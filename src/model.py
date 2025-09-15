"""
Complete BEV Segmentation Model
Combines Backbone, View Transform, BEV Encoder, and Decoder
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import yaml
import os

from .backbone import Backbone
from .view_transform import ViewTransform
from .bev_encoder import BEVEncoder
from .decoder import Decoder
from .temporal_fusion import TemporalFusion


class BEVSegmentationModel(nn.Module):
    """
    Complete BEV Segmentation Model
    Combines all components into end-to-end pipeline
    """
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        """
        Args:
            config_path: Path to configuration file
        """
        super().__init__()
        
        print("Initializing BEV Segmentation Model...")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"  - Config loaded from: {config_path}")
        
        # Extract parameters
        backbone_type = self.config.get('backbone', 'resnet50')
        pretrained = self.config.get('pretrained', True)
        freeze_backbone = self.config.get('freeze_backbone', False)
        num_classes = self.config.get('num_classes', 7)
        bev_height = self.config.get('bev_height', 256)
        bev_width = self.config.get('bev_width', 256)
        
        # Temporal fusion parameters
        self.temporal_config = self.config.get('temporal_fusion', {})
        self.use_temporal = self.temporal_config.get('enabled', False)
        self.num_timesteps = self.temporal_config.get('num_timesteps', 3)
        
        print(f"  - Backbone: {backbone_type}")
        print(f"  - Pretrained: {pretrained}")
        print(f"  - Freeze backbone: {freeze_backbone}")
        print(f"  - Number of classes: {num_classes}")
        print(f"  - BEV size: {bev_height}x{bev_width}")
        
        # Initialize components
        self.backbone = Backbone(
            backbone_type=backbone_type,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
        
        self.view_transform = ViewTransform(
            bev_height=bev_height,
            bev_width=bev_width,
            input_height=12,  # From ResNet50 output
            input_width=12
        )
        
        self.bev_encoder = BEVEncoder(
            input_channels=64,  # From view transform output
            hidden_channels=128,
            output_channels=128
        )
        
        self.decoder = Decoder(
            input_channels=128,  # From BEV encoder output
            num_classes=num_classes,
            bev_height=bev_height,
            bev_width=bev_width
        )
        
        # Initialize temporal fusion if enabled
        if self.use_temporal:
            fusion_method = self.temporal_config.get('fusion_method', 'attention')
            self.temporal_fusion = TemporalFusion(
                feature_dim=64,  # From view transform output
                num_timesteps=self.num_timesteps,
                fusion_method=fusion_method
            )
            print(f"  - Temporal fusion enabled: {fusion_method} method, {self.num_timesteps} timesteps")
        else:
            self.temporal_fusion = None
            print("  - Temporal fusion disabled")
        
        print("  - All components initialized successfully")
        
    def forward(self, camera_images: Dict[str, torch.Tensor], temporal_camera_images: list = None) -> torch.Tensor:
        """
        Forward pass through complete pipeline
        
        Args:
            camera_images: Dict with keys ['front', 'left', 'rear', 'right']
                          Each value is tensor of shape (B, 3, H, W)
            temporal_camera_images: List of temporal camera images for fusion
                                   Each element is a dict like camera_images
        
        Returns:
            segmentation_logits: Segmentation logits (B, num_classes, H', W')
        """
        if self.use_temporal and temporal_camera_images is not None:
            # Temporal fusion mode
            return self._forward_temporal(camera_images, temporal_camera_images)
        else:
            # Single timestep mode
            return self._forward_single(camera_images)
    
    def _forward_single(self, camera_images: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single timestep forward pass"""
        # Step 1: Extract features from camera images
        features = self.backbone(camera_images)
        
        # Step 2: Transform features to BEV space
        bev_features = self.view_transform(features)
        
        # Step 3: Encode BEV features
        encoded_features = self.bev_encoder(bev_features)
        
        # Step 4: Generate segmentation mask
        segmentation_logits = self.decoder(encoded_features)
        
        return segmentation_logits
    
    def _forward_temporal(self, camera_images: Dict[str, torch.Tensor], temporal_camera_images: list) -> torch.Tensor:
        """Temporal fusion forward pass"""
        # Process all timesteps
        temporal_bev_features = []
        
        for t, temporal_images in enumerate(temporal_camera_images):
            # Extract features for this timestep
            features = self.backbone(temporal_images)
            
            # Transform to BEV space
            bev_features = self.view_transform(features)
            
            temporal_bev_features.append(bev_features)
        
        # Fuse temporal features
        fused_bev_features = self.temporal_fusion(temporal_bev_features)
        
        # Encode fused BEV features
        encoded_features = self.bev_encoder(fused_bev_features)
        
        # Generate segmentation mask
        segmentation_logits = self.decoder(encoded_features)
        
        return segmentation_logits
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_type': self.config.get('backbone', 'resnet50'),
            'num_classes': self.config.get('num_classes', 7),
            'bev_size': (self.config.get('bev_height', 256), self.config.get('bev_width', 256))
        }


def test_model():
    """Test the complete model"""
    print("Testing Complete BEV Segmentation Model...")
    
    # Create dummy camera images
    batch_size = 2
    height, width = 384, 384
    
    camera_images = {
        'front': torch.randn(batch_size, 3, height, width),
        'left': torch.randn(batch_size, 3, height, width),
        'rear': torch.randn(batch_size, 3, height, width),
        'right': torch.randn(batch_size, 3, height, width)
    }
    
    print(f"Input camera images:")
    for view, img in camera_images.items():
        print(f"  {view}: {img.shape}")
    
    # Test model
    model = BEVSegmentationModel(config_path='../configs/default.yaml')
    
    # Get model info
    model_info = model.get_model_info()
    print(f"\nModel Information:")
    print(f"  - Total parameters: {model_info['total_parameters']:,}")
    print(f"  - Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"  - Backbone: {model_info['backbone_type']}")
    print(f"  - Number of classes: {model_info['num_classes']}")
    print(f"  - BEV size: {model_info['bev_size']}")
    
    # Forward pass
    with torch.no_grad():
        segmentation_logits = model(camera_images)
    
    print(f"\nOutput segmentation logits: {segmentation_logits.shape}")
    
    # Verify output shape
    expected_shape = (batch_size, 7, 256, 256)
    assert segmentation_logits.shape == expected_shape, f"Expected {expected_shape}, got {segmentation_logits.shape}"
    
    print("Complete model test completed!")


if __name__ == "__main__":
    test_model()
