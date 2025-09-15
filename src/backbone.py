"""
Backbone for BEV Segmentation
Extracts features from multi-view camera images
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any


class Backbone(nn.Module):
    """
    Backbone network for feature extraction from camera images
    Supports ResNet50, ResNet18, and EfficientNet-B0
    """
    
    def __init__(self, backbone_type='resnet50', pretrained=True, freeze_backbone=False):
        """
        Args:
            backbone_type: 'resnet50', 'resnet18', or 'efficientnet'
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone parameters
        """
        super().__init__()
        
        self.backbone_type = backbone_type
        self.freeze_backbone = freeze_backbone
        
        print(f"Initializing {backbone_type} backbone...")
        print(f"  - Pretrained: {pretrained}")
        print(f"  - Freeze backbone: {freeze_backbone}")
        
        # Create backbone
        if backbone_type == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            # Remove final classification layers
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            self.feature_dim = 2048
            self.output_stride = 32
            
        elif backbone_type == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            # Remove final classification layers
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            self.feature_dim = 512
            self.output_stride = 32
            
        elif backbone_type == 'efficientnet':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            # Remove final classification layers
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.feature_dim = 1280
            self.output_stride = 32
            
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            print("  - Freezing backbone parameters...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Feature projection to consistent dimension
        self.feature_proj = nn.Conv2d(
            self.feature_dim, 
            64, 
            kernel_size=1, 
            bias=False
        )
        
        print(f"  - Feature dimension: {self.feature_dim} -> 64")
        print(f"  - Output stride: {self.output_stride}")
        
    def forward(self, camera_images: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract features from multi-view camera images
        
        Args:
            camera_images: Dict with keys ['front', 'left', 'rear', 'right']
                          Each value is tensor of shape (B, 3, H, W)
        
        Returns:
            features: Dict with same keys, each value is (B, 64, H', W')
        """
        features = {}
        
        for view_name, image in camera_images.items():
            # Extract features
            with torch.set_grad_enabled(not self.freeze_backbone):
                view_features = self.backbone(image)
            
            # Project to consistent dimension
            view_features = self.feature_proj(view_features)
            
            features[view_name] = view_features
        
        return features


def test_backbone():
    """Test the backbone implementation"""
    print("Testing Backbone...")
    
    # Create dummy camera images
    batch_size = 2
    height, width = 384, 384
    
    camera_images = {
        'front': torch.randn(batch_size, 3, height, width),
        'left': torch.randn(batch_size, 3, height, width),
        'rear': torch.randn(batch_size, 3, height, width),
        'right': torch.randn(batch_size, 3, height, width)
    }
    
    print(f"Input shapes:")
    for view, img in camera_images.items():
        print(f"  {view}: {img.shape}")
    
    # Test ResNet50
    print("\n--- Testing ResNet50 ---")
    backbone = Backbone('resnet50', pretrained=False)
    features = backbone(camera_images)
    
    print(f"\nOutput shapes:")
    for view, feat in features.items():
        print(f"  {view}: {feat.shape}")
    
    # Test ResNet18
    print("\n--- Testing ResNet18 ---")
    backbone = Backbone('resnet18', pretrained=False)
    features = backbone(camera_images)
    
    print(f"\nOutput shapes:")
    for view, feat in features.items():
        print(f"  {view}: {feat.shape}")
    
    print("\nBackbone test completed!")


if __name__ == "__main__":
    test_backbone()
