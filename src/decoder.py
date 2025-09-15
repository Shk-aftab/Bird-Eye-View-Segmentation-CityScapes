"""
Decoder (Segmentation Head) for BEV Segmentation
Generates final segmentation mask from encoded features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Decoder(nn.Module):
    """
    Decoder for BEV segmentation
    Upsamples encoded features to generate final segmentation mask
    """
    
    def __init__(self, input_channels=128, num_classes=7, bev_height=256, bev_width=256):
        """
        Args:
            input_channels: Number of input channels from BEV encoder
            num_classes: Number of segmentation classes
            bev_height: Target BEV height
            bev_width: Target BEV width
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.bev_height = bev_height
        self.bev_width = bev_width
        
        print(f"Initializing Decoder...")
        print(f"  - Input channels: {input_channels}")
        print(f"  - Number of classes: {num_classes}")
        print(f"  - Target BEV size: {bev_height}x{bev_width}")
        
        # Decoder layers with upsampling
        self.decoder = nn.Sequential(
            # First upsampling: 64x64 -> 128x128
            nn.ConvTranspose2d(input_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Second upsampling: 128x128 -> 256x256
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Final classification layer
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
        
        print(f"  - Decoder architecture: {input_channels} -> 128 -> 64 -> {num_classes}")
        print(f"  - Upsampling: 64x64 -> 128x128 -> 256x256")
        
    def forward(self, encoded_features: torch.Tensor) -> torch.Tensor:
        """
        Generate segmentation mask from encoded features
        
        Args:
            encoded_features: Encoded features from BEV encoder (B, C, H, W)
        
        Returns:
            segmentation_logits: Segmentation logits (B, num_classes, H', W')
        """
        # Process through decoder
        segmentation_logits = self.decoder(encoded_features)
        
        # Verify output shape
        expected_shape = (encoded_features.shape[0], self.num_classes, self.bev_height, self.bev_width)
        if segmentation_logits.shape != expected_shape:
            # Resize to target size if needed
            segmentation_logits = F.interpolate(
                segmentation_logits, 
                size=(self.bev_height, self.bev_width), 
                mode='bilinear', 
                align_corners=False
            )
        
        return segmentation_logits


def test_decoder():
    """Test the decoder implementation"""
    print("Testing Decoder...")
    
    # Create dummy encoded features
    batch_size = 2
    channels = 128
    height, width = 64, 64
    
    encoded_features = torch.randn(batch_size, channels, height, width)
    print(f"Input encoded features: {encoded_features.shape}")
    
    # Test decoder
    decoder = Decoder(input_channels=128, num_classes=7, bev_height=256, bev_width=256)
    segmentation_logits = decoder(encoded_features)
    
    print(f"Output segmentation logits: {segmentation_logits.shape}")
    
    # Verify output shape
    expected_shape = (batch_size, 7, 256, 256)
    assert segmentation_logits.shape == expected_shape, f"Expected {expected_shape}, got {segmentation_logits.shape}"
    
    print("Decoder test completed!")


if __name__ == "__main__":
    test_decoder()
