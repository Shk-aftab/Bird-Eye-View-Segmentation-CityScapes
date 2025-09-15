"""
BEV Encoder for processing projected features
Processes BEV features with CNN layers before segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BEVEncoder(nn.Module):
    """
    BEV Encoder for processing projected features
    Refines spatial features before segmentation
    """
    
    def __init__(self, input_channels=64, hidden_channels=128, output_channels=128):
        """
        Args:
            input_channels: Number of input channels from view transform
            hidden_channels: Number of hidden channels
            output_channels: Number of output channels
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        
        print(f"Initializing BEV Encoder...")
        print(f"  - Input channels: {input_channels}")
        print(f"  - Hidden channels: {hidden_channels}")
        print(f"  - Output channels: {output_channels}")
        
        # Encoder layers
        self.encoder = nn.Sequential(
            # First block: 256x256 -> 128x128
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            # Second block: 128x128 -> 64x64
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            # Third block: 64x64 -> 64x64 (no downsampling)
            nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            
            # Fourth block: 64x64 -> 64x64 (residual-like)
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        
        print(f"  - Encoder architecture: {input_channels} -> {hidden_channels} -> {output_channels}")
        print(f"  - Downsampling: 256x256 -> 128x128 -> 64x64")
        
    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        """
        Process BEV features
        
        Args:
            bev_features: BEV features from view transform (B, C, H, W)
        
        Returns:
            encoded_features: Processed features (B, output_channels, H', W')
        """
        # Process through encoder
        encoded_features = self.encoder(bev_features)
        
        return encoded_features


def test_bev_encoder():
    """Test the BEV encoder implementation"""
    print("Testing BEV Encoder...")
    
    # Create dummy BEV features
    batch_size = 2
    channels = 64
    height, width = 256, 256
    
    bev_features = torch.randn(batch_size, channels, height, width)
    print(f"Input BEV features: {bev_features.shape}")
    
    # Test encoder
    encoder = BEVEncoder(input_channels=64, hidden_channels=128, output_channels=128)
    encoded_features = encoder(bev_features)
    
    print(f"Output encoded features: {encoded_features.shape}")
    
    # Verify output shape
    expected_shape = (batch_size, 128, 64, 64)
    assert encoded_features.shape == expected_shape, f"Expected {expected_shape}, got {encoded_features.shape}"
    
    print("BEV Encoder test completed!")


if __name__ == "__main__":
    test_bev_encoder()
