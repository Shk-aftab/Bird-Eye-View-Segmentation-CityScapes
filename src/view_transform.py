"""
Clean View Transformation Module
Simple implementation for multi-view camera to BEV projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewTransform(nn.Module):
    """
    Simple view transformation for multi-view camera to BEV
    Uses adaptive pooling as a placeholder for proper LSS projection
    """
    
    def __init__(self, 
                 bev_height=256, 
                 bev_width=256,
                 input_height=12,
                 input_width=12):
        """
        Args:
            bev_height: Height of BEV grid
            bev_width: Width of BEV grid  
            input_height: Height of input camera features
            input_width: Width of input camera features
        """
        super().__init__()
        
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.input_height = input_height
        self.input_width = input_width
        
        # Simple projection layers for each view
        self.projection_layers = nn.ModuleDict({
            'front': nn.Sequential(
                nn.AdaptiveAvgPool2d((bev_height, bev_width)),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=True)
            ),
            'left': nn.Sequential(
                nn.AdaptiveAvgPool2d((bev_height, bev_width)),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=True)
            ),
            'rear': nn.Sequential(
                nn.AdaptiveAvgPool2d((bev_height, bev_width)),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=True)
            ),
            'right': nn.Sequential(
                nn.AdaptiveAvgPool2d((bev_height, bev_width)),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        })
        
        # Fusion layer to combine all views
        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 4, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features):
        """
        Forward pass: project camera features to BEV space
        Args:
            features: Dict of camera features {'front': (B,C,H,W), 'left': ..., 'rear': ..., 'right': ...}
        """
        B = next(iter(features.values())).shape[0]
        device = next(iter(features.values())).device
        
        # Process each camera view
        view_features = []
        for view_name, view_feat in features.items():
            if view_name in self.projection_layers:
                # Project to BEV space
                bev_feat = self.projection_layers[view_name](view_feat)
                view_features.append(bev_feat)
        
        # Concatenate all views
        if view_features:
            combined = torch.cat(view_features, dim=1)  # (B, C*4, H, W)
            # Fuse features
            bev_features = self.fusion(combined)
        else:
            # Fallback if no features
            C = next(iter(features.values())).shape[1]
            bev_features = torch.zeros(B, 64, self.bev_height, self.bev_width, device=device)
        
        return bev_features


def test_view_transform():
    """Test the view transformation module"""
    print("Testing ViewTransform...")
    
    # Create dummy features
    B, C, H, W = 2, 64, 96, 96
    features = {
        'front': torch.randn(B, C, H, W),
        'left': torch.randn(B, C, H, W),
        'rear': torch.randn(B, C, H, W),
        'right': torch.randn(B, C, H, W)
    }
    
    # Create view transform
    view_transform = ViewTransform(bev_height=64, bev_width=64)
    
    # Forward pass
    with torch.no_grad():
        bev_features = view_transform(features)
    
    print(f"Input features shape: {[v.shape for v in features.values()]}")
    print(f"Output BEV features shape: {bev_features.shape}")
    print(f"BEV features range: [{bev_features.min():.3f}, {bev_features.max():.3f}]")
    print(f"BEV features mean: {bev_features.mean():.3f}")
    
    print("ViewTransform test completed!")


if __name__ == "__main__":
    test_view_transform()