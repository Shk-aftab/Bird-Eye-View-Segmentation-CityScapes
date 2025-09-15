import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model import BEVPipeline

def test_pipeline():
    print("🧪 Testing BEV Pipeline...")
    
    cfg = {
        "backbone": "resnet50",
        "pretrained": False,  # Skip pretrained for speed
        "freeze_backbone": False,
        "bev_channels": 64,
        "num_classes": 30,
        "bev_height": 50,
        "bev_width": 50,
        "depth_bins": 4
    }
    
    print(f"📋 Config: {cfg}")
    
    print("🧠 Initializing model...")
    model = BEVPipeline(cfg)
    print("✅ Model initialized")
    
    # Create dummy camera inputs
    print("📸 Creating dummy camera inputs...")
    cameras = {
        "front": torch.randn(1, 3, 224, 224),
        "rear": torch.randn(1, 3, 224, 224),
        "left": torch.randn(1, 3, 224, 224),
        "right": torch.randn(1, 3, 224, 224)
    }
    print(f"✅ Camera inputs created - Shapes: {[(k, v.shape) for k, v in cameras.items()]}")
    
    print("🔮 Running forward pass...")
    import time
    start_time = time.time()
    out = model(cameras)
    forward_time = time.time() - start_time
    
    print(f"✅ Forward pass completed in {forward_time:.2f}s")
    print(f"📊 Output shape: {out.shape}")
    print(f"📈 Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    print("🎉 Pipeline test passed!")

if __name__ == "__main__":
    test_pipeline()
