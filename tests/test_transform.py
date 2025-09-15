import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.view_transform import project_to_bev

def test_transform():
    # Dummy features
    feat = torch.randn(1, 64, 32, 32)

    # No calibration (fallback)
    bev = project_to_bev(feat, cfg={"bev_height":100,"bev_width":100,"depth_bins":4})
    print("BEV fallback shape:", bev.shape)

    # With fake calibration
    intrinsics = torch.eye(3).unsqueeze(0)
    extrinsics = torch.eye(4).unsqueeze(0)
    bev = project_to_bev(feat, intrinsics, extrinsics, cfg={"bev_height":100,"bev_width":100,"depth_bins":4})
    print("BEV with calibration shape:", bev.shape)

if __name__ == "__main__":
    test_transform()
