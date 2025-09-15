import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.decoder import SegmentationHead

def test_decoder():
    model = SegmentationHead(in_channels=128, num_classes=30)
    x = torch.randn(1, 128, 200, 200)
    out = model(x)
    print("Decoder out shape:", out.shape)

if __name__ == "__main__":
    test_decoder()
