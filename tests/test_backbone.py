import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.backbone import Backbone

def test_backbone():
    model = Backbone("resnet50", pretrained=False, freeze=False)
    x = torch.randn(1, 3, 224, 224)  # dummy input
    feat = model(x)
    print("Backbone out:", feat.shape)

if __name__ == "__main__":
    test_backbone()
