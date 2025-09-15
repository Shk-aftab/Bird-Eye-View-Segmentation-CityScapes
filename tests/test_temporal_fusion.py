"""
Test Temporal Fusion Integration
"""

import torch
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import BEVSegmentationModel
from src.dataset import create_dataloader
from src.temporal_fusion import test_temporal_fusion


def test_temporal_integration():
    """Test complete temporal fusion integration"""
    print("🧪 Testing Temporal Fusion Integration...")
    print("=" * 50)
    
    # Test 1: Temporal fusion module
    print("\n1. Testing Temporal Fusion Module:")
    test_temporal_fusion()
    
    # Test 2: Model with temporal fusion
    print("\n2. Testing Model with Temporal Fusion:")
    
    # Load config with temporal fusion enabled
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "resnet50_high_res.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = BEVSegmentationModel(config_path)
    print(f"  - Model created with temporal fusion: {model.use_temporal}")
    
    # Test 3: Dataset with temporal sequences
    print("\n3. Testing Dataset with Temporal Sequences:")
    
    data_root = os.path.join(os.path.dirname(__file__), "..", "data", "cam2bev-data-master-1_FRLR", "1_FRLR")
    temporal_config = config.get('temporal_fusion', {})
    
    try:
        # Create temporal dataloader
        train_loader = create_dataloader(
            data_root=data_root,
            split='train',
            batch_size=2,
            num_workers=0,  # Use 0 for testing
            data_percentage=0.01,  # Use only 1% for testing
            use_temporal=True,
            temporal_config=temporal_config
        )
        
        print(f"  - Temporal dataloader created successfully")
        print(f"  - Number of samples: {len(train_loader.dataset)}")
        print(f"  - Number of batches: {len(train_loader)}")
        
        # Test a single batch
        print("\n4. Testing Single Batch:")
        for batch in train_loader:
            print(f"  - Batch keys: {list(batch.keys())}")
            print(f"  - Camera images shape: {batch['camera_images']['front'].shape}")
            
            if 'temporal_camera_images' in batch:
                print(f"  - Temporal camera images: {len(batch['temporal_camera_images'])} timesteps")
                for i, temporal_images in enumerate(batch['temporal_camera_images']):
                    print(f"    Timestep {i}: {temporal_images['front'].shape}")
            
            print(f"  - BEV label shape: {batch['bev_label'].shape}")
            print(f"  - Sample ID: {batch['sample_id']}")
            
            # Test model forward pass
            print("\n5. Testing Model Forward Pass:")
            model.eval()
            with torch.no_grad():
                if 'temporal_camera_images' in batch:
                    outputs = model(batch['camera_images'], batch['temporal_camera_images'])
                    print(f"  - Temporal forward pass successful")
                else:
                    outputs = model(batch['camera_images'])
                    print(f"  - Single timestep forward pass successful")
                
                print(f"  - Output shape: {outputs.shape}")
                print(f"  - Expected shape: (2, 7, 384, 384)")
            
            break  # Only test first batch
        
        print("\n✅ Temporal fusion integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Temporal fusion integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_timestep():
    """Test single timestep mode (backward compatibility)"""
    print("\n🔄 Testing Single Timestep Mode (Backward Compatibility):")
    
    # Load default config (no temporal fusion)
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = BEVSegmentationModel(config_path)
    print(f"  - Model created with temporal fusion: {model.use_temporal}")
    
    # Create single timestep dataloader
    data_root = os.path.join(os.path.dirname(__file__), "..", "data", "cam2bev-data-master-1_FRLR", "1_FRLR")
    train_loader = create_dataloader(
        data_root=data_root,
        split='train',
        batch_size=2,
        num_workers=0,
        data_percentage=0.01,
        use_temporal=False,
        temporal_config=None
    )
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            outputs = model(batch['camera_images'])
            print(f"  - Single timestep forward pass successful")
            print(f"  - Output shape: {outputs.shape}")
            break
    
    print("✅ Single timestep mode test completed!")
    return True


if __name__ == "__main__":
    print("🚀 BEV Segmentation - Temporal Fusion Test Suite")
    print("=" * 60)
    
    # Test temporal fusion integration
    temporal_success = test_temporal_integration()
    
    # Test single timestep mode
    single_success = test_single_timestep()
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"  - Temporal fusion integration: {'✅ PASS' if temporal_success else '❌ FAIL'}")
    print(f"  - Single timestep mode: {'✅ PASS' if single_success else '❌ FAIL'}")
    
    if temporal_success and single_success:
        print("\n🎉 All tests passed! Temporal fusion is ready to use.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
