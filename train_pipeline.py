"""
Complete BEV Segmentation Training Pipeline
Demonstrates the full pipeline with real data
"""

import torch
import yaml
import os
import sys
import argparse
from typing import Dict, Any

# Add src to path
sys.path.append('src')

from src.model import BEVSegmentationModel
from src.dataset import create_dataloader
from src.training import BEVTrainer, FocalLoss, MetricsCalculator
from src.visualization import create_default_visualizer


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='BEV Segmentation Training Pipeline')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--test-only', action='store_true',
                       help='Run only forward pass test, skip training')
    args = parser.parse_args()
    
    print("🚀 BEV Segmentation Pipeline - Complete Training Demo")
    print("=" * 60)
    
    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Configuration loaded from: {config_path}")
    print(f"  - Backbone: {config.get('backbone', 'resnet50')}")
    print(f"  - Batch size: {config.get('batch_size', 4)}")
    print(f"  - Learning rate: {config.get('learning_rate', 1e-4)}")
    print(f"  - Number of classes: {config.get('num_classes', 7)}")
    print(f"  - Data percentage: {config.get('train_data_percentage', 1.0)*100:.1f}%")
    
    # Check if dataset is available
    data_root = config.get('dataset_path', 'data/cam2bev-data-master-1_FRLR/1_FRLR')
    if not os.path.exists(data_root):
        print(f"❌ Dataset not found at: {data_root}")
        print("Please ensure the dataset is available for training")
        return False
    
    # Create data loaders
    print(f"\n📊 Creating data loaders...")
    
    # Check if temporal fusion is enabled
    temporal_config = config.get('temporal_fusion', {})
    use_temporal = temporal_config.get('enabled', False)
    
    train_loader = create_dataloader(
        data_root, 
        split='train', 
        batch_size=config.get('batch_size', 4),
        num_workers=2,
        data_percentage=config.get('train_data_percentage', 1.0),
        use_temporal=use_temporal,
        temporal_config=temporal_config,
        img_height=config.get('img_height', 384),
        img_width=config.get('img_width', 384),
        bev_height=config.get('bev_height', 256),
        bev_width=config.get('bev_width', 256)
    )
    
    val_loader = create_dataloader(
        data_root, 
        split='val', 
        batch_size=config.get('batch_size', 4),
        num_workers=2,
        data_percentage=1.0,  # Always use full validation set
        use_temporal=use_temporal,
        temporal_config=temporal_config,
        img_height=config.get('img_height', 384),
        img_width=config.get('img_width', 384),
        bev_height=config.get('bev_height', 256),
        bev_width=config.get('bev_width', 256)
    )
    
    print(f"  - Training samples: {len(train_loader.dataset)}")
    print(f"  - Validation samples: {len(val_loader.dataset)}")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    
    # Create model
    print(f"\n🏗️ Creating model...")
    model = BEVSegmentationModel(config_path=config_path)
    
    # Get model info
    model_info = model.get_model_info()
    print(f"  - Total parameters: {model_info['total_parameters']:,}")
    print(f"  - Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"  - Model size: {model_info['total_parameters'] * 4 / (1024 * 1024):.1f} MB")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  - Device: {device}")
    
    # Create trainer
    print(f"\n🎯 Creating trainer...")
    trainer = BEVTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        use_wandb=True  # Disable wandb for demo
    )
    
    # Check if test-only mode
    if args.test_only:
        # Test a single forward pass
        print(f"\n🧪 Testing single forward pass...")
        model.eval()
        with torch.no_grad():
            for batch in train_loader:
                # Move data to device
                camera_images = {k: v.to(device) for k, v in batch['camera_images'].items()}
                targets = batch['bev_label'].to(device)
                
                # Handle temporal fusion if enabled
                if 'temporal_camera_images' in batch and use_temporal:
                    temporal_camera_images = []
                    for temporal_batch in batch['temporal_camera_images']:
                        temporal_images = {k: v.to(device) for k, v in temporal_batch.items()}
                        temporal_camera_images.append(temporal_images)
                    outputs = model(camera_images, temporal_camera_images)
                    print(f"  - Temporal fusion enabled: {len(temporal_camera_images)} timesteps")
                else:
                    outputs = model(camera_images)
                    print(f"  - Single timestep mode")
                
                print(f"  - Input shapes:")
                for view, img in camera_images.items():
                    print(f"    {view}: {img.shape}")
                print(f"  - Target shape: {targets.shape}")
                print(f"  - Output shape: {outputs.shape}")
                
                # Test loss calculation
                criterion = FocalLoss(alpha=1.0, gamma=2.0)
                loss = criterion(outputs, targets)
                print(f"  - Focal loss: {loss.item():.4f}")
                
                # Test metrics calculation
                class_names = {
                    0: "unlabeled", 1: "car", 2: "vegetation", 3: "road",
                    4: "terrain", 5: "guard_rail", 6: "sidewalk"
                }
                metrics_calc = MetricsCalculator(7, class_names)
                metrics = metrics_calc.calculate_metrics(outputs, targets)
                
                print(f"  - Accuracy: {metrics['accuracy']:.4f}")
                print(f"  - Mean IoU: {metrics['mean_iou']:.4f}")
                
                # Test visualization
                print(f"  - Creating visualization...")
                visualizer = create_default_visualizer()
                visualizer.visualize_sample(
                    camera_images={k: v[0:1] for k, v in camera_images.items()},
                    bev_features=torch.randn(1, 64, 256, 256),  # Placeholder
                    segmentation_logits=outputs[0:1],
                    ground_truth=targets[0:1],
                    sample_id="training_demo_sample"
                )
                
                break
        
        print(f"\n✅ Single forward pass test completed!")
        print(f"\n🧪 Test-only mode completed!")
        return True
    
    # Ask user if they want to start training
    print(f"\n🎓 Ready to start training!")
    print(f"  - Run path: {trainer.run_path}")
    
    # Print detailed configuration summary
    print(f"\n📋 Configuration Summary:")
    print(f"  - Config file: {config_path}")
    print(f"  - Backbone: {config.get('backbone', 'resnet18')}")
    print(f"  - Image resolution: {config.get('img_height', 256)}x{config.get('img_width', 256)}")
    print(f"  - BEV resolution: {config.get('bev_height', 256)}x{config.get('bev_width', 256)}")
    print(f"  - Batch size: {config.get('batch_size', 4)}")
    print(f"  - Number of workers: {config.get('num_workers', 2)}")
    print(f"  - Learning rate: {config.get('learning_rate', 1e-4)}")
    print(f"  - Backbone LR: {config.get('backbone_lr', 1e-5)}")
    print(f"  - Epochs: {config.get('epochs', 10)}")
    print(f"  - Data percentage: {config.get('train_data_percentage', 1.0)*100:.1f}%")
    print(f"  - Mixed precision: {config.get('mixed_precision', True)}")
    print(f"  - Freeze backbone: {config.get('freeze_backbone', False)}")
    print(f"  - Temporal fusion: {'Enabled' if use_temporal else 'Disabled'}")
    if use_temporal:
        print(f"    - Timesteps: {temporal_config.get('num_timesteps', 3)}")
        print(f"    - Interval: {temporal_config.get('timestep_interval', 500)}ms")
        print(f"    - Method: {temporal_config.get('fusion_method', 'attention')}")
    print(f"  - Training samples: {len(train_loader.dataset):,}")
    print(f"  - Validation samples: {len(val_loader.dataset):,}")
    print(f"  - Training batches: {len(train_loader):,}")
    print(f"  - Validation batches: {len(val_loader):,}")
    print(f"  - Model parameters: {model_info['total_parameters']:,}")
    print(f"  - Model size: {model_info.get('model_size_mb', 0):.1f} MB")
    
    response = input("\nDo you want to start training? (y/n): ").lower().strip()
    
    if response == 'y':
        print(f"\n🚀 Starting training...")
        num_epochs = config.get('epochs', 10)
        trainer.train(num_epochs)
        print(f"\n🎉 Training completed!")
    else:
        print(f"\n👋 Training skipped. Pipeline is ready for use!")
    
    return True


if __name__ == "__main__":
    success = main()
    
    sys.exit(0 if success else 1)
