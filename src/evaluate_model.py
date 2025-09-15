"""
Model Evaluation Script
Loads a trained model checkpoint and evaluates on 20 random validation samples
"""

import torch
import yaml
import os
import sys
import random
import argparse
from typing import Dict, Any
import numpy as np

from .model import BEVSegmentationModel
from .dataset import create_dataloader
from .training import FocalLoss, MetricsCalculator
from .visualization import create_default_visualizer


def load_model_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint"""
    print(f"📁 Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from checkpoint or run directory
    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))  # Go up from checkpoints/ to run directory
    config_path = os.path.join(run_dir, "config.yaml")
    
    if os.path.exists(config_path):
        # Load config from run directory
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"  - Loaded config from run directory: {config_path}")
        # Create model using config file
        model = BEVSegmentationModel(config_path=config_path)
    elif 'config' in checkpoint:
        # Load config from checkpoint
        config = checkpoint['config']
        print(f"  - Loaded config from checkpoint")
        # Create temporary config file
        temp_config_path = os.path.join(run_dir, "temp_config.yaml")
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        model = BEVSegmentationModel(config_path=temp_config_path)
    else:
        # Use default config
        config = {
            'dataset_path': 'data/cam2bev-data-master-1_FRLR/1_FRLR',
            'img_height': 384,
            'img_width': 384,
            'bev_height': 256,
            'bev_width': 256,
            'num_classes': 7,
            'backbone': 'resnet18'
        }
        print(f"  - Using default config")
        # Create temporary config file
        temp_config_path = os.path.join(run_dir, "temp_config.yaml")
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        model = BEVSegmentationModel(config_path=temp_config_path)
    
    model = model.to(device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  - Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        # Get loss from metrics if available
        if 'metrics' in checkpoint and 'loss' in checkpoint['metrics']:
            loss_val = checkpoint['metrics']['loss']
            if hasattr(loss_val, 'item'):
                loss_val = loss_val.item()
            print(f"  - Training loss: {loss_val:.4f}")
        else:
            print(f"  - Training loss: unknown")
    else:
        model.load_state_dict(checkpoint)
        print(f"  - Loaded model state dict")
    
    model.eval()
    print(f"  - Run directory: {run_dir}")
    
    return model, config, run_dir


def evaluate_samples(model, val_loader, device, num_samples=20, run_dir="runs/run_unknown"):
    """Evaluate model on random validation samples"""
    
    # Create evaluation directory inside the run directory
    eval_dir = os.path.join(run_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(os.path.join(eval_dir, "visualizations"), exist_ok=True)
    
    print(f"📁 Saving evaluation results to: {eval_dir}")
    
    # Get random sample indices
    total_samples = len(val_loader.dataset)
    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    print(f"🎯 Evaluating {len(sample_indices)} random samples...")
    
    # Initialize metrics
    class_names = {
        0: "unlabeled", 1: "car", 2: "vegetation", 3: "road",
        4: "terrain", 5: "guard_rail", 6: "sidewalk"
    }
    metrics_calc = MetricsCalculator(7, class_names)
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    
    # Create visualizer (without BEV features)
    visualizer = create_default_visualizer()
    
    all_metrics = []
    total_loss = 0.0
    
    with torch.no_grad():
        for i, sample_idx in enumerate(sample_indices):
            print(f"  Processing sample {i+1}/{len(sample_indices)} (index: {sample_idx})")
            
            # Get specific sample from dataset
            sample = val_loader.dataset[sample_idx]
            
            # Prepare batch data
            camera_images = {}
            for view, img in sample['camera_images'].items():
                camera_images[view] = img.unsqueeze(0).to(device)  # Add batch dimension
            
            targets = sample['bev_label'].unsqueeze(0).to(device)  # Add batch dimension
            sample_id = sample.get('sample_id', f'sample_{sample_idx}')
            
            # Forward pass
            outputs = model(camera_images)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Calculate metrics
            metrics = metrics_calc.calculate_metrics(outputs, targets)
            all_metrics.append(metrics)
            
            # Create visualization (without BEV features)
            print(f"    - Loss: {loss.item():.4f}")
            print(f"    - Accuracy: {metrics['accuracy']:.4f}")
            print(f"    - Mean IoU: {metrics['mean_iou']:.4f}")
            
            # Save visualization
            vis_path = os.path.join(eval_dir, "visualizations", f"eval_{i+1:02d}_{sample_id}.png")
            visualizer.visualize_sample(
                camera_images=camera_images,
                bev_features=None,  # Remove BEV features from visualization
                segmentation_logits=outputs,
                ground_truth=targets,
                sample_id=f"eval_{i+1:02d}_{sample_id}",
                save_path=vis_path
            )
            print(f"    - Saved: {vis_path}")
    
    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    avg_loss = total_loss / len(sample_indices)
    
    print(f"\n📊 Evaluation Results:")
    print(f"  - Samples evaluated: {len(sample_indices)}")
    print(f"  - Average loss: {avg_loss:.4f}")
    print(f"  - Average accuracy: {avg_metrics['accuracy']:.4f}")
    print(f"  - Average mean IoU: {avg_metrics['mean_iou']:.4f}")
    
    # Print per-class IoU
    print(f"\n📈 Per-class IoU:")
    for i, class_name in class_names.items():
        if i < len(avg_metrics.get('per_class_iou', [])):
            iou = avg_metrics['per_class_iou'][i]
            print(f"  - {class_name}: {iou:.4f}")
    
    # Save results to file
    results_file = os.path.join(eval_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Samples evaluated: {len(sample_indices)}\n")
        f.write(f"Average loss: {avg_loss:.4f}\n")
        f.write(f"Average accuracy: {avg_metrics['accuracy']:.4f}\n")
        f.write(f"Average mean IoU: {avg_metrics['mean_iou']:.4f}\n")
        f.write(f"\nPer-class IoU:\n")
        for i, class_name in class_names.items():
            if i < len(avg_metrics.get('per_class_iou', [])):
                iou = avg_metrics['per_class_iou'][i]
                f.write(f"  {class_name}: {iou:.4f}\n")
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"🖼️  Visualizations saved to: {os.path.join(eval_dir, 'visualizations')}")
    
    return avg_metrics, avg_loss


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate BEV Segmentation Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (optional, auto-detected from checkpoint)')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of random samples to evaluate')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (default: auto-detect from checkpoint)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()
    
    print("🔍 BEV Segmentation Model Evaluation")
    print("=" * 50)
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return False
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Load model
    try:
        if args.config:
            # Use provided config file
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            model = BEVSegmentationModel(config_path=args.config)
            model = model.to(device)
            
            # Load checkpoint
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            run_dir = os.path.dirname(os.path.dirname(args.checkpoint))
            print(f"  - Using provided config: {args.config}")
        else:
            # Auto-detect config from checkpoint
            model, config, run_dir = load_model_checkpoint(args.checkpoint, device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Create validation data loader
    print(f"\n📊 Creating validation data loader...")
    data_root = config.get('dataset_path', 'data/cam2bev-data-master-1_FRLR/1_FRLR')
    
    if not os.path.exists(data_root):
        print(f"❌ Dataset not found at: {data_root}")
        return False
    
    val_loader = create_dataloader(
        data_root, 
        split='val', 
        batch_size=1,  # Single sample evaluation
        num_workers=2,
        data_percentage=1.0,
        use_temporal=False,  # Disable temporal for simplicity
        img_height=config.get('img_height', 384),
        img_width=config.get('img_width', 384),
        bev_height=config.get('bev_height', 256),
        bev_width=config.get('bev_width', 256)
    )
    
    print(f"  - Validation samples available: {len(val_loader.dataset)}")
    
    # Determine output directory
    if args.output_dir:
        eval_dir = args.output_dir
    else:
        eval_dir = os.path.join(run_dir, "evaluation")
    
    # Run evaluation
    try:
        avg_metrics, avg_loss = evaluate_samples(
            model, val_loader, device, 
            num_samples=args.num_samples, 
            run_dir=run_dir
        )
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: {eval_dir}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
