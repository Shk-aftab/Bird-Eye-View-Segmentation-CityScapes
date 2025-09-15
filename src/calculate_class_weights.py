"""
Pre-calculate class weights for the dataset and save to config
This avoids recalculating weights every training run
"""

import torch
import yaml
import os
import sys
import random
from typing import Dict, Any
from tqdm import tqdm

from .dataset import create_dataloader


def calculate_class_weights(data_root: str, config_path: str, num_samples: int = 100):
    """
    Calculate class weights from dataset and save to config
    
    Args:
        data_root: Path to dataset
        config_path: Path to config file
        num_samples: Number of batches to sample for weight calculation
    """
    print("🔢 Calculating class weights for dataset...")
    print(f"  - Dataset: {data_root}")
    print(f"  - Config: {config_path}")
    print(f"  - Sample batches: {num_samples}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data loader
    train_loader = create_dataloader(
        data_root, 
        split='train', 
        batch_size=config.get('batch_size', 8),
        num_workers=2,
        data_percentage=config.get('train_data_percentage', 1.0),
        use_temporal=config.get('temporal_fusion', {}).get('enabled', False),
        temporal_config=config.get('temporal_fusion', {}),
        img_height=config.get('img_height', 384),
        img_width=config.get('img_width', 384),
        bev_height=config.get('bev_height', 256),
        bev_width=config.get('bev_width', 256)
    )
    
    print(f"  - Training batches available: {len(train_loader)}")
    
    # Calculate class distribution
    class_counts = torch.zeros(7)  # 7 classes
    total_pixels = 0
    num_sampled = 0
    
    print(f"  - Processing batches with progress tracking...")
    
    # Much more efficient: sample directly from dataset
    dataset = train_loader.dataset
    total_samples = len(dataset)
    samples_per_batch = config.get('batch_size', 8)
    
    # Calculate how many individual samples we need
    num_individual_samples = num_samples * samples_per_batch
    num_individual_samples = min(num_individual_samples, total_samples)
    
    print(f"  - Sampling {num_individual_samples} individual samples...")
    print(f"  - This is equivalent to {num_individual_samples // samples_per_batch} batches")
    
    # Sample random individual samples
    random.seed(42)  # For reproducibility
    sample_indices = random.sample(range(total_samples), num_individual_samples)
    
    # Create progress bar
    pbar = tqdm(total=num_individual_samples, desc="Processing samples", unit="sample")
    
    with torch.no_grad():
        for idx in sample_indices:
            # Get single sample from dataset
            sample = dataset[idx]
            targets = sample['bev_label'].unsqueeze(0)  # Add batch dimension
            
            unique_classes, counts = torch.unique(targets, return_counts=True)
            
            for class_id, count in zip(unique_classes, counts):
                class_counts[class_id] += count.item()
                total_pixels += count.item()
            
            pbar.update(1)
            
            # Update progress bar description
            pbar.set_postfix({
                'pixels': f"{total_pixels:,}",
                'samples': f"{pbar.n}/{num_individual_samples}"
            })
    
    pbar.close()
    
    # Calculate class weights using sqrt of inverse frequency
    class_weights = torch.zeros(7)
    for i in range(7):
        if class_counts[i] > 0:
            class_weights[i] = torch.sqrt(total_pixels / (7 * class_counts[i]))
        else:
            class_weights[i] = 1.0  # Default weight for unseen classes
    
    # Normalize weights to have mean = 1
    class_weights = class_weights / class_weights.mean()
    
    # Convert to list for YAML serialization
    class_weights_list = class_weights.tolist()
    class_distribution = (class_counts / total_pixels * 100).tolist()
    
    print(f"\n📊 Class Distribution:")
    class_names = ['unlabeled', 'car', 'vegetation', 'road', 'terrain', 'guard_rail', 'sidewalk']
    for i, (name, dist, weight) in enumerate(zip(class_names, class_distribution, class_weights_list)):
        print(f"  - {name}: {dist:.2f}% (weight: {weight:.4f})")
    
    print(f"\n🎯 Calculated class weights: {class_weights_list}")
    
    # Add to config
    config['class_weights'] = class_weights_list
    config['class_distribution'] = class_distribution
    config['weight_calculation_info'] = {
        'num_batches_sampled': num_sampled,
        'total_pixels': int(total_pixels),
        'method': 'sqrt_inverse_frequency'
    }
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"\n✅ Class weights saved to config: {config_path}")
    print(f"  - Sampled {num_individual_samples} individual samples")
    print(f"  - Equivalent to {num_individual_samples // samples_per_batch} batches")
    print(f"  - Total pixels: {total_pixels:,}")
    
    return class_weights_list


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate class weights for dataset')
    parser.add_argument('--data-root', type=str, 
                       default='data/cam2bev-data-master-1_FRLR/1_FRLR',
                       help='Path to dataset')
    parser.add_argument('--config', type=str, 
                       default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of batches to sample')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.data_root):
        print(f"❌ Dataset not found: {args.data_root}")
        return False
    
    # Check if config exists
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return False
    
    try:
        class_weights = calculate_class_weights(
            args.data_root, 
            args.config, 
            args.num_samples
        )
        print(f"\n🎉 Class weights calculation completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error calculating class weights: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
