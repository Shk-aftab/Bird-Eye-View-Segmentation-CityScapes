"""
Visualization utilities for BEV Segmentation
Shows 4 camera views, BEV transformation, and ground truth comparison
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, Tuple
import os
from datetime import datetime


class BEVVisualizer:
    """
    Visualization utilities for BEV segmentation pipeline
    """
    
    def __init__(self, class_names: Dict[int, str], class_colors: Dict[int, Tuple[int, int, int]]):
        """
        Args:
            class_names: Mapping from class ID to class name
            class_colors: Mapping from class ID to RGB color
        """
        self.class_names = class_names
        self.class_colors = class_colors
        self.num_classes = len(class_names)
        
        print(f"Initializing BEV Visualizer...")
        print(f"  - Number of classes: {self.num_classes}")
        print(f"  - Class names: {list(class_names.values())}")
        
    def denormalize_image(self, image: torch.Tensor) -> np.ndarray:
        """Denormalize image tensor for visualization"""
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)
        
        denorm_image = image * std + mean
        denorm_image = torch.clamp(denorm_image, 0, 1)
        
        # Convert to numpy and transpose
        return denorm_image.permute(1, 2, 0).cpu().numpy()
    
    def create_class_colormap(self) -> np.ndarray:
        """Create colormap for segmentation classes"""
        colormap = np.zeros((self.num_classes, 3), dtype=np.uint8)
        for class_id, color in self.class_colors.items():
            colormap[class_id] = color
        return colormap
    
    def visualize_sample(self, 
                        camera_images: Dict[str, torch.Tensor],
                        bev_features: Optional[torch.Tensor],
                        segmentation_logits: torch.Tensor,
                        ground_truth: torch.Tensor,
                        sample_id: str,
                        save_path: Optional[str] = None) -> None:
        """
        Visualize a complete sample with 4 camera views, BEV features (optional), and segmentation
        
        Args:
            camera_images: Dict with camera images
            bev_features: BEV features from view transform (optional, can be None)
            segmentation_logits: Predicted segmentation logits
            ground_truth: Ground truth segmentation
            sample_id: Sample identifier
            save_path: Path to save visualization
        """
        
        # Create figure with subplots - adjust layout based on BEV features
        if bev_features is not None:
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        else:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'BEV Segmentation Sample: {sample_id}', fontsize=16)
        
        # Denormalize camera images
        denorm_images = {}
        for view, img in camera_images.items():
            denorm_images[view] = self.denormalize_image(img[0])  # Take first sample
        
        # Plot camera images
        camera_views = ['front', 'left', 'rear', 'right']
        for i, view in enumerate(camera_views):
            axes[0, i].imshow(denorm_images[view])
            axes[0, i].set_title(f'{view.capitalize()} Camera')
            axes[0, i].axis('off')
        
        # Plot BEV features (only if provided)
        if bev_features is not None:
            bev_feat = bev_features[0].cpu().numpy()  # Take first sample
            for i in range(4):
                channel_idx = min(i, bev_feat.shape[0] - 1)
                axes[1, i].imshow(bev_feat[channel_idx], cmap='viridis')
                axes[1, i].set_title(f'BEV Feature Channel {channel_idx}')
                axes[1, i].axis('off')
            
            # Set row index for segmentation plots
            seg_row = 2
        else:
            # Set row index for segmentation plots when no BEV features
            seg_row = 1
        
        # Plot segmentation prediction
        pred_seg = torch.argmax(segmentation_logits[0], dim=0).cpu().numpy()
        axes[seg_row, 0].imshow(pred_seg, cmap='tab10', vmin=0, vmax=self.num_classes-1)
        axes[seg_row, 0].set_title('Predicted Segmentation')
        axes[seg_row, 0].axis('off')
        
        # Plot ground truth
        gt_seg = ground_truth[0].cpu().numpy()
        axes[seg_row, 1].imshow(gt_seg, cmap='tab10', vmin=0, vmax=self.num_classes-1)
        axes[seg_row, 1].set_title('Ground Truth')
        axes[seg_row, 1].axis('off')
        
        # Plot difference
        diff = np.abs(pred_seg - gt_seg)
        axes[seg_row, 2].imshow(diff, cmap='hot')
        axes[seg_row, 2].set_title('Prediction Error')
        axes[seg_row, 2].axis('off')
        
        # Plot class distribution
        pred_classes, pred_counts = np.unique(pred_seg, return_counts=True)
        gt_classes, gt_counts = np.unique(gt_seg, return_counts=True)
        
        axes[seg_row, 3].bar(pred_classes, pred_counts, alpha=0.7, label='Predicted')
        axes[seg_row, 3].bar(gt_classes, gt_counts, alpha=0.7, label='Ground Truth')
        axes[seg_row, 3].set_title('Class Distribution')
        axes[seg_row, 3].set_xlabel('Class ID')
        axes[seg_row, 3].set_ylabel('Pixel Count')
        axes[seg_row, 3].legend()
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()  # Close figure to prevent display during training
    
    def visualize_batch(self, 
                       batch: Dict[str, Any],
                       model_outputs: torch.Tensor,
                       batch_idx: int,
                       save_dir: str) -> None:
        """
        Visualize a batch of samples
        
        Args:
            batch: Batch from dataloader
            model_outputs: Model predictions
            batch_idx: Batch index
            save_dir: Directory to save visualizations
        """
        print(f"Visualizing batch {batch_idx}...")
        
        batch_size = model_outputs.shape[0]
        
        for i in range(min(batch_size, 2)):  # Visualize first 2 samples
            sample_id = batch['sample_id'][i]
            
            # Extract sample data
            camera_images = {view: batch['camera_images'][view][i:i+1] for view in batch['camera_images']}
            ground_truth = batch['bev_label'][i:i+1]
            segmentation_logits = model_outputs[i:i+1]
            
            # Create dummy BEV features for visualization
            bev_features = torch.randn(1, 64, 256, 256)  # Placeholder
            
            # Save path
            save_path = os.path.join(save_dir, f'batch_{batch_idx}_sample_{i}_{sample_id}.png')
            
            # Create visualization
            self.visualize_sample(
                camera_images=camera_images,
                bev_features=bev_features,
                segmentation_logits=segmentation_logits,
                ground_truth=ground_truth,
                sample_id=sample_id,
                save_path=save_path
            )


def create_default_visualizer() -> BEVVisualizer:
    """Create visualizer with default class configuration"""
    
    # Class names for autonomous driving
    class_names = {
        0: "unlabeled",
        1: "car", 
        2: "vegetation",
        3: "road",
        4: "terrain", 
        5: "guard_rail",
        6: "sidewalk"
    }
    
    # Class colors (RGB)
    class_colors = {
        0: (128, 128, 128),  # unlabeled - gray
        1: (0, 0, 142),      # car - dark blue
        2: (107, 142, 35),   # vegetation - green
        3: (128, 64, 128),   # road - purple
        4: (152, 251, 152),  # terrain - light green
        5: (180, 165, 180),  # guard rail - light purple
        6: (244, 35, 232)    # sidewalk - pink
    }
    
    return BEVVisualizer(class_names, class_colors)


def test_visualizer():
    """Test the visualization system"""
    print("Testing BEV Visualizer...")
    
    # Create dummy data
    batch_size = 1
    height, width = 384, 384
    
    camera_images = {
        'front': torch.randn(batch_size, 3, height, width),
        'left': torch.randn(batch_size, 3, height, width),
        'rear': torch.randn(batch_size, 3, height, width),
        'right': torch.randn(batch_size, 3, height, width)
    }
    
    bev_features = torch.randn(batch_size, 64, 256, 256)
    segmentation_logits = torch.randn(batch_size, 7, 256, 256)
    ground_truth = torch.randint(0, 7, (batch_size, 256, 256))
    
    # Create visualizer
    visualizer = create_default_visualizer()
    
    # Test visualization
    visualizer.visualize_sample(
        camera_images=camera_images,
        bev_features=bev_features,
        segmentation_logits=segmentation_logits,
        ground_truth=ground_truth,
        sample_id="test_sample"
    )
    
    print("Visualizer test completed!")


if __name__ == "__main__":
    test_visualizer()
