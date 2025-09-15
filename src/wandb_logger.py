"""
Weights & Biases Integration
Handles experiment tracking, metrics logging, and visualization
"""

import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
import os
from datetime import datetime


class WandBLogger:
    """
    Weights & Biases logger for experiment tracking
    """
    
    def __init__(self, 
                 project_name: str = "bev-segmentation",
                 entity: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 tags: Optional[List[str]] = None):
        """
        Args:
            project_name: Name of the wandb project
            entity: Wandb entity (username or team)
            config: Configuration dictionary
            tags: List of tags for the run
        """
        self.project_name = project_name
        self.entity = entity
        self.config = config or {}
        self.tags = tags or []
        
        print(f"Initializing WandB Logger...")
        print(f"  - Project: {project_name}")
        print(f"  - Entity: {entity}")
        print(f"  - Tags: {tags}")
        
    def init_run(self, run_name: Optional[str] = None) -> None:
        """
        Initialize a new wandb run
        
        Args:
            run_name: Optional custom run name
        """
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"bev_run_{timestamp}"
        
        wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=run_name,
            config=self.config,
            tags=self.tags
        )
        
        print(f"  - Initialized wandb run: {run_name}")
        print(f"  - Run URL: {wandb.run.url}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to wandb
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        try:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
            print(f"  - Logged metrics: {list(metrics.keys())} (step: {step})")
        except Exception as e:
            print(f"  - Warning: Failed to log metrics: {e}")
            # Log without step to avoid ordering issues
            wandb.log(metrics)
    
    def log_model_info(self, model: torch.nn.Module) -> None:
        """
        Log model information to wandb
        
        Args:
            model: PyTorch model
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
        
        wandb.log(model_info)
        print(f"  - Logged model info: {model_info}")
    
    def log_learning_rate(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Log current learning rate
        
        Args:
            optimizer: PyTorch optimizer
        """
        lr = optimizer.param_groups[0]['lr']
        wandb.log({'learning_rate': lr})
    
    def log_gradients(self, model: torch.nn.Module, step: int) -> None:
        """
        Log gradient information
        
        Args:
            model: PyTorch model
            step: Current step
        """
        gradients = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients.append(param.grad.data.cpu().numpy().flatten())
        
        if gradients:
            all_gradients = np.concatenate(gradients)
            
            wandb.log({
                'gradient_norm': np.linalg.norm(all_gradients),
                'gradient_mean': np.mean(all_gradients),
                'gradient_std': np.std(all_gradients),
                'gradient_max': np.max(all_gradients),
                'gradient_min': np.min(all_gradients)
            }, step=step)
    
    def log_images(self, 
                   images: Dict[str, torch.Tensor],
                   ground_truth: torch.Tensor,
                   predictions: torch.Tensor,
                   sample_id: str,
                   step: Optional[int] = None) -> None:
        """
        Log images to wandb
        
        Args:
            images: Dictionary of camera images
            ground_truth: Ground truth segmentation
            predictions: Predicted segmentation
            sample_id: Sample identifier
            step: Optional step number
        """
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Sample: {sample_id}', fontsize=16)
        
        # Denormalize camera images
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        camera_views = ['front', 'left', 'rear', 'right']
        for i, view in enumerate(camera_views):
            if view in images:
                img = images[view][0]  # Take first sample
                denorm_img = img * std + mean
                denorm_img = torch.clamp(denorm_img, 0, 1)
                denorm_img = denorm_img.permute(1, 2, 0).cpu().numpy()
                
                axes[0, i].imshow(denorm_img)
                axes[0, i].set_title(f'{view.capitalize()} Camera')
                axes[0, i].axis('off')
        
        # Ground truth and prediction
        gt = ground_truth[0].cpu().numpy()
        pred = torch.argmax(predictions[0], dim=0).cpu().numpy()
        
        axes[1, 0].imshow(gt, cmap='tab10', vmin=0, vmax=6)
        axes[1, 0].set_title('Ground Truth')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(pred, cmap='tab10', vmin=0, vmax=6)
        axes[1, 1].set_title('Prediction')
        axes[1, 1].axis('off')
        
        # Error map
        error = np.abs(gt - pred)
        axes[1, 2].imshow(error, cmap='hot')
        axes[1, 2].set_title('Error Map')
        axes[1, 2].axis('off')
        
        # Class distribution
        gt_classes, gt_counts = np.unique(gt, return_counts=True)
        pred_classes, pred_counts = np.unique(pred, return_counts=True)
        
        axes[1, 3].bar(gt_classes, gt_counts, alpha=0.7, label='GT')
        axes[1, 3].bar(pred_classes, pred_counts, alpha=0.7, label='Pred')
        axes[1, 3].set_title('Class Distribution')
        axes[1, 3].legend()
        
        plt.tight_layout()
        
        # Log to wandb
        wandb.log({f"sample_{sample_id}": wandb.Image(fig)}, step=step)
        plt.close(fig)
        
        print(f"  - Logged images for sample: {sample_id}")
    
    def log_confusion_matrix(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            class_names: List[str],
                            step: Optional[int] = None) -> None:
        """
        Log confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            step: Optional step number
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        wandb.log({"confusion_matrix": wandb.Image(plt)}, step=step)
        plt.close()
        
        print(f"  - Logged confusion matrix")
    
    def log_histogram(self, 
                     values: np.ndarray, 
                     name: str,
                     step: Optional[int] = None) -> None:
        """
        Log histogram
        
        Args:
            values: Values to plot
            name: Name of the histogram
            step: Optional step number
        """
        plt.figure(figsize=(8, 6))
        plt.hist(values, bins=50, alpha=0.7)
        plt.title(f'Histogram: {name}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        
        wandb.log({name: wandb.Image(plt)}, step=step)
        plt.close()
        
        print(f"  - Logged histogram: {name}")
    
    def finish_run(self) -> None:
        """Finish the current wandb run"""
        wandb.finish()
        print("  - Finished wandb run")


def test_wandb_logger():
    """Test the wandb logger"""
    print("Testing WandB Logger...")
    
    # Create logger
    logger = WandBLogger(
        project_name="test-bev-segmentation",
        config={'test': True, 'backbone': 'resnet50'},
        tags=['test', 'debug']
    )
    
    # Initialize run
    logger.init_run("test_run")
    
    # Log some metrics
    metrics = {
        'loss': 0.5,
        'accuracy': 0.8,
        'miou': 0.6
    }
    logger.log_metrics(metrics)
    
    # Log model info
    model = torch.nn.Linear(10, 1)
    logger.log_model_info(model)
    
    # Finish run
    logger.finish_run()
    
    print("WandB logger test completed!")


if __name__ == "__main__":
    test_wandb_logger()
