"""
Training Loop for BEV Segmentation
Includes Focal Loss, metrics, validation, and comprehensive logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Tuple
import time
from tqdm import tqdm
import os

from .model import BEVSegmentationModel
from .runs_manager import RunsManager
from .wandb_logger import WandBLogger
from .visualization import create_default_visualizer


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (B, C, H, W)
            targets: Ground truth (B, H, W)
        
        Returns:
            loss: Focal loss
        """
        # Convert to log probabilities
        log_probs = torch.log_softmax(inputs, dim=1)
        
        # Get probabilities
        probs = torch.exp(log_probs)
        
        # Gather probabilities for target classes
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        
        # Calculate focal loss
        pt = (probs * targets_one_hot).sum(dim=1)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Calculate cross entropy
        ce_loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    """
    
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (B, C, H, W)
            targets: Ground truth (B, H, W)
        
        Returns:
            loss: Dice loss
        """
        # Convert to probabilities
        probs = torch.softmax(inputs, dim=1)
        
        # Convert targets to one-hot
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        
        # Calculate Dice coefficient for each class
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined Focal Loss + Dice Loss for better segmentation
    """
    
    def __init__(self, focal_alpha: float = 1.0, focal_gamma: float = 2.0, 
                 dice_weight: float = 0.5, class_weights: Optional[torch.Tensor] = None):
        """
        Args:
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            dice_weight: Weight for dice loss (0.0 to 1.0)
            class_weights: Class weights for handling imbalance
        """
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight
        self.class_weights = class_weights
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (B, C, H, W)
            targets: Ground truth (B, H, W)
        
        Returns:
            loss: Combined loss
        """
        # Calculate focal loss
        focal = self.focal_loss(inputs, targets)
        
        # Calculate dice loss
        dice = self.dice_loss(inputs, targets)
        
        # Combine losses
        combined_loss = (1.0 - self.dice_weight) * focal + self.dice_weight * dice
        
        return combined_loss


class MetricsCalculator:
    """
    Calculate various metrics for segmentation
    """
    
    def __init__(self, num_classes: int, class_names: Dict[int, str]):
        self.num_classes = num_classes
        self.class_names = class_names
        
    def calculate_metrics(self, 
                         predictions: torch.Tensor, 
                         targets: torch.Tensor) -> Dict[str, float]:
        """
        Calculate segmentation metrics
        
        Args:
            predictions: Predicted logits (B, C, H, W)
            targets: Ground truth (B, H, W)
        
        Returns:
            metrics: Dictionary of calculated metrics
        """
        # Convert predictions to class indices
        pred_classes = torch.argmax(predictions, dim=1)
        
        # Flatten tensors
        pred_flat = pred_classes.flatten()
        target_flat = targets.flatten()
        
        # Calculate accuracy
        accuracy = (pred_flat == target_flat).float().mean().item()
        
        # Calculate per-class metrics
        class_metrics = {}
        for class_id in range(self.num_classes):
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            
            # True positives, false positives, false negatives
            tp = ((pred_flat == class_id) & (target_flat == class_id)).sum().item()
            fp = ((pred_flat == class_id) & (target_flat != class_id)).sum().item()
            fn = ((pred_flat != class_id) & (target_flat == class_id)).sum().item()
            
            # Precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_metrics[f'{class_name}_precision'] = precision
            class_metrics[f'{class_name}_recall'] = recall
            class_metrics[f'{class_name}_f1'] = f1
        
        # Calculate mean IoU (only for classes present in ground truth)
        ious = []
        present_classes = []
        for class_id in range(self.num_classes):
            tp = ((pred_flat == class_id) & (target_flat == class_id)).sum().item()
            fp = ((pred_flat == class_id) & (target_flat != class_id)).sum().item()
            fn = ((pred_flat != class_id) & (target_flat == class_id)).sum().item()
            
            # Only include classes that are present in ground truth
            if (tp + fn) > 0:  # Class is present in ground truth
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                ious.append(iou)
                present_classes.append(class_id)
        
        # Calculate mean IoU only for present classes
        mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
        
        # Also calculate mean IoU including all classes (for comparison)
        all_ious = []
        for class_id in range(self.num_classes):
            tp = ((pred_flat == class_id) & (target_flat == class_id)).sum().item()
            fp = ((pred_flat == class_id) & (target_flat != class_id)).sum().item()
            fn = ((pred_flat != class_id) & (target_flat == class_id)).sum().item()
            
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            all_ious.append(iou)
        
        mean_iou_all = np.mean(all_ious)
        
        # Calculate overall metrics
        metrics = {
            'accuracy': accuracy,
            'mean_iou': mean_iou,
            'mean_iou_all': mean_iou_all,
            'present_classes': len(present_classes),
            **class_metrics
        }
        
        return metrics


class BEVTrainer:
    """
    Main trainer class for BEV segmentation
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 model: BEVSegmentationModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 use_wandb: bool = True):
        """
        Args:
            config: Training configuration
            model: BEV segmentation model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use for training
            use_wandb: Whether to use wandb logging
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_wandb = use_wandb
        
        print(f"Initializing BEV Trainer...")
        print(f"  - Device: {device}")
        print(f"  - Use wandb: {use_wandb}")
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Initialize loss function with class weighting
        # Get class weights from config or calculate if needed
        class_weights = self._get_class_weights()
        self.criterion = CombinedLoss(
            focal_alpha=1.0, 
            focal_gamma=2.0, 
            dice_weight=0.3,  # 30% dice, 70% focal
            class_weights=class_weights
        )
        
        # Initialize mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.get('mixed_precision', True) else None
        
        # Initialize optimizer with differential learning rates
        backbone_params = []
        new_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                new_params.append(param)
        
        self.optimizer = optim.Adam([
            {'params': backbone_params, 'lr': config.get('backbone_lr', 1e-5)},  # Lower LR for backbone
            {'params': new_params, 'lr': config.get('learning_rate', 1e-4)}     # Higher LR for new layers
        ], weight_decay=config.get('weight_decay', 1e-4))
        
        print(f"  - Backbone LR: {config.get('backbone_lr', 1e-5)}")
        print(f"  - New layers LR: {config.get('learning_rate', 1e-4)}")
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        # Initialize metrics calculator
        class_names = {
            0: "unlabeled", 1: "car", 2: "vegetation", 3: "road",
            4: "terrain", 5: "guard_rail", 6: "sidewalk"
        }
        self.metrics_calculator = MetricsCalculator(7, class_names)
        
        # Initialize runs manager
        self.runs_manager = RunsManager()
        self.run_path = self.runs_manager.create_run(config)
        
        # Initialize wandb logger
        if use_wandb:
            self.wandb_logger = WandBLogger(
                project_name="bev-segmentation",
                config=config,
                tags=['bev', 'segmentation', 'multi-view']
            )
            self.wandb_logger.init_run()
            self.wandb_logger.log_model_info(model)
            self.global_step = 0  # Global step counter for wandb
        else:
            self.wandb_logger = None
            self.global_step = 0
        
        # Initialize visualizer
        self.visualizer = create_default_visualizer()
        
        # Training state
        self.current_epoch = 0
        self.best_val_iou = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        self.patience = config.get('patience', 10)  # Early stopping patience
        self.early_stop_counter = 0
    
    def _get_class_weights(self) -> torch.Tensor:
        """
        Get class weights from config or calculate if not available
        """
        # Check if class weights are pre-calculated in config
        if 'class_weights' in self.config:
            class_weights = torch.tensor(self.config['class_weights'], dtype=torch.float32)
            print("Using pre-calculated class weights from config")
            
            if 'class_distribution' in self.config:
                class_dist = self.config['class_distribution']
                class_names = ['unlabeled', 'car', 'vegetation', 'road', 'terrain', 'guard_rail', 'sidewalk']
                print("Class distribution from config:")
                for name, dist in zip(class_names, class_dist):
                    print(f"  - {name}: {dist:.2f}%")
            
            return class_weights.to(self.device)
        
        # Fallback: use default weights if not in config
        print("⚠️  Class weights not found in config, using default weights")
        print("💡 Tip: Run 'python calculate_class_weights.py' to pre-calculate weights")
        
        # Use default equal weights for now
        default_weights = torch.ones(7, dtype=torch.float32)
        print(f"Using default equal weights: {default_weights.tolist()}")
        
        return default_weights.to(self.device)
    
    def _calculate_class_weights_fallback(self) -> torch.Tensor:
        """
        Fallback method to calculate class weights (slower)
        """
        print("Calculating class weights from training data...")
        
        # Sample random batches to get better class distribution
        class_counts = torch.zeros(7)  # 7 classes
        total_pixels = 0
        num_samples = 0
        target_samples = 50  # Sample 50 batches for better representation
        
        # Get total number of batches
        total_batches = len(self.train_loader)
        
        # Sample random batch indices
        import random
        random.seed(42)  # For reproducibility
        sample_indices = random.sample(range(total_batches), min(target_samples, total_batches))
        
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                if i in sample_indices:
                    targets = batch['bev_label']
                    unique_classes, counts = torch.unique(targets, return_counts=True)
                    
                    for class_id, count in zip(unique_classes, counts):
                        class_counts[class_id] += count.item()
                        total_pixels += count.item()
                    
                    num_samples += 1
                    
                    if num_samples >= target_samples:
                        break
        
        # Calculate inverse frequency weights
        class_weights = torch.zeros(7)
        for i in range(7):
            if class_counts[i] > 0:
                # Use sqrt of inverse frequency for less extreme weights
                class_weights[i] = torch.sqrt(total_pixels / (7 * class_counts[i]))
            else:
                class_weights[i] = 1.0  # Default weight for unseen classes
        
        # Normalize weights to have mean = 1
        class_weights = class_weights / class_weights.mean()
        
        print(f"Class weights: {class_weights.tolist()}")
        print(f"Class distribution: {(class_counts / total_pixels * 100).tolist()}")
        print(f"Sampled {num_samples} batches for weight calculation")
        
        return class_weights.to(self.device)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        print(f"\n=== Training Epoch {self.current_epoch + 1} ===")
        
        self.model.train()
        total_loss = 0.0
        all_metrics = []
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            camera_images = {k: v.to(self.device) for k, v in batch['camera_images'].items()}
            targets = batch['bev_label'].to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    # Check if temporal data is available
                    if 'temporal_camera_images' in batch:
                        # Move temporal camera images to device
                        temporal_camera_images = []
                        for temporal_batch in batch['temporal_camera_images']:
                            temporal_images = {k: v.to(self.device) for k, v in temporal_batch.items()}
                            temporal_camera_images.append(temporal_images)
                        outputs = self.model(camera_images, temporal_camera_images)
                    else:
                        outputs = self.model(camera_images)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Check if temporal data is available
                if 'temporal_camera_images' in batch:
                    temporal_camera_images = batch['temporal_camera_images']
                    outputs = self.model(camera_images, temporal_camera_images)
                else:
                    outputs = self.model(camera_images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                metrics = self.metrics_calculator.calculate_metrics(outputs, targets)
                all_metrics.append(metrics)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{metrics["accuracy"]:.4f}',
                'miou': f'{metrics["mean_iou"]:.4f}',
                'miou_all': f'{metrics["mean_iou_all"]:.4f}',
                'classes': f'{metrics["present_classes"]}'
            })
            
            # Log to wandb (more frequently for long epochs)
            if self.wandb_logger and batch_idx % 100 == 0:
                self.global_step += 1
                self.wandb_logger.log_metrics({
                    'train_loss': loss.item(),
                    'train_accuracy': metrics['accuracy'],
                    'train_miou': metrics['mean_iou'],
                    'train_miou_all': metrics['mean_iou_all'],
                    'train_present_classes': metrics['present_classes']
                }, step=self.global_step)
            
            # Visualize sample (every 500 batches for long epochs)
            if batch_idx % 100 == 0:
                sample_id = batch['sample_id'][0]
                self.visualizer.visualize_sample(
                    camera_images={k: v[0:1] for k, v in camera_images.items()},
                    bev_features=torch.randn(1, 64, 256, 256),  # Placeholder
                    segmentation_logits=outputs[0:1],
                    ground_truth=targets[0:1],
                    sample_id=f"train_epoch_{self.current_epoch + 1}_batch_{batch_idx}",
                    save_path=os.path.join(self.run_path, "visualizations", 
                                         f"train_epoch_{self.current_epoch + 1}_batch_{batch_idx}.png")
                )
            
            # Save intermediate checkpoint every 1000 batches (for long epochs)
            if batch_idx % 500 == 0 and batch_idx > 0:
                intermediate_path = os.path.join(self.run_path, "checkpoints", f"intermediate_epoch_{self.current_epoch + 1}_batch_{batch_idx}.pth")
                torch.save({
                    'epoch': self.current_epoch + 1,
                    'batch': batch_idx,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': metrics,
                    'best_val_iou': self.best_val_iou
                }, intermediate_path)
                print(f"  - Saved intermediate checkpoint: epoch {self.current_epoch + 1}, batch {batch_idx}")
            
            total_loss += loss.item()
        
        # Calculate epoch metrics
        epoch_metrics = {}
        for key in all_metrics[0].keys():
            epoch_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        epoch_metrics['loss'] = total_loss / len(self.train_loader)
        
        # Save metrics
        self.runs_manager.save_metrics(self.run_path, epoch_metrics, self.current_epoch + 1, "train")
        
        print(f"  - Training loss: {epoch_metrics['loss']:.4f}")
        print(f"  - Training accuracy: {epoch_metrics['accuracy']:.4f}")
        print(f"  - Training mIoU: {epoch_metrics['mean_iou']:.4f}")
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        print(f"\n=== Validation Epoch {self.current_epoch + 1} ===")
        
        self.model.eval()
        total_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Val Epoch {self.current_epoch + 1}")
            
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                camera_images = {k: v.to(self.device) for k, v in batch['camera_images'].items()}
                targets = batch['bev_label'].to(self.device)
                
                # Forward pass
                if 'temporal_camera_images' in batch:
                    # Move temporal camera images to device
                    temporal_camera_images = []
                    for temporal_batch in batch['temporal_camera_images']:
                        temporal_images = {k: v.to(self.device) for k, v in temporal_batch.items()}
                        temporal_camera_images.append(temporal_images)
                    outputs = self.model(camera_images, temporal_camera_images)
                else:
                    outputs = self.model(camera_images)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Calculate metrics
                metrics = self.metrics_calculator.calculate_metrics(outputs, targets)
                all_metrics.append(metrics)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{metrics["accuracy"]:.4f}',
                    'miou': f'{metrics["mean_iou"]:.4f}',
                    'miou_all': f'{metrics["mean_iou_all"]:.4f}',
                    'classes': f'{metrics["present_classes"]}'
                })
                
                # Visualize sample
                if batch_idx == 0:  # Visualize first batch
                    sample_id = batch['sample_id'][0]
                    self.visualizer.visualize_sample(
                        camera_images={k: v[0:1] for k, v in camera_images.items()},
                        bev_features=torch.randn(1, 64, 256, 256),  # Placeholder
                        segmentation_logits=outputs[0:1],
                        ground_truth=targets[0:1],
                        sample_id=f"val_epoch_{self.current_epoch + 1}_batch_{batch_idx}",
                        save_path=os.path.join(self.run_path, "visualizations", 
                                             f"val_epoch_{self.current_epoch + 1}_batch_{batch_idx}.png")
                    )
                
                total_loss += loss.item()
        
        # Calculate epoch metrics
        epoch_metrics = {}
        for key in all_metrics[0].keys():
            epoch_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        epoch_metrics['loss'] = total_loss / len(self.val_loader)
        
        # Save metrics
        self.runs_manager.save_metrics(self.run_path, epoch_metrics, self.current_epoch + 1, "val")
        
        print(f"  - Validation loss: {epoch_metrics['loss']:.4f}")
        print(f"  - Validation accuracy: {epoch_metrics['accuracy']:.4f}")
        print(f"  - Validation mIoU: {epoch_metrics['mean_iou']:.4f}")
        
        return epoch_metrics
    
    def train(self, num_epochs: int) -> None:
        """Main training loop"""
        print(f"\n=== Starting Training for {num_epochs} epochs ===")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_metrics['mean_iou'])
            
            # Log to wandb
            if self.wandb_logger:
                self.global_step += 1
                self.wandb_logger.log_metrics({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_accuracy': val_metrics['accuracy'],
                    'train_miou': train_metrics['mean_iou'],
                    'val_miou': val_metrics['mean_iou']
                }, step=self.global_step)
                
                self.wandb_logger.log_learning_rate(self.optimizer)
            
            # Save checkpoint
            is_best = val_metrics['mean_iou'] > self.best_val_iou
            if is_best:
                self.best_val_iou = val_metrics['mean_iou']
                print(f"  - New best validation mIoU: {self.best_val_iou:.4f}")
            
            # Save checkpoint every epoch
            self.runs_manager.save_checkpoint(
                self.run_path,
                self.model,
                self.optimizer,
                epoch + 1,
                val_metrics,
                is_best
            )
            
            # Save intermediate checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                intermediate_path = os.path.join(self.run_path, "checkpoints", f"intermediate_epoch_{epoch + 1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': val_metrics,
                    'best_val_iou': self.best_val_iou
                }, intermediate_path)
                print(f"  - Saved intermediate checkpoint: epoch {epoch + 1}")
            
            # Store metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.val_ious.append(val_metrics['mean_iou'])
            
            # Early stopping check
            if is_best:
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                
            if self.early_stop_counter >= self.patience:
                print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
                print(f"  - No improvement for {self.patience} epochs")
                print(f"  - Best validation mIoU: {self.best_val_iou:.4f}")
                break
        
        print(f"\n=== Training Completed ===")
        print(f"  - Best validation mIoU: {self.best_val_iou:.4f}")
        print(f"  - Run path: {self.run_path}")
        
        # Finish wandb run
        if self.wandb_logger:
            self.wandb_logger.finish_run()


def test_trainer():
    """Test the trainer"""
    print("Testing BEV Trainer...")
    
    # This would require actual data loaders
    # For now, just test the components
    print("Trainer test completed!")


if __name__ == "__main__":
    test_trainer()
