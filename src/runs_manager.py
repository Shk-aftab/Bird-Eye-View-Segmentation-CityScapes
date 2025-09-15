"""
Runs Management System
Handles experiment runs, checkpoints, and metrics storage
"""

import os
import json
import yaml
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import torch


class RunsManager:
    """
    Manages experiment runs, checkpoints, and metrics
    """
    
    def __init__(self, runs_dir: str = "runs"):
        """
        Args:
            runs_dir: Base directory for all runs
        """
        self.runs_dir = runs_dir
        os.makedirs(runs_dir, exist_ok=True)
        
        print(f"Initializing Runs Manager...")
        print(f"  - Runs directory: {self.runs_dir}")
        
    def create_run(self, config: Dict[str, Any], run_name: Optional[str] = None) -> str:
        """
        Create a new experiment run
        
        Args:
            config: Configuration dictionary
            run_name: Optional custom run name
        
        Returns:
            run_path: Path to the created run directory
        """
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"
        
        run_path = os.path.join(self.runs_dir, run_name)
        os.makedirs(run_path, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(run_path, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(run_path, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(run_path, "metrics"), exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(run_path, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"  - Created run: {run_name}")
        print(f"  - Run path: {run_path}")
        
        return run_path
    
    def save_checkpoint(self, 
                       run_path: str, 
                       model: torch.nn.Module, 
                       optimizer: torch.optim.Optimizer,
                       epoch: int, 
                       metrics: Dict[str, float],
                       is_best: bool = False) -> str:
        """
        Save model checkpoint
        
        Args:
            run_path: Path to run directory
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model so far
        
        Returns:
            checkpoint_path: Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(run_path, "checkpoints", f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save as best model if applicable
        if is_best:
            best_path = os.path.join(run_path, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"  - Saved best model at epoch {epoch}")
        
        print(f"  - Saved checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def save_metrics(self, 
                    run_path: str, 
                    metrics: Dict[str, float], 
                    epoch: int,
                    phase: str = "train") -> str:
        """
        Save metrics to CSV file
        
        Args:
            run_path: Path to run directory
            metrics: Metrics dictionary
            epoch: Current epoch
            phase: Phase (train/val)
        
        Returns:
            metrics_path: Path to metrics file
        """
        metrics_path = os.path.join(run_path, "metrics", f"{phase}_metrics.csv")
        
        # Add epoch and phase to metrics
        metrics_data = {
            'epoch': epoch,
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        # Load existing metrics or create new
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            new_row = pd.DataFrame([metrics_data])
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = pd.DataFrame([metrics_data])
        
        # Save metrics
        df.to_csv(metrics_path, index=False)
        
        print(f"  - Saved {phase} metrics for epoch {epoch}")
        return metrics_path
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            checkpoint: Loaded checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"  - Loaded checkpoint from: {checkpoint_path}")
        return checkpoint
    
    def list_runs(self) -> List[Dict[str, Any]]:
        """
        List all available runs
        
        Returns:
            runs: List of run information
        """
        runs = []
        
        if not os.path.exists(self.runs_dir):
            return runs
        
        for run_name in os.listdir(self.runs_dir):
            run_path = os.path.join(self.runs_dir, run_name)
            if not os.path.isdir(run_path):
                continue
            
            # Load run info
            run_info = {
                'name': run_name,
                'path': run_path,
                'created': os.path.getctime(run_path)
            }
            
            # Load config if available
            config_path = os.path.join(run_path, "config.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    run_info['config'] = yaml.safe_load(f)
            
            # Load best metrics if available
            best_model_path = os.path.join(run_path, "best_model.pth")
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path, map_location='cpu')
                run_info['best_metrics'] = checkpoint.get('metrics', {})
                run_info['best_epoch'] = checkpoint.get('epoch', 0)
            
            runs.append(run_info)
        
        # Sort by creation time (newest first)
        runs.sort(key=lambda x: x['created'], reverse=True)
        
        return runs
    
    def get_run_summary(self, run_name: str) -> Dict[str, Any]:
        """
        Get summary of a specific run
        
        Args:
            run_name: Name of the run
        
        Returns:
            summary: Run summary dictionary
        """
        run_path = os.path.join(self.runs_dir, run_name)
        
        if not os.path.exists(run_path):
            raise ValueError(f"Run not found: {run_name}")
        
        summary = {
            'name': run_name,
            'path': run_path,
            'created': os.path.getctime(run_path)
        }
        
        # Load config
        config_path = os.path.join(run_path, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                summary['config'] = yaml.safe_load(f)
        
        # Load best model info
        best_model_path = os.path.join(run_path, "best_model.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location='cpu')
            summary['best_metrics'] = checkpoint.get('metrics', {})
            summary['best_epoch'] = checkpoint.get('epoch', 0)
        
        # Load metrics files
        metrics_dir = os.path.join(run_path, "metrics")
        if os.path.exists(metrics_dir):
            summary['metrics_files'] = os.listdir(metrics_dir)
        
        # Count checkpoints
        checkpoints_dir = os.path.join(run_path, "checkpoints")
        if os.path.exists(checkpoints_dir):
            summary['num_checkpoints'] = len([f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')])
        
        return summary
    
    def cleanup_old_runs(self, keep_last: int = 5) -> List[str]:
        """
        Clean up old runs, keeping only the most recent ones
        
        Args:
            keep_last: Number of recent runs to keep
        
        Returns:
            removed_runs: List of removed run names
        """
        runs = self.list_runs()
        
        if len(runs) <= keep_last:
            print(f"  - No cleanup needed. Only {len(runs)} runs found.")
            return []
        
        # Remove old runs
        removed_runs = []
        for run in runs[keep_last:]:
            run_path = run['path']
            shutil.rmtree(run_path)
            removed_runs.append(run['name'])
            print(f"  - Removed old run: {run['name']}")
        
        print(f"  - Cleaned up {len(removed_runs)} old runs")
        return removed_runs


def test_runs_manager():
    """Test the runs manager"""
    print("Testing Runs Manager...")
    
    # Create runs manager
    manager = RunsManager("test_runs")
    
    # Create a test run
    config = {
        'backbone': 'resnet50',
        'num_classes': 7,
        'batch_size': 4
    }
    
    run_path = manager.create_run(config, "test_run")
    
    # Test metrics saving
    metrics = {
        'loss': 0.5,
        'accuracy': 0.8,
        'miou': 0.6
    }
    
    manager.save_metrics(run_path, metrics, epoch=1, phase="train")
    manager.save_metrics(run_path, metrics, epoch=1, phase="val")
    
    # Test run listing
    runs = manager.list_runs()
    print(f"Found {len(runs)} runs")
    
    # Test run summary
    summary = manager.get_run_summary("test_run")
    print(f"Run summary: {summary}")
    
    # Cleanup
    shutil.rmtree("test_runs")
    print("Runs manager test completed!")


if __name__ == "__main__":
    test_runs_manager()
