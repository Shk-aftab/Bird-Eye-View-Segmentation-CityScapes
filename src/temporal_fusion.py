"""
Temporal Fusion Module for BEV Segmentation
Uses multiple timesteps to improve BEV stability and handle occlusions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np


class TemporalFusion(nn.Module):
    """
    Temporal fusion module that combines features from multiple timesteps
    """
    
    def __init__(self, 
                 feature_dim: int = 64,
                 num_timesteps: int = 3,
                 fusion_method: str = 'attention',
                 hidden_dim: int = 128):
        """
        Args:
            feature_dim: Dimension of input features
            num_timesteps: Number of timesteps to fuse (e.g., 3 for t-2, t-1, t0)
            fusion_method: 'attention', 'conv', or 'lstm'
            hidden_dim: Hidden dimension for fusion layers
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_timesteps = num_timesteps
        self.fusion_method = fusion_method
        
        if fusion_method == 'attention':
            self.fusion = TemporalAttention(feature_dim, num_timesteps, hidden_dim)
        elif fusion_method == 'conv':
            self.fusion = TemporalConv(feature_dim, num_timesteps, hidden_dim)
        elif fusion_method == 'lstm':
            self.fusion = TemporalLSTM(feature_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(self, temporal_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            temporal_features: List of features from different timesteps
                              Each tensor: [B, C, H, W]
        Returns:
            Fused features: [B, C, H, W]
        """
        if len(temporal_features) != self.num_timesteps:
            raise ValueError(f"Expected {self.num_timesteps} timesteps, got {len(temporal_features)}")
        
        return self.fusion(temporal_features)


class TemporalAttention(nn.Module):
    """Attention-based temporal fusion"""
    
    def __init__(self, feature_dim: int, num_timesteps: int, hidden_dim: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_timesteps = num_timesteps
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim * num_timesteps, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_timesteps, 1),
            nn.Softmax(dim=1)
        )
        
        # Feature projection
        self.feature_proj = nn.Conv2d(feature_dim, feature_dim, 1)
    
    def forward(self, temporal_features: List[torch.Tensor]) -> torch.Tensor:
        B, C, H, W = temporal_features[0].shape
        
        # Stack temporal features: [B, C*T, H, W]
        stacked = torch.cat(temporal_features, dim=1)
        
        # Compute attention weights: [B, T, H, W]
        attention_weights = self.attention(stacked)
        
        # Apply attention to features
        weighted_features = []
        for i, features in enumerate(temporal_features):
            weight = attention_weights[:, i:i+1, :, :]  # [B, 1, H, W]
            weighted = features * weight
            weighted_features.append(weighted)
        
        # Sum weighted features
        fused = sum(weighted_features)
        
        # Project to output dimension
        output = self.feature_proj(fused)
        
        return output


class TemporalConv(nn.Module):
    """Convolution-based temporal fusion"""
    
    def __init__(self, feature_dim: int, num_timesteps: int, hidden_dim: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_timesteps = num_timesteps
        
        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(feature_dim, hidden_dim, 
                     kernel_size=(num_timesteps, 3, 3), 
                     padding=(0, 1, 1)),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, feature_dim, 
                     kernel_size=(1, 1, 1)),
        )
    
    def forward(self, temporal_features: List[torch.Tensor]) -> torch.Tensor:
        # Stack temporal features: [B, C, T, H, W]
        stacked = torch.stack(temporal_features, dim=2)
        
        # Apply temporal convolution
        fused = self.temporal_conv(stacked)  # [B, C, 1, H, W]
        
        # Remove temporal dimension
        output = fused.squeeze(2)  # [B, C, H, W]
        
        return output


class TemporalLSTM(nn.Module):
    """LSTM-based temporal fusion"""
    
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, feature_dim)
    
    def forward(self, temporal_features: List[torch.Tensor]) -> torch.Tensor:
        B, C, H, W = temporal_features[0].shape
        
        # Reshape features: [B*H*W, T, C]
        reshaped = torch.stack(temporal_features, dim=1)  # [B, T, C, H, W]
        reshaped = reshaped.permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, T, C]
        reshaped = reshaped.view(B * H * W, -1, C)  # [B*H*W, T, C]
        
        # Apply LSTM
        lstm_out, _ = self.lstm(reshaped)  # [B*H*W, T, hidden_dim]
        
        # Take last timestep output
        last_output = lstm_out[:, -1, :]  # [B*H*W, hidden_dim]
        
        # Project to feature dimension
        projected = self.output_proj(last_output)  # [B*H*W, C]
        
        # Reshape back to spatial dimensions
        output = projected.view(B, C, H, W)
        
        return output


class TemporalDataset:
    """
    Dataset wrapper that provides temporal sequences
    """
    
    def __init__(self, base_dataset, num_timesteps: int = 3, timestep_interval: int = 500):
        """
        Args:
            base_dataset: Base BEV dataset
            num_timesteps: Number of timesteps in sequence
            timestep_interval: Interval between timesteps (in milliseconds)
        """
        self.base_dataset = base_dataset
        self.num_timesteps = num_timesteps
        self.timestep_interval = timestep_interval
        
        # Create temporal sample mapping
        self.temporal_samples = self._create_temporal_samples()
    
    def _create_temporal_samples(self):
        """Create mapping from current sample to temporal sequence"""
        temporal_samples = []
        
        # Create temporal sequences using simple sequential approach
        for i in range(len(self.base_dataset.sample_ids)):
            temporal_sequence = []
            
            # For each timestep, find the appropriate sample
            for t in range(self.num_timesteps):
                # Calculate target index (going backwards in time)
                target_idx = i - (self.num_timesteps - 1 - t)
                
                # Ensure we don't go out of bounds
                if target_idx < 0:
                    target_idx = 0  # Use first available sample
                elif target_idx >= len(self.base_dataset.sample_ids):
                    target_idx = len(self.base_dataset.sample_ids) - 1  # Use last available sample
                
                temporal_sequence.append(target_idx)
            
            temporal_samples.append(temporal_sequence)
        
        return temporal_samples
    
    def _find_closest_timestamp(self, target_timestamp: int, current_idx: int) -> int:
        """Find sample with closest timestamp to target within a local window"""
        min_diff = float('inf')
        closest_idx = current_idx
        
        # Search in a local window around current index to ensure temporal continuity
        search_window = min(100, len(self.base_dataset.sample_ids))  # Search within 100 samples
        start_idx = max(0, current_idx - search_window // 2)
        end_idx = min(len(self.base_dataset.sample_ids), current_idx + search_window // 2)
        
        for i in range(start_idx, end_idx):
            try:
                timestamp = int(self.base_dataset.sample_ids[i].split('_')[-1])
                diff = abs(timestamp - target_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i
            except:
                continue
        
        return closest_idx
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get temporal sequence indices
        temporal_indices = self.temporal_samples[idx]
        
        # Load temporal samples
        temporal_samples = []
        for t_idx in temporal_indices:
            sample = self.base_dataset[t_idx]
            temporal_samples.append(sample)
        
        # Return current sample with temporal context
        current_sample = temporal_samples[-1]  # Current timestep
        temporal_camera_images = [sample['camera_images'] for sample in temporal_samples]
        
        return {
            'camera_images': current_sample['camera_images'],
            'temporal_camera_images': temporal_camera_images,
            'bev_label': current_sample['bev_label'],
            'sample_id': current_sample['sample_id'],
            'temporal_indices': temporal_indices
        }


def test_temporal_fusion():
    """Test temporal fusion module"""
    print("Testing Temporal Fusion...")
    
    # Test parameters
    B, C, H, W = 2, 64, 32, 32
    num_timesteps = 3
    
    # Create test data
    temporal_features = [
        torch.randn(B, C, H, W) for _ in range(num_timesteps)
    ]
    
    # Test different fusion methods
    for method in ['attention', 'conv', 'lstm']:
        print(f"  Testing {method} fusion...")
        
        fusion = TemporalFusion(
            feature_dim=C,
            num_timesteps=num_timesteps,
            fusion_method=method
        )
        
        output = fusion(temporal_features)
        print(f"    Input shape: {[f.shape for f in temporal_features]}")
        print(f"    Output shape: {output.shape}")
        print(f"    ✅ {method} fusion working")
    
    print("Temporal fusion test completed!")


if __name__ == "__main__":
    test_temporal_fusion()
