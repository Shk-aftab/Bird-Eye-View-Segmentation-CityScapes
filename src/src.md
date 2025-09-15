# 🧠 Source Code Documentation

This directory contains the core implementation of the BEV Segmentation Pipeline. Each module is designed to be modular, testable, and well-documented.

## 📁 Module Overview

### **Core Pipeline Components**

**1. Backbone** (`backbone.py`) ✅
- **Input**: 4 camera views (384×384 RGB)
- **Output**: 64-channel features (12×12) per view
- **Options**: ResNet50 (default), ResNet18, EfficientNet-B0
- **Purpose**: Extract 2D features from each camera
- **Parameters**: 24.7M total, 1.2M trainable

**2. View Transform** (`view_transform.py`) ✅
- **Input**: 64-channel features (12×12) × 4 views
- **Output**: 64-channel BEV features (256×256)
- **Method**: Adaptive pooling + fusion (placeholder - not true LSS)
- **Purpose**: Project multi-view features to BEV space
- **Note**: This is a simplified placeholder, not the actual Lift-Splat-Shoot method

**3. BEV Encoder** (`bev_encoder.py`) ✅
- **Input**: 64-channel BEV features (256×256)
- **Output**: 128-channel encoded features (64×64)
- **Purpose**: Process BEV features with CNN layers
- **Architecture**: 64 → 128 → 128 channels

**4. Decoder** (`decoder.py`) ✅
- **Input**: 128-channel encoded features (64×64)
- **Output**: 7-class segmentation (256×256)
- **Purpose**: Generate final segmentation mask
- **Upsampling**: 64×64 → 128×128 → 256×256

**5. Model** (`model.py`) ✅
- **Purpose**: Combine all components into complete pipeline
- **Input**: 4 camera views → **Output**: 7-class BEV segmentation
- **Integration**: End-to-end forward pass

### **Data & Training Components**

**6. Dataset** (`dataset.py`) ✅
- **Classes**: 7 classes for autonomous driving
- **Mapping**: 0:unlabeled, 1:car, 2:vegetation, 3:road, 4:terrain, 5:guard_rail, 6:sidewalk
- **Purpose**: Load multi-view camera images and BEV labels
- **Samples**: 33,199 train + 3,731 validation

**7. Training** (`training.py`) ✅
- **Purpose**: End-to-end training with Focal Loss for class imbalance
- **Features**: Mixed precision, gradient clipping, comprehensive metrics
- **Loss**: Focal Loss (α=1.0, γ=2.0) for class imbalance

### **Utility & Management Components**

**8. Visualization** (`visualization.py`) ✅
- **Purpose**: 4 camera views + BEV comparison visualization
- **Features**: Ground truth vs prediction, class distribution
- **Output**: Comprehensive sample analysis

**9. Runs Management** (`runs_manager.py`) ✅
- **Purpose**: Experiment organization and checkpoint management
- **Features**: Automatic run directories, metrics tracking, CSV export
- **Structure**: `runs/run_TIMESTAMP/` with checkpoints and visualizations

**10. Weights & Biases** (`wandb_logger.py`) ✅
- **Purpose**: Experiment tracking and visualization
- **Features**: Metrics logging, model info, image logging
- **Integration**: Optional wandb support

**11. Temporal Fusion** (`temporal_fusion.py`) ✅
- **Purpose**: Fuse features from multiple timesteps
- **Input**: BEV features from multiple timesteps
- **Output**: Fused BEV features for better stability
- **Methods**: Attention, Conv3D, LSTM fusion
- **Benefits**: Better occlusion handling, temporal consistency

### **Utility Scripts**

**12. Class Weight Calculation** (`calculate_class_weights.py`) ✅
- **Purpose**: Pre-calculate class weights for imbalanced dataset
- **Method**: Inverse frequency weighting with sqrt normalization
- **Output**: Updates config file with calculated weights
- **Usage**: `python -m src.calculate_class_weights --config configs/config.yaml --num-samples 20`

**13. Model Evaluation** (`evaluate_model.py`) ✅
- **Purpose**: Evaluate trained models on validation samples
- **Features**: Random sampling, metrics calculation, visualization generation
- **Output**: Evaluation results and visualizations saved to run directory
- **Usage**: `python -m src.evaluate_model --checkpoint runs/run_TIMESTAMP/checkpoints/best_model.pth --num-samples 20`

## 🔧 Module Dependencies

```
model.py
├── backbone.py
├── view_transform.py
├── bev_encoder.py
├── decoder.py
└── temporal_fusion.py (optional)

training.py
├── model.py
├── dataset.py
├── runs_manager.py
├── wandb_logger.py
└── visualization.py

evaluate_model.py
├── model.py
├── dataset.py
├── training.py (for metrics)
└── visualization.py
```

## 🧪 Testing Individual Modules

```bash
# Test core components
python -m src.backbone
python -m src.view_transform
python -m src.bev_encoder
python -m src.decoder
python -m src.temporal_fusion

# Test complete model
python -m src.model

# Test data loading
python -m src.dataset

# Test utilities
python -m src.calculate_class_weights --help
python -m src.evaluate_model --help
```

## 📊 Expected Output Shapes

| Module | Input Shape | Output Shape | Purpose |
|--------|-------------|--------------|---------|
| **Backbone** | `[B, 3, 384, 384]` × 4 | `[B, 64, 12, 12]` × 4 | Feature extraction |
| **View Transform** | `[B, 64, 12, 12]` × 4 | `[B, 64, 256, 256]` | BEV projection |
| **BEV Encoder** | `[B, 64, 256, 256]` | `[B, 128, 64, 64]` | Feature processing |
| **Decoder** | `[B, 128, 64, 64]` | `[B, 7, 256, 256]` | Segmentation |
| **Temporal Fusion** | `[B, 64, 256, 256]` × T | `[B, 64, 256, 256]` | Multi-timestep fusion |

Where:
- `B` = Batch size
- `T` = Number of timesteps (3 for temporal fusion)
- `384×384` = Input image resolution
- `256×256` = BEV output resolution
- `7` = Number of classes


## ⚠️ Implementation Notes

### **View Transform Limitation**
The current `view_transform.py` implementation is **NOT** the actual Lift-Splat-Shoot (LSS) method. It uses:

**Current Implementation (Placeholder):**
- Adaptive pooling to resize features from 12×12 to 256×256
- Simple concatenation of multi-view features
- Basic convolution for feature fusion

**True LSS Implementation Would Include:**
- **Lift**: Predict depth distributions for each pixel
- **Splat**: Project features to 3D space using camera geometry
- **Shoot**: Rasterize to BEV grid with proper depth handling

### **Future Enhancement**
A proper LSS implementation would significantly improve:
- **Depth awareness** in BEV projection
- **Geometric accuracy** of multi-view fusion
- **Occlusion handling** through proper 3D reasoning

---

For the complete pipeline usage, see the main [README.md](../readme.md) file.
