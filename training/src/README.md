# Training Source Code

PyTorch-based training pipeline for behavioral cloning navigation models.

## Overview

This directory contains the complete training implementation for Phase 1 (baseline behavioral cloning) and Phase 2.5 (DAgger refinement) of the uncertainty-aware navigation system.

## Files

### dataset.py
**Purpose:** ROS bag processing and dataset preparation

**Key Functions:**

**`bag_to_arrays(bag_path, scan_topic, cmd_topic, max_range, num_beams)`**
- Extracts synchronized LiDAR scans and velocity commands from ROS bags
- Handles data cleaning (inf/nan replacement)
- Performs temporal alignment via nearest-neighbor matching
- Returns: (X, y) as numpy arrays

**`save_npz(output_path, *bag_paths)`**
- Converts multiple ROS bags to compressed .npz format
- Combines data from multiple demonstration sessions
- Efficient storage with numpy compression

**`load_npz(npz_path)`**
- Loads preprocessed dataset from .npz file
- Returns: (X, y) tensors ready for training

**Usage:**
```bash
# Convert bags to numpy
python dataset.py data/phase1.npz demo1.bag demo2.bag demo3.bag
```

**Python API:**
```python
from dataset import bag_to_arrays, save_npz, load_npz

# Single bag
X, y = bag_to_arrays('demo.bag')

# Multiple bags
save_npz('combined.npz', 'bag1.bag', 'bag2.bag', 'bag3.bag')

# Load for training
X, y = load_npz('combined.npz')
```

**Data Format:**
- Input: ROS bags with /scan and /cmd_vel topics
- Output: .npz with keys 'X' (LiDAR) and 'y' (velocities)
- X shape: (n_samples, 360)
- y shape: (n_samples, 2) - [linear_vel, angular_vel]

---

### model.py
**Purpose:** Neural network architectures for behavioral cloning

**Classes:**

**`BCPolicy(in_dim=360, hidden_dim=256, out_dim=2, dropout_rate=0.2)`**
- Fully-connected MLP for navigation
- Architecture: Linear(360→256) → ReLU → Dropout → Linear(256→256) → ReLU → Dropout → Linear(256→2)
- Supports deterministic and stochastic (MC Dropout) inference
- Parameters: ~200K trainable weights

**Key Methods:**
```python
model = BCPolicy(hidden_dim=256, dropout_rate=0.2)

# Deterministic prediction
vel = model.predict(lidar_scan, deterministic=True)

# MC Dropout uncertainty estimation (Phase 2)
mean_vel, uncertainty = model.predict_with_uncertainty(lidar_scan, n_samples=20)

# Save/load
model.save('models/phase1.pt')
loaded = BCPolicy.load('models/phase1.pt')
```

**`Conv1DPolicy(in_channels=1, hidden_dim=64, out_dim=2, dropout_rate=0.2)`**
- 1D Convolutional network for sequential LiDAR processing
- Architecture: Conv1D(16)→Pool→Conv1D(32)→Pool→Flatten→Linear(64)→Linear(2)
- Preserves spatial relationships in LiDAR data
- Matches thesis Conv1D CNN architecture

**Features:**
- Xavier weight initialization
- MC Dropout support for uncertainty quantification
- Model checkpointing with configuration
- Parameter counting utility

**Example:**
```python
from model import BCPolicy

# Create model
model = BCPolicy(in_dim=360, hidden_dim=256)
print(f"Parameters: {model.get_num_parameters():,}")

# Forward pass
import torch
x = torch.randn(4, 360)  # Batch of 4 LiDAR scans
y = model(x)  # Output: (4, 2)

# Uncertainty estimation
mean, std = model.predict_with_uncertainty(x, n_samples=20)
overall_uncertainty = std.mean(dim=1)  # Per-sample uncertainty
```

---

### train.py
**Purpose:** Complete training pipeline with validation and logging

**Main Function:**
```python
train(
    data_path="data/phase1.npz",
    output_path="models/policy.pt",
    architecture="mlp",
    hidden_dim=256,
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    val_split=0.2,
    dropout_rate=0.2,
    device="cpu"
)
```

**Features:**
- Train/validation split with monitoring
- Best model checkpointing based on validation loss
- Training history logging (JSON export)
- Support for MLP and Conv1D architectures
- Early stopping capability
- Comprehensive metrics logging

**Command-line Arguments:**
```
--data          Path to .npz dataset
--output        Path to save trained model
--arch          Architecture: 'mlp' or 'conv1d'
--hidden-dim    Hidden layer dimension (default: 256)
--epochs        Training epochs (default: 100)
--batch-size    Batch size (default: 32)
--lr            Learning rate (default: 0.001)
--val-split     Validation fraction (default: 0.2)
--dropout       Dropout rate (default: 0.2)
--device        'cpu' or 'cuda'
--seed          Random seed (default: 42)
```

**Usage Examples:**
```bash
# Phase 1: Baseline behavioral cloning
python train.py \
  --data data/phase1.npz \
  --output models/phase1_bc.pt \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.001

# Phase 2.5: DAgger refinement
python train.py \
  --data data/combined.npz \
  --output models/phase2.5_dagger.pt \
  --epochs 100 \
  --hidden-dim 256 \
  --dropout 0.2

# Conv1D architecture (thesis implementation)
python train.py \
  --arch conv1d \
  --data data/phase1.npz \
  --output models/phase1_conv1d.pt \
  --hidden-dim 64

# GPU training
python train.py --device cuda --batch-size 64

# Custom hyperparameters
python train.py \
  --hidden-dim 512 \
  --lr 0.0001 \
  --dropout 0.3 \
  --val-split 0.3
```

**Output Files:**
- `<output>.pt`: Trained model checkpoint (PyTorch state dict + config)
- `<output>.json`: Training history with losses and best epoch

**Training History JSON:**
```json
{
  "train_losses": [0.015, 0.012, 0.010, ...],
  "val_losses": [0.016, 0.013, 0.011, ...],
  "best_epoch": 87,
  "best_val_loss": 0.006
}
```

---

## Complete Training Workflow

### Phase 1: Baseline Behavioral Cloning
```bash
# Step 1: Convert ROS bags to numpy
python dataset.py \
  data/phase1.npz \
  rosbags/demo1.bag \
  rosbags/demo2.bag \
  rosbags/demo3.bag

# Step 2: Train model
python train.py \
  --data data/phase1.npz \
  --output models/phase1_bc.pt \
  --arch mlp \
  --hidden-dim 256 \
  --epochs 100 \
  --batch-size 32

# Step 3: Verify training
# Check models/phase1_bc.json for training curves
```

**Expected Results:**
- Training samples: ~890 (80% of 1,113)
- Validation samples: ~223 (20% of 1,113)
- Final test loss: ~0.010 MSE
- Training time: ~3 minutes on CPU

---

### Phase 2.5: DAgger Refinement
```bash
# Step 1: Merge Phase 1 + DAgger demonstrations
python dataset.py \
  data/combined.npz \
  rosbags/phase1_demo1.bag \
  rosbags/phase1_demo2.bag \
  rosbags/phase1_demo3.bag \
  rosbags/dagger_demo1.bag \
  rosbags/dagger_demo2.bag

# Step 2: Train on combined dataset
python train.py \
  --data data/combined.npz \
  --output models/phase2.5_dagger.pt \
  --arch mlp \
  --hidden-dim 256 \
  --epochs 100 \
  --batch-size 32 \
  --dropout 0.2

# Step 3: Compare with Phase 1
# Phase 1 test loss: 0.010
# Phase 2.5 test loss: 0.006 (40% improvement)
```

**Expected Results:**
- Training samples: ~1,680 (80% of 2,100)
- Validation samples: ~420 (20% of 2,100)
- Final test loss: ~0.006 MSE
- Training time: ~3 minutes on CPU

---

## Model Deployment

After training, models can be deployed in the ROS2 navigation system:
```python
import torch
from model import BCPolicy

# Load trained model
model = BCPolicy.load('models/phase2.5_dagger.pt', device='cpu')

# Inference with uncertainty
lidar_scan = torch.tensor(scan_data).unsqueeze(0)  # (1, 360)
mean_vel, uncertainty = model.predict_with_uncertainty(lidar_scan, n_samples=20)

linear_vel = mean_vel[0, 0].item()
angular_vel = mean_vel[0, 1].item()
total_uncertainty = uncertainty.mean().item()

print(f"Velocity: [{linear_vel:.3f}, {angular_vel:.3f}]")
print(f"Uncertainty: {total_uncertainty:.3f}")
```

---

## Requirements
```
torch>=1.9.0
numpy>=1.19.0
rosbags>=0.9.0
```

Install: `pip install torch numpy rosbags`

---

## Notes

**Data Preprocessing:**
- LiDAR scans normalized to [0, 1] by dividing by max range
- Invalid measurements (inf/nan) replaced with max_range (3.5m)
- Temporal synchronization uses nearest-neighbor matching

**Training Best Practices:**
- Use validation split to monitor overfitting
- Phase 1 typically converges in 50-100 epochs
- Dropout rate of 0.2 works well for uncertainty estimation
- Batch size of 32 balances speed and stability
- Save best model based on validation loss, not final epoch

**MC Dropout Inference:**
- Use n_samples=20 for good uncertainty estimates
- More samples = better estimates but slower (linear cost)
- Enable dropout during inference: `model.train()`
- Uncertainty = std dev across stochastic predictions

**Troubleshooting:**
- High training loss: Check data normalization, reduce learning rate
- Overfitting: Increase dropout, reduce model size, add more data
- Slow training: Reduce batch size, use GPU, simplify architecture
- Poor uncertainty calibration: Ensure dropout enabled during MC sampling
