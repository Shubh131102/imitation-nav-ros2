# Scripts Directory

Training, evaluation, and data processing scripts for the uncertainty-aware navigation system.

## Files Overview

### data_prep.py
**Purpose:** Load and preprocess LiDAR and velocity data from CSV files

**Key Functions:**
- `load_dataset(data_dir, max_range, downsample)`: Load and synchronize LiDAR scans with velocity commands
- `split_dataset(X, y, test_ratio, random_seed)`: Split data into train/test sets

**Features:**
- Temporal synchronization between LiDAR and velocity topics
- Handles missing data (replaces inf/nan with max range)
- Optional LiDAR downsampling
- Train/test splitting with configurable ratio

**Usage:**
```python
from data_prep import load_dataset, split_dataset

# Load data
X, y = load_dataset("data/expert_demos", max_range=3.5)

# Split into train/test
X_train, X_test, y_train, y_test = split_dataset(X, y, test_ratio=0.2)
```

**Input Format:**
- `scan.csv`: [timestamp, lidar_0, lidar_1, ..., lidar_359]
- `cmd_vel.csv`: [timestamp, vx, wz]

**Output:**
- X: numpy array (n_samples, 360) - LiDAR measurements
- y: numpy array (n_samples, 2) - [linear_velocity, angular_velocity]

---

### train.py
**Purpose:** Train behavioral cloning model on expert demonstrations

**Features:**
- Configurable MLP architecture
- Early stopping to prevent overfitting
- Training history logging and visualization
- Automatic model checkpointing
- Comprehensive metrics (MSE, R², training time)

**Command-line Arguments:**
```
--data              Path to data directory (default: "data")
--output            Path to save trained model (default: "models/bc_mlp.joblib")
--hidden-layers     Hidden layer sizes (default: 128 64)
--max-iter          Maximum training iterations (default: 500)
--learning-rate     Initial learning rate (default: 0.001)
--test-ratio        Test set ratio (default: 0.2)
--seed              Random seed (default: 42)
--plot              Generate training curve plot
```

**Usage Examples:**
```bash
# Basic training
python scripts/train.py

# Custom architecture
python scripts/train.py --hidden-layers 256 128 64 --max-iter 1000

# Phase 1 training
python scripts/train.py --data data/expert_demos --output models/phase1_bc.joblib

# Phase 2.5 training (DAgger)
python scripts/train.py --data data/combined --output models/phase2.5_dagger.joblib --plot

# Custom hyperparameters
python scripts/train.py --learning-rate 0.0001 --test-ratio 0.3
```

**Output Files:**
- `<output>.joblib`: Trained model (scikit-learn format)
- `<output>.json`: Training history and metrics
- `<output>_loss_curve.png`: Training loss visualization (if --plot used)

---

### eval.py
**Purpose:** Comprehensive evaluation of trained models

**Features:**
- Multiple metrics: MSE, MAE, R² (overall and per-velocity)
- Prediction statistics (mean, std, min, max)
- JSON export for result logging
- Supports evaluation on multiple datasets

**Command-line Arguments:**
```
--model         Path to trained model (default: "models/bc_mlp.joblib")
--data          Path to data directory (default: "data")
--output        Path to save results JSON (optional)
--test-ratio    Test set ratio (default: 0.2)
--seed          Random seed (default: 42)
```

**Usage Examples:**
```bash
# Basic evaluation
python scripts/eval.py

# Evaluate specific model
python scripts/eval.py --model models/phase2.5_dagger.joblib --data data/combined

# Save results to JSON
python scripts/eval.py --output results/phase1_eval.json

# Evaluate on different test split
python scripts/eval.py --test-ratio 0.3 --seed 123
```

**Output Format (JSON):**
```json
{
  "overall": {
    "mse": 0.006,
    "mae": 0.05,
    "r2": 0.95
  },
  "linear_velocity": {"mse": 0.003, "r2": 0.96},
  "angular_velocity": {"mse": 0.009, "r2": 0.94},
  "prediction_stats": {
    "linear_vel": {"mean": 0.045, "std": 0.02, "min": 0.0, "max": 0.16},
    "angular_vel": {"mean": 0.01, "std": 0.05, "min": -0.5, "max": 0.5}
  },
  "n_samples": 223
}
```

---

## Complete Training Pipeline

### Phase 1: Baseline Behavioral Cloning
```bash
# 1. Prepare data (assumes rosbag already converted to CSV)
python scripts/data_prep.py

# 2. Train Phase 1 model
python scripts/train.py \
  --data data/expert_demos \
  --output models/phase1_bc.joblib \
  --hidden-layers 128 64 \
  --max-iter 500 \
  --plot

# 3. Evaluate Phase 1
python scripts/eval.py \
  --model models/phase1_bc.joblib \
  --data data/expert_demos \
  --output results/phase1_eval.json
```

### Phase 2.5: DAgger Refinement
```bash
# 1. Merge Phase 1 + DAgger demonstrations
# (Assumes DAgger data collected and placed in data/dagger_demos)
# Manual step: combine CSV files into data/combined/

# 2. Train Phase 2.5 model on combined data
python scripts/train.py \
  --data data/combined \
  --output models/phase2.5_dagger.joblib \
  --hidden-layers 128 64 \
  --max-iter 500 \
  --plot

# 3. Evaluate Phase 2.5
python scripts/eval.py \
  --model models/phase2.5_dagger.joblib \
  --data data/combined \
  --output results/phase2.5_eval.json
```

---

## Requirements
```
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
joblib>=1.0.0
```

Install: `pip install -r requirements.txt`

---

## Notes

**Data Format:**
- All CSV files must have timestamp column ('t' or 'timestamp')
- LiDAR scans: 360 points from 0-359 degrees
- Velocities: linear (vx) and angular (wz)

**Model Format:**
- Models saved using joblib (scikit-learn compatible)
- Can be loaded with: `model = joblib.load('path/to/model.joblib')`

**Training Tips:**
- Use early stopping to prevent overfitting
- Monitor both train and test MSE
- Typical training time: 2-3 minutes on CPU
- Phase 1: ~1,113 samples, Phase 2.5: ~2,100 samples

**Evaluation Best Practices:**
- Always use consistent test_ratio and seed for fair comparison
- Check R² > 0.9 for good tracking performance
- Linear velocity typically has lower MSE than angular velocity
- Test on held-out data not seen during training
