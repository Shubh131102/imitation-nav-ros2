# Scripts Directory

Training, evaluation, and data processing scripts for the uncertainty-aware navigation system.

## Contents

### Data Collection and Processing

**collect_rosbag.py**
- Records synchronized LiDAR scans and velocity commands
- Saves data in ROS bag format
- Usage: `python3 collect_rosbag.py --duration 200`

**rosbag_to_csv.py**
- Converts rosbag files to CSV format
- Performs temporal synchronization between topics
- Handles missing data and outliers
- Usage: `python3 rosbag_to_csv.py --input data.bag --output data.csv`

**preprocess_data.py**
- Normalizes LiDAR scans
- Splits into train/test sets (80/20)
- Handles invalid measurements (inf, nan)
- Usage: `python3 preprocess_data.py --input data.csv --output processed/`

### Training

**train_bc.py**
- Phase 1: Baseline behavioral cloning training
- Arguments: --data, --epochs, --batch-size, --lr
- Outputs: Trained model (.pth) and training curves
- Usage: `python3 train_bc.py --data data/expert_demos --epochs 100`

**train_dagger.py**
- Phase 2.5: DAgger iterative refinement
- Requires expert policy for corrections
- Arguments: --initial-model, --iterations, --beta-decay
- Usage: `python3 train_dagger.py --initial-model models/phase1_bc.pth --iterations 10`

### Evaluation

**evaluate.py**
- Comprehensive evaluation across environments
- Metrics: uncertainty, speed, collisions, success rate
- Generates plots and JSON results
- Usage: `python3 evaluate.py --model models/phase2.5_dagger.pth --env all`

**ablation_study.py**
- Tests different system configurations
- Configs: BC only, BC+Uncertainty, BC+Safety, Full system
- Usage: `python3 ablation_study.py --output results/ablation/`

**visualize_results.py**
- Generates plots from evaluation logs
- Creates comparison figures for report
- Usage: `python3 visualize_results.py --input results/ --output plots/`

### Utilities

**uncertainty_analysis.py**
- Analyzes uncertainty distributions
- Calibration plots and statistics
- Usage: `python3 uncertainty_analysis.py --data logs/phase2.5.json`

**merge_datasets.py**
- Combines Phase 1 and DAgger demonstrations
- Usage: `python3 merge_datasets.py --phase1 data/expert_demos --dagger data/dagger_demos --output data/combined`

## Requirements
```
torch>=1.9.0
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
rospy
rosbag
```

Install: `pip install -r requirements.txt`

## Training Pipeline

**Complete workflow from data to deployment:**
```bash
# 1. Collect data
python3 collect_rosbag.py --duration 200

# 2. Convert to CSV
python3 rosbag_to_csv.py --input demo.bag --output data/raw.csv

# 3. Preprocess
python3 preprocess_data.py --input data/raw.csv --output data/processed/

# 4. Train Phase 1
python3 train_bc.py --data data/processed/ --epochs 100

# 5. Collect DAgger data (manual step with ROS running)

# 6. Train Phase 2.5
python3 train_dagger.py --initial-model models/phase1_bc.pth --iterations 10

# 7. Evaluate
python3 evaluate.py --model models/phase2.5_dagger.pth --env all
```

## Notes

All scripts assume ROS workspace is sourced. Training scripts automatically save checkpoints every 10 epochs. Evaluation generates both numerical metrics (JSON) and visualization plots (PNG).
