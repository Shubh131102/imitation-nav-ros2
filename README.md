# Indoor Navigation via Imitation Learning

Behavior cloning for autonomous indoor navigation using LiDAR-to-velocity mapping on TurtleBot3.

## Overview

Trained MLPRegressor to map 360-degree LiDAR scans directly to velocity commands (linear and angular).

- Dataset: 1,478 samples from expert demonstrations
- Model: MLP (128, 64 hidden units, ReLU activation)
- Performance: R² = 0.95, MSE = 0.0023

## Structure
```
src/          ROS2 nodes and launch files
scripts/      Training and evaluation scripts
models/       Trained models (.joblib)
data/         LiDAR and odometry CSVs (not committed)
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Usage

Train model:
```bash
python scripts/train.py --data data/training.csv
```

Run navigation:
```bash
ros2 launch navigation_pkg navigate.launch.py
```

## Technical Details

Input: 360-dimensional LiDAR scan
Output: [linear_velocity, angular_velocity]
Architecture: Input(360) → Dense(128) → Dense(64) → Output(2)
Loss: Mean Squared Error
