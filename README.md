# Imitation Learning for Indoor Robot Navigation (ROS2 + scikit-learn)

**TL;DR:** Behavior cloning that maps raw 360° LiDAR to velocity commands for indoor navigation (TurtleBot3 in Gazebo).  
**Dataset:** ~1,478 synchronized samples (LiDAR 360-dim → [v_x, w_z])  
**Model:** MLPRegressor (hidden layers [128, 64], ReLU), MSE loss  
**Results:** R² ≈ 0.95 (velocity correlation), smooth tracking; validation MSE ≈ 0.0023

## Repo Layout
- `src/` — ROS/ROS2 code (nodes, launch)
- `scripts/` — data prep, training, evaluation
- `models/` — saved models (`.joblib`)
- `media/` — demo GIF/MP4
- `docs/` — report, diagrams
- `data/` — CSVs (scan, odom, cmd_vel). **Do not commit big/raw data.**

## Quick Start
1) Create a Python venv and install deps:
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
pip install -r requirements.txt
