# Data Directory

This directory contains training datasets, expert demonstrations, and evaluation logs.

## Structure
```
data/
├── expert_demos/          Phase 1 demonstrations (1,113 samples)
├── dagger_demos/          Phase 2.5 demonstrations (987 samples)
├── combined/              Merged dataset (2,100 samples)
└── evaluation/            Test results and metrics
```

## Data Format

### Expert Demonstrations (CSV)

Columns:
- timestamp: ROS time in seconds
- lidar_0 to lidar_359: 360-degree LiDAR range measurements (meters)
- linear_velocity: Forward speed command (m/s)
- angular_velocity: Rotational speed command (rad/s)

Example:
```
timestamp,lidar_0,lidar_1,...,lidar_359,linear_velocity,angular_velocity
1234567.89,3.5,3.5,...,2.1,0.15,0.0
```

### Evaluation Logs (JSON)
```json
{
  "phase": "2.5",
  "environment": "original_world",
  "mean_uncertainty": 0.135,
  "high_uncertainty_pct": 28.6,
  "final_speed": 0.045,
  "collisions": 0,
  "duration": 90.0
}
```

## Data Collection

Phase 1 data collected via manual teleoperation in turtlebot3_world environment (3:17 duration).

Phase 2.5 data collected in high-uncertainty regions (u > 0.20) identified during Phase 2 deployment.

## Notes

Raw rosbag files are not included in repository due to size. CSV files are preprocessed and synchronized.

Total dataset size: Approximately 50MB (combined CSV files).
