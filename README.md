# Uncertainty-Aware Imitation Learning for Indoor Navigation

Master's thesis implementing adaptive autonomous navigation through behavioral cloning with uncertainty quantification for safe indoor robot operation.

## Motivation

Traditional imitation learning suffers from distribution mismatch between training and deployment. This work addresses it by:
1. Using DAgger (Dataset Aggregation) for iterative policy improvement
2. Implementing Monte Carlo Dropout for real-time uncertainty estimation
3. Adaptive speed control based on model confidence

## Key Contributions

**71% reduction in high-uncertainty events** - Robot slows down when model is uncertain
**Zero collision rate** - Combined uncertainty-aware control with LiDAR safety monitoring
**5x faster convergence** - DAgger significantly outperforms standard behavioral cloning
**Real-time performance** - 5 Hz inference with MC Dropout (20 forward passes)

## Technical Architecture

**Perception Pipeline**
- 360-degree LiDAR processing (720 points downsampled to 180)
- Goal-relative positioning in robot frame
- Minimum distance extraction for safety layer

**Neural Network**
```
Input: LiDAR scan (180) + Goal vector (2)
Conv1D(1→32, k=5) + ReLU + Dropout(0.3)
Conv1D(32→64, k=5) + ReLU + Dropout(0.3)
Flatten + Dense(128) + Dropout(0.3)
Output: [v_linear, v_angular]
```

**Uncertainty Quantification**
- Monte Carlo Dropout with 20 stochastic forward passes
- Standard deviation across predictions as uncertainty metric
- Threshold-based speed modulation

**Control Strategy**
```python
if uncertainty > 0.3:
    velocity *= (1 - 0.5 * uncertainty)  # Adaptive slowdown
if min_lidar_distance < 0.4:
    velocity = 0  # Emergency brake
```

## Results

Performance across three environment complexities:

| Environment | Success Rate | Avg Uncertainty | Collisions |
|-------------|--------------|-----------------|------------|
| Simple      | 95%          | 0.12            | 0/50       |
| Medium      | 92%          | 0.18            | 0/50       |
| Complex     | 88%          | 0.24            | 0/50       |

**Comparison with baselines:**
- Behavioral Cloning alone: 78% success, 2.1% collision rate
- BC + DAgger: 85% success, 0.4% collision rate  
- BC + DAgger + Uncertainty (ours): 92% success, 0% collision rate

## Implementation

**Training Pipeline**
1. Expert demonstrations collected via Nav2 in Gazebo
2. Initial behavioral cloning on 1,478 state-action pairs
3. DAgger iterations with beta-decay schedule (10 iterations)
4. Model selection based on validation uncertainty distribution

**Deployment Stack**
- ROS2 Humble on Ubuntu 22.04
- PyTorch 2.0 for neural network
- Custom Nav2 integration for goal handling
- Real-time visualization in RViz

## Repository Structure
```
uncertainty_nav/
├── config/           Navigation and training parameters
├── launch/           ROS2 launch files for sim and deployment
├── models/           Neural network architecture and uncertainty
├── scripts/          Training (BC, DAgger) and evaluation
├── src/              ROS2 nodes (controller, perception, safety)
├── worlds/           Gazebo environments (simple/medium/complex)
└── data/             Expert demonstrations and evaluation logs
```

## Quick Start
```bash
# Build workspace
colcon build --packages-select uncertainty_nav
source install/setup.bash

# Collect expert data
ros2 launch uncertainty_nav collect_demos.launch.py

# Train with DAgger
python3 scripts/train_dagger.py --iterations 10 --beta-decay 0.9

# Deploy navigation
ros2 launch uncertainty_nav deploy.launch.py model:=models/dagger_final.pth
```

## Citation
```bibtex
@mastersthesis{jangle2025uncertainty,
  title={Uncertainty-Aware Imitation Learning for Indoor Autonomous Navigation},
  author={Jangle, Shubham},
  school={University of California, Riverside},
  year={2025}
}
```

## Future Directions

- Multi-robot coordination with distributed uncertainty sharing
- Semantic understanding for context-aware uncertainty
- Transfer to physical TurtleBot3 platform
- Alternative uncertainty methods: ensembles, Bayesian neural networks

---

**Contact:** sjang041@ucr.edu | [LinkedIn](link) | [Portfolio](link)
