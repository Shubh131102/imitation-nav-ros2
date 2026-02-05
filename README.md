# Uncertainty-Aware Imitation Learning for Indoor Navigation

Master's thesis implementing adaptive autonomous navigation through behavioral cloning with Monte Carlo Dropout uncertainty quantification for safe indoor robot operation.

## Overview

This system addresses the fundamental challenge in imitation learning: distribution shift between training and deployment. When robots encounter unfamiliar environments, standard policies produce confident but potentially dangerous predictions. Our solution integrates real-time uncertainty estimation with safety-aware control to enable adaptive navigation that recognizes and responds to its own limitations.

## Key Results

- 43% reduction in prediction uncertainty through targeted DAgger refinement
- 71% decrease in high-uncertainty events (99.8% to 28.6%)
- 5x improvement in navigation speed while maintaining zero collisions
- Well-calibrated uncertainty across diverse environments (0.073 to 0.135)
- Real-time operation at 5 Hz on standard CPU hardware

## System Architecture

The complete system integrates three core components:

**1. Monte Carlo Dropout Uncertainty Estimation**
- 20 stochastic forward passes per prediction
- Combined uncertainty from linear and angular velocity predictions
- Real-time computation at 5 Hz

**2. LiDAR-Based Safety Monitoring**
- 360-degree obstacle proximity analysis
- Risk score computation based on minimum clearance
- Independent environmental hazard assessment

**3. Adaptive Control via Decision Matrix**
- Speed modulation based on uncertainty and risk levels
- Nine control states from full speed to complete stop
- Conservative when uncertain, efficient when confident

## Three-Phase Development

### Phase 1: Baseline Behavior Cloning
- 1,113 expert demonstrations from manual teleoperation
- Conv1D CNN architecture (186,178 parameters)
- Test loss: 0.010 MSE
- Limitation: No uncertainty awareness

### Phase 2: Uncertainty Integration
- Added MC Dropout and safety monitoring
- Decision matrix for velocity modulation
- Detected high uncertainty 99.8% of time
- Result: Safe but impractically slow (0.009 m/s)

### Phase 2.5: DAgger Refinement
- 987 targeted demonstrations in high-uncertainty regions
- Combined dataset: 2,100 samples
- Test loss improved to 0.006 MSE
- Achieved balanced performance: 0.045 m/s with zero collisions

## Technical Specifications

**Platform**
- Robot: TurtleBot3 Burger (differential drive)
- Simulator: Gazebo 11
- Framework: ROS Noetic
- Deep Learning: PyTorch 1.9

**Network Architecture**
```
Input: 360-dim LiDAR scan
Conv1D(16 filters, kernel=5) + ReLU + MaxPool(2)
Conv1D(32 filters, kernel=5) + ReLU + MaxPool(2)
Flatten
Dense(64) + ReLU + Dropout(0.2)
Output: [linear_velocity, angular_velocity]
```

**Uncertainty Computation**
```
u = sqrt(σ²_vx + σ²_ωz)
where σ is std dev across 20 MC samples
```

## Multi-Environment Validation

Tested across three distinct environments without retraining:

| Environment | Mean Uncertainty | Speed (m/s) | Reduction |
|-------------|------------------|-------------|-----------|
| Empty World | 0.073            | 0.160       | 0%        |
| House World | 0.112            | 0.058       | 31%       |
| Original    | 0.135            | 0.045       | 43%       |

Uncertainty scales appropriately with environment complexity, demonstrating calibrated confidence estimation.

## Installation
```bash
# Clone repository
git clone https://github.com/yourusername/uncertainty-aware-navigation.git
cd uncertainty-aware-navigation

# Create ROS workspace
mkdir -p ~/nav_ws/src
cd ~/nav_ws/src
ln -s /path/to/uncertainty-aware-navigation .

# Install dependencies
cd ~/nav_ws
rosdep install --from-paths src --ignore-src -r -y
pip install -r requirements.txt

# Build
colcon build --packages-select uncertainty_nav
source install/setup.bash
```

## Usage

**Collect Expert Demonstrations**
```bash
ros2 launch uncertainty_nav collect_demos.launch.py
```

**Train Initial Model**
```bash
python3 scripts/train_bc.py --data data/expert_demos --epochs 100
```

**Run DAgger Refinement**
```bash
python3 scripts/train_dagger.py --iterations 10 --beta-decay 0.9
```

**Deploy Navigation System**
```bash
ros2 launch uncertainty_nav navigation.launch.py model:=models/dagger_final.pth
```

## Repository Structure
```
uncertainty_nav/
├── config/               Navigation and training parameters
├── data/                 Expert demonstrations and logs
├── launch/               ROS launch files
├── models/               Neural network definitions
├── scripts/              Training and evaluation scripts
├── src/                  ROS nodes (controller, perception, safety)
├── worlds/               Gazebo simulation environments
└── docs/                 Technical report and presentation
```

## Citation
```bibtex
@mastersthesis{jangle2025uncertainty,
  title={Uncertainty-Aware Imitation Learning for Safe Indoor Robot Navigation},
  author={Jangle, Shubham Yogesh},
  school={University of California, Riverside},
  year={2025},
  month={December}
}
```

## License

MIT License

## Contact

Shubham Jangle  
sjang041@ucr.edu  
MS Robotics Engineering, UC Riverside
