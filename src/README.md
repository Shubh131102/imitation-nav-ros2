# Source Directory

ROS2 nodes and packages for uncertainty-aware navigation system.

## Structure
```
src/
├── uncertainty_nav/           Main navigation package
│   ├── uncertainty_nav/       Python package
│   │   ├── __init__.py
│   │   ├── uncertainty_controller.py
│   │   ├── safety_monitor.py
│   │   ├── mc_dropout.py
│   │   └── decision_matrix.py
│   ├── launch/                Launch files
│   ├── config/                Configuration files
│   ├── package.xml
│   └── setup.py
└── README.md
```

## Package Overview

### uncertainty_nav
Main ROS2 package implementing the complete uncertainty-aware navigation system.

**Key Components:**

**uncertainty_controller.py**
- Main navigation controller node
- Integrates MC Dropout policy with adaptive speed control
- Subscribes to /scan, publishes to /cmd_vel
- Real-time operation at 5 Hz

**safety_monitor.py**
- LiDAR-based safety monitoring
- Computes risk scores from obstacle proximity
- Independent safety layer

**mc_dropout.py**
- Monte Carlo Dropout inference
- 20 stochastic forward passes per prediction
- Uncertainty quantification (epistemic uncertainty)

**decision_matrix.py**
- Speed modulation logic
- Maps (uncertainty, risk) → modulation factor
- 9-state decision matrix

## Building the Package
```bash
# Navigate to workspace
cd ~/nav_ws

# Build
colcon build --packages-select uncertainty_nav

# Source
source install/setup.bash
```

## Running the System

**Simulation with Navigation:**
```bash
# Terminal 1: Launch Gazebo
ros2 launch uncertainty_nav simulation.launch.py

# Terminal 2: Run uncertainty-aware controller
ros2 run uncertainty_nav uncertainty_controller \
  --model models/phase2.5_dagger.joblib
```

**Data Collection:**
```bash
ros2 launch uncertainty_nav collect_demos.launch.py
```

## Node Descriptions

### uncertainty_controller
**Subscribed Topics:**
- `/scan` (sensor_msgs/LaserScan): 360-degree LiDAR data
- `/odom` (nav_msgs/Odometry): Robot odometry (optional)

**Published Topics:**
- `/cmd_vel` (geometry_msgs/Twist): Velocity commands
- `/uncertainty` (std_msgs/Float32): Current uncertainty estimate
- `/risk_score` (std_msgs/Float32): Environmental risk score
- `/modulation_factor` (std_msgs/Float32): Speed modulation applied

**Parameters:**
- `model_path`: Path to trained model (.joblib)
- `mc_samples`: Number of MC Dropout samples (default: 20)
- `uncertainty_threshold_high`: High uncertainty threshold (default: 0.15)
- `uncertainty_threshold_med`: Medium uncertainty threshold (default: 0.10)
- `control_frequency`: Control loop rate in Hz (default: 5)

### safety_monitor
**Subscribed Topics:**
- `/scan` (sensor_msgs/LaserScan): LiDAR data

**Published Topics:**
- `/risk_score` (std_msgs/Float32): Computed risk score [0-1]
- `/min_distance` (std_msgs/Float32): Minimum obstacle distance

**Parameters:**
- `critical_distance`: Emergency stop distance (default: 0.05m)
- `warning_distance`: High risk distance (default: 0.10m)
- `safe_distance`: Medium risk distance (default: 0.20m)

## Launch Files

### simulation.launch.py
Launches complete simulation environment with Gazebo and RViz.

**Arguments:**
- `world`: Gazebo world file (default: turtlebot3_world)
- `rviz`: Enable RViz visualization (default: true)
- `robot_x`, `robot_y`: Initial robot position

### navigation.launch.py
Launches navigation system with uncertainty-aware controller.

**Arguments:**
- `model_path`: Path to trained model
- `mc_samples`: Number of MC Dropout samples
- `use_safety_monitor`: Enable independent safety monitoring

### collect_demos.launch.py
Sets up environment for demonstration collection.

## Configuration Files

### navigation_params.yaml
```yaml
uncertainty_controller:
  ros__parameters:
    model_path: "models/phase2.5_dagger.joblib"
    mc_samples: 20
    uncertainty_threshold_high: 0.15
    uncertainty_threshold_med: 0.10
    control_frequency: 5.0

safety_monitor:
  ros__parameters:
    critical_distance: 0.05
    warning_distance: 0.10
    safe_distance: 0.20
```

### decision_matrix.yaml
```yaml
modulation_factors:
  low_uncertainty:
    low_risk: 1.00
    medium_risk: 0.80
    high_risk: 0.50
  medium_uncertainty:
    low_risk: 0.60
    medium_risk: 0.40
    high_risk: 0.20
  high_uncertainty:
    low_risk: 0.30
    medium_risk: 0.20
    high_risk: 0.00
```

## Development

**Creating New Nodes:**
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class MyNavigationNode(Node):
    def __init__(self):
        super().__init__('my_navigation_node')
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
    
    def scan_callback(self, msg):
        # Process LiDAR data
        pass

def main(args=None):
    rclpy.init(args=args)
    node = MyNavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

## Notes

- All nodes operate at 5 Hz for real-time performance
- Models loaded at startup, no runtime training
- System designed for TurtleBot3 Burger platform
- Compatible with ROS2 Humble on Ubuntu 22.04
- CPU-only inference (no GPU required)
