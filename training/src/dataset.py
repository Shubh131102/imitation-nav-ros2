"""
Dataset utilities for loading and preprocessing ROS bag data.

Converts ROS bag files containing LiDAR scans and velocity commands
into numpy arrays suitable for training navigation models.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List
import logging

from rosbags.highlevel import AnyReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def bag_to_arrays(
    bag_path: str,
    scan_topic: str = "/scan",
    cmd_topic: str = "/cmd_vel",
    max_range: float = 3.5,
    num_beams: int = 360
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract synchronized LiDAR scans and velocity commands from ROS bag.
    
    Args:
        bag_path: Path to ROS bag file
        scan_topic: Topic name for LiDAR scans (sensor_msgs/LaserScan)
        cmd_topic: Topic name for velocity commands (geometry_msgs/Twist)
        max_range: Maximum valid LiDAR range, used to replace inf/nan values
        num_beams: Number of LiDAR beams (360 for full scan)
        
    Returns:
        X: LiDAR scans (n_samples, num_beams)
        y: Velocity commands (n_samples, 2) - [linear_vel, angular_vel]
        
    Process:
        1. Read all LiDAR and velocity messages from bag
        2. Clean LiDAR data (replace inf/nan with max_range)
        3. Resample to uniform beam count if needed
        4. Temporally align commands to scans via nearest-neighbor
    """
    scans, cmds = [], []
    
    logger.info(f"Reading bag: {bag_path}")
    
    with AnyReader([Path(bag_path)]) as reader:
        # Get connections for scan and cmd_vel topics
        conns_scan = [c for c in reader.connections if c.topic == scan_topic]
        conns_cmd = [c for c in reader.connections if c.topic == cmd_topic]
        
        if not conns_scan:
            logger.warning(f"No messages found for topic: {scan_topic}")
        if not conns_cmd:
            logger.warning(f"No messages found for topic: {cmd_topic}")
        
        # Read and process LiDAR scans
        scan_msgs = []
        for conn, timestamp, raw_data in reader.messages(conns_scan):
            msg = reader.deserialize(raw_data, conn.msgtype)
            
            # Extract ranges and clean invalid values
            ranges = np.array(msg.ranges, dtype=np.float32)
            ranges[~np.isfinite(ranges)] = max_range
            ranges = np.clip(ranges, 0.0, max_range)
            
            # Resample to fixed number of beams if needed
            if len(ranges) != num_beams:
                indices = np.linspace(0, len(ranges) - 1, num_beams).astype(int)
                ranges = ranges[indices]
            
            scan_msgs.append((timestamp, ranges))
        
        # Read velocity commands
        cmd_msgs = []
        for conn, timestamp, raw_data in reader.messages(conns_cmd):
            msg = reader.deserialize(raw_data, conn.msgtype)
            
            # Extract linear and angular velocities
            linear_vel = getattr(msg.linear, "x", 0.0)
            angular_vel = getattr(msg.angular, "z", 0.0)
            
            cmd_msgs.append((timestamp, np.array([linear_vel, angular_vel], dtype=np.float32)))
        
        # Temporal alignment: match each scan to nearest command
        cmd_timestamps = np.array([t for t, _ in cmd_msgs])
        cmd_values = np.array([v for _, v in cmd_msgs])
        
        for scan_ts, scan_data in scan_msgs:
            # Find nearest command timestamp
            idx = np.abs(cmd_timestamps - scan_ts).argmin()
            scans.append(scan_data)
            cmds.append(cmd_values[idx])
    
    # Stack into arrays
    X = np.stack(scans) if scans else np.zeros((0, num_beams), dtype=np.float32)
    y = np.stack(cmds) if cmds else np.zeros((0, 2), dtype=np.float32)
    
    logger.info(f"Extracted {len(X)} synchronized samples")
    
    return X, y


def save_npz(output_path: str, *bag_paths: str) -> None:
    """
    Convert multiple ROS bags to compressed numpy archive.
    
    Args:
        output_path: Path to save .npz file
        *bag_paths: Variable number of bag file paths to process
        
    Saves:
        Compressed .npz file with keys:
        - 'X': LiDAR scans (total_samples, 360)
        - 'y': Velocity commands (total_samples, 2)
        
    Example:
        save_npz('data/phase1.npz', 'demo1.bag', 'demo2.bag', 'demo3.bag')
    """
    all_scans, all_commands = [], []
    
    logger.info(f"Processing {len(bag_paths)} bag files...")
    
    for bag_path in bag_paths:
        try:
            X, y = bag_to_arrays(bag_path)
            
            if len(X) > 0:
                all_scans.append(X)
                all_commands.append(y)
                logger.info(f"  {bag_path}: {len(X)} samples")
            else:
                logger.warning(f"  {bag_path}: No valid samples found")
                
        except Exception as e:
            logger.error(f"  {bag_path}: Failed - {e}")
            continue
    
    if not all_scans:
        raise ValueError("No valid data found in any bag files")
    
    # Concatenate all data
    X_combined = np.concatenate(all_scans, axis=0)
    y_combined = np.concatenate(all_commands, axis=0)
    
    # Save to compressed archive
    np.savez_compressed(output_path, X=X_combined, y=y_combined)
    
    logger.info(f"Saved: {output_path}")
    logger.info(f"  X shape: {X_combined.shape}")
    logger.info(f"  y shape: {y_combined.shape}")
    logger.info(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")


def load_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from compressed numpy archive.
    
    Args:
        npz_path: Path to .npz file
        
    Returns:
        X: LiDAR scans (n_samples, 360)
        y: Velocity commands (n_samples, 2)
    """
    data = np.load(npz_path)
    X = data['X']
    y = data['y']
    
    logger.info(f"Loaded: {npz_path}")
    logger.info(f"  X shape: {X.shape}")
    logger.info(f"  y shape: {y.shape}")
    
    return X, y


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python dataset.py output.npz bag1.bag [bag2.bag ...]")
        sys.exit(1)
    
    output_file = sys.argv[1]
    bag_files = sys.argv[2:]
    
    save_npz(output_file, *bag_files)
