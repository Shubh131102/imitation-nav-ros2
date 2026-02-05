"""
Data preprocessing for uncertainty-aware navigation.

Loads LiDAR scans and velocity commands from CSV files, performs temporal
synchronization, and prepares training data for behavioral cloning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(
    data_dir: str = "data",
    max_range: float = 3.5,
    downsample: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess LiDAR scans with velocity commands.
    
    Args:
        data_dir: Directory containing scan.csv and cmd_vel.csv
        max_range: Maximum valid LiDAR range for replacing invalid readings
        downsample: Optional - downsample LiDAR from 360 to N points
        
    Returns:
        X: LiDAR scans (n_samples, n_points)
        y: Velocity commands (n_samples, 2) - [linear, angular]
        
    Raises:
        FileNotFoundError: If required CSV files are missing
        KeyError: If timestamp column is missing
    """
    data_dir = Path(data_dir)
    scan_path = data_dir / "scan.csv"
    cmd_path = data_dir / "cmd_vel.csv"
    
    # Validate files exist
    if not scan_path.exists():
        raise FileNotFoundError(f"Missing {scan_path}")
    if not cmd_path.exists():
        raise FileNotFoundError(f"Missing {cmd_path}")
    
    logger.info(f"Loading data from {data_dir}")
    
    # Load CSVs
    scan_df = pd.read_csv(scan_path)
    cmd_df = pd.read_csv(cmd_path)
    
    # Normalize timestamp column
    for df, name in [(scan_df, "scan"), (cmd_df, "cmd")]:
        if "t" not in df.columns:
            if "timestamp" in df.columns:
                df["t"] = df["timestamp"]
            else:
                raise KeyError(f"{name}.csv must have 't' or 'timestamp' column")
    
    # Sort by time
    scan_df = scan_df.sort_values("t").reset_index(drop=True)
    cmd_df = cmd_df.sort_values("t").reset_index(drop=True)
    
    # Identify LiDAR columns
    lidar_cols = [c for c in scan_df.columns if c not in ("t", "timestamp")]
    logger.info(f"Found {len(lidar_cols)} LiDAR points")
    
    # Clean LiDAR data
    scan_df[lidar_cols] = scan_df[lidar_cols].replace([np.inf, -np.inf], np.nan)
    scan_df[lidar_cols] = scan_df[lidar_cols].fillna(max_range)
    
    # Temporal synchronization via nearest-neighbor merge
    merged_df = pd.merge_asof(
        scan_df, 
        cmd_df[["t", "vx", "wz"]], 
        on="t", 
        direction="nearest",
        tolerance=0.1  # Max 100ms time difference
    )
    
    # Remove rows with missing velocity commands
    merged_df = merged_df.dropna(subset=["vx", "wz"])
    
    # Extract arrays
    X = merged_df[lidar_cols].values.astype(np.float32)
    y = merged_df[["vx", "wz"]].values.astype(np.float32)
    
    # Optional downsampling
    if downsample and downsample < X.shape[1]:
        indices = np.linspace(0, X.shape[1] - 1, downsample, dtype=int)
        X = X[:, indices]
        logger.info(f"Downsampled LiDAR to {downsample} points")
    
    logger.info(f"Loaded {X.shape[0]} synchronized samples")
    logger.info(f"LiDAR shape: {X.shape}, Velocity shape: {y.shape}")
    
    return X, y


def split_dataset(
    X: np.ndarray, 
    y: np.ndarray, 
    test_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train and test sets.
    
    Args:
        X: Input features
        y: Target labels
        test_ratio: Fraction of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_seed)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    
    split_idx = int(n_samples * (1 - test_ratio))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    logger.info(f"Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example usage
    X, y = load_dataset("data/expert_demos")
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_ratio=0.2)
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {X.shape[0]}")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"\nVelocity Statistics:")
    print(f"Linear vel - mean: {y[:, 0].mean():.3f}, std: {y[:, 0].std():.3f}")
    print(f"Angular vel - mean: {y[:, 1].mean():.3f}, std: {y[:, 1].std():.3f}")
