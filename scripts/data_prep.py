import pandas as pd
import numpy as np
from pathlib import Path

def load_dataset(data_dir="data"):
    data_dir = Path(data_dir)
    scan_p = data_dir / "scan.csv"
    odom_p = data_dir / "odom.csv"   # optional, not used below
    cmd_p  = data_dir / "cmd_vel.csv"

    if not (scan_p.exists() and cmd_p.exists()):
        raise FileNotFoundError(
            f"Place scan.csv and cmd_vel.csv in {data_dir}. "
            "Columns must include a float timestamp named 't' or 'timestamp'."
        )

    scan = pd.read_csv(scan_p)
    cmd  = pd.read_csv(cmd_p)

    # normalize timestamp column
    for df in (scan, cmd):
        if "t" not in df.columns and "timestamp" in df.columns:
            df["t"] = df["timestamp"]
        elif "t" not in df.columns:
            raise KeyError("Each CSV must have time column 't' or 'timestamp'.")

    scan = scan.sort_values("t")
    cmd  = cmd.sort_values("t")

    # lidar columns = everything except time columns
    lidar_cols = [c for c in scan.columns if c not in ("t", "timestamp")]
    # clean LiDAR (replace inf/nan with max range)
    scan[lidar_cols] = scan[lidar_cols].replace([np.inf, -np.inf], np.nan)
    max_range = np.nanmax(scan[lidar_cols].values)
    scan[lidar_cols] = scan[lidar_cols].fillna(max_range)

    # nearest-time join cmd_vel to lidar
    df = pd.merge_asof(scan, cmd[["t", "vx", "wz"]], on="t", direction="nearest")

    X = df[lidar_cols].values.astype(np.float32)     # 360-d LiDAR
    y = df[["vx", "wz"]].values.astype(np.float32)   # linear & angular vel
    return X, y

if __name__ == "__main__":
    X, y = load_dataset()
    print("X shape:", X.shape, " y shape:", y.shape)
