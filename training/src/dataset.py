import numpy as np
from rosbags.highlevel import AnyReader

def bag_to_arrays(bag_path, scan_topic="/scan", cmd_topic="/cmd_vel",
                  max_range=3.5, num_beams=360):
    scans, cmds = [], []
    with AnyReader([bag_path]) as reader:
        conns_scan = [c for c in reader.connections if c.topic == scan_topic]
        conns_cmd  = [c for c in reader.connections if c.topic == cmd_topic]

        scan_msgs = []
        for conn, ts, raw in reader.messages(conns_scan):
            msg = reader.deserialize(raw, conn.msgtype)
            ranges = np.array(msg.ranges, dtype=np.float32)
            ranges[~np.isfinite(ranges)] = max_range
            ranges = np.clip(ranges, 0.0, max_range)
            if len(ranges) != num_beams:
                idx = np.linspace(0, len(ranges)-1, num_beams).astype(int)
                ranges = ranges[idx]
            scan_msgs.append((ts, ranges))

        cmd_msgs = []
        for conn, ts, raw in reader.messages(conns_cmd):
            msg = reader.deserialize(raw, conn.msgtype)
            v = getattr(msg.linear, "x", 0.0)
            w = getattr(msg.angular, "z", 0.0)
            cmd_msgs.append((ts, np.array([v, w], dtype=np.float32)))

        cmd_ts = np.array([t for t,_ in cmd_msgs])
        cmd_vals = np.array([v for _,v in cmd_msgs])
        for ts, scan in scan_msgs:
            idx = np.abs(cmd_ts - ts).argmin()
            scans.append(scan)
            cmds.append(cmd_vals[idx])

    X = np.stack(scans) if scans else np.zeros((0, num_beams), np.float32)
    y = np.stack(cmds)  if cmds  else np.zeros((0, 2), np.float32)
    return X, y

def save_npz(out_path, *bags):
    Xs, ys = [], []
    for b in bags:
        X, y = bag_to_arrays(b)
        if len(X) > 0:
            Xs.append(X); ys.append(y)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    np.savez_compressed(out_path, X=X, y=y)
    print("Saved:", out_path, X.shape, y.shape)
