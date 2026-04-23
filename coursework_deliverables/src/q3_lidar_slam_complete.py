#!/usr/bin/env python3
"""
COMP0222 Coursework 2 - Question 3: LiDAR SLAM
Covers Q3b (parameter analysis), Q3c (loop closure), Q3d (factor graph).

Data loaded from recorded RPLidar JSONL files.
Sequences used:
  - Basement_1       (indoor)
  - Floor7_Hallway   (indoor large area - Marshgate building)
  - Outdoor_1        (outdoor)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os
import time
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize

try:
    import gtsam
    from gtsam import Pose2, BetweenFactorPose2, PriorFactorPose2, noiseModel
    _HAVE_GTSAM = True
except Exception as _e:
    _HAVE_GTSAM = False
    import sys
    print(
        "\n" + "=" * 72 + "\n"
        "  WARNING: GTSAM not found — Q3d factor-graph will use the SciPy\n"
        "  SLSQP fallback instead of the Levenberg-Marquardt GTSAM solver.\n"
        "  Results (closure error reduction, occupancy grid) will differ\n"
        "  from the submitted figures which were generated with GTSAM.\n"
        "  Install: pip install gtsam>=4.3\n"
        "  Error detail: " + str(_e) + "\n"
        "=" * 72 + "\n",
        file=sys.stderr
    )

# ============================================================
# PATHS
# ============================================================
_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.abspath(os.path.join(_HERE, '..'))
REC1     = os.environ.get('SLAM_REC1', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings', 'tmp_recordings'))
REC2     = os.environ.get('SLAM_REC2', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings2'))
OUT_DIR  = os.environ.get('SLAM_OUT',  os.path.join(_ROOT, 'data', 'q3_results'))
os.makedirs(OUT_DIR, exist_ok=True)

SEQUENCES = {
    # --- tmp_recordings ---
    'Basement_1':      os.path.join(REC1, 'Basement_1',     'lidar', 'scans.jsonl'),
    'Basement_2':      os.path.join(REC1, 'Basement_2',     'lidar', 'scans.jsonl'),
    'Floor7_Hallway':  os.path.join(REC1, 'Floor7_Hallway', 'lidar', 'scans.jsonl'),
    'Outdoor_1':       os.path.join(REC1, 'Outdoor_1',      'lidar', 'scans.jsonl'),
    'Washroom':        os.path.join(REC1, 'Washroom',       'lidar', 'scans.jsonl'),
    # --- tmp_recordings2 ---
    'BikeStorage':     os.path.join(REC2, 'BikeStorage',    'lidar', 'scans.jsonl'),
    'BikeStorage2':    os.path.join(REC2, 'BikeStorage2',   'lidar', 'scans.jsonl'),
    'Entrance2':       os.path.join(REC2, 'Entrance2',      'lidar', 'scans.jsonl'),
    'OnePoolStreet1':  os.path.join(REC2, 'OnePoolStreet1', 'lidar', 'scans.jsonl'),
}

# RPLidar A1 maximum rated range (mm). Actual scan data reaches ~14 000 mm in
# open spaces; the A1 nominal spec is 12 000 mm.
SENSOR_MAX_RANGE_MM = 12000.0

# Blind spot: operator stands 135–225 degrees
BLIND_SPOT_MIN = 135.0
BLIND_SPOT_MAX = 225.0

# ============================================================
# DATA LOADING
# ============================================================
def load_scans(path, max_scans=None):
    scans = []
    with open(path) as f:
        for line in f:
            scans.append(json.loads(line.strip()))
            if max_scans and len(scans) >= max_scans:
                break
    return scans


# ============================================================
# SCAN PROCESSING
# ============================================================
def process_scan(points_raw, max_range_mm=4000.0, angular_step=1):
    """
    Convert raw [(quality, angle_deg, dist_mm),...] to Nx2 XY array (metres).

    Parameters
    ----------
    max_range_mm   : maximum range filter
    angular_step   : 1 = full scan, 2 = every 2nd beam, 3 = every 3rd beam
    """
    if not points_raw:
        return None
    raw = np.array(points_raw)          # (N, 3)
    qualities  = raw[:, 0]
    angles_deg = raw[:, 1]
    dists_mm   = raw[:, 2]

    # Angular downsampling: keep every angular_step-th point
    if angular_step > 1:
        idx = np.arange(len(raw))
        raw       = raw[idx % angular_step == 0]
        angles_deg = raw[:, 1]
        dists_mm   = raw[:, 2]

    # Distance filter
    dist_mask  = (dists_mm > 10) & (dists_mm < max_range_mm)
    # Blind spot filter
    angle_mask = (angles_deg < BLIND_SPOT_MIN) | (angles_deg > BLIND_SPOT_MAX)
    mask = dist_mask & angle_mask

    if np.sum(mask) < 6:
        return None

    a = np.radians(angles_deg[mask])
    d = dists_mm[mask] / 1000.0         # metres
    return np.column_stack((d * np.cos(a), d * np.sin(a)))


def voxel_downsample(pts, voxel_m=0.05):
    """Keep one point per voxel cell."""
    if voxel_m <= 0 or pts is None or len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel_m).astype(int)
    _, first = np.unique(keys, axis=0, return_index=True)
    return pts[first]


# ============================================================
# ICP HELPERS  (from Lab 8)
# ============================================================
def estimate_normals_pca(pts, k=5):
    if len(pts) < k + 1:
        return np.zeros((len(pts), 2))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(pts)
    _, idx = nn.kneighbors(pts)
    normals = np.zeros((len(pts), 2))
    for i in range(len(pts)):
        nbrs = pts[idx[i]]
        c    = nbrs - nbrs.mean(0)
        cov  = c.T @ c / k
        _, vecs = np.linalg.eigh(cov)
        n = vecs[:, 0]
        if np.dot(n, pts[i]) < 0:
            n = -n
        normals[i] = n
    return normals


def solve_point_to_plane(src, dst, normals):
    """Linearised point-to-plane ICP step → 3x3 homogeneous transform."""
    A, b = [], []
    for s, d, n in zip(src, dst, normals):
        cross = s[0] * n[1] - s[1] * n[0]   # 2-D cross product (rotation moment arm)
        A.append([cross, n[0], n[1]])
        b.append(float(np.dot(d - s, n)))
    if not A:
        return np.eye(3)
    A, b = np.array(A), np.array(b)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    c, s_ = np.cos(x[0]), np.sin(x[0])
    T = np.eye(3)
    T[:2, :2] = [[c, -s_], [s_, c]]
    T[:2, 2]  = [x[1], x[2]]
    return T


def icp_scan_to_map(src, map_pts, map_normals, init_pose,
                    max_iter=10, corr_thresh=0.5):
    """Register src into the map frame; returns updated 3x3 pose."""
    src_h = np.vstack([src.T, np.ones(len(src))])       # (3, N)
    pose  = init_pose.copy()
    nn    = NearestNeighbors(n_neighbors=1).fit(map_pts)

    for _ in range(max_iter):
        global_h = pose @ src_h
        global_  = global_h[:2].T
        dists, idx = nn.kneighbors(global_, return_distance=True)
        dists, idx = dists.ravel(), idx.ravel()
        valid = dists < corr_thresh
        if valid.sum() < 8:
            break
        dT = solve_point_to_plane(global_[valid],
                                   map_pts[idx[valid]],
                                   map_normals[idx[valid]])
        pose = dT @ pose
        if np.linalg.norm(dT[:2, 2]) < 1e-3 and abs(np.arctan2(dT[1,0], dT[0,0])) < 1e-3:
            break
    return pose


# ============================================================
# OCCUPANCY GRID  (ray-casting, from Lab 9 concept)
# ============================================================
def build_occupancy_grid(trajectory_xyt, scans_xy_global,
                          cell_m=0.05, grid_m=20.0):
    """
    Build a 2-D log-odds occupancy grid.
    trajectory_xyt: list of [x, y, theta]
    scans_xy_global: list of Nx2 arrays (global frame)
    Returns (grid, origin) where grid is HxW uint8 (0=free,128=unknown,255=occupied).
    """
    half   = grid_m / 2.0
    N_cell = int(grid_m / cell_m)
    log_grid = np.zeros((N_cell, N_cell), dtype=np.float32)
    origin = np.array([-half, -half])

    def w2g(xy):
        g = ((xy - origin) / cell_m).astype(int)
        return np.clip(g, 0, N_cell - 1)

    for pose_xy, scan_global in zip(trajectory_xyt, scans_xy_global):
        if scan_global is None or len(scan_global) == 0:
            continue
        rx, ry = int((pose_xy[0] - origin[0]) / cell_m), int((pose_xy[1] - origin[1]) / cell_m)
        rx = np.clip(rx, 0, N_cell - 1)
        ry = np.clip(ry, 0, N_cell - 1)
        for hit in scan_global:
            hx, hy = w2g(hit)
            # Bresenham ray for free cells
            x0, y0 = rx, ry
            x1, y1 = hx, hy
            dx, dy = abs(x1 - x0), abs(y1 - y0)
            sx     = 1 if x0 < x1 else -1
            sy     = 1 if y0 < y1 else -1
            err    = dx - dy
            cx, cy = x0, y0
            steps  = 0
            while (cx != x1 or cy != y1) and steps < 200:
                if 0 <= cx < N_cell and 0 <= cy < N_cell:
                    log_grid[cy, cx] -= 0.4      # free
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    cx  += sx
                if e2 < dx:
                    err += dx
                    cy  += sy
                steps += 1
            if 0 <= hx < N_cell and 0 <= hy < N_cell:
                log_grid[hy, hx] += 0.85         # occupied

    # Convert log-odds to image (clip prevents overflow in np.exp)
    log_grid = np.clip(log_grid, -20.0, 20.0)
    prob = 1.0 / (1.0 + np.exp(-log_grid))
    img  = np.full((N_cell, N_cell), 128, dtype=np.uint8)
    img[prob > 0.6]  = 0          # occupied → black
    img[prob < 0.4]  = 255        # free → white
    return img, origin


# ============================================================
# CORE SLAM LOOP
# ============================================================
def run_slam(scans_raw, max_range_mm=4000.0, angular_step=1,
             voxel_m=0.0, scan_skip=1,
             corr_thresh=0.5, kf_dist=0.2, kf_angle=0.2,
             icp_iter=10, local_map=20):
    """
    Offline ICP-based laser odometry.

    scan_skip: 1 = use every scan, 2 = skip odd (use even), 3 = use every 3rd
    Returns dict with trajectory, global_scans_xy, keyframe_poses, timestamps.
    """
    pose       = np.eye(3)
    kf_pose    = np.eye(3)
    kf_buf     = []           # list of (pts_global, normals_global)
    map_pts_all = []          # for occupancy grid
    trajectory = [[0.0, 0.0, 0.0]]   # [x, y, theta]
    kf_poses   = [np.eye(3)]
    times      = []

    first = True
    n_total = len(scans_raw)

    for i, s in enumerate(scans_raw):
        # Scan-rate reduction
        if scan_skip > 1 and (i % scan_skip) != 0:
            continue

        t0  = time.perf_counter()
        pts = process_scan(s['points'], max_range_mm=max_range_mm,
                           angular_step=angular_step)
        if pts is None:
            continue
        if voxel_m > 0:
            pts = voxel_downsample(pts, voxel_m)
        if pts is None or len(pts) < 6:
            continue

        if first:
            nrm = estimate_normals_pca(pts)
            kf_buf.append((pts, nrm))
            map_pts_all.append(pts)
            first = False
            continue

        # ICP against local map
        all_pts = np.vstack([k[0] for k in kf_buf])
        all_nrm = np.vstack([k[1] for k in kf_buf])
        prev_pose = pose.copy()
        pose = icp_scan_to_map(pts, all_pts, all_nrm, pose,
                                max_iter=icp_iter, corr_thresh=corr_thresh)

        # Reject diverged ICP steps: RPLidar max range is 8 m so a single
        # scan-to-scan motion > 2 m or > 45° is physically impossible.
        step_dist = np.linalg.norm(pose[:2, 2] - prev_pose[:2, 2])
        step_ang  = abs(np.arctan2(pose[1, 0], pose[0, 0]) -
                        np.arctan2(prev_pose[1, 0], prev_pose[0, 0]))
        step_ang  = min(step_ang, 2 * np.pi - step_ang)
        if step_dist > 2.0 or step_ang > np.radians(45):
            pose = prev_pose

        x, y  = pose[0, 2], pose[1, 2]
        theta = np.arctan2(pose[1, 0], pose[0, 0])
        trajectory.append([x, y, theta])

        # Keyframe decision
        dT   = np.linalg.inv(kf_pose) @ pose
        dist = np.linalg.norm(dT[:2, 2])
        dang = abs(np.arctan2(dT[1, 0], dT[0, 0]))
        if dist > kf_dist or dang > kf_angle:
            # Transform pts to global frame
            h = np.vstack([pts.T, np.ones(len(pts))])
            gpts = (pose @ h)[:2].T
            gnrm = estimate_normals_pca(gpts)
            kf_buf.append((gpts, gnrm))
            map_pts_all.append(gpts)
            kf_poses.append(pose.copy())
            kf_pose = pose.copy()
            if len(kf_buf) > local_map:
                kf_buf.pop(0)

        times.append(time.perf_counter() - t0)
        if i % 200 == 0:
            print(f"  scan {i}/{n_total}, pose ({x:.2f},{y:.2f})")

    return {
        'trajectory':  np.array(trajectory),
        'kf_poses':    kf_poses,
        'map_pts':     map_pts_all,
        'proc_times':  np.array(times) if times else np.array([0.0]),
    }


# ============================================================
# Q3a: TWO-LOOP VERIFICATION
# ============================================================
def _detect_laps(traj):
    """
    Detect lap boundaries by tracking cumulative unwrapped heading.
    Each time the cumulative angle crosses a multiple of 2π, a new lap starts.
    Returns list of frame indices where each lap begins (including 0).
    """
    if len(traj) < 4:
        return [0]
    thetas   = traj[:, 2]
    unwrapped = np.unwrap(thetas)
    total    = unwrapped[-1] - unwrapped[0]
    if abs(total) < 0.5:
        return [0]

    lap_boundaries = [0]
    sign = np.sign(total)
    step = sign * 2 * np.pi
    threshold = step
    for i in range(1, len(unwrapped)):
        delta = unwrapped[i] - unwrapped[0]
        if sign > 0 and delta >= threshold:
            lap_boundaries.append(i)
            threshold += step
        elif sign < 0 and delta <= threshold:
            lap_boundaries.append(i)
            threshold += step
    return lap_boundaries


def plot_two_loop_verification(seq_name, slam_result, out_dir):
    """
    Q3a verification: annotate trajectory with detected lap segments and
    mark the start/end proximity (closure error).  Saves q3a_two_loop_<seq>.png.

    Brief requirement: robot must complete EXACTLY two loops and return to start.
    """
    traj = slam_result['trajectory']
    if len(traj) < 4:
        print(f"  [Q3a] {seq_name}: too few poses, skipping")
        return

    laps  = _detect_laps(traj)
    n_laps = len(laps) - 1 if len(laps) > 1 else 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f'Q3a Two-Loop Verification — {seq_name}\n'
                 f'Detected {n_laps} lap(s), {len(traj)} poses',
                 fontsize=12, fontweight='bold')

    # Left: full trajectory coloured by lap segment
    ax = axes[0]
    colors_lap = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    segments = list(zip(laps, laps[1:] + [len(traj)]))
    for seg_idx, (start, end) in enumerate(segments):
        seg = traj[start:end]
        c   = colors_lap[seg_idx % len(colors_lap)]
        lbl = f'Lap {seg_idx + 1}'
        ax.plot(seg[:, 0], seg[:, 1], color=c, lw=1.5, alpha=0.85, label=lbl)
        ax.plot(*seg[0, :2], 'o', color=c, ms=7)
    ax.plot(*traj[0, :2],  'g^', ms=12, zorder=5, label='Start')
    ax.plot(*traj[-1, :2], 'rs', ms=12, zorder=5, label='End')
    ce = np.hypot(traj[-1, 0] - traj[0, 0], traj[-1, 1] - traj[0, 1])
    ax.set_title(f'Trajectory (closure error = {ce:.2f} m)', fontsize=10)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Right: cumulative heading to show loop completion
    ax2 = axes[1]
    unwrapped = np.unwrap(traj[:, 2])
    ax2.plot(np.degrees(unwrapped), lw=1.5, color='steelblue')
    for b in laps[1:]:
        ax2.axvline(b, color='r', linestyle='--', alpha=0.6)
    ax2.axhline(0,    color='k', linestyle=':', alpha=0.3)
    ax2.axhline(360,  color='gray', linestyle=':', alpha=0.5, label='1 lap')
    ax2.axhline(720,  color='gray', linestyle='--', alpha=0.5, label='2 laps')
    ax2.axhline(-360, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(-720, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Frame'); ax2.set_ylabel('Cumulative heading (°)')
    ax2.set_title('Heading unwrapped — red lines = lap boundaries', fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, f'q3a_two_loop_{seq_name.lower()}.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Q3b: PARAMETER EXPERIMENTS
# ============================================================
def run_q3b(seq_name, scans_raw, out_dir):
    """
    Run all four parameter variations required by Q3b.
    Each subsection gets its own subplot grid.
    """
    print(f"\n=== Q3b Parameter Analysis: {seq_name} ===")

    # ---- 1. Maximum Range ----
    print("  1. Maximum Range")
    res_range = {}
    for rng, label in [(2000.0, '2000mm'), (SENSOR_MAX_RANGE_MM, f'{int(SENSOR_MAX_RANGE_MM)}mm (sensor max)')]:
        r = run_slam(scans_raw, max_range_mm=rng)
        res_range[label] = r
        t = r['trajectory']
        cl = closure_error(t)
        print(f"    range={label}: {len(t)} pts, closure={cl:.3f}m, "
              f"avg_time={r['proc_times'].mean():.4f}s")

    # ---- 2. Angular Resolution ----
    print("  2. Angular Resolution")
    res_angular = {}
    for step, label in [(1, 'Full scan'), (2, 'Every 2nd beam'), (3, 'Every 3rd beam')]:
        r = run_slam(scans_raw, angular_step=step)
        res_angular[label] = r
        t = r['trajectory']
        print(f"    step={step}: {len(t)} pts, closure={closure_error(t):.3f}m")

    # ---- 3. Voxel Grid Downsampling ----
    print("  3. Voxel Grid Downsampling")
    res_voxel = {}
    for v, label in [(0.0, 'None'), (0.05, '0.05m'), (0.10, '0.10m'), (0.20, '0.20m')]:
        r = run_slam(scans_raw, voxel_m=v)
        res_voxel[label] = r
        t = r['trajectory']
        print(f"    voxel={label}: {len(t)} pts, closure={closure_error(t):.3f}m, "
              f"avg_time={r['proc_times'].mean():.4f}s")

    # ---- 4. Scan Rate ----
    print("  4. Scan Rate")
    res_rate = {}
    for skip, label in [(1, 'All scans'), (2, 'Skip odd (50%)'), (3, 'Skip 2/3 (33%)')]:
        r = run_slam(scans_raw, scan_skip=skip)
        res_rate[label] = r
        t = r['trajectory']
        print(f"    skip={skip}: {len(t)} pts, closure={closure_error(t):.3f}m")

    # ---- Plots ----
    _plot_q3b(seq_name, res_range, res_angular, res_voxel, res_rate, out_dir)

    # ---- Per-parameter occupancy grids (required by brief) ----
    _plot_q3b_occupancy_grids(seq_name,
                              [('Max Range',         res_range),
                               ('Angular Resolution', res_angular),
                               ('Voxel Downsampling', res_voxel),
                               ('Scan Rate',          res_rate)],
                              out_dir)

    return res_range, res_angular, res_voxel, res_rate


def _plot_q3b_occupancy_grids(seq_name, groups, out_dir, cell_m=0.05, grid_m=25.0):
    """
    For each parameter group (Max Range / Angular Resolution / Voxel / Scan Rate),
    build an occupancy grid for every variation and save a side-by-side figure.
    This addresses the coursework brief's requirement to show occupancy-grid
    outputs for each parameter setting in Q3b.
    """
    for group_title, res_dict in groups:
        n = len(res_dict)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.2))
        if n == 1:
            axes = [axes]
        fig.suptitle(f'Q3b Occupancy Grids — {seq_name} — {group_title}',
                     fontsize=13, fontweight='bold')
        for ax, (label, r) in zip(axes, res_dict.items()):
            traj  = r['trajectory']
            kf    = r['map_pts']
            # Subsample for speed if very long
            step  = max(1, len(kf) // 400)
            t_sub = traj[::step][:len(kf[::step])]
            k_sub = kf[::step]
            grid, origin = build_occupancy_grid(t_sub, k_sub,
                                                 cell_m=cell_m, grid_m=grid_m)
            ax.imshow(grid, cmap='gray', origin='lower',
                      extent=[origin[0], origin[0] + grid_m,
                              origin[1], origin[1] + grid_m])
            ax.plot(traj[:, 0], traj[:, 1], 'r-', lw=1.0, alpha=0.7)
            ax.plot(*traj[0, :2],  'go', ms=6)
            ax.plot(*traj[-1, :2], 'bs', ms=6)
            ce = closure_error(traj)
            ax.set_title(f'{label}\nclosure={ce:.2f} m', fontsize=9)
            ax.set_xlabel('X (m)', fontsize=8); ax.set_ylabel('Y (m)', fontsize=8)
            ax.set_aspect('equal')
        plt.tight_layout()
        safe = group_title.lower().replace(' ', '_')
        out = os.path.join(out_dir, f'q3b_grids_{seq_name.lower()}_{safe}.png')
        plt.savefig(out, dpi=140, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out}")


def _plot_q3b(seq_name, res_range, res_angular, res_voxel, res_rate, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(f'Q3b Parameter Analysis — {seq_name}', fontsize=14, fontweight='bold')

    pairs = [
        (res_range,   'Max Range',            axes[0, 0], axes[1, 0]),
        (res_angular, 'Angular Resolution',   axes[0, 1], axes[1, 1]),
        (res_voxel,   'Voxel Downsampling',   axes[0, 2], axes[1, 2]),
        (res_rate,    'Scan Rate',            axes[0, 3], axes[1, 3]),
    ]

    colors = plt.cm.tab10.colors
    for res_dict, title, ax_traj, ax_bar in pairs:
        for ci, (label, r) in enumerate(res_dict.items()):
            t = r['trajectory']
            ax_traj.plot(t[:, 0], t[:, 1], color=colors[ci],
                         label=label, lw=1.5, alpha=0.8)
        ax_traj.set_title(title, fontsize=10)
        ax_traj.set_xlabel('X (m)', fontsize=8)
        ax_traj.set_ylabel('Y (m)', fontsize=8)
        ax_traj.legend(fontsize=7)
        ax_traj.set_aspect('equal', adjustable='datalim')
        ax_traj.grid(True, alpha=0.3)

        # Bar: closure error
        labels = list(res_dict.keys())
        errs   = [closure_error(res_dict[lb]['trajectory']) for lb in labels]
        bars   = ax_bar.bar(range(len(labels)), errs,
                            color=colors[:len(labels)], alpha=0.8)
        ax_bar.set_xticks(range(len(labels)))
        ax_bar.set_xticklabels(labels, rotation=20, ha='right', fontsize=7)
        ax_bar.set_ylabel('Closure error (m)', fontsize=8)
        ax_bar.set_title(f'{title} — closure error', fontsize=9)
        ax_bar.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, errs):
            ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    out = os.path.join(out_dir, f'q3b_{seq_name.lower()}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# Q3c: LOOP CLOSURE DETECTION
# ============================================================
def detect_loop_closures(kf_poses, kf_scans_global,
                          min_separation=10, pose_dist_thresh=2.0,
                          icp_corr_thresh=0.4, icp_score_thresh=0.55):
    """
    Compare each keyframe with all earlier ones (at least min_separation away).
    Use pose distance as a candidate filter, then ICP match score to confirm.

    Returns list of (i, j, relative_transform) tuples.
    """
    closures = []
    n = len(kf_poses)

    for j in range(min_separation, n):
        pj = kf_poses[j]
        xj, yj = pj[0, 2], pj[1, 2]

        # Only compare against sufficiently older keyframes. The previous
        # range expression could include invalid future indices for small j.
        for i in range(0, j - min_separation + 1):
            pi = kf_poses[i]
            xi, yi = pi[0, 2], pi[1, 2]
            dist_ij = np.hypot(xj - xi, yj - yi)

            if dist_ij > pose_dist_thresh:
                continue

            # ICP refinement between keyframe scans
            ref = kf_scans_global[i]
            if kf_scans_global[j] is None or ref is None or len(kf_scans_global[j]) < 8 or len(ref) < 8:
                continue

            # icp_scan_to_map expects src in the sensor/local frame; kf_scans_global
            # stores points in the world frame, so transform j's scan back to j's body frame.
            src_world_h = np.vstack([kf_scans_global[j].T, np.ones(len(kf_scans_global[j]))])
            src_local   = (np.linalg.inv(pj) @ src_world_h)[:2].T

            ref_nrm = estimate_normals_pca(ref)
            nn = NearestNeighbors(n_neighbors=1).fit(ref)
            init       = np.linalg.inv(pi) @ pj
            pj_refined = icp_scan_to_map(src_local, ref, ref_nrm, init,
                                         max_iter=15, corr_thresh=icp_corr_thresh)

            # T_rel = Pose_i^{-1} · Pose_j_refined — exactly what BetweenFactorPose2 expects.
            T_rel = np.linalg.inv(pi) @ pj_refined

            # Score = fraction of j's points (world frame) within threshold of ref
            src_local_h = np.vstack([src_local.T, np.ones(len(src_local))])
            aligned = (pj_refined @ src_local_h)[:2].T
            dists, _ = nn.kneighbors(aligned, return_distance=True)
            score = float((dists.ravel() < icp_corr_thresh).mean())

            if score >= icp_score_thresh:
                closures.append((i, j, T_rel, score, dist_ij))

    return closures


def run_q3c(seq_name, slam_result, out_dir):
    """Detect loop closures and produce analysis plot."""
    print(f"\n=== Q3c Loop Closure: {seq_name} ===")

    kf_poses  = slam_result['kf_poses']
    map_pts   = slam_result['map_pts']
    n_kf      = len(kf_poses)
    print(f"  Keyframes: {n_kf}")

    closures = detect_loop_closures(kf_poses, map_pts)

    print(f"  Loop closures detected: {len(closures)}")
    for lc in closures[:5]:
        print(f"    KF {lc[0]} ↔ {lc[1]}  score={lc[3]:.3f}  pose_dist={lc[4]:.3f}m")

    # Plot: trajectory with loop closure arcs
    traj = slam_result['trajectory']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Q3c Loop Closure Detection — {seq_name}', fontweight='bold')

    ax1.plot(traj[:, 0], traj[:, 1], 'b-', lw=1.5, label='Trajectory')
    ax1.plot(*traj[0, :2], 'go', ms=10, label='Start')
    ax1.plot(*traj[-1, :2], 'rs', ms=10, label='End')

    for lc in closures:
        pi = kf_poses[lc[0]]
        pj = kf_poses[lc[1]]
        ax1.plot([pi[0,2], pj[0,2]], [pi[1,2], pj[1,2]],
                 'm-', lw=1, alpha=0.6)
        ax1.plot(pi[0,2], pi[1,2], 'mx', ms=8)

    ax1.set_aspect('equal', adjustable='datalim')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.set_title(f'{len(closures)} loop closures (magenta)')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Score histogram
    if closures:
        scores = [lc[3] for lc in closures]
        ax2.hist(scores, bins=15, color='steelblue', edgecolor='k', alpha=0.8)
        ax2.axvline(0.55, color='r', linestyle='--', label='threshold=0.55')
        ax2.set_xlabel('ICP match score')
        ax2.set_ylabel('Count')
        ax2.set_title('Loop closure score distribution')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No loop closures found', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Loop closure scores')

    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(out_dir, f'q3c_{seq_name.lower()}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

    return closures


# ============================================================
# Q3d: FACTOR GRAPH OPTIMISATION
# ============================================================
def compose_poses(p1, p2):
    """Compose two [x,y,theta] vectors."""
    dx = p2[0] * np.cos(p1[2]) - p2[1] * np.sin(p1[2])
    dy = p2[0] * np.sin(p1[2]) + p2[1] * np.cos(p1[2])
    return np.array([p1[0] + dx, p1[1] + dy, p1[2] + p2[2]])


def pose_error(z_rel, p_i, p_j):
    """Error between observed relative pose and poses i, j."""
    dx  = p_j[0] - p_i[0]
    dy  = p_j[1] - p_i[1]
    dth = p_j[2] - p_i[2]
    ct, st = np.cos(p_i[2]), np.sin(p_i[2])
    ex  =  ct * dx + st * dy - z_rel[0]
    ey  = -st * dx + ct * dy - z_rel[1]
    eth = dth - z_rel[2]
    eth = (eth + np.pi) % (2 * np.pi) - np.pi
    return np.array([ex, ey, eth])


def build_pose_graph(kf_poses, closures,
                      omega_odom=100.0, omega_loop=500.0):
    """Build lists of (i, j, z_rel, omega) factors."""
    factors = []
    n = len(kf_poses)

    # Odometry edges between consecutive keyframes
    for k in range(n - 1):
        Pi = kf_poses[k]
        Pj = kf_poses[k + 1]
        dT = np.linalg.inv(Pi) @ Pj
        z  = np.array([dT[0,2], dT[1,2], np.arctan2(dT[1,0], dT[0,0])])
        factors.append((k, k + 1, z, omega_odom))

    # Loop closure edges
    for lc in closures:
        i, j, T_rel, *_ = lc
        z = np.array([T_rel[0,2], T_rel[1,2], np.arctan2(T_rel[1,0], T_rel[0,0])])
        factors.append((i, j, z, omega_loop))

    return factors


def _optimize_pose_graph_scipy(kf_poses, factors, n_iter=200):
    """
    Scipy SLSQP fallback optimiser (anchors pose 0 with equality constraint).
    """
    n = len(kf_poses)
    x0 = np.zeros(3 * n)
    for k, P in enumerate(kf_poses):
        x0[3*k]   = P[0, 2]
        x0[3*k+1] = P[1, 2]
        x0[3*k+2] = np.arctan2(P[1, 0], P[0, 0])

    def cost(x):
        total = 0.0
        for i, j, z, omega in factors:
            pi = x[3*i:3*i+3]
            pj = x[3*j:3*j+3]
            e  = pose_error(z, pi, pj)
            total += omega * float(e @ e)
        return total

    def anchor(x):
        return x[:3] - x0[:3]

    result = minimize(cost, x0, method='SLSQP',
                      constraints={'type': 'eq', 'fun': anchor},
                      options={'maxiter': n_iter, 'ftol': 1e-9})
    xopt = result.x
    opt_poses = []
    for k in range(n):
        xi, yi, ti = xopt[3*k], xopt[3*k+1], xopt[3*k+2]
        T = np.eye(3)
        T[0,0], T[0,1] =  np.cos(ti), -np.sin(ti)
        T[1,0], T[1,1] =  np.sin(ti),  np.cos(ti)
        T[0,2], T[1,2] =  xi, yi
        opt_poses.append(T)
    return opt_poses


def _optimize_pose_graph_gtsam(kf_poses, factors, verbose=False):
    """
    GTSAM Levenberg-Marquardt pose-graph optimisation (Pose2).
    Anchors pose 0 with a tight Gaussian prior (equivalent to fixing it).
    Information matrix per factor is diag(omega) across (x, y, theta).
    """
    graph   = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Anchor: strong prior on pose 0 so the global frame is fixed
    prior_sigmas = np.array([1e-6, 1e-6, 1e-8])
    prior_noise  = noiseModel.Diagonal.Sigmas(prior_sigmas)
    p0 = kf_poses[0]
    theta0 = np.arctan2(p0[1, 0], p0[0, 0])
    graph.add(PriorFactorPose2(0, Pose2(p0[0, 2], p0[1, 2], theta0), prior_noise))

    # Initial estimates from raw ICP trajectory
    for k, P in enumerate(kf_poses):
        theta_k = np.arctan2(P[1, 0], P[0, 0])
        initial.insert(k, Pose2(P[0, 2], P[1, 2], theta_k))

    # Between factors (odometry + loop closures)
    for i, j, z, omega in factors:
        sigma = 1.0 / np.sqrt(max(omega, 1e-9))
        noise = noiseModel.Diagonal.Sigmas(np.array([sigma, sigma, sigma]))
        graph.add(BetweenFactorPose2(i, j, Pose2(z[0], z[1], z[2]), noise))

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(200)
    params.setRelativeErrorTol(1e-8)
    params.setAbsoluteErrorTol(1e-8)
    if verbose:
        params.setVerbosityLM('SUMMARY')

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    if verbose:
        print(f"    GTSAM: initial error={graph.error(initial):.4f}, "
              f"final error={graph.error(result):.4f}, "
              f"iters={optimizer.iterations()}")

    opt_poses = []
    for k in range(len(kf_poses)):
        p  = result.atPose2(k)
        ct, st = np.cos(p.theta()), np.sin(p.theta())
        T  = np.eye(3)
        T[0, 0], T[0, 1] = ct, -st
        T[1, 0], T[1, 1] = st,  ct
        T[0, 2], T[1, 2] = p.x(), p.y()
        opt_poses.append(T)
    return opt_poses


def optimize_pose_graph(kf_poses, factors, n_iter=200):
    """
    Factor-graph pose-graph optimisation.
    Uses GTSAM (Levenberg-Marquardt) when available, else scipy SLSQP.
    Returns optimised list of 3x3 pose matrices.
    """
    if _HAVE_GTSAM:
        return _optimize_pose_graph_gtsam(kf_poses, factors, verbose=True)
    return _optimize_pose_graph_scipy(kf_poses, factors, n_iter=n_iter)


def closure_error(trajectory):
    """Euclidean distance between start and end pose (metres)."""
    if len(trajectory) < 2:
        return 0.0
    return float(np.linalg.norm(trajectory[-1, :2] - trajectory[0, :2]))


def run_q3d(seq_name, slam_result, closures, out_dir):
    """Factor graph optimisation and before/after comparison."""
    print(f"\n=== Q3d Factor Graph: {seq_name} ===")

    kf_poses = slam_result['kf_poses']
    traj_raw = slam_result['trajectory']

    # Build raw trajectory from kf_poses (x, y, theta)
    raw_xyt = np.array([[P[0,2], P[1,2], np.arctan2(P[1,0], P[0,0])]
                         for P in kf_poses])

    # Compute closure error before optimisation
    err_before = float(np.linalg.norm(raw_xyt[-1, :2] - raw_xyt[0, :2]))
    print(f"  Closure error BEFORE: {err_before:.4f} m")

    factors    = build_pose_graph(kf_poses, closures)
    opt_poses  = optimize_pose_graph(kf_poses, factors)
    opt_xyt    = np.array([[P[0,2], P[1,2], np.arctan2(P[1,0], P[0,0])]
                            for P in opt_poses])
    err_after  = float(np.linalg.norm(opt_xyt[-1, :2] - opt_xyt[0, :2]))
    print(f"  Closure error AFTER:  {err_after:.4f} m")
    print(f"  Improvement:          {(err_before - err_after):.4f} m  "
          f"({100*(err_before - err_after)/max(err_before,1e-6):.1f}%)")

    # Plot: before vs after, plus closure error bar
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Q3d Factor Graph Optimisation — {seq_name}', fontweight='bold')

    ax1, ax2, ax3 = axes
    ax1.plot(raw_xyt[:, 0], raw_xyt[:, 1], 'b-', lw=1.5, label='Before')
    ax1.plot(*raw_xyt[0, :2], 'go', ms=10); ax1.plot(*raw_xyt[-1, :2], 'rs', ms=10)
    ax1.set_title('Before optimisation')
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.grid(True, alpha=0.3); ax1.legend()

    ax2.plot(opt_xyt[:, 0], opt_xyt[:, 1], 'r-', lw=1.5, label='After')
    ax2.plot(*opt_xyt[0, :2], 'go', ms=10); ax2.plot(*opt_xyt[-1, :2], 'rs', ms=10)
    # Loop closure arcs
    for lc in closures:
        pi, pj = opt_poses[lc[0]], opt_poses[lc[1]]
        ax2.plot([pi[0,2], pj[0,2]], [pi[1,2], pj[1,2]], 'm-', lw=1, alpha=0.5)
    ax2.set_title('After optimisation')
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
    ax2.grid(True, alpha=0.3); ax2.legend()

    ax3.bar(['Before', 'After'], [err_before, err_after],
            color=['steelblue', 'tomato'], alpha=0.85, edgecolor='k')
    ax3.set_ylabel('Closure error (m)')
    ax3.set_title('Closure error before vs after\n(Euclidean distance start→end)')
    ax3.grid(True, alpha=0.3, axis='y')
    for xi, val in enumerate([err_before, err_after]):
        ax3.text(xi, val + 0.01, f'{val:.3f} m', ha='center', fontsize=11,
                 fontweight='bold')
    ax3.set_ylim(0, max(err_before, 0.01) * 1.3)

    plt.tight_layout()
    out = os.path.join(out_dir, f'q3d_{seq_name.lower()}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

    # ---- Before/After occupancy grids (required by brief) ----
    _plot_q3d_occupancy_before_after(seq_name, slam_result, opt_poses, out_dir)

    return opt_poses, err_before, err_after


def _transform_local_scans_to_global(kf_global_pts, kf_poses_before, kf_poses_after):
    """
    The `map_pts` stored by run_slam are already in the ORIGINAL global frame.
    To rebuild the map with the *optimised* keyframe poses we first transform
    each keyframe's points back into its local frame using the pre-optimisation
    pose, then re-transform using the post-optimisation pose.
    """
    fixed = []
    n = min(len(kf_global_pts), len(kf_poses_before), len(kf_poses_after))
    for k in range(n):
        P_before = kf_poses_before[k]
        P_after  = kf_poses_after[k]
        pts_g    = kf_global_pts[k]
        if pts_g is None or len(pts_g) == 0:
            fixed.append(pts_g)
            continue
        # World -> local (pre-opt) -> world (post-opt)
        h_g   = np.vstack([pts_g.T, np.ones(len(pts_g))])
        h_l   = np.linalg.inv(P_before) @ h_g
        h_g2  = P_after @ h_l
        fixed.append(h_g2[:2].T)
    return fixed


def _plot_q3d_occupancy_before_after(seq_name, slam_result, opt_poses, out_dir,
                                     cell_m=0.05, grid_m=25.0):
    """
    Build and compare occupancy grids using raw-ICP keyframe poses vs the
    optimised factor-graph keyframe poses. Saves a 1x2 figure.
    """
    kf_poses_before = slam_result['kf_poses']
    kf_pts_world    = slam_result['map_pts']   # already in raw-ICP world frame

    # --- Before grid ---
    traj_before = np.array([[P[0, 2], P[1, 2], np.arctan2(P[1, 0], P[0, 0])]
                             for P in kf_poses_before])
    grid_b, origin_b = build_occupancy_grid(traj_before, kf_pts_world,
                                             cell_m=cell_m, grid_m=grid_m)

    # --- After grid: rebuild points using optimised poses ---
    kf_pts_opt = _transform_local_scans_to_global(kf_pts_world,
                                                   kf_poses_before,
                                                   opt_poses)
    traj_after = np.array([[P[0, 2], P[1, 2], np.arctan2(P[1, 0], P[0, 0])]
                            for P in opt_poses])
    grid_a, origin_a = build_occupancy_grid(traj_after, kf_pts_opt,
                                             cell_m=cell_m, grid_m=grid_m)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f'Q3d Occupancy Grid — Before vs After Factor-Graph Optimisation '
                 f'— {seq_name}', fontweight='bold')

    for ax, grid, origin, traj, title in [
        (axes[0], grid_b, origin_b, traj_before, 'Before (ICP odometry only)'),
        (axes[1], grid_a, origin_a, traj_after,  'After (factor-graph optimised)'),
    ]:
        ax.imshow(grid, cmap='gray', origin='lower',
                  extent=[origin[0], origin[0]+grid_m,
                          origin[1], origin[1]+grid_m])
        ax.plot(traj[:, 0], traj[:, 1], 'r-', lw=1.2, alpha=0.8)
        ax.plot(*traj[0, :2],  'go', ms=8, label='Start')
        ax.plot(*traj[-1, :2], 'bs', ms=8, label='End')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.legend(loc='upper right')
        ax.set_aspect('equal')

    plt.tight_layout()
    out = os.path.join(out_dir, f'q3d_grid_{seq_name.lower()}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# OCCUPANCY GRID MAPS  (for screenshots in report)
# ============================================================
def save_occupancy_grid(seq_name, slam_result, out_dir, cell_m=0.05):
    """Build and save occupancy grid image."""
    print(f"  Building occupancy grid for {seq_name}...")
    traj   = slam_result['trajectory']
    kf_pts = slam_result['map_pts']

    # Use a subset for speed if many keyframes
    step   = max(1, len(kf_pts) // 400)
    t_sub  = traj[::step][:len(kf_pts[::step])]
    k_sub  = kf_pts[::step]

    grid_m = 25.0
    grid, origin = build_occupancy_grid(t_sub, k_sub, cell_m=cell_m, grid_m=grid_m)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(grid, cmap='gray', origin='lower',
              extent=[origin[0], origin[0]+grid_m, origin[1], origin[1]+grid_m])
    # Plot trajectory
    ax.plot(traj[:, 0], traj[:, 1], 'r-', lw=1.2, alpha=0.7, label='Trajectory')
    ax.plot(*traj[0, :2], 'go', ms=8, label='Start')
    ax.plot(*traj[-1, :2], 'bs', ms=8, label='End')
    ax.set_title(f'Occupancy Grid — {seq_name}', fontweight='bold')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.legend(); ax.grid(False)
    plt.tight_layout()
    out = os.path.join(out_dir, f'occupancy_{seq_name.lower()}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# SUMMARY TABLE
# ============================================================
def print_summary(all_results):
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print(f"{'Sequence':<20} {'Scans':>6} {'KFs':>5} {'TrajLen(m)':>11} {'ClosureErr(m)':>14}")
    print("-"*70)
    for name, r in all_results.items():
        t    = r['trajectory']
        n_kf = len(r['kf_poses'])
        if len(t) > 1:
            length = float(np.sum(np.linalg.norm(np.diff(t[:, :2], axis=0), axis=1)))
        else:
            length = 0.0
        ce = closure_error(t)
        print(f"{name:<20} {len(t):>6} {n_kf:>5} {length:>11.2f} {ce:>14.4f}")
    print("="*70)


# ============================================================
# MAIN
# ============================================================
def main():
    all_results = {}

    for seq_name, path in SEQUENCES.items():
        if not os.path.exists(path):
            print(f"[SKIP] Data not found: {path}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {seq_name}")
        print(f"{'='*60}")
        scans = load_scans(path)
        print(f"Loaded {len(scans)} scans")

        # --- Baseline SLAM for Q3c / Q3d ---
        print("Running baseline SLAM...")
        result = run_slam(scans, max_range_mm=4000.0)
        all_results[seq_name] = result

        # --- Q3a: two-loop verification ---
        plot_two_loop_verification(seq_name, result, OUT_DIR)

        # --- Q3b: parameter experiments ---
        run_q3b(seq_name, scans, OUT_DIR)

        # --- Occupancy grid ---
        save_occupancy_grid(seq_name, result, OUT_DIR)

        # --- Q3c: loop closure ---
        closures = run_q3c(seq_name, result, OUT_DIR)

        # --- Q3d: factor graph ---
        run_q3d(seq_name, result, closures, OUT_DIR)

    print_summary(all_results)
    print(f"\nAll output saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
