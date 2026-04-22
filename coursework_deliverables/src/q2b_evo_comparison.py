#!/usr/bin/env python3
"""
Q2b: EVO-based comparison of COLMAP vs ORB-SLAM2 trajectories.
Uses real keyframe timestamps from the recorded RGB streams so COLMAP poses are
compared against the corresponding ORB-SLAM2 poses, rather than against a
synthetically resampled trajectory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from evo.core import trajectory as evo_traj, metrics, sync
from evo.core.metrics import PoseRelation

_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.abspath(os.path.join(_HERE, '..'))
_BASE       = os.environ.get('SLAM_DATA', os.path.join(_ROOT, 'data'))
COLMAP_DIR  = os.path.join(_BASE, 'q2_results', 'colmap_runs')
ORBSLAM_DIR = os.path.join(_BASE, 'q2_results', 'orbslam_runs')
OUT_DIR     = os.path.join(_BASE, 'q2_results')
REC1        = os.environ.get('SLAM_REC1', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings', 'tmp_recordings'))
REC2        = os.environ.get('SLAM_REC2', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings2'))

SEQUENCES = [
    ('OnePoolStreet1',  'outdoor'),
    ('Basement_1',      'indoor'),
    ('Outdoor_1',       'outdoor'),
    ('Basement_2',      'indoor'),
    ('BikeStorage',     'outdoor'),
    ('BikeStorage2',    'outdoor'),
    ('Washroom',        'indoor'),
    ('Entrance2',       'outdoor'),
    ('Floor7_Hallway',  'indoor'),
]

SEQ_COLORS = {'outdoor': '#ff6d00', 'indoor': '#00e5ff'}
SEQUENCE_CAMERA_DIRS = {
    "Basement_1":     os.path.join(REC1, "Basement_1", "camera"),
    "Basement_2":     os.path.join(REC1, "Basement_2", "camera"),
    "Floor7_Hallway": os.path.join(REC1, "Floor7_Hallway", "camera"),
    "Outdoor_1":      os.path.join(REC1, "Outdoor_1", "camera"),
    "Washroom":       os.path.join(REC1, "Washroom", "camera"),
    "BikeStorage":    os.path.join(REC2, "BikeStorage", "camera"),
    "BikeStorage2":   os.path.join(REC2, "BikeStorage2", "camera"),
    "Entrance2":      os.path.join(REC2, "Entrance2", "camera"),
    "OnePoolStreet1": os.path.join(REC2, "OnePoolStreet1", "camera"),
}


def load_tum(path):
    """Load TUM format trajectory → (timestamps, xyz, 4x4 SE(3) poses).
    TUM format: ts tx ty tz qx qy qz qw (camera-in-world).
    """
    ts, xyz, poses = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            if len(p) >= 8:
                tx, ty, tz = float(p[1]), float(p[2]), float(p[3])
                qx, qy, qz, qw = float(p[4]), float(p[5]), float(p[6]), float(p[7])
                R = quat_to_rot(qw, qx, qy, qz)
                T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [tx, ty, tz]
                ts.append(float(p[0]))
                xyz.append([tx, ty, tz])
                poses.append(T)
            elif len(p) >= 4:
                ts.append(float(p[0]))
                xyz.append([float(p[1]), float(p[2]), float(p[3])])
                T = np.eye(4); T[:3, 3] = [float(p[1]), float(p[2]), float(p[3])]
                poses.append(T)
    return (np.array(ts),
            np.array(xyz) if xyz else np.zeros((0, 3)),
            np.array(poses) if poses else np.zeros((0, 4, 4)))


def load_colmap_poses(path):
    """Load COLMAP poses.
    Returns frame names, camera centres, and camera-in-world SE(3) poses.
    Input stores world-to-cam; we invert to get cam-in-world.
    """
    names, positions, se3 = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            if len(p) >= 8:
                names.append(p[0])
                tx, ty, tz = float(p[1]), float(p[2]), float(p[3])
                qx, qy, qz, qw = float(p[4]), float(p[5]), float(p[6]), float(p[7])
                # world-to-cam R
                R_w2c = quat_to_rot(qw, qx, qy, qz)
                t_w2c = np.array([tx, ty, tz])
                # camera-in-world
                R_c2w = R_w2c.T
                t_c2w = -R_w2c.T @ t_w2c
                T = np.eye(4); T[:3, :3] = R_c2w; T[:3, 3] = t_c2w
                positions.append(t_c2w)
                se3.append(T)
    return (names,
            np.array(positions) if positions else np.zeros((0, 3)),
            np.array(se3)       if se3       else np.zeros((0, 4, 4)))


def quat_to_rot(qw, qx, qy, qz):
    """Quaternion to 3x3 rotation matrix."""
    n = qw*qw + qx*qx + qy*qy + qz*qz
    if n < 1e-10:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s*(qy*qy + qz*qz),   s*(qx*qy - qz*qw),   s*(qx*qz + qy*qw)],
        [s*(qx*qy + qz*qw),   1 - s*(qx*qx + qz*qz),   s*(qy*qz - qx*qw)],
        [s*(qx*qz - qy*qw),       s*(qy*qz + qx*qw), 1 - s*(qx*qx + qy*qy)],
    ])


def build_evo_traj(poses_se3, timestamps):
    """Convert SE(3) poses plus real timestamps to an EVO trajectory."""
    return evo_traj.PoseTrajectory3D(
        poses_se3=np.asarray(poses_se3, dtype=float),
        timestamps=np.asarray(timestamps, dtype=float)
    )


def load_rgb_timestamp_map(seq_name):
    """Map image basename -> capture timestamp from the recorded rgb.txt."""
    cam_dir = SEQUENCE_CAMERA_DIRS.get(seq_name)
    if not cam_dir:
        return {}
    rgb_txt = os.path.join(cam_dir, 'rgb.txt')
    if not os.path.exists(rgb_txt):
        return {}

    ts_map = {}
    with open(rgb_txt) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                ts_map[os.path.basename(parts[1])] = float(parts[0])
    return ts_map


def _match_colmap_timestamps(seq_name, colmap_names, colmap_se3):
    """Attach recorded RGB timestamps to each COLMAP keyframe pose."""
    ts_map = load_rgb_timestamp_map(seq_name)
    timestamps, matched_se3 = [], []
    for name, pose in zip(colmap_names, colmap_se3):
        ts = ts_map.get(name)
        if ts is None:
            continue
        timestamps.append(ts)
        matched_se3.append(pose)
    if len(matched_se3) < 5:
        return None, None
    return np.asarray(timestamps, dtype=float), np.asarray(matched_se3, dtype=float)


def _associate_and_align(seq_name, orb_ts, orb_se3, colmap_names, colmap_se3):
    """Associate trajectories by real timestamps and align with scale correction."""
    matched = _match_colmap_timestamps(seq_name, colmap_names, colmap_se3)
    if matched[0] is None:
        return None, None
    ref_ts, ref_se3 = matched
    traj_est = build_evo_traj(orb_se3, orb_ts)
    traj_ref = build_evo_traj(ref_se3, ref_ts)
    traj_ref_s, traj_est_s = sync.associate_trajectories(traj_ref, traj_est,
                                                         max_diff=0.03)
    if len(traj_ref_s.timestamps) < 5:
        return None, None
    traj_est_s.align(traj_ref_s, correct_scale=True)
    return traj_ref_s, traj_est_s


def compute_ate_evo(seq_name, orb_ts, orb_se3, colmap_names, colmap_se3):
    """
    Compute ATE using EVO with three pose relations:
      - translation_part   : standard ATE (metres)
      - rotation_angle_deg : orientation error (degrees)
      - full_transformation: combined SE(3) error
    Returns dict with rmse/median/mean for each, plus aligned trajectories.
    """
    try:
        ref_s, est_s = _associate_and_align(seq_name, orb_ts, orb_se3, colmap_names, colmap_se3)
        if ref_s is None:
            return None, None, None
        out = {}
        for key, rel in [('trans', PoseRelation.translation_part),
                         ('rot',   PoseRelation.rotation_angle_deg),
                         ('full',  PoseRelation.full_transformation)]:
            m = metrics.APE(rel)
            m.process_data((ref_s, est_s))
            s = m.get_all_statistics()
            out[key] = {'rmse': float(s['rmse']),
                         'median': float(s['median']),
                         'mean':   float(s['mean']),
                         'max':    float(s['max'])}
        return out, ref_s, est_s
    except Exception as e:
        print(f"    [EVO] {e}")
        return None, None, None


def plot_trajectory_2d(ax, pts, color, label, lw=1.5, alpha=0.85):
    if len(pts) < 2:
        return
    ax.plot(pts[:, 0], pts[:, 2], color=color, lw=lw, alpha=alpha, label=label)
    ax.plot(*pts[0, [0, 2]], 'o', color=color, ms=5)
    ax.plot(*pts[-1, [0, 2]], 's', color=color, ms=5)


def main():
    results = {}
    for seq, env in SEQUENCES:
        colmap_path  = os.path.join(COLMAP_DIR,  f'{seq}_colmap_poses.txt')
        orbslam_path = os.path.join(ORBSLAM_DIR, f'{seq}_trajectory.txt')
        if not os.path.exists(colmap_path) or not os.path.exists(orbslam_path):
            print(f"  SKIP {seq} (missing files)")
            continue
        colmap_names, colmap_xyz, colmap_se3 = load_colmap_poses(colmap_path)
        orb_ts, orb_xyz, orb_se3             = load_tum(orbslam_path)
        if len(colmap_xyz) < 5 or len(orb_xyz) < 5:
            print(f"  SKIP {seq} (too few poses)")
            continue
        ate, ref_aligned, est_aligned = compute_ate_evo(seq, orb_ts, orb_se3, colmap_names, colmap_se3)
        ate_t   = ate['trans']['rmse'] if ate else float('nan')
        ate_rot = ate['rot']['rmse']   if ate else float('nan')
        ate_full= ate['full']['rmse']  if ate else float('nan')
        n_assoc = len(ref_aligned.timestamps) if ref_aligned is not None else 0
        results[seq] = {
            'env': env,
            'n_colmap': len(colmap_xyz), 'n_orb': len(orb_xyz),
            'n_assoc': n_assoc,
            'ate_rmse':     ate_t,
            'ate_rot_deg':  ate_rot,
            'ate_full':     ate_full,
            'colmap_xyz': colmap_xyz,
            'orb_xyz': orb_xyz,
            'colmap_aligned_xyz': ref_aligned.positions_xyz if ref_aligned is not None else colmap_xyz,
            'orb_aligned_xyz': est_aligned.positions_xyz if est_aligned is not None else orb_xyz,
        }
        print(f"  {seq}: COLMAP={len(colmap_xyz)} ORB={len(orb_xyz)} "
              f"matched={n_assoc} ATE trans={ate_t:.4f}m  rot={ate_rot:.2f}deg  full={ate_full:.4f}")

    if not results:
        print("No sequences processed")
        return

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_seq = len(results)
    n_cols = min(3, n_seq)
    n_rows = (n_seq + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows + 2))
    gs  = gridspec.GridSpec(n_rows + 1, n_cols, figure=fig,
                            height_ratios=[0.35] + [1] * n_rows, hspace=0.45, wspace=0.35)

    fig.suptitle('Q2b: COLMAP vs ORB-SLAM2 Trajectory Comparison (EVO ATE)',
                 fontsize=13, fontweight='bold')

    # Summary table row
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('off')
    headers = ['Sequence', 'Env', 'COLMAP', 'ORB-SLAM2', 'Matched',
               'ATE trans (m)', 'ATE rot (deg)', 'ATE full (SE3)']
    rows    = []
    for seq, r in results.items():
        def _fmt(v, prec=4):
            return f"{v:.{prec}f}" if (v is not None and not np.isnan(v)) else 'N/A'
        rows.append([seq, r['env'], str(r['n_colmap']), str(r['n_orb']), str(r['n_assoc']),
                      _fmt(r['ate_rmse']),
                      _fmt(r['ate_rot_deg'], 2),
                      _fmt(r['ate_full'])])
    tbl = ax_table.table(cellText=rows, colLabels=headers,
                         loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f8f9fa')

    # Trajectory plots
    for i, (seq, r) in enumerate(results.items()):
        row = i // n_cols + 1
        col = i % n_cols
        ax  = fig.add_subplot(gs[row, col])

        color_env = SEQ_COLORS.get(r['env'], 'gray')
        plot_trajectory_2d(ax, r['colmap_aligned_xyz'], '#f39c12', 'COLMAP (matched ref)', lw=1.2)
        plot_trajectory_2d(ax, r['orb_aligned_xyz'],    color_env, 'ORB-SLAM2 (matched+aligned)', lw=1.5)

        if not np.isnan(r['ate_rmse']):
            ate_str = (f"ATE: trans={r['ate_rmse']:.3f}m  "
                       f"rot={r['ate_rot_deg']:.1f}°")
        else:
            ate_str = "ATE: N/A"
        ax.set_title(f"{seq}\n{ate_str}  |  matched={r['n_assoc']}", fontsize=8)
        ax.set_xlabel('X (m)', fontsize=7)
        ax.set_ylabel('Z (m)', fontsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
        ax.set_aspect('equal', adjustable='datalim')

    out = os.path.join(OUT_DIR, 'q2b_colmap_vs_orbslam.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")

    # Print summary
    print("\nQ2b EVO Comparison Summary (timestamp-matched):")
    print(f"  {'Sequence':<18} {'COLMAP':>7} {'ORB':>6} {'Match':>7} "
          f"{'trans (m)':>10} {'rot (deg)':>10} {'full':>10}")
    for seq, r in results.items():
        def _f(v, p=4):
            return f"{v:.{p}f}" if (v is not None and not np.isnan(v)) else 'N/A'
        print(f"  {seq:<18} {r['n_colmap']:>7d} {r['n_orb']:>6d} {r['n_assoc']:>7d} "
              f"{_f(r['ate_rmse']):>10} {_f(r['ate_rot_deg'], 2):>10} "
              f"{_f(r['ate_full']):>10}")


if __name__ == '__main__':
    main()
