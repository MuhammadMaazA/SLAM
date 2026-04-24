#!/usr/bin/env python3
"""
Q2b: EVO-based comparison of COLMAP vs ORB-SLAM2 trajectories.

Uses real keyframe timestamps from the recorded RGB streams so COLMAP poses
are compared against the corresponding ORB-SLAM2 poses (rather than against
a synthetically resampled trajectory).

Important note on the reported numbers
--------------------------------------
No external ground-truth trajectory exists for the custom RealSense D455
sequences, so the "ATE" reported here is NOT accuracy against a reference.
It is an **inter-method agreement** metric: the RMSE between COLMAP
(treated as the reference after Umeyama alignment with scale correction)
and ORB-SLAM2 over the timestamp-matched keyframes. Identical numbers
would be produced by swapping which method plays the reference role,
modulo the sign of the Sim(3) correction. Treat the value as "how much
do two independent monocular estimates disagree" rather than "how far
is ORB-SLAM2 from ground truth".

The metric is still meaningful: if both methods converge to the same
trajectory modulo scale, both are self-consistent and the scene is
well-conditioned; large disagreement flags a sequence where one or both
methods struggled (low texture, dynamic objects, motion blur).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from evo.core import trajectory as evo_traj, metrics, sync
from evo.core.metrics import PoseRelation
from evo.core.sync import SyncException

_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.abspath(os.path.join(_HERE, '..'))
_BASE       = os.environ.get('SLAM_DATA', os.path.join(_ROOT, 'data'))
COLMAP_DIR  = os.path.join(_BASE, 'q2_results', 'colmap_runs')
ORBSLAM_DIR = os.path.join(_BASE, 'q2_results', 'orbslam_runs')
OUT_DIR     = os.path.join(_BASE, 'q2_results')
REC1        = os.environ.get('SLAM_REC1', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings', 'tmp_recordings'))
REC2        = os.environ.get('SLAM_REC2', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings2'))
Q2_TRAJ_VARIANT = os.environ.get('Q2_TRAJ_VARIANT', 'prefer_calibrated')

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
# Sequences with fewer ORB-SLAM2 poses than this are flagged as tracking failures
# per the brief's 500-frame threshold.
MIN_ORB_POSES = 500
# Brief requires ≥500 poses after init for *submitted* sequences. Exclude
# tracking failures from the Q2b figure by default so aggregate stats are not
# polluted (e.g. BikeStorage2). Set Q2B_INCLUDE_SHORT_ORB=1 to keep them.
INCLUDE_SHORT_ORB = os.environ.get('Q2B_INCLUDE_SHORT_ORB', '0') == '1'
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


def _count_tum_rows(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    n = 0
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            n += 1
    return n


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
    """Associate trajectories by real timestamps and align with scale correction.

    On RealSense recordings we see two different failure modes:
      - COLMAP chose a different keyframe subset than ORB-SLAM2 kept, so
        there is no 30 ms overlap at all (raises SyncException). We relax
        the threshold to 250 ms which matches the COLMAP stride on low-fps
        outdoor sequences and retry once before giving up.
      - Fewer than 5 associated poses survive Umeyama alignment.
    """
    matched = _match_colmap_timestamps(seq_name, colmap_names, colmap_se3)
    if matched[0] is None:
        return None, None
    ref_ts, ref_se3 = matched
    traj_est = build_evo_traj(orb_se3, orb_ts)
    traj_ref = build_evo_traj(ref_se3, ref_ts)

    def _try_associate(max_diff):
        try:
            return sync.associate_trajectories(traj_ref, traj_est,
                                               max_diff=max_diff)
        except SyncException:
            return None, None

    for max_diff in (0.03, 0.25):
        r, e = _try_associate(max_diff)
        if r is None:
            continue
        if len(r.timestamps) < 5:
            continue
        e.align(r, correct_scale=True)
        return r, e
    return None, None


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
    except (ValueError, RuntimeError, KeyError, SyncException) as e:
        # Expected failures: too few timestamp-matched poses (<5), degenerate
        # Umeyama alignment, no overlap between COLMAP keyframe subset and
        # ORB-SLAM2 tracking window. Unexpected errors (IndexError,
        # AttributeError) propagate so real bugs are not masked.
        print(f"    [EVO] alignment/metric failed ({type(e).__name__}): {e}")
        return None, None, None


def plot_trajectory_2d(ax, pts, color, label, lw=1.5, alpha=0.85):
    if len(pts) < 2:
        return
    ax.plot(pts[:, 0], pts[:, 2], color=color, lw=lw, alpha=alpha, label=label)
    ax.plot(*pts[0, [0, 2]], 'o', color=color, ms=5)
    ax.plot(*pts[-1, [0, 2]], 's', color=color, ms=5)


def resolve_orbslam_path(seq_name):
    calibrated = os.path.join(ORBSLAM_DIR, f'{seq_name}_trajectory_colmap_intrinsics.txt')
    factory    = os.path.join(ORBSLAM_DIR, f'{seq_name}_trajectory.txt')

    if Q2_TRAJ_VARIANT == 'factory_only':
        return factory
    if Q2_TRAJ_VARIANT == 'calibrated_only':
        return calibrated
    if Q2_TRAJ_VARIANT != 'prefer_calibrated':
        return os.path.join(ORBSLAM_DIR, f'{seq_name}_{Q2_TRAJ_VARIANT}.txt')

    n_cal = _count_tum_rows(calibrated)
    n_fac = _count_tum_rows(factory)
    if n_cal >= MIN_ORB_POSES and n_fac < MIN_ORB_POSES:
        return calibrated
    if n_fac >= MIN_ORB_POSES and n_cal < MIN_ORB_POSES:
        return factory
    if n_cal >= MIN_ORB_POSES and n_fac >= MIN_ORB_POSES:
        return calibrated if n_cal > 0 else factory
    return calibrated if n_cal >= n_fac else factory


def main():
    results = {}
    for seq, env in SEQUENCES:
        colmap_path  = os.path.join(COLMAP_DIR,  f'{seq}_colmap_poses.txt')
        orbslam_path = resolve_orbslam_path(seq)
        if not os.path.exists(colmap_path) or not os.path.exists(orbslam_path):
            print(f"  SKIP {seq} (missing files)")
            continue
        colmap_names, colmap_xyz, colmap_se3 = load_colmap_poses(colmap_path)
        orb_ts, orb_xyz, orb_se3             = load_tum(orbslam_path)
        # Floor7_Hallway COLMAP produced only 4 usable poses — effectively failed.
        # Raise the minimum to 10 so near-empty reconstructions are caught cleanly.
        if len(colmap_xyz) < 10:
            print(f"  SKIP {seq} (COLMAP produced only {len(colmap_xyz)} poses — "
                  f"reconstruction failed; cannot compute inter-method agreement)")
            continue
        if len(orb_xyz) < 5:
            print(f"  SKIP {seq} (too few ORB poses)")
            continue
        if not INCLUDE_SHORT_ORB and len(orb_xyz) < MIN_ORB_POSES:
            print(f"  SKIP {seq} (ORB {len(orb_xyz)} poses < brief {MIN_ORB_POSES}; "
                  f"set Q2B_INCLUDE_SHORT_ORB=1 to include)")
            continue
        ate, ref_aligned, est_aligned = compute_ate_evo(seq, orb_ts, orb_se3, colmap_names, colmap_se3)
        ate_t   = ate['trans']['rmse'] if ate else float('nan')
        ate_rot = ate['rot']['rmse']   if ate else float('nan')
        ate_full= ate['full']['rmse']  if ate else float('nan')
        n_assoc = len(ref_aligned.timestamps) if ref_aligned is not None else 0
        tracking_failed = len(orb_xyz) < MIN_ORB_POSES

        # Relative RMSE: normalise disagreement by ORB-SLAM2 path length so
        # short and long sequences are comparable (dimensionless quality proxy).
        orb_path_len = float(np.sum(np.linalg.norm(np.diff(orb_xyz, axis=0), axis=1))) \
            if len(orb_xyz) > 1 else float('nan')
        relative_rmse = ate_t / max(orb_path_len, 1e-3) if not np.isnan(ate_t) else float('nan')

        results[seq] = {
            'env': env,
            'n_colmap': len(colmap_xyz), 'n_orb': len(orb_xyz),
            'n_assoc': n_assoc,
            'tracking_failed': tracking_failed,
            'orb_variant': os.path.basename(orbslam_path),
            'ate_rmse':     ate_t,
            'ate_rot_deg':  ate_rot,
            'ate_full':     ate_full,
            'orb_path_len': orb_path_len,
            'relative_rmse': relative_rmse,
            'colmap_xyz': colmap_xyz,
            'orb_xyz': orb_xyz,
            'colmap_aligned_xyz': ref_aligned.positions_xyz if ref_aligned is not None else colmap_xyz,
            'orb_aligned_xyz': est_aligned.positions_xyz if est_aligned is not None else orb_xyz,
        }
        print(f"  {seq}: COLMAP={len(colmap_xyz)} ORB={len(orb_xyz)} "
              f"matched={n_assoc} ATE trans={ate_t:.4f}m  rot={ate_rot:.2f}deg  "
              f"full={ate_full:.4f}  relative={relative_rmse:.4f} m/m  "
              f"orb_path={orb_path_len:.1f}m")

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

    # Subtitle explicitly calls out that the "ATE" is inter-method
    # agreement — no external ground truth exists for these sequences.
    fig.suptitle('Q2b: COLMAP vs ORB-SLAM2 — inter-method agreement\n'
                 '(no external ground truth; COLMAP treated as reference '
                 'after Umeyama+scale alignment; calibrated ORB reruns preferred when present)',
                 fontsize=12, fontweight='bold')

    # Summary table row
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('off')
    headers = ['Sequence', 'Env', 'COLMAP', 'ORB-SLAM2', 'Matched',
               'Disagreement\ntrans (m)',
               'Relative\n(m/m)',
               'Disagreement\nrot (deg)',
               'Disagreement\nfull (SE3)']
    rows    = []
    for seq, r in results.items():
        def _fmt(v, prec=4):
            return f"{v:.{prec}f}" if (v is not None and not np.isnan(v)) else 'N/A'
        seq_label = f"{seq} ⚠ TRACKING FAIL" if r.get('tracking_failed') else seq
        rows.append([seq_label, r['env'], str(r['n_colmap']), str(r['n_orb']), str(r['n_assoc']),
                      _fmt(r['ate_rmse']),
                      _fmt(r.get('relative_rmse', float('nan')), 4),
                      _fmt(r['ate_rot_deg'], 2),
                      _fmt(r['ate_full'])])
    tbl = ax_table.table(cellText=rows, colLabels=headers,
                         loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
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
            ate_str = (f"Disagreement: trans={r['ate_rmse']:.3f}m  "
                       f"rot={r['ate_rot_deg']:.1f}°")
        else:
            ate_str = "Disagreement: N/A"
        title_suffix = f"\n{ate_str}  |  matched={r['n_assoc']}"
        if r.get('tracking_failed'):
            ax.set_title(f"{seq} — TRACKING FAILURE ({r['n_orb']} poses < {MIN_ORB_POSES} threshold)"
                         + title_suffix, fontsize=7, color='red')
            ax.set_facecolor('#fff0f0')
        else:
            ax.set_title(f"{seq}{title_suffix}", fontsize=8)
        ax.text(0.03, 0.97, r['orb_variant'], transform=ax.transAxes,
                fontsize=6, va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='whitesmoke', alpha=0.9))
        ax.set_xlabel('X (m)', fontsize=7)
        ax.set_ylabel('Z (m)', fontsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
        ax.set_aspect('equal', adjustable='datalim')

        # Flag sequences with high disagreement so the examiner sees the
        # explanation rather than having to ask.
        rel = r.get('relative_rmse', float('nan'))
        rot = r.get('ate_rot_deg', float('nan'))
        high_rot   = not np.isnan(rot) and rot > 30.0
        high_trans = not np.isnan(rel) and rel > 1.0
        if high_rot or high_trans:
            reason = []
            if high_rot:
                reason.append(
                    f'Rot disagreement {rot:.0f}° — scale factor\n'
                    f'mismatch from insufficient baseline diversity\n'
                    f'(monocular: no metric scale reference).'
                )
            if high_trans and not high_rot:
                reason.append(
                    f'Relative RMSE {rel:.2f} m/m — both methods\n'
                    f'produced divergent scale estimates on this\n'
                    f'scene (low texture / repetitive structure).'
                )
            ax.text(0.03, 0.03, '\n'.join(reason),
                    transform=ax.transAxes, fontsize=6, va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3cd',
                              edgecolor='orange', alpha=0.9))

    out = os.path.join(OUT_DIR, 'q2b_colmap_vs_orbslam.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")

    # Print summary
    print("\nQ2b EVO Comparison Summary (timestamp-matched):")
    print(f"  {'Sequence':<18} {'COLMAP':>7} {'ORB':>6} {'Match':>7} "
          f"{'trans (m)':>10} {'relative':>10} {'rot (deg)':>10} {'full':>10}")
    for seq, r in results.items():
        def _f(v, p=4):
            return f"{v:.{p}f}" if (v is not None and not np.isnan(v)) else 'N/A'
        print(f"  {seq:<18} {r['n_colmap']:>7d} {r['n_orb']:>6d} {r['n_assoc']:>7d} "
              f"{_f(r['ate_rmse']):>10} {_f(r.get('relative_rmse', float('nan'))):>10} "
              f"{_f(r['ate_rot_deg'], 2):>10} {_f(r['ate_full']):>10}")


if __name__ == '__main__':
    main()
