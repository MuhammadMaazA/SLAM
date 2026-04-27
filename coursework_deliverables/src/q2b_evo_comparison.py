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
from matplotlib.collections import LineCollection

from evo.core import trajectory as evo_traj, metrics, sync
from evo.core.metrics import PoseRelation
from evo.core.sync import SyncException

_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.abspath(os.path.join(_HERE, '..'))
_BASE       = os.environ.get('SLAM_DATA', os.path.join(_ROOT, 'data'))
COLMAP_DIR  = os.path.join(_BASE, 'q2_results', 'colmap_runs')
ORBSLAM_DIR = os.path.join(_BASE, 'q2_results', 'orbslam_runs')
OUT_DIR     = os.path.abspath(os.path.join(_ROOT, '..', 'plots'))
REC1        = os.environ.get('SLAM_REC1', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings', 'tmp_recordings'))
REC2        = os.environ.get('SLAM_REC2', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings2'))
Q2_TRAJ_VARIANT = os.environ.get('Q2_TRAJ_VARIANT', 'prefer_calibrated')

SEQUENCES = [
    ('Basement_1',  'indoor'),
    ('Outdoor_1',   'outdoor'),
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


def _reject_outlier_pairs(ref_traj, est_traj, mad_scale=3.0):
    """Remove matched pairs whose per-pair translation error exceeds
    median + mad_scale * MAD.  Returns pruned (ref, est) trajectories."""
    from evo.core import metrics as _m
    from evo.core import trajectory as _t
    met = _m.APE(_m.PoseRelation.translation_part)
    met.process_data((ref_traj, est_traj))
    errors = np.array(met.error)
    median = np.median(errors)
    mad    = np.median(np.abs(errors - median))
    thresh = median + mad_scale * mad
    mask   = errors <= thresh
    if mask.sum() < 5:
        return ref_traj, est_traj, 0  # not enough inliers — keep all
    n_removed = int((~mask).sum())
    idx = np.where(mask)[0]
    ref_poses = np.array([ref_traj.poses_se3[i] for i in idx])
    est_poses = np.array([est_traj.poses_se3[i] for i in idx])
    ref_ts    = ref_traj.timestamps[idx]
    est_ts    = est_traj.timestamps[idx]
    ref_pruned = _t.PoseTrajectory3D(poses_se3=ref_poses, timestamps=ref_ts)
    est_pruned = _t.PoseTrajectory3D(poses_se3=est_poses, timestamps=est_ts)
    # Re-align after pruning so scale correction uses only inliers
    est_pruned.align(ref_pruned, correct_scale=True)
    return ref_pruned, est_pruned, n_removed


def compute_ate_evo(seq_name, orb_ts, orb_se3, colmap_names, colmap_se3):
    """
    Compute ATE using EVO with three pose relations:
      - translation_part   : standard ATE (metres)
      - rotation_angle_deg : orientation error (degrees)
      - full_transformation: combined SE(3) error
    Returns dict with rmse/median/mean for each, plus aligned trajectories.
    Outlier pairs (per-pair trans error > median + 3*MAD) are removed before
    reporting so that a small number of degenerate matches don't dominate RMSE.
    """
    try:
        ref_s, est_s = _associate_and_align(seq_name, orb_ts, orb_se3, colmap_names, colmap_se3)
        if ref_s is None:
            return None, None, None
        ref_s, est_s, n_removed = _reject_outlier_pairs(ref_s, est_s)
        if n_removed:
            print(f"    [outlier rejection] removed {n_removed} pairs (MAD gate)")
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
        return calibrated  # brief requires calibrated intrinsics when available
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

    # Print summary table to stdout
    print("\nQ2b EVO Comparison Summary (timestamp-matched):")
    print(f"  {'Sequence':<18} {'COLMAP':>7} {'ORB':>6} {'Match':>7} "
          f"{'trans (m)':>10} {'relative':>10} {'rot (deg)':>10} {'full':>10}")
    for seq, r in results.items():
        def _f(v, p=4):
            return f"{v:.{p}f}" if (v is not None and not np.isnan(v)) else 'N/A'
        print(f"  {seq:<18} {r['n_colmap']:>7d} {r['n_orb']:>6d} {r['n_assoc']:>7d} "
              f"{_f(r['ate_rmse']):>10} {_f(r.get('relative_rmse', float('nan'))):>10} "
              f"{_f(r['ate_rot_deg'], 2):>10} {_f(r['ate_full']):>10}")

    # ── Plotting helpers ──────────────────────────────────────────────────────
    CMAP_TIME = plt.get_cmap('viridis')   # blue→green→yellow (distinct from plasma)

    seq_labels = {
        'Basement_2':     'Indoor sequence (Basement_2)',
        'OnePoolStreet1': 'Outdoor sequence (OnePoolStreet1)',
    }
    n_seq = len(results)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': '#f7f7f7',
        'axes.facecolor': '#ffffff',
    })

    def gradient_line(ax, pts, cmap, lw=2.2, alpha=0.9, zorder=3):
        """Draw a continuous path coloured by normalised arc-length progress."""
        if len(pts) < 2:
            return None
        xy = pts[:, [0, 2]]   # X-Z plane
        segs = np.stack([xy[:-1], xy[1:]], axis=1)
        t = np.linspace(0, 1, len(pts))
        lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, 1),
                            linewidth=lw, alpha=alpha, zorder=zorder)
        lc.set_array(t[:-1])
        ax.add_collection(lc)
        return lc

    def add_arrows(ax, pts, n_arrows=5, color='#444444', zorder=6):
        """Overlay directional arrows evenly spaced along the path."""
        if len(pts) < 4:
            return
        step = max(1, len(pts) // (n_arrows + 1))
        for i in range(step, len(pts) - 1, step):
            dx = pts[i+1, 0] - pts[i, 0]
            dz = pts[i+1, 2] - pts[i, 2]
            ax.annotate('', xy=(pts[i+1, 0], pts[i+1, 2]),
                        xytext=(pts[i, 0], pts[i, 2]),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.0, mutation_scale=10),
                        zorder=zorder)

    # ── Figure 1: COLMAP trajectories (gradient line + arrows) ───────────────
    fig1, axes1 = plt.subplots(1, n_seq, figsize=(5.8 * n_seq, 5.4),
                               constrained_layout=True)
    fig1.patch.set_facecolor('#f7f7f7')
    if n_seq == 1:
        axes1 = [axes1]

    for ax, (seq, r) in zip(axes1, results.items()):
        pts   = r['colmap_xyz']
        label = seq_labels.get(seq, seq)
        n_poses = len(pts)
        ax.set_facecolor('#ffffff')

        if n_poses >= 2:
            lc = gradient_line(ax, pts, CMAP_TIME, lw=2.4)
            add_arrows(ax, pts, n_arrows=6)
            ax.autoscale_view()

            # colourbar from the LineCollection
            sm = plt.cm.ScalarMappable(cmap=CMAP_TIME, norm=plt.Normalize(0, 1))
            sm.set_array([])
            cb = fig1.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label('elapsed time  (0 = start,  1 = end)', fontsize=8)
            cb.ax.tick_params(labelsize=7)

            # start / end markers
            ax.plot(*pts[0, [0, 2]], 'D', color='#2ecc71', ms=8, zorder=7,
                    markeredgecolor='white', markeredgewidth=0.7, label='start')
            ax.plot(*pts[-1, [0, 2]], 'X', color='#e74c3c', ms=9, zorder=7,
                    markeredgecolor='white', markeredgewidth=0.7, label='end')

        ax.set_title(f'{label}\nCOLMAP SfM trajectory  ({n_poses} keyframes)',
                     fontsize=9, fontweight='bold', pad=10)
        ax.set_xlabel('X  (m)', fontsize=8)
        ax.set_ylabel('Z  (m)', fontsize=8)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.25, linestyle=':', color='#aaaaaa')
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='upper right', framealpha=0.85,
                  edgecolor='#cccccc')

    out1 = os.path.join(OUT_DIR, 'q2b_colmap_trajectories.png')
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"\nSaved: {out1}")

    # ── Figure 2: 2×2 side-by-side (COLMAP | ORB-SLAM2) per sequence ─────────
    seq_list = list(results.items())
    fig2, axes2 = plt.subplots(n_seq, 2, figsize=(10, 5.2 * n_seq),
                               constrained_layout=True)
    fig2.patch.set_facecolor('#f7f7f7')
    if n_seq == 1:
        axes2 = [axes2]  # make it always 2D list

    col_titles = ['COLMAP trajectory', 'ORB-SLAM2 trajectory']

    for row, (seq, r) in enumerate(seq_list):
        label   = seq_labels.get(seq, seq)
        colmap_full = r['colmap_xyz']          # full raw COLMAP trajectory
        colmap_matched = r['colmap_aligned_xyz']  # matched+aligned subset
        orb_matched    = r['orb_aligned_xyz']     # matched+aligned ORB subset

        panels = [
            (colmap_matched, len(colmap_matched), 'COLMAP  (pseudo-GT)'),
            (orb_matched,    len(orb_matched),    'ORB-SLAM2  (EVO-aligned)'),
        ]

        for col, (pts, n_poses, method) in enumerate(panels):
            ax = axes2[row][col]
            ax.set_facecolor('#ffffff')

            # Full COLMAP ghost behind both panels for context
            if len(colmap_full) >= 2:
                ax.plot(colmap_full[:, 0], colmap_full[:, 2],
                        color='#cccccc', lw=0.9, alpha=0.5, zorder=1,
                        label=f'Full COLMAP ({len(colmap_full)} poses)')

            if len(pts) >= 2:
                gradient_line(ax, pts, CMAP_TIME, lw=2.4)
                add_arrows(ax, pts, n_arrows=5)
                ax.autoscale_view()

                sm = plt.cm.ScalarMappable(cmap=CMAP_TIME,
                                           norm=plt.Normalize(0, 1))
                sm.set_array([])
                cb = fig2.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label('elapsed time  (0=start, 1=end)', fontsize=7)
                cb.ax.tick_params(labelsize=6)

                ax.plot(*pts[0, [0, 2]], 'D', color='#2ecc71', ms=7, zorder=7,
                        markeredgecolor='white', markeredgewidth=0.6)
                ax.plot(*pts[-1, [0, 2]], 'X', color='#e74c3c', ms=8, zorder=7,
                        markeredgecolor='white', markeredgewidth=0.6)

            ax.legend(fontsize=6, loc='lower right', framealpha=0.8,
                      edgecolor='#cccccc')

            # EVO metric in title (not overlaid on plot)
            ate_str = ''
            if col == 1 and not np.isnan(r['ate_rmse']):
                ate_str = (f"\nEVO ATE = {r['ate_rmse']:.3f} a.u.   "
                           f"Δrot = {r['ate_rot_deg']:.1f}°   "
                           f"n = {r['n_assoc']} matched")
            ax.set_title(f'{label}\n{method}  ({n_poses} poses){ate_str}',
                         fontsize=8, fontweight='bold', pad=8)
            ax.set_xlabel('X  (arb. units)', fontsize=8)
            ax.set_ylabel('Z  (arb. units)', fontsize=8)
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(True, alpha=0.25, linestyle=':', color='#aaaaaa')
            ax.tick_params(labelsize=7)

    out2 = os.path.join(OUT_DIR, 'q2b_colmap_vs_orbslam.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {out2}")


if __name__ == '__main__':
    main()
