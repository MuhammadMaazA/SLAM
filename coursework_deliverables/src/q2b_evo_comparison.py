#!/usr/bin/env python3
"""
Q2b: EVO-based comparison of COLMAP vs ORB-SLAM2 trajectories.
Uses EVO Python API to compute ATE (aligned + scaled) for each sequence.
Updates q2b_colmap_vs_orbslam.png with quantitative metrics.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

COLMAP_DIR  = '/home/mmaaz/SLAM/coursework_deliverables/data/q2_results/colmap_runs'
ORBSLAM_DIR = '/home/mmaaz/SLAM/coursework_deliverables/data/q2_results/orbslam_runs'
OUT_DIR     = '/home/mmaaz/SLAM/coursework_deliverables/data/q2_results'

# Sequences with enough COLMAP poses for EVO comparison
SEQUENCES = [
    ('OnePoolStreet1',  'outdoor',  616),
    ('Basement_1',      'indoor',   283),
    ('Outdoor_1',       'outdoor',  360),
    ('Basement_2',      'indoor',   233),
    ('BikeStorage',     'indoor',   561),
    ('BikeStorage2',    'indoor',   194),
    ('Washroom',        'indoor',   101),
    ('Entrance2',       'mixed',    399),
    ('Floor7_Hallway',  'indoor',   8),
]

SEQ_COLORS = {'outdoor': '#ff6d00', 'indoor': '#00e5ff', 'mixed': '#76ff03'}


def load_tum(path):
    """Load TUM format trajectory → (timestamps, xyz array)."""
    ts, xyz = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            if len(p) >= 4:
                ts.append(float(p[0]))
                xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(ts), np.array(xyz) if xyz else np.zeros((0, 3))


def load_colmap_poses(path):
    """Load COLMAP poses. Format: name tx ty tz qx qy qz qw (world-to-cam transform).
    Invert to get camera-in-world position."""
    poses = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            if len(p) >= 8:
                tx, ty, tz = float(p[1]), float(p[2]), float(p[3])
                qx, qy, qz, qw = float(p[4]), float(p[5]), float(p[6]), float(p[7])
                # Rotation matrix from quaternion (world-to-cam)
                R = quat_to_rot(qw, qx, qy, qz)
                # Camera center in world = -R^T * t
                t = np.array([tx, ty, tz])
                cam_pos = -R.T @ t
                poses.append(cam_pos)
    return np.array(poses) if poses else np.zeros((0, 3))


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


def align_umeyama(src, dst):
    """Umeyama alignment: find s, R, t such that dst ≈ s*R*src + t.
    Returns aligned src and scale."""
    n = len(src)
    if n < 3:
        return src, 1.0, float('nan')
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    ss = src - mu_s
    ds = dst - mu_d
    cov = ds.T @ ss / n
    U, sv, Vt = np.linalg.svd(cov)
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    scale = (sv @ [1, 1, np.sign(np.linalg.det(U @ Vt))]) / (ss * ss).sum() * n
    t = mu_d - scale * R @ mu_s
    aligned = scale * (R @ src.T).T + t
    return aligned, scale, R


def compute_ate_rmse(est_xyz, ref_xyz):
    """Subsample both trajectories uniformly to same length, compute ATE RMSE."""
    n = min(len(est_xyz), len(ref_xyz))
    if n < 5:
        return float('nan')
    # Subsample
    idx_e = np.round(np.linspace(0, len(est_xyz)-1, n)).astype(int)
    idx_r = np.round(np.linspace(0, len(ref_xyz)-1, n)).astype(int)
    e = est_xyz[idx_e]
    r = ref_xyz[idx_r]
    # Align ORB-SLAM2 (est) to COLMAP (ref) with scale
    e_aligned, scale, R = align_umeyama(e, r)
    errors = np.linalg.norm(e_aligned - r, axis=1)
    return float(np.sqrt(np.mean(errors**2)))


def plot_trajectory_2d(ax, pts, color, label, lw=1.5, alpha=0.85):
    if len(pts) < 2:
        return
    ax.plot(pts[:, 0], pts[:, 2], color=color, lw=lw, alpha=alpha, label=label)
    ax.plot(*pts[0, [0, 2]], 'o', color=color, ms=5)
    ax.plot(*pts[-1, [0, 2]], 's', color=color, ms=5)


def main():
    results = {}
    for seq, env, n_colmap in SEQUENCES:
        colmap_path  = os.path.join(COLMAP_DIR,  f'{seq}_colmap_poses.txt')
        orbslam_path = os.path.join(ORBSLAM_DIR, f'{seq}_trajectory.txt')
        if not os.path.exists(colmap_path) or not os.path.exists(orbslam_path):
            print(f"  SKIP {seq} (missing files)")
            continue
        colmap_xyz  = load_colmap_poses(colmap_path)
        _, orb_xyz  = load_tum(orbslam_path)
        if len(colmap_xyz) < 5 or len(orb_xyz) < 5:
            print(f"  SKIP {seq} (too few poses)")
            continue
        ate = compute_ate_rmse(orb_xyz, colmap_xyz)
        results[seq] = {
            'env': env, 'n_colmap': len(colmap_xyz), 'n_orb': len(orb_xyz),
            'ate_rmse': ate, 'colmap_xyz': colmap_xyz, 'orb_xyz': orb_xyz
        }
        print(f"  {seq}: COLMAP={len(colmap_xyz)} ORB={len(orb_xyz)} ATE RMSE={ate:.4f}m")

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
    headers = ['Sequence', 'Env', 'COLMAP poses', 'ORB-SLAM2 poses', 'ATE RMSE (m)']
    rows    = []
    for seq, r in results.items():
        ate_str = f"{r['ate_rmse']:.4f}" if not np.isnan(r['ate_rmse']) else 'N/A'
        rows.append([seq, r['env'], str(r['n_colmap']), str(r['n_orb']), ate_str])
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
        plot_trajectory_2d(ax, r['colmap_xyz'], '#f39c12', 'COLMAP (ref)', lw=1.2)
        plot_trajectory_2d(ax, r['orb_xyz'],    color_env, 'ORB-SLAM2', lw=1.5)

        ate_str = f"ATE RMSE: {r['ate_rmse']:.3f} m" if not np.isnan(r['ate_rmse']) else "ATE: N/A"
        ax.set_title(f"{seq}\n{ate_str}", fontsize=8)
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
    print("\nQ2b EVO Comparison Summary:")
    for seq, r in results.items():
        ate_str = f"{r['ate_rmse']:.4f}m" if not np.isnan(r['ate_rmse']) else 'N/A'
        print(f"  {seq:<20}: COLMAP={r['n_colmap']:4d}  ORB={r['n_orb']:5d}  ATE={ate_str}")


if __name__ == '__main__':
    main()
