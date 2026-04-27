#!/usr/bin/env python3
"""
Q2b: 3D point-cloud visualisations of the COLMAP sparse reconstructions.

Reads each sequence's COLMAP output (`points3D.txt` for the 3D map, optional
`images.txt` for the camera trajectory) and renders two views:
  - an orthographic top-down scatter (XZ plane)
  - an isometric 3D scatter

The goal is to satisfy the coursework-brief requirement to visualise the
reconstructed 3D map alongside the ORB-SLAM2 trajectory comparison.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.abspath(os.path.join(_HERE, '..'))
BASE    = os.environ.get('SLAM_DATA', os.path.join(_ROOT, 'data'))
COLMAP  = os.path.join(BASE, 'q2_results', 'colmap_runs')
ORB_DIR = os.path.join(BASE, 'q2_results', 'orbslam_runs')
OUT_DIR = os.path.abspath(os.path.join(_ROOT, '..', 'plots'))
os.makedirs(OUT_DIR, exist_ok=True)


def quat_to_rot(qw, qx, qy, qz):
    n = qw*qw + qx*qx + qy*qy + qz*qz
    if n < 1e-10:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s*(qy*qy + qz*qz),   s*(qx*qy - qz*qw),   s*(qx*qz + qy*qw)],
        [s*(qx*qy + qz*qw),   1 - s*(qx*qx + qz*qz),   s*(qy*qz - qx*qw)],
        [s*(qx*qz - qy*qw),       s*(qy*qz + qx*qw), 1 - s*(qx*qx + qy*qy)],
    ])


def load_points3D(path):
    """
    Parse COLMAP points3D.txt. Each data line begins with POINT3D_ID followed
    by X Y Z R G B ERROR TRACK[].
    Returns (Nx3 xyz, Nx3 rgb in [0,1], N errors).
    """
    xyz, rgb, err = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            if len(p) < 8:
                continue
            try:
                x, y, z = float(p[1]), float(p[2]), float(p[3])
                r, g, b = int(p[4]), int(p[5]), int(p[6])
                e       = float(p[7])
            except ValueError:
                continue
            xyz.append([x, y, z])
            rgb.append([r / 255.0, g / 255.0, b / 255.0])
            err.append(e)
    return (np.array(xyz) if xyz else np.zeros((0, 3)),
            np.array(rgb) if rgb else np.zeros((0, 3)),
            np.array(err) if err else np.zeros(0))


def load_camera_centres(images_txt):
    """
    Parse COLMAP images.txt and extract the camera centre of each registered
    image (camera-in-world: C = -R^T t, with R from world-to-cam quaternion).
    COLMAP images.txt uses TWO lines per image; we only need odd (meta) lines.
    """
    if not os.path.exists(images_txt):
        return np.zeros((0, 3))
    centres = []
    with open(images_txt) as f:
        lines = [ln for ln in f if ln.strip() and not ln.startswith('#')]
    # Every other line starting at index 0 is the pose metadata line.
    for ln in lines[::2]:
        p = ln.split()
        if len(p) < 8:
            continue
        try:
            qw, qx, qy, qz = map(float, p[1:5])
            tx, ty, tz     = map(float, p[5:8])
        except ValueError:
            continue
        R = quat_to_rot(qw, qx, qy, qz)
        C = -R.T @ np.array([tx, ty, tz])
        centres.append(C)
    return np.array(centres) if centres else np.zeros((0, 3))


def _filter_inliers(xyz, rgb, err, err_quantile=0.95, bbox_quantile=0.99):
    """Drop points with large reprojection error or extreme spatial outliers."""
    if len(xyz) == 0:
        return xyz, rgb
    if len(err) == len(xyz):
        cutoff = np.quantile(err, err_quantile)
        m = err < cutoff
        xyz = xyz[m]; rgb = rgb[m]
    # Clip spatial outliers (e.g. points at infinity)
    if len(xyz) > 0:
        lo = np.quantile(xyz, 1 - bbox_quantile, axis=0)
        hi = np.quantile(xyz, bbox_quantile,     axis=0)
        m  = np.all((xyz >= lo) & (xyz <= hi), axis=1)
        xyz = xyz[m]; rgb = rgb[m]
    return xyz, rgb


def render_sequence(seq_name):
    sparse_dir = os.path.join(COLMAP, f'{seq_name}_sparse')
    if not os.path.isdir(sparse_dir):
        return None
    pts_path = os.path.join(sparse_dir, 'points3D.txt')
    img_path = os.path.join(sparse_dir, 'images.txt')
    if not os.path.exists(pts_path):
        return None

    xyz, rgb, err = load_points3D(pts_path)
    if len(xyz) < 20:
        print(f"  {seq_name}: only {len(xyz)} points, skipping")
        return None

    cams = load_camera_centres(img_path)
    xyz_f, rgb_f = _filter_inliers(xyz, rgb, err)

    fig = plt.figure(figsize=(15, 6.5))
    fig.suptitle(f'Q2b: COLMAP 3D sparse map — {seq_name} '
                 f'({len(xyz_f)}/{len(xyz)} points)',
                 fontsize=12, fontweight='bold')

    # --- Top-down (XZ) ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(xyz_f[:, 0], xyz_f[:, 2], c=rgb_f, s=1.2, alpha=0.7)
    if len(cams) > 1:
        ax1.plot(cams[:, 0], cams[:, 2], 'r-', lw=1.3, alpha=0.85,
                  label=f'Cameras (n={len(cams)})')
        ax1.plot(cams[0, 0],  cams[0, 2],  'go', ms=7, label='Start')
        ax1.plot(cams[-1, 0], cams[-1, 2], 'bs', ms=7, label='End')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Z (m)')
    ax1.set_title('Top-down view (XZ)')
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.grid(True, alpha=0.3)
    if len(cams) > 1:
        ax1.legend(fontsize=8)

    # --- 3D view ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(xyz_f[:, 0], xyz_f[:, 2], xyz_f[:, 1],
                c=rgb_f, s=1.0, alpha=0.6, depthshade=False)
    if len(cams) > 1:
        ax2.plot(cams[:, 0], cams[:, 2], cams[:, 1],
                 'r-', lw=1.2, alpha=0.9, label='Cameras')
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Z (m)'); ax2.set_zlabel('Y (m)')
    ax2.set_title('Isometric 3D')
    ax2.view_init(elev=20, azim=-60)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'q2b_pointcloud_{seq_name.lower()}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {seq_name}: saved {out}  "
          f"(points={len(xyz_f)}, cameras={len(cams)})")
    return out


def main():
    print("=== Q2b: COLMAP 3D point-cloud rendering ===")
    sequences = sorted([d[:-len('_sparse')] for d in os.listdir(COLMAP)
                        if d.endswith('_sparse')])
    for seq in sequences:
        render_sequence(seq)
    print(f"\nAll figures under: {OUT_DIR}")


if __name__ == '__main__':
    main()
