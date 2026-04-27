#!/usr/bin/env python3
"""
Generate a 3D rotating pointcloud + trajectory visualisation video for Q2c.
Two panels side by side: Basement_1 (indoor) and Outdoor_1 (outdoor).
Each panel shows the COLMAP sparse pointcloud with the ORB-SLAM2 trajectory
overlaid. The camera rotates 360 degrees around the scene.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
import cv2

_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.abspath(os.path.join(_HERE, '..'))
COLMAP_DIR = os.path.join(_ROOT, 'data', 'q2_results', 'colmap_runs')
ORB_DIR    = os.path.join(_ROOT, 'data', 'q2_results', 'orbslam_runs')
OUT        = os.path.join(_ROOT, 'data', 'q2_results', 'q2c_pointcloud_viz.mp4')

SEQUENCES = [
    ('Basement_1', 'indoor',  '#00e5ff', 7330),
    ('Outdoor_1',  'outdoor', '#ff6d00', 17806),
]

FPS       = 20
N_FRAMES  = 300   # 15 s at 20fps — full 360° rotation


def load_points3d(sparse_dir):
    path = os.path.join(sparse_dir, 'points3D.txt')
    xyz, rgb = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            if len(p) >= 7:
                xyz.append([float(p[1]), float(p[2]), float(p[3])])
                rgb.append([int(p[4])/255, int(p[5])/255, int(p[6])/255])
    return np.array(xyz), np.array(rgb)


def load_colmap_poses(path):
    pts = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            if len(p) >= 8:
                tx, ty, tz = float(p[1]), float(p[2]), float(p[3])
                qx, qy, qz, qw = float(p[4]), float(p[5]), float(p[6]), float(p[7])
                n = qw**2+qx**2+qy**2+qz**2
                s = 2/n if n > 1e-10 else 0
                R = np.array([
                    [1-s*(qy**2+qz**2), s*(qx*qy-qz*qw), s*(qx*qz+qy*qw)],
                    [s*(qx*qy+qz*qw),   1-s*(qx**2+qz**2), s*(qy*qz-qx*qw)],
                    [s*(qx*qz-qy*qw),   s*(qy*qz+qx*qw), 1-s*(qx**2+qy**2)],
                ])
                t_c2w = -R.T @ np.array([tx, ty, tz])
                pts.append(t_c2w)
    return np.array(pts) if pts else np.zeros((0, 3))


def load_tum(path):
    pts = []
    if not os.path.exists(path):
        return np.zeros((0, 3))
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            if len(p) >= 4:
                pts.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(pts) if pts else np.zeros((0, 3))


def make_frame(axes_data, azim):
    fig = plt.figure(figsize=(14, 6), facecolor='#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')

    for col, (seq, env, color, n_pts, pts3d, rgb3d, colmap_traj, orb_traj) in enumerate(axes_data):
        ax = fig.add_subplot(1, 2, col+1, projection='3d')
        ax.set_facecolor('#0a0a0a')

        # Pointcloud — downsample for speed
        stride = max(1, len(pts3d) // 4000)
        p = pts3d[::stride]
        c = rgb3d[::stride]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2],
                   c=c, s=0.8, alpha=0.6, linewidths=0, zorder=1)

        # COLMAP trajectory
        if len(colmap_traj) > 1:
            ax.plot(colmap_traj[:, 0], colmap_traj[:, 1], colmap_traj[:, 2],
                    color='white', lw=1.2, alpha=0.7, zorder=3, label='COLMAP')
            ax.scatter(*colmap_traj[0], color='#2ecc71', s=40, zorder=5)
            ax.scatter(*colmap_traj[-1], color='#e74c3c', s=40, zorder=5)

        # ORB-SLAM2 trajectory
        if len(orb_traj) > 1:
            t = np.linspace(0, 1, len(orb_traj))
            cmap = plt.cm.plasma
            for i in range(len(orb_traj)-1):
                ax.plot(orb_traj[i:i+2, 0], orb_traj[i:i+2, 1], orb_traj[i:i+2, 2],
                        color=cmap(t[i]), lw=1.5, alpha=0.9, zorder=4)

        ax.view_init(elev=20, azim=azim)
        ax.set_axis_off()

        label = 'Indoor — Basement\_1' if env == 'indoor' else 'Outdoor — Outdoor\_1'
        ax.set_title(f'{label}\n{n_pts:,} 3D points  |  COLMAP: white  |  ORB-SLAM2: plasma',
                     color='white', fontsize=9, pad=4)

        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('none')

    plt.tight_layout(pad=0.5)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return img


def main():
    print("Loading data...")
    axes_data = []
    for seq, env, color, n_pts in SEQUENCES:
        pts3d, rgb3d = load_points3d(os.path.join(COLMAP_DIR, f'{seq}_sparse'))
        colmap_traj  = load_colmap_poses(os.path.join(COLMAP_DIR, f'{seq}_colmap_poses.txt'))

        # prefer calibrated trajectory
        cal_path = os.path.join(ORB_DIR, f'{seq}_trajectory_colmap_intrinsics.txt')
        fac_path = os.path.join(ORB_DIR, f'{seq}_trajectory.txt')
        orb_path = cal_path if os.path.exists(cal_path) else fac_path
        orb_traj = load_tum(orb_path)

        # align ORB trajectory to COLMAP scale using Umeyama
        if len(orb_traj) > 5 and len(colmap_traj) > 5:
            # simple centroid + scale alignment
            orb_c = orb_traj.mean(0)
            col_c = colmap_traj.mean(0)
            orb_s = np.std(np.linalg.norm(orb_traj - orb_c, axis=1)) + 1e-9
            col_s = np.std(np.linalg.norm(colmap_traj - col_c, axis=1)) + 1e-9
            orb_traj = (orb_traj - orb_c) * (col_s / orb_s) + col_c

        axes_data.append((seq, env, color, n_pts, pts3d, rgb3d, colmap_traj, orb_traj))
        print(f"  {seq}: {len(pts3d)} pts, {len(colmap_traj)} COLMAP poses, {len(orb_traj)} ORB poses")

    # sample one frame to get size
    sample = make_frame(axes_data, azim=30)
    h, w = sample.shape[:2]

    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))

    print(f"Rendering {N_FRAMES} frames...")
    for i in range(N_FRAMES):
        azim = (i / N_FRAMES) * 360
        frame = make_frame(axes_data, azim)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if i % 30 == 0:
            print(f"  frame {i}/{N_FRAMES}")

    writer.release()

    # re-encode with ffmpeg for better compression
    final = OUT.replace('.mp4', '_final.mp4')
    os.system(f'ffmpeg -y -i "{OUT}" -c:v libx264 -crf 20 -pix_fmt yuv420p "{final}" 2>/dev/null')
    if os.path.exists(final):
        os.replace(final, OUT)

    print(f"Saved: {OUT}")


if __name__ == '__main__':
    main()
