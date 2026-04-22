#!/usr/bin/env python3
"""
Q2c: ORB-SLAM2 tracking demonstration — all 9 sequences in 3×3 grid.
Shows live trajectory build-up with COLMAP as faint background reference.
Output: COMP0222_CW2_GRP_1_Visual_SLAM.mp4
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.abspath(os.path.join(_HERE, '..'))
ORBSLAM_DIR = os.environ.get('SLAM_ORBSLAM_OUT', os.path.join(_ROOT, 'data', 'q2_results', 'orbslam_runs'))
COLMAP_DIR  = os.environ.get('SLAM_COLMAP_OUT',  os.path.join(_ROOT, 'data', 'q2_results', 'colmap_runs'))
OUT_VIDEO   = os.environ.get('SLAM_VIDEO_Q2',    os.path.join(_ROOT, '..', 'COMP0222_CW2_GRP_32_Visual_SLAM.mp4'))

SEQUENCES = [
    ('Basement_2',     'indoor'),
    ('Washroom',       'indoor'),
    ('Floor7_Hallway', 'indoor'),
    ('OnePoolStreet1', 'outdoor'),
    ('Outdoor_1',      'outdoor'),
    ('Entrance2',      'outdoor'),
]
ENV_COLORS = {'outdoor': '#ff6d00', 'indoor': '#00e5ff'}

FPS  = 15
N_FRAMES = 360   # 24 s at 15fps
DPI  = 110


def load_tum(path):
    xyz = []
    if not os.path.exists(path):
        return np.zeros((0, 3))
    with open(path) as f:
        for l in f:
            if l.startswith('#') or not l.strip(): continue
            p = l.split()
            if len(p) >= 4:
                xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(xyz) if xyz else np.zeros((0, 3))


def load_colmap(path):
    pts = []
    if not os.path.exists(path):
        return np.zeros((0, 3))
    with open(path) as f:
        for l in f:
            if l.startswith('#') or not l.strip(): continue
            p = l.split()
            if len(p) >= 8:
                tx, ty, tz = float(p[1]), float(p[2]), float(p[3])
                qx, qy, qz, qw = float(p[4]), float(p[5]), float(p[6]), float(p[7])
                n = qw**2+qx**2+qy**2+qz**2; s = 2/n if n > 1e-10 else 0
                R = np.array([
                    [1-s*(qy**2+qz**2), s*(qx*qy-qz*qw), s*(qx*qz+qy*qw)],
                    [s*(qx*qy+qz*qw), 1-s*(qx**2+qz**2), s*(qy*qz-qx*qw)],
                    [s*(qx*qz-qy*qw), s*(qy*qz+qx*qw), 1-s*(qx**2+qy**2)],
                ])
                pts.append(-R.T @ np.array([tx, ty, tz]))
    return np.array(pts) if pts else np.zeros((0, 3))


print("Loading all sequences...")
all_data = []
for seq, env in SEQUENCES:
    orb  = load_tum(os.path.join(ORBSLAM_DIR, f'{seq}_trajectory.txt'))
    colm = load_colmap(os.path.join(COLMAP_DIR, f'{seq}_colmap_poses.txt'))
    all_data.append({'name': seq, 'env': env, 'orb': orb, 'colmap': colm})
    print(f"  {seq}: ORB={len(orb)} poses, COLMAP={len(colm)} pts")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12), facecolor='#0d1117', dpi=DPI)
fig.suptitle('COMP0222 CW2 Group 32  |  ORB-SLAM2 Monocular SLAM  |  Best Sequences',
             color='white', fontsize=13, fontweight='bold', y=0.99)

axes = []
for i in range(6):
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_facecolor('#0d1117')
    for sp in ax.spines.values(): sp.set_edgecolor('#2a2a2a')
    axes.append(ax)
plt.tight_layout(rect=[0, 0.02, 1, 0.97])

# ── Frame dims ────────────────────────────────────────────────────────────────
fig.canvas.draw()
buf = fig.canvas.buffer_rgba()
h0, w0 = np.frombuffer(buf, np.uint8).reshape(
    *fig.canvas.get_width_height()[::-1], 4).shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'avc1')
vw = cv2.VideoWriter(OUT_VIDEO, fourcc, FPS, (w0, h0))
if not vw.isOpened():
    vw = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w0, h0))

# ── Animate ───────────────────────────────────────────────────────────────────
for frame in range(N_FRAMES):
    progress = frame / max(N_FRAMES - 1, 1)

    for i, d in enumerate(all_data):
        ax = axes[i]
        ax.clear()
        ax.set_facecolor('#0d1117')
        for sp in ax.spines.values(): sp.set_edgecolor('#2a2a2a')

        orb   = d['orb']
        colm  = d['colmap']
        color = ENV_COLORS[d['env']]
        n_show = max(2, int(progress * len(orb))) if len(orb) else 0

        # COLMAP reference — faint gray background
        if len(colm) >= 2:
            ax.scatter(colm[:, 0], colm[:, 2], c='#444444', s=1.5, alpha=0.4,
                       zorder=1, label='COLMAP')

        # ORB-SLAM2 trajectory — bright, prominent
        if n_show >= 2:
            sub = orb[:n_show]
            ax.plot(sub[:, 0], sub[:, 2], color=color, lw=2.0, alpha=0.95,
                    zorder=3, label='ORB-SLAM2')
            ax.plot(sub[0, 0], sub[0, 2], 'o', color='#00ff88', ms=6, zorder=5)
            ax.plot(sub[-1, 0], sub[-1, 2], 's', color='#ff3030', ms=6, zorder=5)

        pct = 100 * n_show / max(len(orb), 1)
        ax.set_title(f"{d['name']}\n{d['env']}  |  {n_show}/{len(orb)} poses  ({pct:.0f}%)",
                     color='white', fontsize=7.5, pad=3)
        ax.tick_params(colors='#555', labelsize=5)
        ax.set_aspect('equal', adjustable='datalim')

        if i == 0 and frame == 0:
            ax.legend(fontsize=5.5, facecolor='#1a1a1a', labelcolor='white',
                      loc='upper right', markerscale=1.5)

    # Progress bar text
    fig.texts = [t for t in fig.texts if 'Processing' not in t.get_text()]
    fig.text(0.5, 0.005,
             f'Progress: {100*progress:.0f}%  |  Orange/Cyan/Green = outdoor/indoor/mixed  |  Gray dots = COLMAP reference  |  Colored line = ORB-SLAM2',
             ha='center', color='#777', fontsize=8)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.frombuffer(buf, np.uint8).reshape(h0, w0, 4)
    bgr = arr[:, :, :3][:, :, ::-1].copy()
    vw.write(bgr)

    if frame % 30 == 0:
        print(f"  Q2c frame {frame}/{N_FRAMES}")

vw.release()
plt.close(fig)
size_mb = os.path.getsize(OUT_VIDEO) / 1e6
print(f"\nQ2c video: {OUT_VIDEO}")
print(f"Size: {size_mb:.1f} MB  |  {N_FRAMES} frames @ {FPS}fps = {N_FRAMES/FPS:.0f}s")
