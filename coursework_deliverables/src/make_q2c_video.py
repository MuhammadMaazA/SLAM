#!/usr/bin/env python3
"""
Q2c: Visual SLAM demonstration video.
Each column = one sequence.
Top panel : actual RealSense camera frame (synced by timestamp).
Bottom panel : ORB-SLAM2 trajectory building up, COLMAP as faint reference.
Output: COMP0222_CW2_GRP_32_Visual_SLAM.mp4
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.abspath(os.path.join(_HERE, '..'))
_REC1       = os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data',
                            'tmp_recordings', 'tmp_recordings')
ORBSLAM_DIR = os.path.join(_ROOT, 'data', 'q2_results', 'orbslam_runs')
COLMAP_DIR  = os.path.join(_ROOT, 'data', 'q2_results', 'colmap_runs')
OUT_VIDEO   = os.path.join(_ROOT, '..', 'COMP0222_CW2_GRP_32_Visual_SLAM.mp4')

SEQUENCES = [
    ('Basement_1',     'indoor'),
    ('Floor7_Hallway', 'indoor'),
    ('Outdoor_1',      'outdoor'),
]
ENV_COLORS = {'outdoor': '#ff9500', 'indoor': '#00cfff'}

FPS      = 15
N_FRAMES = 450   # 30 s
DPI      = 100


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_rgb_index(seq):
    """Load rgb.txt → sorted (timestamp, abs_path) list."""
    base = os.path.join(_REC1, seq, 'camera')
    txt  = os.path.join(base, 'rgb.txt')
    entries = []
    with open(txt) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            ts, rel = line.split()
            entries.append((float(ts), os.path.join(base, rel)))
    return sorted(entries, key=lambda x: x[0])


def load_tum(path):
    """Load TUM trajectory → (timestamps_array, xyz_array)."""
    ts, xyz = [], []
    if not os.path.exists(path):
        return np.array([]), np.zeros((0, 3))
    with open(path) as f:
        for l in f:
            if l.startswith('#') or not l.strip():
                continue
            p = l.split()
            if len(p) >= 4:
                ts.append(float(p[0]))
                xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(ts), np.array(xyz) if xyz else np.zeros((0, 3))


def load_colmap(path):
    pts = []
    if not os.path.exists(path):
        return np.zeros((0, 3))
    with open(path) as f:
        for l in f:
            if l.startswith('#') or not l.strip():
                continue
            p = l.split()
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
                pts.append(-R.T @ np.array([tx, ty, tz]))
    return np.array(pts) if pts else np.zeros((0, 3))


def frame_at_progress(rgb_index, progress):
    """Return the BGR image closest to the given normalised progress [0,1]."""
    idx = int(np.clip(progress * (len(rgb_index) - 1), 0, len(rgb_index) - 1))
    path = rgb_index[idx][1]
    img  = cv2.imread(path)
    if img is None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
    return img  # BGR


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading sequences...")
all_data = []
for seq, env in SEQUENCES:
    rgb_idx = load_rgb_index(seq)
    ts, xyz = load_tum(os.path.join(ORBSLAM_DIR, f'{seq}_trajectory.txt'))
    colmap  = load_colmap(os.path.join(COLMAP_DIR, f'{seq}_colmap_poses.txt'))
    all_data.append({
        'name': seq, 'env': env,
        'rgb': rgb_idx, 'ts': ts, 'xyz': xyz, 'colmap': colmap,
    })
    print(f"  {seq}: {len(rgb_idx)} frames  ORB={len(xyz)} poses  COLMAP={len(colmap)}")


# ── Figure layout ─────────────────────────────────────────────────────────────
# 2 rows × 3 cols: top = camera, bottom = trajectory
fig = plt.figure(figsize=(18, 10), facecolor='#0d1117', dpi=DPI)
fig.suptitle(
    'COMP0222 CW2 Group 32  |  Visual SLAM  |  Camera feed + ORB-SLAM2 trajectory',
    color='white', fontsize=12, fontweight='bold', y=0.995)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.08, wspace=0.05,
                       top=0.96, bottom=0.04, left=0.03, right=0.97,
                       height_ratios=[2.2, 1.0])

cam_axes  = [fig.add_subplot(gs[0, i]) for i in range(3)]
traj_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

for ax in cam_axes + traj_axes:
    ax.set_facecolor('#0d1117')
    for sp in ax.spines.values():
        sp.set_edgecolor('#2a2a2a')

fig.canvas.draw()
buf = fig.canvas.buffer_rgba()
h0, w0 = np.frombuffer(buf, np.uint8).reshape(
    *fig.canvas.get_width_height()[::-1], 4).shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'avc1')
vw = cv2.VideoWriter(OUT_VIDEO, fourcc, FPS, (w0, h0))
if not vw.isOpened():
    vw = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w0, h0))


# ── Animation ─────────────────────────────────────────────────────────────────
for frame in range(N_FRAMES):
    progress = frame / max(N_FRAMES - 1, 1)

    for i, d in enumerate(all_data):
        color = ENV_COLORS[d['env']]

        # ── Top: camera frame ────────────────────────────────────────────────
        ax_cam = cam_axes[i]
        ax_cam.clear()
        ax_cam.set_facecolor('#0d1117')
        for sp in ax_cam.spines.values():
            sp.set_edgecolor('#333')
        bgr = frame_at_progress(d['rgb'], progress)
        rgb_img = bgr[:, :, ::-1]
        ax_cam.imshow(rgb_img, aspect='auto')
        ax_cam.set_xticks([]); ax_cam.set_yticks([])
        env_label = '[Indoor]' if d['env'] == 'indoor' else '[Outdoor]'
        ax_cam.set_title(
            f"{d['name']}  |  {env_label}",
            color='white', fontsize=9, fontweight='bold', pad=3)

        # ── Bottom: trajectory ───────────────────────────────────────────────
        ax_tr = traj_axes[i]
        ax_tr.clear()
        ax_tr.set_facecolor('#0d1117')
        for sp in ax_tr.spines.values():
            sp.set_edgecolor('#333')

        xyz  = d['xyz']
        colm = d['colmap']
        n_show = max(2, int(progress * len(xyz))) if len(xyz) else 0

        # ORB-SLAM2 trajectory — draw first so its extent sets axis limits
        if n_show >= 2:
            sub = xyz[:n_show]
            ax_tr.plot(sub[:, 0], sub[:, 2],
                       color=color, lw=2.5, alpha=0.95, zorder=3)
            ax_tr.plot(sub[0, 0],  sub[0, 2],  'o', color='#00ff88', ms=9, zorder=5)
            ax_tr.plot(sub[-1, 0], sub[-1, 2], 's', color='#ff3333', ms=9, zorder=5)
            # Fix axis limits to ORB trajectory with 15% padding
            xpad = max((sub[:, 0].max() - sub[:, 0].min()) * 0.15, 0.3)
            zpad = max((sub[:, 2].max() - sub[:, 2].min()) * 0.15, 0.3)
            ax_tr.set_xlim(sub[:, 0].min() - xpad, sub[:, 0].max() + xpad)
            ax_tr.set_ylim(sub[:, 2].min() - zpad, sub[:, 2].max() + zpad)

        # COLMAP reference — faint, added after limits are fixed
        if len(colm) >= 5:
            ax_tr.scatter(colm[:, 0], colm[:, 2],
                          c='#555555', s=3, alpha=0.5, zorder=1)

        pct = 100 * n_show / max(len(xyz), 1)
        ax_tr.set_title(
            f'ORB-SLAM2  {n_show}/{len(xyz)} poses ({pct:.0f}%)',
            color='#aaa', fontsize=8, pad=3)
        ax_tr.tick_params(colors='#555', labelsize=6)
        ax_tr.set_xlabel('X (m)', color='#666', fontsize=7)
        ax_tr.set_ylabel('Z (m)', color='#666', fontsize=7)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.frombuffer(buf, np.uint8).reshape(h0, w0, 4)
    bgr_frame = arr[:, :, :3][:, :, ::-1].copy()
    vw.write(bgr_frame)

    if frame % 45 == 0:
        print(f"  frame {frame}/{N_FRAMES}  ({100*progress:.0f}%)")

vw.release()
plt.close(fig)
size_mb = os.path.getsize(OUT_VIDEO) / 1e6
print(f"\nDone: {OUT_VIDEO}")
print(f"  {N_FRAMES} frames @ {FPS}fps = {N_FRAMES//FPS}s  |  {size_mb:.1f} MB")
