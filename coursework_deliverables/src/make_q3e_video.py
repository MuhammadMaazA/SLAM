#!/usr/bin/env python3
"""
Q3e: LiDAR SLAM mapping demonstration — all 9 sequences in 3×3 grid.
Shows trajectory + point cloud building simultaneously across all environments.
Output: COMP0222_CW2_GRP_1_LiDAR_SLAM.mp4
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
_REC1 = os.environ.get('SLAM_REC1', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings', 'tmp_recordings'))
_REC2 = os.environ.get('SLAM_REC2', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings2'))

SEQUENCES = {
    'Basement_1':     os.path.join(_REC1, 'Basement_1',     'lidar', 'scans.jsonl'),
    'Outdoor_1':      os.path.join(_REC1, 'Outdoor_1',      'lidar', 'scans.jsonl'),
    'Basement_2':     os.path.join(_REC1, 'Basement_2',     'lidar', 'scans.jsonl'),
    'Floor7_Hallway': os.path.join(_REC1, 'Floor7_Hallway', 'lidar', 'scans.jsonl'),
    'Washroom':       os.path.join(_REC1, 'Washroom',       'lidar', 'scans.jsonl'),
    'BikeStorage':    os.path.join(_REC2, 'BikeStorage',    'lidar', 'scans.jsonl'),
    'BikeStorage2':   os.path.join(_REC2, 'BikeStorage2',   'lidar', 'scans.jsonl'),
    'Entrance2':      os.path.join(_REC2, 'Entrance2',      'lidar', 'scans.jsonl'),
    'OnePoolStreet1': os.path.join(_REC2, 'OnePoolStreet1', 'lidar', 'scans.jsonl'),
}
OUT_VIDEO = os.environ.get('SLAM_VIDEO_Q3', os.path.join(_ROOT, '..', 'COMP0222_CW2_GRP_32_LiDAR_SLAM.mp4'))
MAX_SCANS = 300
MAX_RANGE = 4000
FPS       = 10
N_FRAMES  = 300   # 30 s at 10fps
DPI       = 100

SEQ_COLORS = [
    '#00e5ff', '#76ff03', '#ff6d00', '#ff4081', '#ffeb3b',
    '#b39ddb', '#80cbc4', '#ffcc02', '#ef5350'
]


def parse_scan(raw, max_mm=MAX_RANGE, step=2):
    pts = []
    for q, ang_deg, dist_mm in raw:
        if q < 5 or dist_mm < 50 or dist_mm > max_mm: continue
        a = np.radians(ang_deg)
        pts.append([dist_mm * np.cos(a) / 1000., dist_mm * np.sin(a) / 1000.])
    pts = np.array(pts) if pts else np.zeros((0, 2))
    return pts[::step] if step > 1 and len(pts) else pts


def icp_step(src, dst):
    from sklearn.neighbors import KDTree
    if len(src) < 5 or len(dst) < 5: return np.eye(3)
    T = np.eye(3); s = src.copy()
    tree = KDTree(dst)
    for _ in range(8):
        d, idx = tree.query(s, k=1)
        mask = d[:, 0] < 0.4
        if mask.sum() < 4: break
        m = dst[idx[mask, 0]]; sc = s[mask]
        H = (sc - sc.mean(0)).T @ (m - m.mean(0))
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        t = m.mean(0) - R @ sc.mean(0)
        s = (R @ s.T).T + t
        dT = np.eye(3); dT[:2, :2] = R; dT[:2, 2] = t
        T = dT @ T
        if np.linalg.norm(t) < 5e-4: break
    return T


def apply_T(T, pts):
    if not len(pts): return pts
    h = np.ones((len(pts), 3)); h[:, :2] = pts
    return (T @ h.T).T[:, :2]


def run_slam(name, path):
    print(f"  SLAM: {name}...")
    scans_raw = []
    with open(path) as f:
        for line in f:
            try: scans_raw.append(json.loads(line)['points'])
            except: pass
    scans_raw = scans_raw[:MAX_SCANS]

    poses = [np.array([0., 0.])]; T_world = np.eye(3)
    map_pts = []; ref = None; loops = []
    for i, raw in enumerate(scans_raw):
        pts = parse_scan(raw)
        if len(pts) < 8: continue
        if ref is None:
            ref = pts; map_pts.extend(apply_T(T_world, pts).tolist()); continue
        T = icp_step(pts, ref)
        T_world = T_world @ np.linalg.inv(T)
        pose = T_world[:2, 2].copy()
        poses.append(pose)
        map_pts.extend(apply_T(T_world, pts).tolist())
        ref = pts
        if i > len(scans_raw)//3 and np.linalg.norm(pose) < 0.8:
            loops.append(i)
    print(f"    {len(poses)} poses, {len(loops)} loop events")
    return (np.array(poses),
            np.array(map_pts) if map_pts else np.zeros((0, 2)),
            loops)


# ── Run SLAM for all sequences ─────────────────────────────────────────────────
slam_data = []
seq_names = list(SEQUENCES.keys())
for name in seq_names:
    path = SEQUENCES[name]
    if not os.path.exists(path):
        print(f"  SKIP {name}")
        continue
    poses, map_pts, loops = run_slam(name, path)
    slam_data.append({'name': name, 'poses': poses, 'map': map_pts, 'loops': loops})

n = len(slam_data)
print(f"\nLoaded {n} sequences. Rendering {N_FRAMES} frames at {FPS}fps...")

# ── Figure ─────────────────────────────────────────────────────────────────────
nrows, ncols = 3, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 12), facecolor='#080808', dpi=DPI)
fig.suptitle('COMP0222 CW2 Group 1  |  LiDAR SLAM  |  All 9 Sequences  |  ICP + Loop Closure',
             color='white', fontsize=13, fontweight='bold', y=0.99)
axes = axes.flatten()
for ax in axes:
    ax.set_facecolor('#0d1117')
    for sp in ax.spines.values(): sp.set_edgecolor('#222')
plt.tight_layout(rect=[0, 0.02, 1, 0.97])

# Get frame dimensions
fig.canvas.draw()
buf = fig.canvas.buffer_rgba()
h0, w0 = np.frombuffer(buf, np.uint8).reshape(
    *fig.canvas.get_width_height()[::-1], 4).shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'avc1')
vw = cv2.VideoWriter(OUT_VIDEO, fourcc, FPS, (w0, h0))
if not vw.isOpened():
    vw = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w0, h0))

# ── Animation loop ─────────────────────────────────────────────────────────────
for frame in range(N_FRAMES):
    progress = frame / max(N_FRAMES - 1, 1)

    for i, d in enumerate(slam_data):
        ax = axes[i]
        ax.clear()
        ax.set_facecolor('#0d1117')
        for sp in ax.spines.values(): sp.set_edgecolor('#222')

        poses   = d['poses']
        map_pts = d['map']
        loops   = d['loops']
        c       = SEQ_COLORS[i % len(SEQ_COLORS)]

        n_show = max(2, int(progress * len(poses)))
        p_show = poses[:n_show]
        m_show = map_pts[:n_show*8] if len(map_pts) else np.zeros((0, 2))

        # Map point cloud
        if len(m_show) > 0:
            ax.scatter(m_show[:, 0], m_show[:, 1], c=c, s=0.6, alpha=0.35, zorder=2)

        # Trajectory path
        if len(p_show) > 1:
            ax.plot(p_show[:, 0], p_show[:, 1], color='white', lw=1.4,
                    alpha=0.9, zorder=3)
        ax.plot(*p_show[0], 'o', color='#00ff88', ms=5, zorder=5)
        ax.plot(*p_show[-1], 's', color='#ff3030', ms=5, zorder=5)

        # Loop closure markers
        for li in loops:
            if li < n_show:
                ax.plot(*poses[li], 'y*', ms=9, zorder=6)

        pct = 100 * n_show / len(poses)
        lc  = len([li for li in loops if li < n_show])
        ax.set_title(f"{d['name']}  {pct:.0f}%  |  loops: {lc}",
                     color='white', fontsize=7.5, pad=2)
        ax.tick_params(colors='#444', labelsize=5)
        ax.set_aspect('equal', adjustable='datalim')

    # Hide unused axes
    for j in range(len(slam_data), nrows*ncols):
        axes[j].set_visible(False)

    fig.texts = [t for t in fig.texts if 'Progress' not in t.get_text()]
    fig.text(0.5, 0.005,
             f'Progress: {100*progress:.0f}%  |  White line = robot path  |  Color cloud = LiDAR map  |  Yellow ★ = loop closure',
             ha='center', color='#666', fontsize=8)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.frombuffer(buf, np.uint8).reshape(h0, w0, 4)
    bgr = arr[:, :, :3][:, :, ::-1].copy()
    vw.write(bgr)

    if frame % 30 == 0:
        print(f"  Frame {frame}/{N_FRAMES}")

vw.release()
plt.close(fig)
size_mb = os.path.getsize(OUT_VIDEO) / 1e6
print(f"\nQ3e video: {OUT_VIDEO}")
print(f"Size: {size_mb:.1f} MB  |  {N_FRAMES}f @ {FPS}fps = {N_FRAMES//FPS}s")
