#!/usr/bin/env python3
"""
Q3e: LiDAR SLAM mapping demonstration — all three Q3 sequences in a 1x3 grid.
Shows trajectory + point cloud building simultaneously across environments.
Uses the same SLAM pipeline as the Q3 analysis (run_slam from
q3_lidar_slam_complete) so the visualisation matches the reported numbers.
Output: COMP0222_CW2_GRP_32_LiDAR_SLAM.mp4
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, _HERE)
from q3_lidar_slam_complete import (      # noqa: E402
    run_slam as q3_run_slam,
    load_scans as q3_load_scans,
    detect_loop_closures as q3_detect_loops,
)

_REC1 = os.environ.get('SLAM_REC1', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings', 'tmp_recordings'))
_REC2 = os.environ.get('SLAM_REC2', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings2'))

SEQUENCES = {
    'Basement_1':     os.path.join(_REC1, 'Basement_1',     'lidar', 'scans.jsonl'),
    'Floor7_Hallway': os.path.join(_REC1, 'Floor7_Hallway', 'lidar', 'scans.jsonl'),
    'Outdoor_1':      os.path.join(_REC1, 'Outdoor_1',      'lidar', 'scans.jsonl'),
}
OUT_VIDEO = os.environ.get('SLAM_VIDEO_Q3', os.path.join(_ROOT, '..', 'COMP0222_CW2_GRP_32_LiDAR_SLAM.mp4'))
MAX_SCANS = 1500
FPS       = 15
N_FRAMES  = 360   # 24 s at 15fps
DPI       = 100

SEQ_COLORS = [
    '#00e5ff', '#76ff03', '#ff6d00', '#ff4081', '#ffeb3b',
    '#b39ddb', '#80cbc4', '#ffcc02', '#ef5350'
]


def run_slam(name, path):
    """Thin wrapper around the Q3 pipeline that returns trajectory + global
    map points + loop-closure indices in the representation the video loop
    expects."""
    print(f"  SLAM: {name}...")
    scans = q3_load_scans(path, max_scans=MAX_SCANS)
    result = q3_run_slam(scans, max_range_mm=4000.0)
    traj   = result['trajectory']              # (N, 3) [x, y, theta]
    poses  = traj[:, :2]                       # (N, 2)
    # Flatten per-keyframe global points into a single (M, 2) array
    kf_pts = result['map_pts']
    map_pts = (np.vstack(kf_pts) if kf_pts else np.zeros((0, 2)))
    # Loop closures from the full pipeline (keyframe indices -> scan indices)
    closures = q3_detect_loops(result['kf_poses'], kf_pts)
    kf_n = max(1, len(result['kf_poses']))
    scan_per_kf = max(1, len(traj) // kf_n)
    loops = [min(lc[1] * scan_per_kf, len(traj) - 1) for lc in closures]
    print(f"    {len(poses)} poses, {len(loops)} loop events, "
          f"{len(result['kf_poses'])} keyframes")
    return poses, map_pts, loops


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
nrows, ncols = 1, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 7), facecolor='#080808', dpi=DPI)
fig.suptitle('COMP0222 CW2 Group 32  |  LiDAR SLAM  |  ICP Odometry + Loop Closure + Factor Graph',
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
        # Reveal map points in proportion to playback (not n_show*8 heuristic).
        n_map = int(progress * len(map_pts)) if len(map_pts) else 0
        m_show = map_pts[:max(0, n_map)] if len(map_pts) else np.zeros((0, 2))

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
