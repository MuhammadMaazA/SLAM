#!/usr/bin/env python3
"""
Q3e: LiDAR SLAM mapping demonstration with synced real video.
Top row: actual RGB camera feed for each sequence.
Bottom row: LiDAR SLAM trajectory + map build-up for each sequence.
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
)

_REC1_DEFAULT = os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings')
_REC1_NESTED = os.path.join(_REC1_DEFAULT, 'tmp_recordings')
_REC1_ENV = os.environ.get('SLAM_REC1')
_REC1_CANDIDATES = [p for p in (_REC1_ENV, _REC1_DEFAULT, _REC1_NESTED) if p]
_REC1 = _REC1_CANDIDATES[0]
for _cand in _REC1_CANDIDATES:
    _probe = os.path.join(_cand, 'Basement_1', 'lidar', 'scans.jsonl')
    if os.path.exists(_probe):
        _REC1 = _cand
        break
_REC2 = os.environ.get('SLAM_REC2', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings2'))

SEQUENCES = {
    'Basement_1':     os.path.join(_REC1, 'Basement_1',     'lidar', 'scans.jsonl'),
    'Floor7_Hallway': os.path.join(_REC1, 'Floor7_Hallway', 'lidar', 'scans.jsonl'),
    'Outdoor_1':      os.path.join(_REC1, 'Outdoor_1',      'lidar', 'scans.jsonl'),
}
OUT_VIDEO = os.environ.get('SLAM_VIDEO_Q3', os.path.join(_ROOT, '..', 'COMP0222_CW2_GRP_32_LiDAR_SLAM.mp4'))
MAX_SCANS = None   # use all scans — capping to 1500 missed the second loop on longer sequences
FPS       = 15
N_FRAMES  = 450   # 30 s at 15fps — long enough to show both loops completing
DPI       = 100

SEQ_COLORS = [
    '#00e5ff', '#76ff03', '#ff6d00', '#ff4081', '#ffeb3b',
    '#b39ddb', '#80cbc4', '#ffcc02', '#ef5350'
]



# Per-sequence ICP tuning.  Outdoor open space has sparse geometry so the
# default 0.5m correspondence threshold and 20-KF local map cause ICP to
# freeze — widen both so it can still find matches with distant features.
_SEQ_PARAMS = {
    'Outdoor_1': dict(
        max_range_mm=12000.0, voxel_m=0.05,
        corr_thresh=2.0,      # wider search radius for sparse outdoor geometry
        local_map=60,         # more reference KFs to match against
        max_step_dist=1.5,    # outdoor walking can cover more per scan
        icp_iter=20,
    ),
}
_DEFAULT_PARAMS = dict(max_range_mm=12000.0, voxel_m=0.05)


def run_slam(name, path):
    """Run ICP SLAM and return trajectory + accumulated point cloud."""
    print(f"  SLAM: {name}...")
    scans   = q3_load_scans(path, max_scans=MAX_SCANS)
    params  = _SEQ_PARAMS.get(name, _DEFAULT_PARAMS)
    result  = q3_run_slam(scans, **params)
    traj    = result['trajectory']
    poses   = traj[:, :2]
    kf_pts  = result['map_pts']
    map_pts = np.vstack(kf_pts) if kf_pts else np.zeros((0, 2))
    print(f"    {len(poses)} poses, {len(result['kf_poses'])} keyframes")
    return poses, map_pts


def get_rgb_info(seq_name, scan_path):
    """Return RGB directory, sorted frame list, and count for a sequence."""
    seq_dir = os.path.dirname(os.path.dirname(scan_path))  # .../<sequence>/
    rgb_candidates = [
        os.path.join(seq_dir, 'camera', 'rgb'),
        os.path.join(_REC1_DEFAULT, seq_name, 'camera', 'rgb'),
        os.path.join(_REC1_NESTED, seq_name, 'camera', 'rgb'),
    ]
    for rgb_dir in rgb_candidates:
        if os.path.isdir(rgb_dir):
            frames = sorted(f for f in os.listdir(rgb_dir) if f.lower().endswith('.jpg'))
            if frames:
                return rgb_dir, frames, len(frames)
    return rgb_candidates[0], [], 0


# ── Run SLAM for all sequences ─────────────────────────────────────────────────
slam_data = []
seq_names = list(SEQUENCES.keys())
for name in seq_names:
    path = SEQUENCES[name]
    if not os.path.exists(path):
        print(f"  SKIP {name}")
        continue
    poses, map_pts = run_slam(name, path)
    rgb_dir, rgb_frames, n_rgb = get_rgb_info(name, path)
    if n_rgb == 0:
        print(f"    WARN: no RGB frames found at {rgb_dir}")
    else:
        print(f"    RGB frames: {n_rgb}")
    slam_data.append({
        'name': name,
        'poses': poses,
        'map': map_pts,
        'rgb_dir': rgb_dir,
        'rgb_frames': rgb_frames,
        'n_rgb': n_rgb,
    })

n = len(slam_data)
print(f"\nLoaded {n} sequences. Rendering {N_FRAMES} frames at {FPS}fps...")

# ── Figure ─────────────────────────────────────────────────────────────────────
nrows, ncols = 2, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10), facecolor='#080808',
                         dpi=DPI, squeeze=False)
fig.suptitle('COMP0222 CW2 Group 32  |  LiDAR SLAM + Synced RGB Video',
             color='white', fontsize=13, fontweight='bold', y=0.995)
for ax in axes.flatten():
    ax.set_facecolor('#0d1117')
    for sp in ax.spines.values(): sp.set_edgecolor('#222')
plt.tight_layout(rect=[0, 0.02, 1, 0.965])

# Get frame dimensions
fig.canvas.draw()
buf = fig.canvas.buffer_rgba()
h0, w0 = np.frombuffer(buf, np.uint8).reshape(
    *fig.canvas.get_width_height()[::-1], 4).shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'avc1')
vw = cv2.VideoWriter(OUT_VIDEO, fourcc, FPS, (w0, h0))
if not vw.isOpened():
    vw = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w0, h0))

# ── Pre-compute per-sequence axis limits from full trajectory + map ────────────
# Using stable limits prevents the view from jumping each frame, which makes
# the map-building animation much cleaner to watch.
seq_limits = []
for d in slam_data:
    all_xy = np.vstack([d['poses'], d['map']]) if len(d['map']) else d['poses']
    lo, hi = all_xy.min(0), all_xy.max(0)
    span   = max((hi - lo).max() * 1.15, 2.0)   # add 15% padding
    cx, cy = (lo + hi) / 2
    seq_limits.append((cx - span/2, cx + span/2, cy - span/2, cy + span/2))

# ── Animation loop ─────────────────────────────────────────────────────────────
for frame in range(N_FRAMES):
    progress = frame / max(N_FRAMES - 1, 1)

    for i, d in enumerate(slam_data):
        ax_vid = axes[0, i]
        ax_map = axes[1, i]
        ax_vid.clear()
        ax_map.clear()
        for ax in (ax_vid, ax_map):
            ax.set_facecolor('#0d1117')
            for sp in ax.spines.values():
                sp.set_edgecolor('#222')

        # Top row: actual recorded RGB video
        env = 'Indoor (large)' if d['name'] == 'Floor7_Hallway' else \
              'Outdoor' if d['name'] == 'Outdoor_1' else 'Indoor'
        if d['n_rgb'] > 0:
            rgb_idx = int(progress * (d['n_rgb'] - 1))
            rgb_name = d['rgb_frames'][rgb_idx]
            rgb_path = os.path.join(d['rgb_dir'], rgb_name)
            img_bgr = cv2.imread(rgb_path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                ax_vid.imshow(img_rgb)
            else:
                ax_vid.text(0.5, 0.5, 'RGB frame read failed', ha='center',
                            va='center', color='white', fontsize=9,
                            transform=ax_vid.transAxes)
        else:
            ax_vid.text(0.5, 0.5, 'RGB video unavailable', ha='center',
                        va='center', color='white', fontsize=9,
                        transform=ax_vid.transAxes)
        ax_vid.set_title(f"{d['name']}  [{env}]  RGB video",
                         color='white', fontsize=8, pad=3)
        ax_vid.set_xticks([])
        ax_vid.set_yticks([])

        poses   = d['poses']
        map_pts = d['map']
        c       = SEQ_COLORS[i % len(SEQ_COLORS)]

        n_show = max(2, int(progress * len(poses)))
        p_show = poses[:n_show]
        n_map  = int(progress * len(map_pts)) if len(map_pts) else 0
        m_show = map_pts[:n_map] if n_map > 0 else np.zeros((0, 2))

        # Map point cloud
        if len(m_show) > 0:
            ax_map.scatter(m_show[:, 0], m_show[:, 1], c=c, s=0.6,
                           alpha=0.4, zorder=2)

        # Trajectory path — white line, thicker for clarity
        if len(p_show) > 1:
            ax_map.plot(p_show[:, 0], p_show[:, 1], color='white', lw=1.6,
                        alpha=0.95, zorder=3)
        ax_map.plot(*p_show[0],  'o', color='#00ff88', ms=6, zorder=5, label='Start')
        ax_map.plot(*p_show[-1], 's', color='#ff3030', ms=6, zorder=5, label='Current')

        pct = 100 * n_show / len(poses)
        ax_map.set_title(f"{d['name']}  [{env}]  LiDAR SLAM {pct:.0f}%",
                         color='white', fontsize=8, pad=3)
        ax_map.tick_params(colors='#444', labelsize=5)

        # Use stable limits so the view doesn't jump
        xl0, xl1, yl0, yl1 = seq_limits[i]
        ax_map.set_xlim(xl0, xl1)
        ax_map.set_ylim(yl0, yl1)
        ax_map.set_aspect('equal')

    # Hide unused axes
    for j in range(len(slam_data), ncols):
        axes[0, j].set_visible(False)
        axes[1, j].set_visible(False)

    fig.texts = [t for t in fig.texts if 'Progress' not in t.get_text()]
    fig.text(0.5, 0.005,
             f'Progress: {100*progress:.0f}%  |  Top: recorded RGB  |  Bottom: white=trajectory, colour=LiDAR map',
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
