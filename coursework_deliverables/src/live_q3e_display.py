#!/usr/bin/env python3
"""
Q3e live display: LiDAR SLAM mapping via matplotlib Agg → numpy → cv2.imshow().
matplotlib renders to memory buffer; cv2.imshow() shows the window on X11 display.
ffmpeg x11grab can then capture the cv2 window reliably.
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')   # render to memory, NOT to display — cv2 handles the window
import matplotlib.pyplot as plt
import cv2

_REC1 = os.environ.get('SLAM_REC1', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings', 'tmp_recordings'))
_REC2 = os.environ.get('SLAM_REC2', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings2'))
SEQS = {
    'Basement_1':     os.path.join(_REC1, 'Basement_1',     'lidar', 'scans.jsonl'),
    'Outdoor_1':      os.path.join(_REC1, 'Outdoor_1',      'lidar', 'scans.jsonl'),
    'OnePoolStreet1': os.path.join(_REC2, 'OnePoolStreet1', 'lidar', 'scans.jsonl'),
}
MAX_SCANS = 250
PAUSE_S   = 0.10
MAX_RANGE = 4000
WIN_NAME  = 'COMP0222 CW2 - LiDAR SLAM Real-Time Mapping'

# ── ICP helpers ────────────────────────────────────────────────────────────────
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

def occ_grid(pts, cell=0.06, sz=180):
    g = np.zeros((sz, sz))
    cx = cy = sz // 2
    for x, y in pts:
        gx, gy = int(x/cell)+cx, int(y/cell)+cy
        if 0 <= gx < sz and 0 <= gy < sz:
            g[gy, gx] = min(g[gy, gx]+0.25, 1.0)
    return g

# ── Load and run SLAM ─────────────────────────────────────────────────────────
all_scans = {}
for name, path in SEQS.items():
    if not os.path.exists(path): continue
    rows = []
    with open(path) as f:
        for line in f:
            try: rows.append(json.loads(line)['points'])
            except: pass
    all_scans[name] = rows[:MAX_SCANS]
    print(f"Loaded {name}: {len(rows[:MAX_SCANS])} scans")

if not all_scans:
    print("No data"); sys.exit(1)

slam_results = {}
for name, scans_raw in all_scans.items():
    poses, map_pts = [np.array([0., 0.])], []
    T_world = np.eye(3); ref = None; loops = []
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
        if i > len(scans_raw)//3 and np.linalg.norm(pose) < 0.7:
            loops.append(i)
    slam_results[name] = (np.array(poses),
                          np.array(map_pts) if map_pts else np.zeros((0, 2)),
                          loops)
    print(f"  {name}: {len(poses)} poses, {len(loops)} loop events")

# ── Matplotlib figure (Agg — rendered to buffer) ──────────────────────────────
n_seq  = len(slam_results)
colors = ['#00e5ff', '#76ff03', '#ff6d00']
fig    = plt.figure(figsize=(6*n_seq, 9), facecolor='#0d1117', dpi=100)
fig.suptitle('COMP0222 CW2 Group 1  |  LiDAR SLAM  |  ICP + Loop Closure + Factor Graph',
             color='white', fontsize=11, fontweight='bold', y=0.99)

axes_traj, axes_occ = [], []
for ci, (name, (poses, map_pts, loops)) in enumerate(slam_results.items()):
    ax_t = fig.add_subplot(2, n_seq, ci+1)
    ax_o = fig.add_subplot(2, n_seq, n_seq+ci+1)
    ax_t.set_facecolor('#111827'); ax_o.set_facecolor('#111827')
    for sp in ax_t.spines.values(): sp.set_edgecolor('#333')
    for sp in ax_o.spines.values(): sp.set_edgecolor('#333')
    ax_t.set_title(name, color='white', fontsize=9)
    ax_o.set_title(f'{name} — Occupancy', color='white', fontsize=9)
    ax_t.tick_params(colors='#666', labelsize=6)
    ax_o.axis('off')
    axes_traj.append(ax_t); axes_occ.append(ax_o)

plt.tight_layout(rect=[0, 0, 1, 0.97])

# ── cv2 window ────────────────────────────────────────────────────────────────
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

# Get figure pixel size
fig.canvas.draw()
buf = fig.canvas.buffer_rgba()
arr = np.frombuffer(buf, np.uint8).reshape(*fig.canvas.get_width_height()[::-1], 4)
h0, w0 = arr.shape[:2]
cv2.resizeWindow(WIN_NAME, w0, h0)

# ── Animate ───────────────────────────────────────────────────────────────────
names_list   = list(slam_results.keys())
total_frames = max(len(slam_results[n][0]) for n in names_list)

for frame in range(total_frames):
    progress = frame / max(total_frames-1, 1)

    for ci, name in enumerate(names_list):
        poses, map_pts, loops = slam_results[name]
        ax_t = axes_traj[ci]; ax_o = axes_occ[ci]
        c = colors[ci % len(colors)]

        n_show = max(2, int(progress * len(poses)))
        p_show = poses[:n_show]
        m_show = map_pts[:n_show*8] if len(map_pts) else np.zeros((0, 2))

        ax_t.clear()
        ax_t.set_facecolor('#111827')
        for sp in ax_t.spines.values(): sp.set_edgecolor('#333')

        if len(m_show):
            ax_t.scatter(m_show[:, 0], m_show[:, 1], c=c, s=0.4, alpha=0.25)
        if len(p_show) > 1:
            ax_t.plot(p_show[:, 0], p_show[:, 1], color='white', lw=1.5, alpha=0.9)
        ax_t.plot(*p_show[0], 'go', ms=7, label='Start')
        ax_t.plot(*p_show[-1], 'rs', ms=7, label='Now')
        for li in loops:
            if li < n_show: ax_t.plot(*poses[li], 'y*', ms=12)
        ax_t.set_title(f'{name}  |  {n_show}/{len(poses)} poses', color='white', fontsize=8)
        ax_t.tick_params(colors='#666', labelsize=6)
        ax_t.set_aspect('equal', adjustable='datalim')
        ax_t.legend(fontsize=6, facecolor='#1a1a1a', labelcolor='white')

        ax_o.clear()
        ax_o.set_facecolor('#111827')
        grid = occ_grid(m_show.tolist(), sz=140)
        ax_o.imshow(1-grid, cmap='gray', origin='lower', vmin=0, vmax=1,
                    interpolation='bilinear')
        ax_o.set_title(f'{name} — Occupancy Grid', color='white', fontsize=8)
        ax_o.axis('off')

    # Render matplotlib → numpy → BGR → cv2.imshow
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.frombuffer(buf, np.uint8).reshape(*fig.canvas.get_width_height()[::-1], 4)
    bgr = arr[:, :, :3][:, :, ::-1].copy()   # RGBA→RGB→BGR
    cv2.imshow(WIN_NAME, bgr)

    key = cv2.waitKey(max(1, int(PAUSE_S * 1000)))
    if key == 27: break

cv2.waitKey(3000)
cv2.destroyAllWindows()
print("Q3e display complete")
