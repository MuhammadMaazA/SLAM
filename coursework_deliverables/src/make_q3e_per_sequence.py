#!/usr/bin/env python3
"""
Q3e: Per-sequence side-by-side LiDAR SLAM videos.
Left panel: current raw LiDAR scan (what the sensor sees right now).
Right panel: accumulated map + trajectory building up over time.
Generates one MP4 per sequence in sequence_videos/.
"""
import json, os
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
_REC1 = os.environ.get('SLAM_REC1', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings', 'tmp_recordings'))
_REC2 = os.environ.get('SLAM_REC2', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings2'))

LIDAR_PATHS = {
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
OUT_DIR = os.environ.get('SLAM_VID_OUT', os.path.join(_ROOT, '..', 'sequence_videos'))
MAX_SCANS = 300
MAX_RANGE = 4000
FPS = 12
PANEL_W, PANEL_H = 600, 500
HEADER_H = 40
CANVAS_W = PANEL_W * 2
CANVAS_H = PANEL_H + HEADER_H

BG     = (17, 17, 17)
WHITE  = (230, 230, 230)
GRAY   = (80, 80, 80)
CYAN   = (255, 210, 0)      # map points (BGR)
GREEN  = (0, 210, 60)
RED    = (0, 60, 220)
YELLOW = (0, 220, 220)


# ── ICP helpers ─────────────────────────────────────────────────────────────
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


# ── Draw current scan (left panel) ──────────────────────────────────────────
def draw_scan_panel(scan_pts):
    """Raw LiDAR scan in robot frame — the sensor's eye view."""
    panel = np.full((PANEL_H, PANEL_W, 3), BG, dtype=np.uint8)
    cx, cy = PANEL_W // 2, PANEL_H // 2
    max_r  = MAX_RANGE / 1000.0   # metres

    # Radar rings
    for r_frac in [0.25, 0.5, 0.75, 1.0]:
        r_px = int(r_frac * min(cx, cy) * 0.92)
        cv2.circle(panel, (cx, cy), r_px, (35, 35, 35), 1)
        dist_m = r_frac * max_r
        cv2.putText(panel, f'{dist_m:.1f}m', (cx + r_px + 3, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, GRAY, 1)

    # Cross-hairs
    cv2.line(panel, (cx, 0), (cx, PANEL_H), (30, 30, 30), 1)
    cv2.line(panel, (0, cy), (PANEL_W, cy), (30, 30, 30), 1)

    # Scan points
    if len(scan_pts):
        scale = min(cx, cy) * 0.92 / max_r
        for x, y in scan_pts:
            px = int(cx + x * scale)
            py = int(cy - y * scale)
            if 0 <= px < PANEL_W and 0 <= py < PANEL_H:
                cv2.circle(panel, (px, py), 2, (0, 220, 255), -1)

    # Robot marker
    cv2.circle(panel, (cx, cy), 6, GREEN, -1)
    cv2.putText(panel, 'Current LiDAR Scan', (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 1)
    cv2.putText(panel, f'{len(scan_pts)} pts', (8, PANEL_H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)
    return panel


# ── Draw accumulated map (right panel) ─────────────────────────────────────
def world_to_px(pts, bounds, w, h, pad=25):
    xmin, xmax, ymin, ymax = bounds
    rx = (xmax - xmin) or 1.0; ry = (ymax - ymin) or 1.0
    px = ((pts[:, 0] - xmin) / rx * (w - 2*pad) + pad).astype(int)
    py = ((1.0 - (pts[:, 1] - ymin) / ry) * (h - 2*pad) + pad).astype(int)
    return np.stack([px, py], axis=1)


def draw_map_panel(map_pts, poses, loops, n_show, bounds):
    panel = np.full((PANEL_H, PANEL_W, 3), BG, dtype=np.uint8)
    for gv in np.linspace(0, 1, 6):
        cv2.line(panel, (0, int(gv*PANEL_H)), (PANEL_W, int(gv*PANEL_H)), (28,28,28), 1)
        cv2.line(panel, (int(gv*PANEL_W), 0), (int(gv*PANEL_W), PANEL_H), (28,28,28), 1)

    if len(map_pts) > 0:
        mpx = world_to_px(map_pts, bounds, PANEL_W, PANEL_H)
        for pt in mpx[::3]:   # thin out for speed
            if 0 <= pt[0] < PANEL_W and 0 <= pt[1] < PANEL_H:
                cv2.circle(panel, tuple(pt), 1, CYAN, -1)

    p_show = poses[:n_show]
    if len(p_show) >= 2:
        ppx = world_to_px(p_show, bounds, PANEL_W, PANEL_H)
        valid = [(0 <= p[0] < PANEL_W and 0 <= p[1] < PANEL_H) for p in ppx]
        pts_v = [tuple(ppx[i]) for i in range(len(ppx)) if valid[i]]
        if len(pts_v) >= 2:
            cv2.polylines(panel, [np.array(pts_v, np.int32)], False, WHITE, 2)
        if pts_v:
            cv2.circle(panel, pts_v[0], 7, GREEN, -1)
            cv2.circle(panel, pts_v[-1], 7, RED, -1)

    for li in loops:
        if li < n_show:
            ppx = world_to_px(poses[li:li+1], bounds, PANEL_W, PANEL_H)
            cv2.drawMarker(panel, tuple(ppx[0]), YELLOW, cv2.MARKER_STAR, 14, 2)

    cv2.putText(panel, 'Accumulated Map + Trajectory', (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 1)
    lc = len([l for l in loops if l < n_show])
    cv2.putText(panel, f'{n_show} poses  |  loops: {lc}', (8, PANEL_H-24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)
    cv2.putText(panel, 'Yellow * = loop closure', (8, PANEL_H-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, YELLOW, 1)
    return panel


os.makedirs(OUT_DIR, exist_ok=True)

for seq, lidar_path in LIDAR_PATHS.items():
    print(f"\n=== {seq} ===")
    if not os.path.exists(lidar_path):
        print(f"  SKIP: no lidar data"); continue

    # Load scans
    scans_raw = []
    with open(lidar_path) as f:
        for line in f:
            try: scans_raw.append(json.loads(line)['points'])
            except: pass
    scans_raw = scans_raw[:MAX_SCANS]
    print(f"  {len(scans_raw)} scans")

    # Run SLAM
    poses  = [np.array([0., 0.])]; T_world = np.eye(3)
    map_pts_full = []; ref = None; loops = []
    scan_history = []   # raw scan at each step (robot frame)

    for i, raw in enumerate(scans_raw):
        pts = parse_scan(raw)
        scan_history.append(pts.copy() if len(pts) else np.zeros((0, 2)))
        if len(pts) < 8: continue
        if ref is None:
            ref = pts; map_pts_full.extend(apply_T(T_world, pts).tolist()); continue
        T = icp_step(pts, ref)
        T_world = T_world @ np.linalg.inv(T)
        pose = T_world[:2, 2].copy()
        poses.append(pose)
        map_pts_full.extend(apply_T(T_world, pts).tolist())
        ref = pts
        if i > len(scans_raw)//3 and np.linalg.norm(pose) < 0.8:
            loops.append(len(poses)-1)

    poses    = np.array(poses)
    map_all  = np.array(map_pts_full) if map_pts_full else np.zeros((0, 2))
    print(f"  {len(poses)} poses, {len(loops)} loop events, {len(map_all)} map pts")

    # Stable bounds for map panel
    if len(map_all) > 0:
        pad_f = 0.1 * max(map_all[:, 0].ptp(), map_all[:, 1].ptp(), 0.5)
        bounds = (map_all[:, 0].min()-pad_f, map_all[:, 0].max()+pad_f,
                  map_all[:, 1].min()-pad_f, map_all[:, 1].max()+pad_f)
    else:
        bounds = (-5, 5, -5, 5)

    out_path = os.path.join(OUT_DIR, f'Q3e_{seq}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    vw = cv2.VideoWriter(out_path, fourcc, FPS, (CANVAS_W, CANVAS_H))
    if not vw.isOpened():
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (CANVAS_W, CANVAS_H))

    N = len(scan_history)
    for f_idx in range(N):
        progress = f_idx / max(N - 1, 1)
        n_show   = max(1, int(progress * len(poses)))
        scan_idx = f_idx
        m_show   = map_all[:n_show*8] if len(map_all) else np.zeros((0, 2))

        left  = draw_scan_panel(scan_history[scan_idx])
        right = draw_map_panel(m_show, poses, loops, n_show, bounds)

        body   = np.hstack([left, right])
        header = np.full((HEADER_H, CANVAS_W, 3), (28, 28, 28), np.uint8)
        cv2.putText(header,
                    f'COMP0222 CW2 Group 1  |  LiDAR SLAM  |  {seq}',
                    (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.54, WHITE, 1)
        pct = 100 * n_show / len(poses)
        cv2.putText(header,
                    f'Scan {f_idx+1}/{N}  |  Poses: {n_show}/{len(poses)} ({pct:.0f}%)  |  Sensor: Intel RealSense L515',
                    (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (150, 200, 150), 1)

        vw.write(np.vstack([header, body]))

    vw.release()
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Saved: Q3e_{seq}.mp4  ({size_kb:.0f} KB)")

print("\nAll Q3e per-sequence videos done.")
