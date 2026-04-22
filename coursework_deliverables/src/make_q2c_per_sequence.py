#!/usr/bin/env python3
"""
Q2c: Per-sequence side-by-side videos.
Left panel: actual camera feed from RealSense D455.
Right panel: ORB-SLAM2 trajectory building up (COLMAP = faint gray reference).
Generates one MP4 per sequence in sequence_videos/.
"""
import os
import numpy as np
import cv2

_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.abspath(os.path.join(_HERE, '..'))
_REC1       = os.environ.get('SLAM_REC1', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings', 'tmp_recordings'))
_REC2       = os.environ.get('SLAM_REC2', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings2'))
ORBSLAM_DIR = os.environ.get('SLAM_ORBSLAM_OUT', os.path.join(_ROOT, 'data', 'q2_results', 'orbslam_runs'))
COLMAP_DIR  = os.environ.get('SLAM_COLMAP_OUT',  os.path.join(_ROOT, 'data', 'q2_results', 'colmap_runs'))
OUT_DIR     = os.environ.get('SLAM_VID_OUT',     os.path.join(_ROOT, '..', 'sequence_videos'))

SEQUENCES = {
    'OnePoolStreet1': os.path.join(_REC2, 'OnePoolStreet1', 'camera'),
    'Outdoor_1':      os.path.join(_REC1, 'Outdoor_1',      'camera'),
    'Entrance2':      os.path.join(_REC2, 'Entrance2',       'camera'),
    'Basement_1':     os.path.join(_REC1, 'Basement_1',      'camera'),
    'Basement_2':     os.path.join(_REC1, 'Basement_2',      'camera'),
    'Washroom':       os.path.join(_REC1, 'Washroom',        'camera'),
    'BikeStorage':    os.path.join(_REC2, 'BikeStorage',     'camera'),
    'BikeStorage2':   os.path.join(_REC2, 'BikeStorage2',    'camera'),
    'Floor7_Hallway': os.path.join(_REC1, 'Floor7_Hallway',  'camera'),
}

FPS        = 20
FRAME_STEP = 8      # skip camera frames → fewer frames for faster export
N_FRAMES   = 300    # total output frames per video
PANEL_W, PANEL_H = 640, 480
HEADER_H = 40
CANVAS_W = PANEL_W * 2
CANVAS_H = PANEL_H + HEADER_H

BG     = (17, 17, 17)
WHITE  = (230, 230, 230)
GRAY   = (80, 80, 80)       # COLMAP reference
CYAN   = (255, 210, 0)      # ORB line (BGR: gold/yellow → visible on dark)
GREEN  = (0, 200, 60)
RED    = (0, 60, 220)


def load_rgb(cam_dir):
    paths = []
    rgb_txt = os.path.join(cam_dir, 'rgb.txt')
    if not os.path.exists(rgb_txt):
        return []
    with open(rgb_txt) as f:
        for l in f:
            if l.startswith('#') or not l.strip(): continue
            p = l.split()
            paths.append(os.path.join(cam_dir, p[1]))
    return paths


def load_tum(path):
    xyz = []
    if not os.path.exists(path): return np.zeros((0, 3))
    with open(path) as f:
        for l in f:
            if l.startswith('#') or not l.strip(): continue
            p = l.split()
            if len(p) >= 4:
                xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(xyz) if xyz else np.zeros((0, 3))


def load_colmap(path):
    pts = []
    if not os.path.exists(path): return np.zeros((0, 3))
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


def umeyama_align_xz(src_xyz, dst_xyz):
    """Scale+align src to dst using Umeyama (XZ plane only). Returns aligned src XZ."""
    src = src_xyz[:, [0, 2]];  dst = dst_xyz[:, [0, 2]]
    n   = min(len(src), len(dst))
    if n < 5:
        return src
    idx_s = np.round(np.linspace(0, len(src)-1, n)).astype(int)
    idx_d = np.round(np.linspace(0, len(dst)-1, n)).astype(int)
    s = src[idx_s];  d = dst[idx_d]
    mu_s = s.mean(0);  mu_d = d.mean(0)
    ss = s - mu_s;  ds = d - mu_d
    cov = ds.T @ ss / n
    U, sv, Vt = np.linalg.svd(cov)
    S = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    scale = (sv @ [1, np.sign(np.linalg.det(U @ Vt))]) / (ss * ss).sum() * n
    t = mu_d - scale * R @ mu_s
    return (scale * (R @ src_xyz[:, [0, 2]].T).T + t)


def world_to_px(pts_xz, bounds, w, h, pad=28):
    xmin, xmax, zmin, zmax = bounds
    rx = (xmax - xmin) or 1.0; rz = (zmax - zmin) or 1.0
    px = ((pts_xz[:, 0] - xmin) / rx * (w - 2*pad) + pad).astype(int)
    py = ((1.0 - (pts_xz[:, 1] - zmin) / rz) * (h - 2*pad) + pad).astype(int)
    return np.stack([px, py], axis=1)


def draw_traj(orb_aligned_xz, colmap_xyz, n_orb, bounds, label):
    """orb_aligned_xz: FULL ORB trajectory pre-aligned to COLMAP scale (N×2 XZ)."""
    panel = np.full((PANEL_H, PANEL_W, 3), BG, dtype=np.uint8)
    for gv in np.linspace(0, 1, 6):
        cv2.line(panel, (0, int(gv*PANEL_H)), (PANEL_W, int(gv*PANEL_H)), (30,30,30), 1)
        cv2.line(panel, (int(gv*PANEL_W), 0), (int(gv*PANEL_W), PANEL_H), (30,30,30), 1)
    # COLMAP reference — gray dots
    if len(colmap_xyz) >= 2:
        cpx = world_to_px(colmap_xyz[:, [0, 2]], bounds, PANEL_W, PANEL_H)
        for pt in cpx:
            if 0 <= pt[0] < PANEL_W and 0 <= pt[1] < PANEL_H:
                cv2.circle(panel, tuple(pt), 2, GRAY, -1)
    # ORB-SLAM2 — fixed alignment, reveal first n_orb points
    sub = orb_aligned_xz[:n_orb]
    if len(sub) >= 2:
        apx = world_to_px(sub, bounds, PANEL_W, PANEL_H)
        valid = [(0 <= p[0] < PANEL_W and 0 <= p[1] < PANEL_H) for p in apx]
        pts_v = [tuple(apx[i]) for i in range(len(apx)) if valid[i]]
        if len(pts_v) >= 2:
            cv2.polylines(panel, [np.array(pts_v, np.int32)], False, CYAN, 2)
        if pts_v:
            cv2.circle(panel, pts_v[0], 8, GREEN, -1)
            cv2.circle(panel, pts_v[-1], 8, RED, -1)
    cv2.putText(panel, 'ORB-SLAM2 vs COLMAP (XZ)', (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 1)
    cv2.putText(panel, f'ORB: {n_orb}/{len(orb_aligned_xz)} poses', (8, PANEL_H-36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1)
    cv2.putText(panel, f'COLMAP ref: {len(colmap_xyz)} pts', (8, PANEL_H-18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)
    cv2.circle(panel, (PANEL_W-80, PANEL_H-36), 5, GREEN, -1)
    cv2.putText(panel, 'Start', (PANEL_W-70, PANEL_H-32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREEN, 1)
    cv2.circle(panel, (PANEL_W-80, PANEL_H-20), 5, RED, -1)
    cv2.putText(panel, 'Now', (PANEL_W-70, PANEL_H-16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, RED, 1)
    return panel


os.makedirs(OUT_DIR, exist_ok=True)

for seq, cam_dir in SEQUENCES.items():
    print(f"\n=== {seq} ===")
    orb    = load_tum(os.path.join(ORBSLAM_DIR, f'{seq}_trajectory.txt'))
    colm   = load_colmap(os.path.join(COLMAP_DIR, f'{seq}_colmap_poses.txt'))
    frames = load_rgb(cam_dir)
    frames = frames[::FRAME_STEP]
    print(f"  {len(frames)} cam frames, {len(orb)} ORB poses, {len(colm)} COLMAP pts")

    if len(orb) == 0:
        print(f"  SKIP {seq}: no ORB-SLAM2 data")
        continue

    # Pre-align FULL ORB trajectory to COLMAP once — keeps scale/rotation fixed
    if len(colm) >= 5 and len(orb) >= 5:
        orb_aligned = umeyama_align_xz(orb, colm)   # shape (N, 2) in COLMAP metric coords
        pad_f = 0.12 * max((colm[:,0].ptp()), (colm[:,2].ptp()), 0.5)
        bounds = (colm[:,0].min()-pad_f, colm[:,0].max()+pad_f,
                  colm[:,2].min()-pad_f, colm[:,2].max()+pad_f)
    else:
        # No COLMAP — use ORB's own XZ, own bounds
        orb_aligned = orb[:, [0, 2]]
        pad_f = 0.12 * max((orb[:,0].ptp()), (orb[:,2].ptp()), 0.5)
        bounds = (orb[:,0].min()-pad_f, orb[:,0].max()+pad_f,
                  orb[:,2].min()-pad_f, orb[:,2].max()+pad_f)

    out_path = os.path.join(OUT_DIR, f'Q2c_{seq}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    vw = cv2.VideoWriter(out_path, fourcc, FPS, (CANVAS_W, CANVAS_H))
    if not vw.isOpened():
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (CANVAS_W, CANVAS_H))

    for f_idx in range(N_FRAMES):
        progress = f_idx / max(N_FRAMES - 1, 1)
        n_orb    = max(1, int(progress * len(orb)))
        cam_idx  = min(int(progress * len(frames)), len(frames)-1) if frames else -1

        # Left: camera
        if cam_idx >= 0 and frames:
            img = cv2.imread(frames[cam_idx])
            left = cv2.resize(img, (PANEL_W, PANEL_H)) if img is not None else \
                   np.full((PANEL_H, PANEL_W, 3), BG, dtype=np.uint8)
        else:
            left = np.full((PANEL_H, PANEL_W, 3), BG, dtype=np.uint8)
            cv2.putText(left, 'No camera data', (160, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, GRAY, 2)

        # Right: trajectory (pass pre-aligned XZ array)
        right = draw_traj(orb_aligned, colm, n_orb, bounds, seq)

        # Header
        body   = np.hstack([left, right])
        header = np.full((HEADER_H, CANVAS_W, 3), (28, 28, 28), np.uint8)
        cv2.putText(header,
                    f'COMP0222 CW2 Group 1  |  ORB-SLAM2 Visual SLAM  |  {seq}',
                    (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.54, WHITE, 1)
        pct = 100 * n_orb / len(orb)
        cv2.putText(header,
                    f'Poses: {n_orb}/{len(orb)} ({pct:.0f}%)  |  Sensor: RealSense D455  |  ATE RMSE vs COLMAP shown right',
                    (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (150, 200, 150), 1)

        vw.write(np.vstack([header, body]))

    vw.release()
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Saved: Q2c_{seq}.mp4  ({size_kb:.0f} KB)")

print("\nAll Q2c per-sequence videos done.")
