#!/usr/bin/env python3
"""
Q2c live display: ORB-SLAM2 tracking playback via cv2.imshow() for screen recording.
cv2 uses GTK/SDL and renders reliably to XWayland on WSLg display :0.
Layout: camera feed (left 640×480) | trajectory XZ (right 640×480) | header bar.
"""
import os, sys, time
import numpy as np
import cv2

_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.abspath(os.path.join(_HERE, '..'))
_REC2     = os.environ.get('SLAM_REC2', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings2'))
SEQ       = os.environ.get('SLAM_SEQ', 'OnePoolStreet1')
CAM_DIR   = os.environ.get('SLAM_CAM_DIR',   os.path.join(_REC2, SEQ, 'camera'))
TRAJ_FILE = os.environ.get('SLAM_TRAJ_FILE', os.path.join(_ROOT, 'data', 'q2_results', 'orbslam_runs', f'{SEQ}_trajectory.txt'))
COLMAP    = os.environ.get('SLAM_COLMAP_FILE', os.path.join(_ROOT, 'data', 'q2_results', 'colmap_runs', f'{SEQ}_colmap_poses.txt'))

FRAME_STEP = 6
FPS_TARGET = 20
PANEL_W, PANEL_H = 640, 480
WIN_NAME  = 'COMP0222 CW2 - ORB-SLAM2 Monocular SLAM'

BG        = (17, 17, 17)        # dark background (BGR)
CYAN      = (255, 210, 0)       # ORB trajectory (BGR: gold-ish → actually cyan)
ORANGE    = (0, 140, 255)       # COLMAP reference (BGR)
RED       = (0, 80, 220)        # current pose dot
WHITE     = (230, 230, 230)
GRAY      = (100, 100, 100)


def load_rgb(cam_dir):
    paths = []
    with open(os.path.join(cam_dir, 'rgb.txt')) as f:
        for l in f:
            if l.startswith('#') or not l.strip(): continue
            p = l.split()
            paths.append(os.path.join(cam_dir, p[1]))
    return paths


def load_tum(path):
    xyz = []
    with open(path) as f:
        for l in f:
            if l.startswith('#') or not l.strip(): continue
            p = l.split()
            if len(p) >= 4:
                xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(xyz) if xyz else np.zeros((0, 3))


def load_colmap(path):
    pts = []
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
    """Scale+align src to dst in XZ plane. Returns aligned src as (N,2) XZ array."""
    src = src_xyz[:, [0, 2]]; dst = dst_xyz[:, [0, 2]]
    n   = min(len(src), len(dst))
    if n < 5: return src
    idx_s = np.round(np.linspace(0, len(src)-1, n)).astype(int)
    idx_d = np.round(np.linspace(0, len(dst)-1, n)).astype(int)
    s = src[idx_s]; d = dst[idx_d]
    mu_s = s.mean(0); mu_d = d.mean(0)
    ss = s - mu_s; ds = d - mu_d
    cov = ds.T @ ss / n
    U, sv, Vt = np.linalg.svd(cov)
    S = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    scale = (sv @ [1, np.sign(np.linalg.det(U @ Vt))]) / (ss * ss).sum() * n
    t = mu_d - scale * R @ mu_s
    return (scale * (R @ src_xyz[:, [0, 2]].T).T + t)


def world_to_px(pts_xz, bounds, panel_w, panel_h, pad=30):
    """Map XZ world coords to pixel coords in panel."""
    xmin, xmax, zmin, zmax = bounds
    rx = (xmax - xmin) or 1.0
    rz = (zmax - zmin) or 1.0
    px = ((pts_xz[:, 0] - xmin) / rx * (panel_w - 2*pad) + pad).astype(int)
    py = ((1.0 - (pts_xz[:, 1] - zmin) / rz) * (panel_h - 2*pad) + pad).astype(int)
    return np.stack([px, py], axis=1)


def draw_traj_panel(orb_aligned_xz, colmap_xyz, n_orb, bounds):
    """orb_aligned_xz: full ORB trajectory pre-aligned to COLMAP scale (N×2 XZ)."""
    panel = np.full((PANEL_H, PANEL_W, 3), BG, dtype=np.uint8)

    # Grid lines
    for gv in np.linspace(0, 1, 5):
        y = int(gv * PANEL_H)
        x = int(gv * PANEL_W)
        cv2.line(panel, (0, y), (PANEL_W, y), (35, 35, 35), 1)
        cv2.line(panel, (x, 0), (x, PANEL_H), (35, 35, 35), 1)

    # COLMAP reference dots
    if len(colmap_xyz) >= 2:
        cpx = world_to_px(colmap_xyz[:, [0, 2]], bounds, PANEL_W, PANEL_H)
        for pt in cpx:
            if 0 <= pt[0] < PANEL_W and 0 <= pt[1] < PANEL_H:
                cv2.circle(panel, tuple(pt), 2, ORANGE, -1)

    # ORB-SLAM2 trajectory line (slice pre-aligned array)
    sub = orb_aligned_xz[:n_orb]
    if len(sub) >= 2:
        opx = world_to_px(sub, bounds, PANEL_W, PANEL_H)
        valid = [(0 <= p[0] < PANEL_W and 0 <= p[1] < PANEL_H) for p in opx]
        pts_list = [tuple(opx[i]) for i in range(len(opx)) if valid[i]]
        if len(pts_list) >= 2:
            cv2.polylines(panel, [np.array(pts_list, dtype=np.int32)],
                          False, (0, 220, 255), 2)
        # Start marker
        if valid[0]:
            cv2.circle(panel, pts_list[0], 7, (0, 200, 0), -1)
        # Current pose
        if valid[-1]:
            cv2.circle(panel, pts_list[-1], 7, (0, 60, 220), -1)

    # Legend
    cv2.putText(panel, 'COLMAP ref', (8, PANEL_H-40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ORANGE, 1)
    cv2.putText(panel, 'ORB-SLAM2', (8, PANEL_H-22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
    cv2.putText(panel, 'Trajectory (XZ plane)', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    return panel


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
frames_all = load_rgb(CAM_DIR)
orb_xyz    = load_tum(TRAJ_FILE)
colmap_xyz = load_colmap(COLMAP)
frames     = frames_all[::FRAME_STEP]
print(f"  {len(frames)} camera frames, {len(orb_xyz)} ORB poses, {len(colmap_xyz)} COLMAP pts")

# Align ORB to COLMAP scale once (fixed transform for all frames)
if len(colmap_xyz) >= 5 and len(orb_xyz) >= 5:
    orb_aligned = umeyama_align_xz(orb_xyz, colmap_xyz)  # (N, 2) XZ
    pad_f = 0.12 * max(colmap_xyz[:, 0].ptp(), colmap_xyz[:, 2].ptp(), 0.5)
    BOUNDS = (colmap_xyz[:, 0].min()-pad_f, colmap_xyz[:, 0].max()+pad_f,
              colmap_xyz[:, 2].min()-pad_f, colmap_xyz[:, 2].max()+pad_f)
else:
    orb_aligned = orb_xyz[:, [0, 2]]
    pad_f = 0.12 * max(orb_xyz[:, 0].ptp(), orb_xyz[:, 2].ptp(), 0.5)
    BOUNDS = (orb_xyz[:, 0].min()-pad_f, orb_xyz[:, 0].max()+pad_f,
              orb_xyz[:, 2].min()-pad_f, orb_xyz[:, 2].max()+pad_f)

# ── Window setup ──────────────────────────────────────────────────────────────
HEADER_H = 44
CANVAS_W  = PANEL_W * 2
CANVAS_H  = PANEL_H + HEADER_H

cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, CANVAS_W, CANVAS_H)

# ── Playback loop ─────────────────────────────────────────────────────────────
n_total = len(frames)
delay_ms = max(1, int(1000 / FPS_TARGET))

for i, img_path in enumerate(frames):
    t0 = time.time()
    progress = i / max(n_total - 1, 1)
    n_orb    = max(1, int(progress * len(orb_xyz)))

    # Left panel: camera frame
    img = cv2.imread(img_path)
    if img is not None:
        left = cv2.resize(img, (PANEL_W, PANEL_H))
    else:
        left = np.full((PANEL_H, PANEL_W, 3), BG, dtype=np.uint8)

    # Right panel: trajectory
    right = draw_traj_panel(orb_aligned, colmap_xyz, n_orb, BOUNDS)

    # Combine panels
    body = np.hstack([left, right])

    # Header bar
    header = np.full((HEADER_H, CANVAS_W, 3), (30, 30, 30), dtype=np.uint8)
    cv2.putText(header,
                f'COMP0222 CW2 Group 1  |  ORB-SLAM2 Monocular SLAM  |  {SEQ}',
                (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    pct = 100 * n_orb / len(orb_xyz)
    cv2.putText(header,
                f'Frame {i*FRAME_STEP}/{len(frames_all)}  |  Poses: {n_orb}/{len(orb_xyz)}  |  {pct:.0f}%  |  COLMAP: {len(colmap_xyz)} pts  |  ATE RMSE: 3.19 m',
                (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 200, 160), 1)

    canvas = np.vstack([header, body])
    cv2.imshow(WIN_NAME, canvas)

    elapsed_ms = int((time.time() - t0) * 1000)
    wait = max(1, delay_ms - elapsed_ms)
    key = cv2.waitKey(wait)
    if key == 27:  # ESC to quit early
        break

# Hold final frame
cv2.waitKey(3000)
cv2.destroyAllWindows()
print("Q2c display complete")
