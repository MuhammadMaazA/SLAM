#!/usr/bin/env python3
"""
Simple Q2c video: raw camera feed (left) | 3D pointcloud building up (right).
One video per sequence.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.abspath(os.path.join(_HERE, '..'))
COLMAP_DIR = os.path.join(_ROOT, 'data', 'q2_results', 'colmap_runs')
REC_ROOT   = os.environ.get('SLAM_REC1',
             os.path.join(os.path.expanduser('~'), 'SLAM',
                          'extracted_data', 'tmp_recordings'))

FPS   = 20
W, H  = 1280, 480   # 640 each side

SEQUENCES = {
    'Basement_1': {
        'env': 'Indoor',
        'label': 'Basement_1',
        'location': 'Lift D Room (Marshgate LGF)',
        'color': '#00e5ff',
    },
    'Outdoor_1': {
        'env': 'Outdoor',
        'label': 'Outdoor_1',
        'location': 'Rear courtyard cycle parking (Marshgate)',
        'color': '#ff6d00',
    },
}


def load_points3d(sparse_dir, max_pts=5000):
    path = os.path.join(sparse_dir, 'points3D.txt')
    xyz, rgb = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            p = line.split()
            if len(p) >= 7:
                xyz.append([float(p[1]), float(p[2]), float(p[3])])
                rgb.append([int(p[4])/255, int(p[5])/255, int(p[6])/255])
    xyz = np.array(xyz); rgb = np.array(rgb)
    # remove outliers
    mask = np.all(np.abs(xyz - xyz.mean(0)) < 3*xyz.std(0), axis=1)
    xyz, rgb = xyz[mask], rgb[mask]
    if len(xyz) > max_pts:
        idx = np.random.choice(len(xyz), max_pts, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb


def render_pointcloud(pts3d, rgb3d, n_shown, azim, elev=25, w=640, h=480):
    dpi = 100
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi, facecolor='#080808')
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#080808')

    shown = min(n_shown, len(pts3d))
    if shown > 0:
        ax.scatter(pts3d[:shown, 0], pts3d[:shown, 1], pts3d[:shown, 2],
                   c=rgb3d[:shown], s=4, alpha=0.85,
                   linewidths=0, depthshade=True, zorder=2)

    # fixed axis limits so scene doesn't jump
    c = pts3d.mean(0)
    r = np.percentile(np.linalg.norm(pts3d - c, axis=1), 90) * 1.1
    ax.set_xlim(c[0]-r, c[0]+r)
    ax.set_ylim(c[1]-r, c[1]+r)
    ax.set_zlim(c[2]-r, c[2]+r)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor('none')
    ax.grid(False)

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    ww, hh = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(hh, ww, 3)
    plt.close(fig)
    return cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_RGB2BGR)


def make_video(seq_name):
    cfg    = SEQUENCES[seq_name]
    env    = cfg['env']
    label  = cfg['label']
    location = cfg['location']
    color  = tuple(int(cfg['color'].lstrip('#')[i:i+2], 16) for i in (4,2,0))  # BGR

    rgb_dir = os.path.join(REC_ROOT, seq_name, 'camera', 'rgb')
    frames  = sorted(f for f in os.listdir(rgb_dir) if f.endswith('.jpg'))
    n_raw   = len(frames)
    print(f"{seq_name}: {n_raw} raw frames")

    pts3d, rgb3d = load_points3d(os.path.join(COLMAP_DIR, f'{seq_name}_sparse'))
    n_pts = len(pts3d)
    print(f"{seq_name}: {n_pts} 3D points")

    # shuffle points so they appear scattered (not all from one wall first)
    order = np.random.permutation(n_pts)
    pts3d = pts3d[order]; rgb3d = rgb3d[order]

    # total video = length of raw footage at FPS (max 30s)
    n_frames = min(n_raw, FPS * 30)
    out_path = os.path.join(_ROOT, 'data', 'q2_results',
                            f'q2c_{seq_name.lower()}_simple.mp4')
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             FPS, (W, H))

    for i in range(n_frames):
        progress = i / max(n_frames - 1, 1)

        # ── left: raw frame ───────────────────────────────────────────
        raw_idx  = int(progress * (n_raw - 1))
        raw_path = os.path.join(rgb_dir, frames[raw_idx])
        raw = cv2.imread(raw_path)
        raw = cv2.resize(raw, (W//2, H))

        # Title overlay. Keep this ASCII-only: OpenCV Hershey fonts render
        # some Unicode punctuation as ??? on common Linux builds.
        cv2.putText(raw, f'{env} | {label}', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(raw, f'Location: {location}', (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
        cv2.putText(raw, f'Frame {raw_idx+1}/{n_raw} | raw RGB feed',
                    (10, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (230, 230, 230), 1, cv2.LINE_AA)

        # ── right: pointcloud building up ─────────────────────────────
        n_shown = max(1, int(progress * n_pts))
        azim    = 30 + progress * 270   # sweep 270° over the video
        pcd = render_pointcloud(pts3d, rgb3d, n_shown, azim=azim)

        # title overlay on pointcloud
        pcd_label = f'COLMAP sparse map  ({n_shown:,} / {n_pts:,} points)'
        cv2.putText(pcd, pcd_label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(pcd, f'{env} | {label}', (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
        cv2.putText(pcd, 'Rotating point cloud: scene structure / scale cue',
                    (10, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)

        # ── combine with divider ──────────────────────────────────────
        frame = np.hstack([raw, pcd])
        frame[:, W//2-1:W//2+1] = [60, 60, 60]
        writer.write(frame)

        if i % (FPS * 5) == 0:
            print(f"  {i}/{n_frames}  ({n_shown} pts shown)")

    writer.release()

    # re-encode
    enc = out_path.replace('.mp4', '_enc.mp4')
    if os.system(f'ffmpeg -y -i "{out_path}" -c:v libx264 -crf 20 '
                 f'-pix_fmt yuv420p "{enc}" 2>/dev/null') == 0:
        os.replace(enc, out_path)

    print(f"Saved: {out_path}  ({os.path.getsize(out_path)//1024} KB)")


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else None
    for seq in (SEQUENCES if target is None else [target]):
        make_video(seq)
