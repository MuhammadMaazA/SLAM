#!/usr/bin/env python3
"""
Q2c video — research-paper style.

Layout per sequence (15 s each):
  ┌─────────────────────┬─────────────────────┐
  │  Raw camera feed    │  Trajectory builds   │
  │  (RGB frames)       │  up in sync          │
  └─────────────────────┴─────────────────────┘

Then a final 10 s side-by-side static comparison panel (COLMAP vs ORB-SLAM2)
matching the EVO plot style.

Total ≈ 40 s.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import cv2

_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.abspath(os.path.join(_HERE, '..'))
COLMAP_DIR = os.path.join(_ROOT, 'data', 'q2_results', 'colmap_runs')
ORB_DIR    = os.path.join(_ROOT, 'data', 'q2_results', 'orbslam_runs')
OUT        = os.path.join(_ROOT, 'data', 'q2_results', 'q2c_video.mp4')

REC_ROOT   = os.environ.get('SLAM_REC1',
             os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings'))

W, H = 1280, 480   # final frame size
FPS  = 20
SEQ_SECS   = 15    # seconds per sequence section
COMP_SECS  = 10    # seconds for final comparison panel

CMAP_TIME  = plt.get_cmap('plasma')

PCD_SECS   = 10    # seconds for 3D pointcloud rotation per sequence


# ── helpers ──────────────────────────────────────────────────────────────────

def load_tum(path):
    pts = []
    if not os.path.exists(path): return np.zeros((0,3))
    with open(path) as f:
        for l in f:
            if l.startswith('#') or not l.strip(): continue
            p = l.split()
            if len(p) >= 4:
                pts.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(pts) if pts else np.zeros((0,3))


def load_colmap_poses(path):
    pts = []
    with open(path) as f:
        for l in f:
            if l.startswith('#') or not l.strip(): continue
            p = l.split()
            if len(p) >= 8:
                tx,ty,tz = float(p[1]),float(p[2]),float(p[3])
                qx,qy,qz,qw = float(p[4]),float(p[5]),float(p[6]),float(p[7])
                n = qw**2+qx**2+qy**2+qz**2; s = 2/n if n>1e-10 else 0
                R = np.array([
                    [1-s*(qy**2+qz**2), s*(qx*qy-qz*qw), s*(qx*qz+qy*qw)],
                    [s*(qx*qy+qz*qw),   1-s*(qx**2+qz**2), s*(qy*qz-qx*qw)],
                    [s*(qx*qz-qy*qw),   s*(qy*qz+qx*qw), 1-s*(qx**2+qy**2)],
                ])
                pts.append(-R.T @ np.array([tx,ty,tz]))
    return np.array(pts) if pts else np.zeros((0,3))


def load_points3d(sparse_dir, max_pts=6000):
    """Load COLMAP points3D.txt — returns xyz (N,3) and rgb (N,3) float 0-1."""
    path = os.path.join(sparse_dir, 'points3D.txt')
    xyz, rgb = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            if len(p) >= 7:
                xyz.append([float(p[1]), float(p[2]), float(p[3])])
                rgb.append([int(p[4])/255, int(p[5])/255, int(p[6])/255])
    xyz = np.array(xyz);  rgb = np.array(rgb)
    # remove outliers (beyond 3 std)
    mask = np.all(np.abs(xyz - xyz.mean(0)) < 3*xyz.std(0), axis=1)
    xyz, rgb = xyz[mask], rgb[mask]
    # downsample for render speed
    if len(xyz) > max_pts:
        idx = np.random.choice(len(xyz), max_pts, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb


def make_pointcloud_frame(pts3d, rgb3d, colmap_traj, orb_traj,
                           azim, elev=25, title='', env='indoor',
                           w=W, h=H):
    """One frame of the rotating 3D pointcloud. Returns BGR numpy (h,w,3)."""
    dpi = 100
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi, facecolor='#050505')
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#050505')

    # scatter — actual RGB colours, visible size
    if len(pts3d):
        ax.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2],
                   c=rgb3d, s=3, alpha=0.8, linewidths=0, zorder=1,
                   depthshade=True)

    # COLMAP camera path — white
    if len(colmap_traj) > 1:
        ax.plot(colmap_traj[:,0], colmap_traj[:,1], colmap_traj[:,2],
                color='white', lw=1.5, alpha=0.9, zorder=3, label='COLMAP path')
        ax.scatter(*colmap_traj[0],  color='#2ecc71', s=60, zorder=5,
                   depthshade=False)
        ax.scatter(*colmap_traj[-1], color='#e74c3c', s=60, zorder=5,
                   depthshade=False)

    # ORB-SLAM2 path — plasma coloured
    if len(orb_traj) > 1:
        t = np.linspace(0, 1, len(orb_traj))
        cmap = plt.cm.plasma
        for i in range(0, len(orb_traj)-1, max(1, len(orb_traj)//200)):
            ax.plot(orb_traj[i:i+2,0], orb_traj[i:i+2,1], orb_traj[i:i+2,2],
                    color=cmap(t[i]), lw=1.8, alpha=0.95, zorder=4)

    ax.view_init(elev=elev, azim=azim)

    # clean up panes
    ax.set_axis_off()
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('none')
    ax.grid(False)

    color  = '#00e5ff' if env == 'indoor' else '#ff6d00'
    n_pts  = len(pts3d)
    ax.set_title(f'COLMAP 3D sparse map — {title}  ({n_pts:,} points)\n'
                 f'White = COLMAP path  |  Plasma = ORB-SLAM2 path',
                 color=color, fontsize=10, fontweight='bold', pad=8)

    plt.tight_layout(pad=0.2)
    fig.canvas.draw()
    ww, hh = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(hh, ww, 3)
    plt.close(fig)
    return cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_RGB2BGR)


def read_rgb_frame(rgb_dir, idx):
    path = os.path.join(rgb_dir, f'frame_{idx:06d}.jpg')
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    return img  # BGR


def traj_2d(pts):
    """Return X-Z plane projection (top-down)."""
    if len(pts) == 0: return np.zeros((0,2))
    return pts[:, [0, 2]]


def draw_traj_panel(orb_pts, colmap_pts, n_shown, title, env, w=640, h=480):
    """
    Draw trajectory panel: COLMAP ghost (grey) + ORB-SLAM2 build-up (plasma).
    Returns BGR numpy array (h, w, 3).
    """
    dpi = 100
    fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#0d0d0d')

    orb_2d    = traj_2d(orb_pts)
    colmap_2d = traj_2d(colmap_pts)

    # determine axis limits from full trajectories
    all_pts = []
    if len(orb_2d):    all_pts.append(orb_2d)
    if len(colmap_2d): all_pts.append(colmap_2d)
    if all_pts:
        combined = np.vstack(all_pts)
        cx, cy = combined.mean(0)
        span   = max(combined.ptp(0).max(), 0.5) * 0.65
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)

    # COLMAP ghost
    if len(colmap_2d) > 1:
        ax.plot(colmap_2d[:,0], colmap_2d[:,1],
                color='#555555', lw=0.8, alpha=0.6, zorder=1, label='COLMAP')
        ax.plot(*colmap_2d[0],  'o', color='#888888', ms=4, zorder=2)
        ax.plot(*colmap_2d[-1], 's', color='#888888', ms=4, zorder=2)

    # ORB-SLAM2 build-up
    shown = min(n_shown, len(orb_2d))
    if shown > 1:
        seg_pts = orb_2d[:shown]
        t = np.linspace(0, 1, shown)
        segs = np.stack([seg_pts[:-1], seg_pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap=CMAP_TIME, norm=plt.Normalize(0,1),
                              linewidth=2.0, alpha=0.95, zorder=3)
        lc.set_array(t[:-1])
        ax.add_collection(lc)
        ax.plot(*seg_pts[-1], 'o', color='white', ms=5, zorder=5)

    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')

    env_str = 'Indoor' if env == 'indoor' else 'Outdoor'
    color   = '#00e5ff' if env == 'indoor' else '#ff6d00'
    ax.set_title(f'{env_str} — {title}',
                 color=color, fontsize=11, fontweight='bold', pad=6)

    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color='#555555', lw=1.5, label='COLMAP (reference)'),
        Line2D([0],[0], color=CMAP_TIME(0.7), lw=2,   label='ORB-SLAM2 (plasma)'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=7,
              facecolor='#1a1a1a', edgecolor='#444', labelcolor='white')

    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    ww, hh = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(hh, ww, 3)
    plt.close(fig)
    return cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_RGB2BGR)


def make_title_card(text, sub='', duration_frames=FPS*2):
    """Return list of identical BGR frames with a title card."""
    frames = []
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    fig.patch.set_facecolor('#050505')
    ax.set_facecolor('#050505')
    ax.axis('off')
    ax.text(0.5, 0.55, text, transform=ax.transAxes,
            ha='center', va='center', fontsize=22, fontweight='bold', color='white')
    if sub:
        ax.text(0.5, 0.35, sub, transform=ax.transAxes,
                ha='center', va='center', fontsize=13, color='#aaaaaa')
    fig.canvas.draw()
    ww, hh = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(hh, ww, 3)
    plt.close(fig)
    card = cv2.cvtColor(cv2.resize(img, (W, H)), cv2.COLOR_RGB2BGR)
    return [card.copy() for _ in range(duration_frames)]


def make_comparison_panel(seqs_data, duration_frames=FPS*COMP_SECS):
    """Static side-by-side COLMAP vs ORB-SLAM2 comparison (matches EVO plot style)."""
    n = len(seqs_data)
    fig, axes = plt.subplots(n, 2, figsize=(W/100, H*n/(100*(16/9))), dpi=100)
    fig.patch.set_facecolor('#0d0d0d')
    if n == 1: axes = [axes]

    col_titles = ['COLMAP trajectory', 'ORB-SLAM2 trajectory']
    for row, (seq, env, orb_pts, colmap_pts, ate_str) in enumerate(seqs_data):
        color = '#00e5ff' if env == 'indoor' else '#ff6d00'
        orb_2d    = traj_2d(orb_pts)
        colmap_2d = traj_2d(colmap_pts)

        for col, (pts, method) in enumerate([(colmap_2d,'COLMAP'), (orb_2d,'ORB-SLAM2')]):
            ax = axes[row][col]
            ax.set_facecolor('#0d0d0d')

            # ghost the full colmap on both panels
            if len(colmap_2d) > 1:
                ax.plot(colmap_2d[:,0], colmap_2d[:,1],
                        color='#333333', lw=0.7, alpha=0.5, zorder=1)

            if len(pts) > 1:
                t = np.linspace(0,1,len(pts))
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                lc   = LineCollection(segs, cmap=CMAP_TIME,
                                      norm=plt.Normalize(0,1),
                                      linewidth=2.5, alpha=0.95, zorder=3)
                lc.set_array(t[:-1])
                ax.add_collection(lc)
                ax.autoscale_view()
                ax.plot(*pts[0],  'D', color='#2ecc71', ms=7, zorder=5,
                        markeredgecolor='white', markeredgewidth=0.5)
                ax.plot(*pts[-1], 'X', color='#e74c3c', ms=8, zorder=5,
                        markeredgecolor='white', markeredgewidth=0.5)

            env_str = 'Indoor' if env=='indoor' else 'Outdoor'
            title = f'{env_str} — {seq}\n{method}'
            if col==1 and ate_str:
                title += f'\n{ate_str}'
            ax.set_title(title, color=color, fontsize=8, fontweight='bold', pad=4)
            ax.set_aspect('equal', adjustable='datalim')
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.suptitle('COLMAP vs ORB-SLAM2 Trajectory Comparison',
                 color='white', fontsize=12, fontweight='bold', y=1.0)
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    ww, hh = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(hh, ww, 3)
    plt.close(fig)
    card = cv2.cvtColor(cv2.resize(img, (W, H)), cv2.COLOR_RGB2BGR)
    return [card.copy() for _ in range(duration_frames)]


# ── main ─────────────────────────────────────────────────────────────────────

SEQUENCES = [
    {
        'name': 'Basement_1',
        'env':  'indoor',
        'rgb_dir': os.path.join(REC_ROOT, 'Basement_1', 'camera', 'rgb'),
        'ate_str': 'Trans RMSE: 0.046 a.u.  |  Rot: 3.8°',
    },
    {
        'name': 'Outdoor_1',
        'env':  'outdoor',
        'rgb_dir': os.path.join(REC_ROOT, 'Outdoor_1', 'camera', 'rgb'),
        'ate_str': 'Trans RMSE: 1.488 a.u.  |  Rot: 169.2°  (COLMAP fy inflation)',
    },
]


def main():
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

    seqs_for_comparison = []

    for seq_cfg in SEQUENCES:
        seq  = seq_cfg['name']
        env  = seq_cfg['env']
        rgb_dir = seq_cfg['rgb_dir']
        ate_str = seq_cfg['ate_str']

        print(f"\n=== {seq} ===")

        # load trajectories
        cal_path = os.path.join(ORB_DIR, f'{seq}_trajectory_colmap_intrinsics.txt')
        fac_path = os.path.join(ORB_DIR, f'{seq}_trajectory.txt')
        orb_path = cal_path if os.path.exists(cal_path) else fac_path
        orb_pts    = load_tum(orb_path)
        colmap_pts = load_colmap_poses(
            os.path.join(COLMAP_DIR, f'{seq}_colmap_poses.txt'))
        print(f"  ORB: {len(orb_pts)} poses, COLMAP: {len(colmap_pts)} poses")

        # count available frames
        n_frames_avail = len([f for f in os.listdir(rgb_dir)
                              if f.endswith('.jpg')]) if os.path.isdir(rgb_dir) else 0
        print(f"  RGB frames available: {n_frames_avail}")

        # title card
        env_str  = 'Indoor' if env == 'indoor' else 'Outdoor'
        sub_str  = ('Marshgate Lift D room - closed loop'
                    if env == 'indoor' else
                    'Marshgate back entrance - cycle parking courtyard')
        for frm in make_title_card(f'{env_str}: {seq}', sub_str, FPS*2):
            writer.write(frm)

        # sequence section: raw video left | trajectory right
        n_vid_frames = SEQ_SECS * FPS
        for i in range(n_vid_frames):
            progress = i / max(n_vid_frames - 1, 1)

            # ── left: raw camera feed ──────────────────────────────────
            if n_frames_avail > 0:
                frame_idx = int(progress * (n_frames_avail - 1))
                raw = read_rgb_frame(rgb_dir, frame_idx)
                if raw is None:
                    raw = np.zeros((H, W//2, 3), dtype=np.uint8)
                else:
                    raw = cv2.resize(raw, (W//2, H))
                # overlay label
                cv2.putText(raw, 'Acquired sequence', (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(raw, f'Frame {frame_idx}/{n_frames_avail}', (10, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
            else:
                raw = np.zeros((H, W//2, 3), dtype=np.uint8)
                cv2.putText(raw, 'Raw video unavailable', (20, H//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120,120,120), 1)

            # ── right: trajectory build-up ────────────────────────────
            n_shown = max(2, int(progress * len(orb_pts)))
            traj_panel = draw_traj_panel(orb_pts, colmap_pts, n_shown,
                                         seq, env, w=W//2, h=H)

            # ── combine ───────────────────────────────────────────────
            frame = np.hstack([raw, traj_panel])

            # divider line
            frame[:, W//2-1:W//2+1] = [80, 80, 80]

            writer.write(frame)

            if i % (FPS*3) == 0:
                print(f"  {i}/{n_vid_frames} frames")

        # ── 3D pointcloud rotation section ───────────────────────────
        print(f"  Loading 3D pointcloud...")
        pts3d, rgb3d = load_points3d(os.path.join(COLMAP_DIR, f'{seq}_sparse'))
        print(f"  {len(pts3d)} points after filtering")

        # align ORB trajectory to COLMAP coordinate frame for 3D display
        orb_aligned = orb_pts.copy()
        if len(orb_aligned) > 5 and len(colmap_pts) > 5:
            oc = orb_aligned.mean(0);  cc = colmap_pts.mean(0)
            os_ = np.std(np.linalg.norm(orb_aligned - oc, axis=1)) + 1e-9
            cs_ = np.std(np.linalg.norm(colmap_pts  - cc, axis=1)) + 1e-9
            orb_aligned = (orb_aligned - oc) * (cs_ / os_) + cc

        for frm in make_title_card('3D Point Cloud', f'{seq} — COLMAP sparse reconstruction', FPS):
            writer.write(frm)

        n_pcd_frames = PCD_SECS * FPS
        for i in range(n_pcd_frames):
            azim = 30 + (i / n_pcd_frames) * 360
            frm  = make_pointcloud_frame(pts3d, rgb3d, colmap_pts, orb_aligned,
                                          azim=azim, title=seq, env=env)
            writer.write(frm)
            if i % (FPS*3) == 0:
                print(f"  pcd {i}/{n_pcd_frames}")

        seqs_for_comparison.append((seq, env, orb_pts, colmap_pts, ate_str))

    # ── final comparison panel ────────────────────────────────────────────
    print("\nRendering comparison panel...")
    for frm in make_title_card('COLMAP vs ORB-SLAM2', 'EVO trajectory comparison', FPS*2):
        writer.write(frm)
    for frm in make_comparison_panel(seqs_for_comparison):
        writer.write(frm)

    writer.release()

    # re-encode with ffmpeg
    final = OUT.replace('.mp4', '_enc.mp4')
    ret = os.system(f'ffmpeg -y -i "{OUT}" -c:v libx264 -crf 18 -pix_fmt yuv420p "{final}" 2>/dev/null')
    if ret == 0 and os.path.exists(final):
        os.replace(final, OUT)

    print(f"\nDone: {OUT}")
    print(f"Size: {os.path.getsize(OUT)//1024} KB")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        name = sys.argv[1]
        SEQUENCES[:] = [s for s in SEQUENCES if s['name'] == name]
        OUT = OUT.replace('q2c_video.mp4', f'q2c_{name.lower()}.mp4')
    main()
