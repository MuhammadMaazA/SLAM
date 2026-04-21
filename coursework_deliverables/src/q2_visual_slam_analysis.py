#!/usr/bin/env python3
"""
Q2: Visual SLAM with Custom Collected Data — All 9 Sequences
- Q2a: ORB-SLAM2 trajectories for all sequences (indoor + outdoor)
- Q2b: COLMAP vs ORB-SLAM2 comparison (for sequences where COLMAP succeeded)
- Q2c: Trajectory statistics and analysis for all 9 sequences
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import os

BASE = "/home/mmaaz/SLAM/coursework_deliverables/data"
ORB_DIR  = f"{BASE}/q2_results/orbslam_runs"
COLMAP_DIR = f"{BASE}/q2_results/colmap_runs"
OUT_DIR  = f"{BASE}/q2_results"
os.makedirs(OUT_DIR, exist_ok=True)

REC1 = "/home/mmaaz/SLAM/extracted_data/tmp_recordings/tmp_recordings"
REC2 = "/home/mmaaz/SLAM/extracted_data/tmp_recordings2"

# All 9 sequences with metadata
SEQUENCES = {
    "Basement_1":     {"rec": REC1, "type": "indoor",  "color": "#1f77b4"},
    "Basement_2":     {"rec": REC1, "type": "indoor",  "color": "#aec7e8"},
    "Floor7_Hallway": {"rec": REC1, "type": "indoor",  "color": "#ff7f0e"},
    "Outdoor_1":      {"rec": REC1, "type": "outdoor", "color": "#2ca02c"},
    "Washroom":       {"rec": REC1, "type": "indoor",  "color": "#d62728"},
    "BikeStorage":    {"rec": REC2, "type": "indoor",  "color": "#9467bd"},
    "BikeStorage2":   {"rec": REC2, "type": "indoor",  "color": "#8c564b"},
    "Entrance2":      {"rec": REC2, "type": "mixed",   "color": "#e377c2"},
    "OnePoolStreet1": {"rec": REC2, "type": "outdoor", "color": "#17becf"},
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_tum(path):
    poses = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            v = list(map(float, line.split()))
            if len(v) >= 8:
                poses.append(v[:8])
    return np.array(poses)


def centre(traj):
    pos = traj[:, 1:4] - traj[0, 1:4]
    return pos


def arc_length(traj):
    pos = traj[:, 1:4]
    return float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))


def load_colmap(seq_name):
    path = os.path.join(COLMAP_DIR, f"{seq_name}_colmap_poses.txt")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    poses = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 8:
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                poses.append([tx, ty, tz])
    return np.array(poses) if poses else None


def load_all_orbslam():
    trajs = {}
    for name in SEQUENCES:
        path = os.path.join(ORB_DIR, f"{name}_trajectory.txt")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            trajs[name] = load_tum(path)
    return trajs


# ── Q2a: all ORB-SLAM2 trajectories overview ─────────────────────────────────

def plot_q2a(trajs):
    print("Q2a: plotting all 9 ORB-SLAM2 trajectories …")
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle("Q2a – ORB-SLAM2 Trajectories on All 9 Collected Sequences\n"
                 "(Intel RealSense D455, 848×480 @ 30fps)", fontsize=14, fontweight='bold')

    seq_names = list(SEQUENCES.keys())
    for idx, name in enumerate(seq_names):
        ax = axes[idx // 3][idx % 3]
        meta = SEQUENCES[name]
        color = meta["color"]
        env   = meta["type"]

        if name not in trajs:
            ax.text(0.5, 0.5, f"{name}\n(no trajectory)", ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='red')
            ax.set_title(name)
            continue

        traj = trajs[name]
        pos  = centre(traj)
        t    = traj[:, 0] - traj[0, 0]

        sc = ax.scatter(pos[:, 0], pos[:, 2], c=t, cmap='plasma',
                        s=2, alpha=0.7)
        ax.plot(pos[0, 0], pos[0, 2], 'go', ms=8, label='Start', zorder=5)
        ax.plot(pos[-1, 0], pos[-1, 2], 'rs', ms=8, label='End', zorder=5)

        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Z (m)', fontsize=8)
        ax.set_title(f'{name}\n[{env}] {len(traj)} poses, {arc_length(traj):.1f}m', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='lower right')
        plt.colorbar(sc, ax=ax, label='Time (s)', pad=0.02)

    plt.tight_layout()
    out = f"{OUT_DIR}/q2a_all_trajectories.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved → {out}")


# ── Q2b: COLMAP vs ORB-SLAM2 ──────────────────────────────────────────────────

def plot_q2b(trajs):
    print("Q2b: COLMAP vs ORB-SLAM2 comparison …")

    # Check which sequences have COLMAP results
    colmap_results = {}
    for name in SEQUENCES:
        c = load_colmap(name)
        if c is not None and len(c) > 5:
            colmap_results[name] = c

    n_success = len(colmap_results)
    n_total   = len(SEQUENCES)
    print(f"  COLMAP succeeded on {n_success}/{n_total} sequences: {list(colmap_results.keys())}")

    # Figure layout: left = method comparison table, right = trajectory plots for colmap successes
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Q2b – COLMAP vs ORB-SLAM2: Structure-from-Motion Comparison",
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # --- Method comparison summary table ---
    ax_table = fig.add_subplot(gs[0, :2])
    ax_table.axis('off')
    rows = [["Sequence", "Type", "ORB-SLAM2 poses", "COLMAP poses", "COLMAP result"]]
    for name, meta in SEQUENCES.items():
        orb_poses = len(trajs[name]) if name in trajs else 0
        if name in colmap_results:
            col_poses = len(colmap_results[name])
            col_result = f"{col_poses} poses"
        else:
            col_result = "Failed (insufficient baseline)" if name not in colmap_results else "–"
            col_poses  = 0

        path = os.path.join(COLMAP_DIR, f"{name}_colmap_poses.txt")
        if os.path.exists(path) and os.path.getsize(path) == 0:
            col_result = "Failed"
        elif not os.path.exists(path):
            col_result = "Not run yet"

        rows.append([name, meta["type"], str(orb_poses), str(col_poses) if col_poses else "–", col_result])

    cell_colors = [["#d0e4f7"] * 5]
    for i, row in enumerate(rows[1:]):
        ok = row[4] not in ("Failed", "Not run yet", "Failed (insufficient baseline)")
        cell_colors.append(["white"] * 4 + [("#d0f0d0" if ok else "#ffe0e0")])

    table = ax_table.table(cellText=rows[1:], colLabels=rows[0],
                           cellLoc='center', loc='center',
                           cellColours=cell_colors[1:],
                           colColours=["#d0e4f7"] * 5)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax_table.set_title("COLMAP vs ORB-SLAM2 Results Summary", fontsize=11, fontweight='bold')

    # --- COLMAP methodology explanation ---
    ax_explain = fig.add_subplot(gs[0, 2])
    ax_explain.axis('off')
    txt = ("COLMAP Method:\n\n"
           "• Batch Structure-from-Motion\n"
           "• Keyframes: every 30th frame\n"
           "  (~1fps, ~0.5-1m baseline)\n"
           "• Sequential matcher (overlap=20)\n"
           "• Loop detection enabled\n\n"
           "ORB-SLAM2 Method:\n\n"
           "• Online monocular SLAM\n"
           "• Processes all 30fps frames\n"
           "• Keyframe selection internal\n"
           "• Explicit loop closure\n"
           "• Real-time capable\n\n"
           "Key difference:\n"
           "COLMAP: offline batch\n"
           "ORB-SLAM2: online tracking")
    ax_explain.text(0.05, 0.95, txt, transform=ax_explain.transAxes,
                    fontsize=8.5, va='top', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    # --- Plot COLMAP vs ORB-SLAM2 for sequences where COLMAP worked ---
    plot_idx = 0
    for name in list(colmap_results.keys())[:6]:  # up to 6 comparison plots
        row = 1 + plot_idx // 3
        col = plot_idx % 3
        if row > 2:
            break
        ax = fig.add_subplot(gs[row, col])
        meta = SEQUENCES[name]

        # ORB-SLAM2 trajectory
        if name in trajs:
            pos_orb = centre(trajs[name])
            # normalize scale
            ax.plot(pos_orb[:, 0], pos_orb[:, 2], 'b-', lw=1.2, alpha=0.7, label='ORB-SLAM2')
            ax.plot(pos_orb[0, 0], pos_orb[0, 2], 'bo', ms=7, zorder=5)

        # COLMAP trajectory (centred)
        c = colmap_results[name]
        c_centred = c - c[0]
        ax.plot(c_centred[:, 0], c_centred[:, 2], 'r-', lw=1.2, alpha=0.7, label='COLMAP')
        ax.plot(c_centred[0, 0], c_centred[0, 2], 'ro', ms=7, zorder=5)

        n_orb = len(trajs[name]) if name in trajs else 0
        ax.set_title(f'{name}\nORB: {n_orb} pts | COLMAP: {len(c)} kf', fontsize=8)
        ax.set_xlabel('X (m)', fontsize=7); ax.set_ylabel('Z (m)', fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plot_idx += 1

    # Fill remaining grid cells if fewer than 6 COLMAP successes
    for extra_idx in range(plot_idx, 6):
        row = 1 + extra_idx // 3
        col = extra_idx % 3
        if row > 2:
            break
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
        ax.text(0.5, 0.5, "COLMAP\nnot reconstructed", ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='gray',
                bbox=dict(boxstyle='round', facecolor='whitesmoke'))

    out = f"{OUT_DIR}/q2b_colmap_vs_orbslam.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved → {out}")


# ── Q2c: trajectory statistics for all sequences ──────────────────────────────

def plot_q2c(trajs):
    print("Q2c: trajectory statistics for all 9 sequences …")

    names  = list(trajs.keys())
    n_seqs = len(names)

    path_lengths = [arc_length(trajs[n]) for n in names]
    pose_counts  = [len(trajs[n]) for n in names]
    durations    = [trajs[n][-1, 0] - trajs[n][0, 0] for n in names]
    avg_speeds   = [path_lengths[i] / max(durations[i], 1) for i in range(n_seqs)]
    colors       = [SEQUENCES[n]["color"] for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Q2c – ORB-SLAM2 Statistics Across All 9 Collected Sequences",
                 fontsize=14, fontweight='bold')

    x = np.arange(n_seqs)
    short_names = [n.replace("_", "\n") for n in names]

    # Bar: path lengths
    ax = axes[0, 0]
    bars = ax.bar(x, path_lengths, color=colors, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylabel("Path Length (m)"); ax.set_title("ORB-SLAM2 Tracked Path Length")
    for bar, val in zip(bars, path_lengths):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.1f}m", ha='center', va='bottom', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # Bar: pose counts
    ax = axes[0, 1]
    bars = ax.bar(x, pose_counts, color=colors, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylabel("Number of Poses"); ax.set_title("ORB-SLAM2 Tracked Poses")
    for bar, val in zip(bars, pose_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha='center', va='bottom', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # Bar: duration
    ax = axes[1, 0]
    bars = ax.bar(x, durations, color=colors, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylabel("Sequence Duration (s)"); ax.set_title("Sequence Duration Tracked by ORB-SLAM2")
    for bar, val in zip(bars, durations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.0f}s", ha='center', va='bottom', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # Bar: average speed
    ax = axes[1, 1]
    bars = ax.bar(x, avg_speeds, color=colors, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylabel("Average Speed (m/s)"); ax.set_title("Average Camera Speed per Sequence")
    for bar, val in zip(bars, avg_speeds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.2f}", ha='center', va='bottom', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # Legend: env type
    handles = [
        plt.Rectangle((0,0),1,1, fc="#1f77b4", alpha=0.85, label="indoor"),
        plt.Rectangle((0,0),1,1, fc="#2ca02c", alpha=0.85, label="outdoor"),
        plt.Rectangle((0,0),1,1, fc="#e377c2", alpha=0.85, label="mixed"),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10, title="Environment type")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = f"{OUT_DIR}/q2c_statistics.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved → {out}")


# ── Q2 Summary ────────────────────────────────────────────────────────────────

def plot_q2_summary(trajs):
    print("Q2 summary …")
    # 3×3 grid, colour by environment type
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle("Q2 Summary – ORB-SLAM2 Visual SLAM on 9 Collected Sequences",
                 fontsize=13, fontweight='bold')

    for idx, (name, meta) in enumerate(SEQUENCES.items()):
        ax = axes[idx // 3][idx % 3]
        color = meta["color"]
        if name not in trajs:
            ax.text(0.5, 0.5, f"{name}\nno trajectory", ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(name)
            continue

        traj = trajs[name]
        pos  = centre(traj)
        ax.plot(pos[:, 0], pos[:, 2], color=color, lw=1.5, alpha=0.85)
        ax.plot(pos[0, 0], pos[0, 2], 'go', ms=9, zorder=5)
        ax.plot(pos[-1, 0], pos[-1, 2], 'rs', ms=9, zorder=5)
        ax.set_title(f"{name} [{meta['type']}]", fontsize=9)
        ax.set_xlabel("X (m)", fontsize=8); ax.set_ylabel("Z (m)", fontsize=8)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

        info = (f"Poses: {len(traj)}\n"
                f"Path: {arc_length(traj):.1f} m\n"
                f"Dur: {traj[-1,0]-traj[0,0]:.0f} s")
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=7.5,
                va='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    plt.tight_layout()
    out = f"{OUT_DIR}/q2_summary.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Q2: Visual SLAM – All 9 Collected Sequences")
    print("=" * 60)

    trajs = load_all_orbslam()
    print(f"Loaded ORB-SLAM2 trajectories: {list(trajs.keys())}")

    plot_q2a(trajs)
    plot_q2b(trajs)
    plot_q2c(trajs)
    plot_q2_summary(trajs)

    print("\nAll Q2 plots saved to:", OUT_DIR)
    print("Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith('.png'):
            print(f"  {f}")
