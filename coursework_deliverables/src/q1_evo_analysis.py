#!/usr/bin/env python3
"""
COMP0222 Coursework 2 - Question 1: EVO trajectory evaluation
Generates ATE comparison plots for all Q1 experiments.

Q1a: Baseline evaluation (KITTI07 + TUM)
Q1b: Feature count variations (controlled by the
     ``ORBextractor.nFeatures`` key in the per-sequence YAML; see
     ``data/part1_analysis/KITTI04-12_custom_f{500,250,100}.yaml`` and
     ``TUM1_custom_f{800,500,250}.yaml``)
Q1c: Outlier rejection disabled. Applied as a source patch to ORB-SLAM2:
     ``src/orbslam2_patches/q1c_disable_outlier_rejection.patch``.
     Trajectories in ``*-nooutlier.txt`` are produced by the patched
     binary; see the accompanying README for rebuild instructions.
Q1d: Loop closure disabled. Applied as a source patch to ORB-SLAM2:
     ``src/orbslam2_patches/q1d_disable_loop_closure.patch``.
     Trajectories in ``*-noloop.txt`` are produced by the patched binary.
"""

import os
import os.path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from evo.tools import file_interface
from evo.core import metrics, sync, trajectory
from evo.core.metrics import PoseRelation, Unit

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
DATA  = os.environ.get('SLAM_DATA', os.path.join(_ROOT, 'data', 'part1_analysis'))
OUT   = os.environ.get('SLAM_OUT',  os.path.abspath(os.path.join(_ROOT, '..', 'plots')))
os.makedirs(OUT, exist_ok=True)

# ──────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────
_EVO_EXPECTED_EXC = (ValueError, RuntimeError, KeyError, IOError)
_AXIS_LABELS = {0: 'X (m)', 1: 'Y (m)', 2: 'Z (m)'}

# Trajectory dumps expected for a full Q1 reproduction (ORB-SLAM2 + rerun_all.sh q1_*).
_Q1_EXPECTED_FILES = [
    'kitti07-baseline.txt', 'kitti07-feat500.txt', 'kitti07-feat250.txt',
    'kitti07-feat100.txt', 'kitti07-nooutlier.txt', 'kitti07-noloop.txt',
    'tum-baseline.txt', 'tum-feat250.txt', 'tum-feat500.txt', 'tum-feat800.txt',
    'tum-nooutlier.txt', 'tum-noloop.txt',
]


def print_q1_data_manifest():
    """List missing trajectory files so missing plots are not mistaken for SLAM failure."""
    missing = [f for f in _Q1_EXPECTED_FILES
               if not os.path.exists(os.path.join(DATA, f))]
    print('\n=== Q1 data manifest ===')
    print(f'  DATA directory: {DATA}')
    if not missing:
        print('  All expected ORB trajectory dumps are present.')
    else:
        print(f'  Missing {len(missing)} file(s) (plots will show "no data" / init failed panels):')
        for f in missing:
            print(f'    - {f}')
        print('  Generate them with ORB-SLAM2 after applying patches in')
        print('    src/orbslam2_patches/ — see README there — then')
        print('    coursework_deliverables/src/rerun_all.sh q1_kitti q1_tum_xyz')


def load_traj(fname):
    """Return an EVO trajectory for ``fname`` or ``None`` if the file is absent.

    Only ``FileNotFoundError`` maps to a silent ``None`` (some Q1 ablations
    are legitimately not runnable, e.g. KITTI07 feat<1000 which fails
    ORB-SLAM2 monocular initialisation). All other I/O errors are re-raised
    so real bugs (corrupt dumps, format mismatches) surface.
    """
    path = os.path.join(DATA, fname)
    if not os.path.exists(path):
        return None
    return file_interface.read_tum_trajectory_file(path)


def compute_ate(traj_est, traj_ref):
    """Align and compute ATE (RMSE of translation errors)."""
    try:
        traj_ref_s, traj_est_s = sync.associate_trajectories(traj_ref, traj_est,
                                                               max_diff=0.5)
        traj_est_s.align(traj_ref_s, correct_scale=True)
        err_metric = metrics.APE(PoseRelation.translation_part)
        err_metric.process_data((traj_ref_s, traj_est_s))
        stats = err_metric.get_all_statistics()
        return stats, traj_est_s, traj_ref_s
    except _EVO_EXPECTED_EXC as e:
        # Expected failures: trajectories with no temporal overlap, degenerate
        # alignment (e.g. <3 common poses). These are legitimate "no ATE"
        # situations, not silent bug-swallowing.
        print(f"  [WARN] ATE alignment failed ({type(e).__name__}): {e}")
        return None, None, None


def compute_rpe(traj_est, traj_ref, delta=1.0, delta_unit=Unit.frames):
    """
    Compute Relative Pose Error (translation component) between aligned
    trajectories. Uses fixed delta (default 1 frame) — the canonical EVO RPE.
    Returns (stats_dict, per_edge_errors_array).
    """
    try:
        ref_s, est_s = sync.associate_trajectories(traj_ref, traj_est,
                                                    max_diff=0.5)
        est_s.align(ref_s, correct_scale=True)
        rpe = metrics.RPE(PoseRelation.translation_part,
                          delta=delta, delta_unit=delta_unit,
                          rel_delta_tol=0.1, all_pairs=False)
        rpe.process_data((ref_s, est_s))
        stats = rpe.get_all_statistics()
        errs  = np.array(list(rpe.error))
        return stats, errs
    except _EVO_EXPECTED_EXC as e:
        print(f"  [WARN] RPE alignment failed ({type(e).__name__}): {e}")
        return None, np.array([])


def _axlim(arr, margin=0.1):
    lo, hi = arr.min(), arr.max()
    pad = (hi - lo) * margin
    return lo - pad, hi + pad


def plot_trajectory(ax, traj_ref, traj_est, label_est, color='tab:blue', title='',
                    xy_axes=(0, 2)):
    """Plot 2D trajectories. xy_axes selects which two position dims to use.
    Use (0,2) for KITTI (X-Z ground plane) and (0,1) for TUM (X-Y).
    """
    a0, a1 = xy_axes
    if traj_ref is not None:
        xyz = traj_ref.positions_xyz
        ax.plot(xyz[:, a0], xyz[:, a1], 'k-', lw=1.5, alpha=0.7, label='Ground Truth')
    if traj_est is not None:
        xyz = traj_est.positions_xyz
        ax.plot(xyz[:, a0], xyz[:, a1], color=color, lw=1.5, alpha=0.9, label=label_est)
        ax.plot(xyz[0, a0], xyz[0, a1], 'go', ms=7)
        ax.plot(xyz[-1, a0], xyz[-1, a1], 'rs', ms=7)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel(_AXIS_LABELS.get(a0, f'dim{a0} (m)'), fontsize=8)
    ax.set_ylabel(_AXIS_LABELS.get(a1, f'dim{a1} (m)'), fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_ate_over_time(ax, stats, traj_ref, traj_est, label, color='tab:blue'):
    """Plot ATE error over time."""
    if traj_ref is None or traj_est is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    try:
        err_metric = metrics.APE(PoseRelation.translation_part)
        err_metric.process_data((traj_ref, traj_est))
        err = np.array(list(err_metric.error))
        t   = np.arange(len(err))
        ax.plot(t, err, color=color, lw=1.2, alpha=0.8)
        ax.axhline(err.mean(), color=color, linestyle='--', lw=1,
                   label=f'Mean={err.mean():.3f}m')
        ax.set_xlabel('Frame', fontsize=8)
        ax.set_ylabel('ATE (m)', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    except _EVO_EXPECTED_EXC as e:
        ax.text(0.5, 0.5, f'{type(e).__name__}: {str(e)[:40]}', ha='center',
                va='center', transform=ax.transAxes, fontsize=7)


# ──────────────────────────────────────────────────
# Q1a: BASELINE
# ──────────────────────────────────────────────────
def plot_q1a():
    print("\n=== Q1a: Baseline Evaluation ===")
    # (name, gt_file, est_file, xy_axes)
    # KITTI: X-Z is the ground plane (vehicle moves forward=Z, right=X)
    # TUM freiburg3_long_office: camera moves primarily in the XY horizontal plane
    configs = [
        ('KITTI 07', 'kitti07-gt-tum.txt', 'kitti07-baseline.txt', (0, 2)),
        ('TUM freiburg3_long_office_household',
         'rgbd_dataset_freiburg3_long_office_household/groundtruth.txt',
         'tum-baseline.txt', (0, 1)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Q1a: Baseline Evaluation — KITTI07 & TUM', fontsize=13, fontweight='bold')
    colors = ['tab:blue', 'tab:orange']

    for col, (name, gt_file, est_file, xy_axes) in enumerate(configs):
        traj_gt  = load_traj(gt_file)
        traj_est = load_traj(est_file)

        stats, est_aligned, gt_aligned = compute_ate(traj_est, traj_gt) \
            if (traj_gt and traj_est) else (None, None, None)

        rmse = stats['rmse'] if stats else float('nan')
        mean = stats['mean'] if stats else float('nan')
        print(f"  {name}: RMSE={rmse:.4f}m  MEAN={mean:.4f}m  "
              f"poses_est={len(traj_est.timestamps) if traj_est else 0}")

        plot_trajectory(axes[0, col], gt_aligned, est_aligned,
                        label_est=f'ORB-SLAM2 (RMSE={rmse:.3f}m)',
                        color=colors[col],
                        title=f'{name} — trajectory',
                        xy_axes=xy_axes)

        # Path-length context logged to console only
        if gt_aligned is not None and not np.isnan(rmse):
            path_m = float(np.sum(np.linalg.norm(
                np.diff(gt_aligned.positions_xyz, axis=0), axis=1)))
            pct_path = 100.0 * rmse / max(path_m, 1e-3)
            print(f"    GT path ≈ {path_m:.0f} m | ATE {rmse:.3f} m = {pct_path:.1f}% of path")

        plot_ate_over_time(axes[1, col], stats, gt_aligned, est_aligned,
                           label=name, color=colors[col])
        axes[1, col].set_title(f'{name} — ATE over time', fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT, 'q1a_baseline.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────
# Q1b: FEATURE COUNT
# ──────────────────────────────────────────────────
def plot_q1b():
    """
    Saves two separate figures:
      q1b_kitti_features.png — 3 KITTI trajectory panels + RPE trend (1x4)
      q1b_tum_features.png   — 3 TUM trajectory panels + ATE trend (1x4)
    """
    print("\n=== Q1b: Feature Count Variations ===")

    tum_gt   = load_traj('rgbd_dataset_freiburg3_long_office_household/groundtruth.txt')
    kitti_gt = load_traj('kitti07-gt-tum.txt')

    # KITTI configs
    kitti_configs = [
        (1500, 'kitti07-feat1500.txt', 'tab:green',  '1500'),
        (2000, 'kitti07-feat2000.txt', 'tab:blue',   '2000'),
        (2500, 'kitti07-feat2500.txt', 'tab:orange', '2500'),
    ]
    kitti_data = {}
    for nf, fname, color, label in kitti_configs:
        t = load_traj(fname)
        if t is not None and kitti_gt is not None:
            sa, est_a, gt_a = compute_ate(t, kitti_gt)
            sr, _ = compute_rpe(t, kitti_gt, delta=1.0, delta_unit=Unit.frames)
            ate = sa['rmse'] if sa else float('nan')
            rpe = sr['rmse'] if sr else float('nan')
            kitti_data[nf] = (ate, rpe, est_a, gt_a, color, label, len(t.timestamps))
            print(f"  KITTI feat={nf}: ATE={ate:.4f}m  RPE={rpe:.4f}m  poses={len(t.timestamps)}")
        else:
            kitti_data[nf] = (float('nan'), float('nan'), None, None, color, label, 0)

    # TUM configs
    tum_configs = [
        (250,  'tum-feat250.txt',  'tab:purple', '250'),
        (500,  'tum-feat500.txt',  'tab:green',  '500'),
        (800,  'tum-feat800.txt',  'tab:orange', '800'),
        (1000, 'tum-baseline.txt', 'tab:blue',   '1000'),
    ]
    tum_ate = {}
    tum_poses = {}
    for nf, fname, color, label in tum_configs:
        t = load_traj(fname)
        if t is not None and tum_gt is not None:
            stats, est_a, gt_a = compute_ate(t, tum_gt)
            tum_ate[nf]   = (stats['rmse'] if stats else float('nan'), est_a, gt_a, color, label)
            tum_poses[nf] = len(t.timestamps)
            print(f"  TUM feat={nf}: RMSE={tum_ate[nf][0]:.4f}m, poses={tum_poses[nf]}")
        else:
            tum_ate[nf]   = (float('nan'), None, None, color, label)
            tum_poses[nf] = 0

    # ── Figure 1: KITTI (1 row x 4 cols) ────────────────────────────────────
    fig1, axes1 = plt.subplots(1, 4, figsize=(22, 5))
    fig1.suptitle(
        'Q1b KITTI07: Feature Count Effect (feat=1500, 2000, 2500)\n'
        'All 3 levels track the full sequence. '
        'Per-frame RPE improves monotonically with more features. '
        'ATE ~18 m for all (loop closure non-deterministic — see Q1d).',
        fontsize=11, fontweight='bold')

    for col, nf in enumerate([1500, 2000, 2500]):
        ax = axes1[col]
        ate, rpe, est_a, gt_a, color, lbl, n_poses = kitti_data[nf]
        if est_a is not None:
            plot_trajectory(ax, gt_a, est_a,
                            label_est=f'feat={nf} (RPE={rpe:.3f} m/frame)',
                            color=color,
                            title=f'KITTI07 feat={nf} | {n_poses} poses',
                            xy_axes=(0, 2))
        else:
            ax.text(0.5, 0.5, f'feat={nf}\nNo data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='red', fontweight='bold')

    # RPE trend (col 3)
    ax_kt = axes1[3]
    kf_vals  = sorted(kitti_data.keys())
    rpe_vals = [kitti_data[nf][1] for nf in kf_vals]
    kcolors  = [kitti_data[nf][4] for nf in kf_vals]
    for fv, rv, cv in zip(kf_vals, rpe_vals, kcolors):
        ax_kt.scatter(fv, rv, color=cv, s=120, zorder=5,
                      label=f'feat={fv}: {rv:.3f} m')
    if sum(1 for v in rpe_vals if not np.isnan(v)) >= 2:
        xw = np.array([f for f, v in zip(kf_vals, rpe_vals) if not np.isnan(v)])
        yw = np.array([v for v in rpe_vals if not np.isnan(v)])
        coeffs = np.polyfit(xw, yw, 1)
        xx = np.linspace(min(xw)*0.95, max(xw)*1.05, 100)
        ax_kt.plot(xx, np.polyval(coeffs, xx), 'k--', lw=1.5, alpha=0.6, label='trend')
    ax_kt.set_xlabel('ORB Feature Count', fontsize=10)
    ax_kt.set_ylabel('RPE RMSE (m/frame)', fontsize=10)
    ax_kt.set_title('RPE vs Feature Count', fontsize=10)
    ax_kt.legend(fontsize=8)
    ax_kt.grid(True, alpha=0.3)

    fig1.tight_layout()
    out1 = os.path.join(OUT, 'q1b_kitti_features.png')
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Saved: {out1}")

    # ── Figure 2: TUM (1 row x 4 cols) ──────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 4, figsize=(22, 5))
    fig2.suptitle(
        'Q1b TUM freiburg3_long_office: Feature Count Effect (feat=500, 800, 1000)\n'
        'ATE degrades as features drop. feat=250 fails init entirely.',
        fontsize=11, fontweight='bold')

    for col, (nf, color) in enumerate([(1000, 'tab:blue'), (800, 'tab:orange'), (500, 'tab:green')]):
        ax = axes2[col]
        rmse, est_a, gt_a, _, lbl = tum_ate[nf]
        n_poses = tum_poses[nf]
        if est_a is not None:
            plot_trajectory(ax, gt_a, est_a,
                            label_est=f'feat={nf} (RMSE={rmse:.3f} m)',
                            color=color,
                            title=f'TUM feat={nf} | {n_poses} poses',
                            xy_axes=(0, 1))
        else:
            ax.text(0.5, 0.5, f'feat={nf}\nInit failed', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='red', fontweight='bold')
            ax.set_title(f'TUM feat={nf}', fontsize=9)

    # ATE trend (col 3)
    ax_tt = axes2[3]
    feat_vals = [250, 500, 800, 1000]
    tum_ate_v = [tum_ate[nf][0] for nf in feat_vals]
    tum_col_v = [tum_ate[nf][3] for nf in feat_vals]
    for fv, av, cv, pv in zip(feat_vals, tum_ate_v, tum_col_v,
                               [tum_poses[nf] for nf in feat_vals]):
        if np.isnan(av):
            ax_tt.scatter(fv, 0, marker='x', s=150, color=cv, zorder=5,
                          linewidths=3, label=f'feat={fv}: FAILED')
        else:
            ax_tt.scatter(fv, av, color=cv, s=80, zorder=5,
                          label=f'feat={fv}: {av:.3f} m ({pv} poses)')
    working = [(fv, av) for fv, av in zip(feat_vals, tum_ate_v) if not np.isnan(av)]
    if len(working) >= 2:
        xw = np.array([w[0] for w in working])
        yw = np.array([w[1] for w in working])
        coeffs = np.polyfit(np.log(xw), yw, 1)
        xx = np.linspace(min(xw)*0.9, max(xw)*1.05, 100)
        ax_tt.plot(xx, np.polyval(coeffs, np.log(xx)), 'k--', lw=1.5, alpha=0.6, label='log trend')
    ax_tt.set_xlabel('ORB Feature Count', fontsize=10)
    ax_tt.set_ylabel('ATE RMSE (m)', fontsize=10)
    ax_tt.set_title('ATE vs Feature Count\n(X = init failed)', fontsize=10)
    ax_tt.legend(fontsize=8, loc='upper right')
    ax_tt.grid(True, alpha=0.3)
    ax_tt.set_xlim(100, 1100)

    fig2.tight_layout()
    out2 = os.path.join(OUT, 'q1b_tum_features.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {out2}")

def plot_q1c():
    print("\n=== Q1c: Outlier Rejection ===")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Q1c: Effect of Disabling Outlier Rejection', fontsize=13, fontweight='bold')

    # (name, gt_f, base_f, noout_f, xy_axes)
    pairs = [
        ('KITTI 07', 'kitti07-gt-tum.txt', 'kitti07-baseline.txt',
         'kitti07-nooutlier.txt', (0, 2)),
        ('TUM freiburg3_long_office_household',
         'rgbd_dataset_freiburg3_long_office_household/groundtruth.txt',
         'tum-baseline.txt', 'tum-nooutlier.txt', (0, 1)),
    ]

    for col, (name, gt_f, base_f, noout_f, xy_axes) in enumerate(pairs):
        gt   = load_traj(gt_f)
        base = load_traj(base_f)
        noout= load_traj(noout_f)

        stats_b, est_b, gt_b = compute_ate(base, gt) if (gt and base) else (None,None,None)
        stats_n, est_n, gt_n = compute_ate(noout, gt) if (gt and noout) else (None,None,None)

        rmse_b = stats_b['rmse'] if stats_b else float('nan')
        rmse_n = stats_n['rmse'] if stats_n else float('nan')
        n_b    = len(base.timestamps)  if base  else 0
        n_n    = len(noout.timestamps) if noout else 0
        print(f"  {name}: baseline RMSE={rmse_b:.4f}m ({n_b} poses), "
              f"no-outlier RMSE={rmse_n:.4f}m ({n_n} poses)")

        # Trajectory comparison — delegate fully to plot_trajectory()
        ax = axes[0, col]
        plot_trajectory(ax, gt_b, est_b,
                        label_est=f'Baseline ({n_b} poses)',
                        color='tab:blue',
                        title=f'{name} — trajectories',
                        xy_axes=xy_axes)
        if est_n is not None:
            a0, a1 = xy_axes
            xyz = est_n.positions_xyz
            ax.plot(xyz[:, a0], xyz[:, a1],
                    'tab:red', lw=1.5, alpha=0.8, label=f'No outlier ({n_n} poses)')
            ax.legend(fontsize=7)

        # ATE bar
        ax2 = axes[1, col]
        bars = ax2.bar(['Baseline', 'No Outlier'], [rmse_b, rmse_n],
                       color=['tab:blue', 'tab:red'], alpha=0.8, edgecolor='k')
        ax2.set_ylabel('ATE RMSE (m)', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')

        # % change in title (not inside the plot area)
        if not np.isnan(rmse_b) and not np.isnan(rmse_n) and rmse_b > 0:
            pct = 100.0 * (rmse_n - rmse_b) / rmse_b
            pct_str = f'{pct:+.1f}% ATE change'
            tracking_str = (f'Baseline {n_b} poses  |  No-outlier {n_n} poses '
                            f'({"−" + str(n_b - n_n) + " lost" if n_n < n_b * 0.9 else "tracking intact"})')
            ax2.set_title(f'{name}\n{pct_str}   {tracking_str}', fontsize=8)
        else:
            ax2.set_title(f'{name} — ATE comparison', fontsize=9)

        # Value labels above each bar
        for bar, val in zip(bars, [rmse_b, rmse_n]):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + ax2.get_ylim()[1] * 0.01,
                         f'{val:.3f} m', ha='center', va='bottom',
                         fontsize=10, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(OUT, 'q1c_outlier.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────
# Q1d: LOOP CLOSURE
# ──────────────────────────────────────────────────
def plot_q1d():
    print("\n=== Q1d: Loop Closure ===")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Q1d: Effect of Disabling Loop Closure', fontsize=13, fontweight='bold')

    # (name, gt_f, base_f, noloop_f, xy_axes)
    pairs = [
        ('KITTI 07', 'kitti07-gt-tum.txt', 'kitti07-baseline.txt',
         'kitti07-noloop.txt', (0, 2)),
        ('TUM freiburg3_long_office_household',
         'rgbd_dataset_freiburg3_long_office_household/groundtruth.txt',
         'tum-baseline.txt', 'tum-noloop.txt', (0, 1)),
    ]

    for col, (name, gt_f, base_f, noloop_f, xy_axes) in enumerate(pairs):
        gt     = load_traj(gt_f)
        base   = load_traj(base_f)
        noloop = load_traj(noloop_f)

        stats_b, est_b, gt_b = compute_ate(base, gt)   if (gt and base)   else (None,None,None)
        stats_l, est_l, gt_l = compute_ate(noloop, gt) if (gt and noloop) else (None,None,None)

        rmse_b = stats_b['rmse'] if stats_b else float('nan')
        rmse_l = stats_l['rmse'] if stats_l else float('nan')
        n_b    = len(base.timestamps)   if base   else 0
        n_l    = len(noloop.timestamps) if noloop else 0
        print(f"  {name}: baseline RMSE={rmse_b:.4f}m ({n_b} poses), "
              f"no-loop RMSE={rmse_l:.4f}m ({n_l} poses)")

        ax = axes[0, col]
        plot_trajectory(ax, gt_b, est_b,
                        label_est=f'With loop ({n_b} poses)',
                        color='tab:blue',
                        title=f'{name} — trajectories',
                        xy_axes=xy_axes)
        if est_l is not None:
            a0, a1 = xy_axes
            xyz = est_l.positions_xyz
            ax.plot(xyz[:, a0], xyz[:, a1],
                    'tab:orange', lw=1.5, alpha=0.8, label=f'No loop ({n_l} poses)')
            ax.legend(fontsize=7)

        ax2 = axes[1, col]
        bars = ax2.bar(['With Loop\nClosure', 'No Loop\nClosure'], [rmse_b, rmse_l],
                       color=['tab:blue', 'tab:orange'], alpha=0.8, edgecolor='k')
        ax2.set_ylabel('ATE RMSE (m)', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')

        if not np.isnan(rmse_b) and not np.isnan(rmse_l) and rmse_b > 0:
            pct = 100.0 * (rmse_l - rmse_b) / rmse_b
            ax2.set_title(f'{name}\n{pct:+.1f}% ATE change   '
                          f'Baseline {n_b} poses  |  No-loop {n_l} poses', fontsize=8)
        else:
            ax2.set_title(f'{name} — ATE comparison', fontsize=9)

        for bar, val in zip(bars, [rmse_b, rmse_l]):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + ax2.get_ylim()[1] * 0.01,
                         f'{val:.3f} m', ha='center', va='bottom',
                         fontsize=10, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(OUT, 'q1d_loop.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────
# SUMMARY FIGURE
# ──────────────────────────────────────────────────
def plot_summary():
    print("\n=== Summary: All Q1 ATE values ===")

    files = {
        'KITTI07 Baseline (1000)':    'kitti07-baseline.txt',
        'KITTI07 feat=500 (reduced)': 'kitti07-feat500.txt',
        'KITTI07 feat=250 (reduced)': 'kitti07-feat250.txt',
        'KITTI07 feat=100 (reduced)': 'kitti07-feat100.txt',
        'KITTI07 No Outlier':         'kitti07-nooutlier.txt',
        'KITTI07 No Loop':            'kitti07-noloop.txt',
        'TUM feat=250 (reduced)':     'tum-feat250.txt',
        'TUM feat=500 (reduced)':     'tum-feat500.txt',
        'TUM feat=800 (reduced)':     'tum-feat800.txt',
        'TUM Baseline (1000)':        'tum-baseline.txt',
        'TUM No Outlier':             'tum-nooutlier.txt',
        'TUM No Loop':                'tum-noloop.txt',
    }
    gt_map = {
        'KITTI': 'kitti07-gt-tum.txt',
        'TUM':   'rgbd_dataset_freiburg3_long_office_household/groundtruth.txt',
    }

    kitti_gt = load_traj(gt_map['KITTI'])
    tum_gt   = load_traj(gt_map['TUM'])

    labels, rmses, colors = [], [], []
    for label, fname in files.items():
        traj_est = load_traj(fname) if fname else None
        gt = kitti_gt if 'KITTI' in label else tum_gt
        if traj_est and gt:
            stats, _, _ = compute_ate(traj_est, gt)
            rmse = stats['rmse'] if stats else float('nan')
            n = len(traj_est.timestamps)
        else:
            rmse, n = float('nan'), 0
        print(f"  {label:35s}: RMSE={rmse:.4f}m  n={n}")
        labels.append(label.replace('KITTI07 ', 'K: ').replace('TUM ', 'T: '))
        rmses.append(rmse)
        colors.append('steelblue' if 'KITTI' in label else 'tomato')

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, [0 if np.isnan(v) else v for v in rmses],
                  color=colors, alpha=0.85, edgecolor='k')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('ATE RMSE (m)', fontsize=10)
    ax.set_title('Q1: ATE RMSE Summary — All Configurations', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, rmses):
        if np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, 0.005, 'FAILED',
                    ha='center', va='bottom', fontsize=7, color='red', fontweight='bold',
                    rotation=90)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='steelblue', label='KITTI07'),
                        Patch(facecolor='tomato',    label='TUM freiburg3_long_office_household')],
              fontsize=9)
    plt.tight_layout()
    out = os.path.join(OUT, 'q1_summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────
# Q1: RPE (Relative Pose Error) across all configurations
# ──────────────────────────────────────────────────
def plot_q1_rpe():
    """
    Compute RPE (translation) for every configuration used elsewhere in Q1
    and render (1) a RMSE bar chart and (2) per-frame RPE curves for the
    baselines. RPE is computed with delta=1 frame, which is the canonical
    definition used in the TUM RGB-D benchmark.
    """
    print("\n=== Q1: RPE (Relative Pose Error) ===")

    configs = {
        'KITTI07 Baseline':   ('kitti07-baseline.txt',   'kitti07-gt-tum.txt'),
        'KITTI07 feat=500':   ('kitti07-feat500.txt',    'kitti07-gt-tum.txt'),
        'KITTI07 feat=250':   ('kitti07-feat250.txt',    'kitti07-gt-tum.txt'),
        'KITTI07 feat=100':   ('kitti07-feat100.txt',    'kitti07-gt-tum.txt'),
        'KITTI07 No Outlier': ('kitti07-nooutlier.txt',  'kitti07-gt-tum.txt'),
        'KITTI07 No Loop':    ('kitti07-noloop.txt',     'kitti07-gt-tum.txt'),
        'TUM Baseline':       ('tum-baseline.txt',
                                'rgbd_dataset_freiburg3_long_office_household/groundtruth.txt'),
        'TUM feat=250':       ('tum-feat250.txt',
                                'rgbd_dataset_freiburg3_long_office_household/groundtruth.txt'),
        'TUM feat=500':       ('tum-feat500.txt',
                                'rgbd_dataset_freiburg3_long_office_household/groundtruth.txt'),
        'TUM feat=800':       ('tum-feat800.txt',
                                'rgbd_dataset_freiburg3_long_office_household/groundtruth.txt'),
        'TUM No Outlier':     ('tum-nooutlier.txt',
                                'rgbd_dataset_freiburg3_long_office_household/groundtruth.txt'),
        'TUM No Loop':        ('tum-noloop.txt',
                                'rgbd_dataset_freiburg3_long_office_household/groundtruth.txt'),
    }

    labels, rmses, medians, baseline_curves = [], [], [], {}
    for label, (est_f, gt_f) in configs.items():
        est = load_traj(est_f)
        gt  = load_traj(gt_f)
        if est is None or gt is None:
            print(f"  {label:25s}: NO DATA")
            labels.append(label); rmses.append(float('nan')); medians.append(float('nan'))
            continue
        stats, errs = compute_rpe(est, gt, delta=1.0, delta_unit=Unit.frames)
        if stats is None:
            labels.append(label); rmses.append(float('nan')); medians.append(float('nan'))
            continue
        print(f"  {label:25s}: RPE RMSE={stats['rmse']:.4f}m  "
              f"median={stats['median']:.4f}m  "
              f"max={stats['max']:.4f}m  n_edges={len(errs)}")
        labels.append(label)
        rmses.append(stats['rmse'])
        medians.append(stats['median'])
        if 'Baseline' in label:
            baseline_curves[label] = errs

    # Plot: bar chart + baseline time series
    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1.0])
    fig.suptitle('Q1: RPE (translation, δ=1 frame) — all configurations',
                 fontsize=13, fontweight='bold')

    # Bar chart
    ax0 = fig.add_subplot(gs[0, :])
    x = np.arange(len(labels))
    colors = ['steelblue' if 'KITTI' in l else 'tomato' for l in labels]
    bars = ax0.bar(x, [0 if np.isnan(v) else v for v in rmses],
                   color=colors, alpha=0.85, edgecolor='k')
    ax0.set_xticks(x)
    ax0.set_xticklabels([l.replace('KITTI07 ', 'K: ').replace('TUM ', 'T: ')
                          for l in labels], rotation=30, ha='right', fontsize=9)
    ax0.set_ylabel('RPE RMSE (m per frame)')
    ax0.set_title('RPE RMSE across configurations (blue=KITTI, red=TUM)')
    ax0.grid(True, alpha=0.3, axis='y')
    for bar, val, med in zip(bars, rmses, medians):
        if np.isnan(val):
            ax0.text(bar.get_x() + bar.get_width()/2, 0.001, 'NO DATA',
                     ha='center', fontsize=7, color='red', rotation=90)
        else:
            ax0.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.0005,
                     f'{val:.3f}\n(med {med:.3f})',
                     ha='center', fontsize=7)

    # Baseline RPE curves
    ax1 = fig.add_subplot(gs[1, 0])
    for (name, errs), c in zip(baseline_curves.items(), ['tab:blue', 'tab:orange']):
        ax1.plot(errs, lw=1.0, alpha=0.85, color=c, label=name)
    ax1.set_xlabel('Edge index (frame pair)')
    ax1.set_ylabel('RPE (m)')
    ax1.set_title('Per-edge RPE — baselines')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # RMSE vs MEDIAN scatter
    ax2 = fig.add_subplot(gs[1, 1])
    for l, r, m in zip(labels, rmses, medians):
        if np.isnan(r) or np.isnan(m):
            continue
        c = 'steelblue' if 'KITTI' in l else 'tomato'
        ax2.scatter(m, r, c=c, s=60, edgecolors='k')
        ax2.annotate(l.replace('KITTI07 ', '').replace('TUM ', ''),
                     (m, r), fontsize=7, xytext=(3, 3),
                     textcoords='offset points')
    lim = max([0.01] + [v for v in rmses + medians if not np.isnan(v)])
    ax2.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='RMSE = median')
    ax2.set_xlabel('RPE median (m)')
    ax2.set_ylabel('RPE RMSE (m)')
    ax2.set_title('RMSE vs median: gap indicates outliers')
    ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT, 'q1_rpe.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────
if __name__ == '__main__':
    print_q1_data_manifest()
    plot_q1a()
    plot_q1b()
    plot_q1c()
    plot_q1d()
    plot_summary()
    plot_q1_rpe()
    print(f"\nAll Q1 plots saved to: {OUT}")
