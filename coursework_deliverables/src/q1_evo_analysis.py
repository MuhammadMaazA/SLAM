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
OUT   = os.environ.get('SLAM_OUT',  os.path.join(_ROOT, 'data', 'q1_results'))
os.makedirs(OUT, exist_ok=True)

# ──────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────
_EVO_EXPECTED_EXC = (ValueError, RuntimeError, KeyError, IOError)

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


def plot_trajectory(ax, traj_ref, traj_est, label_est, color='tab:blue', title=''):
    """Plot 2D XY trajectories."""
    if traj_ref is not None:
        xyz = traj_ref.positions_xyz
        ax.plot(xyz[:, 0], xyz[:, 2], 'k-', lw=1.5, alpha=0.7, label='Ground Truth')
    if traj_est is not None:
        xyz = traj_est.positions_xyz
        ax.plot(xyz[:, 0], xyz[:, 2], color=color, lw=1.5, alpha=0.9, label=label_est)
        ax.plot(*xyz[0, [0, 2]], 'go', ms=7)
        ax.plot(*xyz[-1, [0, 2]], 'rs', ms=7)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('X (m)', fontsize=8)
    ax.set_ylabel('Z (m)', fontsize=8)
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
    configs = [
        ('KITTI 07', 'kitti07-gt-tum.txt', 'kitti07-baseline.txt'),
        ('TUM freiburg3_long_office_household', 'rgbd_dataset_freiburg3_long_office_household/groundtruth.txt',
         'tum-baseline.txt'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Q1a: Baseline Evaluation — KITTI07 & TUM', fontsize=13, fontweight='bold')
    colors = ['tab:blue', 'tab:orange']

    for col, (name, gt_file, est_file) in enumerate(configs):
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
                        title=f'{name} — trajectory')
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
    print("\n=== Q1b: Feature Count Variations ===")

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        'Q1b: ORB Feature Count — Impact on Monocular ORB-SLAM2 Tracking\n'
        'KITTI07 (10 Hz, fast vehicle motion): feat<1000 fails the ≥100-inlier RANSAC '
        'initialiser (hard-coded in the UCL ORB-SLAM2 build — verified from run logs '
        '"The map is empty; nothing to save")\n'
        'TUM freiburg3_long_office_household (30 Hz, indoor): feat=250 fails init, '
        'feat=500 tracks 45 % of the sequence then loses, feat=800 tracks fully with '
        '~3× the baseline drift, feat=1000 is the reference',
        fontsize=10.5, fontweight='bold')

    # KITTI07 — explicit feature-count reduction from the 1000-feature baseline.
    kitti_gt = load_traj('kitti07-gt-tum.txt')
    kitti_configs = [
        ('1000 (baseline)', 'kitti07-baseline.txt',  'tab:blue'),
        ('500 (reduced)',   'kitti07-feat500.txt',   'tab:orange'),
        ('250 (reduced)',   'kitti07-feat250.txt',   'tab:red'),
        ('100 (reduced)',   'kitti07-feat100.txt',   'tab:purple'),
    ]

    ate_vals_kitti = []
    for col, (label, fname, color) in enumerate(kitti_configs):
        if label is None:
            axes[0, col].axis('off')
            continue
        traj_est = load_traj(fname) if fname else None
        if traj_est is not None:
            stats, est_a, gt_a = compute_ate(traj_est, kitti_gt)
            rmse = stats['rmse'] if stats else float('nan')
            print(f"  KITTI07 feat={label}: RMSE={rmse:.4f}m, "
                  f"poses={len(traj_est.timestamps)}")
            ate_vals_kitti.append(rmse)
            plot_trajectory(axes[0, col], gt_a, est_a,
                            label_est=f'feat={label} (RMSE={rmse:.3f}m)',
                            color=color,
                            title=f'KITTI07 feat={label}')
        else:
            ate_vals_kitti.append(float('nan'))
            axes[0, col].text(0.5, 0.5,
                              f'feat={label}\nMonocular init failed\n'
                              f'(<100 RANSAC inliers during\n bootstrap; no map saved)',
                              ha='center', va='center',
                              transform=axes[0, col].transAxes,
                              fontsize=11, color='red', fontweight='bold')
            axes[0, col].set_title(f'KITTI07 feat={label}', fontsize=9)
            print(f"  KITTI07 feat={label}: FAILED (monocular init — <100 inliers)")

    # TUM — compare a higher feature count against the corrected 1000-feature
    # baseline and a reduced-count run on the long sequence.
    tum_gt  = load_traj('rgbd_dataset_freiburg3_long_office_household/groundtruth.txt')
    tum_configs = [
        ('250 (reduced)',  'tum-feat250.txt',   'tab:purple'),
        ('500 (reduced)',  'tum-feat500.txt',   'tab:green'),
        ('800 (reduced)',  'tum-feat800.txt',   'tab:orange'),
        ('1000 (baseline)','tum-baseline.txt',  'tab:blue'),
    ]

    for col, (label, fname, color) in enumerate(tum_configs):
        traj_est = load_traj(fname)
        if traj_est is not None:
            stats, est_a, gt_a = compute_ate(traj_est, tum_gt)
            rmse = stats['rmse'] if stats else float('nan')
            print(f"  TUM feat={label}: RMSE={rmse:.4f}m, "
                  f"poses={len(traj_est.timestamps)}")
            plot_trajectory(axes[1, col], gt_a, est_a,
                            label_est=f'feat={label} (RMSE={rmse:.3f}m)',
                            color=color,
                            title=f'TUM feat={label}')
        else:
            axes[1, col].text(0.5, 0.5,
                              f'feat={label}\nMonocular init failed\n'
                              f'(<100 RANSAC inliers during\n bootstrap; no map saved)',
                              ha='center', va='center',
                              transform=axes[1, col].transAxes,
                              fontsize=11, color='red', fontweight='bold')
            axes[1, col].set_title(f'TUM feat={label}', fontsize=9)

    # Row labels
    for row, label in enumerate(['KITTI 07\n(feature count reduction: 1000 → 500 → 250 → 100)',
                                  'TUM freiburg3_long\n(feature count reduction: 250 → 500 → 800 → 1000)']):
        axes[row, 0].set_ylabel(f'{label}\nZ (m)', fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT, 'q1b_features.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────
# Q1c: OUTLIER REJECTION
# ──────────────────────────────────────────────────
def plot_q1c():
    print("\n=== Q1c: Outlier Rejection ===")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Q1c: Effect of Disabling Outlier Rejection', fontsize=13, fontweight='bold')

    pairs = [
        ('KITTI 07',            'kitti07-gt-tum.txt', 'kitti07-baseline.txt',   'kitti07-nooutlier.txt'),
        ('TUM freiburg3_long_office_household',   'rgbd_dataset_freiburg3_long_office_household/groundtruth.txt',
                                 'tum-baseline.txt',    'tum-nooutlier.txt'),
    ]

    for col, (name, gt_f, base_f, noout_f) in enumerate(pairs):
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

        # Trajectory comparison
        ax = axes[0, col]
        if gt_b is not None:
            ax.plot(gt_b.positions_xyz[:, 0], gt_b.positions_xyz[:, 2],
                    'k-', lw=1.5, label='GT')
        if est_b is not None:
            ax.plot(est_b.positions_xyz[:, 0], est_b.positions_xyz[:, 2],
                    'tab:blue', lw=1.5, alpha=0.8, label=f'Baseline ({n_b} poses)')
        if est_n is not None:
            ax.plot(est_n.positions_xyz[:, 0], est_n.positions_xyz[:, 2],
                    'tab:red', lw=1.5, alpha=0.8, label=f'No outlier ({n_n} poses)')
        ax.set_title(f'{name} — trajectories', fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')

        # ATE bar
        ax2 = axes[1, col]
        bars = ax2.bar(['Baseline', 'No Outlier'], [rmse_b, rmse_n],
                       color=['tab:blue', 'tab:red'], alpha=0.8, edgecolor='k')
        ax2.set_ylabel('ATE RMSE (m)', fontsize=8)
        ax2.set_title(f'{name} — ATE comparison', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, [rmse_b, rmse_n]):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                         f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
        ax2.text(0.25, -0.05, f'n={n_b}', ha='center', transform=ax2.transAxes, fontsize=8)

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

    pairs = [
        ('KITTI 07', 'kitti07-gt-tum.txt', 'kitti07-baseline.txt', 'kitti07-noloop.txt'),
        ('TUM freiburg3_long_office_household', 'rgbd_dataset_freiburg3_long_office_household/groundtruth.txt',
         'tum-baseline.txt', 'tum-noloop.txt'),
    ]

    for col, (name, gt_f, base_f, noloop_f) in enumerate(pairs):
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
        if gt_b is not None:
            ax.plot(gt_b.positions_xyz[:, 0], gt_b.positions_xyz[:, 2],
                    'k-', lw=1.5, label='GT')
        if est_b is not None:
            ax.plot(est_b.positions_xyz[:, 0], est_b.positions_xyz[:, 2],
                    'tab:blue', lw=1.5, alpha=0.8, label=f'With loop ({n_b} poses)')
        if est_l is not None:
            ax.plot(est_l.positions_xyz[:, 0], est_l.positions_xyz[:, 2],
                    'tab:orange', lw=1.5, alpha=0.8, label=f'No loop ({n_l} poses)')
        ax.set_title(f'{name} — trajectories', fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')

        ax2 = axes[1, col]
        bars = ax2.bar(['With Loop\nClosure', 'No Loop\nClosure'], [rmse_b, rmse_l],
                       color=['tab:blue', 'tab:orange'], alpha=0.8, edgecolor='k')
        ax2.set_ylabel('ATE RMSE (m)', fontsize=8)
        ax2.set_title(f'{name} — ATE comparison', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, [rmse_b, rmse_l]):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                         f'{val:.3f}m', ha='center', fontsize=9, fontweight='bold')

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
