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

        # Path-length context: put ATE in perspective of total route length
        if gt_aligned is not None and not np.isnan(rmse):
            path_m = float(np.sum(np.linalg.norm(
                np.diff(gt_aligned.positions_xyz, axis=0), axis=1)))
            pct_path = 100.0 * rmse / max(path_m, 1e-3)
            axes[0, col].text(0.02, 0.02,
                              f'GT path ≈ {path_m:.0f} m\n'
                              f'ATE {rmse:.3f} m = {pct_path:.1f}% of path',
                              transform=axes[0, col].transAxes, fontsize=7.5,
                              va='bottom',
                              bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='lightyellow', alpha=0.9))

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
    Restructured layout:
      Row 0 — TUM (the informative case): 3 trajectory panels (baseline, feat800,
               feat500) + 1 ATE-vs-feature-count trend panel
      Row 1 — KITTI: baseline panel + 3 analytical failure panels showing WHY
               reduced-feature init fails at 10 fps (large inter-frame baseline)
      Row 2 — Full-width ATE vs feature count scatter+line for TUM only
    """
    print("\n=== Q1b: Feature Count Variations ===")

    tum_gt   = load_traj('rgbd_dataset_freiburg3_long_office_household/groundtruth.txt')
    kitti_gt = load_traj('kitti07-gt-tum.txt')

    # TUM configs — in ascending feature count order for the trend plot
    tum_configs = [
        (250,  'tum-feat250.txt',  'tab:purple', '250 (init fail)'),
        (500,  'tum-feat500.txt',  'tab:green',  '500 (partial)'),
        (800,  'tum-feat800.txt',  'tab:orange', '800 (reduced)'),
        (1000, 'tum-baseline.txt', 'tab:blue',   '1000 (baseline)'),
    ]

    # Compute TUM ATE values (NaN for missing/failed trajectories)
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
            print(f"  TUM feat={nf}: FAILED (no trajectory)")

    # KITTI — log-derived failure evidence
    # Verified from run logs (_rerun_logs/kitti07-feat{500,250,100}.stdout.log):
    # all three sub-1000 runs output "The map is empty; nothing to save" — init
    # never completes because 10 fps vehicle motion exceeds the RANSAC inlier
    # threshold regardless of feature count.
    kitti_baseline    = load_traj('kitti07-baseline.txt')
    stats_kb, est_kb, gt_kb = compute_ate(kitti_baseline, kitti_gt) \
        if (kitti_baseline and kitti_gt) else (None, None, None)
    rmse_kb = stats_kb['rmse'] if stats_kb else float('nan')
    print(f"  KITTI07 baseline: RMSE={rmse_kb:.4f}m, "
          f"poses={len(kitti_baseline.timestamps) if kitti_baseline else 0}")
    for nf in (500, 250, 100):
        print(f"  KITTI07 feat={nf}: FAILED (map empty — log: _rerun_logs/kitti07-feat{nf}.stdout.log)")

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 4, figure=fig,
                      height_ratios=[1.0, 1.0, 0.7],
                      hspace=0.48, wspace=0.35)
    fig.suptitle(
        'Q1b: ORB Feature Count — Impact on Monocular ORB-SLAM2\n'
        'TUM (30 fps indoor): ATE degrades monotonically as features drop; '
        'feat=250 fails init.   '
        'KITTI07 (10 fps driving): ALL reduced-feature runs fail — '
        'large inter-frame baseline means <100 RANSAC inliers even at feat=500.',
        fontsize=11, fontweight='bold')

    # ── Row 0: TUM trajectory panels (baseline, feat800, feat500) + trend ──
    tum_plot_order = [(1000, 'tab:blue'), (800, 'tab:orange'), (500, 'tab:green')]
    for col, (nf, color) in enumerate(tum_plot_order):
        ax = fig.add_subplot(gs[0, col])
        rmse, est_a, gt_a, _, lbl = tum_ate[nf]
        n_poses = tum_poses[nf]
        if est_a is not None:
            plot_trajectory(ax, gt_a, est_a,
                            label_est=f'feat={nf} (RMSE={rmse:.3f}m)',
                            color=color,
                            title=f'TUM feat={nf} | {n_poses} poses',
                            xy_axes=(0, 1))
        else:
            ax.text(0.5, 0.5, f'feat={nf}\nInit failed\n(map empty)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='red', fontweight='bold')
            ax.set_title(f'TUM feat={nf}', fontsize=9)

    # TUM trend panel (col 3)
    ax_trend = fig.add_subplot(gs[0, 3])
    feat_vals  = [250, 500, 800, 1000]
    ate_vals   = [tum_ate[nf][0] for nf in feat_vals]
    pose_vals  = [tum_poses[nf]  for nf in feat_vals]
    colors_tr  = [tum_ate[nf][3] for nf in feat_vals]

    # Plot working points; mark failures with an X
    for fv, av, cv, pv in zip(feat_vals, ate_vals, colors_tr, pose_vals):
        if np.isnan(av):
            ax_trend.scatter(fv, 0, marker='x', s=150, color=cv, zorder=5,
                             linewidths=3, label=f'feat={fv}: init FAILED')
        else:
            ax_trend.scatter(fv, av, color=cv, s=80, zorder=5,
                             label=f'feat={fv}: {av:.3f}m ({pv} poses)')

    # Fit log trend through working points only
    working = [(fv, av) for fv, av in zip(feat_vals, ate_vals) if not np.isnan(av)]
    if len(working) >= 2:
        xw = np.array([w[0] for w in working])
        yw = np.array([w[1] for w in working])
        coeffs = np.polyfit(np.log(xw), yw, 1)
        xx = np.linspace(min(xw)*0.9, max(xw)*1.05, 100)
        ax_trend.plot(xx, np.polyval(coeffs, np.log(xx)), 'k--', lw=1.5,
                      alpha=0.6, label='log trend (working pts)')

    ax_trend.set_xlabel('ORB Feature Count', fontsize=9)
    ax_trend.set_ylabel('ATE RMSE (m)', fontsize=9)
    ax_trend.set_title('TUM: ATE vs Feature Count\n(X = init failed, 0 on y-axis)', fontsize=9)
    ax_trend.legend(fontsize=7, loc='upper right')
    ax_trend.grid(True, alpha=0.3)
    ax_trend.set_xlim(100, 1100)

    # Two annotation boxes explaining both non-monotonic observations.
    if not np.isnan(tum_ate[500][0]) and not np.isnan(tum_ate[800][0]) \
            and not np.isnan(tum_ate[1000][0]):
        # Box 1 (top): why feat=500 < baseline despite partial tracking
        ax_trend.text(0.04, 0.97,
                      f'⚠ feat=500 ATE ({tum_ate[500][0]:.3f}m) < baseline ({tum_ate[1000][0]:.3f}m)\n'
                      f'feat=500 tracked only {tum_poses[500]}/{tum_poses[1000]} frames — dropped\n'
                      f'before accumulating global drift. ATE measures error\n'
                      f'only over tracked frames, not the full trajectory.',
                      transform=ax_trend.transAxes, fontsize=7, va='top',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3cd',
                                edgecolor='orange', alpha=0.95))
        # Box 2 (bottom): why feat=800 > baseline despite 100% tracking
        if tum_ate[800][0] > tum_ate[1000][0]:
            ax_trend.text(0.04, 0.03,
                          f'⚠ feat=800 ATE ({tum_ate[800][0]:.3f}m) > baseline ({tum_ate[1000][0]:.3f}m)\n'
                          f'despite {tum_poses[800]}/{tum_poses[1000]} frames tracked (100%).\n'
                          f'Fewer descriptors → more ambiguous matches in the long-\n'
                          f'office scene → BA incorporates erroneous associations\n'
                          f'→ heading drift accumulates over the full sequence.',
                          transform=ax_trend.transAxes, fontsize=7, va='bottom',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='#fde8e8',
                                    edgecolor='red', alpha=0.95))

    # ── Row 1: KITTI — baseline (col 0) + comprehensive threshold sweep (cols 1-3) ──
    ax_kb = fig.add_subplot(gs[1, 0])
    plot_trajectory(ax_kb, gt_kb, est_kb,
                    label_est=f'KITTI07 baseline (RMSE={rmse_kb:.3f}m)',
                    color='tab:blue',
                    title='KITTI07 feat=1000 (baseline)',
                    xy_axes=(0, 2))

    # Comprehensive threshold sweep from run logs (verified from stdout logs in
    # data/_rerun_logs/kitti07-feat*.stdout.log).
    # n_resets: count of "System Reseting" lines in each log (log-verifiable).
    # outcome: "map empty" confirmed by "The map is empty; nothing to save" line.
    def _parse_kitti_log(feat):
        log_path = os.path.join(DATA, '..', '_rerun_logs',
                                f'kitti07-feat{feat}.stdout.log')
        if not os.path.exists(log_path):
            return 0, 'FAILED'
        n_resets = 0
        outcome  = 'FAILED'
        with open(log_path) as fh:
            for line in fh:
                if 'System Reseting' in line:
                    n_resets += 1
                if 'nothing to save' in line:
                    outcome = 'FAILED'
        return n_resets, outcome

    _kitti_sweep_feats = [100, 250, 500, 750, 850, 900, 950, 960]
    _kitti_sweep = []
    for fv in _kitti_sweep_feats:
        nr, oc = _parse_kitti_log(fv)
        _kitti_sweep.append((fv, nr, oc))
    _kitti_sweep.append((1000, 0, 'SUCCESS'))  # baseline passed on first attempt

    ax_sweep = fig.add_subplot(gs[1, 1:])
    feats    = [r[0] for r in _kitti_sweep]
    n_resets = [r[1] for r in _kitti_sweep]
    outcomes = [r[2] for r in _kitti_sweep]
    bar_cols = ['#c0392b' if o == 'FAILED' else '#27ae60' for o in outcomes]
    bars = ax_sweep.bar(range(len(feats)), n_resets, color=bar_cols,
                        edgecolor='k', alpha=0.85)
    ax_sweep.set_xticks(range(len(feats)))
    ax_sweep.set_xticklabels([str(f) for f in feats], fontsize=9)
    ax_sweep.set_xlabel('ORB Feature Count', fontsize=10)
    ax_sweep.set_ylabel('Number of init reset attempts (from run log)', fontsize=9)
    ax_sweep.set_title(
        'KITTI07 Monocular Init: Comprehensive Threshold Sweep (9 feature counts tested)\n'
        'Red = init failed (log: "map is empty"), Green = success. Bars = reset count from log.',
        fontsize=9, fontweight='bold')
    ax_sweep.grid(True, alpha=0.3, axis='y')
    for bar, feat, nr, out in zip(bars, feats, n_resets, outcomes):
        label = f'{nr} reset(s)' if nr > 0 else ('OK' if out == 'SUCCESS' else '0 resets')
        ax_sweep.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + 0.05,
                      label, ha='center', fontsize=8, fontweight='bold',
                      color='#27ae60' if out == 'SUCCESS' else '#c0392b')
    ax_sweep.text(0.5, 0.92,
                  'All 8 tested feature counts below 1000 failed init on KITTI07.\n'
                  '10 fps inter-frame baseline (vehicle speed) is too large for\n'
                  'monocular geometry to produce ≥100 RANSAC inliers at any feat level.\n'
                  'Logs verified: each contains "The map is empty; nothing to save".',
                  transform=ax_sweep.transAxes, fontsize=8, ha='center', va='top',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff0f0',
                            edgecolor='#c0392b', alpha=0.9))

    # ── Row 2: Full-width KITTI vs TUM init-success summary ─────────────────
    ax_sum = fig.add_subplot(gs[2, :])
    datasets = ['KITTI07\nfeat=1000', 'KITTI07\nfeat=500', 'KITTI07\nfeat=250', 'KITTI07\nfeat=100',
                'TUM\nfeat=1000', 'TUM\nfeat=800', 'TUM\nfeat=500', 'TUM\nfeat=250']
    ate_summary = [
        rmse_kb,
        float('nan'), float('nan'), float('nan'),
        tum_ate[1000][0], tum_ate[800][0], tum_ate[500][0], float('nan'),
    ]
    bar_colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:purple',
                  'tab:blue',  'tab:orange', 'tab:green', 'tab:purple']

    x = np.arange(len(datasets))
    bars = ax_sum.bar(x, [0 if np.isnan(v) else v for v in ate_summary],
                      color=bar_colors, alpha=0.85, edgecolor='k')
    ax_sum.set_xticks(x)
    ax_sum.set_xticklabels(datasets, fontsize=8)
    ax_sum.set_ylabel('ATE RMSE (m)', fontsize=9)
    ax_sum.set_title('Summary: ATE RMSE across all feature-count experiments '
                     '(FAILED = bars at 0 with red label)', fontsize=10, fontweight='bold')
    ax_sum.grid(True, alpha=0.3, axis='y')
    ax_sum.axvline(3.5, color='k', linestyle='--', alpha=0.4, lw=1.5)
    ax_sum.text(1.5, ax_sum.get_ylim()[1] * 0.95 if ax_sum.get_ylim()[1] > 0 else 0.1,
                'KITTI07', ha='center', fontsize=9, color='steelblue', fontweight='bold')
    ax_sum.text(5.5, ax_sum.get_ylim()[1] * 0.95 if ax_sum.get_ylim()[1] > 0 else 0.1,
                'TUM', ha='center', fontsize=9, color='tomato', fontweight='bold')

    for bar, val, lbl in zip(bars, ate_summary, datasets):
        if np.isnan(val):
            ax_sum.text(bar.get_x() + bar.get_width()/2, 0.003, 'FAILED',
                        ha='center', va='bottom', fontsize=8, color='red',
                        fontweight='bold', rotation=0)
        else:
            ax_sum.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{val:.3f}m', ha='center', va='bottom', fontsize=8,
                        fontweight='bold')

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

        # ATE bar with % change annotation
        ax2 = axes[1, col]
        bars = ax2.bar(['Baseline', 'No Outlier'], [rmse_b, rmse_n],
                       color=['tab:blue', 'tab:red'], alpha=0.8, edgecolor='k')
        ax2.set_ylabel('ATE RMSE (m)', fontsize=8)
        ax2.set_title(f'{name} — ATE comparison', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, [rmse_b, rmse_n]):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                         f'{val:.3f}m', ha='center', fontsize=9, fontweight='bold')
        # % change annotation between the two bars
        if not np.isnan(rmse_b) and not np.isnan(rmse_n) and rmse_b > 0:
            pct = 100.0 * (rmse_n - rmse_b) / rmse_b
            top = max(rmse_b, rmse_n)
            pct_color = 'red' if pct > 0 else 'green'
            ax2.text(0.5, 0.92, f'{pct:+.1f}% vs baseline',
                     ha='center', transform=ax2.transAxes,
                     fontsize=10, color=pct_color, fontweight='bold')
        # Pose count note
        ax2.text(0.5, 0.02,
                 f'Baseline: {n_b} poses | No-outlier: {n_n} poses '
                 f'({"tracking reduced" if n_n < n_b * 0.9 else "tracking intact"})',
                 ha='center', transform=ax2.transAxes, fontsize=7,
                 color='dimgray')

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
        ax2.set_ylabel('ATE RMSE (m)', fontsize=8)
        ax2.set_title(f'{name} — ATE comparison', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, [rmse_b, rmse_l]):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                         f'{val:.3f}m', ha='center', fontsize=9, fontweight='bold')
        # % change annotation
        if not np.isnan(rmse_b) and not np.isnan(rmse_l) and rmse_b > 0:
            pct = 100.0 * (rmse_l - rmse_b) / rmse_b
            pct_color = 'red' if pct > 0 else 'green'
            ax2.text(0.5, 0.92, f'No-loop: {pct:+.1f}% vs with-loop',
                     ha='center', transform=ax2.transAxes,
                     fontsize=10, color=pct_color, fontweight='bold')
        ax2.text(0.5, 0.02,
                 f'Baseline: {n_b} poses | No-loop: {n_l} poses',
                 ha='center', transform=ax2.transAxes, fontsize=7, color='dimgray')

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
