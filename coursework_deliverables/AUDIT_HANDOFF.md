# Audit Handoff

This document is for the next person who audits, extends, or defends this
coursework repository. It records which audit concerns were already known,
which of them had already been fixed, and which additional code changes were
made so the *main pipeline* matches the claims made in the README and oral
defence.

## Why this file exists

The recurring audit risk in this repo was not "do we have a nice explanation?"
but "does the code path that regenerates the submitted outputs actually do the
thing the report says it does?"

The main historical gap was Q2:

- a genuine COLMAP scratch calibration existed,
- per-sequence COLMAP intrinsics could be converted to ORB-SLAM2 YAMLs,
- but the standard batch runner and main Q2 comparison scripts still primarily
  consumed the legacy factory-YAML trajectories.

This handoff makes that status explicit and documents the fix.

## Issues that were already fixed before this handoff

- Q1b KITTI failure panels cite the actual rerun logs.
- Q3b uses the correct RPLidar A1 sensor max range: `12000 mm`.
- Q3c/Q3d are using the current loop-closure return signature consistently.
- `BikeStorage2` is already treated as a failure case in the Q2 narrative.
- A genuine COLMAP scratch calibration exists at:
  `data/q2_results/colmap_runs/Basement_1_scratch_sparse/cameras.txt`
  with `SIMPLE_PINHOLE 848 480 383.70464006076338 424 240`.

## Issues addressed in this handoff

### 1. Q2 calibrated ORB-SLAM2 reruns are now first-class outputs

Files changed:

- `src/run_orbslam_all.sh`
- `src/rerun_all.sh`

What changed:

- `run_orbslam_all.sh` now supports:
  - `YAML_DIR_COLMAP`
  - `Q2_USE_COLMAP_YAMLS=1`
  - `OUT_SUFFIX`
- When `Q2_USE_COLMAP_YAMLS=1`, the batch runner looks for
  `data/q2_results/orbslam_colmap_yaml/{sequence}_D455_colmap.yaml`
  and uses that as the primary ORB-SLAM2 settings file for the sequence.
- The calibrated reruns are saved as:
  `data/q2_results/orbslam_runs/{sequence}_trajectory_colmap_intrinsics.txt`

`rerun_all.sh` step `q2_colmap_intrinsics` now:

1. regenerates the ORB-SLAM2 YAMLs from COLMAP `cameras.txt`,
2. batch-reruns ORB-SLAM2 across the custom sequences using those YAMLs,
3. writes the calibrated trajectory files as normal reproducible outputs.

This closes the earlier "calibration exists only as a claim" gap.

Additional runtime fix made during the audit:

- the repo fallback D455 YAMLs in `data/part1_analysis/` had comments before
  `%YAML:1.0`, which OpenCV rejected as an invalid file;
- those YAML headers were corrected so fresh ORB-SLAM2 reruns can parse them;
- `run_orbslam_all.sh` was also hardened for headless environments by using
  `xvfb-run -a` when available.

### 2. Q2 figures now prefer calibrated reruns automatically

Files changed:

- `src/q2_visual_slam_analysis.py`
- `src/q2b_evo_comparison.py`

What changed:

- Added environment control `Q2_TRAJ_VARIANT`.
- Default is now `prefer_calibrated`.
- Both scripts first look for:
  `{sequence}_trajectory_colmap_intrinsics.txt`
- If that file is missing, they fall back to:
  `{sequence}_trajectory.txt`

This means the default Q2 analysis path now reflects the best audited
available outputs while remaining backward-compatible with older worktrees.

In `q2b_evo_comparison.py`, the selected ORB trajectory filename is also drawn
onto each subplot so an examiner can see whether a panel came from the
calibrated rerun or the legacy factory-YAML run.

### 3. Q3 main pipeline now uses the audited stronger config

File changed:

- `src/q3_lidar_slam_complete.py`

What changed:

- Added:

```python
MAIN_SLAM_CONFIG = {
    'max_range_mm': 12000.0,
    'angular_step': 1,
    'voxel_m': 0.05,
    'scan_skip': 1,
}
```

- The main Q3 pipeline now runs:

```python
result = run_slam(scans, **MAIN_SLAM_CONFIG)
```

instead of the older hard-coded `max_range_mm=4000.0`.

Why:

- Q3b already showed that the historical 4 m default was not the strongest
  general setting for the submitted sequences.
- The audit concern was that the repo "discovers" better settings but still
  generates Q3a/Q3c/Q3d from the weaker baseline.

This fix aligns the main Q3 outputs with the audited best-general settings.

## Reproduction commands

Recommended audit reproduction sequence:

```bash
cd coursework_deliverables
./src/rerun_all.sh q2_colmap q2_colmap_scratch q2_colmap_intrinsics q2_python_plots q3_python
```

If you only want the calibrated Q2 reruns and updated figures:

```bash
cd coursework_deliverables
./src/rerun_all.sh q2_colmap_intrinsics q2_python_plots
```

## Known runtime caveat

The calibrated Q2 rerun path is now wired correctly in code, but one important
distinction remains:

- **pipeline issue fixed:** the repo now generates per-sequence COLMAP YAMLs,
  batch-reruns ORB-SLAM2 against them, and the Q2 analysis scripts prefer the
  resulting calibrated trajectories when present;
- **runtime reproducibility still environment-dependent:** during one audit run
  inside the current Codex sandbox, the fresh ORB-SLAM2 reruns initially failed
  because the repo fallback YAMLs were invalid for OpenCV parsing. That header
  issue is now fixed, and manual headless runs reached real map creation with
  both factory and calibrated YAMLs.

What this means in practice:

- the code path is no longer the weak point;
- the original hard failure mode was identified and fixed;
- if `*_trajectory_colmap_intrinsics.txt` files are still absent after a rerun,
  the next thing to investigate is the ORB-SLAM2 runtime environment itself
  rather than the Python / shell orchestration.

The expected successful outputs are:

- `data/q2_results/orbslam_runs/Basement_1_trajectory_colmap_intrinsics.txt`
- `data/q2_results/orbslam_runs/Outdoor_1_trajectory_colmap_intrinsics.txt`

and any other custom-sequence reruns that track successfully under the
calibrated-first policy.

If those files do not appear, check:

- whether the ORB-SLAM2 binary being used is the same one that originally
  produced the legacy `*_trajectory.txt` files;
- whether its expected runtime libraries / viewer dependencies are available;
- whether the camera-directory CLI convention still matches the installed fork
  of `mono_tum`.

## Honest wording to keep in the oral

These fixes improve the pipeline, but the following statements should still be
made precisely:

- The `Basement_1` scratch calibration is a genuine no-prior calibration using
  `SIMPLE_PINHOLE`.
- Q2b's reported "ATE" is inter-method disagreement, not ground-truth error.
- Q3a loop verification is still heuristic evidence based on heading, laps, and
  closure error, not external motion-capture truth.

## One-paragraph defence summary

If asked whether we knew about the audit gaps:

- yes, we explicitly audited the repo for "claim vs actual code path"
  mismatches;
- we promoted COLMAP-calibrated ORB-SLAM2 reruns into the main Q2 pipeline;
- we updated the Q2 analysis scripts to prefer those reruns by default;
- we updated the main Q3 pipeline to use the stronger audited settings instead
  of the old historical 4 m baseline.

That is the purpose of this file.
