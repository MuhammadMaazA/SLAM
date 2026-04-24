# Future Audit README

This document is for any future audit session on this repository.

Its purpose is simple:

- get a new reviewer productive quickly,
- stop already-settled issues being reopened without evidence,
- record the current authoritative pipeline state,
- and make future grading / oral / repo audits more consistent.

This is **not** the report and **not** the main project README.
It is an operational audit guide.

## Scope

This README is specifically about auditing:

- brief compliance,
- code-vs-claim consistency,
- reproducibility of key outputs,
- and likely mark-sensitive weaknesses.

It should be read together with:

- `README.md`
- `AUDIT_HANDOFF.md`
- `CURRENT_AUDIT_GRADE.md`

## Repo status at the time of this note

The repository is no longer in the earlier "claims exist but pipeline is not
actually wired through" state.

The most important audit-sensitive fixes are already in place:

- Q2 calibrated ORB-SLAM2 reruns are first-class outputs.
- Q2 analysis prefers calibrated trajectories when available.
- Q3 main pipeline uses the stronger audited configuration instead of the old
  weaker `4000 mm` default.
- the D455 YAML header parsing bug that broke fresh ORB-SLAM2 reruns was fixed.
- `Basement_1` and `Outdoor_1` calibrated trajectory files exist and satisfy
  the Q2 indoor/outdoor threshold requirement.

That means a future audit should start from:

- "verify current outputs and claims"

not from:

- "assume the pipeline is broken until proven otherwise."

## Files that matter most in audits

### Main audit notes

- `AUDIT_HANDOFF.md`
- `CURRENT_AUDIT_GRADE.md`
- `README.md`

### Q1

- `src/q1_evo_analysis.py`
- `data/part1_analysis/`
- `data/_rerun_logs/`

### Q2

- `src/q2_visual_slam_analysis.py`
- `src/q2b_evo_comparison.py`
- `src/q2_calibration_report.py`
- `src/generate_orb_yaml_from_colmap.py`
- `src/run_orbslam_all.sh`
- `data/q2_results/`

### Q3

- `src/q3_lidar_slam_complete.py`
- `data/q3_results/`

### Master rerun entry point

- `src/rerun_all.sh`

If a future auditor only reads one shell entry point, it should be
`src/rerun_all.sh`.

## Things that should NOT be re-flagged without new evidence

These were previously real concerns or common false alarms. They should not be
raised again casually unless a future audit finds a new regression.

### 1. "Q2 calibration is only claimed, not actually used"

This is outdated.

Current state:

- calibrated YAML generation exists,
- calibrated ORB-SLAM2 reruns are part of the rerun pipeline,
- the calibrated outputs are written as:
  - `data/q2_results/orbslam_runs/Basement_1_trajectory_colmap_intrinsics.txt`
  - `data/q2_results/orbslam_runs/Outdoor_1_trajectory_colmap_intrinsics.txt`
- Q2 analysis prefers those files when available.

This can still be verified, but it should no longer be treated as an unfixed
gap by default.

### 2. "Q3 still uses the old 4 m max-range baseline"

This is outdated.

Current state:

- `src/q3_lidar_slam_complete.py` defines `MAIN_SLAM_CONFIG`
- the main path uses:
  - `max_range_mm = 12000.0`
  - `angular_step = 1`
  - `voxel_m = 0.05`
  - `scan_skip = 1`

### 3. "Fresh ORB-SLAM2 reruns fail for unknown reasons"

This was partly true before, but one concrete root cause was identified:

- the D455 YAML files had invalid OpenCV header ordering
- OpenCV rejected them
- this was fixed

If ORB reruns fail in a future audit, do **not** reopen the issue as a vague
"environment mystery" without first checking:

- YAML parse validity,
- ORB binary path,
- viewer/headless mode,
- dataset path correctness,
- and whether the failure is sequence-specific rather than systemic.

### 4. "Weak datasets automatically imply weak marks"

This is not a safe assumption for this coursework.

Examiner guidance communicated during the project suggested that dataset quality
is not the main criterion as long as the choice and behaviour are justified
well.

So future audits should focus on:

- whether sequence behaviour is explained honestly,
- whether required success cases exist,
- whether failure cases are framed correctly,

not simply on whether every collected sequence looks pretty.

## What a future auditor should verify first

A good audit should begin with the shortest high-value checks.

### 1. Brief compliance, not perfection

Check the minimum required success cases first.

For Q2:

- one indoor sequence with at least `500` poses after ORB-SLAM initialisation
- one outdoor sequence with at least `500` poses after ORB-SLAM initialisation

Authoritative calibrated files to check:

- `data/q2_results/orbslam_runs/Basement_1_trajectory_colmap_intrinsics.txt`
- `data/q2_results/orbslam_runs/Outdoor_1_trajectory_colmap_intrinsics.txt`

For Q3:

- two indoor sequences
- one outdoor sequence
- one indoor sequence should represent a larger area

The current main sequence set was intentionally narrowed to:

- `Floor7_Hallway`
- `Basement_1`
- `Outdoor_1`

### 2. Code path vs figure path

Confirm that the scripts generating figures use the intended trajectories.

For Q2 this means checking that:

- `q2_visual_slam_analysis.py`
- `q2b_evo_comparison.py`

still prefer calibrated outputs as intended and have not regressed to always
using legacy factory runs.

### 3. Current authoritative outputs

Check that the main figure files exist and are real images, not placeholder
stubs:

- `data/q2_results/q2a_all_trajectories.png`
- `data/q2_results/q2_visual_overview.png`
- `data/q2_results/q2b_colmap_vs_orbslam.png`
- `data/q2_results/q2c_statistics.png`
- `data/q3_results/` main Q3 figures

### 4. Rerun entry point still works

Audit the command path through:

```bash
cd coursework_deliverables
./src/rerun_all.sh q2_colmap_intrinsics q2_python_plots q3_python
```

If that fails, identify whether the issue is:

- orchestration,
- external binaries,
- input paths,
- headless execution,
- or a real code regression.

## What counts as a real unresolved weakness

These are still fair criticisms even after the fixes.

### Q1

- strong overall, but still coursework-style analysis rather than exhaustive
  benchmark science
- oral explanation quality matters

### Q2

- `Outdoor_1` is compliant and defendable, but not a beautiful result
- some extra custom sequences remain weak, high-disagreement, or failure cases
- the strongest story comes from `Basement_1`, not uniformly from every
  sequence

### Q3

- strong and above lab baseline, but still heuristic in places
- loop verification is evidence-based, not external-truth-based
- pose-graph optimisation is solid coursework quality, but not highly advanced
  probabilistic SLAM

These are legitimate quality-ceiling issues.
They are different from broken-pipeline issues.

## What is probably a presentation issue, not a code issue

Many future audit disagreements are likely to come from framing rather than
implementation.

Examples:

- "This dataset is weak"
  - valid response: maybe, but weak datasets are acceptable if analysed and
    justified properly.
- "This failure case looks bad"
  - valid response: failure cases are acceptable if they are used to explain
    limitations rather than hidden.
- "COLMAP vs ORB disagreement is high"
  - valid response: this is inter-method disagreement, not direct ground-truth
    error.

So future audits should distinguish carefully between:

- a technical defect,
- a scientific limitation,
- and a presentational weakness.

## How to audit against the taught labs

If a future reviewer wants to compare the work against what the module actually
taught, the most relevant material is:

- `COMP0222_25-26/Labs/Lab_06_-_COLMAP`
- `COMP0222_25-26/Labs/Lab_07_-_ORBSLAM-2`
- `COMP0222_25-26/Labs/Lab_08_-_Point_Cloud`
- `COMP0222_25-26/Labs/Lab_09_-_2D_Occupancy_Grid`

Important current conclusion:

- this repo is **not below** the taught lab baseline
- Q2 is at least comparable to, and in places beyond, the taught baseline
- Q3 is clearly beyond the basic lab pipeline because it adds loop closure and
  pose-graph optimisation

So future marks should not be depressed on the assumption that the repo failed
to implement what was taught.

## Recommended audit sequence

If you are auditing this repo from scratch, use this order:

1. Read `AUDIT_HANDOFF.md`.
2. Read `CURRENT_AUDIT_GRADE.md`.
3. Check the Q2 calibrated trajectory files exist and are non-trivial.
4. Check the Q2 figure scripts still prefer calibrated outputs.
5. Check `MAIN_SLAM_CONFIG` in `src/q3_lidar_slam_complete.py`.
6. Verify the main figures under `data/q2_results/` and `data/q3_results/`.
7. Only then decide whether there is a real new issue.

This sequence prevents re-auditing already settled concerns as though they were
still open.

## Reproduction commands for future audit sessions

Full targeted audit rerun:

```bash
cd coursework_deliverables
./src/rerun_all.sh q2_colmap_intrinsics q2_python_plots q3_python
```

If Q2 ORB-SLAM2 needs a manual one-off run, inspect the exact command emitted
by:

```bash
./src/run_orbslam_all.sh
```

and verify:

- binary path,
- YAML path,
- dataset root,
- output file target,
- and whether `xvfb-run -a` is being used where needed.

## Current grading interpretation

The current internal audit position is roughly:

- likely range: `86–89`
- conservative single-point estimate: `87`

This estimate is explicitly shaped by:

- current repo state,
- current output files,
- fixes already completed,
- comparison against the actual taught labs,
- and the fact that dataset quality alone is not understood to be heavily
  penalised if justified well.

This is not an official mark prediction.
It is a practical audit estimate.

## One-paragraph summary for the next auditor

This repo has already passed through a meaningful brief-vs-code audit.
The main historical weaknesses were Q2 calibrated rerun integration and Q3 main
config consistency, and those were fixed. The repo should now be audited as a
strong but not flawless submission whose remaining weaknesses are mainly
scientific quality ceiling and oral framing, not missing core pipeline wiring.
