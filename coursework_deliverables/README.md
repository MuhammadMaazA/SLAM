# COMP0222 — Visual and LiDAR SLAM Coursework

End-to-end Visual (ORB-SLAM2 / COLMAP) and 2D LiDAR SLAM pipeline with full
EVO evaluation, factor-graph pose-graph optimisation (GTSAM) and ray-cast
occupancy mapping.

## Directory layout

```
coursework_deliverables/
├── src/                                  # Python + shell entry points
│   ├── q1_evo_analysis.py                # Q1: ATE + RPE across KITTI/TUM
│   ├── q2_visual_slam_analysis.py        # Q2a/Q2c: custom-sequence ORB-SLAM2
│   ├── q2_pointcloud_3d.py               # Q2b: 3D COLMAP sparse renders
│   ├── q2b_evo_comparison.py             # Q2b: COLMAP vs ORB-SLAM2 (EVO)
│   ├── q3_lidar_slam_complete.py         # Q3b/Q3c/Q3d: LiDAR SLAM + PGO
│   ├── factor_graph_optimization.py      # Reference-only (synthetic demo)
│   └── run_orbslam_all.sh                # Batch ORB-SLAM2 runner (Linux)
├── data/
│   ├── part1_analysis/   # Q1: KITTI/TUM trajectory dumps + GT
│   ├── q2_results/       # Q2: ORB-SLAM2 + COLMAP outputs, comparison PNGs
│   └── q3_results/       # Q3: LiDAR plots & occupancy grids
├── docs/
│   ├── ORB_SLAM2_MODIFICATIONS.md        # Q1c/Q1d source diffs
│   └── ...
├── requirements.txt
└── README.md
```

## Install

Python ≥ 3.10 is required. All Python deps are pinned in
`requirements.txt`:

```bash
pip install -r requirements.txt
```

`gtsam` ships as a manylinux/macOS wheel; no additional build step is needed.
ORB-SLAM2 and COLMAP are external C++ binaries — see
`docs/ORB_SLAM2_MODIFICATIONS.md` for the exact ORB-SLAM2 build flags used in
Q1c / Q1d.

## Environment variables

Every script uses environment variables to locate raw data. Defaults point at
`./data/*` so running from the repository root works out of the box:

| Variable     | Purpose                               | Default                 |
|--------------|----------------------------------------|-------------------------|
| `SLAM_DATA`  | Root of per-question data folders      | `./data`                |
| `SLAM_OUT`   | Output directory override              | `./data/<q>_results`    |
| `SLAM_REC1`  | RPLidar recording set #1 (Q3)          | `/home/.../tmp_recordings`    |
| `SLAM_REC2`  | RPLidar recording set #2 (Q3)          | `/home/.../tmp_recordings2`   |

## Reproduce the figures

### One-shot (Linux, recommended)

`src/rerun_all.sh` is an idempotent driver that rebuilds every result from
raw data (ORB-SLAM2 runs, COLMAP runs, Python analysis, factor-graph PGO).
Edit the `USER CONFIG` block at the top of the script to point at your local
ORB-SLAM2 install / KITTI / TUM / D455 recordings, then:

```bash
chmod +x src/rerun_all.sh
./src/rerun_all.sh all                 # everything
./src/rerun_all.sh --dry-run all       # preview commands
./src/rerun_all.sh q1_evo_plots        # just the Python plots for Q1
./src/rerun_all.sh q2_colmap q2_python_plots q3_python    # pick steps
```

Per-step stdout/stderr is captured under `data/_rerun_logs/` and a
`.<step>.ok` marker is written on success.

**Full walkthrough (prerequisites, dataset layout, YAML configs, expected
runtime, common issues, sanity checks):** see
[`docs/RERUN_INSTRUCTIONS.md`](docs/RERUN_INSTRUCTIONS.md).

### Python-only (no external binaries)

If you only want to regenerate the analysis figures from the trajectory
dumps that are already in `data/`:

```bash
python src/q1_evo_analysis.py          # Q1 ATE + RPE (KITTI / TUM)
python src/q2_visual_slam_analysis.py  # Q2 per-sequence statistics & drift
python src/q2b_evo_comparison.py       # Q2b COLMAP vs ORB-SLAM2 (EVO)
python src/q2_pointcloud_3d.py         # Q2b 3D COLMAP point clouds
python src/q3_lidar_slam_complete.py   # Q3 LiDAR SLAM + Q3b/c/d
python src/factor_graph_optimization.py # Synthetic factor-graph demo
```

All scripts are idempotent — figures are regenerated in place.

## Q3 factor-graph backend

`optimize_pose_graph` in `src/q3_lidar_slam_complete.py` uses **GTSAM's
Levenberg-Marquardt** `Pose2` solver when the `gtsam` wheel is importable, and
transparently falls back to a `scipy.optimize.minimize(method='SLSQP')`
implementation otherwise (equality constraint anchors pose 0). The fallback
exists only so the script runs on platforms without a GTSAM build; the
coursework results were generated with GTSAM.

## Metric choices

- **ATE** is computed with `evo`'s Umeyama SE(3) alignment (`correct_scale=True`)
  and three `PoseRelation`s: `translation_part`, `rotation_angle_deg`,
  `full_transformation`.
- **RPE** uses `PoseRelation.translation_part`, `delta=1`, `delta_unit=frames`
  — i.e. the TUM-benchmark definition.
- **Closure error** for LiDAR sequences is the Euclidean distance between the
  first and last 2D pose, reported as a complement to ATE when no external
  ground-truth trajectory is available.

## What's not in this repo

- Raw RPLidar recordings (`tmp_recordings*/`) — too large to ship; referenced
  via `SLAM_REC1` / `SLAM_REC2`.
- KITTI / TUM raw images and groundtruth — downloaded at evaluation time; only
  the ORB-SLAM2 trajectory dumps are included.
- ORB-SLAM2 binary — see `docs/ORB_SLAM2_MODIFICATIONS.md` for the build
  recipe and source patches.
