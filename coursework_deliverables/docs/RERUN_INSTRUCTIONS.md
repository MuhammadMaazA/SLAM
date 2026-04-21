# Linux Rerun Instructions

End-to-end recipe for regenerating every coursework figure from raw data on a
Linux (Ubuntu 20.04 / 22.04) machine using `src/rerun_all.sh`.

> The macOS box you are writing this on cannot run ORB-SLAM2 (X11 viewer +
> Pangolin), and COLMAP is much faster on Linux with a CUDA GPU. Do this on
> the same Ubuntu box where ORB-SLAM2 is already installed.

---

## 1. Prerequisites

### 1.1 System packages

```bash
sudo apt update
sudo apt install -y \
    build-essential cmake git pkg-config \
    libgl1-mesa-dev libglew-dev libpython3-dev python3-dev \
    libeigen3-dev libboost-all-dev libsuitesparse-dev \
    libopencv-dev \
    colmap \
    python3-venv python3-pip
```

### 1.2 Python environment

```bash
cd ~/SLAM/coursework_deliverables   # adjust to your clone path
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # numpy, scipy, sklearn, matplotlib, evo, gtsam
```

Verify:

```bash
python -c "import gtsam, evo.core.metrics as m; print('OK')"
# → OK
```

### 1.3 ORB-SLAM2 (three build variants required for Q1c/Q1d)

Follow `docs/ORB_SLAM2_MODIFICATIONS.md` to apply the Q1c/Q1d source patches
and build **three** ORB-SLAM2 variants:

```bash
export ORBSLAM_SRC=$HOME/ORB_SLAM2
cd $ORBSLAM_SRC

# Baseline — unmodified upstream
cmake -B build_baseline -DCMAKE_BUILD_TYPE=Release
cmake --build build_baseline -j

# Q1c — outlier rejection disabled
cmake -B build_nooutlier -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-DDISABLE_OUTLIER_REJECTION"
cmake --build build_nooutlier -j

# Q1d — loop closure disabled
cmake -B build_noloop -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-DDISABLE_LOOP_CLOSURE"
cmake --build build_noloop -j
```

Each build produces `Examples/Monocular/{mono_kitti, mono_tum}`. Keep them in
their build directories — the driver script will point at the right binaries
per step.

### 1.4 Datasets

Layout expected by the default `USER CONFIG` block:

```
$HOME/SLAM/
├── datasets/
│   ├── kitti/07/image_2/                              # KITTI07 grayscale
│   ├── kitti/07/times.txt
│   ├── rgbd_dataset_freiburg1_xyz/
│   │   ├── rgb/                                       # PNGs
│   │   ├── rgb.txt                                    # timestamps
│   │   └── groundtruth.txt
│   └── rgbd_dataset_freiburg3_long_office_household/  # Q1 "long" sequence
│       ├── rgb/
│       ├── rgb.txt
│       └── groundtruth.txt
├── extracted_data/
│   ├── tmp_recordings/tmp_recordings/
│   │   ├── Basement_1/{camera/, lidar/scans.jsonl}
│   │   ├── Basement_2/…
│   │   └── …
│   └── tmp_recordings2/
│       ├── BikeStorage/…
│       └── …
└── configs/
    ├── KITTI04-12_custom.yaml
    ├── KITTI04-12_custom_f1200.yaml
    ├── KITTI04-12_custom_f1500.yaml
    ├── TUM1_custom.yaml
    ├── TUM1_custom_f800.yaml
    ├── TUM1_custom_f1200.yaml
    ├── TUM1_custom_f1500.yaml
    ├── TUM3_long.yaml
    ├── RealSense_D455.yaml
    └── RealSense_D455_lowthresh.yaml
```

> Feat-count YAMLs differ only in `ORBextractor.nFeatures`. If you don't have
> them yet, copy the baseline YAML and edit that single line.

TUM downloads: `https://cvg.cit.tum.de/rgbd/dataset/download`.
KITTI odometry: `https://www.cvlibs.net/datasets/kitti/eval_odometry.php`.

---

## 2. Configuring `rerun_all.sh`

Open `src/rerun_all.sh` and edit the **USER CONFIG** block near the top if
your paths differ from the defaults. Every option is an env var with a
default, so you can also override per-invocation without editing:

```bash
ORBSLAM_ROOT=$HOME/my_orbslam \
KITTI_SEQ_DIR=/mnt/data/kitti/07 \
./src/rerun_all.sh q1_kitti
```

Key variables:

| Variable                          | Meaning                                    |
|-----------------------------------|--------------------------------------------|
| `ORBSLAM_ROOT`                    | ORB-SLAM2 install prefix (`bin/`, `share/`) |
| `ORBSLAM_BIN_BASELINE` / `_NOOUT` / `_NOLOOP` | Three KITTI mono binaries            |
| `ORBSLAM_BIN_TUM_{BASELINE,NOOUT,NOLOOP}` | Three TUM mono binaries              |
| `ORBSLAM_VOCAB`                   | Path to `ORBvoc.txt`                       |
| `KITTI_SEQ_DIR`                   | KITTI07 image_2 directory                  |
| `TUM_XYZ_DIR`, `TUM_LONG_DIR`     | TUM RGB-D root folders                     |
| `SLAM_REC1`, `SLAM_REC2`          | D455 recording roots                       |
| `KITTI_YAML`, `TUM_XYZ_YAML*`, …  | Per-experiment YAML files                  |
| `PYTHON`                          | Python interpreter (default `python3`)     |
| `DRY_RUN=1` or `--dry-run`        | Print commands only, don't execute         |

---

## 3. Running

### 3.1 Full rebuild

```bash
cd ~/SLAM/coursework_deliverables
source .venv/bin/activate
./src/rerun_all.sh all
```

Expected runtime on a 6-core CPU + single GPU:

| Step                    | Typical wall time |
|-------------------------|-------------------|
| `q1_kitti` (5 runs)     | 15–25 min         |
| `q1_tum_xyz` (6 runs)   | 5–10 min          |
| `q1_tum_long` (3 runs)  | 15–30 min         |
| `q1_evo_plots`          | <10 s             |
| `q2_orbslam` (9 seq)    | 10–20 min         |
| `q2_colmap`  (9 seq)    | 1–4 hours (GPU)   |
| `q2_colmap_intrinsics`  | 1–3 min           |
| `q2_python_plots`       | <30 s             |
| `q3_python`             | 5–10 min          |
| **Total**               | **2–6 hours**     |

### 3.2 Picking individual steps

```bash
./src/rerun_all.sh q1_evo_plots                           # Python plots only
./src/rerun_all.sh q3_python                              # LiDAR + PGO only
./src/rerun_all.sh q2_orbslam q2_colmap q2_python_plots   # rebuild Q2
```

### 3.3 Preview without running

```bash
./src/rerun_all.sh --dry-run all
```

Shows every shell command that would execute. Useful to check paths before a
long run.

### 3.4 Resuming after a failure

Each successful step writes `data/_rerun_logs/.<step>.ok`. If a step fails,
re-run just that step — dependencies are all self-contained. Python steps are
idempotent (they overwrite the output figure); ORB-SLAM2 steps only overwrite
when the new run produces a non-empty trajectory.

---

## 4. What to check after a full run

### 4.1 Expected file layout

```
data/
├── part1_analysis/
│   ├── kitti07-{baseline,feat1200,feat1500,nooutlier,noloop}.txt
│   ├── tum-{baseline,feat800,feat1200,feat1500,nooutlier,noloop}.txt
│   ├── tum_long-{baseline,nooutlier,noloop}.txt
│   ├── kitti07-gt-tum.txt
│   ├── rgbd_dataset_freiburg1_xyz/groundtruth.txt
│   └── rgbd_dataset_freiburg3_long_office_household/groundtruth.txt
├── q1_results/
│   └── q1{a,b,c,d,_summary,_rpe}.png
├── q2_results/
│   ├── orbslam_runs/*_trajectory.txt                  (9 files + 1 intrinsics)
│   ├── colmap_runs/*_{colmap_poses.txt, _sparse/}     (7–9 reconstructions)
│   ├── pointclouds_3d/q2b_pointcloud_*.png            (7 figures)
│   ├── q2_summary.png
│   ├── q2a_all_trajectories.png
│   ├── q2b_colmap_vs_orbslam.png                      (from q2b_evo_comparison.py)
│   ├── q2b_summary_table.png                          (from q2_visual_slam_analysis.py)
│   ├── q2c_statistics.png
│   └── RealSense_D455_colmap_Basement_1.yaml
├── q3_results/
│   ├── q3b_{seq}.png                                  (9 sequences)
│   ├── q3b_grids_{seq}_{group}.png                    (4 × 9 = 36 grids)
│   ├── q3c_{seq}.png
│   ├── q3d_{seq}.png
│   ├── q3d_grid_{seq}.png                             (before/after occupancy)
│   ├── occupancy_{seq}.png
│   └── factor_graph_demo_synthetic.png
└── _rerun_logs/
    ├── .<step>.ok                                     (per-step success markers)
    └── *.stdout.log                                   (per-run ORB-SLAM2 output)
```

### 4.2 Console sanity numbers

Run the Python plots and watch for roughly these values:

| Metric                                      | Expected range        |
|---------------------------------------------|-----------------------|
| KITTI07 baseline ATE RMSE                   | **0.5–5 m**           |
| KITTI07 no-loop ATE RMSE                    | 15–25 m               |
| KITTI07 no-outlier pose count               | ~500 (vs 1096 base)   |
| TUM freiburg1_xyz baseline ATE RMSE         | 0.01–0.05 m           |
| TUM freiburg3_long baseline ATE RMSE        | 0.05–0.15 m           |
| Q3b closure error (Basement_1, default)     | 0.5–1.5 m             |
| Q3d closure error improvement               | ≥ 30 %                |

If the KITTI07 baseline is above **1 m**, the YAML intrinsics are almost
certainly wrong (this is the single most common failure mode — see
§ 5 below).

---

## 5. Common issues

### 5.1 KITTI07 ATE is too high

**Symptom**: baseline RMSE of 4+ m, while published numbers are ~0.5 m.

**Cause**: wrong intrinsics or wrong image dimensions in `KITTI04-12_custom.yaml`.

**Fix**: KITTI sequence 07 is 1226×370. Reference intrinsics from the
dataset's `calib.txt` (camera 0, P0):

```
fx = 718.856,  fy = 718.856,  cx = 607.1928,  cy = 185.2157,  baseline = 0
```

Verify your YAML matches this. Also check that `Camera.RGB = 0` (KITTI is
grayscale).

### 5.2 Empty KeyFrameTrajectory.txt

**Symptom**: ORB-SLAM2 completes but the trajectory file is empty or missing.

**Cause**: tracking never initialised — usually too few features or too little
parallax in the first seconds.

**Fix**: try the `RealSense_D455_lowthresh.yaml` fallback (already referenced
in the D455 runs), or pre-process by skipping the first N frames. For custom
sequences, make sure the camera is *moving* (not static or rotating in place)
during the first ~30 frames.

### 5.3 COLMAP hangs on `exhaustive_matcher`

**Cause**: too many images, matching is O(N²).

**Fix**: `run_colmap_all.py` uses `sequential_matcher` with an overlap
window. Verify no one has switched it to `exhaustive`. For >2000 images,
consider increasing `overlap` from 20 to 40.

### 5.4 GTSAM import error

**Symptom**: `_HAVE_GTSAM = False` printed at script start.

**Fix**: the wheel on PyPI requires `libboost-system1.74+`. On Ubuntu 20.04:

```bash
pip install --upgrade pip
pip install gtsam
# If that still fails, build from source:
#   git clone https://github.com/borglab/gtsam && ...
```

The fallback SciPy SLSQP optimiser still runs if GTSAM is unimportable — the
script will print a warning and numerical results may be slightly worse.

### 5.5 Q3 `SKIP: Data not found`

**Cause**: `SLAM_REC1` / `SLAM_REC2` env vars point somewhere that doesn't
contain `<sequence>/lidar/scans.jsonl`.

**Fix**:

```bash
export SLAM_REC1=/mnt/data/tmp_recordings/tmp_recordings
export SLAM_REC2=/mnt/data/tmp_recordings2
ls $SLAM_REC1/Basement_1/lidar/scans.jsonl   # must exist
./src/rerun_all.sh q3_python
```

### 5.6 `permission denied` on `rerun_all.sh`

```bash
chmod +x src/rerun_all.sh
```

---

## 6. Sanity checks before submission

After all steps finish, run these quick checks:

```bash
# 1. All expected trajectory dumps exist
ls data/part1_analysis/*.txt | wc -l    # ≥ 14 + ground-truths

# 2. All expected figures exist
find data -name '*.png' | wc -l          # ≥ 30

# 3. No mandatory figure is missing
for f in q1_results/q1a_baseline.png q1_results/q1_rpe.png \
         q2_results/q2a_all_trajectories.png \
         q2_results/q2b_colmap_vs_orbslam.png \
         q3_results/q3b_basement_1.png \
         q3_results/q3d_grid_basement_1.png; do
    [[ -f data/$f ]] && echo "OK  $f" || echo "MISSING $f"
done

# 4. Python scripts still import cleanly
python -c "import sys; sys.path.insert(0, 'src'); \
           import q1_evo_analysis, q2_visual_slam_analysis, \
                  q2b_evo_comparison, q2_pointcloud_3d, \
                  q3_lidar_slam_complete, factor_graph_optimization; \
           print('all imports OK')"
```

If all four checks pass, the results directory is ready for the report.

---

## 7. Speeding things up

- **Q2 COLMAP** is the slowest step by a wide margin. If iterating, comment
  out sequences you don't need in `src/run_colmap_all.py`.
- **`q2_orbslam`** runs nine sequences serially; the bottleneck is the
  ORB-SLAM2 viewer. Launch the binary with the `--headless` flag (if your
  build supports it) or edit the call to pass `false` for the `bUseViewer`
  constructor argument in `mono_tum.cc` and rebuild.
- **Q3 Python** scales with total LiDAR scan count. To iterate on factor-graph
  logic alone, add `[:500]` to the scan slice in `run_slam()` temporarily.

---

## 8. Presentation-ready outputs

Most marking rubrics want a small number of "hero" figures:

| Brief question | Recommended hero figure |
|----------------|-------------------------|
| Q1a            | `data/q1_results/q1a_baseline.png` |
| Q1b            | `data/q1_results/q1b_features.png` |
| Q1c            | `data/q1_results/q1c_outlier.png` |
| Q1d            | `data/q1_results/q1d_loop.png` |
| Q1 summary     | `data/q1_results/q1_summary.png` + `q1_rpe.png` |
| Q2a            | `data/q2_results/q2a_all_trajectories.png` |
| Q2b (2D)       | `data/q2_results/q2b_colmap_vs_orbslam.png` |
| Q2b (3D maps)  | `data/q2_results/pointclouds_3d/q2b_pointcloud_*.png` |
| Q2c            | `data/q2_results/q2c_statistics.png` |
| Q3b            | `data/q3_results/q3b_<seq>.png` and `q3b_grids_*_*.png` |
| Q3c            | `data/q3_results/q3c_<seq>.png` |
| Q3d            | `data/q3_results/q3d_<seq>.png` + `q3d_grid_<seq>.png` |

These are the ones to embed in the report PDF and on any poster / slides.
