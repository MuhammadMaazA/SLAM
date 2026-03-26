# COMP0222 Coursework 02 - MASTER GUIDE
# Visual and Lidar SLAM - Complete Reference

> **DEADLINE:** Monday 27 April 2026
> **WEIGHT:** 50% of module
> **FORMAT:** Group work, oral presentation + report
> **This file is the single source of truth for all CW2 work.**

---

## TABLE OF CONTENTS
1. [Environment & Setup Status](#1-environment--setup-status)
2. [Full Mark Breakdown](#2-full-mark-breakdown)
3. [Part 1: Visual SLAM with Datasets (30 marks)](#3-part-1-visual-slam-with-datasets)
4. [Part 2: Visual SLAM with Own Sequences (33 marks)](#4-part-2-visual-slam-with-own-sequences)
5. [Part 3: LiDAR SLAM with Own Sequences (37 marks)](#5-part-3-lidar-slam-with-own-sequences)
6. [Data Collection Plan](#6-data-collection-plan)
7. [Deliverables & Submission](#7-deliverables--submission)
8. [EVO Cheat Sheet](#8-evo-cheat-sheet)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. ENVIRONMENT & SETUP STATUS ✅ COMPLETE

### WSL2 (Ubuntu-24.04) - Primary Work Environment
All ORB-SLAM2 work MUST be done in WSL2 (Windows native build broken).

| Component | Status | Location |
|-----------|--------|----------|
| WSL2 Ubuntu-24.04 | READY | Default distro |
| cmake 3.28, g++ 13.3 | INSTALLED | system |
| eigen3, OpenCV | INSTALLED | apt |
| ORB-SLAM2 (UCL fork) | BUILT | `~/ORB_SLAM2/` |
| ORB-SLAM2 executables | READY | `~/ORB_SLAM2/Install/bin/` |
| EVO | INSTALLED | pip |
| COLMAP | TODO | `sudo apt install colmap` |

### Key Environment Commands
```bash
# Enter WSL
wsl

# Ensure ORB-SLAM2 is on PATH
export PATH=$PATH:$HOME/ORB_SLAM2/Install/bin

# IMPORTANT: Deactivate conda before building ORB-SLAM2
conda deactivate
```

## 2. FULL MARK BREAKDOWN

| Q | Task | Marks | Dependencies | Status |
|---|------|-------|-------------|--------|
| **1a** | Baseline evaluation (KITTI 07 + TUM) | 8 | ORB-SLAM2, EVO, datasets | READY |
| **1b** | Vary ORB features (3 levels x 2 sequences) | 8 | Same as 1a | READY |
| **1c** | Disable outlier rejection | 7 | Modify C++ source, rebuild | READY |
| **1d** | Disable loop closure | 7 | Modify C++ source, rebuild | READY |
| | **Part 1 Total** | **30** | | |
| **2a** | Own data collection (1 indoor + 1 outdoor) | 15 | Intel camera | TODO |
| **2b** | COLMAP vs ORB-SLAM2 comparison via EVO | 15 | COLMAP, ORB-SLAM2, EVO | TODO |
| **2c** | Video of ORB-SLAM2 reconstructing scenes | 3 | Screen recording | TODO |
| | **Part 2 Total** | **33** | | |
| **3a** | LiDAR data collection (2 indoor + 1 outdoor) | 4 | RPLidar sensor | TODO |
| **3b1** | Max range variation (2 values) | 5 | Lab 8/9 code | TODO |
| **3b2** | Angular resolution (full vs n=2, n=3) | 5 | Lab 8/9 code | TODO |
| **3b3** | Voxel grid downsampling (2+ values) | 5 | Lab 8/9 code | TODO |
| **3b4** | Scan rate (skip odd, skip 2/3) | 5 | Lab 8/9 code | TODO |
| **3c** | Loop closure detection | 5 | Custom code | TODO |
| **3d** | Factor graph optimization (GTSAM/G2O) | 5 | GTSAM library | TODO |
| **3e** | Video of mapping results | 3 | Screen recording | TODO |
| | **Part 3 Total** | **37** | | |
| | **COURSEWORK TOTAL** | **100** | | |

## 3. PART 1: VISUAL SLAM WITH DATASETS (30 marks) - READY TO START

### Dataset Status & Setup

**KITTI 07 (COMPULSORY):**
- ❌ **Need KITTI 07 greyscale images** - Download from: https://www.cvlibs.net/datasets/kitti/eval_odometry.php
- Download: "**Download odometry data set (grayscale, 22 GB)**" 
- ✅ Ground truth poses available at `~/SLAM_DATA/KITTI/dataset/poses/07.txt`
- KITTI 07 uses config: `KITTI04-12.yaml`
- KITTI 07 has **a loop** in it - essential for testing loop closure (1d)

**TUM Dataset:**
- ✅ Currently have: `fr1/xyz` (798 frames - sufficient for coursework)
- ✅ Ground truth available in `groundtruth.txt`
- ✅ Successfully tested with ORB-SLAM2

### Experiment 1a: Baseline Evaluation [8 marks]

Run ORB-SLAM2 monocular with default settings on both sequences.

```bash
cd ~/SLAM_DATA

# === KITTI 07 ===
mono_kitti KITTI04-12.yaml KITTI/dataset/sequences/07 kitti07-baseline.txt

# Convert KITTI ground truth to TUM format for EVO
python3 ~/ORB_SLAM2/Evaluation/kitti_to_tum.py \
  KITTI/dataset/poses/07.txt \
  KITTI/dataset/sequences/07/times.txt \
  kitti07-gt-tum.txt

# Convert ORB-SLAM2 output to TUM format
python3 ~/ORB_SLAM2/Evaluation/kitti_to_tum.py \
  kitti07-baseline.txt \
  KITTI/dataset/sequences/07/times.txt \
  kitti07-baseline-tum.txt

# Evaluate ATE
evo_ape tum kitti07-gt-tum.txt kitti07-baseline-tum.txt \
  -p -as --plot_mode xz \
  --save_plot results/kitti07_1a_baseline_ate.png \
  --save_results results/kitti07_1a_baseline.zip

# === TUM ===
mono_tum TUM1.yaml rgbd_dataset_freiburg1_xyz tum-baseline.txt

evo_ape tum rgbd_dataset_freiburg1_xyz/groundtruth.txt \
  tum-baseline.txt -p -as \
  --save_plot results/tum_1a_baseline_ate.png \
  --save_results results/tum_1a_baseline.zip
```

### Experiment 1b: Number of ORB Features [8 marks]

Modify `ORBextractor.nFeatures` in YAML config. Default is 1000.

```bash
# Create working copies of configs
cp ~/ORB_SLAM2/Install/etc/orbslam2/KITTI04-12.yaml ./KITTI04-12_custom.yaml
cp ~/ORB_SLAM2/Install/etc/orbslam2/TUM1.yaml ./TUM1_custom.yaml

# Choose 3 levels: 500, 200, 100
# For each level, edit YAML and run:

# --- 500 features ---
sed -i 's/ORBextractor.nFeatures:.*/ORBextractor.nFeatures: 500/' KITTI04-12_custom.yaml
mono_kitti KITTI04-12_custom.yaml KITTI/dataset/sequences/07 kitti07-feat500.txt

sed -i 's/ORBextractor.nFeatures:.*/ORBextractor.nFeatures: 500/' TUM1_custom.yaml
mono_tum TUM1_custom.yaml rgbd_dataset_freiburg1_xyz tum-feat500.txt

# Convert all outputs to TUM format, then evaluate each with evo_ape
```

### Experiment 1c: Disable Outlier Rejection [7 marks]

**This requires modifying ORB-SLAM2 C++ source code.**

Where to look in `~/ORB_SLAM2/Source/src/`:
- **`Tracking.cc`** - search for outlier rejection after pose estimation
- **`Optimizer.cc`** - in `PoseOptimization()` function
- Look for: `bBad`, `SetBadFlag`, `EraseMapPoint`, RANSAC iterations, `e->chi2()` threshold checks

```bash
# After modifying source:
cd ~/ORB_SLAM2
conda deactivate
./Build.sh

# Run experiments
cd ~/SLAM_DATA
mono_kitti KITTI04-12.yaml KITTI/dataset/sequences/07 kitti07-nooutlier.txt
mono_tum TUM1.yaml rgbd_dataset_freiburg1_xyz tum-nooutlier.txt
```

### Experiment 1d: Disable Loop Closure [7 marks]

**Also requires C++ source modification.**

Where to look:
- **`LoopClosing.cc`** - in the `Run()` method, add `return;` at the very beginning

```bash
# After modifying source:
cd ~/ORB_SLAM2
conda deactivate
./Build.sh

# Run experiments  
cd ~/SLAM_DATA
mono_kitti KITTI04-12.yaml KITTI/dataset/sequences/07 kitti07-noloop.txt
mono_tum TUM1.yaml rgbd_dataset_freiburg1_xyz tum-noloop.txt
```

## 4. PART 2: VISUAL SLAM WITH OWN SEQUENCES (33 marks)

### Requirements
- 2 monocular video sequences: **1 indoor + 1 outdoor**
- Each must be **500+ frames** after ORB-SLAM2 initializes
- Use the **Intel RealSense camera** (borrow at lab sessions)
- Capture as a **continuous video stream** (not individual photos)

### Workflow

#### Step 1: Record Videos
- Steady, slow camera motion (avoid fast rotation)
- Good lighting, textured surfaces
- Include some overlap/revisiting for loop closure
- Aim for 30+ seconds of video at 30fps = 900+ frames

#### Step 2: Extract Frames & Create Timestamps
```bash
ffmpeg -i your_video.mp4 -vf "scale=960:-1" -q:v 2 frames/%06d.png
```

#### Step 3: Calibrate Camera with COLMAP
```bash
colmap automatic_reconstructor --workspace_path colmap_workspace --image_path frames/
```

#### Step 4: Run ORB-SLAM2 on Your Sequences
```bash
mono_tum your_calibration.yaml your_dataset_folder output_trajectory.txt
```

#### Step 5: Compare COLMAP vs ORB-SLAM2 Trajectories
```bash
evo_ape tum colmap_trajectory.txt orbslam_trajectory.txt -p -as
```

## 5. PART 3: LIDAR SLAM WITH OWN SEQUENCES (37 marks)

### Requirements
- **3 sequences:** 2 indoor (one must be large area like Marshgate hallways) + 1 outdoor
- Use **RPLidar** sensor (borrow at lab sessions)
- Each sequence: **start at marker → 2 complete loops → return to exact start**

### Code Base
Built on Lab 8 and Lab 9 code:
- **Lab 8** `rplidar_icp.py`: ICP scan matching, keyframe management, point cloud mapping
- **Lab 9** `occupancy_grid_map_odometry.py`: Probabilistic occupancy grid with Monte Carlo pose optimization

### Experiments

#### 3b1: Maximum Range [5 marks]
Test 2 values: sensor max rated range AND a reduced value.
- RPLidar A1 max range: ~12m; try 12m and 6m

#### 3b2: Angular Resolution [5 marks]
Compare full scan vs downsampled:
- Full scan (all beams)
- Every 2nd beam (n=2)
- Every 3rd beam (n=3)

#### 3b3: Voxel Grid Downsampling [5 marks]
Vary voxel filter size (at least 2 values):
- e.g., 0.05m and 0.1m voxel size

#### 3b4: Scan Rate [5 marks]
Reduce data frequency:
- Skip every odd scan (50% reduction)
- Skip 2 out of 3 scans (67% reduction)

#### 3c: Loop Closure Detection [5 marks]
Develop automatic detection when robot returns to visited location.

#### 3d: Factor Graph Optimization [5 marks]
Integrate odometry + loop closures into a factor graph with GTSAM.

```bash
# Install GTSAM
pip install gtsam
```

## 6. DATA COLLECTION PLAN

### CRITICAL: Collect ALL sensor data before term ends!
- Sensors available **ONLY during lab hours** (Wednesday + Friday)
- **NOT available over holiday or before submission deadline**
- **Each group must collect their own data** (shared data = plagiarism)

### Recommended: Collect Part 2 + Part 3 data TOGETHER
The CW spec recommends using the **same environments** for camera and LiDAR to allow comparison.

### Locations to Choose (need 3 total)

| # | Type | Example | Used For |
|---|------|---------|----------|
| 1 | Indoor small | Lab room, office | Part 2 indoor + Part 3 indoor #1 |
| 2 | Indoor large | Marshgate hallways (REQUIRED for 3) | Part 3 indoor #2 |
| 3 | Outdoor | Courtyard, car park | Part 2 outdoor + Part 3 outdoor |

## 7. DELIVERABLES & SUBMISSION

### Files to Submit
| File | Format | Naming |
|------|--------|--------|
| Report | PDF (2-column LaTeX recommended) | `COMP0222_CW2_GRP_g.pdf` |
| Video | MP4 (Part 2 + Part 3 combined) | `COMP0222_CW2_GRP_g.mp4` |
| Presentation | PPT/KEY/PDF | `COMP0222_CW2_GRP_g.[ppt/key/pdf]` |
| Source code | ZIP (no build files/executables) | `COMP0222_CW2_GRP_g.zip` |

- **2% penalty for wrong file names**
- **2% penalty for late equipment return**
- Replace `g` with your group code

### Oral Presentation
- **Total: 10 minutes** (timed and recorded)
  - 5 minutes on Part 2 (Visual SLAM own sequences)
  - 5 minutes on Part 3 (LiDAR SLAM)
- Followed by **15-20 minutes of questions**
- ALL group members must present

## 8. EVO CHEAT SHEET

```bash
# === ESSENTIAL: Always use -as for monocular SLAM (align + scale) ===

# Absolute Pose Error (ATE/APE) - PRIMARY METRIC for this CW
evo_ape tum groundtruth.txt estimated.txt -p -as \
  --save_plot output.png --save_results output.zip

# Relative Pose Error (RPE) - secondary, good to include
evo_rpe tum groundtruth.txt estimated.txt -p -as

# Visualize trajectories
evo_traj tum estimated.txt --ref=groundtruth.txt -p -as --plot_mode xz

# Compare multiple trajectories
evo_traj tum baseline.txt feat500.txt feat200.txt feat100.txt \
  --ref=groundtruth.txt -p -as --plot_mode xz

# Compare saved results (generates comparison tables)
evo_res result1.zip result2.zip result3.zip -p --save_table comparison.csv

# KITTI format → TUM format conversion
python3 ~/ORB_SLAM2/Evaluation/kitti_to_tum.py input.txt times.txt output.txt
```

## 9. TROUBLESHOOTING

### ORB-SLAM2 Build Issues
- **Always deactivate conda first:** `conda deactivate`
- **Display errors (Pangolin):** `export DISPLAY=:0`
- If build fails, check: `sudo apt install libepoxy-dev libgl1-mesa-glx libsuitesparse-dev`

### ORB-SLAM2 Runtime Issues
- **Tracking lost:** too few features, fast motion, textureless surfaces
- **No initialization:** need sufficient parallax (camera must translate, not just rotate)
- **Non-deterministic:** runs may vary - run important experiments 2-3 times

### EVO Issues
- **Scale mismatch:** always use `-as` for monocular
- **Timestamp mismatch:** ensure both files use same timestamp format
- **Empty trajectory:** ORB-SLAM2 may not output poses for frames where tracking was lost

---

## CURRENT STATUS & NEXT STEPS

### ✅ COMPLETED:
- ORB-SLAM2 built and working in WSL2
- TUM dataset ready (798 frames - sufficient)
- EVO installed and tested
- Ground truth poses for KITTI 07 available
- All Part 1 experiments planned and ready

### 🚨 IMMEDIATE NEED:
- **Download KITTI sequence 07 grayscale images (22 GB)**
- From: "Download odometry data set (grayscale, 22 GB)" on KITTI website

### 🎯 READY TO START:
- Can begin Experiment 1a with TUM dataset immediately
- All experiments 1a-1d are ready once KITTI 07 is downloaded
- Code modification locations identified for experiments 1c and 1d