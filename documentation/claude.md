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
8. [Professor Announcements & Key Info](#8-professor-announcements--key-info)
9. [EVO Cheat Sheet](#9-evo-cheat-sheet)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. ENVIRONMENT & SETUP STATUS

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
| GTSAM (for Part 3) | TODO | needed for factor graph optimization |

### WSL2 Directory Structure
```
~/
├── ORB_SLAM2/                          # ORB-SLAM2 source + build
│   ├── Install/
│   │   ├── bin/                        # mono_tum, mono_kitti, mono_euroc, etc.
│   │   └── etc/orbslam2/              # YAML configs (TUM1.yaml, KITTI04-12.yaml, etc.)
│   ├── Source/
│   │   ├── src/                        # C++ source (Tracking.cc, Optimizer.cc, LoopClosing.cc, System.cc)
│   │   └── include/                    # Headers
│   ├── Evaluation/
│   │   └── kitti_to_tum.py            # KITTI→TUM trajectory format converter
│   └── Build.sh
│
├── SLAM/                               # Copied from Windows - course materials + datasets
│   ├── KITTI.zip                       # Contains KITTI dataset (sequence 00, poses 00-10)
│   ├── TUM.zip                         # Contains TUM fr1/xyz
│   ├── EuRoC.zip                       # EuRoC dataset
│   ├── UCL.zip                         # UCL Kenilworth dataset
│   ├── COMP0222_25-26/                 # Course repo (labs, courseworks)
│   │   └── Labs/
│   │       ├── Lab_08_-_Point_Cloud/   # ICP code for Part 3
│   │       └── Lab_09_-_2D_Occupancy_Grid/  # Occupancy grid code for Part 3
│   └── Week 1-8/                       # Lecture slides
│
├── SLAM_DATA/                          # Working directory for experiments
│   └── KITTI/                          # KITTI dataset (partially set up)
│       └── dataset/
│           ├── poses/                  # 00.txt - 10.txt (ground truth)
│           └── sequences/             # 00.zip (need to extract + get 07)
│
└── results/                            # TODO: create for saving all outputs
```

### Windows Side
```
C:\Users\mmaaz\University\Year 3\Simultaneous Localisation and Mapping\
├── Coursework 2\
│   ├── COMP0222_-_CW_02.pdf           # Official CW2 spec
│   ├── CW2_MASTER_GUIDE.md            # THIS FILE
│   ├── CW2_CONTEXT.md                 # Terminal commands reference
│   └── CHAT_LOG.md                    # Session 1 notes
├── KITTI.zip, TUM.zip, EuRoC.zip, UCL.zip  # Original dataset downloads
└── COMP0222_25-26/                     # Course GitHub repo
```

### Key Environment Commands
```bash
# Enter WSL
wsl

# Ensure ORB-SLAM2 is on PATH
export PATH=$PATH:$HOME/ORB_SLAM2/Install/bin

# Persist PATH (run once)
echo 'export PATH=$PATH:$HOME/ORB_SLAM2/Install/bin' >> ~/.bashrc

# IMPORTANT: Deactivate conda before building ORB-SLAM2
conda deactivate
```

---

## 2. FULL MARK BREAKDOWN

| Q | Task | Marks | Dependencies | Status |
|---|------|-------|-------------|--------|
| **1a** | Baseline evaluation (KITTI 07 + TUM) | 8 | ORB-SLAM2, EVO, datasets | TODO |
| **1b** | Vary ORB features (3 levels x 2 sequences) | 8 | Same as 1a | TODO |
| **1c** | Disable outlier rejection | 7 | Modify C++ source, rebuild | TODO |
| **1d** | Disable loop closure | 7 | Modify C++ source, rebuild | TODO |
| | **Part 1 Total** | **30** | | |
| **2a** | Own data collection (1 indoor + 1 outdoor) | 15 | Intel camera | TODO |
| **2b** | COLMAP vs ORB-SLAM2 comparison via EVO | 15 | COLMAP, ORB-SLAM2, EVO | TODO |
| **2c** | Video of ORB-SLAM2 reconstructing scenes | 3 | Screen recording | TODO |
| **2d** | Oral presentation (5 min) | - | All Part 2 results | TODO |
| | **Part 2 Total** | **33** | | |
| **3a** | LiDAR data collection (2 indoor + 1 outdoor) | 4 | RPLidar sensor | TODO |
| **3b1** | Max range variation (2 values) | 5 | Lab 8/9 code | TODO |
| **3b2** | Angular resolution (full vs n=2, n=3) | 5 | Lab 8/9 code | TODO |
| **3b3** | Voxel grid downsampling (2+ values) | 5 | Lab 8/9 code | TODO |
| **3b4** | Scan rate (skip odd, skip 2/3) | 5 | Lab 8/9 code | TODO |
| **3c** | Loop closure detection | 5 | Custom code | TODO |
| **3d** | Factor graph optimization (GTSAM/G2O) | 5 | GTSAM library | TODO |
| **3e** | Video of mapping results | 3 | Screen recording | TODO |
| **3f** | Oral presentation (5 min) | - | All Part 3 results | TODO |
| | **Part 3 Total** | **37** | | |
| | **COURSEWORK TOTAL** | **100** | | |

---

## 3. PART 1: VISUAL SLAM WITH DATASETS

### Parts 1, 2, and 3 are INDEPENDENT - can be done in any order.

### Dataset Status & Setup

**KITTI 07 (COMPULSORY):**
- The UCL OneDrive has KITTI data but only sequence 00 was included
- **Poses 00-10 are available** at `~/SLAM_DATA/KITTI/dataset/poses/` (07.txt exists)
- **Need KITTI 07 greyscale images** - either:
  - Check UCL OneDrive for a 07.zip: https://liveuclac-my.sharepoint.com/:f:/g/personal/ucacsjj_ucl_ac_uk/IgBQke4GC-8XQIpYcpM4j8hWAdyaaS4_rYaLRudWZhTtWTk?e=dB3bZa
  - Or download from official KITTI: https://www.cvlibs.net/datasets/kitti/eval_odometry.php (greyscale only, ~22GB for all)
- KITTI 07 uses config: `KITTI04-12.yaml`
- KITTI 07 has **a loop** in it - essential for testing loop closure (1d)

**Expected KITTI 07 structure:**
```
~/SLAM_DATA/KITTI/dataset/
├── sequences/07/
│   ├── image_0/        # ~1101 greyscale PNG images (000000.png to 001100.png)
│   ├── calib.txt       # Camera calibration
│   └── times.txt       # Timestamps
└── poses/07.txt        # Ground truth (11 sequences: 00-10)
```

**TUM Dataset:**
- Currently have: `fr1/xyz` (short, ~800 frames but simple scene)
- **Need a LONG sequence** for CW2: recommended `fr3/long_office_household` (~2500 frames)
- Download: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
- TUM fr1 uses `TUM1.yaml`, fr2 uses `TUM2.yaml`, fr3 uses `TUM3.yaml`
- Ground truth is in `groundtruth.txt` inside the dataset folder

**Expected TUM structure:**
```
~/SLAM_DATA/rgbd_dataset_freiburg3_long_office_household/
├── rgb/                # Color images
├── depth/              # Depth (not used for mono)
├── groundtruth.txt     # Ground truth poses
├── rgb.txt             # Image list with timestamps
└── associations.txt    # RGB-depth pairs
```

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
mono_tum TUM3.yaml rgbd_dataset_freiburg3_long_office_household tum-baseline.txt

evo_ape tum rgbd_dataset_freiburg3_long_office_household/groundtruth.txt \
  tum-baseline.txt -p -as \
  --save_plot results/tum_1a_baseline_ate.png \
  --save_results results/tum_1a_baseline.zip
```

**Report:** ATE plots, RMSE/mean/median/max values. Describe each dataset briefly (where recorded, length, characteristics). Interpret where errors are highest and why.

### Experiment 1b: Number of ORB Features [8 marks]

Modify `ORBextractor.nFeatures` in the YAML config. Default is 1000.

```bash
cd ~/SLAM_DATA

# Create working copies of configs
cp ~/ORB_SLAM2/Install/etc/orbslam2/KITTI04-12.yaml ./KITTI04-12_custom.yaml
cp ~/ORB_SLAM2/Install/etc/orbslam2/TUM3.yaml ./TUM3_custom.yaml

# Choose 3 levels, e.g.: 500, 200, 100
# For each level, edit YAML and run:

# --- 500 features ---
sed -i 's/ORBextractor.nFeatures:.*/ORBextractor.nFeatures: 500/' KITTI04-12_custom.yaml
mono_kitti KITTI04-12_custom.yaml KITTI/dataset/sequences/07 kitti07-feat500.txt

sed -i 's/ORBextractor.nFeatures:.*/ORBextractor.nFeatures: 500/' TUM3_custom.yaml
mono_tum TUM3_custom.yaml rgbd_dataset_freiburg3_long_office_household tum-feat500.txt

# --- 200 features ---
sed -i 's/ORBextractor.nFeatures:.*/ORBextractor.nFeatures: 200/' KITTI04-12_custom.yaml
mono_kitti KITTI04-12_custom.yaml KITTI/dataset/sequences/07 kitti07-feat200.txt

sed -i 's/ORBextractor.nFeatures:.*/ORBextractor.nFeatures: 200/' TUM3_custom.yaml
mono_tum TUM3_custom.yaml rgbd_dataset_freiburg3_long_office_household tum-feat200.txt

# --- 100 features ---
sed -i 's/ORBextractor.nFeatures:.*/ORBextractor.nFeatures: 100/' KITTI04-12_custom.yaml
mono_kitti KITTI04-12_custom.yaml KITTI/dataset/sequences/07 kitti07-feat100.txt

sed -i 's/ORBextractor.nFeatures:.*/ORBextractor.nFeatures: 100/' TUM3_custom.yaml
mono_tum TUM3_custom.yaml rgbd_dataset_freiburg3_long_office_household tum-feat100.txt

# Convert all KITTI outputs to TUM format, then evaluate each with evo_ape
# Compare all at once:
evo_traj tum kitti07-baseline-tum.txt kitti07-feat500-tum.txt kitti07-feat200-tum.txt kitti07-feat100-tum.txt \
  --ref=kitti07-gt-tum.txt -p -as --plot_mode xz
```

**Report:** 3 ATE plots per sequence (6 total). Table comparing RMSE across feature counts. Discuss:
- How fewer features → fewer correspondences → worse pose estimation
- At what point does tracking fail completely?
- Which dataset is more sensitive to feature count and why?

### Experiment 1c: Disable Outlier Rejection [7 marks]

**This requires modifying ORB-SLAM2 C++ source code.**

Where to look in `~/ORB_SLAM2/Source/src/`:
- **`Tracking.cc`** - search for outlier rejection after pose estimation
  - Look for: `bBad`, `SetBadFlag`, `EraseMapPoint`, RANSAC iterations
  - After `PoseOptimization()` there's typically a loop marking outliers as bad
- **`Optimizer.cc`** - in `PoseOptimization()` function
  - After optimization iterations, points with high reprojection error are marked as outliers
  - Look for: `e->chi2()` threshold checks, `pMP->SetBadFlag()`, `vpEdgesMono[i]->setLevel(1)`

**Strategy:** Find the code that discards map point observations based on reprojection error after optimization. Comment out the section that marks high-error points as outliers, so ALL matched points are kept regardless of error.

```bash
# After modifying source:
cd ~/ORB_SLAM2
conda deactivate
./Build.sh

# Run experiments
cd ~/SLAM_DATA
mono_kitti KITTI04-12.yaml KITTI/dataset/sequences/07 kitti07-nooutlier.txt
mono_tum TUM3.yaml rgbd_dataset_freiburg3_long_office_household tum-nooutlier.txt

# Evaluate with EVO (same pattern as 1a)
```

**Report:** Compare baseline vs no-outlier-rejection. Expect significantly worse performance. Discuss:
- Role of outliers in corrupting least-squares optimization
- How RANSAC protects against mismatched features
- Specific frames/regions where performance degrades most (use screenshots + timestamps)

### Experiment 1d: Disable Loop Closure [7 marks]

**Also requires C++ source modification.**

Where to look:
- **`System.cc`** - in the constructor, find where the loop closing thread is created:
  ```cpp
  mpLoopCloser = new LoopClosing(...);
  mptLoopClosing = new thread(&LoopClosing::Run, mpLoopCloser);
  ```
  Comment out the thread launch, OR:
- **`LoopClosing.cc`** - in the `Run()` method, add `return;` at the very beginning

**Strategy:** Simplest approach is adding `return;` at the top of `LoopClosing::Run()`.

```bash
# After modifying source:
cd ~/ORB_SLAM2
conda deactivate
./Build.sh

# Run experiments
cd ~/SLAM_DATA
mono_kitti KITTI04-12.yaml KITTI/dataset/sequences/07 kitti07-noloop.txt
mono_tum TUM3.yaml rgbd_dataset_freiburg3_long_office_household tum-noloop.txt

# Evaluate with EVO
```

**Report:** Compare baseline vs no-loop-closure. KITTI 07 has loops, so expect:
- Drift accumulation over time without loop closure corrections
- Trajectory endpoints not aligning with start despite returning to same location
- Discuss: how loop closure detects revisited places (DBoW2 bag of words), how it corrects drift (pose graph optimization)

### Part 1 Output Checklist
```
results/
├── kitti07_1a_baseline_ate.png
├── kitti07_1b_feat500_ate.png
├── kitti07_1b_feat200_ate.png
├── kitti07_1b_feat100_ate.png
├── kitti07_1c_nooutlier_ate.png
├── kitti07_1d_noloop_ate.png
├── tum_1a_baseline_ate.png
├── tum_1b_feat500_ate.png
├── tum_1b_feat200_ate.png
├── tum_1b_feat100_ate.png
├── tum_1c_nooutlier_ate.png
├── tum_1d_noloop_ate.png
├── *.zip (EVO result archives for each)
└── screenshots/ (ORB-SLAM2 GUI screenshots during runs)
```

---

## 4. PART 2: VISUAL SLAM WITH OWN SEQUENCES

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
# Use the script from the UCL dataset or ffmpeg directly:
ffmpeg -i your_video.mp4 -vf "scale=960:-1" -q:v 2 frames/%06d.png
# Create timestamp file matching TUM format
```

#### Step 3: Calibrate Camera with COLMAP
```bash
# Run COLMAP on a subset of frames for calibration
# Then run full COLMAP reconstruction
colmap automatic_reconstructor --workspace_path colmap_workspace --image_path frames/
```
- COLMAP outputs camera intrinsics (focal length, distortion) → convert to ORB-SLAM2 YAML format
- COLMAP also outputs camera trajectory → use for comparison with ORB-SLAM2

#### Step 4: Run ORB-SLAM2 on Your Sequences
```bash
mono_tum your_calibration.yaml your_dataset_folder output_trajectory.txt
```

#### Step 5: Compare COLMAP vs ORB-SLAM2 Trajectories
```bash
# Convert COLMAP output to TUM format
# Then compare with EVO:
evo_ape tum colmap_trajectory.txt orbslam_trajectory.txt -p -as
```

### What to Report (2a - 15 marks)
- Data acquisition strategy: why you chose these locations, camera settings, motion pattern
- What worked and what didn't (failed initializations, lost tracking, etc.)
- Camera calibration process and resulting parameters
- Challenges and how you addressed them

### What to Report (2b - 15 marks)
- EVO comparison plots (COLMAP as reference, ORB-SLAM2 as estimate)
- 3D reconstruction visualizations from both systems
- Discussion of differences: where does ORB-SLAM2 diverge from COLMAP and why?
- RMSE/ATE statistics

### Video (2c - 3 marks)
- Show the raw recorded videos
- Screen-capture of ORB-SLAM2 running and reconstructing the scene
- Visualization of 3D point clouds and camera trajectories
- Think of videos that accompany research papers

---

## 5. PART 3: LIDAR SLAM WITH OWN SEQUENCES

### Requirements
- **3 sequences:** 2 indoor (one must be large area like Marshgate hallways) + 1 outdoor
- Use **RPLidar** sensor (borrow at lab sessions)
- Each sequence: **start at marker → 2 complete loops → return to exact start**
- Mark the floor for consistency, remove marker after

### Code Base
Built on Lab 8 and Lab 9 code:
- **Lab 8** `rplidar_icp.py`: ICP scan matching, keyframe management, point cloud mapping
  - Key params: ICP_MAX_ITER=10, CORRESPONDENCE_THRESH=0.5m, KEYFRAME_DIST/ANGLE_THRESH
  - Blind spot filtering (270° swath)
  - KD-tree for nearest neighbor, PCA normals, point-to-plane ICP
- **Lab 9** `occupancy_grid_map_odometry.py`: Probabilistic occupancy grid with Monte Carlo pose optimization
  - 800x800 cells @ 20mm/cell = 16m x 16m area
  - PoseEstimator with grid search optimization

**Lab code locations in WSL:**
```
~/SLAM/COMP0222_25-26/Labs/Lab_08_-_Point_Cloud/Solution/rplidar_icp.py
~/SLAM/COMP0222_25-26/Labs/Lab_09_-_2D_Occupancy_Grid/Code/occupancy_grid_map_odometry.py
~/SLAM/COMP0222_25-26/Labs/Lab_09_-_2D_Occupancy_Grid/Code/occupancy_grid_map_still_lidar.py
```

### Recording Data
```python
# Use rplidar_recorder_example.py or rplidar_driver.py in live mode
# Data saved as JSON: [[quality, angle_degrees, distance_mm], ...]
# Record at each location, save with descriptive names:
#   indoor_small_room.json
#   indoor_large_hallway.json
#   outdoor_courtyard.json
```

### Experiment 3b1: Maximum Range [5 marks]
Test 2 values: sensor max rated range AND a reduced value.
- RPLidar A1 max range: ~12m; try 12m and 6m
- Filter out points beyond max range threshold before ICP

### Experiment 3b2: Angular Resolution [5 marks]
Compare full scan vs downsampled:
- Full scan (all beams)
- Every 2nd beam (n=2)
- Every 3rd beam (n=3)
```python
# Downsample: scan_points = scan_points[::n]
```

### Experiment 3b3: Voxel Grid Downsampling [5 marks]
Vary voxel filter size (at least 2 values):
- e.g., 0.05m and 0.1m voxel size
- Affects point cloud density and ICP performance

### Experiment 3b4: Scan Rate [5 marks]
Reduce data frequency:
- Skip every odd scan (use scans 0, 2, 4, ... = 50% reduction)
- Skip 2 out of 3 scans (use scans 0, 3, 6, ... = 67% reduction)
```python
# In main loop: if scan_index % 2 != 0: continue  (for 50%)
# Or: if scan_index % 3 != 0: continue  (for 67%)
```

### Experiment 3c: Loop Closure Detection [5 marks]
Develop automatic detection of when robot returns to a visited location.

**Approach ideas:**
- Compare current scan against stored keyframe scans using ICP fitness score
- If ICP converges with very low error against a non-recent keyframe → loop closure detected
- Thresholds to avoid false positives: minimum time/distance since last visit, minimum ICP score
- Provide evidence: graphs of ICP score over time showing spikes at loop closures

### Experiment 3d: Factor Graph Optimization [5 marks]
Integrate odometry + loop closures into a factor graph.

```bash
# Install GTSAM
pip install gtsam
```

**What to do:**
- Create a pose graph: each pose is a node, odometry provides edges between consecutive poses
- Loop closure adds an edge between non-consecutive poses
- Optimize with GTSAM/G2O
- Show before/after comparison of trajectory and occupancy grid
- Quantify: Euclidean distance between start and end pose before vs after optimization

### Part 3 Output Checklist
```
results/
├── For each of 3 sequences, for each parameter variation:
│   ├── point_cloud_map.png
│   ├── occupancy_grid.png
│   └── trajectory.png
├── loop_closure_detection_graph.png
├── factor_graph_before_after.png
└── closure_error_table.txt
```

---

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

### At Each Location, Collect:
1. **Camera video** (Intel RealSense): continuous stream, 500+ frames, slow steady motion
2. **LiDAR scan** (RPLidar): mark start point, walk exactly 2 loops, return to start
3. **Notes**: sketch of environment, dimensions, notable features, start/end position

### Data Collection Checklist
```
[ ] Location 1 (indoor small):
    [ ] Camera video recorded
    [ ] LiDAR data recorded (2 loops, returned to start)
    [ ] Environment notes/sketch

[ ] Location 2 (indoor large - e.g. Marshgate hallways):
    [ ] LiDAR data recorded (2 loops, returned to start)
    [ ] Environment notes/sketch

[ ] Location 3 (outdoor):
    [ ] Camera video recorded
    [ ] LiDAR data recorded (2 loops, returned to start)
    [ ] Environment notes/sketch

[ ] Camera calibration frames recorded (checkerboard or COLMAP calibration set)
[ ] All sensors returned on time (2% penalty if late!)
```

### Lab 10: RealSense Camera
- Lab 10 instructions (tentative) already on GitHub
- Covers interfacing with the Intel RealSense sensor
- Needed for Part 2 data collection

---

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
- Cannot re-run code during presentation - use submitted results only
- Date TBA

### Report Structure (like a research paper Results section)
1. Introduction / Setup
2. Part 1 results (dataset experiments)
3. Part 2 results (own visual SLAM)
4. Part 3 results (own LiDAR SLAM)
5. Analysis and discussion
- Include: EVO plots, RMSE tables, screenshots, parameter comparison tables

---

## 8. PROFESSOR ANNOUNCEMENTS & KEY INFO

### From Moodle (Simon Julier)

**11 March 2026 - Lab 09 and CW2:**
- Lab 09: 2D occupancy grids with RPLidar
- CW2 released, available in GitHub
- Will NOT support sensor work outside lab times
- Will NOT answer CW2 questions outside labs/lectures

**11 March 2026 - Lab 10 (Tentative):**
- RealSense camera lab uploaded early to GitHub
- Instructions for interfacing with Intel RealSense sensor

**4 March 2026 - CW2 Details:**
- Deadline moved to 27 April (start of next term)
- Includes presentation component (in person)
- Sensors only available during lab days
- NOT available over holiday or before submission deadline

**25 February 2026 - Lab 07 (ORB-SLAM2):**
- Data on UCL OneDrive: https://liveuclac-my.sharepoint.com/:f:/g/personal/ucacsjj_ucl_ac_uk/IgBQke4GC-8XQIpYcpM4j8hWAdyaaS4_rYaLRudWZhTtWTk?e=dB3bZa

**24 February 2026 - Lab prep:**
- ORB-SLAM2 skills required for CW2
- Works on WSL2, Linux, Mac

### Key Links
- ORB-SLAM2 UCL fork: https://github.com/UCL/COMP0222-249_25-26_ORB_SLAM2
- UCL OneDrive (datasets): https://liveuclac-my.sharepoint.com/:f:/g/personal/ucacsjj_ucl_ac_uk/IgBQke4GC-8XQIpYcpM4j8hWAdyaaS4_rYaLRudWZhTtWTk?e=dB3bZa
- KITTI odometry: https://www.cvlibs.net/datasets/kitti/eval_odometry.php
- TUM RGB-D: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
- EVO: https://github.com/MichaelGrupp/evo

---

## 9. EVO CHEAT SHEET

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

# Key flags:
#   -p              plot results
#   -as             align and scale (ESSENTIAL for mono SLAM - no absolute scale)
#   -a              align only (for stereo/RGBD which have scale)
#   --plot_mode     xy, xz, yz, xyz (xz for top-down view)
#   --save_plot     save figure as image
#   --save_results  save metrics to zip for later comparison
#   --no_warnings   suppress alignment warnings

# KITTI format → TUM format conversion
python3 ~/ORB_SLAM2/Evaluation/kitti_to_tum.py input.txt times.txt output.txt
```

---

## 10. TROUBLESHOOTING

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

### WSL2 GUI Issues
- For Pangolin/matplotlib display, you may need an X server (VcXsrv, WSLg)
- WSLg should work by default on Windows 11
- If plots don't show: `export DISPLAY=:0` or save plots to files instead of displaying

### Dataset Issues
- KITTI outputs in KITTI format → must convert to TUM with `kitti_to_tum.py`
- TUM datasets are already in TUM format
- EuRoC needs: `evo_traj euroc data.csv --save_as_tum` to convert

---

## TIMELINE / PRIORITY

### NOW (Week 8-9): Part 1 - No hardware needed
1. Get KITTI 07 images (check OneDrive or download)
2. Get TUM long sequence
3. Run all 4 experiments (1a-1d)
4. Save all trajectories and EVO plots

### NEXT LAB SESSIONS (Wed/Fri before term ends): Data Collection
5. Borrow Intel camera + RPLidar
6. Collect Part 2 + Part 3 data at same locations
7. Return sensors on time

### AFTER DATA COLLECTION: Parts 2 & 3 Processing
8. Process camera data: COLMAP calibration + reconstruction, ORB-SLAM2, EVO comparison
9. Process LiDAR data: parameter experiments, loop closure, factor graph
10. Create videos

### FINAL WEEK BEFORE DEADLINE: Polish
11. Write report (2-column LaTeX)
12. Prepare presentation slides (5 min Part 2 + 5 min Part 3)
13. Package source code (no build files)
14. Final video editing (merge Part 2 + Part 3 videos)
