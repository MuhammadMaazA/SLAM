# COMP0222 — Visual and LiDAR SLAM — Group 32

**Submission:** `COMP0222_CW2_GRP_32.mp4` · `COMP0222_CW2_GRP_32.pdf` · `COMP0222_CW2_GRP_32.pptx` · `COMP0222_CW2_GRP_32.zip`

---

## Quick start — regenerate all figures and videos

```bash
pip install -r requirements.txt          # numpy scipy matplotlib scikit-learn evo gtsam opencv-python

# Q1 — EVO evaluation on KITTI07 + TUM
python3 src/q1_evo_analysis.py

# Q2 — Visual SLAM on custom sequences
python3 src/q2_visual_slam_analysis.py
python3 src/q2b_evo_comparison.py
python3 src/q2_calibration_report.py
python3 src/q2_pointcloud_3d.py

# Q3 — LiDAR SLAM (needs data at SLAM_REC1 / SLAM_REC2)
python3 src/q3_lidar_slam_complete.py

# Videos
python3 src/make_q2c_video.py
python3 src/make_q3e_video.py
python3 src/merge_coursework_videos.py
```

Environment variables (all optional — defaults point to `~/SLAM/extracted_data/...`):

| Variable | Purpose |
|---|---|
| `SLAM_REC1` | Path to first recording set (Basement_1, Floor7_Hallway, Outdoor_1, …) |
| `SLAM_REC2` | Path to second recording set (BikeStorage, OnePoolStreet1, …) |
| `SLAM_DATA` | Root of `data/` folder |
| `SLAM_OUT` | Output directory for figures |

---

## What the videos show

### Visual SLAM (`COMP0222_CW2_GRP_32_Visual_SLAM.mp4`, first 30 s of combined)

- **Top row:** live RealSense D455 camera feed synced to the sequence timestamp
- **Bottom row:** ORB-SLAM2 trajectory drawing itself pose-by-pose

**Why the trajectories don't look like perfect rectangles:**
ORB-SLAM2 monocular has no depth reference. It picks its coordinate frame
arbitrarily from the first triangulated frame pair — the scale and orientation
are both arbitrary. A 40 m basement walk may appear as 0.6 units on the plot.
This is a fundamental property of monocular SLAM, not a bug.

**Why Outdoor_1 looks compressed:**
ORB-SLAM2 lost tracking frequently outdoors (bright sky, limited texture).
Only ~584 poses were tracked from ~3,600 frames. The trajectory is all the
algorithm actually produced.

### LiDAR SLAM (`COMP0222_CW2_GRP_32_LiDAR_SLAM.mp4`, last 30 s of combined)

- **Colour point cloud:** accumulated LiDAR map building scan by scan (metric metres)
- **White line:** ICP odometry trajectory
- **Green circle:** start · **Red square:** current position

**Why LiDAR looks so much better than Visual SLAM:**
LiDAR measures real distances in millimetres. ICP has true metric constraints
on all sides of the room simultaneously. Visual SLAM with a single camera has
no depth reference at all.

**Why Outdoor_1 still drifts in LiDAR:**
The aperture/corridor problem (Lab 08 Activity 3B). Open outdoor space has
sparse geometry — no walls to lock onto, flat ground is ambiguous, sky returns
nothing. ICP cannot find a unique minimum for forward motion.

---

## Q1 — Visual SLAM on Benchmarks

### Results

| Config | KITTI 07 ATE | TUM ATE |
|---|---|---|
| Baseline (feat=1000) | **4.39 m** (0.6% of path) | **0.032 m** (0.1% of path) |
| feat=800 | FAILED init | 0.109 m |
| feat=500 | FAILED init | 0.015 m* |
| feat=250 | FAILED init | FAILED init |
| No outlier rejection | 5.76 m (+31%) | 0.474 m (+1381%) |
| No loop closure | 5.11 m (+16%) | 0.038 m (+19%) |

*feat=500 TUM ATE is lower than baseline but misleading — only 45% of frames
tracked. ATE is measured only over tracked frames; the hard late segment was
dropped.

### Why KITTI fails at all sub-1000 feature counts

ORB-SLAM2 monocular initialiser requires ≥100 RANSAC inliers to triangulate
the initial map. At 10 fps vehicle speed the inter-frame baseline is so large
that features scatter too far to match — this fails at every count from 100 to
960. Verified from run logs (`data/_rerun_logs/kitti07-feat*.stdout.log`):
every log ends with *"The map is empty; nothing to save"*. The Q1b figure now
shows log-verified reset counts (0 resets for ≤850, 2 for 900, 6 for 950, 5
for 960) rather than unverifiable map-point estimates.

### Why outlier rejection matters so much more on TUM than KITTI

Indoor scenes (TUM) have glass surfaces, reflective floors, and moving people
— all producing systematically wrong observations. The chi-squared gate
(5.991 monocular) rejects these before they corrupt keyframes. Outdoors
(KITTI) the scene is cleaner so removing the gate hurts less.

---

## Q2 — Visual SLAM on Custom Sequences

### Data collection (9 sequences, UCL Marshgate)

| Sequence | Type | Poses | Calibration | Notes |
|---|---|---|---|---|
| Basement_1 | Indoor | **868** | COLMAP calibrated ✅ | Primary indoor sequence |
| Outdoor_1 | Outdoor | **584** | COLMAP calibrated ✅ | Primary outdoor sequence |
| Floor7_Hallway | Indoor (large) | 617 | Factory intrinsics | COLMAP failed (8 poses, featureless corridor) |
| Basement_2 | Indoor | 2336 | Factory | Extra |
| Washroom | Indoor | 805 | Factory | Extra |
| BikeStorage | Outdoor | 1009 | Factory | Extra |
| Entrance2 | Outdoor | 1134 | Factory | Extra |
| OnePoolStreet1 | Outdoor | 5759 | Factory | Extra |
| BikeStorage2 | Outdoor | 114 | Factory | **FAIL** — night, no texture |

**Brief compliance:** requires 1 indoor + 1 outdoor with ≥500 calibrated poses.
Satisfied by Basement_1 (868 poses) and Outdoor_1 (584 poses).

### Why Floor7_Hallway uses factory intrinsics — and why that's fine

COLMAP recovered only 8 poses from Floor7_Hallway (the featureless Marshgate
corridor provided insufficient texture for feature matching). The resulting
"calibrated" YAML has garbage intrinsics (fx=1003, fy=1138 vs factory fx=426)
that would make ORB-SLAM2 worse. Factory D455 intrinsics are used instead,
cross-validated as accurate by 8 other COLMAP runs on the same camera
(all within 0.3–1.6% of factory values).

**Oral answer if asked:** *"COLMAP needs textured surfaces. The smooth concrete
corridor yielded only 8 reconstructed poses — insufficient for reliable
calibration. We use factory intrinsics validated by 8 other sequences on the
same camera."*

### COLMAP calibration results

| Sequence | COLMAP fx | Factory fx | Δ |
|---|---|---|---|
| Basement_1 (scratch, SIMPLE_PINHOLE) | 383.7 px | 426.7 px | ~10% |
| Basement_1 (OPENCV) | 427.1 px | 426.7 px | 0.1% |
| Outdoor_1 | 424.8 px | 426.7 px | 0.4% |
| Basement_2 | 425.4 px | 426.7 px | 0.3% |

The 10% deviation on the scratch run is expected: SIMPLE_PINHOLE forces equal
fx=fy which is slightly wrong for the D455. Full OPENCV model agrees with
factory within 0.5%.

### Q2b — COLMAP vs ORB-SLAM2 inter-method agreement

**CRITICAL:** The "ATE" reported here is NOT accuracy against ground truth.
No external ground truth exists for custom sequences. It is inter-method
agreement — how much do COLMAP and ORB-SLAM2 disagree after Umeyama Sim(3)
alignment with scale correction.

| Sequence | Trans RMSE | Relative (m/m) | Rot RMSE |
|---|---|---|---|
| Basement_1 | **132 mm** | 0.081 | 40.6° |
| Outdoor_1 | 1835 mm | 1.537 | 168.6° |
| Basement_2 | 3004 mm | 0.856 | 45.9° |
| Floor7_Hallway | — | — | COLMAP FAIL |
| BikeStorage | 2542 mm | 2.692 | 47.7° |

The safest interpretation is:

- `Basement_1` is the strongest indoor agreement case.
- `Outdoor_1` is brief-compliant but shows severe inter-method disagreement.
- These values are useful for discussing monocular consistency and failure
  modes, not for claiming metric accuracy against ground truth.

---

## Q3 — LiDAR SLAM on Custom Sequences

### Sequences (3 required by brief)

| Sequence | Type | Scans | Keyframes | Closure error |
|---|---|---|---|---|
| Basement_1 | Indoor | 1939 | 172 | 0.19 m |
| Floor7_Hallway | Indoor (large area) | 1304 | 135 | 0.16 m |
| Outdoor_1 | Outdoor | 1462 | 52* | 0.94 m |

*Outdoor ICP struggles with sparse geometry — see aperture problem below.

### ICP pipeline (what the code actually does)

1. Load RPLidar scan: `[(quality, angle_deg, dist_mm), ...]`
2. Filter: distance 10–12,000 mm, quality ≥5, blind spot 135–225° masked
3. Voxel centroid downsampling (default 0.05 m)
4. Point-to-plane ICP with Huber IRLS reweighting (matches Lab 08 Appendix C–D exactly)
5. PCA normals: k=5 neighbours, covariance → smallest eigenvector (matches Lab 08 Appendix D)
6. Divergence rejection: relative step >0.5 m or >25° rejected (uses relative transform dT, not absolute heading — bug fixed)
7. Keyframe: Δdist >0.2 m or Δangle >0.2 rad

### Q3b — Key parameter findings

| Parameter | Best | Worst | Ratio |
|---|---|---|---|
| Max range (Basement_1) | 12,000 mm → 0.19 m closure | 2,000 mm → 1.87 m | 10× |
| Max range (Floor7) | 12,000 mm → 0.16 m | 2,000 mm → 2.02 m | 13× |
| Angular (Basement_1) | Full → 0.19 m | Every 3rd → 0.52 m | 2.7× |
| Voxel (Basement_1) | 0.05 m → 0.19 m | 0.20 m → 0.48 m | 2.5× |

Sensor-max range (12,000 mm) is the single most impactful parameter. Short
range clips the wall geometry ICP needs for heading estimation.

### Q3c — Loop closure detection

Four-gate pipeline:
1. Temporal separation: j − i ≥ 10 keyframes
2. Adaptive arc gate: robot must have travelled ≥ min(8 m, 40% total arc)
3. Pose distance: ≤ 2.0 m from candidate anchor
4. ICP overlap score: ≥ 0.70

Threshold 0.70 is empirically justified by the score histogram: sliding-window
false matches score 0.45–0.60; confirmed revisits score 0.75–0.95. Clear
bimodal gap appears consistently across all three sequences.

NMS window of 15 keyframes collapses each physical revisit to one factor,
preventing the pose graph from being over-constrained by near-duplicate loops.

### Q3d — Factor graph optimisation (GTSAM)

| Sequence | Before | After | Reduction |
|---|---|---|---|
| Floor7_Hallway | 0.163 m | 0.061 m | **−62%** |
| Basement_1 | 0.194 m | 0.165 m | −15% |
| Outdoor_1 | 0.941 m | 0.886 m | −6% |

- Odometry edges: 3×3 information matrix from ICP Hessian × inlier count
- Loop edges: per-loop ICP Hessian propagated (not fixed sigma — bug fixed)
- Anchor prior: σ=1e-4 on pose 0
- GTSAM Levenberg-Marquardt, 200 iter, rel.tol=1e-8

**Why Floor7 benefits most (−62%):** well-separated loop revisit points
strongly constrain the drift direction. Factor graph can pull start/end
together cleanly.

**Why Outdoor benefits least (−6%):** noisy ICP in sparse outdoor geometry
produces loop constraints that don't precisely match odometry geometry.

### The corridor / aperture problem (why Floor7 and Outdoor drift)

In a featureless tunnel or open space, a scan at position X looks identical
to a scan at X+10 cm. The ICP error function is flat along that axis — the
algorithm cannot find a unique minimum. This is **Lab 08 Activity 3B** and is
a fundamental physical limitation, not a code bug.

**Oral answer:** *"Floor7_Hallway is a featureless Marshgate corridor — the
aperture problem means ICP can't estimate forward motion. This is why it
benefits most from factor-graph loop closure (−62%): the loop constraints
resolve the ambiguity the odometry couldn't."*

---

## Bugs fixed in this branch

| Bug | Impact | Fix |
|---|---|---|
| Step-angle divergence used absolute world headings — wraps at ±180° | Rejected valid ICP steps near ±180° heading | Use `dT_step = inv(prev_pose) @ pose`; extract angle from relative transform |
| Occupancy grid fixed at 25 m centred at (0,0) | Clipped large trajectories silently | `_grid_extent_from_trajectory()` auto-sizes to bounding box + 3 m margin |
| Lap detection used cumulative heading — fails for hairpin corridors | Detected 0 laps on corridor sequences | Replaced with proximity-based method (returns within 1.5 m of origin after ≥5 m arc) |
| Loop closure edges used fixed sigma regardless of ICP quality | PGO degraded Basement_1 by over-trusting weak loops | Per-loop ICP Hessian propagated through detection; each edge carries its own information matrix |
| Q1b KITTI sweep showed unverifiable `max_map_pts` values | Could not be verified from logs | Replaced with `_parse_kitti_log()` that reads actual reset counts from log files at plot time |
| Video used `max_range_mm=4000` (old lab default) | Trajectory looked broken; 108 spurious loops shown | Fixed to 12,000 mm + per-sequence outdoor ICP params |
| Video `MAX_SCANS=1500` cap cut second loop | Full two-loop path not visible | Removed cap (`None` = all scans) |

---

## Oral cheat-sheet — if the examiner asks…

**"Why does the visual SLAM trajectory look like irregular lines, not a rectangle?"**
Monocular SLAM has no depth reference. The coordinate frame, scale, and
orientation are picked arbitrarily from the first triangulated frame pair.
The trajectory shape is self-consistent but not metric. This is fundamental
to monocular SLAM — LiDAR avoids it because it measures real distances.

**"Why doesn't Floor7_Hallway have a calibrated trajectory?"**
COLMAP recovered only 8 poses (featureless corridor, insufficient texture).
The resulting calibrated YAML had fx=1003 vs factory fx=426 — clearly garbage.
Factory intrinsics are used, validated as accurate by 8 other COLMAP runs
on the same D455 camera (all within 1.6%). The brief requires calibration for
1 indoor + 1 outdoor — satisfied by Basement_1 and Outdoor_1.

**"Why did your Basement_1 PGO get worse in earlier runs?"**
The original code used a fixed sigma for all loop closure edges regardless of
ICP match quality. A borderline match at score=0.71 was trusted the same as
a perfect one at 0.97, causing the solver to over-constrain in the wrong
direction. Fixed by propagating the per-loop ICP Hessian as the information
matrix — weak loops are now automatically down-weighted.

**"Your KITTI feat=500 ATE is lower than baseline — does that mean fewer features is better?"**
No. feat=500 only tracked 45% of frames before losing tracking. It dropped
the hard later segment that causes drift, so the ATE is lower but measured
over an incomplete trajectory. It's a tracking failure that happens to look
good on paper.

**"Why did you choose 0.70 as the loop closure threshold?"**
The ICP score histogram shows a bimodal distribution across all three
sequences: false matches score 0.45–0.60, true revisits score 0.75–0.95.
There is a consistent gap at 0.70. The Q3c figures show this histogram as
direct evidence.

**"What is inter-method agreement and why not ground truth?"**
No external ground truth exists for our custom sequences (no motion capture,
no GPS). COLMAP is treated as the reference after Umeyama Sim(3) alignment
with scale correction. Large rotational disagreement means the two monocular
scales don't match after alignment — both trajectories can be internally
correct while disagreeing on scale.

---

## Directory layout

```
coursework_deliverables/
├── src/
│   ├── q1_evo_analysis.py            Q1: ATE/RPE on KITTI07 + TUM
│   ├── q2_visual_slam_analysis.py    Q2a/Q2c: all 9 sequences
│   ├── q2b_evo_comparison.py         Q2b: COLMAP vs ORB-SLAM2 EVO
│   ├── q2_calibration_report.py      Q2a: COLMAP intrinsics comparison
│   ├── q2_pointcloud_3d.py           Q2b: 3D sparse renders
│   ├── q3_lidar_slam_complete.py     Q3b/Q3c/Q3d: full LiDAR SLAM
│   ├── factor_graph_optimization.py  Synthetic PGO demo (reference)
│   ├── make_q2c_video.py             Visual SLAM video
│   ├── make_q3e_video.py             LiDAR SLAM video
│   ├── merge_coursework_videos.py    Combine both videos
│   ├── rerun_all.sh                  Master rebuild script
│   └── orbslam2_patches/             Q1c / Q1d source patches
├── data/
│   ├── part1_analysis/               Q1: KITTI/TUM trajectories + GT
│   ├── q2_results/                   Q2: ORB runs, COLMAP models, figures
│   └── q3_results/                   Q3: LiDAR SLAM figures
├── report/
│   └── COMP0222_CW2_GRP_32.tex      LaTeX report (compile with pdflatex ×2)
├── presentation/
│   └── COMP0222_CW2_GRP_32_presentation.html   HTML slides (open in browser)
├── RESULTS_GUIDE.md                  How to read every figure + oral answers
└── requirements.txt
```

---

## Dependencies

```
numpy scipy matplotlib scikit-learn opencv-python evo gtsam
```

GTSAM ≥4.3 required for Q3d factor graph. If unavailable, code falls back
to SciPy SLSQP (produces slightly different results).

```bash
pip install gtsam>=4.3   # or: pip install gtsam==4.3.0
```
