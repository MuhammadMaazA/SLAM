# COMP0222 CW2 — Critical Code & Results Review

**Overall Grade: 65 / 100**

---

## Question 1: Visual SLAM with Datasets (30 marks)

### What Exists
- Both KITTI 07 and TUM freiburg1_xyz were run through ORB-SLAM2 (monocular).
- Trajectory files exist for all four conditions (baseline, nooutlier, noloop, feature variants).
- EVO API is used correctly in `q1_evo_analysis.py` — `sync.associate_trajectories`, `APE`, `correct_scale=True` are all proper.
- CSV of ATE RMSE values is present with correct statistics.

---

### Q1a — Baseline Evaluation [8 marks]

- Both sequences run, ATE RMSE values computed, trajectory plots generated.
- The `compute_ate` function does `correct_scale=True` which is correct for monocular.
- KITTI baseline RMSE = 4.39m, TUM = 0.0098m — reasonable numbers.

**Score: 6 / 8**

---

### Q1b — Feature Count [8 marks] — ⚠️ Methodology Error

The brief says *"Reduce the number of ORB features."* Looking at the YAML files and code:

- **KITTI 07:** Baseline = 2000 features → tested 1500, 1200, 750. Direction is correct. However `feat=750` completely fails to initialise, so only 2 usable numerical comparisons exist. The brief asks for "3 graphs per sequence."
- **TUM:** `TUM1_custom.yaml` sets `ORBextractor.nFeatures: 400`. The analysis code labels this as "400 (baseline)" and then tests `feat800`, `feat1200`, `feat1500` — which are all **increases** from the baseline, not reductions. The brief explicitly says reduce. This is the wrong experimental direction for the TUM dataset, and the ATE barely changes (0.0098 → 0.0102m) because adding more features to a simple TUM sequence has negligible effect. The insight is trivial.

**Score: 5 / 8**

---

### Q1c — Outlier Rejection [7 marks] — ⚠️ Missing Code Documentation

- KITTI: 533 poses (vs 1096 baseline) — tracking lost early, RMSE 5.77m vs 4.39m (+31%).
- TUM: 583 poses, RMSE 0.048m vs 0.010m (+391%).
- The direction is correct, degradation is shown.
- However, there is **zero documentation** of *where* in the ORB-SLAM2 codebase outlier rejection was disabled. The brief says *"Your job is to go into the code and find the place."* Markers need to verify the correct location was found (e.g., `CheckFundamental`/`CheckHomography` in `Initializer.cc`, the `OutliersFlag` logic in `Optimizer.cc`, or the inlier/outlier marking in `Tracking.cc`). Without this, the examiner cannot confirm the right function was disabled.

**Score: 5 / 7**

---

### Q1d — Loop Closure [7 marks] — ⚠️ Missing Code Documentation

- KITTI no-loop: RMSE = 18.02m vs 4.39m — dramatic, correct. KITTI 07 is a loop sequence so this is expected.
- TUM no-loop: RMSE = 0.011m vs 0.010m — barely any effect (12% worse), which makes theoretical sense for a short mostly-linear TUM xyz sequence. However 539 poses vs 753 baseline (fewer poses without loop closure) needs to be explained.
- Again, **no documentation of where loop closure was disabled** in the ORB-SLAM2 source. The `LoopClosing.cc` thread and `DetectLoop()`/`CorrectLoop()` functions should be referenced.

**Score: 5 / 7**

---

### Q1 Subtotal: 21 / 30

---

## Question 2: Visual SLAM with Own Sequences (33 marks)

### What Exists
- 9 sequences collected (Basement_1/2, Floor7_Hallway, Outdoor_1, Washroom, BikeStorage, BikeStorage2, Entrance2, OnePoolStreet1) — far exceeds the minimum of 2.
- COLMAP succeeded on 8/9 sequences; Floor7_Hallway produced only 8 poses — effectively failed.
- ORB-SLAM2 trajectories exist for all 9.
- Intel RealSense D455 calibration properly applied.

---

### Q2a — Data Collection [15 marks]

**Frame count requirement — ⚠️ Two sequences below minimum:**

The brief says *"at least 500 frames after ORB-SLAM manages initialisation."*

| Sequence | ORB-SLAM2 Poses | Status |
|---|---|---|
| BikeStorage | 305 | ❌ Below 500 |
| BikeStorage2 | 283 | ❌ Below 500 |
| Floor7_Hallway | 622 | ✅ |
| Outdoor_1 | 606 | ✅ |
| Washroom | 2208 | ✅ |
| Entrance2 | 1134 | ✅ |
| Basement_1 | 4357 | ✅ |
| Basement_2 | 3327 | ✅ |
| OnePoolStreet1 | 5759 | ✅ |

**COLMAP calibration procedure — ⚠️ Requirement not met:**

The D455 was calibrated using factory SDK parameters (`426.675, 426.104, 425.341, 247.517`), not via a COLMAP calibration procedure (e.g., checkerboard calibration run through COLMAP). The brief says *"use COLMAP to calibrate the camera."* Using SDK factory calibration does not fulfil this requirement.

**Score: 10 / 15**

---

### Q2b — Results [15 marks]

**EVO not used for trajectory comparison — ⚠️ Wrong toolchain:**

The `q2b_evo_comparison.py` file implements a manual Umeyama alignment (`align_umeyama`) and RMSE calculation rather than using the EVO Python API. The brief says *"use EVO to compare the trajectories."* The manual implementation produces similar numbers but is not the required toolchain, while EVO is already used correctly in Q1.

**Floor7_Hallway COLMAP near-failure:**

Only 8 poses recovered from the large indoor area (Marshgate hallway) — the COLMAP pipeline for this sequence effectively failed, which weakens the cross-method comparison for the most important scene.

**Score: 9 / 15**

---

### Q2c — Video [3 marks]

Video file `COMP0222_CW2_GRP_1_Visual_SLAM.mp4` exists.

**Score: 2 / 3** *(assumed from file existence)*

---

### Q2 Subtotal: 21 / 33

---

## Question 3: LiDAR SLAM with Own Sequences (37 marks)

### What Exists
- Solid, well-structured LiDAR SLAM implementation in `q3_lidar_slam_complete.py`.
- Point-to-plane ICP correctly linearised (cross-product moment arm for rotation, `lstsq` solution).
- Log-odds occupancy grid with Bresenham ray-casting — correct implementation.
- All 4 parameter variations in Q3b properly set up and run.
- Loop closure detection via pose-distance pre-filter + ICP score verification.
- Factor graph using `scipy.optimize`.

---

### Q3a — Data Collection [4 marks]

- Sequences described, large indoor area (Floor7_Hallway) included.
- No verification that the two-loop requirement was met (see Q3b notes below).

**Score: 3 / 4**

---

### Q3b — Laser Odometry and Mapping [20 marks]

All 4 parameter types are tested:
- Max range (2000mm vs 8000mm sensor max) ✅
- Angular resolution (full, every 2nd, every 3rd beam) ✅
- Voxel grid downsampling (None, 0.05m, 0.10m, 0.20m) ✅
- Scan rate reduction (all scans, skip 50%, skip 67%) ✅

**Two-loop trajectory requirement — ⚠️ Unverified:**

The brief says *"complete exactly two loops of the area, and return to the exact same starting point."* There is no verification or documentation that the recorded sequences actually contain two complete loops. The closure error metric only measures start-to-end Euclidean distance — it does not verify two loops were performed.

**Normal estimation performance issue:**

`estimate_normals_pca` iterates over every point in Python (`for i in range(len(pts))`). For a scan with 300+ points called across thousands of frames, this scales poorly for large sequences.

**Score: 13 / 20**

---

### Q3c — Loop Closure Detection [5 marks]

- Pose-distance pre-filter + ICP score verification is a valid two-stage approach.
- Score threshold (55% of points within 0.4m) is not calibrated against the environment.
- For Floor7_Hallway (long corridor), similar wall patterns at different locations could easily pass a 55% threshold with a 2m pose-distance window — false positives are a real risk and no discussion is provided.

**Score: 3 / 5**

---

### Q3d — Factor Graph Optimisation [5 marks] — 🔴 Critical Bug

```python
# In optimize_pose_graph():
def anchor(x):
    return x[:3] - x0[:3]

result = minimize(cost, x0, method='L-BFGS-B',
                  options={'maxiter': n_iter, 'ftol': 1e-9, 'gtol': 1e-7})
```

The `anchor` function is defined but **never passed** to `minimize`. The `L-BFGS-B` method accepts `bounds`, not `constraints`. The optimiser has no gauge freedom fix — all poses are free to drift globally. This is a significant bug: without an anchor, the optimiser can translate/rotate the entire map arbitrarily. The result may look plausible but is mathematically underconstrained.

**Additionally:** `factor_graph_optimization.py` is a separate class-based implementation whose `demonstrate_factor_graph_optimization()` function uses **true** relative transforms for odometry factors rather than noisy ones. This defeats the purpose of the demonstration — it trivially shows a perfect graph being "optimised."

**Score: 2 / 5**

---

### Q3e — Video [3 marks]

Video file `COMP0222_CW2_GRP_1_LiDAR_SLAM.mp4` exists.

**Score: 2 / 3** *(assumed from file existence)*

---

### Q3 Subtotal: 23 / 37

---

## Cross-Cutting Issues Affecting All Questions

### 1. Hardcoded Absolute Paths

Every script uses `/home/mmaaz/SLAM/...` paths. The submitted code cannot be run on any machine without manual path editing. Examples:

```python
# q3_lidar_slam_complete.py
REC1     = '/home/mmaaz/SLAM/extracted_data/tmp_recordings/tmp_recordings'
OUT_DIR  = '/home/mmaaz/SLAM/coursework_deliverables/data/q3_results'

# q1_evo_analysis.py
DATA = '/home/mmaaz/SLAM/coursework_deliverables/data/part1_analysis'

# q2_visual_slam_analysis.py
BASE = "/home/mmaaz/SLAM/coursework_deliverables/data"
```

### 2. Empty Files

- `project_summary_report.txt` — 0 lines (empty)
- `slam_exam_prep.txt` — 0 lines (empty)

### 3. No ORB-SLAM2 Code Change Documentation

Q1b, Q1c, and Q1d all require the student to *find and modify* specific locations in the ORB-SLAM2 source code. There is no documentation (in the report, README, or code comments) of which files were changed, at which lines, and what modification was made.

---

## Final Score Breakdown

| Section | Description | Max | Score |
|---|---|---|---|
| Q1a | Baseline evaluation | 8 | 6 |
| Q1b | Feature count variations | 8 | 5 |
| Q1c | Outlier rejection | 7 | 5 |
| Q1d | Loop closure disabled | 7 | 5 |
| Q2a | Data collection + calibration | 15 | 10 |
| Q2b | COLMAP vs ORB-SLAM2 results | 15 | 9 |
| Q2c | Video | 3 | 2 |
| Q3a | LiDAR data collection | 4 | 3 |
| Q3b | Parameter analysis | 20 | 13 |
| Q3c | Loop closure detection | 5 | 3 |
| Q3d | Factor graph optimisation | 5 | 2 |
| Q3e | Video | 3 | 2 |
| **Total** | | **100** | **65** |

---

## Priority Fixes Before Submission

| Priority | Issue | Effort |
|---|---|---|
| 🔴 High | Fix factor graph anchor constraint — pass `constraints` to `minimize` or use SLSQP | Low |
| 🔴 High | Document ORB-SLAM2 code changes for Q1b, Q1c, Q1d (file names, line numbers, what was changed) | Low |
| 🔴 High | Fix Q1b TUM methodology — re-run with features 1000 (baseline) → 500 → 200 | Medium |
| 🟡 Medium | Replace manual Umeyama in `q2b_evo_comparison.py` with EVO API (already used in Q1) | Low |
| 🟡 Medium | Address BikeStorage/BikeStorage2 below 500-frame minimum | Medium |
| 🟡 Medium | Document COLMAP calibration procedure vs factory SDK parameters | Low |
| 🟢 Low | Replace all hardcoded `/home/mmaaz/` paths with relative or configurable paths | Low |
| 🟢 Low | Verify two-loop requirement in LiDAR sequences and document it | Low |
