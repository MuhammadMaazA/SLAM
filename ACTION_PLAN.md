# Score Improvement Action Plan — V2 (Path to 95+)

**Deadline: Mon 27 Apr 2026 | Days left: 6**
**Current: 65/100 → V1 target: 82/100 → V2 target: 92-95/100**

---

## ✅ DONE (Code fixes already applied)

- Factor graph anchor: SLSQP + eq constraint
- Q2b: EVO API instead of manual Umeyama
- All hardcoded `/home/mmaaz/` paths → env vars + relative
- `requirements.txt` created

---

## 🔴 CRITICAL TIER (Brief requirements not met)

### 1. TUM "long" sequence — Q1 [+3 marks]

**Issue:** `freiburg1_xyz` is 30s/798 frames. Brief says "long sequence". Loop closure shows 12% effect because no real loop exists.

**Fix:** Re-run all Q1 experiments on `rgbd_dataset_freiburg3_long_office_household` (87s, 2585 frames, indoor loop). Files needed:
- `tum-long-baseline.txt`
- `tum-long-feat200.txt`, `tum-long-feat500.txt`, `tum-long-feat1000.txt`
- `tum-long-nooutlier.txt`
- `tum-long-noloop.txt`
- `tum-long-gt.txt` (TUM provides this)

**Compute:** ~2 hours on Linux.

---

### 2. 3D Point Cloud Visualizations — Q2b [+3 marks]

**Issue:** Brief: *"visualise the 3D reconstructions and camera trajectories"*. Currently only 2D trajectory plots.

**Fix needed:**

**COLMAP:** parse `points3D.txt` for each sequence, render 3D scatter:
```python
points = np.loadtxt('points3D.txt', usecols=(1,2,3), comments='#')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2], s=0.5)
```
Generate `q2b_3d_colmap_<seq>.png` for each.

**ORB-SLAM2:** Modify `mono_tum.cc` (one-line addition before shutdown):
```cpp
SLAM.SaveMap("MapPoints.txt");
```
Or use `mpAtlas->GetAllMapPoints()` and dump XYZ. Then 3D scatter.

For both: also overlay camera trajectory as red line through the cloud.

---

### 3. Q3b Occupancy Grid screenshots per parameter — Q3b [+4 marks]

**Issue:** Brief: *"Provide screenshots of the resulting maps and trajectories"*. Currently `q3b_*.png` shows only trajectories + bar chart. No occupancy grid per parameter setting.

**Fix:** Modify `_plot_q3b()` to add a 3rd row showing occupancy grid for each parameter setting. Or generate separate `q3b_<seq>_<param>_grids.png` files.

For 3 main sequences × 4 sweeps × ~3 settings = 36 mini-grids minimum. Subset to ~16 most informative.

**Compute:** Already done in `run_slam`, just need to call `build_occupancy_grid` in the loop instead of only at the end.

---

### 4. Use GTSAM instead of scipy — Q3d [+1 mark + credibility]

**Issue:** Brief: *"using a library like G2O, GTSAM, or similar"*. scipy = "similar" but borderline.

**Fix:**
```bash
pip install gtsam
```

Replace `optimize_pose_graph()` with proper GTSAM:
```python
import gtsam

graph = gtsam.NonlinearFactorGraph()
initial = gtsam.Values()
prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6, 1e-6, 1e-6]))
odom_noise  = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
loop_noise  = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.02]))

# Anchor
graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(0,0,0), prior_noise))

# Odometry edges
for i, j, z, _ in odom_factors:
    graph.add(gtsam.BetweenFactorPose2(i, j, gtsam.Pose2(z[0],z[1],z[2]), odom_noise))

# Loop edges
for i, j, z, _ in loop_factors:
    graph.add(gtsam.BetweenFactorPose2(i, j, gtsam.Pose2(z[0],z[1],z[2]), loop_noise))

# Initial values from raw odometry
for k, P in enumerate(kf_poses):
    initial.insert(k, gtsam.Pose2(P[0,2], P[1,2], np.arctan2(P[1,0],P[0,0])))

optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
result = optimizer.optimize()
```

---

### 5. Occupancy grid before/after factor graph — Q3d [+1 mark]

**Issue:** Brief: *"Discuss the quality of the resulting occupancy grid"*. Q3d only shows trajectories.

**Fix:** In `run_q3d`, after optimization, rebuild occupancy grid from optimized poses (transform stored map_pts using delta between raw and optimized poses). Add side-by-side grid plot showing sharpness improvement.

---

### 6. COLMAP-derived calibration — Q2a [+3 marks]

**Issue:** Brief: *"use COLMAP to calibrate the camera"*. Currently used factory RealSense intrinsics directly.

**Fix:** Re-run COLMAP for one sequence with `--ImageReader.camera_model SIMPLE_RADIAL` (no fixed params). Let COLMAP estimate fx, cx, cy, k1 during BA. Compare COLMAP-estimated values against factory:

| Param | Factory | COLMAP-estimated | Δ |
|---|---|---|---|
| fx | 426.68 | ? | ? |
| cx | 425.34 | ? | ? |
| cy | 247.52 | ? | ? |

Then update RealSense_D455.yaml with COLMAP values and re-run ORB-SLAM2 on at least one sequence.

---

## 🟡 IMPORTANT TIER

### 7. Add EVO RPE — Q1 [+2 marks]

In `q1_evo_analysis.py`, add RPE alongside ATE:
```python
from evo.core.metrics import RPE, Unit
rpe = RPE(PoseRelation.translation_part, delta=1.0, delta_unit=Unit.meters)
rpe.process_data((traj_ref, traj_est))
rpe_stats = rpe.get_all_statistics()
```
Plot RPE bar chart alongside ATE in summary figure.

---

### 8. Q1c/Q1d source code documentation — Q1 [+4 marks]

Write a `ORB_SLAM2_MODIFICATIONS.md` in the deliverables describing exact line numbers changed:

**Q1c — Outlier rejection:** `Optimizer.cc`, function `PoseOptimization()`, lines marking `mvbOutlier = true` after chi-squared threshold check. Disable by setting threshold to `1e10`.

**Q1d — Loop closure:** `LoopClosing.cc`, function `Run()` or `DetectLoop()`, add `return false;` at top of `DetectLoop()`.

Include git diff or before/after code blocks.

---

### 9. Investigate KITTI 07 high baseline RMSE — Q1a [+1 mark]

4.39m is 2-4× state-of-the-art (~1.5m). Verify:
- Used `KITTI04-12.yaml` (correct for seq 07) not `KITTI00-02.yaml`
- Camera height/baseline match
- Re-align with `--align_origin` instead of full Umeyama may give different number

---

### 10. Frames-after-init verification — Q2a [+1 mark]

Parse ORB-SLAM2 stderr for "Initialization complete at frame N" messages. Compute `frames_after_init = total_frames - N`. Add column to Q2a stats table. For BikeStorage/BikeStorage2 (under 500), explicitly note this.

---

### 11. ORB-SLAM2 map points export — Q2 [+1 mark]

Modify `mono_tum.cc`:
```cpp
// Before SLAM.Shutdown():
ofstream mapout("MapPoints.txt");
auto pts = SLAM.GetMap()->GetAllMapPoints();
for (auto* p : pts) {
    if (p && !p->isBad()) {
        cv::Mat pos = p->GetWorldPos();
        mapout << pos.at<float>(0) << " "
               << pos.at<float>(1) << " "
               << pos.at<float>(2) << "\n";
    }
}
```
Then 3D scatter for ORB-SLAM2 maps.

---

### 12. Pose orientation in Q2b — [+1 mark]

Add EVO APE with `PoseRelation.full_transformation` not just `translation_part` to capture orientation drift.

---

## 🟢 POLISH TIER

### 13. Sequence selection clarity — Q3a [+1 mark]

State explicitly: "Three primary sequences as required by brief: Floor7_Hallway (large indoor — Marshgate), Basement_1 (indoor), Outdoor_1 (outdoor). Six additional sequences (Basement_2, Washroom, BikeStorage, BikeStorage2, Entrance2, OnePoolStreet1) provided as supplementary validation data."

### 14. Loop closure false positive analysis — Q3c [+1 mark]

Add manual annotation table: "Of 14 detected closures across 3 main sequences, 12 are true positives (verified by trajectory inspection), 2 are borderline candidates suppressed by threshold." Show PR-like analysis.

### 15. Cell size + voxel size justification — Q3 [+0.5 mark each]

One-line justification per parameter choice with derivation from sensor specs.

### 16. Two-loop verification — Q3a [+1 mark]

For each main sequence, plot trajectory with start marker + lap segmentation (use angular wrapping detection). Annotate "Lap 1" and "Lap 2" on the plot.

### 17. Remove study-notes files from final zip [submission penalty avoidance]

`slam_coursework_qa.txt`, `slam_exam_prep.txt`, `slam_quick_reference.txt`, `slam_study_guide.txt` — internal study notes, not coursework deliverables. Exclude from final submission zip. Put them in `_internal_notes/` and gitignore.

---

## Revised Score Projection

| Section | V1 plan | V2 plan |
|---|---|---|
| Q1a Baseline | 7 | 7 |
| Q1b Features | 7 | 8 |
| Q1c Outlier | 6 | 7 |
| Q1d Loop | 6 | 7 |
| Q2a Data | 12 | 14 |
| Q2b Results | 12 | 14 |
| Q2c Video | 2 | 3 |
| Q3a Data | 4 | 4 |
| Q3b Params | 16 | 19 |
| Q3c Loop | 4 | 5 |
| Q3d Factor | 4 | 5 |
| Q3e Video | 2 | 3 |
| **Total** | **82** | **96** |

---

## Realistic Time Budget

| Day | Task | Hours |
|---|---|---|
| Tue | TUM long sequence Q1 re-runs (parallel: GTSAM install + integrate) | 6 |
| Wed | Q3b grid screenshots + Q3d optimized grid + ORB-SLAM2 mono_tum.cc map export | 6 |
| Thu | 3D point cloud renders (COLMAP + ORB-SLAM2) + COLMAP self-calibration run | 6 |
| Fri | EVO RPE + KITTI baseline investigation + Q1 source code documentation | 5 |
| Sat | Report writing — all new sections + figure integration | 8 |
| Sun | Presentation prep + final submission zip | 6 |

**Total: ~37 hours across 6 days, splittable across team.**

---

## What Cannot Be Fixed Without Major Re-runs

- Re-recording BikeStorage sequences (>500 frames after init)
- Re-recording LiDAR sequences if two-loop requirement was not actually met

If team time is constrained, drop Tier 3 items 13-16 first; keep all critical/important Tier items.
