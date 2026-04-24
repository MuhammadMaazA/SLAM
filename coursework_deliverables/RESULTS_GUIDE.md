# Results Guide — COMP0222 CW2 Group 32

This guide explains what every result means, how to read each figure,
and what to say about it in the oral. Use it as your cheat-sheet.

---

## How to read the videos

### Visual SLAM video (first 30 s of `COMP0222_CW2_GRP_32.mp4`)

| Panel | What it shows |
|---|---|
| Top row | Live camera feed from the RealSense D455 at that moment |
| Bottom row | ORB-SLAM2 trajectory **drawing itself** as the sequence plays |
| Green circle | Where the camera started |
| Red square | Where the camera is right now |
| Axis units | **Arbitrary** — monocular has no metric scale |

**Why the trajectory doesn't look like a perfect rectangle:**
ORB-SLAM2 monocular has no depth reference. It picks its own coordinate
system from the first two frames it can triangulate. The shape is correct
relative to itself, but the scale and orientation are arbitrary. A 40 m
basement walk might appear as 0.6 units on the plot. This is normal and
expected for monocular SLAM — it is not a bug.

**Why Outdoor_1 looks compressed:**
The outdoor sequence has bright sky (no returns), flat ground (ambiguous),
and limited texture. ORB-SLAM2 lost tracking frequently, so only ~584 poses
are tracked from a ~3600-frame recording.

---

### LiDAR SLAM video (last 30 s of `COMP0222_CW2_GRP_32.mp4`)

| Panel | What it shows |
|---|---|
| Colour point cloud | Accumulated LiDAR map building up scan by scan |
| White line | Robot trajectory estimated by ICP odometry |
| Green circle | Start position |
| Red square | Current position |
| Axis units | **Metric metres** — LiDAR has real scale |

**Why Basement_1 looks great:**
Structured indoor room with walls on all sides. ICP locks onto all four
walls simultaneously → heading is well constrained → the two rectangular
loops are clearly visible and well-aligned.

**Why Floor7_Hallway looks elongated:**
This is the Marshgate level-7 corridor. The white rectangle in the middle
is the robot's path around a smaller rectangular circuit inside the
hallway. The long green walls either side are the corridor. The corridor
problem (aperture problem) means ICP can slide along the corridor axis —
that's why the path isn't a perfect rectangle.

**Why Outdoor_1 still drifts:**
Open outdoor space has sparse geometry: sky = no returns, ground = flat
(ambiguous), far features = weak. Even with wider ICP parameters the
trajectory drifts. This is a physical limitation documented in Lab 08.

---

## Q1 — Visual SLAM on Benchmarks

### Q1a Baseline (`q1a_baseline.png`)

| | KITTI 07 | TUM long office |
|---|---|---|
| ATE RMSE | **4.39 m** | **0.032 m** |
| Path length | 695 m | 30 m |
| ATE as % of path | **0.6%** | **0.1%** |

**What to say:** Both are excellent. KITTI 07 at 0.6% of path length is
competitive with published monocular ORB-SLAM2 results. TUM at 0.1% is
near-perfect on a short indoor sequence.

---

### Q1b Feature Count (`q1b_features.png`)

**TUM side — clean degradation story:**

| Features | ATE | Poses | What happened |
|---|---|---|---|
| 1000 (baseline) | 0.032 m | 2557 | Full tracking |
| 800 | 0.109 m | 2556 | 100% tracking but 3× more drift — fewer descriptors → more ambiguous matches → BA errors |
| 500 | 0.015 m | 1152 | Only 45% of frames tracked (the easy early segment) — ATE looks good but it's a tracking failure |
| 250 | FAILED | 0 | Init failed |

**What to say about feat=500 paradox:** "feat=500 ATE is lower than baseline
but this is misleading — it only tracked 45% of frames before losing tracking.
ATE is measured only over tracked frames. It dropped the hard late segment
that causes drift, so the metric is not comparable."

**KITTI side:**
All 8 sub-1000 feature counts fail to initialise. ORB-SLAM2 monocular needs
≥100 RANSAC inliers to triangulate the initial map. At 10 fps vehicle speed,
the inter-frame baseline is so large that features scatter too far to match
reliably — this fails at every count. Verified from run logs.

---

### Q1c Outlier Rejection (`q1c_outlier.png`)

| | KITTI | TUM |
|---|---|---|
| With outlier rejection | 4.39 m | 0.032 m |
| Without | 5.76 m (+31%) | 0.474 m (+1381%) |

**What to say:** TUM is catastrophically affected because indoor scenes have
glass, reflective surfaces, and moving people — all produce wrong observations
that normally get rejected by the chi-squared gate. Without the gate, they
corrupt every keyframe. KITTI is less affected because outdoor scenes are
cleaner.

---

### Q1d Loop Closure (`q1d_loop.png`)

| | KITTI | TUM |
|---|---|---|
| With loop closure | 4.39 m | 0.032 m |
| Without | 5.11 m (+16%) | 0.038 m (+19%) |

**What to say:** KITTI 07 has one prominent loop near the end of the sequence
— disabling closure correction causes visible trajectory drift there. TUM's
office is a partial loop: closure helps but the sequence is short enough that
drift is modest even without it.

---

## Q2 — Visual SLAM on Custom Sequences

### Q2a Data Collection (`q2a_all_trajectories.png`)

9 sequences collected at UCL Marshgate. Two meet the ≥500 pose threshold
on the calibrated path:
- **Basement_1: 868 poses** ✅ indoor
- **Outdoor_1: 584 poses** ✅ outdoor

BikeStorage2 (night, 114 poses) is an explicit failure case — include it
as evidence you tested challenging conditions.

### Q2a Calibration (`q2a_calibration_report.png`)

COLMAP run with OPENCV model on each sequence. Results:
- Basement_1 scratch calibration: f = 383.7 px vs factory 426.7 px (Δ≈10%)
  — this is expected because SIMPLE_PINHOLE assumes equal fx=fy
- All other sequences: within 0.3–1.6% of factory values
- **This confirms the camera is stable across environments**

### Q2b COLMAP vs ORB-SLAM2 (`q2b_colmap_vs_orbslam.png`)

**CRITICAL: the "ATE" here is NOT accuracy against ground truth.
It is inter-method agreement — how much do COLMAP and ORB-SLAM2 disagree.**

| Sequence | Trans disagreement | Rotation | Verdict |
|---|---|---|---|
| Basement_1 | **132 mm** | 40.6° | Best indoor agreement case |
| Outdoor_1 | 1.84 m | 168.6° | Severe disagreement — hard outdoor monocular case |
| Basement_2 | 3.00 m | 45.9° | High disagreement |
| Floor7 | — | — | COLMAP failed (featureless walls) |
| OnePoolStreet1 | 3.00 m | 2.5° | High translational disagreement |

**How to say this safely in the oral:**
These are inter-method disagreement numbers, not ground-truth error.
`Basement_1` is the strongest agreement case. `Outdoor_1` is still valid for
the brief, but it is a difficult outdoor monocular sequence where the two
methods disagree strongly after alignment. Do not oversell it as "good"
agreement.

---

## Q3 — LiDAR SLAM on Custom Sequences

### Q3a Two-Loop Verification

**Brief requirement: exactly 2 loops, return to start.**

We use proximity-based lap detection: a new lap is counted when the robot
returns within 1.5 m of origin after travelling ≥5 m from the last boundary.

| Sequence | Laps detected | Closure error |
|---|---|---|
| Basement_1 | 2 | 0.19 m |
| Floor7_Hallway | 2 | 0.16 m |
| Outdoor_1 | 2 | 0.94 m |

### Q3b Parameter Analysis — Key Numbers

**Best settings across all sequences:**
- Max range: **12,000 mm** (sensor max) — always better than 2,000 mm
- Angular: **full scan** — every 2nd/3rd degrades ICP normals
- Voxel: **0.05 m** — filters noise, preserves geometry
- Scan rate: **all scans** — more updates = less per-step drift

**Worst case example (Basement_1, max range):**
- 2,000 mm: closure error 1.87 m
- 12,000 mm: closure error 0.19 m
- That's **10× worse** with short range

### Q3c Loop Closure (`q3c_basement_1.png`)

4-gate filter:
1. Temporal: j − i ≥ 10 keyframes
2. Arc: robot must have travelled ≥8 m since the anchor
3. Pose distance: ≤2.0 m from anchor
4. ICP overlap score: ≥0.70

**The histogram is the key evidence.** It shows a bimodal distribution:
- Sliding-window false matches: score 0.45–0.60
- Confirmed revisits: score 0.75–0.95
- Clear gap at 0.70 → threshold is well-placed

### Q3d Factor Graph (`q3d_basement_1.png`, `q3d_floor7_hallway.png`)

GTSAM Levenberg-Marquardt on Pose2 graph.

| Sequence | Before | After | Reduction |
|---|---|---|---|
| Floor7_Hallway | 0.163 m | 0.061 m | **−62%** |
| Basement_1 | 0.194 m | 0.165 m | −15% |
| Outdoor_1 | 0.941 m | 0.886 m | −6% |

**Why Floor7 benefits most:** The corridor layout creates well-separated
loop revisit points that strongly constrain the drift direction. The factor
graph can pull the start and end together cleanly.

**Why Outdoor benefits least:** Noisy ICP in sparse outdoor geometry produces
loop constraints that don't precisely match the odometry. The solver can only
reduce drift by 6%.

---

## Oral presentation cheat-sheet (10 min)

### Q2 — 5 minutes

1. **(30s)** We collected 9 sequences with RealSense D455. Two meet the ≥500 pose brief requirement using calibrated intrinsics.
2. **(60s)** COLMAP calibration pipeline: ran SfM → extracted per-sequence YAMLs → re-ran ORB-SLAM2. Show `q2a_calibration_report.png`.
3. **(60s)** Comparison with EVO: Basement_1 is the strongest indoor agreement case at 132mm translation RMSE, while Outdoor_1 shows severe disagreement at 1.84m. Explain: this is *inter-method agreement*, not accuracy vs ground truth.
4. **(60s)** Floor7 COLMAP failure: featureless white corridors give insufficient texture for SfM matching. Show `q2b_colmap_vs_orbslam.png`.
5. **(30s)** Show 3D point cloud (`q2b_pointcloud_basement_1.png`) — COLMAP reconstructed the basement room structure.

### Q3 — 5 minutes

1. **(30s)** Three sequences: Basement_1 (indoor), Floor7_Hallway (indoor large area, Marshgate), Outdoor_1 (outdoor). Two loops each.
2. **(60s)** Parameter ablation key finding: sensor-max range (12,000 mm) is critical. Short range (2,000 mm) gives 10× worse closure error. Show `q3b_basement_1.png`.
3. **(60s)** Loop closure: 4-gate pipeline. Show score histogram — bimodal gap proves threshold is correct. False matches at 0.45–0.60, true revisits at 0.75–0.95.
4. **(60s)** Factor graph: Floor7 −62%, Basement −15%. Show before/after maps. Explain why Floor7 benefits most (corridor geometry constrains drift direction strongly).
5. **(30s)** Outdoor limitation: aperture problem. LiDAR sees two parallel walls — can't estimate forward motion. This is Lab 08 Activity 3B. That's why outdoor PGO only reduces error by 6%.

---

## If the examiner asks...

**"Why is your visual SLAM trajectory not a rectangle?"**
Monocular SLAM has no metric scale or absolute orientation. ORB-SLAM2 picks
its coordinate frame from the first triangulated frame pair — which is
essentially arbitrary. The trajectory is self-consistent but rotated and
scaled arbitrarily. This is fundamental, not a bug.

**"Why does Floor7_Hallway drift so badly in LiDAR?"**
The corridor problem / aperture problem (Lab 08 Activity 3B). In a featureless
tunnel, sliding the scan forward along the corridor axis produces zero ICP
error change. The algorithm cannot find a unique minimum for forward motion.
This is why we chose it as the large-area sequence — it demonstrates this
limitation clearly, and factor-graph optimisation still reduces closure error
by 62%.

**"What is COLMAP calibration vs factory calibration?"**
Factory: manufacturer's measured intrinsics (fx=426.7 px). COLMAP: we ran
structure-from-motion on our collected data with no prior, and COLMAP
estimated its own intrinsics. The Basement_1 scratch result (f=383.7 px,
Δ≈10%) comes from SIMPLE_PINHOLE assuming equal focal length which is
slightly wrong. Full OPENCV on other sequences agrees within 1.6%.

**"Why did you choose 0.70 as the loop closure threshold?"**
We examined the ICP score histogram empirically across all 3 sequences.
Sliding-window false matches consistently score 0.45–0.60.
Confirmed revisits consistently score 0.75–0.95.
There is a clear bimodal gap around 0.70 in every sequence — this is the
evidence shown in the Q3c figures.

**"Your Basement_1 PGO got worse in a previous run — why?"**
The original code used a fixed sigma for all loop closure edges regardless
of ICP match quality. A borderline loop at score=0.71 was trusted identically
to a perfect one at 0.97 — causing the solver to over-constrain in the wrong
direction. We fixed this by propagating the per-loop ICP Hessian as the
information matrix, so weak loops are down-weighted automatically.
