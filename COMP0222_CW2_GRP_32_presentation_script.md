# COMP0222 CW2 Group 32 — Presentation Script
**Total: ~10 minutes | Q2: slides 1–6 (~5 min) | Q3: slides 7–11 (~5 min)**
*Pace: read naturally, don't rush. One breath between paragraphs.*

---

## Slide 1 — Title (0:00–0:30)

Hi, we're Group 32. This is Coursework 2, Visual and LiDAR SLAM.

We split the talk into two halves. First, monocular visual SLAM — where we collected our own sequences and compared COLMAP with ORB-SLAM2. Second, 2D LiDAR SLAM — where we built a full pipeline from scan matching through to pose-graph optimisation on our own recorded sequences.

Throughout both parts, we focus on failure cases just as much as successes, because understanding why something breaks tells you more about the algorithm than just showing the best result.

---
> **Annotation**
> **Requirement:** present both Q2 and Q3 in 10 minutes total, 5 min each.
> **What we did:** recorded our own data with a RealSense D455 (visual) and RPLIDAR A2M12 (LiDAR), ran the full pipelines, and documented failures with root-cause analysis.
> **Why:** the brief explicitly asks for analysis of settings and failure modes, not just results tables.
---

---

## Slide 2 — Agenda (0:30–0:55)

Part 1: we captured indoor and outdoor sequences, calibrated the camera using COLMAP, ran COLMAP for 3D reconstruction, then ran ORB-SLAM2 with those calibrated parameters and used EVO to compare the two trajectories.

Part 2: we collected three LiDAR sequences, swept four parameters one at a time, implemented four-gate loop closure detection, then fed everything into a GTSAM factor graph.

---
> **Annotation**
> **Requirement:** Q2 — data collection, calibration, COLMAP reconstruction, ORB-SLAM2, EVO comparison. Q3 — parameter analysis, loop closure, pose-graph.
> **How:** each sub-question is its own script block; all numbers cited come directly from output files we can verify.
---

---

## Slide 3 — Part 1: Visual SLAM intro (0:55–1:10)

So Part 1. We recorded two sequences ourselves using an Intel RealSense D455. The key thing we were testing is: how does camera calibration quality affect monocular SLAM behaviour, and does it affect indoor and outdoor scenes differently?

---

---

## Slide 4 — Q2(a): Data Collection and Calibration (1:10–2:30)

We recorded colour video at 848×480 and 30 frames per second. We locked auto-exposure after a two-second warm-up so the camera's apparent intrinsics stayed constant during recording. We walked slowly — about 0.3 to 0.5 metres per second — because at 30 fps that gives only 1 to 1.5 centimetres of motion per frame, which keeps feature matching reliable. We also paused and panned slowly at every corner, which gives COLMAP the multi-view geometry it needs for bundle adjustment.

**Basement_1** is the lower-ground-floor lift and fire-exit core at Marshgate. Concrete walls on all four sides, lift-door frames, emergency signs, high-contrast signage. Dense repeatable texture.

**Outdoor_1** is the cycle-parking courtyard at Marshgate's back exit. Building facade on the right, trees and open sky on the left. Low-texture ground, sunlight glare, moving shadows.

For calibration we ran COLMAP's OPENCV 8-parameter model on each sequence — that gives fx, fy, cx, cy, and four distortion coefficients. We then converted the cameras.txt output directly into an ORB-SLAM2 YAML file and re-ran ORB-SLAM2 with those calibrated parameters, exactly as the brief requires.

The headline result from calibration: fx stayed within 1% of the factory value on both sequences, which confirms the D455 factory calibration is accurate in x. But fy was inflated — Basement_1 got +4.6%, Outdoor_1 got +2.6%. This fy inflation turns out to be the root cause of the outdoor failure we show next.

---
> **Annotation**
> **Requirement (Q2a, 15 marks):** describe data acquisition strategy, settings, what worked, and the calibration process.
> **What we did:** recorded continuous video stream (not still shots) at controlled speed; used COLMAP OPENCV self-calibration on the full sequence; converted cameras.txt → ORB-SLAM2 YAML with q2_calibration_report.py; re-ran ORB-SLAM2 with calibrated params (trajectories saved as *_trajectory_colmap_intrinsics.txt).
> **Why slow walking speed:** inter-frame displacement ≈ 1–1.5 cm at 30 fps keeps ORB feature matches reliable and keeps both methods inside their convergence basins.
> **Why continuous video not still shots:** still shots force the initialiser to pick two frames with a wide baseline which breaks ORB matching.
> **The fy inflation:** COLMAP trades fy against distortion coefficients when depth variation is limited — a known bundle adjustment artefact when calibrating from a walking video rather than a dedicated wall-facing calibration set.
> **Figures to point at:** calibration table (Table 1) — show Basement_1 and Outdoor_1 OPENCV values vs factory.
---

---

## Slide 5 — Q2(b): COLMAP 3D Reconstruction (2:30–3:15)

Now the COLMAP reconstruction results. For Basement_1, COLMAP selected 283 keyframes and produced 7,330 3D points. The top-down view shows the room outline clearly — walls, corners, and doorways all visible.

For Outdoor_1, 360 keyframes and 17,806 points — over twice as many. Most of those points come from the long vertical building facade, which gives COLMAP dense, stable feature matches all the way along. The trees contribute sparser, less stable structure.

One important thing to understand: in this workflow, COLMAP is sequential — it doesn't do global loop closure. So we treat the COLMAP trajectory as a strong reference, not absolute ground truth.

---
> **Annotation**
> **Requirement:** show 3D reconstructions and camera trajectories from COLMAP.
> **What we did:** ran COLMAP automatic reconstruction on each full sequence; extracted camera poses from images.txt; converted to TUM format for EVO.
> **Why Outdoor_1 has more points than Basement_1:** the long facade provides more overlapping multi-view matches over a larger spatial extent. Outdoor ≠ harder for reconstruction; it depends on scene geometry.
> **Figures:** q2b_colmap_trajectories.png — COLMAP SfM trajectories coloured by elapsed time. q2b_pointcloud_outdoor_1.png — 3D sparse map with colour from images.
---

---

## Slide 6 — Q2(b): ORB-SLAM2 Comparison and EVO (3:15–4:50)

Now the inter-method comparison using EVO. We align both trajectories with a Sim(3) Umeyama transform — that handles scale, rotation, and translation together, which is necessary because monocular SLAM has no metric scale.

**Indoors (Basement_1):** 44 timestamp-matched pose pairs. Translation RMSE 0.046 arbitrary units. Rotation disagreement 3.8 degrees. Both pipelines converge to nearly the same loop shape. The fy inflation of +4.6% didn't cause problems because the enclosed room with walls on all four sides gives ORB-SLAM2 enough close-range constraint to absorb the miscalibration.

**Outdoors (Outdoor_1):** 54 matched pairs. Translation RMSE 1.488 arbitrary units. Rotation disagreement 169 degrees. That's the key result — the trajectories are pointing in essentially opposite directions.

Here's the mechanism. ORB-SLAM2 initialises by triangulating a first map from two keyframes. It back-projects each pixel into a 3D ray using the focal length. With fy inflated by 2.6%, vertical parallax appears shallower than it really is — the perceived depth is wrong. In an open outdoor scene with limited close-range geometry, there's nothing to correct that bias, so the initialiser resolves the depth ambiguity into the mirror-image solution, rotating the coordinate frame by about 180 degrees. Once that first map is laid down in the wrong frame, all 584 subsequent poses are tracked consistently within it — ORB-SLAM2 never fails. The flip only shows up when you compare against COLMAP.

The diagnostic: same sequence, same pipeline, factory fy. Rotation disagreement drops to 0.9 degrees. Same scene — the scene is fine. The failure is entirely the calibration artefact.

---
> **Annotation**
> **Requirement (Q2b, 15 marks):** show EVO trajectory comparison plots and analyse results.
> **What we did:** loaded COLMAP keyframe poses (images.txt), re-attached real RGB timestamps from rgb.txt, used EVO sync.associate_trajectories (30ms tolerance), Sim(3) Umeyama alignment with scale, MAD outlier rejection, reported APE trans and rotation RMSE.
> **Why units are arbitrary:** Sim(3) absorbs unknown monocular scale — both methods produce trajectories up to an unknown scale factor, so after alignment the residual is dimensionless.
> **Why the 169° flip happens:** COLMAP OPENCV inflates fy when calibrating from a walking video (limited depth variation → bundle adjuster trades fy against distortion to minimise reprojection error). Inflated fy → wrong vertical parallax → biased depth estimate → monocular initialisation resolves depth ambiguity into mirror-image solution → 180° coordinate frame flip.
> **The diagnostic (factory intrinsics):** proved the scene is well-conditioned by running same sequence with factory fy=426.1 → 0.9° rotation, 89 matched pairs. This rules out scene geometry as the cause.
> **Figures:** q2b_colmap_vs_orbslam.png — side-by-side COLMAP (white/grey ghost) vs ORB-SLAM2 (plasma gradient). Table 2 — EVO numbers. Table 3 — diagnostic comparison table.
---

---

## Slide 7 — Part 2: LiDAR SLAM intro (4:50–5:05)

Part 2. LiDAR SLAM removes monocular scale ambiguity entirely — failures here are geometric and optimisation-related rather than calibration-related. We built our own full pipeline from raw scans through to pose-graph optimisation.

---

---

## Slide 8 — Q3(a)/(b): Data Collection and Parameter Sweep (5:05–6:15)

We used three sequences all collected in the same physical environments as Q2. Basement_1 is the clean indoor baseline. Floor7_Hallway is the large indoor area — a wide corridor over 15 metres long with side passages. Outdoor_1 is the difficult case — open courtyard with limited stable surfaces for the LiDAR.

For each sequence we did exactly two clockwise loops starting and ending on a chalk floor marker, walking at 0.4 to 0.8 m/s.

For the parameter sweep we varied four things one at a time:

**Maximum range** had the biggest effect. Cutting range to 2 metres on Basement_1 sent closure error from 19 cm up to 2.38 metres — a 12× increase. Geometric reason: point-to-plane ICP needs normals pointing in different directions to constrain x and y independently. At 2 metres only the near wall is visible, everything lies on one line, and the solver slides. At full range, all four walls are in view and the constraint is full-rank. Outdoors the relationship inverts — more range means more noisy distant returns that pull the registration in the wrong direction.

**Angular subsampling** showed an interesting result on Floor7. You'd expect full resolution to always be best, but every-2nd-beam actually improved the corridor. Reason: at full resolution you're fitting to micro-surface imperfections on flat walls, which adds noise to the normal estimates. Dropping to half resolution smooths that out without losing the geometric structure.

---
> **Annotation**
> **Requirement (Q3a):** collect sequences, describe environments and data collection strategy. **(Q3b):** parameter analysis with explanation.
> **What we did:** recorded three sequences with RPLIDAR A2M12 (sensor from Lab 09); implemented point-to-plane ICP instead of Lab 09's hill-climbing scan-to-map because ICP gives us a per-step covariance matrix (the ICP Hessian) that we need for the factor graph edge weights in Q3d. Used log-odds occupancy grid instead of Lab 09's simplified linear model for better numerical stability (Bayesian update = addition in log space, can't overflow).
> **Why two loops:** gives two sets of revisit events for loop closure detection; also lets us verify trajectory coherence via closure error.
> **Figures:** Table (Q3a closure errors), q3a_two_loop plots, occupancy grid maps, q3b parameter sweep tables and grid images.
---

---

## Slide 9 — Q3(b): Parameter Findings Summary (6:15–7:05)

The practical conclusion is: no single configuration is universally best. Range is the most scene-sensitive parameter — the optimal setting is roughly the size of the stable surfaces in view. Angular subsampling is mostly harmful in open rooms but can help in narrow corridors by smoothing normal estimates. Voxel filtering is helpful in structured environments where it acts as a denoiser, but harmful where scan density is already limited. Scan skip is mostly irrelevant in well-constrained spaces but can help in corridors by increasing inter-scan displacement enough to make forward translation observable — that's the aperture problem from Lab 09 playing out in real data.

---
> **Annotation**
> **Requirement:** explain why each setting affects performance.
> **The aperture problem:** Lab 09 Activity 3 explicitly covers this. In a straight featureless corridor, consecutive scans look identical in the forward direction — the ICP error function is flat along that axis. Skipping scans increases the baseline and makes forward translation observable.
> **Point-to-plane ICP vs Lab 09 hill-climbing:** we chose ICP because it also outputs the Hessian J^T W J which gives a per-step inverse covariance matrix. This feeds directly into Q3d as the information matrix for each odometry edge. Hill-climbing doesn't produce this.
---

---

## Slide 10 — Q3(c): Loop Closure Detection (7:05–7:55)

Our loop closure pipeline has four gates. First a temporal gate — keyframes must be at least 10 apart to avoid matching with immediate neighbours. Second an adaptive arc gate — the robot must have physically travelled at least 8 metres between the two keyframes, scaled to 40% of total path for shorter sequences. This is the most important filter; without it you get hundreds of false positives from frames that see similar local geometry. Third a pose distance gate — estimated positions within 2 metres. Fourth an ICP overlap score — we align the two scans and measure what fraction of points land within 40 cm of a match. Real revisits score between 0.75 and 0.95. False matches cluster below 0.6. There's a clear gap, and we set threshold at 0.70.

We got 10 closures on Basement_1, 8 on Floor7, and zero on Outdoor_1 — which is correct because the outdoor trajectory froze early and there were no valid revisits to detect.

---
> **Annotation**
> **Requirement (Q3c):** implement and evaluate loop closure detection.
> **What we did:** built a four-gate detection pipeline entirely from scratch — Lab 09 explicitly says "Real SLAM systems have Loop Closure to fix this, but this is a pure odometry script." So loop closure is our addition beyond the lab.
> **Why four gates:** each gate removes a different failure mode. Temporal removes adjacent-frame false positives. Arc distance removes revisits that are physically impossible (haven't walked far enough). Pose distance removes spatially separated candidates. ICP overlap provides the actual geometric verification.
> **Threshold of 0.70:** chosen from observed score distribution — true positives cluster 0.75–0.95, false positives cluster 0.45–0.60. Clean separability on our sequences.
> **Figures:** loop closure score distribution plots, detected loop visualisations.
---

---

## Slide 11 — Q3(d): Factor Graph Optimisation (7:55–9:05)

We fed all the odometry edges and loop closure edges into GTSAM — the same library from Lab 04 — and ran Levenberg-Marquardt. Each keyframe is a 2D pose: x, y, and heading. Odometry edges use the ICP Hessian as the information matrix, so tighter ICP matches carry more weight. Loop closure edges use the per-loop ICP Hessian similarly.

The graph cost decreased by a factor of 8 to 25 on both indoor sequences, which tells you the optimiser is definitely working. But the closure error — the Euclidean distance from start to end — actually got slightly worse. Floor7 went from 16.6 cm to 22.1 cm. Basement_1 from 17 to 20 cm.

This is not a contradiction. GTSAM is minimising the total weighted inconsistency across the whole graph. The closure error is just the residual on one implied edge that the optimiser never explicitly sees. When loop and odometry constraints disagree, the solver finds the globally cheapest compromise — and that compromise can worsen any single metric while improving the sum. The occupancy map also degrades visually after optimisation because every scan gets reprojected from slightly shifted poses, and those misaligned ray-casts cancel each other rather than reinforcing the walls.

The honest conclusion: our loop edge information matrices are weighted too strongly relative to the odometry chain. The fix would be to scale down loop-edge confidence, or add a robust kernel. We documented the failure rather than re-tune, because it demonstrates the underlying mechanics.

---
> **Annotation**
> **Requirement (Q3d):** implement and evaluate pose-graph optimisation.
> **What we did:** used GTSAM (Lab 04 library) with Pose2 variables; added PriorFactor on first pose to fix gauge; BetweenFactors for odometry using ICP Hessian as information matrix (permuted from θ,x,y to x,y,θ for GTSAM convention); BetweenFactors for loop closures using per-loop ICP Hessian.
> **Why the Hessian as information matrix:** the ICP Hessian H = J^T W J is the per-step inverse covariance. Passing it as the edge information matrix tells GTSAM to trust tighter ICP matches more and weaker matches less. This is geometrically principled — a match against a flat featureless wall has near-zero Hessian eigenvalue in the along-wall direction, correctly reflecting that we learned nothing about translation along that wall.
> **Why closure error worsened:** GTSAM minimises sum of weighted residuals across all edges. Closure error is only an implicit constraint — there's no explicit start-to-end edge in the graph. When loop and odometry factor confidences are mismatched, the optimiser improves global cost at the expense of any single metric not directly constrained.
> **Figures:** Q3d closure error table (before/after), occupancy grid comparison (before/after PGO), cost convergence plot.
---

---

## Slide 12 — Closing (9:05–9:55)

To summarise.

In Q2, the dominant factor was calibration sensitivity. COLMAP OPENCV self-calibration inflated fy when run on a walking video — the data simply doesn't provide enough depth variation to pin down fy independently of distortion. Indoors that didn't matter because close-range geometry absorbed the error. Outdoors, where geometry is weaker, it caused a 169-degree initialisation flip. The factory diagnostic confirmed the scene was fine.

In Q3, the dominant factor was geometric constraint quality. Well-enclosed scenes with walls in all directions gave ICP strong, full-rank constraints. Open scenes and corridors broke one or more constraint directions, and no parameter choice fully compensated. And in the factor graph, cost reduction doesn't automatically mean the metric you care about improves — it depends on how consistently the edge information matrices are weighted.

The shared lesson across both parts: algorithm performance is governed by data geometry and calibration quality, not just algorithm choice.

Thank you.

---
> **Annotation**
> **Closing framing:** connects Q2 calibration sensitivity to Q3 geometric constraint sensitivity. The common thread is "observability" — in both cases, the system fails when the data doesn't provide enough independent geometric constraint for the algorithm to recover what it needs.
---

---

## During Video Playback — What to Say

### Q2 video — Basement_1 (raw feed + pointcloud)
"Here's the raw footage on the left and the 3D sparse map building up on the right. The main visual anchors that keep ORB-SLAM2 stable here are the lift-door frames, the high-contrast emergency signs, and the wall bolt-hole patterns. That specific combination — short range, rigid edges, consistent lighting — is why this sequence tracks well. It's not just 'indoor is easier'; it's this exact environment."

### Q2 video — Outdoor_1 (raw feed + pointcloud)
"On the right you can see the reconstruction is denser along the facade side and sparser on the tree side — that geometric asymmetry is part of why the calibration artefact is harder to absorb outdoors. The raw footage also shows sunlight glare and moving shadows which perturb short-term correspondences frame-to-frame."

### Q3 video (if shown)
"The LiDAR map builds up scan by scan. Basement_1 stays coherent throughout. Floor7 is moderate — the corridor geometry limits ICP constraint in the forward direction. Outdoor_1 degrades early because the open courtyard doesn't give enough stable returns from both sides simultaneously."

---

## If They Ask…

**"Why did you use point-to-plane ICP instead of the scan-to-map hill-climbing from Lab 09?"**
Lab 09 uses hill-climbing scan-to-map matching for simplicity. We used point-to-plane ICP because it produces the Hessian J^T W J as a by-product — that's the per-step inverse covariance, which we pass directly to GTSAM as the edge information matrix in Q3d. Hill-climbing doesn't give you that. It also gives more principled convergence guarantees and handles flat-surface sliding better via the Huber loss.

**"Why log-odds instead of Lab 09's linear model?"**
Lab 09 uses a simplified linear accumulation model for speed. Log-odds is the standard Bayesian approach — it's numerically stable (addition, never multiplies small probabilities), saturates gracefully, and can't overflow. The update is the same conceptually, just in log space.

**"You said GTSAM reduces cost 8–25× but closure worsened — isn't that a failure?"**
No — it's expected behaviour when edge information matrices are unbalanced. GTSAM minimises weighted total inconsistency. Closure error is just one implied metric it never directly sees. The result is correct, the weighting is the problem. The fix — reducing loop-edge confidence relative to odometry — is well-understood; we documented the issue rather than masking it.

**"What's the aperture problem in your context?"**
In a straight corridor, consecutive scans look identical in the forward direction — the cost function is flat along the corridor axis and ICP can't estimate forward translation. Lab 09 Activity 3 explicitly calls this out. Our scan-skip finding on Floor7 reflects exactly this: longer gaps between scans give enough parallax to make forward displacement observable.

**"Why does the 169° flip preserve consistent tracking?"**
Because once the first map is initialised in the flipped coordinate frame, all subsequent tracking uses that same frame as reference. ORB-SLAM2 never sees the flip — it just tracks consistently relative to its own initialised frame. The flip only becomes visible when you align against an independently initialised reference (COLMAP). This is a property of monocular SLAM: coordinate frame is arbitrary, set at initialisation, and consistent but not necessarily correct.
