# Graph Report - coursework_deliverables  (2026-04-18)

## Corpus Check
- Large corpus: 201 files · ~955,305 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder, or use --no-semantic to run AST-only.

## Summary
- 591 nodes · 829 edges · 41 communities detected
- Extraction: 71% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_ORB-SLAM2 Core|ORB-SLAM2 Core]]
- [[_COMMUNITY_LiDAR SLAM Pipeline|LiDAR SLAM Pipeline]]
- [[_COMMUNITY_COLMAP SfM|COLMAP SfM]]
- [[_COMMUNITY_Q1 Benchmark Evaluation|Q1 Benchmark Evaluation]]
- [[_COMMUNITY_Q2 Visual SLAM Sequences|Q2 Visual SLAM Sequences]]
- [[_COMMUNITY_Q3 LiDAR Sequences|Q3 LiDAR Sequences]]
- [[_COMMUNITY_KITTI Dataset|KITTI Dataset]]
- [[_COMMUNITY_TUM-RGBD Dataset|TUM-RGBD Dataset]]
- [[_COMMUNITY_Camera Calibration|Camera Calibration]]
- [[_COMMUNITY_Trajectory Analysis|Trajectory Analysis]]
- [[_COMMUNITY_Occupancy Mapping|Occupancy Mapping]]
- [[_COMMUNITY_Loop Closure|Loop Closure]]
- [[_COMMUNITY_Pose Graph Optimisation|Pose Graph Optimisation]]
- [[_COMMUNITY_Feature Extraction|Feature Extraction]]
- [[_COMMUNITY_ICP Scan Matching|ICP Scan Matching]]
- [[_COMMUNITY_Intel RealSense D455|Intel RealSense D455]]
- [[_COMMUNITY_ORB Feature Parameters|ORB Feature Parameters]]
- [[_COMMUNITY_Outlier Rejection|Outlier Rejection]]
- [[_COMMUNITY_EVO Evaluation|EVO Evaluation]]
- [[_COMMUNITY_Sequence Results|Sequence Results]]
- [[_COMMUNITY_Factor Graph|Factor Graph]]
- [[_COMMUNITY_Coursework Structure|Coursework Structure]]
- [[_COMMUNITY_Baseline Experiments|Baseline Experiments]]
- [[_COMMUNITY_Indoor Sequences|Indoor Sequences]]
- [[_COMMUNITY_Outdoor Sequences|Outdoor Sequences]]
- [[_COMMUNITY_BikeStorage Analysis|BikeStorage Analysis]]
- [[_COMMUNITY_Basement Analysis|Basement Analysis]]
- [[_COMMUNITY_Entrance2 Analysis|Entrance2 Analysis]]
- [[_COMMUNITY_OnePoolStreet Analysis|OnePoolStreet Analysis]]
- [[_COMMUNITY_Washroom Analysis|Washroom Analysis]]
- [[_COMMUNITY_Floor7 Hallway Analysis|Floor7 Hallway Analysis]]
- [[_COMMUNITY_ATE Plots|ATE Plots]]
- [[_COMMUNITY_Parameter Sensitivity|Parameter Sensitivity]]
- [[_COMMUNITY_Data Collection|Data Collection]]
- [[_COMMUNITY_Script Utilities|Script Utilities]]
- [[_COMMUNITY_COLMAP Poses|COLMAP Poses]]
- [[_COMMUNITY_ORB-SLAM2 Trajectories|ORB-SLAM2 Trajectories]]
- [[_COMMUNITY_Calibration Data|Calibration Data]]
- [[_COMMUNITY_Analysis Helpers|Analysis Helpers]]
- [[_COMMUNITY_Visualisation|Visualisation]]
- [[_COMMUNITY_Submission Materials|Submission Materials]]

## God Nodes (most connected - your core abstractions)
1. `COLMAPIntegration` - 19 edges
2. `Q2: Visual SLAM with Custom Sequences (33 marks)` - 15 edges
3. `LiDARSLAMDemo` - 12 edges
4. `FactorGraphOptimizer` - 12 edges
5. `ORB-SLAM2` - 12 edges
6. `VisualSLAMDemo` - 11 edges
7. `CourseworkHelper` - 11 edges
8. `Q1: Visual SLAM with Standard Datasets (30 marks)` - 11 edges
9. `Q3: LiDAR SLAM (37 marks)` - 11 edges
10. `ScreenRecordingHelper` - 9 edges

## Surprising Connections (you probably didn't know these)
- `SLAM Exam Preparation Guide` --companion_to--> `SLAM Theory and Practice Study Guide`  [INFERRED]
  coursework_deliverables/data/part1_analysis/slam_exam_prep.txt → coursework_deliverables/data/part1_analysis/slam_study_guide.txt
- `KITTI Timestamps File` --timestamps_for--> `KITTI Sequence 07 Dataset`  [INFERRED]
  coursework_deliverables/data/part1_analysis/times.txt → coursework_deliverables/COURSEWORK_REQUIREMENTS.md
- `Q2: Visual SLAM with Custom Sequences (33 marks)` --collected_sequence--> `Sequence: Basement_1 (indoor, 4612 frames)`  [EXTRACTED]
  coursework_deliverables/COURSEWORK_REQUIREMENTS.md → coursework_deliverables/SUBMISSION_README.md
- `Submission README` --documents--> `COMP0222 Coursework 2: Visual and LiDAR SLAM`  [EXTRACTED]
  coursework_deliverables/SUBMISSION_README.md → coursework_deliverables/COURSEWORK_REQUIREMENTS.md
- `CLAUDE.md Status Report` --status_report_for--> `COMP0222 Coursework 2: Visual and LiDAR SLAM`  [EXTRACTED]
  coursework_deliverables/CLAUDE.md → coursework_deliverables/COURSEWORK_REQUIREMENTS.md

## Hyperedges (group relationships)
- **KITTI-07 Loop Closure Comparison Experiment** —  [INFERRED 0.90]
- **Sequences sharing Camera Group B calibration** —  [INFERRED 1.00]
- **Sequences sharing Camera Group A calibration** —  [INFERRED 1.00]
- **KITTI_calib_group_A_sequences** —  [INFERRED]
- **KITTI_calib_group_B_sequences** —  [INFERRED]
- **SLAM_benchmark_comparison** —  [INFERRED]
- **Custom Indoor Visual SLAM Dataset (COMP0222 Q2a)** —  [INFERRED]
- **ORB-SLAM2 succeeded on all 9 sequences** —  [1.0]
- **COLMAP succeeded on 8 of 9 sequences (Floor7_Hallway near-failed)** —  [1.0]
- **Indoor repetitive-texture environments challenge COLMAP** —  [0.8]
- **All Basement 2 datasets capture the same corridor scene** —  [1.0]
- **All Indoor Room 1 datasets capture the same bare-wall scene** —  [1.0]
- **Both custom sequences are hostile to COLMAP due to low texture and small baseline** —  [0.95]
- **TUM ablation study: all Q1 conditions on freiburg1_xyz** —  [1.0]
- **KITTI ablation study: all Q1 conditions on KITTI seq07** —  [1.0]
- **Cross-dataset comparison: TUM vs KITTI baseline performance** —  [1.0]
- **TUM ATE Cross-Condition Comparison** —  [0.85]
- **KITTI07 ATE Cross-Condition Comparison** —  [0.85]
- **Q3b LiDAR Parameter Sensitivity Sweep - Floor7_Hallway** —  [0.9]
- **All sequences use ICP score threshold 0.55 for loop closure detection** —  [1.0]
- **Pose graph optimisation produced zero improvement in all tested sequences** —  [0.9]
- **Indoor basement sequences produce coherent occupancy maps; short corridor produces noisy map** —  [0.9]
- **Parameter sensitivity analysis on OnePoolStreet1 shows angular resolution is most critical parameter** —  [0.95]
- **All Q3b sequences compared on max_range sensitivity** —  [0.95]
- **Indoor vs Outdoor loop closure performance across sequences** —  [1.0]
- **Occupancy map quality vs drift across sequences** —  [0.9]
- **Q3b Parameter Sensitivity Study** —  [INFERRED]
- **Q3c Loop Closure Detection Study** —  [INFERRED]
- **Q3d Factor Graph Optimisation Study** —  [INFERRED]
- **All Q3 Log-Odds Occupancy Grid Maps** —  [INFERRED]
- **All Q1 ORB-SLAM2 Experiments** —  [INFERRED]
- **All Factor Graph Optimisation Results** —  [INFERRED]
- **LiDAR SLAM Parameter Sensitivity Across Sequences** —  [INFERRED]
- **hyperedge_q1_kitti_all_configs** —  [INFERRED]
- **hyperedge_q1_tum_all_configs** —  [INFERRED]
- **hyperedge_q2_nine_sequences** —  [INFERRED]
- **hyperedge_colmap_comparison_all_sequences** —  [INFERRED]

## Communities

### Community 0 - "ORB-SLAM2 Core"
Cohesion: 0.05
Nodes (58): KITTI07 Baseline, KITTI07 feat=1500, KITTI07 feat=800 (tracking failure), KITTI07 No Loop Closure, KITTI07 No Outlier Rejection, TUM freiburg1_xyz Baseline, TUM freiburg1_xyz feat=1500, TUM freiburg1_xyz feat=800 (+50 more)

### Community 1 - "LiDAR SLAM Pipeline"
Cohesion: 0.07
Nodes (45): Absolute Trajectory Error (ATE), Bundle Adjustment, CLAUDE.md Status Report, COMP0222 Coursework 2: Visual and LiDAR SLAM, COMP0222 Coursework 2 Requirements, DBoW2/3 Bag-of-Words Loop Detection, EVO Trajectory Evaluation Toolkit, Factor Graph / Pose Graph Optimization (+37 more)

### Community 2 - "COLMAP SfM"
Cohesion: 0.08
Nodes (43): Camera Calibration: Intel RealSense D455 (fx=426.67, fy=426.10, cx=425.34, cy=247.52), COLMAP Structure-from-Motion, Dataset: Basement_2 RGB frames, Dataset: indoor_room_1, Finding: COLMAP nearly failed on Floor7_Hallway, Finding: OnePoolStreet1 is largest ORB-SLAM2 trajectory, Finding: ORB-SLAM2 produces denser pose tracks than COLMAP, Finding: Outdoor sequences have higher COLMAP coverage ratio (+35 more)

### Community 3 - "Q1 Benchmark Evaluation"
Cohesion: 0.09
Nodes (37): build_occupancy_grid(), build_pose_graph(), closure_error(), compose_poses(), detect_loop_closures(), estimate_normals_pca(), icp_scan_to_map(), load_scans() (+29 more)

### Community 4 - "Q2 Visual SLAM Sequences"
Cohesion: 0.08
Nodes (19): COLMAPIntegration, main(), Run COLMAP feature extraction, Simulate feature extraction when COLMAP is not available, Initialize COLMAP integration, Run COLMAP feature matching, Simulate feature matching when COLMAP is not available, Run COLMAP sparse reconstruction (+11 more)

### Community 5 - "Q3 LiDAR Sequences"
Cohesion: 0.11
Nodes (31): KITTI Calibration Group A (fx=707.091), KITTI Calibration Group B (fx=718.856), KITTI Calibration Group C (fx=721.538), 1500 ORB Features, 800 ORB Features, Loop Closure Disabled, KITTI Odometry Additional Sequences, KITTI Camera Group A (fx=707.09) (+23 more)

### Community 6 - "KITTI Dataset"
Cohesion: 0.11
Nodes (19): analyze_results(), create_experiment_report(), create_visualizations(), LiDARSLAM, load_lidar_data(), main(), ICP registration of scan to map, Process a full LiDAR sequence (+11 more)

### Community 7 - "TUM-RGBD Dataset"
Cohesion: 0.12
Nodes (13): demonstrate_factor_graph_optimization(), FactorGraphOptimizer, Optimize pose graph using factor graph formulation, Factor graph optimization for LiDAR SLAM loop closure, Calculate Euclidean distance between start and end pose, Create before/after visualization of optimization, Demonstrate factor graph optimization with synthetic data, Add odometry constraint between consecutive poses (+5 more)

### Community 8 - "Camera Calibration"
Cohesion: 0.12
Nodes (12): LiDARSLAMDemo, main(), Simple ICP scan matching, Detect loop closures based on position, Update global map with new scan, Initialize LiDAR SLAM demonstration, Process a single LiDAR frame, Generate synthetic LiDAR scan for demonstration (+4 more)

### Community 9 - "Trajectory Analysis"
Cohesion: 0.13
Nodes (11): main(), Update map with new 3D points, Process a single frame, Initialize Visual SLAM demonstration, Create synthetic image for demonstration, Update real-time visualization, Run the visual SLAM demonstration, Load RGB-D sequence data (+3 more)

### Community 10 - "Occupancy Mapping"
Cohesion: 0.1
Nodes (21): Intel RealSense D455, Q2 Summary — ORB-SLAM2 Visual SLAM on 9 Collected Sequences, Q2a — ORB-SLAM2 Trajectories All 9 Sequences (3x3 grid, XZ plane, color=time), Basement_1 (Q2 summary), Basement_2 centre view (Q2a), Basement_2 (Q2a detail), Basement_2 (Q2 summary), BikeStorage2 (Q2a) (+13 more)

### Community 11 - "Loop Closure"
Cohesion: 0.15
Nodes (10): CourseworkHelper, main(), Analyze completed work and generate summary, Create comprehensive project summary, Provide help for common coursework questions, Check Part 1 ORB-SLAM2 experiments, Check Part 2 visual SLAM with own sequences, Check Part 3 LiDAR SLAM experiments (+2 more)

### Community 12 - "Pose Graph Optimisation"
Cohesion: 0.14
Nodes (20): Q3b Parameter Analysis - BikeStorage, Q3b Parameter Analysis - BikeStorage2, Q3b Parameter Analysis - Entrance2, Q3b Parameter Analysis - Outdoor_1, Q3c Loop Closure - Basement_2, Q3c Loop Closure - Outdoor_1, Occupancy Grid - BikeStorage2, Occupancy Grid - Entrance2 (+12 more)

### Community 13 - "Feature Extraction"
Cohesion: 0.16
Nodes (17): analyze_baseline_for_sfm(), analyze_sequence_for_colmap(), compare_colmap_orbslam2(), create_markdown_summary(), generate_academic_insights(), generate_recommendations(), get_baseline_explanation(), main() (+9 more)

### Community 14 - "ICP Scan Matching"
Cohesion: 0.18
Nodes (9): main(), Print manual recording instructions, Run demonstration and guide recording process, Initialize screen recording helper, Detect current operating system, Check for screen recording tools, Get expected window geometry for demos, Create platform-specific recording script (+1 more)

### Community 15 - "Intel RealSense D455"
Cohesion: 0.21
Nodes (15): COLMAP Analysis Summary, COLMAP Structure-from-Motion Methodology, COLMAP Success Recommendations, COLMAP vs ORB-SLAM2 Comparison, Basement_2/camera RGB sequence, basement2_dataset RGB sequence, Indoor_Room_1/camera RGB sequence, indoor_room_1_dataset RGB sequence (+7 more)

### Community 16 - "ORB Feature Parameters"
Cohesion: 0.29
Nodes (12): compute_ate(), load_traj(), plot_ate_over_time(), plot_q1a(), plot_q1b(), plot_q1c(), plot_q1d(), plot_summary() (+4 more)

### Community 17 - "Outlier Rejection"
Cohesion: 0.21
Nodes (7): main(), Part1Analyzer, Create comprehensive visualization plots, Extract numerical results from evo zip files, Generate detailed text report, Create comprehensive comparison table, Analyze parameter effects

### Community 18 - "EVO Evaluation"
Cohesion: 0.21
Nodes (13): align_trajectories(), calculate_metrics(), compare_sequence(), load_colmap_trajectory(), load_tum_trajectory(), main(), plot_comparison(), Load trajectory in TUM format (+5 more)

### Community 19 - "Sequence Results"
Cohesion: 0.38
Nodes (9): arc_length(), centre(), load_all_orbslam(), load_colmap(), load_tum(), plot_q2_summary(), plot_q2a(), plot_q2b() (+1 more)

### Community 20 - "Factor Graph"
Cohesion: 0.29
Nodes (9): check_sequence_data(), main(), Test screen recording helper functionality, Check if sequence data is available, Test visual SLAM demo functionality, Test LiDAR SLAM demo functionality, test_lidar_slam_demo(), test_screen_recording_helper() (+1 more)

### Community 21 - "Coursework Structure"
Cohesion: 0.36
Nodes (7): analyze_sequence(), load_tum_trajectory(), main(), plot_trajectories(), Load trajectory in TUM format, Analyze trajectory statistics, Plot both trajectories

### Community 22 - "Baseline Experiments"
Cohesion: 0.25
Nodes (8): Basement_1 has the longest tracked path (5.4 m), OnePodStreet1 has the most tracked poses (5279), ORB-SLAM2 has limited tracking on Outdoor_1, Q2c — ORB-SLAM2 Statistics Across All 9 Collected Sequences, Q2c Sequence Duration Tracked by ORB-SLAM2, Q2c Tracked Path Length per Sequence, Q2c Tracked Pose Count per Sequence, Q2c Average Camera Speed per Sequence

### Community 23 - "Indoor Sequences"
Cohesion: 0.29
Nodes (7): Factor Graph Loop Closure Optimisation (synthetic circular trajectory), Occupancy Grid — BikeStorage, Occupancy Grid — Floor7_Hallway, Occupancy Grid — Washroom, Basement_2 LiDAR SLAM Parameter Sensitivity Analysis, Indoor Room 1 (Washroom) LiDAR SLAM Parameter Sensitivity Analysis, Q3d Factor Graph Optimisation — Outdoor_1

### Community 24 - "Outdoor Sequences"
Cohesion: 0.47
Nodes (5): extract_keyframes(), parse_colmap_images(), Copy every `stride`-th frame listed in rgb.txt to kf_dir., Parse COLMAP images.txt → list of (image_name, tx, ty, tz)., run_colmap_sequence()

### Community 25 - "BikeStorage Analysis"
Cohesion: 0.33
Nodes (6): Q3b LiDAR SLAM Parameter Sensitivity - Floor7_Hallway, Max Range Most Influential LiDAR Parameter, Q3b Angular Resolution Parameter - Floor7_Hallway, Q3b Max Range Parameter - Floor7_Hallway, Q3b Scan Rate Parameter - Floor7_Hallway, Q3b Voxel Downsampling Parameter - Floor7_Hallway

### Community 26 - "Basement Analysis"
Cohesion: 0.4
Nodes (5): KITTI07 Condition 1a: Baseline, KITTI07 Condition 1c: No Outlier Rejection, Outlier Rejection Critical for KITTI07, KITTI07 1a Baseline ATE Trajectory Map, KITTI07 1c No-Outlier-Rejection ATE Trajectory Map

### Community 27 - "Entrance2 Analysis"
Cohesion: 0.83
Nodes (4): Intel RealSense D455 Indoor Collection Session, Intel RealSense D455, Basement2 Custom Sequence, Indoor Room 1 Custom Sequence

### Community 28 - "OnePoolStreet Analysis"
Cohesion: 0.5
Nodes (4): Loop Closure - OnePoolStreet1, Occupancy Grid - OnePoolStreet1, Q3b Parameter Analysis - OnePoolStreet1, OnePoolStreet1

### Community 29 - "Washroom Analysis"
Cohesion: 0.5
Nodes (4): Q3b Parameter Analysis - Washroom, Q3d Factor Graph Optimisation - Washroom, Pose Graph Optimisation Provides Modest Error Reduction, Washroom

### Community 30 - "Floor7 Hallway Analysis"
Cohesion: 0.67
Nodes (3): Q3c LiDAR Loop Closure Detection - Floor7_Hallway, Floor7_Hallway Loop Closures Are High Confidence, Q3c Loop Closure Detection Results - Floor7_Hallway

### Community 31 - "ATE Plots"
Cohesion: 0.67
Nodes (3): Loop Closure - Basement_1, Occupancy Grid - Basement_1, Basement_1

### Community 32 - "Parameter Sensitivity"
Cohesion: 0.67
Nodes (3): Q1a: ORB-SLAM2 Baseline ATE Evaluation — KITTI07 & TUM freiburg1_xyz, Q1b: ORB Feature Count Impact on Tracking — ORB-SLAM2, Q1d: Effect of Disabling Loop Closure — ORB-SLAM2

### Community 33 - "Data Collection"
Cohesion: 0.67
Nodes (3): Part 2: Early Q2 Trajectory Plots (Washroom and Basement_2), Basement_2 (early/part2 view), Washroom (early/part2 view)

### Community 34 - "Script Utilities"
Cohesion: 1.0
Nodes (0): 

### Community 35 - "COLMAP Poses"
Cohesion: 1.0
Nodes (1): TUM baseline achieves sub-centimeter ATE accuracy

### Community 36 - "ORB-SLAM2 Trajectories"
Cohesion: 1.0
Nodes (1): KITTI baseline achieves ~4.4m ATE RMSE on outdoor driving

### Community 37 - "Calibration Data"
Cohesion: 1.0
Nodes (1): Angular Resolution Effect on SLAM

### Community 38 - "Analysis Helpers"
Cohesion: 1.0
Nodes (1): Voxel Downsampling Effect on SLAM

### Community 39 - "Visualisation"
Cohesion: 1.0
Nodes (1): Loop Closure Count vs Environment Type

### Community 40 - "Submission Materials"
Cohesion: 1.0
Nodes (1): Scan Rate Effect on SLAM

## Knowledge Gaps
- **142 isolated node(s):** `Copy every `stride`-th frame listed in rgb.txt to kf_dir.`, `Parse COLMAP images.txt → list of (image_name, tx, ty, tz).`, `Initialize LiDAR SLAM demonstration`, `Load LiDAR sequence data`, `Convert raw scan data to XY points` (+137 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Script Utilities`** (2 nodes): `debug_colmap.py`, `debug_image_preparation()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `COLMAP Poses`** (1 nodes): `TUM baseline achieves sub-centimeter ATE accuracy`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `ORB-SLAM2 Trajectories`** (1 nodes): `KITTI baseline achieves ~4.4m ATE RMSE on outdoor driving`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Calibration Data`** (1 nodes): `Angular Resolution Effect on SLAM`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Analysis Helpers`** (1 nodes): `Voxel Downsampling Effect on SLAM`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Visualisation`** (1 nodes): `Loop Closure Count vs Environment Type`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Submission Materials`** (1 nodes): `Scan Rate Effect on SLAM`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Q2: Visual SLAM with Custom Sequences (33 marks)` connect `COLMAP SfM` to `LiDAR SLAM Pipeline`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._
- **Why does `TUM RGB-D Dataset` connect `LiDAR SLAM Pipeline` to `Q3 LiDAR Sequences`?**
  _High betweenness centrality (0.016) - this node is a cross-community bridge._
- **What connects `Copy every `stride`-th frame listed in rgb.txt to kf_dir.`, `Parse COLMAP images.txt → list of (image_name, tx, ty, tz).`, `Initialize LiDAR SLAM demonstration` to the rest of the system?**
  _142 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `ORB-SLAM2 Core` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._
- **Should `LiDAR SLAM Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.07 - nodes in this community are weakly interconnected._
- **Should `COLMAP SfM` be split into smaller, more focused modules?**
  _Cohesion score 0.08 - nodes in this community are weakly interconnected._
- **Should `Q1 Benchmark Evaluation` be split into smaller, more focused modules?**
  _Cohesion score 0.09 - nodes in this community are weakly interconnected._