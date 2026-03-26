# SLAM Directory Structure

## Clean, Organized Directory Layout

The SLAM directory has been thoroughly cleaned and organized with the following structure:

### 📁 Root Directory: `/home/mmaaz/SLAM/`

```
SLAM/
├── COMP0222_25-26/                 # Course repository (official templates)
│   ├── Courseworks/               # Course templates & specifications
│   ├── DevTool/                   # Development tools
│   ├── Labs/                      # Lab materials and code
│   └── Libraries/                 # Course libraries (+ebe, +g2o, etc.)
│
├── Coursework 2/                   # Current coursework (CW2)
│   ├── COMP0222-249_25-26_ORB_SLAM2/  # ORB-SLAM2 implementation
│   └── lidar_scripts/             # LiDAR processing scripts
│
├── Coursework_1_Completed/         # Completed CW1 (Factor Graphs)
│   ├── +cw1/                      # Implementation code
│   ├── Report/                    # Results and figures
│   └── Libraries/                 # CW1-specific libraries
│
├── datasets/                       # All dataset ZIP files
│   ├── KITTI.zip                  # KITTI poses
│   ├── KITTI_odometry_gray.zip    # KITTI grayscale images (11GB)
│   ├── TUM.zip                    # TUM RGB-D dataset
│   ├── EuRoC.zip                  # EuRoC MAV dataset (1.5GB)
│   ├── UCL.zip                    # UCL dataset with COLMAP
│   ├── data_odometry_poses.zip    # Additional KITTI poses
│   └── tmp_recordings.zip         # Recorded sequences (1.8GB)
│
├── extracted_data/                 # Extracted and processed datasets
│   ├── KITTI/                     # Extracted KITTI data
│   ├── TUM/                       # Extracted TUM data
│   ├── rgbd_dataset_freiburg1_xyz/ # TUM RGB-D sequence
│   ├── UCL_extracted/             # UCL with COLMAP scripts
│   ├── EuRoC_extracted/           # EuRoC data
│   ├── joint_sequence_01/         # Combined camera/LiDAR data
│   └── tmp_recordings/            # Processed recordings
│
├── lecture_materials/              # Course lectures organized by week
│   ├── Week 1/                    # Basic concepts, KF SLAM
│   ├── Week 2/                    # Nonlinear SLAM, Factor Graphs
│   ├── Week 3/                    # Factor Graph examples
│   ├── Week 4/                    # Feature matching, loop closing
│   ├── Week 5/                    # COLMAP, ORB-SLAM2
│   ├── Week 6/                    # Performance quantification
│   ├── Week 7/                    # Point clouds, scan matching
│   └── Week 8/                    # Occupancy grid SLAM
│
├── documentation/                  # All documentation files
│   ├── coursework_progress.md     # CW2 progress checklist
│   ├── CRITICAL_KITTI_ISSUE.md    # KITTI data status
│   ├── DIRECTORY_STRUCTURE.md     # This file
│   └── README.md                  # Main documentation
│
└── results/                        # Output results and logs
    ├── tum_test.txt               # TUM trajectory results
    └── *.jpeg                     # Result visualizations
```

## Key Features

### ✅ **Organized Structure**
- Clear separation between course materials, completed work, and current work
- Datasets separated from extracted data
- Documentation centralized
- Results isolated

### ✅ **No Duplicates** 
- Removed duplicate directories and files
- Consolidated overlapping content
- Clean namespace without conflicts

### ✅ **Preserved Work**
- Coursework 1 completed work safely preserved
- All dataset files maintained
- LiDAR scripts and tools intact
- Course repository unchanged

### ✅ **Ready for CW2**
- ORB-SLAM2 implementation ready
- LiDAR scripts organized and tested
- All required datasets available
- Documentation updated

## Dataset Status

| Dataset | Status | Size | Usage |
|---------|--------|------|-------|
| KITTI | ⚠️ Partial | 11GB | CW2 Part 1 (needs extraction) |
| TUM | ✅ Ready | Extracted | CW2 Part 1 |
| EuRoC | ✅ Ready | 1.5GB | CW2 Part 2 |
| UCL | ✅ Ready | With COLMAP | CW2 Part 2 |
| LiDAR | ✅ Ready | 1.8GB | CW2 Part 3 |

## Next Steps

1. **Extract KITTI data**: `unzip KITTI_odometry_gray.zip`
2. **Test ORB-SLAM2**: Verify build and dependencies
3. **Validate datasets**: Ensure all sequences are complete
4. **Begin CW2 implementation**: Follow coursework_progress.md checklist

## Storage Summary
- **Total space**: ~17GB organized data
- **Freed space**: 26GB (removed slam-windows)
- **Critical files**: All preserved and organized