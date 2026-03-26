# COMP0222 SLAM Coursework Environment

## Directory Structure

```
~/SLAM/
├── COMP0222_25-26/                     # Main course repository
│   ├── Labs/
│   │   ├── Lab_08_-_Point_Cloud/       # ICP code for Part 3
│   │   └── Lab_09_-_2D_Occupancy_Grid/ # Occupancy grid code for Part 3
│   └── Courseworks/
│       ├── Coursework_01/              # Factor Graph SLAM (CW1)
│       └── Coursework_02/              # Visual & LiDAR SLAM (CW2)
│
├── Coursework 2/                       # CW2 working directory
│   ├── COMP0222_-_CW_02.pdf           # Official CW2 specification
│   └── lidar_scripts/                  # LiDAR processing scripts
│
├── datasets/                           # Datasets for experiments
│   ├── KITTI.zip → extract sequence 07
│   ├── TUM.zip → extract long sequence
│   ├── EuRoC.zip
│   └── UCL.zip
│
├── results/                            # Results and output files
│   └── [EVO plots, trajectories, screenshots]
│
├── coursework_progress.md              # Progress tracking
└── README.md                          # This file
```

## Key Lab Code Locations (for Part 3)

- **Lab 8**: `~/SLAM/COMP0222_25-26/Labs/Lab_08_-_Point_Cloud/Solution/rplidar_icp.py`
- **Lab 9**: `~/SLAM/COMP0222_25-26/Labs/Lab_09_-_2D_Occupancy_Grid/Code/occupancy_grid_map_odometry.py`

## Quick Start Checklist

### Immediate (Part 1 - No hardware needed):
1. Download KITTI sequence 07 (22GB)
2. Download TUM long sequence (fr3/long_office_household)
3. Install: `sudo apt install colmap`
4. Run all Part 1 experiments (1a-1d)

### Next (Parts 2&3 - Requires sensors):
1. Borrow Intel RealSense camera + RPLidar during lab sessions
2. Collect data for Parts 2 & 3
3. Process and analyze results

## Important Files Integrated

✅ All coursework specifications and lab code integrated
✅ LiDAR processing scripts available
✅ Factor graph SLAM code from CW1 available
✅ Results directory created

## Environment Status

- WSL2 Ubuntu-24.04: ✅ Ready
- ORB-SLAM2: ✅ Built at ~/ORB_SLAM2/
- EVO: ✅ Installed
- COLMAP: ❌ Need to install
- GTSAM: ❌ Need to install for Part 3

**DEADLINE: Monday 27 April 2026 (50% of module grade)**