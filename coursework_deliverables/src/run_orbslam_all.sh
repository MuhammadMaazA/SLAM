#!/bin/bash
# Run ORB-SLAM2 mono_tum on all collected sequences
# Results saved to q2_results/orbslam_runs/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ORBSLAM="${ORBSLAM:-$HOME/ORB_SLAM2/Install/bin/mono_tum}"
YAML_STD="${YAML_STD:-$HOME/ORB_SLAM2/Install/etc/orbslam2/Monocular/RealSense_D455.yaml}"
YAML_LOW="${YAML_LOW:-$HOME/ORB_SLAM2/Install/etc/orbslam2/Monocular/RealSense_D455_lowthresh.yaml}"
REC1="${SLAM_REC1:-$HOME/SLAM/extracted_data/tmp_recordings/tmp_recordings}"
REC2="${SLAM_REC2:-$HOME/SLAM/extracted_data/tmp_recordings2}"
OUT="${SLAM_ORBSLAM_OUT:-$ROOT/data/q2_results/orbslam_runs}"
mkdir -p "$OUT"

run_seq() {
    local name=$1
    local cam_dir=$2
    local outfile="$OUT/${name}_trajectory.txt"
    echo "========================================================"
    echo "Running ORB-SLAM2 on: $name"
    echo "  cam_dir : $cam_dir"
    echo "  output  : $outfile"
    echo "========================================================"

    # Try standard config first, then low-threshold if map is empty
    $ORBSLAM "$YAML_STD" "$cam_dir" "$outfile" 2>&1 | grep -E "Images|tracking|saved|empty|LOST|Init" | tail -5

    if [ ! -f "$outfile" ] || [ ! -s "$outfile" ]; then
        echo "  -> Standard config failed, trying low-threshold config..."
        $ORBSLAM "$YAML_LOW" "$cam_dir" "$outfile" 2>&1 | grep -E "Images|tracking|saved|empty|LOST|Init" | tail -5
    fi

    if [ -f "$outfile" ] && [ -s "$outfile" ]; then
        poses=$(wc -l < "$outfile")
        echo "  SUCCESS: $poses poses saved"
    else
        echo "  FAILED: no trajectory saved"
    fi
}

# tmp_recordings sequences
run_seq "Basement_1"    "$REC1/Basement_1/camera"
run_seq "Basement_2"    "$REC1/Basement_2/camera"
run_seq "Floor7_Hallway" "$REC1/Floor7_Hallway/camera"
run_seq "Outdoor_1"     "$REC1/Outdoor_1/camera"
run_seq "Washroom"      "$REC1/Washroom/camera"

# tmp_recordings2 sequences
run_seq "BikeStorage"   "$REC2/BikeStorage/camera"
run_seq "BikeStorage2"  "$REC2/BikeStorage2/camera"
run_seq "Entrance2"     "$REC2/Entrance2/camera"
run_seq "OnePoolStreet1" "$REC2/OnePoolStreet1/camera"

echo ""
echo "All done. Results in $OUT:"
ls -la "$OUT/"
