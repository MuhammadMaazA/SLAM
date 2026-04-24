#!/bin/bash
# Run ORB-SLAM2 mono_tum on all collected sequences
# Results saved to q2_results/orbslam_runs/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ORBSLAM="${ORBSLAM:-$HOME/ORB_SLAM2/Install/bin/mono_tum}"
YAML_STD="${YAML_STD:-$HOME/ORB_SLAM2/Install/etc/orbslam2/Monocular/RealSense_D455.yaml}"
YAML_LOW="${YAML_LOW:-$HOME/ORB_SLAM2/Install/etc/orbslam2/Monocular/RealSense_D455_lowthresh.yaml}"
YAML_DIR_COLMAP="${YAML_DIR_COLMAP:-$ROOT/data/q2_results/orbslam_colmap_yaml}"
Q2_USE_COLMAP_YAMLS="${Q2_USE_COLMAP_YAMLS:-0}"
OUT_SUFFIX="${OUT_SUFFIX:-_trajectory}"
ORBSLAM_HEADLESS="${ORBSLAM_HEADLESS:-1}"
REC1="${SLAM_REC1:-$HOME/SLAM/extracted_data/tmp_recordings/tmp_recordings}"
REC2="${SLAM_REC2:-$HOME/SLAM/extracted_data/tmp_recordings2}"
OUT="${SLAM_ORBSLAM_OUT:-$ROOT/data/q2_results/orbslam_runs}"
mkdir -p "$OUT"

run_seq() {
    local name=$1
    local cam_dir=$2
    local outfile="$OUT/${name}${OUT_SUFFIX}.txt"
    local yaml_std="$YAML_STD"
    local yaml_low="$YAML_LOW"
    local -a orb_cmd=("$ORBSLAM")
    local yaml_colmap="$YAML_DIR_COLMAP/${name}_D455_colmap.yaml"
    if [ "$Q2_USE_COLMAP_YAMLS" = "1" ] && [ -f "$yaml_colmap" ]; then
        yaml_std="$yaml_colmap"
        # If a calibrated YAML proves too brittle in ORB-SLAM2, fall back to
        # the known-working low-threshold factory config instead of failing the
        # whole audit rerun. This preserves a calibrated-first policy while
        # keeping the batch run robust.
        yaml_low="$YAML_LOW"
    fi
    if [ "$ORBSLAM_HEADLESS" = "1" ] && command -v xvfb-run >/dev/null 2>&1; then
        orb_cmd=(xvfb-run -a "$ORBSLAM")
    fi
    echo "========================================================"
    echo "Running ORB-SLAM2 on: $name"
    echo "  cam_dir : $cam_dir"
    echo "  output  : $outfile"
    echo "  yaml    : $yaml_std"
    echo "  cmd     : ${orb_cmd[*]}"
    echo "========================================================"

    # Try standard config first, then low-threshold if map is empty
    "${orb_cmd[@]}" "$yaml_std" "$cam_dir" "$outfile" 2>&1 | grep -E "Images|tracking|saved|empty|LOST|Init|New Map" | tail -8

    if [ ! -f "$outfile" ] || [ ! -s "$outfile" ]; then
        if [ "$yaml_low" = "$yaml_std" ]; then
            echo "  -> Primary config failed and no alternate fallback YAML is configured."
        else
            echo "  -> Primary config failed, trying fallback YAML: $yaml_low"
            "${orb_cmd[@]}" "$yaml_low" "$cam_dir" "$outfile" 2>&1 | grep -E "Images|tracking|saved|empty|LOST|Init|New Map" | tail -8
        fi
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
