#!/usr/bin/env bash
# =============================================================================
#  rerun_all.sh — regenerate every figure/result in one command.
# -----------------------------------------------------------------------------
# Tested against Ubuntu 22.04 with:
#   - ORB-SLAM2 (external install under $ORBSLAM_ROOT)
#   - COLMAP >= 3.8 on $PATH
#   - evo, gtsam, numpy, scipy, sklearn, matplotlib (see requirements.txt)
#
#  USAGE:
#    Edit the paths in the "USER CONFIG" block below, then:
#      chmod +x src/rerun_all.sh
#      ./src/rerun_all.sh all                    # everything
#      ./src/rerun_all.sh q1                     # just Q1
#      ./src/rerun_all.sh q2_orbslam q3_python   # pick steps by name
#
#  Each step is idempotent — safe to re-run individual steps. Use --dry-run
#  to see the commands without executing them.
# =============================================================================

set -euo pipefail

# Repository root (this script lives in src/, data/ is a sibling)
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"

# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIG
# ──────────────────────────────────────────────────────────────────────────────
# ORB-SLAM2 install prefix (contains bin/ and etc/orbslam2/)
: "${ORBSLAM_ROOT:=$HOME/ORB_SLAM2/Install}"
# Three ORB-SLAM2 binaries — baseline, no-outlier-rejection, no-loop-closure.
# If you only built one tree, point all three at it and rely on the runtime
# config (less clean, but workable).
: "${ORBSLAM_BIN_BASELINE:=$ORBSLAM_ROOT/bin/mono_kitti}"
: "${ORBSLAM_BIN_NOOUT:=$ORBSLAM_ROOT/bin/mono_kitti_nooutlier}"
: "${ORBSLAM_BIN_NOLOOP:=$ORBSLAM_ROOT/bin/mono_kitti_noloop}"
: "${ORBSLAM_BIN_TUM_BASELINE:=$ORBSLAM_ROOT/bin/mono_tum}"
: "${ORBSLAM_BIN_TUM_NOOUT:=$ORBSLAM_ROOT/bin/mono_tum_nooutlier}"
: "${ORBSLAM_BIN_TUM_NOLOOP:=$ORBSLAM_ROOT/bin/mono_tum_noloop}"
# Raw data roots
: "${KITTI_SEQ_DIR:=$HOME/SLAM/extracted_data/KITTI/dataset/sequences/07}"   # KITTI07 image_0/
: "${TUM_XYZ_DIR:=$HOME/SLAM/datasets/TUM/rgbd_dataset_freiburg3_long_office_household}"
: "${TUM_LONG_DIR:=$HOME/SLAM/datasets/TUM/rgbd_dataset_freiburg3_long_office_household}"
: "${SLAM_REC1:=$HOME/SLAM/extracted_data/tmp_recordings/tmp_recordings}"
: "${SLAM_REC2:=$HOME/SLAM/extracted_data/tmp_recordings2}"

# YAMLs (baseline, per-experiment overrides). Feat-count overrides differ only
# in the ORBextractor.nFeatures line.
: "${KITTI_YAML:=$REPO_ROOT/data/part1_analysis/KITTI04-12_custom.yaml}"
: "${KITTI_YAML_F500:=$REPO_ROOT/data/part1_analysis/KITTI04-12_custom_f500.yaml}"
: "${KITTI_YAML_F250:=$REPO_ROOT/data/part1_analysis/KITTI04-12_custom_f250.yaml}"
: "${KITTI_YAML_F100:=$REPO_ROOT/data/part1_analysis/KITTI04-12_custom_f100.yaml}"
: "${TUM_XYZ_YAML:=$REPO_ROOT/data/part1_analysis/TUM1_custom.yaml}"
: "${TUM_XYZ_YAML_F250:=$REPO_ROOT/data/part1_analysis/TUM1_custom_f250.yaml}"
: "${TUM_XYZ_YAML_F500:=$REPO_ROOT/data/part1_analysis/TUM1_custom_f500.yaml}"
: "${TUM_XYZ_YAML_F800:=$REPO_ROOT/data/part1_analysis/TUM1_custom_f800.yaml}"
: "${TUM_LONG_YAML:=$REPO_ROOT/data/part1_analysis/TUM1_custom.yaml}"
: "${D455_YAML:=$REPO_ROOT/data/part1_analysis/RealSense_D455.yaml}"
: "${D455_YAML_LOW:=$REPO_ROOT/data/part1_analysis/RealSense_D455_lowthresh.yaml}"

DATA_DIR="$REPO_ROOT/data"
LOG_DIR="$DATA_DIR/_rerun_logs"
mkdir -p "$LOG_DIR"

export SLAM_DATA="$DATA_DIR"
export SLAM_REC1 SLAM_REC2
PYTHON="${PYTHON:-python3}"
DRY_RUN="${DRY_RUN:-0}"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
log()       { printf '\n\033[1;36m[%s]  %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
warn()      { printf '\033[1;33m[WARN] %s\033[0m\n' "$*"; }
err()       { printf '\033[1;31m[ERR]  %s\033[0m\n' "$*" >&2; }
run()       {
    log "$ $*"
    if [[ "$DRY_RUN" != "1" ]]; then eval "$@"; fi
}
need_file() {
    [[ -e "$1" ]] || { err "Missing path: $1"; return 1; }
}
need_bin()  {
    command -v "$1" >/dev/null 2>&1 || { err "Missing binary on PATH: $1"; return 1; }
}
step_ok()   {
    local name=$1; local marker="$LOG_DIR/.${name}.ok"
    [[ -e "$marker" ]]
}
step_mark() {
    local name=$1; touch "$LOG_DIR/.${name}.ok"
}

# Each orbslam_run() call uses the coursework binaries' CLI:
#   mono_* settings_file sequence_dir results_file
# The binary writes the trajectory directly to the requested output path.
orbslam_run() {
    local bin=$1 yaml=$2 seq_dir=$3 out_trajectory=$4
    need_file "$bin" || return 1
    need_file "$yaml" || return 1
    need_file "$seq_dir" || return 1
    "$bin" "$yaml" "$seq_dir" "$out_trajectory" \
        > "$LOG_DIR/$(basename "$out_trajectory" .txt).stdout.log" 2>&1
    if [[ -s "$out_trajectory" ]]; then
        local n
        n=$(wc -l < "$out_trajectory")
        log "  → $out_trajectory ($n poses)"
    else
        warn "  ORB-SLAM2 produced no trajectory for $out_trajectory"
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# Q1 — ORB-SLAM2 on KITTI07 & TUM long sequence
# ──────────────────────────────────────────────────────────────────────────────
q1_kitti() {
    local OUT="$DATA_DIR/part1_analysis"
    mkdir -p "$OUT"
    log "Q1 KITTI07 — baseline"
    orbslam_run "$ORBSLAM_BIN_BASELINE" "$KITTI_YAML"         "$KITTI_SEQ_DIR" "$OUT/kitti07-baseline.txt"
    log "Q1 KITTI07 — feat=500"
    orbslam_run "$ORBSLAM_BIN_BASELINE" "$KITTI_YAML_F500"    "$KITTI_SEQ_DIR" "$OUT/kitti07-feat500.txt"
    log "Q1 KITTI07 — feat=250"
    orbslam_run "$ORBSLAM_BIN_BASELINE" "$KITTI_YAML_F250"    "$KITTI_SEQ_DIR" "$OUT/kitti07-feat250.txt"
    log "Q1 KITTI07 — feat=100"
    orbslam_run "$ORBSLAM_BIN_BASELINE" "$KITTI_YAML_F100"    "$KITTI_SEQ_DIR" "$OUT/kitti07-feat100.txt"
    log "Q1 KITTI07 — no outlier rejection"
    orbslam_run "$ORBSLAM_BIN_NOOUT"    "$KITTI_YAML"         "$KITTI_SEQ_DIR" "$OUT/kitti07-nooutlier.txt"
    log "Q1 KITTI07 — no loop closure"
    orbslam_run "$ORBSLAM_BIN_NOLOOP"   "$KITTI_YAML"         "$KITTI_SEQ_DIR" "$OUT/kitti07-noloop.txt"
}

q1_tum_xyz() {
    local OUT="$DATA_DIR/part1_analysis"
    log "Q1 TUM freiburg3_long_office_household — baseline"
    orbslam_run "$ORBSLAM_BIN_TUM_BASELINE" "$TUM_XYZ_YAML"        "$TUM_XYZ_DIR" "$OUT/tum-baseline.txt"
    log "Q1 TUM freiburg3_long_office_household — feat=250"
    orbslam_run "$ORBSLAM_BIN_TUM_BASELINE" "$TUM_XYZ_YAML_F250"   "$TUM_XYZ_DIR" "$OUT/tum-feat250.txt"
    log "Q1 TUM freiburg3_long_office_household — feat=500"
    orbslam_run "$ORBSLAM_BIN_TUM_BASELINE" "$TUM_XYZ_YAML_F500"   "$TUM_XYZ_DIR" "$OUT/tum-feat500.txt"
    log "Q1 TUM freiburg3_long_office_household — feat=800"
    orbslam_run "$ORBSLAM_BIN_TUM_BASELINE" "$TUM_XYZ_YAML_F800"   "$TUM_XYZ_DIR" "$OUT/tum-feat800.txt"
    log "Q1 TUM freiburg3_long_office_household — no outlier"
    orbslam_run "$ORBSLAM_BIN_TUM_NOOUT"    "$TUM_XYZ_YAML"        "$TUM_XYZ_DIR" "$OUT/tum-nooutlier.txt"
    log "Q1 TUM freiburg3_long_office_household — no loop"
    orbslam_run "$ORBSLAM_BIN_TUM_NOLOOP"   "$TUM_XYZ_YAML"        "$TUM_XYZ_DIR" "$OUT/tum-noloop.txt"
    # Copy ground-truth for convenient access
    if [[ -f "$TUM_XYZ_DIR/groundtruth.txt" ]]; then
        mkdir -p "$OUT/rgbd_dataset_freiburg3_long_office_household"
        cp -u "$TUM_XYZ_DIR/groundtruth.txt" "$OUT/rgbd_dataset_freiburg3_long_office_household/groundtruth.txt"
    fi
}

q1_tum_long() {
    local OUT="$DATA_DIR/part1_analysis"
    log "Q1 TUM freiburg3_long_office_household — baseline + ablations"
    orbslam_run "$ORBSLAM_BIN_TUM_BASELINE" "$TUM_LONG_YAML" "$TUM_LONG_DIR" "$OUT/tum_long-baseline.txt"
    orbslam_run "$ORBSLAM_BIN_TUM_NOOUT"    "$TUM_LONG_YAML" "$TUM_LONG_DIR" "$OUT/tum_long-nooutlier.txt"
    orbslam_run "$ORBSLAM_BIN_TUM_NOLOOP"   "$TUM_LONG_YAML" "$TUM_LONG_DIR" "$OUT/tum_long-noloop.txt"
    if [[ -f "$TUM_LONG_DIR/groundtruth.txt" ]]; then
        mkdir -p "$OUT/rgbd_dataset_freiburg3_long_office_household"
        cp -u "$TUM_LONG_DIR/groundtruth.txt" \
              "$OUT/rgbd_dataset_freiburg3_long_office_household/groundtruth.txt"
    fi
}

q1_evo_plots() {
    log "Q1 EVO plots + RPE (Python)"
    run "$PYTHON $REPO_ROOT/src/q1_evo_analysis.py"
}

# ──────────────────────────────────────────────────────────────────────────────
# Q2 — ORB-SLAM2 + COLMAP on custom D455 sequences
# ──────────────────────────────────────────────────────────────────────────────
q2_orbslam() {
    log "Q2 ORB-SLAM2 on 9 collected sequences"
    need_file "$SLAM_REC1" || return 1
    need_file "$SLAM_REC2" || return 1
    ORBSLAM="$ORBSLAM_BIN_TUM_BASELINE" \
    YAML_STD="$D455_YAML" YAML_LOW="$D455_YAML_LOW" \
    REC1="$SLAM_REC1" REC2="$SLAM_REC2" \
    OUT="$DATA_DIR/q2_results/orbslam_runs" \
    bash "$_SCRIPT_DIR/run_orbslam_all.sh"
}

q2_colmap() {
    log "Q2 COLMAP sparse reconstructions"
    need_bin colmap || return 1
    run "$PYTHON $REPO_ROOT/src/run_colmap_all.py"
}

q2_colmap_scratch() {
    # Run COLMAP on one sequence with SIMPLE_PINHOLE and no factory prior,
    # so that calibration is derived purely from the images.
    local SEQ="${COLMAP_SCRATCH_SEQ:-Basement_1}"
    log "Q2a COLMAP scratch calibration (${SEQ}, SIMPLE_PINHOLE, no prior)"
    need_bin colmap || return 1
    run "COLMAP_SCRATCH=1 COLMAP_SCRATCH_SEQ=${SEQ} $PYTHON $REPO_ROOT/src/run_colmap_all.py"
}

q2_colmap_intrinsics() {
    # Audit fix: don't stop at "we have calibrated intrinsics". Generate the
    # per-sequence ORB-SLAM2 YAMLs from COLMAP and re-run the custom sequences
    # so the calibrated trajectories are first-class outputs.
    log "Q2a derive ORB-SLAM2 YAMLs from COLMAP intrinsics and re-run calibrated trajectories"
    run "$PYTHON $REPO_ROOT/src/q2_calibration_report.py"
    ORBSLAM="$ORBSLAM_BIN_TUM_BASELINE" \
    YAML_STD="$D455_YAML" YAML_LOW="$D455_YAML_LOW" \
    YAML_DIR_COLMAP="$DATA_DIR/q2_results/orbslam_colmap_yaml" \
    Q2_USE_COLMAP_YAMLS=1 OUT_SUFFIX="_trajectory_colmap_intrinsics" \
    REC1="$SLAM_REC1" REC2="$SLAM_REC2" \
    OUT="$DATA_DIR/q2_results/orbslam_runs" \
    bash "$_SCRIPT_DIR/run_orbslam_all.sh"
}

q2_python_plots() {
    log "Q2 Python plots (visual analysis, EVO, 3D point clouds)"
    run "$PYTHON $REPO_ROOT/src/q2_visual_slam_analysis.py"
    run "$PYTHON $REPO_ROOT/src/q2b_evo_comparison.py"
    run "$PYTHON $REPO_ROOT/src/q2_pointcloud_3d.py"
}

# ──────────────────────────────────────────────────────────────────────────────
# Q3 — LiDAR SLAM
# ──────────────────────────────────────────────────────────────────────────────
q3_python() {
    log "Q3 LiDAR SLAM (ICP, Q3b sweep, Q3c loop, Q3d GTSAM PGO)"
    run "$PYTHON $REPO_ROOT/src/q3_lidar_slam_complete.py"
}

q_factor_graph_demo() {
    log "Synthetic factor-graph GN demo (writes factor_graph_demo_synthetic.png)"
    run "$PYTHON $REPO_ROOT/src/factor_graph_optimization.py"
}

q_merge_coursework_videos() {
    log "Concatenate Q2c + Q3e MP4 (ffmpeg; set SLAM_VIDEO_Q2 / SLAM_VIDEO_Q3 / SLAM_VIDEO_MERGED)"
    need_bin ffmpeg || return 1
    run "$PYTHON $REPO_ROOT/src/merge_coursework_videos.py"
}

# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────
STEPS=(
    q1_kitti
    q1_tum_xyz
    q1_tum_long
    q1_evo_plots
    q2_orbslam
    q2_colmap
    q2_colmap_intrinsics
    q2_python_plots
    q3_python
    q_factor_graph_demo
    q_merge_coursework_videos
)

usage() {
    cat <<USAGE
Usage: $0 [--dry-run] <step> [<step> ...]
       $0 [--dry-run] all

Available steps (run sequentially in listed order when "all"):
  q1_kitti              ORB-SLAM2 on KITTI07 (baseline + 4 ablations)
  q1_tum_xyz            ORB-SLAM2 on TUM freiburg3_long_office_household (baseline + 5 ablations)
  q1_tum_long           ORB-SLAM2 on TUM freiburg3_long_office_household
  q1_evo_plots          Generate all Q1 EVO figures (ATE, RPE, summary)
  q2_orbslam            ORB-SLAM2 on 9 custom D455 sequences
  q2_colmap             COLMAP sparse reconstruction on all sequences
  q2_colmap_scratch     COLMAP on one sequence with SIMPLE_PINHOLE + no factory prior (genuine calibration)
  q2_colmap_intrinsics  Derive ORB-SLAM2 YAML from COLMAP intrinsics, re-run
  q2_python_plots       Q2 visual / EVO / 3D point-cloud figures
  q3_python             Q3 LiDAR SLAM + factor-graph figures
  q_factor_graph_demo   Synthetic 3-pose factor-graph demo (PNG)
  q_merge_coursework_videos  ffmpeg concat of Q2c + Q3e MP4s (optional)
USAGE
}

main() {
    local want=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run) DRY_RUN=1; shift ;;
            -h|--help) usage; exit 0 ;;
            all)       want=("${STEPS[@]}"); shift ;;
            *)         want+=("$1"); shift ;;
        esac
    done
    if [[ ${#want[@]} -eq 0 ]]; then usage; exit 1; fi

    for step in "${want[@]}"; do
        if ! declare -f "$step" >/dev/null; then
            err "Unknown step: $step"; exit 2
        fi
        log "==================  $step  =================="
        "$step" || warn "Step $step reported an error — continuing."
        step_mark "$step"
    done

    log "DONE. Output directory: $DATA_DIR"
    log "Per-step stdout/stderr in: $LOG_DIR"
}

main "$@"
