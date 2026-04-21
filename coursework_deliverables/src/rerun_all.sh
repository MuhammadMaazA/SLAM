#!/usr/bin/env bash
# =============================================================================
#  rerun_all.sh — regenerate every figure/result in one command.
# -----------------------------------------------------------------------------
# Tested against Ubuntu 22.04 with:
#   - ORB-SLAM2 (built from the patched tree described in
#     docs/ORB_SLAM2_MODIFICATIONS.md, installed under $ORBSLAM_ROOT)
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

# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIG
# ──────────────────────────────────────────────────────────────────────────────
# ORB-SLAM2 install prefix (contains bin/ and etc/orbslam2/)
: "${ORBSLAM_ROOT:=$HOME/ORB_SLAM2/Install}"
# Three ORB-SLAM2 binaries — baseline, no-outlier-rejection, no-loop-closure.
# If you only built one tree, point all three at it and rely on the runtime
# config (less clean — see docs/ORB_SLAM2_MODIFICATIONS.md).
: "${ORBSLAM_BIN_BASELINE:=$ORBSLAM_ROOT/bin/mono_kitti}"
: "${ORBSLAM_BIN_NOOUT:=$ORBSLAM_ROOT/bin/mono_kitti_nooutlier}"
: "${ORBSLAM_BIN_NOLOOP:=$ORBSLAM_ROOT/bin/mono_kitti_noloop}"
: "${ORBSLAM_BIN_TUM_BASELINE:=$ORBSLAM_ROOT/bin/mono_tum}"
: "${ORBSLAM_BIN_TUM_NOOUT:=$ORBSLAM_ROOT/bin/mono_tum_nooutlier}"
: "${ORBSLAM_BIN_TUM_NOLOOP:=$ORBSLAM_ROOT/bin/mono_tum_noloop}"
: "${ORBSLAM_VOCAB:=$ORBSLAM_ROOT/share/ORB_SLAM2/Vocabulary/ORBvoc.txt}"

# Raw data roots
: "${KITTI_SEQ_DIR:=$HOME/SLAM/datasets/kitti/07}"                           # KITTI07 image_2/
: "${TUM_XYZ_DIR:=$HOME/SLAM/datasets/rgbd_dataset_freiburg1_xyz}"
: "${TUM_LONG_DIR:=$HOME/SLAM/datasets/rgbd_dataset_freiburg3_long_office_household}"
: "${SLAM_REC1:=$HOME/SLAM/extracted_data/tmp_recordings/tmp_recordings}"
: "${SLAM_REC2:=$HOME/SLAM/extracted_data/tmp_recordings2}"

# YAMLs (baseline, per-experiment overrides). Feat-count overrides differ only
# in the ORBextractor.nFeatures line.
: "${KITTI_YAML:=$HOME/SLAM/configs/KITTI04-12_custom.yaml}"
: "${KITTI_YAML_F1200:=$HOME/SLAM/configs/KITTI04-12_custom_f1200.yaml}"
: "${KITTI_YAML_F1500:=$HOME/SLAM/configs/KITTI04-12_custom_f1500.yaml}"
: "${TUM_XYZ_YAML:=$HOME/SLAM/configs/TUM1_custom.yaml}"
: "${TUM_XYZ_YAML_F800:=$HOME/SLAM/configs/TUM1_custom_f800.yaml}"
: "${TUM_XYZ_YAML_F1200:=$HOME/SLAM/configs/TUM1_custom_f1200.yaml}"
: "${TUM_XYZ_YAML_F1500:=$HOME/SLAM/configs/TUM1_custom_f1500.yaml}"
: "${TUM_LONG_YAML:=$HOME/SLAM/configs/TUM3_long.yaml}"
: "${D455_YAML:=$HOME/SLAM/configs/RealSense_D455.yaml}"
: "${D455_YAML_LOW:=$HOME/SLAM/configs/RealSense_D455_lowthresh.yaml}"

# Repository root (this script lives in src/, data/ is a sibling)
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
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

# Each orbslam_run() call invokes `mono_kitti`/`mono_tum` then moves
# KeyFrameTrajectory.txt to the desired destination. If the output file is
# empty we don't overwrite a previously successful run.
orbslam_run() {
    local bin=$1 yaml=$2 seq_dir=$3 out_trajectory=$4
    need_file "$bin" || return 1
    need_file "$yaml" || return 1
    need_file "$seq_dir" || return 1
    local tmp="$LOG_DIR/$(basename "$out_trajectory")"
    (cd "$LOG_DIR" && "$bin" "$ORBSLAM_VOCAB" "$yaml" "$seq_dir") \
        > "$LOG_DIR/$(basename "$out_trajectory" .txt).stdout.log" 2>&1
    if [[ -s "$LOG_DIR/KeyFrameTrajectory.txt" ]]; then
        mv "$LOG_DIR/KeyFrameTrajectory.txt" "$out_trajectory"
        local n
        n=$(wc -l < "$out_trajectory")
        log "  → $out_trajectory ($n poses)"
    else
        warn "  ORB-SLAM2 produced no trajectory for $out_trajectory"
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# Q1 — ORB-SLAM2 on KITTI07 & TUM (freiburg1_xyz + freiburg3_long)
# ──────────────────────────────────────────────────────────────────────────────
q1_kitti() {
    local OUT="$DATA_DIR/part1_analysis"
    mkdir -p "$OUT"
    log "Q1 KITTI07 — baseline"
    orbslam_run "$ORBSLAM_BIN_BASELINE" "$KITTI_YAML"         "$KITTI_SEQ_DIR" "$OUT/kitti07-baseline.txt"
    log "Q1 KITTI07 — feat=1200"
    orbslam_run "$ORBSLAM_BIN_BASELINE" "$KITTI_YAML_F1200"   "$KITTI_SEQ_DIR" "$OUT/kitti07-feat1200.txt"
    log "Q1 KITTI07 — feat=1500"
    orbslam_run "$ORBSLAM_BIN_BASELINE" "$KITTI_YAML_F1500"   "$KITTI_SEQ_DIR" "$OUT/kitti07-feat1500.txt"
    log "Q1 KITTI07 — no outlier rejection"
    orbslam_run "$ORBSLAM_BIN_NOOUT"    "$KITTI_YAML"         "$KITTI_SEQ_DIR" "$OUT/kitti07-nooutlier.txt"
    log "Q1 KITTI07 — no loop closure"
    orbslam_run "$ORBSLAM_BIN_NOLOOP"   "$KITTI_YAML"         "$KITTI_SEQ_DIR" "$OUT/kitti07-noloop.txt"
}

q1_tum_xyz() {
    local OUT="$DATA_DIR/part1_analysis"
    log "Q1 TUM freiburg1_xyz — baseline"
    orbslam_run "$ORBSLAM_BIN_TUM_BASELINE" "$TUM_XYZ_YAML"        "$TUM_XYZ_DIR" "$OUT/tum-baseline.txt"
    log "Q1 TUM freiburg1_xyz — feat=800"
    orbslam_run "$ORBSLAM_BIN_TUM_BASELINE" "$TUM_XYZ_YAML_F800"   "$TUM_XYZ_DIR" "$OUT/tum-feat800.txt"
    log "Q1 TUM freiburg1_xyz — feat=1200"
    orbslam_run "$ORBSLAM_BIN_TUM_BASELINE" "$TUM_XYZ_YAML_F1200"  "$TUM_XYZ_DIR" "$OUT/tum-feat1200.txt"
    log "Q1 TUM freiburg1_xyz — feat=1500"
    orbslam_run "$ORBSLAM_BIN_TUM_BASELINE" "$TUM_XYZ_YAML_F1500"  "$TUM_XYZ_DIR" "$OUT/tum-feat1500.txt"
    log "Q1 TUM freiburg1_xyz — no outlier"
    orbslam_run "$ORBSLAM_BIN_TUM_NOOUT"    "$TUM_XYZ_YAML"        "$TUM_XYZ_DIR" "$OUT/tum-nooutlier.txt"
    log "Q1 TUM freiburg1_xyz — no loop"
    orbslam_run "$ORBSLAM_BIN_TUM_NOLOOP"   "$TUM_XYZ_YAML"        "$TUM_XYZ_DIR" "$OUT/tum-noloop.txt"
    # Copy ground-truth for convenient access
    if [[ -f "$TUM_XYZ_DIR/groundtruth.txt" ]]; then
        mkdir -p "$OUT/rgbd_dataset_freiburg1_xyz"
        cp -u "$TUM_XYZ_DIR/groundtruth.txt" "$OUT/rgbd_dataset_freiburg1_xyz/groundtruth.txt"
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

q2_colmap_intrinsics() {
    # Regenerate a D455 YAML from COLMAP's auto-estimated intrinsics
    # (addresses the "COLMAP-derived calibration" ask). Picks Basement_1 by
    # default — tweak SEQ to taste.
    local SEQ="${COLMAP_INTRINSIC_SEQ:-Basement_1}"
    local CAM="$DATA_DIR/q2_results/colmap_runs/${SEQ}_sparse/cameras.txt"
    local OUT_YAML="$DATA_DIR/q2_results/RealSense_D455_colmap_${SEQ}.yaml"
    need_file "$CAM" || return 1
    log "Q2a derive ORB-SLAM2 YAML from COLMAP intrinsics (${SEQ})"
    run "$PYTHON - <<PY
import os, sys, re
cam='$CAM'; out='$OUT_YAML'
vals=None
for ln in open(cam):
    if ln.startswith('#') or not ln.strip(): continue
    p=ln.split()
    # PINHOLE: id model W H fx fy cx cy
    if p[1] in ('PINHOLE','SIMPLE_PINHOLE','OPENCV'):
        W,H=int(p[2]),int(p[3])
        if p[1]=='SIMPLE_PINHOLE':
            f, cx, cy = map(float, p[4:7]); fx=fy=f
        else:
            fx,fy,cx,cy = map(float, p[4:8])
        vals=(W,H,fx,fy,cx,cy); break
assert vals, 'Could not parse COLMAP cameras.txt'
W,H,fx,fy,cx,cy=vals
open(out,'w').write(f'''%YAML:1.0
# Auto-generated from COLMAP intrinsics for {SEQ}
Camera.fx: {fx}
Camera.fy: {fy}
Camera.cx: {cx}
Camera.cy: {cy}
Camera.k1: 0.0
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0
Camera.width:  {W}
Camera.height: {H}
Camera.fps: 30.0
Camera.RGB: 1
ORBextractor.nFeatures: 2000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500
''')
print('wrote', out)
PY"
    # Re-run ORB-SLAM2 on the target sequence with the COLMAP YAML
    local OUT_TXT="$DATA_DIR/q2_results/orbslam_runs/${SEQ}_trajectory_colmap_intrinsics.txt"
    orbslam_run "$ORBSLAM_BIN_TUM_BASELINE" "$OUT_YAML" "$SLAM_REC1/${SEQ}/camera" "$OUT_TXT" || \
        orbslam_run "$ORBSLAM_BIN_TUM_BASELINE" "$OUT_YAML" "$SLAM_REC2/${SEQ}/camera" "$OUT_TXT" || true
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
    run "$PYTHON $REPO_ROOT/src/factor_graph_optimization.py"
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
)

usage() {
    cat <<USAGE
Usage: $0 [--dry-run] <step> [<step> ...]
       $0 [--dry-run] all

Available steps (run sequentially in listed order when "all"):
  q1_kitti              ORB-SLAM2 on KITTI07 (baseline + 4 ablations)
  q1_tum_xyz            ORB-SLAM2 on TUM freiburg1_xyz (baseline + 5 ablations)
  q1_tum_long           ORB-SLAM2 on TUM freiburg3_long_office_household
  q1_evo_plots          Generate all Q1 EVO figures (ATE, RPE, summary)
  q2_orbslam            ORB-SLAM2 on 9 custom D455 sequences
  q2_colmap             COLMAP sparse reconstruction on all sequences
  q2_colmap_intrinsics  Derive ORB-SLAM2 YAML from COLMAP intrinsics, re-run
  q2_python_plots       Q2 visual / EVO / 3D point-cloud figures
  q3_python             Q3 LiDAR SLAM + factor-graph figures
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
