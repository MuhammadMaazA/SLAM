#!/usr/bin/env python3
"""
Run COLMAP on all 9 sequences using every-30th-frame keyframe sampling.
Saves poses to data/q2_results/colmap_runs/{seq}_colmap_poses.txt
"""

import os, shutil, subprocess, json
import numpy as np

_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.abspath(os.path.join(_HERE, '..'))
COLMAP  = os.environ.get('COLMAP_BIN', '/usr/bin/colmap')
REC1    = os.environ.get('SLAM_REC1', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings', 'tmp_recordings'))
REC2    = os.environ.get('SLAM_REC2', os.path.join(os.path.expanduser('~'), 'SLAM', 'extracted_data', 'tmp_recordings2'))
OUT_DIR = os.environ.get('SLAM_COLMAP_OUT', os.path.join(_ROOT, 'data', 'q2_results', 'colmap_runs'))
os.makedirs(OUT_DIR, exist_ok=True)

# Camera params (Intel RealSense D455)
CAM_PARAMS = "426.675,426.104,425.341,247.517,-0.0554,0.0644,-0.00105,0.000458"

SEQUENCES = {
    # Indoor — every 15th frame (2fps, shorter baseline step for more coverage)
    "Basement_1":    (REC1 + "/Basement_1/camera",    15),
    "Basement_2":    (REC1 + "/Basement_2/camera",    15),
    "Floor7_Hallway":(REC1 + "/Floor7_Hallway/camera",15),
    "Washroom":      (REC1 + "/Washroom/camera",      15),
    "BikeStorage":   (REC2 + "/BikeStorage/camera",   15),
    "BikeStorage2":  (REC2 + "/BikeStorage2/camera",  15),
    # Outdoor — every 10th frame (3fps, best texture + motion)
    "Outdoor_1":     (REC1 + "/Outdoor_1/camera",     10),
    "Entrance2":     (REC2 + "/Entrance2/camera",     30),  # 12k frames → every 30th → ~400 kf
    "OnePoolStreet1":(REC2 + "/OnePoolStreet1/camera",10),
}


def extract_keyframes(cam_dir, stride, kf_dir):
    """Copy every `stride`-th frame listed in rgb.txt to kf_dir."""
    shutil.rmtree(kf_dir, ignore_errors=True)
    os.makedirs(kf_dir)
    rgb_txt = os.path.join(cam_dir, "rgb.txt")
    selected = []
    with open(rgb_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            selected.append(line.strip().split())
    # every stride-th
    selected = selected[::stride]
    for ts, relpath in selected:
        src = os.path.join(cam_dir, relpath)
        dst = os.path.join(kf_dir, os.path.basename(relpath))
        if os.path.exists(src):
            shutil.copy2(src, dst)
    n_copied = len(os.listdir(kf_dir))
    return n_copied, selected


def parse_colmap_images(images_txt):
    """Parse COLMAP images.txt → list of (image_name, tx, ty, tz)."""
    poses = []
    with open(images_txt) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 9:
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            name = parts[9]
            poses.append((name, tx, ty, tz, qx, qy, qz, qw))
        i += 2  # skip the POINTS2D line
    poses.sort(key=lambda x: x[0])
    return poses


def run_colmap_sequence(seq_name, cam_dir, stride):
    print(f"\n{'='*60}")
    print(f"COLMAP: {seq_name}")

    pose_out = os.path.join(OUT_DIR, f"{seq_name}_colmap_poses.txt")
    if os.path.exists(pose_out) and os.path.getsize(pose_out) > 0:
        n = sum(1 for _ in open(pose_out))
        print(f"  already done ({n} poses)")
        return True, n

    kf_dir  = f"/tmp/colmap_kf/{seq_name}"
    db_path = f"/tmp/colmap_db/{seq_name}.db"
    sp_dir  = f"/tmp/colmap_sp/{seq_name}"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    os.makedirs(sp_dir, exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    # 1. Extract keyframes
    n_kf, selected = extract_keyframes(cam_dir, stride, kf_dir)
    print(f"  Keyframes extracted: {n_kf} (every {stride}th frame)")
    if n_kf < 10:
        print(f"  SKIP: too few keyframes")
        return False, 0

    # 2. Feature extraction
    r = subprocess.run([COLMAP, "feature_extractor",
        "--database_path", db_path,
        "--image_path", kf_dir,
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.camera_params", CAM_PARAMS,
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "0"],
        capture_output=True, text=True)
    print(f"  Feature extraction: {'ok' if r.returncode == 0 else 'FAILED'}")

    # 3. Exhaustive matching (no GPU, no vocabulary file needed)
    r = subprocess.run([COLMAP, "exhaustive_matcher",
        "--database_path", db_path,
        "--SiftMatching.use_gpu", "0"],
        capture_output=True, text=True)
    print(f"  Matching: {'ok' if r.returncode == 0 else 'FAILED'}")

    # 4. Mapper
    r = subprocess.run([COLMAP, "mapper",
        "--database_path", db_path,
        "--image_path", kf_dir,
        "--output_path", sp_dir,
        "--Mapper.init_min_num_inliers", "30",
        "--Mapper.abs_pose_min_num_inliers", "15",
        "--Mapper.ba_local_num_images", "8"],
        capture_output=True, text=True)
    failed = "failed to create sparse model" in (r.stdout + r.stderr)
    print(f"  Mapper: {'FAILED' if failed else 'ok'}")
    if failed:
        print("  COLMAP could not reconstruct this sequence.")
        # Write empty file so we know we tried
        open(pose_out, 'w').close()
        return False, 0

    # 5. Convert binary → text
    model_dir = os.path.join(sp_dir, "0")
    if os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "images.bin")):
        subprocess.run([COLMAP, "model_converter",
            "--input_path", model_dir,
            "--output_path", model_dir,
            "--output_type", "TXT"],
            capture_output=True, text=True)

    # 6. Parse results
    images_txt = os.path.join(sp_dir, "0", "images.txt")
    if not os.path.exists(images_txt):
        print(f"  No images.txt found")
        open(pose_out, 'w').close()
        return False, 0

    poses = parse_colmap_images(images_txt)
    with open(pose_out, 'w') as f:
        f.write("# name tx ty tz qx qy qz qw\n")
        for p in poses:
            f.write(f"{p[0]} {p[1]:.6f} {p[2]:.6f} {p[3]:.6f} {p[4]:.6f} {p[5]:.6f} {p[6]:.6f} {p[7]:.6f}\n")

    print(f"  SUCCESS: {len(poses)} poses → {pose_out}")
    return True, len(poses)


if __name__ == "__main__":
    results = {}
    for seq_name, (cam_dir, stride) in SEQUENCES.items():
        ok, n = run_colmap_sequence(seq_name, cam_dir, stride)
        results[seq_name] = {"success": ok, "poses": n}

    print("\n" + "="*60)
    print("COLMAP Summary:")
    for seq, r in results.items():
        status = f"{r['poses']} poses" if r['success'] else "FAILED"
        print(f"  {seq:<20}: {status}")
    print(f"\nResults in: {OUT_DIR}")
