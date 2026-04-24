#!/usr/bin/env python3
"""
Build an ORB-SLAM2 monocular YAML from COLMAP cameras.txt (PINHOLE or
SIMPLE_PINHOLE). Use after COLMAP scratch or full reconstruction so ORB-SLAM2
runs with intrinsics estimated from images (coursework Q2 calibration loop).

Usage:
  python generate_orb_yaml_from_colmap.py \\
      --cameras path/to/colmap_runs/Basement_1_sparse/cameras.txt \\
      --out path/to/RealSense_D455_from_colmap_Basement_1.yaml

Environment (optional):
  SLAM_DATA  – default repo data/ root for convenience paths.
"""
from __future__ import annotations

import argparse
import os


def parse_colmap_cameras(path: str) -> tuple[int, int, float, float, float, float]:
    """Return (W, H, fx, fy, cx, cy) from the first PINHOLE / SIMPLE_PINHOLE / OPENCV row."""
    vals = None
    for ln in open(path):
        if ln.startswith('#') or not ln.strip():
            continue
        p = ln.split()
        model = p[1]
        if model not in ('PINHOLE', 'SIMPLE_PINHOLE', 'OPENCV'):
            continue
        w, h = int(p[2]), int(p[3])
        if model == 'SIMPLE_PINHOLE':
            f, cx, cy = map(float, p[4:7])
            fx = fy = f
        elif model == 'PINHOLE':
            fx, fy, cx, cy = map(float, p[4:8])
        else:  # OPENCV
            fx, fy, cx, cy = map(float, p[4:8])
        vals = (w, h, fx, fy, cx, cy)
        break
    if vals is None:
        raise ValueError(f'No PINHOLE/SIMPLE_PINHOLE/OPENCV row in {path}')
    return vals


def emit_yaml(w: int, h: int, fx: float, fy: float, cx: float, cy: float,
              source: str, out_path: str) -> None:
    body = f"""%YAML:1.0
# Auto-generated from COLMAP cameras.txt
# Source: {source}
Camera.fx: {fx}
Camera.fy: {fy}
Camera.cx: {cx}
Camera.cy: {cy}
Camera.k1: 0.0
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0
Camera.width:  {w}
Camera.height: {h}
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
"""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as fh:
        fh.write(body)
    print(f'Wrote {out_path}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cameras', required=True, help='COLMAP cameras.txt')
    ap.add_argument('--out', required=True, help='Output ORB-SLAM2 YAML path')
    args = ap.parse_args()
    w, h, fx, fy, cx, cy = parse_colmap_cameras(args.cameras)
    emit_yaml(w, h, fx, fy, cx, cy, os.path.abspath(args.cameras), args.out)


if __name__ == '__main__':
    main()
