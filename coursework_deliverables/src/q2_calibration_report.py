#!/usr/bin/env python3
"""
Q2a: Close the COLMAP calibration loop.

The coursework brief asks us to "calibrate the camera using COLMAP" and
then use the result for the visual SLAM pipeline. COLMAP (via
``run_colmap_all.py``) already estimates a per-sequence OPENCV intrinsics
set when reconstructing each sequence. This script:

  1. Reads every ``{seq}_sparse/cameras.txt`` produced by COLMAP.
  2. Builds a per-sequence comparison of factory D455 intrinsics vs the
     COLMAP self-calibration output.
  3. Writes an ORB-SLAM2 YAML (``{seq}_D455_colmap.yaml``) per sequence
     so the SLAM run uses the refined intrinsics — this is the "use the
     calibration result" half of the brief that would otherwise be left
     open.
  4. Emits ``q2a_calibration_report.png`` summarising fx / fy / cx / cy
     across all sequences, highlighting the factory value for reference.

Factory D455 (from the recording session's ``session_info.json`` / the
RealSense SDK): fx=426.675, fy=426.104, cx=425.341, cy=247.517,
k1=-0.0554, k2=0.0644, p1=-0.00105, p2=0.000458 — OPENCV 5-param model.

The YAML template mirrors the existing ``RealSense_D455.yaml`` in
``data/part1_analysis/`` so the only differences are the intrinsics.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))

COLMAP_DIR = os.path.join(_ROOT, 'data', 'q2_results', 'colmap_runs')
OUT_YAML   = os.path.join(_ROOT, 'data', 'q2_results', 'orbslam_colmap_yaml')
OUT_REPORT = os.path.join(_ROOT, 'data', 'q2_results', 'q2a_calibration_report.png')
os.makedirs(OUT_YAML, exist_ok=True)

# Factory RealSense D455 colour intrinsics at 848x480 (RGB stream).
FACTORY = dict(fx=426.675, fy=426.104, cx=425.341, cy=247.517,
               k1=-0.0554, k2=0.0644, p1=-0.00105, p2=0.000458)


def parse_cameras_txt(path):
    """Return dict(model, width, height, params[list]) or None on failure."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            _, model, w, h, *params = parts
            try:
                return dict(model=model, width=int(w), height=int(h),
                            params=[float(x) for x in params])
            except ValueError:
                return None
    return None


def extract_opencv(cam):
    """Pull (fx, fy, cx, cy, k1, k2, p1, p2) from a parsed OPENCV camera."""
    if cam is None or cam['model'] not in ('OPENCV', 'PINHOLE', 'SIMPLE_PINHOLE'):
        return None
    p = cam['params']
    if cam['model'] == 'OPENCV':
        return dict(fx=p[0], fy=p[1], cx=p[2], cy=p[3],
                    k1=p[4] if len(p) > 4 else 0.0,
                    k2=p[5] if len(p) > 5 else 0.0,
                    p1=p[6] if len(p) > 6 else 0.0,
                    p2=p[7] if len(p) > 7 else 0.0)
    if cam['model'] == 'PINHOLE':
        return dict(fx=p[0], fy=p[1], cx=p[2], cy=p[3],
                    k1=0.0, k2=0.0, p1=0.0, p2=0.0)
    # SIMPLE_PINHOLE: f, cx, cy
    return dict(fx=p[0], fy=p[0], cx=p[1], cy=p[2],
                k1=0.0, k2=0.0, p1=0.0, p2=0.0)


YAML_TEMPLATE = """%YAML:1.0
# ORB-SLAM2 intrinsics for sequence '{seq}' using COLMAP-refined calibration.
# Source: coursework_deliverables/data/q2_results/colmap_runs/{seq}_sparse/cameras.txt
# Model: OPENCV (5-parameter radial-tangential).
# Image size: {width}x{height}

Camera.fx: {fx:.6f}
Camera.fy: {fy:.6f}
Camera.cx: {cx:.6f}
Camera.cy: {cy:.6f}

Camera.k1: {k1:.6f}
Camera.k2: {k2:.6f}
Camera.p1: {p1:.6f}
Camera.p2: {p2:.6f}

Camera.width:  {width}
Camera.height: {height}

Camera.fps: 30.0
Camera.RGB: 1

ORBextractor.nFeatures: 1000
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


def write_yaml(seq, intr, width, height):
    path = os.path.join(OUT_YAML, f'{seq}_D455_colmap.yaml')
    with open(path, 'w') as f:
        f.write(YAML_TEMPLATE.format(seq=seq, width=width, height=height,
                                      **intr))
    return path


def main():
    rows = []
    print("=== Q2a: COLMAP calibration report ===\n")
    for entry in sorted(os.listdir(COLMAP_DIR)):
        if not entry.endswith('_sparse'):
            continue
        seq = entry[:-len('_sparse')]
        cam_txt = os.path.join(COLMAP_DIR, entry, 'cameras.txt')
        cam = parse_cameras_txt(cam_txt)
        intr = extract_opencv(cam)
        if intr is None:
            print(f"  {seq:<20} -- no cameras.txt (mapping failed)")
            continue
        yaml_path = write_yaml(seq, intr, cam['width'], cam['height'])
        dfx = 100.0 * (intr['fx'] - FACTORY['fx']) / FACTORY['fx']
        dfy = 100.0 * (intr['fy'] - FACTORY['fy']) / FACTORY['fy']
        dcx = 100.0 * (intr['cx'] - FACTORY['cx']) / FACTORY['cx']
        dcy = 100.0 * (intr['cy'] - FACTORY['cy']) / FACTORY['cy']
        rows.append((seq, intr, dfx, dfy, dcx, dcy))
        print(f"  {seq:<20} fx={intr['fx']:7.2f} ({dfx:+5.2f}%)  "
              f"fy={intr['fy']:7.2f} ({dfy:+5.2f}%)  "
              f"cx={intr['cx']:7.2f} ({dcx:+5.2f}%)  "
              f"cy={intr['cy']:7.2f} ({dcy:+5.2f}%)  "
              f"-> {os.path.basename(yaml_path)}")

    if not rows:
        print("No COLMAP sparse models found; run run_colmap_all.py first.")
        return

    # ── Plot ─────────────────────────────────────────────────────────────
    seqs = [r[0] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        'Q2a: COLMAP self-calibration vs factory D455 intrinsics\n'
        f'(OPENCV model, {rows[0][1]["fx"]:.0f}×{rows[0][1]["fy"]:.0f} effective focal baseline)',
        fontsize=12, fontweight='bold')

    for ax, key, label, fac in [
        (axes[0, 0], 'fx', 'fx (px)', FACTORY['fx']),
        (axes[0, 1], 'fy', 'fy (px)', FACTORY['fy']),
        (axes[1, 0], 'cx', 'cx (px)', FACTORY['cx']),
        (axes[1, 1], 'cy', 'cy (px)', FACTORY['cy']),
    ]:
        vals = [r[1][key] for r in rows]
        ax.bar(range(len(seqs)), vals, color='tab:blue', alpha=0.8, edgecolor='k')
        ax.axhline(fac, color='tab:red', linestyle='--', lw=1.5,
                   label=f'Factory {key}={fac:.2f}')
        ax.set_xticks(range(len(seqs)))
        ax.set_xticklabels(seqs, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(f'{label} per sequence')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8)
        for i, v in enumerate(vals):
            pct = 100.0 * (v - fac) / fac
            ax.text(i, v, f'{pct:+.1f}%', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT_REPORT, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nReport: {OUT_REPORT}")
    print(f"YAMLs  : {OUT_YAML}/  ({len(rows)} files)")


if __name__ == '__main__':
    main()
