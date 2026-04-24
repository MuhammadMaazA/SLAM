#!/usr/bin/env python3
"""
COMP0222 CW2 – Group 32 Presentation (.pptx)
10 slides, 16:9, dark theme.
Output: COMP0222_CW2_GRP_32.pptx
"""
import os, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
Q1   = os.path.join(_ROOT, 'data', 'q1_results')
Q2   = os.path.join(_ROOT, 'data', 'q2_results')
Q3   = os.path.join(_ROOT, 'data', 'q3_results')
OUT  = os.path.join(_ROOT, '..', 'COMP0222_CW2_GRP_32.pptx')

# Colours (RGB)
DARK   = RGBColor(0x0d, 0x11, 0x17)
CARD   = RGBColor(0x16, 0x1b, 0x22)
ACCENT = RGBColor(0x58, 0xa6, 0xff)
GREEN  = RGBColor(0x3f, 0xb9, 0x50)
ORANGE = RGBColor(0xf7, 0x81, 0x66)
TEXT   = RGBColor(0xe6, 0xed, 0xf3)
MUTED  = RGBColor(0x8b, 0x94, 0x9e)
WHITE  = RGBColor(0xff, 0xff, 0xff)

SW, SH = Inches(13.33), Inches(7.5)   # 16:9 widescreen


# ── helpers ───────────────────────────────────────────────────────────────────
def new_prs():
    prs = Presentation()
    prs.slide_width  = SW
    prs.slide_height = SH
    return prs


def blank_slide(prs):
    layout = prs.slide_layouts[6]   # completely blank
    slide  = prs.slides.add_slide(layout)
    fill   = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = DARK
    return slide


def add_rect(slide, x, y, w, h, fill_rgb=None, line_rgb=None, line_width_pt=0):
    from pptx.util import Pt
    shape = slide.shapes.add_shape(
        1,   # MSO_SHAPE_TYPE.RECTANGLE
        Inches(x), Inches(y), Inches(w), Inches(h))
    shape.line.fill.background() if line_rgb is None else None
    if fill_rgb:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_rgb
    else:
        shape.fill.background()
    if line_rgb:
        shape.line.color.rgb = line_rgb
        shape.line.width = Pt(line_width_pt)
    else:
        shape.line.fill.background()
    return shape


def add_text(slide, text, x, y, w, h,
             size=18, bold=False, color=TEXT,
             align=PP_ALIGN.LEFT, wrap=True):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    p  = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(size)
    run.font.bold  = bold
    run.font.color.rgb = color
    return tb


def add_image(slide, path, x, y, w, h):
    if not os.path.exists(path):
        # placeholder grey box with filename
        add_rect(slide, x, y, w, h, fill_rgb=CARD)
        add_text(slide, f'[missing]\n{os.path.basename(path)}',
                 x+0.05, y+h/2-0.2, w-0.1, 0.4,
                 size=8, color=MUTED, align=PP_ALIGN.CENTER)
        return
    slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))


def card(slide, x, y, w, h, title, bullets, title_color=ACCENT, bullet_size=11):
    add_rect(slide, x, y, w, h, fill_rgb=CARD)
    add_text(slide, title, x+0.1, y+0.08, w-0.2, 0.3,
             size=13, bold=True, color=title_color)
    for i, b in enumerate(bullets):
        add_text(slide, f'• {b}',
                 x+0.1, y+0.42 + i*0.28, w-0.2, 0.28,
                 size=bullet_size, color=TEXT)


def divider(slide, y=0.78):
    add_rect(slide, 0.3, y, 12.73, 0.02, fill_rgb=ACCENT)


def title_bar(slide, title, subtitle=None):
    add_text(slide, title, 0.3, 0.18, 12.73, 0.55,
             size=28, bold=True, color=TEXT, align=PP_ALIGN.CENTER)
    if subtitle:
        add_text(slide, subtitle, 0.3, 0.75, 12.73, 0.35,
                 size=14, color=MUTED, align=PP_ALIGN.CENTER)
    divider(slide, 1.12)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDES
# ═══════════════════════════════════════════════════════════════════════════════
prs = new_prs()


# ── 1. Title ──────────────────────────────────────────────────────────────────
sl = blank_slide(prs)
add_rect(sl, 0, 0, 13.33, 7.5, fill_rgb=DARK)
add_text(sl, 'COMP0222 Coursework 2',
         1, 1.6, 11.33, 0.5, size=16, color=MUTED, align=PP_ALIGN.CENTER)
add_text(sl, 'Visual and LiDAR SLAM',
         1, 2.2, 11.33, 0.9, size=36, bold=True, color=TEXT, align=PP_ALIGN.CENTER)
add_text(sl, 'Group 32',
         1, 3.2, 11.33, 0.5, size=20, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
add_rect(sl, 4.5, 3.85, 4.33, 0.04, fill_rgb=ACCENT)
add_text(sl, 'Q2: Visual SLAM with Custom Sequences   |   Q3: LiDAR SLAM',
         1, 4.05, 11.33, 0.4, size=13, color=MUTED, align=PP_ALIGN.CENTER)
add_text(sl, '10-minute oral presentation  ·  UCL Department of Computer Science  ·  April 2026',
         1, 6.7, 11.33, 0.35, size=10, color=RGBColor(0x44, 0x4d, 0x56),
         align=PP_ALIGN.CENTER)


# ── 2. Q2 Overview ────────────────────────────────────────────────────────────
sl = blank_slide(prs)
title_bar(sl, 'Q2 — Visual SLAM with Own Sequences',
          'Datasets  ·  Camera Calibration  ·  Reconstruction  ·  EVO Comparison')

card(sl, 0.2, 1.35, 4.1, 2.6, 'Data Collection', [
    'Intel RealSense D455 camera',
    '3 sequences (same as LiDAR)',
    'Basement_1  —  indoor',
    'Floor7_Hallway  —  large indoor',
    'Outdoor_1  —  outdoor',
    '≥500 frames after ORB-SLAM init',
], title_color=ACCENT)

card(sl, 4.6, 1.35, 4.1, 2.6, 'Calibration & Method', [
    'COLMAP SfM → cameras.txt intrinsics',
    'fx, fy, cx, cy, k1, k2 extracted',
    'Applied to ORB-SLAM2 YAML config',
    'ORB-SLAM2 Monocular mode',
    'EVO ATE (Umeyama, correct_scale)',
], title_color=GREEN)

card(sl, 9.0, 1.35, 4.1, 2.6, 'Key Findings', [
    'Basement_1: ATE = 0.036 m ✓',
    'Outdoor_1: ATE = 0.262 m ✓',
    'Floor7: COLMAP 8 poses (no texture)',
    'Entrance2: 178° chirality flip',
    '  (monocular ambiguity — expected)',
], title_color=ORANGE)

card(sl, 0.2, 4.15, 12.9, 2.9, 'What We Did', [
    '• Collected 3 custom sequences covering indoor structured, large corridor, and outdoor environments',
    '• Ran COLMAP SfM to calibrate camera and extract 3D sparse reconstructions',
    '• Ran ORB-SLAM2 monocular with COLMAP intrinsics and compared trajectories using EVO ATE',
    '• Floor7_Hallway: COLMAP near-failure (featureless corridor).  Entrance2: ~178° orientation divergence (chirality ambiguity).',
], title_color=ACCENT, bullet_size=10)


# ── 3. Q2b Trajectories ───────────────────────────────────────────────────────
sl = blank_slide(prs)
title_bar(sl, 'Q2b — COLMAP vs ORB-SLAM2 Trajectories',
          'EVO APE with Umeyama alignment and scale correction')
add_image(sl, os.path.join(Q2, 'q2b_colmap_vs_orbslam.png'),
          0.2, 1.3, 8.5, 5.8)
card(sl, 8.9, 1.3, 4.2, 5.8, 'Interpretation', [
    'Basement_1: 0.036 m — very close',
    'Outdoor_1: 0.262 m — acceptable',
    'OnePoolStreet1: 3.0 m — scale drift',
    'Entrance2: rot=178° — chirality flip',
    'Floor7: 0 matched poses (COLMAP fail)',
], title_color=ACCENT, bullet_size=10)


# ── 4. Q2 3D Point Clouds ─────────────────────────────────────────────────────
sl = blank_slide(prs)
title_bar(sl, 'Q2b — 3D Reconstructions (COLMAP Sparse SfM)')
pcs = [
    ('q2b_pointcloud_basement_1.png',   'Basement_1 (indoor)'),
    ('q2b_pointcloud_outdoor_1.png',    'Outdoor_1 (outdoor)'),
    ('q2b_pointcloud_onepoolstreet1.png','OnePoolStreet1 (outdoor)'),
]
for i, (fname, label) in enumerate(pcs):
    add_image(sl, os.path.join(Q2, 'pointclouds_3d', fname),
              0.2 + i*4.37, 1.3, 4.2, 5.4)
    add_text(sl, label, 0.2 + i*4.37, 6.75, 4.2, 0.35,
             size=10, color=MUTED, align=PP_ALIGN.CENTER)


# ── 5. Q3 Overview ────────────────────────────────────────────────────────────
sl = blank_slide(prs)
title_bar(sl, 'Q3 — LiDAR SLAM with Own Sequences',
          'Data Collection  ·  ICP Odometry  ·  Loop Closure  ·  Factor Graph')

card(sl, 0.2, 1.35, 4.1, 2.75, 'Data Collection', [
    'RPLidar A1M8 (max range 12000 mm)',
    '2 indoor + 1 outdoor sequences',
    'Basement_1  —  indoor',
    'Floor7_Hallway  —  large (Marshgate)',
    'Outdoor_1  —  outdoor',
    'Exactly 2 loops, marked start/end',
], title_color=ACCENT)

card(sl, 4.6, 1.35, 4.1, 2.75, 'Laser Odometry', [
    'Huber-robust point-to-plane ICP',
    'Log-odds occupancy grid',
    'Bresenham ray-casting (unbounded)',
    'Keyframe-based local map (20 KF)',
    'ICP divergence rejection',
    '  (>0.5 m or >25° rejected)',
], title_color=GREEN)

card(sl, 9.0, 1.35, 4.1, 2.75, 'Loop Closure + Factor Graph', [
    'Multi-stage gating:',
    '  1. Temporal separation ≥ 10 KF',
    '  2. Adaptive arc-length gate',
    '  3. Pose distance < 2.0 m',
    '  4. ICP match score ≥ 0.70',
    'GTSAM Levenberg-Marquardt',
    'Per-edge 3×3 info (ICP Hessian)',
    'Loop info tighter than odometry',
], title_color=ORANGE)

card(sl, 0.2, 4.3, 12.9, 2.8, 'Parameter Ablations (Q3b)', [
    '① Max range: 2000 mm vs 12000 mm (sensor max)   ② Angular resolution: full / every 2nd / every 3rd beam',
    '③ Voxel grid: None / 0.05 m / 0.10 m / 0.20 m   ④ Scan rate: all scans / 50% / 33%',
    'Each variation tested on all 3 sequences. Occupancy grids and closure error reported for each setting.',
], title_color=ACCENT, bullet_size=11)


# ── 6. Q3b Parameter Analysis ─────────────────────────────────────────────────
sl = blank_slide(prs)
title_bar(sl, 'Q3b — Parameter Analysis', 'Basement_1 (indoor) — closure error per setting')
add_image(sl, os.path.join(Q3, 'q3b_basement_1.png'), 0.2, 1.3, 8.5, 5.8)
card(sl, 9.0, 1.3, 4.1, 5.8, 'Key Findings', [
    'Range 2000mm: misses far walls',
    'Range 12000mm: complete map',
    '',
    'Every 3rd beam: ICP diverges',
    'Full scan: best normal estimates',
    '',
    'Voxel 0.05m: good speed+detail',
    'Voxel 0.20m: map loses features',
    '',
    '33% scans: trajectory drifts',
    'All scans: lowest closure error',
    '',
    'Sensor max range critical for',
    'large indoor environments.',
], title_color=ACCENT, bullet_size=10)


# ── 7. Q3b Occupancy Grids ────────────────────────────────────────────────────
sl = blank_slide(prs)
title_bar(sl, 'Q3b — Occupancy Grid Maps per Parameter (Basement_1)')
grids = [
    ('q3b_grids_basement_1_max_range.png',        'Max Range (2000 vs 12000 mm)'),
    ('q3b_grids_basement_1_angular_resolution.png','Angular Resolution (full / n=2 / n=3)'),
    ('q3b_grids_basement_1_voxel_downsampling.png','Voxel Grid (None / 0.05 / 0.10 / 0.20 m)'),
    ('q3b_grids_basement_1_scan_rate.png',        'Scan Rate (all / 50% / 33%)'),
]
for i, (fname, label) in enumerate(grids):
    r, c = divmod(i, 2)
    add_image(sl, os.path.join(Q3, fname), 0.2 + c*6.6, 1.3 + r*3.0, 6.3, 2.7)
    add_text(sl, label, 0.2 + c*6.6, 4.0 + r*3.0, 6.3, 0.28,
             size=9, color=MUTED, align=PP_ALIGN.CENTER)


# ── 8. Q3c Loop Closure ───────────────────────────────────────────────────────
sl = blank_slide(prs)
title_bar(sl, 'Q3c — Loop Closure Detection',
          'Two-stage filter: pose distance gate + ICP match score confirmation')
add_image(sl, os.path.join(Q3, 'q3c_basement_1.png'), 0.2, 1.3, 6.3, 5.6)
add_image(sl, os.path.join(Q3, 'q3c_outdoor_1.png'),  6.7, 1.3, 6.3, 5.6)
add_text(sl, 'Basement_1 (indoor)',  0.2, 6.9, 6.3, 0.3, size=10, color=MUTED, align=PP_ALIGN.CENTER)
add_text(sl, 'Outdoor_1 (outdoor)', 6.7, 6.9, 6.3, 0.3, size=10, color=MUTED, align=PP_ALIGN.CENTER)


# ── 9. Q3d Factor Graph ───────────────────────────────────────────────────────
sl = blank_slide(prs)
title_bar(sl, 'Q3d — Factor Graph Optimisation',
          'GTSAM Levenberg-Marquardt  ·  Before vs After loop closure')
add_image(sl, os.path.join(Q3, 'q3d_basement_1.png'),      0.2, 1.3, 6.3, 3.2)
add_image(sl, os.path.join(Q3, 'q3d_grid_basement_1.png'), 6.7, 1.3, 6.3, 3.2)
add_text(sl, 'Trajectory: before / after optimisation', 0.2, 4.5, 6.3, 0.3,
         size=10, color=MUTED, align=PP_ALIGN.CENTER)
add_text(sl, 'Occupancy grid: before / after optimisation', 6.7, 4.5, 6.3, 0.3,
         size=10, color=MUTED, align=PP_ALIGN.CENTER)
card(sl, 0.2, 4.9, 12.9, 2.2, 'Factor Graph Details', [
    '• Odometry edges: per-edge 3×3 information from scan-matched ICP Hessian + diagonal regulariser',
    '• Loop edges: higher information than odometry (tighter σ on x, y, θ)',
    '• Anchor prior on pose 0 fixes gauge. Closure error = Euclidean distance start → end pose.',
], title_color=ACCENT, bullet_size=11)


# ── 10. Summary ───────────────────────────────────────────────────────────────
sl = blank_slide(prs)
title_bar(sl, 'Summary')

card(sl, 0.2, 1.35, 6.3, 3.1, 'Q2 — Visual SLAM', [
    'Collected 3 sequences with RealSense D455',
    'COLMAP calibration + SfM reconstruction',
    'ORB-SLAM2 monocular tracking',
    'EVO ATE: Basement_1 = 0.036 m (excellent)',
    'Chirality ambiguity on Entrance2 (~178°)',
    'Floor7 COLMAP near-failure (featureless)',
], title_color=ACCENT)

card(sl, 6.8, 1.35, 6.3, 3.1, 'Q3 — LiDAR SLAM', [
    '2-loop sequences, marked start/end',
    'Point-to-plane ICP + log-odds grid',
    '4 parameter ablations with grid screenshots',
    'Two-stage loop closure (pose dist + ICP)',
    'GTSAM factor graph reduces closure error',
    'Consistent maps across both loops',
], title_color=GREEN)

card(sl, 0.2, 4.65, 6.3, 2.5, 'Environment Comparison', [
    'Indoor (Basement): cleanest maps for both',
    'Outdoor: scale ambiguity (monocular)',
    '  open space challenges LiDAR too',
    'Corridor (Floor7): hard for both systems',
], title_color=ORANGE)

card(sl, 6.8, 4.65, 6.3, 2.5, 'Visual vs LiDAR SLAM', [
    'LiDAR: excels in structured indoor spaces',
    'ORB-SLAM2: more robust to dynamic elements',
    'COLMAP needs texture; LiDAR needs geometry',
    'Both degrade in featureless environments',
], title_color=ORANGE)


# ── Save ──────────────────────────────────────────────────────────────────────
prs.save(OUT)
size_mb = os.path.getsize(OUT) / 1e6
print(f'Saved: {OUT}  ({size_mb:.1f} MB, {prs.slides.__len__()} slides)')
