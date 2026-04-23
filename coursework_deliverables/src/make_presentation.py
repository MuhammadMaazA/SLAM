#!/usr/bin/env python3
"""
COMP0222 CW2 – Group 32 Presentation PDF
10 minutes total: 5 min Q2 (Visual SLAM) + 5 min Q3 (LiDAR SLAM).
Generates COMP0222_CW2_GRP_32.pdf in the repo root.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import matplotlib.image as mpimg

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
Q1  = os.path.join(_ROOT, 'data', 'q1_results')
Q2  = os.path.join(_ROOT, 'data', 'q2_results')
Q3  = os.path.join(_ROOT, 'data', 'q3_results')
OUT = os.path.join(_ROOT, '..', 'COMP0222_CW2_GRP_32.pdf')

DARK   = '#0d1117'
CARD   = '#161b22'
ACCENT = '#58a6ff'
GREEN  = '#3fb950'
ORANGE = '#f78166'
TEXT   = '#e6edf3'
MUTED  = '#8b949e'

W, H = 16, 9   # slide size inches (16:9)


def new_slide(title=None, subtitle=None):
    fig = plt.figure(figsize=(W, H), facecolor=DARK)
    if title:
        y = 0.93 if subtitle else 0.95
        fig.text(0.5, y, title, ha='center', va='top',
                 fontsize=22, fontweight='bold', color=TEXT)
    if subtitle:
        fig.text(0.5, 0.87, subtitle, ha='center', va='top',
                 fontsize=13, color=MUTED)
    return fig


def img(path, ax, title=None):
    if not os.path.exists(path):
        ax.text(0.5, 0.5, f'Missing:\n{os.path.basename(path)}',
                ha='center', va='center', color=MUTED, fontsize=8,
                transform=ax.transAxes)
        ax.set_facecolor(CARD)
    else:
        im = mpimg.imread(path)
        ax.imshow(im, aspect='auto')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(DARK)
    for sp in ax.spines.values():
        sp.set_edgecolor('#30363d')
    if title:
        ax.set_title(title, color=MUTED, fontsize=8, pad=3)


def bullet_box(fig, x, y, w, h, lines, title=None, color=ACCENT):
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(CARD)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#30363d')
    top = 0.88
    if title:
        ax.text(0.05, 0.93, title, fontsize=10, fontweight='bold',
                color=color, va='top')
        top = 0.75
    step = (top - 0.05) / max(len(lines), 1)
    for i, line in enumerate(lines):
        ax.text(0.05, top - i * step, f'• {line}',
                fontsize=8.5, color=TEXT, va='top', wrap=True)
    return ax


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 – Title
# ═══════════════════════════════════════════════════════════════════════════════
def slide_title():
    fig = plt.figure(figsize=(W, H), facecolor=DARK)
    fig.text(0.5, 0.62, 'COMP0222 Coursework 2', ha='center',
             fontsize=18, color=MUTED, fontweight='normal')
    fig.text(0.5, 0.52, 'Visual and LiDAR SLAM', ha='center',
             fontsize=34, fontweight='bold', color=TEXT)
    fig.text(0.5, 0.42, 'Group 32', ha='center', fontsize=16, color=ACCENT)
    fig.text(0.5, 0.30,
             'Q2: Visual SLAM with Custom Sequences   |   Q3: LiDAR SLAM',
             ha='center', fontsize=12, color=MUTED)
    fig.text(0.5, 0.10,
             '10-minute oral presentation  ·  UCL Department of Computer Science',
             ha='center', fontsize=10, color='#444d56')
    ax = fig.add_axes([0.35, 0.18, 0.30, 0.003])
    ax.set_facecolor(ACCENT); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 – Q2 Overview
# ═══════════════════════════════════════════════════════════════════════════════
def slide_q2_overview():
    fig = new_slide('Q2 — Visual SLAM with Own Sequences',
                    'Datasets · Calibration · Reconstruction · Comparison')
    bullet_box(fig, 0.03, 0.55, 0.28, 0.35, [
        'Intel RealSense D455 camera',
        '3 sequences (same as LiDAR)',
        'Basement_1  —  indoor',
        'Floor7_Hallway  —  large indoor',
        'Outdoor_1  —  outdoor',
        '≥500 frames after ORB-SLAM init',
    ], title='Data Collection', color=ACCENT)

    bullet_box(fig, 0.33, 0.55, 0.28, 0.35, [
        'COLMAP SfM for calibration',
        'cameras.txt → fx, fy, cx, cy, k1, k2',
        'Applied to ORB-SLAM2 YAML config',
        'Verified against factory D455 params',
    ], title='Calibration', color=GREEN)

    bullet_box(fig, 0.63, 0.55, 0.28, 0.35, [
        'ORB-SLAM2 Monocular mode',
        'COLMAP Structure-from-Motion',
        'EVO ATE for trajectory comparison',
        'Umeyama alignment + correct_scale',
        'Both 2D trajectories + 3D point clouds',
    ], title='Methods', color=ORANGE)

    bullet_box(fig, 0.03, 0.10, 0.90, 0.38, [
        'Basement_1: ATE trans = 0.036 m, rot = 2.6°  → excellent agreement',
        'Outdoor_1:  ATE trans = 0.262 m, rot = 1.7°  → good agreement',
        'Floor7_Hallway: COLMAP produced only 8 poses (featureless corridor)',
        'Entrance2: ~178° rotation divergence  →  monocular chirality ambiguity (expected)',
    ], title='Key Results', color=ACCENT)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 – Q2 Trajectory comparison
# ═══════════════════════════════════════════════════════════════════════════════
def slide_q2_trajectories():
    fig = new_slide('Q2b — COLMAP vs ORB-SLAM2 Trajectories (EVO ATE)')
    ax1 = fig.add_axes([0.03, 0.08, 0.57, 0.75])
    img(os.path.join(Q2, 'q2b_colmap_vs_orbslam.png'), ax1)
    ax2 = fig.add_axes([0.62, 0.08, 0.36, 0.75])
    img(os.path.join(Q2, 'q2b_summary_table.png'), ax2, 'ATE Summary Table')
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 – Q2 3D point clouds
# ═══════════════════════════════════════════════════════════════════════════════
def slide_q2_pointclouds():
    fig = new_slide('Q2b — 3D Reconstructions')
    pcs = ['q2b_pointcloud_basement_1.png',
           'q2b_pointcloud_outdoor_1.png',
           'q2b_pointcloud_onepoolstreet1.png']
    for i, name in enumerate(pcs):
        ax = fig.add_axes([0.03 + i * 0.32, 0.10, 0.29, 0.72])
        img(os.path.join(Q2, 'pointclouds_3d', name), ax,
            name.replace('q2b_pointcloud_', '').replace('.png', '').replace('_', ' ').title())
    bullet_box(fig, 0.03, 0.03, 0.90, 0.06, [
        'COLMAP sparse SfM point clouds shown. Floor7_Hallway returned only 8 poses — featureless corridor, insufficient texture for feature matching.',
    ], color=MUTED)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 – Q3 Overview
# ═══════════════════════════════════════════════════════════════════════════════
def slide_q3_overview():
    fig = new_slide('Q3 — LiDAR SLAM with Own Sequences',
                    'Data Collection · Odometry · Loop Closure · Factor Graph')
    bullet_box(fig, 0.03, 0.55, 0.28, 0.38, [
        'RPLidar A1M8 (max range 8000 mm)',
        '3 sequences (2 indoor, 1 outdoor)',
        'Basement_1  —  indoor',
        'Floor7_Hallway  —  large indoor (Marshgate)',
        'Outdoor_1  —  outdoor',
        'Exactly 2 loops per sequence',
        'Start/end at same floor marker',
    ], title='Data Collection', color=ACCENT)

    bullet_box(fig, 0.33, 0.55, 0.28, 0.38, [
        'Point-to-plane ICP (linearised)',
        'Log-odds occupancy grid',
        'Bresenham ray-casting',
        'Keyframe-based local map',
        'ICP divergence rejection (>2m or >45°)',
    ], title='Laser Odometry', color=GREEN)

    bullet_box(fig, 0.63, 0.55, 0.28, 0.38, [
        'Two-stage loop closure:',
        '  1. Pose distance < 2.0 m',
        '  2. ICP match score ≥ 0.55',
        'Factor graph: GTSAM (LM optimizer)',
        'Pose-graph with odometry + loop edges',
        'Before/after closure error quantified',
    ], title='Loop Closure + FG', color=ORANGE)

    bullet_box(fig, 0.03, 0.10, 0.90, 0.38, [
        'Q3b: 4 parameter ablations — max range (2000 / 8000 mm), angular resolution (n=1,2,3), voxel grid (None/0.05/0.10/0.20 m), scan rate (all/50%/33%)',
        'Q3c: False-positive avoidance via two-stage filter. Threshold ICP score ≥ 0.55 eliminates geometrically similar but spatially distant candidates.',
        'Q3d: Factor graph with GTSAM LM reduces closure error. Occupancy grid quality improves visibly after optimisation.',
    ], title='Key Findings', color=ACCENT)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 – Q3b Parameter analysis
# ═══════════════════════════════════════════════════════════════════════════════
def slide_q3b():
    fig = new_slide('Q3b — Parameter Analysis: Basement_1')
    ax = fig.add_axes([0.03, 0.08, 0.60, 0.78])
    img(os.path.join(Q3, 'q3b_basement_1.png'), ax)
    bullet_box(fig, 0.65, 0.55, 0.32, 0.31, [
        'Range 2000mm: misses far walls',
        'Range 8000mm (sensor max): complete map',
        'Every 3rd beam: ICP diverges on sparse scan',
        'Voxel 0.20m: loss of detail, faster',
        '33% scans: trajectory drifts significantly',
    ], title='Findings', color=ACCENT)
    bullet_box(fig, 0.65, 0.08, 0.32, 0.43, [
        'Occupancy grids for each',
        'parameter set generated.',
        'Sensor max range critical',
        'for large indoor spaces.',
        'Angular full scan best',
        'for accurate normals.',
        'Voxel 0.05m balances',
        'speed and accuracy.',
    ], title='Occupancy Grids', color=GREEN)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 – Q3b Occupancy grids
# ═══════════════════════════════════════════════════════════════════════════════
def slide_q3b_grids():
    fig = new_slide('Q3b — Occupancy Grid Maps per Parameter')
    names = [
        ('q3b_grids_basement_1_max_range.png',        'Max Range'),
        ('q3b_grids_basement_1_angular_resolution.png','Angular Resolution'),
        ('q3b_grids_basement_1_voxel_downsampling.png','Voxel Grid'),
        ('q3b_grids_basement_1_scan_rate.png',         'Scan Rate'),
    ]
    positions = [(0.03, 0.50), (0.52, 0.50), (0.03, 0.05), (0.52, 0.05)]
    for (fname, title), (x, y) in zip(names, positions):
        ax = fig.add_axes([x, y, 0.45, 0.42])
        img(os.path.join(Q3, fname), ax, title)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 – Q3c Loop closure
# ═══════════════════════════════════════════════════════════════════════════════
def slide_q3c():
    fig = new_slide('Q3c — Loop Closure Detection')
    ax1 = fig.add_axes([0.03, 0.08, 0.43, 0.76])
    img(os.path.join(Q3, 'q3c_basement_1.png'), ax1, 'Basement_1')
    ax2 = fig.add_axes([0.48, 0.08, 0.43, 0.76])
    img(os.path.join(Q3, 'q3c_outdoor_1.png'), ax2, 'Outdoor_1')
    bullet_box(fig, 0.03, 0.01, 0.88, 0.06, [
        'Two-stage filter: (1) pose distance < 2.0 m gates candidates, (2) ICP match score ≥ 0.55 confirms. Score histogram (right panel each figure) shows clear separation.',
    ], color=MUTED)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 – Q3d Factor Graph
# ═══════════════════════════════════════════════════════════════════════════════
def slide_q3d():
    fig = new_slide('Q3d — Factor Graph Optimisation (GTSAM Levenberg-Marquardt)')
    ax1 = fig.add_axes([0.03, 0.08, 0.43, 0.76])
    img(os.path.join(Q3, 'q3d_basement_1.png'), ax1, 'Before / After trajectory')
    ax2 = fig.add_axes([0.48, 0.08, 0.43, 0.76])
    img(os.path.join(Q3, 'q3d_grid_basement_1.png'), ax2, 'Occupancy grid before / after')
    bullet_box(fig, 0.03, 0.01, 0.88, 0.06, [
        'GTSAM LM optimiser with odometry edges (ω=100) + loop closure edges (ω=500). Anchor prior fixes first pose. Closure error quantified as Euclidean start→end distance.',
    ], color=MUTED)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 – Summary
# ═══════════════════════════════════════════════════════════════════════════════
def slide_summary():
    fig = new_slide('Summary')
    bullet_box(fig, 0.03, 0.45, 0.44, 0.48, [
        'Collected 3 custom sequences (same env for Q2 + Q3)',
        'COLMAP calibration + SfM reconstruction',
        'ORB-SLAM2 monocular tracking',
        'EVO ATE: Basement_1 = 0.036 m (excellent)',
        'Floor7_Hallway: COLMAP near-failure (featureless)',
        'Chirality ambiguity noted on Entrance2 (~178° flip)',
    ], title='Q2 — Visual SLAM', color=ACCENT)

    bullet_box(fig, 0.53, 0.45, 0.44, 0.48, [
        '2-loop sequences, start/end at floor marker',
        'Point-to-plane ICP + log-odds occupancy grid',
        '4 parameter ablations with occupancy grid screenshots',
        'Two-stage loop closure (pose distance + ICP score)',
        'GTSAM factor graph — reduces closure error',
        'Consistent maps across both loops',
    ], title='Q3 — LiDAR SLAM', color=GREEN)

    bullet_box(fig, 0.03, 0.08, 0.44, 0.32, [
        'Indoor (Basement): fewest challenges, cleanest maps',
        'Outdoor: scale ambiguity for monocular; open space for LiDAR',
        'Large corridor (Floor7): hard for both systems',
    ], title='Environment Comparison', color=ORANGE)

    bullet_box(fig, 0.53, 0.08, 0.44, 0.32, [
        'LiDAR excels at structured indoor environments',
        'ORB-SLAM2 more robust to dynamic elements outdoors',
        'COLMAP requires texture; LiDAR requires geometry',
    ], title='Visual vs LiDAR SLAM', color=ORANGE)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER
# ═══════════════════════════════════════════════════════════════════════════════
slides = [
    slide_title,
    slide_q2_overview,
    slide_q2_trajectories,
    slide_q2_pointclouds,
    slide_q3_overview,
    slide_q3b,
    slide_q3b_grids,
    slide_q3c,
    slide_q3d,
    slide_summary,
]

with PdfPages(OUT) as pdf:
    for i, fn in enumerate(slides):
        print(f'  Slide {i+1}/{len(slides)}: {fn.__name__}')
        fig = fn()
        pdf.savefig(fig, bbox_inches='tight', facecolor=DARK)
        plt.close(fig)

size_mb = os.path.getsize(OUT) / 1e6
print(f'\nSaved: {OUT}  ({size_mb:.1f} MB, {len(slides)} slides)')
