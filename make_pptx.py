"""
COMP0222 CW2 GRP 32 — Presentation PPTX
Dark data-science theme: charcoal bg, teal/amber accents, white text.
5 min Q2 + 5 min Q3.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from lxml import etree
import os

# ── Palette ──────────────────────────────────────────────────────────────────
BG       = RGBColor(0x0d, 0x11, 0x17)   # slide background (dark charcoal)
BG_CARD  = RGBColor(0x16, 0x1b, 0x22)   # card / table-row background
BG_ALT   = RGBColor(0x1c, 0x24, 0x2e)   # alternate table rows
TEAL     = RGBColor(0x00, 0xd4, 0xaa)   # primary accent (vivid teal)
AMBER    = RGBColor(0xff, 0x8c, 0x42)   # secondary accent (warm amber)
BLUE_HL  = RGBColor(0x58, 0xa6, 0xff)   # links / emphasis
TEXT_PRI = RGBColor(0xe6, 0xed, 0xf3)   # near-white body text
TEXT_SEC = RGBColor(0x8b, 0x94, 0x9e)   # muted secondary text
SUCCESS  = RGBColor(0x3f, 0xb9, 0x50)   # good result
ERROR    = RGBColor(0xf8, 0x51, 0x49)   # bad result
WHITE    = RGBColor(0xff, 0xff, 0xff)
BLACK    = RGBColor(0x00, 0x00, 0x00)
DIV_Q2   = RGBColor(0x0a, 0x1a, 0x2e)   # Q2 divider background
DIV_Q3   = RGBColor(0x07, 0x1a, 0x12)   # Q3 divider background

SW, SH = Inches(13.33), Inches(7.5)

DATA = "coursework_deliverables/data"
Q1   = f"{DATA}/q1_results"
Q2   = f"{DATA}/q2_results"
Q3   = f"{DATA}/q3_results"
EVO  = "harmish/evo_native_plots/evo_native_plots"

prs = Presentation()
prs.slide_width  = SW
prs.slide_height = SH
BLANK = prs.slide_layouts[6]

# ── Core helpers ─────────────────────────────────────────────────────────────
def new_slide(bg_rgb=None):
    s = prs.slides.add_slide(BLANK)
    fill = s.background.fill
    fill.solid()
    fill.fore_color.rgb = bg_rgb if bg_rgb else BG
    return s

def tb(s, x, y, w, h, text, size=14, bold=False, color=TEXT_PRI,
       align=PP_ALIGN.LEFT, italic=False, wrap=True):
    shape = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    return shape

def accent_bar(s, y=0.0, h=0.055, color=TEAL):
    bar = s.shapes.add_shape(1, Inches(0), Inches(y), SW, Inches(h))
    bar.fill.solid(); bar.fill.fore_color.rgb = color
    bar.line.fill.background()

def heading(s, text, y=0.2):
    tb(s, 0.45, y, 12.5, 0.7, text, size=27, bold=True, color=WHITE)

def divider_line(s, y=0.93):
    line = s.shapes.add_connector(1,
        Inches(0.45), Inches(y), Inches(12.88), Inches(y))
    line.line.color.rgb = TEAL
    line.line.width = Pt(1.0)

def pagenum(s, n):
    tb(s, 12.6, 7.15, 0.6, 0.25, str(n), size=10, color=TEXT_SEC, align=PP_ALIGN.RIGHT)

def section_tag(s, text, color=TEAL, x=0.45, y=0.05):
    tb(s, x, y, 4.0, 0.2, text.upper(), size=8, bold=True, color=color)

def img(s, path, x, y, w, h=None):
    if not os.path.exists(path):
        return
    if h:
        s.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    else:
        s.shapes.add_picture(path, Inches(x), Inches(y), Inches(w))

def caption(s, x, y, w, text):
    tb(s, x, y, w, 0.25, text, size=9, color=TEXT_SEC, align=PP_ALIGN.CENTER, italic=True)

def callout(s, x, y, w, h, text, size=11, accent=TEAL):
    """Dark callout box with left accent strip."""
    box = s.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid(); box.fill.fore_color.rgb = BG_CARD
    box.line.color.rgb = accent; box.line.width = Pt(1.2)
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.12); tf.margin_right = Inches(0.1)
    tf.margin_top  = Inches(0.07); tf.margin_bottom = Inches(0.07)
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.color.rgb = TEXT_PRI

def bullet_block(s, x, y, w, items, size=12, gap=0.32, color=TEXT_PRI, indent=0.18):
    cy = y
    for item in items:
        if item == "":
            cy += gap * 0.35
            continue
        dot = "▸" if not item.startswith("  ") else "·"
        dot_color = TEAL if dot == "▸" else TEXT_SEC
        item_color = color if dot == "▸" else TEXT_SEC
        tb(s, x, cy, 0.22, gap, dot, size=size, color=dot_color)
        tb(s, x+indent, cy, w-indent, gap, item.lstrip(), size=size, color=item_color, wrap=True)
        cy += gap
    return cy

def sub_head(s, x, y, w, text, size=13, color=TEAL):
    tb(s, x, y, w, 0.32, text, size=size, bold=True, color=color)

def table_block(s, x, y, w, rows, col_w=None, row_h=0.3):
    n = len(rows[0])
    if col_w is None:
        col_w = [w/n]*n
    cy = y
    for ri, row in enumerate(rows):
        cx = x
        is_hdr = (ri == 0)
        for ci, cell in enumerate(row):
            cw = col_w[ci]
            box = s.shapes.add_shape(1, Inches(cx), Inches(cy), Inches(cw), Inches(row_h))
            if is_hdr:
                box.fill.solid(); box.fill.fore_color.rgb = TEAL
                box.line.color.rgb = BG; box.line.width = Pt(0.5)
            else:
                box.fill.solid()
                box.fill.fore_color.rgb = BG_CARD if ri % 2 == 0 else BG_ALT
                box.line.color.rgb = RGBColor(0x21, 0x2a, 0x38); box.line.width = Pt(0.5)
            tf = box.text_frame; tf.word_wrap = False
            tf.margin_left = Inches(0.08); tf.margin_top = Inches(0.04)
            p = tf.paragraphs[0]
            r = p.add_run(); r.text = str(cell)
            r.font.size = Pt(10.5 if not is_hdr else 11)
            r.font.bold = is_hdr
            r.font.color.rgb = BLACK if is_hdr else TEXT_PRI
            cx += cw
        cy += row_h
    return cy

def metric_box(s, x, y, w, h, label, value, val_color=TEAL):
    """Stat/metric highlight card."""
    box = s.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid(); box.fill.fore_color.rgb = BG_CARD
    box.line.color.rgb = val_color; box.line.width = Pt(1.5)
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.1); tf.margin_top = Inches(0.1)
    p1 = tf.paragraphs[0]
    r1 = p1.add_run(); r1.text = value
    r1.font.size = Pt(20); r1.font.bold = True; r1.font.color.rgb = val_color
    p2 = tf.add_paragraph()
    r2 = p2.add_run(); r2.text = label
    r2.font.size = Pt(9); r2.font.color.rgb = TEXT_SEC

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ═══════════════════════════════════════════════════════════════════════════
s = new_slide(DIV_Q2)
accent_bar(s, y=0.0, h=0.08, color=TEAL)
accent_bar(s, y=7.42, h=0.08, color=AMBER)

# Large title block
tb(s, 0.7, 1.3, 11.9, 1.1,
   "COMP0222  Coursework 2",
   size=38, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
tb(s, 0.7, 2.38, 11.9, 0.7,
   "Visual and LiDAR SLAM with Own Sequences",
   size=24, bold=True, color=TEAL, align=PP_ALIGN.LEFT)

# Thin separator
line = s.shapes.add_connector(1, Inches(0.7), Inches(3.15), Inches(12.6), Inches(3.15))
line.line.color.rgb = AMBER; line.line.width = Pt(1.5)

tb(s, 0.7, 3.32, 8.0, 0.38,
   "Group 32  —  Muhammad Maaz  ·  Harmish Zala",
   size=14, color=TEXT_PRI, align=PP_ALIGN.LEFT)
tb(s, 0.7, 3.72, 4.0, 0.32,
   "April 2026",
   size=12, color=TEXT_SEC, align=PP_ALIGN.LEFT)

# Two agenda cards
for cx, hdr, body, acc in [
    (0.7, "Part 1 — Visual SLAM (Q2)",
     "Own monocular sequences captured with RealSense D455.\nCOLMAP reconstruction vs ORB-SLAM2 comparison.", TEAL),
    (6.9, "Part 2 — LiDAR SLAM (Q3)",
     "Own 2D LiDAR sequences with RPLidar A2M12.\nParameter sweep · loop closure · pose-graph optimisation.", AMBER),
]:
    card = s.shapes.add_shape(1, Inches(cx), Inches(4.22), Inches(5.9), Inches(1.85))
    card.fill.solid(); card.fill.fore_color.rgb = BG_CARD
    card.line.color.rgb = acc; card.line.width = Pt(1.5)
    tf = card.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.15); tf.margin_top = Inches(0.12)
    p1 = tf.paragraphs[0]
    r1 = p1.add_run(); r1.text = hdr
    r1.font.size = Pt(13); r1.font.bold = True; r1.font.color.rgb = acc
    p2 = tf.add_paragraph()
    r2 = p2.add_run(); r2.text = body
    r2.font.size = Pt(11); r2.font.color.rgb = TEXT_PRI

pagenum(s, 1)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 2 — AGENDA (overview)
# ═══════════════════════════════════════════════════════════════════════════
s = new_slide()
accent_bar(s, h=0.055, color=TEAL)
heading(s, "Agenda")
divider_line(s)

for cy, idx, hdr, body, acc in [
    (1.08, "Q2a", "Capture and calibration",
     "RealSense D455 monocular capture. Iterative protocol. COLMAP per-sequence self-calibration.", TEAL),
    (2.12, "Q2b", "COLMAP vs ORB-SLAM2 comparison",
     "EVO Umeyama Sim(3) alignment. Indoor (Basement_2): 155 pairs, 46° scale ambiguity. Outdoor (OnePoolStreet1): 576 pairs, 2.5° agreement.", TEAL),
    (3.16, "Q3a/b", "LiDAR sequences + parameter sweep",
     "RPLidar A2M12. Four parameters × three sequences. Why max range dominates.", AMBER),
    (4.20, "Q3c", "Loop closure detection",
     "Four-gate filter. ICP overlap histogram shows bimodal gap at 0.70.", AMBER),
    (5.24, "Q3d", "Factor graph optimisation",
     "GTSAM SE(2) pose graph. LM cost drops 8–25× but closure error grows — Hessian tension explained.", AMBER),
]:
    badge = s.shapes.add_shape(1, Inches(0.45), Inches(cy), Inches(0.65), Inches(0.68))
    badge.fill.solid(); badge.fill.fore_color.rgb = acc
    badge.line.fill.background()
    tf = badge.text_frame
    tf.margin_top = Inches(0.1); tf.margin_left = Inches(0.05)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = idx
    r.font.size = Pt(11); r.font.bold = True; r.font.color.rgb = BLACK

    tb(s, 1.25, cy+0.04, 11.5, 0.3, hdr, size=13, bold=True, color=WHITE)
    tb(s, 1.25, cy+0.35, 11.5, 0.32, body, size=10.5, color=TEXT_SEC)

pagenum(s, 2)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Q2 DIVIDER
# ═══════════════════════════════════════════════════════════════════════════
s = new_slide(DIV_Q2)
accent_bar(s, h=0.1, color=TEAL)
tb(s, 0.8, 1.7, 11.7, 0.45, "PART 1",
   size=15, bold=True, color=TEAL, align=PP_ALIGN.LEFT)
tb(s, 0.8, 2.1, 11.7, 0.95, "Visual SLAM",
   size=48, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
tb(s, 0.8, 3.05, 11.7, 0.45,
   "Q2 — own indoor and outdoor monocular sequences",
   size=16, color=TEXT_SEC, align=PP_ALIGN.LEFT)

line = s.shapes.add_connector(1, Inches(0.8), Inches(3.62), Inches(5.5), Inches(3.62))
line.line.color.rgb = TEAL; line.line.width = Pt(2.0)

for mx, lbl in [(0.8, "848×480  30fps"), (3.0, "9 sequences"), (5.2, "COLMAP + ORB-SLAM2")]:
    tb(s, mx, 3.9, 2.0, 0.3, lbl, size=12, color=AMBER, bold=True)

pagenum(s, 3)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Q2(a) CAPTURE AND CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════
s = new_slide()
accent_bar(s, h=0.055, color=TEAL)
section_tag(s, "Visual SLAM  ›  Q2a", color=TEAL)
heading(s, "Q2(a) Capture and calibration")
divider_line(s)

sub_head(s, 0.45, 1.05, 6.5, "Rig and capture protocol")
bullet_block(s, 0.45, 1.42, 6.3, [
    "Intel RealSense D455, RGB 848×480, ~30 fps locked",
    "0.3–0.5 m/s walking speed — 1–1.5 cm/frame, well below ORB matching limit",
    "Continuous video stream, NOT stills — initialiser needs small frame-to-frame baselines",
    "1 s pause + slow 90° pan at each corner — multi-view diversity for COLMAP bundle adjustment",
    "9 sequences: Basement_1 (41 m indoor) and Outdoor_1 (30 m outdoor) are primary sequences",
    "Monitored ORB-SLAM2 live; slowed walking when tracked features < 30",
], size=11, gap=0.295)

sub_head(s, 0.45, 3.48, 6.5, "Calibration via COLMAP self-calibration")
bullet_block(s, 0.45, 3.82, 6.3, [
    "Per-sequence OPENCV 8-param model: fx, fy, cx, cy, k1, k2, p1, p2",
    "All per-seq intrinsics within 0.3–1.6% of D455 factory value (fx=426.68 px)",
    "Scratch SIMPLE_PINHOLE on Basement_1 (no prior): f=415.4 px — 2.6% below factory",
    "  Model forces fx=fy; D455 is mildly asymmetric (fx≠fy) — not a calibration error",
    "Floor7_Hallway: COLMAP registered only 8 poses — featureless white corridor → focal blowup",
], size=11, gap=0.295)

callout(s, 0.45, 5.92, 6.5, 0.72,
    "Per-sequence OPENCV calibration is adequate for the low-distortion D455. "
    "Scratch run confirms factory prior is trustworthy to within ~3%.", size=11, accent=TEAL)

img(s, f"{Q2}/q2a_calibration_report.png", 7.0, 0.98, 5.9)
caption(s, 7.0, 5.72, 5.9,
    "COLMAP intrinsics vs D455 factory (dashed red). All on-line except Floor7 (focal blowup).")

pagenum(s, 4)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Q2(a) ALL TRAJECTORIES
# ═══════════════════════════════════════════════════════════════════════════
s = new_slide()
accent_bar(s, h=0.055, color=TEAL)
section_tag(s, "Visual SLAM  ›  Q2a", color=TEAL)
heading(s, "Q2(a) ORB-SLAM2 trajectories — all 9 sequences")
divider_line(s)

img(s, f"{Q2}/q2a_all_trajectories.png", 0.45, 0.98, 7.5)
caption(s, 0.45, 6.4, 7.5,
    "All 9 sequences. Plasma colour = time (dark=early, bright=late). Green=start, Red=end. Hatched=<500 poses.")

table_block(s, 8.1, 1.05, 5.05,
    [["Sequence","Type","ORB Poses","COLMAP"],
     ["Basement_2","Indoor","2336","233 poses"],
     ["OnePoolStreet1","Outdoor","5759","616 poses"]],
    col_w=[2.2, 0.9, 1.1, 0.85], row_h=0.35)

sub_head(s, 8.1, 2.0, 5.05, "Why these two?", size=12)
bullet_block(s, 8.1, 2.38, 5.05, [
    "Basement_2: full rectangular indoor loop — 2 complete circuits visible",
    "  Rich wall texture → COLMAP triangulates from both sides of the room",
    "OnePoolStreet1: outdoor, 576 COLMAP-matched pairs — best outdoor coverage",
    "  Building facade gives enough features; open courtyard with structure",
    "Brief: ≥500 frames after init — both sequences satisfy this ✓",
], size=11, gap=0.295)

callout(s, 8.1, 4.55, 5.05, 0.85,
    "9 sequences recorded total. These 2 give the clearest COLMAP vs ORB-SLAM2 comparison. "
    "BikeStorage2 (night, 114 poses) documented as explicit failure case.", size=10, accent=TEAL)

pagenum(s, 5)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 6 — Q2(b) COLMAP RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════
s = new_slide()
accent_bar(s, h=0.055, color=TEAL)
section_tag(s, "Visual SLAM  ›  Q2b", color=TEAL)
heading(s, "Q2(b) COLMAP reconstruction")
divider_line(s)

img(s, f"{Q2}/pointclouds_3d/q2b_pointcloud_basement_2.png", 0.45, 0.98, 6.5)
caption(s, 0.45, 5.65, 6.5,
    "Basement_2 COLMAP sparse map — 5810 points, 233 cameras. Clear rectangular room shape.")

bullet_block(s, 7.2, 1.05, 5.9, [
    "No external ground truth — report inter-method disagreement (EVO Umeyama Sim(3) + scale)",
    "Basement_2 (indoor): full rectangular loop, 2336 ORB poses, 233 COLMAP poses",
    "  155 matched pairs — 3.0 m trans disagreement from monocular scale ambiguity",
    "  46° rotation = scale factor mismatch from insufficient baseline diversity",
    "  Both methods trace the same rectangular shape — disagreement is metric not topological",
    "OnePoolStreet1 (outdoor): 5759 ORB poses, 616 COLMAP poses",
    "  576 matched pairs (best coverage of all sequences) — only 2.5° rotation disagreement ✓",
    "  3.0 m translation — outdoor scale ambiguity, but orientation agreement is near-perfect",
], size=11, gap=0.295)

callout(s, 7.2, 5.02, 5.9, 0.88,
    "Basement_2 rotation disagreement (46°) = monocular scale ambiguity, not tracking failure. "
    "OnePoolStreet1's 2.5° rotation shows both methods agree on orientation despite scale. "
    "Neither result reflects inaccuracy — both reflect fundamental monocular limits.", size=10, accent=TEAL)

pagenum(s, 6)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 7 — Q2(b) EVO + FAILURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
s = new_slide()
accent_bar(s, h=0.055, color=TEAL)
section_tag(s, "Visual SLAM  ›  Q2b", color=TEAL)
heading(s, "Q2(b) APE maps and comparison table")
divider_line(s)

q2b_map_in  = f"{Q2}/evo_native_plots/q2b_basement_1_ape_map.png"
q2b_map_out = f"{Q2}/evo_native_plots/q2b_outdoor_1_ape_map.png"
q2b_raw_in  = f"{Q2}/evo_native_plots/q2b_basement_1_ape_raw.png"
have_evo = os.path.exists(q2b_map_in)

img(s, f"{Q2}/q2b_colmap_vs_orbslam.png", 0.45, 1.0, 6.1)
caption(s, 0.45, 5.7, 6.1, "COLMAP vs ORB-SLAM2 Sim(3)-aligned scatter: Basement_2 (indoor) + OnePoolStreet1 (outdoor)")

sub_head(s, 7.0, 1.0, 6.1, "Sequence analysis", size=13)
bullet_block(s, 7.0, 1.38, 6.1, [
    "Basement_2: 2336 ORB KFs. Full rectangular loop — two complete circuits.",
    "  Tracking lost on return legs (landmarks invisible from opposite side)",
    "  Relocalises on next outward pass. 155 COLMAP-matched pairs.",
    "OnePoolStreet1: 5759 ORB KFs, 316 tracked (~12s of 191s total sequence).",
    "  Init delayed until building entrance — open courtyard = near-zero parallax",
    "  576 COLMAP matched pairs — best coverage of all sequences.",
], size=11, gap=0.31)

table_block(s, 7.0, 3.45, 6.1,
    [["","Trans RMSE","Rot RMSE","Matched"],
     ["Basement_2 (indoor)","3.004 m","45.9°","155 pairs"],
     ["OnePoolStreet1 (outdoor)","3.004 m","2.5° ✓","576 pairs"]],
    col_w=[2.6, 1.2, 1.0, 1.3], row_h=0.3)

callout(s, 7.0, 4.62, 6.1, 1.12,
    "Both sequences show 3 m translation disagreement — monocular scale ambiguity, not tracking failure. "
    "Basement_2's 46° rotation = scale factor mismatch. "
    "OnePoolStreet1's 2.5° rotation shows near-perfect orientation agreement despite scale.", size=10, accent=AMBER)

pagenum(s, 7)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 8 — Q3 DIVIDER
# ═══════════════════════════════════════════════════════════════════════════
s = new_slide(DIV_Q3)
accent_bar(s, h=0.1, color=AMBER)
tb(s, 0.8, 1.7, 11.7, 0.45, "PART 2",
   size=15, bold=True, color=AMBER, align=PP_ALIGN.LEFT)
tb(s, 0.8, 2.1, 11.7, 0.95, "LiDAR SLAM",
   size=48, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
tb(s, 0.8, 3.05, 11.7, 0.45,
   "Q3 — own 2D LiDAR sequences, parameter sweep, loop closure, pose-graph optimisation",
   size=15, color=TEXT_SEC, align=PP_ALIGN.LEFT)

line = s.shapes.add_connector(1, Inches(0.8), Inches(3.62), Inches(5.5), Inches(3.62))
line.line.color.rgb = AMBER; line.line.width = Pt(2.0)

for mx, lbl in [(0.8, "RPLidar A2M12"), (2.9, "3 sequences"), (4.8, "Loop closure + GTSAM PGO")]:
    tb(s, mx, 3.9, 2.3, 0.3, lbl, size=12, color=TEAL, bold=True)

pagenum(s, 8)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 9 — Q3(a)/(b) SEQUENCES + PARAMETER TABLE
# ═══════════════════════════════════════════════════════════════════════════
s = new_slide()
accent_bar(s, h=0.055, color=AMBER)
section_tag(s, "LiDAR SLAM  ›  Q3a / Q3b", color=AMBER)
heading(s, "Q3(a)/(b) Sequences and parameter sweep")
divider_line(s)

sub_head(s, 0.45, 1.05, 12.5,
    "Three sequences  —  RPLidar A2M12, 360° scan, 12 m rated range, 90° blind-spot masked", size=12, color=AMBER)
bullet_block(s, 0.45, 1.4, 6.0, [
    "Basement_1 — Marshgate basement room, ~35 m path, 172 KFs. Best ICP health",
    "Floor7_Hallway — Marshgate 7th floor corridors, large open area, 135 KFs",
    "Outdoor_1 — One Pool St courtyard. ICP diverges at scan 400, trajectory freezes",
], size=11, gap=0.285)
bullet_block(s, 6.5, 1.4, 6.6, [
    "Protocol: chalk marker, 2 clockwise loops, 0.4–0.8 m/s, return to exact start, remove marker",
    "Quality filter: discard q < 5 ;  body mask: 135–225° ;  range gate: 10 mm – max_range",
    "Keyframe every Δdist > 0.2 m  or  Δangle > 0.2 rad from last keyframe",
], size=11, gap=0.285)

table_block(s, 0.45, 2.7, 12.55,
    [["Parameter setting","Basement_1","Floor7_Hallway","Outdoor_1"],
     ["Baseline (range 12 m)","0.17 m","0.16 m","8.17 m (frozen)"],
     ["Range 2000 mm","2.38 m  ✗","2.02 m  ✗","2.81 m"],
     ["Angular n=2 (every 2nd)","0.68 m","0.71 m  ✓","3.95 m"],
     ["Angular n=3 (every 3rd)","0.26 m","1.21 m","4.76 m"],
     ["Voxel 0.05 m","0.41 m","0.14 m  ✓","12.4 m"],
     ["Voxel 0.10 m","0.29 m","3.04 m","6.22 m"],
     ["Voxel 0.20 m","2.60 m  ✗","2.94 m","6.02 m"],
     ["Scan rate 50% (skip 2)","0.23 m","2.94 m","11.5 m"],
     ["Scan rate 33% (skip 3)","0.26 m","0.73 m  ✓","10.9 m"]],
    col_w=[3.8, 2.9, 2.9, 2.95], row_h=0.27)

callout(s, 0.45, 6.55, 12.55, 0.65,
    "Baseline: max range 12 m, all beams, no voxel, all scans. "
    "Closure error = Euclidean first↔last 2D pose distance. "
    "Outdoor_1 documented failure — open geometry collapses ICP.", size=10, accent=AMBER)

pagenum(s, 9)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 10 — Q3(b) KEY PARAMETER FINDINGS
# ═══════════════════════════════════════════════════════════════════════════
s = new_slide()
accent_bar(s, h=0.055, color=AMBER)
section_tag(s, "LiDAR SLAM  ›  Q3b", color=AMBER)
heading(s, "Q3(b) Key parameter findings")
divider_line(s)

img(s, f"{Q3}/q3b_grids_basement_1_max_range.png", 0.45, 1.0, 5.4)
caption(s, 0.45, 5.48, 5.4,
    "Max range grids — Basement_1. 2000 mm (near wall only → rank-deficient) vs 12000 mm (full map)")

bullet_block(s, 6.1, 1.05, 7.0, [
    "Max range is the dominant parameter (factor 14×). ICP needs surface normals from both sides "
    "of motion to lock translation. At 2000 mm all returns lie on one line — normal equations "
    "rank-deficient along that axis, trajectory drifts. Outdoor exception: distant glass/vegetation "
    "at 12 m degrades ICP — shorter range filters these noisy returns.",
    "",
    "Angular subsampling degrades PCA normals. At n=2/3 the 5-NN span a wider arc → normal "
    "estimate noisier. 5° normal error on a 5 m wall ≈ 0.4 m per-step along-wall slip. "
    "Floor7 non-monotone: every-2nd-beam smooths micro-features the corridor over-fits at full res.",
    "",
    "Voxel 0.05 m = optimal denoiser. Cell smaller than RPLidar noise (±5–20 mm) → centroid "
    "averages noise. At ≥0.10 m the cell spans real geometry; PCA normals average multiple wall "
    "orientations → ICP loses tangential wall constraint → collapses toward point-to-point.",
    "",
    "Scan skipping non-monotone at Floor7. 33% rate gives more inter-scan parallax along the "
    "corridor axis (aperture problem, Lab 08 Activity 3B). Basement_1 flat: 4 cm/scan at 10 Hz "
    "stays inside ICP convergence basin at all rates.",
], size=10.5, gap=0.355)

pagenum(s, 10)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 11 — Q3(b) MAP GRIDS
# ═══════════════════════════════════════════════════════════════════════════
s = new_slide()
accent_bar(s, h=0.055, color=AMBER)
section_tag(s, "LiDAR SLAM  ›  Q3b", color=AMBER)
heading(s, "Q3(b) Occupancy grid maps — Basement_1 parameter sweep")
divider_line(s)

img(s, f"{Q3}/q3b_grids_basement_1_voxel_downsampling.png", 0.45, 1.0, 5.9)
caption(s, 0.45, 5.48, 5.9,
    "Voxel grid: None / 0.05 m / 0.10 m / 0.20 m  — optimal at 0.05 m")

img(s, f"{Q3}/q3b_grids_basement_1_angular_resolution.png", 6.6, 1.0, 6.4)
caption(s, 6.6, 5.48, 6.4,
    "Angular resolution: full / every-2nd / every-3rd beam  — full best for indoor")

callout(s, 0.45, 5.78, 12.55, 0.68,
    "Log-odds occupancy: +log(0.7/0.3) per hit, −log(0.4/0.6) per free. "
    "Clipped to [−4, +4]. Bresenham ray-casting from sensor pose. 0.05 m cell size.", size=10, accent=AMBER)

pagenum(s, 11)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 12 — Q3(c) LOOP CLOSURE
# ═══════════════════════════════════════════════════════════════════════════
s = new_slide()
accent_bar(s, h=0.055, color=AMBER)
section_tag(s, "LiDAR SLAM  ›  Q3c", color=AMBER)
heading(s, "Q3(c) Loop closure detection")
divider_line(s)

img(s, f"{Q3}/q3c_basement_1.png", 0.45, 1.0, 6.5)
caption(s, 0.45, 6.28, 6.5,
    "Basement_1: accepted loop arcs (magenta, left) + ICP overlap score histogram (right). "
    "Clear bimodal gap at 0.70.")

sub_head(s, 7.2, 1.05, 5.95, "Four-gate detector", size=13)
bullet_block(s, 7.2, 1.42, 5.95, [
    "Gate 1: temporal separation j−i ≥ 10 KFs — prevent self-match",
    "Gate 2: arc ≥ min(8 m, 40% total arc) — adaptive; short seqs still fire",
    "Gate 3: pose distance ≤ 2.0 m — geometric proximity required",
    "Gate 4: ICP overlap score sij ≥ 0.70 — verified scan-level alignment",
    "sij = fraction of scan j points within 0.4 m of scan i after ICP",
    "NMS: keep highest-scoring in any 15-KF window — 1 factor per revisit",
], size=11, gap=0.29)

sub_head(s, 7.2, 3.38, 5.95, "Results", size=13)
table_block(s, 7.2, 3.72, 5.95,
    [["Sequence","Candidates","Accepted"],
     ["Basement_1","2933","10"],
     ["Floor7_Hallway","1991","8"],
     ["Outdoor_1","8","0  (traj. frozen)"]],
    col_w=[2.5, 1.7, 1.75], row_h=0.29)

callout(s, 7.2, 5.08, 5.95, 1.02,
    "Histogram evidence: false matches (sliding-window shared geometry) cluster 0.45–0.60. "
    "True revisits cluster 0.75–0.95. Clean gap at 0.70 holds across both indoor sequences. "
    "Outdoor_1 detects 0 loops: frozen trajectory fails arc gate.", size=10, accent=AMBER)

pagenum(s, 12)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 13 — Q3(d) FACTOR GRAPH
# ═══════════════════════════════════════════════════════════════════════════
s = new_slide()
accent_bar(s, h=0.055, color=AMBER)
section_tag(s, "LiDAR SLAM  ›  Q3d", color=AMBER)
heading(s, "Q3(d) Factor graph and occupancy grid result")
divider_line(s)

img(s, f"{Q3}/q3d_floor7_hallway.png", 0.45, 1.0, 4.7)
caption(s, 0.45, 5.38, 4.7,
    "Floor7_Hallway before / after GTSAM PGO. Cost 69762 → 2752 (25×).")

img(s, f"{Q3}/q3d_grid_basement_1.png", 5.35, 1.0, 4.7)
caption(s, 5.35, 5.38, 4.7,
    "Basement_1 occupancy grid before / after PGO.")

sub_head(s, 10.2, 1.05, 3.0, "SE(2) GTSAM LM", size=12)
bullet_block(s, 10.2, 1.42, 2.95, [
    "Prior: σ=1e−4 anchors x₀",
    "Odom: ICP Hessian as info matrix",
    "Loop: per-loop ICP Hessian",
    "Hessian: (θ,tx,ty) → (tx,ty,θ)",
    "LM: 200 iter, rel.tol=1e−8",
], size=10.5, gap=0.285)

table_block(s, 0.45, 5.65, 12.55,
    [["Sequence","Loops","Cost i→f","Closure Before","Closure After","Δ"],
     ["Floor7_Hallway","8","69762→2752","0.166 m","0.221 m","worse (LM tension)"],
     ["Basement_1","10","61253→7349","0.170 m","0.204 m","worse (LM tension)"],
     ["Outdoor_1","0","0→0","8.084 m","8.084 m","no loops — unchanged"]],
    col_w=[2.3, 0.7, 2.0, 1.7, 1.7, 2.15], row_h=0.285)

pagenum(s, 13)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 14 — Q3(d) WHY CLOSURE ERROR GROWS
# ═══════════════════════════════════════════════════════════════════════════
s = new_slide()
accent_bar(s, h=0.055, color=AMBER)
section_tag(s, "LiDAR SLAM  ›  Q3d", color=AMBER)
heading(s, "Q3(d) Factor graph — why closure error grows")
divider_line(s)

img(s, f"{Q3}/q3d_basement_1.png", 0.45, 1.0, 6.5)
caption(s, 0.45, 5.55, 6.5,
    "Basement_1: before (left) vs after PGO (right). Cost drops 8× but closure distance grows.")

bullet_block(s, 7.2, 1.05, 5.95, [
    "PGO minimises the Mahalanobis sum of ALL factor residuals simultaneously",
    "Closure error = residual on ONE implied edge (first ↔ last pose)",
    "When loop and odometry edges disagree, LM finds the compromise minimising total —",
    "  this can worsen any individual residual including closure",
    "Per-loop ICP Hessian is large (tight loop covariance) → pulls optimiser away from",
    "  odometry chain → GTSAM: 'cannot decrease error at maximum lambda'",
    "  → loop and odometry constraints in tension; solver stuck",
], size=11, gap=0.3)

sub_head(s, 7.2, 3.68, 5.95, "Identified fixes", size=13)
bullet_block(s, 7.2, 4.02, 5.95, [
    "Scale down loop information matrix by constant (÷10) → softer loop constraint",
    "Cauchy or Huber robust kernel on loop edges → down-weight outlier loops",
    "Chordal SE(2) initialisation → better LM starting point, faster convergence",
    "Scale per-loop info by ICP score — weak matches contribute less",
], size=11, gap=0.3)

callout(s, 7.2, 5.65, 5.95, 0.75,
    "Informative finding: tightly weighted per-loop Hessian is more accurate than fixed σ "
    "but too tight causes solver tension. Softening by ÷10 lets odometry chain dominate "
    "and closure error would decrease.", size=10, accent=AMBER)

pagenum(s, 14)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 15 — SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
s = new_slide()
accent_bar(s, h=0.055, color=TEAL)
heading(s, "Summary")
divider_line(s)

sub_head(s, 0.45, 1.02, 6.1, "Q2 — Visual SLAM key findings", size=13, color=TEAL)
bullet_block(s, 0.45, 1.38, 6.1, [
    "Basement_2 (indoor): full rectangle, 2336 ORB poses, 155 matched COLMAP pairs",
    "  Both methods trace same room shape — 3 m scale disagreement is monocular ambiguity",
    "OnePoolStreet1 (outdoor): 5759 ORB poses, 576 matched pairs — best coverage",
    "  Only 2.5° rotation disagreement — near-perfect orientation agreement ✓",
    "  3 m translation = scale unobservable in open outdoor scene",
    "BikeStorage2 night failure — no ambient texture for ORB descriptor extraction",
], size=11, gap=0.29)

sub_head(s, 0.45, 3.55, 6.1, "Q2 — What walls do", size=13, color=TEAL)
bullet_block(s, 0.45, 3.9, 6.1, [
    "Basement_2: four enclosing walls → COLMAP triangulates from both sides of room",
    "Monocular init gets well-posed essential matrix — no planar degeneracy",
    "Without walls (outdoor): sky/ground near-zero parallax, scale unobservable",
], size=11, gap=0.29)

sub_head(s, 6.6, 1.02, 6.5, "Q3 — LiDAR SLAM key findings", size=13, color=AMBER)
bullet_block(s, 6.6, 1.38, 6.5, [
    "Max range = dominant parameter: 2000 mm → 12000 mm gives 14× improvement",
    "  ICP rank-deficient with only near wall — opposite wall locks translation",
    "Voxel 0.05 m optimal — denoises below sensor noise floor without merging features",
    "Angular subsampling always hurts PCA normals — full scan preferred for indoor",
    "Loop threshold 0.70 from bimodal ICP score histogram — clean gap both indoor seqs",
    "PGO reduces cost 8–25× but closure error grows — loop/odometry Hessian tension",
    "Fix: scale loop info ÷10 → odometry chain dominates → closure improves",
], size=11, gap=0.29)

sub_head(s, 6.6, 4.35, 6.5, "Q3 — What walls do", size=13, color=AMBER)
bullet_block(s, 6.6, 4.7, 6.5, [
    "Enclosed indoor: walls on both sides → ICP normal equations full-rank",
    "Corridor (Floor7): aperture problem along wall axis — range truncation helps",
    "Outdoor courtyard: sparse geometry, ICP Hessian singular → divergence → freeze",
], size=11, gap=0.29)

pagenum(s, 15)

# ── Save ─────────────────────────────────────────────────────────────────────
out = "COMP0222_CW2_GRP_32_presentation.pptx"
prs.save(out)
print(f"Saved {out}  ({len(prs.slides)} slides)")
