# Current Audit Grade

This note records my best honest estimate of what this coursework should score
**right now**, based on the current repository state after the latest fixes and
reruns.

It is written as an audit memo, not as a guarantee of the official mark.
The real mark will still depend on:

- the oral presentation,
- how clearly the strongest evidence is presented,
- how well the group answers follow-up questions,
- whether the submitted artifacts exactly match the defended outputs.

## Bottom line

My current estimate is:

- **Likely range:** `86–89 / 100`
- **Conservative single-number estimate:** **87 / 100**

I do **not** think the repo is sitting in obvious danger territory anymore.
The major technical audit concern from before, namely:

- "the code claims COLMAP-calibrated Q2 reruns exist, but the rerun path is
  broken or not really used"

has now been materially fixed.

## Why I think it is around 87

The coursework now looks like:

- technically serious,
- substantially more reproducible,
- much more internally consistent than before,
- strong enough to defend well in an oral,
- but still not completely airtight or polished enough for a confident
  `90+` prediction.

The biggest reason it moved up is that the Q2 pipeline now has real evidence
behind the calibration story:

- `Basement_1_trajectory_colmap_intrinsics.txt` exists with `852` poses
- `Outdoor_1_trajectory_colmap_intrinsics.txt` exists with `584` poses

That means the brief-level Q2 requirement of one indoor + one outdoor sequence
with at least 500 frames after initialization is satisfied using the
calibrated-first path that was previously under audit.

## Mark breakdown estimate by question

These are not official marks. They are my best estimate of how an examiner
might view the current state.

### Q1 — Visual SLAM with datasets

Estimated mark:

- **27–28 / 30**

Why it is strong:

- You clearly ran the required KITTI07 and TUM long-sequence experiments.
- The baseline, feature-count, outlier-rejection, and loop-closure ablations
  are all present.
- The analysis is better than average and uses more evidence than most
  coursework submissions do.
- Q1b in particular is now much stronger than a generic "lower features hurts"
  writeup. It shows the actual threshold behavior and log-backed failure story.

Why I am not just calling it 30/30:

- It is still a coursework-style experimental analysis rather than something
  truly exhaustive or publication-grade.
- The exact oral explanation quality will matter a lot here.

### Q2 — Visual SLAM with your own sequences

Estimated mark:

- **27–29 / 33**

Why it is now much better:

- There is a genuine COLMAP scratch calibration result.
- The code now regenerates per-sequence ORB-SLAM2 YAMLs from COLMAP outputs.
- The calibrated rerun path has been debugged and made operational.
- The Q2 figures are regenerated after the calibrated trajectory files exist.
- The required indoor/outdoor threshold condition is now met on the calibrated
  path for:
  - `Basement_1`
  - `Outdoor_1`

Strongest evidence:

- `Basement_1` is your strongest indoor success case.
- It is the best anchor for defending:
  - data collection quality,
  - calibration quality,
  - ORB vs COLMAP comparison quality.

What still keeps Q2 below "easy 30+" territory:

- `Outdoor_1` meets the threshold, but its inter-method disagreement is still
  fairly weak:
  - translation disagreement is not tiny,
  - rotational disagreement is very large,
  - this makes it a valid but not beautiful outdoor case.
- Some of the extra sequences remain messy or clearly weak:
  - `Floor7_Hallway` COLMAP failure
  - `BikeStorage2` failure case
  - several high-disagreement sequences

That does **not** break the brief. But it does keep the quality ceiling below
"obvious distinction" unless the oral framing is very sharp.

### Q3 — LiDAR SLAM with your own sequences

Estimated mark:

- **32–33 / 37**

Why it is strong:

- The repo now clearly focuses on the three brief-required sequences:
  - two indoor,
  - one outdoor,
  - including the required large-area indoor case.
- Q3b is thorough and technically better than many student submissions.
- Q3c and Q3d are real implementations, not placeholders.
- The main Q3 path is now aligned with stronger audited settings instead of the
  weaker historical 4 m default.

Why I am not comfortably calling it 35+:

- Some parts are still heuristic rather than robustly engineered.
- Q3a loop verification is still an evidence-based heuristic, not a perfect
  external truth source.
- Pose-graph optimization is solid coursework quality, but not especially
  advanced or uncertainty-rich compared with a truly top-end implementation.

## What was fixed that materially improved the grade

These are the changes that genuinely moved the score up, not just cosmetic
documentation edits.

### 1. Q2 rerun path was made real

Before:

- calibrated YAML generation existed,
- but the end-to-end rerun path was under audit suspicion.

Now:

- calibrated ORB-SLAM2 reruns are first-class outputs,
- the analysis scripts know how to consume them,
- and the key calibrated trajectory files exist on disk.

That is a real mark improvement.

### 2. YAML parsing bug was found and fixed

The fallback RealSense YAML files had comments before `%YAML:1.0`, which caused
OpenCV parsing failure in fresh ORB reruns.

That means this was not just "the environment is weird." There was a genuine
repo issue affecting rerun reliability, and it has been fixed.

### 3. Headless ORB-SLAM2 execution was hardened

Using `xvfb-run -a` where available makes the rerun path much more defensible
for audit / reproduction.

### 4. Q3 main config is more honest now

The repo no longer discovers stronger settings in Q3b and then silently
generates the main Q3 outputs from a weaker baseline.

That improves both technical integrity and oral defensibility.

## Why I do not think it is safely 90+ yet

A `90+` prediction would require me to feel that the repo is not only correct
and compliant, but also *hard to criticize in the oral*.

I do not think that is fully true yet.

The main remaining reasons:

### 1. Q2 outdoor quality is still defensible, not exceptional

`Outdoor_1` now works well enough to satisfy the brief, which matters a lot.
But it is not a clean "look how well monocular calibration transferred outdoors"
result. It is more of a:

- "the pipeline is real,
- the sequence meets the requirement,
- and the disagreement itself is scientifically interesting"

case.

That is fine, but it is not the same as a beautiful high-mark outdoor success.

### 2. Some extra Q2 sequences may invite awkward questions

Even though they are not required for compliance, they still exist in the repo
and figures. If the examiners focus on the messy ones, your answers need to be
tight and proactive rather than defensive.

### 3. A lot still depends on oral framing

This coursework is especially oral-heavy. A repo that might deserve `87` with a
good defence could drift lower if the presentation over-claims, gets vague, or
centers the wrong sequences.

## Compared to the taught labs

After checking the actual module materials in
`COMP0222_25-26/Labs/`, I think it is important to say this clearly:

- your coursework does **not** look weak relative to what was taught,
- in several places it is **stronger and more complete than the lab baseline**,
- so the remaining lost marks are mostly about result quality and presentation,
  not failure to implement the taught content.

### What the labs actually cover

The most relevant labs are:

- **Lab 06: COLMAP**
  - install and run COLMAP,
  - reconstruct standard scenes,
  - collect your own images,
  - inspect logs and sparse models.
- **Lab 07: ORB-SLAM2 + evo**
  - run ORB-SLAM2 on TUM, EuRoC, and KITTI,
  - vary `ORBextractor.nFeatures`,
  - use `evo_traj`, `evo_ape`, and `evo_rpe`,
  - calibrate a custom camera sequence with COLMAP,
  - run `mono_tum` on the custom sequence.
- **Lab 08: LiDAR Odometry and Point Cloud Mapping**
  - convert polar scans to Cartesian,
  - run scan-to-map ICP,
  - understand drift, degeneracy, corridor failure, and dynamic limits,
  - optionally do parameter sensitivity work.
- **Lab 09: Occupancy Grid Mapping**
  - build occupancy grids,
  - ray-cast free space,
  - perform scan-to-map matching,
  - study parameter tuning and dynamic object effects.

### Where your repo is above the lab baseline

#### Q1

Compared with Lab 07, Q1 goes beyond the basic exercise level because it:

- runs multiple required benchmark datasets,
- performs several ablations rather than one-off demos,
- records evidence for failure cases,
- and wraps the results into a much more explicit comparative analysis.

That is stronger than "run ORB-SLAM2 and plot evo once."

#### Q2

Compared with Labs 06 and 07, Q2 is also above the basic taught level because it:

- calibrates and reruns on your own collected data,
- generates per-sequence intrinsics artifacts,
- compares ORB-SLAM2 and COLMAP across multiple custom sequences,
- and includes additional reconstruction / visualization outputs.

The main weakness in Q2 is **not** "this was never taught and you failed to do
it." The weakness is that some results, especially the outdoor case, are not as
clean as a top-end submission would ideally show.

#### Q3

Compared with Labs 08 and 09, Q3 is clearly beyond the minimum teaching
baseline because it includes:

- a more complete offline pipeline,
- parameter sweeps,
- multiple datasets,
- loop closure detection,
- pose-graph optimization,
- and occupancy mapping integrated into the final workflow.

The labs explicitly say the teaching scripts are "pure odometry" and do **not**
have loop closure. Your coursework does.

### What this means for the mark

So if we benchmark fairly against what was actually taught:

- I do **not** think you are losing marks because the repo is below the lab
  standard,
- I think you are mainly losing marks because some results are less clean than
  the best possible outcomes,
- and because top-band marks still require a very crisp oral defence.

This makes me a bit more confident that the current repo belongs in the
upper-80s conversation rather than the low-80s conversation.

## Best oral strategy from the current state

If I were advising purely for marks, I would recommend this framing:

### Q2

Center these two:

- `Basement_1` as the strongest indoor calibration success
- `Outdoor_1` as the required outdoor case that now meets the threshold

Say explicitly:

- the calibrated reruns now exist,
- `Basement_1` is the cleanest evidence that the calibration pipeline works,
- `Outdoor_1` remains harder and shows the limits of monocular agreement in
  outdoor scenes.

Treat weaker cases as:

- supporting analysis,
- not the headline result.

### Q3

Emphasize:

- you used the three brief-required sequences,
- one indoor is explicitly the large-area case,
- you audited the main config so the final outputs reflect the stronger
  settings rather than the weaker historical defaults.

## If everything goes well in the oral

If the group:

- presents clearly,
- does not over-claim,
- uses `Basement_1` and `Outdoor_1` intelligently,
- answers questions honestly about failure cases,
- and keeps the story focused on the required sequences,

then I think the outcome could plausibly land near the top of this range:

- **88–89**

## If the oral goes badly

If the group:

- centers weak sequences,
- sounds confused about what was recalibrated vs not,
- overstates what the COLMAP-vs-ORB metric means,
- or gets defensive on the failure cases,

then I could imagine the result slipping closer to:

- **83–85**

## Final statement

My best current judgment is:

- **This is now a strong submission.**
- **It looks like upper-80s work, not borderline work.**
- **I would currently call it 87/100, with realistic upside to 89 if defended well.**

That is what I think it should get **right now**, based on the repository and
the fixes completed during this audit.
