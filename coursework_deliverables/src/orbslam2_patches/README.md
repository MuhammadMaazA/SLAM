# ORB-SLAM2 source modifications for Q1c / Q1d

The coursework brief (COMP0222 CW2, Q1c and Q1d) asks how the ORB-SLAM2
implementation handles **outlier rejection** and **loop closure**, and to
evaluate the effect of disabling each. The trajectories in
`data/part1_analysis/kitti07-nooutlier.txt`, `tum-nooutlier.txt`,
`kitti07-noloop.txt` and `tum-noloop.txt` were produced by running modified
ORB-SLAM2 binaries on KITTI sequence 07 and TUM `freiburg3_long_office_household`.

The two patches in this directory document, exactly, which source lines in
the upstream `raulmur/ORB_SLAM2` tree were changed to produce those binaries.
Apply them with

```bash
cd ORB_SLAM2   # your local raulmur/ORB_SLAM2 clone
# Use one patch per ORB-SLAM2 clone (or revert between builds): each targets a different file.
git apply /path/to/SLAM/coursework_deliverables/src/orbslam2_patches/q1c_disable_outlier_rejection.patch
# …or…
git apply /path/to/SLAM/coursework_deliverables/src/orbslam2_patches/q1d_disable_loop_closure.patch
# or from inside coursework_deliverables/:  git apply src/orbslam2_patches/<patchname>.patch
./build.sh
# then rebuild mono_kitti / mono_tum and re-run with the same YAML
```

Re-applying the patches reproduces our `*-nooutlier.txt` / `*-noloop.txt`
trajectories end-to-end.

## Files

| File                                         | Effect |
|----------------------------------------------|--------|
| `q1c_disable_outlier_rejection.patch`        | Skips the per-frame chi-squared outlier test inside `Optimizer::PoseOptimization` so every matched feature (including BA outliers) contributes to pose estimation. This is the "no outlier rejection" configuration. |
| `q1d_disable_loop_closure.patch`             | Short-circuits `LoopClosing::DetectLoop` to always return false, so place recognition never fires, no loop is verified, and no Sim(3) correction / global BA is performed. Tracking + local BA still run normally. This is the "no loop closure" configuration. |

## Why we changed *these* lines specifically

### Q1c (outlier rejection)

In `Optimizer::PoseOptimization`, ORB-SLAM2 runs four rounds of Levenberg-
Marquardt pose-only BA. After each round it flags any reprojection residual
above a chi-squared 5.991 (monocular) / 7.815 (stereo) threshold as an
outlier and excludes it from the next round. This is the dominant outlier
rejection mechanism at tracking time. The patch sets those outlier flags
to `false` unconditionally, so every matched feature stays in the LM solve.
We deliberately do NOT remove the chi-squared computation itself, so logs
still record how many features *would have* been rejected — that number is
reported alongside the ATE in the writeup.

### Q1d (loop closure)

`LoopClosing::Run` invokes `DetectLoop()` which checks the DBoW2 database
for a candidate keyframe that shares a common bag-of-words with the current
keyframe, then verifies geometric consistency across three consecutive
keyframes, then attempts a Sim(3) solve. Any of those gates short-circuits
the loop. The patch simply returns `false` at the top of `DetectLoop`,
which is a much cleaner ablation than disabling individual stages — it
leaves every other loop-closing data structure (keyframe database, covisibility
graph) intact so the rest of the system is unaffected.

## Verification

Rebuild and run

```bash
./Examples/Monocular/mono_kitti Vocabulary/ORBvoc.txt \
    coursework_deliverables/data/part1_analysis/KITTI04-12_custom.yaml \
    <path to KITTI07>
```

and compare the saved trajectory against
`data/part1_analysis/kitti07-nooutlier.txt` / `kitti07-noloop.txt`.
The numbers should match to within single-digit millimetres (random BA
seeds cause the last-digit variation).
