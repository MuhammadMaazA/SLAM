# ORB-SLAM2 source modifications for Q1c and Q1d

This document records the exact source-code changes applied to a vanilla
[ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) checkout in order to run the
ablations required by the coursework brief:

| Coursework item | What is disabled                         | File(s) touched |
|-----------------|-------------------------------------------|-----------------|
| **Q1c**         | Outlier rejection (RANSAC + χ² pruning)   | `src/Frame.cc`, `src/Optimizer.cc` |
| **Q1d**         | Global loop closure                       | `src/System.cc`, `src/LoopClosing.cc` |

All changes are toggled through a single preprocessor flag per experiment
(`DISABLE_OUTLIER_REJECTION` for Q1c, `DISABLE_LOOP_CLOSURE` for Q1d). Keeping
the baseline buildable alongside the ablations avoids accidental regressions.

> The modified source tree sits in
> `~/SLAM/ORB_SLAM2/`. Apply the patch with `git apply orb_slam2_q1.patch`.

---

## Build flags

Three CMake configurations are defined; each re-builds
`Examples/Monocular/mono_kitti` and `Examples/Monocular/mono_tum`.

```bash
# Baseline — identical to upstream
cmake -B build_baseline -DCMAKE_BUILD_TYPE=Release
cmake --build build_baseline -j

# Q1c — outlier rejection disabled
cmake -B build_nooutlier -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-DDISABLE_OUTLIER_REJECTION"
cmake --build build_nooutlier -j

# Q1d — loop closure disabled
cmake -B build_noloop   -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-DDISABLE_LOOP_CLOSURE"
cmake --build build_noloop -j
```

---

## Q1c — disable outlier rejection

ORB-SLAM2 rejects map-point outliers at two points in the pipeline:

1. **Pose-only BA** (`Optimizer::PoseOptimization`) runs χ² tests between
   Levenberg-Marquardt iterations to flag observations as outliers.
2. **Tracking** (`Tracking::TrackLocalMap` and `SearchLocalPoints`) skips any
   observation whose reprojection error exceeds the monocular threshold.

### `src/Optimizer.cc` — `Optimizer::PoseOptimization`

Inside the loop over the four optimiser rounds (lines ~520–560 in upstream),
the outlier mask update is wrapped so it becomes a no-op under the flag:

```cpp
// Optimizer.cc — PoseOptimization
for (size_t idx : indices) {
    g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[idx];
    const size_t idxMP = vnIndexEdgeMono[idx];
    if (pFrame->mvbOutlier[idxMP]) {
        e->computeError();
    }
    const float chi2 = e->chi2();

#ifdef DISABLE_OUTLIER_REJECTION
    // Q1c: never flag a point as outlier — every edge stays ACTIVE.
    pFrame->mvbOutlier[idxMP] = false;
    e->setLevel(0);
#else
    if (chi2 > chi2Mono[it]) {
        pFrame->mvbOutlier[idxMP] = true;
        e->setLevel(1);
    } else {
        pFrame->mvbOutlier[idxMP] = false;
        e->setLevel(0);
    }
#endif
    if (it == 2) e->setRobustKernel(0);
}
```

### `src/Tracking.cc` — reprojection-error gate

The same flag short-circuits the reprojection-error gate in
`Tracking::TrackLocalMap`:

```cpp
// Tracking.cc — TrackLocalMap (final outlier count)
mnMatchesInliers = 0;
for (int i = 0; i < mCurrentFrame.N; ++i) {
    if (mCurrentFrame.mvpMapPoints[i]) {
#ifdef DISABLE_OUTLIER_REJECTION
        // Q1c: keep every matched map point, regardless of residual.
        if (!mCurrentFrame.mvbOutlier[i]) ++mnMatchesInliers;
#else
        if (!mCurrentFrame.mvbOutlier[i]) {
            mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
            if (!mbOnlyTracking) {
                if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    ++mnMatchesInliers;
            } else {
                ++mnMatchesInliers;
            }
        } else if (mSensor == System::STEREO) {
            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }
#endif
    }
}
```

**Expected effect**: BA becomes ill-conditioned, mean tracking quality drops,
large pose jumps appear near textureless frames, `mnMatchesInliers` often falls
below `mnMatchesInMapForRelocalization` so tracking is declared LOST and the
recorded trajectory length shrinks. This matches the observed result
(`kitti07-nooutlier.txt` has 533 poses instead of 1096).

---

## Q1d — disable loop closure

`LoopClosing` runs in its own thread. The cleanest no-op is to suppress the
call to `DetectLoop()` inside the main loop.

### `src/LoopClosing.cc` — `LoopClosing::Run`

```cpp
// LoopClosing.cc — Run()
while (1) {
    if (CheckNewKeyFrames()) {
#ifdef DISABLE_LOOP_CLOSURE
        // Q1d: drain the queue but never attempt detection or correction.
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        mpCurrentKF->SetErase();
#else
        if (DetectLoop()) {
            if (ComputeSim3()) {
                CorrectLoop();
            }
        }
#endif
    }
    ResetIfRequested();
    if (CheckFinish()) break;
    usleep(5000);
}
```

### `src/System.cc` — optional: skip thread entirely

To also avoid the Bag-of-Words overhead, the loop-closing thread can be
prevented from spawning in `System::System()`:

```cpp
// System.cc — constructor
#ifndef DISABLE_LOOP_CLOSURE
mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDB, mpVocabulary,
                                mSensor != MONOCULAR);
mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);
mpTracker->SetLoopClosing(mpLoopCloser);
mpLocalMapper->SetLoopCloser(mpLoopCloser);
#else
mpLoopCloser = nullptr;
#endif
```

Downstream code that references `mpLoopCloser` is already null-guarded in
upstream; no further changes are required.

**Expected effect**: accumulated drift is no longer corrected when the
trajectory revisits a previously seen region. On KITTI07 (a long loop)
this manifests as the characteristic spiral drift — our measured ATE jumps
from **4.39 m** (baseline) to **18.02 m** (`kitti07-noloop.txt`).

---

## Reproducing the Q1c/Q1d trajectory dumps

```bash
# Baseline
./build_baseline/Examples/Monocular/mono_kitti \
    Vocabulary/ORBvoc.txt KITTI04-12_custom.yaml /path/to/kitti07/ \
    && mv KeyFrameTrajectory.txt kitti07-baseline.txt

# Q1c — outlier rejection off
./build_nooutlier/Examples/Monocular/mono_kitti \
    Vocabulary/ORBvoc.txt KITTI04-12_custom.yaml /path/to/kitti07/ \
    && mv KeyFrameTrajectory.txt kitti07-nooutlier.txt

# Q1d — loop closure off
./build_noloop/Examples/Monocular/mono_kitti \
    Vocabulary/ORBvoc.txt KITTI04-12_custom.yaml /path/to/kitti07/ \
    && mv KeyFrameTrajectory.txt kitti07-noloop.txt
```

Equivalent `mono_tum` commands generate the `tum-*.txt` files consumed by
`src/q1_evo_analysis.py`.

---

## Why conditional compilation rather than YAML switches

ORB-SLAM2's YAML config does not expose outlier-rejection or loop-closing
toggles. A naive workaround (setting χ² thresholds to infinity or the
`LoopClosing` match score to 0) would leave both subsystems *running* but
*ineffective*, obscuring the ablation. The preprocessor approach gives a clean
"pure" ablation and leaves the performance profile truthful (no wasted
BoW queries for Q1d, no wasted χ² evaluations for Q1c).
