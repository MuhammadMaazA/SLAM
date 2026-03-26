# COMP0222 Lab 02: Kalman Filter-Based SLAM
## Summary and Key Takeaways

**Author:** Maaz  
**Date:** January 21, 2026  
**Course:** COMP0222 - Robotics and AI Systems

---

## Overview

This lab implements a **Kalman Filter-based SLAM (Simultaneous Localization and Mapping)** system for a point robot called "dotbot". The system uses a linear motion model and Cartesian landmark observations to build a map while localizing the robot.

### Core Concepts
- **SLAM**: Simultaneously estimate robot position and landmark positions
- **State Vector Growth**: Starts as [robot_x, robot_y], grows to [robot_x, robot_y, landmark1_x, landmark1_y, landmark2_x, landmark2_y, ...]
- **Linear System**: Uses standard Kalman Filter (next lab: Extended Kalman Filter for nonlinear systems)

---

## Activity 0: Installation
**Goal:** Verify software installation

**What You See:**
- Blue circle: Robot ground truth position
- Green lines: SLAM sensor measurements
- Red crosses: True landmark positions

**Key Point:** This is just a test - no SLAM algorithm running yet!

---

## Activity 1: Dead Reckoning (Prediction Only)

### What It Does
Robot moves using **dead reckoning** (odometry only) with NO landmark observations.

### Commands
```matlab
>> l2.activity1
```

### Observations

**Behavior of Mean (Estimated Position):**
- ✅ Follows programmed trajectory correctly
- Blue circle moves in straight line
- No systematic drift in mean

**Behavior of Covariance (Uncertainty):**
- ❌ **Grows unboundedly and linearly with time!**
- Red circle keeps expanding
- After 120 seconds: ~1 meter uncertainty
- After 1200 seconds: ~10 meter uncertainty

### Why This Happens

**Dead Reckoning Process:**
```
Position[k+1] = Position[k] + Velocity × ΔT + Process_Noise
```

**Process Noise:**
- Random errors in motion (wheel slip, floor texture, motor variations)
- Each timestep adds noise: `P = F×P×F' + Q`
- Accumulates indefinitely without corrections
- Like walking blindfolded counting steps - eventually you're completely lost!

### Key Takeaways

✅ **Dead reckoning alone causes unbounded uncertainty growth**  
✅ **Process noise accumulates over time**  
✅ **You NEED observations to bound uncertainty**  
✅ **This motivates why SLAM is necessary!**

**Formula:** Uncertainty ≈ √(process_noise × time)

---

## Activity 2: Augmentation Step

### What It Does
Implements **map building** by adding newly observed landmarks to the state vector.

### Commands
```matlab
>> l2.activity2
```

### Implementation

**Location:** `dotbot.SLAMSystem.handleSLAMObservationEvent` (lines 250-275)

**What to Implement:**
```matlab
% Build J matrix (copies state, adds robot position for new landmark)
J = zeros(n + NL, n);
J(1:n, 1:n) = eye(n);      % Copy existing state
J(n+1, 1) = 1;             % new_landmark_x from robot_x
J(n+2, 2) = 1;             % new_landmark_y from robot_y

% Build K matrix (adds measurement)
K = zeros(n + NL, NL);
K(n+1, 1) = 1;             % new_landmark_x += measurement_x
K(n+2, 2) = 1;             % new_landmark_y += measurement_y

% Augment
obj.x = J * obj.x + K * event.data(:, idx(o));
obj.P = J * obj.P * J' + K * event.covariance() * K';
```

### The Math

**Landmark Initialization (Equation 10):**
```
landmark_position = robot_position + measurement_offset + noise
m^i = [x_k + z^i_x,k; y_k + z^i_y,k]
```

**State Vector Growth:**
```
Before: [robot_x, robot_y]                                    (2D)
After:  [robot_x, robot_y, landmark1_x, landmark1_y]          (4D)
Later:  [robot_x, robot_y, lm1_x, lm1_y, lm2_x, lm2_y, ...]   (2+2N dimensional)
```

**Covariance Matrix Growth:**
```
From 2×2 → 4×4 → 6×6 → ... (grows with each new landmark)
Includes cross-correlations between robot and all landmarks!
```

### Observations

**What You See:**
- Green crosses appear (estimated landmark positions)
- Many overlapping red circles (one per landmark)
- **Circles are LARGE** (~5-10 meter radius)

**Why Circles Are Large:**
- Only augmentation, NO updates yet
- Initial uncertainty = robot uncertainty + measurement noise
- No corrections applied
- This is EXPECTED for Activity 2!

### Key Takeaways

✅ **Augmentation adds new landmarks to the map**  
✅ **State vector grows dynamically**  
✅ **Initial landmark uncertainty is high (no corrections yet)**  
✅ **J matrix copies state and links landmark to robot position**  
✅ **K matrix incorporates the measurement**  
✅ **Multiple red circles are normal (one per landmark)**

---

## Activity 3: Update Step

### What It Does
Implements **Kalman Filter updates** to correct estimates when re-observing known landmarks.

### Commands
```matlab
>> l2.activity3
```

### Implementation

**Location:** `dotbot.SLAMSystem.handleSLAMObservationEvent` (lines 224-248)

**What to Implement:**
```matlab
% Build observation matrix for full state
n = length(obj.x);
HS = zeros(NL, n);
HS(:, 1:NP) = Hx;              % Robot Jacobian
HS(:, landmarkIdx) = Hm;       % Landmark Jacobian

% Kalman Filter Update
R = event.covariance();
nu = event.data(:, idx(o)) - zPred;     % Innovation
S = HS * obj.P * HS' + R;               % Innovation covariance
K = obj.P * HS' / S;                    % Kalman gain
obj.x = obj.x + K * nu;                 % Update state
obj.P = obj.P - K * S * K';             % Update covariance
```

### The Math

**Observation Model (Equation 7):**
```
z^i = landmark_i - robot + noise
z^i = [-I, 0, ..., I, ...] × state + noise
```

**HS Matrix Structure:**
```
For state = [robot_x, robot_y, lm1_x, lm1_y, lm2_x, lm2_y]
Observing lm1:
HS = [Hx(1,1), Hx(1,2), Hm(1,1), Hm(1,2),    0,       0    ]
     [Hx(2,1), Hx(2,2), Hm(2,1), Hm(2,2),    0,       0    ]
```

**Standard Kalman Update:**
```
Innovation: ν = z_measured - z_predicted
Kalman Gain: K = P×H'×(H×P×H' + R)^(-1)
State Update: x = x + K×ν
Covariance Update: P = P - K×S×K'
```

### Observations

**What You See:**
- **Red circles SHRINK dramatically!** (from ~10m to ~0.5m radius)
- Green crosses very close to red crosses (accurate!)
- Small uncertainty for ALL landmarks (even unobserved ones!)

**Why This Is Amazing:**
```
Observe landmark 1 →
  ✓ Landmark 1 estimate improves (direct observation)
  ✓ Robot estimate improves (I know where landmark is!)
  ✓ Landmark 2 improves (robot improved, so landmark 2 estimate improves!)
  ✓ ALL estimates improve together via cross-correlations!
```

### Key Takeaways

✅ **Updates dramatically reduce uncertainty**  
✅ **Observing one landmark improves ALL estimates** (via cross-correlations)  
✅ **Green crosses match red crosses (accurate estimates)**  
✅ **Small red circles everywhere = working SLAM!**  
✅ **This is the power of SLAM - information propagates through entire system**  
✅ **Cross-correlations enable joint optimization**

**The Magic:** Update one thing → Everything improves!

---

## Activity 4: Breaking Cross-Correlations

### What It Does
**Deliberately destroys** cross-correlations to show why they're essential.

### Commands
```matlab
>> l2.activity4
```

### The "Muck Up" Code

```matlab
if (obj.muckUp == true)
    for k = 1 : 2 : length(obj.P)
        obj.P(k:k+1, 1:k-1) = 0;      % Zero left cross-correlations
        obj.P(k:k+1, k+2:end) = 0;    % Zero right cross-correlations
    end
end
```

### What This Does

**Normal Covariance Matrix:**
```
P = [σ²_robot    σ_robot,lm1   σ_robot,lm2  ...]
    [σ_robot,lm1 σ²_lm1        σ_lm1,lm2    ...]
    [σ_robot,lm2 σ_lm1,lm2     σ²_lm2       ...]
    
All states are correlated - information flows everywhere!
```

**After "Mucking Up":**
```
P = [σ²_robot    0              0           ...]  ← Robot isolated
    [0           σ²_lm1         0           ...]  ← Landmark1 isolated
    [0           0              σ²_lm2      ...]  ← Landmark2 isolated
    
Block diagonal - NO information sharing!
```

### Observations

**What Happens:**
- ❌ **Landmarks drift apart** from truth
- ❌ **No consistency** between estimates
- ❌ **Large errors accumulate**
- ❌ **Only directly observed landmark improves**
- ❌ **Robot estimate doesn't improve from landmark observations**
- ❌ **System "falls apart" - SLAM fails!**

### Why It Fails

**With Cross-Correlations (Normal):**
```
Kalman Gain K = P × H' / S
  → P has correlations everywhere
  → K has non-zero entries for ALL states
  → ALL states get updated from one observation!
```

**Without Cross-Correlations (Broken):**
```
Kalman Gain K = P_mucked × H' / S
  → P_mucked only has diagonal blocks
  → K only non-zero for observed landmark
  → ONLY observed landmark gets updated!
```

### Analogy

**With Cross-Correlations:**
> Friends in a dark room holding a rope. One person moves → everyone feels it. Information propagates.

**Without Cross-Correlations:**
> Friends in a dark room, no rope. One person moves → nobody else knows. Everyone drifts independently.

### Key Takeaways

✅ **Cross-correlations are THE REASON SLAM works!**  
✅ **They enable information propagation through the system**  
✅ **Without them, SLAM reduces to independent tracking**  
✅ **Maintaining cross-correlations is computationally expensive but essential**  
✅ **This demonstrates why covariance matrix is O(n²) - you can't just track variances!**  
✅ **Some SLAM algorithms approximate correlations, but never eliminate them completely**

**Critical Insight:** It's not enough to track uncertainty of individual elements - you must track how they relate!

---

## Activity 5: Advanced Scenarios

### Commands
```matlab
>> l2.activity5
% Select: a, b, c, or d
```

---

### Scenario A: Two Isolated Islands

**Setup:**
- Landmarks in two separate clusters
- Robot visits island 1, then island 2
- No overlap between islands

**Observations:**
- ✅ Within each island: tight, accurate estimates
- ❌ Between islands: large relative uncertainty
- Islands drift relative to each other
- Each island is internally consistent

**Why:**
- Strong cross-correlations within each island
- Weak/no cross-correlations between islands
- No shared observations to constrain relative positions

**Key Takeaway:**
> **SLAM needs overlapping observations for global consistency**

---

### Scenario B: Loop Closure

**Setup:**
- Landmarks arranged in a loop
- Robot travels around and returns to start
- Re-observes initial landmarks

**Observations:**

**Before Loop Closure:**
- Uncertainty grows as robot moves around loop
- Estimates drift
- Large red circles by end of loop

**During Loop Closure:**
- Robot re-observes start landmarks
- **SUDDEN DRAMATIC CORRECTION!**
- Entire map "snaps" into consistency
- ALL uncertainties plummet
- Even unobserved landmarks improve!

**Why It Happens:**
```
1. Early: Robot at A, sees landmark 1
2. Middle: Robot at Z, uncertainty has grown
3. Loop closes: Robot returns near A, sees landmark 1 again
4. Innovation: "Wait! Landmark 1 should be at X based on my current 
   position, but I measured Y!"
5. Correction propagates through ALL cross-correlations
6. ENTIRE trajectory and map adjust to close the loop
```

**What Constitutes Loop Closure:**
1. ✅ Robot returns to previously visited location
2. ✅ Re-observes previously mapped landmarks
3. ✅ Detects inconsistency (large innovation)
4. ✅ Correction propagates globally via cross-correlations
5. ✅ Entire map adjusts for consistency

**Key Takeaways:**
> **Loop closure enables global consistency correction**  
> **Cross-correlations propagate correction through entire map**  
> **Revisiting locations is crucial for long-term accuracy**

---

### Scenario C: Loop + GPS

**Setup:**
- Same loop as B
- GPS enabled (absolute position measurements)

**Observations:**
- Uncertainty never grows large (GPS bounds it)
- Loop closure effect less dramatic
- Map stays consistent throughout
- Gradual corrections instead of sudden snap

**Why GPS Helps:**
- Provides absolute position: "You are at (10.2, 5.7)"
- Acts as anchor preventing drift
- Bounds uncertainty even without loop closure
- Provides global reference frame

**Comparison:**

| Without GPS (B) | With GPS (C) |
|----------------|--------------|
| Uncertainty grows | Uncertainty bounded |
| Dramatic loop closure | Gradual corrections |
| Relative positioning only | Absolute positioning |
| Depends on loop closure | Less dependent |

**Key Takeaways:**
> **GPS provides global reference frame**  
> **Bounds uncertainty without needing loop closures**  
> **Complements SLAM's local consistency**  
> **Best of both worlds: global + local!**

---

### Scenario D: Loop + GPS + Bearing

**Setup:**
- Same loop as B and C
- GPS + Bearing sensor both enabled

**Observations:**
- **Best performance of all scenarios!**
- Smallest uncertainty circles
- Most accurate estimates
- Very stable throughout trajectory
- Tightest green/red cross alignment

**What Bearing Sensor Adds:**
- Measures angle to fixed beacons: "Beacon at 45°"
- Provides orientation information
- Complements GPS (position) with direction
- Additional constraints on state

**Sensor Fusion Benefits:**

| Sensor | Provides | Limitation |
|--------|----------|------------|
| SLAM landmarks | Relative positions, local consistency | Drift, needs loop closure |
| GPS | Absolute position | No orientation, noise |
| Bearing | Absolute orientation | Needs known beacons |
| **All Together** | Position + orientation + consistency | **Best!** |

**Key Takeaways:**
> **Multiple sensors provide complementary information**  
> **Sensor fusion via Kalman Filter is automatic and optimal**  
> **Redundancy improves robustness**  
> **Real robots use multi-sensor SLAM for best results**  
> **Cross-correlations naturally fuse all sensor data**

---

## Summary of Key Concepts

### 1. Process Noise
- Random errors in motion (wheel slip, motor variations)
- Accumulates over time: `P = F×P×F' + Q`
- Causes unbounded drift without observations
- Standard deviation grows as √time

### 2. Dead Reckoning
- Estimating position by integrating motion
- `Position[k+1] = Position[k] + Velocity × ΔT`
- Process noise causes unbounded uncertainty growth
- Like navigating blindfolded - eventually completely lost

### 3. Augmentation
- Adding newly observed landmarks to state vector
- State grows: 2D → 4D → 6D → 2+2N dimensions
- Initial uncertainty is high (no corrections yet)
- Uses J and K matrices to incorporate measurements

### 4. Update/Correction
- Kalman Filter update when re-observing landmarks
- Reduces uncertainty for ALL states (not just observed!)
- Information propagates via cross-correlations
- This is the "magic" of SLAM

### 5. Cross-Correlations
- **THE MOST IMPORTANT CONCEPT!**
- Represent how estimates are related
- Enable information propagation through entire system
- Stored in off-diagonal elements of covariance matrix P
- Essential for SLAM - without them, system fails completely

### 6. Loop Closure
- Re-observing landmarks after completing a loop
- Enables global consistency correction
- Dramatic reduction in uncertainty
- Critical for long-term SLAM accuracy

### 7. Sensor Fusion
- Combining multiple sensor types (SLAM + GPS + Bearing)
- Each sensor complements others' weaknesses
- Kalman Filter automatically and optimally fuses data
- Cross-correlations handle multi-sensor integration

---

## Mathematical Summary

### State Space
```
x_k = [robot_x, robot_y, lm1_x, lm1_y, ..., lmN_x, lmN_y]^T
Dimension: 2 + 2N (grows with number of landmarks N)
```

### Motion Model
```
x[k+1] = F×x[k] + B×u[k] + w[k]
where w[k] ~ N(0, Q) is process noise
```

### Observation Model
```
z^i = m^i - x_robot + v^i
where v^i ~ N(0, R) is measurement noise
Means: measurement = landmark - robot position
```

### Prediction Step
```
x_predict = F×x + B×u
P_predict = F×P×F' + Q
```

### Augmentation Step (New Landmark)
```
x_new = J×x_old + K×z
P_new = J×P_old×J' + K×R×K'
```

### Update Step (Known Landmark)
```
ν = z - h(x)                    (innovation)
S = H×P×H' + R                  (innovation covariance)
K = P×H'×S^(-1)                 (Kalman gain)
x = x + K×ν                     (state update)
P = P - K×S×K'                  (covariance update)
```

---

## Practical Implementation Tips

### Debug Strategy
1. Set `perturbWithNoise: false` in config for deterministic testing
2. Add print statements to verify matrix dimensions
3. Check that landmarks are being added (`length(obj.x)` increases)
4. Verify uncertainty decreases after updates (`trace(obj.P)` decreases)
5. Visual check: green crosses should match red crosses

### Common Issues

**Activity 2 (Augmentation):**
- Forgot to uncomment bookkeeping lines
- Wrong dimensions for J or K matrices
- Not adding measurement to robot position

**Activity 3 (Update):**
- HS matrix has wrong dimensions (should be 2×n)
- landmarkIdx pointing to wrong state elements
- Forgot to get measurement from `event.data(:, idx(o))`
- Covariance becomes negative definite (use Joseph form)

**Activity 4:**
- Should see system fail - if it works, something's wrong!
- Cross-correlations should be zeroed in P matrix

---

## Real-World Applications

### Where This Matters
- **Autonomous vehicles**: Building maps while driving
- **Drones**: Indoor navigation without GPS
- **Warehouse robots**: Mapping and navigating facilities
- **Mars rovers**: Exploring unknown terrain
- **Vacuum robots**: Learning your home layout

### Extensions Beyond This Lab
- **Extended Kalman Filter (EKF)**: Nonlinear motion/observation models (next lab!)
- **FastSLAM**: Particle filter-based SLAM
- **GraphSLAM**: Pose graph optimization
- **Visual SLAM**: Using camera images as landmarks
- **LiDAR SLAM**: Using laser scans for 3D mapping

---

## Key Equations Reference

### Covariance Propagation
```
P[k+1|k] = F×P[k|k]×F' + Q
```

### Kalman Gain
```
K = P×H'×(H×P×H' + R)^(-1)
```

### Posterior Covariance
```
P = (I - K×H)×P  [Joseph form - more stable]
P = P - K×S×K'   [Alternative form]
```

### Innovation
```
ν = z_measured - h(x_predicted)
```

### Mahalanobis Distance (for outlier detection)
```
d² = ν'×S^(-1)×ν
```

---

## Revision Checklist

- [ ] Understand why dead reckoning causes unbounded uncertainty
- [ ] Can explain process noise and how it accumulates
- [ ] Know how to implement augmentation (J and K matrices)
- [ ] Know how to implement update step (HS matrix and Kalman equations)
- [ ] Understand why cross-correlations are essential
- [ ] Can explain what happens when cross-correlations are removed
- [ ] Know what loop closure is and why it's important
- [ ] Understand how GPS and bearing sensors complement SLAM
- [ ] Can write out the Kalman Filter equations from memory
- [ ] Understand state vector structure and how it grows

---

## Quick Reference

### File Locations
```
Lab_02/+l2/+dotbot/SLAMSystem.m  - Main implementation
  - handleSLAMObservationEvent()  - Activity 2 & 3 code here
  
Lab_02/+l2/activity*.m           - Test scripts
config/activity*.json            - Configuration files
```

### Running Activities
```matlab
>> l2.activity0  % Test installation
>> l2.activity1  % Dead reckoning demo
>> l2.activity2  % Augmentation implementation
>> l2.activity3  % Update implementation  
>> l2.activity4  % Break cross-correlations
>> l2.activity5  % Advanced scenarios (a/b/c/d)
```

### Debug Commands
```matlab
% Check state size
length(obj.x)  % Should grow as landmarks added

% Check uncertainty
trace(obj.P)   % Should decrease after updates

% Check landmarks
numel(obj.landmarkIDStateVectorMap.keys())
```

---

## Final Thoughts

**The Big Picture:**
- SLAM solves the chicken-and-egg problem: need map to localize, need localization to build map
- Kalman Filter elegantly handles both simultaneously
- Cross-correlations are what make it all work
- Multiple sensors make the system robust
- Loop closure enables long-term consistency

**Why This Matters:**
This linear SLAM is a foundation. Real systems use Extended/Unscented Kalman Filters, particle filters, graph optimization, and deep learning. But the core concepts - prediction, update, cross-correlations, loop closure - remain the same.

**Next Steps:**
Lab 03 will implement Extended Kalman Filter (EKF) SLAM with nonlinear motion and observation models - the real-world case!

---

**Good luck with your revision! 🚀**

*Remember: Cross-correlations are the secret sauce that makes SLAM work!*