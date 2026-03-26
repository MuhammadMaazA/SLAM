# COMP0222 Lab 03: Extended Kalman Filter SLAM (EKF-SLAM)
## Complete Summary and Implementation Guide

**Author:** Maaz  
**Date:** January 21, 2026  
**Course:** COMP0222 - Robotics and AI Systems

---

## Quick Overview

This lab implements **EKF-SLAM** for TriangleBot - a robot with orientation using realistic range & bearing sensors. You'll handle nonlinear models using Jacobians (derivatives) to linearize locally.

**Key Achievement:** Built a working SLAM system achieving <0.5m accuracy with realistic nonlinear sensors! 🎯

---

## Activities Completed

✅ **Activity 0:** Installation test  
✅ **Activity 1:** Dead reckoning - observe orientation error amplification  
✅ **Activity 2:** Forward model - compute range & bearing from states  
✅ **Activity 3:** Inverse model - compute landmark from measurement  
✅ **Activity 4:** GPS + Compass fusion demo  
✅ **Activity 5:** Complete EKF with Jacobians - **FULL SLAM WORKING!**

---

## Core Concepts Summary

### 1. Orientation Amplifies Uncertainty

**Key Insight:** Small orientation error → Large position error (grows with distance!)

```
Orientation error: ±0.1 rad (±5.7°)

After 1 meter:   ±0.1m lateral error
After 10 meters:  ±1.0m lateral error
After 100 meters: ±10m lateral error!  ← Disaster!
```

**Why:** `Position error ≈ distance × θ_error`

**Visual:** Uncertainty ellipse elongates perpendicular to motion direction

### 2. Range & Bearing Sensors (Realistic!)

**Range (r):** Distance to landmark
```matlab
r = sqrt((m_x - x)² + (m_y - y)²)
```

**Bearing (β):** Angle to landmark in robot's frame
```matlab
β = atan2(m_y - y, m_x - x) - θ
```

**Critical:** Subtract θ to convert from world frame to robot frame!

### 3. Extended Kalman Filter (EKF)

**Problem:** Standard KF only works for linear models

**Solution:** EKF linearizes nonlinear models using **Jacobians**

**Key Idea:**
```
Nonlinear function f(x) ≈ f(x₀) + J×(x - x₀)
                                  ↑
                              Jacobian J = ∂f/∂x
```

**Jacobian = Matrix of partial derivatives**

### 4. Jacobians You Implemented

**gradHx (2×3):** How observation changes with robot state [x, y, θ]
**gradHm (2×2):** How observation changes with landmark [m_x, m_y]
**gradGx (2×3):** How landmark changes with robot state
**gradGw (2×2):** How landmark changes with measurement

These enable EKF to handle nonlinear models!

---

## Implementation Quick Reference

### Activity 2: Forward Observation Model

```matlab
% In predictSLAMObservation:
dXY = mXY - x(1:2);
r = sqrt(sum(dXY.^2));
beta = atan2(dXY(2), dXY(1)) - x(3);  % ← Subtract θ!
z = [r; beta];
```

### Activity 3: Inverse Observation Model

```matlab
% In predictLandmarkFromSLAMObservation:
phi = x(3) + z(2);  % World angle = θ + β
r = z(1);

mXY = [x(1) + r * cos(phi);
       x(2) + r * sin(phi)];

gradGx = [1,  0,  -r * sin(phi);
          0,  1,   r * cos(phi)];

gradGw = [cos(phi),  -r * sin(phi);
          sin(phi),   r * cos(phi)];
```

### Activity 5: Observation Jacobians

```matlab
% In predictSLAMObservation (after computing z):
dx = dXY(1);
dy = dXY(2);
r2 = r * r;

gradHx = [-dx/r,   -dy/r,    0;      % ∂r/∂[x,y,θ]
           dy/r2,  -dx/r2,  -1];     % ∂β/∂[x,y,θ]

gradHm = [ dx/r,   dy/r;             % ∂r/∂[m_x,m_y]
          -dy/r2,  dx/r2];           % ∂β/∂[m_x,m_y]

gradHw = eye(2);
```

---

## Results Analysis

### Position Error Plot (Activity 5)

**Yellow line:** Actual error  
**Red dashed lines:** ±2σ uncertainty bounds

✅ Error stays within bounds (consistent!)  
✅ Bounds shrink over time (EKF working!)  
✅ Typical error: ±0.2 meters (excellent!)

### Map Plot (Activity 5)

**Green crosses:** True landmarks  
**Yellow crosses:** Estimated landmarks  

✅ Yellow very close to green (<0.5m error)  
✅ No visible uncertainty ellipses (too small!)  
✅ Accurate, confident estimates!

---

## Key Equations

### Motion Model (Nonlinear)
```matlab
x[k+1] = x[k] + ΔT × M(θ[k]) × u[k+1]

M(θ) = [cos(θ)  -sin(θ)  0]  ← Rotation matrix
       [sin(θ)   cos(θ)  0]
       [0        0       1]
```

### Observation Model (Nonlinear)
```matlab
r = sqrt(Δx² + Δy²)
β = atan2(Δy, Δx) - θ
```

### EKF Algorithm
```matlab
% Prediction
x̂ = f(x, u)               % Nonlinear
F = ∂f/∂x                 % Jacobian
P = F×P×F' + Q            % Linearized

% Update
ν = z - h(x,m)            % Nonlinear innovation
H = ∂h/∂x                 % Jacobian (your code!)
S = H×P×H' + R
K = P×H'/S
x = x + K×ν               % Correct estimate
P = P - K×S×K'            % Reduce uncertainty
```

---

## Critical Takeaways

### 1. Orientation Matters!
- Orientation errors amplify with distance
- Create anisotropic (directional) uncertainty
- Correcting θ is as important as correcting position

### 2. Jacobians Enable Nonlinear SLAM
- Linearize models locally using derivatives
- Allow Kalman Filter to work with realistic sensors
- Essential for practical robotics

### 3. EKF Updates Reduce Uncertainty
- Activity 3: Large ellipses (no updates)
- Activity 5: Tiny ellipses (with EKF updates)
- Re-observing landmarks improves ALL estimates

### 4. Cross-Correlations Propagate Information
- Observing one landmark improves robot AND other landmarks
- GPS update improves landmark estimates too
- This is the magic of SLAM!

### 5. Sensor Fusion is Powerful
- SLAM: High-rate, relative measurements
- GPS: Low-rate, absolute position
- Compass: Absolute orientation
- **Together:** Best performance!

---

## Common Mistakes

❌ **Wrong atan2 order:** `atan2(x, y)` → Should be `atan2(y, x)`  
❌ **Forgot to subtract θ:** Bearing in world frame instead of robot frame  
❌ **Wrong Jacobian signs:** Check carefully!  
❌ **Used r instead of r²:** Bearing derivatives need r²  
❌ **Forgot robot offset:** Landmark = robot_position + polar_conversion

---

## Debugging Commands

```matlab
% Run activities
>> l3.activity5  % Full EKF-SLAM

% Edit implementation
>> edit l3.trianglebot.SystemModel

% Check state size (should grow)
>> length(slamSystem.x)

% Check uncertainty (should decrease)
>> trace(slamSystem.P)
```

---

## Lab 02 vs Lab 03 Comparison

| Feature | Lab 02 | Lab 03 |
|---------|--------|--------|
| Robot | Point (x,y) | Oriented (x,y,θ) |
| Filter | Kalman Filter | Extended KF |
| Models | Linear | Nonlinear |
| Sensor | Cartesian | Range & Bearing |
| Jacobians | Not needed | Essential |
| Uncertainty | Circular | Elliptical |
| Realism | Academic | Practical ✅ |

---

## Real-World Applications

- **Self-driving cars** - Navigate with GPS + LiDAR + cameras
- **Drones** - Indoor navigation without GPS
- **Warehouse robots** - Autonomous material handling
- **Mars rovers** - Explore unknown terrain
- **AR/VR** - Track headset position while mapping room

---

## Revision Checklist

- [ ] Understand orientation error amplification
- [ ] Know forward vs inverse models
- [ ] Can derive simple Jacobians
- [ ] Understand EKF linearization concept
- [ ] Know why cross-correlations matter
- [ ] Can implement observation models
- [ ] Understand sensor fusion benefits

---

## Final Summary

**You implemented a complete EKF-SLAM system that:**
- Handles realistic nonlinear sensors (range & bearing)
- Accounts for robot orientation
- Uses Jacobians to linearize locally
- Achieves <0.5m accuracy
- Demonstrates sensor fusion with GPS/compass

**This is foundational to modern autonomous systems!** 🚀

**Key Lesson:**
> *EKF extends Kalman Filter to nonlinear systems using Jacobians. Small implementation details (signs, angle wrapping, r²) matter enormously for correct operation!*

---

**Congratulations on mastering EKF-SLAM!** 🎉

*Now you understand the math behind autonomous vehicles, drones, and robots!*