#!/usr/bin/env python3
"""
Factor Graph Optimization for LiDAR SLAM — standalone synthetic demo.

This module exists only to illustrate the factor-graph concept on a
controlled synthetic loop. The production pose-graph optimisation used for
Q3d lives in `q3_lidar_slam_complete.py` (GTSAM-backed `Pose2` LM).

The demo:
  1. Generates a ground-truth circular trajectory.
  2. Simulates NOISY relative odometry (with per-step Gaussian noise on
     dx, dy, dtheta) — this is the key fix that lets the optimiser have
     real drift to correct.
  3. Adds one loop-closure factor between the first and last pose with
     identity relative transform (the robot is physically back at start).
  4. Runs SLSQP on the negative log-likelihood sum and measures the closure
     error before vs after.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

try:
    import gtsam
    from gtsam import Pose2, BetweenFactorPose2, PriorFactorPose2, noiseModel
    _HAVE_GTSAM = True
except Exception:
    _HAVE_GTSAM = False

class FactorGraphOptimizer:
    def __init__(self):
        """Factor graph optimization for LiDAR SLAM loop closure"""
        self.odometry_factors = []
        self.loop_closure_factors = []
        self.poses = []
        self.optimization_history = []
        
    def add_odometry_factor(self, pose_id1, pose_id2, relative_transform, info_matrix):
        """Add odometry constraint between consecutive poses"""
        factor = {
            'type': 'odometry',
            'pose1': pose_id1,
            'pose2': pose_id2,
            'measurement': relative_transform,
            'information': info_matrix
        }
        self.odometry_factors.append(factor)
    
    def add_loop_closure_factor(self, pose_id1, pose_id2, relative_transform, info_matrix):
        """Add loop closure constraint between distant poses"""
        factor = {
            'type': 'loop_closure', 
            'pose1': pose_id1,
            'pose2': pose_id2,
            'measurement': relative_transform,
            'information': info_matrix
        }
        self.loop_closure_factors.append(factor)
    
    def pose_to_vector(self, pose_matrix):
        """Convert 3x3 pose matrix to [x, y, theta] vector"""
        x = pose_matrix[0, 2]
        y = pose_matrix[1, 2]
        theta = np.arctan2(pose_matrix[1, 0], pose_matrix[0, 0])
        return np.array([x, y, theta])
    
    def vector_to_pose(self, pose_vector):
        """Convert [x, y, theta] vector to 3x3 pose matrix"""
        x, y, theta = pose_vector
        c, s = np.cos(theta), np.sin(theta)
        pose = np.eye(3)
        pose[0:2, 0:2] = [[c, -s], [s, c]]
        pose[0:2, 2] = [x, y]
        return pose
    
    def relative_pose_error(self, pose1_vec, pose2_vec, expected_relative):
        """Calculate error between actual and expected relative pose"""
        pose1 = self.vector_to_pose(pose1_vec)
        pose2 = self.vector_to_pose(pose2_vec)
        
        # Actual relative transform
        actual_relative = np.linalg.inv(pose1) @ pose2
        actual_relative_vec = self.pose_to_vector(actual_relative)
        expected_relative_vec = self.pose_to_vector(expected_relative)
        
        # Calculate error
        error = actual_relative_vec - expected_relative_vec
        
        # Normalize angle error to [-pi, pi]
        error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
        
        return error
    
    def _objective(self, free_vars, anchor_vec):
        """
        Weighted-least-squares cost. Pose 0 is fixed to `anchor_vec`; only
        poses 1..N-1 are free variables. This removes the global gauge
        freedom that would otherwise make the problem under-determined.
        """
        num_free  = len(free_vars) // 3
        poses_vec = np.empty((num_free + 1, 3))
        poses_vec[0]  = anchor_vec
        poses_vec[1:] = free_vars.reshape((num_free, 3))

        total = 0.0
        for factor in self.odometry_factors:
            e = self.relative_pose_error(poses_vec[factor['pose1']],
                                          poses_vec[factor['pose2']],
                                          factor['measurement'])
            total += e.T @ factor['information'] @ e

        for factor in self.loop_closure_factors:
            e = self.relative_pose_error(poses_vec[factor['pose1']],
                                          poses_vec[factor['pose2']],
                                          factor['measurement'])
            total += e.T @ factor['information'] @ e
        return total

    def _optimize_gtsam(self, initial_poses):
        """GTSAM Levenberg-Marquardt on Pose2 — handles angle wrap correctly."""
        graph   = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        p0 = self.pose_to_vector(initial_poses[0])
        prior_noise = noiseModel.Diagonal.Sigmas(np.array([1e-6, 1e-6, 1e-8]))
        graph.add(PriorFactorPose2(0, Pose2(p0[0], p0[1], p0[2]), prior_noise))

        for k, P in enumerate(initial_poses):
            v = self.pose_to_vector(P)
            initial.insert(k, Pose2(v[0], v[1], v[2]))

        def _info_to_sigmas(info):
            # Diagonal information → standard deviations (1/√σ²).
            diag = np.maximum(np.diag(info), 1e-9)
            return np.sqrt(1.0 / diag)

        for f in self.odometry_factors + self.loop_closure_factors:
            z     = self.pose_to_vector(f['measurement'])
            sig   = _info_to_sigmas(np.asarray(f['information'], dtype=float))
            noise = noiseModel.Diagonal.Sigmas(sig)
            graph.add(BetweenFactorPose2(f['pose1'], f['pose2'],
                                          Pose2(z[0], z[1], z[2]), noise))

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(200)
        result = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()

        opt = []
        for k in range(len(initial_poses)):
            p = result.atPose2(k)
            opt.append(self.vector_to_pose(np.array([p.x(), p.y(), p.theta()])))
        return opt, True, 0  # success + iter count unused

    def _optimize_scipy(self, initial_poses, max_iterations):
        """SLSQP fallback with equality constraint to fix pose 0 (used when GTSAM unavailable)."""
        initial_vec = np.array([self.pose_to_vector(P) for P in initial_poses])
        anchor_vec  = initial_vec[0]
        free0       = initial_vec[1:].flatten()
        result = minimize(self._objective, free0, args=(anchor_vec,),
                          method='SLSQP',
                          options={'maxiter': max_iterations, 'ftol': 1e-10})
        free_vec = result.x.reshape((len(initial_poses) - 1, 3))
        opt = [self.vector_to_pose(anchor_vec)] + \
              [self.vector_to_pose(v) for v in free_vec]
        return opt, bool(result.success), int(result.nit)

    def optimize_poses(self, initial_poses, max_iterations=200, backend='auto'):
        """
        Optimize pose graph. Pose 0 is held fixed (anchor).
        `backend` ∈ {'auto', 'gtsam', 'scipy'}:
          - 'auto'  → GTSAM if importable, else SciPy SLSQP
          - 'gtsam' → hard requirement on GTSAM
          - 'scipy' → always use the SciPy SLSQP fallback
        """
        before_poses = [P.copy() for P in initial_poses]
        before_error = self.calculate_closure_error(before_poses)

        use_gtsam = (backend == 'gtsam') or (backend == 'auto' and _HAVE_GTSAM)
        if use_gtsam and not _HAVE_GTSAM:
            raise RuntimeError("GTSAM requested but not importable.")
        if use_gtsam:
            opt_poses, success, iters = self._optimize_gtsam(initial_poses)
        else:
            opt_poses, success, iters = self._optimize_scipy(initial_poses,
                                                              max_iterations)

        after_error = self.calculate_closure_error(opt_poses)
        denom       = max(before_error, 1e-9)
        optimization_result = {
            'backend':              'gtsam' if use_gtsam else 'scipy',
            'success':              success,
            'iterations':           iters,
            'before_closure_error': before_error,
            'after_closure_error':  after_error,
            'improvement_percent':  (before_error - after_error) / denom * 100.0,
            'before_poses':         before_poses,
            'after_poses':          opt_poses,
        }
        self.optimization_history.append(optimization_result)
        return optimization_result
    
    def calculate_closure_error(self, poses):
        """Calculate Euclidean distance between start and end pose"""
        if len(poses) < 2:
            return 0.0
            
        start_pose = poses[0]
        end_pose = poses[-1]
        
        start_position = start_pose[0:2, 2]
        end_position = end_pose[0:2, 2]
        
        return np.linalg.norm(end_position - start_position)
    
    def visualize_optimization_results(self, result, title="Factor Graph Optimization"):
        """Create before/after visualization of optimization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Before optimization
        before_poses = result['before_poses']
        before_positions = np.array([[pose[0, 2], pose[1, 2]] for pose in before_poses])
        
        ax1.plot(before_positions[:, 0], before_positions[:, 1], 'b-o', linewidth=2, markersize=4)
        ax1.scatter(before_positions[0, 0], before_positions[0, 1], c='green', s=100, marker='s', label='Start')
        ax1.scatter(before_positions[-1, 0], before_positions[-1, 1], c='red', s=100, marker='X', label='End')
        ax1.set_title(f'Before Optimization\\nClosure Error: {result["before_closure_error"]:.3f}m')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.legend()
        ax1.grid(True)
        ax1.set_aspect('equal')
        
        # After optimization
        after_poses = result['after_poses']
        after_positions = np.array([[pose[0, 2], pose[1, 2]] for pose in after_poses])
        
        ax2.plot(after_positions[:, 0], after_positions[:, 1], 'r-o', linewidth=2, markersize=4)
        ax2.scatter(after_positions[0, 0], after_positions[0, 1], c='green', s=100, marker='s', label='Start')
        ax2.scatter(after_positions[-1, 0], after_positions[-1, 1], c='red', s=100, marker='X', label='End')
        ax2.set_title(f'After Optimization\\nClosure Error: {result["after_closure_error"]:.3f}m')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.legend()
        ax2.grid(True)
        ax2.set_aspect('equal')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        return fig

def demonstrate_factor_graph_optimization(seed=42, num_poses=20, radius=2.0,
                                           odom_sigma_t=0.05, odom_sigma_a=0.02):
    """
    Synthetic circular loop with NOISY odometry factors.

    Key correction vs the original demo: odometry factors now carry the
    *integrated-noisy* relative transform, not the ground-truth one, so the
    optimiser genuinely has drift to correct when the loop closure is added.
    """
    rng       = np.random.default_rng(seed)
    optimizer = FactorGraphOptimizer()

    # Ground-truth circular trajectory (not shown to the optimiser)
    true_poses = []
    for i in range(num_poses):
        angle = (i / num_poses) * 2 * np.pi
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        theta = angle + np.pi / 2
        T = np.eye(3)
        T[:2, :2] = [[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]]
        T[:2, 2] = [x, y]
        true_poses.append(T)

    # Simulate noisy odometry: take each ground-truth relative transform
    # and corrupt it with zero-mean Gaussian noise.
    noisy_rel = []
    for i in range(num_poses - 1):
        T_rel = np.linalg.inv(true_poses[i]) @ true_poses[i + 1]
        dx_true    = T_rel[0, 2]
        dy_true    = T_rel[1, 2]
        dth_true   = np.arctan2(T_rel[1, 0], T_rel[0, 0])
        dx  = dx_true  + rng.normal(0.0, odom_sigma_t)
        dy  = dy_true  + rng.normal(0.0, odom_sigma_t)
        dth = dth_true + rng.normal(0.0, odom_sigma_a)
        c, s = np.cos(dth), np.sin(dth)
        T_noisy = np.eye(3)
        T_noisy[:2, :2] = [[c, -s], [s, c]]
        T_noisy[:2, 2]  = [dx, dy]
        noisy_rel.append(T_noisy)

    # Integrate noisy odometry → noisy absolute poses (input to optimiser).
    noisy_poses = [true_poses[0].copy()]
    for T_rel in noisy_rel:
        noisy_poses.append(noisy_poses[-1] @ T_rel)

    # Information matrix: higher weight for odometry than loop closure since
    # loop closures are also noisy observations in practice.
    info_odom = np.diag([1.0 / odom_sigma_t**2,
                         1.0 / odom_sigma_t**2,
                         1.0 / odom_sigma_a**2])
    info_loop = info_odom * 0.5

    # Odometry factors carry the NOISY relative transforms.
    for i, T_rel in enumerate(noisy_rel):
        optimizer.add_odometry_factor(i, i + 1, T_rel, info_odom)

    # Loop-closure factor: the physical robot sees pose 0 and pose N-1 as
    # *nearly* coincident (end of the circle). The "true" ground-truth
    # relative transform is the SE(2) transform between the two ground-truth
    # poses — this is what a loop closure detector would estimate from the
    # observation. We add small Gaussian noise to mimic a noisy ICP/match.
    T_rel_gt = np.linalg.inv(true_poses[0]) @ true_poses[-1]
    lc_x  = T_rel_gt[0, 2] + rng.normal(0.0, 0.02)
    lc_y  = T_rel_gt[1, 2] + rng.normal(0.0, 0.02)
    lc_th = np.arctan2(T_rel_gt[1, 0], T_rel_gt[0, 0]) + rng.normal(0.0, 0.01)
    c, s  = np.cos(lc_th), np.sin(lc_th)
    T_lc  = np.eye(3); T_lc[:2, :2] = [[c, -s], [s, c]]; T_lc[:2, 2] = [lc_x, lc_y]
    optimizer.add_loop_closure_factor(0, num_poses - 1, T_lc, info_loop)

    result = optimizer.optimize_poses(noisy_poses)
    fig    = optimizer.visualize_optimization_results(
        result, "Factor Graph Loop Closure Optimization (synthetic, noisy odometry)")
    return optimizer, result, fig


if __name__ == "__main__":
    optimizer, result, fig = demonstrate_factor_graph_optimization()

    print("=== Factor Graph Optimization Results (synthetic demo) ===")
    print(f"Backend:                 {result['backend']}")
    print(f"Optimization Success:    {result['success']}")
    print(f"Iterations:              {result['iterations']}")
    print(f"Before Closure Error:    {result['before_closure_error']:.3f} m")
    print(f"After  Closure Error:    {result['after_closure_error']:.3f} m")
    print(f"Improvement:             {result['improvement_percent']:.1f} %")

    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.abspath(os.path.join(_here, '..'))
    out_dir = os.environ.get('SLAM_OUT',
                              os.path.join(_root, 'data', 'q3_results'))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'factor_graph_demo_synthetic.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved demo figure → {out}")