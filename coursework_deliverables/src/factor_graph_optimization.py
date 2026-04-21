#!/usr/bin/env python3
"""
Factor Graph Optimization for LiDAR SLAM
Implements loop closure optimization using factor graph approach
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json

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
    
    def objective_function(self, poses_flat):
        """Objective function for optimization"""
        total_error = 0.0
        num_poses = len(poses_flat) // 3
        
        # Reshape flat array to poses
        poses_vec = poses_flat.reshape((num_poses, 3))
        
        # Odometry factors
        for factor in self.odometry_factors:
            pose1_vec = poses_vec[factor['pose1']]
            pose2_vec = poses_vec[factor['pose2']]
            error = self.relative_pose_error(pose1_vec, pose2_vec, factor['measurement'])
            
            # Weighted by information matrix
            weighted_error = error.T @ factor['information'] @ error
            total_error += weighted_error
        
        # Loop closure factors
        for factor in self.loop_closure_factors:
            pose1_vec = poses_vec[factor['pose1']]
            pose2_vec = poses_vec[factor['pose2']]
            error = self.relative_pose_error(pose1_vec, pose2_vec, factor['measurement'])
            
            # Higher weight for loop closure constraints
            weighted_error = error.T @ (factor['information'] * 10.0) @ error
            total_error += weighted_error
        
        return total_error
    
    def optimize_poses(self, initial_poses, max_iterations=100):
        """Optimize pose graph using factor graph formulation"""
        # Convert poses to optimization variables
        initial_poses_vec = np.array([self.pose_to_vector(pose) for pose in initial_poses])
        initial_guess = initial_poses_vec.flatten()
        
        # Store before optimization for comparison
        before_poses = initial_poses.copy()
        before_error = self.calculate_closure_error(before_poses)
        
        # Run optimization
        result = minimize(
            self.objective_function, 
            initial_guess,
            method='BFGS',
            options={'maxiter': max_iterations}
        )
        
        # Convert back to poses
        optimized_poses_vec = result.x.reshape((len(initial_poses), 3))
        optimized_poses = [self.vector_to_pose(pose_vec) for pose_vec in optimized_poses_vec]
        
        # Store after optimization for comparison
        after_poses = optimized_poses
        after_error = self.calculate_closure_error(after_poses)
        
        optimization_result = {
            'success': result.success,
            'iterations': result.nit,
            'before_closure_error': before_error,
            'after_closure_error': after_error,
            'improvement_percent': (before_error - after_error) / before_error * 100,
            'before_poses': before_poses,
            'after_poses': after_poses
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

# Demonstration of factor graph optimization
def demonstrate_factor_graph_optimization():
    """Demonstrate factor graph optimization with synthetic data"""
    
    optimizer = FactorGraphOptimizer()
    
    # Create synthetic trajectory with drift
    true_poses = []
    noisy_poses = []
    
    # Generate circular trajectory
    num_poses = 20
    radius = 2.0
    
    for i in range(num_poses):
        angle = (i / num_poses) * 2 * np.pi
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        theta = angle + np.pi/2
        
        # True pose
        true_pose = np.eye(3)
        true_pose[0:2, 0:2] = [[np.cos(theta), -np.sin(theta)], 
                               [np.sin(theta), np.cos(theta)]]
        true_pose[0:2, 2] = [x, y]
        true_poses.append(true_pose)
        
        # Noisy pose with accumulated drift
        drift_x = i * 0.05  # Accumulated drift
        drift_y = i * 0.03
        drift_theta = i * 0.02
        
        noisy_pose = np.eye(3)
        noisy_pose[0:2, 0:2] = [[np.cos(theta + drift_theta), -np.sin(theta + drift_theta)], 
                                [np.sin(theta + drift_theta), np.cos(theta + drift_theta)]]
        noisy_pose[0:2, 2] = [x + drift_x, y + drift_y]
        noisy_poses.append(noisy_pose)
    
    # Add odometry factors
    info_matrix = np.eye(3) * 100  # High confidence in odometry
    
    for i in range(len(noisy_poses) - 1):
        relative_transform = np.linalg.inv(true_poses[i]) @ true_poses[i+1]
        optimizer.add_odometry_factor(i, i+1, relative_transform, info_matrix)
    
    # Add loop closure factor (detect that first and last poses should be close)
    loop_info_matrix = np.eye(3) * 50  # Medium confidence in loop closure
    loop_relative = np.eye(3)  # Should be identity for perfect loop
    optimizer.add_loop_closure_factor(0, num_poses-1, loop_relative, loop_info_matrix)
    
    # Optimize
    result = optimizer.optimize_poses(noisy_poses)
    
    # Visualize results
    fig = optimizer.visualize_optimization_results(result, "Factor Graph Loop Closure Optimization")
    
    return optimizer, result, fig

if __name__ == "__main__":
    optimizer, result, fig = demonstrate_factor_graph_optimization()
    
    print("=== Factor Graph Optimization Results ===")
    print(f"Optimization Success: {result['success']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Before Closure Error: {result['before_closure_error']:.3f}m")
    print(f"After Closure Error: {result['after_closure_error']:.3f}m")
    print(f"Improvement: {result['improvement_percent']:.1f}%")
    
    plt.savefig('/home/mmaaz/SLAM/coursework_deliverables/part3_lidar_slam/factor_graph_optimization.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Factor graph optimization visualization saved!")