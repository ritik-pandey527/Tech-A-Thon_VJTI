import time
import numpy as np
import math
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
from informed_rrt_star import IRrtStar

class PathPlanningMetrics:
    def __init__(self, rrt_star_instance):
        self.rrt_star = rrt_star_instance

    def compute_path_length(self):
        """Calculate the path length"""
        path = self.rrt_star.path
        length = 0.0
        for i in range(1, len(path)):
            length += math.hypot(path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        return length

    def compute_computation_time(self, start_time, end_time):
        """Compute computation time"""
        return end_time - start_time

    def compute_efficiency(self):
        """Calculate efficiency based on number of nodes and iterations"""
        nodes_visited = len(self.rrt_star.V)
        iterations = self.rrt_star.iter_max
        efficiency = nodes_visited / iterations
        return efficiency

    def compute_adaptability(self):
        """Assess adaptability based on changes in the environment (e.g., obstacles)"""
        # Example: Evaluate adaptability by tracking if obstacles caused any planning failure
        adaptability = "High"  # Placeholder, this can be modified based on specific criteria
        return adaptability

    def compute_real_world_applicability(self):
        """Assess real-world applicability based on path smoothness and obstacle avoidance"""
        path_smoothness = self.compute_path_smoothness()
        real_world_applicability = "High" if path_smoothness < 10 else "Low"
        return real_world_applicability

    def compute_path_smoothness(self):
        """Compute path smoothness (less jaggedness means smoother)"""
        path = self.rrt_star.path
        smoothness = 0.0
        for i in range(2, len(path)):
            p0 = path[i-2]
            p1 = path[i-1]
            p2 = path[i]
            # Compute angle between three consecutive points to measure smoothness
            angle = self.compute_angle(p0, p1, p2)
            smoothness += abs(angle)
        return smoothness

    def compute_angle(self, p0, p1, p2):
        """Calculate the angle between three points"""
        vector1 = np.array([p1[0] - p0[0], p1[1] - p0[1]])
        vector2 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return np.arccos(cosine_angle) * 180 / np.pi  # return angle in degrees


def main():
    # Initialize the IRrtStar planning instance
    x_start = (18, 8)
    x_goal = (37, 18)
    rrt_star = IRrtStar(x_start, x_goal, 1, 0.10, 12, 1000)
    
    # Measure computation time
    start_time = time.time()
    rrt_star.planning()
    end_time = time.time()

    # Calculate the metrics
    metrics = PathPlanningMetrics(rrt_star)
    
    # Compute all the parameters
    path_length = metrics.compute_path_length()
    computation_time = metrics.compute_computation_time(start_time, end_time)
    efficiency = metrics.compute_efficiency()
    adaptability = metrics.compute_adaptability()
    real_world_applicability = metrics.compute_real_world_applicability()

    # Print the results
    print(f"Path Length: {path_length:.2f} units")
    print(f"Computation Time: {computation_time:.4f} seconds")
    print(f"Efficiency: {efficiency:.4f} (Nodes visited / Iterations)")
    print(f"Adaptability: {adaptability}")
    print(f"Real-World Applicability: {real_world_applicability}")
    print(f"Path Smoothness: {metrics.compute_path_smoothness():.2f} degrees (less is smoother)")


if __name__ == '__main__':
    main()
