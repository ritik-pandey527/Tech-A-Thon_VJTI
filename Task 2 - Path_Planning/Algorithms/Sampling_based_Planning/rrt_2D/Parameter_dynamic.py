import time
import math
from dynamic_rrt import DynamicRrt
class RRTMetrics:
    def __init__(self, path, vertex, start_time, end_time):
        """
        Initialize the metrics class with path, vertex, start time, and end time.
        
        Args:
            path (list): The final path from start to goal.
            vertex (list): The list of all visited nodes.
            start_time (float): The time when planning started.
            end_time (float): The time when planning ended.
        """
        self.path = path
        self.vertex = vertex
        self.start_time = start_time
        self.end_time = end_time
        
    def compute_path_length(self):
        """
        Compute the length of the path from start to goal.
        
        Returns:
            float: The total path length.
        """
        length = 0
        for i in range(1, len(self.path)):
            x1, y1 = self.path[i-1]
            x2, y2 = self.path[i]
            length += math.hypot(x2 - x1, y2 - y1)
        return length
    
    def compute_computation_time(self):
        """
        Calculate the total time taken to compute the path.
        
        Returns:
            float: Total computation time in seconds.
        """
        return self.end_time - self.start_time
    
    def compute_efficiency(self):
        """
        Efficiency is defined as the ratio of the direct distance from start to goal to the path length.
        
        Returns:
            float: The efficiency of the path.
        """
        x_start, y_start = self.path[0]
        x_goal, y_goal = self.path[-1]
        direct_distance = math.hypot(x_goal - x_start, y_goal - y_start)
        path_length = self.compute_path_length()
        return direct_distance / path_length if path_length > 0 else 0
    
    def compute_adaptability(self):
        """
        Adaptability is measured by the number of nodes re-explored or re-planned.
        
        Returns:
            float: Adaptability metric (proportion of invalid nodes).
        """
        total_nodes = len(self.vertex)
        invalid_nodes = len([node for node in self.vertex if node.flag == "INVALID"])
        return invalid_nodes / total_nodes if total_nodes > 0 else 0
    
    def compute_real_world_applicability(self):
        """
        This metric measures how well the path avoids obstacles and reaches the goal.
        A basic measure is the path efficiency and adaptability combined.
        
        Returns:
            float: Real-world applicability metric.
        """
        efficiency = self.compute_efficiency()
        adaptability = 1 - self.compute_adaptability()
        return 0.5 * efficiency + 0.5 * adaptability
    
    def display_metrics(self):
        """
        Display all the calculated metrics.
        """
        path_length = self.compute_path_length()
        computation_time = self.compute_computation_time()
        efficiency = self.compute_efficiency()
        adaptability = self.compute_adaptability()
        real_world_applicability = self.compute_real_world_applicability()
        
        print("\n=== RRT Path Planning Metrics ===")
        print(f"1. Path Length: {path_length:.2f}")
        print(f"2. Computation Time: {computation_time:.2f} seconds")
        print(f"3. Efficiency: {efficiency:.2f}")
        print(f"4. Adaptability to Dynamic Changes: {adaptability:.2f}")
        print(f"5. Real-World Applicability: {real_world_applicability:.2f}")


def main():
    """
    Example usage of RRTMetrics.
    
    Assume `drrt` is an instance of DynamicRrt and planning is complete.
    """
    
    
    x_start = (2, 2)  # Starting node
    x_goal = (49, 24)  # Goal node
    
    # Start timing the planning process
    start_time = time.time()
    drrt = DynamicRrt(x_start, x_goal, 0.5, 0.1, 0.6, 5000)
    drrt.planning()
    end_time = time.time()  # End timing the planning process
    
    # Collect metrics data
    path = drrt.path  # Extracted path
    vertex = drrt.vertex  # List of visited nodes
    
    # Calculate and display metrics
    metrics = RRTMetrics(path, vertex, start_time, end_time)
    metrics.display_metrics()


if __name__ == "__main__":
    main()