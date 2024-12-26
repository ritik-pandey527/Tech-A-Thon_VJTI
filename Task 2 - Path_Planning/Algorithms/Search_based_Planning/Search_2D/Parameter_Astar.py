import time
import math
import os
import sys
import heapq
from Astar import AStar
# Assuming plotting is imported from your Search_2D module (adjust the path if needed)
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Search_based_Planning/")
from Search_2D import plotting, env


class AStarEvaluator:
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        # Initialize AStar instance
        self.astar = AStar(s_start, s_goal, heuristic_type)
        self.path_length = 0
        self.computation_time = 0
        self.optimal_path = None
        self.optimal_path_length = 0
        self.environment_changes = 0
        self.added_obstacles = 0
        self.removed_obstacles = 0

        # Initialize plotting object for visualization
        self.plot = plotting.Plotting(s_start, s_goal)

    def evaluate_efficiency(self):
        """Calculate computation time and path length"""
        start_time = time.time()

        # Run the A* algorithm
        path, visited = self.astar.searching()

        # Compute computation time
        self.computation_time = time.time() - start_time
        print(f"Computation Time: {self.computation_time:.4f} seconds")

        # Calculate path length
        self.path_length = len(path)
        print(f"Path Length: {self.path_length}")

        # Visualize the path and visited nodes
        self.plot.animation(path, visited, "A*")  # Animation of A* path

    def evaluate_accuracy(self):
        """Calculate accuracy based on optimal path"""
        print("\nEvaluating accuracy...")

        # Compute optimal path
        optimal_path = self.compute_optimal_path()
        self.optimal_path = optimal_path
        self.optimal_path_length = len(optimal_path)
        print(f"Optimal Path Length: {self.optimal_path_length}")

        # Compare path lengths
        accuracy = (self.optimal_path_length / self.path_length) if self.path_length > 0 else 0
        print(f"Accuracy: {accuracy * 100:.2f}%")

    def compute_optimal_path(self):
        """
        Compute the optimal path using a simpler algorithm (e.g., Dijkstra).
        """
        def dijkstra(s_start, s_goal):
            queue = [s_start]
            distances = {s_start: 0}
            previous_nodes = {s_start: None}
            
            while queue:
                current_node = min(queue, key=lambda node: distances[node])
                queue.remove(current_node)
                
                if current_node == s_goal:
                    break
                
                for neighbor in self.astar.get_neighbor(current_node):
                    if neighbor not in distances or distances[current_node] + 1 < distances[neighbor]:
                        distances[neighbor] = distances[current_node] + 1
                        previous_nodes[neighbor] = current_node
                        queue.append(neighbor)
            
            # Reconstruct the optimal path from goal to start
            path = []
            current = s_goal
            while current:
                path.append(current)
                current = previous_nodes.get(current)
            return path[::-1]  # Reverse path to get start to goal

        return dijkstra(self.s_start, self.s_goal)

    def evaluate_adaptability(self):
        """Simulate environment changes and track obstacles"""
        print("\nSimulating environment changes...")

        # Track initial state of obstacles
        initial_obs = set(self.astar.obs)

        # Simulate environment changes (obstacle addition/removal)
        for _ in range(5):  # Simulating 5 environment changes
            # Simulate adding/removing obstacles
            self.simulate_obstacle_change()

            # Track environment changes
            self.environment_changes += 1

        # Track changes in obstacles
        final_obs = set(self.astar.obs)
        self.added_obstacles = len(final_obs - initial_obs)  # Obstacles added
        self.removed_obstacles = len(initial_obs - final_obs)  # Obstacles removed

        print(f"Added Obstacles: {self.added_obstacles}")
        print(f"Removed Obstacles: {self.removed_obstacles}")
        print(f"Total Environment Changes: {self.environment_changes}")

    def simulate_obstacle_change(self):
        """Simulate adding/removing obstacles (you can customize this logic)"""
        # Example of a simple change: toggle a random obstacle
        # Here we're just toggling a specific cell as an obstacle
        # You can adjust this based on the environment you're working with
        x, y = 10, 10  # For example, toggle position (10, 10)
        if (x, y) in self.astar.obs:
            self.astar.obs.remove((x, y))  # Remove obstacle
        else:
            self.astar.obs.add((x, y))  # Add obstacle

    def print_summary(self):
        """Print the summary of evaluation results"""
        print("\nSummary of Results:")
        print(f"Computation Time: {self.computation_time:.4f} seconds")
        print(f"Path Length: {self.path_length}")
        print(f"Optimal Path Length: {self.optimal_path_length}")
        print(f"Accuracy: {self.optimal_path_length / self.path_length * 100:.2f}%")
        print(f"Environment Changes: {self.environment_changes}")
        print(f"Added Obstacles: {self.added_obstacles}")
        print(f"Removed Obstacles: {self.removed_obstacles}")


# Main execution
def main():
    s_start = (5, 5)
    s_goal = (45, 25)

    # Initialize evaluator
    evaluator = AStarEvaluator(s_start, s_goal, "euclidean")

    # Evaluate efficiency (computation time, path length)
    evaluator.evaluate_efficiency()

    # Evaluate accuracy (optimal path comparison)
    evaluator.evaluate_accuracy()

    # Evaluate adaptability (environment changes)
    evaluator.evaluate_adaptability()

    # Print summary of all parameters
    evaluator.print_summary()


if __name__ == "__main__":
    main()
