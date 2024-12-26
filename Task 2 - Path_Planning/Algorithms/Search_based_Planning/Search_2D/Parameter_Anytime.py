import time
import numpy as np
from collections import namedtuple
from Anytime_D_star import ADStar
# Mock event class to simulate the mouse click event
Event = namedtuple('Event', ['xdata', 'ydata'])

class ADStarEvaluator:
    def __init__(self, s_start, s_goal, eps, heuristic_type):
        # Initialize the ADStar object
        self.dstar = ADStar(s_start, s_goal, eps, heuristic_type)
        self.s_start = s_start
        self.s_goal = s_goal
        self.path_length = 0
        self.computation_time = 0
        self.environment_changes = 0
        self.optimal_path = None
        self.optimal_path_length = 0

        # Initial obstacles set (assuming `self.dstar.obs` is a set of obstacles)
        self.initial_obs = set(self.dstar.obs)

    def evaluate_efficiency(self):
        # Start time for computation
        start_time = time.time()

        # Run the AD* algorithm
        self.dstar.run()

        # Calculate computation time
        self.computation_time = time.time() - start_time
        print(f"Computation Time: {self.computation_time} seconds")

        # Calculate path length
        path = self.dstar.extract_path()
        self.path_length = len(path)
        print(f"Path Length: {self.path_length}")

    def evaluate_accuracy(self):
        print("\nEvaluating accuracy...")

        # Compute the optimal path using a simpler algorithm (Dijkstra/A*)
        optimal_path = self.compute_optimal_path()
        self.optimal_path = optimal_path
        self.optimal_path_length = len(optimal_path)

        print(f"Optimal Path Length: {self.optimal_path_length}")

        # Compare the found path length to the optimal path length
        accuracy = self.optimal_path_length / self.path_length if self.path_length > 0 else 0
        print(f"Accuracy: {accuracy * 100:.2f}%")

    def compute_optimal_path(self):
        """
        Compute the optimal path using a simpler pathfinding algorithm (Dijkstra or A*).
        This function will return the optimal path from start to goal, ignoring obstacles.
        """

        def dijkstra(start, goal):
            """
            Simple Dijkstra's algorithm to find the shortest path.
            """
            queue = [start]
            distances = {start: 0}
            previous_nodes = {start: None}
            
            while queue:
                current_node = min(queue, key=lambda node: distances[node])
                queue.remove(current_node)
                
                if current_node == goal:
                    break
                
                for neighbor in self.dstar.get_neighbor(current_node):
                    if neighbor not in distances or distances[current_node] + 1 < distances[neighbor]:
                        distances[neighbor] = distances[current_node] + 1
                        previous_nodes[neighbor] = current_node
                        queue.append(neighbor)
            
            # Reconstruct the optimal path from goal to start
            path = []
            current = goal
            while current:
                path.append(current)
                current = previous_nodes.get(current)
            return path[::-1]  # Reverse the path to start -> goal

        # Compute the optimal path (ignoring obstacles)
        optimal_path = dijkstra(self.s_start, self.s_goal)
        return optimal_path

    def evaluate_adaptability(self):
        # Simulate dynamic environment changes
        print("\nSimulating dynamic changes in the environment...")

        # Add or remove obstacles and track changes
        for _ in range(5):  # Simulate 5 changes
            # Create a mock event with xdata and ydata (simulating a click on (10, 10))
            mock_event = Event(xdata=10, ydata=10)

            # Change position by adding/removing obstacles using the mock event
            self.dstar.on_press(mock_event)  # Pass the mock event here

            # Track environment changes
            self.environment_changes += 1

        # Compare the environment changes
        final_obs = set(self.dstar.obs)
        added = final_obs - self.initial_obs  # Obstacles added
        removed = self.initial_obs - final_obs  # Obstacles removed

        print(f"Added Obstacles: {len(added)}")
        print(f"Removed Obstacles: {len(removed)}")
        print(f"Total Environment Changes: {self.environment_changes}")

        # Update the initial obstacles set to the current state
        self.initial_obs = final_obs

    def evaluate_real_world_applicability(self):
        print("\nEvaluating real-world applicability...")

        # Assume that real-world obstacles and the environment can vary significantly
        dynamic_changes = self.environment_changes > 3  # Example threshold for dynamic changes

        if dynamic_changes:
            print("The algorithm is adaptable to dynamic changes.")
        else:
            print("The algorithm needs further improvements for better adaptability.")

    def print_summary(self):
        print(f"\nSummary:")
        print(f"Computation Time: {self.computation_time} seconds")
        print(f"Path Length: {self.path_length}")
        print(f"Optimal Path Length: {self.optimal_path_length}")
        print(f"Accuracy: {self.optimal_path_length / self.path_length * 100:.2f}%")
        print(f"Environment Changes: {self.environment_changes}")


# Main execution
def main():
    s_start = (5, 5)
    s_goal = (45, 25)

    evaluator = ADStarEvaluator(s_start, s_goal, 2.5, "euclidean")

    # Evaluate efficiency
    evaluator.evaluate_efficiency()

    # Evaluate accuracy
    evaluator.evaluate_accuracy()

    # Evaluate adaptability
    evaluator.evaluate_adaptability()

    # Evaluate real-world applicability
    evaluator.evaluate_real_world_applicability()

    # Print a summary of results
    evaluator.print_summary()


if __name__ == "__main__":
    main()
