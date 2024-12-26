import cv2
import numpy as np
import pygame
import heapq
import time
import random

def extractGraph(input):
    kernel = np.ones((3, 3), np.uint8)
    img = input.copy()
    erosion = cv2.erode(img, kernel, iterations=2)

    GRAY_img = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)

    (thresh, BW_img) = cv2.threshold(GRAY_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return BW_img

def heuristic(a, b):
    """
    Heuristic function using Manhattan distance.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def A_star(graph, start, end, dynamic_changes=False):
    start = start[1], start[0]  # Convert (x, y) to (row, col)
    end = end[1], end[0]  # Convert (x, y) to (row, col)

    height, width = graph.shape

    def isWall(graph, point):
        if graph[point[0], point[1]] < 50:
            return True
        return False

    open_list = []  # Priority queue for A*
    heapq.heappush(open_list, (0, start))  # (f_score, point)

    came_from = {}  # Track path
    g_score = {start: 0}  # Cost from start to a node
    f_score = {start: heuristic(start, end)}  # Estimated cost to the goal

    total_nodes = 0
    start_time = time.time()  # Start time for computation

    while open_list:
        current = heapq.heappop(open_list)[1]  # Get the node with the lowest f_score

        if current == end:
            # Reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            end_time = time.time()
            computation_time = end_time - start_time

            # Path length and efficiency
            path_length = len(path)
            image_area = height * width
            efficiency = total_nodes / image_area

            # Adaptability (simulated by random dynamic obstacles)
            adaptability = "High" if dynamic_changes else "Low"

            # Real-world applicability based on path length
            if path_length < 100:
                real_world_applicability = "High"
            elif path_length < 300:
                real_world_applicability = "Medium"
            else:
                real_world_applicability = "Low"

            print("Total Nodes Explored = ", total_nodes)
            print("Path Length = ", path_length, "steps")
            print("Computation Time = ", round(computation_time, 3), "seconds")
            print("Efficiency (Nodes explored / Image Area): ", round(efficiency, 6), "nodes per pixel")
            print("Adaptability to Dynamic Changes: ", adaptability)
            print("Real-World Applicability: ", real_world_applicability)

            return path[::-1]  # Return reversed path

        neighbors = [
            (current[0], current[1] - 1),
            (current[0] - 1, current[1]),
            (current[0], current[1] + 1),
            (current[0] + 1, current[1])
        ]

        for neighbor in neighbors:
            if neighbor[0] < 0 or neighbor[1] < 0 or neighbor[0] >= height or neighbor[1] >= width:
                continue  # Skip out-of-bounds neighbors
            if isWall(graph, neighbor):
                continue  # Skip walls

            # Simulate dynamic changes (random obstacles)
            if dynamic_changes and random.random() < 0.1:  # 10% chance to add an obstacle
                tx, ty = random.randint(5, height-5), random.randint(5, width-5)
                graph[tx, ty] = 0  # Mark as obstacle

            tentative_g_score = g_score[current] + 1  # Assume uniform cost grid

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                # This path is better
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

            total_nodes += 1

    return []  # No path found

if __name__ == "__main__":
    # Load the image
    imgFile = cv2.imread(r"D:\Techno\PathPlanning\Final_Pathplanning\1.jpg")
    backupImg = imgFile.copy()

    # Initialize Pygame
    pygame.init()

    # Convert the image to RGB for Pygame
    imgRGB = cv2.cvtColor(imgFile, cv2.COLOR_BGR2RGB)
    height, width, _ = imgRGB.shape

    # Create a Pygame window
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Select Start and End Points")

    # Convert the image to Pygame surface
    imgSurface = pygame.surfarray.make_surface(np.flipud(np.rot90(imgRGB)))

    start_point = None
    end_point = None
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()

                if start_point is None:
                    start_point = (x, y)
                    print(f"Start Point Selected: {start_point}")
                elif end_point is None:
                    end_point = (x, y)
                    print(f"End Point Selected: {end_point}")

        # Draw the image and selected points
        screen.blit(imgSurface, (0, 0))

        if start_point:
            pygame.draw.circle(screen, (255, 0, 255), start_point, 5)  # Magenta for start
        if end_point:
            pygame.draw.circle(screen, (0, 255, 255), end_point, 5)  # Cyan for end

        pygame.display.flip()

        # Start processing when both points are selected
        if start_point and end_point:
            running = False

    # Clean up Pygame
    pygame.quit()

    # Extract the graph and compute the path
    graph = extractGraph(imgFile)
    path = A_star(graph, start_point, end_point, dynamic_changes=True)  # Enable dynamic changes

    # Draw the path on the image
    for point in path:
        point = point[1], point[0]  # Convert back to (x, y)
        cv2.circle(imgFile, point, 1, (128, 255, 255), -1)

    cv2.circle(imgFile, end_point, 3, (255, 255, 0), -1)
    cv2.circle(imgFile, start_point, 3, (255, 0, 255), -1)

    # Show the results
    cv2.imshow("Path", imgFile)
    cv2.imshow("Raw Image", backupImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
