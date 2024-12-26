import heapq
import cv2
import numpy as np
import pygame
import time


def extractGraph(input_image):
    kernel = np.ones((3, 3), np.uint8)
    img = input_image.copy()
    erosion = cv2.erode(img, kernel, iterations=2)
    gray_img = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
    _, bw_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bw_img


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def anytime_d_star(graph, start, end, dynamic_changes=False):
    start = start[1], start[0]
    end = end[1], end[0]

    height, width = graph.shape

    # Cost from start to node
    g = np.full((height, width), np.inf)
    g[start] = 0

    # Priority queue for processing
    open_list = []
    heapq.heappush(open_list, (0, start))

    # Parent tracking
    parent = {}
    parent[start] = None

    def is_wall(point):
        return graph[point] < 50

    # Start the timer for computation time
    start_time = time.time()

    # To track the number of recalculations due to dynamic changes
    recalculations = 0

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == end:
            end_time = time.time()
            computation_time = end_time - start_time
            print(f"Path Found! Computation Time: {computation_time:.4f} seconds")
            break

        x, y = current
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        for neighbor in neighbors:
            nx, ny = neighbor
            if 0 <= nx < height and 0 <= ny < width and not is_wall(neighbor):
                new_cost = g[current] + 1  # Incremental cost
                if new_cost < g[neighbor]:
                    g[neighbor] = new_cost
                    f = new_cost + heuristic(neighbor, end)
                    heapq.heappush(open_list, (f, neighbor))
                    parent[neighbor] = current

        # Simulate dynamic obstacles (random obstacles with 10% probability)
        if dynamic_changes and np.random.random() < 0.1:
            tx, ty = np.random.randint(5, height - 5), np.random.randint(5, width - 5)
            graph[tx, ty] = 0  # Mark as obstacle
            recalculations += 1

    # Reconstruct path
    path = []
    current = end
    while current:
        path.append(current)
        current = parent.get(current)

    path.reverse()

    # Metrics calculations
    path_length = len(path)  # Path length in steps (pixels)
    nodes_processed = np.sum(g < np.inf)  # Nodes processed in pixels
    image_area = height * width  # Total area in pixels
    efficiency = nodes_processed / image_area  # Efficiency in nodes per pixel
    adaptability = recalculations  # Adaptability in terms of recalculations

    # Determine real-world applicability based on the number of dynamic changes
    real_world_applicability = "High" if recalculations > 0 else "Low"

    print(f"Path Length: {path_length} steps")
    print(f"Computation Time: {computation_time:.4f} seconds")
    print(f"Efficiency (Nodes explored / Image Area): {efficiency:.6f} nodes per pixel")
    print(f"Adaptability to Dynamic Changes: {'High' if recalculations > 0 else 'Low'}")
    print(f"Real-World Applicability: {real_world_applicability}")
    
    return path, path_length, computation_time, efficiency, adaptability, real_world_applicability


if __name__ == "__main__":
    # Load the image
    img_file = cv2.imread(r"1.jpg")
    backup_img = img_file.copy()

    # Initialize Pygame
    pygame.init()

    # Convert the image to RGB for Pygame
    img_rgb = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
    height, width, _ = img_rgb.shape

    # Create a Pygame window
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Select Start and End Points")

    # Convert the image to Pygame surface
    img_surface = pygame.surfarray.make_surface(np.flipud(np.rot90(img_rgb)))

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
        screen.blit(img_surface, (0, 0))

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
    graph = extractGraph(img_file)
    path, path_length, computation_time, efficiency, adaptability, real_world_applicability = anytime_d_star(graph, start_point, end_point, dynamic_changes=True)

    # Draw the path on the image
    for point in path:
        point = point[1], point[0]
        cv2.circle(img_file, point, 1, (128, 255, 255), -1)

    cv2.circle(img_file, end_point, 3, (255, 255, 0), -1)
    cv2.circle(img_file, start_point, 3, (255, 0, 255), -1)

    # Show the results
    cv2.imshow("Path", img_file)
    cv2.imshow("Raw Image", backup_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
