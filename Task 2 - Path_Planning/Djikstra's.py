import cv2
import numpy as np
import pygame
import heapq
import time

# Utility to measure memory usage (optional, requires psutil)
try:
    import psutil
    MEMORY_TRACKING = True
except ImportError:
    MEMORY_TRACKING = False


def extractGraph(input):
    kernel = np.ones((3, 3), np.uint8)
    img = input.copy()
    erosion = cv2.erode(img, kernel, iterations=2)

    GRAY_img = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)

    (thresh, BW_img) = cv2.threshold(GRAY_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return BW_img


def dijkstra(graph, start, end):
    start = start[1], start[0]
    end = end[1], end[0]

    height, width = graph.shape
    distances = np.full((height, width), np.inf)
    parentPointerX = np.ones((height, width)) * -1
    parentPointerY = np.ones((height, width)) * -1

    def isWall(graph, point):
        if graph[point[0], point[1]] < 50:
            return True
        return False

    def setParent(Target, point):
        parentPointerX[Target[0]][Target[1]] = point[0]
        parentPointerY[Target[0]][Target[1]] = point[1]

    def getParent(point):
        x, y = parentPointerX[point[0]][point[1]], parentPointerY[point[0]][point[1]]
        return x, y

    pq = []  # Priority queue (min-heap)
    heapq.heappush(pq, (0, start))
    distances[start[0]][start[1]] = 0
    setParent(start, (-10, -10))

    TotalNodes = 0
    FOUND_DESTINATION = False

    start_time = time.time()

    if MEMORY_TRACKING:
        process = psutil.Process()
        initial_memory = process.memory_info().rss

    while pq:
        current_distance, current_point = heapq.heappop(pq)
        x, y = current_point

        if (x, y) == end:
            FOUND_DESTINATION = True
            break

        neighbours = [(x, y - 1), (x - 1, y), (x, y + 1), (x + 1, y),
                      (x - 1, y - 1), (x + 1, y + 1), (x - 1, y + 1), (x + 1, y - 1)]

        for neighbour in neighbours:
            tx, ty = neighbour
            if tx < 5 or ty < 5 or tx > (height - 5) or ty > (width - 5):
                continue
            if isWall(graph, neighbour):
                continue

            new_distance = current_distance + 1  # All edges have equal weight (1)
            if new_distance < distances[tx][ty]:
                distances[tx][ty] = new_distance
                heapq.heappush(pq, (new_distance, neighbour))
                setParent((tx, ty), (x, y))

        TotalNodes += 1

    end_time = time.time()
    computation_time = end_time - start_time

    if MEMORY_TRACKING:
        final_memory = process.memory_info().rss
        memory_used = final_memory - initial_memory
        print(f"Memory Usage: {memory_used / 1024:.2f} KB")

    print("Computation Finished")
    print(f"Computation Time: {computation_time:.2f} seconds")

    parent = end
    path = []
    while True:
        newParent = getParent(parent)
        x, y = newParent
        y = int(y)
        x = int(x)

        if parent[0] == -10:
            print("Path Found")
            break
        elif parent[0] == x and parent[1] == y:
            print("Path Not Possible")
            break

        parent = (x, y)
        path.append((x, y))

    print("Total Nodes Checked =", TotalNodes)
    print("Path Length:", len(path))
    return path, computation_time, TotalNodes


def simulateDynamicChanges(graph, obstacle_list):
    for obstacle in obstacle_list:
        x, y = obstacle
        graph[x, y] = 0  # Mark as wall
    return graph


if __name__ == "__main__":
    # Load the image
    imgFile = cv2.imread("1.jpg")
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

    # Extract the graph and simulate dynamic changes
    graph = extractGraph(imgFile)
    dynamic_obstacles = [(50, 50), (100, 100)]  # Example obstacles
    graph = simulateDynamicChanges(graph, dynamic_obstacles)

    # Compute the path
    path, computation_time, total_nodes = dijkstra(graph, start_point, end_point)

    # Draw the path on the image
    for point in path:
        point = int(point[1]), int(point[0])
        cv2.circle(imgFile, point, 2, (128, 255, 255), -1)  # Ensure visibility

    cv2.circle(imgFile, (int(end_point[0]), int(end_point[1])), 5, (255, 255, 0), -1)  # Yellow for end
    cv2.circle(imgFile, (int(start_point[0]), int(start_point[1])), 5, (255, 0, 255), -1)  # Magenta for start

    # Show the results
    cv2.imshow("Path", imgFile)
    cv2.imshow("Raw Image", backupImg)
    print("Real-World Applicability Metrics:")
    print(f"Computation Time: {computation_time:.2f} seconds")
    print(f"Total Nodes Checked: {total_nodes}")
    print(f"Path Length: {len(path)}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
