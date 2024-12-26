import cv2
import numpy as np
import pygame
import time  # To track computation time
import random

def extractGraph(input):
    kernel = np.ones((3, 3), np.uint8)
    img = input.copy()
    erosion = cv2.erode(img, kernel, iterations=2)

    GRAY_img = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)

    (thresh, BW_img) = cv2.threshold(GRAY_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return BW_img

def BFS(graph, start, end, dynamic_changes=False):
    start = start[1], start[0]
    end = end[1], end[0]

    height, width = graph.shape
    levels = (np.zeros(shape=(height, width))) - 1
    parentPointerX = np.ones(shape=(height, width)) * -1
    parentPointerY = np.ones(shape=(height, width)) * -1

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

    Frontier = []
    Frontier.append(start)
    levels[start[0]][start[1]] = 0
    setParent(start, (-10, -10))

    TotalNodes = 0
    FOUND_DESTINATION = False
    start_time = time.time()  # Start time for computation

    while True:
        nextFrontier = []

        for x, y in Frontier:
            # Simulate dynamic changes (random obstacles) in the environment
            if dynamic_changes and random.random() < 0.1:  # 10% chance to introduce an obstacle
                tx, ty = random.randint(5, height-5), random.randint(5, width-5)
                graph[tx, ty] = 0  # Mark as obstacle

            neighbours = [(x, y - 1), (x - 1, y), (x, y + 1), (x + 1, y), (x - 1, y - 1), (x + 1, y + 1), (x - 1, y + 1), (x + 1, y - 1)]
            currentLevel = levels[x][y]

            for neighbour in neighbours:
                tx, ty = neighbour
                if tx < 5 or ty < 5 or tx > (height - 5) or ty > (width - 5):
                    pass
                else:
                    TotalNodes += 1
                    if isWall(graph, neighbour):
                        pass
                    elif levels[tx][ty] == -1:
                        levels[tx][ty] = currentLevel + 1
                        nextFrontier.append(neighbour)
                        setParent((tx, ty), (x, y))
                    elif levels[tx][ty] > currentLevel + 1:
                        levels[tx][ty] = currentLevel + 1
                        nextFrontier.append(neighbour)
                        setParent((tx, ty), (x, y))
                    else:
                        pass

                if x == end[0] and y == end[1]:
                    FOUND_DESTINATION = True
                    break

        del Frontier

        if len(nextFrontier) > 0 and not FOUND_DESTINATION:
            Frontier = nextFrontier.copy()
            del nextFrontier
        else:
            del nextFrontier
            break

    # Calculate path length
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

        del parent

        parent = (x, y)
        path.append((x, y))

    # End time for computation
    end_time = time.time()
    computation_time = end_time - start_time

    print("Total Nodes Checked = ", TotalNodes)
    print("Path Length = ", len(path), "steps")
    print("Computation Time = ", round(computation_time, 3), "seconds")

    # Calculate Efficiency: Nodes explored relative to image area
    image_area = height * width
    efficiency = TotalNodes / image_area
    print("Efficiency (Nodes explored / Image Area): ", round(efficiency, 6), "nodes per pixel")

    # Adaptability (for dynamic changes)
    adaptability = "High" if dynamic_changes else "Low"
    print("Adaptability to Dynamic Changes: ", adaptability)

    # Real-World Applicability (depends on path length and efficiency)
    if len(path) < 100:
        real_world_applicability = "High"
    elif len(path) < 300:
        real_world_applicability = "Medium"
    else:
        real_world_applicability = "Low"
    print("Real-World Applicability: ", real_world_applicability)

    return path


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
    path = BFS(graph, start_point, end_point, dynamic_changes=True)  # Enable dynamic changes

    # Draw the path on the image
    for point in path:
        point = point[1], point[0]
        cv2.circle(imgFile, point, 1, (128, 255, 255), -1)

    cv2.circle(imgFile, end_point, 3, (255, 255, 0), -1)
    cv2.circle(imgFile, start_point, 3, (255, 0, 255), -1)

    # Show the results
    cv2.imshow("Path", imgFile)
    cv2.imshow("Raw Image", backupImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
