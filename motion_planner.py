import numpy as np
import matplotlib.pyplot as plt
import heapq

class MotionPlanner:
    def __init__(self, start, goal, obstacles: list , grid_size=10, resolution=0.1, obstacle_radius=0.5):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles 

        self.grid_size = grid_size
        self.x_min, self.x_max = -grid_size/2, grid_size/2
        self.y_min, self.y_max = -grid_size/2, grid_size/2

        self.resolution = resolution
        self.obstacle_radius = obstacle_radius

    def define_grid(self):
        grid_width = int((self.x_max - self.x_min) / self.resolution)
        grid_height = int((self.y_max - self.y_min) / self.resolution)

        grid = np.zeros((grid_height, grid_width))

        for gy in range(grid_height):
            for gx in range(grid_width):
                point = self.grid_to_world((gx, gy))

                if self.is_collision(point):
                    grid[gy, gx] = 1
        return grid
    
    def world_to_grid(self, point):
        x, y = point

        gx = int((x - self.x_min) / self.resolution)
        gy = int((y - self.y_min) / self.resolution)

        return gx, gy

    def grid_to_world(self, cell):
        gx, gy = cell

        x = self.x_min + (gx + 0.5) * self.resolution
        y = self.y_min + (gy + 0.5) * self.resolution

        return np.array([x, y])
    
    def plan(self, start, goal):
        start_grid = self.world_to_grid(start)
        goal_grid = self.world_to_grid(goal)

        grid = self.define_grid()

        path_grid = self.aStarSearch(grid, start_grid, goal_grid)

        path_world = [self.grid_to_world(cell) for cell in path_grid]

        return path_world

    def is_collision(self, point):
        point = np.array(point)

        # Check wall obstacles
        for wall in self.obstacles:
            p1 = wall["p1"]
            p2 = wall["p2"]
            wall_width = wall["wall_width"]

            if np.linalg.norm(point - p1) <= self.obstacle_radius or np.linalg.norm(point - p2) <= self.obstacle_radius:
                return True

            wall_vec = p2 - p1
            wall_length = np.linalg.norm(wall_vec)

            if wall_length == 0:
                continue

            # Projection parameter along wall
            t = np.dot(point - p1, wall_vec) / wall_length**2

            # Only check points whose closest point lies inside the wall segment
            if 0 <= t <= 1:
                closest_point = p1 + t * wall_vec
                distance_to_wall = np.linalg.norm(point - closest_point)

                if distance_to_wall <= wall_width / 2:
                    return True

        return False
    

    def aStarSearch(self, grid, start, goal, connectivity=8):
        """Search the node that has the lowest combined cost and heuristic first."""
        # from task 4, homework 4
        # Astar: f(n) = g(n) + h(n) | g(n) = cost, h(n) = heuristic distance
        # 4-connectivity
        if connectivity == 4:
            neighbours = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # 8-connectivity
        elif connectivity == 8:
            neighbours = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        else: 
            raise ValueError("Invalid connectivity. Valid options: 4, 8.")
        frontier = []

        #heapq.heappush(frontier, (0, start))
        heapq.heappush(frontier, (self.heuristic(start, goal, connectivity), start))

        prev_state = {}
        path = []
        best_g = {start: 0}
        print(frontier)

        while frontier:
            _, state = heapq.heappop(frontier)

            cx, cy = state

            if state == goal:
                print("Goal found")

                path = [state]
                while state in prev_state:
                    state = prev_state[state]
                    path.append(state)

                path.reverse()
                return path
            
            for i, j in neighbours:
                new_state = cx + i, cy + j
                # skip if out of bounds
                if not (0 <= new_state[0] < grid.shape[1] and 0 <= new_state[1] < grid.shape[0]):
                    continue
                # skip if obstacle
                if grid[new_state[1], new_state[0]] == 1:
                    continue

                if connectivity == 4:
                    # uniform cost
                    new_g = best_g[state] + 1
                
                else:
                    # Diagonal cost is slightly higher than orthogonal cost
                    new_g = best_g[state] + np.sqrt((state[0]-new_state[0])**2 + (state[1]-new_state[1])**2)

                if new_g < best_g.get(new_state, float('inf')):
                    best_g[new_state] = new_g
                    h = self.heuristic(new_state, goal, connectivity)
                    f = new_g + h
                    heapq.heappush(frontier, (f, new_state))
                    prev_state[new_state] = state


        return []

    def heuristic(self, a, b, connectivity=8):
        """Manhatten or Euclidean distance??"""
        x1, y1 = a
        x2, y2 = b
        if connectivity == 4:
            return np.abs(x1 - x2) + np.abs(y1 - y2) # manhattan
        if connectivity == 8:
            return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) # euclidean



    # def plan_path(self):
    #     path = [self.start]
    #     current_point = self.start

    #     while np.linalg.norm(current_point - self.goal) > 0.1:
    #         direction = self.goal - current_point
    #         direction /= np.linalg.norm(direction)  # Normalize the direction
    #         next_point = current_point + direction * 0.1  # Move in small steps

    #         if not self.is_collision(next_point):
    #             path.append(next_point)
    #             current_point = next_point
    #         else:
    #             # If there's a collision, try to find an alternative path
    #             angle = np.random.uniform(0, 2 * np.pi)
    #             next_point = current_point + np.array([np.cos(angle), np.sin(angle)]) * 0.1
    #             if not self.is_collision(next_point):
    #                 path.append(next_point)
    #                 current_point = next_point

    #     path.append(self.goal)
    #     return path

    def plot_path(self, path):
        plt.figure(figsize=(8, 8))
        plt.plot(path[:, 0], path[:, 1], marker='o')
        plt.scatter(self.start[0], self.start[1], color='green', label='Start')
        plt.scatter(self.goal[0], self.goal[1], color='red', label='Goal')
        for obs in self.obstacles:
            circle = plt.Circle(obs, 0.5, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
        plt.xlim(-1, 10)
        plt.ylim(-1, 10)
        plt.legend()
        plt.title('Motion Planning Path')
        plt.grid()
        plt.show()