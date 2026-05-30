import numpy as np
import matplotlib.pyplot as plt
import heapq
from dubins_utils import dubins_path

class MotionPlanner:
    def __init__(self, start, goal, obstacles: list , grid_size=10, obstacle_radius=0.5, robot_radius=0.3, wheelbase=0.5, max_steer_deg=35):
        self.start = np.array(start) # x, y, theta
        self.goal = np.array(goal) # x, y, theta
        self.obstacles = obstacles 

        self.obstacle_radius = obstacle_radius
        self.robot_radius = robot_radius

        self.grid_size = grid_size
        margin = robot_radius + obstacle_radius
        self.x_min, self.x_max = -grid_size/2 + margin, grid_size/2 - margin
        self.y_min, self.y_max = -grid_size/2 + margin, grid_size/2 - margin

        # Car-like param:
        self.wheelbase = wheelbase
        self.max_steer = np.radians(max_steer_deg)
        self.min_turn_radius = wheelbase/np.tan(self.max_steer)


    def sample_random_points(self, num_points):
        points = []
        attempts = 0
        max_attempts = num_points * 10
        while len(points) < num_points and attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(self.x_min, self.x_max)
            y = np.random.uniform(self.y_min, self.y_max)
            if not self.is_collision(np.array([x, y])):
                points.append((x, y))
        return points

    def path_is_free(self, path):
        if not path:
            return False
        for point in path:
            if self.is_collision(point):
                return False
        return True

    def is_collision(self, point):
        point = np.array(point[:2], dtype=float)

        for wall in self.obstacles:
            p1 = np.array(wall["p1"], dtype=float)
            p2 = np.array(wall["p2"], dtype=float)
            wall_width = wall["wall_width"]
            endpoint_radius = wall.get("endpoint_radius", wall_width / 2)
            segment_clearance = wall_width / 2 + self.robot_radius
            endpoint_clearance = endpoint_radius + self.robot_radius

            # Endpoint check — use same clearance as segment
            if np.linalg.norm(point - p1) <= endpoint_clearance:
                return True
            if np.linalg.norm(point - p2) <= endpoint_clearance:
                return True

            wall_vec = p2 - p1
            wall_length = np.linalg.norm(wall_vec)
            if wall_length < 1e-6:
                continue

            t = np.dot(point - p1, wall_vec) / wall_length**2
            if 0 <= t <= 1:
                closest_point = p1 + t * wall_vec
                if np.linalg.norm(point - closest_point) <= segment_clearance:
                    return True

        return False
    
    def requried_radius(self, p1, p2, theta1):
        d = np.linalg.norm(p2 - p1)
        if d < 1e-6:
            return np.inf
        
        # Angle from p1 to p2 in local frame
        alpha = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) - theta1
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        if abs(np.sin(alpha)) < 1e-6:
            return np.inf
        
        return abs(d / (2 * np.sin(alpha)))
    
    def local_planner(self, start, goal, num_steps=30):
        start, goal = np.array(start[:2], dtype=float), np.array(goal[:2], dtype=float)

        seg = goal - start
        theta = np.arctan2(seg[1], seg[0])

        rad = self.requried_radius(start, goal, theta)
        if rad != np.inf and rad < self.min_turn_radius:
            return []

        path = []
        for k in range(num_steps + 1):
            t = k/num_steps
            point = start + (goal-start)*t
            path.append(point)
        return path
    
    def aStarSearch(self, start, edges, nodes, goal):
        """Search the node that has the lowest combined cost and heuristic first."""
        # Base from task 4, homework 4
        # Astar: f(n) = g(n) + h(n) | g(n) = cost, h(n) = heuristic distance
        frontier = []

        heapq.heappush(frontier, (self.heuristic(start, goal, nodes), start))

        prev_state = {}
        path = []
        best_g = {start: 0.0}
        print(frontier)

        while frontier:
            _, state = heapq.heappop(frontier)

            if state == goal:
                print("Goal found")

                path = [state]
                while state in prev_state:
                    state = prev_state[state]
                    path.append(state)

                path.reverse()
                return path
            
            for v in edges.get(state, []):
                new_g = best_g[state] + np.linalg.norm(nodes[state] - nodes[v]) # euclidean distance as cost

                if new_g < best_g.get(v, float('inf')):
                    best_g[v] = new_g
                    h = self.heuristic(v, goal, nodes)
                    f = new_g + h
                    heapq.heappush(frontier, (f, v))
                    prev_state[v] = state
        return []

    def heuristic(self, a, b, nodes):
        """Euclidean distance"""
        return np.linalg.norm((nodes[a] - nodes[b])) # euclidean

    
    def prm_roadmap(self, num_samples=100, k=5):
        # Returns a roadmap
        # N random samples, k nearest neighbours
        
        # Retrieve samples
        samples = self.sample_random_points(num_samples)
        #samples.append(self.start[:2])
        samples.append(tuple(self.start[:2]))
        samples.append(tuple(self.goal[:2]))

        nodes = [np.array(sample, dtype=float) for sample in samples]
        N = len(nodes)

        edges = {}

        # Connect neighbours
        for i in range(N):
            qi = nodes[i]

            distances = []
            for j in range(N):
                if i != j:
                    qj =  nodes[j]
                    dist = np.linalg.norm(qi - qj)
                    distances.append((dist, j))
            distances.sort()

            neighbours = [j for _, j in distances[:k]] # k nearest neighbours

            for j in neighbours:
                qj = nodes[j]

                dist = np.linalg.norm(np.array(qi) - np.array(qj))
                num_steps = max(20, int(dist / (self.robot_radius * 0.5)))

                local_path = self.local_planner(qi, qj, num_steps=num_steps)
                if self.path_is_free(local_path):
                    edges.setdefault(i, []).append(j)
                    edges.setdefault(j, []).append(i)
        return nodes, edges
    
    def global_planner(self, N=100, k=5):
        nodes, edges = self.prm_roadmap(N, k)

        start_idx = len(nodes) - 2
        goal_idx = len(nodes) - 1

        # A* search on roadmap
        path_indicies = self.aStarSearch(start_idx, edges, nodes, goal_idx)

        if not path_indicies:
            print("No path found")
            return []
        
        path = [nodes[i] for i in path_indicies]
        path.pop()
        goal_pose = np.array([self.goal[0], self.goal[1], self.goal[2]], dtype=float)
        path.append(goal_pose)
        return path


