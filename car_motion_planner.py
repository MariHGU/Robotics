import numpy as np
import heapq
from dubins_utils import dubins_path

class CarMotionPlanner:
    def __init__(self, start, goal, obstacles, grid_size=10,
                 obstacle_radius=0.5, robot_radius=0.3,
                 wheelbase=0.5, max_steer_deg=35):
        self.start = np.array(start)       # [x, y, theta]
        self.goal  = np.array(goal)
        self.obstacles = obstacles
        self.robot_radius = robot_radius
        self.grid_size = grid_size

        self.wheelbase   = wheelbase
        self.max_steer   = np.radians(max_steer_deg)
        self.min_turn_radius = wheelbase / np.tan(self.max_steer)

        margin = robot_radius + obstacle_radius
        self.x_min = -grid_size/2 + margin
        self.x_max =  grid_size/2 - margin
        self.y_min = -grid_size/2 + margin
        self.y_max =  grid_size/2 - margin

    def is_collision(self, point):
        point = np.array(point[:2], dtype=float)
        for wall in self.obstacles:
            p1 = np.array(wall["p1"], dtype=float)
            p2 = np.array(wall["p2"], dtype=float)
            wall_width      = wall["wall_width"]
            endpoint_radius = wall.get("endpoint_radius", wall_width / 2)
            seg_clear  = wall_width / 2 + self.robot_radius
            end_clear  = endpoint_radius + self.robot_radius

            if np.linalg.norm(point - p1) <= end_clear: return True
            if np.linalg.norm(point - p2) <= end_clear: return True

            wall_vec    = p2 - p1
            wall_length = np.linalg.norm(wall_vec)
            if wall_length < 1e-6: continue

            t = np.dot(point - p1, wall_vec) / wall_length**2
            if 0 <= t <= 1:
                if np.linalg.norm(point - (p1 + t * wall_vec)) <= seg_clear:
                    return True
        return False

    def path_is_free(self, path):
        if not path:
            return False
        return all(not self.is_collision(p) for p in path)

    def local_planner(self, start, goal, step_size=None):
        """
        Dubins path between two poses [x, y, theta].
        start/goal must be 3-element arrays.
        Returns a list of [x, y] waypoints, or [] if in collision.
        """
        if step_size is None:
            step_size = self.robot_radius * 0.4

        q0 = (float(start[0]), float(start[1]), float(start[2]))
        q1 = (float(goal[0]),  float(goal[1]),  float(goal[2]))

        return dubins_path(q0, q1, self.min_turn_radius, step_size)

    def sample_random_configs(self, num_points):
        configs = []
        attempts = 0
        while len(configs) < num_points and attempts < num_points * 10:
            attempts += 1
            x = np.random.uniform(self.x_min, self.x_max)
            y = np.random.uniform(self.y_min, self.y_max)
            if not self.is_collision(np.array([x, y])):
                theta = np.random.uniform(-np.pi, np.pi)
                configs.append(np.array([x, y, theta]))
        return configs

    def prm_roadmap(self, num_samples=200, k=8):
        nodes = self.sample_random_configs(num_samples)
        nodes.append(self.start.copy())   # index -2
        nodes.append(self.goal.copy())    # index -1
        N = len(nodes)
        edges = {}

        for i in range(N):
            # Nearest neighbours by position only
            dists = sorted(
                [(np.linalg.norm(nodes[i][:2] - nodes[j][:2]), j)
                 for j in range(N) if j != i]
            )
            for _, j in dists[:k]:
                if j in edges.get(i, []):
                    continue  # already connected
                path = self.local_planner(nodes[i], nodes[j])
                if self.path_is_free(path):
                    edges.setdefault(i, []).append(j)
                    edges.setdefault(j, []).append(i)
        return nodes, edges

    def heuristic(self, a, b, nodes):
        return np.linalg.norm(nodes[a][:2] - nodes[b][:2])

    def aStarSearch(self, start, edges, nodes, goal):
        frontier = []
        heapq.heappush(frontier, (self.heuristic(start, goal, nodes), start))
        prev_state = {}
        best_g = {start: 0.0}

        while frontier:
            _, state = heapq.heappop(frontier)
            if state == goal:
                path = [state]
                while state in prev_state:
                    state = prev_state[state]
                    path.append(state)
                return list(reversed(path))
            for v in edges.get(state, []):
                new_g = best_g[state] + np.linalg.norm(nodes[state][:2] - nodes[v][:2])
                if new_g < best_g.get(v, float('inf')):
                    best_g[v] = new_g
                    heapq.heappush(frontier, (new_g + self.heuristic(v, goal, nodes), v))
                    prev_state[v] = state
        return []

    def global_planner(self, N=300, k=8):
        nodes, edges = self.prm_roadmap(N, k)
        start_idx = len(nodes) - 2
        goal_idx  = len(nodes) - 1

        print(f"Start neighbours: {edges.get(start_idx, [])}")
        print(f"Goal  neighbours: {edges.get(goal_idx,  [])}")

        indices = self.aStarSearch(start_idx, edges, nodes, goal_idx)
        if not indices:
            print("No path found")
            return []

        # Return full [x, y, theta] poses along the path
        path = [nodes[i] for i in indices]
        path[-1] = self.goal.copy()   # enforce exact goal pose
        return path