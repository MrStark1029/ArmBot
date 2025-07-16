# astar_path_planner.py

import heapq
import numpy as np

class AStarPlanner:
    def __init__(self, occupancy_grid):
        self.grid = occupancy_grid
        self.grid_h, self.grid_w = occupancy_grid.shape
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def plan(self, start, goal):
        """Plan path using A* from start to goal on occupancy grid."""
        try:
            if not self._is_free(goal) or not self._is_free(start):
                print(f"Start {start} or goal {goal} is blocked.")
                return []

            if start == goal:
                return [start]

            open_set = []
            heapq.heappush(open_set, (0 + self._heuristic(start, goal), 0, start, [start]))
            visited = set()

            while open_set:
                _, cost, current, path = heapq.heappop(open_set)
                if current in visited:
                    continue
                visited.add(current)

                if current == goal:
                    return path

                for dx, dy in self.directions:
                    nx, ny = current[0] + dx, current[1] + dy
                    if self._is_free((nx, ny)) and (nx, ny) not in visited:
                        new_cost = cost + np.hypot(dx, dy)
                        priority = new_cost + self._heuristic((nx, ny), goal)
                        heapq.heappush(open_set, (priority, new_cost, (nx, ny), path + [(nx, ny)]))

            print("No path found.")
            return []
        except Exception as e:
            print(f"Path planning error: {e}")
            return []

    def _heuristic(self, a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def _is_free(self, cell):
        x, y = cell
        if 0 <= x < self.grid_w and 0 <= y < self.grid_h:
            return self.grid[y, x] != 1  # Not occupied
        return False
