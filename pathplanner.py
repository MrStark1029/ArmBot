import heapq
import numpy as np

class AStarPlanner:
    def __init__(self, slam_system):
        """Initialize with SLAM system."""
        self.slam = slam_system
        self.grid = slam_system.get_occupancy_grid_for_planning()
        self.grid_h, self.grid_w = self.grid.shape
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def update_grid(self):
        """Update grid from SLAM system."""
        self.grid = self.slam.get_occupancy_grid_for_planning()
        self.grid_h, self.grid_w = self.grid.shape

    def plan(self, start, goal):
        """Plan path using A* from start to goal on occupancy grid."""
        print(f"=== PATH PLANNING DEBUG ===")
        print(f"Start: {start} (type: {type(start)})")
        print(f"Goal: {goal} (type: {type(goal)})")
        
        try:
            self.update_grid()
            
            # Validate inputs
            if not hasattr(start, '__len__') or len(start) < 2:
                print(f"Invalid start: {start}")
                return []
            if not hasattr(goal, '__len__') or len(goal) < 2:
                print(f"Invalid goal: {goal}")
                return []
            
            # Convert to integers
            start = (int(start[0]), int(start[1]))
            goal = (int(goal[0]), int(goal[1]))
            print(f"Converted - Start: {start}, Goal: {goal}")
            
            # Check bounds
            if not (0 <= start[0] < self.grid_w and 0 <= start[1] < self.grid_h):
                print(f"Start {start} out of bounds (grid: {self.grid_w}x{self.grid_h})")
                return []
            
            if not (0 <= goal[0] < self.grid_w and 0 <= goal[1] < self.grid_h):
                print(f"Goal {goal} out of bounds (grid: {self.grid_w}x{self.grid_h})")
                return []
            
            # Check grid values at start and goal
            start_value = self.grid[start[1], start[0]]
            goal_value = self.grid[goal[1], goal[0]]
            print(f"Grid values - Start: {start_value}, Goal: {goal_value}")
            print(f"Start free: {self._is_free(start)}, Goal free: {self._is_free(goal)}")
            
            # Check surrounding area for debugging
            print("Start area (3x3):")
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    y, x = start[1] + dy, start[0] + dx
                    if 0 <= x < self.grid_w and 0 <= y < self.grid_h:
                        print(f"  ({x},{y}): {self.grid[y, x]}")
            
            if not self._is_free(start) or not self._is_free(goal):
                print(f"Start {start} or goal {goal} is blocked")
                return []

            if start == goal:
                return [start]

            # A* algorithm
            open_set = []
            heapq.heappush(open_set, (self._heuristic(start, goal), 0, start, [start]))
            visited = set()

            while open_set:
                _, cost, current, path = heapq.heappop(open_set)
                if current in visited:
                    continue
                visited.add(current)

                if current == goal:
                    print(f"Path found with {len(path)} waypoints")
                    return path

                for dx, dy in self.directions:
                    nx, ny = current[0] + dx, current[1] + dy
                    if self._is_free((nx, ny)) and (nx, ny) not in visited:
                        new_cost = cost + np.hypot(dx, dy)
                        priority = new_cost + self._heuristic((nx, ny), goal)
                        heapq.heappush(open_set, (priority, new_cost, (nx, ny), path + [(nx, ny)]))

            print("No path found")
            return []
            
        except Exception as e:
            print(f"Path planning error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _heuristic(self, a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def _is_free(self, cell):
        x, y = cell
        if 0 <= x < self.grid_w and 0 <= y < self.grid_h:
            return self.grid[y, x] == 0
        return False
