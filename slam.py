import pybullet as p
import numpy as np
from queue import PriorityQueue
import time
import cv2
from pathplanner import AStarPlanner

class AutonomousNavigator:
    def __init__(self, robot_id, sensor_manager):
        self.robot_id = robot_id
        self.sensor_manager = sensor_manager
        
        # SLAM parameters
        self.cell_size = 0.5  # meters per cell
        self.map_size = 100   # grid cells (50x50m area)
        self.map_center = self.map_size // 2
        self.min_obstacle_dist = 1.0  # meters
        
        # Map and SLAM state
        self.occupancy_grid = np.zeros((self.map_size, self.map_size))  # 0=unknown, -1=free, 1=occupied
        self.visited = set()
        self.frontiers = []

    def get_robot_pose(self):
        """Get robot's pose in world and grid coordinates."""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        x, y = pos[0], pos[1]
        grid_x = int(x / self.cell_size) + self.map_center
        grid_y = int(y / self.cell_size) + self.map_center
        return (x, y), (grid_x, grid_y), orn

    def update_map(self):
        """Update occupancy map using LiDAR data."""
        (x, y), (gx, gy), _ = self.get_robot_pose()
        self.visited.add((gx, gy))

        lidar_data = self.sensor_manager.get_lidar_data()
        if lidar_data is None:
            return

        ranges, angles, _ = lidar_data

        for r, a in zip(ranges, angles):
            if r < self.sensor_manager.lidar_range:
                wx = x + r * np.cos(a)
                wy = y + r * np.sin(a)
                gx1 = int(wx / self.cell_size) + self.map_center
                gy1 = int(wy / self.cell_size) + self.map_center

                if 0 <= gx1 < self.map_size and 0 <= gy1 < self.map_size:
                    self.occupancy_grid[gy1, gx1] = 1
                    self._mark_line_as_free(gx, gy, gx1, gy1)

        self._update_frontiers()

    def _mark_line_as_free(self, x0, y0, x1, y1):
        """Bresenham's line algorithm to mark free space."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        err = dx - dy

        while x != x1 or y != y1:
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                self.occupancy_grid[y, x] = -1
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _update_frontiers(self):
        """Detect unexplored frontiers."""
        self.frontiers.clear()
        for y in range(1, self.map_size - 1):
            for x in range(1, self.map_size - 1):
                if self.occupancy_grid[y, x] == -1:
                    neighbors = self.occupancy_grid[y - 1:y + 2, x - 1:x + 2]
                    if 0 in neighbors:
                        self.frontiers.append((x, y))

    def get_next_frontier(self):
        """Return the closest unexplored frontier."""
        (_, _), (gx, gy), _ = self.get_robot_pose()
        min_dist = float('inf')
        nearest = None
        for fx, fy in self.frontiers:
            if (fx, fy) not in self.visited:
                d = np.hypot(fx - gx, fy - gy)
                if d < min_dist:
                    min_dist = d
                    nearest = (fx, fy)
        if nearest:
            wx = (nearest[0] - self.map_center) * self.cell_size
            wy = (nearest[1] - self.map_center) * self.cell_size
            return (wx, wy)
        return None

    def visualize_map(self):
        """Visualize SLAM map in OpenCV."""
        vis_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        vis_map[self.occupancy_grid == 1] = [0, 0, 255]
        vis_map[self.occupancy_grid == -1] = [255, 255, 255]
        vis_map[self.occupancy_grid == 0] = [50, 50, 50]

        for vx, vy in self.visited:
            vis_map[vy, vx] = [0, 255, 0]

        for fx, fy in self.frontiers:
            vis_map[fy, fx] = [255, 0, 255]

        _, (gx, gy), _ = self.get_robot_pose()
        if 0 <= gx < self.map_size and 0 <= gy < self.map_size:
            cv2.circle(vis_map, (gx, gy), 3, (0, 255, 255), -1)

        vis_map = cv2.resize(vis_map, (self.map_size * 2, self.map_size * 2))
        cv2.imshow("Simulated SLAM Map", vis_map)
        cv2.waitKey(1)

    def step(self):
        """Run one SLAM update step."""
        self.update_map()
        self.visualize_map()
