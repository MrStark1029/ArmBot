import pybullet as p
import numpy as np
import cv2
import math

class TinySLAM:
    def __init__(self, robot_id, sensor_manager):
        self.robot_id = robot_id
        self.sensor_manager = sensor_manager
        
        # TinySLAM parameters
        self.map_size_m = 50.0  # meters
        self.map_resolution = 0.1  # meters per pixel
        self.map_size_px = int(self.map_size_m / self.map_resolution)
        self.map_center = self.map_size_px // 2
        
        # Add cell_size for compatibility with path planner
        self.cell_size = self.map_resolution
        self.map_size = self.map_size_px  # For compatibility
        
        # Occupancy grid: 0.5=unknown, 0=free, 1=occupied
        self.occupancy_grid = np.full((self.map_size_px, self.map_size_px), 0.5, dtype=np.float32)
        
        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        
        # Previous pose for odometry
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_theta = 0.0
        
        # TinySLAM specific parameters
        self.hole_width = 600.0  # Hole width in mm (for scaling)
        self.max_range = 5000.0  # Max range in mm
        self.free_thresh = 0.3
        self.occupied_thresh = 0.7
        
        # Frontiers for exploration
        self.frontiers = []
        
        # Add visited set for path planning compatibility
        self.visited = set()

    def get_occupancy_grid_for_planning(self):
        """Return occupancy grid in format expected by path planner."""
        # Convert probabilistic grid to binary for path planning
        binary_grid = np.zeros_like(self.occupancy_grid, dtype=np.int8)
        binary_grid[self.occupancy_grid > self.occupied_thresh] = 1  # Occupied
        binary_grid[self.occupancy_grid < self.free_thresh] = 0     # Free
        # Treat unknown areas as free for frontier exploration
        binary_grid[(self.occupancy_grid >= self.free_thresh) & 
                   (self.occupancy_grid <= self.occupied_thresh)] = 0  # Unknown treated as free
        return binary_grid

    def world_to_grid(self, world_x, world_y):
        """Convert world coordinates to grid coordinates for path planner compatibility."""
        return self.world_to_map(world_x, world_y)

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates for path planner compatibility."""
        return self.map_to_world(grid_x, grid_y)

    def get_robot_pose(self):
        """Get robot's current pose from PyBullet."""
        try:
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            euler = p.getEulerFromQuaternion(orn)
            return pos[0], pos[1], euler[2]  # x, y, yaw
        except Exception as e:
            print(f"Error getting robot pose: {e}")
            return 0.0, 0.0, 0.0

    def world_to_map(self, world_x, world_y):
        """Convert world coordinates to map pixel coordinates."""
        map_x = int((world_x / self.map_resolution) + self.map_center)
        map_y = int((world_y / self.map_resolution) + self.map_center)
        return map_x, map_y

    def map_to_world(self, map_x, map_y):
        """Convert map pixel coordinates to world coordinates."""
        world_x = (map_x - self.map_center) * self.map_resolution
        world_y = (map_y - self.map_center) * self.map_resolution
        return world_x, world_y

    def update_occupancy(self, scan_ranges, scan_angles):
        """Update occupancy grid using TinySLAM algorithm."""
        robot_map_x, robot_map_y = self.world_to_map(self.robot_x, self.robot_y)
        
        for i, (range_m, angle) in enumerate(zip(scan_ranges, scan_angles)):
            if range_m >= self.sensor_manager.lidar_max_range:
                continue
                
            # Convert to mm for TinySLAM calculations
            range_mm = range_m * 1000
            
            # Calculate end point
            end_x = self.robot_x + range_m * math.cos(self.robot_theta + angle)
            end_y = self.robot_y + range_m * math.sin(self.robot_theta + angle)
            end_map_x, end_map_y = self.world_to_map(end_x, end_y)
            
            # Mark obstacle at end point
            if (0 <= end_map_x < self.map_size_px and 0 <= end_map_y < self.map_size_px):
                self.occupancy_grid[end_map_y, end_map_x] = min(1.0, 
                    self.occupancy_grid[end_map_y, end_map_x] + 0.3)
            
            # Mark free space along the ray using Bresenham's algorithm
            self._mark_ray_free(robot_map_x, robot_map_y, end_map_x, end_map_y)

    def _mark_ray_free(self, x0, y0, x1, y1):
        """Mark cells along a ray as free space using Bresenham's line algorithm."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        err = dx - dy

        while True:
            if (0 <= x < self.map_size_px and 0 <= y < self.map_size_px):
                self.occupancy_grid[y, x] = max(0.0, self.occupancy_grid[y, x] - 0.1)
            
            if x == x1 and y == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def detect_frontiers(self):
        """Detect frontier cells (boundaries between known free and unknown space)."""
        self.frontiers = []
        
        # Create binary maps
        free_map = (self.occupancy_grid < self.free_thresh).astype(np.uint8)
        unknown_map = (np.abs(self.occupancy_grid - 0.5) < 0.1).astype(np.uint8)
        
        # Find frontiers using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        free_dilated = cv2.dilate(free_map, kernel, iterations=1)
        frontier_candidates = cv2.bitwise_and(free_dilated, unknown_map)
        
        # Find contours of frontier regions
        contours, _ = cv2.findContours(frontier_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        robot_map_x, robot_map_y = self.world_to_map(self.robot_x, self.robot_y)
        
        for contour in contours:
            if cv2.contourArea(contour) > 5:  # Minimum frontier size
                # Get centroid of frontier
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Convert to world coordinates
                    world_x, world_y = self.map_to_world(cx, cy)
                    distance = math.sqrt((cx - robot_map_x)**2 + (cy - robot_map_y)**2)
                    
                    self.frontiers.append((world_x, world_y, distance))
        
        # Sort frontiers by distance
        self.frontiers.sort(key=lambda f: f[2])

    def get_next_frontier(self):
        """Get the next frontier to explore."""
        if not self.frontiers:
            return None
        
        # Return closest frontier
        for fx, fy, dist in self.frontiers:
            # Check if frontier is still valid (not already explored)
            map_x, map_y = self.world_to_map(fx, fy)
            if (0 <= map_x < self.map_size_px and 0 <= map_y < self.map_size_px):
                if self.occupancy_grid[map_y, map_x] > 0.4:  # Still unknown
                    return (fx, fy)
        
        return None

    def update_robot_pose(self):
        """Update robot pose from sensors."""
        current_x, current_y, current_theta = self.get_robot_pose()
        
        # Simple odometry update (in a real system, you'd use scan matching)
        self.robot_x = current_x
        self.robot_y = current_y
        self.robot_theta = current_theta

    def update_slam(self):
        """Main SLAM update step."""
        try:
            # Update robot pose
            self.update_robot_pose()
            
            # Get LiDAR data
            lidar_data = self.sensor_manager.get_lidar_data()
            if lidar_data is None or len(lidar_data) < 2:
                return
            
            ranges = np.asarray(lidar_data[0])
            angles = np.asarray(lidar_data[1])
            
            if len(ranges) != len(angles):
                return
            
            # Update occupancy grid
            self.update_occupancy(ranges, angles)
            
            # Detect frontiers
            self.detect_frontiers()
            
        except Exception as e:
            print(f"Error in SLAM update: {e}")

    def visualize_map(self, planned_path=None, current_waypoint_index=0):
        """Visualize the SLAM map with optional path overlay."""
        try:
            # Create visualization image
            vis_map = np.zeros((self.map_size_px, self.map_size_px, 3), dtype=np.uint8)
            
            # Color the map based on occupancy probabilities
            for y in range(self.map_size_px):
                for x in range(self.map_size_px):
                    prob = self.occupancy_grid[y, x]
                    if prob < self.free_thresh:
                        vis_map[y, x] = [255, 255, 255]  # White for free
                    elif prob > self.occupied_thresh:
                        vis_map[y, x] = [0, 0, 0]  # Black for occupied
                    else:
                        gray_val = int(255 * (1 - prob))
                        vis_map[y, x] = [gray_val, gray_val, gray_val]  # Gray for unknown
            
            # Draw planned path if provided
            if planned_path and len(planned_path) > 1:
                path_points = []
                for world_x, world_y in planned_path:
                    map_x, map_y = self.world_to_map(world_x, world_y)
                    if 0 <= map_x < self.map_size_px and 0 <= map_y < self.map_size_px:
                        path_points.append((map_x, map_y))
                
                # Draw path line
                for i in range(len(path_points) - 1):
                    # Use different colors for completed vs remaining path
                    color = (128, 128, 128) if i < current_waypoint_index else (0, 255, 255)  # Gray for completed, cyan for remaining
                    cv2.line(vis_map, path_points[i], path_points[i + 1], color, 2)
                
                # Mark waypoints
                for i, (px, py) in enumerate(path_points):
                    if i == 0:
                        cv2.circle(vis_map, (px, py), 3, (0, 255, 0), -1)  # Green start
                    elif i == len(path_points) - 1:
                        cv2.circle(vis_map, (px, py), 4, (255, 0, 0), -1)  # Blue end
                    elif i == current_waypoint_index:
                        cv2.circle(vis_map, (px, py), 3, (255, 255, 0), -1)  # Yellow for current target waypoint
                    else:
                        color = (128, 128, 128) if i < current_waypoint_index else (0, 255, 255)
                        cv2.circle(vis_map, (px, py), 1, color, -1)  # Gray for completed, cyan for remaining
            
            # Mark robot position
            robot_map_x, robot_map_y = self.world_to_map(self.robot_x, self.robot_y)
            if (0 <= robot_map_x < self.map_size_px and 0 <= robot_map_y < self.map_size_px):
                cv2.circle(vis_map, (robot_map_x, robot_map_y), 4, (0, 0, 255), -1)  # Red circle
                
                # Draw robot orientation
                end_x = robot_map_x + int(15 * math.cos(self.robot_theta))
                end_y = robot_map_y + int(15 * math.sin(self.robot_theta))
                cv2.arrowedLine(vis_map, (robot_map_x, robot_map_y), (end_x, end_y), (0, 0, 255), 2)
            
            # Mark frontiers
            for fx, fy, _ in self.frontiers[:10]:  # Show top 10 frontiers
                map_x, map_y = self.world_to_map(fx, fy)
                if (0 <= map_x < self.map_size_px and 0 <= map_y < self.map_size_px):
                    cv2.circle(vis_map, (map_x, map_y), 2, (255, 0, 255), -1)  # Magenta for frontiers
            
            # Highlight next frontier
            next_frontier = self.get_next_frontier()
            if next_frontier:
                map_x, map_y = self.world_to_map(next_frontier[0], next_frontier[1])
                if (0 <= map_x < self.map_size_px and 0 <= map_y < self.map_size_px):
                    cv2.circle(vis_map, (map_x, map_y), 6, (0, 255, 0), 2)  # Green circle for next target
            
            # Resize for better visibility
            vis_map_resized = cv2.resize(vis_map, (600, 600))
            
            # Add text information
            cv2.putText(vis_map_resized, f"Frontiers: {len(self.frontiers)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_map_resized, f"Pos: ({self.robot_x:.1f}, {self.robot_y:.1f})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if planned_path:
                cv2.putText(vis_map_resized, f"Path: {len(planned_path)} waypoints", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("TinySLAM Map", vis_map_resized)
            cv2.waitKey(1)
            
        except Exception as e:
            print(f"Error visualizing map: {e}")

    def step(self, planned_path=None, current_waypoint_index=0):
        """Run one SLAM step with optional path visualization."""
        self.update_slam()
        self.visualize_map(planned_path, current_waypoint_index)
