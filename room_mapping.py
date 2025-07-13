import numpy as np
import cv2

class RoomMapper:
    def __init__(self, lidar_range=12.0, lidar_resolution=720):
        self.lidar_range = lidar_range
        self.lidar_resolution = lidar_resolution
        self.map_resolution = 0.05  # 5cm per pixel
        self.map_size = int(2 * lidar_range / self.map_resolution)  # Square map
        self.occupancy_map = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
        self.robot_positions = []
    
    def update_map(self, ranges, angles, robot_pose):
        """Update occupancy map with new LiDAR data"""
        # Store robot position for trajectory
        robot_map_pos = self._world_to_map(robot_pose[:2])
        self.robot_positions.append(robot_map_pos)
        
        # Convert LiDAR readings to map coordinates
        for distance, angle in zip(ranges, angles):
            if distance < self.lidar_range:
                # Calculate world coordinates
                x = distance * np.cos(angle + robot_pose[2]) + robot_pose[0]
                y = distance * np.sin(angle + robot_pose[2]) + robot_pose[1]
                
                # Convert to map coordinates
                map_x, map_y = self._world_to_map((x, y))
                
                if 0 <= map_x < self.map_size and 0 <= map_y < self.map_size:
                    self.occupancy_map[map_y, map_x] = 255
    
    def get_obstacle_map(self, ranges, angles):
        """Get obstacle points from LiDAR data"""
        obstacle_points = []
        for distance, angle in zip(ranges, angles):
            if distance < self.lidar_range * 0.95:  # Ignore max range readings
                x = distance * np.cos(angle)
                y = distance * np.sin(angle)
                obstacle_points.append((x, y))
        
        return np.array(obstacle_points) if obstacle_points else None
    
    def visualize_map(self, current_pose=None):
        """Generate visualization of the occupancy map"""
        # Create RGB visualization
        vis_map = cv2.cvtColor(self.occupancy_map, cv2.COLOR_GRAY2BGR)
        
        # Draw robot trajectory
        if len(self.robot_positions) > 1:
            positions = np.array(self.robot_positions)
            cv2.polylines(vis_map, [positions], False, (0, 255, 0), 1)
        
        # Draw current robot position
        if current_pose is not None:
            map_x, map_y = self._world_to_map(current_pose[:2])
            if 0 <= map_x < self.map_size and 0 <= map_y < self.map_size:
                cv2.circle(vis_map, (int(map_x), int(map_y)), 3, (0, 0, 255), -1)
                # Draw robot orientation
                end_x = map_x + 10 * np.cos(current_pose[2])
                end_y = map_y + 10 * np.sin(current_pose[2])
                cv2.line(vis_map, (int(map_x), int(map_y)), 
                        (int(end_x), int(end_y)), (255, 0, 0), 2)
        
        return vis_map
    
    def _world_to_map(self, world_coord):
        """Convert world coordinates to map coordinates"""
        map_coord_x = int((world_coord[0] + self.lidar_range) / self.map_resolution)
        map_coord_y = int((world_coord[1] + self.lidar_range) / self.map_resolution)
        return map_coord_x, map_coord_y
    
    def _map_to_world(self, map_coord):
        """Convert map coordinates to world coordinates"""
        world_x = map_coord[0] * self.map_resolution - self.lidar_range
        world_y = map_coord[1] * self.map_resolution - self.lidar_range
        return world_x, world_y
