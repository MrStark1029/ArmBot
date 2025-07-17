import pybullet as p
import numpy as np
import time
import cv2
import math

class LiDARNavigator:
    def __init__(self, robot_id, sensor_manager):
        self.robot_id = robot_id
        self.sensor_manager = sensor_manager
        
        # Robot control parameters
        self.forward_speed = 1.0
        self.turning_speed = 0.8
        self.safe_distance = 1.0  # meters - minimum distance to obstacles
        
        # Navigation state
        self.enabled = False
        self.running = False
        
        # Simple occupancy map for visualization
        self.map_size = 500  # pixels
        self.map_resolution = 0.1  # meters per pixel
        self.map_center = self.map_size // 2
        self.occupancy_map = np.ones((self.map_size, self.map_size), dtype=np.float32) * 0.5  # Unknown
        
        # Robot trail for visualization
        self.robot_trail = []
        self.max_trail_length = 1000
        
    def set_enabled(self, enabled):
        """Enable or disable navigation"""
        self.enabled = enabled
        if not enabled:
            self.stop()
            self.running = False
        else:
            self.running = True
            
    def is_enabled(self):
        return self.enabled
        
    def is_running(self):
        return self.running
        
    def update(self):
        """Main navigation loop"""
        if not self.enabled:
            return True
            
        try:
            # Get robot pose
            robot_pos, robot_quat = p.getBasePositionAndOrientation(self.robot_id)
            robot_euler = p.getEulerFromQuaternion(robot_quat)
            robot_x, robot_y = robot_pos[0], robot_pos[1]
            robot_theta = robot_euler[2]
            
            # Update trail
            self.robot_trail.append((robot_x, robot_y))
            if len(self.robot_trail) > self.max_trail_length:
                self.robot_trail.pop(0)
            
            # Get LiDAR data
            lidar_data = self.sensor_manager.get_lidar_data()
            ranges = lidar_data['ranges']
            angles = np.linspace(0, 2*np.pi, len(ranges), endpoint=False)
            
            # Update occupancy map with LiDAR data
            self.update_occupancy_map(robot_x, robot_y, robot_theta, ranges, angles)
            
            # Find navigation direction
            target_direction = self.find_navigation_direction(robot_x, robot_y, robot_theta, ranges, angles)
            
            if target_direction is not None:
                # Navigate towards target direction
                self.navigate_to_direction(robot_theta, target_direction)
            else:
                # No clear direction, turn to explore
                self.explore_turn()
                
            # Visualize
            self.visualize_navigation(robot_x, robot_y, robot_theta, ranges, angles, target_direction)
            
        except Exception as e:
            print(f"Navigation error: {e}")
            
        return True
        
    def update_occupancy_map(self, robot_x, robot_y, robot_theta, ranges, angles):
        """Update occupancy map with current LiDAR scan"""
        robot_map_x, robot_map_y = self.world_to_map(robot_x, robot_y)
        
        # Clear area around robot
        cv2.circle(self.occupancy_map, (robot_map_x, robot_map_y), 5, 0.1, -1)
        
        for i, (range_m, angle) in enumerate(zip(ranges, angles)):
            if range_m >= self.sensor_manager.lidar_max_range * 0.95:
                # Max range - mark as free up to max range
                range_m = self.sensor_manager.lidar_max_range * 0.9
                
            # Calculate end point
            world_angle = robot_theta + angle
            end_x = robot_x + range_m * math.cos(world_angle)
            end_y = robot_y + range_m * math.sin(world_angle)
            end_map_x, end_map_y = self.world_to_map(end_x, end_y)
            
            # Mark obstacle if within range
            if range_m < self.sensor_manager.lidar_max_range * 0.95:
                if (0 <= end_map_x < self.map_size and 0 <= end_map_y < self.map_size):
                    cv2.circle(self.occupancy_map, (end_map_x, end_map_y), 2, 1.0, -1)
            
            # Mark free space along ray
            if (0 <= end_map_x < self.map_size and 0 <= end_map_y < self.map_size):
                cv2.line(self.occupancy_map, (robot_map_x, robot_map_y), (end_map_x, end_map_y), 0.1, 1)
    
    def find_navigation_direction(self, robot_x, robot_y, robot_theta, ranges, angles):
        """Find the best direction to navigate based on LiDAR data"""
        # Analyze LiDAR data to find free space directions
        sector_size = len(ranges) // 12  # Divide into 12 sectors (30 degrees each)
        sector_scores = []
        
        for sector in range(12):
            start_idx = sector * sector_size
            end_idx = min((sector + 1) * sector_size, len(ranges))
            
            # Get ranges in this sector
            sector_ranges = ranges[start_idx:end_idx]
            sector_angles = angles[start_idx:end_idx]
            
            # Calculate score for this sector
            min_distance = np.min(sector_ranges)
            avg_distance = np.mean(sector_ranges)
            
            # Score based on distance and exploration potential
            if min_distance > self.safe_distance:
                # Safe to go in this direction
                score = avg_distance
                
                # Bonus for unexplored areas (check occupancy map)
                sector_angle = robot_theta + np.mean(sector_angles)
                check_distance = 3.0  # Look ahead 3 meters
                check_x = robot_x + check_distance * math.cos(sector_angle)
                check_y = robot_y + check_distance * math.sin(sector_angle)
                check_map_x, check_map_y = self.world_to_map(check_x, check_y)
                
                if (0 <= check_map_x < self.map_size and 0 <= check_map_y < self.map_size):
                    if self.occupancy_map[check_map_y, check_map_x] > 0.4:  # Unexplored
                        score += 2.0  # Bonus for unexplored areas
                        
            else:
                score = 0  # Not safe
                
            sector_scores.append((sector, score, sector_angle))
        
        # Find best sector
        sector_scores.sort(key=lambda x: x[1], reverse=True)
        
        if sector_scores[0][1] > 0:
            best_sector_angle = sector_scores[0][2]
            return best_sector_angle
        else:
            return None
    
    def navigate_to_direction(self, current_angle, target_angle):
        """Navigate robot towards target direction"""
        # Calculate angle difference
        angle_diff = target_angle - current_angle
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
            
        # Control logic
        if abs(angle_diff) > 0.2:  # Need to turn
            turn_speed = self.turning_speed * np.sign(angle_diff)
            turn_speed = max(-self.turning_speed, min(self.turning_speed, turn_speed))
            
            print(f"Turning: {np.degrees(angle_diff):.1f}° at speed {turn_speed:.2f}")
            p.setJointMotorControlArray(
                self.robot_id,
                [2, 4, 3, 5],
                p.VELOCITY_CONTROL,
                targetVelocities=[-turn_speed, -turn_speed, turn_speed, turn_speed]
            )
        else:
            # Move forward
            speed = self.forward_speed
            print(f"Moving forward at speed {speed:.2f}")
            p.setJointMotorControlArray(
                self.robot_id,
                [2, 4, 3, 5],
                p.VELOCITY_CONTROL,
                targetVelocities=[speed, speed, speed, speed]
            )
    
    def explore_turn(self):
        """Turn to explore when no clear direction is found"""
        print("No clear direction, exploring...")
        turn_speed = self.turning_speed * 0.5
        p.setJointMotorControlArray(
            self.robot_id,
            [2, 4, 3, 5],
            p.VELOCITY_CONTROL,
            targetVelocities=[-turn_speed, -turn_speed, turn_speed, turn_speed]
        )
    
    def world_to_map(self, world_x, world_y):
        """Convert world coordinates to map coordinates"""
        map_x = int(world_x / self.map_resolution + self.map_center)
        map_y = int(world_y / self.map_resolution + self.map_center)
        return map_x, map_y
        
    def map_to_world(self, map_x, map_y):
        """Convert map coordinates to world coordinates"""
        world_x = (map_x - self.map_center) * self.map_resolution
        world_y = (map_y - self.map_center) * self.map_resolution
        return world_x, world_y
    
    def visualize_navigation(self, robot_x, robot_y, robot_theta, ranges, angles, target_direction):
        """Visualize the navigation state"""
        # Create visualization
        vis_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        
        # Color occupancy map
        for y in range(self.map_size):
            for x in range(self.map_size):
                prob = self.occupancy_map[y, x]
                if prob < 0.2:
                    vis_map[y, x] = [255, 255, 255]  # White for free
                elif prob > 0.8:
                    vis_map[y, x] = [0, 0, 0]  # Black for occupied
                else:
                    gray = int(255 * (1 - prob))
                    vis_map[y, x] = [gray, gray, gray]  # Gray for unknown
        
        # Draw robot trail
        if len(self.robot_trail) > 1:
            trail_points = []
            for tx, ty in self.robot_trail:
                map_x, map_y = self.world_to_map(tx, ty)
                if 0 <= map_x < self.map_size and 0 <= map_y < self.map_size:
                    trail_points.append((map_x, map_y))
            
            for i in range(len(trail_points) - 1):
                cv2.line(vis_map, trail_points[i], trail_points[i + 1], (0, 255, 0), 2)
        
        # Draw robot position
        robot_map_x, robot_map_y = self.world_to_map(robot_x, robot_y)
        if 0 <= robot_map_x < self.map_size and 0 <= robot_map_y < self.map_size:
            cv2.circle(vis_map, (robot_map_x, robot_map_y), 8, (0, 0, 255), -1)
            
            # Draw robot orientation
            end_x = robot_map_x + int(20 * math.cos(robot_theta))
            end_y = robot_map_y + int(20 * math.sin(robot_theta))
            cv2.arrowedLine(vis_map, (robot_map_x, robot_map_y), (end_x, end_y), (0, 0, 255), 3)
            
            # Draw target direction if available
            if target_direction is not None:
                target_end_x = robot_map_x + int(30 * math.cos(target_direction))
                target_end_y = robot_map_y + int(30 * math.sin(target_direction))
                cv2.arrowedLine(vis_map, (robot_map_x, robot_map_y), (target_end_x, target_end_y), (255, 255, 0), 2)
        
        # Draw LiDAR rays
        for i, (range_m, angle) in enumerate(zip(ranges, angles)):
            if i % 4 == 0:  # Draw every 4th ray for clarity
                world_angle = robot_theta + angle
                end_x = robot_x + min(range_m, 5.0) * math.cos(world_angle)  # Limit display range
                end_y = robot_y + min(range_m, 5.0) * math.sin(world_angle)
                end_map_x, end_map_y = self.world_to_map(end_x, end_y)
                
                if (0 <= end_map_x < self.map_size and 0 <= end_map_y < self.map_size):
                    color = (0, 255, 255) if range_m < self.sensor_manager.lidar_max_range * 0.95 else (100, 100, 100)
                    cv2.line(vis_map, (robot_map_x, robot_map_y), (end_map_x, end_map_y), color, 1)
                    if range_m < self.sensor_manager.lidar_max_range * 0.95:
                        cv2.circle(vis_map, (end_map_x, end_map_y), 2, (0, 255, 0), -1)
        
        # Resize and show
        vis_map_resized = cv2.resize(vis_map, (600, 600))
        cv2.putText(vis_map_resized, f"LiDAR Navigation", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_map_resized, f"Pos: ({robot_x:.1f}, {robot_y:.1f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_map_resized, f"Angle: {np.degrees(robot_theta):.1f}°", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("LiDAR Navigation", vis_map_resized)
        cv2.waitKey(1)
    
    def stop(self):
        """Stop robot movement"""
        p.setJointMotorControlArray(
            self.robot_id,
            [2, 4, 3, 5],
            p.VELOCITY_CONTROL,
            targetVelocities=[0] * 4
        )
        self.running = False
