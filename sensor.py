import pybullet as p
import numpy as np
import cv2
from object_detection import ObjectDetector
from camera_utils import CameraManager
from visualization import VisualizationManager
from scipy import ndimage
from scipy.signal import find_peaks
import math

class SensorManager:
    def __init__(self, husky_id, panda_id=None):
        self.husky_id = husky_id
        self.panda_id = panda_id
        
        # Initialize all subsystems
        self.camera_manager = CameraManager()
        self.object_detector = ObjectDetector()
        self.visualizer = VisualizationManager()
        
        # === Navigation Camera Parameters (Main Camera) ===
        self.nav_camera_height = 0.6  # Higher for better room view
        self.nav_camera_offset = [0.4, 0.0, self.nav_camera_height]
        self.nav_img_width = 1280
        self.nav_img_height = 720
        self.nav_fov = 85  # Wide field of view for navigation
        
        # === Manipulation Camera Parameters (Arm-mounted) ===
        self.manip_camera_height = 0.1
        self.manip_camera_offset = [0.1, 0.0, self.manip_camera_height]  # Close to end effector
        self.manip_img_width = 640
        self.manip_img_height = 480
        self.manip_fov = 60  # Narrower for precise manipulation
        
        # === Depth Camera Parameters ===
        self.near_plane = 0.05  # Closer for better manipulation
        self.far_plane = 50.0   # Sufficient for room navigation
        
        # === LiDAR Parameters (Based on Velodyne VLP-16 specifications) ===
        self.lidar_max_range = 12  # Maximum detection range (meters)
        self.lidar_min_range = 0.5   # Minimum detection range (meters)
        self.lidar_angular_resolution = 0.5  # Angular resolution in degrees
        self.lidar_num_rays = int(360 / self.lidar_angular_resolution)  # 720 rays for 0.5° resolution
        self.lidar_height = 0.5  # Height above ground
        self.lidar_offset = [0.0, 0.0, self.lidar_height]  # Position relative to robot base
        
        # LiDAR noise and error parameters
        self.lidar_noise_std = 0.02  # Standard deviation of range noise (2cm)
        self.lidar_min_intensity = 0.1  # Minimum return intensity threshold
        
        # Dynamic obstacle detection parameters
        self.obstacle_threshold = 0.3  # Minimum height difference for obstacle detection
        self.ground_clearance = 0.05   # Height tolerance for ground detection
        
        # Gap detection parameters (based on research papers)
        self.min_gap_width = 0.2  # Minimum navigable gap width (meters)
        self.max_gap_width = 3.0  # Maximum reasonable gap width (meters)
        self.gap_depth_threshold = 0.5  # Minimum depth change to consider a gap
        
        # === Object Detection Parameters ===
        self.object_detection_enabled = True
        
        # === Segmentation Parameters ===
        self.segmentation_enabled = True
        
        # Setup visualization windows
        self.setup_windows()
    
    def setup_windows(self):
        """Setup windows for sensor visualization"""
        self.visualizer.setup_windows()
    
    def get_camera_data(self, camera_type="navigation"):
        """Get RGB and depth data from specified camera"""
        # Get robot position and orientation
        if camera_type == "navigation":
            robot_pos, robot_orn = p.getBasePositionAndOrientation(self.husky_id)
        else:  # manipulation
            # Use Husky position for manipulation camera (mounted on robot)
            robot_pos, robot_orn = p.getBasePositionAndOrientation(self.husky_id)
            
        return self.camera_manager.get_camera_image(robot_pos, robot_orn, camera_type)
    
    def get_lidar_data(self):
        """
        Get LiDAR data using proper 2D LiDAR simulation
        Based on Sick LMS and Velodyne principles
        """
        # Get robot pose
        robot_pos, robot_quat = p.getBasePositionAndOrientation(self.husky_id)
        
        # Convert quaternion to Euler angles for 2D rotation
        robot_euler = p.getEulerFromQuaternion(robot_quat)
        robot_yaw = robot_euler[2]  # Z-axis rotation
        
        # Calculate LiDAR sensor position in world coordinates
        lidar_local_pos = np.array(self.lidar_offset)
        
        # Rotate local position by robot orientation
        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)
        rotation_matrix_2d = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])
        
        lidar_world_pos = np.array(robot_pos) + rotation_matrix_2d @ lidar_local_pos
        
        # Generate ray angles (0 to 2π with specified resolution)
        num_rays = self.lidar_num_rays
        angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
        
        # Pre-allocate arrays for performance
        ranges = np.full(num_rays, self.lidar_max_range, dtype=np.float32)
        intensities = np.zeros(num_rays, dtype=np.float32)
        hit_objects = np.full(num_rays, -1, dtype=np.int32)
        hit_positions = np.zeros((num_rays, 3), dtype=np.float32)
        
        # Batch ray casting for efficiency
        ray_starts = np.tile(lidar_world_pos, (num_rays, 1))
        
        # Calculate ray end points in world coordinates
        ray_directions_local = np.column_stack([
            np.cos(angles),
            np.sin(angles),
            np.zeros(num_rays)
        ])
        
        # Transform ray directions by robot orientation
        ray_directions_world = np.array([rotation_matrix_2d @ ray_dir for ray_dir in ray_directions_local])
        ray_ends = ray_starts + ray_directions_world * self.lidar_max_range
        
        # Perform batch ray casting
        ray_results = p.rayTestBatch(ray_starts.tolist(), ray_ends.tolist())
        
        for i, result in enumerate(ray_results):
            object_id, link_id, hit_fraction, hit_position, hit_normal = result
            
            # Check if ray hit something
            if object_id >= 0:
                # Filter out self-collisions
                if object_id == self.husky_id or (self.panda_id and object_id == self.panda_id):
                    continue
                
                # Calculate actual range
                actual_range = hit_fraction * self.lidar_max_range
                
                # Apply range limits
                if self.lidar_min_range <= actual_range <= self.lidar_max_range:
                    ranges[i] = actual_range
                    hit_objects[i] = object_id
                    hit_positions[i] = hit_position
                    
                    # Calculate intensity based on distance and surface normal
                    # Simplified intensity model: I = I₀ * cos(θ) / r²
                    distance_factor = 1.0 / (actual_range ** 2 + 0.1)  # Avoid division by zero
                    normal_factor = abs(np.dot(hit_normal, -ray_directions_world[i]))
                    intensities[i] = min(1.0, distance_factor * normal_factor * 100)
        
        # Add realistic noise to range measurements
        noise = np.random.normal(0, self.lidar_noise_std, num_rays)
        ranges = np.clip(ranges + noise, self.lidar_min_range, self.lidar_max_range)
        
        # Convert to robot-relative coordinates for easier processing
        relative_angles = angles - robot_yaw
        relative_angles = (relative_angles + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]
        
        return ranges, relative_angles, hit_objects, intensities, hit_positions
    
    def detect_gaps_and_doorways(self, ranges, angles):
        """
        Detect gaps and doorways using gradient-based analysis
        Based on: "Gap Detection for Mobile Robot Navigation" - IEEE
        """
        # Apply median filter to reduce noise
        filtered_ranges = ndimage.median_filter(ranges, size=5)
        
        # Calculate range gradients
        range_gradient = np.gradient(filtered_ranges)
        
        # Detect significant range changes (edges)
        edge_threshold = 0.5  # meters
        edge_indices = np.where(np.abs(range_gradient) > edge_threshold)[0]
        
        gaps = []
        
        # Group consecutive edge points
        if len(edge_indices) > 0:
            # Find groups of consecutive edges
            edge_groups = []
            current_group = [edge_indices[0]]
            
            for i in range(1, len(edge_indices)):
                if edge_indices[i] - edge_indices[i-1] <= 3:  # Allow small gaps
                    current_group.append(edge_indices[i])
                else:
                    if len(current_group) >= 2:
                        edge_groups.append(current_group)
                    current_group = [edge_indices[i]]
            
            if len(current_group) >= 2:
                edge_groups.append(current_group)
            
            # Analyze each edge group for potential doorways
            for group in edge_groups:
                start_idx = group[0]
                end_idx = group[-1]
                
                # Calculate gap properties
                start_angle = angles[start_idx]
                end_angle = angles[end_idx]
                
                # Handle angle wraparound
                angle_diff = end_angle - start_angle
                if angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                elif angle_diff < -np.pi:
                    angle_diff += 2 * np.pi
                
                # Estimate gap width using average distance
                avg_distance = np.mean([filtered_ranges[start_idx], filtered_ranges[end_idx]])
                gap_width = abs(angle_diff) * avg_distance
                
                # Check if this could be a navigable gap
                if self.min_gap_width <= gap_width <= self.max_gap_width:
                    # Additional validation: check depth consistency
                    gap_ranges = filtered_ranges[start_idx:end_idx+1]
                    if len(gap_ranges) > 0:
                        depth_variation = np.std(gap_ranges)
                        if depth_variation < 1.0:  # Consistent depth
                            gaps.append({
                                'start_angle': start_angle,
                                'end_angle': end_angle,
                                'center_angle': (start_angle + end_angle) / 2,
                                'width': gap_width,
                                'distance': avg_distance,
                                'confidence': min(1.0, gap_width / self.max_gap_width),
                                'start_idx': start_idx,
                                'end_idx': end_idx
                            })
        
        return gaps
    
    def detect_obstacles(self, ranges, angles, intensities):
        """
        Detect static and dynamic obstacles using multi-criteria analysis
        """
        obstacles = []
        
        # Apply smoothing filter
        smooth_ranges = ndimage.gaussian_filter1d(ranges, sigma=2.0)
        
        # Detect close obstacles (potential collision risks)
        close_threshold = 2.0  # meters
        close_indices = np.where(smooth_ranges < close_threshold)[0]
        
        if len(close_indices) > 0:
            # Group consecutive close points
            obstacle_groups = []
            current_group = [close_indices[0]]
            
            for i in range(1, len(close_indices)):
                if close_indices[i] - close_indices[i-1] <= 2:
                    current_group.append(close_indices[i])
                else:
                    if len(current_group) >= 3:  # Minimum size for obstacle
                        obstacle_groups.append(current_group)
                    current_group = [close_indices[i]]
            
            if len(current_group) >= 3:
                obstacle_groups.append(current_group)
            
            # Analyze each obstacle group
            for group in obstacle_groups:
                start_idx = group[0]
                end_idx = group[-1]
                
                # Calculate obstacle properties
                obstacle_ranges = smooth_ranges[start_idx:end_idx+1]
                obstacle_angles = angles[start_idx:end_idx+1]
                
                avg_distance = np.mean(obstacle_ranges)
                min_distance = np.min(obstacle_ranges)
                
                # Estimate obstacle size
                angle_span = obstacle_angles[-1] - obstacle_angles[0]
                obstacle_width = abs(angle_span) * avg_distance
                
                obstacles.append({
                    'start_angle': obstacle_angles[0],
                    'end_angle': obstacle_angles[-1],
                    'center_angle': np.mean(obstacle_angles),
                    'min_distance': min_distance,
                    'avg_distance': avg_distance,
                    'width': obstacle_width,
                    'severity': 1.0 - (min_distance / close_threshold),  # Higher for closer obstacles
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
        
        return obstacles
    
    def get_navigation_waypoints(self, target_angle=None):
        """
        Generate navigation waypoints based on LiDAR analysis
        """
        ranges, angles, hit_objects, intensities, hit_positions = self.get_lidar_data()
        
        # Detect gaps and obstacles
        gaps = self.detect_gaps_and_doorways(ranges, angles)
        obstacles = self.detect_obstacles(ranges, angles, intensities)
        
        waypoints = []
        
        # If no target specified, find the largest gap
        if target_angle is None:
            if gaps:
                best_gap = max(gaps, key=lambda g: g['width'] * g['confidence'])
                target_angle = best_gap['center_angle']
            else:
                # No gaps found, head to the direction with maximum clearance
                max_range_idx = np.argmax(ranges)
                target_angle = angles[max_range_idx]
        
        # Generate waypoint in the direction of the target
        waypoint_distance = min(3.0, np.max(ranges) * 0.7)  # Conservative distance
        waypoint_x = waypoint_distance * np.cos(target_angle)
        waypoint_y = waypoint_distance * np.sin(target_angle)
        
        waypoints.append({
            'position': [waypoint_x, waypoint_y],
            'angle': target_angle,
            'distance': waypoint_distance,
            'confidence': 1.0
        })
        
        return waypoints, gaps, obstacles
    
    def detect_objects(self, rgb_image, depth_image):
        """Detect objects in the camera feed"""
        if rgb_image is None:
            return None, [], []
            
        result_image, bottle_positions, cup_positions = self.object_detector.detect_objects(rgb_image, depth_image)
        return result_image, bottle_positions, cup_positions
    
    def visualize_lidar(self, ranges, angles, hit_objects=None, width=800, height=800):
        """Convert LiDAR data to bird's eye view image with object classification"""
        return self.visualizer.visualize_lidar(ranges, angles, self.lidar_max_range, self.lidar_num_rays, hit_objects, width, height)
    
    def update_displays(self):
        """Update all sensor displays - delegated to VisualizationManager"""
        self.visualizer.update_all_displays(self)
    
    def get_bottle_position(self):
        """Get estimated bottle position from camera data"""
        nav_rgb, nav_depth, _ = self.get_camera_data("navigation")
        if nav_rgb is None or nav_depth is None:
            return None
        
        _, bottle_positions, _ = self.detect_objects(nav_rgb, nav_depth)
        if bottle_positions:
            bottle = bottle_positions[0]  # Get the first detected bottle
            return (bottle['position'][0], bottle['position'][1], bottle['depth'])
        return None
    
    def get_cup_position(self):
        """Get estimated cup position from camera data"""
        nav_rgb, nav_depth, _ = self.get_camera_data("navigation")
        if nav_rgb is None or nav_depth is None:
            return None
        
        _, _, cup_positions = self.detect_objects(nav_rgb, nav_depth)
        if cup_positions:
            cup = cup_positions[0]  # Get the first detected cup
            return (cup['position'][0], cup['position'][1], cup['depth'])
        return None
    
    def get_obstacle_map(self):
        """Get obstacle map from LiDAR for path planning"""
        ranges, angles, hit_objects, intensities, hit_positions = self.get_lidar_data()
        obstacles = self.detect_obstacles(ranges, angles, intensities)
        return obstacles
    
    def destroy_windows(self):
        """Clean up visualization windows"""
        self.visualizer.cleanup()
    
    def get_room_map(self):
        """Generate enhanced room map from LiDAR data with proper coordinate system"""
        # Get current LiDAR data
        ranges, angles, hit_objects, intensities, hit_positions = self.get_lidar_data()
        
        # Create occupancy grid map
        map_size = 500  # 500x500 pixel map for better resolution
        map_img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 200  # Light gray background
        center = (map_size // 2, map_size // 2)
        scale = map_size / (2 * self.lidar_max_range)  # pixels per meter
        
        # Convert LiDAR points to map coordinates with proper transformation
        for i, (range_val, angle) in enumerate(zip(ranges, angles)):
            if range_val < self.lidar_max_range:  # Valid measurement
                # Convert polar to cartesian (robot frame)
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                
                # Transform to map coordinates
                map_x = int(x * scale + center[0])
                map_y = int(-y * scale + center[1])  # Flip Y for image coordinates
                
                if 0 <= map_x < map_size and 0 <= map_y < map_size:
                    # Color based on intensity and object type
                    intensity = intensities[i] if i < len(intensities) else 0.5
                    
                    if hit_objects[i] > 0:
                        # Known object
                        color = (0, int(255 * intensity), 0)  # Green
                    else:
                        # Wall or unknown obstacle
                        color = (0, 0, int(255 * intensity))  # Red
                    
                    cv2.circle(map_img, (map_x, map_y), 2, color, -1)
        
        # Draw robot position and orientation
        cv2.circle(map_img, center, 8, (255, 255, 255), -1)  # White center
        cv2.circle(map_img, center, 8, (0, 0, 255), 2)  # Red border
        
        # Draw forward direction arrow
        arrow_end = (center[0] + 20, center[1])
        cv2.arrowedLine(map_img, center, arrow_end, (255, 255, 255), 2)
        
        # Draw range circles and grid
        for radius_m in [2, 5, 10]:
            if radius_m <= self.lidar_max_range:
                pixel_radius = int(radius_m * scale)
                cv2.circle(map_img, center, pixel_radius, (150, 150, 150), 1)
                cv2.putText(map_img, f'{radius_m}m', 
                          (center[0] + pixel_radius - 20, center[1] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # Add coordinate axes
        cv2.line(map_img, (center[0], 0), (center[0], map_size), (180, 180, 180), 1)
        cv2.line(map_img, (0, center[1]), (map_size, center[1]), (180, 180, 180), 1)
        
        # Add title and scale information
        cv2.putText(map_img, 'LiDAR Occupancy Map', (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(map_img, f'Range: {self.lidar_max_range}m | Resolution: {self.lidar_angular_resolution}°', 
                   (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return map_img