import pybullet as p
import numpy as np
import cv2
from object_detection import ObjectDetector
from camera_utils import CameraManager
from visualization import VisualizationManager

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
        
        # === LiDAR Parameters (Optimized for indoor navigation) ===
        self.lidar_range = 12.0  # Good for house rooms
        self.lidar_resolution = 720  # Higher resolution for better obstacle detection
        self.lidar_height = 0.3
        self.lidar_offset = [0.0, 0.0, self.lidar_height]
        
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
        """Get LiDAR data using ray casting"""
        husky_pos, husky_orn = p.getBasePositionAndOrientation(self.husky_id)
        
        # Convert quaternion to rotation matrix
        rotation_matrix = p.getMatrixFromQuaternion(husky_orn)
        rotation_matrix = np.array(rotation_matrix).reshape(3, 3)
        
        # Calculate LiDAR position in world coordinates
        lidar_pos_local = np.array(self.lidar_offset)
        lidar_pos_world = np.array(husky_pos) + rotation_matrix @ lidar_pos_local
        
        # Generate ray directions (360 degrees around Z-axis)
        angles = np.linspace(0, 2*np.pi, self.lidar_resolution, endpoint=False)
        ranges = []
        hit_objects = []
        
        for angle in angles:
            # Ray direction in local coordinates (XY plane)
            ray_dir_local = np.array([np.cos(angle), np.sin(angle), 0])
            # Transform to world coordinates
            ray_dir_world = rotation_matrix @ ray_dir_local
            # End point of ray
            ray_end = lidar_pos_world + ray_dir_world * self.lidar_range
            
            # Cast ray
            ray_result = p.rayTest(lidar_pos_world, ray_end)
            
            if ray_result[0][0] >= 0:  # Hit something
                hit_distance = ray_result[0][2] * self.lidar_range
                ranges.append(hit_distance)
                hit_objects.append(ray_result[0][0])  # Object ID
            else:  # No hit
                ranges.append(self.lidar_range)
                hit_objects.append(-1)
        
        return np.array(ranges), angles, hit_objects
    
    def detect_objects(self, rgb_image, depth_image):
        """Detect objects in the camera feed"""
        if rgb_image is None:
            return None, [], []
            
        result_image, bottle_positions, cup_positions = self.object_detector.detect_objects(rgb_image, depth_image)
        return result_image, bottle_positions, cup_positions
    
    def visualize_lidar(self, ranges, angles, hit_objects=None, width=800, height=800):
        """Convert LiDAR data to bird's eye view image with object classification"""
        return self.visualizer.visualize_lidar(ranges, angles, self.lidar_range, self.lidar_resolution, hit_objects, width, height)
    
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
        return self.room_mapper.get_obstacle_map(*self.get_lidar_data())
    
    def destroy_windows(self):
        """Clean up visualization windows"""
        self.visualizer.cleanup()
    
    def get_room_map(self):
        """Generate a simple room map from LiDAR data"""
        # Get current LiDAR data
        ranges, angles, hit_objects = self.get_lidar_data()
        
        # Create a simple occupancy grid map
        map_size = 400  # 400x400 pixel map
        map_img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 128  # Gray background
        center = (map_size // 2, map_size // 2)
        scale = map_size / (2 * self.lidar_range)
        
        # Convert LiDAR points to map coordinates
        x_coords = ranges * np.cos(angles)
        y_coords = ranges * np.sin(angles)
        
        # Mark occupied cells
        for i, (x, y, range_val) in enumerate(zip(x_coords, y_coords, ranges)):
            if range_val < self.lidar_range:  # Valid hit
                map_x = int(x * scale + center[0])
                map_y = int(y * scale + center[1])
                
                if 0 <= map_x < map_size and 0 <= map_y < map_size:
                    # Draw occupied cell
                    cv2.circle(map_img, (map_x, map_y), 2, (0, 0, 0), -1)  # Black for obstacles
        
        # Draw robot position
        cv2.circle(map_img, center, 5, (0, 0, 255), -1)  # Red for robot
        cv2.arrowedLine(map_img, center, (center[0] + 15, center[1]), (0, 0, 255), 2)
        
        # Add grid lines
        for i in range(0, map_size, 40):
            cv2.line(map_img, (i, 0), (i, map_size), (200, 200, 200), 1)
            cv2.line(map_img, (0, i), (map_size, i), (200, 200, 200), 1)
        
        # Add title
        cv2.putText(map_img, 'Room Map', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(map_img, f'Scale: {self.lidar_range}m', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return map_img