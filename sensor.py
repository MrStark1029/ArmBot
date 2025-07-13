import pybullet as p
import numpy as np
import cv2
from object_detection import ObjectDetector
from room_mapping import RoomMapper
from camera_utils import CameraManager
from visualization import VisualizationManager

class SensorManager:
    def __init__(self, husky_id, panda_id=None):
        self.husky_id = husky_id
        self.panda_id = panda_id
        
        # Initialize all subsystems
        self.camera_manager = CameraManager()
        self.object_detector = ObjectDetector()
        self.room_mapper = RoomMapper()
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
        self.bottle_color_range = [(5, 100, 100), (25, 255, 255)]  # HSV range for bottle detection
        self.cup_color_range = [(20, 100, 100), (30, 255, 255)]    # HSV range for cup detection
        
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
            if self.panda_id is None:
                return None, None, None
            end_effector_state = p.getLinkState(self.panda_id, 11)  # Panda end effector link
            robot_pos = end_effector_state[0]
            robot_orn = end_effector_state[1]
            
        return self.camera_manager.get_camera_image(robot_pos, robot_orn, camera_type)
    
    def get_lidar_data(self):
        """Get LiDAR data using ray casting"""
        husky_pos, husky_orn = p.getBasePositionAndOrientation(self.husky_id)
        return self.room_mapper.get_obstacle_map(husky_pos, husky_orn)
    
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
        """Update all sensor displays"""
        try:
            # Get camera data
            nav_rgb, nav_depth, nav_seg = self.get_camera_data("navigation")
            manip_rgb, manip_depth, manip_seg = self.get_camera_data("manipulation")
            
            # Get LiDAR data
            lidar_data = self.get_lidar_data()
            
            # Process object detection
            if nav_rgb is not None:
                detection_result, bottle_positions, cup_positions = self.detect_objects(nav_rgb, nav_depth)
            else:
                detection_result = None
            
            # Update displays using visualization manager
            self.visualizer.show_camera_feed('Navigation Camera (RGB)', nav_rgb, 'rgb')
            self.visualizer.show_camera_feed('Navigation Camera (Depth)', nav_depth, 'depth')
            self.visualizer.show_camera_feed('Manipulation Camera (RGB)', manip_rgb, 'rgb')
            self.visualizer.show_camera_feed('Manipulation Camera (Depth)', manip_depth, 'depth')
            
            if detection_result is not None:
                self.visualizer.show_camera_feed('Object Detection', detection_result, 'rgb')
            
            if lidar_data is not None:
                ranges, angles, hit_objects = lidar_data
                self.visualizer.show_lidar_view(ranges, angles, hit_objects, self.lidar_range)
                
        except Exception as e:
            print(f"Error updating sensor data: {e}")
    
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