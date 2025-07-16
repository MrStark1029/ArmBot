import cv2
import numpy as np
from scipy import ndimage

class VisualizationManager:
    def __init__(self):
        self.windows_initialized = False
        self.window_positions = {
            'Navigation Camera (RGB)': (0, 0),
            'Navigation Camera (Depth)': (660, 0),
            'Manipulation Camera (RGB)': (0, 380),
            'Manipulation Camera (Depth)': (330, 380),
            'Room Map': (660, 380),
            'Object Detection': (1200, 0)
        }
        self.window_sizes = {
            'Navigation Camera (RGB)': (640, 360),
            'Navigation Camera (Depth)': (320, 240),
            'Manipulation Camera (RGB)': (320, 240),
            'Manipulation Camera (Depth)': (320, 240),
            'Room Map': (520, 520),
            'Object Detection': (640, 360)
        }
    
    def setup_windows(self):
        """Initialize visualization windows"""
        if not self.windows_initialized:
            for window_name in self.window_positions.keys():
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, *self.window_sizes[window_name])
                cv2.moveWindow(window_name, *self.window_positions[window_name])
            self.windows_initialized = True
    
    def show_camera_feed(self, name, image, image_type='rgb'):
        """Display camera feed with appropriate processing"""
        if image is None:
            return
            
        try:
            if image_type == 'rgb':
                display_img = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
            elif image_type == 'depth':
                display_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                display_img = cv2.applyColorMap(display_img, cv2.COLORMAP_PLASMA)
            
            cv2.imshow(name, display_img)
        except Exception as e:
            print(f"Error displaying {name}: {e}")

    def show_object_detection(self, detection_result):
        """Display object detection results"""
        if detection_result is not None:
            self.show_camera_feed('Object Detection', detection_result, 'rgb')
    
    def show_room_map(self, room_map_image):
        """Display room mapping visualization"""
        if room_map_image is not None:
            cv2.imshow('Room Map', room_map_image)
    
    def update_all_displays(self, sensor_manager):
        """Update all sensor displays - moved from SensorManager"""
        try:
            # Get camera data
            nav_rgb, nav_depth, nav_seg = sensor_manager.get_camera_data("navigation")
            manip_rgb, manip_depth, manip_seg = sensor_manager.get_camera_data("manipulation")
            
            # Get enhanced LiDAR data
            ranges, angles, hit_objects, intensities, hit_positions = sensor_manager.get_lidar_data()
            
            # Get room map
            room_map = sensor_manager.get_room_map()
            
            # Process object detection
            detection_result = None
            if nav_rgb is not None:
                try:
                    detection_result, bottle_positions, cup_positions = sensor_manager.detect_objects(nav_rgb, nav_depth)
                    if detection_result is None:
                        detection_result = nav_rgb
                except Exception as e:
                    print(f"Object detection error: {e}")
                    detection_result = nav_rgb
            
            # Update all displays
            self.show_camera_feed('Navigation Camera (RGB)', nav_rgb, 'rgb')
            self.show_camera_feed('Navigation Camera (Depth)', nav_depth, 'depth')
            self.show_camera_feed('Manipulation Camera (RGB)', manip_rgb, 'rgb')
            self.show_camera_feed('Manipulation Camera (Depth)', manip_depth, 'depth')
            
            self.show_object_detection(detection_result)
            
            # Show room map
            self.show_room_map(room_map)
                
        except Exception as e:
            print(f"Error updating sensor data: {e}")
    
    def cleanup(self):
        """Clean up all visualization windows"""
        cv2.destroyAllWindows()
        self.windows_initialized = False