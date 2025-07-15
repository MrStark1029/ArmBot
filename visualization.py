import cv2
import numpy as np
from scipy import ndimage

class VisualizationManager:
    def __init__(self):
        self.windows_initialized = False
        self.window_positions = {
            'Navigation Camera (RGB)': (0, 0),
            'Navigation Camera (Depth)': (650, 0),
            'Manipulation Camera (RGB)': (0, 400),
            'Manipulation Camera (Depth)': (330, 400),
            'LiDAR Navigation View': (970, 0),
            'Room Map': (970, 400),
            'Object Detection': (660, 400)
        }
        self.window_sizes = {
            'Navigation Camera (RGB)': (640, 360),
            'Navigation Camera (Depth)': (320, 240),
            'Manipulation Camera (RGB)': (320, 240),
            'Manipulation Camera (Depth)': (320, 240),
            'LiDAR Navigation View': (500, 500),
            'Room Map': (500, 500),
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
    
    def show_lidar_view(self, ranges, angles, hit_objects=None, intensities=None, lidar_range=15.0):
        """
        Enhanced LiDAR visualization with proper gap detection and obstacle analysis
        """
        width = height = 600
        img = np.zeros((height, width, 3), dtype=np.uint8)
        center = (width // 2, height // 2)
        scale = height / (2 * lidar_range)
        
        # Ensure numpy arrays
        ranges = np.array(ranges) if not isinstance(ranges, np.ndarray) else ranges
        angles = np.array(angles) if not isinstance(angles, np.ndarray) else angles
        
        # Draw coordinate system and grid
        self._draw_lidar_coordinate_system(img, center, scale, lidar_range)
        
        # Detect gaps and obstacles using the sensor manager's algorithms
        try:
            # Create a temporary sensor manager instance for analysis (not ideal, but works)
            from sensor import SensorManager
            temp_sensor = SensorManager(0)  # Dummy ID
            gaps = temp_sensor.detect_gaps_and_doorways(ranges, angles)
            obstacles = temp_sensor.detect_obstacles(ranges, angles, intensities if intensities is not None else np.ones_like(ranges))
            del temp_sensor
        except:
            gaps = []
            obstacles = []
        
        # Draw LiDAR rays and measurements
        for i, (range_val, angle) in enumerate(zip(ranges, angles)):
            if range_val <= 0 or range_val > lidar_range:
                continue
                
            # Convert to cartesian coordinates
            x = range_val * np.cos(angle)
            y = range_val * np.sin(angle)
            
            # Convert to pixel coordinates
            pixel_x = int(x * scale + center[0])
            pixel_y = int(-y * scale + center[1])  # Flip Y for screen coordinates
            
            if 0 <= pixel_x < width and 0 <= pixel_y < height:
                # Determine point color based on analysis
                color = self._get_lidar_point_color(i, range_val, hit_objects, intensities, gaps, obstacles)
                point_size = 2 if range_val < lidar_range * 0.8 else 1
                
                # Draw ray (dimmed)
                cv2.line(img, center, (pixel_x, pixel_y), (40, 40, 40), 1)
                
                # Draw measurement point
                cv2.circle(img, (pixel_x, pixel_y), point_size, color, -1)
                
                # Add distance labels for key points
                if i % 60 == 0:  # Every ~6 degrees
                    cv2.putText(img, f'{range_val:.1f}', 
                              (pixel_x + 3, pixel_y - 3),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Highlight detected gaps
        self._draw_detected_gaps(img, gaps, center, scale, angles, ranges)
        
        # Highlight detected obstacles
        self._draw_detected_obstacles(img, obstacles, center, scale, angles, ranges)
        
        # Add comprehensive information panel
        self._add_lidar_analysis_info(img, ranges, angles, gaps, obstacles, intensities)
        
        cv2.imshow('LiDAR Navigation View', img)
    
    def _draw_lidar_coordinate_system(self, img, center, scale, lidar_range):
        """Draw coordinate system, grid, and reference markers"""
        height, width = img.shape[:2]
        
        # Draw distance circles
        for radius in [1, 3, 5, 8, 12]:
            if radius <= lidar_range:
                pixel_radius = int(radius * scale)
                cv2.circle(img, center, pixel_radius, (70, 70, 70), 1)
                # Distance labels
                cv2.putText(img, f'{radius}m', 
                          (center[0] + pixel_radius - 25, center[1] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Draw angle lines every 45 degrees
        for angle_deg in range(0, 360, 45):
            angle_rad = np.radians(angle_deg)
            end_x = int(center[0] + lidar_range * scale * np.cos(angle_rad))
            end_y = int(center[1] - lidar_range * scale * np.sin(angle_rad))
            cv2.line(img, center, (end_x, end_y), (60, 60, 60), 1)
            
            # Angle labels
            label_radius = lidar_range * scale * 0.85
            label_x = int(center[0] + label_radius * np.cos(angle_rad))
            label_y = int(center[1] - label_radius * np.sin(angle_rad))
            cv2.putText(img, f'{angle_deg}°', 
                      (label_x - 15, label_y + 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
        
        # Draw main axes
        cv2.line(img, (center[0], 0), (center[0], height), (100, 100, 100), 2)
        cv2.line(img, (0, center[1]), (width, center[1]), (100, 100, 100), 2)
        
        # Robot position and orientation
        cv2.circle(img, center, 12, (255, 255, 255), -1)
        cv2.circle(img, center, 12, (0, 0, 255), 2)
        # Forward direction arrow
        cv2.arrowedLine(img, center, (center[0] + 30, center[1]), (255, 255, 255), 3)
        cv2.putText(img, 'ROBOT', (center[0] - 20, center[1] + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _get_lidar_point_color(self, index, range_val, hit_objects, intensities, gaps, obstacles):
        """Determine color for LiDAR point based on analysis results"""
        # Default color (wall/obstacle)
        color = (0, 150, 0)  # Green
        
        # Check if point is part of a detected gap
        for gap in gaps:
            if gap['start_idx'] <= index <= gap['end_idx']:
                return (255, 255, 0)  # Yellow for gaps
        
        # Check if point is part of a detected obstacle
        for obstacle in obstacles:
            if obstacle['start_idx'] <= index <= obstacle['end_idx']:
                severity = obstacle.get('severity', 0.5)
                red_intensity = int(255 * severity)
                return (0, 0, red_intensity)  # Red gradient for obstacles
        
        # Use intensity if available
        if intensities is not None and index < len(intensities):
            intensity = intensities[index]
            intensity_val = int(150 * intensity)
            color = (0, intensity_val, 0)
        
        # Use hit object information if available
        if hit_objects is not None and index < len(hit_objects):
            if hit_objects[index] > 0:
                color = (0, 100, 255)  # Orange for known objects
        
        return color
    
    def _draw_detected_gaps(self, img, gaps, center, scale, angles, ranges):
        """Draw detected gaps and doorways"""
        for i, gap in enumerate(gaps):
            start_angle = gap['start_angle']
            end_angle = gap['end_angle']
            distance = gap['distance']
            width = gap['width']
            confidence = gap['confidence']
            
            # Calculate arc endpoints
            start_x = int(center[0] + distance * scale * np.cos(start_angle))
            start_y = int(center[1] - distance * scale * np.sin(start_angle))
            end_x = int(center[0] + distance * scale * np.cos(end_angle))
            end_y = int(center[1] - distance * scale * np.sin(end_angle))
            
            # Draw gap arc
            color_intensity = int(255 * confidence)
            cv2.line(img, (start_x, start_y), (end_x, end_y), (255, color_intensity, 0), 4)
            
            # Draw gap indicator
            mid_angle = (start_angle + end_angle) / 2
            mid_x = int(center[0] + distance * scale * np.cos(mid_angle))
            mid_y = int(center[1] - distance * scale * np.sin(mid_angle))
            
            cv2.circle(img, (mid_x, mid_y), 8, (255, 255, 0), -1)
            cv2.putText(img, f'GAP{i+1}', (mid_x - 20, mid_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(img, f'{width:.1f}m', (mid_x - 15, mid_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
    
    def _draw_detected_obstacles(self, img, obstacles, center, scale, angles, ranges):
        """Draw detected obstacles with severity indication"""
        for i, obstacle in enumerate(obstacles):
            start_angle = obstacle['start_angle']
            end_angle = obstacle['end_angle']
            min_distance = obstacle['min_distance']
            severity = obstacle['severity']
            
            # Calculate obstacle region
            num_points = 20
            obstacle_angles = np.linspace(start_angle, end_angle, num_points)
            
            # Draw obstacle boundary
            points = []
            for angle in obstacle_angles:
                # Use minimum distance for conservative boundary
                x = int(center[0] + min_distance * scale * np.cos(angle))
                y = int(center[1] - min_distance * scale * np.sin(angle))
                points.append([x, y])
            
            if len(points) > 1:
                points = np.array(points, dtype=np.int32)
                red_intensity = int(255 * severity)
                cv2.polylines(img, [points], False, (0, 0, red_intensity), 3)
                
                # Add obstacle label
                mid_idx = len(points) // 2
                mid_point = points[mid_idx]
                cv2.putText(img, f'OBS{i+1}', 
                          (mid_point[0] - 20, mid_point[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, red_intensity), 1)
                cv2.putText(img, f'{min_distance:.1f}m', 
                          (mid_point[0] - 15, mid_point[1] + 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, red_intensity), 1)
    
    def _add_lidar_analysis_info(self, img, ranges, angles, gaps, obstacles, intensities):
        """Add comprehensive analysis information panel"""
        # Calculate statistics
        valid_ranges = ranges[ranges > 0]
        min_range = np.min(valid_ranges) if len(valid_ranges) > 0 else 0
        max_range = np.max(valid_ranges) if len(valid_ranges) > 0 else 0
        avg_range = np.mean(valid_ranges) if len(valid_ranges) > 0 else 0
        
        close_obstacles = np.sum(ranges < 2.0)
        
        # Info panel
        info_y = 20
        line_height = 16
        
        info_lines = [
            f'LiDAR Analysis Dashboard',
            f'━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
            f'Scan Statistics:',
            f'  • Total rays: {len(ranges)}',
            f'  • Valid readings: {len(valid_ranges)}',
            f'  • Range: {min_range:.2f}m - {max_range:.2f}m',
            f'  • Average range: {avg_range:.2f}m',
            f'',
            f'Detection Results:',
            f'  • Gaps detected: {len(gaps)}',
            f'  • Obstacles detected: {len(obstacles)}',
            f'  • Close obstacles (<2m): {close_obstacles}',
            f'',
            f'Navigation Status:',
            f'  • {"CLEAR PATH" if len(gaps) > 0 else "BLOCKED"}',
            f'  • {"SAFE" if close_obstacles == 0 else "CAUTION"}',
            f'',
            f'Legend:',
            f'  ● Green: Walls/Static obstacles',
            f'  ● Yellow: Detected gaps/doorways',
            f'  ● Red: Dynamic obstacles',
            f'  ● Orange: Known objects'
        ]
        
        # Semi-transparent background
        overlay = img.copy()
        panel_height = len(info_lines) * line_height + 20
        cv2.rectangle(overlay, (10, 10), (280, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        
        # Draw border
        cv2.rectangle(img, (10, 10), (280, panel_height), (100, 100, 100), 2)
        
        # Add text
        for i, line in enumerate(info_lines):
            color = (255, 255, 255)
            font_size = 0.4
            
            if line.startswith('LiDAR Analysis'):
                color = (0, 255, 255)
                font_size = 0.5
            elif '●' in line:
                if 'Green' in line:
                    color = (0, 255, 0)
                elif 'Yellow' in line:
                    color = (255, 255, 0)
                elif 'Red' in line:
                    color = (0, 0, 255)
                elif 'Orange' in line:
                    color = (0, 165, 255)
            elif 'CLEAR PATH' in line:
                color = (0, 255, 0)
            elif 'BLOCKED' in line:
                color = (0, 0, 255)
            elif 'SAFE' in line:
                color = (0, 255, 0)
            elif 'CAUTION' in line:
                color = (0, 255, 255)
            
            cv2.putText(img, line, (15, info_y + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)

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
            self.show_room_map(room_map)
            
            # Show enhanced LiDAR visualization
            self.show_lidar_view(ranges, angles, hit_objects, intensities, sensor_manager.lidar_max_range)
                
        except Exception as e:
            print(f"Error updating sensor data: {e}")
    
    def cleanup(self):
        """Clean up all visualization windows"""
        cv2.destroyAllWindows()
        self.windows_initialized = False