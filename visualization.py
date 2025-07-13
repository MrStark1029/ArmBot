import cv2
import numpy as np

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
            'LiDAR Navigation View': (400, 400),
            'Room Map': (400, 400),
            'Object Detection': (320, 240)
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
            
        if image_type == 'rgb':
            display_img = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        elif image_type == 'depth':
            display_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            display_img = cv2.applyColorMap(display_img, cv2.COLORMAP_PLASMA)
        
        cv2.imshow(name, display_img)
    
    def show_lidar_view(self, ranges, angles, hit_objects=None, lidar_range=12.0):
        """Visualize LiDAR data in bird's eye view"""
        width = height = 800
        img = np.zeros((height, width, 3), dtype=np.uint8)
        center = (width // 2, height // 2)
        scale = height / (2 * lidar_range)
        
        # Convert polar to cartesian coordinates
        x_coords = ranges * np.cos(angles)
        y_coords = ranges * np.sin(angles)
        points = np.column_stack([
            (x_coords * scale + center[0]).astype(int),
            (y_coords * scale + center[1]).astype(int)
        ])
        
        # Draw points
        for i, point in enumerate(points):
            if 0 <= point[0] < width and 0 <= point[1] < height:
                color = (0, 255, 0)  # Default green
                if hit_objects is not None and i < len(hit_objects):
                    if hit_objects[i] > 0:  # Object detected
                        color = (0, 100, 255)  # Orange for objects
                cv2.circle(img, tuple(point), 2, color, -1)
        
        # Draw reference elements
        self._draw_lidar_reference(img, center, scale, lidar_range)
        
        cv2.imshow('LiDAR Navigation View', img)
    
    def _draw_lidar_reference(self, img, center, scale, lidar_range):
        """Draw reference elements on LiDAR visualization"""
        # Draw robot position and orientation
        cv2.circle(img, center, 5, (255, 255, 255), -1)
        cv2.arrowedLine(img, center, (center[0] + 20, center[1]), (255, 255, 255), 2)
        
        # Draw reference circles and labels
        for radius in [1, 2, 3, 5, 10]:
            if radius <= lidar_range:
                cv2.circle(img, center, int(radius * scale), (50, 50, 50), 1)
                cv2.putText(img, f'{radius}m', 
                          (center[0] + int(radius * scale), center[1]),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        # Draw coordinate axes
        cv2.line(img, (center[0], 0), (center[0], img.shape[0]), (30, 30, 30), 1)
        cv2.line(img, (0, center[1]), (img.shape[1], center[1]), (30, 30, 30), 1)
        
        # Add navigation info
        cv2.putText(img, 'LiDAR Navigation View', (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f'Range: {lidar_range}m', (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def visualize_lidar(self, ranges, angles, lidar_range, lidar_resolution, hit_objects=None, width=800, height=800):
        """Convert LiDAR data to bird's eye view image with object classification"""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        center = (width // 2, height // 2)
        scale = height / (2 * lidar_range)
        
        x_coords = ranges * np.cos(angles)
        y_coords = ranges * np.sin(angles)
        
        # Draw LiDAR points with different colors based on object type
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            pixel_x = int(x * scale + center[0])
            pixel_y = int(y * scale + center[1])
            
            if 0 <= pixel_x < width and 0 <= pixel_y < height:
                # Color coding: walls=green, objects=red, unknown=white
                if hit_objects is not None and i < len(hit_objects):
                    if hit_objects[i] == 0:  # Ground plane
                        continue
                    elif hit_objects[i] > 0:  # Object detected
                        color = (0, 100, 255)  # Orange for objects
                    else:
                        color = (0, 255, 0)  # Green for walls/unknown
                else:
                    color = (0, 255, 0)  # Default green
                
                cv2.circle(img, (pixel_x, pixel_y), 2, color, -1)
        
        # Draw robot position and orientation
        cv2.circle(img, center, 5, (255, 255, 255), -1)
        cv2.arrowedLine(img, center, (center[0] + 20, center[1]), (255, 255, 255), 2)
        
        # Draw reference circles for distance estimation
        for radius in [1, 2, 3, 5, 10]:
            if radius <= lidar_range:
                cv2.circle(img, center, int(radius * scale), (50, 50, 50), 1)
                cv2.putText(img, f'{radius}m', (center[0] + int(radius * scale), center[1]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        # Draw coordinate axes
        cv2.line(img, (center[0], 0), (center[0], height), (30, 30, 30), 1)
        cv2.line(img, (0, center[1]), (width, center[1]), (30, 30, 30), 1)
        
        # Add navigation info
        cv2.putText(img, 'LiDAR Navigation View', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f'Range: {lidar_range}m', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f'Resolution: {lidar_resolution}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return img
    
    def cleanup(self):
        """Clean up all visualization windows"""
        cv2.destroyAllWindows()
        self.windows_initialized = False
