import cv2
import numpy as np

class ObjectDetector:
    def __init__(self):
        # HSV color ranges for different objects
        self.bottle_color_range = [(5, 100, 100), (25, 255, 255)]  # HSV range for bottle detection
        self.cup_color_range = [(20, 100, 100), (30, 255, 255)]    # HSV range for cup detection
    
    def detect_objects(self, rgb_image, depth_image=None):
        """Detect bottles and cups in the image"""
        if rgb_image is None:
            return rgb_image
            
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        # Create masks for bottle and cup
        bottle_mask = cv2.inRange(hsv, np.array(self.bottle_color_range[0]), 
                                np.array(self.bottle_color_range[1]))
        cup_mask = cv2.inRange(hsv, np.array(self.cup_color_range[0]), 
                             np.array(self.cup_color_range[1]))
        
        result_image = rgb_image.copy()
        
        # Detect bottles
        bottle_positions = self._detect_bottle(bottle_mask, result_image, depth_image)
        # Detect cups
        cup_positions = self._detect_cup(cup_mask, result_image, depth_image)
        
        return result_image, bottle_positions, cup_positions
    
    def _detect_bottle(self, mask, image, depth_image=None):
        """Detect bottles using contour detection"""
        positions = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small detections
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, 'BOTTLE', (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                center_x, center_y = x + w//2, y + h//2
                depth = None
                
                if depth_image is not None:
                    if 0 <= center_x < depth_image.shape[1] and 0 <= center_y < depth_image.shape[0]:
                        depth = depth_image[center_y, center_x]
                        cv2.putText(image, f'{depth:.2f}m', (x, y + h + 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                positions.append({
                    'type': 'bottle',
                    'position': (center_x, center_y),
                    'depth': depth,
                    'bbox': (x, y, w, h)
                })
        
        return positions
    
    def _detect_cup(self, mask, image, depth_image=None):
        """Detect cups using contour detection"""
        positions = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 300:  # Filter small detections
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, 'CUP', (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                center_x, center_y = x + w//2, y + h//2
                depth = None
                
                if depth_image is not None:
                    if 0 <= center_x < depth_image.shape[1] and 0 <= center_y < depth_image.shape[0]:
                        depth = depth_image[center_y, center_x]
                        cv2.putText(image, f'{depth:.2f}m', (x, y + h + 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                positions.append({
                    'type': 'cup',
                    'position': (center_x, center_y),
                    'depth': depth,
                    'bbox': (x, y, w, h)
                })
        
        return positions
