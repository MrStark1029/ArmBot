import cv2
import numpy as np
import time

class ModeController:
    """
    A GUI-based mode controller that provides clean switching between
    manual and autonomous modes without keyboard interference.
    """
    
    def __init__(self):
        self.current_mode = 'manual'
        self.mode_changed = False
        self.last_mode = 'manual'
        
        # Create the control window
        self.window_name = "Robot Control Mode"
        self.window_width = 400
        self.window_height = 300
        
        # Button properties
        self.button_width = 150
        self.button_height = 50
        self.button_spacing = 20
        
        # Colors
        self.bg_color = (50, 50, 50)
        self.active_color = (0, 255, 0)
        self.inactive_color = (100, 100, 100)
        self.text_color = (255, 255, 255)
        self.hover_color = (150, 150, 150)
        
        # Mouse callback state
        self.mouse_pos = (0, 0)
        self.mouse_clicked = False
        
        # Button positions
        self.manual_button = {
            'x': (self.window_width - self.button_width) // 2,
            'y': 80,
            'width': self.button_width,
            'height': self.button_height
        }
        
        self.auto_button = {
            'x': (self.window_width - self.button_width) // 2,
            'y': 150,
            'width': self.button_width,
            'height': self.button_height
        }
        
        self.exit_button = {
            'x': (self.window_width - self.button_width) // 2,
            'y': 220,
            'width': self.button_width,
            'height': self.button_height
        }
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Initial display
        self.update_display()
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        self.mouse_pos = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_clicked = True
            
            # Check manual button
            if self.point_in_button(x, y, self.manual_button):
                if self.current_mode != 'manual':
                    self.current_mode = 'manual'
                    self.mode_changed = True
                    print("\n" + "="*50)
                    print("SWITCHING TO MANUAL CONTROL MODE")
                    print("="*50)
                    
            # Check autonomous button
            elif self.point_in_button(x, y, self.auto_button):
                if self.current_mode != 'autonomous':
                    self.current_mode = 'autonomous'
                    self.mode_changed = True
                    print("\n" + "="*50)
                    print("SWITCHING TO AUTONOMOUS MODE")
                    print("="*50)
                    
            # Check exit button
            elif self.point_in_button(x, y, self.exit_button):
                self.current_mode = 'exit'
                self.mode_changed = True
                print("\nExiting simulation...")
    
    def point_in_button(self, x, y, button):
        """Check if point is inside button"""
        return (button['x'] <= x <= button['x'] + button['width'] and
                button['y'] <= y <= button['y'] + button['height'])
    
    def get_button_color(self, button_mode):
        """Get color for button based on current mode and hover state"""
        if self.current_mode == button_mode:
            return self.active_color
        elif button_mode == 'exit':
            # Check if mouse is hovering over exit button
            if self.point_in_button(self.mouse_pos[0], self.mouse_pos[1], self.exit_button):
                return (0, 0, 255)  # Red when hovering
            else:
                return (150, 50, 50)  # Dark red
        elif (button_mode == 'manual' and 
              self.point_in_button(self.mouse_pos[0], self.mouse_pos[1], self.manual_button)) or \
             (button_mode == 'autonomous' and 
              self.point_in_button(self.mouse_pos[0], self.mouse_pos[1], self.auto_button)):
            return self.hover_color
        else:
            return self.inactive_color
    
    def draw_button(self, img, button, text, mode):
        """Draw a button with text"""
        color = self.get_button_color(mode)
        
        # Draw button rectangle
        cv2.rectangle(img, 
                     (button['x'], button['y']),
                     (button['x'] + button['width'], button['y'] + button['height']),
                     color, -1)
        
        # Draw button border
        border_color = (255, 255, 255) if self.current_mode == mode else (200, 200, 200)
        cv2.rectangle(img, 
                     (button['x'], button['y']),
                     (button['x'] + button['width'], button['y'] + button['height']),
                     border_color, 2)
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = button['x'] + (button['width'] - text_size[0]) // 2
        text_y = button['y'] + (button['height'] + text_size[1]) // 2
        
        cv2.putText(img, text, (text_x, text_y), font, font_scale, self.text_color, thickness)
    
    def update_display(self):
        """Update the control window display"""
        # Create blank image
        img = np.full((self.window_height, self.window_width, 3), self.bg_color, dtype=np.uint8)
        
        # Draw title
        title = "Robot Control Mode"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        title_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        title_x = (self.window_width - title_size[0]) // 2
        title_y = 40
        
        cv2.putText(img, title, (title_x, title_y), font, font_scale, self.text_color, thickness)
        
        # Draw buttons
        self.draw_button(img, self.manual_button, "MANUAL", 'manual')
        self.draw_button(img, self.auto_button, "AUTONOMOUS", 'autonomous')
        self.draw_button(img, self.exit_button, "EXIT", 'exit')
        
        # Show current mode status
        status_text = f"Current Mode: {self.current_mode.upper()}"
        status_size = cv2.getTextSize(status_text, font, 0.5, 1)[0]
        status_x = (self.window_width - status_size[0]) // 2
        status_y = self.window_height - 20
        
        cv2.putText(img, status_text, (status_x, status_y), font, 0.5, self.text_color, 1)
        
        # Display the image
        cv2.imshow(self.window_name, img)
    
    def update(self):
        """Update the controller - call this in main loop"""
        self.mouse_clicked = False
        
        # Update display
        self.update_display()
        
        # Process events
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key as backup exit
            self.current_mode = 'exit'
            self.mode_changed = True
        
        # Check if mode changed
        if self.mode_changed:
            self.last_mode = self.current_mode
            self.mode_changed = False
            return True
        
        return False
    
    def get_current_mode(self):
        """Get current mode"""
        return self.current_mode
    
    def should_exit(self):
        """Check if should exit"""
        return self.current_mode == 'exit'
    
    def destroy(self):
        """Clean up the window"""
        cv2.destroyWindow(self.window_name)