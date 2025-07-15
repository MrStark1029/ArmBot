import pybullet as p
import time

# Windows-compatible keyboard input
try:
    import msvcrt
    WINDOWS = True
except ImportError:
    WINDOWS = False


class HuskyTeleopController:
    def __init__(self, husky_id):
        """Initialize the teleop controller for existing Husky robot"""
        self.husky_id = husky_id
        
        # Get joint information and find wheel joints
        self.find_wheel_joints()
        
        # Control parameters
        self.husky_wheel_separation = 0.555  # Distance between left and right wheels
        
        # Speed parameters
        self.max_linear_speed = 8.0
        self.max_angular_speed = 4.0
        self.speed_increment = 0.5
        
        # Current speed settings
        self.current_linear_speed = 4.0
        self.current_angular_speed = 2.0
        
        # Control states
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        
        # Enable/disable state for mode switching
        self.enabled = False
        
        # Key state tracking for smooth movement
        self.key_states = {
            'w': False, 's': False, 'a': False, 'd': False,
            'q': False, 'e': False
        }
        self.last_key_time = time.time()
        
        print("HuskyTeleopController initialized successfully!")
        
    def find_wheel_joints(self):
        """Find and identify wheel joints"""
        num_joints = p.getNumJoints(self.husky_id)
        
        self.left_wheels = []
        self.right_wheels = []
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.husky_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            # Look for wheel joints
            if joint_type == p.JOINT_REVOLUTE and 'wheel' in joint_name.lower():
                if 'left' in joint_name.lower():
                    self.left_wheels.append(i)
                elif 'right' in joint_name.lower():
                    self.right_wheels.append(i)
        
        # If we didn't find wheels by name, try common indices
        if not self.left_wheels and not self.right_wheels:
            possible_configs = [
                ([0, 2], [1, 3]),
                ([2, 4], [3, 5]),
                ([0, 1], [2, 3]),
                ([1, 3], [2, 4]),
            ]
            
            for left_idx, right_idx in possible_configs:
                if all(i < num_joints for i in left_idx + right_idx):
                    self.left_wheels = left_idx
                    self.right_wheels = right_idx
                    break
        
    def get_key(self):
        """Get a single key press (Windows compatible)"""
        if WINDOWS:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                return key
        return None
        
    def set_wheel_velocities(self, left_vel, right_vel):
        """Set wheel velocities for differential drive"""
        wheels = self.left_wheels + self.right_wheels
        target_vels = [left_vel] * len(self.left_wheels) + [right_vel] * len(self.right_wheels)
        
        if wheels:
            p.setJointMotorControlArray(
                self.husky_id,
                wheels,
                p.VELOCITY_CONTROL,
                targetVelocities=target_vels,
                forces=[500] * len(wheels)
            )
            
    def stop(self):
        """Stop all robot movement"""
        self.set_wheel_velocities(0, 0)
        
    def compute_wheel_velocities(self, linear_vel, angular_vel):
        """Compute individual wheel velocities from linear and angular velocities"""
        left_vel = linear_vel - (angular_vel * self.husky_wheel_separation) / 2
        right_vel = linear_vel + (angular_vel * self.husky_wheel_separation) / 2
        return left_vel, right_vel
        
    def update_movement(self):
        """Update robot movement based on current velocities"""
        left_vel, right_vel = self.compute_wheel_velocities(
            self.current_linear_vel, self.current_angular_vel
        )
        self.set_wheel_velocities(left_vel, right_vel)
    
    def process_input(self):
        """Process keyboard input and update robot movement. Returns True if should exit."""
        # Don't process input if not enabled
        if not self.enabled:
            return False
            
        key = self.get_key()
        current_time = time.time()
        movement_updated = False
        
        if key:
            self.last_key_time = current_time
            
            # Movement keys - instant response
            if key.lower() in self.key_states:
                # Reset all keys first
                for k in self.key_states:
                    self.key_states[k] = False
                # Set current key
                self.key_states[key.lower()] = True
                movement_updated = True
            
            # Speed control
            elif key in ['+', '=']:
                self.current_linear_speed = min(self.max_linear_speed, 
                                              self.current_linear_speed + self.speed_increment)
                print(f"Speed increased to: {self.current_linear_speed:.1f} m/s")
            elif key in ['-', '_']:
                self.current_linear_speed = max(0.5, 
                                              self.current_linear_speed - self.speed_increment)
                print(f"Speed decreased to: {self.current_linear_speed:.1f} m/s")
            
            # Stop all movement - instant response
            elif key.lower() == 'x':
                for k in self.key_states:
                    self.key_states[k] = False
                movement_updated = True
            
            # Exit
            elif ord(key) == 27:  # ESC key
                return True
        
        # Auto-stop if no key pressed for 0.1 seconds (reduced from 0.2)
        if current_time - self.last_key_time > 0.05:
            keys_were_active = any(self.key_states.values())
            for k in self.key_states:
                self.key_states[k] = False
            if keys_were_active:
                movement_updated = True
        
        # Always update movement to ensure instant response
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        
        if self.key_states['w']:
            self.current_linear_vel = self.current_linear_speed
        elif self.key_states['s']:
            self.current_linear_vel = -self.current_linear_speed
        
        if self.key_states['a']:
            self.current_linear_vel = self.current_linear_speed * 0
            self.current_angular_vel = self.current_angular_speed * 2
        elif self.key_states['d']:
            self.current_linear_vel = self.current_linear_speed * 0
            self.current_angular_vel = -self.current_angular_speed * 2
        
        if self.key_states['q']:
            self.current_angular_vel = self.current_angular_speed * 1.5
        elif self.key_states['e']:
            self.current_angular_vel = -self.current_angular_speed * 1.5
        
        # Update robot movement immediately
        self.update_movement()
        
        return False  # Don't exit
        
    def print_controls(self):
        """Print control instructions"""
        print("\n" + "="*50)
        print("HUSKY TELEOP CONTROLS")
        print("="*50)
        print("Movement:")
        print("  w/W - Forward")
        print("  s/S - Backward") 
        print("  a/A - Turn Left")
        print("  d/D - Turn Right")
        print("  x/X - Stop")
        print("\nRotation:")
        print("  q/Q - Rotate Left (CCW)")
        print("  e/E - Rotate Right (CW)")
        print("\nSpeed Control:")
        print("  +/= - Increase Speed")
        print("  -/_ - Decrease Speed")
        print("\nOther:")
        print("  ESC - Exit")
        print("="*50)
        print(f"Current Linear Speed: {self.current_linear_speed:.1f} m/s")
        print(f"Current Angular Speed: {self.current_angular_speed:.1f} rad/s")
        print("="*50)
        
    def set_enabled(self, enabled):
        """Enable or disable teleop control"""
        self.enabled = enabled
        if not enabled:
            # Stop all movement when disabled
            self.stop()
            # Reset all key states
            for key in self.key_states:
                self.key_states[key] = False
            self.current_linear_vel = 0.0
            self.current_angular_vel = 0.0
        
    def is_enabled(self):
        """Check if teleop control is enabled"""
        return self.enabled