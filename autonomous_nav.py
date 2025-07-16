import pybullet as p
import numpy as np
import time
import cv2
from slam import TinySLAM
from pathplanner import AStarPlanner

class RobotController:
    def __init__(self, robot_id, sensor_manager):
        self.robot_id = robot_id
        self.sensor_manager = sensor_manager
        
        # Initialize SLAM
        self.slam = TinySLAM(robot_id, sensor_manager)
        
        # Initialize path planner
        self.planner = AStarPlanner(self.slam)
        
        # Robot control parameters
        self.forward_speed = 1.5  # m/s - much slower for stability
        self.turning_speed = 1.0  # rad/s - much slower for stability
        self.pos_threshold = 0.25  # meters - increased to avoid getting stuck
        self.angle_threshold = 0.15  # radians - tighter threshold
        
        # Add control state tracking
        self.last_command_time = time.time()
        self.min_command_interval = 0.1  # Minimum time between commands
        self.reorient_mode = False  # Track if we're in reorientation mode
        self.reorient_start_time = 0
        
        # Navigation state
        self.current_path = []
        self.current_target = None
        self.path_index = 0
        
        # Control state for mode switching
        self.enabled = False
        self.running = False
        
    def set_enabled(self, enabled):
        """Enable or disable autonomous navigation"""
        self.enabled = enabled
        if not enabled:
            self.stop()
            self.running = False
        else:
            self.running = True
            
    def is_enabled(self):
        """Check if autonomous navigation is enabled"""
        return self.enabled
        
    def is_running(self):
        """Check if autonomous navigation is currently running"""
        return self.running
        
    def update(self):
        """Main update loop for autonomous navigation"""
        if not self.enabled:
            return True
        
        try:
            # Update SLAM with current path for visualization
            self.slam.step(self.current_path, self.path_index)
            
            # If we don't have a current target or reached it, get new one
            if self.current_target is None or self._reached_target():
                self.current_target = self.slam.get_next_frontier()
                if self.current_target is None:
                    if len(self.slam.frontiers) == 0:
                        print("No frontiers found - exploration complete!")
                        self.running = False
                        return False
                    else:
                        print("No reachable frontiers found")
                        return True
                
                if self.current_target:
                    print(f"New target: {self.current_target}")
                    
                    # Plan path to new target
                    if not self._plan_path_to_target():
                        print("Failed to plan path, looking for new target")
                        self.current_target = None
                        return True
            
            # Follow current path
            if self.current_path and len(self.current_path) > 0:
                self._follow_path()
            else:
                print("No current path available")
                self.current_target = None
                
        except Exception as e:
            print(f"Navigation error: {e}")
            self.current_target = None
            self.current_path = []
            
        return True
        
    def _plan_path_to_target(self):
        """Plan a path to current target using A*"""
        try:
            # Convert world target to grid coordinates
            target_grid_x, target_grid_y = self.slam.world_to_grid(
                self.current_target[0], self.current_target[1]
            )
            
            # Get current robot position in grid coordinates
            robot_x, robot_y, _ = self.slam.get_robot_pose()
            start_grid_x, start_grid_y = self.slam.world_to_grid(robot_x, robot_y)
            
            print(f"Planning from grid ({start_grid_x}, {start_grid_y}) to ({target_grid_x}, {target_grid_y})")
            
            # Plan path
            grid_path = self.planner.plan((start_grid_x, start_grid_y), (target_grid_x, target_grid_y))
            
            if not grid_path:
                print("No path found to target")
                return False
            
            # Convert grid path to world coordinates
            self.current_path = []
            for gx, gy in grid_path:
                world_x, world_y = self.slam.grid_to_world(gx, gy)
                self.current_path.append((world_x, world_y))
            
            # Smooth path by removing waypoints that are too close together
            if len(self.current_path) > 2:
                smoothed_path = [self.current_path[0]]  # Keep start
                for i in range(1, len(self.current_path) - 1):
                    prev_x, prev_y = smoothed_path[-1]
                    curr_x, curr_y = self.current_path[i]
                    distance = np.hypot(curr_x - prev_x, curr_y - prev_y)
                    if distance > 0.3:  # Only keep waypoints at least 30cm apart
                        smoothed_path.append((curr_x, curr_y))
                smoothed_path.append(self.current_path[-1])  # Keep end
                self.current_path = smoothed_path
            
            self.path_index = 0
            print(f"Planned path with {len(self.current_path)} waypoints (smoothed from {len(grid_path)})")
            return True
            
        except Exception as e:
            print(f"Path planning error: {e}")
            return False
    
    def _follow_path(self):
        """Follow the current path"""
        if self.path_index >= len(self.current_path):
            print("Reached end of path")
            return
            
        # Rate limit commands to prevent oscillation
        current_time = time.time()
        if current_time - self.last_command_time < self.min_command_interval:
            return
        self.last_command_time = current_time
            
        try:
            target = self.current_path[self.path_index]
            
            # Get current robot pose
            rx, ry, current_angle = self.slam.get_robot_pose()
            
            # Calculate direction to target
            dx = target[0] - rx
            dy = target[1] - ry
            distance = np.hypot(dx, dy)
            target_angle = np.arctan2(dy, dx)
            
            # Calculate angle difference
            angle_diff = target_angle - current_angle
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            # Debug output (less frequent)
            if self.path_index % 5 == 0 or distance < self.pos_threshold:
                print(f"Waypoint {self.path_index}/{len(self.current_path)-1}: "
                      f"Target: ({target[0]:.2f}, {target[1]:.2f}), "
                      f"Robot: ({rx:.2f}, {ry:.2f}), "
                      f"Distance: {distance:.2f}m, "
                      f"Angle diff: {np.degrees(angle_diff):.1f}°")
            
            # Check if we reached current waypoint
            if distance < self.pos_threshold:
                print(f"Reached waypoint {self.path_index}, moving to next")
                self.path_index += 1
                self.reorient_mode = False  # Reset reorient mode
                return
                
            # Handle reorientation logic
            if distance < 0.5 and abs(angle_diff) > np.pi/2:
                if not self.reorient_mode:
                    print("Starting reorientation")
                    self.reorient_mode = True
                    self.reorient_start_time = current_time
                
                # If we've been reorienting for too long, skip this waypoint
                if current_time - self.reorient_start_time > 3.0:  # 3 seconds timeout
                    print("Reorientation timeout, skipping to next waypoint")
                    self.path_index += 1
                    self.reorient_mode = False
                    return
                
                # Turn in place to face the target
                turn_speed = 0.3 * np.sign(angle_diff)  # Slow, constant turn speed
                print(f"Reorienting: {np.degrees(angle_diff):.1f}° at speed {turn_speed:.2f}")
                p.setJointMotorControlArray(
                    self.robot_id,
                    [2, 4, 3, 5],
                    p.VELOCITY_CONTROL,
                    targetVelocities=[-turn_speed, -turn_speed, turn_speed, turn_speed]
                )
                return
            else:
                self.reorient_mode = False  # Exit reorient mode
                
            # Control robot movement with smoother logic
            if abs(angle_diff) > self.angle_threshold:
                # Need to turn towards target - use gentler proportional control
                turn_speed = min(abs(angle_diff) * 0.5, self.turning_speed) * np.sign(angle_diff)
                if self.path_index % 10 == 0:  # Reduce debug spam
                    print(f"Turning: {np.degrees(angle_diff):.1f}° at speed {turn_speed:.2f}")
                p.setJointMotorControlArray(
                    self.robot_id,
                    [2, 4, 3, 5], 
                    p.VELOCITY_CONTROL,
                    targetVelocities=[-turn_speed, -turn_speed, turn_speed, turn_speed]
                )
            else:
                # Move forward towards target with distance-based speed control
                speed = min(distance * 2, self.forward_speed)  # Gentler speed scaling
                speed = max(speed, 0.3)  # Minimum speed to overcome friction
                if self.path_index % 10 == 0:  # Reduce debug spam
                    print(f"Moving forward at speed {speed:.2f}")
                p.setJointMotorControlArray(
                    self.robot_id,
                    [2, 4, 3, 5],
                    p.VELOCITY_CONTROL,
                    targetVelocities=[speed, speed, speed, speed]
                )
        except Exception as e:
            print(f"Path following error: {e}")
            import traceback
            traceback.print_exc()
    
    def _reached_target(self):
        """Check if we've reached the current target"""
        if self.current_target is None:
            return True
            
        try:
            rx, ry, _ = self.slam.get_robot_pose()
            dx = self.current_target[0] - rx
            dy = self.current_target[1] - ry
            
            return np.hypot(dx, dy) < self.pos_threshold
        except Exception as e:
            print(f"Target check error: {e}")
            return True
    
    def stop(self):
        """Stop robot movement and clear navigation state"""
        p.setJointMotorControlArray(
            self.robot_id,
            [2, 4, 3, 5],
            p.VELOCITY_CONTROL,
            targetVelocities=[0] * 4
        )
        
        self.running = False
        self.current_path = []
        self.current_target = None
        self.path_index = 0
        self.reorient_mode = False