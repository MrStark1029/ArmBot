import pybullet as p
import numpy as np
import time
import cv2
from slam import AutonomousNavigator
from pathplanner import AStarPlanner

class RobotController:
    def __init__(self, robot_id, sensor_manager):
        self.robot_id = robot_id
        self.sensor_manager = sensor_manager
        
        # Initialize SLAM
        self.slam = AutonomousNavigator(robot_id, sensor_manager)
        
        # Robot control parameters
        self.forward_speed = 6  # m/s
        self.turning_speed = 5 # rad/s
        self.pos_threshold = 0.2  # meters
        self.angle_threshold = 0.1  # radians
        
        # Navigation state
        self.current_path = []
        self.current_target = None
        self.path_index = 0
        self.planner = None
        
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
            # Update SLAM
            self.slam.step()
            
            # If we don't have a current target or reached it, get new one
            if self.current_target is None or self._reached_target():
                self.current_target = self.slam.get_next_frontier()
                if self.current_target is None:
                    # Check if we have any frontiers at all
                    if len(self.slam.frontiers) == 0:
                        # Try to find new frontiers by marking current position as explored
                        self.slam._update_frontiers()
                        if len(self.slam.frontiers) == 0:
                            print("No frontiers found - exploration complete!")
                            self.running = False
                            return False
                        else:
                            print(f"Found {len(self.slam.frontiers)} new frontiers")
                            self.current_target = self.slam.get_next_frontier()
                    else:
                        print("No reachable frontiers found")
                        # Try next frontier
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
            self.planner = AStarPlanner(self.slam.occupancy_grid)
            
            # Convert world target to grid coordinates
            target_grid_x = int(self.current_target[0] / self.slam.cell_size) + self.slam.map_center
            target_grid_y = int(self.current_target[1] / self.slam.cell_size) + self.slam.map_center
            
            # Get current robot position
            robot_pose = self.slam.get_robot_pose()
            world_pos, grid_pos, _ = robot_pose
            start_x, start_y = grid_pos
            
            # Plan path
            grid_path = self.planner.plan((start_x, start_y), (target_grid_x, target_grid_y))
            
            if not grid_path:
                print("No path found to target")
                return False
            
            # Convert grid path to world coordinates
            self.current_path = []
            for gx, gy in grid_path:
                world_x = (gx - self.slam.map_center) * self.slam.cell_size
                world_y = (gy - self.slam.map_center) * self.slam.cell_size
                self.current_path.append((world_x, world_y))
            
            self.path_index = 0
            return True
            
        except Exception as e:
            print(f"Path planning error: {e}")
            return False
    
    def _follow_path(self):
        """Follow the current path"""
        if self.path_index >= len(self.current_path):
            return
            
        try:
            target = self.current_path[self.path_index]
            
            # Get current robot pose
            robot_pose = self.slam.get_robot_pose()
            world_pos, grid_pos, orn = robot_pose
            rx, ry = world_pos
            current_angle = p.getEulerFromQuaternion(orn)[2]
            
            # Calculate direction to target
            dx = target[0] - rx
            dy = target[1] - ry
            target_angle = np.arctan2(dy, dx)
            
            # Calculate angle difference
            angle_diff = target_angle - current_angle
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            # Check if we reached current waypoint
            if np.hypot(dx, dy) < self.pos_threshold:
                self.path_index += 1
                return
                
            # Control robot movement
            if abs(angle_diff) > self.angle_threshold:
                turn_speed = self.turning_speed * np.sign(angle_diff)
                p.setJointMotorControlArray(
                    self.robot_id,
                    [2, 4, 3, 5], 
                    p.VELOCITY_CONTROL,
                    targetVelocities=[-turn_speed, -turn_speed, turn_speed, turn_speed]
                )
            else:
                p.setJointMotorControlArray(
                    self.robot_id,
                    [2, 4, 3, 5],
                    p.VELOCITY_CONTROL,
                    targetVelocities=[self.forward_speed] * 4
                )
        except Exception as e:
            print(f"Path following error: {e}")
    
    def _reached_target(self):
        """Check if we've reached the current target"""
        if self.current_target is None:
            return True
            
        try:
            robot_pose = self.slam.get_robot_pose()
            world_pos, grid_pos, _ = robot_pose
            rx, ry = world_pos
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