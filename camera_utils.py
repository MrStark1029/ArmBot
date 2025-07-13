import pybullet as p
import numpy as np

class CameraManager:
    def __init__(self):
        # Navigation Camera Parameters
        self.nav_camera_height = 0.6
        self.nav_camera_offset = [0.4, 0.0, self.nav_camera_height]
        self.nav_img_width = 1280
        self.nav_img_height = 720
        self.nav_fov = 85
        
        # Manipulation Camera Parameters
        self.manip_camera_height = 0.1
        self.manip_camera_offset = [0.1, 0.0, self.manip_camera_height]
        self.manip_img_width = 640
        self.manip_img_height = 480
        self.manip_fov = 60
        
        # Common Parameters
        self.near_plane = 0.05
        self.far_plane = 50.0
    
    def get_camera_image(self, robot_pos, robot_orn, camera_type="navigation"):
        """Get camera image based on robot pose and camera type"""
        # Select camera parameters based on type
        if camera_type == "navigation":
            camera_offset = self.nav_camera_offset
            img_width = self.nav_img_width
            img_height = self.nav_img_height
            fov = self.nav_fov
        else:  # manipulation
            camera_offset = self.manip_camera_offset
            img_width = self.manip_img_width
            img_height = self.manip_img_height
            fov = self.manip_fov
        
        # Calculate camera position and orientation
        rotation_matrix = np.array(p.getMatrixFromQuaternion(robot_orn)).reshape(3, 3)
        camera_pos_world = np.array(robot_pos) + rotation_matrix @ np.array(camera_offset)
        
        # Camera looks forward
        target_offset = np.array([1.0, 0.0, 0.0])
        camera_target = camera_pos_world + rotation_matrix @ target_offset
        camera_up = rotation_matrix @ np.array([0, 0, 1])
        
        # Compute view and projection matrices
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos_world,
            cameraTargetPosition=camera_target,
            cameraUpVector=camera_up
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=img_width/img_height,
            nearVal=self.near_plane,
            farVal=self.far_plane
        )
        
        # Get camera image
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=img_width,
            height=img_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Process image data
        rgb_array = np.array(rgb_img).reshape(height, width, 4)[:, :, :3]
        
        # Convert depth buffer to real depths
        depth_buffer = np.array(depth_img).reshape(height, width)
        depth_real = self.far_plane * self.near_plane / (
            self.far_plane - (self.far_plane - self.near_plane) * depth_buffer)
        
        return rgb_array, depth_real, seg_img
