import numpy as np
import cv2

class Frame:
    
    def __init__(self, frame_id, rgb, depth, camera):
        self.frame_id = frame_id
        self.rgb = rgb.copy()
        self.depth = depth.copy()
        self.camera = camera
        
        # Features
        self.keypoints = []
        self.descriptors = None
        
        # Pose: transformation from world to camera (Tcw)
        self.Tcw = np.eye(4, dtype=np.float64)
        # Inverse pose: transformation from camera to world (Twc)
        self.Twc = np.eye(4, dtype=np.float64)
        
    def set_features(self, keypoints, descriptors):
        """Set keypoints and descriptors"""
        self.keypoints = keypoints
        self.descriptors = descriptors
    
    def set_pose(self, Tcw):
        
        self.Tcw = Tcw.copy()
        self.Twc = np.linalg.inv(Tcw)
    
    def get_camera_center(self):
        """Get camera center in world coordinates"""
        return self.Twc[:3, 3]
    
    def get_rotation(self):
        """Get rotation matrix (world to camera)"""
        return self.Tcw[:3, :3]
    
    def get_translation(self):
        """Get translation vector (world to camera)"""
        return self.Tcw[:3, 3]
    
    def unproject_pixel(self, u, v):
        
        u_int = int(np.round(u))
        v_int = int(np.round(v))
        
        # Check bounds
        if not (0 <= u_int < self.depth.shape[1] and 0 <= v_int < self.depth.shape[0]):
            return None
        
        # Get depth
        z = self.depth[v_int, u_int] / self.camera.depth_scale
        
        # Valid depth check
        if z < 0.3 or z > 5.0:  # Valid range for indoor RGBD
            return None
        
        # Unproject
        x = (u - self.camera.cx) * z / self.camera.fx
        y = (v - self.camera.cy) * z / self.camera.fy
        
        return np.array([x, y, z], dtype=np.float64)
    
    def transform_to_world(self, point_camera):
        
        point_hom = np.append(point_camera, 1.0)
        point_world = self.Twc @ point_hom
        return point_world[:3]
    
    def get_3d_point_world(self, u, v):
        
        point_cam = self.unproject_pixel(u, v)
        if point_cam is None:
            return None
        return self.transform_to_world(point_cam)
    
    def get_color(self, u, v):
        """Get RGB color at pixel (u, v)"""
        v_int = int(np.round(v))
        u_int = int(np.round(u))
        
        if 0 <= v_int < self.rgb.shape[0] and 0 <= u_int < self.rgb.shape[1]:
            # BGR to RGB
            return self.rgb[v_int, u_int][::-1]
        return None