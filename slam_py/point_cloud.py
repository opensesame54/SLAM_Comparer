import numpy as np

class PointCloudGenerator:
    
    def __init__(self, camera):
        self.camera = camera
    
    def generate_sparse(self, frame):
        """Generate sparse point cloud from keypoints"""
        points = []
        colors = []
        
        for kp in frame.keypoints:
            u, v = kp.pt
            
            # Get 3D point in camera frame
            pt_cam = frame.unproject_pixel(u, v)
            if pt_cam is None:
                continue
            
            # Transform to world
            pt_world = frame.transform_to_world(pt_cam)
            
            # Get color
            color = frame.get_color(u, v)
            if color is None:
                continue
            
            points.append(pt_world)
            colors.append(color)
        
        return np.array(points), np.array(colors)
    
    def generate_dense(self, frame, step=5):
        """Generate dense point cloud"""
        height, width = frame.depth.shape
        
        points = []
        colors = []
        
        for v in range(0, height, step):
            for u in range(0, width, step):
                # Get 3D point in camera frame
                pt_cam = frame.unproject_pixel(u, v)
                if pt_cam is None:
                    continue
                
                # Transform to world
                pt_world = frame.transform_to_world(pt_cam)
                
                # Get color
                color = frame.get_color(u, v)
                if color is None:
                    continue
                
                points.append(pt_world)
                colors.append(color)
        
        return np.array(points), np.array(colors)