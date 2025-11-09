import numpy as np

class Camera:
    def __init__(self, fx, fy, cx, cy, width, height, depth_scale=5000.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.depth_scale = depth_scale
        
    @classmethod
    def from_config(cls, config):
        return cls(
            fx=config['fx'],
            fy=config['fy'],
            cx=config['cx'],
            cy=config['cy'],
            width=config['width'],
            height=config['height'],
            depth_scale=config.get('depth_scale', 5000.0)
        )
    
    def get_intrinsic_matrix(self):
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)