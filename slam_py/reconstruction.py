import numpy as np
import os

class Reconstruction:
    def __init__(self):
        self.points = []
        self.colors = []
        self.keyframe_poses = []
        self.keyframe_ids = []
    
    def add_points(self, points, colors):
        if len(points) > 0:
            self.points.append(points)
            self.colors.append(colors)
    
    def add_keyframe(self, frame_id, pose):

        self.keyframe_ids.append(frame_id)
        self.keyframe_poses.append(pose.copy())
    
    def get_all_points(self):
        if len(self.points) == 0:
            return np.array([]), np.array([])
        
        all_points = np.vstack(self.points)
        all_colors = np.vstack(self.colors)
        
        return all_points, all_colors
    
    def save_ply(self, filename):
        points, colors = self.get_all_points()
        
        if len(points) == 0:
            print("Warning: No points to save!")
            return False
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        try:
            with open(filename, 'w') as f:
                # Write PLY header
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                
                # Write point data
                for point, color in zip(points, colors):
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} ")
                    f.write(f"{int(color[0])} {int(color[1])} {int(color[2])}\n")
            
            print(f"\n Saved {len(points)} points to {filename}")
            return True
            
        except Exception as e:
            print(f"\n Error saving PLY: {e}")
            return False
    
    def save_trajectory(self, filename):
        """Save camera trajectory"""
        if len(self.keyframe_poses) == 0:
            print("Warning: No trajectory to save!")
            return False
        
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        try:
            with open(filename, 'w') as f:
                f.write("# frame_id tx ty tz qx qy qz qw\n")
                
                for frame_id, pose in zip(self.keyframe_ids, self.keyframe_poses):
                    t = pose[:3, 3]
                    R = pose[:3, :3]
                    quat = self._rotation_to_quaternion(R)
                    
                    f.write(f"{frame_id} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} ")
                    f.write(f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")
            
            print(f" Saved trajectory with {len(self.keyframe_poses)} poses to {filename}")
            return True
            
        except Exception as e:
            print(f" Error saving trajectory: {e}")
            return False
    
    def _rotation_to_quaternion(self, R):
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        
        return np.array([qx, qy, qz, qw])
    
    def get_point_count(self):
        """Get total number of points"""
        if len(self.points) == 0:
            return 0
        return sum(len(p) for p in self.points)