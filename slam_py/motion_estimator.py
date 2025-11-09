import numpy as np
import cv2
from scipy.spatial import cKDTree

class MotionEstimator:
    
    def __init__(self, camera):
        self.camera = camera
        
    def estimate_motion(self, frame_ref, frame_cur, matches):
       
        if len(matches) < 10:
            return False, None, None
        
        # Get 3D-3D correspondences
        pts_ref = []
        pts_cur = []
        
        for m in matches:
            pt_ref = frame_ref.unproject_pixel(*frame_ref.keypoints[m.queryIdx].pt)
            pt_cur = frame_cur.unproject_pixel(*frame_cur.keypoints[m.trainIdx].pt)
            
            if pt_ref is not None and pt_cur is not None:
                pts_ref.append(pt_ref)
                pts_cur.append(pt_cur)
        
        if len(pts_ref) < 10:
            return False, None, None
        
        pts_ref = np.array(pts_ref)
        pts_cur = np.array(pts_cur)
        
        # RANSAC + SVD rigid transformation
        best_T = None
        best_inliers = None
        best_score = 0
        
        n_iterations = 500
        inlier_threshold = 0.05  # 5cm
        
        for _ in range(n_iterations):
            # Sample 3 points
            if len(pts_ref) < 3:
                break
            
            idx = np.random.choice(len(pts_ref), 3, replace=False)
            
            # Estimate transformation
            T = self._estimate_rigid_transform(pts_ref[idx], pts_cur[idx])
            
            if T is None:
                continue
            
            # Count inliers
            pts_ref_hom = np.hstack([pts_ref, np.ones((len(pts_ref), 1))])
            pts_transformed = (T @ pts_ref_hom.T).T[:, :3]
            
            errors = np.linalg.norm(pts_transformed - pts_cur, axis=1)
            inliers = errors < inlier_threshold
            n_inliers = np.sum(inliers)
            
            if n_inliers > best_score:
                best_score = n_inliers
                best_inliers = inliers
                best_T = T
        
        if best_score < 10:
            return False, None, None
        
        # Refine with all inliers
        T_refined = self._estimate_rigid_transform(
            pts_ref[best_inliers],
            pts_cur[best_inliers]
        )
        
        return True, T_refined, best_inliers
    
    def _estimate_rigid_transform(self, src, dst):
        
        if len(src) < 3:
            return None
        
        # Compute centroids
        centroid_src = np.mean(src, axis=0)
        centroid_dst = np.mean(dst, axis=0)
        
        # Center points
        src_centered = src - centroid_src
        dst_centered = dst - centroid_dst
        
        # Compute cross-covariance matrix
        H = src_centered.T @ dst_centered
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        
        # Rotation
        R = Vt.T @ U.T
        
        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        
        # Translation
        t = centroid_dst - R @ centroid_src
        
        # Build transformation matrix
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T