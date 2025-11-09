import numpy as np
import cv2
from .frame import Frame
from .feature_detector import SIFTDetector
from .feature_matcher import FeatureMatcher
from .motion_estimator import MotionEstimator
from .point_cloud import PointCloudGenerator
from .reconstruction import Reconstruction
from .loop_closure import LoopClosureDetector

class SLAMSystem:
    
    def __init__(self, camera, config):
        self.camera = camera
        self.config = config
        
        # Initialize components
        self.detector = SIFTDetector(**config.get('sift', {}))
        self.matcher = FeatureMatcher(**config.get('matcher', {}))
        self.motion_estimator = MotionEstimator(camera)
        self.pc_generator = PointCloudGenerator(camera)
        self.reconstruction = Reconstruction()
        self.loop_detector = LoopClosureDetector(self.matcher, min_matches=config.get('loop_min_matches', 50))
        
        # State
        self.last_keyframe = None
        self.current_frame = None
        self.frame_count = 0
        self.tracking_lost_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.keyframes = []
        
        # Config
        self.keyframe_interval = config.get('keyframe_interval', 5)
        self.min_matches = config.get('min_matches', 20)
        self.max_tracking_lost = config.get('max_tracking_lost', 10)
        self.use_dense = config.get('use_dense_cloud', False)
        self.dense_step = config.get('dense_cloud_step', 5)
        self.use_loop_closure = config.get('use_loop_closure', True)
        
        # Stats
        self.n_keyframes = 0
        self.total_points = 0
        self.n_loop_closures = 0
        
        print(f"\n{'='*70}")
        print(f"{'RGB-D SLAM System Initialized':^70}")
        print(f"{'='*70}")
        print(f"  Features per frame:      {config.get('sift', {}).get('n_features', 2000)}")
        print(f"  Min matches:             {self.min_matches}")
        print(f"  Keyframe interval:       {self.keyframe_interval}")
        print(f"  Dense reconstruction:    {self.use_dense}")
        if self.use_dense:
            print(f"  Dense step:              {self.dense_step} pixels")
        print(f"  Loop closure:            {self.use_loop_closure}")
        print(f"{'='*70}\n")
    
    def process_frame(self, rgb, depth):
        """Process a single RGB-D frame"""
        self.frame_count += 1
        
        # Create frame
        frame = Frame(self.frame_count, rgb, depth, self.camera)
        
        # Detect features
        kps, desc = self.detector.detect(rgb)
        frame.set_features(kps, desc)
        
        n_features = len(kps)
        
        # Initialize - FIRST FRAME
        if self.last_keyframe is None:
            print(f"Frame {self.frame_count:4d}: Initializing first keyframe...")
            frame.set_pose(np.eye(4))
            self.last_keyframe = frame
            self.current_frame = frame
            
            # Add first keyframe
            self._add_keyframe(frame)
            self.success_count += 1
            
            print(f"Frame {self.frame_count:4d}: {n_features:4d} features | INIT | KF={self.n_keyframes} | Points={self.total_points}")
            return True
        
        # Track against last keyframe
        matches = self.matcher.match(self.last_keyframe.descriptors, desc)
        n_matches = len(matches)
        
        # Check matches
        if n_matches < self.min_matches:
            self.tracking_lost_count += 1
            self.failed_count += 1
            print(f"Frame {self.frame_count:4d}: {n_features:4d} features | {n_matches:4d} matches | LOST ({self.tracking_lost_count})")
            
            # Re-initialize if lost too long
            if self.tracking_lost_count >= self.max_tracking_lost:
                print(f"  └─> Re-initializing with last pose")
                frame.set_pose(self.current_frame.Tcw if self.current_frame else np.eye(4))
                self.last_keyframe = frame
                self._add_keyframe(frame)
                self.tracking_lost_count = 0
                self.success_count += 1
                return True
            
            return False
        
        # Estimate motion
        success, T_kf_to_cur, inliers = self.motion_estimator.estimate_motion(
            self.last_keyframe, frame, matches
        )
        
        if not success:
            self.tracking_lost_count += 1
            self.failed_count += 1
            print(f"Frame {self.frame_count:4d}: {n_features:4d} features | {n_matches:4d} matches | FAILED (motion)")
            return False
        
        n_inliers = np.sum(inliers)
        self.tracking_lost_count = 0
        self.success_count += 1
        
        # Update pose: Tcw = T_kf_to_cur * Tkf_w
        frame.set_pose(T_kf_to_cur @ self.last_keyframe.Tcw)
        
        # Compute motion stats
        t_norm = np.linalg.norm(T_kf_to_cur[:3, 3])
        R = T_kf_to_cur[:3, :3]
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) * 180 / np.pi
        
        # Check keyframe
        is_keyframe = self._need_new_keyframe(t_norm, angle)
        
        status = "KF" if is_keyframe else "  "
        print(f"Frame {self.frame_count:4d}: {n_features:4d} features | {n_matches:4d} matches | {n_inliers:4d} inliers | t={t_norm:5.3f}m r={angle:4.1f}° | {status} | KFs={self.n_keyframes} | Pts={self.total_points:,}")
        
        if is_keyframe:
            # Check for loop closure
            if self.use_loop_closure and self.n_keyframes > 30:
                loop_frame, loop_matches = self.loop_detector.detect_loop(frame)
                if loop_frame is not None:
                    self._handle_loop_closure(frame, loop_frame, loop_matches)
                    self.n_loop_closures += 1
            
            self._add_keyframe(frame)
            self.last_keyframe = frame
        
        self.current_frame = frame
        return True
    
    def _need_new_keyframe(self, translation, rotation):
        # Always add keyframe if this is frame 1 (after initialization)
        if self.n_keyframes == 0:
            return True
        
        # Regular interval
        if self.frame_count - self.last_keyframe.frame_id >= self.keyframe_interval:
            return True
        
        # Large motion
        if translation > 0.1 or rotation > 8.0:
            return True
        
        return False
    
    def _add_keyframe(self, frame):
        print(f"    >> Adding keyframe {frame.frame_id}...")
        
        try:
            # Generate point cloud
            if self.use_dense:
                print(f"    >> Generating dense point cloud (step={self.dense_step})...")
                points, colors = self.pc_generator.generate_dense(frame, self.dense_step)
            else:
                print(f"    >> Generating sparse point cloud from {len(frame.keypoints)} keypoints...")
                points, colors = self.pc_generator.generate_sparse(frame)
            
            print(f"    >> Generated {len(points)} points")
            
            if len(points) > 0:
                self.reconstruction.add_points(points, colors)
                self.reconstruction.add_keyframe(frame.frame_id, frame.Twc)
                self.n_keyframes += 1
                self.total_points += len(points)
                print(f"    >> Keyframe added successfully! Total: {self.n_keyframes} KFs, {self.total_points:,} points")
            else:
                print(f"    >> WARNING: No points generated from keyframe!")
            
            # Add to loop detector
            self.keyframes.append(frame)
            self.loop_detector.add_keyframe(frame)
            
        except Exception as e:
            print(f"    >> ERROR adding keyframe: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_loop_closure(self, current_frame, loop_frame, matches):
        """Handle loop closure by correcting drift"""
        # Estimate relative pose between loop frames
        success, T_loop, inliers = self.motion_estimator.estimate_motion(
            loop_frame, current_frame, matches
        )
        
        if not success:
            return
        
        # Compute expected pose from loop
        expected_Tcw = T_loop @ loop_frame.Tcw
        
        # Compute drift correction
        T_correction = expected_Tcw @ np.linalg.inv(current_frame.Tcw)
        
        # Apply correction to recent keyframes (last 20)
        correction_window = min(20, len(self.keyframes))
        for i in range(len(self.keyframes) - correction_window, len(self.keyframes)):
            kf = self.keyframes[i]
            # Smoothly apply correction based on distance from loop
            weight = 1.0 - (len(self.keyframes) - i) / correction_window
            
            # Apply weighted correction
            kf.set_pose(self._interpolate_poses(kf.Tcw, T_correction @ kf.Tcw, weight))
    
    def _interpolate_poses(self, T1, T2, alpha):
        # Linear interpolation of translation
        t = (1 - alpha) * T1[:3, 3] + alpha * T2[:3, 3]
        
        # Spherical interpolation of rotation (simplified)
        R = (1 - alpha) * T1[:3, :3] + alpha * T2[:3, :3]
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    def get_reconstruction(self):
        return self.reconstruction
    
    def save_reconstruction(self, path):
        return self.reconstruction.save_ply(path)
    
    def save_trajectory(self, path):
        return self.reconstruction.save_trajectory(path)
    
    def get_statistics(self):
        return {
            'frames': self.frame_count,
            'success': self.success_count,
            'failed': self.failed_count,
            'keyframes': self.n_keyframes,
            'points': self.total_points,
            'loop_closures': self.n_loop_closures
        }
    
    def print_statistics(self):
        success_rate = (self.success_count / self.frame_count * 100) if self.frame_count > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"{'SLAM Statistics':^70}")
        print(f"{'='*70}")
        print(f"  Total frames:            {self.frame_count}")
        print(f"  Successful:              {self.success_count} ({success_rate:.1f}%)")
        print(f"  Failed:                  {self.failed_count} ({100-success_rate:.1f}%)")
        print(f"  Keyframes:               {self.n_keyframes}")
        print(f"  Loop closures:           {self.n_loop_closures}")
        print(f"  Total 3D points:         {self.total_points:,}")
        if self.n_keyframes > 0:
            print(f"  Avg points/keyframe:     {self.total_points // self.n_keyframes:,}")
        print(f"{'='*70}\n")