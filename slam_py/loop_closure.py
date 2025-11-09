import numpy as np
import cv2

class LoopClosureDetector:
    """Detect loop closures for pose graph optimization"""
    
    def __init__(self, feature_matcher, min_matches=50):
        self.feature_matcher = feature_matcher
        self.min_matches = min_matches
        self.keyframes = []
    
    def add_keyframe(self, frame):
        """Add keyframe for loop detection"""
        self.keyframes.append(frame)
    
    def detect_loop(self, current_frame, min_temporal_distance=30):
        
        if len(self.keyframes) < min_temporal_distance:
            return None, None
        
        best_matches = None
        best_frame = None
        best_score = 0
        
        # Check keyframes that are far in time but close in space
        for kf in self.keyframes[:-min_temporal_distance]:
            # Skip if too close in time
            if abs(current_frame.frame_id - kf.frame_id) < min_temporal_distance:
                continue
            
            # Match features
            matches = self.feature_matcher.match(
                kf.descriptors,
                current_frame.descriptors
            )
            
            if len(matches) > best_score and len(matches) >= self.min_matches:
                # Check spatial proximity
                dist = np.linalg.norm(
                    current_frame.get_camera_center() - kf.get_camera_center()
                )
                
                # Only consider if spatially close (within 2 meters)
                if dist < 2.0:
                    best_score = len(matches)
                    best_matches = matches
                    best_frame = kf
        
        if best_frame is not None:
            print(f"   Loop closure detected with frame {best_frame.frame_id} ({best_score} matches)")
        
        return best_frame, best_matches