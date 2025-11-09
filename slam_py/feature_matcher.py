import cv2
import numpy as np

class FeatureMatcher:
    def __init__(self, ratio_threshold=0.75, method='FLANN'):
        self.ratio_threshold = ratio_threshold
        self.method = method
        
        if method == 'FLANN':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def match(self, desc1, desc2):
        if len(desc1) == 0 or len(desc2) == 0:
            return []
        
        # Ensure descriptors are float32 for FLANN
        desc1 = desc1.astype(np.float32)
        desc2 = desc2.astype(np.float32)
        
        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches