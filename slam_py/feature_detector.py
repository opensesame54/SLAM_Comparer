import cv2
import numpy as np

class SIFTDetector:
    def __init__(self, n_features=2000, contrast_threshold=0.04, edge_threshold=10, sigma=1.6):
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )
    
    def detect(self, image):
        """Detect SIFT features in image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is None:
            descriptors = np.array([]).reshape(0, 128)
            keypoints = []
        
        return keypoints, descriptors