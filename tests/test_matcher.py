from slam_py.feature_detector import SIFTFeatureDetector
from slam_py.feature_matcher import FeatureMatcher
from slam_py.frame import Frame
import numpy as np
import unittest

class TestFeatureMatcher(unittest.TestCase):
    def setUp(self):
        self.detector = SIFTFeatureDetector()
        self.matcher = FeatureMatcher()

        # Create dummy frames for testing
        self.frame1 = Frame(rgb_image=np.zeros((100, 100, 3), dtype=np.uint8), depth_image=np.zeros((100, 100), dtype=np.float32))
        self.frame2 = Frame(rgb_image=np.zeros((100, 100, 3), dtype=np.uint8), depth_image=np.zeros((100, 100), dtype=np.float32))

        # Detect features
        self.keypoints1, self.descriptors1 = self.detector.detect_features(self.frame1.rgb_image)
        self.keypoints2, self.descriptors2 = self.detector.detect_features(self.frame2.rgb_image)

    def test_match_features(self):
        matches = self.matcher.match_features(self.descriptors1, self.descriptors2)
        self.assertIsNotNone(matches)
        self.assertGreater(len(matches), 0)

    def test_filter_matches(self):
        matches = self.matcher.match_features(self.descriptors1, self.descriptors2)
        filtered_matches = self.matcher.filter_matches(matches)
        self.assertIsNotNone(filtered_matches)
        self.assertLess(len(filtered_matches), len(matches))

if __name__ == '__main__':
    unittest.main()