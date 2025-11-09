from slam_py.feature_detector import SIFTFeatureDetector
import unittest
import cv2
import numpy as np

class TestFeatureDetector(unittest.TestCase):

    def setUp(self):
        self.detector = SIFTFeatureDetector()
        self.image = cv2.imread('data/rgb/sample_image.png', cv2.IMREAD_GRAYSCALE)

    def test_detect_features(self):
        keypoints, descriptors = self.detector.detect_features(self.image)
        self.assertIsNotNone(keypoints)
        self.assertIsNotNone(descriptors)
        self.assertGreater(len(keypoints), 0)
        self.assertEqual(descriptors.shape[1], 128)

    def test_get_keypoints(self):
        keypoints, _ = self.detector.detect_features(self.image)
        keypoints_list = self.detector.get_keypoints(keypoints)
        self.assertIsInstance(keypoints_list, list)
        self.assertGreater(len(keypoints_list), 0)

if __name__ == '__main__':
    unittest.main()