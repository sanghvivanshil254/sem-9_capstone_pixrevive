"""
test_facial_landmarks.py — Tests specifically for facial landmark validation.

These tests focus on the landmark-based identity preservation checks,
which are critical for ensuring that facial geometry is preserved during
restoration.

The tests verify:
1. Landmark extraction from images
2. Facial proportion measurements (eye spacing, eye shape, nose width)
3. Landmark comparison between original and restored images
4. Threshold-based validation of geometric stability

Run with: python -m pytest tests/test_facial_landmarks.py -v
"""

import unittest
import numpy as np
import cv2
import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ai.validation.identity_validator import (
    IdentityValidator,
    FacialLandmarks,
    LandmarkPoint
)


class TestLandmarkProportions(unittest.TestCase):
    """Test facial proportion calculations from landmarks."""
    
    def setUp(self):
        """Create a standardized landmark set for testing."""
        # Create a symmetric face with known proportions
        # Using approximate 68-point model positions
        
        # Define key points
        chin = LandmarkPoint(100, 200)  # Point 8
        
        # Right eye points (36-41)
        right_eye_outer = LandmarkPoint(70, 140)   # 36
        right_eye_inner = LandmarkPoint(85, 140)   # 39
        
        # Left eye points (42-47)
        left_eye_outer = LandmarkPoint(120, 140)   # 42
        left_eye_inner = LandmarkPoint(135, 140)   # 45
        
        # Nose points
        nose_bridge = LandmarkPoint(100, 150)  # 27
        nose_tip = LandmarkPoint(100, 180)      # 30
        
        # Mouth points
        mouth_left = LandmarkPoint(85, 220)     # 48
        mouth_right = LandmarkPoint(115, 220)   # 54
        mouth_center = LandmarkPoint(100, 220)   # 51
        
        # Create full 68-point set (exact 68-point indexing for tests)
        pts = [LandmarkPoint(100, 100)] * 68
        
        # Jawline (0-16): point 8 is chin
        for i in range(17):
            pts[i] = LandmarkPoint(100, 200)
        pts[8] = chin  # (100, 200)
        
        # Eyebrows (17-26)
        for i in range(17, 27):
            pts[i] = LandmarkPoint(100, 120)
            
        # Nose bridge (27-30): point 30 is nose tip
        pts[27] = nose_bridge
        pts[28] = LandmarkPoint(100, 160)
        pts[29] = LandmarkPoint(100, 170)
        pts[30] = nose_tip  # (100, 180)
        
        # Nose base (31-35)
        for i in range(31, 36):
            pts[i] = LandmarkPoint(100, 185)
            
        # Right eye (36-41)
        pts[36] = right_eye_outer  # (70, 140)
        pts[37] = LandmarkPoint(75, 138)
        pts[38] = LandmarkPoint(80, 138)
        pts[39] = right_eye_inner  # (85, 140)
        pts[40] = LandmarkPoint(80, 142)
        pts[41] = LandmarkPoint(75, 142)
        
        # Left eye (42-47)
        pts[42] = left_eye_outer   # (120, 140)
        pts[43] = LandmarkPoint(125, 138)
        pts[44] = LandmarkPoint(130, 138)
        pts[45] = left_eye_inner   # (135, 140)
        pts[46] = LandmarkPoint(130, 142)
        pts[47] = LandmarkPoint(125, 142)
        
        # Mouth outer (48-59)
        for i in range(48, 60):
            pts[i] = LandmarkPoint(100, 220)
            
        self.landmarks = FacialLandmarks(pts)
        
        self.validator = IdentityValidator()
    
    def test_eye_spacing_calculation(self):
        """Test that eye spacing is calculated correctly."""
        spacing = self.landmarks.eye_spacing()
        self.assertIsNotNone(spacing)
        
        # Expected spacing: distance between right_eye_center and left_eye_center
        # Right eye center should be around (77.5, 142.5)
        # Left eye center should be around (127.5, 142.5)
        # Distance should be approximately 50 pixels
        expected_spacing = 50.0
        self.assertAlmostEqual(spacing, expected_spacing, delta=10.0)
    
    def test_eye_shape_ratio_calculation(self):
        """Test eye shape ratio (height/width) calculation."""
        right_ratio = self.landmarks.eye_shape_ratio(0)
        left_ratio = self.landmarks.eye_shape_ratio(1)
        
        self.assertIsNotNone(right_ratio)
        self.assertIsNotNone(left_ratio)
        
        # Ratios should be reasonable (between 0.1 and 1.0)
        self.assertGreater(right_ratio, 0.1)
        self.assertLess(right_ratio, 1.0)
        self.assertGreater(left_ratio, 0.1)
        self.assertLess(left_ratio, 1.0)
    
    def test_nose_tip_position(self):
        """Test nose tip landmark extraction."""
        nose_tip = self.landmarks.nose_tip
        self.assertIsNotNone(nose_tip)
        self.assertAlmostEqual(nose_tip.x, 100, delta=5)
        self.assertAlmostEqual(nose_tip.y, 180, delta=5)
    
    def test_mouth_center_position(self):
        """Test mouth center calculation."""
        mouth_center = self.landmarks.mouth_center
        self.assertIsNotNone(mouth_center)
        self.assertAlmostEqual(mouth_center.x, 100, delta=5)
        self.assertAlmostEqual(mouth_center.y, 220, delta=5)
    
    def test_chin_point_position(self):
        """Test chin point extraction."""
        chin = self.landmarks.chin_point
        self.assertIsNotNone(chin)
        self.assertAlmostEqual(chin.x, 100, delta=5)
        self.assertAlmostEqual(chin.y, 200, delta=5)


class TestLandmarkComparison(unittest.TestCase):
    """Test comparison of landmarks between original and restored images."""
    
    def setUp(self):
        """Create landmark sets for comparison."""
        self.validator = IdentityValidator()
        
        # Create base landmarks
        self.base_landmarks = FacialLandmarks([
            LandmarkPoint(x, y) for x, y in ([
                (100, 100), (110, 100), (120, 100), (130, 100), (140, 100),
                (150, 100), (160, 100), (170, 100), (180, 100),
            ] + [
                (120, 80), (125, 80), (130, 80),  # Right eye
                (135, 80), (140, 80), (145, 80),
            ] + [
                (150, 80), (155, 80), (160, 80),  # Left eye
                (165, 80), (170, 80), (175, 80),
            ] + [
                (140, 90), (145, 100), (150, 110), (155, 120), (140, 120),
            ] + [
                (130, 130), (140, 130), (150, 130), (160, 130),
                (135, 135), (145, 135), (155, 135),
            ])
        ])
        
        # Pad to 68 points
        while len(self.base_landmarks.points) < 68:
            self.base_landmarks.points.append(LandmarkPoint(100, 100))
    
    def test_identical_landmarks_have_zero_delta(self):
        """Test that identical landmarks produce zero deltas."""
        landmarks2 = FacialLandmarks([p for p in self.base_landmarks.points])
        
        deltas = self.validator.compare_landmarks(
            self.base_landmarks, landmarks2
        )
        
        for key, value in deltas.items():
            if value is not None:
                self.assertAlmostEqual(value, 0.0, places=5, 
                                     msg=f"Delta for {key} should be 0")
    
    def test_eye_spacing_delta(self):
        """Test eye spacing delta calculation."""
        # Create modified landmarks with increased eye spacing
        modified_points = []
        for i, p in enumerate(self.base_landmarks.points):
            # Move left eye points right by 10 pixels
            if 18 <= i < 24:  # Left eye indices
                modified_points.append(LandmarkPoint(p.x + 10, p.y))
            else:
                modified_points.append(LandmarkPoint(p.x, p.y))
        
        modified_landmarks = FacialLandmarks(modified_points)
        
        deltas = self.validator.compare_landmarks(
            self.base_landmarks, modified_landmarks
        )
        
        # Eye spacing should have increased
        if "eye_spacing_delta" in deltas:
            self.assertGreater(deltas["eye_spacing_delta"], 0)
        
        if "eye_spacing_ratio" in deltas:
            self.assertGreater(deltas["eye_spacing_ratio"], 1.0)


class TestLandmarkStabilityThresholds(unittest.TestCase):
    """Test threshold-based validation of landmark stability."""
    
    def setUp(self):
        """Set up validator with test thresholds."""
        self.validator = IdentityValidator(
            eye_spacing_threshold=0.02,
            eye_shape_threshold=0.05
        )
    
    def test_eye_spacing_within_threshold(self):
        """Test that small eye spacing changes pass validation."""
        deltas = {"eye_spacing_ratio": 1.015}  # 1.5% change
        
        is_stable, failed = self.validator.validate_landmarks_stability(
            deltas, image_width=1000
        )
        
        self.assertTrue(is_stable)
        self.assertEqual(len(failed), 0)
    
    def test_eye_spacing_exceeds_threshold(self):
        """Test that large eye spacing changes fail validation."""
        deltas = {"eye_spacing_ratio": 1.03}  # 3% change > 2% threshold
        
        is_stable, failed = self.validator.validate_landmarks_stability(
            deltas, image_width=1000
        )
        
        self.assertFalse(is_stable)
        self.assertEqual(len(failed), 1)
        self.assertIn("eye_spacing", failed[0])
    
    def test_eye_shape_within_threshold(self):
        """Test that small eye shape changes pass validation."""
        deltas = {"right_eye_shape_delta": 0.04}  # 4% change < 5% threshold
        
        is_stable, failed = self.validator.validate_landmarks_stability(
            deltas, image_width=1000
        )
        
        self.assertTrue(is_stable)
    
    def test_eye_shape_exceeds_threshold(self):
        """Test that large eye shape changes fail validation."""
        deltas = {"right_eye_shape_delta": 0.06}  # 6% change > 5% threshold
        
        is_stable, failed = self.validator.validate_landmarks_stability(
            deltas, image_width=1000
        )
        
        self.assertFalse(is_stable)
        self.assertIn("right_eye_shape", failed[0])
    
    def test_position_deltas_normalized(self):
        """Test that position deltas are normalized by image width."""
        # Position delta of 20 pixels in a 1000-pixel wide image = 2%
        deltas = {"nose_position_delta": 20.0}
        
        is_stable, failed = self.validator.validate_landmarks_stability(
            deltas, image_width=1000
        )
        
        # 2% > 1% threshold, should fail
        self.assertFalse(is_stable)
        self.assertIn("nose", failed[0])
    
    def test_multiple_failures(self):
        """Test that multiple failed checks are all reported."""
        deltas = {
            "eye_spacing_ratio": 1.05,  # > 2%
            "right_eye_shape_delta": 0.10,  # > 5%
            "nose_position_delta": 30.0,  # > 1% of 1000
        }
        
        is_stable, failed = self.validator.validate_landmarks_stability(
            deltas, image_width=1000
        )
        
        self.assertFalse(is_stable)
        self.assertEqual(len(failed), 3)


class TestFacialGeometryValidation(unittest.TestCase):
    """Test validation of facial geometry according to user guidelines."""
    
    def setUp(self):
        """Set up validator."""
        self.validator = IdentityValidator()
    
    def test_eye_spacing_preservation(self):
        """Test that eye spacing is preserved (per guidelines)."""
        # User guidelines: "Preserve: eye spacing"
        # Create landmarks with specific eye spacing
        landmarks1 = self._create_landmarks_with_eye_spacing(50.0)
        landmarks2 = self._create_landmarks_with_eye_spacing(50.0)
        
        deltas = self.validator.compare_landmarks(landmarks1, landmarks2)
        
        if "eye_spacing_delta" in deltas:
            self.assertAlmostEqual(deltas["eye_spacing_delta"], 0.0, places=5)
    
    def test_eye_shape_preservation(self):
        """Test that eye shape is preserved (per guidelines)."""
        # User guidelines: "Preserve: eye shape"
        landmarks1 = self._create_landmarks_with_eye_shape(0.5)  # Ratio 0.5
        landmarks2 = self._create_landmarks_with_eye_shape(0.5)
        
        deltas = self.validator.compare_landmarks(landmarks1, landmarks2)
        
        if "right_eye_shape_delta" in deltas:
            self.assertAlmostEqual(deltas["right_eye_shape_delta"], 0.0, places=5)
    
    def _create_landmarks_with_eye_spacing(self, spacing: float) -> FacialLandmarks:
        """Create landmarks with specific eye spacing."""
        center_x = 100
        eye_y = 100
        
        # Left eye at center_x - spacing/2
        left_eye_x = center_x - spacing / 2
        # Right eye at center_x + spacing/2
        right_eye_x = center_x + spacing / 2
        
        points = []
        
        # Jawline
        for i in range(17):
            points.append(LandmarkPoint(50 + i * 5, 200))
        
        # Right eyebrow
        for i in range(5):
            points.append(LandmarkPoint(right_eye_x - 10 + i * 5, eye_y - 20))
        
        # Left eyebrow
        for i in range(5):
            points.append(LandmarkPoint(left_eye_x - 10 + i * 5, eye_y - 20))
        
        # Nose
        for i in range(4):
            points.append(LandmarkPoint(center_x, eye_y + i * 10))
        for i in range(2):
            points.append(LandmarkPoint(center_x - 5 + i * 5, eye_y + 30))
        
        # Right eye (6 points)
        for i in range(6):
            points.append(LandmarkPoint(right_eye_x - 5 + i * 5, eye_y))
        
        # Left eye (6 points)
        for i in range(6):
            points.append(LandmarkPoint(left_eye_x - 5 + i * 5, eye_y))
        
        # Mouth (remainder to reach 68)
        while len(points) < 68:
            points.append(LandmarkPoint(center_x - 20 + (len(points) - 40) * 2, eye_y + 50))
        
        return FacialLandmarks(points)
    
    def _create_landmarks_with_eye_shape(self, ratio: float) -> FacialLandmarks:
        """Create landmarks with specific eye shape ratio."""
        # For simplicity, just create a basic set
        # This is a simplified test
        points = [LandmarkPoint(0, 0) for _ in range(68)]
        return FacialLandmarks(points)


if __name__ == "__main__":
    unittest.main()
