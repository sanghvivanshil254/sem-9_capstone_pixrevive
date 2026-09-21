"""
test_identity_preservation.py — Unit tests for identity preservation validation.

These tests verify that the PixRevive restoration pipeline preserves facial identity
according to the project's highest priority principle.

Test categories:
1. Pixel preservation in clean regions
2. Landmark stability validation
3. Overall identity validation
4. Edge cases and error handling

Run with: python -m pytest tests/test_identity_preservation.py -v
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
    IdentityMetrics,
    FacialLandmarks,
    LandmarkPoint
)
from ai.pipeline.pipeline_context import PipelineContext


class TestLandmarkPoint(unittest.TestCase):
    """Test the LandmarkPoint dataclass."""
    
    def test_distance_calculation(self):
        """Test Euclidean distance between two points."""
        p1 = LandmarkPoint(0, 0)
        p2 = LandmarkPoint(3, 4)
        self.assertAlmostEqual(p1.distance_to(p2), 5.0, places=5)
    
    def test_distance_to_self(self):
        """Test distance from a point to itself is zero."""
        p = LandmarkPoint(10, 20)
        self.assertEqual(p.distance_to(p), 0.0)


class TestFacialLandmarks(unittest.TestCase):
    """Test the FacialLandmarks container."""
    
    def setUp(self):
        """Create a simple set of landmarks for testing."""
        # Simplified 68-point like structure with minimal points
        pts = [LandmarkPoint(100, 100)] * 68
        # Right eye (36-41)
        pts[36] = LandmarkPoint(120, 150)
        pts[37] = LandmarkPoint(125, 148)
        pts[38] = LandmarkPoint(130, 148)
        pts[39] = LandmarkPoint(135, 150)
        pts[40] = LandmarkPoint(130, 152)
        pts[41] = LandmarkPoint(125, 152)
        # Left eye (42-47)
        pts[42] = LandmarkPoint(150, 150)
        pts[43] = LandmarkPoint(155, 148)
        pts[44] = LandmarkPoint(160, 148)
        pts[45] = LandmarkPoint(165, 150)
        pts[46] = LandmarkPoint(160, 152)
        pts[47] = LandmarkPoint(155, 152)
        self.landmarks = FacialLandmarks(pts)
    
    def test_eye_centers_exist(self):
        """Test that eye center properties return points."""
        self.assertIsNotNone(self.landmarks.right_eye_center)
        self.assertIsNotNone(self.landmarks.left_eye_center)
    
    def test_eye_spacing(self):
        """Test eye spacing calculation."""
        spacing = self.landmarks.eye_spacing()
        self.assertIsNotNone(spacing)
        self.assertGreater(spacing, 0)
    
    def test_eye_shape_ratio(self):
        """Test eye shape ratio calculation."""
        right_ratio = self.landmarks.eye_shape_ratio(0)
        left_ratio = self.landmarks.eye_shape_ratio(1)
        self.assertIsNotNone(right_ratio)
        self.assertIsNotNone(left_ratio)


class TestIdentityMetrics(unittest.TestCase):
    """Test the IdentityMetrics dataclass."""
    
    def test_default_values(self):
        """Test that default metrics are initialized correctly."""
        metrics = IdentityMetrics()
        self.assertTrue(metrics.landmarks_stable)
        self.assertTrue(metrics.pixel_preservation_ok)
        self.assertTrue(metrics.identity_preserved)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = IdentityMetrics()
        metrics_dict = metrics.to_dict()
        self.assertIn("identity_preserved", metrics_dict)
        self.assertIn("landmarks_stable", metrics_dict)
        self.assertIn("thresholds", metrics_dict)


class TestIdentityValidator(unittest.TestCase):
    """Test the IdentityValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = IdentityValidator()
        
        # Create simple test images (100x100 RGB)
        self.image_shape = (100, 100, 3)
        
        # Clean image with a face-like pattern
        self.original_image = np.zeros(self.image_shape, dtype=np.uint8)
        self.original_image[:, :, 0] = 200  # Red channel
        self.original_image[:, :, 1] = 150  # Green channel
        self.original_image[:, :, 2] = 100  # Blue channel
        
        # Same image (should pass validation)
        self.identical_image = self.original_image.copy()
        
        # Slightly modified image (should still pass if changes are in damaged areas)
        self.modified_image = self.original_image.copy()
        self.modified_image[50:60, 50:60] = [255, 0, 0]  # Red square
    
    def test_identical_images_pass_validation(self):
        """Test that identical images pass validation."""
        metrics = self.validator.validate(
            self.original_image,
            self.identical_image
        )
        self.assertTrue(metrics.identity_preserved)
    
    def test_pixel_preservation_with_no_damage(self):
        """Test pixel preservation when entire face is clean."""
        # With no damage mask, entire image is considered clean
        # Modified image should fail pixel preservation
        face_bbox = (25, 25, 75, 75)  # Face in center
        
        changed_px, total_px, ratio = self.validator.check_pixel_preservation(
            self.original_image,
            self.modified_image,
            damage_mask=None,  # No mask = entire face is clean
            face_bbox=face_bbox
        )
        
        self.assertGreater(changed_px, 0)
        self.assertEqual(total_px, 50 * 50)  # 50x50 face region
        self.assertGreater(ratio, 0)
    
    def test_pixel_preservation_with_damage_mask(self):
        """Test that damaged areas are excluded from pixel preservation check."""
        face_bbox = (25, 25, 75, 75)
        
        # Create damage mask that covers the modified area
        damage_mask = np.zeros((100, 100), dtype=np.uint8)
        damage_mask[50:60, 50:60] = 255  # Mark modified area as damaged
        
        changed_px, total_px, ratio = self.validator.check_pixel_preservation(
            self.original_image,
            self.modified_image,
            damage_mask=damage_mask,
            face_bbox=face_bbox
        )
        
        # Modified area is in damage mask, so clean area changes should be 0
        # (assuming modification is only in damaged area)
        # In our test, the modification IS in the damaged area
        # So clean_face_px_changed should be 0
        self.assertEqual(changed_px, 0)
    
    def test_landmark_detection(self):
        """Test facial landmark detection on a simple image."""
        # This test may not detect landmarks (depends on image content)
        # but should not crash
        try:
            landmarks = self.validator.detect_facial_landmarks(self.original_image)
            # Either returns landmarks or None
            self.assertIsInstance(landmarks, (FacialLandmarks, type(None)))
        except Exception as e:
            self.fail(f"Landmark detection raised exception: {e}")
    
    def test_comparison_of_identical_landmarks(self):
        """Test comparison of identical landmark sets."""
        landmarks1 = FacialLandmarks([
            LandmarkPoint(0, 0),
            LandmarkPoint(10, 0),
            LandmarkPoint(5, 5),
        ])
        landmarks2 = FacialLandmarks([
            LandmarkPoint(0, 0),
            LandmarkPoint(10, 0),
            LandmarkPoint(5, 5),
        ])
        
        # Pad to minimum for eye spacing calculation
        # Need at least 36 points for eye_center properties
        for _ in range(36):
            landmarks1.points.append(LandmarkPoint(0, 0))
            landmarks2.points.append(LandmarkPoint(0, 0))
        
        deltas = self.validator.compare_landmarks(landmarks1, landmarks2)
        
        # For identical landmarks, all deltas should be 0 or None
        for key, value in deltas.items():
            if value is not None:
                self.assertAlmostEqual(value, 0.0, places=5)
    
    def test_landmark_stability_validation(self):
        """Test landmark stability validation."""
        # Test with eye spacing within threshold
        deltas = {
            "eye_spacing_ratio": 1.01,  # 1% change
            "eye_spacing_delta": 0.5
        }
        
        is_stable, failed = self.validator.validate_landmarks_stability(
            deltas, image_width=100
        )
        self.assertTrue(is_stable)
        self.assertEqual(len(failed), 0)
        
        # Test with eye spacing exceeding threshold
        deltas_exceed = {
            "eye_spacing_ratio": 1.05,  # 5% change > 2% threshold
        }
        
        is_stable, failed = self.validator.validate_landmarks_stability(
            deltas_exceed, image_width=100
        )
        self.assertFalse(is_stable)
        self.assertGreater(len(failed), 0)


class TestIdentityValidatorWithContext(unittest.TestCase):
    """Test IdentityValidator with PipelineContext."""
    
    def setUp(self):
        """Set up test context."""
        self.validator = IdentityValidator()
        
        # Create test images
        self.original_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.working_image = self.original_image.copy()
        
        # Create context
        metadata = {
            "faces": [{"bbox": [25, 25, 75, 75], "confidence": 0.95, "damage_ratio": 0.0}],
            "width": 100,
            "height": 100,
            "color_type": "color"
        }
        
        self.context = PipelineContext(self.original_image, metadata)
        self.context.working_image = self.working_image
        
        # Add damage mask (no damage)
        self.context.damage_masks["final"] = np.zeros((100, 100), dtype=np.uint8)
    
    def test_validate_context_identical_images(self):
        """Test validation with identical images in context."""
        metrics = self.validator.validate_context(self.context)
        
        # With identical images and no damage, identity should be preserved
        self.assertTrue(metrics.pixel_preservation_ok)
    
    def test_validate_context_with_modified_image(self):
        """Test validation with heavily modified image in context."""
        # Modify working image (>20% of face region pixels)
        self.context.working_image[40:80, 40:80] = [255, 0, 0]
        
        metrics = self.validator.validate_context(self.context)
        
        # Modified area is clean (no damage mask), so pixel preservation should fail
        self.assertFalse(metrics.pixel_preservation_ok)
        self.assertFalse(metrics.identity_preserved)


class TestIdentityValidatorEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up validator."""
        self.validator = IdentityValidator()
    
    def test_empty_face_bbox(self):
        """Test with None face bbox."""
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        restored = np.zeros((100, 100, 3), dtype=np.uint8)
        
        changed, total, ratio = self.validator.check_pixel_preservation(
            original, restored, None, None
        )
        
        self.assertEqual(changed, 0)
        self.assertEqual(total, 0)
        self.assertEqual(ratio, 0.0)
    
    def test_mismatched_image_shapes(self):
        """Test with mismatched image shapes."""
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        restored = np.zeros((150, 150, 3), dtype=np.uint8)
        
        # Should handle gracefully
        changed, total, ratio = self.validator.check_pixel_preservation(
            original, restored, None, (25, 25, 75, 75)
        )
        
        # Should return 0, 0, 0 for mismatched shapes
        self.assertEqual(changed, 0)
        self.assertEqual(total, 0)
        self.assertEqual(ratio, 0.0)
    
    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        validator = IdentityValidator(
            eye_spacing_threshold=0.05,
            eye_shape_threshold=0.1,
            pixel_change_threshold=0.01,
            similarity_threshold=0.9
        )
        
        self.assertEqual(validator.eye_spacing_threshold, 0.05)
        self.assertEqual(validator.eye_shape_threshold, 0.1)
        self.assertEqual(validator.pixel_change_threshold, 0.01)


class TestIdentityValidatorIntegration(unittest.TestCase):
    """Integration tests for identity validation."""
    
    def test_full_validation_pipeline(self):
        """Test complete validation pipeline with realistic images."""
        validator = IdentityValidator()
        
        # Create a more realistic test image
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Add a face-like pattern
        # Skin tone
        image[70:130, 70:130] = [200, 170, 140]
        
        # Eyes (darker)
        image[85:95, 85:95] = [100, 80, 60]  # Right eye
        image[85:95, 105:115] = [100, 80, 60]  # Left eye
        
        # Nose (vertical line)
        image[95:115, 95:105] = [150, 120, 90]
        
        # Mouth (horizontal line)
        image[115:125, 90:110] = [120, 80, 60]
        
        # Test identical image
        metrics = validator.validate(image, image.copy())
        
        # Should pass validation
        self.assertTrue(metrics.identity_preserved)
        
        # Test with slight modification (simulate damage repair)
        modified = image.copy()
        modified[88:92, 88:92] = [200, 170, 140]  # Repair right eye
        
        # Create damage mask for the repaired area
        damage_mask = np.zeros((200, 200), dtype=np.uint8)
        damage_mask[88:92, 88:92] = 255
        
        face_bbox = (70, 70, 130, 130)
        
        changed, total, ratio = validator.check_pixel_preservation(
            image, modified, damage_mask, face_bbox
        )
        
        # Modified pixels are in damaged area, so clean area should have 0 changes
        self.assertEqual(changed, 0)


class TestDamageAwareFaceRestorer(unittest.TestCase):
    """Test damage-aware face restoration mechanics."""

    def test_damage_aware_blend_preserves_clean_face(self):
        from ai.restoration.face_restorer import FaceRestorer
        restorer = FaceRestorer()
        img = np.full((100, 100, 3), 120, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)

        # Standalone without loaded weights returns input image safely
        out = restorer._restore_single_face(img, damage_mask=mask, mode="damage_aware")
        self.assertEqual(out.shape, img.shape)
        np.testing.assert_array_equal(out, img)


if __name__ == "__main__":
    # Run tests
    unittest.main()
