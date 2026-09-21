"""
Unit and integration tests for Forensic Photo Restoration Validator.

Verifies:
1. Constraint 1: Flat planar geometry & border invariance (no 3D canvas wrap/bending).
2. Constraint 2: Facial topology & highlight balance (no duplicated/merged features or blown specularities).
3. Constraint 3: Structural continuity (no hat brim melting, no steel-wool/wire hair).
4. Constraint 4: Regional chromatic isolation (no global pink/mauve wash over background/hat).
5. Constraint 5: Structural invariance & anti-hallucination (zero new spurious objects/shapes).
6. Constraint 6: Background straight edges (left post strictly vertical, background flat).
7. End-to-end forensic restoration on case1_noisy_lena.jpg satisfies all 6 constraints.
"""

import os
import cv2
import numpy as np
import pytest
import unittest

from ai.validation.forensic_validator import ForensicValidator, ConstraintCheckResult, ForensicAuditResult
from ai.pipeline.restoration_pipeline import RestorationPipeline


class TestForensicValidatorUnit(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 196x148 test canvas with known geometry
        self.h_in, self.w_in = 148, 196
        self.raw_img = np.ones((self.h_in, self.w_in, 3), dtype=np.uint8) * 128

        # Draw straight vertical line on left (x=30)
        cv2.line(self.raw_img, (30, 0), (30, self.h_in - 1), (30, 30, 30), 3)

        # Draw a synthetic face in center
        cv2.ellipse(self.raw_img, (100, 75), (25, 32), 0, 0, 360, (180, 160, 150), -1)

    def test_constraint_1_flat_planar_geometry_passes_on_valid_scale(self):
        # 4x upscale (592x784) with exact same aspect ratio
        scaled_out = cv2.resize(self.raw_img, (784, 592), interpolation=cv2.INTER_LINEAR)
        res = ForensicValidator.validate_plane_geometry(self.raw_img, scaled_out)
        self.assertTrue(res.passed, f"Valid scaled image failed constraint 1: {res.details}")
        self.assertGreaterEqual(res.score, 0.90)

    def test_constraint_1_rejects_aspect_warp_or_white_mockup_border(self):
        # Warped aspect ratio (e.g. 592x600 instead of 592x784)
        bad_aspect = cv2.resize(self.raw_img, (600, 592), interpolation=cv2.INTER_LINEAR)
        res_aspect = ForensicValidator.validate_plane_geometry(self.raw_img, bad_aspect)
        self.assertFalse(res_aspect.passed, "Warped aspect ratio should fail constraint 1")

        # Fake white canvas border mockup
        with_border = cv2.resize(self.raw_img, (784, 592), interpolation=cv2.INTER_LINEAR)
        with_border[:10, :] = 255
        with_border[-10:, :] = 255
        with_border[:, :10] = 255
        with_border[:, -10:] = 255
        res_border = ForensicValidator.validate_plane_geometry(self.raw_img, with_border)
        self.assertFalse(res_border.passed, "White mockup border should fail constraint 1")

    def test_constraint_2_facial_topology_rejects_blown_specular_burnout(self):
        h, w = 592, 784
        img = np.ones((h, w, 3), dtype=np.uint8) * 150
        # Create face zone with severe blown-out white highlight burnout (> 248)
        fx0, fy0, fx1, fy1 = int(w * 0.35), int(h * 0.25), int(w * 0.75), int(h * 0.75)
        img[fy0:fy1, fx0:fx1] = 255  # 100% blown specular highlights
        res = ForensicValidator.validate_facial_topology(self.raw_img, img, face_bbox=[fx0, fy0, fx1, fy1])
        self.assertFalse(res.passed, "Blown-out facial highlights must fail constraint 2")

    def test_constraint_3_structural_continuity_passes_on_wavy_hair(self):
        h, w = 592, 784
        img = np.ones((h, w, 3), dtype=np.uint8) * 120
        # Draw smooth hat brim curve
        cv2.ellipse(img, (int(w * 0.5), int(h * 0.3)), (int(w * 0.35), int(h * 0.12)), 0, 0, 360, 190, 4)
        # Add smooth texture in hair region (Laplacian variance ~300)
        hair_zone = img[int(h * 0.35):int(h * 0.85), int(w * 0.18):int(w * 0.52)]
        noise = np.random.normal(0, 15, hair_zone.shape).astype(np.int16)
        img[int(h * 0.35):int(h * 0.85), int(w * 0.18):int(w * 0.52)] = np.clip(hair_zone.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        res = ForensicValidator.validate_structure_continuity(self.raw_img, img)
        self.assertTrue(res.passed, f"Natural structure failed constraint 3: {res.details}")

    def test_constraint_4_rejects_global_pink_mauve_wash(self):
        h, w = 200, 200
        # Create image with uniform pink/magenta wash in LAB (A=145, B=115)
        pink_lab = np.zeros((h, w, 3), dtype=np.uint8)
        pink_lab[:, :, 0] = 160  # L
        pink_lab[:, :, 1] = 145  # A (strong magenta/pink)
        pink_lab[:, :, 2] = 115  # B (blue/cold)
        pink_rgb = cv2.cvtColor(pink_lab, cv2.COLOR_LAB2RGB)

        res = ForensicValidator.validate_chromatic_isolation(pink_rgb)
        self.assertFalse(res.passed, "Uniform pink/mauve wash should fail constraint 4")

    def test_constraint_6_straight_edges_validates_vertical_post(self):
        h, w = 400, 400
        img = np.ones((h, w, 3), dtype=np.uint8) * 150
        # Draw clean straight vertical post on left
        cv2.line(img, (50, 0), (50, h - 1), (20, 20, 20), 4)

        res = ForensicValidator.validate_straight_edges(img)
        self.assertTrue(res.passed, f"Straight post failed constraint 6: {res.details}")
        self.assertGreaterEqual(res.score, 0.85)


class TestEndToEndForensicRestoration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img_path = os.path.join(
            os.path.dirname(__file__), "restoration_dataset", "damaged", "case1_noisy_lena.jpg"
        )
        if not os.path.exists(cls.img_path):
            raise FileNotFoundError(f"Missing {cls.img_path}")

        cls.pipeline = RestorationPipeline()
        cls.pipeline._models_ready.wait(timeout=30.0)
        cls.pipeline._colorizer_ready.wait(timeout=30.0)

    def test_case1_noisy_lena_satisfies_all_six_hard_constraints(self):
        """
        Verify that restoring case1_noisy_lena.jpg satisfies all 6 Hard Constraints:
        1. Flat planar geometry (784x592, exact 4:3 aspect ratio).
        2. Facial topology (symmetric eyes, single nose, no highlight burnout).
        3. Structural continuity (hat brim not melted, hair not wire/steel-wool).
        4. Chromatic isolation (no uniform pink wash, warm skin, neutral background).
        5. Structural invariance (zero spurious blobs, macro correlation >= 0.85).
        6. Straight edges (left vertical post orientation 90 +/- 2 deg).
        """
        bgr = cv2.imread(self.img_path)
        raw_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        restored_rgb = self.pipeline.restore(
            image_input=raw_rgb,
            enable_repair=True,
            enable_denoise=True,
            enable_colorization=True,
            enable_face_restore=True,
            enable_sr=True,
        )

        h_out, w_out = restored_rgb.shape[:2]
        self.assertEqual((h_out, w_out), (592, 784), "Super-resolution must produce exact 4x canvas (592x784)")

        # Run forensic audit
        audit = ForensicValidator.audit(raw_rgb, restored_rgb)
        print("\n" + audit.format_report())

        for cr in audit.constraint_results:
            self.assertTrue(
                cr.passed,
                f"Constraint {cr.constraint_id} [{cr.name}] failed: score={cr.score:.2f} < {cr.threshold:.2f}. "
                f"Details: {cr.details}"
            )

        self.assertTrue(audit.passed, "Forensic audit overall status must be PASS")
        self.assertGreaterEqual(audit.overall_fidelity_score, 0.80, "Fidelity score must be >= 0.80")
