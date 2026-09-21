"""
Unit tests for Phase 2 of PixRevive Two-Tier Orchestrator:
- FeatureBank numeric descriptors & Fourier analysis.
- DefectProposer candidate generation across closed taxonomy.
- SignalCorroborator deterministic gatekeeping (Proposer != Verifier).
"""

import os
import sys
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ai.orchestrator.perception.taxonomy import DefectCategory, DefectCandidate
from ai.orchestrator.perception.feature_bank import FeatureBank
from ai.orchestrator.perception.defect_proposer import DefectProposer
from ai.orchestrator.perception.corroborator import SignalCorroborator


class TestOrchestratorPhase2(unittest.TestCase):
    def test_feature_bank_global_features(self):
        # Create a sharp checkerboard pattern
        checker = np.indices((100, 100)).sum(axis=0) % 2 * 255
        checker_rgb = cv2.cvtColor(checker.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        features = FeatureBank.extract_global_features(checker_rgb)
        self.assertEqual(features["height"], 100)
        self.assertEqual(features["width"], 100)
        self.assertGreater(features["laplacian_var"], 100.0)
        self.assertLess(features["chroma_std"], 1.0)  # Pure monochrome

    def test_feature_bank_periodic_moire_detection(self):
        # Create a synthetic high-frequency periodic grating (simulating halftone)
        x = np.arange(256)
        y = np.arange(256)
        xx, yy = np.meshgrid(x, y)
        grating = ((np.sin(xx * 0.5) + np.sin(yy * 0.5)) * 127 + 128).astype(np.uint8)
        grating_rgb = cv2.cvtColor(grating, cv2.COLOR_GRAY2RGB)

        features = FeatureBank.extract_global_features(grating_rgb)
        # Should exhibit strong FFT peak-to-median ratio
        self.assertGreater(features["fft_peak_ratio"], 4.5)

    def test_corroborator_rejects_smooth_specular_highlight(self):
        # Create realistic smooth specular highlight (e.g. hat ribbon sheen: 160 base, 220 peak)
        img = np.full((100, 100), 160, dtype=np.uint8)
        cv2.circle(img, (50, 50), 20, 220, -1)
        smooth_highlight = cv2.GaussianBlur(img, (25, 25), 6.0)

        # Candidate proposing this as flaking
        mask = (smooth_highlight > 180).astype(np.uint8) * 255
        candidate = DefectCandidate(
            category=DefectCategory.FLAKING_MIRRORING,
            confidence=0.85,
            mask=mask,
        )

        checked = SignalCorroborator.corroborate(smooth_highlight, candidate)
        # Must be rejected because boundary gradient is smooth (< 35.0)
        self.assertFalse(checked.corroborated)
        self.assertIn("smooth boundary", checked.corroboration_reason)

    def test_corroborator_accepts_sharp_emulsion_flake(self):
        # Create sharp step-edge flake (pure 255 on 0 background with zero blur)
        sharp_img = np.zeros((100, 100), dtype=np.uint8)
        sharp_img[40:60, 40:60] = 255
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255

        candidate = DefectCandidate(
            category=DefectCategory.FLAKING_MIRRORING,
            confidence=0.90,
            mask=mask,
        )

        checked = SignalCorroborator.corroborate(sharp_img, candidate)
        # Step-edge has boundary gradient ~255 > 35.0 -> confirmed
        self.assertTrue(checked.corroborated)
        self.assertIn("Step-edge boundary gradient confirmed", checked.corroboration_reason)

    def test_corroborator_crack_geometry(self):
        # 1. Crack line: 2x30 pixels (aspect ratio = 15.0)
        crack_mask = np.zeros((100, 100), dtype=np.uint8)
        crack_mask[10:40, 50:52] = 255
        crack_cand = DefectCandidate(category=DefectCategory.TEAR_CRACK, confidence=0.8, mask=crack_mask)

        checked_crack = SignalCorroborator.corroborate(np.zeros((100, 100), dtype=np.uint8), crack_cand)
        self.assertTrue(checked_crack.corroborated)

        # 2. Circular blob: 20x20 pixels (aspect ratio = 1.0)
        blob_mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(blob_mask, (50, 50), 10, 255, -1)
        blob_cand = DefectCandidate(category=DefectCategory.TEAR_CRACK, confidence=0.8, mask=blob_mask)

        checked_blob = SignalCorroborator.corroborate(np.zeros((100, 100), dtype=np.uint8), blob_cand)
        self.assertFalse(checked_blob.corroborated)
        self.assertIn("blob shape", checked_blob.corroboration_reason)

    def test_defect_proposer_classifies_monochrome_and_resolution(self):
        proposer = DefectProposer()
        small_gray = np.full((300, 300, 3), 128, dtype=np.uint8)

        candidates = proposer.propose(small_gray)
        categories = [c.category for c in candidates]

        self.assertIn(DefectCategory.UNCOLORED, categories)
        self.assertIn(DefectCategory.LOW_RESOLUTION, categories)


if __name__ == "__main__":
    unittest.main()
