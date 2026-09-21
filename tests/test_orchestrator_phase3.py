"""
Unit tests for Phase 3 of PixRevive Two-Tier Orchestrator:
- ActionRegistry mappings and lookups.
- DetailSynthesizerAction anisotropic micro-grain injection.
- DeskewCropAction scanner margin removal.
- NLMDenoiseAction edge-gated filtering.
"""

import os
import sys
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ai.orchestrator.actions.action_registry import ActionRegistry
from ai.orchestrator.actions.detail_action import DetailSynthesizerAction
from ai.orchestrator.actions.geometry_action import DeskewCropAction
from ai.orchestrator.actions.denoise_action import NLMDenoiseAction
from ai.orchestrator.perception.taxonomy import DefectCategory, DefectCandidate
from ai.pipeline.pipeline_context import PipelineContext


class TestOrchestratorPhase3(unittest.TestCase):
    def test_action_registry_mappings(self):
        registry = ActionRegistry()
        actions = registry.list_all_actions()
        self.assertGreaterEqual(len(actions), 7)

        # Check category lookups
        cand_oversmooth = DefectCandidate(category=DefectCategory.OVER_SMOOTHED, confidence=0.8)
        act = registry.get_for_candidate(cand_oversmooth)
        self.assertIsNotNone(act)
        self.assertEqual(act.name, "DetailSynth")

        cand_crack = DefectCandidate(category=DefectCategory.TEAR_CRACK, confidence=0.9)
        act_crack = registry.get_for_candidate(cand_crack)
        self.assertEqual(act_crack.name, "LaMaInpaint")

    def test_detail_synthesizer_increases_hf_energy(self):
        # Create an over-smoothed image (uniform with subtle gradient, zero high-frequency grain)
        smooth_img = np.full((120, 120, 3), 150, dtype=np.uint8)
        for i in range(120):
            smooth_img[i, :, :] = int(140 + i * 0.1)

        context = PipelineContext(smooth_img)
        pre_lap_var = float(cv2.Laplacian(cv2.cvtColor(smooth_img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())

        action = DetailSynthesizerAction()
        passed, score, details = action.execute(context)

        self.assertTrue(passed)
        post_lap_var = float(cv2.Laplacian(cv2.cvtColor(context.working_image, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
        self.assertGreater(post_lap_var, pre_lap_var)
        self.assertIn("Synthesized micro-grain", details)

    def test_deskew_crop_action(self):
        # Create image with solid white border (top=10, left=10)
        img = np.full((100, 100, 3), 120, dtype=np.uint8)
        img[:10, :, :] = 255
        img[:, :10, :] = 255

        context = PipelineContext(img)
        action = DeskewCropAction()
        passed, score, details = action.execute(context)

        self.assertTrue(passed)
        self.assertTrue(context.image_metadata.get("border_cropped", False))
        h, w = context.working_image.shape[:2]
        self.assertLess(h, 100)
        self.assertLess(w, 100)

    def test_nlm_denoise_action(self):
        # Create noisy flat patch
        img = np.full((80, 80, 3), 128, dtype=np.uint8)
        noise = np.random.normal(0, 15, (80, 80, 3)).astype(np.int16)
        noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        context = PipelineContext(noisy_img)
        action = NLMDenoiseAction()
        passed, score, details = action.execute(context)

        self.assertTrue(passed)
        # Variance of denoised image should be substantially lower than input
        input_var = float(np.var(noisy_img))
        output_var = float(np.var(context.working_image))
        self.assertLess(output_var, input_var)


if __name__ == "__main__":
    unittest.main()
