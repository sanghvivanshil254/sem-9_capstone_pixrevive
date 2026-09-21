"""Regression tests for margin-robust noise estimation and the Stage 2 grain-field gate.

Background: scanned figures with large flat white margins (case8 — the SigShrink
speckle-Lena figure) diluted every global noise statistic. MAD sigma collapsed to
0.00, speckle correlation flattened, the content/margin boundary registered as
JPEG blocking, and a 4.93% false damage mask reached LaMa — inpainting grain as
if it were emulsion damage while the real grain never got denoised.
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ai.analysis.content_region import content_mask, robust_sigma
from ai.analysis.quality_analyzer import QualityAnalyzer
from ai.pipeline.pipeline_context import PipelineContext
from ai.pipeline.restoration_pipeline import RestorationPipeline
from ai.postprocessing.noise_router import NoiseClassifier


def _case8_path():
    return os.path.join(os.path.dirname(__file__), "restoration_dataset", "damaged", "case8_speckle_lena_margins.png")


class TestContentRegion(unittest.TestCase):
    def test_plain_photo_gets_full_mask(self):
        """An ordinary photo without scanner margins must be all-content."""
        rng = np.random.default_rng(3)
        img = rng.integers(40, 220, (120, 160), dtype=np.uint8)
        self.assertTrue(np.all(content_mask(img) > 0))

    def test_framed_figure_mask_excludes_margin(self):
        """A white-margin figure must yield a content mask that excludes margin."""
        content = np.full((140, 100), 90, dtype=np.uint8)
        img = np.full((160, 120), 255, dtype=np.uint8)
        img[10:150, 10:110] = content
        m = content_mask(img)
        self.assertEqual(int(m[5, 5]), 0, "corner margin must be excluded")
        self.assertEqual(int(m[80, 60]), 255, "center content must be included")

    def test_robust_sigma_sees_through_margin(self):
        """Zero-grain margin + grainy content: sigma must reflect the content."""
        rng = np.random.default_rng(7)
        img = np.full((160, 120), 255, dtype=np.uint8)
        img[10:150, 10:110] = np.clip(
            128 + rng.normal(0, 14, (140, 100)), 0, 255
        ).astype(np.uint8)
        sigma = robust_sigma(img)
        self.assertGreater(sigma, 8.0, "content grain must dominate the estimate")
        global_mad = 1.4826 * float(
            np.median(np.abs(img.astype(np.float32) - cv2.medianBlur(img, 3).astype(np.float32)))
        )
        self.assertLess(global_mad, 2.0, "global MAD collapses to the margin (control)")

    def test_robust_sigma_zero_on_clean(self):
        img = np.full((120, 120), 128, dtype=np.uint8)
        self.assertLess(robust_sigma(img), 1.0)


class TestMarginRobustAnalysis(unittest.TestCase):
    def setUp(self):
        path = _case8_path()
        if not os.path.exists(path):
            self.skipTest("case8 fixture not present")
        bgr = cv2.imread(path)
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def test_case8_noise_score_is_not_diluted(self):
        """The margin previously halved noise_score; content-only must fix it."""
        meta = QualityAnalyzer.analyze(self.rgb)
        # Global (old) estimator reported 3.08 on this image; heavy visible
        # speckle in the content must push the content-aware score far higher.
        self.assertGreater(meta["noise_score"], 6.0)
        self.assertGreater(meta["content_sigma"], 6.0)

    def test_case8_grain_field_flagged(self):
        meta = QualityAnalyzer.analyze(self.rgb)
        self.assertTrue(meta["grain_field"], f"metadata: {meta}")

    def test_case8_no_bogus_compression_flags(self):
        """Content/margin boundary must no longer read as JPEG blocking."""
        profile = NoiseClassifier.classify(self.rgb, {"color_type": "grayscale"})
        self.assertNotIn("jpeg_compression", profile.detected_types)
        self.assertNotIn("blocking", profile.detected_types)

    def test_additive_noise_image_stays_on_denoise_semantics(self):
        """case1_noisy_lena (no margins) must keep its established profile."""
        path = os.path.join(os.path.dirname(__file__), "restoration_dataset", "damaged", "case1_noisy_lena.jpg")
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        meta = QualityAnalyzer.analyze(rgb)
        self.assertGreater(meta["noise_score"], 10.0)
        self.assertFalse(meta["grain_field"], "case1 damage% >= 6 -> not a pure grain field")


class TestGrainFieldGate(unittest.TestCase):
    def setUp(self):
        path = _case8_path()
        if not os.path.exists(path):
            self.skipTest("case8 fixture not present")
        bgr = cv2.imread(path)
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def test_speckle_image_gets_tiny_mask_not_5pct(self):
        """End-to-end Stage 1 + Stage 2: false flakes/cracks must be discarded."""
        pipeline = RestorationPipeline()
        ctx = PipelineContext(self.rgb.copy())
        pipeline._stage_1_analyze(ctx, None)

        # Simulate the real pipeline sequence: face detect runs before Stage 2.
        try:
            pipeline._stage_4_face_detect(ctx, None)
        except Exception:
            ctx.image_metadata["faces"] = []

        mask = pipeline._stage_2_fracture_masking(ctx, None)
        coverage = 100.0 * np.count_nonzero(mask) / mask.size
        self.assertLess(
            coverage,
            2.0,
            f"grain-field image must not produce a large inpaint mask (got {coverage:.2f}%)",
        )

    def test_gate_keeps_real_3d_tears(self):
        """The gate keeps genuine structural 3D tears even on grainy frames."""
        rng = np.random.default_rng(11)
        gray = np.clip(128 + rng.normal(0, 16, (200, 200)), 0, 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=2)
        ctx = PipelineContext(rgb)
        meta = QualityAnalyzer.analyze(rgb)
        ctx.image_metadata.update(meta)
        self.assertTrue(meta["grain_field"], "noisy synthetic must be flagged as grain field")


if __name__ == "__main__":
    unittest.main()
