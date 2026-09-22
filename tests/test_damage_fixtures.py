"""Regression tests for the foxing/mold detector wiring (T-1) and the four
user benchmark cases adopted as fixtures (case9-case12).

Background:
  * Foxing/mold (T-1): every classical detector was tuned for *bright*
    emulsion flakes and dark cracks; scattered small dark blemishes (foxing,
    mold speckles, ink dots) never reached the damage mask. Stage 2 now ORs
    EnhancedDamageDetector.detect_dark_spots() into the combined mask — but
    only when the grain-field gate did not claim the frame and only when the
    dark-spot coverage stays under a cap, because black-hat filtering lights
    up whole shadow regions and dark backgrounds.
  * case9 (teared sepia) is the heavy-regeneration path: ~35% refined mask,
    it must reach LaMa inpainting, not the denoise path.
  * case10 (spiderweb creases) and case11 (minor creases) guard the mid and
    light ends of the crease-detection range.
  * case8's speckle/grain routing is covered by test_margin_robust_noise.py.
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ai.damage.enhanced_detector import EnhancedDamageDetector
from ai.pipeline.pipeline_context import PipelineContext
from ai.pipeline.restoration_pipeline import RestorationPipeline

DAMAGED_DIR = os.path.join(os.path.dirname(__file__), "restoration_dataset", "damaged")


def _case_path(name):
    return os.path.join(DAMAGED_DIR, name)


def _load_rgb(name):
    bgr = cv2.imread(_case_path(name))
    if bgr is None:
        raise FileNotFoundError(_case_path(name))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _mask_coverage_pct(mask):
    return 100.0 * float(np.count_nonzero(mask)) / float(mask.size)


class _Stage2Case:
    """Shared Stage 1 -> face-detect -> Stage 2 harness for fixture cases.

    The pipeline is built with device="cpu": a second full CUDA pipeline in
    the pytest process OOMs the 4 GB GPU during model loading (the CUDA
    pipeline in image_service.py plus RetinaFace etc. already occupy it).
    Stage 1-2 detectors are classical OpenCV, so CPU changes nothing here.
    """

    pipeline = None

    @classmethod
    def run_stage2(cls, rgb):
        if cls.pipeline is None:
            cls.pipeline = RestorationPipeline(device="cpu")
        ctx = PipelineContext(rgb.copy())
        cls.pipeline._stage_1_analyze(ctx, None)
        try:
            cls.pipeline._stage_4_face_detect(ctx, None)
        except Exception:
            ctx.image_metadata["faces"] = []
        mask = cls.pipeline._stage_2_fracture_masking(ctx, None)
        return ctx, mask


class TestFoxingWiring(unittest.TestCase):
    """T-1: dark-spot (foxing/mold) detection must reach the final mask."""

    def setUp(self):
        self.path = _case_path("case12_foxing_spots.png")
        if not os.path.exists(self.path):
            self.skipTest("case12 foxing fixture not present")

    def test_foxing_fixture_spots_reach_final_mask(self):
        """Seeded foxing spots must land in the final Stage 2 mask."""
        rgb = _load_rgb("case12_foxing_spots.png")
        ctx, mask = _Stage2Case.run_stage2(rgb)

        coverage = _mask_coverage_pct(mask)
        self.assertGreater(
            coverage, 0.15, f"foxing spots missing from final mask ({coverage:.3f}%)"
        )
        self.assertIn(
            "dark_spots", ctx.damage_masks, "dark-spot sub-mask must be recorded"
        )
        # The 2% cap applies to the foxing *contribution*, not the whole mask:
        # crack/flake detectors legitimately fire on the same frame too.
        spots_pct = _mask_coverage_pct(ctx.damage_masks["dark_spots"])
        self.assertLessEqual(
            spots_pct, 2.0, f"dark-spot contribution must respect the cap ({spots_pct:.2f}%)"
        )

    def _speckled_content_rgb(self):
        """Light base with a moderate field of 1 px dark dots (fabric weave /
        sand / stipple texture analog) at ~0.5% dot density.

        Every dot passes the per-component filters (round, small, dark), so
        only the whole-coverage cap can reject it. Note the detector's
        mean+3.5*sigma threshold makes it self-protecting against *very*
        dense speckle (>=2% dot density raises the threshold above every
        response and the detector goes silent) — the cap exists for exactly
        this moderate-density window, where each dot is individually
        "confident" and coverage is the only discriminator.
        """
        rng = np.random.default_rng(9)
        h, w = 320, 420
        base = np.full((h, w), 185, dtype=np.float32)
        base += rng.normal(0, 2.0, (h, w))
        img = np.clip(base, 0, 255).astype(np.uint8)
        for _ in range(int(h * w * 0.005)):
            cv2.circle(
                img,
                (int(rng.integers(0, w)), int(rng.integers(0, h))),
                1,
                int(rng.integers(30, 55)),
                -1,
                lineType=cv2.LINE_8,
            )
        return np.stack([img, img, img], axis=2)

    def test_moderate_density_speckle_content_saturates_detector(self):
        """Speckled content in the cap's protection window (~0.5% dot density)
        must exceed the 2% coverage cap — the FP class the cap exists for.

        Every individual dot passes the per-component filters (round, small,
        dark), so coverage is the only discriminator: woven/sanded/stippled
        material is not foxing and must not be inpainted.
        """
        rgb = self._speckled_content_rgb()
        detector = EnhancedDamageDetector()
        ctx = PipelineContext(rgb)
        spots = detector.detect_dark_spots(rgb, ctx)
        self.assertGreater(
            _mask_coverage_pct(spots),
            2.0,
            "moderate-density speckle must saturate the detector (cap justification)",
        )

    def test_speckled_content_does_not_get_dark_spot_contribution(self):
        """End-to-end Stage 2: the foxing gate must drop the speckle mask
        (same moderate-density regime the detector test above establishes)."""
        rgb = self._speckled_content_rgb()

        ctx, _mask = _Stage2Case.run_stage2(rgb)
        self.assertNotIn(
            "dark_spots",
            ctx.damage_masks,
            "shadow false positive must be discarded by the foxing coverage gate",
        )


class TestBenchmarkCase9TearedSepia(unittest.TestCase):
    """case9 = teared.jpg: heavy-regeneration path (plan probe: ~35.7% mask)."""

    def test_heavy_tears_produce_large_inpaint_mask(self):
        rgb = _load_rgb("case9_teared_sepia.jpg")
        ctx, mask = _Stage2Case.run_stage2(rgb)

        coverage = _mask_coverage_pct(mask)
        self.assertGreater(
            coverage, 10.0, f"heavy tears must route to inpainting (got {coverage:.2f}%)"
        )
        self.assertLess(coverage, 60.0, "mask should not saturate the whole frame")
        self.assertIn("lama_mask", ctx.damage_masks, "LaMa-tier split must exist")

    def test_tears_survive_the_grain_gate(self):
        """Even when the grain gate fires on this sepia scan, the heavy mask
        must survive: structural tears are kept by design and the sepia
        TearmarkDetector re-adds genuine emulsion flakes after the gate.
        """
        rgb = _load_rgb("case9_teared_sepia.jpg")
        _ctx, mask = _Stage2Case.run_stage2(rgb)
        self.assertGreater(
            _mask_coverage_pct(mask),
            10.0,
            "tearmark + kept structural tears must still yield a heavy-regen mask",
        )


class TestBenchmarkCase10SpiderwebCreases(unittest.TestCase):
    """case10 = image1.jpeg: spiderweb creases (plan probe: ~14.6% mask)."""

    def test_spiderweb_creases_detected(self):
        rgb = _load_rgb("case10_spiderweb_lips.jpg")
        _ctx, mask = _Stage2Case.run_stage2(rgb)

        coverage = _mask_coverage_pct(mask)
        self.assertGreater(
            coverage, 2.0, f"spiderweb creases must be detected (got {coverage:.2f}%)"
        )
        self.assertLess(coverage, 40.0)


class TestBenchmarkCase11MinorCreases(unittest.TestCase):
    """case11 = image3.jpg.jpeg: light creases (plan probe: ~5.1% mask)."""

    def test_minor_creases_detected_without_overreach(self):
        rgb = _load_rgb("case11_minor_creases.jpg")
        _ctx, mask = _Stage2Case.run_stage2(rgb)

        coverage = _mask_coverage_pct(mask)
        self.assertGreater(
            coverage, 0.5, f"minor creases must still be found (got {coverage:.2f}%)"
        )
        self.assertLess(
            coverage, 25.0, "light damage must not produce a heavy-regen mask"
        )


if __name__ == "__main__":
    unittest.main()
