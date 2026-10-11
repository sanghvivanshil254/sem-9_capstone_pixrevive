"""Phase 4A quality-engine unit tests.

Q-3: the colorization hallucination guard detects a pathological single-hue
wash and rolls back to the pre-colorization (luminance-preserving) canvas.

U-6: Stage 2b's face-safe zone is built from the convex hull of detected
landmarks (not the legacy center rectangle). High-variance damage clusters
inside the hull are protected; clusters elsewhere — including inside the old
center rectangle — are smoothed.
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ai.pipeline.pipeline_context import PipelineContext  # noqa: E402
from ai.pipeline.stages.color_stages import ColorStagesMixin  # noqa: E402
from ai.pipeline.restoration_pipeline import RestorationPipeline  # noqa: E402


def _noise_patch(canvas, y0, y1, x0, x1, sigma=60.0, seed=7):
    """Paint a high-variance cluster in-place and return its center coords."""
    rng = np.random.default_rng(seed)
    base = int(canvas[y0:y1, x0:x1].mean())
    patch = np.clip(
        base + rng.normal(0, sigma, (y1 - y0, x1 - x0, 3)), 0, 255
    ).astype(np.uint8)
    canvas[y0:y1, x0:x1] = patch
    return (y0 + y1) // 2, (x0 + x1) // 2


def _local_variance_at(gray_f, cy, cx, win=15):
    half = win // 2
    win_img = gray_f[cy - half:cy + half + 1, cx - half:cx + half + 1]
    return float(win_img.std())


_FACE = {
    "bbox": [80, 90, 170, 180],
    "confidence": 0.99,
    "damage_ratio": 0.0,
    # 5-point landmarks: eyes, nose, mouth corners
    "landmarks": [
        [110.0, 120.0], [150.0, 120.0], [130.0, 145.0],
        [112.0, 160.0], [148.0, 160.0],
    ],
}


class TestHallucinationGuard(unittest.TestCase):
    def test_single_hue_wash_detected(self):
        """A vivid single-hue canvas (the DDColor failure mode) must trip."""
        hsv = np.full((120, 120, 3), 128, dtype=np.uint8)
        hsv[:, :, 0] = 10           # one hue bin (reds, 0-15)
        hsv[:, :, 1] = 220          # vivid
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        self.assertTrue(ColorStagesMixin._detect_colorization_hallucination(rgb))

    def test_natural_multihue_image_passes(self):
        """A normal colorful image spans many hue bins — no trip."""
        rng = np.random.default_rng(2)
        hsv = np.zeros((160, 160, 3), dtype=np.uint8)
        hsv[:, :, 0] = rng.integers(0, 180, (160, 160))
        hsv[:, :, 1] = rng.integers(60, 255, (160, 160))
        hsv[:, :, 2] = 200
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        self.assertFalse(ColorStagesMixin._detect_colorization_hallucination(rgb))

    def test_low_saturation_monochrome_content_passes(self):
        """An overcast single-hue scene with weak chroma must not trip."""
        hsv = np.full((120, 120, 3), 128, dtype=np.uint8)
        hsv[:, :, 0] = 10
        hsv[:, :, 1] = 15           # barely saturated
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        self.assertFalse(ColorStagesMixin._detect_colorization_hallucination(rgb))

    def test_rollback_restores_pre_colorization_canvas(self):
        pre = np.full((64, 64, 3), 90, dtype=np.uint8)
        ctx = PipelineContext(pre.copy(), {"color_type": "grayscale"})
        ctx.working_image = np.full((64, 64, 3), 200, dtype=np.uint8)  # "hallucinated"

        ColorStagesMixin._rollback_to_l_preserved_grayscale(ctx, pre)

        self.assertTrue(np.array_equal(ctx.working_image, pre))
        self.assertTrue(ctx.image_metadata.get("colorization_rolled_back"))
        # N5: the rollback must carry the machine-readable disclaimer too.
        self.assertTrue(ctx.image_metadata.get("colorization_disclaimer"))
        self.assertTrue(
            any("rolled back" in line for line in ctx.processing_log)
        )


def _damage_cluster(canvas, y0, x0, seed=7):
    """Paint a realistic damage cluster: luminance-correlated grain (NLM-
    removable) across the whole 40x40 patch, plus a bright bar running
    from y0+8 down to the patch bottom. The bar's top edge drives the
    15x15 sliding-window variance past the stage's 35.0 gate for every
    point within ~7px of it.

    Returns a probe point 5px below the bar's top edge: the gate's mean
    window (radius 7) still sees the edge there, while the hf measurement
    window (radius 3) stays clear of it — so the only high-frequency
    content at the probe is the grain NLM is supposed to remove.
    """
    rng = np.random.default_rng(seed)
    grain = rng.normal(0, 14, (40, 40, 1))  # correlated across channels
    canvas[y0:y0 + 40, x0:x0 + 40] = np.clip(
        110 + grain, 0, 255
    ).astype(np.uint8)
    bar = np.clip(230 + grain[8:40, :], 0, 255)
    canvas[y0 + 8:y0 + 40, x0:x0 + 40] = bar.astype(np.uint8)
    return y0 + 13, x0 + 20  # inside bar, gate-on via top edge, grain-only hf


def _max_window_diff(a, b, cy, cx, half=12):
    return int(np.abs(
        a[cy - half:cy + half, cx - half:cx + half].astype(int)
        - b[cy - half:cy + half, cx - half:cx + half].astype(int)
    ).max())


def _hf_std(img, cy, cx, half=3):
    """High-frequency residual std over a window. NLM removes grain (this
    drops) while preserving structure like the bar edges (the sliding-window
    variance that gates the stage stays high). The window must sit clear of
    the bar's edges, so the probe point from _damage_cluster is required."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hf = g - cv2.GaussianBlur(g, (5, 5), 0)
    return float(hf[cy - half:cy + half + 1, cx - half:cx + half + 1].std())


class TestFaceSafeZoneFromHull(unittest.TestCase):
    """U-6: the safe zone hugs the landmark hull, not the frame center.

    Stage 2b's NLM is only effective on moderate grain (sigma <~20), so the
    fixtures pair sigma-14 grain with a contrasty bar to trip the variance
    gate — the way real damage clusters carry both grain and structure.
    Assertions compare an in-hull cluster against an identical out-of-hull
    twin, so every test carries its own smoothing control.
    """

    @classmethod
    def setUpClass(cls):
        cls.pipeline = RestorationPipeline(load_models=False)

    def _run_stage_2b(self, faces, clusters=()):
        img = np.full((320, 320, 3), 110, dtype=np.uint8)
        coords = [_damage_cluster(img, y, x) for y, x in clusters]
        ctx = PipelineContext(img.copy(), {"color_type": "color"})
        ctx.working_image = img.copy()
        ctx.image_metadata["faces"] = faces
        self.pipeline._stage_2b_damage_cluster_smooth(ctx, None)
        return img, ctx.working_image, coords

    def test_in_hull_cluster_untouched_while_twin_is_smoothed(self):
        # Twin A sits inside the dilated landmark hull; twin B is far from
        # every protected zone. Same damage, opposite outcomes.
        before, after, coords = self._run_stage_2b(
            [dict(_FACE)], clusters=[(115, 105), (255, 40)]
        )
        (hy, hx), (fy, fx) = coords

        # A: deep inside the safe zone — dz is exactly 0. Only the stage's
        # LAB uint8 roundtrip quantization (<=1/channel) may touch pixels.
        self.assertLessEqual(
            _max_window_diff(before, after, hy, hx), 2,
            "cluster inside the landmark hull must not be touched at all",
        )
        # B: the control — the stage engaged NLM on it: grain removed from
        # the bar interior while the bar structure that gated the zone stays.
        hf_b, hf_a = _hf_std(before, fy, fx), _hf_std(after, fy, fx)
        self.assertGreater(hf_b, 6.0, "fixture grain must be present")
        self.assertLess(
            hf_a, hf_b * 0.75,
            "out-of-hull twin must be measurably smoothed (control)",
        )

    def test_cluster_in_legacy_center_rect_is_now_smoothed(self):
        """U-6 regression: (225, 115) is inside the legacy center rectangle
        (75..225 both axes) but outside the hull's protection — the old code
        blinded exactly this zone."""
        before, after, coords = self._run_stage_2b(
            [dict(_FACE)], clusters=[(205, 95)]
        )
        cy, cx = coords[0]

        hf_b, hf_a = _hf_std(before, cy, cx), _hf_std(after, cy, cx)
        self.assertGreater(hf_b, 6.0, "fixture grain must be present")
        self.assertLess(
            hf_a, hf_b * 0.75,
            "cluster inside the legacy center rect must be smoothed now",
        )

    def test_no_landmarks_falls_back_to_bbox_rect(self):
        face = {"bbox": _FACE["bbox"], "confidence": 0.9, "landmarks": None}
        before, after, coords = self._run_stage_2b(
            [face], clusters=[(115, 105), (255, 40)]
        )
        (by, bx), (fy, fx) = coords

        self.assertLessEqual(
            _max_window_diff(before, after, by, bx), 2,
            "bbox fallback must keep the in-bbox cluster untouched",
        )
        self.assertLess(
            _hf_std(after, fy, fx), _hf_std(before, fy, fx) * 0.75,
            "out-of-bbox twin must be smoothed (control)",
        )

    def test_no_faces_falls_back_to_center_rect(self):
        before, after, coords = self._run_stage_2b(
            [], clusters=[(115, 105), (255, 40)]
        )
        (cy_, cx_), (fy, fx) = coords

        self.assertLessEqual(
            _max_window_diff(before, after, cy_, cx_), 2,
            "center-rect fallback must keep the central cluster untouched",
        )
        self.assertLess(
            _hf_std(after, fy, fx), _hf_std(before, fy, fx) * 0.75,
            "far twin must be smoothed (control)",
        )


if __name__ == "__main__":
    unittest.main()
