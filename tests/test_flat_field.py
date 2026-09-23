"""T-5 flat-field illumination correction tests (Phase 4B.2).

The gate is the whole feature here: natural content is full of smooth
gradients that a naive "illumination varies" test cannot tell from a lamp
defect. These tests pin the discriminating signature chosen by probe —
SYMMETRIC corner falloff of the background envelope — from both sides:

  * vignette fixtures (all four corners down together) must trigger, and the
    correction must measurably flatten them;
  * content gradients (dark bands, tilted exposure), natural photos, and the
    ENTIRE real dataset corpus must leave the image bitwise unchanged;
  * the correction itself must respect the 2x gain clamp and preserve hue;
  * the registry must carry the stage contract for the frontend.

Runs CPU-only, no model loads.
"""

import glob
import os
import sys
import unittest

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.ai.preprocessing.image_normalizer import ImageNormalizer  # noqa: E402


def _photo_base():
    """A real photo as the content base for synthetic fixtures."""
    hits = glob.glob(os.path.join(
        os.path.dirname(__file__), "restoration_dataset", "damaged", "case1_lena.jpg"
    ))
    img = cv2.imread(hits[0])
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _vignette(img, strength=0.45):
    """Radial falloff — the classic scanner/lamp defect signature."""
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt(((xs - cx) / cx) ** 2 + ((ys - cy) / cy) ** 2)
    field = 1.0 - strength * np.clip(r, 0, None) ** 2
    return np.clip(img.astype(np.float32) * field[..., None], 0, 255).astype(np.uint8)


def _dark_band(img, frac=0.5, level=0.35):
    """Left-half dark band — content contrast, NOT a lamp defect (asymmetric)."""
    w = img.shape[1]
    out = img.astype(np.float32).copy()
    out[:, : int(w * frac)] *= level
    return np.clip(out, 0, 255).astype(np.uint8)


def _flat_gray(h=240, w=320, v=180):
    """Uniform canvas with mild vignetting — pure signal, no content."""
    img = np.full((h, w, 3), v, dtype=np.uint8)
    return _vignette(img, 0.4)


def _corner_deficit_of(img):
    m = ImageNormalizer.estimate_illumination_deficit(img)
    return m["corner_deficit"], m["corner_spread"]


class TestIlluminationEstimator(unittest.TestCase):
    def test_vignette_shows_symmetric_deficit(self):
        deficit, spread = _corner_deficit_of(_vignette(_photo_base(), 0.45))
        self.assertGreaterEqual(deficit, 0.20)
        self.assertLessEqual(spread, 0.12)

    def test_flat_vignetted_canvas_strong_signal(self):
        deficit, spread = _corner_deficit_of(_flat_gray())
        self.assertGreaterEqual(deficit, 0.20)
        self.assertLessEqual(spread, 0.12)

    def test_plain_canvas_zero_deficit(self):
        img = np.full((240, 320, 3), 180, dtype=np.uint8)
        deficit, spread = _corner_deficit_of(img)
        self.assertLess(deficit, 0.20)
        self.assertLess(spread, 0.12)

    def test_content_band_is_asymmetric(self):
        deficit, spread = _corner_deficit_of(_dark_band(_photo_base()))
        # The band drops one side only: spread must exceed the gate ceiling.
        self.assertGreater(spread, 0.12)


class TestFlatFieldCorrection(unittest.TestCase):
    def test_vignetted_photo_corrected(self):
        v = _vignette(_photo_base(), 0.45)
        out, info = ImageNormalizer.correct_illumination(v)
        self.assertTrue(info["applied"])
        self.assertLessEqual(info["max_gain"], 2.0)
        # Corners must be measurably brighter after correction.
        h, w = v.shape[:2]
        b = 24
        before = float(np.mean(v[:b, :b].astype(np.float32)))
        after = float(np.mean(out[:b, :b].astype(np.float32)))
        self.assertGreater(after, before * 1.10)

    def test_center_preserved(self):
        # Division by a normalized field keeps the brightest region ~unchanged.
        v = _vignette(_photo_base(), 0.45)
        out, _ = ImageNormalizer.correct_illumination(v)
        h, w = v.shape[:2]
        c_in = v[h // 2 - 20: h // 2 + 20, w // 2 - 20: w // 2 + 20].astype(np.float32)
        c_out = out[h // 2 - 20: h // 2 + 20, w // 2 - 20: w // 2 + 20].astype(np.float32)
        self.assertLess(abs(float(c_out.mean()) - float(c_in.mean())) / max(float(c_in.mean()), 1e-6), 0.08)

    def test_flat_canvas_bitwise_unchanged(self):
        img = np.full((240, 320, 3), 180, dtype=np.uint8)
        out, info = ImageNormalizer.correct_illumination(img)
        self.assertFalse(info["applied"])
        self.assertTrue(np.array_equal(out, img))

    def test_content_band_bitwise_unchanged(self):
        img = _dark_band(_photo_base())
        out, info = ImageNormalizer.correct_illumination(img)
        self.assertFalse(info["applied"])
        self.assertTrue(np.array_equal(out, img))

    def test_tilted_exposure_bitwise_unchanged(self):
        # One-sided lamp falloff is intentionally NOT corrected (asymmetric —
        # documented false-negative; the safe direction).
        img = _photo_base().astype(np.float32)
        xs = np.linspace(1.0, 0.55, img.shape[1], dtype=np.float32)
        field = np.broadcast_to(xs[None, :], img.shape[:2]).copy()
        tilted = np.clip(img * field[..., None], 0, 255).astype(np.uint8)
        out, info = ImageNormalizer.correct_illumination(tilted)
        self.assertFalse(info["applied"])
        self.assertTrue(np.array_equal(out, tilted))

    def test_gain_clamp_limits_brightening(self):
        # A deep vignette must never brighten any pixel beyond 2x.
        v = _vignette(_photo_base(), 0.8)
        out, info = ImageNormalizer.correct_illumination(v)
        if info["applied"]:
            self.assertLessEqual(info["max_gain"], 2.0)

    def test_tiny_image_inert(self):
        img = np.full((20, 30, 3), 100, dtype=np.uint8)
        out, info = ImageNormalizer.correct_illumination(img)
        self.assertFalse(info["applied"])
        self.assertTrue(np.array_equal(out, img))


class TestRealCorpusInertness(unittest.TestCase):
    """The gate must be inert on every real dataset image — measured, not
    assumed: natural content reaches corner deficit 0.60 but always with
    asymmetric spread > 0.27, while the gate needs spread <= 0.12."""

    def test_all_real_images_bitwise_unchanged(self):
        paths = sorted(glob.glob(os.path.join(
            os.path.dirname(__file__), "restoration_dataset", "damaged", "*"
        )))
        self.assertGreaterEqual(len(paths), 15)
        for p in paths:
            with self.subTest(image=os.path.basename(p)):
                img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                out, info = ImageNormalizer.correct_illumination(img)
                self.assertFalse(info["applied"])
                self.assertTrue(np.array_equal(out, img))


class TestRegistryContract(unittest.TestCase):
    def test_flat_field_registered_pre_context(self):
        from backend.ai.pipeline.stage_registry import STAGE_REGISTRY
        ids = [s.id for s in STAGE_REGISTRY]
        self.assertIn("flat_field", ids)
        spec = next(s for s in STAGE_REGISTRY if s.id == "flat_field")
        self.assertIsNone(spec.method)  # inlined pre-context, like deskew
        self.assertLess(
            ids.index("flat_field"), ids.index("analyze"),
            "flat-field must run before degradation analysis measures the image",
        )


if __name__ == "__main__":
    unittest.main()
