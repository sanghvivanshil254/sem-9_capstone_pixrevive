"""T-9 moiré / halftone suppression tests (Phase 4B.3).

HalftoneSuppressor (the Phase 1 D5 salvage) wired as the pipeline's moiré
specialist. Its own FFT gate + MAE safety valve do the discriminating; these
tests pin the contract from both sides:

  * strong synthetic print-screens are detected and their mesh energy drops;
  * EVERY real dataset image — including the macro-edge-heavy case2 landscape
    the plan calls out — must come out bitwise unchanged (no false fires);
  * the registry carries the stage between despeckle and analysis.

Runs CPU-only, no model loads.
"""

import glob
import os
import sys
import unittest

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai.preprocessing.halftone_suppressor import HalftoneSuppressor  # noqa: E402
from ai.pipeline.pipeline_context import PipelineContext  # noqa: E402
from backend.ai.pipeline.stage_registry import STAGE_REGISTRY  # noqa: E402

DATASET_DIR = os.path.join(os.path.dirname(__file__), "restoration_dataset", "damaged")


def _photo_base():
    hits = glob.glob(os.path.join(DATASET_DIR, "case1_lena.jpg"))
    return cv2.cvtColor(cv2.imread(hits[0]), cv2.COLOR_BGR2RGB)


def _halftone(img, period=9, amp=40):
    """Synthetic print-screen: sinusoidal dot lattice on all channels."""
    out = img.astype(np.float32).copy()
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    dots = np.sin(2 * np.pi * xs / period) * np.sin(2 * np.pi * ys / period)
    for c in range(3):
        out[..., c] = np.clip(out[..., c] + amp * dots, 0, 255)
    return out.astype(np.uint8)


def _mesh_energy(rgb):
    """Sharp-peak concentration in the FFT ring — the mesh signature."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(gray))
    log_mag = np.log(np.abs(f) + 1.0)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    ys, xs = np.ogrid[:h, :w]
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    ring = (dist >= max(6, int(min(h, w) * 0.05))) & (dist <= int(min(h, w) * 0.38))
    vals = log_mag[ring]
    return float((vals.max() - vals.mean()) / (vals.std() + 1e-6))


class TestMoireDetection(unittest.TestCase):
    def test_strong_halftone_detected(self):
        img = _halftone(_photo_base())
        has, peaks = HalftoneSuppressor._detect_halftone(
            cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32))
        self.assertTrue(has)
        self.assertGreaterEqual(len(peaks), 1)

    def test_weak_screen_below_gate(self):
        # amp=18 dot screen: below the spectral-peak gate. Accepted limitation:
        # the SR bilateral pre-filter is the backstop for faint screens; the
        # FFT gate stays conservative so natural photo textures never trip it.
        img = _halftone(_photo_base(), period=4, amp=18)
        has, _ = HalftoneSuppressor._detect_halftone(
            cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32))
        self.assertFalse(has)

    def test_all_real_corpus_not_detected(self):
        paths = sorted(glob.glob(os.path.join(DATASET_DIR, "*")))
        self.assertGreaterEqual(len(paths), 15)
        for p in paths:
            with self.subTest(image=os.path.basename(p)):
                img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                has, _ = HalftoneSuppressor._detect_halftone(
                    cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32))
                self.assertFalse(has, "natural photo must never trip the FFT gate")


class TestMoireSuppression(unittest.TestCase):
    def test_suppress_reduces_mesh_on_halftone(self):
        img = _halftone(_photo_base())
        ctx = PipelineContext(img.copy())
        out = HalftoneSuppressor.suppress(ctx)
        self.assertTrue(ctx.image_metadata.get("halftone_suppressed"))
        self.assertLess(_mesh_energy(out), _mesh_energy(img))
        self.assertFalse(np.array_equal(out, img))

    def test_suppress_inert_on_clean_photo(self):
        img = _photo_base()
        ctx = PipelineContext(img.copy())
        out = HalftoneSuppressor.suppress(ctx)
        self.assertFalse(ctx.image_metadata.get("halftone_suppressed"))
        self.assertTrue(np.array_equal(out, img))

    def test_suppress_inert_on_all_real_images(self):
        paths = sorted(glob.glob(os.path.join(DATASET_DIR, "*")))
        for p in paths:
            with self.subTest(image=os.path.basename(p)):
                img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                ctx = PipelineContext(img.copy())
                out = HalftoneSuppressor.suppress(ctx)
                self.assertFalse(ctx.image_metadata.get("halftone_suppressed"))
                self.assertTrue(np.array_equal(out, img))


class TestRegistryContract(unittest.TestCase):
    def test_moire_registered_before_analysis(self):
        ids = [s.id for s in STAGE_REGISTRY]
        self.assertIn("moire", ids)
        spec = next(s for s in STAGE_REGISTRY if s.id == "moire")
        self.assertIsNone(spec.method)  # inlined pre-context, like deskew
        self.assertLess(ids.index("despeckle"), ids.index("moire"))
        self.assertLess(ids.index("moire"), ids.index("analyze"),
                        "moiré must be suppressed before degradation analysis")


if __name__ == "__main__":
    unittest.main()
