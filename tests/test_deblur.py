"""T-3 deblur tests (Phase 4B.6).

The gate is the safety-critical part: Restormer must never touch a sharp
image (it would hallucinate detail), and the whole real corpus must measure
inert. The two-path gate was probe-calibrated:

  path A (defocus): edge-spread acutance >= 0.47 with enough edge evidence —
    corpus max 0.456; defocus sigma>=1.5 measures 0.48+;
  path B (motion): H/V gradient-variance anisotropy >= 2.2 —
    corpus max 1.69; 9-15px motion fixtures 2.4+.

Full-inference tests run on a small canvas (single tile) so no test pays the
multi-tile cost; the 100MB weights load lazily exactly as in production.
"""

import glob
import os
import sys
import unittest

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai.restoration.deblurrer import (  # noqa: E402
    ImageDeblurrer,
    estimate_optical_blur,
    BLUR_SPREAD_GATE,
    MOTION_RATIO_GATE,
)

DATASET_DIR = os.path.join(os.path.dirname(__file__), "restoration_dataset", "damaged")
WEIGHTS_PRESENT = os.path.exists(
    os.path.join("backend", "ai", "weights", "restormer", "motion_deblurring.pth")
)


def _photo_base(size=None):
    hits = glob.glob(os.path.join(DATASET_DIR, "case1_lena.jpg"))
    img = cv2.cvtColor(cv2.imread(hits[0]), cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def _defocus(img, sigma=2.5):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def _motion(img, k=15, vertical=False):
    kernel = np.ones((k, 1) if vertical else (1, k), np.float32) / k
    return cv2.filter2D(img, -1, kernel)


def _sharpness(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))


class TestOpticalBlurGate(unittest.TestCase):
    def test_whole_real_corpus_is_inert(self):
        paths = sorted(glob.glob(os.path.join(DATASET_DIR, "*")))
        self.assertGreaterEqual(len(paths), 15)
        for p in paths:
            with self.subTest(image=os.path.basename(p)):
                img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                m = estimate_optical_blur(img)
                self.assertFalse(m["is_optical"],
                                 f"{p} must never fire the deblur gate: {m}")

    def test_defocus_fixture_fires_path_a(self):
        m = estimate_optical_blur(_defocus(_photo_base()))
        self.assertTrue(m["is_optical"])
        self.assertIn("A", m["fire_path"])
        self.assertGreaterEqual(m["edge_spread"], BLUR_SPREAD_GATE)

    def test_motion_fixture_fires_path_b(self):
        m = estimate_optical_blur(_motion(_photo_base()))
        self.assertTrue(m["is_optical"])
        self.assertIn("B", m["fire_path"])
        self.assertGreaterEqual(m["directional_ratio"], MOTION_RATIO_GATE)

    def test_sharp_but_low_detail_photo_stays_inert(self):
        # The probe's key false-positive case: low Laplacian variance but
        # sharp edges (case11/case4) — the acutance metric must reject it.
        hits = glob.glob(os.path.join(DATASET_DIR, "case11_minor_creases.jpg"))
        img = cv2.cvtColor(cv2.imread(hits[0]), cv2.COLOR_BGR2RGB)
        m = estimate_optical_blur(img)
        self.assertFalse(m["is_optical"])

    def test_weak_defocus_below_gate(self):
        # sigma=1.0 measures ~0.41 — inside the corpus band, below the gate.
        m = estimate_optical_blur(_defocus(_photo_base(), 1.0))
        self.assertFalse(m["is_optical"])


class TestDeblurStage(unittest.TestCase):
    @unittest.skipUnless(WEIGHTS_PRESENT, "deblur weights not downloaded")
    def test_defocus_fixture_improves(self):
        d = ImageDeblurrer()
        dev = "cpu"  # tiny canvas: single tile, CPU is seconds
        d.load(dev)
        base = _photo_base(size=(128, 128))
        blurred = _defocus(base, 2.5)
        out, info = d.deblur(blurred, device=dev)
        self.assertTrue(info["applied"])
        self.assertGreater(_sharpness(out), _sharpness(blurred) * 1.3)

    @unittest.skipUnless(WEIGHTS_PRESENT, "deblur weights not downloaded")
    def test_sharp_input_bitwise_unchanged(self):
        d = ImageDeblurrer()
        d.load("cpu")
        base = _photo_base(size=(128, 128))
        out, info = d.deblur(base, device="cpu")
        self.assertFalse(info["applied"])
        self.assertTrue(np.array_equal(out, base))

    def test_unloaded_model_reports_and_leaves_image(self):
        d = ImageDeblurrer()
        img = _photo_base(size=(64, 64))
        out, info = d.deblur(img, device="cpu")
        self.assertFalse(info["applied"])
        self.assertTrue(np.array_equal(out, img))


class TestStageWiring(unittest.TestCase):
    def test_stage_method_skips_without_loading_on_sharp_image(self):
        from ai.pipeline.restoration_pipeline import RestorationPipeline
        from ai.pipeline.pipeline_context import PipelineContext
        pipeline = RestorationPipeline(device="cpu", load_models=False)
        img = _photo_base(size=(96, 96))
        ctx = PipelineContext(img.copy())
        pipeline._stage_4_5_deblur(ctx, None)
        self.assertTrue(np.array_equal(ctx.working_image, img))
        self.assertIsNone(pipeline._deblur, "gate-inert run must not load 100MB of weights")

    def test_registry_order(self):
        from backend.ai.pipeline.stage_registry import STAGE_REGISTRY
        ids = [s.id for s in STAGE_REGISTRY]
        self.assertIn("deblur", ids)
        self.assertLess(ids.index("denoise"), ids.index("deblur"))
        self.assertLess(ids.index("deblur"), ids.index("colorize"))


if __name__ == "__main__":
    unittest.main()
