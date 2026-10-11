"""Tests for the remediation-plan quality audit + guards
(docs/REMEDIATION_PLAN_AB_G.md VALIDATOR ADDITIONS).

All CPU, synthetic images, no models.
"""

import os
import sys
import unittest

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from ai.validation.quality_audit import (  # noqa: E402
    chroma_bleed_score,
    edge_retention_ratio,
    eye_symmetry,
    hf_retention_ratio,
    residual_speck_count,
    run_quality_audit,
    seam_step_score,
)
from ai.preprocessing.impulse_denoiser import ImpulseDenoiser  # noqa: E402
from ai.restoration.paste_guard import harmonize_eyes  # noqa: E402


def _textured_rgb(h=200, w=200, seed=3):
    rng = np.random.RandomState(seed)
    base = rng.randint(0, 255, (h, w), dtype=np.uint8)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)


def _flat_rgb(h=200, w=200, v=128):
    return np.full((h, w, 3), v, dtype=np.uint8)


class TestHfRetention(unittest.TestCase):
    def test_identical_is_one(self):
        img = _textured_rgb()
        self.assertEqual(hf_retention_ratio(img, img.copy()), 1.0)

    def test_blur_loses_hf(self):
        img = _textured_rgb()
        blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
        r = hf_retention_ratio(img, blurred)
        self.assertIsNotNone(r)
        self.assertLess(r, 0.3)


class TestSpeckCount(unittest.TestCase):
    def test_flat_has_no_specks(self):
        self.assertEqual(residual_speck_count(_flat_rgb()), 0)

    def test_dust_is_counted_and_removal_reduces(self):
        # Realistic scan texture: mild grain (sigma ~6), not uniform noise —
        # the blob pass is sigma-adaptive and must (and does) refuse to fire
        # on high-variance texture fields.
        rng = np.random.RandomState(5)
        base = np.clip(128 + rng.normal(0, 6, (200, 200)), 0, 255).astype(np.uint8)
        img = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
        dirty = img.copy()
        pts = [(rng.randint(10, 190), rng.randint(10, 190)) for _ in range(12)]
        for y, x in pts:
            dirty[y, x] = (250, 250, 250)
        ctx = _FakeCtx(dirty)
        self.assertTrue(ImpulseDenoiser.suppress_isolated_blobs(ctx))
        cleaned = ctx.working_image
        self.assertLess(residual_speck_count(cleaned), residual_speck_count(dirty))

    def test_texture_not_flagged(self):
        img = _textured_rgb()
        # Texture fields must not be removed by the blob pass.
        ctx = _FakeCtx(img)
        meta_rec = ctx.image_metadata.get("blob_inpaint")
        ImpulseDenoiser.suppress_isolated_blobs(ctx)
        # Either a no-op or within the coverage cap; never a wipeout.
        cov = (ctx.image_metadata.get("blob_inpaint") or {}).get("coverage_pct", 0.0)
        self.assertLessEqual(cov, 4.0)
        self.assertIsNone(meta_rec)  # not recorded before the call


class _FakeCtx:
    def __init__(self, img):
        self.working_image = img
        self.image_metadata = {}
        self.logs = []

    def add_log(self, msg):
        self.logs.append(msg)


class TestEyeSymmetry(unittest.TestCase):
    def test_symmetric_eyes(self):
        img = _flat_rgb(240, 240, 110)
        for cx in (80, 160):
            cv2.circle(img, (cx, 90), 10, (40, 40, 40), -1)
            cv2.circle(img, (cx + 3, 88), 2, (240, 240, 240), -1)
        faces = [{"bbox": [50, 40, 190, 200],
                  "landmarks": [[80, 90], [160, 90], [120, 130], [95, 160], [145, 160]]}]
        rec = eye_symmetry(img, faces)
        self.assertEqual(rec["faces_evaluated"], 1)
        self.assertGreater(rec["eye_sharp_ratio"], 0.4)

    def test_one_blind_eye_scores_low(self):
        img = _flat_rgb(240, 240, 110)
        cv2.circle(img, (160, 90), 10, (40, 40, 40), -1)  # right eye only
        faces = [{"bbox": [50, 40, 190, 200],
                  "landmarks": [[80, 90], [160, 90], [120, 130], [95, 160], [145, 160]]}]
        rec = eye_symmetry(img, faces)
        self.assertLess(rec["eye_sharp_ratio"], 0.3)


class TestEyeHarmony(unittest.TestCase):
    def test_blur_weak_eye_is_sharpened(self):
        img = _flat_rgb(240, 240, 110)
        cv2.circle(img, (80, 90), 10, (40, 40, 40), -1)   # left eye
        cv2.circle(img, (160, 90), 10, (40, 40, 40), -1)  # right eye
        weak = cv2.GaussianBlur(img, (0, 0), 2.5)         # both eyes blurred
        cv2.circle(weak, (160, 90), 10, (40, 40, 40), -1)  # right eye re-sharpened
        faces = [{"bbox": [50, 40, 190, 200],
                  "landmarks": [[80, 90], [160, 90], [120, 130], [95, 160], [145, 160]]}]
        out, verdicts = harmonize_eyes(weak, faces, source_mono=True)
        self.assertTrue(verdicts and verdicts[0].get("adjusted"))
        g0 = cv2.cvtColor(weak, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g1 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
        r0 = np.var(cv2.Laplacian(g0[80:100, 66:94], cv2.CV_32F))
        r1 = np.var(cv2.Laplacian(g1[80:100, 66:94], cv2.CV_32F))
        self.assertGreater(r1, r0)  # weak eye got sharper

    def test_no_faces_is_noop(self):
        img = _flat_rgb()
        out, verdicts = harmonize_eyes(img, [], source_mono=True)
        np.testing.assert_array_equal(out, img)
        self.assertEqual(verdicts, [])


class TestChromaBleed(unittest.TestCase):
    def test_gray_image_no_bleed(self):
        g = _textured_rgb()
        self.assertLess(chroma_bleed_score(g), 0.001)

    def test_painted_bleed_detected(self):
        img = _textured_rgb().astype(np.int16)
        # Chroma step without a luminance step: a* jumps mid-image.
        img[:, 100:, 1] = np.clip(img[:, 100:, 1] + 40, 0, 255)
        img[:, 100:, 2] = np.clip(img[:, 100:, 2] + 30, 0, 255)
        score = chroma_bleed_score(img.astype(np.uint8))
        flat = chroma_bleed_score(_flat_rgb())
        self.assertGreater(score, flat)


class TestSeamStep(unittest.TestCase):
    def test_hard_step_detected(self):
        img = _flat_rgb()
        img[:, 100:] = 200
        mask = np.zeros((200, 200), np.uint8)
        mask[:, 100:] = 255
        s = seam_step_score(img, mask)
        self.assertIsNotNone(s)
        self.assertGreater(s, 20.0)


class TestEdgeRetention(unittest.TestCase):
    def test_edge_kept_vs_blurred(self):
        img = _textured_rgb(240, 240)
        cv2.rectangle(img, (60, 180), (180, 200), (30, 30, 30), 3)  # shoulder line
        blurred = cv2.GaussianBlur(img, (0, 0), 4.0)
        faces = [{"bbox": [50, 40, 190, 170], "landmarks": [[80, 90], [160, 90]]}]
        kept = edge_retention_ratio(img, img.copy(), faces)
        lost = edge_retention_ratio(img, blurred, faces)
        self.assertIsNotNone(kept)
        self.assertIsNotNone(lost)
        self.assertGreater(kept, lost)


class TestRunAudit(unittest.TestCase):
    def test_bundle_and_flags(self):
        img = _flat_rgb(240, 240, 110)
        cv2.circle(img, (160, 90), 10, (40, 40, 40), -1)  # blind left eye
        faces = [{"bbox": [50, 40, 190, 200],
                  "landmarks": [[80, 90], [160, 90], [120, 130], [95, 160], [145, 160]]}]
        rec = run_quality_audit(img, img.copy(), faces, None)
        self.assertIn("eye_asymmetry", rec["flags"])
        self.assertIn("residual_speck_count", rec)
        self.assertIn("chroma_bleed_score", rec)

    def test_never_raises(self):
        rec = run_quality_audit(None, None, [], None)
        self.assertIn("version", rec)


if __name__ == "__main__":
    unittest.main()
