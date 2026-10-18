# -*- coding: utf-8 -*-
"""P2: flag-gated post-SR seam feather.

Seam birth trace (FinalMetric definition, common input-frame resolution):
LaMa creates the seam at Stage 3, NLM reduces it, GFPGAN reduces it, and
Real-ESRGAN x4 AMPLIFIES it (+116% lena 14.72 -> 31.8, +68% bw 18.4 -> 30.88).
No stage re-smooths the seam band between Stage 3 and Stage 9. These tests
pin the new post-SR feather: blended only inside the mask boundary band,
facial anatomy zeroed out, default-OFF (PIXREVIVE_POST_SR_SEAM_FEATHER).
"""
import os
import sys
import unittest
import unittest.mock as mock

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "backend"))


def _seam_L_p95(img_bgr, mask):
    """The FinalMetric seam measurement (band ±5px at this resolution)."""
    inner = cv2.erode((mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    outer = cv2.dilate((mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    band = outer & ~inner
    if not band.any():
        return None
    g = cv2.convertScaleAbs(img_bgr)  # keep 3ch
    L = cv2.cvtColor(g, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0]
    delta = np.abs(L - cv2.blur(L, (5, 5)))
    return round(float(np.percentile(delta[band], 95)), 2)


def _hard_seam_canvas():
    """Canvas with a hard L-step across the middle of a mask band."""
    img = np.full((120, 160, 3), 120, np.uint8)
    img[:, 69:] = 165  # hard step straddling the mask edge x=70 (band 68..72)
    mask = np.zeros((120, 160), np.uint8)
    mask[:, 70:90] = 255  # FinalMetric band hugs the step
    return img, mask


class TestPostSrSeamFeather(unittest.TestCase):
    def test_flag_off_is_noop(self):
        from ai.pipeline.stages.finish_stages import FinishStagesMixin

        img, mask = _hard_seam_canvas()
        before = img.copy()
        ctx = mock.MagicMock()
        ctx.working_image = img
        ctx.damage_masks = {"final": mask}
        ctx.image_metadata = {"faces": []}
        ctx.add_log = lambda *a, **k: None
        mixin = FinishStagesMixin()
        with mock.patch.dict(os.environ, {"PIXREVIVE_POST_SR_SEAM_FEATHER": "0"}):
            mixin._post_sr_seam_feather(ctx, mask)
        self.assertTrue(np.array_equal(ctx.working_image, before),
                        "flag OFF must be a strict no-op")

    def test_reduces_seam_without_blurring_outside_band(self):
        from ai.pipeline.stages.finish_stages import FinishStagesMixin

        img, mask = _hard_seam_canvas()
        ctx = mock.MagicMock()
        ctx.working_image = img
        ctx.damage_masks = {"final": mask}
        ctx.image_metadata = {"faces": []}
        ctx.add_log = lambda *a, **k: None
        mixin = FinishStagesMixin()
        with mock.patch.dict(os.environ, {"PIXREVIVE_POST_SR_SEAM_FEATHER": "1"}):
            mixin._post_sr_seam_feather(ctx, mask)
        out = ctx.working_image
        # seam inside the band must fall meaningfully
        self.assertLess(_seam_L_p95(out, mask), _seam_L_p95(img, mask) - 5)
        # far outside the band the pixels must be untouched
        far = np.ones((120, 160), np.uint8)
        far[:, 55:105] = 0
        self.assertTrue(np.array_equal(out[far > 0], img[far > 0]),
                        "feather must not touch pixels outside the band")

    def test_face_anatomy_never_blurred(self):
        from ai.pipeline.stages.finish_stages import FinishStagesMixin

        img, mask = _hard_seam_canvas()
        # put a "face zone" over part of the band: eye landmark circle center
        ctx = mock.MagicMock()
        ctx.working_image = img
        ctx.damage_masks = {"final": mask}
        ctx.image_metadata = {"faces": [{"bbox": [60, 40, 120, 100],
                                          "landmarks": [[95, 60], [105, 60],
                                                        [100, 70], [90, 85],
                                                        [110, 85]]}]}
        ctx.add_log = lambda *a, **k: None
        mixin = FinishStagesMixin()
        with mock.patch.dict(os.environ, {"PIXREVIVE_POST_SR_SEAM_FEATHER": "1"}):
            mixin._post_sr_seam_feather(ctx, mask)
        out = ctx.working_image
        # the landmark eye area must stay identical to the input
        ex, ey = 95, 60
        r = 10
        eye_in = img[ey - r:ey + r, ex - r:ex + r]
        eye_out = out[ey - r:ey + r, ex - r:ex + r]
        self.assertTrue(np.array_equal(eye_out, eye_in),
                        "facial anatomy must be excluded from the feather")


if __name__ == "__main__":
    unittest.main()
