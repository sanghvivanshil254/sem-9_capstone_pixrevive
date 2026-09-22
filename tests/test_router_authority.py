"""Q-1 / D16: the router's run_* flags have real authority.

Three layers are tested:
  1. PipelineRouter.route() selects each of the 6 presets + SAFE_FALLBACK
     from synthetic metadata (no models needed).
  2. Dark-spot evidence (foxing/mold, T-1) escalates a B&W print to a
     repair-granting route — the regression that motivated feeding
     dark_spot_percentage into the router.
  3. RestorationPipeline._effective_stage_flags() ANDs route authority with
     user flags — including the colorization carve-out that keeps B&W
     colorization following the user toggle.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ai.pipeline.pipeline_router import PipelineRouter  # noqa: E402
from ai.pipeline.pipeline_context import PipelineContext  # noqa: E402
from ai.pipeline.restoration_pipeline import RestorationPipeline  # noqa: E402


def _route_for(meta: dict) -> dict:
    ctx = PipelineContext.__new__(PipelineContext)
    ctx.image_metadata = dict(meta)
    ctx.processing_log = []
    return PipelineRouter.route(ctx)


_BASE = {
    "damage_percentage": 0.0,
    "has_white_flakes": False,
    "blur_score": 400.0,
    "noise_score": 0.5,
    "color_type": "color",
}


class TestRouteSelection(unittest.TestCase):
    def _meta(self, **over):
        meta = dict(_BASE)
        meta.update(over)
        return meta

    def test_clean_enhance(self):
        route = _route_for(self._meta())
        self.assertEqual(route["pipeline_name"], "CLEAN_ENHANCE")
        self.assertFalse(route["run_damage_repair"])
        self.assertFalse(route["run_face_restore"])
        self.assertTrue(route["run_sr"])

    def test_noisy_clean(self):
        route = _route_for(self._meta(noise_score=8.0))
        self.assertEqual(route["pipeline_name"], "NOISY_CLEAN")
        self.assertTrue(route["run_denoising"])
        self.assertFalse(route["run_damage_repair"])

    def test_noisy_face(self):
        route = _route_for(self._meta(noise_score=8.0, blur_score=100.0))
        self.assertEqual(route["pipeline_name"], "NOISY_FACE")
        self.assertTrue(route["run_denoising"])
        self.assertTrue(route["run_face_restore"])

    def test_damaged_no_face(self):
        route = _route_for(self._meta(damage_percentage=6.0))
        self.assertEqual(route["pipeline_name"], "DAMAGED_NO_FACE")
        self.assertTrue(route["run_damage_repair"])
        self.assertFalse(route["run_face_restore"])

    def test_damaged_face_via_heavy_damage(self):
        route = _route_for(self._meta(damage_percentage=25.0))
        self.assertEqual(route["pipeline_name"], "DAMAGED_FACE")
        self.assertTrue(route["run_damage_repair"])
        self.assertTrue(route["run_face_restore"])

    def test_damaged_face_via_damage_and_blur(self):
        route = _route_for(self._meta(damage_percentage=6.0, blur_score=100.0))
        self.assertEqual(route["pipeline_name"], "DAMAGED_FACE")

    def test_bw_restore_for_undamaged_grayscale(self):
        route = _route_for(self._meta(color_type="grayscale"))
        self.assertEqual(route["pipeline_name"], "BW_RESTORE")
        self.assertFalse(route["run_damage_repair"])

    def test_damaged_bw_escapes_to_repair_first_route(self):
        route = _route_for(
            self._meta(color_type="grayscale", damage_percentage=6.0)
        )
        self.assertEqual(route["pipeline_name"], "DAMAGED_FACE")
        self.assertTrue(route["run_damage_repair"])

    def test_safe_fallback_moderate_inputs(self):
        route = _route_for(self._meta(noise_score=3.0, blur_score=150.0))
        self.assertEqual(route["pipeline_name"], "SAFE_FALLBACK")


class TestDarkSpotEscalation(unittest.TestCase):
    """Foxing/mold evidence must flip BW_RESTORE to a repair-granting route."""

    def test_dark_spots_escalate_bw_route(self):
        route = _route_for(
            {
                "damage_percentage": 0.0,
                "has_white_flakes": False,
                "blur_score": 400.0,
                "noise_score": 1.0,
                "color_type": "grayscale",
                "dark_spot_percentage": 0.3,
            }
        )
        self.assertEqual(route["pipeline_name"], "DAMAGED_FACE")
        self.assertTrue(route["run_damage_repair"])

    def test_subthreshold_dark_spots_stay_bw_restore(self):
        route = _route_for(
            {
                "damage_percentage": 0.0,
                "has_white_flakes": False,
                "blur_score": 400.0,
                "noise_score": 1.0,
                "color_type": "grayscale",
                "dark_spot_percentage": 0.01,
            }
        )
        self.assertEqual(route["pipeline_name"], "BW_RESTORE")


class TestEffectiveStageFlags(unittest.TestCase):
    """restore() runs a stage only when route AND user both allow it."""

    @staticmethod
    def _flags(route, color_type="color", **user):
        defaults = dict(
            enable_repair=True,
            enable_denoise=True,
            enable_colorization=True,
            enable_face_restore=True,
            enable_sr=True,
        )
        defaults.update(user)
        return RestorationPipeline._effective_stage_flags(
            route, color_type, **defaults
        )

    def test_clean_route_narrows_pipeline(self):
        route = {"run_damage_repair": False, "run_denoising": False,
                 "run_face_restore": False, "run_colorization": False,
                 "run_sr": True}
        flags = self._flags(route)
        self.assertFalse(flags["run_repair"])
        self.assertFalse(flags["run_face"])
        self.assertTrue(flags["run_sr"])

    def test_user_optout_beats_route_grant(self):
        route = {"run_damage_repair": True, "run_denoising": True,
                 "run_face_restore": True, "run_colorization": False,
                 "run_sr": True}
        flags = self._flags(route, enable_repair=False, enable_sr=False)
        self.assertFalse(flags["run_repair"])
        self.assertFalse(flags["run_sr"])
        self.assertTrue(flags["run_face"])

    def test_bw_colorization_follows_user_toggle_despite_route(self):
        route = {"run_damage_repair": False, "run_denoising": True,
                 "run_face_restore": True, "run_colorization": False,
                 "run_sr": True}
        flags = self._flags(route, color_type="grayscale")
        self.assertTrue(flags["run_colorize"])

    def test_color_image_colorization_needs_route_grant(self):
        route = {"run_damage_repair": False, "run_denoising": True,
                 "run_face_restore": True, "run_colorization": False,
                 "run_sr": True}
        flags = self._flags(route, color_type="color")
        self.assertFalse(flags["run_colorize"])

    def test_missing_route_keys_default_to_allow(self):
        flags = self._flags({})
        self.assertTrue(flags["run_repair"])
        self.assertTrue(flags["run_sr"])


if __name__ == "__main__":
    unittest.main()
