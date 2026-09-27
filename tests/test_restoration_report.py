"""Tests for the T-2 Restoration Report builder (plan 5.1, honesty layer).

build_restoration_report() is read-only over PipelineContext evidence, so
these tests run entirely on CPU with synthetic contexts — no models load.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from ai.pipeline.pipeline_context import PipelineContext  # noqa: E402
from ai.report.restoration_report import (  # noqa: E402
    FACE_INPAINT_FLAG_PCT,
    UNRECOVERABLE_COVERAGE_PCT,
    build_restoration_report,
)


def _ctx(size=(64, 64)) -> PipelineContext:
    return PipelineContext(np.zeros((size[0], size[1], 3), dtype=np.uint8), {})


def _mask(size, boxes, value=255) -> np.ndarray:
    """uint8 mask with filled rectangles [(x0, y0, x1, y1), ...]."""
    m = np.zeros((size[0], size[1]), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        m[y0:y1, x0:x1] = value
    return m


class TestRestorationReport(unittest.TestCase):
    def test_empty_context_has_no_claims(self):
        report = build_restoration_report(_ctx())
        self.assertEqual(report["recovered"], [])
        self.assertEqual(report["reconstructed"], [])
        self.assertEqual(report["flagged"], [])
        self.assertFalse(report["colorization_disclaimer"])
        self.assertIn("generated_at", report)

    def test_small_inpaint_is_recovered_not_reconstructed(self):
        ctx = _ctx((100, 100))
        # 8% coverage: below the 25% reconstruction bar.
        ctx.damage_masks["final"] = _mask((100, 100), [(0, 0, 8, 100)])
        report = build_restoration_report(ctx)
        self.assertTrue(any("inpainted" in r for r in report["recovered"]))
        self.assertEqual(report["reconstructed"], [])

    def test_large_inpaint_is_reconstructed(self):
        ctx = _ctx((100, 100))
        ctx.damage_masks["final"] = _mask((100, 100), [(0, 0, 30, 100)])  # 30%
        report = build_restoration_report(ctx)
        self.assertEqual(report["recovered"], [])
        self.assertTrue(any("re-synthesized" in r for r in report["reconstructed"]))

    def test_face_inpaint_flag_uses_worst_face(self):
        ctx = _ctx((200, 200))
        # Face A (left) 50% inpainted, face B (right) untouched.
        ctx.damage_masks["final"] = _mask((200, 200), [(10, 10, 60, 110)])
        ctx.image_metadata["faces"] = [
            {"bbox": [10, 10, 60, 110]},   # 50x100 -> 50% coverage
            {"bbox": [120, 10, 170, 110]},  # untouched
        ]
        report = build_restoration_report(ctx)
        self.assertTrue(
            any("facial regions largely reconstructed" in r for r in report["reconstructed"])
        )

    def test_face_below_threshold_not_flagged(self):
        ctx = _ctx((200, 200))
        # 50x25 box inside the 50x100 face bbox = 25% < 40% threshold.
        ctx.damage_masks["final"] = _mask((200, 200), [(10, 10, 60, 35)])
        ctx.image_metadata["faces"] = [{"bbox": [10, 10, 60, 110]}]
        report = build_restoration_report(ctx)
        self.assertFalse(
            any("facial regions" in r for r in report["reconstructed"] + report["flagged"])
        )

    def test_colorization_sets_disclaimer_and_reconstructed(self):
        ctx = _ctx()
        ctx.image_metadata["color_type"] = "grayscale"
        report = build_restoration_report(ctx)
        self.assertTrue(report["colorization_disclaimer"])
        self.assertTrue(any("color" in r.lower() for r in report["reconstructed"]))

    def test_colorization_rollback_moves_claim_to_flagged(self):
        ctx = _ctx()
        ctx.image_metadata["color_type"] = "grayscale"
        ctx.image_metadata["colorization_rolled_back"] = True
        report = build_restoration_report(ctx)
        self.assertFalse(report["colorization_disclaimer"])
        self.assertFalse(report["reconstructed"])
        self.assertTrue(any("rolled back" in f for f in report["flagged"]))

    def test_colorize_disabled_never_claims_colorization(self):
        # Measured 2026-09-27: colorize=false on a sepia image still produced
        # the "colors added by AI" claim because the report inferred from
        # color_type alone. The claim requires positive evidence (Stage 6's
        # colorization_applied flag).
        ctx = _ctx()
        ctx.image_metadata["color_type"] = "sepia"
        ctx.image_metadata["colorization_applied"] = False
        report = build_restoration_report(ctx)
        self.assertFalse(report["colorization_disclaimer"])
        self.assertFalse(any("colors added" in r for r in report["reconstructed"]))

    def test_colorization_applied_flag_claims_disclaimer(self):
        ctx = _ctx()
        ctx.image_metadata["color_type"] = "grayscale"
        ctx.image_metadata["colorization_applied"] = True
        report = build_restoration_report(ctx)
        self.assertTrue(report["colorization_disclaimer"])

    def test_colorize_disabled_discloses_monochrome_kept(self):
        ctx = _ctx()
        ctx.image_metadata["color_type"] = "sepia"
        ctx.image_metadata["colorization_applied"] = False
        ctx.image_metadata["colorization_skipped_by_flag"] = True
        report = build_restoration_report(ctx)
        self.assertFalse(report["colorization_disclaimer"])
        self.assertTrue(
            any("colorization was disabled" in f for f in report["flagged"]))

    def test_gfpgan_flags(self):
        ctx = _ctx()
        ctx.image_metadata["gfpgan_applied"] = True
        ctx.image_metadata["gfpgan_deferred"] = True
        report = build_restoration_report(ctx)
        self.assertTrue(any("GFPGAN" in r for r in report["reconstructed"]))
        self.assertTrue(any("too small" in f for f in report["flagged"]))

    def test_circuit_breaker_and_identity_warning_flags(self):
        ctx = _ctx()
        ctx.image_metadata["warnings"] = [
            "inpaint_face_structure_erased_reverted",
            "identity_risk",
        ]
        report = build_restoration_report(ctx)
        self.assertTrue(any("reverted" in f for f in report["flagged"]))
        self.assertTrue(any("identity risk" in f for f in report["flagged"]))

    def test_despeckle_and_denoise_evidence_recovered(self):
        ctx = _ctx()
        ctx.image_metadata["specks_suppressed"] = True
        ctx.image_metadata["denoise_applied"] = True
        ctx.damage_masks["tearmark"] = _mask((64, 64), [(0, 0, 64, 2)])  # ~3%
        report = build_restoration_report(ctx)
        self.assertTrue(any("specks" in r for r in report["recovered"]))
        self.assertTrue(any("grain / noise" in r for r in report["recovered"]))
        self.assertTrue(any("surface blemishes" in r for r in report["recovered"]))

    def test_deskew_and_flat_field_recovered(self):
        ctx = _ctx()
        ctx.image_metadata["deskew_applied"] = True
        ctx.image_metadata["deskew_angle_deg"] = 1.25
        ctx.image_metadata["flat_field_applied"] = True
        report = build_restoration_report(ctx)
        self.assertTrue(any("skew corrected" in r for r in report["recovered"]))
        self.assertTrue(any("illumination" in r for r in report["recovered"]))

    def test_quality_service_warnings_land_in_flagged(self):
        ctx = _ctx()
        ctx.processing_log.append("WARNING: Image is very dark (mean brightness=8.3)")
        report = build_restoration_report(ctx)
        self.assertTrue(any("very dark" in f for f in report["flagged"]))

    def test_stage_timings_included(self):
        ctx = _ctx()
        ctx.stage_timings["inpainting"] = 1.234
        report = build_restoration_report(ctx)
        self.assertEqual(report["stage_timings"].get("inpainting"), 1.234)

    def test_never_raises_on_garbage(self):
        ctx = _ctx()
        ctx.image_metadata["faces"] = "not-a-list"  # type: ignore[assignment]
        ctx.damage_masks["final"] = None
        report = build_restoration_report(ctx)
        # Either a clean report or the explicit error dict — never an exception.
        self.assertIsInstance(report, dict)

    def test_threshold_constants(self):
        self.assertEqual(UNRECOVERABLE_COVERAGE_PCT, 25.0)
        self.assertEqual(FACE_INPAINT_FLAG_PCT, 40.0)


if __name__ == "__main__":
    unittest.main()
