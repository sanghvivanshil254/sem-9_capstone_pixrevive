"""Tests for the Stage-11 closed-loop output verifier (2026-09-25).

The verifier answers the project-owner question "how do we know the AI output
matches the original and isn't flaky?": it re-detects residue on the restored
image, micro-cleans it, and flags implausible re-synthesis (incoherent border
color wash, waxy over-smoothed inpaint regions).
"""

import os
import sys
import unittest

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ai.pipeline.pipeline_context import PipelineContext
from ai.validation.output_verifier import (
    detect_residue,
    verify_and_clean,
    verification_flags,
    _check_structural_symmetry,
    RESIDUE_CLEANUP_BAR_PCT,
)


def _speckled_portrait(with_border_wash=False, with_waxy_hair=False):
    """Synthetic portrait-ish frame: textured content + white speckle field."""
    rng = np.random.default_rng(5)
    gray = np.clip(120 + rng.normal(0, 9, (240, 240)), 0, 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=2)

    # Flake field: ~1-1.5% of frame — above the 0.5% cleanup bar but below
    # the 2.5% only-flag cap (250 two-pixel specks on a 240x240 frame ≈ 0.9%).
    ys = rng.integers(10, 230, 250)
    xs = rng.integers(10, 230, 250)
    for y, x in zip(ys, xs):
        rgb[y:y + 2, x:x + 2] = (245, 245, 245)

    if with_border_wash:
        # Cyan wash on the outer 10px only — strictly narrower than the
        # verifier's 14px border band (strip fraction 0.06 of 240), so the
        # wash is border-only and the interior stays neutral. (A 19px paint
        # leaked into the sampled interior and made both sides agree.)
        rgb[:, :10] = (170, 230, 240)  # low R, high B vs warm content
    if with_waxy_hair:
        # A large ultra-smooth blob (texture energy ~0) with textured surround.
        cv2.ellipse(rgb, (120, 60), (55, 28), 0, 0, 360, (95, 80, 70), -1)
        blob = np.zeros((240, 240), np.uint8)
        cv2.ellipse(blob, (120, 60), (55, 28), 0, 0, 360, 255, -1)
        smooth = cv2.GaussianBlur(rgb, (31, 31), 0)
        rgb[blob > 0] = smooth[blob > 0]
    return rgb


class TestResidueDetection(unittest.TestCase):
    def test_speckle_field_is_detected(self):
        rgb = _speckled_portrait()
        mask = detect_residue(rgb)
        pct = 100.0 * np.count_nonzero(mask) / mask.size
        self.assertGreater(pct, RESIDUE_CLEANUP_BAR_PCT,
                           "dense flake field must exceed the cleanup bar")

    def test_clean_textured_image_has_no_residue(self):
        rng = np.random.default_rng(9)
        gray = np.clip(120 + rng.normal(0, 8, (240, 240)), 0, 255).astype(np.uint8)
        rgb = np.stack([gray] * 3, axis=2)
        mask = detect_residue(rgb)
        pct = 100.0 * np.count_nonzero(mask) / mask.size
        self.assertLess(pct, 0.2, "plain texture must not read as residue")


class TestVerifyAndClean(unittest.TestCase):
    def test_cleanup_reduces_residue(self):
        rgb = _speckled_portrait()
        ctx = PipelineContext(rgb.copy())
        record = verify_and_clean(ctx)
        self.assertTrue(record["ran"])
        self.assertGreaterEqual(record["residue_pct_before"], RESIDUE_CLEANUP_BAR_PCT)
        self.assertTrue(record["cleaned"], f"record: {record}")
        self.assertLess(record["residue_pct_after"], record["residue_pct_before"])
        # Sanity: the canvas was actually replaced.
        after = detect_residue(ctx.working_image)
        self.assertEqual(record["residue_pct_after"],
                         round(100.0 * np.count_nonzero(after) / after.size, 3))

    def test_color_edge_incoherence_flagged(self):
        rgb = _speckled_portrait(with_border_wash=True)
        ctx = PipelineContext(rgb.copy())
        record = verify_and_clean(ctx)
        self.assertTrue(record.get("color_edge_incoherent"),
                        f"cyan border wash must be flagged: {record}")
        flags = verification_flags(record)
        self.assertTrue(any("colorization wash" in f for f in flags))

    def test_color_coherent_image_not_flagged(self):
        # Uniform warm tone everywhere — border agrees with content.
        rng = np.random.default_rng(3)
        base = np.clip(150 + rng.normal(0, 7, (240, 240)), 0, 255).astype(np.uint8)
        rgb = np.stack([base, base * 3 // 4, base // 2], axis=2).astype(np.uint8)
        ctx = PipelineContext(rgb.copy())
        record = verify_and_clean(ctx)
        self.assertFalse(record.get("color_edge_incoherent"), f"record: {record}")

    def test_waxy_oversmooth_inpaint_flagged(self):
        rgb = _speckled_portrait(with_waxy_hair=True)
        ctx = PipelineContext(rgb.copy())
        # Simulate the inpaint evidence: the blob region was "re-synthesized".
        mask = np.zeros((240, 240), np.uint8)
        cv2.ellipse(mask, (120, 60), (55, 28), 0, 0, 360, 255, -1)
        ctx.damage_masks["final"] = mask
        record = verify_and_clean(ctx)
        self.assertTrue(record.get("texture_oversmooth"),
                        f"waxy blob must be flagged: {record}")
        flags = verification_flags(record)
        self.assertTrue(any("smoother than their surroundings" in f for f in flags))

    def test_textured_inpaint_not_flagged(self):
        rng = np.random.default_rng(11)
        rgb = np.stack([np.clip(128 + rng.normal(0, 14, (240, 240)), 0, 255)] * 3, axis=2).astype(np.uint8)
        ctx = PipelineContext(rgb.copy())
        mask = np.zeros((240, 240), np.uint8)
        mask[80:160, 80:160] = 255  # inpainted region has the SAME texture
        ctx.damage_masks["final"] = mask
        record = verify_and_clean(ctx)
        self.assertFalse(record.get("texture_oversmooth"), f"record: {record}")

    def test_collar_dropped_on_one_side_flagged(self):
        """The shirt-collar tell: balanced structure in, unbalanced out."""
        rng = np.random.default_rng(4)
        # Input: textured torso spanning BOTH halves (collar continues).
        gray = np.clip(110 + rng.normal(0, 12, (240, 320)), 0, 255).astype(np.uint8)
        original = np.stack([gray] * 3, axis=2)
        # Output: right half's texture wiped into smooth wash (re-synthesis
        # dropped the collar on that side).
        restored = original.copy()
        smoothed = cv2.GaussianBlur(restored, (41, 41), 0)
        restored[:, 190:] = smoothed[:, 190:]
        result = _check_structural_symmetry(original, restored)
        self.assertTrue(result["asymmetry_invented"], f"result: {result}")
        flags = verification_flags({"ran": True, "asymmetry_invented": True,
                                    "input_balance": result["input_balance"],
                                    "output_balance": result["output_balance"]})
        self.assertTrue(any("fades on one side" in f for f in flags))

    def test_one_sided_input_damage_is_explained(self):
        """If the INPUT was already unbalanced, the output may be too."""
        rng = np.random.default_rng(6)
        gray = np.clip(110 + rng.normal(0, 12, (240, 320)), 0, 255).astype(np.uint8)
        original = np.stack([gray] * 3, axis=2)
        smoothed = cv2.GaussianBlur(original, (41, 41), 0)
        original_unbalanced = original.copy()
        original_unbalanced[:, 190:] = smoothed[:, 190:]  # damage ate the right side
        # Output matches the input's imbalance — explained, not invented.
        result = _check_structural_symmetry(original_unbalanced, original_unbalanced.copy())
        self.assertFalse(result["asymmetry_invented"], f"result: {result}")

    def test_verifier_never_raises_on_degenerate_input(self):
        for img in (
            np.zeros((4, 4, 3), np.uint8),
            np.full((64, 64, 3), 255, np.uint8),
        ):
            ctx = PipelineContext(img.copy())
            record = verify_and_clean(ctx)  # must not raise
            self.assertTrue(record["ran"])


if __name__ == "__main__":
    unittest.main()
