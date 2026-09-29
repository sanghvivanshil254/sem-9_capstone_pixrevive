"""Phase V0 tests — semantic_validator.py + stage wiring + report block.

Report-only battery per docs/SEMANTIC_VALIDATION_PLAN.md: findings are
metadata, never pixel edits. Includes the regression for the measured
classify_evidence denominator bug (full-canvas masks deflated lost-share).
"""

import unittest

import numpy as np

from ai.validation.semantic_validator import (
    ORIGINAL_MAX,
    DESTROYED_MIN,
    classify_evidence,
    rule_a1_eye_presence,
    rule_a3_hairline_continuity,
    rule_c1_bilateral_structure,
    run_semantic_validation,
)


def _canvas() -> np.ndarray:
    """400x400 gray-equivalent RGB with a dark torso band + left garment edge."""
    img = np.full((400, 400, 3), 128, np.uint8)
    img[240:400, 80:320] = (90, 90, 90)          # torso band
    img[250:370, 118:123] = (30, 30, 30)         # left-side vertical structure
    return img


def _gray(img: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


FACE = {"bbox": [120, 80, 280, 240]}


class TestEvidenceClassification(unittest.TestCase):
    def test_pristine_is_original(self):
        ev = np.zeros((400, 400), np.uint8)
        region = np.zeros((400, 400), np.uint8)
        region[100:170, 200:270] = 1
        self.assertEqual(classify_evidence(region, ev), "ORIGINAL")

    def test_fully_destroyed_is_destroyed(self):
        ev = np.zeros((400, 400), np.uint8)
        region = np.zeros((400, 400), np.uint8)
        region[100:170, 200:270] = 1
        ev[100:170, 200:270] = 1
        self.assertEqual(classify_evidence(region, ev), "DESTROYED")

    def test_full_canvas_region_denominator_regression(self):
        """Regression (measured V0 bug): region masks are full-canvas shaped;
        the denominator must be in-region pixels, not region_mask.size (which
        deflated a 100%-destroyed quadrant down to 2.4% 'lost')."""
        ev = np.zeros((400, 400), np.uint8)
        region = np.zeros((400, 400), np.uint8)
        region[100:170, 200:270] = 1
        ev[100:170, 200:270] = 1
        # If this were computed over region.size it would be ~0.024 -> ORIGINAL.
        self.assertEqual(classify_evidence(region, ev), "DESTROYED")

    def test_partial_is_ambiguous(self):
        ev = np.zeros((400, 400), np.uint8)
        region = np.zeros((400, 400), np.uint8)
        region[100:170, 200:270] = 1
        sub = region > 0
        ev[sub] = np.where(
            np.random.default_rng(0).random(int(sub.sum())) < 0.4, 1, 0
        ).astype(np.uint8)
        self.assertEqual(classify_evidence(region, ev), "AMBIGUOUS")

    def test_thresholds_ordered(self):
        self.assertLess(ORIGINAL_MAX, DESTROYED_MIN)


class TestRuleA1EyePresence(unittest.TestCase):
    def test_destroyed_quadrant_without_structure_warns(self):
        img = _canvas()
        ev = np.zeros((400, 400), np.uint8)
        ev[100:170, 200:270] = 1  # right eye quadrant destroyed, flat fill
        f = rule_a1_eye_presence(_gray(img), ev, FACE)
        self.assertIsNotNone(f)
        self.assertTrue(any(x["region"] == "right_eye" for x in f))
        right = next(x for x in f if x["region"] == "right_eye")
        self.assertEqual(right["evidence_state"], "DESTROYED")
        self.assertFalse(right["residual_structure"])
        self.assertEqual(right["severity"], "warning")
        self.assertEqual(right["action"], "report_only")

    def test_original_eye_is_silent(self):
        img = _canvas()
        ev = np.zeros((400, 400), np.uint8)
        self.assertIsNone(rule_a1_eye_presence(_gray(img), ev, FACE))

    def test_destroyed_with_residual_structure_is_info(self):
        import cv2
        img = _canvas()
        img[120:140, 225:245] = (20, 20, 20)  # dark blob in right eye quadrant
        ev = np.zeros((400, 400), np.uint8)
        ev[100:170, 200:270] = 1
        f = rule_a1_eye_presence(_gray(img), ev, FACE)
        right = next(x for x in f if x["region"] == "right_eye")
        self.assertTrue(right["residual_structure"])
        self.assertEqual(right["severity"], "info")


class TestRuleA3Hairline(unittest.TestCase):
    def test_destroyed_textured_hair_is_candidate(self):
        rng = np.random.default_rng(3)
        img = _canvas()
        img[20:80, 100:300] = rng.integers(40, 200, (60, 200, 3), dtype=np.uint8)
        ev = np.zeros((400, 400), np.uint8)
        ev[20:80, 100:300] = 1  # textured hair region fully destroyed
        f = rule_a3_hairline_continuity(_gray(img), ev, FACE)
        self.assertIsNotNone(f)
        self.assertEqual(f[0]["rule"], "A3.hairline_continuity")
        self.assertEqual(f[0]["evidence_state"], "DESTROYED")

    def test_destroyed_flat_hair_band_is_bald_note_not_candidate(self):
        img = _canvas()  # flat 128 fill above face — no hair texture
        ev = np.zeros((400, 400), np.uint8)
        ev[20:80, 100:300] = 1
        f = rule_a3_hairline_continuity(_gray(img), ev, FACE)
        self.assertIsNotNone(f)
        self.assertIn("bald", f[0]["verdict"])
        self.assertEqual(f[0]["severity"], "info")

    def test_surviving_hair_is_silent(self):
        rng = np.random.default_rng(3)
        img = _canvas()
        img[20:80, 100:300] = rng.integers(40, 200, (60, 200, 3), dtype=np.uint8)
        ev = np.zeros((400, 400), np.uint8)
        self.assertIsNone(rule_a3_hairline_continuity(_gray(img), ev, FACE))


class TestRuleC1Bilateral(unittest.TestCase):
    def test_no_structure_anywhere_is_silent(self):
        img = np.full((400, 400, 3), 128, np.uint8)  # no torso structure at all
        ev = np.zeros((400, 400), np.uint8)
        ev[250:380, 200:340] = 1
        self.assertIsNone(rule_c1_bilateral_structure(_gray(img), ev, FACE))

    def test_destroyed_mirror_of_structured_side_fires(self):
        # Full-width flat torso (no canvas-boundary edges inside the regions),
        # strong left-only structure, right half destroyed (post-LaMa smooth).
        img = np.full((400, 400, 3), 128, np.uint8)
        img[240:400, :] = (90, 90, 90)         # full-width torso band
        img[250:370, 100:106] = (20, 20, 20)   # left-side structure lines
        img[250:370, 150:156] = (20, 20, 20)
        ev = np.zeros((400, 400), np.uint8)
        ev[250:380, 200:340] = 1               # right torso destroyed
        f = rule_c1_bilateral_structure(_gray(img), ev, FACE)
        self.assertIsNotNone(f)
        self.assertEqual(f[0]["rule"], "C1.bilateral_structure")
        self.assertIn("continuation", f[0]["verdict"])

    def test_symmetric_surviving_structure_is_silent(self):
        """Both sides survive with mirrored structure: evidence governs, no
        finding (Prime Directive — surviving asymmetry/symmetry is truth)."""
        img = np.full((400, 400, 3), 128, np.uint8)
        img[240:400, :] = (90, 90, 90)
        img[250:370, 140:146] = (20, 20, 20)   # left edge
        img[250:370, 254:260] = (20, 20, 20)   # mirrored right edge
        ev = np.zeros((400, 400), np.uint8)    # nothing destroyed
        self.assertIsNone(rule_c1_bilateral_structure(_gray(img), ev, FACE))


class TestRunner(unittest.TestCase):
    def test_pristine_image_yields_zero_findings(self):
        img = _canvas()
        out = run_semantic_validation(img, np.zeros((400, 400), np.uint8), [FACE])
        self.assertEqual(out["findings"], [])
        self.assertEqual(out["summary"], {"eyes": 0, "hair": 0, "garment": 0})
        self.assertEqual(out["version"], "v0-report-only")

    def test_prime_directive_no_findings_touch_original(self):
        """Every finding either references lost evidence (report_only) or is
        downgraded to none_evidence_governs; nothing may propose modifying
        surviving pixels, and no finding may sit on ORIGINAL evidence with an
        actionable action."""
        img = _canvas()
        ev = np.zeros((400, 400), np.uint8)
        ev[100:170, 200:270] = 1
        out = run_semantic_validation(img, ev, [FACE])
        for f in out["findings"]:
            self.assertNotEqual(f.get("action"), "modify_original")
            if f["evidence_state"] == "ORIGINAL":
                self.assertEqual(f.get("action"), "none_evidence_governs")
            else:
                self.assertEqual(f.get("action"), "report_only")
                self.assertIn(f["evidence_state"], ("DESTROYED", "AMBIGUOUS"))

    def test_findings_are_pure_metadata(self):
        """No function may mutate inputs (report-only contract)."""
        img = _canvas()
        ev = np.zeros((400, 400), np.uint8)
        ev[100:170, 200:270] = 1
        img_copy, ev_copy = img.copy(), ev.copy()
        run_semantic_validation(img, ev, [FACE])
        self.assertTrue(np.array_equal(img, img_copy))
        self.assertTrue(np.array_equal(ev, ev_copy))

    def test_no_faces_is_clean(self):
        out = run_semantic_validation(_canvas(), np.zeros((400, 400), np.uint8), [])
        self.assertEqual(out["faces_evaluated"], 0)
        self.assertEqual(out["findings"], [])


if __name__ == "__main__":
    unittest.main()
