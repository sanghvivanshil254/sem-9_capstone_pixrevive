"""Phase 4B.1 (T-4) tests: auto-deskew.

Estimator: recovers the dominant skew from long straight structures, folds
the horizontal and vertical edge families onto one vote, and — critically —
stays inert when the evidence is scattered (natural photos have edges in
every direction; the same lesson as Stage 0's autocorrelation gate).

Corrector: bitwise-inert below the 0.3° plan gate, affine above it, and the
rotation wedges must fall in margins that the border crop then removes.
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ai.preprocessing.image_normalizer import ImageNormalizer  # noqa: E402


def _scan_like_canvas(h=700, w=900, base=205):
    """A matte-white scan canvas with strong black content bars — the long
    straight edges the Hough estimator needs."""
    img = np.full((h, w, 3), base, dtype=np.uint8)
    img[60:110, 80:820] = 25          # long horizontal bar (740 px)
    img[560:610, 80:820] = 25         # second long horizontal bar
    img[140:520, 120:160] = 25        # long vertical bar
    return img


def _rotate(img, deg):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    return cv2.warpAffine(
        img, m, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _bar_angle(img):
    """Ground-truth angle of the top bar: the angle of the principal axis of
    the largest dark connected component, via image moments."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, bin_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img)
    if n < 2:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = cv2.moments((labels == idx).astype(np.uint8))
    if m["m00"] == 0:
        return None
    theta = 0.5 * np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"])
    deg = np.degrees(theta)
    # Principal-axis sign ambiguity; keep in [-45, 45).
    while deg >= 45.0:
        deg -= 90.0
    while deg < -45.0:
        deg += 90.0
    return deg


class TestSkewEstimator(unittest.TestCase):
    def test_skewed_scan_angle_recovered(self):
        # The estimator returns the CORRECTION angle: a scan warped with
        # rotation parameter +t measures -t (edges at -t; applying -t via
        # getRotationMatrix2D restores axis alignment).
        for deg in (2.0, -3.0, 5.0):
            with self.subTest(deg=deg):
                skew = ImageNormalizer.estimate_skew_angle(_rotate(_scan_like_canvas(), deg))
                self.assertIsNotNone(skew)
                self.assertAlmostEqual(skew, -deg, delta=0.8)

    def test_inert_on_straight_scan(self):
        skew = ImageNormalizer.estimate_skew_angle(_scan_like_canvas())
        self.assertAlmostEqual(skew, 0.0, delta=0.15,
                               msg="straight scan must not measure a skew")

    def test_inert_on_natural_photo_evidence(self):
        """Scattered orientation (natural scene) must not produce a dominant
        vote — the estimator fails safe to 0.0."""
        rng = np.random.default_rng(11)
        img = np.full((512, 512, 3), 200, dtype=np.uint8)
        for _ in range(60):  # edges in every direction
            y0, x0 = rng.integers(0, 460, 2)
            y1, x1 = y0 + rng.integers(-200, 200), x0 + rng.integers(-200, 200)
            cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (30, 30, 30), 2)
        self.assertEqual(ImageNormalizer.estimate_skew_angle(img), 0.0)

    def test_inert_on_flat_image(self):
        self.assertEqual(ImageNormalizer.estimate_skew_angle(
            np.full((400, 400, 3), 128, dtype=np.uint8)), 0.0)

    def test_tiny_image_inert(self):
        self.assertEqual(ImageNormalizer.estimate_skew_angle(
            np.full((40, 40, 3), 128, dtype=np.uint8)), 0.0)


class TestDeskewCorrection(unittest.TestCase):
    def test_below_gate_bitwise_unchanged(self):
        img = _scan_like_canvas()
        out, angle = ImageNormalizer.deskew(img)
        self.assertEqual(angle, 0.0)
        self.assertTrue(np.array_equal(out, img))

    def test_3_degree_scan_corrected(self):
        skewed = _rotate(_scan_like_canvas(), 3.0)
        out, angle = ImageNormalizer.deskew(skewed)
        self.assertAlmostEqual(angle, -3.0, delta=0.8)  # correction angle
        self.assertGreater(abs(angle), ImageNormalizer.MIN_SKEW_DEG)
        # The dominant content bar must now be axis-aligned: its principal
        # angle should be near 0 within the interpolation tolerance.
        residual = _bar_angle(out)
        self.assertIsNotNone(residual)
        self.assertLess(abs(residual), 1.0)

    def test_beyond_max_range_never_rotates(self):
        # A true -45deg fold-boundary content orientation must not be read as
        # skew and rotated: |angle| > MAX_SKEW_DEG → inert.
        img = _scan_like_canvas()
        m = cv2.getRotationMatrix2D((450, 350), 30.0, 1.0)
        diag = cv2.warpAffine(img, m, (900, 700),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))
        out, angle = ImageNormalizer.deskew(diag)
        self.assertEqual(angle, 0.0)
        self.assertTrue(np.array_equal(out, diag))

    def test_rotation_wedges_removed_by_border_crop(self):
        """The full T-4 story: deskew THEN crop must leave no white wedges."""
        skewed = _rotate(_scan_like_canvas(), 2.5)
        deskewed, angle = ImageNormalizer.deskew(skewed)
        self.assertNotEqual(angle, 0.0)
        cropped, has_border, _ = ImageNormalizer.detect_and_crop_borders(deskewed)
        # Every border row/col of the crop must be content (dark bar or photo
        # base), not the 255 mat fill the rotation painted.
        self.assertLess(float(cropped[0].mean()), 250.0)
        self.assertLess(float(cropped[-1].mean()), 250.0)
        self.assertLess(float(cropped[:, 0].mean()), 250.0)
        self.assertLess(float(cropped[:, -1].mean()), 250.0)

    def test_deskew_and_crop_net_geometry(self):
        """End-to-end geometry: cropping the deskewed -2.5deg scan recovers
        approximately the original content rectangle."""
        original = _scan_like_canvas()
        skewed = _rotate(original, -2.5)
        deskewed, _ = ImageNormalizer.deskew(skewed)
        cropped, _, _ = ImageNormalizer.detect_and_crop_borders(deskewed)
        # Original bars: h=[60..110, 560..610], first bar top at y=60 of 700.
        # The crop trims the wedge margins; the first bar must sit in the top
        # ~20% of the cropped frame, as in the original (60/700 ≈ 8.6%).
        gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        dark_rows = np.where((gray < 100).mean(axis=1) > 0.5)[0]
        self.assertGreater(len(dark_rows), 0)
        first_bar_top = dark_rows.min() / cropped.shape[0]
        self.assertLess(first_bar_top, 0.20)


if __name__ == "__main__":
    unittest.main()
