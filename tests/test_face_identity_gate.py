"""Tests for the SFace identity gate (Phase H1, design doc section 6).

Two layers:
* Verdict-ladder unit tests run unconditionally (pure thresholds).
* Real YuNet+SFace checks against the case9 portrait fixture are
  weights-gated: they skip when the ONNX files are absent (CI pattern from
  test_deblur.py), and pass fast when they are present.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
)

from ai.validation.face_identity_gate import SFaceIdentityGate  # noqa: E402

FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "restoration_dataset", "damaged", "case9_teared_sepia.jpg",
)


def _gate() -> SFaceIdentityGate:
    return SFaceIdentityGate()


def _weights_present() -> bool:
    return _gate().weights_available()


class TestVerdictLadder(unittest.TestCase):
    """check() thresholds — run with a stubbed compare, no models needed."""

    def _check(self, sim):
        gate = _gate()
        with mock.patch.object(gate, "compare",
                                        return_value={"similarity": sim,
                                                      "reason": "ok",
                                                      "face_a": [0, 0, 10, 10],
                                                      "face_b": [0, 0, 10, 10]}):
            return gate.check(np.zeros((8, 8, 3), np.uint8),
                              np.zeros((8, 8, 3), np.uint8))

    def test_pass_above(self):
        self.assertEqual(self._check(0.42)["action"], "pass")
        self.assertEqual(self._check(0.93)["action"], "pass")

    def test_flag_band_keeps_but_discloses(self):
        for sim in (0.36, 0.41):
            self.assertEqual(self._check(sim)["action"], "flag")

    def test_rollback_below(self):
        self.assertEqual(self._check(0.1)["action"], "rollback")
        self.assertEqual(self._check(0.359)["action"], "rollback")

    def test_uncertified_similarity_is_flag_never_pass(self):
        gate = _gate()
        with mock.patch.object(gate, "compare",
                                return_value={"similarity": None,
                                              "reason": "no_face_detected",
                                              "face_a": None, "face_b": None}):
            out = gate.check(np.zeros((8, 8, 3), np.uint8),
                             np.zeros((8, 8, 3), np.uint8))
        self.assertEqual(out["action"], "flag")
        self.assertEqual(out["reason"], "no_face_detected")

    def test_boundary_is_inclusive_at_pass(self):
        # 0.42 exactly is >= PASS → pass (>= semantics).
        self.assertEqual(self._check(0.42)["action"], "pass")

    def test_custom_thresholds(self):
        gate = _gate()
        with mock.patch.object(gate, "compare",
                                return_value={"similarity": 0.30,
                                              "reason": "ok",
                                              "face_a": None, "face_b": None}):
            out = gate.check(np.zeros((8, 8, 3), np.uint8),
                             np.zeros((8, 8, 3), np.uint8),
                             pass_cosine=0.25, flag_cosine=0.20)
        self.assertEqual(out["action"], "pass")


class TestAllFacesAggregation(unittest.TestCase):
    """Worst-cosine aggregation over every detected face (H3 hardening)."""

    def test_check_reports_per_face_list(self):
        gate = _gate()
        with mock.patch.object(gate, "compare",
                                return_value={"similarity": 0.38, "reason": "ok",
                                              "face_a": [0, 0, 9, 9],
                                              "face_b": [0, 0, 9, 9],
                                              "per_face": [((0, 0, 9, 9), 0.5),
                                                           ((4, 4, 8, 8), 0.38)]}):
            out = gate.check(np.zeros((8, 8, 3), np.uint8),
                             np.zeros((8, 8, 3), np.uint8))
        self.assertEqual(out["action"], "flag")  # worst (0.38) sits in the band
        self.assertEqual(len(out["faces"]), 2)
        self.assertEqual(out["faces"][1], ((4, 4, 8, 8), 0.38))

    def test_env_overrides_ladder(self):
        gate = _gate()
        with mock.patch.object(gate, "compare",
                                return_value={"similarity": 0.30,
                                              "reason": "ok",
                                              "face_a": None, "face_b": None}):
            with mock.patch.dict(os.environ, {"REIMAGINE_PASS_COSINE": "0.25"}):
                out = gate.check(np.zeros((8, 8, 3), np.uint8),
                                 np.zeros((8, 8, 3), np.uint8))
            self.assertEqual(out["action"], "pass")
            with mock.patch.dict(os.environ, {"REIMAGINE_FLAG_COSINE": "0.30"}):
                out = gate.check(np.zeros((8, 8, 3), np.uint8),
                                 np.zeros((8, 8, 3), np.uint8))
            self.assertEqual(out["action"], "flag")  # 0.30 >= env flag 0.30 -> kept, disclosed
        # No env: module constants govern.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REIMAGINE_PASS_COSINE", None)
            os.environ.pop("REIMAGINE_FLAG_COSINE", None)
            gate2 = _gate()
            with mock.patch.object(gate2, "compare",
                                    return_value={"similarity": 0.30,
                                                  "reason": "ok",
                                                  "face_a": None,
                                                  "face_b": None}):
                out = gate2.check(np.zeros((8, 8, 3), np.uint8),
                                  np.zeros((8, 8, 3), np.uint8))
            self.assertEqual(out["action"], "rollback")

    @unittest.skipUnless(_weights_present(),
                         "SFace/YuNet ONNX weights not downloaded yet")
    def test_multi_face_real_pair_aggregates(self):
        import cv2
        gate = _gate()
        base = cv2.imread(FIXTURE)
        assert base is not None
        # A two-face composite of the same portrait: worst-cosine must still
        # be ~1.0 for identical content, and compare(max_faces=0) must find
        # both faces (YuNet may legitimately merge one; accept 1-2).
        h, w = base.shape[:2]
        canvas = np.zeros((h, w * 2, 3), np.uint8)
        canvas[:, :w] = base
        canvas[:, w:] = base
        out = gate.compare(canvas, canvas, max_faces=0)
        self.assertEqual(out["reason"], "ok")
        self.assertGreaterEqual(len(out["per_face"]), 1)
        self.assertGreater(out["similarity"], 0.95)


class TestFailSafe(unittest.TestCase):
    """Any failure path returns uncertified, never raises."""

    def test_missing_weights_gives_flag_reason(self):
        gate = _gate()
        if _weights_present():
            self.skipTest("weights present; nothing to fail")
        out = gate.check(np.zeros((8, 8, 3), np.uint8),
                         np.zeros((8, 8, 3), np.uint8))
        self.assertEqual(out["action"], "flag")
        self.assertEqual(out["reason"], "sface_weights_unavailable")

    def test_compare_never_raises_on_garbage(self):
        if not _weights_present():
            self.skipTest("SFace/YuNet weights not downloaded")
        gate = _gate()
        for junk in (np.zeros((4, 4, 3), np.uint8), np.full((16, 16, 3), 255, np.uint8)):
            out = gate.compare(junk, junk)
            self.assertIn("reason", out)
            self.assertTrue(out["similarity"] is None or
                            isinstance(out["similarity"], float))


@unittest.skipUnless(_weights_present(),
                     "SFace/YuNet ONNX weights not downloaded yet")
class TestRealFaces(unittest.TestCase):
    """Real-model checks against the case9 portrait (weights-gated)."""

    @classmethod
    def setUpClass(cls):
        import cv2
        cls.cv2 = cv2
        cls.image = cv2.imread(FIXTURE)
        assert cls.image is not None, f"fixture unreadable: {FIXTURE}"

    def test_self_similarity_is_high(self):
        gate = _gate()
        out = gate.compare(self.image, self.image)
        self.assertEqual(out["reason"], "ok", out)
        self.assertIsNotNone(out["similarity"])
        assert out["similarity"] is not None
        # Same pixels → same embedding → 1.0 (float tolerance).
        self.assertGreater(out["similarity"], 0.95)

    def test_face_found_in_case9(self):
        gate = _gate()
        out = gate.compare(self.image, self.image)
        self.assertIsNotNone(out["face_a"], "YuNet found no face in case9")

    def test_different_images_score_lower_than_self(self):
        gate = _gate()
        # A *mildly* degraded same-face image must score below self-match —
        # mild enough that YuNet still detects the face (heavy blur erases
        # it entirely, which the detector-fail-safe tests already cover).
        smeared = self.cv2.GaussianBlur(self.image, (7, 7), 2.5)
        smeared = (smeared.astype(np.float32) * 0.92).clip(0, 255).astype(np.uint8)
        out_same = gate.compare(self.image, self.image)
        out_smear = gate.compare(self.image, smeared)
        assert out_same["similarity"] is not None
        assert out_smear["similarity"] is not None, out_smear
        self.assertLess(out_smear["similarity"], out_same["similarity"])

    def test_check_ladder_on_real_pair(self):
        gate = _gate()
        verdict = gate.check(self.image, self.image)
        self.assertIn(verdict["action"], ("pass", "flag"))
        self.assertEqual(verdict["action"], "pass")  # identical image → pass


if __name__ == "__main__":
    unittest.main()
