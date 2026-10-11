"""face_identity_gate.py — SFace identity verification for the re-imagine stage.

Phase H1 (docs/HYBRID_DIFFUSION_DESIGN.md section 6). When Stage 3d
re-imagines destroyed regions, the generated face must still be the *same
person*. CLIP-embedding cosine cannot certify that — two different faces
score 0.6-0.8 — so the active border control is SFace, a trained face
verification model that ships inside the already-pinned opencv-contrib
(cv2.FaceRecognizerSF, zero extra installs):

  YuNet  (cv2.FaceDetectorYN)     finds the largest face in each image
  SFace  (cv2.FaceRecognizerSF)   128-D embedding, cosine match

Verdict ladder (thresholds calibrated on case fixtures in H1 before freeze):
  cosine >= PASS_COSINE        -> "pass"    (recorded in the report)
  FLAG_COSINE..PASS_COSINE     -> "flag"    (kept, disclosed as drifted)
  cosine <  FLAG_COSINE        -> "rollback" (caller re-rolls seed, then
                                              reverts Stage 3d entirely)

The gate never mutates images and never raises into the pipeline: any
detection/ONNX failure returns a None similarity with the reason, which the
caller treats as "cannot certify" -> flag (never a silent pass).
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Anchored to this file, not the process CWD: the uvicorn server runs with
# cwd=backend/ (measured 2026-09-27 — a CWD-relative default broke the gate
# inside the live server while standalone scripts from the repo root worked).
WeightsDirDefault = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "weights", "sface")

YUNET_FILE = "yunet_270face.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"

# Design-doc starting priors; frozen only after the H1 calibration sweep.
PASS_COSINE = 0.42
FLAG_COSINE = 0.36


def _env_float(name: str, default: float) -> float:
    """Env override for the ladder (calibration + host tuning); invalid -> default."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("invalid %s=%r — using default %.2f", name, raw, default)
        return default


def pass_cosine_env() -> float:
    """Effective pass threshold: REIMAGINE_PASS_COSINE overrides the constant."""
    return _env_float("REIMAGINE_PASS_COSINE", PASS_COSINE)


def flag_cosine_env() -> float:
    """Effective flag threshold: REIMAGINE_FLAG_COSINE overrides the constant."""
    return _env_float("REIMAGINE_FLAG_COSINE", FLAG_COSINE)

# YuNet wants reasonably-sized faces; upscale small inputs before detect.
_MIN_DETECT_SIDE = 320
_DETECT_SCORE = 0.6


class SFaceIdentityGate:
    """Compare faces between the original and a re-imagined output."""

    def __init__(self, weights_dir: str | None = None) -> None:
        base = weights_dir or WeightsDirDefault
        self._yunet_path = os.path.join(base, YUNET_FILE)
        self._sface_path = os.path.join(base, SFACE_FILE)
        self._detector = None
        self._recognizer = None

    # -- availability ------------------------------------------------------

    def weights_available(self) -> bool:
        return os.path.exists(self._yunet_path) and os.path.exists(self._sface_path)
    def _ensure_loaded(self) -> bool:
        if self._detector is not None and self._recognizer is not None:
            return True
        if not self.weights_available():
            return False
        try:
            self._detector = cv2.FaceDetectorYN.create(
                self._yunet_path, "", (320, 320), _DETECT_SCORE
            )
            self._recognizer = cv2.FaceRecognizerSF.create(self._sface_path, "")
            return True
        except Exception as exc:  # noqa: BLE001 - never crash the pipeline
            logger.warning("SFace gate failed to load ONNX models: %s", exc)
            self._detector = None
            self._recognizer = None
            return False

    # -- detection ---------------------------------------------------------

    def _largest_face(self, image: np.ndarray):
        """(bbox, aligned_crop) for the largest face, or None.

        image: HxWx3 RGB or BGR uint8 (converted to BGR for cv2).
        """
        if not self._ensure_loaded():
            return None
        bgr = image
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        # YuNet input-size contract: setInputSize before each detect.
        h, w = bgr.shape[:2]
        scale = 1.0
        if max(h, w) < _MIN_DETECT_SIDE:
            scale = _MIN_DETECT_SIDE / max(h, w)
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
        self._detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        ok, faces = self._detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        # Largest by box area; rows are [x, y, w, h, land(10), score].
        areas = faces[:, 2] * faces[:, 3]
        face = faces[int(np.argmax(areas))]
        # alignCrop wants coords in the *detected* image's frame.
        aligned = self._recognizer.alignCrop(bgr, face)
        x, y, bw, bh = (face[0] / scale, face[1] / scale,
                        face[2] / scale, face[3] / scale)
        return (int(x), int(y), int(bw), int(bh)), aligned

    def _all_faces(self, image: np.ndarray, max_faces: int = 0):
        """[(bbox, aligned_crop), ...] area-descending (largest first), or [].

        max_faces=0 means every face YuNet finds; a positive cap keeps the
        N largest. Same detection/upscaling contract as _largest_face.
        """
        if not self._ensure_loaded():
            return []
        bgr = image
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.ndim == 3 and bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        h, w = bgr.shape[:2]
        scale = 1.0
        if max(h, w) < _MIN_DETECT_SIDE:
            scale = _MIN_DETECT_SIDE / max(h, w)
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
        self._detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        ok, faces = self._detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return []
        order = np.argsort(faces[:, 2] * faces[:, 3])[::-1]
        if max_faces and max_faces > 0:
            order = order[:max_faces]
        found = []
        for idx in order:
            face = faces[int(idx)]
            aligned = self._recognizer.alignCrop(bgr, face)
            x, y, bw, bh = (face[0] / scale, face[1] / scale,
                            face[2] / scale, face[3] / scale)
            found.append(((int(x), int(y), int(bw), int(bh)), aligned))
        return found

    # -- comparison --------------------------------------------------------

    def compare(self, image_a: np.ndarray, image_b: np.ndarray,
                max_faces: int = 1) -> dict:
        """SFace cosine between the faces of two images.

        max_faces=1 (default) keeps the legacy largest-face contract;
        max_faces=0 (or a larger cap) verifies EVERY face YuNet finds and
        aggregates pessimistically (Phase-H3 hardening): the score is the
        WORST per-face cosine, so drift on any single face — not just the
        biggest one — can roll the re-imagine back.

        Returns {"similarity": float|None, "reason": str, "face_a",
        "face_b" bbox|None, "faces_a"/"faces_b" bbox lists, "per_face":
        [(bbox, cosine), ...]}. similarity None = could not certify (no
        face, no weights, ONNX failure) — the caller must treat that as a
        flag.
        """
        out: dict = {"similarity": None, "reason": "", "face_a": None,
                     "face_b": None, "faces_a": [], "faces_b": [],
                     "per_face": []}
        if not self._ensure_loaded():
            out["reason"] = "sface_weights_unavailable"
            return out
        try:
            fa_list = self._all_faces(image_a, max_faces)
            fb_list = self._all_faces(image_b, max_faces)
        except Exception as exc:  # noqa: BLE001
            out["reason"] = f"sface_error: {exc}"
            return out
        if not fa_list or not fb_list:
            out["reason"] = ("no_face_detected" if not fa_list and not fb_list
                             else "no_face_in_original" if not fa_list
                             else "no_face_in_result")
            out["faces_a"] = [bbox for bbox, _ in fa_list]
            out["faces_b"] = [bbox for bbox, _ in fb_list]
            out["face_a"] = out["faces_a"][0] if out["faces_a"] else None
            out["face_b"] = out["faces_b"][0] if out["faces_b"] else None
            return out
        out["faces_a"] = [bbox for bbox, _ in fa_list]
        out["faces_b"] = [bbox for bbox, _ in fb_list]
        out["face_a"] = out["faces_a"][0]
        out["face_b"] = out["faces_b"][0]
        try:
            # SFace features are matched bbox-to-bbox in area order: face #k
            # in A against face #k in B. A true crowd re-ordering (same
            # people, swapped positions) can look like drift — disclosed as
            # a low worst-cosine, never silently passed.
            feats_a = [self._recognizer.feature(al) for _, al in fa_list]
            feats_b = [self._recognizer.feature(al) for _, al in fb_list]
            pairs = min(len(feats_a), len(feats_b))
            cosines = [float(self._recognizer.match(
                feats_a[i], feats_b[i], cv2.FaceRecognizerSF_FR_COSINE))
                for i in range(pairs)]
        except Exception as exc:  # noqa: BLE001
            out["reason"] = f"sface_error: {exc}"
            return out
        out["per_face"] = [(fa_list[i][0], round(cosines[i], 4))
                           for i in range(pairs)]
        if len(feats_a) != len(feats_b):
            out["reason"] = "face_count_mismatch"
        out["similarity"] = min(cosines)
        out["reason"] = out["reason"] or "ok"
        return out

    # -- verdict -----------------------------------------------------------

    def check(
        self,
        original: np.ndarray,
        result: np.ndarray,
        pass_cosine: float | None = None,
        flag_cosine: float | None = None,
        max_faces: int = 0,
    ) -> dict:
        """Verdict ladder for the re-imagine loop.

        {"action": "pass"|"flag"|"rollback", "similarity": float|None,
         "reason": str, "faces": per-face [(bbox, cosine)]}. A None
        similarity is a "flag" (cannot certify), never a pass and never by
        itself a rollback.

        Thresholds default to the env-overridable getters (REIMAGINE_PASS_COSINE
        / REIMAGINE_FLAG_COSINE) over the module constants; max_faces=0
        verifies every detected face (worst cosine governs — H3 hardening).
        """
        pc = pass_cosine if pass_cosine is not None else pass_cosine_env()
        fc = flag_cosine if flag_cosine is not None else flag_cosine_env()
        cmp = self.compare(original, result, max_faces=max_faces)
        sim = cmp["similarity"]
        out = {"action": "flag", "similarity": sim, "reason": cmp["reason"],
               "faces": cmp.get("per_face", [])}
        if sim is None:
            out["reason"] = cmp["reason"] or "sface_unavailable"
            return out
        if sim >= pc:
            out["action"] = "pass"
        elif sim >= fc:
            out["action"] = "flag"
        else:
            out["action"] = "rollback"
        return out
