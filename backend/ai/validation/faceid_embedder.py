"""faceid_embedder.py — ArcFace identity embeddings for FaceID conditioning.

Phase H4 (docs/HYBRID_DIFFUSION_DESIGN.md, H4 addendum). The H3 calibration
sweep measured the structural ceiling of plain CLIP-style IP-Adapter
conditioning: seed variance (cosine spread 0.21-0.38) dwarfs any scale effect
(~0.03), so no CLIP-image conditioning reliably preserves identity on
hopeless coverage. FaceID-class conditioning attacks the actual problem: it
feeds generation an *identity* embedding from a trained face-recognition
model (ArcFace) instead of a patch-level CLIP image embedding of a damaged
crop.

Pipeline, zero new pip dependencies (insightface has no cp312 wheels and
needs a C++ toolchain — deliberately avoided):

  YuNet landmarks -> similarity-align to the 112x112 ArcFace template
                  -> w600k_mbt.onnx (buffalo_l recognition model)
                  -> L2-normalized 512-d embedding

The module never raises into the pipeline: every failure returns None with a
reason, which the caller treats as "no identity embedding available" and
degrades to the CLIP-adapter path.
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Anchored to this file, not the process CWD (measured 2026-09-27: the
# uvicorn server runs with cwd=backend/, so CWD-relative weight paths break
# inside the live server while standalone scripts from the repo root work).
WeightsDirDefault = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "weights", "faceid")

# buffalo_l's recognition model. Measured 2026-09-27: the v0.7 release zip
# contains w600k_r50.onnx (ResNet50, 512-d output) — the mbt name does not
# exist in the bundle (the fetcher's strict listing check caught it).
ARCFACE_FILE = "w600k_r50.onnx"

# ArcFace input geometry and the canonical 112x112 landmark template from
# insightface.utils.face_align.arcface_dst (5 points: eyes, nose, mouth
# corners). cv2.estimateSimilarityTransform on these gives the alignment
# insightface performs, without the package.
ARCFACE_SIZE = 112
_ARCFACE_DST = np.array(
    [[38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041]], dtype=np.float32)

# YuNet wants reasonably-sized faces; upscale small inputs before detect.
_MIN_DETECT_SIDE = 320
_DETECT_SCORE = 0.6

_EMBED_DIM = 512


def _arcface_weights_path(weights_dir: str | None = None) -> str:
    base = weights_dir or os.environ.get("PIXREVIVE_FACEID_WEIGHTS",
                                         WeightsDirDefault)
    return os.path.join(base, ARCFACE_FILE)


def _make_yunet():
    """YuNet detector from the sface weights dir (shared with the gate)."""
    from ai.validation.face_identity_gate import WeightsDirDefault as _SFaceDir
    yunet_path = os.path.join(_SFaceDir, "yunet_270face.onnx")
    if not os.path.exists(yunet_path):
        return None
    return cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), _DETECT_SCORE)


class FaceIDEmbedder:
    """YuNet + ArcFace identity embedder (cv2.dnn primary, onnxruntime fallback).

    onnxruntime is preferred when importable (the model card execution
    provider is measurably faster and it sidesteps the old cv2.dnn resize
    quirks); cv2.dnn keeps the module functional with zero extra packages.
    """

    def __init__(self, weights_dir: str | None = None) -> None:
        self._weights_dir = weights_dir
        self._session = None       # onnxruntime session
        self._net = None           # cv2.dnn net
        self._backend = ""
        self._yunet = None

    # -- loading ------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        if self._backend:
            return True
        path = _arcface_weights_path(self._weights_dir)
        if not os.path.exists(path):
            return False
        try:
            import onnxruntime as ort  # type: ignore
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            self._session = ort.InferenceSession(path, so,
                                                 providers=["CPUExecutionProvider"])
            self._backend = "onnxruntime"
            return True
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001 - fall through to cv2.dnn
            logger.warning("FaceID: onnxruntime load failed (%s); trying cv2.dnn", exc)
        try:
            self._net = cv2.dnn.readNetFromONNX(path)
            self._backend = "cv2.dnn"
            return True
        except Exception as exc:  # noqa: BLE001 - never raise into the pipeline
            logger.warning("FaceID: cv2.dnn load failed: %s", exc)
            return False

    def available(self) -> bool:
        return self._ensure_loaded()

    @property
    def backend(self) -> str:
        return self._backend

    # -- detection + alignment ----------------------------------------------

    def _detect_largest(self, bgr: np.ndarray):
        if self._yunet is None:
            self._yunet = _make_yunet()
            if self._yunet is None:
                return None
        h, w = bgr.shape[:2]
        scale = 1.0
        side = max(h, w)
        if side < _MIN_DETECT_SIDE:
            scale = _MIN_DETECT_SIDE / side
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
        self._yunet.setInputSize((bgr.shape[1], bgr.shape[0]))
        ok, faces = self._yunet.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        # Largest by box area (most-identity-bearing face governs).
        best = max(faces, key=lambda f: float(f[2]) * float(f[3]))
        if scale != 1.0:
            best = best.copy()
            best[:4] /= scale
        return best  # [x, y, w, h, score, 5x(x,y) landmarks]

    def align(self, bgr: np.ndarray) -> np.ndarray | None:
        """Similarity-align the largest face to the 112x112 ArcFace template."""
        face = self._detect_largest(bgr)
        if face is None:
            return None
        landmarks = np.asarray(face[5:15], dtype=np.float32).reshape(5, 2)
        try:
            m, _ = cv2.estimateAffinePartial2D(
                landmarks, _ARCFACE_DST, method=cv2.LMEDS)
        except cv2.error:
            m = None
        if m is None:
            return None
        return cv2.warpAffine(bgr, m, (ARCFACE_SIZE, ARCFACE_SIZE),
                              flags=cv2.INTER_CUBIC,
                              borderValue=(0, 0, 0))

    # -- embedding ------------------------------------------------------------

    def embed(self, bgr: np.ndarray) -> np.ndarray | None:
        """L2-normalized 512-d ArcFace embedding of the largest face, or None."""
        if not self._ensure_loaded():
            return None
        aligned = self.align(bgr)
        if aligned is None:
            return None
        blob = cv2.dnn.blobFromImage(
            aligned, 1.0 / 127.5, (ARCFACE_SIZE, ARCFACE_SIZE),
            (127.5, 127.5, 127.5), swapRB=True)
        try:
            if self._backend == "onnxruntime":
                inp = self._session.get_inputs()[0].name
                feat = self._session.run(None, {inp: blob})[0].flatten()
            else:
                self._net.setInput(blob)
                feat = self._net.forward().flatten()
        except Exception as exc:  # noqa: BLE001
            logger.warning("FaceID: forward failed: %s", exc)
            return None
        if feat.shape[0] != _EMBED_DIM:
            logger.warning("FaceID: unexpected embed dim %s", feat.shape)
            return None
        feat = feat.astype(np.float32)
        norm = float(np.linalg.norm(feat))
        if norm < 1e-6:
            return None
        return feat / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized embeddings (dot product)."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
