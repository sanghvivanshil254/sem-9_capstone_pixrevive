"""Tests for faceid_embedder.py — ArcFace identity embeddings (Phase H4).

The real-weights tests are skipped when backend/ai/weights/faceid/w600k_mbt.onnx
is absent (fetched by scripts/fetch_faceid_weights.py); geometry/contract
tests run everywhere.
"""

import os

import cv2
import numpy as np
import pytest

from ai.validation import faceid_embedder as fe
from ai.validation.faceid_embedder import FaceIDEmbedder, cosine

ARCFACE = os.path.join(fe.WeightsDirDefault, fe.ARCFACE_FILE)
HAS_ARCFACE = os.path.exists(ARCFACE)

# SFace weights dir carries the shared YuNet detector.
HAS_YUNET = os.path.exists(os.path.join(
    os.path.dirname(os.path.dirname(fe.WeightsDirDefault)),
    "weights", "sface", "yunet_270face.onnx"))


# Real photo for the real-model tests: YuNet is trained on real faces and
# rightly refuses synthetic cartoons (measured 2026-09-27 — detection None on
# a drawn face). case9 is the standing damaged fixture; it is skipped over
# when absent.
CASE9 = os.path.join("tests", "restoration_dataset", "damaged",
                     "case9_teared_sepia.jpg")
HAS_CASE9 = os.path.exists(CASE9)


def _face_bgr() -> np.ndarray:
    """Synthetic deterministic face-ish image (centered oval + features)."""
    img = np.full((256, 256, 3), 110, np.uint8)
    cv2.ellipse(img, (128, 120), (62, 84), 0, 0, 360, (185, 160, 140), -1)
    cv2.circle(img, (104, 100), 8, (40, 40, 60), -1)
    cv2.circle(img, (152, 100), 8, (40, 40, 60), -1)
    cv2.ellipse(img, (128, 150), (24, 12), 0, 0, 360, (90, 60, 60), -1)
    return img


# ------------------------------------------------------------- templates

def test_arcface_template_shape():
    assert fe._ARCFACE_DST.shape == (5, 2)
    # Canonical template landmarks sit inside the 112x112 canvas.
    assert (fe._ARCFACE_DST >= 0).all() and (fe._ARCFACE_DST <= 112).all()


def test_weights_path_is_file_anchored():
    # Must resolve next to the module (backend/ai/weights/...), not the CWD —
    # the live server runs with cwd=backend/ (measured 2026-09-27).
    assert os.path.isabs(fe.WeightsDirDefault)
    assert "weights" in fe.WeightsDirDefault.replace("\\", "/").split("backend/")[-1]
    assert fe._arcface_weights_path().endswith(fe.ARCFACE_FILE)


def test_embedder_unavailable_without_weights(tmp_path):
    emb = FaceIDEmbedder(weights_dir=str(tmp_path))
    assert emb.available() is False
    assert emb.embed(_face_bgr()) is None
    assert emb.backend == ""


def test_align_returns_none_without_yunet(tmp_path, monkeypatch):
    emb = FaceIDEmbedder(weights_dir=str(tmp_path))
    monkeypatch.setattr(fe, "_make_yunet", lambda: None)
    assert emb.align(_face_bgr()) is None


def test_embed_forward_failure_returns_none(monkeypatch):
    if not HAS_ARCFACE:
        pytest.skip("faceid weights not fetched")
    emb = FaceIDEmbedder()
    assert emb.available()
    monkeypatch.setattr(emb, "align", lambda _b: None)
    assert emb.embed(_face_bgr()) is None


# ------------------------------------------------------------ real model

@pytest.mark.skipif(not (HAS_ARCFACE and HAS_YUNET and HAS_CASE9),
                    reason="faceid/yunet weights or case9 fixture missing")
def test_align_geometry():
    emb = FaceIDEmbedder()
    aligned = emb.align(cv2.imread(CASE9))
    assert aligned is not None
    assert aligned.shape == (112, 112, 3)


@pytest.mark.skipif(not (HAS_ARCFACE and HAS_YUNET and HAS_CASE9),
                    reason="faceid/yunet weights or case9 fixture missing")
def test_embed_contract_and_stability():
    emb = FaceIDEmbedder()
    v1 = emb.embed(cv2.imread(CASE9))
    assert v1 is not None
    assert v1.shape == (512,)
    assert v1.dtype == np.float32
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-4
    # Deterministic for identical input.
    v2 = emb.embed(cv2.imread(CASE9))
    assert cosine(v1, v2) > 0.999


@pytest.mark.skipif(not (HAS_ARCFACE and HAS_YUNET and HAS_CASE9),
                    reason="faceid/yunet weights or case9 fixture missing")
def test_same_face_survives_geometric_perturbation():
    """Identity embedding must be near-invariant to small crops/scales —
    that robustness is the entire point over CLIP-crop conditioning."""
    emb = FaceIDEmbedder()
    img = cv2.imread(CASE9)
    base = emb.embed(img)
    assert base is not None
    # Mild zoom + shift (as the reference-crop path would produce).
    M = np.float32([[1.08, 0, -8], [0, 1.08, -4]])
    other = emb.embed(cv2.warpAffine(img, M, (img.shape[1], img.shape[0])))
    assert other is not None
    assert cosine(base, other) > 0.6


@pytest.mark.skipif(not (HAS_ARCFACE and HAS_YUNET),
                    reason="faceid/yunet weights not fetched")
def test_synthetic_face_is_honestly_undetectable():
    """A drawn face must NOT yield an embedding — silent fabrication would
    poison conditioning. The None path is the contract."""
    emb = FaceIDEmbedder()
    assert emb.embed(_face_bgr()) is None
