"""W1 paste-guard tests — pure cv2/numpy, no GFPGAN/torch (Phase V0.2).

Covers the cases from docs/REMEDIATION_MASTER_PLAN.md W1:
  1. footprint warp matches the analytic inverse-affine region
  2. aligned paste  -> ok (guard would stay silent)
  3. phantom paste  -> suppressed verdict (footprint matches no pipeline face)
  4. misaligned paste -> eyes_contained False (soften path)
  5. seam ring ratio flags a hard-edged paste, not a feathered one
  6. frame-size normalization rescales stale detection coords to the canvas
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ai.restoration.paste_guard import (
    MISALIGNED,
    OK,
    PHANTOM,
    build_footprints,
    changed_region,
    evaluate_pastes,
    feather_seam,
    internal_seam_ratio,
    normalize_faces_to_canvas,
    seam_ring_ratio,
)


def _affine_canvas_to_crop(center_xy, crop_size=512, scale=2.0):
    """Similarity affine: canvas point center_xy -> crop centre."""
    cx, cy = center_xy
    s = scale
    theta = 0.0
    a = s * np.cos(theta)
    b = s * np.sin(theta)
    # x_crop = a*x + b*y + tx ; centre maps to crop centre
    tx = crop_size / 2.0 - (a * cx + b * cy)
    ty = crop_size / 2.0 - (-b * cx + a * cy)
    return np.array([[a, b, tx], [-b, a, ty]], dtype=np.float64)


def _face(bbox, landmarks=None):
    f = {"bbox": list(bbox), "confidence": 0.9, "landmarks": landmarks}
    return f


# ── 1. footprint geometry ─────────────────────────────────────────────────────

def test_footprint_matches_analytic_warp():
    h, w = 200, 300
    center = (150, 100)
    mat = _affine_canvas_to_crop(center, scale=2.0)
    alpha = np.ones((512, 512), np.float32)
    fp = build_footprints([alpha], [mat], (h, w))[0].astype(np.uint8)

    # analytic: crop maps back to a square of size 512/2 = 256 around center
    expect = np.zeros((h, w), np.uint8)
    x0, y0 = int(150 - 128), int(100 - 128)
    expect[max(0, y0):min(h, y0 + 256), max(0, x0):min(w, x0 + 256)] = 1
    inter = np.logical_and(fp > 0, expect > 0).sum()
    union = np.logical_or(fp > 0, expect > 0).sum()
    assert inter / union > 0.95


def test_footprint_empty_for_none_inputs():
    fp = build_footprints([None], [None], (50, 60))[0]
    assert fp.dtype == bool and not fp.any()


# ── 2–4. verdicts ─────────────────────────────────────────────────────────────

def _mk_pipeline_face(center, size=(120, 140)):
    cx, cy = center
    w, h = size
    bbox = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    # YuNet-style 5 points: two eyes, nose, mouth corners
    lm = [[cx - w * 0.2, cy - h * 0.1],
          [cx + w * 0.2, cy - h * 0.1],
          [cx, cy],
          [cx - w * 0.15, cy + h * 0.2],
          [cx + w * 0.15, cy + h * 0.2]]
    return _face(bbox, lm)


def test_aligned_paste_is_ok():
    h, w = 400, 400
    center = (200, 200)
    mat = _affine_canvas_to_crop(center, scale=2.0)  # crop 512 -> canvas 256 box
    alpha = np.ones((512, 512), np.float32)
    fp = build_footprints([alpha], [mat], (h, w))[0]
    face = _mk_pipeline_face(center)
    v = evaluate_pastes([fp], [face], (h, w))[0]
    assert v.verdict == OK
    assert v.eyes_contained is True
    # footprint (256x256) legitimately exceeds the pipeline bbox (120x140):
    # GFPGAN pastes include forehead/hair margin, so ~0.5 is the aligned norm
    assert v.overlap_fraction > 0.3


def test_phantom_paste_is_suppressed():
    h, w = 500, 500
    # footprint far from the pipeline face (canvas box ~154 px)
    mat = _affine_canvas_to_crop((80, 400), scale=3.0)
    alpha = np.ones((512, 512), np.float32)
    fp = build_footprints([alpha], [mat], (h, w))[0]
    face = _mk_pipeline_face((350, 120))
    v = evaluate_pastes([fp], [face], (h, w))[0]
    assert v.verdict == PHANTOM
    assert v.overlap_fraction < 0.1


def test_misaligned_paste_detected():
    h, w = 400, 400
    face = _mk_pipeline_face((200, 200))
    # paste footprint centered above the face so eyes (y~180) fall outside;
    # canvas box ~179 px tall spans y ~ -19..160
    mat = _affine_canvas_to_crop((200, 70), scale=2.6)
    alpha = np.ones((512, 512), np.float32)
    fp = build_footprints([alpha], [mat], (h, w))[0]
    v = evaluate_pastes([fp], [face], (h, w))[0]
    assert v.verdict == MISALIGNED
    assert v.eyes_contained is False


def test_bbox_only_face_skips_eye_check():
    h, w = 400, 400
    center = (200, 200)
    mat = _affine_canvas_to_crop(center, scale=2.0)
    alpha = np.ones((512, 512), np.float32)
    fp = build_footprints([alpha], [mat], (h, w))[0]
    face = _face((130, 130, 270, 270), None)  # no landmarks
    v = evaluate_pastes([fp], [face], (h, w))[0]
    assert v.verdict == OK
    assert v.eyes_contained is None


# ── 5. seam metric ────────────────────────────────────────────────────────────

def test_seam_ratio_flags_hard_edge_not_feathered():
    h, w = 300, 300
    fp = np.zeros((h, w), bool)
    fp[100:200, 100:200] = True
    # hard paste: constant plateaus inside/outside -> gradient ONLY at the band
    after = np.full((h, w), 100, np.uint8)
    after[100:200, 100:200] = 160
    ratio_hard = seam_ring_ratio(after, fp)
    assert ratio_hard is not None and ratio_hard > 3.0

    # feathered: smooth falloff across the boundary -> no band spike.
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - 150) ** 2 + (yy - 150) ** 2)
    bump = np.clip((70 - dist) / 25.0, 0, 1)
    after_f = (100 + bump * 60).astype(np.uint8)
    ratio_soft = seam_ring_ratio(after_f, fp)
    assert ratio_soft is not None and ratio_soft < ratio_hard


def test_feather_seam_blurs_only_the_ring():
    h, w = 200, 200
    fp = np.zeros((h, w), bool)
    fp[60:140, 60:140] = True
    img = np.full((h, w, 3), 100, np.uint8)
    img[60:140, 60:140] = 200  # hard step
    out = feather_seam(img, fp)
    # interior untouched, ring moved toward a gradient
    assert int(out[100, 100, 0]) == 200
    assert 100 < int(out[58, 100, 0]) < 200
    assert int(out[20, 20, 0]) == 100


# ── 5b. internal alpha-cliff metric (the measured E3 signature) ───────────────

def test_internal_seam_flags_block_change_with_edge():
    h, w = 300, 300
    fp = np.zeros((h, w), bool)
    fp[80:220, 80:220] = True
    before = np.full((h, w), 110, np.uint8)
    after = before.copy()
    # block change ending mid-footprint with a hard internal edge (E3 signature)
    after[100:160, 100:200] = 170
    r = internal_seam_ratio(before, after, fp)
    assert r is not None and r > 3.0


def test_internal_seam_quiet_on_smooth_change():
    h, w = 300, 300
    fp = np.zeros((h, w), bool)
    fp[80:220, 80:220] = True
    before = np.full((h, w), 110, np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    bump = np.clip((60 - np.sqrt((xx - 150) ** 2 + (yy - 150) ** 2)) / 20.0, 0, 1)
    after = (before + bump * 50).astype(np.uint8)
    r = internal_seam_ratio(before, after, fp)
    assert r is None or r < 3.0


def test_changed_region_isolates_largest_island():
    h, w = 200, 200
    fp = np.zeros((h, w), bool)
    fp[40:160, 40:160] = True
    before = np.zeros((h, w), np.uint8)
    after = before.copy()
    after[60:90, 60:120] = 200   # big island
    after[130:140, 130:140] = 200  # small island
    comp = changed_region(before, after, fp)
    assert comp is not None
    ys, xs = np.where(comp)
    assert 55 <= xs.min() and xs.max() <= 125 and comp.sum() > 500


# ── 6. frame normalization ────────────────────────────────────────────────────

def test_normalize_faces_rescales_stale_coords():
    faces = [_face((900, 600, 1800, 1500),
                   [[1080, 900], [1620, 900], [1350, 1050],
                    [1200, 1350], [1500, 1350]])]
    out = normalize_faces_to_canvas(faces, (3000, 3900), (600, 780))
    b = out[0]["bbox"]
    assert abs(b[0] - 180) < 1.5 and abs(b[1] - 120) < 1.5
    assert abs(out[0]["landmarks"][0][0] - 216) < 1.5


def test_normalize_faces_passthrough_when_same_scale():
    faces = [_face((10, 10, 50, 50), None)]
    out = normalize_faces_to_canvas(faces, (100, 100), (100, 100))
    assert out[0]["bbox"] == [10, 10, 50, 50]


def test_normalize_handles_missing_frame_size():
    faces = [_face((10, 10, 50, 50), None)]
    out = normalize_faces_to_canvas(faces, None, (100, 100))
    assert out[0]["bbox"] == [10, 10, 50, 50]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
