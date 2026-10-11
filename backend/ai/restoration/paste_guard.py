"""paste_guard.py — canvas-level verification of GFPGAN's paste footprints.

Phase V0.2 / W1 of docs/REMEDIATION_MASTER_PLAN.md.

Measured defect (E3, speckle figure 2026-09-29): GFPGAN pasted a restored
face rect 255x167 px whose footprint did NOT contain the pipeline's detected
eye landmark — a cross-detector disagreement (facexlib RetinaFace inside
GFPGAN vs the pipeline's YuNet faces). The user sees a square patch of
better-looking face floating beside un-restored anatomy.

This module reconciles the two detectors BEFORE the paste is accepted:

  * build_footprints  — warp each face's crop-space alpha to the canvas via
    the inverse affine (exactly what paste_faces_to_input_image will do).
  * evaluate_pastes   — per-face verdicts:
        ok          footprint overlaps a pipeline face and its eyes
        phantom     footprint sits mostly outside every pipeline face bbox
                    (GFPGAN restored a background blob) -> suppress paste
        misaligned  footprint overlaps a pipeline face but that face's eye
                    landmarks fall outside it -> soften the paste
  * seam_ring_ratio   — advisory |before-after| edge metric: a crisp paste
    rectangle shows a bright outer ring; feather the ring when ratio > 3.

Prime directive: the guard only ever *reduces* what GFPGAN changes — it can
suppress or soften a paste, never invent pixels. All functions are pure
numpy/cv2 so they are testable without GFPGAN or torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Verdicts
OK = "ok"
PHANTOM = "phantom"          # paste region matches no pipeline face -> suppress
MISALIGNED = "misaligned"    # eyes of the matched pipeline face outside footprint

# Tunables (measured defaults; W8 may sweep)
# Phantom = footprint essentially matches no pipeline face. The overlap test
# runs against the bbox dilated by BBOX_DILATE_FRACTION so a legitimately
# larger paste (detector margin differences) is not flagged, while a blob in
# the background still scores ~0.
MIN_INSIDE_FRACTION = 0.15
BBOX_DILATE_FRACTION = 0.25
EYE_CONTAINMENT_DILATE = 15  # px dilation around footprint before eye test
SEAM_RING_PX = 8
SEAM_RING_TRIGGER = 3.0      # edge-band gradient vs interior gradient
FEATHER_BLUR_K = 15
FEATHER_BLUR_K_SMALL = 7     # selective cliff feathering: small kernel
FEATHER_STRENGTH = 0.7


@dataclass
class PasteVerdict:
    """One GFPGAN face's canvas-level paste verdict."""

    index: int                       # helper face index
    verdict: str                     # OK | PHANTOM | MISALIGNED
    matched_pipeline_face: Optional[int] = None
    overlap_fraction: float = 0.0    # footprint area inside the matched bbox
    eyes_contained: Optional[bool] = None
    footprint_area: int = 0
    seam_ratio: Optional[float] = None
    detail: Dict[str, object] = field(default_factory=dict)


def build_footprints(
    alpha_masks: Sequence[np.ndarray],
    affines_canvas_to_crop: Sequence[np.ndarray],
    canvas_hw: Tuple[int, int],
) -> List[np.ndarray]:
    """Warp each crop-space alpha mask to a boolean canvas footprint.

    ``affines_canvas_to_crop`` are facexlib's ``helper.affine_matrices[i]``
    (input-image -> aligned 512 crop). The paste itself uses the inverted
    matrices, so the footprint here inverts them back — the exact region the
    paste will touch on the canvas.
    """
    footprints: List[np.ndarray] = []
    for alpha, mat in zip(alpha_masks, affines_canvas_to_crop):
        if alpha is None or mat is None:
            footprints.append(np.zeros(canvas_hw, dtype=bool))
            continue
        inv = cv2.invertAffineTransform(np.asarray(mat, dtype=np.float64)[:2, :])
        warped = cv2.warpAffine(
            (np.asarray(alpha) * 255).astype(np.uint8),
            inv,
            (canvas_hw[1], canvas_hw[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        footprints.append(warped > 64)
    return footprints


def normalize_faces_to_canvas(
    pipeline_faces: Sequence[Dict],
    faces_frame_size: Optional[Sequence[float]],
    canvas_hw: Tuple[int, int],
) -> List[Dict]:
    """Return pipeline faces rescaled into the current canvas frame.

    Detection records coords in ``faces_frame_size``; intermediate stages can
    leave the working canvas at a different scale (measured 3x on the speckle
    figure: faces_frame_size (1026, 1305) vs a 435x342 canvas). Compare in the
    canvas frame or every overlap test silently degenerates.
    """
    if not pipeline_faces:
        return []
    fh = float(faces_frame_size[0]) if faces_frame_size else float(canvas_hw[0])
    fw = float(faces_frame_size[1]) if faces_frame_size else float(canvas_hw[1])
    if fh <= 0 or fw <= 0:
        return list(pipeline_faces)
    sy = canvas_hw[0] / fh
    sx = canvas_hw[1] / fw
    if abs(sx - 1.0) < 0.01 and abs(sy - 1.0) < 0.01:
        return list(pipeline_faces)
    out: List[Dict] = []
    for f in pipeline_faces:
        g = dict(f)
        bbox = g.get("bbox") or g.get("box")
        if bbox and len(bbox) >= 4:
            g["bbox"] = [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]
        lm = g.get("landmarks")
        if lm:
            g["landmarks"] = [[p[0] * sx, p[1] * sy] for p in lm]
        out.append(g)
    return out


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.asarray(mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if n <= 1:
        return mask_u8.astype(bool)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == big


def evaluate_pastes(
    footprints: Sequence[np.ndarray],
    pipeline_faces: Sequence[Dict],
    canvas_hw: Tuple[int, int],
    eye_indices: Tuple[int, int] = (0, 1),
    min_inside_fraction: float = MIN_INSIDE_FRACTION,
    eye_dilate: int = EYE_CONTAINMENT_DILATE,
) -> List[PasteVerdict]:
    """Verdict each GFPGAN footprint against the pipeline's own face list.

    ``pipeline_faces`` entries need ``bbox`` (canvas coords) and optionally
    ``landmarks`` ([[x, y], ...] — first two entries are the eyes for YuNet).
    Faces whose landmarks are missing get bbox-only checks (eyes_contained
    stays None).
    """
    bboxes = []
    bbox_src_idx = []  # original pipeline_faces index per kept bbox
    for fi, f in enumerate(pipeline_faces):
        bbox = f.get("bbox") or f.get("box")
        if not bbox or len(bbox) < 4:
            continue
        x0, y0, x1, y1 = (int(round(v)) for v in bbox[:4])
        x0 = max(0, min(canvas_hw[1] - 1, x0))
        y0 = max(0, min(canvas_hw[0] - 1, y0))
        x1 = max(0, min(canvas_hw[1], x1))
        y1 = max(0, min(canvas_hw[0], y1))
        if x1 <= x0 or y1 <= y0:
            continue
        bboxes.append((x0, y0, x1, y1))
        bbox_src_idx.append(fi)

    verdicts: List[PasteVerdict] = []
    for i, fp in enumerate(footprints):
        fp_b = fp.astype(bool)
        area = int(fp_b.sum())
        v = PasteVerdict(index=i, verdict=OK, footprint_area=area)
        if area == 0 or not bboxes:
            # Nothing pasted (or nothing to reconcile against): keep as OK —
            # the guard never invents problems on images without pipeline faces.
            if area > 0 and not bboxes:
                v.detail["note"] = "no pipeline faces to reconcile; guard passive"
            verdicts.append(v)
            continue

        # Overlap of the footprint with each pipeline bbox, dilated so a
        # somewhat-larger paste is not punished for detector margin differences.
        overlap_fracs = []
        for (x0, y0, x1, y1) in bboxes:
            mx = int((x1 - x0) * BBOX_DILATE_FRACTION)
            my = int((y1 - y0) * BBOX_DILATE_FRACTION)
            dx0, dy0 = max(0, x0 - mx), max(0, y0 - my)
            dx1, dy1 = min(canvas_hw[1], x1 + mx), min(canvas_hw[0], y1 + my)
            inside = fp_b[dy0:dy1, dx0:dx1]
            frac = float(inside.sum()) / area
            overlap_fracs.append(frac)
        best = int(np.argmax(overlap_fracs))
        v.matched_pipeline_face = bbox_src_idx[best]
        v.overlap_fraction = overlap_fracs[best]

        if overlap_fracs[best] < min_inside_fraction:
            v.verdict = PHANTOM
            verdicts.append(v)
            continue

        # Eye containment for the matched pipeline face.
        lm = pipeline_faces[v.matched_pipeline_face].get("landmarks")
        if lm and len(lm) >= max(eye_indices) + 1:
            fp_d = cv2.dilate(fp_b.astype(np.uint8), np.ones((eye_dilate, eye_dilate), np.uint8)) > 0
            contained = []
            for e in eye_indices:
                ex = int(round(float(lm[e][0])))
                ey = int(round(float(lm[e][1])))
                if 0 <= ey < canvas_hw[0] and 0 <= ex < canvas_hw[1]:
                    contained.append(bool(fp_d[ey, ex]))
            v.eyes_contained = all(contained) if contained else None
            if v.eyes_contained is False:
                v.verdict = MISALIGNED
        verdicts.append(v)
    return verdicts


def seam_band(footprint: np.ndarray, band_px: int = SEAM_RING_PX) -> np.ndarray:
    """Band straddling the footprint edge (dilate − erode)."""
    fp = footprint.astype(np.uint8)
    kernel = np.ones((band_px, band_px), np.uint8)
    outer = cv2.dilate(fp, kernel) > 0
    inner = cv2.erode(fp, kernel) > 0
    return outer & ~inner


def seam_ring_ratio(
    after_gray: np.ndarray,
    footprint: np.ndarray,
    band_px: int = SEAM_RING_PX,
) -> Optional[float]:
    """Boundary-gradient ratio of the pasted image: edge band vs interior.

    A hard box paste leaves a visible STEP at the footprint boundary — the
    after-image's gradient spikes on the edge band while the interior is
    smooth. A feathered composite transitions smoothly (ratio ≈ 1). Returns
    None when the footprint is too small to measure.
    """
    return _grad_band_ratio(after_gray, footprint, band_px)


def _grad_band_ratio(
    after_gray: np.ndarray,
    region: np.ndarray,
    band_px: int,
) -> Optional[float]:
    fp = np.asarray(region).astype(np.uint8)
    if fp.sum() < 64:
        return None
    band = seam_band(fp, band_px)
    interior = cv2.erode(fp, np.ones((2 * band_px, 2 * band_px), np.uint8)) > 0
    if band.sum() < 16 or interior.sum() < 16:
        return None
    gx = cv2.Sobel(after_gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(after_gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    inner_mean = float(grad[interior].mean())
    if inner_mean <= 1e-3:
        inner_mean = 1e-3
    return float(grad[band].mean()) / inner_mean


def changed_region(
    before_gray: np.ndarray,
    after_gray: np.ndarray,
    footprint: np.ndarray,
    thresh: int = 8,
) -> Optional[np.ndarray]:
    """Largest connected |before−after| component inside the footprint.

    The user-visible "square patch" defect is an INTERNAL alpha cliff: the
    composite's feature-zone floor restores e.g. the brow while the adjacent
    eye area stays original, so the changed pixels form a hard-edged island
    whose boundary passes right through visible anatomy. This isolates that
    island so the seam metric and feathering can act on the actual artifact
    instead of the (already feathered) footprint edge.
    """
    delta = cv2.absdiff(
        after_gray.astype(np.uint8), before_gray.astype(np.uint8)
    )
    mask = ((delta > thresh) & (np.asarray(footprint) > 0)).astype(np.uint8)
    if mask.sum() < 32:
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (labels == big)
    if comp.sum() < 32:
        return None
    return comp


def internal_seam_ratio(
    before_gray: np.ndarray,
    after_gray: np.ndarray,
    footprint: np.ndarray,
    band_px: int = SEAM_RING_PX,
) -> Optional[float]:
    """Transition SHARPNESS across the changed island's boundary.

    Ratio of the band's 90th-percentile gradient to its 90th-percentile
    |before−after| magnitude. A hard alpha cliff concentrates the whole
    change into 1–2 px, so Sobel's response hits ≈4× the change amplitude
    (ratio ≈ 4) — the measured "square at the eye" signature (E3). A smooth
    ramp spreads the same change over many px (ratio < 1). Edge-vs-interior
    ratios cannot separate the two; sharpness can. Returns None when nothing
    meaningful changed inside the footprint.
    """
    comp = changed_region(before_gray, after_gray, footprint)
    if comp is None:
        return None
    band = seam_band(comp.astype(np.uint8), band_px)
    if band.sum() < 16:
        return None
    after_f = after_gray.astype(np.float32)
    before_f = before_gray.astype(np.float32)
    gx = cv2.Sobel(after_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(after_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_band = np.sqrt(gx ** 2 + gy ** 2)[band]
    delta_band = np.abs(after_f - before_f)[band]
    g90 = float(np.percentile(grad_band, 90))
    d90 = float(np.percentile(delta_band, 90))
    if d90 <= 1e-3:
        return None
    return g90 / d90


def cliff_pixels(
    before_gray: np.ndarray,
    after_gray: np.ndarray,
    band: np.ndarray,
    delta_min: float = 8.0,
    sharpness_min: float = 2.0,
) -> np.ndarray:
    """The band pixels sitting on an actual hard step (per-pixel sharpness).

    Whole-band feathering over-corrects: the changed island is usually a
    sprawling alpha-transition ring whose smooth cosine sections must stay
    untouched. A 1-px step of amplitude D produces Sobel magnitude ≈ 4·D,
    while a smooth ramp spreads the same change (ratio ≪ 1). Selecting only
    band pixels with grad > sharpness_min·|Δ| confines the feather to the
    visible cliff segments — the measured over-blur fix (eye sheet, 2026-09-29).
    """
    before_f = before_gray.astype(np.float32)
    after_f = after_gray.astype(np.float32)
    gx = cv2.Sobel(after_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(after_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    delta = np.abs(after_f - before_f)
    sel = (
        band
        & (grad > sharpness_min * np.maximum(delta, 1.0))
        & (delta > delta_min)
    )
    if sel.sum() < 8:
        return sel
    # small dilation keeps the feathered line continuous
    return cv2.dilate(sel.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def feather_band(
    img_bgr: np.ndarray,
    band: np.ndarray,
    blur_k: int = FEATHER_BLUR_K,
    strength: float = FEATHER_STRENGTH,
) -> np.ndarray:
    """Blur only the given band mask (in-place safe copy)."""
    band = np.asarray(band)
    if band.sum() < 16:
        return img_bgr
    blurred = cv2.GaussianBlur(img_bgr, (blur_k, blur_k), 0)
    out = img_bgr.copy()
    out[band] = (
        blurred[band].astype(np.float32) * strength
        + img_bgr[band].astype(np.float32) * (1.0 - strength)
    ).astype(np.uint8)
    return out


def feather_seam(
    img_bgr: np.ndarray,
    footprint: np.ndarray,
    band_px: int = SEAM_RING_PX,
    blur_k: int = FEATHER_BLUR_K,
    strength: float = FEATHER_STRENGTH,
) -> np.ndarray:
    """Blur only the band straddling the footprint edge (in-place safe copy)."""
    return feather_band(img_bgr, seam_band(footprint, band_px), blur_k, strength)


# ─── A1/B3 eye harmony (docs/REMEDIATION_PLAN_AB_G.md) ───────────────────────
_EYE_SHARP_RATIO_MAX = 1.8   # weaker/stronger Lap-var ratio that triggers help
_EYE_SHARP_PASSES = 4        # deep smudges need more passes; parity stop per pass
_EYE_UNSHARP_AMT = 0.45
_IRIS_SAT_MAX_SEPIA = 70.0   # sepia/mono sources: keep iris chroma subdued
_IRIS_SAT_MAX_COLOR = 110.0


def harmonize_eyes(
    img_bgr: np.ndarray,
    faces: Sequence[Dict],
    source_mono: bool = False,
) -> Tuple[np.ndarray, List[Dict]]:
    """A1: balance the two eyes' sharpness and iris chroma, per face.

    The measured defect: one eye sharp with a defined iris, the other a blur
    (detector geometry put the crop off-center; the alpha floor helped one
    eye more than the other). Two bounded fixes per face:
      1. Sharpness: Laplacian variance in each eye window; if weaker/stronger
         < 1.8, unsharp-mask the weaker window (<= 2 passes) toward parity.
         Never touches the stronger eye.
      2. Iris chroma: mean HSV saturation per iris; desaturate the more vivid
         iris toward the mean, capped by the source class (sepia/mono sources
         keep irises subdued — a saturated modern iris on a sepia portrait is
         synthesis, not restoration).
    Returns (image, verdicts) — verdicts go to metadata/report, never silent.
    """
    out = img_bgr.copy()  # never mutate the caller's canvas
    verdicts: List[Dict] = []
    h, w = out.shape[:2]
    sat_cap = _IRIS_SAT_MAX_SEPIA if source_mono else _IRIS_SAT_MAX_COLOR
    for f in faces:
        bbox = f.get("bbox") or f.get("box")
        lms = f.get("landmarks")
        if not bbox or len(bbox) < 4 or not lms or len(lms) < 2:
            continue
        rec: Dict[str, object] = {"adjusted": False}
        try:
            r = max(5, int(min(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.10))
            wins = []
            for lm in lms[:2]:
                cx, cy = int(round(float(lm[0]))), int(round(float(lm[1])))
                x0, x1 = max(0, cx - r), min(w, cx + r)
                y0, y1 = max(0, cy - r // 2), min(h, cy + r // 2)
                if x1 - x0 >= 3 and y1 - y0 >= 3:
                    wins.append((x0, x1, y0, y1))
            if len(wins) == 2:
                g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
                vs = [float(np.var(cv2.Laplacian(g[y0:y1, x0:x1], cv2.CV_32F)))
                      for (x0, x1, y0, y1) in wins]
                weak, strong = (0, 1) if vs[0] < vs[1] else (1, 0)
                if vs[strong] > 1e-6 and vs[weak] / vs[strong] < _EYE_SHARP_RATIO_MAX:
                    x0, x1, y0, y1 = wins[weak]
                    roi = out[y0:y1, x0:x1]
                    for _ in range(_EYE_SHARP_PASSES):
                        # Stop at parity: one unsharp pass moves the ratio
                        # strongly (measured overshoot to 2.5x when the loop
                        # ran to the trigger bound instead).
                        if vs[weak] / max(vs[strong], 1e-6) >= 0.75:
                            break
                        blur = cv2.GaussianBlur(roi, (0, 0), sigmaX=1.0)
                        roi = np.clip(roi.astype(np.float32)
                                      + _EYE_UNSHARP_AMT * (roi.astype(np.float32) - blur),
                                      0, 255).astype(np.uint8)
                        out[y0:y1, x0:x1] = roi
                        gg = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
                        vs[weak] = float(np.var(cv2.Laplacian(gg[y0:y1, x0:x1], cv2.CV_32F)))
                    rec["sharp_ratio_after"] = round(vs[weak] / max(vs[strong], 1e-6), 3)
                    rec["adjusted"] = True

                # Iris chroma harmonization (mean saturation toward the mean).
                sats = []
                for (x0, x1, y0, y1) in wins:
                    hsv = cv2.cvtColor(out[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                    sats.append(float(np.mean(hsv[:, :, 1])))
                hi = 0 if sats[0] > sats[1] else 1
                if sats[hi] > sat_cap and sats[hi] - sats[1 - hi] > 15.0:
                    x0, x1, y0, y1 = wins[hi]
                    hsv = cv2.cvtColor(out[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).astype(np.float32)
                    target = max((sats[0] + sats[1]) / 2.0, sat_cap * 0.8)
                    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (target / max(sats[hi], 1e-6)), 0, sat_cap)
                    out[y0:y1, x0:x1] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                    rec["iris_sat_before"] = round(sats[hi], 1)
                    rec["iris_sat_after"] = round(target, 1)
                    rec["adjusted"] = True
        except Exception:
            continue
        verdicts.append(rec)
    return out, verdicts
