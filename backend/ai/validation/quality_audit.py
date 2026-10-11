"""quality_audit.py — report-only quality metrics for the remediation plan
(docs/REMEDIATION_PLAN_AB_G.md, VALIDATOR ADDITIONS).

Pure functions over numpy arrays; no pipeline state, no pixel mutation.
Every function is exception-safe and degrades to None/{} on degenerate
input so the Stage 11 record can never be broken by an audit metric.

Metrics:
  eye_symmetry          per-face weaker/stronger eye Laplacian-variance ratio
                        + iris saturation delta (A1/B3)
  hf_retention_ratio    Laplacian-variance ratio output/input (G1)
  residual_speck_count  top-hat + black-hat isolated components (A2)
  chroma_bleed_score    chroma-gradient energy not co-located with luminance
                        edges (G3/B4)
  seam_step_score       |ΔL| 95th percentile across a mask boundary band (A4)
  edge_retention_ratio  Sobel p85 along jaw/neck band, output/input (G2/B6)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def _gray_f32(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _content_mask(gray_u8: np.ndarray) -> np.ndarray:
    try:
        from ai.analysis.content_region import content_mask
        return content_mask(gray_u8) > 0
    except Exception:
        return np.ones(gray_u8.shape[:2], dtype=bool)


def eye_symmetry(rgb: np.ndarray, faces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Weaker/stronger eye sharpness ratio + iris sat delta (output frame)."""
    out: Dict[str, Any] = {"faces_evaluated": 0}
    if rgb is None or rgb.ndim != 3:
        return out
    h, w = rgb.shape[:2]
    for f in faces or []:
        lms = f.get("landmarks")
        if not lms or len(lms) < 2:
            continue
        bbox = f.get("bbox") or f.get("box") or (0, 0, w, h)
        r = max(5, int(min(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.10))
        wins = []
        for lm in lms[:2]:
            cx, cy = int(round(float(lm[0]))), int(round(float(lm[1])))
            x0, x1 = max(0, cx - r), min(w, cx + r)
            y0, y1 = max(0, cy - r // 2), min(h, cy + r // 2)
            if x1 - x0 >= 3 and y1 - y0 >= 3:
                wins.append((x0, x1, y0, y1))
        if len(wins) != 2:
            continue
        out["faces_evaluated"] += 1
        g = _gray_f32(rgb)
        vs = [float(np.var(cv2.Laplacian(g[y0:y1, x0:x1], cv2.CV_32F)))
              for (x0, x1, y0, y1) in wins]
        if max(vs) > 1e-6:
            out["eye_sharp_ratio"] = round(min(vs) / max(vs), 3)
        sats = []
        for (x0, x1, y0, y1) in wins:
            hsv = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)
            sats.append(float(np.mean(hsv[:, :, 1])))
        out["iris_sat_delta"] = round(abs(sats[0] - sats[1]), 2)
        break
    return out


def hf_retention_ratio(input_rgb: np.ndarray, output_rgb: np.ndarray) -> Optional[float]:
    """Laplacian variance ratio output/input on the input content region."""
    if input_rgb is None or output_rgb is None:
        return None
    h, w = input_rgb.shape[:2]
    out_at_in = cv2.resize(output_rgb, (w, h), interpolation=cv2.INTER_AREA)
    g_in = _gray_f32(input_rgb)
    g_out = _gray_f32(out_at_in)
    cm = _content_mask(g_in.astype(np.uint8))
    if not cm.any():
        return None
    v_in = float(np.var(cv2.Laplacian(g_in, cv2.CV_32F)[cm]))
    v_out = float(np.var(cv2.Laplacian(g_out, cv2.CV_32F)[cm]))
    if v_in <= 1e-6:
        return None
    return round(v_out / v_in, 3)


def residual_speck_count(rgb: np.ndarray, max_area: int = 400,
                         ref_size: Optional[tuple] = None) -> int:
    """Isolated bright/dark outlier components on the content region (A2).

    ``ref_size`` (h, w): when given, the image is first resized to that scale
    so counts are comparable between an SR-scale output and its input — top-hat
    at 4x scale counts every grain pixel as a component (measured 2474 on the
    speckle figure's output canvas vs 333 at input scale).
    """
    if rgb is None or rgb.ndim != 3:
        return 0
    if ref_size is not None and rgb.shape[:2] != tuple(ref_size):
        rgb = cv2.resize(rgb, (int(ref_size[1]), int(ref_size[0])),
                         interpolation=cv2.INTER_AREA)
    g = _gray_f32(rgb)
    cm = _content_mask(g.astype(np.uint8))
    if not cm.any():
        return 0
    g8 = g.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(g8, cv2.MORPH_TOPHAT, k)
    blackhat = cv2.morphologyEx(g8, cv2.MORPH_BLACKHAT, k)
    speck = ((tophat > 28) | (blackhat > 28)) & cm
    num, _, stats, _ = cv2.connectedComponentsWithStats(speck.astype(np.uint8), 8)
    return int(sum(1 for i in range(1, num)
                   if 2 <= stats[i, cv2.CC_STAT_AREA] <= max_area))


def chroma_bleed_score(rgb: np.ndarray) -> Optional[float]:
    """Fraction of content pixels with chroma gradient but no luminance edge."""
    if rgb is None or rgb.ndim != 3:
        return None
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lC, aC, bC = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    cm = _content_mask(lC.astype(np.uint8))
    if not cm.any():
        return None
    gl = np.abs(cv2.Sobel(lC, cv2.CV_32F, 1, 0, 3)) + np.abs(cv2.Sobel(lC, cv2.CV_32F, 0, 1, 3))
    gc = np.sqrt((cv2.Sobel(aC, cv2.CV_32F, 1, 0, 3)) ** 2 + (cv2.Sobel(aC, cv2.CV_32F, 0, 1, 3)) ** 2
                 + (cv2.Sobel(bC, cv2.CV_32F, 1, 0, 3)) ** 2 + (cv2.Sobel(bC, cv2.CV_32F, 0, 1, 3)) ** 2)
    bleed = (gc > 6.0) & (gl < 4.0) & cm
    return round(float(np.mean(bleed)), 5)


def seam_step_score(rgb: np.ndarray, mask: Optional[np.ndarray]) -> Optional[float]:
    """95th-pct |ΔL| across a 5px dilate-minus-erode band of the given mask."""
    if rgb is None or mask is None or not np.any(mask > 0):
        return None
    g = _gray_f32(rgb)
    cm = _content_mask(g.astype(np.uint8))
    m = (mask > 0).astype(np.uint8)
    if m.shape[:2] != g.shape[:2]:
        m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    inner = cv2.erode(m, np.ones((5, 5), np.uint8)) > 0
    outer = cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0
    band = outer & ~inner & cm
    if not band.any():
        return None
    delta = np.abs(g - cv2.blur(g, (5, 5)))
    return round(float(np.percentile(delta[band], 95)), 2)


def edge_retention_ratio(
    input_rgb: np.ndarray,
    output_rgb: np.ndarray,
    faces: List[Dict[str, Any]],
) -> Optional[float]:
    """Jaw/neck band Sobel p85, output vs input (contour-lost detector, G2).

    The band comes from face bboxes that live in the OUTPUT frame (the audit
    runs on the final canvas); both images are compared on the input frame
    with the band placed content-relatively (N8 lesson: never compare frames,
    compare content).
    """
    if input_rgb is None or output_rgb is None:
        return None
    h, w = input_rgb.shape[:2]
    out_at_in = cv2.resize(output_rgb, (w, h), interpolation=cv2.INTER_AREA)
    g_in = _gray_f32(input_rgb)
    g_out = _gray_f32(out_at_in)

    # Content-relative band: face bbox (output frame) -> relative rect within
    # the output's content box -> same relative rect within the input's
    # content box.
    rel = None
    for f in faces or []:
        bbox = f.get("bbox")
        if bbox and len(bbox) >= 4:
            ob = _content_bbox(output_rgb)
            if ob is None:
                return None
            ox0, oy0, ox1, oy1 = ob
            ow_c, oh_c = max(1, ox1 - ox0), max(1, oy1 - oy0)
            x0, y0, x1, y1 = bbox[:4]
            rel = (
                (x0 - ox0) / ow_c, (y0 + 0.70 * (y1 - y0) - oy0) / oh_c,
                (x1 - ox0) / ow_c, (min(y1 + 0.15 * (y1 - y0), oy1) - oy0) / oh_c,
            )
            break
    if rel is None:
        rel = (0.15, 0.70, 0.85, 0.95)
        ib = _content_bbox(input_rgb)
        if ib is not None:
            ix0, iy0, ix1, iy1 = ib
            iw_c, ih_c = max(1, ix1 - ix0), max(1, iy1 - iy0)
            band = np.zeros((h, w), dtype=bool)
            bx0, by0 = ix0 + int(rel[0] * iw_c), iy0 + int(rel[1] * ih_c)
            bx1, by1 = ix0 + int(rel[2] * iw_c), iy0 + int(rel[3] * ih_c)
            band[max(0, by0):min(h, by1), max(0, bx0):min(w, bx1)] = True
        else:
            band = np.zeros((h, w), dtype=bool)
            band[int(rel[1] * h):int(rel[3] * h), int(rel[0] * w):int(rel[2] * w)] = True
    else:
        ib = _content_bbox(input_rgb)
        band = np.zeros((h, w), dtype=bool)
        if ib is not None:
            ix0, iy0, ix1, iy1 = ib
            iw_c, ih_c = max(1, ix1 - ix0), max(1, iy1 - iy0)
            bx0, by0 = ix0 + int(rel[0] * iw_c), iy0 + int(rel[1] * ih_c)
            bx1, by1 = ix0 + int(rel[2] * iw_c), iy0 + int(rel[3] * ih_c)
            band[max(0, by0):min(h, by1), max(0, bx0):min(w, bx1)] = True
        else:
            band[int(rel[1] * h):int(rel[3] * h), int(rel[0] * w):int(rel[2] * w)] = True
    if not band.any():
        return None

    def _p85(g):
        gr = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3) ** 2
                     + cv2.Sobel(g, cv2.CV_32F, 0, 1, 3) ** 2)
        return float(np.percentile(gr[band], 85))

    base = _p85(g_in)
    if base <= 1e-6:
        return None
    return round(_p85(g_out) / base, 3)


def _content_bbox(rgb: np.ndarray) -> Optional[tuple]:
    try:
        g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    except Exception:
        try:
            g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        except Exception:
            return None
    cm = _content_mask(g)
    ys, xs = np.where(cm)
    if ys.size < 32:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def run_quality_audit(
    input_rgb: Optional[np.ndarray],
    output_rgb: np.ndarray,
    faces: List[Dict[str, Any]],
    damage_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Bundle every metric for the Stage 11 record (report-only)."""
    audit: Dict[str, Any] = {"version": "v1-report-only"}
    try:
        audit["eye_symmetry"] = eye_symmetry(output_rgb, faces or [])
        if input_rgb is not None:
            audit["hf_retention_ratio"] = hf_retention_ratio(input_rgb, output_rgb)
            audit["edge_retention_ratio"] = edge_retention_ratio(
                input_rgb, output_rgb, faces or [])
        ref = None
        if input_rgb is not None:
            ref = input_rgb.shape[:2]
        audit["residual_speck_count"] = residual_speck_count(output_rgb, ref_size=ref)
        audit["chroma_bleed_score"] = chroma_bleed_score(output_rgb)
        if damage_mask is not None and np.any(damage_mask > 0):
            audit["seam_step_score"] = seam_step_score(output_rgb, damage_mask)
        # Thresholded advisory flags (consumed by the restoration report).
        flags: List[str] = []
        es = (audit.get("eye_symmetry") or {}).get("eye_sharp_ratio")
        if isinstance(es, (int, float)) and es < 0.5:
            flags.append("eye_asymmetry")
        er = audit.get("edge_retention_ratio")
        if isinstance(er, (int, float)) and er < 0.6:
            flags.append("contour_lost")
        hf = audit.get("hf_retention_ratio")
        if isinstance(hf, (int, float)) and hf < 0.55:
            flags.append("hf_retention_low")
        cb = audit.get("chroma_bleed_score")
        if isinstance(cb, (int, float)) and cb > 0.03:
            flags.append("chroma_bleed")
        audit["flags"] = flags
    except Exception as exc:  # audit must never break verification
        audit["error"] = f"{type(exc).__name__}: {exc}"
    return audit
