"""output_verifier.py — Closed-loop output verification (post-restoration).

The repair stages trust their own masks: whatever the detectors missed on the
INPUT stays in the OUTPUT. This module closes the loop — it inspects the
RESTORED image with the same class of detectors that drove the repair and:

1. Residue re-detection + micro-cleanup
   Bright-outlier specks on the final image (emulsion-flake survivors) are
   re-detected; above an honest bar they get ONE tiny Telea pass limited to
   the residual pixels — never a full re-inpaint. Face bboxes are excluded
   from cleanup (GFPGAN owns that canvas; smearing it is worse than specks).
2. Color-coherence check ("colour on one side, not the other")
   Damaged margins that survive as borders often receive a colorization wash
   that disagrees with the photo content (measured: cyan strips against a warm
   portrait). Compares LAB a/b chroma of the outer border strip vs the inner
   content; a large disagreement is FLAGGED (recolouring is not attempted —
   an automatic fix would be a second hallucination).
3. Texture-plausibility check ("hair like softy ice-cream")
   Re-synthesized (inpainted) regions whose local gradient energy collapsed
   far below their surround read as waxy/plastic. Faces are excluded (skin
   smoothing is expected). Flagged, not altered.

Everything is recorded in context.image_metadata["output_verification"] and
surfaced by the Restoration Report. The verifier is designed to never raise —
a verification failure must not destroy a finished restoration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Residue (bright-outlier % of frame) above which cleanup fires, and the cap
# beyond which we only flag (a huge residue means the repair failed upstream;
# inpainting 10% of the output here would be a second, uncontrolled repair).
RESIDUE_CLEANUP_BAR_PCT = 0.5
RESIDUE_CLEANUP_CAP_PCT = 2.5
# In-face residue worth telling the user about (faces are excluded from cleanup).
IN_FACE_RESIDUE_FLAG_PCT = 0.3
# LAB a/b disagreement between border strip and content that reads as an
# incoherent color wash (measured ~25 on the cyan-margin case; real photos
# with genuinely two-tone lighting rarely exceed ~12 on stripped borders).
COLOR_EDGE_DELTA_AB = 18.0
# Border-strip width as a fraction of the short side (former scanner margins /
# damage bands live here).
COLOR_EDGE_STRIP_FRACTION = 0.06
# Texture-energy ratio (inpaint region vs its surround ring) below which
# re-synthesis reads as over-smoothed.
TEXTURE_RATIO_BAR = 0.35
# Inpaint coverage below which the texture check is meaningless.
TEXTURE_MIN_INPAINT_PCT = 1.0


def _to_gray_f32(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def detect_residue(rgb: np.ndarray) -> np.ndarray:
    """Binary mask of bright-outlier specks (emulsion-flake survivors).

    Same statistic the audit used to measure the problem: pixels far above
    their 11px local mean, formed into small compact components. Structural
    content (elongated edges, large areas) is excluded — a white shirt collar
    is not residue.
    """
    gray = _to_gray_f32(rgb)
    local_mean = cv2.GaussianBlur(gray, (11, 11), 0)
    candidates = ((gray - local_mean) > 30.0).astype(np.uint8) * 255
    if not np.any(candidates):
        return candidates

    h, w = gray.shape
    max_area = max(16, int(h * w * 0.005))  # single specks, not regions
    mask = np.zeros_like(candidates)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area > max_area:
            continue
        # Compact bright dots: not filament-thin and not huge bounding boxes.
        if max(cw, ch) > 8 * max(1, min(cw, ch)):
            continue
        mask[labels == i] = 255
    return mask


def _face_boxes(context) -> List[np.ndarray]:
    boxes: List[np.ndarray] = []
    for f in (context.image_metadata.get("faces") or []):
        bbox = f.get("bbox") if isinstance(f, dict) else None
        if bbox and len(bbox) >= 4:
            boxes.append(np.array(bbox[:4], dtype=np.int32))
    return boxes


def _face_mask(shape, boxes: List[np.ndarray]) -> np.ndarray:
    m = np.zeros(shape[:2], dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(shape[1] - 1, int(x1)), min(shape[0] - 1, int(y1))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = 255
    return m


def _pct(mask: np.ndarray) -> float:
    return 100.0 * float(np.count_nonzero(mask)) / max(1, mask.size)


def _check_structural_symmetry(original_rgb: np.ndarray, restored_rgb: np.ndarray) -> Dict[str, Any]:
    """Left-right structural continuity (the 'shirt collar on one side only' tell).

    Re-synthesis can silently drop structure on one side of the frame: the
    input showed a collar/shirt continuing on both sides, the restored output
    keeps texture on the left and fades into wash on the right. Measure
    per-side edge-energy (Sobel magnitude pooled over columns) on both images
    and compare the sides:

      * input balanced  + output balanced  -> fine
      * input unbalanced (real damage was one-sided) -> fine, explained
      * input balanced  + output UNBALANCED -> NEW asymmetry the pipeline
        invented — flagged.
    """
    def _side_energy(rgb: np.ndarray) -> tuple:
        gray = _to_gray_f32(rgb)
        h, w = gray.shape
        mid = w // 2
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        # Weighted to the lower-middle band where collars/torso live, so sky
        # and margins do not dominate.
        band = mag[int(h * 0.35): int(h * 0.95), :]
        left = float(np.mean(band[:, : mid - w // 10]))
        right = float(np.mean(band[:, mid + w // 10:]))
        return left, right

    li, ri = _side_energy(original_rgb)
    lo, ro = _side_energy(restored_rgb)

    def _ratio(l: float, r: float) -> float:
        m = max(l, r, 1e-6)
        return min(l, r) / m

    in_balance = _ratio(li, ri)
    out_balance = _ratio(lo, ro)
    # The output may be MORE balanced than the input (repair removes
    # one-sided damage) but must not lose relative balance: a >25% RELATIVE
    # drop flags (measured on the teared-sepia output: 0.84 -> 0.62, a 26%
    # relative drop — the collar-fade case; absolute 0.30 was too lax).
    invented = bool(
        in_balance > 0 and (in_balance - out_balance) / in_balance > 0.25
    )
    return {
        "input_balance": round(in_balance, 3),
        "output_balance": round(out_balance, 3),
        "asymmetry_invented": invented,
    }


def _check_color_coherence(rgb: np.ndarray) -> Dict[str, Any]:
    """Border-strip vs content chroma disagreement (the cyan-margin tell)."""
    h, w = rgb.shape[:2]
    strip = max(4, int(min(h, w) * COLOR_EDGE_STRIP_FRACTION))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    inner = lab[strip:h - strip, strip:w - strip]
    if inner.size == 0:
        return {"delta_ab": 0.0, "incoherent": False}

    # Chroma stats per side. Returns (mean_a_offset, mean_b_offset,
    # chromatic_fraction). A side with almost no chromatic pixels is reported
    # as its measured (near-neutral) vector plus a low fraction — the CALLER
    # decides that "one side coloured, the other not" is itself incoherence
    # (the damaged-border wash tell), not absence of evidence.
    def _chroma_stats(block: np.ndarray) -> tuple:
        # Accepts (N, 3) (boolean-indexed border) or (H, W, 3) (interior
        # slice) — flatten to pixels first.
        flat = block.reshape(-1, block.shape[-1]).astype(np.float32)
        a, b = flat[:, 1] - 128.0, flat[:, 2] - 128.0
        sat = np.hypot(a, b)
        sel = sat > 12.0
        frac = float(np.count_nonzero(sel)) / max(1, flat.shape[0])
        if float(np.count_nonzero(sel)) < 50:
            return 0.0, 0.0, frac
        return float(np.mean(a[sel])), float(np.mean(b[sel])), frac

    # Per-side check (union dilutes a single-side wash — measured on the
    # teared-sepia case: left-margin cyan wash read 10.0 via the union but
    # ~20 per-side). Interior = the frame minus ALL side bands.
    interior_lab = inner
    s_inner = _chroma_stats(interior_lab)
    ca, cb, fi = s_inner

    worst = 0.0
    for side_mask in (
        (slice(None), slice(None, strip)),
        (slice(None), slice(-strip, None)),
        (slice(None, strip), slice(None)),
        (slice(-strip, None), slice(None)),
    ):
        va, vb, fa = _chroma_stats(lab[side_mask])
        if fa < 0.02 or fi < 0.02:
            # One side effectively neutral: neutral-vs-coloured IS the
            # disagreement (wash applied where the photo has none).
            chromatic = (va, vb) if fa >= fi else (ca, cb)
            delta = float(np.hypot(*chromatic))
        else:
            delta = float(np.hypot(va - ca, vb - cb))
        worst = max(worst, delta)
    return {"delta_ab": round(worst, 2), "incoherent": bool(worst > COLOR_EDGE_DELTA_AB)}


def _check_texture_plausibility(
    rgb: np.ndarray, inpaint_mask: Optional[np.ndarray], face_mask: np.ndarray
) -> Dict[str, Any]:
    """Re-synthesized regions must not be textureless against their surround."""
    empty = {"ratio": None, "oversmooth": False}
    if inpaint_mask is None or _pct(inpaint_mask > 0) < TEXTURE_MIN_INPAINT_PCT:
        return empty

    region = (inpaint_mask > 0) & (face_mask == 0)
    if float(np.count_nonzero(region)) < 500:
        return empty

    # Ring around the region (dilate minus region), also face-free.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    dilated = cv2.dilate((region.astype(np.uint8) * 255), kernel) > 0
    ring = dilated & ~region & (face_mask == 0)
    if float(np.count_nonzero(ring)) < 500:
        return empty

    gray = _to_gray_f32(rgb)
    energy = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    # Local pooling so single pixels cannot decide the verdict.
    pooled = cv2.blur(energy, (15, 15))
    e_in = float(np.mean(pooled[region]))
    e_out = float(np.mean(pooled[ring])) + 1e-6
    ratio = e_in / e_out
    return {"ratio": round(ratio, 3), "oversmooth": bool(ratio < TEXTURE_RATIO_BAR)}


def verify_and_clean(context) -> Dict[str, Any]:
    """Run all checks on the final image; micro-clean residue outside faces.

    Returns the verification record; never raises (callers still wrap it).
    """
    rgb = context.working_image
    result: Dict[str, Any] = {
        "ran": True,
        "residue_pct_before": 0.0,
        "residue_pct_after": 0.0,
        "cleaned": False,
        "in_face_residue_pct": 0.0,
    }

    residue = detect_residue(rgb)
    residue_pct = _pct(residue)
    result["residue_pct_before"] = round(residue_pct, 3)
    result["residue_pct_after"] = result["residue_pct_before"]  # honest default

    faces = _face_mask(rgb.shape, _face_boxes(context))
    in_face = residue & (faces > 0)
    result["in_face_residue_pct"] = round(_pct(in_face), 3)

    cleanup_mask = residue & (faces == 0)
    if residue_pct >= RESIDUE_CLEANUP_BAR_PCT and _pct(cleanup_mask) <= RESIDUE_CLEANUP_CAP_PCT:
        dilated = cv2.dilate(cleanup_mask, np.ones((3, 3), np.uint8), iterations=2)
        cleaned = cv2.inpaint(rgb, dilated, 3, cv2.INPAINT_TELEA)
        context.working_image = cleaned
        rgb = cleaned
        result["cleaned"] = True
        after = detect_residue(rgb)
        result["residue_pct_after"] = round(_pct(after), 3)
        context.add_log(
            f"Stage 11 OutputVerify: residue {result['residue_pct_before']:.2f}% "
            f"-> {result['residue_pct_after']:.2f}% (targeted micro-cleanup)."
        )
    elif residue_pct > RESIDUE_CLEANUP_CAP_PCT:
        context.add_log(
            f"Stage 11 OutputVerify: residue {residue_pct:.2f}% exceeds the cleanup "
            f"cap ({RESIDUE_CLEANUP_CAP_PCT:.1f}%) — flagged, not re-repaired."
        )

    color = _check_color_coherence(rgb)
    result["color_edge_delta_ab"] = color["delta_ab"]
    result["color_edge_incoherent"] = color["incoherent"]

    # Structural left-right continuity ("collar on one side only"): compares
    # the input's own balance with the output's — ratios are scale-invariant,
    # so the 1x input vs the 4x output compare directly.
    try:
        symmetry = _check_structural_symmetry(context.original_image, rgb)
        result["input_balance"] = symmetry["input_balance"]
        result["output_balance"] = symmetry["output_balance"]
        result["asymmetry_invented"] = symmetry["asymmetry_invented"]
    except Exception:
        result["asymmetry_invented"] = False

    masks = context.damage_masks or {}
    final_mask = masks.get("final")
    texture = _check_texture_plausibility(rgb, final_mask, faces)
    result["texture_ratio"] = texture.get("ratio")
    result["texture_oversmooth"] = texture.get("oversmooth", False)

    return result


def verification_flags(record: Dict[str, Any]) -> List[str]:
    """Human-readable flags for the Restoration Report."""
    flags: List[str] = []
    if not record.get("ran"):
        return flags
    if record.get("cleaned"):
        flags.append(
            f"post-verification removed residual specks "
            f"({record.get('residue_pct_before', 0):.1f}% → {record.get('residue_pct_after', 0):.1f}% of frame)"
        )
    elif record.get("residue_pct_before", 0) >= RESIDUE_CLEANUP_BAR_PCT:
        flags.append(
            f"residual specks remain after restoration "
            f"({record.get('residue_pct_before', 0):.1f}% of frame) — repair was incomplete"
        )
    if record.get("in_face_residue_pct", 0) >= IN_FACE_RESIDUE_FLAG_PCT:
        flags.append(
            f"specks remain inside face regions ({record.get('in_face_residue_pct'):.1f}%)"
        )
    if record.get("color_edge_incoherent"):
        flags.append(
            f"colorization wash on damaged borders disagrees with the photo content "
            f"(Δab {record.get('color_edge_delta_ab', 0):.0f}) — colors near the edges are implausible"
        )
    if record.get("texture_oversmooth"):
        flags.append(
            "re-synthesized regions are smoother than their surroundings — "
            "texture there is plausible, not recovered"
        )
    if record.get("asymmetry_invented"):
        flags.append(
            "structure that continued across the frame in the original "
            "(e.g. clothing) fades on one side of the output — re-synthesis "
            f"invented the imbalance (balance {record.get('input_balance', 0):.2f} \u2192 "
            f"{record.get('output_balance', 0):.2f})"
        )
    return flags
