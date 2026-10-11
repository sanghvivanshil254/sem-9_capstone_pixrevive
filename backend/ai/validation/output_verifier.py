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

import os
from typing import Any, Dict, List, Optional, Tuple

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


def dissolve_face_drips(rgb: np.ndarray, faces_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Dissolve compact bright residue INSIDE face zones (RC-2, 2026-10-01).

    The teared portrait's semantic audit found 163 bright residual patches on
    the face — whitish "drips" GFPGAN's damage-aware reconstruction lifted but
    never fully removed. The shipped cleanup excludes faces entirely because
    GFPGAN owns that canvas; but compact BRIGHT OUTLIER specks (far above an
    11px local mean, small, non-filament) are unambiguously residue, even on a
    reconstructed face, and Telea at radius 2 on their exact pixels cannot
    smear anatomy the way broad edits did.

    Safety: eyes and mouths are additionally zeroed from the mask by the caller
    (landmark circles), components are capped at 0.15% of the face area, and
    the whole pass is capped at 1.0% of face area per image.
    Returns (cleaned_rgb, pixels_dissolved).
    """
    residue = detect_residue(rgb)
    cand = residue & (faces_mask > 0)
    if not np.any(cand):
        return rgb, 0
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8) * 255, connectivity=8)
    face_area = max(1, int(np.count_nonzero(faces_mask)))
    per_comp_cap = max(24, int(face_area * 0.0015))
    total_cap = int(face_area * 0.01)
    keep = np.zeros(rgb.shape[:2], dtype=np.uint8)
    used = 0
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > per_comp_cap or used + area > total_cap:
            continue
        keep[labels == i] = 255
        used += area
    if used < 8:
        return rgb, 0
    dilated = cv2.dilate(keep, np.ones((3, 3), np.uint8), iterations=1)
    cleaned = cv2.inpaint(rgb, dilated, 2, cv2.INPAINT_TELEA)
    return cleaned, used


# Bright-fill survivor dissolve (2026-10-02 white-blob debug): flat bright
# fields the inpaint left INSIDE the repair zone. Scale-relative geometry
# (1341px reference: k_in=31, k_ring=93).
BRIGHT_FILL_COVERAGE_CAP_PCT = 3.0   # above this, flag only (repair failed upstream)


def dissolve_bright_fill_fields(
    rgb: np.ndarray, faces_mask: np.ndarray, damage_mask: Optional[np.ndarray]
) -> Tuple[np.ndarray, int, int]:
    """Dissolve flat bright infill surviving INSIDE the repaired zone (RC-B).

    The teared portrait's head blob had two halves (probe
    .freebuff/probe_white_blob2.py): 49% dust the mask never covered, and 51%
    INSIDE the mask — LaMa fill that inherited brightness from the dust
    context around it (masked-subset bright px mean 204, p99 230). The first
    half is a detection problem (fixed at source in Stage 2,
    EnhancedDamageDetector.detect_dust_colonies); this function closes the
    loop on the second: if the REPAIRED region itself still reads as a flat
    bright field that pops out against its surround, the repair failed there
    and a targeted Telea pass removes it.

    Detection (validated on the real output, .freebuff/probe_fix2_v3.py):
    candidate components form where blur(gray,k_in) > 185 inside a dilated
    damage mask (so real bright content outside the repair zone — paper
    margins, collars, highlights — can never be touched) and outside faces.
    A component is dissolved only when it is compact, mid-size, mean >= 180,
    flat inside (pooled Laplacian <= 0.85x its annulus) AND pops out against
    its true annulus by >= 25 gray levels (the real blob's fill core measured
    mean 216 vs annulus 138, popout 78, tex 0.54; every other structure in
    the frame failed at least one gate). Components are budgeted
    strongest-first under a frame-coverage cap.

    Returns (cleaned_rgb, pixels_dissolved, components_dissolved).
    """
    import os

    if os.environ.get("PIXREVIVE_BRIGHT_FILL", "1").strip().lower() in (
        "0", "false", "no", "off"
    ):
        return rgb, 0, 0
    if damage_mask is None or not np.any(damage_mask > 0):
        return rgb, 0, 0  # no repair zone to verify

    h, w = rgb.shape[:2]
    side = min(h, w)
    k_in = max(15, int(side * 0.023)) | 1
    k_near = max(9, int(side * 0.007)) | 1
    k_out = max(31, int(side * 0.047)) | 1
    amin = max(120, int(h * w * 0.0004))
    amax = int(h * w * 0.02)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    zone = cv2.dilate(
        (damage_mask > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_in, k_in)),
        iterations=1,
    ) * 255
    zone[faces_mask > 0] = 0

    bright = cv2.blur(gray, (k_in, k_in))
    cand = ((bright > 185.0) & (zone > 0)).astype(np.uint8)
    cand = cv2.morphologyEx(
        cand, cv2.MORPH_OPEN, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)))
    if not np.any(cand):
        return rgb, 0, 0

    energy = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    pooled = cv2.blur(energy, (15, 15))
    se_near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_near, k_near))
    se_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_out, k_out))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    comps = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < amin or area > amax:
            continue
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        if max(cw, ch) / max(1, min(cw, ch)) >= 3.0:
            continue
        m = labels == i
        # True annulus: outside a near ring, inside the far ring — excludes
        # the component itself so a large patch's own brightness cannot
        # cancel its popout (the v2 blur-of-blur failure).
        near = cv2.dilate(m.astype(np.uint8), se_near, iterations=1)
        out = cv2.dilate(m.astype(np.uint8), se_out, iterations=1)
        ann = (out > 0) & (near == 0)
        if int(ann.sum()) < 50:
            continue
        mean_in = float(gray[m].mean())
        if mean_in < 180.0:
            continue  # bright is part of the signature; gray fills stay
        popout = mean_in - float(gray[ann].mean())
        if popout < 25.0:
            continue
        tex = float(pooled[m].mean()) / (float(pooled[ann].mean()) + 1e-6)
        if tex > 0.85:
            continue  # interior as textured as its surround: real content
        comps.append((popout, m, area))
    if not comps:
        return rgb, 0, 0

    comps.sort(reverse=True, key=lambda t: t[0])
    total_cap = int(h * w * BRIGHT_FILL_COVERAGE_CAP_PCT / 100.0)
    keep = np.zeros((h, w), dtype=np.uint8)
    used = 0
    n_comp = 0
    for _pop, m, area in comps:
        if used + area > total_cap:
            continue
        keep[m] = 255
        used += area
        n_comp += 1
    if used < amin:
        return rgb, 0, 0

    # Dissolve the FULL bright extent connected to each accepted core: the
    # blur that formed the candidates softened the patch edge, so a rim-only
    # mask makes Telea source from the artifact's own penumbra and the fill
    # saturates bright (measured: rim-only capped the patch at ~183 over a
    # 138 surround). k_ext bounds the reach; connectivity to the core keeps
    # it on-patch. Where the surround is clean (fill sat on dark hair), the
    # drop is near-total; where the surround is itself surviving dust, this
    # pass only dims the artifact — the source fix (Stage 2 dust colonies)
    # is what removes the class.
    k_ext = max(31, int(side * 0.047)) | 1
    grown = cv2.dilate(keep, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (k_ext, k_ext)), iterations=1)
    bright_ext = ((bright > 165.0) & (zone > 0) & (grown > 0)).astype(np.uint8)
    num_e, lbl_e, _, _ = cv2.connectedComponentsWithStats(
        bright_ext, connectivity=8)
    seeds = np.unique(lbl_e[(keep > 0) & (lbl_e > 0)])
    ext = (np.isin(lbl_e, seeds).astype(np.uint8) * 255
           if seeds.size else keep)
    dilated = cv2.dilate(ext, np.ones((9, 9), np.uint8), iterations=1)
    cleaned = cv2.inpaint(rgb, dilated, max(5, side // 200), cv2.INPAINT_TELEA)
    return cleaned, int(np.count_nonzero(ext)), n_comp


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

    # ── In-face drip dissolve (RC-2, PIXREVIVE_FACE_DRIP=0 reverts) ───────
    if os.environ.get("PIXREVIVE_FACE_DRIP", "1").strip().lower() not in ("0", "false", "no", "off"):
        try:
            if np.any(in_face):
                # Zero out eyes/mouth from the dissolve mask (landmarks when
                # available, anthropometric fallback otherwise) — enamel,
                # sclera and specular glints read as "bright residue" but are
                # anatomy.
                safe = faces.copy()
                for f in (context.image_metadata.get("faces") or []):
                    bbox = f.get("bbox") if isinstance(f, dict) else None
                    if not bbox or len(bbox) < 4:
                        continue
                    x0, y0, x1, y1 = (int(v) for v in bbox[:4])
                    fw, fh = max(1, x1 - x0), max(1, y1 - y0)
                    lms = f.get("landmarks")
                    if lms and len(lms) >= 5:
                        for idx_eye in (0, 1):
                            ex, ey = int(round(lms[idx_eye][0])), int(round(lms[idx_eye][1]))
                            r = max(8, int(min(fw, fh) * 0.09))
                            cv2.circle(safe, (ex, ey), r, 0, -1)
                        ml = np.array(lms[3], dtype=np.float32)
                        mr = np.array(lms[4], dtype=np.float32)
                        mc = (ml + mr) / 2.0
                        mw = float(np.linalg.norm(mr - ml))
                        ang = float(np.degrees(np.arctan2(mr[1] - ml[1], mr[0] - ml[0])))
                        cv2.ellipse(safe, (int(round(mc[0])), int(round(mc[1]))),
                                    (int(max(10, mw * 0.60)), int(max(6, mw * 0.30))),
                                    ang, 0, 360, 0, -1)
                    else:
                        for fx in (0.32, 0.68):
                            ex, ey = int(x0 + fx * fw), int(y0 + 0.38 * fh)
                            r = max(8, int(min(fw, fh) * 0.09))
                            cv2.circle(safe, (ex, ey), r, 0, -1)
                        cv2.ellipse(safe, (int(x0 + 0.50 * fw), int(y0 + 0.76 * fh)),
                                    (int(max(10, fw * 0.24)), int(max(6, fh * 0.11))),
                                    0, 0, 360, 0, -1)
                cleaned, n_px = dissolve_face_drips(rgb, safe)
                if n_px > 0:
                    context.working_image = cleaned
                    rgb = cleaned
                    result["face_drips_dissolved_px"] = n_px
                    result["cleaned"] = True
                    after = detect_residue(rgb)
                    result["residue_pct_after"] = round(_pct(after), 3)
                    context.add_log(
                        f"Stage 11 OutputVerify: dissolved {n_px} in-face residue px "
                        f"(drip cleanup, eyes/mouth protected)."
                    )
        except Exception as exc:  # never let cleanup destroy a finished restore
            context.add_log(f"Stage 11 OutputVerify: face drip cleanup skipped: {exc}")

    # ── Bright-fill survivor dissolve (RC-B, PIXREVIVE_BRIGHT_FILL=0 reverts) ─
    try:
        cleaned, n_px, n_comp = dissolve_bright_fill_fields(
            rgb, faces, (context.damage_masks or {}).get("final"))
        if n_px > 0:
            context.working_image = cleaned
            rgb = cleaned
            result["bright_fill_dissolved_px"] = n_px
            result["bright_fill_dissolved_components"] = n_comp
            result["cleaned"] = True
            after = detect_residue(rgb)
            result["residue_pct_after"] = round(_pct(after), 3)
            context.add_log(
                f"Stage 11 OutputVerify: dissolved {n_px} px of flat bright "
                f"infill surviving inside the repair zone ({n_comp} field(s))."
            )
    except Exception as exc:  # never let cleanup destroy a finished restore
        context.add_log(f"Stage 11 OutputVerify: bright-fill cleanup skipped: {exc}")

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

    # Remediation-plan validator additions (docs/REMEDIATION_PLAN_AB_G.md):
    # report-only quality audit — eye symmetry, HF retention, residual specks,
    # chroma bleed, seam steps, contour retention. Never alters pixels.
    try:
        from ai.validation.quality_audit import run_quality_audit
        result["quality_audit"] = run_quality_audit(
            context.original_image if getattr(context, "original_image", None) is not None else None,
            rgb,
            context.image_metadata.get("faces") or [],
            final_mask,
        )
    except Exception as _qa_exc:
        result["quality_audit"] = {"ran": False, "error": str(_qa_exc)}

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
    if record.get("bright_fill_dissolved_px"):
        flags.append(
            f"post-verification dissolved {record.get('bright_fill_dissolved_px')} px of "
            f"flat bright infill the repair left inside the damage zone"
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
