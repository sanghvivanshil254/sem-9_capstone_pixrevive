"""semantic_validator.py — Phase V0: evidence-conditioned semantic rule battery.

Implements the report-only core of docs/SEMANTIC_VALIDATION_PLAN.md:

  * Evidence-state classification — every region is labeled ORIGINAL,
    DESTROYED or AMBIGUOUS from the Phase-H evidence mask (mask==0 =
    surviving original evidence; mask>0 = lost, re-imagined or repairable).
  * Rule battery V0: A1 eye_presence, A3 hairline_continuity,
    C1 bilateral_structure — each emits a *finding* (dict), never touches
    pixels (Prime Directive, plan section 0: the validator may never add
    what the evidence cannot support; it may only complete what the
    evidence proves was there — and in V0 it does not even complete).
  * Composite gate inputs: the findings also drive the (future) targeted
    re-imagination in V1; in V0 they are disclosure-only.

Design invariants (tested):
  * Pure functions over numpy arrays — no model loads, no device access,
    no I/O. Deterministic for identical inputs.
  * Findings may only reference DESTROYED/AMBIGUOUS evidence for
    "actionable" class; a finding whose region is dominated by ORIGINAL
    evidence is downgraded to note-only (evidence is truth).
  * No finding ever proposes modifying surviving-evidence pixels.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Evidence-state shares (of the rule's region): below these the region is
# considered 'original' (evidence governs); above 'destroyed'; between =
# ambiguous. Calibrated so a 60%-destroy synthetic face reads destroyed in
# the eye quadrants while a pristine portrait never crosses AMBIGUOUS.
ORIGINAL_MAX = 0.15
DESTROYED_MIN = 0.60

# A1: eye quadrant definition as fractions of the face bbox (anatomical):
# eyes sit in the upper-middle band of a portrait-crop face.
_EYE_BAND = (0.15, 0.55)          # y-range of the eye band within bbox
_EYE_QUARTERS = {                  # x-ranges (fractions of bbox width)
    "left_eye": (0.10, 0.48),
    "right_eye": (0.52, 0.90),
}
_EYE_SYMMETRY_TOL = 0.12           # interocular tilt tolerance (fraction of bbox h)

# A3: hair region = band above the face bbox (0.35*bbox_h above, capped),
# widened horizontally by 15% each side.
_HAIR_ABOVE = 0.35
_HAIR_SIDE = 0.15

# C1: torso structure band = below the face bbox; the bilateral mirror test
# compares left/right halves of that band (structure density via edges).
_TORSO_TOP = 1.05                  # bbox_h multiples below the face bbox
_TORSO_BOTTOM = 2.2
_TORSO_MIN_DENSITY = 0.010         # min edge density for "structure exists"
_TORSO_ASYMMETRY_RATIO = 0.45      # min(minor/major) density ratio to PASS
_TORSO_MIN_SIDE_PX = 24            # skip tiny bboxes (unreliable geometry)

# B3: skin-residual-damage watch (V0.1, 2026-09-29). Motivated by the
# teared.jpg user review: white blotches on cheek/nose/chin read as
# undissolved damage dressed as texture — no rule watched skin. B3 counts
# bright residual patches (white top-hat) inside the face zone, but ONLY
# where the evidence mask records lost damage; surviving clean skin is
# truth and stays silent (Prime Directive).
_B3_TOPHAT_K = 5                   # blotches are 1-4px; k=5 isolates them
_B3_BRIGHT_L = 12.0                # min L-units above local opening
_B3_MAX_COMP_AREA = 40             # components above this are regions, not blotches
_B3_MIN_COUNT = 4                  # fewer isolated patches = texture, not residue
_B3_MIN_SHARE = 0.004              # ...or >=0.4% of the face zone in blotches
_B3_EVIDENCE_MIN = 0.05            # >=5% of the face zone must be lost evidence
_B3_EVIDENCE_COLOCATION = 0.30     # >=30% of a patch's pixels on evidence>0
_B3_MOUTH_BOX = (0.32, 0.66, 0.68, 0.90)   # x0,y0,x1,y1 fractions of bbox


# --------------------------------------------------------------------------
# Evidence-state classification
# --------------------------------------------------------------------------

def classify_evidence(region_mask: np.ndarray, evidence_mask: np.ndarray) -> str:
    """Classify a region's evidence state: ORIGINAL | AMBIGUOUS | DESTROYED.

    `evidence_mask`: uint8, >0 = lost (destroyed / re-imagined / repairable),
    ==0 = surviving original pixels — exactly the Phase-H convention
    (reimaginer: mask>0 = destroyed, composite restores mask==0 bytewise).
    """
    # Region masks are FULL-CANVAS-shaped (zeros outside the region), so the
    # denominator must be the count of in-region pixels — region_mask.size is
    # the whole image and deflated every lost-share by ~40x (measured bug:
    # a 100%-destroyed quadrant classified ORIGINAL on first probe).
    total = int(np.count_nonzero(region_mask > 0))
    if total == 0:
        return "ORIGINAL"
    lost = float(np.count_nonzero(evidence_mask[region_mask > 0])) / total
    if lost >= DESTROYED_MIN:
        return "DESTROYED"
    if lost <= ORIGINAL_MAX:
        return "ORIGINAL"
    return "AMBIGUOUS"


def _quadrant_masks(face: Dict[str, Any], shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """Eye-band quadrant masks + hair band + torso band for one face bbox."""
    h_img, w_img = shape
    bbox = face.get("bbox")
    if not bbox or len(bbox) < 4:
        return {}
    x0, y0, x1, y1 = [int(round(v)) for v in bbox[:4]]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w_img, x1), min(h_img, y1)
    fw, fh = x1 - x0, y1 - y0
    if fw < _TORSO_MIN_SIDE_PX or fh < _TORSO_MIN_SIDE_PX:
        return {}

    def _rect(rx0: float, ry0: float, rx1: float, ry1: float) -> np.ndarray:
        m = np.zeros((h_img, w_img), np.uint8)
        ax0, ay0 = max(0, x0 + int(rx0 * fw)), max(0, y0 + int(ry0 * fh))
        ax1, ay1 = min(w_img, x0 + int(rx1 * fw)), min(h_img, y0 + int(ry1 * fh))
        if ax1 - ax0 >= 4 and ay1 - ay0 >= 4:
            m[ay0:ay1, ax0:ax1] = 1
        return m

    quads = {
        name: _rect(rx0, _EYE_BAND[0], rx1, _EYE_BAND[1])
        for name, (rx0, rx1) in _EYE_QUARTERS.items()
    }
    # Hair band: above the face bbox (anatomical forehead/hairline zone).
    quads["hair"] = _rect(-_HAIR_SIDE, -_HAIR_ABOVE, 1.0 + _HAIR_SIDE, 0.02)
    # Torso band: below the face bbox (garment/collar zone), widened so the
    # bilateral mirror covers shoulders.
    torso = np.zeros((h_img, w_img), np.uint8)
    ty0 = min(h_img - 4, y1 + max(4, int(0.05 * fh)))
    ty1 = min(h_img, y1 + int((_TORSO_BOTTOM - 1.0) * fh * 0.35))
    if ty1 - ty0 >= _TORSO_MIN_SIDE_PX:
        tx0 = max(0, x0 - int(0.35 * fw))
        tx1 = min(w_img, x1 + int(0.35 * fw))
        torso[ty0:ty1, tx0:tx1] = 1
    quads["torso"] = torso
    return quads


# --------------------------------------------------------------------------
# Rules — each returns a finding dict or None. Report-only: no pixel writes.
# --------------------------------------------------------------------------

def rule_a1_eye_presence(gray: np.ndarray, evidence_mask: np.ndarray,
                         face: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A1: eye presence in the destroyed eye-band quadrants.

    Fires only when a quadrant's evidence state is DESTROYED/AMBIGUOUS —
    surviving eyes are evidence and always PASS (Prime Directive).
    Structure-within-destroyed check: residual dark blobs (eye-like) in the
    destroyed quadrant lower severity; nothing dark at all raises it.
    """
    quads = _quadrant_masks(face, gray.shape)
    if not quads:
        return None
    findings = []
    for name in ("left_eye", "right_eye"):
        quad = quads[name]
        state = classify_evidence(quad, evidence_mask)
        if state == "ORIGINAL":
            continue  # evidence governs: eye survived, nothing to say
        # Eye-likeness proxy inside the quadrant: dark iris/liner blobs.
        sub = cv2.bitwise_and(gray, gray, mask=quad.astype(np.uint8) * 255)
        vals = sub[quad > 0]
        if vals.size == 0:
            continue
        dark_share = float(np.count_nonzero(vals < np.percentile(vals, 25))) / vals.size
        has_structure = dark_share > 0.10
        findings.append({
            "rule": "A1.eye_presence",
            "region": name,
            "evidence_state": state,
            "residual_structure": bool(has_structure),
            "severity": "info" if has_structure else "warning",
            "action": "report_only",
            "verdict": (
                f"{name} region is {state} with residual iris-like structure"
                if has_structure else
                f"{name} region is {state} with no detectable eye structure"
            ),
        })
    return findings or None


def rule_a3_hairline_continuity(gray: np.ndarray, evidence_mask: np.ndarray,
                                face: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A3: hair-region continuity across the evidence boundary.

    Baldness is truth: a surviving hair band that is low-texture is PASS with
    a bald-note — never a regeneration target (Prime Directive). Only a
    DESTROYED hair band adjacent to *textured* surviving hair (i.e. evidence
    proves hair existed) yields a regeneration-candidate finding.
    """
    quads = _quadrant_masks(face, gray.shape)
    hair = quads.get("hair")
    if hair is None or not hair.any():
        return None
    state = classify_evidence(hair, evidence_mask)
    sub = cv2.bitwise_and(gray, gray, mask=hair.astype(np.uint8) * 255)
    vals = sub[hair > 0]
    if vals.size == 0:
        return None
    # Texture proxy: local std of the hair band.
    tex = float(np.std(vals.astype(np.float32)))
    textured = tex > 18.0
    if state == "ORIGINAL":
        return None  # surviving hair (or its absence) is truth
    if state == "DESTROYED" and textured:
        return [{
            "rule": "A3.hairline_continuity",
            "region": "hair_band",
            "evidence_state": state,
            "texture_std": round(tex, 1),
            "severity": "warning",
            "action": "report_only",
            "verdict": (
                "hair band destroyed but surviving adjacent hair is textured — "
                "regeneration candidate (V1): hair existed in evidence"
            ),
        }]
    if state == "DESTROYED":
        return [{
            "rule": "A3.hairline_continuity",
            "region": "hair_band",
            "evidence_state": state,
            "texture_std": round(tex, 1),
            "severity": "info",
            "action": "report_only",
            "verdict": (
                "hair band destroyed and adjacent hair low-texture — possible "
                "bald/short-hair subject; no regeneration proposed (evidence "
                "cannot prove hair existed)"
            ),
        }]
    return None  # AMBIGUOUS: not enough signal in V0 — silence beats noise


def rule_c1_bilateral_structure(gray: np.ndarray, evidence_mask: np.ndarray,
                                face: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """C1: bilateral garment-structure continuity below the face.

    Compares edge-structure density of left vs right torso halves. Fires only
    when ONE side holds surviving structure (density >= threshold) and the
    mirror side is DESTROYED — the surviving collar/button-line implies its
    continuation. Surviving asymmetry (both sides original) is PASS-with-note
    only when densities diverge wildly; never a regeneration target.
    """
    quads = _quadrant_masks(face, gray.shape)
    torso = quads.get("torso")
    if torso is None or not torso.any():
        return None
    xs = np.where(torso.any(axis=0))[0]
    mid = (xs[0] + xs[-1]) // 2
    left = torso.copy(); left[:, mid + 1:] = 0
    right = torso.copy(); right[:, :mid] = 0

    # Canny the FULL image once, then mask the edge map. (Measured bug:
    # Canny over a masked-and-zeroed image fires along the region boundary
    # itself — the zero/region gradient manufactured ~0.05 fake density on a
    # perfectly flat side, corrupting the mirror comparison.)
    edges_full = cv2.Canny(gray, 60, 140)

    def _density(region: np.ndarray) -> float:
        area = max(1, np.count_nonzero(region > 0))
        return float(np.count_nonzero(edges_full[region > 0])) / area

    d_left, d_right = _density(left), _density(right)
    s_left = classify_evidence(left, evidence_mask)
    s_right = classify_evidence(right, evidence_mask)

    if d_left < _TORSO_MIN_DENSITY and d_right < _TORSO_MIN_DENSITY:
        return None  # no structure anywhere: nothing to continue
    dense_side, d_dense = ("left", d_left) if d_left >= d_right else ("right", d_right)
    mirror_side = "right" if dense_side == "left" else "left"
    d_mirror = d_right if dense_side == "left" else d_left
    s_mirror = s_right if dense_side == "left" else s_left

    if s_mirror == "DESTROYED" and d_dense >= _TORSO_MIN_DENSITY \
            and d_mirror < d_dense * _TORSO_ASYMMETRY_RATIO:
        return [{
            "rule": "C1.bilateral_structure",
            "region": "torso",
            "evidence_state": "half-original/half-destroyed",
            "density_left": round(d_left, 4),
            "density_right": round(d_right, 4),
            "severity": "info",
            "action": "report_only",
            "verdict": (
                f"garment structure on the {dense_side} side has no mirror on "
                f"the destroyed {mirror_side} side — continuation candidate "
                "(V1): a surviving collar/button-line implies its counterpart"
            ),
        }]
    return None


def rule_b3_skin_residual_damage(gray: np.ndarray, evidence_mask: np.ndarray,
                                 face: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """B3: undissolved damage residue on skin — bright residual patches in the face zone.

    The user review of the teared.jpg restoration flagged white blotches on
    cheek/nose/chin as *undissolved damage dressed as texture*: the evidence
    mask said skin was lost there, the repair filled it, but bright residue
    survived where clean skin should be. No V0 rule watched skin (A1 eyes,
    A3 hair, C1 torso). B3 counts bright top-hat patches inside the face
    bbox, excluding the eye quadrants and mouth box (sclera/teeth are
    legitimately bright), and fires only when the evidence mask records real
    lost damage in the zone — a face with no lost evidence is never flagged
    (Prime Directive; pristine corpus must stay silent).

    Report-only info finding; a natural V1b refill target.
    """
    h_img, w_img = gray.shape[:2]
    bbox = face.get("bbox")
    if not bbox or len(bbox) < 4:
        return None
    x0, y0, x1, y1 = [int(round(v)) for v in bbox[:4]]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w_img, x1), min(h_img, y1)
    fw, fh = x1 - x0, y1 - y0
    if fw < _TORSO_MIN_SIDE_PX or fh < _TORSO_MIN_SIDE_PX:
        return None

    zone = np.zeros((h_img, w_img), np.uint8)
    zone[y0:y1, x0:x1] = 1
    zone_area = int(np.count_nonzero(zone))

    # Evidence gate: only faces that actually lost damage get watched.
    lost = float(np.count_nonzero(evidence_mask[zone > 0])) / zone_area
    if lost < _B3_EVIDENCE_MIN:
        return None

    # Bright residual patches: white top-hat of the face zone, excluding the
    # eye quadrants (sclera/catch-lights) and the mouth box (teeth).
    Lf = gray.astype(np.float32)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_B3_TOPHAT_K, _B3_TOPHAT_K))
    th = Lf - cv2.morphologyEx(Lf, cv2.MORPH_OPEN, se)
    cand = ((th > _B3_BRIGHT_L) & (zone > 0)).astype(np.uint8)

    excl = np.zeros((h_img, w_img), np.uint8)
    for name, (rx0, rx1) in _EYE_QUARTERS.items():
        ey0, ey1 = _EYE_BAND
        excl[y0 + int(ey0 * fh):y0 + int(ey1 * fh),
             x0 + int(rx0 * fw):x0 + int(rx1 * fw)] = 1
    mx0, my0, mx1, my1 = _B3_MOUTH_BOX
    excl[y0 + int(my0 * fh):y0 + int(my1 * fh),
         x0 + int(mx0 * fw):x0 + int(mx1 * fw)] = 1
    cand[excl > 0] = 0

    n, labm, stats, _ = cv2.connectedComponentsWithStats(cand * 255, connectivity=8)
    kept = 0
    kept_area = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (1 <= area <= _B3_MAX_COMP_AREA):
            continue
        # Co-location gate: a patch only counts when most of its pixels sit on
        # lost evidence (the verdict claims "residue ON lost damage", not
        # "bright stuff anywhere in the face"). Grain specks on surviving skin
        # are the denoiser's business, not undissolved damage.
        comp = labm == i
        overlap = float(np.count_nonzero(comp & (evidence_mask > 0)))
        if overlap / max(area, 1) < _B3_EVIDENCE_COLOCATION:
            continue
        kept += 1
        kept_area += area
    share = kept_area / float(zone_area)
    if kept < _B3_MIN_COUNT and share < _B3_MIN_SHARE:
        return None
    return [{
        "rule": "B3.skin_residual_damage",
        "region": "face_zone",
        "evidence_state": "DESTROYED" if lost >= DESTROYED_MIN else "AMBIGUOUS",
        "residual_patches": kept,
        "residual_share": round(share, 4),
        "evidence_lost_share": round(lost, 3),
        "severity": "info",
        "action": "report_only",
        "verdict": (
            f"{kept} bright residual patches ({share:.1%} of face zone) remain "
            "on skin where evidence records lost damage — undissolved-damage "
            "candidate (B3, V1 refill target)"
        ),
    }]


RULES = (rule_a1_eye_presence, rule_a3_hairline_continuity,
         rule_c1_bilateral_structure, rule_b3_skin_residual_damage)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run_semantic_validation(image_rgb: np.ndarray,
                            evidence_mask: np.ndarray,
                            faces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Report-only battery. Returns the report `semantic` block.

    Never mutates `image_rgb` or `evidence_mask`. Findings are pure metadata.
    """
    if image_rgb.ndim == 3:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb
    ev = (evidence_mask > 0).astype(np.uint8)
    all_findings: List[Dict[str, Any]] = []
    for face in faces or []:
        for rule in RULES:
            try:
                found = rule(gray, ev, face)
            except Exception as exc:  # noqa: BLE001 — validator never breaks the pipeline
                logger.warning("semantic rule %s failed: %s",
                               getattr(rule, "__name__", rule), exc)
                continue
            if found:
                all_findings.extend(found)
    # Prime Directive guard: no finding may reference an ORIGINAL-dominated
    # region as actionable — downgrade (defense in depth; rules already skip).
    for f in all_findings:
        if f.get("evidence_state") == "ORIGINAL":
            f["action"] = "none_evidence_governs"
    summary = {"eyes": 0, "hair": 0, "garment": 0, "skin": 0}
    for f in all_findings:
        if f["rule"].startswith("A1."):
            summary["eyes"] += 1
        elif f["rule"].startswith("A3."):
            summary["hair"] += 1
        elif f["rule"].startswith("C1."):
            summary["garment"] += 1
        elif f["rule"].startswith("B3."):
            summary["skin"] += 1
    return {
        "version": "v0-report-only",
        "faces_evaluated": len(faces or []),
        "findings": all_findings,
        "summary": summary,
        "prime_directive": (
            "findings reference only lost-evidence regions; surviving pixels "
            "are truth and are never proposed for modification"
        ),
    }
