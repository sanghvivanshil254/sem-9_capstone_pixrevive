"""restoration_report.py — T-2 Restoration Report (plan 5.1): the honesty layer.

The pipeline repairs some damage, reconstructs (invents plausible content
for) the rest, and occasionally cannot fix something at all. Until now that
distinction lived only in scattered processing-log lines; users saw a single
output image with no account of what happened to it.

This module turns the evidence the stages already record into one
machine-readable dict stored in ``context.image_metadata
["restoration_report"]``:

  recovered     — damage the pipeline removed using pixels that were still
                  there (denoise, speck suppression, seam smoothing, defhade).
  reconstructed — regions where content was *invented* (LaMa inpaint
                  coverage, DDColor colorization of B&W/sepia, GFPGAN face
                  synthesis). Plausible, not original.
  flagged       — conditions a reviewer should know about: inpaint coverage
                  over the frame, inpaint inside face regions, GFPGAN
                  deferred faces, hallucination rollbacks, circuit-breaker
                  reverts, identity risk, quality-gate warnings.

No stage behavior changes: this is read-only over metadata the pipeline
already computes, so it can never regress the canvas. The report is
attached at the very end of ``restore()``, after quality validation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from ai.pipeline.pipeline_context import PipelineContext

# Inpaint coverage (% of frame) above which substantial content was invented
# rather than repaired — the "reconstructed, not recovered" honesty bar.
UNRECOVERABLE_COVERAGE_PCT = 25.0
# Inpaint coverage inside a face bbox above which facial content is largely
# synthetic even with GFPGAN applied afterwards.
FACE_INPAINT_FLAG_PCT = 40.0

# damage_masks keys whose pixels were re-synthesized (inpainting), vs keys
# whose pixels were cleaned in place (denoise-style operations).
_INPAINT_MASK_KEYS = ("final", "enhanced", "lama_mask")
_DENOISE_MASK_KEYS = ("dark_spots", "tearmark")


def _mask_coverage_pct(mask: Optional[np.ndarray]) -> float:
    """Coverage of a mask in percent of the frame; 0.0 for empty/None."""
    if mask is None:
        return 0.0
    arr = np.asarray(mask)
    if arr.size == 0:
        return 0.0
    return float(np.sum(arr > 0) / arr.size * 100.0)


def _face_union_inpaint_pct(
    faces: List[Dict[str, Any]], mask: Optional[np.ndarray]
) -> Optional[float]:
    """Worst-case inpaint coverage inside any single face bbox (percent).

    Union-per-face (not union-over-faces): one heavily damaged face should
    flag even when a second face is pristine.
    """
    if mask is None or not np.any(mask > 0) or not faces:
        return None
    h, w = mask.shape[:2]
    worst: Optional[float] = None
    for face in faces:
        bbox = face.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in bbox[:4]]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        roi = mask[y0:y1, x0:x1]
        pct = _mask_coverage_pct(roi)
        if worst is None or pct > worst:
            worst = pct
    return worst


def _pct(value: Optional[float]) -> Optional[float]:
    return round(value, 2) if value is not None else None


def build_restoration_report(context: PipelineContext) -> Dict[str, Any]:
    """Classify the run's evidence into recovered / reconstructed / flagged.

    Read-only over ``context``; returns a JSON-serializable dict. Never
    raises — a report must not be able to fail a restoration that already
    succeeded.
    """
    try:
        return _build(context)
    except Exception as exc:  # pragma: no cover — defensive by contract
        return {"error": f"report generation failed: {exc}"}


def _build(context: PipelineContext) -> Dict[str, Any]:
    meta = context.image_metadata
    masks = context.damage_masks

    inpaint_pct = max(_mask_coverage_pct(masks.get(k)) for k in _INPAINT_MASK_KEYS)
    denoise_pct = max(_mask_coverage_pct(masks.get(k)) for k in _DENOISE_MASK_KEYS)
    faces = meta.get("faces") or []
    face_inpaint_pct = _face_union_inpaint_pct(faces, masks.get("final"))

    color_type = meta.get("color_type", "color")
    # Evidence-based: only claim "colors added by AI" when colorization
    # actually ran. restore() seeds colorization_applied=False at flag time
    # and Stage 6 flips it to True on success, so absent flag = legacy test
    # context (keep the old color_type inference for them). The rollback
    # flag still vetoes. Measured 2026-09-27: colorize=false on sepia input
    # used to emit the disclaimer because color_type alone was trusted.
    # N5 (re-audit 2026-09-30): a Q-3 rollback means colorization ran and
    # failed — that is exactly the case the disclaimer exists for (the output
    # stayed grayscale, but the model DID invent chroma and it was discarded).
    # Legacy test contexts without colorization_applied keep the old
    # color_type inference.
    rolled_back = bool(meta.get("colorization_rolled_back", False))
    was_colorized = rolled_back or (
        bool(meta.get("colorization_applied", False))
        or ("colorization_applied" not in meta
            and color_type in ("grayscale", "sepia"))
    )
    gfpgan_applied = bool(meta.get("gfpgan_applied", False))
    gfpgan_deferred = bool(meta.get("gfpgan_deferred", False))

    # ── Recovered: damage removed using pixels that were still there ────────
    recovered: List[str] = []
    if meta.get("deskew_applied"):
        recovered.append(
            f"scan skew corrected ({meta.get('deskew_angle_deg', 0):+.2f}°)"
        )
    if meta.get("flat_field_applied"):
        recovered.append("uneven illumination flattened")
    if inpaint_pct > 0.0 and inpaint_pct < UNRECOVERABLE_COVERAGE_PCT:
        recovered.append(
            f"structural damage inpainted ({inpaint_pct:.1f}% of frame)"
        )
    if meta.get("specks_suppressed"):
        recovered.append("dust specks / impulse noise suppressed")
    if denoise_pct > 0.5:
        recovered.append(f"surface blemishes cleaned ({denoise_pct:.1f}% of frame)")
    if meta.get("denoise_applied"):
        recovered.append("grain / noise reduced")

    # ── Reconstructed: plausible content invented by generative models ─────
    reconstructed: List[str] = []
    if inpaint_pct >= UNRECOVERABLE_COVERAGE_PCT:
        reconstructed.append(
            f"large damaged regions re-synthesized ({inpaint_pct:.1f}% of frame)"
        )
    if face_inpaint_pct is not None and face_inpaint_pct >= FACE_INPAINT_FLAG_PCT:
        reconstructed.append(
            f"facial regions largely reconstructed ({face_inpaint_pct:.1f}% of a face area)"
        )
    if was_colorized and not rolled_back:
        reconstructed.append(
            "colors added by AI — plausible, not the original colors"
        )
    if gfpgan_applied:
        reconstructed.append("facial details enhanced by GFPGAN")

    # ── Flagged: everything a reviewer should know about ────────────────────
    flags: List[str] = []
    if meta.get("colorization_skipped_by_flag") and color_type in ("grayscale", "sepia"):
        flags.append(
            "colorization was disabled for this run — "
            "output keeps the original monochrome tones"
        )
    warnings = meta.get("warnings") or []
    warnings = warnings or []
    # Remediation-plan validator flags (report-only): quality-audit advisories
    # computed in Stage 11 (eye asymmetry, contour loss, HF retention, chroma
    # bleed) become plain-language disclosure lines.
    _ov = meta.get("output_verification") or {}
    _qa = _ov.get("quality_audit") or {} if isinstance(_ov, dict) else {}
    _AUDIT_FLAG_LINES = {
        "eye_asymmetry": "eyes differ in sharpness beyond threshold",
        "contour_lost": "jawline/shoulder contours lost in restoration",
        "hf_retention_low": "fine texture retention below threshold",
        "chroma_bleed": "color bleeding across luminance edges detected",
    }
    for _f in (_qa.get("flags") or []):
        if _f in _AUDIT_FLAG_LINES and _AUDIT_FLAG_LINES[_f] not in flags:
            flags.append(_AUDIT_FLAG_LINES[_f])
    if "inpaint_face_structure_erased_reverted" in warnings:
        flags.append(
            "inpainting erased facial structure in a face region — that region "
            "was reverted to its pre-inpaint state"
        )
    if meta.get("colorization_rolled_back"):
        flags.append(
            "colorization produced an implausible wash and was rolled back — "
            "output stays grayscale"
        )
    elif meta.get("colorization_disclaimer") and not was_colorized:
        # Stage 6 seeds the disclaimer flag directly on the rollback path;
        # mirror it here so the flag never outlives its claim.
        flags.append(
            "colorization produced an implausible wash and was rolled back — "
            "output stays grayscale"
        )
    # G3 (docs/REMEDIATION_PLAN_AB_G.md): a grayscale/sepia source has NO
    # recoverable true colors — anything the colorizer produced is synthetic
    # and must be disclosed as such, not as "plausible".
    if was_colorized and not rolled_back and color_type in ("grayscale", "sepia"):
        flags.append(
            "source is monochrome (" + color_type + ") — all colors are "
            "synthetic; the original colors are unrecoverable"
        )
    # RC-4 (debug session 2026-10-01): chroma synthesis skipped on a low-res
    # noisy grayscale source — disclose that the monochrome look is a decision,
    # not an omission.
    if meta.get("colorize_skipped_lowres_gray"):
        flags.append(
            "source is a low-resolution noisy grayscale image — colorization "
            "was skipped because every color would be invented (enable "
            "colorize explicitly to override)"
        )
    if gfpgan_deferred:
        flags.append(
            "faces too small for reliable restoration before upscaling — "
            "restoration applied after 4× super-resolution"
        )
    if "identity_risk" in warnings:
        flags.append("facial landmark stability below threshold — identity risk")

    # Closed-loop output verification (Stage 11): residue, color-coherence,
    # texture-plausibility findings on the FINAL image.
    from ai.validation.output_verifier import verification_flags

    flags.extend(verification_flags(meta.get("output_verification") or {}))

    # Quality-gate warnings recorded by QualityService (very dark/bright,
    # low dynamic range, solid-block fill, accidental downscale).
    for log_line in reversed(context.processing_log[-60:]):
        if log_line.startswith("WARNING:"):
            flags.append(log_line[len("WARNING:"):].strip().rstrip(".").lower())

    # ── Phase H1: re-imagine disclosure (hybrid evidence-preserving diffusion) ─
    # Additive block; existing keys untouched. When the tier ran, the report
    # states plainly how much of the output is original evidence vs AI
    # re-imagined, whether evidence was preserved bytewise, and which resource
    # tier/identity verdict governed the run (docs/HYBRID_DIFFUSION_DESIGN.md).
    reimagine_block: Optional[Dict[str, Any]] = None
    reim = meta.get("reimagine") or {}
    if reim.get("error"):
        # A stage-level crash must be disclosed, not swallowed: the user gets
        # an honest "the beta failed" instead of a silent no-op.
        reimagine_block = {
            "mode": "not_applied",
            "status": "error",
            "reason": reim.get("error"),
        }
    elif reim.get("ran") or reim.get("status"):
        identity = reim.get("identity") or {}
        reimagine_block = {
            "mode": "hybrid-restoration" if reim.get("status") == "done" else "not_applied",
            "status": reim.get("status"),
            "reason": reim.get("reason"),
            "real_pixels_pct": reim.get("real_pixels_pct"),
            "generated_pixels_pct": reim.get("generated_pixels_pct"),
            "evidence_preserved": reim.get("evidence_preserved"),
            "generator": reim.get("generator"),
            "seed": reim.get("seed"),
            # H4: "faceid" (ArcFace-embedded conditioning), "clip" (image-crop
            # adapter) or "none" — which identity signal guided generation.
            "identity_conditioning": reim.get("identity_conditioning"),
            # "Show what's real" (design §7): relative API path to the
            # colorized evidence-zone overlay for hybrid runs.
            "evidence_overlay": reim.get("evidence_overlay"),
            "identity": {
                "action": identity.get("action"),
                "similarity": identity.get("similarity"),
            } if identity else None,
            "resource": reim.get("resource"),
            "coverage": reim.get("coverage"),
        }
        if reim.get("status") == "done":
            reconstructed.append(
                f"{reim.get('generated_pixels_pct', 0):.0f}% of the frame re-imagined "
                "by AI (evidence-preserving diffusion) — plausible, not original"
            )
            if identity.get("action") == "flag":
                flags.append(
                    "re-imagined face drifted from the original "
                    f"(SFace cosine {identity.get('similarity')}) — kept, disclosed"
                )
        elif reim.get("status") == "skipped_degraded":
            flags.append("re-imagine was attempted but rolled back (identity could "
                         "not be preserved) — output is the standard restoration")

    report: Dict[str, Any] = {
        "recovered": recovered,
        "reconstructed": reconstructed,
        "flagged": flags,
        "colorization_disclaimer": bool(was_colorized),
        "reimagine": reimagine_block,
        # Phase V0 (docs/SEMANTIC_VALIDATION_PLAN.md): report-only semantic
        # findings — evidence-conditioned anatomy/garment plausibility.
        # Disclosure-only; findings never alter pixels and never fire in V0.
        "semantic": meta.get("semantic"),
        "stage_timings": dict(context.stage_timings),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return report
