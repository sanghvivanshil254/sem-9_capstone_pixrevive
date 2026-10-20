"""
pipeline_router.py — Degradation-aware pipeline routing.

Analyzes image metadata from QualityAnalyzer and determines which pipeline
stages should run. Returns a route dict consumed by restoration_pipeline.py.

Route selection is based on:
  - damage_percentage and has_white_flakes  →  run_damage_repair
  - blur_score and face presence            →  run_face_restore
  - color_type and enable_colorization      →  run_colorization
  - noise_score and noise_profile           →  run_denoising
  - resolution and enable_restoration       →  run_sr
"""

import os
import logging
from typing import Dict, Any

from ai.pipeline.pipeline_context import PipelineContext

logger = logging.getLogger(__name__)

# ── Active route presets ──────────────────────────────────────────────────────
# Keep only the restoration paths that are currently safe and verified.
# Legacy/over-broad combinations are intentionally removed to avoid dead code.

_ROUTES = {
    "CLEAN_ENHANCE": {
        "pipeline_name": "CLEAN_ENHANCE",
        "run_damage_repair": False,
        "run_face_restore": False,
        "run_colorization": False,
        "run_denoising": False,
        "run_sr": True,
        "description": "Clean image — enhancement and SR only",
    },
    "NOISY_CLEAN": {
        "pipeline_name": "NOISY_CLEAN",
        "run_damage_repair": False,
        "run_face_restore": False,
        "run_colorization": False,
        "run_denoising": True,
        "run_sr": True,
        "description": "Noisy image without structural damage",
    },
    "NOISY_FACE": {
        "pipeline_name": "NOISY_FACE",
        "run_damage_repair": False,
        "run_face_restore": True,
        "run_colorization": False,
        "run_denoising": True,
        "run_sr": True,
        "description": "Noisy face — denoise and face restore",
    },
    "DAMAGED_NO_FACE": {
        "pipeline_name": "DAMAGED_NO_FACE",
        "run_damage_repair": True,
        "run_face_restore": False,
        "run_colorization": False,
        "run_denoising": True,
        "run_sr": True,
        "description": "Structural damage, no face detected",
    },
    "DAMAGED_FACE": {
        "pipeline_name": "DAMAGED_FACE",
        "run_damage_repair": True,
        "run_face_restore": True,
        "run_colorization": False,
        "run_denoising": True,
        "run_sr": True,
        "description": "Structural damage with face detected",
    },
    "BW_RESTORE": {
        "pipeline_name": "BW_RESTORE",
        "run_damage_repair": False,
        "run_face_restore": True,
        "run_colorization": False,
        "run_denoising": True,
        "run_sr": True,
        "description": "Black-and-white portrait restore without forced colorization",
    },
}

# Default fallback — stay conservative and safe for legacy/unknown images.
_DEFAULT_ROUTE = {
    "pipeline_name": "SAFE_FALLBACK",
    "run_damage_repair": False,
    "run_face_restore": True,
    "run_colorization": False,
    "run_denoising": True,
    "run_sr": True,
    "description": "Safe fallback — no risky auto-colorization",
}


class PipelineRouter:
    """Selects the optimal pipeline route based on image analysis."""

    @staticmethod
    def route(context: PipelineContext) -> Dict[str, Any]:
        """
        Analyze image metadata and return a route plan dict.

        Authority contract (Q-1): the returned run_* flags are consumed by
        restore() and ANDed with the user's enable_* flags — a stage runs only
        when BOTH the route and the user allow it. The route can therefore
        narrow the pipeline on clean images (skip LaMa/GFPGAN) but can never
        override an explicit user opt-out. Stage-internal refinements (e.g.
        Stage 8's noise gate) remain in their stage bodies.
        """
        meta = context.image_metadata
        damage_pct     = float(meta.get("damage_percentage", 0.0))
        has_flakes     = bool(meta.get("has_white_flakes", False))
        blur_score     = float(meta.get("blur_score", 999.0))
        noise_score    = float(meta.get("noise_score", 0.0))
        color_type     = meta.get("color_type", "color")
        is_noisy       = noise_score > 6.0
        is_blurry      = blur_score < 200.0
        has_damage     = has_flakes or damage_pct >= 5.0
        heavy_damage   = damage_pct >= 20.0
        dark_spots     = float(meta.get("dark_spot_percentage", 0.0))

        _use_disc = os.environ.get("PIXREVIVE_STRUCTURE_DISCRIMINATOR", "1").strip().lower() not in ("0", "false", "no", "off")
        has_scratches  = bool(meta.get("has_scratches", False))
        has_creases    = bool(meta.get("has_creases", False))

        if _use_disc:
            has_damage = has_flakes or damage_pct >= 5.0 or has_scratches
            # Distinguish genuine foxing spots / crease shadows on smooth paper (noise_score < 6.0, dark_spots >= 0.12)
            # Exclude saturated color photos misclassified as sepia from B&W foxing escalation
            is_monochrome_toned = (color_type == "grayscale") or (color_type == "sepia" and float(meta.get("mean_saturation", 0.0)) <= 70.0)
            foxing_damage = (dark_spots >= 0.12 and noise_score < 6.0 and is_monochrome_toned)
            needs_bw_repair = heavy_damage or has_damage or foxing_damage or has_creases
        else:
            needs_bw_repair = heavy_damage or has_damage or dark_spots >= 0.05

        # Route selection logic
        if color_type in ("grayscale", "sepia"):
            # Damaged B&W/sepia photos MUST repair scratches before colorization.
            # Routing them straight to BW_RESTORE skips damage repair and causes
            # DDColor to colorize bright scratch pixels with wrong hues. Dark-spot
            # evidence (foxing/mold stipple) escalates too: a spotted print can
            # sit under the %-thresholds while the repair is still wanted.
            if needs_bw_repair:
                route = _ROUTES["DAMAGED_FACE"].copy()
                route["description"] = "Damaged B&W — repair scratches first, then restore"
            else:
                route = _ROUTES["BW_RESTORE"].copy()
        elif heavy_damage:
            route = _ROUTES["DAMAGED_FACE"].copy()
        elif has_damage and is_blurry:
            route = _ROUTES["DAMAGED_FACE"].copy()
        elif has_damage:
            route = _ROUTES["DAMAGED_NO_FACE"].copy()
        elif is_noisy and is_blurry:
            route = _ROUTES["NOISY_FACE"].copy()
        elif is_noisy:
            route = _ROUTES["NOISY_CLEAN"].copy()
        elif not is_noisy and not has_damage and blur_score > 200.0:
            route = _ROUTES["CLEAN_ENHANCE"].copy()
        else:
            route = _DEFAULT_ROUTE.copy()

        # Log the decision
        context.add_log(
            f"PipelineRouter: selected route '{route['pipeline_name']}' — "
            f"damage={damage_pct:.1f}%, noise={noise_score:.1f}, "
            f"blur={blur_score:.1f}, color_type={color_type}"
        )

        # Store route in metadata for downstream stages
        meta["pipeline_route"] = route
        return route
