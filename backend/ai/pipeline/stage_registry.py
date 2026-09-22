"""
stage_registry.py — Single source of truth for pipeline stage metadata.

Before this registry existed, the stage contract lived in four places that
drifted independently: the `on_progress(...)` calls scattered through
`restoration_pipeline.py`, the STAGES array in `frontend/src/components/
ProgressBar.jsx`, the KNOWN_STAGES set in `scripts/check_progress_stream.py`,
and the stage table in the docs. The frontend could not label a backend stage
id it had never heard of (this actually happened: `damage_detect` rendered as
"Processing...").

`STAGE_REGISTRY` is the ordered list of stages in *progress-emission* order.
Consumers:
  * `restoration_pipeline.restore()` — iterates the registry for gated stages
    (progress emission + method dispatch) instead of hardcoding them.
  * `scripts/export_stage_list.py` — writes `frontend/src/generated/stages.json`
    so the frontend imports the same list the backend emits.
  * `scripts/check_progress_stream.py` — validates live SSE streams against
    the registry instead of a hand-maintained set.
  * Future T-stages (deskew, flat-field, deblur) register here as first-class
    entries — insertion position and nominal percentage are explicit data,
    not code edits scattered across four files.

Note on ordering: the registry order matches the order progress events are
*emitted* (what the progress bar mirrors), which is not identical to the
order `restore()` *calls* stage methods (analysis and face detection run
before damage masking, but damage masking emits at 20% while face detection
emits at 12%). The `progress_pct` values are what keep the bar monotonic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class StageSpec:
    """One pipeline stage's contract with the progress system."""

    id: str                   # progress event stage id (frontend must know it)
    name: str                 # human-readable stage name
    method: Optional[str] = None  # RestorationPipeline method; None = inlined in restore()
    progress_pct: int = 0         # nominal percentage when the stage starts
    progress_message: str = ""    # default progress message
    debug_snapshot: Optional[str] = None  # DebugLogger.save_stage snapshot name, if any
    route_gate: Optional[str] = None      # pipeline_flags key that enables the stage; None = always run
    extra: Dict[str, Any] = field(default_factory=dict)  # future metadata (gates, tier, etc.)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "method": self.method,
            "progress_pct": self.progress_pct,
            "progress_message": self.progress_message,
            "debug_snapshot": self.debug_snapshot,
            "route_gate": self.route_gate,
        }


# Ordered by progress-emission order (see module docstring).
STAGE_REGISTRY: List[StageSpec] = [
    StageSpec(
        id="despeckle",
        name="Impulse & speck suppression",
        method="_stage_0_despeckle",
        progress_pct=3,
        progress_message="Suppressing impulse noise and specks...",
        debug_snapshot="00b_after_despeckle",
    ),
    StageSpec(
        id="analyze",
        name="Degradation analysis",
        method="_stage_1_analyze",
        progress_pct=10,
        progress_message="Analyzing image degradation profile...",
        debug_snapshot="01_after_analyze",
    ),
    StageSpec(
        id="face_detect",
        name="Early face & landmark detection",
        method="_stage_4_face_detect",
        progress_pct=12,
        progress_message="Detecting faces and facial landmarks...",
        debug_snapshot="04_after_face_detect",
    ),
    StageSpec(
        id="damage_detect",
        name="Multi-scale fracture masking",
        method="_stage_2_fracture_masking",
        progress_pct=20,
        progress_message="Detecting fractures and structural tears...",
        debug_snapshot="02_after_damage_detect",
        route_gate="enable_repair",
    ),
    StageSpec(
        id="inpaint",
        name="LaMa deep inpainting",
        method="_stage_3_lama_inpaint",
        progress_pct=30,
        progress_message="Inpainting structural damage with LaMa...",
        debug_snapshot="03_after_inpaint",
        route_gate="enable_repair",
    ),
    StageSpec(
        id="damage_cluster_smooth",
        name="High-variance damage cluster smoothing",
        method="_stage_2b_damage_cluster_smooth",
        progress_pct=35,
        progress_message="Smoothing high-variance damage clusters...",
        debug_snapshot="02b_after_damage_cluster_smooth",
        # Gate is None because restore() gates 2b on inpaint coverage too —
        # the loop must not re-run it behind enable_denoise alone. It still
        # EMITS progress, so it must stay in the frontend export.
        route_gate=None,
    ),
    StageSpec(
        id="refresh_faces",
        name="Post-inpaint face re-detection",
        method="_stage_3b_refresh_faces",
        progress_pct=0,
        progress_message="",
        debug_snapshot="03b_after_refresh_faces",
        route_gate=None,
        # Internal orchestration stage: no progress event is emitted for it,
        # so it is excluded from the frontend list and the stream audit.
        extra={"internal": True},
    ),
    StageSpec(
        id="denoise",
        name="Spatially-gated 3-zone denoising",
        method="_stage_5_spatially_gated_denoise",
        progress_pct=50,
        progress_message="Applying spatially-gated 3-zone denoising...",
        debug_snapshot="05_after_denoise",
        route_gate="enable_denoise",
    ),
    StageSpec(
        id="colorize",
        name="DDColor colorization",
        method="_stage_6_colorization",
        progress_pct=65,
        progress_message="Colorizing with DDColor (ICCV 2023)...",
        debug_snapshot="06_after_colorize",
        route_gate="enable_colorization",
    ),
    StageSpec(
        id="face_restore",
        name="GFPGAN face restoration",
        method="_stage_7_face_restoration",
        progress_pct=78,
        progress_message="Restoring facial details with GFPGAN...",
        debug_snapshot="07_after_face_restore",
        route_gate="enable_face_restore",
    ),
    StageSpec(
        id="pre_sr",
        name="Pre-SR anti-aliasing",
        method="_stage_8_presr_antialiasing",
        progress_pct=85,
        progress_message="Applying pre-SR anti-aliasing filter...",
        debug_snapshot="08_after_pre_sr",
        route_gate="enable_sr",
    ),
    StageSpec(
        id="super_res",
        name="Real-ESRGAN super-resolution",
        method="_stage_9_super_resolution",
        progress_pct=90,
        progress_message="Upscaling with Real-ESRGAN via SuperResolution.restore()...",
        debug_snapshot="09_after_super_res",
        route_gate="enable_sr",
    ),
    StageSpec(
        id="sharpen",
        name="Post-SR luminance sharpening",
        method="_stage_10_post_sr_sharpen",
        progress_pct=95,
        progress_message="Enhancing internal borders and contours...",
        debug_snapshot="10_after_sharpen",
        route_gate="enable_sr",
    ),
    StageSpec(
        id="projector_grade",
        name="Projector exhibition grading",
        method="_stage_10b_projector_grade",
        progress_pct=98,
        progress_message="Applying projector exhibition grading...",
        debug_snapshot="10b_after_projector_grade",
        route_gate=None,
    ),
]

# Terminal events (not stages — the dot row hides at 100%).
EDGE_EVENTS: List[StageSpec] = [
    StageSpec(id="loading", name="Loading AI models", progress_pct=0),
    StageSpec(id="complete", name="Complete", progress_pct=100),
    StageSpec(id="done", name="Complete", progress_pct=100),
]


def get_stage(stage_id: str) -> Optional[StageSpec]:
    """Return the StageSpec for a progress id, or None if unknown."""
    for spec in STAGE_REGISTRY:
        if spec.id == stage_id:
            return spec
    for spec in EDGE_EVENTS:
        if spec.id == stage_id:
            return spec
    return None


def frontend_stages() -> List[StageSpec]:
    """Registry stages the frontend should render (internal stages excluded)."""
    return [s for s in STAGE_REGISTRY if not s.extra.get("internal")]


def stage_ids() -> List[str]:
    """All ids a live stream may emit: frontend stages first, then terminals."""
    return [s.id for s in frontend_stages()] + [s.id for s in EDGE_EVENTS]


def registry_to_json() -> Dict[str, Any]:
    """Serialize the registry for frontend consumption (stages.json)."""
    return {
        "stages": [s.to_dict() for s in frontend_stages()],
        "edge_events": [s.to_dict() for s in EDGE_EVENTS],
    }
