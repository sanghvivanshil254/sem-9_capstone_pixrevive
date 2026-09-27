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
        # T-4 auto-deskew: the FIRST registry entry proving the T-stage
        # extension design. method=None because deskew must run before the
        # border crop — i.e. before PipelineContext even exists — so restore()
        # executes it ahead of the inline prelude. The registry still owns its
        # position, id, and contract, and the frontend export lists it.
        id="deskew",
        name="Auto-deskew (scan geometry correction)",
        method=None,  # inlined in restore() before border crop (pre-context)
        progress_pct=1,
        progress_message="Detecting and correcting scan skew...",
    ),
    StageSpec(
        # T-5 flat-field: second registry-registered taxonomy stage. Like
        # deskew, it runs pre-context (before analysis measures the image) so
        # damage thresholds see material, not lamp geometry. Gated on symmetric
        # corner deficit of the background envelope; inert on the whole real
        # corpus by design (see ImageNormalizer.correct_illumination).
        id="flat_field",
        name="Flat-field illumination correction",
        method=None,  # inlined in restore() after despeckle (pre-context)
        progress_pct=2,
        progress_message="Flattening scan illumination...",
    ),
    StageSpec(
        id="despeckle",
        name="Impulse & speck suppression",
        method="_stage_0_despeckle",
        progress_pct=3,
        progress_message="Suppressing impulse noise and specks...",
        debug_snapshot="00b_after_despeckle",
    ),
    StageSpec(
        # T-9 moire: the Phase 1 salvage (D5) — HalftoneSuppressor's FFT notch
        # wired as the moire specialist, ahead of the SR bilateral pre-filter.
        # Runs inline right after despeckle (before analysis): a periodic mesh
        # is texture, not damage, and every downstream detector + the colorizer
        # misread it as content. Self-gating (FFT spectral peaks + MAE safety
        # valve); probe-verified inert on all 17 real dataset photos including
        # the macro-edge-heavy case2 landscape.
        id="moire",
        name="Moiré & halftone suppression",
        method=None,  # inlined in restore() after despeckle (pre-context)
        progress_pct=4,
        progress_message="Suppressing halftone and moiré patterns...",
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
        # NOTE: listed BEFORE cast_fix because that is the actual runtime
        # order (restore() calls face detection right after analysis, the
        # cast fix after it). The registry once listed cast_fix first, which
        # made the emitted progress percent go 12 -> 11 — a ProgressBar
        # monotonicity break found by the 2026-09-25 robustness audit.
        progress_pct=11,
        progress_message="Detecting faces and facial landmarks...",
        debug_snapshot="04_after_face_detect",
    ),
    StageSpec(
        # T-7 dye-fade rebalance: input-side gray-world/channel-curve cast fix
        # for COLOR-classified prints, using the logic salvaged from the
        # deleted Enhancer (D17 -> ColorArtifactReducer). Runs after analysis
        # (needs color_type) and after early face detection, before damage
        # routing/detection: sepia tone is historic character and is excluded
        # BY TYPE — documented false-negative, the safe direction.
        id="cast_fix",
        name="Dye-fade color-cast rebalancing",
        method=None,  # inlined in restore() after analysis (needs color_type)
        progress_pct=12,
        progress_message="Rebalancing faded print colors...",
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
        # T-3 deblur (the taxonomy's last unaddressed damage class): vendored
        # einops-free Restormer with the official motion_deblurring checkpoint
        # (GitHub release -> backend/ai/weights/restormer/, runtime-rule
        # compliant). STRICT two-path gate: path A fires on defocus (edge-
        # spread acutance), path B on motion (H/V gradient anisotropy) —
        # probe-calibrated so the whole real corpus stays bitwise inert.
        # Weights load lazily on first fire (100MB VRAM on a 4GB card), so
        # /api/health reports 'deblur: skipped-or-unavailable' until a blurred
        # image actually arrives.
        id="deblur",
        name="Optical blur correction (Restormer)",
        method="_stage_4_5_deblur",
        progress_pct=56,
        progress_message="Correcting optical blur...",
        debug_snapshot="05b_after_deblur",
        route_gate="enable_denoise",  # blur correction shares denoise authority
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
        # T-6 deband runs inside _stage_8_presr_antialiasing (gated on the
        # noise router's banding score) — the SR step amplifies whatever
        # quantization stairs survive, so pre-SR is the last cheap exit.
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
    StageSpec(
        # Closed-loop verification (2026-09-25): the repair stages trust their
        # own masks, so input-side misses survived into every output. This
        # stage re-detects residue ON THE OUTPUT, micro-cleans it (Telea,
        # residual pixels only, faces excluded), and runs the color-coherence
        # and texture-plausibility checks that surface implausible
        # re-synthesis in the Restoration Report.
        id="output_verify",
        name="Closed-loop output verification",
        method="_stage_11_output_verify",
        progress_pct=99,
        progress_message="Verifying the restored output...",
        debug_snapshot="11_after_output_verify",
        route_gate=None,
    ),
    StageSpec(
        # Phase H1 — re-imagine (opt-in beta): hybrid evidence-preserving
        # diffusion for UNRECOVERABLE damage (>= 35% of frame or >= 55% of a
        # face is Destroy-class). Runs as the FINAL content stage: the
        # evidence contract out[E] == pre-reimagine[E] must hold on the
        # shipped image, so nothing may modify pixels after this stage.
        # Skips gracefully when off / not opted-in / coverage below the
        # hopeless bar / resources or weights unavailable — the report
        # carries the reason either way (docs/HYBRID_DIFFUSION_DESIGN.md).
        id="reimagine",
        name="Re-imagine lost regions (beta)",
        method="_stage_12_reimagine",
        progress_pct=99,
        progress_message="Re-imagining unrecoverable regions (beta)...",
        debug_snapshot="12_after_reimagine",
        route_gate="enable_reimagine",
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
