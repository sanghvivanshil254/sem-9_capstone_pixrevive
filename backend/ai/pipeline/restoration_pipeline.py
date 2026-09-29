"""
restoration_pipeline.py — Exhibition-grade photo restoration pipeline (v12 — Diagnostic Pass).

DIAGNOSTIC STAGE SNAPSHOT EXECUTION SEQUENCE & ORDER:
---------------------------------------------------
When PIXREVIVE_DEBUG=1 is set, the following numbered PNG snapshots are saved
to `debug_output/` sequentially:

00_input_raw.png                   — Raw input image before any processing
00b_after_despeckle.png            — After Stage 0 Impulse / Speck Suppression
01_after_analyze.png               — After Stage 1 Degradation Analysis
02_after_damage_detect.png         — After Stage 2 Multi-Scale Fracture Masking
02_before_inpaint.png              — Working image immediately prior to Stage 3 LaMa Inpainting
03_after_inpaint.png               — After Stage 3 LaMa Inpainting + Seam Smoothing
02b_after_damage_cluster_smooth.png— After Stage 2b High-Variance Damage Zone Smoothing
04_after_face_detect.png           — After Stage 4 RetinaFace Face Detection
04b_before_denoise.png             — Working image immediately prior to Stage 5 Spatially-Gated Denoising
05_after_denoise.png               — After Stage 5 Spatially-Gated 3-Zone Denoising
05c_before_colorize.png            — Working image immediately prior to Stage 6 DDColor Colorization
06_after_colorize.png              — After Stage 6 DDColor Colorization
06c_before_face_restore.png        — Working image immediately prior to Stage 7 GFPGAN Face Restoration
07_after_face_restore.png          — After Stage 7 GFPGAN Face Restoration
07b_before_pre_sr.png              — Working image immediately prior to Stage 8 Pre-SR Anti-Aliasing
08_after_pre_sr.png                — After Stage 8 Pre-SR Bilateral Anti-Aliasing
08b_before_super_res.png           — Working image immediately prior to Stage 9 Real-ESRGAN Super-Resolution
09_after_super_res.png             — After Stage 9 Real-ESRGAN x4+ Super-Resolution (4x upscale)
09b_before_sharpen.png             — Working image immediately prior to Stage 10 Post-SR Sharpening
10_after_sharpen.png               — After Stage 10 Post-SR Luminance Sharpening
10_before_projector_grade.png      — Working image immediately prior to Stage 10b Projector Grading
10b_after_projector_grade.png      — Final restored output after Stage 10b Projector Exhibition Grading

Summary outputs in `debug_output/`:
02_enhanced_mask.png / 04_final_mask.png — Binary crack masks
05_final_working.png                     — Final image
06_timings.txt                           — Per-stage execution duration in seconds
07_full_log.txt                          — Detailed execution logs
08_metadata.json                         — Quality scores, face bboxes, and image dimensions
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from typing import Dict, Any, Optional, Callable, List

import cv2
import numpy as np
import torch
from PIL import Image

from ai.pipeline.pipeline_context import PipelineContext
from ai.pipeline.pipeline_router import PipelineRouter
from ai.pipeline.stage_registry import STAGE_REGISTRY
from ai.preprocessing.image_normalizer import ImageNormalizer
from ai.preprocessing.halftone_suppressor import HalftoneSuppressor
from ai.analysis.quality_analyzer import QualityAnalyzer
from ai.analysis.face_detector import FaceDetector
from ai.damage.enhanced_detector import EnhancedDamageDetector
from ai.damage.tearmark_detector import TearmarkDetector
from ai.damage.mask_refiner import MaskRefiner
from ai.restoration.inpainting_engine import InpaintingEngine
from ai.restoration.face_restorer import FaceRestorer
from ai.colorization.ddcolor_colorizer import DDColorColorizer
from ai.colorization.color_refiner import ColorArtifactReducer
from ai.restoration.super_resolution import SuperResolution
from ai.postprocessing.enhancer import Harmonizer
from ai.pipeline.debug_logger import DebugLogger
from ai.pipeline.stages._helpers import _refresh_sharpness
from ai.pipeline.stages.preprocess_stages import PreprocessStagesMixin
from ai.pipeline.stages.damage_stages import DamageStagesMixin
from ai.pipeline.stages.color_stages import ColorStagesMixin
from ai.pipeline.stages.finish_stages import FinishStagesMixin

logger = logging.getLogger(__name__)


class RestorationPipeline(
    PreprocessStagesMixin,
    DamageStagesMixin,
    ColorStagesMixin,
    FinishStagesMixin,
):
    """Production Photo Restoration Pipeline with Projector Exhibition Grading
    & Diagnostic Instrumentation.

    Thin orchestrator: model lifecycle, device management, health, routing and
    the restore()/process() drivers. Stage implementations live in
    ai/pipeline/stages/ mixins (preprocess / damage / color / finish); the
    stage contract — ids, percentages, snapshots, route gates — lives in
    stage_registry.py.
    """

    def __init__(self, device: Optional[str] = None, load_models: bool = True) -> None:
        # Precedence: explicit argument > PIXREVIVE_DEVICE env var > auto pick.
        # Tests force CPU (several full pipelines loading on one 4 GB GPU OOM
        # nondeterministically); production leaves the env var unset.
        env_device = os.getenv("PIXREVIVE_DEVICE", "").strip().lower()
        chosen = device or env_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = chosen if chosen in ("cuda", "cpu") else "cpu"

        self.router = PipelineRouter()
        self.enhanced_detector = EnhancedDamageDetector()
        self.tearmark_detector = TearmarkDetector()
        self.face_detector = FaceDetector(device=self._device)
        self.face_restorer = FaceRestorer()
        self.colorizer = DDColorColorizer()
        self.super_resolution = SuperResolution()

        self._colorizer_ready = threading.Event()
        self._models_ready = threading.Event()
        self._model_load_failures: Dict[str, str] = {}
        self.inpainting_engine: Optional[InpaintingEngine] = None
        self._lama = None
        self._deblur = None  # T-3: lazy — loads on first gate-fire only

        if load_models:
            self.initialise_models()
        else:
            # No loaders will ever run: resolve the readiness events NOW so
            # stages that wait on them (colorization's 60s timeout, Stage 7's
            # model wait) fail fast into their graceful-degradation paths
            # instead of burning the full timeout per run. Found by the
            # robustness audit 2026-09-25.
            self._colorizer_ready.set()
            self._models_ready.set()

    def initialise_models(self) -> None:
        """Load deep learning models asynchronously in background threads."""
        def _record_failure(name: str, exc: Exception) -> None:
            """Remember a loader failure so /api/health can report per-model truth."""
            logger.warning("%s load failed: %s", name, exc)
            self._model_load_failures[name] = str(exc)

        def _load_all_fast():
            try:
                try:
                    from ai.restoration.lama_restorer import LamaRestorer
                    self._lama = LamaRestorer()
                    self._lama.load(self._device)
                except Exception as exc:
                    _record_failure("lama", exc)
                    self._lama = None

                try:
                    self.face_restorer.load(self._device)
                except Exception as exc:
                    _record_failure("gfpgan", exc)

                try:
                    self.super_resolution.load(self._device)
                except Exception as exc:
                    _record_failure("realesrgan", exc)

                self.inpainting_engine = InpaintingEngine(self._lama)
                logger.info("[OK] Fast pipeline models loaded on %s.", self._device)
            finally:
                self._models_ready.set()

        def _load_colorizer():
            try:
                self.colorizer.load(self._device)
            except Exception as exc:
                _record_failure("ddcolor", exc)
            finally:
                self._colorizer_ready.set()

        threading.Thread(target=_load_all_fast, name="fast-model-loader", daemon=True).start()
        threading.Thread(target=_load_colorizer, name="colorizer-loader", daemon=True).start()

    @staticmethod
    def _should_colorize(enable_colorization: bool, color_type: str) -> bool:
        """Return whether colorization should run based on user flag and image color type."""
        return enable_colorization and (color_type in ("grayscale", "sepia"))

    @staticmethod
    def _effective_stage_flags(
        route: Dict[str, Any],
        color_type: str,
        *,
        enable_repair: bool,
        enable_denoise: bool,
        enable_colorization: bool,
        enable_face_restore: bool,
        enable_sr: bool,
    ) -> Dict[str, bool]:
        """AND the router's run_* authority with the user's enable_* flags (Q-1).

        A stage runs only when BOTH the route and the user allow it, so a clean
        image's route narrows the pipeline while an explicit user opt-out always
        wins. Colorization is the one carve-out: no route preset carries
        run_colorization (it is meaningless on color images — Stage 6 self-skips
        those), but the B&W/sepia restoration brief REQUIRES colorization to
        follow the user toggle, so the route cannot veto it there.
        """
        _is_bw = color_type in ("grayscale", "sepia")
        return {
            "run_repair": enable_repair and bool(route.get("run_damage_repair", True)),
            "run_denoise": enable_denoise and bool(route.get("run_denoising", True)),
            "run_face": enable_face_restore and bool(route.get("run_face_restore", True)),
            "run_colorize": enable_colorization and (
                _is_bw or bool(route.get("run_colorization", True))
            ),
            "run_sr": enable_sr and bool(route.get("run_sr", True)),
        }

    def health_status(self) -> Dict[str, Any]:
        """Per-model load truth: a boolean per model plus each failed loader's error.

        Consumers (main.py health endpoint, Navbar degraded-mode pill) must be
        able to distinguish "all models loaded" from "degraded: which one".
        """
        defect_ready = (
            getattr(self.enhanced_detector, "defect_segmenter", None) is not None
            and getattr(self.enhanced_detector.defect_segmenter, "is_loaded", False)
        )
        return {
            "status": "ready" if self._models_ready.is_set() else "loading",
            "models": {
                "defect_segmenter": defect_ready,
                "lama": self._lama is not None and getattr(self._lama, "is_loaded", False),
                "deblur": self._deblur is not None and self._deblur.is_loaded(),
                "gfpgan": getattr(self.face_restorer, "gfpganer", None) is not None,
                "realesrgan": getattr(self.super_resolution, "realesrgan", None) is not None or self.super_resolution.is_loaded,
                "ddcolor": self.colorizer.is_loaded,
            },
            "load_errors": dict(self._model_load_failures),
            "device": self._device,
        }

    def is_colorizer_ready(self) -> bool:
        """Return whether DDColor colorization model is fully loaded."""
        return self._colorizer_ready.is_set() and self.colorizer.is_loaded

    def is_ready(self) -> bool:
        """Return whether fast model set is loaded."""
        return self._models_ready.is_set()

    # ── Device management (U-5 / plan 3.3) ────────────────────────────────
    def _reload_models_on(self, device: str) -> None:
        """Synchronously (re)load every model onto `device`.

        Used by _ensure_device when a request's device_preference disagrees
        with the device models are currently loaded on. Failures are recorded
        the same way startup records them (visible via /api/health).
        """
        try:
            from ai.restoration.lama_restorer import LamaRestorer
            self._lama = LamaRestorer()
            self._lama.load(device)
        except Exception as exc:
            logger.warning("LaMa reload failed: %s", exc)
            self._model_load_failures["lama"] = str(exc)
            self._lama = None

        try:
            self.face_restorer.load(device)
        except Exception as exc:
            self._model_load_failures["gfpgan"] = str(exc)

        try:
            self.super_resolution.load(device)
        except Exception as exc:
            self._model_load_failures["realesrgan"] = str(exc)

        try:
            self.colorizer.load(device)
        except Exception as exc:
            self._model_load_failures["ddcolor"] = str(exc)
        finally:
            self._colorizer_ready.set()

        try:
            from ai.analysis.face_detector import FaceDetector
            self.face_detector = FaceDetector(device=device)
        except Exception as exc:
            self._model_load_failures["face_detector"] = str(exc)

        self.inpainting_engine = InpaintingEngine(self._lama)
        self._device = device
        self._models_ready.set()
        logger.info("[OK] Models reloaded on %s per device preference.", device)

    def _ensure_device(self, preference: str) -> str:
        """Resolve a request's device_preference; lazily reload on mismatch.

        'auto' keeps whatever is loaded. 'cuda'/'cpu' honor the preference;
        'cuda' falls back to the current device (with a log) when CUDA is
        unavailable. Returns the effective device.
        """
        pref = (preference or "auto").strip().lower()
        if pref not in ("auto", "cuda", "cpu"):
            pref = "auto"
        if pref == "auto" or pref == self._device:
            return self._device
        if pref == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "device_preference='cuda' but CUDA is unavailable - staying on %s.",
                self._device,
            )
            return self._device
        logger.info(
            "device_preference='%s' differs from loaded device '%s' - reloading models.",
            pref,
            self._device,
        )
        self._reload_models_on(pref)
        return self._device

    def _wait_for_models(self, timeout: float = 300.0) -> bool:
        """Wait for background model loading threads to complete."""
        return self._models_ready.wait(timeout=timeout)

    # ── Stage Implementations (0 to 10b) ──────────────────────────────────────

    def restore(
        self,
        image_input: Any,
        external_metadata: Optional[Dict[str, Any]] = None,
        enable_repair: bool = True,
        enable_denoise: bool = True,
        enable_colorization: bool = True,
        enable_face_restore: bool = True,
        enable_sr: bool = True,
        enable_reimagine: bool = False,
        on_progress: Optional[Callable[[str, int, str], None]] = None,
    ) -> np.ndarray:
        """
        Execute full restoration pipeline with exhibition grading and per-stage diagnostic snapshots.
        
        Args:
            image_input: File bytes, PIL Image, or numpy RGB array
            external_metadata: Optional dict of external metadata to pass into PipelineContext
            enable_repair: Enable LaMa structural inpainting
            enable_denoise: Enable spatially-gated multi-zone denoising
            enable_colorization: Enable DDColor colorization for B&W/sepia
            enable_face_restore: Enable GFPGAN face restoration
            enable_sr: Enable Real-ESRGAN x4 super-resolution and post-SR sharpening
            enable_reimagine: Opt-in beta (Phase H1): re-imagine unrecoverable
                damage regions with evidence-preserving diffusion after the
                standard chain (docs/HYBRID_DIFFUSION_DESIGN.md)
            on_progress: Progress callback (stage_name, percent, message)

        Returns:
            Restored RGB uint8 numpy array at full 4x resolution
        """
        if isinstance(image_input, bytes):
            img_pil = Image.open(io.BytesIO(image_input)).convert("RGB")
            img_rgb = np.array(img_pil)
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            img_rgb = image_input.copy()
            # Robustness audit (2026-09-25): raw-array callers can hand us
            # 2D grayscale or float arrays (NaN included). The HTTP path can
            # never produce these (PIL .convert("RGB") normalizes), but the
            # library API must not crash with raw OpenCV channel/depth
            # errors — normalize exactly like the bytes path does.
            if img_rgb.ndim == 2:
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
            elif img_rgb.ndim == 3 and img_rgb.shape[2] == 4:
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
            if img_rgb.dtype != np.uint8:
                arr = np.nan_to_num(img_rgb.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
                img_rgb = np.clip(arr, 0, 255).astype(np.uint8)
            if img_rgb.ndim != 3 or img_rgb.shape[2] != 3 or img_rgb.size == 0:
                raise ValueError(f"Unsupported image array shape: {img_rgb.shape}")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -- T-4: Auto-deskew (registry stage 'deskew', pre-context) -------------
        # Skew must be corrected before the border crop so the rotation wedges
        # land in the margins and get cropped away. Gated: inert (bitwise
        # unchanged) unless dominant skew >= 0.3deg and <= 8deg.
        try:
            img_rgb, deskew_angle = ImageNormalizer.deskew(img_rgb)
            if on_progress:
                on_progress("deskew", 1, "Detecting and correcting scan skew...")
            if abs(deskew_angle) > 0:
                deskew_meta: Dict[str, Any] = {
                    "deskew_applied": True,
                    "deskew_angle_deg": round(deskew_angle, 3),
                }
            else:
                deskew_meta = {"deskew_applied": False}
        except Exception as exc:
            logger.warning("T-4 deskew failed, using uncorrected input: %s", exc)
            deskew_meta = {"deskew_applied": False, "deskew_error": str(exc)}

        # Auto-detect and crop scanner framing margins / solid canvas borders
        img_rgb, has_border, border_bbox = ImageNormalizer.detect_and_crop_borders(img_rgb)

        # -- T-5: Flat-field illumination (registry stage 'flat_field') --------
        # Runs AFTER the border crop and still pre-context: a solid dark photo
        # border is exactly a symmetric corner deficit and would fake a lamp
        # defect (and poison the gain field). Gated on symmetric corner
        # deficit (>=20% with spread <=12%) — inert on all 17 real dataset
        # images by design; fires on scanner vignetting. After the crop the
        # context image is lamp-corrected before analysis measures it.
        try:
            img_rgb, ff_info = ImageNormalizer.correct_illumination(img_rgb)
            ff_info["flat_field_applied"] = ff_info.pop("applied", False)
            flat_field_meta: Dict[str, Any] = ff_info
            if on_progress:
                on_progress("flat_field", 2, "Flattening scan illumination...")
        except Exception as exc:
            logger.warning("T-5 flat-field failed, using uncorrected input: %s", exc)
            flat_field_meta = {"flat_field_applied": False, "flat_field_error": str(exc)}

        meta = dict(external_metadata) if external_metadata else {}
        if has_border:
            meta["border_cropped"] = True
            meta["border_bbox"] = border_bbox
        meta.update(deskew_meta)
        meta.update(flat_field_meta)

        context = PipelineContext(img_rgb, meta)
        if deskew_meta.get("deskew_applied"):
            context.add_log(
                f"Stage -1 Deskew: corrected {deskew_meta['deskew_angle_deg']:+.2f}deg "
                "dominant scan skew (T-4)."
            )
        if flat_field_meta.get("flat_field_applied"):
            context.add_log(
                f"Stage -1 FlatField: corrected symmetric illumination falloff "
                f"(deficit={flat_field_meta.get('corner_deficit')}, "
                f"spread={flat_field_meta.get('corner_spread')}) (T-5)."
            )
        if has_border:
            context.add_log(
                f"Stage 0 Pre: Auto-cropped solid framing margins "
                f"(y_min={border_bbox[0]}, y_max={border_bbox[1]}, x_min={border_bbox[2]}, x_max={border_bbox[3]})."
            )
        context.pipeline_flags.update({
            "enable_repair": enable_repair,
            "enable_denoise": enable_denoise,
            "enable_colorization": enable_colorization,
            "enable_face_restore": enable_face_restore,
            "enable_sr": enable_sr,
            "enable_reimagine": enable_reimagine,
        })
        self.last_context = context

        DebugLogger.save_stage(context, "00_input_raw")

        # -- Stage 0: Impulse / Speck Suppression --------------------------------
        # Ungated stages (despeckle, moire, analyze, face detect) are called
        # directly: restore() threads their outputs (faces, route, damage mask)
        # through inline orchestration the registry cannot express.
        self._stage_0_despeckle(context, on_progress)

        # -- T-9: Moiré / halftone suppression (registry stage 'moire') ---------
        # The Phase 1 salvage: HalftoneSuppressor's FFT spectral-peak detector
        # + Gaussian notch rejection, self-gating (fires only on genuine
        # periodic mesh; probe-verified inert on all 17 real dataset photos
        # including the macro-edge-heavy case2 landscape, and its MAE safety
        # valve rejects drastic false positives). Runs before analysis so the
        # damage detectors and colorizer see the mesh as suppressed texture
        # rather than content.
        try:
            if on_progress:
                on_progress("moire", 4, "Suppressing halftone and moiré patterns...")
            moire_img = HalftoneSuppressor.suppress(context)
            context.working_image = moire_img
        except Exception as exc:
            logger.warning("T-9 moiré suppression failed, keeping input: %s", exc)

        # -- Stage 1: Degradation Analysis ---------------------------------------
        self._stage_1_analyze(context, on_progress)

        # -- Stage 1b / Early Face Detection --------------------------------------
        # Runs early on the input canvas so faces and landmarks are available to
        # MaskRefiner (Stage 2) and DamageClusterSmooth (Stage 2b), preventing
        # destructive inpainting of facial anatomy.
        faces = self._stage_4_face_detect(context, on_progress)

        # -- T-7: Dye-fade rebalance (registry stage 'cast_fix') -----------------
        # Input-side cast correction for COLOR-classified prints only; sepia
        # tone is historic character and is excluded by color_type (documented
        # false-negative, the safe direction). Uses the logic salvaged from the
        # deleted Enhancer (D17), now in ColorArtifactReducer.
        try:
            from ai.colorization.color_refiner import ColorArtifactReducer as _CAR
            color_type = context.image_metadata.get("color_type", "color")
            if on_progress:
                on_progress("cast_fix", 12, "Rebalancing faded print colors...")
            fixed_img, cast_info = _CAR.rebalance_dye_fade(
                context.working_image, color_type=color_type, context=context
            )
            if cast_info.get("applied"):
                context.working_image = fixed_img
            context.image_metadata["cast_fix"] = cast_info
        except Exception as exc:
            logger.warning("T-7 cast fix failed, keeping input: %s", exc)

        route = self.router.route(context)
        context.add_log(
            f"PipelineRouter: '{route['pipeline_name']}' - {route['description']}"
        )
        context.image_metadata["route_name"] = route.get("pipeline_name", "unknown")
        flags = self._effective_stage_flags(
            route,
            context.image_metadata.get("color_type", "color"),
            enable_repair=enable_repair,
            enable_denoise=enable_denoise,
            enable_colorization=enable_colorization,
            enable_face_restore=enable_face_restore,
            enable_sr=enable_sr,
        )
        run_repair = flags["run_repair"]
        run_denoise = flags["run_denoise"]
        run_face = flags["run_face"]
        run_colorize_flag = flags["run_colorize"]
        run_sr = flags["run_sr"]
        # Report evidence contract: record the colorization DECISION before
        # dispatch. Stage 6 flips it to True only on success, so the report's
        # "colors added by AI" claim always rests on what actually ran —
        # colorize=false on a sepia image can no longer emit the disclaimer
        # (measured 2026-09-27 via the API with the pre-fix build).
        context.image_metadata["colorization_applied"] = False
        if not run_colorize_flag:
            context.image_metadata["colorization_skipped_by_flag"] = True

        # -- Stage 2: Multi-Scale Fracture Masking (returns the damage mask) -----
        damage_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        if run_repair:
            damage_mask = self._stage_2_fracture_masking(context, on_progress)

        # -- Stage 3: LaMa Deep Inpainting (needs the mask + canvas, not flags) --
        if run_repair and np.any(damage_mask > 0):
            self._stage_3_lama_inpaint(context, damage_mask, on_progress)

        # -- Stage 2b: High-Variance Damage Zone Smoother -------------------------
        if run_denoise:
            noise_score = float(context.image_metadata.get("noise_score", 0.0))
            inpaint_coverage = float(np.sum(damage_mask > 0) / max(damage_mask.size, 1) * 100) if damage_mask is not None else 0.0
            if noise_score > 3.0 and inpaint_coverage < 30.0:
                self._stage_2b_damage_cluster_smooth(context, on_progress)
            elif noise_score <= 3.0:
                context.add_log(
                    f"Stage 2b: noise_score={noise_score:.2f} <= 3.0 - "
                    "skipping (clean image, no grain clusters to smooth)."
                )
            else:
                context.add_log(
                    f"Stage 2b: inpaint_coverage={inpaint_coverage:.1f}% >= 30% - "
                    "skipping to preserve LaMa inpaint result."
                )

        # -- Stage 3b: Post-Inpaint Face Re-Detection ------------------------------
        faces = self._stage_3b_refresh_faces(context, faces, on_progress)

        # ========================================================================
        # Registry-driven stage dispatch (A-2).
        # Linear, flag-gated stages iterate STAGE_REGISTRY: each StageSpec
        # carries the method and the pipeline_flags gate, so adding a stage is
        # a registry entry, not a hardcoded block. Progress is still emitted by
        # the stage bodies themselves (percent/message live in the registry for
        # the frontend export; the bodies remain the single emitters to avoid
        # double emissions). Inline composition stays here.
        #
        # The prelude above already ran despeckle/analyze/face_detect/
        # damage_detect/inpaint inline (their outputs are threaded through
        # local variables the loop cannot see) — those specs are skipped here
        # so no stage executes twice. (A loop re-run of inpaint would bind
        # on_progress into the mask parameter and poison the mask store.)
        # ========================================================================
        _ran_inline = {
            "deskew",  # pre-context: runs before PipelineContext exists (T-4)
            "despeckle", "analyze", "face_detect", "damage_detect", "inpaint",
            "damage_cluster_smooth", "refresh_faces",
        }
        for spec in STAGE_REGISTRY:
            if spec.id in _ran_inline:
                continue
            if spec.method is None:
                # Contract: method=None stages are dispatched by restore()
                # itself (inline or pre-context). getattr(self, None) would
                # raise; skipping keeps the loop total over the registry.
                continue

            # Route authority (Q-1): registry gates read the ANDed flags.
            gate = spec.route_gate
            if gate is not None:
                gate_value = {
                    "enable_repair": run_repair,
                    "enable_denoise": run_denoise,
                    "enable_colorization": run_colorize_flag,
                    "enable_face_restore": run_face,
                    "enable_sr": run_sr,
                }.get(gate, context.pipeline_flags.get(gate, False))
                if not gate_value:
                    context.add_log(
                        f"PipelineRouter: stage '{spec.id}' skipped "
                        f"(gate '{gate}' not granted by route+user flags)."
                    )
                    continue

            # Stage 8's original contract: bilateral anti-aliasing only on
            # noisy scans; quiet images keep their natural sharpness.
            if spec.id == "pre_sr":
                noise_score = float(context.image_metadata.get("noise_score", 0.0))
                if noise_score <= 2.5:
                    context.add_log(
                        "Stage 8 Pre-SR: noise_score <= 2.5 - preserving natural sharpness."
                    )
                    continue
                context.add_log(
                    f"Stage 8 Pre-SR: noise_score {noise_score:.2f} > 2.5 - "
                    "running pre-SR smoothing pass to keep grain from being amplified."
                )

            method = getattr(self, spec.method)
            # on_progress is passed BY KEYWORD: Stage 5's second positional
            # slot is `faces`, not the callback — a positional call bound the
            # callback into `faces` and Stage 5 died with "'function' object
            # is not iterable" on every registry-dispatched run (measured
            # 2026-09-29, offline run of the speckle figure).
            method(context, on_progress=on_progress)

            # -- inline composition that belongs to a specific stage -----------
            if spec.id == "denoise":
                try:
                    from ai.postprocessing.noise_router import NoiseRouter
                    noise_profile = NoiseRouter.analyze(context)
                    if noise_profile.has_noise:
                        NoiseRouter.reduce(context, noise_profile)
                        context.add_log(
                            f"Stage 4b NoiseRouter: applied filter(s) for "
                            f"{', '.join(noise_profile.detected_types[:3])}."
                        )
                    else:
                        context.add_log("Stage 4b NoiseRouter: no actionable noise detected.")
                except Exception as exc:
                    context.add_log(f"Stage 4b NoiseRouter failed: {exc}")
            elif spec.id == "colorize":
                _was_colorized = context.image_metadata.get("color_type", "color") in (
                    "grayscale", "sepia"
                )
                if _was_colorized:
                    try:
                        refined = ColorArtifactReducer.reduce_artifacts(
                            context.working_image, context=context
                        )
                        context.working_image = refined
                        context.add_log("Stage 6c ColorArtifactReducer: banding, saturation, edge & cast artifacts reduced.")
                    except Exception as exc:
                        context.add_log(f"Stage 6c ColorArtifactReducer failed: {exc}")
            elif spec.id == "super_res":
                try:
                    context.sync_geometry_to_working_image()
                    context.add_log("Stage 9 post-SR: geometry synced to SR resolution.")
                except Exception as exc:
                    context.add_log(f"Stage 9 sync_geometry failed: {exc}")
                if run_repair and np.any(context.damage_masks.get("final", np.zeros(1)) > 0):
                    try:
                        Harmonizer.harmonize(context)
                        context.add_log("Stage 9b Harmonizer: inpaint seam blending applied at SR resolution.")
                    except Exception as exc:
                        context.add_log(f"Stage 9b Harmonizer failed: {exc}")

        # -- Post-SR Deferred Face Restoration (9c) ------------------------------
        if run_face and context.pipeline_flags.get("face_restore_deferred_to_post_sr", False):
            context.pipeline_flags["face_restore_deferred_to_post_sr"] = False
            context.pipeline_flags["_running_deferred_face_restore"] = True
            try:
                self._stage_7_face_restoration(context, on_progress)
                context.add_log("Stage 9c Post-SR FaceRestoration: Executed deferred face restoration on super-resolved canvas.")
            finally:
                context.pipeline_flags["_running_deferred_face_restore"] = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Output quality gate + (debug-only) perceptual metrics (Q-6).
        try:
            from ai.analysis.quality_service import QualityService

            QualityService.validate(context)
            if os.getenv("PIXREVIVE_DEBUG", "").strip() in ("1", "true", "True"):
                QualityService.compute_metrics(context)
        except Exception as exc:
            logger.warning("QualityService post-check failed: %s", exc)

        DebugLogger.save_debug_images(context)

        # Plan 5.3 (default=wire): IdentityValidator runs as a debug-only,
        # advisory post-SR check — it never rolls the canvas back (that
        # failure mode is documented in docs/restoration_debug_plan.md §9);
        # it only feeds an identity-risk flag into the Restoration Report.
        if os.getenv("PIXREVIVE_DEBUG", "").strip() in ("1", "true", "True"):
            try:
                from ai.validation.identity_validator import IdentityValidator

                identity_metrics = IdentityValidator().validate_context(context)
                if not identity_metrics.identity_preserved:
                    warnings_list = context.image_metadata.setdefault("warnings", [])
                    if "identity_risk" not in warnings_list:
                        warnings_list.append("identity_risk")
                    context.add_log(
                        "[DEBUG] IdentityValidator: landmark stability below threshold "
                        "— flagged identity_risk in the restoration report."
                    )
            except Exception as exc:
                logger.warning("IdentityValidator debug check failed: %s", exc)

        # -- T-2 Restoration Report (plan 5.1): machine-readable honesty layer --
        # Classifies the run's evidence into recovered / reconstructed /
        # flagged and stores it in metadata; the API layer forwards it to the
        # UI's collapsible Restoration Report card.
        try:
            from ai.report.restoration_report import build_restoration_report

            context.image_metadata["restoration_report"] = build_restoration_report(context)
        except Exception as exc:
            logger.warning("Restoration report generation failed: %s", exc)

        if on_progress:
            on_progress("complete", 100, "Restoration complete!")

        return context.working_image

    def process(
        self,
        image_bytes: Any,
        external_metadata: Optional[Dict[str, Any]] = None,
        enable_restoration: bool = True,
        enable_repair: bool = True,
        enable_colorization: bool = True,   # Fix 13: aligned with restore() defaults
        enable_denoise: bool = True,         # Fix 13: aligned with restore() defaults
        enable_sr: bool = True,
        enable_reimagine: bool = False,      # Phase H1 opt-in beta
        device_preference: str = "auto",
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        on_progress: Optional[Callable[[str, int, str], None]] = None,
        **kwargs: Any,
    ) -> bytes:
        """
        Process an image through the restoration pipeline and return full 4x resolution PNG bytes.
        
        Args:
            image_bytes: Input image data (raw bytes, PIL Image, or numpy array)
            external_metadata: Optional extra metadata dictionary
            enable_restoration: Enable GFPGAN face restoration
            enable_repair: Enable LaMa fracture / scratch repair
            enable_colorization: Enable DDColor colorization
            enable_denoise: Enable spatially-gated multi-zone denoising
            enable_sr: Enable Real-ESRGAN x4 super-resolution and sharpening
            device_preference: Execution device preference ('auto', 'cuda', 'cpu')
            progress_callback: Callback for progress updates (stage, percent, message)
            on_progress: Alias for progress_callback

        Returns:
            PNG encoded bytes of the restored image at full 4x resolution
        """
        cb = progress_callback or on_progress

        # U-5: honor the request's device preference before any stage runs.
        # 'auto' (the default) is a no-op; an explicit mismatch lazily reloads.
        self._ensure_device(device_preference)

        restored_rgb = self.restore(
            image_input=image_bytes,
            external_metadata=external_metadata,
            enable_repair=enable_repair,
            enable_denoise=enable_denoise,
            enable_colorization=enable_colorization,
            enable_face_restore=enable_restoration,
            enable_sr=enable_sr,
            enable_reimagine=enable_reimagine,
            on_progress=cb,
        )

        out_pil = Image.fromarray(restored_rgb)
        buf = io.BytesIO()
        out_pil.save(buf, format="PNG")
        return buf.getvalue()
