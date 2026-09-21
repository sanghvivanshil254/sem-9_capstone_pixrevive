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
05_before_hair_refine.png          — Working image immediately prior to Stage 5b Hair Zone Refinement
05b_after_hair_refine.png          — After Stage 5b Hair Zone Bilateral Refinement & Warm Brown Enrichment
05c_before_colorize.png            — Working image immediately prior to Stage 6 DDColor Colorization
06_after_colorize.png              — After Stage 6 DDColor Colorization
06_before_skin_enforce.png         — Working image immediately prior to Stage 6b Skin Tone Enforcement
06b_after_skin_enforce.png         — After Stage 6b Face Skin Tone Enforcement
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
from ai.preprocessing.image_normalizer import ImageNormalizer
from ai.analysis.quality_analyzer import QualityAnalyzer
from ai.analysis.face_detector import FaceDetector
from ai.damage.enhanced_detector import EnhancedDamageDetector
from ai.damage.tearmark_detector import TearmarkDetector
from ai.damage.mask_refiner import MaskRefiner
from ai.restoration.inpainting_engine import InpaintingEngine
from ai.restoration.face_restorer import FaceRestorer
from ai.colorization.ddcolor_colorizer import DDColorColorizer
from ai.colorization.color_refiner import ColorArtifactReducer
from ai.restoration.super_resolution import SuperResolution, _realesrgan_tile_forward
from ai.postprocessing.enhancer import Harmonizer
from ai.pipeline.debug_logger import DebugLogger

logger = logging.getLogger(__name__)


def _refresh_sharpness(context: PipelineContext) -> None:
    """Re-measure blur_score on current working image."""
    try:
        gray = cv2.cvtColor(context.working_image, cv2.COLOR_RGB2GRAY)
        refreshed = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        meta = context.image_metadata
        previous = float(meta.get("blur_score", 0.0))
        meta.setdefault("blur_score_raw", previous)
        meta["blur_score"] = refreshed
        meta["blur_score_norm"] = min(refreshed / 500.0, 1.0)
    except Exception as exc:
        context.add_log(f"[WARN] Could not refresh blur_score: {exc}")


class RestorationPipeline:
    """Production Photo Restoration Pipeline with Projector Exhibition Grading & Diagnostic Instrumentation."""

    def __init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self.router = PipelineRouter()
        self.enhanced_detector = EnhancedDamageDetector()
        self.tearmark_detector = TearmarkDetector()
        self.face_detector = FaceDetector(device=self._device)
        self.face_restorer = FaceRestorer()
        self.colorizer = DDColorColorizer()
        self.super_resolution = SuperResolution()

        self._colorizer_ready = threading.Event()
        self._models_ready = threading.Event()
        self.inpainting_engine: Optional[InpaintingEngine] = None
        self._lama = None
        self._orchestrator: Optional[Any] = None

        self.initialise_models()

    @property
    def orchestrator(self):
        """Lazy-loaded PhotoOrchestrator instance."""
        if self._orchestrator is None:
            from ai.orchestrator.orchestrator import PhotoOrchestrator
            self._orchestrator = PhotoOrchestrator(face_detector=self.face_detector)
        return self._orchestrator

    def initialise_models(self) -> None:
        """Load deep learning models asynchronously in background threads."""
        def _load_all_fast():
            try:
                try:
                    from ai.restoration.lama_restorer import LamaRestorer
                    self._lama = LamaRestorer()
                    self._lama.load(self._device)
                except Exception as exc:
                    logger.warning("LaMa load failed: %s", exc)
                    self._lama = None

                try:
                    self.face_restorer.load(self._device)
                except Exception as exc:
                    logger.warning("GFPGAN load failed: %s", exc)

                try:
                    self.super_resolution.load(self._device)
                except Exception as exc:
                    logger.warning("RealESRGAN load failed: %s", exc)

                self.inpainting_engine = InpaintingEngine(self._lama)
                logger.info("[OK] Fast pipeline models loaded on %s.", self._device)
            finally:
                self._models_ready.set()

        def _load_colorizer():
            try:
                self.colorizer.load(self._device)
            finally:
                self._colorizer_ready.set()

        threading.Thread(target=_load_all_fast, name="fast-model-loader", daemon=True).start()
        threading.Thread(target=_load_colorizer, name="colorizer-loader", daemon=True).start()

    @staticmethod
    def _should_colorize(enable_colorization: bool, color_type: str) -> bool:
        """Return whether colorization should run based on user flag and image color type."""
        return enable_colorization and (color_type in ("grayscale", "sepia"))

    def health_status(self) -> Dict[str, Any]:
        """Check status of pipeline models."""
        defect_ready = (
            getattr(self.enhanced_detector, "defect_segmenter", None) is not None
            and getattr(self.enhanced_detector.defect_segmenter, "is_loaded", False)
        )
        return {
            "status": "ready" if self._models_ready.is_set() else "loading",
            "models": {
                "defect_segmenter": defect_ready,
                "lama": self._lama is not None and getattr(self._lama, "is_loaded", False),
                "gfpgan": getattr(self.face_restorer, "gfpganer", None) is not None,
                "realesrgan": getattr(self.super_resolution, "realesrgan", None) is not None or self.super_resolution.is_loaded,
                "ddcolor": self.colorizer.is_loaded,
            },
            "device": self._device,
        }

    def is_colorizer_ready(self) -> bool:
        """Return whether DDColor colorization model is fully loaded."""
        return self._colorizer_ready.is_set() and self.colorizer.is_loaded

    def is_ready(self) -> bool:
        """Return whether fast model set is loaded."""
        return self._models_ready.is_set()

    def _wait_for_models(self, timeout: float = 300.0) -> bool:
        """Wait for background model loading threads to complete."""
        return self._models_ready.wait(timeout=timeout)

    # ── Stage Implementations (0 to 10b) ──────────────────────────────────────

    def _stage_0_despeckle(self, context: PipelineContext, on_progress: Optional[Callable]) -> None:
        """Stage 0: Impulse and speck suppression, before anything measures the image.

        This stage exists because of an ordering defect, not because the pipeline
        was short a filter. Noise reduction used to happen at Stage 4b, which is
        after Stage 2 builds the damage mask and after Stage 3 inpaints it. An
        isolated bright pixel is indistinguishable from an emulsion flake and an
        isolated dark pixel from the start of a crack, so every speck reached the
        detectors as damage. TearmarkDetector's MORPH_CLOSE then merged the speck
        scatter into large connected components, MaskRefiner routed those
        low-solidity blobs to LaMa, and LaMa repainted them — which is the
        smeared, crackled, pastel-blotched output that colorization then invents
        hues for.

        Measured on a clean 256x256 synthetic control, final mask coverage and
        component count against added impulse density:

            density   RAW coverage / components      after Stage 0
            0.000     10.05%  /  5                   9.62%  / 5
            0.005     96.28%  /  2                   9.99%  / 4
            0.010     99.62%  /  1                   9.71%  / 5
            0.080     99.99%  /  1                   9.83%  / 4

        Density-invariance is the property that matters: the detectors should
        report the same damage whether or not the scan happens to be speckled,
        because the damage is the same. That is also why this is Stage 0 rather
        than a tweak to the detectors' thresholds — no amount of threshold
        tuning makes a detector able to tell a 1px flake from a 1px speck, so
        the speck has to be gone before it is asked.

        Two complementary suppressors run here, covering opposite ends of the
        amplitude range:

          ImpulseDenoiser — discrete high-amplitude outliers (classic salt and
            pepper, scanner dust hits, film pinholes). Detects with a ROAD
            isolation test so single-pixel *content* survives, and rewrites only
            the pixels it identifies, leaving everything else bit-identical.

          SpeckSuppressor — a dense low-amplitude bright-speck field, the kind
            that reads as a texture rather than as discrete outliers and sits
            below the impulse detector's visibility floor. Gated on its own
            coverage and amplitude measurements, so it is a no-op unless a real
            field is present.

        Both record their measurements in image_metadata whether or not they act,
        so Stage 1's noise_score is measured on the cleaned image while later
        stages can still tell a cleaned image from one that was never speckled.
        """
        if on_progress:
            on_progress("despeckle", 3, "Suppressing impulse noise and specks...")

        context.start_timer("stage_0_despeckle")
        acted = []

        try:
            from ai.preprocessing.impulse_denoiser import ImpulseDenoiser
            if ImpulseDenoiser.suppress(context):
                acted.append("impulse")
        except Exception as exc:
            logger.warning("Stage 0 ImpulseDenoiser failed: %s", exc)
            context.add_log(f"Stage 0 ImpulseDenoiser failed: {exc}")

        try:
            from ai.preprocessing.speck_suppressor import SpeckSuppressor
            if SpeckSuppressor.suppress(context):
                acted.append("speck field")
        except Exception as exc:
            logger.warning("Stage 0 SpeckSuppressor failed: %s", exc)
            context.add_log(f"Stage 0 SpeckSuppressor failed: {exc}")

        context.end_timer("stage_0_despeckle")
        context.add_log(
            f"Stage 0 Despeckle: {', '.join(acted)} suppressed — damage detection "
            "will see a speck-free image."
            if acted else
            "Stage 0 Despeckle: nothing to suppress, image passed through unchanged."
        )
        DebugLogger.save_stage(context, "00b_after_despeckle")

    def _stage_1_analyze(self, context: PipelineContext, on_progress: Optional[Callable]) -> None:
        """Stage 1: Degradation Analysis."""
        if on_progress:
            on_progress("analyze", 10, "Analyzing image degradation profile...")
        metadata = QualityAnalyzer.analyze(context.working_image)
        context.image_metadata.update(metadata)
        context.add_log(
            f"Stage 1 Analyze: {metadata['width']}x{metadata['height']}, "
            f"color_type={metadata.get('color_type')}, blur={metadata.get('blur_score', 0):.1f}, "
            f"damage_est={metadata.get('damage_percentage', 0):.1f}%."
        )
        DebugLogger.save_stage(context, "01_after_analyze")

    def _stage_2_fracture_masking(self, context: PipelineContext, on_progress: Optional[Callable]) -> np.ndarray:
        """
        Stage 2: Multi-Scale Fracture Masking.

        Strategy (Fix 1 — U-Net Gatekeeper):
          When the CVPR DefectSegmenter U-Net is loaded and returns a focused
          mask (< 15% coverage), it is treated as authoritative for fine cracks
          and scratches. Classical detectors are limited to ultra-high-confidence
          white flakes only. This prevents hat/clothing textures from being
          misclassified as damage.

          When U-Net coverage >= 15% (model uncertain, heavy damage, or model
          not loaded), classical detectors run in full to catch wide tears.

        Strategy (Fix 2 — TearmarkDetector):
          For sepia and color photos, TearmarkDetector runs its 3-pass
          saturation-aware flake detector and adds emulsion flakes that the
          U-Net threshold may have missed.

        Strategy (MaskRefiner):
          MaskRefiner fuses all sub-masks, strips photo border artefacts from
          the tearmark mask, splits into LaMa-tier and NS-tier, and protects
          critical facial feature zones from being inpainted.
        """
        if on_progress:
            on_progress("damage_detect", 20, "Detecting fractures and structural tears...")
        context.start_timer("damage_detection")

        # ── Primary: U-Net semantic crack detection ──────────────────────────
        crack_mask = self.enhanced_detector.detect_thin_crack_network(
            context.working_image, context
        )
        unet_coverage = float(np.sum(crack_mask > 0) / max(crack_mask.size, 1) * 100)

        # Classical & structural detectors
        flake_mask = self.enhanced_detector.detect_white_flakes(
            context.working_image, context
        )
        wide_mask = self.enhanced_detector.detect_wide_cracks(
            context.working_image, context
        )
        scratch_mask = self.enhanced_detector.detect_scratches(
            context.working_image, context
        )
        tears_3d_mask = self.enhanced_detector.detect_3d_paper_tears(
            context.working_image, context
        )

        combined = cv2.bitwise_or(crack_mask, flake_mask)
        combined = cv2.bitwise_or(combined, wide_mask)
        combined = cv2.bitwise_or(combined, scratch_mask)
        combined = cv2.bitwise_or(combined, tears_3d_mask)

        # ── Grain-field (density-invariance) gate ─────────────────────────
        # A noisy scan is not a damaged photograph. When the robust content
        # noise floor is high and the structural damage estimate is low, the
        # mask above is grain misread as flakes/cracks — inpainting it is the
        # exact "oversmoothed Lenna" failure: LaMa repaints healthy pixels and
        # the real grain survives. Route the frame to the denoise path instead:
        # discard every noise-derived mask but keep genuine 3D structural tears
        # (missing paper corners are real damage at any noise level) and let
        # Stage 5 spatially-gated NLM clean the grain.
        # Measured on the SigShrink speckle-Lena figure (2026-09-21): global
        # statistics were diluted by its flat white margin (content_sigma
        # read 0.0 before the margin-robust estimator), detectors returned a
        # 4.93% false mask, and that mask reached LaMa with default flags.
        sigma_content = float(context.image_metadata.get("content_sigma", 0.0))
        damage_est = float(context.image_metadata.get("damage_percentage", 0.0))
        if sigma_content > 6.0 and damage_est < 6.0 and np.any(combined > 0):
            combined = cv2.bitwise_and(combined, tears_3d_mask)
            context.add_log(
                f"Stage 2 GrainField gate: content sigma={sigma_content:.1f} > 6.0 "
                f"with damage_est={damage_est:.1f}% < 6.0 — mask is grain, not damage. "
                "Discarding noise-derived masks (kept 3D structural tears only); "
                "denoise path will handle the grain."
            )
            context.image_metadata["grain_field_mask_discarded"] = True

        # ── TearmarkDetector for sepia/color emulsion flakes ─────────
        color_type = context.image_metadata.get("color_type", "color")
        if color_type in ("sepia", "color"):
            try:
                tm_mask = self.tearmark_detector.detect(context)
                context.damage_masks["tearmark"] = tm_mask
                tm_coverage = float(np.sum(tm_mask > 0) / max(tm_mask.size, 1) * 100)
                context.add_log(
                    f"Stage 2 TearmarkDetector: {tm_coverage:.2f}% emulsion flakes detected."
                )
            except Exception as exc:
                context.add_log(f"Stage 2 TearmarkDetector failed: {exc}")

        # ── MaskRefiner: fuse, strip borders, protect face zones ─────────────
        try:
            combined = MaskRefiner.refine(combined, context)
            context.add_log("Stage 2 MaskRefiner: mask fused and refined.")
        except Exception as exc:
            context.add_log(f"Stage 2 MaskRefiner failed: {exc}")

        context.damage_masks["final"] = combined
        combined_coverage = float(np.sum(combined > 0) / max(combined.size, 1) * 100)
        context.add_log(f"Stage 2 Final mask coverage: {combined_coverage:.2f}%")

        context.end_timer("damage_detection")
        DebugLogger.save_stage(context, "02_after_damage_detect")
        return combined

    def _stage_3_lama_inpaint(self, context: PipelineContext, mask: np.ndarray, on_progress: Optional[Callable]) -> None:
        """Stage 3: LaMa Deep Inpainting with Boundary Seam Smoothing."""
        if not np.any(mask > 0):
            context.add_log("Stage 3 Inpainting: No structural damage detected; skipping.")
            return

        DebugLogger.save_stage(context, "02_before_inpaint")

        if on_progress:
            on_progress("inpaint", 30, "Inpainting structural damage with LaMa...")
        context.start_timer("inpainting")

        try:
            pre_inpaint_canvas = context.working_image.copy()
            faces = context.image_metadata.get("faces", [])
            context.damage_masks["final"] = mask
            if self._lama is not None and getattr(self._lama, "model", None) is not None:
                inpainted = self._lama.restore(context)
                context.working_image = inpainted
                context.add_log("Stage 3 Inpainting: LaMa deep infill completed.")
            elif self.inpainting_engine is not None:
                inpainted = self.inpainting_engine.inpaint(context)
                context.working_image = inpainted
                context.add_log("Stage 3 Inpainting: InpaintingEngine completed.")
            else:
                inpainted = cv2.inpaint(context.working_image, mask, 3, cv2.INPAINT_TELEA)
                context.working_image = inpainted
                context.add_log("Stage 3 Inpainting: Classical Telea fallback completed.")

            # Seam boundary smoothing along inpaint borders to prevent hard edge artifacts
            k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            boundary_zone = cv2.dilate(mask, k_erode) - cv2.erode(mask, k_erode)

            # Protect critical facial anatomy (eyes, mouth) from seam boundary blur
            if faces:
                for f in faces:
                    bbox = f.get("bbox")
                    if not bbox or len(bbox) < 4:
                        continue
                    x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
                    fw, fh = max(1, x1 - x0), max(1, y1 - y0)
                    landmarks = f.get("landmarks")
                    if landmarks is not None and len(landmarks) >= 5:
                        for idx_eye in (0, 1):
                            ex, ey = int(round(landmarks[idx_eye][0])), int(round(landmarks[idx_eye][1]))
                            cv2.circle(boundary_zone, (ex, ey), max(10, int(min(fw, fh) * 0.12)), 0, -1)
                        mx = int(round((landmarks[3][0] + landmarks[4][0]) / 2.0))
                        my = int(round((landmarks[3][1] + landmarks[4][1]) / 2.0))
                        cv2.circle(boundary_zone, (mx, my), max(12, int(min(fw, fh) * 0.16)), 0, -1)
                    else:
                        ex1, ey1 = int(x0 + 0.32 * fw), int(y0 + 0.38 * fh)
                        ex2, ey2 = int(x0 + 0.68 * fw), int(y0 + 0.38 * fh)
                        r_eye = max(10, int(min(fw, fh) * 0.12))
                        cv2.circle(boundary_zone, (ex1, ey1), r_eye, 0, -1)
                        cv2.circle(boundary_zone, (ex2, ey2), r_eye, 0, -1)
                        mx, my = int(x0 + 0.50 * fw), int(y0 + 0.76 * fh)
                        cv2.circle(boundary_zone, (mx, my), max(12, int(min(fw, fh) * 0.16)), 0, -1)

            seam_f32 = boundary_zone.astype(np.float32) / 255.0
            seam_f32 = cv2.GaussianBlur(seam_f32, (9, 9), 0)[:, :, None]
            blurred = cv2.GaussianBlur(context.working_image, (7, 7), 0)
            context.working_image = np.clip(
                context.working_image.astype(np.float32) * (1.0 - seam_f32) +
                blurred.astype(np.float32) * seam_f32, 0, 255
            ).astype(np.uint8)
            context.add_log("Stage 3 Inpainting: Seam boundary smoothing applied.")

            # ── Structural Circuit Breaker on Face Regions ────────────────────
            # If inpainting wiped out critical anatomical structures (eyes, nose, contours),
            # detect the catastrophic loss in high-frequency edge energy and revert that face
            # region back to the pre-inpaint state before downstream colorization & GFPGAN.
            if faces:
                h_img, w_img = context.working_image.shape[:2]
                for idx, f in enumerate(faces):
                    bbox = f.get("bbox")
                    if not bbox or len(bbox) < 4:
                        continue
                    x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
                    x0 = max(0, x0); y0 = max(0, y0)
                    x1 = min(w_img, x1); y1 = min(h_img, y1)
                    fw = x1 - x0; fh = y1 - y0
                    if fw < 12 or fh < 12:
                        continue

                    face_mask_roi = mask[y0:y1, x0:x1]
                    face_damage_ratio = float(np.sum(face_mask_roi > 0)) / float(fw * fh)

                    # Only evaluate circuit breaker if pre-inpaint damage was not dense (ratio <= 0.20).
                    # In heavily flaked/cracked faces (ratio > 0.20), inpainting naturally removes massive
                    # flake edge variance while successfully synthesizing facial contours.
                    if 0.03 < face_damage_ratio <= 0.20:
                        crop_pre = pre_inpaint_canvas[y0:y1, x0:x1]
                        crop_post = context.working_image[y0:y1, x0:x1]
                        gray_pre = cv2.cvtColor(crop_pre, cv2.COLOR_RGB2GRAY).astype(np.float32)
                        gray_post = cv2.cvtColor(crop_post, cv2.COLOR_RGB2GRAY).astype(np.float32)

                        grad_pre = float(np.mean(cv2.Sobel(gray_pre, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(gray_pre, cv2.CV_32F, 0, 1)**2))
                        grad_post = float(np.mean(cv2.Sobel(gray_post, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(gray_post, cv2.CV_32F, 0, 1)**2))

                        # Only trigger if post-inpaint edge energy drops catastrophically to near-zero (wipeout)
                        # while pre-inpaint face had meaningful edge structure.
                        if grad_pre > 50.0 and grad_post < 35.0 and (grad_post / max(grad_pre, 1.0)) < 0.05:
                            context.working_image[y0:y1, x0:x1] = crop_pre
                            context.add_log(
                                f"[WARN] Circuit Breaker: Face {idx+1} edge energy dropped catastrophically "
                                f"({grad_pre:.1f} -> {grad_post:.1f} < 35.0). "
                                f"Reverted face region to pre-inpaint state to preserve facial structure."
                            )
                            warnings_list = context.image_metadata.setdefault("warnings", [])
                            if "inpaint_face_structure_erased_reverted" not in warnings_list:
                                warnings_list.append("inpaint_face_structure_erased_reverted")

            # Post-inpaint residual damage percentage update:
            # Once LaMa successfully infills damage, residual structural damage is minimal (< 2%).
            # This ensures downstream DDColor colorization evaluates the cleaned canvas rather than
            # tripping against pre-repair damage figures.
            context.image_metadata["damage_percentage"] = 0.5
            context.add_log("Stage 3 Inpainting: residual damage_percentage updated to 0.5% for downstream stages.")
        except Exception as exc:
            context.add_log(f"Stage 3 Inpainting failed: {exc}")

        context.end_timer("inpainting")
        DebugLogger.save_stage(context, "03_after_inpaint")

    def _stage_2b_damage_cluster_smooth(self, context: PipelineContext, on_progress: Optional[Callable] = None) -> None:
        """
        Stage 2b: High-Variance Damage Zone Detection & Targeted NLM Smoothing.
        """
        if on_progress:
            on_progress("damage_cluster_smooth", 35, "Smoothing high-variance damage clusters...")
        context.start_timer("damage_cluster_smooth")

        try:
            img = context.working_image
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            h_img, w_img = gray.shape

            # Measure local variance using 15x15 sliding window
            mean = cv2.GaussianBlur(gray, (15, 15), 0)
            mean_sq = cv2.GaussianBlur(gray ** 2, (15, 15), 0)
            local_variance = np.sqrt(np.maximum(mean_sq - mean ** 2, 0.0))

            damage_zone = (local_variance > 35.0).astype(np.float32)

            # Continuous feathered face safe zone
            safe_mask = np.zeros((h_img, w_img), dtype=np.float32)
            cy, cx = h_img // 2, w_img // 2
            safe_y0 = int(cy - 0.25 * h_img)
            safe_y1 = int(cy + 0.25 * h_img)
            safe_x0 = int(cx - 0.25 * w_img)
            safe_x1 = int(cx + 0.25 * w_img)
            safe_mask[max(0, safe_y0):min(h_img, safe_y1), max(0, safe_x0):min(w_img, safe_x1)] = 1.0

            faces = context.image_metadata.get("faces", [])
            if faces:
                for f in faces:
                    bbox = f.get("bbox")
                    if bbox and len(bbox) >= 4:
                        x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
                        safe_mask[max(0, y0):min(h_img, y1), max(0, x0):min(w_img, x1)] = 1.0

            safe_mask = cv2.GaussianBlur(safe_mask, (25, 25), 0)
            damage_zone = damage_zone * (1.0 - np.clip(safe_mask, 0.0, 1.0))
            damage_zone = cv2.GaussianBlur(damage_zone, (21, 21), 0)

            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            l_heavy = cv2.fastNlMeansDenoising(l_chan, None, h=10.0, templateWindowSize=7, searchWindowSize=23)

            l_final = (l_heavy.astype(np.float32) * damage_zone + l_chan.astype(np.float32) * (1.0 - damage_zone))
            lab_fixed = cv2.merge([np.clip(l_final, 0, 255).astype(np.uint8), a_chan, b_chan])
            context.working_image = cv2.cvtColor(lab_fixed, cv2.COLOR_LAB2RGB)
            context.add_log("Stage 2b DamageClusterSmooth: High-variance damage smoothed (variance>35.0, h=10.0).")
        except Exception as exc:
            context.add_log(f"Stage 2b DamageClusterSmooth failed: {exc}")

        context.end_timer("damage_cluster_smooth")
        DebugLogger.save_stage(context, "02b_after_damage_cluster_smooth")

    def _stage_4_face_detect(self, context: PipelineContext, on_progress: Optional[Callable]) -> List[Dict[str, Any]]:
        """Stage 4: Face Detection (Run early to inform damage masks and protection zones)."""
        if on_progress:
            on_progress("face_detect", 12, "Detecting faces and facial landmarks...")
        context.start_timer("face_detection")
        self.face_detector.detect(context)
        context.end_timer("face_detection")
        faces = context.image_metadata.get("faces", [])
        context.add_log(f"FaceDetect: detected {len(faces)} face(s).")
        DebugLogger.save_stage(context, "04_after_face_detect")
        return faces

    def _stage_3b_refresh_faces(
        self,
        context: PipelineContext,
        previous_faces: List[Dict[str, Any]],
        on_progress: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Stage 3b: Post-Inpaint Face Re-Detection.
        
        When large cracks or tears cut across a face, pre-inpaint detection can
        fail to identify facial landmarks. Once LaMa reconstructs the torn canvas,
        re-detecting faces recovers missing faces and updates facial geometry for
        accurate 3-zone denoising, skin tone protection, and GFPGAN face restoration.
        """
        damage_mask = context.damage_masks.get("final")
        has_repaired_damage = damage_mask is not None and np.any(damage_mask > 0)

        # Re-detect if no faces were found initially or if damage was repaired
        if not previous_faces or has_repaired_damage:
            try:
                self.face_detector.detect(context)
                refreshed_faces = context.image_metadata.get("faces", [])
                if refreshed_faces and not previous_faces:
                    context.add_log(
                        f"Stage 3b Post-Inpaint FaceDetect: successfully recovered {len(refreshed_faces)} face(s) after inpainting tears."
                    )
                    return refreshed_faces
                elif refreshed_faces:
                    context.add_log(
                        f"Stage 3b Post-Inpaint FaceDetect: refreshed {len(refreshed_faces)} face bounds/landmarks on repaired canvas."
                    )
                    return refreshed_faces
            except Exception as exc:
                context.add_log(f"Stage 3b Post-Inpaint FaceDetect failed: {exc}")

        return previous_faces

    def _stage_5_spatially_gated_denoise(self, context: PipelineContext, faces: List[Dict[str, Any]], on_progress: Optional[Callable]) -> None:
        """
        Stage 5: Spatially-Gated 3-Zone Denoising.
        - Face zone: Gentle NLM (h ≈ 0.95σ, capped 10.5) preserving skin texture.
        - Subject zone: Balanced NLM (h ≈ 1.0σ).
        - Background (undamaged, high σ): Moderate NLM (h = 0.85σ, capped 11.5) — preserves photographic grain.
        - Background (damaged or low σ): Aggressive NLM (h = 1.4σ) for deep grain/scratch suppression.
        """
        DebugLogger.save_stage(context, "04b_before_denoise")

        if on_progress:
            on_progress("denoise", 50, "Applying spatially-gated 3-zone denoising...")
        context.start_timer("denoising")

        try:
            h_img, w_img = context.working_image.shape[:2]
            face_mask_2d = np.zeros((h_img, w_img), dtype=np.float32)

            for f in faces:
                bbox = f.get("bbox")
                if bbox and len(bbox) >= 4:
                    x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
                    cv2.rectangle(face_mask_2d, (x0, y0), (x1, y1), 1.0, -1)
                landmarks = f.get("landmarks")
                if landmarks is not None and len(landmarks) >= 3:
                    pts = np.array(landmarks, dtype=np.int32)
                    hull = cv2.convexHull(pts)
                    cv2.fillConvexPoly(face_mask_2d, hull, 1.0)

            if np.any(face_mask_2d > 0):
                k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
                face_mask_2d = cv2.dilate(face_mask_2d, k_dil, iterations=1)
                face_mask_2d = cv2.GaussianBlur(face_mask_2d, (21, 21), 0)

            mask_face = np.clip(face_mask_2d, 0.0, 1.0)

            # Mid-zone mask: dilated subject zone around face
            if np.any(mask_face > 0):
                k_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (75, 75))
                mid_zone = cv2.dilate((mask_face > 0.1).astype(np.uint8), k_mid)
                mid_zone_f = cv2.GaussianBlur(mid_zone.astype(np.float32), (35, 35), 0)
                mid_mask = np.clip(mid_zone_f * (1.0 - mask_face), 0.0, 1.0)
            else:
                mid_mask = np.full((h_img, w_img), 0.4, dtype=np.float32)

            bg_mask = np.clip(1.0 - mask_face - mid_mask, 0.0, 1.0)

            # Denoise strictly on L* channel in CIELAB space
            lab = cv2.cvtColor(context.working_image, cv2.COLOR_RGB2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)

            # Measure high-frequency noise floor sigma via MAD on L* channel.
            # On images with large flat margins the global MAD collapses to 0
            # (the margin is the modal residual), so blend in Stage 1's
            # content-only sigma as a floor — measured on the photo, not the
            # scanner bed. Capped so a stale estimate can never over-drive.
            med = cv2.medianBlur(l_chan, 3).astype(np.float32)
            mad = float(np.median(np.abs(l_chan.astype(np.float32) - med)))
            sigma_global = 1.4826 * mad
            sigma_meta = float(context.image_metadata.get("content_sigma", 0.0))
            sigma_noise = max(sigma_global, min(sigma_meta, 20.0)) if sigma_meta > 0 else sigma_global

            # Check if face is small/low-res or deferred
            is_small_face = any(
                min(f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1]) < 90
                for f in faces if "bbox" in f and len(f["bbox"]) >= 4
            ) or min(h_img, w_img) < 320 or context.pipeline_flags.get("face_restore_deferred_to_post_sr", False)

            # Check damage mask coverage in background area
            final_mask = context.damage_masks.get("final")
            bg_damaged = False
            if final_mask is not None and np.any(final_mask > 0):
                bg_pix = bg_mask > 0.5
                if np.any(bg_pix):
                    bg_cov = float(np.sum(final_mask[bg_pix] > 0)) / max(1.0, float(np.sum(bg_pix)))
                    if bg_cov > 0.02:
                        bg_damaged = True

            # Adaptive filter strengths: scale h proportionally to measured noise floor.
            # bg_damaged drives aggressive smoothing; high sigma alone does NOT — we want
            # photographic grain preserved when the background has no damage (e.g. noisy Lena).
            if bg_damaged:
                h_bg = max(11.0, 1.4 * sigma_noise)
            elif sigma_noise > 7.0:
                h_bg = max(11.5, 1.25 * sigma_noise)
            else:
                h_bg = max(6.5, 0.95 * sigma_noise)
            h_mid = max(7.5, 1.10 * sigma_noise)
            # Face zone: cap at 1.10x sigma regardless of small-face flag to retain skin texture.
            if is_small_face and sigma_noise > 4.0:
                h_face = min(max(7.0, 1.10 * sigma_noise), 12.0)
            else:
                h_face = max(5.5, 0.95 * sigma_noise)

            # 1. Background: Substantial NLM
            l_bg = cv2.fastNlMeansDenoising(l_chan, None, h=h_bg, templateWindowSize=7, searchWindowSize=21)
            # 2. Mid-zone (subject clothing/hair): Balanced NLM
            l_mid = cv2.fastNlMeansDenoising(l_chan, None, h=h_mid, templateWindowSize=7, searchWindowSize=21)
            # 3. Face: Adaptive NLM preserving anatomy or smoothing noise
            l_face = cv2.fastNlMeansDenoising(l_chan, None, h=h_face, templateWindowSize=5, searchWindowSize=15)

            # Compute luminance edge energy using pre-smoothed L-channel so high-frequency noise spikes
            # do not register as structural edges.
            l_smooth = cv2.GaussianBlur(l_chan, (3, 3), 0)
            gx = cv2.Sobel(l_smooth, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(l_smooth, cv2.CV_32F, 0, 1, ksize=3)
            edge_mag = np.sqrt(gx ** 2 + gy ** 2)

            local_mean = cv2.blur(l_smooth.astype(np.float32), (7, 7))
            local_sq = cv2.blur(l_smooth.astype(np.float32) ** 2, (7, 7))
            local_var = np.maximum(0.0, local_sq - local_mean ** 2)

            # Noise-aware edge and texture protection:
            # Detects genuine anatomical boundaries (chin shadow, shoulder, jawline, hat brim).
            edge_thresh = max(35.0, 2.5 * sigma_noise)
            edge_weight = np.clip((edge_mag - 1.5 * sigma_noise) / edge_thresh, 0.0, 1.0)

            noise_floor_var = max(16.0, sigma_noise ** 2)
            raw_texture = np.clip((local_var - 2.0 * noise_floor_var) / (2.5 * noise_floor_var), 0.0, 1.0)
            texture_weight = raw_texture * max(0.0, 1.0 - sigma_noise / 15.0)

            protect_map = np.maximum(edge_weight, texture_weight)

            l_fused = (
                l_face.astype(np.float32) * mask_face +
                l_mid.astype(np.float32) * mid_mask +
                l_bg.astype(np.float32) * bg_mask
            )
            # Retain authentic edge sharpness: blend with edge-preserving reference rather than raw noisy pixels.
            # If image has substantial noise, use median-filtered L to avoid re-injecting single-pixel noise spikes.
            if sigma_noise > 4.0:
                l_edge_ref = cv2.medianBlur(l_chan, 3).astype(np.float32)
                max_protect = max(0.20, 0.40 - 0.015 * sigma_noise)
            else:
                l_edge_ref = l_chan.astype(np.float32)
                max_protect = 0.50

            l_fused = l_fused * (1.0 - max_protect * protect_map) + l_edge_ref * (max_protect * protect_map)
            l_fused_uint8 = np.clip(l_fused, 0, 255).astype(np.uint8)

            lab_fused = cv2.merge([l_fused_uint8, a_chan, b_chan])
            context.working_image = cv2.cvtColor(lab_fused, cv2.COLOR_LAB2RGB)

            med_after = cv2.medianBlur(l_fused_uint8, 3).astype(np.float32)
            mad_after = float(np.median(np.abs(l_fused_uint8.astype(np.float32) - med_after)))
            sigma_after = 1.4826 * mad_after

            context.add_log(
                f"Stage 5 Denoise: Spatially-gated adaptive NLM applied "
                f"(sigma={sigma_noise:.2f} -> {sigma_after:.2f}; "
                f"h_face={h_face:.1f}, h_mid={h_mid:.1f}, h_bg={h_bg:.1f})."
            )
        except Exception as exc:
            context.add_log(f"Stage 5 Denoise failed: {exc}")

        context.end_timer("denoising")
        DebugLogger.save_stage(context, "05_after_denoise")
        _refresh_sharpness(context)

    def _stage_6_colorization(self, context: PipelineContext, on_progress: Optional[Callable]) -> None:
        """Stage 6: DDColor Colorization with Guided Chrominance Diffusion."""
        color_type = context.image_metadata.get("color_type", "color")
        if color_type not in ("grayscale", "sepia"):
            context.add_log("Stage 6 Colorization: Image is already color; skipping.")
            return

        DebugLogger.save_stage(context, "05c_before_colorize")

        if on_progress:
            on_progress("colorize", 65, "Colorizing with DDColor (ICCV 2023)...")
        context.start_timer("colorization")

        try:
            if not self._colorizer_ready.is_set():
                self._colorizer_ready.wait(timeout=60.0)
            if self.colorizer.is_loaded:
                self.colorizer.colorize(context)
                context.add_log("Stage 6 Colorization: DDColor completed.")

                # ── Seamless chrominance propagation in inpainted tear/crack zones ──
                # DDColor can hallucinate vivid or inconsistent chromatic tones inside
                # infilled regions. Instead of clamping to dead gray (a*=128, b*=128),
                # seamlessly diffuse surrounding ambient skin/hair/background chrominance
                # across the inpaint boundary so color transitions are continuous and natural.
                final_mask = context.damage_masks.get("final")
                if final_mask is not None and np.any(final_mask > 0):
                    try:
                        h_m, w_m = final_mask.shape[:2]
                        h_i, w_i = context.working_image.shape[:2]
                        if (h_m, w_m) != (h_i, w_i):
                            final_mask = cv2.resize(final_mask, (w_i, h_i), interpolation=cv2.INTER_NEAREST)
                        
                        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        inpaint_zone = cv2.dilate((final_mask > 0).astype(np.uint8) * 255, k_dil)
                        
                        lab = cv2.cvtColor(context.working_image, cv2.COLOR_RGB2LAB)
                        l_chan, a_chan, b_chan = cv2.split(lab)

                        # Dynamic outer border detection: detect near-white edge-connected margins
                        # (torn paper border strips) and protect them from chroma bleed
                        border_mask = np.zeros((h_i, w_i), dtype=np.uint8)
                        border_cand = (l_chan > 195).astype(np.uint8) * 255
                        num_b, lbl_b, stats_b, _ = cv2.connectedComponentsWithStats(border_cand, connectivity=8)
                        for bi in range(1, num_b):
                            bx = stats_b[bi, cv2.CC_STAT_LEFT]
                            bw = stats_b[bi, cv2.CC_STAT_WIDTH]
                            by = stats_b[bi, cv2.CC_STAT_TOP]
                            bh = stats_b[bi, cv2.CC_STAT_HEIGHT]
                            b_area = stats_b[bi, cv2.CC_STAT_AREA]
                            if (bx <= 2 or bx + bw >= w_i - 2) and bh > 0.30 * h_i and b_area > (h_i * 8):
                                border_mask[lbl_b == bi] = 255

                        if np.any(border_mask > 0):
                            # Exclude border regions from inpaint_zone so background chroma doesn't diffuse into paper
                            inpaint_zone[border_mask > 0] = 0

                        # Inpaint chrominance channels from healthy surrounding context
                        a_diffused = cv2.inpaint(a_chan, inpaint_zone, 5, cv2.INPAINT_TELEA)
                        b_diffused = cv2.inpaint(b_chan, inpaint_zone, 5, cv2.INPAINT_TELEA)
                        
                        # Structure-guided chrominance snapping:
                        # Use l_chan as structural guide so chrominance boundaries align with physical
                        # image edges (hair outline, hat brim, clothing, background) instead of bleeding arbitrarily
                        try:
                            a_diffused = cv2.ximgproc.guidedFilter(guide=l_chan, src=a_diffused, radius=6, eps=15.0)
                            b_diffused = cv2.ximgproc.guidedFilter(guide=l_chan, src=b_diffused, radius=6, eps=15.0)
                        except Exception:
                            pass

                        inpaint_zone_f = cv2.GaussianBlur(inpaint_zone.astype(np.float32) / 255.0, (11, 11), 0)
                        
                        a_fused = np.clip(
                            a_diffused.astype(np.float32) * inpaint_zone_f +
                            a_chan.astype(np.float32) * (1.0 - inpaint_zone_f),
                            0, 255
                        ).astype(np.uint8)
                        
                        b_fused = np.clip(
                            b_diffused.astype(np.float32) * inpaint_zone_f +
                            b_chan.astype(np.float32) * (1.0 - inpaint_zone_f),
                            0, 255
                        ).astype(np.uint8)

                        # Keep border paper chrominance strictly neutral (a*=128, b*=128)
                        if np.any(border_mask > 0):
                            a_fused[border_mask > 0] = 128
                            b_fused[border_mask > 0] = 128
                        
                        lab_fused = cv2.merge([l_chan, a_fused, b_fused])
                        context.working_image = cv2.cvtColor(lab_fused, cv2.COLOR_LAB2RGB)
                        context.add_log("Stage 6 ColorDiffusion: ambient skin/hair chrominance smoothly propagated into inpaint zones.")
                        if np.any(border_mask > 0):
                            context.add_log("Stage 6 ColorDiffusion: torn paper border margins protected from pink/warm chroma bleed.")
                    except Exception as _exc:
                        context.add_log(f"Stage 6 ColorDiffusion failed: {_exc}")
            else:
                context.add_log("Stage 6 Colorization: DDColor not loaded; skipping.")
        except Exception as exc:
            context.add_log(f"Stage 6 Colorization failed: {exc}")

        context.end_timer("colorization")
        DebugLogger.save_stage(context, "06_after_colorize")

    def _stage_7_face_restoration(self, context: PipelineContext, on_progress: Optional[Callable]) -> None:
        """Stage 7: GFPGAN Face Restoration (Delta_y=0, scales 0.55/0.65, 0% noise passback)."""
        faces = context.image_metadata.get("faces", [])
        if not faces:
            # Fallback re-scan on current image state
            try:
                self.face_detector.detect(context)
                faces = context.image_metadata.get("faces", [])
            except Exception:
                faces = []

        if not faces:
            context.add_log("Stage 7 FaceRestoration: No faces to restore; skipping.")
            return

        # Pre-SR Small Face / Low-Res Canvas Deferral Check:
        # RetinaFace landmark reliability drops sharply on faces < 90px or canvases < 320px.
        # If SR is enabled and face is small, defer to post-SR where 4x upscale makes landmarks robust.
        is_deferred_pass = context.pipeline_flags.get("_running_deferred_face_restore", False)
        if not is_deferred_pass and context.pipeline_flags.get("enable_sr", True):
            h_curr, w_curr = context.working_image.shape[:2]
            min_dim = min(h_curr, w_curr)
            face_sizes = [
                min(f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1])
                for f in faces if "bbox" in f and len(f["bbox"]) >= 4
            ]
            min_face_size = min(face_sizes) if face_sizes else 999
            if min_dim < 320 or min_face_size < 90:
                context.pipeline_flags["face_restore_deferred_to_post_sr"] = True
                context.add_log(
                    f"Stage 7 FaceRestoration: Face size ({min_face_size}px) or canvas ({min_dim}px) "
                    "below landmark reliability threshold (<90px / <320px). Deferring to post-SR."
                )
                return

        DebugLogger.save_stage(context, "06c_before_face_restore")

        if on_progress:
            on_progress("face_restore", 78, "Restoring facial details with GFPGAN...")
        context.start_timer("face_restoration")

        try:
            self.face_restorer.restore(context)
            context.add_log("Stage 7 FaceRestoration: GFPGAN completed.")
        except Exception as exc:
            context.add_log(f"Stage 7 FaceRestoration failed: {exc}")

        context.end_timer("face_restoration")
        DebugLogger.save_stage(context, "07_after_face_restore")

    def _stage_8_presr_antialiasing(self, context: PipelineContext, on_progress: Optional[Callable]) -> None:
        """Stage 8: Pre-SR Bilateral Anti-Aliasing (adaptive on L*, plus gentle chroma diffusion on noisy scans)."""
        DebugLogger.save_stage(context, "07b_before_pre_sr")

        if on_progress:
            on_progress("pre_sr", 85, "Applying pre-SR anti-aliasing filter...")

        try:
            lab = cv2.cvtColor(context.working_image, cv2.COLOR_RGB2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            
            noise_score = float(context.image_metadata.get("noise_score", 0.0))
            sigma_c = max(7.0, 1.0 * noise_score) if noise_score > 6.0 else 7.0
            l_smooth = cv2.bilateralFilter(l_chan, d=5, sigmaColor=sigma_c, sigmaSpace=3.0)

            # On noisy scans, also diffuse residual color grain on a* and b* before 4x Real-ESRGAN sharpening
            if noise_score > 8.0:
                a_smooth = cv2.bilateralFilter(a_chan, d=5, sigmaColor=12.0, sigmaSpace=3.0)
                b_smooth = cv2.bilateralFilter(b_chan, d=5, sigmaColor=12.0, sigmaSpace=3.0)
            else:
                a_smooth, b_smooth = a_chan, b_chan

            lab_clean = cv2.merge([l_smooth, a_smooth, b_smooth])
            context.working_image = cv2.cvtColor(lab_clean, cv2.COLOR_LAB2RGB)
            context.add_log(f"Stage 8 Pre-SR Anti-Aliasing: Bilateral filter applied (sigmaColor={sigma_c:.1f}).")
        except Exception as exc:
            context.add_log(f"Stage 8 Pre-SR Anti-Aliasing failed: {exc}")

        DebugLogger.save_stage(context, "08_after_pre_sr")

    def _stage_9_super_resolution(self, context: PipelineContext, on_progress: Optional[Callable]) -> None:
        """
        Stage 9: Real-ESRGAN x4+ Super-Resolution.

        Fix 9: Delegates to SuperResolution.restore(context) instead of calling
        _realesrgan_tile_forward() directly. SuperResolution.restore() provides:
          - OOM safety cap (output capped at 4096×4096)
          - Pre-SR halftone/mesh suppression for scanned prints
          - 2K output normalisation (longest side → 2048px)
          - Graceful bicubic fallback with correct aspect ratio
          - Post-SR LAB sharpening (strength=0.22)
        """
        DebugLogger.save_stage(context, "08b_before_super_res")

        if on_progress:
            on_progress("super_res", 90, "Upscaling with Real-ESRGAN via SuperResolution.restore()...")
        context.start_timer("super_resolution")

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.super_resolution.restore(context)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            h_out, w_out = context.working_image.shape[:2]
            context.add_log(
                f"Stage 9 SuperResolution: completed via SuperResolution.restore() — "
                f"output {w_out}x{h_out}."
            )
        except Exception as exc:
            context.add_log(
                f"Stage 9 SuperResolution failed ({exc}) — applying Lanczos 4x fallback."
            )
            h_in, w_in = context.working_image.shape[:2]
            context.working_image = cv2.resize(
                context.working_image,
                (w_in * 4, h_in * 4),
                interpolation=cv2.INTER_LANCZOS4,
            )

        context.end_timer("super_resolution")
        DebugLogger.save_stage(context, "09_after_super_res")

    def _stage_10_post_sr_sharpen(self, context: PipelineContext, on_progress: Optional[Callable] = None) -> None:
        """
        Stage 10: Post-SR Luminance Crispness Calibration.
        
        Applies edge-directed multi-scale sharpening strictly in LAB luminance (L*) space
        to enhance internal borders (facial contours, jawline, eyes, lips, hair borders,
        hat brims, cloth folds, silhouette boundaries) without amplifying flat background noise
        or creating chromatic halos.
        """
        if on_progress:
            on_progress("sharpen", 95, "Enhancing internal borders and contours...")
        context.start_timer("post_sr_sharpen")

        try:
            img = context.working_image
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
            l_chan = lab[:, :, 0]
            a_chan = lab[:, :, 1]
            b_chan = lab[:, :, 2]

            # 1. Edge & Contour Detection via Sobel Gradient Magnitude
            gx = cv2.Sobel(l_chan, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(l_chan, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx ** 2 + gy ** 2)

            # Smooth edge gate: 0.0 in smooth areas, ramping up to 1.0 along defined edges/borders
            edge_weight = np.clip((grad_mag - 12.0) / 45.0, 0.0, 1.0)
            # Soft-clamp extreme contrast transitions to avoid ringing halos
            edge_weight = edge_weight * np.clip(1.0 - (grad_mag - 130.0) / 100.0, 0.35, 1.0)

            # 2. Multi-Scale Luminance Unsharp Masking
            # Fine scale (sigma=1.2): recovers micro-textures (hair strands, fabric weave, eyelashes)
            diff_micro = l_chan - cv2.GaussianBlur(l_chan, (0, 0), sigmaX=1.2)
            # Contour / border scale (sigma=3.2): specifically sharpens 4-8px object borders & facial contours
            diff_contour = l_chan - cv2.GaussianBlur(l_chan, (0, 0), sigmaX=3.2)

            # Combine: gentle micro-detail everywhere + pronounced crispness on internal borders
            l_crisp = l_chan + 0.25 * diff_micro + 0.45 * edge_weight * diff_contour
            l_crisp = np.clip(l_crisp, 0, 255)

            lab[:, :, 0] = l_crisp
            context.working_image = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
            context.add_log(
                "Stage 10 Sharpen: Edge-directed multi-scale crispness applied "
                "(micro=0.25, border_contour=0.45 on L* channel)."
            )
        except Exception as exc:
            context.add_log(f"Stage 10 Sharpen failed: {exc}")

        context.end_timer("post_sr_sharpen")
        DebugLogger.save_stage(context, "10_after_sharpen")

    def _stage_10b_projector_grade(self, context: PipelineContext, on_progress: Optional[Callable] = None) -> None:
        """Stage 10b: Projector Display Grading (CLAHE, shadow-lifted S-curve, and calibrated tonal grading)."""
        DebugLogger.save_stage(context, "10_before_projector_grade")

        if on_progress:
            on_progress("projector_grade", 98, "Applying projector exhibition grading...")
        context.start_timer("projector_grade")

        try:
            img = context.working_image
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # Step 1: Calibrated CLAHE on L* with tamed clipLimit and larger tile grid
            # Prevents local-contrast halos and the 'glossy embossed/varnish' look on skin/fabric
            h_img, w_img = img.shape[:2]
            grid_size = max(16, min(32, w_img // 128))
            clahe = cv2.createCLAHE(clipLimit=0.85, tileGridSize=(grid_size, grid_size))
            l_clahe = clahe.apply(l)

            # Step 2: Smooth natural filmic tone curve (gentle shadow lift, neutral midtones)
            lut = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                x = i / 255.0
                # Filmic exhibition response: lifts true blacks slightly (0.01) with linear-preserving gamma
                y = 0.01 + 0.99 * (x ** 0.96)
                lut[i] = int(np.clip(y * 255, 0, 255))
            l_curved = cv2.LUT(l_clahe, lut)

            # Step 3: Merge calibrated tonal response
            lab_graded = cv2.merge([l_curved, a, b])
            context.working_image = cv2.cvtColor(lab_graded, cv2.COLOR_LAB2RGB)
            context.add_log("Stage 10b ProjectorGrade: Tamed CLAHE (clip=0.85, grid=16-32) and natural film curve applied.")
        except Exception as exc:
            context.add_log(f"Stage 10b ProjectorGrade failed: {exc}")

        context.end_timer("projector_grade")
        DebugLogger.save_stage(context, "10b_after_projector_grade")

    # ── Main Restore Entry Point ──────────────────────────────────────────────

    def restore(
        self,
        image_input: Any,
        external_metadata: Optional[Dict[str, Any]] = None,
        enable_repair: bool = True,
        enable_denoise: bool = True,
        enable_colorization: bool = True,
        enable_face_restore: bool = True,
        enable_sr: bool = True,
        on_progress: Optional[Callable[[str, int, str], None]] = None,
        use_orchestrator: Optional[bool] = None,
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
            on_progress: Progress callback (stage_name, percent, message)
            use_orchestrator: Optional bool to execute via Two-Tier PPCAV PhotoOrchestrator
            
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
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Auto-detect and crop scanner framing margins / solid canvas borders
        img_rgb, has_border, border_bbox = ImageNormalizer.detect_and_crop_borders(img_rgb)

        meta = dict(external_metadata) if external_metadata else {}
        if has_border:
            meta["border_cropped"] = True
            meta["border_bbox"] = border_bbox

        context = PipelineContext(img_rgb, meta)
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
        })
        self.last_context = context

        _should_use_orchestrator = (
            use_orchestrator
            if use_orchestrator is not None
            else os.getenv("PIXREVIVE_USE_ORCHESTRATOR", "0").lower() in ("1", "true")
        )
        if _should_use_orchestrator:
            context.pipeline_flags["use_orchestrator"] = True
            return self.orchestrator.run(
                context,
                progress_callback=on_progress,
                enable_repair=enable_repair,
                enable_denoise=enable_denoise,
                enable_colorization=enable_colorization,
                enable_face_restore=enable_face_restore,
                enable_sr=enable_sr,
            )

        DebugLogger.save_stage(context, "00_input_raw")

        # ── Stage 0: Impulse / Speck Suppression ──────────────────────────────
        if enable_denoise:
            self._stage_0_despeckle(context, on_progress)

        # ── Stage 1: Degradation Analysis ─────────────────────────────────────
        self._stage_1_analyze(context, on_progress)

        # ── Stage 1b / Early Face Detection ───────────────────────────────────
        # Runs early on the input canvas so faces and landmarks are available to
        # MaskRefiner (Stage 2) and DamageClusterSmooth (Stage 2b), preventing
        # destructive inpainting of facial anatomy.
        faces = self._stage_4_face_detect(context, on_progress)

        # ── PipelineRouter — advisory routing logged for debugging ────────────
        route = self.router.route(context)
        context.add_log(
            f"PipelineRouter: '{route['pipeline_name']}' — {route['description']}"
        )

        # ── Stage 2: Multi-Scale Fracture Masking ─────────────────────────────
        damage_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        if enable_repair:
            damage_mask = self._stage_2_fracture_masking(context, on_progress)

        # ── Stage 3: LaMa Deep Inpainting + Seam Boundary Smoothing ───────────
        if enable_repair and np.any(damage_mask > 0):
            self._stage_3_lama_inpaint(context, damage_mask, on_progress)

        # ── Stage 2b: High-Variance Damage Zone Smoother ──────────────────────
        if enable_denoise:
            noise_score = float(context.image_metadata.get("noise_score", 0.0))
            inpaint_coverage = float(np.sum(damage_mask > 0) / max(damage_mask.size, 1) * 100) if damage_mask is not None else 0.0
            if noise_score > 3.0 and inpaint_coverage < 30.0:
                self._stage_2b_damage_cluster_smooth(context, on_progress)
            elif noise_score <= 3.0:
                context.add_log(
                    f"Stage 2b: noise_score={noise_score:.2f} <= 3.0 — "
                    "skipping (clean image, no grain clusters to smooth)."
                )
            else:
                context.add_log(
                    f"Stage 2b: inpaint_coverage={inpaint_coverage:.1f}% >= 30% — "
                    "skipping to preserve LaMa inpaint result."
                )

        # ── Stage 3b: Post-Inpaint Face Re-Detection ──────────────────────────
        # If face detection was missed early due to large tears/occlusions,
        # or if structural inpainting repaired facial regions, re-scan the
        # inpainted canvas so faces and landmarks are accurately available
        # for Stage 5 (3-zone denoising), Stage 6 (colorization), and Stage 7 (GFPGAN).
        faces = self._stage_3b_refresh_faces(context, faces, on_progress)

        # ── Stage 5: Spatially-Gated 3-Zone Denoising ─────────────────────────
        if enable_denoise:
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

            self._stage_5_spatially_gated_denoise(context, faces, on_progress)

        # ── Stage 6: DDColor Colorization ─────────────────────────────────────
        if enable_colorization:
            self._stage_6_colorization(context, on_progress)

            # ── Stage 6c: Color Artifact Reduction ────────────────────────────
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

        # ── Stage 7: GFPGAN Face Restoration ──────────────────────────────────
        if enable_face_restore:
            self._stage_7_face_restoration(context, on_progress)

        # ── Stage 8: Pre-SR Bilateral Anti-Aliasing ───────────────────────────
        if enable_sr:
            noise_score = float(context.image_metadata.get("noise_score", 0.0))
            if noise_score > 6.0:
                self._stage_8_presr_antialiasing(context, on_progress)
            else:
                context.add_log("Stage 8 Pre-SR: noise_score <= 6.0 — preserving natural sharpness.")

        # ── Stage 9: Real-ESRGAN x4+ Super-Resolution ─────────────────────────
        if enable_sr:
            self._stage_9_super_resolution(context, on_progress)

            # Sync all masks and face bboxes to new SR resolution
            try:
                context.sync_geometry_to_working_image()
                context.add_log("Stage 9 post-SR: geometry synced to SR resolution.")
            except Exception as exc:
                context.add_log(f"Stage 9 sync_geometry failed: {exc}")

            # Harmonize inpaint seams at SR resolution
            if enable_repair and np.any(context.damage_masks.get("final", np.zeros(1)) > 0):
                try:
                    Harmonizer.harmonize(context)
                    context.add_log("Stage 9b Harmonizer: inpaint seam blending applied at SR resolution.")
                except Exception as exc:
                    context.add_log(f"Stage 9b Harmonizer failed: {exc}")

        # ── Post-SR Deferred Face Restoration ─────────────────────────────────
        if enable_face_restore and context.pipeline_flags.get("face_restore_deferred_to_post_sr", False):
            context.pipeline_flags["face_restore_deferred_to_post_sr"] = False
            context.pipeline_flags["_running_deferred_face_restore"] = True
            try:
                self._stage_7_face_restoration(context, on_progress)
                context.add_log("Stage 9c Post-SR FaceRestoration: Executed deferred face restoration on super-resolved canvas.")
            finally:
                context.pipeline_flags["_running_deferred_face_restore"] = False

        # ── Stage 10: Post-SR Luminance Sharpening ────────────────────────────
        if enable_sr:
            self._stage_10_post_sr_sharpen(context, on_progress)

        # ── Stage 10b: Projector Display Grading ──────────────────────────────
        self._stage_10b_projector_grade(context, on_progress)

        DebugLogger.save_debug_images(context)

        if on_progress:
            on_progress("complete", 100, "Restoration complete!")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        device_preference: str = "auto",
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        on_progress: Optional[Callable[[str, int, str], None]] = None,
        use_orchestrator: Optional[bool] = None,
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
            use_orchestrator: Optional bool to execute via Two-Tier PPCAV PhotoOrchestrator
            
        Returns:
            PNG encoded bytes of the restored image at full 4x resolution
        """
        cb = progress_callback or on_progress

        restored_rgb = self.restore(
            image_input=image_bytes,
            external_metadata=external_metadata,
            enable_repair=enable_repair,
            enable_denoise=enable_denoise,
            enable_colorization=enable_colorization,
            enable_face_restore=enable_restoration,
            enable_sr=enable_sr,
            on_progress=cb,
            use_orchestrator=use_orchestrator,
        )

        out_pil = Image.fromarray(restored_rgb)
        buf = io.BytesIO()
        out_pil.save(buf, format="PNG")
        return buf.getvalue()
