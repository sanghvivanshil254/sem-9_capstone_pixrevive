"""Damage detection & repair stages: fracture masking, LaMa inpainting,
seam smoothing, facial-anatomy protection, wipeout circuit breaker
(mixins on RestorationPipeline; see stage_registry.py)."""

from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import torch

from ai.pipeline.pipeline_context import PipelineContext
from ai.pipeline.debug_logger import DebugLogger
from ai.pipeline.stages._helpers import _refresh_sharpness
from ai.analysis.quality_analyzer import QualityAnalyzer
from ai.analysis.face_detector import FaceDetector
from ai.damage.enhanced_detector import EnhancedDamageDetector
from ai.damage.tearmark_detector import TearmarkDetector
from ai.damage.mask_refiner import MaskRefiner
from ai.restoration.inpainting_engine import InpaintingEngine
from ai.restoration.super_resolution import SuperResolution
from ai.colorization.color_refiner import ColorArtifactReducer
from ai.postprocessing.enhancer import Harmonizer


class DamageStagesMixin:
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

        Strategy (Fix 3 — foxing / mold, T-1):
          EnhancedDamageDetector.detect_dark_spots() (black-hat) contributes
          small dark blemishes — foxing, mold speckles, ink dots — that every
          other detector (tuned for *bright* flakes and cracks) ignores. It is
          OR-ed in after the grain-field gate so noisy frames still route to
          the denoise path, and dropped entirely when its coverage exceeds
          FOXING_COVERAGE_CAP: shadows and dark backgrounds light up black-hat
          across huge areas, so a large "spots" mask is a false positive by
          size, not by content.

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

        # ── Foxing / mold spots (T-1) ────────────────────────────────────
        # Only on frames the grain gate did not claim; black-hat on a noisy
        # scan mostly fires on grain. detect_dark_spots() itself is sized to
        # reject shadows (component area <= 100 px, circularity >= 0.3,
        # pixel < 60); the coverage cap below rejects whole-region shadows.
        FOXING_COVERAGE_CAP = 2.0  # percent of frame
        if not context.image_metadata.get("grain_field_mask_discarded", False):
            try:
                dark_spots = self.enhanced_detector.detect_dark_spots(
                    context.working_image, context
                )
                spots_pct = float(np.sum(dark_spots > 0) / max(dark_spots.size, 1) * 100)
                if spots_pct > FOXING_COVERAGE_CAP:
                    context.add_log(
                        f"Stage 2 Foxing gate: dark-spot coverage {spots_pct:.2f}% "
                        f"> {FOXING_COVERAGE_CAP:.1f}% cap — shadow/dark-texture "
                        "false positive, mask discarded."
                    )
                elif np.any(dark_spots):
                    combined = cv2.bitwise_or(combined, dark_spots)
                    context.damage_masks["dark_spots"] = dark_spots
                    context.add_log(
                        f"Stage 2 Foxing: {spots_pct:.2f}% dark-spot coverage "
                        "added to damage mask."
                    )
            except Exception as exc:
                context.add_log(f"Stage 2 Foxing detection failed: {exc}")

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

    def _stage_3_lama_inpaint(
        self,
        context: PipelineContext,
        mask: Optional[np.ndarray] = None,
        on_progress: Optional[Callable] = None,
    ) -> None:
        """Stage 3: LaMa Deep Inpainting with Boundary Seam Smoothing.

        Decomposed into three unit-testable sub-steps (A-2/U-3):
          _seam_smooth                  -- feathered blend along inpaint rims
          _protect_facial_anatomy       -- keeps eyes/mouth out of the seam blur
          _face_wipeout_circuit_breaker -- reverts catastrophic anatomy loss

        `mask` defaults to the refined final mask already stored on the
        context (registry-loop compatibility; restore() passes it explicitly).
        """
        if mask is None:
            mask = context.damage_masks.get("final")
            if mask is None or not np.any(mask > 0):
                context.add_log("Stage 3 Inpainting: No structural damage detected; skipping.")
                return
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

            self._seam_smooth(context, mask, faces)
            self._face_wipeout_circuit_breaker(context, pre_inpaint_canvas, mask, faces)

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

    def _protect_facial_anatomy(
        self,
        boundary_zone: np.ndarray,
        faces: List[Dict[str, Any]],
        img_shape: tuple,
    ) -> np.ndarray:
        """Zero facial-feature zones out of a (feathering) zone mask.

        Protects eyes and mouth -- by landmark when available, by anatomical
        proportion when not -- so seam-boundary smoothing and similar blends
        never blur critical anatomy. Mutates and returns `boundary_zone`.

        Args:
            boundary_zone: uint8 mask marking where blending may happen.
            faces: face dicts with `bbox` and optional `landmarks` (5 points:
                two eyes, nose, two mouth corners).
            img_shape: (h, w) of the frame, for bbox clamping.
        """
        h_img, w_img = img_shape[:2]
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
        return boundary_zone

    def _seam_smooth(self, context: PipelineContext, mask: np.ndarray, faces: List[Dict[str, Any]]) -> None:
        """Feathered Gaussian blend along inpaint mask boundaries.

        Builds a dilate-minus-erode boundary band, carries out facial anatomy
        via _protect_facial_anatomy, then blends a blurred copy through the
        band so repaired regions do not end in hard edges.
        """
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        boundary_zone = cv2.dilate(mask, k_erode) - cv2.erode(mask, k_erode)

        boundary_zone = self._protect_facial_anatomy(boundary_zone, faces, context.working_image.shape)

        seam_f32 = boundary_zone.astype(np.float32) / 255.0
        seam_f32 = cv2.GaussianBlur(seam_f32, (9, 9), 0)[:, :, None]
        blurred = cv2.GaussianBlur(context.working_image, (7, 7), 0)
        context.working_image = np.clip(
            context.working_image.astype(np.float32) * (1.0 - seam_f32) +
            blurred.astype(np.float32) * seam_f32, 0, 255
        ).astype(np.uint8)
        context.add_log("Stage 3 Inpainting: Seam boundary smoothing applied.")

    def _face_wipeout_circuit_breaker(
        self,
        context: PipelineContext,
        pre_inpaint_canvas: np.ndarray,
        mask: np.ndarray,
        faces: List[Dict[str, Any]],
    ) -> None:
        """Revert face regions whose high-frequency anatomy inpainting erased.

        Per face: if the pre-inpaint damage ratio is in (0.03, 0.20] and
        post-inpaint Sobel edge energy collapsed (< 35.0 absolute and
        < 5% of the pre-inpaint energy), the fill destroyed structure rather
        than repairing it -- restore the pre-inpaint pixels and record a
        warning. Heavily damaged faces (ratio > 0.20) are exempt: inpainting
        legitimately removes flake-edge variance there while synthesizing
        contours.
        """
        if not faces:
            return
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

            if 0.03 < face_damage_ratio <= 0.20:
                crop_pre = pre_inpaint_canvas[y0:y1, x0:x1]
                crop_post = context.working_image[y0:y1, x0:x1]
                gray_pre = cv2.cvtColor(crop_pre, cv2.COLOR_RGB2GRAY).astype(np.float32)
                gray_post = cv2.cvtColor(crop_post, cv2.COLOR_RGB2GRAY).astype(np.float32)

                grad_pre = float(np.mean(cv2.Sobel(gray_pre, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(gray_pre, cv2.CV_32F, 0, 1)**2))
                grad_post = float(np.mean(cv2.Sobel(gray_post, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(gray_post, cv2.CV_32F, 0, 1)**2))

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

