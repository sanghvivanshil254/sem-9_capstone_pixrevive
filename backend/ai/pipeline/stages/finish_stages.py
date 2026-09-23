"""Finishing stages: 3-zone denoise, GFPGAN face restoration, pre-SR
anti-aliasing, super-resolution, sharpening, exhibition grading
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
from ai.restoration.deblurrer import ImageDeblurrer, estimate_optical_blur
from ai.colorization.color_refiner import ColorArtifactReducer
from ai.postprocessing.enhancer import Harmonizer


class FinishStagesMixin:
    def _stage_4_5_deblur(self, context: PipelineContext, on_progress: Optional[Callable]) -> None:
        """Stage 4.5 (T-3): Restormer deblur, strictly gated on optical blur.

        Two-path probe-calibrated gate (see deblurrer.estimate_optical_blur):
        path A = defocus via edge-spread acutance, path B = motion via H/V
        gradient anisotropy. The whole real corpus measures inert. The 100MB
        Restormer loads lazily on the FIRST gate-fire: images that don't need
        it never pay the VRAM.
        """
        DebugLogger.save_stage(context, "05_before_deblur")
        if on_progress:
            on_progress("deblur", 56, "Correcting optical blur...")

        measurement = estimate_optical_blur(context.working_image)
        if not measurement["is_optical"]:
            context.add_log(
                f"Stage 4.5 Deblur: not optical blur "
                f"(spread={measurement.get('edge_spread')}, "
                f"dir={measurement.get('directional_ratio')}) - skipping."
            )
            return

        try:
            if self._deblur is None:
                self._deblur = ImageDeblurrer()
            if not self._deblur.is_loaded():
                context.add_log("Stage 4.5 Deblur: loading Restormer weights (first fire)...")
                self._deblur.load(self._device)
            context.start_timer("deblurring")
            result, info = self._deblur.deblur(context.working_image, device=self._device)
            if info.get("applied"):
                context.working_image = result
                context.add_log(
                    f"Stage 4.5 Deblur: Restormer applied (path={info.get('fire_path')}, "
                    f"spread={info.get('edge_spread')}, dir={info.get('directional_ratio')})."
                )
            else:
                context.add_log(f"Stage 4.5 Deblur skipped: {info.get('reason', info.get('error'))}")
        except FileNotFoundError as exc:
            context.add_log(f"Stage 4.5 Deblur unavailable: {exc}")
            self._model_load_failures["deblur"] = str(exc)
        except Exception as exc:
            context.add_log(f"Stage 4.5 Deblur failed: {exc}")
            self._model_load_failures["deblur"] = str(exc)

    def _stage_5_spatially_gated_denoise(
        self,
        context: PipelineContext,
        faces: Optional[List[Dict[str, Any]]] = None,
        on_progress: Optional[Callable] = None,
    ) -> None:
        """
        Stage 5: Spatially-Gated 3-Zone Denoising.
        - Face zone: Gentle NLM (h ≈ 0.95σ, capped 10.5) preserving skin texture.
        - Subject zone: Balanced NLM (h ≈ 1.0σ).
        - Background (undamaged, high σ): Moderate NLM (h = 0.85σ, capped 11.5) — preserves photographic grain.
        - Background (damaged or low σ): Aggressive NLM (h = 1.4σ) for deep grain/scratch suppression.

        `faces` defaults to the faces already recorded in context metadata, so
        the registry loop can call this stage uniformly (method(context)).
        """
        if faces is None:
            faces = context.image_metadata.get("faces", [])
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

            # -- T-6: Banding pass (registry stage 'deband') -----------------
            # Plan T-6: dedicated bit-depth-expansion + dithering pass applied
            # pre-SR when banding_score is high. Gated on the noise router's
            # banding measurement (corpus max 0.503; gate 0.6 — synthetic
            # posterization reads 1.0). Super-resolution then amplifies
            # whatever stairs remain, so this is the last cheap exit.
            try:
                from ai.postprocessing.deband import deband
                debanded, deband_info = deband(context.working_image)
                if deband_info.get("applied"):
                    context.working_image = debanded
                    context.add_log(
                        f"Stage 8 Deband: quantization stairs smoothed "
                        f"(score={deband_info.get('banding_score')}) (T-6)."
                    )
            except Exception as deband_exc:
                context.add_log(f"Stage 8 Deband skipped: {deband_exc}")
        except Exception as exc:
            context.add_log(f"Stage 8 Pre-SR Anti-Aliasing failed: {exc}")

        DebugLogger.save_stage(context, "08_after_pre_sr")

    def _stage_9_super_resolution(self, context: PipelineContext, on_progress: Optional[Callable] = None) -> None:
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

