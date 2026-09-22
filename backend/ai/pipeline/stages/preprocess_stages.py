"""Preprocess & analysis stages: despeckle, analyze, face detection,
damage-cluster smoothing, post-inpaint face refresh (mixins on
RestorationPipeline; see stage_registry.py for the stage contract)."""

from typing import Any, Callable, Dict, List, Optional

import logging

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

logger = logging.getLogger(__name__)


class PreprocessStagesMixin:
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

    def _stage_2b_damage_cluster_smooth(self, context: PipelineContext, on_progress: Optional[Callable] = None, _compat=None) -> None:
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

            # Continuous feathered face safe zone (U-6).
            # With landmarks available the safe zone is the convex hull of the
            # 5-point facial set, dilated to anatomical scale — it hugs the
            # actual face instead of the frame center. The legacy center
            # rectangle survives ONLY as the no-face fallback: a center rect
            # on an off-center portrait used to blind exactly the zone where
            # the damage clusters were.
            safe_mask = np.zeros((h_img, w_img), dtype=np.float32)

            faces = context.image_metadata.get("faces", [])
            hull_points = []
            face_span = 0.0
            bbox_rects = []
            for f in faces:
                lm = f.get("landmarks")
                if lm:
                    hull_points.extend([(int(px), int(py)) for px, py in lm])
                    pts = np.asarray(lm, dtype=np.float32)
                    face_span = max(face_span, float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])))
                else:
                    bbox = f.get("bbox")
                    if bbox and len(bbox) >= 4:
                        bbox_rects.append([int(v) for v in bbox[:4]])

            if hull_points:
                hull = cv2.convexHull(np.array(hull_points, dtype=np.int32))
                cv2.fillConvexPoly(safe_mask, hull, 1.0)
                # Eyes/nose/mouth sit inside the 5-point hull, but forehead,
                # cheeks and chin extend well beyond it — dilate by ~0.6× the
                # landmark spread so the whole anatomical region is protected.
                grow = max(9, int(0.6 * face_span) | 1)
                kernel_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))
                safe_u8 = (safe_mask * 255.0).astype(np.uint8)
                safe_u8 = cv2.dilate(safe_u8, kernel_grow)
                safe_mask = safe_u8.astype(np.float32) / 255.0
            elif bbox_rects:
                for x0, y0, x1, y1 in bbox_rects:
                    safe_mask[max(0, y0):min(h_img, y1), max(0, x0):min(w_img, x1)] = 1.0
            else:
                cy, cx = h_img // 2, w_img // 2
                safe_y0 = int(cy - 0.25 * h_img)
                safe_y1 = int(cy + 0.25 * h_img)
                safe_x0 = int(cx - 0.25 * w_img)
                safe_x1 = int(cx + 0.25 * w_img)
                safe_mask[max(0, safe_y0):min(h_img, safe_y1), max(0, safe_x0):min(w_img, safe_x1)] = 1.0

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

