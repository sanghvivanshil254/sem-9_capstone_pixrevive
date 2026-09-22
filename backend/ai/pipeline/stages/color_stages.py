"""Colorization stage: DDColor transform + chrominance diffusion
(mixin on RestorationPipeline; see stage_registry.py)."""

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


# Q-3 hallucination-rollback thresholds.
_SINGLE_HUE_SAT_THRESHOLD = 0.88   # fraction of sampled pixels within one 15° hue bin
_MIN_SATURATION_FRACTION = 0.10    # below this mean saturation the wash can't be "vivid"
_MIN_PIX_COUNT = 1000              # skip the guard on tiny canvases (noise-dominated)


class ColorStagesMixin:
    @staticmethod
    def _detect_colorization_hallucination(rgb: np.ndarray) -> bool:
        """Q-3: pathological single-hue wash detector (>88% within one hue bin).

        Mirrors the guard documented for the legacy ECCV16 colorizer: when
        DDColor hallucinates one dominant color, nearly every sufficiently
        saturated pixel lands in a single 15° hue bin. The saturation floor
        keeps genuinely monochromatic content (an overcast sky, a red curtain
        filling the frame) from tripping the guard on low-chroma images.
        """
        try:
            if rgb.ndim != 3 or rgb.shape[0] * rgb.shape[1] < _MIN_PIX_COUNT:
                return False
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            sat = hsv[:, :, 1].astype(np.float32)
            if float(np.mean(sat)) < _MIN_SATURATION_FRACTION * 255.0:
                return False
            vivid = sat > 40
            if not np.any(vivid):
                return False
            hues = hsv[:, :, 0][vivid].astype(np.int32)
            counts = np.bincount(hues // 15, minlength=12)
            dominant = int(counts.max()) / float(hues.size)
            return dominant > _SINGLE_HUE_SAT_THRESHOLD
        except Exception:
            return False

    @staticmethod
    def _rollback_to_l_preserved_grayscale(context: PipelineContext, pre_color: np.ndarray) -> None:
        """Q-3 fallback: revert to the pre-colorization canvas with a warning.

        The working canvas before Stage 6 is grayscale-identity in L, so
        restoring it preserves the luminance DDColor was supposed to keep
        while discarding the hallucinated chroma entirely.
        """
        context.working_image = pre_color.copy()
        context.image_metadata["colorization_rolled_back"] = True
        context.add_log(
            "Stage 6 Colorization: pathological single-hue wash detected — "
            "rolled back to luminance-preserved grayscale. Colors are not "
            "trustworthy for this input."
        )

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

        pre_colorization_canvas = context.working_image.copy()
        try:
            if not self._colorizer_ready.is_set():
                self._colorizer_ready.wait(timeout=60.0)
            if self.colorizer.is_loaded:
                self.colorizer.colorize(context)
                context.add_log("Stage 6 Colorization: DDColor completed.")

                # ── Q-3 hallucination rollback guard ────────────────────────
                # A pathological single-hue wash means DDColor invented chroma
                # wholesale; the luminance-preserving pre-colorization canvas
                # is strictly more honest. Runs before any chroma diffusion so
                # downstream stages never refine a hallucinated wash.
                if self._detect_colorization_hallucination(context.working_image):
                    self._rollback_to_l_preserved_grayscale(
                        context, pre_colorization_canvas
                    )
                    context.end_timer("colorization")
                    DebugLogger.save_stage(context, "06_after_colorize")
                    return

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

