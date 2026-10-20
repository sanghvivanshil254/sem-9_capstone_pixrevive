"""Colorization stage: DDColor transform + chrominance diffusion
(mixin on RestorationPipeline; see stage_registry.py)."""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import threading
import cv2
import numpy as np
import os

from ai.pipeline.pipeline_context import PipelineContext
from ai.pipeline.debug_logger import DebugLogger


# Q-3 hallucination-rollback thresholds.
_SINGLE_HUE_SAT_THRESHOLD = 0.88   # fraction of sampled pixels within one 15° hue bin
_MIN_SATURATION_FRACTION = 0.10    # below this mean saturation the wash can't be "vivid"
_MIN_PIX_COUNT = 1000              # skip the guard on tiny canvases (noise-dominated)


# RC-4 (debug session 2026-10-01): inventing chroma for a LOW-RESOLUTION or
# NOISY grayscale source produces the posterized blue/orange blob wash seen on
# the SigShrink speckle-Lena figure (435x342 paper figure, noise_score 3.7).
# DDColor has no evidence for any color there; every hue is synthesis, SR then
# quantizes the wash into flat plates. The Q-3 single-hue guard only catches
# the degenerate one-hue case — the Lena wash used four. This gate skips
# colorization when a grayscale source lacks the two things chroma synthesis
# needs: enough pixels and a low noise floor. Measured: the speckle figure's
# grain reads as noise_score 3.7 / sigma 1.48 — REAL speckle that the global
# score under-rates — so the bar for small grayscale canvases is deliberately
# low (2.5). PIXREVIVE_GRAY_SKIP=0 reverts.
def _gray_skip_warranted(context: PipelineContext) -> bool:
    if os.environ.get("PIXREVIVE_GRAY_SKIP", "1").strip().lower() in (
        "0", "false", "no", "off"
    ):
        return False
    # Fix 2 / Option 3: bypass gray_skip when PIXREVIVE_COLOR_TYPE_FIX is enabled
    if os.environ.get("PIXREVIVE_COLOR_TYPE_FIX", "1").strip().lower() not in (
        "0", "false", "no", "off"
    ):
        return False
    meta = context.image_metadata
    if meta.get("color_type") not in ("grayscale", "monochrome"):
        return False
    h, w = context.working_image.shape[:2]
    if h * w > 1_200_000:
        return False  # ≥1.2 MP: enough evidence for chroma synthesis
    noise = float(meta.get("noise_score", 0.0) or 0.0)
    sigma = float(meta.get("content_sigma", 0.0) or 0.0)
    profile = meta.get("noise_profile") or {}
    grain_types = {"grain_floor", "speckle", "gaussian", "salt_and_pepper",
                   "random_valued_impulse", "impulse", "poisson", "shot"}
    detected = set(profile.get("detected_types") or [])
    return noise > 2.5 or sigma > 4.0 or bool(detected & grain_types)


class ColorStagesMixin:
    if TYPE_CHECKING:  # pragma: no cover — host-object contract only
        from ai.colorization.ddcolor_colorizer import DDColorColorizer

    # Host-object contract (pyright, plan 6.3): these attributes live on
    # RestorationPipeline, which composes this mixin.
    _colorizer_ready: threading.Event
    colorizer: "DDColorColorizer"

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
        # N5 (re-audit 2026-09-30): the report builder's disclaimer gate is
        # evidence-based (colorization_applied / color_type), so a rollback
        # that returned early left colorization_disclaimer False on a B&W
        # output — case9's sepia portrait shipped grayscale with no "colors
        # are not trustworthy" disclosure. Seed the flag here; the builder
        # also derives it from colorization_rolled_back as a backstop.
        context.image_metadata["colorization_disclaimer"] = True
        context.add_log(
            "Stage 6 Colorization: pathological single-hue wash detected — "
            "rolled back to luminance-preserved grayscale. Colors are not "
            "trustworthy for this input."
        )

    @staticmethod
    def _detect_background_chroma_wash(
        rgb: np.ndarray, faces: list, sepia: bool = True,
        pre_rgb: Optional[np.ndarray] = None,
    ) -> bool:
        """RC-4b: invented background wash on sepia sources (pink/lavender field).

        DDColor's chroma on a mottled sepia scan is per-mottle hallucination:
        the wash passes Q-3 when it uses several hue bins (measured: pink
        background + warm skin on the teared portrait), but physically the
        aged paper is LOW-CHROMA. Faces are excluded (skin legitimately
        carries chroma). Two firing conditions:

        1. Strong-field: a large share of the background deviates >15 in
           Lab a/b (|ab| ~20-30 on the measured wash vs <10 for aged paper).
        2. Sepia-drift (measured SHIP6 regression): the wash passed guard 1
           (|ab| only ~10) but REPLACED the paper's own warmth — background
           b dropped +31 -> +4 with a magenta lean (a=+8.4) — reading as a
           pink cast at viewing scale. When `pre_rgb` (the pre-colorization
           canvas) carries genuine sepia warmth (bg b >= 8), a colorized
           result that destroys that warmth (b < 60% of it) or turns the
           paper magenta-dominant (a > 4 and a > b) is hallucination.
        """
        try:
            if not sepia or rgb.ndim != 3:
                return False
            h, w = rgb.shape[:2]
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            a, b = lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0
            bg = np.ones((h, w), bool)
            for f in faces:
                bbox = f.get("bbox") if isinstance(f, dict) else None
                if bbox and len(bbox) >= 4:
                    x0, y0, x1, y1 = (int(v) for v in bbox[:4])
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(w, x1), min(h, y1)
                    if x1 > x0 and y1 > y0:
                        bg[y0:y1, x0:x1] = False
            if bg.sum() < 1000:
                return False
            dev = np.sqrt(a[bg] ** 2 + b[bg] ** 2)
            # Condition 1: strong chroma field on the paper.
            if float((dev > 15.0).mean()) > 0.35:
                return True
            # Condition 2: sepia-warmth destroyed / hue flipped.
            if pre_rgb is not None and pre_rgb.ndim == 3:
                pre_lab = cv2.cvtColor(pre_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
                pa, pb = pre_lab[:, :, 1] - 128.0, pre_lab[:, :, 2] - 128.0
                pre_b = float(pb[bg].mean())
                post_a, post_b = float(a[bg].mean()), float(b[bg].mean())
                if pre_b >= 8.0:  # the source really is warm sepia paper
                    if post_b < 0.60 * pre_b:
                        return True
                    if post_a > 4.0 and post_a > post_b:
                        return True
            return False
        except Exception:
            return False

    def _stage_6_colorization(self, context: PipelineContext, on_progress: Optional[Callable]) -> None:
        """Stage 6: DDColor Colorization with Guided Chrominance Diffusion."""
        color_type = context.image_metadata.get("color_type", "color")
        if color_type not in ("grayscale", "sepia"):
            context.add_log("Stage 6 Colorization: Image is already color; skipping.")
            return

        if _gray_skip_warranted(context):
            context.image_metadata["colorize_skipped_lowres_gray"] = True
            context.add_log(
                "Stage 6 Colorization: low-resolution noisy grayscale source — "
                "skipping chroma synthesis (invented color on this input is "
                "hallucination, and SR posterizes it into blobs). Output stays "
                "grayscale."
            )
            DebugLogger.save_stage(context, "06_after_colorize")
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
                _color_type_fix = os.environ.get("PIXREVIVE_COLOR_TYPE_FIX", "1").strip().lower() not in (
                    "0", "false", "no", "off"
                )
                # Fix 2 / Option 3: bypass sepia background chroma wash rollback when colorization is requested
                _allow_sepia_wash_rollback = not _color_type_fix

                single_hue_wash = self._detect_colorization_hallucination(context.working_image)
                bg_wash = (
                    _allow_sepia_wash_rollback
                    and color_type == "sepia"
                    and self._detect_background_chroma_wash(
                        context.working_image,
                        context.image_metadata.get("faces") or [],
                        pre_rgb=pre_colorization_canvas,
                    )
                )

                if single_hue_wash or bg_wash:
                    wash_kind = (
                        "single-hue"
                        if single_hue_wash
                        else "background chroma field"
                    )
                    self._rollback_to_l_preserved_grayscale(
                        context, pre_colorization_canvas
                    )
                    context.image_metadata["colorization_wash_kind"] = wash_kind
                    context.add_log(
                        f"Stage 6 Colorization: pathological {wash_kind} wash "
                        "detected — rolled back to luminance-preserved grayscale. "
                        "Colors are not trustworthy for this input."
                    )
                    context.end_timer("colorization")
                    DebugLogger.save_stage(context, "06_after_colorize")
                    return

                # Positive evidence flag for the Restoration Report: the
                # colorization claim ("colors added by AI") must rest on
                # colorization actually having RUN, not on the input being
                # B&W/sepia. Measured 2026-09-27: colorize=false on a sepia
                # image still produced the disclaimer because the report
                # inferred from color_type alone.
                context.image_metadata["colorization_applied"] = True

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
                        
                        # Structure-guided chrominance snapping (J-7, 2026-10-12):
                        # originally guided on RAW l_chan so chrominance boundaries
                        # would align with physical edges. On damaged scans that
                        # backfired: the guide is the damage-ragged luminance
                        # (fractured hair outlines, torn paper edges), so guided
                        # filtering SNAPPED chroma to those ragged edges — the
                        # measured posterised cut-paper fringe on hair/jaw/
                        # shoulders. Now the guide is a smoothed L (structure
                        # survives, damage fringe is blurred away) with a larger
                        # radius/eps so chroma follows broad anatomical zones,
                        # not speckle-scale edges. PIXREVIVE_CHROMA_SNAP_RAW=1
                        # reverts to the raw-guide behaviour.
                        if os.environ.get("PIXREVIVE_CHROMA_SNAP_RAW", "0").strip().lower() in ("1", "true", "yes", "on"):
                            _snap_guide = l_chan
                            _snap_radius, _snap_eps = 6, 15.0
                        else:
                            _snap_guide = cv2.GaussianBlur(l_chan, (0, 0), sigmaX=max(2.0, 0.008 * max(l_chan.shape[:2])))
                            _snap_radius, _snap_eps = 10, 60.0
                        try:
                            a_diffused = cv2.ximgproc.guidedFilter(guide=_snap_guide, src=a_diffused, radius=_snap_radius, eps=_snap_eps)
                            b_diffused = cv2.ximgproc.guidedFilter(guide=_snap_guide, src=b_diffused, radius=_snap_radius, eps=_snap_eps)
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

