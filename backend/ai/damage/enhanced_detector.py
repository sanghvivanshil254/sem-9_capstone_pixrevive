"""
enhanced_detector.py — Enhanced damage detection for severe white flakes and scratches.

This module provides multi-scale damage detection specifically targeting:
- Severe white emulsion flakes
- Heavy scratch patterns (including wide diagonal cracks)
- Fine and wide branching crack networks (Hessian + multi-scale Black-Hat morphology)
- True isolated dark spots (dust, mold)
- Border edge damage and peeling

Key design points (v6):
1. Multi-scale Black-Hat kernel bank {9, 17, 27, 37} px to bridge wide fractures (18-35px).
2. Crack thresholding is a robust *global* outlier test on the black-hat response
   (mean + k*sigma, floored by median + k*MAD and a perceptual minimum depth),
   escalated if it floods. The v5 Sauvola-style local threshold collapsed to
   0.78 * local_mean and masked 85-99% of every frame it was given; see
   _thin_cracks_classical for the full analysis.
3. Components must be crack-shaped: long enough to matter, and at least 4x longer
   than their mean stroke width, which rejects the dark blobs (eyes, nostrils,
   hair) a 37px black-hat responds to as strongly as it does to a fracture.
4. Dilation margins scale with the defect (3x3 for hairlines, 7x7 for wide tears)
   instead of the v5 fixed 11x11 plus a second 19x19, which turned a 1px crack
   into an 11px band.
5. A coverage ceiling discards the mask entirely rather than hand most of the
   frame to the inpainter — under-detecting is the cheaper failure in a fallback.

Note: this module assumes impulse noise has already been removed (Stage 0,
ai.preprocessing.impulse_denoiser). An isolated bright speck is indistinguishable
from an emulsion flake and an isolated dark speck from the start of a crack, so
running these detectors on a speckled image produces a mask that morphological
closing merges into frame-filling blobs.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from ai.pipeline.pipeline_context import PipelineContext
from ai.damage.defect_segmenter import DefectSegmenter

logger = logging.getLogger(__name__)


class EnhancedDamageDetector:
    """
    Enhanced detector for severe damage patterns in archival photographs.
    
    Detects:
    - White flakes (emulsion damage)
    - Black spots (dust, mold)
    - Scratches (linear damage) — both wide and thin
    - Crack networks (multi-scale Black-Hat and Hessian)
    - Stains and edge damage
    """

    def __init__(self, use_defect_segmenter: bool = True) -> None:
        self.min_flake_size: int = 3
        self.max_flake_size: float = 0.05
        self.scratch_sensitivity: float = 0.7
        self.edge_protection_radius: int = 2
        # Set False to force the classical crack path. Assigning
        # `detector.defect_segmenter = None` after construction does *not* do
        # this: detect_thin_crack_network lazily rebuilds the segmenter, so the
        # learned path comes back. The regression suite needs the classical
        # branch to be reachable on a machine where the weights are present.
        self.use_defect_segmenter: bool = use_defect_segmenter
        self.defect_segmenter: Optional[DefectSegmenter] = None
        if use_defect_segmenter:
            try:
                self.defect_segmenter = DefectSegmenter()
            except Exception as exc:
                logger.warning(f"DefectSegmenter init failed: {exc}")

    def detect_white_flakes(self, img: np.ndarray, context: PipelineContext) -> np.ndarray:
        """Detect white emulsion flakes using adaptive thresholding."""
        h, w = img.shape[:2]
        color_type = context.image_metadata.get("color_type", "color")

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0].astype(np.float32)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        S = hsv[:, :, 1].astype(np.float32)
        V = hsv[:, :, 2].astype(np.float32)

        mean_L = float(np.mean(L))

        # Noise-adaptive parameters to avoid classifying raw grain as emulsion flakes
        noise_score = float(context.image_metadata.get("noise_score", 0.0))
        if noise_score > 6.0:
            L_filtered = cv2.medianBlur(np.clip(L, 0, 255).astype(np.uint8), 3).astype(np.float32)
            diff_thresh = max(26.0, 20.0 + (noise_score - 6.0) * 1.5)
            effective_min_size = max(10, int(noise_score * 1.5))
        else:
            L_filtered = L
            diff_thresh = 18.0
            effective_min_size = self.min_flake_size

        window_size = max(15, int(min(h, w) * 0.025) | 1)
        L_blur = cv2.GaussianBlur(L_filtered, (window_size, window_size), 0)
        local_diff = L_filtered - L_blur

        if color_type in ["monochrome", "grayscale"]:
            # In monochrome photographs, saturation is ~0 everywhere, so brightness cannot be
            # distinguished from photographic tone by saturation alone. Emulsion loss is a
            # localized anomaly that exposes the high-contrast bright base paper substrate.
            # Require paper-white substrate brightness and step boundary gradient to prevent
            # misclassifying specular highlights (e.g. hat ribbon, cheek shine) as emulsion flakes.
            diff_thresh_mono = max(24.0, 20.0 + (noise_score - 3.0) * 1.5 if noise_score > 3.0 else 24.0)
            flake_mask = (local_diff > diff_thresh_mono) & (V > 215.0) & (L > max(mean_L + 35.0, 205.0))
        else:
            # In sepia or color photographs, exposed base paper lacks saturation AND is bright
            brightness_threshold = min(max(mean_L + 25.0, 175.0), 220.0)
            sat_threshold = 45 if color_type == "sepia" else 55
            color_flake = (V > 170.0) & (S < sat_threshold) & (L > brightness_threshold) & (local_diff > 10.0)
            local_flake = (local_diff > diff_thresh) & (V > 150.0)
            flake_mask = color_flake | local_flake

        flake_uint8 = flake_mask.astype(np.uint8) * 255

        # Filter out tiny noise specks (<4px) and massive non-flake regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(flake_uint8, connectivity=8)
        max_area = int(h * w * self.max_flake_size)
        if color_type in ["monochrome", "grayscale"] and num_labels > 1:
            sobel_x = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            k3 = np.ones((3, 3), np.uint8)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < effective_min_size or area > max_area:
                    flake_uint8[labels == i] = 0
                    continue
                comp_mask = (labels == i).astype(np.uint8)
                dil = cv2.dilate(comp_mask, k3)
                bnd = dil - comp_mask
                if np.any(bnd > 0):
                    mean_bnd_grad = float(np.mean(grad_mag[bnd > 0]))
                    if mean_bnd_grad < 35.0:  # Smooth photographic highlight falloff, not chipped emulsion
                        flake_uint8[labels == i] = 0
        else:
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < effective_min_size or area > max_area:
                    flake_uint8[labels == i] = 0

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        flake_uint8 = cv2.dilate(flake_uint8, k, iterations=1)

        # Zero out ocular and oral zones: teeth and sclera are natural human features, not emulsion flakes
        faces = context.image_metadata.get("faces", [])
        for f in faces:
            lms = f.get("landmarks")
            bbox = f.get("bbox")
            if lms and len(lms) >= 5:
                for idx_eye in (0, 1):
                    ex, ey = int(round(lms[idx_eye][0])), int(round(lms[idx_eye][1]))
                    cv2.circle(flake_uint8, (ex, ey), max(8, int(min(h, w) * 0.035)), 0, -1)
                ml, mr = np.array(lms[3], dtype=np.float32), np.array(lms[4], dtype=np.float32)
                mc = (ml + mr) / 2.0
                mw = float(np.linalg.norm(mr - ml))
                ang = float(np.degrees(np.arctan2(mr[1] - ml[1], mr[0] - ml[0])))
                axes = (int(max(10, mw * 0.58)), int(max(6, mw * 0.28)))
                cv2.ellipse(flake_uint8, (int(round(mc[0])), int(round(mc[1]))), axes, ang, 0, 360, 0, -1)
            elif bbox and len(bbox) >= 4:
                bx0, by0, bx1, by1 = [int(v) for v in bbox[:4]]
                bfw, bfh = max(1, bx1 - bx0), max(1, by1 - by0)
                ex1, ey1 = int(bx0 + 0.32 * bfw), int(by0 + 0.38 * bfh)
                ex2, ey2 = int(bx0 + 0.68 * bfw), int(by0 + 0.38 * bfh)
                cv2.circle(flake_uint8, (ex1, ey1), max(8, int(min(bfw, bfh) * 0.08)), 0, -1)
                cv2.circle(flake_uint8, (ex2, ey2), max(8, int(min(bfw, bfh) * 0.08)), 0, -1)
                mx, my = int(bx0 + 0.50 * bfw), int(by0 + 0.76 * bfh)
                axes = (int(max(10, bfw * 0.22)), int(max(6, bfh * 0.10)))
                cv2.ellipse(flake_uint8, (mx, my), axes, 0, 0, 360, 0, -1)

        return flake_uint8

    def detect_wide_cracks(self, img: np.ndarray, context: PipelineContext) -> np.ndarray:
        """Detect wide cracks and structural tears with resolution-aware scaling."""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        scale = max(0.25, min(h, w) / 500.0)
        noise_score = float(context.image_metadata.get("noise_score", 0.0))

        # Scale directional kernel sizes with resolution to prevent oversizing on low-res images
        scales = [
            (max(7, int(15 * scale) | 1), max(2, int(3 * scale))),
            (max(11, int(25 * scale) | 1), max(3, int(5 * scale))),
            (max(15, int(35 * scale) | 1), max(3, int(7 * scale))),
        ]
        combined = np.zeros_like(gray)

        for kw, kh in scales:
            for angle_deg in (0, 45, 90, 135):
                if angle_deg == 0:
                    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kh, kw))
                elif angle_deg == 90:
                    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
                else:
                    sz = max(kw, kh)
                    k = np.zeros((sz, sz), dtype=np.uint8)
                    angle_rad = np.deg2rad(angle_deg)
                    cx, cy = sz // 2, sz // 2
                    for px in range(sz):
                        for py in range(sz):
                            dx, dy = px - cx, py - cy
                            rx = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
                            ry = -dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
                            if abs(rx) <= kh // 2 and abs(ry) <= kw // 2:
                                k[py, px] = 1
                    k = k.astype(np.uint8)
                    if k.sum() == 0:
                        continue

                tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
                blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
                combined = np.maximum(combined, tophat)
                combined = np.maximum(combined, blackhat)

        mean_val = float(np.mean(combined.astype(np.float32)))
        std_val = float(np.std(combined.astype(np.float32)))
        sigma_mult = 2.8 if noise_score > 3.0 else 2.2
        threshold = mean_val + sigma_mult * std_val

        _, mask = cv2.threshold(combined, threshold, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)

        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, k3, iterations=1 if scale < 0.6 else 2)

        min_aspect = 2.5 if noise_score > 3.0 else 2.2
        min_comp_area = max(12, int(20 * scale))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            aspect = max(width, height) / max(1, min(width, height))
            if area < min_comp_area or aspect < min_aspect:
                mask[labels == i] = 0

        context.add_log(f"WideCrackDetector: {float(np.sum(mask > 0) / mask.size * 100):.2f}% coverage")
        return mask

    def detect_3d_paper_tears(self, img: np.ndarray, context: PipelineContext) -> np.ndarray:
        """
        Detect severe 3D physical paper tears, ripped edges, and associated shadow creases.
        Scaled dynamically to avoid false-positive detections on noisy, low-resolution images.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()
        scale = max(0.25, min(h, w) / 500.0)
        noise_score = float(context.image_metadata.get("noise_score", 0.0))
        color_type = context.image_metadata.get("color_type", "color")

        # Pre-filter noise grain to prevent noise peaks from triggering substrate highlights
        if noise_score > 3.0:
            gray_proc = cv2.medianBlur(gray, 3)
            bright_diff = max(20.0, 15.0 + (noise_score - 3.0) * 1.5)
            grad_thresh = max(52.0, 45.0 + (noise_score - 3.0) * 2.0)
            bh_thresh = 26
        else:
            gray_proc = gray
            bright_diff = 15.0
            grad_thresh = 45.0
            bh_thresh = 22

        # 1. Exposed white paper fibers / substrate
        k_local = max(9, int(min(h, w) * 0.035) | 1)
        local_mean = cv2.blur(gray_proc, (k_local, k_local))
        if color_type in ["monochrome", "grayscale"]:
            p_bright_thresh = 210
            b_diff_thresh = max(22.0, bright_diff)
        else:
            p_bright_thresh = 185
            b_diff_thresh = bright_diff
        paper_bright = (gray_proc > p_bright_thresh) & (gray_proc > (local_mean.astype(np.float32) + b_diff_thresh))

        # 2. Shadow creases and dark bevel edges along torn paper flaps
        # Dynamically scale black-hat structuring element sizes with image resolution
        base_shadow_sizes = [9, 17, 25]
        k_shadow_sizes = [max(3, int(ks * scale) | 1) for ks in base_shadow_sizes]
        shadow_masks = []
        for ks in k_shadow_sizes:
            k_s = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
            bh = cv2.morphologyEx(gray_proc, cv2.MORPH_BLACKHAT, k_s)
            shadow_masks.append(bh > bh_thresh)
        shadow_combined = np.maximum.reduce(shadow_masks) if shadow_masks else np.zeros((h, w), dtype=bool)

        # 3. High gradient along torn paper edges
        grad_x = cv2.Sobel(gray_proc, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_proc, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        strong_edges = grad_mag > grad_thresh

        # 4. Bridge bright substrate with adjacent dark shadow bevels (scale-adaptive radius)
        # A true 3D physical paper tear consists of exposed white substrate paired with a shadow crease/bevel.
        # Highlights or noise peaks that have no co-located shadow bevel are photographic features, not 3D tears.
        raw_tear = paper_bright.astype(np.uint8) * 255
        k_bridge_sz = max(3, int(11 * scale) | 1)
        k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_bridge_sz, k_bridge_sz))
        tear_dilated = cv2.dilate(raw_tear, k_bridge, iterations=1)

        shadow_adjacent = cv2.bitwise_and(
            shadow_combined.astype(np.uint8) * 255, tear_dilated
        )

        k_check_sz = max(3, int(7 * scale) | 1)
        k_check = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_check_sz, k_check_sz))
        shadow_dilated = cv2.dilate(shadow_adjacent, k_check, iterations=1)
        verified_bright = cv2.bitwise_and(raw_tear, shadow_dilated)

        # Include edge gradients ONLY if they bridge verified tear substrate and shadow
        edge_adjacent = cv2.bitwise_and(
            strong_edges.astype(np.uint8) * 255, shadow_dilated
        )

        combined_tear = cv2.bitwise_or(verified_bright, shadow_adjacent)
        combined_tear = cv2.bitwise_or(combined_tear, edge_adjacent)

        # 5. Connect fragmented tear paths with morphological closing (scale-adaptive)
        k_close_sz = max(3, int(9 * scale) | 1)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close_sz, k_close_sz))
        combined_tear = cv2.morphologyEx(combined_tear, cv2.MORPH_CLOSE, k_close, iterations=1 if scale < 0.6 else 2)

        # 5. Filter non-elongated clusters and photographic textures
        # A true 3D paper tear is an elongated linear fracture (rip/crease).
        # Broad 2D textures (hat brims, skin highlights, garment folds) must NOT
        # be accepted merely because their area is large.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_tear, connectivity=8)
        min_tear_area = max(20, int(min(h, w) * 0.03))
        min_tear_len = max(20, int(min(h, w) * 0.035))
        clean_tear = np.zeros_like(combined_tear)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w_c = stats[i, cv2.CC_STAT_WIDTH]
            h_c = stats[i, cv2.CC_STAT_HEIGHT]
            length = max(w_c, h_c)
            aspect = length / max(1, min(w_c, h_c))
            elongation = (length * length) / max(1, area)
            # True tears must be elongated fractures, not broad 2D texture blobs
            if area >= min_tear_area and length >= min_tear_len and aspect >= 2.5 and elongation >= 2.8:
                clean_tear[labels == i] = 255

        # Scale-adaptive inpaint boundary overlap dilation
        k_dil_sz = max(3, int(7 * scale) | 1)
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dil_sz, k_dil_sz))
        final_tear = cv2.dilate(clean_tear, k_dil, iterations=1)

        # Protect ocular and oral zones: teeth paired with dark mouth cavity are not 3D paper tears
        faces = context.image_metadata.get("faces", [])
        for f in faces:
            lms = f.get("landmarks")
            bbox = f.get("bbox")
            if lms and len(lms) >= 5:
                for idx_eye in (0, 1):
                    ex, ey = int(round(lms[idx_eye][0])), int(round(lms[idx_eye][1]))
                    cv2.circle(final_tear, (ex, ey), max(8, int(min(h, w) * 0.035)), 0, -1)
                ml, mr = np.array(lms[3], dtype=np.float32), np.array(lms[4], dtype=np.float32)
                mc = (ml + mr) / 2.0
                mw = float(np.linalg.norm(mr - ml))
                ang = float(np.degrees(np.arctan2(mr[1] - ml[1], mr[0] - ml[0])))
                axes = (int(max(10, mw * 0.58)), int(max(6, mw * 0.28)))
                cv2.ellipse(final_tear, (int(round(mc[0])), int(round(mc[1]))), axes, ang, 0, 360, 0, -1)
            elif bbox and len(bbox) >= 4:
                bx0, by0, bx1, by1 = [int(v) for v in bbox[:4]]
                bfw, bfh = max(1, bx1 - bx0), max(1, by1 - by0)
                ex1, ey1 = int(bx0 + 0.32 * bfw), int(by0 + 0.38 * bfh)
                ex2, ey2 = int(bx0 + 0.68 * bfw), int(by0 + 0.38 * bfh)
                cv2.circle(final_tear, (ex1, ey1), max(8, int(min(bfw, bfh) * 0.08)), 0, -1)
                cv2.circle(final_tear, (ex2, ey2), max(8, int(min(bfw, bfh) * 0.08)), 0, -1)
                mx, my = int(bx0 + 0.50 * bfw), int(by0 + 0.76 * bfh)
                axes = (int(max(10, bfw * 0.22)), int(max(6, bfh * 0.10)))
                cv2.ellipse(final_tear, (mx, my), axes, 0, 0, 360, 0, -1)

        cov = float(np.sum(final_tear > 0) / final_tear.size * 100)
        if cov > 15.0:
            context.add_log(
                f"3DPaperTearDetector: {cov:.1f}% coverage exceeds 15% safety ceiling — "
                "response is dominated by photographic texture, suppressing false positive mask."
            )
            return np.zeros_like(final_tear)

        context.add_log(f"3DPaperTearDetector: {cov:.2f}% coverage.")
        return final_tear

    def detect_thin_crack_network(self, img: np.ndarray, context: PipelineContext) -> np.ndarray:
        """
        Detect fine branching crack networks, scratches, and structural tears using
        learned semantic defect segmentation (U-Net).
        """
        if self.use_defect_segmenter and self.defect_segmenter is None:
            try:
                self.defect_segmenter = DefectSegmenter()
            except Exception as exc:
                logger.warning(f"DefectSegmenter load error: {exc}")

        if self.defect_segmenter is not None and self.defect_segmenter.is_loaded:
            raw_mask = self.defect_segmenter.segment(img, prob_threshold=0.45, min_area=15)
            # Gentle 5x5 elliptical dilation to guarantee boundary overlap for inpainting
            k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(raw_mask, k_dil, iterations=1)
            cov = float(np.sum(mask > 0) / mask.size * 100)
            context.add_log(f"LearnedDefectDetector (U-Net): {cov:.2f}% coverage")
            return mask

        # ── Classical fallback if model weights are missing ──────────────────
        # This path is a safety net, so it is tuned to be conservative: a missed
        # crack leaves a visible line in the output, but a false positive gets
        # repainted by LaMa, and a false positive covering most of the frame
        # repaints the entire photograph. Under-detecting is the cheaper mistake.
        return self._thin_cracks_classical(img, context)

    # Minimum depth, in grey levels, of a black-hat response that can count as a
    # crack. A crack is a physical gap in the emulsion: it reads substantially
    # darker than the surface around it. Below this a response is ordinary tonal
    # variation. A perceptual floor, identical for every image.
    _MIN_CRACK_DEPTH = 22.0

    # A crack is thin. length^2 / area is the ratio of a component's longest
    # extent to its mean stroke width, so this asks for something at least four
    # times longer than it is wide — which excludes the dark blobs (eyes,
    # nostrils, shadow pockets, hair masses) that a 37px black-hat also responds
    # to just as strongly as it does to a fracture.
    _MIN_CRACK_ELONGATION = 4.0

    # A classical detector claiming this much of the frame is cracked has stopped
    # detecting cracks and started detecting content. Past this point the mask is
    # discarded rather than handed to the inpainter.
    _MAX_FALLBACK_COVERAGE = 0.20

    def _thin_cracks_classical(self, img: np.ndarray, context: PipelineContext) -> np.ndarray:
        """Multi-scale black-hat crack detection without a learned segmenter.

        Threshold rationale
        -------------------
        The previous threshold was a Sauvola-style expression:

            max(mean_bh * (1 + 0.22 * (std_bh / r_dyn - 1)), 8.5)

        with `r_dyn = max(std_bh)` taken over the *whole* image. Because r_dyn is
        a global maximum, `std_bh / r_dyn` is ~0 at essentially every pixel, so
        the bracket collapsed to (1 - 0.22) and the threshold became
        `0.78 * mean_bh` — "flag every pixel whose black-hat response exceeds 78%
        of its own local average". In a flat region the local average of a
        black-hat response is near zero, so that condition is satisfied by
        arbitrarily small fluctuations: it selects roughly half the frame before
        the dilations widen it. Measured coverage on inputs with no crack network
        at all was 85.68% on a synthetic gradient, 99.76% and 99.45% on two real
        photographs, and still 94.93% on a heavily pre-smoothed copy — which is
        the proof that the flooding was structural to the threshold rather than a
        reaction to noisy input.

        Sauvola is also the wrong tool here. It is a text-binarization method
        that assumes dark glyphs on a light page and normalises by local
        contrast. A black-hat response is already a contrast measure, and it is
        non-negative and sparse, so what it needs is an outlier criterion on its
        own distribution. This uses a global one, robust on both tails: mean +
        3 sigma catches the broad case, median + 6 MAD keeps a handful of very
        strong responses from inflating sigma and hiding the rest, and
        _MIN_CRACK_DEPTH floors both so a low-contrast image cannot drive the
        threshold down to noise. The same mean + k*sigma form is what
        detect_wide_cracks and detect_dark_spots in this module already use.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = gray.shape
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0.5)

        bh_stack = []
        for k_size in [9, 17, 27, 37]:
            k_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            bh = cv2.morphologyEx(
                gray_blur.astype(np.uint8), cv2.MORPH_BLACKHAT, k_elem
            ).astype(np.float32)
            bh_stack.append(bh)

        bh_fused = np.maximum.reduce(bh_stack)

        mean_bh = float(np.mean(bh_fused))
        std_bh = float(np.std(bh_fused))
        med_bh = float(np.median(bh_fused))
        mad_bh = float(np.median(np.abs(bh_fused - med_bh))) / 0.6745

        # Escalate the threshold if the first pass floods. Two attempts, then
        # give up — see the coverage guard at the end.
        mask = np.zeros((h, w), dtype=np.uint8)
        for sigma_k in (3.0, 5.0, 8.0):
            threshold = max(
                mean_bh + sigma_k * std_bh,
                med_bh + 2.0 * sigma_k * mad_bh,
                self._MIN_CRACK_DEPTH,
            )
            candidate = self._crack_components(
                (bh_fused >= threshold).astype(np.uint8) * 255, h, w
            )
            coverage = float(np.sum(candidate > 0)) / float(h * w)
            if coverage <= self._MAX_FALLBACK_COVERAGE:
                mask = candidate
                break
            context.add_log(
                f"ThinCrackNetworkDetector (classical): {coverage:.1%} coverage at "
                f"mean+{sigma_k:.0f}sigma (threshold {threshold:.1f}) exceeds the "
                f"{self._MAX_FALLBACK_COVERAGE:.0%} ceiling — raising the threshold."
            )
        else:
            # Every threshold flooded. The response is dominated by image content,
            # not by fractures, so there is nothing here worth inpainting. An
            # empty mask leaves cracks unrepaired; a flooding mask would have LaMa
            # repaint the photograph.
            context.add_log(
                "ThinCrackNetworkDetector (classical): response is dominated by "
                "image content at every threshold — returning an empty mask "
                "rather than sending most of the frame to the inpainter."
            )
            return np.zeros((h, w), dtype=np.uint8)

        cov = float(np.sum(mask > 0) / mask.size * 100.0)
        context.add_log(
            f"ThinCrackNetworkDetector (Classical Fallback): {cov:.2f}% coverage"
        )
        return mask

    def _crack_components(self, binary: np.ndarray, h: int, w: int) -> np.ndarray:
        """Keep only components shaped like cracks, and dilate each by its own
        stroke width.

        The previous code dilated every accepted component with an 11x11 ellipse
        and then any component over 300px again with a 19x19 ellipse. A 1px crack
        became an 11px band and a branching network became a solid region, which
        is how a plausible-looking thin mask turned into a frame-filling one.
        Inpainting only needs a small margin so the fill blends at the boundary,
        and that margin should scale with the defect: dilating a hairline by 5px
        on each side discards ten times more intact emulsion than it repairs.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        # Thin and wide cracks get different margins, so collect them separately
        # and dilate each set once — two morphology passes regardless of count.
        thin = np.zeros_like(binary)
        wide = np.zeros_like(binary)
        min_length = max(14, int(min(h, w) * 0.02))
        max_area = int(h * w * 0.02)

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            w_box = int(stats[i, cv2.CC_STAT_WIDTH])
            h_box = int(stats[i, cv2.CC_STAT_HEIGHT])
            length = max(w_box, h_box)

            if length < min_length or area > max_area:
                continue
            # length^2 / area ~ length / mean stroke width.
            if (length * length) / float(max(area, 1)) < self._MIN_CRACK_ELONGATION:
                continue

            stroke = area / float(max(length, 1))
            if stroke > 4.5:
                continue
            if stroke >= 3.0:
                wide[labels == i] = 255
            else:
                thin[labels == i] = 255

        out = np.zeros_like(binary)
        if np.any(thin):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            out = cv2.bitwise_or(out, cv2.dilate(thin, k, iterations=1))
        if np.any(wide):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            out = cv2.bitwise_or(out, cv2.dilate(wide, k, iterations=1))
        return out

    def detect_scratches(self, img: np.ndarray, context: PipelineContext) -> np.ndarray:
        """Detect linear scratches using multi-angle directional filtering."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]
        color_type = context.image_metadata.get("color_type", "color")

        kernels_3x3 = {
            0: np.array([[-1, -1, -1], [ 2,  2,  2], [-1, -1, -1]], dtype=np.float32),
            45: np.array([[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]], dtype=np.float32),
            90: np.array([[-1,  2, -1], [-1,  2, -1], [-1,  2, -1]], dtype=np.float32),
            135: np.array([[-1, -1,  2], [-1,  2, -1], [ 2, -1, -1]], dtype=np.float32),
        }

        scratch_masks = []
        for angle, kernel in kernels_3x3.items():
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            abs_filtered = np.abs(filtered)
            mean_val = float(np.mean(abs_filtered.astype(np.float32)))
            std_val = float(np.std(abs_filtered.astype(np.float32)))
            threshold = mean_val + 3.0 * std_val
            _, mask = cv2.threshold(abs_filtered, threshold, 255, cv2.THRESH_BINARY)
            scratch_masks.append(mask.astype(np.uint8))

        combined = np.zeros_like(gray, dtype=np.uint8)
        for mask in scratch_masks:
            combined = cv2.bitwise_or(combined, mask)

        min_brightness = 215 if color_type in ["monochrome", "grayscale"] else 180
        bright_mask = gray > min_brightness
        combined = cv2.bitwise_and(combined, bright_mask.astype(np.uint8) * 255)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            length = max(width, height)
            thickness = max(1, min(width, height))
            aspect_ratio = length / float(thickness)
            if area < 10 or aspect_ratio < 3.0 or length < 15:
                combined[labels == i] = 0

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.dilate(combined, k, iterations=1)
        return combined

    def detect_dark_spots(self, img: np.ndarray, context: PipelineContext) -> np.ndarray:
        """Detect small dark spots (dust, mold, ink dots) using black-hat filtering."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]

        kernel_size = max(5, int(min(h, w) * 0.008) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        mean_val = float(np.mean(blackhat.astype(np.float32)))
        std_val = float(np.std(blackhat.astype(np.float32)))
        threshold = mean_val + 3.5 * std_val
        _, spot_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
        spot_mask = spot_mask.astype(np.uint8)

        dark_pixels = gray < 60
        spot_mask = cv2.bitwise_and(spot_mask, dark_pixels.astype(np.uint8) * 255)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(spot_mask, connectivity=8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            circularity = min(width, height) / max(1, max(width, height))
            if area < 4 or area > 100 or circularity < 0.3:
                spot_mask[labels == i] = 0

        return spot_mask

    def detect_dust_spots(self, img: np.ndarray, context: PipelineContext) -> np.ndarray:
        """Alias for detect_dark_spots for API compatibility."""
        return self.detect_dark_spots(img, context)

    def detect_edge_damage(self, img: np.ndarray, context: PipelineContext) -> np.ndarray:
        """Detect genuine physical damage near photo borders."""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        noise_score = float(context.image_metadata.get("noise_score", 0.0))

        if noise_score > 6.0:
            gray = cv2.medianBlur(gray, 3)
            bright_thresh = max(55, int(45 + (noise_score - 6.0) * 2.0))
            dark_thresh = max(60, int(50 + (noise_score - 6.0) * 2.0))
            min_area = max(20, int(noise_score * 2.5))
        else:
            bright_thresh = 45
            dark_thresh = 50
            min_area = 12

        border_width = max(4, int(min(h, w) * 0.015))
        edge_mask = np.zeros((h, w), dtype=np.uint8)
        edge_mask[:border_width, :] = 255
        edge_mask[-border_width:, :] = 255
        edge_mask[:, :border_width] = 255
        edge_mask[:, -border_width:] = 255

        kernel_size = max(7, int(min(h, w) * 0.01) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        _, bright_mask = cv2.threshold(tophat, bright_thresh, 255, cv2.THRESH_BINARY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, dark_mask = cv2.threshold(blackhat, dark_thresh, 255, cv2.THRESH_BINARY)

        damage_mask = cv2.bitwise_or(bright_mask.astype(np.uint8), dark_mask.astype(np.uint8))
        edge_damage = cv2.bitwise_and(damage_mask, edge_mask)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_damage, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                edge_damage[labels == i] = 0

        return edge_damage

    def combined_damage_mask(self, img: np.ndarray, context: PipelineContext) -> np.ndarray:
        """Alias for detect() that computes the full multi-category damage mask."""
        return self.detect(context)

    def detect(self, context: PipelineContext) -> np.ndarray:
        """Main detection method — combines all damage types."""
        img = context.working_image
        flakes = self.detect_white_flakes(img, context)
        scratches = self.detect_scratches(img, context)
        wide_cracks = self.detect_wide_cracks(img, context)
        tears_3d = self.detect_3d_paper_tears(img, context)
        thin_cracks = self.detect_thin_crack_network(img, context)
        spots = self.detect_dark_spots(img, context)
        edge_damage = self.detect_edge_damage(img, context)

        combined = cv2.bitwise_or(flakes, scratches)
        combined = cv2.bitwise_or(combined, wide_cracks)
        combined = cv2.bitwise_or(combined, tears_3d)
        combined = cv2.bitwise_or(combined, thin_cracks)
        combined = cv2.bitwise_or(combined, spots)
        combined = cv2.bitwise_or(combined, edge_damage)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

        # ── Face-Region False-Positive Guard ─────────────────────────────────
        # Legitimate deep tears crossing a face are elongated, linear structures.
        # Dense image noise and facial contours (shadows, eyes, nostrils) generate
        # small, low-contrast, non-linear blob false-positives. Filter them out inside
        # any detected face bounding box so downstream inpainting doesn't erase facial anatomy.
        faces = context.image_metadata.get("faces", [])
        h_img, w_img = combined.shape[:2]
        pruned_face_px = 0
        if faces:
            for f in faces:
                bbox = f.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
                x0 = max(0, x0); y0 = max(0, y0)
                x1 = min(w_img, x1); y1 = min(h_img, y1)
                fw = x1 - x0; fh = y1 - y0
                if fw <= 0 or fh <= 0:
                    continue

                face_roi = combined[y0:y1, x0:x1]
                if not np.any(face_roi > 0):
                    continue

                num_f, lbl_f, stats_f, _ = cv2.connectedComponentsWithStats(face_roi, connectivity=8)
                min_face_crack_length = max(10, int(min(fw, fh) * 0.15))

                for fi in range(1, num_f):
                    area_fi = stats_f[fi, cv2.CC_STAT_AREA]
                    w_fi = stats_f[fi, cv2.CC_STAT_WIDTH]
                    h_fi = stats_f[fi, cv2.CC_STAT_HEIGHT]
                    len_fi = max(w_fi, h_fi)
                    aspect_fi = len_fi / max(1, min(w_fi, h_fi))
                    elongation_fi = (len_fi * len_fi) / max(1, area_fi)

                    # True tears crossing a face are thin and elongated. Reject small or round/blobby noise detections.
                    is_valid_face_crack = (len_fi >= min_face_crack_length) and (aspect_fi >= 2.5 or elongation_fi >= 3.0)
                    if not is_valid_face_crack:
                        face_roi[lbl_f == fi] = 0
                        pruned_face_px += area_fi

                combined[y0:y1, x0:x1] = face_roi

        if pruned_face_px > 0:
            context.add_log(
                f"EnhancedDamageDetector: Pruned {pruned_face_px} false-positive noise/feature px inside face region(s)."
            )

        coverage = float(np.sum(combined > 0) / combined.size * 100.0)
        context.add_log(f"EnhancedDamageDetector: {coverage:.2f}% coverage.")
        context.damage_masks["enhanced"] = combined
        return combined
