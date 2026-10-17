"""
color_refiner.py — Advanced color refinement and artifact reduction.

This module addresses color artifacts seen in current results:
1. Color banding in smooth gradients
2. Unnatural saturation in certain areas
3. Pixelation in color transitions
4. Color bleeding at edges

Key improvements:
- Multi-stage color smoothing
- Anti-banding with noise dithering
- Smart saturation adjustment
- Edge-aware color blending
"""

import os

import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from ai.pipeline.pipeline_context import PipelineContext


class BandingReducer:
    """
    Reduces color banding artifacts in smooth gradients.
    """
    
    @staticmethod
    def detect_banding(img: np.ndarray, threshold: float = 0.02) -> np.ndarray:
        """
        Detect areas with color banding.
        
        Banding appears in smooth gradient areas with low variance
        but visible color steps.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = lab[:, :, 0]
        
        # Calculate local variance
        L_f32 = L.astype(np.float32)
        blurred = cv2.GaussianBlur(L_f32, (5, 5), 0)
        var_L = cv2.GaussianBlur(((L_f32 - blurred) ** 2).astype(np.float32), (5, 5), 0)
        
        # Smooth areas with low variance are prone to banding
        smooth_mask = var_L < 10
        
        # Check for visible steps (high second derivative)
        grad_L = (cv2.Sobel(L_f32, cv2.CV_32F, 1, 0, ksize=3) + cv2.Sobel(L_f32, cv2.CV_32F, 0, 1, ksize=3)).astype(np.float32)
        grad2_L = cv2.Sobel(grad_L, cv2.CV_32F, 1, 0, ksize=3) + cv2.Sobel(grad_L, cv2.CV_32F, 0, 1, ksize=3)
        banding_indicator = np.abs(grad2_L) > threshold * 255
        
        banding_mask = smooth_mask & banding_indicator
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        banding_mask = cv2.morphologyEx(
            banding_mask.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE, kernel, iterations=1
        ) > 0
        
        return banding_mask
    
    @staticmethod
    def reduce_banding(img: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reduce color banding by applying soft continuous smoothing in detected banding zones.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # If no mask provided, detect banding automatically
        if mask is None:
            mask = BandingReducer.detect_banding(img)
        
        if not np.any(mask):
            return img

        # Continuous blurred weight map to avoid hard boundaries
        mask_f = mask.astype(np.float32)
        mask_soft = cv2.GaussianBlur(mask_f, (9, 9), 0)
        if not np.any(mask_soft > 0.01):
            return img

        # Only smooth colour channels (A, B) — touching L destroys grain/texture.
        # Keep blend cap at 0.15 (was 0.30) to stay conservative and preserve texture.
        smoothed_A = cv2.GaussianBlur(lab[:, :, 1], (5, 5), 0)
        smoothed_B = cv2.GaussianBlur(lab[:, :, 2], (5, 5), 0)

        blend = np.clip(mask_soft * 0.15, 0.0, 1.0)
        lab[:, :, 1] = lab[:, :, 1] * (1.0 - blend) + smoothed_A * blend
        lab[:, :, 2] = lab[:, :, 2] * (1.0 - blend) + smoothed_B * blend

        lab = np.clip(lab, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


class ColorSmoother:
    """
    Smooths color transitions and reduces pixelation artifacts.
    """
    
    @staticmethod
    def smooth_colors(img: np.ndarray, 
                      strength: float = 0.3,
                      mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply edge-preserving color smoothing.
        
        Uses bilateral filter in LAB space to smooth colors
        while preserving edges.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Apply bilateral filter to A and B channels (color channels)
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        
        # Create edge mask for preservation
        if mask is None:
            # Use luminance edges to guide smoothing
            l_channel = lab[:, :, 0]
            edges = cv2.Canny(np.clip(l_channel, 0, 255).astype(np.uint8), 30, 100)
            edge_mask = edges > 0
        else:
            edge_mask = mask
        
        # Apply bilateral filter
        a_smoothed = cv2.bilateralFilter(
            a_channel.astype(np.uint8), 
            d=5, sigmaColor=15, sigmaSpace=5
        ).astype(np.float32)
        b_smoothed = cv2.bilateralFilter(
            b_channel.astype(np.uint8), 
            d=5, sigmaColor=15, sigmaSpace=5
        ).astype(np.float32)
        
        # Blend with original based on edge mask
        if np.any(edge_mask):
            a_final = np.where(edge_mask, a_channel, 
                              a_channel * (1 - strength) + a_smoothed * strength)
            b_final = np.where(edge_mask, b_channel,
                              b_channel * (1 - strength) + b_smoothed * strength)
        else:
            a_final = a_channel * (1 - strength) + a_smoothed * strength
            b_final = b_channel * (1 - strength) + b_smoothed * strength
        
        lab[:, :, 1] = a_final
        lab[:, :, 2] = b_final
        
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        return result


class EdgeAwareColorBleedFixer:
    """
    Fix color bleeding caused by low-resolution chrominance prediction.
    
    ECCV16 produces AB color channels at ~56×56 resolution, upscaled to full
    resolution. This causes color to bleed across luminance edges (e.g., hair
    color bleeds into background, skin tone bleeds into clothing).
    
    Solution: Use the high-resolution luminance (L) channel as a guide to
    constrain chrominance (A, B) channels via joint bilateral filtering.
    The L channel preserves sharp edges that the AB channels lost during
    upscaling.
    """
    
    @staticmethod
    def fix_bleeding(img: np.ndarray, 
                     context: Optional[PipelineContext] = None,
                     strength: float = 0.7) -> np.ndarray:
        """
        Apply luminance-guided chrominance edge correction.
        
        Args:
            img: RGB uint8 image (post-colorization)
            context: Pipeline context for logging
            strength: Blend factor (0 = no fix, 1 = full fix)
            
        Returns:
            RGB uint8 image with reduced color bleeding
        """
        h, w = img.shape[:2]
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = lab[:, :, 0]
        A = lab[:, :, 1]
        B = lab[:, :, 2]
        
        # 1. Detect luminance edges — these are the boundaries where
        #    color should NOT bleed across
        edges = cv2.Canny(np.clip(L, 0, 255).astype(np.uint8), 30, 80)
        
        # Dilate edges to create a "boundary zone" where we clamp chrominance
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_zone = cv2.dilate(edges, edge_kernel, iterations=1)
        edge_mask = edge_zone > 0
        
        if not np.any(edge_mask):
            return img
        
        # 2. For each edge pixel, compute the chrominance of each side
        #    and clamp edge pixels to match the side they're on based on
        #    luminance similarity.
        
        # Use a large guided bilateral filter: sigma_color is LOW (tight
        # in luminance space) and sigma_space is moderate. This smooths
        # chrominance only among pixels with similar luminance.
        # Adaptive kernel size based on image resolution
        d = max(7, min(15, int(min(h, w) * 0.01) | 1))
        sigma_color = 20.0   # Tight: only mix chrominance between luminance-similar pixels
        sigma_space = max(7.0, min(h, w) * 0.005)
        
        # Joint bilateral filter: filter A and B using L as the guide
        L_u8 = np.clip(L, 0, 255).astype(np.uint8)
        A_u8 = np.clip(A, 0, 255).astype(np.uint8)
        B_u8 = np.clip(B, 0, 255).astype(np.uint8)
        
        # Fix 4: Use luminance-weighted separable passes as fallback to approximate
        # joint bilateral behavior.  Plain bilateral without a luma guide bleeds across
        # luminance edges rather than along them.
        if hasattr(cv2, 'ximgproc'):
            A_filtered = cv2.ximgproc.jointBilateralFilter(
                L_u8, A_u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space
            ).astype(np.float32)
            B_filtered = cv2.ximgproc.jointBilateralFilter(
                L_u8, B_u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space
            ).astype(np.float32)
        else:
            A_filtered = cv2.bilateralFilter(A_u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space).astype(np.float32)
            B_filtered = cv2.bilateralFilter(B_u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space).astype(np.float32)
            # Pull back to original at strong luminance edges to prevent cross-boundary bleed
            _edges_strong = cv2.Canny(L_u8, 40, 120)
            _edge_supp = cv2.GaussianBlur((_edges_strong > 0).astype(np.float32), (5, 5), 0)
            A_filtered = A_filtered * (1.0 - _edge_supp) + A.astype(np.float32) * _edge_supp
            B_filtered = B_filtered * (1.0 - _edge_supp) + B.astype(np.float32) * _edge_supp
        
        # 3. Apply correction primarily in the edge boundary zone,
        #    with gentle correction in non-edge areas
        edge_mask_f = edge_mask.astype(np.float32)
        
        # In edge zones: stronger correction (pull color to luminance-guided value)
        # Outside edge zones: gentle correction (prevent large-scale color drift)
        edge_strength = strength
        # Fix 4: Reduced non-edge strength 0.2x → 0.08x to prevent global chroma drift.
        non_edge_strength = strength * 0.08
        
        blend_map = np.where(edge_mask_f > 0, edge_strength, non_edge_strength)
        
        A_out = A * (1.0 - blend_map) + A_filtered * blend_map
        B_out = B * (1.0 - blend_map) + B_filtered * blend_map
        
        lab[:, :, 1] = A_out
        lab[:, :, 2] = B_out
        
        result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        if context:
            edge_pct = float(np.sum(edge_mask) / edge_mask.size * 100)
            context.add_log(
                f"EdgeAwareColorBleedFixer: corrected bleeding across {edge_pct:.1f}% "
                f"edge pixels (d={d}, strength={strength:.2f})"
            )
        
        return result


class ChromaOutlierSuppressor:
    """
    Suppresses isolated color splotches, rust/orange patches on dark or neutral surfaces,
    and unnatural saturated blobs produced by deep colorization models.
    """
    @staticmethod
    def suppress_splotches(
        img: np.ndarray,
        context: Optional[PipelineContext] = None,
        strength: float = 0.85,
    ) -> np.ndarray:
        h, w = img.shape[:2]
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = lab[:, :, 0]
        A = lab[:, :, 1]
        B = lab[:, :, 2]

        L_u8 = np.clip(L, 0, 255).astype(np.uint8)
        A_u8 = np.clip(A, 0, 255).astype(np.uint8)
        B_u8 = np.clip(B, 0, 255).astype(np.uint8)

        # 1. Luminance-guided joint bilateral filter on chroma
        # d≤9 / sigma_s tighter: suppress hard splotches only, not fine texture (was d≤15).
        d = max(5, min(9, int(min(h, w) * 0.010) | 1))
        sigma_c = 10.0
        sigma_s = max(4.0, min(h, w) * 0.010)

        if hasattr(cv2, "ximgproc"):
            A_smooth = cv2.ximgproc.jointBilateralFilter(L_u8, A_u8, d=d, sigmaColor=sigma_c, sigmaSpace=sigma_s).astype(np.float32)
            B_smooth = cv2.ximgproc.jointBilateralFilter(L_u8, B_u8, d=d, sigmaColor=sigma_c, sigmaSpace=sigma_s).astype(np.float32)
        else:
            A_smooth = cv2.bilateralFilter(A_u8, d=d, sigmaColor=sigma_c, sigmaSpace=sigma_s).astype(np.float32)
            B_smooth = cv2.bilateralFilter(B_u8, d=d, sigmaColor=sigma_c, sigmaSpace=sigma_s).astype(np.float32)

        # 2. Identify face zones to protect natural skin tones
        face_mask = np.zeros((h, w), dtype=np.float32)
        if context:
            faces = context.image_metadata.get("faces", [])
            for face in faces:
                bbox = face.get("bbox")
                if bbox and len(bbox) >= 4:
                    x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
                    cx = (x0 + x1) // 2
                    cy = (y0 + y1) // 2
                    ax_w = max(10, int((x1 - x0) * 0.50))
                    ax_h = max(10, int((y1 - y0) * 0.60))
                    cv2.ellipse(face_mask, (cx, cy), (ax_w, ax_h), 0, 0, 360, 1.0, -1)
            if np.any(face_mask > 0):
                face_mask = cv2.GaussianBlur(face_mask, (15, 15), 0)

        # 3. Suppress chroma only for NON-DARK pixels — dark hair/clothing (L < 80)
        #    are intentional color deviations and must NOT be treated as outliers.
        # Fix 2b: Dark pixel protection — allow hair to differ from bilateral mean.
        dark_pixel_guard = np.clip((L_u8.astype(np.float32) - 55.0) / 35.0, 0.0, 1.0)  # 0.0 for L<55, 1.0 for L>90
        chroma_dev = np.sqrt((A - A_smooth) ** 2 + (B - B_smooth) ** 2)
        # Reduced cap: 0.12 instead of 0.25, threshold 36.0 instead of 24.0 — only pull true splotches toward smooth,
        # leave normal chroma variation (fine texture, hair, skin tones) intact.
        outlier_weight = np.clip(chroma_dev / 36.0, 0.0, 0.12) * strength * dark_pixel_guard

        A_out = A * (1.0 - outlier_weight) + A_smooth * outlier_weight
        B_out = B * (1.0 - outlier_weight) + B_smooth * outlier_weight

        lab[:, :, 1] = np.clip(A_out, 0, 255)
        lab[:, :, 2] = np.clip(B_out, 0, 255)

        if context:
            context.add_log("ChromaOutlierSuppressor: preserved authentic full-scene colorization.")

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


class FacialSkinToneProtector:
    """
    Multi-face landmark-gated skin tone protection.
    
    DDColor applies a global color transform without per-face landmark guidance,
    which can produce unnatural cyan, green, or sickly blue casts on facial skin,
    especially in multi-face or mixed-lighting portraits.
    
    This class creates soft elliptical skin masks per detected face, excludes
    ocular (eyes) and oral (lips, teeth) landmark zones, and gently guides
    unnatural chrominance back into authentic portrait skin bounds in LAB space:
      a* in [130, 170] (healthy warm/red tones, avoiding green/cyan casts)
      b* in [130, 175] (natural melanin/yellow tones, avoiding cold blue casts)
    with smooth Gaussian boundary feathering.
    """

    @staticmethod
    def protect_skin_tones(
        img: np.ndarray,
        faces: Optional[List[Dict[str, Any]]] = None,
        context: Optional[PipelineContext] = None,
        strength: float = 0.40,
    ) -> np.ndarray:
        if faces is None and context is not None:
            faces = context.image_metadata.get("faces", [])
        if not faces:
            return img

        h, w = img.shape[:2]
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = lab[:, :, 0]
        A = lab[:, :, 1]
        B = lab[:, :, 2]

        total_skin_mask = np.zeros((h, w), dtype=np.float32)

        for face in faces:
            bbox = face.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
            x0 = max(0, min(w, x0))
            x1 = max(0, min(w, x1))
            y0 = max(0, min(h, y0))
            y1 = max(0, min(h, y1))
            fw = x1 - x0
            fh = y1 - y0
            if fw < 16 or fh < 16:
                continue

            face_skin = np.zeros((h, w), dtype=np.float32)
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            ax_w = max(8, int(fw * 0.40))
            ax_h = max(8, int(fh * 0.48))
            cv2.ellipse(face_skin, (cx, cy), (ax_w, ax_h), 0, 0, 360, 1.0, -1)

            # Exclude ocular & oral landmarks if present
            landmarks = face.get("landmarks")
            if landmarks and len(landmarks) >= 5:
                # Eyes (indices 0 and 1)
                r_eye = max(4, int(min(fw, fh) * 0.09))
                for idx_eye in (0, 1):
                    ex = int(round(landmarks[idx_eye][0]))
                    ey = int(round(landmarks[idx_eye][1]))
                    if 0 <= ex < w and 0 <= ey < h:
                        cv2.circle(face_skin, (ex, ey), r_eye, 0.0, -1)

                # Mouth (indices 3 and 4)
                ml = np.array([landmarks[3][0], landmarks[3][1]], dtype=np.float32)
                mr = np.array([landmarks[4][0], landmarks[4][1]], dtype=np.float32)
                mc = (ml + mr) / 2.0
                mw = float(np.linalg.norm(mr - ml))
                ang = float(np.degrees(np.arctan2(mr[1] - ml[1], mr[0] - ml[0])))
                cv2.ellipse(
                    face_skin,
                    (int(round(mc[0])), int(round(mc[1]))),
                    (int(max(8, mw * 0.55)), int(max(5, mw * 0.28))),
                    ang,
                    0,
                    360,
                    0.0,
                    -1,
                )

            # Smooth individual face skin mask
            k_smooth = max(7, int(min(fw, fh) * 0.10) | 1)
            face_skin = cv2.GaussianBlur(face_skin, (k_smooth, k_smooth), 0)
            total_skin_mask = np.maximum(total_skin_mask, face_skin)

        if not np.any(total_skin_mask > 0.01):
            return img

        # Target natural skin bounds in OpenCV LAB space (neutral=128):
        # A: [130, 170]
        # B: [130, 175]
        # Detect pixels within the skin mask that fall outside natural skin chrominance
        # (e.g. green/cyan casts: A < 128; cold blue casts: B < 128; over-saturated magenta: A > 175)
        unnatural_chroma = (A < 128.0) | (B < 128.0) | (A > 175.0) | (B > 180.0)

        # Only adjust pixels that have some luminance (avoid crushed shadows or hair)
        luma_valid = (L > 45.0) & (L < 240.0)
        active_correction_mask = total_skin_mask * unnatural_chroma.astype(np.float32) * luma_valid.astype(np.float32)

        if not np.any(active_correction_mask > 0.01):
            return img

        # Soft blend factor
        blend = np.clip(active_correction_mask * strength, 0.0, 1.0)

        target_A = np.clip(A, 132.0, 165.0)
        target_B = np.clip(B, 134.0, 170.0)

        A_out = A * (1.0 - blend) + target_A * blend
        B_out = B * (1.0 - blend) + target_B * blend

        lab[:, :, 1] = np.clip(A_out, 0, 255)
        lab[:, :, 2] = np.clip(B_out, 0, 255)

        if context:
            affected_px = float(np.sum(blend > 0.05) / blend.size * 100)
            context.add_log(
                f"FacialSkinToneProtector: protected skin tones across {len(faces)} face(s) "
                f"({affected_px:.1f}% pixels corrected, strength={strength:.2f})"
            )

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


class BackgroundNeutralizer:
    """
    Suppresses unnatural global chromatic washes (such as pink/mauve/purple casts)
    in non-subject background regions of colorized photographs.
    
    Ensures that background walls, poles, and framing retain neutral gray/beige/taupe
    tones, while fully preserving subject skin tones, warm straw/tan hats (where b* >= a*),
    and dark hair.
    """

    @staticmethod
    def neutralize_background(
        img: np.ndarray,
        faces: Optional[List[Dict[str, Any]]] = None,
        context: Optional[PipelineContext] = None,
        strength: float = 0.60,
    ) -> np.ndarray:
        if faces is None and context is not None:
            faces = context.image_metadata.get("faces", [])

        h, w = img.shape[:2]
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = lab[:, :, 0]
        A = lab[:, :, 1]
        B = lab[:, :, 2]

        # 1. Build subject protection mask (faces + expanded neck/shoulder region)
        subject_mask = np.zeros((h, w), dtype=np.float32)
        if faces:
            for face in faces:
                bbox = face.get("bbox")
                if bbox and len(bbox) >= 4:
                    x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(w, x1), min(h, y1)
                    fw, fh = x1 - x0, y1 - y0
                    # Face oval
                    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                    cv2.ellipse(subject_mask, (cx, cy), (int(fw * 0.65), int(fh * 0.70)), 0, 0, 360, 1.0, -1)
                    # Neck and upper torso/shoulder downward extension
                    shoulder_y1 = min(h, y1 + int(fh * 0.90))
                    shoulder_x0 = max(0, x0 - int(fw * 0.50))
                    shoulder_x1 = min(w, x1 + int(fw * 0.50))
                    cv2.rectangle(subject_mask, (shoulder_x0, y0 + int(fh * 0.5)), (shoulder_x1, shoulder_y1), 0.85, -1)
                    # Headwear, hat, and hair upward extension
                    hat_y0 = max(0, y0 - int(fh * 1.30))
                    hat_x0 = max(0, x0 - int(fw * 0.70))
                    hat_x1 = min(w, x1 + int(fw * 0.70))
                    cv2.rectangle(subject_mask, (hat_x0, hat_y0), (hat_x1, y0 + int(fh * 0.3)), 0.85, -1)

            if np.any(subject_mask > 0):
                k_smooth = max(15, int(min(h, w) * 0.08) | 1)
                subject_mask = cv2.GaussianBlur(subject_mask, (k_smooth, k_smooth), 0)

        # 2. Warm object protection: straw/tan hats, wood, gold jewelry where b* > a* + 2
        # (yellow/amber dominant chrominance is intentional scene content, not mauve wash)
        warm_object_mask = ((B > A + 2.0) & (B > 131.0)).astype(np.float32)
        if np.any(warm_object_mask > 0):
            warm_object_mask = cv2.GaussianBlur(warm_object_mask, (9, 9), 0)

        # 3. Detect unnatural magenta/purple/mauve cast in background:
        # Magenta/pink wash: A > 136 and A > B + 2.0
        unnatural_bg_cast = (A > 136.0) & (A > B + 2.0)

        # Eligible background pixels: not in subject mask, not in warm object mask
        bg_weight = np.clip((1.0 - subject_mask) * (1.0 - warm_object_mask * 0.85), 0.0, 1.0)
        correction_weight = bg_weight * unnatural_bg_cast.astype(np.float32) * strength

        if not np.any(correction_weight > 0.01):
            return img

        # Target neutral chromatic center (128, 128) with slight warm-gray offset (128, 129.5)
        target_A = 128.0
        target_B = 129.5

        A_out = A * (1.0 - correction_weight) + target_A * correction_weight
        B_out = B * (1.0 - correction_weight) + target_B * correction_weight

        lab[:, :, 1] = np.clip(A_out, 0, 255)
        lab[:, :, 2] = np.clip(B_out, 0, 255)

        if context:
            corrected_pct = float(np.sum(correction_weight > 0.05) / correction_weight.size * 100)
            context.add_log(f"BackgroundNeutralizer: neutral gray-beige tone restored to background ({corrected_pct:.1f}% px).")

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


class ColorArtifactReducer:
    """
    Main class for reducing color artifacts in colorized imagery.
    """
    
    @staticmethod
    def reduce_artifacts(img: np.ndarray, 
                        context: Optional[PipelineContext] = None,
                        enable_bleed_reduction: bool = True,
                        enable_splotch_suppression: bool = True,
                        enable_banding_reduction: bool = True,
                        enable_color_cast_fix: bool = True,
                        enable_skin_protection: bool = True,
                        enable_background_neutralization: bool = True,
                        faces: Optional[List[Dict[str, Any]]] = None,
                        **kwargs) -> np.ndarray:
        """
        Apply essential color artifact reduction without geometric synthetic masks.
        """
        result = img.copy()

        # G3 (docs/REMEDIATION_PLAN_AB_G.md): luminance-guided chroma
        # alignment — smooth a/b with L* as the guided-filter guide so hue
        # changes can only occur where luminance changes (stops color bleeding
        # across luminance edges; sharpens chin/neck and collar/shirt hue
        # separation). PIXREVIVE_CHROMA_DISCIPLINE=0 reverts.
        if os.environ.get("PIXREVIVE_CHROMA_DISCIPLINE", "1").strip().lower() not in ("0", "false", "no", "off"):
            try:
                lab_g = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(np.float32)
                l_g = lab_g[:, :, 0].astype(np.uint8)
                a_f = cv2.ximgproc.guidedFilter(
                    guide=l_g, src=lab_g[:, :, 1].astype(np.float32), radius=8, eps=600.0)
                b_f = cv2.ximgproc.guidedFilter(
                    guide=l_g, src=lab_g[:, :, 2].astype(np.float32), radius=8, eps=600.0)
                # Blend 60% of the guided (edge-aligned) chroma back.
                lab_g[:, :, 1] = np.clip(0.4 * lab_g[:, :, 1] + 0.6 * a_f, 0, 255)
                lab_g[:, :, 2] = np.clip(0.4 * lab_g[:, :, 2] + 0.6 * b_f, 0, 255)
                result = cv2.cvtColor(lab_g.astype(np.uint8), cv2.COLOR_LAB2RGB)
            except Exception:
                pass  # ximgproc unavailable — the bleed fixer below still runs

        # Step 0: Edge-aware color bleed reduction (joint bilateral filter guided by L*)
        if enable_bleed_reduction:
            result = EdgeAwareColorBleedFixer.fix_bleeding(result, context=context, strength=0.75)
        
        # Step 1: Chroma outlier & splotch suppression (luminance-guided)
        if enable_splotch_suppression:
            result = ChromaOutlierSuppressor.suppress_splotches(result, context=context, strength=0.85)

        # Step 2: Continuous color banding reduction
        if enable_banding_reduction:
            result = BandingReducer.reduce_banding(result)

        # Step 3: Fix unnatural magenta/purple casts with soft feathered face protection
        if enable_color_cast_fix:
            result = ColorArtifactReducer.fix_color_cast(result, strength=0.40, context=context)
            # Step 3b: whole-image cast neutralization (salvaged from Enhancer,
            # D17/T-7) — removes DDColor's subtle global hue bias while
            # preserving naturally warm portraits via the |avg-128| <= 10 skip.
            result = ColorArtifactReducer.neutralize_global_cast(result, strength=0.6, context=context)
            
        # Step 4: Multi-face landmark-guided skin tone protection
        if enable_skin_protection:
            result = FacialSkinToneProtector.protect_skin_tones(result, faces=faces, context=context, strength=0.40)

        # Step 5: Background chromatic wash neutralization
        if enable_background_neutralization:
            result = BackgroundNeutralizer.neutralize_background(result, faces=faces, context=context, strength=0.60)
        
        return result


    
    @staticmethod
    def fix_color_cast(img: np.ndarray, 
                       strength: float = 0.40,
                       context: Optional[PipelineContext] = None) -> np.ndarray:
        """
        Fix color casts (magenta/purple/cyan leakage) using continuous feathered face protection.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        h, w = img.shape[:2]
        
        A = lab[:, :, 1]
        B = lab[:, :, 2]
        
        # Soft continuous face protection weight
        face_weight = np.zeros((h, w), dtype=np.float32)
        if context:
            faces = context.image_metadata.get("faces", [])
            for face in faces:
                bbox = face.get("bbox")
                if bbox and len(bbox) >= 4:
                    x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
                    x0 = max(0, min(w, x0))
                    x1 = max(0, min(w, x1))
                    y0 = max(0, min(h, y0))
                    y1 = max(0, min(h, y1))
                    face_weight[y0:y1, x0:x1] = 1.0
            if np.any(face_weight > 0):
                k_smooth = max(15, int(min(h, w) * 0.05) | 1)
                face_weight = cv2.GaussianBlur(face_weight, (k_smooth, k_smooth), 0)

        # Detect unnatural magenta, violet, neon purple, and neon cyan casts
        extreme_purple = (A > 140) & (B < 125)
        extreme_magenta = (A > 145) & (A > B + 8)
        extreme_cyan = (A < 112) & (B < 120)
        
        raw_cast_mask = (extreme_purple | extreme_magenta | extreme_cyan).astype(np.float32)
        
        if np.any(raw_cast_mask > 0):
            # Smooth cast transition to prevent hard boundary contours
            cast_soft = cv2.GaussianBlur(raw_cast_mask, (11, 11), 0)
            eff_strength = np.clip(cast_soft * (1.0 - face_weight) * strength, 0.0, 1.0)
            
            if np.any(eff_strength > 0.01):
                if context:
                    cast_ratio = float(np.sum(eff_strength > 0.05) / eff_strength.size) * 100
                    context.add_log(f"ColorArtifactReducer: feathered color cast fix ({cast_ratio:.1f}% pixels, strength={strength:.2f})")
                    if hasattr(context, "debug_masks"):
                        context.debug_masks["color_cast_weight"] = eff_strength
                
                A = A * (1.0 - eff_strength) + 128.0 * eff_strength
                B = B * (1.0 - eff_strength) + 128.0 * eff_strength
                
                lab[:, :, 1] = np.clip(A, 0, 255)
                lab[:, :, 2] = np.clip(B, 0, 255)
        
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Phase-1 repair (2026-10-12): the L-weighted subtract can overshoot.
    # Measured on image1 (sepia-flaked portrait): input bg a*=-1.6 b*=+17.5
    # (a modest paper warmth), avgB = 138 > 138-10 = sanit gate fires; after
    # a full L-weighted correction the FINAL canvas lands on bg b*=-12.5 —
    # the paper goes COLD (lavender), the exact opposite defect, because the
    # grey print's chroma is tiny-heterogeneous and the L-weighting
    # amplifies bright zones. Cap the per-channel correction at 60% of the
    # measured cast so a modest warmth can never invert sign; measured:
    # with cap, image1 bg lands b*≈+11 (warm, natural) instead of -12.5.
    # PIXREVIVE_CAST_MAX_FRAC overrides (1.0 = uncapped legacy).
    CAST_MAX_FRAC_DEFAULT = 0.60

    @staticmethod
    def neutralize_global_cast(img: np.ndarray,
                               strength: float = 0.7,
                               context: Optional[PipelineContext] = None) -> np.ndarray:
        """Neutralise a whole-image color cast in LAB space.

        Salvaged from the deleted `Enhancer.correct_color_cast` (D17) so the
        logic survives for T-7 channel-curve rebalancing. Complements
        `fix_color_cast`, which only clips extreme *local* casts: this removes
        a global hue bias (e.g. DDColor's warm shift) with luminance-weighted
        partial correction, and skips silently when the average is within the
        natural warm-portrait range (|avg - 128| <= 10 on a or b).

        Phase-1 (see CAST_MAX_FRAC_DEFAULT) caps the actual applied shift to
        a fraction of the measured cast, so a modest paper warmth never inverts
        into a cold cast on the other side.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        avg_a = float(np.mean(lab[:, :, 1]))
        avg_b = float(np.mean(lab[:, :, 2]))

        cast_a = avg_a - 128.0
        cast_b = avg_b - 128.0

        # Skip correction if cast is within natural warm-portrait range
        if abs(cast_a) <= 10.0 and abs(cast_b) <= 10.0:
            return img

        try:
            max_frac = float(os.environ.get("PIXREVIVE_CAST_MAX_FRAC", "")) if os.environ.get("PIXREVIVE_CAST_MAX_FRAC") else ColorArtifactReducer.CAST_MAX_FRAC_DEFAULT
        except (TypeError, ValueError):
            max_frac = ColorArtifactReducer.CAST_MAX_FRAC_DEFAULT
        max_frac = float(np.clip(max_frac, 0.0, 1.0))

        # Partial correction capped at max_frac of the measured cast — the
        # applied channel shift cannot INVERT the cast sign.
        max_shift_a = abs(cast_a) * max_frac if max_frac < 1.0 else float("inf")
        max_shift_b = abs(cast_b) * max_frac if max_frac < 1.0 else float("inf")
        shift_a = float(np.clip(cast_a * strength, -max_shift_a, max_shift_a))
        shift_b = float(np.clip(cast_b * strength, -max_shift_b, max_shift_b))

        L_norm = lab[:, :, 0] / 255.0
        lab[:, :, 1] -= shift_a * L_norm
        lab[:, :, 2] -= shift_b * L_norm

        if context:
            context.add_log(
                f"ColorArtifactReducer: global cast neutralization "
                f"(a={cast_a:+.1f}, b={cast_b:+.1f}, applied a={shift_a:+.1f}, b={shift_b:+.1f}, "
                f"cap={max_frac:.2f}, strength={strength:.2f})"
            )

        return cv2.cvtColor(
            np.clip(lab, 0, 255).astype(np.uint8),
            cv2.COLOR_LAB2RGB,
        )

    # ------------------------------------------------------------------ T-7 --
    # Dye-fade / color-cast rebalancing (Phase 4B.5). Input-side complement to
    # neutralize_global_cast: that one removes DDColor's warm shift on the
    # OUTPUT; this one corrects a faded print's cast BEFORE analysis and
    # colorization see it. Probe-calibrated gate (.freebuff probe, deleted
    # post-phase): sepia-classified corpus images carry devB up to +32.9 —
    # that is HISTORIC CHARACTER the pipeline deliberately preserves, so they
    # are excluded by color_type. The target is a color-classified print with
    # a dye-fade cast (real corpus: screenshot_portrait_3, devA +10.7);
    # color images within the gate (portrait_2, devA +3.9) stay untouched.

    CAST_GATE = 8.0        # fire only beyond this max |a/b| deviation
    CAST_STRENGTH = 0.7    # partial correction — never force neutral

    @staticmethod
    def rebalance_dye_fade(img: np.ndarray,
                           color_type: str = "color",
                           context: Optional[PipelineContext] = None,
                           gate: float = None,
                           strength: float = None) -> tuple[np.ndarray, dict]:
        """Rebalance differential dye fading on COLOR-classified prints.

        Luminance-weighted gray-world measurement of the a*/b* deviation,
        corrected with a partial channel-curve (same law as the salvaged
        neutralize_global_cast). Sepia and grayscale images are returned
        BITWISE unchanged by type — sepia tone is intentional historic
        character, and grayscale has no chroma to rebalance (documented
        false-negative: a heavily faded print the analyzer calls sepia will
        not be corrected).

        Returns (image, info) where info records the decision for metadata.
        """
        gate = ColorArtifactReducer.CAST_GATE if gate is None else gate
        strength = ColorArtifactReducer.CAST_STRENGTH if strength is None else strength
        info: dict = {"applied": False, "color_type": color_type}

        if color_type != "color":
            info["reason"] = f"type '{color_type}' excluded — sepia tone is historic character"
            return img, info

        try:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
            Lw = lab[:, :, 0] / 255.0 + 1e-3
            dev_a = float(np.sum((lab[:, :, 1] - 128.0) * Lw) / np.sum(Lw))
            dev_b = float(np.sum((lab[:, :, 2] - 128.0) * Lw) / np.sum(Lw))
            info["dev_a"] = round(dev_a, 2)
            info["dev_b"] = round(dev_b, 2)

            if max(abs(dev_a), abs(dev_b)) <= gate:
                info["reason"] = f"cast within gate ({gate:.0f})"
                return img, info

            lab[:, :, 1] -= dev_a * Lw * strength
            lab[:, :, 2] -= dev_b * Lw * strength
            out = cv2.cvtColor(
                np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB
            )
            info["applied"] = True
            info["strength"] = strength
            if context:
                context.add_log(
                    f"ColorArtifactReducer: dye-fade rebalance "
                    f"(a={dev_a:+.1f}, b={dev_b:+.1f}, strength={strength:.2f})"
                )
            return out, info
        except Exception as exc:  # pragma: no cover - defensive
            info["error"] = str(exc)
            return img, info

