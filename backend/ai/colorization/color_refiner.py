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

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
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


class SaturationBalancer:
    """
    Balances and normalizes saturation to prevent unnatural colors.
    """
    
    @staticmethod
    def analyze_saturation(img: np.ndarray) -> dict:
        """
        Analyze saturation distribution across the image.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        saturation = hsv[:, :, 1]
        
        return {
            'mean_sat': float(np.mean(saturation)),
            'std_sat': float(np.std(saturation)),
            'max_sat': float(np.max(saturation)),
            'min_sat': float(np.min(saturation)),
            'median_sat': float(np.median(saturation))
        }
    
    @staticmethod
    def balance_saturation(img: np.ndarray, 
                           target_mean: float = 80.0,
                           max_saturation: float = 200.0,
                           preserve_highlights: bool = True) -> np.ndarray:
        """
        Balance saturation to achieve natural, film-like results.
        
        Args:
            img: Input RGB image
            target_mean: Target mean saturation value (0-255)
            max_saturation: Maximum allowed saturation
            preserve_highlights: Whether to preserve high saturation in highlights
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        
        current_mean = float(np.mean(s))
        
        # If saturation is too high, scale it down
        if current_mean > target_mean:
            scale_factor = target_mean / (current_mean + 1e-6)
            s = s * scale_factor
        
        # Cap maximum saturation
        s = np.clip(s, 0, max_saturation)
        
        # Preserve saturation in well-lit areas
        if preserve_highlights:
            # Identify highlight areas (high value)
            highlight_mask = v > 200
            # Allow higher saturation in highlights
            s[highlight_mask] = np.clip(s[highlight_mask] * 1.2, 0, 255)
        
        hsv[:, :, 0] = h
        hsv[:, :, 1] = np.clip(s, 0, 255)
        hsv[:, :, 2] = v
        
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return result


class ColorSmoother:
    """
    Smooths color transitions and reduces pixelation artifacts.
    """
    
    @staticmethod
    def detect_color_noise(img: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Detect areas with color noise (isolated color pixels).
        """
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Calculate color difference between each pixel and its neighbors
        h, w = lab.shape[:2]
        color_diff = np.zeros((h, w))
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                shifted = np.roll(lab, (dy, dx), axis=(0, 1))
                diff = np.linalg.norm(lab - shifted, axis=2)
                color_diff += diff
        
        # Normalize
        color_diff = color_diff / 8.0  # 8 neighbors
        
        # Identify isolated color pixels
        noise_mask = color_diff > threshold * 255
        
        return noise_mask
    
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


class GradientPreservingSmoother:
    """
    Smooths images while preserving important gradients and edges.
    """
    
    @staticmethod
    def smooth_preserving_gradients(img: np.ndarray,
                                    iterations: int = 2,
                                    strength: float = 0.3) -> np.ndarray:
        """
        Apply smoothing that preserves strong gradients and edges.
        """
        result = img.copy()
        
        for _ in range(iterations):
            # Apply selective smoothing based on gradient strength
            lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(np.float32)
            L = lab[:, :, 0]
            
            # Calculate gradient magnitude
            L_f32 = L.astype(np.float32)
            grad_x = cv2.Sobel(L_f32, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(L_f32, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
            
            # Normalize gradient magnitude
            grad_norm = grad_mag / (np.max(grad_mag) + 1e-6)
            
            # Create protection mask - protect areas with strong gradients
            protect_mask = grad_norm > 0.1
            
            # Apply Gaussian blur to smooth areas
            smoothed_L = cv2.GaussianBlur(L_f32, (5, 5), 0)
            
            # Blend based on gradient protection
            L_result = np.where(
                protect_mask,
                L,  # Keep original in high-gradient areas
                L * (1 - strength) + smoothed_L * strength  # Smooth in flat areas
            )
            
            lab[:, :, 0] = L_result
            result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        return result
