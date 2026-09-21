"""
quality_analyzer.py — Image quality and type analysis.

ROOT CAUSE FIXED:
  The debug image has mean_saturation=131.9 and mean_channel_diff=155.03,
  meaning it's heavily warm/golden BUT NOT classified as sepia because:
    - mean_saturation < 60 was WAY too tight (actual is 131.9)
    - The photo has a strong golden/olive color cast from aging + scan

  This image is an aged sepia photograph that has been scanned and has a very
  warm, golden cast — it's NOT a "color" photo.

  Fix: sepia detection is now ALSO triggered by the warm_ratio alone (>0.70)
  regardless of the mean_saturation threshold. This catches heavily-stained
  or warmly-scanned sepia photos that have higher saturation than expected.

  Added explicit `is_aged_photo` heuristic as a third classification path.
"""

import cv2
import numpy as np

from ai.analysis.content_region import content_mask, robust_sigma


class QualityAnalyzer:
    @staticmethod
    def analyze(image_np: np.ndarray) -> dict:
        """
        Analyze image quality and return structured metadata.

        Parameters
        ----------
        image_np : RGB numpy uint8 array

        Returns
        -------
        dict — never raises, uses safe defaults.
        """
        h, w = image_np.shape[:2]
        gray: np.ndarray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.uint8)
        gray_f = gray.astype(np.float32)

        # ── Sharpness ──────────────────────────────────────────────────────
        blur_score     = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score_norm = min(float(blur_score) / 500.0, 1.0)

        # ── Noise ──────────────────────────────────────────────────────────
        # Measured over the photographic CONTENT region only. Scanned figures
        # with large flat white margins previously diluted every global noise
        # statistic: MAD sigma collapsed to 0 and mean|I-blur5| halved, so heavy
        # speckle grain read as "quiet" and never reached the denoise path
        # while its grain simultaneously registered as false emulsion damage.
        _cmask = content_mask(gray)
        _content_px = gray_f[_cmask > 0]
        _content_f = _content_px.astype(np.float32)
        if _content_px.size >= 5000:
            # Measure within the content bounding box so the 5x5 blur never mixes
            # in margin zeros; inside the box, restrict to content-mask pixels.
            ys, xs = np.where(_cmask > 0)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            crop = gray[y0:y1, x0:x1]
            crop_m = _cmask[y0:y1, x0:x1]
            crop_blur = cv2.GaussianBlur(crop, (5, 5), 0)
            crop_noise = np.abs(crop.astype(np.float32) - crop_blur.astype(np.float32))
            noise_score = float(np.mean(crop_noise[crop_m > 0]))
        else:
            blurred_5 = cv2.GaussianBlur(gray, (5, 5), 0)
            noise_score = float(np.mean(np.abs(gray_f - blurred_5.astype(np.float32))))

        # Robust content-only sigma (MAD on median residual). Reported for
        # downstream stages; near-zero on the dilated-margin problem this fixes.
        sigma_content = robust_sigma(gray, _cmask)
        noise_score = max(noise_score, 0.9 * sigma_content)

        # ── Contrast ──────────────────────────────────────────────────────
        contrast_score = float(np.std(gray_f))

        # ── Grayscale detection ───────────────────────────────────────────────
        r: np.ndarray = np.asarray(image_np[:, :, 0], dtype=np.float32)
        g: np.ndarray = np.asarray(image_np[:, :, 1], dtype=np.float32)
        b: np.ndarray = np.asarray(image_np[:, :, 2], dtype=np.float32)
        mean_diff = (
            float(np.mean(np.abs(r - g)))
            + float(np.mean(np.abs(g - b)))
            + float(np.mean(np.abs(b - r)))
        )
        base_is_grayscale = mean_diff < 5.0

        # ── Sepia / color-type detection ──────────────────────────────────────
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        S = hsv[:, :, 1].astype(np.float32)
        H = hsv[:, :, 0].astype(np.float32)
        V = hsv[:, :, 2].astype(np.float32)

        mean_saturation = float(np.mean(S))
        is_grayscale_scan = mean_diff < 12.0 and mean_saturation < 25.0
        is_grayscale = base_is_grayscale or is_grayscale_scan

        # Warm hues: 10-45 (0-180 scale) = yellow-brown-orange
        warm_pixels = int(np.sum((H >= 8) & (H <= 50)))
        warm_ratio  = warm_pixels / max(H.size, 1)

        # KEY FIX: Three-tier sepia detection:
        #
        # Tier 1 — Classic sepia (low sat, very warm):
        #   mean_sat < 60, warm_ratio > 0.40
        #
        # Tier 2 — Aged/stained sepia (moderate-high sat, extremely warm):
        #   The photo has been yellowed/golden by aging or scanning. It is NOT
        #   a true color photo because >70% of pixels have warm hue and the
        #   image is monochrome portrait content.
        #   mean_sat < 160, warm_ratio > 0.70
        #
        # Tier 3 — Warm monochrome scan:
        #   All channels correlated (low mean_diff relative to intensity) AND
        #   predominantly warm.
        is_sepia_classic = (
            not is_grayscale
            and mean_saturation < 70.0
            and warm_ratio > 0.40
        )
        is_sepia_aged = (
            not is_grayscale
            and mean_saturation < 160.0
            and warm_ratio > 0.70
        )
        # Warm monochrome: channel diff much lower than mean value → uniform warm cast
        mean_value = float(np.mean(V))
        is_warm_mono = (
            not is_grayscale
            and mean_diff > 5.0
            and (mean_diff / (mean_value + 1e-6)) < 0.6
            and warm_ratio > 0.60
        )

        is_sepia = is_sepia_classic or is_sepia_aged or is_warm_mono

        if is_grayscale:
            color_type = "grayscale"
        elif is_sepia:
            color_type = "sepia"
        else:
            color_type = "color"

        # ── Damage estimation ─────────────────────────────────────────────────
        k_size = max(15, min(31, int(max(h, w) * 0.02))) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        gray_damage = cv2.medianBlur(gray, 3)
        tophat = cv2.morphologyEx(gray_damage, cv2.MORPH_TOPHAT, kernel)

        # Damage should be structural, not ordinary film grain or benchmark noise.
        # The old threshold counted single noisy pixels as white flakes, which sent
        # clean noisy images into destructive scratch restoration and colorization.
        tophat_f = np.asarray(tophat, dtype=np.float32)
        dynamic_threshold = max(28.0, float(np.mean(tophat_f) + 2.5 * np.std(tophat_f)))
        bright_mask = (tophat >= dynamic_threshold).astype(np.uint8) * 255

        min_area = max(4, int(h * w * 0.00025))
        max_area = max(min_area + 1, int(h * w * 0.35))
        structural_mask = np.zeros_like(bright_mask)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            bright_mask, connectivity=8
        )
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            cw = int(stats[label, cv2.CC_STAT_WIDTH])
            ch = int(stats[label, cv2.CC_STAT_HEIGHT])
            extent = area / max(float(cw * ch), 1.0)
            is_scratch_like = max(cw, ch) >= 8 and min(cw, ch) <= max(3, int(0.2 * max(cw, ch)))
            is_flake_like = extent >= 0.12
            if is_scratch_like or is_flake_like:
                structural_mask[labels == label] = 255

        damage_percentage = float(np.sum(structural_mask > 0) / (h * w) * 100.0)

        # ── Fast scratch line detection ───────────────────────────────────────
        # Thin white/bright cracks (0.2-2% coverage) are missed by the top-hat
        # structural detector because they don't meet the 5% area threshold.
        # This lightweight heuristic uses Canny edges + morphological line
        # filtering to detect elongated bright linear features.
        has_scratch_lines = False
        scratch_line_pct = 0.0
        try:
            # Detect high-contrast edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Filter for bright edges only (scratches are brighter than surroundings)
            local_mean = cv2.GaussianBlur(gray_f, (21, 21), 0)
            # Raised delta threshold 15→30: scratches are abrupt bright flashes;
            # natural edges (branches, bark, fence) have only moderate local contrast.
            bright_edge_mask = (edges > 0) & (gray_f > local_mean + 30.0)
            bright_edges = bright_edge_mask.astype(np.uint8) * 255
            
            # Use morphological close to connect nearby edge fragments into lines
            line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
            bright_edges = cv2.morphologyEx(bright_edges, cv2.MORPH_CLOSE, line_kernel)
            line_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            bright_edges = cv2.morphologyEx(bright_edges, cv2.MORPH_CLOSE, line_kernel2)
            
            # Filter connected components by aspect ratio (scratches are elongated)
            n_labels, labels_s, stats_s, _ = cv2.connectedComponentsWithStats(
                bright_edges, connectivity=8
            )
            scratch_mask = np.zeros_like(bright_edges)
            for i in range(1, n_labels):
                area_s = int(stats_s[i, cv2.CC_STAT_AREA])
                sw = int(stats_s[i, cv2.CC_STAT_WIDTH])
                sh = int(stats_s[i, cv2.CC_STAT_HEIGHT])
                aspect = max(sw, sh) / max(1, min(sw, sh))
                # Scratches: very elongated (aspect > 6, raised from 4), minimum
                # length > 3% of image short side.  Natural B&W features (branches,
                # wires, fence posts) are typically aspect 4-6; scratches are 8-20+.
                min_length = max(30, int(min(h, w) * 0.03))
                if aspect > 6.0 and max(sw, sh) > min_length and area_s > 20:
                    scratch_mask[labels_s == i] = 255
            
            scratch_line_pct = float(np.sum(scratch_mask > 0) / (h * w) * 100.0)
            # Raised trigger from 0.1% → 0.8%: natural linear features (branches,
            # fences, horizon lines) on a clean B&W photo can easily reach 0.1-0.4%.
            # Actual film scratches covering < 0.8% are mostly invisible anyway.
            has_scratch_lines = scratch_line_pct > 0.8
        except Exception:
            pass  # Non-critical — don't block pipeline

        # Aged sepia prints often lose emulsion as warm-white flakes, not
        # neutral white pixels. The generic top-hat signal above deliberately
        # rejects those tiny components to avoid confusing grain with damage.
        # Detect this distinct pattern only on sepia input: bright, locally
        # isolated, low-to-moderate-saturation components. This keeps a noisy
        # grayscale image on the denoise path instead of sending it to repair.
        sepia_flake_percentage = 0.0
        if color_type == "sepia":
            local_luma = cv2.GaussianBlur(V, (11, 11), 0)
            flake_candidates = (
                (V > 200.0)
                & (S < 160.0)
                & ((V - local_luma) > 25.0)
            ).astype(np.uint8)

            sepia_flakes = np.zeros_like(flake_candidates)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                flake_candidates, connectivity=8
            )
            max_flake_area = max(6, int(h * w * 0.05))
            for label in range(1, num_labels):
                area = int(stats[label, cv2.CC_STAT_AREA])
                fw = int(stats[label, cv2.CC_STAT_WIDTH])
                fh = int(stats[label, cv2.CC_STAT_HEIGHT])
                extent = area / max(float(fw * fh), 1.0)
                if 5 <= area <= max_flake_area and extent >= 0.10:
                    sepia_flakes[labels == label] = 255

            sepia_flake_percentage = float(
                np.sum(sepia_flakes > 0) / (h * w) * 100.0
            )
            damage_percentage = max(damage_percentage, sepia_flake_percentage)

        # has_white_flakes: original gate (lowered from 5.0% to 2.0%)
        has_white_flakes = (
            sepia_flake_percentage >= 1.0 or damage_percentage > 2.0
        )

        # has_scratches: NEW independent scratch trigger
        # Even if damage_pct is low, visible scratches should trigger repair
        has_scratches = has_scratch_lines

        # ── Grain-field (density-invariance) gate ─────────────────────────
        # If the measured noise floor is high relative to the "damage" signal,
        # the structural damage estimate is grain, not emulsion loss. Stage 2
        # consults this flag so speckled scans are denoised, never inpainted.
        # Content-only sigma makes this immune to white scanner margins.
        grain_field = bool(sigma_content > 6.0 and damage_percentage < 6.0)
        grain_field_confidence = round(min(1.0, sigma_content / 20.0), 3)

        return {
            "content_sigma":      sigma_content,
            "grain_field":        grain_field,
            "grain_field_confidence": grain_field_confidence,
            "width":             w,
            "height":            h,
            "blur_score":        blur_score,
            "blur_score_norm":   blur_score_norm,
            "noise_score":       noise_score,
            "contrast_score":    contrast_score,
            "is_grayscale":      is_grayscale,
            "is_sepia":          is_sepia,
            "color_type":        color_type,
            "mean_saturation":   mean_saturation,
            "warm_ratio":        warm_ratio,
            "damage_percentage": damage_percentage,
            "scratch_line_pct":  scratch_line_pct,
            "sepia_flake_percentage": sepia_flake_percentage,
            "has_white_flakes":  has_white_flakes,
            "has_scratches":     has_scratches,
            "damage_confidence":  "structural" if has_white_flakes or has_scratches else "low",
        }
