"""
halftone_suppressor.py — Remove halftone / screen-dot mesh from scanned photos via 2D FFT Frequency Notch Filtering.

Old photographs printed using halftone, rosette screens, canvas, or honeycomb paper textures
have repeating dot, mesh, or grid patterns. When these images are processed by super-resolution
or inpainting models, the periodic texture is treated as content, producing severe honeycomb artifacts.

Strategy:
1. 2D FFT Spectral Peak Detection: Analyze the 2D Fourier spectrum of the luminance channel.
   Periodic mesh patterns create sharp symmetric energy spikes (spectral peaks) in the mid-frequency range.
2. Frequency Domain Notch Rejection: Construct Gaussian notch filters centered at the detected
   peak locations in frequency space and apply inverse FFT. This removes periodic mesh cleanly
   without blurring non-periodic structural edges.
3. Spatial Anisotropic Cleanup: Follow up with an edge-preserving filter to remove residual low-amplitude mesh.
"""

import cv2
import numpy as np
from ai.pipeline.pipeline_context import PipelineContext


class HalftoneSuppressor:
    """
    Detect and suppress repeating halftone / honeycomb / mesh texture in scanned photos.
    """

    @staticmethod
    def _detect_halftone(gray_f: np.ndarray) -> tuple[bool, list]:
        """
        Detect periodic mesh pattern via 2D FFT spectral peak analysis.
        Returns (has_halftone, peak_coordinates).
        """
        h, w = gray_f.shape
        scale = min(1.0, 512.0 / max(h, w))
        if scale < 1.0:
            small = cv2.resize(
                gray_f, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        else:
            small = gray_f

        sh, sw = small.shape
        cy, cx = sh // 2, sw // 2

        # 2D FFT shifted magnitude spectrum
        fft_shift = np.fft.fftshift(np.fft.fft2(small))
        mag = np.abs(fft_shift)
        log_mag = np.log(mag + 1.0)

        # Mid-frequency annular ring (5% - 35% of max radius)
        min_dim = min(sh, sw)
        r_min = max(6, int(min_dim * 0.05))
        r_max = int(min_dim * 0.38)

        ys, xs = np.ogrid[:sh, :sw]
        dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        ring_mask = (dist >= r_min) & (dist <= r_max)

        ring_vals = log_mag[ring_mask]
        if ring_vals.size == 0:
            return False, []

        mean_val = float(np.mean(ring_vals))
        std_val = float(np.std(ring_vals))

        # Halftone spectral peaks exceed mean + 3.0 * std
        threshold = mean_val + max(3.0 * std_val, 1.2)
        peak_mask = ring_mask & (log_mag > threshold)

        # Perform local maximum suppression to locate peak centers
        dilated = cv2.dilate(log_mag, np.ones((5, 5), np.uint8))
        local_max = (log_mag == dilated) & peak_mask

        y_peaks, x_peaks = np.where(local_max)
        peaks = []
        for y, x in zip(y_peaks, x_peaks):
            # Scale coordinates back to original image dimensions
            orig_y = int(y / scale)
            orig_x = int(x / scale)
            peaks.append((orig_y, orig_x))

        # Check peak count and energy ratio
        max_peak = float(np.max(ring_vals)) if ring_vals.size > 0 else 0.0
        peak_ratio = (max_peak - mean_val) / (std_val + 1e-6)

        has_halftone = (len(peaks) >= 4 and peak_ratio > 5.0) or (peak_ratio > 6.0)
        return has_halftone, peaks

    @staticmethod
    def _apply_fft_notch(img_np: np.ndarray, gray_f: np.ndarray) -> np.ndarray:
        """
        Apply 2D FFT Notch Filtering to remove periodic mesh frequencies.
        """
        h, w = gray_f.shape
        cy, cx = h // 2, w // 2

        fft_shift = np.fft.fftshift(np.fft.fft2(gray_f))
        mag = np.abs(fft_shift)
        log_mag = np.log(mag + 1.0)

        min_dim = min(h, w)
        r_min = max(6, int(min_dim * 0.04))
        r_max = int(min_dim * 0.40)

        ys, xs = np.ogrid[:h, :w]
        dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        ring_mask = (dist >= r_min) & (dist <= r_max)

        ring_vals = log_mag[ring_mask]
        if ring_vals.size == 0:
            return img_np

        mean_val = float(np.mean(ring_vals))
        std_val = float(np.std(ring_vals))

        threshold = mean_val + max(2.8 * std_val, 1.0)
        peak_mask = ring_mask & (log_mag > threshold)

        dilated = cv2.dilate(log_mag, np.ones((7, 7), np.uint8))
        local_max = (log_mag == dilated) & peak_mask

        y_peaks, x_peaks = np.where(local_max)

        # Construct Gaussian Notch Rejection Mask
        notch_mask = np.ones((h, w), dtype=np.float32)
        sigma = max(3.0, min_dim * 0.008)

        for y_p, x_p in zip(y_peaks, x_peaks):
            # Ignore DC center
            if np.sqrt((y_p - cy) ** 2 + (x_p - cx) ** 2) < r_min:
                continue
            d2 = (ys - y_p) ** 2 + (xs - x_p) ** 2
            notch_mask *= (1.0 - np.exp(-d2 / (2.0 * sigma ** 2)))

        # Apply notch mask to Fourier spectrum
        fft_filtered = fft_shift * notch_mask
        L_notch = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_filtered)))
        L_notch = np.clip(L_notch, 0, 255).astype(np.float32)

        # Spatial anisotropic refinement for residual mesh
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        L_orig = lab[:, :, 0].astype(np.float32)

        # Combine FFT notch luminance with original color channels
        # Mild guided filter on notch luminance to smooth residual grid lines
        L_smooth = cv2.bilateralFilter(L_notch.astype(np.uint8), d=7, sigmaColor=25, sigmaSpace=9).astype(np.float32)

        # Blend notch luminance based on local gradient strength
        grad_x = cv2.Sobel(L_orig, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(L_orig, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_norm = np.clip(grad_mag / (grad_mag.max() + 1e-6), 0.0, 1.0)

        # Stronger suppression on uniform/mesh areas, preserving sharp structural boundaries
        # REDUCED from 0.85 to 0.50 — previous value overwrote 85% of luminance with filtered result
        blend_notch = 0.50 * (1.0 - grad_norm * 0.5)
        L_final = L_orig * (1.0 - blend_notch) + L_smooth * blend_notch
        L_final = np.clip(L_final, 0, 255).astype(np.uint8)

        # Safety check: if the filtered result is drastically different from original,
        # the detection was a false positive — reject the filtering entirely.
        mae = float(np.mean(np.abs(L_final.astype(np.float32) - L_orig)))
        if mae > 15.0:
            # Filtering changed the image too much — likely a false positive
            return img_np

        lab_out = lab.copy()
        lab_out[:, :, 0] = L_final
        return cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)

    @classmethod
    def suppress(cls, context: PipelineContext) -> np.ndarray:
        """
        Check for halftone pattern via 2D FFT and suppress periodic mesh if detected.
        """
        img_np = context.working_image
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

        has_halftone, peaks = cls._detect_halftone(gray)

        if has_halftone or context.image_metadata.get("force_halftone_suppression", False):
            context.add_log(
                f"HalftoneSuppressor: suppressing periodic mesh via 2D FFT Notch Filter "
                f"[{len(peaks)} spectral peaks detected]."
            )
            suppressed = cls._apply_fft_notch(img_np, gray)
            context.working_image = suppressed
            context.image_metadata["halftone_suppressed"] = True
            return suppressed
        else:
            context.add_log(
                "HalftoneSuppressor: no periodic mesh detected — skipping."
            )
            context.image_metadata["halftone_suppressed"] = False
            return img_np

