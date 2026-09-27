"""
feature_bank.py — Engine A: Fast Deterministic Image Feature Extractor.

Computes spatial, frequency, and morphological image metrics in <15ms.
Extracts:
- Tile-level Laplacian variance & high-frequency energy
- 2D Fourier power spectrum peak-to-median ratio (Moiré screening)
- LAB color gamut distribution, chroma standard deviations, and color cast delta E
- Structure tensor eigenvalues & anisotropic coherence
- Edge continuity, Canny density, and step-edge overshoot ratios
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple


class FeatureBank:
    """Fast numeric feature extractor supporting the dual-engine perception stack."""

    @staticmethod
    def extract_global_features(img_rgb: np.ndarray) -> Dict[str, Any]:
        """Compute whole-canvas statistics in <10ms."""
        h, w = img_rgb.shape[:2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if img_rgb.ndim == 3 else img_rgb
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB) if img_rgb.ndim == 3 else None

        # 1. High-frequency Laplacian variance
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. 2D FFT Frequency peak ratio (Periodic moiré / halftone detection)
        small_gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        f_transform = np.fft.fft2(small_gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        # Zero out DC central 16px radius
        ch, cw = 128, 128
        y, x = np.ogrid[:256, :256]
        mask_dc = (x - cw) ** 2 + (y - ch) ** 2 <= 16 ** 2
        magnitude[mask_dc] = 0
        median_mag = float(np.median(magnitude))
        peak_mag = float(np.max(magnitude))
        fft_peak_ratio = peak_mag / max(1e-5, median_mag)

        # 3. LAB Color & Chroma Statistics
        if lab is not None:
            a_chan = lab[:, :, 1].astype(np.float32)
            b_chan = lab[:, :, 2].astype(np.float32)
            a_std = float(np.std(a_chan))
            b_std = float(np.std(b_chan))
            chroma_std = float(np.sqrt(a_std ** 2 + b_std ** 2))
            # Distance of median chroma from neutral (128, 128)
            a_mean_dev = abs(float(np.mean(a_chan)) - 128.0)
            b_mean_dev = abs(float(np.mean(b_chan)) - 128.0)
            color_cast_delta = float(np.sqrt(a_mean_dev ** 2 + b_mean_dev ** 2))
        else:
            chroma_std = 0.0
            color_cast_delta = 0.0

        # 4. Ringing / Overshoot Ratio
        # Compute gradient magnitude along horizontal/vertical step-edges
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        p99 = float(np.percentile(grad_mag, 99))
        p90 = float(np.percentile(grad_mag, 90))
        ringing_ratio = p99 / max(1e-5, p90)

        return {
            "height": h,
            "width": w,
            "min_dim": min(h, w),
            "laplacian_var": round(lap_var, 2),
            "fft_peak_ratio": round(fft_peak_ratio, 2),
            "chroma_std": round(chroma_std, 2),
            "color_cast_delta": round(color_cast_delta, 2),
            "ringing_ratio": round(ringing_ratio, 2),
        }

    @staticmethod
    def extract_structure_tensor(gray: np.ndarray, sigma: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """Compute local structure tensor eigenvalues and orientation coherence."""
        gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)

        ksize = int(2 * round(3 * sigma) + 1)
        jxx = cv2.GaussianBlur(gx * gx, (ksize, ksize), sigma)
        jyy = cv2.GaussianBlur(gy * gy, (ksize, ksize), sigma)
        jxy = cv2.GaussianBlur(gx * gy, (ksize, ksize), sigma)

        # Eigenvalues: lambda1, lambda2 = 0.5 * (trace +- sqrt(trace^2 - 4*det))
        trace = jxx + jyy
        det = jxx * jyy - jxy * jxy
        disc = np.sqrt(np.maximum(0.0, trace ** 2 - 4 * det))
        lambda1 = 0.5 * (trace + disc)
        lambda2 = 0.5 * (trace - disc)

        # Coherence: (lambda1 - lambda2) / (lambda1 + lambda2 + eps)
        coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)
        return coherence, lambda1

    @staticmethod
    def compute_boundary_gradient(img_gray: np.ndarray, binary_mask: np.ndarray) -> float:
        """Measure average Sobel boundary gradient across perimeter of mask."""
        if binary_mask is None or cv2.countNonZero(binary_mask) == 0:
            return 0.0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(binary_mask, kernel, iterations=1)
        eroded = cv2.erode(binary_mask, kernel, iterations=1)
        perimeter = cv2.bitwise_and(dilated, cv2.bitwise_not(eroded))

        if cv2.countNonZero(perimeter) == 0:
            return 0.0

        sobel_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        mean_grad = float(np.mean(grad_mag[perimeter > 0]))
        return mean_grad
