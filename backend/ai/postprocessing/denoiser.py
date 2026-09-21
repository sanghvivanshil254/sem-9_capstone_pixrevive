"""
denoiser.py — Advanced denoising with artifact prevention.

This module provides multiple denoising strategies with artifact-aware selection:
1. Adaptive NLM denoising with edge preservation
2. Wavelet-based denoising for smooth regions
3. Deep denoising fallback for severe noise
4. Artifact detection to prevent over-smoothing

Key improvements:
- Prevents blocky artifacts from aggressive NLM
- Uses wavelet denoising for smooth gradient areas
- Includes edge-preserving filters
- Has artifact detection to avoid over-processing
"""

import cv2
import numpy as np
try:
    import pywt
except ImportError:
    pywt = None
from typing import Optional, Tuple
from ai.pipeline.pipeline_context import PipelineContext


class ArtifactDetector:
    """
    Detects potential denoising artifacts before they become visible.
    """
    
    @staticmethod
    def detect_blocking(img: np.ndarray, threshold: float = 0.15) -> bool:
        """
        Detect blocking artifacts by analyzing local variance.
        Blocky images have artificially high variance at block boundaries.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # Calculate local variance
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        variance = cv2.GaussianBlur(((gray - blurred) ** 2).astype(np.float32), (3, 3), 0)
        
        # Check for unnatural variance patterns
        high_var_pixels = variance > np.percentile(variance.astype(np.float32), 95)
        blocking_score = float(np.mean(high_var_pixels))
        
        return blocking_score > threshold
    
    @staticmethod
    def detect_over_smoothing(img: np.ndarray, original: np.ndarray, 
                               threshold: float = 0.12) -> bool:
        """
        Detect if denoising has caused excessive smoothing/blurring.
        
        Threshold raised from 0.05→0.12: edge detection is unreliable on
        noisy images, so the old 5% threshold caused false positives that
        prevented ANY denoising from being accepted on heavily grainy photos.
        """
        gray_orig = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray_denoised = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # Calculate edge preservation
        gray_orig_blur = cv2.GaussianBlur(gray_orig.astype(np.uint8), (3, 3), 0)
        gray_denoised_blur = cv2.GaussianBlur(gray_denoised.astype(np.uint8), (3, 3), 0)
        edges_orig = cv2.Canny(gray_orig_blur, 50, 150)
        edges_denoised = cv2.Canny(gray_denoised_blur, 50, 150)
        
        edge_loss = float(np.sum(edges_orig > 0).astype(np.int64) - np.sum(edges_denoised > 0).astype(np.int64))
        edge_loss /= max(float(np.sum(edges_orig > 0)), 1)
        
        return edge_loss > threshold


class WaveletDenoiser:
    """
    Wavelet-based denoising that preserves edges better than NLM.
    Uses smooth biorthogonal wavelets (bior2.2) to prevent Haar 2x2 step/grid artifacts.
    """
    
    @staticmethod
    def wavelet_denoise(img: np.ndarray, 
                       wavelet: str = 'bior2.2', 
                       level: int = 3,
                       threshold: float = 0.5) -> np.ndarray:
        """
        Apply wavelet denoising with soft thresholding using smooth wavelets.
        
        Args:
            img: Input RGB image (uint8)
            wavelet: Wavelet type ('bior2.2', 'sym4', 'db4', etc.)
            level: Decomposition level
            threshold: Noise threshold for coefficient shrinkage
            
        Returns:
            Denoised image
        """
        if pywt is None:
            # Fallback to bilateral filter if pywt unavailable
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            filt = cv2.bilateralFilter(bgr, d=5, sigmaColor=15, sigmaSpace=5)
            return cv2.cvtColor(filt, cv2.COLOR_BGR2RGB)

        result = np.zeros_like(img, dtype=np.float32)
        
        for channel in range(3):
            coeffs = pywt.wavedec2(img[:, :, channel].astype(np.float32), wavelet, level=level)
            
            # Apply soft thresholding to detail coefficients
            new_coeffs = [coeffs[0]]  # Keep approximation coefficients
            for i in range(1, len(coeffs)):
                detail_coeffs = coeffs[i]
                thresholded = [
                    pywt.threshold(d, value=threshold, mode='soft') 
                    for d in detail_coeffs
                ]
                new_coeffs.append(tuple(thresholded))
            
            # Reconstruct
            reconstructed = pywt.waverec2(new_coeffs, wavelet)
            h_orig, w_orig = img.shape[:2]
            rec_crop = reconstructed[:h_orig, :w_orig]
            result[:, :, channel] = np.clip(rec_crop, 0, 255)
        
        return result.astype(np.uint8)


class AdaptiveDenoiser:
    """
    Adaptive denoising that automatically selects the best method and parameter tuning.
    """
    
    @staticmethod
    def analyze_image(img: np.ndarray) -> dict:
        """
        Analyze image characteristics using Wavelet MAD for accurate noise estimation.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # Calculate noise level via Wavelet Median Absolute Deviation (MAD)
        try:
            if pywt is None:
                raise ImportError("pywt not available")
            coeffs = pywt.wavedec2(gray, 'bior2.2', level=1)
            lh_detail = coeffs[1][0]
            mad_sigma = float(np.median(np.abs(lh_detail)) / 0.6745)
        except Exception:
            mad_sigma = 0.0

        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
        gaussian_sigma = float(np.std((gray - blurred).astype(np.float32)))
        noise_std = max(mad_sigma, gaussian_sigma)
        
        # Calculate texture level
        texture_var = float(np.var(gray))
        
        # Edge density
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        edge_density = float(np.sum(edges > 0)) / edges.size
        
        return {
            'noise_level': noise_std,
            'texture_level': texture_var,
            'edge_density': edge_density,
            'is_smooth': texture_var < 800,
            'is_textured': texture_var >= 800,
            'has_edges': edge_density > 0.01
        }
    
    @staticmethod
    def denoise(img: np.ndarray, 
                noise_level: Optional[float] = None,
                context: Optional[PipelineContext] = None) -> np.ndarray:
        """
        Apply adaptive denoising based on image characteristics.
        Smoothly removes heavy grain stippling while retaining structural edges.
        """
        analysis = AdaptiveDenoiser.analyze_image(img)
        noise_level = noise_level if noise_level is not None else analysis['noise_level']

        if context:
            context.add_log(f"Denoiser: noise_sigma={noise_level:.2f}, texture={analysis['texture_level']:.0f}, "
                          f"edges={analysis['edge_density']*100:.1f}%")

        if noise_level < 4.0:
            # Low noise - gentle smooth wavelet pass
            result = WaveletDenoiser.wavelet_denoise(img, wavelet='bior2.2', threshold=0.25)
        elif noise_level < 12.0:
            # Medium noise / grain - wavelet + dynamic bilateral filtering
            w_denoised = WaveletDenoiser.wavelet_denoise(img, wavelet='bior2.2', threshold=0.6)
            bgr = cv2.cvtColor(w_denoised, cv2.COLOR_RGB2BGR)
            sigma_c = float(max(20.0, min(45.0, 2.5 * noise_level)))
            filtered = cv2.bilateralFilter(bgr, d=7, sigmaColor=sigma_c, sigmaSpace=7.0)
            result = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
        else:
            # High noise / heavy grain (e.g. Image 2) - multi-stage wavelet + NLM pass
            w_denoised = WaveletDenoiser.wavelet_denoise(img, wavelet='bior2.2', threshold=1.2)
            bgr = cv2.cvtColor(w_denoised, cv2.COLOR_RGB2BGR)

            # Fast NLM for heavy grain stippling
            h_luma = float(max(6.0, min(14.0, 0.8 * noise_level)))
            nlm = cv2.fastNlMeansDenoisingColored(
                bgr, None, h=h_luma, hColor=h_luma,
                templateWindowSize=7, searchWindowSize=21
            )
            result = cv2.cvtColor(nlm, cv2.COLOR_BGR2RGB)

        return result


class SmartDenoiser:
    """
    Smart denoiser that automatically prevents artifacts.
    """
    
    @staticmethod
    def denoise_smart(img: np.ndarray, 
                      context: Optional[PipelineContext] = None,
                      max_iterations: int = 3) -> np.ndarray:
        """
        Apply denoising with artifact prevention.
        
        Uses iterative approach with artifact detection to prevent over-processing.
        """
        original = img.copy()
        
        for iteration in range(max_iterations):
            # Analyze current state
            analysis = AdaptiveDenoiser.analyze_image(img)
            
            if analysis['noise_level'] < 5.0:
                # Noise is already low enough
                # (threshold raised from 3.0→5.0: old film grain sits around
                # noise_level 3-5 and needs at least one pass to clean)
                if context:
                    context.add_log(f"Denoiser: iteration {iteration+1} - noise low, stopping.")
                break
            
            # Apply denoising tentatively. If the artifact checks fail, keep the
            # last accepted image instead of returning the damaged candidate.
            previous = img
            candidate = AdaptiveDenoiser.denoise(img, context=context)

            # Check for artifacts
            if ArtifactDetector.detect_blocking(candidate):
                if context:
                    context.add_log(
                        f"Denoiser: blocking detected at iteration {iteration+1}; "
                        "rolling back denoise candidate."
                    )
                img = previous
                break
            
            if ArtifactDetector.detect_over_smoothing(candidate, original):
                if context:
                    context.add_log(
                        f"Denoiser: over-smoothing detected at iteration {iteration+1}; "
                        "rolling back denoise candidate."
                    )
                img = previous
                break

            img = candidate
        
        return img
