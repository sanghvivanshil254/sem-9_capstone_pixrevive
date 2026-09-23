"""deband.py — T-6 banding / posterization pass (Phase 4B.4).

Banding (visible quantization stairs in smooth gradients — skies, studio
backdrops) survives the pipeline because every stage works in 8-bit: the
steps ARE the values. The fix is a bit-depth expansion in effect: smooth the
stepped luminance in a wider working range, add a small triangular-PDF dither
so the re-quantization steps stop aligning, and re-quantize to 8-bit. Applied
pre-SR (inside Stage 8) only when the existing noise_router._banding_score
measures genuine quantization structure — probe-calibrated so the entire real
corpus (max 0.503, foxing blotches) stays below the 0.6 gate while synthetic
posterization reads 1.0.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ai.postprocessing.noise_router import _banding_score

BANDING_GATE = 0.6
SMOOTH_SIGMA = 1.1   # sigma for the gradient-domain smoothing (px)
DITHER_AMP = 0.6     # triangular dither amplitude in 8-bit units


def banding_score(gray: np.ndarray) -> float:
    """Public wrapper over noise_router's banding measurement."""
    return float(_banding_score(gray))


def deband(img: np.ndarray,
           gate: float = BANDING_GATE,
           score: Optional[float] = None) -> tuple[np.ndarray, dict]:
    """Suppress quantization banding on smooth gradients. Returns (img, info).

    Gated: the image is returned BITWISE unchanged unless _banding_score
    measures genuine quantization structure (>= gate). The correction smooths
    luminance only (chroma untouched), works in float so the steps dissolve,
    and re-quantizes with triangular-PDF dither so surviving gradients no
    longer carry aligned stairs.
    """
    info: dict = {"applied": False}
    try:
        s = banding_score(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)) if score is None else float(score)
        info["banding_score"] = round(s, 3)
        if s < gate:
            return img, info

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = lab[:, :, 0]

        # Dissolve the stairs: gradient-domain smoothing of luminance only.
        L_smooth = cv2.GaussianBlur(L, (0, 0), SMOOTH_SIGMA)

        # Triangular-PDF dither (sum of two uniforms) centered on zero, in
        # 8-bit units — decorrelates the re-quantization steps from the image.
        rng = np.random.default_rng(12345)  # deterministic; banding fix must
        dith = (rng.random(L.shape) + rng.random(L.shape) - 1.0) * DITHER_AMP

        lab[:, :, 0] = np.clip(L_smooth + dith, 0, 255)
        out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

        info["applied"] = True
        info["score_after"] = round(
            banding_score(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)), 3
        )
        return out, info
    except Exception as exc:  # pragma: no cover - defensive
        info["error"] = str(exc)
        return img, info
