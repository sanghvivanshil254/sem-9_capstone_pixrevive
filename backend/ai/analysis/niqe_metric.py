"""niqe_metric.py — Reference-free image naturalness (NIQE).

Lower is better (natural images cluster around 3–8 on photographic content).
PixRevive's benchmark set has no ground truth, so NIQE is the reference-free
ruler (Q-6) used by the analyzer, the QualityService debug metrics, and
scripts/benchmark_quality.py.

basicsr 1.4.2 still imports ``torchvision.transforms.functional_tensor``,
a module removed in newer torchvision. The import shim lives here so every
consumer gets the same one-line fix instead of three divergent copies.
"""

import logging
import sys
import types
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _ensure_basicsr_importable() -> None:
    """Shim the removed torchvision module before basicsr is imported."""
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except ImportError:
        from torchvision.transforms import functional as _tf

        stub = types.ModuleType("torchvision.transforms.functional_tensor")
        stub.rgb_to_grayscale = _tf.rgb_to_grayscale
        sys.modules.setdefault(
            "torchvision.transforms.functional_tensor", stub
        )


def niqe_score(image_gray: np.ndarray) -> Optional[float]:
    """NIQE of a grayscale uint8 image; None when it cannot be computed.

    The score is computed on a downscaled copy when the image is larger than
    512 px on the long side (bounded runtime) and is fully guarded — metric
    failures must never break a pipeline run or a benchmark sweep.
    """
    try:
        gray = np.asarray(image_gray)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]
        scale = 512.0 / max(h, w)
        if scale < 1.0:
            gray = cv2.resize(
                gray,
                (max(96, int(w * scale)), max(96, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        if gray.shape[0] < 96 or gray.shape[1] < 96:
            return None

        _ensure_basicsr_importable()
        from basicsr.metrics.niqe import calculate_niqe

        return round(float(calculate_niqe(gray, crop_border=0)), 3)
    except Exception as exc:  # SVD non-convergence on degenerate patches, etc.
        logger.debug("NIQE computation failed: %s", exc)
        return None
