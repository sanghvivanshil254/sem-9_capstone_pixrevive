"""content_region.py — Margin-robust statistics helpers.

Shared by QualityAnalyzer and NoiseClassifier. Many old photographs are scanned
figures / photos with large flat white (or black) scanner margins. Global
statistics computed over the whole frame are diluted by those margins:

- A robust median estimator over the whole frame lands on the constant margin
  value, so sigma collapses to 0.00 even when the photo content is heavily
  grained.
- Correlation-based speckle detectors see half their samples carrying zero
  residual, flattening the brightness-residual correlation to ~0.
- Compression detectors read the sharp content/margin boundary as 8x8 block
  edges, flagging clean images as JPEG-damaged.

`content_mask` isolates the photographic content region so every estimator can
measure the photo instead of the scanner bed. It is intentionally cheap (no
model, no big convolutions) and conservative: when in doubt it returns "all
content" (mask of ones), so behaviour on normal photos is unchanged.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# Fraction of the frame a content component must span to be considered the
# photo rather than a stamp/logo/text caption on the scanner bed.
_MIN_CONTENT_AREA_FRACTION = 0.15


def content_mask(
    img: np.ndarray,
    margin_tolerance: float = 6.0,
    closing_ksize: int = 31,
) -> np.ndarray:
    """Return a uint8 mask (255 = photographic content) for `img` (RGB or gray).

    The detector looks for the dominant flat-intensity "margin" value — the
    near-saturated white (or near-black) scanner border — using a histogram of
    extreme luminance bins, then treats everything that is not margin-adjacent
    as content. On images without such margins (most real photos) the near-white
    bin is small and disconnected, so the function falls through to an all-ones
    mask and every caller behaves exactly as before.

    Parameters
    ----------
    margin_tolerance:
        Pixels within this luminance distance of the margin value count as
        margin. Scanner borders are usually >=250; 6 tolerates vignetting and
        soft scan edges.
    closing_ksize:
        Morphological closing kernel that fills interior detail (faces, skies)
        which happens to match the margin value so the mask is one solid
        content region rather than swiss cheese.
    """
    if img is None or img.size == 0:
        h = w = 0
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    if h < 32 or w < 32:
        return np.ones((h, w), dtype=np.uint8)

    gray_u8 = gray if gray.dtype == np.uint8 else np.clip(gray, 0, 255).astype(np.uint8)

    margin_val = _dominant_margin_value(gray_u8)
    if margin_val is None:
        return np.ones((h, w), dtype=np.uint8)

    margin_band = np.abs(gray_u8.astype(np.int16) - int(margin_val)) <= margin_tolerance
    margin_band_u8 = margin_band.astype(np.uint8) * 255

    # Keep only margin components that actually touch the frame border (real
    # scanner margins always do; a white wedding dress does not).
    num, labels, stats, _ = cv2.connectedComponentsWithStats(margin_band_u8, connectivity=4)
    frame_margin = np.zeros_like(margin_band_u8)
    for i in range(1, num):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        touches = x <= 1 or y <= 1 or (x + cw) >= (w - 2) or (y + ch) >= (h - 2)
        if touches and stats[i, cv2.CC_STAT_AREA] > 0.02 * h * w:
            frame_margin[labels == i] = 255

    if float(np.count_nonzero(frame_margin)) / float(h * w) < 0.04:
        # No significant frame-touching margin — ordinary photo, keep everything.
        return np.ones((h, w), dtype=np.uint8)

    # Solidify the margin strip (bleeds through soft scan edges), THEN invert.
    # NOTE: a CLOSE on the inverted image would be wrong here — scanner margins
    # are often only 10-20px wide, thinner than any kernel large enough to
    # close interior detail holes, so closing would fill the margin back in and
    # silently return an all-content mask (measured 2026-09-21).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_ksize, closing_ksize))
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    frame_margin = cv2.morphologyEx(frame_margin, cv2.MORPH_DILATE, small, iterations=1)
    content = cv2.morphologyEx(255 - frame_margin, cv2.MORPH_OPEN, kernel)

    # Reject pathological results: content must be a sensible chunk of the frame.
    area_frac = float(np.count_nonzero(content)) / float(h * w)
    if area_frac < _MIN_CONTENT_AREA_FRACTION or area_frac > 0.995:
        return np.ones((h, w), dtype=np.uint8)

    return content


def _dominant_margin_value(gray_u8: np.ndarray) -> Optional[int]:
    """Return the luminance value of a flat scanner margin, or None.

    Scanner beds are near-white (250+) or, rarely, near-black (<8). A real
    photo rarely has >8% of pixels inside those bands; a figure with margins
    has 20-60%.
    """
    hist = cv2.calcHist([gray_u8], [0], None, [256], [0, 256]).ravel()
    total = float(hist.sum())
    if total <= 0:
        return None

    white_mass = float(hist[244:].sum()) / total
    black_mass = float(hist[:10].sum()) / total
    if white_mass >= 0.08:
        # Precise value: brightest bin with meaningful mass (handles 255 vs 250).
        lo, hi = 244, 256
        top = 255
    elif black_mass >= 0.08:
        lo, hi = 0, 10
        top = 0
    else:
        return None

    band = hist[lo:hi]
    best = int(np.argmax(band)) + lo
    if float(band[best - lo]) / total < 0.05:
        return None
    if top == 0 and best > 10:
        return None
    return best


def robust_sigma(gray_u8: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """MAD-based noise sigma computed over content pixels only.

    Uses the high-pass residual (pixel - 3x3 median) which is the same statistic
    Stage 5 uses, but restricted to content. Returns 0.0 when content is flat.

    Lena-mottle rescue branch: a dense 1-2px halftone mottle field can lurk
    *below* the 3x3-median residual — a 1px speck's residual is ~0 whenever
    its 3x3 window contains another speck within a few gray levels of the
    local base. On the synthetic pure-255-frame probe the field covers 27%
    of content windows at amplitude 6-13, yet the plain median saturates on
    the flat floor (residual 0 at the 62nd percentile) and the classic
    estimator reports sigma 0.00. Every Stage-5 NLM h = max(floor, k*sigma)
    then floors at 5.5-6.5, the mottle survives into mid-res content, and
    SR+sharpen re-amplifies it as a dotted-moss texture.

    This branch engages ONLY when the classic estimate is truly COLLAPSED
    (sigma_mad < 1.0, median saturated on the flat floor). A higher gate
    (e.g. < 2.0) hijacks real fixtures: the shipped case8/lena-figure input
    measures sigma_mad 1.48 with 20% windows covered by 20-amplitude specks —
    those are sparse speckle = DAMAGE on the repair path, and re-labeling
    them a dense grain field flips the route and leaves the specks in.
    """
    if gray_u8.ndim != 2:
        gray_u8 = cv2.cvtColor(gray_u8, cv2.COLOR_RGB2GRAY)
    gray_u8 = np.clip(gray_u8, 0, 255).astype(np.uint8)
    m = mask if mask is not None else content_mask(gray_u8)
    med = cv2.medianBlur(gray_u8, 3).astype(np.float32)
    residual = np.abs(gray_u8.astype(np.float32) - med)
    sel = residual[m > 0]
    if sel.size < 500:
        sel = residual.ravel()
    mad = float(np.median(sel))
    sigma_mad = 1.4826 * mad

    # Mottle rescue: only when the median SATURATED on the flat floor.
    # Single-speck amplitude floor: specks below this are sensor/quantization
    # noise (clean-scan residual p50 = 0.0, p95 < 2).
    if sigma_mad < 1.0:
        _speck_amp = 4.0
        hostile = (residual >= _speck_amp).astype(np.float32)
        cnt = cv2.boxFilter(hostile, -1, (3, 3), normalize=False)
        core = cv2.erode((m > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        cover = float((cnt[core] >= 2).mean()) if bool(core.any()) else 0.0
        if cover >= 0.05:
            # Population amplitude: per-pixel MAD over the hostile speck
            # population only (robust to the flat floor dominating the median).
            hostiles = residual[(m > 0) & (residual >= _speck_amp)]
            if hostiles.size >= 100:
                host_mad = float(np.median(np.abs(hostiles - np.median(hostiles))))
                sigma_speck = 1.4826 * host_mad
            else:
                sigma_speck = 0.0
            p95 = float(np.percentile(hostiles, 95)) if hostiles.size >= 100 else 0.0
            sigma_mottle = max(sigma_speck, min(p95, 16.0) * 0.55)
            # Coverage-weighted blend: a sparse 5% speckle adds little; a
            # dense field (27% on the probe) biases the sigma up to the
            # real speck population so Stage 5's NLM actually acts on it.
            blend = float(np.clip((cover - 0.05) / 0.25, 0.0, 1.0))
            return float(min(sigma_mad + blend * sigma_mottle, 20.0))

    return sigma_mad
