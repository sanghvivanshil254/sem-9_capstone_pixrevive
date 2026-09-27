"""
noise_router.py - Conservative noise classification and cleanup routing.

The router groups many named sensor/compression noise types into practical
restoration families, then applies the least destructive filter for each family.
Every candidate is quality-gated against the previous image before it is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ai.pipeline.pipeline_context import PipelineContext


_RANDOM_NOISE_TYPES = [
    "gaussian",
    "uniform",
    "poisson",
    "shot",
    "thermal",
    "read",
    "amplifier",
    "reset",
    "rayleigh",
    "exponential",
    "gamma",
    "luminance",
]
_IMPULSE_NOISE_TYPES = [
    "salt_and_pepper",
    "impulse",
    "random_valued_impulse",
]
_PERIODIC_NOISE_TYPES = [
    "periodic",
    "sinusoidal",
    "striping",
    "interference",
    "fixed_pattern",
    "prnu",
    "dark_current",
]
_COMPRESSION_NOISE_TYPES = [
    "jpeg_compression",
    "blocking",
    "ringing",
    "quantization",
    "banding",
]
_COLOR_NOISE_TYPES = [
    "color",
    "chroma",
]

SUPPORTED_NOISE_TYPES = tuple(
    _RANDOM_NOISE_TYPES
    + _IMPULSE_NOISE_TYPES
    + ["speckle"]
    + _PERIODIC_NOISE_TYPES
    + _COMPRESSION_NOISE_TYPES
    + _COLOR_NOISE_TYPES
)


@dataclass
class NoiseProfile:
    detected_types: List[str] = field(default_factory=list)
    family_scores: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def has_noise(self) -> bool:
        return bool(self.detected_types)

    def to_dict(self) -> Dict[str, object]:
        return {
            "detected_types": list(self.detected_types),
            "family_scores": dict(self.family_scores),
            "metrics": dict(self.metrics),
        }


class NoiseClassifier:
    @staticmethod
    def classify(img: np.ndarray, metadata: Optional[Dict[str, object]] = None) -> NoiseProfile:
        metadata = metadata or {}
        gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = gray.shape

        # ── Content-region masking (margin-robust statistics) ─────────────
        # Scanned figures with flat white margins previously poisoned every
        # statistic here: residual MAD collapsed toward the margin's zero,
        # the speckle correlation flattened, and the content/margin boundary
        # registered as JPEG blocking. Measure the photo, not the scanner bed.
        from ai.analysis.content_region import content_mask
        cmask = content_mask(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        content_only = (cmask > 0)

        blurred_for_residual: np.ndarray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2).astype(np.float32)
        residual: np.ndarray = gray - blurred_for_residual
        residual_abs: np.ndarray = np.abs(residual)
        if content_only.any():
            res_content = residual[content_only]
            residual_std = float(np.std(res_content))
            residual_mad = float(np.median(np.abs(res_content - float(np.median(res_content)))) / 0.6745)
        else:
            residual_std = float(np.std(residual))
            residual_mad = float(np.median(np.abs(residual - float(np.median(residual)))) / 0.6745)
        edge_density = _edge_density(gray.astype(np.uint8))
        texture_var = float(np.var(gray[content_only])) if content_only.any() else float(np.var(gray))

        med = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
        med_diff = np.abs(gray - med)
        if content_only.any():
            md_content = med_diff[content_only]
            impulse_ratio = float(np.mean(md_content > 25.0))
            random_impulse_ratio = float(np.mean(md_content > 40.0))
        else:
            impulse_ratio = float(np.mean(med_diff > 25.0))
            random_impulse_ratio = float(np.mean(med_diff > 40.0))

        periodic_score, striping_score = _periodic_scores(gray)
        blocking_score, blocking_score_content = _blocking_score(gray, return_content=True, content_mask=cmask)
        ringing_score = _ringing_score(gray)
        banding_score = _banding_score(gray)
        chroma_score = _chroma_noise_score(img)
        quantization_score = max(banding_score, blocking_score_content * 0.6)

        random_score = _normalize(residual_mad, 3.0, 18.0)
        speckle_score = max(
            _speckle_score(gray, residual_abs, content_mask=cmask),
            _speckle_slope_score(gray, content_mask=cmask),
        )
        # BUG FIX 2: Old gaussian_like demanded impulse_ratio < 0.015, so heavy
        # grain images (which simultaneously have high impulse AND random scores)
        # fell through both detectors and triggered nothing. Decouple the two.
        gaussian_like = random_score > 0.18

        # Degenerate-input guard: a content region with essentially no texture
        # cannot exhibit compression, banding, periodic interference, ringing,
        # or speckle — every ratio-based detector divides by ~zero there and
        # reports spurious positives (measured 2026-09-25: a flat scanned figure
        # fired seven noise families; a flat marginless control fired two).
        # Texture variance is measured over the content region, so real photos
        # (var in the hundreds+) are never affected.
        if texture_var < 2.0 and residual_std < 0.5:
            periodic_score = 0.0
            striping_score = 0.0
            blocking_score = 0.0
            ringing_score = 0.0
            banding_score = 0.0
            quantization_score = 0.0
            speckle_score = 0.0

        family_scores = {
            "random": random_score,
            "impulse": _normalize(max(impulse_ratio, random_impulse_ratio * 0.35), 0.004, 0.06),
            "periodic": max(periodic_score, striping_score),
            "compression": max(blocking_score, ringing_score, banding_score, quantization_score),
            "color": chroma_score,
            "speckle": speckle_score,
        }

        detected: List[str] = []
        if family_scores["impulse"] > 0.18:
            detected.extend(_IMPULSE_NOISE_TYPES)
        if gaussian_like:
            detected.extend(_RANDOM_NOISE_TYPES)
        if speckle_score > 0.22:
            detected.append("speckle")
        if periodic_score > 0.20:
            detected.extend(["periodic", "sinusoidal", "interference"])
        if striping_score > 0.20:
            detected.extend(["striping", "fixed_pattern", "prnu", "dark_current"])
        if blocking_score > 0.14:
            detected.extend(["jpeg_compression", "blocking"])
        if ringing_score > 0.18:
            detected.append("ringing")
        if banding_score > 0.18:
            detected.extend(["banding", "quantization"])
        if chroma_score > 0.20:
            detected.extend(_COLOR_NOISE_TYPES)

        if metadata.get("color_type") == "grayscale":
            detected = [name for name in detected if name not in _COLOR_NOISE_TYPES]

        # Keep stable order and remove duplicates.
        unique_detected = []
        for name in detected:
            if name not in unique_detected:
                unique_detected.append(name)

        metrics = {
            "residual_std": residual_std,
            "residual_mad": residual_mad,
            "edge_density": edge_density,
            "texture_var": texture_var,
            "impulse_ratio": impulse_ratio,
            "random_impulse_ratio": random_impulse_ratio,
            "periodic_score": periodic_score,
            "striping_score": striping_score,
            "blocking_score": blocking_score,
            "ringing_score": ringing_score,
            "banding_score": banding_score,
            "quantization_score": quantization_score,
            "chroma_score": chroma_score,
            "speckle_score": speckle_score,
            "height": float(h),
            "width": float(w),
        }
        return NoiseProfile(unique_detected, family_scores, metrics)


class NoiseRouter:
    @staticmethod
    def analyze(context: PipelineContext) -> NoiseProfile:
        profile = NoiseClassifier.classify(context.working_image, context.image_metadata)
        context.image_metadata["noise_profile"] = profile.to_dict()
        if profile.has_noise:
            detected = ", ".join(profile.detected_types)
            context.add_log(f"NoiseRouter: detected {detected}.")
        else:
            context.add_log("NoiseRouter: no actionable noise family detected.")
        return profile

    @staticmethod
    def reduce(context: PipelineContext, profile: Optional[NoiseProfile] = None) -> np.ndarray:
        profile = profile or NoiseRouter.analyze(context)
        if not profile.has_noise:
            return context.working_image

        current = context.working_image
        original = current.copy()
        stages: List[Tuple[str, np.ndarray]] = []

        if _has_any(profile, _IMPULSE_NOISE_TYPES):
            stages.append(("impulse_median", _reduce_impulse(current, profile)))

        if _has_any(profile, _PERIODIC_NOISE_TYPES):
            stages.append(("periodic_fft_notch", _reduce_periodic(stages[-1][1] if stages else current, profile)))

        if _has_any(profile, _COMPRESSION_NOISE_TYPES):
            stages.append(("compression_deblock_dering", _reduce_compression(stages[-1][1] if stages else current, profile)))

        if _has_any(profile, _COLOR_NOISE_TYPES):
            stages.append(("chroma_luma_separated", _reduce_chroma(stages[-1][1] if stages else current, profile)))

        if _has_any(profile, _RANDOM_NOISE_TYPES) or "speckle" in profile.detected_types:
            stages.append(("random_noise_edge_aware", _reduce_random(stages[-1][1] if stages else current, profile)))

        accepted = current
        for stage_name, candidate in stages:
            passed, reason = _candidate_passes(accepted, candidate, original, context, stage_name)
            if passed:
                accepted = candidate
                context.add_log(f"NoiseRouter: accepted {stage_name} ({reason}).")
            else:
                context.add_log(f"NoiseRouter: rejected {stage_name} ({reason}).")

        context.working_image = accepted
        return accepted


def _has_any(profile: NoiseProfile, names: List[str]) -> bool:
    return any(name in profile.detected_types for name in names)


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _edge_density(gray_u8: np.ndarray) -> float:
    edges = cv2.Canny(gray_u8, 60, 150)
    return float(np.mean(edges > 0))


def _periodic_scores(gray: np.ndarray) -> Tuple[float, float]:
    h, w = gray.shape
    small = gray
    if max(h, w) > 512:
        scale = 512.0 / max(h, w)
        small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    small_f: np.ndarray = small.astype(np.float32)
    window_y = np.hanning(small_f.shape[0])[:, None]
    window_x = np.hanning(small_f.shape[1])[None, :]
    spectrum = np.fft.fftshift(np.fft.fft2((small_f - float(np.mean(small_f))) * window_y * window_x))
    mag: np.ndarray = np.log1p(np.abs(spectrum)).astype(np.float32)
    cy, cx = np.array(mag.shape) // 2
    radius = max(8, min(mag.shape) // 18)
    mag[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1] = 0.0
    threshold = float(np.mean(mag) + 4.0 * np.std(mag))
    peak_ratio = float(np.mean(mag > threshold))
    peak_strength = float(np.max(mag) / (float(np.mean(mag)) + float(np.std(mag)) + 1e-6))
    periodic_score = min(1.0, peak_ratio * 320.0 + max(0.0, peak_strength - 3.2) / 8.0)

    col_profile: np.ndarray = np.mean(small_f, axis=0).astype(np.float32)
    row_profile: np.ndarray = np.mean(small_f, axis=1).astype(np.float32)
    col_hp: np.ndarray = col_profile - cv2.GaussianBlur(col_profile.reshape(1, -1), (1, 31), 0).ravel().astype(np.float32)
    row_hp: np.ndarray = row_profile - cv2.GaussianBlur(row_profile.reshape(-1, 1), (31, 1), 0).ravel().astype(np.float32)
    striping_score = min(
        1.0,
        max(float(np.std(col_hp)), float(np.std(row_hp))) / (float(np.std(small_f)) + 1e-6),
    )
    return periodic_score, striping_score


def _blocking_score(
    gray: np.ndarray,
    return_content: bool = False,
    content_mask: Optional[np.ndarray] = None,
) -> float:
    """JPEG 8x8 boundary-energy ratio.

    With `return_content=True` also returns a second value restricted to the
    content region. Scanned figures produce enormous spurious blocking scores
    from the content/margin boundary — a straight vertical or horizontal edge
    that happens to align with the 8-pixel grid far more often than chance —
    so the content-restricted value is the honest one for those images.
    """
    if gray.shape[0] < 16 or gray.shape[1] < 16:
        return (0.0, 0.0) if return_content else 0.0

    def _ratio(g: np.ndarray) -> float:
        vertical = np.abs(np.diff(g, axis=1))
        horizontal = np.abs(np.diff(g, axis=0))
        boundary_v = vertical[:, 7::8]
        boundary_h = horizontal[7::8, :]
        non_boundary_v = np.delete(vertical, np.arange(7, vertical.shape[1], 8), axis=1)
        non_boundary_h = np.delete(horizontal, np.arange(7, horizontal.shape[0], 8), axis=0)
        b = (float(np.mean(boundary_v)) + float(np.mean(boundary_h))) / 2.0
        nb = (float(np.mean(non_boundary_v)) + float(np.mean(non_boundary_h))) / 2.0
        return _normalize(b / (nb + 1e-6), 1.08, 1.8)

    whole = _ratio(gray)
    if not return_content:
        return whole

    if content_mask is not None and (content_mask > 0).any() and not (content_mask > 0).all():
        # Erode the mask so the content/margin boundary itself is excluded.
        eroded = cv2.erode(content_mask, np.ones((9, 9), np.uint8))
        if (eroded > 0).any():
            inner = gray.copy()
            inner[eroded == 0] = float(np.mean(gray[eroded > 0]))
            return whole, _ratio(inner)
    return whole, whole


def _ringing_score(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray.astype(np.uint8), 80, 180)
    if not np.any(edges):
        return 0.0
    dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    rim = (dilated > 0) & (edges == 0)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    if not np.any(rim):
        return 0.0
    return _normalize(float(np.mean(lap[rim])) / (float(np.mean(lap)) + 1e-6), 1.3, 3.2)


def _banding_score(gray: np.ndarray) -> float:
    smooth: np.ndarray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.4).astype(np.float32)
    residual: np.ndarray = np.abs(gray - smooth)
    smooth_ratio = float(np.mean(residual < 1.5))
    unique_ratio = len(np.unique(gray.astype(np.uint8))) / 256.0
    return float(np.clip((smooth_ratio - 0.30) * (1.0 - unique_ratio) * 2.5, 0.0, 1.0))


def _chroma_noise_score(img: np.ndarray) -> float:
    lab: np.ndarray = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    a: np.ndarray = lab[:, :, 1]
    b: np.ndarray = lab[:, :, 2]
    a_res: np.ndarray = a - cv2.GaussianBlur(a, (0, 0), sigmaX=1.2).astype(np.float32)
    b_res: np.ndarray = b - cv2.GaussianBlur(b, (0, 0), sigmaX=1.2).astype(np.float32)
    return _normalize((float(np.std(a_res)) + float(np.std(b_res))) / 2.0, 2.5, 13.0)


def _speckle_score(
    gray: np.ndarray,
    residual_abs: np.ndarray,
    content_mask: Optional[np.ndarray] = None,
) -> float:
    """Brightness-residual correlation, restricted to the content region.

    The margin pixels of a scanned figure carry zero residual by construction,
    which flattened the global correlation to ~0 and hid real speckle.
    """
    mean_blur: np.ndarray = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0).astype(np.float32)
    if content_mask is not None and (content_mask > 0).any():
        sel = content_mask > 0
        x = mean_blur[sel]
        y = residual_abs[sel]
    else:
        x = mean_blur.ravel()
        y = residual_abs.ravel()
    if x.size < 500:
        x, y = mean_blur.ravel(), residual_abs.ravel()
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        corr = 0.0
    return _normalize(float(corr), 0.15, 0.55)


def _speckle_slope_score(
    gray: np.ndarray,
    content_mask: Optional[np.ndarray] = None,
) -> float:
    """Multiplicative-noise detector: brightness-dependence of the noise floor.

    Divides the content region into luminance quintiles by local mean and
    measures the growth of the robust residual scale from darkest to brightest.
    Multiplicative (speckle) noise has noise_std proportional to signal, so the
    growth ratio is large (≈2-3x). Additive noise is flat (≈1x); clean images
    have no measurable residual anywhere. Measured calibration (2026-09-21):

        flat gray            slope  0.00
        additive gaussian    slope  0.56
        multiplicative speck slope  1.85-1.88

    unlike the old global-correlation score, this survives large flat margins
    because only content pixels are binned.
    """
    mean_blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0).astype(np.float32)
    high = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2).astype(np.float32)
    residual_abs = np.abs(gray - high)

    if content_mask is not None and (content_mask > 0).any():
        sel = content_mask > 0
        means = mean_blur[sel]
        resids = residual_abs[sel]
    else:
        means = mean_blur.ravel()
        resids = residual_abs.ravel()
    if means.size < 2000:
        return 0.0

    edges = np.percentile(means, [20, 40, 60, 80])
    bins = np.digitize(means, edges)
    scales = []
    for b in range(5):
        s = resids[bins == b]
        scales.append(float(np.median(s)) if s.size >= 200 else np.nan)
    scales = np.asarray(scales, dtype=np.float64)
    if np.isnan(scales).any() or scales[0] <= 1e-3:
        return 0.0
    growth = (scales[-1] - scales[0]) / max(scales[0], 1e-3)
    return _normalize(float(growth), 0.9, 1.9)


def _reduce_impulse(img: np.ndarray, profile: NoiseProfile) -> np.ndarray:
    """Impulse / salt-and-pepper removal, delegated to ai.preprocessing.impulse_denoiser.

    There is deliberately one implementation of impulse removal in the codebase.
    Stage 0 runs it before damage detection (which is where it matters, because a
    speck reaching Stage 2 gets masked as an emulsion flake and repainted by
    LaMa); this hook stays because the earlier stages can expose impulses that
    were not visible in the input — LaMa seams, a colorizer hallucinating a hot
    pixel — and because NoiseRouter's quality gate is a useful second opinion.
    Sharing the implementation means both call sites get the same guarantees.

    Two independent defects in the previous local-min/max version:

    1. The impulse test `(cur != med) & ((cur <= lo) | (cur >= hi))` flagged any
       pixel that was the minimum or maximum of its own k x k window. That is
       roughly 2/9 of all pixels in textured content whether or not noise is
       present — a local extremum is what an eyelash, a catch-light, or a hair
       strand *is*. Measured on a noise-free test image it rewrote 1.34% of
       pixels and destroyed 58% of the high-frequency energy, and its output
       PSNR sat at ~28.5 dB for every input density from 0 to 0.10: the filter
       converged to the same degraded image regardless of the noise, and at low
       density it made the image 3.45 dB *worse* than the input.

    2. `unresolved &= ~med_usable` inverted the bookkeeping. It cleared the
       grow-window flag wherever the median *was* usable, which is almost
       everywhere, so the loop broke after the first iteration and k never
       reached 5 or 7 no matter what max_k said. Dense noise was therefore
       under-treated at exactly the same time clean detail was over-treated.

    The replacement uses ROAD (Rank-Ordered Absolute Differences) to require
    that a candidate be unlike *most* of its neighbourhood rather than merely
    extreme within it, which is the distinction the old test could not make. See
    impulse_denoiser for the full rationale.

    `profile` is no longer consulted: the denoiser derives its own threshold from
    the image's robust noise scale, which is a better estimate than a family
    score bucketed into a fixed max_k. The parameter stays for call compatibility.
    """
    from ai.preprocessing.impulse_denoiser import ImpulseDenoiser

    return ImpulseDenoiser.apply(img)[0]


def _reduce_periodic(img: np.ndarray, profile: NoiseProfile) -> np.ndarray:
    channels = cv2.split(img)
    filtered = [_fft_notch_channel(channel) for channel in channels]
    out = cv2.merge(filtered)
    strength = 0.35 if profile.family_scores.get("periodic", 0.0) < 0.5 else 0.55
    return cv2.addWeighted(img, 1.0 - strength, out, strength, 0)


def _fft_notch_channel(channel: np.ndarray) -> np.ndarray:
    arr = channel.astype(np.float32)
    h, w = arr.shape
    spectrum = np.fft.fftshift(np.fft.fft2(arr - np.mean(arr)))
    mag = np.log1p(np.abs(spectrum))
    cy, cx = h // 2, w // 2
    dc_radius = max(8, min(h, w) // 22)
    mag[cy - dc_radius:cy + dc_radius + 1, cx - dc_radius:cx + dc_radius + 1] = 0.0
    threshold = np.mean(mag) + 4.2 * np.std(mag)
    coords = np.argwhere(mag > threshold)
    mask = np.ones((h, w), dtype=np.float32)
    for y, x in coords[:24]:
        if abs(y - cy) < dc_radius and abs(x - cx) < dc_radius:
            continue
        cv2.circle(mask, (int(x), int(y)), 3, 0.0, -1)
        sy, sx = 2 * cy - int(y), 2 * cx - int(x)
        if 0 <= sy < h and 0 <= sx < w:
            cv2.circle(mask, (sx, sy), 3, 0.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.2)
    restored = np.fft.ifft2(np.fft.ifftshift(spectrum * mask)).real + np.mean(arr)
    return np.clip(restored, 0, 255).astype(np.uint8)


def _reduce_compression(img: np.ndarray, profile: NoiseProfile) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    smooth_l = cv2.bilateralFilter(l, d=5, sigmaColor=12, sigmaSpace=5)
    # Blend mostly in flat regions so text, eyes, and hair edges survive.
    edges = cv2.Canny(l, 60, 150)
    flat = cv2.GaussianBlur((edges == 0).astype(np.float32), (9, 9), 0)
    strength = 0.22 + 0.18 * profile.family_scores.get("compression", 0.0)
    l_out = (l.astype(np.float32) * (1.0 - flat * strength) + smooth_l.astype(np.float32) * flat * strength)
    return cv2.cvtColor(cv2.merge([np.clip(l_out, 0, 255).astype(np.uint8), a, b]), cv2.COLOR_LAB2RGB)


def _reduce_chroma(img: np.ndarray, profile: NoiseProfile) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    strength = 0.25 + 0.25 * profile.family_scores.get("color", 0.0)
    for channel in (1, 2):
        src = lab[:, :, channel].astype(np.uint8)
        smooth = cv2.bilateralFilter(src, d=5, sigmaColor=10, sigmaSpace=5).astype(np.float32)
        lab[:, :, channel] = src.astype(np.float32) * (1.0 - strength) + smooth * strength
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _reduce_random(img: np.ndarray, profile: NoiseProfile) -> np.ndarray:
    random_score = max(profile.family_scores.get("random", 0.0), profile.family_scores.get("speckle", 0.0))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    h_luma = 3.0 + 4.0 * random_score
    l_nlm = cv2.fastNlMeansDenoising(l, None, h=h_luma, templateWindowSize=7, searchWindowSize=21)
    l_bilat = cv2.bilateralFilter(l_nlm, d=5, sigmaColor=8 + 12 * random_score, sigmaSpace=5)
    edges = cv2.Canny(l, 45, 130)
    protect = cv2.GaussianBlur((edges > 0).astype(np.float32), (7, 7), 0)
    strength = 0.30 + 0.25 * random_score
    l_out = l.astype(np.float32) * (1.0 - strength * (1.0 - protect)) + l_bilat.astype(np.float32) * strength * (1.0 - protect)
    return cv2.cvtColor(cv2.merge([np.clip(l_out, 0, 255).astype(np.uint8), a, b]), cv2.COLOR_LAB2RGB)


# Above this fraction of lost Canny edges a denoise candidate is rejected no
# matter how much residual noise it removes. Removing noise never requires
# erasing most of the image's structure. This ceiling used to carry an exemption
# for the impulse stage, on the theory that when an image is mostly speckle the
# lost "edges" are the speck outlines. That reasoning was sound but the exemption
# was unbounded: it let a candidate through at 58.7% edge loss, which had wiped a
# 57x75 px face down to a featureless blob. The premise is also obsolete now that
# _reduce_impulse only rewrites pixels it has positively identified as impulses —
# a precise filter cannot destroy half the frame's edges, so if one ever reports
# that much loss it is malfunctioning and should be rejected, not exempted.
_MAX_EDGE_LOSS = 0.55

# Minimum fraction of a face region's high-frequency energy that must survive.
# Some loss is the noise leaving; a collapse to a featureless patch is not.
_MIN_FACE_DETAIL_RETENTION = 0.40


def _detail_energy(gray: np.ndarray) -> float:
    """High-frequency energy of a grayscale patch. Smearing drives this to ~0."""
    g = gray.astype(np.float32)
    return float(np.std(g - cv2.GaussianBlur(g, (0, 0), 1.5)))


def _face_structure_survives(
    previous: np.ndarray,
    candidate: np.ndarray,
    context: PipelineContext,
) -> Tuple[bool, str]:
    """
    Reject a denoise candidate that flattens a detected face.

    Pixel-difference guards cannot be used on a heavily noisy face: cleaning it
    legitimately changes pixels by a lot. What must not happen is the face
    losing its structure, so compare high-frequency energy inside each face box
    instead of raw pixel deltas.
    """
    faces = context.image_metadata.get("faces") or []
    h, w = previous.shape[:2]
    for face in faces:
        bbox = face.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        prev_crop = cv2.cvtColor(previous[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        cand_crop = cv2.cvtColor(candidate[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        prev_detail = _detail_energy(prev_crop)
        cand_detail = _detail_energy(cand_crop)
        if prev_detail <= 1e-3:
            continue
        retention = cand_detail / prev_detail
        if retention < _MIN_FACE_DETAIL_RETENTION:
            return False, (
                f"face structure collapsed (detail retention {retention:.0%} < "
                f"{_MIN_FACE_DETAIL_RETENTION:.0%}, {prev_detail:.1f}->{cand_detail:.1f})"
            )
    return True, "face structure preserved"


def _candidate_passes(
    previous: np.ndarray,
    candidate: np.ndarray,
    original: np.ndarray,
    context: PipelineContext,
    stage_name: str,
) -> Tuple[bool, str]:
    if candidate.shape != previous.shape or candidate.dtype != np.uint8:
        return False, "invalid candidate shape or dtype"

    prev_gray: np.ndarray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY).astype(np.float32)
    cand_gray: np.ndarray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY).astype(np.float32)
    orig_gray: np.ndarray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)

    prev_blur: np.ndarray = cv2.GaussianBlur(prev_gray, (3, 3), 0).astype(np.float32)
    cand_blur: np.ndarray = cv2.GaussianBlur(cand_gray, (3, 3), 0).astype(np.float32)
    prev_edges = cv2.Canny(prev_blur.astype(np.uint8), 60, 150)
    cand_edges = cv2.Canny(cand_blur.astype(np.uint8), 60, 150)
    edge_loss = (float(np.sum(prev_edges > 0)) - float(np.sum(cand_edges > 0))) / max(float(np.sum(prev_edges > 0)), 1.0)
    mean_luma_shift = abs(float(np.mean(cand_gray)) - float(np.mean(prev_gray)))
    original_luma_shift = abs(float(np.mean(cand_gray)) - float(np.mean(orig_gray)))

    if edge_loss > 0.18:
        # Edge loss above 18% on a heavily noisy image is expected and desirable:
        # most of the "edges" being lost ARE the noise spikes. But this branch
        # used to accept ANY edge loss — 58.7% wiped a 57x75 px face down to a
        # featureless blob and still passed, because global residual had dropped.
        # Two things bound it now: an absolute ceiling, and the face-structure
        # check below (which this branch previously returned before reaching).
        prev_blur12: np.ndarray = cv2.GaussianBlur(prev_gray, (0, 0), 1.2).astype(np.float32)
        cand_blur12: np.ndarray = cv2.GaussianBlur(cand_gray, (0, 0), 1.2).astype(np.float32)
        residual_prev = float(np.std(prev_gray - prev_blur12))
        residual_cand = float(np.std(cand_gray - cand_blur12))
        if edge_loss > _MAX_EDGE_LOSS:
            return False, (
                f"{stage_name}: edge loss {edge_loss:.1%} exceeds the "
                f"{_MAX_EDGE_LOSS:.0%} ceiling — structural collapse, not noise "
                f"removal"
            )
        if not (residual_cand < residual_prev * 0.80 and mean_luma_shift <= 6.0):
            return False, f"{stage_name}: edge loss {edge_loss:.1%}"
        ok, why = _face_structure_survives(previous, candidate, context)
        if not ok:
            return False, why
        return True, f"noise reduced despite edge_loss={edge_loss:.1%}, residual {residual_prev:.2f}->{residual_cand:.2f}"
    if mean_luma_shift > 8.0 or original_luma_shift > 10.0:
        return False, f"luma shift {mean_luma_shift:.1f}/{original_luma_shift:.1f}"

    if context.image_metadata.get("faces"):
        for face in context.image_metadata.get("faces", []):
            bbox = face.get("bbox")
            if not bbox:
                continue
            x0, y0, x1, y1 = [int(v) for v in bbox]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(previous.shape[1], x1), min(previous.shape[0], y1)
            if x1 <= x0 or y1 <= y0:
                continue
            face_diff = float(np.mean(np.abs(candidate[y0:y1, x0:x1].astype(np.float32) - previous[y0:y1, x0:x1].astype(np.float32))))
            # BUG FIX 4: face_diff threshold of 7.0 was rejecting valid denoising.
            # A noisy face has per-pixel variance >> 7.0; cleaning it up WILL change
            # pixel values by >7 on average. Threshold raised to 18.0 so only
            # genuinely destructive changes (blurring, structural loss) are blocked.
            if face_diff > 18.0:
                return False, f"face region changed too much ({face_diff:.1f})"

    prev_blur12_f: np.ndarray = cv2.GaussianBlur(prev_gray, (0, 0), 1.2).astype(np.float32)
    cand_blur12_f: np.ndarray = cv2.GaussianBlur(cand_gray, (0, 0), 1.2).astype(np.float32)
    orig_blur12_f: np.ndarray = cv2.GaussianBlur(orig_gray, (0, 0), 1.2).astype(np.float32)
    residual_prev = float(np.std(prev_gray - prev_blur12_f))
    residual_cand = float(np.std(cand_gray - cand_blur12_f))
    residual_orig = float(np.std(orig_gray - orig_blur12_f))
    # BUG FIX 4b: the old check compared candidate residual against the PREVIOUS
    # stage (which might also be noisy). A multi-stage pipeline would always pass
    # this check even if every stage made things slightly worse, as long as it
    # wasn't WORSE than the immediately prior noisy version. Now we measure
    # against the original so the overall reduction is always confirmed.
    if residual_cand > residual_orig * 1.02:
        return False, f"noise not reduced vs original (orig={residual_orig:.2f}, cand={residual_cand:.2f})"

    return True, f"edge_loss={edge_loss:.1%}, residual {residual_prev:.2f}->{residual_cand:.2f}"
