"""
speck_suppressor.py — Suppress a dense field of fine bright specks (emulsion
loss, paper-weave highlights, salt noise) with a white top-hat.

Why this exists alongside HalftoneSuppressor
--------------------------------------------
HalftoneSuppressor notches *periodic* mesh out of the FFT. A worn print's
emulsion loss is not periodic: it is an aperiodic scatter of 1-2px bright
specks, so the spectral peak test finds nothing and the mesh survives. On a
447x447 sepia portrait HalftoneSuppressor reported "no periodic mesh detected"
while ~40% of pixels carried a >6 L-unit bright speck.

Why a top-hat rather than mask-and-inpaint
------------------------------------------
Mask-and-inpaint is the right tool for discrete holes. It is the wrong tool for
a dense lacy field: the mask reaches 20-30% of the frame, and both LaMa and the
Gaussian pyramid fill turn that into mush (the first attempt smeared the top of
a subject's head grey). A white top-hat isolates bright structures *smaller than
the structuring element* and nothing else, so subtracting it removes the speckle
while leaving every feature larger than the kernel — eyes, hair mass, collar
edges — bit-identical outside the specks.

The kernel is deliberately fixed at 3px. Measured on a real worn print, k=3
removes 94% of the speck amplitude for a 33% cost in total high-frequency
energy; k=5 and k=7 remove barely more speckle but cost 58-68%, which reads as
waxy skin and turns the remaining flakes into blobby light patches. Larger
emulsion losses are left for the detect → refine → inpaint chain, which handles
discrete damage well and does so more cleanly once this step has taken the fine
field out from underneath it.

The correction is soft-gated: it ramps in from `threshold` to `2 * threshold`
rather than switching at it, so genuine small highlights (catch-lights, teeth,
jewellery) below the threshold survive and specks do not get left with a hard
rim. Only L is touched; chroma is carried through untouched.
"""

from __future__ import annotations

import cv2
import numpy as np

from ai.pipeline.pipeline_context import PipelineContext

# Structuring element side, in pixels. See the module docstring for why this is
# not scaled with resolution: it targets 1-2px specks, which is what emulsion
# loss and paper weave produce at any scan size.
_KERNEL = 3

# Base L-channel threshold floor.
_MIN_THRESHOLD = 6.0

# Minimum coverage to consider a field of specks.
_MIN_COVERAGE = 0.08
_MIN_AMPLITUDE = 5.0

# Spatial autocorrelation cutoff:
# Uncorrelated stochastic Gaussian/sensor noise has rho < 0.18.
# Physical emulsion blemishes have rho >= 0.20.
_MIN_SPATIAL_AUTOCORR = 0.18

# Macroscopic scene edge co-location cutoff:
# If > 55% of candidate peaks sit on macroscopic scene edges (macro gradient > 25.0),
# the features are natural scene geometry (branches, horizons) rather than a distributed
# emulsion blemish field.
_MAX_EDGE_COLOCATION = 0.55


def _tophat(L: np.ndarray, k: int = _KERNEL) -> np.ndarray:
    """Bright structures smaller than a k x k ellipse."""
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return L - cv2.morphologyEx(L, cv2.MORPH_OPEN, se)


class SpeckSuppressor:
    """Remove a dense fine bright-speck field, in place on the context."""

    @staticmethod
    def measure(rgb: np.ndarray) -> dict:
        """Analyze the image and return multi-factor metrics for speck field detection."""
        L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 1. Robust estimation of high-frequency noise floor sigma via MAD
        med = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
        mad = float(np.median(np.abs(gray - med)))
        sigma_est = 1.4826 * mad
        adaptive_thresh = max(_MIN_THRESHOLD, 1.8 * sigma_est)

        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_KERNEL, _KERNEL))
        th_white = L - cv2.morphologyEx(L, cv2.MORPH_OPEN, se)
        th_black = cv2.morphologyEx(L, cv2.MORPH_CLOSE, se) - L

        amplitude = float(np.std(th_white))
        cand_mask = th_white > adaptive_thresh
        coverage = float(np.mean(cand_mask))

        # 2. Spatial Autocorrelation (normalized lag-1 horizontal + vertical)
        th_c = th_white - np.mean(th_white)
        denom = np.sum(th_c ** 2) + 1e-6
        ac_h = np.sum(th_c[:, :-1] * th_c[:, 1:]) / denom
        ac_v = np.sum(th_c[:-1, :] * th_c[1:, :]) / denom
        spatial_autocorr = float((ac_h + ac_v) / 2.0)

        # 3. Macroscopic Scene Edge Co-location
        # Heavy Gaussian blur (15x15) isolates macroscopic structural edges (e.g. tree branches)
        # without capturing localized 1-3px speck blemishes.
        macro = cv2.GaussianBlur(gray, (15, 15), 0)
        gx = cv2.Sobel(macro, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(macro, cv2.CV_32F, 0, 1)
        macro_edge_mag = np.sqrt(gx**2 + gy**2)

        cand_count = int(np.sum(cand_mask))
        if cand_count > 0:
            edge_colocation = float(np.sum(cand_mask & (macro_edge_mag > 25.0)) / cand_count)
        else:
            edge_colocation = 0.0

        # 4. Tail Asymmetry (telemetry / diagnostic logging only — not a hard gate)
        w_tail = int(np.sum(th_white > adaptive_thresh))
        b_tail = int(np.sum(th_black > adaptive_thresh))
        tail_asymmetry = float(w_tail / max(b_tail, 1))

        return {
            "amplitude": round(amplitude, 2),
            "coverage": round(coverage, 4),
            "adaptive_threshold": round(adaptive_thresh, 2),
            "spatial_autocorr": round(spatial_autocorr, 4),
            "edge_colocation": round(edge_colocation, 4),
            "tail_asymmetry": round(tail_asymmetry, 3),
        }

    @staticmethod
    def apply(
        rgb: np.ndarray,
        threshold: float = _MIN_THRESHOLD,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Subtract the soft-gated white top-hat from L, preserving chroma."""
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = lab[:, :, 0]
        th = _tophat(L)
        # Ramp 0 -> 1 across [threshold, 2 * threshold] so the correction eases in
        gain = np.clip((th - threshold) / max(threshold, 1e-6), 0.0, 1.0)
        lab[:, :, 0] = np.clip(L - th * gain * strength, 0.0, 255.0)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    @staticmethod
    def suppress(context: PipelineContext) -> bool:
        """Suppress the speck field on context.working_image if verified present."""
        rgb = context.working_image
        metrics = SpeckSuppressor.measure(rgb)
        context.image_metadata["speck_field"] = metrics

        amplitude = metrics["amplitude"]
        coverage = metrics["coverage"]
        adaptive_thresh = metrics["adaptive_threshold"]
        spatial_autocorr = metrics["spatial_autocorr"]
        edge_colocation = metrics["edge_colocation"]

        # Gating checks:
        if coverage < _MIN_COVERAGE or amplitude < _MIN_AMPLITUDE:
            context.add_log(
                f"SpeckSuppressor: no dense speck field "
                f"(amplitude={amplitude:.1f}, coverage={coverage:.1%}) — skipping."
            )
            return False

        if spatial_autocorr < _MIN_SPATIAL_AUTOCORR:
            context.add_log(
                f"SpeckSuppressor: rejected stochastic noise field "
                f"(spatial_autocorr={spatial_autocorr:.3f} < {_MIN_SPATIAL_AUTOCORR}) — "
                f"passing through to Stage 5 NLM/Bilateral denoiser."
            )
            return False

        if edge_colocation > _MAX_EDGE_COLOCATION:
            context.add_log(
                f"SpeckSuppressor: rejected macroscopic scene edges "
                f"(edge_colocation={edge_colocation:.1%} > {_MAX_EDGE_COLOCATION:.1%}) — "
                f"preserving natural scene structures."
            )
            return False

        context.working_image = SpeckSuppressor.apply(rgb, threshold=adaptive_thresh)
        after_metrics = SpeckSuppressor.measure(context.working_image)
        context.image_metadata["speck_field"]["amplitude_after"] = after_metrics["amplitude"]
        context.image_metadata["speck_field"]["coverage_after"] = after_metrics["coverage"]
        context.add_log(
            f"SpeckSuppressor: verified speck field suppressed "
            f"(k={_KERNEL}, tau={adaptive_thresh:.1f}, rho={spatial_autocorr:.3f}, edge_coloc={edge_colocation:.1%}; "
            f"amplitude {amplitude:.1f} -> {after_metrics['amplitude']:.1f}, "
            f"coverage {coverage:.1%} -> {after_metrics['coverage']:.1%})."
        )
        return True
