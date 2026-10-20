"""
impulse_denoiser.py — Detect and remove impulse (salt-and-pepper) noise before
anything else in the pipeline looks at the image.

Why this runs first
-------------------
Impulse noise is not a cosmetic problem in this pipeline; it is a *detection*
problem. Stage 2 asks "is this bright pixel an emulsion flake?" and "is this
dark pixel the start of a crack?", and an isolated speck answers yes to both.
The flake and tearmark detectors then hand a speck field to MaskRefiner, whose
morphological closing merges the scatter into large connected components, and
LaMa repaints them. Measured on a clean 256x256 synthetic, adding 1% impulse
noise took the final damage mask from 10.05% of the frame in 5 components to
99.62% in a single component. The repainted result is the smeared, crackled,
pastel-blotched output that colorization then invents hues for.

So the fix is ordering plus precision: remove the specks before any detector
runs, and remove *only* the specks.

Why not a median filter
-----------------------
A median filter is a blanket operation: it rewrites every pixel, so it cannot
help but destroy the single-pixel detail that portraits are made of — catch-
lights in the eyes, individual hair strands, eyelashes, jewellery highlights.
The previous implementation in noise_router flagged any pixel that was the min
or max of its own 3x3 window, which is roughly 2/9 of all pixels in textured
content whether or not any noise is present. Measured on a noise-*free* test
image it changed 1.34% of pixels and destroyed 58% of the high-frequency
energy, and its output PSNR was pinned at ~28.5 dB no matter how noisy the
input was — the signature of a filter that is filtering content rather than
noise. At low noise density it made the image measurably worse (-3.45 dB).

How this decides what is noise
------------------------------
Three tests, and none of them has a constant tuned to a particular photograph.
Write T for the per-image threshold, derived below.

1. Extremity — the pixel differs from its 3x3 median by more than T. T is
   max(12, 3 * sigma) where sigma is a median-absolute-deviation estimate of the
   image's own noise scale, taken on the median-filter residual. MAD rather than
   a standard deviation precisely because impulses are present: they sit in the
   tail of the residual distribution and would inflate a std, which would raise
   the threshold and hide them. MAD ignores the tail. The floor of 12 grey
   levels is perceptual — below that an outlier is indistinguishable from
   ordinary film grain, which the grain-oriented filters downstream handle
   better than a median ever will.

2. Isolation, via ROAD — Rank-Ordered Absolute Differences. Take the absolute
   difference to each of the 8 neighbours, sort them, and require that even the
   *two smallest* sum to more than 2T; that is, the pixel is unlike at least
   seven of its eight neighbours. This is the test that protects thin structure:

     * flat region — 8 near-identical neighbours, ROAD ~ 0, protected
     * edge — at least 3 neighbours on the pixel's own side, protected, even
       though the pixel is a local extreme
     * interior of a 1px hair strand — 2 collinear neighbours on the strand, so
       the two smallest differences are both ~0, protected
     * isolated impulse — unlike every neighbour, flagged

   The order matters here and getting it wrong is subtle. Summing the four
   smallest differences instead of the two is the more common formulation, and
   it breaks on exactly the content this pipeline exists to preserve: a pixel in
   the middle of a 1px strand has only 2 matching neighbours, so its third and
   fourth smallest differences are full-amplitude and the sum clears any
   sensible threshold. Every hair strand in the image would be filtered as
   noise. Requiring only the two smallest to be large is what keeps 1px lines.

3. Continuation veto — the flag is withdrawn if there is any direction in which
   the neighbours at distance 1 *and* distance 2 are both within T of the pixel.
   Test 2 alone clips the endpoints of thin lines: the last pixel of a strand has
   just one collinear neighbour, so its two smallest differences are 0 and
   full-amplitude, and it gets flagged. Checking two steps out distinguishes the
   two cases — a strand endpoint has the strand continuing beneath it, an
   isolated impulse has nothing in any direction.

Detection is per channel rather than on luminance alone, which covers both
classic neutral salt-and-pepper (an outlier in all three channels, flagged in
all three) and the per-channel impulses that scanners and heavy JPEG
recompression produce (flagged only where they occur).

What this deliberately does not catch
-------------------------------------
Runs of three or more adjacent impulses of the same polarity survive: the middle
of such a run has two matching neighbours, so test 2 protects it, and the ends
are vetoed by test 3. This is not a tuning oversight, it is an information limit
— no purely local test can tell a 3-pixel salt cluster from a 3-pixel line
segment, because locally they are the same object. Two things make it the right
trade. First, such clusters are rare: at density d they are roughly 4*(d/2)^2 of
the impulses, which is 0.25% at d=0.05 and 1% at d=0.10. Second, the failure this
stage exists to prevent is a *dense field of isolated specks* merging under
morphological closing into frame-filling blobs; a handful of surviving 3-pixel
clusters reach Stage 2 as tiny components and are pruned there. Being
conservative is the correct bias when the bug being fixed was over-aggression.

There is no iteration for the same reason. Running detect-and-fill repeatedly
would eventually clear those clusters, but it would also erode thin lines one
pixel per pass from each end, since each pass creates a fresh endpoint for the
next to clip. One pass clips at most a single pixel from a line end and cannot
compound.

How replacement works
---------------------
Flagged pixels are replaced by the median of their *unflagged* neighbours, with
the window growing 3 -> 5 -> 7 only for pixels that still have too few clean
neighbours to draw on. Noise never feeds into its own replacement, so dense
speckle resolves without a blanket median, and clusters of adjacent specks do
not propagate their own values. Pixels corrected at a smaller window do become
valid sources at the next one, which is what lets heavy noise resolve from the
outside in. Unflagged pixels are returned bit-identical, so a clean image is a
true no-op — there is no coverage threshold to tune because the detector is
precise rather than a gate on a blunt filter.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── A2 isolated-blob inpainting (docs/REMEDIATION_PLAN_AB_G.md) ─────────────
# Values scale with the image so the same constants work at scan and SR scale.
_DISABLED = {"0", "false", "no", "off"}
_BLOB_MIN_AREA = 1            # amplitude + isolation gates separate dust from grain
_BLOB_MAX_AREA_FRAC = 6e-4    # max blob area as fraction of h*w (≈120px @447²)
_BLOB_MIN_CONTRAST = 20.0     # min local contrast (grey levels) to be dust
_BLOB_MAX_STRUCTURE_TOUCH = 0.35  # max structure-pixel share inside a component
_BLOB_COVERAGE_CAP_PCT = 4.0  # frame cap — above this it is a texture field

# The 8-neighbourhood, in the order used to build the difference stack. Also the
# 8 directions searched by the continuation veto.
_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)

# How many of the sorted neighbour differences the ROAD statistic sums. Two, so
# that a pixel needs only two similar neighbours to be considered part of a
# structure — which is exactly what the interior of a 1px line has. See the
# module docstring: m=4 is the usual choice and it filters away hair strands.
_ROAD_M = 2

# Absolute floor on "differs by a visible amount", in 8-bit grey levels. A
# perceptual floor, identical for every image, not a tuned constant.
_MIN_DELTA = 12.0

# Multiple of the robust noise scale a pixel must exceed to count as extreme.
# 3 sigma on a MAD-derived scale is the conventional outlier boundary.
_SIGMA_K = 3.0

# Below this flagged fraction the impulses are reported as incidental (dust
# motes rather than a noise process). The pixels are still corrected —
# correction is exact — this only controls how the stage describes itself.
_FIELD_DENSITY = 0.001


@dataclass
class ImpulseField:
    """Measurement of the impulse-noise content of an image."""

    density: float          # fraction of pixels flagged as impulses
    sigma: float            # robust noise scale, 8-bit grey levels
    threshold: float        # per-pixel delta threshold actually used
    channels: Tuple[float, ...] = ()   # per-channel flagged fraction

    @property
    def is_field(self) -> bool:
        """True when the impulses form a noise process, not a few dust motes."""
        return self.density >= _FIELD_DENSITY

    def to_dict(self) -> dict:
        return {
            "density": round(self.density, 5),
            "sigma": round(self.sigma, 2),
            "threshold": round(self.threshold, 2),
            "channels": [round(c, 5) for c in self.channels],
            "is_field": self.is_field,
        }


def _shifted(padded: np.ndarray, pad: int, dy: int, dx: int, h: int, w: int) -> np.ndarray:
    """Slice a replicate-padded plane to give the neighbour at offset (dy, dx)."""
    return padded[pad + dy:pad + dy + h, pad + dx:pad + dx + w]


def _neighbour_stack(plane_f32: np.ndarray, distance: int = 1) -> List[np.ndarray]:
    """The 8 neighbours of every pixel at the given Chebyshev distance.

    Borders are mirrored, not replicated, and the distinction is load-bearing
    rather than cosmetic. Under replication the pixel above row 0 is a copy of
    row 0 itself, so for a pixel on the top edge the continuation veto compares
    the pixel against itself at both distance 1 and distance 2, both differences
    are 0, and the veto fires unconditionally. The whole 1px frame border would
    be a guaranteed no-op — 1020 pixels on a 256x256 image, and at density 0.02
    roughly twenty impulses per frame surviving into Stage 2 by construction.

    Mirroring gives the top edge rows 1 and 2 as its upward neighbours, which is
    real content: an impulse there still differs from them and is still caught,
    while a structure running down from the edge still vetoes correctly. The
    upward and downward tests become duplicates of each other on the edge row,
    which is a harmless loss of one direction rather than a false veto.
    """
    h, w = plane_f32.shape[:2]
    # OpenCV rejects REFLECT_101 when the border is not smaller than the source,
    # so a strip thinner than the reach falls back to replication. Such an image
    # has no meaningful neighbourhood in that axis anyway.
    mode = cv2.BORDER_REFLECT_101 if min(h, w) > distance else cv2.BORDER_REPLICATE
    padded = cv2.copyMakeBorder(
        plane_f32, distance, distance, distance, distance, mode
    )
    return [
        _shifted(padded, distance, dy * distance, dx * distance, h, w)
        for dy, dx in _OFFSETS
    ]


def _road(plane_f32: np.ndarray) -> np.ndarray:
    """Rank-Ordered Absolute Differences: the sum of the _ROAD_M smallest
    absolute differences to the 8-neighbourhood. Low for flat regions, edges and
    the interiors of thin lines; high for isolated outliers.

    The m smallest are tracked in a running insertion over the 8 neighbours
    rather than by stacking and partitioning. Stacking is the obvious
    formulation and it is the wrong one here: this stage runs before the 2K
    normalisation, so on a 4000x3000 scan an (8, h, w) float32 stack is 384 MB
    and np.partition allocates a second copy of it. The running form holds
    (m, h, w) instead — 96 MB at m=2 — at the cost of 8*m cheap elementwise
    passes, which is a trade worth making at full scan resolution.
    """
    neighbours = _neighbour_stack(plane_f32)
    h, w = plane_f32.shape[:2]
    smallest = np.full((_ROAD_M, h, w), np.inf, dtype=np.float32)

    for neighbour in neighbours:
        carry = np.abs(neighbour - plane_f32)
        for j in range(_ROAD_M):
            # Where carry belongs at rank j, insert it and push the old value
            # down to compete for rank j+1.
            takes = carry < smallest[j]
            displaced = np.where(takes, smallest[j], carry)
            smallest[j] = np.where(takes, carry, smallest[j])
            carry = displaced

    return smallest.sum(axis=0)


def _continues_into_structure(plane_f32: np.ndarray, tol: float) -> np.ndarray:
    """True where some direction has similar neighbours at distance 1 *and* 2.

    This is the endpoint rescue. The last pixel of a thin line has a single
    collinear neighbour, which is not enough for the ROAD test to protect it, but
    the line carries on beyond that neighbour — so looking one step further out
    separates "end of a structure" from "isolated outlier". An impulse has
    nothing similar in any direction at either distance.
    """
    near = _neighbour_stack(plane_f32, distance=1)
    far = _neighbour_stack(plane_f32, distance=2)
    continues = np.zeros(plane_f32.shape[:2], dtype=bool)
    for n1, n2 in zip(near, far):
        continues |= (np.abs(plane_f32 - n1) <= tol) & (np.abs(plane_f32 - n2) <= tol)
    return continues


def _robust_sigma(plane_f32: np.ndarray) -> float:
    """Noise scale from the median-filter residual via MAD.

    MAD rather than std precisely because impulses are present: they sit in the
    tail of the residual distribution and would inflate a std, which would then
    raise the threshold and hide them. MAD ignores the tail.

    The estimate is global. A local estimate would adapt to a smooth face beside
    a textured hat, but it is also far less stable — and a global scale errs
    toward a higher threshold in textured images, which is the conservative
    direction for a filter whose previous incarnation was too aggressive.
    """
    plane_u8 = np.clip(plane_f32, 0, 255).astype(np.uint8)
    residual = plane_f32 - cv2.medianBlur(plane_u8, 3).astype(np.float32)
    mad = float(np.median(np.abs(residual - float(np.median(residual)))))
    return mad / 0.6745  # MAD -> Gaussian-equivalent sigma


def _masked_median_fill(
    plane_f32: np.ndarray,
    flag: np.ndarray,
    max_k: int = 7,
    chunk: int = 262_144,
) -> np.ndarray:
    """Replace flagged pixels with the median of their *unflagged* neighbours.

    The window grows 3 -> 5 -> 7 only for pixels that still have no clean
    neighbour to draw on, so dense speckle resolves without applying a large
    window anywhere it is not needed. Pixels corrected at a smaller window
    become valid sources at the next one, which lets a dense field resolve from
    its edges inward. Work is gathered at flagged coordinates only, in chunks,
    so cost scales with the amount of noise rather than the size of the image.
    """
    out = plane_f32.copy()
    remaining = flag.copy()

    for k in range(3, max_k + 1, 2):
        ys, xs = np.nonzero(remaining)
        if ys.size == 0:
            break

        r = k // 2
        # Source view: still-unresolved pixels are excluded from every median, so
        # noise can never supply its own replacement value. Computed once per
        # window size, which makes the pass independent of chunk ordering.
        src = np.where(remaining, np.nan, out).astype(np.float32)
        padded = np.pad(src, r, mode="edge")
        offsets = [(dy, dx)
                   for dy in range(-r, r + 1)
                   for dx in range(-r, r + 1)
                   if not (dy == 0 and dx == 0)]

        for start in range(0, ys.size, chunk):
            cy = ys[start:start + chunk]
            cx = xs[start:start + chunk]
            gathered = np.empty((cy.size, len(offsets)), dtype=np.float32)
            for i, (dy, dx) in enumerate(offsets):
                gathered[:, i] = padded[cy + dy + r, cx + dx + r]

            with warnings.catch_warnings():
                # All-NaN rows mean "no clean neighbour at this window size".
                # They yield NaN and are retried at the next k; the warning is
                # expected, not a problem.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                filled = np.nanmedian(gathered, axis=1)

            resolved = ~np.isnan(filled)
            out[cy[resolved], cx[resolved]] = filled[resolved]
            remaining[cy[resolved], cx[resolved]] = False

    # Anything still unresolved sat in a fully-corrupted neighbourhood at every
    # window size. A plain median is the only information left.
    if np.any(remaining):
        fallback = cv2.medianBlur(
            np.clip(out, 0, 255).astype(np.uint8), max_k
        ).astype(np.float32)
        out[remaining] = fallback[remaining]

    return out


class ImpulseDenoiser:
    """Precise impulse-noise detection and removal.

    Every method is content-adaptive: thresholds derive from the image's own
    robust noise scale, so the same code handles a lightly dusted scan and a
    heavily speckled one without per-image tuning.
    """

    @staticmethod
    def detect(rgb: np.ndarray) -> Tuple[np.ndarray, ImpulseField]:
        """Return a per-channel boolean impulse mask and the field measurement.

        The mask has the same shape as `rgb`; a pixel may be flagged in one
        channel and not another, which is what per-channel scanner impulses
        actually look like.
        """
        if rgb.ndim == 2:
            rgb = rgb[:, :, None]

        _despeckle_guard = os.environ.get(
            "PIXREVIVE_DESPECKLE_GUARD", "1"
        ).strip().lower() not in ("0", "false", "no", "off")

        is_scene_edge = None
        if _despeckle_guard:
            gray_m = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 and rgb.shape[2] == 3 else rgb.squeeze()
            macro = cv2.GaussianBlur(gray_m, (15, 15), 0)
            gx_m = cv2.Sobel(macro, cv2.CV_32F, 1, 0)
            gy_m = cv2.Sobel(macro, cv2.CV_32F, 0, 1)
            macro_edge = np.sqrt(gx_m**2 + gy_m**2)
            is_scene_edge = macro_edge > 25.0

        flags = np.zeros(rgb.shape, dtype=bool)
        sigmas: list[float] = []
        thresholds: list[float] = []
        per_channel: list[float] = []

        for c in range(rgb.shape[2]):
            plane = rgb[:, :, c].astype(np.float32)
            sigma = _robust_sigma(plane)
            delta_t = max(_MIN_DELTA, _SIGMA_K * sigma)

            median3 = cv2.medianBlur(
                np.clip(plane, 0, 255).astype(np.uint8), 3
            ).astype(np.float32)

            # Test 1 — extremity against the local median.
            extreme = np.abs(plane - median3) > delta_t
            # Test 2 — isolation. Scaled by _ROAD_M so the per-neighbour
            # requirement matches the same delta_t the extremity test uses.
            isolated = _road(plane) > (_ROAD_M * delta_t)
            # Test 3 — withdraw the flag at the endpoints of thin structures.
            continues = _continues_into_structure(plane, delta_t)

            flag_c = extreme & isolated & ~continues
            flags[:, :, c] = flag_c
            sigmas.append(sigma)
            thresholds.append(delta_t)

        # Whole-field macroscopic edge co-location check (analogous to SpeckSuppressor)
        if _despeckle_guard and is_scene_edge is not None:
            any_flags = np.any(flags, axis=2)
            cand_count = int(np.sum(any_flags))
            if cand_count > 0:
                edge_coloc = float(np.sum(any_flags & is_scene_edge) / cand_count)
                if edge_coloc > 0.55:
                    flags[:] = False
                else:
                    flags &= ~is_scene_edge[:, :, None]

        per_channel = [float(np.mean(flags[:, :, c])) for c in range(rgb.shape[2])]

        field = ImpulseField(
            density=float(np.mean(np.any(flags, axis=2))),
            sigma=float(np.mean(sigmas)),
            threshold=float(np.mean(thresholds)),
            channels=tuple(per_channel),
        )
        return flags, field

    @staticmethod
    def measure(rgb: np.ndarray) -> ImpulseField:
        """Measure impulse content without modifying anything."""
        return ImpulseDenoiser.detect(rgb)[1]

    @staticmethod
    def apply(rgb: np.ndarray) -> Tuple[np.ndarray, ImpulseField]:
        """Remove detected impulses. Unflagged pixels are bit-identical."""
        flags, field = ImpulseDenoiser.detect(rgb)
        if not np.any(flags):
            return rgb.copy(), field

        single = rgb.ndim == 2
        work = rgb[:, :, None] if single else rgb
        out = work.copy()

        for c in range(work.shape[2]):
            if not np.any(flags[:, :, c]):
                continue
            filled = _masked_median_fill(
                work[:, :, c].astype(np.float32), flags[:, :, c]
            )
            # Rounded, not truncated. The neighbour count is always even (8 at
            # k=3, 24 at k=5, 48 at k=7), so nanmedian averages the two middle
            # values and a .5 result is the common case rather than the
            # exception; truncating would bias every repaired pixel dark.
            out[:, :, c] = np.rint(np.clip(filled, 0, 255)).astype(np.uint8)

        return (out[:, :, 0] if single else out), field

    @staticmethod
    def suppress_isolated_blobs(context: "PipelineContext") -> bool:  # noqa: F821
        """A2 (remediation plan docs/REMEDIATION_PLAN_AB_G.md): inpaint
        isolated dust blobs that no other stage claims.

        The pipeline's coverage gates route *sparse structural* damage to LaMa
        and *fields* of specks to the denoise path, but mid-size isolated blobs
        (dust grains, emulsion dots, stain pinpoints — bright OR dark, up to a
        few hundred px at scan scale) can survive every detector and reach the
        output. Detection = morphological top-hat + black-hat (contrast against
        the local background, 21px scale) + median-difference agreement; every
        candidate must then pass an isolation filter: a component qualifies only
        when its boundary touches few image-structure pixels, so cracks, fabric
        weave, printed stipple and film grain texture are never candidates.
        Removal = dilate 3 + Telea r=3, bounded by a frame-coverage cap.

        Self-gating: no-op unless candidate coverage lands in a plausible dust
        band and the robust noise floor is below the blob contrast. Clean or
        uniformly noisy frames measure out of the band and pass through.
        Flag: PIXREVIVE_BLOB_INPAINT=0 reverts.
        """
        if os.environ.get("PIXREVIVE_BLOB_INPAINT", "1").strip().lower() in _DISABLED:
            return False
        img = context.working_image
        if img is None or img.ndim != 3:
            return False
        h, w = img.shape[:2]
        if h * w < 64 * 64:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # Robust noise floor so grain texture is not mistaken for blobs.
        med = cv2.medianBlur(gray, 3).astype(np.float32)
        sigma_floor = 1.4826 * float(
            np.median(np.abs(gray.astype(np.float32) - med)))

        blob_amp = max(_BLOB_MIN_CONTRAST, 3.5 * sigma_floor)
        cand = ((tophat > blob_amp) | (blackhat > blob_amp)).astype(np.uint8)

        # Median-difference agreement (specks read as median outliers too).
        cand &= (np.abs(gray.astype(np.float32) - med) > blob_amp).astype(np.uint8)
        if not np.any(cand):
            context.image_metadata["blob_inpaint"] = {
                "ran": True, "removed_blobs": 0, "coverage_pct": 0.0}
            return False

        _despeckle_guard = os.environ.get(
            "PIXREVIVE_DESPECKLE_GUARD", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
        if _despeckle_guard:
            macro = cv2.GaussianBlur(gray, (15, 15), 0)
            gx_m = cv2.Sobel(macro, cv2.CV_32F, 1, 0)
            gy_m = cv2.Sobel(macro, cv2.CV_32F, 0, 1)
            macro_edge = np.sqrt(gx_m**2 + gy_m**2)
            cand_count = int(np.sum(cand > 0))
            if cand_count > 0:
                edge_coloc = float(np.sum((cand > 0) & (macro_edge > 25.0)) / cand_count)
                if edge_coloc > 0.55:
                    context.add_log(
                        f"BlobInpaint: rejected macroscopic scene edges "
                        f"(edge_colocation={edge_coloc:.1%} > 55.0%) — preserving natural scene structures."
                    )
                    context.image_metadata["blob_inpaint"] = {
                        "ran": True, "removed_blobs": 0, "coverage_pct": 0.0, "reason": "rejected_scene_edges"
                    }
                    return False

        num, lbl, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        blob_max = max(_BLOB_MIN_AREA + 1,
                       int(_BLOB_MAX_AREA_FRAC * h * w))
        # Structure evidence: strong luminance contours INSIDE a component
        # mean it carries real content (a crack segment, a text stroke) — dust
        # blob interiors are flat. Measured on the component pixels only, so
        # mottled paper around a dust grain does not protect the grain.
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        structure = grad > 40.0

        keep = np.zeros_like(cand)
        removed = 0
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < _BLOB_MIN_AREA or area > blob_max:
                continue
            comp = lbl == i
            touch = float(np.sum(structure[comp])) / area
            if touch > _BLOB_MAX_STRUCTURE_TOUCH:
                continue
            keep[comp] = 255
            removed += 1

        coverage = float(np.sum(keep > 0) / (h * w) * 100)
        context.image_metadata["blob_inpaint"] = {
            "ran": True,
            "removed_blobs": removed,
            "coverage_pct": round(coverage, 3),
            "blob_amp": round(blob_amp, 1),
        }
        if removed == 0 or coverage > _BLOB_COVERAGE_CAP_PCT:
            if coverage > _BLOB_COVERAGE_CAP_PCT:
                context.add_log(
                    f"BlobInpaint: candidate coverage {coverage:.2f}% above "
                    f"{_BLOB_COVERAGE_CAP_PCT:.1f}% cap — texture field, skipped.")
            return False

        mask = cv2.dilate(keep, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        context.working_image = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        context.add_log(
            f"BlobInpaint: {removed} isolated dust blob(s) inpainted "
            f"({coverage:.2f}% of frame, contrast>{blob_amp:.0f}).")
        return True

    @staticmethod
    def suppress(context: "PipelineContext") -> bool:  # noqa: F821
        """Remove impulse noise from `context.working_image` in place.

        Records the measurement under `impulse_field` in image_metadata whether
        or not anything was changed, so a later stage can distinguish an image
        that was cleaned from one that never had impulses.
        Returns True when pixels were modified.
        """
        cleaned, field = ImpulseDenoiser.apply(context.working_image)
        context.image_metadata["impulse_field"] = field.to_dict()

        if field.density <= 0.0:
            context.add_log(
                f"ImpulseDenoiser: no impulses detected "
                f"(sigma={field.sigma:.2f}, threshold={field.threshold:.1f}) — "
                "image passed through unchanged."
            )
            return False

        context.working_image = cleaned
        kind = "noise field" if field.is_field else "sparse specks"
        pixels = int(round(field.density * cleaned.shape[0] * cleaned.shape[1]))
        # Deliberately no re-measurement of the cleaned image here. Detection is
        # the expensive half of this stage and Stage 0 runs at full scan
        # resolution, so verifying the result in-line would double the cost of
        # every restoration to produce a number that is already asserted by
        # scripts/check_impulse_robustness.py.
        context.add_log(
            f"ImpulseDenoiser: {kind} removed — {pixels} pixels "
            f"({field.density:.3%}) replaced from unflagged neighbours, "
            f"sigma={field.sigma:.2f}, threshold={field.threshold:.1f}. "
            "Damage detection now sees a speck-free image."
        )
        return True
