# -*- coding: utf-8 -*-
"""structure.py — oriented-structure measurement shared by the G1/G2 guards.

The remediation guards (docs/REMEDIATION_PLAN_AB_G.md) must protect real
contours (collar edge, jawline, shoulder line) without protecting *noise
fields* or *damage clusters* — both of which also light up a plain Sobel
magnitude map. The discriminator is orientation coherence from the structure
tensor: a line or edge has one dominant gradient orientation in its
neighborhood (coherence near 1.0), while isotropic noise and 2D damage blobs
have energy in all directions (coherence near 0.0).

Coherence definition (standard):
    C = ((λ1 − λ2) / (λ1 + λ2))²
where λ1 ≥ λ2 ≥ 0 are the structure-tensor eigenvalues of the Gaussian-
smoothed outer product of the gradient. C ∈ [0, 1].

`structure_gate(gray)` combines magnitude and coherence: a pixel is
"structure" when the pre-smoothed gradient is strong AND locally oriented.
Regime-verified on the remediation fixtures: jaw/collar edges land in the
0.5–0.9 band; speckle noise < 0.15; LaMa tear-fill margins < 0.3.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# Magnitude floor (gradient, 8-bit scale) below which nothing counts as
# structure regardless of coherence — protects flat zones absolutely.
_MAG_FLOOR = 45.0
# Coherence floor: below this the local energy is isotropic (noise/blob).
_COHERENCE_FLOOR = 0.25


def structure_coherence(gray_f32: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Per-pixel orientation coherence C ∈ [0, 1] (see module docstring).

    The tensor is built from a MEDIAN-FILTERED copy of the input (5×5):
    genuine lines and edges survive a median with their orientation intact,
    while isolated noise samples are replaced — so iid grain collapses to
    incoherence instead of riding Sobel's internal pixel-correlation into
    spuriously oriented pockets (measured: pure σ=15 Gaussian noise previously
    cleared the coherence floor and defeated the whole gate).
    """
    med = cv2.medianBlur(np.clip(gray_f32, 0, 255).astype(np.uint8), 5).astype(np.float32)
    gx = cv2.Sobel(med, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(med, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    # Eigenvalues of [[jxx, jxy], [jxy, jyy]].
    tr = jxx + jyy
    det = jxx * jyy - jxy * jxy
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    l1 = tr / 2.0 + disc
    l2 = tr / 2.0 - disc
    denom = np.maximum(l1 + l2, 1e-6)
    coh = ((l1 - l2) / denom) ** 2
    return np.clip(coh, 0.0, 1.0)


def structure_gate(
    gray_f32: np.ndarray,
    mag_floor: float = _MAG_FLOOR,
    coh_floor: float = _COHERENCE_FLOOR,
    smooth: float = 1.0,
) -> np.ndarray:
    """Soft gate in [0, 1]: strong AND locally oriented gradient energy.

    Returns 0 for flat regions, isotropic noise and 2D damage blobs; ~1 on
    lines, edges and ridges. Blend-ready as a per-pixel protection weight.
    """
    med = cv2.medianBlur(np.clip(gray_f32, 0, 255).astype(np.uint8), 5).astype(np.float32)
    gx = cv2.Sobel(med, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(med, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    coh = structure_coherence(gray_f32)
    mag_w = np.clip((mag - mag_floor) / max(1.0, 2.0 * mag_floor), 0.0, 1.0)
    coh_w = np.clip((coh - coh_floor) / max(1e-3, 1.0 - coh_floor), 0.0, 1.0)
    gate = mag_w * coh_w
    if smooth > 0:
        gate = cv2.GaussianBlur(gate, (0, 0), smooth)
    return np.clip(gate, 0.0, 1.0)
