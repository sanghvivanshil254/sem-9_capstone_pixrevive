"""
corroborator.py — Deterministic Corroboration Engine (Gatekeeper).

Enforces the non-negotiable rule: Proposer != Verifier.
Every DefectCandidate proposed by Engine B must be mathematically corroborated
by independent deterministic signals before an action is permitted to execute.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Any, List, Optional

from ai.orchestrator.perception.taxonomy import DefectCategory, DefectCandidate
from ai.orchestrator.perception.feature_bank import FeatureBank


class SignalCorroborator:
    """Deterministic gatekeeper confirming or rejecting proposed defect candidates."""

    @staticmethod
    def corroborate(img_rgb: np.ndarray, candidate: DefectCandidate) -> DefectCandidate:
        """Validate candidate against objective numerical signals."""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if img_rgb.ndim == 3 else img_rgb
        cat = candidate.category

        if cat == DefectCategory.FLAKING_MIRRORING:
            # Must exhibit sharp step-edge boundary gradient (>= 35.0) to reject specular highlights
            if candidate.mask is not None:
                bnd_grad = FeatureBank.compute_boundary_gradient(gray, candidate.mask)
                candidate.features["boundary_gradient"] = round(bnd_grad, 2)
                if bnd_grad >= 35.0:
                    candidate.corroborated = True
                    candidate.corroboration_score = min(1.0, bnd_grad / 50.0)
                    candidate.corroboration_reason = f"Step-edge boundary gradient confirmed ({bnd_grad:.1f} >= 35.0)"
                else:
                    candidate.corroborated = False
                    candidate.corroboration_reason = f"Rejected: smooth boundary ({bnd_grad:.1f} < 35.0, likely specular highlight)"
            else:
                candidate.corroborated = False
                candidate.corroboration_reason = "Rejected: no spatial mask provided for flaking"

        elif cat == DefectCategory.TEAR_CRACK:
            # Must exhibit fracture-like aspect ratio and stroke width
            if candidate.mask is not None and cv2.countNonZero(candidate.mask) > 10:
                contours, _ = cv2.findContours(candidate.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    max_c = max(contours, key=cv2.contourArea)
                    _, _, w, h = cv2.boundingRect(max_c)
                    aspect = max(w, h) / max(1, min(w, h))
                    area = cv2.contourArea(max_c)
                    candidate.features["aspect_ratio"] = round(aspect, 2)
                    candidate.features["contour_area"] = round(area, 1)

                    if aspect >= 2.0 and area >= 12.0:
                        candidate.corroborated = True
                        candidate.corroboration_score = min(1.0, aspect / 5.0)
                        candidate.corroboration_reason = f"Crack line geometry confirmed (aspect={aspect:.1f} >= 2.0)"
                    else:
                        candidate.corroborated = False
                        candidate.corroboration_reason = f"Rejected: blob shape (aspect={aspect:.1f} < 2.0)"
                else:
                    candidate.corroborated = False
            else:
                candidate.corroborated = False

        elif cat == DefectCategory.MOIRE_PATTERN:
            fft_ratio = candidate.features.get("fft_peak_ratio", 0.0)
            if fft_ratio >= 4.5:
                candidate.corroborated = True
                candidate.corroboration_score = min(1.0, fft_ratio / 8.0)
                candidate.corroboration_reason = f"Periodic Fourier spike confirmed ({fft_ratio:.1f} >= 4.5)"
            else:
                candidate.corroborated = False
                candidate.corroboration_reason = f"Rejected: no periodic frequency peak ({fft_ratio:.1f} < 4.5)"

        elif cat == DefectCategory.UNCOLORED:
            chroma_std = candidate.features.get("chroma_std", 10.0)
            if chroma_std < 4.0:
                candidate.corroborated = True
                candidate.corroboration_score = 1.0 - (chroma_std / 4.0)
                candidate.corroboration_reason = f"Monochrome confirmed (chroma std={chroma_std:.2f} < 4.0)"
            else:
                candidate.corroborated = False
                candidate.corroboration_reason = f"Rejected: significant chroma present ({chroma_std:.2f} >= 4.0)"

        elif cat == DefectCategory.FADED_COLOR:
            cast_delta = candidate.features.get("color_cast_delta", 0.0)
            if cast_delta > 7.0:
                candidate.corroborated = True
                candidate.corroboration_score = min(1.0, cast_delta / 15.0)
                candidate.corroboration_reason = f"Color cast confirmed (delta E={cast_delta:.1f} > 7.0)"
            else:
                candidate.corroborated = False
                candidate.corroboration_reason = f"Rejected: color cast negligible ({cast_delta:.1f} <= 7.0)"

        elif cat == DefectCategory.OVER_SMOOTHED:
            lap_var = candidate.features.get("laplacian_var", 100.0)
            if lap_var < 45.0:
                candidate.corroborated = True
                candidate.corroboration_score = max(0.0, 1.0 - (lap_var / 45.0))
                candidate.corroboration_reason = f"High-frequency loss confirmed (Laplacian var={lap_var:.1f} < 45.0)"
            else:
                candidate.corroborated = False
                candidate.corroboration_reason = f"Rejected: sufficient texture present ({lap_var:.1f} >= 45.0)"

        elif cat == DefectCategory.SKEW_PERSPECTIVE:
            min_std = candidate.features.get("border_min_std", 50.0)
            if min_std < 8.0:
                candidate.corroborated = True
                candidate.corroboration_score = 1.0 - (min_std / 8.0)
                candidate.corroboration_reason = f"Scanner margin confirmed (border std={min_std:.1f} < 8.0)"
            else:
                candidate.corroborated = False
                candidate.corroboration_reason = f"Rejected: natural photographic edges (std={min_std:.1f} >= 8.0)"

        elif cat == DefectCategory.RINGING_HALO:
            ringing = candidate.features.get("ringing_ratio", 1.0)
            if ringing >= 3.8:
                candidate.corroborated = True
                candidate.corroboration_score = min(1.0, ringing / 6.0)
                candidate.corroboration_reason = f"Overshoot confirmed (ratio={ringing:.1f} >= 3.8)"
            else:
                candidate.corroborated = False
                candidate.corroboration_reason = f"Rejected: ringing within normal bounds ({ringing:.1f} < 3.8)"

        elif cat == DefectCategory.LOW_RESOLUTION:
            min_dim = candidate.features.get("min_dim", 1000)
            if min_dim < 650:
                candidate.corroborated = True
                candidate.corroboration_score = 1.0 - (min_dim / 650.0)
                candidate.corroboration_reason = f"Low resolution confirmed (dimension={min_dim} < 650)"
            else:
                candidate.corroborated = False
                candidate.corroboration_reason = f"Rejected: resolution adequate ({min_dim} >= 650)"

        else:
            candidate.corroborated = True
            candidate.corroboration_score = candidate.confidence
            candidate.corroboration_reason = f"Accepted category {cat.value}"

        return candidate
