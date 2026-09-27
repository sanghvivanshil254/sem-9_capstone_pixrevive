"""
defect_proposer.py — Engine B: Lightweight Closed-Taxonomy Candidate Classifier.

Consumes global numeric features from FeatureBank and specialized classical detectors
to propose a prioritized list of DefectCandidate objects.
Acts strictly as a Proposer (candidates must be confirmed by Corroborator).
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Any, List, Optional

from ai.orchestrator.perception.taxonomy import DefectCategory, DefectCandidate
from ai.orchestrator.perception.feature_bank import FeatureBank
from ai.damage.enhanced_detector import EnhancedDamageDetector


class DefectProposer:
    """Lightweight rule and feature-based proposer generating defect candidates."""

    def __init__(self) -> None:
        self.detector = EnhancedDamageDetector()

    def propose(self, img_rgb: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> List[DefectCandidate]:
        """Generate candidate list of proposed defects from image and feature metrics."""
        candidates: List[DefectCandidate] = []
        features = FeatureBank.extract_global_features(img_rgb)
        h, w = features["height"], features["width"]

        # 1. Check Skew / Scanner Framing Margins
        # Check border row/col standard deviation
        top_std = float(np.std(img_rgb[:12, :, :]))
        bot_std = float(np.std(img_rgb[-12:, :, :]))
        left_std = float(np.std(img_rgb[:, :12, :]))
        right_std = float(np.std(img_rgb[:, -12:, :]))
        if min(top_std, bot_std, left_std, right_std) < 8.0:
            candidates.append(
                DefectCandidate(
                    category=DefectCategory.SKEW_PERSPECTIVE,
                    confidence=0.88,
                    features={"border_min_std": min(top_std, bot_std, left_std, right_std)},
                )
            )

        # 2. Check Periodic Moiré / Halftone Patterns
        if features["fft_peak_ratio"] >= 4.5:
            candidates.append(
                DefectCandidate(
                    category=DefectCategory.MOIRE_PATTERN,
                    confidence=min(1.0, 0.5 + (features["fft_peak_ratio"] - 4.5) * 0.1),
                    features={"fft_peak_ratio": features["fft_peak_ratio"]},
                )
            )

        # 3. Check Monochrome / Color Mode
        if features["chroma_std"] < 4.0:
            candidates.append(
                DefectCandidate(
                    category=DefectCategory.UNCOLORED,
                    confidence=0.95,
                    features={"chroma_std": features["chroma_std"]},
                )
            )
        elif features["color_cast_delta"] > 8.0:
            candidates.append(
                DefectCandidate(
                    category=DefectCategory.FADED_COLOR,
                    confidence=0.80,
                    features={"color_cast_delta": features["color_cast_delta"]},
                )
            )

        # 4. Check Low Resolution
        if features["min_dim"] < 650:
            candidates.append(
                DefectCandidate(
                    category=DefectCategory.LOW_RESOLUTION,
                    confidence=0.90,
                    features={"min_dim": features["min_dim"]},
                )
            )

        # 5. Check Over-Smoothed (Detail flattened at moderate/high resolution)
        if features["min_dim"] >= 650 and features["laplacian_var"] < 45.0:
            candidates.append(
                DefectCandidate(
                    category=DefectCategory.OVER_SMOOTHED,
                    confidence=0.85,
                    features={"laplacian_var": features["laplacian_var"]},
                )
            )

        # 6. Check Ringing / Overshoot
        if features["ringing_ratio"] >= 3.8:
            candidates.append(
                DefectCandidate(
                    category=DefectCategory.RINGING_HALO,
                    confidence=0.75,
                    features={"ringing_ratio": features["ringing_ratio"]},
                )
            )

        # 7. Check Physical Fractures, Tears & Cracks via EnhancedDetector
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        try:
            damage_results = self.detector.detect_all(img_rgb)
            if damage_results:
                for dtype, mask in damage_results.items():
                    if mask is not None and cv2.countNonZero(mask) > 30:
                        # Extract bounding box of defect cluster
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            largest = max(contours, key=cv2.contourArea)
                            x, y, bw, bh = cv2.boundingRect(largest)
                            coverage_pct = (cv2.countNonZero(mask) / (h * w)) * 100.0

                            cat = (
                                DefectCategory.FLAKING_MIRRORING
                                if dtype == "white_flakes"
                                else DefectCategory.HOLE_MISSING
                                if dtype == "paper_tears"
                                else DefectCategory.TEAR_CRACK
                            )

                            candidates.append(
                                DefectCandidate(
                                    category=cat,
                                    confidence=min(0.95, 0.60 + coverage_pct * 0.1),
                                    bbox=[x, y, x + bw, y + bh],
                                    mask=mask,
                                    features={"damage_type": dtype, "coverage_pct": coverage_pct},
                                )
                            )
        except Exception:
            pass

        return candidates
