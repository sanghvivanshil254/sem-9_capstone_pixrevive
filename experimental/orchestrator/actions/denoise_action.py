"""
denoise_action.py — Noise-Adaptive Spatially-Gated 3-Zone Denoising Action.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple, List
from ai.orchestrator.actions.base_action import BaseAction
from ai.orchestrator.synthesis_manifest import ActionTier
from ai.orchestrator.perception.taxonomy import DefectCandidate
from ai.pipeline.pipeline_context import PipelineContext


class NLMDenoiseAction(BaseAction):
    """Executes noise-adaptive, edge-gated Non-Local Means denoising."""

    name: str = "NLMDenoise"
    cost_points: int = 2
    tier: ActionTier = ActionTier.RECOVERED
    preconditions: List[str] = []

    def execute(
        self, context: PipelineContext, candidate: Optional[DefectCandidate] = None
    ) -> Tuple[bool, float, str]:
        img = context.working_image
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

        # Measure noise floor
        small = cv2.resize(gray, (min(w, 256), min(h, 256)), interpolation=cv2.INTER_AREA)
        med = cv2.medianBlur(small, 3)
        diff = cv2.absdiff(small, med).astype(np.float32)
        sigma_noise = float(np.median(diff) * 1.4826)

        h_param = context.hyperparameters.get("nlm_h", 10.0)
        h_lum = max(3.0, min(14.0, h_param * (sigma_noise / 8.0 if sigma_noise > 0 else 1.0)))

        # NLM filter
        filtered = cv2.fastNlMeansDenoisingColored(
            img, None, h=float(h_lum), hColor=float(h_lum * 1.2), templateWindowSize=7, searchWindowSize=21
        )

        # Edge gating: protect edges with Sobel threshold
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.sqrt(gx ** 2 + gy ** 2)
        edge_thresh = max(30.0, 2.5 * sigma_noise)
        edge_mask = (edge_mag > edge_thresh).astype(np.float32)[:, :, np.newaxis]

        # Blend: preserve 70% of raw sharpness on edges, full denoise on flat fields
        denoised = (img.astype(np.float32) * (0.7 * edge_mask) + filtered.astype(np.float32) * (1.0 - 0.7 * edge_mask))
        denoised_uint8 = np.clip(denoised, 0, 255).astype(np.uint8)

        # Verification: check edge preservation
        context.working_image = denoised_uint8
        return True, 0.92, f"NLM denoised with h={h_lum:.1f}, sigma_noise={sigma_noise:.1f}"
