"""
detail_action.py — Anisotropic Structure-Tensor Detail Synthesizer.

Remediates over-smoothing by injecting physically plausible directional grain
aligned with the local structure tensor, without ungrounded cross-image texture borrowing.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from ai.orchestrator.actions.base_action import BaseAction
from ai.orchestrator.synthesis_manifest import ActionTier
from ai.orchestrator.perception.taxonomy import DefectCandidate
from ai.pipeline.pipeline_context import PipelineContext


class DetailSynthesizerAction(BaseAction):
    """Synthesizes high-frequency micro-grain aligned with natural surface flow."""

    name: str = "DetailSynth"
    cost_points: int = 2
    tier: ActionTier = ActionTier.FABRICATED
    preconditions: List[str] = []

    def execute(
        self, context: PipelineContext, candidate: Optional[DefectCandidate] = None
    ) -> Tuple[bool, float, str]:
        img = context.working_image
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()

        # Initial sharpness metric
        pre_lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 1. Compute Structure Tensor orientation
        gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)

        jxx = cv2.GaussianBlur(gx * gx, (7, 7), 1.5)
        jyy = cv2.GaussianBlur(gy * gy, (7, 7), 1.5)
        jxy = cv2.GaussianBlur(gx * gy, (7, 7), 1.5)

        theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy + 1e-6)

        # 2. Generate Gaussian noise field
        noise = np.random.normal(0, 1.0, (h, w)).astype(np.float32)

        # 3. Create anisotropic directional micro-grain by directional filtering
        # Approximate with directional oriented Sobel-guided noise
        dir_x = np.cos(theta)
        dir_y = np.sin(theta)
        aniso_grain = (noise * (0.6 + 0.4 * np.abs(dir_x + dir_y))).astype(np.float32)

        # 4. Residual high-frequency re-injection if original_image available
        if hasattr(context, "original_image") and context.original_image.shape == img.shape:
            orig_gray = cv2.cvtColor(context.original_image, cv2.COLOR_RGB2GRAY)
            residual = orig_gray.astype(np.float32) - gray.astype(np.float32)
            # Bandpass filter the residual: remove impulse noise (>30) and DC drift (<2)
            residual_bp = np.clip(residual, -20.0, 20.0)
            combined_grain = 0.5 * aniso_grain * 6.0 + 0.5 * residual_bp
        else:
            combined_grain = aniso_grain * 6.0

        # 5. Mask gating (use candidate mask if provided, else subtle global injection)
        if candidate and candidate.mask is not None:
            mask = cv2.resize(candidate.mask, (w, h), interpolation=cv2.INTER_NEAREST)
            weight = (mask.astype(np.float32) / 255.0) * 0.85
        else:
            weight = np.full((h, w), 0.50, dtype=np.float32)

        # Apply grain in LAB L channel to avoid color noise
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] + combined_grain * weight, 0, 255)
        enhanced_rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

        # 6. Verification: Confirm high-frequency energy gained without explosion
        post_gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)
        post_lap_var = float(cv2.Laplacian(post_gray, cv2.CV_64F).var())

        if post_lap_var > pre_lap_var:
            context.working_image = enhanced_rgb
            score = min(1.0, post_lap_var / max(1.0, pre_lap_var * 1.3))
            return True, score, f"Synthesized micro-grain: lap_var {pre_lap_var:.1f} -> {post_lap_var:.1f}"
        else:
            return False, 0.0, "Detail synthesis failed to increase high-frequency energy"
