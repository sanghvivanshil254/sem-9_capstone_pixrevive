"""
inpaint_action.py — Deep Fourier Inpainting Action (LaMa Engine).
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple, List
from ai.orchestrator.actions.base_action import BaseAction
from ai.orchestrator.synthesis_manifest import ActionTier
from ai.orchestrator.perception.taxonomy import DefectCandidate
from ai.pipeline.pipeline_context import PipelineContext
from ai.restoration.inpainting_engine import InpaintingEngine


class LaMaInpaintAction(BaseAction):
    """Executes Fourier-domain deep inpainting on verified damage masks."""

    name: str = "LaMaInpaint"
    cost_points: int = 3
    tier: ActionTier = ActionTier.FABRICATED
    preconditions: List[str] = []

    def __init__(self, engine: Optional[InpaintingEngine] = None):
        self.engine = engine or InpaintingEngine()

    def execute(
        self, context: PipelineContext, candidate: Optional[DefectCandidate] = None
    ) -> Tuple[bool, float, str]:
        # Determine target mask
        mask = None
        if candidate and candidate.mask is not None:
            mask = candidate.mask
        elif "final" in context.damage_masks:
            mask = context.damage_masks["final"]
        elif "crack" in context.damage_masks:
            mask = context.damage_masks["crack"]

        if mask is None or cv2.countNonZero(mask) == 0:
            return True, 1.0, "No damage mask present; inpainting skipped"

        h, w = context.working_image.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Apply dilation radius from hyperparameters
        radius = context.hyperparameters.get("inpaint_dilation_radius", 3)
        if radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        else:
            dilated_mask = mask

        inpainted = self.engine.inpaint(context.working_image, dilated_mask, context=context)

        # Seam feathering: smooth 3px border along inpaint mask
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        seam_zone = cv2.morphologyEx(dilated_mask, cv2.MORPH_GRADIENT, edge_kernel)
        if cv2.countNonZero(seam_zone) > 0:
            blurred_seam = cv2.GaussianBlur(inpainted, (5, 5), 1.2)
            alpha = (seam_zone.astype(np.float32) / 255.0)[:, :, np.newaxis]
            inpainted = (inpainted.astype(np.float32) * (1.0 - 0.5 * alpha) + blurred_seam.astype(np.float32) * (0.5 * alpha)).astype(np.uint8)

        context.working_image = inpainted
        return True, 0.88, f"Inpainted {cv2.countNonZero(mask)} pixels using Fourier infill"
