"""
superres_action.py — Terminal Super-Resolution & Exhibition Polish Action (Tier 2).
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple, List
from ai.orchestrator.actions.base_action import BaseAction
from ai.orchestrator.synthesis_manifest import ActionTier
from ai.orchestrator.perception.taxonomy import DefectCandidate
from ai.pipeline.pipeline_context import PipelineContext
from ai.restoration.super_resolution import SuperResolution
from ai.postprocessing.enhancer import Harmonizer


class RealESRGANSuperResAction(BaseAction):
    """Executes Real-ESRGAN x4plus upscaling and terminal projector exhibition grading."""

    name: str = "RealESRGANSuperRes"
    cost_points: int = 5
    tier: ActionTier = ActionTier.FABRICATED
    preconditions: List[str] = []

    def __init__(self, sr_engine: Optional[SuperResolution] = None):
        self.sr_engine = sr_engine or SuperResolution()
        self.harmonizer = Harmonizer()

    def execute(
        self, context: PipelineContext, candidate: Optional[DefectCandidate] = None
    ) -> Tuple[bool, float, str]:
        img = context.working_image
        h, w = img.shape[:2]

        upscaled = self.sr_engine.upscale(img, context=context)

        # Synchronize context geometry to 4x upscaled working image
        context.working_image = upscaled
        context.sync_geometry_to_working_image()

        # Terminal exhibition grade (CLAHE + shadow lifting)
        lab = cv2.cvtColor(upscaled, cv2.COLOR_RGB2LAB)
        l_chan = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(l_chan)
        graded = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        context.working_image = graded
        uh, uw = graded.shape[:2]
        return True, 0.95, f"Completed terminal super-resolution: {w}x{h} -> {uw}x{uh}"
