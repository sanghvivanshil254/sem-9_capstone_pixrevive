"""
colorize_action.py — Deep DDColor Colorization and Chrominance Refinement Action.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple, List
from ai.orchestrator.actions.base_action import BaseAction
from ai.orchestrator.synthesis_manifest import ActionTier
from ai.orchestrator.perception.taxonomy import DefectCandidate
from ai.pipeline.pipeline_context import PipelineContext
from ai.colorization.ddcolor_colorizer import DDColorColorizer
from ai.colorization.color_refiner import ColorArtifactReducer


class DDColorizeAction(BaseAction):
    """Executes DDColor neural colorization with edge-aware chroma bleed fixing."""

    name: str = "DDColorize"
    cost_points: int = 4
    tier: ActionTier = ActionTier.FABRICATED
    preconditions: List[str] = []

    def __init__(self, colorizer: Optional[DDColorColorizer] = None):
        self.colorizer = colorizer or DDColorColorizer()
        self.artifact_reducer = ColorArtifactReducer()

    def execute(
        self, context: PipelineContext, candidate: Optional[DefectCandidate] = None
    ) -> Tuple[bool, float, str]:
        img = context.working_image

        colorized = self.colorizer.colorize(img, context=context)

        # Apply edge bleed reduction and artifact reduction
        faces = context.image_metadata.get("faces", [])
        refined = self.artifact_reducer.reduce_artifacts(colorized, faces=faces)

        # Opacity blend
        opacity = context.hyperparameters.get("colorize_opacity", 0.85)
        if opacity < 1.0:
            final_color = cv2.addWeighted(refined, opacity, img, 1.0 - opacity, 0.0)
        else:
            final_color = refined

        context.working_image = final_color
        return True, 0.85, f"Applied DDColor colorization with opacity {opacity:.2f}"
