"""
geometry_action.py — Scanner Margin Detection and Cropping Action.
"""

from __future__ import annotations

from typing import Optional, Tuple, List
from ai.orchestrator.actions.base_action import BaseAction
from ai.orchestrator.synthesis_manifest import ActionTier
from ai.orchestrator.perception.taxonomy import DefectCandidate
from ai.pipeline.pipeline_context import PipelineContext
from ai.preprocessing.image_normalizer import ImageNormalizer


class DeskewCropAction(BaseAction):
    """Detects and crops solid scanner canvas borders and framing margins."""

    name: str = "DeskewCrop"
    cost_points: int = 1
    tier: ActionTier = ActionTier.RECOVERED
    preconditions: List[str] = []

    def execute(
        self, context: PipelineContext, candidate: Optional[DefectCandidate] = None
    ) -> Tuple[bool, float, str]:
        img = context.working_image
        cropped, has_border, crop_bbox = ImageNormalizer.detect_and_crop_borders(img)

        if has_border and crop_bbox is not None:
            context.working_image = cropped
            context.image_metadata["border_cropped"] = True
            context.image_metadata["border_bbox"] = crop_bbox
            context.sync_geometry_to_working_image()
            return True, 1.0, f"Cropped scanner canvas border to bbox {crop_bbox}"
        else:
            return True, 0.9, "No solid scanner margins detected; canvas intact"
