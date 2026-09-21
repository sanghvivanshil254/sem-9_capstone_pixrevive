"""
face_action.py — Damage-Aware Facial Restoration Action (GFPGAN).
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple, List
from ai.orchestrator.actions.base_action import BaseAction
from ai.orchestrator.synthesis_manifest import ActionTier
from ai.orchestrator.perception.taxonomy import DefectCandidate
from ai.pipeline.pipeline_context import PipelineContext
from ai.restoration.face_restorer import FaceRestorer
from ai.analysis.face_detector import FaceDetector


class GFPGANFaceAction(BaseAction):
    """Executes damage-aware generative facial restoration with anatomical chin fade."""

    name: str = "GFPGANRestore"
    cost_points: int = 3
    tier: ActionTier = ActionTier.PARTIALLY_FABRICATED
    preconditions: List[str] = []

    def __init__(self, restorer: Optional[FaceRestorer] = None, detector: Optional[FaceDetector] = None):
        self.restorer = restorer or FaceRestorer()
        self.detector = detector

    def execute(
        self, context: PipelineContext, candidate: Optional[DefectCandidate] = None
    ) -> Tuple[bool, float, str]:
        img = context.working_image
        faces = context.image_metadata.get("faces", [])

        # If faces haven't been detected yet, detect them now
        if not faces and self.detector is not None:
            self.detector.detect(context)
            faces = context.image_metadata.get("faces", [])

        if not faces:
            return True, 1.0, "No faces detected in canvas; face restore skipped"

        blend_weight = context.hyperparameters.get("gfpgan_blend_weight", 0.85)
        damage_mask = context.damage_masks.get("final")

        restored_canvas = self.restorer.restore_faces(
            image=img,
            faces=faces,
            damage_mask=damage_mask,
            fidelity_weight=blend_weight,
            context=context,
        )

        context.working_image = restored_canvas
        return True, 0.90, f"Restored {len(faces)} face(s) with blend fidelity {blend_weight:.2f}"
