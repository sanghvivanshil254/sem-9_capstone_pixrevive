"""
action_registry.py — Central Registry of Modular Restoration Actions.

Maps DefectCategory enum types to executable BaseAction instances.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from ai.orchestrator.actions.base_action import BaseAction
from ai.orchestrator.actions.geometry_action import DeskewCropAction
from ai.orchestrator.actions.denoise_action import NLMDenoiseAction
from ai.orchestrator.actions.inpaint_action import LaMaInpaintAction
from ai.orchestrator.actions.detail_action import DetailSynthesizerAction
from ai.orchestrator.actions.colorize_action import DDColorizeAction
from ai.orchestrator.actions.face_action import GFPGANFaceAction
from ai.orchestrator.actions.superres_action import RealESRGANSuperResAction
from ai.orchestrator.perception.taxonomy import DefectCategory, DefectCandidate


class ActionRegistry:
    """Registry maintaining instances and category mappings for all actions."""

    def __init__(self) -> None:
        self._actions_by_name: Dict[str, BaseAction] = {}
        self._category_map: Dict[DefectCategory, str] = {}
        self._register_default_actions()

    def _register_default_actions(self) -> None:
        # Register action instances
        deskew = DeskewCropAction()
        denoise = NLMDenoiseAction()
        inpaint = LaMaInpaintAction()
        detail = DetailSynthesizerAction()
        colorize = DDColorizeAction()
        face = GFPGANFaceAction()
        sr = RealESRGANSuperResAction()

        self.register(deskew)
        self.register(denoise)
        self.register(inpaint)
        self.register(detail)
        self.register(colorize)
        self.register(face)
        self.register(sr)

        # Map defect categories to default handling actions
        self.bind_category(DefectCategory.SKEW_PERSPECTIVE, deskew.name)
        self.bind_category(DefectCategory.TEAR_CRACK, inpaint.name)
        self.bind_category(DefectCategory.FLAKING_MIRRORING, inpaint.name)
        self.bind_category(DefectCategory.HOLE_MISSING, inpaint.name)
        self.bind_category(DefectCategory.OVER_SMOOTHED, detail.name)
        self.bind_category(DefectCategory.UNCOLORED, colorize.name)
        self.bind_category(DefectCategory.FADED_COLOR, colorize.name)

    def register(self, action: BaseAction) -> None:
        self._actions_by_name[action.name] = action

    def bind_category(self, category: DefectCategory, action_name: str) -> None:
        self._category_map[category] = action_name

    def get_by_name(self, name: str) -> Optional[BaseAction]:
        return self._actions_by_name.get(name)

    def get_for_candidate(self, candidate: DefectCandidate) -> Optional[BaseAction]:
        action_name = self._category_map.get(candidate.category)
        if action_name:
            return self._actions_by_name.get(action_name)
        return None

    def list_all_actions(self) -> List[BaseAction]:
        return list(self._actions_by_name.values())
