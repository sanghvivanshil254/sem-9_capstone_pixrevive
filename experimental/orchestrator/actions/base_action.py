"""
base_action.py — Abstract Base Action for Modular Restoration Pipeline.

Defines the contract for all modular actions:
- Cost points (budget tracking)
- Provenance tier (Recovered vs Fabricated)
- Execution preconditions
- Verification check after execution
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

from ai.orchestrator.synthesis_manifest import ActionTier
from ai.orchestrator.perception.taxonomy import DefectCandidate

if TYPE_CHECKING:
    from ai.pipeline.pipeline_context import PipelineContext


class BaseAction(ABC):
    """Abstract base class for all restorative and generative actions."""

    name: str = "BaseAction"
    cost_points: int = 1
    tier: ActionTier = ActionTier.RECOVERED
    preconditions: List[str] = []

    def check_preconditions(self, context: PipelineContext) -> Tuple[bool, str]:
        """Verify that necessary upstream actions or states have completed."""
        committed = {
            a.action for a in getattr(context.synthesis_manifest, "actions", [])
            if a.status == "COMMITTED"
        }
        for pre in self.preconditions:
            if pre not in committed:
                return False, f"Missing precondition '{pre}'"
        return True, "Preconditions satisfied"

    @abstractmethod
    def execute(
        self, context: PipelineContext, candidate: Optional[DefectCandidate] = None
    ) -> Tuple[bool, float, str]:
        """
        Execute action on context.working_image.
        Returns:
            (passed: bool, verification_score: float, details: str)
        """
        pass
