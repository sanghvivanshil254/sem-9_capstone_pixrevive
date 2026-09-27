"""
transaction.py — Atomic Transaction State Machine for Canvas Restoration.

Guarantees transactional integrity for all image alterations:
- Checkpoints working_image and damage_masks before any action executes.
- If verification fails or an exception occurs, automatically rolls back pixels
  to the exact byte-level pre-action state.
- Automatically adjusts step-down parameters for progressive remediation.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import numpy as np

from ai.orchestrator.synthesis_manifest import ActionStatus, ActionTier

if TYPE_CHECKING:
    from ai.pipeline.pipeline_context import PipelineContext

logger = logging.getLogger(__name__)


class CanvasTransaction:
    """Context manager guaranteeing atomic rollback upon verification rejection."""

    def __init__(
        self,
        context: PipelineContext,
        action_name: str,
        auto_rollback_on_exception: bool = True,
    ) -> None:
        self.context: PipelineContext = context
        self.action_name: str = action_name
        self.auto_rollback_on_exception: bool = auto_rollback_on_exception

        self.backup_image: Optional[np.ndarray] = None
        self.backup_masks: Dict[str, np.ndarray] = {}
        self.is_committed: bool = False
        self.is_rolled_back: bool = False

    def __enter__(self) -> CanvasTransaction:
        # Snapshot pixel buffer and all active damage masks
        self.backup_image = self.context.working_image.copy()
        self.backup_masks = {
            k: v.copy() for k, v in self.context.damage_masks.items() if v is not None
        }
        return self

    def commit(
        self,
        tier: ActionTier | str = ActionTier.RECOVERED,
        bbox: Optional[List[int]] = None,
        confidence: float = 1.0,
        verification_score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Confirm that the action passed verification and commit pixel changes."""
        if self.is_rolled_back:
            raise RuntimeError(f"Cannot commit rolled-back transaction '{self.action_name}'")

        self.is_committed = True
        if hasattr(self.context, "synthesis_manifest") and self.context.synthesis_manifest:
            self.context.synthesis_manifest.log_action(
                action=self.action_name,
                status=ActionStatus.COMMITTED,
                tier=tier,
                bbox=bbox,
                confidence=confidence,
                verification_score=verification_score,
                metadata=metadata or {},
            )
        self.context.add_log(f"[TRANSACTION COMMIT] {self.action_name} (Score: {verification_score:.2f})")

    def rollback(
        self,
        reason: str,
        step_down_param: Optional[str] = None,
        tier: ActionTier | str = ActionTier.RECOVERED,
        bbox: Optional[List[int]] = None,
    ) -> None:
        """Revert working_image and masks to their exact pre-action state."""
        if self.is_committed:
            raise RuntimeError(f"Cannot rollback already committed transaction '{self.action_name}'")

        if self.backup_image is not None:
            self.context.working_image = self.backup_image.copy()

        self.context.damage_masks = {
            k: v.copy() for k, v in self.backup_masks.items()
        }
        self.is_rolled_back = True

        if hasattr(self.context, "synthesis_manifest") and self.context.synthesis_manifest:
            self.context.synthesis_manifest.log_action(
                action=self.action_name,
                status=ActionStatus.ROLLED_BACK,
                tier=tier,
                bbox=bbox,
                reason=reason,
            )

        if step_down_param and hasattr(self.context, "step_down_parameter"):
            self.context.step_down_parameter(self.action_name, step_down_param)

        self.context.add_log(f"[TRANSACTION ROLLBACK] {self.action_name} — Reason: {reason}")
        logger.warning("Rolled back action '%s': %s", self.action_name, reason)

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None and self.auto_rollback_on_exception and not self.is_committed:
            self.rollback(reason=f"Exception encountered: {exc_val}")
            # Do not suppress exception so callers/orchestrators can handle or record it
            return False

        # If user exited context without explicit commit or rollback, roll back for safety
        if not self.is_committed and not self.is_rolled_back:
            self.rollback(reason="Transaction exited without explicit commit")

        return False
