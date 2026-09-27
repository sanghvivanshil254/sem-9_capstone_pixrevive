"""
synthesis_manifest.py — Provenance and ethical authenticity manifest.

Logs every action executed by the orchestrator, its spatial bounding box,
verification score, and classification tier (Recovered vs. Fabricated).
Outputs synthesis_manifest.json alongside the restored image.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, List, Optional


class ActionTier(str, Enum):
    RECOVERED = "Recovered"                     # Pure recovery: NLM denoise, deskew, optical contrast
    FABRICATED = "Fabricated"                   # Deep generative infill: LaMa inpaint, DDColor guess
    PARTIALLY_FABRICATED = "Partially_Fabricated" # Hybrid: GFPGAN blend with original face


class ActionStatus(str, Enum):
    COMMITTED = "COMMITTED"
    ROLLED_BACK = "ROLLED_BACK"
    SKIPPED = "SKIPPED"


@dataclass
class ActionRecord:
    order: int
    action: str
    status: str
    tier: str
    bbox: Optional[List[int]] = None
    confidence: float = 1.0
    verification_score: float = 1.0
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class SynthesisManifest:
    """Provenance tracker recording all restorative and generative actions."""

    def __init__(self, version: str = "2.0", image_hash: Optional[str] = None):
        self.version: str = version
        self.image_hash: Optional[str] = image_hash
        self.actions: List[ActionRecord] = []
        self._action_counter: int = 0

    def log_action(
        self,
        action: str,
        status: ActionStatus | str,
        tier: ActionTier | str = ActionTier.RECOVERED,
        bbox: Optional[List[int]] = None,
        confidence: float = 1.0,
        verification_score: float = 1.0,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActionRecord:
        """Register an action attempt or committed operation."""
        self._action_counter += 1
        status_val = status.value if isinstance(status, ActionStatus) else str(status)
        tier_val = tier.value if isinstance(tier, ActionTier) else str(tier)

        record = ActionRecord(
            order=self._action_counter,
            action=action,
            status=status_val,
            tier=tier_val,
            bbox=bbox,
            confidence=float(confidence),
            verification_score=float(verification_score),
            reason=reason,
            metadata=metadata or {},
        )
        self.actions.append(record)
        return record

    @property
    def authenticity_score(self) -> float:
        """Calculate overall authenticity score (0.0 to 1.0).

        Ratio of recovered operations to total committed operations.
        """
        committed = [a for a in self.actions if a.status == ActionStatus.COMMITTED.value]
        if not committed:
            return 1.0

        recovered_count = sum(
            1.0 if a.tier == ActionTier.RECOVERED.value else
            0.5 if a.tier == ActionTier.PARTIALLY_FABRICATED.value else
            0.0
            for a in committed
        )
        return round(recovered_count / len(committed), 4)

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to serializable dictionary."""
        return {
            "version": self.version,
            "image_hash": self.image_hash,
            "overall_authenticity_score": self.authenticity_score,
            "total_actions_logged": len(self.actions),
            "committed_actions": [
                asdict(a) for a in self.actions if a.status == ActionStatus.COMMITTED.value
            ],
            "rolled_back_actions": [
                asdict(a) for a in self.actions if a.status == ActionStatus.ROLLED_BACK.value
            ],
            "all_actions": [asdict(a) for a in self.actions],
        }

    def to_json(self, indent: int = 2) -> str:
        """Export manifest as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str) -> None:
        """Write manifest JSON to disk."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())
