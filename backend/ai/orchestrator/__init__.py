"""
PixRevive Orchestrator Module (v2.0)
Two-Tier Closed-Loop PPCAV (Perceive -> Propose -> Corroborate -> Act -> Verify) System.
"""

from ai.orchestrator.synthesis_manifest import SynthesisManifest, ActionTier, ActionStatus
from ai.orchestrator.transaction import CanvasTransaction
from ai.orchestrator.loop_controller import LoopController


def __getattr__(name: str):
    if name == "PhotoOrchestrator":
        from ai.orchestrator.orchestrator import PhotoOrchestrator
        return PhotoOrchestrator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "PhotoOrchestrator",
    "SynthesisManifest",
    "ActionTier",
    "ActionStatus",
    "CanvasTransaction",
    "LoopController",
]
