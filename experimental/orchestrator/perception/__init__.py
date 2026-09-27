"""
Perception package for PixRevive Orchestrator.
Includes closed taxonomy, Engine A (FeatureBank), Engine B (DefectProposer), and SignalCorroborator.
"""

from ai.orchestrator.perception.taxonomy import DefectCategory, DefectCandidate
from ai.orchestrator.perception.feature_bank import FeatureBank
from ai.orchestrator.perception.defect_proposer import DefectProposer
from ai.orchestrator.perception.corroborator import SignalCorroborator

__all__ = [
    "DefectCategory",
    "DefectCandidate",
    "FeatureBank",
    "DefectProposer",
    "SignalCorroborator",
]
