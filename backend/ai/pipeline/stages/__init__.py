"""Stage mixin modules split out of restoration_pipeline.py (plan 3.5).

Ordering and metadata remain in ai/pipeline/stage_registry.py.
"""

from .preprocess_stages import PreprocessStagesMixin
from .damage_stages import DamageStagesMixin
from .color_stages import ColorStagesMixin
from .finish_stages import FinishStagesMixin

__all__ = [
    "PreprocessStagesMixin",
    "DamageStagesMixin",
    "ColorStagesMixin",
    "FinishStagesMixin",
]
