"""
taxonomy.py — Closed Problem Taxonomy for PixRevive Orchestrator.

Defines the exact closed enum of supported defect categories and candidate representation.
Prevents open-ended hallucination by enforcing discrete, auditable damage states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
import numpy as np


class DefectCategory(str, Enum):
    OVER_SMOOTHED      = "over_smoothed"       # High-frequency detail flattened, waxy or plastic appearance
    TEAR_CRACK         = "tear_crack"          # Physical fracture, crack line, hairline tear, scratch
    STAIN_FOXING       = "stain_foxing"        # Chemical oxidation spots, water marks, mold blotches
    HOLE_MISSING       = "hole_missing"        # Emulsion loss, paper perforation, missing chunk
    FLAKING_MIRRORING  = "flaking_mirroring"   # Chipped emulsion, silver mirroring, metallic sheen
    GLARE_REFLECTION   = "glare_reflection"    # Flash blowout, specular hot-spot
    SKEW_PERSPECTIVE   = "skew_perspective"    # Scanner canvas margin or perspective distortion
    MOIRE_PATTERN      = "moire_pattern"       # Periodic halftone screening pattern
    LOW_RESOLUTION     = "low_resolution"      # Pixelation, undersampled native raster
    UNCOLORED          = "uncolored"           # True monochrome / grayscale photo
    FADED_COLOR        = "faded_color"         # Color cast, sepia shift, faded photographic dyes
    WAXY_OVERPROCESSED = "waxy_overprocessed"  # Uncanny face restoration mask / AI beautification
    RINGING_HALO       = "ringing_halo"        # Overshoot halos from aggressive sharpening
    NONE               = "none"                # Region is clean or within normal variance


@dataclass
class DefectCandidate:
    """Represents a proposed defect candidate identified during the Perceive pass."""
    category: DefectCategory
    confidence: float
    bbox: Optional[List[int]] = None           # [x0, y0, x1, y1]
    mask: Optional[np.ndarray] = None          # Spatial binary mask (uint8 0 or 255)
    features: Dict[str, Any] = field(default_factory=dict)
    corroborated: bool = False
    corroboration_score: float = 0.0
    corroboration_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "confidence": round(float(self.confidence), 3),
            "bbox": self.bbox,
            "has_mask": self.mask is not None,
            "corroborated": self.corroborated,
            "corroboration_score": round(float(self.corroboration_score), 3),
            "corroboration_reason": self.corroboration_reason,
            "features": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.features.items()
                if not isinstance(v, np.ndarray)
            },
        }
