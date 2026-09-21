"""
Colorization module for PixRevive.

DDColor (ICCV 2023) is the sole colorizer — it replaced the retired ECCV16/Caffe
model.

Components:
- ddcolor_colorizer.py: DDColor dual-decoder transformer colorization
- ddcolor/:             vendored upstream inference sources (Apache-2.0)
- color_refiner.py:     Color artifact reduction and refinement

Sources and weights both live in the project workspace — never ~/.cache, and
never ModelScope. Both are produced by a one-time installer:

    python backend/ai/install_ddcolor.py

    backend/ai/colorization/ddcolor/       vendored inference code
    backend/ai/weights/ddcolor/*.pth       checkpoint (~870 MB)

If either is missing the colorizer logs the installer command and reports
is_loaded=False; the rest of the pipeline runs normally without colour.
"""

from .ddcolor_colorizer import DDColorColorizer
from .color_refiner import (
    ColorArtifactReducer,
    BandingReducer,
    SaturationBalancer,
    ColorSmoother,
    FacialSkinToneProtector,
    BackgroundNeutralizer,
)
