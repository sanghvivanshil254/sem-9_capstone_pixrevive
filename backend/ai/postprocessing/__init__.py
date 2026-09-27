"""
Postprocessing module for PixRevive.

This module contains image refinement components:
- enhancer.py: Harmonizer — guided-filter seam blending along inpainting rims
- noise_router.py: NoiseClassifier / NoiseRouter — detection + routed reduction

(The former wavelet denoiser stack — WaveletDenoiser / AdaptiveDenoiser /
SmartDenoiser / ArtifactDetector — was removed per plan decision gate 5.4:
nothing dispatched to it; Stage 5 spatially-gated NLM via NoiseRouter covers
that space. PyWavelets remains a dependency of no module and can be dropped
from requirements when convenient.)
"""

from .enhancer import Harmonizer
from .noise_router import NoiseClassifier, NoiseProfile, NoiseRouter
