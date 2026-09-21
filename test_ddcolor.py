"""
test_ddcolor.py - Smoke test for the DDColor colorizer.

Verifies that:
  1. DDColorColorizer loads without error (model files must be present)
  2. colorize() returns an RGB image of the correct shape
  3. Safety guard correctly skips tiny images

Run from the project root:
    python test_ddcolor.py
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from ai.pipeline.pipeline_context import PipelineContext
from ai.colorization.ddcolor_colorizer import DDColorColorizer


def test_safety_guard():
    """Safety guard rejects 64x64 images without loading the model."""
    colorizer = DDColorColorizer()
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    ctx = PipelineContext(img, {"damage_percentage": 0.0})

    reason = DDColorColorizer.colorization_safety_reason(ctx)
    assert reason is not None and "source too small" in reason, (
        f"Expected 'source too small' in safety reason, got: {reason}"
    )
    print("[OK] Safety guard test passed.")


def test_colorize():
    """Full colorization on a 512x512 synthetic grayscale image."""
    colorizer = DDColorColorizer()
    print("Loading DDColor colorizer...")
    colorizer.load("cpu")

    if not colorizer.is_loaded:
        print(
            "[SKIP] DDColor not loaded — weights may not be downloaded yet.\n"
            "       Run: python backend/ai/install_ddcolor.py"
        )
        return

    print("Colorizer loaded on:", colorizer.device)

    # Synthetic grayscale image (512×512 satisfies the size guard)
    gray = np.full((512, 512), 128, dtype=np.uint8)
    rgb  = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    ctx = PipelineContext(rgb, {"damage_percentage": 5.0})

    print("Running colorize()...")
    out = colorizer.colorize(ctx)

    assert out.shape == rgb.shape, f"Shape mismatch: {out.shape} != {rgb.shape}"
    assert out.dtype == np.uint8,  f"Wrong dtype: {out.dtype}"

    print(f"[OK] Colorization complete. Output shape: {out.shape}")
    for log in ctx.processing_log:
        print("  LOG:", log)


if __name__ == "__main__":
    test_safety_guard()
    test_colorize()
    print("\nAll DDColor smoke tests passed.")
