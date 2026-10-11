"""Step 5: DDColor isolation A/B — stain-inheritance hypothesis.

Runs the DDColor pipeline directly on the same teared.jpg face crop with:
  (a) STAINED input:          the crop as-is, LONLY disabled, discipline off
  (b) L-ONLY input:            a/b forced to 128 before the network
Reports mean a*/b* for the face-crop (skin) and a lower background band of
the same output. No pipeline stages; only DDColorColorizer.colorize().
"""
import cv2
import numpy as np
import os
import sys

REPO = r"D:\Desktop\project\PixRevive"
sys.path.insert(0, REPO + r"\backend")
os.chdir(REPO + r"\backend")

crop = cv2.imread(REPO + r"\test demos\teared.jpg", cv2.IMREAD_COLOR)
h, w = crop.shape[:2]
# Audit face ROI [176,357,595,699] (0-1000 normalized) — shrink slightly so the
# border band stays neutral-paper (a genuine 'background of the same photo').
face = crop[int(0.176 * h):int(0.595 * h), int(0.357 * w):int(0.699 * w)]
# lower third of the crop = below the chin/collar area (bg-ish in crop coords)
bg_band = face[int(face.shape[0] * 0.70):, :]
crop_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

from ai.colorization.ddcolor_colorizer import DDColorColorizer
from ai.pipeline.pipeline_context import PipelineContext


def run(mode: str) -> dict:
    colorizer = DDColorColorizer()
    colorizer.load("cuda")
    ctx = PipelineContext(crop_rgb.copy(), {"faces": [], "color_type": "sepia"})
    if mode == "stained":
        os.environ["PIXREVIVE_DDCOLOR_LONLY"] = "0"
    else:
        os.environ["PIXREVIVE_DDCOLOR_LONLY"] = "1"
    os.environ["PIXREVIVE_CHROMA_DISCIPLINE"] = "0"
    os.environ["PIXREVIVE_CHROMA_VALIDATOR"] = "0"
    out = colorizer.colorize(ctx)
    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    a, b = lab[:, :, 1] - 128, lab[:, :, 2] - 128
    bg_frac = 0.70
    bg_b = b[int(b.shape[0] * bg_frac):, :].mean()
    bg_a = a[int(a.shape[0] * bg_frac):, :].mean()
    skin_a, skin_b = a.mean(), b.mean()
    return {"mode": mode, "skin_a": round(float(skin_a), 2), "skin_b": round(float(skin_b), 2),
            "bg_a": round(float(bg_a), 2), "bg_b": round(float(bg_b), 2)}


if __name__ == "__main__":
    import json
    results = []
    for mode in ("stained", "lonly"):
        try:
            results.append(run(mode))
        except Exception as exc:
            results.append({"mode": mode, "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps(results, indent=2))
