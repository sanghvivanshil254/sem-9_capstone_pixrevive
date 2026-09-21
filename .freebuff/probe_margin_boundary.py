"""Final discriminator probe: A/B clean-vs-speckled figure twins + margin-band attribution.

case7_lena.jpg (clean figure) and case8 (speckled figure) are the same artwork.
If detectors fire on case7 too, the mask is figure content, not noise. Also
measures how much of each mask lies within the margin band.

Run:  PYTHONPATH=backend "C:/Program Files/Python312/python.exe" .freebuff/probe_margin_boundary.py
"""
import sys, os

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import cv2
import numpy as np

from ai.analysis.quality_analyzer import QualityAnalyzer
from ai.analysis.content_region import content_mask
from ai.pipeline.pipeline_context import PipelineContext
from ai.damage.enhanced_detector import EnhancedDamageDetector
from ai.damage.tearmark_detector import TearmarkDetector

FILES = {
    "case7 CLEAN figure": "tests/restoration_dataset/damaged/case7_lena.jpg",
    "case8 SPECKLED figure": "tests/restoration_dataset/damaged/case8_speckle_lena_margins.png",
    "case3 REAL flakes": "tests/restoration_dataset/damaged/case3_sepia_flaked_portrait.jpg",
    "case5 REAL spiderweb": "tests/restoration_dataset/damaged/case5_spiderweb_creases_portrait.jpg",
    "teared REAL": r"C:\Users\sangh\Downloads\teared.jpg",
}

for label, path in FILES.items():
    bgr = cv2.imread(path)
    if bgr is None:
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ctx = PipelineContext(rgb)
    meta = QualityAnalyzer.analyze(rgb)
    ctx.image_metadata.update(meta)

    det = EnhancedDamageDetector(use_defect_segmenter=False)
    parts = {
        "flakes": det.detect_white_flakes(rgb, ctx),
        "wide": det.detect_wide_cracks(rgb, ctx),
        "3d": det.detect_3d_paper_tears(rgb, ctx),
        "thin": det.detect_thin_crack_network(rgb, ctx),
    }
    combined = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for m in parts.values():
        combined = cv2.bitwise_or(combined, m)
    if meta.get("color_type") in ("sepia", "color"):
        combined = cv2.bitwise_or(combined, TearmarkDetector().detect(ctx))

    total = 100.0 * np.count_nonzero(combined) / combined.size

    # margin band = frame-touching near-white regions (recompute cheaply)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cmask = content_mask(gray)
    margin_band = (cmask == 0)
    margin_frac = 100.0 * np.count_nonzero(combined[margin_band]) / max(1, np.count_nonzero(combined))

    per = "  ".join(f"{k}={100.0 * np.count_nonzero(v) / v.size:5.2f}" for k, v in parts.items())
    print(f"{label:24s} total={total:6.2f}%  in_margin_band={margin_frac:5.1f}%   {per}")
