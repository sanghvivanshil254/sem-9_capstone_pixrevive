"""Find a detection pre-filter that kills JPEG-artifact false masks but keeps real damage.

For each candidate canvas transform, measure total detector-mask coverage on
case8 (false damage target: want LOW) vs the real-damage cases (want HIGH).

Run:  PYTHONPATH=backend "C:/Program Files/Python312/python.exe" .freebuff/probe_deblock_detection.py
"""
import sys, os

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import cv2
import numpy as np

from ai.analysis.quality_analyzer import QualityAnalyzer
from ai.pipeline.pipeline_context import PipelineContext
from ai.damage.enhanced_detector import EnhancedDamageDetector
from ai.damage.tearmark_detector import TearmarkDetector
from ai.postprocessing.noise_router import NoiseClassifier

FILES = {
    "case8 FALSE": "tests/restoration_dataset/damaged/case8_speckle_lena_margins.png",
    "case3 REAL": "tests/restoration_dataset/damaged/case3_sepia_flaked_portrait.jpg",
    "case4 REAL": "tests/restoration_dataset/damaged/case4_diagonal_tear_portrait.jpg",
    "case5 REAL": "tests/restoration_dataset/damaged/case5_spiderweb_creases_portrait.jpg",
    "teared REAL": r"C:\Users\sangh\Downloads\teared.jpg",
    "image1 REAL": r"C:\Users\sangh\Downloads\image1.jpeg",
    "image3 REAL": r"C:\Users\sangh\Downloads\image3.jpg.jpeg",
}

TRANSFORMS = {
    "none": lambda rgb: rgb,
    "median3": lambda rgb: cv2.medianBlur(rgb, 3),
    "bilateral(9,25)": lambda rgb: cv2.bilateralFilter(rgb, 9, 25, 5),
    "bilateral(9,45)": lambda rgb: cv2.bilateralFilter(rgb, 9, 45, 5),
}


def mask_on(rgb, ctx_meta, canvas):
    ctx = PipelineContext(canvas, dict(ctx_meta))
    det = EnhancedDamageDetector(use_defect_segmenter=False)
    m = np.zeros(canvas.shape[:2], dtype=np.uint8)
    for x in (
        det.detect_thin_crack_network(canvas, ctx),
        det.detect_white_flakes(canvas, ctx),
        det.detect_wide_cracks(canvas, ctx),
        det.detect_scratches(canvas, ctx),
        det.detect_3d_paper_tears(canvas, ctx),
    ):
        m = cv2.bitwise_or(m, x)
    if ctx_meta.get("color_type") in ("sepia", "color"):
        m = cv2.bitwise_or(m, TearmarkDetector().detect(ctx))
    return 100.0 * np.count_nonzero(m) / m.size


for label, path in FILES.items():
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    meta = QualityAnalyzer.analyze(rgb)
    prof = NoiseClassifier.classify(rgb, meta)
    comp = prof.family_scores.get("compression", 0.0)
    line = f"{label:14s} comp={comp:.2f} |"
    for tname, tf in TRANSFORMS.items():
        cov = mask_on(rgb, meta, tf(rgb))
        line += f"  {tname}={cov:6.2f}%"
    print(line)
