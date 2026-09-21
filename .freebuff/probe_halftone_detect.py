"""Test HalftoneSuppressor as a detection pre-filter (T-9 candidate).

For each image: measure Stage-2 mask coverage raw vs after halftone notch.
Want: case7/case8 in-content false flakes drop; real cases unchanged.

Run:  PYTHONPATH=backend "C:/Program Files/Python312/python.exe" .freebuff/probe_halftone_detect.py
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
from ai.preprocessing.halftone_suppressor import HalftoneSuppressor

FILES = {
    "case7 CLEAN fig": "tests/restoration_dataset/damaged/case7_lena.jpg",
    "case8 SPECK fig": "tests/restoration_dataset/damaged/case8_speckle_lena_margins.png",
    "case3 REAL flakes": "tests/restoration_dataset/damaged/case3_sepia_flaked_portrait.jpg",
    "case4 REAL tear": "tests/restoration_dataset/damaged/case4_diagonal_tear_portrait.jpg",
    "case5 REAL web": "tests/restoration_dataset/damaged/case5_spiderweb_creases_portrait.jpg",
    "teared REAL": r"C:\Users\sangh\Downloads\teared.jpg",
    "image1 REAL": r"C:\Users\sangh\Downloads\image1.jpeg",
    "image3 REAL": r"C:\Users\sangh\Downloads\image3.jpg.jpeg",
}


def stage2_mask(rgb, ctx):
    det = EnhancedDamageDetector(use_defect_segmenter=False)
    m = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for x in (
        det.detect_thin_crack_network(rgb, ctx),
        det.detect_white_flakes(rgb, ctx),
        det.detect_wide_cracks(rgb, ctx),
        det.detect_scratches(rgb, ctx),
        det.detect_3d_paper_tears(rgb, ctx),
    ):
        m = cv2.bitwise_or(m, x)
    if ctx.image_metadata.get("color_type") in ("sepia", "color"):
        m = cv2.bitwise_or(m, TearmarkDetector().detect(ctx))
    return m


for label, path in FILES.items():
    bgr = cv2.imread(path)
    if bgr is None:
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    meta = QualityAnalyzer.analyze(rgb)

    # raw detection
    ctx = PipelineContext(rgb, dict(meta))
    raw = stage2_mask(rgb, ctx)

    # halftone-notched detection
    ctx2 = PipelineContext(rgb.copy(), dict(meta))
    acted = HalftoneSuppressor.suppress(ctx2)
    notched_canvas = ctx2.working_image
    notched = stage2_mask(notched_canvas, ctx2)

    cm = content_mask(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    in_margin_raw = 100.0 * np.count_nonzero(raw[cm == 0]) / max(1, np.count_nonzero(raw))

    cov_raw = 100.0 * np.count_nonzero(raw) / raw.size
    cov_notch = 100.0 * np.count_nonzero(notched) / notched.size
    print(f"{label:18s} ht_acted={str(acted):5s}  raw={cov_raw:6.2f}% (margin {in_margin_raw:4.1f}%)  notched={cov_notch:6.2f}%")
