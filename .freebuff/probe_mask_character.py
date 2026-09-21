"""Measure Stage-2 mask component character across real vs false damage images.

Hypothesis: real creases/flakes form few large elongated components; grain/JPEG
false positives form many small round ones. Measure before designing the gate.

Run:  PYTHONPATH=backend "C:/Program Files/Python312/python.exe" .freebuff/probe_mask_character.py
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

FILES = {
    "case8 SPECKLE-figure (false damage)": "tests/restoration_dataset/damaged/case8_speckle_lena_margins.png",
    "case1 noisy lena (additive, grain)": "tests/restoration_dataset/damaged/case1_noisy_lena.jpg",
    "case3 sepia flakes (REAL)": "tests/restoration_dataset/damaged/case3_sepia_flaked_portrait.jpg",
    "case4 diagonal tear (REAL)": "tests/restoration_dataset/damaged/case4_diagonal_tear_portrait.jpg",
    "case5 spiderweb (REAL)": "tests/restoration_dataset/damaged/case5_spiderweb_creases_portrait.jpg",
    "teared.jpg (REAL, hardest)": r"C:\Users\sangh\Downloads\teared.jpg",
    "image1 spiderweb (REAL)": r"C:\Users\sangh\Downloads\image1.jpeg",
    "image3 minor creases (REAL)": r"C:\Users\sangh\Downloads\image3.jpg.jpeg",
}

for label, path in FILES.items():
    bgr = cv2.imread(path)
    if bgr is None:
        print(label, "unreadable")
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ctx = PipelineContext(rgb)
    meta = QualityAnalyzer.analyze(rgb)
    ctx.image_metadata.update(meta)
    try:
        from ai.analysis.face_detector import FaceDetector
        FaceDetector().detect(ctx)
    except Exception:
        ctx.image_metadata["faces"] = []

    det = EnhancedDamageDetector(use_defect_segmenter=False)
    combined = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for m in (
        det.detect_thin_crack_network(rgb, ctx),
        det.detect_white_flakes(rgb, ctx),
        det.detect_wide_cracks(rgb, ctx),
        det.detect_scratches(rgb, ctx),
        det.detect_3d_paper_tears(rgb, ctx),
    ):
        combined = cv2.bitwise_or(combined, m)
    if meta.get("color_type") in ("sepia", "color"):
        combined = cv2.bitwise_or(combined, TearmarkDetector().detect(ctx))

    cov = 100.0 * np.count_nonzero(combined) / combined.size
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    ws = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    hs = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    if areas.size:
        elong = np.maximum(ws, hs) / np.maximum(1.0, np.minimum(ws, hs))
        small = areas < 60
        frac_small_area = 100.0 * areas[small].sum() / areas.sum()
        print(f"=== {label}")
        print(f"  coverage={cov:5.2f}%  components={num-1:5d}  "
              f"median_area={np.median(areas):6.1f}px  p90_area={np.percentile(areas, 90):7.1f}px")
        print(f"  median_elongation={np.median(elong):5.2f}  "
              f"area_in_tiny_components(<60px)={frac_small_area:5.1f}%  "
              f"count_tiny={int(small.sum())}")
        print(f"  sigma/content: noise={meta.get('noise_score'):.2f} sigma={meta.get('content_sigma'):.2f} "
              f"grain_field={meta.get('grain_field')} damage%={meta.get('damage_percentage'):.2f}")
    else:
        print(f"=== {label}: empty mask (coverage {cov:.2f}%)")
