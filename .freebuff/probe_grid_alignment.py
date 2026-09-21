"""Test JPEG block-grid alignment of Stage-2 masks as the artifact discriminator.

JPEG artifacts hug 8x8 block boundaries; real creases/flakes have random
orientation. For each mask, compute the fraction of mask pixels lying on
vertical or horizontal boundary lines, maximized over grid phase (the image may
have been cropped/rescaled after compression).

Run:  PYTHONPATH=backend "C:/Program Files/Python312/python.exe" .freebuff/probe_grid_alignment.py
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
    "case8 SPECKLE-figure (FALSE damage)": "tests/restoration_dataset/damaged/case8_speckle_lena_margins.png",
    "case1 noisy lena (grain)": "tests/restoration_dataset/damaged/case1_noisy_lena.jpg",
    "case3 sepia flakes (REAL)": "tests/restoration_dataset/damaged/case3_sepia_flaked_portrait.jpg",
    "case4 diagonal tear (REAL)": "tests/restoration_dataset/damaged/case4_diagonal_tear_portrait.jpg",
    "case5 spiderweb (REAL)": "tests/restoration_dataset/damaged/case5_spiderweb_creases_portrait.jpg",
    "teared.jpg (REAL hardest)": r"C:\Users\sangh\Downloads\teared.jpg",
    "image1 spiderweb (REAL)": r"C:\Users\sangh\Downloads\image1.jpeg",
    "image3 minor creases (REAL)": r"C:\Users\sangh\Downloads\image3.jpg.jpeg",
}


def grid_alignment(mask: np.ndarray):
    """Best-phase fraction of mask pixels on 8x8 boundary lines."""
    ys, xs = np.nonzero(mask)
    if ys.size < 200:
        return 0.0, 0.0
    h, w = mask.shape
    best = 0.0
    best_phase = (0, 0)
    for dy in range(8):
        rows_on = ((ys + dy) % 8 == 0)
        for dx in range(8):
            cols_on = ((xs + dx) % 8 == 0)
            frac = float(np.count_nonzero(rows_on | cols_on)) / ys.size
            if frac > best:
                best = frac
                best_phase = (dy, dx)
    return best, best_phase


for label, path in FILES.items():
    bgr = cv2.imread(path)
    if bgr is None:
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ctx = PipelineContext(rgb)
    meta = QualityAnalyzer.analyze(rgb)
    ctx.image_metadata.update(meta)
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

    align, phase = grid_alignment(combined)
    cov = 100.0 * np.count_nonzero(combined) / combined.size
    print(f"{label:42s} cov={cov:6.2f}%  grid_alignment={align:6.3f} (phase {phase})")
