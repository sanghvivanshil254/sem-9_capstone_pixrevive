"""Test mask COHERENCE as the real-vs-false damage discriminator.

coherence = area of mask inside the largest connected cluster of dilate(mask,5)
            / total mask area.
Real damage (tears, crease webs, flake fields) is spatially clumped -> HIGH.
Grain / JPEG-artifact dust is scattered -> LOW.

Run:  PYTHONPATH=backend "C:/Program Files/Python312/python.exe" .freebuff/probe_coherence.py
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
    "case8 SPECKLE-fig (FALSE)": "tests/restoration_dataset/damaged/case8_speckle_lena_margins.png",
    "case1 additive grain": "tests/restoration_dataset/damaged/case1_noisy_lena.jpg",
    "case3 sepia flakes (REAL)": "tests/restoration_dataset/damaged/case3_sepia_flaked_portrait.jpg",
    "case4 diagonal tear (REAL)": "tests/restoration_dataset/damaged/case4_diagonal_tear_portrait.jpg",
    "case5 spiderweb (REAL)": "tests/restoration_dataset/damaged/case5_spiderweb_creases_portrait.jpg",
    "teared.jpg (REAL)": r"C:\Users\sangh\Downloads\teared.jpg",
    "image1 spiderweb (REAL)": r"C:\Users\sangh\Downloads\image1.jpeg",
    "image3 minor creases (REAL)": r"C:\Users\sangh\Downloads\image3.jpg.jpeg",
}


def coherence(mask, dilate_ksize=5):
    ys_count = int(np.count_nonzero(mask))
    if ys_count < 50:
        return 0.0, 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize * 2 + 1, dilate_ksize * 2 + 1))
    dil = cv2.dilate(mask, k)
    num, lbl = cv2.connectedComponents(dil, connectivity=8)
    if num <= 1:
        return 0.0, 0
    # largest dilated cluster
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    biggest = int(np.argmax(counts))
    biggest_zone = lbl == biggest
    area_in_biggest = int(np.count_nonzero(mask[biggest_zone]))
    return area_in_biggest / ys_count, num - 1


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

    cov = 100.0 * np.count_nonzero(combined) / combined.size
    coh, nclust = coherence(combined)
    print(f"{label:28s} cov={cov:6.2f}%  coherence={coh:5.3f}  clusters={nclust}")
