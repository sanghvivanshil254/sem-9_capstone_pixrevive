"""Probe the 4 user-supplied damage cases with PixRevive's own analyzers.

Reports what Stage 1 (QualityAnalyzer), NoiseRouter classification, and the
classical Stage 2 detectors (no U-Net weights required) would conclude for each.
Run:  PYTHONPATH=backend "C:/Program Files/Python312/python.exe" .freebuff/probe_user_cases.py
"""
import sys, os

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import cv2
import numpy as np

from ai.analysis.quality_analyzer import QualityAnalyzer
from ai.pipeline.pipeline_context import PipelineContext
from ai.postprocessing.noise_router import NoiseClassifier
from ai.damage.enhanced_detector import EnhancedDamageDetector
from ai.damage.tearmark_detector import TearmarkDetector
from ai.damage.mask_refiner import MaskRefiner

CASES = {
    "teared.jpg (HARDEST)": r"C:\Users\sangh\Downloads\teared.jpg",
    "SigShrink lena speckle (HARD)": r"C:\Users\sangh\Downloads\SigShrink-denoising-of-the-Lena-image-corrupted-by-speckle-noise-The-SWT-with-four.png",
    "image1.jpeg spiderweb creases (MEDIUM)": r"C:\Users\sangh\Downloads\image1.jpeg",
    "image3.jpg.jpeg minor creases (EASY)": r"C:\Users\sangh\Downloads\image3.jpg.jpeg",
}


def cov(mask):
    if mask is None:
        return 0.0
    return 100.0 * np.count_nonzero(mask) / mask.size


for label, path in CASES.items():
    print("=" * 78)
    print(label, "|", os.path.basename(path))
    bgr = cv2.imread(path)
    if bgr is None:
        print("  !! could not read file")
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"  size: {rgb.shape[1]}x{rgb.shape[0]}")

    ctx = PipelineContext(rgb)
    meta = QualityAnalyzer.analyze(rgb)
    ctx.image_metadata.update(meta)
    print(
        f"  Stage1: color_type={meta.get('color_type')} blur={meta.get('blur_score', 0):.1f} "
        f"noise={meta.get('noise_score', 0):.1f} damage_est={meta.get('damage_percentage', 0):.1f}% "
        f"flakes={meta.get('has_white_flakes')}"
    )

    profile = NoiseClassifier.classify(rgb, ctx.image_metadata)
    print(f"  NoiseRouter: types={profile.detected_types[:6]} has_noise={profile.has_noise}")

    det = EnhancedDamageDetector(use_defect_segmenter=False)
    parts = {
        "thin_cracks": det.detect_thin_crack_network(rgb, ctx),
        "white_flakes": det.detect_white_flakes(rgb, ctx),
        "wide_cracks": det.detect_wide_cracks(rgb, ctx),
        "scratches": det.detect_scratches(rgb, ctx),
        "3d_tears": det.detect_3d_paper_tears(rgb, ctx),
    }
    for name, m in parts.items():
        print(f"    {name:12s} coverage={cov(m):6.2f}%")

    if meta.get("color_type") in ("sepia", "color"):
        tm = TearmarkDetector().detect(ctx)
        print(f"    tearmark     coverage={cov(tm):6.2f}%")

    combined = np.zeros_like(parts["thin_cracks"])
    for m in parts.values():
        combined = cv2.bitwise_or(combined, m)
    refined = MaskRefiner.refine(combined, ctx)
    print(f"  FINAL refined mask coverage: {cov(refined):.2f}%  (LaMa tier decision)")
    print(f"  Face box present in metadata: {'faces' in ctx.image_metadata and len(ctx.image_metadata.get('faces', []))}")
