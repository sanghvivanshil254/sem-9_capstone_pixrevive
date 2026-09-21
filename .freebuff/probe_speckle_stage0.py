"""Verify the speckle misdiagnosis in REAL pipeline order.

Production runs Stage 0 (ImpulseDenoiser + SpeckSuppressor) BEFORE Stage 1
analysis and Stage 2 detection. This probe replicates that order on the
SigShrink speckle-Lena image and reports what changes.

Run:  PYTHONPATH=backend "C:/Program Files/Python312/python.exe" .freebuff/probe_speckle_stage0.py
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
from ai.preprocessing.impulse_denoiser import ImpulseDenoiser
from ai.preprocessing.speck_suppressor import SpeckSuppressor

IMG = r"C:\Users\sangh\Downloads\SigShrink-denoising-of-the-Lena-image-corrupted-by-speckle-noise-The-SWT-with-four.png"


def cov(mask):
    return 100.0 * np.count_nonzero(mask) / mask.size


def measure(rgb, tag):
    ctx = PipelineContext(rgb)
    meta = QualityAnalyzer.analyze(rgb)
    ctx.image_metadata.update(meta)
    profile = NoiseClassifier.classify(rgb, ctx.image_metadata)
    det = EnhancedDamageDetector(use_defect_segmenter=False)
    m = {
        "flakes": det.detect_white_flakes(rgb, ctx),
        "wide": det.detect_wide_cracks(rgb, ctx),
        "tears3d": det.detect_3d_paper_tears(rgb, ctx),
        "thin": det.detect_thin_crack_network(rgb, ctx),
    }
    total = np.zeros_like(m["flakes"])
    for x in m.values():
        total = cv2.bitwise_or(total, x)
    print(f"[{tag}]")
    print(f"  Stage1: noise={meta.get('noise_score', 0):.2f} damage_est={meta.get('damage_percentage', 0):.1f}% flakes={meta.get('has_white_flakes')}")
    print(f"  NoiseRouter: {profile.detected_types[:6]}")
    print(f"  detectors: " + "  ".join(f"{k}={cov(v):.2f}%" for k, v in m.items()))
    print(f"  TOTAL damage mask: {cov(total):.2f}%")
    return cov(total), meta, profile


bgr = cv2.imread(IMG)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

cov_before, meta_before, prof_before = measure(rgb, "RAW (no Stage 0) — what my earlier probe measured")

# ── Real pipeline order: Stage 0 first ────────────────────────────────────────
ctx0 = PipelineContext(rgb.copy())
acted_impulse = ImpulseDenoiser.suppress(ctx0)
acted_speck = SpeckSuppressor.suppress(ctx0)
print(f"\nStage 0 on speckle image: ImpulseDenoiser acted={acted_impulse}, SpeckSuppressor acted={acted_speck}")
print(f"  speck metrics recorded: {ctx0.image_metadata.get('speck_field', {})}")
cleaned = ctx0.working_image
changed_px = int(np.sum(np.any(cleaned != rgb, axis=2)))
print(f"  pixels rewritten by Stage 0: {changed_px} ({100.0 * changed_px / rgb.shape[0] / rgb.shape[1]:.2f}% of frame)")

cov_after, meta_after, prof_after = measure(cleaned, "AFTER Stage 0 (real pipeline order)")

print("\n=== VERDICT ===")
print(f"damage mask: {cov_before:.2f}% -> {cov_after:.2f}%")
print(f"noise_score: {meta_before.get('noise_score'):.2f} -> {meta_after.get('noise_score'):.2f}")
print(f"noise types raw: {prof_before.detected_types[:6]}")
print(f"noise types post-Stage0: {prof_after.detected_types[:6]}")
