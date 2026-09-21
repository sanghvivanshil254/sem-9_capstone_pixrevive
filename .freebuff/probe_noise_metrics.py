"""Print raw noise metrics for SigShrink vs controls to locate the underestimating estimator.

Run:  PYTHONPATH=backend "C:/Program Files/Python312/python.exe" .freebuff/probe_noise_metrics.py
"""
import sys, os

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import cv2
import numpy as np

from ai.analysis.quality_analyzer import QualityAnalyzer
from ai.postprocessing.noise_router import NoiseClassifier
from ai.pipeline.pipeline_context import PipelineContext

FILES = {
    "SigShrink SPECKLE lena": r"C:\Users\sangh\Downloads\SigShrink-denoising-of-the-Lena-image-corrupted-by-speckle-noise-The-SWT-with-four.png",
    "case1_noisy_lena (additive, protected)": "tests/restoration_dataset/damaged/case1_noisy_lena.jpg",
    "case5 spiderweb (clean-ish)": "tests/restoration_dataset/damaged/case5_spiderweb_creases_portrait.jpg",
    "synth heavy speck s=0.6": ".freebuff/_synth/heavy_speck.png",
}

for label, path in FILES.items():
    bgr = cv2.imread(path)
    if bgr is None:
        print(label, "unreadable")
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    meta = QualityAnalyzer.analyze(rgb)
    prof = NoiseClassifier.classify(rgb, meta)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # QualityAnalyzer-style estimator
    blur5 = cv2.GaussianBlur(gray, (5, 5), 0)
    qa_style = float(np.mean(np.abs(gray - blur5)))
    # MAD-based robust sigma (Stage 5 style)
    med3 = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    mad_sigma = 1.4826 * float(np.median(np.abs(gray - med3)))
    # high-frequency energy ratio
    lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F).var()

    print(f"=== {label}")
    print(f"  QualityAnalyzer: noise_score={meta.get('noise_score'):.2f} blur_score={meta.get('blur_score'):.1f} damage%={meta.get('damage_percentage')}")
    print(f"  qa_style mean|I-blur5| = {qa_style:.2f}")
    print(f"  MAD sigma (Stage5 style) = {mad_sigma:.2f}")
    print(f"  Laplacian var = {lap:.1f}")
    m = prof.metrics
    print(f"  NoiseRouter: residual_mad={m.get('residual_mad'):.2f} impulse_ratio={m.get('impulse_ratio'):.4f} "
          f"speckle_score={m.get('speckle_score'):.3f} blocking={m.get('blocking_score'):.3f} "
          f"ringing={m.get('ringing_score'):.3f} banding={m.get('banding_score'):.3f}")
    print(f"  family_scores: { {k: round(v, 3) for k, v in prof.family_scores.items()} }")
    print(f"  detected: {prof.detected_types}")
