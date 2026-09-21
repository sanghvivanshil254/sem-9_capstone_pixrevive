"""Probe case7/case8: does content_mask actually exclude the real scanner margin?

Checks: margin histogram mass, detected margin value, content mask fraction,
content vs global sigma, and compression flags with/without masking.
Run: PYTHONPATH=backend "C:/Program Files/Python312/python.exe" .freebuff/probe_case78_content.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import cv2
import numpy as np

from ai.analysis.content_region import _dominant_margin_value, content_mask, robust_sigma
from ai.analysis.quality_analyzer import QualityAnalyzer

BASE = os.path.join(os.path.dirname(__file__), "..", "tests", "restoration_dataset", "damaged")
CASES = {
    "case7_lena": "case7_lena.jpg",
    "case8_speckle_lena": "case8_speckle_lena_margins.png",
}

for name, fname in CASES.items():
    path = os.path.join(BASE, fname)
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = float(hist.sum())
    print(f"\n=== {name} ({w}x{h}) ===")
    print(f"  >=244 mass: {hist[244:].sum()/total:.3f}   >=200 mass: {hist[200:].sum()/total:.3f}   argmax bin: {int(np.argmax(hist))}")
    mv = _dominant_margin_value(gray)
    print(f"  _dominant_margin_value: {mv}")

    m = content_mask(rgb)
    frac = float(np.mean(m > 0))
    print(f"  content_mask: all-ones={bool(np.all(m == 1))}, content frac={frac:.3f}")

    sig_content = robust_sigma(gray, m)
    sig_global = robust_sigma(gray, np.ones_like(m))
    print(f"  sigma: content={sig_content:.2f}  global={sig_global:.2f}")

    meta = QualityAnalyzer.analyze(rgb)
    print(f"  QualityAnalyzer: noise_score={meta['noise_score']:.2f} content_sigma={meta.get('content_sigma', float('nan')):.2f} "
          f"grain_field={meta.get('grain_field')} conf={meta.get('grain_field_confidence', 0):.3f} damage%={meta['damage_percentage']:.2f}")
