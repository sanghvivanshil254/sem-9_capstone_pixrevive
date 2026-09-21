"""Calibrate the multiplicative-speckle slope statistic across available images.

Slope = relative growth of robust residual magnitude from darkest to brightest
local-mean quintile. Multiplicative noise => noise scales with brightness =>
large slope. Additive noise / clean images => flat.

Run:  PYTHONPATH=backend "C:/Program Files/Python312/python.exe" .freebuff/probe_slope_calibration.py
"""
import sys, os, glob

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import cv2
import numpy as np


def multiplicative_slope(gray: np.ndarray):
    gray_f = gray.astype(np.float32)
    mean_blur = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=2.0)
    residual = gray_f - cv2.GaussianBlur(gray_f, (0, 0), sigmaX=1.2)
    residual_abs = np.abs(residual)

    edges = np.percentile(mean_blur, [20, 40, 60, 80])
    bins = np.digitize(mean_blur.ravel(), edges)
    ravel_res = residual_abs.ravel()

    scales = []
    for b in range(5):
        sel = ravel_res[bins == b]
        scales.append(float(np.median(sel)) if sel.size >= 500 else np.nan)
    scales = np.array(scales, dtype=np.float64)
    if np.isnan(scales).any() or scales[0] <= 1e-3:
        return 0.0, scales
    growth = (scales[-1] - scales[0]) / max(scales[0], 1e-3)
    return float(growth), scales


def report(path, label):
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"{label:38s} !! unreadable")
        return
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    slope, scales = multiplicative_slope(gray)
    sc = " ".join(f"{s:5.2f}" for s in scales)
    print(f"{label:38s} slope={slope:6.3f}   per-quintile residual medians: [{sc}]")


# Synthetic controls
rng = np.random.default_rng(42)
base = cv2.imread("tests/restoration_dataset/damaged/case1_noisy_lena.jpg")
base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)

clean = np.full_like(base_gray, 128.0)
add_gauss = np.clip(base_gray + rng.normal(0, 15, base_gray.shape), 0, 255).astype(np.uint8)
speck = np.clip(base_gray * (1.0 + rng.normal(0, 0.35, base_gray.shape)), 0, 255).astype(np.uint8)
tmp = ".freebuff/_synth"
os.makedirs(tmp, exist_ok=True)
cv2.imwrite(f"{tmp}/clean_flat.png", clean.astype(np.uint8))
cv2.imwrite(f"{tmp}/additive_gauss.png", add_gauss)
cv2.imwrite(f"{tmp}/multiplicative_speck.png", speck)
heavy_speck = np.clip(base_gray * (1.0 + rng.normal(0, 0.6, base_gray.shape)), 0, 255).astype(np.uint8)
cv2.imwrite(f"{tmp}/heavy_speck.png", heavy_speck)

print("=== SYNTHETIC CONTROLS ===")
report(f"{tmp}/clean_flat.png", "flat gray (zero noise)")
report(f"{tmp}/additive_gauss.png", "case1 + ADDITIVE gaussian")
report(f"{tmp}/multiplicative_speck.png", "case1 * MULTIPLICATIVE speck s=0.35")
report(f"{tmp}/heavy_speck.png", "case1 * heavy speck s=0.60")

print("\n=== REAL FILES ===")
report(r"C:\Users\sangh\Downloads\SigShrink-denoising-of-the-Lena-image-corrupted-by-speckle-noise-The-SWT-with-four.png", "SigShrink SPECKLE lena (target)")
for p in sorted(glob.glob("tests/restoration_dataset/damaged/*")):
    report(p, "fixture: " + os.path.basename(p)[:30])
for p in [r"C:\Users\sangh\Downloads\teared.jpg", r"C:\Users\sangh\Downloads\image1.jpeg", r"C:\Users\sangh\Downloads\image3.jpg.jpeg"]:
    report(p, "user: " + os.path.basename(p))
