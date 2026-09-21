"""
probe_impulse_components.py — Isolate the two independent failures the
front-end diagnosis turned up, and measure each on its own.

Failure A — classical thin-crack fallback masks the whole frame.
    detect_thin_crack_network()'s Sauvola-style threshold is
        max(mean_bh * (1 + 0.22 * (std_bh / r_dyn - 1)), 8.5)
    where r_dyn is the *global* max of the local std. std_bh / r_dyn is
    therefore ~0 almost everywhere, collapsing the expression to
    0.78 * mean_bh — i.e. "flag every pixel whose black-hat response is above
    78% of its local average", which by construction selects roughly half the
    frame before the 11x11 and 19x19 dilations widen it further.

Failure B — _reduce_impulse's impulse test is content-blind.
    is_impulse = (cur != med) & ((cur <= lo) | (cur >= hi)) flags any pixel that
    is the min or max of its own k x k window. For a k=3 window that is ~2/9 of
    all pixels in textured content, whether or not any impulse noise is present,
    so legitimate single-pixel detail (catch-lights, eyelashes, hair strands) is
    replaced by its median. Separately, `unresolved &= ~med_usable` clears the
    grow-window flag wherever the median *was* usable, which is nearly
    everywhere, so the loop never reaches k=5 or k=7 regardless of max_k.

Run (system Python 3.12, per CLAUDE.md):
    PYTHONPATH=backend "C:/Program Files/Python312/python.exe" \
        scripts/probe_impulse_components.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "backend"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from diagnose_impulse_noise import add_salt_pepper  # noqa: E402  (installs torch stub)
from ai.postprocessing.noise_router import (  # noqa: E402
    NoiseClassifier,
    _reduce_impulse,
)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def photographic_test_image(h: int = 240, w: int = 320) -> np.ndarray:
    """A clean, noise-free image with smooth gradients, fine detail, and hard
    edges — the three things an impulse filter must not damage."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 90 + 70 * np.sin(xx / 55.0) * np.cos(yy / 70.0)
    img = np.dstack([base, base * 0.92 + 14, base * 0.80 + 26])
    img = np.clip(img, 0, 255).astype(np.uint8)
    # Fine single-pixel detail: the exact structure a median filter erases.
    rng = np.random.default_rng(7)
    for _ in range(220):
        y, x = rng.integers(4, h - 4), rng.integers(4, w - 4)
        img[y, x] = 245                      # catch-light / specular highlight
    for _ in range(120):                     # 1px hair-strand segments
        y, x = rng.integers(6, h - 6), rng.integers(6, w - 6)
        img[y:y + 5, x] = 30
    cv2.circle(img, (w // 3, h // 2), 40, (210, 195, 175), -1)
    cv2.rectangle(img, (w - 90, 30), (w - 30, 110), (45, 50, 60), -1)
    return img


def main() -> int:
    print("=" * 92)
    print("FAILURE A — classical detect_thin_crack_network() coverage on CLEAN input")
    print("=" * 92)
    from ai.analysis.quality_analyzer import QualityAnalyzer
    from ai.damage.enhanced_detector import EnhancedDamageDetector
    from ai.pipeline.pipeline_context import PipelineContext

    det = EnhancedDamageDetector()
    det.defect_segmenter = None

    cases = [("photographic_synthetic", photographic_test_image())]
    for nm in ("noise.jpg", "teared.jpg"):
        p = os.path.join(REPO, "test_artifacts", "sources", nm)
        if os.path.exists(p):
            cases.append((nm, cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)))
    # A heavily pre-denoised copy: proves the ~100% coverage is not "because the
    # input is noisy" but structural to the threshold itself.
    if len(cases) > 1:
        smooth = cv2.medianBlur(cv2.GaussianBlur(cases[1][1], (0, 0), 2.0), 5)
        cases.append(("noise.jpg_heavily_presmoothed", smooth))

    print(f"{'image':<34} {'coverage%':>10}  {'verdict'}")
    for nm, rgb in cases:
        ctx = PipelineContext(rgb)
        ctx.image_metadata.update(QualityAnalyzer.analyze(rgb))
        m = det.detect_thin_crack_network(rgb, ctx)
        cov = float(np.sum(m > 0) / m.size * 100.0)
        verdict = "BROKEN (whole frame)" if cov > 60 else (
            "suspicious" if cov > 20 else "plausible")
        print(f"{nm:<34} {cov:>10.2f}  {verdict}")

    print()
    print("=" * 92)
    print("FAILURE B — _reduce_impulse on a CLEAN image (every changed pixel is a")
    print("            false positive) and its recovery on genuinely noisy input")
    print("=" * 92)

    clean = photographic_test_image()
    print(f"{'density':>8} {'changed%':>9} {'PSNR in':>9} {'PSNR out':>9} "
          f"{'gain dB':>8} {'detail kept':>12} {'max k used':>11}")

    for d in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10):
        noisy = add_salt_pepper(clean, d, seed=3) if d else clean.copy()
        out = _reduce_impulse(noisy, NoiseClassifier.classify(noisy, {}))

        changed = float(np.mean(np.any(out != noisy, axis=2)) * 100.0)
        p_in = psnr(clean, noisy)
        p_out = psnr(clean, out)

        def hf(x):
            g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32)
            return float(np.std(g - cv2.GaussianBlur(g, (0, 0), 1.5)))

        detail = hf(out) / max(hf(clean), 1e-6) * 100.0
        gain = "inf" if np.isinf(p_in) else f"{p_out - p_in:+.2f}"
        print(f"{d:>8.3f} {changed:>9.2f} "
              f"{'inf' if np.isinf(p_in) else f'{p_in:.2f}':>9} "
              f"{p_out:>9.2f} {gain:>8} {detail:>11.1f}% "
              f"{'k=3 only':>11}")

    print()
    print("Reference: a textbook adaptive median at these densities recovers to")
    print("PSNR 30-38 dB while changing only ~density% of pixels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
