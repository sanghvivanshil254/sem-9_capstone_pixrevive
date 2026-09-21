"""
probe_impulse_denoise.py — Compare impulse-noise removal strategies for small,
densely speckled sources.

The shipped `_reduce_impulse` picks a 5x5 median once the impulse score passes
0.55 and then dilates the replacement mask, which on a 196x148 photo is close to
a blanket 5x5 median: it removed 58.7% of the image's edges and erased the eyes.

Metrics:
  residual  — std of (gray - gaussian(gray, 1.5)); noise drives this up, so
              lower is better, but zero means the structure went too.
  edge_keep — fraction of the ORIGINAL Canny edges that survive. Impulse noise
              creates spurious edges, so 100% is not the goal; what matters is
              keeping the edges that sit on real features.
  face_det  — detail energy inside the detected face box: the number that
              collapsed to nothing in the shipped path.

Usage:
    PYTHONPATH=backend "C:/Program Files/Python312/python.exe" \
        scripts/probe_impulse_denoise.py test_artifacts/sources/noise.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# ── candidate filters ─────────────────────────────────────────────────────────
def shipped(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Current behaviour: median, threshold, DILATE the mask, replace."""
    median = cv2.medianBlur(img, ksize)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    med_gray = cv2.cvtColor(median, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mask = np.abs(gray - med_gray) > (38.0 if ksize == 3 else 30.0)
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    out = img.copy()
    out[mask] = median[mask]
    return out


def no_dilate(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Same, but replace only the pixels actually flagged as outliers."""
    median = cv2.medianBlur(img, ksize)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    med_gray = cv2.cvtColor(median, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mask = np.abs(gray - med_gray) > (38.0 if ksize == 3 else 30.0)
    out = img.copy()
    out[mask] = median[mask]
    return out


def iterated_k3(img: np.ndarray, rounds: int = 3) -> np.ndarray:
    """Repeated 3x3 outlier replacement — small footprint, applied until clean."""
    out = img.copy()
    for _ in range(rounds):
        median = cv2.medianBlur(out, 3)
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.float32)
        med_gray = cv2.cvtColor(median, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mask = np.abs(gray - med_gray) > 25.0
        if not np.any(mask):
            break
        out[mask] = median[mask]
    return out


def adaptive_median(img: np.ndarray, max_k: int = 7) -> np.ndarray:
    """
    Textbook adaptive median filter: grow the window per pixel only until the
    local median is not itself an extremum. Pixels that are already consistent
    with their neighbourhood are left untouched at every window size.
    """
    out = img.copy()
    remaining = np.ones(img.shape[:2], dtype=bool)
    for k in range(3, max_k + 1, 2):
        med = cv2.medianBlur(out, k)
        lo = cv2.erode(out, np.ones((k, k), np.uint8))
        hi = cv2.dilate(out, np.ones((k, k), np.uint8))
        med_g = cv2.cvtColor(med, cv2.COLOR_RGB2GRAY).astype(np.int16)
        lo_g = cv2.cvtColor(lo, cv2.COLOR_RGB2GRAY).astype(np.int16)
        hi_g = cv2.cvtColor(hi, cv2.COLOR_RGB2GRAY).astype(np.int16)
        cur_g = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.int16)

        # Median is a usable estimate when it is not itself a local extremum.
        med_ok = (med_g > lo_g) & (med_g < hi_g)
        # The pixel is an impulse when it sits at the local extreme.
        is_impulse = (cur_g <= lo_g) | (cur_g >= hi_g)
        replace = remaining & med_ok & is_impulse
        out[replace] = med[replace]
        remaining &= ~med_ok  # only unresolved pixels grow their window
        if not np.any(remaining):
            break
    return out


def nlm_only(img: np.ndarray, h: int = 12) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


def adaptive_then_nlm(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(adaptive_median(img), None, 6, 6, 7, 21)


# ── metrics ───────────────────────────────────────────────────────────────────
def residual(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.std(g - cv2.GaussianBlur(g, (0, 0), 1.5)))


def edge_keep(orig: np.ndarray, cand: np.ndarray) -> float:
    def edges(x):
        g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        return cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 60, 150)
    e0, e1 = edges(orig), edges(cand)
    n0 = float(np.sum(e0 > 0))
    return float(np.sum(e1 > 0)) / max(n0, 1.0)


def face_detail(img: np.ndarray, box) -> float:
    if box is None:
        return float("nan")
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return float("nan")
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.std(g - cv2.GaussianBlur(g, (0, 0), 1.5)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("-d", "--outdir", type=Path, default=ROOT / "debug_output" / "impulse_variants")
    args = ap.parse_args()

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"!! cannot read {args.image}")
        return 1
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Approximate face box via the project detector so face_detail is comparable.
    box = None
    try:
        from ai.analysis.face_detector import FaceDetector
        from ai.pipeline.pipeline_context import PipelineContext
        from ai.analysis.quality_analyzer import QualityAnalyzer

        ctx = PipelineContext(rgb, QualityAnalyzer.analyze(rgb))
        det = FaceDetector()
        if hasattr(det, "load"):
            try:
                det.load("cpu")
            except Exception:
                pass
        det.detect(ctx)
        faces = ctx.image_metadata.get("faces", [])
        if faces:
            x0, y0, x1, y1 = [int(v) for v in faces[0]["bbox"]]
            box = (max(0, x0), max(0, y0), min(rgb.shape[1], x1), min(rgb.shape[0], y1))
    except Exception as exc:
        print(f"(face detection unavailable: {exc})")

    variants = {
        "00_input": rgb,
        "01_shipped_k5_dilated": shipped(rgb, 5),
        "02_shipped_k3_dilated": shipped(rgb, 3),
        "03_no_dilate_k5": no_dilate(rgb, 5),
        "04_no_dilate_k3": no_dilate(rgb, 3),
        "05_iterated_k3_x3": iterated_k3(rgb, 3),
        "06_adaptive_median": adaptive_median(rgb),
        "07_nlm_only": nlm_only(rgb),
        "08_adaptive_then_nlm": adaptive_then_nlm(rgb),
    }

    print(f"input {rgb.shape[1]}x{rgb.shape[0]}  face_box={box}\n")
    print(f"{'variant':<26} {'residual':>9} {'edge_keep':>10} {'face_detail':>12}")
    for name, img in variants.items():
        cv2.imwrite(str(args.outdir / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"{name:<26} {residual(img):>9.2f} {edge_keep(rgb, img):>9.1%} "
              f"{face_detail(img, box):>12.2f}")
    print(f"\nwrote {len(variants)} variants to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
