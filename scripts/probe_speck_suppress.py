"""
probe_speck_suppress.py — Knock down a dense fine bright-speckle field with a
morphological operator instead of mask-and-inpaint.

Why: on test_artifacts/sources/teared.jpg the emulsion loss is not a handful of
discrete flakes, it is a fine bright texture over the whole print (hair, shirt,
and the woven studio backdrop alike). Mask-and-inpaint handles discrete holes
well, but a lacy 20-30% mask drives LaMa and the Gaussian pyramid fill toward
mush — the 447x447 run smeared the top of the head grey.

A white top-hat isolates bright structures smaller than the structuring element,
which is exactly what this speckle is. Subtracting it suppresses the field while
leaving anything larger than the kernel (eyes, hair mass, collar) untouched.

Metrics:
  speck_energy — std of (L - opening(L, 5)); the bright-speck amplitude itself.
                 This is the number we want down.
  detail       — std of (L - gaussian(L, 1.5)); total high-frequency energy.
                 Should fall much less than speck_energy, or structure went too.
  eye_detail   — detail inside the detected face box; the canary for over-filtering.

Usage:
    PYTHONPATH=backend "C:/Program Files/Python312/python.exe" \
        scripts/probe_speck_suppress.py test_artifacts/sources/teared.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def suppress_specks(
    rgb: np.ndarray,
    k: int = 5,
    threshold: float = 8.0,
    strength: float = 1.0,
    soft: bool = True,
) -> np.ndarray:
    """Subtract the white top-hat of L, keeping chroma untouched.

    threshold sets how bright a speck must be before it is touched, so genuine
    small highlights (catchlights, teeth) below it survive. `soft` ramps the
    correction in over the threshold instead of switching at it, which avoids
    leaving a hard rim around every speck.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]

    opened = cv2.morphologyEx(L, cv2.MORPH_OPEN, _ellipse(k))
    tophat = L - opened

    if soft:
        # Ramp from 0 at `threshold` to full strength one threshold above it.
        gain = np.clip((tophat - threshold) / max(threshold, 1e-6), 0.0, 1.0)
    else:
        gain = (tophat > threshold).astype(np.float32)

    lab[:, :, 0] = np.clip(L - tophat * gain * strength, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def suppress_specks_protected(
    rgb: np.ndarray,
    box: Optional[Tuple[int, int, int, int]],
    k: int = 5,
    threshold: float = 8.0,
    strength: float = 1.0,
    face_strength: float = 0.5,
) -> np.ndarray:
    """Full strength outside the face box, eased off inside it."""
    full = suppress_specks(rgb, k, threshold, strength)
    if box is None:
        return full
    weak = suppress_specks(rgb, k, threshold, face_strength)
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    alpha = np.zeros((h, w), np.float32)
    alpha[y0:y1, x0:x1] = 1.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(h, w) * 0.02)[:, :, None]
    return (weak.astype(np.float32) * alpha
            + full.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


# ── metrics ───────────────────────────────────────────────────────────────────
def _L(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)


def speck_energy(img: np.ndarray, k: int = 5) -> float:
    L = _L(img)
    return float(np.std(L - cv2.morphologyEx(L, cv2.MORPH_OPEN, _ellipse(k))))


def detail(img: np.ndarray) -> float:
    L = _L(img)
    return float(np.std(L - cv2.GaussianBlur(L, (0, 0), 1.5)))


def crop_detail(img: np.ndarray, box) -> float:
    if box is None:
        return float("nan")
    x0, y0, x1, y1 = box
    sub = img[y0:y1, x0:x1]
    return detail(sub) if sub.size else float("nan")


def find_face(rgb: np.ndarray):
    try:
        from ai.analysis.face_detector import FaceDetector
        from ai.analysis.quality_analyzer import QualityAnalyzer
        from ai.pipeline.pipeline_context import PipelineContext

        ctx = PipelineContext(rgb, QualityAnalyzer.analyze(rgb))
        det = FaceDetector()
        try:
            det.load("cpu")
        except Exception:
            pass
        det.detect(ctx)
        faces = ctx.image_metadata.get("faces", [])
        if not faces:
            return None
        x0, y0, x1, y1 = (int(v) for v in faces[0]["bbox"])
        h, w = rgb.shape[:2]
        return (max(0, x0), max(0, y0), min(w, x1), min(h, y1))
    except Exception as exc:
        print(f"(face detection unavailable: {exc})")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("-d", "--outdir", type=Path,
                    default=ROOT / "debug_output" / "speck_suppress")
    args = ap.parse_args()

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"!! cannot read {args.image}")
        return 1
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    args.outdir.mkdir(parents=True, exist_ok=True)
    box = find_face(rgb)
    print(f"{args.image.name}  {rgb.shape[1]}x{rgb.shape[0]}  face={box}\n")

    variants: Dict[str, np.ndarray] = {"00_input": rgb}
    for k in (3, 5, 7):
        for th in (6.0, 10.0):
            variants[f"k{k}_t{int(th)}"] = suppress_specks(rgb, k=k, threshold=th)
    variants["k5_t6_hard"] = suppress_specks(rgb, k=5, threshold=6.0, soft=False)
    variants["k5_t6_faceprot"] = suppress_specks_protected(rgb, box, k=5, threshold=6.0)
    variants["k7_t6_then_k3"] = suppress_specks(
        suppress_specks(rgb, k=7, threshold=6.0), k=3, threshold=6.0
    )

    base_speck, base_det = speck_energy(rgb), detail(rgb)
    print(f"{'variant':<20} {'speck':>7} {'d_speck':>8} {'detail':>7} "
          f"{'d_detail':>9} {'face_det':>9}")
    tiles = []
    for name, img in variants.items():
        sp, dt = speck_energy(img), detail(img)
        print(f"{name:<20} {sp:>7.2f} {sp / base_speck - 1:>+7.0%} {dt:>7.2f} "
              f"{dt / base_det - 1:>+8.0%} {crop_detail(img, box):>9.2f}")
        cv2.imwrite(str(args.outdir / f"{name}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        lab = img.copy()
        cv2.putText(lab, name, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 0), 1, cv2.LINE_AA)
        tiles.append(lab)

    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(args.outdir / "_grid.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"\nwrote {len(variants)} variants + {args.outdir / '_grid.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
