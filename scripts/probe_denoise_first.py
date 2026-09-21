"""
probe_denoise_first.py — Test whether the "damage" QualityAnalyzer reports on a
noisy source survives impulse-noise removal.

Hypothesis: on test_artifacts/sources/noise.jpg the 2.97% "damage" and the
has_white_flakes / has_scratches triggers are impulse noise, not structure. If
so, measuring damage AFTER a cheap median denoise collapses the estimate, while
on a genuinely torn photo (teared.jpg) the estimate survives.

Usage:
    PYTHONPATH=backend "C:/Program Files/Python312/python.exe" \
        scripts/probe_denoise_first.py [image ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from ai.analysis.quality_analyzer import QualityAnalyzer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = [
    ROOT / "test_artifacts" / "sources" / "noise.jpg",
    ROOT / "test_artifacts" / "sources" / "teared.jpg",
]


def row(tag: str, m: dict) -> str:
    return (
        f"  {tag:<22} damage={m['damage_percentage']:>6.2f}%  "
        f"scratch={m['scratch_line_pct']:>5.2f}%  "
        f"flakes={str(m['has_white_flakes']):<5} "
        f"scratches={str(m['has_scratches']):<5} "
        f"noise={m['noise_score']:>5.2f}  blur={m['blur_score']:>8.1f}"
    )


def probe(path: Path) -> None:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"!! cannot read {path}")
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    print(f"\n=== {path.name}  {w}x{h} ===")
    print(row("as-uploaded", QualityAnalyzer.analyze(rgb)))

    # Impulse-noise removal only — a median filter leaves tears and scratches
    # intact (they are wider than the kernel) but erases salt-and-pepper specks.
    for k in (3, 5):
        med = cv2.medianBlur(rgb, k)
        print(row(f"median k={k}", QualityAnalyzer.analyze(med)))

    # Edge-preserving alternative for comparison.
    nlm = cv2.fastNlMeansDenoisingColored(rgb, None, 10, 10, 7, 21)
    print(row("fastNlMeans h=10", QualityAnalyzer.analyze(nlm)))


def main(paths: list[Path]) -> int:
    for p in paths:
        probe(p)
    return 0


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]] or DEFAULT_IMAGES
    raise SystemExit(main(args))
