"""
diagnose_small_input.py — Show how QualityAnalyzer/PipelineRouter behave as the
input resolution shrinks.

The user's failing run had a 196x148 source that was routed DAMAGED_FACE with
halftone detected. This script re-runs the analyzer over the same image at
several scales so the resolution-dependence of the damage estimate is visible.

Usage:
    PYTHONPATH=backend "C:/Program Files/Python312/python.exe" \
        scripts/diagnose_small_input.py [image ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from ai.analysis.quality_analyzer import QualityAnalyzer
from ai.pipeline.pipeline_context import PipelineContext
from ai.pipeline.pipeline_router import PipelineRouter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = [
    ROOT / "test_artifacts" / "sources" / "teared.jpg",
    ROOT / "test_artifacts" / "sources" / "noise.jpg",
]
# Long-edge targets. 196 is the user's failing source width.
SCALES = [196, 320, 512, 800, 1200]


def analyse(rgb: np.ndarray) -> tuple[dict, str]:
    meta = QualityAnalyzer.analyze(rgb)
    ctx = PipelineContext(rgb, dict(meta))
    route = PipelineRouter.route(ctx)
    return meta, route["pipeline_name"]


def main(paths: list[Path]) -> int:
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"!! cannot read {path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = rgb.shape[:2]
        print(f"\n=== {path.name}  (native {w0}x{h0}) ===")
        print(
            f"{'size':>10} {'damage%':>8} {'scratch%':>9} {'flakes':>7} "
            f"{'scr':>4} {'blur':>9} {'noise':>6} {'type':>10} {'route':>16}"
        )
        for target in SCALES:
            scale = target / max(w0, h0)
            if scale >= 1.0:
                small = rgb
            else:
                small = cv2.resize(
                    rgb, (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            m, route = analyse(small)
            print(
                f"{m['width']:>4}x{m['height']:<5} "
                f"{m['damage_percentage']:>8.2f} {m['scratch_line_pct']:>9.2f} "
                f"{str(m['has_white_flakes']):>7} {str(m['has_scratches']):>4} "
                f"{m['blur_score']:>9.1f} {m['noise_score']:>6.2f} "
                f"{m['color_type']:>10} {route:>16}"
            )
    return 0


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]] or DEFAULT_IMAGES
    raise SystemExit(main(args))
