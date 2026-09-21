"""
trace_pipeline.py — Run the REAL pipeline (learned U-Net path included) on one or
more images, dump every stage snapshot, and print the processing log plus
per-sub-mask coverage.

This exists because scripts/check_impulse_robustness.py deliberately stubs torch
and runs `EnhancedDamageDetector(use_defect_segmenter=False)`. That covers the
classical branch, which is the branch that was fixed — but the shipped pipeline
runs the learned DefectSegmenter, GFPGAN, LaMa, DDColor and Real-ESRGAN, and none
of those are exercised there. Anything that only misbehaves with the real models
loaded is invisible to that suite.

Run:
    PYTHONPATH=backend "C:/Program Files/Python312/python.exe" \
        scripts/trace_pipeline.py test_artifacts/sources/noise.jpg

With no arguments it traces the two canonical fixtures.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("PIXREVIVE_DEBUG", "1")

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "backend"))

DEFAULTS = [
    os.path.join(REPO, "test_artifacts", "sources", "noise.jpg"),
    os.path.join(REPO, "test_artifacts", "sources", "teared.jpg"),
]


def _cov(mask) -> float:
    if mask is None or not isinstance(mask, np.ndarray) or mask.size == 0:
        return float("nan")
    return float(np.count_nonzero(mask > 0)) / mask.size * 100.0


def trace(path: str) -> None:
    from ai.pipeline.restoration_pipeline import RestorationPipeline
    from ai.pipeline.debug_logger import DebugLogger

    global _PIPE
    if _PIPE is None:
        print("loading models (LaMa, GFPGAN, Real-ESRGAN, DDColor, U-Net)...")
        _PIPE = RestorationPipeline()
        t0 = time.time()
        _PIPE._models_ready.wait(timeout=600)
        _PIPE._colorizer_ready.wait(timeout=600)
        print(f"models ready in {time.time() - t0:.1f}s: {_PIPE.health_status()['models']}")

    stem = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(REPO, "debug_output", f"trace_{stem}")
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        try:
            os.remove(os.path.join(out_dir, f))
        except OSError:
            pass
    DebugLogger.DEBUG_DIR = out_dir

    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    print(f"\n{'=' * 92}\n{path}   {rgb.shape[1]}x{rgb.shape[0]}\n{'=' * 92}")

    # The pipeline keeps its context internal, so capture it off the stage that
    # always runs. Simplest reliable hook: wrap DebugLogger.save_stage.
    captured = {}
    _orig = DebugLogger.save_stage.__func__

    def _spy(cls, context, stage_name):
        captured["ctx"] = context
        return _orig(cls, context, stage_name)

    DebugLogger.save_stage = classmethod(_spy)
    try:
        t0 = time.time()
        _PIPE.restore(
            image_input=rgb,
            external_metadata={"case_name": f"trace_{stem}"},
            enable_repair=True,
            enable_denoise=True,
            enable_colorization=True,
            enable_face_restore=True,
            enable_sr=True,
        )
        elapsed = time.time() - t0
    finally:
        DebugLogger.save_stage = classmethod(_orig)

    ctx = captured.get("ctx")
    print(f"\n--- processing log ({elapsed:.1f}s) ---")
    for line in (ctx.processing_log if ctx else []):
        print(f"  {line}")

    if ctx:
        print("\n--- damage_masks coverage ---")
        for k, m in ctx.damage_masks.items():
            shape = getattr(m, "shape", None)
            print(f"  {k:<14} {_cov(m):>7.2f}%   shape={shape}")

        print("\n--- metadata ---")
        for k in (
            "width", "height", "color_type", "blur_score", "noise_score",
            "damage_percentage", "contrast_score", "brightness",
        ):
            if k in ctx.image_metadata:
                print(f"  {k:<20} {ctx.image_metadata[k]}")
        print(f"  faces                {len(ctx.image_metadata.get('faces', []))}")

    print(f"\nstage snapshots -> {out_dir}")


_PIPE = None


def main() -> int:
    paths = sys.argv[1:] or DEFAULTS
    for p in paths:
        if not os.path.exists(p):
            print(f"missing: {p}")
            continue
        trace(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
