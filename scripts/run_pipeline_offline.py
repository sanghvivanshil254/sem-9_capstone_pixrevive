"""
run_pipeline_offline.py — Run the restoration pipeline on a local file without
the HTTP server, printing the full processing log and the emitted progress.

Used to reproduce quality regressions against a known source image. Combine
with PIXREVIVE_DEBUG=1 to dump per-stage images into debug_output/.

Usage:
    PYTHONPATH=backend PIXREVIVE_DEBUG=1 "C:/Program Files/Python312/python.exe" \
        scripts/run_pipeline_offline.py test_artifacts/sources/noise.jpg -o out.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "debug_output" / "result.png")
    ap.add_argument("--no-colorize", action="store_true")
    ap.add_argument("--no-restore", action="store_true")
    ap.add_argument("--no-repair", action="store_true")
    args = ap.parse_args()

    from ai.pipeline.restoration_pipeline import RestorationPipeline

    pipeline = RestorationPipeline()
    print("waiting for models...")
    if not pipeline._wait_for_models(timeout=300.0):
        print("!! fast models failed to load")
        return 1
    if not pipeline.is_colorizer_ready():
        print("waiting for colorizer...")
        pipeline._colorizer_ready.wait(timeout=300.0)
    print(f"models ready (colorizer={pipeline.is_colorizer_ready()})\n")

    def on_progress(stage: str, pct: int, msg: str) -> None:
        print(f"  [{pct:3d}%] {stage:<16} {msg}")

    data = args.image.read_bytes()
    t0 = time.time()
    out = pipeline.process(
        data,
        enable_restoration=not args.no_restore,
        enable_repair=not args.no_repair,
        enable_colorization=not args.no_colorize,
        enable_denoise=False,
        progress_callback=on_progress,
    )
    elapsed = time.time() - t0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(out)
    print(f"\nwrote {args.output} ({len(out)/1024:.0f} KB) in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
