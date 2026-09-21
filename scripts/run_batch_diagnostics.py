"""
run_batch_diagnostics.py — Clean old outputs and run per-stage pipeline diagnostics on all test cases.
"""

import argparse
import os
import shutil
import time
import logging
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure debug is ON
os.environ["PIXREVIVE_DEBUG"] = "1"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("BatchDiagnostics")

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT_DIR = os.path.join(WORKSPACE_DIR, "tests", "restoration_dataset", "damaged")
DEFAULT_DEBUG_ROOT = os.path.join(WORKSPACE_DIR, "debug_output")
TEMP_RESULTS = os.path.join(WORKSPACE_DIR, "backend", "temp_results")


def clean_old_outputs(debug_root: str):
    """Remove all previous debug PNGs, logs, and temporary results."""
    logger.info("Cleaning old outputs in %s...", debug_root)
    if os.path.exists(debug_root):
        for item in os.listdir(debug_root):
            item_path = os.path.join(debug_root, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)
            elif item.endswith((".png", ".jpg", ".txt", ".json")):
                try:
                    os.remove(item_path)
                except OSError:
                    pass
        logger.info("Cleared debug_output directory.")

    if os.path.exists(TEMP_RESULTS):
        for item in os.listdir(TEMP_RESULTS):
            item_path = os.path.join(TEMP_RESULTS, item)
            if os.path.isfile(item_path):
                try:
                    os.remove(item_path)
                except OSError:
                    pass
        logger.info("Cleared temp_results directory.")


def get_test_cases(input_dir: str):
    """Discover all test image files in input_dir."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    cases = []
    if not os.path.exists(input_dir):
        logger.error("Input directory does not exist: %s", input_dir)
        return cases

    for entry in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(entry)[1].lower()
        if ext in valid_exts:
            case_name = os.path.splitext(entry)[0]
            cases.append((case_name, os.path.join(input_dir, entry)))
    return cases


def run_diagnostics(input_dir: str = DEFAULT_INPUT_DIR, debug_root: str = DEFAULT_DEBUG_ROOT):
    clean_old_outputs(debug_root)

    test_cases = get_test_cases(input_dir)
    if not test_cases:
        logger.warning("No test cases found in %s", input_dir)
        return

    from ai.pipeline.restoration_pipeline import RestorationPipeline
    from ai.pipeline.debug_logger import DebugLogger

    logger.info("Initializing RestorationPipeline and waiting for all models...")
    pipeline = RestorationPipeline()

    # Wait for models to finish loading
    t0 = time.time()
    while not pipeline._models_ready.is_set() or not pipeline._colorizer_ready.is_set():
        time.sleep(0.5)
        if time.time() - t0 > 60:
            logger.warning("Timeout waiting for background models.")
            break
    logger.info("Models ready! Starting batch diagnostics on %d cases.", len(test_cases))

    summary_records = []

    for case_name, img_path in test_cases:
        if not os.path.exists(img_path):
            logger.error("Input file not found: %s", img_path)
            continue

        case_dir = os.path.join(debug_root, case_name)
        os.makedirs(case_dir, exist_ok=True)
        DebugLogger.DEBUG_DIR = case_dir

        logger.info("==================================================")
        logger.info("STARTING TEST CASE: %s (%s)", case_name, os.path.basename(img_path))
        logger.info("==================================================")

        img_pil = Image.open(img_path).convert("RGB")
        img_rgb = np.array(img_pil)
        h, w = img_rgb.shape[:2]
        logger.info("Input resolution: %dx%d", w, h)

        start_time = time.time()
        try:
            restored_rgb = pipeline.restore(
                image_input=img_rgb,
                external_metadata={"case_name": case_name, "filename": os.path.basename(img_path)},
                enable_repair=True,
                enable_denoise=True,
                enable_colorization=True,
                enable_face_restore=True,
                enable_sr=True,
            )
            elapsed = time.time() - start_time
            out_h, out_w = restored_rgb.shape[:2]
            logger.info("SUCCESS: %s restored in %.2fs -> Output: %dx%d", case_name, elapsed, out_w, out_h)
            
            stage_files = [f for f in os.listdir(case_dir) if f.endswith(".png")]
            stage_files.sort()
            summary_records.append({
                "case": case_name,
                "status": "SUCCESS",
                "in_size": f"{w}x{h}",
                "out_size": f"{out_w}x{out_h}",
                "time_sec": round(elapsed, 2),
                "stages_saved": len(stage_files),
                "stage_files": stage_files,
            })
        except Exception as exc:
            logger.exception("FAILED: %s - %s", case_name, exc)
            summary_records.append({
                "case": case_name,
                "status": f"FAILED: {exc}",
                "in_size": f"{w}x{h}",
                "out_size": "N/A",
                "time_sec": round(time.time() - start_time, 2),
                "stages_saved": 0,
                "stage_files": [],
            })

    print("\n" + "=" * 80)
    print("BATCH DIAGNOSTICS SUMMARY")
    print("=" * 80)
    for rec in summary_records:
        print(f"[{rec['status']}] {rec['case']:<32} In: {rec['in_size']:<10} Out: {rec['out_size']:<10} Time: {rec['time_sec']}s (Stages: {rec['stages_saved']})")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run PixRevive batch diagnostics")
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR, help="Directory containing test images")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_DEBUG_ROOT, help="Output directory for debug snapshots")
    args = parser.parse_args()

    run_diagnostics(input_dir=args.input_dir, debug_root=args.out_dir)


if __name__ == "__main__":
    main()
