---
name: pixrevive-restoration-debugger
description: Debug PixRevive image restoration quality failures. Use when investigating noisy, degraded, sepia, grayscale, or damaged-photo outputs; tracing artifacts through masks, inpainting, GFPGAN face restoration, DDColor colorization, RealESRGAN upscaling, quality gates, frontend upload controls, or Python 3.12 environment issues.
---

# PixRevive Restoration Debugger

## Overview

Use this skill to debug PixRevive end to end without guessing. Work from recorded pipeline stages, numeric metadata, and stage logs before changing model parameters.

## Workflow

1. Confirm the runtime first:
   - Use global Python 3.12 via `python`, not `backend/venv`.
   - Check `python --version`, `where.exe python`, and `python -c "import cv2; print(cv2.__version__)"`.
   - Treat missing OpenCV APIs such as `CascadeClassifier` or `ximgproc` as environment failures, not model behavior.

2. Read the pipeline in execution order:
   - `backend/ai/pipeline/restoration_pipeline.py`
   - `backend/ai/preprocessing/`
   - `backend/ai/damage/`
   - `backend/ai/restoration/`
   - `backend/ai/colorization/`
   - `backend/ai/postprocessing/`
   - `backend/ai/validation/`

3. Inspect the latest debug artifacts:
   - `backend/debug_output/07_full_log.txt`
   - `backend/debug_output/08_metadata.json`
   - stage PNGs from `01_original.png` through `11_after_colorize.png`
   - Beware stale files: some masks may remain from older runs if the current stage did not overwrite them.

4. Localize the first bad stage:
   - If masks cover real facial structure, debug damage detection and mask refinement.
   - If output becomes green/pink/purple, debug contamination removal, DDColor chroma blending, and color refinement.
   - If details become waxy or identity changes, debug GFPGAN blending and identity validation.
   - If texture becomes honeycomb or smeared, debug halftone suppression and SR prefiltering.

5. Validate with the lightest checks first:
   - `python -m compileall -q backend\ai backend\routers backend\services tests`
   - `python -m unittest discover tests`
   - `npm run --prefix frontend lint`
   - `npm run --prefix frontend build`

## Debug Rules

- Preserve identity over beauty. A better-looking face that changes eye spacing, eye shape, nose, or mouth position is a failure.
- Prefer routing and guards over stacking more post-processing.
- Do not run chroma cleanup on grayscale-only output unless colorization is requested.
- Avoid broad masks on faces; inpaint only verified damage regions.
- Record exact stage where artifacts first appear before tuning thresholds.

## Reference

Read `references/failure-checklist.md` when diagnosing a visual artifact or planning restoration-quality work.
