# PixRevive Restoration Debug Plan

## Goal

Restore degraded, noisy, scratched, faded, grayscale, or sepia images into natural-looking, identity-preserving, colorized/upscaled outputs.

The project already has most required modules. The main gap is control: stages run too statically, quality gates are too weak, and some post-processing touches images it should leave alone.

## Current Diagnosis

1. `PipelineRouter` is a placeholder.
   - The docs describe degradation-based routing, but `backend/ai/pipeline/pipeline_router.py` always logs "Static Baseline Pipeline".
   - Result: denoise, repair, contamination cleanup, face restoration, colorization, and SR decisions are scattered across stage code instead of one route plan.

2. Debug artifacts can mislead.
   - `backend/debug_output` contains stale files from previous runs.
   - Example: `02_classical_mask.png` was `447x447` while the latest logged input was `196x148`, so it could not belong to the same run.

3. The environment must stay consistent.
   - Use global Python 3.12 only.
   - `backend/venv` is broken and points to missing `C:\Python314\python.exe`.
   - OpenCV is pinned to `opencv-contrib-python==4.12.0.88` so `CascadeClassifier` and `ximgproc` are available.
   - Keep plain `opencv-python` uninstalled to avoid conflicting `cv2` wheels.

4. Damage analysis is over-sensitive.
   - Latest debug metadata reported `damage_percentage=77.7%` on a small grayscale noisy image, while enhanced mask coverage was about `9.9%`.
   - This means the high-level damage estimate is not reliable enough for routing.

5. Inpainting had a deterministic crash path.
   - `_multiscale_fill` could fail because OpenCV collapsed a one-channel blur from `(h,w,1)` to `(h,w)`, breaking NumPy broadcasting.
   - Fixed in `backend/ai/restoration/inpainting_engine.py`.

6. Color cleanup/refinement was too broad.
   - Grayscale images could be interpreted as sepia-like because hue is zero when saturation is zero.
   - Final color artifact reduction ran even for non-colorized grayscale output.
   - Fixed in `backend/ai/colorization/advanced_contamination_remover.py` and `backend/ai/pipeline/restoration_pipeline.py`.

7. Color edge detection had a numeric bug.
   - `ColorSmoother` multiplied LAB L by `255` even though L was already in `0..255`, then cast to `uint8`.
   - Fixed in `backend/ai/colorization/color_refiner.py`.

8. Quality control is not visual enough.
   - `QualityService.validate` only catches empty, NaN, all-black, or all-white outputs.
   - It does not catch green/pink color fields, severe hue shifts, identity drift, or over-smoothed faces.

9. "No changes" in the frontend can be caused by final rollback.
   - After OpenCV was fixed, `IdentityValidator` started running on real faces.
   - It compared original-size faces against 2K upscaled faces, marked identity as failed, and the pipeline returned `original_image`.
   - The validator now resizes restored output to original dimensions for final identity validation.
   - Final pipeline rollback now happens only on explicit landmark instability, not expected pixel-level changes from GFPGAN/color/SR.

## Fix Plan

### Phase 1: Stabilize Runtime

- Rebuild dependencies for global Python 3.12.
- Replace or pin OpenCV to a package that provides required APIs:
  - `opencv-contrib-python` if `ximgproc.guidedFilter` is required.
  - Verify `cv2.CascadeClassifier` exists or replace landmark validation with facexlib/RetinaFace-derived checks.
- Delete or ignore `backend/venv` until it is rebuilt for Python 3.12.
- Add a startup health check that reports:
  - Python executable and version.
  - OpenCV version and required API availability.
  - model weight existence.
  - ECCV16/GFPGAN/RealESRGAN/LaMa load status.

### Phase 2: Make Routing Real

Implement a `PipelineRoute` object in `pipeline_router.py` with explicit booleans:

- `run_halftone`
- `run_damage_detection`
- `run_inpaint`
- `run_denoise`
- `run_face_restore`
- `run_decontaminate`
- `run_colorize`
- `run_color_refine`
- `run_super_resolution`

Routing should use `QualityAnalyzer` plus user options. The pipeline should obey the route instead of every stage deciding independently.

### Phase 3: Fix Damage Masks Before Models

- Treat `damage_percentage` as diagnostic only until recalibrated.
- Add mask metrics:
  - total coverage
  - face coverage
  - largest component percentage
  - border-touching percentage
  - thin-line vs blob ratio
- Add hard safety valves:
  - skip inpainting if total mask is too high unless the image is clearly torn.
  - skip or shrink mask inside eyes, nose, and mouth unless local damage confidence is high.
  - route diffuse noise to denoising, not inpainting.

### Phase 4: Protect Identity

- Make identity validation fail "unknown" instead of pass when landmarks cannot run.
- Validate face checks before and after:
  - inpainting
  - GFPGAN
  - colorization
  - SR
- Keep rollback stage-specific. If GFPGAN fails identity, roll back only GFPGAN, not the entire restoration.
- Prefer CodeFormer later for controllable fidelity when time permits; GFPGAN can stay as the fast baseline.

### Phase 5: Colorization Discipline

- Only decontaminate sepia/color scans, never plain grayscale.
- ECCV16 should receive clean luminance, not inpaint/face outputs with chroma side effects.
- Add chroma sanity checks:
  - max green/magenta contaminated area
  - mean LAB A/B drift from pre-colorization
  - saturation percentile caps
- Run color artifact reduction only when the output is expected to be color.

### Phase 6: Super-Resolution Last

- Keep RealESRGAN last.
- Do not tune SR until pre-SR stages look clean.
- Add pre-SR checks for halftone/mesh, mask seams, and color artifacts, because SR amplifies all of them.

## Verification Checklist

Run after every backend change:

```powershell
python -m compileall -q backend\ai backend\routers backend\services tests
python -m unittest discover tests
npm run --prefix frontend lint
npm run --prefix frontend build
```

Restart the backend after Python code changes:

```powershell
cd /d D:\Desktop\kapstone\kapstone\PixRevive\backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

For visual checks, process one image per bucket:

- grayscale noisy image
- sepia scratched portrait
- color photo with mild fading
- heavily flaked portrait
- low-resolution clean face

For each image, compare stage PNGs and record the first bad stage.

## Support Added

- Project skill: `.codex/skills/pixrevive-restoration-debugger/SKILL.md`
- Skill checklist: `.codex/skills/pixrevive-restoration-debugger/references/failure-checklist.md`

Use `$pixrevive-restoration-debugger` for future focused debugging passes.
