# PixRevive Failure Checklist

## First Checks

- Confirm global Python 3.12 is active: `python --version`.
- Do not use `backend/venv` if its launcher points to a missing Python.
- Confirm OpenCV has expected APIs: `CascadeClassifier`, `ximgproc` if guided filtering is required.
- Clear or timestamp `backend/debug_output` before comparing stage files, because stale masks can remain from previous runs.

## Stage Localization

- `01_original.png`: verify orientation, RGB order, and input dimensions.
- `02_pre_halftone.png` and `03_after_halftone.png`: compare for unwanted blur or luminance shifts.
- `04_final_mask.png`: check mask coverage. Any mask over eyes, lips, nose, or intact skin must be justified by visible damage.
- `05_after_inpaint.png`: first place to catch hallucinated skin, rewritten facial features, or filled texture errors.
- `09_after_face_restore.png`: check identity, eye shape, mouth, and nose before colorization.
- `11_after_colorize.png`: check ECCV16 artifacts before SR.
- `05_final_working.png`: final post-SR output; compare against earlier stages to isolate SR amplification.

## Common Root Causes

- Green or pink sheets: chroma cleanup/refinement applied too broadly, LAB A/B channel mistakes, or ECCV16 AB scaling mismatch.
- Sepia image turns gray too early: contamination removal neutralized chroma before colorization or before the user wanted it.
- B&W image gets color artifacts while colorization is off: color cleanup/refinement ran on grayscale output.
- Face changes identity: mask touched clean facial pixels, GFPGAN blend weight too high, or validation skipped landmarks.
- Mask misses flakes: detector thresholds too strict, saturation guard excludes faded white damage, or halftone suppression ran before detection.
- Mask attacks real detail: texture/noise is being treated as damage; route to denoising instead of inpainting.
- Inpaint crashes on large masks: inspect NumPy broadcasting and mask/image shape after resize.
- SR makes honeycomb texture: halftone/mesh not suppressed before RealESRGAN.

## Fix Order

1. Fix environment and deterministic crashes.
2. Fix routing so only needed stages run.
3. Fix mask precision before tuning inpainting.
4. Fix face identity validation before increasing GFPGAN strength.
5. Fix colorization and color cleanup after grayscale/sepia classification is reliable.
6. Fix SR last, because SR amplifies earlier mistakes.

## Minimum Verification

- `python -m compileall -q backend\ai backend\routers backend\services tests`
- `python -m unittest discover tests`
- `npm run --prefix frontend lint`
- `npm run --prefix frontend build`
