# Current PixRevive Pipeline (Baseline)

## Overview
This document records the existing pipeline implemented in `backend/ai_model/inference.py` prior to the `restoration-v2` refactoring.

## Pipeline Flow

```text
Input (Bytes -> PIL RGB)
  ↓
Stage 0: Denoising (Optional)
  - Uses `cv2.fastNlMeansDenoisingColored`
  ↓
Stage 0.5: Generative Inpainting (Optional)
  - Damage Detection: Brightness contrast (OpenCV)
  - Stage A: TELEA inpainting
  - Stage B: RealESRGAN texture generation
  - Stage C: LAB color preservation
  ↓
Stage 1: RealESRGAN 4x Super Resolution
  - Uses `RRDBNet` (RealESRGAN_x4plus)
  - Downscales input if > 512x512 to prevent OOM
  ↓
Stage 2: Gentle Contrast
  - Uses CLAHE on the L channel of LAB color space
  ↓
Stage 3: Light Sharpening
  - Adaptive sharpen using Gaussian blur unsharp mask
  ↓
Stage 4: ECCV16 Colorization (Opt-in)
  - Uses `ImageColorizer` (ECCV16 model)
  ↓
JPEG Output (Quality 95)
```

## Model Details
* **Super Resolution Model Name**: RealESRGAN_x4plus (RRDBNet)
* **Colorization Model Name**: ECCV16
* **Input Resolution Limit**: Downscales images larger than 512x512 before SR.
* **Device Used**: Auto-detect (CUDA if available, fallback to CPU)
* **Memory Management**: Tiled inference for RealESRGAN (tile_size=192), aggressive downscaling if pixel count > 262,144 pixels.
