# PixRevive — Complete Project Documentation
> **For Exam Preparation | Capstone Project**  
> AI-Powered Photo Restoration Web Application

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Backend — Frameworks & Packages](#4-backend--frameworks--packages)
5. [Frontend — Frameworks & Packages](#5-frontend--frameworks--packages)
6. [AI Models Used](#6-ai-models-used)
7. [The AI Pipeline — Stage-by-Stage](#7-the-ai-pipeline--stage-by-stage)
8. [Pipeline Routing (Degradation-Aware)](#8-pipeline-routing-degradation-aware)
9. [Module Breakdown](#9-module-breakdown)
10. [API Endpoints](#10-api-endpoints)
11. [Data Flow — End-to-End](#11-data-flow--end-to-end)
12. [Key Algorithms & Techniques](#12-key-algorithms--techniques)
13. [Precautions & Safety Mechanisms](#13-precautions--safety-mechanisms)
14. [Instructions to Run the Project](#14-instructions-to-run-the-project)
15. [Model Weights & Files](#15-model-weights--files)
16. [Important Design Decisions](#16-important-design-decisions)
17. [Glossary](#17-glossary)

---

## 1. Project Overview

**PixRevive** is a full-stack web application that uses deep learning and classical computer vision to automatically restore old, damaged, and black-and-white photographs. Users upload a photo; the system analyzes it, routes it through an intelligent multi-stage AI pipeline, and returns a high-quality restored PNG.

### Core Features
| Feature | Description |
|---|---|
| **Damage Repair** | Removes scratches, white flakes, dust, and emulsion damage |
| **Face Restoration** | Restores degraded facial details using GFPGAN |
| **Colorization** | Converts grayscale/sepia photos to natural color using ECCV16 |
| **Super-Resolution** | 4x upscaling using RealESRGAN with tiled inference |
| **Smart Denoising** | Removes film grain, noise while preserving edges |
| **Enhancement** | CLAHE contrast + Unsharp Mask (USM) sharpening |

### Project Type
- **Capstone / Karstone Project**
- Full-stack application (React frontend + FastAPI backend + AI pipeline)
- Modular, production-grade architecture

---

## 2. System Architecture

```
+-------------------------------------------------------------+
|                     FRONTEND (React + Vite)                 |
|   ImageUploader -> ProgressBar -> Result Display            |
|   Runs on: http://localhost:5173                             |
+------------------------+------------------------------------+
                         |  HTTP / Server-Sent Events (SSE)
                         v
+-------------------------------------------------------------+
|                   BACKEND (FastAPI)                         |
|   main.py -> Routers -> ImageService -> RestorationPipeline |
|   Runs on: http://localhost:8000                             |
+------------------------+------------------------------------+
                         |
                         v
+-------------------------------------------------------------+
|                 AI PIPELINE (18 Stages)                     |
|                                                             |
|  Normalize -> Analyze -> Route -> Face Detect -> Noise      |
|  -> Damage Repair -> Denoising -> Face Restore -> Colorize  |
|  -> Color Artifact Reduce -> Super-Resolution               |
|  -> Harmonize -> Enhance -> Color Refine -> Debug Log       |
|  -> Identity Validate -> Quality Control -> PNG Encode      |
+-------------------------------------------------------------+
```

### Directory Structure

```
PixRevive/
├── backend/
│   ├── main.py                         # FastAPI app entry point
│   ├── routers/
│   │   ├── image_router.py             # Core image processing endpoints
│   │   └── admin_router.py             # Model management / admin panel
│   ├── services/
│   │   └── image_service.py            # Thin orchestration layer (singleton)
│   └── ai/
│       ├── pipeline/
│       │   ├── restoration_pipeline.py # 18-stage pipeline (main engine)
│       │   ├── pipeline_router.py      # Degradation-aware routing
│       │   ├── pipeline_context.py     # Shared pipeline state
│       │   └── debug_logger.py         # Debug image saving
│       ├── preprocessing/
│       │   ├── image_normalizer.py     # Load, validate, orient, convert to RGB
│       │   └── halftone_suppressor.py  # Remove halftone mesh artifacts
│       ├── analysis/
│       │   ├── quality_analyzer.py     # Blur/noise/contrast/color-type analysis
│       │   ├── face_detector.py        # OpenCV Haar/DNN face detection
│       │   └── quality_service.py      # Output quality validation
│       ├── damage/
│       │   ├── enhanced_detector.py    # Multi-scale damage detection
│       │   ├── tearmark_detector.py    # Tearline / border damage detection
│       │   └── mask_refiner.py         # Morphological mask cleanup
│       ├── restoration/
│       │   ├── scratch_restorer.py     # Microsoft "Old Photos" model
│       │   ├── inpainting_engine.py    # LaMa + classical CV cascade
│       │   ├── lama_restorer.py        # LaMa large-mask inpainting
│       │   ├── face_restorer.py        # GFPGAN face restoration
│       │   ├── super_resolution.py     # RealESRGAN x4 tiled
│       │   ├── rrdbnet.py              # RRDB network definition
│       │   └── pixel_preserver.py      # Non-damaged pixel protection
│       ├── colorization/
│       │   ├── eccv16_colorizer.py     # ECCV16 Caffe DNN colorizer
│       │   └── color_refiner.py        # Color artifact reduction
│       ├── postprocessing/
│       │   ├── denoiser.py             # Wavelet + NLM + bilateral denoising
│       │   ├── noise_router.py         # Noise type classification & routing
│       │   └── enhancer.py             # CLAHE + USM + Harmonizer
│       ├── validation/
│       │   └── identity_validator.py   # Facial landmark identity check
│       ├── models/
│       │   ├── base_model.py           # Abstract base class for restorers
│       │   └── zeroscratches/          # Microsoft scratch removal model files
│       └── weights/                    # Downloaded model weight files
│           ├── GFPGANv1.4.pth
│           ├── RealESRGAN_x4plus.pth
│           ├── big-lama.pt
│           ├── facelib/
│           └── eccv16/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageUploader.jsx   # Main upload/restore UI
│   │   │   ├── Navbar.jsx
│   │   │   ├── ProgressBar.jsx
│   │   │   └── Footer.jsx
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   ├── Documentation.jsx
│   │   │   └── WeightsAdmin.jsx    # Admin panel for model management
│   │   └── services/               # Axios API call utilities
│   └── index.html
├── tests/                          # Backend unit tests
├── e2e/                            # End-to-end Playwright tests
└── restart_all.bat / start_dev.bat # Windows startup scripts
```

---

## 3. Technology Stack

| Layer | Technology | Version |
|---|---|---|
| **Frontend Framework** | React | ^19.2.7 |
| **Frontend Build Tool** | Vite | ^8.1.1 |
| **CSS Framework** | TailwindCSS | ^4.3.2 |
| **Frontend Routing** | React Router DOM | ^7.18.1 |
| **HTTP Client** | Axios | ^1.18.1 |
| **Icons** | Lucide React | ^1.24.0 |
| **Backend Framework** | FastAPI | Latest |
| **Backend Server** | Uvicorn (ASGI) | Latest |
| **Deep Learning** | PyTorch | >=2.0.0 |
| **Computer Vision** | OpenCV (contrib) | 4.12.0.88 |
| **Image Processing** | Pillow | >=9.5.0 |
| **Numerical Computing** | NumPy | >=1.24.0 |
| **Scientific Computing** | SciPy | >=1.10.0 |
| **Image Science** | scikit-image | >=0.20.0 |
| **Wavelet Processing** | PyWavelets (pywt) | Included |
| **E2E Testing** | Playwright | Latest |

---

## 4. Backend — Frameworks & Packages

### FastAPI
The backend is built with **FastAPI**, a modern, high-performance Python web framework.

- Automatic API docs (Swagger at `/docs`)
- Async support for non-blocking I/O
- Type validation via Pydantic
- **CORS Middleware** configured: `allow_origins=["*"]` (development only)
- **StaticFiles** middleware serves training data directory

### Python Packages (ai/requirements.txt)

| Package | Purpose |
|---|---|
| `torch>=2.0.0` | Core deep learning framework — runs GFPGAN, RealESRGAN, LaMa |
| `torchvision>=0.15.0` | Vision transforms used by GFPGAN loading |
| `torchaudio>=2.0.0` | Required for torch environment completeness |
| `opencv-contrib-python==4.12.0.88` | **Critical** — DNN inference (ECCV16), guided filter, Canny edges, morphological ops |
| `pillow>=9.5.0` | Image I/O — loading, saving PNG with embedded metadata |
| `numpy>=1.24.0` | All array operations throughout the pipeline |
| `scipy>=1.10.0` | Scientific filters |
| `scikit-image>=0.20.0` | SSIM metrics, image quality |
| `basicsr>=1.4.2` | Backbone for RealESRGAN architecture |
| `facexlib>=0.3.0` | Face detection/parsing models used internally by GFPGAN |
| `gfpgan>=1.3.8` | GFPGAN face restoration library |
| `realesrgan>=0.3.0` | Real-ESRGAN super-resolution library |
| `omegaconf>=2.3.0` | Config management for models |
| `huggingface_hub>=0.14.0` | Model weight downloading from HuggingFace |
| `pyyaml>=6.0` | YAML config parsing |
| `einops>=0.6.1` | Tensor rearrangement for attention-based models |

### Application Architecture (Backend)

```
main.py  (FastAPI App)
|
+-- /api/*           --> image_router.py     (upload, process, stream, result)
+-- /api/admin/*     --> admin_router.py     (weight management, model status)
+-- /health          --> built-in health check

image_router.py  -->  ImageService.process_image()
ImageService     -->  RestorationPipeline.process()  [singleton]
```

---

## 5. Frontend — Frameworks & Packages

### React 19 + Vite 8
- **React**: UI library for building the component tree
- **Vite**: Build tool / dev server for instant HMR
- **React Router DOM v7**: Client-side routing between Home, Documentation, WeightsAdmin

### Key Frontend Components

| Component | File | Purpose |
|---|---|---|
| **ImageUploader** | `ImageUploader.jsx` | Drag-and-drop upload, option toggles (restore/repair/colorize/denoise), progress display, before/after comparison |
| **ProgressBar** | `ProgressBar.jsx` | Visual progress indicator powered by SSE stream |
| **Navbar** | `Navbar.jsx` | Top navigation bar |
| **WeightsAdmin** | `WeightsAdmin.jsx` | Admin panel for downloading/managing AI model weights |
| **Documentation** | `Documentation.jsx` | In-app user documentation page |

### Communication with Backend
- **Axios**: Used for regular HTTP requests (upload, status polling)
- **Server-Sent Events (SSE)**: Real-time progress streaming via `/api/process-stream`
- SSE events: `progress` (stage updates), `complete` (done), `error` (failure)

---

## 6. AI Models Used

### 1. ECCV16 Colorizer (Caffe DNN)
- **Full Name:** "Colorful Image Colorization" — Zhang et al., ECCV 2016
- **Type:** Caffe CNN model, loaded via OpenCV DNN (`cv2.dnn.readNetFromCaffe`)
- **Purpose:** Converts grayscale/sepia photos to natural color
- **Input:** L channel (lightness) from LAB color space, resized to 224×224
- **Output:** AB channels (color) upscaled back to original resolution
- **Files Needed:**
  - `colorization_deploy_v2.prototxt` — network architecture
  - `colorization_release_v2.caffemodel` — weights (~128MB)
  - `pts_in_hull.npy` — 313 color cluster prior probability points
- **Key Technical Detail:** Predicts 313 ab-space color bins. Class rebalancing layer (`class8_ab`) with prior probabilities loaded manually.
- **Safety Guards:** Skipped if image < 96px on any side OR damage > 35%

### 2. GFPGAN v1.4 (Face Restoration GAN)
- **Full Name:** Generative Facial Prior GAN
- **Type:** GAN (Generative Adversarial Network) — PyTorch
- **Purpose:** Restores degraded, blurry, or damaged face regions
- **Architecture:** `"clean"` arch, channel multiplier = 2, upscale = 1 (SR done separately)
- **Weight File:** `GFPGANv1.4.pth` (~332MB)
- **Dependency Models:** facexlib face detection/parsing models in `weights/facelib/`
- **Identity Preservation:**
  - Adaptive blend weight: 30% (sharp) / 40% (medium) / 50% (blurry)
  - SSIM check: if SSIM < 0.55, weight drops to 22%
  - 22% high-frequency skin texture from original added back to prevent "plastic" look
  - Face mask feathered with 35×35 Gaussian blur for soft edges

### 3. RealESRGAN x4 (Super-Resolution)
- **Full Name:** Real-Enhanced Super-Resolution GAN
- **Type:** RRDB (Residual in Residual Dense Block) Network — PyTorch
- **Purpose:** 4x spatial upscaling — makes low-res photos crisp and sharp
- **Weight File:** `RealESRGAN_x4plus.pth`
- **Inference:** Tiled inference — tile_size=256, tile_pad=32 (prevents OOM)
- **Output Cap:** Max 4096×4096, then normalized to 2K (longest side = 2048)
- **Key Fix vs. Old Version:** Removed destructive MAX_INPUT_PIXELS=512×512 hard cap

### 4. LaMa (Large Mask Inpainting)
- **Full Name:** LaMa — Resolution-robust Large Mask inpainting
- **Type:** PyTorch TorchScript model (`torch.jit.load`)
- **Purpose:** Fills large damaged areas with plausible content
- **Weight File:** `big-lama.pt`
- **Requirements:** 8-pixel alignment (pad_factor=8)
- **Mask Dilation:** 7×7 kernel ensures coverage of damage edges
- **Pixel Preservation:** Only inpainted pixels are written back; undamaged areas untouched

### 5. Microsoft "Old Photos Back to Life" (zeroscratches)
- **Library:** `zeroscratches` (wraps Microsoft CVPR 2020 model)
- **Purpose:** Primary scratch/damage restorer — end-to-end neural network for old photos
- **Fallback:** If unavailable, classical cascade (EnhancedDamageDetector + InpaintingEngine) takes over

### 6. Face Detection
- Uses **OpenCV DNN** or **facexlib's** pretrained face detection
- Face bounding boxes stored in `context.image_metadata["faces"]` as `{"bbox": [x0, y0, x1, y1]}`

---

## 7. The AI Pipeline — Stage-by-Stage

The `RestorationPipeline.process()` method executes **18 ordered stages**.  
Models are loaded in **background threads** at startup (server responds immediately).

```
Stage | Name                            | Progress %
------|---------------------------------|-----------
  1   | Normalize & Analyze             | 5-10%
  2   | Color-type Normalization        | (part of stage 1)
  3   | Degradation-aware Route         | (internal)
  4   | Face Detection                  | 12%
  5   | Noise Analysis                  | (internal)
  6   | Damage Repair                   | 20-35%
  7   | Smart Denoising                 | 60-65%
  8   | Face Restoration (GFPGAN)       | 55%
  9   | Colorization (ECCV16)           | 75-78%
 10   | Color Artifact Reduction (pre)  | 82%
 11   | Super-Resolution (RealESRGAN)   | 83%
 12   | Harmonization                   | 85%
 13   | Enhancement (CLAHE + USM)       | 88%
 14   | Color Artifact Reduction (post) | 91%
 15   | Debug Logging                   | (conditional)
 16   | Identity Validation             | 93% (debug only)
 17   | Quality Control                 | 95%
 18   | PNG Encode                      | 98-100%
```

### Detailed Stage Descriptions

#### Stage 1 — Normalize & Analyze
- `ImageNormalizer.normalize()`: Loads safely, EXIF orientation correction, handles transparency (composites over white), converts to RGB numpy array.
- `QualityAnalyzer.analyze()`: Computes `blur_score`, `noise_score`, `contrast_score`, `color_type`, `damage_percentage`, `has_white_flakes`, `width`, `height`.
- All metadata stored in `PipelineContext`.

#### Stage 2 — Color-type Normalization
- **Grayscale:** Strips any tint → true grayscale → 3-channel RGB
- **Sepia + colorization requested:** Strips sepia warmth → neutral grayscale for clean ECCV16 input
- **Color:** Preserves original colors as-is

#### Stage 3 — Degradation-aware Route
- `PipelineRouter.route()` reads metadata and selects one of 6 preset pipeline routes (see Section 8)

#### Stage 4 — Face Detection
- `FaceDetector.detect()` stores face bounding boxes in context metadata

#### Stage 5 — Noise Analysis
- `NoiseRouter.analyze()` classifies noise type and stores `noise_profile`

#### Stage 6 — Damage Repair
**Condition:** `enable_repair=True` AND (`has_white_flakes=True` OR `damage_pct >= 5%`)

**Primary Path:** `ScratchRestorer.restore()` — Microsoft neural network (zeroscratches)

**Fallback Classical Cascade:**
1. `EnhancedDamageDetector.detect()` — multi-scale morphological damage mask
2. `TearmarkDetector.detect()` — tear lines / border damage
3. `HalftoneSuppressor.suppress()` — removes halftone printing patterns
4. `MaskRefiner.refine()` — morphological mask cleanup
5. `InpaintingEngine.inpaint()` — LaMa large-mask inpainting

#### Stage 7 — Smart Denoising
**Triggers:** `enable_denoise=True` OR grayscale image with `noise_score > 6.0` OR `noise_profile.has_noise=True`

**Denoising Strategy by Noise Level (sigma):**
| Noise Level | Strategy |
|---|---|
| sigma < 4 | Gentle wavelet pass (bior2.2, threshold=0.25) |
| sigma 4–12 | Wavelet + dynamic bilateral filter |
| sigma > 12 | Multi-stage wavelet + Fast NLM |

**Artifact Prevention:** `ArtifactDetector` checks blocking/over-smoothing; rolls back if detected.
**Stylization Pass:** `cv2.stylization()` after denoising (sigma_s=60, sigma_r=0.45) for residual grain.

#### Stage 8 — Face Restoration (GFPGAN)
- Runs only when: faces detected AND (blurry OR has damage)
- Adaptive blend weight 30–50% based on blur_score
- SSIM < 0.55 → weight drops to 22% (identity safety)
- 22% high-freq skin texture from original added back
- 35×35 Gaussian feathering on face mask for soft edges

#### Stage 9 — Colorization (ECCV16)
- Only for `color_type` = grayscale/sepia AND `enable_colorization=True`
- Image → LAB space → L channel → ECCV16 → AB prediction → merge → RGB
- Saturation capped at 180 (HSV) to prevent oversaturation
- **Rollback:** If >88% of high-saturation pixels share one hue → revert to original
- Color applied at **90% opacity** for natural look

#### Stage 10 — Color Artifact Reduction (Pre-SR)
- `ColorArtifactReducer.reduce_artifacts()` on AB channels in LAB space
- Bilateral filtering of color channels only

#### Stage 11 — Super-Resolution
- `SuperResolution.restore()` — RealESRGAN x4 with tiled inference
- Tile size: 256px, overlap padding: 32px
- Output cap: 4096×4096, then normalized to 2K (longest side = 2048)

#### Stage 12 — Harmonization
- `Harmonizer.harmonize()` — smooths seams between restored patches and original
- `cv2.ximgproc.guidedFilter` (radius=10, eps=100) on dilated rim of damage mask
- Skipped if no damage mask exists

#### Stage 13 — Enhancement
- **CLAHE:** Applied only when `contrast_score < 30.0`
- **USM (Unsharp Mask):** Applied only when `blur_score >= 60` (skipped on soft images to avoid halos)
  - Parameters: percent=80, threshold=5 (gentler than previous version)

#### Stage 14 — Color Artifact Reduction (Post-SR)
- Same as Stage 10, applied again after super-resolution to clean upscaling artifacts

#### Stage 15 — Debug Logging
- Saves intermediate stage images to disk only when `PIXREVIVE_DEBUG=1`

#### Stage 16 — Identity Validation
- Facial landmark stability check (debug-only)
- Warning logged but NEVER blocks the pipeline

#### Stage 17 — Quality Control
- `QualityService.validate()` — structural quality check
- If fails: **reverts to original image** as safe fallback

#### Stage 18 — PNG Encode
- Lossless PNG with embedded text metadata:
  - `Processing-Log`, `Pipeline-Time`, `Identity-Preserved`, `Pipeline-Version`, `Pipeline-Route`

---

## 8. Pipeline Routing (Degradation-Aware)

`PipelineRouter` selects one of 6 routes based on analyzed image properties — avoids running expensive operations unnecessarily.

### Route Selection Logic

| Condition | Route Selected |
|---|---|
| color_type is grayscale or sepia | **BW_RESTORE** |
| damage_percentage >= 20% (heavy) | **DAMAGED_FACE** |
| Has damage AND blurry | **DAMAGED_FACE** |
| Has damage (no blur) | **DAMAGED_NO_FACE** |
| Noisy AND blurry | **NOISY_FACE** |
| Noisy only | **NOISY_CLEAN** |
| Clean, sharp | **CLEAN_ENHANCE** |
| All other | **SAFE_FALLBACK** |

### Route Stage Flags

| Route | Damage Repair | Face Restore | Denoise | SR |
|---|:---:|:---:|:---:|:---:|
| CLEAN_ENHANCE | No | No | No | Yes |
| NOISY_CLEAN | No | No | Yes | Yes |
| NOISY_FACE | No | Yes | Yes | Yes |
| DAMAGED_NO_FACE | Yes | No | Yes | Yes |
| DAMAGED_FACE | Yes | Yes | Yes | Yes |
| BW_RESTORE | No | Yes | Yes | Yes |
| SAFE_FALLBACK | No | Yes | Yes | Yes |

> **Colorization is NEVER auto-routed** — always user-opt-in via `enable_colorization` flag.

---

## 9. Module Breakdown

### `pipeline/pipeline_context.py`
Shared mutable state for one pipeline request:
- `working_image`: current numpy array being processed
- `image_metadata`: dict with all analysis results
- `damage_masks`: dict of damage masks (raw, refined, final)
- `processing_log`: list of log strings from each stage
- `start_timer()` / `end_timer()`: stage timing utilities

### `preprocessing/image_normalizer.py`
- Sets `Image.MAX_IMAGE_PIXELS = 100,000,000` (pixel bomb prevention)
- Calls `img.verify()` then re-opens (verify() leaves file unstable)
- Applies `ImageOps.exif_transpose()` for correct orientation
- Composites transparent images over white background before RGB conversion

### `preprocessing/halftone_suppressor.py`
- Detects and suppresses halftone dot patterns from print scans
- FFT-based filtering to remove periodic dot-pattern frequency components

### `analysis/quality_analyzer.py`
Returns metadata dict:
- `blur_score`: Laplacian variance (higher = sharper)
- `noise_score`: mean deviation from 5x5 Gaussian blur
- `contrast_score`: std deviation of grayscale
- `color_type`: "grayscale" | "sepia" | "color"
- `damage_percentage`: estimated % damaged pixels
- `has_white_flakes`: boolean for emulsion damage

**Color Type Detection:**
- Grayscale: mean RGB channel difference < 5.0
- Sepia: warm pixel ratio > 0.70 OR (low saturation + warm hues)
- Color: everything else

### `damage/enhanced_detector.py`
Multi-stage detector:
1. White flakes (bright, low-saturation, high local contrast)
2. Black spots (dark relative to local background)
3. Scratches (linear morphological patterns)
4. Edge-aware filtering to prevent false positives
5. Confidence-weighted binary mask output

### `postprocessing/denoiser.py`
Three denoising classes:
- **`ArtifactDetector`**: Detects blocking (local variance analysis) and over-smoothing (Canny edge loss)
- **`WaveletDenoiser`**: `pywt.wavedec2()` / `waverec2()` with soft thresholding. Uses `bior2.2` (smooth, prevents Haar step artifacts)
- **`AdaptiveDenoiser`**: Selects strategy based on noise sigma
- **`SmartDenoiser`**: Iterative wrapper; rolls back if candidate is worse than original

### `postprocessing/enhancer.py`
- **`Harmonizer`**: Guided filter seam blending at damage mask boundaries
- **`Enhancer`**: CLAHE adaptive contrast, USM sharpening, optional color cast correction

### `colorization/color_refiner.py`
`ColorArtifactReducer.reduce_artifacts()`:
- Reduces color bleeding at edges after colorization
- Bilateral filtering on AB channels of LAB image
- Applied pre-SR and post-SR when colorization occurred

---

## 10. API Endpoints

All endpoints prefixed with `/api`.

### `POST /api/upload`
- **Purpose:** Validate and acknowledge file upload (no processing)
- **Input:** `multipart/form-data` with `file`
- **Validation:** MIME type (JPEG/PNG/WebP), max 25MB
- **Returns:** `{ filename, content_type, size_bytes, message }`

### `POST /api/process`
- **Purpose:** Synchronous — blocks until complete, returns PNG bytes
- **Input:** `file` + form fields: `restore`, `repair`, `colorize`, `denoise` (bools), `device` (str)
- **Returns:** PNG binary, `Content-Disposition: attachment; filename="restored.png"`

### `POST /api/process-stream`
- **Purpose:** Async streaming via SSE — returns progress events, then task_id
- **Input:** Same as `/api/process`
- **Returns:** `text/event-stream`
  - `event: progress` → `{ stage, percent, message }`
  - `event: complete` → `{ percent: 100, message, task_id }`
  - `event: error` → `{ message }`
- **Architecture:** Thread → asyncio.Queue → SSE generator

### `GET /api/process/{task_id}/result`
- **Purpose:** Download PNG after stream completes
- **Returns:** PNG file response

### `GET /api/process/{task_id}/status`
- **Purpose:** Poll task status (non-SSE fallback)
- **Returns:** `{ task_id, status, progress[], error }`

### `GET /health` or `GET /api/health`
- **Returns:** `{ status, models: { fast_models, colorizer, ddcolor }, ready }`

### Task Management
- **TTL:** Tasks older than **30 minutes** auto-pruned (memory + temp file deleted)
- **In-memory store:** `_tasks` dict → `{ status, progress[], result_path, created_at }`

---

## 11. Data Flow — End-to-End

```
User Browser
  |
  | 1. Select image, choose options (restore/repair/colorize/denoise)
  | 2. POST /api/process-stream (multipart form)
  v
FastAPI image_router.py
  |
  | 3. Validate MIME type & file size (max 25MB)
  | 4. Create task_id (UUID hex, 12 chars)
  | 5. Start background thread: ImageService.process_image()
  | 6. Open SSE stream back to browser
  v
ImageService.process_image()
  |
  | 7. Forward to RestorationPipeline.process()
  v
RestorationPipeline.process()
  |
  | 8.  Wait for models (max 120s if still loading)
  | 9.  Stage 1: ImageNormalizer.normalize() -> RGB numpy array
  | 10. Stage 1: QualityAnalyzer.analyze() -> metadata dict
  | 11. Stage 2: Strip sepia/grayscale tints
  | 12. Stage 3: PipelineRouter.route() -> route preset
  | 13. Stage 4: FaceDetector.detect() -> face bboxes
  | 14. Stage 5: NoiseRouter.analyze() -> noise_profile
  |
  | [If damage detected]
  | 15. Stage 6: ScratchRestorer -> OR -> classical cascade
  |
  | [If denoising needed]
  | 16. Stage 7: NoiseRouter.reduce() -> SmartDenoiser fallback
  |              -> cv2.stylization() grain removal
  |
  | [If faces found]
  | 17. Stage 8: FaceRestorer (GFPGAN) + identity blend
  |
  | [If colorization requested]
  | 18. Stage 9:  ECCV16Colorizer.colorize() (LAB -> AB prediction)
  | 19. Stage 10: ColorArtifactReducer (pre-SR)
  |
  | 20. Stage 11: SuperResolution.restore() (RealESRGAN x4 tiled)
  | 21. Stage 12: Harmonizer.harmonize() (guided filter seam blend)
  | 22. Stage 13: Enhancer.enhance() (CLAHE + USM)
  |
  | [If colorization was applied]
  | 23. Stage 14: ColorArtifactReducer (post-SR)
  |
  | 24. Stages 15-17: Debug / Identity / Quality Control
  | 25. Stage 18: Encode as PNG with metadata in text chunks
  |
  | 26. Return PNG bytes
  v
image_router.py
  |
  | 27. Save PNG to temp_results/{task_id}.png
  | 28. Emit SSE "complete" event with task_id
  v
User Browser
  |
  | 29. On "complete": GET /api/process/{task_id}/result
  | 30. Display before/after comparison
```

---

## 12. Key Algorithms & Techniques

### Laplacian Variance (Blur Detection)
```python
blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
```
- High variance → sharp (many edges, high frequency detail)
- Low variance → blurry

### Wavelet Denoising (bior2.2)
```python
coeffs = pywt.wavedec2(channel, 'bior2.2', level=3)
thresholded = pywt.threshold(d, value=threshold, mode='soft')
reconstructed = pywt.waverec2(new_coeffs, 'bior2.2')
```
- `bior2.2` chosen over `haar` to prevent 2x2 block/step artifacts
- Soft thresholding: shrinks coefficients toward zero without hard cuts

### Non-Local Means (NLM) Denoising
```python
cv2.fastNlMeansDenoisingColored(bgr, None, h=h_luma, hColor=h_luma,
    templateWindowSize=7, searchWindowSize=21)
```
- Compares each pixel to similar patches across the full image
- `h` controls filtering strength (higher → more smoothing, risk of blurring)

### CLAHE
- Divides image into tiles, equalizes histogram per tile
- Clips contrast limit to prevent noise amplification
- Applied globally only when `contrast_score < 30.0`

### Guided Filter (Harmonizer)
```python
cv2.ximgproc.guidedFilter(img, img, radius=10, eps=100)
```
- Edge-preserving filter — smooths regions while respecting natural edges
- Blends damage mask rim boundaries seamlessly

### SSIM (Structural Similarity Index)
- Measured on face ROI between original and GFPGAN-restored versions
- Components: luminance, contrast, structure
- Threshold: SSIM < 0.55 → reduce blend weight to 22%

### LAB Color Space
- ECCV16 works in LAB: L (lightness), A and B (color channels)
- Model predicts AB only; original L preserved → natural brightness retained

### Tiled Inference (RealESRGAN)
- Split into 256×256 tiles with 32px overlap
- Each tile processed independently
- Stitch using no-overlap output regions
- Prevents OOM on large inputs without quality loss from pre-downscaling

---

## 13. Precautions & Safety Mechanisms

### Input Validation

| Check | Limit |
|---|---|
| File MIME type | JPEG, PNG, WebP only |
| File size | Max **25MB** |
| Image pixel count | Max **100,000,000** pixels |
| Image format | Verified with `PIL.Image.verify()` before processing |

### Processing Safety

| Mechanism | Description |
|---|---|
| Model loading timeout | Max 120 seconds; RuntimeError if exceeded |
| Colorization size guard | ECCV16 skipped if image < 96px on any side |
| Colorization damage guard | ECCV16 skipped if damage > 35% (color bleeding risk) |
| Colorization rollback | Reverted if pathological single-hue saturation (>88%) |
| Face identity guard (SSIM) | GFPGAN blend reduced to 22% if SSIM < 0.55 |
| GFPGAN conditional run | Skipped on sharp, undamaged images to preserve identity |
| Denoising artifact detection | Rolls back if blocking or over-smoothing detected |
| Quality control fallback | Returns original if output fails structural validation |
| Exception isolation | Every stage wrapped in try/except; failures logged, not propagated |
| Pixel Preserver | LaMa output composited only onto damage areas; undamaged pixels untouched |

### Memory Management
- Task results older than **30 minutes** auto-pruned
- RealESRGAN output capped at **4096×4096** before 2K normalization
- Models loaded in background threads; server starts immediately
- CUDA cache cleared on model unload

### Security
- **Pixel bomb prevention:** `Image.MAX_IMAGE_PIXELS = 100,000,000`
- **CORS:** Restricted in production (currently `"*"` for development only)
- **File validation:** MIME type + Pillow content verification

---

## 14. Instructions to Run the Project

### Prerequisites
- Python 3.9+ (with venv support)
- Node.js 18+ (with npm)
- Windows OS (startup scripts are .bat)
- GPU (optional; CUDA recommended for speed)

### Step 1: Clone Repository
```bash
git clone <repo-url>
cd PixRevive
```

### Step 2: Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install fastapi uvicorn
pip install -r ai/requirements.txt
```

### Step 3: Download AI Model Weights
```bash
# Option A: Script
python backend/ai/download_weights.py

# Option B: ECCV16 only
python backend/ai/install_eccv16.py

# Option C: Use WeightsAdmin UI after server starts
```

Required files in `backend/ai/weights/`:
- `GFPGANv1.4.pth`
- `RealESRGAN_x4plus.pth`
- `big-lama.pt`
- `facelib/` (detection + parsing models — auto-downloaded)
- `eccv16/colorization_deploy_v2.prototxt`
- `eccv16/colorization_release_v2.caffemodel`
- `eccv16/pts_in_hull.npy`

### Step 4: Frontend Setup
```bash
cd frontend
npm install
```

### Step 5: Start Both Servers

**Option A — Batch script (Windows):**
```bat
.\start_dev.bat
```

**Option B — Manual:**
```bash
# Terminal 1: Backend
cd backend
venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
cd frontend
npm run dev
```

### Step 6: Open in Browser
- Frontend: http://localhost:5173
- API docs (Swagger): http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Debug Mode
```bat
set PIXREVIVE_DEBUG=1
REM Then restart backend — saves stage images per request
```

### Running Tests
```bash
# Backend unit tests
cd backend && pytest tests/

# E2E tests
npx playwright test
```

---

## 15. Model Weights & Files

| Model | File | Size | Source |
|---|---|---|---|
| GFPGAN v1.4 | `GFPGANv1.4.pth` | ~332MB | GitHub: TencentARC/GFPGAN |
| RealESRGAN x4 | `RealESRGAN_x4plus.pth` | ~67MB | GitHub: xinntao/Real-ESRGAN |
| LaMa | `big-lama.pt` | ~200MB | HuggingFace: smartywu/big-lama |
| ECCV16 Caffe | `colorization_release_v2.caffemodel` | ~128MB | richzhang/colorization |
| ECCV16 Prototxt | `colorization_deploy_v2.prototxt` | ~4KB | richzhang/colorization |
| ECCV16 Prior | `pts_in_hull.npy` | ~2KB | richzhang/colorization |
| Face Detection | `detection_Resnet50_Final.pth` | ~109MB | facexlib (auto-downloaded) |
| Face Parsing | `parsing_parsenet.pth` | ~85MB | facexlib (auto-downloaded) |

---

## 16. Important Design Decisions

### Why an 18-Stage Pipeline?
Each stage is modular and independently toggled. The pipeline router skips unnecessary stages (e.g., no face restoration on landscapes), reducing processing time from minutes to seconds.

### Why Background Thread Loading?
AI models take 20–60 seconds to load. Background threads allow FastAPI to start immediately and respond to health checks while models load in parallel. A `threading.Event` (`_models_ready`) gates processing until ready.

### Why PNG Output (Not JPEG)?
PNG is lossless — restoration work should never re-introduce JPEG compression artifacts. PNG text chunks also carry pipeline metadata for debugging.

### Why Tiled Inference for RealESRGAN?
Running a full 4K image through RRDB in one pass exceeds GPU VRAM. Tiling with overlap padding avoids OOM AND avoids quality loss from downscaling.

### Why Two Color Artifact Reduction Passes?
Colorization introduces AB channel bleeding at edges. SR upscaling amplifies these. Running refinement before SR (reduce source artifacts) and after SR (clean upscaling artifacts) produces cleaner results.

### Why is Colorization Never Auto-Routed?
Colorization is an irreversible artistic decision. Auto-colorizing a photo the user wanted in black-and-white would be incorrect. It is always **user-opt-in only**.

### Why zeroscratches Over Pure Classical CV?
Classical mask-then-inpaint struggles with complex overlapping damage that is hard to mask accurately. The Microsoft end-to-end model was trained on real old photos and produces significantly better results, especially for scratches and emulsion flakes.

### Why SSE for Progress?
Processing takes 30–120 seconds. Streaming progress over Server-Sent Events gives real-time stage-by-stage feedback (e.g., "Detecting faces... 12%", "Colorizing... 75%") instead of a blank loading screen.

---

## 17. Glossary

| Term | Definition |
|---|---|
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization — local contrast enhancement |
| **USM** | Unsharp Mask — sharpening by subtracting a blurred version from original |
| **NLM** | Non-Local Means — denoising by comparing patches across the image |
| **SSIM** | Structural Similarity Index — image quality metric (0–1, higher is more similar) |
| **LAB** | Perceptual color space: L=lightness, A=green-red axis, B=blue-yellow axis |
| **GAN** | Generative Adversarial Network — ML model with generator + discriminator |
| **RRDB** | Residual in Residual Dense Block — backbone architecture of RealESRGAN |
| **Inpainting** | Filling missing/damaged image regions with plausible content |
| **Halftone** | Printing pattern of tiny dots (newspaper/magazine) that can fake damage |
| **EXIF** | Metadata embedded in photos; includes orientation flag |
| **Caffe** | Early deep learning framework; ECCV16 was trained in Caffe |
| **TorchScript** | PyTorch model serialization format used by LaMa (`big-lama.pt`) |
| **SSE** | Server-Sent Events — one-way real-time HTTP streaming server to browser |
| **VRAM** | Video RAM — GPU memory; limited resource for deep learning |
| **OOM** | Out of Memory — error when model/image exceeds available RAM/VRAM |
| **Feathering** | Gaussian-blurring a mask edge for smooth, seamless blending |
| **Tiled inference** | Processing large images in overlapping tiles to avoid memory limits |
| **Pixel Preserver** | Composites inpainted result only onto damaged regions |
| **Pipeline Route** | Preset configuration dict controlling which stages run |
| **PipelineContext** | Shared mutable state object passed through all 18 pipeline stages |
| **bior2.2** | Biorthogonal 2.2 wavelet — smooth wavelet basis preventing step artifacts |
| **soft threshold** | Wavelet technique that shrinks coefficients toward zero smoothly |
| **Guided filter** | Edge-preserving smoothing filter used for seam blending |
| **SSIM identity check** | Comparing face region before/after GFPGAN to detect identity drift |

---

*Document generated for PixRevive Capstone Project — August 2026*
*Pipeline Version: v3.0 | Backend: FastAPI + PyTorch | Frontend: React 19 + Vite 8*
