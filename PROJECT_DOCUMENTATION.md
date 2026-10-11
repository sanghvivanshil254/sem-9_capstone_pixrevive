# PixRevive — Complete Project Documentation

*Verified against the code on 2026-09-22. Pipeline order, routes, endpoints, and
weights in this document are taken from `backend/ai/pipeline/restoration_pipeline.py`,
`pipeline_router.py`, `backend/routers/*`, and `backend/ai/download_weights.py`.*

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Backend — Frameworks & Packages](#4-backend--frameworks--packages)
5. [Frontend — Frameworks & Packages](#5-frontend--frameworks--packages)
6. [AI Models Used](#6-ai-models-used)
7. [The AI Pipeline — Stage by Stage](#7-the-ai-pipeline--stage-by-stage)
8. [Pipeline Routing](#8-pipeline-routing)
9. [Module Breakdown](#9-module-breakdown)
10. [API Endpoints](#10-api-endpoints)
11. [Data Flow — End to End](#11-data-flow--end-to-end)
12. [Key Algorithms & Techniques](#12-key-algorithms--techniques)
13. [Precautions & Safety Mechanisms](#13-precautions--safety-mechanisms)
14. [Running the Project](#14-running-the-project)
15. [Model Weights](#15-model-weights)
16. [Capabilities & Limitations](#16-capabilities--limitations)

---

## 1. Project Overview

PixRevive restores scanned photographs: it repairs *physical* damage (tears,
cracks, flakes, foxing) first, then denoises, colorizes, restores faces, and
upscales — streaming per-stage progress to the browser over Server-Sent Events.

### Core Features

- 14-stage restoration pipeline with damage-repair-before-colorization ordering
- DDColor (ICCV 2023) colorization for grayscale and sepia scans
- LaMa inpainting of structural damage, with a face-wipeout circuit breaker
- GFPGAN v1.4 face restoration, damage-aware and deferred for small faces
- Real-ESRGAN x4 upscaling, 2K-normalized, tiled
- Monotonic SSE progress stream audited by `scripts/check_progress_stream.py`
- Health endpoint reporting per-model load truth (`load_errors`)

### Project Type

- **Backend:** Python 3.12 / FastAPI (no virtual environments — system interpreter only)
- **Frontend:** React 19 + Vite
- **Inference:** PyTorch (CUDA when available, CPU fallback)

---

## 2. System Architecture

```
Browser (React)
   │  POST /api/process-stream (multipart)
   ▼
FastAPI (backend/main.py)
   ├── routers/image_router.py      /api/upload · /api/process · /api/process-stream · /api/process/{id}/result|status
   └── routers/admin_router.py      /api/admin/weights · /api/admin/training-data · (training = honest 501 stubs)
   ▼
services/image_service.py           singleton façade, task store, GPU semaphore
   ▼
ai/pipeline/restoration_pipeline.py 14-stage orchestrator (stage methods, timers, debug snapshots)
   ├── ai/pipeline/pipeline_router.py    route selection per image
   ├── ai/pipeline/pipeline_context.py   shared state: working_image, damage_masks, metadata, timers
   └── stage modules: ai/preprocessing · ai/damage · ai/restoration · ai/colorization · ai/postprocessing · ai/validation
   ▼
ai/weights/                          all model checkpoints (git-ignored, downloaded by download_weights.py)
```

### Directory Structure

See [CODEBASE_MAP.md](CODEBASE_MAP.md) for the full annotated tree. Top level:

```
backend/
  main.py                     FastAPI app, CORS allowlist, /health
  routers/                    image_router.py, admin_router.py
  services/image_service.py   pipeline façade + task store
  ai/
    pipeline/                 restoration_pipeline, pipeline_router, pipeline_context, debug_logger
    preprocessing/            impulse_denoiser, speck_suppressor, halftone_suppressor, image_normalizer
    damage/                   enhanced_detector, mask_refiner, defect_segmenter, tearmark_detector
    analysis/                 quality_analyzer, face_detector, content_region, quality_service
    restoration/              face_restorer, super_resolution, inpainting_engine, lama_restorer, rrdbnet
    colorization/             ddcolor_colorizer, color_refiner, vendored ddcolor/
    postprocessing/           denoiser, enhancer (Harmonizer), noise_router
    validation/               identity_validator, forensic_validator
    weights/                  checkpoints (git-ignored)
frontend/                     React 19 + Vite (components: ImageUploader, ProgressBar, Navbar, Footer)
tests/                        pytest suite + restoration_dataset fixtures (case1–case12)
scripts/                      check_progress_stream.py, benchmark_quality.py
e2e/                          Playwright specs (run against CDP-attached Chrome)
```

---

## 3. Technology Stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI + Uvicorn |
| Language | Python 3.12 (system interpreter at `C:\Program Files\Python312\python.exe`) |
| Deep learning | PyTorch, torchvision, basicsr 1.4.2 (pinned — GFPGAN/facexlib need it) |
| Imaging | OpenCV (`opencv-contrib-python`, guided filter via `cv2.ximgproc`), NumPy, Pillow |
| Faces | GFPGAN, facexlib, RetinaFace |
| Colorization | DDColor (vendored inference source) |
| Frontend | React 19, Vite, Tailwind CSS, lucide-react |
| Streaming | Server-Sent Events over `StreamingResponse` |
| Testing | pytest (backend), Playwright-over-CDP (e2e) |

---

## 4. Backend — Frameworks & Packages

### FastAPI

- App factory in `backend/main.py`; routers mounted at `/api` (image) and `/api/admin` (admin).
- CORS is an explicit allowlist (`allow_credentials=True` forbids `*`); override via `PIXREVIVE_CORS_ORIGINS`.
- `/static/training_data` is mounted from disk.
- File-type and size validation live in `image_router.py` (`ALLOWED_MIME_TYPES`, `MAX_FILE_SIZE`).
- A GPU semaphore serializes restorations so concurrent requests queue instead of OOM-ing
  (`image_router.py` `_gpu_semaphore(1)`; VRAM-aware concurrency was considered under
  O-2 and deliberately kept as a plain semaphore — a 4 GB card serves one job at a time).

### Key Python Packages (`backend/requirements.txt`)

| Package | Role |
|---|---|
| `torch` / `torchvision` | All model inference |
| `opencv-contrib-python` | Classical CV: black-hat/top-hat detection, NLM, CLAHE, guided filter |
| `basicsr==1.4.2` | Required by GFPGAN/facexlib (pinned — do not let DDColor installs overwrite it) |
| `gfpgan`, `facexlib` | Face restoration stack |
| `realesrgan` | Super-resolution wrapper |
| `huggingface_hub`, `pyyaml` | Weight downloads, config |

### Application Architecture (Backend)

- **Singleton pipeline.** `services/image_service.py` constructs one `RestorationPipeline`; models load on daemon threads at import and `/api/health` reports their true state.
- **PipelineContext.** A single mutable context carries `working_image`, `damage_masks` (sub-masks + `final`/`lama_mask`/`ns_mask`), `image_metadata`, faces/landmarks, timers, and the processing log. `sync_geometry_to_working_image()` rescales masks after any resolution change.
- **Route then stages.** `PipelineRouter.route()` picks one of 6 named route plans; `restore()` executes stages whose route gates (or the user's explicit flags) enable them.
- **Debug snapshots.** With `PIXREVIVE_DEBUG=1`, `DebugLogger.save_stage()` writes numbered PNGs per stage to `debug_output/`.

---

## 5. Frontend — Frameworks & Packages

### React 19 + Vite 8

- Pages: `Home` (restore), `Documentation`, `WeightsAdmin`.
- Components: `ImageUploader` (dropzone), `ProgressBar`, `Navbar` (health pill), `Footer`.

### Key Frontend Components

- **`ProgressBar.jsx`** — mirrors the backend stage emission order (`STAGES` with nominal percentages, `EDGE_LABELS` for `loading`/`complete`/`done`). Resolves the active stage by id, falling back to percent; marks skipped stages done. Must stay in sync with the backend — `scripts/check_progress_stream.py` enforces the contract.
- **`Navbar.jsx`** — polls `/health` every 5 s and renders one of four states: `loading`, `online` (green), `degraded` (amber — some model loader failed; reads `load_errors`), `offline` (red + retry).
- **`WeightsAdmin.jsx`** — admin UI over `/api/admin/weights/*` and `/api/admin/training-data/*`.

### Communication with Backend

- `POST /api/process-stream` (multipart) → SSE events `progress` / `complete` / `error`; then `GET /api/process/{task_id}/result` for the PNG.
- `POST /api/process` for a one-shot synchronous restore.

---

## 6. AI Models Used

### 1. DDColor (ICCV 2023) — Colorization

- `ai/colorization/ddcolor_colorizer.py` wrapping the vendored `ai/colorization/ddcolor/` source; checkpoint `weights/ddcolor/ddcolor_modelscope.pth`.
- 512-px input, learned color head, chroma blended back onto the working luminance; smooth chrominance diffusion across inpainted zones. This is the only colorizer in the codebase.

### 2. GFPGAN v1.4 — Face Restoration

- `ai/restoration/face_restorer.py`, checkpoint `weights/GFPGANv1.4.pth` (plus `facelib/` and `CodeFormer/` support weights).
- Multi-face, damage-aware blending (fidelity 0.85). Faces < 90 px are deferred to a post-super-resolution pass (stage 9c).

### 3. Real-ESRGAN x4plus — Super-Resolution

- `ai/restoration/super_resolution.py`, checkpoint `weights/RealESRGAN_x4plus.pth`, RRDBNet architecture (`rrdbnet.py`).
- Tiled inference; images are normalized toward 2K before ×4 and capped at 4096 px.

### 4. Big-LaMa — Inpainting

- `ai/restoration/lama_restorer.py` loading TorchScript `weights/big-lama.pt`; orchestrated by `inpainting_engine.py` with an OpenCV Telea fallback.
- Fills the refined damage mask; seam feathering and a face-wipeout circuit breaker protect anatomical regions.

### 5. Semantic Defect Segmenter (CVPR 2020 "Bringing Old Photos Back to Life")

- `ai/damage/defect_segmenter.py`, checkpoint `weights/defect_detect/FT_Epoch_latest.pt`.
- Stage 2 gatekeeper: when loaded and confident (< 15% coverage), its mask is authoritative for fine cracks/scratches; classical detectors then only contribute ultra-high-confidence flakes.

### 6. Face Detection

- `ai/analysis/face_detector.py`: RetinaFace (`facelib/detection_Resnet50_Final.pth`) with a Haar fallback; all faces with bounding boxes + 5 landmarks, feeding protection zones in stages 2, 5, and 7.

---

## 7. The AI Pipeline — Stage by Stage

All stages live in `backend/ai/pipeline/restoration_pipeline.py`. Route flags and
user toggles decide which run; ordering is fixed.

| # | Stage | Method | What it does |
|---|---|---|---|
| 0 | Impulse & speck suppression | `_stage_0_despeckle` | ROAD-test salt/pepper filter + SpeckSuppressor (autocorrelation + edge co-location). Runs **first** so specks are never misread as damage. |
| 1 | Degradation analysis | `_stage_1_analyze` | `QualityAnalyzer`: blur score, noise σ (margin-robust MAD), damage %, `color_type` ∈ {grayscale, sepia, color}, grain-field flag. |
| 1b | Early face detection | `_stage_4_face_detect` | RetinaFace + Haar; bboxes, 5 landmarks, ocular/oral protection zones. |
| 2 | Multi-scale fracture masking | `_stage_2_fracture_masking` | U-Net gatekeeper + classical white-flake/wide-crack/scratch/3D-tear detectors + TearmarkDetector (sepia/color) + **dark-spot (foxing/mold) detector with a 2% coverage cap** + grain-field gate + `MaskRefiner` fusion (border strip, face-zone protection, LaMa/NS tier split). |
| 3 | LaMa inpainting | `_stage_3_lama_inpaint` | Big-LaMa fill of the LaMa-tier mask, seam feathering, edge-energy face-wipeout circuit breaker. |
| 2b | Damage cluster smoothing | `_stage_2b_damage_cluster_smooth` | Variance-gated smoothing of inpaint perimeters outside face safe-zones. |
| 3b | Face re-detection | `_stage_3b_refresh_faces` | Re-scan post-inpaint canvas; recovers faces obscured by pre-repair damage. |
| 5 | Spatially-gated denoise | `_stage_5_spatially_gated_denoise` | 3-zone NLM (face adaptive / mid / background) scaled to measured σ; background guard preserves film grain on clean areas. |
| 6 | Colorization | `_stage_6_colorization` | DDColor (only when `color_type` ∈ {grayscale, sepia} and colorization enabled). |
| 6c | Color artifact reduction | inline (`ColorArtifactReducer`) | Edge-aware bleed fix, chroma outlier suppression, banding reduction, face-protected local cast fix, global cast neutralization. |
| 7 | Face restoration | `_stage_7_face_restoration` | GFPGAN v1.4 damage-aware composite; small faces deferred. |
| 8 | Pre-SR anti-aliasing | `_stage_8_presr_antialiasing` | Bilateral filter on L* (+ chroma when noisy) before upscaling. |
| 9 | Super-resolution | `_stage_9_super_resolution` | Real-ESRGAN x4plus tiled; Harmonizer blends inpaint seams. |
| 9c | Deferred face pass | inline | GFPGAN on the 4× canvas for previously-small faces. |
| 10 | Post-SR sharpening | `_stage_10_post_sr_sharpen` | Luminance-only LAB unsharp mask. |
| 10b | Projector exhibition grading | `_stage_10b_projector_grade` | Exhibition CLAHE + shadow-lifting tone curve. |
| 11 | Identity & quality validation | `_stage_11_validate` (+ orchestrator identity check) | IdentityValidator (SSIM + landmark distance), ForensicValidator residual checks. |
| 12 | PNG encode | inline | Final PNG (lossless) + result download. |

### Stage 2 detector wiring (current truth)

`detect_dark_spots()` **is wired** into Stage 2 (foxing/mold, T-1): it runs when
the grain-field gate has not claimed the frame, its contribution is capped at
2% coverage (larger dark responses are shadow/dark-texture false positives and
are discarded), and the accepted sub-mask is stored as
`context.damage_masks["dark_spots"]`.

---

## 8. Pipeline Routing

`ai/pipeline/pipeline_router.py` inspects Stage 1 metadata:

- `has_damage = has_white_flakes or damage_pct >= 5.0`; `heavy_damage = damage_pct >= 20.0`
- `is_noisy = noise_score > 6.0`; `is_blurry = blur_score < 200.0`

### Routes

| Route | Damage repair | Faces | Denoise | SR | Notes |
|---|---|---|---|---|---|
| `CLEAN_ENHANCE` | – | – | – | ✔ | Clean images |
| `NOISY_CLEAN` | – | – | ✔ | ✔ | Noise, no damage |
| `NOISY_FACE` | – | ✔ | ✔ | ✔ | Noisy face |
| `DAMAGED_NO_FACE` | ✔ | – | ✔ | ✔ | Damage, no face |
| `DAMAGED_FACE` | ✔ | ✔ | ✔ | ✔ | Damage + face |
| `BW_RESTORE` | – | ✔ | ✔ | ✔ | Undamaged B&W/sepia, no forced colorization |
| `SAFE_FALLBACK` | – | – | – | ✔ | Conservative default for unknown profiles |

**Repair-before-restore invariant:** damaged grayscale/sepia photos are routed
to `DAMAGED_FACE` ("repair scratches first, then restore") — never straight to
`BW_RESTORE`, which skips repair and lets DDColor colorize scratch pixels with
wrong hues. Routes are advisory: the user's explicit enable/disable flags win.

---

## 9. Module Breakdown

### `ai/pipeline/pipeline_router.py`
Route selection (above). Stores the chosen plan in `metadata["pipeline_route"]`.

### `ai/pipeline/pipeline_context.py`
State container + `sync_geometry_to_working_image()` (mask/bbox/landmark rescale after resolution changes) + processing log.

### `ai/preprocessing/impulse_denoiser.py`
ROAD statistic for salt/pepper outliers; density-invariance regression-tested (`check_impulse_robustness.py`).

### `ai/preprocessing/speck_suppressor.py`
Dense low-amplitude speck fields via autocorrelation + edge co-location.

### `ai/preprocessing/halftone_suppressor.py`
Frequency-domain notch suppression for scanned halftone prints (Phase 4B will wire it as the moiré specialist).

### `ai/preprocessing/image_normalizer.py`
Bit-depth/channel normalization and `detect_and_crop_borders()` for scanner margins.

### `ai/analysis/quality_analyzer.py`
Blur (Laplacian variance), noise σ (content-region robust MAD), color-type classification, grain-field flag, `content_region.py` margin-robust statistics (the case8 fix).

### `ai/damage/enhanced_detector.py`
Classical detectors: white flakes, wide cracks, directional scratches, 3D paper tears, thin crack networks (Hessian + multi-scale black-hat), **dark spots (foxing/mold)**, edge damage.

### `ai/damage/mask_refiner.py`
Fuses sub-masks, strips photo-border artifacts, protects ocular/oral zones, splits into LaMa-tier and NS-tier masks.

### `ai/damage/tearmark_detector.py`
3-pass saturation-aware emulsion-flake detector for sepia/color photos.

### `ai/damage/defect_segmenter.py`
U-Net learned defect segmentation (Stage 2 gatekeeper).

### `ai/postprocessing/noise_router.py`
`NoiseClassifier`/`NoiseRouter` — 30 declared noise types routed to the right filter family.

### `ai/postprocessing/denoiser.py`
Wavelet/adaptive denoiser stack; `SmartDenoiser` used where route flags request extra denoising.

### `ai/postprocessing/enhancer.py`
`Harmonizer` — guided-filter seam blending along inpainting rims. (The former `Enhancer` class was dead code and was removed; its global-cast logic lives on in `ColorArtifactReducer.neutralize_global_cast`.)

### `ai/colorization/color_refiner.py`
`ColorArtifactReducer` (bleed, splotch, banding, local + **global** cast with face protection), `FacialSkinToneProtector`, `BackgroundNeutralizer`.

### `ai/validation/*`
`identity_validator.py` (SSIM + landmark distance), `forensic_validator.py` (residual defect & inpaint quality scoring).

---

## 10. API Endpoints

Base URL `http://localhost:8000`. Interactive docs at `/docs` (FastAPI).

| Endpoint | Purpose |
|---|---|
| `POST /api/upload` | Upload helper (validated, staged for processing) |
| `POST /api/process` | Synchronous restore → PNG bytes |
| `POST /api/process-stream` | SSE restore: `progress` events (`stage`, `percent`, `message`), terminal `complete`/`error` |
| `GET /api/process/{task_id}/result` | Restored PNG |
| `GET /api/process/{task_id}/status` | Task status snapshot |
| `GET /api/health` (alias `/health`) | **Per-model truth**: `status`, `models` booleans (`defect_segmenter`, `lama`, `gfpgan`, `realesrgan`, `ddcolor`), `load_errors` map, `device`, `ready` |
| `POST /api/admin/weights/upload` | Upload/replace a checkpoint |
| `GET /api/admin/weights/status` | On-disk weights inventory |
| `GET/POST/PUT/DELETE /api/admin/training-data/pairs…` | Training-pair CRUD |
| `POST /api/admin/training/start` · `GET /api/admin/training/status` | **Honest 501 stubs** — no trainer exists; endpoints say so instead of pretending |

Admin endpoints are protected by `PIXREVIVE_ADMIN_TOKEN`.

### Health truth

`main.py` delegates to `pipeline.health_status()`. If a background loader
raises, its name and error text appear in `load_errors`; the Navbar renders an
amber **degraded** pill instead of a green "all loaded".

---

## 11. Data Flow — End to End

1. Upload via `/api/process-stream` (multipart: file + flags `restore/repair/colorize/denoise/device`).
2. Task id issued; GPU semaphore acquired; `PipelineContext` created.
3. Stage 1 analysis → router picks the route plan.
4. Stages execute per route/flags, emitting `progress` events (stage id + monotonic percent — audited by `scripts/check_progress_stream.py`).
5. Terminal `complete` (with task id) or `error` event.
6. Client fetches `/api/process/{task_id}/result` (lossless PNG).
7. Task store holds status/result until pruned (Phase 3 adds startup pruning).

---

## 12. Key Algorithms & Techniques

- **ROAD statistic** — rank-ordered absolute deviations for impulse-noise immunity before any analysis.
- **Margin-robust noise estimation** — `content_region.robust_sigma` computes σ on the content mask only; flat scanner margins no longer dilute the estimate (case8 regression).
- **Black-hat / top-hat morphology** — small dark (foxing) / bright (flake) features vs. a scale-adapted elliptical kernel; multi-scale variants for crack networks.
- **Non-Local Means, 3-zone** — separate strength per face / mid / background region; σ-scaled; grain-preserving background guard.
- **DDColor** — transformer-based learned colorization (ICCV 2023).
- **Guided filter (Harmonizer)** — edge-aware seam blending on inpaint rims.
- **SSIM** — identity preservation scoring.
- **LAB color space** — all color correction operates on perceptual channels; luminance-only sharpening avoids chroma halos.
- **Tiled inference** — Real-ESRGAN tiles with overlap for 2K→4K+ outputs within GPU memory.

---

## 13. Precautions & Safety Mechanisms

### Input validation
MIME allowlist + size cap; reject early with 400s.

### Processing safety
- **Face-wipeout circuit breaker** — aborts inpainting whose mask would erase facial anatomy (edge-energy ratio thresholds).
- **Foxing coverage cap** — dark-spot contributions > 2% of frame are treated as shadow/dark-texture false positives and discarded.
- **Grain-field gate** — high content σ with low structural damage estimate routes noise to denoising instead of inpainting.
- **GPU semaphore** — serializes heavy restorations.
- **Model-load isolation** — each loader's failure is recorded, never crashes the app; degraded mode is visible.

### Memory management
Tiled SR, 4096 px output cap, 2K normalization, explicit context image swaps.

### Security
CORS allowlist (credentials-safe), admin token, no shell-outs from request handlers, static mount limited to `training_data`.

---

## 14. Running the Project

### Prerequisites

- Windows with Python 3.12 at `C:\Program Files\Python312\python.exe` — **no virtual environments, ever**; no bare `python`/`pip`/`pytest` (PATH may resolve the wrong interpreter).
- Node.js for the frontend.
- ~2.4 GB free for model weights on the project drive.

### Step 1 — Backend dependencies

```powershell
& "C:\Program Files\Python312\python.exe" -m pip install -r backend/requirements.txt
```

### Step 2 — Model weights

```powershell
& "C:\Program Files\Python312\python.exe" backend/ai/download_weights.py
```

Downloads (GitHub releases / Hugging Face, `urllib` — never a hub cache) into
`backend/ai/weights/`: GFPGANv1.4, RealESRGAN_x4plus, big-lama, CodeFormer +
facelib, DDColor, defect_detect. The alternative WeightsAdmin UI exists, and
DDColor auto-download is controlled by `PIXREVIVE_DDCOLOR_AUTODOWNLOAD`.

### Step 3 — Start the backend

```powershell
cd backend
$env:PYTHONPATH='backend'
& "C:\Program Files\Python312\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Models load on background threads; `GET /api/health` shows per-model truth.

### Step 4 — Start the frontend

```powershell
cd frontend
npm install
npm run dev
```

### Debug mode

`PIXREVIVE_DEBUG=1` writes numbered stage snapshots to `debug_output/` (e.g. `02_after_damage_detect.png`).

### Running tests

```powershell
$env:PYTHONPATH='backend'
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests/
```

E2E specs in `e2e/` run against an already-running Chrome attached via CDP on port 9222 (Playwright driver downloads are blocked in this environment — see `AGENTS.md`).

---

## 15. Model Weights

Canonical manifest: `backend/ai/download_weights.py` → `backend/ai/weights/`
(git-ignored; re-downloadable at any time).

| Path under `weights/` | Model | Pipeline use |
|---|---|---|
| `GFPGANv1.4.pth` | GFPGAN v1.4 | Stages 7 / 9c |
| `RealESRGAN_x4plus.pth` | Real-ESRGAN x4 | Stage 9 |
| `big-lama.pt` (+ `big-lama/` files) | Big-LaMa TorchScript | Stage 3 |
| `ddcolor/ddcolor_modelscope.pth` | DDColor ICCV 2023 | Stage 6 |
| `defect_detect/FT_Epoch_latest.pt` | Semantic defect segmenter (CVPR 2020) | Stage 2 gatekeeper |
| `CodeFormer/codeformer.pth` | CodeFormer | GFPGAN support |
| `facelib/detection_Resnet50_Final.pth`, `facelib/parsing_parsenet.pth` | RetinaFace + parsing | Face detection / parsing |

Note: `realesr-general-x4v3.pth` may exist on disk but is **not** referenced by
the code (removed from the manifest in Phase 1); the retired Microsoft
scratch-removal stack was dropped from `requirements.txt` in Phase 1.

---

## 16. Capabilities & Limitations

Derived from the per-damage-type coverage audit in
[docs/DAMAGE_TAXONOMY_MAPPING.md](docs/DAMAGE_TAXONOMY_MAPPING.md).

### Recovered (deterministic, wired stages)

- **Noise:** impulse/salt-pepper, speckle, Gaussian & film grain (grain deliberately preserved on clean backgrounds), chroma noise, JPEG blocking/ringing; banding partially.
- **Physical damage:** surface scratches, dust/debris, creases & folds, small tears, emulsion cracking/crazing, 3D torn corners/edges.
- **Foxing / mold speckles:** black-hat dark-spot detector wired into Stage 2 with shadow protection (component size + circularity filters, 2% coverage cap).
- **Tone & color:** overall fading (CLAHE + filmic curve), yellowing/sepia neutralization, unnatural casts — local (face-protected) and global (DDColor bias).
- **Faces:** multi-face detection, landmark-guided protection, GFPGAN with damage-aware blending, identity validation.

### Reconstructed (generative — plausible, not guaranteed)

- **Missing regions** via LaMa: small/medium losses fill with high-confidence plausible content; **large fills are fabrication** and should be treated as such (labeling large-coverage fills is planned T-2 work).
- **Damaged faces** via GFPGAN: identity accuracy is fundamentally generative; validators score but cannot guarantee likeness.

### Out of scope (honest gaps)

- Motion / out-of-focus blur (no deblur stage yet — planned T-3).
- Dewarping/curl (T-4), flat-field vignetting (T-5), channel realignment (T-8).
- Illegible text (no OCR), double exposure/ghosting (under-determined — flagged, not processed), silver mirroring beyond tone correction.

---

*Related documents: [README.md](README.md) · [CODEBASE_MAP.md](CODEBASE_MAP.md) · [docs/DAMAGE_TAXONOMY_MAPPING.md](docs/DAMAGE_TAXONOMY_MAPPING.md) · [docs/EXECUTION_PLAN.md](docs/EXECUTION_PLAN.md)*
