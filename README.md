# PixRevive

AI-powered photo restoration for scanned prints: repair physical damage first,
then denoise, colorize, restore faces, and upscale — with per-stage progress
streamed to a React frontend.

**The ordering is the product.** Damage is repaired *before* colorization
(DDColor would otherwise colorize bright scratch pixels with wrong hues), and
impulse noise is suppressed *before* damage detection (specks are
indistinguishable from emulsion flakes once they reach the detectors).

---

## Architecture

`backend/ai/pipeline/restoration_pipeline.py` orchestrates 14 ordered stages:

| # | Stage | Method / Module |
|---|---|---|
| 0 | Impulse & speck suppression | `_stage_0_despeckle` — ROAD filter + SpeckSuppressor |
| 1 | Degradation analysis | `_stage_1_analyze` — `QualityAnalyzer` (blur, noise σ, damage %, color type) |
| 1b | Early face & landmark detection | `_stage_4_face_detect` — RetinaFace + Haar |
| 2 | Multi-scale fracture masking | `_stage_2_fracture_masking` — U-Net DefectSegmenter, classical detectors, TearmarkDetector, dark-spot (foxing) detector, `MaskRefiner` |
| 3 | LaMa deep inpainting | `_stage_3_lama_inpaint` — Big-LaMa + seam feathering + face-wipeout circuit breaker |
| 2b | Damage cluster smoothing | `_stage_2b_damage_cluster_smooth` |
| 3b | Post-inpaint face re-detection | `_stage_3b_refresh_faces` |
| 5 | Spatially-gated 3-zone denoise | `_stage_5_spatially_gated_denoise` — face / mid / background NLM scaled to measured σ |
| 6 | Colorization | `_stage_6_colorization` — DDColor (ICCV 2023) |
| 6c | Color artifact reduction | inline — `ColorArtifactReducer` (bleed, splotch, banding, local + global cast) |
| 7 | Face restoration | `_stage_7_face_restoration` — GFPGAN v1.4, damage-aware |
| 8 | Pre-SR anti-aliasing | `_stage_8_presr_antialiasing` |
| 9 | Super-resolution | `_stage_9_super_resolution` — Real-ESRGAN x4plus (tiled, 2K-normalized) |
| 9c | Deferred face pass | inline — GFPGAN on the 4× canvas for small faces |
| 10 | Post-SR sharpening | `_stage_10_post_sr_sharpen` |
| 10b | Projector exhibition grading | `_stage_10b_projector_grade` |

Routing (`pipeline_router.py`) picks one of 6 named routes per image
(`CLEAN_ENHANCE`, `NOISY_CLEAN`, `NOISY_FACE`, `DAMAGED_NO_FACE`,
`DAMAGED_FACE`, `BW_RESTORE`, plus a conservative `SAFE_FALLBACK`). Damaged
B&W/sepia photos are deliberately routed to `DAMAGED_FACE` — **repairs first,
then restore** — never straight to `BW_RESTORE`.

---

## Requirements

- **Python**: exactly the system interpreter `C:\Program Files\Python312\python.exe`.
  Never a virtual environment; never bare `python`/`pip`/`pytest` (PATH may
  resolve the wrong interpreter).
- **Node** for the Vite + React frontend.
- Model weights in `backend/ai/weights/` (see below) — never in `~/.cache`.

## Setup

```powershell
# 1. Backend dependencies (system interpreter only)
& "C:\Program Files\Python312\python.exe" -m pip install -r backend/requirements.txt

# 2. Model weights (~2.4 GB) — downloads into backend/ai/weights/
& "C:\Program Files\Python312\python.exe" backend/ai/download_weights.py

# 3. Frontend
cd frontend; npm install
```

## Run

```powershell
# Backend (from backend/, with PYTHONPATH=backend)
$env:PYTHONPATH='backend'
& "C:\Program Files\Python312\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000

# Frontend
cd frontend; npm run dev
```

## Test

```powershell
$env:PYTHONPATH='backend'
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests/
```

End-to-end SSE audit (requires the backend running):

```powershell
$env:PYTHONPATH='backend'
& "C:\Program Files\Python312\python.exe" scripts/check_progress_stream.py
```

E2E browser specs live in `e2e/` (Playwright is NOT installed; use an
already-running Chrome via CDP — see `AGENTS.md`).

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Per-model load truth: boolean per model + `load_errors` map |
| `POST /api/process` | Synchronous restore; returns PNG bytes |
| `POST /api/process-stream` | SSE progress stream; terminal `complete`/`error` event, then fetch result |
| `GET /api/process/{task_id}/result` | Restored PNG |
| `GET /api/process/{task_id}/status` | Task status |
| `/api/admin/*` | Weight upload, training-data pairs; `/api/admin/training/start` is an honest 501 stub (no trainer exists) |

## Environment variables

| Variable | Purpose |
|---|---|
| `PIXREVIVE_CORS_ORIGINS` | Comma-separated CORS allowlist override |
| `PIXREVIVE_ADMIN_TOKEN` | Admin router auth |
| `PIXREVIVE_DEBUG` | Debug stage snapshots to `debug_output/` |
| `PIXREVIVE_USE_ORCHESTRATOR` | Enable the experimental orchestrator path |
| `PIXREVIVE_DDCOLOR_WEIGHTS` / `_VARIANT` / `_AUTODOWNLOAD` | DDColor checkpoint selection |

## Model weights

All checkpoints live under `backend/ai/weights/` (git-ignored; re-download with
`backend/ai/download_weights.py`):

| File | Model | Used by |
|---|---|---|
| `GFPGANv1.4.pth` | GFPGAN v1.4 | Stage 7/9c face restoration |
| `RealESRGAN_x4plus.pth` | Real-ESRGAN x4 | Stage 9 super-resolution |
| `big-lama.pt` | Big-LaMa (TorchScript) | Stage 3 inpainting |
| `ddcolor/ddcolor_modelscope.pth` | DDColor (ICCV 2023) | Stage 6 colorization |
| `defect_detect/FT_Epoch_latest.pt` | Semantic defect segmenter (CVPR 2020) | Stage 2 U-Net gatekeeper |
| `CodeFormer/codeformer.pth`, `facelib/*.pth` | CodeFormer + face lib | GFPGAN support weights |

## Capabilities & limitations

Derived from the damage-taxonomy coverage audit
(`docs/DAMAGE_TAXONOMY_MAPPING.md`):

**Recovered (wired stages)**
- Impulse/salt-pepper noise, speckle, Gaussian/film grain (grain preserved on clean backgrounds), chroma noise, JPEG blocking
- Surface scratches, dust/debris, creases & folds, small tears, emulsion cracking/crazing, 3D torn corners
- Foxing / mold speckles (black-hat dark-spot detector with a shadow false-positive cap)
- Overall fading, yellowing/sepia shift, unnatural color casts (local + global, face-protected)
- Faces: detection, landmark-guided protection, GFPGAN restoration with damage-aware blending and identity checks

**Reconstructed (generative — flag, don't trust blindly)**
- Missing regions via LaMa: plausible infills for small/medium losses; large filled areas are *fabrication*, not recovery
- Damaged faces: GFPGAN output is generative; identity accuracy is not guaranteed

**Out of scope (documented gaps)**
- Motion / out-of-focus blur (no deblur stage), dewarping/curling, channel realignment, flat-field vignetting, illegible text, double exposure

## More documentation

- [CODEBASE_MAP.md](CODEBASE_MAP.md) — navigation map with the full module index
- [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) — deep dive: architecture, models, endpoints
- [docs/DAMAGE_TAXONOMY_MAPPING.md](docs/DAMAGE_TAXONOMY_MAPPING.md) — per-damage-type coverage audit
- [docs/EXECUTION_PLAN.md](docs/EXECUTION_PLAN.md) — active work plan
