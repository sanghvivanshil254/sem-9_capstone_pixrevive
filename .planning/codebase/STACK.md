# Technology Stack

**Analysis Date:** 2026-09-27 (full refresh; previous map 2026-09-19)

## Languages

**Primary:**
- Python 3.12 (`C:\Program Files\Python312\python.exe`) — entire backend, now a 16-stage
  restoration pipeline (Stage 12 re-imagine added post-baseline) + hybrid diffusion tier.
  System interpreter only; virtual environments prohibited by policy (`CLAUDE.md`, `AGENTS.md`).
  Pytest runner: `C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe`.
- JavaScript / JSX (React 19) — frontend SPA (`frontend/src/`), Vite build
  (`frontend/vite.config.js`), Playwright e2e suite (`e2e/`).

**Secondary:**
- HTML5 / Tailwind CSS v4 — UI markup + styling (`frontend/src/index.css`).
- Batch scripts — dev launchers (`start_dev.bat`, `restart_all.bat`).

## Runtime

- Backend: system Python 3.12 x86_64 on Windows; no venvs. `PYTHONPATH=backend` required.
- Frontend: Node.js + npm (Vite 7 dev server on 5173, production build in `frontend/dist/`).
- GPU host: GTX 1650 4 GB — resource-tiered inference is designed around this limit
  (`backend/ai/restoration/resource_profiler.py`).

## Frameworks & Key Libraries

**Core (backend/requirements.txt pins):**
- FastAPI + uvicorn — REST + SSE API (`backend/main.py`, `backend/routers/`).
- PyTorch ≥2.0 (measured 2.13.0+cu126) — inference runtime for LaMa, DDColor, GFPGAN,
  Real-ESRGAN, SD diffusion.
- **diffusers ==0.40.0** — SD1.5/SDXL inpaint pipelines, IP-Adapter + FaceID conditioning
  (`backend/ai/restoration/reimaginer.py`, `faceid_adapter.py`). Added in H1; the pinned
  version matters (attention-processor surface differs across versions).
- transformers ≥4.44,<6 · accelerate ≥1.0 (sequential CPU offload) · safetensors ≥0.4.
- opencv (contrib) — YuNet face detection, SFace recognition, cv2.dnn ONNX inference
  (ArcFace), classic CV stages.
- psutil ≥6.0 — resource-tier measurements.
- numpy / Pillow — image arrays, PNG persistence.

**Frontend:** React 19, react-router-dom 7, Vite, Tailwind v4, lucide-react icons.

## Model Weights (all vendored under `backend/ai/weights/`)

Per policy: weights live in the repo tree, fetched via `urllib.request` direct URLs —
never HuggingFace hub caches, never ModelScope (`scripts/fetch_reimagine_weights.py`,
`scripts/fetch_faceid_weights.py`).

| Family | Files | Used by |
|---|---|---|
| LaMa big-lama | `big-lama/`, `big-lama.pt` | Stage 3 inpainting (`lama_restorer.py`) |
| GFPGAN 1.4 / facexlib | `GFPGANv1.4.pth`, `facelib/` | Stage 8 face restore; `basicsr==1.4.2` pinned — never pip-install DDColor |
| Real-ESRGAN | `RealESRGAN_x4plus.pth`, `realesr-general-x4v3.pth` | Stage 9 super-resolution |
| DDColor (ICCV 2023, vendored) | `ddcolor/` | Stage 7 colorization (`backend/ai/colorization/`) |
| SD1.5 inpaint (fp16 safetensors) | `sd15_inpaint/` | Stage 12 re-imagine generator (MINIMAL tier) |
| SDXL inpaint (optional) | `sdxl_inpaint/` (absent on this host by design) | STANDARD/AMPLE generator |
| IP-Adapter (CLIP) | `ip_adapter/models/` + `image_encoder/` | CLIP-crop identity conditioning fallback |
| **FaceID (H4)** | `ip_adapter_faceid/models/ip-adapter-faceid_sd15.bin` (92 MB), `models_lora/…lora.safetensors` (48 MB), `faceid/w600k_r50.onnx` (166 MB) | ArcFace embeddings + identity-token conditioning (`faceid_embedder.py`, `faceid_adapter.py`) |
| YuNet / SFace | `sface/yunet_270face.onnx`, `face_recognition_sface_2021dec.onnx` | face detect, identity gate, embedder landmarks |

## Resource Tiers (measured, `resource_profiler.py`)

| Tier | Floors | Generator | Notes |
|---|---|---|---|
| MINIMAL | ≥3.0 GB VRAM, ≥2.2 GB RAM | SD1.5, fp32 + sequential CPU offload | measured 0.76 GB VRAM, ~165 s/25-step @512; fp16 diffusion is NaN-broken on this torch/GPU |
| STANDARD | ≥8/10 GB | SDXL (absent here) | falls back to SD1.5 when weights missing |
| AMPLE | ≥12/8 GB | SDXL tiled | unverifiable on this host |

Overrides: `REIMAGINE_FORCE_TIER` (never waives floors), `REIMAGINE_IP_SCALE`,
`REIMAGINE_FACEID=0`, `REIMAGINE_PASS_COSINE`/`REIMAGINE_FLAG_COSINE`.

## Configuration

- `backend/main.py` — FastAPI app, CORS, static mounts, startup model preload,
  `/api/health` (readiness = models loaded + `load_errors` empty).
- `frontend/vite.config.js` — dev server 5173 → API proxy 8000.
- `.claude/settings.json` + `scripts/hooks/guard_python_runtime.py` — PreToolUse Bash hook
  mechanically enforcing interpreter rules 1–3.
