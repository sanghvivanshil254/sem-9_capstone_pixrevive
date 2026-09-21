# Technology Stack

**Analysis Date:** 2026-09-19

## Languages

**Primary:**
- Python 3.12 (`C:\Program Files\Python312\python.exe`) - Entire backend service, 14-stage restoration pipeline, computer vision and deep learning inference, automated regression tests. Enforced system interpreter without virtual environments (`CLAUDE.md`, `GEMINI.md`, `AGENTS.md`).
- JavaScript / JSX (ECMAScript 2022+ / React 19) - Frontend single-page application (`frontend/src/`), Vite build configuration (`frontend/vite.config.js`), and Playwright test suite (`e2e/`, `playwright.config.js`).

**Secondary:**
- HTML5 & CSS3 / Modern Tailwind CSS v4 - Frontend interface markup and styling system (`frontend/src/index.css`, `frontend/src/App.css`).
- Batch Scripting (`.bat`) - Developer launcher scripts (`start_dev.bat`, `restart_all.bat`).

## Runtime

**Environment:**
- Backend: System Python 3.12 (x86_64) on Windows. Virtual environments (`venv`, `.venv`, `conda`) are strictly prohibited by repository policy.
- Frontend & Tooling: Node.js (v18+ or v20+) with npm.

**Package Manager:**
- Backend: `pip` via system Python (`"C:\Program Files\Python312\python.exe" -m pip install -r backend/requirements.txt`).
- Frontend: `npm` using lockfile (`package-lock.json` in root and `frontend/package-lock.json`).

## Frameworks

**Core:**
- FastAPI (`fastapi>=0.115`, `uvicorn>=0.30`) - High-performance asynchronous REST and Server-Sent Events (SSE) web API (`backend/main.py`, `backend/routers/image_router.py`, `backend/routers/admin_router.py`).
- PyTorch & TorchVision (`torch>=2.0.0`, `torchvision>=0.15.0`) - Deep learning execution runtime for LaMa, DDColor, GFPGAN, and Real-ESRGAN (`backend/ai/pipeline/restoration_pipeline.py`).
- React 19 (`react^19.2.7`, `react-dom^19.2.7`, `react-router-dom^7.18.1`) - Declarative component UI for interactive restoration, progress visualization, and administration (`frontend/src/App.jsx`).

**Testing:**
- Pytest (`pytest` via `C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe`) - Automated unit, regression, and identity preservation test suite (`tests/`).
- Playwright (`@playwright/test^1.62.1`) - End-to-end browser test automation against Chromium, Firefox, and WebKit (`playwright.config.js`, `e2e/`).

**Build/Dev:**
- Vite 8 (`vite^8.1.1`, `@vitejs/plugin-react^6.0.3`) - Frontend dev server and asset bundler (`frontend/vite.config.js`).
- Tailwind CSS 4 (`tailwindcss^4.3.2`, `@tailwindcss/vite^4.3.2`, `@tailwindcss/postcss^4.3.2`) - Utility-first CSS compiler (`frontend/src/index.css`).
- Oxlint (`oxlint^1.71.0`) - Fast JavaScript linter (`frontend/package.json`).

## Key Dependencies

**Critical:**
- `opencv-contrib-python==4.12.0.88` - Fundamental image processing, morphology, color space conversion (RGB, LAB), and guided filtering (`cv2.ximgproc.guidedFilter` in `backend/ai/postprocessing/enhancer.py`). Contrib build strictly required.
- `basicsr==1.4.2` - Backbone vision restoration library pinned strictly for GFPGAN and facexlib interoperability (`backend/requirements.txt`).
- `facexlib>=0.3.0` & `gfpgan>=1.3.8` - RetinaFace ResNet50 landmark extraction and damage-aware face restoration (`backend/ai/analysis/face_detector.py`, `backend/ai/restoration/face_restorer.py`).
- `timm>=0.9.0` - ConvNeXt vision transformer backbone used by the vendored DDColor model (`backend/ai/colorization/ddcolor/`).
- `big-lama.pt` (TorchScript) - Deep Fourier convolution inpainting engine (`backend/ai/restoration/lama_restorer.py`).
- `RealESRGAN_x4plus.pth` - RRDBNet deep generative 4x super-resolution model (`backend/ai/restoration/super_resolution.py`).
- `PyWavelets>=1.4.0` - Discrete wavelet transform denoiser for texture preservation (`backend/ai/postprocessing/denoiser.py`).
- `scikit-image>=0.20.0` & `scipy>=1.10.0` - Scientific image metrics (SSIM, peak signal-to-noise) and optimization math (`backend/ai/validation/identity_validator.py`).

**Infrastructure:**
- `pydantic>=2.7` - Request and response schema validation for FastAPI endpoints (`backend/routers/admin_router.py`).
- `python-multipart>=0.0.9` - Streaming multipart form data parser for multi-megabyte image uploads (`backend/routers/image_router.py`).
- `huggingface_hub>=0.14.0` - Weights resolution and download utility (`backend/ai/install_ddcolor.py`).
- `lucide-react^1.24.0` - Icon system for React components (`frontend/src/components/`).
- `axios^1.18.1` - Promise-based HTTP client for browser requests (`frontend/src/pages/WeightsAdmin.jsx`).

## Configuration

**Environment Configuration:**
- `PYTHONPATH`: Must include `backend/` directory (`$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"`).
- `PIXREVIVE_DEBUG`: Set to `1` to enable sequential intermediate stage PNG snapshots and timing metrics saved to `debug_output/`.
- `PIXREVIVE_ADMIN_TOKEN`: Optional bearer token protecting administrative upload endpoints in `backend/routers/admin_router.py`.
- `PIXREVIVE_LOAD_SCRATCH_RESTORER`: Flag for optional `zeroscratches` loading (defaults to bypassed).
