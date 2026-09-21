# External Integrations

**Analysis Date:** 2026-09-19

## APIs & External Services

**Model Hubs & Weight Distribution:**
- Hugging Face Hub - Model weight acquisition for DDColor checkpoint (`piddnad/DDColor-models` -> `backend/ai/weights/ddcolor/ddcolor_modelscope.pth`).
  - SDK/Client: Direct HTTP via `urllib.request` in `backend/ai/install_ddcolor.py`.
  - Auth: None (publicly accessible repository weights).
- GitHub Releases - Model weight hosting for GFPGAN and Real-ESRGAN releases.
  - SDK/Client: `urllib.request` downloading to `backend/ai/weights/`.
  - Auth: None (public asset downloads).

## Data Storage

**Databases:**
- None - PixRevive does not require an external SQL or NoSQL database (PostgreSQL, SQLite, MySQL, Redis, MongoDB).
- Metadata Registry: Local JSON flat file (`backend/training_data/registry.json`) maintained for training datasets.

**File Storage:**
- Local filesystem only:
  - Weight Storage: `backend/ai/weights/` stores all offline neural network checkpoints (`GFPGANv1.4.pth`, `RealESRGAN_x4plus.pth`, `big-lama.pt`, `CodeFormer/codeformer.pth`, `ddcolor/ddcolor_modelscope.pth`).
  - In-Flight & Temp Results: `backend/temp_results/` stores exported PNG results for task retrieval via `/api/process/{task_id}/result`.
  - Diagnostic Snapshots: `debug_output/` captures intermediate stage visual outputs and logs when `PIXREVIVE_DEBUG=1`.
  - Training Data: `backend/training_data/bw` and `backend/training_data/color`.

**Caching:**
- In-memory process cache:
  - Task state dictionary: `_tasks` in `backend/routers/image_router.py` guarded by `_tasks_lock = threading.Lock()`.
  - Background ZIP upload tracker: `upload_progress_store` in `backend/routers/admin_router.py` with automated 1-hour TTL pruning.
  - Model weights singleton: `RestorationPipeline` held in memory by `backend/services/image_service.py` to eliminate per-request cold-starts.

## Authentication & Identity

**Auth Provider:**
- Custom / Unauthenticated Development Mode:
  - Current API routes (`/api/process`, `/api/process-stream`, `/api/health`, `/api/admin/*`) are designed for local evaluation, research, and self-hosted execution without mandatory bearer tokens or OAuth.
  - Admin endpoints optionally verify `X-Admin-Token` if `PIXREVIVE_ADMIN_TOKEN` is configured.
  - CORS allowlist enforced in `backend/main.py` using `ALLOWED_ORIGINS` to control browser cross-origin requests.

## Monitoring & Observability

**Error Tracking:**
- Standard Python `logging` module configured across backend components (`backend/routers/image_router.py`, `backend/ai/pipeline/restoration_pipeline.py`).
- No external third-party APM or exception reporting service (e.g., Sentry, Datadog) is currently integrated.

**Logs:**
- Standard stream logging via Uvicorn console.
- Diagnostic Logger (`backend/ai/pipeline/debug_logger.py`): When `PIXREVIVE_DEBUG=1`, outputs execution duration timings (`06_timings.txt`), comprehensive pipeline logs (`07_full_log.txt`), and JSON metadata (`08_metadata.json`) in `debug_output/`.

## CI/CD & Deployment

**Hosting:**
- Self-hosted local workstation or server deployment.
- Backend runs on `http://127.0.0.1:8000` via Uvicorn ASGI server.
- Frontend runs on `http://localhost:5173` (Vite dev server) or preview on `http://localhost:4173`.

**CI Pipeline:**
- GitHub Actions: `.github/workflows/playwright.yml` executes Playwright browser tests on push and pull requests against `main` and `master` branches, uploading `playwright-report` artifacts.
