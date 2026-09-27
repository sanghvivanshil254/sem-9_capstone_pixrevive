# External Integrations

**Analysis Date:** 2026-09-27 (full refresh)

PixRevive is **network-outbound-minimal**: no SaaS APIs, no databases, no auth providers,
no webhooks. All AI inference is local. The only outbound network calls are weight
downloads (one-shot, urllib direct, user-triggered or script-run).

## Weight-Download Endpoints (urllib direct, no hub libraries)

| Script | Hosts | Artifacts |
|---|---|---|
| `scripts/fetch_reimagine_weights.py` | huggingface.co (`stable-diffusion-v1-5/stable-diffusion-inpainting`, `diffusers/sdxl-inpaint-0.1`, `stabilityai/sdxl-base-1.0`), h94/IP-Adapter | SD1.5/SDXL inpaint repos, CLIP IP-Adapter + image encoder |
| `scripts/fetch_faceid_weights.py` | huggingface.co (`immich-app/buffalo_l` — recognition/model.onnx mirror), h94/IP-Adapter-FaceID (files at **repo root** — `models/…` 404s) | `faceid/w600k_r50.onnx` (ArcFace 512-d), FaceID adapter `.bin` + LoRA |
| `backend/ai/colorization/ddcolor_colorizer.py` | GitHub release (opt-in via `PIXREVIVE_DDCOLOR_AUTODOWNLOAD=1`) | DDColor checkpoints |

Download discipline (`.part` temp + `os.replace`, strict manifest listing, size checks);
the fetcher's strict listing check caught two URL/name errors pre-extraction (measured
2026-09-27: buffalo_l ships `w600k_r50.onnx`, not "mbt"; FaceID files live at repo root).

## Browser Test Integration

- Playwright e2e suite (`e2e/`, `playwright.config.js`) **must not download browsers** —
  CDN driver downloads fail in this environment. Connects to an already-running Chrome via
  CDP on `127.0.0.1:9222` (`.agents/rules/browser_cdp.md`).

## Internal API Surface (frontend ↔ backend)

Frontend talks only to the local FastAPI server — same-origin via Vite proxy in dev:

- `POST /api/process` (sync, form: file + toggles incl. `reimagine`)
- `POST /api/process-stream` (SSE: `progress`/`complete`/`error` events; `complete`
  carries the full restoration report JSON)
- `GET /api/presets` — 6 presets incl. `reimagine_beta`
- `GET /api/process/{task_id}/evidence` — hybrid evidence-zone overlay PNG
  (green = original, magenta = re-imagined; "Show what's real" toggle in the UI)
- `GET /api/health` — readiness contract used by the frontend status pill
- `backend/routers/admin_router.py` — admin endpoints (token-enforced, tests in
  `tests/test_admin_router.py`)

## Cross-Component Contracts Worth Knowing

- **Report JSON** (`backend/ai/report/restoration_report.py`) is the frontend's data
  source for `RestorationReportCard.jsx` — additive fields only (latest:
  `reimagine.identity_conditioning ∈ {faceid, clip, none}`, surfaced 2026-09-27).
- **SSE event ordering**: worker thread → `call_soon_threadsafe` → generator; terminal
  events are `complete`/`error` only. Consumer must treat anything else as progress.
- **Health readiness** gates frontend "AI Fully Loaded" pill; lazy models (deblur, ddcolor)
  report `false` until first use — by design, not failure.
