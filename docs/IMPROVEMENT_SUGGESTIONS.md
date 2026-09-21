# PixRevive — Improvement Suggestions

Prioritized by impact. Each item states the problem, the concrete change, and the
effort/risk. Cross-references: dead-code items are `D#` in `docs/DEAD_CODE_REMOVAL.md`.

---

## A. Correctness & trust (bugs lying in wait)

**U-1 · Model-load failure is reported as "ready".**
`initialise_models()` swallows per-model exceptions (`logger.warning`) and always sets
`_models_ready` in `finally`; `/api/health` then reports `ready: true` while LaMa, GFPGAN,
or RealESRGAN silently degraded to slow fallbacks (Telea, Lanczos). Change: track a
`loaded_models: set[str]` and report per-model booleans; expose a degraded-mode banner in
the UI. *Effort: S · Risk: none — pure observability.*

**U-2 · Health payload shape.**
`main.py` returns `"ddcolor"` duplicating `"colorizer"` and hard-codes `"fast_models"`.
Align keys with `health_status()` in the pipeline (defect_segmenter/lama/gfpgan/realesrgan/ddcolor)
and have `/api/health` delegate to it. *Effort: XS.*

**U-3 · `_stage_3_lama_inpaint` is a 200-line god-method.**
Face-zone circuit breaker, seam feathering, and eye/mouth protection logic are inline.
Extract `_seam_smooth()`, `_protect_facial_anatomy(mask, faces)`,
`_face_wipeout_circuit_breaker(...)` — each independently unit-testable. Currently only the
happy path is ever tested end-to-end. *Effort: M · Big testability win.*

**U-4 · Per-request `RestorationPipeline()` in hot paths.**
Fine for prod (singleton in `image_service`), but `scripts/` and several tests construct
fresh pipelines (≈30–60s model load each). Provide a `restore_offline()` module function or
a `--no-models` flag for geometry/mask-only tests. *Effort: S.*

**U-5 · Implement or delete `device_preference` (D12).**
The Form field exists in both endpoints and the UI device dropdown does nothing. Two options:
(a) honor it — pass to `FaceDetector`/`FaceRestorer`/`SuperResolution`/`DDColorColorizer.load()`
and reconstruct lazily when it changes; (b) drop it from the API. Recommend (a) — it's a
advertised feature users may rely on. *Effort: M.*

**U-6 · Stage-2b safe zone ignores detected faces' certainty.**
`_stage_2b_damage_cluster_smooth` blends a fixed center-rectangle (±25% of image) with face
bboxes; on group photos this protects empty background between faces and on close-ups the
center box covers the whole frame making the stage a no-op. Use the Stage-1b convex-hull
face mask (already computed for Stage 5) instead. *Effort: S.*

**U-7 · Progress percent mapping drift.**
`_emit` percentages are hard-coded per stage (3/10/12/20/30/50/65/78/85/90/95/98/100) and
skip from 30→50 when repair is off etc. The UI maps stages to nominal `%` — any reordering
silently desyncs. Keep the CLAUDE.md contract but add a `scripts/check_progress_stream.py`
assertion that emitted percentages are **strictly increasing in emission order** (it currently
checks monotonicity of the stream, not that the stage order matches ProgressBar's). *Effort: S.*

**U-8 · `restore()` ignores `enable_restoration` aliasing bugs.**
`process()` maps `enable_restoration` → `enable_face_restore` — three bool params
(`enable_restoration`, `enable_face_restore`) mean the same thing depending on entry point;
`image_router` passes `restore=`. Rename to one canonical `enable_face_restore` everywhere
(the frontend toggle is literally labeled restore/repair/denoise/colorize — good, but keep
the API field names in sync via Form aliases with a deprecation note). *Effort: S.*

**U-9 · In-memory task store is single-process, TTL-only-on-stream.**
`_prune_old_tasks()` runs only when a new `/process-stream` arrives; a `/process`-only
server never prunes, and `GET /status` for an expired task returns 404 without cleanup of
its file. Move pruning to a FastAPI startup/repeat timer (or prune inside `get_status`
too), and include the file delete when status is fetched post-TTL. *Effort: S.*

**U-10 · CORS: no production origin story.**
`main.py` default allowlist is localhost-only (good), but there's no documented env for
deployment; `PIXREVIVE_CORS_ORIGINS` exists — document it in a README ops section and add
a startup log of effective origins. *Effort: XS.*

---

## B. Restoration quality (image-engineering wins)

**Q-1 · Give the router real authority (fixes D16).**
`PipelineRouter.route()` computes 6 route presets that nothing consumes. Wire the flags:
`restore()` should consult `route["run_damage_repair"]` etc. ANDed with user `enable_*`
flags. This makes "clean image, skip LaMa+GFPGAN" a data-driven decision instead of the
ad-hoc `noise_score > 6.0` / damage checks sprinkled through `restore()`. This is the single
highest-leverage architecture fix that doesn't change output quality. *Effort: M.*

**Q-2 · Semantic material gating is half-built — finish it.**
`CODEBASE_MAP.md` still says segmentation is absent, but `DefectSegmenter` (U-Net) exists
and only gates damage masking. Use its segmentation class tokens (or a cheap color/hue
prior) to (a) gate Stage 5 NLM strength per material (skin/hair/sky need different h),
(b) prevent DDColor chroma diffusion from crossing hair/clothing boundaries
(Stage 6 uses only luminance-guided filtering today). *Effort: L · Big quality win on
portraits.*

**Q-3 · Colorization Rollback guard from the old ECCV16 design was lost.**
PROJECT_DOCUMENTATION documents a >88% single-hue saturation rollback that protected against
DDColor/ECCV16 hallucinating one dominant color — this guard doesn't exist in the DDColor
path. Re-add: compute HSV hue histogram post-colorization; if pathological, fall back to
L-channel preserved grayscale + log a warning. *Effort: S.*

**Q-4 · Seam smoothing (Stage 3) blurs the whole inpaint boundary including textured regions.**
The 9×9 Gaussian on the dilated boundary band softens real texture (fabric weave, brick).
Use `Harmonizer`'s guided filter (already imported at SR stage) at native resolution instead
of raw Gaussian, and restrict to non-face zones (face anatomy already protected — reuse the
helper from U-3). *Effort: S.*

**Q-5 · SR fallback upscales 4× unconditionally on failure.**
`_stage_9_super_resolution` except-branch always does Lanczos 4× even for a 3000px input
→ 12000px monster. Respect the 2K normalization target on the fallback path too. *Effort: XS.*

**Q-6 · Perceptual metric instrumentation.**
Add optional LPIPS/NIQE scoring (CPU-OK via `torch` already present) computed in
`QualityService` (currently dead, D6) and written into debug metadata `08_metadata.json` per
run. You can't tune the 10-stage parameter space blind — build the ruler first, then use
`tests/restoration_dataset/damaged/case1..6` as the regression benchmark. *Effort: M ·
Multiplies every future tuning decision.*

**Q-7 · Batch mode in the UI.**
Families restore albums, one photo at a time today. Backend task store already supports
arbitrary task ids; add `/api/process-batch` (multipart list) and a queue UI. GPU semaphore
already serializes inference. *Effort: L (backend) + M (UI).*

**Q-8 · Presets instead of 4 toggles.**
"Repair/Denoise/Colorize/Face/SR" booleans are power-user flags. Add named presets
(`Fast preview`, `Standard`, `Exhibition` = today's defaults, `Archive B&W`) that set the
toggles + tunable strengths server-side. Keeps flags for experts, gives normal users sane
choices. *Effort: S.*

---

## C. Architecture & code health

**A-1 · Resolve the two-engines problem (D8/D19).**
Running `orchestrator/` (PPCAV loop) vs the waterfall pipeline doubles maintenance. Decide:
- **Option 1 (recommended):** declare waterfall the production engine; move `orchestrator/`
  to `experimental/` with a README; keep tests green.
- **Option 2:** promote orchestrator behind a stable flag, delete the waterfall's
  unreachable branches (pipeline_router presets), and cut over once parity is proven on
  case1–6.
Either way, `PipelineContext` stops carrying orchestrator-only state (D19).

**A-2 · Single stage registry.**
Define stages once (`STAGE_REGISTRY: list[StageSpec]` with id/name/method/progress%/route
flags); generate the ProgressBar STAGES array, the docstring snapshot list, and the
`_emit` percentages from it. Kills D1/D14/I3 permanently and makes `check_progress_stream.py`
authoritative. *Effort: M · prevents an entire class of doc/UI drift.*

**A-3 · Split `restoration_pipeline.py` (1,349 lines).**
Move the 10 stage methods into `backend/ai/pipeline/stages/` modules grouped by concern
(`preprocess_stages.py`, `damage_stages.py`, `color_stages.py`, `finish_stages.py`);
`RestorationPipeline` keeps orchestration only. Pure mechanical refactor, tests keep passing.
*Effort: M.*

**A-4 · Type-check the AI layer.**
`pyrightconfig.json` exists at root but backend modules are largely untyped. Add
`from __future__ import annotations` + light typing to `pipeline/`, `damage/`, and enable
pyright basic mode in CI for the **app layer only** (routers/services/pipeline) — don't
boil the ocean on numpy-heavy math. *Effort: M incremental.*

**A-5 · Config dataclass instead of `image_metadata` stringly-typed dict.**
Metadata keys (`blur_score`, `damage_percentage`, `noise_score`, `color_type`, …) are
scattered magic strings. Introduce `@dataclass AnalysisReport` produced once by
QualityAnalyzer and passed read-only; keeps `image_metadata` for free-form extras. *Effort: M.*

**A-6 · Frontend: extract API client and kill hardcoded URLs.**
12 `localhost:8000` literals across `ImageUploader.jsx`, `Navbar.jsx`, `WeightsAdmin.jsx`
(see D-search). Create `frontend/src/api/client.js` reading `import.meta.env.VITE_API_BASE`
with dev fallback; fixes LAN/prod deployment. *Effort: S.*

**A-7 · Frontend: split ImageUploader (490 lines).**
Extract `useRestorationJob()` hook (SSE handling, abort, object-URL lifecycle) and
`ResultCard`/`OptionsPanel` components; the current component mixes transport, state, and
three UI concerns. *Effort: M.*

---

## D. Operations & safety

**O-1 · Zip-slip guard in `_looks_like_torch_checkpoint` path is good; extend to model
hot-swap of LaMa/DDColor.** Admin upload currently covers GFPGAN/RealESRGAN only; add
big-lama.pt and ddcolor with the same atomic staging + restart note. *Effort: S.*

**O-2 · Request concurrency guard is global.** `_gpu_semaphore(1)` serializes everything —
correct for 8GB VRAM, but small images could pass in parallel with a VRAM budget check
(`torch.cuda.mem_get_info`). Optional; document first. *Effort: M.*

**O-3 · Observability: structured per-stage timings to the client.**
`stage_timings` already exists in context — include it in the `complete` SSE event and
display "inpainting 8.2s · colorize 11.3s …" in the UI. Cheap and great for demos/capstone.
*Effort: S.*

**O-4 · CI is Playwright-only.** `.github/workflows/playwright.yml` runs e2e; add a
pytest job (Windows runner, Python 3.12, the absolute-path pytest) running
`tests/ -k "not slow"` on PRs. The regression suite is the project's real safety net.
*Effort: S.*

**O-5 · README.md at repo root doesn't exist.** The only run instructions live in stale
docs (PROJECT_DOCUMENTATION with venv instructions that violate repo rules). Write a real
README: architecture diagram, current 14-stage table, run commands (system Python 3.12, no
venv), weights download, test commands. *Effort: S.*

---

## Top-5 if you only do five things

1. **A-2 Stage registry** — kills the whole doc/UI/backend drift class.
2. **Q-1 Router authority** — makes routing real; simplifies `restore()`.
3. **U-1/U-2 honest health endpoint** — you lose hours to "ready but degraded" states.
4. **D21/D22 junkyard cleanup + O-5 README** — the repo currently hides its real design
   under 80 scratch files and stale docs.
5. **Q-6 perceptual metrics on the case1–6 benchmark** — everything after this becomes
   measurable engineering instead of eyeballing.
