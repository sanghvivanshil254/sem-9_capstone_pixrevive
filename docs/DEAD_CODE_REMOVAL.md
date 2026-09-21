# PixRevive — Dead Code Removal Report

**Date:** 2026-09-21 · **Method:** full-file audit of every Python/JSX module, import-graph
tracing (`rg` over all `.py`/`.jsx`), and cross-check against `CODEBASE_MAP.md`,
`PROJECT_DOCUMENTATION.md`, and the test suite.

Risk scale: 🟢 **zero-risk** (nothing imports it, nothing references it) ·
🟡 **safe with a small fix** (a test or doc references it — update that first) ·
🔴 **decision needed** (works but is redundant/abandoned — confirm before deleting).

---

## Part 1 — Documentation ghost stages (stale prose, not code)

The pipeline header docstring in `backend/ai/pipeline/restoration_pipeline.py` (lines 1–47)
lists snapshots for **Stage 5b Hair Zone Refinement** and **Stage 6b Skin Tone Enforcement**.
Grep confirms these stages **do not exist anywhere in the code** (`05_before_hair_refine`,
`06_before_skin_enforce` are never emitted). `PROJECT_DOCUMENTATION.md` is worse: it
documents an **18-stage ECCV16 + zeroscratches pipeline** that no longer exists —
`eccv16_colorizer.py`, `scratch_restorer.py`, `image_router.py`→`admin_router.py` module
names, the venv setup instructions, `ScratchRestorer`, `PipelineRouter` routes table
(BW_RESTORE with `run_damage_repair: False` — contradicted by current code), and PNG
metadata chunk encoding that `process()` does not do.

| # | Item | Location | Evidence | Risk |
|---|---|---|---|---|
| D1 | Stage 5b/6b docstring entries | `restoration_pipeline.py` header | stage methods absent | 🟢 |
| D2 | Entire ECCV16/zeroscratches/18-stage narrative | `PROJECT_DOCUMENTATION.md` | contradicts code (grep: zero refs to eccv16, scratch_restorer, zeroscratches) | 🟡 (tests/docs may quote it) |
| D3 | Baseline doc of the pre-v2 `ai_model/inference.py` system | `docs/current_pipeline.md` | that module was deleted long ago | 🟢 |
| D4 | "Semantic Segmentation: Currently Absent" claim | `CODEBASE_MAP.md` §3 | false since `DefectSegmenter` U-Net was added | 🟢 |

---

## Part 2 — Unused backend modules (zero import references)

| # | Item | Location | Evidence | Risk |
|---|---|---|---|---|
| D5 | **`HalftoneSuppressor`** (FFT notch filter) | `backend/ai/preprocessing/halftone_suppressor.py` | Grep for `HalftoneSuppressor\.` and `from ai.preprocessing.halftone_suppressor` across the whole repo: **0 call sites**. The SR module has its own separate `_suppress_halftone()` bilateral pre-filter, which is the only halftone code that runs. `CODEBASE_MAP.md` lists this module as an active preprocessor — wrong. | 🟢 (keep as future reference or delete; it is fully self-contained) |
| D6 | **`quality_service.py` / `QualityService`** | `backend/ai/analysis/quality_service.py` | Only the definition exists. No import in pipeline, routers, scripts, or tests. The "Stage 17 Quality Control / revert-to-original" behavior described in PROJECT_DOCUMENTATION.md **is not wired** — a doc-only feature. | 🟢 |
| D7 | **`IdentityValidator` never runs in production** | `backend/ai/validation/identity_validator.py` (517 lines) | Imported only by `validation/__init__.py` and 3 test files. No pipeline stage calls it. The ProgressBar UI still advertises an `identity_check` stage that can never fire (see D14). | 🟡 (tests depend on it; either wire it into the pipeline behind debug flag, or delete module + its 3 test files) |
| D8 | **`orchestrator/` package (26 files) is dormant** | `backend/ai/orchestrator/**` | Only reachable via `PIXREVIVE_USE_ORCHESTRATOR=1` env var or `use_orchestrator=True` param — never set by any router or test default. Tests `test_orchestrator_phase1–4.py` exercise it directly, so it's tested but **unreachable from the web app**. Two parallel restoration engines are maintained for zero user-facing benefit. | 🔴 decide: promote it (replace waterfall), or quarantine it (move to `experimental/`) |
| D9 | `realesr-general-x4v3.pth` download entry | `backend/ai/download_weights.py` DOWNLOADS list | Grep for `realesr-general` in `backend/ai/restoration/`: 0 uses. ~5 MB dead weight in the manifest. | 🟢 |
| D10 | `zeroscratches` dependency | `backend/requirements.txt` (optional section) + `PROJECT_DOCUMENTATION.md` §6 | The scratch_restorer module was deleted; `ScratchRestorer` appears nowhere. Startup flag `PIXREVIVE_LOAD_SCRATCH_RESTORER` is mentioned in requirements comment but grep finds no consumer. | 🟢 remove from requirements |
| D11 | `torchaudio` in docs' package list | `PROJECT_DOCUMENTATION.md` §4 | Not in actual `backend/requirements.txt`; doc-only. | 🟢 |

---

## Part 3 — Dead parameters & vestigial code paths

| # | Item | Location | Evidence | Risk |
|---|---|---|---|---|
| D12 | **`device_preference` parameter is ignored** | `restoration_pipeline.py:1307` ← `image_service.py` ← both router endpoints | The parameter is threaded through 3 layers and **never read** in `process()`/`restore()` — device is chosen solely by `torch.cuda.is_available()` in `__init__`. Users' "auto/cpu/cuda" Form field in `/api/process` does nothing. | 🟢 remove param + Form field, or implement device override |
| D13 | **`_realesrgan_tile_forward` imported but unused in pipeline** | `restoration_pipeline.py:70` | Pipeline delegates to `SuperResolution.restore()`; only `admin_router.py:812` (inside dead benchmark code, see D15) and `scripts/probe_sr_variants.py` use the symbol. | 🟢 drop from the import |
| D14 | **UI stages that never fire** | `frontend/src/components/ProgressBar.jsx` STAGES array | `smart_denoise`, `upscale`, `harmonize`, `enhance`, `final_refine`, `identity_check`, `encode` — grep of backend `_emit(...)`/`on_progress(...)` calls: none of these ids are emitted by the current pipeline (harmonize logs a string but never emits progress). Dead dot entries mislead users; harmless but stale. Also `refine_mask`, `color_refine` — check `check_progress_stream.py` when touching. | 🟢 prune after re-running `scripts/check_progress_stream.py` |
| D15 | **Training benchmark: unreachable + mocked metrics** | `admin_router.py` `async_run_training_pipeline` (lines ~708–900), `/training/start`, `/training/status` | Function body sets status "failed / not available in this build" then `return`, leaving ~150 lines of unreachable code (marked `# Unreachable — placeholder below kept for reference`) that fabricates "accuracy" numbers. Frontend `WeightsAdmin.jsx` renders loss charts from these fake epochs. | 🟢 delete the unreachable block + h5py comment; decide fate of the endpoint pair |
| D16 | Route dict flags computed but never consulted | `pipeline_router.py` `_ROUTES` (`run_damage_repair` etc.) | `route()` stores `meta["pipeline_route"]` and the pipeline logs it, but no stage reads `run_*` flags — gating is done from raw metadata + user `enable_*` flags. The 6 presets are advisory logging only. | 🟡 either wire flags into `restore()` stage gating (making the router real) or reduce `_ROUTES` to a name string |
| D17 | `Enhancer` class + `Harmonizer.enhance()`-style helpers | `backend/ai/postprocessing/enhancer.py` (class at line 82) | Only `Harmonizer` is imported by the pipeline (`restoration_pipeline.py:71`). `Enhancer.enhance()`, `texture_restore`, `gentle_contrast`, `adaptive_sharpen`, `correct_color_cast` have no production call sites (CLAUHE/sharpening now live in `_stage_10b_projector_grade` / `_stage_10_post_sr_sharpen`). | 🟡 delete class or rewire as configurable finisher; update `postprocessing/__init__.py` |
| D18 | `WaveletDenoiser`, `AdaptiveDenoiser`, `SmartDenoiser`, `ArtifactDetector` | `backend/ai/postprocessing/denoiser.py` | Production denoising is Stage 0/2b/5 (impulse/speck/NLM) + NoiseRouter. The wavelet/smart stack is used **only by tests** (`test_restoration_regressions.py`). | 🔴 decide: remove + drop tests, or keep as alternative denoiser for NoiseRouter dispatch |
| D19 | `SynthesisManifest`/`CanvasTransaction`/`step_down_parameter` on every PipelineContext | `pipeline_context.py:8-9, 44-53, 78-96` | Only the dormant orchestrator consumes these; every standard request allocates them for nothing. | 🔴 tied to D8 decision |
| D20 | Legacy root-level `__pycache__` + `backend/ai/__pycache__`-style dirs, `debug_output/`, `test_artifacts/`, `test-results/`, `backend/temp_results/` build outputs | repo root | Check `.gitignore` coverage; several artifacts were committed at least once (the find shows them tracked). | 🟢 gitignore + `git rm -r --cached` |

---

## Part 4 — Junkyard directories (biggest win)

| # | Item | Location | Evidence | Risk |
|---|---|---|---|---|
| D21 | **~80 one-off diagnostic/scratch scripts** | `test_artifacts/*.py` (`scratch_*`, `test_*`, `probe_*`, `diagnose_*`, `run_*` …) | Every file is a historical experiment around case1–case6 debugging; none imported by the app; many reference deleted modules/stage layouts. They also import pipeline internals (`_stage_0_despeckle` etc.) so they rot silently. | 🟢 move the 2–3 still-valuable ones (`benchmark_runner.py`, `full_dataset_sanity_sweep.py`) into `scripts/diagnostics/`, delete the rest |
| D22 | Committed run artifacts (logs, metadata, timings) | `test_artifacts/**/06_timings.txt` etc., `debug_output/*` | Outputs, not source. | 🟢 delete from git, gitignore |
| D23 | `backend/tests/run_pipeline_tests.py` — dead harness | `backend/tests/` | Manual runner requiring a `test_data/input` folder that does not exist; duplicates `tests/` pytest suite; imports `device_preference` dead param. Also note **two different `tests/` dirs exist** (`tests/` = real pytest suite, `backend/tests/` = this + `metrics.py`). | 🟢 delete runner; keep `metrics.py` by relocating under `tests/` |
| D24 | `docs/tasks/` (empty `progress.txt`, PRD stub) + `.planning/codebase/*` snapshots | docs/, .planning/ | Empty or one-time analysis snapshots. `.planning/CONCERNS.md` has useful history — merge its resolved-bug notes into docs, then delete. | 🟢 |
| D25 | `frontend/src/pages/Documentation.jsx` accuracy | frontend | In-app docs likely mirror the stale PROJECT_DOCUMENTATION claims (verify text when touching). | 🟡 refresh content |

---

## Part 5 — Inconsistencies (not dead, but lying to the reader)

| # | Item | Detail |
|---|---|---|
| I1 | `CODEBASE_MAP.md` vs reality | Map omits the whole `orchestrator/` package, `admin_router.py`, `ImpulseDenoiser`/`SpeckSuppressor`, `ImageNormalizer.detect_and_crop_borders`, and lists a nonexistent `frontend/src/services/` and components that don't exist (`ComparisonSlider.jsx`, `SettingsPanel.jsx` — actual UI is a single `ImageUploader.jsx` with inline comparison/settings). |
| I2 | `main.py` health payload | `"ddcolor": colorizer_ready` duplicates `"colorizer"`; `"ready": fast_models_ready` returns true even when LaMa/GFPGAN/SR individually failed to load (loader threads swallow exceptions and set the event in `finally`). See IMPROVEMENT_SUGGESTIONS.md U-2. |
| I3 | Stage numbering chaos | Pipeline uses 0, 1, 1b, 2, 2b, 3, 3b, 4, 5, 4b(in log strings), 6, 6c, 7, 8, 9, 9b, 9c, 10, 10b; `restoration_pipeline.py` docstrings call itself "14-stage", PROJECT_DOCUMENTATION says 18. The router uses named routes. Pick one scheme. |
| I4 | `image_service.py` comment drift | Comments reference "Fix 11/12/13" with no defect register anywhere. |

---

## Recommended removal order (mechanical checklist)

1. **Snapshot branch first** (no git repo is currently initialized in this checkout — `git init` and commit before any deletion, so every removal is revertible).
2. Gitignore + untrack artifacts: `debug_output/`, `test_artifacts/`, `test-results/`, `backend/temp_results/`, `__pycache__/` (D20, D22).
3. Delete `test_artifacts/*.py` after salvaging `benchmark_runner.py`, `full_dataset_sanity_sweep.py` into `scripts/diagnostics/` (D21).
4. Delete `backend/tests/run_pipeline_tests.py`; move `metrics.py` → `tests/metrics.py` (D23).
5. Prune `download_weights.py` realesr-general entry; remove `zeroscratches` from requirements (D9, D10).
6. Delete the unreachable benchmark body in `admin_router.py` (D15) — keep endpoints only if the training UI story is still wanted.
7. Remove `device_preference` plumbing end-to-end **or** implement it (D12 — implementing is the better fix, see IMPROVEMENTS U-5).
8. Fix stale docs: `restoration_pipeline.py` header (D1), rewrite `PROJECT_DOCUMENTATION.md` against current code (D2), delete `docs/current_pipeline.md` (D3), correct `CODEBASE_MAP.md` (D4/I1).
9. Decision gates (schedule as a team decision, not silent deletion): orchestrator (D8/D19), router flags (D16), Enhancer class (D17), wavelet denoiser stack (D18), IdentityValidator (D7).
10. Re-run the full pytest suite + `scripts/check_progress_stream.py` after each numbered step.
