# PixRevive — Whole-Pipeline Robustness Audit Report

**Date:** 2026-09-25 · **Scope:** every layer of the restoration system ·
**Verdict: PASS — 98/98 audit checks, 213/213 pytest, live U-7 contract PASS**

Harness: `scripts/robustness_audit.py` (kept in-repo, re-runnable any time).
Machine-readable results: `test_artifacts/robustness_audit.json`.

```bash
PYTHONPATH=backend "C:/Program Files/Python312/python.exe" scripts/robustness_audit.py            # incl. GPU
PYTHONPATH=backend PIXREVIVE_DEVICE=cpu "C:/Program Files/Python312/python.exe" scripts/robustness_audit.py --skip-gpu
```

The audit found **5 real defects** (all fixed and regression-covered) plus
2 documented-but-safe behaviors. Details per layer below.

---

## Layer 1 — Static hygiene

| Check | Result |
|---|---|
| `py_compile` over **every** `.py` in `backend/`, `scripts/`, `tests/`, `experimental/` (697 files) | PASS — zero syntax errors anywhere, including the archived orchestrator |

## Layer 2 — Synthetic input battery (14 pathological images)

Every image went through the **full** `restore()` path on CPU:

| Input | Result |
|---|---|
| solid black / white / 128-gray fields | PASS |
| 1×1, 3×3, 8×8 tiny images | PASS (upscale path clamps gracefully) |
| 16×16 gradient tile | PASS |
| 100×75 small photo, 121×99 odd dimensions | PASS |
| 300×2 / 2×300 extreme strips | PASS |
| heavy Gaussian grain field (260×300) | PASS |
| **2-D grayscale array** | **BUG #1 — fixed** (was a raw OpenCV "bad number of channels" crash) |
| **float64 array with NaNs** | **BUG #2 — fixed** (was a raw OpenCV depth crash) |
| 4-channel RGBA | PASS |
| PNG bytes / PIL Image / ndarray / `None` | PASS — polymorphic inputs OK; `None` rejected with a clean `ValueError` |

**Fix:** `restore()` now normalizes raw-array inputs exactly like the HTTP path
does (GRAY→RGB, RGBA→RGB, `nan_to_num` + clip to uint8, explicit shape error).

## Layer 3 — Toggle matrix (2 base images × all 32 toggle combinations = 64 runs)

All 64 combinations of `enable_{repair,denoise,colorization,face_restore,sr}`
completed with sane RGB uint8 output — no combination deadlocks, crashes, or
produces an invalid frame.

## Layer 4 — API surface, SSE contract, honesty, concurrency

| Check | Result |
|---|---|
| `/api/health`, `/api/presets` | PASS |
| Bad MIME upload → 400 | PASS |
| 26 MB oversize upload → 400 | PASS |
| Truncated PNG → SSE `error` event (never hangs) | PASS |
| Unknown task id → 404 | PASS |
| Admin upload with invalid model type → 400 | PASS |
| **SSE strict-JSON parse** of every `data:` line | PASS — zero malformed events |
| Every stage id known to the registry & frontend (`ProgressBar.jsx`) | PASS |
| **Monotonic progress percent** | **BUG #3 — fixed** (was 12 → 11, see below) |
| `complete` event carries `task_id` + `restoration_report`; result is a real PNG | PASS |
| **Report honesty:** torn image discloses inpainting/re-synthesis; clean gradient discloses none; all keys present | PASS |
| **Concurrency:** 8 simultaneous `/api/process` requests | PASS — 8/8 images, serialized by the GPU semaphore, 6.2 s total |

**Bug #3 detail:** the registry listed `cast_fix` (11 %) before `face_detect`
(12 %) but the runtime runs face detection first — emitted progress went
**12 → 11**, breaking the ProgressBar monotonicity contract on every color
photo. Fixed in the registry (truthful order + percentages), the inline
`on_progress("cast_fix", 12, …)` literal, and the regenerated
`frontend/src/generated/stages.json`; a registry-order test now pins it.

## Layer 5 — GPU spot check (GTX 1650 4 GB)

| Check | Result |
|---|---|
| case10 (spiderweb creases, 478×641) end-to-end on CUDA | PASS — 1527×2048 out, **55.2 s**, **peak 3569 MB** VRAM |

Peak VRAM sits ~430 MB under the card limit — the closest-to-the-edge metric in
the system, and precisely why the `_gpu_semaphore(1)` serialization (documented
under O-2) is mandatory rather than optional.

## Layer 6 — Live-server contract (U-7) + full suite

| Check | Result |
|---|---|
| `scripts/check_progress_stream.py` against a live uvicorn backend | **PASS** — 17/17 events monotonic, all ids known to the frontend, stream terminates with `complete`, 3194 KB PNG downloadable, end-to-end 135 s (CPU device) |
| Full pytest suite after all fixes | **213 passed + 105 subtests, 0 failures** (archived orchestrator tests excluded from collection by design) |
| Pyright basic (backend core) | **0 errors** |

## Defects found & fixed (summary)

| # | Severity | Defect | Fix |
|---|---|---|---|
| 1 | **High** | Degraded-mode stall: with `load_models=False` (tests, audits, offline tools) the colorization stage waited the full **60 s** `Event` timeout per run before degrading | Readiness events resolve immediately when no loader will ever run; 60.8 s → **0.75 s** (80×) |
| 2 | High (library API) | 2-D grayscale arrays crashed `restore()` with a raw OpenCV error | Normalized to RGB at the boundary |
| 3 | High (library API) | float/NaN arrays crashed `restore()` with a raw OpenCV depth error | `nan_to_num` + clip to uint8 |
| 4 | Medium | Progress percent went **12 → 11** (registry/runtime order mismatch) — visible as a backwards progress bar on every color photo | Registry now tells the truth; pcts fixed; export regenerated; test pins the order |
| 5 | Low | Degenerate-input noise detection: a perfectly flat scanned figure fired **7 spurious noise families** (ratio detectors divide by ~zero) | Texture-variance guard in `NoiseClassifier` (found in the Phase 6 pre-work, regression-tested) |

## Documented-but-safe behaviors (not defects)

- **Case9/case12 colorization rollback** — the Q-3 hallucination guard detects
  DDColor's single-hue wash and rolls back to grayscale *by design*; the
  Restoration Report discloses it. Improving DDColor's output on those cases is
  a model-quality project, not a robustness issue.
- **4:3 pane letterboxing in the UI** — cosmetic, noted for the frontend polish
  backlog.

## Reproduction & artifacts

- Audit harness: `scripts/robustness_audit.py` — self-contained, exception-isolated
  per check, writes `test_artifacts/robustness_audit.json`, exit code 0 only on
  a fully green run.
- Live-server log: `test_artifacts/uvicorn_audit.log`.
- The five fixes are in `backend/ai/pipeline/restoration_pipeline.py`,
  `backend/ai/pipeline/stage_registry.py`, `backend/ai/pipeline/restoration_pipeline.py`
  (cast_fix pct), `frontend/src/generated/stages.json` (regenerated), and
  `tests/test_banding_and_cast.py` (order pin).
