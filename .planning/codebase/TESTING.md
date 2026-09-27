# Testing

**Analysis Date:** 2026-09-27 (full refresh; previous map 2026-09-19)

## Current State (measured)

**341 passed, 1 skipped, 105 subtests** (`PIXREVIVE_DEVICE=cpu`, full suite ~2.5 min).
342 test functions across 27 files in `tests/`. The single skip is the SFace
weights-unavailable guard when ONNX files are absent.

## How to Run (exact forms — the runtime rules apply)

```bash
# Full suite (backend must be PAUSED — see host quirk below)
PYTHONPATH=backend PIXREVIVE_DEVICE=cpu \
  "C:/Users/sangh/AppData/Roaming/Python/Python312/Scripts/pytest.exe" tests/ -q

# Targeted
... tests/test_reimagine_stage.py tests/test_reimaginer.py -q
```

**Host quirk (documented, not a code bug):** running the suite while the uvicorn server
holds torch weights causes intermittent access-violation segfaults in torch model init
(RAM/VRAM coexistence on the 7.3 GB host). Procedure: pause backend → run suite → restart.
Kill/restart: `taskkill //PID <pid> //F`, then `cd backend && nohup env PYTHONPATH=.
PIXREVIVE_DEVICE=cuda <python> -m uvicorn main:app --port 8000 &` (the launching shell
often times out at 10 s — poll `curl http://127.0.0.1:8000/api/health` until
`"ready":true` instead of trusting the shell).

## Test Layers

| Layer | Files | What they pin |
|---|---|---|
| Unit — tiering | `test_resource_profiler.py` (37) | tier floors, overrides, never-waive semantics |
| Unit — hybrid engine | `test_reimaginer.py` (25) | composite bytewise invariant, reroll ladder, conditioning priority, adapter two-way sync |
| Unit — identity | `test_face_identity_gate.py` (14), `test_faceid_embedder.py` (9+3 real-weights), `test_faceid_adapter.py` (12) | gate ladder + all-faces aggregation + env overrides; ArcFace embed geometry/perturbation stability; FaceID processor contracts + checkpoint geometry + dim-mismatch safety |
| Unit — stage | `test_reimagine_stage.py` (18) | coverage gate, VRAM bridge calls, evidence overlay persistence, error disclosure |
| Unit — report | `test_restoration_report.py` | disclosure blocks incl. colorization evidence flag, skip-reason classes |
| Unit — pipeline core | damage fixtures, regressions, router authority, stage registry structure, output verifier, quality engine, progress stream, admin/auth | pre-existing protections, all still green |
| E2E (manual/live) | `scripts/e2e_reimagine_case9.py`, `scripts/smoke_faceid.py`, `scripts/sweep_ip_scale.py`, `scripts/bench_reimagine.py` | real-pipeline runs through the API/fresh process; not part of the suite (GPU + RAM floors + minutes-long) |

## Conventions for New Tests

- Fixture images: `tests/restoration_dataset/damaged|pristine/` (case9 is the standing
  hybrid fixture — 60%-destroy synthetic mask construction in the E2E scripts).
- Synthetic drawn faces are **not** detectable by YuNet (trained on real faces) —
  real-face tests use real fixtures; a test asserts the synthetic is honestly undetectable.
- Real-weights tests skip cleanly when ONNX/bin files are absent — the suite must pass
  on a fresh checkout without weights.
- CPU-determinism: `PIXREVIVE_DEVICE=cpu` for the suite; generation tests use small
  step counts or mock the pipe; seeds are pinned via `task_id` hashing.

## Live Verification Discipline (beyond unit tests)

Every H-phase landed with a **live API verification** (SSE run through
`/api/process-stream` on a probe image) before commit; regressions found that way
(5 integration-audit bugs + 2 hunt fixes) were reproduced live first, then pinned with
unit tests. Keep that loop for any Stage-12/conditioning change.
