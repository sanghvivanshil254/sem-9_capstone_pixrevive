# experimental/orchestrator — Two-Tier PPCAV PhotoOrchestrator (ARCHIVED)

Archived 2026-09-25 per execution-plan decision gate **5.2 (A-1/D8/D19)**.

## What this is

An alternative restoration engine: a perception→manifest→action loop
(`orchestrator.py`, `actions/`, `perception/`, `transaction.py`,
`synthesis_manifest.py`, `loop_controller.py`) built in parallel to the
waterfall pipeline. It was **never the default** (`PIXREVIVE_USE_ORCHESTRATOR`
env hook, off by default) and never achieved benchmark parity, so it has been
removed from the active code path and parked here.

## Why it was parked (not deleted)

- The waterfall pipeline shipped all taxonomy stages (4B) and the Restoration
  Report (5.1) on top of the stage registry; the orchestrator never consumed
  them.
- Removing it from `backend/` cut a whole-tree dynamic-typing surface from the
  type discipline (plan 6.3) and removed the only consumers of
  `PipelineContext` transaction state (see below).
- **Reversible:** if it is ever revived, promotion requires parity evidence
  through the benchmark ruler (`scripts/benchmark_quality.py`) on cases 1–12
  plus its own test suite. Run it standalone from this directory; do not wire
  it back into `restore()` without that evidence.

## How to run (standalone)

The tree keeps its original package layout, so it is importable as
`ai.orchestrator` by temporarily adding the archive to the backend path — e.g.
run from a scratch checkout, or symlink/copy it back to `backend/ai/orchestrator`
locally (do not commit that):

```
cp -r experimental/orchestrator backend/ai/orchestrator
PYTHONPATH=backend "C:/Users/sangh/AppData/Roaming/Python/Python312/Scripts/pytest.exe" experimental/orchestrator/tests/
rm -rf backend/ai/orchestrator
```

Its tests are kept here for provenance; they are outside `tests/` and are not
collected by the project CI job.

## Removed from the live pipeline

- `restore()` / `process()` no longer accept `use_orchestrator` and never
  branch to `PhotoOrchestrator.run()`; the `PIXREVIVE_USE_ORCHESTRATOR` env
  hook is gone.
- `PipelineContext` no longer carries orchestrator-only state:
  `synthesis_manifest`, `hyperparameters`, `begin_transaction()`,
  `step_down_parameter()` — they moved with the experiment (the orchestrator's
  own copy of the context contract lives in its tests).
