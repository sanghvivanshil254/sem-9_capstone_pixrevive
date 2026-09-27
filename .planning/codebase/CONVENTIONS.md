# Conventions

**Analysis Date:** 2026-09-27 (full refresh; previous map 2026-09-19)

## Non-Negotiable Runtime Rules (enforced, `CLAUDE.md`/`AGENTS.md`)

1. Interpreter is **always** `C:/Program Files/Python312/python.exe`; pytest via
   `C:/Users/sangh/AppData/Roaming/Python/Python312/Scripts/pytest.exe`; backend code
   needs `PYTHONPATH=backend`. Never bare `python`/`pip`/`pytest`.
2. **No virtual environments** of any kind.
3. Model weights live in `backend/ai/weights/`, fetched by `urllib.request` direct URLs —
   never hub caches. Downloads use `.part` + `os.replace`.
4. Never `pip install` DDColor or IP-Adapter (would break pinned `basicsr==1.4.2` /
   duplicate diffusers trees) — vendor reimplementations instead
   (`backend/ai/colorization/ddcolor/`, `backend/ai/restoration/faceid_adapter.py`).

Enforced mechanically by the PreToolUse Bash hook (`.claude/settings.json` →
`scripts/hooks/guard_python_runtime.py`); self-test:
`python scripts/hooks/guard_python_runtime.py --self-test`.

## Error-Handling Pattern (the house style)

**Never raise into the pipeline.** Every stage/feature boundary catches broadly, logs,
and *discloses*:

- Stages return records, not exceptions: `{"status": "done"|"skipped"|"skipped_degraded",
  "reason": ..., ...}` (`reimaginer.py::reimagine`, the identity gate, output verifier).
- Silent failure is a bug: Stage-12 errors log a full traceback **and** surface as
  `status: "error"` in the report block (added after the integration audit found a
  swallowed exception).
- Degradation must be visible: conditioning downgrades set
  `_STATE.conditioning_used`, which overrides the *planned* `identity_conditioning` in
  the record — the report states what actually ran.
- "Cannot certify" ≠ "pass": the SFace gate returns `similarity: None` with a reason on
  any detection/ONNX failure, and the caller treats that as flag, never pass.

## Honesty & Disclosure Conventions (project-specific, strictly held)

- Measured claims carry dates and conditions in comments: "measured 2026-09-27",
  "8 E2E points", "0.76 GB VRAM". No folklore numbers.
- Negative results are documented, not hidden (design doc §11/§12.1: IP-scale sweep and
  FaceID-on-hopeless-coverage both measured and recorded).
- User-facing reports distinguish outcome classes ("no unrecoverable damage detected" vs
  "below the bar" vs "rolled back") — never a bare 0.0%.
- Evidence claims are proven, not asserted: bytewise composite assertions, evidence
  overlay PNG served for audit.

## Code Style

- Python: stdlib + numpy/cv2/torch idioms; type hints on public functions; docstrings
  cite the design-doc section and measured findings; `# noqa: BLE001` marks the
  intentional broad catches (always paired with a log-and-degrade body).
- Modules anchor paths to `os.path.dirname(os.path.abspath(__file__))`, **never CWD**
  (the uvicorn server runs with `cwd=backend/` — CWD-relative paths broke the gate and
  weights resolution in production, measured).
- Env-var feature flags follow `REIMAGINE_*` / `PIXREVIVE_*` naming, always with a
  safe default and a validation-with-warning fallback.
- Frontend: function components + hooks; report rendering reads `report.*` fields
  additively (new backend fields must not break old cards); Tailwind utility classes.

## Git & Artifacts

- `test_artifacts/` and `backend/temp_results/` are gitignored working areas; docs live
  in `docs/`; the GSD map in `.planning/codebase/` is committed.
- Commit style: imperative subject naming the phase + intent; footer
  `🤖 Generated with Codebuff / Co-Authored-By: Codebuff`.
