# Codebase Structure

**Analysis Date:** 2026-09-27 (full refresh; previous map 2026-09-19)

## Repository Layout (verified by directory listing)

```
PixRevive/
├── backend/                    # FastAPI + 16-stage pipeline (73 .py files, 67 under ai/)
│   ├── main.py                 # app factory, CORS, static mounts, health
│   ├── requirements.txt        # pins incl. diffusers==0.40.0, accelerate, psutil
│   ├── routers/                # image_router (7 routes), admin_router (12 routes)
│   ├── services/               # image_service.py — task orchestration layer
│   ├── ai/
│   │   ├── pipeline/           # restoration_pipeline.py, pipeline_context.py,
│   │   │                       #   stage_registry.py (StageSpec list = contract),
│   │   │                       #   stages/{prep,color,damage,finish}_stages.py, debug_logger.py
│   │   ├── restoration/        # lama_restorer, super_resolution, face_restorer, deblurrer,
│   │   │                       #   inpainting_engine,
│   │   │                       #   reimaginer.py (H1 engine),
│   │   │                       #   resource_profiler.py (tiering),
│   │   │                       #   faceid_adapter.py (H4 conditioning)
│   │   ├── validation/         # face_identity_gate.py (SFace),
│   │   │                       #   faceid_embedder.py (H4 ArcFace),
│   │   │                       #   output_verifier.py, forensic_validator.py,
│   │   │                       #   identity_validator.py
│   │   ├── analysis/           # face_detector.py (RetinaFace, lazy-reload), quality engine
│   │   ├── damage/             # enhanced_detector.py (taxonomy detectors), defect_segmenter
│   │   ├── colorization/       # ddcolor_colorizer.py + vendored ddcolor/
│   │   ├── report/             # restoration_report.py — report JSON builder
│   │   └── weights/            # ALL model weights (gitignored artifacts, fetched by scripts)
│   ├── temp_results/           # per-task result PNGs + evidence overlays (swept)
│   └── debug_output/           # optional per-run stage debug artifacts
├── frontend/
│   ├── src/
│   │   ├── App.jsx, main.jsx
│   │   ├── components/         # ImageUploader, OptionsPanel (reimagine checkbox),
│   │   │                       #   ResultCard ("Show what's real" overlay toggle),
│   │   │                       #   RestorationReportCard (hybrid card + disclosures)
│   │   └── hooks/useRestorationJob.js   # job state machine, SSE consumer, presets
│   └── vite.config.js
├── scripts/                    # fetch_reimagine_weights.py, fetch_faceid_weights.py,
│                               #   bench_reimagine.py, e2e_reimagine_case9.py,
│                               #   sweep_ip_scale.py, smoke_faceid.py, hooks/
├── tests/                      # 27 test files, 342 test functions (341 pass + 1 weights-skip)
│   └── restoration_dataset/    # damaged/ + pristine/ fixtures incl. case9_teared_sepia.jpg
├── docs/                       # HYBRID_DIFFUSION_DESIGN.md (v2 + §11/§12 measured addenda),
│                               #   EXECUTION_PLAN.md, DAMAGE_TAXONOMY_MAPPING.md, ...
├── e2e/                        # Playwright suite (CDP-attached Chrome, no downloads)
├── .planning/codebase/         # THIS MAP (7 docs)
├── CODEBASE_MAP.md             # legacy navigation map (still maintained)
└── CLAUDE.md / AGENTS.md       # runtime rules (enforced by PreToolUse hook)
```

## Navigation Shortcuts (fastest routes into the code)

- "Where does X happen in the pipeline?" → `backend/ai/pipeline/stage_registry.py`
  (id → method name → pct), then `backend/ai/pipeline/stages/`.
- "How does the hybrid beta work?" → `finish_stages.py::_stage_12_reimagine` →
  `backend/ai/restoration/reimaginer.py::reimagine`.
- "Where are identity decisions made?" → `backend/ai/validation/face_identity_gate.py`
  (verdicts) + `faceid_embedder.py` (embeddings) + `reimaginer.py` (conditioning priority).
- "Where does the UI get its numbers?" → `backend/ai/report/restoration_report.py`
  (field names = frontend contract).
- "Why is X slow/refused?" → `resource_profiler.py` floors + `uvicorn*.log` in
  `test_artifacts/`.

## Changed-Since-Baseline Modules (H1–H4)

New: `reimaginer.py`, `resource_profiler.py`, `face_identity_gate.py`,
`faceid_embedder.py`, `faceid_adapter.py`, `restoration_report.py` (reimagine block),
`smoke_faceid.py`, `sweep_ip_scale.py`, both fetchers, 5 new test files.
Heavily modified: `finish_stages.py` (Stage 12), `stage_registry.py`, `restoration_pipeline.py`
(flags), `image_router.py`/`image_service.py` (reimagine + evidence + task_id threading),
frontend `OptionsPanel/ResultCard/RestorationReportCard/useRestorationJob`.
