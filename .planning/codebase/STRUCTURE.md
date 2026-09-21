# Codebase Structure

**Analysis Date:** 2026-09-19

## Directory Layout

```
PixRevive/
├── backend/                  # FastAPI backend and core AI pipeline
│   ├── ai/                   # Vision and deep learning modules
│   │   ├── analysis/         # Quality metrics, face and landmark detection
│   │   ├── colorization/     # DDColor colorizer & chrominance refinement
│   │   ├── damage/           # Crack, scratch, tear, and flake detectors
│   │   ├── models/           # Base model interface definitions
│   │   ├── pipeline/         # 14-stage restoration orchestrator & context
│   │   ├── postprocessing/   # 3-zone NLM denoising, sharpening, exhibition grade
│   │   ├── preprocessing/    # Normalization, impulse & speckle suppressors
│   │   ├── restoration/      # LaMa inpainting, GFPGAN, Real-ESRGAN
│   │   ├── validation/       # SSIM, forensic, and identity preservation validators
│   │   └── weights/          # Deep learning model weight checkpoints (.pth, .pt)
│   ├── routers/              # FastAPI endpoint routers (image, admin)
│   ├── services/             # Application orchestration service singletons
│   ├── temp_results/         # Generated PNG results (auto-pruned by TTL)
│   ├── training_data/        # Local training datasets (bw and color)
│   ├── main.py               # FastAPI application entry point & CORS
│   └── requirements.txt      # Python runtime dependencies
├── frontend/                 # React 19 + Vite + Tailwind CSS v4 frontend
│   ├── src/                  # React source code
│   │   ├── components/       # Reusable UI widgets (ProgressBar, ImageUploader)
│   │   ├── pages/            # Application pages (Home, WeightsAdmin, Documentation)
│   │   ├── App.jsx           # Root layout and client routes
│   │   └── main.jsx          # Client entry point
│   ├── package.json          # Frontend dependencies and scripts
│   └── vite.config.js        # Vite bundler configuration
├── tests/                    # Pytest test suite
│   ├── restoration_dataset/  # Standard test fixtures (damaged case1 to case6)
│   ├── test_admin_router.py
│   ├── test_color_refiner.py
│   ├── test_compat.py
│   ├── test_facial_landmarks.py
│   ├── test_forensic_restoration.py
│   ├── test_identity_preservation.py
│   ├── test_image_router.py
│   ├── test_progress_stream.py
│   └── test_restoration_regressions.py
├── e2e/                      # Playwright end-to-end browser test specifications
│   └── pixrevive.spec.js     # User interaction and pipeline flow tests
├── test_artifacts/           # Diagnostic test scripts and comparison images
├── debug_output/             # Diagnostic stage PNG snapshots and execution logs
├── docs/                     # Documentation and architecture diagrams
├── scripts/                  # Maintenance, verification, and runtime guard scripts
├── .agents/                  # Workspace agent rules and runtime skills
├── .claude/                  # Claude Code settings, hooks, and launch configuration
├── .planning/                # GSD project plans, roadmaps, and codebase maps
│   └── codebase/             # Seven structured codebase reference documents
├── CLAUDE.md                 # Canonical instructions for AI agents
├── AGENTS.md                 # Agent instructions for Codex/alternative tools
├── GEMINI.md                 # Agent instructions for Gemini CLI
├── CODEBASE_MAP.md           # Quick codebase navigation map (`map` shorthand)
├── pixrevive_orchestrator_plan.md # v2.0 Two-Tier PPCAV Orchestrator Specification
├── package.json              # Root Playwright dev dependencies and scripts
├── playwright.config.js      # Playwright browser runner configuration
└── pyrightconfig.json        # Pyright static analysis configuration
```

## Directory Purposes & Key Locations

### Backend Core (`backend/ai/`)
- `preprocessing/`: Initial signal cleanup before neural models.
  - `image_normalizer.py`: Aspect ratio, bit-depth normalization, and scanner canvas margin auto-cropping.
  - `impulse_denoiser.py`: Rank-Ordered Absolute Differences (ROAD) filter for dust/pinhole outliers.
  - `speck_suppressor.py`: Spatial autocorrelation speck-field suppressor.
- `damage/`: Multi-scale defect detection and mask refinement.
  - `enhanced_detector.py`: Detects scratches, wide cracks, 3D paper tears, and white flakes with boundary Sobel verification.
  - `mask_refiner.py`: Binary morphology fusion, border artifact trimming, and ocular/oral safe-zone exclusion.
  - `defect_segmenter.py`: U-Net semantic crack segmentation.
- `restoration/`: Generative deep learning repair engines.
  - `inpainting_engine.py` & `lama_restorer.py`: TorchScript LaMa deep Fourier convolution inpainting with edge-energy circuit breaker.
  - `face_restorer.py`: GFPGAN v1.4 multi-face restoration with anatomical `chin_fade` mask to protect submental jawline shadow.
  - `super_resolution.py`: Real-ESRGAN x4plus tiled inference with dynamic VRAM tile sizing (512/256/192px).
- `colorization/`: Chrominance inference and refinement.
  - `ddcolor_colorizer.py`: ICCV 2023 DDColor global transform with ambient infill diffusion.
  - `color_refiner.py`: `BandingReducer`, `SaturationBalancer`, and `EdgeAwareColorBleedFixer`.
- `validation/`: Quantitative restoration auditing.
  - `forensic_validator.py`: Hard constraints auditor (line warping, face symmetry, texture fidelity).
  - `identity_validator.py`: SSIM and landmark displacement validator.

### Web & API (`backend/routers/`)
- `routers/image_router.py`: Handles `/api/process` and SSE `/api/process-stream` endpoints, guarded by a GPU concurrency semaphore.
- `routers/admin_router.py`: Weight management and training data upload with secure zip extraction.

### Frontend (`frontend/src/`)
- `components/ComparisonSlider.jsx`: Interactive before/after split slider.
- `components/ProgressBar.jsx`: Monotonic stage progress visualizer.
- `components/SettingsPanel.jsx`: Pipeline toggle configuration.
- `pages/Home.jsx`: Main application landing and processing workspace.
- `pages/WeightsAdmin.jsx`: Administrative dashboard for neural weight assets.
