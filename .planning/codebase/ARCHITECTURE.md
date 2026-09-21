<!-- refreshed: 2026-09-19 -->
# Architecture

**Analysis Date:** 2026-09-19

## System Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Presentation Layer (Vite + React 19)               │
├──────────────────────────┬──────────────────────────┬───────────────────────┤
│    `ImageUploader.jsx`   │  `ComparisonSlider.jsx`  │  `ProgressBar.jsx`    │
│  (Upload & User Controls)│ (Before / After Viewer)  │  (Monotonic SSE Dot)  │
└────────────┬─────────────┴────────────┬─────────────┴───────────┬───────────┘
             │                          │                         │
             ▼                          ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   API & Transport Layer (FastAPI / Uvicorn)                 │
│  - `backend/main.py` (CORS allowlist, static mounts, health checks)         │
│  - `backend/routers/image_router.py` (/api/process, /api/process-stream)    │
│  - `backend/routers/admin_router.py` (/api/admin/weights, /training_data)   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               Service Orchestrator & Context (Singleton)                    │
│  - `backend/services/image_service.py` (`ImageService` wrapper)             │
│  - `backend/ai/pipeline/pipeline_context.py` (`PipelineContext` state)      │
│  - Transitioning to Two-Tier PPCAV Closed-Loop Orchestrator (`v2.0 Plan`)    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            Exhibition-Grade 14-Stage Restoration Pipeline                   │
│             (`backend/ai/pipeline/restoration_pipeline.py`)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Stage -1: Framing Margin Detection & Auto-Cropping (`ImageNormalizer`)      │
│ Stage 0: Impulse & Speckle Suppression (`ImpulseDenoiser`, `SpeckSuppressor`)│
│ Stage 1: Degradation & Color Analysis (`QualityAnalyzer`)                   │
│ Stage 1b/4: Early RetinaFace Face & 5-Landmark Detection (`FaceDetector`)   │
│ Stage 2: Multi-Scale Fracture Masking (`EnhancedDetector`, `MaskRefiner`)   │
│          - Boundary Sobel gradient verification rejects specular highlights │
│ Stage 3: LaMa Deep Inpainting & Seam Smoothing (`InpaintingEngine`)         │
│ Stage 2b: High-Variance Damage Cluster Smoothing                            │
│ Stage 3b: Post-Inpaint Face Re-Detection                                    │
│ Stage 5: Spatially-Gated 3-Zone Denoising (Face / Mid / Background NLM)      │
│          - Noise-floor edge gating & variance protection preserves texture  │
│ Stage 6: DDColor Global Colorization & Chrominance Infill Diffusion         │
│ Stage 6c: Color Artifact Reduction (Banding, Saturation, Bleed Fixers)      │
│          - EdgeAwareColorBleedFixer guided by luminance boundary gradients  │
│ Stage 7: GFPGAN Damage-Aware Face Restoration                               │
│          - Anatomical chin_fade mask protects submental jawline shadow      │
│ Stage 8: Pre-SR Bilateral Anti-Aliasing (L* and chroma filtering)           │
│ Stage 9: Real-ESRGAN x4plus Super-Resolution & Inpaint Seam Harmonizer      │
│ Stage 9c: Post-SR Deferred Face Restoration (for small faces <90px)         │
│ Stage 10: Contrast-Adaptive LAB Luminance Sharpening                        │
│ Stage 10b: Projector Exhibition Grade (CLAHE clip 1.6 + Shadow S-Curve)     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Storage & Neural Weight Assets                            │
│  - Model weights: `backend/ai/weights/` (LaMa, GFPGAN, RealESRGAN, DDColor) │
│  - Temp outputs: `backend/temp_results/` (Task TTL 30-min auto-pruned)      │
│  - Diagnostics: `debug_output/` (Sequential numbered stage snapshots)       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Architectural Layers

### 1. Presentation Layer (Frontend)
- Built with React 19 and Vite 8, utilizing Tailwind CSS v4.
- Provides interactive drag-and-drop image upload, before/after split comparisons (`ComparisonSlider.jsx`), and real-time Server-Sent Events progress visualization (`ProgressBar.jsx`).
- Settings panel exposes user toggles: `enable_repair`, `enable_denoise`, `enable_colorization`, `enable_face_restore`, and `enable_sr`.

### 2. API & Transport Layer
- FastAPI asynchronous framework running on Uvicorn.
- Endpoints:
  - `POST /api/process`: Asynchronous task submission returning `task_id`.
  - `GET /api/process/{task_id}/status`: Polling status endpoint.
  - `GET /api/process/{task_id}/result`: PNG artifact retrieval.
  - `POST /api/process-stream`: High-throughput Server-Sent Events (SSE) streaming monotonic progress percentage and status text directly to the UI.
  - `GET /api/health`: System health and model availability check.
  - `POST /api/admin/weights/upload`: Chunked background upload for model weights.

### 3. Pipeline Core & State Orchestration
- `PipelineContext` (`backend/ai/pipeline/pipeline_context.py`): Central state container passed across all stages. Retains `original_image`, mutable `working_image`, `damage_masks`, `image_metadata`, and stage execution logs.
- `RestorationPipeline` (`backend/ai/pipeline/restoration_pipeline.py`): The core 1,309-line processing pipeline orchestrating all stages in strict diagnostic sequence.
- Diagnostic snapshots: Enabled via `PIXREVIVE_DEBUG=1`, saving sequential stage PNGs (00_input_raw.png through 10b_after_projector_grade.png) and timing reports in `debug_output/`.

### 4. Evolution to v2.0 Two-Tier PPCAV Closed-Loop Orchestrator
The planned architecture documented in `pixrevive_orchestrator_plan.md` replaces the feedforward waterfall with:
- **Tier 1 (Inner Loop):** Native-resolution Perceive $\to$ Propose $\to$ Corroborate $\to$ Act $\to$ Verify loop with atomic `CanvasTransaction` rollbacks and a 20 Cost Point budget.
- **Tier 2 (Terminal Pass):** One-way Real-ESRGAN $4\times$ super-resolution, seam harmonization, and projector exhibition grade CLAHE.
- **Synthesis Manifest:** Outputting `synthesis_manifest.json` alongside restored images to document authentic vs. generative fabrication.
