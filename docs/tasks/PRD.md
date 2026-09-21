# PixRevive — Product Requirements Document (PRD) for Ralph Loop Execution

**Document Status:** Approved for Ralph Execution  
**Target Repository:** PixRevive (14-Stage Archival Photo Restoration)  
**Creation Date:** 2026-09-18  
**Reference Artifacts:** `.planning/codebase/` (`STACK.md`, `ARCHITECTURE.md`, `STRUCTURE.md`, `CONVENTIONS.md`, `TESTING.md`, `INTEGRATIONS.md`, `CONCERNS.md`), `plan.md`, `CODEBASE_MAP.md`, `CLAUDE.md`.

---

## 1. Executive Summary & Ralph Loop Operating Guidelines

### 1.1 Objective
This PRD defines the concrete, actionable sequence of engineering tasks required to harden, optimize, and complete the PixRevive exhibition-grade photo restoration system. Each task is self-contained, ordered strictly by dependency, and sized specifically for single-iteration completion by an autonomous agent (Ralph Loop).

### 1.2 Mandatory Ralph Ground Rules (Repository Invariants)
All Ralph loop iterations MUST adhere to the following non-negotiable rules:
1. **Python Runtime:** Use the system Python 3.12 interpreter only (`C:\Program Files\Python312\python.exe`). Never create or activate virtual environments (`venv`, `.venv`, `conda`).
2. **Pytest Executable:** Use `C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe` with `$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"`.
3. **Weights Locality:** All model weights must reside in `backend/ai/weights/`. No downloads to `~/.cache` or user home folders.
4. **DDColor Dependency Guard:** Colorization is DDColor (ICCV 2023) vendored into `backend/ai/colorization/ddcolor/`. Never run `pip install ddcolor` or upgrade `basicsr` beyond `basicsr==1.4.2`.
5. **Stage Monotonicity:** Any change affecting pipeline progress stages must update `STAGES` in `frontend/src/components/ProgressBar.jsx` and pass `scripts/check_progress_stream.py`.

---

## 2. Execution Task Matrix Overview

| Task ID | Track | Task Title | Dependencies | Complexity |
|---|---|---|---|---|
| **FIX-01** | Debt | Centralize `torchvision.transforms.functional_tensor` Shim | None | Low |
| **FIX-02** | Bug | Harden Server Restart & Task Result Pruning Lifecycle | None | Low |
| **FIX-03** | Debt | Add Automated Unit Tests for `image_router.py` API Endpoints | FIX-01, FIX-02 | Medium |
| **FIX-04** | Security | Harden & Test `admin_router.py` Zip and Upload Bounds | None | Medium |
| **FIX-05** | Perf | Dynamic Low-VRAM Tile Sizing & Memory Guard in Real-ESRGAN | None | Medium |
| **FEAT-01** | Pipeline | Multi-Class Damage Classification Map (Chapter 6) | FIX-01 | Medium |
| **FEAT-02** | Pipeline | Integrate Multi-Class Damage Map into `MaskRefiner` | FEAT-01 | Medium |
| **FEAT-03** | Color | Face Landmark-Gated Skin Tone Protection in DDColor Refiner | None | Medium |
| **FEAT-04** | Face | Multi-Face Landmark Protection & Deferred Restoration Guard | FEAT-03 | Medium |
| **FEAT-05** | UI/API | Full Synchronization of Progress Stages with Monotonic Validator | FIX-03 | Low |
| **TEST-01** | Quality | End-to-End SSE Progress Stream Monotonicity Automated Test | FEAT-05 | Low |
| **TEST-02** | Quality | Quantitative Metric Benchmark Harness (PSNR, SSIM, LPIPS) | FEAT-02, FEAT-03 | Medium |
| **TEST-03** | Quality | Visual Debugging Snapshot Verification (`PIXREVIVE_DEBUG=1`) | None | Low |

---

## 3. Track 1: Bugs & Technical Debt Remediation (FIX)

---

### [TASK-FIX-01] Centralize `torchvision.transforms.functional_tensor` Compatibility Shim

- **Category:** Technical Debt
- **Dependencies:** None
- **Affected Files:**
  - `backend/ai/compat.py` (NEW)
  - `backend/ai/analysis/face_detector.py` (MODIFY)
  - `backend/ai/restoration/face_restorer.py` (MODIFY)
  - `tests/test_restoration_regressions.py` (MODIFY)

#### Context & Problem
Upstream `basicsr==1.4.2` and `gfpgan>=1.3.8` require `torchvision.transforms.functional_tensor`, which was removed in modern TorchVision. Currently, both `face_detector.py` and `face_restorer.py` implement duplicate monkey-patches into `sys.modules`. If any script imports GFPGAN or facexlib before one of these two files, it crashes with `ModuleNotFoundError`.

#### Objective
Create a single canonical compatibility module `backend/ai/compat.py` that reliably installs the `torchvision.transforms.functional_tensor` alias before any external library can trigger the import error. Update all modules to import `compat` first.

#### Acceptance Criteria
- [ ] `backend/ai/compat.py` is created with proper docstrings and idempotent `sys.modules` monkey-patching.
- [ ] `face_detector.py` and `face_restorer.py` remove redundant duplicate inline monkey-patches and import `ai.compat`.
- [ ] Direct import of `basicsr` or `gfpgan` in a standalone script succeeds without prior manual patching.
- [ ] A dedicated unit test in `tests/test_restoration_regressions.py` asserts that `torchvision.transforms.functional_tensor` is available in `sys.modules` and points to `torchvision.transforms.functional`.

#### Verification Commands
```powershell
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Program Files\Python312\python.exe" -c "import ai.compat, basicsr, gfpgan; print('Shim OK')"
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests/test_restoration_regressions.py -k test_opencv_runtime_has_required_apis
```

---

### [TASK-FIX-02] Harden Server Restart & Task Result Pruning Lifecycle

- **Category:** Bug / Reliability
- **Dependencies:** None
- **Affected Files:**
  - `backend/routers/image_router.py` (MODIFY)
  - `tests/test_image_router.py` (NEW)

#### Context & Problem
In `backend/routers/image_router.py`, `_sweep_orphaned_results()` and `_prune_old_tasks()` run during module import and stream generation. However, if a task fails or an unhandled OS error occurs while deleting files in `_results_dir`, task metadata may become desynchronized or throw unhandled exceptions during file deletion. Additionally, task cleanup lacks structured unit testing.

#### Objective
Ensure atomic, error-resilient task deletion with comprehensive try/except guards for Windows file locks (`PermissionError`), enforce strict file lifetime checks, and prevent memory leaks in `_tasks`.

#### Acceptance Criteria
- [ ] `_prune_old_tasks` safely handles file lock contention without raising unhandled exceptions.
- [ ] `_sweep_orphaned_results` validates that files slated for deletion are strictly within `temp_results` and have a valid hexadecimal `.png` format.
- [ ] Unit tests in `tests/test_image_router.py` verify that expired tasks (>1800s) are evicted from `_tasks` and corresponding files deleted.

#### Verification Commands
```powershell
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests/test_image_router.py -v
```

---

### [TASK-FIX-03] Add Automated Unit Tests for `image_router.py` Endpoints

- **Category:** Technical Debt / Test Coverage
- **Dependencies:** FIX-01, FIX-02
- **Affected Files:**
  - `tests/test_image_router.py` (MODIFY/EXPAND)
  - `backend/routers/image_router.py` (READ ONLY)

#### Context & Problem
While Playwright exercises the full UI, there are no FastAPI `TestClient` unit tests verifying HTTP status codes, file-size guards (25MB limit), invalid MIME type rejection (400), or status query endpoints (`/api/process/{task_id}/status`) in isolation.

#### Objective
Implement comprehensive `TestClient` suite for `backend/routers/image_router.py`.

#### Acceptance Criteria
- [ ] Test `POST /api/upload` with valid JPEG/PNG (returns 200).
- [ ] Test `POST /api/upload` with invalid file type (e.g. `.txt`, `.pdf`) and assert 400 Bad Request.
- [ ] Test `POST /api/upload` with payload exceeding 25 MB and assert 400 Bad Request with size detail.
- [ ] Test `GET /api/process/{task_id}/status` for nonexistent task (returns 404).
- [ ] Test `GET /api/health` and verify schema structure (`status`, `models`, `ready`).

#### Verification Commands
```powershell
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests/test_image_router.py -v
```

---

### [TASK-FIX-04] Harden & Test `admin_router.py` Zip and Upload Bounds

- **Category:** Security / Reliability
- **Dependencies:** None
- **Affected Files:**
  - `backend/routers/admin_router.py` (MODIFY)
  - `tests/test_admin_router.py` (NEW)

#### Context & Problem
`backend/routers/admin_router.py` handles multi-gigabyte model weights and dataset ZIP uploads. While upload ceilings exist (`MAX_ZIP_UPLOAD_BYTES = 2GB`), the zip extraction path-traversal (Zip-Slip) verification relies on regex `\.\./` which may miss Windows-style backslash traversals (`..\`). Additionally, `admin_router.py` has 0% automated test coverage.

#### Objective
1. Normalize and resolve canonical archive paths against `os.path.commonpath` to ensure complete immunity against Zip-Slip across both POSIX and Windows paths.
2. Provide automated tests for weights inspection, bounded file writer, and invalid archive rejection.

#### Acceptance Criteria
- [ ] Zip extraction validates every member path using `os.path.commonpath([dest_dir, target_path]) == dest_dir`.
- [ ] A mock zip containing directory traversal members (`../evil.txt`, `..\evil.txt`) is rejected with an explicit error.
- [ ] Unit tests verify `GET /api/admin/weights/status` returns expected model dictionary.
- [ ] Uploads exceeding `MAX_WEIGHTS_UPLOAD_BYTES` abort with HTTP 413.

#### Verification Commands
```powershell
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests/test_admin_router.py -v
```

---

### [TASK-FIX-05] Dynamic Low-VRAM Tile Sizing & Memory Guard in Real-ESRGAN

- **Category:** Performance / Stability
- **Dependencies:** None
- **Affected Files:**
  - `backend/ai/restoration/super_resolution.py` (MODIFY)
  - `tests/test_restoration_regressions.py` (MODIFY)

#### Context & Problem
When running Real-ESRGAN 4x upscaling on GPUs with <= 4GB VRAM or when scaling images with dimensions >= 1024px to 4K, fixed tile sizes (512px) can trigger CUDA Out Of Memory errors.

#### Objective
Add a VRAM detection helper that inspects `torch.cuda.get_device_properties` (if CUDA is active) and dynamically adjusts tile size (256px for <= 4GB VRAM, 512px for >= 8GB VRAM) with automatic fallback to CPU processing if CUDA allocation fails.

#### Acceptance Criteria
- [ ] `SuperResolution.enhance()` dynamically determines safe tile size based on available VRAM or user device preference.
- [ ] If a `torch.cuda.OutOfMemoryError` occurs during tiled forward pass, the method catches it, frees CUDA cache (`torch.cuda.empty_cache()`), logs a warning, and re-executes the forward pass on CPU.
- [ ] Regression test verifies that tile size calculation returns 256 for simulated 4GB VRAM and 512 for 8GB VRAM.

#### Verification Commands
```powershell
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests/test_restoration_regressions.py -k test_target_long_side
```

---

## 4. Track 2: Architecture & Pipeline Enhancements (FEAT)

---

### [TASK-FEAT-01] Multi-Class Damage Classification Map (Chapter 6 Specification)

- **Category:** Feature / Core Pipeline
- **Dependencies:** FIX-01
- **Affected Files:**
  - `backend/ai/damage/damage_classifier.py` (NEW)
  - `backend/ai/damage/enhanced_detector.py` (MODIFY)
  - `backend/ai/pipeline/pipeline_context.py` (MODIFY)
  - `tests/test_damage_classification.py` (NEW)

#### Context & Problem
`plan.md` Chapter 6 specifies: "Do not produce only: damaged, clean. Use: 0 = clean, 1 = scratch, 2 = dust/speck, 3 = white flake, 4 = stain/tear, 5 = missing region, 6 = color contamination. Your mask should conceptually be: `damage_map[y][x] = class_id`." Currently, PixRevive collapses all detected damages into binary 0/255 masks, losing semantic information about what algorithm should repair each area.

#### Objective
Implement `DamageClassifier` and update `PipelineContext` to store a multi-class integer damage tensor `damage_map: np.ndarray (uint8)` alongside the fused binary mask.

#### Acceptance Criteria
- [ ] Class constants defined: `DAMAGE_CLEAN=0`, `DAMAGE_SCRATCH=1`, `DAMAGE_SPECK=2`, `DAMAGE_WHITE_FLAKE=3`, `DAMAGE_TEAR=4`, `DAMAGE_MISSING=5`, `DAMAGE_COLOR_CONTAMINATION=6`.
- [ ] `EnhancedDamageDetector` populates `context.damage_masks["damage_map"]` with individual label IDs for detected regions before merging into the binary inpaint mask.
- [ ] `PipelineContext` image metadata records percentage breakdown per damage class.
- [ ] Unit tests verify that synthetic images with simulated scratches, specks, and tears receive distinct non-zero class IDs in `damage_map`.

#### Verification Commands
```powershell
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests/test_damage_classification.py -v
```

---

### [TASK-FEAT-02] Integrate Multi-Class Damage Map into `MaskRefiner` & Inpainting Engine

- **Category:** Feature / Core Pipeline
- **Dependencies:** FEAT-01
- **Affected Files:**
  - `backend/ai/damage/mask_refiner.py` (MODIFY)
  - `backend/ai/restoration/inpainting_engine.py` (MODIFY)
  - `tests/test_restoration_regressions.py` (MODIFY)

#### Context & Problem
With multi-class damage maps available from FEAT-01, `MaskRefiner` should apply specialized morphological operations per class: fine scratches require 1px dilation, whereas large missing regions/tears require smooth boundary dilation and contour feathering.

#### Objective
Update `MaskRefiner` to consume the multi-class map, applying class-specific kernel sizes and preserving thin scratch detail while smoothing wide fracture perimeters.

#### Acceptance Criteria
- [ ] Scratches (class 1) are dilated with a small $3\times3$ cross kernel to prevent over-dilation.
- [ ] Tears/Missing regions (class 4, 5) receive $5\times5$ elliptical dilation and Gaussian edge feathering.
- [ ] The final binary mask passed to LaMa retains all damaged classes while eliminating speck scatter.
- [ ] Regression test verifies that scratch line width is preserved without ballooning.

#### Verification Commands
```powershell
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests/test_restoration_regressions.py -k test_multiscale_fill
```

---

### [TASK-FEAT-03] Face Landmark-Gated Skin Tone Protection in DDColor Refiner

- **Category:** Feature / Colorization
- **Dependencies:** None
- **Affected Files:**
  - `backend/ai/colorization/color_refiner.py` (MODIFY)
  - `backend/ai/pipeline/restoration_pipeline.py` (MODIFY)
  - `tests/test_color_refiner.py` (NEW)

#### Context & Problem
As documented in `CODEBASE_MAP.md` Section 3: "Stage 6 (DDColor) does not currently consume face landmarks, applying a global transformation." Historical portraits colorized by DDColor can suffer from unnatural color casts (e.g. green or cyan bleed) across facial skin, oral cavities, and eyes.

#### Objective
Connect `FaceDetector` bounding boxes and 5 landmarks (eyes, nose, mouth corners) into `ColorArtifactReducer` (`color_refiner.py`). Implement facial skin tone chrominance bounding in LAB space ($a^* \in [130, 170]$, $b^* \in [130, 175]$) specifically within detected face ellipses, softly blended at boundaries.

#### Acceptance Criteria
- [ ] `ColorArtifactReducer` accepts optional `faces` metadata from `PipelineContext`.
- [ ] Detected face zones are clamped against extreme color casts in the $a^*$ (green-red) and $b^*$ (blue-yellow) channels.
- [ ] Eye and oral landmark regions are exempted from skin-tone warming to preserve natural sclera and lips.
- [ ] Unit tests verify that a synthetic face region with an injected green color cast is restored to natural warm tone bounds.

#### Verification Commands
```powershell
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests/test_color_refiner.py -v
```

---

### [TASK-FEAT-04] Multi-Face Landmark Protection & Deferred Restoration Guard

- **Category:** Feature / Face Restoration
- **Dependencies:** FEAT-03
- **Affected Files:**
  - `backend/ai/restoration/face_restorer.py` (MODIFY)
  - `backend/ai/pipeline/restoration_pipeline.py` (MODIFY)
  - `tests/test_facial_landmarks.py` (MODIFY)

#### Context & Problem
Group photographs contain multiple faces with varying resolutions. Small background faces (< 90px) get corrupted if processed with full-resolution GFPGAN prior to super-resolution. Furthermore, when multiple faces are present, landmark protection zones must be independently generated for each face.

#### Objective
Ensure `FaceRestorer` iterates through all detected faces independently, applies fidelity scaling proportional to face bounding box size, and correctly queues low-resolution faces for Stage 9c deferred restoration.

#### Acceptance Criteria
- [ ] In images with $N \ge 2$ faces, every face bounding box is evaluated individually.
- [ ] Faces with `width < 90px` or `height < 90px` are bypassed in Stage 7 and restored in Stage 9c on the 4x upscaled canvas.
- [ ] Test in `test_facial_landmarks.py` asserts that multi-face detection produces non-overlapping protection masks for all subjects.

#### Verification Commands
```powershell
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests/test_facial_landmarks.py -v
```

---

### [TASK-FEAT-05] Full Synchronization of Progress Stages with Monotonic Validator

- **Category:** Reliability / UI Contract
- **Dependencies:** FIX-03
- **Affected Files:**
  - `backend/ai/pipeline/restoration_pipeline.py` (MODIFY)
  - `frontend/src/components/ProgressBar.jsx` (MODIFY)
  - `scripts/check_progress_stream.py` (MODIFY)

#### Context & Problem
`CLAUDE.md` Rule 5 requires: "`_emit(stage, percent, message)` in `restoration_pipeline.py` clamps percentages so they never decrease. Stage ids must stay in sync with `STAGES` in `ProgressBar.jsx`." Any drift between backend stage IDs and frontend stage IDs causes UI dots to remain grey or misalign during restoration.

#### Objective
Audit all `_emit()` calls across stages 0 through 10b in `restoration_pipeline.py`. Ensure every emitted stage ID matches `STAGES` in `ProgressBar.jsx` exactly, and confirm nominal percentages are strictly increasing.

#### Acceptance Criteria
- [ ] Every stage ID emitted in `restoration_pipeline.py` exists in `frontend/src/components/ProgressBar.jsx:STAGES`.
- [ ] No stage emission percentage is smaller than the preceding stage's percentage.
- [ ] Running `scripts/check_progress_stream.py` against a test image completes with 0 errors and validates monotonicity.

#### Verification Commands
```powershell
& "C:\Program Files\Python312\python.exe" scripts/check_progress_stream.py
```

---

## 5. Track 3: Verification, Test Harness & Benchmarking (TEST)

---

### [TASK-TEST-01] End-to-End SSE Progress Stream Automated Pytest

- **Category:** Quality Assurance
- **Dependencies:** FEAT-05
- **Affected Files:**
  - `tests/test_progress_stream.py` (NEW)
  - `backend/routers/image_router.py` (READ ONLY)

#### Context & Problem
Currently, progress stream validation relies on manual execution of `scripts/check_progress_stream.py`. There is no automated Pytest that exercises the `/api/process-stream` SSE endpoint in the regular test suite.

#### Objective
Create a Pytest module using `httpx` or `TestClient` with streaming response consumption to assert:
1. Event stream starts with `progress` events.
2. Percentages monotonically increase from 0 to 100.
3. Event stream terminates cleanly with `("done", filename)` or `complete`.

#### Acceptance Criteria
- [ ] `tests/test_progress_stream.py` runs within the standard pytest suite without external server invocation.
- [ ] Test validates SSE data formatting (`event: progress\ndata: {...}\n\n`).
- [ ] Asserts that task ID generated in stream can be fetched from `/api/process/{task_id}/status`.

#### Verification Commands
```powershell
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests/test_progress_stream.py -v
```

---

### [TASK-TEST-02] Quantitative Metric Benchmark Harness (PSNR, SSIM, LPIPS)

- **Category:** Benchmarking / Quality Control
- **Dependencies:** FEAT-02, FEAT-03
- **Affected Files:**
  - `scripts/benchmark_dataset.py` (NEW)
  - `tests/test_restoration_quality.py` (NEW)

#### Context & Problem
`plan.md` Chapter 21 mandates: "Do not judge improvements only by: 'This image looks better to me.' That is subjective. Compare PSNR, SSIM, processing time, and memory usage against `tests/restoration_dataset/damaged/` (cases 1 through 6)."

#### Objective
Create an automated benchmarking script that runs cases 1 through 6 through the full pipeline, records quantitative metrics (PSNR, SSIM, execution time per stage), and outputs a clean markdown comparison table in `test_results/benchmark_report.md`.

#### Acceptance Criteria
- [ ] Script `scripts/benchmark_dataset.py` loads standard damaged test cases (`case1` to `case6`).
- [ ] Computes and records:
  - Identity SSIM on face crops.
  - Overall pixel preservation in undamaged regions (> 95% target).
  - Millisecond execution duration per stage.
- [ ] Outputs summary table to `test_results/benchmark_report.md`.
- [ ] A lightweight version is registered in `tests/test_restoration_quality.py` asserting minimum thresholds.

#### Verification Commands
```powershell
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Program Files\Python312\python.exe" scripts/benchmark_dataset.py
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests/test_restoration_quality.py -v
```

---

### [TASK-TEST-03] Visual Debugging Snapshot Verification (`PIXREVIVE_DEBUG=1`)

- **Category:** Diagnostic Tooling
- **Dependencies:** None
- **Affected Files:**
  - `tests/test_debug_logger.py` (NEW)
  - `backend/ai/pipeline/debug_logger.py` (READ ONLY)

#### Context & Problem
When `PIXREVIVE_DEBUG=1` is set, `RestorationPipeline` saves 18+ sequential numbered PNG snapshots and logs to `debug_output/`. If stage renaming occurs or a stage fails to log its snapshot, the visual debugging pipeline silently skips frames.

#### Objective
Create an automated test that enables `PIXREVIVE_DEBUG=1`, executes a mock pipeline restoration, and asserts the existence of all numbered snapshot artifacts (`00_input_raw.png` through `10b_after_projector_grade.png`, `06_timings.txt`, `07_full_log.txt`, `08_metadata.json`).

#### Acceptance Criteria
- [ ] Test runs with `os.environ["PIXREVIVE_DEBUG"] = "1"` against a test fixture.
- [ ] Asserts all expected filenames in `debug_output/` are generated and have non-zero file sizes.
- [ ] Asserts `06_timings.txt` contains timing entries for all active stages.
- [ ] Cleans up test artifacts in temporary test directory after completion.

#### Verification Commands
```powershell
$env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
& "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests/test_debug_logger.py -v
```

---

## 6. Ralph Loop Execution Protocol

When Ralph executes tasks from this PRD, it MUST follow this strict iteration sequence:

1. **Pick the lowest-numbered uncompleted Task ID** whose dependencies are fully satisfied.
2. **Review task requirements and acceptance criteria** without making extraneous architectural changes.
3. **Execute modifications** using repository conventions (PEP 8, type annotations, descriptive docstrings, no venvs).
4. **Execute the task's specific Verification Command** using system Python 3.12.
5. **Run the regression test suite:**
   ```powershell
   $env:PYTHONPATH = "d:\Desktop\kapstone\kapstone\PixRevive\backend"
   & "C:\Users\sangh\AppData\Roaming\Python\Python312\Scripts\pytest.exe" tests/ -v
   ```
6. **Mark acceptance criteria as checked** in this PRD.
7. **Commit task changes** with message format: `fix/feat(task-id): <description>`.
8. **Handoff to next Ralph iteration.**
