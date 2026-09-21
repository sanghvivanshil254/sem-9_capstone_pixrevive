# Coding Conventions

**Analysis Date:** 2026-09-19

## Naming Patterns

**Files:**
- Backend modules: `snake_case.py` (e.g., `backend/ai/preprocessing/image_normalizer.py`, `backend/routers/image_router.py`).
- Frontend components: `PascalCase.jsx` (e.g., `frontend/src/components/ProgressBar.jsx`, `frontend/src/pages/WeightsAdmin.jsx`).
- Config and scripts: `camelCase.js` or `kebab-case.js` (e.g., `playwright.config.js`, `e2e/pixrevive.spec.js`).
- Test files: `test_<feature>.py` (e.g., `tests/test_restoration_regressions.py`, `tests/test_color_refiner.py`).

**Functions:**
- Python functions and methods: `snake_case` (e.g., `process_image()`, `detect_and_crop_borders()`, `validate_context()`).
- Python private methods / internal helpers: Prefixed with a single leading underscore (e.g., `_stage_0_despeckle()`, `_prune_old_tasks()`, `_sweep_orphaned_results()`).
- JavaScript functions: `camelCase` (e.g., `labelFor()`, `handleUpload()`, `setViewportSize()`).

**Variables:**
- Local variables: `snake_case` in Python (`working_image`, `task_id`), `camelCase` in JavaScript (`currentIdx`, `isUploading`).
- Global constants: `UPPER_SNAKE_CASE` (e.g., `MAX_FILE_SIZE = 25 * 1024 * 1024` in `backend/routers/image_router.py`, `STAGES` in `frontend/src/components/ProgressBar.jsx`).
- Private module constants: `_UPPER_SNAKE_CASE` (e.g., `_TASK_TTL_SECONDS = 1800`).

**Types & Classes:**
- Python classes: `PascalCase` (e.g., `RestorationPipeline`, `PipelineContext`, `QualityAnalyzer`, `EnhancedDamageDetector`).
- React components: `PascalCase` (e.g., `function ProgressBar({ progress })`).

## Code Style

**Formatting:**
- Python: PEP 8 compliant, 4-space indentation, double quotes for strings, type annotations enabled via `from __future__ import annotations`.
- JavaScript: 2-space indentation, single quotes for string literals, semicolons enforced.

**Linting:**
- Python: Pyright static analysis configured in `pyrightconfig.json` targeting Python 3.12.
- JavaScript: Oxlint configured via `npm run lint` (`oxlint` v1.71.0) in `frontend/package.json`.

## Import Organization

**Python Import Order:**
1. Future imports:
   ```python
   from __future__ import annotations
   ```
2. Standard library modules:
   ```python
   import os
   import time
   import logging
   import threading
   from typing import Dict, Any, Optional, Callable
   ```
3. Third-party libraries:
   ```python
   import cv2
   import numpy as np
   import torch
   from fastapi import APIRouter, UploadFile, File
   ```
4. Local application imports (using absolute or relative paths from `backend`):
   ```python
   from ai.pipeline.pipeline_context import PipelineContext
   from ai.analysis.quality_analyzer import QualityAnalyzer
   from ai.damage.enhanced_detector import EnhancedDamageDetector
   ```

## Error Handling Patterns

**Backend:**
- Use specific exceptions where possible (`ValueError`, `FileNotFoundError`, `torch.cuda.OutOfMemoryError`).
- Log caught exceptions with detailed context using `logger.error(..., exc_info=True)` or `logger.warning(...)`.
- When deep learning operations fail (e.g., LaMa infill or DDColor), implement graceful fallbacks (e.g., OpenCV Telea inpainting or grayscale pass-through) rather than crashing the HTTP request.
- File system cleanup operations (e.g. Windows file locks) catch `PermissionError` and `OSError` cleanly.

**Frontend:**
- Async fetch and EventSource operations are wrapped in `try/catch` blocks.
- Failed SSE streams trigger user-friendly toast notifications and reset processing state.
