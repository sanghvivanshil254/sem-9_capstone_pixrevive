"""
image_router.py — FastAPI endpoints for photo restoration.

Changes vs. previous version:
  - Result files saved as .png (lossless) instead of .jpg.
  - FileResponse returns image/png with filename restored.png.
  - Added output dimension validation before sending result.
  - Added task TTL cleanup: tasks older than 30 min auto-pruned from _tasks.
  - MAX_FILE_SIZE raised from 10 MB to 25 MB to accommodate high-res uploads.
  - Proper error logging for all failure paths.
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import Response, StreamingResponse, FileResponse
from services.image_service import ImageService

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 25 * 1024 * 1024   # 25 MB — accommodate high-res scans

# In-memory result store: task_id → {"status": ..., "result_path": ..., ...}
# Fix 12: Protected by _tasks_lock to prevent race conditions under concurrent requests.
_tasks: dict = {}
_tasks_lock = threading.Lock()  # Fix 12: guards all _tasks reads/writes
_gpu_semaphore = threading.Semaphore(1)  # Serializes GPU/CPU-heavy inference passes
_results_dir = os.path.join(os.path.dirname(__file__), "..", "temp_results")
os.makedirs(_results_dir, exist_ok=True)

# Task TTL: prune tasks older than 30 min to prevent unbounded memory growth
_TASK_TTL_SECONDS = 1800


def _prune_old_tasks() -> None:
    """Remove completed tasks older than _TASK_TTL_SECONDS from memory."""
    now = time.time()
    with _tasks_lock:
        expired = [
            tid for tid, t in _tasks.items()
            if t.get("status") in ("done", "error")
            and now - t.get("created_at", now) > _TASK_TTL_SECONDS
        ]
        for tid in expired:
            result_path = _tasks[tid].get("result_path")
            if result_path and os.path.exists(result_path):
                try:
                    os.remove(result_path)
                except (OSError, PermissionError) as err:
                    logger.debug("Could not remove expired task file %s: %s", result_path, err)
            del _tasks[tid]
    if expired:
        logger.debug("Pruned %d expired task(s).", len(expired))


def _sweep_orphaned_results() -> None:
    """Delete result files left behind by a previous process.

    _tasks lives in memory only, so every restart orphans the PNGs written by
    the previous run — _prune_old_tasks can never reach them and the directory
    grows without bound. Sweep anything older than the task TTL at import time.
    """
    now = time.time()
    removed = 0
    try:
        entries = os.listdir(_results_dir)
    except OSError:
        return
    results_dir_abs = os.path.abspath(_results_dir)
    for name in entries:
        if not name.endswith(".png"):
            continue
        path = os.path.abspath(os.path.join(_results_dir, name))
        if os.path.commonpath([results_dir_abs, path]) != results_dir_abs:
            continue
        try:
            if now - os.path.getmtime(path) > _TASK_TTL_SECONDS:
                os.remove(path)
                removed += 1
        except (OSError, PermissionError):
            pass
    if removed:
        logger.info("Swept %d orphaned result file(s) from %s.", removed, _results_dir)


_sweep_orphaned_results()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Validate and receive an image file."""
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. "
                   "Allowed: JPEG, PNG, WebP.",
        )
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(content)//1024} KB). "
                   f"Max size is {MAX_FILE_SIZE//1024//1024} MB.",
        )
    return {
        "filename":     file.filename,
        "content_type": file.content_type,
        "size_bytes":   len(content),
        "message":      "File received successfully.",
    }


@router.post("/process")
async def process_image(
    file:     UploadFile = File(...),
    restore:  bool = Form(True),
    repair:   bool = Form(True),
    colorize: bool = Form(True),
    denoise:  bool = Form(False),
    device:   str  = Form("auto"),
):
    """
    Synchronous endpoint — processes image and returns PNG bytes directly.
    Use /process-stream for large images to receive progress updates.
    """
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type.")
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max {MAX_FILE_SIZE//1024//1024} MB.",
        )

    try:
        with _gpu_semaphore:
            processed_bytes = ImageService.process_image(
                content,
                enable_restoration=restore,
                enable_repair=repair,
                enable_colorization=colorize,
                enable_denoise=denoise,
                device_preference=device,
                filename=file.filename or "unknown",
            )
        return Response(
            content=processed_bytes,
            media_type="image/png",
            headers={"Content-Disposition": 'attachment; filename="restored.png"'},
        )
    except Exception as exc:
        logger.exception("Synchronous processing failed.")
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")


@router.post("/process-stream")
async def process_image_stream(
    file:     UploadFile = File(...),
    restore:  bool = Form(True),
    repair:   bool = Form(True),
    colorize: bool = Form(True),
    denoise:  bool = Form(False),
    device:   str  = Form("auto"),
):
    """
    Stream pipeline progress via Server-Sent Events, then provide result URL.

    Events:
      progress  → {"stage": str, "percent": int, "message": str}
      complete  → {"percent": 100, "message": str, "task_id": str}
      error     → {"message": str}

    After `complete`, call GET /process/{task_id}/result to download the image.
    """
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type.")
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max {MAX_FILE_SIZE//1024//1024} MB.",
        )

    _prune_old_tasks()

    task_id = uuid.uuid4().hex[:12]
    with _tasks_lock:  # Fix 12: thread-safe initial write
        _tasks[task_id] = {
            "status":     "queued",
            "progress":   [],
            "created_at": time.time(),
        }

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(item) -> None:
            """Hand an event to the event loop from the worker thread.

            asyncio.Queue is NOT thread-safe: calling put_nowait directly from
            the worker thread can miss the loop wakeup and stall the SSE stream.
            call_soon_threadsafe is the supported cross-thread handoff.
            """
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def progress_callback(stage: str, percent: int, message: str) -> None:
            with _tasks_lock:  # Fix 12
                _tasks[task_id]["progress"].append((stage, percent, message))
            emit(("progress", stage, percent, message))

        def run() -> None:
            try:
                with _tasks_lock:  # Fix 12
                    _tasks[task_id]["status"] = "processing"
                with _gpu_semaphore:
                    result_bytes = ImageService.process_image(
                        content,
                        enable_restoration=restore,
                        enable_repair=repair,
                        enable_colorization=colorize,
                        enable_denoise=denoise,
                        device_preference=device,
                        progress_callback=progress_callback,
                        filename=file.filename or "unknown",
                    )
                # Save lossless PNG result
                result_filename = f"{task_id}.png"
                result_path = os.path.join(_results_dir, result_filename)
                with open(result_path, "wb") as fp:
                    fp.write(result_bytes)

                with _tasks_lock:  # Fix 12
                    _tasks[task_id].update({
                        "status":      "done",
                        "result_path": result_path,
                    })
                emit(("done", result_filename))

            except Exception as exc:
                logger.exception(f"Task {task_id} failed.")
                with _tasks_lock:  # Fix 12
                    _tasks[task_id].update({"status": "error", "error": str(exc)})
                emit(("error", str(exc)))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        while True:
            event = await queue.get()
            if event[0] == "progress":
                _, stage, pct, msg = event
                yield (
                    f"event: progress\n"
                    f"data: {json.dumps({'stage': stage, 'percent': pct, 'message': msg})}\n\n"
                )
            elif event[0] == "done":
                yield (
                    f"event: complete\n"
                    f"data: {json.dumps({'percent': 100, 'message': 'Restoration complete', 'task_id': task_id})}\n\n"
                )
                break
            elif event[0] == "error":
                _, err_msg = event
                yield (
                    f"event: error\n"
                    f"data: {json.dumps({'message': err_msg})}\n\n"
                )
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            # Progress is only useful live. Without these a caching layer or an
            # nginx reverse proxy in front of uvicorn buffers the whole stream
            # and the client gets every event at once, after the work is done.
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/process/{task_id}/result")
async def get_result(task_id: str):
    """Download the processed PNG result for a given task ID."""
    with _tasks_lock:  # Fix 12: thread-safe read
        task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task["status"] == "error":
        raise HTTPException(
            status_code=500,
            detail=task.get("error", "Processing failed."),
        )
    if task["status"] != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Task not yet complete (status: {task['status']}).",
        )

    result_path = task.get("result_path")
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=500, detail="Result file not found.")

    return FileResponse(
        result_path,
        media_type="image/png",
        filename="restored.png",
        headers={"Content-Disposition": 'attachment; filename="restored.png"'},
    )


@router.get("/process/{task_id}/status")
async def get_status(task_id: str):
    """Poll task status (non-SSE fallback for environments without SSE support)."""
    with _tasks_lock:  # Fix 12: thread-safe read
        task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return {
        "task_id":  task_id,
        "status":   task["status"],
        "progress": task.get("progress", []),
        "error":    task.get("error"),
    }
