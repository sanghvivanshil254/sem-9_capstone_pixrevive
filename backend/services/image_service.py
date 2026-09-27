"""
image_service.py — Thin orchestration layer between routers and pipeline.

Holds the singleton RestorationPipeline instance and forwards requests.
"""

from typing import Any, Dict, Optional, Callable
from ai.pipeline.restoration_pipeline import RestorationPipeline

pipeline = RestorationPipeline()


class ImageService:
    @staticmethod
    def get_last_restoration_report() -> Optional[Dict[str, Any]]:
        """T-2 Restoration Report of the most recent process_image() call.

        The GPU semaphore serializes pipeline runs, so reading this right
        after process_image() returns (same worker thread) is safe.
        """
        ctx = getattr(pipeline, "last_context", None)
        if ctx is None:
            return None
        return ctx.image_metadata.get("restoration_report")

    @staticmethod
    def process_image(
        image_bytes: bytes,
        enable_restoration:  bool = True,
        enable_repair:       bool = True,
        enable_colorization: bool = True,   # Fix 11: aligned with restore() default
        enable_denoise:      bool = True,   # Fix 11: aligned with restore() default
        enable_reimagine:    bool = False,  # Phase H1 opt-in beta
        device_preference:   str  = "auto",
        progress_callback:   Optional[Callable[[str, int, str], None]] = None,
        filename:            str  = "unknown",
        task_id:             Optional[str] = None,  # Stage-12 evidence overlay key
    ) -> bytes:
        import uuid
        return pipeline.process(
            image_bytes,
            external_metadata={
                "filename": filename,
                # Stage 12 keys the "Show what's real" overlay artifact to
                # this id; the SSE router passes its real task_id so the
                # /api/process/{id}/evidence URL matches the report.
                "task_id": task_id or uuid.uuid4().hex[:12],
            },
            enable_restoration=enable_restoration,
            enable_repair=enable_repair,
            enable_colorization=enable_colorization,
            enable_denoise=enable_denoise,
            enable_reimagine=enable_reimagine,
            device_preference=device_preference,
            progress_callback=progress_callback,
        )