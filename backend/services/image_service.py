"""
image_service.py — Thin orchestration layer between routers and pipeline.

Holds the singleton RestorationPipeline instance and forwards requests.
"""

from typing import Optional, Callable
from ai.pipeline.restoration_pipeline import RestorationPipeline

pipeline = RestorationPipeline()


class ImageService:
    @staticmethod
    def process_image(
        image_bytes: bytes,
        enable_restoration:  bool = True,
        enable_repair:       bool = True,
        enable_colorization: bool = True,   # Fix 11: aligned with restore() default
        enable_denoise:      bool = True,   # Fix 11: aligned with restore() default
        device_preference:   str  = "auto",
        progress_callback:   Optional[Callable[[str, int, str], None]] = None,
        filename:            str  = "unknown",
    ) -> bytes:
        return pipeline.process(
            image_bytes,
            external_metadata={"filename": filename},
            enable_restoration=enable_restoration,
            enable_repair=enable_repair,
            enable_colorization=enable_colorization,
            enable_denoise=enable_denoise,
            device_preference=device_preference,
            progress_callback=progress_callback,
        )