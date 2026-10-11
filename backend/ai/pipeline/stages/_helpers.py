"""Shared module-level helpers for the pipeline stage mixins."""

import cv2

from ai.pipeline.pipeline_context import PipelineContext


def _refresh_sharpness(context: PipelineContext) -> None:
    """Re-measure blur_score on current working image."""
    try:
        gray = cv2.cvtColor(context.working_image, cv2.COLOR_RGB2GRAY)
        refreshed = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        meta = context.image_metadata
        previous = float(meta.get("blur_score", 0.0))
        meta.setdefault("blur_score_raw", previous)
        meta["blur_score"] = refreshed
        meta["blur_score_norm"] = min(refreshed / 500.0, 1.0)
    except Exception as exc:
        context.add_log(f"[WARN] Could not refresh blur_score: {exc}")
