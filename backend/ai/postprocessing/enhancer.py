"""
enhancer.py — Guided-filter seam blending along inpainting boundaries.

The `Enhancer` class that used to live here (adaptive USM, CLAHE,
micro-texture injection, color-cast correction) was deleted as dead code
(D17): CLAHE and sharpening live in `_stage_10b_projector_grade` /
`_stage_10_post_sr_sharpen`, and its color-cast logic was salvaged into
`ColorArtifactReducer.neutralize_global_cast` before deletion (T-7).
`Harmonizer` survives — the pipeline calls it for seam blending.
"""

from typing import Optional

import cv2
import numpy as np
from ai.pipeline.pipeline_context import PipelineContext


def _mask_matching(mask: Optional[np.ndarray], img_np: np.ndarray) -> Optional[np.ndarray]:
    """Return `mask` resized to img_np's frame, or None if unusable.

    Guards against a mask built at a different resolution than the image it is
    about to be combined with (e.g. a mask produced before super-resolution).
    """
    if mask is None:
        return None
    if mask.shape[:2] == img_np.shape[:2]:
        return mask
    resized = cv2.resize(
        mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    return (resized > 0).astype(np.uint8) * 255


class Harmonizer:
    @staticmethod
    def harmonize(context: PipelineContext) -> np.ndarray:
        """
        Gentle guided filter seam blending along inpainting boundary rims.
        """
        img_np = context.working_image
        mask = _mask_matching(context.damage_masks.get("final"), img_np)
        if mask is None or not np.any(mask):
            return img_np

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(mask, k)
        rim = cv2.subtract(dilated, mask)

        if not np.any(rim):
            return img_np

        try:
            guided = cv2.ximgproc.guidedFilter(img_np, img_np, radius=4, eps=20)
            rim_f = rim.astype(np.float32) / 255.0
            rim_f = cv2.GaussianBlur(rim_f, (5, 5), 0)
            rim_f = np.stack([rim_f] * 3, axis=2)
            harmonized = (img_np * (1.0 - rim_f) + guided * rim_f).astype(np.uint8)
            context.working_image = harmonized
        except Exception:
            pass

        return context.working_image
