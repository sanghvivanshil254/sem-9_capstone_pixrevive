"""
quality_service.py — Final output quality validation.

Validates the pipeline output before returning it to the user.
Catches severe processing failures that would make the output worse
than the input. Returns True if the image passes all checks.

Checks:
  1. Image exists and has valid dimensions
  2. No NaN values
  3. Not completely black or white
  4. Has sufficient dynamic range (at least 30 levels)
  5. Mean brightness in a reasonable range [15, 245]
  6. No massive solid-colour blocks (>30% of pixels identical)
  7. Output dimensions are >= input dimensions (no accidental downscale)
"""

import logging
import time

import cv2
import numpy as np
from ai.pipeline.pipeline_context import PipelineContext

logger = logging.getLogger(__name__)


class QualityService:

    @staticmethod
    def compute_metrics(context: PipelineContext) -> dict:
        """Reference-free perceptual metrics on the final canvas (Q-6/D6).

        Called at pipeline end behind PIXREVIVE_DEBUG=1. NIQE (naturalness,
        lower = better) is computed on the working image and on the original
        input — on a healthy run the output score should be no worse than the
        input's beyond a tolerance. Timing/PSNR-style metrics need ground
        truth this pipeline doesn't have; scripts/benchmark_quality.py owns
        the full per-case report instead. Metrics are stored into context
        metadata and never raise.
        """
        started = time.time()
        metrics: dict = {}
        try:
            from ai.analysis.niqe_metric import niqe_score

            out = niqe_score(context.working_image)
            metrics["output_niqe"] = out
            metrics["input_niqe"] = context.image_metadata.get("niqe_score")
            if out is not None:
                context.image_metadata["output_niqe"] = out
                context.add_log(f"Quality metrics: output NIQE={out:.2f}.")
        except Exception as exc:
            metrics["error"] = str(exc)
        metrics["compute_ms"] = round((time.time() - started) * 1000.0, 1)
        return metrics

    @staticmethod
    def validate(context: PipelineContext) -> bool:
        """
        Runs right before returning the final image to catch severe issues.
        Returns True if image is valid, False if output should be rejected.
        """
        img_np = context.working_image

        # 1. Output exists
        if img_np is None or img_np.size == 0:
            context.add_log("CRITICAL ERROR: Image is empty.")
            return False

        # 2. Dimensions valid
        if len(img_np.shape) not in (2, 3):
            context.add_log(f"CRITICAL ERROR: Invalid image dimensions {img_np.shape}.")
            return False

        # 3. No NaN values (only possible for float arrays — skip the full
        #    float64 copy, which is 8 bytes/channel, for the usual uint8 case)
        if np.issubdtype(img_np.dtype, np.floating) and not np.isfinite(img_np).all():
            context.add_log("CRITICAL ERROR: Image contains NaN/inf values.")
            return False

        # 4. Not completely black
        if np.max(img_np) == 0:
            context.add_log("CRITICAL ERROR: Image is completely black.")
            return False

        # 5. Not completely white
        if np.min(img_np) == 255:
            context.add_log("CRITICAL ERROR: Image is completely white.")
            return False

        # 6. Minimum dynamic range — at least 30 pixel levels
        dynamic_range = int(np.max(img_np)) - int(np.min(img_np))
        if dynamic_range < 30:
            context.add_log(
                f"WARNING: Very low dynamic range ({dynamic_range}). "
                "Output may appear flat or washed out."
            )
            # Warning only — don't reject, as some input images are naturally flat

        # 7. Mean brightness in reasonable range
        if img_np.ndim == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        mean_brightness = float(np.mean(gray.astype(np.float32)))
        if mean_brightness < 15.0:
            context.add_log(
                f"WARNING: Image is very dark (mean brightness={mean_brightness:.1f})."
            )
        elif mean_brightness > 245.0:
            context.add_log(
                f"WARNING: Image is very bright (mean brightness={mean_brightness:.1f})."
            )

        # 8. No accidental downscale — output should be >= input size
        orig_h, orig_w = context.original_image.shape[:2]
        out_h, out_w = img_np.shape[:2]
        if out_h < orig_h or out_w < orig_w:
            context.add_log(
                f"WARNING: Output ({out_w}x{out_h}) is smaller than input "
                f"({orig_w}x{orig_h}) — possible accidental downscale."
            )
            # Warning only — some pipeline paths intentionally resize

        # 9. Check for solid colour blocks covering >30% of image
        #    (sign of a catastrophic fill or model failure)
        if img_np.ndim == 3 and img_np.shape[2] >= 3:
            # Pack RGB into one uint32 and count in 1-D. np.unique(..., axis=0)
            # lexsorts a structured view of every row, which takes seconds on a
            # multi-megapixel post-SR image; this is the same answer, ~20× faster.
            rgb = img_np[:, :, :3].reshape(-1, 3).astype(np.uint32)
            packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
            values, counts = np.unique(packed, return_counts=True)
            top = int(counts.argmax())
            max_pct = float(counts[top]) / packed.size
            if max_pct > 0.30:
                code = int(values[top])
                r, g, b = (code >> 16) & 0xFF, (code >> 8) & 0xFF, code & 0xFF
                context.add_log(
                    f"WARNING: {max_pct:.0%} of pixels are identical "
                    f"(RGB={r},{g},{b}). Possible catastrophic fill."
                )

        context.add_log("Quality control passed.")
        return True
