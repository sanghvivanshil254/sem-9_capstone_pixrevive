from __future__ import annotations

import time
import cv2
import numpy as np
from typing import Dict, Any, Optional


class PipelineContext:
    def __init__(self, original_image: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """
        Args:
            original_image: The source image in RGB format (numpy array).
            metadata: Additional info like original filename, etc.
        """
        # Original input data (never modified)
        self.original_image: np.ndarray = original_image

        # The current working state of the image (RGB numpy array)
        self.working_image: np.ndarray = original_image.copy()

        # Analysis results (quality scores, resolution, etc.)
        self.image_metadata: Dict[str, Any] = metadata or {}

        # Masks for different damage types
        self.damage_masks: Dict[str, np.ndarray] = {}

        # Spatial masks for debug auditing (registered by passes creating masks)
        self.debug_masks: Dict[str, np.ndarray] = {}
        self._last_debug_image: Optional[np.ndarray] = None

        # Intermediate/final outputs from different models
        self.model_outputs: Dict[str, Any] = {}

        # Dynamic pipeline control flags (e.g. deferred passes)
        self.pipeline_flags: Dict[str, Any] = {}

        # Log of processing steps
        self.processing_log: list[str] = ["Pipeline Context initialized."]

        # Per-stage timing (stage_name → elapsed seconds)
        self.stage_timings: Dict[str, float] = {}
        self._timer_starts: Dict[str, float] = {}

        # Orchestrator provenance/transaction state (synthesis_manifest,
        # hyperparameters, begin_transaction, step_down_parameter) moved to
        # experimental/orchestrator with the parked Two-Tier engine (plan 5.2).

    def add_log(self, message: str):
        """Add an entry to the processing log."""
        self.processing_log.append(message)

    def sync_geometry_to_working_image(self) -> None:
        """Rescale damage masks, face boxes, and landmarks to match working_image.

        Stages that change resolution (super-resolution, the 2K resize fallback)
        leave every mask, face bbox, and landmark array at the pre-resize scale.
        Downstream stages combine those masks with working_image pixel-for-pixel,
        so the mismatch either raises a broadcast error in a mask-combining stage
        or is silently swallowed (Harmonizer). Call this immediately after any
        stage that resizes working_image.
        """
        h, w = self.working_image.shape[:2]

        stale = [
            (name, m) for name, m in self.damage_masks.items()
            if m is not None and m.shape[:2] != (h, w)
        ]
        for name, mask in stale:
            resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            # Keep masks strictly binary — feathering happens at the point of use.
            self.damage_masks[name] = ((resized > 0).astype(np.uint8) * 255)

        faces = self.image_metadata.get("faces") or []
        prev = self.image_metadata.get("faces_frame_size")
        if faces and prev and tuple(prev) != (h, w):
            prev_h, prev_w = prev
            sx = w / max(prev_w, 1)
            sy = h / max(prev_h, 1)
            for face in faces:
                bbox = face.get("bbox")
                if bbox and len(bbox) >= 4:
                    x0, y0, x1, y1 = bbox[:4]
                    face["bbox"] = [
                        max(0, min(w, int(round(x0 * sx)))),
                        max(0, min(h, int(round(y0 * sy)))),
                        max(0, min(w, int(round(x1 * sx)))),
                        max(0, min(h, int(round(y1 * sy)))),
                    ]

                landmarks = face.get("landmarks")
                if landmarks is not None and len(landmarks) > 0:
                    scaled_lms = []
                    for pt in landmarks:
                        if len(pt) >= 2:
                            lx, ly = pt[:2]
                            scaled_lms.append([
                                max(0, min(w, int(round(lx * sx)))),
                                max(0, min(h, int(round(ly * sy)))),
                            ])
                    if isinstance(landmarks, np.ndarray):
                        face["landmarks"] = np.array(scaled_lms, dtype=landmarks.dtype)
                    else:
                        face["landmarks"] = scaled_lms

        if faces:
            self.image_metadata["faces_frame_size"] = (h, w)

        if stale:
            names = ", ".join(name for name, _ in stale)
            self.add_log(f"Geometry sync: rescaled masks [{names}] to {w}x{h}.")

    def start_timer(self, stage_name: str):
        """Record the start time for a named pipeline stage."""
        self._timer_starts[stage_name] = time.perf_counter()

    def end_timer(self, stage_name: str):
        """Record the elapsed time for a named stage and log it."""
        start = self._timer_starts.pop(stage_name, None)
        if start is not None:
            elapsed = time.perf_counter() - start
            self.stage_timings[stage_name] = round(elapsed, 3)
            self.add_log(f"[TIMING] {stage_name}: {elapsed:.3f}s")


