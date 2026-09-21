"""
debug_logger.py — Per-stage debug image capture for the PixRevive pipeline.

Saves numbered PNG images after every pipeline stage so you can diagnose
exactly where artifacts appear.

Usage:
    Call DebugLogger.save_stage(context, "07_after_face_restore") from any stage.
    Call DebugLogger.save_debug_images(context) at end-of-pipeline for final summary.

Output directory: debug_output/ (relative to CWD, i.e. backend/)
"""

import os
import time
import json
import cv2
import numpy as np
from ai.pipeline.pipeline_context import PipelineContext

_DEBUG_ENV_VAR = "PIXREVIVE_DEBUG"

def _debug_enabled() -> bool:
    """Debug mode is on when PIXREVIVE_DEBUG env var is set to 1/true/yes.
    
    Default is OFF (0) in production. Set PIXREVIVE_DEBUG=1 to enable
    per-stage debug image snapshots.
    """
    val = os.environ.get(_DEBUG_ENV_VAR, "0").lower()
    return val in ("1", "true", "yes", "on")


class DebugLogger:
    DEBUG_DIR = "debug_output"

    @staticmethod
    def _debug_enabled_flag() -> bool:
        """Public accessor for debug mode — used by pipeline for conditional steps."""
        return _debug_enabled()

    # ── Low-level helpers ──────────────────────────────────────────────────────

    @classmethod
    def _ensure_dir(cls) -> None:
        os.makedirs(cls.DEBUG_DIR, exist_ok=True)

    @classmethod
    def _save_rgb(cls, name: str, img: np.ndarray, context: "PipelineContext") -> None:
        """Save an RGB uint8 numpy array as a BGR PNG."""
        try:
            path = os.path.join(cls.DEBUG_DIR, name)
            cv2.imwrite(path, cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
        except Exception as exc:
            context.add_log(f"DebugLogger: could not save {name}: {exc}")

    @classmethod
    def _save_mask(cls, name: str, mask: np.ndarray, context: "PipelineContext") -> None:
        try:
            path = os.path.join(cls.DEBUG_DIR, name)
            cv2.imwrite(path, mask)
        except Exception as exc:
            context.add_log(f"DebugLogger: could not save {name}: {exc}")

    # ── Per-stage capture ──────────────────────────────────────────────────────

    @classmethod
    def save_stage(cls, context: "PipelineContext", stage_name: str) -> None:
        """
        Save the current working_image as a numbered stage PNG and
        save a false-color chroma-delta map (|Δa*| + |Δb*|) vs the previous stage.

        Filename format: {stage_name}_{image_slug}.png and {stage_name}_chromadelta_{image_slug}.png
        """
        if not _debug_enabled():
            return
        cls._ensure_dir()
        # Derive a clean image slug from metadata
        slug = (
            context.image_metadata.get("case_name")
            or context.image_metadata.get("filename", "img")
        )
        # Strip extension and sanitize: keep only alphanum + underscore
        slug = os.path.splitext(slug)[0]
        slug = "".join(c if c.isalnum() or c == "_" else "_" for c in slug)
        slug = slug.strip("_")[:32]  # cap length
        filename = f"{stage_name}_{slug}.png"
        h, w = context.working_image.shape[:2]
        context.add_log(f"[DEBUG] Stage snapshot: {filename} ({w}x{h})")
        cls._save_rgb(filename, context.working_image, context)

        # Chroma-delta map vs previous stage snapshot (clamped to max 2048 for memory safety)
        curr_for_debug = context.working_image
        if max(h, w) > 2048:
            sc = 2048.0 / max(h, w)
            curr_for_debug = cv2.resize(curr_for_debug, (max(1, round(w * sc)), max(1, round(h * sc))), interpolation=cv2.INTER_AREA)

        last_img = getattr(context, "_last_debug_image", None)
        if last_img is not None:
            try:
                curr_lab = cv2.cvtColor(curr_for_debug, cv2.COLOR_RGB2LAB).astype(np.float32)
                prev_img = last_img
                if prev_img.shape[:2] != curr_for_debug.shape[:2]:
                    prev_img = cv2.resize(prev_img, (curr_for_debug.shape[1], curr_for_debug.shape[0]), interpolation=cv2.INTER_NEAREST)
                prev_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB).astype(np.float32)

                delta_chroma = np.abs(curr_lab[:, :, 1] - prev_lab[:, :, 1]) + np.abs(curr_lab[:, :, 2] - prev_lab[:, :, 2])
                # Scale so delta 30 maps to full scale 255
                delta_norm = np.clip(delta_chroma * (255.0 / 30.0), 0, 255).astype(np.uint8)
                delta_colored_bgr = cv2.applyColorMap(delta_norm, cv2.COLORMAP_TURBO)
                delta_colored_rgb = cv2.cvtColor(delta_colored_bgr, cv2.COLOR_BGR2RGB)
                delta_filename = f"{stage_name}_chromadelta_{slug}.png"
                cls._save_rgb(delta_filename, delta_colored_rgb, context)
            except Exception as exc:
                context.add_log(f"DebugLogger: chroma-delta calculation failed: {exc}")

        context._last_debug_image = curr_for_debug.copy()

    # ── End-of-pipeline summary ────────────────────────────────────────────────

    @classmethod
    def save_debug_images(cls, context: "PipelineContext") -> None:
        """
        Save a full debug snapshot at end-of-pipeline:
          01_original.png
          02_enhanced_mask.png
          03_tearmark_mask.png
          04_final_mask.png
          05_final_working.png
          06_timings.txt
          07_full_log.txt
          08_metadata.json
          and all registered context.debug_masks.
        """
        if not _debug_enabled():
            return
        cls._ensure_dir()

        # ── 01: original input ─────────────────────────────────────────────────
        cls._save_rgb("01_original.png", context.original_image, context)

        # ── 02: raw damage mask from the enhanced detector ─────────────────────
        if "enhanced" in context.damage_masks:
            cls._save_mask("02_enhanced_mask.png", context.damage_masks["enhanced"], context)

        # ── 03: tearmark mask ──────────────────────────────────────────────────
        if "tearmark" in context.damage_masks:
            cls._save_mask("03_tearmark_mask.png", context.damage_masks["tearmark"], context)

        # ── 04: final fused mask ───────────────────────────────────────────────
        if "final" in context.damage_masks:
            cls._save_mask("04_final_mask.png", context.damage_masks["final"], context)

        # ── Spatial debug masks ────────────────────────────────────────────────
        debug_masks = getattr(context, "debug_masks", {})
        for m_name, m_arr in debug_masks.items():
            if m_arr is not None:
                if m_arr.dtype == bool:
                    m_u8 = (m_arr.astype(np.uint8) * 255)
                elif np.issubdtype(m_arr.dtype, np.floating):
                    m_u8 = (np.clip(m_arr, 0.0, 1.0) * 255).astype(np.uint8)
                else:
                    m_u8 = m_arr.astype(np.uint8)
                cls._save_mask(f"mask_{m_name}.png", m_u8, context)

        # ── 05: final working image ────────────────────────────────────────────
        cls._save_rgb("05_final_working.png", context.working_image, context)

        # ── 06: stage timing summary ───────────────────────────────────────────
        if context.stage_timings:
            lines = ["=== Pipeline Stage Timings ==="]
            total = 0.0
            for stage, elapsed in context.stage_timings.items():
                lines.append(f"  {stage:<35} {elapsed:.3f}s")
                total += elapsed
            lines.append(f"  {'TOTAL':<35} {total:.3f}s")
            try:
                with open(os.path.join(cls.DEBUG_DIR, "06_timings.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
            except Exception as exc:
                context.add_log(f"DebugLogger: could not save timings: {exc}")

        # ── 07: full processing log ────────────────────────────────────────────
        try:
            with open(os.path.join(cls.DEBUG_DIR, "07_full_log.txt"), "w", encoding="utf-8") as f:
                for line in context.processing_log:
                    f.write(line + "\n")
        except Exception as exc:
            context.add_log(f"DebugLogger: could not save log: {exc}")

        # ── 08: metadata JSON ─────────────────────────────────────────────────
        try:
            safe_meta = {}
            for k, v in context.image_metadata.items():
                try:
                    json.dumps(v)
                    safe_meta[k] = v
                except (TypeError, ValueError):
                    safe_meta[k] = str(v)
            with open(os.path.join(cls.DEBUG_DIR, "08_metadata.json"), "w", encoding="utf-8") as f:
                json.dump(safe_meta, f, indent=2)
        except Exception as exc:
            context.add_log(f"DebugLogger: could not save metadata: {exc}")

        context.add_log(f"[DEBUG] Full debug snapshot saved to '{cls.DEBUG_DIR}/'")
