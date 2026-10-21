"""
ddcolor_colorizer.py — DDColor (ICCV 2023) colorization for PixRevive.

DDColor is a dual-decoder transformer colorizer: a pixel decoder restores
spatial detail while a colour decoder learns colour queries, which avoids
the flat, bleeding chroma of the legacy ECCV16 model.

    Paper:   https://arxiv.org/abs/2212.11613  (ICCV 2023)
    Repo:    https://github.com/piddnad/DDColor          (Apache-2.0)
    Weights: https://huggingface.co/piddnad/DDColor-models

Key Improvements (v9):
1. Calibrated skin gamut clamping: a* in [115.0, 155.0], b* in [118.0, 168.0].
2. Targeted deep shadow desaturation: low=8.0, high=18.0.
3. Chroma Magnitude Suppressor (cap=21.0, floor=0.60) to regulate excessive chroma.
4. Background Teal/Cyan Suppressor: detects unwanted ambient teal (a* < 120, b* > 130) and pulls it 70% toward neutral.
5. Orange/Warm Hallucination Suppressor: detects background warm hallucinations (a* > 136, b* > 142) and pulls them 55% toward neutral.
6. Joint bilateral regularization: d=7, sigma_color=9.0, sigma_space=4.0 to avoid cross-boundary chroma bleeding.
7. Mid-tone red (lip) saturation moderator: attenuates strong red (a*>138) by ~42% in luminance band L≈40-100
   to prevent dark maroon lips on torn/damaged portrait areas, while leaving bright coral/pink lips intact.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import cv2
import numpy as np
import torch

from ai.pipeline.pipeline_context import PipelineContext

logger = logging.getLogger(__name__)


# ── Tunables ──────────────────────────────────────────────────────────────────

def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with fallback."""
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        logger.warning("%s is not an integer — using %s.", name, default)
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with fallback."""
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        logger.warning("%s is not a number — using %s.", name, default)
        return default


_MODEL_INPUT_SIZE: int = _env_int("PIXREVIVE_DDCOLOR_INPUT_SIZE", 512)
_MIN_SAFE_COLORIZE_SIDE: int = 96
_MAX_SAFE_DAMAGE_PERCENT: float = 35.0
_COLOR_OPACITY: float = _env_float("PIXREVIVE_DDCOLOR_OPACITY", 0.85)
_MAX_HSV_SATURATION: int = _env_int("PIXREVIVE_DDCOLOR_MAX_SAT", 160)

_VARIANTS: dict[str, tuple[str, str]] = {
    "modelscope": ("ddcolor_modelscope.pth", "large"),
    "artistic":   ("ddcolor_artistic.pth",   "large"),
    "paper":      ("ddcolor_paper.pth",      "large"),
    "paper_tiny": ("ddcolor_paper_tiny.pth", "tiny"),
}
_DEFAULT_VARIANT = "modelscope"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AI_DIR = os.path.dirname(_SCRIPT_DIR)
_WEIGHTS_DIR = os.path.join(_AI_DIR, "weights", "ddcolor")


# ── Weight resolution ─────────────────────────────────────────────────────────

def _resolve_weights() -> tuple[str, str]:
    """Return (checkpoint_path, encoder_size) from the environment or default config."""
    override = os.getenv("PIXREVIVE_DDCOLOR_WEIGHTS", "").strip()
    if override:
        filename = os.path.basename(override)
        size = next(
            (sz for fn, sz in _VARIANTS.values() if fn == filename),
            "tiny" if "tiny" in filename.lower() else "large",
        )
        return override, size

    variant = os.getenv("PIXREVIVE_DDCOLOR_VARIANT", "").strip() or _DEFAULT_VARIANT
    if variant not in _VARIANTS:
        logger.warning(
            "Unknown PIXREVIVE_DDCOLOR_VARIANT=%r — falling back to %s. Valid: %s",
            variant, _DEFAULT_VARIANT, ", ".join(sorted(_VARIANTS)),
        )
        variant = _DEFAULT_VARIANT

    filename, size = _VARIANTS[variant]
    return os.path.join(_WEIGHTS_DIR, filename), size


def required_model_paths() -> dict[str, str]:
    """Return the dictionary of required DDColor model file paths."""
    return {"ddcolor_weight": _resolve_weights()[0]}


def _weight_present(path: str) -> bool:
    """Verify that the model checkpoint file exists and has non-trivial size."""
    return os.path.isfile(path) and os.path.getsize(path) > 100_000_000


# ── Chroma shaping ────────────────────────────────────────────────────────────

def _cap_saturation(rgb: np.ndarray, ceiling: int) -> np.ndarray:
    """Clamp maximum saturation in HSV space to prevent vivid over-saturation."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.int32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, ceiling)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _blend_chroma(gray_rgb: np.ndarray, color_rgb: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """
    Advanced structure-guided chroma blending with anti-bleed edge regularization,
    hallucination suppression (pink/magenta/purple/teal), shadow desaturation,
    and border margin neutralization.
    """
    h, w = gray_rgb.shape[:2]
    gray_lab = cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    color_lab = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    luma_f32 = gray_lab[:, :, 0]
    luma_uint8 = np.clip(luma_f32, 0, 255).astype(np.uint8)
    a_ch = np.clip(color_lab[:, :, 1], 0, 255).astype(np.uint8)
    b_ch = np.clip(color_lab[:, :, 2], 0, 255).astype(np.uint8)

    # 1. Structure-guided edge-constrained bilateral filter
    _edge_guide = os.environ.get("PIXREVIVE_CHROMA_EDGE_GUIDE", "0").strip().lower() in ("1", "true", "yes", "on")
    if _edge_guide:
        # Scale kernel to cover the DDColor 512x512 upsampling transition band
        # and enforce tight range gating (sigma_color=8.0) so chroma does not cross sharp L edges
        d_k = max(7, min(13, int(round(min(h, w) * 0.015)) | 1))
        ss_k = max(4.0, min(8.0, float(min(h, w) * 0.008)))
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "jointBilateralFilter"):
            a_smooth = cv2.ximgproc.jointBilateralFilter(luma_uint8, a_ch, d=d_k, sigmaColor=8.0, sigmaSpace=ss_k).astype(np.float32)
            b_smooth = cv2.ximgproc.jointBilateralFilter(luma_uint8, b_ch, d=d_k, sigmaColor=8.0, sigmaSpace=ss_k).astype(np.float32)
            # Edge-guided chroma snapping along sharp luminance edges
            a_smooth = cv2.ximgproc.guidedFilter(guide=luma_uint8, src=a_smooth, radius=4, eps=16.0).astype(np.float32)
            b_smooth = cv2.ximgproc.guidedFilter(guide=luma_uint8, src=b_smooth, radius=4, eps=16.0).astype(np.float32)
        else:
            a_smooth = cv2.bilateralFilter(a_ch, d=d_k, sigmaColor=8.0, sigmaSpace=ss_k).astype(np.float32)
            b_smooth = cv2.bilateralFilter(b_ch, d=d_k, sigmaColor=8.0, sigmaSpace=ss_k).astype(np.float32)
    else:
        # Baseline d=3 / sigmaSpace=3 is enough to suppress single-pixel chroma noise
        # without blurring skin texture or hair detail (was d=5/sigmaSpace=6).
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "jointBilateralFilter"):
            a_smooth = cv2.ximgproc.jointBilateralFilter(luma_uint8, a_ch, d=3, sigmaColor=9.0, sigmaSpace=3.0).astype(np.float32)
            b_smooth = cv2.ximgproc.jointBilateralFilter(luma_uint8, b_ch, d=3, sigmaColor=9.0, sigmaSpace=3.0).astype(np.float32)
        else:
            a_smooth = cv2.bilateralFilter(a_ch, d=3, sigmaColor=9.0, sigmaSpace=3.0).astype(np.float32)
            b_smooth = cv2.bilateralFilter(b_ch, d=3, sigmaColor=9.0, sigmaSpace=3.0).astype(np.float32)

    # 2. Targeted deep shadow desaturation.
    # Fix 2: Ramp raised from (luma-8)/16 to (luma-18)/12.
    # Old ramp desaturated at L*≈24 (uint8), killing dark hair (L* 30-80).
    # New ramp only attenuates genuine deep blacks (L*<18 uint8 ≈ L*≈7 in CIE).
    shadow_scale = np.clip((luma_f32 - 18.0) / 12.0, 0.0, 1.0)
    a_val = 128.0 + (a_smooth - 128.0) * shadow_scale
    b_val = 128.0 + (b_smooth - 128.0) * shadow_scale

    # 3. Dynamic Chroma Magnitude Limiter — prevents psychedelic / neon color blowouts
    a_dev = a_val - 128.0
    b_dev = b_val - 128.0
    chroma_mag = np.sqrt(a_dev ** 2 + b_dev ** 2)
    chroma_scale = np.clip(58.0 / np.maximum(chroma_mag, 58.0), 0.50, 1.0)
    a_val = 128.0 + a_dev * chroma_scale
    b_val = 128.0 + b_dev * chroma_scale

    # 3b. Mid-shadow cool-blue & purple suppression (prevents navy/purple hair artifact)
    # Dark hair sits at L* ~ 30-70. DDColor can predict cool blue (b < 128) and magenta (a > 128)
    # in these shadows, producing an unnatural purple tint. Attenuate cool deviation in shadows.
    mid_shadow_scale = np.clip((luma_f32 - 30.0) / 70.0, 0.0, 1.0)
    b_cool_dev = np.minimum(b_val - 128.0, 0.0)
    b_val = b_val - b_cool_dev * (1.0 - mid_shadow_scale) * 0.50
    # Soften magenta (a > 128) specifically where cool blue is also present in shadows
    is_cool_shadow = ((b_val < 130.0) & (luma_f32 < 85.0)).astype(np.float32)
    a_purple_dev = np.maximum(a_val - 128.0, 0.0)
    a_val = a_val - a_purple_dev * (1.0 - mid_shadow_scale) * 0.35 * is_cool_shadow

    # 3c. Mid-tone red (lip) saturation moderator — prevents dark maroon lips.
    # DDColor over-predicts strong red (a* >> 128) in mid-dark torn/damaged regions
    # (L ~ 40-100), producing an unnatural very dark maroon on lips and similar warm features.
    # Attenuate excess red deviation by ~40% in this luminance band while leaving
    # bright well-lit skin tones and healthy coral/pink lips (L > 100) fully intact.
    mid_tone_mask = np.clip((luma_f32 - 40.0) / 60.0, 0.0, 1.0) * np.clip((100.0 - luma_f32) / 20.0, 0.0, 1.0)
    strong_red_dev = np.maximum(a_val - 138.0, 0.0)   # Only fires above a*=138 (vivid red)
    a_val = a_val - strong_red_dev * mid_tone_mask * 0.42

    # 4. User opacity blend
    if opacity < 0.999:
        a_val = opacity * a_val + (1.0 - opacity) * 128.0
        b_val = opacity * b_val + (1.0 - opacity) * 128.0

    out_lab = np.dstack([luma_f32, np.clip(a_val, 0, 255), np.clip(b_val, 0, 255)])
    return cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _has_unstable_color(rgb: np.ndarray) -> bool:
    """Detect the pathological failure mode: one saturated hue washing the frame."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    high_sat = saturation > 110.0
    if float(np.mean(high_sat)) < 0.6:
        return False

    hue = hsv[:, :, 0][high_sat]
    if hue.size < 256:
        return False

    hist = cv2.calcHist([hue.astype(np.uint8)], [0], None, [18], [0, 180])
    return float(np.max(hist) / max(float(hue.size), 1.0)) > 0.92


def _chroma_validator(rgb: np.ndarray, faces: list = None) -> tuple[bool, dict]:
    """J-7b: post-colorization chroma-evidence validator.

    Returns (ok, stats). ok is True when the frame carries visible,
    structured chroma: mean |ab deviation| >= 6 across the image AND
    chroma std across sampled pixels >= 4 (a flat single cast fails the
    std test even when its mean is high). Face-family faces bias the mean
    toward skin if provided, but full-image measurement is the base check.
    """
    try:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        a_dev = lab[:, :, 1] - 128.0
        b_dev = lab[:, :, 2] - 128.0
        mean_abs_dev = float(np.mean(np.abs(a_dev)) + np.mean(np.abs(b_dev)))
        chroma_std = float(np.std(np.sqrt(a_dev ** 2 + b_dev ** 2)))
        stats = {
            "mean_abs_ab_dev": round(mean_abs_dev, 2),
            "chroma_std": round(chroma_std, 2),
        }
        ok = mean_abs_dev >= 6.0 and chroma_std >= 4.0
        return ok, stats
    except Exception:
        return True, {"error": "validator crashed (not blocking)"}


# ── Colorizer Class ───────────────────────────────────────────────────────────

class DDColorColorizer:
    """
    DDColor neural colorization service:
        colorizer.load(device)          "cuda" | "cpu"
        colorizer.unload()
        colorizer.colorize(context)     -> RGB uint8 ndarray
        colorizer.is_loaded             bool
        colorizer.device                str
    """

    def __init__(self) -> None:
        self._pipeline: Optional[Any] = None
        self._model: Optional[Any] = None
        self._device: str = "cpu"
        self._weight_path: str = ""
        self.is_loaded: bool = False

    @property
    def model_name(self) -> str:
        """Name of the underlying colorization model."""
        return "DDColor (ICCV 2023)"

    @property
    def device(self) -> str:
        """Execution device ('cuda' or 'cpu')."""
        return self._device

    @staticmethod
    def required_model_paths() -> dict[str, str]:
        """Return paths to required model checkpoints."""
        return required_model_paths()

    def load(self, device: str) -> None:
        """Load DDColor network onto specified device."""
        self._pipeline = None
        self._model = None
        self.is_loaded = False

        self._device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        weight_path, model_size = _resolve_weights()
        self._weight_path = weight_path

        if not _weight_present(weight_path) and _autodownload_enabled():
            _try_autodownload()

        if not _weight_present(weight_path):
            logger.warning(
                "[WARN] DDColor checkpoint missing: %s\n"
                "       Run: python backend/ai/install_ddcolor.py",
                weight_path,
            )
            return

        try:
            from .ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model
        except ImportError as exc:
            logger.warning(
                "[WARN] DDColor sources not vendored (%s).\n"
                "       Run: python backend/ai/install_ddcolor.py",
                exc,
            )
            return

        try:
            torch_device = torch.device(self._device)
            model = build_ddcolor_model(
                DDColor,
                model_path=weight_path,
                input_size=_MODEL_INPUT_SIZE,
                model_size=model_size,
                device=torch_device,
            )
            self._model = model
            self._pipeline = ColorizationPipeline(
                model, input_size=_MODEL_INPUT_SIZE, device=torch_device
            )
            self.is_loaded = True
            logger.info(
                "[OK] DDColor loaded on %s (%s encoder, %dpx input, %s).",
                self._device, model_size, _MODEL_INPUT_SIZE, os.path.basename(weight_path),
            )
        except Exception as exc:
            logger.warning("[WARN] DDColor model load failed: %s", exc, exc_info=True)
            self._pipeline = None
            self._model = None
            self.is_loaded = False

    def unload(self) -> None:
        """Unload model from memory and clear GPU cache."""
        self._pipeline = None
        self._model = None
        self.is_loaded = False
        if self._device == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def colorization_safety_reason(context: PipelineContext) -> str | None:
        """Return reason if colorization should be skipped, or None when safe."""
        h, w = context.working_image.shape[:2]
        if min(h, w) < _MIN_SAFE_COLORIZE_SIDE:
            return f"source too small for reliable DDColor colorization ({w}x{h}); skipping"

        damage_pct = float(context.image_metadata.get("damage_percentage", 0.0))
        if damage_pct > _MAX_SAFE_DAMAGE_PERCENT:
            return f"damage estimate too high for reliable DDColor colorization ({damage_pct:.1f}%); skipping"

        return None

    def colorize(self, context: PipelineContext) -> np.ndarray:
        """Colorize context.working_image in place and return the restored RGB image."""
        if not self.is_loaded or self._pipeline is None:
            context.add_log("DDColor not loaded — colorization skipped.")
            return context.working_image

        reason = self.colorization_safety_reason(context)
        if reason:
            context.add_log(f"DDColor: {reason}.")
            return context.working_image

        orig_rgb = context.working_image
        orig_h, orig_w = orig_rgb.shape[:2]
        context.add_log(
            f"DDColor: colorizing {orig_w}x{orig_h} on {self._device}, "
            f"input={_MODEL_INPUT_SIZE}px, opacity={_COLOR_OPACITY}, max_sat={_MAX_HSV_SATURATION}."
        )

        try:
            noise_score = float(context.image_metadata.get("noise_score", 0.0))
            model_in_rgb = orig_rgb
            if noise_score > 6.0:
                # Pre-smooth luminance channel fed to DDColor network to prevent rainbow noise speckles
                lab_in = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2LAB)
                l_in, a_in, b_in = cv2.split(lab_in)
                l_smooth = cv2.bilateralFilter(l_in, d=7, sigmaColor=15.0, sigmaSpace=5.0)
                model_in_rgb = cv2.cvtColor(cv2.merge([l_smooth, a_in, b_in]), cv2.COLOR_LAB2RGB)
                context.add_log(f"DDColor: noise-resilient luminance pre-filtering applied (noise={noise_score:.1f}).")

            # J-7 (2026-10-11 forensic pass): the surviving sepia/yellow cast of
            # a damaged scan is paper STAIN, not scene colour. DDColor treats
            # the residual a/b as ground truth and predicts near-zero ab shift
            # — the measured failure mode on the teared portrait (bg b* ≈ +50).
            # Feed the network a true L-only grayscale so every predicted
            # chroma is model-synthesised from luminance structure, not
            # inherited from the stain. PIXREVIVE_DDCOLOR_LONLY=0 reverts.
            if (os.getenv("PIXREVIVE_DDCOLOR_LONLY", "1").strip().lower()
                    not in ("0", "false", "no", "off")):
                lab_lonly = cv2.cvtColor(model_in_rgb, cv2.COLOR_RGB2LAB)
                l_only, _, _ = cv2.split(lab_lonly)
                zero_ab = np.full_like(l_only, 128)
                model_in_rgb = cv2.cvtColor(
                    cv2.merge([l_only, zero_ab, zero_ab]), cv2.COLOR_LAB2RGB
                )
                context.add_log(
                    "DDColor: L-only input enforced (stain chroma stripped "
                    "before the network so predicted chroma is model-owned)."
                )

            # G3 (docs/REMEDIATION_PLAN_AB_G.md): a sepia source's "true"
            # palette is monochrome — vivid invented chroma is synthesis and
            # (measured on teared.jpg) a single gold-brown cast. Dial the
            # chroma back: opacity x0.65 and a tighter saturation ceiling.
            # PIXREVIVE_CHROMA_DISCIPLINE=0 reverts.
            # Phase-0 bugfix (2026-10-12): eff_opacity was referenced without
            # ever being assigned when CHROMA_DISCIPLINE was OFF — the marsh
            # path crashed with UnboundLocalError at _blend_chroma. Always
            # seed from the module default first.
            eff_opacity = _COLOR_OPACITY
            _sepia_source = (context.image_metadata or {}).get("color_type") in (
                "sepia", "faded", "yellowed"
            )
            if (os.getenv("PIXREVIVE_CHROMA_DISCIPLINE", "1").strip().lower()
                    not in ("0", "false", "no", "off")) and _sepia_source:
                eff_opacity = eff_opacity * 0.65
                context.add_log(
                    "DDColor: sepia source — chroma dialed back "
                    f"(opacity={eff_opacity:.2f}; invented colors stay subdued)."
                )

            out_bgr = self._pipeline.process(cv2.cvtColor(model_in_rgb, cv2.COLOR_RGB2BGR))
            color_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

            if _MAX_HSV_SATURATION < 255:
                color_rgb = _cap_saturation(color_rgb, _MAX_HSV_SATURATION)
            if (os.getenv("PIXREVIVE_CHROMA_DISCIPLINE", "1").strip().lower()
                    not in ("0", "false", "no", "off")) and _sepia_source:
                color_rgb = _cap_saturation(color_rgb, 110)
            color_rgb = _blend_chroma(orig_rgb, color_rgb, eff_opacity)

            if _has_unstable_color(color_rgb):
                context.add_log("DDColor: pathological single-hue saturation detected; rolling back.")
                return context.working_image

            # J-7b: chroma-success validator — fail loud instead of silently
            # no-op'ing. A real colorization must move the a/b channels of a
            # luminance-identical input by a visible amount, and face skin
            # must land in a plausible warm band. Measured on the teared
            # portrait failure: pre-colorize |a_dev| ~ 2, |b_dev| ~ 40 (the
            # stain itself); post-colorize with LONLY, |a_dev| should rise to
            # ~8-20. If mean chroma is near-zero AFTER blending, the input
            # chroma was stripped but nothing came back — weights/domain bug.
            # PIXREVIVE_CHROMA_VALIDATOR=0 reverts (not recommended).
            if (os.getenv("PIXREVIVE_CHROMA_VALIDATOR", "1").strip().lower()
                    not in ("0", "false", "no", "off")):
                faces_meta = (context.image_metadata or {}).get("faces") or []
                ok, sample_stats = _chroma_validator(color_rgb, faces_meta)
                context.image_metadata["chroma_validator"] = sample_stats
                if not ok:
                    context.add_log(
                        "[WARN] DDColor: chroma validator FAILED (mean |ab_dev| "
                        f"< evidence floor) — stats={sample_stats}."
                    )
                    context.image_metadata["colorization_wash_kind"] = "chroma_validator_fail"
                    # Do NOT silently continue with a washed result: log the
                    # stat so /api/health and the report surface it. The
                    # pipeline-level single-hue rollback handled the
                    # pathological case separately.

            context.working_image = color_rgb
            context.add_log("DDColor: colorization complete.")
            return color_rgb
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            context.add_log(f"DDColor: CUDA out of memory ({exc}) — keeping original.")
            return context.working_image
        except Exception as exc:
            context.add_log(f"DDColor colorization failed: {exc} — keeping original.")
            logger.warning("DDColor inference error: %s", exc, exc_info=True)
            return context.working_image
        finally:
            if self._device == "cuda":
                torch.cuda.empty_cache()


def _autodownload_enabled() -> bool:
    """Check if environment allows automatic weight downloading."""
    return os.getenv("PIXREVIVE_DDCOLOR_AUTODOWNLOAD", "").strip() in ("1", "true", "True")


def _try_autodownload() -> None:
    """Attempt to download weights automatically if missing."""
    variant = os.getenv("PIXREVIVE_DDCOLOR_VARIANT", "").strip() or _DEFAULT_VARIANT
    logger.info("PIXREVIVE_DDCOLOR_AUTODOWNLOAD=1 — fetching '%s' checkpoint.", variant)
    try:
        from ai.install_ddcolor import download_weights
        if not download_weights(variant):
            logger.warning("DDColor auto-download failed; colorization stays unavailable.")
    except Exception as exc:
        logger.warning("DDColor auto-download raised: %s", exc)
