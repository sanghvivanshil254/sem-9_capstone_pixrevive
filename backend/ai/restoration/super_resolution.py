"""
super_resolution.py — RealESRGAN ×4 with proper tile inference.

Key fixes vs. previous version:
  - Removed the destructive MAX_INPUT_PIXELS=512×512 hard cap that was
    downscaling images before upscaling, destroying all fine detail.
  - Replaced with a true pad-and-tile forward pass (tile_size=256, tile_pad=32)
    that can handle any input resolution without downscaling.
  - The ×4 OUTPUT is capped at 4096×4096 (the tile buffer is preallocated at
    output size, so capping the input instead let it reach ~3.2 GB).
  - The ×4 model is skipped entirely when the source already meets the target
    long side — the result would only be downscaled again.
  - Final 2K normalisation (longest side → 2048) happens here so downstream
    stages always receive a consistent 2K image.
"""

import os
import logging
import math
import torch
import numpy as np
from PIL import Image
from ai.models.base_model import BaseRestorer
from ai.pipeline.pipeline_context import PipelineContext

# ─── 2K output targets ────────────────────────────────────────────────────────
_2K_LONG_SIDE = 2048    # longest side in pixels
_MIN_LONG_SIDE = 1024   # never return smaller than this


def _configured_long_side() -> int | None:
    """Optional output-resolution override via PIXREVIVE_TARGET_LONG_SIDE.

    Regulated: values >4096 are clamped to 4096 (the hard ceiling is
    _MAX_SR_OUTPUT_PIXELS anyway) and invalid values fall back to the
    default 2K behaviour.
    """
    raw = os.environ.get("PIXREVIVE_TARGET_LONG_SIDE", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value < 256:
        return None
    return min(value, 4096)


def _configured_long_side() -> int | None:
    """Optional output-resolution override via PIXREVIVE_TARGET_LONG_SIDE.

    Regulated: values >4096 are clamped (солютly capped by _MAX_SR_OUTPUT_PIXELS
    anyway) and invalid values fall back to the default 2K behavior.
    """
    raw = os.environ.get("PIXREVIVE_TARGET_LONG_SIDE", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value < 256:
        return None
    return min(value, 4096)


def _realesrgan_tile_forward(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    tile_size: int = 256,
    tile_pad: int = 32,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Process a tensor through `model` in overlapping tiles to avoid OOM.

    Parameters
    ----------
    model       : RRDBNet (eval mode, on `device`)
    img_tensor  : float32 BCHW in [0, 1], on CPU (moved per tile)
    tile_size   : spatial size of each tile (pixels, before padding)
    tile_pad    : overlap padding on each side
    device      : target device for inference

    Returns
    -------
    float32 BCHW tensor, ×4 spatial dimensions, on CPU, clamped [0, 1]
    """
    batch, channel, height, width = img_tensor.shape
    output_h = height * 4
    output_w = width * 4
    output = torch.zeros((batch, channel, output_h, output_w), dtype=torch.float32)
    # A4 (docs/REMEDIATION_PLAN_AB_G.md): accumulate overlapping tiles with a
    # cosine-feathered weight instead of replace-style writes — adjacent tiles
    # see different padding context and disagree slightly; replace-writes turn
    # that disagreement into hard tile seams (visible as grid blocks on
    # posterized zones). PIXREVIVE_SEAM_QUALITY=0 reverts to hard writes.
    _seam_quality = os.environ.get("PIXREVIVE_SEAM_QUALITY", "1").strip().lower() not in ("0", "false", "no", "off")
    acc = torch.zeros_like(output) if _seam_quality else None
    wsum = torch.zeros((1, 1, output_h, output_w), dtype=torch.float32) if _seam_quality else None

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    for iy in range(tiles_y):
        for ix in range(tiles_x):
            # Source tile bounds (without padding)
            x1 = ix * tile_size
            x2 = min(x1 + tile_size, width)
            y1 = iy * tile_size
            y2 = min(y1 + tile_size, height)

            # Padded source bounds (for overlap context)
            x1p = max(x1 - tile_pad, 0)
            x2p = min(x2 + tile_pad, width)
            y1p = max(y1 - tile_pad, 0)
            y2p = min(y2 + tile_pad, height)

            tile_in = img_tensor[:, :, y1p:y2p, x1p:x2p].to(device)
            with torch.no_grad():
                tile_out = model(tile_in).clamp(0.0, 1.0).cpu()

            # Destination in output (×4 scale)
            dx1 = x1 * 4
            dx2 = x2 * 4
            dy1 = y1 * 4
            dy2 = y2 * 4

            # Source in tile_out (remove padding on ×4 scale)
            ox1 = (x1 - x1p) * 4
            ox2 = ox1 + (x2 - x1) * 4
            oy1 = (y1 - y1p) * 4
            oy2 = oy1 + (y2 - y1) * 4

            # Full padded tile is blended, not just its core: weight 1 across
            # the core, cosine taper to ~0 only across the PAD ring (which the
            # neighboring tile's core covers). Every output pixel therefore
            # receives weight-1 coverage from at least one tile — the first
            # version tapered to 0 at each tile's CORE edge, so the seams
            # (where the taper met) had zero total weight and rendered as dark
            # cross lines on both remediation fixtures.
            th_p, tw_p = tile_out.shape[2], tile_out.shape[3]
            wy = torch.ones(th_p, dtype=torch.float32)
            wx = torch.ones(tw_p, dtype=torch.float32)
            ramp = tile_pad * 4
            if ramp > 1:
                t = torch.linspace(0.0, 1.0, ramp)
                cos_ramp = 0.5 - 0.5 * torch.cos(t * math.pi)
                if y1p > 0:
                    wy[:ramp] = torch.minimum(wy[:ramp], cos_ramp)
                if y2p < height:
                    wy[-ramp:] = torch.minimum(wy[-ramp:], cos_ramp.flip(0))
                if x1p > 0:
                    wx[:ramp] = torch.minimum(wx[:ramp], cos_ramp)
                if x2p < width:
                    wx[-ramp:] = torch.minimum(wx[-ramp:], cos_ramp.flip(0))
            w2d = wy[:, None] * wx[None, :]
            acc[:, :, y1p * 4:y2p * 4, x1p * 4:x2p * 4] += tile_out[0] * w2d
            wsum[:, :, y1p * 4:y2p * 4, x1p * 4:x2p * 4] += w2d

    if _seam_quality:
        output = acc / torch.clamp(wsum, min=1e-6)

    return output


def _suppress_halftone(img_np: np.ndarray) -> np.ndarray:
    """
    Suppress halftone / mesh / screen-pattern texture from old scanned photos
    BEFORE super-resolution.

    Old photographs printed with halftone screens have a repeating dot or mesh
    pattern at ~5-30 px frequency.  When RealESRGAN upscales these it treats
    the pattern as content and produces a honeycombed texture artifact.

    Strategy: apply a bilateral filter (edge-preserving) only when a periodic
    high-frequency texture is detected across the image.  Faces, edges, and
    damage boundaries are preserved; flat periodic mesh is smoothed.

    Returns the filtered image; if no halftone pattern is detected,
    returns the original unchanged (same object reference).
    """
    import cv2
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # High-frequency energy map
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    hf = np.abs(gray - blurred)

    hf_mean = float(np.mean(hf))
    hf_std  = float(np.std(hf))

    # Halftone heuristic: high mean HF energy AND extremely uniform distribution
    # (energy is evenly distributed = periodic dot screen mesh across the print).
    cv_ratio = hf_std / (hf_mean + 1e-6)
    has_halftone = (hf_mean > 10.0) and (cv_ratio < 0.90)

    if not has_halftone:
        return img_np   # no periodic texture — return original unchanged

    # Bilateral filter: preserves edges, smooths flat periodic regions.
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    filtered_bgr = cv2.bilateralFilter(bgr, d=5, sigmaColor=10, sigmaSpace=5)
    return cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)


def _post_sr_sharpen(img_np: np.ndarray, strength: float = 0.20) -> np.ndarray:
    """Post-super-resolution sharpening in LAB space.

    Sharpens only the L (luminance) channel to add crispness without
    introducing colour fringing.  Uses an unsharp mask with radius=1.5
    so fine detail (hair, eyes, fabric) is enhanced without amplifying
    any residual texture.
    """
    import cv2
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]
    blurred_L = cv2.GaussianBlur(L, (0, 0), sigmaX=1.5)
    L_sharp = np.clip(L + strength * (L - blurred_L), 0, 255)
    lab[:, :, 0] = L_sharp
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def _resize_to_long_side(img_np: np.ndarray, target_long_side: int) -> np.ndarray:
    """Resize so the longest side equals target_long_side."""
    h, w = img_np.shape[:2]
    long_side = max(h, w)
    target_long_side = max(1, target_long_side)

    if long_side == target_long_side:
        return img_np  # already perfect

    scale = target_long_side / long_side
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    pil = Image.fromarray(img_np)
    pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.array(pil)


def _target_long_side_for_source(source_long_side: int) -> int:
    """Avoid inventing detail by over-enlarging tiny sources.

    PIXREVIVE_TARGET_LONG_SIDE (256..4096) overrides the default; tiny-source
    multipliers still apply so small scans are never blown up beyond a 4x
    inventable-detail bound, unless the env override is explicit.
    """
    override = _configured_long_side()
    if override is not None:
        if source_long_side < 256:
            return min(override, max(1, source_long_side * 4))
        return override
    if source_long_side < 256:
        return max(1, source_long_side * 4)
    if source_long_side < 512:
        return min(_2K_LONG_SIDE, max(_MIN_LONG_SIDE, source_long_side * 3))
    return _2K_LONG_SIDE


def _determine_tile_size(device: str) -> int:
    """Choose tile size based on device and available VRAM."""
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            total_vram = torch.cuda.get_device_properties(0).total_memory
            if total_vram >= 8 * 1024 * 1024 * 1024:
                return 512
            return 256
        except Exception:
            return 256
    return 192


# Ceiling on the ×4 OUTPUT tensor. _realesrgan_tile_forward preallocates a
# float32 BCHW buffer of (h*4, w*4), so this is what actually bounds memory:
# 4096² × 3 × 4 B ≈ 201 MB. Capping the *input* at 4096² instead (as this file
# used to) lets that buffer reach 16384² ≈ 3.2 GB and OOM the process.
_MAX_SR_OUTPUT_PIXELS = 4096 * 4096
_SR_SCALE = 4


def _clamp_max_pixels(img_np: np.ndarray, max_pixels: int) -> np.ndarray:
    """Downscale (aspect-preserving) so h*w <= max_pixels."""
    h, w = img_np.shape[:2]
    if h * w <= max_pixels:
        return img_np

    scale = math.sqrt(max_pixels / (h * w))
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    pil = Image.fromarray(img_np)
    pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.array(pil)


def _clamp_for_sr_output(img_np: np.ndarray) -> np.ndarray:
    """Cap the input so the ×4 output stays within _MAX_SR_OUTPUT_PIXELS."""
    return _clamp_max_pixels(img_np, _MAX_SR_OUTPUT_PIXELS // (_SR_SCALE ** 2))


class SuperResolution(BaseRestorer):
    """
    RealESRGAN ×4 super-resolution with tiled inference.

    Works on any input size without pre-downscaling.  Enforces 2K output
    (longest side = 2048 px) after upscaling.
    """

    def __init__(self) -> None:
        self.realesrgan: torch.nn.Module | None = None
        self.is_loaded: bool = False
        self.device: str = "cpu"

    @property
    def model_name(self) -> str:
        return "RealESRGAN x4plus"

    def load(self, device: str) -> None:
        self.device = device
        try:
            from ai.restoration.rrdbnet import RRDBNet

            weights_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "weights", "RealESRGAN_x4plus.pth",
            )

            net = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23, num_grow_ch=32,
            ).to(device)

            if os.path.exists(weights_path):
                sd = torch.load(weights_path, map_location=device, weights_only=True)
                # Handle various checkpoint formats
                if "params_ema" in sd:
                    sd = sd["params_ema"]
                elif "params" in sd:
                    sd = sd["params"]
                net.load_state_dict(sd, strict=True)
                net.eval()
                self.realesrgan = net
                self.is_loaded = True
                logging.getLogger(__name__).info("RealESRGAN x4plus loaded on %s.", device)
            else:
                logging.getLogger(__name__).warning("RealESRGAN weights not found at: %s", weights_path)
                self.realesrgan = None

        except Exception as exc:
            logging.getLogger(__name__).warning("RealESRGAN failed to load: %s", exc)
            self.realesrgan = None

    def unload(self) -> None:
        self.realesrgan = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def restore(self, context: PipelineContext) -> np.ndarray:
        if not self.is_loaded or self.realesrgan is None:
            context.add_log("RealESRGAN not loaded — skipping super-resolution.")
            h, w = context.working_image.shape[:2]
            target_long = _target_long_side_for_source(max(h, w))
            context.working_image = _resize_to_long_side(context.working_image, target_long)
            return context.working_image

        img_np = context.working_image
        orig_h, orig_w = img_np.shape[:2]
        target_long = _target_long_side_for_source(max(orig_h, orig_w))

        # ── Pre-SR: suppress halftone / mesh texture ──────────────────────────
        # Old scanned photos often have a repeating screen pattern that ESRGAN
        # amplifies into a honeycomb artifact.  Bilateral filter suppresses the
        # periodic mesh while preserving real edges (faces, text, damage).
        img_np_sr = _suppress_halftone(img_np)
        if img_np_sr is not img_np:
            context.add_log(
                "SR: halftone/mesh pattern detected — applied bilateral pre-filter."
            )

        # Nothing to gain from a ×4 pass when the source already meets the target:
        # the result would only be downscaled again, at ~16× the memory and time.
        if max(orig_h, orig_w) >= target_long:
            context.add_log(
                f"SR: source {orig_w}×{orig_h} already at/above target long side "
                f"{target_long} — skipping ×4 model, resampling directly."
            )
            out_np = _post_sr_sharpen(_resize_to_long_side(img_np_sr, target_long), strength=0.38)
            context.working_image = out_np
            context.add_log(f"SR: final output size {out_np.shape[1]}×{out_np.shape[0]}.")
            return out_np

        # Safety cap: bound the ×4 OUTPUT (not the input) to prevent OOM.
        img_np_sr = _clamp_for_sr_output(img_np_sr)
        capped_h, capped_w = img_np_sr.shape[:2]
        if (capped_h, capped_w) != (orig_h, orig_w):
            context.add_log(
                f"SR: capped input from {orig_w}×{orig_h} to {capped_w}×{capped_h} "
                f"so the ×{_SR_SCALE} output fits {_MAX_SR_OUTPUT_PIXELS:,} px."
            )

        tile_size = _determine_tile_size(self.device)
        context.add_log(
            f"SR: running RealESRGAN ×4 on {capped_w}×{capped_h} "
            f"(tile_size={tile_size}, tile_pad=32, device={self.device})."
        )

        try:
            arr = img_np_sr.astype(np.float32) / 255.0
            t = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0)

            out = _realesrgan_tile_forward(
                self.realesrgan, t,
                tile_size=tile_size, tile_pad=32,
                device=self.device,
            )
            out_np = (
                out.squeeze(0).permute(1, 2, 0).numpy() * 255
            ).round().astype(np.uint8)

            context.add_log(
                f"SR: upscaled to {out_np.shape[1]}×{out_np.shape[0]}. "
                "Now resizing to 2K."
            )

        except (torch.cuda.OutOfMemoryError, RuntimeError) as cuda_exc:
            is_oom = isinstance(cuda_exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(cuda_exc).lower()
            if is_oom and self.device != "cpu":
                context.add_log(f"SR: CUDA OOM ({cuda_exc}) — retrying inference on CPU.")
                torch.cuda.empty_cache()
                try:
                    cpu_net = self.realesrgan.to("cpu")
                    out = _realesrgan_tile_forward(
                        cpu_net, t,
                        tile_size=192, tile_pad=16,
                        device="cpu",
                    )
                    out_np = (
                        out.squeeze(0).permute(1, 2, 0).numpy() * 255
                    ).round().astype(np.uint8)
                    try:
                        self.realesrgan.to(self.device)
                    except Exception:
                        pass
                except Exception as cpu_exc:
                    context.add_log(f"RealESRGAN CPU fallback failed: {cpu_exc} — falling back to bicubic.")
                    src_h, src_w = img_np_sr.shape[:2]
                    scale = max(2.0, _MIN_LONG_SIDE / max(max(src_h, src_w), 1))
                    new_w = max(1, round(src_w * scale))
                    new_h = max(1, round(src_h * scale))
                    pil = Image.fromarray(img_np_sr)
                    out_np = np.array(pil.resize((new_w, new_h), Image.Resampling.BICUBIC))
            else:
                context.add_log(f"RealESRGAN failed: {cuda_exc} — falling back to bicubic.")
                src_h, src_w = img_np_sr.shape[:2]
                scale = max(2.0, _MIN_LONG_SIDE / max(max(src_h, src_w), 1))
                new_w = max(1, round(src_w * scale))
                new_h = max(1, round(src_h * scale))
                pil = Image.fromarray(img_np_sr)
                out_np = np.array(pil.resize((new_w, new_h), Image.Resampling.BICUBIC))
        except Exception as exc:
            context.add_log(f"RealESRGAN failed: {exc} — falling back to bicubic.")
            src_h, src_w = img_np_sr.shape[:2]
            scale = max(2.0, _MIN_LONG_SIDE / max(max(src_h, src_w), 1))
            new_w = max(1, round(src_w * scale))
            new_h = max(1, round(src_h * scale))
            pil = Image.fromarray(img_np_sr)
            out_np = np.array(pil.resize((new_w, new_h), Image.Resampling.BICUBIC))

        # ── Post-SR: LAB-space sharpening for HD crispness ───────────────────
        # Apply gentle luminance sharpening to restore fine detail crispness
        # that the bilateral pre-filter or SR model may have softened slightly.
        out_np = _post_sr_sharpen(out_np, strength=0.38)

        if max(out_np.shape[:2]) != target_long:
            context.add_log(
                f"SR: resizing final output to long side {target_long} "
                "based on source resolution."
            )
        out_np = _resize_to_long_side(out_np, target_long)
        context.working_image = out_np
        context.add_log(
            f"SR: final output size {out_np.shape[1]}×{out_np.shape[0]}."
        )
        return out_np
