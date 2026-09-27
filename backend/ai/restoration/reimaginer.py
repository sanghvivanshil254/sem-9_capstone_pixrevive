"""reimaginer.py — the re-imagine stage engine (Phase H1, design v2).

Hybrid evidence-preserving diffusion (docs/HYBRID_DIFFUSION_DESIGN.md):
when damage is so extensive that any "repair" is fabrication anyway, this
module re-imagines ONLY the destroyed regions (mask M) with a diffusion
inpainting model, while every surviving pixel (evidence E) is composited
back unconditionally and asserted bytewise:

    out[E] == original[E]        (hard invariant, checked every run)

Contract facts a maintainer needs:

* The caller (pipeline stage) must already hold the GPU semaphore — this
  module never touches concurrency; DDColor/GFPGAN must not fight it.
* Resource tiering comes from ai.restoration.resource_profiler.profile(),
  evaluated at first gate-fire (lazy) — never at import time.
* Models are lazy-loaded per tier and cached for the process lifetime
  (relatively cheap after the first image), but unload_reimagine_models()
  is available for hosts that need the VRAM back mid-run.
* NaN self-healing: fp16 diffusion can silently emit an all-black frame
  (measured on SD1.5). Every generated tile is checked; a poisoned tile is
  regenerated once at seed+1; a poisoned SDXL canvas falls back to the
  SD1.5 pipeline for that image; a still-poisoned result is reported as
  skipped — never shipped as black pixels.
* Identity: SFace cosine between the original's and the result's largest
  face (ai.validation.face_identity_gate). >= pass / flag band / < rollback
  with at most 2 re-rolls (seed+1), then revert to the LaMa canvas.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import cv2
import numpy as np
import torch

from ai.restoration.resource_profiler import ResourceTier, profile
from ai.validation.face_identity_gate import SFaceIdentityGate

logger = logging.getLogger(__name__)

# Anchor weights to THIS file, not the process CWD: the uvicorn server runs
# with cwd=backend/, where the old relative "backend/ai/weights" resolved to
# backend/backend/... and every live Stage-12 run reported "sd15 weights
# missing" (measured 2026-09-27). Standalone scripts run from the repo root
# and masked the bug.
WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "weights")
SD15_DIR = os.path.join(WEIGHTS, "sd15_inpaint")
SDXL_DIR = os.path.join(WEIGHTS, "sdxl_inpaint")
IP_ADAPTER_DIR = os.path.join(WEIGHTS, "ip_adapter", "models")
# Mirrors the h94/IP-Adapter repo layout: the CLIP image encoder lives INSIDE
# the `models` subfolder, next to the adapter weights. diffusers resolves
# image_encoder_folder relative to `subfolder` (measured failure 2026-09-27:
# encoder at ip_adapter/ root -> "no file named model.safetensors" -> silent
# unconditioned generation -> guaranteed SFace rollback).
IP_ENCODER_DIR = os.path.join(WEIGHTS, "ip_adapter", "models", "image_encoder")
# Identity conditioning strength: how hard generation is pulled toward the
# input's surviving face. H3 calibration verdict (measured, 2026-09-27, 8
# E2E points on case9's 60%-destroy mask): seed variance (cosine spread
# 0.21-0.38 across seeds) DOMINATES any scale effect (0.5->1.0 spread ~0.03
# on a fixed seed; means 0.22-0.30, no point clears the 0.36 rollback
# floor). No scale reliably passes hopeless coverage — the SFace gate
# refusing is the designed behavior, and higher scales over-bake the
# DAMAGED reference face. 0.5 stays: the diffusers-recommended SD1.5
# prior, least over-bake risk on real (milder) damage. REIMAGINE_IP_SCALE
# overrides per-run for hosts that want to experiment (sweep harness:
# scripts/sweep_ip_scale.py). The H4-class fix is FaceID-style conditioning.
IP_ADAPTER_SCALE = 0.5


def _ip_scale() -> float:
    raw = os.environ.get("REIMAGINE_IP_SCALE")
    if not raw:
        return IP_ADAPTER_SCALE
    try:
        return min(1.5, max(0.0, float(raw)))
    except ValueError:
        logger.warning("invalid REIMAGINE_IP_SCALE=%r — using default %.2f",
                       raw, IP_ADAPTER_SCALE)
        return IP_ADAPTER_SCALE


# --- FaceID-class identity conditioning (Phase H4) -------------------------
# The H3 sweep measured the structural ceiling of CLIP-crop conditioning
# (seed variance 0.21-0.38 dwarfs any scale effect); FaceID conditioning
# attacks the real problem by feeding generation a trained *identity*
# embedding (ArcFace 512-d) instead of a CLIP embedding of a damaged crop.
# Weights: backend/ai/weights/ip_adapter_faceid/ (scripts/fetch_faceid_weights.py).
FACEID_BIN = os.path.join(WEIGHTS, "ip_adapter_faceid", "models",
                          "ip-adapter-faceid_sd15.bin")
FACEID_LORA = os.path.join(WEIGHTS, "ip_adapter_faceid", "models_lora",
                           "ip-adapter-faceid_sd15_lora.safetensors")


def _faceid_enabled() -> bool:
    """FaceID conditioning opt-in (weights present; REIMAGINE_FACEID=0 disables).

    Default ON when the weights exist. It REPLACES the CLIP adapter while
    active — the processor slot is exclusive — and cleanly falls back when
    weights are absent, the install fails, or the image yields no identity
    embedding (faceid_adapter.FaceIDAdapter is the installer).
    """
    if os.environ.get("REIMAGINE_FACEID") == "0":
        return False
    return os.path.exists(FACEID_BIN)

DEFAULT_STEPS = 25
DEFAULT_GUIDANCE = 7.5
PROMPT = ("a clean vintage photograph of the same person, sharp natural detail, "
          "consistent lighting and tone")
NEGATIVE = "blurry, watermark, text, artifacts, oversaturated"
# Ladder: >= pass / flag band / < rollback (face_identity_gate defaults).
MAX_REROLLS = 2
# A tile/canvas this share of near-black pixels is NaN-poisoned (measured
# failure mode: 100%; genuine dark photos are never >40% pure zero-RGB).
_BLACK_SHARE = 0.60


class _ReimagineState:
    """Process-lazy model cache. One pipeline per generator kind."""

    def __init__(self) -> None:
        self.pipes: dict[str, Any] = {}
        self.tier: ResourceTier | None = None
        self.reason: str = ""
        self.measurements: dict = {}
        self.faceid: Any | None = None   # live FaceIDAdapter, when installed
        # What the LAST generation actually used (honesty for the record:
        # 'faceid' | 'clip' | 'none' — set by run_pass, read by reimagine()).
        self.conditioning_used: str | None = None
        # True only after a successful load_ip_adapter(); the presence check
        # alone lies (weights on disk != adapter installed on the UNet). Once
        # the adapter IS installed, diffusers' UNet REQUIRES image embeds —
        # generating without ip_adapter_image crashes with
        # "TypeError: argument of type 'NoneType' is not iterable"
        # (added_cond_kwargs=None, measured 2026-09-27). Track it explicitly.
        self.ip_loaded = False

    def reset(self) -> None:
        self.pipes = {}
        self.tier = None
        self.ip_loaded = False
        self.faceid = None
        self.conditioning_used = None


_STATE = _ReimagineState()


def _weights_present() -> bool:
    return (os.path.isdir(SD15_DIR)
            and os.path.exists(os.path.join(SD15_DIR, "model_index.json")))


# CLIP ViT-H/14 vision-tower geometry for the h94/IP-Adapter image encoder
# (2.528 GB fp32 = 632M params). transformers needs this config on disk; the
# upstream repo does not ship it. `projection_dim` 1024 matches the
# `image_proj` state-dict the adapter expects.
_IP_ENCODER_CONFIG: dict = {
    "_name_or_path": "./image_encoder",
    "architectures": ["CLIPVisionModelWithProjection"],
    "attention_dropout": 0.0,
    "dropout": 0.0,
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "image_size": 224,
    "initializer_factor": 1.0,
    "initializer_range": 0.02,
    "intermediate_size": 5120,
    "layer_norm_eps": 1e-05,
    "model_type": "clip_vision_model",
    "num_attention_heads": 16,
    "num_channels": 3,
    "num_hidden_layers": 32,
    "patch_size": 14,
    "projection_dim": 1024,
    "torch_dtype": "float32",
    "transformers_version": "4.28.0.dev0",
}


def _ensure_ip_encoder_config() -> None:
    """Write the encoder config.json if the weights folder lacks it."""
    cfg_path = os.path.join(IP_ENCODER_DIR, "config.json")
    if os.path.exists(os.path.join(IP_ENCODER_DIR, "model.safetensors")) \
            and not os.path.exists(cfg_path):
        try:
            os.makedirs(IP_ENCODER_DIR, exist_ok=True)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(_IP_ENCODER_CONFIG, f, indent=2)
            logger.info("re-imagine: wrote missing IP image-encoder config.json")
        except OSError as exc:  # noqa: BLE001 - read-only FS: gate will disable IP
            logger.warning("re-imagine: could not write encoder config (%s)", exc)


def _ip_adapter_present() -> bool:
    # The h94/IP-Adapter repo ships the image encoder as bare safetensors with
    # no config.json; transformers then guesses a ViT-B geometry and dies on a
    # size mismatch (measured 2026-09-27). Vendor the correct CLIP ViT-H/14
    # vision config here and self-heal fresh checkouts.
    _ensure_ip_encoder_config()
    return (os.path.isdir(IP_ADAPTER_DIR)
            and os.path.exists(os.path.join(IP_ADAPTER_DIR, "ip-adapter_sd15.safetensors"))
            and os.path.isdir(IP_ENCODER_DIR)
            and os.path.exists(os.path.join(IP_ENCODER_DIR, "model.safetensors")))


def _largest_face_crop(image_rgb: np.ndarray):
    """Aligned-crop-able bbox of the largest face, via the SFace gate's YuNet.

    Returns a BGR crop for the IP-Adapter reference image, or None when no
    face is detectable (conditioning then silently doesn't apply).
    """
    gate = SFaceIdentityGate()
    if not gate.weights_available():
        return None
    try:
        found = gate._largest_face(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if not found:
            return None
        (x, y, w, h), _aligned = found
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return bgr[max(0, y):y + h, max(0, x):x + w]
    except Exception:  # noqa: BLE001 - conditioning is best-effort
        return None


def _sdxl_weights_present() -> bool:
    return (os.path.isdir(SDXL_DIR)
            and os.path.exists(os.path.join(SDXL_DIR, "model_index.json"))
            and os.path.exists(os.path.join(
                SDXL_DIR, "unet", "diffusion_pytorch_model.fp16.safetensors")))


def _load_pipeline(kind: str) -> Any:
    """Load and configure one generator per its tier settings."""
    from diffusers import (DDIMScheduler, StableDiffusionInpaintPipeline,
                           StableDiffusionXLInpaintPipeline)

    tier = _STATE.tier
    assert tier is not None and tier.runnable
    sdxl = kind == "sdxl"
    path = SDXL_DIR if sdxl else SD15_DIR
    cls = StableDiffusionXLInpaintPipeline if sdxl else StableDiffusionInpaintPipeline
    torch_dtype = torch.float16 if tier.dtype == "fp16" else torch.float32
    logger.info("re-imagine: loading %s pipeline (tier=%s, dtype=%s) ...",
                kind, tier.name, tier.dtype)
    pipe = cls.from_pretrained(
        path,
        torch_dtype=torch_dtype,
        variant="fp16",  # fp16-filed weights; torch_dtype upcasts when fp32
        safety_checker=None,
        requires_safety_checker=False,
    )
    if not sdxl:
        # SD1.5's legacy UNet NaNs under fp16+PNDM; DDIM is stable.
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Identity conditioning (design doc section 3): load IP-Adapter when the
    # weights are present. The image encoder (~1.7 GB fp32) streams through
    # RAM once at load; without it the SFace gate rolls everything back
    # (measured on case9: best cosine -0.099 -> skipped_degraded).
    #
    # ORDER MATTERS, and slicing + IP-Adapter are mutually exclusive in
    # diffusers 0.40 (both measured 2026-09-27):
    #   * slicing AFTER load: SlicedAttnProcessor replaces the IP-Adapter
    #     processors, which alone can split the (text, image) embed tuple ->
    #     "'tuple' object has no attribute 'shape'" mid-generation.
    #   * slicing BEFORE load: load_ip_adapter re-instantiates
    #     SlicedAttnProcessor without slice_size -> load fails, we silently
    #     go unconditioned, and the identity gate rolls everything back.
    # Resolution: when identity conditioning is available it wins; SDPA
    # scaled-dot-product attention is already memory-efficient and
    # sequential CPU offload dominates the VRAM budget on MINIMAL.
    # Identity conditioning is decided ONCE per pipe load, BEFORE the
    # CPU-offload hooks snapshot the UNet (smoke-measured 2026-09-27: a
    # processor swap AFTER enable_sequential_cpu_offload leaves the new
    # processors' weights outside the offload snapshot -> 'mat2 is on cpu'
    # crash). FaceID wins when its weights are present; the CLIP adapter is
    # the fallback. Faceless images need NO adapter surgery — they generate
    # with null identity tokens (proj(zeros), the reference's uncond path).
    faceid_ready = False
    if not sdxl and _faceid_enabled():
        from ai.restoration.faceid_adapter import FaceIDAdapter
        fid = FaceIDAdapter(
            pipe, FACEID_BIN,
            lora_path=FACEID_LORA if os.path.exists(FACEID_LORA) else None)
        if fid.install():
            _STATE.faceid = fid
            faceid_ready = True
            logger.info("re-imagine: FaceID conditioning active (scale %.2f)",
                        _ip_scale())
        else:
            _STATE.faceid = None
            logger.warning("re-imagine: FaceID install failed — CLIP fallback")
    ip_ok = (not sdxl and not faceid_ready and _ip_adapter_present())
    if ip_ok:
        try:
            pipe.load_ip_adapter(
                os.path.join(WEIGHTS, "ip_adapter"),
                subfolder="models",
                image_encoder_folder="image_encoder",
                weight_name="ip-adapter_sd15.safetensors",
            )
            scale = _ip_scale()
            pipe.set_ip_adapter_scale(scale)
            _STATE.ip_loaded = True
            logger.info("re-imagine: IP-Adapter loaded (scale %.2f)", scale)
        except Exception as exc:  # noqa: BLE001 - generate unconditioned rather than fail
            ip_ok = False
            logger.warning("re-imagine: IP-Adapter load failed (%s) — "
                           "identity conditioning disabled", exc)
    if tier.attention_slicing and not ip_ok and not faceid_ready:
        # Slicing replaces attention processors and breaks BOTH adapter
        # kinds (measured 2026-09-27 for CLIP; same mechanics apply).
        pipe.enable_attention_slicing()

    if tier.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif tier.offload == "model":
        pipe.enable_model_cpu_offload()
    if tier.tiled_vae and hasattr(pipe, "enable_vae_tiling"):
        try:
            pipe.enable_vae_tiling()
        except Exception:  # noqa: BLE001 - optional optimization
            pass
    _STATE.pipes[kind] = pipe
    return pipe


def _tile_grid(size: int, tile_px: int, overlap: int) -> list[tuple[int, int]]:
    """Feathered tile origins covering [0, size) without exceeding the canvas."""
    if tile_px <= 0 or size <= tile_px:
        return [(0, 0)]
    step = tile_px - overlap
    coords = list(range(0, max(size - tile_px, 0) + 1, step))
    if coords[-1] != size - tile_px:
        coords.append(size - tile_px)
    return [(x, y) for y in coords for x in coords]


def _feather_alpha(size: int, tile_px: int, overlap: int) -> np.ndarray:
    """Cosine-feathered weight window (1.0 core -> 0.0 at tile edges)."""
    ramp = np.ones(tile_px, np.float32)
    if overlap > 0 and tile_px > 2 * overlap:
        edge = (1 + np.cos(np.linspace(np.pi, 0, overlap, dtype=np.float32))) / 2
        ramp[:overlap] = edge
        ramp[-overlap:] = edge[::-1]
    return np.outer(ramp, ramp)


def _black_share(arr: np.ndarray) -> float:
    return float((arr.max(axis=2) == 0).mean()) if arr.size else 1.0


def _generate(image_rgb: np.ndarray, mask: np.ndarray, kind: str,
              seed: int, size_px: int, steps: int, guidance: float,
              ip_image=None, id_embed=None) -> np.ndarray:
    """One full-canvas generation pass (tiles internally when tier demands).

    ip_image: PIL image of the input's surviving face (identity reference);
    None when no face was found or IP-Adapter is not loaded.
    id_embed: L2-normalized 512-d ArcFace embedding — FaceID conditioning
    (H4). The FaceID pipe was already fitted at _load_pipeline (BEFORE the
    offload snapshot; no runtime processor swapping). When None on a FaceID
    pipe, generation runs with null identity tokens — identity-neutral and
    split-safe. On a CLIP pipe the pre-H4 ip_image path applies unchanged.
    """
    from PIL import Image

    tier = _STATE.tier
    assert tier is not None
    pipe = _STATE.pipes.get(kind) or _load_pipeline(kind)
    h, w = image_rgb.shape[:2]

    # Adapter state sync (measured 2026-09-27, live-server find): a loaded
    # IP-Adapter makes the UNet REQUIRE image embeds — generating without
    # them crashes (added_cond_kwargs=None -> "'NoneType' is not iterable").
    # When this image has no detectable face (nothing to condition ON) the
    # adapter must be unloaded first; when a later image does have a face,
    # it is reloaded here. Keeps the cached pipeline honest across requests.
    # --- conditioning state (decided at pipe load; NO runtime swapping) ---
    # FaceID pipe: always generate with 4 trailing identity tokens — real
    # embeds when a face exists, null embeds (proj(zeros), the reference
    # uncond path) when it doesn't. The processor split stays aligned and
    # the offload snapshot is never touched. CLIP pipe: the existing
    # two-way sync (adapter installed <=> ip_image supplied, measured rule
    # 2026-09-27).
    faceid_state = "on" if (kind != "sdxl" and _STATE.faceid is not None) \
        else "off"
    if faceid_state == "on":
        _STATE.faceid.set_scale(_ip_scale())
    elif kind != "sdxl":
        # CLIP-adapter sync (unchanged behavior):
        if _STATE.ip_loaded and ip_image is None:
            try:
                pipe.unload_ip_adapter()
            except Exception as exc:  # noqa: BLE001
                logger.warning("re-imagine: IP-Adapter unload failed (%s)", exc)
            _STATE.ip_loaded = False
        elif ip_image is not None and not _STATE.ip_loaded \
                and _ip_adapter_present():
            try:
                pipe.load_ip_adapter(
                    os.path.join(WEIGHTS, "ip_adapter"),
                    subfolder="models",
                    image_encoder_folder="image_encoder",
                    weight_name="ip-adapter_sd15.safetensors",
                )
                pipe.set_ip_adapter_scale(_ip_scale())
                _STATE.ip_loaded = True
            except Exception as exc:  # noqa: BLE001 - degrade to unconditioned
                logger.warning("re-imagine: IP-Adapter reload failed (%s)", exc)
                _STATE.ip_loaded = False

    if kind == "sdxl":
        # SDXL is trained at 1024; anything below ~512 degrades it. On
        # MINIMAL (tile_px 704) tiles run at tier.tile_px; STANDARD/AMPLE
        # run the canvas at its natural (<=1024) size.
        gen_size = size_px
    else:
        gen_size = 512

    def run_pass(img: np.ndarray, msk: np.ndarray, s: int, sd: int) -> np.ndarray:
        nonlocal faceid_state  # degraded-encode downgrade propagates out
        canvas = cv2.resize(img, (s, s), interpolation=cv2.INTER_CUBIC)
        m = cv2.resize(msk, (s, s), interpolation=cv2.INTER_NEAREST)
        kwargs: dict = dict(
            prompt=PROMPT, negative_prompt=NEGATIVE,
            image=Image.fromarray(canvas).convert("RGB"),
            mask_image=Image.fromarray(m).convert("L"),
            num_inference_steps=steps, guidance_scale=guidance,
            generator=torch.Generator(device="cpu").manual_seed(sd),
        )
        if faceid_state == "on":
            # H4: project the identity embedding to 4 tokens and CONCAT to
            # text embeds (cond + uncond). Faceless input -> null embeds
            # for both (proj(zeros): identity-neutral, split stays aligned).
            with torch.inference_mode():
                text, neg = pipe.encode_prompt(
                    PROMPT, device="cpu", num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=NEGATIVE)
                if id_embed is not None:
                    cond_embed = torch.as_tensor(
                        id_embed, dtype=torch.float32).unsqueeze(0)
                else:
                    cond_embed = torch.zeros((1, 512))
                enc = _STATE.faceid.encode(
                    cond_embed, device=text.device, dtype=text.dtype)
                null_enc = _STATE.faceid.encode(
                    torch.zeros((1, 512)),
                    device=text.device, dtype=text.dtype)
            if enc is not None and null_enc is not None:
                tokens, _ = enc
                uncond_tokens, _ = null_enc
                del kwargs["prompt"], kwargs["negative_prompt"]
                kwargs["prompt_embeds"] = torch.cat([text, tokens], dim=1)
                kwargs["negative_prompt_embeds"] = torch.cat(
                    [neg, uncond_tokens], dim=1)
            else:
                # Encoding failed: fall back to plain text prompts rather
                # than crash — the SFace gate still judges the result.
                logger.warning("re-imagine: FaceID encode failed — text-only")
                faceid_state = "degraded"
        elif ip_image is not None and _STATE.ip_loaded:
            kwargs["ip_adapter_image"] = ip_image
        _STATE.conditioning_used = (
            ("faceid" if id_embed is not None else "none")
            if faceid_state == "on"
            else "clip" if (ip_image is not None and _STATE.ip_loaded)
            else "none")
        out = pipe(**kwargs).images[0]
        arr = np.asarray(out)
        if s != img.shape[1] or s != img.shape[0]:
            arr = cv2.resize(arr, (img.shape[1], img.shape[0]),
                             interpolation=cv2.INTER_CUBIC)
        return arr

    tile = tier.tile_px if kind == "sdxl" else 0
    if tile and (h > tile or w > tile):
        overlap = 128
        alpha = _feather_alpha(tile, tile, overlap)
        acc = np.zeros((h, w, 3), np.float32)
        wsum = np.zeros((h, w, 1), np.float32)
        # _tile_grid returns (x, y) pairs per axis; take the coordinates
        # apart so the sweep is a true cross-product over both axes.
        xs = [c[0] for c in _tile_grid(w, tile, overlap)]
        ys = [c[1] for c in _tile_grid(h, tile, overlap)]
        for y0 in ys:
            for x0 in xs:
                tile_img = image_rgb[y0:y0 + tile, x0:x0 + tile]
                tile_msk = mask[y0:y0 + tile, x0:x0 + tile]
                if not (tile_msk > 0).any():
                    continue  # pure evidence tile — nothing to imagine
                sd = (seed + y0 * 131 + x0) % (2**31)
                gen = run_pass(tile_img, tile_msk, min(tile, gen_size), sd)
                a = alpha[:gen.shape[0], :gen.shape[1]][..., None]
                acc[y0:y0 + gen.shape[0], x0:x0 + gen.shape[1]] += gen * a
                wsum[y0:y0 + gen.shape[0], x0:x0 + gen.shape[1]] += a
        canvas = acc / np.maximum(wsum, 1e-6)
        out = np.where((mask[..., None] > 0), canvas, image_rgb).astype(np.uint8)
        return out
    return run_pass(image_rgb, mask, gen_size, seed)


def _composite_evidence(original: np.ndarray, generated: np.ndarray,
                        mask: np.ndarray) -> np.ndarray:
    """out[M] = generated[M]; out[E] = original[E] — enforced bytewise."""
    out = generated.astype(np.uint8).copy()
    keep = mask == 0
    out[keep] = original[keep]
    # Hard invariant: evidence pixels are bit-identical, always.
    if not np.array_equal(out[keep], original[keep]):
        raise AssertionError("evidence-preservation invariant violated")
    return out


def reimagine(original_rgb: np.ndarray,
              mask: np.ndarray,
              task_id: str = "adhoc",
              prompt_extra: str = "",
              steps: int = DEFAULT_STEPS,
              guidance: float = DEFAULT_GUIDANCE,
              tier: ResourceTier | None = None,
              rerolls: int = MAX_REROLLS) -> dict:
    """Re-imagine destroyed regions of `original_rgb` (mask>0 = destroyed).

    Returns a record dict (never raises into the pipeline):
      status: "done" | "skipped" | "skipped_degraded"
      result: np.ndarray (H,W,3 uint8)          — only when status == "done"
      evidence_preserved / real_pixels_pct / identity / resource / rollback...
    """
    record: dict[str, Any] = {
        "status": "skipped", "result": None,
        "evidence_preserved": True, "identity": None,
        "resource": None, "rollback": None,
    }

    if tier is None:
        tier, reason, measurements = profile()
    else:
        reason, measurements = "supplied", {}
    record["resource"] = {
        "tier": tier.name,
        "reason": reason,
        **measurements,
        "config": {
            "offload": tier.offload,
            "attention_slicing": tier.attention_slicing,
            "tiled_vae": tier.tiled_vae,
            "tile_px": tier.tile_px,
            "dtype": tier.dtype,
            "generator": "sdxl" if tier.use_sdxl else "sd15",
        },
    }

    if not tier.runnable:
        record["reason"] = f"re-imagine unavailable: {tier.name} ({reason})"
        return record
    if not _weights_present():
        record["reason"] = "re-imagine unavailable: sd15 weights missing"
        return record
    if tier.tile_px and not _sdxl_weights_present() and tier.offload != "none":
        # SDXL requested by tier but absent: MINIMAL falls back to SD1.5
        # (documented fallback, design doc section 5).
        pass

    _STATE.tier, _STATE.reason, _STATE.measurements = tier, reason, measurements
    # Tier decides the generator: MINIMAL pins SD1.5 (measured — fp16 SDXL is
    # NaN-broken on this torch/GPU class and fp32 SDXL exceeds host RAM),
    # STANDARD/AMPLE take SDXL when its weights are present.
    kind = ("sdxl" if (tier.use_sdxl and _sdxl_weights_present()) else "sd15")
    if kind == "sd15" and not _weights_present():
        record["reason"] = "re-imagine unavailable: no diffusion weights"
        return record

    seed = int(hashlib.sha256(str(task_id).encode()).hexdigest()[:8], 16)
    original = np.ascontiguousarray(original_rgb)
    mask = (mask > 127).astype(np.uint8) * 255
    real_pct = float((mask == 0).mean() * 100)

    gate = SFaceIdentityGate()
    # Identity reference: the input's surviving face crop (same YuNet the
    # verifier uses). run_pass only passes it when the adapter actually
    # loaded (hasattr pipe.ip_adapter), so a failed adapter load silently
    # degrades to unconditioned generation — which the SFace gate then
    # judges honestly.
    face_crop = _largest_face_crop(original)
    ip_image = None
    if face_crop is not None and _ip_adapter_present():
        from PIL import Image as _PILImage
        ip_image = _PILImage.fromarray(
            cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)).convert("RGB")
    record["identity_conditioned"] = ip_image is not None

    # Identity EMBEDDING for FaceID conditioning (H4): computed from the
    # same BGR crop. When present it governs; the CLIP-adapter path stays
    # as fallback (embedder unavailable / embed fails / REIMAGINE_FACEID=0).
    id_embed = None
    if ip_image is not None and _faceid_enabled():
        from ai.validation.faceid_embedder import FaceIDEmbedder
        _emb = FaceIDEmbedder()
        if _emb.available():
            id_embed = _emb.embed(face_crop)  # 512-d, L2-normalized (or None)
    record["identity_conditioning"] = (
        "faceid" if (_STATE.faceid is not None)
        else ("clip" if ip_image is not None else "none"))

    attempt, final = 0, None
    degraded = False
    rerolls = max(0, int(rerolls))
    while attempt <= rerolls:
        sd = seed + attempt
        try:
            gen = _generate(original, mask, kind, sd,
                            size_px=min(1024, max(original.shape[:2])),  # canvas cap
                            steps=steps, guidance=guidance,
                            ip_image=ip_image, id_embed=id_embed)
        except torch.cuda.OutOfMemoryError:
            logger.warning("re-imagine OOM on %s tier — degrading to SD1.5", tier.name)
            if kind == "sdxl" and _weights_present():
                kind, degraded = "sd15", True
                _STATE.pipes.pop("sdxl", None)
                continue
            record["reason"] = "re-imagine unavailable: GPU out of memory"
            return record
        if _black_share(gen) > _BLACK_SHARE:
            # NaN-poisoned frame: self-heal once at seed+1, then degrade.
            if attempt == 0 and kind == "sdxl":
                logger.warning("SDXL produced a NaN frame; falling back to SD1.5")
                kind, degraded = "sd15", True
                continue
            attempt += 1
            continue
        candidate = _composite_evidence(original, gen, mask)
        verdict = gate.check(original, candidate, max_faces=0) \
            if gate.weights_available() else \
            {"action": "flag", "similarity": None, "reason": "sface_unavailable"}
        if verdict["action"] == "rollback":
            if attempt < rerolls:
                attempt += 1
                continue
            # Last attempt also rolled back: revert entirely — a hybrid
            # result the identity gate rejected must never ship.
            record["status"] = "skipped_degraded"
            record["identity"] = verdict
            record["reason"] = (
                "re-imagine rolled back: identity could not be preserved "
                f"after {attempt + 1} attempts (best SFace cosine "
                f"{verdict.get('similarity')})")
            if _STATE.conditioning_used:
                record["identity_conditioning"] = _STATE.conditioning_used
            return record
        final = candidate
        record["identity"] = verdict
        break

    if final is None:
        record["status"] = "skipped_degraded"
        record["reason"] = ("re-imagine rolled back: identity could not be "
                            f"preserved after {rerolls + 1} attempts")
        if _STATE.conditioning_used:
            record["identity_conditioning"] = _STATE.conditioning_used
        return record

    record["status"] = "done"
    record["result"] = final
    if _STATE.conditioning_used:
        # What generation ACTUALLY used (install failures and encode
        # degradations may have downgraded the planned conditioning).
        record["identity_conditioning"] = _STATE.conditioning_used
    record["real_pixels_pct"] = round(real_pct, 1)
    record["generated_pixels_pct"] = round(100.0 - real_pct, 1)
    record["evidence_preserved"] = bool(
        np.array_equal(final[mask == 0], original[mask == 0]))
    record["seed"] = seed
    record["generator"] = kind + ("+sd15_fallback" if degraded else "")
    record["steps"] = steps
    return record


def unload_reimagine_models() -> None:
    """Free VRAM/RAM held by the re-imagine pipelines (caller holds semaphore)."""
    if _STATE.faceid is not None:
        try:
            _STATE.faceid.uninstall()
        except Exception:  # noqa: BLE001 - teardown is best-effort
            pass
        _STATE.faceid = None
    for kind in list(_STATE.pipes):
        pipe = _STATE.pipes.pop(kind, None)
        del pipe
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
