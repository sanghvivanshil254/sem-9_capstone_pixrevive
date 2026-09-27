"""resource_profiler.py — runtime resource tiering for the re-imagine stage.

Phase H1 of the hybrid evidence-preserving diffusion design
(docs/HYBRID_DIFFUSION_DESIGN.md section 4). The re-imagine stage's precheck
and offload strategy must be decided at runtime from what the host actually
has free — not assumed from one dev box's spec sheet.

Design notes a maintainer needs:

* Units are gibibytes (GiB, 1024**3) everywhere; the report block labels them
  "gb" for brevity. Free-not-total is the decision variable: other models
  (LaMa, GFPGAN, DDColor) hold VRAM while we profile, so a 4 GB card with
  2.5 GB genuinely free must not be treated as a 4 GB card.
* Profiling happens at first gate-fire and is re-run per invocation — never
  cached at import time. Specs do not change mid-process, but *usage* does on
  shared hosts.
* RAM matters as much as VRAM: CPU-offloaded weights live in host memory. A
  tier whose RAM minimum fails steps DOWN a tier rather than OOM-ing the
  host — slower is recoverable, a dead backend is not.
* "no_gpu" doubles as the not-runnable sentinel tier. When CUDA exists but
  every tier floor fails, selection returns NO_GPU with a distinguishing
  reason string ("insufficient_resources: ...") so the report can say exactly
  why the stage skipped.
* REIMAGINE_FORCE_TIER pins a host to a tier ("minimal"|"standard"|"ample",
  plus "off" to disable). An override never waives a tier's minimums: a
  forced tier that fails its own floor steps down like autodetect would.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import psutil
import torch

logger = logging.getLogger(__name__)

GiB = 1024.0**3

_OVERRIDE_ENV = "REIMAGINE_FORCE_TIER"


@dataclass(frozen=True)
class ResourceTier:
    """One rung of the re-imagine resource ladder (design doc section 4.2)."""

    name: str
    offload: str  # "none" | "model" | "sequential"
    attention_slicing: bool
    tiled_vae: bool
    tile_px: int  # latent-tile edge for chunked generation (0 = untiled)
    use_controlnet: bool
    min_vram_gb: float | None  # None = no CUDA requirement
    min_ram_gb: float | None
    # Measured on the reference box (torch 2.13 + GTX 1650): fp16 diffusion
    # produces NaN -> all-black frames in EVERY offload/scheduler config,
    # while fp32 + sequential offload is stable at 0.76 GB and is also the
    # FASTEST measured config (full-resident fp32 spills to shared memory).
    # MINIMAL therefore pins fp32 and SD1.5; hosts with modern cards (where
    # fp16 is healthy) keep fp16 + SDXL. The reimaginer's NaN self-heal
    # remains the runtime safety net for hosts we have not measured.
    dtype: str = "fp16"          # "fp16" | "fp32" — diffusion compute dtype
    use_sdxl: bool = True        # False = SD1.5 fallback (design doc section 5)
    runnable: bool = True  # NO_GPU and below-floor outcomes are not runnable


# Tier ladder, highest first. Order is the step-down path.
AMPLE = ResourceTier(
    name="ample", offload="none", attention_slicing=False, tiled_vae=False,
    tile_px=1024, use_controlnet=True, min_vram_gb=12.0, min_ram_gb=8.0,
)
STANDARD = ResourceTier(
    name="standard", offload="model", attention_slicing=False, tiled_vae=False,
    tile_px=1024, use_controlnet=True, min_vram_gb=8.0, min_ram_gb=10.0,
)
MINIMAL = ResourceTier(
    name="minimal", offload="sequential", attention_slicing=True, tiled_vae=True,
    tile_px=704, use_controlnet=False, min_vram_gb=3.0, min_ram_gb=2.2,
    dtype="fp32", use_sdxl=False,  # measured: fp16 NaNs on this torch/GPU class
    # min_ram_gb 2.2 is MEASURED: the fp32 sequential-offload probe's python
    # process peaked at ~2.1 GB host RAM (weights stream through RAM, they do
    # not all residency at once). The first-calibrated 8 GB floor was
    # intuition and would have permanently barred this 7.3 GB reference box —
    # the exact machine the tier exists for; observed free RAM here fluctuates
    # 3.1-3.8 GB run-to-run. The floor answers "will it physically fit", not
    # "is the host otherwise busy"; backstops for tighter hosts: the
    # reimaginer's OOM catch degrades gracefully and the report records it.
)
NO_GPU = ResourceTier(
    name="no_gpu", offload="none", attention_slicing=False, tiled_vae=False,
    tile_px=0, use_controlnet=False, min_vram_gb=None, min_ram_gb=None,
    runnable=False,
)

_LADDER: tuple[ResourceTier, ...] = (AMPLE, STANDARD, MINIMAL)

_VALID_OVERRIDES = {t.name for t in _LADDER} | {"off"}


def detect_gpu_vram_free_gb() -> tuple[float, float] | None:
    """(free_gb, total_gb) currently visible to CUDA, or None when no GPU.

    Uses torch.cuda.mem_get_info(): what CUDA sees free *right now* — the
    honest number on shared hosts where other models hold memory. A probe
    failure degrades to None (skip path), never raises into the pipeline.
    """
    try:
        if not torch.cuda.is_available():
            return None
        free_b, total_b = torch.cuda.mem_get_info()
        return free_b / GiB, total_b / GiB
    except Exception as exc:  # noqa: BLE001 - profiling must never crash the stage
        logger.warning("VRAM probe failed, treating as no GPU: %s", exc)
        return None


def detect_system_ram_free_gb() -> float:
    """Free system RAM (GiB) available to the process, 0.0 if undetectable.

    psutil.virtual_memory().available is "available", not merely "free":
    reclaimable page cache is included, which is what offload can actually
    take. 0.0 (the impossible value) fails every RAM minimum and forces the
    graceful-skip path — the safe direction for an unknown host.
    """
    try:
        return psutil.virtual_memory().available / GiB
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAM probe failed, treating as zero headroom: %s", exc)
        return 0.0


def _read_override() -> str | None:
    raw = os.environ.get(_OVERRIDE_ENV, "").strip().lower()
    return raw or None


def _select_with_reason(
    vram_free_gb: float | None,
    ram_free_gb: float,
    override: str | None,
) -> tuple[ResourceTier, str]:
    """Shared selection core. Returns (tier, reason-for-the-report)."""
    if override is None:
        override = _read_override()

    if override == "off":
        return NO_GPU, "override:off"
    if override is not None and override not in _VALID_OVERRIDES:
        logger.warning(
            "REIMAGINE_FORCE_TIER=%r is not one of %s — ignoring override",
            override, sorted(_VALID_OVERRIDES),
        )
        override = None

    if vram_free_gb is None:
        return NO_GPU, "no_gpu_detected"

    ladder = _LADDER
    reason = "autodetect"
    if override is not None:
        forced_idx = next(i for i, t in enumerate(_LADDER) if t.name == override)
        # The override pins the *ceiling*: start at the forced tier, keep its
        # minimums enforceable, and allow stepping down the normal ladder.
        ladder = _LADDER[forced_idx:]
        reason = f"override:{override}"

    for tier in ladder:
        assert tier.min_vram_gb is not None and tier.min_ram_gb is not None
        if vram_free_gb < tier.min_vram_gb:
            continue  # next rung down
        if ram_free_gb < tier.min_ram_gb:
            # RAM step-down: VRAM qualifies but host RAM does not. Rather
            # than OOM-ing the host on offload, try the next rung's smaller
            # appetite. Recorded in the reason for the audit trail.
            reason = (
                f"ram_stepped_down:{tier.name}_needs_{tier.min_ram_gb:.1f}gb_free"
                if reason == "autodetect" else reason
            )
            continue
        return tier, reason

    return NO_GPU, (
        f"insufficient_resources: vram_free {vram_free_gb:.2f}gb / "
        f"ram_free {ram_free_gb:.2f}gb below every tier floor"
    )


def select_resource_tier(
    vram_free_gb: float | None,
    ram_free_gb: float,
    override: str | None = None,
) -> ResourceTier:
    """Pick the re-imagine tier for the detected resources.

    See _select_with_reason for semantics; this wrapper keeps the simple
    contract from the design doc, and select_resource_tier_with_reason()
    exposes the audit string for the report block.
    """
    return _select_with_reason(vram_free_gb, ram_free_gb, override)[0]


def select_resource_tier_with_reason(
    vram_free_gb: float | None,
    ram_free_gb: float,
    override: str | None = None,
) -> tuple[ResourceTier, str]:
    """(tier, reason) — the reason is recorded in the report's resource block."""
    return _select_with_reason(vram_free_gb, ram_free_gb, override)


def profile() -> tuple[ResourceTier, str, dict]:
    """Full profiling pass at gate-fire time.

    Returns (tier, reason, measurements) where measurements carries the raw
    detected numbers for the report's resource block (design doc section 4.4).
    """
    vram = detect_gpu_vram_free_gb()
    ram = detect_system_ram_free_gb()
    override = _read_override()
    vram_free = vram[0] if vram else None
    vram_total = vram[1] if vram else 0.0
    tier, reason = select_resource_tier_with_reason(vram_free, ram, override)
    measurements = {
        "vram_free_gb": round(vram_free, 2) if vram_free is not None else None,
        "vram_total_gb": round(vram_total, 2) if vram else None,
        "ram_free_gb": round(ram, 2),
        "override": override,
    }
    logger.info(
        "re-imagine tier selected: %s (%s; %s)", tier.name, reason, measurements
    )
    return tier, reason, measurements
