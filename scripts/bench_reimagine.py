"""bench_reimagine.py — the deciding benchmark for the v2 generator stack.

Measures, on the actual hardware (GTX 1650, 4 GB):
  * SD1.5-inpaint @ 512px   — the v1 config, MINIMAL tier settings
  * SDXL-inpaint  @ 704px   — one MINIMAL-tier tile pass of the v2 stack
for wall-clock time and peak VRAM (torch.cuda.max_memory_allocated), using
each tier's real generation settings (sequential CPU-offload, attention
slicing, tiled VAE when available).

Output: test_artifacts/bench_reimagine.json + the inpainted PNGs. The
generator decision (docs/HYBRID_DIFFUSION_DESIGN.md section 5) reads this file.

Usage:
  PYTHONPATH=backend "C:/Program Files/Python312/python.exe" scripts/bench_reimagine.py [--sd15-only|--sdxl-only] [--steps 25]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

WEIGHTS = os.path.join("backend", "ai", "weights")
OUT_DIR = os.path.join("test_artifacts", "bench_reimagine")

SEED = 20260926
GUIDANCE = 7.5


def make_inputs(size: int):
    """Deterministic test canvas + center-square mask (30% of frame)."""
    rng = np.random.default_rng(SEED)
    yy, xx = np.mgrid[0:size, 0:size]
    image = np.stack([
        128 + 60 * np.sin(xx / size * 3.1),
        128 + 60 * np.sin(yy / size * 2.3),
        128 + 60 * np.sin((xx + yy) / size * 1.7),
    ], axis=-1).clip(0, 255).astype(np.uint8)
    mask = np.zeros((size, size), np.uint8)
    lo, hi = int(size * 0.35), int(size * 0.65)
    mask[lo:hi, lo:hi] = 255
    return image, mask


def peak_vram_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**3


def bench_leg(pipeline_dir: str, sdxl: bool, size: int, steps: int,
              offload: str = "sequential", dtype: str = "fp16",
              no_reduced_reduction: bool = False) -> dict:
    from diffusers import (StableDiffusionInpaintPipeline,
                           StableDiffusionXLInpaintPipeline)

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    cls = StableDiffusionXLInpaintPipeline if sdxl else StableDiffusionInpaintPipeline
    t0 = time.time()
    pipe = cls.from_pretrained(
        os.path.join(WEIGHTS, pipeline_dir),
        torch_dtype=torch_dtype,
        variant="fp16",  # fp16-FILED weights on disk; torch_dtype upcasts for fp32
        safety_checker=None,   # NSFW checker is irrelevant here and costs VRAM
        requires_safety_checker=False,
    )
    # The legacy SD1.5-inpaint UNet produces NaN latents under fp16 + PNDM
    # (measured: an all-black frame). DDIM is numerically stable.
    if not sdxl:
        from diffusers import DDIMScheduler
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if no_reduced_reduction:
        # Known NaN cure on some Turing cards: fp16 matmuls accumulate in
        # reduced precision by default; disabling restores full accumulation.
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    load_s = time.time() - t0

    # Offload strategy (diagnostic axis — NaN isolation).
    if offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload == "model":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")  # "none": everything resident on GPU
    if dtype == "fp16":
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass

    image, mask = make_inputs(size)
    from PIL import Image
    pil_img = Image.fromarray(image).convert("RGB")
    pil_mask = Image.fromarray(mask).convert("L")
    gen = torch.Generator(device="cpu").manual_seed(SEED)

    def run() -> np.ndarray:
        out = pipe(prompt="a clean vintage photograph, sharp detail",
                   negative_prompt="blurry, watermark, text",
                   image=pil_img, mask_image=pil_mask,
                   num_inference_steps=steps, guidance_scale=GUIDANCE,
                   generator=gen).images[0]
        return np.asarray(out)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    arr = run()                      # warmup (includes offload-graph setup)
    warm_s = time.time() - t0
    warm_vram = peak_vram_gb()

    torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    t0 = time.time()
    arr = run()                      # measured pass
    run_s = time.time() - t0
    run_vram = peak_vram_gb()

    # Output sanity: NaN-poisoned latents cast to an all-black frame (measured
    # failure mode of fp16+PNDM). A black result must not enter the decision.
    black_pct = float((arr.max(axis=2) == 0).mean() * 100)
    if black_pct > 99.0:
        raise RuntimeError(
            f"output is {black_pct:.0f}% black — numerically unstable pass; "
            "scheduler/dtype fix required before this leg can be trusted"
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    Image.fromarray(arr).save(os.path.join(OUT_DIR, f"{'sdxl' if sdxl else 'sd15'}.png"))

    return {
        "pipeline": "sdxl_inpaint" if sdxl else "sd15_inpaint",
        "size_px": size, "steps": steps, "black_pct": round(black_pct, 2),
        "offload": offload, "dtype": dtype,
        "no_reduced_reduction": no_reduced_reduction,
        "load_s": round(load_s, 1),
        "warmup_s": round(warm_s, 1), "warmup_vram_gb": round(warm_vram, 2),
        "run_s": round(run_s, 1), "run_vram_gb": round(run_vram, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sd15-only", action="store_true")
    ap.add_argument("--sdxl-only", action="store_true")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--dtype", default="fp32", choices=["fp16", "fp32"],
                    help="fp32 default: fp16 diffusion is NaN-broken on this "
                         "torch/GPU class (measured, see resource_profiler)")
    ap.add_argument("--probe", action="store_true",
                    help="short NaN-isolation matrix over offload/dtype axes")
    args = ap.parse_args()

    if args.probe:
        # Round 2: the failure is fp16 itself on this torch/GPU combo (black
        # in ALL offload modes). Race the two known cures at 10 steps:
        matrix = [
            dict(offload="sequential", dtype="fp16", no_reduced_reduction=True),
            dict(offload="sequential", dtype="fp32"),
            dict(offload="none", dtype="fp32"),
        ]
        rows = []
        for cfg in matrix:
            tag = (f"offload={cfg['offload']} dtype={cfg['dtype']} "
                   f"noRed={cfg.get('no_reduced_reduction', False)}")
            print(f"--- probe {tag} ---", flush=True)
            try:
                row = bench_leg("sd15_inpaint", sdxl=False, size=512,
                                steps=10, **cfg)
                row["tag"] = tag
                rows.append(row)
                print(json.dumps(row), flush=True)
            except Exception as exc:  # noqa: BLE001 - a failed config IS the data
                rows.append({"tag": tag, "error": str(exc)[:200]})
                print(f"FAILED: {str(exc)[:200]}", flush=True)
        with open(os.path.join("test_artifacts", "bench_probe.json"), "w") as f:
            json.dump(rows, f, indent=2)
        return

    results: list[dict] = []
    if not args.sdxl_only:
        print("=== SD1.5-inpaint, 512px, MINIMAL-tier settings ===", flush=True)
        results.append(bench_leg("sd15_inpaint", sdxl=False, size=512,
                                 steps=args.steps, dtype=args.dtype))
        print(json.dumps(results[-1], indent=2), flush=True)
    if not args.sd15_only:
        print("=== SDXL-inpaint, 704px single tile, MINIMAL-tier settings ===", flush=True)
        try:
            results.append(bench_leg("sdxl_inpaint", sdxl=True, size=704,
                                     steps=args.steps, dtype=args.dtype))
            print(json.dumps(results[-1], indent=2), flush=True)
        except Exception as exc:  # noqa: BLE001 - OOM is itself a measurement
            print(f"SDXL leg FAILED: {exc}", flush=True)
            results.append({"pipeline": "sdxl_inpaint", "error": str(exc)[:300]})

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join("test_artifacts", "bench_reimagine.json"), "w") as f:
        json.dump({"seed": SEED, "guidance": GUIDANCE, "results": results}, f, indent=2)
    print("saved test_artifacts/bench_reimagine.json", flush=True)


if __name__ == "__main__":
    main()
