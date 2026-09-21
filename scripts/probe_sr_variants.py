"""
probe_sr_variants.py — Compare super-resolution variants on the pre-SR image.

The 196x148 noise.jpg case survives the pipeline in good shape up to
12_after_color_refine_pre_sr.png and is destroyed by the RealESRGAN x4plus pass.
This probe upscales that same image several ways so the responsible factor
(the model itself vs the halftone bilateral pre-filter) is visible side by side.

Usage:
    PYTHONPATH=backend "C:/Program Files/Python312/python.exe" \
        scripts/probe_sr_variants.py debug_output/12_after_color_refine_pre_sr.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ai.restoration.super_resolution import (
    _realesrgan_tile_forward,
    _suppress_halftone,
)

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "backend" / "ai" / "weights"


def load_rrdb(device: str) -> torch.nn.Module:
    from ai.restoration.rrdbnet import RRDBNet

    net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32).to(device)
    sd = torch.load(WEIGHTS / "RealESRGAN_x4plus.pth", map_location=device, weights_only=True)
    sd = sd.get("params_ema", sd.get("params", sd))
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


def esrgan_x4(img: np.ndarray, net: torch.nn.Module, device: str) -> np.ndarray:
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        out = _realesrgan_tile_forward(net, t, tile_size=256, tile_pad=32, device=device)
    arr = out.squeeze(0).permute(1, 2, 0).numpy()
    return (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def lanczos(img: np.ndarray, scale: int = 4) -> np.ndarray:
    h, w = img.shape[:2]
    return np.array(Image.fromarray(img).resize((w * scale, h * scale), Image.Resampling.LANCZOS))


def sharpness(img: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def detail_energy(img: np.ndarray) -> float:
    """High-frequency energy: std of (image - blurred). Smears drive this to ~0."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.std(g - cv2.GaussianBlur(g, (0, 0), 1.5)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("-d", "--outdir", type=Path, default=ROOT / "debug_output" / "sr_variants")
    args = ap.parse_args()

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"!! cannot read {args.image}")
        return 1
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    args.outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_rrdb(device)

    pre = _suppress_halftone(rgb)
    halftone_fired = pre is not rgb

    variants = {
        "00_input": rgb,
        "01_lanczos_x4": lanczos(rgb),
        "02_esrgan_x4_raw": esrgan_x4(rgb, net, device),
        "03_esrgan_x4_with_halftone_prefilter": esrgan_x4(pre, net, device),
    }
    # A blend of ESRGAN structure with Lanczos texture, for reference.
    e = variants["02_esrgan_x4_raw"].astype(np.float32)
    l = variants["01_lanczos_x4"].astype(np.float32)
    variants["04_esrgan_lanczos_blend_50"] = np.clip(0.5 * e + 0.5 * l, 0, 255).astype(np.uint8)

    print(f"input {rgb.shape[1]}x{rgb.shape[0]}  halftone_prefilter_fired={halftone_fired}  device={device}\n")
    print(f"{'variant':<40} {'size':>12} {'sharpness':>11} {'detail':>8}")
    for name, img in variants.items():
        out = args.outdir / f"{name}.png"
        cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"{name:<40} {img.shape[1]:>5}x{img.shape[0]:<6} "
              f"{sharpness(img):>11.1f} {detail_energy(img):>8.2f}")
    print(f"\nwrote {len(variants)} variants to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
