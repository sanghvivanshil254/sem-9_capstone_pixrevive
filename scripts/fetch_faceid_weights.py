#!/usr/bin/env python3
"""fetch_faceid_weights.py — H4 FaceID conditioning weights (urllib, no hub libs).

Downloads (project rule: urllib.request against GitHub/HuggingFace, weights
live under backend/ai/weights/, never a hub cache):

  1. InsightFace buffalo_l release zip (GitHub) -> extract ONLY
     w600k_r50.onnx, the recognition model our embedder runs via cv2.dnn /
     onnxruntime (512-d embeddings). The rest of the bundle (SCRFD detector
     etc.) is discarded — detection stays on YuNet, which we already ship.
  2. h94/IP-Adapter-FaceID SD1.5 adapter (HF) + its LoRA — the conditioning
     side that consumes the 512-d identity embedding.

Usage:
  python scripts/fetch_faceid_weights.py            # everything
  python scripts/fetch_faceid_weights.py --arcface-only
  python scripts/fetch_faceid_weights.py --skip-lora
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(REPO_ROOT, "backend", "ai", "weights")

BUFFALO_URL = ("https://github.com/deepinsight/insightface/releases/"
               "download/v0.7/buffalo_l.zip")
# Single-file mirror (fp32 buffalo_l recognizer, byte-equivalent weights):
# GitHub's release CDN can crawl at ~100 KB/s while HF typically saturates.
BUFFALO_MIRROR = ("https://huggingface.co/immich-app/buffalo_l/resolve/main/"
                  "recognition/model.onnx")
# h94/IP-Adapter-FaceID layout differs from h94/IP-Adapter: files live at the
# repo ROOT (models/... 404s — measured 2026-09-27 via HEAD requests).
FACEID_BASE = "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main"
FACEID_ADAPTER = f"{FACEID_BASE}/ip-adapter-faceid_sd15.bin"
FACEID_LORA = f"{FACEID_BASE}/ip-adapter-faceid_sd15_lora.safetensors"

FACEID_DIR = os.path.join(WEIGHTS, "ip_adapter_faceid", "models")
LORA_DIR = os.path.join(WEIGHTS, "ip_adapter_faceid", "models_lora")
ARCFACE_DIR = os.path.join(WEIGHTS, "faceid")
# Measured 2026-09-27: buffalo_l v0.7 ships the ResNet50 recognizer as
# w600k_r50.onnx (512-d); there is no w600k_mbt in the bundle.
ARCFACE_FILE = "w600k_r50.onnx"

CHUNK = 1 << 20


def _fetch(url: str, dest: str) -> int:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    if os.path.exists(dest):
        print(f"[skip] {dest} exists")
        return 0
    print(f"[get ] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "PixRevive/1.1"})
    total = 0
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
            if total % (100 * CHUNK) < CHUNK:
                print(f"       {total / (1 << 20):8.1f} MiB", end="\r", flush=True)
    print(f"       {total / (1 << 20):8.1f} MiB done")
    os.replace(tmp, dest)
    return total


def fetch_arcface() -> None:
    dest = os.path.join(ARCFACE_DIR, ARCFACE_FILE)
    if os.path.exists(dest):
        print(f"[skip] {dest} exists")
        return
    # Fast path: HF single-file mirror (no 280 MB zip to buffer).
    try:
        total = _fetch(BUFFALO_MIRROR, dest)
        if total:
            print(f"[ ok ] {dest} ({os.path.getsize(dest) / (1 << 20):.1f} MiB, mirror)")
            return
    except Exception as exc:  # noqa: BLE001 - fall back to the zip
        print(f"[warn] mirror failed ({exc}); falling back to buffalo_l.zip")
    print(f"[get ] buffalo_l bundle (extracting {ARCFACE_FILE} only)")
    req = urllib.request.Request(BUFFALO_URL, headers={"User-Agent": "PixRevive/1.1"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = [n for n in zf.namelist()
                 if os.path.basename(n) == ARCFACE_FILE]
        if not names:
            print(f"[fail] {ARCFACE_FILE} not found in buffalo_l.zip "
                  f"(contents: {zf.namelist()})", file=sys.stderr)
            raise SystemExit(1)
        os.makedirs(ARCFACE_DIR, exist_ok=True)
        with zf.open(names[0]) as src, open(dest + ".part", "wb") as out:
            while True:
                chunk = src.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
    os.replace(dest + ".part", dest)
    print(f"[ ok ] {dest} ({os.path.getsize(dest) / (1 << 20):.1f} MiB)")


def fetch_adapter(skip_lora: bool) -> None:
    _fetch(FACEID_ADAPTER, os.path.join(FACEID_DIR, "ip-adapter-faceid_sd15.bin"))
    if not skip_lora:
        _fetch(FACEID_LORA,
               os.path.join(LORA_DIR, "ip-adapter-faceid_sd15_lora.safetensors"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arcface-only", action="store_true",
                    help="only the ArcFace recognition ONNX")
    ap.add_argument("--skip-lora", action="store_true",
                    help="skip the FaceID LoRA (used only with use_lora=True)")
    args = ap.parse_args()
    fetch_arcface()
    if not args.arcface_only:
        fetch_adapter(args.skip_lora)
    print("[done] FaceID weights present.")


if __name__ == "__main__":
    main()
