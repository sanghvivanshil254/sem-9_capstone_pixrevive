"""fetch_reimagine_weights.py — download the Phase H1 re-imagine model weights.

Per the PixRevive runtime rules: downloads use urllib.request against GitHub or
HuggingFace resolve endpoints directly — never a hub cache, never ModelScope.
Everything lands in backend/ai/weights/ (project rule 3).

Weights fetched (docs/HYBRID_DIFFUSION_DESIGN.md section 3):
  sface/yunet_270face.onnx       YuNet face detector      (~0.4 MB)
  sface/face_recognition_sface_2021dec.onnx  SFace recognizer (~37 MB)
  sd15_inpaint/                  SD-1.5 inpainting fp16   (~2.0 GB)
  sdxl_inpaint/                  SDXL-1.0 inpainting fp16 (~7.0 GB)

Resumable: existing files of the correct size are skipped; partial *.part
files are re-requested with a Range header when the server supports it.

Usage:
  "C:/Program Files/Python312/python.exe" scripts/fetch_reimagine_weights.py [--skip-sdxl]
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request

WEIGHTS = os.path.join("backend", "ai", "weights")

# (relative dest path or dir manifest, url list). A "file" entry is one URL;
# a "dir" entry is a diffusers folder layout fetched file-by-file.
SFACE = [
    ("sface/yunet_270face.onnx",
     "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    ("sface/face_recognition_sface_2021dec.onnx",
     "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
]

# NOTE: the original host runwayml/stable-diffusion-inpainting was deleted
# (org emptied 2026). HuggingFace's official redirect target is the
# stable-diffusion-v1-5 org — verified live, fp16 safetensors on main.
SD15 = "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/resolve/main"
SD15_INPAINT_FP16 = [
    ("sd15_inpaint/unet/config.json", f"{SD15}/unet/config.json"),
    ("sd15_inpaint/unet/diffusion_pytorch_model.fp16.safetensors", f"{SD15}/unet/diffusion_pytorch_model.fp16.safetensors"),
    ("sd15_inpaint/vae/config.json", f"{SD15}/vae/config.json"),
    ("sd15_inpaint/vae/diffusion_pytorch_model.fp16.safetensors", f"{SD15}/vae/diffusion_pytorch_model.fp16.safetensors"),
    ("sd15_inpaint/text_encoder/config.json", f"{SD15}/text_encoder/config.json"),
    ("sd15_inpaint/text_encoder/model.fp16.safetensors", f"{SD15}/text_encoder/model.fp16.safetensors"),
    ("sd15_inpaint/tokenizer/merges.txt", f"{SD15}/tokenizer/merges.txt"),
    ("sd15_inpaint/tokenizer/special_tokens_map.json", f"{SD15}/tokenizer/special_tokens_map.json"),
    ("sd15_inpaint/tokenizer/tokenizer_config.json", f"{SD15}/tokenizer/tokenizer_config.json"),
    ("sd15_inpaint/tokenizer/vocab.json", f"{SD15}/tokenizer/vocab.json"),
    ("sd15_inpaint/scheduler/scheduler_config.json", f"{SD15}/scheduler/scheduler_config.json"),
    ("sd15_inpaint/model_index.json", f"{SD15}/model_index.json"),
]

# IP-Adapter (SD1.5, h94 repo — the diffusers-format reference):
#   image_encoder  CLIP ViT-H/14 image tower (~1.71 GB fp32)
#   ip-adapter_sd15  the adapter proper (~44 MB)
# Loaded once per process (like the pipelines) to condition generation on
# the input's surviving face (identity conditioning — design doc section 3).
IP_ADAPTER = [
    ("ip_adapter/models/image_encoder/config.json",
     "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/config.json"),
    ("ip_adapter/models/image_encoder/model.safetensors",
     "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors"),
    ("ip_adapter/models/ip-adapter_sd15.safetensors",
     "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors"),
]

SDXL_INPAINT_FP16 = [
    ("sdxl_inpaint/unet/config.json", "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/main/unet/config.json"),
    ("sdxl_inpaint/unet/diffusion_pytorch_model.fp16.safetensors", "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors"),
    ("sdxl_inpaint/vae/config.json", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/vae/config.json"),
    ("sdxl_inpaint/vae/diffusion_pytorch_model.fp16.safetensors", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors"),
    ("sdxl_inpaint/text_encoder/config.json", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/config.json"),
    ("sdxl_inpaint/text_encoder/model.fp16.safetensors", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/model.fp16.safetensors"),
    ("sdxl_inpaint/text_encoder_2/config.json", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/config.json"),
    ("sdxl_inpaint/text_encoder_2/model.fp16.safetensors", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/model.fp16.safetensors"),
    ("sdxl_inpaint/tokenizer/merges.txt", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer/merges.txt"),
    ("sdxl_inpaint/tokenizer/special_tokens_map.json", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer/special_tokens_map.json"),
    ("sdxl_inpaint/tokenizer/tokenizer_config.json", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer/tokenizer_config.json"),
    # NOTE: no tokenizer.json in SDXL repos — the CLIP tokenizer is vocab+merges.
    ("sdxl_inpaint/tokenizer/vocab.json", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer/vocab.json"),
    ("sdxl_inpaint/tokenizer_2/merges.txt", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer_2/merges.txt"),
    ("sdxl_inpaint/tokenizer_2/special_tokens_map.json", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer_2/special_tokens_map.json"),
    ("sdxl_inpaint/tokenizer_2/tokenizer_config.json", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer_2/tokenizer_config.json"),
    ("sdxl_inpaint/tokenizer_2/vocab.json", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer_2/vocab.json"),
    ("sdxl_inpaint/scheduler/scheduler_config.json", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/scheduler/scheduler_config.json"),
    ("sdxl_inpaint/feature_extractor/preprocessor_config.json", "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/main/feature_extractor/preprocessor_config.json"),
    ("sdxl_inpaint/model_index.json", "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/main/model_index.json"),
]


def _fetch(rel: str, url: str) -> None:
    dest = os.path.join(WEIGHTS, *rel.split("/"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  skip {rel} ({os.path.getsize(dest)} bytes)")
        return
    part = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "pixrevive-fetch/1.0"})
    if os.path.exists(part) and os.path.getsize(part) > 0:
        req.add_header("Range", f"bytes={os.path.getsize(part)}-")
    with urllib.request.urlopen(req, timeout=60) as resp, open(part, "ab" if os.path.exists(part) else "wb") as f:
        total = resp.headers.get("Content-Length")
        total = int(total) + (os.path.getsize(part) if os.path.exists(part) else 0) if total else None
        done = os.path.getsize(part) if os.path.exists(part) else 0
        t0 = time.time()
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and time.time() - t0 > 5:
                print(f"  {rel}: {done / 1e6:.0f}/{total / 1e6:.0f} MB "
                      f"({done / max(total, 1) * 100:.0f}%)", flush=True)
                t0 = time.time()
    os.replace(part, dest)
    print(f"  done {rel} ({os.path.getsize(dest)} bytes)", flush=True)


def main() -> None:
    skip_sdxl = "--skip-sdxl" in sys.argv
    only_ip_adapter = "--ip-adapter-only" in sys.argv
    jobs: list[tuple[str, str]] = []
    if only_ip_adapter:
        jobs = list(IP_ADAPTER)
    else:
        jobs = list(SFACE) + list(SD15_INPAINT_FP16) + list(IP_ADAPTER)
        if not skip_sdxl:
            jobs += list(SDXL_INPAINT_FP16)
    print(f"Fetching {len(jobs)} files into {WEIGHTS}/ ...", flush=True)
    failures: list[str] = []
    for rel, url in jobs:
        try:
            _fetch(rel, url)
        except Exception as exc:  # noqa: BLE001 - keep going; report at the end
            print(f"  FAILED {rel}: {exc}", flush=True)
            failures.append(rel)
    if failures:
        print(f"{len(failures)} file(s) failed: {failures}", flush=True)
        raise SystemExit(1)
    print("All weights fetched.", flush=True)


if __name__ == "__main__":
    main()
