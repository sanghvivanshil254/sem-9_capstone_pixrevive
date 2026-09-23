# -*- coding: utf-8 -*-
"""
download_weights.py - Download all PixRevive AI model weights.

Downloads (all to backend/ai/weights/ as the single canonical location):
  1. GFPGANv1.4          — face restoration        (~349 MB)
  2. RealESRGAN x4plus   — super-resolution         ( ~67 MB)
  3. big-lama             — LaMa inpainting          (~196 MB)
  4. CodeFormer           — alternate face restorer  (~377 MB)
  5. detection_Resnet50   — face detector (facelib)  (~109 MB)
  6. parsing_parsenet     — face parser  (facelib)   ( ~85 MB)
  7. DDColor              — transformer colorization (~870 MB)

Note: DDColor also needs its inference sources vendored, which this script does
not do. Run the dedicated installer instead — it fetches the sources, downloads
this same checkpoint and verifies both:
    python backend/ai/install_ddcolor.py
This script only downloads the weight file.

Usage:
    python backend/ai/download_weights.py
    python backend/ai/download_weights.py --force   # re-download even if file exists
    python backend/ai/download_weights.py --weights_dir path/to/dir
"""

import os
import sys
import io
import hashlib
import argparse
import ssl
import urllib.request

# Force UTF-8 stdout on Windows to avoid charmap codec errors
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Canonical download manifest ──────────────────────────────────────────────
# (url, relative_path_under_weights_dir, expected_sha256_prefix_or_None)
DOWNLOADS = [
    (
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        "GFPGANv1.4.pth",
        None,   # skip checksum (large file, trust GitHub release)
    ),
    (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x4plus.pth",
        None,
    ),
    (
        "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
        "big-lama.pt",
        None,
    ),
    (
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        os.path.join("CodeFormer", "codeformer.pth"),
        None,
    ),
    (
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        os.path.join("facelib", "detection_Resnet50_Final.pth"),
        None,
    ),
    (
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",

        # T-3 deblur (Phase 4B.6): Restormer motion deblurring checkpoint.
        # Loaded lazily by ai.restoration.deblurrer on first gate-fire only.
        (
            "https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth",
            os.path.join("restormer", "motion_deblurring.pth"),
            None,   # skip checksum (large file, trust GitHub release)
        ),
        os.path.join("facelib", "parsing_parsenet.pth"),
        None,
    ),
    # DDColor (ICCV 2023) — replaces the retired ECCV16/Caffe colorizer.
    # Stored in weights/ddcolor/ (project workspace, not ~/.cache).
    (
        "https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_modelscope.pth",
        os.path.join("ddcolor", "ddcolor_modelscope.pth"),
        None,
    ),
    # Learned Semantic Defect Segmenter (Microsoft Bringing Old Photos Back to Life / CVPR 2020)
    # Stored in weights/defect_detect/ (project workspace, not ~/.cache).
    (
        "https://huggingface.co/databuzzword/bringing-old-photos-back-to-life/resolve/main/Global/checkpoints/detection/FT_Epoch_latest.pt",
        os.path.join("defect_detect", "FT_Epoch_latest.pt"),
        None,
    ),
]


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _progress_hook(count: int, block_size: int, total_size: int):
    if total_size <= 0:
        downloaded = count * block_size
        sys.stdout.write(f"\r  {_human_bytes(downloaded)} downloaded ...")
    else:
        downloaded = min(count * block_size, total_size)
        pct = int(downloaded * 100 / total_size)
        filled = pct // 5
        bar = "#" * filled + "-" * (20 - filled)
        sys.stdout.write(
            f"\r  [{bar}] {pct:3d}%  {_human_bytes(downloaded)}/{_human_bytes(total_size)}"
        )
    sys.stdout.flush()


def _sha256_prefix(path: str, length: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def download_file(url: str, out_path: str, force: bool = False, expected_sha: str | None = None) -> bool:
    """
    Download url → out_path.
    Returns True on success, False on failure.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path) and not force:
        size = os.path.getsize(out_path)
        if size > 0:
            print(f"  [SKIP] Already exists ({_human_bytes(size)}): {os.path.basename(out_path)}")
            return True
        else:
            print(f"  [WARN] Zero-byte file found — re-downloading: {os.path.basename(out_path)}")

    print(f"  Downloading: {url}")
    ctx = ssl._create_unverified_context()
    try:
        urllib.request.urlretrieve(url, out_path, _progress_hook)
        print()  # newline after progress bar
        size = os.path.getsize(out_path)
        print(f"  [OK] Saved {os.path.basename(out_path)} ({_human_bytes(size)})")

        if expected_sha:
            actual = _sha256_prefix(out_path)
            if actual == expected_sha:
                print(f"  [OK] SHA256 verified: {actual}...")
            else:
                print(f"  [WARN] SHA256 mismatch: expected={expected_sha} got={actual}")

        return True
    except Exception as exc:
        print(f"\n  [ERROR] Failed: {exc}")
        # Remove partial file
        if os.path.exists(out_path):
            os.remove(out_path)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download PixRevive AI model weights.")
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights"),
        help="Directory to store weights (default: backend/ai/weights/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist",
    )
    args = parser.parse_args()

    weights_dir = args.weights_dir
    os.makedirs(weights_dir, exist_ok=True)

    print("=" * 60)
    print("  PixRevive — Fresh AI Weight Downloader")
    print(f"  Target directory: {weights_dir}")
    print(f"  Force re-download: {args.force}")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    for idx, (url, rel_path, expected_sha) in enumerate(DOWNLOADS, 1):
        out_path = os.path.join(weights_dir, rel_path)
        name = os.path.basename(out_path)
        print(f"\n[{idx}/{len(DOWNLOADS)}] {name}")
        ok = download_file(url, out_path, force=args.force, expected_sha=expected_sha)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"  Done! {success_count} succeeded, {fail_count} failed.")
    if fail_count:
        print("  [!] Some downloads failed. Run again to retry.")
    print("  DDColor weights are stored under weights/ddcolor/.")
    print("  Run: python backend/ai/install_ddcolor.py  (first time only — vendors the sources).")
    print("  Restart the backend to load fresh weights.")
    print("=" * 60)


if __name__ == "__main__":
    main()
