# -*- coding: utf-8 -*-
"""
install_ddcolor.py — one-time setup for the DDColor colorizer (ICCV 2023).

What this does
--------------
  1. Vendors the official DDColor inference sources from GitHub
     (github.com/piddnad/DDColor, Apache-2.0) into
     backend/ai/colorization/ddcolor/, rewriting their imports so the tree is
     self-contained.
  2. Downloads the checkpoint from the author's official HuggingFace repo
     (huggingface.co/piddnad/DDColor-models) into
     backend/ai/weights/ddcolor/.
  3. Verifies the result: builds the network, checks the checkpoint keys match,
     and runs one real forward pass.

Why vendor instead of `pip install git+https://github.com/piddnad/DDColor`?
DDColor's setup.py packages **basicsr 1.3.4.6** with CUDA C++ extensions.
Installing it would overwrite the site-packages `basicsr` 1.4.2 that GFPGAN and
facexlib depend on (breaking face restoration and face detection) and would
require a CUDA/C++ toolchain. Vendoring the six inference modules avoids both.

No ModelScope. No ~/.cache. No venv. Everything lands in the project workspace.

Run (system Python 3.12 — never a venv):
    & "C:\\Program Files\\Python312\\python.exe" backend/ai/install_ddcolor.py

Options:
    --variant {modelscope,artistic,paper,paper_tiny}   default: modelscope
    --ref GIT_REF     pin a different DDColor commit/branch (default: DEFAULT_REF)
    --force           re-vendor sources and re-download the checkpoint
    --skip-weights    only vendor the sources
    --skip-verify     skip the build + forward-pass check
"""

from __future__ import annotations

import argparse
import io
import os
import ssl
import sys
import urllib.error
import urllib.request

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))          # backend/ai
BACKEND_DIR  = os.path.dirname(SCRIPT_DIR)                         # backend
VENDOR_DIR   = os.path.join(SCRIPT_DIR, "colorization", "ddcolor")
WEIGHTS_DIR  = os.path.join(SCRIPT_DIR, "weights", "ddcolor")

# ── Upstream source (pinned for reproducibility) ──────────────────────────────
REPO        = "piddnad/DDColor"
DEFAULT_REF = "2adb63f2656ac41cbdf7b894cddd94121a3faf13"   # master @ 2026-01-17
LICENSE     = "Apache-2.0 (DDColor — Alibaba DAMO Academy)"

# (path in the upstream repo, path under VENDOR_DIR)
VENDOR_FILES = [
    ("ddcolor/model.py",                                     "model.py"),
    ("ddcolor/pipeline.py",                                   "pipeline.py"),
    ("basicsr/archs/ddcolor_arch_utils/unet.py",              "arch_utils/unet.py"),
    ("basicsr/archs/ddcolor_arch_utils/convnext.py",           "arch_utils/convnext.py"),
    ("basicsr/archs/ddcolor_arch_utils/transformer_utils.py",  "arch_utils/transformer_utils.py"),
    ("basicsr/archs/ddcolor_arch_utils/position_encoding.py",  "arch_utils/position_encoding.py"),
]

# The only edit applied to vendored code: absolute basicsr imports → relative.
IMPORT_REWRITES = [
    ("from basicsr.archs.ddcolor_arch_utils.", "from .arch_utils."),
]

# ── Checkpoints (official HuggingFace repo of the DDColor author) ─────────────
HF_REPO = "piddnad/DDColor-models"
VARIANTS = {
    # name          filename                    approx MB   encoder
    "modelscope":  ("ddcolor_modelscope.pth",       870,    "convnext-l"),
    "artistic":    ("ddcolor_artistic.pth",         870,    "convnext-l"),
    "paper":       ("ddcolor_paper.pth",            870,    "convnext-l"),
    "paper_tiny":  ("ddcolor_paper_tiny.pth",       210,    "convnext-t"),
}
DEFAULT_VARIANT = "modelscope"

PACKAGE_INIT = '''"""
Vendored DDColor inference package (ICCV 2023).

Fetched from https://github.com/piddnad/DDColor by
backend/ai/install_ddcolor.py — see _VENDOR_INFO.md for the pinned revision.
Licensed Apache-2.0. Do not hand-edit the vendored modules; re-run the
installer with --force to refresh them.

The upstream modules live in a `basicsr` package that would collide with the
site-packages basicsr 1.4.2 required by GFPGAN, so their imports are rewritten
to stay inside this directory.
"""

from .model import DDColor
from .pipeline import (
    ColorizationPipeline,
    build_ddcolor_model,
    load_checkpoint_state_dict,
)

__all__ = [
    "DDColor",
    "ColorizationPipeline",
    "build_ddcolor_model",
    "load_checkpoint_state_dict",
]
'''

ARCH_UTILS_INIT = '''"""Vendored DDColor architecture utilities (Apache-2.0). Generated — see ../_VENDOR_INFO.md."""
'''


# ── Console helpers ──────────────────────────────────────────────────────────

def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _bar(done: int, total: int) -> None:
    if total <= 0:
        sys.stdout.write(f"\r  {_human_bytes(done)} downloaded ...")
    else:
        pct = int(done * 100 / total)
        filled = pct // 5
        sys.stdout.write(
            f"\r  [{'#' * filled}{'-' * (20 - filled)}] {pct:3d}%  "
            f"{_human_bytes(done)}/{_human_bytes(total)}"
        )
    sys.stdout.flush()


# ── HTTP ─────────────────────────────────────────────────────────────────────

def _open(url: str, timeout: int = 120):
    """Open a URL with certificate verification, falling back to unverified.

    Corporate TLS interception and stale Windows root stores are common enough
    that a hard failure here would be unhelpful, but silently skipping
    verification for a 870 MB checkpoint is worse — so the fallback is loud.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "pixrevive-installer"})
    try:
        return urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context())
    except urllib.error.URLError as exc:
        if not isinstance(getattr(exc, "reason", None), ssl.SSLError):
            raise
        print(f"\n  [WARN] TLS verification failed ({exc.reason}); retrying unverified.")
        return urllib.request.urlopen(req, timeout=timeout, context=ssl._create_unverified_context())


def _fetch_text(url: str) -> str:
    with _open(url, timeout=60) as resp:
        return resp.read().decode("utf-8")


def _download(url: str, dest: str, force: bool = False) -> bool:
    """Stream url → dest through a .part file so a partial download is never
    mistaken for a finished one."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.isfile(dest) and not force:
        size = os.path.getsize(dest)
        if size > 1_000_000:
            print(f"  [SKIP] Already present ({_human_bytes(size)}): {os.path.basename(dest)}")
            return True
        print(f"  [WARN] Truncated file found — re-downloading: {os.path.basename(dest)}")

    part = dest + ".part"
    print(f"  URL:  {url}")
    print(f"  Dest: {dest}")
    try:
        with _open(url) as resp, open(part, "wb") as fh:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                _bar(done, total)
        print()
        if total and done != total:
            raise IOError(f"incomplete download: got {done} of {total} bytes")
        os.replace(part, dest)
        print(f"  [OK] Saved {os.path.basename(dest)} ({_human_bytes(os.path.getsize(dest))})")
        return True
    except Exception as exc:
        print(f"\n  [ERROR] Download failed: {exc}")
        for path in (part, dest):
            if os.path.exists(path) and path.endswith(".part"):
                os.remove(path)
        return False


# ── Step 1: vendor the upstream sources ──────────────────────────────────────

def vendor_sources(ref: str, force: bool = False) -> bool:
    print(f"\n[1/3] Vendoring DDColor sources from github.com/{REPO} @ {ref[:12]}")
    print(f"  Dest: {VENDOR_DIR}")

    os.makedirs(os.path.join(VENDOR_DIR, "arch_utils"), exist_ok=True)
    ok = True

    for remote_rel, local_rel in VENDOR_FILES:
        local_path = os.path.join(VENDOR_DIR, local_rel.replace("/", os.sep))
        if os.path.isfile(local_path) and not force:
            print(f"  [SKIP] {local_rel}")
            continue

        url = f"https://raw.githubusercontent.com/{REPO}/{ref}/{remote_rel}"
        try:
            source = _fetch_text(url)
        except Exception as exc:
            print(f"  [ERROR] {local_rel}: {exc}")
            ok = False
            continue

        rewritten = source
        for old, new in IMPORT_REWRITES:
            rewritten = rewritten.replace(old, new)

        header = (
            "# ─── VENDORED FILE — DO NOT EDIT BY HAND ───────────────────────────────────\n"
            f"# Source : https://github.com/{REPO}/blob/{ref}/{remote_rel}\n"
            f"# License: {LICENSE}\n"
            "# Fetched by backend/ai/install_ddcolor.py (re-run with --force to refresh).\n"
            "# Only edit applied: `basicsr.archs.ddcolor_arch_utils.*` imports rewritten to\n"
            "# relative imports so this tree cannot collide with site-packages basicsr.\n"
            "# ──────────────────────────────────────────────────────────────────────────────\n"
        )
        with open(local_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(header + rewritten)
        changed = "rewritten" if rewritten != source else "verbatim"
        print(f"  [OK] {local_rel}  ({len(rewritten.splitlines())} lines, {changed})")

    # Package markers are ours, not upstream's — write them if absent.
    for rel, content in (
        ("__init__.py", PACKAGE_INIT),
        (os.path.join("arch_utils", "__init__.py"), ARCH_UTILS_INIT),
    ):
        path = os.path.join(VENDOR_DIR, rel)
        if force or not os.path.isfile(path):
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(content)
            print(f"  [OK] {rel.replace(os.sep, '/')}  (generated)")

    info = os.path.join(VENDOR_DIR, "_VENDOR_INFO.md")
    with open(info, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            f"# Vendored DDColor sources\n\n"
            f"- Upstream: https://github.com/{REPO}\n"
            f"- Revision: `{ref}`\n"
            f"- License:  {LICENSE}\n"
            f"- Fetched by: `backend/ai/install_ddcolor.py`\n\n"
            f"## Files\n\n"
            + "".join(f"- `{local}` ← `{remote}`\n" for remote, local in VENDOR_FILES)
            + "\n## Local modifications\n\n"
            "Imports of `basicsr.archs.ddcolor_arch_utils.*` are rewritten to relative\n"
            "imports (`.arch_utils.*`). Nothing else is changed.\n\n"
            "Upstream's modules ship inside a `basicsr` package that is a *different*\n"
            "project from the PyPI `basicsr==1.4.2` this backend installs for GFPGAN and\n"
            "facexlib. Installing DDColor with pip would overwrite it, so the six\n"
            "inference modules are vendored here instead.\n\n"
            "Refresh with:\n\n"
            "```\npython backend/ai/install_ddcolor.py --force\n```\n"
        )
    print(f"  [OK] _VENDOR_INFO.md  (provenance recorded)")
    return ok


# ── Step 2: download the checkpoint ──────────────────────────────────────────

def download_weights(variant: str, force: bool = False) -> bool:
    filename, approx_mb, encoder = VARIANTS[variant]
    print(f"\n[2/3] Downloading checkpoint '{variant}' (~{approx_mb} MB, {encoder}) from HuggingFace")
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/{filename}"
    return _download(url, os.path.join(WEIGHTS_DIR, filename), force=force)


# ── Step 3: verify ───────────────────────────────────────────────────────────

def verify(variant: str) -> bool:
    filename, _, encoder = VARIANTS[variant]
    weight_path = os.path.join(WEIGHTS_DIR, filename)
    print("\n[3/3] Verifying: build network, match checkpoint keys, run a forward pass")
    print("  (CPU build of the convnext-l encoder takes ~30 s — this is a one-time check.)")

    if BACKEND_DIR not in sys.path:
        sys.path.insert(0, BACKEND_DIR)

    try:
        import time

        import numpy as np
        import torch

        from ai.colorization.ddcolor import (  # type: ignore[import-not-found]
            DDColor,
            ColorizationPipeline,
            build_ddcolor_model,
            load_checkpoint_state_dict,
        )

        state_dict = load_checkpoint_state_dict(weight_path, map_location="cpu")
        print(f"  [OK] Checkpoint parsed: {len(state_dict)} tensors")

        t0 = time.time()
        model = build_ddcolor_model(
            DDColor,
            model_path=weight_path,
            input_size=512,
            model_size="tiny" if encoder == "convnext-t" else "large",
            device=torch.device("cpu"),
        )
        print(f"  [OK] Network built and weights loaded in {time.time() - t0:.1f}s")

        # build_ddcolor_model loads with strict=False; report what did not match
        # so a wrong-variant checkpoint cannot pass silently.
        report = model.load_state_dict(state_dict, strict=False)
        missing = [k for k in report.missing_keys if not k.startswith(("mean", "std"))]
        if missing or report.unexpected_keys:
            print(f"  [WARN] key mismatch — missing={len(missing)} unexpected={len(report.unexpected_keys)}")
            for k in (missing + list(report.unexpected_keys))[:5]:
                print(f"         {k}")
        else:
            print("  [OK] All checkpoint keys matched the network")

        # A horizontal luminance ramp: a real signal, so a dead model shows up.
        ramp = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
        bgr = np.stack([ramp] * 3, axis=-1)

        t0 = time.time()
        out = ColorizationPipeline(model, input_size=512).process(bgr)
        print(f"  [OK] Forward pass on 256x256 in {time.time() - t0:.1f}s → {out.shape} {out.dtype}")

        if out.shape != bgr.shape or out.dtype != np.uint8:
            print("  [ERROR] Unexpected output shape/dtype.")
            return False

        chroma = float(np.abs(out.astype(np.int16) - np.stack([ramp] * 3, -1).astype(np.int16)).mean())
        print(f"  [OK] Mean chroma deviation from grayscale input: {chroma:.2f}")
        return True
    except Exception as exc:
        print(f"  [ERROR] Verification failed: {type(exc).__name__}: {exc}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def _force_utf8_stdout() -> None:
    """Windows consoles default to cp1252 and choke on the box-drawing output.

    Done here rather than at import time: ddcolor_colorizer imports
    download_weights from this module, and reassigning the server's stdout as a
    side effect of that import would be hostile.
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            return
        except Exception:
            pass
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main() -> None:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="Install DDColor into the PixRevive workspace.")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default=DEFAULT_VARIANT,
                        help=f"checkpoint to download (default: {DEFAULT_VARIANT})")
    parser.add_argument("--ref", default=DEFAULT_REF, help="DDColor git ref to vendor from")
    parser.add_argument("--force", action="store_true", help="re-vendor sources and re-download")
    parser.add_argument("--skip-weights", action="store_true", help="only vendor the sources")
    parser.add_argument("--skip-verify", action="store_true", help="skip the build + forward check")
    args = parser.parse_args()

    print("=" * 74)
    print("  DDColor Colorizer Setup  (ICCV 2023 — piddnad/DDColor, Apache-2.0)")
    print(f"  Sources : {VENDOR_DIR}")
    print(f"  Weights : {WEIGHTS_DIR}")
    print(f"  Variant : {args.variant}")
    print("  No ModelScope, no ~/.cache, no venv.")
    print("=" * 74)

    vendored = vendor_sources(args.ref, force=args.force)
    fetched = True if args.skip_weights else (download_weights(args.variant, force=args.force) if vendored else False)
    verified = True
    if not args.skip_verify:
        verified = verify(args.variant) if (vendored and fetched and not args.skip_weights) else False

    print("\n" + "=" * 74)
    if vendored and fetched and verified:
        print("  [DONE] DDColor is ready for local inference.")
        print("  Restart the backend to activate it.")
    else:
        failed = [
            name for name, ok in
            (("vendor sources", vendored), ("download weights", fetched), ("verification", verified))
            if not ok
        ]
        print(f"  [PARTIAL] Failed: {', '.join(failed)}")
        print("  Check the log above and re-run. Network errors are safe to retry.")
    print("=" * 74)


if __name__ == "__main__":
    main()
