"""
impulse_testkit.py — Shared fixtures for the impulse-noise regression checks.

Kept separate from the check script so the synthetic images, the degradation
model and the metrics are defined in exactly one place, and so a change to a
diagnostic script cannot silently alter what the regression suite measures.

Everything here is OpenCV + NumPy only. The torch stub lets the classical
detectors be exercised on a machine without the deep-learning stack, which is
the point: these are checks on classical CV behaviour and they should run in CI
without 2 GB of model weights.
"""

from __future__ import annotations

import sys
import types
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ── torch stub ────────────────────────────────────────────────────────────────
# enhanced_detector imports defect_segmenter, which imports torch purely to
# define a U-Net class at module scope. When real torch is importable this is a
# no-op; otherwise a permissive stand-in satisfies every way the module touches
# torch at import time (subclassing nn.Module, @torch.no_grad() as a decorator,
# torch.cuda.is_available(), tensor type annotations).

def install_torch_stub() -> bool:
    """Install a stub `torch` if the real one is unusable. True if stubbed.

    Catches Exception rather than ModuleNotFoundError on purpose: an
    installed-but-unloadable torch (missing CUDA DLL, ABI mismatch against the
    local numpy) raises ImportError or OSError, and that is precisely the case
    the stub exists to survive. Letting it propagate would kill the suite at
    module-import time, before any check has run and before there is a report to
    print.
    """
    try:
        import torch  # noqa: F401
        return False
    except Exception:
        pass

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            # Act as an identity decorator when used as one. `@torch.no_grad()`
            # would otherwise replace the decorated method with this stub
            # instance, so the class would import cleanly and then return a
            # _Stub instead of an array if the method were ever reached.
            if len(a) == 1 and not k and callable(a[0]):
                return a[0]
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __getattr__(self, _name):
            return _Stub()

        @classmethod
        def __class_getitem__(cls, _item):
            return cls

    class _StubModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _Stub()

    for name in ("torch", "torch.nn", "torch.nn.functional", "torch.nn.init"):
        sys.modules[name] = _StubModule(name)
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]
    sys.modules["torch.nn"].Module = _Stub
    sys.modules["torch"].cuda = types.SimpleNamespace(is_available=lambda: False)
    return True


# ── degradation models ────────────────────────────────────────────────────────

def add_salt_pepper(rgb: np.ndarray, density: float, seed: int = 0) -> np.ndarray:
    """Classic impulse noise: a fraction `density` of pixels forced to 0 or 255.

    Applied per-pixel (all three channels together), which is what a scanner dust
    hit or a film emulsion pinhole does, and what makes the specks read as
    neutral black and white.
    """
    if density <= 0.0:
        return rgb.copy()
    rng = np.random.default_rng(seed)
    out = rgb.copy()
    r = rng.random(rgb.shape[:2])
    out[r < density / 2.0] = 0
    out[(r >= density / 2.0) & (r < density)] = 255
    return out


def add_channel_impulse(rgb: np.ndarray, density: float, seed: int = 0) -> np.ndarray:
    """Per-channel impulse noise — each channel corrupted independently.

    This is what heavy JPEG recompression and some scanner sensors produce, and
    it is the case a luminance-only detector misses: a channel forced to 255
    while the other two are untouched shifts hue without necessarily making the
    pixel a luminance outlier.
    """
    if density <= 0.0:
        return rgb.copy()
    rng = np.random.default_rng(seed)
    out = rgb.copy()
    for c in range(out.shape[2]):
        r = rng.random(out.shape[:2])
        out[:, :, c][r < density / 2.0] = 0
        out[:, :, c][(r >= density / 2.0) & (r < density)] = 255
    return out


# ── synthetic fixtures ────────────────────────────────────────────────────────

def flat_gradient(h: int = 256, w: int = 256) -> np.ndarray:
    """Smooth gradients plus two hard-edged solids, no fine detail, no damage.

    Any damage coverage reported on this is a pure false positive, and any pixel
    the impulse denoiser changes on it is a pure false positive too.
    """
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = np.linspace(30, 220, w)[None, :]
    img[:, :, 1] = np.linspace(40, 200, h)[:, None]
    img[:, :, 2] = 120
    cv2.circle(img, (w // 2, h // 2), h // 4, (200, 180, 160), -1)
    cv2.rectangle(img, (20, 20), (80, 90), (90, 90, 90), -1)
    return img


def thin_structure(h: int = 256, w: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """A clean image built from 1px-wide structure, plus the mask of that structure.

    Returns (image, structure_mask). The structure is the thing an impulse filter
    must not touch: single-pixel-wide lines at several orientations, standing in
    for hair strands, eyelashes, and engraved detail. The lines have hard
    endpoints, which is the case that a naive isolation test clips.
    """
    img = np.full((h, w, 3), 150, np.uint8)
    # A mid-tone background with gentle shading, so the lines are not sitting on
    # a mathematically perfect constant.
    img[:, :, 0] = np.clip(140 + np.linspace(-20, 20, w)[None, :], 0, 255)
    img[:, :, 1] = np.clip(150 + np.linspace(-15, 15, h)[:, None], 0, 255)
    img[:, :, 2] = 145

    mask = np.zeros((h, w), np.uint8)
    rng = np.random.default_rng(11)
    for _ in range(40):                      # vertical / horizontal strands
        y, x = int(rng.integers(10, h - 30)), int(rng.integers(10, w - 30))
        length = int(rng.integers(6, 22))
        if rng.random() < 0.5:
            img[y:y + length, x] = 40
            mask[y:y + length, x] = 255
        else:
            img[y, x:x + length] = 40
            mask[y, x:x + length] = 255
    for _ in range(20):                      # diagonal strands
        y, x = int(rng.integers(10, h - 30)), int(rng.integers(10, w - 30))
        length = int(rng.integers(6, 18))
        for i in range(length):
            img[y + i, x + i] = 235
            mask[y + i, x + i] = 255
    return img, mask


def photographic(h: int = 240, w: int = 320) -> np.ndarray:
    """Smooth gradients, fine detail, and hard edges — a stand-in for a portrait."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 90 + 70 * np.sin(xx / 55.0) * np.cos(yy / 70.0)
    img = np.clip(np.dstack([base, base * 0.92 + 14, base * 0.80 + 26]), 0, 255).astype(np.uint8)
    rng = np.random.default_rng(7)
    for _ in range(120):                     # 1px strand segments
        y, x = int(rng.integers(6, h - 8)), int(rng.integers(6, w - 6))
        img[y:y + 5, x] = 30
    cv2.circle(img, (w // 3, h // 2), 40, (210, 195, 175), -1)
    cv2.rectangle(img, (w - 90, 30), (w - 30, 110), (45, 50, 60), -1)
    return img


# ── metrics ───────────────────────────────────────────────────────────────────

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def detail_energy(rgb: np.ndarray) -> float:
    """High-frequency energy. Smearing drives this toward zero."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.std(g - cv2.GaussianBlur(g, (0, 0), 1.5)))


def changed_fraction(before: np.ndarray, after: np.ndarray) -> float:
    """Fraction of pixels the filter modified in any channel."""
    return float(np.mean(np.any(before != after, axis=2)))


def coverage(mask: np.ndarray) -> float:
    """Fraction of the frame a binary mask occupies."""
    if mask is None or mask.size == 0:
        return 0.0
    return float(np.sum(mask > 0)) / float(mask.size)


def component_count(mask: np.ndarray) -> int:
    if mask is None or not np.any(mask > 0):
        return 0
    n, _, _, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    return max(0, int(n) - 1)


def real_sources(repo: str, names: Tuple[str, ...] = ()) -> List[Tuple[str, np.ndarray]]:
    """Load whichever test_artifacts/sources images are present.

    Optional by design: the regression suite must pass on a clean checkout that
    does not ship sample photographs, so real images add coverage when available
    rather than being a prerequisite.
    """
    import os
    base = os.path.join(repo, "test_artifacts", "sources")
    if not os.path.isdir(base):
        return []
    wanted = names or tuple(sorted(
        f for f in os.listdir(base) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ))
    out: List[Tuple[str, np.ndarray]] = []
    for name in wanted:
        path = os.path.join(base, name)
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        out.append((name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return out
