"""
diagnose_impulse_noise.py — Instrument the damage-detection front end against
impulse (salt-and-pepper) noise.

Why this script exists
----------------------
Stage 2 of the restoration pipeline (fracture masking) runs *before* any noise
reduction: NoiseRouter is not invoked until Stage 4b. Every isolated bright
speck therefore reaches the flake/tearmark/crack detectors looking exactly like
an emulsion flake, and every isolated dark speck looks like a dust spot or the
start of a crack. The detectors' morphological closing then merges the speck
scatter into large connected components, MaskRefiner routes those to LaMa, and
LaMa repaints them — which is where the smeared, crackled, pastel-blotched
output comes from.

This script measures that claim instead of asserting it. For each test image it
reports detector coverage at a sweep of impulse-noise densities, both on the raw
noisy image and on the same image after impulse removal, so the causal link
between speck density and mask coverage is visible as a curve rather than a
single anecdote.

Run (system Python 3.12, per CLAUDE.md):
    PYTHONPATH=backend "C:/Program Files/Python312/python.exe" \
        scripts/diagnose_impulse_noise.py
"""

from __future__ import annotations

import os
import sys
import types
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ── torch stub ────────────────────────────────────────────────────────────────
# enhanced_detector imports defect_segmenter, which imports torch purely to
# define the U-Net. We only exercise the classical detectors here, so a stub
# keeps the script runnable on a CPU-only box without the DL stack. When real
# torch is importable this block is a no-op.
try:  # pragma: no cover
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover

    class _Stub:
        """Permissive stand-in: subclassable, callable, and usable as a
        context manager or decorator, which covers every way defect_segmenter
        touches torch at import time."""

        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
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

    for _name in ("torch", "torch.nn", "torch.nn.functional", "torch.nn.init"):
        sys.modules[_name] = _StubModule(_name)
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]
    sys.modules["torch.nn"].Module = _Stub
    sys.modules["torch"].cuda = types.SimpleNamespace(is_available=lambda: False)


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO, "backend") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "backend"))

from ai.analysis.quality_analyzer import QualityAnalyzer  # noqa: E402
from ai.damage.mask_refiner import MaskRefiner  # noqa: E402
from ai.damage.tearmark_detector import TearmarkDetector  # noqa: E402
from ai.pipeline.pipeline_context import PipelineContext  # noqa: E402
from ai.postprocessing.noise_router import NoiseClassifier, NoiseRouter  # noqa: E402


# ── synthetic degradation ─────────────────────────────────────────────────────

def add_salt_pepper(rgb: np.ndarray, density: float, seed: int = 0) -> np.ndarray:
    """Classic impulse noise: a fraction `density` of pixels forced to 0 or 255.

    Applied per-pixel (all three channels together) rather than per-channel,
    which is what a scanner dust hit or a film emulsion pinhole actually does
    and what makes the specks read as neutral black/white.
    """
    rng = np.random.default_rng(seed)
    out = rgb.copy()
    r = rng.random(rgb.shape[:2])
    out[r < density / 2.0] = 0
    out[(r >= density / 2.0) & (r < density)] = 255
    return out


def coverage(mask: np.ndarray) -> float:
    """Percentage of the frame a binary mask occupies."""
    if mask is None or mask.size == 0:
        return 0.0
    return float(np.sum(mask > 0) / mask.size * 100.0)


def blob_stats(mask: np.ndarray) -> Tuple[int, int]:
    """(component count, largest component area) for a binary mask."""
    if mask is None or not np.any(mask > 0):
        return 0, 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return 0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(n - 1), int(areas.max())


# ── detector probes ───────────────────────────────────────────────────────────

def probe(rgb: np.ndarray) -> Dict[str, object]:
    """Run the classical Stage-2 detectors on `rgb` and report what they claim."""
    from ai.damage.enhanced_detector import EnhancedDamageDetector

    ctx = PipelineContext(rgb)
    meta = QualityAnalyzer.analyze(rgb)
    ctx.image_metadata.update(meta)

    det = EnhancedDamageDetector()
    det.defect_segmenter = None  # force the classical path — no weights here

    flakes = det.detect_white_flakes(rgb, ctx)
    wide = det.detect_wide_cracks(rgb, ctx)
    thin = det.detect_thin_crack_network(rgb, ctx)
    spots = det.detect_dark_spots(rgb, ctx)

    tm = TearmarkDetector().detect(ctx)

    combined = cv2.bitwise_or(flakes, thin)
    combined = cv2.bitwise_or(combined, wide)
    combined = cv2.bitwise_or(combined, spots)
    ctx.damage_masks["tearmark"] = tm
    refined = MaskRefiner.refine(combined, ctx)

    profile = NoiseClassifier.classify(rgb, ctx.image_metadata)
    n_comp, biggest = blob_stats(refined)

    return {
        "color_type": meta["color_type"],
        "noise_score": meta["noise_score"],
        "damage_pct": meta["damage_percentage"],
        "impulse_family": profile.family_scores.get("impulse", 0.0),
        "impulse_ratio": profile.metrics.get("impulse_ratio", 0.0),
        "flakes": coverage(flakes),
        "wide": coverage(wide),
        "thin": coverage(thin),
        "spots": coverage(spots),
        "tearmark": coverage(tm),
        "final": coverage(refined),
        "components": n_comp,
        "largest_blob_pct": biggest / refined.size * 100.0,
        "mask": refined,
    }


def clean_impulse(rgb: np.ndarray) -> np.ndarray:
    """Run only the NoiseRouter impulse branch, as the pipeline would at 4b."""
    ctx = PipelineContext(rgb)
    ctx.image_metadata.update(QualityAnalyzer.analyze(rgb))
    profile = NoiseClassifier.classify(rgb, ctx.image_metadata)
    from ai.postprocessing.noise_router import _reduce_impulse

    return _reduce_impulse(rgb, profile)


# ── report ────────────────────────────────────────────────────────────────────

_ROW = "{:>7} {:>6} {:>7} {:>7} {:>7} {:>7} {:>8} {:>7} {:>7} {:>8}"


def header(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    print(_ROW.format("density", "impls", "flakes", "wide", "thin", "spots",
                      "tearmrk", "FINAL", "comps", "bigblob"))


def emit(label: str, r: Dict[str, object]) -> None:
    print(_ROW.format(
        label,
        f"{r['impulse_family']:.2f}",
        f"{r['flakes']:.2f}",
        f"{r['wide']:.2f}",
        f"{r['thin']:.2f}",
        f"{r['spots']:.2f}",
        f"{r['tearmark']:.2f}",
        f"{r['final']:.2f}",
        str(r["components"]),
        f"{r['largest_blob_pct']:.2f}",
    ))


def main() -> int:
    art = os.path.join(REPO, "test_artifacts", "sources")
    out = os.path.join(REPO, "debug_output", "impulse_diagnosis")
    os.makedirs(out, exist_ok=True)

    images: List[Tuple[str, np.ndarray]] = []
    for name in ("noise.jpg", "teared.jpg"):
        p = os.path.join(art, name)
        if os.path.exists(p):
            bgr = cv2.imread(p)
            images.append((name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

    # A synthetic control: smooth gradient + geometric structure, no damage at
    # all. Any coverage reported on this is a pure false positive.
    ctrl = np.zeros((256, 256, 3), np.uint8)
    ctrl[:, :, 0] = np.linspace(30, 220, 256)[None, :]
    ctrl[:, :, 1] = np.linspace(40, 200, 256)[:, None]
    ctrl[:, :, 2] = 120
    cv2.circle(ctrl, (128, 128), 60, (200, 180, 160), -1)
    cv2.rectangle(ctrl, (20, 20), (80, 90), (90, 90, 90), -1)
    images.append(("synthetic_clean", ctrl))

    densities = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08)

    for name, rgb in images:
        header(f"{name}  {rgb.shape[1]}x{rgb.shape[0]}   RAW (as Stage 2 sees it today)")
        for d in densities:
            noisy = add_salt_pepper(rgb, d) if d else rgb
            r = probe(noisy)
            emit(f"{d:.3f}", r)
            cv2.imwrite(
                os.path.join(out, f"{name}_d{d:.3f}_raw_mask.png"), r["mask"]
            )
            cv2.imwrite(
                os.path.join(out, f"{name}_d{d:.3f}_input.png"),
                cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR),
            )

        header(f"{name}   AFTER impulse removal (what Stage 2 should see)")
        for d in densities:
            noisy = add_salt_pepper(rgb, d) if d else rgb
            r = probe(clean_impulse(noisy))
            emit(f"{d:.3f}", r)
            cv2.imwrite(
                os.path.join(out, f"{name}_d{d:.3f}_cleaned_mask.png"), r["mask"]
            )

    print(f"\nMasks and inputs written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
