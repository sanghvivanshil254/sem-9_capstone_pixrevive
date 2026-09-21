"""
probe_stage0.py — Attribute the crackle/mosaic texture to a specific Stage-0 operator.

Stage 0 runs two suppressors in sequence: ImpulseDenoiser (isolated outliers) then
SpeckSuppressor (dense bright-speck field). The traced pipeline shows the mosaic
already present in 00b_after_despeckle, so one of these two makes it. This runs
each alone and measures what is left behind.

Two measurements matter.

  residual impulse density — the fraction of pixels that are still local extremes
    by a *polarity-split* criterion, so salt and pepper are counted separately.
    ImpulseDenoiser's own `density` reports what it *flagged*, which says nothing
    about what it missed.

  clustered fraction — of those impulses, how many have a same-polarity neighbour.
    ImpulseDenoiser's ROAD isolation test protects any pixel with two similar
    neighbours, so a clustered impulse is one it cannot flag by construction. This
    is the number that predicts whether the design assumption holds on an image.

Run:
    PYTHONPATH=backend "C:/Program Files/Python312/python.exe" scripts/probe_stage0.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "backend"))

from ai.preprocessing.impulse_denoiser import ImpulseDenoiser  # noqa: E402
from ai.preprocessing.speck_suppressor import SpeckSuppressor  # noqa: E402

OUT = os.path.join(REPO, "debug_output", "probe_stage0")


def impulse_stats(rgb: np.ndarray, t: float = 30.0) -> dict:
    """Polarity-split local-extreme census, independent of ImpulseDenoiser.

    A pixel counts as salt when it exceeds the median of a 5x5 window by t, and
    pepper when it falls below by t. A 5x5 median rather than 3x3 because at high
    density the 3x3 median is itself corrupted, which is precisely how a dense
    field hides from a 3x3-based detector.
    """
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    med = cv2.medianBlur(g.astype(np.uint8), 5).astype(np.float32)
    d = g - med
    salt = d > t
    pepper = d < -t

    def clustered(mask: np.ndarray) -> float:
        if not mask.any():
            return 0.0
        k = np.ones((3, 3), np.uint8)
        neighbours = cv2.filter2D(mask.astype(np.uint8), -1, k, borderType=cv2.BORDER_REFLECT_101)
        # subtract self, so >=1 means at least one same-polarity neighbour
        return float(np.mean((neighbours - mask.astype(np.uint8))[mask] >= 1))

    return {
        "salt": float(np.mean(salt)),
        "pepper": float(np.mean(pepper)),
        "total": float(np.mean(salt | pepper)),
        "salt_clustered": clustered(salt),
        "pepper_clustered": clustered(pepper),
    }


def row(label: str, rgb: np.ndarray) -> None:
    s = impulse_stats(rgb)
    amp, cov = SpeckSuppressor.measure(rgb)
    field = ImpulseDenoiser.measure(rgb)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    print(
        f"  {label:<34} "
        f"salt {s['salt']:>6.2%} (clust {s['salt_clustered']:>5.0%})  "
        f"pepper {s['pepper']:>6.2%} (clust {s['pepper_clustered']:>5.0%})  "
        f"| ID would flag {field.density:>6.2%} sigma {field.sigma:>5.1f} T {field.threshold:>5.1f}  "
        f"| tophat amp {amp:>5.1f} cov {cov:>5.1%}  mean {g.mean():>5.1f}"
    )


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for name in ("noise.jpg", "teared.jpg"):
        path = os.path.join(REPO, "test_artifacts", "sources", name)
        if not os.path.exists(path):
            continue
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        stem = os.path.splitext(name)[0]

        print(f"\n{'=' * 118}\n{name}  {rgb.shape[1]}x{rgb.shape[0]}\n{'=' * 118}")
        row("00 raw input", rgb)

        after_id, _ = ImpulseDenoiser.apply(rgb)
        row("01 after ImpulseDenoiser", after_id)

        after_speck = SpeckSuppressor.apply(after_id)
        row("02 after SpeckSuppressor (as shipped)", after_speck)

        # Counterfactual A: skip SpeckSuppressor entirely.
        row("A  ImpulseDenoiser only", after_id)

        # Counterfactual B: iterate ImpulseDenoiser to attack clusters.
        it = rgb
        for _ in range(4):
            it, _ = ImpulseDenoiser.apply(it)
        row("B  ImpulseDenoiser x4, no speck", it)

        # Counterfactual C: textbook adaptive median (grows 3->7 at each pixel).
        # A reference point, not a proposal: it is a blanket filter.
        amf = rgb.copy()
        for k in (3, 5, 7):
            med = cv2.medianBlur(amf, k)
            g = cv2.cvtColor(amf, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gm = cv2.cvtColor(med, cv2.COLOR_RGB2GRAY).astype(np.float32)
            bad = (np.abs(g - gm) > 30.0)[:, :, None]
            amf = np.where(bad, med, amf)
        row("C  adaptive median 3/5/7 reference", amf)

        for tag, img in (
            ("00_raw", rgb),
            ("01_after_impulse", after_id),
            ("02_after_speck", after_speck),
            ("B_impulse_x4", it),
            ("C_adaptive_median", amf),
        ):
            cv2.imwrite(
                os.path.join(OUT, f"{stem}_{tag}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )

    print(f"\nimages -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
