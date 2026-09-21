"""
probe_flake_masks.py — Find a flake mask that actually covers the white emulsion
loss on a sepia scan.

Two shipped detectors both miss it on test_artifacts/sources/teared.jpg:

  * TearmarkDetector needs `S < S_local - 3`, which is False at every pixel once
    the pipeline has desaturated the sepia input (S == 0 => 0 < -3). Both of its
    passes report exactly 0.00%.
  * EnhancedDamageDetector.detect_white_flakes needs L > 235 and V > 225, but the
    flakes on this scan are cream rather than paper-white (L p99 = 241), so it
    finds 0.5%.

This probe renders candidate masks as red overlays so the thresholds can be
chosen by looking at them rather than guessed.

Usage:
    PYTHONPATH=backend "C:/Program Files/Python312/python.exe" \
        scripts/probe_flake_masks.py test_artifacts/sources/teared.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _scales(L: np.ndarray, h: int, w: int) -> Dict[str, np.ndarray]:
    base = max(h, w)
    ks = {
        "s": max(9, int(base * 0.015)) | 1,
        "m": max(21, int(base * 0.035)) | 1,
        "l": max(41, int(base * 0.065)) | 1,
        "xl": max(81, int(base * 0.12)) | 1,
    }
    return {k: cv2.blur(L, (v, v)) for k, v in ks.items()}


def _clean(mask: np.ndarray, h: int, w: int, max_frac: float = 0.20) -> np.ndarray:
    """Close pinholes, drop specks, drop components larger than max_frac."""
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k3, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    lo, hi = 4, int(h * w * max_frac)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < lo or area > hi:
            out[labels == i] = 0
    return out


def build_candidates(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1].astype(np.float32)

    lm = _scales(L, h, w)
    mean, std, med = float(L.mean()), float(L.std()), float(np.median(L))
    floor = min(max(mean + 2.5 * std, med + 20), 200)
    # A percentile floor tracks the actual bright tail instead of assuming the
    # flakes reach 200+; on this scan p85 is well below the mean+2.5σ cap.
    p85, p80, p75 = (float(np.percentile(L, q)) for q in (85, 80, 75))
    S_local = cv2.blur(S, (max(21, int(max(h, w) * 0.035)) | 1,) * 2)

    # Relative desaturation: flakes are paper showing through, so they are less
    # saturated than their surroundings. Scaled by the local spread so it still
    # discriminates on a weakly tinted scan, and inert when S carries no signal.
    s_std = float(S.std())
    sat_rel = (S < (S_local - max(2.0, 0.15 * s_std))) if s_std > 2.0 else np.ones_like(S, bool)

    out: Dict[str, np.ndarray] = {}

    def add(name: str, m: np.ndarray, max_frac: float = 0.20) -> None:
        out[name] = _clean((m.astype(np.uint8) * 255), h, w, max_frac)

    # Shipped behaviour, for reference.
    add("a_shipped_pass1", (L - lm["s"] > 18) & (L - lm["m"] > 14) & (L - lm["l"] > 10)
        & (L > floor) & (S < 80) & (S < S_local - 3))
    add("b_shipped_pass2", (L - lm["xl"] > 8) & (L > max(floor - 10, med + 15))
        & (S < 80) & (S < S_local - 2))

    # Enhanced detector's near-pure-white rule.
    add("c_enhanced_white", (L > 235) & (hsv[:, :, 2].astype(np.float32) > 225)
        & (L - cv2.GaussianBlur(L, (11, 11), 0) > 35), 0.05)

    # Percentile-floored multi-scale contrast, saturation used relatively.
    add("d_pct85_multiscale", (L - lm["s"] > 12) & (L - lm["m"] > 9)
        & (L > p85) & sat_rel)
    add("e_pct80_multiscale", (L - lm["s"] > 10) & (L - lm["m"] > 8)
        & (L > p80) & sat_rel)
    add("f_pct80_coarse", (L - lm["xl"] > 6) & (L > p80) & sat_rel)
    add("g_pct75_coarse", (L - lm["xl"] > 5) & (L > p75) & sat_rel)

    # Union of the two percentile families — the shape the fix would ship.
    union = (((L - lm["s"] > 12) & (L - lm["m"] > 9) & (L > p85))
             | ((L - lm["xl"] > 6) & (L > p80))) & sat_rel
    add("h_union_pct", union)

    # The percentile families above all mask the forehead, cheeks and nose: a
    # global brightness floor cannot tell a bright flake on a dark shirt from a
    # smoothly bright face. Local contrast can — a flake is much brighter than
    # its immediate neighbourhood, skin is not. So keep the multi-scale contrast
    # test and relax the absolute floor instead of the other way round.
    ms = (L - lm["s"] > 18) & (L - lm["m"] > 14) & (L - lm["l"] > 10)
    add("i_ms_nofloor", ms & sat_rel)
    add("j_ms_medfloor", ms & (L > med) & sat_rel)
    ms_soft = (L - lm["s"] > 14) & (L - lm["m"] > 11) & (L - lm["l"] > 8)
    add("k_ms_soft_medfloor", ms_soft & (L > med) & sat_rel)
    ms_softer = (L - lm["s"] > 11) & (L - lm["m"] > 9) & (L - lm["l"] > 6)
    add("l_ms_softer_medfloor", ms_softer & (L > med) & sat_rel)
    # Coarse-scale companion for broad clumps, still local rather than absolute.
    add("m_ms_soft_plus_coarse",
        (ms_soft | ((L - lm["xl"] > 14) & (L - lm["m"] > 8))) & (L > med) & sat_rel)

    print(f"  floor(mean+2.5s,cap200)={floor:.0f}  p75={p75:.0f} p80={p80:.0f} "
          f"p85={p85:.0f}  median={med:.0f}  S.std={s_std:.1f}  "
          f"sat_rel={100*sat_rel.mean():.1f}%")
    return out


def overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    sel = mask > 0
    out[sel] = (0.35 * out[sel] + 0.65 * np.array([255, 40, 40])).astype(np.uint8)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("-d", "--outdir", type=Path,
                    default=ROOT / "debug_output" / "flake_masks")
    args = ap.parse_args()

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"!! cannot read {args.image}")
        return 1
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"{args.image.name}  {rgb.shape[1]}x{rgb.shape[0]}")
    cands = build_candidates(rgb)

    print(f"\n{'candidate':<22} {'coverage':>9}  {'components':>10}")
    tiles = []
    for name, mask in cands.items():
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cov = float(np.sum(mask > 0)) / mask.size
        print(f"{name:<22} {cov:>8.2%}  {n - 1:>10}")
        ov = overlay(rgb, mask)
        cv2.imwrite(str(args.outdir / f"{name}.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(args.outdir / f"{name}_mask.png"), mask)
        lab = ov.copy()
        cv2.putText(lab, f"{name} {cov:.1%}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 0), 1, cv2.LINE_AA)
        tiles.append(lab)

    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    grid_path = args.outdir / "_grid.png"
    cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"\nwrote {len(cands)} overlays + {grid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
