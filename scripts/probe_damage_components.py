"""
probe_damage_components.py — Characterise the connected components that
QualityAnalyzer counts as "damage", to separate impulse noise from real damage.

Real damage (tears, scratches, emulsion loss) forms a few large, spatially
clustered components. Impulse/film noise forms thousands of tiny components
spread uniformly over the whole frame. This probe prints the statistics that
tell the two apart.

Usage:
    PYTHONPATH=backend "C:/Program Files/Python312/python.exe" \
        scripts/probe_damage_components.py [image ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = [
    ROOT / "test_artifacts" / "sources" / "noise.jpg",
    ROOT / "test_artifacts" / "sources" / "teared.jpg",
]


def probe(path: Path) -> None:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"!! cannot read {path}")
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Mirror QualityAnalyzer.analyze's damage estimation exactly.
    k_size = max(15, min(31, int(max(h, w) * 0.02))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    tophat = cv2.morphologyEx(cv2.medianBlur(gray, 3), cv2.MORPH_TOPHAT, kernel)
    tophat_f = tophat.astype(np.float32)
    thr = max(28.0, float(np.mean(tophat_f) + 2.5 * np.std(tophat_f)))
    bright = (tophat >= thr).astype(np.uint8) * 255

    min_area = max(4, int(h * w * 0.00025))
    max_area = max(min_area + 1, int(h * w * 0.35))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(bright, connectivity=8)

    kept_areas, kept_cents = [], []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        extent = area / max(float(cw * ch), 1.0)
        scratch_like = max(cw, ch) >= 8 and min(cw, ch) <= max(3, int(0.2 * max(cw, ch)))
        flake_like = extent >= 0.12
        if scratch_like or flake_like:
            kept_areas.append(area)
            kept_cents.append(cents[i])

    kept_areas_a = np.asarray(kept_areas, dtype=np.float64)
    total_px = h * w
    damage_px = float(kept_areas_a.sum()) if kept_areas else 0.0

    print(f"\n=== {path.name}  {w}x{h} ({total_px} px) ===")
    print(f"  tophat kernel     : {k_size} px  ({k_size / max(h, w) * 100:.1f}% of long edge)")
    print(f"  bright threshold  : {thr:.1f}")
    print(f"  raw components    : {n - 1}")
    print(f"  min/max area gate : {min_area} .. {max_area} px")
    print(f"  kept components   : {len(kept_areas)}")
    print(f"  damage coverage   : {damage_px / total_px * 100:.2f}%")
    if not kept_areas:
        return
    print(f"  component area    : median={np.median(kept_areas_a):.0f} "
          f"mean={kept_areas_a.mean():.1f} max={kept_areas_a.max():.0f}")
    print(f"  largest component : {kept_areas_a.max() / total_px * 100:.3f}% of frame")
    print(f"  density           : {len(kept_areas) / (total_px / 10000):.1f} components "
          f"per 100x100 px block")

    # Spatial uniformity: how evenly the components spread over a 4x4 grid.
    # Uniform spread (noise) -> occupancy near 1.0 and low coefficient of
    # variation. Clustered damage -> many empty cells, high CV.
    gy, gx = 4, 4
    cell = np.zeros((gy, gx), dtype=np.int32)
    for cx, cy in kept_cents:
        j = min(gx - 1, int(cx / w * gx))
        i = min(gy - 1, int(cy / h * gy))
        cell[i, j] += 1
    occupancy = float(np.count_nonzero(cell)) / cell.size
    cv_ = float(cell.std() / (cell.mean() + 1e-6))
    print(f"  grid occupancy    : {occupancy:.2f} (1.00 = every cell has damage)")
    print(f"  grid CV           : {cv_:.2f} (low = uniform => noise, high = clustered)")
    print(f"  per-cell counts   :\n{cell}")


def main(paths: list[Path]) -> int:
    for p in paths:
        probe(p)
    return 0


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]] or DEFAULT_IMAGES
    raise SystemExit(main(args))
