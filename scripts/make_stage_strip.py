"""
make_stage_strip.py — Build contact sheets of pipeline stage snapshots and chroma-delta heatmaps.

Usage:
    & "C:/Program Files/Python312/python.exe" scripts/make_stage_strip.py --dir debug_output/case4_diagonal_tear_portrait -o strip_case4.png
    & "C:/Program Files/Python312/python.exe" scripts/make_stage_strip.py debug_output/01_original.png debug_output/06_after_colorize.png -o strip.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

LABEL_H = 24


def zoom(img: np.ndarray, target_long: int) -> np.ndarray:
    h, w = img.shape[:2]
    factor = max(1, int(round(target_long / max(h, w))))
    return cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_NEAREST)


def build_strip_from_images(image_paths: list[Path], output_path: Path, target_long: int = 500, cols: int = 4):
    tiles = []
    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        z = zoom(img, target_long)
        canvas = np.full((z.shape[0] + LABEL_H, z.shape[1], 3), 245, dtype=np.uint8)
        canvas[LABEL_H:, :] = z
        label = p.stem[:45]
        cv2.putText(canvas, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (30, 30, 30), 1, cv2.LINE_AA)
        tiles.append(canvas)

    if not tiles:
        print(f"No valid images to build strip for {output_path}")
        return 1

    cell_h = max(t.shape[0] for t in tiles)
    cell_w = max(t.shape[1] for t in tiles)
    padded = []
    for t in tiles:
        c = np.full((cell_h, cell_w, 3), 245, dtype=np.uint8)
        c[: t.shape[0], : t.shape[1]] = t
        padded.append(c)

    cols = max(1, cols)
    rows = [padded[i : i + cols] for i in range(0, len(padded), cols)]
    row_imgs = []
    for r in rows:
        while len(r) < cols:
            r.append(np.full((cell_h, cell_w, 3), 245, dtype=np.uint8))
        row_imgs.append(np.hstack(r))
    grid = np.vstack(row_imgs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)
    print(f"Wrote {output_path} ({grid.shape[1]}x{grid.shape[0]}, {len(tiles)} tiles)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Make stage strip contact sheet")
    ap.add_argument("images", nargs="*", type=Path, help="Image files to combine")
    ap.add_argument("--dir", type=Path, default=None, help="Directory containing stage snapshots")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output image file")
    ap.add_argument("--long", type=int, default=500, help="Target long side per tile")
    ap.add_argument("--cols", type=int, default=4, help="Number of columns in strip")
    args = ap.parse_args()

    images = list(args.images)
    if args.dir and args.dir.is_dir():
        # Find all numbered stage PNGs in directory
        found = [p for p in args.dir.glob("*.png") if not p.name.startswith("strip")]
        found.sort(key=lambda p: p.name)
        images.extend(found)

    if not images:
        # Default: scan debug_output for all case directories and build strips
        debug_root = Path("debug_output")
        if debug_root.is_dir():
            for case_dir in sorted(debug_root.iterdir()):
                if case_dir.is_dir():
                    case_images = sorted([p for p in case_dir.glob("*.png") if not p.name.startswith("strip")])
                    if case_images:
                        out_file = case_dir / "00_stage_strip.png"
                        build_strip_from_images(case_images, out_file, args.long, args.cols)
            return 0
        else:
            print("No images or debug_output directory found.")
            return 1

    output = args.output or Path("debug_output/stage_strip.png")
    return build_strip_from_images(images, output, args.long, args.cols)


if __name__ == "__main__":
    raise SystemExit(main())
