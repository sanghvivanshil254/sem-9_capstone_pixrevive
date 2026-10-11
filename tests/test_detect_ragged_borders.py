"""Committed coverage for ImageNormalizer.detect_ragged_borders (plan §2.5 item 1).

Cases ported from the 2026-10-12 session smoke tests:
- ragged cream L/R strips on a mid-tone portrait -> crop fires
- clean borderless image -> inert
- interior flakes only (no strips) -> inert
- PIXREVIVE_RAGGED_BORDER=0 -> disabled
Every case asserts both the hit/miss AND the crop bbox when hit.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from ai.preprocessing.image_normalizer import ImageNormalizer  # noqa: E402


def _portrait(colour=120) -> np.ndarray:
    img = np.full((600, 600, 3), colour, np.uint8)
    cv2.circle(img, (300, 300), 150, (int(colour * 0.7), colour, int(colour * 1.4) % 256), -1)
    cv2.circle(img, (250, 260), 14, (40, 40, 45), -1)
    cv2.circle(img, (350, 260), 14, (40, 40, 45), -1)
    return img


def _add_cream_strips(img: np.ndarray, width: int = 58, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    for x in range(width):
        col_bright = int(195 + rng.integers(-18, 25))
        mask = rng.random(h) > 0.25 + 0.5 * x / width
        img[:, x][mask] = np.clip(col_bright + rng.integers(-10, 10), 0, 255)
        img[:, w - 1 - x][mask] = np.clip(col_bright + rng.integers(-10, 10), 0, 255)
    return img


def test_ragged_left_right_strips_crop():
    img = _add_cream_strips(_portrait(), seed=0)
    out, hit, bbox = ImageNormalizer.detect_ragged_borders(img)
    assert hit is True
    y0, y1, x0, x1 = bbox
    assert x0 >= 55 and x1 <= 545, f"expected inner crop x in [55,545], got {bbox}"
    assert out.shape[1] <= 500, f"expected cropped width <= 500, got {out.shape[1]}"


def test_clean_image_inert():
    img = _portrait(140)
    out, hit, _ = ImageNormalizer.detect_ragged_borders(img)
    assert hit is False
    assert out.shape == img.shape


def test_interior_flakes_only_inert():
    rng = np.random.default_rng(1)
    img = _portrait(130)
    for _ in range(30):
        x, y, s = rng.integers(150, 450), rng.integers(150, 450), rng.integers(4, 17)
        img[max(0, y - s):y + s, max(0, x - s):x + s] = 235
    out, hit, _ = ImageNormalizer.detect_ragged_borders(img)
    assert hit is False


def test_flag_off_disables():
    img = _add_cream_strips(_portrait(), seed=2)
    old = os.environ.get("PIXREVIVE_RAGGED_BORDER")
    try:
        os.environ["PIXREVIVE_RAGGED_BORDER"] = "0"
        out, hit, _ = ImageNormalizer.detect_ragged_borders(img)
        assert hit is False
        assert out.shape == img.shape
    finally:
        if old is None:
            os.environ.pop("PIXREVIVE_RAGGED_BORDER", None)
        else:
            os.environ["PIXREVIVE_RAGGED_BORDER"] = old
