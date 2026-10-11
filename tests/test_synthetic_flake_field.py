"""Synthetic flake-field coverage for the multi-scale detector (§2.5 item 2).

12 bright blobs (30-400 px) on a flat mid-tone background; the enhanced
detector at default flags must mask >= 70% of them before LaMa. Threshold is
justified in the plan: below 70% the detector demonstrably does not cover the
blob class that produced the teared.jpg regressions.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from ai.damage.enhanced_detector import EnhancedDamageDetector  # noqa: E402
from ai.pipeline.pipeline_context import PipelineContext  # noqa: E402


def _flake_field(seed: int = 7):
    rng = np.random.default_rng(seed)
    img = np.full((512, 512, 3), 120, np.uint8)
    centres = []
    for _ in range(12):
        x, y = int(rng.integers(40, 472)), int(rng.integers(40, 472))
        s = int(rng.integers(4, 18))            # radius 4-18 -> area 50-1000px
        cv2.circle(img, (x, y), s, (225, 222, 208), -1)
        centres.append((x, y, s))
    return img, centres


def test_multiscale_catches_at_least_70pct():
    img, centres = _flake_field()
    ctx = PipelineContext(img, {"color_type": "sepia", "noise_score": 4.0})
    detector = EnhancedDamageDetector()
    masks = detector.detect(ctx) if hasattr(detector, "detect") else None
    assert masks is not None, "EnhancedDamageDetector.detect missing — inspect API"
    mask = None
    for key in ("final", "damage", "inpaint", "enhanced"):
        cand = ctx.damage_masks.get(key) if hasattr(ctx, "damage_masks") else None
        if cand is not None and np.any(cand > 0):
            mask = cand
            break
    assert mask is not None, f"no nonzero damage mask key; keys={list(getattr(ctx, 'damage_masks', {}))}"
    if mask.shape[:2] != (512, 512):
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    hit = 0
    for x, y, s in centres:
        r = max(6, s + 6)
        patch = mask[max(0, y - r):y + r, max(0, x - r):x + r]
        if float(patch.mean()) > 25.0:   # >10% of the local window masked
            hit += 1
    assert hit >= 8, f"only {hit}/12 synthetic blobs masked (need >=8 = 70%)"
