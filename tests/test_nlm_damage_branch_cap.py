"""NLM damage-branch cap test (§2.5 item 4).

Feeds a high-sigma fixture with bg_damaged=True into Stage 5 and asserts the
h_bg printed in the log respects PIXREVIVE_NLM_CAP (the damage branch is the
only branch where the cap binds; the clean log contract h_face=5.5, h_mid=7.5,
h_bg=6.5 is pinned by test_restoration_regressions and must stay intact).
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from ai.pipeline.pipeline_context import PipelineContext  # noqa: E402
from ai.pipeline.restoration_pipeline import RestorationPipeline  # noqa: E402


def _stage5_run(meta_overrides: dict):
    import cv2
    rng = np.random.default_rng(42)
    clean = np.full((100, 100, 3), 128, np.uint8)
    noise = rng.normal(0, 15, (100, 100, 3)).astype(np.int16)
    noisy = np.clip(clean.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    ctx = PipelineContext(noisy)
    ctx.image_metadata.update({"faces": [], **meta_overrides})
    pipeline = RestorationPipeline()
    pipeline._stage_5_spatially_gated_denoise(ctx, [], None)
    logs = [l for l in ctx.processing_log if "h_face=" in l]
    assert logs, "Stage 5 log line missing"
    m = re.search(r"h_bg=([\d.]+)", logs[-1])
    return float(m.group(1))


def test_damage_branch_respects_cap(monkeypatch):
    # A damage mask key whose bg coverage > 2% triggers the damage branch.
    import cv2
    meta = {"content_sigma": 12.0}
    # damage 'final' mask: >2% coverage in the bg zone
    # bg_mask is computed from faces==[] -> whole-frame 0.4 mid + rest bg; give
    # a mask covering the bottom rows only
    ctx = PipelineContext(np.full((100, 100, 3), 128, np.uint8))
    ctx.image_metadata.update({"faces": [], **meta})
    mask = np.zeros((100, 100), np.uint8)
    mask[:, 60:] = 255           # 40% of frame
    ctx.damage_masks["final"] = mask
    monkeypatch.setenv("PIXREVIVE_NLM_CAP", "14.0")
    pipeline = RestorationPipeline()
    pipeline._stage_5_spatially_gated_denoise(ctx, [], None)
    logs = [l for l in ctx.processing_log if "h_face=" in l]
    import re
    h_bg = float(re.search(r"h_bg=([\d.]+)", logs[-1]).group(1))
    assert h_bg <= 14.0, f"damage-branch h_bg {h_bg} exceeded cap 14"


def test_cap_zero_disables(monkeypatch):
    import cv2
    ctx = PipelineContext(np.full((100, 100, 3), 128, np.uint8))
    ctx.image_metadata.update({"faces": [], "content_sigma": 12.0})
    mask = np.zeros((100, 100), np.uint8)
    mask[:, 60:] = 255
    ctx.damage_masks["final"] = mask
    monkeypatch.setenv("PIXREVIVE_NLM_CAP", "0")
    pipeline = RestorationPipeline()
    pipeline._stage_5_spatially_gated_denoise(ctx, [], None)
    logs = [l for l in ctx.processing_log if "h_face=" in l]
    import re
    h_bg = float(re.search(r"h_bg=([\d.]+)", logs[-1]).group(1))
    assert h_bg > 14.0, f"cap=0 must not constrain, got h_bg={h_bg}"
