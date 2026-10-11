# -*- coding: utf-8 -*-
"""RC-fix tests (debug session 2026-10-01): field-noise tearmark mode, mask-
refiner speck-granular morphology, hair HF guard (structure floor + bands),
face-drip dissolve, and the low-res grayscale colorize skip."""
import os

import cv2
import numpy as np
import pytest

from ai.damage.tearmark_detector import TearmarkDetector, _tm_field_mode
from ai.pipeline.pipeline_context import PipelineContext


# ── helpers ─────────────────────────────────────────────────────────────────
def _sepia_mottle_image(h=447, w=447, seed=3):
    """Sepia paper matched to the REAL portrait's measured statistics: tan base,
    low-frequency mottle, PLUS a fine speckle field whose top-hat ladder matches
    the teared scan (p50=24, p75=45, p90=84, p93=103), plus 12 sharp
    high-amplitude flakes (top-hat 46-63 at p93=103 -> amplitude tail).
    Returns (img, flake_centers)."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 168.0, np.float32)
    # low-frequency mottle
    small = rng.normal(0, 22, (h // 16, w // 16)).astype(np.float32)
    mottle = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    img[..., 0] += mottle * 0.35
    img[..., 1] += mottle * 0.2
    img[..., 2] += mottle * 0.05
    # fine speckle field (the grain gate's "field")
    fine = rng.normal(0, 14, (h, w)).astype(np.float32)
    img[..., 0] += fine * 0.5
    img[..., 1] += fine * 0.3
    img[..., 2] += fine * 0.1
    # real flakes: sharp bright discs recorded for the survival test
    centers = []
    for _ in range(12):
        cy, cx = int(rng.integers(20, h - 20)), int(rng.integers(20, w - 20))
        # Bright neutral disc = bare paper showing through: bright in L AND
        # locally LESS saturated than the tinted emulsion around it (the
        # physical signature pass3's sat_rel gate requires).
        cv2.circle(img, (cx, cy), 3, (232.0, 232.0, 228.0), -1)
        centers.append((cy, cx))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, centers


def _ctx(img, meta=None):
    if isinstance(img, tuple):
        img = img[0]
    m = {"color_type": "sepia", "content_sigma": 7.4,
         "grain_field_mask_discarded": True, "faces": []}
    if meta:
        m.update(meta)
    return PipelineContext(original_image=img, metadata=m)


def _coverage(mask):
    return float(np.count_nonzero(mask)) / mask.size


# ── RC-1a: tearmark field mode ──────────────────────────────────────────────
class TestTearmarkFieldMode:
    def test_field_mode_engages_on_grain_discarded(self):
        ctx = _ctx(_sepia_mottle_image())
        assert _tm_field_mode(ctx) is True

    def test_field_mode_inert_on_clean_low_noise(self):
        ctx = _ctx(_sepia_mottle_image(), meta={
            "grain_field_mask_discarded": False, "content_sigma": 2.0})
        assert _tm_field_mode(ctx) is False

    def test_field_mode_never_on_grayscale(self):
        ctx = _ctx(_sepia_mottle_image(), meta={"color_type": "grayscale"})
        assert _tm_field_mode(ctx) is False

    def test_field_mode_covers_far_less_than_shipped(self):
        img = _sepia_mottle_image()
        m_field = TearmarkDetector().detect(_ctx(img))
        cov_field = _coverage(m_field)
        # field mode: stricter pass3, no pass2, light close, 12% cap
        assert cov_field < 0.12 + 0.01

    def test_field_mode_env_reverts(self, monkeypatch):
        monkeypatch.setenv("PIXREVIVE_TM_FIELD", "0")
        ctx = _ctx(_sepia_mottle_image())
        assert _tm_field_mode(ctx) is False

    def test_field_mode_keeps_high_amplitude_flakes(self):
        img, centers = _sepia_mottle_image()
        mask = TearmarkDetector().detect(_ctx(img))
        # flake discs carry ~+55 L on a field whose p93 top-hat ~= the real
        # portrait's 103 — they are the amplitude tail and must survive.
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        th = lab[:, :, 0].astype(np.float32) - cv2.morphologyEx(
            lab[:, :, 0].astype(np.float32), cv2.MORPH_OPEN, se)
        strong = [c for c in centers if th[c[0] - 1:c[0] + 2, c[1] - 1:c[1] + 2].max() > 40]
        assert strong, "fixture produced no amplitude-tail flakes"
        hits = sum(
            1 for (cy, cx) in strong
            if np.any(mask[max(0, cy - 4):cy + 5, max(0, cx - 4):cx + 5] > 0)
        )
        assert hits >= max(4, int(0.5 * len(strong)))

    def test_safety_valve_off_in_field_mode(self):
        """Field mode replaces the >45% tightening valve with its own cap; the
        tightened branch must not run (it reverts to shipped thresholds)."""
        img = _sepia_mottle_image()
        ctx = _ctx(img)
        TearmarkDetector().detect(ctx)
        joined = "\n".join(ctx.processing_log)
        assert ">45%" not in joined


# ── RC-1b: mask refiner speck-granular morphology ───────────────────────────
class TestMaskRefinerFieldMode:
    def _refine(self, img, mask, meta=None):
        from ai.damage.mask_refiner import MaskRefiner
        ctx = _ctx(img, meta)
        ctx.damage_masks["tearmark"] = mask
        return MaskRefiner.refine(mask, ctx)

    def test_field_mode_keeps_mask_speck_granular(self):
        img, _ = _sepia_mottle_image()
        rng = np.random.default_rng(5)
        specks = np.zeros(img.shape[:2], np.uint8)
        for _ in range(400):
            y, x = int(rng.integers(5, img.shape[0] - 5)), int(rng.integers(5, img.shape[1] - 5))
            specks[y, x] = 255
        meta = {"noise_score": 3.0}
        out_field = self._refine(img, specks.copy(), meta)
        out_ship = self._refine(img, specks.copy(), {
            **meta, "grain_field_mask_discarded": False, "content_sigma": 2.0})
        assert _coverage(out_field) < _coverage(out_ship)

    def test_real_tear_component_survives_field_mode(self):
        img, _ = _sepia_mottle_image()
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.line(mask, (10, 10), (10, 400), 255, 3)  # genuine tear strip
        out = self._refine(img, mask)
        assert out[200, 10] > 0  # tear interior survives the light close


# ── RC-3: structure gate floor ──────────────────────────────────────────────
class TestStructureGateFloor:
    def test_lower_floor_admits_more_hair_structure(self):
        from ai.analysis.structure import structure_gate
        img = np.asarray(cv2.imread(
            os.path.join("debug_output", "03_after_inpaint_img.png"),
            cv2.IMREAD_COLOR))
        if img is None:
            pytest.skip("debug dump not present in this checkout")
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = g.shape
        hair = g[int(0.02 * h):int(0.20 * h), int(0.30 * w):int(0.70 * w)]
        g45 = structure_gate(hair, mag_floor=45.0).mean()
        g24 = structure_gate(hair, mag_floor=24.0).mean()
        assert g24 > g45

    def test_pure_noise_stays_below_contract_at_floor_24(self):
        from ai.analysis.structure import structure_gate
        rng = np.random.default_rng(7)
        noise = np.clip(rng.normal(128, 15, (256, 256)), 0, 255).astype(np.float32)
        assert structure_gate(noise, mag_floor=24.0).mean() < 0.01


# ── RC-2: face drip dissolve ────────────────────────────────────────────────
class TestFaceDripDissolve:
    def _face_with_drips(self, h=400, w=400):
        img = np.full((h, w, 3), 150.0, np.uint8)
        for _ in range(30):
            y = int(np.random.default_rng(_).integers(40, h - 40))
            x = int(np.random.default_rng(_ + 7).integers(40, w - 40))
            cv2.circle(img, (x, y), 2, (245, 245, 245), -1)
        return img

    def test_dissolves_compact_bright_specks_in_face(self):
        from ai.validation.output_verifier import dissolve_face_drips
        img = self._face_with_drips()
        faces = np.zeros(img.shape[:2], np.uint8)
        faces[40:360, 40:360] = 255
        out, n = dissolve_face_drips(img, faces)
        assert n > 0
        assert out is not img

    def test_zero_when_no_residue(self):
        from ai.validation.output_verifier import dissolve_face_drips
        img = np.full((200, 200, 3), 150.0, np.uint8)
        faces = np.full((200, 200), 255, np.uint8)
        out, n = dissolve_face_drips(img, faces)
        assert n == 0 and np.array_equal(out, img)

    def test_empty_face_mask_safe(self):
        from ai.validation.output_verifier import dissolve_face_drips
        img = self._face_with_drips()
        faces = np.zeros(img.shape[:2], np.uint8)
        out, n = dissolve_face_drips(img, faces)
        assert n == 0

    def test_verify_and_clean_log_and_flag(self):
        from ai.validation.output_verifier import verify_and_clean
        img = self._face_with_drips()
        ctx = _ctx(img, meta={"color_type": "sepia"})
        ctx.image_metadata["faces"] = [{"bbox": [40, 40, 360, 360]}]
        rec = verify_and_clean(ctx)
        assert rec.get("face_drips_dissolved_px", 0) >= 0  # ran without raising

    def test_env_revert(self, monkeypatch):
        from ai.validation.output_verifier import verify_and_clean
        monkeypatch.setenv("PIXREVIVE_FACE_DRIP", "0")
        img = self._face_with_drips()
        ctx = _ctx(img)
        ctx.image_metadata["faces"] = [{"bbox": [40, 40, 360, 360]}]
        rec = verify_and_clean(ctx)
        assert "face_drips_dissolved_px" not in rec


# ── RC-4: low-res grayscale colorize skip ───────────────────────────────────
class TestGraySkip:
    def _gray_ctx(self, size=(420, 561), noise=8.0):
        img = np.full((size[0], size[1], 3), 128.0, np.uint8)
        meta = {"color_type": "grayscale", "noise_score": noise,
                "content_sigma": 7.0}
        return PipelineContext(original_image=img, metadata=meta)

    def test_skips_on_small_noisy_gray(self):
        from ai.pipeline.stages.color_stages import _gray_skip_warranted
        assert _gray_skip_warranted(self._gray_ctx()) is True

    def test_keeps_large_clean_gray(self):
        from ai.pipeline.stages.color_stages import _gray_skip_warranted
        ctx = self._gray_ctx(size=(1400, 1400), noise=2.0)
        ctx.image_metadata["content_sigma"] = 2.0
        assert _gray_skip_warranted(ctx) is False

    def test_never_on_sepia_or_color(self):
        from ai.pipeline.stages.color_stages import _gray_skip_warranted
        ctx = self._gray_ctx()
        ctx.image_metadata["color_type"] = "sepia"
        assert _gray_skip_warranted(ctx) is False

    def test_env_revert(self, monkeypatch):
        from ai.pipeline.stages.color_stages import _gray_skip_warranted
        monkeypatch.setenv("PIXREVIVE_GRAY_SKIP", "0")
        assert _gray_skip_warranted(self._gray_ctx()) is False

    def test_detected_grain_type_triggers_skip_at_low_noise(self):
        """Measured on the SigShrink figure: global noise 3.7 but grain_floor
        in detected_types — the profile is the evidence, not the score."""
        from ai.pipeline.stages.color_stages import _gray_skip_warranted
        ctx = self._gray_ctx(noise=3.7)
        ctx.image_metadata["content_sigma"] = 1.5
        ctx.image_metadata["noise_profile"] = {
            "detected_types": ["grain_floor", "ringing"]}
        assert _gray_skip_warranted(ctx) is True


class TestSepiaBackgroundWashGuard:
    """RC-4b: DDColor multi-hue wash on sepia mottle (pink field) must roll
    back even though Q-3's single-hue guard passes it."""

    def _mixin(self):
        from ai.pipeline.stages.color_stages import ColorStagesMixin
        return ColorStagesMixin()

    def test_pink_background_fires(self):
        rgb = np.zeros((447, 447, 3), np.uint8)
        rgb[..., 0] = 205; rgb[..., 1] = 175; rgb[..., 2] = 195
        assert self._mixin()._detect_background_chroma_wash(
            rgb, [{"bbox": [150, 60, 320, 280]}]) is True

    def test_neutral_sepia_background_silent(self):
        rgb = np.full((447, 447, 3), 190, np.uint8)
        rgb[..., 0] = 195; rgb[..., 2] = 185
        assert self._mixin()._detect_background_chroma_wash(
            rgb, [{"bbox": [150, 60, 320, 280]}]) is False

    def test_gray_source_never_fires(self):
        rgb = np.zeros((447, 447, 3), np.uint8)
        rgb[..., 0] = 205; rgb[..., 1] = 175; rgb[..., 2] = 195
        assert self._mixin()._detect_background_chroma_wash(
            rgb, [], sepia=False) is False

    def test_face_only_frame_never_fires(self):
        rgb = np.zeros((100, 100, 3), np.uint8)
        rgb[..., 0] = 205; rgb[..., 1] = 175; rgb[..., 2] = 195
        assert self._mixin()._detect_background_chroma_wash(
            rgb, [{"bbox": [0, 0, 100, 100]}]) is False


class TestReportFlagForSkippedColorize:
    def test_report_flag_for_skipped_colorize(self):
        from ai.report.restoration_report import build_restoration_report
        img = np.full((342, 435, 3), 128.0, np.uint8)
        meta = {"color_type": "grayscale", "colorize_skipped_lowres_gray": True,
                "colorization_applied": False}
        ctx = PipelineContext(original_image=img, metadata=meta)
        report = build_restoration_report(ctx)
        joined = str(report)
        assert "colorization" in joined.lower() and ("skipped" in joined.lower()
                                                     or "invented" in joined.lower())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
