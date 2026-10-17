# -*- coding: utf-8 -*-
"""Dust-colony / bright-fill fix tests (debug session 2026-10-02, white blob
on head): Stage-2 dense dust-colony re-admission after the grain gate, and
the Stage-11 bright-infill survivor dissolve inside the repair zone."""
import cv2
import numpy as np
import pytest

from ai.damage.enhanced_detector import EnhancedDamageDetector
from ai.pipeline.pipeline_context import PipelineContext
from ai.validation.output_verifier import dissolve_bright_fill_fields


# ── helpers ─────────────────────────────────────────────────────────────────
def _sepia_noise_frame(h=447, w=447, seed=3):
    """Sepia paper with a low-frequency mottle + fine speckle field, matching
    the teared portrait's measured character (tan base, noisy)."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 168.0, np.float32)
    small = rng.normal(0, 22, (h // 16, w // 16)).astype(np.float32)
    mottle = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    img[..., 0] += mottle * 0.35
    img[..., 1] += mottle * 0.2
    img[..., 2] += mottle * 0.05
    fine = rng.normal(0, 14, (h, w)).astype(np.float32)
    img[..., 0] += fine * 0.5
    img[..., 1] += fine * 0.3
    img[..., 2] += fine * 0.1
    return np.clip(img, 0, 255).astype(np.uint8)


def _add_dust_colony(img, cy, cx, ry=22, rx=30, amp=60, seed=7):
    """Dense colony of mid-size bright dust blobs inside a DARKER substrate
    patch — the class signature from the real input (head colony inside dark
    hair: ring ~110, colony ~178, popout ~50-68). The dark patch extends one
    ring-width past the dust so the detector's surround IS the substrate,
    exactly as hair surrounds the real colony."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    # dark substrate (hair-like), extending past the dust field
    ry_s, rx_s = ry + 20, rx + 20
    patch = out[cy - ry_s:cy + ry_s, cx - rx_s:cx + rx_s].astype(np.float32)
    patch *= 0.55
    out[cy - ry_s:cy + ry_s, cx - rx_s:cx + rx_s] = np.clip(
        patch, 0, 255).astype(np.uint8)
    # dense bright dust inside it — DISCRETE grains (like the real colony,
    # 51 seed islands, probe_island_gate.py), so disc coverage stays below
    # the fusion point and the grains do not merge into one smooth blob.
    # Granularity contract (P3/C.1 amendment): the detector requires >=5 seed
    # islands per component, calibrated on the REAL head colony (51/7/23
    # islands). Grains must therefore stay SMALL enough to remain discrete:
    # r=3-5 discs at ~25% coverage merged into only 3 islands on the noisy
    # fixture, tripping the calibrated gate with synthetic — not behavioural
    # — data. r=2-4 discs keep 8+ islands (fix, 2026-10-17).
    sub = out[cy - ry:cy + ry, cx - rx:cx + rx].astype(np.float32)
    # Two-scale disc layout (meets BOTH detector gates, measured 2026-10-17):
    # r=2 discs on a 6px grid + r=1 dots at cell centres. Seed coverage
    # 0.40 > 0.35 keeps the density-fusion field welded across the region
    # while 1-2px separations keep every grain a discrete seed island
    # (synthetic sweep: cov 0.405, dens>0.35 on 73% of the region, 648
    # islands). Single-scale layouts fail one gate or the other: spacing 5
    # r=2 self-extinguishes nothing but fuses (1 island); spacing 6 r=2
    # keeps islands but dens tops out 0.39 < 0.35 needed.
    for gy in range(6, 2 * ry - 2, 6):
        for gx in range(6, 2 * rx - 2, 6):
            cv2.circle(sub, (gx, gy), 2,
                       (min(255, 150 + amp),) * 2 + (min(255, 140 + amp),), -1)
            cv2.circle(sub, (min(2 * rx - 3, gx + 3), min(2 * ry - 3, gy + 3)), 1,
                       (min(255, 150 + amp),) * 2 + (min(255, 140 + amp),), -1)
    out[cy - ry:cy + ry, cx - rx:cx + rx] = np.clip(sub, 0, 255).astype(np.uint8)
    return out


def _ctx(img, meta=None):
    m = {"color_type": "sepia", "content_sigma": 7.4, "noise_score": 9.8,
         "damage_percentage": 3.0, "grain_field_mask_discarded": True,
         "faces": []}
    if meta:
        m.update(meta)
    return PipelineContext(original_image=img, metadata=m)


def _det():
    return EnhancedDamageDetector(use_defect_segmenter=False)


# ── Fix 1: Stage-2 dust colonies ────────────────────────────────────────────
class TestDetectDustColonies:
    def test_colony_on_dark_substrate_is_masked(self):
        img = _add_dust_colony(_sepia_noise_frame(), 60, 200)
        mask = _det().detect_dust_colonies(img, _ctx(img))
        assert float(np.count_nonzero(mask)) / mask.size > 0.005
        y0, y1, x0, x1 = 38, 82, 170, 230
        roi = mask[y0:y1, x0:x1] > 0
        assert roi.mean() > 0.3  # colony region substantially covered

    def test_clean_frame_is_inert(self):
        img = _sepia_noise_frame(seed=11)
        mask = _det().detect_dust_colonies(img, _ctx(img))
        assert float(np.count_nonzero(mask)) / mask.size < 0.02

    def test_bright_on_bright_is_not_flagged(self):
        # uniform bright paper with a slightly brighter collar band: the
        # popout test must reject it (ring is equally bright).
        img = _sepia_noise_frame(seed=5)
        img[250:330, 100:250] = (215, 213, 205)
        img[255:325, 110:240] = (222, 220, 212)
        mask = _det().detect_dust_colonies(img, _ctx(img))
        cov = float(np.count_nonzero(mask[250:330, 100:250])) / (80 * 150)
        assert cov < 0.2

    def test_revert_flag(self, monkeypatch):
        monkeypatch.setenv("PIXREVIVE_DUST_COLONIES", "0")
        img = _add_dust_colony(_sepia_noise_frame(), 60, 200)
        mask = _det().detect_dust_colonies(img, _ctx(img))
        assert np.count_nonzero(mask) == 0

    def test_real_head_colony_regression(self):
        """Real-image regression (P3/C.1 amendment): the >=5-island gate is
        calibrated on the REAL teared-sepia head colony (51/7/23 islands,
        probe_island_gate.py). This test pins the REAL image so any future
        fixture retune is judged against real data, not synthetic.

        Measured on case9_teared_sepia.jpg (447px): colony ROI (head blob,
        x185-263 y11-78 @447 ref) coverage 0.536, frame frac 0.0751.
        """
        import os
        path = os.path.join(os.path.dirname(__file__),
                            "restoration_dataset", "damaged", "case9_teared_sepia.jpg")
        if not os.path.exists(path):
            import pytest
            pytest.skip("case9 real fixture not present")
        img_bgr = cv2.imread(path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = _det().detect_dust_colonies(img, _ctx(img))
        h, w = mask.shape
        # head-blob ROI scaled from the 447px reference, slightly padded
        ky0, ky1 = int(11 / 447 * h * 1.2), int(78 / 447 * h * 1.4)
        kx0, kx1 = int(185 / 447 * w * 0.9), int(263 / 447 * w * 1.3)
        roi_cover = float(mask[ky0:ky1, kx0:kx1].mean() / 255)
        assert roi_cover > 0.30, \
            f"real head colony not masked (ROI cover {roi_cover:.3f})"
        frame_frac = float(np.count_nonzero(mask)) / mask.size
        assert frame_frac < 0.12, \
            f"detector over-fired on real frame ({frame_frac:.4f})"

    def test_synthetic_colony_island_count_contract(self):
        """Document the granularity contract the synthetic fixture must meet:
        the detector's >=5-seed-island gate is calibrated on real colonies
        (51/7/23 islands). The retuned synthetic grains (r=2-4) must produce
        >= 5 islands in the colony component so the test fails LOUDLY if a
        future fixture edit silently re-merges the grains into 3 islands
        (that failure mode is what hid the P3 fixture/gate mismatch).
        """
        img = _add_dust_colony(_sepia_noise_frame(), 60, 200)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = gray.shape
        side = min(h, w)
        k_exc = max(19, int(side * 0.056)) | 1
        bg = cv2.GaussianBlur(gray, (k_exc, k_exc), 0)
        seed = ((gray - bg) > 20.0).astype(np.uint8)
        cy, cx, ry, rx = 60, 200, 22, 30
        islands = int(np.unique(cv2.connectedComponents(seed, connectivity=8)[1][
            cy - ry:cy + ry, cx - rx:cx + rx]).max())
        # colony disc region alone must carry at least 5 discrete islands
        n_lbl = cv2.connectedComponents(
            seed[(cy - ry):(cy + ry), (cx - rx):(cx + rx)], connectivity=8)[0] - 1
        assert n_lbl >= 5, \
            f"colony fixture merged into {n_lbl} islands; detector needs >=5"


# ── Fix 2: Stage-11 bright-fill survivor dissolve ──────────────────────────
class TestDissolveBrightFillFields:
    def _final_canvas(self, seed=4):
        """Final-like canvas: dark hair field + flat bright fill patch inside
        the damage zone + textured bright patch inside the damage zone."""
        rng = np.random.default_rng(seed)
        img = np.full((1341, 1341, 3), 120.0, np.float32)
        img += rng.normal(0, 6, img.shape).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
        # flat bright fill (the artifact) inside the repair zone
        cv2.rectangle(img, (555, 60), (700, 190), (215, 210, 205), -1)
        cv2.GaussianBlur(img, (0, 0), 2.0, dst=img)
        # textured bright patch (real content) inside the repair zone
        tex = img[300:420, 555:700]
        tex[...] = (
            tex.astype(np.float32)
            + rng.normal(0, 45, tex.shape).astype(np.float32)
        ).clip(0, 255).astype(np.uint8)
        cv2.GaussianBlur(img, (0, 0), 0.8, dst=img)
        damage = np.zeros((1341, 1341), np.uint8)
        damage[40:500, 540:720] = 255  # repair zone covers both patches
        return img, damage

    def test_flat_fill_dissolved_textured_kept(self):
        img, damage = self._final_canvas()
        faces = np.zeros((1341, 1341), np.uint8)
        cleaned, n_px, n_comp = dissolve_bright_fill_fields(img, faces, damage)
        assert n_px > 0 and n_comp >= 1
        # the flat fill patch changed substantially
        diff = np.abs(cleaned[60:190, 555:700].astype(int)
                      - img[60:190, 555:700].astype(int)).mean()
        assert diff > 3.0
        # the textured bright patch was NOT touched
        diff_tex = np.abs(cleaned[300:420, 555:700].astype(int)
                          - img[300:420, 555:700].astype(int)).mean()
        assert diff_tex == 0.0

    def test_no_damage_mask_is_noop(self):
        img, _ = self._final_canvas()
        faces = np.zeros((1341, 1341), np.uint8)
        cleaned, n_px, n_comp = dissolve_bright_fill_fields(img, faces, None)
        assert n_px == 0 and n_comp == 0
        assert np.array_equal(cleaned, img)

    def test_faces_protected(self):
        img, damage = self._final_canvas()
        faces = np.zeros((1341, 1341), np.uint8)
        faces[40:500, 540:720] = 255  # cover the fill patch
        cleaned, n_px, n_comp = dissolve_bright_fill_fields(img, faces, damage)
        assert n_px == 0
        assert np.array_equal(cleaned, img)

    def test_revert_flag(self, monkeypatch):
        monkeypatch.setenv("PIXREVIVE_BRIGHT_FILL", "0")
        img, damage = self._final_canvas()
        faces = np.zeros((1341, 1341), np.uint8)
        cleaned, n_px, n_comp = dissolve_bright_fill_fields(img, faces, damage)
        assert n_px == 0
        assert np.array_equal(cleaned, img)

    def test_real_output_blob_is_dissolved(self):
        """Regression anchor: the actual teared-portrait output had a flat
        bright fill core (masked-subset p99=230) that this pass must catch."""
        # synthetic stand-in with the measured geometry: 86x101 flat patch at
        # mean 222 over a ~135 surround, tex ratio 0.61 (probe_fix2_v2).
        rng = np.random.default_rng(9)
        img = np.full((1341, 1341, 3), 135.0, np.float32)
        img += rng.normal(0, 5, img.shape).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.rectangle(img, (605, 80), (690, 180), (222, 220, 216), -1)
        cv2.GaussianBlur(img, (0, 0), 2.0, dst=img)
        damage = np.zeros((1341, 1341), np.uint8)
        damage[40:300, 540:760] = 255
        faces = np.zeros((1341, 1341), np.uint8)
        cleaned, n_px, n_comp = dissolve_bright_fill_fields(img, faces, damage)
        assert n_px > 500
        # the fill patch itself is where the pixels changed (frame-mean deltas
        # are meaningless on a 1.7MP canvas)
        diff = np.abs(cleaned[80:180, 605:690].astype(int)
                      - img[80:180, 605:690].astype(int)).mean()
        assert diff > 10.0
