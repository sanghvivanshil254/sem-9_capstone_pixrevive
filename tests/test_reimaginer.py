"""Tests for the re-imagine stage engine (Phase H1, design doc sections 1-6).

The evidence-preservation contract, tile coverage, identity rollback ladder
and NaN self-healing are tested with mocked generation — no diffusion
weights, runs unconditionally. Skip paths need nothing at all.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
)

from ai.restoration import reimaginer as reim  # noqa: E402
from ai.restoration.resource_profiler import MINIMAL, NO_GPU, AMPLE  # noqa: E402


def _canvas(size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    img = np.full((size, size, 3), 200, np.uint8)
    img[:, : size // 2] = (40, 90, 180)  # distinct left half
    mask = np.zeros((size, size), np.uint8)
    lo, hi = size // 4, size // 2
    mask[lo:hi, lo:hi] = 255  # ~15% destroyed region
    return img, mask


class TestEvidenceComposite(unittest.TestCase):
    """The core contract: out[E] == original[E], bytewise."""

    def test_evidence_restored_bytewise_mask_from_generation(self):
        img, mask = _canvas()
        gen = np.full_like(img, 7)
        out = reim._composite_evidence(img, gen, mask)
        keep = mask == 0
        self.assertTrue(np.array_equal(out[keep], img[keep]))
        self.assertTrue(np.array_equal(out[mask > 0], gen[mask > 0]))

    def test_composite_does_not_mutate_inputs(self):
        img, mask = _canvas()
        img0, gen0 = img.copy(), None
        gen = np.full_like(img, 7)
        reim._composite_evidence(img, gen, mask)
        self.assertTrue(np.array_equal(img, img0))
        del gen0

    def test_fully_destroyed_image_is_all_generated(self):
        img, _ = _canvas(64)
        mask = np.full((64, 64), 255, np.uint8)
        gen = np.full_like(img, 99)
        out = reim._composite_evidence(img, gen, mask)
        self.assertTrue(np.array_equal(out, gen))


class TestTiling(unittest.TestCase):
    """Tile grid covers the canvas exactly; alpha feathers correctly."""

    def test_small_canvas_single_tile(self):
        self.assertEqual(reim._tile_grid(256, 704, 128), [(0, 0)])

    def test_practical_1024_canvas_two_tiles_per_axis(self):
        # The re-imagine canvas is capped at 1024; with 704 tiles that is
        # exactly two coordinates per axis with a 384px feathered overlap.
        grid = reim._tile_grid(1024, 704, 128)
        self.assertEqual(sorted(grid), [(0, 0), (0, 320), (320, 0), (320, 320)])

    def test_awkward_size_last_tile_is_flush(self):
        grid = reim._tile_grid(1500, 704, 128)
        self.assertEqual(grid[-1][0], 1500 - 704)
        for x, y in grid:
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x + 704, 1500)

    def test_union_of_tiles_covers_canvas(self):
        w = h = 1024
        tile, ov = 704, 128
        cover = np.zeros((h, w), bool)
        xs = [c[0] for c in reim._tile_grid(w, tile, ov)]
        ys = [c[1] for c in reim._tile_grid(h, tile, ov)]
        for y in ys:
            for x in xs:
                cover[y:y + tile, x:x + tile] = True
        self.assertTrue(cover.all())

    def test_feather_alpha_shape_and_bounds(self):
        a = reim._feather_alpha(704, 704, 128)
        self.assertEqual(a.shape, (704, 704))
        self.assertAlmostEqual(float(a[352, 352]), 1.0, places=5)
        self.assertAlmostEqual(float(a[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(a[0, 352]), 0.0, places=5)  # edge row
        self.assertAlmostEqual(float(a[100, 100]), float(a[100, 703 - 100]),
                               places=5)  # symmetric ramps

    def test_black_share(self):
        black = np.zeros((8, 8, 3), np.uint8)
        noisy = np.full((8, 8, 3), 128, np.uint8)
        self.assertAlmostEqual(reim._black_share(black), 1.0)
        self.assertAlmostEqual(reim._black_share(noisy), 0.0)


class TestSkipPaths(unittest.TestCase):
    """Every unavailable path degrades to a record, never a raise."""

    def test_no_gpu_tier_skips_with_reason(self):
        img, mask = _canvas()
        rec = reim.reimagine(img, mask, task_id="t1", tier=NO_GPU)
        self.assertEqual(rec["status"], "skipped")
        self.assertIn("unavailable", rec["reason"])
        self.assertIsNone(rec["result"])
        self.assertEqual(rec["resource"]["tier"], "no_gpu")

    def test_missing_weights_skips(self):
        img, mask = _canvas()
        with mock.patch.object(reim, "_weights_present", return_value=False):
            rec = reim.reimagine(img, mask, task_id="t1", tier=MINIMAL)
        self.assertEqual(rec["status"], "skipped")
        self.assertIn("weights", rec["reason"])

    def test_resource_block_records_config(self):
        img, mask = _canvas()
        rec = reim.reimagine(img, mask, task_id="t1", tier=NO_GPU)
        cfg = rec["resource"]["config"]
        self.assertIn("offload", cfg)
        self.assertIn("tile_px", cfg)


class _FakeGate:
    """Scriptable stand-in for SFaceIdentityGate."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def weights_available(self):
        return True

    def check(self, original, result, **kw):
        self.calls += 1
        return self.script.pop(0)


class TestHappyPath(unittest.TestCase):
    """Mocked generation: contract, identity recording, determinism."""

    def _patch_env(self, gate):
        return (
            mock.patch.object(reim, "_weights_present", return_value=True),
            mock.patch.object(reim, "_sdxl_weights_present", return_value=False),
            mock.patch.object(reim, "SFaceIdentityGate", return_value=gate),
        )

    def test_done_result_honors_contract(self):
        img, mask = _canvas()
        good_gen = img.copy()
        good_gen[mask > 0] = (10, 20, 30)
        gate = _FakeGate([{"action": "pass", "similarity": 0.61, "reason": "ok"}])
        p1, p2, p3 = self._patch_env(gate)
        with p1, p2, p3, mock.patch.object(reim, "_generate", return_value=good_gen):
            rec = reim.reimagine(img, mask, task_id="case-x", tier=MINIMAL)
        self.assertEqual(rec["status"], "done")
        self.assertTrue(rec["evidence_preserved"])
        keep = mask == 0
        self.assertTrue(np.array_equal(rec["result"][keep], img[keep]))
        self.assertTrue(np.array_equal(rec["result"][mask > 0],
                                       good_gen[mask > 0]))
        self.assertEqual(rec["identity"]["action"], "pass")
        self.assertEqual(rec["generator"], "sd15")
        # mask is rows 64..127 of 256, cols 64..127 → 64*64 / 256² = 6.25%.
        self.assertAlmostEqual(rec["real_pixels_pct"], 100.0 - 6.25, delta=0.1)

    def test_seed_is_deterministic_per_task(self):
        img, mask = _canvas()
        gate = _FakeGate([{"action": "pass", "similarity": 0.7, "reason": "ok"}] * 3)
        seeds = []
        for task in ("same", "same", "other"):
            gen = img.copy()
            p1, p2, p3 = self._patch_env(gate)
            with p1, p2, p3, mock.patch.object(reim, "_generate",
                                               return_value=gen):
                rec = reim.reimagine(img, mask, task_id=task, tier=MINIMAL)
            seeds.append((task, rec["seed"]))
        self.assertEqual(seeds[0][1], seeds[1][1])
        self.assertNotEqual(seeds[0][1], seeds[2][1])

    def test_flag_verdict_still_ships_but_discloses(self):
        img, mask = _canvas()
        gen = img.copy()
        gate = _FakeGate([{"action": "flag", "similarity": 0.38, "reason": "ok"}])
        p1, p2, p3 = self._patch_env(gate)
        with p1, p2, p3, mock.patch.object(reim, "_generate", return_value=gen):
            rec = reim.reimagine(img, mask, task_id="t", tier=MINIMAL)
        self.assertEqual(rec["status"], "done")
        self.assertEqual(rec["identity"]["action"], "flag")

    def test_uncertified_identity_is_disclosed_not_passed(self):
        img, mask = _canvas()
        gen = img.copy()
        gate = _FakeGate([{"action": "flag", "similarity": None,
                           "reason": "sface_unavailable"}])
        p1, p2, p3 = self._patch_env(gate)
        with p1, p2, p3, mock.patch.object(reim, "_generate", return_value=gen):
            rec = reim.reimagine(img, mask, task_id="t", tier=MINIMAL)
        self.assertEqual(rec["status"], "done")
        self.assertIsNone(rec["identity"]["similarity"])


class TestRollbackLadder(unittest.TestCase):
    """Rollback re-rolls at seed+1, then reverts — bounded, never loops."""

    def _run(self, gate, tier=MINIMAL):
        img, mask = _canvas()
        gen = img.copy()
        p1, p2, p3 = self._patch_env_shared(gate)
        with p1, p2, p3, mock.patch.object(reim, "_generate", return_value=gen):
            return reim.reimagine(img, mask, task_id="t", tier=tier)

    def _patch_env_shared(self, gate):
        return (
            mock.patch.object(reim, "_weights_present", return_value=True),
            mock.patch.object(reim, "_sdxl_weights_present", return_value=False),
            mock.patch.object(reim, "SFaceIdentityGate", return_value=gate),
        )

    def test_rollback_recovers_on_second_roll(self):
        gate = _FakeGate([
            {"action": "rollback", "similarity": 0.2, "reason": "ok"},
            {"action": "pass", "similarity": 0.55, "reason": "ok"},
        ])
        rec = self._run(gate)
        self.assertEqual(rec["status"], "done")
        self.assertEqual(gate.calls, 2)
        self.assertEqual(rec["identity"]["similarity"], 0.55)

    def test_permanent_rollback_degrades_after_max_rerolls(self):
        gate = _FakeGate([{"action": "rollback", "similarity": 0.1,
                           "reason": "ok"}] * (reim.MAX_REROLLS + 1))
        rec = self._run(gate)
        self.assertEqual(rec["status"], "skipped_degraded")
        self.assertIn("identity", rec["reason"])
        self.assertEqual(gate.calls, reim.MAX_REROLLS + 1)
        self.assertIsNone(rec["result"])
        # The reverted record must disclose the identity verdict it failed on.
        self.assertEqual(rec["identity"]["action"], "rollback")

    def test_rerolls_param_bounds_the_ladder(self):
        """rerolls=0: single attempt, no re-roll even on rollback (H3)."""
        gate = _FakeGate([{"action": "rollback", "similarity": 0.1,
                           "reason": "ok"}] * 3)
        img, mask = _canvas()
        gen = img.copy()
        p1, p2, p3 = self._patch_env_shared(gate)
        with p1, p2, p3, mock.patch.object(reim, "_generate", return_value=gen):
            rec = reim.reimagine(img, mask, task_id="t", tier=MINIMAL, rerolls=0)
        self.assertEqual(rec["status"], "skipped_degraded")
        self.assertEqual(gate.calls, 1)

    def test_rerolls_param_negative_clamps_to_zero(self):
        gate = _FakeGate([{"action": "rollback", "similarity": 0.1,
                           "reason": "ok"}] * 3)
        img, mask = _canvas()
        gen = img.copy()
        p1, p2, p3 = self._patch_env_shared(gate)
        with p1, p2, p3, mock.patch.object(reim, "_generate", return_value=gen):
            rec = reim.reimagine(img, mask, task_id="t", tier=MINIMAL, rerolls=-5)
        self.assertEqual(gate.calls, 1)  # clamped, never an infinite/absent loop


class TestNaNHealing(unittest.TestCase):
    """A poisoned (black) frame is retried once, then degraded — never shipped."""

    def _patch_env(self, gate=None):
        gate = gate or _FakeGate([{"action": "pass", "similarity": 0.6,
                                   "reason": "ok"}] * 5)
        return (
            mock.patch.object(reim, "_weights_present", return_value=True),
            mock.patch.object(reim, "_sdxl_weights_present", return_value=False),
            mock.patch.object(reim, "SFaceIdentityGate", return_value=gate),
        )

    def test_black_frame_retried_at_seed_plus_one(self):
        img, mask = _canvas()
        good = img.copy()
        black = np.zeros_like(img)
        p = self._patch_env()
        with p[0], p[1], p[2], \
             mock.patch.object(reim, "_generate",
                               side_effect=[black, good]) as gen_mock:
            rec = reim.reimagine(img, mask, task_id="t", tier=MINIMAL)
        self.assertEqual(rec["status"], "done")
        self.assertEqual(gen_mock.call_count, 2)

    def test_permanent_black_never_ships_black(self):
        img, mask = _canvas()
        black = np.zeros_like(img)
        p = self._patch_env()
        with p[0], p[1], p[2], \
             mock.patch.object(reim, "_generate", return_value=black) as gm:
            rec = reim.reimagine(img, mask, task_id="t", tier=MINIMAL)
        self.assertNotEqual(rec["status"], "done")
        self.assertIsNone(rec["result"])
        self.assertEqual(gm.call_count, reim.MAX_REROLLS + 1)

    def test_minimal_pins_sd15_even_when_sdxl_weights_exist(self):
        # Measured tier policy: MINIMAL runs SD1.5 up front (fp16 SDXL is
        # NaN-broken on the reference torch/GPU class; fp32 SDXL exceeds RAM).
        img, mask = _canvas()
        good = img.copy()
        gate = _FakeGate([{"action": "pass", "similarity": 0.6, "reason": "ok"}])
        p = self._patch_env(gate)
        with p[0], p[1], p[2], \
             mock.patch.object(reim, "_generate", return_value=good) as gm, \
             mock.patch.object(reim, "_sdxl_weights_present",
                               return_value=True):
            rec = reim.reimagine(img, mask, task_id="t", tier=MINIMAL)
        self.assertEqual(rec["status"], "done")
        self.assertEqual(rec["generator"], "sd15")
        self.assertEqual(rec["resource"]["config"]["generator"], "sd15")
        self.assertEqual(gm.call_count, 1)

    def test_sdxl_tier_black_frame_falls_back_to_sd15(self):
        # On an SDXL tier (AMPLE), a NaN-poisoned SDXL frame degrades to the
        # SD1.5 pipeline for that image (design doc section 5 fallback).
        img, mask = _canvas()
        good = img.copy()
        gate = _FakeGate([{"action": "pass", "similarity": 0.6, "reason": "ok"}])
        p = self._patch_env(gate)
        with p[0], p[1], p[2], \
             mock.patch.object(reim, "_generate",
                               side_effect=[np.zeros_like(img), good]) as gm, \
             mock.patch.object(reim, "_sdxl_weights_present",
                               return_value=True):
            rec = reim.reimagine(img, mask, task_id="t", tier=AMPLE)
        self.assertEqual(rec["status"], "done")
        self.assertIn("sd15_fallback", rec["generator"])
        # First call was SDXL, second the SD1.5 retry.
        self.assertEqual(gm.call_count, 2)


class TestUnload(unittest.TestCase):
    def test_unload_clears_cache(self):
        reim._STATE.pipes["sd15"] = object()
        reim.unload_reimagine_models()
        self.assertEqual(reim._STATE.pipes, {})


if __name__ == "__main__":
    unittest.main()
