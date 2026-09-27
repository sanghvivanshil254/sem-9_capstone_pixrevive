"""Tier-selection matrix for resource_profiler (design doc section 4.2).

Pure-Python + psutil/torch stubs — no model weights, runs unconditionally in
CI. detect_* are monkeypatched per-case; select_resource_tier* is the unit
under test. Boundaries are tested exactly AT each threshold (>= semantics).
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
)

from ai.restoration import resource_profiler as rp  # noqa: E402


def select(vram, ram, override=None):
    return rp.select_resource_tier(vram, ram, override)


class TestAutodetect(unittest.TestCase):
    """The ladder: ample → standard → minimal → skip, plus every boundary."""

    def test_no_gpu_skips(self):
        self.assertIs(select(None, 64.0).name, "no_gpu")

    def test_ample_vram(self):
        self.assertIs(select(24.0, 32.0).name, "ample")

    def test_standard_vram(self):
        self.assertIs(select(11.9, 16.0).name, "standard")

    def test_minimal_vram(self):
        self.assertIs(select(7.9, 16.0).name, "minimal")

    # -- exact boundaries (>= semantics) --

    def test_boundary_at_12gb_is_ample(self):
        self.assertIs(select(12.0, 32.0).name, "ample")

    def test_boundary_just_under_12gb(self):
        self.assertIs(select(11.99, 32.0).name, "standard")

    def test_boundary_at_8gb_is_standard(self):
        self.assertIs(select(8.0, 32.0).name, "standard")

    def test_boundary_just_under_8gb(self):
        self.assertIs(select(7.99, 32.0).name, "minimal")

    def test_boundary_at_3gb_is_minimal(self):
        self.assertIs(select(3.0, 32.0).name, "minimal")

    def test_boundary_just_under_3gb_skips(self):
        self.assertIs(select(2.99, 32.0).name, "no_gpu")

    # -- RAM minimums and the step-down rule --

    def test_ample_vram_low_ram_steps_to_minimal(self):
        # RAM 7.9 GB fails ample's 8 GB and standard's 10 GB floors, but
        # MINIMAL's measured 3.2 GB floor accepts it → step down to minimal.
        self.assertIs(select(24.0, 7.9).name, "minimal")

    def test_ample_vram_but_no_ram_skips(self):
        # Below even MINIMAL's measured 2.2 GB RAM floor → skip.
        self.assertIs(select(24.0, 2.0).name, "no_gpu")

    def test_standard_vram_low_ram_steps_to_minimal(self):
        # 11 GB VRAM is standard-eligible, standard needs 10 GB RAM: < 10 → minimal.
        self.assertIs(select(11.0, 9.5).name, "minimal")

    def test_minimal_vram_low_ram_skips(self):
        # Nothing below minimal to step to.
        self.assertIs(select(4.0, 2.0).name, "no_gpu")

    def test_huge_vram_but_no_ram_skips(self):
        self.assertIs(select(24.0, 0.5).name, "no_gpu")

    def test_minimal_ram_boundary_is_measured_floor(self):
        # Exactly at the 2.2 GB measured floor → runnable (>= semantics).
        self.assertIs(select(4.0, 2.2).name, "minimal")
        self.assertIs(select(4.0, 2.19).name, "no_gpu")

    def test_boundary_ram_at_ample_minimum(self):
        # 8.0 GB RAM == ample's minimum → ample runs (>=).
        self.assertIs(select(24.0, 8.0).name, "ample")

    def test_tiny_vram_fine_ram_skips(self):
        # The dev-box floor: 4 GB card near-full → not runnable.
        self.assertIs(select(1.5, 16.0).name, "no_gpu")


class TestTierSettings(unittest.TestCase):
    """Each tier carries the generation config the reimaginer consumes."""

    def test_minimal_is_the_4gb_config(self):
        t = rp.MINIMAL
        self.assertEqual((t.offload, t.attention_slicing, t.tiled_vae, t.tile_px, t.use_controlnet),
                         ("sequential", True, True, 704, False))
        # RAM floor 2.2 GB is measured (probe peak 2.1 GB, streaming), not intuited.
        self.assertEqual((t.min_vram_gb, t.min_ram_gb), (3.0, 2.2))
        # Measured on torch 2.13 + GTX 1650: fp16 diffusion NaNs; SD1.5 fp32
        # sequential-offload is the validated MINIMAL configuration.
        self.assertEqual((t.dtype, t.use_sdxl), ("fp32", False))

    def test_standard_is_model_offload(self):
        t = rp.STANDARD
        self.assertEqual((t.offload, t.attention_slicing, t.tile_px, t.use_controlnet),
                         ("model", False, 1024, True))
        self.assertEqual((t.min_vram_gb, t.min_ram_gb), (8.0, 10.0))

    def test_ample_is_bare_metal(self):
        t = rp.AMPLE
        self.assertEqual((t.offload, t.attention_slicing, t.tiled_vae, t.use_controlnet),
                         ("none", False, False, True))
        self.assertEqual((t.min_vram_gb, t.min_ram_gb), (12.0, 8.0))

    def test_no_gpu_is_not_runnable(self):
        self.assertFalse(rp.NO_GPU.runnable)
        self.assertTrue(all(t.runnable for t in rp._LADDER))

    def test_ladder_is_ordered_high_to_low(self):
        vrams = [t.min_vram_gb for t in rp._LADDER]
        self.assertEqual(vrams, sorted(vrams, reverse=True))


class TestOverride(unittest.TestCase):
    """REIMAGINE_FORCE_TIER pins the ceiling but never waives minimums."""

    def setUp(self):
        os.environ.pop("REIMAGINE_FORCE_TIER", None)

    def tearDown(self):
        os.environ.pop("REIMAGINE_FORCE_TIER", None)

    def test_override_off_skips_even_with_gpus(self):
        self.assertIs(select(24.0, 64.0, override="off").name, "no_gpu")

    def test_override_to_standard_from_ample(self):
        self.assertIs(select(24.0, 64.0, override="standard").name, "standard")

    def test_override_to_ample_needs_ample_vram(self):
        # Forced ample on an 8 GB box: ample's 12 GB floor fails → step down.
        self.assertIs(select(8.0, 32.0, override="ample").name, "standard")

    def test_override_respects_ram_step_down(self):
        # Forced standard with 9 GB RAM < standard's 10 GB floor → minimal
        # (which accepts 9 GB). The override pins the ceiling, not the floors.
        self.assertIs(select(16.0, 9.0, override="standard").name, "minimal")

    def test_override_env_var_is_read(self):
        with mock.patch.dict(os.environ, {"REIMAGINE_FORCE_TIER": "MINIMAL"}):
            # Case-insensitive; 4 GB box → minimal regardless of ample-class VRAM.
            self.assertIs(select(24.0, 64.0).name, "minimal")

    def test_override_env_var_off(self):
        with mock.patch.dict(os.environ, {"REIMAGINE_FORCE_TIER": "off"}):
            self.assertIs(select(24.0, 64.0).name, "no_gpu")

    def test_invalid_override_falls_back_to_autodetect(self):
        with mock.patch.dict(os.environ, {"REIMAGINE_FORCE_TIER": "turbo"}):
            self.assertIs(select(24.0, 64.0).name, "ample")

    def test_override_cannot_lift_a_failing_floor(self):
        # "off"-like intent: forced minimal but 2.9 GB free → still skipped.
        self.assertIs(select(2.9, 32.0, override="minimal").name, "no_gpu")


class TestDetectAndProfile(unittest.TestCase):
    """detect_* fallbacks and the profile() aggregation."""

    def test_detect_gpu_returns_none_on_exception(self):
        with mock.patch.object(rp.torch, "cuda") as fake:
            fake.is_available.return_value = True
            fake.mem_get_info.side_effect = RuntimeError("driver died")
            self.assertIsNone(rp.detect_gpu_vram_free_gb())

    def test_detect_gpu_none_when_unavailable(self):
        with mock.patch.object(rp.torch, "cuda") as fake:
            fake.is_available.return_value = False
            self.assertIsNone(rp.detect_gpu_vram_free_gb())

    def test_detect_ram_falls_back_to_zero(self):
        with mock.patch.object(rp.psutil, "virtual_memory", side_effect=OSError):
            self.assertEqual(rp.detect_system_ram_free_gb(), 0.0)

    def test_profile_returns_measurements(self):
        with mock.patch.object(rp, "detect_gpu_vram_free_gb", return_value=(3.6, 4.0)), \
             mock.patch.object(rp, "detect_system_ram_free_gb", return_value=11.2), \
             mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REIMAGINE_FORCE_TIER", None)
            tier, reason, m = rp.profile()
        self.assertEqual(tier.name, "minimal")
        self.assertEqual(reason, "autodetect")
        self.assertEqual(m, {"vram_free_gb": 3.6, "vram_total_gb": 4.0,
                             "ram_free_gb": 11.2, "override": None})

    def test_profile_reports_insufficient_reason(self):
        with mock.patch.object(rp, "detect_gpu_vram_free_gb", return_value=(1.0, 4.0)), \
             mock.patch.object(rp, "detect_system_ram_free_gb", return_value=16.0):
            os.environ.pop("REIMAGINE_FORCE_TIER", None)
            tier, reason, _ = rp.profile()
        self.assertEqual(tier.name, "no_gpu")
        self.assertTrue(reason.startswith("insufficient_resources"))

    def test_profile_reason_records_ram_step_down(self):
        # 11 GB VRAM is standard-eligible; 9 GB RAM fails standard's 10 GB
        # floor, minimal accepts → tier=minimal, reason names the step-down.
        with mock.patch.object(rp, "detect_gpu_vram_free_gb", return_value=(11.0, 12.0)), \
             mock.patch.object(rp, "detect_system_ram_free_gb", return_value=9.0):
            os.environ.pop("REIMAGINE_FORCE_TIER", None)
            tier, reason, _ = rp.profile()
        self.assertEqual(tier.name, "minimal")
        self.assertTrue(reason.startswith("ram_stepped_down"))
        self.assertIn("standard", reason)


if __name__ == "__main__":
    unittest.main()
