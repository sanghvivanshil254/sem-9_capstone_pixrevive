"""Phase 3 structural tests: stage registry integrity (A-2), the Stage 3
decomposition (U-3: _seam_smooth / _protect_facial_anatomy /
_face_wipeout_circuit_breaker), and device management (U-5: device_preference
threading with lazy reload).

Runs against a CPU-forced pipeline (tests/conftest.py) to keep the 4 GB GPU
out of the equation.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import torch

from ai.pipeline.pipeline_context import PipelineContext
from ai.pipeline.restoration_pipeline import RestorationPipeline
from ai.pipeline.stage_registry import (
    EDGE_EVENTS,
    STAGE_REGISTRY,
    frontend_stages,
    get_stage,
    registry_to_json,
    stage_ids,
)


class TestStageRegistry(unittest.TestCase):
    """A-2: the registry is the single source of truth and must stay sane."""

    def test_ids_are_unique(self):
        ids = [s.id for s in STAGE_REGISTRY] + [s.id for s in EDGE_EVENTS]
        self.assertEqual(len(ids), len(set(ids)), f"duplicate stage ids: {ids}")

    def test_progress_percentages_are_monotonic(self):
        # Only the frontend-visible stages need sorted percentages: the
        # internal damage-cluster sub-step fires at pct 0 inside stage 4, and
        # internal events are invisible to clients anyway.
        pcts = [s.progress_pct for s in frontend_stages()]
        self.assertEqual(
            pcts,
            sorted(pcts),
            f"frontend stage list must be ordered by emission percentage, got {pcts}",
        )
        # And every registry entry must still carry a sane pct.
        for spec in STAGE_REGISTRY:
            self.assertTrue(0 <= spec.progress_pct <= 100, f"{spec.id} pct out of range")

    def test_every_stage_method_exists_on_pipeline(self):
        pipeline = RestorationPipeline(device="cpu", load_models=False)
        for spec in STAGE_REGISTRY:
            if spec.method is None:
                continue
            self.assertTrue(
                hasattr(pipeline, spec.method),
                f"registry stage {spec.id!r} points at missing method {spec.method!r}",
            )

    def test_frontend_export_excludes_internal_stages(self):
        external_ids = {s.id for s in frontend_stages()}
        internal_ids = {s.id for s in STAGE_REGISTRY} - external_ids
        # refresh_faces: re-detection sub-step; semantic_validate: Phase-V0
        # report-only battery (no user-facing progress dot by design).
        self.assertEqual(internal_ids, {"refresh_faces", "semantic_validate"})
        # Exported ids + edge events == everything a live stream may emit.
        self.assertEqual(set(stage_ids()), external_ids | {e.id for e in EDGE_EVENTS})

    def test_registry_to_json_roundtrip(self):
        payload = registry_to_json()
        self.assertIn("stages", payload)
        self.assertIn("edge_events", payload)
        by_id = {s["id"]: s for s in payload["stages"]}
        self.assertIn("damage_detect", by_id)
        self.assertEqual(by_id["damage_detect"]["progress_pct"], 20)
        self.assertEqual(by_id["super_res"]["route_gate"], "enable_sr")

    def test_frontend_stages_json_is_in_sync(self):
        # N7 (re-audit 2026-09-30): the committed export went stale (missing
        # the Phase-H1 `reimagine` stage) and ProgressBar fell back to
        # "Processing..." for an emitted stage. Pin the generated file to the
        # registry; regen with scripts/export_stage_list.py.
        import json

        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "frontend", "src", "generated", "stages.json",
        )
        with open(out_path, encoding="utf-8") as fh:
            on_disk = json.load(fh)
        self.assertEqual(on_disk, registry_to_json())

    def test_get_stage_resolves_stages_and_edges(self):
        self.assertEqual(get_stage("colorize").id, "colorize")
        self.assertEqual(get_stage("complete").id, "complete")
        self.assertIsNone(get_stage("no_such_stage"))


class TestStage3Decomposition(unittest.TestCase):
    """U-3: seam smoothing, facial-anatomy protection, wipeout breaker."""

    def setUp(self):
        self.pipeline = RestorationPipeline(device="cpu", load_models=False)

    def test_protect_facial_anatomy_zeroes_eye_and_mouth_zones(self):
        zone = np.full((200, 200), 255, dtype=np.uint8)
        face = {"bbox": [60, 60, 140, 140]}  # no landmarks -> proportional path
        out = self.pipeline._protect_facial_anatomy(zone, [face], (200, 200, 3))

        fw = fh = 80
        eyes = [
            (60 + int(0.32 * fw), 60 + int(0.38 * fh)),
            (60 + int(0.68 * fw), 60 + int(0.38 * fh)),
        ]
        mouth = (60 + int(0.50 * fw), 60 + int(0.76 * fh))
        for cx, cy in eyes + [mouth]:
            self.assertEqual(int(out[cy, cx]), 0, f"anatomy point ({cx},{cy}) not protected")
        # Non-facial area stays blendable
        self.assertEqual(int(out[10, 10]), 255)

    def test_protect_facial_anatomy_uses_landmarks_when_present(self):
        zone = np.full((200, 200), 255, dtype=np.uint8)
        face = {
            "bbox": [60, 60, 140, 140],
            "landmarks": [(80, 90), (120, 90), (100, 110), (90, 125), (110, 125)],
        }
        out = self.pipeline._protect_facial_anatomy(zone, [face], (200, 200, 3))
        self.assertEqual(int(out[90, 80]), 0)   # left eye
        self.assertEqual(int(out[90, 120]), 0)  # right eye
        self.assertEqual(int(out[125, 100]), 0)  # mouth midpoint

    def test_seam_smooth_blends_only_the_boundary_band(self):
        rng = np.random.default_rng(11)
        img = np.clip(rng.normal(60, 35, (120, 120, 3)), 0, 255).astype(np.uint8)
        mask = np.zeros((120, 120), dtype=np.uint8)
        mask[40:80, 40:80] = 255
        ctx = PipelineContext(img.copy(), {"color_type": "color"})
        ctx.working_image = img.copy()

        self.pipeline._seam_smooth(ctx, mask, faces=[])

        after = ctx.working_image
        self.assertFalse(np.array_equal(img, after), "seam smoothing must alter the canvas")
        # Mask interior (far from any edge) must be untouched by the band.
        self.assertTrue(np.array_equal(after[60, 60], img[60, 60]))
        self.assertTrue(
            any("Seam boundary smoothing applied" in line for line in ctx.processing_log)
        )

    def test_face_wipeout_circuit_breaker_reverts_catastrophic_loss(self):
        rng = np.random.default_rng(3)
        pre = np.clip(128 + rng.normal(0, 30, (200, 200, 3)), 0, 255).astype(np.uint8)
        post = np.full((200, 200, 3), 128, dtype=np.uint8)  # anatomical wipeout

        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[70:90, 70:110] = 255  # ~12.5% of the 80x80 face bbox: breaker range

        ctx = PipelineContext(post.copy(), {"color_type": "color"})
        ctx.working_image = post.copy()
        faces = [{"bbox": [60, 60, 140, 140]}]

        self.pipeline._face_wipeout_circuit_breaker(ctx, pre, mask, faces)

        self.assertTrue(
            np.array_equal(ctx.working_image[60:140, 60:140], pre[60:140, 60:140]),
            "face region must be reverted to the pre-inpaint canvas",
        )
        self.assertIn(
            "inpaint_face_structure_erased_reverted",
            ctx.image_metadata.get("warnings", []),
        )

    def test_face_wipeout_circuit_breaker_ignores_clean_faces(self):
        rng = np.random.default_rng(3)
        pre = np.clip(128 + rng.normal(0, 30, (200, 200, 3)), 0, 255).astype(np.uint8)
        ctx = PipelineContext(pre.copy(), {"color_type": "color"})
        ctx.working_image = pre.copy()
        faces = [{"bbox": [60, 60, 140, 140]}]
        mask = np.zeros((200, 200), dtype=np.uint8)  # no damage in the face

        self.pipeline._face_wipeout_circuit_breaker(ctx, pre.copy(), mask, faces)

        self.assertTrue(np.array_equal(ctx.working_image, pre))
        self.assertNotIn("warnings", ctx.image_metadata)


class TestDeviceManagement(unittest.TestCase):
    """U-5: device_preference resolution with lazy reload."""

    def setUp(self):
        self.pipeline = RestorationPipeline(device="cpu", load_models=False)
        self.assertEqual(self.pipeline._device, "cpu")

    def test_auto_keeps_current_device(self):
        self.assertEqual(self.pipeline._ensure_device("auto"), "cpu")
        self.assertEqual(self.pipeline._ensure_device(""), "cpu")

    def test_same_device_preference_skips_reload(self):
        with mock.patch.object(self.pipeline, "_reload_models_on") as reload_mock:
            self.assertEqual(self.pipeline._ensure_device("cpu"), "cpu")
        reload_mock.assert_not_called()

    def test_cuda_preference_without_cuda_falls_back(self):
        with mock.patch.object(self.pipeline, "_reload_models_on") as reload_mock, \
             mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(self.pipeline._ensure_device("cuda"), "cpu")
        reload_mock.assert_not_called()

    def test_explicit_mismatch_triggers_lazy_reload(self):
        # Simulate a machine with CUDA and a pipeline currently on cpu.
        with mock.patch.object(self.pipeline, "_reload_models_on") as reload_mock, \
             mock.patch.object(torch.cuda, "is_available", return_value=True):
            reload_mock.side_effect = lambda device: setattr(self.pipeline, "_device", device)
            self.assertEqual(self.pipeline._ensure_device("cuda"), "cuda")
        reload_mock.assert_called_once_with("cuda")

    def test_invalid_preference_is_treated_as_auto(self):
        self.assertEqual(self.pipeline._ensure_device("tpu"), "cpu")


if __name__ == "__main__":
    unittest.main()
