"""Stage-level tests for the re-imagine pipeline stage (Phase H1).

Covers the gate contract, registry wiring, report disclosure and API plumbing
with the diffusion engine mocked — no weights, runs unconditionally.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
)

from ai.pipeline.pipeline_context import PipelineContext  # noqa: E402
from ai.pipeline.stage_registry import STAGE_REGISTRY, get_stage  # noqa: E402
from ai.report.restoration_report import build_restoration_report  # noqa: E402
from ai.restoration.resource_profiler import MINIMAL  # noqa: E402


def _ctx(frame_pct: float = 0.10, face_pct: float | None = None):
    img = np.full((128, 128, 3), 180, np.uint8)
    ctx = PipelineContext(img, {"filename": "t.png"})
    side = int(128 * (frame_pct ** 0.5))
    mask = np.zeros((128, 128), np.uint8)
    mask[:side, :side] = 255
    ctx.damage_masks["final"] = mask
    faces = []
    if face_pct is not None:
        # A 32x32 face bbox at (0,0); damage occupies face_pct of it.
        faces.append({"bbox": [0, 0, 32, 32, 1.0]})
        covered = int(32 * 32 * face_pct / 100)
        mask.reshape(-1)[:covered] = 255
    ctx.image_metadata["faces"] = faces
    return ctx


class _FakeResult(dict):
    pass


def _done_result():
    return _FakeResult(
        status="done", result=np.full((128, 128, 3), 90, np.uint8),
        real_pixels_pct=93.8, generated_pixels_pct=6.2, evidence_preserved=True,
        identity={"action": "pass", "similarity": 0.55, "reason": "ok"},
        resource={"tier": "minimal", "reason": "autodetect",
                  "config": {"dtype": "fp32", "generator": "sd15"}},
        seed=42, generator="sd15", steps=25,
    )


class TestRegistryWiring(unittest.TestCase):
    def test_reimagine_stage_registered_and_gated(self):
        spec = get_stage("reimagine")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.method, "_stage_12_reimagine")
        self.assertEqual(spec.route_gate, "enable_reimagine")

    def test_progress_stays_monotonic(self):
        # Only emitting stages (pct > 0) participate; refresh_faces is a
        # pct-0 non-emitting entry by long-standing design (the live U-7
        # stream validator checks emitted events only).
        pcts = [s.progress_pct for s in STAGE_REGISTRY if s.progress_pct > 0]
        self.assertEqual(pcts, sorted(pcts), "registry percentages must be non-decreasing")
        self.assertEqual(pcts[-1], 99)  # reimagine/output_verify sit at 99, complete=100


class TestStageGate(unittest.TestCase):
    def _run_stage(self, ctx, **engine_ret):
        from ai.pipeline.stages.finish_stages import FinishStagesMixin  # type: ignore
        inst = object.__new__(FinishStagesMixin)
        ret = _done_result()
        ret.update(engine_ret)
        with mock.patch("ai.restoration.reimaginer.reimagine", return_value=ret):
            inst._stage_12_reimagine(ctx, on_progress=None)
        return ctx.image_metadata.get("reimagine", {})

    def test_below_coverage_bar_skips_without_engine_call(self):
        ctx = _ctx(frame_pct=0.05)  # 0.39% of frame — far below 35%
        with mock.patch("ai.restoration.reimaginer.reimagine") as eng:
            rec = self._run_stage(ctx)
        eng.assert_not_called()
        self.assertEqual(rec["status"], "skipped")
        self.assertIn("coverage", rec["reason"])

    def test_hopeless_frame_coverage_fires_engine(self):
        ctx = _ctx(frame_pct=0.62)  # 38% of frame — above the 35% bar
        rec = self._run_stage(ctx)
        self.assertEqual(rec["status"], "done")

    def test_hopeless_face_coverage_fires_engine(self):
        ctx = _ctx(frame_pct=0.05, face_pct=80.0)  # 80% of the face destroyed
        rec = self._run_stage(ctx)
        self.assertEqual(rec["status"], "done")
        self.assertGreaterEqual(rec["coverage"]["worst_face_pct"], 55.0)

    def test_engine_skip_is_recorded(self):
        ctx = _ctx(frame_pct=0.62)
        rec = self._run_stage(ctx, status="skipped", reason="unavailable: no_gpu")
        self.assertEqual(rec["status"], "skipped")
        self.assertIn("no_gpu", rec["reason"])

    def test_engine_rollback_is_recorded_verbatim(self):
        ctx = _ctx(frame_pct=0.62)
        rec = self._run_stage(ctx, status="skipped_degraded",
                              reason="re-imagine rolled back: identity")
        self.assertEqual(rec["status"], "skipped_degraded")

    def test_working_image_swapped_on_done(self):
        ctx = _ctx(frame_pct=0.62)
        self._run_stage(ctx)
        self.assertEqual(int(ctx.working_image.mean()), 90)

    def test_stage_failure_never_raises(self):
        ctx = _ctx(frame_pct=0.62)
        from ai.pipeline.stages.finish_stages import FinishStagesMixin  # type: ignore
        inst = object.__new__(FinishStagesMixin)
        with mock.patch("ai.restoration.reimaginer.reimagine",
                        side_effect=RuntimeError("boom")):
            inst._stage_12_reimagine(ctx, on_progress=None)  # must not raise
        self.assertIn("error", ctx.image_metadata["reimagine"])

    def test_done_run_persists_evidence_overlay(self):
        """'Show what's real' (design §7): a hybrid run writes the overlay
        PNG and records its API path in the record."""
        import os
        import tempfile

        from ai.pipeline.stages import finish_stages as fs

        ctx = _ctx(frame_pct=0.62)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(fs, "EVIDENCE_DIR", tmp), \
                 mock.patch("ai.restoration.reimaginer.reimagine",
                            return_value=_done_result()):
                inst = object.__new__(fs.FinishStagesMixin)
                inst._stage_12_reimagine(ctx, on_progress=None)
            rec = ctx.image_metadata["reimagine"]
            self.assertTrue(rec["evidence_overlay"].startswith("/api/process/"))
            written = [f for f in os.listdir(tmp) if f.endswith("_evidence.png")]
            self.assertEqual(len(written), 1)
        # The report block exposes the overlay URL for the frontend toggle.
        rep = build_restoration_report(ctx)
        self.assertTrue(rep["reimagine"]["evidence_overlay"].startswith(
            "/api/process/"))


class TestReportDisclosure(unittest.TestCase):
    def _report(self, reimagine_meta):
        ctx = _ctx(frame_pct=0.62)
        ctx.image_metadata["reimagine"] = reimagine_meta
        return build_restoration_report(ctx)

    def test_done_block_discloses_evidence_split(self):
        rep = self._report(dict(_done_result(), ran=True))
        block = rep["reimagine"]
        self.assertEqual(block["mode"], "hybrid-restoration")
        self.assertEqual(block["status"], "done")
        self.assertAlmostEqual(block["real_pixels_pct"], 93.8)
        self.assertTrue(block["evidence_preserved"])
        self.assertEqual(block["resource"]["tier"], "minimal")
        self.assertEqual(block["identity"]["similarity"], 0.55)
        self.assertTrue(any("6% of the frame re-imagined" in s
                            for s in rep["reconstructed"]))

    def test_flagged_identity_surfaces_in_flags(self):
        base = _done_result()
        base["identity"] = {"action": "flag", "similarity": 0.38, "reason": "ok"}
        rep = self._report(dict(base, ran=True))
        self.assertTrue(any("drifted" in f for f in rep["flagged"]))

    def test_rollback_disclosed_as_flag(self):
        rep = self._report({"ran": True, "status": "skipped_degraded",
                            "reason": "identity could not be preserved"})
        self.assertEqual(rep["reimagine"]["mode"], "not_applied")
        self.assertTrue(any("rolled back" in f for f in rep["flagged"]))

    def test_no_reimagine_meta_means_null_block(self):
        rep = build_restoration_report(_ctx())
        self.assertIsNone(rep["reimagine"])

    def test_block_absent_from_error_reports_shape(self):
        rep = self._report({"ran": True, "status": "skipped", "reason": "below bar"})
        self.assertEqual(rep["reimagine"]["mode"], "not_applied")
        self.assertIn("below bar", rep["reimagine"]["reason"])


class TestAPIPlumbing(unittest.TestCase):
    def test_restore_signature_accepts_enable_reimagine(self):
        import inspect
        from ai.pipeline.restoration_pipeline import RestorationPipeline
        sig = inspect.signature(RestorationPipeline.restore)
        self.assertIn("enable_reimagine", sig.parameters)
        self.assertIs(sig.parameters["enable_reimagine"].default, False)

    def test_process_signature_accepts_enable_reimagine(self):
        import inspect
        from ai.pipeline.restoration_pipeline import RestorationPipeline
        sig = inspect.signature(RestorationPipeline.process)
        self.assertIn("enable_reimagine", sig.parameters)

    def test_service_forwards_the_flag(self):
        import inspect
        from services import image_service
        src = inspect.getsource(image_service.ImageService.process_image)
        self.assertIn("enable_reimagine", src)

    def test_router_forms_expose_the_flag(self):
        import inspect
        import routers.image_router as router
        src = inspect.getsource(router)
        self.assertEqual(src.count("reimagine: bool = Form(False)"), 2)


if __name__ == "__main__":
    unittest.main()
