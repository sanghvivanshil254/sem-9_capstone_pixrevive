"""
Unit & Integration tests for Phase 4 of PixRevive Two-Tier Orchestrator:
- PhotoOrchestrator end-to-end execution.
- Monotonic progress callback emission.
- SynthesisManifest logging and provenance auditing.
- Toggle flag enforcement (disable repair, disable sr).
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ai.orchestrator.orchestrator import PhotoOrchestrator
from ai.orchestrator.actions.action_registry import ActionRegistry
from ai.orchestrator.actions.base_action import BaseAction
from ai.orchestrator.synthesis_manifest import ActionTier
from ai.orchestrator.perception.taxonomy import DefectCategory, DefectCandidate
from ai.pipeline.pipeline_context import PipelineContext


class MockFailingAction(BaseAction):
    name = "MockFailing"
    cost_points = 2
    tier = ActionTier.FABRICATED

    def execute(self, context, candidate=None):
        context.working_image[0, 0] = 0
        return False, 0.0, "Artificial verification rejection"


class TestOrchestratorPhase4(unittest.TestCase):
    def setUp(self):
        # Create a simple test canvas
        self.sample_img = np.full((120, 120, 3), 160, dtype=np.uint8)
        self.context = PipelineContext(self.sample_img)

    def test_orchestrator_monotonic_progress_emission(self):
        orchestrator = PhotoOrchestrator(max_cost_points=10)
        progress_events = []

        def callback(stage, pct, msg):
            progress_events.append((stage, pct, msg))

        # Run with SR disabled to keep test execution fast
        restored = orchestrator.run(
            self.context,
            progress_callback=callback,
            enable_repair=True,
            enable_denoise=False,
            enable_colorization=False,
            enable_face_restore=False,
            enable_sr=False,
        )

        self.assertEqual(restored.shape, self.sample_img.shape)
        self.assertGreater(len(progress_events), 2)

        # Check monotonic non-decreasing percentages
        percentages = [p[1] for p in progress_events]
        for i in range(1, len(percentages)):
            self.assertGreaterEqual(
                percentages[i], percentages[i - 1],
                f"Progress non-monotonic: {percentages[i-1]} -> {percentages[i]}"
            )
        self.assertEqual(percentages[-1], 100)

    def test_orchestrator_manifest_generation(self):
        orchestrator = PhotoOrchestrator(max_cost_points=10)
        orchestrator.run(
            self.context,
            enable_repair=True,
            enable_denoise=False,
            enable_colorization=False,
            enable_face_restore=False,
            enable_sr=False,
        )

        manifest = self.context.synthesis_manifest
        manifest_dict = manifest.to_dict()

        self.assertIn("overall_authenticity_score", manifest_dict)
        self.assertIn("committed_actions", manifest_dict)
        self.assertIsInstance(manifest_dict["committed_actions"], list)

    def test_orchestrator_handles_rollback_gracefully(self):
        registry = ActionRegistry()
        failing_act = MockFailingAction()
        registry.register(failing_act)
        registry.bind_category(DefectCategory.OVER_SMOOTHED, failing_act.name)

        # Create smooth image that will trigger over-smoothed candidate
        smooth = np.full((700, 700, 3), 128, dtype=np.uint8)
        context = PipelineContext(smooth)

        orchestrator = PhotoOrchestrator(registry=registry, max_cost_points=6)
        orchestrator.run(
            context,
            enable_repair=False,
            enable_denoise=False,
            enable_colorization=False,
            enable_face_restore=False,
            enable_sr=False,
        )

        # Confirm rollback was logged in synthesis manifest
        rolled_back = [
            a for a in context.synthesis_manifest.actions if a.status == "ROLLED_BACK"
        ]
        self.assertGreaterEqual(len(rolled_back), 1)
        self.assertIn("Artificial verification rejection", rolled_back[0].reason)

    def test_restoration_pipeline_delegates_to_orchestrator(self):
        from ai.pipeline.restoration_pipeline import RestorationPipeline
        pipeline = RestorationPipeline()

        test_img = np.full((100, 100, 3), 180, dtype=np.uint8)
        events = []

        def cb(stage, pct, msg):
            events.append((stage, pct, msg))

        # Test restore with use_orchestrator=True
        restored = pipeline.restore(
            test_img,
            enable_repair=False,
            enable_denoise=False,
            enable_colorization=False,
            enable_face_restore=False,
            enable_sr=False,
            on_progress=cb,
            use_orchestrator=True,
        )

        self.assertEqual(restored.shape, test_img.shape)
        self.assertTrue(pipeline.last_context.pipeline_flags.get("use_orchestrator", False))
        self.assertIsNotNone(pipeline.last_context.synthesis_manifest)
        self.assertGreater(len(events), 0)

        # Test process with use_orchestrator=True returning PNG bytes
        png_bytes = pipeline.process(
            test_img,
            enable_repair=False,
            enable_denoise=False,
            enable_colorization=False,
            enable_restoration=False,
            enable_sr=False,
            use_orchestrator=True,
        )
        self.assertTrue(png_bytes.startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()

