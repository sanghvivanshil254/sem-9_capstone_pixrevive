"""
Unit tests for Phase 1 of PixRevive Two-Tier Orchestrator:
- SynthesisManifest provenance tracking & scoring.
- CanvasTransaction atomic rollback & step-down curves.
- LoopController budget caps, oscillation detection, and monotonic progress.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ai.orchestrator.synthesis_manifest import SynthesisManifest, ActionTier, ActionStatus
from ai.orchestrator.transaction import CanvasTransaction
from ai.orchestrator.loop_controller import LoopController
from ai.pipeline.pipeline_context import PipelineContext


class TestOrchestratorPhase1(unittest.TestCase):
    def setUp(self):
        self.sample_img = np.full((100, 100, 3), 128, dtype=np.uint8)
        self.context = PipelineContext(self.sample_img)

    def test_manifest_logging_and_authenticity(self):
        manifest = SynthesisManifest()
        manifest.log_action("DeskewCrop", ActionStatus.COMMITTED, ActionTier.RECOVERED)
        manifest.log_action("NLMDenoise", ActionStatus.COMMITTED, ActionTier.RECOVERED)
        manifest.log_action("LaMaInpaint", ActionStatus.COMMITTED, ActionTier.FABRICATED)
        manifest.log_action("GFPGANRestore", ActionStatus.COMMITTED, ActionTier.PARTIALLY_FABRICATED)
        manifest.log_action("FailedAction", ActionStatus.ROLLED_BACK, ActionTier.FABRICATED, reason="Seam error")

        # 4 committed actions: 2 recovered (1.0 each), 1 partially (0.5), 1 fabricated (0.0)
        # Authenticity score = (1.0 + 1.0 + 0.5 + 0.0) / 4 = 2.5 / 4 = 0.625
        self.assertAlmostEqual(manifest.authenticity_score, 0.625)

        data = manifest.to_dict()
        self.assertEqual(data["total_actions_logged"], 5)
        self.assertEqual(len(data["committed_actions"]), 4)
        self.assertEqual(len(data["rolled_back_actions"]), 1)
        self.assertIn("overall_authenticity_score", data)

    def test_canvas_transaction_commit(self):
        with self.context.begin_transaction("TestCommit") as tx:
            self.context.working_image[10:20, 10:20] = 255
            tx.commit(tier=ActionTier.RECOVERED, verification_score=0.95)

        # Pixel modification should be preserved
        self.assertEqual(self.context.working_image[15, 15, 0], 255)
        self.assertEqual(len(self.context.synthesis_manifest.actions), 1)
        self.assertEqual(self.context.synthesis_manifest.actions[0].status, "COMMITTED")

    def test_canvas_transaction_rollback_reverts_pixels_exactly(self):
        original_copy = self.context.working_image.copy()
        expected_mask = np.zeros((100, 100), dtype=np.uint8)
        self.context.damage_masks["test_mask"] = expected_mask.copy()

        with self.context.begin_transaction("TestRollback") as tx:
            # Corrupt image heavily
            self.context.working_image[:, :] = 0
            self.context.damage_masks["test_mask"][:, :] = 255
            tx.rollback(reason="Verification rejected high artifacting", step_down_param="nlm_h")

        # Must be 100% byte-for-byte identical to original
        np.testing.assert_array_equal(self.context.working_image, original_copy)
        np.testing.assert_array_equal(self.context.damage_masks["test_mask"], expected_mask)

        # Manifest must reflect rollback
        actions = self.context.synthesis_manifest.actions
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].status, "ROLLED_BACK")
        self.assertEqual(actions[0].reason, "Verification rejected high artifacting")

        # nlm_h must have stepped down from 10.0 to 7.0
        self.assertEqual(self.context.hyperparameters["nlm_h"], 7.0)

    def test_canvas_transaction_auto_rollback_on_exception(self):
        original_copy = self.context.working_image.copy()

        try:
            with self.context.begin_transaction("FailingAction") as tx:
                self.context.working_image[0, 0] = 99
                raise ValueError("Simulated neural crash")
        except ValueError:
            pass

        # Image should be rolled back despite the exception
        np.testing.assert_array_equal(self.context.working_image, original_copy)
        self.assertEqual(self.context.synthesis_manifest.actions[-1].status, "ROLLED_BACK")

    def test_loop_controller_budget_and_oscillation(self):
        controller = LoopController(max_cost_points=10, max_action_retries=3, oscillation_window=4)

        # Register Action A (cost 3)
        can_run, _ = controller.can_execute("ActionA", cost_points=3)
        self.assertTrue(can_run)
        controller.register_execution("ActionA", cost_points=3, metric_value=0.5)

        # Register Action B (cost 4)
        can_run, _ = controller.can_execute("ActionB", cost_points=4)
        self.assertTrue(can_run)
        controller.register_execution("ActionB", cost_points=4, metric_value=0.6)

        # Attempt Action A again -> should detect oscillation A -> B -> A
        can_run, msg = controller.can_execute("ActionA", cost_points=3)
        self.assertFalse(can_run)
        self.assertIn("Oscillation detected", msg)

        # Register Action C (cost 3) -> budget hits 3 + 4 + 3 = 10 CP
        can_run, _ = controller.can_execute("ActionC", cost_points=3)
        self.assertTrue(can_run)
        controller.register_execution("ActionC", cost_points=3, metric_value=0.8)

        # Budget should now be completely depleted
        self.assertTrue(controller.is_budget_depleted)
        can_run, msg = controller.can_execute("ActionD", cost_points=1)
        self.assertFalse(can_run)
        self.assertIn("Budget exceeded", msg)

    def test_loop_controller_monotonic_progress(self):
        controller = LoopController(max_cost_points=20)
        p1 = controller.get_tier1_progress(unresolved_defect_ratio=0.8)
        controller.register_execution("Action1", cost_points=5)
        p2 = controller.get_tier1_progress(unresolved_defect_ratio=0.5)
        controller.register_execution("Action2", cost_points=5)
        p3 = controller.get_tier1_progress(unresolved_defect_ratio=0.2)

        self.assertGreaterEqual(p2, p1)
        self.assertGreaterEqual(p3, p2)
        self.assertLessEqual(p3, 85)


if __name__ == "__main__":
    unittest.main()
