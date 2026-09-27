"""
orchestrator.py — Master Two-Tier Closed-Loop Restoration Orchestrator (v2.0).

Coordinates:
- Tier 1: Inner PPCAV loop (Perceive -> Propose -> Corroborate -> Act -> Verify)
          at native resolution (max 2048px), guarded by atomic CanvasTransaction
          and bounded by a 20 Cost Point budget.
- Tier 2: One-way terminal super-resolution (Real-ESRGAN x4plus) and projector
          exhibition grading.
- Provenance tracking outputting synthesis_manifest.json alongside restored image.
- Monotonic SSE progress percentage streaming.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional, Callable, List, Tuple
import numpy as np
import cv2

from ai.pipeline.pipeline_context import PipelineContext
from ai.orchestrator.perception.taxonomy import DefectCategory, DefectCandidate
from ai.orchestrator.perception.feature_bank import FeatureBank
from ai.orchestrator.perception.defect_proposer import DefectProposer
from ai.orchestrator.perception.corroborator import SignalCorroborator
from ai.orchestrator.actions.action_registry import ActionRegistry
from ai.orchestrator.actions.base_action import BaseAction
from ai.orchestrator.transaction import CanvasTransaction
from ai.orchestrator.loop_controller import LoopController
from ai.orchestrator.synthesis_manifest import ActionStatus, ActionTier
from ai.analysis.quality_analyzer import QualityAnalyzer
from ai.analysis.face_detector import FaceDetector

logger = logging.getLogger(__name__)


class PhotoOrchestrator:
    """Production Two-Tier PPCAV Orchestrator replacing static feedforward waterfalls."""

    def __init__(
        self,
        registry: Optional[ActionRegistry] = None,
        max_cost_points: int = 20,
        face_detector: Optional[FaceDetector] = None,
    ) -> None:
        self.registry: ActionRegistry = registry or ActionRegistry()
        self.proposer: DefectProposer = DefectProposer()
        self.corroborator: SignalCorroborator = SignalCorroborator()
        self.analyzer: QualityAnalyzer = QualityAnalyzer()
        self._face_detector: Optional[FaceDetector] = face_detector
        self.max_cost_points: int = max_cost_points

    @property
    def face_detector(self) -> FaceDetector:
        if self._face_detector is None:
            self._face_detector = FaceDetector()
        return self._face_detector

    def run(
        self,
        context: PipelineContext,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        enable_repair: bool = True,
        enable_denoise: bool = True,
        enable_colorization: bool = True,
        enable_face_restore: bool = True,
        enable_sr: bool = True,
    ) -> np.ndarray:
        """Execute full Two-Tier closed-loop restoration."""
        start_time = time.perf_counter()

        def _emit_progress(stage: str, pct: int, msg: str):
            if progress_callback:
                try:
                    progress_callback(stage, pct, msg)
                except Exception as exc:
                    logger.debug("Progress callback failed: %s", exc)

        _emit_progress("analyze", 10, "Perceiving image quality & baseline degradation...")

        # Run baseline analysis
        analysis_meta = self.analyzer.analyze(context.working_image)
        context.image_metadata.update(analysis_meta)

        # Detect faces early (FaceDetector updates context.image_metadata['faces'] in-place)
        if enable_face_restore or enable_repair:
            try:
                self.face_detector.detect(context)
            except Exception as exc:
                context.add_log(f"Face detection skipped or failed: {exc}")

        controller = LoopController(max_cost_points=self.max_cost_points)

        # ── TIER 1: Inner PPCAV Iterative Repair Loop (Native Resolution) ───────────
        context.add_log("=== Starting Tier 1: Inner PPCAV Closed-Loop ===")
        iteration = 0

        while not controller.is_budget_depleted and iteration < 15:
            iteration += 1
            cur_img = context.working_image

            # Step 1: Propose Candidates
            candidates = self.proposer.propose(cur_img, metadata=context.image_metadata)

            # Step 2: Corroborate Candidates (Proposer != Verifier)
            verified_candidates: List[DefectCandidate] = []
            for cand in candidates:
                # Check user toggle flags
                if cand.category in [DefectCategory.TEAR_CRACK, DefectCategory.FLAKING_MIRRORING, DefectCategory.HOLE_MISSING] and not enable_repair:
                    continue
                if cand.category == DefectCategory.UNCOLORED and not enable_colorization:
                    continue

                checked = self.corroborator.corroborate(cur_img, cand)
                if checked.corroborated:
                    verified_candidates.append(checked)

            # Check for convergence
            if not verified_candidates:
                context.add_log(f"[TIER 1 CONVERGED] No unresolved defects at iteration {iteration}")
                break

            # Sort by corroboration score * confidence
            verified_candidates.sort(
                key=lambda c: (c.corroboration_score * c.confidence), reverse=True
            )
            target_candidate = verified_candidates[0]

            # Step 3: Action Selection & Precondition Check
            action: Optional[BaseAction] = self.registry.get_for_candidate(target_candidate)
            if not action:
                context.add_log(f"[WARN] No registered action for category {target_candidate.category.value}")
                continue

            can_run, reason = controller.can_execute(action.name, action.cost_points, target_candidate.bbox)
            if not can_run:
                context.add_log(f"[SKIP] Action '{action.name}' skipped: {reason}")
                # Remove candidate from consideration this round
                verified_candidates.pop(0)
                if not verified_candidates:
                    break
                target_candidate = verified_candidates[0]
                action = self.registry.get_for_candidate(target_candidate)
                if not action:
                    break

            # Step 4 & 5: Act & Verify inside CanvasTransaction
            progress_pct = controller.get_tier1_progress(
                unresolved_defect_ratio=len(verified_candidates) / 10.0
            )
            _emit_progress("repair", progress_pct, f"Applying {action.name} ({target_candidate.category.value})...")

            with context.begin_transaction(action.name) as tx:
                try:
                    passed, v_score, v_msg = action.execute(context, candidate=target_candidate)
                    if passed:
                        tx.commit(
                            tier=action.tier,
                            bbox=target_candidate.bbox,
                            confidence=target_candidate.confidence,
                            verification_score=v_score,
                            metadata={"category": target_candidate.category.value, "details": v_msg},
                        )
                        controller.register_execution(
                            action.name,
                            action.cost_points,
                            bbox=target_candidate.bbox,
                            metric_value=v_score,
                            passed=True,
                        )
                    else:
                        tx.rollback(
                            reason=v_msg,
                            step_down_param="nlm_h" if "denoise" in action.name.lower() else None,
                            tier=action.tier,
                            bbox=target_candidate.bbox,
                        )
                        controller.register_execution(
                            action.name,
                            action.cost_points,
                            bbox=target_candidate.bbox,
                            metric_value=0.0,
                            passed=False,
                        )
                except Exception as exc:
                    tx.rollback(reason=f"Execution error: {exc}", tier=action.tier)
                    controller.register_execution(
                        action.name,
                        action.cost_points,
                        bbox=target_candidate.bbox,
                        metric_value=0.0,
                        passed=False,
                    )

        # Standard denoise & face restore pass if enabled and not already committed
        committed_actions = {
            a.action for a in context.synthesis_manifest.actions if a.status == ActionStatus.COMMITTED.value
        }

        if enable_denoise and "NLMDenoise" not in committed_actions and not controller.is_budget_depleted:
            denoise_act = self.registry.get_by_name("NLMDenoise")
            if denoise_act:
                _emit_progress("denoise", 60, "Applying noise-adaptive 3-zone NLM denoising...")
                with context.begin_transaction("NLMDenoise") as tx:
                    passed, score, msg = denoise_act.execute(context)
                    if passed:
                        tx.commit(tier=denoise_act.tier, verification_score=score)
                        controller.register_execution("NLMDenoise", denoise_act.cost_points)

        if enable_face_restore and "GFPGANRestore" not in committed_actions and not controller.is_budget_depleted:
            face_act = self.registry.get_by_name("GFPGANRestore")
            if face_act:
                _emit_progress("face", 75, "Restoring facial anatomy and landmarks...")
                with context.begin_transaction("GFPGANRestore") as tx:
                    passed, score, msg = face_act.execute(context)
                    if passed:
                        tx.commit(tier=face_act.tier, verification_score=score)
                        controller.register_execution("GFPGANRestore", face_act.cost_points)

        # ── TIER 2: Terminal Expansion & Exhibition Polish ─────────────────────────
        context.add_log("=== Starting Tier 2: Terminal Super-Resolution & Exhibition Polish ===")
        _emit_progress("super_res", 85, "Executing terminal Real-ESRGAN super-resolution...")

        if enable_sr:
            sr_action = self.registry.get_by_name("RealESRGANSuperRes")
            if sr_action:
                with context.begin_transaction("RealESRGANSuperRes") as tx:
                    passed, score, msg = sr_action.execute(context)
                    if passed:
                        tx.commit(tier=sr_action.tier, verification_score=score)
                        context.add_log(f"[TIER 2] {msg}")

        _emit_progress("enhance", 95, "Applying final projector exhibition grade...")
        # Post-process tone & unsharp mask (built into superres or applied gently)
        elapsed = time.perf_counter() - start_time
        context.add_log(f"[COMPLETE] Two-Tier Orchestrator finished in {elapsed:.2f}s")
        _emit_progress("complete", 100, "Restoration complete.")

        return context.working_image
