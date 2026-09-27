"""
loop_controller.py — Budget, Oscillation, and Convergence Controller.

Enforces:
1. Global Budget Cap (Default: 20 Cost Points) to prevent run-away latency.
2. Per-category / per-action retry limits (Max 3 attempts per region).
3. 4-step sliding window oscillation detection (breaking Action A ↔ Action B cycles).
4. Diminishing returns detection (< 2% improvement between consecutive passes).
5. Monotonic progress percentage calculation for SSE streaming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class ActionExecutionRecord:
    iteration: int
    action_name: str
    cost_points: int
    bbox: Optional[List[int]] = None
    metric_value: float = 0.0
    passed: bool = True


class LoopController:
    """Controls the inner restoration loop termination, oscillation, and budgeting."""

    def __init__(
        self,
        max_cost_points: int = 20,
        max_action_retries: int = 3,
        oscillation_window: int = 4,
        min_diminishing_delta: float = 0.02,
    ) -> None:
        self.max_cost_points: int = max_cost_points
        self.max_action_retries: int = max_action_retries
        self.oscillation_window: int = oscillation_window
        self.min_diminishing_delta: float = min_diminishing_delta

        self.accumulated_cost_points: int = 0
        self.iteration: int = 0
        self.history: List[ActionExecutionRecord] = []
        self.action_counts: Dict[str, int] = {}
        self.blacklisted_actions: set[str] = set()
        self._last_progress: float = 10.0

    def can_execute(
        self, action_name: str, cost_points: int, bbox: Optional[List[int]] = None
    ) -> Tuple[bool, str]:
        """Check if an action is eligible to fire under budget and loop guardrails."""
        if action_name in self.blacklisted_actions:
            return False, f"Action '{action_name}' is blacklisted due to oscillation"

        if self.accumulated_cost_points + cost_points > self.max_cost_points:
            return False, f"Budget exceeded: {self.accumulated_cost_points} + {cost_points} > {self.max_cost_points} CP"

        attempts = self.action_counts.get(action_name, 0)
        if attempts >= self.max_action_retries:
            return False, f"Max retries reached for '{action_name}' ({attempts}/{self.max_action_retries})"

        if self.check_oscillation(action_name, bbox):
            self.blacklisted_actions.add(action_name)
            return False, f"Oscillation detected for '{action_name}'; blacklisted"

        return True, "Eligible"

    def register_execution(
        self,
        action_name: str,
        cost_points: int,
        bbox: Optional[List[int]] = None,
        metric_value: float = 0.0,
        passed: bool = True,
    ) -> None:
        """Record an executed action into history and deplete budget."""
        self.iteration += 1
        self.accumulated_cost_points += cost_points
        self.action_counts[action_name] = self.action_counts.get(action_name, 0) + 1

        record = ActionExecutionRecord(
            iteration=self.iteration,
            action_name=action_name,
            cost_points=cost_points,
            bbox=bbox,
            metric_value=metric_value,
            passed=passed,
        )
        self.history.append(record)

    def check_oscillation(self, action_name: str, bbox: Optional[List[int]] = None) -> bool:
        """Check if action alternates back and forth within the sliding window."""
        if len(self.history) < 2:
            return False

        window = self.history[-self.oscillation_window:]
        actions = [r.action_name for r in window]

        # Check if candidate would complete A -> B -> A pattern
        if len(actions) >= 2 and actions[-1] != action_name and actions[-2] == action_name:
            return True

        # Check for immediate repeat failure
        if len(actions) >= 1 and actions[-1] == action_name and not window[-1].passed:
            return True

        return False

    def check_diminishing_returns(self, action_name: str, new_metric: float) -> bool:
        """Return True if repeated attempts fail to achieve meaningful metric gains."""
        matching = [r for r in self.history if r.action_name == action_name and r.passed]
        if not matching:
            return False

        last_metric = matching[-1].metric_value
        improvement = abs(new_metric - last_metric)
        return improvement < self.min_diminishing_delta

    @property
    def is_budget_depleted(self) -> bool:
        return self.accumulated_cost_points >= self.max_cost_points

    def is_converged(
        self, unresolved_defect_count: int, max_defect_confidence: float = 0.0
    ) -> bool:
        """Check if the restoration loop has achieved mathematical convergence."""
        if unresolved_defect_count == 0:
            return True
        if max_defect_confidence < 0.35:
            return True
        if self.is_budget_depleted:
            return True
        return False

    def get_tier1_progress(self, unresolved_defect_ratio: float = 0.0) -> int:
        """Compute strictly monotonic progress percentage for Tier 1 (10% to 85%)."""
        budget_fraction = min(1.0, self.accumulated_cost_points / max(1, self.max_cost_points))
        defect_fraction = max(0.0, 1.0 - unresolved_defect_ratio)

        # Weighted blend between budget depleted and defects resolved
        raw_pct = 10.0 + 75.0 * (0.6 * budget_fraction + 0.4 * defect_fraction)
        monotonic_pct = max(self._last_progress, raw_pct)
        self._last_progress = min(85.0, monotonic_pct)
        return int(round(self._last_progress))

    def summary(self) -> Dict[str, Any]:
        """Return controller execution summary."""
        return {
            "total_iterations": self.iteration,
            "accumulated_cost_points": self.accumulated_cost_points,
            "max_cost_points": self.max_cost_points,
            "budget_depleted": self.is_budget_depleted,
            "blacklisted_actions": list(self.blacklisted_actions),
            "action_counts": dict(self.action_counts),
        }
