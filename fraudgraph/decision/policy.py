"""The decision policy (A6, task 4.2): three layers, applied in order.

1. Expected cost: the cheapest action for this payment (APPROVE / STEP_UP / REVIEW /
   BLOCK), a per-transaction threshold that depends on the amount.
2. Conformal override (needs a conformal model):
   - set {fraud}             -> at least REVIEW                    reason "conformal_fraud"
   - set {legit, fraud} / {} -> at least `uncertain_action`        reason "model_uncertain"
   - set {legit}             -> keep the cost decision
   "At least" uses severity APPROVE < STEP_UP < REVIEW < BLOCK, so a BLOCK is never
   softened. `uncertain_action` is REVIEW (a human) or STEP_UP (challenge the customer).
3. Review capacity: at most `review_capacity_per_day` REVIEWs per 24-step day. The
   serving path is first-come (a counter); offline we also report an oracle that keeps
   the reviews with the largest expected savings E[APPROVE] - E[REVIEW]. Overflow
   falls back to the cheapest non-review action, reason "capacity_overflow".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fraudgraph.decision.conformal import ConformalModel, set_labels
from fraudgraph.decision.costs import (
    APPROVE,
    BLOCK,
    REVIEW,
    SEVERITY,
    STEP_UP,
    CostParams,
    expected_costs,
)

STEPS_PER_DAY = 24
REASONS = ("expected_cost", "conformal_fraud", "model_uncertain", "capacity_overflow")


@dataclass(frozen=True)
class PolicyConfig:
    costs: CostParams
    conformal: ConformalModel | None = None
    uncertain_action: int = REVIEW
    review_capacity_per_day: int | None = None
    capacity_mode: str = "first_come"  # or "oracle" (offline upper bound only)

    def __post_init__(self) -> None:
        if self.uncertain_action not in (REVIEW, STEP_UP):
            raise ValueError("uncertain_action must be REVIEW or STEP_UP")
        if self.uncertain_action == STEP_UP and not self.costs.step_up_enabled:
            raise ValueError("uncertain_action STEP_UP needs costs.step_up_enabled")
        if self.capacity_mode not in ("first_come", "oracle"):
            raise ValueError("capacity_mode must be first_come or oracle")


@dataclass
class Decisions:
    actions: np.ndarray  # action index per row
    reasons: np.ndarray  # reason string per row
    sets: np.ndarray | None = None  # conformal membership [n, 2], if used
    expected: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))


def at_least(actions: np.ndarray, floor: int) -> np.ndarray:
    """Raise every action to at least `floor` severity (never lowers one)."""
    severity = np.vectorize(SEVERITY.__getitem__)(actions)
    return np.where(severity < SEVERITY[floor], floor, actions)


def decide(p: np.ndarray, amount: np.ndarray, step: np.ndarray, config: PolicyConfig) -> Decisions:
    """Batch decisions for rows in step order (capacity is applied per 24-step day)."""
    expected = expected_costs(p, amount, config.costs)
    actions = np.argmin(expected, axis=1)
    reasons = np.full(len(p), "expected_cost", dtype=object)
    sets = None

    if config.conformal is not None:
        sets = config.conformal.sets(p)
        label = set_labels(sets)
        fraud_only = label == "fraud"
        raised = at_least(actions, REVIEW)
        reasons[fraud_only & (raised != actions)] = "conformal_fraud"
        actions = np.where(fraud_only, raised, actions)
        unsure = (label == "uncertain") | (label == "empty")
        raised = at_least(actions, config.uncertain_action)
        reasons[unsure & (raised != actions)] = "model_uncertain"
        actions = np.where(unsure, raised, actions)

    if config.review_capacity_per_day is not None:
        actions, reasons = apply_capacity(actions, reasons, expected, step, config)
    return Decisions(actions=actions, reasons=reasons, sets=sets, expected=expected)


def apply_capacity(
    actions: np.ndarray,
    reasons: np.ndarray,
    expected: np.ndarray,
    step: np.ndarray,
    config: PolicyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    capacity = config.review_capacity_per_day
    assert capacity is not None
    actions, reasons = actions.copy(), reasons.copy()
    fallback = _cheapest_non_review(expected)
    savings = expected[:, APPROVE] - expected[:, REVIEW]
    day = (np.asarray(step) - 1) // STEPS_PER_DAY
    for d in np.unique(day):
        reviews = np.flatnonzero((day == d) & (actions == REVIEW))
        if len(reviews) <= capacity:
            continue
        if config.capacity_mode == "oracle":
            keep = reviews[np.argsort(-savings[reviews], kind="stable")[:capacity]]
        else:  # rows arrive in step order: the first `capacity` reviews get analysts
            keep = reviews[:capacity]
        overflow = np.setdiff1d(reviews, keep)
        actions[overflow] = fallback[overflow]
        reasons[overflow] = "capacity_overflow"
    return actions, reasons


def _cheapest_non_review(expected: np.ndarray) -> np.ndarray:
    masked = expected.copy()
    masked[:, REVIEW] = np.inf
    return np.argmin(masked, axis=1)


class OnlinePolicy:
    """Per-request version for the API: same layers, first-come review capacity."""

    def __init__(self, config: PolicyConfig) -> None:
        if config.capacity_mode != "first_come":
            raise ValueError("online capacity can only be first-come")
        self.config = config
        self._day: int | None = None
        self._reviews = 0

    def decide(self, p: float, amount: float, step: int) -> tuple[int, str, np.ndarray | None]:
        batch = decide(
            np.array([p]),
            np.array([amount]),
            np.array([step]),
            PolicyConfig(
                costs=self.config.costs,
                conformal=self.config.conformal,
                uncertain_action=self.config.uncertain_action,
            ),
        )
        action, reason = int(batch.actions[0]), str(batch.reasons[0])
        capacity = self.config.review_capacity_per_day
        if action == REVIEW and capacity is not None:
            day = (step - 1) // STEPS_PER_DAY
            if day != self._day:
                self._day, self._reviews = day, 0
            if self._reviews >= capacity:
                action = int(_cheapest_non_review(batch.expected)[0])
                reason = "capacity_overflow"
            else:
                self._reviews += 1
        sets = None if batch.sets is None else batch.sets[0]
        return action, reason, sets


__all__ = [
    "APPROVE",
    "BLOCK",
    "REVIEW",
    "STEP_UP",
    "Decisions",
    "OnlinePolicy",
    "PolicyConfig",
    "decide",
]
