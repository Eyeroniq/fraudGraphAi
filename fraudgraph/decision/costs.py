"""The money-based cost model (A6) and the decisions made from it (task 1.5).

| action  | cost if legit | cost if fraud                   |
|---------|---------------|---------------------------------|
| APPROVE | 0             | a                               |
| REVIEW  | c_review      | c_review + (1 - r) * a          |
| BLOCK   | c_friction    | c_block                         |
| STEP_UP | c_step_up     | c_step_up + (1 - r_step) * a    |

STEP_UP (task 4.2, Juspay track J3) challenges the customer (OTP / 3DS) instead of
sending the payment to a human: much cheaper for a legit customer than a review, but
it stops less fraud. It is off unless `step_up_enabled`.

Everything is vectorised over rows: `a` is the transaction amount.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

ACTIONS = ("APPROVE", "REVIEW", "BLOCK", "STEP_UP")
APPROVE, REVIEW, BLOCK, STEP_UP = 0, 1, 2, 3
# How strongly an action interferes with the payment (for "at least X" overrides).
SEVERITY = {APPROVE: 0, STEP_UP: 1, REVIEW: 2, BLOCK: 3}


@dataclass(frozen=True)
class CostParams:
    c_review: float = 5.0
    c_friction: float = 50.0
    catch_rate: float = 0.9  # r: share of reviewed fraud an analyst stops
    c_block: float = 0.0
    step_up_enabled: bool = False
    c_step_up: float = 200.0  # OTP / 3DS friction and abandonment for a legit customer
    step_up_catch_rate: float = 0.7  # share of fraud a step-up challenge stops

    def enabled_actions(self) -> tuple[int, ...]:
        return (
            (APPROVE, REVIEW, BLOCK, STEP_UP) if self.step_up_enabled else (APPROVE, REVIEW, BLOCK)
        )

    def scaled(self, **factors: float) -> CostParams:
        values = asdict(self)
        for name, factor in factors.items():
            values[name] = values[name] * factor
        return CostParams(**values)


def cost_table(amount: np.ndarray, costs: CostParams) -> tuple[np.ndarray, np.ndarray]:
    """(cost_if_legit, cost_if_fraud), each shaped [rows, 4 actions]."""
    n = len(amount)
    legit = np.empty((n, len(ACTIONS)))
    fraud = np.empty((n, len(ACTIONS)))
    legit[:, APPROVE], fraud[:, APPROVE] = 0.0, amount
    legit[:, REVIEW] = costs.c_review
    fraud[:, REVIEW] = costs.c_review + (1.0 - costs.catch_rate) * amount
    legit[:, BLOCK], fraud[:, BLOCK] = costs.c_friction, costs.c_block
    legit[:, STEP_UP] = costs.c_step_up
    fraud[:, STEP_UP] = costs.c_step_up + (1.0 - costs.step_up_catch_rate) * amount
    return legit, fraud


def expected_costs(p: np.ndarray, amount: np.ndarray, costs: CostParams) -> np.ndarray:
    """E[cost] per action, shape [rows, 4]; disabled actions are +inf (never chosen)."""
    legit, fraud = cost_table(amount, costs)
    expected = np.asarray((1.0 - p)[:, None] * legit + p[:, None] * fraud)
    if not costs.step_up_enabled:
        expected[:, STEP_UP] = np.inf
    return expected


def expected_cost_decisions(p: np.ndarray, amount: np.ndarray, costs: CostParams) -> np.ndarray:
    """Per-transaction rule: the action with the lowest expected cost (ties -> cheaper index)."""
    return np.asarray(np.argmin(expected_costs(p, amount, costs), axis=1))


def realised_costs(
    actions: np.ndarray, y: np.ndarray, amount: np.ndarray, costs: CostParams
) -> np.ndarray:
    """What each decision actually cost, given the true label."""
    legit, fraud = cost_table(amount, costs)
    rows = np.arange(len(actions))
    return np.asarray(np.where(y.astype(bool), fraud[rows, actions], legit[rows, actions]))


def threshold_decisions(score: np.ndarray, threshold: float) -> np.ndarray:
    """The classic single global threshold: alert (REVIEW) at or above it, else APPROVE."""
    return np.where(score >= threshold, REVIEW, APPROVE)


def best_global_threshold(
    score: np.ndarray, y: np.ndarray, amount: np.ndarray, costs: CostParams
) -> tuple[float, float]:
    """Threshold minimising total realised cost (alert -> REVIEW). Returns (t, cost).

    Sweeps every distinct score: flagging the top-k rows costs
    sum(fraud amounts not flagged) + c_review * k + (1 - r) * sum(flagged fraud amounts).
    """
    order = np.argsort(-score, kind="stable")
    s, yy, aa = score[order], y[order].astype(bool), amount[order]
    fraud_amount = np.where(yy, aa, 0.0)
    total_fraud = fraud_amount.sum()
    flagged_fraud = np.cumsum(fraud_amount)
    k = np.arange(1, len(s) + 1)
    cost_flag_k = (
        (total_fraud - flagged_fraud)
        + costs.c_review * k
        + (1.0 - costs.catch_rate) * flagged_fraud
    )
    # Only cut between distinct scores (ties are flagged together).
    last_of_group = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    candidates = cost_flag_k[last_of_group]
    best = int(np.argmin(candidates))
    if total_fraud <= candidates[best]:  # flagging nothing is cheapest
        return float(np.inf), float(total_fraud)
    return float(s[last_of_group[best]]), float(candidates[best])


def prior_shift(p: np.ndarray, prior_from: float, prior_to: float) -> np.ndarray:
    """Re-weight calibrated probabilities from one fraud prevalence to another."""
    odds_ratio = (prior_to / prior_from) / ((1.0 - prior_to) / (1.0 - prior_from))
    odds = p / np.clip(1.0 - p, 1e-12, None) * odds_ratio
    return np.asarray(odds / (1.0 + odds))
