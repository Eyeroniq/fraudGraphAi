from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fraudgraph.decision.costs import (
    APPROVE,
    BLOCK,
    REVIEW,
    CostParams,
    best_global_threshold,
    expected_cost_decisions,
    prior_shift,
    realised_costs,
    threshold_decisions,
)
from fraudgraph.decision.fit_policy import fit_policy
from fraudgraph.models.calibrate import expected_calibration_error, fit_calibrator

COSTS = CostParams(c_review=10.0, c_friction=100.0, catch_rate=0.9, c_block=0.0)


def test_expected_cost_rule_by_hand() -> None:
    p = np.array([0.0001, 0.05, 0.95])
    amount = np.array([1_000.0, 1_000.0, 1_000.0])
    # p=0.0001: APPROVE 0.1, REVIEW 10.01, BLOCK ~100   -> APPROVE
    # p=0.05:   APPROVE 50,  REVIEW 15,    BLOCK 95     -> REVIEW
    # p=0.95:   APPROVE 950, REVIEW 105,   BLOCK 5      -> BLOCK
    assert expected_cost_decisions(p, amount, COSTS).tolist() == [APPROVE, REVIEW, BLOCK]


def test_expected_cost_threshold_depends_on_amount() -> None:
    p = np.array([0.02, 0.02])
    amount = np.array([100.0, 100_000.0])
    # Same p: small amount -> APPROVE (E=2); large amount -> BLOCK (E=98 < REVIEW 210).
    assert expected_cost_decisions(p, amount, COSTS).tolist() == [APPROVE, BLOCK]


def test_realised_costs_by_hand() -> None:
    actions = np.array([APPROVE, APPROVE, REVIEW, REVIEW, BLOCK, BLOCK])
    y = np.array([0, 1, 0, 1, 0, 1])
    amount = np.full(6, 1_000.0)
    assert realised_costs(actions, y, amount, COSTS) == pytest.approx([0, 1000, 10, 110, 100, 0])


def test_global_threshold_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.1).astype(np.int8)
    score = np.round(rng.random(400) * 0.5 + y * 0.3, 3)
    amount = rng.lognormal(6, 1, 400)
    t, cost = best_global_threshold(score, y, amount, COSTS)

    brute = min(
        realised_costs(threshold_decisions(score, c), y, amount, COSTS).sum()
        for c in np.r_[np.unique(score), np.inf]
    )
    assert cost == pytest.approx(brute)
    assert realised_costs(threshold_decisions(score, t), y, amount, COSTS).sum() == pytest.approx(
        cost
    )


def test_prior_shift_identity_and_direction() -> None:
    p = np.array([0.001, 0.1, 0.5])
    assert prior_shift(p, 0.01, 0.01) == pytest.approx(p)
    assert (prior_shift(p, 0.01, 0.05) > p).all()


def test_isotonic_calibrator_is_monotone_and_bounded() -> None:
    rng = np.random.default_rng(1)
    score = rng.random(5_000)
    y = (rng.random(5_000) < score**3).astype(np.int8)
    cal = fit_calibrator(score, y, "isotonic")
    p = cal(np.sort(score))

    assert (np.diff(p) >= -1e-12).all()
    assert p.min() >= 0
    assert p.max() <= 1
    assert expected_calibration_error(y, cal(score)) < 0.02


def test_platt_is_available_as_comparison() -> None:
    rng = np.random.default_rng(2)
    score = rng.random(2_000)
    y = (rng.random(2_000) < score).astype(np.int8)
    assert fit_calibrator(score, y, "platt")(np.array([0.1, 0.9])).tolist() == sorted(
        fit_calibrator(score, y, "platt")(np.array([0.1, 0.9])).tolist()
    )


def synthetic_predictions(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for split, (start, end) in {
        "calib_a": (1, 48),
        "calib_b": (49, 96),
        "test": (97, 144),
        "late_test": (145, 192),
    }.items():
        n = 3_000
        y = rng.random(n) < 0.02
        rows.append(
            pl.DataFrame(
                {
                    "txn_id": [f"{split}{i}" for i in range(n)],
                    "step": rng.integers(start, end + 1, n),
                    "split": split,
                    "is_fraud": y,
                    "score_m": np.clip(rng.normal(0.1 + 0.5 * y, 0.15), 0, 1),
                    "amount": rng.lognormal(8, 1, n),
                }
            )
        )
    return pl.concat(rows)


def test_fit_policy_outputs() -> None:
    _, policy, report = fit_policy(
        synthetic_predictions(), "m", COSTS, block_steps=6, resamples=100, seed=0
    )

    assert policy["rule"] == "expected_cost"
    assert np.isfinite(policy["global_threshold"])
    rules = report["test"]
    assert set(rules) == {
        "approve_all",
        "global_threshold",
        "expected_cost",
        "expected_cost_prior_shift",
    }
    for r in rules.values():
        lo, hi = r["cost"]["ci95"]
        assert lo <= r["cost"]["total"] <= hi
    assert rules["expected_cost"]["cost"]["total"] < rules["approve_all"]["cost"]["total"]
    assert {row["parameter"] for row in report["sensitivity"]} == {
        "c_review",
        "c_friction",
        "catch_rate",
    }
