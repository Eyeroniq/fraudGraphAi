from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fraudgraph.decision import conformal
from fraudgraph.decision.conformal import ConformalModel, conformal_quantile, coverage, set_labels
from fraudgraph.decision.costs import (
    APPROVE,
    BLOCK,
    REVIEW,
    STEP_UP,
    CostParams,
    expected_cost_decisions,
)
from fraudgraph.decision.policy import OnlinePolicy, PolicyConfig, decide

COSTS = CostParams(
    c_review=10.0, c_friction=100.0, step_up_enabled=True, c_step_up=2.0, step_up_catch_rate=0.7
)


# ------------------------------------------------------------------ conformal


def test_finite_sample_quantile() -> None:
    scores = np.arange(1, 10, dtype=float)  # 9 rows
    assert conformal_quantile(scores, 0.1) == 9.0  # ceil(10 * 0.9) = 9th smallest
    assert conformal_quantile(scores, 0.5) == 5.0  # ceil(10 * 0.5) = 5th
    assert math.isinf(conformal_quantile(scores[:5], 0.1))  # too few rows


def calibrated_sample(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.beta(0.3, 8.0, n)  # mostly small probabilities, like fraud
    return p, (rng.random(n) < p).astype(int)


@pytest.mark.parametrize("alpha", [0.05, 0.1])
def test_mondrian_covers_each_class(alpha: float) -> None:
    p, y = calibrated_sample(200_000, 0)
    model = conformal.fit(p[:100_000], y[:100_000], alpha=alpha, mondrian=True)
    cov = coverage(model.sets(p[100_000:]), y[100_000:])
    assert cov["fraud"] >= 1 - alpha - 0.01
    assert cov["legit"] >= 1 - alpha - 0.01


def test_marginal_coverage_can_fail_the_rare_class() -> None:
    """Why Mondrian (D7): marginal coverage holds overall but not for fraud."""
    p, y = calibrated_sample(200_000, 1)
    marginal = conformal.fit(p[:100_000], y[:100_000], alpha=0.1, mondrian=False)
    cov = coverage(marginal.sets(p[100_000:]), y[100_000:])
    assert cov["overall"] >= 0.89
    assert cov["fraud"] < 0.8


def test_aps_scores_by_hand() -> None:
    scores = conformal.nonconformity(np.array([[0.8, 0.2], [0.3, 0.7]]), "aps")
    assert scores.tolist() == [[0.8, 1.0], [1.0, 0.7]]


def test_set_labels_and_json_round_trip(tmp_path: Path) -> None:
    sets = np.array([[True, False], [False, True], [True, True], [False, False]])
    assert set_labels(sets).tolist() == ["legit", "fraud", "uncertain", "empty"]

    model = ConformalModel(0.05, "lac", True, (0.3, math.inf), (100, 3))
    conformal.save(model, tmp_path / "c.json")
    assert conformal.load(tmp_path / "c.json") == model


def test_marginal_matches_mapie() -> None:
    """Our marginal LAC sets equal MAPIE's SplitConformalClassifier on the same data."""
    pytest.importorskip("mapie")
    from mapie.classification import SplitConformalClassifier
    from sklearn.base import BaseEstimator, ClassifierMixin

    class Fixed(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
        """Returns stored probabilities; X is the row index."""

        def __init__(self, p: np.ndarray | None = None) -> None:
            self.p = p

        def fit(self, x: np.ndarray, y: np.ndarray) -> Fixed:
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            assert self.p is not None
            return conformal.class_probabilities(self.p[x[:, 0].astype(int)])

        def predict(self, x: np.ndarray) -> np.ndarray:
            return self.predict_proba(x).argmax(axis=1)

    p, y = calibrated_sample(20_000, 2)
    idx = np.arange(len(p))[:, None]
    est = Fixed(p).fit(idx, y)
    mapie = SplitConformalClassifier(
        estimator=est, confidence_level=0.9, conformity_score="lac", prefit=True
    )
    mapie.conformalize(idx[:10_000], y[:10_000])
    _, mapie_sets = mapie.predict_set(idx[10_000:])

    ours = conformal.fit(p[:10_000], y[:10_000], alpha=0.1, mondrian=False).sets(p[10_000:])
    agreement = (ours == mapie_sets[:, :, 0]).all(axis=1).mean()
    assert agreement > 0.999  # equal up to tie-handling at the quantile


# ------------------------------------------------------------------ policy


def test_step_up_only_when_enabled() -> None:
    p, amount = np.array([0.02]), np.array([1_000.0])
    # E: APPROVE 20, REVIEW 12, BLOCK 98, STEP_UP 2 + 0.02*0.3*1000 = 8
    assert expected_cost_decisions(p, amount, COSTS).tolist() == [STEP_UP]
    no_step = CostParams(c_review=10.0, c_friction=100.0)
    assert expected_cost_decisions(p, amount, no_step).tolist() == [REVIEW]


def model_with(q_legit: float, q_fraud: float) -> ConformalModel:
    return ConformalModel(0.1, "lac", True, (q_legit, q_fraud), (1000, 100))


def test_conformal_override_branches() -> None:
    # LAC: legit in set iff p <= q_legit ; fraud in set iff 1 - p <= q_fraud
    model = model_with(q_legit=0.5, q_fraud=0.6)
    p = np.array([0.001, 0.45, 0.55, 0.999])  # legit-only, uncertain, fraud-only, fraud-only
    amount = np.array([1.0, 1.0, 1.0, 1e9])
    cfg = PolicyConfig(costs=COSTS, conformal=model, uncertain_action=REVIEW)
    out = decide(p, amount, np.ones(4, dtype=int), cfg)

    assert set_labels(out.sets).tolist() == ["legit", "uncertain", "fraud", "fraud"]  # type: ignore[arg-type]
    assert out.actions[0] == APPROVE
    assert (out.actions[1], out.reasons[1]) == (REVIEW, "model_uncertain")
    assert (out.actions[2], out.reasons[2]) == (REVIEW, "conformal_fraud")
    assert out.actions[3] == BLOCK  # never softened to REVIEW
    assert out.reasons[3] == "expected_cost"


def test_empty_set_counts_as_uncertain_and_can_step_up() -> None:
    model = model_with(q_legit=0.1, q_fraud=0.1)  # p = 0.5 fits neither class
    cfg = PolicyConfig(costs=COSTS, conformal=model, uncertain_action=STEP_UP)
    out = decide(np.array([0.5]), np.array([1.0]), np.array([1]), cfg)
    assert set_labels(out.sets).tolist() == ["empty"]  # type: ignore[arg-type]
    assert (out.actions[0], out.reasons[0]) == (STEP_UP, "model_uncertain")


def test_step_up_uncertain_action_requires_step_up_costs() -> None:
    with pytest.raises(ValueError, match="step_up_enabled"):
        PolicyConfig(costs=CostParams(), uncertain_action=STEP_UP)


def capacity_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Four REVIEW-worthy payments in one day, savings increasing with amount.
    costs_no_step = CostParams(c_review=10.0, c_friction=1e9)
    p = np.full(4, 0.05)
    amount = np.array([300.0, 400.0, 2_000.0, 1_000.0])
    assert (expected_cost_decisions(p, amount, costs_no_step) == REVIEW).all()
    return p, amount, np.array([1, 2, 3, 4])


@pytest.mark.parametrize(("mode", "kept"), [("first_come", [0, 1]), ("oracle", [2, 3])])
def test_review_capacity(mode: str, kept: list[int]) -> None:
    p, amount, step = capacity_case()
    cfg = PolicyConfig(
        costs=CostParams(c_review=10.0, c_friction=1e9),
        review_capacity_per_day=2,
        capacity_mode=mode,
    )
    out = decide(p, amount, step, cfg)
    assert np.flatnonzero(out.actions == REVIEW).tolist() == kept
    overflow = out.reasons == "capacity_overflow"
    assert overflow.sum() == 2
    assert (out.actions[overflow] == APPROVE).all()  # cheapest non-review here


def test_capacity_resets_each_day() -> None:
    p, amount, _ = capacity_case()
    cfg = PolicyConfig(costs=CostParams(c_review=10.0, c_friction=1e9), review_capacity_per_day=2)
    out = decide(p, amount, np.array([1, 2, 25, 26]), cfg)  # two days, two reviews each
    assert (out.actions == REVIEW).all()


def test_online_policy_matches_batch_first_come() -> None:
    rng = np.random.default_rng(5)
    n = 300
    p, amount = rng.beta(0.5, 5, n), rng.lognormal(6, 1, n)
    step = np.sort(rng.integers(1, 73, n))
    cfg = PolicyConfig(costs=COSTS, conformal=model_with(0.4, 0.7), review_capacity_per_day=5)
    batch = decide(p, amount, step, cfg)
    online = OnlinePolicy(cfg)
    actions = [
        online.decide(float(a), float(b), int(c))[0]
        for a, b, c in zip(p, amount, step, strict=True)
    ]
    assert actions == batch.actions.tolist()


def test_conformal_report_end_to_end(tmp_path: Path) -> None:
    from fraudgraph.decision import conformal_report
    from tests.unit.test_policy import synthetic_predictions

    costs = CostParams(c_review=2_000.0, c_friction=20_000.0, step_up_enabled=True)
    calibrator, chosen, policy, report = conformal_report.run(
        synthetic_predictions(),
        "m",
        costs,
        alpha=0.1,
        score="lac",
        mondrian=True,
        uncertain_action="REVIEW",
        capacity=50,
        block_steps=6,
        resamples=50,
        seed=0,
    )
    conformal_report.save(tmp_path / "a", tmp_path / "r", calibrator, chosen, policy, report)
    conformal_report.write_plots(report, tmp_path / "r")

    assert len(report["coverage_grid"]) == 2 * 2 * 4
    assert report["done_when"]["target"] == pytest.approx(0.9)
    assert {"global_threshold", "+ capacity 50/day, oracle"} <= set(report["policies_test"])
    capped = report["policies_test"]["+ capacity 50/day, first-come"]
    assert "capacity_overflow" in capped["reason_share"]
    assert conformal.load(tmp_path / "a" / "conformal.json") == chosen
    assert (tmp_path / "r" / "coverage_per_day.png").is_file()
