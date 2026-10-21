from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pytest

from fraudgraph.config import REPO_ROOT, SplitSettings
from fraudgraph.data.prepare import to_canonical
from fraudgraph.data.splits import assign_split, plan_splits
from fraudgraph.data.synth import SynthConfig, generate
from fraudgraph.explain import quality, stability
from fraudgraph.explain.reasons import Group, ReasonCodes, Variant, contributions, gap_after_top_k
from fraudgraph.features.backfill import backfill
from fraudgraph.features.spec import FeatureSpec

REASONS = REPO_ROOT / "configs" / "reason_codes.yaml"


@pytest.fixture(scope="module")
def toy() -> tuple[lgb.Booster, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4_000, 4))
    y = (2 * x[:, 0] + x[:, 1] + rng.normal(0, 0.5, 4_000) > 1.5).astype(int)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "seed": 0},
        lgb.Dataset(x, y, feature_name=["a", "b", "c", "d"]),
        num_boost_round=60,
    )
    return booster, x, y


def test_pred_contrib_equals_shap_tree_explainer(
    toy: tuple[lgb.Booster, np.ndarray, np.ndarray],
) -> None:
    shap = pytest.importorskip("shap")
    booster, x, _ = toy
    ours, bias = contributions(booster, x[:300])
    reference = np.asarray(shap.TreeExplainer(booster).shap_values(x[:300]))
    assert np.max(np.abs(ours - reference)) < 1e-6
    margin = booster.predict(x[:300], raw_score=True)
    assert np.allclose(ours.sum(axis=1) + bias, margin)  # additivity


def codes_for(features: list[str]) -> ReasonCodes:
    groups = [
        Group(
            "ONE",
            "One",
            ("a", "b"),
            (Variant("a is {a:.1f}", ("a", 9.0)), Variant("b is {b:.0f}", None)),
        ),
        Group("TWO", "Two", ("c",), (Variant("c={c:.0f}", None),)),
        Group("THREE", "Three", ("d",), (Variant("missing {nope}", None),)),
    ]
    return ReasonCodes(groups, features)


def test_grouping_sums_feature_attributions() -> None:
    codes = codes_for(["a", "b", "c", "d"])
    grouped = codes.grouped(np.array([[1.0, 2.0, -0.5, 0.25]]))
    assert grouped.tolist() == [[3.0, -0.5, 0.25]]


def test_reasons_are_risk_increasing_and_templated() -> None:
    codes = codes_for(["a", "b", "c", "d"])
    values = {"a": 1.0, "b": 7.0, "c": 3.0, "d": 0.0}
    reasons = codes.reasons(np.array([3.0, -0.5, 0.25]), values)
    assert [r.code for r in reasons] == ["ONE", "THREE"]  # TWO lowers risk: not a reason
    assert reasons[0].text == "b is 7"  # the `when a == 9` variant does not match
    assert reasons[1].text == "Three"  # unknown placeholder falls back to the label


def test_when_variant_matches() -> None:
    codes = codes_for(["a", "b", "c", "d"])
    (reason,) = codes.reasons(
        np.array([1.0, 0.0, 0.0]), {"a": 9.0, "b": 0.0, "c": 0.0, "d": 0.0}, k=1
    )
    assert reason.text == "a is 9.0"


def test_every_model_feature_needs_a_group() -> None:
    with pytest.raises(ValueError, match="without a reason group"):
        codes_for(["a", "b", "c", "d", "e"])


def test_shipped_reason_codes_cover_all_engine_features() -> None:
    from fraudgraph.features.engine import FeatureEngine
    from fraudgraph.features.state import InMemoryStore

    names = list(FeatureEngine(FeatureSpec(), InMemoryStore(168)).feature_names)
    codes = ReasonCodes.load(REASONS, names)
    assert len(codes.codes) == 8


def test_gap_and_jaccard_helpers() -> None:
    g = np.array([[5.0, -4.0, 3.0, 2.9], [1.0, 1.0, 1.0, 1.0]])
    assert gap_after_top_k(g).tolist() == pytest.approx([0.1, 0.0])
    a, _ = stability.top_k_sets(g)
    assert a == frozenset({0, 1, 2})
    assert stability.jaccard(frozenset({0, 1, 2}), frozenset({0, 1, 3})) == pytest.approx(0.5)


def test_stability_threshold_separates_fragile_rows() -> None:
    gap = np.r_[np.full(50, 0.01), np.full(50, 1.0)]
    jac = np.r_[np.full(50, 0.4), np.full(50, 1.0)]
    out = stability.stability_threshold(gap, jac)
    assert out["balanced_accuracy"] == pytest.approx(1.0)
    assert 0.01 < out["threshold"] <= 1.0


def test_seed_stability_is_high_for_a_clear_signal(
    toy: tuple[lgb.Booster, np.ndarray, np.ndarray],
) -> None:
    booster, x, y = toy
    codes = codes_for(["a", "b", "c", "d"])
    models = stability.refit_models(booster, x, y, n_models=3)
    out = stability.seed_stability(models, x[:200], codes)
    assert out["median_jaccard"] == pytest.approx(1.0)  # three groups, top-3 is all of them
    assert out["median_tau"] > 0.3


def test_shap_beats_random_on_faithfulness(toy: tuple[lgb.Booster, np.ndarray, np.ndarray]) -> None:
    booster, x, _ = toy
    rows = x[:200]
    mean = x.mean(axis=0)
    shap_f = quality.faithfulness_correlation(
        booster, rows, contributions(booster, rows)[0], mean, subset=2
    )
    rand_f = quality.faithfulness_correlation(
        booster, rows, np.random.default_rng(1).normal(size=rows.shape), mean, subset=2
    )
    assert np.nanmean(shap_f) > 0.6
    assert abs(np.nanmean(rand_f)) < 0.15


def test_bootstrap_ci_brackets_mean() -> None:
    out = quality.bootstrap_ci(np.arange(100, dtype=float))
    assert out["ci95"][0] <= out["mean"] <= out["ci95"][1]


def test_explain_report_end_to_end_on_fixture(tmp_path: Path) -> None:
    pytest.importorskip("shap")
    pytest.importorskip("quantus")
    from fraudgraph.explain import report as explain_report
    from fraudgraph.pipeline import fixture_bundle

    bundle = fixture_bundle(tmp_path / "b", trials=1)
    booster = lgb.Booster(model_file=str(bundle / "model.txt"))
    calibrator = joblib.load(bundle / "calibrator.joblib")
    _, policy_config = explain_report.policy_from_artifacts(bundle)
    cfg = SynthConfig()
    events = to_canonical(generate(cfg).frame)
    features, _ = backfill(events, FeatureSpec())
    plan = plan_splits(events, SplitSettings(dense_last_step=(cfg.days - cfg.sparse_days) * 24), 12)
    table = assign_split(features, plan).join(
        events.select("txn_id", "amount", "type"), on="txn_id", how="left"
    )
    codes = ReasonCodes.load(REASONS, booster.feature_name())

    report, background = explain_report.run(
        table, booster, codes, calibrator, policy_config,
        horizon_steps=168, n_flagged=150, n_quality=40, n_models=3, resamples=50,
    )  # fmt: skip
    explain_report.save(report, background, REASONS, tmp_path / "a", tmp_path / "r")

    assert report["shap_check"]["max_abs_diff_vs_shap_tree_explainer"] < 1e-6
    assert set(report["quality"]) == {"shap", "random", "permutation"}
    assert {"faithfulness", "monotonicity", "max_sensitivity"} <= set(report["quality"]["shap"])
    assert report["examples"]
    assert json.loads((tmp_path / "a" / "explain.json").read_text())["stability_gap_threshold"] >= 0
