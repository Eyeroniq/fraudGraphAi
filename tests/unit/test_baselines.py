from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from fraudgraph.config import SplitSettings
from fraudgraph.data.prepare import to_canonical
from fraudgraph.data.splits import plan_splits
from fraudgraph.data.synth import SynthConfig, generate
from fraudgraph.features.backfill import backfill
from fraudgraph.features.novelty import destination_novelty
from fraudgraph.features.spec import FeatureSpec
from fraudgraph.models.baselines import HOUR_FEATURE, search_trials
from fraudgraph.models.train import SCORED_SPLITS, run_baselines, save, training_table

HORIZON = 168


def test_destination_novelty_rules() -> None:
    #           same step   seen     gap = H   gap = H+1
    steps = {"a": 1, "b": 1, "c": 2, "d": 170, "e": 339}
    frame = pl.DataFrame(
        {"txn_id": list(steps), "step": list(steps.values()), "name_dest": ["D1"] * 5}
    )
    new = dict(destination_novelty(frame, HORIZON).iter_rows())

    assert new["a"]
    assert new["b"]
    assert not new["c"]  # received at step 1
    assert not new["d"]  # last seen at step 2: 170 - 2 = 168 = horizon, still remembered
    assert new["e"]  # last seen at step 170: 339 - 170 = 169 > horizon, forgotten


def test_search_trials_are_seeded_and_start_from_default() -> None:
    first, second = search_trials(6, seed=3), search_trials(6, seed=3)

    assert first == second
    assert first[0]["weighting"] == "none"
    assert len({tuple(sorted(t.items())) for t in first}) == 6


@pytest.fixture(scope="module")
def table() -> pl.DataFrame:
    cfg = SynthConfig()
    events = to_canonical(generate(cfg).frame)
    features, _ = backfill(events, FeatureSpec())
    plan = plan_splits(events, SplitSettings(dense_last_step=(cfg.days - cfg.sparse_days) * 24), 12)
    return training_table(events, features, plan, HORIZON)


@pytest.fixture(scope="module")
def run(table: pl.DataFrame):  # type: ignore[no-untyped-def]
    return run_baselines(table, horizon_steps=HORIZON, trials=3, seed=0)


def test_all_baselines_with_and_without_hour(run) -> None:  # type: ignore[no-untyped-def]
    results, _, models = run
    names = [r.name for r in results]

    assert names == ["rule_new_dest", "logreg", "lgbm", "logreg_no_hour", "lgbm_no_hour"]
    by_name = {r.name: r for r in results}
    assert HOUR_FEATURE in by_name["lgbm"].features
    assert HOUR_FEATURE not in by_name["lgbm_no_hour"].features
    assert set(models) == {"logreg", "lgbm", "logreg_no_hour", "lgbm_no_hour"}


def test_lgbm_selects_the_best_trial_on_train_val(run) -> None:  # type: ignore[no-untyped-def]
    results, _, _ = run
    lgbm = next(r for r in results if r.name == "lgbm")

    assert len(lgbm.search) == 3
    assert lgbm.val_ap == max(t["val_ap"] for t in lgbm.search)
    assert lgbm.params["num_boost_round"] >= 1


def test_predictions_cover_only_scored_splits(run, table: pl.DataFrame) -> None:  # type: ignore[no-untyped-def]
    _, predictions, _ = run

    assert set(predictions["split"].unique()) == set(SCORED_SPLITS)
    assert predictions.height == table.filter(pl.col("split") != "train_fit").height
    score_columns = [c for c in predictions.columns if c.startswith("score_")]
    assert len(score_columns) == 5
    assert predictions.select(score_columns).null_count().sum_horizontal().item() == 0


def test_new_destination_rule_beats_prevalence_on_fixture(run) -> None:  # type: ignore[no-untyped-def]
    results, predictions, _ = run
    val = predictions.filter(pl.col("split") == "train_val")
    prevalence = val["is_fraud"].mean()
    rule = next(r for r in results if r.name == "rule_new_dest")

    assert rule.val_ap > 2 * prevalence  # type: ignore[operator]


def test_save_writes_models_predictions_and_summary(run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    results, predictions, models = run
    save(tmp_path, results, predictions, models, {"trials": 3})

    assert (tmp_path / "lgbm.txt").is_file()
    assert (tmp_path / "logreg_no_hour.joblib").is_file()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["preview_ap"]) == {r.name for r in results}
    assert pl.read_parquet(tmp_path / "predictions.parquet").height == predictions.height
