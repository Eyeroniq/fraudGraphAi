"""Assemble the training table and run the Phase 1 baselines (task 1.3)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from fraudgraph.data.splits import SplitPlan, assign_split
from fraudgraph.features.novelty import destination_novelty
from fraudgraph.models.baselines import (
    KEY_COLUMNS,
    BaselineResult,
    ap,
    feature_sets,
    fit_lgbm,
    fit_logreg,
    labels,
    matrix,
    rule_scores,
)

SCORED_SPLITS = ("train_val", "calib_a", "calib_b", "test", "late_test")


def training_table(
    events: pl.DataFrame, features: pl.DataFrame, plan: SplitPlan, horizon_steps: int
) -> pl.DataFrame:
    """Feature rows + split + `dest_is_new`; embargoed rows are dropped."""
    novelty = destination_novelty(events, horizon_steps)
    return (
        assign_split(features, plan)
        .filter(pl.col("split").is_not_null())
        .join(novelty, on="txn_id", how="left", maintain_order="left")
    )


def run_baselines(
    table: pl.DataFrame, *, horizon_steps: int, trials: int, seed: int
) -> tuple[list[BaselineResult], pl.DataFrame, dict[str, Any]]:
    """Fit every baseline on train_fit, select on train_val, score all later splits.

    Returns the results, a predictions frame (one score column per model) and the
    fitted model objects keyed by name.
    """
    train = table.filter(pl.col("split") == "train_fit")
    val = table.filter(pl.col("split") == "train_val")
    scored = table.filter(pl.col("split").is_in(SCORED_SPLITS))
    y_train, y_val = labels(train), labels(val)

    results: list[BaselineResult] = []
    scores: dict[str, np.ndarray] = {}
    models: dict[str, Any] = {}

    started = time.perf_counter()
    scores["rule_new_dest"] = rule_scores(scored)
    results.append(
        BaselineResult(
            name="rule_new_dest",
            features=["dest_is_new"],
            val_ap=ap(y_val, rule_scores(val)),
            seconds=round(time.perf_counter() - started, 1),
        )
    )

    for suffix, features in feature_sets(table).items():
        x_train, x_val = (matrix(t, features, horizon_steps) for t in (train, val))
        x_scored = matrix(scored, features, horizon_steps)

        started = time.perf_counter()
        logreg = fit_logreg(x_train, y_train, seed)
        name = f"logreg{suffix}"
        models[name] = logreg
        scores[name] = logreg.predict_proba(x_scored)[:, 1]
        results.append(
            BaselineResult(
                name=name,
                features=features,
                val_ap=ap(y_val, logreg.predict_proba(x_val)[:, 1]),
                params={"class_weight": "balanced", "scaler": "standard"},
                seconds=round(time.perf_counter() - started, 1),
            )
        )

        started = time.perf_counter()
        booster, result = fit_lgbm(
            x_train, y_train, x_val, y_val, features, trials=trials, seed=seed
        )
        name = f"lgbm{suffix}"
        result.name = name
        result.seconds = round(time.perf_counter() - started, 1)
        models[name] = booster
        scores[name] = np.asarray(booster.predict(x_scored, num_iteration=booster.best_iteration))
        results.append(result)

    predictions = scored.select(KEY_COLUMNS).with_columns(
        [pl.Series(f"score_{k}", v, dtype=pl.Float64) for k, v in scores.items()]
    )
    return results, predictions, models


def preview_ap(
    predictions: pl.DataFrame, results: list[BaselineResult]
) -> dict[str, dict[str, float]]:
    """Plain AP per split, for the PR description only (1.4 adds CIs)."""
    out: dict[str, dict[str, float]] = {}
    y_by_split = {
        s: predictions.filter(pl.col("split") == s) for s in ("train_val", "test", "late_test")
    }
    for r in results:
        out[r.name] = {
            s: ap(labels(part), part[f"score_{r.name}"].to_numpy())
            for s, part in y_by_split.items()
        }
    return out


def save(
    out_dir: Path,
    results: list[BaselineResult],
    predictions: pl.DataFrame,
    models: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        if name.startswith("lgbm"):
            model.save_model(str(out_dir / f"{name}.txt"), num_iteration=model.best_iteration)
        else:
            joblib.dump(model, out_dir / f"{name}.joblib")
    predictions.write_parquet(out_dir / "predictions.parquet", compression="zstd")
    summary = {
        **meta,
        "preview_ap": preview_ap(predictions, results),
        "results": [asdict(r) for r in results],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


CHAMPION_EXCLUDED = ("g_dest_known_fraud_in",)  # needs online labels (7.3), see graph_study.md


def champion_features(table: pl.DataFrame) -> list[str]:
    """T+O+G1 minus the label-dependent G1-b feature: the 3.6 winner that is servable now."""
    return [
        c for c in table.columns if c.startswith(("t_", "o_", "g_")) and c not in CHAMPION_EXCLUDED
    ]


def train_champion(
    table: pl.DataFrame, *, horizon_steps: int, trials: int, seed: int
) -> tuple[BaselineResult, np.ndarray, Any]:
    """Fit the champion on train_fit, select on train_val; scores for SCORED_SPLITS rows."""
    features = champion_features(table)
    train = table.filter(pl.col("split") == "train_fit")
    val = table.filter(pl.col("split") == "train_val")
    scored = table.filter(pl.col("split").is_in(SCORED_SPLITS))
    started = time.perf_counter()
    booster, result = fit_lgbm(
        matrix(train, features, horizon_steps),
        labels(train),
        matrix(val, features, horizon_steps),
        labels(val),
        features,
        trials=trials,
        seed=seed,
    )
    result.name = "lgbm_g1"
    result.seconds = round(time.perf_counter() - started, 1)
    scores = np.asarray(
        booster.predict(
            matrix(scored, features, horizon_steps), num_iteration=booster.best_iteration
        )
    )
    return result, scores, booster
