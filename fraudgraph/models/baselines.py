"""Phase 1 baselines (task 1.3): B-rule, B-lr, B-gbm, each with and without hour.

Model selection only ever looks at `train_val`: models are fitted on `train_fit`,
LightGBM early-stops and is tuned on `train_val`, and calib/test are only scored.
Proper metrics with confidence intervals come from the evaluation harness (1.4).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

HOUR_FEATURE = "t_hour"
KEY_COLUMNS = ("txn_id", "step", "split", "is_fraud")


@dataclass
class BaselineResult:
    name: str
    features: list[str]
    val_ap: float
    params: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    search: list[dict[str, Any]] = field(default_factory=list)


def feature_sets(table: pl.DataFrame) -> dict[str, list[str]]:
    """The tabular feature set, and the same set without hour (the ablation)."""
    names = [c for c in table.columns if c.startswith(("t_", "o_"))]
    return {"": names, "_no_hour": [c for c in names if c != HOUR_FEATURE]}


def matrix(table: pl.DataFrame, features: list[str], horizon_steps: int) -> np.ndarray:
    """Features as float64, with model-agnostic fills for "no history" NaNs."""
    filled = table.select(features).with_columns(
        [
            pl.col(c).fill_nan(float(horizon_steps + 1))
            for c in features
            if c == "o_steps_since_last"
        ]
        + [pl.col(c).fill_nan(0.0) for c in features if c == "o_amount_z"]
    )
    return filled.to_numpy().astype(np.float64)


def labels(table: pl.DataFrame) -> np.ndarray:
    return table["is_fraud"].cast(pl.Int8).to_numpy()


def ap(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score))


# ----------------------------------------------------------------- B-rule


def rule_scores(table: pl.DataFrame) -> np.ndarray:
    """1 when the destination is new (no incoming transaction in the horizon), else 0.
    The type condition holds by construction: the table only has TRANSFER/CASH_OUT."""
    return table["dest_is_new"].cast(pl.Float64).to_numpy()


# ----------------------------------------------------------------- B-lr


def fit_logreg(x: np.ndarray, y: np.ndarray, seed: int) -> Pipeline:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=seed),
    )
    model.fit(x, y)
    return model


# ----------------------------------------------------------------- B-gbm

LGBM_BASE: dict[str, Any] = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
    "num_threads": 0,
}
SEARCH_SPACE: dict[str, list[Any]] = {
    "num_leaves": [15, 31, 63, 127],
    "min_data_in_leaf": [20, 50, 100, 200, 500],
    "feature_fraction": [0.7, 0.85, 1.0],
    "bagging_fraction": [0.7, 0.85, 1.0],
    "lambda_l2": [0.0, 1.0, 10.0],
    "weighting": ["none", "sqrt", "balanced"],  # scale_pos_weight: 1, sqrt(neg/pos), neg/pos
}
DEFAULT_TRIAL = {
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "lambda_l2": 0.0,
    "weighting": "none",
}


def search_trials(trials: int, seed: int) -> list[dict[str, Any]]:
    """Trial 0 is a sensible default; the rest are a seeded random sample of the space."""
    rng = np.random.default_rng(seed)
    out = [dict(DEFAULT_TRIAL)]
    seen = {tuple(sorted(DEFAULT_TRIAL.items()))}
    while len(out) < trials:
        trial = {k: v[int(rng.integers(len(v)))] for k, v in SEARCH_SPACE.items()}
        key = tuple(sorted(trial.items()))
        if key not in seen:
            seen.add(key)
            out.append(trial)
    return out


def fit_lgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    *,
    trials: int,
    seed: int,
    max_rounds: int = 2_000,
    early_stopping: int = 100,
) -> tuple[lgb.Booster, BaselineResult]:
    """Random search on train_val AP; each trial early-stops on train_val."""
    pos = max(int(y_train.sum()), 1)
    ratio = (len(y_train) - pos) / pos
    weights = {"none": 1.0, "sqrt": math.sqrt(ratio), "balanced": ratio}
    train_set = lgb.Dataset(
        x_train, y_train, feature_name=feature_names, params={"feature_pre_filter": False}
    )
    val_set = lgb.Dataset(x_val, y_val, reference=train_set, params={"feature_pre_filter": False})

    best: tuple[float, lgb.Booster, dict[str, Any]] | None = None
    log = []
    for i, trial in enumerate(search_trials(trials, seed)):
        params = {
            **LGBM_BASE,
            **{k: v for k, v in trial.items() if k != "weighting"},
            "bagging_freq": 1 if trial["bagging_fraction"] < 1 else 0,
            "scale_pos_weight": weights[trial["weighting"]],
            "seed": seed + i,
        }
        started = time.perf_counter()
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=max_rounds,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
        )
        score = ap(y_val, np.asarray(booster.predict(x_val, num_iteration=booster.best_iteration)))
        log.append(
            {
                **trial,
                "val_ap": score,
                "rounds": booster.best_iteration,
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
        if best is None or score > best[0]:
            best = (score, booster, {**params, "num_boost_round": booster.best_iteration})

    assert best is not None
    score, booster, params = best
    result = BaselineResult(name="", features=feature_names, val_ap=score, params=params)
    result.search = log
    return booster, result
