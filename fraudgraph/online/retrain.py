"""LightGBM retraining shared by the drift study (6.6) and the worker (7.4).

The policy from `docs/drift_study.md`: retrain the champion's model class every day on
the labels known so far from the last week, with legit rows subsampled and up-weighted.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np

KEEP_PARAMS = (
    "objective", "num_leaves", "min_data_in_leaf", "feature_fraction", "bagging_fraction",
    "bagging_freq", "lambda_l2", "scale_pos_weight", "learning_rate",
)  # fmt: skip


def lgbm_params(booster: lgb.Booster) -> dict[str, Any]:
    """The champion's hyper-parameters, made deterministic and quiet."""
    return {k: v for k, v in booster.params.items() if k in KEEP_PARAMS} | {
        "verbosity": -1,
        "deterministic": True,
        "force_row_wise": True,
    }


def fit_like(
    booster: lgb.Booster,
    x: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    *,
    legit_keep: float,
    rng: np.random.Generator,
) -> lgb.Booster:
    """Same parameters and number of trees as `booster`, on `rows` of (x, y): every fraud,
    a `legit_keep` share of legit rows weighted 1 / legit_keep."""
    keep = rows & (y | (rng.random(len(y)) < legit_keep))
    weight = np.where(y[keep], 1.0, 1.0 / legit_keep)
    data = lgb.Dataset(
        x[keep], y[keep].astype(int), weight=weight, feature_name=booster.feature_name()
    )
    params = lgbm_params(booster) | {"seed": int(rng.integers(1 << 30))}
    return lgb.train(params, data, num_boost_round=booster.current_iteration())
