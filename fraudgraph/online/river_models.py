"""Online learners on the same engine features as the batch models (task 6.3).

Candidates: logistic regression (+ standard scaling) and a Hoeffding adaptive tree, both
behind random undersampling of legit rows (fraud is ~0.2 % of traffic). They are warm-started
on train_fit, then learn test-then-train from delayed labels as the stream advances.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fraudgraph.features.matrix import fill_values

CANDIDATES = ("logreg", "hoeffding_tree")


def make_model(name: str, seed: int = 0) -> Any:
    from river import compose, imblearn, linear_model, optim, preprocessing, tree

    base: Any
    if name == "logreg":
        base = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression(optimizer=optim.SGD(0.01), l2=1e-4),
        )
    elif name == "hoeffding_tree":
        base = tree.HoeffdingAdaptiveTreeClassifier(grace_period=200, seed=seed)
    else:
        raise ValueError(f"unknown river model {name!r}")
    return imblearn.RandomUnderSampler(base, desired_dist={False: 0.9, True: 0.1}, seed=seed)


def row_dicts(x: np.ndarray, features: list[str], horizon_steps: int) -> list[dict[str, float]]:
    """Matrix rows -> river feature dicts (NaN "no history" values get the shared fills)."""
    fills = fill_values(horizon_steps)
    out = []
    for row in x:
        d = {}
        for name, value in zip(features, row, strict=True):
            v = float(value)
            d[name] = fills.get(name, 0.0) if math.isnan(v) else v
        out.append(d)
    return out


def predict(model: Any, rows: list[dict[str, float]]) -> np.ndarray:
    return np.array([model.predict_proba_one(r).get(True, 0.0) for r in rows])


def warm_start(model: Any, rows: list[dict[str, float]], y: np.ndarray, seed: int = 0) -> Any:
    """Learn on a shuffled pass over the training rows (the undersampler drops most legit)."""
    order = np.random.default_rng(seed).permutation(len(rows))
    for i in order:
        model.learn_one(rows[i], bool(y[i]))
    return model
