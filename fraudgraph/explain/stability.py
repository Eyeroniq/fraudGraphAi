"""Explanation stability (task 5.2).

- Seed/bootstrap: refit the champion (same hyper-parameters, same number of trees) on
  bootstrap samples with different seeds; per flagged payment, compare the top-3 reason
  groups (Jaccard) and the full group ranking (Kendall tau) across models.
- Input perturbation: small Gaussian noise (5 % of each feature's std; binary features
  fixed) -> how much the grouped attribution moves relative to the score.
- Background sensitivity: interventional SHAP with three different background samples.
- API flag: a payment's explanation is marked unstable when the gap between its 3rd and
  4th group is small; the threshold is learned here from the seed study.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import lightgbm as lgb
import numpy as np
from scipy.stats import kendalltau

from fraudgraph.explain.reasons import TOP_K, ReasonCodes, contributions, gap_after_top_k

STABLE_JACCARD = 0.8


def top_k_sets(group_shap: np.ndarray, k: int = TOP_K) -> list[frozenset[int]]:
    """Indices of the k groups with the largest |SHAP| per row."""
    order = np.argsort(-np.abs(group_shap), axis=1, kind="stable")[:, :k]
    return [frozenset(row.tolist()) for row in order]


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def refit_models(
    booster: lgb.Booster,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_models: int = 5,
    seed: int = 0,
) -> list[lgb.Booster]:
    """Same hyper-parameters and tree count, bootstrap samples, different seeds."""
    keep = (
        "objective", "num_leaves", "min_data_in_leaf", "feature_fraction", "bagging_fraction",
        "bagging_freq", "lambda_l2", "scale_pos_weight", "learning_rate",
    )  # fmt: skip
    params = {k: v for k, v in booster.params.items() if k in keep}
    params |= {"verbosity": -1, "deterministic": True, "force_row_wise": True}
    rounds = booster.current_iteration()
    rng = np.random.default_rng(seed)
    models = []
    for i in range(n_models):
        rows = rng.integers(0, len(x_train), len(x_train))
        data = lgb.Dataset(x_train[rows], y_train[rows], feature_name=booster.feature_name())
        models.append(lgb.train({**params, "seed": seed + i}, data, num_boost_round=rounds))
    return models


def seed_stability(models: list[lgb.Booster], x: np.ndarray, codes: ReasonCodes) -> dict[str, Any]:
    grouped = [codes.grouped(contributions(m, x)[0]) for m in models]
    tops = [top_k_sets(g) for g in grouped]
    n = len(x)
    jac = np.zeros(n)
    tau = np.zeros(n)
    pairs = list(combinations(range(len(models)), 2))
    for a, b in pairs:
        jac += np.array([jaccard(tops[a][i], tops[b][i]) for i in range(n)])
        tau += np.array([kendalltau(grouped[a][i], grouped[b][i]).statistic for i in range(n)])
    jac /= len(pairs)
    tau = np.nan_to_num(tau / len(pairs), nan=1.0)
    return {
        "per_row_jaccard": jac,
        "per_row_tau": tau,
        "gap": gap_after_top_k(grouped[0]),
        "median_jaccard": float(np.median(jac)),
        "mean_jaccard": float(np.mean(jac)),
        "share_jaccard_ge_0.8": float(np.mean(jac >= STABLE_JACCARD)),
        "median_tau": float(np.median(tau)),
    }


def perturbation_stability(
    booster: lgb.Booster,
    x: np.ndarray,
    codes: ReasonCodes,
    feature_std: np.ndarray,
    *,
    scale: float = 0.05,
    repeats: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    binary = np.array([np.isin(np.unique(x[:, j]), [0.0, 1.0]).all() for j in range(x.shape[1])])
    base_contrib, base_bias = contributions(booster, x)
    base_group = codes.grouped(base_contrib)
    base_margin = base_contrib.sum(axis=1) + base_bias
    base_top = top_k_sets(base_group)
    rel_attr, d_margin, jac = [], [], []
    for _ in range(repeats):
        noise = rng.normal(0, 1, x.shape) * (scale * feature_std)[None, :]
        noise[:, binary] = 0.0
        c, bias = contributions(booster, x + noise)
        g = codes.grouped(c)
        rel_attr.append(
            np.linalg.norm(g - base_group, axis=1) / (np.linalg.norm(base_group, axis=1) + 1e-12)
        )
        d_margin.append(np.abs(c.sum(axis=1) + bias - base_margin))
        top = top_k_sets(g)
        jac.append([jaccard(a, b) for a, b in zip(base_top, top, strict=True)])
    rel = np.mean(rel_attr, axis=0)
    dm = np.mean(d_margin, axis=0)
    return {
        "median_relative_attribution_change": float(np.median(rel)),
        "median_margin_change": float(np.median(dm)),
        "median_top3_jaccard": float(np.median(np.mean(jac, axis=0))),
        "share_top3_unchanged": float(np.mean(np.mean(jac, axis=0) == 1.0)),
    }


def background_stability(
    booster: lgb.Booster,
    x: np.ndarray,
    backgrounds: list[np.ndarray],
    codes: ReasonCodes,
) -> dict[str, Any]:
    """Interventional SHAP with different background samples (needs `shap`)."""
    import shap

    tops = []
    for bg in backgrounds:
        masker = shap.maskers.Independent(bg, max_samples=len(bg))  # use the whole sample
        explainer = shap.TreeExplainer(
            booster, data=masker, feature_perturbation="interventional", model_output="raw"
        )
        values = np.asarray(explainer.shap_values(x, check_additivity=False))
        tops.append(top_k_sets(codes.grouped(values)))
    pairs = list(combinations(range(len(backgrounds)), 2))
    jac = np.mean(
        [[jaccard(tops[a][i], tops[b][i]) for i in range(len(x))] for a, b in pairs], axis=0
    )
    return {"median_top3_jaccard": float(np.median(jac)), "mean_top3_jaccard": float(np.mean(jac))}


def stability_threshold(gap: np.ndarray, per_row_jaccard: np.ndarray) -> dict[str, float]:
    """Gap threshold that best separates stable (Jaccard >= 0.8) from unstable rows
    (balanced accuracy). Used by the API's `explanation_stable` flag."""
    stable = per_row_jaccard >= STABLE_JACCARD
    best = {"threshold": 0.0, "balanced_accuracy": 0.5}
    if stable.all() or (~stable).all():
        return best
    for t in np.unique(np.quantile(gap[np.isfinite(gap)], np.linspace(0.02, 0.98, 49))):
        predicted = gap >= t
        tpr = np.mean(predicted[stable])
        tnr = np.mean(~predicted[~stable])
        acc = float((tpr + tnr) / 2)
        if acc > best["balanced_accuracy"]:
            best = {"threshold": float(t), "balanced_accuracy": acc}
    best["share_flagged_unstable"] = float(np.mean(gap < best["threshold"]))
    return best
