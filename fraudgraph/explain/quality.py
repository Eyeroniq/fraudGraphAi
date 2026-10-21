"""Explanation quality (task 5.3): is SHAP faithful, robust and model-dependent, and does it
clearly beat trivial attributions?

Methods compared per payment: SHAP (TreeSHAP on the margin), random Gaussian
attributions, and global permutation importance (the same vector for every payment).

Metrics:
- Faithfulness correlation (own, tabular): replace random 3-feature subsets with training
  means; correlate the attribution mass removed with the drop in the margin. Higher is
  better. (Quantus's version perturbs with image-style per-row baselines, which is
  meaningless when columns have different units.)
- Model randomisation (own): rank correlation between attributions from the real model and
  from a model trained on shuffled labels. Lower is better: attributions must depend on
  what the model learned.
- Quantus 0.6 on standardised inputs (the wrapper un-standardises before predicting):
  MonotonicityCorrelation (higher better), MaxSensitivity and LocalLipschitzEstimate
  (lower better; note they reward constant explanations, so permutation importance wins
  them trivially), RandomLogit (trivial for two classes: the other class is the negation).

Per-payment scores get 95 % bootstrap CIs over payments.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr

from fraudgraph.explain.reasons import contributions

Explainer = Callable[[np.ndarray], np.ndarray]  # x [n, F] -> attributions [n, F]


def margin(booster: lgb.Booster, x: np.ndarray) -> np.ndarray:
    return np.asarray(
        booster.predict(x, raw_score=True, num_iteration=booster.best_iteration or None)
    )


def permutation_importance(booster: lgb.Booster, x: np.ndarray, seed: int = 0) -> np.ndarray:
    """Mean |change in margin| when one column is shuffled (global, non-negative)."""
    rng = np.random.default_rng(seed)
    base = margin(booster, x)
    out = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        shuffled = x.copy()
        shuffled[:, j] = rng.permutation(shuffled[:, j])
        out[j] = np.mean(np.abs(margin(booster, shuffled) - base))
    return out


def explainers(
    booster: lgb.Booster, global_importance: np.ndarray, seed: int = 0
) -> dict[str, Explainer]:
    rng = np.random.default_rng(seed)
    return {
        "shap": lambda x: contributions(booster, x)[0],
        "random": lambda x: rng.normal(0, 1, x.shape),
        "permutation": lambda x: np.broadcast_to(global_importance, x.shape).copy(),
    }


def faithfulness_correlation(
    booster: lgb.Booster,
    x: np.ndarray,
    attributions: np.ndarray,
    train_mean: np.ndarray,
    *,
    subset: int = 3,
    runs: int = 30,
    seed: int = 0,
) -> np.ndarray:
    """Per-row Pearson correlation between removed attribution mass and margin drop."""
    rng = np.random.default_rng(seed)
    n, f = x.shape
    base = margin(booster, x)
    removed = np.zeros((n, runs))
    drop = np.zeros((n, runs))
    for r in range(runs):
        idx = np.array([rng.choice(f, subset, replace=False) for _ in range(n)])
        rows = np.arange(n)[:, None]
        x_pert = x.copy()
        x_pert[rows, idx] = train_mean[idx]
        removed[:, r] = attributions[rows, idx].sum(axis=1)
        drop[:, r] = base - margin(booster, x_pert)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = removed - removed.mean(axis=1, keepdims=True)
        b = drop - drop.mean(axis=1, keepdims=True)
        corr = (a * b).sum(axis=1) / np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    return np.asarray(corr)


def randomisation_correlation(real: np.ndarray, shuffled_model: np.ndarray) -> np.ndarray:
    """Per-row Spearman correlation between attributions of the real and a label-shuffled model."""
    out = np.array([spearmanr(a, b).statistic for a, b in zip(real, shuffled_model, strict=True)])
    return np.asarray(np.nan_to_num(out, nan=1.0))  # constant rows (e.g. permutation) = identical


def quantus_scores(
    booster: lgb.Booster,
    x: np.ndarray,
    explain: Explainer,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, np.ndarray]:
    """Quantus metrics on standardised inputs shaped (n, 1, F)."""
    import quantus
    import torch

    std = np.where(std > 0, std, 1.0)

    def to_x(z: np.ndarray) -> np.ndarray:
        return np.asarray(np.asarray(z).reshape(len(z), -1) * std + mean)

    class Wrapped(torch.nn.Module):
        def forward(self, z: torch.Tensor) -> torch.Tensor:
            p = np.asarray(booster.predict(to_x(z.detach().cpu().numpy())))
            return torch.tensor(np.stack([1 - p, p], axis=1), dtype=torch.float32)

    def explain_func(model: Any, inputs: np.ndarray, targets: np.ndarray, **_: Any) -> np.ndarray:
        attr = explain(to_x(inputs)).astype(np.float32)
        sign = np.where(np.asarray(targets).reshape(-1) == 1, 1.0, -1.0)[:, None]
        return (attr * sign).reshape(len(inputs), 1, -1)

    z = ((x - mean) / std).astype(np.float32).reshape(len(x), 1, -1)
    y = np.ones(len(x), dtype=int)  # explain the fraud score
    a = explain_func(None, z, y)
    model = Wrapped().eval()
    kw: dict[str, Any] = {"disable_warnings": True, "display_progressbar": False}
    metrics = {
        "monotonicity": quantus.MonotonicityCorrelation(
            nr_samples=5, features_in_step=1, perturb_baseline="black", **kw
        ),
        "max_sensitivity": quantus.MaxSensitivity(nr_samples=5, lower_bound=0.05, **kw),
        "local_lipschitz": quantus.LocalLipschitzEstimate(nr_samples=5, **kw),
        "random_logit": quantus.RandomLogit(
            num_classes=2, similarity_func=quantus.similarity_func.correlation_spearman, **kw
        ),
    }
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, metric in metrics.items():
            scores = metric(
                model=model,
                x_batch=z,
                y_batch=y,
                a_batch=a,
                explain_func=explain_func,
                device="cpu",
                channel_first=True,
            )
            out[name] = np.asarray(scores, dtype=np.float64)
    return out


def bootstrap_ci(values: np.ndarray, resamples: int = 1_000, seed: int = 0) -> dict[str, Any]:
    v = values[np.isfinite(values)]
    if not len(v):
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    rng = np.random.default_rng(seed)
    means = rng.choice(v, (resamples, len(v)), replace=True).mean(axis=1)
    return {
        "mean": float(v.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "n": len(v),
    }
