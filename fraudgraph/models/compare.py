"""Evaluation harness (task 1.4): rare-event metrics with block-bootstrap CIs.

Resampling whole time blocks (default: the block size in splits.json) respects time
correlation. Drawing blocks with replacement is the same as giving every row the
number of times its block was drawn as a weight, so each resample is a weighted
metric over rows sorted once: fast enough for 1,000 resamples on the full test set.

No accuracy anywhere; ROC-AUC is reported only as a footnote.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PRECISION_TARGETS = (0.5, 0.8, 0.9)


@dataclass(frozen=True)
class Curve:
    """Points of the PR curve at each distinct score (highest score first)."""

    tp: np.ndarray  # weighted true positives flagged at or above each threshold
    fp: np.ndarray
    thresholds: np.ndarray


class Ranked:
    """Rows sorted once by score; metrics take per-row weights (bootstrap counts)."""

    def __init__(self, y: np.ndarray, score: np.ndarray, amount: np.ndarray | None = None):
        order = np.argsort(-score, kind="stable")
        self.y = y[order].astype(np.float64)
        self.score = score[order].astype(np.float64)
        self.amount = None if amount is None else amount[order].astype(np.float64)
        self.order = order
        # Last row of each group of tied scores: thresholds sit between distinct scores.
        distinct = np.flatnonzero(np.diff(self.score)) if len(self.score) else np.array([])
        self.group_end = np.r_[distinct, len(self.score) - 1].astype(np.int64)

    def curve(self, w: np.ndarray | None = None) -> Curve:
        w = np.ones_like(self.y) if w is None else w
        tp = np.cumsum(self.y * w)[self.group_end]
        fp = np.cumsum((1.0 - self.y) * w)[self.group_end]
        return Curve(tp=tp, fp=fp, thresholds=self.score[self.group_end])

    def metrics(self, w: np.ndarray | None = None) -> dict[str, float]:
        c = self.curve(w)
        positives = c.tp[-1] if len(c.tp) else 0.0
        if positives <= 0:
            return {
                "ap": float("nan"),
                **{f"recall_at_p{p}": float("nan") for p in PRECISION_TARGETS},
            }
        precision = c.tp / np.maximum(c.tp + c.fp, 1e-12)
        recall = c.tp / positives
        # AP = sum over thresholds of (recall step) x precision, as in sklearn.
        ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
        out = {"ap": ap}
        for p in PRECISION_TARGETS:
            ok = precision >= p
            out[f"recall_at_p{p}"] = float(recall[ok].max()) if ok.any() else 0.0
        return out


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Point estimate only (footnote metric)."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")


def topk_flags(score: np.ndarray, blocks: np.ndarray, k_per_block: int) -> np.ndarray:
    """True for the `k_per_block` highest scores inside each block (the alert budget)."""
    flags = np.zeros(len(score), dtype=bool)
    order = np.lexsort((-score, blocks))  # by block, then score descending
    sorted_blocks = blocks[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_blocks)) + 1]
    rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    flags[order[rank < k_per_block]] = True
    return flags


@dataclass(frozen=True)
class Evaluation:
    point: dict[str, float]
    ci: dict[str, tuple[float, float]]

    def as_dict(self) -> dict[str, Any]:
        return {
            name: {"value": value, "ci95": list(self.ci.get(name, (np.nan, np.nan)))}
            for name, value in self.point.items()
        }


def block_weights(blocks: np.ndarray, resamples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw blocks with replacement. Returns (block index per row, counts[resamples, n_blocks])."""
    unique, index = np.unique(blocks, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(resamples, len(unique)))
    counts = np.zeros((resamples, len(unique)), dtype=np.float64)
    for r in range(resamples):
        counts[r] = np.bincount(draws[r], minlength=len(unique))
    return index, counts


def evaluate(
    y: np.ndarray,
    score: np.ndarray,
    blocks: np.ndarray,
    *,
    amount: np.ndarray | None = None,
    alerts_per_block: int = 25,
    resamples: int = 1_000,
    seed: int = 0,
) -> Evaluation:
    """Point metrics + 95 % block-bootstrap CIs for one model on one split."""
    ranked = Ranked(y, score, amount)
    flagged = topk_flags(score, blocks, alerts_per_block)[ranked.order]
    block_of_row, counts = block_weights(blocks[ranked.order], resamples, seed)

    def extra(w: np.ndarray) -> dict[str, float]:
        alerts = float((flagged * w).sum())
        hits = float((flagged * ranked.y * w).sum())
        out = {"precision_at_k": hits / alerts if alerts else float("nan")}
        if ranked.amount is not None:
            fraud_amount = float((ranked.y * ranked.amount * w).sum())
            caught = float((flagged * ranked.y * ranked.amount * w).sum())
            out["fraud_amount_caught_share_at_k"] = caught / fraud_amount if fraud_amount else 0.0
        return out

    ones = np.ones_like(ranked.y)
    point = {**ranked.metrics(ones), **extra(ones), "prevalence": float(y.mean())}
    samples: dict[str, list[float]] = {k: [] for k in point if k != "prevalence"}
    for r in range(resamples):
        w = counts[r][block_of_row]
        values = {**ranked.metrics(w), **extra(w)}
        for k in samples:
            samples[k].append(values[k])
    ci = {k: _percentile_ci(v) for k, v in samples.items()}
    point["roc_auc_footnote"] = roc_auc(y, score)
    return Evaluation(point=point, ci=ci)


def paired_difference(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    blocks: np.ndarray,
    *,
    resamples: int = 1_000,
    seed: int = 0,
) -> dict[str, Any]:
    """AP(a) - AP(b) with a paired block bootstrap (the same resamples for both)."""
    ra, rb = Ranked(y, score_a), Ranked(y, score_b)
    block_index, counts = block_weights(blocks, resamples, seed)
    wa_index, wb_index = block_index[ra.order], block_index[rb.order]
    diffs = [
        ra.metrics(counts[r][wa_index])["ap"] - rb.metrics(counts[r][wb_index])["ap"]
        for r in range(resamples)
    ]
    point = ra.metrics()["ap"] - rb.metrics()["ap"]
    lo, hi = _percentile_ci(diffs)
    return {
        "delta_ap": point,
        "ci95": [lo, hi],
        "share_of_resamples_a_better": float(np.mean(np.array(diffs) > 0)),
    }


def pr_band(
    y: np.ndarray,
    score: np.ndarray,
    blocks: np.ndarray,
    *,
    resamples: int = 200,
    seed: int = 0,
    grid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Precision at a recall grid: point curve plus 2.5/97.5 % bootstrap band."""
    grid = np.linspace(0.0, 1.0, 101) if grid is None else grid
    ranked = Ranked(y, score)
    block_of_row, counts = block_weights(blocks[ranked.order], resamples, seed)

    def precision_at(w: np.ndarray) -> np.ndarray:
        c = ranked.curve(w)
        if c.tp[-1] <= 0:
            return np.full_like(grid, np.nan)
        recall = c.tp / c.tp[-1]
        precision = c.tp / np.maximum(c.tp + c.fp, 1e-12)
        # Interpolated precision: best precision at recall >= r.
        best = np.maximum.accumulate(precision[::-1])[::-1]
        idx = np.searchsorted(recall, grid, side="left")
        return np.where(idx < len(best), best[np.minimum(idx, len(best) - 1)], 0.0)

    curves = np.array([precision_at(counts[r][block_of_row]) for r in range(resamples)])
    return {
        "recall": grid,
        "precision": precision_at(np.ones_like(ranked.y)),
        "low": np.nanpercentile(curves, 2.5, axis=0),
        "high": np.nanpercentile(curves, 97.5, axis=0),
    }


def reliability(y: np.ndarray, prob: np.ndarray, bins: int = 15) -> dict[str, np.ndarray]:
    """Mean predicted vs observed rate in equal-count bins of the predicted probability."""
    order = np.argsort(prob, kind="stable")
    chunks = np.array_split(order, bins)
    return {
        "predicted": np.array([prob[c].mean() for c in chunks]),
        "observed": np.array([y[c].mean() for c in chunks]),
        "count": np.array([len(c) for c in chunks]),
    }


def _percentile_ci(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if not len(arr):
        return (float("nan"), float("nan"))
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
