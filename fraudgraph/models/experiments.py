"""Graph study (task 3.6): the experiment matrix, judged by the pre-registered rule (3.0).

    E0  T+O                      LightGBM (baseline)
    E1  T+O+G1                   LightGBM               <- primary comparison vs E0
    E2  T+O+G1+G2 (GNN emb.)     LightGBM (hybrid)      secondary (Holm)
    E3  causal edge GNN          end-to-end             secondary (Holm)
    E4  E0 vs E1 with injected mule rings in the test period
    E5a/b  E0 + sender / all balances: artifact references only

Rule: over 3 seeds (scores averaged per row), the graph variant wins if the paired
block-bootstrap 95 % CI of dAP > 0 AND of d(test cost) < 0, AND it fits the latency
budget (D5). Each model's cost uses its own isotonic calibration on calib_a and the
expected-cost rule. Ablations: no hour, no G1-b, drop each G1 sub-family.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from fraudgraph.decision.costs import CostParams, expected_cost_decisions, realised_costs
from fraudgraph.features.matrix import frame_to_matrix
from fraudgraph.models.baselines import HOUR_FEATURE, fit_lgbm
from fraudgraph.models.calibrate import fit_calibrator
from fraudgraph.models.compare import Ranked, block_weights

SPLITS = ("train_fit", "train_val", "calib_a", "calib_b", "test", "late_test")
SCORED = ("train_val", "calib_a", "test", "late_test")
G1_FAMILIES: dict[str, Callable[[str], bool]] = {
    "G1-a": lambda c: c.startswith("g_") and c not in {"g_dest_known_fraud_in", "g_pair_n"},
    "G1-b": lambda c: c == "g_dest_known_fraud_in",
    "G1-c": lambda c: c == "g_pair_n",
}
ARTIFACT_SENDER = ("oldbalance_orig", "a_amount_eq_balance", "a_amount_ratio")
ARTIFACT_ALL = (
    *ARTIFACT_SENDER,
    "newbalance_orig",
    "oldbalance_dest",
    "newbalance_dest",
    "a_error_orig",
    "a_error_dest",
)


def feature_columns(table: pl.DataFrame) -> dict[str, list[str]]:
    base = [c for c in table.columns if c.startswith(("t_", "o_"))]
    graph = [c for c in table.columns if c.startswith("g_")]
    sets = {
        "E0": base,
        "E1": base + graph,
        "E0_no_hour": [c for c in base if c != HOUR_FEATURE],
        "E1_no_hour": [c for c in base + graph if c != HOUR_FEATURE],
        "E1_no_G1b": base + [c for c in graph if c != "g_dest_known_fraud_in"],
        "E5a_sender_balance": base + list(ARTIFACT_SENDER),
        "E5b_all_balances": base + list(ARTIFACT_ALL),
    }
    for family, keep in G1_FAMILIES.items():
        if family != "G1-b":  # E1 minus G1-b is exactly E1_no_G1b
            sets[f"E1_minus_{family}"] = base + [c for c in graph if not keep(c)]
    return sets


def with_artifacts(table: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """Balance columns (excluded by D2) for the artifact-reference runs only."""
    raw = events.select(
        "txn_id",
        "amount",
        "oldbalance_orig",
        "newbalance_orig",
        "oldbalance_dest",
        "newbalance_dest",
    ).with_columns(
        a_amount_eq_balance=(pl.col("amount") == pl.col("oldbalance_orig")).cast(pl.Float32),
        a_amount_ratio=pl.col("amount") / (pl.col("oldbalance_orig") + 1),
        a_error_orig=pl.col("oldbalance_orig") - pl.col("amount") - pl.col("newbalance_orig"),
        a_error_dest=pl.col("oldbalance_dest") + pl.col("amount") - pl.col("newbalance_dest"),
    )
    return table.join(raw, on="txn_id", how="left", maintain_order="left")


# --------------------------------------------------------------------- fitting


def fit_lgbm_scores(
    parts: dict[str, pl.DataFrame],
    features: list[str],
    *,
    horizon: int,
    trials: int,
    seed: int,
    extra: dict[str, np.ndarray] | None = None,
    train_mask: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], lgb.Booster, float]:
    """Fit on train_fit (optionally a row mask), select on train_val, score SCORED splits.

    `extra` adds dense columns per split (the GNN embeddings for the hybrid).
    """

    def x(split: str) -> np.ndarray:
        m = frame_to_matrix(parts[split], features, horizon)
        if extra is not None:
            m = np.hstack([m, extra[split]])
        return m

    x_train, y_train = x("train_fit"), _y(parts["train_fit"])
    if train_mask is not None:
        x_train, y_train = x_train[train_mask], y_train[train_mask]
    names = features + (
        [f"g2_emb_{i}" for i in range(extra["train_val"].shape[1])] if extra else []
    )
    booster, result = fit_lgbm(
        x_train, y_train, x("train_val"), _y(parts["train_val"]), names, trials=trials, seed=seed
    )
    scores = {
        s: np.asarray(booster.predict(x(s), num_iteration=booster.best_iteration)) for s in SCORED
    }
    return scores, booster, result.val_ap


# --------------------------------------------------------------------- judging


def costs_per_row(
    scores: dict[str, np.ndarray], parts: dict[str, pl.DataFrame], costs: CostParams
) -> np.ndarray:
    """Realised test cost per row under the model's own calibrated expected-cost policy."""
    calibrator = fit_calibrator(scores["calib_a"], _y(parts["calib_a"]))
    p = calibrator(scores["test"])
    amount = parts["test"]["amount"].to_numpy()
    actions = expected_cost_decisions(p, amount, costs)
    return realised_costs(actions, _y(parts["test"]), amount, costs)


def paired_test(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    cost_a: np.ndarray,
    cost_b: np.ndarray,
    blocks: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """dAP = AP(a) - AP(b) and dCost = cost(a) - cost(b), paired block bootstrap."""
    ra, rb = Ranked(y, score_a), Ranked(y, score_b)
    index, counts = block_weights(blocks, resamples, seed)
    d_ap = np.array(
        [
            ra.metrics(counts[r][index[ra.order]])["ap"]
            - rb.metrics(counts[r][index[rb.order]])["ap"]
            for r in range(resamples)
        ]
    )
    per_block = np.bincount(index, weights=cost_a - cost_b)
    d_cost = counts @ per_block
    return {
        "delta_ap": ra.metrics()["ap"] - rb.metrics()["ap"],
        "delta_ap_ci95": [float(np.percentile(d_ap, 2.5)), float(np.percentile(d_ap, 97.5))],
        "delta_ap_p": _two_sided_p(d_ap),
        "delta_cost": float((cost_a - cost_b).sum()),
        "delta_cost_ci95": [float(np.percentile(d_cost, 2.5)), float(np.percentile(d_cost, 97.5))],
        "delta_cost_p": _two_sided_p(-d_cost),
    }


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


def verdict(test: dict[str, Any], latency_ok: bool) -> str:
    ap_win = test["delta_ap_ci95"][0] > 0
    cost_win = test["delta_cost_ci95"][1] < 0
    if ap_win and cost_win and latency_ok:
        return "WIN"
    return "no significant gain"


def latency_p95_ms(fn: Any, inputs: list[Any], warmup: int = 50) -> float:
    for item in inputs[:warmup]:
        fn(item)
    times = []
    for item in inputs:
        started = time.perf_counter()
        fn(item)
        times.append((time.perf_counter() - started) * 1_000)
    return float(np.percentile(times, 95))


def gain_by_family(booster: lgb.Booster) -> dict[str, float]:
    gains = booster.feature_importance("gain")
    total = float(gains.sum()) or 1.0
    out: dict[str, float] = {}
    for name, gain in zip(booster.feature_name(), gains, strict=True):
        family = {"t": "transaction", "o": "origin", "g": "graph G1"}.get(
            name.split("_")[0], "other"
        )
        if name.startswith("g2_"):
            family = "GNN embedding G2"
        out[family] = out.get(family, 0.0) + float(gain) / total
    return {k: round(v, 4) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}


def _y(frame: pl.DataFrame) -> np.ndarray:
    return frame["is_fraud"].cast(pl.Int8).to_numpy()


def _two_sided_p(samples: np.ndarray) -> float:
    """Bootstrap p-value for 'delta > 0' being real (two-sided)."""
    share = float(np.mean(samples > 0))
    return min(1.0, 2 * min(share, 1 - share))


def params_of(cost: CostParams) -> dict[str, float]:
    return asdict(cost)
