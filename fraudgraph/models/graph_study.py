"""`fgx graph-study`: run the 3.6 experiment matrix on PaySim and judge it (see
`models/experiments.py` for the matrix and the pre-registered rule)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from fraudgraph.data.scenarios import inject_mule_rings
from fraudgraph.data.splits import SplitPlan, assign_split
from fraudgraph.decision.costs import CostParams
from fraudgraph.features.backfill import MODELLED_TYPES, LabelQueue, backfill, events_from
from fraudgraph.features.engine import FeatureEngine
from fraudgraph.features.matrix import frame_to_matrix, row_to_vector
from fraudgraph.features.spec import FeatureSpec
from fraudgraph.features.state import InMemoryStore
from fraudgraph.graph.sampler import TemporalNeighbours
from fraudgraph.graph.train import GNNConfig, fit, out_of_fold_embeddings, predict
from fraudgraph.models.compare import Ranked
from fraudgraph.models.experiments import (
    SCORED,
    SPLITS,
    costs_per_row,
    feature_columns,
    fit_lgbm_scores,
    gain_by_family,
    holm,
    latency_p95_ms,
    paired_test,
    verdict,
    with_artifacts,
)

LATENCY_BUDGET_MS = 50.0


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run_graph_study(
    events: pl.DataFrame,
    features: pl.DataFrame,
    plan: SplitPlan,
    costs: CostParams,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    trials: int = 20,
    resamples: int = 1_000,
    gnn: GNNConfig | None = None,
) -> dict[str, Any]:
    spec = FeatureSpec()
    horizon = spec.horizon_steps
    table = with_artifacts(assign_split(features, plan), events)
    parts = {s: table.filter(pl.col("split") == s) for s in SPLITS}
    sets = feature_columns(table)
    scores: dict[str, list[dict[str, np.ndarray]]] = {}
    boosters: dict[str, list[Any]] = {}
    val_ap: dict[str, list[float]] = {}
    gains: dict[str, dict[str, float]] = {}

    def run_lgbm(name: str, columns: list[str], seed: int, **kw: Any) -> None:
        log(f"{name} seed {seed}")
        s, booster, v = fit_lgbm_scores(
            parts, columns, horizon=horizon, trials=trials, seed=seed, **kw
        )
        scores.setdefault(name, []).append(s)
        boosters.setdefault(name, []).append(booster)
        val_ap.setdefault(name, []).append(v)
        if seed == seeds[0]:
            gains[name] = gain_by_family(booster)

    for seed in seeds:
        for name in ("E0", "E1"):
            run_lgbm(name, sets[name], seed)
    for name in (
        "E0_no_hour",
        "E1_no_hour",
        "E1_no_G1b",
        "E1_minus_G1-a",
        "E1_minus_G1-c",
        "E5a_sender_balance",
        "E5b_all_balances",
    ):
        run_lgbm(name, sets[name], seeds[0])

    # ---------------------------------------------------------------- GNN (E3) + hybrid (E2)
    log("building causal neighbourhoods")
    base_cfg = gnn or GNNConfig()
    neighbours = TemporalNeighbours(
        events, horizon_steps=horizon, label_delay_steps=spec.label_delay_steps
    )
    ids = {s: parts[s]["txn_id"].to_numpy() for s in SPLITS}
    batches = {s: neighbours.batch(ids[s], base_cfg.k) for s in SCORED}
    conv_choice: dict[str, float] = {}
    probes = {}
    for conv in ("sage", "gine"):
        probes[conv] = fit(
            neighbours, ids["train_fit"], ids["train_val"], _cfg(base_cfg, conv, seeds[0])
        )
        conv_choice[conv] = probes[conv][1].best_val_ap
        log(f"GNN {conv} seed {seeds[0]}: val AP {conv_choice[conv]:.4f}")
    conv = max(conv_choice, key=lambda c: conv_choice[c])
    gnn_models = []
    for seed in seeds:
        cfg = _cfg(base_cfg, conv, seed)
        if seed == seeds[0]:
            model, report = probes[conv]  # already fitted with this exact config
        else:
            model, report = fit(neighbours, ids["train_fit"], ids["train_val"], cfg)
        gnn_models.append(model)
        val_ap.setdefault("E3", []).append(report.best_val_ap)
        e3, emb = {}, {}
        for s in SCORED:
            e3[s], emb[s] = predict(model, batches[s])
        scores.setdefault("E3", []).append(e3)
        log(f"E3 seed {seed}: val AP {report.best_val_ap:.4f}")
        oof, has = out_of_fold_embeddings(
            neighbours,
            ids["train_fit"],
            parts["train_fit"]["step"].to_numpy(),
            ids["train_val"],
            cfg,
            embargo_steps=spec.label_delay_steps,
        )
        emb["train_fit"] = oof
        run_lgbm("E2", sets["E1"], seed, extra=emb, train_mask=has)

    # ---------------------------------------------------------------- judge on test
    log("judging")
    y = parts["test"]["is_fraud"].cast(pl.Int8).to_numpy()
    blocks = (parts["test"]["step"].to_numpy() - 1) // plan.bootstrap_block_steps
    mean_scores = {
        name: {s: np.mean([run[s] for run in runs], axis=0) for s in SCORED}
        for name, runs in scores.items()
    }
    cost = {name: costs_per_row(sc, parts, costs) for name, sc in mean_scores.items()}
    summary: dict[str, Any] = {}
    for name, sc in mean_scores.items():
        summary[name] = {
            "seeds": len(scores[name]),
            "val_ap": val_ap.get(name),
            "test_ap": Ranked(y, sc["test"]).metrics()["ap"],
            "late_test_ap": Ranked(
                parts["late_test"]["is_fraud"].cast(pl.Int8).to_numpy(), sc["late_test"]
            ).metrics()["ap"],
            "test_cost": float(cost[name].sum()),
            "gain_by_family": gains.get(name),
        }

    comparisons = {
        name: paired_test(
            y,
            mean_scores[name]["test"],
            mean_scores["E0"]["test"],
            cost[name],
            cost["E0"],
            blocks,
            resamples=resamples,
            seed=0,
        )
        for name in mean_scores
        if name != "E0"
    }
    # Secondary comparisons: Holm across E2/E3, for both conditions of the rule.
    for metric in ("delta_ap_p", "delta_cost_p"):
        for n, p in holm({n: comparisons[n][metric] for n in ("E2", "E3")}).items():
            comparisons[n][f"{metric}_holm"] = p

    log("latency")
    latency = measure_latency(
        events, plan, boosters, gnn_models[0], neighbours, ids["test"], base_cfg.k
    )
    budget_ok = {n: latency[n] < LATENCY_BUDGET_MS for n in latency}
    verdicts = {
        "E1": verdict(comparisons["E1"], budget_ok["E1"]),
        **{
            n: verdict(comparisons[n], budget_ok[n])
            if max(comparisons[n]["delta_ap_p_holm"], comparisons[n]["delta_cost_p_holm"]) < 0.05
            else "no significant gain (Holm)"
            for n in ("E2", "E3")
        },
    }

    log("E4: injected mule rings")
    e4 = mule_ring_scenario(events, plan, boosters, sets, resamples=resamples)

    return {
        "rule": "win iff CI(dAP) > 0 and CI(dCost) < 0 vs E0, seed-averaged, and p95 latency "
        f"< {LATENCY_BUDGET_MS} ms; E2/E3 Holm-corrected",
        "seeds": list(seeds),
        "trials": trials,
        "gnn": {"conv_selected": conv, "val_ap_by_conv": conv_choice, "config": base_cfg.__dict__},
        "summary": summary,
        "comparisons_vs_E0": comparisons,
        "latency_p95_ms": latency,
        "verdicts": verdicts,
        "E4_mule_rings": e4,
        "test": {"rows": len(y), "fraud": int(y.sum()), "blocks": int(np.unique(blocks).size)},
    }


def measure_latency(
    events: pl.DataFrame,
    plan: SplitPlan,
    boosters: dict[str, list[Any]],
    gnn_model: Any,
    neighbours: TemporalNeighbours,
    test_ids: np.ndarray,
    k: int,
    n: int = 2_000,
) -> dict[str, float]:
    """p95 per-transaction latency: engine features + model, on live state at test time."""
    spec = FeatureSpec()
    engine = FeatureEngine(spec, InMemoryStore(spec.horizon_steps))
    labels = LabelQueue(spec.label_delay_steps)
    start = plan.ranges["test"].start
    sample = []
    for ev, is_fraud in zip(events_from(events), events["is_fraud"].to_list(), strict=True):
        if ev.step >= start:
            if ev.type in MODELLED_TYPES:
                sample.append(ev)
            if len(sample) >= n:
                break
            continue
        labels.release(engine, ev.step)
        engine.apply(ev)
        if is_fraud:
            labels.push(ev)

    out = {}
    for name in ("E0", "E1", "E2"):
        booster = boosters[name][0]
        columns = booster.feature_name()
        engine_cols = [c for c in columns if not c.startswith("g2_")]
        pad = len(columns) - len(engine_cols)

        def score_one(
            ev: Any, booster: Any = booster, cols: list[str] = engine_cols, pad: int = pad
        ) -> None:
            x = row_to_vector(engine.features(ev), cols, spec.horizon_steps)
            if pad:
                x = np.hstack([x, np.zeros((1, pad))])  # GNN embedding cost added below
            booster.predict(x)

        out[name] = latency_p95_ms(score_one, sample)

    def gnn_one(txn_id: str) -> None:
        predict(gnn_model, neighbours.batch(np.array([txn_id]), k), chunk=1)

    gnn_ms = latency_p95_ms(gnn_one, list(test_ids[:n]))
    out["E3"] = gnn_ms  # causal neighbourhood + forward pass
    out["E2"] = out["E2"] + gnn_ms
    return out


def mule_ring_scenario(
    events: pl.DataFrame,
    plan: SplitPlan,
    boosters: dict[str, list[Any]],
    sets: dict[str, list[str]],
    *,
    resamples: int,
) -> dict[str, Any]:
    """E4: inject rings into the test period; score with models trained on clean data."""
    test = plan.ranges["test"]
    injected_events, injected = inject_mule_rings(
        events, first_step=test.start, last_step=test.end, seed=0
    )
    spec = FeatureSpec()
    table, _ = backfill(injected_events, spec)
    part = (
        assign_split(table, plan)
        .filter(pl.col("split") == "test")
        .join(injected_events.select("txn_id", "amount"), on="txn_id", how="left")
    )
    y = part["is_fraud"].cast(pl.Int8).to_numpy()
    is_ring = part["txn_id"].is_in(injected).to_numpy()
    blocks = (part["step"].to_numpy() - 1) // plan.bootstrap_block_steps
    out: dict[str, Any] = {"injected_edges": len(injected), "test_rows": part.height}
    mean = {}
    for name in ("E0", "E1"):
        x = frame_to_matrix(part, sets[name], spec.horizon_steps)
        mean[name] = np.mean(
            [b.predict(x, num_iteration=b.best_iteration) for b in boosters[name]], axis=0
        )
        top = mean[name] >= np.quantile(mean[name], 0.99)
        out[name] = {
            "test_ap": Ranked(y, mean[name]).metrics()["ap"],
            "ring_ap": Ranked(is_ring.astype(np.int8), mean[name]).metrics()["ap"],
            "ring_recall_top1pct": float(top[is_ring].mean()),
        }
    zeros = np.zeros(len(y))
    out["E1_vs_E0"] = paired_test(
        y, mean["E1"], mean["E0"], zeros, zeros, blocks, resamples=resamples, seed=0
    )
    return out


def _cfg(base: GNNConfig, conv: str, seed: int) -> GNNConfig:
    return GNNConfig(**{**base.__dict__, "conv": conv, "seed": seed})


def save(results: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
