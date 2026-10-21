from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fraudgraph.config import SplitSettings
from fraudgraph.data.prepare import to_canonical
from fraudgraph.data.scenarios import inject_mule_rings
from fraudgraph.data.splits import plan_splits
from fraudgraph.data.synth import SynthConfig, generate
from fraudgraph.decision.costs import CostParams
from fraudgraph.features.backfill import backfill
from fraudgraph.features.spec import FeatureSpec
from fraudgraph.graph.train import GNNConfig
from fraudgraph.models.experiments import holm, verdict
from fraudgraph.models.graph_study import run_graph_study


def test_holm_adjustment() -> None:
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_verdict_needs_both_cis_and_latency() -> None:
    win = {"delta_ap_ci95": [0.01, 0.05], "delta_cost_ci95": [-50.0, -1.0]}
    assert verdict(win, latency_ok=True) == "WIN"
    assert verdict(win, latency_ok=False) == "no significant gain"
    assert verdict({**win, "delta_ap_ci95": [-0.01, 0.05]}, True) == "no significant gain"
    assert verdict({**win, "delta_cost_ci95": [-50.0, 1.0]}, True) == "no significant gain"


def test_mule_ring_injection_adds_labelled_new_destinations() -> None:
    events = to_canonical(generate().frame)
    merged, injected = inject_mule_rings(events, first_step=150, last_step=190, rings=2, seed=1)
    added = merged.filter(pl.col("txn_id").is_in(injected))

    assert merged.height == events.height + 2 * 3 * 15
    assert added["is_fraud"].all()
    assert added["step"].is_between(150, 190).all()
    assert merged["step"].is_sorted()
    assert not added["name_dest"].is_in(events["name_dest"]).any()
    assert (added["amount"] < added["oldbalance_orig"]).all()


def test_graph_study_runs_end_to_end_on_fixture() -> None:
    cfg = SynthConfig()
    events = to_canonical(generate(cfg).frame)
    features, _ = backfill(events, FeatureSpec())
    plan = plan_splits(events, SplitSettings(dense_last_step=(cfg.days - cfg.sparse_days) * 24), 12)
    results = run_graph_study(
        events,
        features,
        plan,
        CostParams(c_review=2_000.0, c_friction=20_000.0),
        seeds=(0, 1),
        trials=2,
        resamples=50,
        gnn=GNNConfig(epochs=2, hidden=16, embedding=8, legit_per_fraud=10),
    )

    assert {"E0", "E1", "E2", "E3", "E5a_sender_balance"} <= set(results["summary"])
    assert results["summary"]["E1"]["seeds"] == 2
    assert set(results["verdicts"]) == {"E1", "E2", "E3"}
    assert "delta_ap_p_holm" in results["comparisons_vs_E0"]["E2"]
    assert "delta_cost_p_holm" in results["comparisons_vs_E0"]["E3"]
    assert "E1_minus_G1-b" not in results["summary"]  # same set as E1_no_G1b
    assert all(np.isfinite(v) for v in results["latency_p95_ms"].values())
    assert results["E4_mule_rings"]["injected_edges"] == 20 * 3 * 15


def test_ring_ids_refuse_sizes_that_would_collide() -> None:
    events = to_canonical(generate().frame)
    with pytest.raises(ValueError, match="fixed-width"):
        inject_mule_rings(events, first_step=150, last_step=190, destinations_per_ring=100)
