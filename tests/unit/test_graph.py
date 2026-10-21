from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch

from fraudgraph.data.prepare import to_canonical
from fraudgraph.data.synth import SynthConfig, generate
from fraudgraph.graph.models import TemporalEdgeGNN
from fraudgraph.graph.sampler import NEIGHBOUR_FEATURES, TemporalNeighbours
from fraudgraph.graph.train import GNNConfig, fit, out_of_fold_embeddings, predict

STEPS_AGO = NEIGHBOUR_FEATURES.index("steps_ago")
KNOWN = NEIGHBOUR_FEATURES.index("known_fraud")


def tiny() -> pl.DataFrame:
    #   txn  step dest  fraud
    rows = [
        ("T0", 1, "D", True),
        ("T1", 2, "D", False),
        ("T2", 2, "D", False),  # same step as T1
        ("T3", 30, "D", False),
        ("T4", 190, "D", False),  # 160 steps after T3 (in), 189 after T0 (out)
        ("T5", 5, "E", False),
    ]
    return pl.DataFrame(
        {
            "txn_id": [r[0] for r in rows],
            "step": [r[1] for r in rows],
            "type": ["TRANSFER"] * len(rows),
            "amount": [100.0] * len(rows),
            "name_dest": [r[2] for r in rows],
            "is_fraud": [r[3] for r in rows],
        }
    )


@pytest.fixture
def neighbours() -> TemporalNeighbours:
    return TemporalNeighbours(tiny(), horizon_steps=168, label_delay_steps=24)


def test_neighbours_are_strictly_earlier_same_destination(neighbours: TemporalNeighbours) -> None:
    batch = neighbours.batch(np.array(["T2", "T3", "T5"]), k=4)
    t2, t3, t5 = batch.mask.sum(axis=1)
    assert t2 == 1  # only T0: T1 shares its step
    assert t3 == 3  # T0, T1, T2
    assert t5 == 0  # other destination


def test_horizon_drops_old_neighbours(neighbours: TemporalNeighbours) -> None:
    batch = neighbours.batch(np.array(["T4"]), k=4)
    assert batch.mask.sum() == 1  # only T3: T0..T2 are more than 168 steps back


def test_last_k_are_the_most_recent(neighbours: TemporalNeighbours) -> None:
    batch = neighbours.batch(np.array(["T3"]), k=2)
    steps_ago = batch.neighbours[0, :, STEPS_AGO] * 168
    assert np.allclose(sorted(steps_ago), [28, 28])  # T1 and T2, not T0


def test_neighbour_label_respects_delay(neighbours: TemporalNeighbours) -> None:
    # T0 is fraud at step 1: known at 25, visible from 26. T3 is at step 30 -> visible.
    batch = neighbours.batch(np.array(["T1", "T3"]), k=4)
    assert batch.neighbours[0, :, KNOWN].sum() == 0  # step 2: not known yet
    assert batch.neighbours[1, :, KNOWN].sum() == 1


def test_unknown_txn_id_is_an_error(neighbours: TemporalNeighbours) -> None:
    with pytest.raises(KeyError):
        neighbours.batch(np.array(["NOPE"]), k=2)


@pytest.mark.parametrize("conv", ["sage", "gine"])
def test_model_shapes_and_empty_neighbourhoods(conv: str) -> None:
    model = TemporalEdgeGNN(conv, hidden=8, embedding=4)
    edge = torch.randn(3, 5)
    nbs = torch.randn(3, 4, len(NEIGHBOUR_FEATURES))
    mask = torch.tensor([[True, True, False, False], [False] * 4, [True] * 4])
    assert model(edge, nbs, mask).shape == (3,)
    assert model.embed(edge, nbs, mask).shape == (3, 4)


@pytest.fixture(scope="module")
def fixture_setup() -> tuple[TemporalNeighbours, pl.DataFrame]:
    cfg = SynthConfig()
    events = to_canonical(generate(cfg).frame)
    neighbours = TemporalNeighbours(events, horizon_steps=168, label_delay_steps=12)
    modelled = events.filter(pl.col("type").is_in(["TRANSFER", "CASH_OUT"]))
    return neighbours, modelled


def test_gnn_learns_on_fixture(fixture_setup: tuple[TemporalNeighbours, pl.DataFrame]) -> None:
    neighbours, modelled = fixture_setup
    train = modelled.filter(pl.col("step") <= 120)
    val = modelled.filter(pl.col("step").is_between(121, 192))
    config = GNNConfig(epochs=8, hidden=32, embedding=8, legit_per_fraud=20, seed=0)

    model, report = fit(neighbours, train["txn_id"].to_numpy(), val["txn_id"].to_numpy(), config)
    prob, emb = predict(model, neighbours.batch(val["txn_id"].to_numpy(), config.k))

    prevalence = float(val["is_fraud"].mean())  # type: ignore[arg-type]
    assert report.best_val_ap > 2 * prevalence
    assert emb.shape == (val.height, 8)
    assert np.isfinite(prob).all()


def test_out_of_fold_embeddings_skip_the_first_fold(
    fixture_setup: tuple[TemporalNeighbours, pl.DataFrame],
) -> None:
    neighbours, modelled = fixture_setup
    train = modelled.filter(pl.col("step") <= 120)
    val = modelled.filter(pl.col("step").is_between(121, 192))
    config = GNNConfig(epochs=2, hidden=8, embedding=4, legit_per_fraud=5)

    emb, has = out_of_fold_embeddings(
        neighbours,
        train["txn_id"].to_numpy(),
        train["step"].to_numpy(),
        val["txn_id"].to_numpy(),
        config,
        folds=3,
        embargo_steps=12,
    )
    first_fold = train["step"].to_numpy() < np.quantile(train["step"].to_numpy(), 1 / 3)
    assert not has[first_fold].any()
    assert has[~first_fold].mean() > 0.9
    assert np.abs(emb[~has]).sum() == 0
