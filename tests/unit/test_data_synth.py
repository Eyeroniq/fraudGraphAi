from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pytest

from fraudgraph.data.prepare import prepare
from fraudgraph.data.synth import SynthConfig, SynthResult, generate, write_csv

# sha256 of the default fixture CSV. If this changes, every golden number built on the
# fixture changes too: update it only on purpose (e.g. a deliberate generator change).
GOLDEN_SHA256 = "f814975c2949a6c7a0fbfc8c6c8606eeea146e1cbffdb8c85abbfed4fda31e02"


@pytest.fixture(scope="module")
def result() -> SynthResult:
    return generate()


@pytest.fixture(scope="module")
def canonical(result: SynthResult, tmp_path_factory: pytest.TempPathFactory) -> pl.DataFrame:
    """The fixture after the real prepare step (so it also passes the PaySim schema)."""
    root = tmp_path_factory.mktemp("synth")
    parquet, _, _ = prepare(write_csv(result, root / "raw.csv"), root / "out")
    return pl.read_parquet(parquet)


def test_rebuilds_byte_identically_from_seed(tmp_path: Path) -> None:
    first = write_csv(generate(), tmp_path / "a.csv").read_bytes()
    second = write_csv(generate(), tmp_path / "b.csv").read_bytes()

    assert first == second
    assert hashlib.sha256(first).hexdigest() == GOLDEN_SHA256


def test_other_seed_gives_other_data() -> None:
    assert not generate(SynthConfig(seed=8)).frame.equals(generate().frame)


def test_size_and_fraud_only_in_modelled_types(canonical: pl.DataFrame) -> None:
    assert 18_000 <= canonical.height <= 22_000
    fraud_types = set(canonical.filter("is_fraud")["type"].unique().to_list())
    assert fraud_types == {"TRANSFER", "CASH_OUT"}


def test_planted_ring_is_present(result: SynthResult, canonical: pl.DataFrame) -> None:
    cfg = SynthConfig()
    assert len(result.ring_destinations) == cfg.rings * cfg.ring_destinations

    for dest in result.ring_destinations:
        start, end = result.ring_bursts[dest]
        incoming = canonical.filter(pl.col("name_dest") == dest)
        # Brand new: nothing reaches it before the burst; everything arrives inside it.
        assert incoming["step"].is_between(start, end).all()
        assert incoming["name_orig"].n_unique() == cfg.ring_senders
        assert incoming["is_fraud"].all()
        # Ring transfers do not drain the sender: only graph features can see them.
        assert (incoming["amount"] < incoming["oldbalance_orig"]).all()


def test_ring_stands_out_on_burst_in_degree(result: SynthResult, canonical: pl.DataFrame) -> None:
    """No legit destination gets as many distinct senders inside a burst window."""
    window = SynthConfig().ring_burst_steps
    edges = canonical.filter(pl.col("type").is_in(["TRANSFER", "CASH_OUT"]))
    starts = edges.select("name_dest", pl.col("step").alias("start")).unique()
    burst_in_degree = (
        edges.join(starts, on="name_dest")
        .filter(pl.col("step").is_between(pl.col("start"), pl.col("start") + window - 1))
        .group_by("name_dest", "start")
        .agg(senders=pl.col("name_orig").n_unique())
        .group_by("name_dest")
        .agg(pl.col("senders").max())
    )
    ring = burst_in_degree.filter(pl.col("name_dest").is_in(result.ring_destinations))
    others = burst_in_degree.filter(~pl.col("name_dest").is_in(result.ring_destinations))

    assert ring["senders"].min() > others["senders"].max()


def test_drain_artifact_and_single_use_origins(canonical: pl.DataFrame) -> None:
    drains = canonical.with_columns(drain=pl.col("amount") == pl.col("oldbalance_orig"))
    assert drains.filter(~pl.col("is_fraud"))["drain"].sum() == 0
    assert drains.filter("is_fraud")["drain"].sum() > 0
    assert canonical["name_orig"].n_unique() == canonical.height


def test_daily_cycle_for_legit_but_not_fraud(canonical: pl.DataFrame) -> None:
    hourly = canonical.with_columns(hour=pl.col("step") % 24)
    legit = hourly.filter(~pl.col("is_fraud"))
    day_hours = legit.filter(pl.col("hour").is_between(10, 20)).height
    night_hours = legit.filter(pl.col("hour").is_between(2, 6)).height
    assert day_hours > 50 * night_hours

    drain_fraud = hourly.filter(pl.col("is_fraud") & (pl.col("type") == "CASH_OUT"))
    assert drain_fraud.filter(pl.col("hour").is_between(2, 6)).height > 0


def test_late_days_are_sparse_for_legit_only(canonical: pl.DataFrame) -> None:
    cfg = SynthConfig()
    dense_last_step = (cfg.days - cfg.sparse_days) * 24
    late = canonical.filter(pl.col("step") > dense_last_step)
    dense = canonical.filter(pl.col("step") <= dense_last_step)

    assert late["is_fraud"].mean() > 5 * dense["is_fraud"].mean()  # type: ignore[operator]
