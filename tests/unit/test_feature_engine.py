from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from fraudgraph.data.prepare import to_canonical
from fraudgraph.data.synth import generate
from fraudgraph.features.backfill import MODELLED_TYPES, LabelQueue, backfill, events_from
from fraudgraph.features.engine import FeatureEngine
from fraudgraph.features.spec import Event, FeatureSpec
from fraudgraph.features.state import InMemoryStore, OutOfOrderError

SPEC = FeatureSpec()


def engine() -> FeatureEngine:
    return FeatureEngine(SPEC, InMemoryStore(SPEC.horizon_steps))


def ev(step: int, amount: float = 100.0, orig: str = "C1", type_: str = "TRANSFER") -> Event:
    return Event(f"T{step}-{orig}-{amount}", step, type_, amount, orig, "C999")


def test_transaction_features() -> None:
    feats = engine().features(ev(26, amount=10_000_000.0, type_="CASH_OUT"))

    assert feats["t_log_amount"] == pytest.approx(math.log1p(1e7))
    assert feats["t_is_transfer"] == 0.0
    assert feats["t_hour"] == 2.0
    assert feats["t_amount_at_cap"] == 1.0


def test_new_origin_has_empty_history() -> None:
    feats = engine().features(ev(5))

    assert feats["o_first_seen"] == 1.0
    assert feats["o_n_1w"] == 0.0
    assert math.isnan(feats["o_steps_since_last"])
    assert math.isnan(feats["o_amount_z"])


def test_events_in_the_same_step_do_not_see_each_other() -> None:
    e = engine()
    e.apply(ev(10, 50.0))

    feats = e.features(ev(10, 70.0))

    assert feats["o_first_seen"] == 1.0
    assert feats["o_n_1w"] == 0.0


def test_windows_count_only_strictly_earlier_steps() -> None:
    e = engine()
    for step, amount in [(10, 100.0), (20, 200.0), (30, 300.0)]:
        e.apply(ev(step, amount))

    feats = e.features(ev(31, 400.0))

    assert feats["o_n_1h"] == 1.0  # only step 30
    assert feats["o_n_1d"] == 3.0  # steps 7..30
    assert feats["o_log_sum_1d"] == pytest.approx(math.log1p(600.0))
    assert feats["o_steps_since_last"] == 1.0
    assert feats["o_first_seen"] == 0.0
    std = np.std([100.0, 200.0, 300.0])
    assert feats["o_amount_z"] == pytest.approx((400.0 - 200.0) / std)


def test_history_beyond_horizon_is_forgotten_and_evicted() -> None:
    e = engine()
    store = e.store
    assert isinstance(store, InMemoryStore)
    e.apply(ev(1))
    e.apply(ev(2, orig="C2"))

    later = 1 + SPEC.horizon_steps + 1
    feats = e.features(ev(later))
    e.apply(ev(later + 5, orig="C3"))  # advancing time triggers eviction

    assert feats["o_first_seen"] == 1.0
    assert "o:C1" not in store._buckets
    assert set(store._buckets) == {"o:C3", "d:C999"}  # only the newest event is left


def test_features_is_pure() -> None:
    e = engine()
    e.apply(ev(3))
    before = dict(e.store._buckets)  # type: ignore[attr-defined]

    first, second = e.features(ev(4)), e.features(ev(4))

    assert first == second
    assert e.store._buckets == before  # type: ignore[attr-defined]


def test_out_of_order_event_is_rejected() -> None:
    e = engine()
    e.apply(ev(10))
    with pytest.raises(OutOfOrderError):
        e.apply(ev(9, orig="C2"))


def test_spec_rejects_window_longer_than_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        FeatureSpec(windows=(1, 200), horizon_steps=168)


# ---------------------------------------------------------------- backfill on the fixture


@pytest.fixture(scope="module")
def fixture_frame() -> pl.DataFrame:
    return to_canonical(generate().frame)


def test_backfill_covers_modelled_rows_with_aligned_labels(fixture_frame: pl.DataFrame) -> None:
    table, report = backfill(fixture_frame, SPEC)

    modelled = fixture_frame.filter(pl.col("type").is_in(list(MODELLED_TYPES)))
    assert report.events == fixture_frame.height
    assert table.height == modelled.height
    assert table["txn_id"].to_list() == modelled["txn_id"].to_list()
    assert table["is_fraud"].to_list() == modelled["is_fraud"].to_list()
    assert table.columns[3:] == list(FeatureEngine(SPEC, InMemoryStore(168)).feature_names)


def repeating_stream(seed: int = 0, n: int = 3_000) -> pl.DataFrame:
    """Few accounts, many repeats, same-step clashes and gaps longer than the horizon."""
    rng = np.random.default_rng(seed)
    steps = np.sort(rng.integers(1, 500, size=n))
    return pl.DataFrame(
        {
            "txn_id": [f"T{i:07d}" for i in range(n)],
            "step": steps.astype(np.int32),
            "type": rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN"], size=n),
            "amount": np.round(rng.lognormal(10, 1, size=n), 2),
            "name_orig": [f"C{k}" for k in rng.integers(0, 20, size=n)],
            "name_dest": [f"C{1000 + k}" for k in rng.integers(0, 50, size=n)],
            "is_fraud": rng.random(n) < 0.05,
        }
    )


def test_backfill_features_only_use_strictly_earlier_steps() -> None:
    """Recompute features for sampled events from a store fed only with earlier steps:
    they must match the backfill exactly (no peeking at the same or later steps)."""
    frame = repeating_stream()
    table, _ = backfill(frame, SPEC)
    events = list(events_from(frame))
    by_id = {row["txn_id"]: row for row in table.iter_rows(named=True)}
    modelled = [e for e in events if e.type in MODELLED_TYPES]
    sample = [modelled[i] for i in np.random.default_rng(1).choice(len(modelled), 60, False)]
    assert any(by_id[e.txn_id]["o_n_1w"] > 0 for e in sample)  # history is exercised

    fraud = dict(zip(frame["txn_id"].to_list(), frame["is_fraud"].to_list(), strict=True))
    for target in sample:
        fresh = engine()
        labels = LabelQueue(SPEC.label_delay_steps)
        for e in events:
            if e.step >= target.step:
                break
            labels.release(fresh, e.step)
            fresh.apply(e)
            if fraud[e.txn_id]:
                labels.push(e)
        labels.release(fresh, target.step)
        stored = by_id[target.txn_id]
        for name, value in fresh.features(target).items():
            if math.isnan(value):
                assert stored[name] is None or math.isnan(stored[name]), name
            else:
                assert stored[name] == pytest.approx(value, rel=1e-5, abs=1e-5), name
