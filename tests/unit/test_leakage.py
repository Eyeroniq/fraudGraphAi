"""Leakage and parity tests (task 3.3). Results are only trusted if these pass."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fraudgraph.data.prepare import to_canonical
from fraudgraph.data.synth import SynthConfig, generate
from fraudgraph.features.backfill import MODELLED_TYPES, LabelQueue, backfill, events_from
from fraudgraph.features.engine import FeatureEngine
from fraudgraph.features.novelty import destination_novelty
from fraudgraph.features.spec import Event, FeatureSpec
from fraudgraph.features.state import InMemoryStore

SPEC = FeatureSpec()


def fresh() -> FeatureEngine:
    return FeatureEngine(SPEC, InMemoryStore(SPEC.horizon_steps))


def same(a: dict[str, float], b: dict[str, float]) -> bool:
    return all((math.isnan(a[k]) and math.isnan(b[k])) or a[k] == pytest.approx(b[k]) for k in a)


# ------------------------------------------------------------------ property tests

accounts = st.sampled_from([f"C{i}" for i in range(6)])
event_st = st.builds(
    lambda step, typ, amount, orig, dest: (step, typ, amount, orig, dest),
    st.integers(1, 60),
    st.sampled_from(["TRANSFER", "CASH_OUT", "PAYMENT"]),
    st.floats(1, 1e6, allow_nan=False),
    accounts,
    accounts,
)


@settings(max_examples=150, deadline=None)
@given(
    history=st.lists(event_st, min_size=1, max_size=40),
    extra=st.lists(event_st, max_size=8),
    target=event_st,
)
def test_same_step_events_and_labels_never_change_a_score(history, extra, target) -> None:  # type: ignore[no-untyped-def]
    """Features of `target` are fixed by strictly earlier steps: applying any events, or
    learning any labels, at the target's own step before scoring it changes nothing."""
    t_step = max(s for s, *_ in history) + 1
    tgt = Event("TGT", t_step, target[1], target[2], target[3], target[4])
    past = sorted(history)

    engine_a, engine_b = fresh(), fresh()
    for i, (step, typ, amount, orig, dest) in enumerate(past):
        for e in (engine_a, engine_b):
            e.apply(Event(f"H{i}", step, typ, amount, orig, dest))

    before = engine_a.features(tgt)
    for i, (_, typ, amount, orig, dest) in enumerate(extra):
        engine_b.apply(Event(f"X{i}", t_step, typ, amount, orig, dest))
        engine_b.apply_label(dest, True, t_step)
    after = engine_b.features(tgt)

    assert same(before, after)


@settings(max_examples=60, deadline=None)
@given(steps=st.lists(st.integers(1, 50), min_size=2, max_size=30))
def test_features_only_depend_on_strictly_earlier_steps(steps: list[int]) -> None:
    """Scoring the last event after a full replay equals scoring it from a replay cut
    just before its step."""
    events = [
        Event(f"E{i}", s, "TRANSFER", 100.0 + i, f"C{i % 3}", f"C{(i * 7) % 4}")
        for i, s in enumerate(sorted(steps))
    ]
    target = events[-1]
    full, cut = fresh(), fresh()
    for e in events[:-1]:
        full.apply(e)
        if e.step < target.step:
            cut.apply(e)
    assert same(full.features(target), cut.features(target))


# ------------------------------------------------------------------ label delay


def test_label_is_invisible_before_delay_and_visible_after() -> None:
    delay = SPEC.label_delay_steps
    engine = fresh()
    labels = LabelQueue(delay)
    fraud = Event("F", 10, "TRANSFER", 500.0, "C1", "MULE")
    engine.apply(fraud)
    labels.push(fraud)

    seen = {}
    for step in (11, 10 + delay, 10 + delay + 1):
        labels.release(engine, step)
        probe = Event(f"P{step}", step, "TRANSFER", 1.0, f"C{step}", "MULE")
        seen[step] = engine.features(probe)["g_dest_known_fraud_in"]
        engine.apply(probe)

    assert seen[11] == 0.0  # label not known yet
    assert seen[10 + delay] == 0.0  # known at s + L, visible only from the next step
    assert seen[10 + delay + 1] == 1.0


# ------------------------------------------------------------------ parity


@pytest.fixture(scope="module")
def fixture_events() -> pl.DataFrame:
    return to_canonical(generate(SynthConfig()).frame)


@pytest.fixture(scope="module")
def fixture_features(fixture_events: pl.DataFrame) -> pl.DataFrame:
    table, _ = backfill(fixture_events, SPEC)
    return table


def test_engine_novelty_matches_bulk_novelty(
    fixture_events: pl.DataFrame, fixture_features: pl.DataFrame
) -> None:
    bulk = destination_novelty(fixture_events, SPEC.horizon_steps)
    joined = fixture_features.select("txn_id", "g_dest_is_new").join(bulk, on="txn_id")
    assert joined.height == fixture_features.height
    assert (joined["g_dest_is_new"] == joined["dest_is_new"].cast(pl.Float32)).all()


def test_backfill_equals_online_replay_for_2000_events(
    fixture_events: pl.DataFrame, fixture_features: pl.DataFrame
) -> None:
    """The API path (event by event, labels as they arrive) reproduces the backfill."""
    head = fixture_events.head(2_000)
    engine = fresh()
    labels = LabelQueue(SPEC.label_delay_steps)
    stored = {r["txn_id"]: r for r in fixture_features.iter_rows(named=True)}
    checked = 0
    for ev, is_fraud in zip(events_from(head), head["is_fraud"].to_list(), strict=True):
        labels.release(engine, ev.step)
        if ev.type in MODELLED_TYPES:
            online = engine.features(ev)
            for name, value in online.items():
                expected = stored[ev.txn_id][name]
                if math.isnan(value):
                    assert expected is None or math.isnan(expected), name
                else:  # backfill stores float32
                    assert expected == pytest.approx(value, rel=1e-6, abs=1e-6), name
            checked += 1
        engine.apply(ev)
        if is_fraud:
            labels.push(ev)
    assert checked > 500


def test_planted_ring_has_many_distinct_senders(fixture_features: pl.DataFrame) -> None:
    result = generate(SynthConfig())
    events = to_canonical(result.frame).select("txn_id", "name_dest")
    feats = fixture_features.join(events, on="txn_id")
    ring = feats.filter(pl.col("name_dest").is_in(result.ring_destinations))
    legit = feats.filter(~pl.col("is_fraud"))

    ring_last = ring.group_by("name_dest").agg(pl.col("g_dest_senders_1d").max())
    # Same-step senders are invisible to each other, so the last ring payment sees at
    # least the senders of the earlier burst steps: about half of them or more.
    assert ring_last["g_dest_senders_1d"].min() >= SynthConfig().ring_senders // 2
    assert ring["g_dest_senders_1d"].mean() > 3 * legit["g_dest_senders_1d"].mean()  # type: ignore[operator]
    # A day-long window also catches a few busy legit receivers, so compare the typical
    # ring destination with the legit 99th percentile rather than the weakest ring.
    legit_p99 = float(np.percentile(legit["g_dest_senders_1d"].to_numpy(), 99))
    assert float(ring_last["g_dest_senders_1d"].median()) > legit_p99  # type: ignore[arg-type]
