"""RedisStore (task 7.1): same answers as InMemoryStore, atomic, idempotent.

Runs on fakeredis (with its Lua engine) everywhere, and on a real Redis under the
`integration` marker (CI service container).
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterator

import fakeredis
import pytest
import redis

from fraudgraph.data.prepare import to_canonical
from fraudgraph.data.synth import generate
from fraudgraph.features.backfill import LabelQueue, events_from
from fraudgraph.features.engine import FeatureEngine
from fraudgraph.features.redis_store import RedisStore
from fraudgraph.features.spec import Event, FeatureSpec
from fraudgraph.features.state import InMemoryStore, OutOfOrderError
from tests.backends import service_url

SPEC = FeatureSpec()


@pytest.fixture(params=["fake", pytest.param("real", marks=pytest.mark.integration)])
def client(request: pytest.FixtureRequest) -> Iterator[redis.Redis]:
    if request.param == "fake":
        yield fakeredis.FakeRedis()
        return
    real = redis.Redis.from_url(service_url("TEST_REDIS_URL"))
    yield real
    real.close()


@pytest.fixture
def store(client: redis.Redis) -> Iterator[RedisStore]:
    prefix = f"test:{uuid.uuid4().hex[:8]}:"  # isolated namespace on a shared server
    yield RedisStore(client, SPEC.horizon_steps, prefix=prefix)
    for key in client.scan_iter(f"{prefix}*"):
        client.delete(key)


def same(a: dict[str, float], b: dict[str, float]) -> bool:
    return a.keys() == b.keys() and all(
        (math.isnan(a[k]) and math.isnan(b[k])) or a[k] == pytest.approx(b[k], rel=1e-12) for k in a
    )


def test_features_match_the_in_memory_store(store: RedisStore) -> None:
    """Parity (3.3) on the synthetic fixture: every event, with delayed labels (a short
    delay so labels land inside the sample; fakeredis runs Lua slowly)."""
    frame = to_canonical(generate().frame).head(1_200)
    fraud = set(frame.filter(frame["is_fraud"])["txn_id"].to_list())
    memory = FeatureEngine(SPEC, InMemoryStore(SPEC.horizon_steps))
    remote = FeatureEngine(SPEC, store)
    queues = LabelQueue(2), LabelQueue(2)

    for ev in events_from(frame):
        for engine, queue in zip((memory, remote), queues, strict=True):
            queue.release(engine, ev.step)
        assert same(memory.features(ev), remote.features(ev)), ev.txn_id
        for engine, queue in zip((memory, remote), queues, strict=True):
            engine.apply(ev)
            if ev.txn_id in fraud:
                queue.push(ev)

    assert fraud
    assert store.latest_step == memory.store.latest_step
    assert list(store.client.scan_iter(f"{store.prefix}h:f:*"))  # labels reached Redis


def test_retried_event_is_counted_once(store: RedisStore) -> None:
    engine = FeatureEngine(SPEC, store)
    ev = Event("T1", 5, "TRANSFER", 100.0, "C1", "C2")
    assert engine.apply(ev) is True
    assert engine.apply(ev) is False
    (bucket,) = store.history("d:C2", 6)
    assert (bucket.n, bucket.total, bucket.members) == (1, 100.0, ["C1"])


def test_labels_are_counted_once_per_txn(store: RedisStore) -> None:
    engine = FeatureEngine(SPEC, store)
    for _ in range(2):
        engine.apply_label("C2", True, 7, txn_id="T1")
    engine.apply_label("C2", True, 7, txn_id="T2")
    assert sum(b.n for b in store.history("f:C2", 8)) == 2


def test_out_of_order_step_is_rejected_without_writing(store: RedisStore) -> None:
    engine = FeatureEngine(SPEC, store)
    engine.apply(Event("T1", 10, "TRANSFER", 1.0, "C1", "C2"))
    with pytest.raises(OutOfOrderError):
        engine.apply(Event("T2", 9, "TRANSFER", 1.0, "C3", "C4"))
    assert store.history("d:C4", 11) == []
    assert engine.apply(Event("T2", 10, "TRANSFER", 1.0, "C3", "C4")) is True  # not marked


def test_same_step_bucket_accumulates(store: RedisStore) -> None:
    for i, amount in enumerate((3.0, 4.0)):
        store.apply(f"T{i}", [("d:C9", 2, amount, f"C{i}")])
    (bucket,) = store.history("d:C9", 3)
    assert (bucket.n, bucket.total, bucket.total_sq) == (2, 7.0, 25.0)
    assert bucket.members == ["C0", "C1"]
    assert store.history("d:C9", 2) == []  # same step never sees itself


def test_old_buckets_are_trimmed_and_keys_expire(store: RedisStore, client: redis.Redis) -> None:
    store.add("d:C1", 1, 1.0)
    store.add("d:C1", 1 + SPEC.horizon_steps + 1, 2.0)
    fields = client.hkeys(f"{store.prefix}h:d:C1")
    assert not any(f.startswith(b"1:") for f in fields)
    assert 0 < client.ttl(f"{store.prefix}h:d:C1") <= store.ttl_seconds
    store.apply("T1", [("d:C2", 200, 1.0, None)])
    assert 0 < client.ttl(f"{store.prefix}seen:T1") <= store.seen_ttl_seconds


def test_writes_of_one_unit_share_a_step(store: RedisStore) -> None:
    with pytest.raises(ValueError, match="share a step"):
        store.apply("T1", [("d:C1", 1, 1.0, None), ("d:C2", 2, 1.0, None)])
