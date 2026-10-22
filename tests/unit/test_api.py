"""Full API (7.3) and its security (7.7), on the synthetic fixture and in-memory SQLite."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import fakeredis
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from fraudgraph.config import get_settings
from fraudgraph.data.prepare import to_canonical
from fraudgraph.data.synth import generate
from fraudgraph.features.matrix import row_to_vector
from fraudgraph.features.redis_store import RedisStore
from fraudgraph.pipeline import fixture_bundle
from services.api.main import create_app
from services.api.security import ApiKeys, RedactFilter
from tests.api_helpers import KEYS, api_settings, auth

N_EVENTS = 2_500


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return fixture_bundle(tmp_path_factory.mktemp("bundles"), trials=2)


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    frame = to_canonical(generate().frame).head(N_EVENTS)
    return frame.to_dicts()


def request_of(row: dict[str, Any]) -> dict[str, Any]:
    keys = ("txn_id", "step", "type", "amount", "name_orig", "name_dest")
    return {k: row[k] for k in keys}


@pytest.fixture(scope="module")
def stream(bundle: Path, rows: list[dict[str, Any]]) -> TestClient:
    """One app that has scored the first N fixture events (in batches)."""
    client = TestClient(create_app(bundle, api_settings()), headers=auth("write"))
    for start in range(0, len(rows), 1_000):
        batch = [request_of(r) for r in rows[start : start + 1_000]]
        response = client.post("/v1/score/batch", json={"transactions": batch})
        assert response.status_code == 200, response.text
        assert response.json()["failed"] == 0
    return client


@pytest.fixture
def client(bundle: Path) -> TestClient:
    return TestClient(create_app(bundle, api_settings()), headers=auth("write"))


def score(client: TestClient, **overrides: Any) -> dict[str, Any]:
    body = {
        "txn_id": "T1",
        "step": 30,
        "type": "TRANSFER",
        "amount": 181000.0,
        "name_orig": "C1305486145",
        "name_dest": "C553264065",
        **overrides,
    }
    response = client.post("/v1/score", json=body)
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result


# ----------------------------------------------------------------------- auth (7.7)


def test_keys_are_required_and_scoped(bundle: Path) -> None:
    client = TestClient(create_app(bundle, api_settings()))
    body = {"txn_id": "A", "step": 1, "type": "PAYMENT", "amount": 1.0}
    body |= {"name_orig": "C1", "name_dest": "M1"}

    missing = client.post("/v1/score", json=body)
    wrong = client.post("/v1/score", json=body, headers={"X-API-Key": "nope"})
    reader = client.post("/v1/score", json=body, headers=auth("read"))
    writer = client.post("/v1/score", json=body, headers=auth("write"))
    admin_reads = client.get("/v1/models", headers=auth("admin"))

    assert (missing.status_code, missing.json()["error"]["code"]) == (401, "unauthorized")
    assert wrong.status_code == 401
    assert (reader.status_code, reader.json()["error"]["code"]) == (403, "forbidden")
    assert writer.status_code == 200
    assert admin_reads.status_code == 200
    assert client.get("/health").status_code == 200  # ops routes stay open


def test_no_configured_keys_fails_closed(bundle: Path) -> None:
    client = TestClient(create_app(bundle, api_settings(api_keys="")))
    assert client.get("/v1/models", headers=auth("admin")).status_code == 401
    open_client = TestClient(create_app(bundle, api_settings(api_keys="", auth_enabled=False)))
    assert open_client.get("/v1/models").status_code == 200


@pytest.mark.parametrize("entry", ["root:" + "0" * 64, "read:abc", "read"])
def test_bad_key_entries_are_rejected(entry: str) -> None:
    with pytest.raises(ValueError, match="scope"):
        ApiKeys.parse(entry)


def test_keys_are_only_stored_hashed() -> None:
    settings = api_settings()
    assert all(key not in settings.api_keys for key in KEYS.values())


def test_rate_limit_answers_429_with_the_error_model(bundle: Path) -> None:
    client = TestClient(
        create_app(bundle, api_settings(rate_limit_read="2/minute")), headers=auth("read")
    )
    codes = [client.get("/v1/models").status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert client.get("/v1/models").json()["error"]["code"] == "rate_limited"
    other_key = client.get("/v1/models", headers=auth("admin"))
    assert other_key.status_code == 200  # limits are per key


def test_oversized_bodies_are_rejected(bundle: Path) -> None:
    client = TestClient(
        create_app(bundle, api_settings(max_body_bytes=1_024)), headers=auth("write")
    )
    big = {"transactions": [{"txn_id": f"T{i}"} for i in range(200)]}
    response = client.post("/v1/score/batch", json=big)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"

    def chunks() -> Iterator[bytes]:  # no Content-Length: the stream itself is capped
        yield b'{"transactions": ['
        for _ in range(100):
            yield b'{"txn_id": "T"},'
        yield b"]}"

    streamed = client.post(
        "/v1/score/batch", content=chunks(), headers={"content-type": "application/json"}
    )
    assert streamed.status_code == 413


def test_cors_rejects_wildcards(bundle: Path) -> None:
    with pytest.raises(ValueError, match="explicitly"):
        create_app(bundle, api_settings(cors_origins="*"))


def test_cors_preflight_allows_the_key_header(bundle: Path) -> None:
    client = TestClient(create_app(bundle, api_settings(cors_origins="http://localhost:3000")))
    preflight = client.options(
        "/v1/score",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-api-key,content-type",
        },
    )
    assert preflight.status_code == 200
    assert "x-api-key" in preflight.headers["access-control-allow-headers"].lower()


def test_api_keys_are_redacted_from_logs() -> None:
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0, '%s - "GET %s HTTP/1.1" 101', None, None
    )
    record.args = ("127.0.0.1:5000", "/ws/live?api_key=fg_secret&x=1")
    RedactFilter().filter(record)
    assert "fg_secret" not in record.getMessage()
    assert "api_key=***" in record.getMessage()


def test_unknown_routes_use_the_error_model(client: TestClient) -> None:
    response = client.get("/v1/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


# ---------------------------------------------------------------- scoring and reads


def test_scored_transaction_is_stored_with_features(client: TestClient) -> None:
    scored = score(client, txn_id="T42")
    detail = client.get("/v1/transactions/T42").json()

    assert detail["transaction"]["name_dest"] == "C553264065"
    assert detail["prediction"]["p_fraud"] == scored["p_fraud"]
    assert detail["prediction"]["decision"] == scored["decision"]
    assert detail["prediction"]["reasons"] == scored["reasons"]
    assert detail["features"]["g_dest_is_new"] == 1.0
    assert detail["features"]["o_steps_since_last"] is None  # NaN stored as null
    assert detail["label"] is None
    assert client.get("/v1/transactions/NOPE").status_code == 404


def test_retry_is_not_written_twice(client: TestClient) -> None:
    first = score(client, txn_id="R1")
    second = score(client, txn_id="R1", amount=1.0)
    assert first == second
    assert client.get("/v1/transactions/R1").json()["transaction"]["amount"] == 181000.0


def test_batch_reports_failures_per_item(client: TestClient) -> None:
    items = [
        {"txn_id": "B1", "step": 50, "type": "TRANSFER", "amount": 10.0},
        {"txn_id": "B2", "step": 40, "type": "TRANSFER", "amount": 10.0},  # out of order
        {"txn_id": "B3", "step": 51, "type": "PAYMENT", "amount": 10.0},
    ]
    for item in items:
        item |= {"name_orig": "C1", "name_dest": "C2"}
    body = client.post("/v1/score/batch", json={"transactions": items}).json()

    assert (body["scored"], body["failed"]) == (2, 1)
    assert body["results"][1] == {
        "txn_id": "B2",
        "error": {"code": "out_of_order", "message": "step 40 arrived after step 50"},
    }
    assert client.get("/v1/transactions/B3").status_code == 200
    assert client.get("/v1/transactions/B2").status_code == 404


def test_batch_size_is_capped(client: TestClient) -> None:
    item = {"txn_id": "X", "step": 1, "type": "PAYMENT", "amount": 1.0}
    item |= {"name_orig": "C1", "name_dest": "M1"}
    response = client.post("/v1/score/batch", json={"transactions": [item] * 1_001})
    assert response.status_code == 422


def test_explanation_adds_up_to_the_stored_margin(
    stream: TestClient, bundle: Path, rows: list[dict[str, Any]]
) -> None:
    from fraudgraph.serving.scorer import Scorer

    flagged = [
        r for r in rows[-600:] if r["type"] in {"TRANSFER", "CASH_OUT"}
    ]  # modelled types only
    detail = stream.get(f"/v1/transactions/{flagged[0]['txn_id']}").json()
    explained = stream.get(f"/v1/transactions/{flagged[0]['txn_id']}/explain").json()

    scorer = Scorer(bundle)
    feats = {k: np.nan if v is None else v for k, v in detail["features"].items()}
    x = row_to_vector(feats, scorer.features, scorer.spec.horizon_steps)
    margin = float(np.asarray(scorer.booster.predict(x, raw_score=True))[0])
    total = explained["bias"] + sum(explained["contributions"].values())

    assert total == pytest.approx(margin, abs=1e-6)
    assert sum(explained["groups"].values()) == pytest.approx(
        sum(explained["contributions"].values()), abs=1e-9
    )
    stored = detail["prediction"]["reasons"]
    if stored and stored[0]["impact"] > 0:  # feature reasons, not a policy reason
        assert explained["reasons"] == stored


def test_graph_only_looks_back_in_time(stream: TestClient, rows: list[dict[str, Any]]) -> None:
    target = rows[-1]
    graph = stream.get(f"/v1/transactions/{target['txn_id']}/graph?hops=2").json()
    one_hop = stream.get(f"/v1/transactions/{target['txn_id']}/graph?hops=1").json()

    ids = {n["id"] for n in graph["nodes"]}
    assert {target["name_orig"], target["name_dest"]} <= ids
    assert all(e["step"] <= target["step"] for e in graph["edges"])
    assert target["txn_id"] in {e["txn_id"] for e in graph["edges"]}
    assert len(one_hop["edges"]) <= len(graph["edges"])
    centers = {n["id"] for n in graph["nodes"] if n["center"]}
    assert centers == {target["name_orig"], target["name_dest"]}
    assert stream.get(f"/v1/transactions/{target['txn_id']}/graph?hops=3").status_code == 422


# ----------------------------------------------------------------------- cases, labels


def test_review_queue_is_sorted_and_paginated(stream: TestClient) -> None:
    first = stream.get("/v1/cases?limit=5").json()
    second = stream.get("/v1/cases?limit=5&offset=5").json()

    assert first["total"] > 5, "the fixture stream should open review cases"
    priorities = [c["priority"] for c in first["items"] + second["items"]]
    assert priorities == sorted(priorities, reverse=True)
    assert {c["case_id"] for c in first["items"]}.isdisjoint(c["case_id"] for c in second["items"])
    assert all(c["status"] == "open" and c["decision"] == "REVIEW" for c in first["items"])
    assert stream.get("/v1/cases?status=bogus").status_code == 422


def test_resolving_a_case_creates_an_analyst_label(stream: TestClient) -> None:
    case = stream.get("/v1/cases?limit=1").json()["items"][0]
    resolved = stream.post(
        f"/v1/cases/{case['case_id']}/resolve",
        json={"verdict": "fraud", "assignee": "analyst-1", "notes": "mule ring"},
    )
    body = resolved.json()

    assert resolved.status_code == 200, resolved.text
    assert body["case"]["status"] == "resolved"
    assert body["case"]["verdict"] == "fraud"
    assert body["label"] | {"labelled_at_step": 0} == {
        "txn_id": case["txn_id"],
        "is_fraud": True,
        "source": "analyst",
        "labelled_at_step": 0,
    }
    assert body["label"]["labelled_at_step"] >= case["step"]
    again = stream.post(f"/v1/cases/{case['case_id']}/resolve", json={"verdict": "legit"})
    assert again.json()["error"]["code"] == "already_resolved"
    assert stream.get("/v1/cases/999999").status_code == 404
    resolved_list = stream.get("/v1/cases?status=resolved").json()
    assert case["case_id"] in {c["case_id"] for c in resolved_list["items"]}


def test_labels_are_idempotent_and_reach_the_feature_state(
    client: TestClient,
) -> None:
    score(client, txn_id="L1", step=10, name_dest="C77")
    assert (
        client.post(
            "/v1/labels", json={"txn_id": "nope", "observed_fraud": True, "step": 40}
        ).json()["error"]["code"]
        == "unknown_txn"
    )
    early = client.post("/v1/labels", json={"txn_id": "L1", "observed_fraud": True, "step": 9})
    assert early.status_code == 422

    first = client.post("/v1/labels", json={"txn_id": "L1", "observed_fraud": True, "step": 34})
    second = client.post("/v1/labels", json={"txn_id": "L1", "observed_fraud": False, "step": 35})
    assert (first.status_code, first.json()["created"]) == (201, True)
    assert (second.status_code, second.json()["created"]) == (200, False)
    assert second.json()["label"]["is_fraud"] is True  # first label wins

    later = score(client, txn_id="L2", step=36, name_orig="C5", name_dest="C77")
    detail = client.get("/v1/transactions/L2").json()
    assert later["txn_id"] == "L2"
    assert detail["features"]["g_dest_known_fraud_in"] == 1.0


def test_performance_counts_labelled_decisions(client: TestClient) -> None:
    score(client, txn_id="P1", step=10, name_dest="C90")
    client.post("/v1/labels", json={"txn_id": "P1", "observed_fraud": True, "step": 34})
    body = client.get("/v1/monitoring/performance?window=7d").json()

    (champion,) = body["models"]
    assert champion["role"] == "champion"
    assert champion["n_labelled"] == 1
    assert champion["tp"] + champion["fn"] == 1
    assert body["flagged"] == ["STEP_UP", "REVIEW", "BLOCK"]
    assert client.get("/v1/monitoring/performance?window=0h").status_code == 422


# ------------------------------------------------------------- monitoring, models, sim


def test_drift_is_empty_until_the_worker_runs(client: TestClient) -> None:
    body = client.get("/v1/monitoring/drift?window=24h").json()
    assert body == {"window_steps": 24, "since_step": 0, "windows": [], "alarms": []}


def test_reports_are_served_from_the_reports_dir(
    bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FG_PATHS__REPORTS_DIR", str(tmp_path))
    get_settings.cache_clear()
    (tmp_path / "evidently").mkdir()
    (tmp_path / "evidently" / "d3_window_12.html").write_text("<html>ok</html>")
    client = TestClient(create_app(bundle, api_settings()), headers=auth("read"))

    report = client.get("/v1/monitoring/reports/d3_window_12")
    assert report.status_code == 200
    assert report.text == "<html>ok</html>"
    assert client.get("/v1/monitoring/reports/d3_window_12?format=json").status_code == 404
    assert client.get("/v1/monitoring/reports/..%2Fsecret").status_code in {404, 422}


def test_models_list_the_served_champion(client: TestClient, bundle: Path) -> None:
    models = client.get("/v1/models").json()
    current = client.get("/v1/models/current").json()

    assert [(m["version"], m["stage"]) for m in models] == [(bundle.name, "champion")]
    assert current["version"] == bundle.name
    assert "t_log_amount" in current["features"]


def test_admin_routes_need_admin_and_are_pending(bundle: Path) -> None:
    client = TestClient(create_app(bundle, api_settings()))
    for path, body in (
        ("/v1/models/v1/promote", None),
        ("/v1/sim/start", None),
        ("/v1/sim/stop", None),
        ("/v1/sim/scenario", {"name": "D3"}),
    ):
        assert client.post(path, json=body, headers=auth("write")).status_code == 403
        pending = client.post(path, json=body, headers=auth("admin"))
        assert (pending.status_code, pending.json()["error"]["code"]) == (501, "not_implemented")


def test_ready_reports_each_backend(client: TestClient) -> None:
    body = client.get("/ready").json()
    assert body["checks"] == {"model": True, "db": True}


# ----------------------------------------------------------------------- WebSocket


def test_live_stream_pushes_decisions(client: TestClient) -> None:
    with client.websocket_connect(f"/ws/live?api_key={KEYS['read']}") as ws:
        score(client, txn_id="W1")
        message = ws.receive_json()
    assert message["kind"] == "decision"
    assert message["txn_id"] == "W1"


def test_live_stream_needs_a_read_key(bundle: Path) -> None:
    client = TestClient(create_app(bundle, api_settings()))  # no default key header
    with client.websocket_connect("/ws/live") as ws, pytest.raises(WebSocketDisconnect) as closed:
        ws.receive_json()
    assert closed.value.code == 1008


# --------------------------------------------------------------- Redis-backed state


def test_retry_on_another_worker_does_not_double_count(bundle: Path) -> None:
    server = fakeredis.FakeServer()

    def worker() -> TestClient:
        store = RedisStore(fakeredis.FakeRedis(server=server), 168)
        return TestClient(create_app(bundle, api_settings(), store=store), headers=auth("write"))

    a, b = worker(), worker()
    score(a, txn_id="D1", step=5, name_dest="C9")
    score(b, txn_id="D1", step=5, name_dest="C9")  # retried on another worker
    store = RedisStore(fakeredis.FakeRedis(server=server), 168)
    (bucket,) = store.history("d:C9", 6)
    assert bucket.n == 1


# ------------------------------------------------------------------------- contract


A7_PATHS = {
    "/v1/score",
    "/v1/score/batch",
    "/v1/transactions/{txn_id}",
    "/v1/transactions/{txn_id}/explain",
    "/v1/transactions/{txn_id}/graph",
    "/v1/cases",
    "/v1/cases/{case_id}/resolve",
    "/v1/labels",
    "/v1/monitoring/drift",
    "/v1/monitoring/performance",
    "/v1/monitoring/reports/{report_id}",
    "/v1/models",
    "/v1/models/current",
    "/v1/models/{version}/promote",
    "/v1/sim/start",
    "/v1/sim/stop",
    "/v1/sim/scenario",
    "/health",
    "/ready",
}


def test_openapi_documents_every_route_with_tags_and_auth(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert set(schema["paths"]) >= A7_PATHS
    assert schema["components"]["securitySchemes"]["ApiKey"]["name"] == "X-API-Key"
    for path, methods in schema["paths"].items():
        for op in methods.values():
            assert op["tags"], path
            if path.startswith("/v1/"):
                assert "401" in op["responses"], path
    examples = schema["components"]["schemas"]["ScoreRequest"]["examples"]
    assert examples[0]["txn_id"] == "T000123"
