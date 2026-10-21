from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraudgraph.data.prepare import to_canonical
from fraudgraph.data.synth import generate
from fraudgraph.features.backfill import MODELLED_TYPES, events_from
from fraudgraph.features.spec import Event
from fraudgraph.pipeline import fixture_bundle
from fraudgraph.serving.scorer import BundleMismatchError, Scorer
from services.api.main import create_app, latest_bundle


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return fixture_bundle(tmp_path_factory.mktemp("bundles"), trials=2)


@pytest.fixture(scope="module")
def events() -> list[Event]:
    return list(events_from(to_canonical(generate().frame)))


def test_scorer_matches_offline_pipeline(bundle: Path, events: list[Event]) -> None:
    """Online scoring (engine + bundle) equals scoring a batch-backfilled row."""
    from fraudgraph.features.backfill import backfill
    from fraudgraph.features.matrix import frame_to_matrix
    from fraudgraph.features.spec import FeatureSpec

    scorer = Scorer(bundle)
    online: dict[str, float] = {}
    for ev in events[:4_000]:
        if ev.type in MODELLED_TYPES:
            online[ev.txn_id] = scorer.probability(ev)
        scorer.engine.apply(ev)

    frame = to_canonical(generate().frame).head(4_000)
    table, _ = backfill(frame, FeatureSpec())
    x = frame_to_matrix(table, scorer.features, scorer.spec.horizon_steps)
    offline = scorer.calibrator(np.asarray(scorer.booster.predict(x)))

    assert list(online) == table["txn_id"].to_list()
    assert np.allclose(list(online.values()), offline, rtol=1e-9, atol=1e-12)


def test_decision_has_expected_cost_for_every_action(bundle: Path, events: list[Event]) -> None:
    scorer = Scorer(bundle)
    modelled = next(e for e in events if e.type in MODELLED_TYPES)
    decision = scorer.score(modelled)

    assert decision.decision in {"APPROVE", "REVIEW", "BLOCK"}
    assert set(decision.expected_cost) == {"APPROVE", "REVIEW", "BLOCK"}
    assert decision.expected_cost[decision.decision] == min(decision.expected_cost.values())
    assert 0.0 <= decision.p_fraud <= 1.0
    assert decision.model_version == bundle.name


def test_unmodelled_types_are_approved_but_update_state(bundle: Path) -> None:
    scorer = Scorer(bundle)
    payment = Event("T1", 5, "PAYMENT", 10.0, "C1", "M1")
    decision = scorer.score(payment)

    assert (decision.decision, decision.decision_reason) == ("APPROVE", "type_not_modelled")
    assert scorer.engine.store.history("o:C1", 6)


def test_engine_version_mismatch_is_refused(bundle: Path, tmp_path: Path) -> None:
    copy = tmp_path / bundle.name
    shutil.copytree(bundle, copy)
    spec_path = copy / "feature_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["engine_version"] = "0.0.1"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    # Keep the manifest hash valid so only the version check can fail.
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    import hashlib

    manifest["files"]["feature_spec.json"] = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BundleMismatchError):
        Scorer(copy)


# ------------------------------------------------------------------------ API


def payload(**overrides: object) -> dict[str, object]:
    body = {
        "txn_id": "T9000001",
        "step": 30,
        "type": "TRANSFER",
        "amount": 181000.0,
        "name_orig": "C1305486145",
        "name_dest": "C553264065",
    }
    return {**body, **overrides}


@pytest.fixture
def client(bundle: Path) -> TestClient:
    return TestClient(create_app(bundle))


def test_health_and_ready(client: TestClient, bundle: Path) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["model_version"] == bundle.name


def test_score_returns_a_decision(client: TestClient) -> None:
    response = client.post("/v1/score", json=payload())
    body = response.json()

    assert response.status_code == 200
    assert body["txn_id"] == "T9000001"
    assert body["decision"] in {"APPROVE", "REVIEW", "BLOCK"}
    assert body["reasons"] == []
    assert body["latency_ms"] >= 0


def test_score_is_idempotent_on_txn_id(client: TestClient) -> None:
    first = client.post("/v1/score", json=payload(txn_id="T1"))
    second = client.post("/v1/score", json=payload(txn_id="T1", amount=5.0))
    assert first.json() == second.json()


def test_validation_errors_use_the_error_model(client: TestClient) -> None:
    response = client.post("/v1/score", json=payload(amount=-1))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert "amount" in response.json()["error"]["message"]

    bad_id = client.post("/v1/score", json=payload(name_dest="X1"))
    assert bad_id.status_code == 422


def test_out_of_order_step_is_rejected(client: TestClient) -> None:
    assert client.post("/v1/score", json=payload(txn_id="A", step=50)).status_code == 200
    late = client.post("/v1/score", json=payload(txn_id="B", step=40))
    assert late.status_code == 409
    assert late.json()["error"]["code"] == "out_of_order"


def test_not_ready_without_a_bundle(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "missing"))
    assert client.get("/ready").status_code == 503
    response = client.post("/v1/score", json=payload())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "not_ready"


def test_latest_bundle_picks_newest_manifest(tmp_path: Path) -> None:
    for name, created in (("a", "2026-01-01T00:00:00+00:00"), ("b", "2026-02-01T00:00:00+00:00")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.json").write_text(json.dumps({"created_at": created}))
    assert latest_bundle(tmp_path) == tmp_path / "b"
    assert latest_bundle(tmp_path / "empty") is None


def test_openapi_documents_the_score_contract(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/v1/score" in schema["paths"]
    props = schema["components"]["schemas"]["ScoreResponse"]["properties"]
    assert {"p_fraud", "decision", "expected_cost", "model_version"} <= set(props)


def test_cors_allows_only_the_configured_origins(
    bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FG_CORS_ORIGINS", "http://localhost:3000")
    client = TestClient(create_app(bundle))
    preflight = {"Access-Control-Request-Method": "POST"}

    ok = client.options("/v1/score", headers={"Origin": "http://localhost:3000", **preflight})
    bad = client.options("/v1/score", headers={"Origin": "http://evil.example", **preflight})

    assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "access-control-allow-origin" not in bad.headers


def test_cors_can_be_disabled(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FG_CORS_ORIGINS", "")
    client = TestClient(create_app(bundle))
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert "access-control-allow-origin" not in response.headers
