"""Worker (7.4) and champion/challenger (7.5) end to end: API + worker on the synthetic
fixture and in-memory SQLite, with the worker replaying the stream through the API."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fraudgraph.config import WorkerSettings
from fraudgraph.data.prepare import to_canonical
from fraudgraph.data.synth import generate
from fraudgraph.pipeline import fixture_bundle
from services.api.main import create_app
from services.worker.jobs import Worker
from tests.api_helpers import api_settings, auth

LAST_STEP = 150
INJECT_AT = 97


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return fixture_bundle(tmp_path_factory.mktemp("bundles"), trials=2)


def make_app(bundle: Path) -> tuple[Any, TestClient]:
    settings = api_settings(challenger_refresh_s=0, rate_limit_batch="100000/minute")
    app = create_app(bundle, settings)
    return app, TestClient(app)


def poster(client: TestClient) -> Any:
    def post(path: str, body: Any, scope: str) -> tuple[int, Any]:
        response = client.post(path, json=body, headers=auth(scope))
        return response.status_code, response.json()

    return post


@pytest.fixture(scope="module")
def run(bundle: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Replay steps 1-150 with D3 (mule rings) from step 97; one tick per 12 steps."""
    app, client = make_app(bundle)
    ctx = app.state.ctx
    worker = Worker(
        log=ctx.log,
        settings=WorkerSettings(
            evidently=False,
            label_delay_steps=6,
            min_fraud_labels=5,
            min_shadow_labels=50,
        ),
        bundles_root=bundle.parent,
        reports_dir=tmp_path_factory.mktemp("reports"),
        post=poster(client),
        events=to_canonical(generate().frame),
    )
    admin = {"headers": auth("admin")}
    start = {"first_step": 1, "last_step": LAST_STEP, "steps_per_tick": 12}
    assert client.post("/v1/sim/start", json=start, **admin).status_code == 200
    ticks = []
    for _ in range(40):
        if ctx.log.sim().next_step >= INJECT_AT and ctx.log.sim().scenario == "D0":
            client.post("/v1/sim/scenario", json={"name": "D3"}, **admin)
        ticks.append(worker.tick())
        if not ctx.log.sim().running:
            break
    windows, alarms = ctx.log.drift(0)
    return {
        "client": client,
        "ctx": ctx,
        "ticks": ticks,
        "windows": windows,
        "alarms": alarms,
        "models": {m.version: m.stage for m in ctx.log.models()},
    }


def test_replayer_streams_every_step_and_labels(run: dict[str, Any]) -> None:
    replayed = [s for t in run["ticks"] for s in t["replayed"]]
    assert replayed == list(range(1, LAST_STEP + 1))
    assert run["ctx"].log.latest_step() == LAST_STEP
    sim = run["client"].get("/v1/sim/status", headers=auth("read")).json()
    assert (sim["running"], sim["scenario"], sim["scenario_start"]) == (False, "D3", INJECT_AT)
    performance = run["client"].get("/v1/monitoring/performance?window=1w", headers=auth("read"))
    assert sum(m["n_labelled"] for m in performance.json()["models"]) > 1_000


def test_drift_monitor_writes_day_windows_and_alarms_after_the_rings(
    run: dict[str, Any],
) -> None:
    ends = sorted({w.end_step for w in run["windows"]})
    assert ends == [72, 96, 120, 144]  # reference = steps 1-48, then full days only
    assert all(e - s == 23 for s, e in {(w.start_step, w.end_step) for w in run["windows"]})
    assert any(a.step > INJECT_AT for a in run["alarms"])
    drift = run["client"].get("/v1/monitoring/drift?window=1w", headers=auth("read")).json()
    assert len(drift["windows"]) == len(run["windows"])


def test_daily_retrain_registers_challengers_and_the_better_one_is_promoted(
    run: dict[str, Any],
) -> None:
    retrains = [t["retrain"] for t in run["ticks"] if t["retrain"] and "challenger" in t["retrain"]]
    assert len(retrains) >= 3
    verdicts = [v for t in run["ticks"] for v in t["judged"]]
    assert verdicts, "no challenger reached the shadow-label minimum"
    for v in verdicts:  # the rule: promote iff the challenger's realised cost is lower
        assert v.get("promoted", False) == (v["cost_challenger"] < v["cost_champion"])
        assert v.get("retired", False) == (v["cost_challenger"] >= v["cost_champion"])
    current = run["client"].get("/v1/models/current", headers=auth("read")).json()["version"]
    champions = [v for v, stage in run["models"].items() if stage == "champion"]
    assert champions == [current]
    if any(v.get("promoted") for v in verdicts):
        assert "-r" in current  # a retrained bundle is being served
    metrics = run["client"].get("/metrics").text
    assert f'fraudgraph_model_info{{model_version="{current}"}} 1.0' in metrics


# ------------------------------------------------------------------ 7.5 in isolation


def copy_bundle(bundle: Path, version: str) -> Path:
    """Same model under another version (the manifest is not part of its own hashes)."""
    target = bundle.parent / version
    shutil.copytree(bundle, target, dirs_exist_ok=True)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    manifest["version"] = version
    (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return target


def test_challenger_scores_in_shadow_then_promotion_swaps_the_model(bundle: Path) -> None:
    app, client = make_app(bundle)
    ctx = app.state.ctx
    challenger = copy_bundle(bundle, "shadow-test-1")
    ctx.log.register_challenger("shadow-test-1", {}, challenger.name)
    events = to_canonical(generate().frame).filter(
        to_canonical(generate().frame)["type"].is_in(["TRANSFER", "CASH_OUT"])
    )
    batch = events.head(200).select("txn_id", "step", "type", "amount", "name_orig", "name_dest")
    response = client.post(
        "/v1/score/batch", json={"transactions": batch.to_dicts()}, headers=auth("write")
    )
    assert response.status_code == 200
    first = response.json()["results"][0]
    assert first["model_version"] == ctx.scorer.version != "shadow-test-1"
    pairs = ctx.log.shadow_pairs("shadow-test-1", since_step=0)
    assert pairs == []  # nothing labelled yet, but shadow rows exist:
    with ctx.log.sessions() as session:
        from services.api.db import ShadowPrediction

        shadow = session.query(ShadowPrediction).filter_by(model_version="shadow-test-1").all()
    assert len(shadow) == 200
    assert all(s.p_fraud >= 0 for s in shadow)

    old = ctx.scorer.version
    promoted = client.post("/v1/models/shadow-test-1/promote", headers=auth("admin"))
    assert promoted.json() == {"version": "shadow-test-1", "previous": old}
    stages = {m.version: m.stage for m in ctx.log.models()}
    assert (stages["shadow-test-1"], stages[old]) == ("champion", "archived")
    after = client.post(
        "/v1/score",
        json=events.slice(200, 1).select(batch.columns).to_dicts()[0],
        headers=auth("write"),
    )
    assert after.json()["model_version"] == "shadow-test-1"
    assert "shadow-test-1" not in ctx.challengers


def test_restart_serves_the_registered_champion_not_the_newest_bundle(
    bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fraudgraph.config import get_settings
    from services.api.db import Base, make_engine

    root = tmp_path / "artifacts" / "bundles"
    champion = copy_bundle(bundle, "restart-champion")
    shutil.copytree(champion, root / champion.name)
    newer = copy_bundle(bundle, "restart-newer")
    manifest = json.loads((newer / "manifest.json").read_text(encoding="utf-8"))
    manifest["created_at"] = "2999-01-01T00:00:00+00:00"  # newest on disk
    (newer / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    shutil.copytree(newer, root / newer.name)
    url = f"sqlite:///{(tmp_path / 'log.db').as_posix()}"
    Base.metadata.create_all(make_engine(url))
    monkeypatch.setenv("FG_PATHS__ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.delenv("FG_BUNDLE_DIR", raising=False)
    get_settings.cache_clear()

    first = create_app(None, api_settings(database_url=url))
    assert first.state.ctx.scorer.version == "restart-newer"  # empty log: newest bundle
    first.state.ctx.log.register_model("restart-champion", {}, champion.name)
    restarted = create_app(None, api_settings(database_url=url))
    assert restarted.state.ctx.scorer.version == "restart-champion"
