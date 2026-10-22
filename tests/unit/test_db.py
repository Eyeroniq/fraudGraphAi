"""Decision log (task 7.2): Alembic schema equals the models; writes are idempotent.

SQLite everywhere; a real Postgres under the `integration` marker (CI service container).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine import Engine

from fraudgraph.config import REPO_ROOT
from fraudgraph.serving.scorer import Decision
from services.api.db import (
    Base,
    DecisionLog,
    Label,
    Prediction,
    ReviewCase,
    Scored,
    Transaction,
)
from tests.backends import service_url


def upgraded(url: str) -> Engine:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.cmd_opts = None
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    engine = create_engine(url)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
    return engine


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.integration)])
def engine(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Engine]:
    if request.param == "sqlite":
        engine = upgraded(f"sqlite:///{tmp_path / 'fg.db'}")
        yield engine
        engine.dispose()
        return
    base = create_engine(service_url("TEST_DATABASE_URL"), isolation_level="AUTOCOMMIT")
    schema = f"t_{uuid.uuid4().hex[:8]}"  # isolated schema per test
    with base.connect() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    url = base.url.update_query_dict({"options": f"-csearch_path={schema}"})
    engine = upgraded(url.render_as_string(hide_password=False))
    yield engine
    engine.dispose()
    with base.connect() as conn:
        conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
    base.dispose()


def test_migrations_match_the_models(engine: Engine) -> None:
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []


def scored(txn_id: str, decision: str = "REVIEW", step: int = 5) -> Scored:
    d = Decision(
        txn_id=txn_id,
        model_version="m1",
        p_fraud=0.3,
        decision=decision,
        decision_reason="expected_cost",
        expected_cost={"APPROVE": 900.0, "REVIEW": 2_100.0},
        prediction_set=["legit", "fraud"],
        reasons=[{"code": "DEST_HISTORY", "text": "new receiver", "impact": 0.5}],
        explanation_stable=True,
        latency_ms=1.0,
        features={"g_dest_age": float("nan"), "t_log_amount": 8.0},
    )
    txn = {"txn_id": txn_id, "step": step, "type": "TRANSFER", "amount": 3_000.0}
    txn |= {"name_orig": "C1", "name_dest": "C2"}
    return Scored(txn=txn, decision=d, features=d.features)


def count(engine: Engine, model: type[Base]) -> int:
    with engine.connect() as conn:
        return int(conn.scalar(select(func.count()).select_from(model)) or 0)


def test_writes_are_idempotent_on_txn_id(engine: Engine) -> None:
    log = DecisionLog(engine)
    log.record([scored("T1"), scored("T2", decision="APPROVE")])
    log.record([scored("T1")])  # a retried request

    assert (count(engine, Transaction), count(engine, Prediction)) == (2, 2)
    assert count(engine, ReviewCase) == 1
    _, prediction, label = log.transaction("T1")
    assert prediction is not None
    assert prediction.features == {"g_dest_age": None, "t_log_amount": 8.0}
    assert (prediction.set_legit, prediction.set_fraud) == (True, True)
    assert label is None


def test_first_label_wins(engine: Engine) -> None:
    log = DecisionLog(engine)
    log.record([scored("T1")])
    assert log.add_label("T1", True, "chargeback", 29) is True
    assert log.add_label("T1", False, "analyst", 30) is False
    assert count(engine, Label) == 1
    assert log.performance(0) == {"m1": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}}


def test_cases_page_and_resolve(engine: Engine) -> None:
    log = DecisionLog(engine)
    log.record([scored(f"T{i}") for i in range(5)])
    page = log.cases(status="open", assignee=None, sort="priority", limit=2, offset=0)
    assert (page.total, len(page.items)) == (5, 2)

    case = page.items[0][0]
    resolved = log.resolve_case(case.case_id, "fraud", "a1", None)
    assert resolved is not None
    assert resolved.status == "resolved"
    assert log.cases(status="open", assignee=None, sort="created", limit=10, offset=0).total == 4
    assert log.resolve_case(9_999, "fraud", None, None) is None


def test_registering_a_new_champion_archives_the_old_one(engine: Engine) -> None:
    log = DecisionLog(engine)
    log.register_model("m1", {"ap": 0.3}, "m1")
    log.register_model("m2", {"ap": 0.4}, "m2")
    log.register_model("m2", {"ap": 0.4}, "m2")  # restart: no change
    stages = {m.version: m.stage for m in log.models()}
    assert stages == {"m1": "archived", "m2": "champion"}
