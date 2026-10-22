"""Postgres decision log (task 7.2, A8): SQLAlchemy 2 models and the queries the API runs.

Writes are idempotent on `txn_id` (insert, do nothing on conflict) and happen in a
background task after the response is sent, so scoring latency never waits on the
database. Alembic (`migrations/`) owns the schema; SQLite is only used in tests and for
runs without Docker, where the tables are created directly.

`predictions` also keeps the transaction's `step` (not in A8) so the `(decision, step)`
index and the monitoring windows do not need a join.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    create_engine,
    func,
    or_,
    select,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from fraudgraph.serving.scorer import Decision

JSONType = JSON().with_variant(postgresql.JSONB(), "postgresql")
FLAGGED = ("STEP_UP", "REVIEW", "BLOCK")  # anything not approved counts as a positive


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSONType, list[Any]: JSONType}


class Transaction(Base):
    __tablename__ = "transactions"

    txn_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    step: Mapped[int] = mapped_column(Integer, index=True)
    type: Mapped[str] = mapped_column(String(16))
    amount: Mapped[float] = mapped_column(Float)
    name_orig: Mapped[str] = mapped_column(String(32), index=True)
    name_dest: Mapped[str] = mapped_column(String(32), index=True)
    oldbalance_orig: Mapped[float | None] = mapped_column(Float)
    received_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    scenario: Mapped[str | None] = mapped_column(String(32))


class Prediction(Base):
    """Champion decisions only; challengers go to `shadow_predictions`."""

    __tablename__ = "predictions"
    __table_args__ = (Index("ix_predictions_decision_step", "decision", "step"),)

    txn_id: Mapped[str] = mapped_column(ForeignKey("transactions.txn_id"), primary_key=True)
    step: Mapped[int] = mapped_column(Integer, index=True)
    model_version: Mapped[str] = mapped_column(String(128))
    p_fraud: Mapped[float] = mapped_column(Float)
    set_legit: Mapped[bool | None] = mapped_column(Boolean)
    set_fraud: Mapped[bool | None] = mapped_column(Boolean)
    decision: Mapped[str] = mapped_column(String(16))
    decision_reason: Mapped[str] = mapped_column(String(32))
    expected_cost: Mapped[dict[str, Any]]
    features: Mapped[dict[str, Any]]
    reasons: Mapped[list[Any]]
    explanation_stable: Mapped[bool | None] = mapped_column(Boolean)
    latency_ms: Mapped[float] = mapped_column(Float)


class ShadowPrediction(Base):
    __tablename__ = "shadow_predictions"

    txn_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_version: Mapped[str] = mapped_column(String(128), primary_key=True)
    p_fraud: Mapped[float] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String(16))


class Label(Base):
    __tablename__ = "labels"

    txn_id: Mapped[str] = mapped_column(ForeignKey("transactions.txn_id"), primary_key=True)
    is_fraud: Mapped[bool] = mapped_column(Boolean)
    source: Mapped[str] = mapped_column(String(16))  # analyst | chargeback | simulator
    labelled_at_step: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewCase(Base):
    __tablename__ = "review_cases"
    __table_args__ = (Index("ix_review_cases_status_priority", "status", "priority"),)

    case_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    txn_id: Mapped[str] = mapped_column(ForeignKey("transactions.txn_id"), unique=True)
    priority: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="open")
    assignee: Mapped[str | None] = mapped_column(String(64))
    verdict: Mapped[str | None] = mapped_column(String(16))
    notes: Mapped[str | None] = mapped_column(String(2000))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    resolved_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))


class DriftWindow(Base):
    __tablename__ = "drift_windows"

    window_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    start_step: Mapped[int] = mapped_column(Integer)
    end_step: Mapped[int] = mapped_column(Integer, index=True)
    feature: Mapped[str] = mapped_column(String(64))
    psi: Mapped[float | None] = mapped_column(Float)
    ks_stat: Mapped[float | None] = mapped_column(Float)
    ks_p: Mapped[float | None] = mapped_column(Float)
    alarm: Mapped[bool] = mapped_column(Boolean, default=False)


class Alarm(Base):
    __tablename__ = "alarms"

    alarm_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(16))  # input | prediction | performance
    detector: Mapped[str] = mapped_column(String(32))
    feature: Mapped[str | None] = mapped_column(String(64))
    step: Mapped[int] = mapped_column(Integer, index=True)
    details: Mapped[dict[str, Any]]


class ModelVersion(Base):
    __tablename__ = "model_versions"

    version: Mapped[str] = mapped_column(String(128), primary_key=True)
    stage: Mapped[str] = mapped_column(String(16))  # champion | challenger | archived
    metrics: Mapped[dict[str, Any]]
    bundle_uri: Mapped[str] = mapped_column(String(512))
    promoted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))


# ---------------------------------------------------------------------------- helpers


def make_engine(url: str | None) -> Engine:
    """Engine for `url`; None is a private in-memory SQLite database (tables created)."""
    if url is None:
        engine = create_engine(
            "sqlite+pysqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        return engine
    return create_engine(url, pool_pre_ping=True)


def _insert(session: Session, model: type[Base], rows: list[dict[str, Any]]) -> int:
    """INSERT ... ON CONFLICT DO NOTHING; returns the number of rows written.

    Rows are counted from RETURNING, not `rowcount`: on Postgres a multi-row insert can
    run in SQLAlchemy's "insertmanyvalues" mode, where `rowcount` is not reliable.
    """
    if not rows:
        return 0
    dialect = session.get_bind().dialect.name
    insert = postgresql.insert if dialect == "postgresql" else sqlite.insert
    table = cast(Table, model.__table__)
    stmt = insert(table).values(rows).on_conflict_do_nothing().returning(*table.primary_key.columns)
    return len(session.execute(stmt).all())


def _json_safe(features: dict[str, float]) -> dict[str, float | None]:
    """NaN ("no history") is not valid JSON in Postgres: store it as null."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in features.items()}


@dataclass(frozen=True)
class Scored:
    """What one scoring call hands to the background writer."""

    txn: dict[str, Any]  # Transaction columns
    decision: Decision
    features: dict[str, float]


@dataclass(frozen=True)
class Page:
    items: list[Any]
    total: int


class DecisionLog:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.sessions = sessionmaker(engine, expire_on_commit=False)

    def ping(self) -> bool:
        with self.engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return True

    # ---------------------------------------------------------------- writes

    def record(self, scored: Sequence[Scored]) -> None:
        """Store transactions, champion predictions and review cases (idempotent)."""
        txns, preds, cases = [], [], []
        for s in scored:
            d = s.decision
            txns.append(s.txn)
            sets = d.prediction_set
            preds.append(
                {
                    "txn_id": d.txn_id,
                    "step": s.txn["step"],
                    "model_version": d.model_version,
                    "p_fraud": d.p_fraud,
                    "set_legit": None if sets is None else "legit" in sets,
                    "set_fraud": None if sets is None else "fraud" in sets,
                    "decision": d.decision,
                    "decision_reason": d.decision_reason,
                    "expected_cost": d.expected_cost,
                    "features": _json_safe(s.features),
                    "reasons": d.reasons,
                    "explanation_stable": d.explanation_stable,
                    "latency_ms": d.latency_ms,
                }
            )
            if d.decision == "REVIEW":
                cases.append({"txn_id": d.txn_id, "priority": review_priority(d)})
        with self.sessions.begin() as session:
            _insert(session, Transaction, txns)
            _insert(session, Prediction, preds)
            _insert(session, ReviewCase, cases)

    def add_label(self, txn_id: str, is_fraud: bool, source: str, step: int) -> bool:
        """Store a label; False if the transaction already had one (first label wins)."""
        row = {"txn_id": txn_id, "is_fraud": is_fraud, "source": source, "labelled_at_step": step}
        with self.sessions.begin() as session:
            return _insert(session, Label, [row]) == 1

    def resolve_case(
        self, case_id: int, verdict: str, assignee: str | None, notes: str | None
    ) -> ReviewCase | None:
        with self.sessions.begin() as session:
            case = session.get(ReviewCase, case_id)
            if case is None:
                return None
            case.status = "resolved"
            case.verdict = verdict
            case.assignee = assignee or case.assignee
            case.notes = notes
            case.resolved_at = utcnow()
            return case

    def register_model(self, version: str, metrics: dict[str, Any], bundle_uri: str) -> None:
        """Record the bundle the API serves as champion (others become archived)."""
        with self.sessions.begin() as session:
            for other in session.scalars(
                select(ModelVersion).where(
                    ModelVersion.stage == "champion", ModelVersion.version != version
                )
            ):
                other.stage = "archived"
            current = session.get(ModelVersion, version)
            if current is None:
                session.add(
                    ModelVersion(
                        version=version,
                        stage="champion",
                        metrics=metrics,
                        bundle_uri=bundle_uri,
                        promoted_at=utcnow(),
                    )
                )
            elif current.stage != "champion":
                current.stage, current.promoted_at = "champion", utcnow()

    # ---------------------------------------------------------------- reads

    def transaction(self, txn_id: str) -> tuple[Transaction, Prediction | None, Label | None]:
        with self.sessions() as session:
            txn = session.get(Transaction, txn_id)
            if txn is None:
                raise KeyError(txn_id)
            return txn, session.get(Prediction, txn_id), session.get(Label, txn_id)

    def cases(
        self,
        *,
        status: str | None,
        assignee: str | None,
        sort: Literal["priority", "created"],
        limit: int,
        offset: int,
    ) -> Page:
        query = select(ReviewCase, Transaction, Prediction).join(
            Transaction, Transaction.txn_id == ReviewCase.txn_id
        )
        query = query.outerjoin(Prediction, Prediction.txn_id == ReviewCase.txn_id)
        if status is not None:
            query = query.where(ReviewCase.status == status)
        if assignee is not None:
            query = query.where(ReviewCase.assignee == assignee)
        order = (
            (ReviewCase.priority.desc(), ReviewCase.case_id)
            if sort == "priority"
            else (ReviewCase.created_at.desc(), ReviewCase.case_id.desc())
        )
        with self.sessions() as session:
            total = session.scalar(select(func.count()).select_from(query.subquery())) or 0
            rows = session.execute(query.order_by(*order).limit(limit).offset(offset)).all()
        return Page(items=[tuple(r) for r in rows], total=int(total))

    def case(self, case_id: int) -> ReviewCase | None:
        with self.sessions() as session:
            return session.get(ReviewCase, case_id)

    def ego_graph(
        self, txn: Transaction, hops: int, max_edges: int
    ) -> list[tuple[Transaction, Label | None]]:
        """Transactions within `hops` of the txn's two accounts, up to its step."""
        frontier = {txn.name_orig, txn.name_dest}
        seen_accounts = set(frontier)
        edges: dict[str, tuple[Transaction, Label | None]] = {}
        with self.sessions() as session:
            for _ in range(hops):
                if not frontier or len(edges) >= max_edges:
                    break
                rows = session.execute(
                    select(Transaction, Label)
                    .outerjoin(Label, Label.txn_id == Transaction.txn_id)
                    .where(
                        Transaction.step <= txn.step,
                        or_(
                            Transaction.name_orig.in_(frontier),
                            Transaction.name_dest.in_(frontier),
                        ),
                    )
                    .order_by(Transaction.step.desc(), Transaction.txn_id)
                    .limit(max_edges - len(edges))
                ).all()
                frontier = set()
                for t, label in rows:
                    edges.setdefault(t.txn_id, (t, label))
                    for account in (t.name_orig, t.name_dest):
                        if account not in seen_accounts:
                            seen_accounts.add(account)
                            frontier.add(account)
        return list(edges.values())

    def latest_step(self) -> int | None:
        with self.sessions() as session:
            return session.scalar(select(func.max(Transaction.step)))

    def performance(self, since_step: int) -> dict[str, dict[str, int]]:
        """Confusion counts of labelled decisions per model version (champion + shadow)."""
        out: dict[str, dict[str, int]] = {}
        with self.sessions() as session:
            champion = session.execute(
                select(Prediction.model_version, Prediction.decision, Label.is_fraud)
                .join(Label, Label.txn_id == Prediction.txn_id)
                .where(Prediction.step >= since_step)
            ).all()
            shadow = session.execute(
                select(ShadowPrediction.model_version, ShadowPrediction.decision, Label.is_fraud)
                .join(Label, Label.txn_id == ShadowPrediction.txn_id)
                .join(Transaction, Transaction.txn_id == ShadowPrediction.txn_id)
                .where(Transaction.step >= since_step)
            ).all()
        for version, decision, is_fraud in (*champion, *shadow):
            counts = out.setdefault(version, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            flagged = decision in FLAGGED
            key = ("tp" if is_fraud else "fp") if flagged else ("fn" if is_fraud else "tn")
            counts[key] += 1
        return out

    def drift(self, since_step: int) -> tuple[list[DriftWindow], list[Alarm]]:
        with self.sessions() as session:
            windows = session.scalars(
                select(DriftWindow)
                .where(DriftWindow.end_step >= since_step)
                .order_by(DriftWindow.end_step, DriftWindow.feature)
            ).all()
            alarms = session.scalars(
                select(Alarm).where(Alarm.step >= since_step).order_by(Alarm.step)
            ).all()
        return list(windows), list(alarms)

    def models(self) -> list[ModelVersion]:
        with self.sessions() as session:
            return list(
                session.scalars(
                    select(ModelVersion).order_by(ModelVersion.promoted_at.desc().nulls_last())
                ).all()
            )


def review_priority(decision: Decision) -> float:
    """Expected savings of a review over approving (A6, capacity ranking)."""
    costs = decision.expected_cost
    return round(costs.get("APPROVE", 0.0) - costs.get("REVIEW", 0.0), 4)
