"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-10-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "alarms",
        sa.Column("alarm_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("detector", sa.String(length=32), nullable=False),
        sa.Column("feature", sa.String(length=64), nullable=True),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column(
            "details",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("alarm_id"),
    )
    op.create_index(op.f("ix_alarms_step"), "alarms", ["step"], unique=False)
    op.create_table(
        "drift_windows",
        sa.Column("window_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("start_step", sa.Integer(), nullable=False),
        sa.Column("end_step", sa.Integer(), nullable=False),
        sa.Column("feature", sa.String(length=64), nullable=False),
        sa.Column("psi", sa.Float(), nullable=True),
        sa.Column("ks_stat", sa.Float(), nullable=True),
        sa.Column("ks_p", sa.Float(), nullable=True),
        sa.Column("alarm", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("window_id"),
    )
    op.create_index(op.f("ix_drift_windows_end_step"), "drift_windows", ["end_step"], unique=False)
    op.create_table(
        "model_versions",
        sa.Column("version", sa.String(length=128), nullable=False),
        sa.Column("stage", sa.String(length=16), nullable=False),
        sa.Column(
            "metrics",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("bundle_uri", sa.String(length=512), nullable=False),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("version"),
    )
    op.create_table(
        "shadow_predictions",
        sa.Column("txn_id", sa.String(length=64), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("p_fraud", sa.Float(), nullable=False),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.PrimaryKeyConstraint("txn_id", "model_version"),
    )
    op.create_table(
        "transactions",
        sa.Column("txn_id", sa.String(length=64), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(length=16), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("name_orig", sa.String(length=32), nullable=False),
        sa.Column("name_dest", sa.String(length=32), nullable=False),
        sa.Column("oldbalance_orig", sa.Float(), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("scenario", sa.String(length=32), nullable=True),
        sa.PrimaryKeyConstraint("txn_id"),
    )
    op.create_index(op.f("ix_transactions_name_dest"), "transactions", ["name_dest"], unique=False)
    op.create_index(op.f("ix_transactions_name_orig"), "transactions", ["name_orig"], unique=False)
    op.create_index(op.f("ix_transactions_step"), "transactions", ["step"], unique=False)
    op.create_table(
        "labels",
        sa.Column("txn_id", sa.String(length=64), nullable=False),
        sa.Column("is_fraud", sa.Boolean(), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("labelled_at_step", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["txn_id"],
            ["transactions.txn_id"],
        ),
        sa.PrimaryKeyConstraint("txn_id"),
    )
    op.create_table(
        "predictions",
        sa.Column("txn_id", sa.String(length=64), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("p_fraud", sa.Float(), nullable=False),
        sa.Column("set_legit", sa.Boolean(), nullable=True),
        sa.Column("set_fraud", sa.Boolean(), nullable=True),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("decision_reason", sa.String(length=32), nullable=False),
        sa.Column(
            "expected_cost",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "features",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "reasons",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("explanation_stable", sa.Boolean(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(
            ["txn_id"],
            ["transactions.txn_id"],
        ),
        sa.PrimaryKeyConstraint("txn_id"),
    )
    op.create_index(
        "ix_predictions_decision_step", "predictions", ["decision", "step"], unique=False
    )
    op.create_index(op.f("ix_predictions_step"), "predictions", ["step"], unique=False)
    op.create_table(
        "review_cases",
        sa.Column("case_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("txn_id", sa.String(length=64), nullable=False),
        sa.Column("priority", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("assignee", sa.String(length=64), nullable=True),
        sa.Column("verdict", sa.String(length=16), nullable=True),
        sa.Column("notes", sa.String(length=2000), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["txn_id"],
            ["transactions.txn_id"],
        ),
        sa.PrimaryKeyConstraint("case_id"),
        sa.UniqueConstraint("txn_id"),
    )
    op.create_index(
        "ix_review_cases_status_priority", "review_cases", ["status", "priority"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_review_cases_status_priority", table_name="review_cases")
    op.drop_table("review_cases")
    op.drop_index(op.f("ix_predictions_step"), table_name="predictions")
    op.drop_index("ix_predictions_decision_step", table_name="predictions")
    op.drop_table("predictions")
    op.drop_table("labels")
    op.drop_index(op.f("ix_transactions_step"), table_name="transactions")
    op.drop_index(op.f("ix_transactions_name_orig"), table_name="transactions")
    op.drop_index(op.f("ix_transactions_name_dest"), table_name="transactions")
    op.drop_table("transactions")
    op.drop_table("shadow_predictions")
    op.drop_table("model_versions")
    op.drop_index(op.f("ix_drift_windows_end_step"), table_name="drift_windows")
    op.drop_table("drift_windows")
    op.drop_index(op.f("ix_alarms_step"), table_name="alarms")
    op.drop_table("alarms")
