"""simulation control

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sim_control",
        sa.Column("sim_id", sa.Integer(), nullable=False),
        sa.Column("running", sa.Boolean(), nullable=False),
        sa.Column("scenario", sa.String(length=8), nullable=False),
        sa.Column("scenario_start", sa.Integer(), nullable=False),
        sa.Column("first_step", sa.Integer(), nullable=False),
        sa.Column("last_step", sa.Integer(), nullable=False),
        sa.Column("next_step", sa.Integer(), nullable=False),
        sa.Column("steps_per_tick", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("sim_id"),
    )


def downgrade() -> None:
    op.drop_table("sim_control")
