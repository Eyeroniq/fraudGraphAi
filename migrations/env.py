"""Alembic environment: the models in `services.api.db` are the target schema."""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from fraudgraph.config import get_settings
from services.api.db import Base

config = context.config
if config.config_file_name is not None:
    from logging.config import fileConfig

    fileConfig(config.config_file_name, disable_existing_loggers=False)


def database_url() -> str:
    url = context.get_x_argument(as_dictionary=True).get("url") or get_settings().api.database_url
    if not url:
        raise SystemExit("set FG_API__DATABASE_URL (or pass -x url=...)")
    return url


def run_offline() -> None:
    context.configure(url=database_url(), target_metadata=Base.metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    connectable = config.attributes.get("connection") or create_engine(database_url())
    if hasattr(connectable, "connect"):
        with connectable.connect() as connection:
            context.configure(connection=connection, target_metadata=Base.metadata)
            with context.begin_transaction():
                context.run_migrations()
    else:
        context.configure(connection=connectable, target_metadata=Base.metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
