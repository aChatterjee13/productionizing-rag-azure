"""Alembic environment: async migrations driven by ragcore.settings.

The database URL always comes from :func:`ragcore.settings.get_settings`, never from
``alembic.ini``, so a migration cannot be applied to a different database than the
application talks to and no credential is committed.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Importing the models registers every table on Base.metadata.
from ragcore.db import models as _models  # noqa: F401
from ragcore.db.base import Base
from ragcore.settings import get_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata

_settings = get_settings()
config.set_main_option("sqlalchemy.url", _settings.alembic_database_url)


def _include_object(_object, name, type_, _reflected, _compare_to) -> bool:
    """Filter objects out of autogenerate comparisons.

    Args:
        _object: The schema object being considered.
        name: Object name.
        type_: Object type, e.g. ``"table"``.
        _reflected: Whether the object came from the database.
        _compare_to: The object it is being compared against.

    Returns:
        False for Alembic's own bookkeeping table, True otherwise.
    """
    return not (type_ == "table" and name == "alembic_version")


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to a database (``alembic --sql``)."""
    context.configure(
        url=_settings.alembic_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_object=_include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection: Connection) -> None:
    """Run migrations on an established synchronous connection.

    Args:
        connection: A connection produced by ``AsyncConnection.run_sync``.
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_object=_include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def _run_async_migrations() -> None:
    """Create an async engine and run migrations through it."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Apply migrations against the configured database."""
    asyncio.run(_run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
