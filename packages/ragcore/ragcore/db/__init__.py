"""Relational persistence: engine, ORM models and tenant-scoped repositories.

Importing this package pulls SQLAlchemy and asyncpg, so keep it out of hot import
paths that only need models or settings.

    from ragcore.db import get_session, repositories
    from ragcore.db.models import Document
"""

from ragcore.db import repositories
from ragcore.db.base import (
    Base,
    check_database,
    dispose_engine,
    get_engine,
    get_session,
    get_sessionmaker,
    metadata,
    session_scope,
)

__all__ = [
    "Base",
    "check_database",
    "dispose_engine",
    "get_engine",
    "get_session",
    "get_sessionmaker",
    "metadata",
    "repositories",
    "session_scope",
]
