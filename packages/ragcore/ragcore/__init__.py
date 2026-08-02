"""ragcore: shared contracts and infrastructure for the productionizing-rag platform.

Deliberately light at import time. Settings, errors, logging and the exception
hierarchy are re-exported here; anything that pulls a heavy dependency (Qdrant,
FastEmbed, the Anthropic SDK, SQLAlchemy) must be imported from its own submodule so
that ``import ragcore`` stays fast and never fails because an optional extra is
missing:

    from ragcore import get_settings                # cheap
    from ragcore.models import ChunkPayload         # cheap
    from ragcore.vectorstore.filters import build_acl_filter   # pulls qdrant-client
    from ragcore.db.base import get_session         # pulls sqlalchemy + asyncpg
"""

from ragcore.errors import (
    AuthError,
    ConfigError,
    GuardrailBlocked,
    RagError,
    RetrievalError,
    TenantMismatchError,
    ToolExecutionError,
)
from ragcore.logging import bind_request_context, configure_logging, get_logger
from ragcore.settings import Settings, get_settings

__version__ = "0.1.0"

__all__ = [
    "AuthError",
    "ConfigError",
    "GuardrailBlocked",
    "RagError",
    "RetrievalError",
    "Settings",
    "TenantMismatchError",
    "ToolExecutionError",
    "__version__",
    "bind_request_context",
    "configure_logging",
    "get_logger",
    "get_settings",
]
