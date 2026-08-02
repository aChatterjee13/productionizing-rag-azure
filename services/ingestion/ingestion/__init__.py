"""Serverless, multi-tenant, ACL-aware delta ingestion (requirement #1).

Import layout mirrors the pipeline stages:

    from ingestion.pipeline import run_ingest          # the operational entry point
    from ingestion.pipeline import run_ingestion       # multi-tenant timer variant
    from ingestion.connectors import get_connector     # blob/local/sharepoint/http/sql
    from ingestion.delta import get_manifest_store     # delta state
    from ingestion.parse import parse_document         # bytes -> ParsedBlocks
    from ingestion.chunk import chunk_document         # blocks -> chunks
    from ingestion.enrich import enrich_document       # summary/keywords/PII/language
    from ingestion.upsert import RunUpserter           # embed, dedupe, write, lineage

Nothing heavy is imported here. ``ingestion.pipeline`` pulls SQLAlchemy, Qdrant and
FastEmbed transitively, so ``import ingestion`` stays cheap for the Azure Functions
host's cold start and for tests that only exercise chunking or delta logic.
``run_ingest`` is re-exported through a module-level ``__getattr__`` so
``from ingestion import run_ingest`` works without paying that cost at import time.

``function_app.py`` at the service root is a thin adapter: every trigger delegates to
:mod:`ingestion.pipeline`, so the nightly Azure run and a local CLI invocation execute
identical code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ingestion.pipeline import run_ingest

__version__ = "0.1.0"

__all__ = ["__version__", "run_ingest"]

#: Lazily re-exported names -> the module that defines them.
_LAZY: dict[str, str] = {"run_ingest": "ingestion.pipeline"}


def __getattr__(name: str) -> Any:
    """Resolve a lazily re-exported symbol on first access.

    Args:
        name: Attribute being looked up on the package.

    Returns:
        The resolved object.

    Raises:
        AttributeError: If the package does not export that name.
    """
    module_name = _LAZY.get(name)
    if module_name is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    """List the package's public names, including the lazy ones.

    Returns:
        Sorted attribute names.
    """
    return sorted({*globals(), *_LAZY})
