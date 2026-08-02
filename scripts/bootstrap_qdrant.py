#!/usr/bin/env python3
"""Create the Qdrant collections and payload indexes the platform needs.

Idempotent: it delegates to :func:`ragcore.vectorstore.collections.ensure_collections`,
which creates ``rag_chunks`` (named ``dense`` + ``sparse`` vectors, IDF modifier on the
sparse side, ``payload_m`` tuned for tenant-partitioned search) plus ``rag_memories``
and ``rag_semantic_cache``, then adds every payload index retrieval filters on.

    uv run python scripts/bootstrap_qdrant.py
    uv run python scripts/bootstrap_qdrant.py --verify-only
    uv run python scripts/bootstrap_qdrant.py --drop-existing   # destroys data

Collection names, vector size, HNSW parameters and the Qdrant URL all come from
``ragcore.settings`` — nothing is hard-coded here.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ragcore.logging import configure_logging, get_logger
from ragcore.settings import Settings, get_settings
from ragcore.vectorstore.collections import ensure_collections, get_client

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qdrant_client import AsyncQdrantClient

logger = get_logger("scripts.bootstrap_qdrant")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2


def collection_names(settings: Settings) -> tuple[str, ...]:
    """Return the collections this platform owns, in creation order.

    Args:
        settings: Resolved platform settings.

    Returns:
        The chunk, memory and semantic-cache collection names.
    """
    return (
        settings.qdrant_chunks_collection,
        settings.qdrant_memories_collection,
        settings.qdrant_cache_collection,
    )


async def probe(client: AsyncQdrantClient, settings: Settings) -> None:
    """Verify the Qdrant endpoint answers before doing any work.

    Args:
        client: Connected Qdrant client.
        settings: Resolved platform settings, used for the error message.

    Raises:
        ConnectionError: If the endpoint cannot be reached or refuses the API key.
    """
    try:
        await client.get_collections()
    except Exception as exc:
        msg = (
            f"cannot reach Qdrant at {settings.qdrant_url}: {exc}. "
            "Check RAG_QDRANT_URL and RAG_QDRANT_API_KEY, and that the service is up "
            "(`make up` locally, or the qdrant Container App in Azure)."
        )
        raise ConnectionError(msg) from exc


@dataclass(slots=True)
class CollectionSummary:
    """What one collection looks like right now.

    Attributes:
        name: Collection name.
        error: Why the collection could not be described, when it is missing.
        points: Indexed point count.
        dense_vectors: Names of the dense named vectors.
        sparse_vectors: Names of the sparse named vectors.
        payload_indexes: Payload fields that carry an index.
    """

    name: str
    error: str | None = None
    points: int = 0
    dense_vectors: list[str] = field(default_factory=list)
    sparse_vectors: list[str] = field(default_factory=list)
    payload_indexes: list[str] = field(default_factory=list)


async def describe(client: AsyncQdrantClient, name: str) -> CollectionSummary:
    """Summarise one collection for the console report.

    Args:
        client: Connected Qdrant client.
        name: Collection name.

    Returns:
        A :class:`CollectionSummary`; ``error`` is set when the collection is absent.
    """
    try:
        info = await client.get_collection(name)
    except Exception as exc:
        return CollectionSummary(name=name, error=str(exc))

    vectors = info.config.params.vectors
    dense_names = sorted(vectors) if isinstance(vectors, dict) else ["(unnamed)"]
    sparse = info.config.params.sparse_vectors or {}
    schema = info.payload_schema or {}
    return CollectionSummary(
        name=name,
        points=info.points_count or 0,
        dense_vectors=dense_names,
        sparse_vectors=sorted(sparse),
        payload_indexes=sorted(schema),
    )


async def drop(client: AsyncQdrantClient, names: tuple[str, ...]) -> None:
    """Delete the platform collections.

    Args:
        client: Connected Qdrant client.
        names: Collections to delete.
    """
    for name in names:
        deleted = await client.delete_collection(collection_name=name)
        print(f"  dropped {name}: {deleted}")


async def bootstrap(
    settings: Settings,
    *,
    verify_only: bool = False,
    drop_existing: bool = False,
) -> int:
    """Ensure every collection and payload index exists, then report the layout.

    Args:
        settings: Resolved platform settings.
        verify_only: Report the current layout without creating anything.
        drop_existing: Delete the collections first. Destroys all indexed data.

    Returns:
        A process exit code.
    """
    names = collection_names(settings)
    client = await get_client(settings)
    try:
        await probe(client, settings)
        print(f"qdrant {settings.qdrant_url} reachable")

        if drop_existing:
            print("dropping existing collections")
            await drop(client, names)

        if verify_only:
            print("verify-only: not creating anything")
        else:
            await ensure_collections(client, settings)
            print("collections and payload indexes ensured")

        missing: list[str] = []
        for name in names:
            summary = await describe(client, name)
            if summary.error is not None:
                missing.append(name)
                print(f"  {name}: MISSING ({summary.error})")
                continue
            print(
                f"  {name}: points={summary.points} "
                f"dense={summary.dense_vectors} "
                f"sparse={summary.sparse_vectors}"
            )
            print(
                f"    payload indexes ({len(summary.payload_indexes)}): "
                f"{', '.join(summary.payload_indexes)}"
            )

        if missing:
            logger.error("qdrant_bootstrap_incomplete", missing=missing)
            return EXIT_FAILED
        return EXIT_OK
    finally:
        await client.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="bootstrap_qdrant.py",
        description=(
            "Create the rag_chunks / rag_memories / rag_semantic_cache collections "
            "and every payload index retrieval filters on."
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="report the current collection layout without creating anything",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="delete the collections before creating them (DESTROYS ALL DATA)",
    )
    parser.add_argument(
        "--allow-production",
        action="store_true",
        help="permit --drop-existing when RAG_ENV=production",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings)

    if args.drop_existing and settings.is_production and not args.allow_production:
        print(
            "refusing to drop collections while RAG_ENV=production; "
            "pass --allow-production if you really mean it",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    try:
        return asyncio.run(
            bootstrap(
                settings,
                verify_only=args.verify_only,
                drop_existing=args.drop_existing,
            )
        )
    except ConnectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
