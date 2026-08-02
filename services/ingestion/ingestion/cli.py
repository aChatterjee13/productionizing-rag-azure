"""Command-line entry point: ``python -m ingestion.cli run --tenant <id>``.

The CLI is a thin argv adapter over :func:`ingestion.pipeline.run_ingest`, exactly as
``function_app.py`` is a thin trigger adapter over the same coroutine. Nothing here
decides anything the Azure run would decide differently — including the working-hours
guard, which :func:`run_ingest` evaluates for every caller.

    python -m ingestion.cli run --tenant tenant-acme --source-type local --force
    python -m ingestion.cli run --tenant tenant-acme --source-id src-policies --dry-run
    python -m ingestion.cli run --tenant tenant-acme --root ./data/documents --dry-run

With ``--source-type local`` (or ``--root``) and no matching row in
``source_configs``, an ephemeral local-filesystem source is synthesised from
``RAG_INGEST_LOCAL_ROOT`` so a first run needs no database seeding.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from ragcore.logging import configure_logging, get_logger
from ragcore.models.acl import Classification
from ragcore.models.document import (
    IngestRunSummary,
    IngestTrigger,
    SourceConfig,
    SourceType,
)
from ragcore.settings import Settings, get_settings

__all__ = ["build_parser", "local_source_config", "main"]

_log = get_logger("ingestion.cli")

#: Process exit codes. Kept identical to ``scripts/run_ingest_local.py`` so a wrapper
#: can treat either entry point the same way.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2


def _emit(text: str) -> None:
    """Write one line to stdout.

    Args:
        text: The line to write, without a trailing newline.
    """
    sys.stdout.write(f"{text}\n")


def local_source_config(
    settings: Settings,
    *,
    tenant_id: str,
    root: Path,
    include_globs: Sequence[str] = (),
    exclude_globs: Sequence[str] = (),
    classification: Classification = Classification.INTERNAL,
) -> SourceConfig:
    """Build an ephemeral source config for the local-filesystem connector.

    Args:
        settings: Process settings; supplies the default document language.
        tenant_id: Tenant the ingested documents belong to.
        root: Directory to walk.
        include_globs: Patterns to include; empty means the connector default.
        exclude_globs: Patterns to skip.
        classification: Default classification for documents from this run.

    Returns:
        A :class:`~ragcore.models.document.SourceConfig` the pipeline can consume with
        no ``source_configs`` row present.
    """
    options: dict[str, object] = {"root": str(root)}
    if include_globs:
        options["include_globs"] = list(include_globs)
    if exclude_globs:
        options["exclude_globs"] = list(exclude_globs)
    return SourceConfig(
        source_id=f"local-{root.name or 'documents'}",
        tenant_id=tenant_id,
        source_type=SourceType.LOCAL,
        name=f"Local filesystem: {root}",
        options=options,
        default_classification=classification,
        inherit_source_permissions=False,
        doc_type="document",
        tags=["local"],
        language=settings.pii_language,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        A parser exposing the ``run`` subcommand.
    """
    parser = argparse.ArgumentParser(
        prog="ingestion.cli",
        description="Run the ingestion pipeline outside Azure Functions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one ingestion pass")
    run.add_argument("--tenant", required=True, help="tenant id to ingest for")
    run.add_argument("--source-id", default=None, help="run one registered source")
    run.add_argument(
        "--source-type",
        default=None,
        choices=[member.value for member in SourceType],
        help="run every registered source of this type",
    )
    run.add_argument(
        "--root",
        default=None,
        help="directory for an ephemeral local source (default RAG_INGEST_LOCAL_ROOT)",
    )
    run.add_argument(
        "--include", action="append", default=None, metavar="GLOB", help="include glob"
    )
    run.add_argument(
        "--exclude", action="append", default=None, metavar="GLOB", help="exclude glob"
    )
    run.add_argument(
        "--classification",
        default=Classification.INTERNAL.value,
        choices=[level.value for level in Classification],
        help="default classification for an ephemeral local source",
    )
    run.add_argument(
        "--force", action="store_true", help="override the working-hours guard"
    )
    run.add_argument(
        "--dry-run", action="store_true", help="fetch and parse but write nothing"
    )
    run.add_argument(
        "--full-scan",
        action="store_true",
        help="ignore stored delta cursors so deletions are detected",
    )
    return parser


def _print_summaries(summaries: Sequence[IngestRunSummary]) -> None:
    """Print one block per run summary.

    Args:
        summaries: Summaries returned by the pipeline.
    """
    if not summaries:
        _emit("no ingestion runs were started")
        return
    for summary in summaries:
        duration = summary.duration_seconds
        duration_text = f"{duration:.1f}s" if duration is not None else "unfinished"
        _emit(
            f"run {summary.run_id} source={summary.source_id or '(all)'} "
            f"status={summary.status.value} trigger={summary.trigger.value} "
            f"({duration_text})"
        )
        _emit(
            f"  documents seen={summary.documents_seen} "
            f"created={summary.documents_created} "
            f"updated={summary.documents_updated} "
            f"deleted={summary.documents_deleted} "
            f"skipped={summary.documents_skipped} "
            f"failed={summary.documents_failed}"
        )
        _emit(
            f"  chunks upserted={summary.chunks_upserted} "
            f"deleted={summary.chunks_deleted} "
            f"tokens={summary.tokens_embedded} "
            f"duplicates_dropped={summary.duplicates_dropped} "
            f"pii_documents={summary.pii_documents}"
        )
        if summary.metrics.get("dry_run"):
            _emit(
                "  dry run: would write "
                f"{int(summary.metrics.get('chunks_planned', 0))} chunks / "
                f"{int(summary.metrics.get('tokens_planned', 0))} tokens"
            )
        if summary.skip_reason:
            _emit(f"  skipped because: {summary.skip_reason}")
        if summary.error_message:
            _emit(f"  error: {summary.error_message}")
        for error in summary.errors[:10]:
            _emit(f"    - {error}")


async def _run(args: argparse.Namespace, settings: Settings) -> int:
    """Execute the ``run`` subcommand.

    Args:
        args: Parsed arguments.
        settings: Process settings.

    Returns:
        A process exit code.
    """
    from ingestion.pipeline import resolve_sources, run_ingest

    source_type = SourceType(args.source_type) if args.source_type else None
    sources: list[SourceConfig] | None = None
    root: Path | None = args.resolved_root

    if root is not None:
        try:
            registered = await resolve_sources(
                tenant_id=args.tenant, source_type=SourceType.LOCAL, settings=settings
            )
        except Exception as exc:
            # No database yet is the normal state of a first local run, and a dry run
            # needs none at all. Fall back to the ephemeral source rather than failing.
            _log.warning("cli.source_lookup_failed", error=type(exc).__name__)
            registered = []
        if not registered:
            if not args.root_exists:
                _emit(f"error: local source root {root} does not exist")
                return EXIT_FAILED
            sources = [
                local_source_config(
                    settings,
                    tenant_id=args.tenant,
                    root=root,
                    include_globs=args.include or (),
                    exclude_globs=args.exclude or (),
                    classification=Classification(args.classification),
                )
            ]
            _emit(f"using an ephemeral local source rooted at {root}")

    summaries = await run_ingest(
        tenant_id=args.tenant,
        source_id=args.source_id,
        source_type=None if sources else source_type,
        sources=sources,
        trigger=IngestTrigger.MANUAL,
        force=args.force,
        dry_run=args.dry_run,
        full_scan=args.full_scan,
        settings=settings,
    )
    _print_summaries(summaries)

    if any(summary.skip_reason for summary in summaries):
        return EXIT_REFUSED
    if any(summary.documents_failed or summary.error_message for summary in summaries):
        return EXIT_FAILED
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument vector excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` on failure, ``2`` when the working-hours guard refused
        the run.
    """
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    settings = get_settings()
    configure_logging(settings)

    # Filesystem probing happens here rather than inside the coroutine: blocking
    # stat() calls have no business on the event loop.
    args.resolved_root = None
    args.root_exists = False
    wants_local = args.root is not None or args.source_type == SourceType.LOCAL.value
    if wants_local and args.source_id is None:
        args.resolved_root = (
            Path(args.root or settings.ingest_local_root).expanduser().resolve()
        )
        args.root_exists = args.resolved_root.is_dir()

    try:
        return asyncio.run(_run(args, settings))
    except Exception as exc:
        _log.exception("cli.failed", error=type(exc).__name__)
        _emit(f"error: {type(exc).__name__}")
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
