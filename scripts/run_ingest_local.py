#!/usr/bin/env python3
"""Run one ingestion pass outside Azure Functions.

Same pipeline the timer-triggered Function App runs — connector fetch, parse, chunk,
PII scan, dedupe, embed, upsert, manifest update — driven from a shell so it can be
debugged with a normal debugger and pointed at a local directory.

    uv run python scripts/run_ingest_local.py --tenant tenant-acme --force
    uv run python scripts/run_ingest_local.py --root ./data/documents --dry-run
    uv run python scripts/run_ingest_local.py --source-id src-acme-policies

By default it builds an ephemeral ``SourceConfig`` for the local-filesystem connector
rooted at ``RAG_INGEST_LOCAL_ROOT``, so nothing needs to be registered in the database
first. Pass ``--source-id`` to run a source that *is* registered instead.

The working-hours guard is honoured exactly as the scheduled run honours it
(``Settings.may_start_scheduled_ingest``): a run started inside working hours is
refused unless ``--force`` is given, so this script cannot accidentally hammer a source
during the business day.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import runpy
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ragcore.db import repositories as repo
from ragcore.db import session_scope
from ragcore.logging import configure_logging, get_logger
from ragcore.models.acl import Classification
from ragcore.models.document import (
    IngestRunSummary,
    IngestTrigger,
    SourceConfig,
    SourceType,
)
from ragcore.settings import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

logger = get_logger("scripts.run_ingest_local")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
EXIT_NO_ENTRYPOINT = 3

#: Where the ingestion service is expected to expose the operational entry point,
#: in priority order. See "Addendum B" in docs/CONTRACTS.md for the signature.
_ENTRYPOINTS: tuple[tuple[str, str], ...] = (
    ("ingestion.pipeline", "run_ingest"),
    ("ingestion", "run_ingest"),
    ("ingestion.orchestrator", "run_ingest"),
)

#: Module providing the argv-driven fallback (`python -m ingestion.cli run ...`).
_CLI_MODULE = "ingestion.cli"


class IngestRunner(Protocol):
    """The callable ``services/ingestion`` exposes for operational runs."""

    async def __call__(
        self,
        *,
        tenant_id: str,
        source_id: str | None = None,
        source_type: SourceType | None = None,
        sources: Sequence[SourceConfig] | None = None,
        trigger: IngestTrigger = IngestTrigger.MANUAL,
        force: bool = False,
        dry_run: bool = False,
        settings: Settings | None = None,
    ) -> list[IngestRunSummary]:
        """Run ingestion for one tenant and return one summary per source."""
        ...


def resolve_runner() -> tuple[IngestRunner | None, str]:
    """Find the ingestion entry point on the import path.

    Returns:
        A ``(runner, location)`` pair. ``runner`` is None when no candidate module
        exposes ``run_ingest``, in which case ``location`` explains what was tried.
    """
    tried: list[str] = []
    for module_name, attribute in _ENTRYPOINTS:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            tried.append(f"{module_name} (import failed: {exc})")
            continue
        runner = getattr(module, attribute, None)
        if callable(runner):
            return runner, f"{module_name}.{attribute}"
        tried.append(f"{module_name} (no {attribute})")
    return None, "; ".join(tried)


def local_source_config(
    settings: Settings,
    *,
    tenant_id: str,
    root: Path,
    include_globs: Sequence[str],
    exclude_globs: Sequence[str],
    classification: Classification,
) -> SourceConfig:
    """Build an ephemeral source config for the local-filesystem connector.

    Args:
        settings: Resolved platform settings; supplies the default language.
        tenant_id: Tenant the ingested documents belong to.
        root: Directory to walk.
        include_globs: Patterns to include; empty means the connector default.
        exclude_globs: Patterns to skip.
        classification: Default classification applied to documents from this run.

    Returns:
        A :class:`~ragcore.models.document.SourceConfig` the pipeline can consume
        without a database row existing for it.
    """
    options: dict[str, Any] = {"root": str(root)}
    if include_globs:
        options["include_globs"] = list(include_globs)
    if exclude_globs:
        options["exclude_globs"] = list(exclude_globs)
    return SourceConfig(
        source_id=f"local-{root.name or 'documents'}",
        tenant_id=tenant_id,
        source_type=SourceType.LOCAL,
        name=f"Local filesystem: {root}",
        enabled=True,
        options=options,
        default_classification=classification,
        inherit_source_permissions=False,
        doc_type="document",
        tags=["local"],
        language=settings.pii_language,
    )


async def ensure_tenant(settings: Settings, tenant_id: str) -> None:
    """Make sure the tenant row exists before documents reference it.

    Every tenant-scoped table has a foreign key onto ``tenants``, so a first local run
    against a fresh database fails without this.

    Args:
        settings: Resolved platform settings.
        tenant_id: Tenant to create or refresh.
    """
    async with session_scope(settings) as session:
        await repo.upsert_tenant(
            session,
            tenant_id=tenant_id,
            name=tenant_id,
            settings_blob={"created_by": "scripts/run_ingest_local.py"},
        )
        await session.commit()


def describe_root(root: Path) -> str:
    """Summarise what the local connector will find.

    Args:
        root: Directory the connector will walk.

    Returns:
        A human-readable description of the directory contents.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not root.exists():
        msg = (
            f"local source root {root} does not exist. Create it, point --root "
            "somewhere else, or set RAG_INGEST_LOCAL_ROOT."
        )
        raise FileNotFoundError(msg)
    files = [path for path in root.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    return f"{len(files)} files, {total_bytes / 1024:.1f} KiB under {root}"


def print_summaries(summaries: Sequence[IngestRunSummary]) -> None:
    """Print one line per ingestion run plus its counters.

    Args:
        summaries: Summaries returned by the pipeline.
    """
    if not summaries:
        print("no ingestion runs were started")
        return
    for summary in summaries:
        duration = summary.duration_seconds
        duration_text = f"{duration:.1f}s" if duration is not None else "unfinished"
        print(
            f"run {summary.run_id} source={summary.source_id or '(all)'} "
            f"status={summary.status.value} trigger={summary.trigger.value} "
            f"({duration_text})"
        )
        print(
            f"  documents seen={summary.documents_seen} "
            f"created={summary.documents_created} "
            f"updated={summary.documents_updated} "
            f"deleted={summary.documents_deleted} "
            f"skipped={summary.documents_skipped} "
            f"failed={summary.documents_failed}"
        )
        print(
            f"  chunks upserted={summary.chunks_upserted} "
            f"deleted={summary.chunks_deleted} "
            f"tokens={summary.tokens_embedded} "
            f"duplicates_dropped={summary.duplicates_dropped} "
            f"pii_documents={summary.pii_documents}"
        )
        if summary.skip_reason:
            print(f"  skipped because: {summary.skip_reason}")
        if summary.error_message:
            print(f"  error: {summary.error_message}")
        for error in summary.errors[:10]:
            print(f"    - {error}")


def run_via_cli(argv: Sequence[str]) -> int:
    """Delegate to ``ingestion.cli`` when no importable ``run_ingest`` exists.

    Args:
        argv: Arguments to pass to the ingestion CLI, excluding the program name.

    Returns:
        The CLI's exit code.

    Raises:
        ModuleNotFoundError: If ``ingestion.cli`` cannot be imported either.
    """
    print(f"delegating to `python -m {_CLI_MODULE} {' '.join(argv)}`")
    module = importlib.import_module(_CLI_MODULE)
    entry = getattr(module, "main", None)
    if callable(entry):
        result = entry(list(argv))
        return int(result or EXIT_OK)
    original = sys.argv
    sys.argv = [_CLI_MODULE, *argv]
    try:
        runpy.run_module(_CLI_MODULE, run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or EXIT_OK)
    finally:
        sys.argv = original
    return EXIT_OK


async def run(args: argparse.Namespace, settings: Settings) -> int:
    """Execute one ingestion pass.

    Args:
        args: Parsed command-line arguments.
        settings: Resolved platform settings.

    Returns:
        A process exit code.
    """
    runner, location = resolve_runner()
    if runner is None:
        print(
            "could not find an ingestion entry point.\n"
            f"  tried: {location}\n"
            "  services/ingestion must expose the coroutine documented in "
            "docs/CONTRACTS.md Addendum B:\n"
            "    async def run_ingest(*, tenant_id, source_id=None, source_type=None,\n"
            "                         sources=None, trigger=IngestTrigger.MANUAL,\n"
            "                         force=False, dry_run=False, settings=None\n"
            "                         ) -> list[IngestRunSummary]\n"
            f"  or run `python -m {_CLI_MODULE} run --help` directly.",
            file=sys.stderr,
        )
        return EXIT_NO_ENTRYPOINT
    print(f"using ingestion entry point {location}")

    allowed, reason = settings.may_start_scheduled_ingest(datetime.now(UTC))
    if args.force:
        print(f"working-hours guard: {reason} (overridden by --force)")
    elif not allowed:
        print(
            f"refusing to start: {reason}. Working hours are "
            f"{settings.ingest_working_hours_start:02d}:00-"
            f"{settings.ingest_working_hours_end:02d}:00 "
            f"{settings.ingest_timezone} on days "
            f"{settings.ingest_working_days}. Pass --force to override.",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    else:
        print(f"working-hours guard: {reason}")

    sources: list[SourceConfig] | None = None
    if args.source_id is None:
        # Resolved and validated in main(): filesystem calls stay out of the event loop.
        sources = [
            local_source_config(
                settings,
                tenant_id=args.tenant,
                root=args.resolved_root,
                include_globs=args.include or (),
                exclude_globs=args.exclude or (),
                classification=Classification(args.classification),
            )
        ]

    if args.ensure_tenant and not args.dry_run:
        await ensure_tenant(settings, args.tenant)

    summaries = await runner(
        tenant_id=args.tenant,
        source_id=args.source_id,
        source_type=None if args.source_id else SourceType.LOCAL,
        sources=sources,
        trigger=IngestTrigger.MANUAL,
        force=args.force,
        dry_run=args.dry_run,
        settings=settings,
    )
    print_summaries(summaries)

    failed = [s for s in summaries if s.documents_failed or s.error_message]
    return EXIT_FAILED if failed else EXIT_OK


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="run_ingest_local.py",
        description=(
            "Run the ingestion pipeline outside Azure Functions, against the "
            "local-filesystem connector by default."
        ),
    )
    parser.add_argument(
        "--tenant",
        default="tenant-acme",
        help="tenant id the ingested documents belong to (default: tenant-acme)",
    )
    parser.add_argument(
        "--source-id",
        default=None,
        help="run a source registered in source_configs instead of a local directory",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="directory for the local connector (default: RAG_INGEST_LOCAL_ROOT)",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        metavar="GLOB",
        help="glob to include; repeatable",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="GLOB",
        help="glob to exclude; repeatable",
    )
    parser.add_argument(
        "--classification",
        default=Classification.INTERNAL.value,
        choices=[level.value for level in Classification],
        help="default classification for documents from this run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="override the working-hours guard",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and parse but write nothing to Qdrant or PostgreSQL",
    )
    parser.add_argument(
        "--no-ensure-tenant",
        dest="ensure_tenant",
        action="store_false",
        help="do not create the tenant row before ingesting",
    )
    parser.add_argument(
        "--via-cli",
        action="store_true",
        help=f"delegate straight to `python -m {_CLI_MODULE} run ...`",
    )
    parser.set_defaults(ensure_tenant=True)
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

    args.resolved_root = None
    if args.source_id is None and not args.via_cli:
        try:
            args.resolved_root = (
                Path(args.root or settings.ingest_local_root).expanduser().resolve()
            )
            print(describe_root(args.resolved_root))
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_FAILED

    if args.via_cli:
        cli_argv = ["run", "--tenant", args.tenant]
        if args.source_id:
            cli_argv += ["--source-id", args.source_id]
        else:
            cli_argv += ["--source-type", SourceType.LOCAL.value]
        if args.force:
            cli_argv.append("--force")
        if args.dry_run:
            cli_argv.append("--dry-run")
        try:
            return run_via_cli(cli_argv)
        except ModuleNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_NO_ENTRYPOINT

    try:
        return asyncio.run(run(args, settings))
    except Exception as exc:
        logger.exception("local_ingest_failed")
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
