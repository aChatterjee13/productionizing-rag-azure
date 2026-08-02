#!/usr/bin/env python3
"""End-to-end smoke test: ingest -> search -> chat -> citations, plus ACL negatives.

Runs against a **live** API (local, or a deployed Container App) and asserts the whole
path works and that the security boundary holds:

1. ``/health`` and ``/readyz`` answer.
2. **Ingest.** A document with a unique canary is uploaded through
   ``POST /documents``, then polled through ``POST /search`` until it is retrievable.
3. **Search.** A seeded query returns chunks, and every chunk is inside the caller's
   tenant and at or below the caller's clearance.
4. **Chat + citations.** The SSE stream emits tokens and a ``citations`` event, and
   every citation points at a chunk the same turn retrieved.
5. **ACL negatives.** The ``PUBLIC``-clearance intern persona cannot retrieve or be
   told any canary from a document above its clearance, and the other tenant's analyst
   never sees an Acme document.

Personas come from :mod:`seed_demo_tenant`, so the fixture and the assertions can never
drift apart. Run ``scripts/seed_demo_tenant.py`` first.

    uv run python scripts/smoke_test.py --base-url http://localhost:8000
    uv run python scripts/smoke_test.py --base-url https://... --skip-ingest
    uv run python scripts/smoke_test.py --auth-mode bearer   # SMOKE_TOKEN_<PERSONA>

Exit code 0 means every check passed; 1 means at least one failed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ragcore.logging import configure_logging, get_logger
from ragcore.models.acl import Classification
from ragcore.settings import Settings, get_settings

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from seed_demo_tenant import (  # noqa: E402 - sibling script, path set just above
    CANARIES,
    PERSONAS,
    TENANT_ACME,
    forbidden_canaries_for,
)

logger = get_logger("scripts.smoke_test")

EXIT_OK = 0
EXIT_FAILED = 1

#: Multipart field names tried in order by the upload step. The contract fixes the
#: endpoint (``POST /documents``, multipart) but not the field name.
_UPLOAD_FIELD_NAMES = ("file", "files", "upload", "document")


class SmokeCheckError(Exception):
    """Raised when an assertion about the running system does not hold."""


@dataclass(slots=True)
class CheckResult:
    """Outcome of one smoke check.

    Attributes:
        name: Human-readable check name.
        passed: Whether the check held.
        detail: Supporting detail, or the failure reason.
        seconds: Wall-clock duration.
    """

    name: str
    passed: bool
    detail: str = ""
    seconds: float = 0.0


@dataclass(slots=True)
class SmokeContext:
    """Shared state for a smoke run.

    Attributes:
        client: HTTP client bound to the API base URL.
        settings: Resolved platform settings.
        prefix: Versioned API prefix, e.g. ``/api/v1``.
        auth_mode: ``dev`` (unsigned dev principal header) or ``bearer``.
        tokens: Persona key -> bearer token, used when ``auth_mode`` is ``bearer``.
        results: Accumulated check results.
        uploaded_canary: Canary embedded in the document uploaded by the ingest check.
        uploaded_title: Title of that document.
    """

    client: httpx.AsyncClient
    settings: Settings
    prefix: str
    auth_mode: str
    tokens: dict[str, str] = field(default_factory=dict)
    results: list[CheckResult] = field(default_factory=list)
    uploaded_canary: str = ""
    uploaded_title: str = ""


# --------------------------------------------------------------------------- helpers
def require(condition: bool, message: str) -> None:
    """Fail the current check when a condition does not hold.

    Args:
        condition: What must be true.
        message: Failure message.

    Raises:
        SmokeCheckError: When ``condition`` is false.
    """
    if not condition:
        raise SmokeCheckError(message)


def auth_headers(context: SmokeContext, persona: str) -> dict[str, str]:
    """Build the request headers that authenticate as one persona.

    Args:
        context: Smoke run context.
        persona: Persona key from :data:`seed_demo_tenant.PERSONAS`.

    Returns:
        Headers carrying either the dev principal or a bearer token.

    Raises:
        SmokeCheckError: If bearer mode is selected and no token is configured.
    """
    if context.auth_mode == "bearer":
        token = context.tokens.get(persona)
        require(
            bool(token),
            f"no bearer token for persona {persona!r}: set "
            f"SMOKE_TOKEN_{persona.upper()} or use --auth-mode dev",
        )
        return {"Authorization": f"Bearer {token}"}
    principal = PERSONAS[persona]
    header = context.settings.entra_dev_principal_header
    return {header: principal.model_dump_json()}


def payload_of(chunk: dict[str, Any]) -> dict[str, Any]:
    """Extract the chunk payload from a ``RetrievedChunk`` JSON object.

    Args:
        chunk: One entry of ``RetrievalResult.chunks``.

    Returns:
        The ``payload`` object, or the chunk itself when it is already flat.
    """
    payload = chunk.get("payload")
    if isinstance(payload, dict):
        return payload
    return chunk


def chunk_text(chunk: dict[str, Any]) -> str:
    """Concatenate every text-bearing field of a retrieved chunk.

    Args:
        chunk: One entry of ``RetrievalResult.chunks``.

    Returns:
        Text, contextual header and summary joined, for substring assertions.
    """
    payload = payload_of(chunk)
    parts = [
        str(payload.get("text") or ""),
        str(payload.get("contextual_header") or ""),
        str(payload.get("summary") or ""),
    ]
    return "\n".join(parts)


async def search(
    context: SmokeContext,
    persona: str,
    query: str,
    **extra: Any,
) -> dict[str, Any]:
    """Call ``POST /search`` as one persona.

    Args:
        context: Smoke run context.
        persona: Persona key.
        query: Query text.
        **extra: Additional body fields (for example ``filters``).

    Returns:
        The parsed ``RetrievalResult`` body.

    Raises:
        SmokeCheckError: If the endpoint does not answer 200.
    """
    body: dict[str, Any] = {"query": query, **extra}
    response = await context.client.post(
        f"{context.prefix}/search",
        json=body,
        headers=auth_headers(context, persona),
    )
    require(
        response.status_code == httpx.codes.OK,
        f"POST {context.prefix}/search as {persona} returned "
        f"{response.status_code}: {response.text[:300]}",
    )
    return response.json()


@dataclass(slots=True)
class ChatTurn:
    """What one streamed chat turn produced.

    Attributes:
        answer: Concatenated ``token`` event text.
        citations: Citation objects from the ``citations`` event.
        retrieved_chunk_ids: Chunk ids seen in the ``retrieval`` event.
        events: Every event name received, in order.
        guardrails: Guardrail events received.
        errors: Payloads of any ``error`` events.
    """

    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    guardrails: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


def _collect_citations(data: Any) -> list[dict[str, Any]]:
    """Normalise the payload of a ``citations`` SSE event into a list.

    Args:
        data: Decoded event data.

    Returns:
        A list of citation objects.
    """
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        inner = data.get("citations")
        if isinstance(inner, list):
            return [item for item in inner if isinstance(item, dict)]
        return [data]
    return []


def _collect_chunk_ids(data: Any) -> list[str]:
    """Pull chunk ids out of the payload of a ``retrieval`` SSE event.

    Args:
        data: Decoded event data (a ``RetrievalResult.without_text()`` object).

    Returns:
        Every ``chunk_id`` found, de-duplicated, order preserved.
    """
    if not isinstance(data, dict):
        return []
    found: list[str] = []
    for chunk in data.get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        chunk_id = payload_of(chunk).get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id not in found:
            found.append(chunk_id)
    return found


async def chat(
    context: SmokeContext,
    persona: str,
    message: str,
    *,
    read_timeout: float,
    session_id: str | None = None,
) -> ChatTurn:
    """Run one streamed chat turn and decode the SSE events.

    Args:
        context: Smoke run context.
        persona: Persona key.
        message: User message.
        read_timeout: Read timeout in seconds for the stream. Not named ``timeout``,
            which would read as an async cancellation budget rather than a socket read.
        session_id: Existing session to continue, if any.

    Returns:
        The decoded :class:`ChatTurn`.

    Raises:
        SmokeCheckError: If the endpoint does not answer 200.
    """
    body: dict[str, Any] = {"message": message, "stream": True}
    if session_id:
        body["session_id"] = session_id

    turn = ChatTurn()
    headers = {
        **auth_headers(context, persona),
        "Accept": "text/event-stream",
    }
    async with context.client.stream(
        "POST",
        f"{context.prefix}/chat",
        json=body,
        headers=headers,
        timeout=httpx.Timeout(read_timeout, connect=30.0),
    ) as response:
        require(
            response.status_code == httpx.codes.OK,
            f"POST {context.prefix}/chat as {persona} returned {response.status_code}",
        )
        event_name = ""
        data_lines: list[str] = []

        def flush() -> None:
            """Decode one accumulated SSE event into the turn."""
            nonlocal event_name, data_lines
            if not data_lines:
                event_name = ""
                return
            raw = "\n".join(data_lines)
            data_lines = []
            name = event_name or "message"
            event_name = ""
            turn.events.append(name)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = raw
            if name == "token":
                if isinstance(data, dict):
                    turn.answer += str(data.get("text") or "")
                else:
                    turn.answer += str(data)
            elif name == "citations":
                turn.citations.extend(_collect_citations(data))
            elif name == "retrieval":
                turn.retrieved_chunk_ids.extend(_collect_chunk_ids(data))
            elif name == "guardrail" and isinstance(data, dict):
                turn.guardrails.append(data)
            elif name == "error" and isinstance(data, dict):
                turn.errors.append(data)

        async for line in response.aiter_lines():
            if line == "":
                flush()
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip())
        flush()
    return turn


# ---------------------------------------------------------------------------- checks
async def check_health(context: SmokeContext) -> str:
    """Assert the liveness and readiness probes answer.

    Args:
        context: Smoke run context.

    Returns:
        A detail string naming the paths that answered.

    Raises:
        SmokeCheckError: If neither the prefixed nor the bare path answers 200.
    """
    answered: list[str] = []
    for name in ("health", "readyz"):
        candidates = (f"{context.prefix}/{name}", f"/{name}")
        for path in candidates:
            response = await context.client.get(path)
            if response.status_code == httpx.codes.OK:
                answered.append(path)
                break
        else:
            raise SmokeCheckError(
                f"neither {candidates[0]} nor {candidates[1]} returned 200"
            )
    return ", ".join(answered)


async def check_ingest(context: SmokeContext, *, wait_seconds: float) -> str:
    """Upload a document and wait until it is retrievable.

    This is the ingest half of the end-to-end path: upload through the API, then poll
    ``/search`` for the document's unique canary until the chunk is indexed.

    Args:
        context: Smoke run context.
        wait_seconds: Seconds to wait for the document to become searchable.

    Returns:
        A detail string with the document title and how long indexing took.

    Raises:
        SmokeCheckError: If the upload is rejected or the document never becomes
            searchable.
    """
    canary = f"CANARY-SMOKE-{uuid.uuid4().hex[:12].upper()}"
    title = f"Smoke test note {canary}"
    context.uploaded_canary = canary
    context.uploaded_title = title
    content = (
        f"# {title}\n\n"
        "This note was created by scripts/smoke_test.py to prove that the ingestion "
        "path indexes an uploaded document and that retrieval can find it again.\n\n"
        f"The smoke canary for this run is {canary}. The agreed smoke-test answer is "
        "that the canary value is unique per run.\n"
    )

    last: httpx.Response | None = None
    accepted = None
    for field_name in _UPLOAD_FIELD_NAMES:
        response = await context.client.post(
            f"{context.prefix}/documents",
            files={field_name: (f"{canary.lower()}.md", content, "text/markdown")},
            headers=auth_headers(context, "acme_admin"),
        )
        last = response
        if response.status_code in {
            httpx.codes.OK,
            httpx.codes.CREATED,
            httpx.codes.ACCEPTED,
        }:
            accepted = field_name
            break

    require(
        accepted is not None,
        f"POST {context.prefix}/documents rejected every multipart field name "
        f"{_UPLOAD_FIELD_NAMES}: last status "
        f"{last.status_code if last else 'n/a'} "
        f"{last.text[:300] if last else ''}",
    )

    deadline = time.monotonic() + wait_seconds
    attempts = 0
    while time.monotonic() < deadline:
        attempts += 1
        result = await search(context, "acme_admin", canary)
        for chunk in result.get("chunks") or []:
            if canary in chunk_text(chunk):
                elapsed = wait_seconds - (deadline - time.monotonic())
                return (
                    f"uploaded as multipart field {accepted!r}, searchable after "
                    f"{elapsed:.1f}s ({attempts} polls)"
                )
        await asyncio.sleep(2.0)

    raise SmokeCheckError(
        f"uploaded document {title!r} was not searchable within "
        f"{wait_seconds:.0f}s; "
        "ingestion accepted the upload but never indexed it"
    )


async def check_search_scoping(context: SmokeContext) -> str:
    """Assert search returns chunks and never breaches tenant or clearance.

    Args:
        context: Smoke run context.

    Returns:
        A detail string with the number of chunks inspected.

    Raises:
        SmokeCheckError: If nothing is returned, or a chunk is out of tenant or
            above the caller's clearance.
    """
    persona = "acme_engineer"
    principal = PERSONAS[persona]
    result = await search(context, persona, "What is the daily meal allowance?")
    chunks = result.get("chunks") or []
    require(
        len(chunks) > 0,
        "search returned no chunks for a question the seeded corpus answers; "
        "run scripts/seed_demo_tenant.py",
    )

    clearance = principal.clearance_rank()
    for chunk in chunks:
        payload = payload_of(chunk)
        require(
            payload.get("tenant_id") == principal.tenant_id,
            f"cross-tenant chunk {payload.get('chunk_id')} from tenant "
            f"{payload.get('tenant_id')!r} returned to {principal.tenant_id!r}",
        )
        rank = payload.get("classification_rank")
        require(
            isinstance(rank, int) and rank <= clearance,
            f"chunk {payload.get('chunk_id')} has classification_rank {rank} above "
            f"the caller's clearance {clearance}",
        )
        require(
            not payload.get("is_deleted", False),
            f"soft-deleted chunk {payload.get('chunk_id')} was returned",
        )

    forbidden = forbidden_canaries_for(principal)
    joined = "\n".join(chunk_text(chunk) for chunk in chunks)
    leaked = [token for token in forbidden if token in joined]
    require(not leaked, f"search leaked canaries {leaked} to {persona}")

    return f"{len(chunks)} chunks, all in {principal.tenant_id} at rank <= {clearance}"


async def check_chat_citations(context: SmokeContext, *, read_timeout: float) -> str:
    """Assert a chat turn streams an answer with verifiable citations.

    Args:
        context: Smoke run context.
        read_timeout: Stream read timeout in seconds.

    Returns:
        A detail string with the answer length and citation count.

    Raises:
        SmokeCheckError: If the stream produces no answer, no citations, or a citation
            that points at a chunk the same turn never retrieved.
    """
    persona = "acme_engineer"
    turn = await chat(
        context,
        persona,
        "What is the current daily meal allowance, and which policy says so?",
        read_timeout=read_timeout,
    )

    require(not turn.errors, f"chat stream emitted error events: {turn.errors}")
    require(
        "done" in turn.events,
        f"chat stream never emitted a 'done' event; saw {sorted(set(turn.events))}",
    )
    require(
        len(turn.answer.strip()) > 0,
        f"chat stream produced no token text; events: {sorted(set(turn.events))}",
    )
    require(
        len(turn.citations) > 0,
        "chat stream produced no citations for a question the corpus answers; "
        "requirement #9 needs every answer grounded in cited chunks",
    )

    if turn.retrieved_chunk_ids:
        retrieved = set(turn.retrieved_chunk_ids)
        unknown = [
            citation.get("chunk_id")
            for citation in turn.citations
            if citation.get("chunk_id") not in retrieved
        ]
        require(
            not unknown,
            f"citations reference chunks that were not retrieved this turn: {unknown}",
        )

    markers = [citation.get("marker") for citation in turn.citations]
    return (
        f"{len(turn.answer)} chars, {len(turn.citations)} citations "
        f"{markers}, events={sorted(set(turn.events))}"
    )


async def check_low_clearance_denied(
    context: SmokeContext, *, read_timeout: float
) -> str:
    """Assert the ``PUBLIC``-clearance persona cannot reach restricted content.

    Args:
        context: Smoke run context.
        read_timeout: Stream read timeout in seconds.

    Returns:
        A detail string naming the canaries that were correctly withheld.

    Raises:
        SmokeCheckError: If a restricted chunk or canary reaches the intern.
    """
    persona = "acme_intern"
    principal = PERSONAS[persona]
    forbidden = forbidden_canaries_for(principal)
    require(
        len(forbidden) > 0,
        "the intern persona has nothing to be denied; the fixture is wrong",
    )

    probes = (
        "What are the FY25 engineering salary bands and their base ranges?",
        "Summarise security incident 2025-0142 and how it was contained.",
        "What does clause 7 of the contractor NDA addendum say?",
    )

    for probe in probes:
        result = await search(context, persona, probe)
        chunks = result.get("chunks") or []
        for chunk in chunks:
            payload = payload_of(chunk)
            require(
                payload.get("tenant_id") == principal.tenant_id,
                f"cross-tenant chunk returned to {persona}: "
                f"{payload.get('tenant_id')!r}",
            )
            rank = payload.get("classification_rank")
            require(
                isinstance(rank, int) and rank <= principal.clearance_rank(),
                f"chunk {payload.get('chunk_id')} at rank {rank} returned to a "
                f"principal cleared for {principal.clearance_rank()}",
            )
        text = "\n".join(chunk_text(chunk) for chunk in chunks)
        leaked = [token for token in forbidden if token in text]
        require(not leaked, f"search leaked {leaked} to {persona} for {probe!r}")

    turn = await chat(context, persona, probes[0], read_timeout=read_timeout)
    leaked = [token for token in forbidden if token in turn.answer]
    require(not leaked, f"chat leaked {leaked} to {persona}")
    for citation in turn.citations:
        document_id = citation.get("document_id")
        require(
            document_id not in CANARIES or CANARIES[document_id] not in forbidden,
            f"chat cited restricted document {document_id} to {persona}",
        )

    return f"{len(forbidden)} canaries withheld across {len(probes)} probes + 1 chat"


async def check_cross_tenant_isolation(context: SmokeContext) -> str:
    """Assert the other tenant's analyst never sees an Acme document.

    The two tenants carry a nearly identical travel policy with different amounts, so a
    leak surfaces as the wrong document id rather than a plausible-looking answer.

    Args:
        context: Smoke run context.

    Returns:
        A detail string with the number of chunks inspected.

    Raises:
        SmokeCheckError: If any Acme document or canary reaches the Globex analyst.
    """
    persona = "globex_analyst"
    principal = PERSONAS[persona]
    result = await search(context, persona, "What is the daily meal allowance?")
    chunks = result.get("chunks") or []

    acme_documents = [
        payload_of(chunk).get("document_id")
        for chunk in chunks
        if str(payload_of(chunk).get("tenant_id")) == TENANT_ACME
    ]
    require(
        not acme_documents,
        f"tenant leak: {persona} received Acme documents {acme_documents}",
    )
    for chunk in chunks:
        require(
            payload_of(chunk).get("tenant_id") == principal.tenant_id,
            f"chunk from tenant {payload_of(chunk).get('tenant_id')!r} returned to "
            f"{principal.tenant_id!r}",
        )

    text = "\n".join(chunk_text(chunk) for chunk in chunks)
    leaked = [
        token
        for document_id, token in CANARIES.items()
        if document_id.startswith("doc-acme-") and token in text
    ]
    require(not leaked, f"tenant leak: Acme canaries {leaked} reached {persona}")

    return f"{len(chunks)} chunks, none from {TENANT_ACME}"


# ----------------------------------------------------------------------------- runner
async def run_check(
    context: SmokeContext,
    name: str,
    coro: Any,
) -> bool:
    """Await one check coroutine and record its result.

    Args:
        context: Smoke run context.
        name: Check name for the report.
        coro: The awaitable performing the check; it returns a detail string.

    Returns:
        True when the check passed.
    """
    started = time.monotonic()
    try:
        detail = await coro
    except SmokeCheckError as exc:
        context.results.append(
            CheckResult(name, False, str(exc), time.monotonic() - started)
        )
        return False
    except Exception as exc:
        context.results.append(
            CheckResult(
                name,
                False,
                f"{type(exc).__name__}: {exc}",
                time.monotonic() - started,
            )
        )
        return False
    context.results.append(CheckResult(name, True, detail, time.monotonic() - started))
    return True


def load_tokens() -> dict[str, str]:
    """Read per-persona bearer tokens from the environment.

    Returns:
        Persona key -> token for every ``SMOKE_TOKEN_<PERSONA>`` variable that is set.
    """
    tokens: dict[str, str] = {}
    for persona in PERSONAS:
        value = os.environ.get(f"SMOKE_TOKEN_{persona.upper()}")
        if value:
            tokens[persona] = value
    return tokens


async def run(args: argparse.Namespace, settings: Settings) -> int:
    """Execute every smoke check against a running API.

    Args:
        args: Parsed command-line arguments.
        settings: Resolved platform settings.

    Returns:
        A process exit code.
    """
    prefix = args.prefix or settings.api_prefix
    timeouts = httpx.Timeout(args.timeout, connect=30.0)
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        timeout=timeouts,
        follow_redirects=True,
    ) as client:
        context = SmokeContext(
            client=client,
            settings=settings,
            prefix=prefix,
            auth_mode=args.auth_mode,
            tokens=load_tokens(),
        )

        print(f"smoke test against {args.base_url}{prefix} (auth={args.auth_mode})")
        if args.auth_mode == "dev":
            print(
                "  dev auth: sending an unsigned principal in "
                f"{settings.entra_dev_principal_header!r}; the API must be running "
                "with RAG_ENTRA_DEV_MODE=true"
            )

        healthy = await run_check(context, "health + readyz", check_health(context))
        if not healthy:
            report(context)
            return EXIT_FAILED

        if args.skip_ingest:
            context.results.append(
                CheckResult("ingest -> searchable", True, "skipped (--skip-ingest)")
            )
        else:
            await run_check(
                context,
                "ingest -> searchable",
                check_ingest(context, wait_seconds=args.ingest_timeout),
            )

        await run_check(context, "search scoping", check_search_scoping(context))
        await run_check(
            context,
            "chat + citations",
            check_chat_citations(context, read_timeout=args.timeout),
        )
        await run_check(
            context,
            "low clearance denied",
            check_low_clearance_denied(context, read_timeout=args.timeout),
        )
        await run_check(
            context,
            "cross-tenant isolation",
            check_cross_tenant_isolation(context),
        )

        report(context)
        return EXIT_OK if all(r.passed for r in context.results) else EXIT_FAILED


def report(context: SmokeContext) -> None:
    """Print the check report.

    Args:
        context: Smoke run context holding the results.
    """
    print()
    width = max((len(r.name) for r in context.results), default=10)
    for result in context.results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"  [{status}] {result.name:<{width}}  {result.seconds:6.2f}s  "
            f"{result.detail}"
        )
    failed = [r for r in context.results if not r.passed]
    print()
    print(f"{len(context.results) - len(failed)}/{len(context.results)} checks passed")
    if failed:
        print("failed checks:", ", ".join(r.name for r in failed), file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="smoke_test.py",
        description=(
            "End-to-end smoke test: ingest -> search -> chat -> citations, plus the "
            "ACL negatives that prove a low-clearance user and another tenant cannot "
            "reach restricted content."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SMOKE_BASE_URL", "http://localhost:8000"),
        help="API base URL (default: $SMOKE_BASE_URL or http://localhost:8000)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="API prefix override (default: ragcore settings api_prefix)",
    )
    parser.add_argument(
        "--auth-mode",
        choices=("dev", "bearer"),
        default=os.environ.get("SMOKE_AUTH_MODE", "dev"),
        help="how to authenticate: unsigned dev principal, or SMOKE_TOKEN_<PERSONA>",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("SMOKE_TIMEOUT", "300")),
        help="per-request timeout in seconds (a chat turn can run minutes)",
    )
    parser.add_argument(
        "--ingest-timeout",
        type=float,
        default=float(os.environ.get("SMOKE_INGEST_TIMEOUT", "180")),
        help="seconds to wait for an uploaded document to become searchable",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="skip the upload step and test search/chat against the seeded corpus only",
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

    # Sanity-check the fixture before touching the network: a broken persona table
    # would make every ACL assertion meaningless.
    intern = PERSONAS["acme_intern"]
    if intern.max_classification is not Classification.PUBLIC:
        print(
            "fixture error: the intern persona must have PUBLIC clearance for the "
            "ACL negative checks to mean anything",
            file=sys.stderr,
        )
        return EXIT_FAILED

    try:
        return asyncio.run(run(args, settings))
    except httpx.HTTPError as exc:
        print(f"error: cannot reach {args.base_url}: {exc}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
