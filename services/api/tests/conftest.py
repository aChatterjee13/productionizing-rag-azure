"""Shared fixtures for the API test-suite.

Three things are set up here, and each exists because the alternative would make
a test prove less than it appears to:

* **A real database.** A temporary sqlite file, migrated from
  ``Base.metadata``, and the *real* repositories. Tenant scoping is a security
  boundary, so the tests that assert it must run against real SQL rather than a
  mock that would happily agree with a broken predicate.
* **A real pipeline.** ``POST /chat`` runs the actual
  :class:`~app.rag.orchestrator.Orchestrator` — input guard, transform, OOD gate,
  context assembly, citation verification, output guard, persistence — with only
  the two genuinely external dependencies faked: Qdrant (via the retriever seam)
  and Anthropic (via an injected client). What is asserted is therefore the real
  event order and the real persistence, not a script.
* **Dev-mode principals.** ``entra_dev_mode`` is on, so a test authenticates by
  sending ``Principal.model_dump_json()`` in the dev header, exactly as
  ``scripts/smoke_test.py`` does. :mod:`tests.test_auth` additionally exercises
  the real RS256 path against a generated key pair.

Module-level environment defaults are deliberately narrow — Redis off, no
Anthropic key, a temp database — so the sibling unit suites in this directory,
which construct their own :class:`~ragcore.settings.Settings`, are unaffected.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Environment. Applied at collection time, before any test module imports, and
# limited to values that cannot change what the sibling unit suites measure.
# ---------------------------------------------------------------------------
_DB_PATH = Path(tempfile.mkdtemp(prefix="rag-api-tests-")) / "api.sqlite3"
os.environ.setdefault("RAG_ENV", "local")
os.environ["RAG_DATABASE_URL_OVERRIDE"] = f"sqlite+aiosqlite:///{_DB_PATH}"
os.environ["RAG_ENTRA_DEV_MODE"] = "true"
os.environ["RAG_REDIS_ENABLED"] = "false"
os.environ["RAG_LANGFUSE_ENABLED"] = "false"
# A developer's real key must never reach a test run: it would turn a unit test
# into a billed API call the first time a fake is forgotten.
os.environ.pop("RAG_ANTHROPIC_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from app.deps import reset_rate_limiter, settings_dependency  # noqa: E402
from app.main import create_app  # noqa: E402
from app.rag.context import ContextAssembler  # noqa: E402
from app.rag.memory.long_term import LongTermMemoryStore  # noqa: E402
from app.rag.memory.semantic_cache import SemanticCache  # noqa: E402
from app.rag.memory.short_term import (  # noqa: E402
    InMemorySessionStore,
    ShortTermMemory,
    TokenCounter,
)
from app.rag.orchestrator import Orchestrator  # noqa: E402
from ragcore.db import repositories as repo  # noqa: E402
from ragcore.db.base import Base, get_engine, get_sessionmaker  # noqa: E402
from ragcore.llm.client import (  # noqa: E402
    LLMResponse,
    LLMUsage,
    StreamEvent,
    StreamEventType,
)
from ragcore.models.acl import AccessControl, Classification, Principal  # noqa: E402
from ragcore.models.chunk import ChunkPayload  # noqa: E402
from ragcore.models.retrieval import (  # noqa: E402
    MetadataFilter,
    RetrievalResult,
    RetrievedChunk,
)
from ragcore.observability.langfuse import NoopTracer  # noqa: E402
from ragcore.settings import Settings, get_settings  # noqa: E402

# --------------------------------------------------------------------- fixture
# The demo fixture's tenants and personas, so a golden item and a test name the
# same identifiers. Mirrors `scripts/seed_demo_tenant.py`.
TENANT_A = "tenant-acme"
TENANT_B = "tenant-globex"

# The scripted answer quotes the fake chunk verbatim, so stage 11 verifies the
# citation for real. It is also deliberately free of tokens the PII recognisers
# treat as credential-shaped, so the stage 12 egress scan leaves it alone and the
# test asserts on persistence rather than on redaction.
ANSWER_SENTENCE = "Meals on a local trip are paid up to sixty euro a day [1]."
ANSWER_FACT = "sixty euro a day"
CHUNK_A_TEXT = (
    "Travel policy, 2025 edition. Meals on a local trip are paid up to sixty "
    "euro a day. Keep your itemised bills."
)
CHUNK_B_TEXT = (
    "Globex travel rules. Meals on a local trip are paid up to thirty euro a "
    "day. CANARY-GLOBEX-TRAVEL-1E44."
)


# ------------------------------------------------------------------ principals


def make_principal(
    *,
    tenant_id: str,
    user_id: str,
    roles: Sequence[str] = ("rag.user",),
    groups: Sequence[str] = (),
    clearance: Classification = Classification.INTERNAL,
) -> Principal:
    """Build a test principal.

    Args:
        tenant_id: Owning tenant.
        user_id: Entra object id.
        roles: App roles.
        groups: Entra group object ids.
        clearance: Clearance ceiling.

    Returns:
        The principal.
    """
    return Principal(
        user_id=user_id,
        tenant_id=tenant_id,
        roles=list(roles),
        groups=list(groups),
        email=f"{user_id}@example.test",
        display_name=user_id,
        max_classification=clearance,
    )


@pytest.fixture
def acme_user() -> Principal:
    """An ordinary user in tenant A."""
    return make_principal(
        tenant_id=TENANT_A, user_id="user-acme-eng", groups=["g-acme-engineering"]
    )


@pytest.fixture
def acme_admin() -> Principal:
    """An administrator in tenant A."""
    return make_principal(
        tenant_id=TENANT_A,
        user_id="user-acme-admin",
        roles=["rag.admin", "rag.user"],
        groups=["g-acme-engineering", "g-acme-hr"],
        clearance=Classification.RESTRICTED,
    )


@pytest.fixture
def globex_user() -> Principal:
    """An ordinary user in tenant B."""
    return make_principal(
        tenant_id=TENANT_B, user_id="user-globex", groups=["g-globex-operations"]
    )


def auth_headers(principal: Principal) -> dict[str, str]:
    """Build the dev-mode authentication header for a principal.

    Args:
        principal: The caller to impersonate.

    Returns:
        Headers carrying the JSON principal, matching what
        ``scripts/smoke_test.py`` sends.
    """
    return {get_settings().entra_dev_principal_header: principal.model_dump_json()}


# -------------------------------------------------------------------- chunking


def make_chunk(
    *,
    tenant_id: str,
    document_id: str,
    text: str,
    score: float = 0.92,
    classification: Classification = Classification.PUBLIC,
    allowed_groups: Sequence[str] = (),
    title: str = "Travel policy",
) -> RetrievedChunk:
    """Build a retrieved chunk for the fake retriever.

    Args:
        tenant_id: Owning tenant.
        document_id: Owning document.
        text: Chunk text, which citation verification runs against.
        score: Final relevance score, on the ``[0, 1]`` scale the OOD gate uses.
        classification: Sensitivity label.
        allowed_groups: Groups permitted to read it.
        title: Document title.

    Returns:
        The chunk.
    """
    access = AccessControl(
        tenant_id=tenant_id,
        allowed_groups=list(allowed_groups),
        classification=classification,
    )
    payload = ChunkPayload.from_access_control(
        access,
        chunk_id=f"{document_id}::0000",
        document_id=document_id,
        chunk_index=0,
        source_type="blob",
        source_id="seed",
        source_uri=f"https://example.test/{document_id}",
        title=title,
        text=text,
        doc_type="policy",
        language="en",
    )
    return RetrievedChunk(
        payload=payload,
        dense_score=None,
        sparse_score=None,
        fusion_score=score,
        rerank_score=score,
        final_score=score,
        retrieval_stage="rerank",
    )


@dataclass
class RetrievalSpy:
    """Records what the pipeline asked the retriever for.

    Attributes:
        chunks: Every chunk the fake corpus holds, across tenants.
        calls: One ``(principal, queries)`` entry per invocation.
    """

    chunks: list[RetrievedChunk]
    calls: list[tuple[Principal, list[str]]]

    async def __call__(
        self,
        principal: Principal,
        queries: Sequence[str],
        filters: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Stand in for :func:`app.rag.retriever.retrieve`.

        The tenant predicate is applied here rather than being assumed, so a
        handler that failed to pass the principal down would return nothing and
        the test would fail loudly instead of passing on a mock's goodwill.

        Args:
            principal: The caller the handler passed down.
            queries: Probes issued.
            filters: Metadata filter.
            **kwargs: Ignored retriever options.

        Returns:
            The chunks visible to this principal.
        """
        del filters, kwargs
        self.calls.append((principal, list(queries)))
        visible = [
            chunk
            for chunk in self.chunks
            if chunk.payload.access_control().permits(principal)
        ]
        return RetrievalResult(
            chunks=visible,
            queries_used=list(queries),
            filter_applied={"tenant_id": principal.tenant_id},
            total_candidates=len(self.chunks),
            after_dedupe=len(visible),
            after_rerank=len(visible),
            latency_ms={"total": 1.0},
        )


# ------------------------------------------------------------------- fake LLM


class FakeLLM:
    """Deterministic stand-in for :class:`~ragcore.llm.LLMClient`.

    Streams a scripted answer whose one factual sentence quotes the fake chunk
    verbatim, so stage 11 verifies the citation for real rather than being
    bypassed.
    """

    def __init__(self, settings: Settings, *, answer: str = ANSWER_SENTENCE) -> None:
        """Initialise the fake.

        Args:
            settings: Active settings, exposed like the real client's.
            answer: Text to stream.
        """
        self._settings = settings
        self.answer = answer
        self.stream_calls: list[dict[str, Any]] = []

    @property
    def settings(self) -> Settings:
        """Settings this client was built from."""
        return self._settings

    async def count_tokens(
        self,
        *,
        system: Any = None,
        messages: Sequence[Any],
        model: str | None = None,
        tools: Sequence[Any] | None = None,
    ) -> int:
        """Estimate a token count without a network call.

        Args:
            system: System blocks.
            messages: Conversation turns.
            model: Ignored.
            tools: Tool definitions.

        Returns:
            A characters/4 estimate, which is only ever compared against a budget
            in these tests.
        """
        del model
        size = len(str(system or "")) + len(str(messages)) + len(str(tools or ""))
        return max(1, size // 4)

    def _usage(self) -> LLMUsage:
        """Build the usage this fake reports.

        Returns:
            A small, non-zero usage so the ``usage`` event is meaningful.
        """
        return LLMUsage(
            input_tokens=120,
            output_tokens=len(self.answer) // 4 + 1,
            cache_read_tokens=64,
            cache_write_tokens=0,
            model=self._settings.anthropic_model_main,
            settings=self._settings,
        )

    async def stream(self, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        """Stream the scripted answer.

        Args:
            **kwargs: The request the orchestrator built; recorded for assertions.

        Yields:
            ``TEXT`` deltas, then ``USAGE``, then one ``DONE``.
        """
        self.stream_calls.append(kwargs)
        for word in self.answer.split(" "):
            yield StreamEvent(type=StreamEventType.TEXT, text=f"{word} ")
        usage = self._usage()
        yield StreamEvent(type=StreamEventType.USAGE, usage=usage)
        yield StreamEvent(
            type=StreamEventType.DONE,
            usage=usage,
            stop_reason="end_turn",
            response=LLMResponse(
                text=self.answer,
                tool_calls=[],
                stop_reason="end_turn",
                usage=usage,
                refused=False,
                raw=None,
            ),
        )

    async def complete(self, **kwargs: Any) -> LLMResponse:
        """Return the scripted answer without streaming.

        Args:
            **kwargs: Ignored.

        Returns:
            The response.
        """
        del kwargs
        usage = self._usage()
        return LLMResponse(
            text=self.answer,
            tool_calls=[],
            stop_reason="end_turn",
            usage=usage,
            refused=False,
            raw=None,
        )

    async def structured(self, *, schema: type, **kwargs: Any) -> Any:
        """Refuse structured calls.

        The pipeline treats every structured call as best-effort, so raising here
        proves the degradation paths are wired rather than merely present.

        Args:
            schema: The requested model.
            **kwargs: Ignored.

        Raises:
            RuntimeError: Always.
        """
        del schema, kwargs
        msg = "structured output is not available in tests"
        raise RuntimeError(msg)

    async def classify(self, *, labels: Sequence[str], **kwargs: Any) -> str:
        """Return the safest label.

        Args:
            labels: Candidate labels, safest first.
            **kwargs: Ignored.

        Returns:
            ``labels[0]``.
        """
        del kwargs
        return labels[0]

    async def aclose(self) -> None:
        """No transport to release."""


# ---------------------------------------------------------------- application


@pytest.fixture
def api_settings() -> Settings:
    """Settings for the API under test.

    Returns:
        The process settings with the external subsystems this suite does not
        stand up switched off, so nothing reaches Qdrant, Redis or Anthropic by
        accident.
    """
    get_settings.cache_clear()
    base = get_settings()
    return base.model_copy(
        update={
            "memory_enabled": False,
            "memory_cache_enabled": False,
            "tool_enabled": False,
            "rerank_enabled": False,
            "langfuse_enabled": False,
            "api_metrics_enabled": True,
            # Read through `tool_config`'s Settings-first lookup. Self-hosted MCP
            # discovery dials a child process or an HTTP endpoint, and a test run
            # must not reach for either. Set here rather than in the environment,
            # which the sibling MCP unit suite reads.
            "tool_mcp_local_enabled": False,
        }
    )


@pytest.fixture
def fake_llm(api_settings: Settings) -> FakeLLM:
    """The scripted model client injected into the pipeline."""
    return FakeLLM(api_settings)


@pytest.fixture
def corpus() -> list[RetrievedChunk]:
    """A two-tenant fake corpus, one document each."""
    return [
        make_chunk(
            tenant_id=TENANT_A,
            document_id="doc-acme-travel-2025",
            text=CHUNK_A_TEXT,
        ),
        make_chunk(
            tenant_id=TENANT_B,
            document_id="doc-globex-travel-policy",
            text=CHUNK_B_TEXT,
            title="Globex travel policy",
        ),
    ]


@pytest.fixture
def retrieval(
    monkeypatch: pytest.MonkeyPatch, corpus: list[RetrievedChunk]
) -> RetrievalSpy:
    """Replace the Qdrant-backed retriever with a tenant-aware fake.

    Args:
        monkeypatch: pytest's patcher.
        corpus: The fake corpus.

    Returns:
        The spy, so a test can assert which principal reached retrieval.
    """
    spy = RetrievalSpy(chunks=list(corpus), calls=[])
    monkeypatch.setattr("app.rag.orchestrator.retrieve", spy)
    monkeypatch.setattr("app.routers.search.retrieve", spy)
    return spy


@pytest.fixture
def orchestrator(api_settings: Settings, fake_llm: FakeLLM) -> Orchestrator:
    """A pipeline wired to the fakes.

    Args:
        api_settings: Settings for the run.
        fake_llm: The scripted model client.

    Returns:
        An orchestrator whose only fakes are the two external services.
    """
    counter = TokenCounter(fake_llm, settings=api_settings)
    short_term = ShortTermMemory(
        settings=api_settings,
        store=InMemorySessionStore(),
        counter=counter,
        llm=fake_llm,
    )
    return Orchestrator(
        settings=api_settings,
        llm=fake_llm,
        tracer=NoopTracer(),
        short_term=short_term,
        assembler=ContextAssembler(
            settings=api_settings,
            llm=fake_llm,
            counter=counter,
            short_term=short_term,
        ),
        long_term=LongTermMemoryStore(settings=api_settings, llm=fake_llm),
        cache=SemanticCache(settings=api_settings),
    )


@pytest.fixture(autouse=True)
def _reset_caches() -> Iterator[None]:
    """Drop cached singletons around every test so fixtures cannot leak."""
    yield
    get_settings.cache_clear()


@pytest.fixture
async def database(api_settings: Settings) -> AsyncIterator[None]:
    """Create the schema and clear it between tests.

    Args:
        api_settings: Settings naming the temporary sqlite file.

    Yields:
        None, with an empty schema in place.
    """
    engine = get_engine(api_settings)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    # `users` and `documents` carry a foreign key to `tenants`, and sqlite is
    # configured with foreign-key enforcement on, so the fixture provisions the
    # two demo tenants the same way the first authenticated request would.
    factory = get_sessionmaker(api_settings)
    async with factory() as session:
        for tenant_id in (TENANT_A, TENANT_B):
            await repo.upsert_tenant(session, tenant_id=tenant_id, name=tenant_id)
        await session.commit()

    yield

    async with engine.begin() as connection:
        for table in reversed(Base.metadata.sorted_tables):
            await connection.execute(table.delete())


@pytest.fixture
async def api(
    api_settings: Settings,
    database: None,
    orchestrator: Orchestrator,
    retrieval: RetrievalSpy,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[AsyncClient]:
    """An HTTP client bound to the application, with the fakes installed.

    Args:
        api_settings: Settings for the run.
        database: Ensures the schema exists.
        orchestrator: The pipeline the chat route will use.
        retrieval: Ensures the retriever is faked before the app runs.
        monkeypatch: pytest's patcher.

    Yields:
        An :class:`httpx.AsyncClient` speaking to the app in-process.
    """
    del database, retrieval
    # Every route that reaches the model or the session window must reach the
    # *same* fakes, or a test would exercise a half-real pipeline.
    monkeypatch.setattr(
        "app.routers.chat.get_orchestrator", lambda _settings=None: orchestrator
    )
    monkeypatch.setattr(
        "app.routers.sessions.get_short_term_memory",
        lambda _settings=None: orchestrator.short_term,
    )
    monkeypatch.setattr(
        "app.routers.sessions.get_context_assembler",
        lambda _settings=None: orchestrator.assembler,
    )
    application = create_app(api_settings)
    application.dependency_overrides[settings_dependency] = lambda: api_settings

    transport = ASGITransport(app=application)
    async with AsyncClient(
        transport=transport, base_url="http://api.test", timeout=30.0
    ) as client:
        client.headers["accept"] = "application/json"
        yield client
    await reset_rate_limiter()


@pytest.fixture
async def db_session(
    api_settings: Settings, database: None
) -> AsyncIterator[AsyncSession]:
    """A session on the temporary database, for seeding rows directly.

    Args:
        api_settings: Settings naming the temporary sqlite file.
        database: Ensures the schema exists.

    Yields:
        An :class:`~sqlalchemy.ext.asyncio.AsyncSession`; the test commits.
    """
    del database
    factory = get_sessionmaker(api_settings)
    session = factory()
    try:
        yield session
    finally:
        await session.close()


# ---------------------------------------------------------------- SSE parsing


def parse_sse(body: str) -> list[tuple[str, str]]:
    """Parse an SSE response body into ``(event, data)`` pairs.

    Comment lines (heartbeats) and the ``retry:`` hint are skipped, so a test
    asserts on the semantic event sequence rather than on transport noise.

    Args:
        body: The raw response text.

    Returns:
        The events in arrival order.
    """
    events: list[tuple[str, str]] = []
    name = ""
    data: list[str] = []
    for raw in body.split("\n"):
        line = raw.rstrip("\r")
        if not line:
            if name:
                events.append((name, "\n".join(data)))
            name, data = "", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            name = value
        elif field == "data":
            data.append(value)
    if name:
        events.append((name, "\n".join(data)))
    return events


def event_names(events: Sequence[tuple[str, str]]) -> list[str]:
    """Project parsed SSE events onto their names.

    Args:
        events: Parsed events.

    Returns:
        The names, in order.
    """
    return [name for name, _ in events]


def payload_for(events: Sequence[tuple[str, str]], name: str) -> Mapping[str, Any]:
    """Return the first payload for an event name.

    Args:
        events: Parsed events.
        name: Event name.

    Returns:
        The decoded JSON payload.

    Raises:
        AssertionError: When the event never arrived.
    """
    for event, data in events:
        if event == name:
            return json.loads(data)
    msg = f"no {name!r} event in {[e for e, _ in events]}"
    raise AssertionError(msg)
