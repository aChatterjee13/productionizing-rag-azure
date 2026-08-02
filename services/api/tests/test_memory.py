"""Short-term and long-term memory.

The load-bearing assertion here is the consent gate: with ``memory_consent=False``
nothing is read and nothing is written. The Qdrant client, the embedder and the model
used in those tests all raise on contact, so "nothing happened" is proved rather than
inferred from an empty result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import models as qm

from app.rag.memory.consolidate import MemoryConsolidator, ProfileDraft
from app.rag.memory.long_term import (
    ExtractedMemory,
    LongTermMemoryStore,
    MemoryExtraction,
)
from app.rag.memory.short_term import (
    InMemorySessionStore,
    SessionWindow,
    ShortTermMemory,
    TokenCounter,
    summarise_turns,
)
from ragcore.embeddings import cosine_similarity
from ragcore.models.acl import Principal
from ragcore.models.chat import Message, Role
from ragcore.models.memory import LongTermMemory, MemoryKind, UserProfile
from ragcore.settings import Settings
from ragcore.vectorstore import DENSE, point_id_for_memory

TENANT = "tenant-acme"
OTHER_TENANT = "tenant-globex"
USER = "user-1"

VOCAB = [
    "munich",
    "office",
    "bullet",
    "points",
    "travel",
    "policy",
    "german",
    "expenses",
    "vpn",
    "salary",
    "concur",
    "approval",
    "director",
]


# ---------------------------------------------------------------------- fakes
def _vector(text: str) -> list[float]:
    words = {token.strip(".,;:").lower() for token in text.split()}
    vector = [1.0 if token in words else 0.0 for token in VOCAB]
    if not any(vector):
        vector = [1e-6] * len(VOCAB)
    return vector


class FakeEmbedder:
    """Bag-of-words embedder: deterministic, and cosine still means something."""

    dim = len(VOCAB)

    async def embed_query(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(dense=_vector(text))

    async def embed_documents(self, texts: list[str]) -> list[SimpleNamespace]:
        return [SimpleNamespace(dense=_vector(text)) for text in texts]


@dataclass
class _FakeResponse:
    text: str
    refused: bool = False


class FakeLLM:
    """Counts tokens by words and returns a scripted structured result."""

    def __init__(self, structured_result: Any = None, summary: str = "summary") -> None:
        """Initialise the fake with the structured result it will return."""
        self.structured_result = structured_result
        self.summary = summary
        self.structured_calls = 0
        self.complete_calls = 0

    async def count_tokens(self, *, system: Any = None, messages: Any, **_: Any) -> int:
        total = 5
        for message in messages:
            content = message.get("content", "")
            total += 4 + len(str(content).split())
        return total

    async def complete(self, **_: Any) -> _FakeResponse:
        self.complete_calls += 1
        return _FakeResponse(text=self.summary)

    async def structured(self, **_: Any) -> Any:
        self.structured_calls += 1
        if self.structured_result is None:
            msg = "no structured result scripted"
            raise AssertionError(msg)
        return self.structured_result


class ExplodingLLM:
    """Any contact is a test failure."""

    def __getattr__(self, name: str) -> Any:
        """Return a coroutine that fails the test when awaited."""

        async def _boom(*_: Any, **__: Any) -> Any:
            msg = f"llm.{name} must not be called"
            raise AssertionError(msg)

        return _boom


class ExplodingQdrant:
    """Any contact is a test failure."""

    def __getattr__(self, name: str) -> Any:
        """Return a coroutine that fails the test when awaited."""

        async def _boom(*_: Any, **__: Any) -> Any:
            msg = f"qdrant.{name} must not be called"
            raise AssertionError(msg)

        return _boom


def _must_values(qfilter: qm.Filter) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for condition in qfilter.must or []:
        key = getattr(condition, "key", None)
        match = getattr(condition, "match", None)
        if key and match is not None and hasattr(match, "value"):
            values[key] = match.value
    return values


class FakeQdrant:
    """In-memory stand-in for the subset of the Qdrant API these paths use."""

    def __init__(self) -> None:
        """Start with an empty collection."""
        self.points: dict[str, dict[str, Any]] = {}
        self.deleted: list[str] = []

    def seed(self, memory: LongTermMemory) -> str:
        point_id = point_id_for_memory(memory.memory_id)
        self.points[point_id] = {
            "payload": memory.to_qdrant_payload(),
            "vector": {DENSE: _vector(memory.text)},
        }
        return point_id

    def _matching(self, qfilter: qm.Filter) -> list[tuple[str, dict[str, Any]]]:
        wanted = _must_values(qfilter)
        matched: list[tuple[str, dict[str, Any]]] = []
        for point_id, record in self.points.items():
            payload = record["payload"]
            if all(payload.get(key) == value for key, value in wanted.items()):
                matched.append((point_id, record))
        return matched

    async def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        using: str,
        query_filter: qm.Filter,
        limit: int,
        score_threshold: float | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> SimpleNamespace:
        scored = []
        for point_id, record in self._matching(query_filter):
            score = cosine_similarity(query, record["vector"][DENSE])
            if score_threshold is not None and score < score_threshold:
                continue
            scored.append(
                SimpleNamespace(id=point_id, score=score, payload=record["payload"])
            )
        scored.sort(key=lambda point: point.score, reverse=True)
        return SimpleNamespace(points=scored[:limit])

    async def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter: qm.Filter,
        limit: int,
        with_payload: bool = True,
        with_vectors: bool = False,
        **_: Any,
    ) -> tuple[list[SimpleNamespace], None]:
        records = [
            SimpleNamespace(
                id=point_id,
                payload=record["payload"],
                vector=record["vector"] if with_vectors else None,
            )
            for point_id, record in self._matching(scroll_filter)
        ]
        return records[:limit], None

    async def retrieve(
        self,
        *,
        collection_name: str,
        ids: list[str],
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(id=point_id, payload=self.points[point_id]["payload"])
            for point_id in ids
            if point_id in self.points
        ]

    async def upsert(
        self, *, collection_name: str, points: list[Any], wait: bool = True
    ) -> None:
        for point in points:
            self.points[str(point.id)] = {
                "payload": dict(point.payload),
                "vector": dict(point.vector),
            }

    async def set_payload(
        self,
        *,
        collection_name: str,
        payload: dict[str, Any],
        points: list[str],
        wait: bool = True,
    ) -> None:
        for point_id in points:
            if point_id in self.points:
                self.points[point_id]["payload"].update(payload)

    async def delete(
        self, *, collection_name: str, points_selector: Any, wait: bool = True
    ) -> None:
        for point_id in getattr(points_selector, "points", []):
            self.points.pop(str(point_id), None)
            self.deleted.append(str(point_id))


# --------------------------------------------------------------------- helpers
def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "env": "local",
        "redis_enabled": False,
        "langfuse_enabled": False,
        "memory_min_salience": 0.25,
        "memory_top_k": 3,
        "memory_ttl_days": 30,
        "memory_max_per_user": 50,
        "memory_dedupe_threshold": 0.9,
    }
    base.update(overrides)
    return Settings(**base)


def make_principal(tenant: str = TENANT, user: str = USER) -> Principal:
    return Principal(user_id=user, tenant_id=tenant, roles=["rag.user"])


def make_memory(
    memory_id: str,
    text: str,
    *,
    salience: float = 0.8,
    kind: MemoryKind = MemoryKind.FACT,
    tenant: str = TENANT,
    user: str = USER,
    last_used_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> LongTermMemory:
    return LongTermMemory(
        memory_id=memory_id,
        user_id=user,
        tenant_id=tenant,
        kind=kind,
        text=text,
        salience=salience,
        last_used_at=last_used_at or datetime.now(UTC),
        expires_at=expires_at,
        pii_redacted=True,
    )


def build_store(
    settings: Settings,
    *,
    client: Any = None,
    llm: Any = None,
) -> LongTermMemoryStore:
    return LongTermMemoryStore(
        settings=settings,
        client=client if client is not None else FakeQdrant(),
        embedder=FakeEmbedder(),
        llm=llm if llm is not None else FakeLLM(),
    )


# ------------------------------------------------------------- short-term memory
async def test_record_turn_refuses_unredacted_content() -> None:
    settings = make_settings()
    short_term = ShortTermMemory(
        settings=settings,
        store=InMemorySessionStore(),
        counter=TokenCounter(FakeLLM(), settings=settings),
        llm=FakeLLM(),
    )
    window = SessionWindow.empty(tenant_id=TENANT, user_id=USER, session_id="s1")

    with pytest.raises(ValueError, match="PII redaction"):
        await short_term.record_turn(
            window,
            role=Role.USER,
            content="my card is 4111 1111 1111 1111",
            pii_redacted=False,
            message_id="m0",
        )
    assert window.turns == []


async def test_window_round_trips_through_the_store_and_is_tenant_scoped() -> None:
    settings = make_settings()
    store = InMemorySessionStore()
    short_term = ShortTermMemory(
        settings=settings,
        store=store,
        counter=TokenCounter(FakeLLM(), settings=settings),
        llm=FakeLLM(),
    )
    principal = make_principal()
    window = SessionWindow.empty(tenant_id=TENANT, user_id=USER, session_id="s1")
    await short_term.record_turn(
        window,
        role=Role.USER,
        content="where do I file expenses",
        pii_redacted=True,
        message_id="m0",
    )

    reloaded = await short_term.load(principal=principal, session_id="s1")
    assert [turn.message_id for turn in reloaded.turns] == ["m0"]
    assert reloaded.turns[0].token_count > 0

    # A principal from another tenant cannot read this window: the store key is
    # tenant-scoped, so the lookup simply misses.
    other = await short_term.load(
        principal=make_principal(tenant=OTHER_TENANT), session_id="s1"
    )
    assert other.turns == []
    assert other.tenant_id == OTHER_TENANT


async def test_pinned_turns_survive_every_suppression_path() -> None:
    window = SessionWindow.empty(tenant_id=TENANT, user_id=USER, session_id="s1")
    for index in range(6):
        window.append(
            Message(
                message_id=f"m{index}",
                session_id="s1",
                role=Role.USER if index % 2 == 0 else Role.ASSISTANT,
                content=f"turn {index}",
                token_count=5,
                pinned=index == 0,
            )
        )

    assert [turn.message_id for turn in window.suppressible(keep_live=2)] == [
        "m1",
        "m2",
        "m3",
    ]
    retired = window.suppress(
        [f"m{index}" for index in range(6)], summary="s", summary_tokens=1
    )
    assert "m0" not in [turn.message_id for turn in retired]
    assert [turn.message_id for turn in window.turns] == ["m0"]
    assert window.suppressed_count == 5
    assert window.compaction_events == 1
    assert window.turns_since_compaction == 0


async def test_summarise_turns_degrades_to_a_deterministic_summary() -> None:
    settings = make_settings()

    class Broken:
        async def complete(self, **_: Any) -> Any:
            msg = "upstream is down"
            raise RuntimeError(msg)

    turns = [
        Message(
            message_id="m0",
            session_id="s1",
            role=Role.USER,
            content="what is the meal allowance",
        )
    ]
    summary = await summarise_turns(turns, llm=Broken(), settings=settings)

    assert "meal allowance" in summary
    assert summary.startswith("user:")


# -------------------------------------------------------------- the consent gate
async def test_consent_false_reads_nothing() -> None:
    settings = make_settings()
    store = LongTermMemoryStore(
        settings=settings,
        client=ExplodingQdrant(),
        embedder=FakeEmbedder(),
        llm=ExplodingLLM(),
    )
    profile = UserProfile(user_id=USER, tenant_id=TENANT, memory_consent=False)

    recall = await store.recall(make_principal(), "what do you know", profile=profile)

    assert recall.memories == []
    assert recall.consent is False
    assert recall.reason == "no_consent"


async def test_consent_false_writes_nothing() -> None:
    settings = make_settings()
    client = FakeQdrant()
    llm = ExplodingLLM()
    store = LongTermMemoryStore(
        settings=settings, client=client, embedder=FakeEmbedder(), llm=llm
    )
    profile = UserProfile(user_id=USER, tenant_id=TENANT, memory_consent=False)

    stored = await store.write_back(
        principal=make_principal(),
        session_id="s1",
        user_text="I work in the Munich office and prefer bullet points",
        assistant_text="Noted.",
        profile=profile,
    )

    assert stored == []
    assert client.points == {}


async def test_unknown_consent_is_treated_as_no_consent() -> None:
    settings = make_settings()
    store = LongTermMemoryStore(
        settings=settings,
        client=ExplodingQdrant(),
        embedder=FakeEmbedder(),
        llm=ExplodingLLM(),
    )

    recall = await store.recall(make_principal(), "anything")
    written = await store.write_back(
        principal=make_principal(), session_id="s1", user_text="I like bullets"
    )

    assert recall.reason == "consent_unknown"
    assert recall.consent is False
    assert written == []


async def test_memory_disabled_short_circuits() -> None:
    settings = make_settings(memory_enabled=False)
    store = LongTermMemoryStore(
        settings=settings,
        client=ExplodingQdrant(),
        embedder=FakeEmbedder(),
        llm=ExplodingLLM(),
    )
    profile = UserProfile(user_id=USER, tenant_id=TENANT, memory_consent=True)

    recall = await store.recall(make_principal(), "anything", profile=profile)
    assert recall.reason == "disabled"


# ------------------------------------------------------------------- recall path
async def test_recall_is_salience_weighted_and_skips_faded_or_expired() -> None:
    settings = make_settings(memory_top_k=2, memory_min_salience=0.3)
    client = FakeQdrant()
    now = datetime.now(UTC)
    client.seed(make_memory("m-strong", "Works in the Munich office", salience=0.9))
    client.seed(make_memory("m-weak", "Munich office coffee machine", salience=0.05))
    client.seed(
        make_memory(
            "m-expired",
            "Munich office desk booking",
            salience=0.9,
            expires_at=now - timedelta(days=1),
        )
    )
    client.seed(make_memory("m-other", "Prefers bullet points", salience=0.8))
    store = build_store(settings, client=client)
    profile = UserProfile(user_id=USER, tenant_id=TENANT)

    recall = await store.recall(
        make_principal(), "munich office", profile=profile, now=now
    )

    ids = [memory.memory_id for memory in recall.memories]
    assert ids[0] == "m-strong"
    assert "m-weak" not in ids
    assert "m-expired" not in ids
    assert len(ids) <= 2
    assert recall.reason == "ok"
    # Usage bookkeeping was written back.
    payload = client.points[point_id_for_memory("m-strong")]["payload"]
    assert payload["hit_count"] == 1
    assert payload["last_used_at"] is not None


async def test_recall_never_crosses_the_tenant_boundary() -> None:
    settings = make_settings()
    client = FakeQdrant()
    client.seed(make_memory("mine", "Munich office", tenant=TENANT))
    client.seed(make_memory("theirs", "Munich office", tenant=OTHER_TENANT))
    store = build_store(settings, client=client)

    recall = await store.recall(
        make_principal(),
        "munich office",
        profile=UserProfile(user_id=USER, tenant_id=TENANT),
    )

    assert [memory.memory_id for memory in recall.memories] == ["mine"]


# --------------------------------------------------------------- write-back path
async def test_write_back_redacts_before_storing() -> None:
    settings = make_settings()
    client = FakeQdrant()
    llm = FakeLLM(
        structured_result=MemoryExtraction(
            memories=[
                ExtractedMemory(
                    kind="fact",
                    text="Reachable at bob@example.com for travel questions",
                    salience=0.7,
                )
            ]
        )
    )
    store = build_store(settings, client=client, llm=llm)

    stored = await store.write_back(
        principal=make_principal(),
        session_id="s1",
        user_text="travel policy question",
        assistant_text="answered",
        profile=UserProfile(user_id=USER, tenant_id=TENANT),
    )

    assert len(stored) == 1
    memory = stored[0]
    assert memory.pii_redacted is True
    assert "bob@example.com" not in memory.text
    assert "<EMAIL_ADDRESS>" in memory.text
    assert memory.expires_at is not None
    # And the same redacted text is what reached the vector store.
    payload = client.points[point_id_for_memory(memory.memory_id)]["payload"]
    assert "bob@example.com" not in payload["text"]


async def test_write_back_supersedes_a_near_duplicate_instead_of_appending() -> None:
    settings = make_settings(memory_dedupe_threshold=0.8)
    client = FakeQdrant()
    old = make_memory("old-1", "Works in the Munich office", salience=0.6)
    client.seed(old)
    llm = FakeLLM(
        structured_result=MemoryExtraction(
            memories=[
                ExtractedMemory(
                    kind="fact", text="Works in the Munich office now", salience=0.9
                )
            ]
        )
    )
    store = build_store(settings, client=client, llm=llm)

    stored = await store.write_back(
        principal=make_principal(),
        session_id="s1",
        user_text="I moved to the Munich office",
        profile=UserProfile(user_id=USER, tenant_id=TENANT),
    )

    assert len(stored) == 1
    assert stored[0].supersedes == "old-1"
    # Superseding replaces rather than appends: the old point is gone.
    assert point_id_for_memory("old-1") not in client.points
    assert point_id_for_memory(stored[0].memory_id) in client.points


async def test_write_back_skips_an_exact_repeat() -> None:
    settings = make_settings(memory_dedupe_threshold=0.8)
    client = FakeQdrant()
    client.seed(make_memory("old-1", "Works in the Munich office"))
    llm = FakeLLM(
        structured_result=MemoryExtraction(
            memories=[ExtractedMemory(kind="fact", text="Works in the Munich office")]
        )
    )
    store = build_store(settings, client=client, llm=llm)

    stored = await store.write_back(
        principal=make_principal(),
        session_id="s1",
        user_text="reminder about Munich",
        profile=UserProfile(user_id=USER, tenant_id=TENANT),
    )

    assert stored == []
    assert len(client.points) == 1


async def test_remember_refuses_a_foreign_tenant() -> None:
    settings = make_settings()
    store = build_store(settings)

    with pytest.raises(ValueError, match="different tenant"):
        await store.remember(
            make_memory("x", "Munich office", tenant=OTHER_TENANT),
            principal=make_principal(),
        )


async def test_forget_will_not_delete_another_tenants_point() -> None:
    settings = make_settings()
    client = FakeQdrant()
    client.seed(make_memory("theirs", "Munich office", tenant=OTHER_TENANT, user="u9"))
    store = build_store(settings, client=client)

    removed = await store.forget(make_principal(), "theirs")

    assert removed is False
    assert point_id_for_memory("theirs") in client.points


async def test_expire_and_prune_bound_growth() -> None:
    settings = make_settings(memory_max_per_user=2, memory_salience_decay_per_day=0.0)
    client = FakeQdrant()
    now = datetime.now(UTC)
    client.seed(
        make_memory("gone", "Munich office", expires_at=now - timedelta(days=1))
    )
    for index, salience in enumerate((0.2, 0.5, 0.9)):
        client.seed(make_memory(f"keep-{index}", f"policy {index}", salience=salience))
    store = build_store(settings, client=client)
    principal = make_principal()

    expired = await store.expire(principal, now=now)
    pruned = await store.prune(principal, now=now)

    assert expired == 1
    assert pruned == 1
    remaining = {record["payload"]["memory_id"] for record in client.points.values()}
    assert remaining == {"keep-1", "keep-2"}


# ------------------------------------------------------------------ consolidation
async def test_consolidation_merges_near_duplicates_and_decays() -> None:
    settings = make_settings(
        memory_dedupe_threshold=0.8, memory_salience_decay_per_day=0.1
    )
    client = FakeQdrant()
    now = datetime.now(UTC)
    stale = now - timedelta(days=3)
    client.seed(
        make_memory("dup-a", "Prefers bullet points", salience=0.9, last_used_at=stale)
    )
    client.seed(
        make_memory("dup-b", "Prefers bullet points", salience=0.4, last_used_at=stale)
    )
    client.seed(
        make_memory(
            "solo", "Works in the Munich office", salience=0.7, last_used_at=stale
        )
    )
    llm = FakeLLM(structured_result=ProfileDraft(summary="An engineer in Munich."))
    store = build_store(settings, client=client, llm=llm)
    consolidator = MemoryConsolidator(settings=settings, store=store, llm=llm)

    report = await consolidator.consolidate_user(
        tenant_id=TENANT,
        user_id=USER,
        profile=UserProfile(user_id=USER, tenant_id=TENANT),
        now=now,
    )

    assert report.merged == 1
    assert report.decayed >= 1
    assert report.changed
    surviving = {record["payload"]["memory_id"] for record in client.points.values()}
    assert "dup-b" not in surviving
    # No database session was supplied, so the profile is computed but not persisted.
    assert report.profile_refreshed is False


async def test_consolidation_respects_consent() -> None:
    settings = make_settings()
    store = LongTermMemoryStore(
        settings=settings,
        client=ExplodingQdrant(),
        embedder=FakeEmbedder(),
        llm=ExplodingLLM(),
    )
    consolidator = MemoryConsolidator(
        settings=settings, store=store, llm=ExplodingLLM()
    )

    report = await consolidator.consolidate_user(
        tenant_id=TENANT,
        user_id=USER,
        profile=UserProfile(user_id=USER, tenant_id=TENANT, memory_consent=False),
    )

    assert report.skipped_reason == "no_consent"
    assert report.changed is False
