"""Long-term, per-user memory — pipeline stages 2 and 13, requirement #2.

Durable preferences, facts, entities and episodes live as dense points in
``rag_memories``. Recall is salience-weighted and decayed; write-back is a single
structured ``MODEL_FAST`` call whose output supersedes near-duplicates instead of
piling up a second contradictory copy.

Four invariants:

* **Consent is a hard gate.** When ``UserProfile.memory_consent`` is False nothing is
  read and nothing is written — not the Qdrant probe, not the extraction call, not
  the PostgreSQL row. When consent cannot be determined (no profile and no database
  session) the safe reading is "no consent", so memory stays off rather than
  defaulting on.
* **Tenancy is enforced in the query.** Every read uses
  :func:`ragcore.vectorstore.filters.build_memory_filter`, which puts tenant *and*
  user in ``must``; no filter is hand-rolled here. Deletes resolve the point first
  and verify its tenant and user before removing it.
* **Every memory is PII-scanned before storage.** :meth:`LongTermMemoryStore.remember`
  is the only write path and it redacts first, so a memory can never carry a raw
  identifier into a prompt, a payload or a log line.
* **Growth is bounded.** ``supersedes`` replaces rather than appends, ``expires_at``
  enforces ``memory_ttl_days``, and anything past ``memory_max_per_user`` is pruned
  lowest-decayed-salience first.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import models as qm

from ragcore.embeddings import cosine_similarity, get_embedding_provider
from ragcore.llm import get_llm_client
from ragcore.llm.client import LLMClient, LLMRefusedError
from ragcore.llm.prompts import MEMORY_EXTRACTION_SYSTEM, prompt_metadata
from ragcore.models.acl import Principal
from ragcore.models.memory import LongTermMemory, MemoryKind, UserProfile
from ragcore.observability import observe_cache_lookup
from ragcore.pii import PIIDetector, get_pii_detector
from ragcore.settings import Settings, get_settings
from ragcore.vectorstore import (
    DENSE,
    build_memory_filter,
    dense_search,
    get_client,
    point_id_for_memory,
    upsert_points,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qdrant_client import AsyncQdrantClient
    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "EXTRACTION_CALL_NAME",
    "ExtractedMemory",
    "LongTermMemoryStore",
    "MemoryExtraction",
    "MemoryRecall",
    "get_long_term_memory",
    "reset_long_term_memory",
]

_log = structlog.get_logger(__name__)

#: Trace/operation name for the write-back extraction call.
EXTRACTION_CALL_NAME = "rag.memory_extraction"

#: Shortest text worth storing as a memory. Anything shorter is noise, not a fact.
_MIN_MEMORY_CHARS = 8


def _utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    Returns:
        The current moment in UTC.
    """
    return datetime.now(UTC)


def _coerce_kind(value: str) -> MemoryKind:
    """Map a model-supplied label onto a :class:`MemoryKind`.

    Args:
        value: Raw label from the extraction call.

    Returns:
        The matching kind, defaulting to ``FACT`` for anything unrecognised — a
        mislabelled memory is still useful, an exception is not.
    """
    candidate = (value or "").strip().lower()
    for kind in MemoryKind:
        if candidate == kind.value:
            return kind
    return MemoryKind.FACT


def _parse_memory(payload: Mapping[str, Any] | None) -> LongTermMemory | None:
    """Rebuild a memory from a Qdrant payload.

    Args:
        payload: Raw point payload.

    Returns:
        The parsed memory, or None when the payload is missing or malformed. A point
        that cannot be parsed is skipped rather than failing recall.
    """
    if not payload:
        return None
    known = set(LongTermMemory.model_fields)
    cleaned = {key: value for key, value in payload.items() if key in known}
    try:
        return LongTermMemory.model_validate(cleaned)
    except ValueError:
        _log.warning("memory_payload_unparseable")
        return None


class ExtractedMemory(BaseModel):
    """One candidate memory from the write-back extraction call.

    Deliberately flat and primitive. Structured outputs strip most JSON Schema
    keywords, so an enum or a datetime here is the shape most likely to come back
    unusable; the fields are validated into a :class:`LongTermMemory` afterwards,
    where a bad value can be corrected instead of failing the call.
    """

    model_config = ConfigDict(extra="ignore")

    kind: str = Field(
        default="fact", description="preference | fact | entity | episode."
    )
    text: str = Field(
        default="",
        description="Self-contained third-person sentence, free of identifiers.",
    )
    salience: float = Field(
        default=0.5, description="Importance in 0..1; clamped on the way in."
    )
    supersedes_text: str = Field(
        default="",
        description="Existing memory this replaces, quoted; resolved to an id here.",
    )


class MemoryExtraction(BaseModel):
    """Wire schema for the structured write-back call."""

    model_config = ConfigDict(extra="ignore")

    memories: list[ExtractedMemory] = Field(
        default_factory=list, description="Durable memories worth keeping, possibly []."
    )


class MemoryRecall(BaseModel):
    """What stage 2 loaded, and why."""

    model_config = ConfigDict(extra="forbid")

    memories: list[LongTermMemory] = Field(
        default_factory=list, description="Selected memories, best first."
    )
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="memory_id -> combined similarity/salience score.",
    )
    profile: UserProfile | None = Field(
        default=None, description="The profile consent was read from."
    )
    consent: bool = Field(
        default=False, description="Whether long-term memory was allowed to be read."
    )
    candidates: int = Field(
        default=0, ge=0, description="Points returned by the dense probe."
    )
    latency_ms: float = Field(default=0.0, ge=0.0, description="Recall latency.")
    reason: str = Field(
        default="",
        description=(
            "'ok' | 'disabled' | 'no_consent' | 'consent_unknown' | 'empty_query' | "
            "'error'."
        ),
    )

    @property
    def has_memories(self) -> bool:
        """Whether anything was recalled.

        Returns:
            True when at least one memory was selected.
        """
        return bool(self.memories)


class LongTermMemoryStore:
    """Dense recall, structured write-back and lifecycle for ``rag_memories``."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: AsyncQdrantClient | None = None,
        embedder: Any | None = None,
        llm: LLMClient | None = None,
        detector: PIIDetector | None = None,
    ) -> None:
        """Initialise the store.

        Args:
            settings: Active settings. Defaults to the process settings.
            client: Qdrant client. Resolved lazily when omitted.
            embedder: Embedding provider. Resolved lazily when omitted.
            llm: Client used for the extraction call.
            detector: PII detector used before every write.
        """
        self._settings = settings or get_settings()
        self._client = client
        self._embedder = embedder
        self._llm = llm
        self._detector = detector

    @property
    def settings(self) -> Settings:
        """Settings this store was built from.

        Returns:
            The bound settings.
        """
        return self._settings

    @property
    def collection(self) -> str:
        """Qdrant collection holding memories.

        Returns:
            ``settings.qdrant_memories_collection``.
        """
        return self._settings.qdrant_memories_collection

    async def qdrant(self) -> AsyncQdrantClient:
        """Resolve the Qdrant client.

        Exposed so the consolidation job can reuse the same connection rather than
        opening a second one.

        Returns:
            The shared async client.
        """
        if self._client is None:
            self._client = await get_client(self._settings)
        return self._client

    def _embedding_provider(self) -> Any:
        """Resolve the embedding provider.

        Returns:
            The cached provider singleton.
        """
        if self._embedder is None:
            self._embedder = get_embedding_provider(self._settings)
        return self._embedder

    def _pii(self) -> PIIDetector:
        """Resolve the PII detector.

        Returns:
            The cached detector.
        """
        if self._detector is None:
            self._detector = get_pii_detector(self._settings)
        return self._detector

    def _client_llm(self) -> LLMClient:
        """Resolve the LLM client.

        Returns:
            The bound or process client.
        """
        if self._llm is None:
            self._llm = get_llm_client(self._settings)
        return self._llm

    async def _embed(self, text: str) -> list[float]:
        """Embed one text with the dense model.

        Args:
            text: Text to embed.

        Returns:
            The dense vector.
        """
        embedded = await self._embedding_provider().embed_query(text)
        return list(embedded.dense)

    # -------------------------------------------------------------------- consent
    async def resolve_profile(
        self,
        principal: Principal,
        *,
        profile: UserProfile | None = None,
        db_session: AsyncSession | None = None,
    ) -> UserProfile | None:
        """Resolve the profile that decides consent.

        Args:
            principal: The caller.
            profile: An already-loaded profile, which wins.
            db_session: Database session used to load one when ``profile`` is None.

        Returns:
            The profile, or None when it could not be determined. None means "no
            consent" everywhere in this module.
        """
        if profile is not None:
            return profile
        if db_session is None:
            return None
        from ragcore.db import repositories as repo

        return await repo.get_or_create_profile(
            db_session, tenant_id=principal.tenant_id, user_id=principal.user_id
        )

    def consent_gate(self, profile: UserProfile | None) -> str:
        """Evaluate the consent and feature gates.

        A missing profile is treated as "no consent". Defaulting the other way would
        mean a database hiccup silently turns memory on for someone who switched it
        off, which is exactly the failure a consent gate exists to prevent.

        Args:
            profile: The resolved profile, or None.

        Returns:
            ``"ok"`` when memory may be used, otherwise the blocking reason:
            ``"disabled"``, ``"consent_unknown"`` or ``"no_consent"``.
        """
        if not self._settings.memory_enabled:
            return "disabled"
        if profile is None:
            return "consent_unknown"
        if not profile.memory_consent:
            return "no_consent"
        return "ok"

    # --------------------------------------------------------------------- recall
    async def recall(
        self,
        principal: Principal,
        query: str,
        *,
        profile: UserProfile | None = None,
        top_k: int | None = None,
        kinds: Iterable[MemoryKind | str] | None = None,
        db_session: AsyncSession | None = None,
        now: datetime | None = None,
        touch: bool = True,
    ) -> MemoryRecall:
        """Load the most useful memories for this turn (pipeline stage 2).

        Candidates come from a dense probe scoped to the caller's tenant and user,
        are decayed by time since last use, filtered by ``memory_min_salience`` and
        ranked by a blend of similarity and salience
        (``memory_recall_similarity_weight``).

        Args:
            principal: The caller.
            query: The user's question, used as the dense probe.
            profile: Pre-loaded profile. Required for consent unless ``db_session``
                is given.
            top_k: Memories to return. Defaults to ``settings.memory_top_k``.
            kinds: Restrict to these memory kinds.
            db_session: Database session used to load the profile and to record hits.
            now: Reference time for decay and expiry. Defaults to now (UTC).
            touch: Update ``hit_count`` and ``last_used_at`` on the selected
                memories. Disable for read-only inspection.

        Returns:
            The recall result, always populated with ``consent`` and ``reason`` so a
            caller can tell "nothing stored" from "not allowed to look".
        """
        started = time.perf_counter()
        resolved = await self.resolve_profile(
            principal, profile=profile, db_session=db_session
        )
        gate = self.consent_gate(resolved)
        if gate != "ok":
            return MemoryRecall(profile=resolved, consent=False, reason=gate)

        limit = self._settings.memory_top_k if top_k is None else top_k
        if limit <= 0 or not query.strip():
            return MemoryRecall(
                profile=resolved,
                consent=True,
                reason="empty_query" if not query.strip() else "ok",
            )

        reference = now or _utcnow()
        oversample = int(self._settings.memory_recall_oversample)
        try:
            client = await self.qdrant()
            vector = await self._embed(query)
            points = await dense_search(
                client,
                collection=self.collection,
                dense=vector,
                qfilter=build_memory_filter(principal, kinds, now=reference),
                limit=max(limit, limit * oversample),
            )
        except Exception:
            _log.warning(
                "memory_recall_failed",
                tenant_id=principal.tenant_id,
                exc_info=True,
            )
            observe_cache_lookup(cache="memory", hit=False)
            return MemoryRecall(profile=resolved, consent=True, reason="error")

        weight = float(self._settings.memory_recall_similarity_weight)
        decay = self._settings.memory_salience_decay_per_day
        floor = self._settings.memory_min_salience

        ranked: list[tuple[float, LongTermMemory]] = []
        for point in points:
            memory = _parse_memory(point.payload)
            if memory is None or memory.is_expired(reference):
                continue
            salience = memory.decayed_salience(decay_per_day=decay, now=reference)
            if salience < floor:
                continue
            similarity = float(point.score or 0.0)
            ranked.append((weight * similarity + (1.0 - weight) * salience, memory))

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[:limit]
        memories = [memory for _, memory in selected]
        observe_cache_lookup(cache="memory", hit=bool(memories))

        if touch and memories:
            await self.touch(principal, memories, db_session=db_session, now=reference)

        return MemoryRecall(
            memories=memories,
            scores={memory.memory_id: score for score, memory in selected},
            profile=resolved,
            consent=True,
            candidates=len(points),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            reason="ok",
        )

    async def touch(
        self,
        principal: Principal,
        memories: Sequence[LongTermMemory],
        *,
        db_session: AsyncSession | None = None,
        now: datetime | None = None,
    ) -> int:
        """Record that memories were used: ``hit_count`` and ``last_used_at``.

        Failures are logged and swallowed. Usage bookkeeping is valuable but it is
        not worth failing a chat turn for.

        Args:
            principal: The caller; supplies the tenant and user scope.
            memories: Memories that were injected into the prompt.
            db_session: Optional session for the relational mirror.
            now: Reference time. Defaults to now (UTC).

        Returns:
            The number of memories updated in Qdrant.
        """
        if not memories:
            return 0
        moment = now or _utcnow()
        boost = self._settings.memory_salience_boost_on_hit
        updated = 0
        try:
            client = await self.qdrant()
        except Exception:
            _log.warning("memory_touch_client_failed", exc_info=True)
            return 0

        for memory in memories:
            if memory.tenant_id != principal.tenant_id:
                _log.error(
                    "memory_tenant_mismatch",
                    expected=principal.tenant_id,
                    memory_id=memory.memory_id,
                )
                continue
            touched = memory.touch(boost=boost, now=moment)
            try:
                await client.set_payload(
                    collection_name=self.collection,
                    payload={
                        "hit_count": touched.hit_count,
                        "last_used_at": touched.last_used_at.isoformat()
                        if touched.last_used_at
                        else None,
                        "salience": touched.salience,
                    },
                    points=[point_id_for_memory(memory.memory_id)],
                    wait=False,
                )
            except Exception:
                _log.warning("memory_touch_failed", memory_id=memory.memory_id)
                continue
            updated += 1
            if db_session is not None:
                await self._mirror(touched, db_session=db_session)
        return updated

    async def set_salience(
        self,
        principal: Principal,
        memory: LongTermMemory,
        value: float,
        *,
        db_session: AsyncSession | None = None,
    ) -> bool:
        """Rewrite one memory's stored salience.

        Used by the consolidation job to persist decay, so the stored value stays
        honest for pruning order and for the admin surface. It is a payload update,
        not a re-embed: the text did not change.

        Args:
            principal: The caller; the memory must belong to them.
            memory: The memory to update.
            value: New salience, clamped to 0..1.
            db_session: Optional session for the relational mirror.

        Returns:
            True when the payload was written.
        """
        if (
            memory.tenant_id != principal.tenant_id
            or memory.user_id != principal.user_id
        ):
            _log.error(
                "memory_salience_ownership_denied", tenant_id=principal.tenant_id
            )
            return False
        clamped = min(1.0, max(0.0, value))
        try:
            client = await self.qdrant()
            await client.set_payload(
                collection_name=self.collection,
                payload={"salience": clamped},
                points=[point_id_for_memory(memory.memory_id)],
                wait=False,
            )
        except Exception:
            _log.warning("memory_salience_update_failed", memory_id=memory.memory_id)
            return False
        if db_session is not None:
            await self._mirror(
                memory.model_copy(update={"salience": clamped}),
                db_session=db_session,
            )
        return True

    # ---------------------------------------------------------------- write path
    async def remember(
        self,
        memory: LongTermMemory,
        *,
        principal: Principal,
        db_session: AsyncSession | None = None,
    ) -> LongTermMemory:
        """Store one memory, PII-scanning it first.

        This is the only write path into ``rag_memories``, which is what makes the
        "every memory is redacted before storage" guarantee checkable in one place.

        Args:
            memory: The memory to store.
            principal: The caller; its tenant must match the memory's.
            db_session: Optional session for the ``user_memories`` mirror.

        Returns:
            The stored memory, with redacted text and ``pii_redacted=True``.

        Raises:
            ValueError: If the memory belongs to a different tenant or user.
        """
        if memory.tenant_id != principal.tenant_id:
            msg = "refusing to store a memory for a different tenant"
            raise ValueError(msg)
        if memory.user_id != principal.user_id:
            msg = "refusing to store a memory for a different user"
            raise ValueError(msg)

        redacted, report = self._pii().scan_and_redact(memory.text)
        stored = memory.model_copy(update={"text": redacted, "pii_redacted": True})
        if report.has_pii:
            _log.info(
                "memory_redacted",
                tenant_id=principal.tenant_id,
                entity_types=report.entity_types,
            )

        client = await self.qdrant()
        vector = await self._embed(stored.text)
        await upsert_points(
            client,
            collection=self.collection,
            points=[
                qm.PointStruct(
                    id=point_id_for_memory(stored.memory_id),
                    vector={DENSE: vector},
                    payload=stored.to_qdrant_payload(),
                )
            ],
            settings=self._settings,
        )
        if stored.supersedes:
            await self._delete_points(principal, [stored.supersedes])
        if db_session is not None:
            await self._mirror(stored, db_session=db_session)
        return stored

    async def write_back(
        self,
        *,
        principal: Principal,
        session_id: str | None,
        user_text: str,
        assistant_text: str = "",
        profile: UserProfile | None = None,
        db_session: AsyncSession | None = None,
        now: datetime | None = None,
    ) -> list[LongTermMemory]:
        """Extract and store durable memories from one turn (pipeline stage 13).

        A single structured ``MODEL_FAST`` call proposes candidates; each is
        PII-redacted, embedded, and compared against the user's existing memories.
        A candidate whose cosine similarity clears ``memory_dedupe_threshold``
        *supersedes* the older memory instead of being appended next to it, which is
        what keeps the store from growing without bound.

        Args:
            principal: The caller.
            session_id: Session the memories came from, for provenance.
            user_text: The user's turn, already PII-redacted by stage 1.
            assistant_text: The answer, already PII-redacted by stage 12.
            profile: Pre-loaded profile. Required for consent unless ``db_session``
                is given.
            db_session: Database session for the relational mirror.
            now: Reference time. Defaults to now (UTC).

        Returns:
            The memories actually stored, possibly empty — which is the common and
            correct outcome for most turns.
        """
        resolved = await self.resolve_profile(
            principal, profile=profile, db_session=db_session
        )
        gate = self.consent_gate(resolved)
        if gate != "ok" or not self._settings.memory_extract_enabled:
            _log.debug(
                "memory_write_back_skipped",
                tenant_id=principal.tenant_id,
                reason=gate if gate != "ok" else "extract_disabled",
            )
            return []
        if not user_text.strip():
            return []

        candidates = await self._extract(user_text, assistant_text)
        if not candidates:
            return []

        moment = now or _utcnow()
        expires_at = (
            moment + timedelta(days=self._settings.memory_ttl_days)
            if self._settings.memory_ttl_days
            else None
        )
        existing = await self._existing(principal, now=moment)
        stored: list[LongTermMemory] = []
        for candidate in candidates:
            memory = await self._build_memory(
                candidate,
                principal=principal,
                session_id=session_id,
                existing=existing,
                expires_at=expires_at,
                now=moment,
            )
            if memory is None:
                continue
            stored.append(
                await self.remember(memory, principal=principal, db_session=db_session)
            )

        if stored:
            await self.prune(principal, db_session=db_session, now=moment)
        _log.info(
            "memory_write_back",
            tenant_id=principal.tenant_id,
            proposed=len(candidates),
            stored=len(stored),
        )
        return stored

    async def _extract(
        self, user_text: str, assistant_text: str
    ) -> list[ExtractedMemory]:
        """Run the structured extraction call.

        Args:
            user_text: The user's turn.
            assistant_text: The answer.

        Returns:
            Candidate memories, or an empty list on refusal or failure. The text is
            redacted defensively before it is sent, because a caller that forgot
            stage 1 must not be able to leak an identifier into a model call.
        """
        detector = self._pii()
        redacted_user, _ = detector.scan_and_redact(user_text)
        redacted_answer, _ = detector.scan_and_redact(assistant_text)
        payload = (
            f"<user_turn>\n{redacted_user.strip()}\n</user_turn>\n\n"
            f"<assistant_turn>\n{redacted_answer.strip()}\n</assistant_turn>"
        )
        try:
            result = await self._client_llm().structured(
                system=MEMORY_EXTRACTION_SYSTEM,
                messages=[{"role": "user", "content": payload}],
                schema=MemoryExtraction,
                model=self._settings.anthropic_model_fast,
                effort=self._settings.anthropic_effort_fast,
                name=EXTRACTION_CALL_NAME,
                metadata=prompt_metadata("memory_extraction"),
            )
        except LLMRefusedError:
            _log.info("memory_extraction_refused")
            return []
        except Exception:
            _log.warning("memory_extraction_failed", exc_info=True)
            return []
        return [
            candidate
            for candidate in result.memories
            if len(candidate.text.strip()) >= _MIN_MEMORY_CHARS
        ]

    async def _existing(
        self, principal: Principal, *, now: datetime
    ) -> list[tuple[LongTermMemory, list[float]]]:
        """Load the user's current memories with their vectors.

        Args:
            principal: The caller.
            now: Reference time for the expiry filter.

        Returns:
            ``(memory, dense_vector)`` pairs, empty on any failure.
        """
        try:
            return await self.scan(principal, now=now, with_vectors=True)
        except Exception:
            _log.warning("memory_existing_scan_failed", exc_info=True)
            return []

    async def _build_memory(
        self,
        candidate: ExtractedMemory,
        *,
        principal: Principal,
        session_id: str | None,
        existing: Sequence[tuple[LongTermMemory, list[float]]],
        expires_at: datetime | None,
        now: datetime,
    ) -> LongTermMemory | None:
        """Turn a candidate into a storable memory, resolving ``supersedes``.

        Args:
            candidate: The extracted candidate.
            principal: The caller.
            session_id: Provenance.
            existing: The user's current memories and vectors.
            expires_at: TTL for the new memory.
            now: Reference time.

        Returns:
            The memory to store, or None when it merely repeats one already held.
        """
        text = candidate.text.strip()
        if len(text) < _MIN_MEMORY_CHARS:
            return None

        supersedes: str | None = None
        if existing:
            try:
                vector = await self._embed(text)
            except Exception:
                _log.warning("memory_dedupe_embed_failed", exc_info=True)
                vector = []
            if vector:
                best_id, best_score = "", 0.0
                for memory, other in existing:
                    if not other:
                        continue
                    score = cosine_similarity(vector, other)
                    if score > best_score:
                        best_id, best_score = memory.memory_id, score
                if best_id and best_score >= self._settings.memory_dedupe_threshold:
                    matched = next(
                        memory for memory, _ in existing if memory.memory_id == best_id
                    )
                    if matched.text.strip().casefold() == text.casefold():
                        return None
                    supersedes = best_id

        return LongTermMemory(
            memory_id=uuid.uuid4().hex,
            user_id=principal.user_id,
            tenant_id=principal.tenant_id,
            kind=_coerce_kind(candidate.kind),
            text=text,
            salience=min(1.0, max(0.0, candidate.salience)),
            source_session_id=session_id,
            supersedes=supersedes,
            created_at=now,
            last_used_at=now,
            expires_at=expires_at,
        )

    # ------------------------------------------------------------------ lifecycle
    async def scan(
        self,
        principal: Principal,
        *,
        limit: int | None = None,
        include_expired: bool = False,
        with_vectors: bool = False,
        now: datetime | None = None,
    ) -> list[tuple[LongTermMemory, list[float]]]:
        """Page through a user's memories.

        Args:
            principal: The caller; supplies the tenant and user scope.
            limit: Maximum memories to return. Defaults to
                ``memory_consolidate_batch_size``.
            include_expired: Include memories past ``expires_at`` (cleanup only).
            with_vectors: Fetch the dense vectors too, for similarity work.
            now: Reference time for the expiry filter.

        Returns:
            ``(memory, vector)`` pairs; the vector is empty when not requested.
        """
        client = await self.qdrant()
        page = limit or int(self._settings.memory_consolidate_batch_size)
        points, _ = await client.scroll(
            collection_name=self.collection,
            scroll_filter=build_memory_filter(
                principal, include_expired=include_expired, now=now or _utcnow()
            ),
            limit=page,
            with_payload=True,
            with_vectors=with_vectors,
        )
        results: list[tuple[LongTermMemory, list[float]]] = []
        for point in points:
            memory = _parse_memory(point.payload)
            if memory is None:
                continue
            results.append((memory, _dense_of(point.vector)))
        return results

    async def forget(
        self,
        principal: Principal,
        memory_id: str,
        *,
        db_session: AsyncSession | None = None,
    ) -> bool:
        """Delete one memory (``DELETE /memory/items/{id}``).

        Args:
            principal: The caller; ownership is verified before deletion.
            memory_id: Memory to delete.
            db_session: Optional session for the relational soft delete.

        Returns:
            True when the point was removed.
        """
        removed = await self._delete_points(principal, [memory_id])
        if db_session is not None:
            from ragcore.db import repositories as repo

            await repo.delete_memory(
                db_session,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                memory_id=memory_id,
            )
        return removed > 0

    async def expire(
        self,
        principal: Principal,
        *,
        now: datetime | None = None,
        db_session: AsyncSession | None = None,
    ) -> int:
        """Remove memories past their TTL.

        Args:
            principal: The caller.
            now: Reference time. Defaults to now (UTC).
            db_session: Optional session for the relational soft delete.

        Returns:
            The number of memories removed.
        """
        reference = now or _utcnow()
        stale = [
            memory.memory_id
            for memory, _ in await self.scan(
                principal, include_expired=True, now=reference
            )
            if memory.is_expired(reference)
        ]
        if not stale:
            return 0
        removed = await self._delete_points(principal, stale)
        if db_session is not None:
            from ragcore.db import repositories as repo

            for memory_id in stale:
                await repo.delete_memory(
                    db_session,
                    tenant_id=principal.tenant_id,
                    user_id=principal.user_id,
                    memory_id=memory_id,
                )
        _log.info("memory_expired", tenant_id=principal.tenant_id, removed=removed)
        return removed

    async def prune(
        self,
        principal: Principal,
        *,
        db_session: AsyncSession | None = None,
        now: datetime | None = None,
    ) -> int:
        """Enforce ``memory_max_per_user``, lowest decayed salience first.

        Args:
            principal: The caller.
            db_session: Optional session for the relational soft delete.
            now: Reference time. Defaults to now (UTC).

        Returns:
            The number of memories removed.
        """
        cap = self._settings.memory_max_per_user
        reference = now or _utcnow()
        held = await self.scan(principal, limit=cap + 1, now=reference)
        if len(held) <= cap:
            return 0
        decay = self._settings.memory_salience_decay_per_day
        ordered = sorted(
            held,
            key=lambda item: item[0].decayed_salience(
                decay_per_day=decay, now=reference
            ),
        )
        victims = [memory.memory_id for memory, _ in ordered[: len(held) - cap]]
        removed = await self._delete_points(principal, victims)
        if db_session is not None:
            from ragcore.db import repositories as repo

            for memory_id in victims:
                await repo.delete_memory(
                    db_session,
                    tenant_id=principal.tenant_id,
                    user_id=principal.user_id,
                    memory_id=memory_id,
                )
        _log.info("memory_pruned", tenant_id=principal.tenant_id, removed=removed)
        return removed

    async def _delete_points(
        self, principal: Principal, memory_ids: Sequence[str]
    ) -> int:
        """Delete memory points after verifying ownership.

        Qdrant deletes by point id, which carries no tenant. Each point is therefore
        retrieved and its payload checked against the caller's tenant *and* user
        before anything is removed, so an id guessed from another tenant deletes
        nothing.

        Args:
            principal: The caller.
            memory_ids: Logical memory ids to remove.

        Returns:
            The number of points deleted.
        """
        if not memory_ids:
            return 0
        client = await self.qdrant()
        point_ids = [point_id_for_memory(memory_id) for memory_id in memory_ids]
        try:
            records = await client.retrieve(
                collection_name=self.collection,
                ids=point_ids,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            _log.warning("memory_delete_retrieve_failed", exc_info=True)
            return 0

        owned: list[Any] = []
        for record in records:
            payload = record.payload or {}
            if (
                payload.get("tenant_id") == principal.tenant_id
                and payload.get("user_id") == principal.user_id
            ):
                owned.append(record.id)
            else:
                _log.error(
                    "memory_delete_ownership_denied", tenant_id=principal.tenant_id
                )
        if not owned:
            return 0
        await client.delete(
            collection_name=self.collection,
            points_selector=qm.PointIdsList(points=owned),
            wait=True,
        )
        return len(owned)

    async def _mirror(
        self, memory: LongTermMemory, *, db_session: AsyncSession
    ) -> None:
        """Mirror a memory into the ``user_memories`` table.

        Args:
            memory: The memory to write.
            db_session: Active database session. The caller commits.
        """
        from ragcore.db import repositories as repo

        try:
            await repo.upsert_memory(db_session, memory, tenant_id=memory.tenant_id)
        except Exception:
            _log.warning("memory_mirror_failed", memory_id=memory.memory_id)


def _dense_of(vector: Any) -> list[float]:
    """Extract the named dense vector from a Qdrant record.

    Args:
        vector: The record's ``vector`` attribute, which is a mapping of named
            vectors for every collection in this platform.

    Returns:
        The dense vector, or an empty list when absent.
    """
    if isinstance(vector, Mapping):
        candidate = vector.get(DENSE)
        return list(candidate) if candidate else []
    if isinstance(vector, Sequence) and not isinstance(vector, str):
        return [float(value) for value in vector]
    return []


_STORES: dict[str, LongTermMemoryStore] = {}


def get_long_term_memory(settings: Settings | None = None) -> LongTermMemoryStore:
    """Return the process-wide long-term memory store.

    ``Settings`` is unhashable, so the cache key is the Qdrant endpoint plus the
    memories collection.

    Args:
        settings: Active settings. Defaults to the process settings.

    Returns:
        The cached store.
    """
    cfg = settings or get_settings()
    key = f"{cfg.qdrant_url}|{cfg.qdrant_memories_collection}"
    existing = _STORES.get(key)
    if existing is None:
        existing = LongTermMemoryStore(settings=cfg)
        _STORES[key] = existing
    return existing


def reset_long_term_memory() -> None:
    """Drop the cached stores. Test helper."""
    _STORES.clear()
