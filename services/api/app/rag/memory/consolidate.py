"""Periodic memory consolidation — the maintenance half of requirement #2.

Write-back (stage 13) is deliberately conservative: it runs once per turn, sees one
turn, and supersedes only what it can recognise from that turn. Left alone, a memory
store still drifts — near-duplicates accumulate from different phrasings, salience
never decays for memories nobody uses, expired entries linger, and the rolling profile
goes stale.

This job fixes all four, per user, inside one tenant:

* **merge** near-duplicates (cosine ≥ ``memory_dedupe_threshold``), keeping the
  memory with the higher decayed salience and superseding the other;
* **decay** salience by ``memory_salience_decay_per_day`` since last use, persisting
  the decayed value so recall does not have to recompute history;
* **expire** entries past ``expires_at`` and prune past ``memory_max_per_user``;
* **refresh** the rolling :class:`~ragcore.models.memory.UserProfile` from what
  survives, with a single ``MODEL_FAST`` structured call.

It is safe to run repeatedly and it respects the same consent gate as everything else:
a user with ``memory_consent=False`` has nothing to consolidate, and this job will not
create anything for them.

Invoke it from a scheduled task, an admin endpoint, or opportunistically after
write-back. It never raises into a caller — every failure is counted in
:attr:`ConsolidationReport.errors`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.rag.memory import optional_setting
from app.rag.memory.long_term import LongTermMemoryStore, get_long_term_memory
from ragcore.embeddings import cosine_similarity
from ragcore.llm import get_llm_client
from ragcore.llm.client import LLMClient, LLMRefusedError
from ragcore.llm.prompts import PROFILE_SUMMARY_SYSTEM, prompt_metadata
from ragcore.models.acl import Principal
from ragcore.models.memory import LongTermMemory, UserProfile
from ragcore.settings import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "PROFILE_CALL_NAME",
    "ConsolidationReport",
    "MemoryConsolidator",
    "ProfileDraft",
]

_log = structlog.get_logger(__name__)

#: Trace/operation name for the profile-refresh call.
PROFILE_CALL_NAME = "rag.profile_summary"

#: Salience changes smaller than this are not worth a write.
_SALIENCE_EPSILON = 1e-6


def _utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    Returns:
        The current moment in UTC.
    """
    return datetime.now(UTC)


class ProfileDraft(BaseModel):
    """Wire schema for the rolling-profile refresh call.

    Flat and primitive for the same reason as the memory extraction schema:
    structured outputs strip most JSON Schema keywords, so anything richer is the
    shape most likely to come back unusable.
    """

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(
        default="", description="Compact third-person persona paragraph."
    )
    preferred_style: str = Field(
        default="", description="Preferred answer style, empty when unknown."
    )
    preferred_language: str = Field(
        default="", description="Preferred language, empty when unknown."
    )
    top_topics: list[str] = Field(
        default_factory=list, description="Topics the user returns to."
    )


class ConsolidationReport(BaseModel):
    """What one consolidation pass did, for logging and for the admin surface."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(description="Tenant that was consolidated.")
    user_id: str = Field(description="User that was consolidated.")
    scanned: int = Field(default=0, ge=0, description="Memories examined.")
    merged: int = Field(
        default=0, ge=0, description="Near-duplicates folded into a survivor."
    )
    decayed: int = Field(
        default=0, ge=0, description="Memories whose salience was rewritten."
    )
    expired: int = Field(default=0, ge=0, description="Memories past their TTL.")
    pruned: int = Field(
        default=0, ge=0, description="Memories removed to honour memory_max_per_user."
    )
    profile_refreshed: bool = Field(
        default=False, description="Whether the rolling profile was regenerated."
    )
    skipped_reason: str = Field(
        default="",
        description="'disabled' | 'no_consent' | 'consent_unknown' when nothing ran.",
    )
    duration_ms: float = Field(default=0.0, ge=0.0, description="Pass duration.")
    errors: list[str] = Field(
        default_factory=list,
        description="Exception type names. Never a message: those can quote content.",
    )

    @property
    def changed(self) -> bool:
        """Whether the pass altered anything.

        Returns:
            True when memories were merged, decayed, expired or pruned, or the
            profile was refreshed.
        """
        return bool(
            self.merged
            or self.decayed
            or self.expired
            or self.pruned
            or self.profile_refreshed
        )


class MemoryConsolidator:
    """Periodic maintenance over one user's long-term memories."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        store: LongTermMemoryStore | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        """Initialise the consolidator.

        Args:
            settings: Active settings. Defaults to the process settings.
            store: Long-term memory store. Defaults to the process store.
            llm: Client used for the profile refresh.
        """
        self._settings = settings or get_settings()
        self._store = store or get_long_term_memory(self._settings)
        self._llm = llm

    @property
    def settings(self) -> Settings:
        """Settings this consolidator was built from.

        Returns:
            The bound settings.
        """
        return self._settings

    def _client(self) -> LLMClient:
        """Resolve the LLM client.

        Returns:
            The bound or process client.
        """
        if self._llm is None:
            self._llm = get_llm_client(self._settings)
        return self._llm

    async def consolidate_user(
        self,
        *,
        tenant_id: str,
        user_id: str,
        db_session: AsyncSession | None = None,
        profile: UserProfile | None = None,
        now: datetime | None = None,
        refresh_profile: bool = True,
    ) -> ConsolidationReport:
        """Run one consolidation pass for a single user.

        Args:
            tenant_id: Owning tenant; every query is scoped to it.
            user_id: Owning user.
            db_session: Database session for the relational mirrors and the profile.
                Omit to consolidate Qdrant only.
            profile: Pre-loaded profile. Required for consent unless ``db_session``
                is given.
            now: Reference time for decay and expiry. Defaults to now (UTC).
            refresh_profile: Regenerate the rolling persona summary.

        Returns:
            What the pass did. Failures are counted rather than raised: a maintenance
            job must not be able to take the API down.
        """
        started = time.perf_counter()
        reference = now or _utcnow()
        principal = Principal(user_id=user_id, tenant_id=tenant_id)
        report = ConsolidationReport(tenant_id=tenant_id, user_id=user_id)

        resolved = await self._store.resolve_profile(
            principal, profile=profile, db_session=db_session
        )
        gate = self._store.consent_gate(resolved)
        if gate != "ok":
            report.skipped_reason = gate
            report.duration_ms = (time.perf_counter() - started) * 1000.0
            return report

        try:
            report.expired = await self._store.expire(
                principal, now=reference, db_session=db_session
            )
        except Exception as exc:
            _log.warning("consolidate_expire_failed", tenant_id=tenant_id)
            report.errors.append(type(exc).__name__)

        held: list[tuple[LongTermMemory, list[float]]] = []
        try:
            held = await self._store.scan(
                principal,
                limit=int(
                    optional_setting(self._settings, "memory_consolidate_batch_size")
                ),
                with_vectors=True,
                now=reference,
            )
        except Exception as exc:
            _log.warning("consolidate_scan_failed", tenant_id=tenant_id)
            report.errors.append(type(exc).__name__)
        report.scanned = len(held)

        survivors = held
        if held:
            try:
                survivors, merged = await self._merge(
                    principal, held, now=reference, db_session=db_session
                )
                report.merged = merged
            except Exception as exc:
                _log.warning("consolidate_merge_failed", tenant_id=tenant_id)
                report.errors.append(type(exc).__name__)

        if survivors:
            try:
                report.decayed = await self._decay(
                    principal, survivors, now=reference, db_session=db_session
                )
            except Exception as exc:
                _log.warning("consolidate_decay_failed", tenant_id=tenant_id)
                report.errors.append(type(exc).__name__)

        try:
            report.pruned = await self._store.prune(
                principal, db_session=db_session, now=reference
            )
        except Exception as exc:
            _log.warning("consolidate_prune_failed", tenant_id=tenant_id)
            report.errors.append(type(exc).__name__)

        if refresh_profile and resolved is not None:
            try:
                report.profile_refreshed = await self.refresh_profile(
                    resolved,
                    [memory for memory, _ in survivors],
                    db_session=db_session,
                )
            except Exception as exc:
                _log.warning("consolidate_profile_failed", tenant_id=tenant_id)
                report.errors.append(type(exc).__name__)

        report.duration_ms = (time.perf_counter() - started) * 1000.0
        _log.info(
            "memory_consolidated",
            tenant_id=tenant_id,
            scanned=report.scanned,
            merged=report.merged,
            decayed=report.decayed,
            expired=report.expired,
            pruned=report.pruned,
            profile_refreshed=report.profile_refreshed,
            duration_ms=round(report.duration_ms, 1),
            errors=len(report.errors),
        )
        return report

    async def consolidate_users(
        self,
        *,
        tenant_id: str,
        user_ids: Sequence[str],
        db_session: AsyncSession | None = None,
        now: datetime | None = None,
    ) -> list[ConsolidationReport]:
        """Consolidate several users in one tenant, sequentially.

        Sequential on purpose: the pass is I/O-light per user and embarrassingly
        interruptible, and a burst of concurrent Qdrant scrolls competes with live
        chat traffic for the same connection pool.

        Args:
            tenant_id: Owning tenant.
            user_ids: Users to consolidate.
            db_session: Database session shared by every pass. The caller commits.
            now: Reference time. Defaults to now (UTC).

        Returns:
            One report per user, in input order.
        """
        reference = now or _utcnow()
        return [
            await self.consolidate_user(
                tenant_id=tenant_id,
                user_id=user_id,
                db_session=db_session,
                now=reference,
            )
            for user_id in user_ids
        ]

    # ---------------------------------------------------------------------- merge
    async def _merge(
        self,
        principal: Principal,
        held: Sequence[tuple[LongTermMemory, list[float]]],
        *,
        now: datetime,
        db_session: AsyncSession | None,
    ) -> tuple[list[tuple[LongTermMemory, list[float]]], int]:
        """Fold near-duplicate memories into their strongest sibling.

        Comparison is over the stored dense vectors, so no re-embedding is needed.
        The survivor is the memory with the higher decayed salience; it inherits the
        loser's ``hit_count`` so reuse evidence is not lost, and records the loser in
        ``supersedes`` so the audit trail says what replaced what.

        Args:
            principal: The caller.
            held: ``(memory, vector)`` pairs from the scan.
            now: Reference time.
            db_session: Optional session for the relational mirror.

        Returns:
            A ``(survivors, merged_count)`` pair.
        """
        threshold = self._settings.memory_dedupe_threshold
        decay = self._settings.memory_salience_decay_per_day
        ordered = sorted(
            held,
            key=lambda item: item[0].decayed_salience(decay_per_day=decay, now=now),
            reverse=True,
        )

        survivors: list[tuple[LongTermMemory, list[float]]] = []
        merged = 0
        for memory, vector in ordered:
            position = -1
            if vector:
                for index, (other, other_vector) in enumerate(survivors):
                    if not other_vector or other.kind is not memory.kind:
                        continue
                    if cosine_similarity(vector, other_vector) >= threshold:
                        position = index
                        break
            if position < 0:
                survivors.append((memory, vector))
                continue

            keeper, keeper_vector = survivors[position]
            updated = keeper.model_copy(
                update={
                    "hit_count": keeper.hit_count + memory.hit_count,
                    "supersedes": memory.memory_id,
                    "salience": max(keeper.salience, memory.salience),
                }
            )
            survivors[position] = (updated, keeper_vector)
            # `remember` applies `supersedes` by deleting the losing point, so the
            # merge is one write rather than a write plus an orphan.
            await self._store.remember(
                updated, principal=principal, db_session=db_session
            )
            merged += 1
        return survivors, merged

    async def _decay(
        self,
        principal: Principal,
        held: Sequence[tuple[LongTermMemory, list[float]]],
        *,
        now: datetime,
        db_session: AsyncSession | None,
    ) -> int:
        """Persist the decayed salience of each memory.

        Recall decays on read as well, so this is not required for correctness — it
        makes the stored value honest, which is what the pruning order and the admin
        surface both read.

        Args:
            principal: The caller.
            held: ``(memory, vector)`` pairs to update.
            now: Reference time.
            db_session: Optional session for the relational mirror.

        Returns:
            The number of memories whose stored salience changed.
        """
        decay = self._settings.memory_salience_decay_per_day
        if decay <= 0:
            return 0
        changed = 0
        for memory, _ in held:
            decayed = memory.decayed_salience(decay_per_day=decay, now=now)
            if abs(decayed - memory.salience) < _SALIENCE_EPSILON:
                continue
            if await self._store.set_salience(
                principal, memory, decayed, db_session=db_session
            ):
                changed += 1
        return changed

    # -------------------------------------------------------------------- profile
    async def refresh_profile(
        self,
        profile: UserProfile,
        memories: Sequence[LongTermMemory],
        *,
        db_session: AsyncSession | None = None,
    ) -> bool:
        """Regenerate the rolling persona summary from surviving memories.

        Args:
            profile: The current profile.
            memories: The user's surviving memories.
            db_session: Session used to persist the result. Without one the refresh
                is computed and discarded, which is only useful in tests.

        Returns:
            True when a new profile was produced and persisted.
        """
        minimum = int(optional_setting(self._settings, "memory_profile_min_memories"))
        if len(memories) < minimum:
            return False

        rendered = "\n".join(
            f"- ({memory.kind.value}) {memory.text}" for memory in memories
        )
        payload = (
            f"<memories>\n{rendered}\n</memories>\n\n"
            f"<current_profile>\n{profile.summary or '(none)'}\n</current_profile>"
        )
        try:
            draft = await self._client().structured(
                system=PROFILE_SUMMARY_SYSTEM,
                messages=[{"role": "user", "content": payload}],
                schema=ProfileDraft,
                model=self._settings.anthropic_model_fast,
                effort=self._settings.anthropic_effort_fast,
                name=PROFILE_CALL_NAME,
                metadata=prompt_metadata("profile_summary"),
            )
        except LLMRefusedError:
            _log.info("profile_refresh_refused", tenant_id=profile.tenant_id)
            return False
        except Exception:
            _log.warning("profile_refresh_failed", tenant_id=profile.tenant_id)
            return False

        if not draft.summary.strip():
            return False
        updated = profile.model_copy(
            update={
                "summary": draft.summary.strip(),
                "preferred_style": draft.preferred_style.strip()
                or profile.preferred_style,
                "preferred_language": draft.preferred_language.strip()
                or profile.preferred_language,
                "top_topics": [topic for topic in draft.top_topics if topic.strip()]
                or profile.top_topics,
                "updated_at": _utcnow(),
            }
        )
        if db_session is None:
            return False
        from ragcore.db import repositories as repo

        await repo.save_profile(db_session, updated)
        return True
