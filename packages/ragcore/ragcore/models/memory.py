"""Personalisation: long-term memory, the rolling user profile, and the semantic cache.

The semantic cache (requirement #2) caches the *retrieval plan and chunk ids*, never
the rendered answer. Cached chunk ids are always re-fetched through the live ACL
filter, so a principal who lost access to a document simply gets fewer chunks instead
of a stale leak.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "LongTermMemory",
    "MemoryKind",
    "SemanticCacheEntry",
    "UserProfile",
    "normalize_query",
]


def _utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    Returns:
        The current moment in UTC.
    """
    return datetime.now(UTC)


def normalize_query(query: str) -> str:
    """Canonicalise a query for semantic-cache keying.

    Lowercases, collapses whitespace and strips trailing punctuation so trivially
    different phrasings of the same question share a cache entry. Embedding
    similarity does the real work; this only removes noise.

    Args:
        query: Raw user query.

    Returns:
        The normalised query string.
    """
    collapsed = " ".join(query.lower().split())
    return collapsed.rstrip("?!. ")


class MemoryKind(StrEnum):
    """What sort of durable fact a memory holds."""

    PREFERENCE = "preference"
    """A standing instruction, e.g. "answer in bullet points"."""

    FACT = "fact"
    """A stable fact about the user, e.g. "I work in the Munich office"."""

    ENTITY = "entity"
    """A salient entity the user repeatedly cares about."""

    EPISODE = "episode"
    """A summary of a past resolved conversation."""


class LongTermMemory(BaseModel):
    """A durable, user-scoped memory retrieved by dense similarity and salience."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(description="Unique memory id; also the Qdrant point id.")
    user_id: str = Field(description="Owning user (Entra oid).")
    tenant_id: str = Field(description="Owning tenant id; the security boundary.")
    kind: MemoryKind = Field(description="preference | fact | entity | episode.")
    text: str = Field(description="Memory text, always PII-redacted before storage.")
    salience: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance in 0..1. Decays with age, boosted on reuse.",
    )
    source_session_id: str | None = Field(
        default=None, description="Session this memory was extracted from."
    )
    supersedes: str | None = Field(
        default=None,
        description="memory_id this replaces, when write-back found a near-duplicate.",
    )
    hit_count: int = Field(
        default=0,
        ge=0,
        description="Times this memory has been injected into a prompt.",
    )
    created_at: datetime = Field(default_factory=_utcnow, description="Creation time.")
    last_used_at: datetime | None = Field(
        default=None, description="Last time this memory was injected."
    )
    expires_at: datetime | None = Field(
        default=None, description="Expiry, or None for no expiry."
    )
    pii_redacted: bool = Field(
        default=False, description="Whether text has been through PII redaction."
    )

    def decayed_salience(
        self, *, decay_per_day: float, now: datetime | None = None
    ) -> float:
        """Salience adjusted for time since last use.

        Args:
            decay_per_day: Salience subtracted per elapsed day
                (``settings.memory_salience_decay_per_day``).
            now: Reference time. Defaults to now (UTC).

        Returns:
            The decayed salience, clamped to 0..1.
        """
        reference = now or _utcnow()
        anchor = self.last_used_at or self.created_at
        anchor = anchor if anchor.tzinfo else anchor.replace(tzinfo=UTC)
        days = max(0.0, (reference - anchor).total_seconds() / 86_400.0)
        return max(0.0, min(1.0, self.salience - decay_per_day * days))

    def is_expired(self, now: datetime | None = None) -> bool:
        """Whether this memory has passed its expiry.

        Args:
            now: Reference time. Defaults to now (UTC).

        Returns:
            True when ``expires_at`` is set and in the past.
        """
        if self.expires_at is None:
            return False
        expiry = (
            self.expires_at
            if self.expires_at.tzinfo
            else self.expires_at.replace(tzinfo=UTC)
        )
        return (now or _utcnow()) >= expiry

    def touch(self, *, boost: float = 0.0, now: datetime | None = None) -> Self:
        """Return a copy marked as used, optionally boosting salience.

        Args:
            boost: Salience to add (``settings.memory_salience_boost_on_hit``).
            now: Reference time. Defaults to now (UTC).

        Returns:
            A copy with ``hit_count`` incremented and ``last_used_at`` refreshed.
        """
        return self.model_copy(
            update={
                "hit_count": self.hit_count + 1,
                "last_used_at": now or _utcnow(),
                "salience": max(0.0, min(1.0, self.salience + boost)),
            }
        )

    def to_qdrant_payload(self) -> dict[str, object]:
        """Serialise for the ``rag_memories`` collection payload.

        Returns:
            A JSON-safe mapping with datetimes as RFC 3339 strings.
        """
        return self.model_dump(mode="json")


class UserProfile(BaseModel):
    """Rolling, LLM-maintained persona for one user within one tenant."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(description="Owning user (Entra oid).")
    tenant_id: str = Field(description="Owning tenant id.")
    summary: str = Field(
        default="",
        description="Rolling persona summary, regenerated every N turns by MODEL_FAST.",
    )
    preferred_style: str | None = Field(
        default=None,
        description="Preferred answer style, e.g. 'concise bullet points'.",
    )
    preferred_language: str | None = Field(
        default=None, description="Preferred response language (ISO 639-1)."
    )
    top_topics: list[str] = Field(
        default_factory=list, description="Topics this user asks about most."
    )
    memory_consent: bool = Field(
        default=True,
        description=(
            "User-controlled switch. When false, stage 13 memory write-back is "
            "skipped entirely and no long-term memory is stored."
        ),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, description="Last update time."
    )

    @classmethod
    def empty(cls, *, user_id: str, tenant_id: str) -> Self:
        """Build a blank profile for a first-time user.

        Args:
            user_id: Entra object id.
            tenant_id: Owning tenant id.

        Returns:
            A profile with consent granted and no learned preferences.
        """
        return cls(user_id=user_id, tenant_id=tenant_id, summary="")

    @property
    def has_content(self) -> bool:
        """Whether this profile is worth injecting into a prompt.

        Returns:
            True when any learned field is populated.
        """
        return bool(
            self.summary
            or self.preferred_style
            or self.preferred_language
            or self.top_topics
        )


class SemanticCacheEntry(BaseModel):
    """Requirement #2: faster retrieval for similar queries.

    Caches the *retrieval plan and chunk ids*, never the rendered answer, so ACLs are
    re-checked on reuse. A hit requires cosine similarity at or above
    ``settings.memory_cache_threshold`` (default 0.94) **and** an exact
    ``filter_fingerprint`` match.
    """

    model_config = ConfigDict(extra="forbid")

    cache_id: str = Field(description="Unique entry id; also the Qdrant point id.")
    tenant_id: str = Field(description="Owning tenant id; the security boundary.")
    user_id: str | None = Field(
        default=None,
        description="Owning user, or None for a tenant-wide entry.",
    )
    normalized_query: str = Field(
        description="Canonicalised query text (see normalize_query)."
    )
    transformed_queries: list[str] = Field(
        default_factory=list, description="Queries the transformer produced."
    )
    chunk_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Chunk ids to re-fetch. Always re-filtered through build_acl_filter on "
            "reuse: a principal who lost access gets fewer chunks, never a stale leak."
        ),
    )
    filter_fingerprint: str = Field(
        description=(
            "Hash of the MetadataFilter plus the principal's clearance. Must match "
            "exactly for a hit."
        )
    )
    hit_count: int = Field(default=0, ge=0, description="Times this entry was reused.")
    created_at: datetime = Field(default_factory=_utcnow, description="Creation time.")
    last_used_at: datetime = Field(
        default_factory=_utcnow, description="Last reuse time."
    )
    ttl_seconds: int = Field(default=86_400, gt=0, description="Entry lifetime.")

    @property
    def expires_at(self) -> datetime:
        """When this entry stops being eligible for reuse.

        Returns:
            ``created_at`` plus :attr:`ttl_seconds`.
        """
        created = (
            self.created_at
            if self.created_at.tzinfo
            else self.created_at.replace(tzinfo=UTC)
        )
        return created + timedelta(seconds=self.ttl_seconds)

    def is_expired(self, now: datetime | None = None) -> bool:
        """Whether this entry has aged out.

        Args:
            now: Reference time. Defaults to now (UTC).

        Returns:
            True when the TTL has elapsed.
        """
        return (now or _utcnow()) >= self.expires_at

    def matches(self, *, fingerprint: str, similarity: float, threshold: float) -> bool:
        """Decide whether this entry may serve a query.

        Args:
            fingerprint: Fingerprint computed for the incoming request.
            similarity: Cosine similarity between the incoming query and this entry.
            threshold: Minimum similarity (``settings.memory_cache_threshold``).

        Returns:
            True only when the fingerprint matches exactly, the similarity clears the
            threshold, and the entry has not expired.
        """
        return (
            fingerprint == self.filter_fingerprint
            and similarity >= threshold
            and not self.is_expired()
        )

    def touch(self, now: datetime | None = None) -> Self:
        """Return a copy marked as reused.

        Args:
            now: Reference time. Defaults to now (UTC).

        Returns:
            A copy with ``hit_count`` incremented and ``last_used_at`` refreshed.
        """
        return self.model_copy(
            update={"hit_count": self.hit_count + 1, "last_used_at": now or _utcnow()}
        )

    @staticmethod
    def make_cache_id(
        *, tenant_id: str, normalized_query: str, fingerprint: str
    ) -> str:
        """Derive a deterministic cache id.

        Keying on the tenant means an identical query in two tenants can never share
        an entry, which keeps the cache inside the security boundary by construction.

        Args:
            tenant_id: Owning tenant id.
            normalized_query: Output of :func:`normalize_query`.
            fingerprint: Filter fingerprint for the request.

        Returns:
            A 32-character hex id.
        """
        digest = hashlib.sha256(
            f"{tenant_id}\x00{normalized_query}\x00{fingerprint}".encode()
        )
        return digest.hexdigest()[:32]
