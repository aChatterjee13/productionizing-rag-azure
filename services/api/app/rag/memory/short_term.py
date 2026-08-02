"""Short-term session memory — pipeline stages 2 and 9, requirement #5.

This module owns the *live conversation window*: the turns that are actually replayed
to the model, the rolling summary that stands in for the turns that are not, the pins
that protect a turn from ever being suppressed, and the token accounting that makes
all of it measurable.

Three properties it is built around:

* **Tokens are measured, never estimated.** :class:`TokenCounter` calls
  ``LLMClient.count_tokens`` (Claude's own tokenizer) and memoises the result by
  content digest, so a stable conversation costs one measurement per *new* item
  rather than one per turn. ``tiktoken`` is the wrong tokenizer for Claude and is
  never used; the only fallback is the characters/4 degradation *inside*
  ``LLMClient`` when the API is unreachable, which is that client's documented
  contract.
* **Suppression is lossless.** An over-budget turn is folded into the rolling summary
  and marked ``suppressed``; it is never truncated and never silently dropped. Pinned
  turns are excluded from every suppression path — in the window, in the store and in
  the SQL update (``repositories.suppress_messages`` filters on ``pinned``).
* **Nothing raw is persisted.** :meth:`ShortTermMemory.record_turn` refuses content
  that has not passed PII redaction, exactly as ``repositories.append_message`` does.
  The session store is persistence: Redis is a database.

The store is Redis-backed with a working in-memory fallback. The fallback is
per-process and therefore not shared between API replicas — a miss simply re-hydrates
the window from PostgreSQL, which is always the source of truth.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

import structlog
from pydantic import BaseModel, ConfigDict, Field

from ragcore.llm import get_llm_client
from ragcore.llm.client import LLMClient
from ragcore.llm.prompts import SESSION_SUMMARY_SYSTEM, prompt_metadata
from ragcore.models.acl import Principal
from ragcore.models.chat import Message, Role
from ragcore.settings import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "CompactionOutcome",
    "InMemorySessionStore",
    "RedisSessionStore",
    "SessionStore",
    "SessionWindow",
    "ShortTermMemory",
    "TokenCounter",
    "get_short_term_memory",
    "reset_short_term_memory",
    "summarise_turns",
]

_log = structlog.get_logger(__name__)

#: Trace/operation name for the rolling-summary call.
SUMMARY_CALL_NAME = "rag.session_summary"

#: Token cost attributed to text that is blank and therefore never sent.
_EMPTY_TOKENS = 0

#: Characters of a retired turn kept by the deterministic fallback summary.
_FALLBACK_TURN_CHARS = 400


def _utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    Returns:
        The current moment in UTC.
    """
    return datetime.now(UTC)


def _fingerprint(text: str) -> str:
    """Hash a text for the token-count memo.

    The memo key is a digest rather than the text itself, so the cache never holds a
    second copy of user content in memory.

    Args:
        text: Text about to be measured.

    Returns:
        A 32-character hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _jsonable(value: Any) -> Any:
    """Reduce a request payload to something ``json.dumps`` can key on.

    Args:
        value: Arbitrary system or message payload.

    Returns:
        A JSON-safe projection; unknown objects fall back to ``str``.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence):
        return [_jsonable(item) for item in value]
    return str(value)


# --------------------------------------------------------------------- counting
class TokenCounter:
    """Measures prompt tokens with Claude's tokenizer, memoised by content.

    Context assembly needs a per-item token cost to pack a budget, and one exact
    measurement of the finished payload to prove the budget was respected. Both go
    through here so the number of API round trips stays bounded: repeated turns,
    repeated chunks and an unchanged system prompt are measured once per process.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        settings: Settings | None = None,
        model: str | None = None,
    ) -> None:
        """Initialise the counter.

        Args:
            llm: Client used for measurement. Defaults to the process client.
            settings: Active settings. Defaults to the process settings.
            model: Model whose tokenizer is used. Defaults to
                ``settings.anthropic_model_main``, which is the model the assembled
                prompt is actually sent to.
        """
        self._settings = settings or get_settings()
        self._llm = llm if llm is not None else get_llm_client(self._settings)
        self._model = model or self._settings.anthropic_model_main
        self._limit = int(self._settings.context_token_cache_entries)
        self._semaphore = asyncio.Semaphore(
            int(self._settings.context_token_count_concurrency)
        )
        self._memo: OrderedDict[str, int] = OrderedDict()
        self._calls = 0

    @property
    def calls(self) -> int:
        """Number of measurement round trips made so far.

        Returns:
            The count of uncached ``count_tokens`` calls, which is where a cost or
            latency regression shows up.
        """
        return self._calls

    @property
    def model(self) -> str:
        """Model whose tokenizer is used.

        Returns:
            The resolved model id.
        """
        return self._model

    def _remember(self, key: str, value: int) -> None:
        """Store a measurement, evicting the least recently used entry.

        Args:
            key: Content digest.
            value: Measured token count.
        """
        self._memo[key] = value
        self._memo.move_to_end(key)
        while len(self._memo) > self._limit:
            self._memo.popitem(last=False)

    def peek(self, text: str, *, role: str = "user") -> int | None:
        """Return a memoised measurement without measuring.

        Args:
            text: Text to look up.
            role: Role the text would be sent under.

        Returns:
            The cached token count, or None when it has not been measured.
        """
        if not text.strip():
            return _EMPTY_TOKENS
        return self._memo.get(f"{role}:{_fingerprint(text)}")

    async def count_text(self, text: str, *, role: str = "user") -> int:
        """Measure one rendered block as it would appear in a message.

        Args:
            text: Rendered block text.
            role: Role the block would be sent under. Framing differs slightly
                between roles, so it participates in the memo key.

        Returns:
            Token count, or 0 for blank text, which is never sent.
        """
        if not text.strip():
            return _EMPTY_TOKENS
        key = f"{role}:{_fingerprint(text)}"
        cached = self._memo.get(key)
        if cached is not None:
            self._memo.move_to_end(key)
            return cached
        async with self._semaphore:
            self._calls += 1
            tokens = await self._llm.count_tokens(
                messages=[{"role": role, "content": text}], model=self._model
            )
        self._remember(key, tokens)
        return tokens

    async def count_many(
        self, texts: Sequence[str], *, role: str = "user"
    ) -> list[int]:
        """Measure several blocks concurrently.

        Args:
            texts: Rendered block texts.
            role: Role the blocks would be sent under.

        Returns:
            Token counts in the same order as ``texts``.
        """
        if not texts:
            return []
        return list(
            await asyncio.gather(*(self.count_text(t, role=role) for t in texts))
        )

    async def count_message(self, message: Message) -> int:
        """Measure a stored turn.

        Args:
            message: The turn to measure.

        Returns:
            Token count for the turn as it will be replayed.
        """
        return await self.count_text(message.content, role=message.role.value)

    async def count_prompt(
        self,
        *,
        system: Any,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> int:
        """Measure a complete request payload exactly.

        This is the number :attr:`~ragcore.models.chat.ContextStats.window_tokens`
        reports and the number the budget check compares against: per-item sums
        ignore the framing the API adds around a whole request.

        Args:
            system: System prompt as text or blocks.
            messages: The full ``messages`` array.
            tools: Tool definitions, which also consume prompt tokens.

        Returns:
            Prompt tokens for the whole request.
        """
        if not messages:
            return _EMPTY_TOKENS
        payload = json.dumps(
            {
                "system": _jsonable(system),
                "messages": _jsonable(list(messages)),
                "tools": _jsonable(list(tools or [])),
            },
            sort_keys=True,
            default=str,
        )
        key = f"prompt:{_fingerprint(payload)}"
        cached = self._memo.get(key)
        if cached is not None:
            self._memo.move_to_end(key)
            return cached
        async with self._semaphore:
            self._calls += 1
            tokens = await self._llm.count_tokens(
                system=system,
                messages=[dict(message) for message in messages],
                model=self._model,
                tools=[dict(tool) for tool in tools] if tools else None,
            )
        self._remember(key, tokens)
        return tokens


# ----------------------------------------------------------------- the window
class SessionWindow(BaseModel):
    """The live conversation window for one session.

    Holds only the turns still replayed to the model. Suppressed turns leave the
    window and live on in PostgreSQL (with ``suppressed=True``) and, in condensed
    form, in :attr:`rolling_summary`.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(description="Owning tenant id; the security boundary.")
    user_id: str = Field(description="Owning user (Entra oid).")
    session_id: str = Field(description="Session this window belongs to.")
    turns: list[Message] = Field(
        default_factory=list, description="Live turns, oldest first."
    )
    rolling_summary: str = Field(
        default="", description="Condensed stand-in for the suppressed turns."
    )
    summary_tokens: int = Field(
        default=0, ge=0, description="Measured token cost of the rolling summary."
    )
    compaction_events: int = Field(
        default=0, ge=0, description="Compactions performed on this session."
    )
    turns_since_compaction: int = Field(
        default=0,
        ge=0,
        description=(
            "Turns appended since the last compaction. Drives the periodic "
            "suppression requirement #5 asks for."
        ),
    )
    suppressed_count: int = Field(
        default=0, ge=0, description="Turns folded into the summary so far."
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, description="Last mutation time."
    )

    @classmethod
    def empty(cls, *, tenant_id: str, user_id: str, session_id: str) -> Self:
        """Build an empty window for a new session.

        Args:
            tenant_id: Owning tenant id.
            user_id: Owning user id.
            session_id: Session id.

        Returns:
            A window with no turns and no summary.
        """
        return cls(tenant_id=tenant_id, user_id=user_id, session_id=session_id)

    @property
    def live_turns(self) -> list[Message]:
        """Turns eligible to be replayed.

        Returns:
            Every turn that is pinned or not suppressed, oldest first.
        """
        return [turn for turn in self.turns if turn.is_live]

    @property
    def pinned_turns(self) -> list[Message]:
        """Turns that may never be suppressed.

        Returns:
            The pinned turns, oldest first.
        """
        return [turn for turn in self.turns if turn.pinned]

    @property
    def total_tokens(self) -> int:
        """Measured token cost of the live window.

        Returns:
            Sum of ``token_count`` over the live turns plus the summary.
        """
        return sum(turn.token_count for turn in self.live_turns) + self.summary_tokens

    def recent(self, count: int) -> list[Message]:
        """Return the newest ``count`` live turns.

        Args:
            count: How many turns to take. Non-positive returns nothing.

        Returns:
            The turns, oldest first.
        """
        if count <= 0:
            return []
        return self.live_turns[-count:]

    def history_pairs(self, count: int | None = None) -> list[tuple[str, str]]:
        """Render the window as ``(role, content)`` pairs.

        Args:
            count: Optional cap on the number of newest turns included.

        Returns:
            Pairs suitable for :func:`ragcore.llm.prompts.render_history`.
        """
        turns = self.live_turns if count is None else self.recent(count)
        return [(turn.role.value, turn.content) for turn in turns]

    def append(self, message: Message) -> Self:
        """Add a turn to the window.

        Args:
            message: The turn to add.

        Returns:
            This window, mutated in place for call-site convenience.
        """
        self.turns.append(message)
        self.turns_since_compaction += 1
        self.updated_at = _utcnow()
        return self

    def replace(self, message: Message) -> bool:
        """Replace a stored turn in place, matched on ``message_id``.

        Args:
            message: The replacement turn.

        Returns:
            True when a turn with that id was present.
        """
        for index, turn in enumerate(self.turns):
            if turn.message_id == message.message_id:
                self.turns[index] = message
                self.updated_at = _utcnow()
                return True
        return False

    def pin(self, message_id: str, *, pinned: bool = True) -> bool:
        """Pin or unpin a turn.

        Args:
            message_id: Turn to change.
            pinned: True to pin, False to unpin.

        Returns:
            True when a turn was found and changed.
        """
        for index, turn in enumerate(self.turns):
            if turn.message_id == message_id:
                self.turns[index] = turn.model_copy(update={"pinned": pinned})
                self.updated_at = _utcnow()
                return True
        return False

    def suppressible(self, *, keep_live: int) -> list[Message]:
        """Choose the oldest turns that may be retired.

        Pinned turns are never returned, however tight the budget is, and the newest
        ``keep_live`` turns are always protected.

        Args:
            keep_live: Number of newest turns to protect.

        Returns:
            The retirable turns, oldest first.
        """
        live = self.live_turns
        protected = {turn.message_id for turn in live[-max(keep_live, 0) :]}
        return [
            turn
            for turn in live
            if not turn.pinned and turn.message_id not in protected
        ]

    def suppress(
        self,
        message_ids: Iterable[str],
        *,
        summary: str | None = None,
        summary_tokens: int | None = None,
    ) -> list[Message]:
        """Retire turns from the live window.

        Args:
            message_ids: Turns to retire. Pinned turns in this set are ignored, which
                is the invariant that makes ``pinned`` a guarantee rather than a hint.
            summary: New rolling summary standing in for the retired turns.
            summary_tokens: Measured token cost of ``summary``.

        Returns:
            The turns that were actually retired, oldest first.
        """
        wanted = set(message_ids)
        retired: list[Message] = []
        remaining: list[Message] = []
        for turn in self.turns:
            if turn.message_id in wanted and not turn.pinned:
                retired.append(turn.model_copy(update={"suppressed": True}))
                continue
            remaining.append(turn)
        if not retired and summary is None:
            return []

        self.turns = remaining
        self.suppressed_count += len(retired)
        if summary is not None:
            self.rolling_summary = summary
            self.summary_tokens = max(0, summary_tokens or 0)
        if retired:
            self.compaction_events += 1
            self.turns_since_compaction = 0
        self.updated_at = _utcnow()
        return retired

    def trim(self, max_turns: int) -> list[Message]:
        """Drop the oldest non-pinned turns beyond a hard cap, without summarising.

        This is the guard rail used only when ``context_compaction_enabled`` is off:
        the operator has asked for turns to leave the window rather than be folded
        into a summary. The rows stay in PostgreSQL marked ``suppressed``.

        Args:
            max_turns: Maximum live turns to keep.

        Returns:
            The turns removed, oldest first.
        """
        live = self.live_turns
        if max_turns < 0 or len(live) <= max_turns:
            return []
        excess = len(live) - max_turns
        victims = [turn.message_id for turn in self.suppressible(keep_live=max_turns)]
        return self.suppress(victims[:excess])

    def to_payload(self) -> dict[str, Any]:
        """Serialise for the session store.

        Returns:
            A JSON-safe mapping.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild a window from its stored form.

        Args:
            payload: Output of :meth:`to_payload`.

        Returns:
            The parsed window.
        """
        return cls.model_validate(dict(payload))


# ------------------------------------------------------------------- the store
@runtime_checkable
class SessionStore(Protocol):
    """Key-value persistence for the live window, keyed by tenant and session."""

    async def load(
        self, *, tenant_id: str, session_id: str
    ) -> dict[str, Any] | None:  # pragma: no cover - protocol
        """Read a stored window.

        Args:
            tenant_id: Owning tenant id.
            session_id: Session to read.

        Returns:
            The stored payload, or None on a miss.
        """
        ...

    async def save(
        self,
        *,
        tenant_id: str,
        session_id: str,
        payload: Mapping[str, Any],
        ttl_seconds: int,
    ) -> None:  # pragma: no cover - protocol
        """Write a window.

        Args:
            tenant_id: Owning tenant id.
            session_id: Session to write.
            payload: Output of :meth:`SessionWindow.to_payload`.
            ttl_seconds: Entry lifetime.
        """
        ...

    async def delete(
        self, *, tenant_id: str, session_id: str
    ) -> None:  # pragma: no cover - protocol
        """Remove a window.

        Args:
            tenant_id: Owning tenant id.
            session_id: Session to remove.
        """
        ...

    async def close(self) -> None:  # pragma: no cover - protocol
        """Release any transport the store owns."""
        ...


def _session_key(prefix: str, tenant_id: str, session_id: str) -> str:
    """Build the tenant-scoped store key.

    The tenant is the first segment, so a key can never be written under one tenant
    and read back under another even if session ids collided.

    Args:
        prefix: Configured key prefix.
        tenant_id: Owning tenant id.
        session_id: Session id.

    Returns:
        The composed key.
    """
    return f"{prefix}{tenant_id}:{session_id}"


class InMemorySessionStore:
    """Process-local fallback store.

    Correct but not shared: another API replica sees a miss and re-hydrates from
    PostgreSQL, which is authoritative. Entries carry an expiry so a long-lived
    process does not accumulate abandoned sessions.
    """

    def __init__(self, *, prefix: str = "rag:session:") -> None:
        """Initialise the store.

        Args:
            prefix: Key prefix, mirroring the Redis layout.
        """
        self._prefix = prefix
        self._data: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def load(self, *, tenant_id: str, session_id: str) -> dict[str, Any] | None:
        """Read a stored window.

        Args:
            tenant_id: Owning tenant id.
            session_id: Session to read.

        Returns:
            The stored payload, or None when absent or expired.
        """
        key = _session_key(self._prefix, tenant_id, session_id)
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, payload = entry
            if expires_at <= time.monotonic():
                self._data.pop(key, None)
                return None
            return json.loads(json.dumps(payload))

    async def save(
        self,
        *,
        tenant_id: str,
        session_id: str,
        payload: Mapping[str, Any],
        ttl_seconds: int,
    ) -> None:
        """Write a window.

        Args:
            tenant_id: Owning tenant id.
            session_id: Session to write.
            payload: Output of :meth:`SessionWindow.to_payload`.
            ttl_seconds: Entry lifetime.
        """
        key = _session_key(self._prefix, tenant_id, session_id)
        async with self._lock:
            self._data[key] = (time.monotonic() + ttl_seconds, dict(payload))

    async def delete(self, *, tenant_id: str, session_id: str) -> None:
        """Remove a window.

        Args:
            tenant_id: Owning tenant id.
            session_id: Session to remove.
        """
        async with self._lock:
            self._data.pop(_session_key(self._prefix, tenant_id, session_id), None)

    async def close(self) -> None:
        """Drop every entry."""
        async with self._lock:
            self._data.clear()


class RedisSessionStore:
    """Redis-backed session store.

    ``redis`` is an optional import: when it is unavailable, or a call fails, the
    store degrades to "miss" and :func:`get_short_term_memory` selects
    :class:`InMemorySessionStore` instead. A cache outage costs latency, never
    correctness — PostgreSQL still holds every turn.
    """

    def __init__(self, url: str, *, prefix: str = "rag:session:") -> None:
        """Initialise the store.

        Args:
            url: Redis URL, e.g. ``redis://localhost:6379/0``.
            prefix: Key prefix.
        """
        self._url = url
        self._prefix = prefix
        self._client: Any | None = None
        self._available = True

    @property
    def available(self) -> bool:
        """Whether the store believes Redis is usable.

        Returns:
            False once the ``redis`` package could not be imported.
        """
        return self._available

    def probe(self) -> bool:
        """Build the client eagerly so store selection can fall back.

        Returns:
            True when a Redis client could be constructed.
        """
        return self._connect() is not None

    def _connect(self) -> Any | None:
        """Build the Redis client lazily.

        Returns:
            The client, or None when ``redis`` is not installed.
        """
        if self._client is not None or not self._available:
            return self._client
        try:
            import redis.asyncio as redis_asyncio
        except ImportError:
            _log.warning("redis_import_failed_using_memory_session_store")
            self._available = False
            return None
        self._client = redis_asyncio.from_url(self._url, decode_responses=True)
        return self._client

    async def load(self, *, tenant_id: str, session_id: str) -> dict[str, Any] | None:
        """Read a stored window.

        Args:
            tenant_id: Owning tenant id.
            session_id: Session to read.

        Returns:
            The stored payload, or None on a miss or a transport error.
        """
        client = self._connect()
        if client is None:
            return None
        key = _session_key(self._prefix, tenant_id, session_id)
        try:
            raw = await client.get(key)
        except Exception:
            _log.warning("redis_session_load_failed", exc_info=True)
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            _log.warning("redis_session_payload_corrupt")
            return None

    async def save(
        self,
        *,
        tenant_id: str,
        session_id: str,
        payload: Mapping[str, Any],
        ttl_seconds: int,
    ) -> None:
        """Write a window.

        Args:
            tenant_id: Owning tenant id.
            session_id: Session to write.
            payload: Output of :meth:`SessionWindow.to_payload`.
            ttl_seconds: Entry lifetime.
        """
        client = self._connect()
        if client is None:
            return
        key = _session_key(self._prefix, tenant_id, session_id)
        try:
            await client.set(key, json.dumps(payload), ex=max(1, ttl_seconds))
        except Exception:
            _log.warning("redis_session_save_failed", exc_info=True)

    async def delete(self, *, tenant_id: str, session_id: str) -> None:
        """Remove a window.

        Args:
            tenant_id: Owning tenant id.
            session_id: Session to remove.
        """
        client = self._connect()
        if client is None:
            return
        try:
            await client.delete(_session_key(self._prefix, tenant_id, session_id))
        except Exception:
            _log.warning("redis_session_delete_failed", exc_info=True)

    async def close(self) -> None:
        """Close the Redis connection pool."""
        if self._client is None:
            return
        try:
            await self._client.aclose()
        except Exception:
            _log.debug("redis_session_close_failed", exc_info=True)
        finally:
            self._client = None


# --------------------------------------------------------------------- summary
def _extractive_summary(
    turns: Sequence[Message], *, current: str, max_chars: int
) -> str:
    """Build a deterministic stand-in summary without a model call.

    Used when summarisation is unavailable (no API key, a refusal, a transport
    error). It is worse prose than the model's, but it keeps suppression *lossless
    enough* to continue the conversation, which is the property that matters — the
    alternative is dropping turns with no trace of them at all.

    Args:
        turns: Turns being retired, oldest first.
        current: The existing rolling summary.
        max_chars: Hard cap on the returned text.

    Returns:
        The merged summary text.
    """
    lines: list[str] = []
    if current.strip():
        lines.append(current.strip())
    for turn in turns:
        text = " ".join(turn.content.split())
        if not text:
            continue
        lines.append(f"{turn.role.value}: {text[:_FALLBACK_TURN_CHARS]}")
    merged = "\n".join(lines)
    if len(merged) <= max_chars:
        return merged
    # Keep the tail: the newest retired turns are the ones later turns refer back to.
    return merged[-max_chars:]


async def summarise_turns(
    turns: Sequence[Message],
    *,
    current_summary: str = "",
    llm: LLMClient | None = None,
    settings: Settings | None = None,
    max_tokens: int | None = None,
) -> str:
    """Fold retired turns into the rolling summary with ``MODEL_FAST``.

    Args:
        turns: Turns being retired, oldest first. Their content must already be
            PII-redacted, because it is sent to the model and traced.
        current_summary: The summary being updated.
        llm: Client to use. Defaults to the process client.
        settings: Active settings. Defaults to the process settings.
        max_tokens: Length budget stated to the model. Defaults to
            ``settings.context_summary_max_tokens``.

    Returns:
        The updated summary. On any failure the deterministic extractive summary is
        returned instead, so compaction never fails a chat turn.
    """
    cfg = settings or get_settings()
    budget = max_tokens or cfg.context_summary_max_tokens
    # The prompt cannot count its own tokens, so the budget is stated as an
    # approximate word count and enforced afterwards by re-measurement.
    max_chars = budget * 4
    if not turns:
        return current_summary

    client = llm
    if client is None:
        try:
            client = get_llm_client(cfg)
        except Exception:
            _log.warning("session_summary_client_unavailable", exc_info=True)
            return _extractive_summary(
                turns, current=current_summary, max_chars=max_chars
            )

    transcript = "\n\n".join(
        f"{turn.role.value}: {turn.content.strip()}" for turn in turns if turn.content
    )
    user_turn = (
        f"<current_summary>\n{current_summary.strip() or '(none)'}\n"
        "</current_summary>\n\n"
        f"<retiring_turns>\n{transcript}\n</retiring_turns>\n\n"
        f"<budget>\nKeep the updated summary under roughly {budget // 2} words.\n"
        "</budget>"
    )
    try:
        response = await client.complete(
            system=SESSION_SUMMARY_SYSTEM,
            messages=[{"role": "user", "content": user_turn}],
            model=cfg.anthropic_model_fast,
            effort=cfg.anthropic_effort_fast,
            max_tokens=budget,
            thinking=False,
            name=SUMMARY_CALL_NAME,
            metadata=prompt_metadata("session_summary"),
        )
    except Exception:
        _log.warning("session_summary_failed", turns=len(turns), exc_info=True)
        return _extractive_summary(turns, current=current_summary, max_chars=max_chars)

    if response.refused or not response.text.strip():
        _log.warning("session_summary_refused", turns=len(turns))
        return _extractive_summary(turns, current=current_summary, max_chars=max_chars)
    return response.text.strip()


class CompactionOutcome(BaseModel):
    """What one compaction pass did, for ``ContextStats`` and for logging."""

    model_config = ConfigDict(extra="forbid")

    retired: list[Message] = Field(
        default_factory=list, description="Turns folded into the summary."
    )
    summary: str = Field(default="", description="The updated rolling summary.")
    summary_tokens: int = Field(
        default=0, ge=0, description="Measured token cost of the summary."
    )
    reason: str = Field(
        default="",
        description=(
            "Why compaction ran: 'ratio' | 'periodic' | 'max_turns' | 'forced' | "
            "'overflow'. Empty when nothing was retired."
        ),
    )

    @property
    def happened(self) -> bool:
        """Whether anything was retired.

        Returns:
            True when at least one turn left the live window.
        """
        return bool(self.retired)


# ------------------------------------------------------------------- the façade
class ShortTermMemory:
    """Session-window store, rolling summary, pinning and token accounting."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        store: SessionStore | None = None,
        counter: TokenCounter | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        """Initialise the façade.

        Args:
            settings: Active settings. Defaults to the process settings.
            store: Backing store. Defaults to Redis with an in-memory fallback.
            counter: Token counter. Defaults to a fresh one bound to ``llm``.
            llm: Client used for summarisation and token counting.
        """
        self._settings = settings or get_settings()
        self._llm = llm
        self._counter = counter or TokenCounter(llm, settings=self._settings)
        self._store = store if store is not None else _build_store(self._settings)

    @property
    def settings(self) -> Settings:
        """Settings this façade was built from.

        Returns:
            The bound settings.
        """
        return self._settings

    @property
    def counter(self) -> TokenCounter:
        """The shared token counter.

        Returns:
            The counter, so context assembly reuses one memo.
        """
        return self._counter

    @property
    def store(self) -> SessionStore:
        """The backing store.

        Returns:
            The store in use.
        """
        return self._store

    # ------------------------------------------------------------------ loading
    async def load(
        self,
        *,
        principal: Principal,
        session_id: str,
        db_session: AsyncSession | None = None,
    ) -> SessionWindow:
        """Load the live window, hydrating from PostgreSQL on a cache miss.

        Args:
            principal: The caller. The tenant is part of the store key and of every
                SQL predicate, so a window is never read across tenants.
            session_id: Session to load.
            db_session: Database session used to re-hydrate on a miss. Omit for a
                purely in-memory conversation (tests, or a brand-new session).

        Returns:
            The live window, empty when the session is new.
        """
        payload = await self._store.load(
            tenant_id=principal.tenant_id, session_id=session_id
        )
        if payload is not None:
            try:
                cached = SessionWindow.from_payload(payload)
            except ValueError:
                _log.warning("session_window_corrupt", session_id=session_id)
            else:
                if cached.tenant_id == principal.tenant_id:
                    return cached
                _log.error(
                    "session_window_tenant_mismatch",
                    session_id=session_id,
                    expected=principal.tenant_id,
                )

        window = SessionWindow.empty(
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            session_id=session_id,
        )
        if db_session is not None:
            await self._hydrate(window, db_session=db_session)
            await self.save(window)
        return window

    async def _hydrate(
        self, window: SessionWindow, *, db_session: AsyncSession
    ) -> None:
        """Fill an empty window from the durable store.

        Args:
            window: Window to populate in place.
            db_session: Active database session.
        """
        from ragcore.db import repositories as repo

        window.turns = list(
            await repo.list_session_messages(
                db_session,
                tenant_id=window.tenant_id,
                session_id=window.session_id,
                limit=self._settings.context_max_history_turns,
                include_suppressed=False,
                ascending=True,
            )
        )
        row = await repo.get_session_row(
            db_session,
            tenant_id=window.tenant_id,
            user_id=window.user_id,
            session_id=window.session_id,
        )
        if row is not None:
            window.rolling_summary = row.rolling_summary or ""
            window.summary_tokens = int(row.summary_tokens or 0)
            window.compaction_events = int(row.compaction_events or 0)

        unmeasured = [
            index
            for index, turn in enumerate(window.turns)
            if turn.token_count <= 0 and turn.content.strip()
        ]
        if unmeasured:
            counts = await self._counter.count_many(
                [window.turns[index].content for index in unmeasured]
            )
            for index, tokens in zip(unmeasured, counts, strict=True):
                window.turns[index] = window.turns[index].model_copy(
                    update={"token_count": tokens}
                )
        if window.rolling_summary and not window.summary_tokens:
            window.summary_tokens = await self._counter.count_text(
                window.rolling_summary
            )

    async def save(self, window: SessionWindow) -> None:
        """Persist the window to the session store.

        Args:
            window: Window to write.
        """
        await self._store.save(
            tenant_id=window.tenant_id,
            session_id=window.session_id,
            payload=window.to_payload(),
            ttl_seconds=self._settings.redis_ttl_seconds,
        )

    async def clear(self, *, principal: Principal, session_id: str) -> None:
        """Drop a session's cached window.

        Args:
            principal: The caller; supplies the tenant scope.
            session_id: Session to clear.
        """
        await self._store.delete(tenant_id=principal.tenant_id, session_id=session_id)

    # ------------------------------------------------------------------- writing
    async def record_turn(
        self,
        window: SessionWindow,
        *,
        role: Role,
        content: str,
        pii_redacted: bool,
        message_id: str,
        pinned: bool = False,
        citations: Sequence[Any] | None = None,
        tool_calls: Sequence[Any] | None = None,
        created_at: datetime | None = None,
        persist: bool = True,
    ) -> Message:
        """Append a turn, measuring its token cost.

        Args:
            window: Window to append to.
            role: Author of the turn.
            content: Turn text, already PII-redacted.
            pii_redacted: Must be True. This is an assertion that the redaction pass
                ran, not a formatting hint — the session store is persistence.
            message_id: Stable message id, shared with the ``chat_messages`` row.
            pinned: Mark the turn as never-suppressible.
            citations: Verified citations for an assistant turn.
            tool_calls: Tool invocations made while producing the turn.
            created_at: Explicit creation time; defaults to now.
            persist: Write the window back to the store immediately.

        Returns:
            The appended :class:`~ragcore.models.chat.Message`.

        Raises:
            ValueError: If ``pii_redacted`` is False.
        """
        if not pii_redacted:
            msg = (
                "refusing to store turn content that has not passed PII redaction: "
                "pass pii_redacted=True only after running the redaction pass"
            )
            raise ValueError(msg)

        tokens = await self._counter.count_text(content, role=role.value)
        message = Message(
            message_id=message_id,
            session_id=window.session_id,
            role=role,
            content=content,
            citations=list(citations or []),
            tool_calls=list(tool_calls or []),
            token_count=tokens,
            created_at=created_at or _utcnow(),
            pinned=pinned,
        )
        window.append(message)
        if not self._settings.context_compaction_enabled:
            window.trim(self._settings.context_max_history_turns)
        if persist:
            await self.save(window)
        return message

    async def pin(
        self, window: SessionWindow, message_id: str, *, pinned: bool = True
    ) -> bool:
        """Pin or unpin a turn and persist the change.

        Args:
            window: Window to change.
            message_id: Turn to pin.
            pinned: True to pin, False to unpin.

        Returns:
            True when the turn was found.
        """
        changed = window.pin(message_id, pinned=pinned)
        if changed:
            await self.save(window)
        return changed

    # --------------------------------------------------------------- compaction
    def select_for_suppression(
        self, window: SessionWindow, *, keep_live: int | None = None
    ) -> list[Message]:
        """Choose which turns a compaction pass would retire.

        Exposed so ``POST /sessions/{id}/compact`` can report a dry run without
        mutating anything.

        Args:
            window: Window to inspect.
            keep_live: Newest turns to protect. Defaults to
                ``settings.context_min_live_turns``.

        Returns:
            The retirable turns, oldest first.
        """
        protect = (
            self._settings.context_min_live_turns if keep_live is None else keep_live
        )
        return window.suppressible(keep_live=protect)

    async def compact(
        self,
        window: SessionWindow,
        *,
        keep_live: int | None = None,
        reason: str = "periodic",
        message_ids: Sequence[str] | None = None,
        persist: bool = True,
    ) -> CompactionOutcome:
        """Fold the oldest non-pinned turns into the rolling summary.

        Args:
            window: Window to compact.
            keep_live: Newest turns to protect. Defaults to
                ``settings.context_min_live_turns``.
            reason: Why compaction ran, recorded on the outcome.
            message_ids: Explicit turns to retire. When omitted the oldest
                non-pinned turns beyond ``keep_live`` are chosen.
            persist: Write the window back to the store afterwards.

        Returns:
            What happened, including the updated summary and its measured cost.
        """
        if message_ids is None:
            victims = self.select_for_suppression(window, keep_live=keep_live)
        else:
            wanted = set(message_ids)
            victims = [
                turn
                for turn in window.live_turns
                if turn.message_id in wanted and not turn.pinned
            ]
        if not victims:
            return CompactionOutcome(
                summary=window.rolling_summary,
                summary_tokens=window.summary_tokens,
            )

        if self._settings.context_compaction_enabled:
            summary = await summarise_turns(
                victims,
                current_summary=window.rolling_summary,
                llm=self._llm,
                settings=self._settings,
            )
        else:
            # Compaction disabled still suppresses: the turns leave the window and
            # remain in PostgreSQL with suppressed=True, but no summary is produced.
            summary = window.rolling_summary
        summary_tokens = await self._counter.count_text(summary) if summary else 0
        retired = window.suppress(
            [turn.message_id for turn in victims],
            summary=summary,
            summary_tokens=summary_tokens,
        )
        if persist:
            await self.save(window)
        _log.info(
            "context_compacted",
            session_id=window.session_id,
            reason=reason,
            retired=len(retired),
            summary_tokens=summary_tokens,
            live_turns=len(window.live_turns),
        )
        return CompactionOutcome(
            retired=retired,
            summary=summary,
            summary_tokens=summary_tokens,
            reason=reason if retired else "",
        )

    async def persist_compaction(
        self,
        window: SessionWindow,
        outcome: CompactionOutcome,
        *,
        db_session: AsyncSession,
    ) -> int:
        """Mirror a compaction into PostgreSQL.

        Args:
            window: The compacted window.
            outcome: What :meth:`compact` did.
            db_session: Active database session. The caller commits.

        Returns:
            The number of rows marked suppressed.
        """
        if not outcome.retired:
            return 0
        from ragcore.db import repositories as repo

        return await repo.suppress_messages(
            db_session,
            tenant_id=window.tenant_id,
            session_id=window.session_id,
            message_ids=[turn.message_id for turn in outcome.retired],
            rolling_summary=outcome.summary,
            summary_tokens=outcome.summary_tokens,
        )

    async def aclose(self) -> None:
        """Release the backing store's transport."""
        await self._store.close()


def _build_store(settings: Settings) -> SessionStore:
    """Select the session store for this process.

    Args:
        settings: Active settings.

    Returns:
        A Redis store when Redis is enabled and importable, otherwise the in-memory
        fallback. The choice is logged once, so a production deployment silently
        running on the fallback is visible rather than mysterious.
    """
    prefix = str(settings.redis_session_prefix)
    if not settings.redis_enabled:
        _log.info("session_store_selected", store="memory", reason="redis_disabled")
        return InMemorySessionStore(prefix=prefix)
    store = RedisSessionStore(settings.redis_url, prefix=prefix)
    if not store.probe():
        _log.warning(
            "session_store_selected", store="memory", reason="redis_import_failed"
        )
        return InMemorySessionStore(prefix=prefix)
    _log.info("session_store_selected", store="redis")
    return store


_SHORT_TERM: dict[str, ShortTermMemory] = {}


def get_short_term_memory(settings: Settings | None = None) -> ShortTermMemory:
    """Return the process-wide short-term memory façade.

    ``Settings`` is a pydantic model and therefore unhashable, so the cache key is
    the Redis URL plus the session prefix — the only fields that change which store
    gets built.

    Args:
        settings: Active settings. Defaults to the process settings.

    Returns:
        The cached façade.
    """
    cfg = settings or get_settings()
    key = f"{cfg.redis_url}|{cfg.redis_session_prefix}"
    existing = _SHORT_TERM.get(key)
    if existing is None:
        existing = ShortTermMemory(settings=cfg)
        _SHORT_TERM[key] = existing
    return existing


def reset_short_term_memory() -> None:
    """Drop the cached façades. Test helper."""
    _SHORT_TERM.clear()
