"""The binding between the golden set and the real pipeline (requirement #8).

`docs/CONTRACTS.md` Addendum E documents ``eval_pipeline_target`` as
``"app.rag.orchestrator:run_turn"``. That callable has never existed: the module
exposes :class:`app.rag.orchestrator.Orchestrator` with :meth:`~Orchestrator.run`
and :meth:`~Orchestrator.stream`, both of which take a
:class:`~ragcore.models.chat.ChatRequest` rather than a bare question. Resolving the
documented target therefore raised ``EvalHarnessError: ... is not callable`` and the
whole validation layer — including the ten ``acl_negative`` items that are the
end-to-end proof of multi-tenant isolation — could not run at all. Addendum P also
expects ``eval.harness.run_evaluation`` for ``POST /eval/runs``, and this module did
not exist, so that endpoint answered 503 unconditionally.

This module closes both gaps:

* :class:`EvalHarness` builds an :class:`~app.rag.orchestrator.Orchestrator`, turns a
  persona into a :class:`~ragcore.models.acl.Principal`, runs one golden item through
  the real thirteen stages and projects the finished
  :class:`~app.rag.orchestrator.ChatTurn` onto the harness's
  :class:`~eval.run_eval.TurnOutcome` — answer, retrieved chunk ids, tool invocations
  and usage.
* :func:`run_turn` is the module-level coroutine :data:`DEFAULT_PIPELINE_TARGET`
  names, so ``module:attribute`` resolution lands on a callable that exists and
  binds structurally the way Addendum E requires.
* :func:`run_evaluation` is the entry point ``POST /eval/runs`` imports.
* :func:`offline_environment` installs the fakes that let the whole golden set — and
  therefore :mod:`eval.ci_gate` — execute with no Anthropic key, no Qdrant and no
  FastEmbed download, while still driving the *real* orchestrator, retriever,
  guardrails and ACL filters.

**Configuring the target.** ``eval_pipeline_target`` belongs on
:class:`ragcore.settings.Settings`; until it is declared there,
``Settings(extra="ignore")`` silently drops ``RAG_EVAL_PIPELINE_TARGET`` and
:data:`eval.EVAL_SETTING_DEFAULTS` still carries the stale default. So
:func:`resolve_pipeline_target` reads, in order: an explicit argument, the
:data:`PIPELINE_TARGET_ENV` environment variable, the ``eval_pipeline_target``
settings field *if it exists*, then :data:`DEFAULT_PIPELINE_TARGET`. The settings
field wins the moment it is declared; nothing here needs to change then.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import math
import os
import re
import sys
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from eval.run_eval import (
    EvalHarnessError,
    PipelineRunner,
    TurnOutcome,
    load_personas,
    new_run_id,
    run_eval,
)
from ragcore.llm.client import LLMResponse, LLMUsage, StreamEvent, StreamEventType
from ragcore.logging import get_logger
from ragcore.models.acl import Principal
from ragcore.models.chat import ChatRequest, GuardrailAction
from ragcore.models.eval import EvalRun, GoldenItem
from ragcore.settings import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.rag.orchestrator import ChatTurn, Orchestrator
    from ragcore.models.chunk import ChunkPayload
    from ragcore.models.retrieval import MetadataFilter

__all__ = [
    "DEFAULT_PIPELINE_TARGET",
    "PIPELINE_TARGET_ENV",
    "EvalHarness",
    "FakeEmbeddingProvider",
    "FakeQdrantClient",
    "ScriptedLLM",
    "demo_chunk_payloads",
    "new_run_id",
    "offline_environment",
    "offline_settings",
    "outcome_from_turn",
    "reset_harness_cache",
    "resolve_pipeline_target",
    "run_evaluation",
    "run_turn",
]

_log = get_logger(__name__)

#: Environment variable that overrides the pipeline target. Read here rather than
#: through :func:`eval.eval_setting` because ``eval_pipeline_target`` is not a
#: ``Settings`` field yet and ``Settings`` is ``extra="ignore"``, so the variable
#: would otherwise be discarded without a word.
PIPELINE_TARGET_ENV = "RAG_EVAL_PIPELINE_TARGET"

#: ``module:attribute`` of the coroutine one golden item is run through. Points at
#: :func:`run_turn` below, which drives the real orchestrator.
DEFAULT_PIPELINE_TARGET = "eval.harness:run_turn"

#: Repository root, used to locate the demo fixture script.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_pipeline_target(
    *, target: str | None = None, settings: Settings | None = None
) -> str:
    """Resolve the ``module:attribute`` a golden item is executed through.

    Args:
        target: Explicit override, normally a constructor argument. Wins outright.
        settings: Resolved settings. ``eval_pipeline_target`` is consulted only when
            the field actually exists, so promoting it to
            :class:`ragcore.settings.Settings` starts working with no code change.

    Returns:
        The target string, defaulting to :data:`DEFAULT_PIPELINE_TARGET`.
    """
    explicit = (target or "").strip()
    if explicit:
        return explicit
    from_env = os.environ.get(PIPELINE_TARGET_ENV, "").strip()
    if from_env:
        return from_env
    cfg = settings or get_settings()
    from_settings = str(getattr(cfg, "eval_pipeline_target", "") or "").strip()
    if from_settings:
        return from_settings
    return DEFAULT_PIPELINE_TARGET


def outcome_from_turn(turn: ChatTurn, *, settings: Settings) -> TurnOutcome:
    """Project a finished :class:`~app.rag.orchestrator.ChatTurn` onto a scoring view.

    Read explicitly rather than through :func:`eval.run_eval.coerce_outcome`'s
    structural adapter, because the orchestrator reports its spend as a summed
    mapping on :attr:`ChatTurn.usage` rather than as
    :class:`~ragcore.llm.LLMUsage` records, and the structural reader would score
    every item at zero cost.

    Args:
        turn: The finished turn.
        settings: Resolved settings, used to price the reconstructed usage record.

    Returns:
        The normalised :class:`~eval.run_eval.TurnOutcome`. ``answer`` is the
        post-stage-12 text, i.e. what the user would have seen.
    """
    usage = dict(turn.usage or {})
    records: list[LLMUsage] = []
    if usage:
        records.append(
            LLMUsage(
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                cache_read_tokens=int(usage.get("cache_read_tokens", 0)),
                cache_write_tokens=int(usage.get("cache_write_tokens", 0)),
                model=str(usage.get("model") or settings.anthropic_model_main),
                settings=settings,
            )
        )
    # `ChatTurn` carries no `refused` flag, so it is derived from the guardrail
    # decisions the turn published: a blocked input and an out-of-domain refusal both
    # emit a `block` event. `refusal_correct` ORs this with its own textual check.
    refused = any(
        str(event.action) == GuardrailAction.BLOCK.value for event in turn.guardrails
    )
    return TurnOutcome(
        answer=turn.message.content,
        chunks=list(turn.retrieval.chunks) if turn.retrieval is not None else [],
        citations=list(turn.message.citations),
        tools_invoked=[call.tool_name for call in turn.message.tool_calls],
        usages=records,
        cost_usd=float(usage["cost_usd"]) if "cost_usd" in usage else None,
        refused=refused,
        trace_id=turn.trace_id,
        guardrails=list(turn.guardrails),
        raw=turn,
    )


class EvalHarness:
    """Runs one golden item through the real pipeline as a persona.

    Implements :class:`~eval.run_eval.PipelineRunner`, so it can be handed straight
    to :func:`eval.run_eval.run_eval` as ``runner=``.

    Attributes:
        settings: Resolved settings the orchestrator was built from.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        orchestrator: Orchestrator | None = None,
        personas: Mapping[str, Principal] | None = None,
        personas_path: str | Path | None = None,
    ) -> None:
        """Build a harness.

        Args:
            settings: Resolved settings.
            orchestrator: Pipeline to drive. Defaults to the process-wide one, which
                is the same object ``POST /chat`` uses.
            personas: Persona key to principal. Loaded from ``eval_personas_path``
                on first use when omitted.
            personas_path: Persona file to load when ``personas`` is not given.
        """
        self.settings = settings or get_settings()
        self._orchestrator = orchestrator
        self._personas: dict[str, Principal] | None = (
            dict(personas) if personas is not None else None
        )
        self._personas_path = personas_path

    @property
    def orchestrator(self) -> Orchestrator:
        """The pipeline this harness drives.

        Returns:
            The injected orchestrator, or the process-wide one.

        Raises:
            EvalHarnessError: If ``services/api`` is not importable, which makes the
                harness unable to exercise anything real.
        """
        if self._orchestrator is None:
            try:
                from app.rag.orchestrator import get_orchestrator
            except ImportError as exc:  # pragma: no cover - broken install only
                msg = (
                    "app.rag.orchestrator is required to run the golden set; install "
                    "the workspace with `uv sync --all-packages`"
                )
                raise EvalHarnessError(msg) from exc
            self._orchestrator = get_orchestrator(self.settings)
        return self._orchestrator

    @property
    def personas(self) -> dict[str, Principal]:
        """Persona key to principal.

        Returns:
            The persona map, loaded on first use.
        """
        if self._personas is None:
            self._personas = load_personas(self._personas_path, settings=self.settings)
        return self._personas

    def principal_for(self, persona: str) -> Principal:
        """Resolve a persona key to the principal the turn runs as.

        Args:
            persona: A key of the persona file, i.e. ``GoldenItem.as_user``.

        Returns:
            The principal — the same shape the Entra token resolver produces, so the
            ACL filter cannot tell an evaluated turn from a signed-in one.

        Raises:
            EvalHarnessError: If the persona file does not define that key.
        """
        principal = self.personas.get(persona)
        if principal is None:
            known = ", ".join(sorted(self.personas)) or "none"
            msg = f"persona {persona!r} is not defined; known personas: {known}"
            raise EvalHarnessError(msg)
        return principal

    async def run_question(
        self,
        question: str,
        *,
        principal: Principal,
        session_id: str | None = None,
        allow_tools: bool = True,
        filters: MetadataFilter | None = None,
    ) -> TurnOutcome:
        """Run one question through every pipeline stage.

        Args:
            question: The user turn.
            principal: The caller to run as.
            session_id: Existing session to continue. A fresh session per item is the
                default, so one item cannot contaminate the next through the window.
            allow_tools: Permit the tool loop.
            filters: Facet filter applied on top of the ACL filter.

        Returns:
            The normalised :class:`~eval.run_eval.TurnOutcome`.
        """
        request = ChatRequest(
            message=question,
            session_id=session_id,
            filters=filters,
            allow_tools=allow_tools,
            stream=False,
        )
        turn = await self.orchestrator.run(request, principal=principal)
        return outcome_from_turn(turn, settings=self.settings)

    async def run_item(
        self, item: GoldenItem, *, principal: Principal | None = None
    ) -> TurnOutcome:
        """Run one golden item as its own persona.

        Args:
            item: The golden item.
            principal: Persona override; resolved from ``item.as_user`` when omitted.

        Returns:
            The normalised outcome.
        """
        return await self.run_question(
            item.question, principal=principal or self.principal_for(item.as_user)
        )

    async def __call__(self, *, item: GoldenItem, principal: Principal) -> TurnOutcome:
        """Execute one item — the :class:`~eval.run_eval.PipelineRunner` protocol.

        Args:
            item: The golden item.
            principal: The persona the runner resolved.

        Returns:
            The normalised outcome.
        """
        return await self.run_item(item, principal=principal)


#: Harnesses cached per settings fingerprint, so ``run_turn`` does not rebuild an
#: orchestrator (and its model client, registry and session store) for every item.
_HARNESSES: dict[str, EvalHarness] = {}


def _harness_key(settings: Settings) -> str:
    """Cache key for :func:`get_harness`.

    Args:
        settings: Resolved settings.

    Returns:
        The fields that change which collaborators an orchestrator builds.
    """
    return "|".join(
        str(part)
        for part in (
            settings.anthropic_model_main,
            settings.qdrant_url,
            settings.database_url,
            settings.redis_url,
            settings.tool_registry_path,
        )
    )


def get_harness(settings: Settings | None = None) -> EvalHarness:
    """Return the shared harness for a settings instance.

    Args:
        settings: Resolved settings.

    Returns:
        A cached :class:`EvalHarness`.
    """
    cfg = settings or get_settings()
    key = _harness_key(cfg)
    harness = _HARNESSES.get(key)
    if harness is None:
        harness = EvalHarness(settings=cfg)
        _HARNESSES[key] = harness
    return harness


def reset_harness_cache() -> None:
    """Drop every cached harness. Used by the offline fixture and by tests."""
    _HARNESSES.clear()


async def run_turn(
    *,
    message: str,
    principal: Principal,
    settings: Settings | None = None,
    session_id: str | None = None,
    allow_tools: bool = True,
    filters: MetadataFilter | None = None,
) -> TurnOutcome:
    """Run one question through the real pipeline.

    This is the callable :data:`DEFAULT_PIPELINE_TARGET` names. Its signature is the
    structural contract Addendum E specifies: a question under ``message``, a
    principal under ``principal``, and ``settings`` / ``session_id`` / ``allow_tools``
    / ``filters`` filled in by the runner when it has an opinion.

    Args:
        message: The user turn.
        principal: The caller to run as.
        settings: Resolved settings.
        session_id: Existing session to continue.
        allow_tools: Permit the tool loop.
        filters: Facet filter applied on top of the ACL filter.

    Returns:
        The normalised :class:`~eval.run_eval.TurnOutcome`.
    """
    harness = get_harness(settings)
    return await harness.run_question(
        message,
        principal=principal,
        session_id=session_id,
        allow_tools=allow_tools,
        filters=filters,
    )


async def run_evaluation(
    *,
    tenant_id: str,
    golden_set_path: str | None = None,
    sample_size: int | None = None,
    notes: str | None = None,
    settings: Settings | None = None,
    runner: PipelineRunner | None = None,
    run_id: str | None = None,
) -> EvalRun:
    """Run the golden set and return the contract's :class:`EvalRun`.

    The entry point ``POST /api/v1/eval/runs`` imports, as pinned by Addendum P of
    `docs/CONTRACTS.md`.

    Args:
        tenant_id: Tenant the run row is filed under.
        golden_set_path: Golden file; defaults to ``eval_golden_path``.
        sample_size: Evaluate at most this many items.
        notes: Free-form note stored with the run.
        settings: Resolved settings.
        runner: Pipeline runner override, mainly for tests.
        run_id: Pre-allocated run id, so a caller that started the run in the
            background can already name it.

    Returns:
        The completed run, gate verdict recorded on ``gate_passed``.
    """
    artifacts = await run_eval(
        golden_path=golden_set_path,
        limit=sample_size,
        tenant_id=tenant_id,
        notes=notes,
        settings=settings,
        runner=runner,
        run_id=run_id,
    )
    return artifacts.run


# --------------------------------------------------------------------------
# Offline fixture: the whole golden set, no external service.
#
# The gate has to be runnable — in CI, in a container build, on a laptop with no
# Anthropic key — or it is not a gate. What is faked here is exactly what leaves the
# process: the Anthropic API, Qdrant, and the two FastEmbed model downloads. The
# orchestrator, the retriever, `build_acl_filter`, the in-process ACL mirror, the
# guardrails, the citation verifier and the metrics are all the real ones, which is
# what makes an `acl_negative` result mean anything.
#
# What it proves and what it does not: `acl_leak` depends on the ACL filter, so it is
# a real measurement here. The quality metrics depend on the embedder, so under a
# lexical fixture they are fixture artefacts — read them as "the harness scored
# something", never as "retrieval is this good".
# --------------------------------------------------------------------------

#: Dimensionality of the fake dense vector. Small enough to be free, large enough
#: that hashing collisions do not dominate the similarity.
_FAKE_DIM = 512

#: Tokeniser shared by the fake embedder and the scripted model.
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

#: Reciprocal-rank-fusion constant Qdrant uses, mirrored so a branch's contribution
#: falls off with rank the way the real fusion's does.
_RRF_K = 60

#: ``retrieval_fusion_score_scale``'s default. The fake multiplies its cosine by this
#: so the retriever's ``fusion_score / scale`` projection recovers the cosine.
_FUSION_SCORE_SCALE = 0.05


def _tokens(text: str) -> list[str]:
    """Split text into comparable lower-case tokens.

    Args:
        text: Any text.

    Returns:
        The tokens, in order.
    """
    return [match.group(0).casefold() for match in _WORD_RE.finditer(text)]


def _bucket(token: str) -> int:
    """Hash a token onto a vector dimension.

    Args:
        token: A lower-case token.

    Returns:
        A dimension index in ``[0, _FAKE_DIM)``.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % _FAKE_DIM


class FakeEmbeddingProvider:
    """Deterministic TF-IDF embedder standing in for the bge-m3 provider.

    Implements :class:`ragcore.embeddings.EmbeddingProvider`. Cosine over these
    vectors is lexical overlap weighted by inverse document frequency, which
    separates "the corpus covers this" from "it does not" well enough for the gate
    to mean something, and costs nothing to compute.

    Attributes:
        idf: Token to inverse-document-frequency weight, fitted on the corpus.
    """

    def __init__(self, corpus: Sequence[str] = ()) -> None:
        """Fit the weights.

        Args:
            corpus: Documents to derive the IDF weights from. An empty corpus gives
                every token weight 1.0, i.e. plain lexical overlap.
        """
        documents = [set(_tokens(text)) for text in corpus]
        total = len(documents)
        self.idf: dict[str, float] = {}
        for document in documents:
            for token in document:
                self.idf[token] = self.idf.get(token, 0.0) + 1.0
        for token, count in self.idf.items():
            self.idf[token] = math.log((total + 1.0) / (count + 1.0)) + 1.0

    def vector(self, text: str) -> list[float]:
        """Embed one text.

        Args:
            text: The text to embed.

        Returns:
            An L2-normalised dense vector of length :data:`_FAKE_DIM`.
        """
        vector = [0.0] * _FAKE_DIM
        for token in _tokens(text):
            vector[_bucket(token)] += self.idf.get(token, 1.0)
        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            return vector
        return [component / norm for component in vector]

    def _sparse(self, text: str) -> Any:
        """Build the sparse half of an embedding.

        Args:
            text: The text to embed.

        Returns:
            A :class:`ragcore.embeddings.SparseVec` over the same hashed buckets.
        """
        from ragcore.embeddings import SparseVec

        weights: dict[int, float] = {}
        for token in _tokens(text):
            index = _bucket(token)
            weights[index] = weights.get(index, 0.0) + self.idf.get(token, 1.0)
        ordered = sorted(weights)
        return SparseVec(indices=ordered, values=[weights[key] for key in ordered])

    async def embed_documents(self, texts: Sequence[str]) -> list[Any]:
        """Embed a batch of documents.

        Args:
            texts: Texts to embed.

        Returns:
            One :class:`ragcore.embeddings.Embedded` per input.
        """
        from ragcore.embeddings import Embedded

        return [
            Embedded(dense=self.vector(text), sparse=self._sparse(text))
            for text in texts
        ]

    async def embed_query(self, text: str) -> Any:
        """Embed one query.

        Args:
            text: The query.

        Returns:
            The :class:`ragcore.embeddings.Embedded` pair.
        """
        return (await self.embed_documents([text]))[0]

    async def embed_dense(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch, dense half only.

        Args:
            texts: Texts to embed.

        Returns:
            One dense vector per input.
        """
        return [self.vector(text) for text in texts]

    def cosine(self, left: str, right: str) -> float:
        """Cosine similarity between two texts.

        Args:
            left: First text.
            right: Second text.

        Returns:
            The similarity, in ``[0, 1]``.
        """
        first, second = self.vector(left), self.vector(right)
        return max(0.0, sum(a * b for a, b in zip(first, second, strict=True)))


class FakeReranker:
    """Calibrated cross-encoder stand-in.

    The offline fixture cannot download a 600 MB cross-encoder, and
    ``NoopReranker`` is not a substitute: it hands every candidate a synthetic
    rank-proxy score, so :func:`app.rag.guardrails.ood.relevance_signals` reads
    ``max_score == 1.0`` and stage 6 can never fire — with the reranker off, the
    out-of-domain gate is effectively off too.

    Scores are emitted as **probabilities**, not raw logits, because that is the
    scale stage 6 is written against: ``guardrail_ood_min_score`` (0.35) and
    ``guardrail_ood_mean_score_min`` (0.2) are both ``[0, 1]`` numbers compared
    directly against :attr:`RetrievedChunk.rerank_score`.

    The mapping is a stated fixture calibration, not a measurement: lexical overlap
    is not semantic relevance, so the *refusal boundary* under this fixture lands
    where lexical overlap puts it. What the fixture proves is that stage 6 runs and
    that the ACL metrics hold; it does not prove retrieval quality.

    Attributes:
        embedder: Provider supplying the similarity.
    """

    #: Similarity the fixture treats as "an even bet that this passage answers the
    #: question", i.e. where the emitted probability crosses 0.5.
    _CENTRE = 0.30

    #: Logistic gain. Cross-encoders are confident; a flatter gain would squash
    #: every candidate into the middle of the scale and nothing would ever refuse.
    _GAIN = 12.0

    def __init__(self, embedder: FakeEmbeddingProvider) -> None:
        """Bind the reranker to the fixture's embedder.

        Args:
            embedder: Provider supplying the similarity.
        """
        self.embedder = embedder

    def probability(self, query: str, document: str) -> float:
        """Score one (query, document) pair.

        Args:
            query: The search query.
            document: The candidate text.

        Returns:
            The calibrated relevance probability, in ``(0, 1)``.
        """
        logit = self._GAIN * (self.embedder.cosine(query, document) - self._CENTRE)
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit))))

    async def rerank(
        self, query: str, documents: Sequence[str], top_n: int
    ) -> list[Any]:
        """Score candidates against the query.

        Args:
            query: The search query.
            documents: Candidate texts, in retrieval order.
            top_n: Maximum results to return.

        Returns:
            Up to ``top_n`` :class:`ragcore.rerank.RerankResult` values, best first.
        """
        from ragcore.rerank import RerankResult

        scored = [
            RerankResult(index=index, score=self.probability(query, document))
            for index, document in enumerate(documents)
        ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_n]


def _condition_matches(
    condition: Any, payload: Mapping[str, Any], point_id: Any
) -> bool:
    """Evaluate one Qdrant condition against a stored payload.

    Only the shapes :mod:`ragcore.vectorstore.filters` actually produces are
    implemented — ``MatchValue``, ``MatchAny``, ``Range``, ``IsEmpty``, ``HasId`` and
    nested filters. An unrecognised condition raises rather than being ignored: a
    filter clause the fake silently drops is a filter clause the gate stops testing.

    Args:
        condition: A ``qdrant_client.models`` condition.
        payload: The point's stored payload.
        point_id: The point's id, for ``HasIdCondition``.

    Returns:
        Whether the condition holds.

    Raises:
        EvalHarnessError: If the condition kind is not implemented.
    """
    from qdrant_client import models as qm

    if isinstance(condition, qm.Filter):
        return _filter_matches(condition, payload, point_id)
    if isinstance(condition, qm.HasIdCondition):
        return point_id in set(condition.has_id)
    if isinstance(condition, qm.IsEmptyCondition):
        value = payload.get(condition.is_empty.key)
        return value is None or value == [] or value == ""
    if isinstance(condition, qm.FieldCondition):
        value = payload.get(condition.key)
        values = value if isinstance(value, list) else [value]
        if condition.match is not None:
            match = condition.match
            if isinstance(match, qm.MatchValue):
                return match.value in values
            present = {item for item in values if item is not None}
            if isinstance(match, qm.MatchAny):
                return bool(set(match.any) & present)
            if isinstance(match, qm.MatchExcept):
                excluded = getattr(match, "except_", None) or []
                return not (set(excluded) & present)
        if condition.range is not None and isinstance(condition.range, qm.Range):
            number = value if isinstance(value, (int, float)) else None
            if number is None:
                return False
            bounds = condition.range
            if bounds.lte is not None and number > bounds.lte:
                return False
            if bounds.lt is not None and number >= bounds.lt:
                return False
            if bounds.gte is not None and number < bounds.gte:
                return False
            return not (bounds.gt is not None and number <= bounds.gt)
    msg = f"the offline Qdrant fake does not implement {type(condition).__name__}"
    raise EvalHarnessError(msg)


def _filter_matches(qfilter: Any, payload: Mapping[str, Any], point_id: Any) -> bool:
    """Evaluate a composed Qdrant filter against a stored payload.

    Args:
        qfilter: The filter, normally from
            :func:`ragcore.vectorstore.filters.build_acl_filter`.
        payload: The point's stored payload.
        point_id: The point's id.

    Returns:
        Whether the point survives the filter, applying ``must`` / ``must_not`` /
        ``min_should`` exactly as Qdrant does.
    """
    if qfilter is None:
        return True

    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        return list(value) if isinstance(value, list) else [value]

    for condition in _as_list(qfilter.must):
        if not _condition_matches(condition, payload, point_id):
            return False
    for condition in _as_list(qfilter.must_not):
        if _condition_matches(condition, payload, point_id):
            return False

    minimum = qfilter.min_should
    should = _as_list(qfilter.should)
    if minimum is not None:
        matched = sum(
            1
            for condition in minimum.conditions
            if _condition_matches(condition, payload, point_id)
        )
        return matched >= minimum.min_count
    if should:
        return any(
            _condition_matches(condition, payload, point_id) for condition in should
        )
    return True


@dataclass(frozen=True, slots=True)
class _QueryResponse:
    """What ``AsyncQdrantClient.query_points`` returns, as far as callers read it.

    Attributes:
        points: The scored points, best first.
    """

    points: list[Any]


class FakeQdrantClient:
    """An in-memory Qdrant that honours the real filters.

    Enough of the surface for retrieval and the out-of-domain gate:
    ``query_points`` (hybrid prefetch plus fusion, and the vector read-back MMR
    uses) and ``scroll`` (the coverage sample). Every read is filtered through
    :func:`_filter_matches`, so weakening
    :func:`ragcore.vectorstore.filters.build_acl_filter` genuinely lets foreign
    points out of this store — which is the whole point of testing against it.

    Attributes:
        payloads: Stored payloads keyed by deterministic point id.
    """

    def __init__(
        self, chunks: Sequence[ChunkPayload], *, embedder: FakeEmbeddingProvider
    ) -> None:
        """Load the corpus.

        Args:
            chunks: Chunk payloads to serve.
            embedder: Provider used to vectorise them, so a stored vector and a
                query vector live in the same space.
        """
        from ragcore.vectorstore.collections import point_id_for_chunk

        self._embedder = embedder
        self.payloads: dict[Any, dict[str, Any]] = {}
        self._vectors: dict[Any, list[float]] = {}
        for chunk in chunks:
            point_id = point_id_for_chunk(chunk.chunk_id)
            self.payloads[point_id] = chunk.to_qdrant_payload()
            self._vectors[point_id] = embedder.vector(chunk.embed_text)

    def _visible(self, qfilter: Any) -> list[Any]:
        """List the point ids a filter admits.

        Args:
            qfilter: The filter to apply.

        Returns:
            Matching point ids, in insertion order.
        """
        return [
            point_id
            for point_id, payload in self.payloads.items()
            if _filter_matches(qfilter, payload, point_id)
        ]

    @staticmethod
    def _branch_vector(branch: Any) -> list[float]:
        """Project one prefetch branch's query onto the dense space.

        The sparse branch is expressed as a ``SparseVector`` over the same hashed
        buckets the fake embedder uses, so densifying it keeps the two branches in
        one comparable space while still ranking independently.

        Args:
            branch: A ``qdrant_client.models.Prefetch``.

        Returns:
            An L2-normalised vector of length :data:`_FAKE_DIM`.
        """
        if isinstance(branch.query, list):
            return [float(component) for component in branch.query]
        indices = getattr(branch.query, "indices", None) or []
        values = getattr(branch.query, "values", None) or []
        vector = [0.0] * _FAKE_DIM
        for index, value in zip(indices, values, strict=False):
            if 0 <= int(index) < _FAKE_DIM:
                vector[int(index)] = float(value)
        norm = math.sqrt(sum(item * item for item in vector))
        return [item / norm for item in vector] if norm else vector

    def _similarity(self, point_id: Any, query: Sequence[float]) -> float:
        """Cosine similarity between a stored point and a query vector.

        Args:
            point_id: The stored point.
            query: The query vector.

        Returns:
            Cosine similarity, clamped to ``[0, 1]``.
        """
        stored = self._vectors.get(point_id, [])
        if not stored or not query:
            return 0.0
        dot = sum(a * b for a, b in zip(stored, query, strict=False))
        return max(0.0, min(1.0, dot))

    async def query_points(
        self,
        *,
        collection_name: str,
        prefetch: Sequence[Any] | None = None,
        query: Any = None,
        query_filter: Any = None,
        limit: int = 10,
        with_payload: Any = True,
        with_vectors: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Answer a hybrid search or a vector read-back.

        Args:
            collection_name: Ignored; the fake holds one collection.
            prefetch: Per-branch prefetch queries. Each branch is ranked
                independently and the branches are fused with reciprocal rank
                fusion, the way the real ``FusionQuery`` does.
            query: The outer query. Only ``FusionQuery`` and "no query" are used by
                the retriever.
            query_filter: The ACL filter, applied to every branch and to the fused
                set.
            limit: Maximum points to return.
            with_payload: Ignored; the fake always returns the payload.
            with_vectors: Return the stored dense vector, as the MMR read-back asks.
            **kwargs: Ignored query options.

        Returns:
            A :class:`_QueryResponse` holding ``qdrant_client.models.ScoredPoint``
            values, best first.
        """
        del collection_name, query, with_payload, kwargs
        from qdrant_client import models as qm

        from ragcore.vectorstore.collections import DENSE

        admitted = self._visible(query_filter)
        branches = [self._branch_vector(branch) for branch in prefetch or []]
        if branches:
            fused: dict[Any, float] = dict.fromkeys(admitted, 0.0)
            best: dict[Any, float] = dict.fromkeys(admitted, 0.0)
            for branch, vector in zip(prefetch or [], branches, strict=True):
                pool = {
                    point_id: self._similarity(point_id, vector)
                    for point_id in self._visible(branch.filter)
                    if point_id in fused
                }
                ranked = sorted(pool, key=lambda pid: pool[pid], reverse=True)
                for rank, point_id in enumerate(ranked, start=1):
                    if pool[point_id] <= 0.0:
                        continue
                    fused[point_id] += 1.0 / (_RRF_K + rank)
                    best[point_id] = max(best[point_id], pool[point_id])
            # Reciprocal rank fusion decides *which* points come back — a point no
            # branch touched scores zero and is dropped. The score itself is the best
            # branch cosine times `retrieval_fusion_score_scale`, so the retriever's
            # projection of a fusion score onto `[0, 1]` recovers the cosine. A raw
            # RRF value would instead put a chunk nothing in the question touches near
            # the top of that scale purely because it was the only survivor of the ACL
            # filter, and everything downstream reads that projection as relevance.
            scored = {
                point_id: _FUSION_SCORE_SCALE * best[point_id]
                for point_id, value in fused.items()
                if value > 0.0
            }
        else:
            scored = dict.fromkeys(admitted, 1.0)

        order = sorted(scored, key=lambda pid: scored[pid], reverse=True)[:limit]
        points = [
            qm.ScoredPoint(
                id=point_id,
                version=1,
                score=scored[point_id],
                payload=dict(self.payloads[point_id]),
                vector={DENSE: list(self._vectors[point_id])} if with_vectors else None,
            )
            for point_id in order
        ]
        return _QueryResponse(points=points)

    async def scroll(
        self,
        collection_name: str,
        *,
        scroll_filter: Any = None,
        limit: int = 10,
        with_payload: Any = True,
        with_vectors: bool = False,
        **kwargs: Any,
    ) -> tuple[list[Any], None]:
        """Page through the points a filter admits.

        Args:
            collection_name: Ignored.
            scroll_filter: The filter to apply.
            limit: Maximum points to return.
            with_payload: Ignored; the fake always returns the payload.
            with_vectors: Ignored; the coverage sample does not need vectors.
            **kwargs: Ignored scroll options.

        Returns:
            ``(records, next_offset)`` with ``next_offset`` always None.
        """
        del collection_name, with_payload, with_vectors, kwargs
        from qdrant_client import models as qm

        records = [
            qm.Record(id=point_id, payload=dict(self.payloads[point_id]))
            for point_id in self._visible(scroll_filter)[:limit]
        ]
        return records, None


class ScriptedLLM:
    """Deterministic, extractive stand-in for :class:`~ragcore.llm.LLMClient`.

    The answer is the first sentence of the top-numbered source in the prompt, with
    its ``[1]`` marker attached. Two consequences, both deliberate:

    * citations verify for real, because the sentence is quoted verbatim from the
      chunk stage 11 checks it against;
    * anything the retriever should not have returned is quoted straight into the
      answer, canary token included, so a weakened ACL filter surfaces as an
      ``acl_leak`` finding rather than as a silent pass.

    Attributes:
        settings: Settings the client reports, as the real client does.
        calls: One entry per ``stream``/``complete`` invocation.
    """

    #: Sentence terminators used to clip the quoted span.
    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    #: A numbered-source header line inside the rendered ``<sources>`` block.
    _HEADER_RE = re.compile(r"^\[(\d+)\]\s")

    #: The nonce fence that brackets untrusted source text. Matched loosely so the
    #: fixture keeps working if the marker's spelling changes.
    _FENCE_RE = re.compile(r"^<{2,}.*>{2,}$")

    def __init__(self, settings: Settings) -> None:
        """Initialise the scripted client.

        Args:
            settings: Active settings, exposed like the real client's.
        """
        self._settings = settings
        self.calls: list[dict[str, Any]] = []

    @property
    def settings(self) -> Settings:
        """Settings this client was built from.

        Returns:
            The bound settings.
        """
        return self._settings

    def _answer_for(self, kwargs: Mapping[str, Any]) -> str:
        """Compose the answer from the prompt's ``<sources>`` block.

        Args:
            kwargs: The request the orchestrator built.

        Returns:
            A one-sentence answer citing source ``[1]``, or an empty string when the
            prompt carried no sources at all.
        """
        text = ""
        for message in kwargs.get("messages") or []:
            content = message.get("content", "") if isinstance(message, Mapping) else ""
            if isinstance(content, str):
                text += content
            else:
                text += "\n".join(
                    str(block.get("text", ""))
                    for block in content
                    if isinstance(block, Mapping)
                )
        block = re.search(r"<sources[^>]*>(.*?)</sources[^>]*>", text, re.DOTALL)
        if block is None:
            return ""
        lines = [line.strip() for line in block.group(1).splitlines() if line.strip()]
        body: list[str] = []
        inside = False
        for line in lines:
            header = self._HEADER_RE.match(line)
            if header is not None:
                if inside:
                    break
                inside = header.group(1) == "1"
                continue
            if inside and not self._FENCE_RE.match(line):
                body.append(line)
        if not body:
            return ""
        sentence = self._SENTENCE_RE.split(" ".join(body))[0].strip()
        return f"{sentence} [1]"

    def _usage(self, answer: str) -> LLMUsage:
        """Build the usage this client reports.

        Args:
            answer: The generated text.

        Returns:
            A small, non-zero usage record so cost is a real number.
        """
        return LLMUsage(
            input_tokens=200,
            output_tokens=max(1, len(answer) // 4),
            cache_read_tokens=0,
            cache_write_tokens=0,
            model=self._settings.anthropic_model_main,
            settings=self._settings,
        )

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
            A characters/4 estimate, which is only compared against a budget.
        """
        del model
        size = len(str(system or "")) + len(str(messages)) + len(str(tools or ""))
        return max(1, size // 4)

    async def stream(self, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        """Stream the extracted answer.

        Args:
            **kwargs: The request the orchestrator built.

        Yields:
            ``TEXT`` deltas, then ``USAGE``, then one ``DONE``.
        """
        self.calls.append(dict(kwargs))
        answer = self._answer_for(kwargs)
        for word in answer.split(" "):
            yield StreamEvent(type=StreamEventType.TEXT, text=f"{word} ")
        usage = self._usage(answer)
        yield StreamEvent(type=StreamEventType.USAGE, usage=usage)
        yield StreamEvent(
            type=StreamEventType.DONE,
            usage=usage,
            stop_reason="end_turn",
            response=LLMResponse(
                text=answer,
                tool_calls=[],
                stop_reason="end_turn",
                usage=usage,
                refused=False,
                raw=None,
            ),
        )

    async def complete(self, **kwargs: Any) -> LLMResponse:
        """Return the extracted answer without streaming.

        Args:
            **kwargs: The request the orchestrator built.

        Returns:
            The response.
        """
        self.calls.append(dict(kwargs))
        answer = self._answer_for(kwargs)
        usage = self._usage(answer)
        return LLMResponse(
            text=answer,
            tool_calls=[],
            stop_reason="end_turn",
            usage=usage,
            refused=False,
            raw=None,
        )

    async def structured(self, *, schema: type, **kwargs: Any) -> Any:
        """Refuse structured calls.

        Every structured call in the pipeline — query transform, contradiction
        detection, the judges — is documented as best-effort, so refusing here keeps
        the offline run on the degradation paths rather than on a scripted happy
        path that would not exist in production.

        Args:
            schema: The requested model.
            **kwargs: Ignored.

        Raises:
            RuntimeError: Always.
        """
        del schema, kwargs
        msg = "structured output is not available in the offline eval fixture"
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


def _seed_module() -> Any:
    """Import ``scripts/seed_demo_tenant.py``, the demo fixture's source of truth.

    Loaded by path rather than imported: ``scripts/`` is deliberately not a package,
    and duplicating the corpus here would let the golden set and the fixture drift.

    Returns:
        The loaded module.

    Raises:
        EvalHarnessError: If the script cannot be located or executed.
    """
    cached = sys.modules.get("seed_demo_tenant")
    if cached is not None:
        return cached
    path = _REPO_ROOT / "scripts" / "seed_demo_tenant.py"
    spec = importlib.util.spec_from_file_location("seed_demo_tenant", path)
    if spec is None or spec.loader is None:
        msg = f"{path} is not importable; the offline fixture needs the demo corpus"
        raise EvalHarnessError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules["seed_demo_tenant"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop("seed_demo_tenant", None)
        msg = f"{path} could not be loaded: {exc}"
        raise EvalHarnessError(msg) from exc
    return module


def demo_chunk_payloads(settings: Settings | None = None) -> list[ChunkPayload]:
    """Build every demo-fixture chunk, across both tenants.

    The fake store is loaded with *all* of them, restricted by nothing. Whether a
    persona sees a chunk is decided by ``build_acl_filter`` and the retriever's
    in-process mirror, exactly as in production — which is what makes the
    ``acl_negative`` items a real test rather than a restatement of the fixture.

    Args:
        settings: Resolved settings; supplies the chunking options.

    Returns:
        Chunk payloads in document order.
    """
    cfg = settings or get_settings()
    seed = _seed_module()
    chunks: list[ChunkPayload] = []
    for document in seed.DEMO_DOCUMENTS:
        chunks.extend(seed.build_chunks(document, cfg))
    with contextlib.suppress(Exception):  # pii annotation is metadata, not content
        seed.annotate_pii(chunks, cfg)
    return chunks


def offline_settings(base: Settings | None = None) -> Settings:
    """Derive the settings the offline fixture runs under.

    Args:
        base: Settings to derive from. Defaults to the process settings.

    Returns:
        A copy with the subsystems the fixture does not stand up switched off:
        Redis, Langfuse, the tool registry, the cross-encoder, long-term memory and
        the semantic cache. Everything that decides *what a persona may see* is left
        exactly as configured.
    """
    cfg = base or get_settings()
    return cfg.model_copy(
        update={
            "langfuse_enabled": False,
            "redis_enabled": False,
            "memory_enabled": False,
            "memory_cache_enabled": False,
            # Self-hosted MCP discovery dials a child process or an HTTP endpoint;
            # the declarative registry itself stays on, so `tool_available` is true
            # and stage 6 routes a tool-servable question instead of refusing it.
            "tool_mcp_local_enabled": False,
            "eval_persist_results": False,
        }
    )


@contextlib.contextmanager
def _patched(module: Any, name: str, value: Any) -> Iterator[None]:
    """Temporarily rebind a module attribute.

    Args:
        module: The module to patch.
        name: Attribute name.
        value: Replacement value.

    Yields:
        None, with the replacement installed.
    """
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


@contextlib.contextmanager
def offline_environment(*, settings: Settings | None = None) -> Iterator[EvalHarness]:
    """Install the fakes and yield a harness bound to them.

    Faked: the Anthropic API (:class:`ScriptedLLM`), Qdrant
    (:class:`FakeQdrantClient` over the demo fixture) and the two FastEmbed model
    downloads (:class:`FakeEmbeddingProvider`, :class:`FakeReranker`). Real: the
    orchestrator's thirteen stages, the retriever's fusion/dedupe/rerank/MMR/cap
    pipeline, ``build_acl_filter``, the in-process ACL mirror, every guardrail, the
    citation verifier and every metric.

    What this fixture *does* prove is that the harness runs and that the security
    metrics hold: ``acl_leak`` depends on the ACL filter, not on how good the
    embedder is. What it does not prove is answer quality — the fixture's embedder
    is lexical, so ``semantic_similarity`` and the refusal boundary land where
    lexical overlap puts them, not where ``bge-m3`` would.

    Args:
        settings: Settings to derive the fixture from; :func:`offline_settings` is
            applied on top.

    Yields:
        An :class:`EvalHarness` whose orchestrator is wired to the fakes.
    """
    import app.rag.retriever as retriever_module
    import eval.ragas_adapter as ragas_module
    import eval.run_eval as run_eval_module
    import eval.semantic as semantic_module
    import ragcore.vectorstore.client as client_module
    from app.rag.context import ContextAssembler
    from app.rag.memory.long_term import LongTermMemoryStore
    from app.rag.memory.semantic_cache import SemanticCache
    from app.rag.memory.short_term import (
        InMemorySessionStore,
        ShortTermMemory,
        TokenCounter,
    )
    from app.rag.orchestrator import Orchestrator
    from ragcore.observability.langfuse import NoopTracer

    cfg = offline_settings(settings)
    chunks = demo_chunk_payloads(cfg)
    embedder = FakeEmbeddingProvider([chunk.embed_text for chunk in chunks])
    store = FakeQdrantClient(chunks, embedder=embedder)
    llm = ScriptedLLM(cfg)

    async def _client(_settings: Settings | None = None) -> FakeQdrantClient:
        return store

    def _embedder(_settings: Settings | None = None) -> FakeEmbeddingProvider:
        return embedder

    reranker = FakeReranker(embedder)

    def _reranker(_settings: Settings | None = None) -> FakeReranker:
        return reranker

    judge = ragas_module.RagasAdapter(cfg, llm=llm)

    def _adapter(_settings: Settings | None = None) -> Any:
        return judge

    counter = TokenCounter(llm, settings=cfg)
    short_term = ShortTermMemory(
        settings=cfg, store=InMemorySessionStore(), counter=counter, llm=llm
    )
    orchestrator = Orchestrator(
        settings=cfg,
        llm=llm,
        tracer=NoopTracer(),
        short_term=short_term,
        assembler=ContextAssembler(
            settings=cfg, llm=llm, counter=counter, short_term=short_term
        ),
        long_term=LongTermMemoryStore(settings=cfg, llm=llm),
        cache=SemanticCache(settings=cfg),
    )

    with (
        _patched(retriever_module, "get_client", _client),
        _patched(retriever_module, "get_embedding_provider", _embedder),
        _patched(retriever_module, "get_reranker", _reranker),
        _patched(client_module, "get_client", _client),
        _patched(semantic_module, "get_embedding_provider", _embedder),
        _patched(run_eval_module, "get_ragas_adapter", _adapter),
    ):
        _log.info(
            "eval_offline_fixture_ready",
            chunks=len(chunks),
            documents=len({chunk.document_id for chunk in chunks}),
        )
        yield EvalHarness(settings=cfg, orchestrator=orchestrator)
