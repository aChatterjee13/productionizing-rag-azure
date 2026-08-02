"""RAGAS with Claude as the judge, and a native fallback that scores the same five.

Requirement #8 names RAGAS, but a scoring library must never be able to fail a
build on its own. So this module has two backends for the same
:class:`RagasScores`:

* **ragas** — the real package, driven through a :class:`ragas.llms.BaseRagasLLM`
  adapter that routes every judge call through :class:`ragcore.llm.LLMClient`, so
  the judge obeys the platform's LLM_FACTS rules (no sampling parameters, nested
  effort, server-side fallbacks, refusal handling) and lands in Langfuse like every
  other model call. Only the *embedding-free* metrics go this way —
  ``faithfulness``, ``context_precision``, ``context_recall`` — because the
  embedding-dependent ones would drag a second embedding stack into the harness.
* **native** — Claude-judged reimplementations of all five metrics, using
  :mod:`ragcore.embeddings` wherever RAGAS would use its own embeddings.
  ``answer_relevancy`` and ``answer_correctness`` always come from here.

The RAGAS import is isolated in :func:`load_ragas`: a missing extra, a moved symbol
or a changed constructor signature degrades to native with one loud warning instead
of breaking CI.

Every judge call is a structured (JSON-schema) completion — no assistant prefill,
no temperature — and every prompt is a module-level constant carrying
:data:`JUDGE_PROMPT_VERSION`, so a score change can be attributed to a prompt
revision.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from eval import eval_setting
from eval.semantic import embed_texts
from ragcore.embeddings import cosine_similarity
from ragcore.llm import LLMClient, get_llm_client
from ragcore.logging import get_logger
from ragcore.settings import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "JUDGE_PROMPT_VERSION",
    "RAGAS_LLM_ONLY_METRICS",
    "RAGAS_METRIC_NAMES",
    "RagasAdapter",
    "RagasSample",
    "RagasScores",
    "get_ragas_adapter",
    "load_ragas",
    "reset_ragas_cache",
]

_log = get_logger(__name__)

#: Bumped whenever a judge prompt below changes, so a score shift can be blamed on
#: the prompt rather than on the pipeline.
JUDGE_PROMPT_VERSION = "eval-judge-v1"

#: The five metrics this adapter produces, in report order.
RAGAS_METRIC_NAMES: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)

#: Metrics the RAGAS backend may serve. The other two need embeddings, and the
#: harness embeds only through :mod:`ragcore.embeddings`.
RAGAS_LLM_ONLY_METRICS: tuple[str, ...] = (
    "faithfulness",
    "context_precision",
    "context_recall",
)

_JUDGE_RULES = (
    "You are a strict evaluation judge for a retrieval-augmented generation "
    "system. Judge only what the inputs say. Never use outside knowledge, never "
    "guess, and never reward fluent writing. Return the requested JSON object and "
    "nothing else."
)

FAITHFULNESS_SYSTEM = (
    f"{_JUDGE_RULES}\n\n"
    "Task: break the answer into atomic factual claims, then decide for each "
    "claim whether the retrieved context directly supports it. A claim is "
    "supported only if the context states it or entails it without inference from "
    "outside knowledge. Numbers must match exactly. Ignore hedging, greetings and "
    "citation markers such as [1]; they are not claims."
)

RELEVANCY_SYSTEM = (
    f"{_JUDGE_RULES}\n\n"
    "Task: read the answer and write the questions it is a direct answer to. "
    "Write them as standalone questions using only the answer's own wording. Also "
    "flag the answer as noncommittal when it evades, refuses, or says it does not "
    "know."
)

CONTEXT_PRECISION_SYSTEM = (
    f"{_JUDGE_RULES}\n\n"
    "Task: for each numbered context, decide whether it was useful for producing "
    "the reference answer to the question. A context is useful only if it carries "
    "information the reference answer states. Related-but-unused background is "
    "not useful."
)

CONTEXT_RECALL_SYSTEM = (
    f"{_JUDGE_RULES}\n\n"
    "Task: split the reference answer into atomic statements, then decide for "
    "each whether it can be attributed to the retrieved context. Attribution "
    "means the context contains the statement's information, not merely the same "
    "topic."
)

CORRECTNESS_SYSTEM = (
    f"{_JUDGE_RULES}\n\n"
    "Task: compare the answer with the reference answer as sets of factual "
    "statements. Classify each statement as: true_positive (present in both, and "
    "consistent), false_positive (asserted by the answer but absent from or "
    "contradicted by the reference), false_negative (present in the reference but "
    "missing from the answer). A statement with a different number is a "
    "false_positive and a false_negative, never a true_positive."
)


class RagasSample(BaseModel):
    """One item to score: question, answer, retrieved contexts and ground truth."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(default="", description="Golden item id, for tracing.")
    question: str = Field(description="The question that was asked.")
    answer: str = Field(default="", description="The answer the pipeline produced.")
    contexts: list[str] = Field(
        default_factory=list, description="Retrieved chunk texts, in ranked order."
    )
    ground_truth: str = Field(default="", description="The reference answer.")
    semantic_similarity: float | None = Field(
        default=None,
        description=(
            "Embedding cosine against the ground truth, supplied by the caller so "
            "answer correctness does not re-embed what the runner already scored."
        ),
    )


class RagasScores(BaseModel):
    """The five RAGAS metrics, plus which backend produced them."""

    model_config = ConfigDict(extra="forbid")

    faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_relevancy: float | None = Field(default=None, ge=0.0, le=1.0)
    context_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    context_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_correctness: float | None = Field(default=None, ge=0.0, le=1.0)
    backends: dict[str, str] = Field(
        default_factory=dict,
        description="Metric name -> 'ragas' | 'native' | 'unavailable'.",
    )
    degraded: bool = Field(
        default=False, description="True when RAGAS was asked for but not used."
    )
    degraded_reason: str = Field(
        default="", description="Why RAGAS was not used, when it was not."
    )

    @property
    def backend(self) -> str:
        """Summarise which implementation produced these scores.

        Returns:
            ``"ragas"`` when at least one metric came from the package, otherwise
            ``"native"``.
        """
        return "ragas" if "ragas" in self.backends.values() else "native"

    def as_mapping(self) -> dict[str, float]:
        """Flatten the populated metrics.

        Returns:
            Metric name to value, skipping the ones that were not measured.
        """
        return {
            name: float(value)
            for name in RAGAS_METRIC_NAMES
            if (value := getattr(self, name)) is not None
        }


# --------------------------------------------------------------- judge schemas
class _ClaimVerdict(BaseModel):
    """One atomic claim and whether the context supports it."""

    claim: str = Field(description="The atomic claim, quoted from the answer.")
    supported: bool = Field(description="Whether the context supports it.")


class _FaithfulnessVerdict(BaseModel):
    """Every claim in the answer, judged against the context."""

    claims: list[_ClaimVerdict] = Field(default_factory=list)


class _RelevancyVerdict(BaseModel):
    """Questions the answer answers, and whether it dodged."""

    questions: list[str] = Field(default_factory=list)
    noncommittal: bool = Field(default=False)


class _ContextVerdict(BaseModel):
    """Whether one numbered context was useful."""

    index: int = Field(description="1-based index of the context.")
    useful: bool = Field(description="Whether it fed the reference answer.")


class _ContextPrecisionVerdict(BaseModel):
    """Per-context usefulness verdicts."""

    verdicts: list[_ContextVerdict] = Field(default_factory=list)


class _RecallStatement(BaseModel):
    """One reference statement and whether the context carries it."""

    statement: str = Field(description="Atomic statement from the reference.")
    attributed: bool = Field(description="Whether the context supports it.")


class _ContextRecallVerdict(BaseModel):
    """Per-statement attribution verdicts."""

    statements: list[_RecallStatement] = Field(default_factory=list)


class _CorrectnessVerdict(BaseModel):
    """Statement-level comparison of the answer with the reference."""

    true_positive: list[str] = Field(default_factory=list)
    false_positive: list[str] = Field(default_factory=list)
    false_negative: list[str] = Field(default_factory=list)


# ------------------------------------------------------------- the RAGAS bridge
class _RagasBackend:
    """A loaded RAGAS package plus the metric objects we drive.

    Attributes:
        metrics: Metric name to the RAGAS metric instance.
        sample_cls: ``ragas.dataset_schema.SingleTurnSample``.
    """

    __slots__ = ("metrics", "sample_cls")

    def __init__(self, metrics: dict[str, Any], sample_cls: Any) -> None:
        """Store the resolved RAGAS objects.

        Args:
            metrics: Metric name to instance.
            sample_cls: The single-turn sample class.
        """
        self.metrics = metrics
        self.sample_cls = sample_cls


#: Metric name -> RAGAS class names to try, newest naming first.
_RAGAS_METRIC_CLASSES: dict[str, tuple[str, ...]] = {
    "faithfulness": ("Faithfulness",),
    "context_precision": (
        "LLMContextPrecisionWithReference",
        "ContextPrecision",
    ),
    "context_recall": ("LLMContextRecall", "ContextRecall"),
}

#: ``(backend, reason)`` cached per settings fingerprint so the warning is logged
#: once per process rather than once per item.
_BACKEND_CACHE: dict[tuple[Any, ...], tuple[_RagasBackend | None, str]] = {}


def _backend_key(settings: Settings) -> tuple[Any, ...]:
    """Build the cache key for a resolved backend.

    Args:
        settings: Resolved settings.

    Returns:
        A hashable key covering everything that changes the backend.
    """
    return (
        bool(eval_setting(settings, "eval_ragas_enabled")),
        str(settings.eval_judge_model),
        str(eval_setting(settings, "eval_judge_effort")),
    )


def _make_ragas_llm(base_cls: Any, client: LLMClient, settings: Settings) -> Any:
    """Build the ``BaseRagasLLM`` adapter that routes RAGAS through Claude.

    Args:
        base_cls: ``ragas.llms.BaseRagasLLM``.
        client: The platform LLM client.
        settings: Resolved settings.

    Returns:
        An instance RAGAS can call.
    """
    model = str(settings.eval_judge_model)
    effort = str(eval_setting(settings, "eval_judge_effort"))
    from langchain_core.outputs import Generation, LLMResult

    class _ClaudeRagasLLM(base_cls):  # type: ignore[misc, valid-type]
        """Adapter presenting :class:`ragcore.llm.LLMClient` as a RAGAS LLM."""

        def __init__(self) -> None:
            """Initialise without a LangChain model — Claude is called directly."""
            self.run_config = None

        def is_finished(self, response: Any) -> bool:
            """Report generation as complete.

            Args:
                response: The result object RAGAS just received.

            Returns:
                Always True: :meth:`LLMClient.complete` returns whole messages.
            """
            del response
            return True

        def generate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float | None = None,
            stop: Sequence[str] | None = None,
            callbacks: Any = None,
        ) -> Any:
            """Refuse the synchronous path.

            Args:
                prompt: RAGAS prompt value.
                n: Requested generations.
                temperature: Ignored; sampling parameters are a 400 on Claude.
                stop: Ignored.
                callbacks: Ignored.

            Raises:
                NotImplementedError: The harness is async end to end.
            """
            del prompt, n, temperature, stop, callbacks
            msg = "the eval judge is async-only; use agenerate_text"
            raise NotImplementedError(msg)

        async def agenerate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float | None = None,
            stop: Sequence[str] | None = None,
            callbacks: Any = None,
        ) -> Any:
            """Run one RAGAS prompt through Claude.

            Args:
                prompt: RAGAS prompt value; ``to_string()`` renders it.
                n: Generations requested. Claude is called once per generation.
                temperature: Ignored — ``temperature`` returns 400 on the main
                    models, and a judge should not sample anyway.
                stop: Ignored; structured judging does not need stop sequences.
                callbacks: Ignored.

            Returns:
                A LangChain ``LLMResult`` carrying the generations.
            """
            del temperature, stop, callbacks
            text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
            generations = []
            for _ in range(max(1, int(n))):
                response = await client.complete(
                    system=_JUDGE_RULES,
                    messages=[{"role": "user", "content": text}],
                    model=model,
                    effort=effort,
                    name="eval.judge.ragas",
                    metadata={"prompt_version": JUDGE_PROMPT_VERSION},
                )
                generations.append(Generation(text=response.text))
            return LLMResult(generations=[generations])

    return _ClaudeRagasLLM()


def load_ragas(settings: Settings | None = None) -> tuple[_RagasBackend | None, str]:
    """Load RAGAS and build the metric objects, or explain why we cannot.

    Every failure mode — the extra is not installed, a class was renamed, a
    constructor changed, ``langchain-core`` is absent — is caught here and turned
    into a reason string. The caller then uses the native implementations.

    Args:
        settings: Resolved settings.

    Returns:
        ``(backend, reason)``. ``backend`` is None when RAGAS is unusable, and
        ``reason`` is empty when it is usable.
    """
    cfg = settings or get_settings()
    key = _backend_key(cfg)
    cached = _BACKEND_CACHE.get(key)
    if cached is not None:
        return cached

    result: tuple[_RagasBackend | None, str]
    if not eval_setting(cfg, "eval_ragas_enabled"):
        result = (None, "disabled by eval_ragas_enabled")
    else:
        try:
            result = (_build_backend(cfg), "")
        except ImportError as exc:
            result = (None, f"ragas extra not installed ({exc.__class__.__name__})")
        except Exception as exc:  # any RAGAS change degrades, not fails
            result = (None, f"ragas incompatible ({exc.__class__.__name__}: {exc})")

    if result[0] is None:
        _log.warning(
            "ragas_backend_unavailable",
            reason=result[1],
            fallback="native Claude-judged metrics",
        )
    else:
        _log.info("ragas_backend_ready", metrics=sorted(result[0].metrics))
    _BACKEND_CACHE[key] = result
    return result


def _build_backend(settings: Settings) -> _RagasBackend:
    """Import RAGAS and instantiate the LLM-only metrics.

    Args:
        settings: Resolved settings.

    Returns:
        A ready :class:`_RagasBackend`.

    Raises:
        ImportError: If RAGAS or its LangChain dependency is missing.
        RuntimeError: If no supported metric class could be resolved.
    """
    import ragas.metrics as ragas_metrics
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import BaseRagasLLM

    judge = _make_ragas_llm(BaseRagasLLM, get_llm_client(settings), settings)
    metrics: dict[str, Any] = {}
    for name, candidates in _RAGAS_METRIC_CLASSES.items():
        for class_name in candidates:
            metric_cls = getattr(ragas_metrics, class_name, None)
            if metric_cls is None:
                continue
            try:
                metrics[name] = metric_cls(llm=judge)
            except Exception:  # try the next candidate spelling
                _log.debug("ragas_metric_construct_failed", metric=class_name)
                continue
            break
    if not metrics:
        msg = "no supported RAGAS metric classes found"
        raise RuntimeError(msg)
    return _RagasBackend(metrics=metrics, sample_cls=SingleTurnSample)


# ------------------------------------------------------------------- the adapter
class RagasAdapter:
    """Scores one :class:`RagasSample` with RAGAS where possible, natively always.

    Attributes:
        settings: Resolved settings.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
    ) -> None:
        """Build an adapter.

        Args:
            settings: Resolved settings.
            llm: LLM client override, mainly for tests.
        """
        self.settings = settings or get_settings()
        self._llm = llm
        self._backend, self._reason = load_ragas(self.settings)

    @property
    def available(self) -> bool:
        """Whether the RAGAS package backend loaded.

        Returns:
            True when RAGAS itself will serve the LLM-only metrics.
        """
        return self._backend is not None

    @property
    def llm(self) -> LLMClient:
        """The judge client.

        Returns:
            The injected client, or the process-wide one.
        """
        if self._llm is None:
            self._llm = get_llm_client(self.settings)
        return self._llm

    # ------------------------------------------------------------------ public
    async def score(self, sample: RagasSample) -> RagasScores:
        """Score one sample on all five metrics.

        Args:
            sample: The item to score.

        Returns:
            A :class:`RagasScores`. A metric the judge could not produce stays
            None rather than defaulting to zero — an unmeasured metric must not
            look like a failing one.
        """
        scores = RagasScores(
            degraded=self._backend is None
            and bool(eval_setting(self.settings, "eval_ragas_enabled")),
            degraded_reason=self._reason,
        )
        if not sample.answer.strip() and not sample.contexts:
            scores.backends = dict.fromkeys(RAGAS_METRIC_NAMES, "unavailable")
            return scores

        results = await asyncio.gather(
            self._faithfulness(sample),
            self._answer_relevancy(sample),
            self._context_precision(sample),
            self._context_recall(sample),
            self._answer_correctness(sample),
        )
        for name, (value, backend) in zip(RAGAS_METRIC_NAMES, results, strict=True):
            setattr(scores, name, value)
            scores.backends[name] = backend
        return scores

    async def score_many(
        self, samples: Sequence[RagasSample], *, concurrency: int | None = None
    ) -> list[RagasScores]:
        """Score several samples with bounded concurrency.

        Args:
            samples: Items to score.
            concurrency: Parallel samples; defaults to ``eval_max_concurrency``.

        Returns:
            One score set per sample, in input order.
        """
        limit = concurrency or int(self.settings.eval_max_concurrency)
        semaphore = asyncio.Semaphore(max(1, limit))

        async def _one(sample: RagasSample) -> RagasScores:
            async with semaphore:
                return await self.score(sample)

        return list(await asyncio.gather(*(_one(sample) for sample in samples)))

    # ------------------------------------------------------------------ judges
    def _contexts(self, sample: RagasSample) -> list[str]:
        """Clip the contexts handed to a judge call.

        Args:
            sample: The item being scored.

        Returns:
            At most ``eval_judge_max_contexts`` contexts, each clipped to
            ``eval_judge_max_context_chars``.
        """
        max_contexts = int(eval_setting(self.settings, "eval_judge_max_contexts"))
        max_chars = int(eval_setting(self.settings, "eval_judge_max_context_chars"))
        return [text[:max_chars] for text in sample.contexts[:max_contexts]]

    async def _judge(
        self,
        *,
        system: str,
        payload: dict[str, Any],
        schema: type[BaseModel],
        name: str,
    ) -> Any | None:
        """Run one structured judge call.

        Args:
            system: The judge system prompt.
            payload: JSON-serialisable inputs for the judge.
            schema: Pydantic model the judge must return.
            name: Trace/metric label, e.g. ``"eval.judge.faithfulness"``.

        Returns:
            The parsed verdict, or None when the judge refused, timed out or
            returned something unparseable. A judge failure must degrade the score
            to "not measured", never to zero.
        """
        try:
            return await self.llm.structured(
                system=system,
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    }
                ],
                schema=schema,
                model=str(self.settings.eval_judge_model),
                effort=str(eval_setting(self.settings, "eval_judge_effort")),
                name=name,
                metadata={"prompt_version": JUDGE_PROMPT_VERSION},
            )
        except Exception:  # a judge failure is not a pipeline failure
            _log.warning("eval_judge_failed", judge=name, exc_info=True)
            return None

    async def _ragas_metric(self, name: str, sample: RagasSample) -> float | None:
        """Score one metric with the RAGAS package.

        Args:
            name: Metric name.
            sample: The item to score.

        Returns:
            The score, or None when RAGAS could not produce one.
        """
        backend = self._backend
        if backend is None or name not in backend.metrics:
            return None
        try:
            single = backend.sample_cls(
                user_input=sample.question,
                response=sample.answer,
                retrieved_contexts=self._contexts(sample),
                reference=sample.ground_truth,
            )
            value = await backend.metrics[name].single_turn_ascore(single)
        except Exception:  # degrade this metric to native
            _log.warning("ragas_metric_failed", metric=name, exc_info=True)
            return None
        return _bounded(value)

    async def _faithfulness(self, sample: RagasSample) -> tuple[float | None, str]:
        """Score faithfulness of the answer to the retrieved context.

        Args:
            sample: The item to score.

        Returns:
            ``(score, backend)``.
        """
        value = await self._ragas_metric("faithfulness", sample)
        if value is not None:
            return value, "ragas"
        contexts = self._contexts(sample)
        if not contexts or not sample.answer.strip():
            return None, "unavailable"
        verdict = await self._judge(
            system=FAITHFULNESS_SYSTEM,
            payload={"answer": sample.answer, "contexts": contexts},
            schema=_FaithfulnessVerdict,
            name="eval.judge.faithfulness",
        )
        if verdict is None:
            return None, "unavailable"
        if not verdict.claims:
            # No factual claims to be unfaithful with — a refusal, or a pure
            # clarifying question. Scoring zero here would punish the safe answer.
            return 1.0, "native"
        supported = sum(1 for claim in verdict.claims if claim.supported)
        return _bounded(supported / len(verdict.claims)), "native"

    async def _answer_relevancy(self, sample: RagasSample) -> tuple[float | None, str]:
        """Score how directly the answer addresses the question.

        Mirrors the RAGAS construction: generate the questions the answer would be
        an answer to, embed them, and average their cosine similarity with the real
        question. A noncommittal answer scores zero by definition.

        Args:
            sample: The item to score.

        Returns:
            ``(score, backend)``.
        """
        if not sample.answer.strip() or not sample.question.strip():
            return None, "unavailable"
        count = int(eval_setting(self.settings, "eval_relevancy_probe_questions"))
        verdict = await self._judge(
            system=RELEVANCY_SYSTEM,
            payload={
                "answer": sample.answer,
                "questions_to_generate": max(1, count),
            },
            schema=_RelevancyVerdict,
            name="eval.judge.answer_relevancy",
        )
        if verdict is None:
            return None, "unavailable"
        if verdict.noncommittal:
            return 0.0, "native"
        probes = [text for text in verdict.questions if text.strip()][: max(1, count)]
        if not probes:
            return None, "unavailable"
        vectors = await embed_texts([sample.question, *probes], settings=self.settings)
        reference = vectors[0]
        similarities = [
            max(0.0, cosine_similarity(reference, vector)) for vector in vectors[1:]
        ]
        return _bounded(sum(similarities) / len(similarities)), "native"

    async def _context_precision(self, sample: RagasSample) -> tuple[float | None, str]:
        """Score how much of the retrieved context was actually useful.

        Args:
            sample: The item to score.

        Returns:
            ``(score, backend)``. The native path is RAGAS's average precision at
            k: a useful chunk ranked first is worth more than one ranked last.
        """
        value = await self._ragas_metric("context_precision", sample)
        if value is not None:
            return value, "ragas"
        contexts = self._contexts(sample)
        if not contexts:
            return None, "unavailable"
        verdict = await self._judge(
            system=CONTEXT_PRECISION_SYSTEM,
            payload={
                "question": sample.question,
                "reference_answer": sample.ground_truth or sample.answer,
                "contexts": [
                    {"index": index, "text": text}
                    for index, text in enumerate(contexts, start=1)
                ],
            },
            schema=_ContextPrecisionVerdict,
            name="eval.judge.context_precision",
        )
        if verdict is None:
            return None, "unavailable"
        useful = {
            item.index
            for item in verdict.verdicts
            if item.useful and 1 <= item.index <= len(contexts)
        }
        if not useful:
            return 0.0, "native"
        hits = 0
        precision_sum = 0.0
        for rank in range(1, len(contexts) + 1):
            if rank in useful:
                hits += 1
                precision_sum += hits / rank
        return _bounded(precision_sum / len(useful)), "native"

    async def _context_recall(self, sample: RagasSample) -> tuple[float | None, str]:
        """Score how much of the reference answer the context can support.

        Args:
            sample: The item to score.

        Returns:
            ``(score, backend)``.
        """
        value = await self._ragas_metric("context_recall", sample)
        if value is not None:
            return value, "ragas"
        contexts = self._contexts(sample)
        if not contexts or not sample.ground_truth.strip():
            return None, "unavailable"
        verdict = await self._judge(
            system=CONTEXT_RECALL_SYSTEM,
            payload={"reference_answer": sample.ground_truth, "contexts": contexts},
            schema=_ContextRecallVerdict,
            name="eval.judge.context_recall",
        )
        if verdict is None:
            return None, "unavailable"
        if not verdict.statements:
            return None, "unavailable"
        attributed = sum(1 for item in verdict.statements if item.attributed)
        return _bounded(attributed / len(verdict.statements)), "native"

    async def _answer_correctness(
        self, sample: RagasSample
    ) -> tuple[float | None, str]:
        """Score the answer against the reference answer.

        RAGAS's definition: a weighted sum of statement-level F1 and embedding
        similarity. The similarity half comes from :mod:`eval.semantic`, which is
        why this metric never uses the RAGAS backend.

        Args:
            sample: The item to score.

        Returns:
            ``(score, backend)``.
        """
        if not sample.ground_truth.strip() or not sample.answer.strip():
            return None, "unavailable"
        verdict = await self._judge(
            system=CORRECTNESS_SYSTEM,
            payload={"answer": sample.answer, "reference_answer": sample.ground_truth},
            schema=_CorrectnessVerdict,
            name="eval.judge.answer_correctness",
        )
        if verdict is None:
            return None, "unavailable"
        true_positive = len(verdict.true_positive)
        penalty = 0.5 * (len(verdict.false_positive) + len(verdict.false_negative))
        denominator = true_positive + penalty
        f1 = true_positive / denominator if denominator else 0.0

        f1_weight = float(eval_setting(self.settings, "eval_correctness_f1_weight"))
        sim_weight = float(
            eval_setting(self.settings, "eval_correctness_similarity_weight")
        )
        similarity = sample.semantic_similarity
        if similarity is None:
            total = f1_weight or 1.0
            return _bounded(f1 * f1_weight / total), "native"
        total = f1_weight + sim_weight or 1.0
        combined = (f1 * f1_weight + similarity * sim_weight) / total
        return _bounded(combined), "native"


def _bounded(value: Any) -> float | None:
    """Coerce a judge or RAGAS output onto ``[0, 1]``.

    Args:
        value: Raw score; RAGAS returns ``float('nan')`` for "not applicable".

    Returns:
        The clamped float, or None when the value is not a finite number.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN: RAGAS's "could not score"
        return None
    return max(0.0, min(1.0, number))


#: Adapters cached per settings fingerprint; building one loads RAGAS.
_ADAPTER_CACHE: dict[tuple[Any, ...], RagasAdapter] = {}


def get_ragas_adapter(settings: Settings | None = None) -> RagasAdapter:
    """Return the shared adapter for a settings instance.

    Args:
        settings: Resolved settings.

    Returns:
        A cached :class:`RagasAdapter`.
    """
    cfg = settings or get_settings()
    key = _backend_key(cfg)
    adapter = _ADAPTER_CACHE.get(key)
    if adapter is None:
        adapter = RagasAdapter(cfg)
        _ADAPTER_CACHE[key] = adapter
    return adapter


def reset_ragas_cache() -> None:
    """Drop the cached backend and adapters. Test helper."""
    _BACKEND_CACHE.clear()
    _ADAPTER_CACHE.clear()
