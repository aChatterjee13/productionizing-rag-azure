"""Evaluation models for requirement #8: RAGAS plus semantic similarity vs a golden set.

The golden set deliberately mixes categories so the harness exercises the guardrails,
not just happy-path retrieval: ``out_of_domain`` items must be refused,
``acl_negative`` items must return nothing for the given persona, ``pii`` items must
come back redacted, and ``tool_required`` items must invoke a named tool.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "EvalCategory",
    "EvalResult",
    "EvalRun",
    "GoldenItem",
    "MetricScores",
]


def _utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    Returns:
        The current moment in UTC.
    """
    return datetime.now(UTC)


class EvalCategory(StrEnum):
    """What a golden item is testing."""

    IN_DOMAIN = "in_domain"
    OUT_OF_DOMAIN = "out_of_domain"
    PII = "pii"
    CONTRADICTION = "contradiction"
    ACL_NEGATIVE = "acl_negative"
    TOOL_REQUIRED = "tool_required"


class GoldenItem(BaseModel):
    """One question in the golden set, with its expectations."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(description="Stable item id, used to diff runs over time.")
    question: str = Field(description="The question to ask.")
    ground_truth: str = Field(description="Reference answer for correctness scoring.")
    expected_document_ids: list[str] = Field(
        default_factory=list, description="Documents that should be retrieved."
    )
    expected_chunk_ids: list[str] = Field(
        default_factory=list, description="Chunks that should be retrieved."
    )
    must_contain: list[str] = Field(
        default_factory=list, description="Substrings the answer must include."
    )
    must_not_contain: list[str] = Field(
        default_factory=list,
        description=(
            "Substrings the answer must not include. Used for ACL-negative and PII "
            "cases: a leaked value here fails the item outright."
        ),
    )
    as_user: str = Field(
        description="Persona key from the golden file, resolved to a Principal."
    )
    tenant_id: str = Field(description="Tenant the question is asked in.")
    category: str = Field(
        description=(
            "in_domain | out_of_domain | pii | contradiction | acl_negative | "
            "tool_required."
        )
    )
    expect_refusal: bool = Field(
        default=False,
        description="The answer must be an explicit out-of-corpus refusal.",
    )
    expect_tool: str | None = Field(
        default=None, description="Tool name that must be invoked to answer."
    )

    @property
    def category_enum(self) -> EvalCategory:
        """Typed view of :attr:`category`.

        Returns:
            The parsed category, falling back to ``IN_DOMAIN`` for unknown values so
            a hand-edited golden file cannot crash the whole run.
        """
        try:
            return EvalCategory(self.category)
        except ValueError:
            return EvalCategory.IN_DOMAIN


class MetricScores(BaseModel):
    """Per-item metric scores. None means the metric was not applicable or not run."""

    model_config = ConfigDict(extra="forbid")

    faithfulness: float | None = Field(
        default=None, description="RAGAS: is the answer grounded in the context?"
    )
    answer_relevancy: float | None = Field(
        default=None, description="RAGAS: does the answer address the question?"
    )
    context_precision: float | None = Field(
        default=None, description="RAGAS: are retrieved chunks relevant?"
    )
    context_recall: float | None = Field(
        default=None, description="RAGAS: was the needed context retrieved?"
    )
    answer_correctness: float | None = Field(
        default=None, description="RAGAS: does the answer match the ground truth?"
    )
    semantic_similarity: float | None = Field(
        default=None, description="Sentence-embedding cosine against the ground truth."
    )
    citation_validity: float | None = Field(
        default=None,
        description=(
            "Fraction of cited spans that actually occur in their cited chunks."
        ),
    )
    acl_leak: float | None = Field(
        default=None,
        description=(
            "1.0 means no leak. Anything below 1.0 is a cross-tenant or "
            "over-clearance chunk reaching the answer and fails the gate."
        ),
    )
    refusal_correct: float | None = Field(
        default=None,
        description="1.0 when refusal behaviour matched expect_refusal, else 0.0.",
    )
    latency_ms: float = Field(
        default=0.0, ge=0.0, description="End-to-end item latency."
    )
    cost_usd: float = Field(
        default=0.0, ge=0.0, description="Anthropic spend for the item."
    )

    def as_mapping(self) -> dict[str, float]:
        """Flatten to a name -> value mapping, skipping unmeasured metrics.

        Returns:
            The populated metrics, ready for aggregation.
        """
        return {
            name: value
            for name, value in self.model_dump().items()
            if isinstance(value, (int, float))
        }


class EvalResult(BaseModel):
    """The outcome of evaluating one golden item."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(description="Golden item this result belongs to.")
    answer: str = Field(default="", description="Answer the pipeline produced.")
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list, description="Chunk ids the pipeline retrieved."
    )
    scores: MetricScores = Field(
        default_factory=MetricScores, description="Metric scores for this item."
    )
    passed: bool = Field(default=False, description="Whether every assertion held.")
    failures: list[str] = Field(
        default_factory=list,
        description="Human-readable reasons this item failed, one per assertion.",
    )
    trace_id: str | None = Field(
        default=None, description="Langfuse trace id for drill-down."
    )

    @classmethod
    def error(cls, item_id: str, message: str) -> Self:
        """Build a failed result for an item that raised before scoring.

        Args:
            item_id: Golden item id.
            message: Redacted failure description.

        Returns:
            A failing :class:`EvalResult`.
        """
        return cls(item_id=item_id, passed=False, failures=[message])


class EvalRun(BaseModel):
    """A full evaluation run over the golden set, and whether it passes the CI gate."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(description="Unique run id.")
    started_at: datetime = Field(default_factory=_utcnow, description="Run start time.")
    finished_at: datetime | None = Field(default=None, description="Run end time.")
    git_sha: str | None = Field(
        default=None, description="Commit under test, for regression tracking."
    )
    config_fingerprint: str = Field(
        default="",
        description=(
            "Hash of the retrieval/generation settings that shaped this run, so a "
            "score change can be attributed to config rather than code."
        ),
    )
    results: list[EvalResult] = Field(
        default_factory=list, description="Per-item results."
    )
    aggregate: dict[str, float] = Field(
        default_factory=dict, description="Mean of each metric across scored items."
    )
    gate_passed: bool = Field(
        default=False, description="Whether every configured threshold was met."
    )

    @property
    def item_count(self) -> int:
        """Number of items evaluated.

        Returns:
            The result count.
        """
        return len(self.results)

    @property
    def passed_count(self) -> int:
        """Number of items that passed every assertion.

        Returns:
            The count of passing results.
        """
        return sum(1 for result in self.results if result.passed)

    @property
    def failed_count(self) -> int:
        """Number of items that failed at least one assertion.

        Returns:
            The count of failing results.
        """
        return self.item_count - self.passed_count

    @property
    def pass_rate(self) -> float:
        """Fraction of items that passed.

        Returns:
            Passed over total, or 0.0 for an empty run.
        """
        if not self.results:
            return 0.0
        return self.passed_count / self.item_count

    @property
    def total_cost_usd(self) -> float:
        """Total Anthropic spend across the run.

        Returns:
            Summed per-item cost.
        """
        return sum(result.scores.cost_usd for result in self.results)

    @property
    def duration_seconds(self) -> float | None:
        """Wall-clock run duration.

        Returns:
            Seconds between start and finish, or None while still running.
        """
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    def compute_aggregate(self) -> dict[str, float]:
        """Average every populated metric across items and store the result.

        Returns:
            The aggregate mapping, also assigned to :attr:`aggregate`. Includes
            derived ``pass_rate``, ``item_count`` and ``total_cost_usd`` entries so
            the CI gate has everything it needs in one place.
        """
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for result in self.results:
            for name, value in result.scores.as_mapping().items():
                sums[name] = sums.get(name, 0.0) + float(value)
                counts[name] = counts.get(name, 0) + 1
        aggregate = {name: sums[name] / counts[name] for name in sums}
        aggregate["pass_rate"] = self.pass_rate
        aggregate["item_count"] = float(self.item_count)
        aggregate["total_cost_usd"] = self.total_cost_usd
        self.aggregate = aggregate
        return aggregate

    def failures_by_category(self, items: dict[str, GoldenItem]) -> dict[str, int]:
        """Count failures per golden-item category.

        Args:
            items: Golden items keyed by ``item_id``.

        Returns:
            A mapping of category name to failure count, useful for spotting that
            (say) every ``acl_negative`` item regressed at once.
        """
        counts: dict[str, int] = {}
        for result in self.results:
            if result.passed:
                continue
            item = items.get(result.item_id)
            category = item.category if item else "unknown"
            counts[category] = counts.get(category, 0) + 1
        return counts
