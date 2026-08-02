"""Run artefacts and the markdown / HTML reports rendered from them.

Three views of the same run, because three different people read it:

* the **aggregate table** — one number per metric, with the gate's verdict beside
  it, for whoever is looking at a red build;
* the **per-category breakdown** — an ``acl_negative`` regression has to be
  distinguishable from a quality dip, so categories are never averaged away;
* the **worst items**, each with its Langfuse trace id, for whoever has to fix it.

Plus an A/B section when a baseline run is supplied: a metric that moved by more
than ``eval_regression_tolerance`` is called out, and so is every item that flipped
from pass to fail.

The HTML is fully self-contained — inline CSS, no fonts, no scripts, no network —
so it survives being attached to a CI artefact or emailed.
"""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from eval import eval_setting
from eval.ci_gate import GateReport
from ragcore.logging import get_logger
from ragcore.models.eval import EvalResult, EvalRun
from ragcore.settings import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "REPORT_METRIC_ORDER",
    "EvalItemDiagnostics",
    "EvalRunArtifacts",
    "ItemDelta",
    "MetricDelta",
    "ReportPaths",
    "RunComparison",
    "category_aggregate",
    "compare_runs",
    "load_artifacts",
    "render_html",
    "render_markdown",
    "worst_items",
    "write_reports",
]

_log = get_logger(__name__)

#: Metric column order shared by every table in the report.
REPORT_METRIC_ORDER: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
    "semantic_similarity",
    "citation_validity",
    "acl_leak",
    "refusal_correct",
    "tool_correct",
    "retrieval_recall",
    "latency_ms",
    "cost_usd",
    "pass_rate",
)

#: Metrics where a larger number is worse.
_LOWER_IS_BETTER: frozenset[str] = frozenset({"latency_ms", "cost_usd"})


class EvalItemDiagnostics(BaseModel):
    """Per-item context that :class:`~ragcore.models.eval.EvalResult` cannot carry.

    ``EvalResult`` and ``MetricScores`` are contract models with ``extra="forbid"``,
    which is exactly right — but the harness still needs to report the category, the
    persona, the tool that was (or was not) invoked and why an item was skipped. That
    lives here, keyed by ``item_id`` alongside the run.
    """

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(description="Golden item id.")
    category: str = Field(default="", description="Golden item category.")
    persona: str = Field(default="", description="Persona key the item ran as.")
    tenant_id: str = Field(default="", description="Tenant the item ran in.")
    question: str = Field(default="", description="The question, from the fixture.")
    expected_tool: str | None = Field(
        default=None, description="Tool the item says is required."
    )
    tools_invoked: list[str] = Field(
        default_factory=list, description="Tools the turn actually called."
    )
    tool_correct: float | None = Field(
        default=None, description="1.0 when the expected tool was invoked."
    )
    retrieval_recall: float | None = Field(
        default=None, description="Fraction of expected evidence retrieved."
    )
    missing_document_ids: list[str] = Field(
        default_factory=list, description="Expected documents never retrieved."
    )
    retrieved_document_ids: list[str] = Field(
        default_factory=list, description="Documents behind the retrieved chunks."
    )
    refused: bool = Field(default=False, description="The answer was a refusal.")
    expect_refusal: bool = Field(default=False, description="A refusal was required.")
    skipped: bool = Field(
        default=False, description="The item did not run; see skip_reason."
    )
    skip_reason: str = Field(default="", description="Why the item was skipped.")
    ragas_backend: str = Field(
        default="", description="'ragas' or 'native' for this item's judge."
    )
    degraded_reason: str = Field(
        default="", description="Why RAGAS was not used, when it was not."
    )
    answer_preview: str = Field(
        default="",
        description=(
            "Clipped answer as emitted by stage 12, i.e. after the PII egress "
            "redaction. The harness never stores a pre-redaction answer."
        ),
    )


class EvalRunArtifacts(BaseModel):
    """Everything one evaluation run produced, serialised as one JSON document."""

    model_config = ConfigDict(extra="forbid")

    run: EvalRun = Field(description="The contract-shaped run and its results.")
    diagnostics: dict[str, EvalItemDiagnostics] = Field(
        default_factory=dict, description="Per-item extras, keyed by item id."
    )
    category_aggregate: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Metric means per golden-item category."
    )
    golden_path: str = Field(default="", description="Golden file that was run.")
    personas_path: str = Field(default="", description="Persona file that was used.")
    baseline_path: str | None = Field(
        default=None, description="Baseline run compared against, when any."
    )
    gate: GateReport | None = Field(
        default=None, description="Gate outcome, when the gate ran."
    )
    skipped_items: list[str] = Field(
        default_factory=list, description="Items that did not run."
    )

    def diagnostics_for(self, item_id: str) -> EvalItemDiagnostics:
        """Look up an item's diagnostics, or an empty record.

        Args:
            item_id: Golden item id.

        Returns:
            The stored diagnostics, or a blank one so rendering never branches.
        """
        return self.diagnostics.get(item_id) or EvalItemDiagnostics(item_id=item_id)


class MetricDelta(BaseModel):
    """How one aggregate metric moved between two runs."""

    model_config = ConfigDict(extra="forbid")

    metric: str = Field(description="Aggregate key.")
    current: float | None = Field(default=None, description="This run's value.")
    baseline: float | None = Field(default=None, description="Baseline value.")
    delta: float | None = Field(default=None, description="Current minus baseline.")
    regression: bool = Field(
        default=False, description="Moved the wrong way by more than the tolerance."
    )
    improvement: bool = Field(
        default=False, description="Moved the right way by more than the tolerance."
    )


class ItemDelta(BaseModel):
    """An item whose pass/fail verdict changed between two runs."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(description="Golden item id.")
    category: str = Field(default="", description="Golden item category.")
    was_passing: bool = Field(default=False, description="Baseline verdict.")
    is_passing: bool = Field(default=False, description="Current verdict.")
    failures: list[str] = Field(
        default_factory=list, description="Current failure reasons."
    )
    trace_id: str | None = Field(default=None, description="Current trace id.")


class RunComparison(BaseModel):
    """A/B comparison of a run against a baseline."""

    model_config = ConfigDict(extra="forbid")

    baseline_run_id: str = Field(default="", description="Baseline run id.")
    tolerance: float = Field(
        default=0.0, description="Movement below this is treated as noise."
    )
    metrics: list[MetricDelta] = Field(
        default_factory=list, description="Per-metric movement, in report order."
    )
    regressions: list[ItemDelta] = Field(
        default_factory=list, description="Items that went from pass to fail."
    )
    fixes: list[ItemDelta] = Field(
        default_factory=list, description="Items that went from fail to pass."
    )
    new_items: list[str] = Field(
        default_factory=list, description="Items absent from the baseline."
    )
    missing_items: list[str] = Field(
        default_factory=list, description="Baseline items absent from this run."
    )

    @property
    def has_regressions(self) -> bool:
        """Whether anything regressed.

        Returns:
            True when a metric or an item regressed.
        """
        return bool(self.regressions) or any(delta.regression for delta in self.metrics)


class ReportPaths(BaseModel):
    """Where the three artefacts were written."""

    model_config = ConfigDict(extra="forbid")

    json_path: str = Field(description="Machine-readable run artefact.")
    markdown_path: str = Field(description="Markdown report.")
    html_path: str = Field(description="Self-contained HTML report.")


def category_aggregate(
    run: EvalRun, diagnostics: Mapping[str, EvalItemDiagnostics]
) -> dict[str, dict[str, float]]:
    """Average each metric within each golden-item category.

    Args:
        run: The completed run.
        diagnostics: Per-item diagnostics supplying the category.

    Returns:
        Category name to a metric mapping, with ``item_count`` and ``pass_rate``
        added per category.
    """
    buckets: dict[str, list[EvalResult]] = {}
    for result in run.results:
        record = diagnostics.get(result.item_id)
        category = (record.category if record else "") or "unknown"
        buckets.setdefault(category, []).append(result)

    output: dict[str, dict[str, float]] = {}
    for category, results in sorted(buckets.items()):
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for result in results:
            for name, value in result.scores.as_mapping().items():
                sums[name] = sums.get(name, 0.0) + float(value)
                counts[name] = counts.get(name, 0) + 1
            record = diagnostics.get(result.item_id)
            if record is not None and record.tool_correct is not None:
                sums["tool_correct"] = sums.get("tool_correct", 0.0) + float(
                    record.tool_correct
                )
                counts["tool_correct"] = counts.get("tool_correct", 0) + 1
            if record is not None and record.retrieval_recall is not None:
                sums["retrieval_recall"] = sums.get("retrieval_recall", 0.0) + float(
                    record.retrieval_recall
                )
                counts["retrieval_recall"] = counts.get("retrieval_recall", 0) + 1
        metrics = {name: sums[name] / counts[name] for name in sums}
        passed = sum(1 for result in results if result.passed)
        metrics["item_count"] = float(len(results))
        metrics["pass_rate"] = passed / len(results) if results else 0.0
        output[category] = metrics
    return output


def _item_score(result: EvalResult) -> float:
    """Rank an item for the "worst items" table.

    Args:
        result: One item's result.

    Returns:
        The mean of its bounded metric scores; 0.0 when nothing was scored, so an
        item that blew up sorts to the top of the worst list.
    """
    values = [
        value
        for name, value in result.scores.as_mapping().items()
        if name not in {"latency_ms", "cost_usd"}
    ]
    return sum(values) / len(values) if values else 0.0


def worst_items(run: EvalRun, *, limit: int) -> list[EvalResult]:
    """Pick the items most worth looking at.

    Args:
        run: The completed run.
        limit: How many to return.

    Returns:
        Failing items first, then the lowest mean score, capped at ``limit``.
    """
    ordered = sorted(
        run.results, key=lambda result: (result.passed, _item_score(result))
    )
    return ordered[: max(0, limit)]


def _category_of(records: Mapping[str, EvalItemDiagnostics], item_id: str) -> str:
    """Look up an item's category from the diagnostics.

    Args:
        records: Per-item diagnostics.
        item_id: Golden item id.

    Returns:
        The category, or an empty string when the item has no diagnostics.
    """
    record = records.get(item_id)
    return record.category if record is not None else ""


def _delta_label(delta: MetricDelta) -> str:
    """Label a metric movement for the markdown table.

    Args:
        delta: The measured movement.

    Returns:
        ``REGRESSION``, ``better`` or ``=``.
    """
    if delta.regression:
        return "REGRESSION"
    return "better" if delta.improvement else "="


def compare_runs(
    current: EvalRun,
    baseline: EvalRun,
    *,
    diagnostics: Mapping[str, EvalItemDiagnostics] | None = None,
    settings: Settings | None = None,
) -> RunComparison:
    """Diff a run against a previous one.

    Args:
        current: The run just produced.
        baseline: The run to compare against.
        diagnostics: Current-run diagnostics, for the category on an item delta.
        settings: Resolved settings supplying ``eval_regression_tolerance``.

    Returns:
        A :class:`RunComparison`.
    """
    cfg = settings or get_settings()
    tolerance = float(eval_setting(cfg, "eval_regression_tolerance"))
    records = diagnostics or {}

    current_aggregate = current.aggregate or current.compute_aggregate()
    baseline_aggregate = baseline.aggregate or baseline.compute_aggregate()

    metrics: list[MetricDelta] = []
    names = [name for name in REPORT_METRIC_ORDER if name in current_aggregate]
    names.extend(
        name
        for name in sorted(baseline_aggregate)
        if name not in current_aggregate and name in REPORT_METRIC_ORDER
    )
    for name in names:
        now = current_aggregate.get(name)
        before = baseline_aggregate.get(name)
        delta = None if now is None or before is None else now - before
        regression = improvement = False
        if delta is not None:
            # Latency and cost are the two metrics where "up" is the bad direction.
            inverted = name in _LOWER_IS_BETTER
            worse = delta > tolerance if inverted else delta < -tolerance
            better = delta < -tolerance if inverted else delta > tolerance
            # A relative tolerance is the honest test for a metric that is not on a
            # 0..1 scale; latency in milliseconds would otherwise always "regress".
            if name in _LOWER_IS_BETTER and before:
                worse = (delta / before) > tolerance
                better = (delta / before) < -tolerance
            regression, improvement = worse, better
        metrics.append(
            MetricDelta(
                metric=name,
                current=now,
                baseline=before,
                delta=delta,
                regression=regression,
                improvement=improvement,
            )
        )

    baseline_results = {result.item_id: result for result in baseline.results}
    current_results = {result.item_id: result for result in current.results}
    regressions: list[ItemDelta] = []
    fixes: list[ItemDelta] = []
    for item_id, result in current_results.items():
        previous = baseline_results.get(item_id)
        if previous is None:
            continue
        if previous.passed and not result.passed:
            regressions.append(
                ItemDelta(
                    item_id=item_id,
                    category=_category_of(records, item_id),
                    was_passing=True,
                    is_passing=False,
                    failures=list(result.failures),
                    trace_id=result.trace_id,
                )
            )
        elif not previous.passed and result.passed:
            fixes.append(
                ItemDelta(
                    item_id=item_id,
                    category=_category_of(records, item_id),
                    was_passing=False,
                    is_passing=True,
                    trace_id=result.trace_id,
                )
            )

    return RunComparison(
        baseline_run_id=baseline.run_id,
        tolerance=tolerance,
        metrics=metrics,
        regressions=sorted(regressions, key=lambda delta: delta.item_id),
        fixes=sorted(fixes, key=lambda delta: delta.item_id),
        new_items=sorted(set(current_results) - set(baseline_results)),
        missing_items=sorted(set(baseline_results) - set(current_results)),
    )


# ------------------------------------------------------------------- rendering
def _fmt(value: float | None, metric: str = "") -> str:
    """Format one metric value for a table.

    Args:
        value: The value, or None when unmeasured.
        metric: Metric name, so latency and cost get their own units.

    Returns:
        A short display string.
    """
    if value is None:
        return "-"
    if metric == "latency_ms":
        return f"{value:,.0f} ms"
    if metric == "cost_usd":
        return f"${value:.4f}"
    if metric in {"item_count"}:
        return f"{value:.0f}"
    return f"{value:.3f}"


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    """Render a markdown table.

    Args:
        headers: Column headers.
        rows: Row cells.

    Returns:
        The table as markdown, or an empty-state line when there are no rows.
    """
    body = list(rows)
    if not body:
        return "_no rows_\n"
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines) + "\n"


def _metric_names(artifacts: EvalRunArtifacts) -> list[str]:
    """Metrics present in this run, in report order.

    Args:
        artifacts: The run artefacts.

    Returns:
        The ordered metric names.
    """
    aggregate = artifacts.run.aggregate
    return [name for name in REPORT_METRIC_ORDER if name in aggregate]


def render_markdown(
    artifacts: EvalRunArtifacts,
    *,
    comparison: RunComparison | None = None,
    settings: Settings | None = None,
) -> str:
    """Render the full markdown report.

    Args:
        artifacts: The run artefacts, including the gate outcome.
        comparison: Baseline comparison, when a baseline was supplied.
        settings: Resolved settings supplying the worst-item count.

    Returns:
        The report as markdown.
    """
    cfg = settings or get_settings()
    run = artifacts.run
    limit = int(eval_setting(cfg, "eval_report_worst_items"))
    gate = artifacts.gate

    parts: list[str] = []
    parts.append(f"# Evaluation run `{run.run_id}`\n")
    verdict = "n/a" if gate is None else ("PASS" if gate.passed else "FAIL")
    parts.append(
        "\n".join(
            [
                f"- **Gate:** {verdict}",
                f"- **Items:** {run.item_count} "
                f"({run.passed_count} passed, {run.failed_count} failed, "
                f"{len(artifacts.skipped_items)} skipped)",
                f"- **Pass rate:** {run.pass_rate:.1%}",
                f"- **Started:** {run.started_at.isoformat()}",
                f"- **Duration:** {_fmt_duration(run.duration_seconds)}",
                f"- **Cost:** ${run.total_cost_usd:.4f}",
                f"- **Commit:** `{run.git_sha or 'unknown'}`",
                f"- **Config fingerprint:** `{run.config_fingerprint or 'unknown'}`",
                f"- **Golden set:** `{artifacts.golden_path}`",
            ]
        )
        + "\n"
    )

    parts.append("\n## Aggregate\n")
    names = _metric_names(artifacts)
    gate_by_metric = (
        {check.metric: check for check in gate.checks} if gate is not None else {}
    )
    rows = []
    for name in names:
        check = gate_by_metric.get(name)
        threshold = "-"
        status = "-"
        if check is not None:
            comparator = "<=" if check.direction == "max" else ">="
            threshold = f"{comparator} {_fmt(check.limit, name)}"
            status = check.status
        rows.append(
            [f"`{name}`", _fmt(run.aggregate.get(name), name), threshold, status]
        )
    parts.append(_markdown_table(["metric", "value", "threshold", "status"], rows))

    if gate is not None and gate.item_failures:
        parts.append("\n### Hard per-item failures\n")
        parts.append("\n".join(f"- {failure}" for failure in gate.item_failures) + "\n")

    parts.append("\n## Per category\n")
    category_rows = []
    for category, metrics in artifacts.category_aggregate.items():
        row = [f"`{category}`", _fmt(metrics.get("item_count"), "item_count")]
        row.append(_fmt(metrics.get("pass_rate")))
        row.extend(
            _fmt(metrics.get(name), name) for name in names if name != "pass_rate"
        )
        category_rows.append(row)
    others = [name for name in names if name != "pass_rate"]
    headers = ["category", "items", "pass_rate", *others]
    parts.append(_markdown_table(headers, category_rows))

    if comparison is not None:
        parts.append(f"\n## Versus baseline `{comparison.baseline_run_id}`\n")
        delta_rows = [
            [
                f"`{delta.metric}`",
                _fmt(delta.baseline, delta.metric),
                _fmt(delta.current, delta.metric),
                _fmt(delta.delta, delta.metric),
                _delta_label(delta),
            ]
            for delta in comparison.metrics
        ]
        parts.append(
            _markdown_table(["metric", "baseline", "current", "delta", ""], delta_rows)
        )
        if comparison.regressions:
            parts.append("\n### Items that regressed\n")
            parts.append(
                _markdown_table(
                    ["item", "category", "failures", "trace"],
                    [
                        [
                            f"`{delta.item_id}`",
                            delta.category or "-",
                            "; ".join(delta.failures) or "-",
                            f"`{delta.trace_id}`" if delta.trace_id else "-",
                        ]
                        for delta in comparison.regressions
                    ],
                )
            )
        if comparison.fixes:
            parts.append("\n### Items that started passing\n")
            parts.append(
                "\n".join(f"- `{delta.item_id}`" for delta in comparison.fixes) + "\n"
            )
        if comparison.new_items or comparison.missing_items:
            parts.append(
                f"\nNew items: {len(comparison.new_items)}; "
                f"items missing versus baseline: {len(comparison.missing_items)}.\n"
            )

    parts.append(f"\n## Worst {limit} items\n")
    worst_rows = []
    for result in worst_items(run, limit=limit):
        record = artifacts.diagnostics_for(result.item_id)
        worst_rows.append(
            [
                f"`{result.item_id}`",
                record.category or "-",
                record.persona or "-",
                "pass" if result.passed else "FAIL",
                _fmt(result.scores.faithfulness),
                _fmt(result.scores.semantic_similarity),
                _fmt(result.scores.citation_validity),
                _fmt(result.scores.acl_leak),
                "; ".join(result.failures) or "-",
                f"`{result.trace_id}`" if result.trace_id else "-",
            ]
        )
    parts.append(
        _markdown_table(
            [
                "item",
                "category",
                "persona",
                "verdict",
                "faith",
                "sim",
                "cite",
                "acl",
                "failures",
                "trace",
            ],
            worst_rows,
        )
    )

    if artifacts.skipped_items:
        parts.append("\n## Skipped items\n")
        parts.append(
            "\n".join(
                f"- `{item_id}`: "
                f"{artifacts.diagnostics_for(item_id).skip_reason or 'skipped'}"
                for item_id in artifacts.skipped_items
            )
            + "\n"
        )

    return "".join(parts)


def _fmt_duration(seconds: float | None) -> str:
    """Format a run duration.

    Args:
        seconds: Duration in seconds, or None while running.

    Returns:
        A short human string.
    """
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


_HTML_STYLE = """
:root { color-scheme: light dark; --fg:#111827; --bg:#ffffff; --muted:#6b7280;
  --line:#e5e7eb; --pass:#047857; --fail:#b91c1c; --warn:#b45309;
  --chip:#f3f4f6; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e5e7eb; --bg:#0b0f19; --muted:#9ca3af; --line:#1f2937;
    --pass:#34d399; --fail:#f87171; --warn:#fbbf24; --chip:#111827; }
}
* { box-sizing: border-box; }
body { margin:0; padding:2rem 1.25rem; background:var(--bg); color:var(--fg);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
main { max-width: 72rem; margin: 0 auto; }
h1 { font-size:1.5rem; margin:0 0 .25rem; }
h2 { font-size:1.1rem; margin:2rem 0 .5rem; border-bottom:1px solid var(--line);
  padding-bottom:.25rem; }
h3 { font-size:1rem; margin:1.25rem 0 .5rem; }
code, .mono { font-family: ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.92em; }
.meta { color:var(--muted); margin:0 0 1rem; }
.chips { display:flex; flex-wrap:wrap; gap:.5rem; margin:.75rem 0 0; padding:0;
  list-style:none; }
.chips li { background:var(--chip); border:1px solid var(--line); border-radius:999px;
  padding:.15rem .6rem; }
.scroll { overflow-x:auto; -webkit-overflow-scrolling:touch; }
table { border-collapse:collapse; width:100%; min-width:32rem; }
th, td { text-align:left; padding:.4rem .6rem; border-bottom:1px solid var(--line);
  vertical-align:top; white-space:nowrap; }
td.wrap { white-space:normal; min-width:16rem; }
th { font-weight:600; color:var(--muted); font-size:.82rem; text-transform:uppercase;
  letter-spacing:.04em; }
.pass { color:var(--pass); font-weight:600; }
.fail { color:var(--fail); font-weight:600; }
.warn { color:var(--warn); font-weight:600; }
.verdict { display:inline-block; padding:.2rem .7rem; border-radius:.4rem;
  font-weight:700; }
.verdict.ok { background:var(--pass); color:var(--bg); }
.verdict.bad { background:var(--fail); color:var(--bg); }
footer { margin-top:2.5rem; color:var(--muted); font-size:.85rem; }
"""


def _esc(value: object) -> str:
    """HTML-escape any value.

    Args:
        value: The value to render.

    Returns:
        The escaped string form.
    """
    return html.escape(str(value), quote=True)


def _html_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    wrap_columns: Sequence[int] = (),
) -> str:
    """Render an HTML table inside a horizontally scrollable wrapper.

    Args:
        headers: Column headers (already escaped or plain text).
        rows: Row cells; each cell may contain a pre-escaped ``<span>``.
        wrap_columns: Column indices allowed to wrap onto several lines.

    Returns:
        The table markup.
    """
    body = list(rows)
    if not body:
        return "<p class='meta'>no rows</p>"
    head = "".join(f"<th>{_esc(header)}</th>" for header in headers)
    lines = [f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>"]
    for row in body:
        cells = "".join(
            f"<td class='wrap'>{cell}</td>"
            if index in wrap_columns
            else f"<td>{cell}</td>"
            for index, cell in enumerate(row)
        )
        lines.append(f"<tr>{cells}</tr>")
    lines.append("</tbody></table></div>")
    return "".join(lines)


def _status_span(status: str) -> str:
    """Colour a status word.

    Args:
        status: ``pass``, ``FAIL``, ``HARD FAIL`` or ``n/a``.

    Returns:
        A span element.
    """
    css = "pass" if status == "pass" else ("fail" if "FAIL" in status else "warn")
    return f"<span class='{css}'>{_esc(status)}</span>"


def render_html(
    artifacts: EvalRunArtifacts,
    *,
    comparison: RunComparison | None = None,
    settings: Settings | None = None,
) -> str:
    """Render the self-contained HTML report.

    Args:
        artifacts: The run artefacts, including the gate outcome.
        comparison: Baseline comparison, when a baseline was supplied.
        settings: Resolved settings supplying the worst-item count.

    Returns:
        A complete HTML document with inline styles and no external requests.
    """
    cfg = settings or get_settings()
    run = artifacts.run
    gate = artifacts.gate
    limit = int(eval_setting(cfg, "eval_report_worst_items"))
    names = _metric_names(artifacts)

    verdict_html = "<span class='verdict'>gate not run</span>"
    if gate is not None:
        css = "ok" if gate.passed else "bad"
        verdict_html = f"<span class='verdict {css}'>{_esc(gate.summary())}</span>"

    gate_by_metric = (
        {check.metric: check for check in gate.checks} if gate is not None else {}
    )
    aggregate_rows = []
    for name in names:
        check = gate_by_metric.get(name)
        threshold = "-"
        status = "-"
        if check is not None:
            comparator = "≤" if check.direction == "max" else "≥"
            threshold = f"{comparator} {_fmt(check.limit, name)}"
            status = _status_span(check.status)
        aggregate_rows.append(
            [
                f"<code>{_esc(name)}</code>",
                _esc(_fmt(run.aggregate.get(name), name)),
                _esc(threshold),
                status,
                f"<code>{_esc(check.source)}</code>" if check is not None else "-",
            ]
        )

    others = [name for name in names if name != "pass_rate"]
    category_headers = ["category", "items", "pass_rate", *others]
    category_rows = []
    for category, metrics in artifacts.category_aggregate.items():
        row = [
            f"<code>{_esc(category)}</code>",
            _esc(_fmt(metrics.get("item_count"), "item_count")),
            _esc(_fmt(metrics.get("pass_rate"))),
        ]
        row.extend(
            _esc(_fmt(metrics.get(name), name)) for name in names if name != "pass_rate"
        )
        category_rows.append(row)

    worst_rows = []
    for result in worst_items(run, limit=limit):
        record = artifacts.diagnostics_for(result.item_id)
        worst_rows.append(
            [
                f"<code>{_esc(result.item_id)}</code>",
                _esc(record.category or "-"),
                _esc(record.persona or "-"),
                "<span class='pass'>pass</span>"
                if result.passed
                else "<span class='fail'>FAIL</span>",
                _esc(_fmt(result.scores.faithfulness)),
                _esc(_fmt(result.scores.semantic_similarity)),
                _esc(_fmt(result.scores.citation_validity)),
                _esc(_fmt(result.scores.acl_leak)),
                _esc("; ".join(result.failures) or "-"),
                f"<code>{_esc(result.trace_id)}</code>" if result.trace_id else "-",
            ]
        )

    sections = [
        "<h2>Aggregate</h2>",
        _html_table(
            ["metric", "value", "threshold", "status", "source"], aggregate_rows
        ),
    ]
    if gate is not None and gate.item_failures:
        sections.append("<h3>Hard per-item failures</h3><ul>")
        sections.extend(
            f"<li class='fail'>{_esc(item)}</li>" for item in gate.item_failures
        )
        sections.append("</ul>")

    sections.append("<h2>Per category</h2>")
    sections.append(_html_table(category_headers, category_rows))

    if comparison is not None:
        sections.append(
            f"<h2>Versus baseline <code>{_esc(comparison.baseline_run_id)}</code></h2>"
        )
        delta_rows = []
        for delta in comparison.metrics:
            if delta.regression:
                marker = "<span class='fail'>regression</span>"
            elif delta.improvement:
                marker = "<span class='pass'>better</span>"
            else:
                marker = "<span class='meta'>=</span>"
            delta_rows.append(
                [
                    f"<code>{_esc(delta.metric)}</code>",
                    _esc(_fmt(delta.baseline, delta.metric)),
                    _esc(_fmt(delta.current, delta.metric)),
                    _esc(_fmt(delta.delta, delta.metric)),
                    marker,
                ]
            )
        sections.append(
            _html_table(["metric", "baseline", "current", "delta", ""], delta_rows)
        )
        if comparison.regressions:
            sections.append("<h3>Items that regressed</h3>")
            sections.append(
                _html_table(
                    ["item", "category", "failures", "trace"],
                    [
                        [
                            f"<code>{_esc(delta.item_id)}</code>",
                            _esc(delta.category or "-"),
                            _esc("; ".join(delta.failures) or "-"),
                            f"<code>{_esc(delta.trace_id)}</code>"
                            if delta.trace_id
                            else "-",
                        ]
                        for delta in comparison.regressions
                    ],
                    wrap_columns=(2,),
                )
            )

    sections.append(f"<h2>Worst {limit} items</h2>")
    sections.append(
        _html_table(
            [
                "item",
                "category",
                "persona",
                "verdict",
                "faith",
                "sim",
                "cite",
                "acl",
                "failures",
                "trace",
            ],
            worst_rows,
            wrap_columns=(8,),
        )
    )

    if artifacts.skipped_items:
        sections.append("<h2>Skipped items</h2><ul>")
        sections.extend(
            f"<li><code>{_esc(item_id)}</code>: "
            f"{_esc(artifacts.diagnostics_for(item_id).skip_reason or 'skipped')}</li>"
            for item_id in artifacts.skipped_items
        )
        sections.append("</ul>")

    chips = [
        f"items {run.item_count}",
        f"passed {run.passed_count}",
        f"failed {run.failed_count}",
        f"skipped {len(artifacts.skipped_items)}",
        f"pass rate {run.pass_rate:.1%}",
        f"cost ${run.total_cost_usd:.4f}",
        f"duration {_fmt_duration(run.duration_seconds)}",
        f"commit {run.git_sha or 'unknown'}",
    ]
    chip_html = "".join(f"<li>{_esc(chip)}</li>" for chip in chips)

    generated = datetime.now(UTC).isoformat(timespec="seconds")
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Eval run {_esc(run.run_id)}</title>"
        f"<style>{_HTML_STYLE}</style></head><body><main>"
        f"<h1>Evaluation run <code>{_esc(run.run_id)}</code></h1>"
        f"<p class='meta'>golden set <code>{_esc(artifacts.golden_path)}</code> · "
        f"config <code>{_esc(run.config_fingerprint or 'unknown')}</code></p>"
        f"<p>{verdict_html}</p>"
        f"<ul class='chips'>{chip_html}</ul>"
        + "".join(sections)
        + f"<footer>Generated {_esc(generated)} by services/eval. "
        "Answers shown here are post-egress-redaction.</footer>"
        "</main></body></html>"
    )


def write_reports(
    artifacts: EvalRunArtifacts,
    *,
    directory: str | Path | None = None,
    comparison: RunComparison | None = None,
    settings: Settings | None = None,
) -> ReportPaths:
    """Write the JSON artefact and both rendered reports.

    Args:
        artifacts: The run artefacts.
        directory: Output directory; defaults to ``eval_report_dir``.
        comparison: Baseline comparison, when a baseline was supplied.
        settings: Resolved settings.

    Returns:
        The three paths written.
    """
    cfg = settings or get_settings()
    target = Path(directory or eval_setting(cfg, "eval_report_dir"))
    target.mkdir(parents=True, exist_ok=True)
    stem = artifacts.run.run_id

    json_path = target / f"{stem}.json"
    markdown_path = target / f"{stem}.md"
    html_path = target / f"{stem}.html"

    json_path.write_text(
        json.dumps(artifacts.model_dump(mode="json"), indent=2, sort_keys=False),
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_markdown(artifacts, comparison=comparison, settings=cfg),
        encoding="utf-8",
    )
    html_path.write_text(
        render_html(artifacts, comparison=comparison, settings=cfg), encoding="utf-8"
    )

    # Stable names so CI can link "the latest run" without knowing the run id.
    (target / "latest.json").write_text(
        json_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (target / "latest.md").write_text(
        markdown_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (target / "latest.html").write_text(
        html_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    _log.info(
        "eval_report_written",
        run_id=stem,
        directory=str(target),
        items=artifacts.run.item_count,
    )
    return ReportPaths(
        json_path=str(json_path),
        markdown_path=str(markdown_path),
        html_path=str(html_path),
    )


def load_artifacts(path: str | Path) -> EvalRunArtifacts:
    """Read a previously written run artefact.

    Accepts both the artefact envelope and a bare serialised
    :class:`~ragcore.models.eval.EvalRun`, so ``--baseline`` works against an older
    file that only stored the run.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed artefacts.

    Raises:
        ValueError: If the document is not a JSON object.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        msg = f"{path} does not contain an evaluation run"
        raise ValueError(msg)
    if "run" in document:
        return EvalRunArtifacts.model_validate(document)
    return EvalRunArtifacts(run=EvalRun.model_validate(document))
