"""The CI gate: per-metric thresholds, two of them hard.

Every threshold is a :class:`ragcore.settings.Settings` field. Two of them are
different in kind from the rest:

* ``acl_leak`` must be **1.0**. A single chunk from another tenant, above the
  persona's clearance, or from a document the persona is denied is a security
  defect; there is no "mostly isolated".
* ``refusal_correct`` must be at least **0.95**. Answering a question the corpus
  cannot support is the failure mode RAG is built to avoid, and a refusal
  regression is never a rounding error.

Configuration may *tighten* those two but never loosen them: the effective floor is
``max(configured minimum, hard floor)``. Everything else — faithfulness, relevancy,
precision, recall, correctness, similarity, citation validity, tool correctness,
retrieval recall and the latency budget — is a soft threshold read straight from
settings.

A metric nothing measured is reported as ``not measured`` and does not fail the
build: an absent score is not a zero. It is still printed, because a metric that
silently stops being measured is how a gate quietly stops gating.

Two ways in. ``python -m eval.ci_gate <run.json>`` judges a run someone else wrote,
which is what a pipeline does after ``python -m eval.run_eval``. ``python -m
eval.ci_gate --run`` executes the golden set first and then judges it, and
``--offline`` executes it against :func:`eval.harness.offline_environment` — the
real orchestrator, retriever, guardrails and ACL filters driven by a faked
Anthropic, a faked Qdrant and a faked embedder. A gate that cannot be run without a
live stack is a gate nobody runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from eval import eval_setting
from ragcore.logging import configure_logging, get_logger
from ragcore.models.eval import EvalRun
from ragcore.settings import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from eval.report import EvalRunArtifacts

__all__ = [
    "EXIT_FAILED",
    "EXIT_GATE_FAILED",
    "EXIT_OK",
    "HARD_METRICS",
    "GateCheck",
    "GateReport",
    "Threshold",
    "apply_gate",
    "evaluate_gate",
    "format_gate_table",
    "gate_thresholds",
    "main",
    "run_and_gate",
]

_log = get_logger(__name__)

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_FAILED = 2

#: Metrics whose floor cannot be configured away.
HARD_METRICS: tuple[str, ...] = ("acl_leak", "refusal_correct")


@dataclass(frozen=True, slots=True)
class Threshold:
    """One gate condition.

    Attributes:
        metric: Aggregate key the condition reads.
        limit: The boundary value.
        direction: ``"min"`` when the metric must be at least ``limit``,
            ``"max"`` when it must be at most ``limit`` (the latency budget).
        hard: Whether a breach is a build-stopping correctness failure.
        source: Settings field the limit came from, printed in the table so an
            argument about a threshold is an argument about a config value.
    """

    metric: str
    limit: float
    direction: str = "min"
    hard: bool = False
    source: str = ""


class GateCheck(BaseModel):
    """The outcome of one threshold against one run."""

    model_config = ConfigDict(extra="forbid")

    metric: str = Field(description="Aggregate key checked.")
    limit: float = Field(description="Threshold value.")
    direction: str = Field(default="min", description="'min' or 'max'.")
    value: float | None = Field(
        default=None, description="Measured aggregate, None when not measured."
    )
    measured: bool = Field(default=False, description="Whether anything scored it.")
    passed: bool = Field(default=True, description="Whether the condition held.")
    hard: bool = Field(default=False, description="Whether a breach stops the build.")
    source: str = Field(default="", description="Settings field behind the limit.")

    @property
    def status(self) -> str:
        """Single-word status for the table.

        Returns:
            ``"pass"``, ``"FAIL"``, ``"HARD FAIL"`` or ``"n/a"``.
        """
        if not self.measured:
            return "n/a"
        if self.passed:
            return "pass"
        return "HARD FAIL" if self.hard else "FAIL"


class GateReport(BaseModel):
    """Everything the gate decided, and why."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="False when the gate is off.")
    passed: bool = Field(default=True, description="Whether the build may proceed.")
    checks: list[GateCheck] = Field(
        default_factory=list, description="One entry per threshold, in table order."
    )
    item_failures: list[str] = Field(
        default_factory=list,
        description="Per-item hard failures, e.g. an ACL leak on one golden item.",
    )
    unmeasured: list[str] = Field(
        default_factory=list, description="Metrics no item produced a score for."
    )

    @property
    def hard_failures(self) -> list[GateCheck]:
        """Hard thresholds that were breached.

        Returns:
            The failing hard checks.
        """
        return [check for check in self.checks if check.hard and not check.passed]

    @property
    def soft_failures(self) -> list[GateCheck]:
        """Soft thresholds that were breached.

        Returns:
            The failing soft checks.
        """
        return [check for check in self.checks if not check.hard and not check.passed]

    def summary(self) -> str:
        """One-line verdict.

        Returns:
            A short human-readable summary.
        """
        if not self.enabled:
            return "eval gate disabled (RAG_EVAL_GATE_ENABLED=false)"
        if self.passed:
            return "eval gate passed"
        parts = []
        if self.hard_failures:
            parts.append(f"{len(self.hard_failures)} hard")
        if self.soft_failures:
            parts.append(f"{len(self.soft_failures)} soft")
        if self.item_failures:
            parts.append(f"{len(self.item_failures)} item")
        return "eval gate FAILED: " + ", ".join(parts) + " failure(s)"


def gate_thresholds(settings: Settings | None = None) -> list[Threshold]:
    """Build the ordered threshold list from settings.

    Args:
        settings: Resolved settings.

    Returns:
        Every condition the gate applies, hard ones first.
    """
    cfg = settings or get_settings()
    hard_acl = float(eval_setting(cfg, "eval_hard_min_acl_leak"))
    hard_refusal = float(eval_setting(cfg, "eval_hard_min_refusal_correct"))
    # eval_max_acl_leak is expressed as a tolerated leak *rate*; the metric is
    # scored the other way up (1.0 == clean), so the floor is its complement.
    configured_acl = 1.0 - float(cfg.eval_max_acl_leak)
    return [
        Threshold(
            metric="acl_leak",
            limit=max(configured_acl, hard_acl),
            hard=True,
            source="eval_max_acl_leak / eval_hard_min_acl_leak",
        ),
        Threshold(
            metric="refusal_correct",
            limit=max(float(cfg.eval_min_refusal_correct), hard_refusal),
            hard=True,
            source="eval_min_refusal_correct / eval_hard_min_refusal_correct",
        ),
        Threshold(
            metric="faithfulness",
            limit=float(cfg.eval_min_faithfulness),
            source="eval_min_faithfulness",
        ),
        Threshold(
            metric="answer_relevancy",
            limit=float(cfg.eval_min_answer_relevancy),
            source="eval_min_answer_relevancy",
        ),
        Threshold(
            metric="context_precision",
            limit=float(cfg.eval_min_context_precision),
            source="eval_min_context_precision",
        ),
        Threshold(
            metric="context_recall",
            limit=float(cfg.eval_min_context_recall),
            source="eval_min_context_recall",
        ),
        Threshold(
            metric="answer_correctness",
            limit=float(cfg.eval_min_answer_correctness),
            source="eval_min_answer_correctness",
        ),
        Threshold(
            metric="semantic_similarity",
            limit=float(cfg.eval_min_semantic_similarity),
            source="eval_min_semantic_similarity",
        ),
        Threshold(
            metric="citation_validity",
            limit=float(cfg.eval_min_citation_validity),
            source="eval_min_citation_validity",
        ),
        Threshold(
            metric="tool_correct",
            limit=float(eval_setting(cfg, "eval_min_tool_correct")),
            source="eval_min_tool_correct",
        ),
        Threshold(
            metric="retrieval_recall",
            limit=float(eval_setting(cfg, "eval_min_retrieval_recall")),
            source="eval_min_retrieval_recall",
        ),
        Threshold(
            metric="latency_ms",
            limit=float(cfg.eval_max_latency_ms),
            direction="max",
            source="eval_max_latency_ms",
        ),
    ]


def _item_hard_failures(run: EvalRun) -> list[str]:
    """Collect per-item breaches of a hard rule.

    An aggregate mean can hide one catastrophic item: forty clean answers and one
    cross-tenant leak still averages 0.976. ACL leaks are therefore also checked
    item by item.

    Args:
        run: The completed run.

    Returns:
        One string per offending item.
    """
    failures: list[str] = []
    for result in run.results:
        leak = result.scores.acl_leak
        if leak is not None and leak < 1.0:
            reasons = [
                reason
                for reason in result.failures
                if reason.startswith(("acl_leak:", "forbidden_present:"))
            ]
            detail = ", ".join(reasons) if reasons else "acl_leak below 1.0"
            failures.append(f"{result.item_id}: {detail}")
    return failures


def evaluate_gate(
    run: EvalRun,
    *,
    settings: Settings | None = None,
) -> GateReport:
    """Apply every threshold to a completed run.

    Args:
        run: The run to judge. ``run.aggregate`` must already be computed.
        settings: Resolved settings.

    Returns:
        A :class:`GateReport`. When the gate is disabled the checks are still
        evaluated and printed; only the verdict is forced to pass.
    """
    cfg = settings or get_settings()
    aggregate = run.aggregate or run.compute_aggregate()
    checks: list[GateCheck] = []
    unmeasured: list[str] = []

    for threshold in gate_thresholds(cfg):
        raw = aggregate.get(threshold.metric)
        measured = raw is not None
        value = float(raw) if measured else None
        if not measured:
            unmeasured.append(threshold.metric)
            passed = True
        elif threshold.direction == "max":
            passed = value <= threshold.limit
        else:
            passed = value >= threshold.limit
        checks.append(
            GateCheck(
                metric=threshold.metric,
                limit=threshold.limit,
                direction=threshold.direction,
                value=value,
                measured=measured,
                passed=passed,
                hard=threshold.hard,
                source=threshold.source,
            )
        )

    item_failures = _item_hard_failures(run)
    enabled = bool(cfg.eval_gate_enabled)
    clean = all(check.passed for check in checks) and not item_failures
    report = GateReport(
        enabled=enabled,
        passed=clean or not enabled,
        checks=checks,
        item_failures=item_failures,
        unmeasured=unmeasured,
    )
    _log.info(
        "eval_gate_evaluated",
        enabled=enabled,
        passed=report.passed,
        hard_failures=len(report.hard_failures),
        soft_failures=len(report.soft_failures),
        item_failures=len(item_failures),
    )
    return report


def apply_gate(
    run: EvalRun,
    *,
    settings: Settings | None = None,
) -> GateReport:
    """Evaluate the gate and record the verdict on the run.

    Args:
        run: The run to judge; ``gate_passed`` is set in place.
        settings: Resolved settings.

    Returns:
        The :class:`GateReport`.
    """
    report = evaluate_gate(run, settings=settings)
    run.gate_passed = report.passed
    return report


def _cell(value: str, width: int) -> str:
    """Pad a table cell.

    Args:
        value: Cell text.
        width: Column width.

    Returns:
        The left-aligned, padded text.
    """
    return value.ljust(width)


def format_gate_table(report: GateReport, *, run: EvalRun | None = None) -> str:
    """Render the gate outcome as a readable fixed-width table.

    Args:
        report: The gate report.
        run: The run, for the summary line's item counts.

    Returns:
        The table plus a verdict line, ready for a CI log.
    """
    headers = ("metric", "value", "threshold", "status", "source")
    rows: list[tuple[str, str, str, str, str]] = []
    for check in report.checks:
        comparator = "<=" if check.direction == "max" else ">="
        value = "not measured" if not check.measured else f"{check.value:.4f}"
        rows.append(
            (
                check.metric,
                value,
                f"{comparator} {check.limit:.4f}",
                check.status,
                check.source,
            )
        )

    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        if rows
        else len(headers[column])
        for column in range(len(headers))
    ]
    lines = [
        "  ".join(_cell(headers[column], widths[column]) for column in range(5)),
        "  ".join("-" * widths[column] for column in range(5)),
    ]
    lines.extend(
        "  ".join(_cell(row[column], widths[column]) for column in range(5))
        for row in rows
    )

    if report.item_failures:
        lines.append("")
        lines.append("hard per-item failures:")
        lines.extend(f"  {failure}" for failure in report.item_failures)
    if report.unmeasured:
        lines.append("")
        lines.append("not measured: " + ", ".join(report.unmeasured))
    if run is not None:
        lines.append("")
        lines.append(
            f"items {run.item_count}  passed {run.passed_count}  "
            f"failed {run.failed_count}  pass_rate {run.pass_rate:.2%}  "
            f"cost ${run.total_cost_usd:.4f}"
        )
    lines.append("")
    lines.append(report.summary())
    return "\n".join(lines)


def _load_run(path: Path) -> EvalRun:
    """Read an :class:`~ragcore.models.eval.EvalRun` from a run artefact file.

    Accepts both the harness's artefact envelope (``{"run": {...}, ...}``) and a
    bare serialised run, so a hand-trimmed file still gates.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed run.

    Raises:
        ValueError: If the document holds no recognisable run.
    """
    document: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, dict) and "run" in document:
        document = document["run"]
    if not isinstance(document, dict):
        msg = f"{path} does not contain an EvalRun object"
        raise ValueError(msg)
    return EvalRun.model_validate(document)


async def run_and_gate(
    *,
    offline: bool = False,
    golden_path: str | Path | None = None,
    categories: Sequence[str] | None = None,
    limit: int | None = None,
    report_dir: str | Path | None = None,
    settings: Settings | None = None,
) -> EvalRunArtifacts:
    """Execute the golden set and apply the gate to the run it produced.

    Args:
        offline: Run inside :func:`eval.harness.offline_environment`, which fakes
            Anthropic, Qdrant and the embedder and leaves everything that decides
            what a persona may see untouched.
        golden_path: Golden file; defaults to ``eval_golden_path``.
        categories: Restrict the run to these categories.
        limit: Evaluate at most this many items.
        report_dir: Where the JSON/markdown/HTML artefacts land.
        settings: Resolved settings.

    Returns:
        The completed :class:`~eval.report.EvalRunArtifacts`, ``gate`` populated.
    """
    # Imported lazily: `eval.run_eval` imports this module, so a module-scope import
    # either way round would be a cycle.
    from eval.run_eval import run_eval

    common: dict[str, Any] = {
        "golden_path": golden_path,
        "categories": categories,
        "limit": limit,
        "report_dir": report_dir,
        "gate": True,
    }
    if not offline:
        return await run_eval(settings=settings, **common)

    from eval.harness import offline_environment

    with offline_environment(settings=settings) as harness:
        return await run_eval(
            settings=harness.settings, runner=harness, persist=False, **common
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the standalone gate CLI arguments.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="python -m eval.ci_gate",
        description=(
            "Apply the evaluation thresholds to a run written by `python -m "
            "eval.run_eval`, or execute the golden set first with --run/--offline. "
            "Exits non-zero when the gate fails."
        ),
    )
    parser.add_argument(
        "run",
        nargs="?",
        default=None,
        help="path to a run JSON file (the artefact envelope or a bare EvalRun)",
    )
    parser.add_argument(
        "--run",
        dest="execute",
        action="store_true",
        help="execute the golden set against the live stack, then gate it",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "execute the golden set against the offline fixture (faked Anthropic, "
            "Qdrant and embedder; real orchestrator, guardrails and ACL filters)"
        ),
    )
    parser.add_argument("--golden", default=None, help="path to the golden set YAML")
    parser.add_argument(
        "--category",
        action="append",
        default=None,
        metavar="NAME",
        help="only run this category; repeatable",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="evaluate at most N items"
    )
    parser.add_argument(
        "--report-dir", default=None, help="directory for the JSON/markdown/HTML"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Gate a previously written run, or execute the golden set and gate that.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` when the gate passes, ``1`` when it fails, ``2`` on an error.
    """
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings)

    execute = args.execute or args.offline
    if execute and args.run:
        print("error: pass a run file or --run/--offline, not both", file=sys.stderr)
        return EXIT_FAILED
    if not execute and not args.run:
        print(
            "error: pass a run file, or --run/--offline to execute the golden set",
            file=sys.stderr,
        )
        return EXIT_FAILED

    if execute:
        try:
            artifacts = asyncio.run(
                run_and_gate(
                    offline=args.offline,
                    golden_path=args.golden,
                    categories=args.category,
                    limit=args.limit,
                    report_dir=args.report_dir,
                    settings=settings,
                )
            )
        except Exception as exc:  # the CLI reports, it does not traceback
            _log.exception("eval_gate_run_failed")
            print(f"error: {exc.__class__.__name__}: {exc}", file=sys.stderr)
            return EXIT_FAILED
        run = artifacts.run
        report = artifacts.gate or apply_gate(run, settings=settings)
    else:
        try:
            run = _load_run(Path(args.run))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_FAILED
        report = apply_gate(run, settings=settings)

    print(format_gate_table(report, run=run))
    return EXIT_OK if report.passed else EXIT_GATE_FAILED


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
