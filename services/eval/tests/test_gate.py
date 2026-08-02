from __future__ import annotations

from datetime import UTC, datetime

import pytest

from eval.ci_gate import (
    HARD_METRICS,
    apply_gate,
    evaluate_gate,
    format_gate_table,
    gate_thresholds,
)
from eval.report import (
    EvalItemDiagnostics,
    EvalRunArtifacts,
    category_aggregate,
    compare_runs,
    render_html,
    render_markdown,
    worst_items,
    write_reports,
)
from ragcore.models.eval import EvalResult, EvalRun, MetricScores
from ragcore.settings import get_settings


@pytest.fixture
def settings():
    return get_settings()


def _result(
    item_id: str,
    *,
    passed: bool = True,
    acl_leak: float = 1.0,
    refusal_correct: float = 1.0,
    faithfulness: float | None = 0.95,
    failures: list[str] | None = None,
) -> EvalResult:
    return EvalResult(
        item_id=item_id,
        answer="an answer",
        retrieved_chunk_ids=["doc-acme-travel-2025::0000"],
        scores=MetricScores(
            faithfulness=faithfulness,
            answer_relevancy=0.9,
            context_precision=0.9,
            context_recall=0.9,
            answer_correctness=0.9,
            semantic_similarity=0.9,
            citation_validity=0.95,
            acl_leak=acl_leak,
            refusal_correct=refusal_correct,
            latency_ms=1200.0,
            cost_usd=0.01,
        ),
        passed=passed,
        failures=failures or [],
        trace_id=f"trace-{item_id}",
    )


def _run(results: list[EvalResult], *, run_id: str = "eval-test") -> EvalRun:
    run = EvalRun(
        run_id=run_id,
        started_at=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
        finished_at=datetime(2026, 8, 1, 12, 5, tzinfo=UTC),
        git_sha="abc123",
        config_fingerprint="fingerprint",
        results=results,
    )
    run.compute_aggregate()
    return run


def _diagnostics(run: EvalRun, category: str = "in_domain") -> dict:
    return {
        result.item_id: EvalItemDiagnostics(
            item_id=result.item_id,
            category=category,
            persona="acme_engineer",
            tenant_id="tenant-acme",
            question="q?",
            tool_correct=1.0,
            retrieval_recall=1.0,
        )
        for result in run.results
    }


# ------------------------------------------------------------------- thresholds
def test_acl_leak_and_refusal_are_the_hard_gates(settings):
    thresholds = {
        threshold.metric: threshold for threshold in gate_thresholds(settings)
    }
    hard = {name for name, threshold in thresholds.items() if threshold.hard}
    assert hard == set(HARD_METRICS)
    assert thresholds["acl_leak"].limit == pytest.approx(1.0)
    assert thresholds["refusal_correct"].limit >= 0.95


def test_configuration_can_tighten_a_hard_gate_but_not_loosen_it(settings):
    loosened = settings.model_copy(
        update={"eval_min_refusal_correct": 0.10, "eval_max_acl_leak": 0.5}
    )
    thresholds = {
        threshold.metric: threshold for threshold in gate_thresholds(loosened)
    }
    assert thresholds["refusal_correct"].limit >= 0.95
    assert thresholds["acl_leak"].limit == pytest.approx(1.0)

    tightened = settings.model_copy(update={"eval_min_refusal_correct": 0.99})
    thresholds = {
        threshold.metric: threshold for threshold in gate_thresholds(tightened)
    }
    assert thresholds["refusal_correct"].limit == pytest.approx(0.99)


def test_latency_is_a_maximum_not_a_minimum(settings):
    thresholds = {
        threshold.metric: threshold for threshold in gate_thresholds(settings)
    }
    assert thresholds["latency_ms"].direction == "max"


# ------------------------------------------------------------------------- gate
def test_clean_run_passes(settings):
    run = _run([_result("a"), _result("b")])
    report = evaluate_gate(run, settings=settings)
    assert report.passed
    assert not report.hard_failures
    assert not report.item_failures


def test_single_acl_leak_fails_the_build_even_when_the_mean_looks_fine(settings):
    results = [_result(f"ok-{index}") for index in range(40)]
    results.append(
        _result(
            "leak",
            passed=False,
            acl_leak=0.0,
            failures=["acl_leak:cross_tenant:c1"],
        )
    )
    run = _run(results)
    # The mean is still 40/41 = 0.976, which a naive threshold would wave through.
    assert run.aggregate["acl_leak"] > 0.97
    report = evaluate_gate(run, settings=settings)
    assert not report.passed
    assert [check.metric for check in report.hard_failures] == ["acl_leak"]
    assert any("leak" in failure for failure in report.item_failures)


def test_refusal_regression_below_the_hard_floor_fails(settings):
    results = [_result(f"ok-{index}") for index in range(9)]
    results.append(_result("wrong-refusal", passed=False, refusal_correct=0.0))
    run = _run(results)
    report = evaluate_gate(run, settings=settings)
    assert not report.passed
    assert {check.metric for check in report.hard_failures} == {"refusal_correct"}


def test_unmeasured_metric_is_reported_but_does_not_fail(settings):
    run = _run([_result("a", faithfulness=None)])
    report = evaluate_gate(run, settings=settings)
    assert "faithfulness" in report.unmeasured
    assert report.passed
    check = next(check for check in report.checks if check.metric == "faithfulness")
    assert check.status == "n/a"


def test_disabled_gate_still_evaluates_but_passes(settings):
    disabled = settings.model_copy(update={"eval_gate_enabled": False})
    run = _run([_result("leak", passed=False, acl_leak=0.0)])
    report = evaluate_gate(run, settings=disabled)
    assert report.passed
    assert report.enabled is False
    assert report.hard_failures  # the breach is still visible


def test_apply_gate_records_the_verdict(settings):
    run = _run([_result("leak", passed=False, acl_leak=0.0)])
    report = apply_gate(run, settings=settings)
    assert run.gate_passed is report.passed is False


def test_gate_table_is_readable(settings):
    run = _run([_result("leak", passed=False, acl_leak=0.0, failures=["acl_leak:x:y"])])
    table = format_gate_table(apply_gate(run, settings=settings), run=run)
    assert "HARD FAIL" in table
    assert "acl_leak" in table
    assert "eval gate FAILED" in table
    assert "items 1" in table


# ----------------------------------------------------------------------- report
def test_category_aggregate_keeps_categories_separate():
    run = _run([_result("a"), _result("b", passed=False, acl_leak=0.0)])
    diagnostics = _diagnostics(run)
    diagnostics["b"].category = "acl_negative"
    aggregate = category_aggregate(run, diagnostics)
    assert set(aggregate) == {"in_domain", "acl_negative"}
    assert aggregate["acl_negative"]["acl_leak"] == pytest.approx(0.0)
    assert aggregate["in_domain"]["pass_rate"] == pytest.approx(1.0)
    assert aggregate["in_domain"]["tool_correct"] == pytest.approx(1.0)


def test_worst_items_puts_failures_first():
    run = _run([_result("good"), _result("bad", passed=False, faithfulness=0.1)])
    assert [result.item_id for result in worst_items(run, limit=2)] == ["bad", "good"]


def test_compare_runs_flags_regressions(settings):
    baseline = _run([_result("a"), _result("b")], run_id="eval-baseline")
    current = _run(
        [_result("a"), _result("b", passed=False, faithfulness=0.10)],
        run_id="eval-current",
    )
    comparison = compare_runs(
        current, baseline, diagnostics=_diagnostics(current), settings=settings
    )
    assert comparison.baseline_run_id == "eval-baseline"
    assert [delta.item_id for delta in comparison.regressions] == ["b"]
    faithfulness = next(
        delta for delta in comparison.metrics if delta.metric == "faithfulness"
    )
    assert faithfulness.regression is True
    assert comparison.has_regressions


def test_compare_runs_ignores_noise_below_the_tolerance(settings):
    baseline = _run([_result("a", faithfulness=0.900)], run_id="eval-baseline")
    current = _run([_result("a", faithfulness=0.895)], run_id="eval-current")
    comparison = compare_runs(current, baseline, settings=settings)
    faithfulness = next(
        delta for delta in comparison.metrics if delta.metric == "faithfulness"
    )
    assert faithfulness.regression is False
    assert faithfulness.improvement is False


def _artifacts(run: EvalRun, settings) -> EvalRunArtifacts:
    diagnostics = _diagnostics(run)
    return EvalRunArtifacts(
        run=run,
        diagnostics=diagnostics,
        category_aggregate=category_aggregate(run, diagnostics),
        golden_path="services/eval/golden/golden_set.yaml",
        personas_path="services/eval/golden/personas.yaml",
        gate=apply_gate(run, settings=settings),
        skipped_items=["gi-092-order-status"],
    )


def test_markdown_report_covers_every_section(settings):
    run = _run([_result("a"), _result("b", passed=False, acl_leak=0.0)])
    artifacts = _artifacts(run, settings)
    markdown = render_markdown(artifacts, settings=settings)
    assert "# Evaluation run" in markdown
    assert "## Aggregate" in markdown
    assert "## Per category" in markdown
    assert "Worst" in markdown
    assert "Skipped items" in markdown
    assert "trace-b" in markdown


def test_html_report_is_self_contained(settings):
    run = _run([_result("a")])
    html_text = render_html(_artifacts(run, settings), settings=settings)
    assert html_text.startswith("<!doctype html>")
    assert "<style>" in html_text
    for forbidden in ("http://", "https://", "<script"):
        assert forbidden not in html_text
    assert "prefers-color-scheme" in html_text


def test_html_report_escapes_content(settings):
    run = _run([_result("<script>alert(1)</script>", passed=False)])
    html_text = render_html(_artifacts(run, settings), settings=settings)
    assert "<script>alert(1)</script>" not in html_text
    assert "&lt;script&gt;" in html_text


def test_write_reports_emits_three_files_plus_latest(tmp_path, settings):
    run = _run([_result("a")])
    paths = write_reports(
        _artifacts(run, settings), directory=tmp_path, settings=settings
    )
    for path in (paths.json_path, paths.markdown_path, paths.html_path):
        assert path.startswith(str(tmp_path))
    assert (tmp_path / "latest.json").is_file()
    assert (tmp_path / "latest.md").is_file()
    assert (tmp_path / "latest.html").is_file()


def test_run_artifacts_round_trip(tmp_path, settings):
    from eval.report import load_artifacts

    run = _run([_result("a")])
    paths = write_reports(
        _artifacts(run, settings), directory=tmp_path, settings=settings
    )
    restored = load_artifacts(paths.json_path)
    assert restored.run.run_id == run.run_id
    assert restored.run.aggregate["acl_leak"] == pytest.approx(1.0)
    assert restored.gate is not None
