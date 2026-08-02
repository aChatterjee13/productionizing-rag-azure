"""The validation layer actually executes.

Before this suite existed the harness could not run at all: the documented
``eval_pipeline_target`` named ``app.rag.orchestrator:run_turn``, which the module
has never defined, so every run died with ``EvalHarnessError: ... is not callable``
and the ten ``acl_negative`` golden items — the end-to-end proof that one tenant
cannot read another's documents — had never once been executed.

Everything below runs the *real* orchestrator, retriever, guardrails, ACL filters,
citation verifier and metrics. Only the things that leave the process are faked:
Anthropic, Qdrant and the two FastEmbed model downloads.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from qdrant_client import models as qm

from eval.ci_gate import run_and_gate
from eval.harness import (
    DEFAULT_PIPELINE_TARGET,
    PIPELINE_TARGET_ENV,
    offline_environment,
    resolve_pipeline_target,
)
from eval.run_eval import (
    EvalHarnessError,
    OrchestratorRunner,
    load_golden_set,
    run_eval,
)
from ragcore.models.chunk import ChunkPayload
from ragcore.models.retrieval import RetrievalStage, RetrievedChunk
from ragcore.settings import get_settings
from ragcore.vectorstore.filters import build_acl_filter

ACL_NEGATIVE = "acl_negative"
OUT_OF_DOMAIN = "out_of_domain"

#: The target `docs/CONTRACTS.md` documents. It has never been importable.
CONTRACT_TARGET = "app.rag.orchestrator:run_turn"


# ------------------------------------------------------------- pipeline target
def test_default_pipeline_target_resolves_to_a_real_coroutine() -> None:
    """The runner binds to a callable that exists and accepts what it must.

    This is the whole defect: with the documented default the same call raised
    ``EvalHarnessError: 'app.rag.orchestrator:run_turn' is not callable`` and no
    golden item ever ran.
    """
    runner = OrchestratorRunner()
    assert runner.target == DEFAULT_PIPELINE_TARGET
    function, static = runner._resolve()
    assert callable(function)
    assert static["_message_param"] == "message"
    assert static["_principal_param"] == "principal"
    assert static["settings"] is runner.settings


def test_the_contract_target_still_fails_loudly() -> None:
    """A target that cannot be imported is a wiring error naming its own fix."""
    runner = OrchestratorRunner(target=CONTRACT_TARGET)
    with pytest.raises(EvalHarnessError, match=PIPELINE_TARGET_ENV):
        runner._resolve()


def test_environment_variable_overrides_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``RAG_EVAL_PIPELINE_TARGET`` is honoured.

    ``Settings`` is ``extra="ignore"`` and has no ``eval_pipeline_target`` field, so
    the variable used to be discarded silently: there was no escape hatch at all.
    The environment beats a declared field too, which is what makes it an override.
    """
    monkeypatch.setenv(PIPELINE_TARGET_ENV, "some.module:entry_point")
    promoted = get_settings().model_copy(
        update={"eval_pipeline_target": "promoted.module:run"}
    )
    assert resolve_pipeline_target() == "some.module:entry_point"
    assert resolve_pipeline_target(settings=promoted) == "some.module:entry_point"
    assert OrchestratorRunner().target == "some.module:entry_point"


def test_explicit_target_beats_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A constructor argument wins over the environment."""
    monkeypatch.setenv(PIPELINE_TARGET_ENV, "from.env:target")
    assert resolve_pipeline_target(target="from.arg:target") == "from.arg:target"


def test_settings_field_wins_once_it_is_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A promoted ``eval_pipeline_target`` takes over with no code change here."""
    monkeypatch.delenv(PIPELINE_TARGET_ENV, raising=False)
    promoted = get_settings().model_copy(
        update={"eval_pipeline_target": "promoted.module:run"}
    )
    assert resolve_pipeline_target(settings=promoted) == "promoted.module:run"


# ------------------------------------------------------------- the API binding
def test_the_api_can_import_the_harness_it_advertises() -> None:
    """``POST /eval/runs`` answered 503 forever because this module was missing."""
    from app.routers.eval import _harness

    harness = _harness()
    assert callable(harness.run_evaluation)
    assert callable(harness.new_run_id)
    assert harness.new_run_id() != harness.new_run_id()


def test_post_eval_runs_starts_a_background_run_and_names_it() -> None:
    """The handler answers with the id it allocated, and queues the work."""
    from fastapi import BackgroundTasks

    from app.routers.eval import _execute_run, start_run
    from app.schemas.requests import EvalRunRequest
    from ragcore.models.acl import Principal

    background = BackgroundTasks()
    payload = asyncio.run(
        start_run(
            body=EvalRunRequest(sample_size=1, notes="unit"),
            principal=Principal(
                user_id="u", tenant_id="tenant-acme", roles=["rag.admin"]
            ),
            session=object(),
            settings=get_settings(),
            background=background,
        )
    )
    assert payload["run_id"].startswith("eval-")
    assert payload["results"] == []
    assert payload["finished_at"] is None
    assert len(background.tasks) == 1
    task = background.tasks[0]
    assert task.func is _execute_run
    assert task.kwargs["run_id"] == payload["run_id"]


def test_run_evaluation_files_the_run_under_the_id_the_caller_was_given(
    tmp_path: Any,
) -> None:
    """The poll target and the response agree, which is what makes polling work."""
    from eval.harness import new_run_id, run_evaluation

    allocated = new_run_id()
    with offline_environment() as harness:
        run = asyncio.run(
            run_evaluation(
                tenant_id="tenant-acme",
                sample_size=1,
                notes="unit",
                settings=harness.settings.model_copy(
                    update={"eval_report_dir": str(tmp_path)}
                ),
                runner=harness,
                run_id=allocated,
            )
        )
    assert run.run_id == allocated
    assert run.item_count == 1
    assert run.aggregate["acl_leak"] == pytest.approx(1.0)


# ------------------------------------------------------------- offline fixture
@pytest.fixture(scope="module")
def clean_run(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Run the whole golden set once, offline, and gate it.

    Args:
        tmp_path_factory: pytest's directory factory, so the run's artefacts do not
            land in the repository's report directory.

    Returns:
        The gated ``EvalRunArtifacts``.
    """
    directory = tmp_path_factory.mktemp("eval-reports")
    return asyncio.run(run_and_gate(offline=True, report_dir=directory))


def test_the_gate_runs_end_to_end_over_the_whole_golden_set(clean_run: Any) -> None:
    """The gate executes: every threshold is evaluated against a real run."""
    artifacts = clean_run
    assert artifacts.gate is not None
    metrics = {check.metric for check in artifacts.gate.checks}
    assert {"acl_leak", "refusal_correct", "citation_validity", "latency_ms"} <= metrics

    categories = {record.category for record in artifacts.diagnostics.values()}
    assert categories == {
        "in_domain",
        OUT_OF_DOMAIN,
        "pii",
        "contradiction",
        ACL_NEGATIVE,
        "tool_required",
    }
    # Every item either produced a result or was explicitly skipped with a reason.
    assert artifacts.run.item_count + len(artifacts.skipped_items) == len(
        artifacts.diagnostics
    )
    assert artifacts.run.item_count >= 50
    for item_id in artifacts.skipped_items:
        assert artifacts.diagnostics[item_id].skip_reason


def test_acl_leak_is_one_on_the_seeded_data(clean_run: Any) -> None:
    """Nothing crossed a tenant, a clearance or a deny list. The hard gate holds."""
    artifacts = clean_run
    assert artifacts.run.aggregate["acl_leak"] == pytest.approx(1.0)
    check = next(check for check in artifacts.gate.checks if check.metric == "acl_leak")
    assert check.hard and check.passed and check.status == "pass"
    assert artifacts.gate.item_failures == []

    negatives = [
        result
        for result in artifacts.run.results
        if artifacts.diagnostics[result.item_id].category == ACL_NEGATIVE
    ]
    assert len(negatives) == 10, "the ten multi-tenant isolation items must all run"
    for result in negatives:
        assert result.scores.acl_leak == pytest.approx(1.0), result.item_id
        assert not [
            failure for failure in result.failures if failure.startswith("acl_leak:")
        ]


def test_acl_negative_items_are_refused_rather_than_answered(clean_run: Any) -> None:
    """An unauthorised question is declined, and no canary reaches the answer."""
    artifacts = clean_run
    for result in artifacts.run.results:
        record = artifacts.diagnostics[result.item_id]
        if record.category != ACL_NEGATIVE:
            continue
        assert result.scores.refusal_correct == pytest.approx(1.0), result.item_id
        assert "CANARY-" not in result.answer


def test_refusal_correct_fires_for_an_out_of_domain_item(tmp_path: Any) -> None:
    """A question nothing in the corpus answers is refused, and scored as such."""
    with offline_environment() as harness:
        artifacts = asyncio.run(
            run_eval(
                item_ids=["gi-050-share-price"],
                settings=harness.settings,
                runner=harness,
                persist=False,
                report_dir=tmp_path,
            )
        )
    result = artifacts.run.results[0]
    assert result.item_id == "gi-050-share-price"
    assert artifacts.diagnostics[result.item_id].category == OUT_OF_DOMAIN
    assert artifacts.diagnostics[result.item_id].refused is True
    assert result.scores.refusal_correct == pytest.approx(1.0)
    assert "share price" not in result.answer.casefold()


def test_the_harness_reports_answer_chunks_tools_and_usage() -> None:
    """One item, one turn, and everything the scorers need comes back."""
    with offline_environment() as harness:
        items = {
            item.item_id: item for item in load_golden_set(settings=harness.settings)
        }
        item = items["gi-001-meal-allowance-current"]
        outcome = asyncio.run(harness.run_item(item))

    assert outcome.answer.strip()
    assert outcome.retrieved_chunk_ids()
    assert all("::" in chunk_id for chunk_id in outcome.retrieved_chunk_ids())
    assert set(item.expected_document_ids) <= set(outcome.retrieved_document_ids())
    assert outcome.usages and outcome.resolved_cost() > 0.0
    assert outcome.tools_invoked == []
    assert all(chunk.payload.tenant_id == item.tenant_id for chunk in outcome.chunks)


# ------------------------------------------------- the ACL filter under attack
def _tenant_blind_filter(
    principal: Any, extra: Any = None, *, include_deleted: bool = False
) -> qm.Filter:
    """Build the ACL filter with the tenant clause removed.

    Args:
        principal: The caller.
        extra: Optional metadata filter.
        include_deleted: Passed through.

    Returns:
        The real filter minus ``tenant_id`` — the regression this suite exists to
        catch.
    """
    original = build_acl_filter(principal, extra, include_deleted=include_deleted)
    must = [
        condition
        for condition in original.must or []
        if not (
            isinstance(condition, qm.FieldCondition) and condition.key == "tenant_id"
        )
    ]
    return qm.Filter(
        must=must,
        must_not=original.must_not,
        should=original.should,
        min_should=original.min_should,
    )


def _acl_blind_candidate(
    point: Any, *, principal: Any, settings: Any, ledger: Any
) -> RetrievedChunk | None:
    """Map a scored point to a candidate without re-checking the ACL.

    Args:
        point: The scored point.
        principal: The caller. Unused — that omission is the simulated defect.
        settings: Active settings. Unused.
        ledger: Drop ledger. Unused.

    Returns:
        The candidate, or None when the payload will not parse.
    """
    del principal, settings, ledger
    try:
        payload = ChunkPayload.from_qdrant_payload(point.payload)
    except ValueError:
        return None
    return RetrievedChunk(
        payload=payload,
        fusion_score=float(point.score or 0.0),
        retrieval_stage=RetrievalStage.FUSION.value,
    )


#: Golden items whose persona and target document live in different tenants.
ISOLATION_ITEMS = [
    "gi-085-globex-asks-acme-allowance",
    "gi-086-acme-asks-globex-forklift",
    "gi-089-globex-asks-acme-onboarding",
]


def _run_isolation_gate(tmp_path: Any) -> Any:
    """Run the cross-tenant golden items offline and gate them.

    Args:
        tmp_path: Directory for the run's artefacts.

    Returns:
        The gated artefacts.
    """
    with offline_environment() as harness:
        return asyncio.run(
            run_eval(
                item_ids=ISOLATION_ITEMS,
                settings=harness.settings,
                runner=harness,
                persist=False,
                report_dir=tmp_path,
            )
        )


def test_breaking_only_the_qdrant_filter_is_caught_by_the_in_process_mirror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Defence in depth: one broken layer does not leak."""
    monkeypatch.setattr("app.rag.retriever.build_acl_filter", _tenant_blind_filter)
    artifacts = _run_isolation_gate(tmp_path)
    assert artifacts.run.aggregate["acl_leak"] == pytest.approx(1.0)
    assert artifacts.gate.item_failures == []


def test_breaking_both_acl_layers_turns_the_gate_red(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """With both enforcement points weakened the gate must go hard red.

    This is what proves the ``acl_negative`` items measure something: weaken the
    tenant predicate in the Qdrant filter *and* the retriever's in-process mirror,
    and foreign chunks reach the answer, ``acl_leak`` collapses and the build stops.
    """
    monkeypatch.setattr("app.rag.retriever.build_acl_filter", _tenant_blind_filter)
    monkeypatch.setattr("app.rag.retriever._to_candidate", _acl_blind_candidate)

    artifacts = _run_isolation_gate(tmp_path)

    assert artifacts.run.aggregate["acl_leak"] < 1.0
    check = next(check for check in artifacts.gate.checks if check.metric == "acl_leak")
    assert check.hard and not check.passed and check.status == "HARD FAIL"
    assert artifacts.gate.item_failures
    assert not artifacts.gate.passed

    leaked = [
        result for result in artifacts.run.results if result.scores.acl_leak == 0.0
    ]
    assert leaked, "no item detected the leak"
    assert any(
        failure.startswith("acl_leak:cross_tenant:")
        for result in leaked
        for failure in result.failures
    )


# ------------------------------------------------------------------- the fakes
@pytest.fixture
def offline() -> Iterator[Any]:
    """An offline harness, torn down after the test.

    Yields:
        The harness.
    """
    with offline_environment() as harness:
        yield harness


def test_the_fake_store_serves_both_tenants_and_the_filter_separates_them(
    offline: Any,
) -> None:
    """The fixture is loaded with everything; only the ACL filter narrows it."""
    from app.rag.retriever import get_client

    store = asyncio.run(get_client(offline.settings))
    tenants = {payload["tenant_id"] for payload in store.payloads.values()}
    assert tenants == {"tenant-acme", "tenant-globex"}

    intern = offline.principal_for("acme_intern")
    visible = store._visible(build_acl_filter(intern))
    documents = {store.payloads[point]["document_id"] for point in visible}
    assert documents == {
        "doc-acme-travel-2023",
        "doc-acme-travel-2025",
        "doc-acme-onboarding",
    }


def test_an_unimplemented_filter_clause_fails_loudly() -> None:
    """A condition the fake cannot evaluate must raise, never be silently ignored.

    A filter clause the fake drops is a filter clause the gate stops testing.
    """
    from eval.harness import _filter_matches

    unsupported = qm.Filter(
        must=[qm.FieldCondition(key="tags", values_count=qm.ValuesCount(gte=1))]
    )
    with pytest.raises(EvalHarnessError, match="does not implement"):
        _filter_matches(unsupported, {"tenant_id": "t", "tags": ["a"]}, 1)
