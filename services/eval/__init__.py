"""Golden-set validation harness (requirement #8).

The package is importable as ``eval`` — ``services/eval`` *is* the package
directory, which is what the repository layout in `docs/CONTRACTS.md` and the ruff
``known-first-party`` list both assume.

Modules:

* :mod:`eval.run_eval` -- the async, bounded-concurrency runner and the CLI. Runs
  every golden item through the real orchestrator with the item's persona.
* :mod:`eval.metrics` -- the metrics RAGAS does not give us: citation validity,
  ACL leak, refusal correctness, tool correctness, latency and cost.
* :mod:`eval.ragas_adapter` -- RAGAS with Claude as the judge, degrading to native
  Claude-judged implementations of the same five metrics.
* :mod:`eval.semantic` -- sentence-embedding cosine similarity through
  :mod:`ragcore.embeddings`. No second embedding stack.
* :mod:`eval.ci_gate` -- per-metric thresholds, with ACL leak and refusal
  correctness as hard gates.
* :mod:`eval.report` -- markdown and self-contained HTML reports.

**Tunables.** Every threshold, limit, model name and path is a
:class:`ragcore.settings.Settings` field. The knobs introduced with this harness are
listed in :data:`EVAL_SETTING_DEFAULTS` and read through :func:`eval_setting`, which
prefers the real settings field and falls back to the documented default until that
field is declared — the same pattern ``app.rag.rag_setting`` and
``app.rag.guardrails.guardrail_setting`` use, and for the same reason:
:mod:`ragcore.settings` is owned by another component. Nothing here hard-codes a
threshold at a call site.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from ragcore.settings import Settings

if TYPE_CHECKING:  # pragma: no cover - import cycles only matter to type checkers
    from importlib.machinery import ModuleSpec

    from eval.ci_gate import GateReport, evaluate_gate, format_gate_table
    from eval.metrics import AclLeakReport, acl_leak, citation_validity, tool_correct
    from eval.ragas_adapter import RagasAdapter, RagasSample, RagasScores
    from eval.report import write_reports
    from eval.run_eval import EvalRunArtifacts, run_eval
    from eval.semantic import semantic_similarity

__all__ = [
    "EVAL_SETTING_DEFAULTS",
    "AclLeakReport",
    "EvalRunArtifacts",
    "GateReport",
    "RagasAdapter",
    "RagasSample",
    "RagasScores",
    "acl_leak",
    "citation_validity",
    "eval_setting",
    "evaluate_gate",
    "format_gate_table",
    "run_eval",
    "semantic_similarity",
    "tool_correct",
    "write_reports",
]

#: Tunables this harness needs that ``ragcore.settings`` does not declare yet, with
#: the value that applies until it does. Keys are real ``RAG_``-prefixed setting
#: names, so exporting ``RAG_EVAL_REPORT_DIR`` starts working the moment the field
#: exists on :class:`~ragcore.settings.Settings`. Documented in Addendum E of
#: `docs/CONTRACTS.md`.
EVAL_SETTING_DEFAULTS: dict[str, Any] = {
    # ------------------------------------------------------------------ wiring
    # ``module:attribute`` of the coroutine one golden item is run through. The
    # harness must exercise the *real* pipeline, so this points at the wrapper
    # that drives the orchestrator through all thirteen stages, not at an HTTP
    # client or a stub.
    #
    # `docs/CONTRACTS.md` (Addendum E) documents this default as
    # `app.rag.orchestrator:run_turn`, but `app.rag.orchestrator` exposes no
    # module-level `run_turn` — the orchestrator's entry point is the
    # `Orchestrator` class, so that string raises AttributeError on resolution.
    # `eval.harness:run_turn` is the callable that actually satisfies the
    # binding contract, and it is what `Settings.eval_pipeline_target` defaults
    # to. Kept in sync here so the shim and the settings field cannot disagree.
    "eval_pipeline_target": "eval.harness:run_turn",
    # Persona file resolved relative to the repository root when relative.
    "eval_personas_path": "services/eval/golden/personas.yaml",
    # Where run JSON, markdown and HTML land. CI uploads this directory.
    "eval_report_dir": "services/eval/reports",
    # Wall-clock ceiling for one golden item, so one wedged turn cannot hang a run.
    "eval_item_timeout_seconds": 300.0,
    # Persist EvalRun/EvalResult rows. A database failure degrades to a warning:
    # the gate's verdict must not depend on the reporting store being up.
    "eval_persist_results": True,
    # Tenant recorded on the run row. None means "the tenant most golden items in
    # this run belong to", which is what a single-tenant CI run wants.
    "eval_run_tenant_id": None,
    # Skip (rather than fail) a ``tool_required`` item whose expected tool is not
    # registered for that persona. A REST/MCP tool needs a `tools.yaml` and live
    # credentials; a build without them should report the gap, not a red gate.
    "eval_skip_unregistered_tools": True,
    # Run the ``tool_required`` items whose tool is a REST or MCP tool. Off by
    # default: the example registry points at unreachable hosts, and an answer
    # produced from a failed tool call is indistinguishable in a score from one the
    # model made up. Set it once the registry points at live backends. The built-in
    # tools (``search_corpus``, ``current_context``) always run.
    "eval_tool_required_live": False,
    # ------------------------------------------------------------------- RAGAS
    # Use the RAGAS package for the LLM-only metrics when it imports cleanly.
    "eval_ragas_enabled": True,
    # Effort for every judge call. Judging is a classification job, not a
    # reasoning showcase, and the judge runs once per metric per item.
    "eval_judge_effort": "medium",
    # Questions generated from the answer for the native answer-relevancy metric,
    # each embedded and compared with the real question.
    "eval_relevancy_probe_questions": 3,
    # RAGAS answer-correctness weighting: statement F1 against ground truth, plus
    # embedding similarity. The two weights must sum to 1.0.
    "eval_correctness_f1_weight": 0.75,
    "eval_correctness_similarity_weight": 0.25,
    # Contexts handed to a judge call, and the per-context character cap. Judge
    # prompts are the harness's main token cost.
    "eval_judge_max_contexts": 12,
    "eval_judge_max_context_chars": 4000,
    # ------------------------------------------------------------------- gates
    # Hard floors. ``eval_min_*`` settings may tighten these but never loosen
    # them: an ACL leak or a wrong refusal is a correctness failure, not a quality
    # dip, so the build fails at anything below them.
    "eval_hard_min_acl_leak": 1.0,
    "eval_hard_min_refusal_correct": 0.95,
    # Soft floor for ``tool_correct``; absent from the gate when nothing measured
    # it (every tool item skipped). Deliberately lenient: whether a model reaches
    # for a tool when the prompt already carries the answer is a behavioural
    # preference, not a correctness property. Raise it once the tool registry
    # points at live backends and ``eval_tool_required_live`` is on.
    "eval_min_tool_correct": 0.5,
    # Aggregate floor for expected-document retrieval recall.
    "eval_min_retrieval_recall": 0.7,
    # A baseline comparison flags a metric as a regression once it drops by more
    # than this. Below it, the move is noise from a non-deterministic judge.
    "eval_regression_tolerance": 0.02,
    # ------------------------------------------------------------------ report
    # Worst-performing items shown in full, with their trace ids.
    "eval_report_worst_items": 10,
    # Characters of the emitted answer stored on a result. The answer has already
    # been through the stage 12 PII egress scan; clipping keeps a report readable
    # and keeps a run artefact from becoming a corpus copy.
    "eval_report_answer_chars": 600,
    # Prefix for every Langfuse score name pushed by the harness.
    "eval_score_prefix": "eval.",
}

#: Where each lazily re-exported symbol lives.
_LAZY_EXPORTS: dict[str, str] = {
    "AclLeakReport": "eval.metrics",
    "acl_leak": "eval.metrics",
    "citation_validity": "eval.metrics",
    "tool_correct": "eval.metrics",
    "RagasAdapter": "eval.ragas_adapter",
    "RagasSample": "eval.ragas_adapter",
    "RagasScores": "eval.ragas_adapter",
    "semantic_similarity": "eval.semantic",
    "GateReport": "eval.ci_gate",
    "evaluate_gate": "eval.ci_gate",
    "format_gate_table": "eval.ci_gate",
    "write_reports": "eval.report",
    "EvalRunArtifacts": "eval.run_eval",
    "run_eval": "eval.run_eval",
}

#: Module name ``.github/workflows/ci.yml`` and the root ``Makefile`` invoke.
_RUN_ALIAS = "eval.run"


def eval_setting(settings: Settings, name: str) -> Any:
    """Read a harness tunable, preferring the settings field over the default.

    Args:
        settings: Active settings object.
        name: A key of :data:`EVAL_SETTING_DEFAULTS`, e.g. ``"eval_report_dir"``.

    Returns:
        ``getattr(settings, name)`` when the field exists and is set, otherwise the
        documented default.

    Raises:
        KeyError: If ``name`` is not a declared tunable. That is a typo at the call
            site, not a configuration problem, so it fails loudly.
    """
    if name not in EVAL_SETTING_DEFAULTS:
        msg = f"unknown eval tunable {name!r}"
        raise KeyError(msg)
    value = getattr(settings, name, None)
    return EVAL_SETTING_DEFAULTS[name] if value is None else value


class _RunModuleAliasFinder:
    """Meta-path finder that resolves ``eval.run`` to ``eval/run_eval.py``.

    ``python -m eval.run`` is the command the CI workflow and the ``make eval``
    target invoke, while the file `docs/CONTRACTS.md` names is ``run_eval.py``.
    Rather than ship a second module that re-exports the first — two importable
    copies of the same CLI is how the two drift apart — the alias is resolved by
    the import system: ``-m`` re-executes the source file as ``__main__``, which is
    exactly what it does for any module.
    """

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        """Resolve the alias, and nothing else.

        Args:
            fullname: Fully qualified module name being imported.
            path: Parent package search path. Unused.
            target: Module being reloaded, if any. Unused.

        Returns:
            A spec for ``eval/run_eval.py`` under the aliased name, or None so the
            normal finders handle every other module.
        """
        del path, target
        if fullname != _RUN_ALIAS:
            return None
        source = Path(__file__).with_name("run_eval.py")
        if not source.is_file():  # pragma: no cover - only in a broken install
            return None
        return importlib.util.spec_from_file_location(fullname, source)


def _install_run_alias() -> None:
    """Register the ``eval.run`` alias exactly once."""
    if any(isinstance(finder, _RunModuleAliasFinder) for finder in sys.meta_path):
        return
    sys.meta_path.append(_RunModuleAliasFinder())


_install_run_alias()


def __getattr__(name: str) -> Any:
    """Resolve a lazily re-exported submodule symbol.

    Importing :mod:`eval` must stay cheap: the API service imports
    :mod:`eval.run_eval` to serve ``POST /eval/runs``, and the CLI imports only what
    it needs.

    Args:
        name: Attribute being imported from this package.

    Returns:
        The resolved object.

    Raises:
        AttributeError: If the name is not part of the public surface.
    """
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the public surface, including the lazily loaded names.

    Returns:
        Sorted attribute names.
    """
    return sorted(set(__all__) | set(globals()))
