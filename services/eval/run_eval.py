"""The golden-set runner: every item, through the real pipeline, as its persona.

What this module guarantees, and why each one matters:

* **The real orchestrator.** An eval harness that re-implements retrieval measures
  the harness. Each item is executed through the coroutine
  :func:`eval.harness.resolve_pipeline_target` names — by default
  :func:`eval.harness.run_turn`, which drives
  :meth:`app.rag.orchestrator.Orchestrator.run` — with the persona's
  :class:`~ragcore.models.acl.Principal`, so the ACL filter, the guardrails, the
  tool loop and the citation verifier are the production ones.
* **Bounded concurrency.** ``eval_max_concurrency`` items run at once; each item is
  bounded by ``eval_item_timeout_seconds`` so one wedged turn cannot hang a run.
* **Traceability.** Every item opens a Langfuse trace, the turn's spans nest inside
  it, and every metric is pushed back as a score on that trace. A regression is
  therefore attributable to one item of one run, not just to "the eval got worse".
* **No unredacted content.** Answers are read back post-stage-12, and the harness
  stores a clipped preview, never the retrieved chunk text.

Both entry points live here: :func:`run_eval` for ``POST /eval/runs``, and
:func:`main` for ``python -m eval.run_eval`` (also reachable as ``python -m
eval.run``, the spelling CI uses).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import yaml

from eval import eval_setting
from eval.ci_gate import (
    EXIT_FAILED,
    EXIT_GATE_FAILED,
    EXIT_OK,
    apply_gate,
    format_gate_table,
    gate_thresholds,
)
from eval.metrics import (
    EvalDependencyError,
    acl_leak,
    citation_validity,
    cost_from_usages,
    refusal_correct,
    retrieval_recall,
    substring_failures,
    tool_correct,
)
from eval.ragas_adapter import RagasSample, get_ragas_adapter
from eval.report import (
    EvalItemDiagnostics,
    EvalRunArtifacts,
    RunComparison,
    category_aggregate,
    compare_runs,
    load_artifacts,
    write_reports,
)
from eval.semantic import semantic_similarity
from ragcore.errors import ConfigError
from ragcore.logging import configure_logging, get_logger
from ragcore.models.acl import Principal
from ragcore.models.eval import (
    EvalCategory,
    EvalResult,
    EvalRun,
    GoldenItem,
    MetricScores,
)
from ragcore.observability import get_tracer
from ragcore.settings import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Mapping, Sequence

    from ragcore.llm import LLMUsage
    from ragcore.models.chat import GuardrailEvent
    from ragcore.models.retrieval import Citation, RetrievedChunk

__all__ = [
    "EvalHarnessError",
    "EvalItemDiagnostics",
    "EvalRunArtifacts",
    "OrchestratorRunner",
    "PipelineRunner",
    "TurnOutcome",
    "config_fingerprint",
    "load_golden_set",
    "load_personas",
    "main",
    "new_run_id",
    "run_eval",
    "select_items",
]

_log = get_logger(__name__)

#: Repository root, used to resolve the relative paths in settings.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Settings prefixes that change what a run measures. Their values are hashed into
#: ``EvalRun.config_fingerprint`` so a score move can be attributed to config.
_FINGERPRINT_PREFIXES: tuple[str, ...] = (
    "anthropic_model",
    "chunk_",
    "context_",
    "embedding_",
    "guardrail_",
    "qt_",
    "rerank_",
    "retrieval_",
    "tool_",
)

#: Attribute names a turn result might use for each thing the harness needs.
_ANSWER_ATTRS = ("answer", "text", "content", "output", "final_answer")
_CHUNK_ATTRS = ("chunks", "retrieved_chunks", "sources", "context_chunks")
_CITATION_ATTRS = ("citations",)
_TOOL_ATTRS = ("tool_calls", "tools_invoked", "tool_invocations", "tools")
_USAGE_ATTRS = ("usages", "usage", "llm_usages")
_MESSAGE_PARAMS = ("message", "question", "query", "text", "user_message", "prompt")
_PRINCIPAL_PARAMS = ("principal", "user", "caller", "identity")

#: Named in the wiring errors below so an operator is told how to fix the target
#: rather than only that it is wrong. Mirrors ``eval.harness.PIPELINE_TARGET_ENV``;
#: duplicated as a literal because importing the harness at module scope would be a
#: cycle, and a variable name in an error message is not a tunable.
_TARGET_ENV = "RAG_EVAL_PIPELINE_TARGET"


class EvalHarnessError(ConfigError):
    """The harness cannot run: bad wiring, an unloadable golden set, no personas."""


@dataclass(slots=True)
class TurnOutcome:
    """What one pipeline turn produced, normalised for scoring.

    Attributes:
        answer: The answer as emitted to the user, i.e. after the stage 12 output
            guard has redacted PII.
        chunks: Chunks handed to generation, in the order they were numbered.
        citations: Citations the pipeline attached.
        tools_invoked: Names of tools the turn actually called.
        usages: Per-call token accounting.
        cost_usd: Cost the pipeline reported; derived from ``usages`` when absent.
        refused: The pipeline's own refusal signal (OOD gate or model refusal).
        trace_id: Langfuse trace id for the turn.
        citation_report: The pipeline's ``CitationReport``, when it exposes one, so
            citation validity is not recomputed with different settings.
        guardrails: Guardrail events the turn emitted.
        raw: The untouched return value, for debugging.
    """

    answer: str = ""
    chunks: list[RetrievedChunk] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    tools_invoked: list[str] = field(default_factory=list)
    usages: list[LLMUsage] = field(default_factory=list)
    cost_usd: float | None = None
    refused: bool = False
    trace_id: str | None = None
    citation_report: Any | None = None
    guardrails: list[GuardrailEvent] = field(default_factory=list)
    raw: Any = None

    def resolved_cost(self) -> float:
        """Cost of the turn in USD.

        Returns:
            The pipeline's own figure when it reported one, otherwise the sum over
            :attr:`usages`.
        """
        if self.cost_usd is not None:
            return float(self.cost_usd)
        return cost_from_usages(self.usages)

    def retrieved_chunk_ids(self) -> list[str]:
        """Chunk ids handed to generation.

        Returns:
            The ids in retrieval order.
        """
        return [chunk.payload.chunk_id for chunk in self.chunks]

    def retrieved_document_ids(self) -> list[str]:
        """Distinct document ids behind the retrieved chunks.

        Returns:
            The ids, de-duplicated, in first-seen order.
        """
        seen: dict[str, None] = {}
        for chunk in self.chunks:
            seen.setdefault(chunk.payload.document_id, None)
        return list(seen)


class PipelineRunner(Protocol):
    """Executes one golden item and returns a normalised outcome."""

    async def __call__(self, *, item: GoldenItem, principal: Principal) -> TurnOutcome:
        """Run one item.

        Args:
            item: The golden item.
            principal: The persona to run as.

        Returns:
            The turn's :class:`TurnOutcome`.
        """
        ...


def _first_attr(obj: Any, names: Sequence[str]) -> Any:
    """Read the first present, non-None attribute or mapping key.

    Args:
        obj: The object to inspect.
        names: Candidate names, most specific first.

    Returns:
        The value, or None when none of the names is populated.
    """
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _tool_names(value: Any) -> list[str]:
    """Extract tool names from whatever shape the turn reported.

    Args:
        value: A list of ``ToolCall`` objects, ``ToolResult`` objects, dicts or
            plain strings.

    Returns:
        The tool names, in call order.
    """
    if value is None:
        return []
    names: list[str] = []
    for entry in value if isinstance(value, (list, tuple)) else [value]:
        if isinstance(entry, str):
            names.append(entry)
            continue
        name = _first_attr(entry, ("tool_name", "name"))
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _usage_list(value: Any) -> list[LLMUsage]:
    """Normalise the usage field to a list.

    Args:
        value: One usage record, several, or None.

    Returns:
        A list of usage records.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [item for item in value if hasattr(item, "cost_usd")]
    return [value] if hasattr(value, "cost_usd") else []


def coerce_outcome(result: Any) -> TurnOutcome:
    """Normalise whatever the orchestrator returned into a :class:`TurnOutcome`.

    The orchestrator's return type is owned by ``services/api``; this adapter reads
    it structurally so a field rename there degrades one metric rather than the
    whole harness. A result that carries no answer at all is still an outcome — an
    empty answer scores as a refusal, which is exactly what it is.

    Args:
        result: The orchestrator's return value, or a :class:`TurnOutcome`.

    Returns:
        The normalised outcome.
    """
    if isinstance(result, TurnOutcome):
        return result

    message = _first_attr(result, ("message",))
    answer = _first_attr(result, _ANSWER_ATTRS)
    if answer is None and message is not None:
        answer = _first_attr(message, ("content", "text"))

    retrieval = _first_attr(result, ("retrieval", "retrieval_result"))
    chunks = _first_attr(result, _CHUNK_ATTRS)
    if chunks is None and retrieval is not None:
        chunks = _first_attr(retrieval, ("chunks",))

    citations = _first_attr(result, _CITATION_ATTRS)
    if citations is None and message is not None:
        citations = _first_attr(message, ("citations",))

    tools = _first_attr(result, _TOOL_ATTRS)
    if tools is None and message is not None:
        tools = _first_attr(message, ("tool_calls",))

    return TurnOutcome(
        answer=str(answer or ""),
        chunks=list(chunks or []),
        citations=list(citations or []),
        tools_invoked=_tool_names(tools),
        usages=_usage_list(_first_attr(result, _USAGE_ATTRS)),
        cost_usd=_first_attr(result, ("cost_usd",)),
        refused=bool(_first_attr(result, ("refused", "is_refusal")) or False),
        trace_id=_first_attr(result, ("trace_id",)),
        citation_report=_first_attr(result, ("citation_report", "citations_report")),
        guardrails=list(_first_attr(result, ("guardrails", "guardrail_events")) or []),
        raw=result,
    )


class OrchestratorRunner:
    """Calls the real pipeline entry point named by the resolved pipeline target.

    The target is resolved as ``module:attribute`` and called with whichever of
    ``message`` / ``principal`` / ``settings`` / ``session_id`` / ``stream`` /
    ``allow_tools`` / ``filters`` its signature accepts, so the harness binds to the
    orchestrator without dictating its parameter list. If the target does not take a
    question and a principal, that is a wiring error and it says so.

    Resolution order is :func:`eval.harness.resolve_pipeline_target`'s: the
    constructor argument, then ``RAG_EVAL_PIPELINE_TARGET``, then the
    ``eval_pipeline_target`` settings field once it exists, then
    :data:`eval.harness.DEFAULT_PIPELINE_TARGET`. It deliberately does **not** go
    through :func:`eval.eval_setting`: that helper's documented fallback names
    ``app.rag.orchestrator:run_turn``, a callable the orchestrator has never
    exposed, so every run died on it before reaching an item.

    Attributes:
        settings: Resolved settings.
        target: The ``module:attribute`` string in use.
    """

    def __init__(
        self,
        *,
        target: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Build a runner.

        Args:
            target: ``module:attribute`` override; wins over environment and
                settings.
            settings: Resolved settings.
        """
        # Imported here because `eval.harness` imports this module at load time.
        from eval.harness import resolve_pipeline_target

        self.settings = settings or get_settings()
        self.target = resolve_pipeline_target(target=target, settings=self.settings)
        self._callable: Callable[..., Any] | None = None
        self._kwargs: dict[str, Any] | None = None

    def _resolve(self) -> tuple[Callable[..., Any], dict[str, Any]]:
        """Import the target and work out how to call it.

        Returns:
            The coroutine function and the static keyword arguments it accepts.

        Raises:
            EvalHarnessError: If the target cannot be imported, is not callable, or
                does not accept a question and a principal.
        """
        if self._callable is not None and self._kwargs is not None:
            return self._callable, self._kwargs

        module_name, _, attribute = self.target.partition(":")
        if not module_name or not attribute:
            msg = (
                f"the eval pipeline target must be 'module:attribute', got "
                f"{self.target!r}; set {_TARGET_ENV}"
            )
            raise EvalHarnessError(msg)
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            msg = (
                f"cannot import {module_name!r} for eval pipeline target "
                f"{self.target!r}; install the workspace with "
                "`uv sync --all-packages`"
            )
            raise EvalHarnessError(msg) from exc
        function = getattr(module, attribute, None)
        if not callable(function):
            msg = (
                f"{self.target!r} is not callable; set {_TARGET_ENV} to a real "
                "'module:attribute' coroutine"
            )
            raise EvalHarnessError(msg)

        signature = inspect.signature(function)
        accepted = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        takes_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        message_param = next(
            (name for name in _MESSAGE_PARAMS if name in accepted), None
        )
        principal_param = next(
            (name for name in _PRINCIPAL_PARAMS if name in accepted), None
        )
        if message_param is None or principal_param is None:
            if not takes_kwargs:
                msg = (
                    f"{self.target!r} must accept a question "
                    f"({'|'.join(_MESSAGE_PARAMS)}) and a principal "
                    f"({'|'.join(_PRINCIPAL_PARAMS)}); it accepts {sorted(accepted)}"
                )
                raise EvalHarnessError(msg)
            message_param = message_param or _MESSAGE_PARAMS[0]
            principal_param = principal_param or _PRINCIPAL_PARAMS[0]

        static: dict[str, Any] = {
            "_message_param": message_param,
            "_principal_param": principal_param,
        }
        # Anything the orchestrator offers that the harness has an opinion about.
        optional = {
            "settings": self.settings,
            "stream": False,
            "allow_tools": True,
            "filters": None,
            "session_id": None,
            "persist": False,
        }
        for name, value in optional.items():
            if name in accepted:
                static[name] = value
        self._callable, self._kwargs = function, static
        _log.info(
            "eval_pipeline_target_resolved",
            target=self.target,
            message_param=message_param,
            principal_param=principal_param,
            extra_kwargs=sorted(set(static) - {"_message_param", "_principal_param"}),
        )
        return function, static

    async def __call__(self, *, item: GoldenItem, principal: Principal) -> TurnOutcome:
        """Run one golden item through the pipeline.

        Args:
            item: The golden item.
            principal: The persona to run as.

        Returns:
            The normalised :class:`TurnOutcome`.
        """
        function, static = self._resolve()
        kwargs = {
            key: value for key, value in static.items() if not key.startswith("_")
        }
        kwargs[static["_message_param"]] = item.question
        kwargs[static["_principal_param"]] = principal
        result = function(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return coerce_outcome(result)


# ------------------------------------------------------------------ golden set
def _resolve_path(path: str | Path) -> Path:
    """Resolve a configured path against the cwd, then the repository root.

    Args:
        path: Absolute or relative path.

    Returns:
        The first existing candidate, or the cwd-relative path so the caller's
        error message names what it actually looked for.
    """
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    rooted = _REPO_ROOT / candidate
    return rooted if rooted.exists() else candidate


def _load_yaml(path: Path) -> Any:
    """Read a YAML document.

    Args:
        path: File to read.

    Returns:
        The parsed document.

    Raises:
        EvalHarnessError: If the file is missing or not valid YAML.
    """
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        msg = f"{path} does not exist"
        raise EvalHarnessError(msg) from exc
    except yaml.YAMLError as exc:
        msg = f"{path} is not valid YAML: {exc}"
        raise EvalHarnessError(msg) from exc


def load_personas(
    path: str | Path | None = None, *, settings: Settings | None = None
) -> dict[str, Principal]:
    """Load the persona file into principals.

    Each persona is a :class:`~ragcore.models.acl.Principal`: the same shape the
    Entra token resolver produces, so an evaluated turn is indistinguishable from a
    real one as far as the ACL filter is concerned.

    Args:
        path: Persona file; defaults to ``eval_personas_path``.
        settings: Resolved settings.

    Returns:
        Persona key to principal.

    Raises:
        EvalHarnessError: If the file is malformed or empty.
    """
    cfg = settings or get_settings()
    target = _resolve_path(path or eval_setting(cfg, "eval_personas_path"))
    document = _load_yaml(target)
    if isinstance(document, dict) and "personas" in document:
        document = document["personas"]
    if not isinstance(document, dict) or not document:
        msg = f"{target} does not define any personas"
        raise EvalHarnessError(msg)
    personas: dict[str, Principal] = {}
    for key, payload in document.items():
        try:
            personas[str(key)] = Principal.model_validate(payload)
        except Exception as exc:  # re-raised with the offending key
            msg = f"persona {key!r} in {target} is not a valid Principal: {exc}"
            raise EvalHarnessError(msg) from exc
    return personas


def load_golden_set(
    path: str | Path | None = None, *, settings: Settings | None = None
) -> list[GoldenItem]:
    """Load and validate the golden set.

    Args:
        path: Golden file; defaults to ``eval_golden_path``.
        settings: Resolved settings.

    Returns:
        The items in file order.

    Raises:
        EvalHarnessError: If the file is malformed, empty, or has duplicate ids.
    """
    cfg = settings or get_settings()
    target = _resolve_path(path or cfg.eval_golden_path)
    document = _load_yaml(target)
    if isinstance(document, dict):
        document = document.get("items")
    if not isinstance(document, list) or not document:
        msg = f"{target} does not define an 'items' list"
        raise EvalHarnessError(msg)

    items: list[GoldenItem] = []
    seen: set[str] = set()
    for index, payload in enumerate(document):
        try:
            item = GoldenItem.model_validate(payload)
        except Exception as exc:  # re-raised with the offending index
            msg = f"golden item #{index} in {target} is invalid: {exc}"
            raise EvalHarnessError(msg) from exc
        if item.item_id in seen:
            msg = f"duplicate golden item id {item.item_id!r} in {target}"
            raise EvalHarnessError(msg)
        seen.add(item.item_id)
        items.append(item)
    return items


def select_items(
    items: Sequence[GoldenItem],
    *,
    categories: Sequence[str] | None = None,
    item_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[GoldenItem]:
    """Filter the golden set down to what this run should execute.

    Args:
        items: Every loaded item.
        categories: Keep only these categories.
        item_ids: Keep only these ids.
        limit: Keep at most this many, after filtering.

    Returns:
        The selected items, in file order.
    """
    wanted_categories = {value.strip() for value in categories or () if value.strip()}
    wanted_ids = {value.strip() for value in item_ids or () if value.strip()}
    selected = [
        item
        for item in items
        if (not wanted_categories or item.category in wanted_categories)
        and (not wanted_ids or item.item_id in wanted_ids)
    ]
    if limit is not None and limit > 0:
        selected = selected[:limit]
    return selected


def new_run_id() -> str:
    """Mint an evaluation run id.

    Returns:
        A sortable timestamped id with a random tail, so two runs started in the
        same second cannot overwrite each other's artefacts. ``POST /eval/runs``
        allocates one up front so the response can name the run it just started.
    """
    return f"eval-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def config_fingerprint(settings: Settings) -> str:
    """Hash the settings that shape what a run measures.

    Args:
        settings: Resolved settings.

    Returns:
        32 hex characters. Two runs with the same fingerprint differ by code, not
        by configuration — which is the distinction a regression hunt needs first.
    """
    payload = {
        name: value
        for name, value in sorted(settings.model_dump(mode="json").items())
        if name.startswith(_FINGERPRINT_PREFIXES)
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    )
    return digest.hexdigest()[:32]


def _git_sha() -> str | None:
    """Read the commit under test without shelling out.

    Returns:
        The CI-provided SHA, the checked-out SHA from ``.git``, or None.
    """
    for variable in ("GITHUB_SHA", "GIT_COMMIT", "RAG_GIT_SHA"):
        value = os.environ.get(variable)
        if value:
            return value.strip()
    head = _REPO_ROOT / ".git" / "HEAD"
    try:
        content = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if content.startswith("ref:"):
        ref = _REPO_ROOT / ".git" / content.removeprefix("ref:").strip()
        try:
            return ref.read_text(encoding="utf-8").strip()
        except OSError:
            return None
    return content or None


def _available_tool_names(principal: Principal, settings: Settings) -> set[str] | None:
    """List the tool names registered for a persona.

    Args:
        principal: The persona.
        settings: Resolved settings.

    Returns:
        Lower-cased tool names — the REST/builtin tools plus each MCP server's name
        and its exposed tools — or None when the registry cannot be consulted, in
        which case nothing is skipped.
    """
    try:
        from app.rag.tools.registry import get_tool_registry
    except ImportError:  # pragma: no cover - broken install only
        return None
    try:
        registry = get_tool_registry(settings)
        names = {tool.name.casefold() for tool in registry.tools_for(principal)}
        for spec in registry.mcp_specs_for(principal):
            names.add(spec.name.casefold())
            names.update(name.casefold() for name in spec.allowed_tools)
    except Exception:  # a registry problem must not fail the run
        _log.warning("eval_tool_registry_unavailable", exc_info=True)
        return None
    return names


def _builtin_tool_names(settings: Settings) -> set[str]:
    """Names of the tools that are registered on every deployment.

    Args:
        settings: Resolved settings. Unused today; taken so the lookup can become
            configuration-driven without changing every call site.

    Returns:
        Lower-cased built-in tool names.
    """
    del settings
    try:
        from app.rag.tools.builtin import CONTEXT_TOOL_NAME, SEARCH_TOOL_NAME
    except ImportError:  # pragma: no cover - broken install only
        return {"search_corpus", "current_context"}
    return {SEARCH_TOOL_NAME.casefold(), CONTEXT_TOOL_NAME.casefold()}


def _skip_reason(
    item: GoldenItem, *, tool_names: set[str] | None, settings: Settings
) -> str:
    """Decide whether a ``tool_required`` item can run in this environment.

    Args:
        item: The golden item.
        tool_names: Tools registered for the persona, or None when unknown.
        settings: Resolved settings.

    Returns:
        The reason to skip, or an empty string to run the item.
    """
    expected = (item.expect_tool or "").casefold()
    if not expected:
        return ""
    if expected in _builtin_tool_names(settings):
        return ""
    if not eval_setting(settings, "eval_tool_required_live"):
        return (
            f"tool {item.expect_tool!r} is a REST/MCP tool; set "
            "RAG_EVAL_TOOL_REQUIRED_LIVE=true once the registry points at live "
            "backends"
        )
    if (
        tool_names is not None
        and eval_setting(settings, "eval_skip_unregistered_tools")
        and expected not in tool_names
    ):
        return (
            f"tool {item.expect_tool!r} is not registered for persona "
            f"{item.as_user!r}; configure services/api/config/tools.yaml to run it"
        )
    return ""


# --------------------------------------------------------------------- scoring
def _threshold_failures(
    scores: MetricScores,
    *,
    settings: Settings,
    quality_metrics: bool,
) -> list[str]:
    """Apply the gate thresholds to one item.

    Args:
        scores: The item's scores.
        settings: Resolved settings.
        quality_metrics: Whether the RAGAS-style quality thresholds apply. They do
            not for an item that is supposed to be refused: "faithful to the
            context" is not a meaningful bar for an answer that correctly declines.

    Returns:
        One failure string per breached threshold.
    """
    quality = {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
        "semantic_similarity",
    }
    failures: list[str] = []
    for threshold in gate_thresholds(settings):
        if threshold.metric in quality and not quality_metrics:
            continue
        value = getattr(scores, threshold.metric, None)
        if value is None:
            continue
        if threshold.direction == "max":
            if value > threshold.limit:
                failures.append(f"{threshold.metric}:{value:.0f}>{threshold.limit:.0f}")
        elif value < threshold.limit:
            failures.append(f"{threshold.metric}:{value:.3f}<{threshold.limit:.3f}")
    return failures


def _score_names(scores: MetricScores) -> Iterable[tuple[str, float]]:
    """Yield the populated metric values.

    Args:
        scores: The item's scores.

    Returns:
        Name/value pairs for everything that was measured.
    """
    return scores.as_mapping().items()


async def evaluate_item(
    item: GoldenItem,
    principal: Principal,
    *,
    runner: PipelineRunner,
    settings: Settings,
    tool_names: set[str] | None = None,
) -> tuple[EvalResult | None, EvalItemDiagnostics]:
    """Run one golden item and score it.

    Args:
        item: The golden item.
        principal: The persona to run as.
        runner: How to execute the item.
        settings: Resolved settings.
        tool_names: Tools registered for this persona; used to skip a
            ``tool_required`` item whose REST/MCP tool is not configured.

    Returns:
        ``(result, diagnostics)``. ``result`` is None when the item was skipped, so
        a skipped item never lands in an aggregate.
    """
    diagnostics = EvalItemDiagnostics(
        item_id=item.item_id,
        category=item.category,
        persona=item.as_user,
        tenant_id=item.tenant_id,
        question=item.question,
        expected_tool=item.expect_tool,
        expect_refusal=item.expect_refusal,
    )

    skip = _skip_reason(item, tool_names=tool_names, settings=settings)
    if skip:
        diagnostics.skipped = True
        diagnostics.skip_reason = skip
        _log.info(
            "eval_item_skipped",
            item_id=item.item_id,
            tool=item.expect_tool,
            reason=skip,
        )
        return None, diagnostics

    tracer = get_tracer(settings)
    started = time.perf_counter()
    timeout = float(eval_setting(settings, "eval_item_timeout_seconds"))

    async with tracer.trace(
        "eval.item",
        user_id=principal.user_id,
        tenant_id=principal.tenant_id,
        tags=["eval", item.category],
        metadata={
            "item_id": item.item_id,
            "persona": item.as_user,
            "category": item.category,
            "expect_refusal": item.expect_refusal,
            "expect_tool": item.expect_tool,
        },
    ) as handle:
        failures: list[str] = []
        try:
            async with asyncio.timeout(timeout):
                outcome = await runner(item=item, principal=principal)
        except TimeoutError:
            outcome = TurnOutcome()
            failures.append(f"timeout:{timeout:.0f}s")
            _log.error("eval_item_timeout", item_id=item.item_id, timeout=timeout)
        except EvalDependencyError:
            raise
        except Exception as exc:  # one bad item must not stop the run
            outcome = TurnOutcome()
            failures.append(f"error:{exc.__class__.__name__}")
            _log.error("eval_item_failed", item_id=item.item_id, exc_info=True)

        latency_ms = (time.perf_counter() - started) * 1000.0
        trace_id = outcome.trace_id or handle.trace_id

        scores, extra_failures, diagnostics = await _score_outcome(
            item=item,
            principal=principal,
            outcome=outcome,
            diagnostics=diagnostics,
            latency_ms=latency_ms,
            settings=settings,
        )
        failures.extend(extra_failures)

        result = EvalResult(
            item_id=item.item_id,
            answer=diagnostics.answer_preview,
            retrieved_chunk_ids=outcome.retrieved_chunk_ids(),
            scores=scores,
            passed=not failures,
            failures=failures,
            trace_id=trace_id,
        )
        _push_scores(
            tracer,
            result=result,
            diagnostics=diagnostics,
            trace_id=trace_id,
            settings=settings,
        )
        handle.update(
            output={
                "passed": result.passed,
                "failures": failures,
                "category": item.category,
            }
        )

    _log.info(
        "eval_item_scored",
        item_id=item.item_id,
        category=item.category,
        passed=result.passed,
        failures=len(failures),
        latency_ms=round(latency_ms, 1),
    )
    return result, diagnostics


async def _score_outcome(
    *,
    item: GoldenItem,
    principal: Principal,
    outcome: TurnOutcome,
    diagnostics: EvalItemDiagnostics,
    latency_ms: float,
    settings: Settings,
) -> tuple[MetricScores, list[str], EvalItemDiagnostics]:
    """Score one turn against its golden item.

    Args:
        item: The golden item.
        principal: The persona.
        outcome: The normalised turn outcome.
        diagnostics: The record to fill in.
        latency_ms: Measured end-to-end latency.
        settings: Resolved settings.

    Returns:
        ``(scores, failures, diagnostics)``.
    """
    failures: list[str] = []
    answer = outcome.answer

    leak = acl_leak(
        principal=principal,
        chunks=outcome.chunks,
        answer=answer,
        citations=outcome.citations,
        forbidden_terms=item.must_not_contain,
        settings=settings,
    )
    failures.extend(leak.failure_reasons())

    refusal = refusal_correct(
        answer=answer,
        expect_refusal=item.expect_refusal,
        refused=outcome.refused,
        settings=settings,
    )
    if refusal.score < 1.0:
        failures.append(
            "refusal_missing" if item.expect_refusal else "refusal_unexpected"
        )
    elif item.expect_refusal and not refusal.acceptable:
        failures.append("refusal_unhelpful:" + ",".join(refusal.reasons))

    literal_failures = substring_failures(
        answer=answer,
        must_contain=item.must_contain,
        must_not_contain=item.must_not_contain,
    )
    # A forbidden term is already reported as an ACL leak; do not count it twice.
    failures.extend(
        failure
        for failure in literal_failures
        if not failure.startswith("forbidden_present:")
    )

    recall = retrieval_recall(
        expected_document_ids=item.expected_document_ids,
        expected_chunk_ids=item.expected_chunk_ids,
        retrieved_chunk_ids=outcome.retrieved_chunk_ids(),
        retrieved_document_ids=outcome.retrieved_document_ids(),
    )
    failures.extend(f"retrieval_miss:{doc}" for doc in recall.missing_document_ids)
    failures.extend(f"retrieval_miss_chunk:{cid}" for cid in recall.missing_chunk_ids)

    tools = tool_correct(
        expect_tool=item.expect_tool, tools_invoked=outcome.tools_invoked
    )
    if tools is not None and tools < 1.0:
        failures.append(f"tool_missing:{item.expect_tool}")

    similarity = await semantic_similarity(answer, item.ground_truth, settings=settings)
    validity = citation_validity(
        answer=answer,
        chunks=outcome.chunks,
        report=outcome.citation_report,
        settings=settings,
    )

    scores = MetricScores(
        semantic_similarity=similarity,
        citation_validity=validity,
        acl_leak=leak.score,
        refusal_correct=refusal.score,
        latency_ms=latency_ms,
        cost_usd=outcome.resolved_cost(),
    )

    # RAGAS-style quality metrics are meaningless for an item whose correct answer
    # is a refusal: there is no faithful answer to a question the corpus cannot
    # support, and scoring one would punish the behaviour we require.
    if not item.expect_refusal and answer.strip():
        adapter = get_ragas_adapter(settings)
        ragas_scores = await adapter.score(
            RagasSample(
                item_id=item.item_id,
                question=item.question,
                answer=answer,
                contexts=[chunk.payload.text for chunk in outcome.chunks],
                ground_truth=item.ground_truth,
                semantic_similarity=similarity,
            )
        )
        scores.faithfulness = ragas_scores.faithfulness
        scores.answer_relevancy = ragas_scores.answer_relevancy
        scores.context_precision = ragas_scores.context_precision
        scores.context_recall = ragas_scores.context_recall
        scores.answer_correctness = ragas_scores.answer_correctness
        diagnostics.ragas_backend = ragas_scores.backend
        diagnostics.degraded_reason = ragas_scores.degraded_reason

    failures.extend(
        _threshold_failures(
            scores, settings=settings, quality_metrics=not item.expect_refusal
        )
    )

    preview_chars = int(eval_setting(settings, "eval_report_answer_chars"))
    diagnostics.tools_invoked = list(outcome.tools_invoked)
    diagnostics.tool_correct = tools
    diagnostics.retrieval_recall = recall.score
    diagnostics.missing_document_ids = recall.missing_document_ids
    diagnostics.retrieved_document_ids = outcome.retrieved_document_ids()
    diagnostics.refused = refusal.refused
    diagnostics.answer_preview = answer[:preview_chars]
    return scores, failures, diagnostics


def _push_scores(
    tracer: Any,
    *,
    result: EvalResult,
    diagnostics: EvalItemDiagnostics,
    trace_id: str | None,
    settings: Settings,
) -> None:
    """Send every measured score to Langfuse against the turn's trace.

    Args:
        tracer: The process tracer; a no-op when Langfuse is disabled.
        result: The scored item.
        diagnostics: Extras that are not part of ``MetricScores``.
        trace_id: Trace the scores attach to.
        settings: Resolved settings supplying the score-name prefix.
    """
    prefix = str(eval_setting(settings, "eval_score_prefix"))
    comment = ("; ".join(result.failures) or None) if not result.passed else None
    for name, value in _score_names(result.scores):
        tracer.score(
            name=f"{prefix}{name}",
            value=float(value),
            comment=comment,
            trace_id=trace_id,
        )
    for name, value in (
        ("tool_correct", diagnostics.tool_correct),
        ("retrieval_recall", diagnostics.retrieval_recall),
    ):
        if value is not None:
            tracer.score(
                name=f"{prefix}{name}",
                value=float(value),
                comment=comment,
                trace_id=trace_id,
            )
    tracer.score(
        name=f"{prefix}passed",
        value=1.0 if result.passed else 0.0,
        comment=comment,
        trace_id=trace_id,
    )


# ------------------------------------------------------------------- the runner
def _extra_aggregates(
    diagnostics: Mapping[str, EvalItemDiagnostics],
) -> dict[str, float]:
    """Average the metrics that live outside ``MetricScores``.

    Args:
        diagnostics: Per-item records.

    Returns:
        The extra aggregate keys, omitting any nothing measured.
    """
    extras: dict[str, float] = {}
    for name in ("tool_correct", "retrieval_recall"):
        values = [
            float(value)
            for record in diagnostics.values()
            if not record.skipped and (value := getattr(record, name)) is not None
        ]
        if values:
            extras[name] = sum(values) / len(values)
    return extras


async def _persist(
    artifacts: EvalRunArtifacts,
    *,
    tenant_id: str,
    settings: Settings,
    notes: str | None,
) -> None:
    """Write the run and its results to PostgreSQL.

    A reporting store being down must never change a gate's verdict, so every
    failure here is logged and swallowed.

    Args:
        artifacts: The completed run artefacts.
        tenant_id: Tenant the run row belongs to.
        settings: Resolved settings.
        notes: Free-form note stored on the run row.
    """
    try:
        from ragcore.db import repositories as repo
        from ragcore.db import session_scope
    except ImportError:  # pragma: no cover - broken install only
        _log.warning("eval_persist_unavailable")
        return

    run = artifacts.run
    try:
        async with session_scope(settings) as session:
            await repo.write_eval_run(
                session,
                tenant_id=tenant_id,
                run_id=run.run_id,
                started_at=run.started_at,
                finished_at=run.finished_at,
                git_sha=run.git_sha,
                config_fingerprint=run.config_fingerprint,
                golden_set_path=artifacts.golden_path,
                aggregate=run.aggregate,
                gate_passed=run.gate_passed,
                item_count=run.item_count,
                passed_count=run.passed_count,
                failed_count=run.failed_count,
                total_cost_usd=run.total_cost_usd,
                notes=notes,
            )
            for result in run.results:
                record = artifacts.diagnostics_for(result.item_id)
                await repo.write_eval_result(
                    session,
                    tenant_id=record.tenant_id or tenant_id,
                    run_id=run.run_id,
                    item_id=result.item_id,
                    answer=result.answer,
                    scores=result.scores.model_dump(mode="json"),
                    passed=result.passed,
                    failures=result.failures,
                    retrieved_chunk_ids=result.retrieved_chunk_ids,
                    category=record.category,
                    question=record.question,
                    latency_ms=result.scores.latency_ms,
                    cost_usd=result.scores.cost_usd,
                    trace_id=result.trace_id,
                )
            await session.commit()
    except Exception:  # persistence is reporting, not the verdict
        _log.warning("eval_persist_failed", run_id=run.run_id, exc_info=True)


def _dominant_tenant(items: Sequence[GoldenItem]) -> str:
    """Pick the tenant a run's row should be filed under.

    Args:
        items: The selected items.

    Returns:
        The most common tenant id, or an empty string for an empty selection.
    """
    counts: dict[str, int] = {}
    for item in items:
        counts[item.tenant_id] = counts.get(item.tenant_id, 0) + 1
    return max(counts, key=lambda key: counts[key]) if counts else ""


async def run_eval(
    *,
    golden_path: str | Path | None = None,
    personas_path: str | Path | None = None,
    categories: Sequence[str] | None = None,
    item_ids: Sequence[str] | None = None,
    limit: int | None = None,
    baseline_path: str | Path | None = None,
    gate: bool = True,
    report_dir: str | Path | None = None,
    concurrency: int | None = None,
    tenant_id: str | None = None,
    persist: bool | None = None,
    notes: str | None = None,
    settings: Settings | None = None,
    runner: PipelineRunner | None = None,
    run_id: str | None = None,
) -> EvalRunArtifacts:
    """Evaluate the golden set end to end.

    Args:
        golden_path: Golden file; defaults to ``eval_golden_path``.
        personas_path: Persona file; defaults to ``eval_personas_path``.
        categories: Restrict the run to these categories.
        item_ids: Restrict the run to these item ids.
        limit: Evaluate at most this many items after filtering; defaults to
            ``eval_sample_size``.
        baseline_path: A previous run's JSON to compare against.
        gate: Apply the CI gate and record the verdict on the run.
        report_dir: Where to write the artefacts; defaults to ``eval_report_dir``.
        concurrency: Items in flight; defaults to ``eval_max_concurrency``.
        tenant_id: Tenant the run row is filed under.
        persist: Write ``eval_runs``/``eval_results`` rows.
        notes: Free-form note stored with the run.
        settings: Resolved settings.
        runner: Pipeline runner override; defaults to :class:`OrchestratorRunner`.
        run_id: Pre-allocated run id. ``POST /eval/runs`` allocates one so it can
            answer with the id before the run finishes; omitted, a fresh one is
            minted here.

    Returns:
        The :class:`~eval.report.EvalRunArtifacts`, already written to disk.

    Raises:
        EvalHarnessError: If evaluation is disabled, the golden set is unusable, or
            an item names a persona the persona file does not define.
    """
    cfg = settings or get_settings()
    if not cfg.eval_enabled:
        msg = "evaluation is disabled (RAG_EVAL_ENABLED=false)"
        raise EvalHarnessError(msg)

    golden_file = _resolve_path(golden_path or cfg.eval_golden_path)
    personas_file = _resolve_path(
        personas_path or eval_setting(cfg, "eval_personas_path")
    )
    items = load_golden_set(golden_file, settings=cfg)
    personas = load_personas(personas_file, settings=cfg)

    selected = select_items(
        items,
        categories=categories,
        item_ids=item_ids,
        limit=limit if limit is not None else cfg.eval_sample_size,
    )
    if not selected:
        msg = "no golden items matched the selection"
        raise EvalHarnessError(msg)

    missing = sorted({item.as_user for item in selected} - set(personas))
    if missing:
        msg = f"golden items reference undefined personas: {', '.join(missing)}"
        raise EvalHarnessError(msg)

    for item in selected:
        principal = personas[item.as_user]
        if principal.tenant_id != item.tenant_id:
            msg = (
                f"item {item.item_id!r} claims tenant {item.tenant_id!r} but persona "
                f"{item.as_user!r} belongs to {principal.tenant_id!r}"
            )
            raise EvalHarnessError(msg)

    execute = runner or OrchestratorRunner(settings=cfg)
    tool_cache: dict[str, set[str] | None] = {}
    for persona_key in {item.as_user for item in selected if item.expect_tool}:
        tool_cache[persona_key] = _available_tool_names(personas[persona_key], cfg)

    run = EvalRun(
        run_id=run_id or new_run_id(),
        started_at=datetime.now(UTC),
        git_sha=_git_sha(),
        config_fingerprint=config_fingerprint(cfg),
    )
    _log.info(
        "eval_run_started",
        run_id=run.run_id,
        items=len(selected),
        golden=str(golden_file),
        concurrency=concurrency or cfg.eval_max_concurrency,
    )

    semaphore = asyncio.Semaphore(max(1, concurrency or int(cfg.eval_max_concurrency)))

    async def _one(item: GoldenItem) -> tuple[EvalResult | None, EvalItemDiagnostics]:
        async with semaphore:
            return await evaluate_item(
                item,
                personas[item.as_user],
                runner=execute,
                settings=cfg,
                tool_names=tool_cache.get(item.as_user),
            )

    outcomes = await asyncio.gather(*(_one(item) for item in selected))

    diagnostics: dict[str, EvalItemDiagnostics] = {}
    skipped: list[str] = []
    for result, record in outcomes:
        diagnostics[record.item_id] = record
        if result is None:
            skipped.append(record.item_id)
            continue
        run.results.append(result)

    run.finished_at = datetime.now(UTC)
    run.compute_aggregate()
    run.aggregate.update(_extra_aggregates(diagnostics))
    run.aggregate["skipped_count"] = float(len(skipped))

    artifacts = EvalRunArtifacts(
        run=run,
        diagnostics=diagnostics,
        category_aggregate=category_aggregate(run, diagnostics),
        golden_path=str(golden_file),
        personas_path=str(personas_file),
        baseline_path=str(baseline_path) if baseline_path else None,
        skipped_items=skipped,
    )

    if gate:
        artifacts.gate = apply_gate(run, settings=cfg)

    comparison: RunComparison | None = None
    if baseline_path:
        try:
            baseline = load_artifacts(_resolve_path(baseline_path))
        except (OSError, ValueError) as exc:
            _log.warning(
                "eval_baseline_unreadable",
                path=str(baseline_path),
                error=str(exc),
            )
        else:
            comparison = compare_runs(
                run, baseline.run, diagnostics=diagnostics, settings=cfg
            )

    write_reports(artifacts, directory=report_dir, comparison=comparison, settings=cfg)

    should_persist = (
        bool(eval_setting(cfg, "eval_persist_results")) if persist is None else persist
    )
    if should_persist:
        await _persist(
            artifacts,
            tenant_id=tenant_id
            or str(eval_setting(cfg, "eval_run_tenant_id") or "")
            or _dominant_tenant(selected),
            settings=cfg,
            notes=notes,
        )

    tracer = get_tracer(cfg)
    prefix = str(eval_setting(cfg, "eval_score_prefix"))
    async with tracer.trace(
        "eval.run",
        tags=["eval", "run"],
        metadata={
            "run_id": run.run_id,
            "items": run.item_count,
            "skipped": len(skipped),
            "git_sha": run.git_sha,
            "config_fingerprint": run.config_fingerprint,
        },
    ) as handle:
        for name, value in run.aggregate.items():
            tracer.score(
                name=f"{prefix}run.{name}", value=float(value), trace_id=handle.trace_id
            )
        handle.update(output={"gate_passed": run.gate_passed})

    _log.info(
        "eval_run_finished",
        run_id=run.run_id,
        items=run.item_count,
        passed=run.passed_count,
        failed=run.failed_count,
        skipped=len(skipped),
        gate_passed=run.gate_passed,
        cost_usd=round(run.total_cost_usd, 6),
    )
    return artifacts


# ------------------------------------------------------------------------- CLI
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the runner's command-line arguments.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="python -m eval.run_eval",
        description=(
            "Run the golden set through the real RAG pipeline, score it with RAGAS "
            "plus semantic similarity, and optionally apply the CI gate."
        ),
    )
    parser.add_argument("--golden", default=None, help="path to the golden set YAML")
    parser.add_argument("--personas", default=None, help="path to the persona YAML")
    parser.add_argument(
        "--category",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "only run this category; repeatable. One of: "
            + ", ".join(category.value for category in EvalCategory)
        ),
    )
    parser.add_argument(
        "--item",
        action="append",
        default=None,
        metavar="ID",
        help="only run this item id; repeatable",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="evaluate at most N items"
    )
    parser.add_argument(
        "--baseline",
        default=None,
        metavar="PATH",
        help="previous run JSON to A/B against",
    )
    parser.add_argument(
        "--concurrency", type=int, default=None, help="items evaluated in parallel"
    )
    parser.add_argument(
        "--report-dir", default=None, help="directory for the JSON/markdown/HTML"
    )
    parser.add_argument("--tenant", default=None, help="tenant id for the run row")
    parser.add_argument("--notes", default=None, help="note stored with the run")
    gate_group = parser.add_mutually_exclusive_group()
    gate_group.add_argument(
        "--gate",
        dest="gate",
        action="store_true",
        default=True,
        help="apply the CI gate and exit non-zero on failure (default)",
    )
    gate_group.add_argument(
        "--no-gate", dest="gate", action="store_false", help="score without gating"
    )
    parser.add_argument(
        "--no-persist",
        dest="persist",
        action="store_false",
        default=None,
        help="do not write eval_runs / eval_results rows",
    )
    parser.add_argument(
        "--json",
        dest="print_json",
        action="store_true",
        help="print the aggregate as JSON instead of the gate table",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` when the run completed and the gate passed (or was not requested),
        ``1`` when the gate failed, ``2`` when the harness itself could not run.
    """
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings)
    try:
        artifacts = asyncio.run(
            run_eval(
                golden_path=args.golden,
                personas_path=args.personas,
                categories=args.category,
                item_ids=args.item,
                limit=args.limit,
                baseline_path=args.baseline,
                gate=args.gate,
                report_dir=args.report_dir,
                concurrency=args.concurrency,
                tenant_id=args.tenant,
                persist=args.persist,
                notes=args.notes,
                settings=settings,
            )
        )
    except EvalHarnessError as exc:
        print(f"error: {exc}")
        return EXIT_FAILED
    except Exception as exc:  # the CLI reports, it does not traceback
        _log.exception("eval_run_failed")
        print(f"error: {exc.__class__.__name__}: {exc}")
        return EXIT_FAILED

    if args.print_json:
        print(json.dumps(artifacts.run.aggregate, indent=2, sort_keys=True))
    elif artifacts.gate is not None:
        print(format_gate_table(artifacts.gate, run=artifacts.run))
    else:
        print(
            f"{artifacts.run.item_count} items, "
            f"{artifacts.run.passed_count} passed, "
            f"{artifacts.run.failed_count} failed "
            f"({artifacts.run.pass_rate:.1%})"
        )

    if artifacts.gate is not None and not artifacts.gate.passed:
        return EXIT_GATE_FAILED
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
