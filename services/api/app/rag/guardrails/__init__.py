"""Guardrails for the RAG pipeline — requirement #9's edge cases.

Four stages of `docs/CONTRACTS.md`'s ordered pipeline live here:

* **stage 1** :mod:`app.rag.guardrails.input_guard` — redact-before-log PII,
  size cap, language detection and the prompt-injection scan on the user turn.
* **stage 6** :mod:`app.rag.guardrails.ood` — out-of-domain adjudication and the
  refusal that names what the corpus *does* cover.
* **stage 7** :mod:`app.rag.guardrails.contradiction` — claim clustering,
  conflict detection and recency/authority resolution, surfaced with both
  citations.
* **stage 12** :mod:`app.rag.guardrails.output_guard` — PII egress, the
  defence-in-depth clearance check, the groundedness gate and refusal quality.

:mod:`app.rag.guardrails.injection` is used by stages 1 *and* 5/9: indirect
injection arriving through a poisoned document is the threat that matters, so
retrieved chunk text is scanned as well as the user turn and is always rendered
inside a delimited, clearly-labelled untrusted block.

Submodules are re-exported lazily through a module-level ``__getattr__`` so that
importing this package costs nothing and so that submodules can import
:func:`guardrail_setting` from it without a circular import.

**Tunables.** Every threshold, limit and toggle is a ``ragcore.settings`` field.
The fields listed in :data:`GUARDRAIL_DEFAULTS` are the ones added by Addendum G
of `docs/CONTRACTS.md`; they are read through :func:`guardrail_setting`, which
falls back to the documented default when the running ``Settings`` build predates
the addendum. Nothing in this package hard-codes a threshold at a call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragcore.settings import Settings

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for type checkers
    from app.rag.guardrails.contradiction import (
        ClaimCluster,
        ConflictResolution,
        ConflictVerdict,
        Contradiction,
        ContradictionReport,
        check_contradictions,
        cluster_claims,
        render_contradiction_notes,
        resolve_conflict,
    )
    from app.rag.guardrails.injection import (
        INJECTION_DROP_REASON,
        UNTRUSTED_BLOCK_CLOSE,
        UNTRUSTED_BLOCK_OPEN,
        UNTRUSTED_PREAMBLE,
        InjectionPattern,
        InjectionSignal,
        InjectionVerdict,
        RetrievedScan,
        normalise_unicode,
        sanitise_untrusted,
        scan_retrieved,
        scan_text,
        scan_user_turn,
        wrap_untrusted,
    )
    from app.rag.guardrails.input_guard import (
        LANGUAGE_GUARDRAIL_KIND,
        InputDecision,
        detect_language,
        run_input_guard,
    )
    from app.rag.guardrails.ood import (
        CoverageItem,
        DomainCoverage,
        OODVerdict,
        RelevanceSignals,
        clear_coverage_cache,
        fallback_refusal,
        relevance_signals,
        run_ood_gate,
        tenant_coverage,
    )
    from app.rag.guardrails.output_guard import (
        CLEARANCE_BLOCK_MESSAGE,
        GROUNDEDNESS_BLOCK_MESSAGE,
        PII_BLOCK_MESSAGE,
        ClearanceReport,
        ClearanceViolation,
        OutputDecision,
        RefusalQuality,
        assess_refusal,
        check_clearance,
        citation_validity_score,
        run_output_guard,
    )

__all__ = [
    "CLEARANCE_BLOCK_MESSAGE",
    "GROUNDEDNESS_BLOCK_MESSAGE",
    "GUARDRAIL_DEFAULTS",
    "GUARDRAIL_PROMPT_VERSION",
    "INJECTION_DROP_REASON",
    "LANGUAGE_GUARDRAIL_KIND",
    "PII_BLOCK_MESSAGE",
    "UNTRUSTED_BLOCK_CLOSE",
    "UNTRUSTED_BLOCK_OPEN",
    "UNTRUSTED_PREAMBLE",
    "ClaimCluster",
    "ClearanceReport",
    "ClearanceViolation",
    "ConflictResolution",
    "ConflictVerdict",
    "Contradiction",
    "ContradictionReport",
    "CoverageItem",
    "DomainCoverage",
    "InjectionPattern",
    "InjectionSignal",
    "InjectionVerdict",
    "InputDecision",
    "OODVerdict",
    "OutputDecision",
    "RefusalQuality",
    "RelevanceSignals",
    "RetrievedScan",
    "assess_refusal",
    "check_clearance",
    "check_contradictions",
    "citation_validity_score",
    "clear_coverage_cache",
    "cluster_claims",
    "detect_language",
    "fallback_refusal",
    "guardrail_setting",
    "normalise_unicode",
    "relevance_signals",
    "render_contradiction_notes",
    "resolve_conflict",
    "run_input_guard",
    "run_ood_gate",
    "run_output_guard",
    "sanitise_untrusted",
    "scan_retrieved",
    "scan_text",
    "scan_user_turn",
    "tenant_coverage",
    "wrap_untrusted",
]


#: Guardrail tunables added by Addendum G of `docs/CONTRACTS.md`, with the
#: default that applies when the running ``Settings`` build does not declare the
#: field yet. Keys are real ``RAG_*`` setting names: exporting
#: ``RAG_GUARDRAIL_OOD_COLLAPSE_SPREAD`` works as soon as the field exists on
#: :class:`ragcore.settings.Settings`.
GUARDRAIL_DEFAULTS: dict[str, Any] = {
    # ------------------------------------------------------------- stage 1
    "guardrail_input_min_chars": 1,
    "guardrail_input_truncate": False,
    "guardrail_input_normalise_unicode": True,
    "guardrail_input_credential_entities": ["API_KEY", "JWT"],
    "guardrail_language_detect_enabled": True,
    "guardrail_language_min_confidence": 0.35,
    "guardrail_allowed_languages": [],
    # ------------------------------------------------------- injection (1/5)
    "guardrail_injection_scan_retrieved": True,
    "guardrail_injection_retrieved_block_threshold": 0.5,
    "guardrail_injection_quarantine_retrieved": True,
    "guardrail_injection_max_scan_chars": 20_000,
    "guardrail_injection_classifier_enabled": False,
    "guardrail_injection_classifier_benign_factor": 0.5,
    # ------------------------------------------------------------- stage 6
    "guardrail_ood_mean_score_min": 0.2,
    "guardrail_ood_min_candidates": 1,
    "guardrail_ood_collapse_enabled": True,
    "guardrail_ood_collapse_top_k": 5,
    "guardrail_ood_collapse_spread": 0.05,
    "guardrail_ood_collapse_max_score": 0.55,
    "guardrail_ood_classifier_enabled": False,
    "guardrail_ood_coverage_sample": 512,
    "guardrail_ood_coverage_ttl_seconds": 900,
    "guardrail_ood_coverage_max_items": 8,
    "guardrail_ood_llm_refusal": False,
    # ------------------------------------------------------------- stage 7
    "guardrail_contradiction_min_chunks": 2,
    "guardrail_contradiction_max_clusters": 6,
    "guardrail_contradiction_max_pairs": 4,
    "guardrail_contradiction_llm_enabled": True,
    "guardrail_contradiction_snippet_chars": 600,
    # ------------------------------------------------------------ stage 12
    "guardrail_output_block_below_groundedness": 0.2,
    "guardrail_output_leak_span_chars": 60,
    "guardrail_output_pii_ignore_entities": ["DATE_TIME", "LOCATION"],
    "guardrail_refusal_check_enabled": True,
    "guardrail_refusal_min_chars": 40,
}

#: Version of the guardrail-owned system prompts, recorded on traced calls.
#: Bumped whenever :data:`app.rag.guardrails.injection.INJECTION_CLASSIFIER_SYSTEM`
#: changes.
GUARDRAIL_PROMPT_VERSION = "1.0.0"


def guardrail_setting(settings: Settings, name: str) -> Any:
    """Read a guardrail tunable, falling back to its documented default.

    Every guardrail threshold is an operator concern, so it is read from
    :class:`ragcore.settings.Settings` rather than written at the call site. The
    fields in :data:`GUARDRAIL_DEFAULTS` were introduced with this package; this
    accessor keeps the guardrails working against a ``Settings`` build that has
    not declared them yet, while still honouring an operator override the moment
    the field exists.

    Args:
        settings: Process settings.
        name: A key of :data:`GUARDRAIL_DEFAULTS`.

    Returns:
        The configured value, or the documented default.

    Raises:
        KeyError: If ``name`` is not a declared guardrail tunable — that is a
            typo at the call site, not a configuration problem.
    """
    if name not in GUARDRAIL_DEFAULTS:
        msg = f"unknown guardrail tunable {name!r}"
        raise KeyError(msg)
    return getattr(settings, name, GUARDRAIL_DEFAULTS[name])


_LAZY_EXPORTS: dict[str, str] = {
    "ClaimCluster": "contradiction",
    "ConflictResolution": "contradiction",
    "ConflictVerdict": "contradiction",
    "Contradiction": "contradiction",
    "ContradictionReport": "contradiction",
    "check_contradictions": "contradiction",
    "cluster_claims": "contradiction",
    "render_contradiction_notes": "contradiction",
    "resolve_conflict": "contradiction",
    "INJECTION_DROP_REASON": "injection",
    "UNTRUSTED_BLOCK_CLOSE": "injection",
    "UNTRUSTED_BLOCK_OPEN": "injection",
    "UNTRUSTED_PREAMBLE": "injection",
    "InjectionPattern": "injection",
    "InjectionSignal": "injection",
    "InjectionVerdict": "injection",
    "RetrievedScan": "injection",
    "normalise_unicode": "injection",
    "sanitise_untrusted": "injection",
    "scan_retrieved": "injection",
    "scan_text": "injection",
    "scan_user_turn": "injection",
    "wrap_untrusted": "injection",
    "LANGUAGE_GUARDRAIL_KIND": "input_guard",
    "InputDecision": "input_guard",
    "detect_language": "input_guard",
    "run_input_guard": "input_guard",
    "CoverageItem": "ood",
    "DomainCoverage": "ood",
    "OODVerdict": "ood",
    "RelevanceSignals": "ood",
    "clear_coverage_cache": "ood",
    "fallback_refusal": "ood",
    "relevance_signals": "ood",
    "run_ood_gate": "ood",
    "tenant_coverage": "ood",
    "CLEARANCE_BLOCK_MESSAGE": "output_guard",
    "GROUNDEDNESS_BLOCK_MESSAGE": "output_guard",
    "PII_BLOCK_MESSAGE": "output_guard",
    "ClearanceReport": "output_guard",
    "ClearanceViolation": "output_guard",
    "OutputDecision": "output_guard",
    "RefusalQuality": "output_guard",
    "assess_refusal": "output_guard",
    "check_clearance": "output_guard",
    "citation_validity_score": "output_guard",
    "run_output_guard": "output_guard",
}


def __getattr__(name: str) -> Any:
    """Resolve a public symbol from its submodule on first access.

    Args:
        name: Attribute being looked up on the package.

    Returns:
        The requested symbol.

    Raises:
        AttributeError: If the name is not exported by this package.
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    from importlib import import_module

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the package's public names for tab-completion and ``dir()``.

    Returns:
        The sorted contents of ``__all__``.
    """
    return sorted(__all__)
