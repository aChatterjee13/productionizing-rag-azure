"""Every documented tunable is a real, env-settable ``Settings`` field.

`docs/CONTRACTS.md` states the rule twice — once in the ``ragcore.settings``
section ("Every tunable in this document is a settings field with a sane default —
nothing is hard-coded at a call site") and again in each addendum that adds knobs.
The documentation audit found 122 documented tunables that were *not* fields: they
lived in six private ``*_DEFAULTS`` dicts (Addenda G, R, M, P and E), so
``Settings`` — which is ``extra="ignore"`` — silently discarded the ``RAG_*``
variable an operator exported. Retuning an OOD cutoff, a groundedness floor or a
PII confidence needed a code change and a redeploy.

:data:`PROMOTED` is the inventory those addenda document, name for documented
default. Each entry asserts three things: the field exists, its default is
unchanged (the behaviour the suite encodes), and ``RAG_<NAME>`` reaches it.

The one deliberate difference from the old tables is ``eval_pipeline_target``:
Addendum E documented ``"app.rag.orchestrator:run_turn"``, a callable that has
never existed, so :data:`eval.harness.DEFAULT_PIPELINE_TARGET` is the value the
harness actually resolves and therefore the value the field defaults to.

This module imports nothing service-private: it pins the *contract*, so it runs
wherever ``ragcore`` does.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from ragcore.settings import Settings

#: Documented tunable -> documented default, grouped by the private table that
#: used to own it. 122 audited keys plus the two ``ingest_function_*`` fields the
#: admin dispatch route reads.
PROMOTED: dict[str, Any] = {
    # --- Addendum G, app.rag.guardrails.GUARDRAIL_DEFAULTS: 34
    "guardrail_input_min_chars": 1,
    "guardrail_input_truncate": False,
    "guardrail_input_normalise_unicode": True,
    "guardrail_input_credential_entities": ["API_KEY", "JWT"],
    "guardrail_language_detect_enabled": True,
    "guardrail_language_min_confidence": 0.35,
    "guardrail_allowed_languages": [],
    "guardrail_injection_scan_retrieved": True,
    "guardrail_injection_retrieved_block_threshold": 0.5,
    "guardrail_injection_quarantine_retrieved": True,
    "guardrail_injection_max_scan_chars": 20000,
    "guardrail_injection_classifier_enabled": False,
    "guardrail_injection_classifier_benign_factor": 0.5,
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
    "guardrail_contradiction_min_chunks": 2,
    "guardrail_contradiction_max_clusters": 6,
    "guardrail_contradiction_max_pairs": 4,
    "guardrail_contradiction_llm_enabled": True,
    "guardrail_contradiction_snippet_chars": 600,
    "guardrail_output_block_below_groundedness": 0.2,
    "guardrail_output_leak_span_chars": 60,
    "guardrail_output_pii_ignore_entities": ["DATE_TIME", "LOCATION"],
    "guardrail_refusal_check_enabled": True,
    "guardrail_refusal_min_chars": 40,
    # --- Addendum R, app.rag.RAG_SETTING_DEFAULTS: 15
    "qt_hyde_max_query_words": 12,
    "retrieval_fusion_score_scale": 0.05,
    "retrieval_query_concurrency": 4,
    "retrieval_mmr_embed_fallback": True,
    "citation_fuzzy_threshold": 0.55,
    "citation_quote_min_ratio": 0.9,
    "citation_token_recall_threshold": 0.6,
    "citation_number_check": True,
    "citation_min_span_chars": 12,
    "citation_max_span_chars": 600,
    "citation_min_claim_words": 5,
    "citation_min_coverage": 0.5,
    "citation_anchor_tokens": 3,
    "citation_max_windows": 24,
    "citation_strip_unresolved_markers": True,
    # --- Addendum M, app.rag.memory.EXTRA_SETTING_DEFAULTS: 14
    "context_compact_every_n_turns": 6,
    "context_token_count_concurrency": 8,
    "context_token_cache_entries": 4096,
    "context_fit_max_passes": 3,
    "context_min_retrieved_chunks": 1,
    "context_duplicate_penalty": 0.35,
    "context_cache_history_breakpoint": True,
    "context_cache_history_min_tokens": 1024,
    "redis_session_prefix": "rag:session:",
    "memory_cache_max_entries": 500,
    "memory_consolidate_batch_size": 200,
    "memory_recall_oversample": 4,
    "memory_recall_similarity_weight": 0.7,
    "memory_profile_min_memories": 3,
    # --- Addendum P, app.API_SETTING_DEFAULTS: 19
    "api_request_id_header": "x-request-id",
    "api_trace_id_header": "x-trace-id",
    "api_cors_allow_credentials": True,
    "api_cors_allow_headers": ["*"],
    "api_cors_expose_headers": ["x-request-id", "x-trace-id", "retry-after"],
    "api_cors_max_age_seconds": 600,
    "api_gzip_min_bytes": 1024,
    "api_rate_limit_enabled": True,
    "api_rate_limit_burst": 20,
    "api_rate_limit_prefix": "rag:ratelimit:",
    "api_sse_retry_ms": 2000,
    "api_sse_heartbeat_comment": "heartbeat",
    "api_problem_type_base": "https://productionizing-rag.dev/problems",
    "api_docs_enabled": True,
    "api_default_page_size": 50,
    "api_max_page_size": 200,
    "api_warm_models": True,
    "api_ensure_collections": True,
    "api_readiness_timeout_seconds": 5.0,
    # --- Addendum P, app.auth.principal.AUTH_SETTING_DEFAULTS: 17
    "entra_clearance_roles": {
        "rag.admin": "restricted",
        "rag.restricted": "restricted",
        "rag.confidential": "confidential",
        "rag.internal": "internal",
        "rag.public": "public",
    },
    "entra_clearance_groups": {},
    "entra_default_classification": "internal",
    "entra_accept_v1_issuer": True,
    "entra_required_scope": None,
    "entra_pin_tenant": True,
    "entra_jwks_refresh_min_seconds": 60.0,
    "entra_jwks_timeout_seconds": 10.0,
    "entra_group_overage_lookup": True,
    "entra_group_cache_seconds": 900.0,
    "entra_group_page_size": 999,
    "entra_group_lookup_timeout_seconds": 10.0,
    "entra_group_lookup_path": "/me/transitiveMemberOf/microsoft.graph.group",
    "entra_client_secret": None,
    "entra_managed_identity_endpoint": (
        "http://169.254.169.254/metadata/identity/oauth2/token"
    ),
    "entra_managed_identity_api_version": "2018-02-01",
    "entra_graph_token_skew_seconds": 60.0,
    # --- Addendum E, eval.EVAL_SETTING_DEFAULTS: 23
    "eval_pipeline_target": "eval.harness:run_turn",
    "eval_personas_path": "services/eval/golden/personas.yaml",
    "eval_report_dir": "services/eval/reports",
    "eval_item_timeout_seconds": 300.0,
    "eval_persist_results": True,
    "eval_run_tenant_id": None,
    "eval_skip_unregistered_tools": True,
    "eval_tool_required_live": False,
    "eval_ragas_enabled": True,
    "eval_judge_effort": "medium",
    "eval_relevancy_probe_questions": 3,
    "eval_correctness_f1_weight": 0.75,
    "eval_correctness_similarity_weight": 0.25,
    "eval_judge_max_contexts": 12,
    "eval_judge_max_context_chars": 4000,
    "eval_hard_min_acl_leak": 1.0,
    "eval_hard_min_refusal_correct": 0.95,
    "eval_min_tool_correct": 0.5,
    "eval_min_retrieval_recall": 0.7,
    "eval_regression_tolerance": 0.02,
    "eval_report_worst_items": 10,
    "eval_report_answer_chars": 600,
    "eval_score_prefix": "eval.",
    # --- Addendum P/S, read by POST /admin/ingest/trigger: 2
    "ingest_function_url": None,
    "ingest_function_key": None,
}

#: A cross-section of the promoted knobs, exported as ``RAG_<NAME>`` would be,
#: with the value the field must then carry. Covers int, float, bool, str and
#: JSON-list parsing, and every guardrail threshold the audit called out as a
#: genuine operational problem.
ENV_SAMPLE: list[tuple[str, str, Any]] = [
    ("RAG_GUARDRAIL_OOD_COLLAPSE_SPREAD", "0.12", 0.12),
    ("RAG_GUARDRAIL_OUTPUT_BLOCK_BELOW_GROUNDEDNESS", "0.45", 0.45),
    ("RAG_GUARDRAIL_OOD_COVERAGE_SAMPLE", "1024", 1024),
    ("RAG_GUARDRAIL_CONTRADICTION_LLM_ENABLED", "false", False),
    ("RAG_MEMORY_CACHE_MAX_ENTRIES", "25", 25),
    ("RAG_API_RATE_LIMIT_PREFIX", "acme:rl:", "acme:rl:"),
    ("RAG_EVAL_PIPELINE_TARGET", "acme.pipeline:run", "acme.pipeline:run"),
    (
        "RAG_GUARDRAIL_OUTPUT_PII_IGNORE_ENTITIES",
        '["DATE_TIME"]',
        ["DATE_TIME"],
    ),
]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Strip every ``RAG_*`` variable so a developer's shell cannot skew a default.

    Returns:
        The monkeypatch handle, for setting the variables under test.
    """
    for name in [key for key in os.environ if key.startswith("RAG_")]:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_every_documented_tunable_is_a_settings_field() -> None:
    """The audit's 122 knobs plus the two ingest-dispatch ones are real fields."""
    declared = Settings.model_fields
    missing = sorted(name for name in PROMOTED if name not in declared)
    assert missing == [], f"{len(missing)} documented tunables are not fields"


def test_promoted_tunables_keep_their_documented_defaults(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """Promotion must not move a single default: the suite encodes them."""
    del clean_env
    settings = Settings(_env_file=None)
    changed = {
        name: (getattr(settings, name), expected)
        for name, expected in PROMOTED.items()
        if getattr(settings, name) != expected
    }
    assert changed == {}


def test_every_promoted_tunable_documents_itself() -> None:
    """A field without a description is not a documented tunable."""
    declared = Settings.model_fields
    undocumented = sorted(
        name for name in PROMOTED if not (declared[name].description or "").strip()
    )
    assert undocumented == []


@pytest.mark.parametrize(("variable", "raw", "expected"), ENV_SAMPLE)
def test_promoted_tunables_are_env_settable(
    clean_env: pytest.MonkeyPatch, variable: str, raw: str, expected: Any
) -> None:
    """``RAG_<NAME>`` reaches the field instead of being silently discarded."""
    clean_env.setenv(variable, raw)
    settings = Settings(_env_file=None)
    assert getattr(settings, variable.removeprefix("RAG_").lower()) == expected


def test_a_groundedness_block_floor_above_the_notice_threshold_is_refused() -> None:
    """The hard floor sits underneath the annotate threshold, never above it."""
    with pytest.raises(ValueError, match="guardrail_output_block_below_groundedness"):
        Settings(
            _env_file=None,
            guardrail_min_groundedness=0.4,
            guardrail_output_block_below_groundedness=0.7,
        )


def test_retrieved_text_may_not_be_judged_more_leniently_than_the_user_turn() -> None:
    """Indirect injection is the threat that matters; its threshold must be lower."""
    with pytest.raises(
        ValueError, match="guardrail_injection_retrieved_block_threshold"
    ):
        Settings(
            _env_file=None,
            guardrail_injection_block_threshold=0.6,
            guardrail_injection_retrieved_block_threshold=0.9,
        )


def test_an_inverted_citation_span_window_is_refused() -> None:
    """A min span above the max would verify nothing at all."""
    with pytest.raises(ValueError, match="citation_min_span_chars"):
        Settings(
            _env_file=None,
            citation_min_span_chars=700,
            citation_max_span_chars=600,
        )


def test_a_default_page_size_above_the_ceiling_is_refused() -> None:
    """The ceiling is what stops an unbounded list; the default cannot beat it."""
    with pytest.raises(ValueError, match="api_default_page_size"):
        Settings(_env_file=None, api_default_page_size=500, api_max_page_size=200)


def test_answer_correctness_weights_must_sum_to_one() -> None:
    """Addendum E: the two weights are a partition, not two independent knobs."""
    with pytest.raises(ValueError, match=r"must sum to 1\.0"):
        Settings(
            _env_file=None,
            eval_correctness_f1_weight=0.75,
            eval_correctness_similarity_weight=0.5,
        )
