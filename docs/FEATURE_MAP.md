# Feature map

Each of the ten original requirements, mapped to the files and functions that
implement it and the tests that prove it, with an honest assessment of what is
production-ready and what is scaffolded.

**Test-suite context, so the "proven by" column can be read correctly.** 632 tests
pass in ~9 s. All of them are **in-process**: `pytest.mark.integration` appears nowhere
in the repository, despite `ci.yml`'s comment claiming integration-marked tests run
against the live Postgres and Qdrant services it starts. `services/api/tests` does run
the **real** `Orchestrator` through the real FastAPI app over real SQLite, with exactly
two fakes — a deterministic `FakeLLM` and a `retrieve` spy that applies
`AccessControl.permits` itself so a handler that failed to pass the principal down
returns nothing and fails loudly. So the pipeline wiring is genuinely covered; the
Qdrant query layer and the SQL repositories are not covered against real engines.
`scripts/smoke_test.py` covers that path but is not run by CI.

Legend: **Production-ready** = implemented, defended, and covered by tests that would
catch a regression. **Solid, unproven at scale** = implemented and covered, but never
exercised against a real engine/service in CI. **Scaffolded** = present and shaped
correctly but not executable or not wired.

---

## 1. Multi-tenant ACL ingestion, serverless, nightly delta refresh

| Aspect | Where |
|---|---|
| Entry points | `ingestion/pipeline.py`: `run_ingest`, `run_ingestion`, `plan_source`, `process_document`, `process_documents`, `finalize_run`, `dry_run_source`, `start_run`, `ingest_single_document`, `ingest_uploaded_document`, `resolve_sources` |
| Serverless triggers | `services/ingestion/function_app.py`: `ingest_timer`, `ingest_http`, `ingest_orchestrator`, `ingest_plan_activity`, `ingest_document_activity`, `ingest_finalize_activity`, `ingest_blob`, `ingest_retry` |
| Schedule + guard | `ragcore/settings.py`: `is_within_working_hours`, `may_start_scheduled_ingest`; `pipeline.guard_scheduled_run`; `_validate_cron` field validator |
| Delta | `ingestion/delta.py`: `classify_document`, `detect_deletions`, `manifest_entry_for`, `BlobManifestStore`, `LocalManifestStore`, `mirror_items_to_postgres`, `tombstone_documents` |
| Connectors | `ingestion/connectors/`: `blob.py`, `sharepoint.py`, `http_crawler.py`, `sql_source.py`, `base.py`, `get_connector` |
| ACL resolution | `ingestion/acl.py`: `access_control_from_sidecar`, `_from_metadata`, `_from_graph_permissions`, `merge_with_source_defaults`, `acl_fingerprint` |
| Writes | `ingestion/upsert.py`: `RunUpserter`, `QdrantChunkWriter`, `chunk_id_for`; `ragcore/vectorstore/writer.py`: `upsert_chunks`, `soft_delete_document`, `update_access_control`, `tombstone_missing`, `hard_delete_by_filter` |
| Tenant assertion | `pipeline._assert_tenant`; `writer.upsert_chunks` (`TenantMismatchError` on a batch spanning tenants) |
| CLI | `ingestion/cli.py: main`; `scripts/run_ingest_local.py` |
| Proven by | `test_delta.py` (32): `test_touched_but_unchanged_file_is_skipped_not_reindexed`, `test_acl_only_change_avoids_reembedding`, `test_classification_change_is_an_acl_change`, `test_deleted_at_source_is_a_delete`, `test_reappearing_document_is_recreated`. `test_connectors.py` (38): `test_document_id_is_deterministic_and_tenant_scoped`, `test_local_connector_resolves_the_sidecar_acl`, `test_local_connector_skips_the_read_when_the_etag_has_not_moved`, `test_registry_maps_every_source_type`. `test_chunk.py` (20). |

**Assessment.** The delta logic, ACL resolution, chunk identity and the local/blob
connectors are **production-ready**: two-tier delta (listing metadata, then content
hash), ACL-only reindex without re-embedding, ETag-conditioned manifest writes with
merge-and-retry on 412, and deletion detection that correctly refuses to fire without
a full scan.

**Solid, unproven at scale:** the SharePoint Graph delta and SQL watermark connectors
are real code with real ACL mapping, but their tests are constructed against recorded
shapes rather than a live Graph tenant or database. The HTTP crawler honours robots.txt
and conditional GET but has never been run against a large site in this repository.

**Scaffolded:** nothing in the Durable Functions wiring has ever executed — there is no
Functions host in CI and `function_app.py` has no tests. The orchestrator's
determinism discipline (no clock, config or I/O outside an activity; batch size carried
in the plan envelope) is correct by inspection, not by test. `ingest_blob`'s
longest-prefix source resolution and `ingest_retry`'s two message shapes are likewise
untested.

**Known gap:** `POST /admin/ingest/trigger` runs `run_ingest` **inline in the API
process** rather than dispatching to the Function App, so a large full scan occupies an
API replica.

---

## 2. Personalisation, long-term memory, faster retrieval for similar queries

| Aspect | Where |
|---|---|
| Long-term memory | `app/rag/memory/long_term.py`: `LongTermMemoryStore.recall`, `remember`, `write_back`, `touch`, `forget`, `expire`, `prune`, `consent_gate`, `resolve_profile` |
| Consolidation | `app/rag/memory/consolidate.py`: `MemoryConsolidator.consolidate_user`, `refresh_profile` |
| Semantic cache | `app/rag/memory/semantic_cache.py`: `SemanticCache.probe`, `store`, `evict`, `invalidate`, `fingerprint` |
| Cache re-authorisation | `ragcore/vectorstore/filters.build_acl_filter_for_chunk_ids`; `app/rag/retriever.retrieve_by_ids`; `orchestrator._cache_resolver` |
| Keying | `ragcore/models/memory.py`: `normalize_query`, `SemanticCacheEntry.make_cache_id`, `.matches`, `LongTermMemory.decayed_salience`, `.touch`; `filters.filter_fingerprint` |
| Profile / consent API | `app/routers/memory.py`: `GET/PUT /memory/profile`, `GET /memory/items`, `DELETE /memory/items/{id}`, `PUT /memory/consent` |
| Proven by | `test_memory.py` (18): `test_consent_false_reads_nothing`, `test_consent_false_writes_nothing`, `test_unknown_consent_is_treated_as_no_consent`, `test_recall_is_salience_weighted_and_skips_faded_or_expired`, `test_recall_never_crosses_the_tenant_boundary`, `test_write_back_redacts_before_storing`, `test_write_back_supersedes_a_near_duplicate_instead_of_appending`. `test_semantic_cache.py` (14): `test_hit_with_downgraded_principal_returns_fewer_chunks`, `test_all_chunks_revoked_falls_through_to_a_miss`, `test_clearance_downgrade_changes_the_fingerprint_and_misses`, `test_cross_tenant_probe_never_matches`, `test_only_the_retrieval_plan_is_cached`, `test_eviction_removes_expired_then_least_used`. |

**Assessment. Production-ready design, and the tests are the good kind.** The two
decisions that matter are both enforced and tested: the consent gate **fails closed**
on an unresolvable profile (`consent_unknown` reads and writes nothing), and the cache
stores only the plan — `test_only_the_retrieval_plan_is_cached` asserts no chunk text
or answer is persisted, and `test_hit_with_downgraded_principal_returns_fewer_chunks`
exercises the exact scenario the design exists for.

**Solid, unproven at scale:** all Qdrant interaction is faked. Salience decay,
eviction ordering and consolidation have never run over a real, aged store — a
`memory_max_per_user=500` prune pass or a 500-entry cache bucket eviction is untested
against real latency.

---

## 3. Efficient in-session context management

| Aspect | Where |
|---|---|
| Assembly | `app/rag/context.py`: `ContextAssembler.assemble`, `rank_sources`, `_pack`, `_fit`, `_shed`, `_shed_weakest_source`, `_compaction_reason`, `_tool_result_edit`, `_suppress_overflow`, `_render`, `_finalise` |
| Budget | `ContextBudget.from_settings`, `Settings.context_prompt_budget_tokens`, `.context_compact_threshold_tokens` |
| Measurement | `app/rag/memory/short_term.py: TokenCounter` (`count_text`, `count_many`, `count_message`, `count_prompt`, `peek`) → `LLMClient.count_tokens` |
| Prompt shape | `ragcore/llm/prompts.py`: `ANSWER_SYSTEM`, `render_numbered_sources`, `build_answer_user_turn` |
| Context edit | `ragcore/llm/client.clear_tool_uses_edit` |
| Surfaced to UI | `ContextStats` on the `context_stats` SSE event; `web/src/components/ContextMeter.tsx` |
| Proven by | `test_context.py` (13): `test_budget_is_never_exceeded_under_heavy_pressure`, `test_budget_respected_with_the_real_answer_prompt`, `test_suppression_preserves_pinned_turns_and_produces_a_summary`, `test_periodic_compaction_fires_without_budget_pressure`, `test_no_compaction_below_the_floor`, `test_exact_duplicate_chunks_are_dropped_and_audited`, `test_near_duplicates_are_shed_before_novel_chunks`, `test_stale_tool_results_are_cleared_by_the_context_edit`, `test_cache_control_sits_on_the_stable_prefix_only`, `test_message_sequence_is_valid_after_arbitrary_suppression`, `test_budget_is_derived_entirely_from_settings` |

**Assessment. Production-ready, and the strongest-tested part of the repository.**
Nothing is estimated — every count is an exact `count_tokens` call, and
`test_budget_is_never_exceeded_under_heavy_pressure` is an adversarial test rather than
a happy path. Shedding is dedup-aware and every drop is audited.
`test_message_sequence_is_valid_after_arbitrary_suppression` covers the failure mode
that actually bites in production (a history starting on an assistant turn after
suppression).

**Caveat:** three of the tuning knobs that shape it —
`context_compact_every_n_turns`, `context_duplicate_penalty`,
`context_fit_max_passes` — are compile-time constants, not settings.

---

## 4. API / MCP tool calling for data that is not in the index

| Aspect | Where |
|---|---|
| Registry | `app/rag/tools/registry.py`: `ToolRegistry`, `build_registry`, `get_tool_registry`, `register_tool`, `load_tool_document`, `ToolPolicy`, `LocalMcpServerSpec`, `RateLimiter`, `redact_arguments`, `tool_config`, `ToolTuning` |
| REST | `app/rag/tools/rest_tool.py`: `RestExecutor.prepare`/`execute`, `validate_arguments`, `render_template`, `project_response`, `SecretResolver`, `TokenProvider`, `CircuitBreaker` |
| MCP | `app/rag/tools/mcp_client.py`: `build_connector_request`, `RemoteMcpConnector`, `LocalMcpClient.discover`/`call`, `translate_mcp_tool`, `local_tool_name` |
| Built-ins | `app/rag/tools/builtin.py`: `search_tool_spec`, `context_tool_spec`, `qdrant_retrieve`, `filter_from_arguments`, `BuiltinExecutor` |
| Routing + dispatch | `app/rag/tools/router.py`: `decide_route`, `LoopGuard`, `ToolContext.build`, `ToolPlan.request_kwargs`, `ToolDispatcher.plan`/`screen`/`dispatch` |
| Loop | `orchestrator._generate` (interleaved with streamed generation), `_tool_request_kwargs`, `_safe_arguments`, `_tool_call_record` |
| Config | `services/api/config/tools.example.yaml` |
| Proven by | `test_tools.py` (43): `test_shipped_example_config_loads`, `test_tenant_filtering_beats_every_role`, `test_restricted_content_never_forwarded`, `test_rate_limiter_is_tenant_scoped`, `test_redact_arguments_masks_secrets_and_pii`, `test_require_raises_for_denied_tool`, `test_anthropic_tools_exclude_remote_mcp`. `test_mcp_client.py` (21). |

**Assessment. Production-ready for the REST path and the built-ins.** The gate order in
`dispatch` (exposure → loop guard → rate limit → egress screen → execute → trace →
audit) is tested, undeclared arguments are rejected with or without `jsonschema`,
path placeholders are mandatory while query/body placeholders drop out, secrets are
never values, and `execute` never raises.

**Solid, unproven:** the remote MCP connector emits the correct `mcp_servers` +
`mcp_toolset` + beta triple and is unit-tested, but has never been sent to the API —
`tool_mcp_enabled` is `false` by default. The self-hosted MCP path over the official
SDK is likewise tested against a fake session, not a real stdio server.

**Scaffolded:** `tools.example.yaml` points at unreachable example hosts, so the OAuth
on-behalf-of and managed-identity token paths, the circuit breaker under real
failures, and the response-size cap while streaming have never run against a live
endpoint. The `tool_required` golden items are skipped by default for exactly this
reason.

---

## 5. Short-term memory and periodic context suppression

| Aspect | Where |
|---|---|
| Session window | `app/rag/memory/short_term.py`: `SessionWindow` (`live_turns`, `pinned_turns`, `suppressible`, `suppress`, `pin`, `history_pairs`, `to_payload`/`from_payload`), `ShortTermMemory.load`/`save`/`record_turn`/`compact`/`persist_compaction`, `summarise_turns` |
| Stores | `InMemorySessionStore`, `RedisSessionStore`, `_session_key` (`<prefix><tenant>:<session>`) |
| Periodic trigger | `context._compaction_reason` → `"periodic"` at `context_compact_every_n_turns = 6` |
| SQL mirror | `ragcore/db/repositories.suppress_messages` (refuses to suppress a pinned turn), `list_session_messages(include_suppressed=…)` |
| API | `POST /api/v1/sessions/{id}/compact`, `GET /sessions/{id}/messages` (includes `suppressed` flags) |
| Proven by | `test_memory.py`: `test_record_turn_refuses_unredacted_content`, `test_window_round_trips_through_the_store_and_is_tenant_scoped`, `test_pinned_turns_survive_every_suppression_path`, `test_summarise_turns_degrades_to_a_deterministic_summary`. `test_context.py`: `test_periodic_compaction_fires_without_budget_pressure`, `test_no_compaction_below_the_floor`. `test_api_smoke.py`: `test_session_lifecycle`. |

**Assessment. Production-ready.** Suppression rather than truncation, a floor that is
never compacted, tenant-first keys with a mismatch discarded, `pii_redacted=True`
enforced at the store boundary, and a summariser that degrades to a deterministic
extractive summary rather than failing a turn.

**Solid, unproven:** `RedisSessionStore` is exercised through its interface, not
against a real Redis, and Redis is disabled in both the deployed environments
(`RAG_REDIS_ENABLED=false` in `containerapps.bicep`) — so the production path is
actually the in-process fallback, which means the session-window fast path does not
survive a replica switch. Correctness holds (PostgreSQL re-hydration is authoritative);
the caching benefit does not.

---

## 6. Hybrid retrieval: semantic + BM25 + reranking + metadata filtering

| Aspect | Where |
|---|---|
| Collections | `ragcore/vectorstore/collections.py`: `ensure_collections`, `CHUNK_PAYLOAD_INDEXES` (17), `_hnsw_config` (`m=0` + `payload_m`), `_sparse_vectors_config` (`Modifier.IDF`), `point_id_for_chunk` |
| Fusion | `ragcore/vectorstore/hybrid.py`: `hybrid_search` (two `Prefetch` + `FusionQuery`), `dense_search`, `resolve_fusion` |
| Embeddings | `ragcore/embeddings/fastembed_provider.py`: `FastEmbedProvider.embed_documents`/`embed_query`/`embed_sparse`/`warm_up` (thread-offloaded), `cosine_similarity` |
| Rerank | `ragcore/rerank/cross_encoder.py`: `CrossEncoderReranker`, `NoopReranker` — neither drops candidates |
| Orchestration | `app/rag/retriever.py`: `retrieve`, `retrieve_by_ids`, `_union`, `_candidate`, `_dedupe`, `_rerank`, `_mmr`, `_relevance`; drop constants `DROP_CANDIDATE_LIMIT`, `DROP_RERANK_MIN_SCORE`, `DROP_RERANK_TOP_N`, `DROP_MAX_PER_DOCUMENT`, `DROP_TOP_N`, `DROP_DELETED`, `DROP_ACL` |
| Diversity | `app/rag/mmr.py`: `maximal_marginal_relevance`, `mmr_select`, `normalise_scores` |
| Filters | `ragcore/vectorstore/filters.build_acl_filter` + `_metadata_conditions`; `MetadataFilter.merged_with`, `.fingerprint_payload` |
| Dedupe | `ragcore/dedupe.py`: `content_sha256`, `simhash64`, `simhash_hex`, `hamming64`, `is_near_duplicate`, `dedupe_chunks`, `normalise_text`, `shingles` |
| Proven by | `test_hybrid.py` (13). `test_filters.py` (28). `test_dedupe.py` (29). `test_retriever.py` (19): `test_every_candidate_is_either_kept_or_dropped_with_a_reason`, `test_scores_are_ordered_and_normalised_for_the_ood_gate`, `test_the_union_keeps_the_best_score_per_chunk`, `test_rerank_bounds_are_enforced_by_the_retriever_not_the_reranker`, `test_one_document_cannot_monopolise_the_result`, `test_mmr_diversifies_a_pool_of_near_identical_chunks`, `test_the_acl_filter_comes_from_ragcore_and_reaches_every_branch`, `test_a_cross_tenant_chunk_is_discarded_and_never_reported`. |

**Assessment. Production-ready design, and the two decisions most often got wrong are
both correct here:** `Modifier.IDF` on the sparse vector (without it BM25 silently
scores term frequency only, which looks like a working branch while ranking badly) and
server-side RRF via a single Query API call rather than Python-side fusion.
`test_every_candidate_is_either_kept_or_dropped_with_a_reason` is a strong invariant —
`kept + dropped` reconstructs the input, so no chunk vanishes silently.

**Solid, unproven at scale.** This is where the missing integration tests hurt most.
`test_hybrid.py` asserts the **request shape** sent to Qdrant; nothing in CI verifies
that Qdrant, holding two tenants' points, actually returns only one tenant's, or that
the `is_tenant` partition is being used. `scripts/bootstrap_qdrant.py` and
`scripts/smoke_test.py` do check this, and `ci.yml` starts a Qdrant service and runs
bootstrap — but no test then queries it.

Also unproven: the `sigmoid(rerank_score)` calibration of `final_score` against a real
`bge-reranker-v2-m3` (CI sets `RAG_RERANK_ENABLED=false`, so `NoopReranker` runs), and
`retrieval_fusion_score_scale = 0.05` as the RRF→[0,1] divisor, which is a
compile-time constant and is what `guardrail_ood_min_score` is compared against.

---

## 7. React interface

| Aspect | Where |
|---|---|
| Shell | `web/src/App.tsx`, `components/Layout.tsx`, `store/settings.ts` (view state) |
| Auth | `web/src/auth/msal.ts`, `auth/AuthProvider.tsx`, `hooks/usePrincipal.ts` |
| Transport | `web/src/api/client.ts` (bearer + **one** retry on 401 after a forced silent refresh), `api/sse.ts` (`fetch` + `ReadableStream`, never `EventSource` — it cannot send an `Authorization` header), `api/types.ts` |
| Chat | `hooks/useChatStream.ts`, `store/chat.ts`, `components/ChatPanel.tsx`, `Composer.tsx`, `MessageBubble.tsx`, `SessionSidebar.tsx` |
| Observability surfaces | `RetrievalInspector.tsx`, `SourceDrawer.tsx`, `CitationList.tsx`, `ContextMeter.tsx`, `GuardrailBanner.tsx`, `ToolTrace.tsx` |
| Memory / admin / eval | `MemoryPanel.tsx`, `AdminDocuments.tsx`, `AdminIngestion.tsx`, `EvalDashboard.tsx`, `FilterBar.tsx` |
| Build | `web/vite.config.ts` (dev proxy with SSE buffering disabled), `web/Dockerfile`, `web/nginx.conf` (envsubst template, `proxy_buffering off` on `/api/`) |
| Proven by | CI `web` job: `npm run typecheck` (`tsc --noEmit` on both tsconfigs) + `vite build`. **No unit, component or e2e tests.** |

**Assessment. Solid, unproven.** ~7 800 lines of TypeScript that typecheck and build
cleanly, cover every documented endpoint and every SSE event, degrade gracefully on an
absent `PUT /memory/profile` (404/405/501 → a notice that preferences can only be
learned from conversation), tolerate unknown SSE event names and unknown object keys,
and implement the documented reconnect policy (retry the whole request only while no
`token` event has arrived; after the first token, fail the turn and offer an explicit
resend, because there is no resume-from-offset contract).

**But there is not one test.** Every behavioural claim above is by inspection. There is
also no `lint` script in `web/package.json` (CI runs `npm run lint --if-present`, which
silently passes), and `engines.node` says `>=20.19` while CI uses Node 25 and the
contract specifies Node 25 / npm 11.

The `EvalDashboard` is wired to `GET /eval/runs` and `POST /eval/runs`; the latter
always returns 503 today (see #8), so the "start a run" affordance cannot succeed.

---

## 8. RAGAS + semantic-similarity validation against a golden set

| Aspect | Where |
|---|---|
| Runner | `services/eval/run_eval.py`: `run_eval`, `OrchestratorRunner`, `coerce_outcome`, `load_golden_set`, `load_personas`, `select_items`, `config_fingerprint`, `_threshold_failures`, `main` |
| Metrics | `services/eval/metrics.py`: `citation_validity`, `acl_leak`, `refusal_correct`, `tool_correct`, `retrieval_recall`, `substring_failures`, `cost_from_usages`. `services/eval/semantic.py`: `semantic_similarity`, `semantic_similarity_batch` |
| RAGAS | `services/eval/ragas_adapter.py`: `RagasAdapter.score`/`score_many`, `load_ragas`, native `_faithfulness`/`_answer_relevancy`/`_context_precision`/`_context_recall`/`_answer_correctness` |
| Gate | `services/eval/ci_gate.py`: `HARD_METRICS`, `gate_thresholds`, `evaluate_gate`, `apply_gate`, `_item_hard_failures`, `format_gate_table`, `main` |
| Reports | `services/eval/report.py`: `category_aggregate`, `worst_items`, `compare_runs`, `render_markdown`, `render_html`, `write_reports`, `load_artifacts` |
| Data | `services/eval/golden/golden_set.yaml` (59 items), `personas.yaml` (4), `golden/README.md` |
| API | `app/routers/eval.py`: `GET /eval/runs`, `GET /eval/runs/{id}`, `POST /eval/runs` |
| Proven by | `test_gate.py` (19): `test_acl_leak_and_refusal_are_the_hard_gates`, `test_configuration_can_tighten_a_hard_gate_but_not_loosen_it`, `test_single_acl_leak_fails_the_build_even_when_the_mean_looks_fine`, `test_unmeasured_metric_is_reported_but_does_not_fail`, `test_disabled_gate_still_evaluates_but_passes`. `test_metrics.py` (38): `test_golden_set_is_large_enough_and_covers_every_category`, `test_personas_match_the_seeder`, `test_items_reference_defined_personas_in_their_own_tenant`, `test_expected_documents_are_visible_to_the_persona`, `test_acl_negative_items_forbid_the_right_canaries`. |

**Assessment. SCAFFOLDED — this is the largest gap in the repository.**

The pieces are all there and are well built: the gate's hard-floor arithmetic
(`max(configured, hard)`), the per-item ACL check that an aggregate mean would hide,
the "unmeasured is not zero" rule, the golden set's binding tests against the seeder,
and the metric implementations that reuse **production** code
(`app.rag.citations.extract_citations`, `output_guard.check_clearance`,
`ragcore.embeddings`) so the harness cannot disagree with production about what
"cited" or "permitted" means.

**But it cannot execute.** `eval_pipeline_target` defaults to
`app.rag.orchestrator:run_turn`, which does not exist; `OrchestratorRunner._resolve`
raises `EvalHarnessError` and the CLI exits 2. The value cannot be set from the
environment because `eval_pipeline_target` is not a `Settings` field and `Settings` is
`extra="ignore"`. Consequently: the CI `eval` job cannot pass, the 10 `acl_negative`
items have never actually run against the pipeline, and none of the twelve thresholds
has ever been measured against a real answer.

**Also scaffolded:** `POST /api/v1/eval/runs` imports `eval.harness.run_evaluation`;
`services/eval/harness.py` does not exist, so the route always answers
`503 eval_unavailable`. And RAGAS itself is not installed (optional extra), so all five
RAGAS-family metrics would run on the native Claude-judge path — which is a supported
mode, but it means the "RAGAS" half of the requirement is untested.

Fix is small: add a `run_turn` coroutine to the orchestrator module (or make
`eval_pipeline_target` a real setting), and add `services/eval/harness.py` exposing
`run_evaluation`.

---

## 9. Guardrails: PII, out-of-domain, contradictions, dedupe, citations, lineage, Langfuse

| Aspect | Where |
|---|---|
| PII | `ragcore/pii/detector.py`: `PIIDetector.analyze`/`redact`/`scan_and_redact`/`pseudonym`/`verify`; `ragcore/pii/recognizers.py`: `CUSTOM_RECOGNIZERS`, `BASELINE_RECOGNIZERS`, `luhn_check`, `iban_check`, `verhoeff_check`, `jwt_check` |
| PII enforcement | `ragcore/logging.py`: `redact_secrets`, `guard_raw_content`, `RAW_CONTENT_KEYS`; `repositories.append_message`/`write_feedback` (`pii_redacted=True` required); `ingestion/enrich.scan_and_redact` |
| Input guard | `app/rag/guardrails/input_guard.py`: `run_input_guard`, `detect_language`, `InputDecision` (`text` vs `redacted_text`) |
| Injection | `app/rag/guardrails/injection.py`: `scan_text`, `scan_user_turn`, `scan_retrieved`, `wrap_untrusted`, `sanitise_untrusted`, `normalise_unicode`, `PATTERNS` |
| OOD | `app/rag/guardrails/ood.py`: `run_ood_gate`, `relevance_signals`, `tenant_coverage`, `fallback_refusal`, `clear_coverage_cache` |
| Contradictions | `app/rag/guardrails/contradiction.py`: `cluster_claims`, `resolve_conflict`, `check_contradictions`, `render_contradiction_notes` |
| Output guard | `app/rag/guardrails/output_guard.py`: `run_output_guard`, `check_clearance`, `citation_validity_score`, `assess_refusal` |
| Citations | `app/rag/citations.py`: `build_source_block`, `parse_markers`, `verify_span`, `extract_citations`, `strip_unresolved_markers`, `append_uncertainty_notice` |
| Dedupe | `ragcore/dedupe.py` (see #6); `ingestion/upsert.py` three-layer dedupe; `context.rank_sources` |
| Lineage | `ragcore/observability/lineage.py`: `LineageRecord`, `record_lineage`, `subject_provenance`, `document_provenance`; `orchestrator._record_lineage`; `GET /documents/{id}/lineage` |
| Langfuse | `ragcore/observability/langfuse.py`: `Tracer`, `NoopTracer`, `LangfuseTracer`, `get_tracer`, `traced`, `FAILURE_BUDGET`; `ragcore/observability/metrics.py` |
| Proven by | `test_pii.py` (31). `test_guardrails.py` (31): `test_injection_in_retrieved_chunk_is_quarantined`, `test_retrieved_threshold_is_stricter_than_the_user_turn`, `test_untrusted_wrapper_neutralises_a_forged_delimiter`, `test_sanitise_strips_invisible_smuggling`, `test_ordinary_corpus_prose_does_not_flag`, `test_input_guard_redacts_before_logging`, `test_input_guard_masks_credentials_even_in_the_prompt`. `test_contradiction.py` (15). `test_citations.py` (23): `test_a_fabricated_quote_is_not_verifiable`, `test_a_number_the_source_does_not_contain_fails_the_guard`, `test_a_faithful_paraphrase_verifies_below_full_confidence`. `test_api_smoke.py`: `test_a_blocked_turn_still_closes_the_stream`. |

**Assessment. Mostly production-ready, with two real holes.**

Production-ready: the citation verifier (exact → bounded fuzzy → numeric guard, with a
stricter `verbatim` path for explicit quotes, and a `CitationDrop` that deliberately
carries no content because stage 11 runs before the egress scan); the output guard's
fail-closed clearance check using a *second* implementation of the ACL rule; the
redact-before-log rule enforced mechanically by a structlog processor rather than by
convention; the contradiction resolver's discriminator order (`effective_from` →
`source_modified_at` → `authority` → `recency` → `indeterminate`, recency **before**
authority on purpose, because a superseded policy is still a policy); and the OOD gate's
score-collapse detector, whose coverage summary is sampled **through
`build_acl_filter`** so a refusal can never advertise a document the caller cannot see.

**Hole 1 — indirect injection has no structural containment on the answer path.**
`wrap_untrusted` is called from exactly two places, the OOD adjudicator and the
contradiction adjudicator; retrieved chunk text reaches the generation prompt through
`prompts.render_numbered_sources`, which emits a plain `<sources>` block with no
delimiter neutralisation and no `sanitise_untrusted` pass. The contract states this
wrapping is applied to "every piece of retrieved or tool-returned text that enters a
prompt". It is not. `test_untrusted_wrapper_neutralises_a_forged_delimiter` proves the
*function* works; nothing proves it is called where it matters.

**Hole 2 — Prometheus metrics are dead.** `prometheus-client` is not a `ragcore`
dependency, so `PROMETHEUS_AVAILABLE is False`, `/metrics` serves one comment line, and
all seventeen `rag_*` series — `rag_guardrail_events_total` included — record nothing.
Langfuse works when configured but is sampled, so guardrail activity is currently not
countable.

**Scaffolded:** Presidio is not installed (`pii` extra), so PII detection is
regex-only: no `PERSON` or `LOCATION`. The optional LLM PII verification, injection
classifier and OOD classifier are all off by default and untested end to end. Reads
are not audited — `audit_log` covers seven mutating actions only.

---

## 10. Query transformation

| Aspect | Where |
|---|---|
| Implementation | `app/rag/query_transform.py`: `transform_query`, `fallback_transform`, `is_abstract_query`, `merge_filters`, `TransformedQuery`, `QueryTransformPayload` |
| Prompt | `ragcore/llm/prompts.py`: `QUERY_TRANSFORM_SYSTEM`, `HYDE_SYSTEM`, `render_history` |
| Structured call | `ragcore/llm/client.LLMClient.structured` (JSON-Schema derivation, keyword stripping, `additionalProperties: false`) |
| Wiring | `orchestrator._pipeline` stage 3; `merge_filters(request.filters, plan.metadata_filter)` |
| Proven by | `test_query_transform.py` (15) |

**Assessment. Production-ready.** One `claude-sonnet-5` structured call returns intent,
`needs_retrieval`, `needs_tools`, `tool_hints`, `rewritten` (pronoun/ellipsis resolved
against the trimmed short-term window), `sub_questions`, `hyde_passage`,
`metadata_filter` and `is_out_of_domain`.

Two design choices that are right and are tested: **`transform_query` never raises** —
every failure returns `fallback_transform` with `degraded=True` and a reason in
`{disabled, empty_query, llm_error, invalid_response}`, a degraded plan keeps
`needs_retrieval=True`, and stage 6 must not read a degraded plan's
`is_out_of_domain` as evidence. And **`merge_filters` is fill-in, not intersection**:
`MetadataFilter.merged_with` would intersect `["standard"]` with `["policy"]` to `[]`,
which the validator normalises to *no constraint*, silently widening the search the
user narrowed. Classification and `exclude_pii` still combine strictly. Extracted
`doc_type`/`source_type` facets are vocabulary-checked, because a filter nothing can
match hides the answer.

---

## Cross-cutting

| Concern | Where | Assessment |
|---|---|---|
| HTTP surface | `app/main.py`, `app/middleware.py` (pure-ASGI `RequestContextMiddleware`, RFC 7807 handlers), `app/deps.py`, `app/routers/**`, `app/schemas/**` | **Production-ready.** Problem+JSON everywhere, 422 never echoes pydantic's `input`, 500 names only the exception class, 404-not-403 on cross-tenant document and session probes, startup never aborts except on dev-mode-in-production. Proven by `test_api_smoke.py` (16) and `test_auth.py` (16). |
| Model client | `ragcore/llm/client.py`, `pricing.py`, `prompts.py` | **Production-ready by inspection, 39 tests.** Beta flags, adaptive thinking, no sampling parameters, refusal handled before `content[0]` is indexed, retries only on `RateLimitError`/5xx/`APIConnectionError` with `max_retries=0` on the SDK so backoff is not doubled, `stream()` retries only while nothing has been yielded, `count_tokens` never uses `tiktoken`. Never exercised against the real API in CI (`-m "not llm"`). |
| Database | `ragcore/db/{base,models,repositories}.py`, one Alembic revision `0001` creating all 17 tables | **Solid, unproven.** No repository tests at all. Migrations run in CI against real Postgres, and `test_api_smoke.py`/`test_auth.py` exercise the repositories through the API over SQLite — so the SQL is executed, but `TenantMismatchError`, `upsert_document`'s version bump and `suppress_messages`' pinned-turn refusal have no direct tests. `JSON_TYPE` is JSONB on PostgreSQL and JSON on SQLite, so behaviour is not identical between the tested and deployed engines. |
| Infrastructure | `infra/azure/**` | **Solid, compiled, never deployed here.** CI runs `az bicep build` on the template and both parameter files with linter warnings as errors, plus `shellcheck` on `deploy.sh`. Correct on the things that matter (Qdrant pinned to one replica with mandatory Azure Files persistence, internal-only, no public PostgreSQL endpoint, no secret in any output, RBAC per identity). No deployment has been verified from this repository. |
| Configuration | `ragcore/settings.py` (208 fields, all in `.env.example`) + six in-code default tables (122 keys, **none** a settings field) | **Half-scaffolded.** The 208 real fields are complete, validated and cross-checked (`Settings` refuses to construct on dev-mode-in-production, insecure-HTTP-in-production, and unsatisfiable budget/chunk relationships). The 122 documented "fallback" keys cannot be set from the environment at all, because `Settings` is `extra="ignore"`. See [ARCHITECTURE §6](ARCHITECTURE.md#6-configuration-reality). |
| Developer entry points | `Makefile`, `scripts/**` | **Three targets broken.** `make bootstrap`/`seed`/`smoke` invoke `scripts/bootstrap.py`/`seed.py`/`smoke.py`, which do not exist; the real names are `bootstrap_qdrant.py`, `seed_demo_tenant.py`, `smoke_test.py`. The scripts themselves are good — `smoke_test.py` in particular covers the ingest→search→chat→citations path *plus* the ACL negatives, which is the only place the real Qdrant ACL path is exercised. It is not run by CI. |
| CI | `.github/workflows/ci.yml` | **Four of five jobs work.** `lint`, `test` (coverage floor 70 %), `bicep` and `web` are real gates. `eval` cannot pass (#8). The `test` job's comment claims integration-marked tests run against the live Postgres and Qdrant services it starts; no such tests exist. |

---

## Summary

| Requirement | Verdict |
|---|---|
| 1 · ACL ingestion, serverless, nightly delta | Production-ready core; Durable Functions wiring untested |
| 2 · Personalisation + memory + similar-query cache | Production-ready design and tests; Qdrant path faked |
| 3 · In-session context management | **Production-ready** — best-tested area |
| 4 · API / MCP tool calling | Production-ready for REST + built-ins; MCP unproven live |
| 5 · Short-term memory + periodic suppression | Production-ready; Redis disabled in the deployed config |
| 6 · Hybrid + rerank + metadata filtering | Production-ready design; **no integration test against a real Qdrant** |
| 7 · React interface | Complete and type-safe; **zero tests** |
| 8 · Golden-set validation + CI gate | **Scaffolded — cannot execute** |
| 9 · PII / OOD / contradictions / citations / lineage / Langfuse | Mostly production-ready; **injection wrapping missing on the answer path**, **metrics dead** |
| 10 · Query transformation | **Production-ready** |

The four changes that would move the most: add `run_turn` (or make
`eval_pipeline_target` a real setting) so the eval gate runs; fix the three Makefile
paths; add `prometheus-client`; and wrap/sanitise retrieved text in
`render_numbered_sources`. Each is a small diff, and together they turn three of the
four "scaffolded" verdicts above into working gates.
