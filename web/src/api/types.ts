/**
 * TypeScript mirrors of the pydantic contracts in `docs/CONTRACTS.md`.
 *
 * Field names match the Python models exactly (snake_case) so a response can be
 * assigned without any translation layer. Anything the API may legitimately omit is
 * declared optional rather than being invented client-side.
 */

// --------------------------------------------------------------------- ACL

/** `ragcore.models.acl.Classification` — ordered least to most sensitive. */
export type Classification = 'public' | 'internal' | 'confidential' | 'restricted';

/** Classification labels in rank order. */
export const CLASSIFICATIONS: readonly Classification[] = [
  'public',
  'internal',
  'confidential',
  'restricted',
];

/** `ragcore.models.acl.ADMIN_ROLE`. */
export const ADMIN_ROLE = 'rag.admin';

/** `ragcore.models.acl.Principal` as echoed by `GET /me`. */
export interface Principal {
  user_id: string;
  tenant_id: string;
  roles: string[];
  groups: string[];
  email: string | null;
  display_name: string | null;
  max_classification: Classification;
}

// ------------------------------------------------------------------- chunks

/**
 * `ragcore.models.chunk.ChunkPayload`.
 *
 * `text`, `contextual_header` and `summary` are optional because
 * `RetrievalResult.without_text()` strips them from the SSE `retrieval` event.
 */
export interface ChunkPayload {
  chunk_id: string;
  document_id: string;
  chunk_index: number;
  tenant_id: string;
  allowed_roles: string[];
  allowed_groups: string[];
  allowed_users: string[];
  denied_users: string[];
  classification: string;
  classification_rank: number;
  source_type: string;
  source_id: string;
  source_uri: string;
  title: string;
  section_path: string[];
  page: number | null;
  text?: string;
  contextual_header?: string;
  summary?: string | null;
  keywords: string[];
  doc_type: string;
  tags: string[];
  author: string | null;
  language: string;
  content_sha256: string;
  simhash: string;
  token_count: number;
  source_modified_at: string | null;
  effective_from: string | null;
  effective_to: string | null;
  created_at: string;
  updated_at: string;
  version: number;
  is_deleted: boolean;
  pii_types: string[];
  pii_redacted: boolean;
  ingest_run_id: string;
}

// ---------------------------------------------------------------- retrieval

/** `ragcore.models.retrieval.RetrievalStage`. */
export type RetrievalStage = 'dense' | 'sparse' | 'fusion' | 'rerank' | 'cache' | 'tool';

/** `ragcore.models.retrieval.MetadataFilter`. */
export interface MetadataFilter {
  doc_types?: string[] | null;
  source_types?: string[] | null;
  tags?: string[] | null;
  authors?: string[] | null;
  languages?: string[] | null;
  document_ids?: string[] | null;
  section_prefix?: string | null;
  date_from?: string | null;
  date_to?: string | null;
  max_classification?: Classification | null;
  exclude_pii?: boolean;
}

/** `ragcore.models.retrieval.RetrievedChunk`. */
export interface RetrievedChunk {
  payload: ChunkPayload;
  dense_score: number | null;
  sparse_score: number | null;
  fusion_score: number;
  rerank_score: number | null;
  final_score: number;
  retrieval_stage: string;
  dropped_reason: string | null;
}

/** `ragcore.models.retrieval.Citation`. */
export interface Citation {
  marker: string;
  document_id: string;
  chunk_id: string;
  title: string;
  source_uri: string;
  section_path: string[];
  page: number | null;
  quoted_span: string | null;
  char_start: number | null;
  char_end: number | null;
  confidence: number;
}

/** `ragcore.models.retrieval.RetrievalResult`. */
export interface RetrievalResult {
  chunks: RetrievedChunk[];
  queries_used: string[];
  filter_applied: Record<string, unknown>;
  total_candidates: number;
  after_dedupe: number;
  after_rerank: number;
  latency_ms: Record<string, number>;
  cache_hit: boolean;
  dropped: RetrievedChunk[];
}

/** Body of `POST /api/v1/search`. */
export interface SearchRequest {
  query: string;
  filters?: MetadataFilter | null;
  top_n?: number | null;
}

// --------------------------------------------------------------------- chat

/** `ragcore.models.chat.Role`. */
export type Role = 'user' | 'assistant';

/** `ragcore.models.tool.ToolKind`. */
export type ToolKind = 'retrieval' | 'rest' | 'mcp';

/** `ragcore.models.chat.ToolCall`. */
export interface ToolCall {
  tool_call_id: string;
  tool_name: string;
  kind: string;
  arguments: Record<string, unknown>;
  result_summary: string | null;
  is_error: boolean;
  latency_ms: number;
  created_at: string;
}

/** `ragcore.models.chat.Message`. */
export interface Message {
  message_id: string;
  session_id: string;
  role: Role;
  content: string;
  citations: Citation[];
  tool_calls: ToolCall[];
  token_count: number;
  created_at: string;
  suppressed: boolean;
  pinned: boolean;
}

/** `ragcore.models.chat.ChatRequest`. */
export interface ChatRequest {
  message: string;
  session_id?: string | null;
  filters?: MetadataFilter | null;
  allow_tools?: boolean;
  stream?: boolean;
}

/** `ragcore.models.chat.ContextStats`. */
export interface ContextStats {
  window_tokens: number;
  budget_tokens: number;
  system_tokens: number;
  history_tokens: number;
  retrieved_tokens: number;
  memory_tokens: number;
  summary_tokens: number;
  messages_live: number;
  messages_suppressed: number;
  compaction_events: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  tool_results_cleared?: number;
}

/** `ragcore.models.chat.GuardrailStage`. */
export type GuardrailStage = 'input' | 'retrieval' | 'output';

/** `ragcore.models.chat.GuardrailKind`. */
export type GuardrailKind =
  | 'pii'
  | 'injection'
  | 'ood'
  | 'contradiction'
  | 'classification'
  | 'groundedness'
  | 'size';

/** `ragcore.models.chat.GuardrailAction`. */
export type GuardrailAction = 'allow' | 'redact' | 'block' | 'warn' | 'clarify';

/** `ragcore.models.chat.GuardrailEvent`. */
export interface GuardrailEvent {
  stage: string;
  kind: string;
  action: string;
  detail: string;
  entities: string[];
  score?: number | null;
  created_at?: string;
}

// ---------------------------------------------------------------- SSE events

/** `ragcore.models.chat.SSEEvent` — the `event:` field of the chat stream. */
export type SSEEventName =
  | 'session'
  | 'retrieval'
  | 'thinking'
  | 'tool_call'
  | 'tool_result'
  | 'token'
  | 'citations'
  | 'context_stats'
  | 'guardrail'
  | 'usage'
  | 'done'
  | 'error';

/** `event: session` payload. */
export interface SessionEventPayload {
  session_id: string;
  title?: string;
}

/** `event: token` payload. */
export interface TokenEventPayload {
  text: string;
}

/** `event: thinking` payload. Accepts either an incremental or whole-block field. */
export interface ThinkingEventPayload {
  text?: string;
  delta?: string;
}

/** `event: tool_call` payload. */
export interface ToolCallEventPayload {
  tool_call_id: string;
  tool_name: string;
  kind?: string;
  arguments?: Record<string, unknown>;
  created_at?: string;
}

/** `event: tool_result` payload. */
export interface ToolResultEventPayload {
  tool_call_id: string;
  tool_name?: string;
  kind?: string;
  is_error?: boolean;
  latency_ms?: number;
  result_summary?: string | null;
  error_message?: string | null;
  http_status?: number | null;
  truncated?: boolean;
}

/** `event: citations` payload — a bare list or an envelope. */
export type CitationsEventPayload = Citation[] | { citations: Citation[] };

/** `event: usage` payload — `ragcore.llm.LLMUsage` plus derived cost. */
export interface UsageEventPayload {
  model?: string;
  input_tokens?: number;
  output_tokens?: number;
  cache_read_tokens?: number;
  cache_write_tokens?: number;
  cost_usd?: number;
  latency_ms?: number;
}

/** `event: done` payload. */
export interface DoneEventPayload {
  session_id?: string;
  message_id?: string;
  stop_reason?: string;
  refused?: boolean;
  trace_id?: string | null;
}

/** `event: error` payload. */
export interface ErrorEventPayload {
  message?: string;
  detail?: string;
  error?: string;
  code?: string;
}

/** Discriminated union of everything `useChatStream` understands. */
export type ChatStreamEvent =
  | { type: 'session'; data: SessionEventPayload }
  | { type: 'retrieval'; data: RetrievalResult }
  | { type: 'thinking'; data: ThinkingEventPayload }
  | { type: 'tool_call'; data: ToolCallEventPayload }
  | { type: 'tool_result'; data: ToolResultEventPayload }
  | { type: 'token'; data: TokenEventPayload }
  | { type: 'citations'; data: Citation[] }
  | { type: 'context_stats'; data: ContextStats }
  | { type: 'guardrail'; data: GuardrailEvent }
  | { type: 'usage'; data: UsageEventPayload }
  | { type: 'done'; data: DoneEventPayload }
  | { type: 'error'; data: ErrorEventPayload }
  | { type: 'unknown'; name: string; data: unknown };

/** Non-streaming (`stream: false`) chat response. */
export interface ChatResponse {
  session_id: string;
  message: Message;
  retrieval?: RetrievalResult | null;
  context_stats?: ContextStats | null;
  guardrails?: GuardrailEvent[];
  usage?: UsageEventPayload | null;
  trace_id?: string | null;
}

// ----------------------------------------------------------------- sessions

/** A row of `GET /api/v1/sessions`, mirroring the `chat_sessions` table. */
export interface SessionSummary {
  session_id: string;
  tenant_id?: string;
  user_id?: string;
  title: string;
  created_at?: string;
  updated_at?: string;
  last_message_at?: string | null;
  message_count?: number;
  compaction_events?: number;
  summary_tokens?: number;
  rolling_summary?: string;
  total_input_tokens?: number;
  total_output_tokens?: number;
  total_cache_read_tokens?: number;
  total_cost_usd?: number;
  is_archived?: boolean;
}

/** Result of `POST /api/v1/sessions/{id}/compact`. */
export interface CompactionResult {
  session_id: string;
  messages_suppressed?: number;
  compaction_events?: number;
  summary_tokens?: number;
  context_stats?: ContextStats | null;
}

// ------------------------------------------------------------------ memory

/** `ragcore.models.memory.MemoryKind`. */
export type MemoryKind = 'preference' | 'fact' | 'entity' | 'episode';

/** `ragcore.models.memory.LongTermMemory`. */
export interface LongTermMemory {
  memory_id: string;
  user_id: string;
  tenant_id: string;
  kind: MemoryKind;
  text: string;
  salience: number;
  source_session_id: string | null;
  supersedes?: string | null;
  hit_count: number;
  created_at: string;
  last_used_at: string | null;
  expires_at: string | null;
  pii_redacted: boolean;
}

/** `ragcore.models.memory.UserProfile`. */
export interface UserProfile {
  user_id: string;
  tenant_id: string;
  summary: string;
  preferred_style: string | null;
  preferred_language: string | null;
  top_topics: string[];
  memory_consent: boolean;
  updated_at: string;
}

/** Body of `PUT /api/v1/memory/profile` (Addendum B). */
export interface ProfileUpdate {
  summary?: string;
  preferred_style?: string | null;
  preferred_language?: string | null;
  top_topics?: string[];
}

// --------------------------------------------------------------- documents

/** A row of `GET /api/v1/documents`, mirroring the `documents` table. */
export interface DocumentSummary {
  document_id: string;
  tenant_id?: string;
  source_id?: string | null;
  source_type: string;
  source_uri: string;
  title: string;
  doc_type: string;
  language: string;
  author?: string | null;
  classification: string;
  classification_rank?: number;
  tags: string[];
  chunk_count: number;
  token_count: number;
  page_count?: number | null;
  size_bytes?: number;
  version: number;
  content_sha256?: string;
  is_deleted: boolean;
  pii_redacted?: boolean;
  pii_types?: string[];
  source_modified_at?: string | null;
  created_at?: string;
  updated_at?: string;
  blob_url?: string | null;
  ingest_run_id?: string | null;
}

/** Metadata accompanying a `POST /api/v1/documents` multipart upload. */
export interface UploadOptions {
  title?: string;
  doc_type?: string;
  tags?: string[];
  classification?: Classification;
  language?: string;
  allowed_roles?: string[];
  allowed_groups?: string[];
}

/** `ragcore.observability.lineage.LineageRecord`. */
export interface LineageRecord {
  lineage_id: string;
  kind: string;
  tenant_id: string;
  user_id?: string | null;
  session_id?: string | null;
  trace_id?: string | null;
  subject_id: string;
  parents: string[];
  operation: string;
  actor: string;
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown>;
  metrics: Record<string, number>;
  created_at: string;
}

/**
 * `GET /api/v1/documents/{id}/lineage`.
 *
 * `document_provenance()` returns an open mapping, so unknown keys are preserved and
 * rendered generically instead of being dropped.
 */
export interface DocumentProvenance {
  document_id?: string;
  source_uri?: string;
  records?: LineageRecord[];
  chunk_ids?: string[];
  [key: string]: unknown;
}

// --------------------------------------------------------------- ingestion

/** `ragcore.models.document.IngestTrigger`. */
export type IngestTrigger = 'timer' | 'queue' | 'http' | 'manual' | 'reindex' | 'upload';

/** `ragcore.models.document.IngestStatus`. */
export type IngestStatus = 'running' | 'succeeded' | 'failed' | 'partial' | 'skipped';

/** `ragcore.models.document.IngestRunSummary`. */
export interface IngestRunSummary {
  run_id: string;
  tenant_id: string;
  source_id: string | null;
  trigger: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  documents_seen: number;
  documents_created: number;
  documents_updated: number;
  documents_deleted: number;
  documents_skipped: number;
  documents_failed: number;
  chunks_upserted: number;
  chunks_deleted: number;
  tokens_embedded: number;
  duplicates_dropped: number;
  pii_documents: number;
  forced?: boolean;
  within_working_hours?: boolean;
  skip_reason?: string | null;
  error_message?: string | null;
  errors?: string[];
  metrics?: Record<string, number>;
}

/** A row of `GET /api/v1/admin/sources`, mirroring `source_configs`. */
export interface SourceConfigSummary {
  source_id: string;
  tenant_id: string;
  source_type: string;
  name: string;
  enabled: boolean;
  doc_type?: string;
  tags?: string[];
  default_classification?: string;
  cron_override?: string | null;
  timezone_override?: string | null;
  cursor?: string | null;
  cursor_updated_at?: string | null;
  last_run_id?: string | null;
  last_run_at?: string | null;
  last_status?: string | null;
}

/** `GET /api/v1/admin/schedule`. */
export interface ScheduleInfo {
  ingest_cron?: string;
  ingest_timezone?: string;
  ingest_enabled?: boolean;
  ingest_working_hours_start?: number;
  ingest_working_hours_end?: number;
  within_working_hours?: boolean;
  may_start?: boolean;
  reason?: string;
  next_run_at?: string | null;
}

/** Body of `POST /api/v1/admin/ingest/trigger`. */
export interface IngestTriggerRequest {
  source_id?: string | null;
  force?: boolean;
  full_scan?: boolean;
}

// ------------------------------------------------------------------- eval

/** `ragcore.models.eval.EvalCategory`. */
export type EvalCategory =
  | 'in_domain'
  | 'out_of_domain'
  | 'pii'
  | 'contradiction'
  | 'acl_negative'
  | 'tool_required';

/** `ragcore.models.eval.MetricScores`. */
export interface MetricScores {
  faithfulness: number | null;
  answer_relevancy: number | null;
  context_precision: number | null;
  context_recall: number | null;
  answer_correctness: number | null;
  semantic_similarity: number | null;
  citation_validity: number | null;
  acl_leak: number | null;
  refusal_correct: number | null;
  latency_ms: number;
  cost_usd: number;
}

/** `ragcore.models.eval.EvalResult`, plus the `category` column of `eval_results`. */
export interface EvalResultRow {
  item_id: string;
  answer: string;
  retrieved_chunk_ids: string[];
  scores: Partial<MetricScores>;
  passed: boolean;
  failures: string[];
  trace_id?: string | null;
  category?: string | null;
  question?: string | null;
  latency_ms?: number;
  cost_usd?: number;
}

/** `ragcore.models.eval.EvalRun`, plus the counters stored on `eval_runs`. */
export interface EvalRun {
  run_id: string;
  started_at: string;
  finished_at: string | null;
  git_sha?: string | null;
  config_fingerprint?: string;
  golden_set_path?: string | null;
  results?: EvalResultRow[];
  aggregate: Record<string, number>;
  gate_passed: boolean;
  item_count?: number;
  passed_count?: number;
  failed_count?: number;
  total_cost_usd?: number;
  notes?: string | null;
}

/** Body of `POST /api/v1/eval/runs`. */
export interface EvalRunRequest {
  golden_set_path?: string | null;
  sample_size?: number | null;
  notes?: string | null;
}

/** Pass/fail tally for one golden-item category. */
export interface CategoryTally {
  category: string;
  total: number;
  passed: number;
  failed: number;
}

// ---------------------------------------------------------------- feedback

/** Body of `POST /api/v1/feedback`. `rating` is +1 or -1 (`feedback.rating`). */
export interface FeedbackRequest {
  session_id?: string | null;
  message_id?: string | null;
  rating: 1 | -1;
  comment?: string | null;
  tags?: string[];
}

/** `GET /health` / `GET /readyz`. */
export interface HealthResponse {
  status?: string;
  checks?: Record<string, boolean | string>;
  [key: string]: unknown;
}
