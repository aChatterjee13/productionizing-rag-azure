# CONTRACTS — authoritative interface spec

Every service in this repo is built against this file. **Do not invent alternative
signatures.** If something you need is missing, add it here in the same style, then
implement it.

Stack decisions (fixed):

- **Cloud:** Azure. Qdrant on Azure Container Apps; ingestion on Azure Functions
  (Python v2 programming model, Durable Functions, timer + queue + HTTP triggers);
  Azure Database for PostgreSQL Flexible Server; Azure Blob Storage; Azure Key Vault;
  Log Analytics. Langfuse self-hosted (Container App) or Langfuse Cloud.
- **Embeddings / sparse / rerank:** local FastEmbed only.
  - dense: `BAAI/bge-m3` (1024-dim, cosine)
  - sparse: `Qdrant/bm25` (true BM25 with IDF modifier, computed in Qdrant)
  - rerank: `Xenova/bge-reranker-v2-m3` via FastEmbed `TextCrossEncoder`
- **Identity:** Microsoft Entra ID only. RS256 JWT validated against Entra JWKS.
  React app uses MSAL. Tenant + roles + groups come from token claims.
- **Ingestion connectors (all four are real, not stubs):** Azure Blob + local
  filesystem, SharePoint/OneDrive via Microsoft Graph delta query, HTTP/sitemap
  crawler with conditional GET, SQL source with watermark column.
- **LLM:** Anthropic Claude. See `LLM_FACTS` below.

Environment: Python 3.13, `uv` for dependency management, Node 25 / npm 11 for the
web app. **Docker is not installed on the build machine** — write `docker-compose.yml`
and `Dockerfile`s, but never try to run them.

---

## Repository layout

```
productionizing-rag/
├── pyproject.toml                # uv workspace root
├── docker-compose.yml
├── Makefile
├── .env.example
├── .gitignore
├── README.md
├── packages/ragcore/             # shared library: contracts + infrastructure
├── services/api/                 # FastAPI: chat, search, admin, memory, eval
├── services/ingestion/           # Azure Functions: serverless delta ingestion
├── services/eval/                # RAGAS + semantic-similarity validation harness
├── web/                          # React + Vite + TS + Tailwind + MSAL
├── infra/azure/                  # Bicep IaC
├── scripts/                      # bootstrap, seed, smoke test
├── .github/workflows/ci.yml
└── docs/
```

`packages/ragcore` is importable as `ragcore`. `services/api` is importable as
`app`. Both are uv workspace members; `ragcore` is a path dependency of the others.

---

## LLM_FACTS (Anthropic API — current, do not substitute from memory)

Model IDs are exact and carry **no date suffix**:

| Constant | Model ID | Use |
|---|---|---|
| `MODEL_MAIN` | `claude-opus-5` | answer generation, agentic tool loop, contradiction resolution |
| `MODEL_FAST` | `claude-sonnet-5` | query transformation, summarisation, memory extraction |
| `MODEL_CHEAP` | `claude-haiku-4-5` | classification: routing, OOD, PII verification (200K ctx) |

Rules that are easy to get wrong:

- **Adaptive thinking:** `thinking={"type": "adaptive"}`. On `claude-opus-5` thinking
  is **on by default** — omitting the field runs adaptive.
  `thinking={"type": "enabled", "budget_tokens": N}` returns **400**.
  `thinking={"type": "disabled"}` is accepted only at effort `high` or lower.
- **Effort:** `output_config={"effort": "low"|"medium"|"high"|"xhigh"|"max"}` —
  nested inside `output_config`, never top-level. Default `high`.
- **No sampling parameters.** `temperature`, `top_p`, `top_k` return **400** on
  `claude-opus-5`. Steer with prompting.
- **No assistant prefill.** A trailing `role: "assistant"` message returns 400.
  Use structured outputs instead.
- **Structured output:** `output_config={"format": {"type": "json_schema", "schema": ...}}`,
  or `client.messages.parse(..., output_format=PydanticModel)` →
  `response.parsed_output`.
- **Prompt caching:** `cache_control={"type": "ephemeral"}` on the last stable block.
  Render order is `tools` → `system` → `messages`; anything before a breakpoint that
  changes per request destroys the cache. Minimum cacheable prefix on
  `claude-opus-5` is **512 tokens**. Verify with
  `response.usage.cache_read_input_tokens`.
- **Streaming:** use `client.messages.stream(...)`; `stream.get_final_message()` for
  the accumulated message. Stream anything with `max_tokens > ~16000`.
- **Token counting:** `client.messages.count_tokens(model=..., system=..., messages=...)`
  → `.input_tokens`. **Never use `tiktoken`** — it is the wrong tokenizer for Claude.
- **MCP connector (remote MCP servers, server-side):** requires **both** halves plus
  the beta flag, on the beta endpoint:
  ```python
  client.beta.messages.create(
      betas=["mcp-client-2025-11-20"],
      mcp_servers=[{"type": "url", "name": "svc", "url": "https://…/mcp",
                    "authorization_token": "…"}],
      tools=[{"type": "mcp_toolset", "mcp_server_name": "svc"}],
      ...)
  ```
  Passing `mcp_servers` without a matching `mcp_toolset` entry is a validation error.
- **Refusal handling:** a declined request returns HTTP 200 with
  `stop_reason == "refusal"` and possibly-`None` `stop_details`. **Check
  `stop_reason` before reading `response.content`** — indexing `content[0]`
  unconditionally crashes. Opt into server-side fallback by default:
  `client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", ...)`.
- **Context editing (beta `context-management-2025-06-27`):**
  `context_management={"edits": [{"type": "clear_tool_uses_20250919"}]}` clears old
  tool results. **Compaction is separate** — `{"type": "compact_20260112"}` with beta
  `compact-2026-01-12`; when using compaction you must append the whole
  `response.content` (compaction blocks included) back into `messages`.
- **Errors:** catch a most-specific-first chain —
  `anthropic.NotFoundError` → `RateLimitError` → `APIStatusError` →
  `APIConnectionError`. Do not string-match error messages.
- **`max_tokens` defaults:** 16000 non-streaming, 64000 streaming, 256 for
  classification calls.

---

## `ragcore.settings`

`pydantic-settings` `BaseSettings`, `env_prefix="RAG_"`, reads `.env`. Exposed as
`get_settings()` (`functools.lru_cache`). Field groups: `qdrant_*`, `postgres_*`,
`redis_*`, `anthropic_*`, `entra_*`, `langfuse_*`, `azure_*`, `embedding_*`,
`rerank_*`, `retrieval_*`, `context_*`, `memory_*`, `guardrail_*`, `ingest_*`,
`eval_*`. Every tunable in this document is a settings field with a sane default —
nothing is hard-coded at a call site.

Schedule config (requirement #1, "delta refresh daily outside working hours,
configurable"): `ingest_cron` (default `"0 30 2 * * *"` — 02:30, six-field NCRONTAB
as Azure Functions requires), `ingest_timezone` (default `"UTC"`),
`ingest_working_hours_start` / `_end` (default 8 / 18), `ingest_max_parallel_docs`,
`ingest_batch_size`, `ingest_enabled`. A guard refuses to start a scheduled run
inside working hours unless `ingest_force` is set.

## `ragcore.models.acl`

```python
class Classification(StrEnum):          # ordered, comparable via .rank
    PUBLIC = "public"; INTERNAL = "internal"
    CONFIDENTIAL = "confidential"; RESTRICTED = "restricted"

class AccessControl(BaseModel):
    tenant_id: str
    allowed_roles: list[str] = []       # empty == no role restriction
    allowed_groups: list[str] = []      # Entra group object IDs
    allowed_users: list[str] = []       # Entra user object IDs (oid)
    denied_users: list[str] = []        # explicit deny always wins
    classification: Classification = Classification.INTERNAL

class Principal(BaseModel):
    user_id: str                        # Entra `oid`
    tenant_id: str                      # Entra `tid`
    roles: list[str] = []               # app roles
    groups: list[str] = []              # group object IDs
    email: str | None = None
    display_name: str | None = None
    max_classification: Classification = Classification.INTERNAL
    def clearance_rank(self) -> int
    def is_admin(self) -> bool          # "rag.admin" in roles
```

Semantics: a chunk is visible iff `tenant_id` matches **and** the principal is not in
`denied_users` **and** (`allowed_roles`/`allowed_groups`/`allowed_users` are all empty
OR the principal matches at least one) **and**
`chunk.classification.rank <= principal.clearance_rank()`.

## `ragcore.models.chunk`

`ChunkPayload` is the Qdrant payload. Flat — nested objects cannot be payload-indexed.

```python
class ChunkPayload(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    tenant_id: str
    allowed_roles: list[str]
    allowed_groups: list[str]
    allowed_users: list[str]
    denied_users: list[str]
    classification: str  # Classification value
    classification_rank: int  # denormalised for range filtering
    source_type: str  # blob|sharepoint|http|sql|upload
    source_id: str
    source_uri: str
    title: str
    section_path: list[str]
    page: int | None
    text: str
    contextual_header: str  # prepended at embed time, stored separately
    summary: str | None
    keywords: list[str]
    doc_type: str
    tags: list[str]
    author: str | None
    language: str
    content_sha256: str
    simhash: str  # 16-hex-char unsigned 64-bit
    token_count: int
    source_modified_at: datetime | None
    effective_from: datetime | None  # recency / contradiction resolution
    effective_to: datetime | None
    created_at: datetime
    updated_at: datetime
    version: int
    is_deleted: bool
    pii_types: list[str]
    pii_redacted: bool
    ingest_run_id: str
```

## `ragcore.models.retrieval`

```python
class MetadataFilter(BaseModel):
    doc_types: list[str] | None = None
    source_types: list[str] | None = None
    tags: list[str] | None = None
    authors: list[str] | None = None
    languages: list[str] | None = None
    document_ids: list[str] | None = None
    section_prefix: str | None = None
    date_from: datetime | None = None  # against source_modified_at
    date_to: datetime | None = None
    max_classification: Classification | None = None
    exclude_pii: bool = False


class RetrievedChunk(BaseModel):
    payload: ChunkPayload
    dense_score: float | None
    sparse_score: float | None
    fusion_score: float
    rerank_score: float | None
    final_score: float
    retrieval_stage: str  # dense|sparse|fusion|rerank|cache|tool
    dropped_reason: str | None = None  # set for audited drops, never silent


class Citation(BaseModel):
    marker: str  # "[1]"
    document_id: str
    chunk_id: str
    title: str
    source_uri: str
    section_path: list[str]
    page: int | None
    quoted_span: str | None  # verbatim supporting text
    char_start: int | None
    char_end: int | None
    confidence: float


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk]
    queries_used: list[str]  # after transformation
    filter_applied: dict
    total_candidates: int
    after_dedupe: int
    after_rerank: int
    latency_ms: dict[str, float]
    cache_hit: bool = False
```

## `ragcore.models.chat`

```python
class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    message_id: str
    session_id: str
    role: Role
    content: str
    citations: list[Citation] = []
    tool_calls: list[ToolCall] = []
    token_count: int = 0
    created_at: datetime
    suppressed: bool = False  # dropped from the live window
    pinned: bool = False  # never suppressed


class ToolCall(BaseModel):
    tool_call_id: str
    tool_name: str
    kind: str  # rest|mcp|retrieval
    arguments: dict
    result_summary: str | None
    is_error: bool = False
    latency_ms: float
    created_at: datetime


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    filters: MetadataFilter | None = None
    allow_tools: bool = True
    stream: bool = True


class ContextStats(BaseModel):
    """Surfaced to the UI — requirement #3/#5 must be observable, not implicit."""

    window_tokens: int
    budget_tokens: int
    system_tokens: int
    history_tokens: int
    retrieved_tokens: int
    memory_tokens: int
    summary_tokens: int
    messages_live: int
    messages_suppressed: int
    compaction_events: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class GuardrailEvent(BaseModel):
    stage: str  # input|retrieval|output
    kind: str  # pii|injection|ood|contradiction|classification|groundedness
    action: str  # allow|redact|block|warn|clarify
    detail: str
    entities: list[str] = []
```

**SSE event names** emitted by `POST /api/v1/chat` (`event:` field, JSON `data:`):
`session` (session_id), `retrieval` (RetrievalResult minus text), `thinking`,
`tool_call`, `tool_result`, `token` (`{"text": …}`), `citations`, `context_stats`,
`guardrail`, `usage`, `done`, `error`. The web client must handle unknown events by
ignoring them.

## `ragcore.models.memory`

```python
class MemoryKind(StrEnum):
    PREFERENCE = "preference"  # "answer in bullet points"
    FACT = "fact"  # "I work in the Munich office"
    ENTITY = "entity"  # salient entities the user cares about
    EPISODE = "episode"  # summary of a past resolved conversation


class LongTermMemory(BaseModel):
    memory_id: str
    user_id: str
    tenant_id: str
    kind: MemoryKind
    text: str
    salience: float = 0.5  # 0..1, decays; boosted on reuse
    source_session_id: str | None
    supersedes: str | None = None  # memory_id this replaces
    hit_count: int = 0
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None
    pii_redacted: bool = False


class UserProfile(BaseModel):
    user_id: str
    tenant_id: str
    summary: str  # rolling LLM-maintained persona
    preferred_style: str | None
    preferred_language: str | None
    top_topics: list[str] = []
    memory_consent: bool = True  # user can switch long-term memory off
    updated_at: datetime


class SemanticCacheEntry(BaseModel):
    """Requirement #2: faster retrieval for similar queries — caches the *retrieval
    plan and chunk ids*, never the rendered answer, so ACLs are re-checked on reuse."""

    cache_id: str
    tenant_id: str
    user_id: str | None
    normalized_query: str
    transformed_queries: list[str]
    chunk_ids: list[str]
    filter_fingerprint: str  # hash of MetadataFilter + principal clearance
    hit_count: int = 0
    created_at: datetime
    last_used_at: datetime
    ttl_seconds: int
```

Cache reuse rule: a hit requires cosine similarity ≥ `memory_cache_threshold`
(default `0.94`) **and** an exact `filter_fingerprint` match. Cached `chunk_ids` are
re-fetched through the normal ACL filter — a principal who lost access gets fewer
chunks rather than a stale leak.

## `ragcore.models.eval`

```python
class GoldenItem(BaseModel):
    item_id: str
    question: str
    ground_truth: str
    expected_document_ids: list[str] = []
    expected_chunk_ids: list[str] = []
    must_contain: list[str] = []
    must_not_contain: list[str] = []
    as_user: str  # persona key from the golden file
    tenant_id: str
    category: (
        str  # in_domain|out_of_domain|pii|contradiction|acl_negative|tool_required
    )
    expect_refusal: bool = False
    expect_tool: str | None = None


class MetricScores(BaseModel):
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None
    answer_correctness: float | None
    semantic_similarity: float | None  # sentence-embedding cosine
    citation_validity: float | None  # cited spans exist in cited chunks
    acl_leak: float | None  # 1.0 == no leak
    refusal_correct: float | None
    latency_ms: float
    cost_usd: float


class EvalResult(BaseModel):
    item_id: str
    answer: str
    retrieved_chunk_ids: list[str]
    scores: MetricScores
    passed: bool
    failures: list[str]
    trace_id: str | None


class EvalRun(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: datetime | None
    git_sha: str | None
    config_fingerprint: str
    results: list[EvalResult]
    aggregate: dict[str, float]
    gate_passed: bool
```

---

## `ragcore.vectorstore`

```python
# collections.py
CHUNKS = "rag_chunks"; MEMORIES = "rag_memories"; SEMANTIC_CACHE = "rag_semantic_cache"
DENSE = "dense"; SPARSE = "sparse"

async def get_client(settings) -> AsyncQdrantClient
async def ensure_collections(client, settings) -> None
```

`ensure_collections` is idempotent and creates:

- `rag_chunks` — named vectors `dense` (1024, COSINE) and `sparse`
  (`SparseVectorParams(modifier=Modifier.IDF)` — **required** for BM25 scoring),
  `hnsw_config.payload_m` set with `m=0` on the global index so tenant-scoped search
  stays fast, and payload indexes on: `tenant_id` (keyword, `is_tenant=True`),
  `allowed_roles`, `allowed_groups`, `allowed_users`, `denied_users`,
  `classification_rank` (integer), `document_id`, `source_type`, `doc_type`, `tags`,
  `language`, `author`, `is_deleted` (bool), `content_sha256`,
  `source_modified_at` (datetime), `section_path`, `pii_types`.
- `rag_memories` — dense only; payload indexes `tenant_id` (`is_tenant=True`),
  `user_id`, `kind`, `expires_at`.
- `rag_semantic_cache` — dense only; payload indexes `tenant_id` (`is_tenant=True`),
  `user_id`, `filter_fingerprint`.

```python
# filters.py — the single security chokepoint. Nothing else may build ACL filters.
def build_acl_filter(principal: Principal,
                     extra: MetadataFilter | None = None,
                     *, include_deleted: bool = False) -> qm.Filter

def build_memory_filter(principal: Principal, kinds=None) -> qm.Filter
def build_cache_filter(principal: Principal, fingerprint: str) -> qm.Filter
def filter_fingerprint(principal: Principal, extra: MetadataFilter | None) -> str
```

`build_acl_filter` composition: `must` = tenant match + `is_deleted == False` +
`classification_rank <= principal.clearance_rank()` + every `MetadataFilter` clause;
`must_not` = `denied_users` contains `principal.user_id`; `should` (with
`min_should=1`) = the permissive branch — `allowed_users` matches, or `allowed_groups`
intersects, or `allowed_roles` intersects, or the document is unrestricted (all three
lists empty, expressed as `IsEmpty` conditions). Ship unit tests that prove a
principal from tenant A can never match a tenant-B point and that `denied_users`
overrides an otherwise-matching group.

```python
# hybrid.py
async def hybrid_search(client, *, collection: str, query_text: str,
                        dense: list[float], sparse: SparseVec,
                        qfilter: qm.Filter, limit: int,
                        prefetch_limit: int, fusion: str = "rrf") -> list[qm.ScoredPoint]
```

Implemented with the Qdrant **Query API**: two `Prefetch` branches (dense `using=DENSE`,
sparse `using=SPARSE`), each carrying the same `qfilter`, fused server-side via
`FusionQuery(fusion=Fusion.RRF)` (or `DBSF` when `fusion="dbsf"`). Never fuse in
Python — the point of server-side fusion is that filtering and scoring stay in Qdrant.

## `ragcore.embeddings`

```python
class SparseVec(BaseModel): indices: list[int]; values: list[float]
class Embedded(BaseModel): dense: list[float]; sparse: SparseVec

class EmbeddingProvider(Protocol):
    dim: int
    async def embed_documents(self, texts: Sequence[str]) -> list[Embedded]
    async def embed_query(self, text: str) -> Embedded
    async def embed_dense(self, texts: Sequence[str]) -> list[list[float]]

def get_embedding_provider(settings) -> EmbeddingProvider   # cached singleton
```

`FastEmbedProvider` wraps `TextEmbedding("BAAI/bge-m3")` and
`SparseTextEmbedding("Qdrant/bm25")`. FastEmbed is synchronous and CPU-bound — run it
in a thread (`anyio.to_thread.run_sync`) so the event loop is never blocked, and batch
by `embedding_batch_size`. Models are lazily loaded once per process.

## `ragcore.rerank`

```python
class RerankResult(BaseModel): index: int; score: float
class Reranker(Protocol):
    async def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RerankResult]
def get_reranker(settings) -> Reranker
```

`CrossEncoderReranker` wraps FastEmbed `TextCrossEncoder("Xenova/bge-reranker-v2-m3")`,
also thread-offloaded. Include a `NoopReranker` selected by
`rerank_enabled=False` for cheap CI runs.

## `ragcore.llm`

```python
@dataclass
class LLMUsage: input_tokens: int; output_tokens: int
                cache_read_tokens: int; cache_write_tokens: int
                model: str
    def cost_usd(self) -> float          # per-model table; cache reads at 0.1x

@dataclass
class LLMResponse: text: str; tool_calls: list[dict]; stop_reason: str
                   usage: LLMUsage; refused: bool; raw: Any

class LLMClient:
    async def complete(self, *, system, messages, tools=None, mcp_servers=None,
                       model=MODEL_MAIN, effort="high", max_tokens=16000,
                       cache_system=True, thinking=True,
                       context_management=None) -> LLMResponse
    def stream(self, *, ... same ...) -> AsyncIterator[StreamEvent]
    async def structured(self, *, system, messages, schema: type[T],
                         model=MODEL_FAST, effort="medium") -> T
    async def classify(self, *, system, text, labels: list[str]) -> str   # MODEL_CHEAP
    async def count_tokens(self, *, system, messages, model=MODEL_MAIN) -> int

def get_llm_client(settings) -> LLMClient
```

`LLMClient` obligations: always set `betas=["server-side-fallback-2026-07-01"]` +
`fallbacks="default"`; check `stop_reason == "refusal"` and return
`refused=True` with empty text rather than raising or indexing `content[0]`; put
`cache_control` on the final system block only; never pass `temperature`/`top_p`;
retry `RateLimitError`/5xx with jittered backoff; record every call as a Langfuse
generation with model, usage, cost and latency.

`ragcore.llm.prompts` holds every system prompt as a module-level constant with a
`PROMPT_VERSION` string, so prompts are cacheable (stable prefix) and traceable.

## `ragcore.observability`

```python
class Tracer:
    def trace(self, name, *, user_id=None, session_id=None, tenant_id=None,
              tags=None, metadata=None) -> AbstractAsyncContextManager[TraceHandle]
    def span(self, name, *, input=None, metadata=None) -> AbstractAsyncContextManager[SpanHandle]
    def generation(self, name, *, model, input, output, usage, metadata=None) -> None
    def score(self, *, name, value, comment=None, trace_id=None) -> None
    def current_trace_id(self) -> str | None
def get_tracer(settings) -> Tracer
def traced(name: str)            # decorator for async functions
```

Must degrade to a no-op tracer when `langfuse_enabled=False` or keys are absent — the
app never fails because observability is down. Trace/span ids propagate via
`contextvars` so nothing has to thread a handle through call signatures.

```python
# lineage.py — requirement #9 "complete traceability and lineage"
class LineageRecord(BaseModel):
    lineage_id: str; kind: str      # ingest|retrieval|generation|tool|eval
    tenant_id: str; user_id: str | None; session_id: str | None
    trace_id: str | None
    subject_id: str                 # document_id / chunk_id / message_id
    parents: list[str]              # upstream ids — chunk -> document -> source_uri
    operation: str; actor: str
    inputs: dict; outputs: dict; metrics: dict
    created_at: datetime
async def record_lineage(session, record) -> None
async def document_provenance(session, document_id) -> dict   # full upstream chain
```

## `ragcore.pii`

```python
class PIIFinding(BaseModel): entity_type: str; start: int; end: int
                             score: float; snippet: str
class PIIReport(BaseModel):
    findings: list[PIIFinding]; entity_types: list[str]
    has_pii: bool; max_score: float
class PIIDetector:
    def analyze(self, text: str, *, language="en") -> PIIReport
    def redact(self, text: str, report: PIIReport, *, mode="mask") -> str
    async def verify(self, text, report) -> PIIReport     # optional LLM pass, MODEL_CHEAP
def get_pii_detector(settings) -> PIIDetector
```

Presidio `AnalyzerEngine` + `AnonymizerEngine` plus extra regex recognisers for
IBAN, Aadhaar, PAN, SWIFT, API keys and JWTs. `mode` ∈ `mask` (`<EMAIL_ADDRESS>`),
`hash` (stable HMAC so redacted values still join), `partial` (last 4 kept).
Presidio must be an optional import: if unavailable, fall back to the regex-only
recogniser set and log a warning — the package must still import.

## `ragcore.dedupe`

```python
def content_sha256(text: str) -> str
def simhash64(text: str, *, shingle=4) -> int
def simhash_hex(text: str) -> str
def hamming64(a: int, b: int) -> int
def is_near_duplicate(a: str, b: str, *, max_distance=3) -> bool   # hex simhashes
def dedupe_chunks(chunks: list[T], *, key, simhash_key,
                  max_distance=3) -> tuple[list[T], list[tuple[T, str]]]
```

`dedupe_chunks` returns kept items and `(dropped, reason)` pairs — drops are always
returned for audit, never silently discarded.

## `ragcore.db`

SQLAlchemy 2.0 async (`asyncpg`) + Alembic. Tables: `tenants`, `users`,
`source_configs`, `documents`, `ingest_runs`, `ingest_items`, `chat_sessions`,
`chat_messages`, `tool_invocations`, `user_profiles`, `user_memories`,
`semantic_cache_meta`, `lineage_records`, `eval_runs`, `eval_results`, `feedback`,
`audit_log`. Every tenant-scoped table has a `tenant_id` column, an index on it, and
a composite index on `(tenant_id, <natural key>)`. `get_session()` async dependency;
one initial Alembic migration that creates everything.

---

## HTTP API surface (`services/api`, prefix `/api/v1`)

| Method | Path | Notes |
|---|---|---|
| POST | `/chat` | SSE stream (events above); `stream=false` returns one JSON body |
| POST | `/search` | retrieval only → `RetrievalResult`; no generation |
| GET | `/me` | echoes resolved `Principal` |
| GET | `/sessions` · GET `/sessions/{id}` · DELETE `/sessions/{id}` | |
| GET | `/sessions/{id}/messages` | includes `suppressed` flags |
| POST | `/sessions/{id}/compact` | force context compaction |
| GET/POST | `/documents` | list / upload (multipart) |
| DELETE | `/documents/{id}` · POST `/documents/{id}/reindex` | |
| GET | `/documents/{id}/lineage` | full provenance chain |
| GET | `/memory/profile` · GET `/memory/items` · DELETE `/memory/items/{id}` | |
| PUT | `/memory/consent` | toggle long-term memory |
| POST | `/feedback` | thumbs + comment → Langfuse score |
| GET/POST | `/eval/runs` · GET `/eval/runs/{id}` | |
| GET | `/admin/tenants` · `/admin/sources` · `/admin/schedule` | role `rag.admin` |
| POST | `/admin/ingest/trigger` · GET `/admin/ingest/runs` | |
| GET | `/health` · `/readyz` · `/metrics` | Prometheus text on `/metrics` |

Auth: `Authorization: Bearer <Entra JWT>` on everything except `/health`,
`/readyz`, `/metrics`. Validation checks signature against cached Entra JWKS
(`https://login.microsoftonline.com/{tenant}/discovery/v2.0/keys`), `iss`, `aud`,
`exp`/`nbf`, and extracts `oid`, `tid`, `roles`, `groups`. A dev-only mode
(`entra_dev_mode=true`) accepts an unsigned header for local runs and **must log a
loud warning and refuse to start when `env == "production"`**.

## RAG pipeline behaviour (`services/api/app/rag`)

Ordered stages in `orchestrator.py`. Each stage is a Langfuse span and may short-circuit.

1. **Input guard** — PII scan on the user turn (redact-before-log, never store raw),
   prompt-injection and jailbreak heuristics, size cap.
2. **Memory load** — user profile + top-`memory_top_k` long-term memories
   (dense search in `rag_memories`, salience-weighted), plus session short-term window.
3. **Query transform** (requirement #10) — a single `MODEL_FAST` structured call
   returns `TransformedQuery`: `intent`, `needs_retrieval`, `needs_tools`,
   `tool_hints`, `rewritten` (context-resolved standalone question),
   `sub_questions` (decomposition, ≤ `qt_max_subqueries`), `hyde_passage`
   (only when `qt_hyde_enabled` and the query is sparse/abstract),
   `metadata_filter` (extracted facets → `MetadataFilter`), `is_out_of_domain`.
   Pronoun/ellipsis resolution against the short-term window happens here.
4. **Semantic cache probe** — `rag_semantic_cache` lookup; on hit skip stage 5's
   candidate generation but still apply the live ACL filter to cached `chunk_ids`.
5. **Retrieve** — for each query: dense + sparse prefetch → server-side RRF →
   metadata + ACL filter (always `build_acl_filter`) → union across sub-questions →
   exact-hash and simhash dedupe → cross-encoder rerank → MMR diversity →
   top-`retrieval_top_n`. Record `total_candidates`/`after_dedupe`/`after_rerank` and
   every `dropped_reason`.
6. **OOD gate** — if retrieval max score < `guardrail_ood_min_score` or the
   transformer flagged out-of-domain, and no tool can serve the query, answer with an
   explicit "outside the indexed corpus" refusal plus what *is* covered. Never
   hallucinate a fallback answer.
7. **Contradiction check** — cluster retrieved chunks by claim; when chunks disagree,
   prefer the one with the newer `effective_from`/`source_modified_at` and higher
   authority `doc_type`, and surface the conflict in the answer with both citations
   rather than silently picking one. Emits a `GuardrailEvent(kind="contradiction")`.
8. **Tool loop** (requirement #4) — Claude tool-use loop over the role-filtered
   registry: retrieval tool, declarative REST tools, and remote MCP servers via the
   `mcp_servers` + `mcp_toolset` connector. Bounded by `tool_max_iterations`;
   every invocation persisted to `tool_invocations` and traced.
9. **Context assembly** (requirements #3/#5) — `context.py` packs to a token budget
   measured with `count_tokens`, in priority order: system prompt → pinned messages →
   rolling summary → long-term memory → retrieved chunks (dedup-aware, largest
   marginal-value first) → recent turns. On overflow it *suppresses* rather than
   truncates: oldest non-pinned turns are folded into the rolling summary
   (`MODEL_FAST`), tool results older than `context_tool_result_ttl_turns` are cleared
   via the `clear_tool_uses_20250919` context edit, and `ContextStats` reports exactly
   what happened. Compaction triggers at `context_compact_at_ratio` (default 0.75) of
   budget — periodic and proactive, not only on hard overflow.
10. **Generate** — `MODEL_MAIN`, streamed, cached system prefix, numbered-source
    prompt that requires `[n]` markers.
11. **Citation extraction** (requirement #9) — map every `[n]` to its chunk, verify the
    quoted span actually occurs in that chunk (normalised match), drop unverifiable
    citations and flag them. Groundedness gate: if `citation_validity` <
    `guardrail_min_groundedness`, append an uncertainty notice.
12. **Output guard** — PII egress scan, classification check (never emit content above
    the principal's clearance even if a filter bug let it through — defence in depth),
    then persist message, lineage, usage and Langfuse scores.
13. **Memory write-back** — extract durable preferences/facts (`MODEL_FAST`,
    structured), deduplicate against existing memories via `supersedes`, decay
    salience, and update the rolling profile. Skipped entirely when
    `memory_consent=False`.

## Requirement → implementation map

| # | Requirement | Primary location |
|---|---|---|
| 1 | Multi-tenant ACL ingestion, serverless, nightly delta | `services/ingestion/**`, `ragcore/vectorstore/filters.py` |
| 2 | User personalisation + long-term memory + fast similar-query retrieval | `app/rag/memory/long_term.py`, `semantic_cache.py` |
| 3 | Efficient in-session context management | `app/rag/context.py` |
| 4 | API / MCP tool calling for un-indexed data | `app/rag/tools/**` |
| 5 | Short-term memory + periodic context suppression | `app/rag/memory/short_term.py`, `context.py` |
| 6 | Hybrid semantic + BM25 + rerank + metadata filtering | `ragcore/vectorstore/hybrid.py`, `app/rag/retriever.py` |
| 7 | React interface | `web/**` |
| 8 | RAGAS + semantic-similarity validation vs golden set | `services/eval/**` |
| 9 | PII, OOD, contradictions, dedupe, citations, lineage, Langfuse | `app/rag/guardrails/**`, `citations.py`, `ragcore/pii`, `ragcore/dedupe`, `ragcore/observability` |
| 10 | Query transformation | `app/rag/query_transform.py` |

---

# Addendum A — contract-layer definitions (owner: ragcore/models, ragcore/db, root)

Everything below was referenced but not fully defined above. It is now implemented in
`packages/ragcore`. Treat these signatures the same as the ones earlier in this file.

## `ragcore.models.document`

```python
class SourceType(StrEnum):        # also re-exported from ragcore.models.chunk
    BLOB = "blob"; SHAREPOINT = "sharepoint"; HTTP = "http"
    SQL = "sql"; UPLOAD = "upload"; LOCAL = "local"

class BlockKind(StrEnum):
    HEADING; PARAGRAPH; LIST_ITEM; TABLE; CODE; CAPTION; QUOTE; FOOTNOTE; OTHER

class IngestAction(StrEnum):  CREATE; UPDATE; DELETE; SKIP; ACL_ONLY
class IngestStatus(StrEnum):  RUNNING; SUCCEEDED; FAILED; PARTIAL; SKIPPED
class IngestTrigger(StrEnum): TIMER; QUEUE; HTTP; MANUAL; REINDEX; UPLOAD

class SourceConfig(BaseModel):
    source_id: str; tenant_id: str; source_type: SourceType
    name: str; enabled: bool = True
    options: dict[str, Any] = {}          # connector-specific; see below
    default_classification: Classification = Classification.INTERNAL
    default_allowed_roles / _groups / _users / denied_users: list[str] = []
    inherit_source_permissions: bool = True   # map SharePoint item ACLs onto chunks
    doc_type: str = "document"; tags: list[str] = []; language: str = "en"
    cron_override: str | None; timezone_override: str | None
    cursor: str | None                    # deltaLink / watermark / blob marker
    cursor_updated_at: datetime | None
    last_run_id / last_run_at / last_status
    created_at: datetime; updated_at: datetime
    def default_access_control(self) -> AccessControl
    def option(self, key, default=None) -> Any
    def require_option(self, key) -> Any   # raises ValueError when absent

class SourceDocument(BaseModel):          # raw fetch result, pre-parse
    document_id: str; tenant_id: str; source_id: str
    source_type: SourceType; source_uri: str
    title: str = ""
    content_bytes: bytes | None; content_text: str | None
    media_type: str; filename: str | None; size_bytes: int
    etag: str | None; content_sha256: str
    source_modified_at / effective_from / effective_to: datetime | None
    access_control: AccessControl         # source defaults merged with item ACLs
    doc_type: str; tags: list[str]; author: str | None; language: str | None
    blob_url: str | None                  # archived raw copy
    metadata: dict[str, Any]; deleted: bool; fetched_at: datetime
    @property has_content -> bool
    def text_or_empty(self) -> str

class ParsedBlock(BaseModel):
    kind: BlockKind; text: str; order: int
    level: int | None                     # heading depth, HEADING only
    page: int | None; section_path: list[str]; metadata: dict[str, Any]
    @property token_estimate -> int       # pre-tokenisation, ~len/4

class ParsedDocument(BaseModel):
    document_id / tenant_id / source_id / source_type / source_uri
    title: str; blocks: list[ParsedBlock]         # auto-sorted by .order
    doc_type: str; language: str; author: str | None
    tags / keywords: list[str]; summary: str | None
    page_count: int | None; access_control: AccessControl
    content_sha256: str
    source_modified_at / effective_from / effective_to: datetime | None
    metadata: dict[str, Any]; parsed_at: datetime
    @property full_text -> str
    def blocks_for_page(self, page: int) -> list[ParsedBlock]

class IngestManifestEntry(BaseModel):
    document_id: str; source_uri: str; content_sha256: str
    etag: str | None; source_modified_at: datetime | None
    acl_fingerprint: str                  # ACL-only change -> ACL_ONLY reindex
    version: int; chunk_count: int; token_count: int
    last_run_id: str; last_seen_at: datetime; is_deleted: bool
    def decide(self, *, content_sha256, etag=None,
               acl_fingerprint="") -> IngestAction

class IngestManifest(BaseModel):          # persisted per (tenant_id, source_id)
    tenant_id: str; source_id: str
    entries: dict[str, IngestManifestEntry]        # keyed by document_id
    cursor: str | None; last_run_id: str | None
    last_full_scan_at: datetime | None; updated_at: datetime; version: int
    def get(self, document_id) -> IngestManifestEntry | None
    def upsert(self, entry) -> Self
    def missing_document_ids(self, seen: set[str]) -> list[str]   # deletions
    def mark_deleted(self, document_id, run_id) -> None
    @property live_count -> int

class IngestRunSummary(BaseModel):        # mirrored into the ingest_runs table
    run_id: str; tenant_id: str; source_id: str | None
    trigger: IngestTrigger; status: IngestStatus
    started_at: datetime; finished_at: datetime | None
    documents_seen / _created / _updated / _deleted / _skipped / _failed: int
    chunks_upserted / chunks_deleted / tokens_embedded: int
    duplicates_dropped / pii_documents: int
    forced: bool; within_working_hours: bool
    skip_reason: str | None                # "disabled" | "working_hours"
    error_message: str | None; errors: list[str]; metrics: dict[str, float]
    @property duration_seconds -> float | None
    def finish(self, status=None) -> Self  # derives PARTIAL/FAILED/SUCCEEDED
```

`SourceConfig.options` keys by `source_type`:

| `source_type` | required | optional |
|---|---|---|
| `blob` | `container` | `prefix`, `account_url` |
| `local` | `root` | `include_globs`, `exclude_globs` |
| `sharepoint` | `site_id`, `drive_id` | `folder_path` |
| `http` | `start_urls` **or** `sitemap_url` | `allow_domains`, `max_depth` |
| `sql` | `dsn_secret_ref`, `query`, `watermark_column`, `id_column` | `text_columns`, `title_column` |

## `ragcore.models.tool`

```python
class ToolKind(StrEnum):  RETRIEVAL = "retrieval"; REST = "rest"; MCP = "mcp"
class ToolAuth(StrEnum):
    NONE; BEARER; API_KEY; BASIC; ENTRA_OBO; MANAGED_IDENTITY

class RestToolSpec(BaseModel):
    method: str = "GET"                   # validated + upper-cased
    url_template: str                     # "https://api/orders/{order_id}"
    headers: dict[str, str] = {}
    query_template: dict[str, str] = {}
    body_template: dict[str, Any] | None
    auth: ToolAuth = ToolAuth.NONE
    auth_secret_ref: str | None           # Key Vault secret NAME, never a value
    auth_header_name: str = "Authorization"; auth_scope: str | None
    timeout_seconds: float = 20.0
    response_json_path: str | None        # dotted path, e.g. "data.items"
    success_status_codes: list[int] = [200, 201, 202, 204]
    verify_tls: bool = True
    @property placeholders -> set[str]    # every {name} in url/query/body

class McpServerSpec(BaseModel):
    name: str; url: str                   # https:// enforced
    authorization_token_ref: str | None   # Key Vault secret NAME
    allowed_tools: list[str] = []         # empty exposes everything
    beta_flag: str = "mcp-client-2025-11-20"
    def to_connector_entries(self, authorization_token=None
        ) -> tuple[dict, dict]            # (mcp_servers entry, mcp_toolset entry)

class ToolSpec(BaseModel):
    name: str; kind: ToolKind; description: str
    input_schema: dict[str, Any]
    tenant_id: str | None                 # None = all tenants
    allowed_roles: list[str] = []          # empty = all roles
    enabled: bool = True
    rest: RestToolSpec | None; mcp: McpServerSpec | None
    max_result_chars: int = 8000
    cacheable: bool = False; cache_ttl_seconds: int = 60
    def is_allowed_for(self, principal: Principal) -> bool
    def to_anthropic_tool(self) -> dict   # raises for kind=mcp

class ToolResult(BaseModel):
    tool_call_id: str; tool_name: str; kind: ToolKind
    content: str                          # what the model sees
    structured: dict[str, Any] | None     # parsed payload, for lineage
    is_error: bool = False; error_message: str | None
    http_status: int | None; latency_ms: float
    truncated: bool = False; created_at: datetime
    @classmethod failure(...) -> Self
    def truncate(self, limit: int) -> Self
    def to_result_summary(self, limit=500) -> str
    def to_tool_result_block(self) -> dict
```

Validation the tool layer relies on: a `kind`/transport mismatch raises, and every
`{placeholder}` in a `RestToolSpec` template must be declared in the owning
`ToolSpec.input_schema.properties` — so the model cannot inject an undeclared URL
segment. `ToolResult` is always returned with `is_error=True` rather than raising out
of the loop, so a failed tool lets the model recover instead of killing the turn.

## `ragcore.models.chunk` — additions

```python
class ChunkPayload(BaseModel):
    # ... fields exactly as specified earlier ...
    @classmethod from_access_control(cls, ac: AccessControl, **fields) -> Self
    def access_control(self) -> AccessControl
    def with_access_control(self, ac) -> Self        # ACL-only reindex path
    def to_qdrant_payload(self) -> dict              # datetimes -> RFC 3339
    @classmethod from_qdrant_payload(cls, payload) -> Self   # raises on empty
    @property embed_text -> str                      # header + "\n\n" + text
    @property section_label -> str
    @property recency_at -> datetime                 # effective_from > modified > created
    @property classification_level -> Classification
    def is_effective_at(self, moment=None) -> bool
```

**`from_access_control` is the only supported way to populate the flat ACL fields.**
ACL keys passed in `**fields` are overwritten by `ac`, so a caller cannot
desynchronise the flat fields from their nested source. `classification_rank` is
re-derived from `classification` by a model validator and cannot disagree with it.

## `ragcore.models.acl` — additions

`Classification` overrides `__lt__/__le__/__gt__/__ge__` to compare by `.rank`, so
`PUBLIC < RESTRICTED`, `sorted()` yields least-to-most sensitive, and
`max(a, b)` picks the stricter label. `str` behaviour is preserved
(`Classification.PUBLIC == "public"`). Also `Classification.from_rank(int)` (clamped)
and `Classification.max_rank()`.

`ADMIN_ROLE = "rag.admin"` is a module constant; `Principal.is_admin(admin_role=ADMIN_ROLE)`
takes an optional override so the contract signature `is_admin(self) -> bool` still holds.
`AccessControl.permits(principal)` is the in-process mirror of `build_acl_filter` for
defence-in-depth (stage 12) — it is **not** a substitute for filtering in Qdrant.
`AccessControl.to_flat()` / `.from_flat()` / `.merged_with()` round-trip the flat form.

## `ragcore.models.retrieval` / `chat` — additions

* `RetrievalStage(StrEnum)`: `DENSE|SPARSE|FUSION|RERANK|CACHE|TOOL`.
* `MetadataFilter.merged_with(other)` intersects two filters and never widens either;
  `.fingerprint_payload()` is the order-stable form `filter_fingerprint()` hashes.
* `RetrievalResult.dropped: list[RetrievedChunk]` carries audited drops with reasons;
  `.without_text()` is the serialisation for the SSE `retrieval` event (strips `text`,
  `contextual_header` and `summary`); `.max_score` feeds the OOD gate.
* `RetrievedChunk.drop(reason)` and `.to_citation(marker)`.
* `SSEEvent(StrEnum)` enumerates the SSE event names; `GuardrailStage`,
  `GuardrailKind` and `GuardrailAction` enumerate the `GuardrailEvent` string fields.
* `ContextStats.tool_results_cleared` counts results removed by the
  `clear_tool_uses_20250919` context edit; `.utilisation` and `.cache_hit_ratio` derive.

## `ragcore.models.memory` — additions

`normalize_query(str) -> str` canonicalises a query for cache keying.
`SemanticCacheEntry.make_cache_id(tenant_id=, normalized_query=, fingerprint=)` derives
a deterministic id **keyed on the tenant**, so an identical query in two tenants can
never share an entry. `.matches(fingerprint=, similarity=, threshold=)` implements the
reuse rule (exact fingerprint match AND similarity ≥ threshold AND not expired).
`LongTermMemory.decayed_salience(decay_per_day=, now=)` and `.touch(boost=)`.

## `ragcore.db` — table/model naming

Tables are exactly the 17 named earlier. ORM class names add a `Row` suffix where a
pydantic model already owns the plain name, so both can be imported together:

| table | ORM class | table | ORM class |
|---|---|---|---|
| `tenants` | `Tenant` | `user_profiles` | `UserProfileRow` |
| `users` | `User` | `user_memories` | `UserMemory` |
| `source_configs` | `SourceConfigRow` | `semantic_cache_meta` | `SemanticCacheMeta` |
| `documents` | `Document` | `lineage_records` | `LineageRecordRow` |
| `ingest_runs` | `IngestRun` | `eval_runs` | `EvalRunRow` |
| `ingest_items` | `IngestItem` | `eval_results` | `EvalResultRow` |
| `chat_sessions` | `ChatSession` | `feedback` | `Feedback` |
| `chat_messages` | `ChatMessage` | `audit_log` | `AuditLog` |
| `tool_invocations` | `ToolInvocation` | | |

```python
# ragcore.db.base
JSON_TYPE                 # JSONB on PostgreSQL, JSON on sqlite — use for every JSON column
class Base(DeclarativeBase)
metadata                  # = Base.metadata, Alembic target
def get_engine(settings=None) -> AsyncEngine
def get_sessionmaker(settings=None) -> async_sessionmaker[AsyncSession]
async def get_session() -> AsyncIterator[AsyncSession]      # FastAPI dependency
def session_scope(settings=None) -> AbstractAsyncContextManager[AsyncSession]
async def dispose_engine(settings=None) -> None              # shutdown hook
async def check_database(settings=None) -> bool              # /readyz probe
```

Sessions are handed out with **no transaction begun**; the caller commits. Repositories
`flush()` but never `commit()`.

```python
# ragcore.db.repositories — every function takes tenant_id and filters on it
def new_id() -> str                       # 32-char hex, for any *_id column
async def upsert_tenant(session, *, tenant_id, name, ...) -> Tenant
async def upsert_user(session, principal: Principal) -> User
async def upsert_document(session, *, tenant_id, document_id, source_uri,
        source_type, access_control: AccessControl, ...) -> Document
async def mark_documents_deleted(session, *, tenant_id, document_ids) -> int
async def list_source_configs(session, *, tenant_id, enabled_only=True) -> list[...]
async def record_ingest_run(session, summary: IngestRunSummary) -> IngestRun
async def record_ingest_item(session, *, tenant_id, run_id, source_uri,
        action, status, ...) -> IngestItem
async def get_or_create_session(session, *, tenant_id, user_id,
        session_id=None, title="") -> ChatSession
async def get_session_row(session, *, tenant_id, user_id, session_id) -> ChatSession | None
async def list_sessions(session, *, tenant_id, user_id, ...) -> list[ChatSession]
async def delete_session(session, *, tenant_id, user_id, session_id) -> bool
async def append_message(session, *, tenant_id, session_id, user_id, role: Role,
        content, pii_redacted: bool, ...) -> ChatMessage
async def list_session_messages(session, *, tenant_id, session_id, limit=None,
        include_suppressed=True, ascending=True) -> list[Message]
async def count_session_messages(session, *, tenant_id, session_id) -> int
async def suppress_messages(session, *, tenant_id, session_id, message_ids,
        rolling_summary=None, summary_tokens=None) -> int
async def write_tool_invocation(session, *, tenant_id, tool_call_id, tool_name,
        kind, arguments, ...) -> ToolInvocation
async def get_or_create_profile(session, *, tenant_id, user_id) -> UserProfile
async def save_profile(session, profile: UserProfile) -> UserProfileRow
async def set_memory_consent(session, *, tenant_id, user_id, consent) -> UserProfileRow
async def upsert_memory(session, memory: LongTermMemory, *, tenant_id) -> UserMemory
async def list_memories(session, *, tenant_id, user_id, ...) -> list[UserMemory]
async def delete_memory(session, *, tenant_id, user_id, memory_id) -> bool
async def upsert_cache_meta(session, *, tenant_id, cache_id, normalized_query,
        filter_fingerprint, chunk_ids, ...) -> SemanticCacheMeta
async def record_lineage(session, *, tenant_id, kind, subject_id, operation,
        actor, parents=None, ...) -> LineageRecordRow
async def write_eval_run(session, *, tenant_id, run_id, started_at, ...) -> EvalRunRow
async def write_eval_result(session, *, tenant_id, run_id, item_id, answer,
        scores, passed, ...) -> EvalResultRow
async def write_feedback(session, *, tenant_id, user_id, rating,
        pii_redacted: bool, ...) -> Feedback
async def write_audit(session, *, tenant_id, action, outcome="allow", ...) -> AuditLog
```

Three behaviours callers must know:

1. **`append_message` and `write_feedback` require `pii_redacted=True`** before they
   will persist free text (`ValueError` otherwise). Run redaction first, then pass the
   flag — it is an assertion that the pass ran, not a formatting hint.
2. **Fetch-then-check raises `TenantMismatchError`** rather than returning `None` when
   a row exists under a different tenant, so a cross-tenant probe is auditable.
   `upsert_document` additionally rejects an `access_control` whose `tenant_id`
   disagrees with the `tenant_id` argument.
3. **`upsert_document` bumps `version`** when `content_sha256` changes, clears
   `is_deleted` when a document reappears at source, and `upsert_memory` applies
   `supersedes` by soft-deleting the superseded row and setting `superseded_by`.
   `set_memory_consent(consent=False)` soft-deletes the user's memories.

## `ragcore.settings` — additions beyond the field groups listed earlier

Extra groups: `qt_*` (query transform), `pii_*`, `dedupe_*`, `chunk_*`, `tool_*`,
`api_*`. Extra derived members:

```python
@property is_production / is_local -> bool
@property database_url / alembic_database_url -> str
@property entra_issuer / entra_jwks_url / expected_audience -> str | None
@property context_prompt_budget_tokens -> int      # budget - reserve_output
@property context_compact_threshold_tokens -> int  # ratio of the above
@property langfuse_ready -> bool                   # enabled AND both keys present
def price_for_model(self, model: str) -> tuple[float, float]   # USD per MTok
def is_within_working_hours(self, now: datetime | None = None) -> bool
def may_start_scheduled_ingest(self, now=None) -> tuple[bool, str]
```

`is_within_working_hours` treats a naive `now` as local to `ingest_timezone`, returns
False on a non-working weekday, and handles a window that wraps midnight
(`start=22, end=6`). `may_start_scheduled_ingest` is the guard ingestion calls; its
reason is one of `"ok" | "disabled" | "forced" | "working_hours"` and belongs in
`IngestRunSummary.skip_reason`.

Startup validation refuses to construct `Settings` when `entra_dev_mode` or
`tool_allow_insecure_http` is true while `env == "production"`, and when a budget or
chunk-size relationship is unsatisfiable. `Settings` is unhashable (pydantic model),
so it must never be an `lru_cache` key — cache on `database_url` instead.

## Workspace layout (root `pyproject.toml`)

The root project is **virtual** (`[tool.uv] package = false`) and owns no importable
code. Members: `packages/ragcore`, `services/api`, `services/ingestion`,
`services/eval`. Install everything with **`uv sync --all-packages --all-groups`**
(what `make setup` runs) rather than listing members as root dependencies, so each
member owns its own distribution name. `ragcore` is declared in
`[tool.uv.sources]` as `{ workspace = true }`; a service depends on it by adding
`ragcore` to its own `dependencies`.

Shared dev deps live in the root `[dependency-groups] dev`. ruff (line-length 88,
Google docstrings) and pytest (`asyncio_mode = "auto"`) are configured once at the
root and apply to every member. Presidio is an optional extra on ragcore:
`uv sync --extra pii`.

`docker-compose.yml` expects two Dockerfiles that other phases own:
`services/api/Dockerfile` (build context = repo root, so it can copy
`packages/ragcore`) and `web/Dockerfile` (build context = `web/`, serving the built
assets on port 80).

---

# Addendum B — retrieval substrate (owner: ragcore/vectorstore, ragcore/embeddings, ragcore/rerank, ragcore/dedupe)

Everything the earlier sections specify for these modules is implemented exactly as
written. This addendum records the decisions those sections left open, plus the
helpers added so that no downstream module has to hand-roll a filter, a point id or a
Qdrant request. Treat these signatures the same as the ones earlier in this file.

## Collection layout decisions

* **`get_client` lives in `vectorstore/client.py`** and is re-exported from
  `vectorstore/collections.py` and from `ragcore.vectorstore`, so all three import
  paths work. It caches one `AsyncQdrantClient` per endpoint (`Settings` is
  unhashable, so the key is a tuple of the `qdrant_*` fields with the API key reduced
  to a digest). Also `close_client(settings=None)`, `close_all_clients()` (shutdown
  hook) and `check_qdrant(client=None, settings=None) -> bool` (the `/readyz` probe;
  never raises).
* **All three collections carry a single _named_ dense vector called `dense`** — never
  an unnamed default vector. `rag_memories` and `rag_semantic_cache` are "dense only"
  in the sense of having no sparse vector, not in the sense of being unnamed. Every
  query therefore passes `using=DENSE`, and every upsert uses
  `vector={DENSE: [...]}`.
* **Point ids are UUIDv5-derived.** Qdrant accepts only unsigned integers and UUIDs,
  but `chunk_id` / `memory_id` / `cache_id` are opaque strings.

  ```python
  # collections.py
  POINT_ID_NAMESPACE: uuid.UUID
  def stable_point_id(value: str, *, namespace=POINT_ID_NAMESPACE) -> str
  def point_id_for_chunk(chunk_id: str) -> str
  def point_id_for_memory(memory_id: str) -> str
  def point_id_for_cache(cache_id: str) -> str
  ```

  A value that already parses as a UUID is returned canonicalised; anything else is
  hashed. The mapping is deterministic, which is what makes re-ingesting a chunk an
  upsert rather than a duplicate. **Nothing reads the point id back** — the logical id
  is always in the payload.
* `CollectionSpec` / `collection_specs(settings)` expose the declarative layout, and
  `CHUNK_PAYLOAD_INDEXES` / `MEMORY_PAYLOAD_INDEXES` / `CACHE_PAYLOAD_INDEXES` are the
  `(field_name, field_schema)` tuples `ensure_collections` applies. `ensure_collections`
  tolerates a concurrent creator (HTTP 409), skips indexes that already exist, logs
  `created` vs `found` per collection, and warns when an existing `tenant_id` index
  lacks `is_tenant=True`.
* `classification_rank` is indexed with `IntegerIndexParams(range=True)` because the
  ACL filter range-filters it. `is_deleted` is a `BoolIndexParams`,
  `source_modified_at` a `DatetimeIndexParams`, everything else keyword.

## `ragcore.vectorstore.filters` — additions

```python
MIN_SHOULD = 1                 # the permissive branch's threshold
FINGERPRINT_VERSION = "v1"     # bumped when fingerprint inputs change

def build_acl_filter_for_chunk_ids(principal, chunk_ids, extra=None, *,
                                   include_deleted=False) -> qm.Filter
def build_tenant_filter(tenant_id, *, document_ids=None, exclude_document_ids=None,
                        source_id=None, chunk_ids=None,
                        include_deleted=True, only_deleted=False) -> qm.Filter
def build_memory_filter(principal, kinds=None, *,
                        include_expired=False, now=None) -> qm.Filter
def classification_ceiling(principal, extra=None) -> Classification
def serialise_filter(qfilter) -> dict     # for RetrievalResult.filter_applied
```

* `build_acl_filter` emits **both** `should=[...]` and
  `min_should=MinShould(conditions=[...], min_count=1)` over the same permissive
  branch. Qdrant ANDs the two gates, so the pair is semantically identical to `should`
  alone; stating `min_count` explicitly is what the unit test pins.
* The permissive branch omits the `allowed_groups` / `allowed_roles` arms when the
  principal has no groups / roles, rather than sending an empty `MatchAny`.
* `build_acl_filter_for_chunk_ids` is **the** semantic-cache re-fetch path (stage 4).
  It composes the full ACL filter and adds a `HasIdCondition` over the derived point
  ids. An empty `chunk_ids` matches nothing, which is the safe reading. `chunk_id`
  itself is deliberately **not** payload-indexed — `HasIdCondition` is indexed by
  construction.
* `build_tenant_filter` is for ingestion and admin **writes**. It is *not* an ACL
  filter (no deny list, no permissive branch, no clearance) and must never serve a
  read to a principal. It lives in `filters.py` so that no module outside it ever
  constructs a `qm.Filter`.
* `MetadataFilter.section_prefix` becomes `MatchValue` on the `section_path` keyword
  array: the heading must appear somewhere in the root-to-leaf breadcrumb, i.e. the
  chunk is at or beneath that heading. That is the section-subtree reading, and the
  only one a keyword index answers without a payload scan.
* `filter_fingerprint` hashes canonical JSON of
  `{version, tenant_id, clearance_rank, filter: extra.fingerprint_payload()}` →
  32 hex chars. Group and role membership deliberately do **not** contribute, because
  cached chunk ids are always re-filtered through the live ACL filter on reuse.

## `ragcore.vectorstore.hybrid` — additions

```python
FUSIONS: dict[str, qm.Fusion]
def resolve_fusion(fusion: str) -> qm.Fusion          # raises ValueError

async def dense_search(client, *, collection: str, dense: Sequence[float],
                       qfilter: qm.Filter, limit: int,
                       score_threshold: float | None = None,
                       using: str = DENSE) -> list[qm.ScoredPoint]
```

* `hybrid_search` takes `sparse: SparseVec | None`. A None or empty sparse vector
  (a stop-word-only query) **drops the sparse prefetch branch** rather than sending an
  empty `SparseVector`, which would match nothing.
* The `qfilter` is set on both prefetch branches **and** on the outer `query_filter`.
  Redundant by design: the tenant boundary should survive an edit to either layer.
* `query_text` is carried for tracing only. It is raw user content, so only its
  **length** is ever logged.
* `dense_search` is what `rag_memories` and `rag_semantic_cache` use. Pass
  `score_threshold=settings.memory_cache_threshold` on the cache probe so the engine
  enforces the similarity floor.

## `ragcore.vectorstore.writer`

```python
class ChunkPoint(BaseModel):
    payload: ChunkPayload
    dense: list[float]                    # rejected when empty
    sparse: SparseVec | None = None
    @property point_id -> str
    def to_point_struct(self) -> qm.PointStruct

async def upsert_points(client, *, collection, points: Sequence[qm.PointStruct],
        batch_size=None, wait=True, settings=None) -> int
async def upsert_chunks(client, *, collection, chunks: Sequence[ChunkPoint],
        batch_size=None, wait=True, settings=None) -> int
async def count_chunks(client, *, collection, qfilter, exact=True) -> int
async def soft_delete_document(client, *, collection, tenant_id, document_id,
        run_id=None, wait=True) -> int
async def soft_delete_documents(client, *, collection, tenant_id, document_ids,
        run_id=None, wait=True) -> int
async def hard_delete_document(client, *, collection, tenant_id, document_id,
        wait=True) -> int
async def hard_delete_by_filter(client, *, collection, qfilter, wait=True) -> None
async def update_access_control(client, *, collection, tenant_id, document_id,
        access_control: AccessControl, run_id=None, wait=True) -> int
async def tombstone_missing(client, *, collection, tenant_id,
        manifest_document_ids: Sequence[str], source_id=None, run_id=None,
        page_size=None, settings=None) -> list[str]
```

Behaviours callers must know:

1. **`upsert_chunks` raises `TenantMismatchError` when a batch spans tenants.** A batch
   is the retry unit, and a partially-correct retry is how cross-tenant rows appear.
2. **Soft delete is the default.** It sets `is_deleted=True`, `updated_at` and
   (when given) `ingest_run_id` via a filtered `set_payload`, and returns the number of
   chunks affected — 0 when they were already tombstoned. `hard_delete_by_filter`
   refuses a filter with no `must` clause, so an unscoped purge is impossible.
3. **`update_access_control` is the `ACL_ONLY` reindex path** — writes
   `AccessControl.to_flat()` plus `updated_at`, no re-embedding. Raises
   `TenantMismatchError` when `access_control.tenant_id` disagrees.
4. **`tombstone_missing` is scroll-and-diff, not `MatchExcept`.** Pass the manifest's
   live document ids; it scrolls the tenant/source's indexed `document_id` values,
   tombstones the difference, and returns the tombstoned ids sorted (feed them into
   `IngestRunSummary.documents_deleted`). **Always pass `source_id`** for a per-source
   run, or documents owned by the tenant's other sources get tombstoned. An empty
   `manifest_document_ids` tombstones everything for that tenant/source and logs a
   warning, because that is also what a broken connector looks like.

## `ragcore.embeddings` — additions

```python
SparseVec.empty() -> Self ; .is_empty -> bool ; .nnz -> int
def truncate_for_embedding(text: str, max_chars: int) -> str
def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float
def reset_embedding_providers() -> None        # test helper

class FastEmbedProvider:                       # implements EmbeddingProvider
    async def embed_sparse(self, texts) -> list[SparseVec]
    async def warm_up(self) -> None            # load models at startup, not on turn 1
```

* `SparseVec` validates that `indices` and `values` are the same length.
* Documents go through FastEmbed `embed`, queries through `query_embed`. For
  `Qdrant/bm25` these genuinely differ, and the IDF factor is applied by Qdrant via
  `Modifier.IDF` — embedding a query with `embed` double-weights terms.
* `cosine_similarity` exists for the off-Qdrant comparisons: memory write-back's
  `memory_dedupe_threshold` and the eval harness's `semantic_similarity`.
* `get_embedding_provider` caches on a tuple of `embedding_*` fields. Import of
  `ragcore.embeddings` never pulls fastembed/onnxruntime; a missing fastembed raises
  `ConfigError` on first embed, not at import.

## `ragcore.rerank` — additions

`NoopReranker` lives in `rerank/cross_encoder.py` alongside `CrossEncoderReranker` and
is re-exported from `ragcore.rerank`. Both expose `async warm_up()`.

**Neither reranker drops candidates.** `rerank_min_score` and
`rerank_candidate_limit` are enforced by `app/rag/retriever.py`, which records a
`dropped_reason` for each excluded chunk — filtering inside the reranker would make
those drops invisible, and requirement #9 asks for every drop to be auditable. So the
retriever must:

* truncate candidates to `rerank_candidate_limit` before calling, dropping with
  reason `"rerank:candidate_limit"`;
* apply `rerank_min_score` to the returned results, dropping with reason
  `"rerank:min_score"`.

`RerankResult.score` is model-dependent and comparable only to other scores from the
same call — never to a cosine or fusion score. `NoopReranker` returns descending
scores in `(0, 1]` so downstream ordering and `final_score` arithmetic are unchanged.
Also `reset_rerankers()` for tests.

## `ragcore.dedupe` — additions

```python
SIMHASH_BITS = 64
def normalise_text(text: str) -> str          # NFKC + casefold + non-word runs -> " "
def shingles(text: str, *, width: int = 4) -> list[str]
def simhash64(text: str, *, shingle: int = 4) -> int
def simhash_hex(text: str, *, shingle: int = 4) -> str
```

* Both hash layers operate on `normalise_text`, so a re-export with different wrapping
  or Unicode composition does not look like new content.
* Shingles are hashed with BLAKE2b, never `hash()`, whose per-process salt would
  invalidate a stored fingerprint across restarts.
* Empty or word-free text has simhash 0 and `simhash_hex` `"0000000000000000"`.
  A **blank** stored simhash means "no fingerprint" and never compares equal —
  `is_near_duplicate` returns False when either side is blank or unparseable, and
  `dedupe_chunks` skips the near-duplicate layer for such a candidate. Ingestion opts
  chunks below `dedupe_min_chunk_chars` out this way.
* `dedupe_chunks` drop reasons are exactly `"duplicate:sha256"` and
  `"duplicate:simhash:<distance>"`. First occurrence wins, so callers pass candidates
  best-first (fusion score descending) and the survivor is the higher-ranked copy.
  `kept + dropped` always reconstructs the input.


---

# Addendum W — web client contract (owner: `web/`)

The React app (requirement #7) is a pure consumer of the HTTP surface above. This
section pins the few shapes the earlier sections name but do not spell out, plus the
one endpoint the web app needs that the table lacked. **Nothing here changes an
existing signature.**

## Browser environment variables

Only `VITE_*` variables reach the bundle; there is no client secret (PKCE public
client). Names match the `# web` block of the root `.env.example`.

| Variable | Default | Meaning |
|---|---|---|
| `VITE_API_BASE_URL` | `""` | API origin; empty means same-origin (dev proxy / co-hosted nginx). |
| `VITE_API_PROXY_TARGET` | `http://localhost:8000` | Dev-server proxy target for `/api`. |
| `VITE_API_PREFIX` | `/api/v1` | Must equal `api_prefix`. |
| `VITE_API_TIMEOUT_MS` | `30000` | Non-streaming request timeout. |
| `VITE_STREAM_IDLE_TIMEOUT_MS` | `120000` | SSE silence tolerated before the stream fails. |
| `VITE_STREAM_MAX_RETRIES` / `VITE_STREAM_RETRY_BASE_MS` | `2` / `500` | Stream retry budget and backoff base. |
| `VITE_PAGE_SIZE` | `50` | `limit` sent to list endpoints. |
| `VITE_CONTEXT_WARN_RATIO` | `0.75` | Mirrors `context_compact_at_ratio` for the context meter's warning. |
| `VITE_ENTRA_CLIENT_ID` / `_TENANT_ID` / `_AUTHORITY_HOST` | — | MSAL app registration. |
| `VITE_ENTRA_API_SCOPE` | — | Scope requested for the API access token. |
| `VITE_ENTRA_REDIRECT_URI` / `_POST_LOGOUT_REDIRECT_URI` | app origin | Registered SPA redirect URIs. |
| `VITE_ENTRA_CACHE_LOCATION` | `sessionStorage` | `sessionStorage` or `localStorage`. |
| `VITE_DEV_MODE` | `false` | Skips MSAL and sends the unsigned dev principal instead. |
| `VITE_DEV_PRINCIPAL_HEADER` | `x-dev-principal` | Must equal `entra_dev_principal_header`. |
| `VITE_DEV_PRINCIPAL` | — | JSON `Principal` sent in that header when dev mode is on. |

The client attaches `Authorization: Bearer <token>` from MSAL silent acquisition and
**retries a request exactly once on `401` after a forced refresh**. `EventSource` is
never used — it cannot send an `Authorization` header — so the chat stream is read with
`fetch` + `ReadableStream`.

## SSE `data:` payloads

Event names are exactly `SSEEvent`. Each `data:` line is a JSON object unless noted;
the client ignores unknown event names and unknown object keys.

| event | payload |
|---|---|
| `session` | `{"session_id": str, "title"?: str}` |
| `retrieval` | `RetrievalResult.without_text()` |
| `thinking` | `{"text": str}` (`delta` also accepted) |
| `tool_call` | `{"tool_call_id": str, "tool_name": str, "kind": "rest"\|"mcp"\|"retrieval", "arguments": dict}` |
| `tool_result` | `{"tool_call_id": str, "is_error": bool, "latency_ms": float, "result_summary": str\|null, "error_message"?: str\|null, "http_status"?: int\|null, "truncated"?: bool}` |
| `token` | `{"text": str}` |
| `citations` | `list[Citation]` (an `{"citations": [...]}` envelope is also accepted) |
| `context_stats` | `ContextStats` |
| `guardrail` | `GuardrailEvent` |
| `usage` | `{"model": str, "input_tokens": int, "output_tokens": int, "cache_read_tokens": int, "cache_write_tokens": int, "cost_usd": float}` |
| `done` | `{"session_id": str, "message_id": str, "stop_reason"?: str, "refused"?: bool, "trace_id"?: str\|null}` |
| `error` | `{"detail": str}` (`message` / `error` also accepted) |

`done` must carry `message_id` so the UI can attach feedback to the persisted turn;
without it the client falls back to a local id and hides the thumbs.

Reconnect policy: the client retries the **whole** request only while no `token` event
has arrived. After the first token it fails the turn and offers an explicit resend
rather than risking duplicated text — there is no resume-from-offset contract.

## Request / response shapes the table left open

```jsonc
// POST /search
{"query": str, "filters": MetadataFilter | null, "top_n": int | null}   // -> RetrievalResult

// POST /chat with stream=false  -> one JSON body
{"session_id": str, "message": Message, "retrieval": RetrievalResult | null,
 "context_stats": ContextStats | null, "guardrails": [GuardrailEvent],
 "usage": {...} | null, "trace_id": str | null}

// PUT /memory/consent
{"memory_consent": bool}                                               // -> UserProfile

// POST /feedback   (rating is the integer column `feedback.rating`)
{"session_id": str | null, "message_id": str | null, "rating": 1 | -1,
 "comment": str | null, "tags": [str]}

// POST /documents  (multipart/form-data)
// file=<binary>, title?, doc_type?, language?, classification?,
// tags (repeated), allowed_roles (repeated), allowed_groups (repeated)
// -> the created document row

// POST /documents/{id}/reindex  -> IngestRunSummary
// POST /admin/ingest/trigger
{"source_id": str | null, "force": bool, "full_scan": bool}             // -> IngestRunSummary

// GET /admin/schedule
{"ingest_cron": str, "ingest_timezone": str, "ingest_enabled": bool,
 "ingest_working_hours_start": int, "ingest_working_hours_end": int,
 "within_working_hours": bool, "may_start": bool, "reason": str,
 "next_run_at": str | null}          // reason is may_start_scheduled_ingest()'s reason

// POST /sessions/{id}/compact
{"session_id": str, "messages_suppressed": int, "compaction_events": int,
 "summary_tokens": int, "context_stats": ContextStats | null}

// POST /eval/runs
{"golden_set_path": str | null, "sample_size": int | null, "notes": str | null}  // -> EvalRun
```

List endpoints (`/sessions`, `/sessions/{id}/messages`, `/documents`, `/memory/items`,
`/eval/runs`, `/admin/ingest/runs`, `/admin/sources`) may return either a bare JSON
array or an envelope keyed `items` / `results` / `data`; the client accepts both and
sends `limit` (and `source_id` / `q` / `include_suppressed` where meaningful).

`GET /eval/runs/{id}` must include `results`, and each result should carry the
`eval_results.category` column so the dashboard can show pass/fail per golden-item
category — an `acl_negative` regression has to be distinguishable from a quality dip.

## New endpoint owned by this addendum

| Method | Path | Notes |
|---|---|---|
| PUT | `/memory/profile` | Body `{"summary"?: str, "preferred_style"?: str\|null, "preferred_language"?: str\|null, "top_topics"?: [str]}` → `UserProfile`. User-editable subset of the profile; the rolling `summary` stays model-maintained unless explicitly sent. Tenant/user come from the principal, never the body. |

The web client degrades gracefully when this route is absent (404/405/501): it shows a
notice that preferences can only be learned from conversation, so an API build without
it still works.


---

# Addendum L — model, observability and PII layer (owner: `ragcore/llm`, `ragcore/observability`, `ragcore/pii`)

Everything below is implemented. It refines the earlier `ragcore.llm`,
`ragcore.observability` and `ragcore.pii` sections rather than replacing them: the
signatures there still hold, and the notes here fix the details a caller has to know.

## Dependency additions

`packages/ragcore/pyproject.toml` needs **`prometheus-client>=0.21`** in
`dependencies`. `ragcore.observability.metrics` imports it defensively and no-ops when
it is absent, so a build without it still runs — but `/metrics` then serves a single
comment line instead of samples.

## `ragcore.llm.pricing`

```python
MODEL_MAIN = "claude-opus-5"; MODEL_FAST = "claude-sonnet-5"
MODEL_CHEAP = "claude-haiku-4-5"          # exact ids from LLM_FACTS
DEFAULT_CACHE_READ_MULTIPLIER = 0.1; DEFAULT_CACHE_WRITE_MULTIPLIER = 1.25

@dataclass(frozen=True)
class ModelPricing:
    input_per_mtok: float; output_per_mtok: float
    cache_read_multiplier: float = 0.1; cache_write_multiplier: float = 1.25
    @property cache_read_per_mtok / cache_write_per_mtok -> float
    def cost_usd(self, *, input_tokens=0, output_tokens=0,
                 cache_read_tokens=0, cache_write_tokens=0) -> float

MODEL_PRICING: dict[str, ModelPricing]     # 5/25, 3/15, 1/5 USD per MTok
def pricing_for(model, settings=None) -> ModelPricing
def estimate_cost_usd(*, model, input_tokens=0, output_tokens=0,
                      cache_read_tokens=0, cache_write_tokens=0,
                      settings=None) -> float
```

`settings.anthropic_price_per_mtok` always wins over the module table, so rates are an
operator concern. `usage.input_tokens` excludes cached tokens, so the three input
buckets are priced independently and summed.

## `ragcore.llm.client` — signature details

**`None` means "use settings".** `model`, `effort` and `max_tokens` default to `None`
and resolve to the settings field for that call kind, which is how "no threshold,
model name or limit is hard-coded at a call site" and the contract's
`model=MODEL_MAIN, effort="high", max_tokens=16000` defaults are both satisfied
(the settings defaults *are* those values). Passing `model=MODEL_MAIN` explicitly
also works.

```python
async def complete(*, system=None, messages, tools=None, mcp_servers=None,
                   model=None, effort=None, max_tokens=None, cache_system=True,
                   thinking=True, context_management=None, tool_choice=None,
                   name="llm.complete", metadata=None) -> LLMResponse
def   stream(...)      # same, + name="llm.stream"; max_tokens defaults to
                       # anthropic_max_tokens_streaming (64000)
async def structured(*, system=None, messages, schema: type[T], model=None,
                     effort=None, max_tokens=None, cache_system=True,
                     thinking=True, name="llm.structured", metadata=None) -> T
async def classify(*, system, text, labels, model=None,
                   name="llm.classify", metadata=None) -> str
async def count_tokens(*, system=None, messages, model=None, tools=None) -> int
async def aclose() -> None
@property settings / raw_client
```

* `system` accepts a `str`, one block mapping, or a sequence of either. Any
  `cache_control` a caller puts on a block is **stripped**, then one breakpoint is
  placed on the final block (`cache_system and settings.anthropic_cache_system`).
* `thinking=True` sends `{"type": "adaptive", "display": settings.anthropic_thinking_display}`.
  `thinking=False` sends `{"type": "disabled"}` only at effort `low`/`medium`/`high`;
  at `xhigh`/`max` the field is **omitted with a warning**, because disabling above
  `high` is a 400 and omitting it runs adaptive.
* `LLMResponse` adds `thinking: str` and `stop_details: Any` to the contract fields.
  `LLMUsage` adds `settings` (excluded from equality) plus `total_tokens`,
  `prompt_tokens` and `as_dict()`; `cost_usd()` follows the **serving** model from
  `response.model`, which server-side fallback can change.
* `structured` derives its JSON Schema from the pydantic model, strips the keywords
  structured outputs reject (numeric/string/array constraints, `default`, `pattern`,
  …) and forces `additionalProperties: false` on every object node. A refusal raises
  `LLMRefusedError(RagError, status_code=422, code="llm_refused")` because there is no
  value to return.
* `classify` requires `labels` **safest-first**: a refusal, an unparseable response or
  an unknown label logs a warning and returns `labels[0]`. It never raises into the
  pipeline.
* `count_tokens` degrades to a characters/4 estimate (never `tiktoken`) and logs when
  the API is unreachable after retries, so context assembly cannot hard-fail.
* Retries: `RateLimitError`, 5xx and `APIConnectionError` only; `NotFoundError` and
  other 4xx never. The SDK client is built with `max_retries=0` so backoff is not
  applied twice. `stream()` retries only while **nothing has been yielded**; after the
  first event a failure emits a `StreamEvent(type=ERROR)` and re-raises.

```python
class StreamEventType(StrEnum):
    TEXT
    THINKING
    TOOL_USE
    USAGE
    REFUSAL
    DONE
    ERROR


@dataclass
class StreamEvent:
    type: StreamEventType
    text: str = ""  # delta for TEXT/THINKING
    index: int | None = None  # content-block index
    tool_call: dict | None = None  # {"id","name","input","kind"[,"server_name"]}
    usage: LLMUsage | None = None  # set on USAGE and DONE
    stop_reason: str | None = None
    refused: bool = False
    response: LLMResponse | None = None  # set on DONE
    error: str | None = None  # exception class name on ERROR
```

Event order: interleaved `THINKING`/`TEXT` deltas and completed `TOOL_USE` calls, then
`REFUSAL` if the model declined, then `USAGE`, then exactly one `DONE` carrying the
assembled `LLMResponse`. Map these onto the SSE names in the earlier section
(`thinking`, `token`, `tool_call`, `usage`, `done`, `error`).

Beta flags and context edits:

```python
BETA_SERVER_FALLBACK = "server-side-fallback-2026-07-01"   # + fallbacks="default"
BETA_CONTEXT_MANAGEMENT = "context-management-2025-06-27"
BETA_COMPACTION = "compact-2026-01-12"
CLEAR_TOOL_USES_EDIT = "clear_tool_uses_20250919"
COMPACT_EDIT = "compact_20260112"
def clear_tool_uses_edit(*, clear_tool_inputs=False) -> dict   # context_management=
def compaction_edit() -> dict
```

Because fallbacks are on by default, **every** call goes through
`client.beta.messages`; the stable namespace is used only when fallbacks, MCP and
context management are all off (and for `count_tokens`).

## `ragcore.llm.prompts`

`PROMPT_VERSION` plus `PROMPT_VERSIONS: dict[str, str]` (per-prompt versions) and
`prompt_metadata(name) -> {"prompt", "prompt_version", "prompt_suite"}`, which callers
pass as `metadata=` so every traced generation names its prompt. Constants:
`ANSWER_SYSTEM`, `OOD_REFUSAL_SYSTEM`, `UNCERTAINTY_NOTICE`, `QUERY_TRANSFORM_SYSTEM`,
`HYDE_SYSTEM`, `OOD_ADJUDICATION_SYSTEM`, `CONTRADICTION_SYSTEM`,
`PII_VERIFICATION_SYSTEM`, `TOOL_ROUTING_SYSTEM`, `MEMORY_EXTRACTION_SYSTEM`,
`SESSION_SUMMARY_SYSTEM`, `PROFILE_SUMMARY_SYSTEM`.

Optional user-turn helpers (volatile content must never enter the cached system
prefix): `SourceSnippet(marker, title, text, source_uri, section_path, page, doc_type,
effective_from)`, `render_numbered_sources(snippets) -> str` (the exact numbered form
`ANSWER_SYSTEM` requires, including an explicit "no sources were retrieved" marker),
`render_history(turns)` and `build_answer_user_turn(*, question, sources, memory="",
summary="", preferences="", notes="")`.

## `ragcore.observability` — additions

```python
# langfuse.py
@dataclass TraceHandle: trace_id: str | None; name: str
    def update(**fields) -> None; def score(*, name, value, comment=None) -> None
@dataclass SpanHandle: span_id / trace_id: str | None; name: str
    def update(**fields) -> None
class Tracer:            # the base class IS the working no-op
    ... contract methods ...
    def event(name, **fields) -> None
    def update_observation(obj, **fields) -> None
    def flush() / shutdown() -> None
    enabled: bool
class NoopTracer(Tracer); class LangfuseTracer(Tracer)
def get_current_trace_id() -> str | None      # module-level, no tracer needed
def current_span_id() -> str | None
def flush_tracer(settings=None) / shutdown_tracer(settings=None) -> None
def reset_tracer_cache() -> None              # tests
FAILURE_BUDGET = 20    # consecutive SDK failures then the tracer disables itself
```

`get_tracer` is cached on `(langfuse_ready, host, public_key, sample_rate)` because
`Settings` is unhashable. Every SDK call is wrapped defensively and probes both the v2
(`trace()`/`span()`/`generation()`) and v3 (`start_span()`/`start_generation()`)
spellings. `langfuse_sample_rate` is applied per trace.

**Content policy (binding on all callers):** trace/span/generation payloads carry
redacted or structural data only. `LLMClient` deliberately sends message counts, roles,
block sizes, tool names, betas, token counts and cost — never prompt or answer text,
which has not necessarily passed PII redaction at that point.

```python
# lineage.py
DEFAULT_PROVENANCE_DEPTH = 8
class LineageKind: INGEST|RETRIEVAL|GENERATION|TOOL|EVAL (+ ALL)
class LineageRecord(BaseModel)   # fields exactly as specified earlier; lineage_id and
    # created_at default themselves, and trace_id is inherited from the ambient
    # Langfuse trace when omitted
async def record_lineage(session, record) -> None      # flushes, never commits
async def subject_provenance(session, *, tenant_id, subject_id, max_depth=None) -> dict
async def document_provenance(session, document_id, *, tenant_id,
                              max_depth=None) -> dict
```

`document_provenance` takes **`tenant_id` as a required keyword** — the contract
signature omitted it, and multi-tenancy is a security boundary: a document under
another tenant reads as `found=False` rather than as someone else's provenance. It
returns `{document_id, tenant_id, found, generated_at, document, ingest_run,
ingest_items, records, chain, ancestors, depth, truncated}`, all JSON-serialisable;
serve `found=False` as 404. The walk is upward only (chunk → document → source_uri),
depth-bounded, and needs no JSON-containment SQL, so it behaves identically on
PostgreSQL and sqlite.

```python
# metrics.py
METRICS_CONTENT_TYPE: str; REGISTRY: CollectorRegistry; PROMETHEUS_AVAILABLE: bool
def render_metrics() -> bytes            # body for GET /metrics
def set_build_info(*, service, env, release=None) -> None
def observe_http_request(*, route, method, status_code, latency_ms) -> None
def observe_pipeline_stage(*, stage, latency_ms) -> None
def observe_retrieval_stage(*, stage, latency_ms, candidates=None) -> None
def observe_llm_call(*, model, operation, input_tokens=0, output_tokens=0,
                     cache_read_tokens=0, cache_write_tokens=0, cost_usd=0.0,
                     latency_ms=0.0, outcome="ok") -> None
def observe_cache_lookup(*, cache, hit) -> None
def observe_guardrail(*, stage, kind, action) -> None
def observe_tool_invocation(*, tool, kind, latency_ms, is_error=False) -> None
def observe_ingest_run(*, trigger, status) -> None
def observe_ingest_documents(*, source_type, action, count=1) -> None
```

Series: `rag_build_info`, `rag_http_requests_total`,
`rag_http_request_duration_seconds`, `rag_pipeline_stage_duration_seconds`,
`rag_retrieval_stage_duration_seconds`, `rag_retrieval_candidates`,
`rag_llm_calls_total`, `rag_llm_tokens_total`, `rag_llm_cost_usd_total`,
`rag_llm_call_cost_usd`, `rag_llm_call_duration_seconds`, `rag_cache_lookups_total`,
`rag_guardrail_events_total`, `rag_tool_invocations_total`,
`rag_tool_invocation_duration_seconds`, `rag_ingest_runs_total`,
`rag_ingest_documents_total`. Latency is passed in **milliseconds** and recorded in
**seconds**. Cache hit rate is `rag_cache_lookups_total{result="hit"}` over the sum;
`observe_llm_call` also records a `cache="prompt"` lookup so prompt-cache
effectiveness is visible without extra call sites. Route labels must be route
**templates**, never rendered paths.

## `ragcore.pii` — additions

`PIIFinding.snippet` is a **partially-masked preview**, never the matched value, so a
report is safe to log and persist (`redact` works from offsets against the original
text, so nothing is lost). Redaction is implemented in-process for all three modes
rather than through Presidio's `AnonymizerEngine`, so output is byte-identical with and
without the optional extra — `hash` mode is a join key and must not vary by
deployment.

```python
class PIIFinding(BaseModel): ... + @property length
class PIIReport(BaseModel):  ... + @classmethod empty() / from_findings(list)
                                 + def spans() -> list[tuple[int, int]]
class PIIVerdict(BaseModel): index: int; is_pii: bool; confidence: float = 1.0
class PIIVerificationResult(BaseModel): verdicts: list[PIIVerdict]
class PIIDetector:
    def analyze(text, *, language="en") -> PIIReport
    def redact(text, report, *, mode="mask") -> str        # ValueError on other modes
    def scan_and_redact(text, *, mode=None, language="en") -> tuple[str, PIIReport]
    def pseudonym(value, *, entity_type="VALUE") -> str    # the hash-mode token
    async def verify(text, report) -> PIIReport
    @property presidio_enabled -> bool
def get_pii_detector(settings) -> PIIDetector              # cached
def reset_pii_detector_cache() -> None                     # tests
```

`analyze` filters by `pii_score_threshold` and the `pii_entities` allowlist, then
merges overlapping findings (highest score wins, ties to the longer span). `verify`
may only ever *remove* false positives: any failure returns the input report
unchanged, so protection never disappears because a model call failed.

```python
# recognizers.py
CONTEXT_WINDOW = 64; CONTEXT_BOOST = 0.15
@dataclass(frozen=True) RegexMatch: entity_type; start; end; score; value
@dataclass(frozen=True) RegexRecognizer:
    entity_type; pattern; score=0.6; validator=None; context=(); require_context=False
    group=0; trim_retry=False; name=""
    def find(text) -> list[RegexMatch]
CUSTOM_RECOGNIZERS    # AADHAAR, PAN, SWIFT_CODE, JWT, API_KEY (6 patterns)
BASELINE_RECOGNIZERS  # EMAIL_ADDRESS, CREDIT_CARD, IBAN_CODE, US_SSN,
                      # PHONE_NUMBER, IP_ADDRESS - used in regex-only mode
REGEX_RECOGNIZERS = CUSTOM + BASELINE
def scan(text, *, recognizers=REGEX_RECOGNIZERS, entities=None, min_score=0.0)
def build_presidio_recognizers(*, language="en", recognizers=None) -> list
def luhn_check(v) / iban_check(v) / verhoeff_check(v) / jwt_check(v) -> bool
def verhoeff_checksum(v) -> int
```

Entity names match `settings.pii_entities` exactly. Precision comes from checksums
(Luhn, ISO 7064 mod-97, Verhoeff, base64url JWT header parse), required context for
intrinsically ambiguous patterns (`SWIFT_CODE`), and `trim_retry`, which re-validates
progressively shorter prefixes because greedy regex matching knows nothing about
checksums (`4111 1111 1111 1111 2029` would otherwise fail Luhn as a whole).

---

# Addendum I — infrastructure, containers and operational scripts (owner: `infra/azure`, Dockerfiles, `scripts/`, CI)

Owner: `infra/azure/**`, `services/api/Dockerfile`, `web/Dockerfile`, `web/nginx.conf`,
`scripts/**`, `.github/workflows/ci.yml`, `.dockerignore`. Everything below is either
implemented in those files or is a requirement they place on a neighbouring component.

## `services/ingestion` — operational entry point

`scripts/run_ingest_local.py` and the admin trigger need one importable coroutine that
runs the same pipeline the timer trigger runs:

```python
# ingestion.pipeline (re-exported from `ingestion`)
async def run_ingest(
    *,
    tenant_id: str,
    source_id: str | None = None,          # run one registered source
    source_type: SourceType | None = None, # or every source of one type
    sources: Sequence[SourceConfig] | None = None,  # or these configs verbatim,
                                           # without a source_configs row existing
    trigger: IngestTrigger = IngestTrigger.MANUAL,
    force: bool = False,                   # override the working-hours guard
    dry_run: bool = False,                 # fetch + parse, write nothing
    settings: Settings | None = None,
) -> list[IngestRunSummary]                # one summary per source processed
```

`sources=` is what makes a local run self-sufficient: the caller builds a
`SourceConfig(source_type=LOCAL, options={"root": …})` and no database row is required.
`run_ingest` must still call `Settings.may_start_scheduled_ingest()` and record the
result in `IngestRunSummary.skip_reason` rather than silently running.

`ingestion.cli` exposes `main(argv: Sequence[str] | None = None) -> int` supporting
`run --tenant <id> [--source-id <id>|--source-type <type>] [--force] [--dry-run]`, which
is the form the root `Makefile` already invokes. `run_ingest_local.py` falls back to it
when `run_ingest` is absent.

## Chunk ids in the seeded fixture

Chunk ids are opaque, readable strings — `f"{document_id}::{index:04d}"` — so a golden
item can name one. The UUID translation Qdrant needs stays where Addendum B puts it
(`point_id_for_chunk`, applied by `vectorstore.writer.ChunkPoint`); the seeder does not
mint point ids itself, which is what keeps a re-seed after a real ingestion run an
in-place upsert rather than a duplicate.

The seeder writes through `vectorstore.writer.upsert_chunks` grouped **by tenant** (a
batch that spans tenants raises `TenantMismatchError`), and `--purge` deletes through
`filters.build_tenant_filter` + `writer.hard_delete_by_filter`. No script constructs a
`qm.Filter` of its own.

## Request shapes the scripts bind to

Addendum W owns the request/response bodies. The operational scripts bind to exactly
two of them and will break if they change: `POST /search` takes
`{"query", "filters", "top_n"}`, and `POST /documents` is `multipart/form-data` with the
binary in a part named **`file`** (the smoke test tries `file` first and falls back to
`files`/`upload`/`document` so a differently-named part still produces a useful error
rather than a false failure). Tenant and ACL always come from the `Principal`.

## Demo fixture — the contract the golden set binds to

`scripts/seed_demo_tenant.py` is the single source of truth. Import from it rather than
retyping identifiers:

```python
from seed_demo_tenant import (
    PERSONAS,
    DEMO_TENANTS,
    DEMO_DOCUMENTS,
    CANARIES,
    CONTRADICTION_PAIR,
    documents_visible_to,
    forbidden_canaries_for,
)
```

Tenants: `tenant-acme`, `tenant-globex`. The tenant id doubles as the Entra `tid` of the
dev-mode personas, so a seeded payload matches a resolved `Principal` with no
translation.

Personas — the keys are exactly what `GoldenItem.as_user` carries:

| key | roles | groups | clearance |
|---|---|---|---|
| `acme_admin` | `rag.admin`, `rag.user` | `g-acme-engineering`, `g-acme-hr` | `restricted` |
| `acme_engineer` | `rag.user` | `g-acme-engineering` | `confidential` |
| `acme_intern` | `rag.user` | `g-acme-interns` | `public` |
| `globex_analyst` | `rag.user` | `g-globex-operations` | `confidential` |

Documents:

| document_id | tenant | doc_type | classification | restriction |
|---|---|---|---|---|
| `doc-acme-travel-2023` | acme | policy | public | none (effective to 2025-03-31) |
| `doc-acme-travel-2025` | acme | policy | public | none (effective from 2025-04-01) |
| `doc-acme-onboarding` | acme | handbook | public | none |
| `doc-acme-hr-contact` | acme | note | internal | none — contains PII |
| `doc-acme-remote-work` | acme | standard | internal | none |
| `doc-acme-vpn-runbook` | acme | runbook | internal | group `g-acme-engineering` |
| `doc-acme-salary-bands` | acme | standard | confidential | group `g-acme-hr` |
| `doc-acme-contractor-nda` | acme | contract | confidential | group `g-acme-engineering`, **denies `acme_engineer`** |
| `doc-acme-security-incident` | acme | runbook | restricted | role `rag.admin` |
| `doc-globex-travel-policy` | globex | policy | public | none |
| `doc-globex-warehouse-safety` | globex | standard | internal | none |

Resulting visibility, which the ACL negative tests assert:

* `acme_intern` sees only `doc-acme-travel-2023`, `doc-acme-travel-2025`,
  `doc-acme-onboarding`.
* `acme_engineer` additionally sees `doc-acme-hr-contact`, `doc-acme-remote-work`,
  `doc-acme-vpn-runbook` — but **not** `doc-acme-contractor-nda` (explicit deny beats
  the matching group), not `doc-acme-salary-bands` (wrong group), not
  `doc-acme-security-incident` (clearance and role).
* `acme_admin` sees every Acme document. `globex_analyst` sees only Globex documents.

`CANARIES` maps a document id to a unique token embedded in its text
(`CANARY-ACME-SALARY-7F3A`, `CANARY-ACME-INCIDENT-9B21`, `CANARY-ACME-NDA-4C88`,
`CANARY-ACME-VPN-5A17`, `CANARY-GLOBEX-TRAVEL-1E44`, `CANARY-GLOBEX-SAFETY-2D55`).
`forbidden_canaries_for(principal)` returns the tokens that principal must never see —
use them as `GoldenItem.must_not_contain` for the `acl_negative` category.

`CONTRADICTION_PAIR` is `("doc-acme-travel-2023", "doc-acme-travel-2025")`: the meal
allowance is EUR 45 in the 2023 edition and EUR 60 in the 2025 edition. Stage 7 must
prefer the newer `effective_from` and cite both. The Globex travel policy says EUR 30 in
near-identical wording, so a cross-tenant leak shows up as a wrong number rather than a
plausible answer. The onboarding handbook repeats the travel policy's expense paragraph
verbatim, which is what dedupe collapses.

## Operational scripts

| script | purpose |
|---|---|
| `scripts/bootstrap_qdrant.py` | `ensure_collections` + report; `--verify-only`, `--drop-existing` |
| `scripts/seed_demo_tenant.py` | the fixture above; `--purge`, `--dry-run` |
| `scripts/smoke_test.py` | ingest → search → chat → citations + ACL negatives; `--base-url`, `--skip-ingest`, `--auth-mode dev\|bearer` |
| `scripts/run_ingest_local.py` | one ingestion pass outside Azure Functions |

The root `Makefile` targets `bootstrap`, `seed`, `smoke` and `ingest-local` must point
at these four paths; earlier drafts referenced shorter names (`scripts/bootstrap.py`,
`scripts/seed.py`, `scripts/smoke.py`) that do not exist.

`smoke_test.py` authenticates by sending `Principal.model_dump_json()` in the
`entra_dev_principal_header`, so the target API must run with `RAG_ENTRA_DEV_MODE=true`;
`--auth-mode bearer` reads `SMOKE_TOKEN_<PERSONA>` instead.

## Container images

`services/api/Dockerfile` — build context is the **repository root**. It resolves the uv
workspace into `/opt/venv` (non-editable, so no source is needed at runtime), runs as uid
10001, and starts `uvicorn app.main:app --host $RAG_API_HOST --port $RAG_API_PORT`. The
API must therefore expose `app.main:app` and answer `GET /health` and `GET /readyz`
without auth. FastEmbed weights are cached in `$RAG_EMBEDDING_CACHE_DIR`
(`/home/app/.cache/fastembed`); mount a volume there or accept a re-download per restart.

`web/Dockerfile` — build context is `web/`. Vite `VITE_*` values are **build arguments**
(`VITE_API_BASE_URL`, `VITE_ENTRA_CLIENT_ID`, `VITE_ENTRA_TENANT_ID`,
`VITE_ENTRA_API_SCOPE`, `VITE_DEV_MODE`); the image then serves the built assets with
`nginx-unprivileged` on port 80. `web/nginx.conf` is an envsubst template with two
runtime variables — `API_UPSTREAM` (default `http://api:8000`) and `NGINX_PORT` (default
`80`) — and it reverse-proxies `/api/` with buffering disabled so the SSE chat stream is
not batched. Building with an empty `VITE_API_BASE_URL` makes the SPA same-origin through
that proxy, which is the deployed configuration.

`web/package.json` must expose a `build` script (CI fails without it) and may expose
`lint` and `typecheck`, which CI runs with `--if-present`. `web/` should also ship its
own `.dockerignore`; the root one does not apply to the `web/` build context, and the
Dockerfile is written to be correct without it (`npm ci` recreates `node_modules` from
scratch, so a host `node_modules` cannot leak in).

## Deployment outputs → settings

`main.bicep` outputs exactly what the workloads need, and `containerapps.bicep` /
`functions.bicep` already wire them into `RAG_*` app settings:

| output | setting |
|---|---|
| `qdrantUrl` | `RAG_QDRANT_URL` (internal `http://…internal.<domain>`) |
| `postgresHost` · `postgresUser` · `postgresDatabase` | `RAG_POSTGRES_HOST` · `_USER` · `_DB` |
| `blobEndpoint` | `RAG_AZURE_BLOB_ACCOUNT_URL` |
| `sourcesContainerName` · `rawContainerName` · `manifestsContainerName` | `RAG_AZURE_BLOB_CONTAINER` · `_RAW_CONTAINER` · `RAG_INGEST_MANIFEST_CONTAINER` |
| `ingestQueueName` | `RAG_AZURE_STORAGE_QUEUE_NAME` |
| `keyVaultUri` | `RAG_AZURE_KEY_VAULT_URL` |
| `apiIdentityClientId` / `ingestionIdentityClientId` | `RAG_AZURE_CLIENT_ID` per workload |
| `apiUrl` · `webUrl` | smoke test target · MSAL redirect URI |

The ingestion Function App binds its timer trigger to `%RAG_INGEST_CRON%` (six-field
NCRONTAB) and reads `WEBSITE_TIME_ZONE` from `RAG_INGEST_TIMEZONE`, so the host schedule
and `Settings.is_within_working_hours()` agree on the timezone.


---

# Addendum S — serverless delta ingestion (owner: `services/ingestion`)

Requirement #1, end to end. Everything the earlier sections and Addendum I say about
ingestion is implemented; this addendum pins the surface a downstream module binds to
and the decisions those sections left open. **Nothing here changes an existing
signature.**

## Operational entry points

```python
# ingestion.pipeline — re-exported lazily from `ingestion`
async def run_ingest(*, tenant_id: str,
                     source_id: str | None = None,
                     source_type: SourceType | None = None,
                     sources: Sequence[SourceConfig] | None = None,
                     trigger: IngestTrigger = IngestTrigger.MANUAL,
                     force: bool = False,
                     dry_run: bool = False,
                     full_scan: bool = False,          # added here
                     settings: Settings | None = None) -> list[IngestRunSummary]

async def run_ingestion(*, tenant_id=None, source_ids=None,
                        trigger=IngestTrigger.MANUAL, forced=False, settings=None,
                        now=None, enforce_schedule=False,
                        since=None) -> list[IngestRunSummary]
```

* `run_ingest` is the coroutine Addendum I specifies, with one extra keyword:
  **`full_scan`** clears each source's stored delta cursor for that run, so the
  enumeration is complete and deletion detection becomes sound again. It is what
  `POST /admin/ingest/trigger`'s `full_scan` maps onto.
* `run_ingest` **always** evaluates `Settings.may_start_scheduled_ingest()`. A refusal
  returns exactly one `IngestRunSummary` with `status=SKIPPED` and
  `skip_reason ∈ {"working_hours", "disabled"}` — never an empty list, so "refused" and
  "nothing to do" are distinguishable. `force=True` overrides `working_hours` but never
  `disabled`.
* `sources=` is used verbatim and needs no `source_configs` row. Every supplied config
  must carry the run's `tenant_id`; a mismatch raises `ValueError` rather than being
  corrected, because multi-tenancy is a security boundary.
* `run_ingestion` is the **multi-tenant** variant the nightly timer uses (`tenant_id`
  omitted = every active tenant) and is the only one with `enforce_schedule=False`
  behaviour, for an operator-initiated cross-tenant sweep.

```python
# ingestion.cli — required by Addendum I
def main(argv: Sequence[str] | None = None) -> int
# run --tenant <id> [--source-id <id> | --source-type <type>] [--root <dir>]
#     [--include GLOB] [--exclude GLOB] [--classification <level>]
#     [--force] [--dry-run] [--full-scan]
# exit codes: 0 ok, 1 failed, 2 refused by the working-hours guard
```

With `--source-type local` (or `--root`) and no matching `source_configs` row, the CLI
synthesises an ephemeral local source rooted at `RAG_INGEST_LOCAL_ROOT`, so
`make ingest-local` works on a machine with an empty database.

**Dry-run semantics.** `dry_run=True` enumerates, resolves ACLs, classifies, fetches,
parses, PII-scans, chunks and dedupes for real, and writes nothing — no Qdrant client
is opened, no `documents`/`ingest_runs`/`ingest_items` row is written, and the manifest
is read but never saved. The returned summary is a **projection**: `documents_*` say
what the run *would* do, `chunks_upserted`/`tokens_embedded` stay 0 because nothing was
written, and the would-be totals are in `metrics` as `chunks_planned`,
`tokens_planned`, `deletions_planned`, alongside `metrics["dry_run"] == 1.0`.
Enrichment is skipped (no LLM spend); the PII scan still runs.

## Pipeline phases and the fan-out payloads

```python
class DocumentTask(BaseModel)     # run_id, tenant_id, source, descriptor,
                                  # planned_action, reason, previous_entry, forced,
                                  # trigger — serialisable, one Durable activity's input
class DocumentOutcome(BaseModel)  # what happened to one document + drop_reasons
class SourcePlan(BaseModel)       # tasks, skipped, deletions, cursor, full_scan

async def plan_source(source, *, run_id, settings=None, trigger=..., forced=False,
                      since=None, store=None) -> SourcePlan
async def process_document(task, *, settings=None, upserter=None,
                           connector=None) -> DocumentOutcome
async def process_documents(tasks, *, settings=None,
                            share_dedupe_state=True) -> list[DocumentOutcome]
async def finalize_run(plan, outcomes, *, source, settings=None, store=None,
                       started_at=None, within_working_hours=False) -> IngestRunSummary
async def dry_run_source(source, *, ...) -> IngestRunSummary
async def resolve_sources(*, tenant_id, source_id=None, source_type=None,
                          sources=None, settings=None) -> list[SourceConfig]
async def start_run(source, *, trigger, forced=False,
                    settings=None) -> tuple[str, datetime]
async def ingest_single_document(*, tenant_id, source_id, source_uri,
                                 trigger=IngestTrigger.UPLOAD, settings=None,
                                 forced=False) -> IngestRunSummary
async def ingest_uploaded_document(*, tenant_id, source_id, filename, payload,
                                   access_control, media_type=None, title="",
                                   doc_type="document", tags=None, author=None,
                                   language=None, settings=None,
                                   run_id=None) -> DocumentOutcome
```

`ingest_uploaded_document` is the `POST /documents` path: the ACL comes from the
uploading principal, `source_type` is `upload`, and the bytes go through the same
parse → enrich → chunk → dedupe → embed → write pipeline, so an uploaded chunk is
indistinguishable from a crawled one.

**Two-tier delta, and why a re-run writes nothing.** Phase 1 classifies on listing
metadata only (ETag, `last_modified`, ACL fingerprint) — a provably unchanged document
is never downloaded. Phase 2 re-classifies on the **content hash** once the bytes are
in hand, so a file that was touched, copied or re-uploaded without being edited is
skipped before parsing or embedding. `IngestManifestEntry.decide` (which treats a moved
ETag as an update) is used only when no hash is available.

**Idempotency and resumability.** Document ids are `sha256(tenant \x00 source_uri)[:32]`,
chunk ids are positional, point ids derive from chunk ids, the `documents` row is an
upsert and manifest folding is last-write-wins per document. A Durable activity retry,
a re-queued document or a re-run of a failed run therefore repeats work rather than
duplicating it. `start_run` writes the `running` row *before* any document is touched,
so an invocation killed mid-flight still leaves an auditable record.

## Chunk and point identity

Chunk ids are `f"{document_id}::{index:04d}"` (`ingestion.upsert.chunk_id_for`), the
form Addendum I pins so a golden item can name one. The Qdrant point id is derived from
it by `ragcore.vectorstore.writer.ChunkPoint.point_id` → `point_id_for_chunk`; ingestion
never mints a point id. `chunk_index` is the position **after** dedupe, so indexes are
contiguous from zero and a re-seed of the demo fixture over a real run is an in-place
upsert.

## `ingestion.upsert`

```python
class ChunkWriter(Protocol)       # upsert_chunks / prune_document /
                                  # soft_delete_document / rewrite_access_control /
                                  # find_by_content_hash / close
class QdrantChunkWriter           # the implementation; delegates to
                                  # ragcore.vectorstore.writer for every write
class NullChunkWriter             # the dry-run writer; opens no client
class DocumentUpsertResult        # chunks_written/_deleted, tokens_embedded,
                                  # duplicates_dropped, drop_reasons, chunk_ids
class RunUpserter                 # run-scoped dedupe state + writes + lineage
async def get_chunk_writer(settings) -> ChunkWriter
def chunk_id_for(document_id: str, chunk_index: int) -> str
```

* Drop reasons are `duplicate_exact_run`, `duplicate_exact_corpus`,
  `duplicate_simhash` and `empty_after_normalisation`. `empty_after_normalisation` is
  counted in `drop_reasons` but **not** in `duplicates_dropped`.
* Dedupe is three layers: exact hash within the run, exact hash against the tenant's
  already-indexed corpus (the `content_sha256` payload index — this is what makes
  cross-document dedupe survive the Durable fan-out, where each activity is a separate
  process), and banded simhash LSH within the run at `dedupe_max_distance`.
* **Pruning hard-deletes, document deletion tombstones.** After an upsert, chunk
  *positions* the document no longer has are purged with
  `hard_delete_by_filter(build_tenant_filter(tenant, document_ids=[…], chunk_ids=stale))`
  — they are stale rows of a document being replaced, not content removed at source.
  A document that disappeared goes through `writer.soft_delete_document`, which sets
  `is_deleted=True` and keeps the lineage.
* The only `qm.Filter` this service composes is the content-hash dedupe probe, and it
  is built by extending `build_tenant_filter(tenant_id, include_deleted=False).must`
  with one `content_sha256` clause. The tenant boundary and the tombstone exclusion
  always come from `ragcore.vectorstore.filters`.
* Lineage: one `ingest`/`ingest_document` record per document with
  `parents=[source_uri]`, plus one `ingest`/`chunk` record per chunk with
  `parents=[document_id, source_uri]` — the chain
  `GET /documents/{id}/lineage` walks.

## `ingestion.delta`

```python
class DeltaDecision(BaseModel)    # document_id, source_uri, action, reason,
                                  # content_sha256, acl_fingerprint, previous_version
                                  # + .needs_content, .next_version
class ChangeSet(BaseModel)        # decisions, deletions, full_scan, .counts()
class ManifestStore(Protocol)     # load / save / close
class BlobManifestStore           # ETag-conditioned writes, merge-and-retry on 412
class LocalManifestStore          # filesystem mirror for local runs and tests
def get_manifest_store(settings) -> ManifestStore
def classify_document(manifest, doc, *, acl_fingerprint, not_modified=False,
                      force=False) -> DeltaDecision
def detect_deletions(manifest, seen, *, full_scan, enabled=True) -> list[str]
def manifest_entry_for(doc, *, run_id, acl_fingerprint, version,
                       chunk_count=0, token_count=0,
                       content_sha256=None) -> IngestManifestEntry
async def mirror_items_to_postgres(session, *, tenant_id, run_id, decisions,
                                   statuses=None, metrics=None) -> int
async def tombstone_documents(*, tenant_id, document_ids, manifest, run_id,
                              upserter, session=None) -> int
def manifest_summary(manifest) -> str    # counts only, never a URI
```

Manifests live at `<tenant_id>/<source_id>.json` in
`settings.ingest_manifest_container` (Blob) or under
`dirname(ingest_local_root)/<container>/` locally — tenant-first, so the container is
already tenant-partitioned. A corrupt or foreign-tenant manifest degrades the run to a
full rescan; it never fails it. Reason strings, which land in `ingest_items.reason`:
`new`, `content_changed`, `unchanged`, `unchanged_touched`, `acl_changed`,
`deleted_at_source`, `forced`, `not_modified`, `reappeared`,
`missing_from_full_scan`.

**Deletion detection requires a full scan.** A SharePoint `deltaLink` pass or a SQL
watermark pass deliberately does not mention unchanged documents, so
`detect_deletions` returns `[]` unless `connector.performed_full_scan` is true.

## `ingestion.connectors`

```python
@runtime_checkable
class SourceConnector(Protocol)   # list_documents / fetch / resolve_acl / close
                                  # + source_type, supports_delta
class BaseConnector(ABC)          # descriptor(), apply_fetched(), resolve_acl(),
                                  # prime_delta_state(entries), known_etag(),
                                  # known_modified_at(), cursor, performed_full_scan,
                                  # within_size_limit(), async context manager
@dataclass FetchedContent         # content_bytes | content_text, media_type, etag,
                                  # source_modified_at, size_bytes, metadata,
                                  # not_modified, .sha256
class ConnectorError(RagError)    # status_code 502, code "connector_error"
def get_connector(source, settings) -> BaseConnector   # the only SourceType -> class map
def make_document_id(tenant_id, source_uri) -> str     # sha256(tenant\0uri)[:32]
def guess_media_type(name, *, default="application/octet-stream") -> str
def azure_credential(settings) -> DefaultAzureCredential
async def resolve_secret(ref, settings) -> str | None  # Key Vault name, then env
```

Two delta styles: a connector with a server-side cursor (`SharePointConnector`,
`SqlSourceConnector`; `supports_delta = True`) advances `.cursor` and the pipeline
persists it to the manifest and to `source_configs.cursor`. A connector without one
(`AzureBlobConnector`, `LocalFilesystemConnector`, `HttpCrawlerConnector`) is primed
with the previous run's manifest entries via `prime_delta_state()` and does conditional
requests, returning `FetchedContent(not_modified=True)` instead of a payload.

Per-connector specifics worth knowing:

| connector | delta signal | ACL source |
|---|---|---|
| `AzureBlobConnector` | blob ETag + `last_modified`; listing includes metadata and index tags | `<name>.acl.json` sidecar → blob metadata (`acl_*`) → index tags → source defaults |
| `LocalFilesystemConnector` | `st_size`-`st_mtime_ns` synthetic ETag; `include_globs`/`exclude_globs` | `<name>.acl.json` sidecar → source defaults |
| `SharePointConnector` | Graph `/drives/{id}/root/delta`, `@odata.deltaLink` persisted as the cursor; tombstones carry the same document id as the live item | Graph `/items/{id}/permissions` → Entra group and user object ids; organisation-wide links leave the document unrestricted |
| `HttpCrawlerConnector` | `If-None-Match` / `If-Modified-Since`; 304 → skip | source config only (a web page has no ACL) |
| `SqlSourceConnector` | `watermark_column`, always a `:watermark` **bind parameter**; the column name is identifier-validated | per-row `acl_*` columns merged with source defaults; a row whose `tenant_column` disagrees with the source is **dropped, not indexed** |

`robots.txt` is honoured per host, the crawl is bounded by `ingest_http_max_pages` and
`ingest_http_concurrency`, and off-domain URLs are never fetched. Document identity
survives a SharePoint rename because the id is derived from the drive item id rather
than from the display path.

## `ingestion.acl`

```python
SIDECAR_SUFFIX = ".acl.json"; ACL_METADATA_PREFIX = "acl_"
def sidecar_name_for(name) -> str
def parse_identifier_list(value) -> list[str]        # JSON array or , ; | delimited
def access_control_from_mapping(data, source) -> AccessControl | None
def access_control_from_sidecar(raw, source) -> AccessControl | None
def access_control_from_metadata(metadata, tags, source) -> AccessControl | None
def access_control_from_graph_permissions(permissions, source) -> AccessControl | None
def merge_with_source_defaults(item, source) -> AccessControl
def acl_fingerprint(ac: AccessControl) -> str        # order-insensitive, tenant-scoped
```

Sidecar / metadata keys are `classification`, `allowed_roles`, `allowed_groups`,
`allowed_users`, `denied_users` (metadata keys additionally accept the `acl_` prefix;
blob index tags win over blob metadata for the same key). An ACL declaring a foreign
`tenant_id` is refused, never rewritten. `acl_fingerprint` is what turns a
permissions-only change into an `ACL_ONLY` reindex — payload rewrite, no re-embedding.

## `ingestion.parse`, `ingestion.chunk`, `ingestion.enrich`

```python
# parse.py
def parse_document(doc: SourceDocument, settings=None) -> ParsedDocument
class ParseError(RagError); class SectionTracker
def normalise_text(text) -> str; def render_table(rows) -> str
# per-format: parse_pdf / parse_docx / parse_pptx / parse_xlsx / parse_html /
#             parse_markdown / parse_csv / parse_text
```

PDF via pdfplumber with a pypdf fallback, DOCX via python-docx, PPTX via python-pptx,
XLSX via openpyxl, HTML via selectolax with a BeautifulSoup fallback. Every block
carries `kind`, `order`, heading `level`, `section_path` and `page`. Tables are rendered
to pipe-delimited text with the header row retained, so a table split across chunks
keeps its header.

```python
# chunk.py
class ChunkDraft(BaseModel)   # chunk_index, text, contextual_header, section_path,
                              # page, token_count, block_kinds + .embed_text
def chunk_document(parsed, settings=None, *, doc_summary=None) -> list[ChunkDraft]
def build_contextual_header(title, section_path, summary=None) -> str
def estimate_tokens(text) -> int; def split_sentences(text) -> list[str]
```

The header is `"<title> > <section> > <subsection>"` plus an optional one-line document
summary. It is **prepended only for embedding** (`ChunkPayload.embed_text`) and stored
in its own `contextual_header` payload field, so a retrieved chunk's `text` is the
document's own words. Headings start a new chunk when `chunk_respect_headings`; tables
are never merged with prose and are split by rows with the header repeated; runt
sections below `chunk_min_tokens` are merged forwards rather than emitted as fragments.
`estimate_tokens` is a cheap local estimate, never `tiktoken`.

```python
# enrich.py
class DocumentInsights(BaseModel)     # summary, keywords, doc_type
class EnrichmentCache                 # keyed by content_sha256
async def enrich_document(parsed, *, settings=None, llm=None, detector=None,
                          cache=None) -> ParsedDocument
async def enrich_documents(documents, *, ...) -> list[ParsedDocument]
def scan_and_redact(parsed, settings, detector=None
                    ) -> tuple[ParsedDocument, list[str], bool]
def detect_language(text, *, default="en") -> str
def extract_effective_dates(text) -> tuple[datetime | None, datetime | None]
def heuristic_insights(parsed, settings) -> DocumentInsights
def allowed_doc_types(settings) -> list[str]
```

One `MODEL_FAST` structured call per distinct content hash produces summary, keywords
and `doc_type`; `doc_type` labels are `guardrail_doc_type_authority` plus `"document"`,
so the classifier and the contradiction resolver can never drift apart. When no API key
is configured, or the model refuses, `heuristic_insights` supplies all three and the
run continues.

**PII: redact before anything is persisted.** `scan_and_redact` runs over every parsed
block *before* enrichment, chunking, embedding, logging or lineage, so no raw value
reaches a prompt, a payload or a log line. The outcome travels as
`ParsedDocument.metadata["pii_types"]` (list) and `["pii_redacted"]` (bool), which
`RunUpserter` copies into `ChunkPayload.pii_types` / `pii_redacted`. Error messages
persisted into `ingest_runs.errors` are reduced to the exception type (plus the message
only for `RagError`, whose messages are contractually content-free), because an
exception can quote the document.

## Azure Functions triggers (`function_app.py`)

| function | trigger | behaviour |
|---|---|---|
| `ingest_timer` | timer `%RAG_INGEST_CRON%` | guard first; on refusal records a skipped run, otherwise starts the orchestrator |
| `ingest_http` | `POST /api/ingest/trigger` | `wait=true` runs inline (`run_ingest` for one tenant, `run_ingestion` across tenants); otherwise starts the orchestrator and returns the Durable status URLs |
| `ingest_orchestrator` | orchestration | plan activity → waves of `ingest_batch_size` document activities → finalise activity, one summary per source |
| `ingest_plan_activity` · `ingest_document_activity` · `ingest_finalize_activity` | activity | the three pipeline phases |
| `ingest_blob` | blob `%RAG_AZURE_BLOB_CONTAINER%/{name}` | immediate single-document ingestion; the owning source is resolved by longest matching prefix, `.acl.json` sidecars are ignored |
| `ingest_retry` | queue `%RAG_AZURE_STORAGE_QUEUE_NAME%` | a serialised `DocumentTask` (retry one document) or `{tenant_id, source_id, source_uri, force?}` (reindex one) |

The HTTP body accepts `tenant_id`, `source_id`/`source_ids`, `force`, `full_scan`,
`dry_run`, `wait` and `enforce_schedule`. Orchestrator determinism is preserved by
keeping every clock read, settings read and I/O inside an activity — the batch size
travels in the plan envelope rather than being read from the environment during replay.

**Three settings are consumed as binding expressions** and must exist as app settings,
not only in `.env`, or the host refuses to index the functions: `RAG_INGEST_CRON`,
`RAG_AZURE_BLOB_CONTAINER`, `RAG_AZURE_STORAGE_QUEUE_NAME`.

## Notes for neighbouring components

* The root `Makefile`'s `ingest-local` target (`python -m ingestion.cli run
  --source-type local --tenant … --force`) works as written; `scripts/run_ingest_local.py`
  resolves `ingestion.pipeline.run_ingest` and never needs its CLI fallback.
* `services/ingestion/Dockerfile` copies the repository `README.md`, so the build
  context must contain one.
* **`services/ingestion/tests/` deliberately has no `__init__.py`.** Under pytest's
  default `prepend` import mode, two test directories both named `tests` and both
  carrying an `__init__.py` resolve to the same module name (`tests.*`) and the
  single root `pytest` run CI executes fails at collection — `packages/ragcore/tests`
  already claims that name. Any service adding tests should do the same, or the root
  `[tool.pytest.ini_options]` should set `importmode = "importlib"`, after which
  package markers are safe again.
* `ingestion` imports nothing heavy at package import time; `run_ingest` is re-exported
  through a module-level `__getattr__`. Import `ingestion.pipeline` directly if you want
  the cost up front (the Functions host does).


---

# Addendum T — tool calling, REST and MCP (owner: `services/api/app/rag/tools`)

Requirement #4, pipeline stage 8. Everything the earlier sections specify for
`app/rag/tools/**` is implemented; this addendum records what those sections left
open. Nothing here changes an existing signature.

## Dependency additions

`services/api/pyproject.toml` needs **`httpx>=0.28`** (the REST executor) and
**`mcp>=1.2`** (the official MCP Python SDK, used only for *self-hosted* servers).
`PyYAML` and `jsonschema` are **optional**: the registry ships a restricted YAML
subset parser and a built-in JSON-Schema validator, and uses the real libraries when
they are installed. A missing `mcp` disables the local MCP path with a warning; it
never blocks import.

## `ragcore.settings` — tool tunables this layer adds

`Settings` already carries `tool_enabled`, `tool_max_iterations`,
`tool_timeout_seconds`, `tool_max_result_chars`, `tool_registry_path`,
`tool_mcp_enabled`, `tool_mcp_beta_flag` and `tool_allow_insecure_http`. The rest of
the layer's tunables live on `registry.ToolTuning`, a `pydantic-settings`
`BaseSettings` with the **same** `env_prefix="RAG_"` and `.env` file, so an operator
configures them identically. `tool_config(settings) -> ToolConfig` merges the two and
**`Settings` always wins**, so moving any of these onto `Settings` later is a no-op
for callers. No call site hard-codes any of them.

| field | default | meaning |
|---|---|---|
| `tool_retry_attempts` | `2` | retries after the first REST attempt |
| `tool_retry_backoff_seconds` / `tool_retry_max_backoff_seconds` | `0.25` / `4.0` | jittered backoff base and ceiling |
| `tool_circuit_failure_threshold` / `tool_circuit_reset_seconds` | `5` / `30.0` | breaker open threshold and cool-down |
| `tool_rate_limit_per_minute` | `30` | per-**tenant**, per-tool token bucket |
| `tool_max_classification` | `internal` | default egress ceiling |
| `tool_max_response_bytes` | `2_000_000` | hard cap on a REST body, enforced while streaming |
| `tool_max_projected_items` | `25` | list length kept after projection |
| `tool_response_pii_scan` | `true` | redact PII in a tool response before the prompt |
| `tool_secret_cache_seconds` | `300.0` | TTL of a resolved Key Vault secret |
| `tool_oauth_expiry_skew_seconds` | `60.0` | refresh an OAuth token this early |
| `tool_oauth_token_url_template` | `{authority}/{tenant}/oauth2/v2.0/token` | client-credentials endpoint |
| `tool_oauth_client_secret_ref` | `None` | Key Vault secret holding the client secret |
| `tool_managed_identity_endpoint` / `_api_version` | IMDS / `2018-02-01` | managed-identity token source |
| `tool_secret_env_prefix` | `RAG_TOOL_SECRET_` | environment fallback when Key Vault is absent |
| `tool_loop_repeat_limit` | `2` | identical `(tool, arguments)` calls tolerated |
| `tool_router_min_confidence` | `0.45` | retrieval confidence below which tools are attempted |
| `tool_mcp_local_enabled` | `true` | allow self-hosted MCP over the `mcp` SDK |
| `tool_mcp_discovery_ttl_seconds` | `300.0` | TTL of a cached local tool listing |
| `tool_mcp_connect_timeout_seconds` | `10.0` | connect timeout for a local session |
| `tool_mcp_disable_seconds` | `120.0` | cool-down after an unreachable MCP server |
| `tool_result_cache_max_entries` | `256` | entries in the cacheable-tool cache |
| `tool_result_summary_chars` | `500` | length of the `tool_invocations.result_summary` preview |
| `tool_user_agent` / `tool_follow_redirects` | `productionizing-rag-tools/0.1` / `false` | outbound request hygiene |

## `app.rag.tools.registry`

```python
NEVER_FORWARD: Classification = Classification.RESTRICTED
DEFAULT_CONFIG_FILENAMES = ("tools.yaml", "tools.example.yaml")

class ToolPolicy(BaseModel)           # per-tool overrides + the egress gate
    timeout_seconds / rate_limit_per_minute / retry_attempts
    circuit_failure_threshold / circuit_reset_seconds
    max_classification: Classification | None
    allow_pii_in_arguments: bool = False
    response_pii_scan: bool | None
    require_admin: bool = False
    def resolved_max_classification(*, kind, default) -> Classification

class LocalMcpServerSpec(BaseModel)   # self-hosted: transport stdio|streamable_http
    def is_allowed_for(principal) -> bool

@dataclass(frozen=True) class RegisteredTool
    spec / policy / timeout_seconds / rate_limit_per_minute
    max_classification / max_result_chars / server_name
    def may_receive(classification) -> bool
    def is_allowed_for(principal) -> bool

class RateLimiter                      # per (tenant_id, tool) token bucket
class ToolRegistry
    tools_for(principal) / get(name, principal) / require(name, principal)
    anthropic_tools(principal) / mcp_specs_for(principal)
    local_mcp_servers_for(principal) / register(tool) / resolve_policy(spec, policy)

def tool_config(settings=None) -> ToolConfig
def register_tool(spec, *, policy=None, config=None, settings=None) -> RegisteredTool
def build_registry(settings=None, *, document=None, path=None, extra_tools=()) -> ToolRegistry
def get_tool_registry(settings=None) -> ToolRegistry        # cached; adds the built-ins
def reset_tool_registry_cache() -> None
def load_tool_document(path) -> dict
def config_search_paths(settings=None) -> list[Path]
def redact_arguments(arguments, *, detector=None, settings=None, max_chars=512) -> dict
```

* **Tenant is checked before role.** `ToolSpec.tenant_id` / `LocalMcpServerSpec.tenant_id`
  pin a tool to one tenant; a principal from another tenant cannot see it, call it, or
  learn it exists — `require()` raises the same message for "unknown" and "denied".
* **`RESTRICTED` never leaves the platform.** `resolved_max_classification` clamps any
  non-`retrieval` tool's ceiling below `NEVER_FORWARD` even when the config asks for
  more, and `may_receive` re-checks at dispatch time.
* `redact_arguments` runs two passes — credential-looking keys are replaced outright,
  every remaining string goes through the PII detector. It is the only shape of
  arguments that is ever logged, traced or persisted.
* The config file is YAML or JSON. When PyYAML is absent a restricted subset parser
  handles the shipped format: block mappings, block sequences, quoted and bare
  scalars, inline JSON for flow collections, `#` comments. **No block scalars
  (`|`, `>`), anchors or multi-document files** — `config/tools.example.yaml` stays
  inside that subset and a test asserts both parsers agree.

## `app.rag.tools.rest_tool`

```python
class InvalidToolArgumentsError(ToolExecutionError)   # 400, tool_invalid_arguments
def validate_arguments(schema, arguments, *, tool_name) -> dict
def coerce_arguments(schema, arguments) -> dict
def render_template(template, arguments, *, quote_values=False, strict=True) -> str
def project_response(payload, path, *, max_items=None) -> Any
class SecretResolver;  class TokenProvider;  class CircuitBreaker / CircuitState
@dataclass(frozen=True) class PreparedRequest
class RestExecutor
    async prepare(tool, arguments) -> PreparedRequest
    async execute(tool, *, tool_call_id, arguments) -> ToolResult
    async aclose() -> None
def get_rest_executor(settings=None) -> RestExecutor
async def reset_rest_executor() -> None                # shutdown hook
```

* Validation **always rejects undeclared properties** unless the schema sets
  `additionalProperties: true`, with or without `jsonschema` installed, so an
  argument the model invented can never reach a template.
* URL-path values are percent-encoded with `safe=""`; a query or body placeholder
  whose argument is absent drops out instead of rendering empty. A missing **path**
  placeholder is an error — a hole in a path is a different resource.
* Auth mapping: `bearer`/`api_key`/`basic` resolve `auth_secret_ref` through Key Vault
  (`azure_key_vault_url` + the `azure-keyvault-secrets` SDK) and fall back to
  `$RAG_TOOL_SECRET_<REF>`; `entra_obo` and `managed_identity` perform an OAuth2
  client-credentials grant / IMDS fetch for `auth_scope`, cached per scope until
  `tool_oauth_expiry_skew_seconds` before expiry.
* Non-`https` is refused unless `tool_allow_insecure_http` (itself refused in
  production by `Settings`). Redirects are not followed by default.
* `response_json_path` is the JMESPath subset `a.b[0].c` / `a.b[*].c`; an unresolved
  path yields `None` rather than raising, so an upstream shape change degrades the
  tool and not the turn. List results are capped at `tool_max_projected_items`.
* The response is PII-scanned before it becomes tool-result content, then truncated
  to `RegisteredTool.max_result_chars`.
* `execute` **never raises**. Invalid arguments, an open circuit, a timeout and a bad
  status all return `ToolResult(is_error=True)`.

## `app.rag.tools.mcp_client`

```python
LOCAL_NAME_SEPARATOR = "__"
def mcp_sdk_available() -> bool
@dataclass(frozen=True) class ConnectorRequest
    betas: list[str]; mcp_servers: list[dict]; tools: list[dict]
    @property is_empty -> bool ; def as_kwargs() -> dict
def build_connector_request(specs, *, tokens=None, beta_flag=None) -> ConnectorRequest
class RemoteMcpConnector
    async build(registry, principal) -> ConnectorRequest
def local_tool_name(server, tool) -> str ; def split_local_tool_name(name) -> tuple[str, str]
def translate_mcp_tool(server, tool) -> dict
def render_mcp_content(outcome) -> tuple[str, dict | None]
@dataclass(frozen=True) class LocalMcpTool      # name/description/input_schema/server/policy/limits
@dataclass class McpToolListing
class LocalMcpClient
    async discover(spec, *, force=False) -> McpToolListing
    async discovered_tools(registry, principal) -> list[LocalMcpTool]
    async call(spec, *, tool_call_id, tool_name, arguments,
               max_result_chars, timeout_seconds=None) -> ToolResult
    def is_disabled(server) -> bool ; def disable(server, *, reason) -> None
```

* **(a) Remote.** `build_connector_request` emits `mcp_servers`, the matching
  `mcp_toolset` tool entries and `betas` **as one object** — there is no way to obtain
  one half. `as_kwargs()` returns `{}` when no server applies, so a turn without MCP
  never sends an empty `mcp_servers`. `McpServerSpec.allowed_tools` becomes
  `default_config: {"enabled": false}` plus per-tool `configs`.
* **(b) Self-hosted.** A `LocalMcpTool` is deliberately **not** a `ToolSpec` of
  `kind="mcp"`: that shape describes a remote server Anthropic dials and requires an
  https URL, which a stdio child process does not have. `LocalMcpTool` exposes the
  same surface the loop needs (`name`, `kind == ToolKind.MCP`, `policy`,
  `timeout_seconds`, `rate_limit_per_minute`, `max_classification`,
  `max_result_chars`, `is_allowed_for`, `may_receive`, `to_anthropic_tool`) and is
  recorded with `kind="mcp"` on `tool_invocations`. It reaches the model as an
  ordinary **client-side** tool definition, because the platform holds the connection.
* Local tool names are `"<server>__<tool>"`, sanitised to `[A-Za-z0-9_-]`.
* Discovery is cached per server for `tool_mcp_discovery_ttl_seconds`. An unreachable
  server is **disabled for `tool_mcp_disable_seconds`**, not raised: the model simply
  does not see those tools this turn, and a disabled server is not dialled again.

## `app.rag.tools.builtin`

```python
SEARCH_TOOL_NAME = "search_corpus" ; CONTEXT_TOOL_NAME = "current_context"
class RetrieveFn(Protocol)
    async __call__(*, query, principal, filters, top_n) -> RetrievalResult
def search_tool_spec(settings=None) -> ToolSpec
def context_tool_spec(settings=None) -> ToolSpec
def builtin_tools(settings=None) -> list[RegisteredTool]     # always registered
def filter_from_arguments(arguments, *, base=None) -> MetadataFilter | None
async def qdrant_retrieve(*, query, principal, filters, top_n, settings=None) -> RetrievalResult
def render_chunks(result, *, snippet_chars) -> str
class BuiltinExecutor
    async execute(tool, *, tool_call_id, arguments, principal,
                  base_filter=None, retrieve=None) -> ToolResult
```

Both built-ins are `kind="retrieval"` and stay in-process, so their egress ceiling is
`RESTRICTED`. `filter_from_arguments` intersects the model's facets with the turn's
filter via `MetadataFilter.merged_with`, so a tool-issued search can only **narrow**
what the user could already search. The orchestrator should inject its stage-5
retriever as `ToolContext.retrieve`; with nothing injected the tool falls back to
`qdrant_retrieve`, a real `build_acl_filter` + `hybrid_search` query that skips only
rerank/MMR.

## `app.rag.tools.router`

```python
type ExposedTool = RegisteredTool | LocalMcpTool

class RouteMode(StrEnum): RETRIEVAL_ONLY | TOOLS_ONLY | BOTH | NEITHER
@dataclass(frozen=True) class RouteDecision
    mode / reason / max_iterations / retrieval_confidence
    tool_hints / candidate_tools
    @property use_tools / use_retrieval ; def as_metadata() -> dict
def decide_route(transformed, *, tools=(), retrieval=None, allow_tools=True,
                 settings=None, config=None) -> RouteDecision

def argument_signature(tool_name, arguments) -> str
class LoopVerdict(StrEnum): ALLOW | REPEAT | BLOCKED | EXHAUSTED
@dataclass class LoopGuard
    max_iterations / repeat_limit / iterations / seen / broken_reason
    @classmethod from_config(config, *, max_iterations=None)
    def begin_iteration() -> bool ; def check(tool_name, arguments) -> LoopVerdict

@dataclass class ToolContext
    principal / registry / settings / config
    session / session_id / message_id            # AsyncSession for tool_invocations
    base_filter / retrieve / context_classification
    tracer / detector / exposed / guard
    @classmethod build(principal, *, settings=None, registry=None, **overrides)

@dataclass(frozen=True) class ToolPlan
    tools / connector / exposed ; def request_kwargs() -> dict

class ToolDispatcher
    async plan(ctx) -> ToolPlan
    def screen(ctx, tool, arguments) -> str | None
    async dispatch(ctx, *, tool_call_id, tool_name, arguments) -> ToolResult
```

* `decide_route` picks tools when the transform set `needs_tools`/`tool_hints`, when
  `RetrievalResult.max_score < tool_router_min_confidence`, or when the transform said
  retrieval is unnecessary. `allow_tools=False` and an empty exposed set always win. A
  `degraded` transform's flags are not treated as evidence.
* `LoopGuard.check` returns `REPEAT` on the duplicate that trips `repeat_limit` and
  `BLOCKED` after that; the dispatcher answers both with an `is_error` result telling
  the model to stop repeating, which is what actually breaks the cycle.
* `ToolPlan.request_kwargs()` is what the orchestrator splats into the model call: it
  concatenates client-side definitions with the connector's `mcp_toolset` entries and
  adds `mcp_servers` + `betas` only when a remote server applies.
* **`ToolDispatcher.dispatch` is the only path to an executed tool.** Order: exposure
  check (tenant + role) → loop guard → per-tenant rate limit → egress screen
  (classification ceiling, then PII in arguments unless `allow_pii_in_arguments`) →
  execute → Langfuse span named `tool.<name>` carrying **redacted** arguments →
  `observe_tool_invocation` → `write_tool_invocation`. Every failure path returns
  `ToolResult(is_error=True)`; a failed audit write is logged and never fails the turn.
* The orchestrator must set `ToolContext.context_classification` to the highest
  `Classification` present in the assembled context. It defaults to `PUBLIC`, which is
  permissive — the gate is only as good as that value.

---

# Addendum R — query transformation, retrieval and citations (owner: `services/api/app/rag/{query_transform,retriever,mmr,citations}.py`)

Pipeline stages 3, 5 and 11, plus the MMR primitive stage 5 uses. Requirements #10,
#6 and the citation half of #9. Nothing here changes an existing signature; it pins
the surface the orchestrator, the semantic cache, the guardrails and the eval harness
bind to.

## `app.rag` — the package itself

```python
RAG_SETTING_DEFAULTS: dict[str, Any]
def rag_setting(settings: Settings, name: str) -> Any   # KeyError on an unknown name
```

Same pattern as `guardrail_setting` and `app.rag.memory.optional_setting`: the
tunables below are read through `rag_setting`, which prefers the real
`ragcore.settings` field and falls back to the documented default until that field
exists. Keys are real `RAG_`-prefixed setting names, so
`RAG_CITATION_FUZZY_THRESHOLD=0.7` starts working the moment the field is declared.
Nothing in these modules reads a threshold, limit or model name from anywhere else.

| setting | default | meaning |
|---|---|---|
| `qt_hyde_max_query_words` | `12` | At or below this, a query is "sparse" and HyDE may fire. |
| `retrieval_fusion_score_scale` | `0.05` | Divisor mapping a fusion score onto the `[0,1]` relevance scale. RRF reference; set near `2.0` for `dbsf`. |
| `retrieval_query_concurrency` | `4` | Probes embedded and searched in parallel. |
| `retrieval_mmr_embed_fallback` | `True` | Re-embed a candidate locally when Qdrant did not return its vector. |
| `citation_fuzzy_threshold` | `0.55` | Similarity a paraphrase must reach against its best window. |
| `citation_quote_min_ratio` | `0.9` | Similarity an explicitly quoted literal must reach. |
| `citation_token_recall_threshold` | `0.6` | Content-token recall that may stand in for character similarity. |
| `citation_number_check` | `True` | Every multi-digit number a cited sentence asserts must occur in the cited chunk. |
| `citation_min_span_chars` | `12` | Below this a span borrows the preceding sentence. |
| `citation_max_span_chars` | `600` | Cap on the span compared against a chunk. |
| `citation_min_claim_words` | `5` | Sentence length at which a sentence counts as a factual claim. |
| `citation_min_coverage` | `0.5` | Claim coverage separating `grounded` from `partial`. |
| `citation_anchor_tokens` | `3` | Rare tokens used to anchor the fuzzy window search. |
| `citation_max_windows` | `24` | Upper bound on windows scored per span. |
| `citation_strip_unresolved_markers` | `True` | Remove `[n]` markers no citation survived for. |

Public symbols are re-exported lazily from `app.rag`; `import app.rag` pulls neither
`qdrant-client` nor `anthropic`. Sibling packages (`guardrails`, `memory`, `tools`)
and `orchestrator`/`context` are **not** re-exported here — import them directly.

## `app.rag.query_transform` (stage 3, requirement #10)

```python
class QueryTransformPayload(BaseModel)   # flat wire schema for the structured call
class TransformedQuery(BaseModel):
    intent: str; needs_retrieval: bool; needs_tools: bool; tool_hints: list[str]
    rewritten: str; sub_questions: list[str]; hyde_passage: str
    metadata_filter: MetadataFilter | None; is_out_of_domain: bool; confidence: float
    degraded: bool; degraded_reason: str
    @property queries -> list[str]       # rewritten first, then sub-questions

async def transform_query(message, *, history=(), settings=None, llm=None,
                          now=None) -> TransformedQuery
def fallback_transform(message, *, reason, settings=None) -> TransformedQuery
def is_abstract_query(text, *, settings=None) -> bool
def merge_filters(base, extracted) -> MetadataFilter | None
```

* **`transform_query` never raises.** Every failure — disabled, empty query, refusal,
  malformed response, unreachable API — returns `fallback_transform` with
  `degraded=True` and `degraded_reason ∈ {"disabled", "empty_query", "llm_error",
  "invalid_response"}`. A degraded plan keeps `needs_retrieval=True`; stage 6 must not
  read a degraded plan's `is_out_of_domain` as evidence.
* Exactly **one** `MODEL_FAST` structured call. `history` accepts `Message` objects,
  `(role, content)` pairs or mappings, oldest first, trimmed to `qt_history_turns`,
  and is what pronoun and ellipsis resolution runs against.
* `metadata_filter` holds **only** what the transformer extracted; it is never
  pre-merged. The orchestrator calls `merge_filters(user_filter, plan.metadata_filter)`,
  which is **fill-in, not intersection** — `MetadataFilter.merged_with` would intersect
  `["standard"]` with `["policy"]` to `[]`, which the validator normalises to *no
  constraint*, silently widening the search the user narrowed. Classification and
  `exclude_pii` still combine strictly.
* `doc_type` and `source_type` facets are vocabulary-checked against
  `guardrail_doc_type_authority ∪ {"document"}` and `SourceType`; an unknown value is
  dropped, because a filter nothing can match hides the answer.

## `app.rag.retriever` (stage 5, requirement #6)

```python
async def retrieve(principal, queries, filters=None, *, top_n=None, hyde_passage="",
                   rerank_query=None, include_deleted=None, settings=None,
                   client=None, embedder=None, reranker=None,
                   collection=None) -> RetrievalResult

async def retrieve_by_ids(principal, chunk_ids, filters=None, *, queries=(),
                          top_n=None, include_deleted=None, settings=None,
                          client=None, collection=None) -> RetrievalResult
```

Order: embed each probe once → `hybrid_search` per probe with the single
`build_acl_filter` result → union keeping the best score per chunk → `dedupe_chunks` →
cross-encoder rerank → MMR → per-document cap → `retrieval_top_n`.

* **`queries_used` lists every probe actually issued**, including the HyDE passage
  when one was supplied. Probes are capped at `qt_max_subqueries + 1` plus the HyDE
  passage, de-duplicated case-insensitively.
* **Drop-reason vocabulary** (module constants, all appearing in
  `RetrievalResult.dropped[*].dropped_reason`): `"duplicate:sha256"` and
  `"duplicate:simhash:<distance>"` from `ragcore.dedupe`, plus `DROP_CANDIDATE_LIMIT`
  = `"rerank:candidate_limit"`, `DROP_RERANK_MIN_SCORE` = `"rerank:min_score"`,
  `DROP_RERANK_TOP_N` = `"rerank:top_n"`, `DROP_MAX_PER_DOCUMENT` =
  `"max_per_document"`, `DROP_TOP_N` = `"top_n"`, `DROP_DELETED` = `"deleted"`.
* **`DROP_ACL` = `"acl"` is never returned.** A candidate that fails the in-process
  ACL mirror (`AccessControl.permits`) or whose `tenant_id` disagrees with the
  principal's is counted and logged at **error** level and then discarded — it is not
  placed in `dropped`, because `RetrievalResult.without_text()` is serialised to the
  client and would leak the title, URI and section path of a document the principal
  may not see. Reaching that branch means the Qdrant filter is broken.
* **`final_score` is always in `[0, 1]`**, which is the scale `guardrail_ood_min_score`
  and `MetricScores` are expressed in. With a real cross-encoder it is
  `sigmoid(rerank_score)` — `bge-reranker-v2-m3` is trained with binary
  cross-entropy, so the logistic transform is the model's own relevance estimate.
  Otherwise (rerank disabled, or `NoopReranker`, whose scores encode input order only)
  it is `min(1, fusion_score / retrieval_fusion_score_scale)`. `rerank_score` keeps the
  raw value and is comparable only within one call.
* `dense_score` / `sparse_score` stay `None` on the fused path: fusion happens
  server-side, so the per-branch scores never leave Qdrant.
* `latency_ms` buckets: `embed`, `search`, `dedupe`, `rerank`, `vectors`, `mmr`,
  `total` (a bucket is absent when its stage did not run). The same numbers go to
  `observe_retrieval_stage`.
* MMR reads candidate vectors back from Qdrant through
  `build_acl_filter_for_chunk_ids` — even that bookkeeping read is tenant-scoped —
  and re-embeds anything the read missed when `retrieval_mmr_embed_fallback` is set.
  A vector it cannot obtain contributes no redundancy signal rather than failing.
* One failing probe degrades the search and is logged; **all** probes failing raises
  `RetrievalError` (503). A rerank failure degrades to fusion order.
* **`retrieve_by_ids` is the stage 4 cache-hit path.** It re-fetches through
  `build_acl_filter_for_chunk_ids`, returns `cache_hit=True`, restores the cached
  ranking order, and scores rank-derived `final_score = (n - rank) / n`.
  `total_candidates` is the number of ids asked for, so `total_candidates -
  len(chunks)` is exactly what the live ACL filter removed. An empty `chunk_ids`
  issues no query.

## `app.rag.mmr`

```python
@dataclass(frozen=True) class MMRSelection: order; kept; dropped; scores; lambda_mult
def maximal_marginal_relevance(vectors, relevance, *, top_n=None, lambda_mult=None,
                               settings=None) -> MMRSelection
def mmr_select(...) -> list[int]
def normalise_scores(scores) -> list[float]
```

Returns a **full ordering**, never a truncated one: the retriever needs the losers in
rank order to attribute a drop reason to each. Relevance is min-max normalised inside
the pool because cross-encoder logits and RRF scores are on incomparable scales.
`lambda_mult` defaults to `retrieval_mmr_lambda`.

## `app.rag.citations` (stage 11, requirement #9)

```python
class CitationVerdict(StrEnum): GROUNDED|PARTIAL|UNGROUNDED|NOT_APPLICABLE
@dataclass(frozen=True) class MarkerRef:  number; marker; start; end
@dataclass(frozen=True) class SpanMatch:  text; start; end; ratio; exact
@dataclass(frozen=True) class SourceBlock:
    text; snippets; chunks; def chunk_for(n) -> RetrievedChunk | None; @property size
class CitationDrop(BaseModel):
    marker; number; chunk_id; document_id; reason; span_chars; best_ratio
class CitationReport(BaseModel):
    citations: list[Citation]; dropped: list[CitationDrop]; cleaned_answer: str
    markers_attempted; markers_verified; unknown_markers: list[int]
    claim_sentences; cited_sentences
    citation_validity; coverage; groundedness; verdict; needs_uncertainty_notice
    @property cited_chunk_ids -> list[str]

def build_source_block(chunks, *, settings=None) -> SourceBlock
def parse_markers(answer) -> list[MarkerRef]
def verify_span(span, chunk_text, *, threshold=None, verbatim=False,
                check_numbers=None, settings=None) -> SpanMatch | None
def extract_citations(answer, sources, *, settings=None) -> CitationReport
def strip_unresolved_markers(answer, keep) -> str
def append_uncertainty_notice(answer, report, *, settings=None) -> str
def format_marker(n) -> str                      # "[n]"
DROP_UNKNOWN_MARKER | DROP_EMPTY_SPAN | DROP_SPAN_NOT_FOUND | DROP_QUOTE_NOT_FOUND
```

* **Markers are positional**: source `n` is `chunks[n - 1]`, rendered through
  `ragcore.llm.prompts.render_numbered_sources`. Snippet text is the chunk's own
  words clipped to `retrieval_snippet_chars`; verification always runs against the
  **full** `ChunkPayload.text`, so clipping can only make the verifier more permissive.
* **`quoted_span` always comes from the chunk**, with `char_start`/`char_end` indexing
  `ChunkPayload.text` — `chunk.text[char_start:char_end] == quoted_span` holds for
  every surviving citation. `confidence` is the match ratio (1.0 for an exact
  normalised match).
* Verification is: exact containment on NFKC + casefolded + punctuation-collapsed
  forms; then a bounded fuzzy window search scored on the better of character
  similarity and content-token recall; then a numeric guard — every multi-digit number
  the span asserts must occur in the chunk. An explicitly quoted literal takes the
  `verbatim` path (`citation_quote_min_ratio`, no token recall), so a fabricated quote
  is caught. The numeric guard is skipped automatically when one sentence cites several
  sources.
* **`CitationDrop` carries no content.** Stage 11 runs before the stage 12 PII egress
  scan, so the failing span is deliberately not recorded, logged or traced.
* `citation_validity` = verified marker occurrences ÷ attempted. With no markers at
  all it is `0.0` when the answer contains claim sentences and `1.0` when it does not
  (a refusal), so the contract's gate — `citation_validity <
  guardrail_min_groundedness` → append `UNCERTAINTY_NOTICE` — fires on a confidently
  uncited answer. That gate's outcome is `needs_uncertainty_notice`.
* `coverage` = claim sentences carrying a verified citation ÷ claim sentences;
  `groundedness` = `min(citation_validity, coverage)`. `verdict` is
  `NOT_APPLICABLE` (no claims, no markers) → `UNGROUNDED` (validity below the gate) →
  `PARTIAL` (coverage below `citation_min_coverage`) → `GROUNDED`.
* One `Citation` per surviving marker (the highest-confidence instance wins);
  `markers_attempted`/`markers_verified` count every occurrence. `cleaned_answer`
  strips markers no citation survived for; using it is the orchestrator's decision.


---

# Addendum M — context management and memory (owner: `services/api/app/rag/context.py`, `services/api/app/rag/memory/**`)

Requirements #2, #3 and #5; pipeline stages 2, 4, 9 and 13. Everything the earlier
sections say about these modules is implemented; this addendum pins the surface
downstream code binds to and the decisions those sections left open. **Nothing here
changes an existing signature.**

## `ragcore.settings` — fields this layer needs

These are read through `app.rag.memory.optional_setting(settings, name)`, which prefers
a real `Settings` field and otherwise falls back to
`app.rag.memory.EXTRA_SETTING_DEFAULTS`. The moment the settings owner declares the
`RAG_`-prefixed field, the field wins and the table stops being consulted — so no
threshold, limit or key prefix is ever a literal at a call site.

| field | default | meaning |
|---|---|---|
| `context_compact_every_n_turns` | `6` | **Periodic** suppression (requirement #5): compaction fires every N appended turns even when the budget is nowhere near full. |
| `context_token_count_concurrency` | `8` | Parallel `count_tokens` measurements. |
| `context_token_cache_entries` | `4096` | Bound on the token-count memo (LRU, keyed by content digest, never by content). |
| `context_fit_max_passes` | `3` | Exact-measurement shed passes before the prompt is stripped to system + question. |
| `context_min_retrieved_chunks` | `1` | Chunks kept even under extreme pressure, when any were retrieved. |
| `context_duplicate_penalty` | `0.35` | Marginal-value multiplier for a simhash near-duplicate chunk; makes it the first thing shed. |
| `context_cache_history_breakpoint` | `true` | Place a second `cache_control` breakpoint at the end of the stable history prefix. |
| `context_cache_history_min_tokens` | `1024` | Minimum measured history tokens before that breakpoint is worth writing (the API minimum cacheable prefix on `claude-opus-5` is 512). |
| `redis_session_prefix` | `"rag:session:"` | Redis key prefix for the session window. Keys are `<prefix><tenant_id>:<session_id>` — tenant first. |
| `memory_cache_max_entries` | `500` | Semantic-cache entries kept per `(tenant_id, filter_fingerprint)` bucket before LRU-ish eviction. |
| `memory_consolidate_batch_size` | `200` | Memories scanned per consolidation pass, per user. |
| `memory_recall_oversample` | `4` | Candidate oversampling factor for salience-weighted recall. |
| `memory_recall_similarity_weight` | `0.7` | Blend of dense similarity vs decayed salience when ranking recall candidates. |
| `memory_profile_min_memories` | `3` | Memories required before the rolling profile is regenerated. |

## `app.rag.memory` (package)

```python
EXTRA_SETTING_DEFAULTS: dict[str, Any]
def optional_setting(settings, name) -> Any
def render_memory_block(memories, *, max_items=None) -> str   # "- (kind) text" lines
def render_preferences(profile) -> str
```

Submodules are re-exported lazily through a module-level `__getattr__`, so importing
`app.rag.memory` pulls neither `qdrant-client` nor SQLAlchemy.

## `app.rag.memory.short_term` — stages 2 and 9

```python
class TokenCounter:                       # Claude's tokenizer, memoised by digest
    def __init__(self, llm=None, *, settings=None, model=None)
    @property calls -> int ; @property model -> str
    def peek(text, *, role="user") -> int | None
    async def count_text(text, *, role="user") -> int
    async def count_many(texts, *, role="user") -> list[int]
    async def count_message(message: Message) -> int
    async def count_prompt(*, system, messages, tools=None) -> int

class SessionWindow(BaseModel):           # live turns only; suppressed ones leave it
    tenant_id; user_id; session_id
    turns: list[Message]; rolling_summary: str; summary_tokens: int
    compaction_events: int; turns_since_compaction: int; suppressed_count: int
    updated_at: datetime
    @classmethod empty(*, tenant_id, user_id, session_id) -> Self
    @property live_turns / pinned_turns / total_tokens
    def recent(count) / history_pairs(count=None) / append(message) / replace(message)
    def pin(message_id, *, pinned=True) -> bool
    def suppressible(*, keep_live) -> list[Message]
    def suppress(message_ids, *, summary=None, summary_tokens=None) -> list[Message]
    def trim(max_turns) -> list[Message]        # only when compaction is disabled
    def to_payload() / from_payload(payload)

class SessionStore(Protocol)              # load / save / delete / close
class InMemorySessionStore ; class RedisSessionStore   # `.probe()`, `.available`
class CompactionOutcome(BaseModel)        # retired, summary, summary_tokens, reason

class ShortTermMemory:
    def __init__(self, *, settings=None, store=None, counter=None, llm=None)
    @property settings / counter / store
    async def load(*, principal, session_id, db_session=None) -> SessionWindow
    async def save(window) -> None
    async def clear(*, principal, session_id) -> None
    async def record_turn(window, *, role, content, pii_redacted, message_id,
                          pinned=False, citations=None, tool_calls=None,
                          created_at=None, persist=True) -> Message
    async def pin(window, message_id, *, pinned=True) -> bool
    def select_for_suppression(window, *, keep_live=None) -> list[Message]
    async def compact(window, *, keep_live=None, reason="periodic",
                      message_ids=None, persist=True) -> CompactionOutcome
    async def persist_compaction(window, outcome, *, db_session) -> int
    async def aclose() -> None

async def summarise_turns(turns, *, current_summary="", llm=None, settings=None,
                          max_tokens=None) -> str
def get_short_term_memory(settings=None) -> ShortTermMemory
def reset_short_term_memory() -> None
```

Behaviours callers must know:

1. **`record_turn` raises `ValueError` unless `pii_redacted=True`**, exactly as
   `repositories.append_message` does. The session store is persistence.
2. **The store key is `<prefix><tenant_id>:<session_id>`** and a cached window whose
   `tenant_id` disagrees with the principal is discarded and logged, never returned.
3. **Redis is optional.** A missing `redis` package or a failed call degrades to
   `InMemorySessionStore`/"miss"; PostgreSQL re-hydration is authoritative and the
   choice is logged once at startup.
4. **`summarise_turns` never fails a turn.** A refusal, an unreachable API or a missing
   key falls back to a deterministic extractive summary so suppression still produces
   *something* that stands in for the retired turns.
5. `message_id` is supplied by the caller so the window and the `chat_messages` row
   share an id; `persist_compaction` mirrors suppression through
   `repositories.suppress_messages`, which itself refuses to suppress a pinned turn.

## `app.rag.context` — stage 9

```python
DROP_DUPLICATE_EXACT = "context:duplicate:sha256"
DROP_DUPLICATE_NEAR  = "context:duplicate:simhash"
DROP_RETRIEVED_CAP   = "context:retrieved_cap"
DROP_BUDGET          = "context:budget"
GAP_NOTICE: str      # constant user turn inserted when history starts on an assistant

@dataclass(frozen=True, slots=True)
class ContextBudget:
    total_tokens; reserve_output_tokens; prompt_tokens; compact_at_tokens
    retrieved_tokens; memory_tokens; summary_tokens
    min_live_turns; max_history_turns; compact_every_n_turns
    tool_result_ttl_turns; min_retrieved_chunks
    @classmethod from_settings(settings) -> ContextBudget
    def headroom(used) -> int

@dataclass(slots=True)
class PackedSource: chunk; text; tokens; novelty; marker
    @property value -> float      # final_score * novelty
    @property density -> float    # value / tokens — what packing optimises

@dataclass(slots=True)
class AssembledContext:
    system: list[dict]; messages: list[dict]; stats: ContextStats
    sources: list[RetrievedChunk]      # kept, in citation-marker order
    dropped: list[RetrievedChunk]      # every drop carries a dropped_reason
    memories: list[LongTermMemory]; rolling_summary: str
    suppressed_message_ids: list[str]; context_management: dict | None
    cache_system: bool; compaction_reason: str; user_turn: str
    @property compacted -> bool
    def request_kwargs() -> dict       # system/messages/cache_system[/context_management]

class ContextAssembler:
    def __init__(*, settings=None, llm=None, counter=None, short_term=None)
    @property settings / counter / short_term ; def budget() -> ContextBudget
    async def rank_sources(chunks) -> tuple[list[PackedSource], list[RetrievedChunk]]
    async def assemble(*, principal, window, question, chunks=(), memories=(),
                       profile=None, notes="", system=None, tools=None,
                       force_compaction=False, db_session=None) -> AssembledContext

def get_context_assembler(settings=None) -> ContextAssembler
def reset_context_assemblers() -> None
```

Decisions this module pins:

* **Nothing is estimated.** The system prompt is measured by differencing two exact
  `count_tokens` calls against a constant probe message; each candidate block is
  measured on its own; the finished payload is measured once. `ContextStats.window_tokens`
  is that final exact number. `tiktoken` is never used, and no characters/4 heuristic
  appears at any call site here.
* **Packing order** is the contract's: system → pinned turns → rolling summary →
  long-term memory → retrieved chunks (best marginal value per token first) → recent
  turns. Chunks are rendered in *relevance* order so `[n]` markers stay meaningful,
  even though they are *selected* by density.
* **Shedding order** (after the exact measurement overshoots): near-duplicate/low-density
  chunks above `context_min_retrieved_chunks` → memory lines → the remaining chunks →
  non-pinned history turns → the rolling summary. Each pass sheds until the estimated
  saving covers the overshoot, then re-measures, so the number of `count_tokens` calls
  is bounded by `context_fit_max_passes`. Pinned turns are never shed.
* **Compaction reasons** are `"forced" | "max_turns" | "periodic" | "ratio" | "overflow"`
  (empty when none ran) and always retire down to `context_min_live_turns`. The floor is
  never compacted, so `force_compaction` on a session at or below the floor is a no-op.
* **Tool results**: `ContextStats.tool_results_cleared` counts the tool calls on live
  turns older than `context_tool_result_ttl_turns`, and `context_management` is
  `clear_tool_uses_edit()` whenever that count is non-zero. Replayed history is rendered
  as plain text, so nothing stale is re-sent from the assembler either; the edit is what
  clears the blocks the stage 8 tool loop appends to the same payload.
* **Caching**: the last system block carries `cache_control` (LLMClient re-normalises,
  so this is belt-and-braces and makes the payload self-describing); a second breakpoint
  goes on the last *stable* history message when that prefix is worth it. Volatile
  content only ever appears in the final user turn, via
  `prompts.build_answer_user_turn`.
* **Message sequence repair**: suppression can leave a history that starts on an
  assistant turn or contains two consecutive turns of the same role, both of which the
  Messages API rejects. Consecutive same-role turns are merged and a leading assistant
  turn is preceded by the constant `GAP_NOTICE` — nothing is discarded to make the
  sequence valid.
* `ContextStats.messages_live` is the number of messages actually sent;
  `messages_suppressed` is the session's cumulative suppressed count.
  `cache_read_tokens`/`cache_write_tokens` are left at 0 for the caller to fill in from
  `response.usage` after the generation call.

## `app.rag.memory.long_term` — stages 2 and 13

```python
class ExtractedMemory(BaseModel)   # kind, text, salience, supersedes_text (flat wire)
class MemoryExtraction(BaseModel)  # memories: list[ExtractedMemory]
class MemoryRecall(BaseModel):
    memories; scores: dict[str, float]; profile; consent: bool
    candidates: int; latency_ms: float; reason: str
    @property has_memories -> bool

class LongTermMemoryStore:
    def __init__(*, settings=None, client=None, embedder=None, llm=None, detector=None)
    @property settings / collection ; async def qdrant() -> AsyncQdrantClient
    async def resolve_profile(principal, *, profile=None, db_session=None) -> UserProfile | None
    def consent_gate(profile) -> str      # "ok" | "disabled" | "consent_unknown" | "no_consent"
    async def recall(principal, query, *, profile=None, top_k=None, kinds=None,
                     db_session=None, now=None, touch=True) -> MemoryRecall
    async def touch(principal, memories, *, db_session=None, now=None) -> int
    async def set_salience(principal, memory, value, *, db_session=None) -> bool
    async def remember(memory, *, principal, db_session=None) -> LongTermMemory
    async def write_back(*, principal, session_id, user_text, assistant_text="",
                         profile=None, db_session=None, now=None) -> list[LongTermMemory]
    async def scan(principal, *, limit=None, include_expired=False,
                   with_vectors=False, now=None) -> list[tuple[LongTermMemory, list[float]]]
    async def forget(principal, memory_id, *, db_session=None) -> bool
    async def expire(principal, *, now=None, db_session=None) -> int
    async def prune(principal, *, db_session=None, now=None) -> int

def get_long_term_memory(settings=None) -> LongTermMemoryStore
def reset_long_term_memory() -> None
```

* **The consent gate is hard and fails closed.** `consent_gate` returns
  `"consent_unknown"` when no profile could be resolved (no `profile=` and no
  `db_session=`), and *nothing* is read or written in that case. Defaulting the other
  way would let a database hiccup silently re-enable memory for someone who switched it
  off. `recall` and `write_back` both report the blocking reason, so "nothing stored"
  and "not allowed to look" are distinguishable.
* **`remember` is the only write path**, and it PII-scans and redacts before embedding
  or upserting, so `pii_redacted=True` on a stored memory is an assertion the pass ran.
  Write-back additionally redacts the turn text it sends to `MODEL_FAST`.
* **Recall ranking** is `w * cosine + (1 - w) * decayed_salience` with
  `w = memory_recall_similarity_weight`, after dropping anything below
  `memory_min_salience` or past `expires_at`. Selected memories are touched
  (`hit_count`, `last_used_at`, `+memory_salience_boost_on_hit`), best-effort.
* **Deletes are ownership-checked.** Qdrant deletes by point id, which carries no
  tenant, so the point is retrieved and its payload's `tenant_id`/`user_id` verified
  before removal. `remember` applies `supersedes` by deleting the superseded point, so
  an update is one write rather than a write plus an orphan.
* Growth is bounded by `supersedes`, `expire()` (TTL from `memory_ttl_days`) and
  `prune()` (`memory_max_per_user`, lowest decayed salience first).

## `app.rag.memory.semantic_cache` — stage 4

```python
CACHE_LABEL = "semantic"        # the `cache=` label on rag_cache_lookups_total

ChunkResolver = Callable[[Principal, Sequence[str], MetadataFilter | None],
                         Awaitable[Sequence[RetrievedChunk]]]

async def retrieve_by_ids(client, *, principal, chunk_ids, extra=None,
                          settings=None) -> list[RetrievedChunk]

class CacheProbe(BaseModel):
    hit: bool; similarity: float; entry: SemanticCacheEntry | None
    chunks: list[RetrievedChunk]; revoked_chunk_ids: list[str]
    transformed_queries: list[str]; fingerprint: str; normalized_query: str
    latency_ms: float; reason: str
    @property usable -> bool     # hit AND at least one chunk survived the ACL re-fetch

class SemanticCache:
    def __init__(*, settings=None, client=None, embedder=None, resolver=None)
    @property settings / collection
    def fingerprint(principal, extra) -> str
    async def probe(principal, query, *, extra=None, now=None,
                    db_session=None, resolver=None) -> CacheProbe
    async def store(principal, query, *, chunk_ids, transformed_queries=(),
                    extra=None, now=None, db_session=None) -> SemanticCacheEntry | None
    async def evict(principal, *, fingerprint, now=None) -> int
    async def invalidate(principal, *, fingerprint) -> int

def get_semantic_cache(settings=None) -> SemanticCache
def reset_semantic_cache() -> None
```

* Cached ids become chunks **only** through `build_acl_filter_for_chunk_ids` — the
  full ACL filter plus a `HasIdCondition`. Chunks come back in the cached order with
  descending `final_score` and `retrieval_stage="cache"`; the ids the filter withheld
  are reported in `CacheProbe.revoked_chunk_ids`.
* **`ChunkResolver` is the seam to `app.rag.retriever.retrieve_by_ids`.** Addendum R
  owns the richer stage-4 re-fetch, which returns a full `RetrievalResult` with the
  retriever's own score scale and drop vocabulary; the local `retrieve_by_ids` here is
  the client-level primitive and the default, so this module imports neither the
  retriever nor FastEmbed. The orchestrator should inject the retriever's version:

  ```python
  async def _resolve(p, ids, filters):
      return (await retriever.retrieve_by_ids(p, ids, filters)).chunks


  probe = await cache.probe(principal, question, extra=filters, resolver=_resolve)
  ```

  Both paths compose the same ACL filter, so the security property does not depend on
  which one runs. A resolver that raises degrades to an empty result, which reads
  downstream as a miss.
* **A hit with zero surviving chunks is reported as a miss** (`reason="empty_after_acl"`,
  `usable=False`) so the pipeline retrieves normally instead of answering from an empty
  plan.
* `probe` reasons: `"hit" | "miss" | "disabled" | "empty_query" | "below_threshold" |
  "expired" | "fingerprint_mismatch" | "empty_after_acl" | "error"`. Every outcome also
  records `observe_cache_lookup(cache="semantic", hit=…)`.
* **Only the plan is cached** — `normalized_query`, `transformed_queries`, `chunk_ids`,
  `filter_fingerprint`, counters. Never the answer, and never chunk text. An answer
  cannot be re-authorised; a chunk id can.
* **A clearance change misses, a group change hits.** `filter_fingerprint` includes the
  clearance rank but deliberately not group or role membership, so losing a group still
  matches the entry and is caught by the live ACL re-fetch — which is precisely the
  path that must be exercised, and is.
* Eviction is per `(tenant_id, filter_fingerprint)` bucket, enumerated with
  `build_cache_filter`: expired entries first, then the excess over
  `memory_cache_max_entries` ordered by `hit_count` then `last_used_at`. It runs after
  every `store`.

## `app.rag.memory.consolidate` — the periodic maintenance job

```python
class ProfileDraft(BaseModel)        # summary, preferred_style, preferred_language, top_topics
class ConsolidationReport(BaseModel):
    tenant_id; user_id; scanned; merged; decayed; expired; pruned
    profile_refreshed: bool; skipped_reason: str; duration_ms: float
    errors: list[str]                # exception type names only, never messages
    @property changed -> bool

class MemoryConsolidator:
    def __init__(*, settings=None, store=None, llm=None)
    async def consolidate_user(*, tenant_id, user_id, db_session=None, profile=None,
                               now=None, refresh_profile=True) -> ConsolidationReport
    async def consolidate_users(*, tenant_id, user_ids, db_session=None,
                                now=None) -> list[ConsolidationReport]
    async def refresh_profile(profile, memories, *, db_session=None) -> bool
```

Merges near-duplicates (cosine over the *stored* vectors, so no re-embedding), persists
decayed salience, expires and prunes, and regenerates the rolling profile with one
`MODEL_FAST` structured call. It honours the same consent gate, never raises into a
caller (failures are counted in `errors`), and is safe to run repeatedly. Call it from a
scheduled task or an admin endpoint; `refresh_profile` only persists when a
`db_session` is supplied.

## Notes for neighbouring components

* **The orchestrator owns the ordering.** `assemble()` mutates the window in place and
  returns `suppressed_message_ids`; pass a `db_session` to have suppression mirrored
  into `chat_messages` during assembly, or persist it yourself afterwards.
* `ContextStats.cache_read_tokens` / `cache_write_tokens` are the caller's to fill from
  `LLMResponse.usage` before the `context_stats` SSE event is emitted.
* `services/api/tests/` deliberately has no `__init__.py`, for the reason Addendum S
  gives.

---

# Addendum G — guardrails (owner: `services/api/app/rag/guardrails/**`)

Requirement #9's edge cases: pipeline stages **1** (input guard), **5/6** (indirect
injection and the out-of-domain gate), **7** (contradictions) and **12** (output
guard). Nothing here changes an existing signature; it pins the surface the
orchestrator, the eval harness and the SSE layer bind to.

## `app.rag.guardrails` — the package itself

```python
GUARDRAIL_DEFAULTS: dict[str, Any]
GUARDRAIL_PROMPT_VERSION: str          # version of this package's own prompts
def guardrail_setting(settings: Settings, name: str) -> Any   # KeyError if unknown
```

Same pattern as `rag_setting` and `app.rag.memory.optional_setting`: every tunable
below is read through `guardrail_setting`, which prefers the real `ragcore.settings`
field and falls back to the documented default until that field is declared. Keys are
real `RAG_`-prefixed names, so `RAG_GUARDRAIL_OOD_COLLAPSE_SPREAD=0.08` starts working
the moment the field exists. **No threshold, limit, entity list or model name is
written at a call site in this package.**

Submodules are re-exported lazily through a module-level `__getattr__`, so
`import app.rag.guardrails` costs nothing and submodules can import
`guardrail_setting` from the package without a cycle.

| setting | default | meaning |
|---|---|---|
| `guardrail_input_min_chars` | `1` | Below this the turn is refused with `clarify`. |
| `guardrail_input_truncate` | `False` | Truncate rather than refuse an oversized turn. |
| `guardrail_input_normalise_unicode` | `True` | NFKC + invisible-character strip before scanning. |
| `guardrail_input_credential_entities` | `["API_KEY", "JWT"]` | Entity types masked out of the **prompt** text as well as the persisted text. |
| `guardrail_language_detect_enabled` | `True` | Run the dependency-free language guess. |
| `guardrail_language_min_confidence` | `0.35` | Below this the language policy is not enforced. |
| `guardrail_allowed_languages` | `[]` | Empty = all. A detected language outside the list warns; it never blocks. |
| `guardrail_injection_scan_retrieved` | `True` | Scan retrieved chunk text for indirect injection. |
| `guardrail_injection_retrieved_block_threshold` | `0.5` | Quarantine threshold for **retrieved** text (stricter than the user turn's `0.8`). |
| `guardrail_injection_quarantine_retrieved` | `True` | Drop a flagged chunk; `False` keeps it and only warns. |
| `guardrail_injection_max_scan_chars` | `20000` | Cap on characters inspected per passage. |
| `guardrail_injection_classifier_enabled` | `False` | Optional `MODEL_CHEAP` adjudication. |
| `guardrail_injection_classifier_benign_factor` | `0.5` | Score multiplier when the classifier says `benign`. |
| `guardrail_ood_mean_score_min` | `0.2` | Mean top-k relevance floor. |
| `guardrail_ood_min_candidates` | `1` | Fewer candidates than this is automatically out of domain. |
| `guardrail_ood_collapse_enabled` | `True` | Enable score-distribution collapse detection. |
| `guardrail_ood_collapse_top_k` | `5` | Window the collapse test looks at. |
| `guardrail_ood_collapse_spread` | `0.05` | Max−min within the window at or below which the distribution is "collapsed". |
| `guardrail_ood_collapse_max_score` | `0.55` | Collapse only counts when the best score is at or below this. |
| `guardrail_ood_classifier_enabled` | `False` | Optional `MODEL_CHEAP` adjudication when the signals disagree. |
| `guardrail_ood_coverage_sample` | `512` | Chunks scrolled to build the coverage summary. |
| `guardrail_ood_coverage_ttl_seconds` | `900` | Coverage cache TTL. |
| `guardrail_ood_coverage_max_items` | `8` | Facet values listed per refusal. |
| `guardrail_ood_llm_refusal` | `False` | Phrase the refusal with `MODEL_FAST` instead of the deterministic template. |
| `guardrail_contradiction_min_chunks` | `2` | Below this stage 7 is skipped. |
| `guardrail_contradiction_max_clusters` | `6` | Claim clusters kept. |
| `guardrail_contradiction_max_pairs` | `4` | Cross-document pairs adjudicated per turn — the model-call budget. |
| `guardrail_contradiction_llm_enabled` | `True` | Use the structured `MODEL_MAIN` detector. |
| `guardrail_contradiction_snippet_chars` | `600` | Passage length sent to the adjudicator. |
| `guardrail_output_block_below_groundedness` | `0.2` | Hard floor: below it the answer is withheld, not annotated. |
| `guardrail_output_leak_span_chars` | `60` | Verbatim overlap treated as an over-clearance leak. |
| `guardrail_output_pii_ignore_entities` | `["DATE_TIME", "LOCATION"]` | Entity types not redacted on egress: an effective date and an office name are policy content, and masking them would gut every answer. |
| `guardrail_refusal_check_enabled` | `True` | Assess refusal quality. |
| `guardrail_refusal_min_chars` | `40` | Shorter than this, a refusal is reported as bare. |

Vocabulary extensions this package makes (both string fields, both additive):
`GuardrailEvent.kind` gains **`"language"`** for the stage-1 language notice, and
`RetrievedChunk.dropped_reason` gains **`"guardrail:injection"`** for a quarantined
chunk.

## `app.rag.guardrails.injection` (stages 1 and 5)

```python
UNTRUSTED_BLOCK_OPEN  = "<<<BEGIN_UNTRUSTED_CONTENT>>>"
UNTRUSTED_BLOCK_CLOSE = "<<<END_UNTRUSTED_CONTENT>>>"
UNTRUSTED_PREAMBLE: str ; INJECTION_DROP_REASON = "guardrail:injection"
INJECTION_CLASSIFIER_SYSTEM: str ; PATTERNS: tuple[InjectionPattern, ...]

@dataclass(frozen=True) class InjectionPattern: name; category; pattern; weight; description
class InjectionSignal(BaseModel):  name; category; weight; count
class InjectionVerdict(BaseModel):
    score; action; signals; source; scanned_chars; truncated; classifier_label
    @property blocked / flagged / categories / detail
    def to_event(*, stage) -> GuardrailEvent
class RetrievedScan(BaseModel):
    kept; quarantined; verdicts: dict[str, InjectionVerdict]; events
    @property flagged_chunk_ids / quarantined_chunk_ids

def scan_text(text, *, settings=None, source="user_turn",
              block_threshold=None, warn_threshold=None) -> InjectionVerdict
async def scan_user_turn(text, *, settings=None, llm=None) -> InjectionVerdict
async def scan_retrieved(chunks, *, settings=None, llm=None) -> RetrievedScan
def wrap_untrusted(text, *, label="retrieved document content", marker=None,
                   source_uri=None, include_preamble=True) -> str
def sanitise_untrusted(text) -> str
def normalise_unicode(text) -> str
def pattern_by_name(name) -> InjectionPattern | None
```

* **Every piece of retrieved or tool-returned text that enters a prompt is wrapped**
  by `wrap_untrusted`. That is structural defence and does not depend on detection
  working. `sanitise_untrusted` neutralises the delimiters inside the payload, so a
  document cannot close the block early, and strips invisible characters (zero-width,
  bidi controls, the Unicode Tags block) — none of which changes the visible words a
  citation span has to match.
* Scores combine as noisy-OR, `1 − Π(1 − wᵢ)`, over distinct patterns. A pattern at or
  above `guardrail_injection_retrieved_block_threshold` quarantines a chunk alone;
  weaker ones need company. Patterns are anchored to clause boundaries or to
  assistant-directed objects wherever ordinary corpus prose would otherwise trip them
  ("the runbook explains how to disable access controls" must not quarantine the
  runbook).
* **A signal never carries an excerpt of the match** — it is untrusted content and
  signals are logged, traced and streamed before anything redacts them.
* The optional classifier adjudicates the ambiguous band on the user turn; on
  retrieved content the band is widened past the block threshold so it can *attenuate*
  a false positive on security documentation.

## `app.rag.guardrails.input_guard` (stage 1)

```python
EMPTY_INPUT_MESSAGE / OVERSIZE_MESSAGE / INJECTION_BLOCK_MESSAGE: str
LANGUAGE_GUARDRAIL_KIND = "language"

class InputDecision(BaseModel):
    allowed; action; text; redacted_text; refusal
    original_chars; truncated; language; language_confidence
    pii_types; pii_redacted; credential_types; injection; events
    @property blocked
    def raise_if_blocked() -> None          # raises GuardrailBlocked

def detect_language(text, *, default="en") -> tuple[str, float]
async def run_input_guard(message, *, principal, settings=None,
                          detector=None) -> InputDecision
```

**`text` versus `redacted_text` is the whole point of this stage.** `text` is the
prompt-safe turn — the user's own words, normalised and size-capped, with only
credential-shaped entities masked, because redacting "my email is …" out of the
question would change what was asked. `redacted_text` is the only form that may be
logged, traced or persisted, and it is what `repositories.append_message` and
`short_term.record_turn` want alongside `pii_redacted=True`. Order is fixed:
normalise → cap → redact → scan, so every log line this function emits is already
safe.

## `app.rag.guardrails.ood` (stage 6)

```python
class CoverageItem(BaseModel):  value; count
class DomainCoverage(BaseModel):
    tenant_id; doc_types; tags; titles; documents_sampled; chunks_sampled; generated_at
    @property is_empty ; def describe(*, max_items=8) -> str
class RelevanceSignals(BaseModel):
    candidate_count; max_score; mean_score; top_k_spread; score_source
    below_min_score; below_mean_score; collapsed; too_few_candidates
    @property weak / reason
class OODVerdict(BaseModel):
    is_out_of_domain; needs_tool; reason; detail; confidence
    signals; coverage; refusal; classifier_label; events

def relevance_signals(result_or_chunks, *, settings=None) -> RelevanceSignals
async def tenant_coverage(principal, *, settings=None, client=None,
                          extra=None) -> DomainCoverage
def fallback_refusal(coverage, *, question="", needs_tool=False,
                     settings=None) -> str
async def run_ood_gate(*, question, result, principal, transformed=None,
                       tool_available=False, settings=None, llm=None,
                       client=None) -> OODVerdict
def clear_coverage_cache() -> None
```

* Three evidence sources: retrieval signals, `TransformedQuery.is_out_of_domain`
  (**ignored when `degraded`**), and the optional cheap adjudicator, which runs only
  when the first two disagree. `reason` is one of `disabled`, `no_candidates`,
  `low_max_score`, `collapsed_distribution`, `low_mean_score`, `classifier`,
  `classifier_needs_tool`, `tool_can_serve`, `sufficient_evidence`, or a
  `<signal>+transformer_flag` composite.
* **Collapse** is the signal absolute thresholds miss: top-k spread ≤
  `guardrail_ood_collapse_spread` with the best score ≤
  `guardrail_ood_collapse_max_score` and at least three candidates. Healthy retrieval
  has a clear winner; "everything equally mediocre" means the ranker found nothing to
  prefer.
* `tenant_coverage` samples **through `build_acl_filter`**, so a refusal can never
  advertise a document type or title the caller is not cleared for. It is cached per
  (tenant, clearance, roles, groups), and every failure returns an empty coverage and
  logs — a turn must not die while composing an apology. `clear_coverage_cache()`
  after a bulk ingest or reindex.
* A weak-evidence query that a registered tool could serve returns
  `is_out_of_domain=False, needs_tool=True`; the orchestrator routes to stage 8. Pass
  `tool_available` from the role-filtered tool registry.
* On refusal, `OODVerdict.refusal` is a **complete user-facing answer** — it states the
  boundary, names what is indexed, and offers a next step. The orchestrator emits it
  instead of running generation and must not append citations to it.

## `app.rag.guardrails.contradiction` (stage 7)

```python
class ClaimCluster(BaseModel):  cluster_id; chunk_ids; document_ids; key_terms
                                @property size / is_cross_document
class ConflictVerdict(BaseModel):    # the MODEL_MAIN structured schema
    conflicts; subject; statement_a; statement_b; distinguishing_scope; confidence
class ConflictResolution(BaseModel):
    winner; loser; basis; gap_days; superseded; winner_authority; loser_authority
    @property winner_chunk_id / loser_chunk_id
class Contradiction(BaseModel):
    subject; current_chunk_id; superseded_chunk_id
    current_statement; superseded_statement; basis; gap_days; superseded
    confidence; detection; citations: list[Citation]      # both sides, current first
    @property markers
class ContradictionReport(BaseModel):
    checked; clusters; contradictions; pairs_examined; degraded; notes; events
    @property has_conflicts

def cluster_claims(chunks, *, settings=None) -> list[ClaimCluster]
def resolve_conflict(left, right, *, settings=None) -> ConflictResolution
async def check_contradictions(chunks, *, question="", markers=None,
                               settings=None, llm=None) -> ContradictionReport
def render_contradiction_notes(report) -> str
```

* `basis` is the first discriminator that separated the pair, in the contract's order:
  `effective_from` → `source_modified_at` → `authority` → `recency` →
  `indeterminate`. Recency is checked **before** authority on purpose: a superseded
  policy is still a policy, and ranking by type first would let last year's policy
  override this year's standard. `superseded=True` when the gap reaches
  `guardrail_contradiction_recency_days`; otherwise the disagreement is live.
* Candidate pairs come from cross-document claim clusters **plus** sub-threshold pairs
  that disagree about a shared quantity or flip a shared obligation's polarity — the
  conflicts that matter most are usually surrounded by rewritten prose, which is what
  drags similarity below the clustering bar. Two chunks of the same document are never
  a contradiction.
* `detection` is `"llm"` or `"heuristic"`. `degraded=True` means a model call was
  *expected* and failed; a deployment with `guardrail_contradiction_llm_enabled=False`
  or no API key runs the documented deterministic path and is **not** degraded.
* **`markers=` must be the orchestrator's real chunk-id → `[n]` mapping** (positional
  markers from `build_source_block`), or the surfaced note will cite different numbers
  from the answer. Defaults to `[1]`-based positional markers over `chunks`.
* **`notes` is the surfacing mechanism**: pass it as
  `prompts.build_answer_user_turn(..., notes=report.notes)`. It instructs the model to
  give the current position with its citation and add one sentence citing the
  conflicting source. Resolving a conflict without surfacing it is the failure mode
  this stage exists to prevent.

## `app.rag.guardrails.output_guard` (stage 12)

```python
CLEARANCE_BLOCK_MESSAGE / PII_BLOCK_MESSAGE / GROUNDEDNESS_BLOCK_MESSAGE: str

class ClearanceViolation(BaseModel):
    kind; chunk_id; document_id; classification; classification_rank
    principal_rank; detail                     # kind: chunk | citation | leaked_span
class ClearanceReport(BaseModel):
    violations; checked_chunks; checked_citations ; @property ok / leaked
class RefusalQuality(BaseModel):  is_refusal; acceptable; reasons
class OutputDecision(BaseModel):
    text; redacted_text; blocked; action; citations; dropped_citations
    pii_types; pii_redacted; groundedness; groundedness_applicable
    uncertainty_appended; clearance; refusal; events

def citation_validity_score(answer, citations, chunks) -> float
def check_clearance(*, answer, citations, chunks, principal, settings=None) -> ClearanceReport
def assess_refusal(text, *, settings=None) -> RefusalQuality
async def run_output_guard(*, answer, citations, chunks, principal, settings=None,
                           detector=None, groundedness=None) -> OutputDecision
```

* Order: refusal quality → classification → PII egress → groundedness. **A
  classification violation fails closed**: the answer is replaced with
  `CLEARANCE_BLOCK_MESSAGE`, the offending citations are moved to
  `dropped_citations`, and the event is logged at `error` level naming the chunk,
  document and both ranks — if this ever fires, `build_acl_filter` is broken and the
  log line says so. `guardrail_enforce_classification_on_output=False` downgrades the
  action to `warn` and leaves the answer alone; that is an operator's decision, not a
  default.
* `check_clearance` uses `AccessControl.permits` — deliberately a *second*
  implementation of the Qdrant filter's rule, so a bug in one is visible against the
  other — plus a windowed verbatim-span search for `guardrail_output_leak_span_chars`
  of over-clearance text in the answer.
* **Pass `groundedness=CitationReport.citation_validity`** from stage 11. Omitting it
  makes this module recompute a coarser score from the answer's markers, and only in
  that self-computed case does "sources retrieved, claims made, no markers at all"
  force `0.0`. Stage 12 owns the uncertainty notice; appending is **idempotent**, so
  calling `citations.append_uncertainty_notice` first is harmless but redundant.
* The canned messages above are operator-authored and are **not** PII-scanned;
  scanning them could only produce false positives.
* `redacted_text` is the only form that may be persisted, traced or logged. Emit
  `text`; store `redacted_text` with `pii_redacted=True`.

## Orchestrator wiring

1. `run_input_guard(...)` → on `blocked`, stream `decision.events`, emit
   `decision.refusal` and stop. Otherwise prompt with `decision.text` and persist
   `decision.redacted_text`.
5. after retrieval: `scan_retrieved(result.chunks)` → prompt with `scan.kept`, append
   `scan.quarantined` to `RetrievalResult.dropped`, stream `scan.events`.
6. `run_ood_gate(question=plan.rewritten, result=..., principal=..., transformed=plan,
   tool_available=bool(plan_tools))` → on `is_out_of_domain`, emit `verdict.refusal`
   and stop; on `needs_tool`, continue into stage 8.
7. `check_contradictions(kept_chunks, question=plan.rewritten, markers=...)` → pass
   `report.notes` into the answer user turn, stream `report.events`.
12. `run_output_guard(answer=..., citations=report.citations, chunks=kept_chunks,
    principal=..., groundedness=report.citation_validity)` → emit `decision.text`,
    persist `decision.redacted_text`, stream `decision.events`, score
    `decision.groundedness` into Langfuse.

All guardrail events are `GuardrailEvent`s and go out on the `guardrail` SSE event;
each is mirrored into `observe_guardrail` so `rag_guardrail_events_total` counts them
without an extra call site.

---

# Addendum E — evaluation harness (owner: `services/eval`)

Requirement #8, end to end. Nothing here changes an existing signature; it pins the
surface CI, the API and a future golden-set author bind to.

## Package, entry points and layout

`services/eval` **is** the package: `services/eval/__init__.py` makes it importable as
`eval` (the name the repository layout and ruff's `known-first-party` list already
use). Modules: `run_eval`, `metrics`, `ragas_adapter`, `semantic`, `ci_gate`,
`report`, plus `golden/`.

```bash
python -m eval.run_eval --golden services/eval/golden/golden_set.yaml --gate
python -m eval.run            # same module; the spelling ci.yml and `make eval` use
python -m eval.ci_gate services/eval/reports/latest.json   # gate a saved run
rag-eval --category acl_negative --no-gate                 # console script
```

`eval.run` is an import alias, not a second file: `eval/__init__.py` installs a
meta-path finder that resolves `eval.run` to `eval/run_eval.py`. Two importable
copies of one CLI is how the two drift apart.

Exit codes for both CLIs: `0` ok, `1` gate failed, `2` the harness could not run.

## `eval` — package surface

```python
EVAL_SETTING_DEFAULTS: dict[str, Any]
def eval_setting(settings: Settings, name: str) -> Any    # KeyError on a typo
```

Same pattern as `app.rag.rag_setting` and `app.rag.guardrails.guardrail_setting`:
the knobs below are read through `eval_setting`, which prefers the real
`ragcore.settings` field and falls back to the documented default until that field
exists. Keys are real `RAG_`-prefixed names, so `RAG_EVAL_REPORT_DIR=…` works the
moment the field is declared. The `eval_*` fields already on `Settings`
(`eval_enabled`, `eval_golden_path`, `eval_max_concurrency`, `eval_judge_model`,
`eval_gate_enabled`, every `eval_min_*`, `eval_max_acl_leak`, `eval_max_latency_ms`,
`eval_sample_size`) are read directly.

| setting | default | meaning |
|---|---|---|
| `eval_pipeline_target` | `"app.rag.orchestrator:run_turn"` | `module:attribute` of the coroutine one golden item is run through. |
| `eval_personas_path` | `services/eval/golden/personas.yaml` | Persona file. |
| `eval_report_dir` | `services/eval/reports` | JSON + markdown + HTML land here; CI uploads it. |
| `eval_item_timeout_seconds` | `300.0` | Ceiling for one item. |
| `eval_persist_results` | `True` | Write `eval_runs` / `eval_results` rows. |
| `eval_run_tenant_id` | `None` | Tenant on the run row; None means the dominant tenant of the selection. |
| `eval_skip_unregistered_tools` | `True` | Skip a `tool_required` item whose tool is not in the registry. |
| `eval_tool_required_live` | `False` | Run `tool_required` items that name a REST/MCP tool. Off by default; the built-ins always run. |
| `eval_ragas_enabled` | `True` | Use the RAGAS package for the LLM-only metrics when it imports. |
| `eval_judge_effort` | `"medium"` | `output_config.effort` for every judge call. |
| `eval_relevancy_probe_questions` | `3` | Questions generated from the answer for answer relevancy. |
| `eval_correctness_f1_weight` / `eval_correctness_similarity_weight` | `0.75` / `0.25` | Answer-correctness weighting. |
| `eval_judge_max_contexts` / `eval_judge_max_context_chars` | `12` / `4000` | Judge prompt budget. |
| `eval_hard_min_acl_leak` | `1.0` | Hard floor; configuration may tighten, never loosen. |
| `eval_hard_min_refusal_correct` | `0.95` | Hard floor; same rule. |
| `eval_min_tool_correct` | `0.5` | Soft floor. Lenient on purpose — see below. |
| `eval_min_retrieval_recall` | `0.7` | Soft floor for expected-document recall. |
| `eval_regression_tolerance` | `0.02` | Movement below this is judge noise, not a regression. |
| `eval_report_worst_items` | `10` | Items shown in full, with their trace ids. |
| `eval_report_answer_chars` | `600` | Answer preview stored on a result. |
| `eval_score_prefix` | `"eval."` | Prefix for every Langfuse score name. |

## The orchestrator binding

**This is the one thing `services/api` must satisfy.** The harness resolves
`eval_pipeline_target` and calls it once per item. It binds structurally, so the
orchestrator keeps ownership of its own signature:

* the callable must accept a question under one of
  `message | question | query | text | user_message | prompt`, and a
  `ragcore.models.acl.Principal` under one of `principal | user | caller | identity`.
  Anything else it declares out of
  `settings`, `stream` (passed `False`), `allow_tools` (`True`), `filters` (`None`),
  `session_id` (`None`), `persist` (`False`) is filled in; anything it does not
  declare is not passed. A target missing both required parameters raises
  `EvalHarnessError` naming what it does accept.
* the return value (awaited if awaitable) is read structurally by
  `eval.run_eval.coerce_outcome` into `TurnOutcome`:

  | `TurnOutcome` field | read from |
  |---|---|
  | `answer` | `answer` \| `text` \| `content` \| `output` \| `final_answer` \| `message.content` |
  | `chunks` | `chunks` \| `retrieved_chunks` \| `sources` \| `context_chunks` \| `retrieval.chunks` |
  | `citations` | `citations` \| `message.citations` |
  | `tools_invoked` | `tool_calls` \| `tools_invoked` \| `tool_invocations` \| `tools` \| `message.tool_calls`, each entry's `tool_name`/`name` |
  | `usages` | `usages` \| `usage` \| `llm_usages` (`LLMUsage` records) |
  | `cost_usd` | `cost_usd`, else summed from `usages` |
  | `refused` | `refused` \| `is_refusal` |
  | `trace_id` | `trace_id` |
  | `citation_report` | `citation_report` \| `citations_report` (an `app.rag.citations.CitationReport`) |

  A field the orchestrator does not expose degrades one metric; it never breaks the
  run. `answer` must be the **post-stage-12** text — the harness stores a clipped
  preview of it and never sees a pre-redaction answer.
* an in-process runner can be injected instead: `run_eval(runner=...)` takes any
  `PipelineRunner`, i.e. `async def (*, item: GoldenItem, principal: Principal) -> TurnOutcome`.

## `eval.run_eval`

```python
async def run_eval(*, golden_path=None, personas_path=None, categories=None,
                   item_ids=None, limit=None, baseline_path=None, gate=True,
                   report_dir=None, concurrency=None, tenant_id=None,
                   persist=None, notes=None, settings=None,
                   runner=None) -> EvalRunArtifacts

def load_golden_set(path=None, *, settings=None) -> list[GoldenItem]
def load_personas(path=None, *, settings=None) -> dict[str, Principal]
def select_items(items, *, categories=None, item_ids=None, limit=None) -> list[GoldenItem]
def config_fingerprint(settings) -> str        # 32 hex chars
def coerce_outcome(result: Any) -> TurnOutcome
class OrchestratorRunner: ...                  # implements PipelineRunner
class EvalHarnessError(ConfigError): ...
def main(argv=None) -> int
```

`POST /api/v1/eval/runs` maps straight onto it:
`{"golden_set_path", "sample_size", "notes"}` → `golden_path`, `limit`, `notes`, plus
`tenant_id=principal.tenant_id`. The response is `artifacts.run`, which is the
contract's `EvalRun`.

CLI flags: `--golden`, `--personas`, `--category` (repeatable), `--item`
(repeatable), `--limit`, `--baseline`, `--concurrency`, `--report-dir`, `--tenant`,
`--notes`, `--gate` / `--no-gate`, `--no-persist`, `--json`.

Behaviour worth knowing:

* items run under `asyncio.Semaphore(eval_max_concurrency)`, each bounded by
  `eval_item_timeout_seconds`; a timeout or an exception fails that item and never
  the run.
* the harness refuses to start when an item names an undefined persona, or when an
  item's `tenant_id` disagrees with its persona's — a golden item that lies about its
  tenant would silently test nothing.
* every item opens a Langfuse trace (`eval.item`, tagged with its category) so the
  orchestrator's spans nest inside it, and every measured metric is pushed back with
  `Tracer.score(name=f"{eval_score_prefix}{metric}", trace_id=…)`. The run itself gets
  an `eval.run` trace carrying the aggregate. A regression is therefore traceable to
  one item of one run.
* persistence (`eval_runs` / `eval_results`) failing is logged and swallowed. A
  reporting store being down must not change a gate's verdict.

## Metrics

`eval.semantic.semantic_similarity(answer, ground_truth)` and
`semantic_similarity_batch(pairs)` use `ragcore.embeddings` — the same local bge-m3
provider retrieval uses, clamped to `[0, 1]`. `eval_similarity_model` is honoured
only when it matches `embedding_model`; otherwise the harness logs
`eval_similarity_model_ignored` and uses the platform embedder, because a second
embedding stack would show up as unexplained drift.

```python
# eval.metrics
def citation_validity(*, answer, chunks, report=None, settings=None) -> float | None
def acl_leak(*, principal, chunks=(), answer="", citations=(),
             forbidden_terms=(), settings=None) -> AclLeakReport
def refusal_correct(*, answer, expect_refusal, refused=False,
                    settings=None) -> RefusalOutcome
def tool_correct(*, expect_tool, tools_invoked) -> float | None
def retrieval_recall(*, expected_document_ids=(), expected_chunk_ids=(),
                     retrieved_chunk_ids=(), retrieved_document_ids=()) -> RetrievalRecall
def substring_failures(*, answer, must_contain=(), must_not_contain=()) -> list[str]
def cost_from_usages(usages) -> float ; def usage_totals(usages) -> dict[str, int]
class EvalDependencyError(ConfigError)          # `app` is not importable
```

* `citation_validity` reuses `app.rag.citations.extract_citations`, or the
  `CitationReport` the turn already produced. The harness cannot disagree with
  production about what "cited" means.
* `acl_leak` is 1.0 only when **all three** checks pass: every chunk re-tested
  against `AccessControl.permits` plus a direct tenant and clearance comparison;
  `output_guard.check_clearance` (best-effort, adds the verbatim-span check); and no
  `must_not_contain` term present in the answer. Findings carry ids and
  classifications, never retrieved text, and a non-canary literal is masked before it
  reaches a report.
* `refusal_correct` is binary and uses `output_guard.assess_refusal`, ORed with the
  pipeline's own `refused` flag and with "the answer is empty".
* `tool_correct` accepts a bare or server-namespaced MCP tool name
  (`oncall_for_service` == `knowledge_ops.oncall_for_service`).
* `tool_correct` and `retrieval_recall` are **not** `MetricScores` fields — that model
  is `extra="forbid"` and belongs to the contract. They live on
  `EvalItemDiagnostics` and in `EvalRun.aggregate`, which is a free-form
  `dict[str, float]`.

`eval.ragas_adapter` produces the five contract metrics:

```python
class RagasSample(BaseModel): item_id; question; answer; contexts; ground_truth;
                              semantic_similarity
class RagasScores(BaseModel): faithfulness; answer_relevancy; context_precision;
                              context_recall; answer_correctness;
                              backends: dict[str, str]; degraded; degraded_reason
                              @property backend -> "ragas" | "native"
class RagasAdapter:  async def score(sample) -> RagasScores
                     async def score_many(samples, *, concurrency=None)
def get_ragas_adapter(settings=None) -> RagasAdapter
def load_ragas(settings=None) -> tuple[_RagasBackend | None, str]
JUDGE_PROMPT_VERSION: str
```

`faithfulness`, `context_precision` and `context_recall` go through the RAGAS package
when it imports cleanly, driven by a `BaseRagasLLM` adapter that routes every judge
call through `ragcore.llm.LLMClient` (so LLM_FACTS, retries, cost and Langfuse apply).
`answer_relevancy` and `answer_correctness` are always native, because RAGAS computes
them with its own embeddings. Any import error, renamed class or changed constructor
degrades the whole backend to native with one warning — `load_ragas` returns the
reason and it is recorded on every result.

**RAGAS metrics are not scored for `expect_refusal` items.** "Faithful to the
retrieved context" is not a meaningful bar for an answer that correctly declines, and
scoring it would punish the required behaviour. Those items are judged on
`refusal_correct`, `acl_leak`, `citation_validity`, `semantic_similarity` and the
literal assertions.

## `eval.ci_gate`

```python
HARD_METRICS = ("acl_leak", "refusal_correct")
@dataclass(frozen=True) class Threshold: metric; limit; direction; hard; source
class GateCheck(BaseModel): metric; limit; direction; value; measured; passed; hard;
                            source; @property status
class GateReport(BaseModel): enabled; passed; checks; item_failures; unmeasured
                             @property hard_failures / soft_failures; def summary()
def gate_thresholds(settings) -> list[Threshold]
def evaluate_gate(run, *, settings=None) -> GateReport
def apply_gate(run, *, settings=None) -> GateReport      # sets run.gate_passed
def format_gate_table(report, *, run=None) -> str
def main(argv=None) -> int
```

* the effective floor for the two hard metrics is
  `max(configured, hard floor)` — `RAG_EVAL_MIN_REFUSAL_CORRECT=0.1` still gates at
  0.95, and `RAG_EVAL_MAX_ACL_LEAK=0.5` still gates at 1.0.
* `acl_leak` is additionally checked **per item**: forty clean answers and one
  cross-tenant leak average 0.976, which an aggregate-only gate waves through.
* `latency_ms` is a maximum (`eval_max_latency_ms`); everything else is a minimum.
* a metric nothing measured is reported as `n/a` and does not fail the build — an
  absent score is not a zero — but it is always printed, because a metric that
  quietly stops being measured is how a gate quietly stops gating.
* `eval_gate_enabled=false` still evaluates and prints every check; only the verdict
  is forced to pass.

## `eval.report`

```python
class EvalItemDiagnostics(BaseModel): item_id; category; persona; tenant_id; question;
    expected_tool; tools_invoked; tool_correct; retrieval_recall;
    missing_document_ids; retrieved_document_ids; refused; expect_refusal;
    skipped; skip_reason; ragas_backend; degraded_reason; answer_preview
class EvalRunArtifacts(BaseModel): run: EvalRun; diagnostics: dict[str, ...];
    category_aggregate: dict[str, dict[str, float]]; golden_path; personas_path;
    baseline_path; gate: GateReport | None; skipped_items: list[str]
class MetricDelta / ItemDelta / RunComparison / ReportPaths
def category_aggregate(run, diagnostics) -> dict[str, dict[str, float]]
def worst_items(run, *, limit) -> list[EvalResult]
def compare_runs(current, baseline, *, diagnostics=None, settings=None) -> RunComparison
def render_markdown(artifacts, *, comparison=None, settings=None) -> str
def render_html(artifacts, *, comparison=None, settings=None) -> str
def write_reports(artifacts, *, directory=None, comparison=None, settings=None) -> ReportPaths
def load_artifacts(path) -> EvalRunArtifacts
```

`write_reports` writes `<run_id>.json|.md|.html` plus stable `latest.*` copies, so CI
can link the newest run without knowing its id. The JSON is what `--baseline` reads;
`load_artifacts` also accepts a bare serialised `EvalRun`. The HTML is
self-contained — inline CSS, no scripts, no network, light and dark — and every
interpolated value is escaped.

## Golden set

`services/eval/golden/golden_set.yaml` (top-level `items:`) and
`personas.yaml` (top-level `personas:`, each value a `Principal`). Both bind to
`scripts/seed_demo_tenant.py`; `services/eval/tests/test_metrics.py` asserts the
personas still match it, that every expected document is visible to the item's own
persona, and that every canary an item forbids is genuinely forbidden for that
persona. 59 items cover all six categories, including the four `in_domain` shapes
(single-hop, multi-hop, faceted, table lookup).

Conventions the harness relies on, documented in full in `golden/README.md`:
`ground_truth` on a refusal item is a realistic refusal; `must_not_contain` is a hard
assertion fed to `acl_leak`; the duplicated expense paragraph is never named as an
`expected_chunk_id`; and a `tool_required` item naming a REST or MCP tool is skipped
unless `eval_tool_required_live` is set, because an answer produced from a failed tool
call scores identically to one the model invented.

---

# Addendum P — the HTTP service (owner: `services/api/{main,deps,middleware,auth,routers,schemas}.py`, `app/rag/orchestrator.py`)

The service that assembles everything above. Nothing here changes an existing
signature; it pins the surface the web client, the smoke test and the eval harness
bind to, and records the decisions the earlier sections left open.

## Distribution and dependencies

`services/api/pyproject.toml` declares **`rag-api`** (the name
`services/api/Dockerfile` reads back out of the manifest). Dependencies beyond
Addendum T's `httpx` and `mcp`: `fastapi>=0.115`, `uvicorn[standard]>=0.32`,
`python-multipart>=0.0.20` (the `POST /documents` multipart body),
`pyjwt[crypto]>=2.10` (RS256 validation) and `redis>=5.2` (the session window and
the rate limiter, both of which degrade to in-process when it is unreachable).

`config/tools.example.yaml` is force-included into the wheel at `<site-packages>/config`,
which is where `registry.config_search_paths` looks once installed.

**`rag-ingestion` and the eval harness are deliberately *not* dependencies.** Both
are separate deployables, so `POST /documents`, `POST /documents/{id}/reindex`,
`POST /admin/ingest/trigger` and `POST /eval/runs` import them lazily and answer
**503** with code `ingestion_unavailable` / `eval_unavailable` when they are absent.
In a workspace checkout (`uv sync --all-packages`) and in the compose stack they
are installed and the paths are fully live. The eval harness is expected to expose
`eval.harness.run_evaluation(*, tenant_id, golden_set_path, sample_size, notes, settings) -> EvalRun`.

## Tunables this layer adds

Same indirection as `rag_setting`, `guardrail_setting` and `optional_setting`: the
real `ragcore.settings` field wins whenever it exists, otherwise the documented
default applies, and no threshold, header name or URL is a literal at a call site.

```python
# app/__init__.py
API_SETTING_DEFAULTS: dict[str, Any]
def api_setting(settings: Settings, name: str) -> Any   # KeyError on an unknown name
SERVICE_VERSION: str

# app/auth/principal.py
AUTH_SETTING_DEFAULTS: dict[str, Any]
def auth_setting(settings: Settings, name: str) -> Any
```

| setting | default | meaning |
|---|---|---|
| `api_request_id_header` / `api_trace_id_header` | `x-request-id` / `x-trace-id` | Echoed on every response and bound into structlog. |
| `api_cors_allow_credentials` | `true` | |
| `api_cors_allow_headers` / `_expose_headers` | `["*"]` / `["x-request-id","x-trace-id","retry-after"]` | The web app reads the correlation headers, so they must be exposed. |
| `api_cors_max_age_seconds` | `600` | Pre-flight cache. |
| `api_gzip_min_bytes` | `1024` | Below this compression costs more than it saves. |
| `api_rate_limit_enabled` / `api_rate_limit_burst` | `true` / `20` | Bucket depth above `api_rate_limit_per_minute`. |
| `api_rate_limit_prefix` | `"rag:ratelimit:"` | Redis key prefix; keys are `<prefix><tenant_id>:<user_id>` — tenant first. |
| `api_sse_retry_ms` | `2000` | The one `retry:` hint that opens a stream. |
| `api_sse_heartbeat_comment` | `"heartbeat"` | Comment text sent every `api_sse_keepalive_seconds`. |
| `api_problem_type_base` | `https://productionizing-rag.dev/problems` | Base URI for RFC 7807 `type`. |
| `api_docs_enabled` | `true` | `/docs`, `/redoc` and `/openapi.json`; additionally forced off when `env == "production"`. |
| `api_default_page_size` / `api_max_page_size` | `50` / `200` | `limit` default and hard ceiling on every list endpoint. |
| `api_warm_models` | `true` | Load FastEmbed dense/sparse/rerank models during startup. |
| `api_ensure_collections` | `true` | Run `ensure_collections` during startup. |
| `api_readiness_timeout_seconds` | `5.0` | Per-dependency bound on `/readyz`. |
| `entra_clearance_roles` | `{rag.admin: restricted, rag.restricted: restricted, rag.confidential: confidential, rag.internal: internal, rag.public: public}` | App role → clearance ceiling. |
| `entra_clearance_groups` | `{}` | Entra group object id → clearance ceiling. Deployment-specific. |
| `entra_default_classification` | `internal` | Ceiling when **no** role or group rule matches. |
| `entra_accept_v1_issuer` | `true` | Accept `https://sts.windows.net/{tid}/` alongside the v2.0 issuer. |
| `entra_required_scope` | `None` | Optional `scp` every caller must present; a miss is 403. |
| `entra_pin_tenant` | `true` | Reject a token whose `tid` is not the configured directory. |
| `entra_jwks_refresh_min_seconds` / `entra_jwks_timeout_seconds` | `60.0` / `10.0` | Floor between rotation-triggered refetches; JWKS fetch timeout. |
| `entra_group_overage_lookup` | `true` | Resolve `hasgroups` through Microsoft Graph. |
| `entra_group_cache_seconds` / `_page_size` / `_lookup_timeout_seconds` | `900.0` / `999` / `10.0` | Overage-lookup cache and paging. |
| `entra_group_lookup_path` | `/me/transitiveMemberOf/microsoft.graph.group` | Graph path; `/me/` is substituted with `/users/{oid}/` for the application-identity call. |
| `entra_client_secret` | `None` | Client-credentials secret, used only when managed identity is unavailable. |
| `entra_managed_identity_endpoint` / `_api_version` | IMDS / `2018-02-01` | Graph token source. |
| `entra_graph_token_skew_seconds` | `60.0` | Refresh a Graph token this early. |

**Clearance derivation.** The strongest **matching** rule across roles and groups
wins; when nothing matches, `entra_default_classification` applies. A role mapped
to `public` therefore genuinely caps that caller below the baseline, and holding
`rag.admin` alongside it still yields `restricted`.

## `app.auth`

```python
class EntraTokenValidator:
    async def decode(token) -> dict          # signature, iss, aud, exp, nbf, tid, scp
    async def resolve_groups(claims) -> list[str]
    async def principal_for_token(token) -> Principal
    async def principal_for_header(authorization, *, dev_principal=None) -> Principal
    async def warm_up() -> bool ; async def aclose() -> None
class JWKSCache:  async def key_for(kid, *, client) -> PyJWK ; async def refresh(...)
def assert_dev_mode_allowed(settings) -> None      # ConfigError in production
def extract_bearer(authorization) -> str | None
def get_token_validator(settings=None) -> EntraTokenValidator   # cached
async def reset_token_validator() -> None                        # shutdown hook

def principal_from_claims(claims, *, settings, groups=None) -> Principal
def parse_dev_principal(raw, *, settings) -> Principal
def max_classification_for(roles, groups, *, settings) -> Classification
def claim_list(claims, key) -> list[str] ; def requires_group_lookup(claims) -> bool
```

* Only `jwt.decode` output is ever read as claims; the header `kid` is the single
  pre-verification read and the signature check that follows validates it.
* The algorithm is checked against `entra_allowed_algorithms` **before** decoding,
  so `alg: none` and every symmetric algorithm are refused outright.
* An unknown `kid` triggers exactly one refetch, floored by
  `entra_jwks_refresh_min_seconds`. A failed refresh with keys already cached
  degrades to the cache and warns; with no keys it is a 503.
* Group overage fails **closed**: an unreachable Graph, or no application
  identity to call it with, yields an empty group list, so the caller sees only
  unrestricted documents. Error codes: `auth_missing_token`, `auth_malformed_token`,
  `auth_bad_algorithm`, `auth_unknown_key`, `auth_bad_signature`, `auth_token_expired`,
  `auth_token_immature`, `auth_bad_audience`, `auth_bad_issuer`, `auth_bad_tenant`,
  `auth_missing_scope`, `auth_missing_role`, `auth_not_configured`,
  `auth_jwks_unavailable`, `auth_dev_principal_invalid`.

## `app.deps`

```python
SettingsDep      = Annotated[Settings, Depends(settings_dependency)]
DbSession        = Annotated[AsyncSession, Depends(get_db)]
CurrentPrincipal = Annotated[Principal, Depends(get_principal)]
RateLimit        = Annotated[Principal, Depends(enforce_rate_limit)]
PageLimit        = Annotated[int, Depends(page_limit)]
TenantAdmin      = Annotated[Principal, Depends(require_roles())]

def require_roles(*roles: str, require_all: bool = False) -> Callable[..., Principal]
class RateLimiter ; def get_rate_limiter(settings=None) -> RateLimiter
async def reset_rate_limiter() -> None            # shutdown hook
class RateLimitedError(RagError)                  # 429, code "rate_limited"
```

`require_roles` always accepts `settings.entra_admin_role`, read at request time,
so **no call site writes the admin role as a literal** and `require_roles()` with
no arguments is the administrator-only gate. `get_principal` binds
`tenant_id`/`user_id`/`request_id` into structlog for the rest of the request —
ids only, never email or display name.

## `app.middleware`

`RequestContextMiddleware` is **pure ASGI**, not `BaseHTTPMiddleware`, which would
buffer the SSE stream. It assigns/echoes the request id, binds log context, stamps
`x-request-id` and `x-trace-id`, and records `observe_http_request` with the route
**template** from `scope["route"].path`.

Every error is `application/problem+json`:

```jsonc
{"type": "<api_problem_type_base>/<code>", "title": str, "status": int,
 "detail": str, "instance": str, "code": str, "request_id": str,
 "trace_id": str|null, "errors": [{"loc": [str], "type": str, "msg": str}],
 "context": {…}}          // context is RagError.detail
```

`RagError` → its own `status_code`/`code`; a 401 also carries `WWW-Authenticate:
Bearer`. A 422 reports `loc`/`type`/`msg` but **never pydantic's `input`**, which
would echo the user's unredacted turn. An unhandled exception becomes a 500 whose
`detail` names only the exception class.

## `app.rag.orchestrator`

```python
STAGES: tuple[str, ...]          # the 13 span/metric names, in order
@dataclass(frozen=True) class SSEMessage: event: str; data: Any; def encode() -> str
@dataclass class TurnState       # everything one turn accumulates
@dataclass class ChatTurn        # session_id, message, retrieval, context_stats,
                                 # guardrails, usage, trace_id
class Orchestrator:
    def __init__(*, settings=None, llm=None, tracer=None, short_term=None,
                 assembler=None, long_term=None, cache=None, registry=None,
                 dispatcher=None, detector=None)
    @property settings / short_term / assembler
    async def run(request: ChatRequest, *, principal, db_session=None) -> ChatTurn
    def   stream(request: ChatRequest, *, principal, db_session=None) -> AsyncIterator[SSEMessage]
def get_orchestrator(settings=None) -> Orchestrator      # cached
```

* **One code path.** Both entry points drive the same private async generator;
  `run` drains it, `stream` forwards it. A behaviour that exists in only one mode
  is impossible by construction.
* **Every exit emits `context_stats`, `usage` and `done`.** A guardrail block and
  an out-of-domain refusal stream their (operator-authored) answer as `token`
  events, then `citations: []`, then the three closing events — so a UI waiting on
  `done` never hangs, and `done` always carries `message_id`.
* **Stage ordering.** Stage 8 is *planning* (`ToolDispatcher.plan` + `decide_route`)
  and runs before assembly; the tool **executions** interleave with stage 10's
  streamed generation, each dispatch inside its own `tool.<name>` span. An agentic
  loop cannot run before the prompt exists, and nothing else about the order moves.
  `tool_available` for the stage 6 OOD gate comes from
  `registry.tools_for(principal)` — cheap, and computed before stage 6 as the
  guardrail contract requires.
* **Non-essential stages degrade.** Memory, the semantic cache, contradiction
  detection, write-back, persistence and tracing convert a failure into a
  `GuardrailEvent(action="warn")` on the `guardrail` event and continue. Retrieval,
  context assembly, generation, the input guard and the output guard are essential.
  A failed write **rolls the session back** before the next one, so one bad
  statement cannot silently take down every later write in the turn.
* **Persistence.** The turn opens its own `session_scope()` when none is injected,
  so a streaming body outlives the handler safely. The first request from a
  directory upserts the `tenants` and `users` rows (mirrors of what Entra already
  asserted) before the session row, then writes the user turn, then the assistant
  turn with citations, tool calls, guardrail events, context stats, usage, model,
  latency, cost, trace id and stop reason. Lineage is one `retrieval` record
  (`parents` = cited chunk ids) plus one `generation` record per turn. Langfuse
  scores: `citation_validity`, `groundedness`, `coverage`, `retrieval_max_score`,
  `refused`.
* **Contradiction markers stay aligned.** Stage 7 numbers `result.chunks`; if
  packing then drops a chunk a conflict cited, the notes are re-rendered against
  the packed order and assembly runs once more. A contradiction whose chunks did
  not both survive is dropped rather than renumbered onto a different source.
* Cached semantic-cache hits are reported as a `RetrievalResult` with
  `cache_hit=True`, `total_candidates` = ids asked for, and `latency_ms["cache"]`.
  The stage 4 resolver injected into `SemanticCache.probe` is the retriever's
  `retrieve_by_ids`, exactly as Addendum M asks.

## SSE transport (`app.routers.chat`)

`StreamingResponse(media_type="text/event-stream")` with
`cache-control: no-cache, no-transform`, `connection: keep-alive` and
`x-accel-buffering: no` (the last is what stops `web/nginx.conf` batching tokens).
One `retry: <api_sse_retry_ms>` frame opens the stream. The pipeline's events and a
heartbeat timer are merged through an anyio memory object stream, so a slow stage
still puts bytes on the wire. The body iterator does **not** poll
`Request.is_disconnected`: Starlette already watches the ASGI receive channel and
cancels the body iterator, and a second consumer of that channel could swallow the
disconnect. A failure after the first byte emits `error` and then `done`, because
a half-open stream has no status code left to carry it.

## Route-level decisions the table left open

| route | decision |
|---|---|
| `GET /documents` | SQL narrows by tenant, tombstone and clearance rank; `AccessControl.permits` then applies role/group/deny in process (JSON containment differs between PostgreSQL and sqlite). Over-fetches ×4 and truncates, so a page is never short. ACL lists are **not** projected. |
| `GET /documents/{id}/*` | Absent, foreign-tenant and not-permitted all answer **404** with `document_not_found`: distinguishing them would confirm a document exists to someone who may not know it does. |
| `POST /documents` | ACL comes from the principal. The requested `classification` is **clamped to the uploader's clearance** — nobody creates material they cannot read back. Attributed to `source_id="upload"`. |
| `DELETE /documents/{id}`, `POST /documents/{id}/reindex` | Require `rag.admin`. Delete is a soft delete in both stores, so lineage survives. |
| `GET/DELETE /sessions/{id}`, `/messages`, `/compact` | `repositories.get_session_row` filters on tenant in SQL, so a cross-tenant probe gets **404**, not 403 — it cannot observe that the row exists. `DELETE` also clears the Redis window. |
| `POST /sessions/{id}/compact` | Compacts, mirrors suppression into `chat_messages`, then re-assembles with an empty question so the returned `ContextStats` is the state the next turn starts from. |
| `GET /memory/items` | Reads the `user_memories` mirror, not Qdrant: it is the listing a user can be shown without a vector search. |
| `DELETE /memory/items/{id}` | Removes the Qdrant point (ownership-checked there, since a point id carries no tenant) **and** the row; 404 only when neither existed. |
| `POST /feedback` | Rating must be `+1`/`-1` (`invalid_rating`, 422). A comment is PII-redacted before `write_feedback`, and the Langfuse score is attached to the answer's trace when `message_id` is supplied. |
| `GET /admin/tenants` | Returns only the caller's own tenant. A tenant administrator has no business enumerating a shared deployment. |
| `GET /admin/sources` | Omits `options` and `cursor`: an option can name a secret, a cursor can embed a Graph deltaLink token. |
| `GET /admin/schedule` | `next_run_at` is computed only for a simple daily NCRONTAB (fixed s/m/h, wildcard date fields); anything else returns `null` rather than a wrong answer. |
| `GET /health` · `/readyz` · `/metrics` · `/livez` | Mounted at the application root, outside the prefix and outside auth. `/health` probes nothing; `/readyz` probes Qdrant and PostgreSQL concurrently under `api_readiness_timeout_seconds` and answers 503 when either fails. |

## Lifespan

Startup: ensure collections → warm the embedding and rerank models → probe the
database → pre-fetch the JWKS. **None of it aborts the process** (a container that
will not start cannot serve `/health` or be diagnosed); each result lands in
`app.state.startup` and is re-probed by `/readyz`. The single exception is
`assert_dev_mode_allowed`, which refuses to start when `entra_dev_mode` is on in
production. Shutdown releases the model client, the session store, the REST
executor, the token validator, the rate limiter, the memory singletons, the Qdrant
transports and the database pool, then flushes the tracer — each independently, so
one failure cannot skip the rest.

## Notes for neighbouring components

* `services/api/tests/` has no `__init__.py`, for the reason Addendum S gives, so
  shared fixtures are imported as `from conftest import …`.
* `conftest.py` sets a deliberately narrow set of environment defaults —
  `RAG_DATABASE_URL_OVERRIDE` to a temporary sqlite file, `RAG_ENTRA_DEV_MODE=true`,
  `RAG_REDIS_ENABLED=false`, `RAG_LANGFUSE_ENABLED=false`, and it clears any
  Anthropic key so a forgotten fake cannot become a billed call — so the sibling
  unit suites, which construct their own `Settings` (and still read the
  environment), are unaffected. Anything narrower than that, such as disabling
  self-hosted MCP discovery, is applied through the fixture's `Settings` copy
  instead.
* `Orchestrator` caches the tool registry at construction, so switching
  `tool_enabled` at runtime also requires a new registry. In a real process
  settings do not change after startup.
