# productionizing-rag-azure

A multi-tenant, ACL-aware RAG platform built to a written interface spec
(`docs/CONTRACTS.md`): hybrid retrieval over Qdrant, serverless delta ingestion on
Azure Functions, an agentic FastAPI chat surface backed by Anthropic Claude,
Azure OpenAI or Ollama, and a React client.

**Status.** The retrieval, guardrail, context, memory, tool, ingestion and API layers
are implemented and unit-tested (857 tests, all in-process, ~18 s; 74 % line coverage).
The four defects this section used to list — Makefile targets pointing at filenames
that did not exist, the evaluation gate failing to resolve its pipeline target,
`/metrics` serving nothing because `prometheus-client` was missing, and ~122
documented tunables being compile-time constants — are fixed, and
`packages/ragcore/tests/test_ops_wiring.py` now fails if any of them regresses.

What still needs a live environment rather than a unit test: the golden-set gate runs
against real Postgres, Qdrant and the Anthropic API, so it is exercised in CI (job
`eval`) rather than locally by default. See
[`docs/FEATURE_MAP.md`](docs/FEATURE_MAP.md) for the full production-ready vs
scaffolded breakdown.

## The ten capabilities, one line each

| # | Capability | One line |
|---|---|---|
| 1 | Multi-tenant ACL ingestion, serverless, nightly delta | Four real connectors (Blob/local, SharePoint Graph delta, HTTP crawler, SQL watermark) fan out through Durable Functions on a 02:30 NCRONTAB timer, refuse to run inside working hours, and write tenant/role/group/deny/classification onto every chunk payload. |
| 2 | Personalisation, long-term memory, fast similar-query retrieval | Salience-weighted dense recall from `rag_memories` with decay, TTL and a hard consent gate, plus a semantic cache that stores the retrieval *plan* (chunk ids) — never an answer — so every reuse is re-authorised against the live ACL filter. |
| 3 | Efficient in-session context management | `context.py` packs to a token budget measured with Claude's own `count_tokens` (never a chars/4 heuristic), in priority order, and sheds by marginal value per token with an audited reason for every drop. |
| 4 | API / MCP tool calling for un-indexed data | A declarative YAML registry of REST tools plus remote MCP (Anthropic connector) and self-hosted MCP (official SDK), behind one dispatcher that enforces tenant, role, loop, rate-limit, classification-egress and PII gates in that order. |
| 5 | Short-term memory and periodic suppression | A Redis-backed session window (in-process fallback) that suppresses rather than truncates: old turns fold into a rolling summary every 6 turns *and* at 75 % of budget, stale tool results are cleared with the `clear_tool_uses_20250919` context edit. |
| 6 | Hybrid semantic + BM25 + rerank + metadata filtering | Dense `bge-m3` and sparse `Qdrant/bm25` prefetch branches fused **server-side** by the Qdrant Query API (RRF or DBSF), then simhash dedupe, `bge-reranker-v2-m3` cross-encoder, MMR and a per-document cap. |
| 7 | React interface | Vite + React 19 + Tailwind + MSAL: streaming chat, retrieval inspector, source drawer, citation list, context meter, guardrail banner, tool trace, memory panel, admin documents/ingestion and an eval dashboard. |
| 8 | RAGAS + semantic-similarity validation vs a golden set | 59 golden items over six categories run through the real pipeline, scored with RAGAS (or native Claude judges) plus bge-m3 cosine, gated in CI with two hard floors. Needs live Postgres, Qdrant and an API key, so it runs in CI (job `eval`) rather than in the unit suite. |
| 9 | PII, OOD, contradictions, dedupe, citations, lineage, Langfuse | Presidio-or-regex PII with redact-before-log enforced by a structlog processor, an out-of-domain gate with score-collapse detection, recency/authority contradiction resolution that cites both sides, span-verified citations and lineage rows per turn. |
| 10 | Query transformation | One `claude-sonnet-5` structured call produces intent, rewrite, sub-questions, HyDE passage, extracted facets and an out-of-domain flag — and never raises: every failure degrades to a documented fallback plan. |

## Component diagram

```mermaid
flowchart TB
    subgraph client["Browser"]
        WEB["web/ — React 19 + Vite + Tailwind<br/>MSAL PKCE, fetch+ReadableStream SSE"]
    end

    subgraph api["services/api — FastAPI (Container App)"]
        MW["middleware<br/>request id · RFC 7807 · metrics"]
        AUTH["auth/entra<br/>RS256 · JWKS cache · Graph group overage"]
        ORCH["rag/orchestrator<br/>13 ordered stages"]
        RET["rag/retriever · mmr · citations"]
        GRD["rag/guardrails<br/>input · injection · ood · contradiction · output"]
        MEM["rag/memory<br/>short_term · long_term · semantic_cache"]
        CTX["rag/context — token-budget packing"]
        TOOL["rag/tools<br/>registry · rest · mcp · router"]
    end

    subgraph ing["services/ingestion — Azure Functions (Python v2 + Durable)"]
        TRG["timer · http · blob · queue triggers"]
        PIPE["pipeline: plan → fan-out → finalize"]
        CONN["connectors: blob/local · sharepoint · http · sql"]
        PARSE["parse → enrich(PII) → chunk → dedupe → upsert"]
    end

    subgraph core["packages/ragcore — shared library"]
        FILT["vectorstore/filters<br/>THE ACL chokepoint"]
        VEC["vectorstore<br/>collections · hybrid · writer"]
        LLM["llm client · prompts · pricing"]
        PII["pii · dedupe · observability · db"]
    end

    subgraph data["Data plane"]
        QD[("Qdrant<br/>rag_chunks · rag_memories · rag_semantic_cache")]
        PG[("PostgreSQL<br/>17 tables")]
        RD[("Redis<br/>session window · rate limit")]
        BLOB[("Blob<br/>documents · raw · manifests")]
    end

    subgraph ext["External"]
        ANT["LLM provider<br/>Anthropic · Azure OpenAI · Ollama"]
        LF["Langfuse"]
        KV["Key Vault + Managed Identity"]
        MCP["Remote / self-hosted MCP servers"]
    end

    WEB -->|"Bearer JWT"| MW --> AUTH --> ORCH
    ORCH --> RET & GRD & MEM & CTX & TOOL
    RET --> FILT --> VEC --> QD
    MEM --> VEC
    CTX --> LLM --> ANT
    TOOL --> MCP
    ORCH --> PG
    MEM --> RD
    TRG --> PIPE --> CONN --> PARSE --> VEC
    PIPE --> BLOB
    PIPE --> PG
    core --> LF
    api --> KV
    ing --> KV

    subgraph evalpkg["services/eval"]
        EV["golden set · RAGAS · semantic · CI gate"]
    end
    EV -.->|"imports the orchestrator in-process"| ORCH
```

## Quickstart from a clean clone

Prerequisites: Python 3.13 + [uv](https://docs.astral.sh/uv/), Node 20.19+ (CI uses
25), Docker for the local data plane.

```bash
# 1. environment + workspace
make setup                       # cp .env.example .env; uv sync --all-packages --all-groups

# 2. data plane (Qdrant, PostgreSQL, Redis, Langfuse + its ClickHouse/MinIO)
make up                          # waits for qdrant/postgres/redis to report healthy

# 3. schema
make migrate                     # alembic -c packages/ragcore/ragcore/db/alembic.ini upgrade head

# 4. Qdrant collections + payload indexes
make bootstrap                   # scripts/bootstrap_qdrant.py

# 5. demo tenants, personas and the golden corpus
make seed                        # scripts/seed_demo_tenant.py

# 6. API on http://localhost:8000  (docs at /docs)
make api

# 7. web on http://localhost:5173  (separate shell)
make web

# 8. end-to-end check incl. ACL negatives
make smoke                       # scripts/smoke_test.py
```

### Makefile targets

`docs/CONTRACTS.md` Addendum I pins the four script paths the Makefile must invoke;
earlier drafts referenced shorter names that never existed, so `make bootstrap`,
`make seed` and `make smoke` failed at the shell. They now point at
`scripts/bootstrap_qdrant.py`, `scripts/seed_demo_tenant.py` and
`scripts/smoke_test.py`, and `test_ops_wiring.py` fails if a target ever again names
a script that is not a real file — including the `python -m` and `uvicorn` targets,
which are checked by importing what they name.

The first `make api` (or first ingestion run) downloads the FastEmbed weights for
`BAAI/bge-m3`, `Qdrant/bm25` and `Xenova/bge-reranker-v2-m3` into
`RAG_EMBEDDING_CACHE_DIR` (~2.5 GB, several minutes). Startup never aborts on a
failure — check `GET /readyz`, not `GET /health`, for dependency state.

## Choosing an LLM provider

`RAG_LLM_PROVIDER` selects the chat backend. Retrieval is unaffected: embeddings and
reranking are local FastEmbed models on every provider, so switching does **not**
touch your Qdrant collections or require re-ingestion.

### Anthropic (default)

```bash
RAG_LLM_PROVIDER=anthropic
RAG_ANTHROPIC_API_KEY=sk-ant-...
```

Nothing else to set. This is the only provider with extended thinking, prompt
caching, remote MCP and context compaction, and the only one whose token counts come
from the model's own counting endpoint.

### Azure OpenAI

```bash
RAG_LLM_PROVIDER=azure_openai
RAG_AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com
RAG_AZURE_OPENAI_API_KEY=...        # omit to use managed identity
RAG_AZURE_OPENAI_API_VERSION=2024-10-21

# Aliases name *deployments*, not model families — Azure routes on deployment name.
RAG_LLM_MODEL_ALIASES={"claude-opus-5":"gpt-4o","claude-sonnet-5":"gpt-4o-mini","claude-haiku-4-5":"gpt-4o-mini"}

# Anthropic-only features must be off, or startup refuses to proceed.
RAG_ANTHROPIC_THINKING=false
RAG_ANTHROPIC_CACHE_SYSTEM=false
RAG_TOOL_MCP_ENABLED=false
RAG_CONTEXT_COMPACTION_ENABLED=false
```

### Ollama

```bash
RAG_LLM_PROVIDER=ollama
RAG_OLLAMA_BASE_URL=http://localhost:11434     # '/v1' is appended
RAG_LLM_MODEL_ALIASES={"claude-opus-5":"llama3.1:70b","claude-sonnet-5":"llama3.1:8b","claude-haiku-4-5":"llama3.1:8b"}
RAG_ANTHROPIC_THINKING=false
RAG_ANTHROPIC_CACHE_SYSTEM=false
RAG_TOOL_MCP_ENABLED=false
RAG_CONTEXT_COMPACTION_ENABLED=false
```

Pull the models first (`ollama pull llama3.1:8b`). Tool calling needs a model trained
for it — `llama3.1`, `qwen2.5` and `mistral-nemo` work; many smaller models will
simply never emit a tool call, which degrades the agentic loop to plain RAG without
erroring.

### Why the aliases are mandatory

Every call site names a Claude model directly (`claude-opus-5` for generation,
`claude-sonnet-5` for query transformation, `claude-haiku-4-5` for classification).
Without an alias for each of the three, the request would reach Azure or Ollama as
`claude-opus-5` and be rejected there — once per turn, forever. Settings therefore
refuse to construct if any slot is unmapped, so the failure lands at startup instead.

The same reasoning covers the four feature flags: a deployment that asks for extended
thinking on a provider that has none would quietly do less than its configuration
says, so that combination is refused rather than ignored. This mirrors how
`entra_dev_mode` is refused in production.

### What differs per provider

| | Anthropic | Azure OpenAI | Ollama |
|---|---|---|---|
| Chat, streaming, tool calling | ✅ | ✅ | ✅ (model-dependent) |
| Structured output | ✅ tool-forced | ✅ strict JSON schema | ⚠️ JSON mode + prompt |
| Token counting | ✅ exact, via API | ✅ exact, `tiktoken` | ⚠️ approximate, padded 15 % |
| `clear_tool_uses` context edit | ✅ server-side | ✅ applied client-side | ✅ applied client-side |
| Extended thinking | ✅ | ❌ refused | ❌ refused |
| Prompt caching | ✅ | ❌ refused | ❌ refused |
| Remote MCP connector | ✅ | ❌ refused | ❌ refused |
| Context compaction | ✅ | ❌ refused | ❌ refused |
| Per-token cost reporting | ✅ | ✅ | ✅ zero (local) |

Ollama's token count is `tiktoken` scaled by 1.15, because Llama-family tokenisers
disagree with `o200k_base`. The context packer treats the number as a budget, so it
is biased to over-count: that wastes a little of the window, where under-counting
would overflow it and fail the request.

`clear_tool_uses` has no OpenAI equivalent, so the provider applies it to the outgoing
messages itself — stale tool results have their bodies replaced with a marker before
the request is sent. They are replaced rather than removed because OpenAI rejects an
assistant turn whose `tool_calls` have no answering `tool` message.

**None of this has been exercised against a live Azure or Ollama endpoint.** The 65
provider tests stub the SDK, so the translation layer is verified but the first real
call is not.

## Environment variables that must be set

Everything is a `RAG_`-prefixed field on `ragcore.settings.Settings`. `.env.example`
documents 324 of the 367 fields, and `test_settings_tunables.py` fails if any entry
there is not a real field — so the file never drifts into documenting a setting that
does not exist. The 43 undocumented fields are internal defaults nobody is expected to
override. These are the ones with no usable default:

| Variable | Why |
|---|---|
| `RAG_ANTHROPIC_API_KEY` | Every model call. Omit only to let the SDK resolve `ANTHROPIC_API_KEY`. Without a key, query transform, enrichment, contradiction adjudication and generation degrade or fail. |
| `RAG_ENTRA_TENANT_ID` · `RAG_ENTRA_CLIENT_ID` | Token issuer and expected audience. Without them the API cannot validate a real JWT (`auth_not_configured`). |
| `RAG_QDRANT_URL` (+ `RAG_QDRANT_API_KEY` off-localhost) | The vector store. |
| `RAG_POSTGRES_HOST` / `_USER` / `_PASSWORD` / `_DB` | Sessions, messages, documents, lineage, audit. Or set `RAG_DATABASE_URL_OVERRIDE`. |
| `RAG_PII_HASH_SECRET` | HMAC key for `hash`-mode redaction. It is a **join key**: change it and previously redacted values stop matching. The shipped value is a placeholder. |
| `RAG_ENV` | `local` \| `dev` \| `staging` \| `production`. `production` makes `Settings` refuse to construct when `entra_dev_mode` or `tool_allow_insecure_http` is true, and turns `/docs` off. |

Local-only, must never reach production:

| Variable | Effect |
|---|---|
| `RAG_ENTRA_DEV_MODE=true` | Accepts an unsigned `x-dev-principal` header instead of a JWT. `Settings` refuses to construct, and the API refuses to start, when this is on with `RAG_ENV=production`. |
| `RAG_TOOL_ALLOW_INSECURE_HTTP=true` | Permits non-HTTPS REST tool targets. Same production refusal. |

Optional but load-bearing: `RAG_LANGFUSE_ENABLED` + `RAG_LANGFUSE_PUBLIC_KEY` +
`RAG_LANGFUSE_SECRET_KEY` (tracing degrades to a no-op without all three),
`RAG_REDIS_ENABLED` (session window and rate limiter degrade to in-process),
`RAG_RERANK_ENABLED=false` (swaps in `NoopReranker` for cheap CI),
`RAG_PII_USE_PRESIDIO=false` (regex-only recognisers; Presidio is the `pii` extra:
`uv sync --extra pii`).

Browser variables are `VITE_*` only and are read at build time — see the `# web`
block of `.env.example` and `web/.env.example`.

## Running the eval gate

```bash
uv run python -m eval.run --golden services/eval/golden/golden_set.yaml --gate
# or: make eval        (this target's path is correct)
```

**This currently fails.** The harness resolves `eval_pipeline_target`, whose default
is `app.rag.orchestrator:run_turn`; the orchestrator module exposes
`Orchestrator.run`, `Orchestrator.stream` and `get_orchestrator`, but no module-level
`run_turn`. The run aborts with:

```
EvalHarnessError: 'app.rag.orchestrator:run_turn' is not callable
```

exit code 2 ("the harness could not run"). There is **no environment override**:
`eval_pipeline_target` is not a `Settings` field and `Settings` is configured
`extra="ignore"`, so `RAG_EVAL_PIPELINE_TARGET` is silently discarded. The two ways
to run it today are to add a `run_turn(*, message, principal, …)` coroutine to
`app/rag/orchestrator.py`, or to call `run_eval(runner=…)` in process with your own
`PipelineRunner`. Everything downstream of the runner — scoring, the gate, the
JSON/Markdown/HTML reports, the baseline comparison — is implemented and unit-tested.

Related: `POST /api/v1/eval/runs` answers `503 eval_unavailable` because it imports
`eval.harness.run_evaluation`, and `services/eval/harness.py` does not exist.

See [`docs/EVALUATION.md`](docs/EVALUATION.md) for the golden-set schema, every metric
and the gate thresholds.

## Checks

```bash
make lint        # ruff check + ruff format --check   (clean)
make test        # pytest across every workspace member (857 pass, ~18 s)
make typecheck   # mypy (not run by CI)
```

CI (`.github/workflows/ci.yml`) runs five jobs: `lint`, `test` (with live Postgres and
Qdrant services and `--cov-fail-under=70`), `bicep` (compile + linter-warnings-as-errors
+ shellcheck), `web` (typecheck + production build) and `eval` (the gate — see above).

## Documentation

| Document | Contents |
|---|---|
| [`docs/CONTRACTS.md`](docs/CONTRACTS.md) | The authoritative interface spec everything was written against. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Request lifecycle, ingestion lifecycle, Qdrant design and why, context/memory design, tool execution model. |
| [`docs/SECURITY.md`](docs/SECURITY.md) | Threat model, tenant isolation enforcement points, ACL truth table, PII at every hop, injection, secrets, residual risks. |
| [`docs/OPERATIONS.md`](docs/OPERATIONS.md) | Bicep deploy, ingestion schedule, monitoring, scaling and cost, backup/restore, five-failure runbook. |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Golden-set schema, metrics, gate thresholds, adding an item, baselines, reading the report. |
| [`docs/FEATURE_MAP.md`](docs/FEATURE_MAP.md) | Requirement → files/functions/tests, with an honest production-ready vs scaffolded column. |

Where this documentation and `docs/CONTRACTS.md` disagree, **this documentation
describes what the code does** and says so explicitly; the contract describes what it
was supposed to do.
