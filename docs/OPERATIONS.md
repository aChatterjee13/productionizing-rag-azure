# Operations

Deploying, scheduling, monitoring, scaling, backing up and un-breaking this platform.

Read [`docs/SECURITY.md` §10](SECURITY.md#10-residual-risks) before treating any of
this as production-ready. Three operational facts up front, because they change how
you use everything below:

* **`/metrics` serves nothing.** `prometheus-client` is not a `ragcore` dependency, so
  every `observe_*` call is a no-op and all seventeen `rag_*` series are absent.
  Langfuse and structured logs are your only telemetry today.
* **122 tunables are compile-time constants** — including the readiness timeout, the
  page-size ceiling and every guardrail threshold outside the ~30 that are real
  `Settings` fields. See [ARCHITECTURE §6](ARCHITECTURE.md#6-configuration-reality).
* **`make bootstrap`, `make seed` and `make smoke` invoke filenames that do not
  exist.** Use `scripts/bootstrap_qdrant.py`, `scripts/seed_demo_tenant.py`,
  `scripts/smoke_test.py`.

---

## 1. Deploying with Bicep

One resource-group deployment. `infra/azure/main.bicep` composes nine modules;
`deploy.sh` is an idempotent wrapper.

### Prerequisites

* `az` CLI logged in, `az bicep install` done.
* Container images built and pushed for `rag-api` (build context = repository root, so
  it can copy `packages/ragcore`) and `rag-web` (build context = `web/`, `VITE_*`
  values are **build arguments** — build with an empty `VITE_API_BASE_URL` so the SPA
  is same-origin through the nginx `/api` proxy, which is the deployed configuration).
* Entra app registration: directory GUID and API client id.

### Run it

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export ENTRA_TENANT_ID=<directory guid>
export ENTRA_CLIENT_ID=<api app registration client id>
# optional: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST, PII_HASH_SECRET

./infra/azure/deploy.sh \
  --env dev --resource-group rg-rag-dev --location westeurope \
  --acr myregistry \
  --api-image myregistry.azurecr.io/rag-api:sha-abc1234 \
  --web-image myregistry.azurecr.io/rag-web:sha-abc1234 \
  --what-if                       # drop to actually deploy
```

Flags: `-e/--env {dev|prod}`, `-g/--resource-group`, `-l/--location`,
`-s/--subscription`, `--api-image`, `--web-image`, `--acr`, `--acr-login-server`,
`--what-if`, `--rotate-secrets`, `--no-deployer-grant`, `-y/--yes`.

`deploy.sh` preflights the CLI, login, subscription, parameter file, images and Entra
identifiers; grants the deploying principal **Key Vault Secrets Officer**; **reuses**
the Qdrant API key and PostgreSQL admin password already in the environment's Key
Vault (generating them only on the very first deployment — this is what makes a
redeploy safe); and uses a *stable* deployment name so repeated runs update one
deployment record. No secret is ever echoed.

### Post-deployment, in order

```bash
# 1. Migrations. PostgreSQL has NO public endpoint by design, so run these from
#    inside the VNet: a Container Apps job, a jump box, or a Bastion session.
uv run alembic -c packages/ragcore/ragcore/db/alembic.ini upgrade head

# 2. Qdrant collections and payload indexes (idempotent).
uv run python scripts/bootstrap_qdrant.py
uv run python scripts/bootstrap_qdrant.py --verify-only   # report without creating

# 3. Ingestion code.
func azure functionapp publish func-rag-<env>-<token> --python

# 4. Prove it end to end, ACL negatives included.
uv run python scripts/smoke_test.py --base-url https://<apiUrl>
```

`bootstrap_qdrant.py --drop-existing` destroys all data and refuses to run when
`RAG_ENV=production` unless `--allow-production` is also passed.

### What gets created

See `infra/azure/README.md` for the full resource, RBAC and naming tables. The shape:
Log Analytics + App Insights; a VNet with four subnets and a PostgreSQL private DNS
zone; a storage account (containers `rag-documents`, `rag-raw`, `rag-manifests`,
`function-releases`, `rag-eval-reports`; queue `rag-ingest` + poison; file share
`qdrant-storage`); an RBAC Key Vault; four user-assigned managed identities;
PostgreSQL Flexible Server v16 with **private access only**; a Container Apps
environment with `api`, `web` (external ingress) and `qdrant` (internal only); and a
Flex Consumption Function App on Python 3.13 with a Durable task hub.

`<token>` is `uniqueString(subscription, resourceGroup, environmentName)`, so names
are stable across redeploys and unique across environments.

### Template outputs → settings

`containerapps.bicep` and `functions.bicep` already wire these into `RAG_*` app
settings; the mapping matters when you are debugging a misconfigured workload:

| Output | Setting |
|---|---|
| `qdrantUrl` | `RAG_QDRANT_URL` — `http://ca-rag-<env>-qdrant.internal.<domain>`, no port, no public FQDN |
| `postgresHost` · `postgresUser` · `postgresDatabase` | `RAG_POSTGRES_HOST` · `_USER` · `_DB` |
| `blobEndpoint` | `RAG_AZURE_BLOB_ACCOUNT_URL` |
| `sourcesContainerName` · `rawContainerName` · `manifestsContainerName` | `RAG_AZURE_BLOB_CONTAINER` · `_RAW_CONTAINER` · `RAG_INGEST_MANIFEST_CONTAINER` |
| `ingestQueueName` | `RAG_AZURE_STORAGE_QUEUE_NAME` |
| `keyVaultUri` | `RAG_AZURE_KEY_VAULT_URL` |
| `apiIdentityClientId` / `ingestionIdentityClientId` | `RAG_AZURE_CLIENT_ID` per workload |
| `apiUrl` · `webUrl` | smoke-test target · MSAL redirect URI |

`prod.bicepparam` sets `ragEnv = 'production'`, which makes `ragcore.settings.Settings`
refuse to construct when `entra_dev_mode` or `tool_allow_insecure_http` is true, and
turns `/docs`, `/redoc` and `/openapi.json` off.

CI compiles the template on every PR (`bicep` job: `az bicep build`, both parameter
files, **linter warnings treated as failures**, and `shellcheck --severity=warning` on
`deploy.sh`).

---

## 2. Ingestion schedule and the working-hours guard

### Configuration

| Setting | Default | Meaning |
|---|---|---|
| `RAG_INGEST_ENABLED` | `true` | Master switch. `false` → every run returns `SKIPPED` with `skip_reason="disabled"`, and `force` does **not** override it. |
| `RAG_INGEST_CRON` | `0 30 2 * * *` | **Six-field NCRONTAB** (`sec min hour day month dow`), as the Functions host requires. Validated by a `Settings` field validator that rejects anything without exactly six fields. Consumed as a binding expression `%RAG_INGEST_CRON%`, so it must exist as an **app setting**, not only in `.env`. |
| `RAG_INGEST_TIMEZONE` | `UTC` | IANA name, validated against the tz database. `functions.bicep` also sets `WEBSITE_TIME_ZONE` from it, so the host schedule and `Settings.is_within_working_hours()` agree. |
| `RAG_INGEST_WORKING_HOURS_START` / `_END` | `8` / `18` | Local to `ingest_timezone`. |
| `RAG_INGEST_WORKING_DAYS` | `[0,1,2,3,4]` | Monday=0 … Sunday=6, JSON list. |
| `RAG_INGEST_FORCE` | `false` | Overrides the working-hours refusal. Never overrides `disabled`. |
| `RAG_INGEST_MAX_PARALLEL_DOCS` | `8` (dev param 4, prod 8) | In-process concurrency inside one activity batch. |
| `RAG_INGEST_BATCH_SIZE` | `32` (dev param 16, prod 32) | Durable fan-out wave size. Travels in the plan envelope, not read during replay. |
| `RAG_INGEST_DELETE_MISSING` | `true` | Enable tombstoning of documents absent from a **full scan**. |
| `RAG_INGEST_REINDEX_ON_ACL_CHANGE` | `true` | An ACL-only change becomes a payload rewrite, no re-embedding. |
| `RAG_INGEST_MAX_DOCUMENT_BYTES` | `26214400` (25 MiB) | Larger documents are skipped with `reason="too_large"`. |
| `RAG_INGEST_RETRY_ATTEMPTS` | `3` | |
| `RAG_INGEST_HTTP_MAX_PAGES` / `_CONCURRENCY` / `_TIMEOUT_SECONDS` | `500` / `4` / `20` | Crawler bounds. |

### The guard

`Settings.may_start_scheduled_ingest(now) -> (allowed, reason)`:

```
not ingest_enabled                         -> (False, "disabled")
not is_within_working_hours(now)           -> (True,  "ok")
ingest_force                               -> (True,  "forced")
otherwise                                  -> (False, "working_hours")
```

`is_within_working_hours` treats a naive `now` as local to `ingest_timezone`, returns
False on a non-working weekday, and handles a window that wraps midnight
(`start=22, end=6`).

A refusal is **recorded, not silent**: `ingest_timer` calls
`run_ingestion(enforce_schedule=True)` so exactly one `IngestRunSummary` with
`status=SKIPPED` and `skip_reason ∈ {"working_hours", "disabled"}` reaches
`ingest_runs`. `run_ingest` never returns an empty list on a refusal, so "refused" and
"nothing to do" are distinguishable.

### Inspecting and changing the schedule

```bash
curl -H "Authorization: Bearer $TOKEN" https://<api>/api/v1/admin/schedule
```

```jsonc
{"ingest_cron": "0 30 2 * * *", "ingest_timezone": "UTC", "ingest_enabled": true,
 "ingest_working_hours_start": 8, "ingest_working_hours_end": 18,
 "within_working_hours": false, "may_start": true, "reason": "ok",
 "next_run_at": "2026-08-03T02:30:00+00:00"}
```

`next_run_at` is computed only for a **simple daily NCRONTAB** (fixed s/m/h, wildcard
date fields); anything else returns `null` rather than a wrong answer.

To change the schedule you redeploy with a different `ingestCron`/`ingestTimezone`
parameter, or set the app setting directly and restart the Function App — the value is
a host binding expression, so a running host does not pick it up live.

---

## 3. Running, monitoring and retrying an ingestion run

### Starting one

| Path | Command | Notes |
|---|---|---|
| Scheduled | — | `ingest_timer` on `%RAG_INGEST_CRON%`, every active tenant. |
| Admin API | `POST /api/v1/admin/ingest/trigger` `{"source_id": null, "force": false, "full_scan": false}` | Requires `rag.admin`. **Runs inline in the API process** via `pipeline.run_ingest` — it does not hand off to the Function App. A large full scan will occupy an API replica for minutes and is rate-limited like any other request. Returns the first `IngestRunSummary`; the rest are in `GET /admin/ingest/runs`. |
| Function HTTP | `POST https://<func>/api/ingest/trigger` | Body adds `wait`, `dry_run`, `enforce_schedule`, `source_ids`. `wait=false` (default) starts the orchestrator and returns Durable status URLs. |
| Blob drop | copy into `rag-documents` | `ingest_blob` indexes it in seconds; the owning source is resolved by longest matching prefix, `.acl.json` sidecars ignored. |
| Queue | enqueue to `rag-ingest` | A serialised `DocumentTask` (retry one document) or `{tenant_id, source_id, source_uri, force?}` (reindex one). |
| Local | `make ingest-local` or `uv run python -m ingestion.cli run --tenant tenant-acme --source-type local --force` | Also `--root`, `--include`, `--exclude`, `--classification`, `--dry-run`, `--full-scan`. Exit codes: `0` ok, `1` failed, `2` refused by the guard. |
| Local, scripted | `uv run python scripts/run_ingest_local.py --tenant tenant-acme --force` | Same pipeline, resolves `ingestion.pipeline.run_ingest`. |

Always start with a dry run against an unfamiliar source:

```bash
uv run python -m ingestion.cli run --tenant tenant-acme --source-id src-x --dry-run
```

It enumerates, resolves ACLs, fetches, parses, **PII-scans** and dedupes for real and
writes nothing. `metrics.dry_run == 1.0`, and `chunks_planned`, `tokens_planned`,
`deletions_planned` tell you what a real run would do. Enrichment is skipped, so it
costs no Claude tokens.

### Monitoring

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "https://<api>/api/v1/admin/ingest/runs?limit=20"
```

Each row is an `IngestRunSummary`: `run_id`, `tenant_id`, `source_id`, `trigger`,
`status` (`RUNNING|SUCCEEDED|PARTIAL|FAILED|SKIPPED`), timings,
`documents_seen/_created/_updated/_deleted/_skipped/_failed`,
`chunks_upserted/_deleted`, `tokens_embedded`, `duplicates_dropped`, `pii_documents`,
`forced`, `within_working_hours`, `skip_reason`, `error_message`, `errors[]`,
`metrics{}`.

`start_run` writes the `RUNNING` row **before** any document is touched, so an
invocation killed mid-flight still leaves an auditable record — a `RUNNING` row with
an old `started_at` and no `finished_at` is an aborted run, not a running one.

Per-document detail lives in `ingest_items` (`source_uri`, `action`, `status`,
`reason`). The reason vocabulary — `new`, `content_changed`, `unchanged`,
`unchanged_touched`, `acl_changed`, `deleted_at_source`, `forced`, `not_modified`,
`reappeared`, `missing_from_full_scan` — is how you tell "nothing changed" from
"nothing was seen".

In Azure, watch the Durable orchestration instance (`ingest_orchestrator`), the
`ingest_document_activity` failure count, and the `rag-ingest-poison` queue depth.

### Retrying

* **One document**: enqueue `{"tenant_id", "source_id", "source_uri", "force": true}`
  to `rag-ingest`, or `POST /api/v1/documents/{id}/reindex` (admin).
* **One document, exact same task**: enqueue the serialised `DocumentTask` from the
  failed activity — `ingest_retry` detects it by the presence of `descriptor` and
  `source` keys and calls `process_document` directly, skipping replanning.
* **One source**: `POST /admin/ingest/trigger {"source_id": "...", "force": true}`.
* **Everything, soundly**: `full_scan: true`. This clears the stored delta cursor so
  enumeration is complete — the only way deletion detection becomes correct again
  after a cursor-based connector has been running.

Retrying is always safe. Document ids are `sha256(tenant \x00 source_uri)[:32]`,
chunk ids are positional, point ids derive from chunk ids, the `documents` row is an
upsert, and manifest folding is last-write-wins per document.

### Manifests

`<tenant_id>/<source_id>.json` in the `rag-manifests` container. Blob writes are
ETag-conditioned and merge-and-retry on a 412. A corrupt or foreign-tenant manifest
**degrades the run to a full rescan** rather than failing it — so if you see an
unexpectedly expensive run, check whether the manifest blob was replaced or truncated.
Deleting a manifest forces a full re-ingest of that source (idempotent, but it will
re-embed everything and cost tokens).

---

## 4. Langfuse: what to look at and what to alert on

Tracing is on when `langfuse_enabled` **and** both keys are present
(`Settings.langfuse_ready`); otherwise `get_tracer` returns a working no-op and the
app never fails because observability is down. `FAILURE_BUDGET = 20` consecutive SDK
failures disables the tracer for the process. `langfuse_sample_rate` is applied per
trace, so at anything below 1.0 your dashboards are sampled and are not an audit log.

### Trace shape

One `chat.turn` trace per turn, tagged `chat`, carrying `user_id`, `session_id`,
`tenant_id` and metadata (`allow_tools`, `stream`, `has_filters`, `message_chars` —
never the message). Nested spans, in order: `memory_load`, `query_transform`,
`cache_probe`, `rag.retrieve`, `ood_gate`, `contradiction`, `tool_plan`,
`context_assembly`, `generate` (once per tool iteration), `tool.<name>` per dispatch,
`citations`, `output_guard`, `memory_write_back`. Generations are recorded by
`LLMClient` with model, usage, cost and latency.

Scores emitted per turn: `citation_validity`, `groundedness`, `coverage`,
`retrieval_max_score`, `refused`. Evaluation runs add `eval.<metric>` scores on an
`eval.item` trace nested under an `eval.run` trace.

### Dashboards worth building

| Dashboard | Built from |
|---|---|
| **Answer quality** | `groundedness`, `citation_validity`, `coverage` over time, split by tenant. A drop in `citation_validity` without a drop in `retrieval_max_score` means the model stopped citing; the reverse means retrieval regressed. |
| **Refusals** | `refused` score rate, split by the `comment` (`guardrail`, `out_of_domain`). A rising `out_of_domain` rate is usually an ingestion failure, not a user-behaviour change. |
| **Retrieval health** | `retrieval_max_score` distribution; `cache_hit` rate from the `cache_probe` span; `rag.retrieve` span latency by bucket. |
| **Cost** | Generation cost per trace, split by model and by tenant. `cache_read_input_tokens` share of input tokens is the prompt-cache hit rate — see §5. |
| **Tool behaviour** | `tool.<name>` span count, error rate and latency; iterations per turn. |
| **Latency** | `chat.turn` p50/p95 and the per-stage breakdown. Stage 9 (`context_assembly`) is dominated by `count_tokens` round trips; stage 5 by embedding + Qdrant. |

### Alert on

1. **`refused` rate > baseline** for a tenant — most often "the nightly ingestion
   silently did nothing", which the OOD gate correctly reports as "outside the indexed
   corpus".
2. **`groundedness` below `guardrail_min_groundedness` (0.6) on a rising fraction of
   turns** — retrieval or prompt regression.
3. **Any stage-12 clearance violation.** This is a log line, not a metric: search for
   `error` level with the clearance-violation event from
   `output_guard.check_clearance`. If it ever fires, `build_acl_filter` is broken.
4. **Any `retrieval.acl_rejected`** error log from `retriever._candidate` — same
   meaning, one layer earlier.
5. **`pipeline_stage_degraded`** warnings, especially `stage=retrieve` or
   `stage=context_assembly`.
6. **`persistence_degraded` / `database_unavailable`** — turns are being answered but
   not recorded.
7. **`entra_dev_principal_accepted`** in any environment you did not intend.
8. **Ingestion**: `status IN ('FAILED','PARTIAL')` in `ingest_runs`; a `RUNNING` row
   older than the expected run duration; `documents_seen == 0` on a source that should
   have content; a non-empty poison queue.
9. **Cost per turn** above a per-tenant ceiling.

Once `prometheus-client` is installed, the equivalent metric alerts are
`rag_guardrail_events_total{action="block"}`, `rag_cache_lookups_total{result="hit"}`
over the sum, `rag_llm_cost_usd_total`, `rag_ingest_runs_total{status="failed"}` and
`rag_pipeline_stage_duration_seconds`. Latency is passed in **milliseconds** and
recorded in **seconds**; route labels are route templates, never rendered paths.

### Health endpoints

`/health` and `/livez` probe nothing — they answer as long as the process is up.
`/readyz` probes **Qdrant and PostgreSQL concurrently**, each bounded by
`api_readiness_timeout_seconds = 5.0 (constant)`, and answers 503 when either fails
with a per-dependency `checks`/`detail` breakdown. Gate traffic on `/readyz`, not
`/health`.

Startup deliberately **never aborts** (except `assert_dev_mode_allowed`): a container
that will not start cannot serve `/health` or be diagnosed. Each startup step's verdict
lands in `app.state.startup` and is re-probed by `/readyz`.

---

## 5. Scaling and cost

### Qdrant must not scale to zero

`qdrant.bicep` pins `minReplicas: 1` and `maxReplicas: 1`, in both dev and prod, and
this is not a sizing choice:

* **`minReplicas` ≥ 1** because a cold start drops the HNSW index from memory. The
  next retrieval would either stall for the rebuild or return nothing, and "returns
  nothing" is indistinguishable from "the corpus does not cover this" — the OOD gate
  would confidently tell the user their question is out of domain.
* **`maxReplicas` = 1** because two replicas mounting the same Azure Files share
  would corrupt the write-ahead log. Horizontal scale means a real Qdrant cluster with
  one volume per node, which this template deliberately does not pretend to do.
* **Persistence is mandatory.** `/qdrant/storage` is an Azure Files volume backed by
  the `qdrant-storage` share (32 GiB dev, 256 GiB prod). Without it a new revision
  starts with empty collections and every retrieval silently returns nothing.

Qdrant is internal-only: `http://ca-rag-<env>-qdrant.internal.<defaultDomain>`, no
public FQDN. Sizing: 1 vCPU / 2 GiB dev, 2 vCPU / 4 GiB prod,
`qdrantMaxOptimizationThreads` 1 / 2.

### The other components

| Component | Dev | Prod | Scaling shape |
|---|---|---|---|
| API Container App | 1 vCPU / 2 GiB, 1–2 replicas, 20 concurrent requests | 2 vCPU / 4 GiB, 2–10 replicas | Scales on concurrent requests. Each replica loads bge-m3 + bm25 + the cross-encoder into memory — that is most of the 2 GiB floor. Mount or bake `RAG_EMBEDDING_CACHE_DIR` or every cold start re-downloads ~2.5 GB of weights. |
| Web Container App | 0.25 vCPU / 0.5 GiB, 1–2 | 0.5 / 1 GiB, 2–5 | Static nginx-unprivileged on port 80. |
| Function App | Flex Consumption FC1, 2048 MB, max 40 instances | 4096 MB, max 100 | Scales to zero between nightly runs. `ingest_batch_size` bounds the Durable fan-out; `ingest_max_parallel_docs` bounds concurrency inside one activity. |
| PostgreSQL | `Standard_B2s` Burstable, 32 GiB, 7-day backups | `Standard_D2ds_v5` GeneralPurpose, 128 GiB, zone-redundant HA, 35-day geo-redundant backups | Private access only. |
| Redis | **none** | **none** | `RAG_REDIS_ENABLED=false` on both workloads. The session window and the rate limiter degrade to in-process, which means **per-replica** rate limiting and a session window that does not survive a replica switch (PostgreSQL re-hydration is authoritative, so correctness holds; the Redis fast path just misses). Add Azure Cache for Redis and flip the setting if you need cross-replica coordination. |

The two always-on pieces are Qdrant (one replica, by design) and the API minimum
replica. Everything else is consumption-priced. Log Analytics is capped at 1 GB/day in
dev and uncapped in prod.

### Token cost drivers, in order

Rates (`ragcore.llm.pricing`, overridable via `RAG_ANTHROPIC_PRICE_PER_MTOK`), USD per
million tokens:

| Model | Input | Output | Cache read (0.1×) | Cache write (1.25×) |
|---|---|---|---|---|
| `claude-opus-5` (MODEL_MAIN) | 5.00 | 25.00 | 0.50 | 6.25 |
| `claude-sonnet-5` (MODEL_FAST) | 3.00 | 15.00 | 0.30 | 3.75 |
| `claude-haiku-4-5` (MODEL_CHEAP) | 1.00 | 5.00 | 0.10 | 1.25 |

1. **The answer prompt on `claude-opus-5`.** Up to 104 000 prompt tokens
   (`context_prompt_budget_tokens`), of which the retrieved-source block is budgeted at
   55 % (~57 200). This dominates everything else by an order of magnitude.
2. **The tool loop.** Every iteration re-sends the whole prompt. `tool_max_iterations`
   is 6, so a worst-case turn is 7 full prompts.
3. **Contradiction adjudication** — up to `guardrail_contradiction_max_pairs = 4
   (constant)` structured `MODEL_MAIN` calls per turn, each carrying two 600-character
   passages. Turn it off with `RAG_GUARDRAIL_CONTRADICTION_ENABLED=false` if your
   corpus does not have genuine version conflicts.
4. **Ingestion enrichment** — one `MODEL_FAST` structured call **per distinct content
   hash** (summary, keywords, doc_type), memoised by `EnrichmentCache`. A re-ingest of
   unchanged content costs nothing because phase-2 delta skips before enrichment.
5. **Query transform** — one `MODEL_FAST` structured call per turn.
6. **Memory write-back and summarisation** — `MODEL_FAST`, per turn / per compaction.
7. **`count_tokens`** — free, but it is a network round trip; `context_token_count_concurrency
   = 8 (constant)` bounds the parallelism and results are memoised by content digest.
8. **Classification calls** (`MODEL_CHEAP`, `max_tokens=256`) — negligible, and both
   optional classifiers (injection, OOD) are off by default.

### How caching cuts them

**Prompt caching** is the main lever. `cache_control={"type": "ephemeral"}` goes on the
last system block, and a second breakpoint on the last stable history message once
that prefix exceeds 1 024 measured tokens. Cache reads bill at **0.1×** and writes at
1.25×, so a stable prefix pays for itself after two turns.

This only works because of a design rule that is easy to break: **volatile content
never enters the cached prefix.** Sources, memory, the rolling summary, preferences,
contradiction notes and the question all live in the *final user turn*
(`prompts.build_answer_user_turn`). Render order is `tools` → `system` → `messages`;
anything before a breakpoint that changes per request destroys the cache. If you add
per-request text to the system prompt, the cache silently stops working and your cost
roughly triples — verify with `response.usage.cache_read_input_tokens`, which the
platform surfaces as `ContextStats.cache_read_tokens` on the `context_stats` SSE event
and as `cache_read_tokens` on the `usage` event.

The **semantic cache** cuts retrieval cost, not token cost: a hit at cosine ≥ 0.94 with
a matching filter fingerprint skips embedding and the Qdrant hybrid query, but the
answer is still generated. It never caches an answer, on purpose — an answer cannot be
re-authorised, a chunk id can.

The **enrichment cache** (keyed by `content_sha256`) and the two delta tiers are what
keep a nightly re-run of an unchanged corpus at approximately zero token cost.

Other levers: `RAG_RETRIEVAL_TOP_N` (8) and `RAG_RETRIEVAL_SNIPPET_CHARS` (1200)
directly size the source block; `RAG_CONTEXT_RETRIEVED_BUDGET_RATIO` (0.55) caps it;
`RAG_ANTHROPIC_EFFORT` (`high`) can be lowered; `RAG_RERANK_ENABLED=false` saves CPU
and latency but not tokens.

---

## 6. Backup and restore

### Qdrant

Qdrant's own snapshot API is the supported path; the Azure Files share is the durable
substrate.

```bash
# Snapshot each collection (QDRANT_URL is internal-only — run from inside the VNet)
for c in rag_chunks rag_memories rag_semantic_cache; do
  curl -X POST -H "api-key: $QDRANT_API_KEY" "$QDRANT_URL/collections/$c/snapshots"
done
# Snapshots are written to /qdrant/storage/snapshots, which is on the same share.
```

`qdrant.bicep` sets the snapshots path to `/qdrant/storage/snapshots`, so snapshots
live on the same Azure Files share as the data. **Copy them off that share** — an
Azure Files backup or a `az storage file copy` into a separate account — or a share
failure loses both.

Restore: stop the Container App revision, restore the share (Azure Files soft delete /
share snapshots), or start a revision and `PUT /collections/{name}/snapshots/recover`.
Then always re-run `scripts/bootstrap_qdrant.py --verify-only` to confirm the payload
indexes and especially that `tenant_id` still carries `is_tenant=True`.

**The cheap alternative: rebuild.** The vector store is fully derivable. Delete the
collections, `bootstrap_qdrant.py`, then run ingestion with `full_scan: true` against
every source. That costs embedding time (CPU, no API spend — FastEmbed is local) and
re-runs enrichment for content whose hash is not already in the `documents` table.
For a corpus that is small enough, this is a better recovery story than snapshot
juggling because it also repairs any drift between Qdrant and PostgreSQL.

### PostgreSQL

Flexible Server automated backups: 7 days dev, **35 days geo-redundant** prod. Restore
is point-in-time through `az postgres flexible-server restore`, which creates a **new
server** — you then repoint `RAG_POSTGRES_HOST` and redeploy, or update the app
setting and restart.

For a logical export (schema migration rehearsal, cross-environment copy), run
`pg_dump` from inside the VNet; there is no public endpoint to dump from outside.

What is in PostgreSQL that is *not* derivable and therefore genuinely needs the
backup: `chat_sessions`, `chat_messages`, `tool_invocations`, `user_profiles`,
`user_memories`, `feedback`, `audit_log`, `eval_runs`, `eval_results`. What *is*
derivable: `documents`, `ingest_runs`, `ingest_items`, `lineage_records` (from a
re-ingest), `semantic_cache_meta`.

### Blob

`rag-documents` holds the source corpus and `rag-raw` the archived raw copies — enable
soft delete and versioning on the account. `rag-manifests` is derivable: deleting a
manifest forces a full rescan of that source, which is a valid (if expensive) repair.

### Order of restore

1. PostgreSQL (the `documents` table is what tells ingestion what it already knows).
2. Blob containers.
3. Qdrant — snapshot restore, or bootstrap + full-scan re-ingest.
4. `scripts/bootstrap_qdrant.py --verify-only`.
5. `scripts/smoke_test.py --base-url …` — it includes the ACL negatives, which is
   exactly what you want to re-verify after a restore.

---

## 7. Runbook: the five most likely failures

### 7.1 Every answer is "outside the indexed corpus"

**Symptom.** `refused` rate spikes; the OOD gate's refusal names little or no
coverage; `retrieval_max_score` near zero.

**Almost always one of:**

| Cause | Check | Fix |
|---|---|---|
| Qdrant restarted without its volume | `/readyz` shows `qdrant: true` but `GET /collections` shows 0 points | Restore the share, or bootstrap + full-scan re-ingest. Confirm the `qdrant-storage` volume mount survived the revision. |
| Collections exist without payload indexes | `scripts/bootstrap_qdrant.py --verify-only` | Re-run `bootstrap_qdrant.py` (idempotent). |
| `tenant_id` index lost `is_tenant=True` | grep logs for `qdrant.index.tenant_not_partitioned` | The collection must be recreated; an index cannot be converted in place. |
| Nightly ingestion silently did nothing | `GET /admin/ingest/runs` — `status=SKIPPED`, `skip_reason` | `disabled` → `RAG_INGEST_ENABLED=true`. `working_hours` → the cron drifted into the window, or `ingest_timezone` and `WEBSITE_TIME_ZONE` disagree. |
| Sparse vector created without `Modifier.IDF` | collection info shows no `modifier` on `sparse` | Recreate the collection. Symptom is subtler: hybrid results are poor rather than empty. |
| The user genuinely narrowed the filter to nothing | `filter_applied` on the `retrieval` SSE event | Nothing to fix. |

The OOD refusal is deliberately informative: it names what *is* indexed, sampled
**through `build_acl_filter`**, so it never advertises a document type the caller
cannot see. `clear_coverage_cache()` after a bulk ingest — the coverage summary is
cached for `guardrail_ood_coverage_ttl_seconds = 900 (constant)`.

### 7.2 An ingestion run failed or is stuck

**Symptom.** `ingest_runs` shows `FAILED`/`PARTIAL`, or a `RUNNING` row with an old
`started_at` and no `finished_at`.

1. `GET /admin/ingest/runs` → read `error_message` and `errors[]`. Remember these are
   **exception type names only** (plus the message for `RagError`), by design — the
   detail is in Application Insights / Log Analytics, not the run row.
2. `ingest_items` for that `run_id` → which documents, which `action`, which `reason`.
3. `PARTIAL` means some documents failed: retry just those by enqueueing their
   `DocumentTask` or `{tenant_id, source_id, source_uri}` to `rag-ingest`.
4. A stuck `RUNNING` row is an aborted invocation. Check the Durable instance status;
   Durable will retry activities itself. If the orchestration is genuinely dead,
   re-trigger the source — every write is idempotent, so a re-run repeats work rather
   than duplicating it.
5. `plan.enumeration_failed` in the logs means the connector broke during listing.
   `plan_source` captures it in `error_message` rather than raising, so one broken
   source cannot abort a multi-source nightly run — but that source did nothing.
6. Poison queue non-empty → messages that failed repeatedly. Inspect, fix the cause,
   re-enqueue.

**Do not** delete the manifest as a first move. An absent manifest forces a full
rescan, which re-embeds everything; it is a repair, not a diagnostic.

### 7.3 Documents deleted at source are still being returned

**Expected.** Deletion detection requires a **full scan**. A SharePoint `deltaLink`
pass or a SQL watermark pass deliberately does not mention unchanged documents, so
`detect_deletions` returns `[]` unless `connector.performed_full_scan` is true.

Fix: `POST /admin/ingest/trigger {"source_id": "...", "full_scan": true}`, or
`ingestion.cli run --full-scan`. That clears the stored cursor so enumeration is
complete.

Also check `RAG_INGEST_DELETE_MISSING` is `true`. And note the guardrail in
`tombstone_missing`: an **empty** `manifest_document_ids` tombstones everything for
that tenant/source and logs a warning, because that is also what a broken connector
looks like.

### 7.4 Cost per turn jumped

1. **Check the prompt cache first.** `usage.cache_read_tokens` on the `usage` SSE
   event, or `cache_read_input_tokens` in Langfuse. A hit rate that fell to zero means
   something volatile entered the cached prefix — a changed system prompt, a per-turn
   value injected into `system`, or a `PROMPT_VERSION` bump (which is intended to
   invalidate, once).
2. **Check tool iterations.** More iterations = more full prompts. Look at the
   `tool.<name>` span count per trace and at `LoopGuard` warnings
   (`tool_loop_guard_stop`, `tool_calls_not_dispatched`).
3. **Check the source block size.** `ContextStats.retrieved_tokens` against the ~57 200
   sub-budget. A rise usually means chunking changed (`chunk_target_tokens`) or
   `retrieval_snippet_chars` was raised.
4. **Check contradiction adjudication.** Up to 4 `MODEL_MAIN` calls per turn. A corpus
   that suddenly has many near-duplicate versioned documents will trip it constantly.
5. **Check enrichment.** One `MODEL_FAST` call per *distinct content hash*. If a source
   is producing a new hash for unchanged content (a timestamp in the body, a re-export
   with different whitespace), every run re-enriches everything. `normalise_text` in
   the dedupe layer absorbs whitespace and Unicode composition changes, but not an
   embedded date.

### 7.5 The chat stream hangs, or tokens arrive in one lump

| Symptom | Cause | Fix |
|---|---|---|
| Tokens arrive all at once at the end | A proxy is buffering | The API sets `x-accel-buffering: no` and `cache-control: no-transform`; `web/nginx.conf` sets `proxy_buffering off` on `/api/`. Check any proxy *between* them — App Gateway, Front Door, a corporate proxy. |
| Stream opens then silently stops | Idle timeout | A heartbeat comment goes out every `api_sse_keepalive_seconds = 15`. Confirm the intermediary's idle timeout exceeds that and that comments are not stripped. |
| Client hangs forever waiting for `done` | Would be a bug — every exit path emits `context_stats`, `usage` and `done` | Check for an `error` event immediately before; a failure after the first byte emits `error` then `done`, because a half-open stream has no status code left. |
| 401 mid-stream | Token expired between the initial fetch and a retry | The client retries a request exactly once on 401 after a forced MSAL silent refresh, and only while **no `token` event has arrived**. After the first token it fails the turn and offers an explicit resend — there is no resume-from-offset contract. |
| Very slow first token | Model warm-up on a cold API replica | `api_warm_models = True (constant)` loads FastEmbed dense/sparse/rerank during startup. If the weights cache is not mounted, that is a ~2.5 GB download per cold start. Mount `RAG_EMBEDDING_CACHE_DIR`. |

### Honourable mentions

* **`/metrics` returns one comment line.** Not a failure — `prometheus-client` is not
  installed. Add it to `packages/ragcore/pyproject.toml` dependencies.
* **`POST /eval/runs` returns 503 `eval_unavailable`.** `eval.harness.run_evaluation`
  does not exist. See [EVALUATION.md](EVALUATION.md).
* **`make bootstrap` / `make seed` / `make smoke` fail with "no such file".** The
  Makefile points at `scripts/bootstrap.py`, `scripts/seed.py`, `scripts/smoke.py`;
  the real names are `bootstrap_qdrant.py`, `seed_demo_tenant.py`, `smoke_test.py`.
* **API refuses to start with a `ConfigError` about dev mode.** `RAG_ENTRA_DEV_MODE` is
  true with `RAG_ENV=production`. This is the one startup check permitted to abort the
  process, and it is correct.
