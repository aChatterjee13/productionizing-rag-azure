# Security

What the code enforces, where, and what it does not. Function names are given so a
claim can be checked against the source rather than believed.

This system has **not** been through a production security review. The
[Residual risks](#residual-risks) section is deliberately long and is the most
important part of this document.

---

## 1. Threat model

### Assets

| Asset | Where |
|---|---|
| Tenant-scoped document content | Qdrant `rag_chunks` payload `text`, Blob raw copies, `documents` rows |
| Classified content (`confidential`, `restricted`) | same, gated by `classification_rank` |
| Personal data in documents and conversations | chunk text, `chat_messages`, `user_memories`, `feedback` |
| Per-user long-term memory and profile | Qdrant `rag_memories`, `user_profiles` |
| Credentials | Key Vault: Anthropic key, Qdrant key, PostgreSQL password, Langfuse keys, PII HMAC secret, per-tool secrets |
| The corpus itself | as a competitive asset and as an injection surface |

### Actors and what they can try

| Actor | Capability | Primary threat |
|---|---|---|
| Authenticated user, tenant A | Any API call with a valid Entra JWT | Reach tenant B's data; reach material above their clearance; reach a document their group is denied |
| Authenticated user, low clearance | Same | Extract restricted content indirectly — through a summary, a citation span, a tool argument, a semantic-cache reuse, or an error message |
| Tenant administrator (`rag.admin`) | Admin surface, uploads, reindex, ingest trigger | Enumerate other tenants; exfiltrate through a tool config; escalate clearance |
| Document author / content supplier | Can put text into an indexed document | **Indirect prompt injection**: make the assistant ignore instructions, exfiltrate other retrieved context, or misuse a tool |
| External REST/MCP endpoint | Returns tool responses | Same injection surface, plus a hostile response body |
| Network attacker | Sees traffic, may reach an endpoint | Token theft, replay, MITM on tool egress |
| Operator error | Misconfiguration | `entra_dev_mode` in production; a tool pointed at a public host; an over-broad ACL default |

### Explicit non-goals in this codebase

* No rate limiting or abuse detection on cost (a user can drive Claude spend up to the
  per-minute request limit, not a token budget).
* No data-residency or per-tenant encryption key separation. One Qdrant collection,
  one PostgreSQL database, one Blob account, shared by all tenants.
* No audit of *reads*. `audit_log` records seven mutating actions (below); a retrieval
  is traced but not audited.
* No key rotation automation beyond `deploy.sh --rotate-secrets`.

---

## 2. Tenant isolation, and exactly where it is enforced

The tenant boundary is enforced at **six** places. Only the first is load-bearing;
the rest exist so that a bug in one is visible against another.

### (1) `ragcore.vectorstore.filters.build_acl_filter` — the chokepoint

`packages/ragcore/ragcore/vectorstore/filters.py`. This is the only supported way to
produce a filter for `rag_chunks`, and it is **not parameterised by tenant**: the
tenant comes from the `Principal`, so no caller can substitute one. Its siblings:

| Function | Purpose |
|---|---|
| `build_acl_filter(principal, extra, *, include_deleted=False)` | Every chunk read. |
| `build_acl_filter_for_chunk_ids(principal, chunk_ids, extra)` | The semantic-cache re-fetch (stage 4) and the MMR vector read-back. Composes the full ACL filter and adds a `HasIdCondition` over derived point ids. An empty `chunk_ids` matches nothing. |
| `build_memory_filter(principal, kinds, …)` | `rag_memories`: tenant **and** `user_id` both in `must`. |
| `build_cache_filter(principal, fingerprint)` | `rag_semantic_cache`: tenant + exact fingerprint in `must`; `should` matches the caller's own entries or tenant-wide (`user_id` empty). |
| `build_tenant_filter(tenant_id, …)` | **Writes only.** Ingestion and admin. No deny list, no permissive branch, no clearance. It lives in this module so nothing outside it ever constructs a `qm.Filter`, but it must never serve a read to a principal. |
| `filter_fingerprint(principal, extra)` | Cache key: `sha256(canonical_json({version, tenant_id, clearance_rank, filter}))[:32]`. |
| `classification_ceiling(principal, extra)` | Typed view of the effective ceiling, so stage 12 re-checks against exactly the ceiling the filter used. |

Proven by `packages/ragcore/tests/test_filters.py` (28 tests), including that a
tenant-A principal can never match a tenant-B point and that `denied_users` overrides
an otherwise-matching group.

### (2) `ragcore.models.acl.AccessControl.permits` — the in-process mirror

A second, independent implementation of the same rule, written so a bug in one is
visible against the other. Used by:

* `app/rag/retriever._candidate` — every candidate returned by Qdrant is re-checked
  (`payload.tenant_id == principal.tenant_id` **and** `permits(principal)`). A
  failure is counted, logged at **error** level with `chunk_id`/`document_id` only,
  and the chunk is discarded. Reaching that branch means `build_acl_filter` is broken.
* `app/rag/guardrails/output_guard.check_clearance` — stage 12, below.
* `app/routers/documents.list_documents` — SQL narrows by tenant, tombstone and
  clearance rank; role/group/deny are then applied in process, because JSON
  containment differs between PostgreSQL and SQLite. The listing over-fetches ×4 and
  truncates so a page is never short, and ACL lists are not projected into the
  response.
* `services/eval/metrics.acl_leak` — the golden set's ACL negatives.

### (3) `ragcore.db.repositories` — SQL

Every repository function takes `tenant_id` and filters on it. A fetch-then-check that
finds a row under a *different* tenant raises `TenantMismatchError` rather than
returning `None`, so a cross-tenant probe is auditable. `upsert_document` additionally
rejects an `access_control` whose `tenant_id` disagrees with the `tenant_id` argument.

### (4) `ragcore.vectorstore.writer` — writes

`upsert_chunks` raises `TenantMismatchError` when a batch spans tenants — a batch is
the retry unit, and a partially-correct retry is how cross-tenant rows appear.
`update_access_control` raises when `access_control.tenant_id` disagrees.
`hard_delete_by_filter` refuses a filter with no `must` clause, so an unscoped purge
is impossible.

### (5) Ingestion — `ingestion.pipeline._assert_tenant`

An ACL resolved from a sidecar, blob metadata, Graph permissions or a SQL row that
declares a foreign `tenant_id` raises `ValueError` and fails the document. It is never
"corrected". `SqlSourceConnector` drops a row whose `tenant_column` disagrees with the
source rather than indexing it.

### (6) Keys and cached state

* Session-window store key is `<prefix><tenant_id>:<session_id>` — tenant first — and
  `ShortTermMemory.load` **discards and logs** a cached window whose `tenant_id`
  disagrees with the principal.
* `SemanticCacheEntry.make_cache_id` is keyed on the tenant, so an identical query in
  two tenants can never share an entry, and `filter_fingerprint` includes the tenant
  again as defence in depth.
* Rate-limit keys are `<prefix><tenant_id>:<user_id>` so one tenant's traffic cannot
  evict another's counters in a shared Redis.
* Memory deletes are ownership-checked: a Qdrant point id carries no tenant, so
  `LongTermMemoryStore` retrieves the point and verifies its payload's
  `tenant_id`/`user_id` before removing it.

### Where tenancy is *not* enforced

* `GET /admin/tenants` returns **only the caller's own tenant** by design — a tenant
  administrator has no business enumerating a shared deployment.
* `GET /admin/sources` omits `options` and `cursor`: an option can name a secret, a
  cursor can embed a Graph `deltaLink` token.
* `GET /documents/{id}/*` answers **404 `document_not_found`** for absent,
  foreign-tenant and not-permitted alike. Distinguishing them would confirm a document
  exists to someone who may not know it does. `GET/DELETE /sessions/{id}` likewise
  answers 404, not 403.

---

## 3. ACL filter semantics

`build_acl_filter` composition:

```
must     = tenant_id == principal.tenant_id
         + is_deleted == false                      (unless include_deleted)
         + classification_rank <= effective_clearance
         + every MetadataFilter clause
must_not = denied_users contains principal.user_id
should   = allowed_users contains principal.user_id
         | (allowed_roles IS EMPTY and allowed_groups IS EMPTY and allowed_users IS EMPTY)
         | allowed_groups matches any of principal.groups     (omitted if no groups)
         | allowed_roles  matches any of principal.roles      (omitted if no roles)
min_should = MinShould(conditions=should, min_count=1)
```

`must_not` and `should` are ANDed by Qdrant, so satisfying the permissive branch
cannot rescue a denied user. `min_should` is redundant with `should` (Qdrant already
requires one `should` to match) but is stated explicitly so a future edit turning
`should` into an AND fails the unit test that pins `min_count == 1`.

The permissive branch omits the group/role arms when the principal has none, rather
than sending an empty `MatchAny`.

### Truth table

`R` = the chunk's `allowed_roles`, `G` = `allowed_groups`, `U` = `allowed_users`,
`D` = `denied_users`. "Match" means non-empty intersection with the principal's
corresponding claim (or `principal.user_id ∈ U`).

| # | Tenant | `is_deleted` | `rank ≤ clearance` | `user ∈ D` | R,G,U all empty | U match | G match | R match | Visible |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ≠ | – | – | – | – | – | – | – | **no** |
| 2 | = | true | – | – | – | – | – | – | **no** |
| 3 | = | false | no | – | – | – | – | – | **no** |
| 4 | = | false | yes | **yes** | yes | – | – | – | **no** — deny beats unrestricted |
| 5 | = | false | yes | **yes** | no | yes | yes | yes | **no** — deny beats every allow |
| 6 | = | false | yes | no | **yes** | – | – | – | **yes** — unrestricted in tenant |
| 7 | = | false | yes | no | no | **yes** | no | no | **yes** |
| 8 | = | false | yes | no | no | no | **yes** | no | **yes** |
| 9 | = | false | yes | no | no | no | no | **yes** | **yes** |
| 10 | = | false | yes | no | no | no | no | no | **no** — restricted, no arm matched |

Rows 4 and 5 are the ones worth testing, and are: `doc-acme-contractor-nda` in the
demo fixture grants `g-acme-engineering` **and** denies `acme_engineer`, who is in
that group. `packages/ragcore/tests/test_filters.py` asserts it at the filter level;
the golden set asserts it end to end via canary tokens.

### Effective clearance

`_effective_clearance(principal, extra) = min(principal.clearance_rank(),
extra.max_classification.rank)`. A `MetadataFilter` may only **narrow**: a caller
sending `max_classification: restricted` cannot widen their own clearance.

---

## 4. Classification and clearance

`Classification` is an ordered `StrEnum` with `__lt__/__le__/__gt__/__ge__` overridden
to compare by `.rank`, so `PUBLIC(0) < INTERNAL(1) < CONFIDENTIAL(2) < RESTRICTED(3)`,
`sorted()` yields least-to-most sensitive and `max(a, b)` picks the stricter label.
`str` behaviour is preserved (`Classification.PUBLIC == "public"`).
`classification_rank` is denormalised onto the chunk payload so Qdrant can range-filter
it, and a model validator re-derives it from `classification` so the two cannot
disagree.

### Deriving a principal's clearance

`app/auth/principal.max_classification_for(roles, groups, settings)`: the **strongest
matching** rule across roles and groups wins; when nothing matches,
`entra_default_classification` applies. So a role mapped to `public` genuinely caps
that caller below the baseline, and holding `rag.admin` alongside it still yields
`restricted`.

Default map (all `(constant)` — see the configuration caveat in §10):

| Role | Ceiling |
|---|---|
| `rag.admin` | `restricted` |
| `rag.restricted` | `restricted` |
| `rag.confidential` | `confidential` |
| `rag.internal` | `internal` |
| `rag.public` | `public` |
| *(no match)* | `internal` (`entra_default_classification`) |

`entra_clearance_groups` is empty by default — group ids are deployment-specific.

### Where the ceiling is enforced

1. **Qdrant** — `classification_rank <= clearance` in `must`.
2. **In-process, post-retrieval** — `AccessControl.permits` in `retriever._candidate`.
3. **SQL** — `Document.classification_rank <= principal.clearance_rank()` in
   `GET /documents`.
4. **Upload** — `POST /documents` clamps the requested classification to
   `min(requested, principal.max_classification)` and logs
   `upload_classification_clamped`. Nobody creates material they cannot read back.
5. **Tool egress** — `RegisteredTool.may_receive(context_classification)`;
   `RESTRICTED` never leaves the platform for a non-`retrieval` tool, whatever the
   config says (`NEVER_FORWARD`).
6. **Output guard (stage 12)** — `check_clearance` re-tests every cited chunk with
   `AccessControl.permits` **and** scans the answer for a verbatim overlap of
   `guardrail_output_leak_span_chars = 60 (constant)` characters with over-clearance
   text. A violation **fails closed**: the answer is replaced with
   `CLEARANCE_BLOCK_MESSAGE`, the offending citations move to `dropped_citations`, and
   the event is logged at `error` level naming the chunk, the document and both ranks.
   If this ever fires, `build_acl_filter` is broken and the log line says so.
   `guardrail_enforce_classification_on_output=false` downgrades the action to `warn`
   and leaves the answer alone — that is an operator's decision, not a default.

---

## 5. Authentication

`app/auth/entra.py`.

* **Algorithm is checked before decoding.** `jwt.get_unverified_header` is the single
  pre-verification read; the `alg` is matched against `entra_allowed_algorithms`
  before `jwt.decode` runs, so `alg: none` and every symmetric algorithm are refused
  outright rather than relying on the library default. Algorithm confusion is the
  classic JWT break.
* `jwt.decode` requires `exp`, `iss`, `aud`; validates signature against the cached
  JWKS, `iss` (v2.0 and optionally the v1.0 `sts.windows.net` form), `aud`
  (`entra_audience` or `entra_client_id`), `exp`/`nbf` with
  `entra_leeway_seconds = 60`.
* **Only `jwt.decode` output is read as claims.** Nothing anywhere reads an unverified
  payload.
* An unknown `kid` triggers exactly one refetch, floored by
  `entra_jwks_refresh_min_seconds = 60 (constant)` — without a floor, a stream of
  tokens signed by a key that will never exist becomes a DoS on the Microsoft
  endpoint. A failed refresh with keys already cached degrades to the cache and warns;
  with no keys it is a 503.
* **Group overage fails closed.** When the token carries `hasgroups` instead of
  `groups`, the validator resolves them through Microsoft Graph. An unreachable Graph,
  a missing application identity, or any error returns an **empty group list**, so the
  caller sees only unrestricted documents — never "all".
* Error codes are enumerated and never echo the token: `auth_missing_token`,
  `auth_malformed_token`, `auth_bad_algorithm`, `auth_unknown_key`,
  `auth_bad_signature`, `auth_token_expired`, `auth_token_immature`,
  `auth_bad_audience`, `auth_bad_issuer`, `auth_bad_tenant`, `auth_missing_scope`,
  `auth_missing_role`, `auth_not_configured`, `auth_jwks_unavailable`,
  `auth_dev_principal_invalid`.

### Dev mode

`entra_dev_mode=true` accepts an unsigned `Principal` JSON in the
`x-dev-principal` header. It is defended three ways:

1. `Settings._validate_consistency` **refuses to construct** when `entra_dev_mode` and
   `env == "production"` — so an environment mutated after deploy fails loudly.
2. `assert_dev_mode_allowed(settings)` in the FastAPI lifespan is the one startup
   check permitted to abort the process.
3. Every accepted dev principal logs `entra_dev_principal_accepted` at **warning**
   level with the env and header name.

`bicepparam` sets `ragEnv = 'production'` for prod, and `containerapps.bicep` wires
`RAG_ENTRA_DEV_MODE` explicitly.

### Authorisation

`app.deps.require_roles(*roles)` always accepts `settings.entra_admin_role`, read at
request time, so no call site writes the admin role as a literal and
`require_roles()` with no arguments is the administrator-only gate. Admin-gated:
`/admin/*`, `DELETE /documents/{id}`, `POST /documents/{id}/reindex`.

---

## 6. PII handling, hop by hop

| # | Hop | What runs | What is stored / sent |
|---|---|---|---|
| 1 | **Ingestion, pre-everything** | `ingestion.enrich.scan_and_redact` runs over every parsed block **before** enrichment, chunking, embedding, logging or lineage | Only redacted text reaches the LLM enrichment call, the chunk payload, the embedding and the Blob-archived derivative. Outcome travels as `ParsedDocument.metadata["pii_types"]`/`["pii_redacted"]` → `ChunkPayload.pii_types`/`pii_redacted`. |
| 2 | **Ingestion errors** | `_redact_error` | `ingest_runs.errors` carries the exception **type** only, plus the message for `RagError` (whose messages are contractually content-free) — an arbitrary exception can quote the document. |
| 3 | **User turn (stage 1)** | `run_input_guard`: NFKC-normalise + invisible-character strip → size cap → redact → scan | Produces **two** strings. `text` is prompt-safe: the user's own words with only credential-shaped entities (`API_KEY`, `JWT`) masked, because redacting "my email is …" out of the question would change what was asked. `redacted_text` is the **only** form that may be logged, traced or persisted. |
| 4 | **Persistence** | `repositories.append_message` / `write_feedback` raise `ValueError` unless `pii_redacted=True` | The flag is an assertion that the pass ran, not a formatting hint. `short_term.record_turn` enforces the same. |
| 5 | **Logging** | `ragcore.logging.guard_raw_content` structlog processor | Any event carrying `text`, `content`, `message`, `query`, `question`, `answer`, `prompt`, `chunk_text`, `user_input`, `document_text` or `snippet` has that value replaced with `***omitted-unredacted***` **unless the same event sets `pii_redacted=True`**. Redact-before-log is a mechanical property of the pipeline, not a rule each call site must remember. |
| 6 | **Tracing** | Content policy in `ragcore.observability` | `LLMClient` sends message counts, roles, block sizes, tool names, betas, token counts and cost — never prompt or answer text, which has not necessarily passed redaction at that point. |
| 7 | **Tool arguments** | `registry.redact_arguments` (two passes: credential-looking keys replaced outright, then every remaining string through the detector) | The only shape of arguments ever logged, traced, streamed on the `tool_call` SSE event, or persisted to `tool_invocations`. |
| 8 | **Tool responses** | `RestExecutor` PII-scans the body before it becomes tool-result content (`tool_response_pii_scan=true`) | Then truncated to the tool's `max_result_chars`. |
| 9 | **Memory write** | `LongTermMemoryStore.remember` PII-scans and redacts **before** embedding or upserting; write-back also redacts the turn text sent to `claude-sonnet-5` | `pii_redacted=True` on a stored memory is an assertion the pass ran. |
| 10 | **Answer egress (stage 12)** | `run_output_guard` PII scan, ignoring `DATE_TIME` and `LOCATION` `(constant)` | Emit `decision.text`; store `decision.redacted_text` with `pii_redacted=True`. An effective date and an office name are policy content — masking them would gut every answer. |
| 11 | **Citation drops (stage 11)** | `CitationDrop` carries **no content** | Stage 11 runs before the stage 12 egress scan, so a failing span is deliberately not recorded, logged or traced. |
| 12 | **Feedback** | `POST /feedback` redacts the comment before `write_feedback` | |

### Detector

`ragcore.pii.PIIDetector`. Presidio (`AnalyzerEngine`) when `presidio-analyzer` is
importable **and** `pii_use_presidio`; otherwise the regex-only recogniser set with a
warning — the package always imports. Presidio is an optional extra
(`uv sync --extra pii`) and **is not installed in this checkout**, so the current
behaviour is regex-only: no NLP-backed `PERSON` or `LOCATION` detection.

Regex recognisers cover `EMAIL_ADDRESS`, `CREDIT_CARD` (Luhn), `IBAN_CODE`
(ISO 7064 mod-97), `US_SSN`, `PHONE_NUMBER`, `IP_ADDRESS`, plus custom `AADHAAR`
(Verhoeff), `PAN`, `SWIFT_CODE` (requires context), `API_KEY` and `JWT` (base64url
header parse). `trim_retry` re-validates progressively shorter prefixes, because
greedy regex matching knows nothing about checksums.

Three redaction modes: `mask` (`<EMAIL_ADDRESS>`), `hash` (stable HMAC keyed on
`pii_hash_secret`, so redacted values still join) and `partial` (last
`pii_partial_keep_chars=4` kept). Redaction is implemented in-process for all three
rather than through Presidio's `AnonymizerEngine`, so output is byte-identical with
and without the extra — `hash` mode is a join key and must not vary by deployment.
`PIIFinding.snippet` is a partially-masked preview, never the matched value, so a
report is safe to log and persist.

The optional LLM verification pass (`pii_llm_verify_enabled`, off by default,
`claude-haiku-4-5`) may only ever **remove** false positives: any failure returns the
input report unchanged, so protection never disappears because a model call failed.

---

## 7. Prompt injection

### Direct (the user turn)

`app/rag/guardrails/injection.scan_text` scores a turn against a pattern table
(`PATTERNS`) combining as noisy-OR, `1 − Π(1 − wᵢ)`, over distinct patterns.
`guardrail_injection_block_threshold = 0.8` blocks with `INJECTION_BLOCK_MESSAGE`;
`guardrail_injection_warn_threshold = 0.5` warns. Patterns are anchored to clause
boundaries or assistant-directed objects so ordinary corpus prose does not trip them
("the runbook explains how to disable access controls" must not quarantine the
runbook). **A signal never carries an excerpt of the match** — it is untrusted content
and signals are logged, traced and streamed before anything redacts them.

The turn is also NFKC-normalised and stripped of invisible characters (zero-width,
bidi controls, the Unicode Tags block) by `sanitise_untrusted` before scanning, so a
homoglyph or tag-block smuggling attempt is normalised away first.

### Indirect (poisoned documents and tool output)

This is the harder half, and it is where the strongest and the weakest parts of the
implementation both sit.

**What runs.** After stage 5, `scan_retrieved(result.chunks)` scores every retrieved
chunk's text with a **stricter** threshold than the user turn:
`guardrail_injection_retrieved_block_threshold = 0.5 (constant)`. A chunk at or above
it is **quarantined** — removed from `result.chunks`, appended to
`RetrievalResult.dropped` with `dropped_reason = "guardrail:injection"`, and surfaced
as a `guardrail` SSE event. Scanning is capped at
`guardrail_injection_max_scan_chars = 20 000 (constant)` per passage. An optional
`claude-haiku-4-5` adjudicator (`guardrail_injection_classifier_enabled=false` by
default) can *attenuate* a false positive on security documentation.

Tool responses go through the same registry-level PII scan and are truncated, and the
`ANSWER_SYSTEM` and tool-routing prompts both instruct the model to treat source text
and tool output as untrusted data, never as instructions.

**⚠ divergence — what does not run.** `docs/CONTRACTS.md` states that "every piece of
retrieved or tool-returned text that enters a prompt is wrapped by `wrap_untrusted`",
described as "structural defence that does not depend on detection working". In the
code, `wrap_untrusted` is called from exactly two places, and neither is the answer
path:

* `app/rag/guardrails/ood.py` — the OOD adjudication prompt;
* `app/rag/guardrails/contradiction.py` — the pairwise conflict adjudication prompt.

Retrieved chunk text reaches the **generation** prompt through
`ragcore.llm.prompts.render_numbered_sources`, which emits a plain `<sources>` block:
no `<<<BEGIN_UNTRUSTED_CONTENT>>>` delimiters, and no `sanitise_untrusted` pass over
the chunk text. A document containing the literal string `</sources>` can therefore
close the block early and have the remainder of its text read as prompt-level
material. Chunk text was invisible-character-stripped at ingestion time only if
`sanitise_untrusted` ran there — it does not; only the user turn is sanitised.

So on the main answer path the injection defence is: quarantine at score ≥ 0.5, plus a
system-prompt instruction. That is meaningfully weaker than the contract claims. The
fix is small — wrap and sanitise inside `render_numbered_sources` (or in
`context._snippet_for`) — but it changes the exact text a citation span is verified
against, so it needs the citation tests re-run alongside.

### Blast radius if injection succeeds

Bounded, but not zero:

* The model cannot widen retrieval: a tool-issued `search_corpus` call is intersected
  with the turn's filter via `MetadataFilter.merged_with` and runs under the same
  `build_acl_filter`, as the same principal. It can only narrow.
* The model cannot reach a tool it is not exposed to (tenant then role), cannot exceed
  `tool_max_iterations=6` or the repeat limit, and cannot forward `RESTRICTED` context
  to any non-retrieval tool.
* It **can** cause an incorrect or manipulated answer, cause a plausible-looking
  fabricated citation (which stage 11 will usually drop, lowering
  `citation_validity`), and cause a REST tool to be called with attacker-influenced
  arguments within that tool's declared schema.

---

## 8. Secret management

**No secret is a template parameter value, an output, or a committed file.**
`params/*.bicepparam` read every secret from the process environment with
`readEnvironmentVariable`, so the parameter files are safe to commit.

Six Key Vault secrets, RBAC-mode vault, soft delete on, purge protection in prod:

| Secret | Consumed as |
|---|---|
| `anthropic-api-key` | `RAG_ANTHROPIC_API_KEY` |
| `qdrant-api-key` | `QDRANT__SERVICE__API_KEY`, `RAG_QDRANT_API_KEY` |
| `postgres-admin-password` | `RAG_POSTGRES_PASSWORD` |
| `langfuse-public-key` / `langfuse-secret-key` | `RAG_LANGFUSE_*` |
| `pii-hash-secret` | `RAG_PII_HASH_SECRET` |

A secret resource is created **only when its parameter is non-empty**, so omitting an
optional secret on a redeploy leaves the stored value alone rather than overwriting it
with a placeholder. Container Apps reference secrets by versionless Key Vault URI
(`secrets[].keyVaultUrl` + `secrets[].identity`); the Function App uses
`@Microsoft.KeyVault(SecretUri=…)` with `keyVaultReferenceIdentity`.

Four user-assigned managed identities (`api`, `web`, `ingest`, `qdrant`) with
least-privilege role assignments — see the RBAC table in `infra/azure/README.md`. The
Function App carries **two** identities: the pre-provisioned user-assigned one for
data-plane access (role assignments must exist before the app first starts) plus a
system-assigned one.

Per-tool secrets are never values: `RestToolSpec.auth_secret_ref` and
`McpServerSpec.authorization_token_ref` name a Key Vault **secret**, resolved by
`SecretResolver` through `azure-keyvault-secrets` and cached for
`tool_secret_cache_seconds=300`, falling back to `$RAG_TOOL_SECRET_<REF>` when Key
Vault is not configured. `entra_obo` and `managed_identity` do a client-credentials
grant / IMDS fetch instead.

`deploy.sh` reads existing Qdrant and PostgreSQL credentials **back out of Key Vault**
on a redeploy and passes the same values, so a redeploy never silently rotates a
credential; `--rotate-secrets` is the explicit opt-in.

**One deliberate exception.** The Azure Files mount backing `/qdrant/storage` needs a
storage account key — Container Apps has no managed-identity option for Azure Files —
so `allowSharedKeyAccess` stays `true` and the key is resolved *inside* `qdrant.bicep`
with `listKeys()`, never as a parameter and never as an output. Every application data
path still uses managed identity. This is a real residual exposure: anyone who can
read the storage account's keys can read the raw vector store.

---

## 9. What is logged versus redacted

### Redacted or dropped automatically

| Mechanism | Effect |
|---|---|
| `redact_secrets` structlog processor | Any event key matching `secret\|password\|passwd\|token\|api_key\|apikey\|access_key\|private_key\|client_secret\|connection_string\|sas\|credential\|authorization\|auth_header\|cookie\|session_key\|signing_key\|jwt\|bearer` is replaced with `***redacted***<last 4>`; nested mappings and lists are walked recursively. |
| `guard_raw_content` structlog processor | The ten content keys listed in §6 are replaced with `***omitted-unredacted***` unless the event sets `pii_redacted=True`. |
| `Principal.audit_identity()` | Emits tenant, user id, role/group **counts**, clearance and `is_admin` — deliberately not email or display name. |
| `_redact_error` (ingestion) | Exception type only, except for `RagError`. |
| `Orchestrator._stage` degradation events | Name the exception **class**, never its text: an exception raised while handling user content can quote that content, and this value is streamed to the client and written to `chat_messages`. |
| RFC 7807 422 responses | Report `loc`/`type`/`msg` but never pydantic's `input`, which would echo the user's unredacted turn. |
| Unhandled 500 | `detail` names only the exception class. |
| `hybrid_search` tracing | `query_text` is raw user content, so only its **length** is logged. |

### Logged in the clear (by design)

Tenant ids, user object ids, session ids, message ids, document ids, chunk ids,
request ids, trace ids, route templates, status codes, latencies, token counts, cost,
model names, stage names, drop reasons, guardrail kinds/actions, filter structures
(`serialise_filter` — ids and facet values, never text), classification labels and
ranks.

### Audit rows

`repositories.write_audit` is called from seven places: memory item delete, memory
consent change, session delete, document upload, document delete, document reindex,
and admin ingest trigger. **Reads are not audited** — a retrieval produces a Langfuse
trace and lineage rows, but no `audit_log` entry, so "who read what" is only
answerable while traces are retained.

---

## 10. Residual risks

Ordered roughly by how much they would matter in a real deployment.

1. **The evaluation gate does not run**, so none of the security assertions it encodes
   are actually enforced by CI. `eval_pipeline_target` defaults to
   `app.rag.orchestrator:run_turn`, which does not exist, and the value cannot be
   overridden from the environment. The golden set's ten `acl_negative` items, the
   hard `acl_leak >= 1.0` floor and the per-item leak check are all written and
   unit-tested — and all currently unexecuted end to end. *This is the single largest
   gap between claimed and demonstrated security.*

2. **Indirect prompt injection has no structural containment on the answer path**
   (§7). Detection-only defence against poisoned documents is a known-weak posture.

3. **No integration tests.** `pytest.mark.integration` appears nowhere in the
   repository despite `ci.yml` claiming the integration-marked tests "run for real"
   against live Postgres and Qdrant. Every one of the 632 tests is in-process. The
   ACL filter is proven as a *filter object*; it has never been proven against a real
   Qdrant instance holding two tenants' points in this test suite. `scripts/smoke_test.py`
   does exercise that path, but it is not run by CI.

4. **122 security-relevant tunables are compile-time constants** (§ARCHITECTURE-6).
   Among them: the Entra role→clearance map, the retrieved-text injection threshold,
   the output PII ignore list, the OOD collapse detector, the API page-size ceiling,
   the JWKS refresh floor and every citation-verification threshold. An operator
   cannot tighten any of them without a code change and a redeploy.

5. **Observability is dead.** `prometheus-client` is not a `ragcore` dependency, so
   `PROMETHEUS_AVAILABLE` is `False`, `/metrics` returns a single comment line, and
   all seventeen `rag_*` metric series — including `rag_guardrail_events_total`, which
   is how you would notice an attack — record nothing. Langfuse still works when
   configured, but it is sampled and retained, not a security log.

6. **Reads are not audited.** There is no durable record of which principal retrieved
   which chunk. Answering "did user X ever see document Y" requires Langfuse traces
   within their retention window plus `lineage_records`, and lineage records the
   *cited* chunks, not everything retrieved.

7. **Shared multi-tenancy at every layer.** One Qdrant collection, one PostgreSQL
   database, one Blob account, one Redis. Isolation is entirely logical. There is no
   per-tenant encryption key, no data-residency control, and a Qdrant-level bug in
   filter evaluation is a cross-tenant breach.

8. **The Qdrant storage account key exists** and grants raw read of the vector store
   outside the ACL layer (§8).

9. **PII detection is regex-only in the current checkout.** Without the `pii` extra
   there is no `PERSON` or `LOCATION` recognition, so a name in free text is neither
   redacted at ingestion nor at egress. `RAG_PII_HASH_SECRET` also still carries its
   placeholder value in `.env.example`.

10. **No cost or token abuse control.** Rate limiting is per request per minute
    (60/user), not per token or per dollar. A user with a valid token can drive
    `claude-opus-5` spend, and a `tool_required` loop can amplify it up to six
    iterations per turn.

11. **`GET /documents` applies role/group/deny in process after a SQL page.** It
    over-fetches ×4 and truncates, which is correct for correctness but means a
    principal with almost no visibility does a 4× read of the tenant's document table
    per page. It is a performance and information-timing consideration, not a leak.

12. **Dev mode is on in `.env.example`** (`RAG_ENTRA_DEV_MODE=true`). The production
    guard is solid, but a `staging` or `dev` environment inherits the unsigned-header
    path unless explicitly disabled.

13. **Error-message oracle on `/admin/*`.** `require_roles` distinguishes 401 from 403,
    which is correct behaviour but does tell an authenticated caller that an admin
    surface exists. The document and session routes deliberately do not.

### What a real production sign-off would need

**Blocking:**

* Make the eval gate executable, wire it into CI as a required check, and require the
  `acl_negative` category to pass on every PR.
* Add integration tests that stand up a real Qdrant with two tenants' points and prove
  `build_acl_filter` in the engine — including the deny-beats-group case, the
  clearance ceiling and the `is_tenant` partition actually being used.
* Wrap and sanitise retrieved and tool-returned text on the generation path, and add a
  golden item whose document contains `</sources>` and an instruction, asserting the
  instruction is not followed.
* Declare the 122 constants as `Settings` fields so security thresholds are
  operator-controllable and reviewable in config.
* Add `prometheus-client` and alert on `rag_guardrail_events_total{action="block"}`,
  `retrieval.acl_rejected` and the stage-12 clearance-violation error log.
* Install and enable Presidio, or accept and document that names are not detected.
* Rotate `pii_hash_secret` away from the placeholder and record the rotation
  procedure (it invalidates existing hash-mode joins).

**Strongly recommended:**

* Durable read auditing: a row per retrieval with principal, filter fingerprint and
  returned chunk ids, retained independently of Langfuse.
* A token/cost budget per tenant and per user, enforced before the model call.
* An external penetration test focused on cross-tenant retrieval, clearance bypass via
  summarisation/citation spans, and indirect injection through an uploaded document.
* Threat-model review of the tool registry as a deployment artefact: today a
  `tools.yaml` change is a code-adjacent config change with egress implications and no
  separate approval path.
* Decide and document the data-retention policy for `chat_messages`,
  `user_memories`, `lineage_records` and Langfuse traces. Nothing expires today except
  memories (`memory_ttl_days=365`) and cache entries.
* Confirm the Azure Files shared-key exposure is acceptable, or move Qdrant to a
  storage path that supports managed identity.
