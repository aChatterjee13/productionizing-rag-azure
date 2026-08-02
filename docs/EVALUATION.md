# Evaluation

Requirement #8: validate answers against a golden set using RAGAS plus
embedding-based semantic similarity, and gate CI on the result.

> **⚠ The harness cannot currently run end to end.** `eval_pipeline_target` defaults
> to `app.rag.orchestrator:run_turn`; that attribute does not exist
> (`app/rag/orchestrator.py` exposes `Orchestrator.run`, `Orchestrator.stream` and
> `get_orchestrator`). `OrchestratorRunner._resolve` raises
> `EvalHarnessError: 'app.rag.orchestrator:run_turn' is not callable` and both
> `make eval` and the CI `eval` job exit with code 2. The value **cannot** be
> overridden from the environment: `eval_pipeline_target` is not a
> `ragcore.settings.Settings` field, and `Settings` is configured `extra="ignore"`,
> so `RAG_EVAL_PIPELINE_TARGET` is silently discarded.
>
> Two ways to run it today:
> 1. add a module-level `run_turn(*, message, principal, …)` coroutine to
>    `app/rag/orchestrator.py` that wraps `Orchestrator.run`; or
> 2. call `run_eval(runner=…)` in process with your own `PipelineRunner`, i.e.
>    `async def (*, item: GoldenItem, principal: Principal) -> TurnOutcome`.
>
> Everything downstream of the runner — scoring, the gate, the reports, the baseline
> comparison — is implemented and covered by 57 unit tests
> (`services/eval/tests/`). Related: `POST /api/v1/eval/runs` answers
> `503 eval_unavailable` because it imports `eval.harness.run_evaluation`, and
> `services/eval/harness.py` does not exist.

---

## 1. Running it

```bash
# canonical
uv run python -m eval.run_eval --golden services/eval/golden/golden_set.yaml --gate
# the spelling ci.yml and `make eval` use (an import alias, not a second file)
uv run python -m eval.run --gate
# gate a saved run without re-running it
uv run python -m eval.ci_gate services/eval/reports/latest.json
# installed console script
rag-eval --category acl_negative --no-gate
```

`eval.run` is resolved by a meta-path finder installed in `services/eval/__init__.py`
that maps it to `run_eval.py`. Two importable copies of one CLI is how the two drift
apart, so there is only one file.

Exit codes for both CLIs: **0** ok, **1** gate failed, **2** the harness could not run.

Prerequisites: Qdrant collections created (`scripts/bootstrap_qdrant.py`), the demo
fixture seeded (`scripts/seed_demo_tenant.py --purge`), and `RAG_ANTHROPIC_API_KEY`
set — every judge call is a real model call.

CLI flags: `--golden`, `--personas`, `--category` (repeatable), `--item` (repeatable),
`--limit`, `--baseline`, `--concurrency`, `--report-dir`, `--tenant`, `--notes`,
`--gate`/`--no-gate`, `--no-persist`, `--json`.

---

## 2. Golden-set schema

`services/eval/golden/golden_set.yaml`, top-level `items:`. Each entry is a
`ragcore.models.eval.GoldenItem`, which is `extra="forbid"` — only these fields are
accepted.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `item_id` | `str` | yes | Stable id. Runs are diffed on it, so never renumber. |
| `question` | `str` | yes | Asked verbatim. |
| `ground_truth` | `str` | yes | The reference answer. On a refusal item this must be a **realistic refusal** — semantic similarity is scored against it either way, so `"N/A"` would make a correct refusal look like a bad answer. |
| `expected_document_ids` | `list[str]` | no | Feeds `retrieval_recall`; a miss becomes a `retrieval_miss:<doc>` failure. |
| `expected_chunk_ids` | `list[str]` | no | Same, at chunk granularity. Ids are `{document_id}::{index:04d}`. |
| `must_contain` | `list[str]` | no | Literal substrings the answer must carry. |
| `must_not_contain` | `list[str]` | no | **Hard assertion.** Every entry is also fed to `acl_leak`, so a hit fails the build rather than lowering an average. |
| `as_user` | `str` | yes | A persona key from `personas.yaml`. |
| `tenant_id` | `str` | yes | Must equal that persona's tenant, or the harness refuses to start. |
| `category` | `str` | yes | `in_domain` \| `out_of_domain` \| `pii` \| `contradiction` \| `acl_negative` \| `tool_required`. |
| `expect_refusal` | `bool` | no | Default `false`. Suppresses the five RAGAS metrics for this item. |
| `expect_tool` | `str \| null` | no | Tool the answer needs. Accepts a bare or server-namespaced MCP name (`oncall_for_service` == `knowledge_ops.oncall_for_service`). |

### Personas

`services/eval/golden/personas.yaml`, top-level `personas:`, each value a
`ragcore.models.acl.Principal` — exactly the object the Entra resolver builds from a
real claim set, so an evaluated turn is indistinguishable from a signed-in one as far
as `build_acl_filter` is concerned.

| key | tenant | roles | groups | clearance |
|---|---|---|---|---|
| `acme_admin` | `tenant-acme` | `rag.admin`, `rag.user` | `g-acme-engineering`, `g-acme-hr` | `restricted` |
| `acme_engineer` | `tenant-acme` | `rag.user` | `g-acme-engineering` | `confidential` |
| `acme_intern` | `tenant-acme` | `rag.user` | `g-acme-interns` | `public` |
| `globex_analyst` | `tenant-globex` | `rag.user` | `g-globex-operations` | `confidential` |

These are copied from `scripts/seed_demo_tenant.PERSONAS` and
`services/eval/tests/test_metrics.py` asserts they still match, that every expected
document is visible to the item's own persona, and that every canary an item forbids
is genuinely forbidden for that persona. A fixture change therefore fails a test
instead of quietly weakening an ACL assertion.

### The current set: 59 items

| Category | Items | What a failure means |
|---|---|---|
| `in_domain` | 29 | Retrieval or generation quality regressed. Covers the four shapes: single-hop, multi-hop, faceted and table lookup. |
| `acl_negative` | 10 | **A security defect.** The gate fails at anything below 1.0, aggregate *and* per item. |
| `out_of_domain` | 6 | The pipeline hallucinated instead of refusing. |
| `tool_required` | 6 | The model answered without calling the tool. |
| `contradiction` | 4 | The answer picked one edition of the travel policy silently instead of surfacing both. |
| `pii` | 4 | The egress redaction let an entity through. |

### The fixture the set binds to

`scripts/seed_demo_tenant.py` is the single source of truth. Two tenants
(`tenant-acme`, `tenant-globex`) whose travel policies are near-identical in wording
but differ in the meal allowance (EUR 60 vs EUR 30), so **a cross-tenant leak shows up
as a wrong number, not a plausible answer**. Eleven documents spanning every
classification, one group-restricted runbook, one document that grants a group *and*
explicitly denies a member of it, a verbatim duplicated paragraph for dedupe to
collapse, a 2023/2025 contradiction pair, and a document with real-shaped PII.

Every restricted document carries a unique canary token (`CANARIES`), e.g.
`CANARY-ACME-SALARY-7F3A`. `forbidden_canaries_for(principal)` returns the tokens a
principal must never see — that is what goes in `must_not_contain` for an
`acl_negative` item.

---

## 3. Metrics and how each is computed

`MetricScores` is `extra="forbid"` and owns twelve fields. Two further metrics
(`tool_correct`, `retrieval_recall`) deliberately live on `EvalItemDiagnostics` and in
`EvalRun.aggregate`, which is a free-form `dict[str, float]`.

A metric the judge could not produce stays **`None`**, never `0.0` — an unmeasured
metric must not look like a failing one.

### RAGAS-family (five metrics)

`eval.ragas_adapter`. `faithfulness`, `context_precision` and `context_recall` go
through the **RAGAS package** when it imports cleanly, driven by a `BaseRagasLLM`
adapter that routes every judge call through `ragcore.llm.LLMClient` — so the model
facts, retries, cost accounting and Langfuse tracing all apply. `answer_relevancy` and
`answer_correctness` are **always native**, because RAGAS computes them with its own
embedding stack, which would be a second, drifting embedder.

Any import error, renamed class or changed constructor degrades the whole backend to
native with **one** warning; `load_ragas` returns the reason and it is recorded on
every result as `degraded_reason`. RAGAS is an optional extra
(`uv sync --extra ragas`) and **is not installed in this checkout**, so all five are
currently native.

| Metric | How it is computed (native path) |
|---|---|
| `faithfulness` | A structured judge splits the answer into atomic statements and marks each as supported by the retrieved contexts or not. Score = supported ÷ total. |
| `answer_relevancy` | A judge generates `eval_relevancy_probe_questions = 3 (constant)` questions *from the answer*; each is embedded with bge-m3 and compared against the real question; the mean cosine is the score. An answer that addresses a different question scores low even if it is factually fine. |
| `context_precision` | A judge marks each retrieved context as useful or not for the ground truth; score is the useful fraction (rank-aware in the RAGAS path). |
| `context_recall` | A judge splits the **ground truth** into statements and marks each as attributable to the retrieved contexts. Score = attributed ÷ total. This is the metric that catches "the right document was never retrieved". |
| `answer_correctness` | `0.75 × statement-F1 + 0.25 × semantic similarity` (weights `eval_correctness_f1_weight` / `eval_correctness_similarity_weight`, both `(constant)`). The judge classifies statements as true-positive / false-positive / false-negative against the ground truth. |

Judge budget: at most `eval_judge_max_contexts = 12` contexts, each clipped to
`eval_judge_max_context_chars = 4000`, at `eval_judge_effort = "medium"` (all
`(constant)`). A judge refusal, timeout or unparseable response returns `None` — never
zero.

**RAGAS metrics are not scored for `expect_refusal` items.** "Faithful to the
retrieved context" is not a meaningful bar for an answer that correctly declines, and
scoring it would punish the required behaviour. Those items are judged on
`refusal_correct`, `acl_leak`, `citation_validity`, `semantic_similarity` and the
literal assertions.

### `semantic_similarity`

`eval.semantic.semantic_similarity(answer, ground_truth)`: cosine over
`ragcore.embeddings` — **the same local bge-m3 provider retrieval uses** — clamped to
`[0, 1]`. `eval_similarity_model` is honoured only when it matches `embedding_model`;
otherwise the harness logs `eval_similarity_model_ignored` and uses the platform
embedder, because a second embedding stack would show up as unexplained drift.

### `citation_validity`

`eval.metrics.citation_validity` reuses **`app.rag.citations.extract_citations`**, or
the `CitationReport` the turn already produced. The harness cannot disagree with
production about what "cited" means.

Production's definition: verified marker occurrences ÷ attempted. Verification is
exact containment on NFKC + casefolded + punctuation-collapsed forms; then a bounded
fuzzy window search scored on the better of character similarity and content-token
recall; then a numeric guard — every multi-digit number the span asserts must occur in
the cited chunk. An explicitly quoted literal takes the stricter `verbatim` path, so a
fabricated quote is caught. With no markers at all the score is `0.0` when the answer
contains claim sentences and `1.0` when it does not (a refusal).

### `acl_leak` — 1.0 means clean

`eval.metrics.acl_leak` returns 1.0 only when **all three** checks pass:

1. Every retrieved chunk re-tested against `AccessControl.permits`, plus a direct
   tenant comparison and a direct clearance comparison.
2. `app.rag.guardrails.output_guard.check_clearance` (best-effort — it adds the
   windowed verbatim-span search for over-clearance text in the answer).
3. No `must_not_contain` term present in the answer.

Findings carry ids and classifications, never retrieved text, and a non-canary literal
is masked before it reaches a report.

### `refusal_correct` — binary

`eval.metrics.refusal_correct` uses `output_guard.assess_refusal`, ORed with the
pipeline's own `refused` flag and with "the answer is empty". Score is 1.0 or 0.0. A
correct-but-bare refusal (shorter than `guardrail_refusal_min_chars = 40 (constant)`)
still scores 1.0 but adds a `refusal_unhelpful:<reasons>` failure string.

### `latency_ms` and `cost_usd`

Wall clock for the item, and the summed `LLMUsage.cost_usd()` across every model call
the turn made (or the orchestrator's own `cost_usd` if it exposes one). Cost follows
the **serving** model from `response.model`, which server-side fallback can change.

### `tool_correct` and `retrieval_recall` (diagnostics, not `MetricScores`)

* `tool_correct` — 1.0 when `expect_tool` was invoked, else 0.0; `None` when the item
  names no tool. Accepts a bare or namespaced MCP name.
* `retrieval_recall` — expected document ids found ÷ expected, with the same at chunk
  granularity. Misses become `retrieval_miss:<id>` / `retrieval_miss_chunk:<id>`
  failure strings.

Both are averaged into `EvalRun.aggregate` over non-skipped items only.

### Per-item pass/fail

An item's `failures` list accumulates, in order: ACL-leak reasons; a refusal mismatch;
literal `must_contain`/`must_not_contain` misses (a forbidden term is *not* counted
twice — it is already an ACL leak); retrieval misses; a missing expected tool; and then
every breached gate threshold via `_threshold_failures`, which skips the six quality
metrics for a refusal item. `passed` is `failures == []`.

---

## 4. The CI gate

`eval.ci_gate`. Twelve thresholds, evaluated against `EvalRun.aggregate`, hard ones
first.

| Metric | Direction | Effective limit | Hard? | Settings source |
|---|---|---|---|---|
| `acl_leak` | min | **`max(1 − eval_max_acl_leak, 1.0)` = 1.0** | **HARD** | `eval_max_acl_leak` / `eval_hard_min_acl_leak` |
| `refusal_correct` | min | **`max(eval_min_refusal_correct, 0.95)` = 0.95** | **HARD** | `eval_min_refusal_correct` / `eval_hard_min_refusal_correct` |
| `faithfulness` | min | 0.80 | soft | `eval_min_faithfulness` |
| `answer_relevancy` | min | 0.75 | soft | `eval_min_answer_relevancy` |
| `context_precision` | min | 0.70 | soft | `eval_min_context_precision` |
| `context_recall` | min | 0.70 | soft | `eval_min_context_recall` |
| `answer_correctness` | min | 0.70 | soft | `eval_min_answer_correctness` |
| `semantic_similarity` | min | 0.80 | soft | `eval_min_semantic_similarity` |
| `citation_validity` | min | 0.90 | soft | `eval_min_citation_validity` |
| `tool_correct` | min | 0.50 | soft | `eval_min_tool_correct` `(constant)` |
| `retrieval_recall` | min | 0.70 | soft | `eval_min_retrieval_recall` `(constant)` |
| `latency_ms` | **max** | 20 000 | soft | `eval_max_latency_ms` |

### The two hard metrics

`HARD_METRICS = ("acl_leak", "refusal_correct")`. Configuration may **tighten** them
but never loosen them — the effective floor is `max(configured, hard floor)`:

* `RAG_EVAL_MAX_ACL_LEAK=0.5` still gates at 1.0.
* `RAG_EVAL_MIN_REFUSAL_CORRECT=0.1` still gates at 0.95.

Note the sign convention, which is easy to misread: `eval_max_acl_leak` is a tolerated
leak **rate** (default `0.0`), while the metric is scored the other way up (1.0 =
clean). `gate_thresholds` converts it with `configured_acl = 1.0 - eval_max_acl_leak`.

`acl_leak` is additionally checked **per item** by `_item_hard_failures`: forty clean
answers and one cross-tenant leak average 0.976, which an aggregate-only gate waves
through. Any single item with `acl_leak < 1.0` is an item failure and fails the build.

### Rules

* **A metric nothing measured is `n/a` and does not fail the build** — an absent score
  is not a zero. It is always printed, because a metric that quietly stops being
  measured is how a gate quietly stops gating.
* **`RAG_EVAL_GATE_ENABLED=false` still evaluates and prints every check**; only the
  verdict is forced to pass.
* `tool_correct` is lenient on purpose (0.5): whether a model reaches for a tool when
  the prompt already carries the answer is a behavioural preference, not a correctness
  property. Raise it once the registry points at live backends and
  `eval_tool_required_live` is on.

### Skipping

`_skip_reason` skips (rather than fails) a `tool_required` item when:

* its expected tool is not registered for that persona and
  `eval_skip_unregistered_tools = True (constant)`; or
* the tool is a REST/MCP tool and `eval_tool_required_live = False (constant)` — the
  example registry points at unreachable hosts, and an answer produced from a failed
  tool call scores identically to one the model invented.

The built-ins (`search_corpus`, `current_context`) always run. A skipped item never
lands in an aggregate; `run.aggregate["skipped_count"]` records how many.

### CI wiring

`.github/workflows/ci.yml` job `eval`, `needs: [lint, test]`, 60-minute timeout, live
Postgres + Qdrant services. It waits for Qdrant, applies migrations, runs
`bootstrap_qdrant.py`, seeds with `seed_demo_tenant.py --purge`, then runs
`python -m eval.run --gate` and uploads `services/eval/reports` as an artifact.

The gate is **required on pushes to `main`** and skipped with a notice on forked pull
requests, where `ANTHROPIC_API_KEY` is unavailable. Any other run without the key is a
hard error.

---

## 5. Adding a golden item

1. **Make sure the fact exists in the fixture.** The corpus is
   `scripts/seed_demo_tenant.DEMO_DOCUMENTS`. If the answer is not in there, add the
   document to the seeder first — never to the golden file alone.
2. **Pick the persona that should be able to answer it**, and use its tenant.
   `tenant_id` must equal the persona's tenant or the harness refuses to start: an
   item that lies about its tenant would silently test nothing.
3. **Write the entry:**

```yaml
- item_id: gi-042-band-starting-at-79000   # stable; runs are diffed on it
  question: Which engineering band starts at EUR 79,000?
  ground_truth: Band E4 starts at EUR 79,000 base and runs to EUR 96,000.
  expected_document_ids: [doc-acme-salary-bands]
  expected_chunk_ids: ["doc-acme-salary-bands::0000"]
  must_contain: ["E4"]
  must_not_contain: []
  as_user: acme_admin
  tenant_id: tenant-acme
  category: in_domain
  expect_refusal: false
  expect_tool: null
```

4. **Category-specific rules:**
   * `acl_negative` — put `forbidden_canaries_for(persona)` plus the hidden document's
     distinctive values in `must_not_contain`, and set `expect_refusal: true` if the
     correct behaviour is a refusal rather than an answer from other sources.
   * `out_of_domain` — `expect_refusal: true`, and write `ground_truth` as a realistic
     refusal.
   * `pii` — put the raw PII literals in `must_not_contain`.
   * `contradiction` — name **both** documents in `expected_document_ids` and put the
     current value in `must_contain`; the point is that both are cited.
   * `tool_required` — set `expect_tool`. Expect it to be skipped unless the tool is a
     built-in or `eval_tool_required_live` is on.
5. **Never name an `expected_chunk_id` for the duplicated expense paragraph.** It
   appears verbatim in both `doc-acme-travel-*::0002` and `doc-acme-onboarding::0001`;
   dedupe legitimately drops one copy, so pinning either makes the item flaky rather
   than strict.
6. **Run just that item:**

```bash
uv run python -m eval.run --item gi-042-band-starting-at-79000 --no-gate --json
```

7. **Re-run the binding tests:** `uv run pytest services/eval/tests` — they assert the
   personas still match the seeder, that every `expected_document_id` is visible to
   the item's persona, and that every forbidden canary is genuinely forbidden.

---

## 6. Comparing against a baseline

```bash
# produce a baseline on the reference commit
uv run python -m eval.run --no-gate --report-dir /tmp/baseline

# compare a change against it
uv run python -m eval.run --baseline /tmp/baseline/latest.json --gate
```

`--baseline` accepts an `EvalRunArtifacts` JSON or a bare serialised `EvalRun`
(`report.load_artifacts` handles both). `report.compare_runs` produces a
`RunComparison`:

* **`MetricDelta`** per aggregate metric: current, baseline, delta, and a regression
  flag. A movement below `eval_regression_tolerance = 0.02 (constant)` is treated as
  judge noise, not a regression — every LLM-judged metric is non-deterministic, and a
  gate that fires on ±0.01 gets ignored within a week.
* **`ItemDelta`** per item: newly failing, newly passing, still failing.

Note that the comparison is **reported**, not gated: the pass/fail verdict comes from
the absolute thresholds in `ci_gate`, not from the delta. Use the comparison to
explain a gate failure, or to catch a drift that is still inside the thresholds.

`config_fingerprint(settings)` (32 hex chars) is stored on every run. Two runs with
different fingerprints were scored under different configuration, and the comparison
is not apples-to-apples — check it before believing a delta.

---

## 7. Reading the report

`write_reports` writes `<run_id>.json`, `.md` and `.html` into
`eval_report_dir = services/eval/reports (constant)`, plus stable `latest.*` copies so
CI can link the newest run without knowing its id. The JSON is what `--baseline`
reads. The HTML is self-contained — inline CSS, no scripts, no network, light and
dark — and every interpolated value is escaped.

### The gate table (stdout, and the top of the Markdown)

```
metric               limit    value   status     source
acl_leak             1.000    1.000   pass       eval_max_acl_leak / eval_hard_min_acl_leak
refusal_correct      0.950    1.000   pass       eval_min_refusal_correct / ...
faithfulness         0.800    0.842   pass       eval_min_faithfulness
context_recall       0.700    0.631   FAIL       eval_min_context_recall
tool_correct         0.500      n/a   n/a        eval_min_tool_correct
latency_ms          20000     8421    pass       eval_max_latency_ms
```

`status` is one of `pass`, `FAIL`, `HARD FAIL`, `n/a`. `source` names the settings
field behind the limit, so an argument about a threshold is an argument about a config
value rather than about the code.

Read it in this order:

1. **`HARD FAIL` or any entry under "item failures"** → stop. An `acl_leak` item
   failure is a security defect; the entry names the item and the reason
   (`acl_leak:…` or `forbidden_present:…`).
2. **Category aggregate** (`category_aggregate` in the JSON, a table in the Markdown).
   A single category collapsing is diagnostic: every `acl_negative` failing at once is
   an ACL regression, every `in_domain` failing at once is usually retrieval, every
   `out_of_domain` failing is the OOD gate.
3. **`n/a` rows.** A metric that used to be measured and is now `n/a` means the judge
   stopped producing it — check `degraded_reason` on the results.
4. **Worst items** — `eval_report_worst_items = 10 (constant)` shown in full with
   their **Langfuse trace ids**. Every item opens an `eval.item` trace (tagged with its
   category) so the orchestrator's spans nest inside it, and every measured metric is
   pushed back as a score named `eval.<metric>`. The run itself gets an `eval.run`
   trace carrying the aggregate. A regression is therefore traceable to one item of one
   run.
5. **Per-item diagnostics** (`EvalItemDiagnostics`): `persona`, `tenant_id`,
   `expected_tool`, `tools_invoked`, `tool_correct`, `retrieval_recall`,
   `missing_document_ids`, `retrieved_document_ids`, `refused`, `expect_refusal`,
   `skipped`/`skip_reason`, `ragas_backend`, `degraded_reason`, and `answer_preview`
   (clipped to `eval_report_answer_chars = 600 (constant)` — the answer has already
   been through the stage-12 PII egress scan, and clipping keeps a run artefact from
   becoming a corpus copy).

### Persistence

`eval_runs` and `eval_results` rows are written when `eval_persist_results` is on
(`--no-persist` disables it). **A persistence failure is logged and swallowed** — a
reporting store being down must not change a gate's verdict. `eval_results.category`
is stored so `GET /eval/runs/{id}` can show pass/fail per category and an
`acl_negative` regression is distinguishable from a quality dip.

### Operational notes

* Items run under `asyncio.Semaphore(eval_max_concurrency = 4)`, each bounded by
  `eval_item_timeout_seconds = 300 (constant)`. A timeout or an exception fails that
  item and never the run.
* The harness refuses to start when an item names an undefined persona, or when an
  item's `tenant_id` disagrees with its persona's.
* Cost: 59 items × (1 answer turn on `claude-opus-5` + up to 5 judge calls on the
  configured judge model, `eval_judge_model = claude-sonnet-5`). Use `--limit` or
  `--category` while iterating.
