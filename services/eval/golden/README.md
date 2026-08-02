# Golden set

`golden_set.yaml` is the question set requirement #8 validates against, and
`personas.yaml` is the identity each question is asked under. Both bind to
`scripts/seed_demo_tenant.py`, which is the single source of truth for the demo
corpus. Seed it before evaluating:

```bash
uv run python scripts/bootstrap_qdrant.py
uv run python scripts/seed_demo_tenant.py --purge
uv run python -m eval.run_eval --gate
```

## Personas

Copied verbatim from `seed_demo_tenant.PERSONAS`. Each is a
`ragcore.models.acl.Principal`, so an evaluated turn goes through exactly the same
`build_acl_filter` path a signed-in Entra user does.

| key | tenant | roles | groups | clearance | sees |
|---|---|---|---|---|---|
| `acme_admin` | `tenant-acme` | `rag.admin`, `rag.user` | engineering, hr | `restricted` | every Acme document |
| `acme_engineer` | `tenant-acme` | `rag.user` | engineering | `confidential` | public + internal + the VPN runbook; **not** the NDA (explicit deny), the salary bands (wrong group) or the incident report |
| `acme_intern` | `tenant-acme` | `rag.user` | interns | `public` | the two travel policies and the onboarding handbook |
| `globex_analyst` | `tenant-globex` | `rag.user` | operations | `confidential` | Globex documents only |

`services/eval/tests/test_metrics.py` asserts this file still matches the seeder, so
a fixture change fails a test instead of quietly weakening an ACL assertion.

## Categories

| category | items | what a failure means |
|---|---|---|
| `in_domain` | single-hop, multi-hop, faceted and table-lookup questions | retrieval or generation quality regressed |
| `out_of_domain` | nothing indexed answers it | the pipeline hallucinated instead of refusing |
| `pii` | the answer touches the HR duty roster | the egress redaction let an entity through |
| `contradiction` | the 2023/2025 meal allowance pair | the answer picked one edition silently |
| `acl_negative` | a persona asks for material it may not read | **a security defect** — the gate fails at anything below 1.0 |
| `tool_required` | the answer needs un-indexed live data | the model answered without calling the tool |

## Writing an item

`GoldenItem` is `extra="forbid"`, so only the contract's fields are accepted:

```yaml
- item_id: gi-042-band-starting-at-79000   # stable; runs are diffed on it
  question: Which engineering band starts at EUR 79,000?
  ground_truth: Band E4 starts at EUR 79,000 base and runs to EUR 96,000.
  expected_document_ids: [doc-acme-salary-bands]
  expected_chunk_ids: ["doc-acme-salary-bands::0000"]   # {document_id}::{index:04d}
  must_contain: ["E4"]
  must_not_contain: []
  as_user: acme_admin          # a key from personas.yaml
  tenant_id: tenant-acme       # must equal that persona's tenant
  category: in_domain
  expect_refusal: false
  expect_tool: null
```

Rules the harness relies on:

1. **`ground_truth` on a refusal item is a realistic refusal.** Semantic similarity
   and answer correctness are scored against it either way, so "N/A" would make a
   correct refusal look like a bad answer.
2. **`must_not_contain` is a hard assertion.** Every entry is also fed to the
   ACL-leak metric, so a hit fails the build rather than lowering an average. Put
   the persona's forbidden canary tokens
   (`seed_demo_tenant.forbidden_canaries_for`) there for `acl_negative`, and the
   raw PII literals for `pii`.
3. **Do not name an `expected_chunk_id` for the duplicated expense paragraph.** It
   appears verbatim in both `doc-acme-travel-*::0002` and
   `doc-acme-onboarding::0001`; dedupe legitimately drops one copy, so pinning
   either one makes the item flaky rather than strict.
4. **RAGAS metrics are not scored for `expect_refusal` items.** "Faithful to the
   retrieved context" is not a meaningful bar for an answer that correctly declines,
   and scoring it would punish the behaviour the contract requires. Those items are
   judged on `refusal_correct`, `acl_leak` and the literal assertions.
5. **A `tool_required` item naming a REST or MCP tool is skipped by default.** The
   example registry points at unreachable hosts, and an answer built from a failed
   tool call scores the same as one the model invented. Set
   `RAG_EVAL_TOOL_REQUIRED_LIVE=true` once the registry points at live backends. The
   two built-in tools (`current_context`, `search_corpus`) always run.

## Chunk ids

`{document_id}::{index:04d}`, zero-based in section order:

| document | chunks |
|---|---|
| `doc-acme-travel-2023` / `doc-acme-travel-2025` | `0000` meal allowance · `0001` flights and rail · `0002` expense claims |
| `doc-acme-onboarding` | `0000` first week · `0001` claiming expenses · `0002` equipment |
| `doc-acme-hr-contact` | `0000` who to contact |
| `doc-acme-remote-work` | `0000` eligibility · `0001` home office equipment |
| `doc-acme-vpn-runbook` · `doc-acme-salary-bands` · `doc-acme-contractor-nda` · `doc-acme-security-incident` · `doc-globex-warehouse-safety` | `0000` |
| `doc-globex-travel-policy` | `0000` meal allowance · `0001` approvals |

## Running a subset

```bash
uv run python -m eval.run_eval --category acl_negative --category pii --no-gate
uv run python -m eval.run_eval --item gi-070-allowance-changed --no-gate
uv run python -m eval.run_eval --limit 10 --baseline services/eval/reports/latest.json
```
