from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.metrics import (
    acl_leak,
    cost_from_usages,
    retrieval_recall,
    substring_failures,
    tool_correct,
    usage_totals,
)
from eval.run_eval import (
    TurnOutcome,
    coerce_outcome,
    load_golden_set,
    load_personas,
    select_items,
)
from eval.semantic import semantic_similarity, semantic_similarity_batch
from ragcore.dedupe import content_sha256, simhash_hex
from ragcore.llm import LLMUsage
from ragcore.models.acl import AccessControl, Classification, Principal
from ragcore.models.chunk import ChunkPayload
from ragcore.models.eval import EvalCategory
from ragcore.models.retrieval import RetrievedChunk
from ragcore.settings import get_settings

_REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_PATH = _REPO_ROOT / "services" / "eval" / "golden" / "golden_set.yaml"
PERSONAS_PATH = _REPO_ROOT / "services" / "eval" / "golden" / "personas.yaml"

CHUNK_ID_SUFFIX_LENGTH = 4


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def golden(settings):
    return load_golden_set(GOLDEN_PATH, settings=settings)


@pytest.fixture(scope="module")
def personas(settings):
    return load_personas(PERSONAS_PATH, settings=settings)


@pytest.fixture(scope="module")
def seed_module():
    """Import scripts/seed_demo_tenant.py, the fixture's source of truth."""
    path = _REPO_ROOT / "scripts" / "seed_demo_tenant.py"
    spec = importlib.util.spec_from_file_location("seed_demo_tenant", path)
    if spec is None or spec.loader is None:  # pragma: no cover - broken checkout
        pytest.skip("seed_demo_tenant.py is not importable")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("seed_demo_tenant", module)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # pragma: no cover - optional deps missing locally
        pytest.skip(f"seed_demo_tenant dependencies unavailable: {exc}")
    return module


def _chunk(
    *,
    chunk_id: str = "doc-acme-travel-2025::0000",
    document_id: str = "doc-acme-travel-2025",
    tenant_id: str = "tenant-acme",
    text: str = "Employees claim a flat daily meal allowance of EUR 60 per full day.",
    classification: Classification = Classification.PUBLIC,
    allowed_groups: tuple[str, ...] = (),
    denied_users: tuple[str, ...] = (),
) -> RetrievedChunk:
    access = AccessControl(
        tenant_id=tenant_id,
        allowed_groups=list(allowed_groups),
        denied_users=list(denied_users),
        classification=classification,
    )
    payload = ChunkPayload.from_access_control(
        access,
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        source_type="blob",
        source_id="src-acme-policies",
        source_uri="https://acme.example/policies/travel-2025.md",
        title="Acme Travel and Expense Policy (2025 edition)",
        section_path=["Meal allowance"],
        page=None,
        text=text,
        contextual_header="Acme Travel and Expense Policy > Meal allowance",
        keywords=["travel"],
        doc_type="policy",
        tags=["travel"],
        author="Acme People Operations",
        language="en",
        content_sha256=content_sha256(text),
        simhash=simhash_hex(text),
        token_count=max(1, len(text) // 4),
        version=1,
        is_deleted=False,
        ingest_run_id="test",
    )
    return RetrievedChunk(payload=payload, fusion_score=0.5, final_score=0.9)


def _engineer() -> Principal:
    return Principal(
        user_id="11111111-1111-4111-8111-000000000002",
        tenant_id="tenant-acme",
        roles=["rag.user"],
        groups=["g-acme-engineering"],
        max_classification=Classification.CONFIDENTIAL,
    )


# ------------------------------------------------------------------ golden set
def test_golden_set_is_large_enough_and_covers_every_category(golden):
    assert len(golden) >= 40
    categories = {item.category for item in golden}
    assert categories == {category.value for category in EvalCategory}


def test_golden_ids_are_unique_and_stable(golden):
    ids = [item.item_id for item in golden]
    assert len(ids) == len(set(ids))
    assert all(item_id.startswith("gi-") for item_id in ids)


def test_every_item_has_a_ground_truth(golden):
    assert all(item.ground_truth.strip() for item in golden)


def test_personas_match_the_seeder(personas, seed_module):
    assert set(personas) == set(seed_module.PERSONAS)
    for key, principal in personas.items():
        expected = seed_module.PERSONAS[key]
        assert principal.model_dump() == expected.model_dump(), key


def test_items_reference_defined_personas_in_their_own_tenant(golden, personas):
    for item in golden:
        assert item.as_user in personas, item.item_id
        assert personas[item.as_user].tenant_id == item.tenant_id, item.item_id


def test_expected_documents_exist_in_the_fixture(golden, seed_module):
    known = {document.document_id for document in seed_module.DEMO_DOCUMENTS}
    for item in golden:
        for document_id in item.expected_document_ids:
            assert document_id in known, f"{item.item_id} -> {document_id}"


def test_expected_chunk_ids_are_real_chunks(golden, seed_module):
    sections = {
        document.document_id: len(document.sections)
        for document in seed_module.DEMO_DOCUMENTS
    }
    for item in golden:
        for chunk_id in item.expected_chunk_ids:
            document_id, separator, index = chunk_id.rpartition("::")
            assert separator == "::", chunk_id
            assert document_id in sections, chunk_id
            assert len(index) == CHUNK_ID_SUFFIX_LENGTH, chunk_id
            assert 0 <= int(index) < sections[document_id], chunk_id


def test_expected_documents_are_visible_to_the_persona(golden, personas, seed_module):
    """An item may not expect a document its own persona cannot read."""
    for item in golden:
        if not item.expected_document_ids:
            continue
        visible = {
            document.document_id
            for document in seed_module.documents_visible_to(personas[item.as_user])
        }
        for document_id in item.expected_document_ids:
            assert document_id in visible, f"{item.item_id} expects a hidden document"


def test_acl_negative_items_forbid_the_right_canaries(golden, personas, seed_module):
    """Every canary an item forbids must genuinely be forbidden for that persona."""
    all_canaries = set(seed_module.CANARIES.values())
    covered = set()
    for item in golden:
        forbidden = set(seed_module.forbidden_canaries_for(personas[item.as_user]))
        for term in item.must_not_contain:
            if term in all_canaries:
                assert term in forbidden, f"{item.item_id} forbids a visible canary"
                covered.add(term)
    # Every canary in the fixture is exercised by at least one negative item.
    assert covered == all_canaries


def test_acl_negative_category_names_at_least_one_forbidden_value(golden):
    negatives = [item for item in golden if item.category == "acl_negative"]
    assert len(negatives) >= 8
    assert all(item.must_not_contain for item in negatives)


def test_pii_items_forbid_the_fixture_pii_literals(golden):
    pii_items = [item for item in golden if item.category == "pii"]
    assert pii_items
    for item in pii_items:
        assert any("@" in term or "+" in term for term in item.must_not_contain)


def test_out_of_domain_items_all_expect_a_refusal(golden):
    for item in golden:
        if item.category == "out_of_domain":
            assert item.expect_refusal, item.item_id


def test_tool_items_name_a_tool(golden):
    tool_items = [item for item in golden if item.category == "tool_required"]
    assert len(tool_items) >= 4
    assert all(item.expect_tool for item in tool_items)


def test_contradiction_items_expect_both_editions(golden, seed_module):
    older, newer = seed_module.CONTRADICTION_PAIR
    items = [item for item in golden if item.category == "contradiction"]
    assert items
    assert any(
        older in item.expected_document_ids and newer in item.expected_document_ids
        for item in items
    )


def test_select_items_filters_and_limits(golden):
    negatives = select_items(golden, categories=["acl_negative"])
    assert negatives and all(item.category == "acl_negative" for item in negatives)
    single = select_items(golden, item_ids=[golden[0].item_id])
    assert [item.item_id for item in single] == [golden[0].item_id]
    assert len(select_items(golden, limit=3)) == 3


# --------------------------------------------------------------------- acl_leak
def test_acl_leak_is_clean_for_a_permitted_chunk():
    report = acl_leak(
        principal=_engineer(),
        chunks=[_chunk()],
        answer="The daily meal allowance is EUR 60.",
    )
    assert report.clean
    assert report.score == 1.0
    assert report.checked_chunks == 1


def test_acl_leak_catches_a_cross_tenant_chunk():
    report = acl_leak(
        principal=_engineer(),
        chunks=[
            _chunk(
                tenant_id="tenant-globex",
                document_id="doc-globex-travel-policy",
            )
        ],
        answer="EUR 30.",
    )
    assert report.score == 0.0
    assert [finding.kind for finding in report.findings] == ["cross_tenant"]


def test_acl_leak_catches_an_over_clearance_chunk():
    report = acl_leak(
        principal=_engineer(),
        chunks=[
            _chunk(
                chunk_id="doc-acme-security-incident::0000",
                document_id="doc-acme-security-incident",
                classification=Classification.RESTRICTED,
            )
        ],
        answer="A token leaked.",
    )
    assert report.score == 0.0
    assert report.findings[0].kind == "over_clearance"


def test_acl_leak_catches_an_explicit_deny():
    principal = _engineer()
    report = acl_leak(
        principal=principal,
        chunks=[
            _chunk(
                chunk_id="doc-acme-contractor-nda::0000",
                document_id="doc-acme-contractor-nda",
                classification=Classification.CONFIDENTIAL,
                allowed_groups=("g-acme-engineering",),
                denied_users=(principal.user_id,),
            )
        ],
        answer="Clause 7 says...",
    )
    assert report.score == 0.0
    assert report.findings[0].kind == "acl_denied"


def test_acl_leak_catches_a_canary_in_the_answer():
    report = acl_leak(
        principal=_engineer(),
        chunks=[],
        answer="Reference CANARY-ACME-SALARY-7F3A. Band E3 runs from EUR 62,000.",
        forbidden_terms=["CANARY-ACME-SALARY-7F3A"],
    )
    assert report.score == 0.0
    assert report.findings[0].kind == "forbidden_term"
    # A synthetic canary is safe to print; the failure line names it.
    assert "CANARY-ACME-SALARY-7F3A" in report.failure_reasons()[0]


def test_acl_leak_masks_a_pii_literal_in_its_finding():
    report = acl_leak(
        principal=_engineer(),
        chunks=[],
        answer="Write to priya.raman@acme.example for help.",
        forbidden_terms=["priya.raman@acme.example"],
    )
    assert report.score == 0.0
    assert "priya.raman@acme.example" not in report.failure_reasons()[0]


# ---------------------------------------------------------------------- refusal
def test_refusal_correct_matches_expectation():
    pytest.importorskip("app.rag.guardrails.output_guard")
    from eval.metrics import refusal_correct

    refused = refusal_correct(
        answer=(
            "I don't have anything on that in the indexed corpus, which covers the "
            "travel policy and the onboarding handbook."
        ),
        expect_refusal=True,
    )
    assert refused.score == 1.0
    assert refused.refused is True
    assert refused.acceptable is True

    answered = refusal_correct(
        answer="The daily meal allowance is EUR 60 per full day.",
        expect_refusal=True,
    )
    assert answered.score == 0.0
    assert answered.refused is False


def test_refusal_correct_penalises_an_unexpected_refusal():
    pytest.importorskip("app.rag.guardrails.output_guard")
    from eval.metrics import refusal_correct

    outcome = refusal_correct(answer="I cannot answer that.", expect_refusal=False)
    assert outcome.score == 0.0


def test_empty_answer_counts_as_a_refusal():
    pytest.importorskip("app.rag.guardrails.output_guard")
    from eval.metrics import refusal_correct

    assert refusal_correct(answer="   ", expect_refusal=True).score == 1.0


def test_pipeline_refusal_signal_is_honoured():
    pytest.importorskip("app.rag.guardrails.output_guard")
    from eval.metrics import refusal_correct

    outcome = refusal_correct(
        answer="Outside the indexed corpus.", expect_refusal=True, refused=True
    )
    assert outcome.refused is True
    assert outcome.score == 1.0


# ------------------------------------------------------------------- tool + text
def test_tool_correct_handles_bare_and_namespaced_names():
    assert tool_correct(expect_tool=None, tools_invoked=[]) is None
    assert tool_correct(expect_tool="order_status", tools_invoked=[]) == 0.0
    assert (
        tool_correct(expect_tool="order_status", tools_invoked=["order_status"]) == 1.0
    )
    assert (
        tool_correct(
            expect_tool="oncall_for_service",
            tools_invoked=["knowledge_ops.oncall_for_service"],
        )
        == 1.0
    )
    assert (
        tool_correct(expect_tool="order_status", tools_invoked=["Order_Status"]) == 1.0
    )


def test_substring_assertions():
    failures = substring_failures(
        answer="The allowance is EUR 60 per day.",
        must_contain=["60", "receipt"],
        must_not_contain=["EUR 45"],
    )
    assert failures == ["missing_required:receipt"]

    leaked = substring_failures(
        answer="Reference CANARY-ACME-VPN-5A17.",
        must_not_contain=["CANARY-ACME-VPN-5A17"],
    )
    assert leaked == ["forbidden_present:CANARY-ACME-VPN-5A17"]


def test_retrieval_recall_reports_what_was_missed():
    recall = retrieval_recall(
        expected_document_ids=["doc-a", "doc-b"],
        retrieved_document_ids=["doc-a"],
        retrieved_chunk_ids=["doc-a::0000"],
    )
    assert recall.score == pytest.approx(0.5)
    assert recall.missing_document_ids == ["doc-b"]

    assert retrieval_recall().score is None


def test_cost_and_usage_totals():
    usages = [
        LLMUsage(input_tokens=1000, output_tokens=500, model="claude-haiku-4-5"),
        LLMUsage(input_tokens=200, output_tokens=100, model="claude-haiku-4-5"),
    ]
    assert cost_from_usages(usages) > 0.0
    totals = usage_totals(usages)
    assert totals == {
        "input": 1200,
        "output": 600,
        "cache_read": 0,
        "cache_write": 0,
        "calls": 2,
    }


# -------------------------------------------------------------------- semantic
class _StubEmbedder:
    """Returns a fixed vector per text, so cosine is deterministic."""

    dim = 3

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls = 0

    async def embed_dense(self, texts):
        self.calls += 1
        return [self.vectors.get(text, [0.0, 0.0, 1.0]) for text in texts]

    async def embed_documents(self, texts):  # pragma: no cover - unused
        raise NotImplementedError

    async def embed_query(self, text):  # pragma: no cover - unused
        raise NotImplementedError


async def test_semantic_similarity_is_one_for_identical_vectors():
    embedder = _StubEmbedder({"a": [1.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0]})
    assert await semantic_similarity("a", "b", embedder=embedder) == pytest.approx(1.0)


async def test_semantic_similarity_clamps_negatives_to_zero():
    embedder = _StubEmbedder({"a": [1.0, 0.0, 0.0], "b": [-1.0, 0.0, 0.0]})
    assert await semantic_similarity("a", "b", embedder=embedder) == pytest.approx(0.0)


async def test_semantic_similarity_returns_none_for_an_empty_side():
    embedder = _StubEmbedder({})
    assert await semantic_similarity("", "reference", embedder=embedder) is None
    assert embedder.calls == 0


async def test_semantic_similarity_batches_in_one_call():
    embedder = _StubEmbedder({"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0]})
    scores = await semantic_similarity_batch(
        [("a", "a"), ("", "b"), ("a", "b")], embedder=embedder
    )
    assert embedder.calls == 1
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] is None
    assert scores[2] == pytest.approx(0.0)


# ------------------------------------------------------------------- citations
def test_citation_validity_uses_the_pipeline_verifier():
    pytest.importorskip("app.rag.citations")
    from eval.metrics import citation_validity

    chunk = _chunk()
    grounded = citation_validity(
        answer=(
            "Employees claim a flat daily meal allowance of EUR 60 per full day. [1]"
        ),
        chunks=[chunk],
    )
    assert grounded == pytest.approx(1.0)

    fabricated = citation_validity(
        answer="The allowance is EUR 250 per day and doubles on weekends. [1]",
        chunks=[chunk],
    )
    assert fabricated < 1.0


def test_citation_validity_prefers_a_report_the_pipeline_already_produced():
    from eval.metrics import citation_validity

    class _Report:
        citation_validity = 0.42

    score = citation_validity(answer="x [1]", chunks=[], report=_Report())
    assert score == pytest.approx(0.42)


# ------------------------------------------------------------- outcome coercion
def test_coerce_outcome_reads_a_structural_result():
    retrieval = SimpleNamespace(chunks=[_chunk()])
    message = SimpleNamespace(
        content="The allowance is EUR 60. [1]",
        citations=[],
        tool_calls=[{"tool_name": "current_context"}],
    )
    result = SimpleNamespace(
        message=message,
        retrieval=retrieval,
        trace_id="trace-1",
        usage=LLMUsage(input_tokens=10, output_tokens=5, model="claude-opus-5"),
    )

    outcome = coerce_outcome(result)
    assert outcome.answer.startswith("The allowance")
    assert outcome.retrieved_chunk_ids() == ["doc-acme-travel-2025::0000"]
    assert outcome.retrieved_document_ids() == ["doc-acme-travel-2025"]
    assert outcome.tools_invoked == ["current_context"]
    assert outcome.trace_id == "trace-1"
    assert outcome.resolved_cost() > 0.0


def test_coerce_outcome_passes_a_turn_outcome_through():
    outcome = TurnOutcome(answer="hello")
    assert coerce_outcome(outcome) is outcome


def test_coerce_outcome_survives_an_empty_result():
    outcome = coerce_outcome(object())
    assert outcome.answer == ""
    assert outcome.chunks == []
    assert outcome.resolved_cost() == 0.0
