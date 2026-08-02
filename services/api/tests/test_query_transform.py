"""Stage 3 must never be able to break a chat turn — and must bound what it returns.

The behaviours pinned here are the ones the rest of the pipeline relies on:
transformation degrades to the raw query on *any* failure, the sub-question count is
bounded by settings, HyDE only fires when it is both enabled and useful, extracted
facets are vocabulary-checked, and a facet the user set is never widened.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.rag.query_transform import (
    QueryTransformPayload,
    fallback_transform,
    is_abstract_query,
    merge_filters,
    transform_query,
)
from ragcore.models.chat import Message, Role
from ragcore.models.retrieval import MetadataFilter
from ragcore.settings import Settings


def make_settings(**overrides: Any) -> Settings:
    """Build settings isolated from the developer's own .env."""
    return Settings(_env_file=None, **overrides)


class RecordingLLM:
    """Stands in for ``LLMClient.structured``, recording what it was asked."""

    def __init__(self, payload: Any) -> None:
        """Store the payload (or exception) the fake call will produce."""
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    async def structured(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


# --------------------------------------------------------------------- fallbacks
async def test_llm_failure_falls_back_to_the_raw_query() -> None:
    cfg = make_settings()
    llm = RecordingLLM(RuntimeError("anthropic is down"))

    plan = await transform_query("What is the meal allowance?", settings=cfg, llm=llm)

    assert plan.degraded is True
    assert plan.degraded_reason == "llm_error"
    assert plan.rewritten == "What is the meal allowance?"
    assert plan.queries == ["What is the meal allowance?"]
    # A degraded plan still retrieves: searching and finding nothing is recoverable,
    # answering from parametric knowledge is not.
    assert plan.needs_retrieval is True
    assert plan.metadata_filter is None
    assert plan.confidence == 0.0


async def test_a_response_of_the_wrong_shape_falls_back() -> None:
    cfg = make_settings()
    # A bare dict raises AttributeError, not ValueError, inside plan construction.
    llm = RecordingLLM({"rewritten": "not a payload object"})

    plan = await transform_query("Which VPN client do we use?", settings=cfg, llm=llm)

    assert plan.degraded is True
    assert plan.degraded_reason == "invalid_response"
    assert plan.rewritten == "Which VPN client do we use?"


async def test_disabled_transformation_never_calls_the_model() -> None:
    cfg = make_settings(qt_enabled=False)
    llm = RecordingLLM(QueryTransformPayload(rewritten="should not be used"))

    plan = await transform_query("Anything at all", settings=cfg, llm=llm)

    assert llm.calls == []
    assert plan.degraded_reason == "disabled"
    assert plan.rewritten == "Anything at all"


async def test_empty_query_short_circuits() -> None:
    cfg = make_settings()
    llm = RecordingLLM(QueryTransformPayload())

    plan = await transform_query("   ", settings=cfg, llm=llm)

    assert llm.calls == []
    assert plan.degraded_reason == "empty_query"
    assert plan.needs_retrieval is False
    assert plan.queries == []


def test_fallback_transform_is_a_working_plan() -> None:
    plan = fallback_transform("  raw words  ", reason="llm_error")

    assert plan.rewritten == "raw words"
    assert plan.degraded is True
    assert plan.is_out_of_domain is False


# ------------------------------------------------------------------- happy paths
async def test_history_is_offered_for_pronoun_resolution() -> None:
    cfg = make_settings()
    llm = RecordingLLM(
        QueryTransformPayload(
            intent="contractor travel allowance",
            rewritten="What is the meal allowance for contractors?",
            confidence=0.8,
        )
    )
    history = [
        Message(
            message_id="m1",
            session_id="s1",
            role=Role.USER,
            content="What is the meal allowance?",
            created_at=datetime.now(UTC),
        ),
        Message(
            message_id="m2",
            session_id="s1",
            role=Role.ASSISTANT,
            content="EUR 60 per day for employees.",
            created_at=datetime.now(UTC),
        ),
    ]

    plan = await transform_query(
        "and what about contractors?", history=history, settings=cfg, llm=llm
    )

    assert plan.degraded is False
    assert plan.rewritten == "What is the meal allowance for contractors?"
    turn = llm.calls[0]["messages"][0]["content"]
    assert "EUR 60 per day for employees." in turn
    assert "and what about contractors?" in turn
    # Exactly one model call: stage 3 is a single MODEL_FAST round trip.
    assert len(llm.calls) == 1


async def test_sub_questions_are_bounded_and_deduplicated() -> None:
    cfg = make_settings(qt_max_subqueries=2)
    llm = RecordingLLM(
        QueryTransformPayload(
            rewritten="Compare the 2023 and 2025 travel policies",
            sub_questions=[
                "What did the 2023 travel policy allow?",
                "compare the 2023 and 2025 travel policies",  # restates the main one
                "What does the 2025 travel policy allow?",
                "What changed between them?",
            ],
        )
    )

    plan = await transform_query("compare the policies", settings=cfg, llm=llm)

    assert plan.sub_questions == [
        "What did the 2023 travel policy allow?",
        "What does the 2025 travel policy allow?",
    ]
    assert len(plan.queries) == 3


async def test_hyde_is_suppressed_when_disabled_or_unnecessary() -> None:
    payload = QueryTransformPayload(
        rewritten="How does the company think about remote work?",
        hyde_passage="Remote work is encouraged where the role allows it...",
    )

    disabled = await transform_query(
        "how do we think about remote work?",
        settings=make_settings(qt_hyde_enabled=False),
        llm=RecordingLLM(payload),
    )
    assert disabled.hyde_passage == ""

    enabled = await transform_query(
        "how do we think about remote work?",
        settings=make_settings(qt_hyde_enabled=True),
        llm=RecordingLLM(payload),
    )
    assert enabled.hyde_passage.startswith("Remote work is encouraged")

    concrete = await transform_query(
        "concrete question",
        settings=make_settings(qt_hyde_enabled=True),
        llm=RecordingLLM(
            QueryTransformPayload(
                rewritten=(
                    "Which exact firmware version does the ACME-4471 gateway require "
                    "before the Q3-FY24 rollout can proceed in the Munich datacentre?"
                ),
                hyde_passage="The gateway requires firmware 4.4.1 ...",
            )
        ),
    )
    assert concrete.hyde_passage == ""


async def test_hyde_passage_is_clipped_to_the_budget() -> None:
    cfg = make_settings(qt_hyde_enabled=True, qt_hyde_max_chars=40)
    llm = RecordingLLM(
        QueryTransformPayload(rewritten="what is our culture?", hyde_passage="x" * 500)
    )

    plan = await transform_query("what is our culture?", settings=cfg, llm=llm)

    assert len(plan.hyde_passage) == 40


@pytest.mark.parametrize(
    ("text", "abstract"),
    [
        ("what is our culture?", True),
        ("", False),
        (
            "Explain in detail how the organisation approaches the balance between "
            "autonomy and oversight when teams are distributed across countries",
            True,
        ),
        (
            "Explain the Q3-FY24 rollout plan for the ACME-4471 gateway including "
            "firmware 4.4.1 and the Munich datacentre maintenance window schedule",
            False,
        ),
    ],
)
def test_is_abstract_query(text: str, abstract: bool) -> None:
    assert is_abstract_query(text, settings=make_settings()) is abstract


# ----------------------------------------------------------------------- facets
async def test_facets_are_vocabulary_checked_and_dates_resolved() -> None:
    cfg = make_settings()
    llm = RecordingLLM(
        QueryTransformPayload(
            rewritten="2024 finance policy documents",
            filter_doc_types=["policy", "not-a-real-doc-type"],
            filter_source_types=["sharepoint", "carrier pigeon"],
            filter_tags=["finance", "finance"],
            filter_date_from="2024",
            filter_date_to="2024",
        )
    )

    plan = await transform_query("2024 policy docs from finance", settings=cfg, llm=llm)

    extracted = plan.metadata_filter
    assert extracted is not None
    assert extracted.doc_types == ["policy"]
    assert extracted.source_types == ["sharepoint"]
    assert extracted.tags == ["finance"]
    assert extracted.date_from == datetime(2024, 1, 1, tzinfo=UTC)
    # A bare year as an upper bound covers the whole year, not just its first instant.
    assert extracted.date_to == datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC)


async def test_an_inverted_date_range_is_dropped_not_raised() -> None:
    cfg = make_settings()
    llm = RecordingLLM(
        QueryTransformPayload(
            rewritten="anything",
            filter_date_from="2025-06-01",
            filter_date_to="2024-01-01",
        )
    )

    plan = await transform_query("anything", settings=cfg, llm=llm)

    assert plan.degraded is False
    assert plan.metadata_filter is None


def test_merge_filters_never_widens_a_user_facet() -> None:
    user = MetadataFilter(doc_types=["standard"])
    extracted = MetadataFilter(doc_types=["policy"], tags=["finance"])

    merged = merge_filters(user, extracted)

    assert merged is not None
    # Intersecting would produce [], which the validator normalises to "no
    # constraint" — silently widening the search the user narrowed.
    assert merged.doc_types == ["standard"]
    assert merged.tags == ["finance"]


def test_merge_filters_keeps_the_stricter_classification() -> None:
    from ragcore.models.acl import Classification

    user = MetadataFilter(max_classification=Classification.CONFIDENTIAL)
    extracted = MetadataFilter(
        max_classification=Classification.PUBLIC, exclude_pii=True
    )

    merged = merge_filters(user, extracted)

    assert merged is not None
    assert merged.max_classification == Classification.PUBLIC
    assert merged.exclude_pii is True


def test_merge_filters_passes_through_when_one_side_is_empty() -> None:
    user = MetadataFilter(tags=["hr"])
    assert merge_filters(user, None) is user
    assert merge_filters(None, user) is user
    assert merge_filters(None, None) is None
