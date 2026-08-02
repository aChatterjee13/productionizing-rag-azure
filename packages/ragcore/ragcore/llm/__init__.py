"""Anthropic model access: client, prompts and pricing.

``ragcore.llm`` is the only place in the platform that talks to the Anthropic
API. Import :func:`get_llm_client` rather than constructing ``AsyncAnthropic``
anywhere else, so the LLM_FACTS rules in `docs/CONTRACTS.md` (adaptive thinking,
nested effort, no sampling parameters, no prefill, cache breakpoint placement,
server-side fallbacks, refusal handling, retries, tracing) are enforced once.

System prompts live in :mod:`ragcore.llm.prompts` as versioned constants so the
cached prefix stays byte-stable.
"""

from __future__ import annotations

from ragcore.llm import prompts
from ragcore.llm.client import (
    BETA_COMPACTION,
    BETA_CONTEXT_MANAGEMENT,
    BETA_SERVER_FALLBACK,
    CLEAR_TOOL_USES_EDIT,
    COMPACT_EDIT,
    FALLBACKS_DEFAULT,
    LLMClient,
    LLMRefusedError,
    LLMResponse,
    LLMUsage,
    StreamEvent,
    StreamEventType,
    clear_tool_uses_edit,
    compaction_edit,
    get_llm_client,
    reset_llm_client_cache,
)
from ragcore.llm.pricing import (
    MODEL_CHEAP,
    MODEL_FAST,
    MODEL_MAIN,
    MODEL_PRICING,
    ModelPricing,
    estimate_cost_usd,
    pricing_for,
)
from ragcore.llm.prompts import PROMPT_VERSION, prompt_metadata

__all__ = [
    "BETA_COMPACTION",
    "BETA_CONTEXT_MANAGEMENT",
    "BETA_SERVER_FALLBACK",
    "CLEAR_TOOL_USES_EDIT",
    "COMPACT_EDIT",
    "FALLBACKS_DEFAULT",
    "MODEL_CHEAP",
    "MODEL_FAST",
    "MODEL_MAIN",
    "MODEL_PRICING",
    "PROMPT_VERSION",
    "LLMClient",
    "LLMRefusedError",
    "LLMResponse",
    "LLMUsage",
    "ModelPricing",
    "StreamEvent",
    "StreamEventType",
    "clear_tool_uses_edit",
    "compaction_edit",
    "estimate_cost_usd",
    "get_llm_client",
    "pricing_for",
    "prompt_metadata",
    "prompts",
    "reset_llm_client_cache",
]
