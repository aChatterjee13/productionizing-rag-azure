"""Per-model Anthropic pricing so reported cost is real, not a guess.

The table mirrors the model table in `docs/CONTRACTS.md` (LLM_FACTS) and the
defaults of :attr:`ragcore.settings.Settings.anthropic_price_per_mtok`. Settings
always win: :func:`pricing_for` consults
:meth:`ragcore.settings.Settings.price_for_model` first so operators can adjust
rates without a code change, and this module only supplies the fallback.

Billing rules encoded here:

* ``usage.input_tokens`` excludes cached tokens, so the three input buckets are
  summed independently rather than nested.
* cache **reads** bill at ``0.1x`` the input rate.
* cache **writes** bill at ``1.25x`` the input rate (5-minute ephemeral TTL).
"""

from __future__ import annotations

from dataclasses import dataclass

from ragcore.settings import Settings

__all__ = [
    "DEFAULT_CACHE_READ_MULTIPLIER",
    "DEFAULT_CACHE_WRITE_MULTIPLIER",
    "MODEL_CHEAP",
    "MODEL_FAST",
    "MODEL_MAIN",
    "MODEL_PRICING",
    "ModelPricing",
    "estimate_cost_usd",
    "pricing_for",
]

#: Answer generation, agentic tool loop, contradiction resolution.
MODEL_MAIN = "claude-opus-5"
#: Query transformation, summarisation, memory extraction.
MODEL_FAST = "claude-sonnet-5"
#: Classification: routing, out-of-domain, PII verification (200K context).
MODEL_CHEAP = "claude-haiku-4-5"

#: Cache-read tokens bill at this fraction of the uncached input rate.
DEFAULT_CACHE_READ_MULTIPLIER = 0.1
#: Cache-write tokens bill at this multiple of the uncached input rate.
DEFAULT_CACHE_WRITE_MULTIPLIER = 1.25

_TOKENS_PER_MTOK = 1_000_000.0


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """USD rates for one model, per million tokens.

    Attributes:
        input_per_mtok: USD per million uncached input tokens.
        output_per_mtok: USD per million output tokens.
        cache_read_multiplier: Multiplier applied to the input rate for tokens
            served from the prompt cache.
        cache_write_multiplier: Multiplier applied to the input rate for tokens
            written to the prompt cache.
    """

    input_per_mtok: float
    output_per_mtok: float
    cache_read_multiplier: float = DEFAULT_CACHE_READ_MULTIPLIER
    cache_write_multiplier: float = DEFAULT_CACHE_WRITE_MULTIPLIER

    @property
    def cache_read_per_mtok(self) -> float:
        """USD per million cache-read tokens.

        Returns:
            The discounted input rate.
        """
        return self.input_per_mtok * self.cache_read_multiplier

    @property
    def cache_write_per_mtok(self) -> float:
        """USD per million cache-write tokens.

        Returns:
            The premium input rate charged when a prefix is first cached.
        """
        return self.input_per_mtok * self.cache_write_multiplier

    def cost_usd(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """Compute the USD cost of one call.

        Args:
            input_tokens: Uncached input tokens (``usage.input_tokens``).
            output_tokens: Generated tokens, thinking included.
            cache_read_tokens: ``usage.cache_read_input_tokens``.
            cache_write_tokens: ``usage.cache_creation_input_tokens``.

        Returns:
            Total cost in USD.
        """
        total = (
            max(input_tokens, 0) * self.input_per_mtok
            + max(output_tokens, 0) * self.output_per_mtok
            + max(cache_read_tokens, 0) * self.cache_read_per_mtok
            + max(cache_write_tokens, 0) * self.cache_write_per_mtok
        )
        return total / _TOKENS_PER_MTOK


#: Fallback rate table, keyed by exact model id (no date suffix).
MODEL_PRICING: dict[str, ModelPricing] = {
    MODEL_MAIN: ModelPricing(input_per_mtok=5.0, output_per_mtok=25.0),
    MODEL_FAST: ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0),
    MODEL_CHEAP: ModelPricing(input_per_mtok=1.0, output_per_mtok=5.0),
}


def pricing_for(model: str, settings: Settings | None = None) -> ModelPricing:
    """Resolve the rate card for a model.

    Args:
        model: Exact Anthropic model id, e.g. ``"claude-opus-5"``.
        settings: Settings whose ``anthropic_price_per_mtok`` and cache
            multipliers take precedence. When omitted the module table is used.

    Returns:
        A :class:`ModelPricing`. Unknown models fall back to the MODEL_MAIN rate
        so cost accounting over-reports rather than silently reporting zero.
    """
    if settings is not None:
        input_rate, output_rate = settings.price_for_model(model)
        return ModelPricing(
            input_per_mtok=input_rate,
            output_per_mtok=output_rate,
            cache_read_multiplier=settings.anthropic_cache_read_multiplier,
            cache_write_multiplier=settings.anthropic_cache_write_multiplier,
        )
    return MODEL_PRICING.get(model, MODEL_PRICING[MODEL_MAIN])


def estimate_cost_usd(
    *,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    settings: Settings | None = None,
) -> float:
    """Price one call without constructing a :class:`ModelPricing` first.

    Args:
        model: Exact Anthropic model id.
        input_tokens: Uncached input tokens.
        output_tokens: Generated tokens.
        cache_read_tokens: Tokens served from the prompt cache.
        cache_write_tokens: Tokens written to the prompt cache.
        settings: Optional settings override for the rate table.

    Returns:
        Total cost in USD.
    """
    return pricing_for(model, settings).cost_usd(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )
