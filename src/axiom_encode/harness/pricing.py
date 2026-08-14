"""Model pricing helpers for eval and trace analysis.

Pricing rates are loaded from ``pricing_rates.toml`` (co-located with this
module). The TOML file carries a ``version`` and ``effective_date`` so that
rate changes can be tracked in source control without touching Python code.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .encoding_db import TokenUsage

_PRICING_RATES_PATH = Path(__file__).with_name("pricing_rates.toml")


@dataclass(frozen=True)
class ModelPricing:
    """Per-million token rates for a model family."""

    input_per_million: float
    output_per_million: float
    cache_read_per_million: float = 0.0
    cache_create_per_million: float = 0.0
    max_input_tokens: int | None = None


@dataclass(frozen=True)
class UsageCostBreakdown:
    """Token-cost breakdown for a single model invocation."""

    non_cached_input_tokens: int
    input_cost_usd: float
    cache_read_cost_usd: float
    cache_creation_cost_usd: float
    output_cost_usd: float
    reasoning_output_cost_usd: float
    total_cost_usd: float


@dataclass(frozen=True)
class PricingRates:
    """Full pricing rate bundle loaded from ``pricing_rates.toml``."""

    version: int
    effective_date: str
    models: dict[str, ModelPricing]


def _load_pricing_rates(path: Path = _PRICING_RATES_PATH) -> PricingRates:
    """Parse ``pricing_rates.toml`` into a :class:`PricingRates` bundle."""
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    raw_models = data.get("models", {}) or {}
    models: dict[str, ModelPricing] = {}
    for name, rates in raw_models.items():
        models[name] = ModelPricing(
            input_per_million=float(rates.get("input_per_million", 0.0)),
            output_per_million=float(rates.get("output_per_million", 0.0)),
            cache_read_per_million=float(rates.get("cache_read_per_million", 0.0)),
            cache_create_per_million=float(rates.get("cache_create_per_million", 0.0)),
            max_input_tokens=(
                int(rates["max_input_tokens"]) if "max_input_tokens" in rates else None
            ),
        )

    return PricingRates(
        version=int(data.get("version", 0)),
        effective_date=str(data.get("effective_date", "")),
        models=models,
    )


@lru_cache(maxsize=1)
def _cached_pricing_rates() -> PricingRates:
    return _load_pricing_rates()


def get_pricing_rates() -> PricingRates:
    """Return the cached pricing rate bundle loaded from TOML."""
    return _cached_pricing_rates()


_MODEL_PRICING: dict[str, ModelPricing] = get_pricing_rates().models


def get_model_pricing(model: str) -> ModelPricing | None:
    """Return pricing metadata for a supported model string."""
    normalized = (model or "").strip()
    if not normalized:
        return None
    if normalized == "gpt-5.6":
        # The unsuffixed API alias routes to the Sol model. Keep this exact so
        # unrelated siblings such as gpt-5.6-luna cannot inherit Sol pricing.
        normalized = "gpt-5.6-sol"

    if normalized in _MODEL_PRICING:
        return _MODEL_PRICING[normalized]

    for prefix, pricing in _MODEL_PRICING.items():
        # Require a variant boundary so lexical siblings such as
        # gpt-5.6-solstice cannot inherit gpt-5.6-sol pricing.
        if normalized == prefix or normalized.startswith(f"{prefix}-"):
            return pricing

    return None


def estimate_usage_cost_usd(model: str, usage: TokenUsage | None) -> float | None:
    """Estimate cost for a usage bundle if model pricing is known."""
    breakdown = estimate_usage_cost_breakdown(model, usage)
    if breakdown is None:
        return None
    return breakdown.total_cost_usd


def estimate_usage_cost_breakdown(
    model: str,
    usage: TokenUsage | None,
    *,
    enforce_context_tier: bool = True,
) -> UsageCostBreakdown | None:
    """Estimate a detailed cost breakdown for a usage bundle.

    ``enforce_context_tier`` only makes sense for single-request usage; pass
    ``False`` when ``usage`` aggregates several requests, where the summed
    input count says nothing about any one request's context tier.
    """
    if usage is None:
        return None

    pricing = get_model_pricing(model)
    if pricing is None:
        return None
    if (
        enforce_context_tier
        and pricing.max_input_tokens is not None
        and usage.input_tokens > pricing.max_input_tokens
    ):
        # The configured rate is intentionally bounded to its published context
        # tier. Unknown longer-context rates must not be silently underestimated.
        return None

    non_cached_input_tokens = max(
        usage.input_tokens - usage.cache_read_tokens - usage.cache_creation_tokens,
        0,
    )
    input_cost_usd = non_cached_input_tokens * pricing.input_per_million / 1_000_000
    cache_read_cost_usd = (
        usage.cache_read_tokens * pricing.cache_read_per_million / 1_000_000
    )
    cache_creation_cost_usd = (
        usage.cache_creation_tokens * pricing.cache_create_per_million / 1_000_000
    )
    output_cost_usd = usage.output_tokens * pricing.output_per_million / 1_000_000
    reasoning_output_cost_usd = (
        usage.reasoning_output_tokens * pricing.output_per_million / 1_000_000
    )

    return UsageCostBreakdown(
        non_cached_input_tokens=non_cached_input_tokens,
        input_cost_usd=input_cost_usd,
        cache_read_cost_usd=cache_read_cost_usd,
        cache_creation_cost_usd=cache_creation_cost_usd,
        output_cost_usd=output_cost_usd,
        reasoning_output_cost_usd=reasoning_output_cost_usd,
        total_cost_usd=(
            input_cost_usd
            + cache_read_cost_usd
            + cache_creation_cost_usd
            + output_cost_usd
        ),
    )
