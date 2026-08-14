"""Smoke tests for the TOML-backed pricing rate loader."""

from __future__ import annotations

from axiom_encode.harness.encoding_db import TokenUsage
from axiom_encode.harness.pricing import (
    ModelPricing,
    PricingRates,
    _load_pricing_rates,
    estimate_usage_cost_breakdown,
    estimate_usage_cost_usd,
    get_model_pricing,
    get_pricing_rates,
)


def test_pricing_rates_load_has_expected_shape():
    rates = get_pricing_rates()
    assert isinstance(rates, PricingRates)
    assert rates.version >= 1
    assert rates.effective_date  # non-empty string
    assert rates.models, "pricing_rates.toml should declare at least one model"

    for name, pricing in rates.models.items():
        assert isinstance(name, str) and name
        assert isinstance(pricing, ModelPricing)
        assert pricing.input_per_million >= 0.0
        assert pricing.output_per_million >= 0.0
        assert pricing.cache_read_per_million >= 0.0
        assert pricing.cache_create_per_million >= 0.0
        assert pricing.max_input_tokens is None or pricing.max_input_tokens > 0


def test_known_models_resolve_via_public_api():
    # The public API must keep working after the TOML extraction.
    opus = get_model_pricing("opus")
    assert opus is not None
    assert opus.input_per_million > 0

    # Prefix matching is part of the existing contract.
    extended = get_model_pricing("claude-opus-4-6-some-variant")
    assert extended is not None

    terra = get_model_pricing("gpt-5.6-terra")
    sol = get_model_pricing("gpt-5.6-sol")
    base_alias = get_model_pricing("gpt-5.6")
    assert terra == ModelPricing(2.0, 12.0, 0.20, 2.50, 272000)
    assert sol == ModelPricing(5.0, 30.0, 0.50, 6.25, 272000)
    assert base_alias == sol
    assert get_model_pricing("gpt-5.6-luna") is None


def test_prefix_fallback_requires_variant_boundary():
    # Dash-suffixed variants keep inheriting their family's pricing...
    assert get_model_pricing("gpt-5.6-terra-2") == get_model_pricing("gpt-5.6-terra")
    # ...but lexical siblings that merely share leading characters do not.
    assert get_model_pricing("gpt-5.6-solstice") is None
    assert get_model_pricing("gpt-5.6-terra2") is None


def test_context_tier_gate_can_be_skipped_for_aggregated_usage():
    over_boundary = TokenUsage(input_tokens=272001, output_tokens=100)

    assert estimate_usage_cost_breakdown("gpt-5.6-terra", over_boundary) is None
    breakdown = estimate_usage_cost_breakdown(
        "gpt-5.6-terra",
        over_boundary,
        enforce_context_tier=False,
    )
    assert breakdown is not None
    assert breakdown.total_cost_usd > 0


def test_gpt_5_6_standard_pricing_fails_closed_above_short_context_tier():
    at_boundary = TokenUsage(input_tokens=272000, output_tokens=100)
    over_boundary = TokenUsage(input_tokens=272001, output_tokens=100)

    assert estimate_usage_cost_usd("gpt-5.6-terra", at_boundary) is not None
    assert estimate_usage_cost_usd("gpt-5.6-sol", at_boundary) is not None
    assert estimate_usage_cost_usd("gpt-5.6", at_boundary) is not None
    assert estimate_usage_cost_usd("gpt-5.6-terra", over_boundary) is None
    assert estimate_usage_cost_usd("gpt-5.6-sol", over_boundary) is None
    assert estimate_usage_cost_usd("gpt-5.6", over_boundary) is None


def test_load_pricing_rates_is_reparseable():
    # _load_pricing_rates bypasses the cache, so calling twice must still work
    # and produce structurally identical output.
    first = _load_pricing_rates()
    second = _load_pricing_rates()
    assert first.version == second.version
    assert first.effective_date == second.effective_date
    assert set(first.models) == set(second.models)
