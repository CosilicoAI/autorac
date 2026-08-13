"""Smoke tests for the TOML-backed pricing rate loader."""

from __future__ import annotations

from axiom_encode.harness.pricing import (
    ModelPricing,
    PricingRates,
    _load_pricing_rates,
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
    assert terra == ModelPricing(2.0, 12.0, 0.20, 2.50)
    assert sol == ModelPricing(5.0, 30.0, 0.50, 6.25)


def test_load_pricing_rates_is_reparseable():
    # _load_pricing_rates bypasses the cache, so calling twice must still work
    # and produce structurally identical output.
    first = _load_pricing_rates()
    second = _load_pricing_rates()
    assert first.version == second.version
    assert first.effective_date == second.effective_date
    assert set(first.models) == set(second.models)
