"""Tests for the per-run cost summary (axiom-encode#1495).

Includes the parity pin: the standalone script's pricing math must equal
``axiom_encode.harness.pricing.estimate_usage_cost_breakdown``, and every
model the harness dispatches by default must have rates on file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axiom_encode.constants import (
    DEFAULT_MODEL,
    DEFAULT_OPENAI_ESCALATION_MODEL,
    DEFAULT_OPENAI_MODEL,
)
from axiom_encode.harness.encoding_db import TokenUsage
from axiom_encode.harness.pricing import (
    estimate_usage_cost_breakdown,
    get_model_pricing,
)
from scripts.summarize_run_cost import (
    DEFAULT_PRICING_PATH,
    Attempt,
    attempt_cost_usd,
    collect_attempts,
    load_rates,
    parse_trace,
    rates_for_model,
    render_markdown,
)


def _write_trace(
    root: Path,
    runner: str,
    slug: str,
    *,
    model: str,
    input_tokens: int,
    cached_tokens: int = 0,
    output_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> Path:
    trace_dir = root / "lane" / "traces" / runner
    trace_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "json_result": {
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_tokens_details": {"cached_tokens": cached_tokens},
                "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
            }
        },
    }
    path = trace_dir / f"{slug}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestDefaultModelsArePriced:
    """Model bumps in constants.py must ship pricing rates alongside."""

    @pytest.mark.parametrize(
        "model",
        [DEFAULT_OPENAI_MODEL, DEFAULT_OPENAI_ESCALATION_MODEL, DEFAULT_MODEL],
    )
    def test_harness_pricing_covers_default_model(self, model: str) -> None:
        assert get_model_pricing(model) is not None, (
            f"{model} has no rates in pricing_rates.toml; cost telemetry "
            "would silently report it as unpriced"
        )

    @pytest.mark.parametrize(
        "model",
        [DEFAULT_OPENAI_MODEL, DEFAULT_OPENAI_ESCALATION_MODEL, DEFAULT_MODEL],
    )
    def test_script_rates_cover_default_model(self, model: str) -> None:
        models, _ = load_rates(DEFAULT_PRICING_PATH)
        assert rates_for_model(models, model) is not None


class TestPricingParity:
    """Standalone math must match the harness breakdown exactly."""

    @pytest.mark.parametrize(
        ("model", "usage"),
        [
            (
                "gpt-5.6-terra",
                TokenUsage(input_tokens=60_793, output_tokens=1_431),
            ),
            (
                "gpt-5.6-sol",
                TokenUsage(
                    input_tokens=78_035,
                    output_tokens=6_126,
                    cache_read_tokens=12_000,
                ),
            ),
            (
                "claude-opus-4-6",
                TokenUsage(
                    input_tokens=50_000,
                    output_tokens=2_000,
                    cache_read_tokens=30_000,
                    cache_creation_tokens=10_000,
                ),
            ),
        ],
    )
    def test_total_matches_harness(self, model: str, usage: TokenUsage) -> None:
        attempt = Attempt(
            trace_path="t.json",
            model=model,
            input_tokens=usage.input_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_creation_tokens=usage.cache_creation_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=0,
        )
        models, _ = load_rates(DEFAULT_PRICING_PATH)
        script_total = attempt_cost_usd(attempt, rates_for_model(models, model))
        harness = estimate_usage_cost_breakdown(model, usage)
        assert harness is not None
        assert script_total == pytest.approx(harness.total_cost_usd, abs=1e-12)

    def test_observed_1491_attempts_price_as_expected(self) -> None:
        """Regression pin against the #1491 measured attempt (zero cache)."""
        models, _ = load_rates(DEFAULT_PRICING_PATH)
        terra = Attempt(
            trace_path="t.json",
            model="gpt-5.6-terra",
            input_tokens=60_793,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            output_tokens=1_431,
            reasoning_tokens=0,
        )
        sol = Attempt(
            trace_path="s.json",
            model="gpt-5.6-sol",
            input_tokens=78_035,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            output_tokens=6_126,
            reasoning_tokens=0,
        )
        terra_cost = attempt_cost_usd(terra, rates_for_model(models, terra.model))
        sol_cost = attempt_cost_usd(sol, rates_for_model(models, sol.model))
        assert terra_cost == pytest.approx((60_793 * 2.0 + 1_431 * 12.0) / 1e6)
        assert sol_cost == pytest.approx((78_035 * 5.0 + 6_126 * 30.0) / 1e6)


class TestTraceParsing:
    def test_parse_openai_responses_trace(self, tmp_path: Path) -> None:
        path = _write_trace(
            tmp_path,
            "openai",
            "us-ky-141-020",
            model="gpt-5.6-terra",
            input_tokens=60_793,
            cached_tokens=100,
            output_tokens=1_431,
            reasoning_tokens=900,
        )
        attempt = parse_trace(path)
        assert attempt is not None
        assert attempt.model == "gpt-5.6-terra"
        assert attempt.input_tokens == 60_793
        assert attempt.cache_read_tokens == 100
        assert attempt.output_tokens == 1_431
        assert attempt.reasoning_tokens == 900

    def test_parse_error_trace_without_usage(self, tmp_path: Path) -> None:
        trace_dir = tmp_path / "lane" / "traces" / "openai"
        trace_dir.mkdir(parents=True)
        path = trace_dir / "boom.json"
        path.write_text(
            json.dumps(
                {
                    "model": "gpt-5.6-terra",
                    "json_result": {"error": {"message": "quota"}},
                }
            ),
            encoding="utf-8",
        )
        attempt = parse_trace(path)
        assert attempt is not None
        assert attempt.error == "quota"
        assert not attempt.has_usage

    def test_parse_cache_write_tokens_spelling(self, tmp_path: Path) -> None:
        """PR #1492's explicit-cache traces record writes as cache_write_tokens."""
        trace_dir = tmp_path / "lane" / "traces" / "openai"
        trace_dir.mkdir(parents=True)
        path = trace_dir / "cw.json"
        path.write_text(
            json.dumps(
                {
                    "model": "gpt-5.6-terra",
                    "json_result": {
                        "usage": {
                            "input_tokens": 1000,
                            "output_tokens": 10,
                            "cache_write_tokens": 900,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        attempt = parse_trace(path)
        assert attempt is not None
        assert attempt.cache_creation_tokens == 900

    def test_variant_boundary_matching(self) -> None:
        models, _ = load_rates(DEFAULT_PRICING_PATH)
        assert rates_for_model(models, "gpt-5.6-sol") is not None
        assert rates_for_model(models, "gpt-5.6-sol-fast") is not None
        assert rates_for_model(models, "gpt-5.6-solstice") is None
        # The unsuffixed API alias folds onto Sol, mirroring PR #1492.
        assert rates_for_model(models, "gpt-5.6") == rates_for_model(
            models, "gpt-5.6-sol"
        )

    def test_parse_rejects_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        assert parse_trace(path) is None

    def test_collect_walks_trace_layout(self, tmp_path: Path) -> None:
        _write_trace(tmp_path, "openai", "a", model="gpt-5.6-terra", input_tokens=10)
        _write_trace(tmp_path, "openai", "b", model="gpt-5.6-sol", input_tokens=20)
        attempts = collect_attempts([tmp_path])
        assert [attempt.model for attempt in attempts] == [
            "gpt-5.6-terra",
            "gpt-5.6-sol",
        ]

    def test_collect_missing_root(self, tmp_path: Path) -> None:
        assert collect_attempts([tmp_path / "absent"]) == []


class TestRendering:
    def test_render_totals_and_footer(self, tmp_path: Path) -> None:
        _write_trace(
            tmp_path,
            "openai",
            "a",
            model="gpt-5.6-terra",
            input_tokens=1_000_000,
            output_tokens=0,
        )
        models, meta = load_rates(DEFAULT_PRICING_PATH)
        markdown, total = render_markdown(collect_attempts([tmp_path]), models, meta)
        assert total == pytest.approx(2.0)
        assert "Priced total: $2.0000" in markdown
        assert "Rates v2" in markdown

    def test_render_flags_unpriced_model(self, tmp_path: Path) -> None:
        _write_trace(
            tmp_path,
            "openai",
            "a",
            model="model-without-rates",
            input_tokens=5,
        )
        models, meta = load_rates(DEFAULT_PRICING_PATH)
        markdown, total = render_markdown(collect_attempts([tmp_path]), models, meta)
        assert total == 0.0
        assert "Unpriced model(s): model-without-rates" in markdown

    def test_render_no_traces(self) -> None:
        models, meta = load_rates(DEFAULT_PRICING_PATH)
        markdown, total = render_markdown([], models, meta)
        assert "No model traces found" in markdown
        assert total == 0.0
