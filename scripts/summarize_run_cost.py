#!/usr/bin/env python3
"""Append a model token/cost summary for one CI encode run to the step summary.

Motivation (axiom-encode#1495): CI encode spend was only attributable
through billing-email forensics. This script makes every targeted
re-encode run print what it consumed.

It reads the per-attempt traces the harness writes under
``<output-root>/**/traces/<runner>/*.json`` — for the OpenAI Responses
backend the trace carries ``model`` plus the raw response payload under
``json_result`` (usage keys: ``input_tokens``, ``output_tokens``,
``input_tokens_details.cached_tokens``,
``output_tokens_details.reasoning_tokens``) — and prices them with
``pricing_rates.toml`` from this checkout.

The math intentionally mirrors
``axiom_encode.harness.pricing.estimate_usage_cost_breakdown``;
``tests/test_run_cost_summary.py`` pins parity. It is reimplemented here
stdlib-only because in CI the encode binary is the provisioned
``/opt/axiom-verification`` runtime, whose pinned checkout may not carry
current rates, and because this script must run even when the encode step
failed before the encoder was importable.

Telemetry must never break an encode run: ``main`` always exits 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PRICING_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "axiom_encode"
    / "harness"
    / "pricing_rates.toml"
)


@dataclass(frozen=True)
class Attempt:
    """Token usage recorded for one model invocation trace."""

    trace_path: str
    model: str
    input_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    output_tokens: int
    reasoning_tokens: int
    error: str | None = None

    @property
    def has_usage(self) -> bool:
        return any(
            (
                self.input_tokens,
                self.cache_read_tokens,
                self.cache_creation_tokens,
                self.output_tokens,
            )
        )


@dataclass(frozen=True)
class Rates:
    """Per-million-token USD rates for one model prefix."""

    input_per_million: float
    output_per_million: float
    cache_read_per_million: float
    cache_create_per_million: float


def load_rates(path: Path) -> tuple[dict[str, Rates], dict]:
    """Load ``pricing_rates.toml`` into prefix-matchable rate entries."""
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    models: dict[str, Rates] = {}
    for name, raw in (data.get("models") or {}).items():
        models[name] = Rates(
            input_per_million=float(raw.get("input_per_million", 0.0)),
            output_per_million=float(raw.get("output_per_million", 0.0)),
            cache_read_per_million=float(raw.get("cache_read_per_million", 0.0)),
            cache_create_per_million=float(raw.get("cache_create_per_million", 0.0)),
        )
    meta = {
        "version": data.get("version"),
        "effective_date": data.get("effective_date"),
    }
    return models, meta


def rates_for_model(models: dict[str, Rates], model: str) -> Rates | None:
    """Exact-then-variant-prefix matching, mirroring ``get_model_pricing``.

    Prefix matches require a ``-`` variant boundary so lexical siblings
    (e.g. a hypothetical gpt-5.6-solstice) cannot inherit gpt-5.6-sol
    rates — same rule PR #1492 adopts in the harness, including its fold
    of the unsuffixed ``gpt-5.6`` API alias onto the Sol model.
    """
    normalized = (model or "").strip()
    if not normalized:
        return None
    if normalized == "gpt-5.6":
        normalized = "gpt-5.6-sol"
    if normalized in models:
        return models[normalized]
    for prefix, rates in models.items():
        if normalized.startswith(f"{prefix}-"):
            return rates
    return None


def parse_trace(trace_path: Path) -> Attempt | None:
    """Parse one trace file into an :class:`Attempt`, or None if unreadable."""
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    model = str(payload.get("model") or "")
    json_result = payload.get("json_result")
    usage: dict = {}
    if isinstance(json_result, dict) and isinstance(json_result.get("usage"), dict):
        usage = json_result["usage"]
    elif isinstance(payload.get("usage"), dict):
        usage = payload["usage"]

    def _int(value: object) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    input_details = usage.get("input_tokens_details")
    if not isinstance(input_details, dict):
        input_details = {}
    output_details = usage.get("output_tokens_details")
    if not isinstance(output_details, dict):
        output_details = {}
    error = None
    if isinstance(json_result, dict) and isinstance(json_result.get("error"), dict):
        error = str(json_result["error"].get("message") or "error")
    return Attempt(
        trace_path=str(trace_path),
        model=model,
        input_tokens=_int(usage.get("input_tokens")),
        cache_read_tokens=_int(
            input_details.get("cached_tokens") or usage.get("cache_read_input_tokens")
        ),
        cache_creation_tokens=_int(
            usage.get("cache_creation_input_tokens")
            or usage.get("cache_write_tokens")
            or input_details.get("cache_write_tokens")
        ),
        output_tokens=_int(usage.get("output_tokens")),
        reasoning_tokens=_int(output_details.get("reasoning_tokens")),
        error=error,
    )


def attempt_cost_usd(attempt: Attempt, rates: Rates | None) -> float | None:
    """Total cost mirroring ``estimate_usage_cost_breakdown``'s total."""
    if rates is None:
        return None
    non_cached_input = max(
        attempt.input_tokens
        - attempt.cache_read_tokens
        - attempt.cache_creation_tokens,
        0,
    )
    return (
        non_cached_input * rates.input_per_million
        + attempt.cache_read_tokens * rates.cache_read_per_million
        + attempt.cache_creation_tokens * rates.cache_create_per_million
        + attempt.output_tokens * rates.output_per_million
    ) / 1_000_000


def collect_attempts(roots: list[Path]) -> list[Attempt]:
    attempts: list[Attempt] = []
    for root in roots:
        if not root.is_dir():
            continue
        for trace_path in sorted(root.rglob("traces/*/*.json")):
            attempt = parse_trace(trace_path)
            if attempt is not None:
                attempts.append(attempt)
    return attempts


def render_markdown(
    attempts: list[Attempt], models: dict[str, Rates], meta: dict
) -> tuple[str, float]:
    """Render the summary table; return (markdown, priced total USD)."""
    lines = ["### Model spend for this run", ""]
    if not attempts:
        lines.append("No model traces found (encode never reached a model call).")
        return "\n".join(lines) + "\n", 0.0
    lines.append("| Trace | Model | Input | Cached | Output | Reasoning | Est. USD |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    total = 0.0
    unpriced: set[str] = set()
    for attempt in attempts:
        cost = attempt_cost_usd(attempt, rates_for_model(models, attempt.model))
        if cost is None:
            if attempt.model and attempt.has_usage:
                unpriced.add(attempt.model)
            cost_cell = "unpriced"
        else:
            total += cost
            cost_cell = f"{cost:.4f}"
        label = Path(attempt.trace_path).name
        if attempt.error:
            label += " (error)"
        lines.append(
            f"| {label} | {attempt.model or '?'} | {attempt.input_tokens:,} "
            f"| {attempt.cache_read_tokens:,} | {attempt.output_tokens:,} "
            f"| {attempt.reasoning_tokens:,} | {cost_cell} |"
        )
    lines.append("")
    lines.append(f"**Priced total: ${total:.4f}**")
    if unpriced:
        lines.append(
            "⚠️ Unpriced model(s): "
            + ", ".join(sorted(unpriced))
            + " — add rates to `pricing_rates.toml`."
        )
    lines.append(
        f"_Rates v{meta.get('version')} effective {meta.get('effective_date')}; "
        "estimates from recorded usage, not the provider invoice._"
    )
    return "\n".join(lines) + "\n", total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces-root",
        action="append",
        default=[],
        help="Directory tree containing traces/<runner>/*.json (repeatable)",
    )
    parser.add_argument(
        "--pricing",
        default=str(DEFAULT_PRICING_PATH),
        help="Path to pricing_rates.toml",
    )
    args = parser.parse_args(argv)
    try:
        models, meta = load_rates(Path(args.pricing))
        attempts = collect_attempts([Path(root) for root in args.traces_root])
        markdown, total = render_markdown(attempts, models, meta)
        print(markdown)
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path:
            with open(summary_path, "a", encoding="utf-8") as fh:
                fh.write(markdown)
        output_path = os.environ.get("GITHUB_OUTPUT")
        if output_path:
            with open(output_path, "a", encoding="utf-8") as fh:
                fh.write(f"run_cost_usd={total:.4f}\n")
    except Exception as error:  # noqa: BLE001 - telemetry must not fail runs
        print(f"run-cost-summary: skipped ({error})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
