"""Tests for observability helpers."""

from axiom_encode.harness.encoding_db import TokenUsage
from axiom_encode.harness.observability import (
    _base_llm_attributes,
    _normalize_otlp_endpoint,
    extract_reasoning_entries,
    extract_reasoning_output_tokens,
)


def test_extract_reasoning_output_tokens_from_codex_token_count_events():
    trace = {
        "provider": "openai",
        "events": [
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {"reasoning_output_tokens": 17},
                    },
                },
            }
        ],
    }

    assert extract_reasoning_output_tokens(trace) == 17


def test_extract_reasoning_output_tokens_from_usage_details():
    trace = {
        "provider": "openai",
        "json_result": {
            "usage": {
                "completion_tokens_details": {
                    "reasoning_tokens": 9,
                }
            }
        },
    }

    assert extract_reasoning_output_tokens(trace) == 9


def test_extract_reasoning_entries_from_codex_reasoning_items():
    trace = {
        "provider": "openai",
        "events": [
            {
                "type": "item.completed",
                "item": {
                    "id": "item_4",
                    "type": "reasoning",
                    "text": "Need to reconcile section 151 with the qualifying-child count.",
                },
            }
        ],
    }

    entries = extract_reasoning_entries(trace)

    assert len(entries) == 1
    assert entries[0].provider == "openai"
    assert entries[0].source == "events.item.completed"
    assert entries[0].item_id == "item_4"
    assert "section 151" in entries[0].text


def test_extract_reasoning_entries_from_anthropic_thinking_blocks():
    trace = {
        "provider": "anthropic",
        "response_blocks": [
            {
                "id": "thinking_1",
                "type": "thinking",
                "thinking": "I should use the imported threshold instead of restating it.",
            }
        ],
    }

    entries = extract_reasoning_entries(trace)

    assert len(entries) == 1
    assert entries[0].provider == "anthropic"
    assert entries[0].source == "response_blocks"
    assert entries[0].item_type == "thinking"
    assert "imported threshold" in entries[0].text


def test_extract_reasoning_entries_from_openai_response_payload():
    trace = {
        "provider": "openai",
        "json_result": {
            "output": [
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": "Need to emit only one .yaml file and let the orchestrator write it.",
                        }
                    ],
                }
            ]
        },
    }

    entries = extract_reasoning_entries(trace)

    assert len(entries) == 1
    assert entries[0].provider == "openai"
    assert entries[0].source == "json_result.output"
    assert entries[0].item_id == "rs_1"
    assert "orchestrator write it" in entries[0].text


def test_normalize_otlp_endpoint_appends_traces_path():
    assert (
        _normalize_otlp_endpoint("http://localhost:6006")
        == "http://localhost:6006/v1/traces"
    )
    assert (
        _normalize_otlp_endpoint("http://localhost:6006/v1/traces")
        == "http://localhost:6006/v1/traces"
    )


def test_base_llm_attributes_include_reasoning_and_cost_breakdown():
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=40,
        cache_read_tokens=60,
        reasoning_output_tokens=11,
    )

    attrs = _base_llm_attributes(
        provider="openai",
        model="gpt-5.4",
        usage=usage,
        cost_usd=1.23,
    )

    assert attrs["gen_ai.system"] == "openai"
    assert attrs["gen_ai.usage.input_tokens"] == 100
    assert attrs["gen_ai.usage.output_tokens"] == 40
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 60
    assert attrs["axiom_encode.usage.reasoning_output_tokens"] == 11
    assert attrs["axiom_encode.cost.total_usd"] == 1.23
    assert attrs["axiom_encode.usage.non_cached_input_tokens"] == 40


def test_base_llm_attributes_price_aggregated_usage_past_single_request_tier():
    # Usage summed across attempts may exceed one request's context tier; the
    # breakdown must still accompany a known total.
    usage = TokenUsage(input_tokens=300_000, output_tokens=40)

    attrs = _base_llm_attributes(
        provider="openai",
        model="gpt-5.6-terra",
        usage=usage,
        cost_usd=1.0,
    )

    assert attrs["axiom_encode.cost.total_usd"] == 1.0
    assert "axiom_encode.cost.input_usd" in attrs


def test_base_llm_attributes_omit_breakdown_when_total_cost_unknown():
    usage = TokenUsage(input_tokens=100, output_tokens=40)

    attrs = _base_llm_attributes(
        provider="openai",
        model="gpt-5.4",
        usage=usage,
        cost_usd=None,
    )

    assert "axiom_encode.cost.total_usd" not in attrs
    assert "axiom_encode.cost.input_usd" not in attrs
