"""Optional OTLP observability export for Axiom Encode runs and evals."""

from __future__ import annotations

import atexit
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any, Mapping

from .encoding_db import TokenUsage
from .pricing import estimate_usage_cost_breakdown


@dataclass(frozen=True)
class ReasoningEntry:
    """A provider-exposed reasoning item or summary."""

    text: str
    provider: str | None = None
    source: str | None = None
    item_id: str | None = None
    item_type: str | None = None


@dataclass(frozen=True)
class ObservabilityConfig:
    """Configuration for optional OTLP trace export."""

    endpoint: str
    headers: tuple[tuple[str, str], ...]
    service_name: str
    project_name: str


class ObservabilityClient:
    """Thin wrapper around an OTLP span exporter."""

    def __init__(self, config: ObservabilityConfig):
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        resource = Resource.create(
            {
                "service.name": config.service_name,
                "axiom_encode.project_name": config.project_name,
            }
        )
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(
            endpoint=config.endpoint,
            headers=dict(config.headers) or None,
        )
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        self._trace = trace
        self._provider = provider
        self._tracer = provider.get_tracer("axiom_encode.observability")

        atexit.register(self._provider.shutdown)

    def emit_span(
        self,
        name: str,
        *,
        start_time_ns: int | None = None,
        end_time_ns: int | None = None,
        attributes: Mapping[str, Any] | None = None,
        events: list[tuple[str, Mapping[str, Any]]] | None = None,
        error: str | None = None,
    ) -> None:
        """Emit a single client span."""
        from opentelemetry.trace import SpanKind, Status, StatusCode

        span = self._tracer.start_span(
            name,
            kind=SpanKind.CLIENT,
            start_time=start_time_ns,
        )
        try:
            for key, value in (attributes or {}).items():
                if value is None:
                    continue
                if isinstance(value, (str, bool, int, float)):
                    span.set_attribute(key, value)

            for event_name, payload in events or []:
                safe_payload = {
                    key: value
                    for key, value in payload.items()
                    if isinstance(value, (str, bool, int, float))
                }
                span.add_event(event_name, safe_payload)

            if error:
                span.add_event("axiom_encode.error", {"message": error})
                span.set_status(Status(StatusCode.ERROR, error))
            else:
                span.set_status(Status(StatusCode.OK))
        finally:
            span.end(end_time=end_time_ns)


_CLIENT_LOCK = Lock()
_CLIENT_CACHE_KEY: tuple[str, tuple[tuple[str, str], ...], str, str] | None = None
_CLIENT: ObservabilityClient | None = None
_IMPORT_ERROR_REPORTED = False


def extract_reasoning_output_tokens(trace_payload: Mapping[str, Any] | None) -> int:
    """Extract reasoning-output token counts from a provider-native trace."""
    if not trace_payload:
        return 0

    usage = (trace_payload.get("json_result") or {}).get("usage") or {}
    reasoning_tokens = _reasoning_tokens_from_usage(usage)
    if reasoning_tokens:
        return reasoning_tokens

    events = trace_payload.get("events")
    if not isinstance(events, list):
        return 0

    max_total_reasoning = 0
    summed_last_reasoning = 0

    for event in events:
        if not isinstance(event, dict):
            continue

        if event.get("type") == "event_msg":
            payload = event.get("payload") or {}
            if payload.get("type") != "token_count":
                continue
            info = payload.get("info") or {}
            total_usage = info.get("total_token_usage") or {}
            last_usage = info.get("last_token_usage") or {}
            max_total_reasoning = max(
                max_total_reasoning,
                int(total_usage.get("reasoning_output_tokens", 0) or 0),
            )
            summed_last_reasoning += int(
                last_usage.get("reasoning_output_tokens", 0) or 0
            )
            continue

        if event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
            max_total_reasoning = max(
                max_total_reasoning,
                _reasoning_tokens_from_usage(usage),
            )

    return max_total_reasoning or summed_last_reasoning


def extract_reasoning_entries(
    trace_payload: Mapping[str, Any] | None,
) -> list[ReasoningEntry]:
    """Extract provider-exposed reasoning summaries from a trace payload."""
    if not trace_payload:
        return []

    provider = trace_payload.get("provider")
    provider_name = provider if isinstance(provider, str) else None
    entries: list[ReasoningEntry] = []
    seen: set[str] = set()

    response_blocks = trace_payload.get("response_blocks")
    if isinstance(response_blocks, list):
        for block in response_blocks:
            if not isinstance(block, Mapping):
                continue
            if block.get("thinking"):
                _append_reasoning_entry(
                    entries,
                    seen,
                    text=block.get("thinking"),
                    provider=provider_name,
                    source="response_blocks",
                    item_id=_as_optional_str(block.get("id")),
                    item_type=_as_optional_str(block.get("type")),
                )
            _extract_reasoning_from_payload(
                block,
                entries,
                seen,
                provider=provider_name,
                source="response_blocks",
            )

    json_result = trace_payload.get("json_result")
    if isinstance(json_result, Mapping):
        _extract_reasoning_from_payload(
            json_result,
            entries,
            seen,
            provider=provider_name,
            source="json_result",
        )

    events = trace_payload.get("events")
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, Mapping):
                continue

            item = event.get("item")
            if isinstance(item, Mapping):
                _extract_reasoning_from_payload(
                    item,
                    entries,
                    seen,
                    provider=provider_name,
                    source=f"events.{event.get('type', 'item')}",
                )

            response = event.get("response")
            if isinstance(response, Mapping):
                _extract_reasoning_from_payload(
                    response,
                    entries,
                    seen,
                    provider=provider_name,
                    source=f"events.{event.get('type', 'response')}",
                )

            _extract_reasoning_from_payload(
                event,
                entries,
                seen,
                provider=provider_name,
                source=f"events.{event.get('type', 'event')}",
            )

    return entries


def emit_eval_result(result: Any, trace_payload: Mapping[str, Any] | None) -> None:
    """Export one eval result to OTLP if configured."""
    client = _get_client()
    if client is None:
        return

    usage = TokenUsage(
        input_tokens=getattr(result, "input_tokens", 0),
        output_tokens=getattr(result, "output_tokens", 0),
        cache_read_tokens=getattr(result, "cache_read_tokens", 0),
        cache_creation_tokens=getattr(result, "cache_creation_tokens", 0),
        reasoning_output_tokens=(
            getattr(result, "reasoning_output_tokens", 0)
            or extract_reasoning_output_tokens(trace_payload)
        ),
    )

    start_time_ns, end_time_ns = _window_from_duration_ms(
        getattr(result, "duration_ms", 0)
    )
    metrics = getattr(result, "metrics", None)
    attributes = _base_llm_attributes(
        provider=getattr(result, "backend", None),
        model=getattr(result, "model", None),
        usage=usage,
        cost_usd=getattr(result, "actual_cost_usd", None)
        or getattr(result, "estimated_cost_usd", None),
    )
    attributes.update(
        {
            "openinference.span.kind": "LLM",
            "axiom_encode.kind": "eval",
            "axiom_encode.citation": getattr(result, "citation", None),
            "axiom_encode.runner": getattr(result, "runner", None),
            "axiom_encode.mode": getattr(result, "mode", None),
            "axiom_encode.success": getattr(result, "success", None),
            "axiom_encode.failure_kind": getattr(result, "failure_kind", None),
            "axiom_encode.timed_out": getattr(result, "timed_out", False),
            "axiom_encode.timeout_stage": getattr(result, "timeout_stage", None),
            "axiom_encode.timeout_reason": getattr(result, "timeout_reason", None),
            "axiom_encode.timeout_seconds": getattr(result, "timeout_seconds", None),
            "axiom_encode.timeout_attempts": getattr(result, "timeout_attempts", 0),
            "axiom_encode.output_file": getattr(result, "output_file", None),
            "axiom_encode.trace_file": getattr(result, "trace_file", None),
            "axiom_encode.retrieved_files_count": len(
                getattr(result, "retrieved_files", []) or []
            ),
            "axiom_encode.unexpected_access_count": len(
                getattr(result, "unexpected_accesses", []) or []
            ),
            "axiom_encode.retry_count": getattr(result, "retry_count", 0),
            "axiom_encode.compile_pass": getattr(metrics, "compile_pass", None),
            "axiom_encode.ci_pass": getattr(metrics, "ci_pass", None),
            "axiom_encode.grounded_numeric_count": getattr(
                metrics, "grounded_numeric_count", None
            ),
            "axiom_encode.ungrounded_numeric_count": getattr(
                metrics, "ungrounded_numeric_count", None
            ),
            "axiom_encode.embedded_source_present": getattr(
                metrics, "embedded_source_present", None
            ),
            "axiom_encode.reasoning_entry_count": len(
                extract_reasoning_entries(trace_payload)
            ),
        }
    )

    events: list[tuple[str, Mapping[str, Any]]] = []
    retrieved_files = getattr(result, "retrieved_files", []) or []
    if retrieved_files:
        events.append(
            (
                "axiom_encode.retrieved_files",
                {"files_json": json.dumps(retrieved_files, sort_keys=True)},
            )
        )
    unexpected_accesses = getattr(result, "unexpected_accesses", []) or []
    if unexpected_accesses:
        events.append(
            (
                "axiom_encode.unexpected_accesses",
                {"commands_json": json.dumps(unexpected_accesses, sort_keys=True)},
            )
        )
    events.extend(_reasoning_span_events(trace_payload))

    client.emit_span(
        "axiom_encode.eval",
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        attributes=attributes,
        events=events,
        error=getattr(result, "error", None),
    )


def emit_agent_run(
    *,
    citation: str | None,
    session_id: str | None,
    agent_run: Any,
) -> None:
    """Export one orchestrator agent run to OTLP if configured."""
    client = _get_client()
    if client is None:
        return

    usage = getattr(agent_run, "total_tokens", None)
    if usage is None:
        usage = TokenUsage(
            reasoning_output_tokens=extract_reasoning_output_tokens(
                getattr(agent_run, "provider_trace", None)
            )
        )
    elif usage.reasoning_output_tokens == 0:
        usage.reasoning_output_tokens = extract_reasoning_output_tokens(
            getattr(agent_run, "provider_trace", None)
        )

    attributes = _base_llm_attributes(
        provider=(getattr(agent_run, "provider_trace", {}) or {}).get("provider"),
        model=(getattr(agent_run, "provider_trace", {}) or {}).get("model"),
        usage=usage,
        cost_usd=getattr(agent_run, "total_cost", None),
    )
    attributes.update(
        {
            "openinference.span.kind": "LLM",
            "axiom_encode.kind": "agent_run",
            "axiom_encode.citation": citation,
            "axiom_encode.session_id": session_id,
            "axiom_encode.agent_type": getattr(agent_run, "agent_type", None),
            "axiom_encode.phase": getattr(
                getattr(agent_run, "phase", None), "value", None
            ),
            "axiom_encode.message_count": len(getattr(agent_run, "messages", []) or []),
            "axiom_encode.reasoning_entry_count": len(
                extract_reasoning_entries(getattr(agent_run, "provider_trace", None))
            ),
        }
    )

    events: list[tuple[str, Mapping[str, Any]]] = []
    provider_trace = getattr(agent_run, "provider_trace", None)
    if provider_trace:
        summary = {
            "provider": provider_trace.get("provider"),
            "backend": provider_trace.get("backend"),
        }
        events.append(
            (
                "axiom_encode.provider_trace",
                {"summary_json": json.dumps(summary, sort_keys=True)},
            )
        )
        events.extend(_reasoning_span_events(provider_trace))

    client.emit_span(
        "axiom_encode.agent_run",
        start_time_ns=_datetime_to_ns(getattr(agent_run, "started_at", None)),
        end_time_ns=_datetime_to_ns(getattr(agent_run, "ended_at", None)),
        attributes=attributes,
        events=events,
        error=getattr(agent_run, "error", None),
    )


def _base_llm_attributes(
    *,
    provider: str | None,
    model: str | None,
    usage: TokenUsage | None,
    cost_usd: float | None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "gen_ai.system": provider,
        "gen_ai.response.model": model,
    }

    if usage is not None:
        attrs["gen_ai.usage.input_tokens"] = usage.input_tokens
        attrs["gen_ai.usage.output_tokens"] = usage.output_tokens
        attrs["gen_ai.usage.cache_read.input_tokens"] = usage.cache_read_tokens
        attrs["gen_ai.usage.cache_creation.input_tokens"] = usage.cache_creation_tokens
        attrs["axiom_encode.usage.reasoning_output_tokens"] = (
            usage.reasoning_output_tokens
        )

    if cost_usd is not None:
        attrs["axiom_encode.cost.total_usd"] = cost_usd

    if usage is not None and model:
        breakdown = estimate_usage_cost_breakdown(model, usage)
        if breakdown is not None:
            attrs.update(
                {
                    "axiom_encode.cost.input_usd": breakdown.input_cost_usd,
                    "axiom_encode.cost.cache_read_usd": breakdown.cache_read_cost_usd,
                    "axiom_encode.cost.cache_creation_usd": (
                        breakdown.cache_creation_cost_usd
                    ),
                    "axiom_encode.cost.output_usd": breakdown.output_cost_usd,
                    "axiom_encode.cost.reasoning_output_usd": (
                        breakdown.reasoning_output_cost_usd
                    ),
                    "axiom_encode.usage.non_cached_input_tokens": (
                        breakdown.non_cached_input_tokens
                    ),
                }
            )

    return attrs


def _get_client() -> ObservabilityClient | None:
    global _CLIENT, _CLIENT_CACHE_KEY, _IMPORT_ERROR_REPORTED

    config = _config_from_env()
    if config is None:
        return None

    cache_key = (
        config.endpoint,
        config.headers,
        config.service_name,
        config.project_name,
    )

    with _CLIENT_LOCK:
        if _CLIENT is not None and _CLIENT_CACHE_KEY == cache_key:
            return _CLIENT

        try:
            _CLIENT = ObservabilityClient(config)
            _CLIENT_CACHE_KEY = cache_key
            return _CLIENT
        except ImportError:
            if not _IMPORT_ERROR_REPORTED:
                print(
                    "Axiom Encode observability disabled: install axiom_encode[observability]",
                    file=os.sys.stderr,
                )
                _IMPORT_ERROR_REPORTED = True
            return None


def _config_from_env() -> ObservabilityConfig | None:
    if os.environ.get("AXIOM_ENCODE_DISABLE_OBSERVABILITY") == "1":
        return None

    traces_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if traces_endpoint:
        endpoint = traces_endpoint
    else:
        endpoint = _normalize_otlp_endpoint(
            os.environ.get("AXIOM_ENCODE_OTLP_ENDPOINT")
            or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        )

    if not endpoint:
        return None

    headers = _parse_headers(
        os.environ.get("AXIOM_ENCODE_OTLP_HEADERS")
        or os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")
        or ""
    )

    return ObservabilityConfig(
        endpoint=endpoint,
        headers=headers,
        service_name=os.environ.get("OTEL_SERVICE_NAME", "axiom_encode"),
        project_name=os.environ.get("AXIOM_ENCODE_PROJECT_NAME", "axiom_encode"),
    )


def _normalize_otlp_endpoint(endpoint: str | None) -> str | None:
    if not endpoint:
        return None
    stripped = endpoint.rstrip("/")
    if stripped.endswith("/v1/traces"):
        return stripped
    return f"{stripped}/v1/traces"


def _parse_headers(raw_headers: str) -> tuple[tuple[str, str], ...]:
    headers: list[tuple[str, str]] = []
    for part in raw_headers.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            headers.append((key, value))
    return tuple(headers)


def _window_from_duration_ms(duration_ms: int) -> tuple[int, int]:
    end_time_ns = time.time_ns()
    start_time_ns = end_time_ns - max(int(duration_ms), 0) * 1_000_000
    return start_time_ns, end_time_ns


def _datetime_to_ns(value: datetime | None) -> int | None:
    if value is None:
        return None
    return int(value.timestamp() * 1_000_000_000)


def _reasoning_tokens_from_usage(usage: Mapping[str, Any]) -> int:
    completion_details = (
        usage.get("completion_tokens_details")
        or usage.get("output_tokens_details")
        or {}
    )
    return int(completion_details.get("reasoning_tokens", 0) or 0)


def _extract_reasoning_from_payload(
    payload: Mapping[str, Any],
    entries: list[ReasoningEntry],
    seen: set[str],
    *,
    provider: str | None,
    source: str,
) -> None:
    """Extract reasoning texts from a generic provider payload."""
    payload_type = _as_optional_str(payload.get("type"))
    if payload_type == "reasoning":
        _append_reasoning_entry(
            entries,
            seen,
            text=payload.get("text"),
            provider=provider,
            source=source,
            item_id=_as_optional_str(payload.get("id")),
            item_type=payload_type,
        )
        for summary in _iter_reasoning_summary_texts(payload.get("summary")):
            _append_reasoning_entry(
                entries,
                seen,
                text=summary,
                provider=provider,
                source=source,
                item_id=_as_optional_str(payload.get("id")),
                item_type=payload_type,
            )

    content = payload.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, Mapping):
                _extract_reasoning_from_payload(
                    item,
                    entries,
                    seen,
                    provider=provider,
                    source=f"{source}.content",
                )

    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if isinstance(item, Mapping):
                _extract_reasoning_from_payload(
                    item,
                    entries,
                    seen,
                    provider=provider,
                    source=f"{source}.output",
                )


def _append_reasoning_entry(
    entries: list[ReasoningEntry],
    seen: set[str],
    *,
    text: Any,
    provider: str | None,
    source: str,
    item_id: str | None,
    item_type: str | None,
) -> None:
    """Add a normalized reasoning item when it is non-empty and novel."""
    if not isinstance(text, str):
        return

    normalized = text.strip()
    if not normalized or normalized in seen:
        return

    seen.add(normalized)
    entries.append(
        ReasoningEntry(
            text=normalized,
            provider=provider,
            source=source,
            item_id=item_id,
            item_type=item_type,
        )
    )


def _iter_reasoning_summary_texts(summary: Any) -> list[str]:
    """Flatten reasoning summary payloads into plain text."""
    if isinstance(summary, str):
        return [summary]
    if isinstance(summary, Mapping):
        text = summary.get("text")
        return [text] if isinstance(text, str) else []
    if isinstance(summary, list):
        texts: list[str] = []
        for item in summary:
            texts.extend(_iter_reasoning_summary_texts(item))
        return texts
    return []


def _reasoning_span_events(
    trace_payload: Mapping[str, Any] | None,
) -> list[tuple[str, Mapping[str, Any]]]:
    """Convert extracted reasoning entries into compact OTLP events."""
    events: list[tuple[str, Mapping[str, Any]]] = []
    for index, entry in enumerate(
        extract_reasoning_entries(trace_payload)[:10], start=1
    ):
        events.append(
            (
                "axiom_encode.reasoning",
                {
                    "index": index,
                    "source": entry.source or "",
                    "item_type": entry.item_type or "",
                    "text": _truncate_attr(entry.text, 1000),
                },
            )
        )
    return events


def _truncate_attr(text: str, limit: int) -> str:
    """Keep OTLP event attributes reasonably small."""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _as_optional_str(value: Any) -> str | None:
    """Convert an identifier-like value to string when present."""
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)
