"""
Encoder backends - abstraction for different Claude invocation methods.

Two backends:
1. ClaudeCodeBackend - uses Claude Code CLI (subprocess), works with Max subscription
2. AgentSDKBackend - uses Claude API (anthropic SDK), enables parallelization

Both use embedded prompts -- no external plugin dependencies.
"""

import asyncio
import hashlib
import json
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from axiom_encode.codex_cli import resolve_codex_cli
from axiom_encode.constants import DEFAULT_CLI_MODEL, DEFAULT_MODEL
from axiom_encode.prompts.encoder import get_encoder_prompt

from .encoding_db import TokenUsage
from .eval_evidence import scrub_attestation_signing_keys
from .observability import extract_reasoning_output_tokens
from .pricing import estimate_usage_cost_usd


@dataclass
class EncoderRequest:
    """Input for an encoding operation."""

    citation: str
    source_text: str
    output_path: Path
    corpus_citation_path: str | None = None
    model: str = DEFAULT_CLI_MODEL
    timeout: int = 300
    require_complete_source_unit: bool = False


@dataclass
class EncoderResponse:
    """Output from an encoding operation."""

    rulespec_content: str
    success: bool
    error: Optional[str] = None
    duration_ms: int = 0
    tokens: Optional[TokenUsage] = None
    cost_usd: Optional[float] = None
    trace: Optional[dict[str, Any]] = None
    codex_cli_version: Optional[str] = None
    codex_cli_sha256: Optional[str] = None


@dataclass
class PredictionScores:
    """Predicted scores from the encoder."""

    rulespec_reviewer: float = 7.0
    formula_reviewer: float = 7.0
    parameter_reviewer: float = 7.0
    integration_reviewer: float = 7.0
    ci_pass: bool = True
    policyengine_match: Optional[float] = None
    confidence: float = 0.5


class EncoderBackend(ABC):
    """Abstract base class for encoder backends."""

    @abstractmethod
    def encode(self, request: EncoderRequest) -> EncoderResponse:
        """Encode a source text to RuleSpec format (synchronous)."""
        pass  # pragma: no cover

    @abstractmethod
    def predict(self, citation: str, source_text: str) -> PredictionScores:
        """Predict quality scores before encoding."""
        pass  # pragma: no cover


def parse_claude_cli_json_output(
    output: str,
    model: str,
) -> Optional[dict[str, Any]]:
    """Parse Claude CLI JSON output when available."""
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict) or payload.get("type") != "result":
        return None

    usage = payload.get("usage", {}) or {}
    tokens = TokenUsage(
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        cache_read_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
        cache_creation_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
        reasoning_output_tokens=extract_reasoning_output_tokens(
            {"provider": "anthropic", "json_result": payload}
        ),
    )
    return {
        "text": payload.get("result", "") or "",
        "tokens": (
            tokens
            if (
                tokens.total_tokens
                or tokens.cache_creation_tokens
                or tokens.cache_read_tokens
            )
            else None
        ),
        "cost_usd": payload.get("total_cost_usd"),
        "is_error": bool(payload.get("is_error")),
        "trace": {
            "provider": "anthropic",
            "backend": "cli",
            "model": model,
            "json_result": payload,
        },
    }


class ClaudeCodeBackend(EncoderBackend):
    """
    Backend using Claude Code CLI (subprocess).

    Works with Max subscription - no API billing.
    Uses embedded prompts -- no external plugin needed.
    """

    def __init__(
        self,
        cwd: Optional[Path] = None,
    ):
        self.cwd = cwd or Path.cwd()

    def encode(self, request: EncoderRequest) -> EncoderResponse:
        """Encode using Claude Code CLI with embedded encoder prompt."""
        start = time.time()

        request.output_path.parent.mkdir(parents=True, exist_ok=True)

        prompt = get_encoder_prompt(
            request.citation,
            str(request.output_path),
            corpus_citation_path=request.corpus_citation_path,
            require_complete_source_unit=request.require_complete_source_unit,
        )
        prompt += f"\n\nSource Text:\n{request.source_text}\n"

        output, returncode = self._run_claude_code(
            prompt=prompt,
            model=request.model,
            timeout=request.timeout,
            writable_dir=request.output_path.parent,
        )

        duration_ms = int((time.time() - start) * 1000)
        parsed = self._parse_claude_json_output(output, request.model)
        result_text = parsed["text"] if parsed else output
        token_usage = parsed["tokens"] if parsed else None
        cost_usd = parsed["cost_usd"] if parsed else None
        trace = parsed["trace"] if parsed else None

        if returncode != 0 or (parsed and parsed.get("is_error")):
            return EncoderResponse(
                rulespec_content="",
                success=False,
                error=result_text or output,
                duration_ms=duration_ms,
                tokens=token_usage,
                cost_usd=cost_usd,
                trace=trace,
            )

        # Check if file was created.
        if request.output_path.exists():
            rulespec_content = request.output_path.read_text()
        else:
            rulespec_content = result_text

        # Clean up markdown code blocks
        rulespec_content = re.sub(r"^```\w*\n", "", rulespec_content)
        rulespec_content = re.sub(r"\n```$", "", rulespec_content)
        rulespec_content = rulespec_content.strip()

        return EncoderResponse(
            rulespec_content=rulespec_content,
            success=True,
            error=None,
            duration_ms=duration_ms,
            tokens=token_usage,
            cost_usd=cost_usd,
            trace=trace,
        )

    def predict(self, citation: str, source_text: str) -> PredictionScores:
        """Predict scores using Claude Code CLI."""
        prompt = f"""Predict quality scores for encoding the following source text into RuleSpec.

Citation: {citation}

Source Text:
{source_text[:2000]}{"..." if len(source_text) > 2000 else ""}

Score each dimension from 1-10. Output ONLY valid JSON:
{{
  "rulespec_reviewer": <float 1-10>,
  "formula_reviewer": <float 1-10>,
  "parameter_reviewer": <float 1-10>,
  "integration_reviewer": <float 1-10>,
  "ci_pass": <boolean>,
  "policyengine_match": <float 0-1>,
  "confidence": <float 0-1>
}}
"""

        try:
            output, returncode = self._run_claude_code(
                prompt=prompt,
                model=DEFAULT_CLI_MODEL,
                timeout=60,
            )
            parsed_output = self._parse_claude_json_output(output, DEFAULT_CLI_MODEL)
            text_output = parsed_output["text"] if parsed_output else output

            # Parse JSON from output
            json_match = re.search(r"\{[^{}]*\}", text_output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in output")

            return PredictionScores(
                rulespec_reviewer=float(data.get("rulespec_reviewer", 7.0)),
                formula_reviewer=float(data.get("formula_reviewer", 7.0)),
                parameter_reviewer=float(data.get("parameter_reviewer", 7.0)),
                integration_reviewer=float(data.get("integration_reviewer", 7.0)),
                ci_pass=bool(data.get("ci_pass", True)),
                policyengine_match=data.get("policyengine_match"),
                confidence=float(data.get("confidence", 0.5)),
            )

        except Exception:
            # Return defaults on error
            return PredictionScores(confidence=0.3)

    def _run_claude_code(
        self,
        prompt: str,
        model: str = DEFAULT_CLI_MODEL,
        timeout: int = 300,
        writable_dir: Path | None = None,
    ) -> tuple[str, int]:
        """Run Claude in one isolated output directory without shell authority."""
        execution_root = (writable_dir or self.cwd).resolve()
        tools = "Read,Write,Edit" if writable_dir is not None else ""
        permission_mode = "acceptEdits" if writable_dir is not None else "dontAsk"
        cmd = [
            "claude",
            "--print",
            "--output-format",
            "json",
            "--permission-mode",
            permission_mode,
            "--safe-mode",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--no-chrome",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers": {}}',
            "--tools",
            tools,
            "--allowed-tools",
            tools,
        ]

        if model:
            cmd.extend(["--model", model])

        cmd.extend(["-p", prompt])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=execution_root,
                env=scrub_attestation_signing_keys(),
            )
            return result.stdout + result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return f"Timeout after {timeout}s", 1
        except FileNotFoundError:
            return (
                "Claude CLI not found - install with: npm install -g @anthropic-ai/claude-code",
                1,
            )
        except Exception as e:
            return f"Error running Claude CLI: {e}", 1

    def _parse_claude_json_output(
        self,
        output: str,
        model: str,
    ) -> Optional[dict[str, Any]]:
        """Parse Claude CLI JSON output when available."""
        return parse_claude_cli_json_output(output, model)


class CodexCLIBackend(EncoderBackend):
    """
    Backend using the local Codex CLI.

    This provides a GPT-backed comparison path without a direct API integration.
    """

    def __init__(
        self,
        cwd: Optional[Path] = None,
        sandbox: str = "workspace-write",
    ):
        self.cwd = cwd or Path.cwd()
        self.sandbox = sandbox

    def encode(self, request: EncoderRequest) -> EncoderResponse:
        """Encode using codex exec with structured JSONL output."""
        start = time.time()
        request.output_path.parent.mkdir(parents=True, exist_ok=True)

        prompt = get_encoder_prompt(
            request.citation,
            str(request.output_path),
            corpus_citation_path=request.corpus_citation_path,
            require_complete_source_unit=request.require_complete_source_unit,
        )
        prompt += f"\n\nSource Text:\n{request.source_text}\n"

        codex_cli_version, codex_cli_sha256 = self._trusted_cli_provenance()
        output, returncode = self._run_codex_exec(
            prompt=prompt,
            model=request.model,
            timeout=request.timeout,
            writable_dir=request.output_path.parent,
        )
        duration_ms = int((time.time() - start) * 1000)
        parsed = self._parse_codex_json_output(output, request.model)
        result_text = parsed["text"] if parsed else output
        token_usage = parsed["tokens"] if parsed else None
        cost_usd = (
            parsed["cost_usd"]
            if parsed and parsed.get("cost_usd") is not None
            else estimate_usage_cost_usd(request.model, token_usage)
        )
        trace = parsed["trace"] if parsed else None

        if returncode != 0:
            return EncoderResponse(
                rulespec_content="",
                success=False,
                error=result_text or output,
                duration_ms=duration_ms,
                tokens=token_usage,
                cost_usd=cost_usd,
                trace=trace,
                codex_cli_version=codex_cli_version,
                codex_cli_sha256=codex_cli_sha256,
            )

        if request.output_path.exists():
            rulespec_content = request.output_path.read_text()
        else:
            rulespec_content = result_text

        rulespec_content = re.sub(r"^```\w*\n", "", rulespec_content)
        rulespec_content = re.sub(r"\n```$", "", rulespec_content)
        rulespec_content = rulespec_content.strip()

        return EncoderResponse(
            rulespec_content=rulespec_content,
            success=True,
            error=None,
            duration_ms=duration_ms,
            tokens=token_usage,
            cost_usd=cost_usd,
            trace=trace,
            codex_cli_version=codex_cli_version,
            codex_cli_sha256=codex_cli_sha256,
        )

    @staticmethod
    def _trusted_cli_provenance() -> tuple[str | None, str | None]:
        if not (
            os.environ.get("AXIOM_ENCODE_TRUSTED_RUNTIME") == "1"
            and os.environ.get("CODEX_HOME")
        ):
            return None, None
        executable = Path(resolve_codex_cli()).resolve(strict=True)
        expected_digest = os.environ.get("AXIOM_ENCODE_TRUSTED_CODEX_SHA256")
        version = os.environ.get("AXIOM_ENCODE_TRUSTED_CODEX_VERSION")
        if not expected_digest or not version:
            raise RuntimeError("Trusted runtime omitted Codex CLI provenance")
        digest = hashlib.sha256(executable.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise RuntimeError(
                "Trusted Codex CLI changed after supervisor verification"
            )
        return version, digest

    def predict(self, citation: str, source_text: str) -> PredictionScores:
        """Predict scores using Codex CLI."""
        return PredictionScores(confidence=0.5)

    def _run_codex_exec(
        self,
        prompt: str,
        model: str,
        timeout: int,
        writable_dir: Path,
    ) -> tuple[str, int]:
        """Run codex exec with writes confined to the output directory."""
        execution_root = writable_dir.resolve()
        cmd = [
            resolve_codex_cli(),
            "exec",
            "--json",
            "--model",
            model,
            "-C",
            str(execution_root),
            "--sandbox",
            self.sandbox,
        ]
        cmd.append(prompt)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=execution_root,
                env=scrub_attestation_signing_keys(),
            )
            return result.stdout + result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return f'{{"type":"error","message":"Timeout after {timeout}s"}}', 1
        except FileNotFoundError:
            return '{"type":"error","message":"codex CLI not found"}', 1
        except Exception as e:
            return f'{{"type":"error","message":"Error running codex CLI: {e}"}}', 1

    def _parse_codex_json_output(
        self,
        output: str,
        model: str,
    ) -> Optional[dict[str, Any]]:
        """Parse Codex JSONL event stream."""
        events: list[dict[str, Any]] = []
        assistant_messages: list[str] = []
        usage_payload: dict[str, Any] | None = None
        last_error: str | None = None

        for line in output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            if isinstance(payload, dict):
                events.append(payload)
                if payload.get("type") == "item.completed":
                    item = payload.get("item", {}) or {}
                    if item.get("type") == "agent_message" and item.get("text"):
                        assistant_messages.append(item["text"])
                elif payload.get("type") == "turn.completed":
                    usage_payload = payload.get("usage") or {}
                elif payload.get("type") == "error":
                    last_error = payload.get("message") or "codex exec error"

        if not events:
            return None

        tokens = None
        if usage_payload is not None:
            tokens = TokenUsage(
                input_tokens=int(usage_payload.get("input_tokens", 0) or 0),
                output_tokens=int(usage_payload.get("output_tokens", 0) or 0),
                cache_read_tokens=int(usage_payload.get("cached_input_tokens", 0) or 0),
                reasoning_output_tokens=extract_reasoning_output_tokens(
                    {
                        "provider": "openai",
                        "events": events,
                    }
                ),
            )
            if not tokens.total_tokens and not tokens.cache_read_tokens:
                tokens = None

        return {
            "text": "\n".join(assistant_messages).strip() or last_error or "",
            "tokens": tokens,
            "cost_usd": estimate_usage_cost_usd(model, tokens),
            "trace": {
                "provider": "openai",
                "backend": "codex-cli",
                "model": model,
                "events": events,
            },
        }


class AgentSDKBackend(EncoderBackend):
    """
    Backend using Claude API (anthropic SDK).

    Requires ANTHROPIC_API_KEY - pay per token.
    Enables parallelization for batch encoding.
    Uses embedded prompts -- no external plugin needed.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required for AgentSDKBackend")
        self.model = model or DEFAULT_MODEL

    def encode(self, request: EncoderRequest) -> EncoderResponse:
        """Synchronous encode using API (runs async under the hood)."""
        return asyncio.run(self.encode_async(request))

    async def encode_async(self, request: EncoderRequest) -> EncoderResponse:
        """Async encode using Claude API with embedded prompts."""
        start = time.time()

        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=self.api_key)

            prompt = get_encoder_prompt(
                request.citation,
                str(request.output_path),
                corpus_citation_path=request.corpus_citation_path,
                require_complete_source_unit=request.require_complete_source_unit,
            )
            prompt += f"\n\nSource Text:\n{request.source_text}\n"

            response = await client.messages.create(
                model=self.model,
                max_tokens=16384,
                messages=[{"role": "user", "content": prompt}],
            )

            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason in {"max_tokens", "model_context_window_exceeded"}:
                return EncoderResponse(
                    rulespec_content="",
                    success=False,
                    error=(
                        "Agent API response was truncated "
                        f"(stop_reason={stop_reason}); refusing partial output"
                    ),
                    duration_ms=int((time.time() - start) * 1000),
                )

            result_content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    result_content += block.text

            token_usage = TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            duration_ms = int((time.time() - start) * 1000)

            # Check if file was created.
            if request.output_path.exists():
                rulespec_content = request.output_path.read_text()
            else:
                rulespec_content = result_content

            return EncoderResponse(
                rulespec_content=rulespec_content,
                success=True,
                error=None,
                duration_ms=duration_ms,
                tokens=token_usage if token_usage.total_tokens > 0 else None,
            )

        except ImportError:
            return EncoderResponse(
                rulespec_content="",
                success=False,
                error="anthropic SDK not installed. Run: pip install anthropic",
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return EncoderResponse(
                rulespec_content="",
                success=False,
                error=str(e),
                duration_ms=int((time.time() - start) * 1000),
            )

    async def encode_batch(
        self,
        requests: List[EncoderRequest],
        max_concurrent: int = 5,
    ) -> List[EncoderResponse]:
        """
        Encode multiple source units in parallel.

        This is the key advantage of the API backend - parallelization.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def encode_with_limit(req: EncoderRequest) -> EncoderResponse:
            async with semaphore:
                return await self.encode_async(req)

        tasks = [encode_with_limit(req) for req in requests]
        return await asyncio.gather(*tasks)

    def predict(self, citation: str, source_text: str) -> PredictionScores:
        """Predict scores using API."""
        # For now, use defaults - prediction is less critical than encoding
        return PredictionScores(confidence=0.5)
