"""
Tests for encoder backend abstraction.

Updated for self-contained backends (no plugin dependencies).
"""

import hashlib
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Import what we're testing
from axiom_encode.harness.backends import (
    AgentSDKBackend,
    ClaudeCodeBackend,
    CodexCLIBackend,
    EncoderBackend,
    EncoderRequest,
    EncoderResponse,
)
from axiom_encode.prompts import ENCODER_PROMPT, get_encoder_prompt
from axiom_encode.prompts.encoder import (
    _FORMULA_PROTOCOL,
    _TESTS_PROTOCOL,
    _assemble,
)


def test_assembled_prompt_has_no_duplicated_or_glued_blocks():
    """Regression guards for the protocol-block assembly (codex review).

    The prompt is composed by concatenating named protocol blocks; these
    assert the *assembled* string is clean, since a bad block boundary or a
    mis-partitioned block is invisible when you only inspect the blocks.
    """
    # Output-shape and prose-ban guidance must appear exactly once.
    assert ENCODER_PROMPT.count("Emit only RuleSpec YAML") == 1
    assert ENCODER_PROMPT.count("Do not emit Python code, markdown fences") == 1
    # No duplicated heading from a block that re-states the core header.
    assert "Hard requirements:Hard requirements:" not in ENCODER_PROMPT
    assert ENCODER_PROMPT.count("Hard requirements:") == 1
    # No block boundary glued two sentences together (the SOURCE_SCOPE /
    # composition seam previously produced "proxy.- If source text").
    assert "proxy.- If source text" not in ENCODER_PROMPT


def test_formula_and_tests_protocols_are_distinct_and_populated():
    assert _FORMULA_PROTOCOL.strip(), "_FORMULA_PROTOCOL must not be empty"
    assert "Put formulas under" in _FORMULA_PROTOCOL
    assert "min(uncapped_amount, max(cap_a," in _FORMULA_PROTOCOL
    # Test guidance lives in the tests protocol, formula guidance does not.
    assert "Emit only RuleSpec YAML" in _TESTS_PROTOCOL
    assert "Put formulas under" not in _TESTS_PROTOCOL


def test_assemble_strips_blocks_and_cannot_glue_or_stack():
    assembled = _assemble("- alpha statement.", "  - beta statement.  ", "", "- gamma.")
    assert assembled == "- alpha statement.\n\n- beta statement.\n\n- gamma."
    # No empty block leaves a triple newline, and no boundary glues words.
    assert "\n\n\n" not in assembled
    assert "statement.- beta" not in assembled


@pytest.fixture(autouse=True)
def mock_sdk_env(tmp_path):
    """Provide API key for AgentSDKBackend tests."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        yield


class TestEncoderBackendInterface:
    """Test the abstract backend interface."""

    def test_backend_is_abstract(self):
        """EncoderBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EncoderBackend()

    def test_request_dataclass(self):
        """EncoderRequest holds encoding inputs."""
        req = EncoderRequest(
            citation="26 USC 32",
            source_text="The earned income tax credit...",
            output_path=Path("/tmp/test.yaml"),
        )
        assert req.citation == "26 USC 32"
        assert req.source_text.startswith("The earned")
        assert req.output_path == Path("/tmp/test.yaml")

    def test_response_dataclass(self):
        """EncoderResponse holds encoding outputs."""
        resp = EncoderResponse(
            rulespec_content="eitc:\n  entity: TaxUnit",
            success=True,
            error=None,
            duration_ms=1500,
        )
        assert resp.success
        assert "eitc" in resp.rulespec_content
        assert resp.duration_ms == 1500


def _assert_encoder_prompt_topics(prompt: str) -> None:
    """Check durable prompt themes without freezing every wording tweak."""
    required_topics = [
        "module.proof_validation.required: true",
        "Do not emit `source_url`",
        "every source-stated amount, rate, threshold, cap, and limit",
        '"per taxpayer per beneficiary"',
        "Do not treat the final interval row as open-ended",
        "Do not invent synthetic row-number constants for text-label tables",
        "Do not use an `indexed_by` table with numeric keys merely to encode rows whose",
        "Use a negative sentinel such as `-1`",
        "source text `133%` should be",
        "kind: derived_relation",
        '"This source is about SNAP" is not enough',
        "Any rule that uses `entity: <filtered-entity>`",
        "Treat stale unit-scoped imports for that base as",
        "self-employment, wage, net-earnings, compensation, remuneration",
        "earned income of an individual shall be",
        "Never drop the jurisdiction prefix",
        "Importing an adjacent upstream output only as proof",
        "is not an executable dependency",
        "RuleSpec document-root `inputs`",
        "never nested under `module`",
        "purpose-limited replacement rate",
        "not as `section_<cited>_*`",
        "predicate for the excepted category",
        "`subject to`, or `notwithstanding`",
        "`subject to paragraph (c)`",
        "toggle each gate at least once",
        "Validation fails if a direct local `#input.*_exception_applies`",
        "Compute each expected `output:` value by evaluating the emitted RuleSpec",
        "flat\n  threshold with a percentage of excess income",
        "`clause_ii_provides_otherwise`",
        "paragraph_d_2_methodology_limit_satisfied",
        "dependency graph remains acyclic",
        "copied context file already exports the operative legal condition",
        "Do not\n  recreate it as a local factual input",
        "Axiom formulas have no date literal type",
        "Never use `post_YYYY`, `pre_YYYY`, `after_YYYY`, `before_YYYY`",
        "Do not write `else if` or `elif`",
        "min(source_amount, cap)",
        "compute the uncapped base amount separately",
        "min(uncapped_amount, max(cap_a,",
        "Do not\n  reconstruct the cited section's amount locally",
        "Only include `blocked_by` entries when you know the exact RuleSpec output",
        "us:statutes/us-ca/17000",
        "direct release-bound corpus source text",
    ]
    for topic in required_topics:
        assert topic in prompt


def test_generic_encoder_prompt_includes_durable_rule_spec_guidance():
    prompt = get_encoder_prompt(
        citation="26 USC 63(c)(5)",
        output_path="statutes/26/63/c/5.yaml",
        corpus_citation_path="us/statute/26/63",
    )

    _assert_encoder_prompt_topics(ENCODER_PROMPT)
    _assert_encoder_prompt_topics(prompt)
    assert "For 26 USC 1402(a)(12)" not in ENCODER_PROMPT
    normalized_prompt = " ".join(prompt.split())
    assert "Include `us/statute/26/63` in `module.source_verification`." in prompt
    assert (
        "module.source_verification.corpus_citation_path: us/statute/26/63"
        in normalized_prompt
    )
    assert (
        "Use that exact same `us/statute/26/63` value in every source-backed proof"
        in prompt
    )
    assert "do not append subsection markers or other path segments" in prompt
    assert "Never emit `corpus_citation_paths`" in normalized_prompt
    assert "corpus resolver under this one canonical path" in normalized_prompt
    assert "Target citation/source id: 26 USC 63(c)(5)" in prompt
    assert "Expected output path: statutes/26/63/c/5.yaml" in prompt


def test_complete_source_prompt_requires_explicit_branch_inventory():
    prompt = get_encoder_prompt(
        citation="42 USC 1437c-1",
        output_path="statutes/42/1437c-1.yaml",
        corpus_citation_path="us/statute/42/1437c–1",
        require_complete_source_unit=True,
    )
    normalized_prompt = " ".join(prompt.split())

    assert "inventory every top-level structural branch" in normalized_prompt
    assert "Do not return while any branch is absent" in normalized_prompt
    assert "absolute output path preserves the branch label" in normalized_prompt


class TestClaudeCodeBackend:
    """Test the Claude Code CLI backend (subprocess approach)."""

    def test_backend_inherits_interface(self):
        """ClaudeCodeBackend implements EncoderBackend."""
        backend = ClaudeCodeBackend()
        assert isinstance(backend, EncoderBackend)

    def test_encode_calls_subprocess(self):
        """encode() calls claude CLI via subprocess."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="test:\n  entity: TaxUnit",
                stderr="",
                returncode=0,
            )

            backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test statute",
                    output_path=Path("/tmp/test.yaml"),
                )
            )

            assert mock_run.called
            # Should use 'claude' command
            cmd = mock_run.call_args[0][0]
            assert "claude" in cmd

    def test_encode_uses_embedded_prompt(self):
        """encode() uses embedded encoder prompt (no plugin agent)."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="test:\n  entity: TaxUnit",
                stderr="",
                returncode=0,
            )

            backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=Path("/tmp/test.yaml"),
                )
            )

            cmd = mock_run.call_args[0][0]
            # Should NOT include --agent or --plugin-dir (self-contained)
            assert "--agent" not in cmd
            assert "--plugin-dir" not in cmd
            # Should include --print and -p flags
            assert "--print" in cmd
            assert "-p" in cmd

    def test_encode_confines_claude_to_output_directory(self, tmp_path):
        backend = ClaudeCodeBackend(cwd=tmp_path / "mutable-checkout")
        output = tmp_path / "isolated-output" / "rule.yaml"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="rules: []", stderr="", returncode=0)
            backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=output,
                )
            )

        cmd = mock_run.call_args[0][0]
        kwargs = mock_run.call_args.kwargs
        assert "bypassPermissions" not in cmd
        assert cmd[cmd.index("--permission-mode") + 1] == "acceptEdits"
        assert cmd[cmd.index("--tools") + 1] == "Read,Write,Edit"
        assert "--safe-mode" in cmd
        assert "--no-session-persistence" in cmd
        assert kwargs["cwd"] == output.parent.resolve()

    def test_predict_returns_scores(self):
        """predict() returns score predictions."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout='{"rulespec_reviewer": 8.0, "formula_reviewer": 7.5, "confidence": 0.8}',
                stderr="",
                returncode=0,
            )

            scores = backend.predict(
                citation="26 USC 32",
                source_text="EITC rules...",
            )

            assert scores.rulespec_reviewer == 8.0
            assert scores.confidence == 0.8


class TestAgentSDKBackend:
    """Test the Claude API backend (anthropic SDK)."""

    def test_backend_inherits_interface(self):
        """AgentSDKBackend implements EncoderBackend."""
        backend = AgentSDKBackend()
        assert isinstance(backend, EncoderBackend)

    def test_requires_api_key(self):
        """AgentSDKBackend requires ANTHROPIC_API_KEY."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove API key from env
            import os

            if "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]

            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                AgentSDKBackend(api_key=None)

    @pytest.mark.asyncio
    async def test_encode_async(self):
        """encode_async() uses anthropic SDK for async encoding."""
        backend = AgentSDKBackend(api_key="test-key")

        # Create mock for anthropic
        mock_anthropic = Mock()
        mock_client = Mock()

        mock_response = Mock()
        mock_response.content = [Mock(text="test:\n  entity: TaxUnit")]
        mock_response.usage = Mock(input_tokens=100, output_tokens=50)

        mock_client.messages = Mock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.AsyncAnthropic = Mock(return_value=mock_client)

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            resp = await backend.encode_async(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=Path("/tmp/test.yaml"),
                )
            )

            assert resp.success
            assert resp.tokens is not None

    @pytest.mark.asyncio
    async def test_encode_batch_parallel(self):
        """encode_batch() runs multiple encodings in parallel."""
        backend = AgentSDKBackend(api_key="test-key")

        requests = [
            EncoderRequest(
                citation=f"26 USC {i}",
                source_text=f"Statute {i}",
                output_path=Path(f"/tmp/test{i}.yaml"),
            )
            for i in range(5)
        ]

        with patch.object(backend, "encode_async") as mock_encode:
            mock_encode.return_value = EncoderResponse(
                rulespec_content="test",
                success=True,
                error=None,
                duration_ms=100,
            )

            responses = await backend.encode_batch(requests, max_concurrent=3)

            # All 5 should complete
            assert len(responses) == 5
            # encode_async should be called 5 times
            assert mock_encode.call_count == 5

    @pytest.mark.asyncio
    async def test_encode_batch_respects_concurrency_limit(self):
        """encode_batch() respects max_concurrent limit."""
        backend = AgentSDKBackend(api_key="test-key")
        concurrent_count = 0
        max_seen = 0

        async def track_concurrency(req):
            nonlocal concurrent_count, max_seen
            concurrent_count += 1
            max_seen = max(max_seen, concurrent_count)
            import asyncio

            await asyncio.sleep(0.01)  # Simulate work
            concurrent_count -= 1
            return EncoderResponse(
                rulespec_content="test",
                success=True,
                error=None,
                duration_ms=10,
            )

        with patch.object(backend, "encode_async", side_effect=track_concurrency):
            requests = [
                EncoderRequest(
                    citation=f"26 USC {i}",
                    source_text=f"Statute {i}",
                    output_path=Path(f"/tmp/test{i}.yaml"),
                )
                for i in range(10)
            ]

            await backend.encode_batch(requests, max_concurrent=3)

            # Should never exceed max_concurrent
            assert max_seen <= 3


class TestClaudeCodeBackendAdditional:
    """Additional tests for ClaudeCodeBackend to cover missing lines."""

    def test_encode_with_nonzero_returncode(self):
        """Test encode returns error when CLI returns non-zero."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="Error: failed to encode",
                stderr="",
                returncode=1,
            )

            resp = backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=Path("/tmp/test.yaml"),
                )
            )

            assert not resp.success
            assert resp.error is not None
            assert resp.rulespec_content == ""

    def test_encode_reads_file_when_exists(self, tmp_path):
        """Test encode reads from output_path when file exists."""
        backend = ClaudeCodeBackend()
        output_path = tmp_path / "output.yaml"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="CLI output ignored",
                stderr="",
                returncode=0,
            )
            # Create the file as if Claude wrote it
            output_path.write_text("file_var:\n  entity: TaxUnit\n")

            resp = backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=output_path,
                )
            )

            assert resp.success
            assert "file_var" in resp.rulespec_content

    def test_predict_no_json_in_output(self):
        """Test predict returns defaults when no JSON found in output."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="No JSON here, just plain text",
                stderr="",
                returncode=0,
            )

            scores = backend.predict("26 USC 1", "Statute text")
            # Should return defaults on error
            assert scores.confidence == 0.3

    def test_predict_returns_defaults_on_exception(self):
        """Test predict returns defaults when exception occurs."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("unexpected error")

            scores = backend.predict("26 USC 1", "Statute text")
            assert scores.confidence == 0.3

    def test_run_claude_code_uses_print_flag(self):
        """Test _run_claude_code uses --print flag (self-contained, no plugin)."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="test", stderr="", returncode=0)

            backend._run_claude_code("test prompt")

            cmd = mock_run.call_args[0][0]
            assert "--print" in cmd
            assert "--plugin-dir" not in cmd
            # Current Claude CLI versions validate --mcp-config as a record
            # with an mcpServers key; a bare "{}" fails every invocation.
            assert cmd[cmd.index("--mcp-config") + 1] == '{"mcpServers": {}}'

    def test_run_claude_code_timeout(self):
        """Test _run_claude_code handles timeout."""
        import subprocess

        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=300)

            output, code = backend._run_claude_code("test", timeout=300)
            assert "Timeout" in output
            assert code == 1

    def test_run_claude_code_file_not_found(self):
        """Test _run_claude_code handles missing CLI."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            output, code = backend._run_claude_code("test")
            assert "not found" in output
            assert code == 1

    def test_run_claude_code_generic_error(self):
        """Test _run_claude_code handles generic exception."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = OSError("Permission denied")

            output, code = backend._run_claude_code("test")
            assert "Error" in output
            assert code == 1


class TestAgentSDKBackendAdditional:
    """Additional tests for AgentSDKBackend to cover missing lines."""

    @pytest.mark.asyncio
    async def test_encode_async_with_usage(self):
        """Test encode_async captures token usage."""
        backend = AgentSDKBackend(api_key="test-key")

        mock_anthropic = Mock()
        mock_client = Mock()

        mock_response = Mock()
        mock_response.content = [Mock(text="encoded")]
        mock_response.usage = Mock(input_tokens=100, output_tokens=50)

        mock_client.messages = Mock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.AsyncAnthropic = Mock(return_value=mock_client)

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            resp = await backend.encode_async(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=Path("/tmp/nonexistent.yaml"),
                )
            )

            assert resp.success
            assert resp.tokens is not None
            assert resp.tokens.input_tokens == 100
            assert resp.tokens.output_tokens == 50

    @pytest.mark.asyncio
    async def test_encode_async_reads_file_if_exists(self, tmp_path):
        """Test encode_async reads from output_path if it exists."""
        backend = AgentSDKBackend(api_key="test-key")

        output_path = tmp_path / "output.yaml"
        output_path.write_text("file_content:\n  entity: TaxUnit\n")

        mock_anthropic = Mock()
        mock_client = Mock()

        mock_response = Mock()
        mock_response.content = [Mock(text="ignored")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)

        mock_client.messages = Mock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.AsyncAnthropic = Mock(return_value=mock_client)

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            resp = await backend.encode_async(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=output_path,
                )
            )

            assert resp.success
            assert "file_content" in resp.rulespec_content

    @pytest.mark.asyncio
    async def test_encode_async_import_error(self):
        """Test encode_async handles missing anthropic import."""
        backend = AgentSDKBackend(api_key="test-key")

        import builtins

        orig_import = builtins.__import__

        def no_anthropic_import(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("No module named 'anthropic'")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_anthropic_import):
            resp = await backend.encode_async(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=Path("/tmp/test.yaml"),
                )
            )

            assert not resp.success
            assert "not installed" in resp.error

    @pytest.mark.asyncio
    async def test_encode_async_generic_error(self):
        """Test encode_async handles generic exception."""
        backend = AgentSDKBackend(api_key="test-key")

        mock_anthropic = Mock()
        mock_client = Mock()
        mock_client.messages = Mock()
        mock_client.messages.create = AsyncMock(
            side_effect=RuntimeError("Connection failed")
        )
        mock_anthropic.AsyncAnthropic = Mock(return_value=mock_client)

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            resp = await backend.encode_async(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=Path("/tmp/test.yaml"),
                )
            )

            assert not resp.success
            assert "Connection failed" in resp.error


class TestAgentSDKPrediction:
    """Test AgentSDKBackend.predict() method."""

    def test_predict_returns_default_scores(self):
        """Test predict returns default PredictionScores."""
        backend = AgentSDKBackend(api_key="test-key")
        scores = backend.predict("26 USC 1", "Statute text")
        assert scores.confidence == 0.5


class TestCodexCLIBackend:
    """Test the Codex CLI backend."""

    def test_encode_calls_codex_exec(self):
        backend = CodexCLIBackend(cwd=Path("/tmp/work"))

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout='{"type":"item.completed","item":{"type":"agent_message","text":"test:\\n  entity: TaxUnit"}}\n{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5,"cached_input_tokens":2}}',
                stderr="",
                returncode=0,
            )

            backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test statute",
                    output_path=Path("/tmp/output/test.yaml"),
                    model="gpt-5.4",
                )
            )

            cmd = mock_run.call_args[0][0]
            assert Path(cmd[0]).name == "codex"
            assert cmd[1:3] == ["exec", "--json"]
            assert "--model" in cmd
            assert "gpt-5.4" in cmd
            assert "--add-dir" not in cmd
            assert cmd[cmd.index("-C") + 1] == str(Path("/tmp/output").resolve())
            assert mock_run.call_args.kwargs["cwd"] == Path("/tmp/output").resolve()

    def test_trusted_subscription_records_pinned_cli_provenance(
        self, tmp_path, monkeypatch
    ):
        binary = tmp_path / "codex"
        binary.write_bytes(b"pinned-codex")
        monkeypatch.setattr(
            "axiom_encode.harness.backends.resolve_codex_cli", lambda: str(binary)
        )
        monkeypatch.setenv("AXIOM_ENCODE_TRUSTED_RUNTIME", "1")
        monkeypatch.setenv("CODEX_HOME", str(tmp_path / "runtime-home"))
        digest = hashlib.sha256(b"pinned-codex").hexdigest()
        monkeypatch.setenv("AXIOM_ENCODE_TRUSTED_CODEX_VERSION", "codex-cli 0.test")
        monkeypatch.setenv("AXIOM_ENCODE_TRUSTED_CODEX_SHA256", digest)
        version, digest = CodexCLIBackend._trusted_cli_provenance()
        assert version == "codex-cli 0.test"
        assert digest == hashlib.sha256(b"pinned-codex").hexdigest()

    def test_trusted_subscription_executes_bound_cli_not_path_decoy(
        self, tmp_path, monkeypatch
    ):
        trusted = tmp_path / "trusted-codex"
        trusted.write_text("#!/bin/sh\nexit 0\n")
        trusted.chmod(0o700)
        decoy_dir = tmp_path / "decoy-bin"
        decoy_dir.mkdir()
        decoy_marker = tmp_path / "decoy-executed"
        decoy = decoy_dir / "codex"
        decoy.write_text(f"#!/bin/sh\ntouch {decoy_marker}\nexit 99\n")
        decoy.chmod(0o700)
        monkeypatch.setenv("AXIOM_ENCODE_TRUSTED_RUNTIME", "1")
        monkeypatch.setenv("CODEX_HOME", str(tmp_path / "runtime-home"))
        monkeypatch.setenv("AXIOM_ENCODE_TRUSTED_CODEX_BIN", str(trusted))
        monkeypatch.setenv("PATH", f"{decoy_dir}{os.pathsep}{os.environ['PATH']}")

        output, returncode = CodexCLIBackend()._run_codex_exec(
            "prompt", "gpt-test", 10, tmp_path
        )

        assert returncode == 0, output
        assert not decoy_marker.exists()

    def test_encode_parses_jsonl_usage(self):
        backend = CodexCLIBackend(cwd=Path("/tmp/work"))

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout='{"type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"reasoning_output_tokens":7}}}}\n{"type":"item.completed","item":{"type":"agent_message","text":"test:\\n  entity: TaxUnit"}}\n{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5,"cached_input_tokens":2}}',
                stderr="",
                returncode=0,
            )

            response = backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test statute",
                    output_path=Path("/tmp/output/test.yaml"),
                    model="gpt-5.4",
                )
            )

            assert response.success
            assert response.tokens is not None
            assert response.tokens.input_tokens == 10
            assert response.tokens.output_tokens == 5
            assert response.tokens.cache_read_tokens == 2
            assert response.tokens.reasoning_output_tokens == 7
            assert response.trace is not None
            assert response.trace["provider"] == "openai"


class TestBackendContract:
    """Test that each backend returns the shared response contract."""

    def test_both_backends_return_encoder_response(self):
        """Both backends return EncoderResponse from encode()."""
        # This ensures the abstraction is clean

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="test:\n  entity: TaxUnit",
                stderr="",
                returncode=0,
            )

            cli_backend = ClaudeCodeBackend()
            cli_resp = cli_backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=Path("/tmp/test.yaml"),
                )
            )

            assert isinstance(cli_resp, EncoderResponse)

    def test_sync_wrapper_for_sdk_backend(self):
        """AgentSDKBackend.encode() provides sync wrapper."""
        backend = AgentSDKBackend(api_key="test-key")

        with patch.object(backend, "encode_async") as mock_async:
            mock_async.return_value = EncoderResponse(
                rulespec_content="test",
                success=True,
                error=None,
                duration_ms=100,
            )

            # Sync encode() should work
            resp = backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=Path("/tmp/test.yaml"),
                )
            )

            assert isinstance(resp, EncoderResponse)


class TestCodexAuthPreflight:
    """Codex-backend auth preflight helpers (encode#1054)."""

    def test_auth_path_defaults_to_home_codex(self):
        from axiom_encode.codex_cli import codex_auth_json_path

        with patch.dict(os.environ, {}, clear=True):
            assert codex_auth_json_path() == Path.home() / ".codex" / "auth.json"

    def test_trusted_runtime_resolution_uses_supervisor_bound_path(self, monkeypatch):
        from axiom_encode.codex_cli import resolve_codex_cli

        monkeypatch.setenv("AXIOM_ENCODE_TRUSTED_RUNTIME", "1")
        monkeypatch.setenv("CODEX_HOME", "/protected/runtime-codex-home")
        monkeypatch.setenv("AXIOM_ENCODE_CODEX_BIN", "/hostile/override")
        monkeypatch.setenv("AXIOM_ENCODE_TRUSTED_CODEX_BIN", "/trusted/bin/codex")
        assert resolve_codex_cli() == "/trusted/bin/codex"

    def test_auth_path_honors_codex_home(self, tmp_path):
        from axiom_encode.codex_cli import codex_auth_json_path

        with patch.dict(os.environ, {"CODEX_HOME": str(tmp_path)}, clear=True):
            assert codex_auth_json_path() == tmp_path / "auth.json"

    def test_auth_error_when_no_file_and_no_key(self, tmp_path):
        from axiom_encode.codex_cli import codex_auth_error

        with patch.dict(os.environ, {"CODEX_HOME": str(tmp_path)}, clear=True):
            error = codex_auth_error()
        assert error is not None
        assert "Codex backend requires authentication" in error
        assert str(tmp_path / "auth.json") in error

    def test_auth_ok_when_file_present(self, tmp_path):
        from axiom_encode.codex_cli import codex_auth_error

        (tmp_path / "auth.json").write_text('{"OPENAI_API_KEY": "sk-test"}\n')
        with patch.dict(os.environ, {"CODEX_HOME": str(tmp_path)}, clear=True):
            assert codex_auth_error() is None

    def test_auth_ok_when_openai_api_key_set(self, tmp_path):
        from axiom_encode.codex_cli import codex_auth_error

        # No auth.json, but OPENAI_API_KEY is enough for the Codex CLI.
        with patch.dict(
            os.environ,
            {"CODEX_HOME": str(tmp_path), "OPENAI_API_KEY": "sk-test"},
            clear=True,
        ):
            assert codex_auth_error() is None
