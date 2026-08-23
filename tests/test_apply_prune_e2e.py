"""End-to-end prune-on-supersede through the real apply path (axiom-encode#1282).

These drive ``_apply_generated_encoding_result`` — staging, signing, the live
install transaction, and the post-install closure — with predecessor records
present, on the existing ``TestCmdEncode`` harness.  The first revision of
prune-on-supersede planned inside the isolated signing stage and never fired;
the second planned correctly but the transaction allowlist rejected the
targets and the closure rolled jurisdiction-tree retirements back.  Helper
tests cannot catch either class; these can.  (Originally contributed by the
round-3 Fable defensive audit.)
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from axiom_encode import manifest_audit
from axiom_encode.cli import _apply_generated_encoding_result, _sha256_file
from tests.test_cli import (  # noqa: F401 - autouse fixtures re-exported on purpose
    APPLIED_ENCODING_SIGNING_PUBLIC_KEY_ENV,
    AXIOM_ENCODE_TEST_VERSION,
    TEST_APPLY_PUBLIC_KEY_B64,
    TEST_PINNED_ENCODER_IDENTITY,
    _pin_test_encoder_execution_identity,
    _test_eval_evidence_keys,
)
from tests.test_cli import (
    TestCmdEncode as _CmdEncodeHarness,
)

REL = "regulations/18-nycrr/387/12/f/3/v/c.yaml"
OLD_RULE = "format: rulespec/v1\nrules: [old]\n"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class TestApplyPruneEndToEnd(_CmdEncodeHarness):
    """Inherits the harness (autouse preflight fixture, helpers), not its tests."""

    def _run(self, tmp_path, *, setup):
        output_root = tmp_path / "out"
        policy_checkout = tmp_path / "rulespec-us"
        policy_repo = policy_checkout / "us-ny"
        generated = (
            output_root
            / "codex-test-model"
            / "regulations"
            / "18-nycrr"
            / "387"
            / "12"
            / "f"
            / "3"
            / "v"
            / "c.yaml"
        )
        generated.parent.mkdir(parents=True)
        generated.write_text("format: rulespec/v1\nrules: []\n")
        generated.with_name("c.test.yaml").write_text("[]\n")
        policy_repo.mkdir(parents=True)
        result = self._make_eval_result(True)
        result.output_file = str(generated)
        result.context_manifest_file = str(tmp_path / "context.json")
        result.trace_file = str(tmp_path / "trace.json")
        result.generation_prompt_sha256 = "prompt-sha"
        self._write_result_context(result, tmp_path)
        Path(result.trace_file).write_text("{}\n")
        corpus_path, result.source_attestation = self._bind_apply_source_release(
            policy_checkout,
            tmp_path,
            citation_path="us-ny/regulation/18-nycrr/387/12/f/3/v/c",
        )
        setup(policy_checkout, policy_repo)
        self._record_apply_validation(
            result,
            output_root=output_root,
            policy_repo_path=policy_repo,
            corpus_path=corpus_path,
        )
        with (
            patch.dict(
                os.environ,
                {APPLIED_ENCODING_SIGNING_PUBLIC_KEY_ENV: TEST_APPLY_PUBLIC_KEY_B64},
            ),
            patch(
                "axiom_encode.cli._git_repo_provenance",
                return_value={
                    "root": "/repo/axiom-encode",
                    "commit": TEST_PINNED_ENCODER_IDENTITY["commit"],
                    "dirty_tracked": False,
                },
            ),
            patch(
                "axiom_encode.cli._require_axiom_encode_version_provenance",
                return_value={
                    "version": AXIOM_ENCODE_TEST_VERSION,
                    "version_commit": "version123",
                    "identity_source": "git",
                },
            ),
        ):
            applied = _apply_generated_encoding_result(
                result,
                output_root=output_root,
                policy_repo_path=policy_repo,
                corpus_path=corpus_path,
                run_id="run-123",
            )
        return policy_checkout, policy_repo, applied

    def test_jurisdiction_tree_and_root_tree_predecessors(self, tmp_path, capsys):
        def setup(checkout: Path, content_root: Path) -> None:
            target = content_root / REL
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(OLD_RULE)
            sibling = content_root / "regulations/18-nycrr/387/12/f/3/v/d.yaml"
            sibling.write_text("format: rulespec/v1\nrules: [d]\n")
            # Jurisdiction-tree predecessor, fully superseded: deleted through
            # the transaction (dead layout, delete-only allowance).
            juris = content_root / (
                ".axiom/encoding-manifests/regulations/18-nycrr/387/12/f/3/v/c.json"
            )
            juris.parent.mkdir(parents=True)
            juris.write_text(
                json.dumps({"applied_files": [{"path": REL, "sha256": _sha(OLD_RULE)}]})
                + "\n"
            )
            # Legacy root predecessor keyed country-relative, partially
            # superseded: its claim on c.yaml is retired in place.
            root = checkout / (
                ".axiom/encoding-manifests/regulations/18-nycrr/387/12/f/3/v/c.json"
            )
            root.parent.mkdir(parents=True)
            root.write_text(
                json.dumps(
                    {
                        "applied_files": [
                            {"path": REL, "sha256": _sha(OLD_RULE)},
                            {
                                "path": "regulations/18-nycrr/387/12/f/3/v/d.yaml",
                                "sha256": _sha256_file(sibling),
                            },
                        ]
                    }
                )
                + "\n"
            )

        checkout, content_root, _applied = self._run(tmp_path, setup=setup)
        out = capsys.readouterr().out
        juris = content_root / (
            ".axiom/encoding-manifests/regulations/18-nycrr/387/12/f/3/v/c.json"
        )
        root = checkout / (
            ".axiom/encoding-manifests/regulations/18-nycrr/387/12/f/3/v/c.json"
        )
        assert not juris.exists(), "jurisdiction-tree predecessor should be retired"
        assert root.exists(), "partially superseded root record must survive"
        assert (checkout / manifest_audit.RATCHET_RELATIVE_PATH).is_file()
        assert "Retired superseded apply manifest" in out
        assert not (checkout / ".axiom/.apply-transaction").exists()
        final = manifest_audit.audit_repository(checkout)
        assert final.passed, [str(f) for f in final.findings]
        assert final.retired == 1

    def test_nested_tree_predecessor_is_retired_in_place(self, tmp_path):
        """A record the transaction cannot delete is retired by entry instead."""
        state: dict[str, Path] = {}

        def setup(checkout: Path, content_root: Path) -> None:
            target = content_root / REL
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(OLD_RULE)
            nested = content_root / (
                "regulations/.axiom/encoding-manifests/18-nycrr/387/12/f/3/v/c.json"
            )
            nested.parent.mkdir(parents=True)
            nested.write_text(
                json.dumps(
                    {
                        "applied_files": [
                            {
                                "path": "18-nycrr/387/12/f/3/v/c.yaml",
                                "sha256": _sha(OLD_RULE),
                            }
                        ]
                    }
                )
                + "\n"
            )
            state.update(checkout=checkout, nested=nested)

        checkout, _content_root, _applied = self._run(tmp_path, setup=setup)
        assert state["nested"].exists()
        assert not (checkout / ".axiom/.apply-transaction").exists()
        final = manifest_audit.audit_repository(checkout)
        assert final.passed, [str(f) for f in final.findings]
        assert final.retired == 1

    def test_predecessor_changed_after_planning_rolls_everything_back(self, tmp_path):
        from axiom_encode import cli

        state: dict[str, object] = {}

        def setup(checkout: Path, content_root: Path) -> None:
            target = content_root / REL
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(OLD_RULE)
            juris = checkout / (
                ".axiom/encoding-manifests/regulations/18-nycrr/387/12/f/3/v/c.json"
            )
            juris.parent.mkdir(parents=True)
            juris.write_text(
                json.dumps({"applied_files": [{"path": REL, "sha256": _sha(OLD_RULE)}]})
                + "\n"
            )
            state.update(juris=juris, target=target, checkout=checkout)

        real_install = cli._install_apply_transaction

        def racing_install(files, **kw):
            # Another writer expands the predecessor after planning, before
            # the locked install.
            juris = state["juris"]
            juris.write_text(
                json.dumps(
                    {
                        "applied_files": [
                            {"path": REL, "sha256": _sha(OLD_RULE)},
                            {
                                "path": "regulations/18-nycrr/387/12/f/3/v/z.yaml",
                                "sha256": "ab" * 32,
                            },
                        ]
                    }
                )
                + "\n"
            )
            state["grown"] = juris.read_bytes()
            return real_install(files, **kw)

        with patch(
            "axiom_encode.cli._install_apply_transaction", side_effect=racing_install
        ):
            with pytest.raises(RuntimeError) as info:
                self._run(tmp_path, setup=setup)
        message = str(info.value)
        assert (
            "plan changed before the install lock" in message
            or "changed after validation" in message
        ), message
        checkout = state["checkout"]
        assert state["juris"].read_bytes() == state["grown"]
        assert state["target"].read_text() == OLD_RULE
        assert not (
            checkout
            / ".axiom/encoding-manifests/us-ny/regulations/18-nycrr/387/12/f/3/v/c.json"
        ).exists()
        assert not (checkout / ".axiom/.apply-transaction").exists()
        assert not (checkout / manifest_audit.RATCHET_RELATIVE_PATH).exists()

    def test_install_that_would_worsen_the_audit_rolls_back(self, tmp_path):
        """The generic backstop: an apply must never add an audit finding.

        With the planner neutralized, a partially superseded predecessor would
        be left stale; the post-install audit diff must catch it and roll the
        whole transaction back.
        """
        from axiom_encode import manifest_audit as ma

        state: dict[str, object] = {}

        def setup(checkout: Path, content_root: Path) -> None:
            target = content_root / REL
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(OLD_RULE)
            sibling = content_root / "regulations/18-nycrr/387/12/f/3/v/d.yaml"
            sibling.write_text("format: rulespec/v1\nrules: [d]\n")
            root = checkout / (
                ".axiom/encoding-manifests/regulations/18-nycrr/387/12/f/3/v/c.json"
            )
            root.parent.mkdir(parents=True)
            root.write_text(
                json.dumps(
                    {
                        "applied_files": [
                            {"path": REL, "sha256": _sha(OLD_RULE)},
                            {
                                "path": "regulations/18-nycrr/387/12/f/3/v/d.yaml",
                                "sha256": _sha256_file(sibling),
                            },
                        ]
                    }
                )
                + "\n"
            )
            state.update(checkout=checkout, target=target, root=root)

        empty = ma.PrunePlan(retire={}, disclose=(), drop_keys=())
        with patch("axiom_encode.manifest_audit.plan_prune", return_value=empty):
            with pytest.raises(RuntimeError) as info:
                self._run(tmp_path, setup=setup)
        assert "would add" in str(info.value) and "mismatch" in str(info.value)
        checkout = state["checkout"]
        assert state["target"].read_text() == OLD_RULE
        assert not (checkout / ".axiom/.apply-transaction").exists()
        assert not (
            checkout
            / ".axiom/encoding-manifests/us-ny/regulations/18-nycrr/387/12/f/3/v/c.json"
        ).exists()


# Subclassing the harness re-collects its ~400 tests under this class; mask
# every inherited test so only the end-to-end prune cases here are collected.
for _name in dir(_CmdEncodeHarness):
    if _name.startswith("test_") and _name not in TestApplyPruneEndToEnd.__dict__:
        setattr(TestApplyPruneEndToEnd, _name, None)
