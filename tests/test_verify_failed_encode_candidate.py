"""Tests for the signed-apply failed-candidate artifact boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_failed_encode_candidate import verify_candidate_directory


def _candidate_directory(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "failed-encode"
    rulespec = root / "de/statutes/estg/66.yaml"
    rulespec.parent.mkdir(parents=True)
    rulespec.write_text("format: rulespec/v1\nrules: []\n")
    tests = rulespec.with_suffix(".test.yaml")
    tests.write_text("[]\n")
    metadata: dict[str, object] = {
        "schema": "axiom-encode/failed-encode-candidate/v1",
        "citation": "de/statute/estg/66",
        "path": "de/statutes/estg/66.yaml",
        "issues": ["[complete-source-unit:structure] missing branch"],
        "rulespec_sha256": hashlib.sha256(rulespec.read_bytes()).hexdigest(),
        "tests_sha256": hashlib.sha256(tests.read_bytes()).hexdigest(),
        "encoder_version": "0.2.1692",
        "attempt_count": 4,
    }
    (root / "issues.json").write_text(json.dumps(metadata) + "\n")
    return root, metadata


def test_verifies_exact_checksum_bound_candidate_directory(tmp_path):
    root, metadata = _candidate_directory(tmp_path)

    result = verify_candidate_directory(root, citation="de/statute/estg/66")

    assert result == {
        "root": str(root),
        "path": metadata["path"],
        "rulespec_sha256": metadata["rulespec_sha256"],
        "tests_sha256": metadata["tests_sha256"],
    }


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("rulespec_sha256", "RuleSpec SHA-256 mismatch"),
        ("tests_sha256", "tests SHA-256 mismatch"),
    ],
)
def test_rejects_issues_json_sha_mismatch(tmp_path, field, message):
    root, metadata = _candidate_directory(tmp_path)
    metadata[field] = "0" * 64
    (root / "issues.json").write_text(json.dumps(metadata) + "\n")

    with pytest.raises(ValueError, match=message):
        verify_candidate_directory(root, citation="de/statute/estg/66")


def test_rejects_candidate_from_a_slug_collision_with_another_citation(tmp_path):
    root, _metadata = _candidate_directory(tmp_path)

    with pytest.raises(ValueError, match="citation mismatch"):
        verify_candidate_directory(root, citation="de/statute/estg-66")


def test_rejects_backslash_in_noncanonical_candidate_path(tmp_path):
    root, metadata = _candidate_directory(tmp_path)
    rulespec = root / str(metadata["path"])
    tests = rulespec.with_suffix(".test.yaml")
    noncanonical_rulespec = rulespec.with_name(r"66\branch.yaml")
    noncanonical_tests = noncanonical_rulespec.with_suffix(".test.yaml")
    rulespec.rename(noncanonical_rulespec)
    tests.rename(noncanonical_tests)
    metadata["path"] = noncanonical_rulespec.relative_to(root).as_posix()
    (root / "issues.json").write_text(json.dumps(metadata) + "\n")

    with pytest.raises(ValueError, match="not canonical"):
        verify_candidate_directory(root, citation="de/statute/estg/66")


@pytest.mark.parametrize(
    "extra_path",
    [
        Path(".github/workflows/injected.yml"),
        Path("scripts/injected.py"),
        Path("_axiom/private.txt"),
        Path("diagnostics.log"),
    ],
)
def test_rejects_every_path_outside_exact_three_file_allowlist(
    tmp_path, extra_path
):
    root, _metadata = _candidate_directory(tmp_path)
    extra = root / extra_path
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("must not cross artifact boundary\n")

    with pytest.raises(ValueError, match="exactly the rejected RuleSpec"):
        verify_candidate_directory(root, citation="de/statute/estg/66")


def test_rejects_symlinked_candidate_file(tmp_path):
    root, _metadata = _candidate_directory(tmp_path)
    tests = root / "de/statutes/estg/66.test.yaml"
    tests.unlink()
    external = tmp_path / "external.test.yaml"
    external.write_text("secret\n")
    tests.symlink_to(external)

    with pytest.raises(ValueError, match="symlink|regular file"):
        verify_candidate_directory(root, citation="de/statute/estg/66")
