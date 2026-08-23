from __future__ import annotations

import hashlib
import io
import json
import tarfile
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.extract_repair_candidate import (
    MAX_ARCHIVE_MEMBERS,
    MAX_CANDIDATE_BYTES,
    SINGLE_TARGET_MODE_FIELDS,
    _expected_module_path,
    extract_candidate,
)


def _archive(tmp_path: Path, *, candidate: bytes | None = None) -> tuple[Path, dict]:
    candidate = candidate or b"format: rulespec/v1\nrules: []\n"
    tests = b"[]\n"
    repair = json.dumps(
        {
            "schema_version": "axiom-encode/repair-manifest/v1",
            "citation": "us/statute/42/1437c\u20131",
            "runner": "openai-gpt-5.6-sol",
        }
    ).encode()
    payloads = {
        "target/openai-gpt-5.6-sol/statutes/42/1437c-1.yaml": candidate,
        "target/openai-gpt-5.6-sol/statutes/42/1437c-1.test.yaml": tests,
        "target/openai-gpt-5.6-sol/statutes/42/1437c-1.repair.json": repair,
    }
    metadata = {
        "schema": "axiom-encode/failed-reencode-diagnostics/v1",
        "citation": "us/statute/42/1437c\u20131",
        "country": "us",
        "encoder_commit": "a" * 40,
        "corpus_ref": "b" * 40,
        "rules_engine_ref": "c" * 40,
        "rulespec_ref": "d" * 40,
        "replace_rulespec_path": "us/statutes/42/1437c-1.yaml",
        "workflow_run_id": "1234",
        "workflow_run_attempt": 1,
        "failed_steps": ["encode_apply"],
        "generated_lanes": ["target"],
        **SINGLE_TARGET_MODE_FIELDS,
        "source_bundle_input": "[]",
        "files": [
            {
                "path": path,
                "size": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
            for path, body in sorted(payloads.items())
        ],
    }
    archive = tmp_path / "failure.tar"
    with tarfile.open(archive, "w") as bundle:
        members = {"metadata.json": json.dumps(metadata).encode()}
        members.update({f"generated/{path}": body for path, body in payloads.items()})
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            bundle.addfile(info, io.BytesIO(body))
    return archive, metadata


def _add_retained_candidate(
    source_archive: Path,
    destination: Path,
    metadata: dict,
    *,
    candidate: bytes,
    tests: bytes = b"[]\n",
) -> Path:
    module = "statutes/42/1437c-1.yaml"
    root = "target/final-rejected-candidate"
    issues = json.dumps(
        {
            "schema": "axiom-encode/failed-encode-candidate/v1",
            "citation": "us/statute/42/1437c\u20131",
            "path": module,
            "issues": ["best candidate still needs one repair"],
            "rulespec_sha256": hashlib.sha256(candidate).hexdigest(),
            "tests_sha256": hashlib.sha256(tests).hexdigest(),
            "encoder_version": "0.2.1707",
            "attempt_count": 4,
        }
    ).encode()
    payloads = {
        f"{root}/issues.json": issues,
        f"{root}/{module}": candidate,
        f"{root}/{module.removesuffix('.yaml')}.test.yaml": tests,
    }
    metadata["files"].extend(
        {
            "path": path,
            "size": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        for path, body in sorted(payloads.items())
    )
    metadata["files"].sort(key=lambda entry: entry["path"])
    with (
        tarfile.open(source_archive, "r") as source,
        tarfile.open(destination, "w") as target,
    ):
        for member in source.getmembers():
            extracted = source.extractfile(member)
            assert extracted is not None
            body = extracted.read()
            if member.name == "metadata.json":
                body = json.dumps(metadata).encode()
                member.size = len(body)
            target.addfile(member, io.BytesIO(body))
        for path, body in payloads.items():
            info = tarfile.TarInfo(f"generated/{path}")
            info.size = len(body)
            target.addfile(info, io.BytesIO(body))
    return destination


def _rewrite_metadata(
    source_archive: Path,
    destination: Path,
    metadata: dict,
) -> Path:
    with (
        tarfile.open(source_archive, "r") as source,
        tarfile.open(destination, "w") as target,
    ):
        for member in source.getmembers():
            extracted = source.extractfile(member)
            assert extracted is not None
            body = extracted.read()
            if member.name == "metadata.json":
                body = json.dumps(metadata).encode()
                member.size = len(body)
            target.addfile(member, io.BytesIO(body))
    return destination


def _args(tmp_path: Path, archive: Path, **overrides) -> Namespace:
    values = {
        "archive": archive,
        "destination": tmp_path / "extracted",
        "citation": "us/statute/42/1437c\u20131",
        "country": "us",
        "encoder_commit": "a" * 40,
        "corpus_ref": "b" * 40,
        "rules_engine_ref": "c" * 40,
        "rulespec_ref": "d" * 40,
        "allow_rulespec_base_advance": False,
        "atomic_source_json": "[]",
        "replace_rulespec_path": "us/statutes/42/1437c-1.yaml",
        "workflow_run_id": "1234",
    }
    values.update(overrides)
    return Namespace(**values)


def test_extracts_checksum_bound_final_candidate(tmp_path):
    archive, _ = _archive(tmp_path)

    result = extract_candidate(_args(tmp_path, archive))

    root = Path(result["root"])
    assert result["path"] == "statutes/42/1437c-1.yaml"
    assert result["source_rulespec_ref"] == "d" * 40
    assert (root / result["path"]).read_text() == "format: rulespec/v1\nrules: []\n"
    assert (root / "statutes/42/1437c-1.test.yaml").read_text() == "[]\n"


def test_prefers_integrity_bound_retained_best_candidate(tmp_path):
    archive, metadata = _archive(tmp_path, candidate=b"final regression\n")
    retained = b"strongest retained candidate\n"
    replacement = _add_retained_candidate(
        archive,
        tmp_path / "retained.tar",
        metadata,
        candidate=retained,
    )

    result = extract_candidate(_args(tmp_path, replacement))

    root = Path(result["root"])
    assert result["runner"] == "retained-best"
    assert (root / result["path"]).read_bytes() == retained


def test_rejects_tampered_retained_best_candidate(tmp_path):
    archive, metadata = _archive(tmp_path)
    replacement = _add_retained_candidate(
        archive,
        tmp_path / "retained.tar",
        metadata,
        candidate=b"strongest retained candidate\n",
    )
    candidate_entry = next(
        entry
        for entry in metadata["files"]
        if entry["path"].endswith("final-rejected-candidate/statutes/42/1437c-1.yaml")
    )
    candidate_entry["sha256"] = "0" * 64
    tampered = _rewrite_metadata(
        replacement,
        tmp_path / "tampered-retained.tar",
        metadata,
    )

    with pytest.raises(ValueError, match="digest mismatch"):
        extract_candidate(_args(tmp_path, tampered))


def test_state_rulespec_path_maps_to_lane_relative_candidate_path():
    assert (
        _expected_module_path("us", "us-la/statutes/47/297/4.yaml")
        == "statutes/47/297/4.yaml"
    )


@pytest.mark.parametrize(
    "replace_rulespec_path",
    (
        "ca-la/statutes/47/297/4.yaml",
        "us-/statutes/47/297/4.yaml",
        "us-_/statutes/47/297/4.yaml",
        "us-LA/statutes/47/297/4.yaml",
        "us-la_foo/statutes/47/297/4.yaml",
        "us-la//statutes/47/297/4.yaml",
        "us-la/./statutes/47/297/4.yaml",
        "us-la/other/47/297/4.yaml",
        "us-la/statutes/47/297/4.test.yaml",
        "us-la/statutes/../297/4.yaml",
        "/us-la/statutes/47/297/4.yaml",
    ),
)
def test_rejects_noncanonical_state_rulespec_path(replace_rulespec_path):
    with pytest.raises(ValueError, match="not canonical"):
        _expected_module_path("us", replace_rulespec_path)


@pytest.mark.parametrize(
    "atomic_source_input",
    [
        "[]",
        '{"canonical_refresh_bundle":[]}',
        json.dumps(
            {
                "schema": "axiom-encode/atomic-source-transaction/v2",
                "source_bundle": [],
                "canonical_refresh_bundle": [],
                "primary_required_test_cases": [],
            }
        ),
    ],
)
def test_extracts_new_atomic_source_metadata(tmp_path, atomic_source_input):
    archive, metadata = _archive(tmp_path)
    del metadata["source_bundle_input"]
    metadata["atomic_source_input"] = atomic_source_input
    replacement = _rewrite_metadata(
        archive,
        tmp_path / "new-atomic-source.tar",
        metadata,
    )

    result = extract_candidate(_args(tmp_path, replacement))

    assert result["source_rulespec_ref"] == "d" * 40


def _rewrite_as_source_preflight(
    source_archive: Path,
    destination: Path,
    metadata: dict,
    atomic_source_input: str,
) -> Path:
    metadata.pop("source_bundle_input")
    metadata["atomic_source_input"] = atomic_source_input
    metadata["generated_lanes"] = ["target-preflight"]
    metadata["files"] = [
        {**entry, "path": entry["path"].replace("target/", "target-preflight/", 1)}
        for entry in metadata["files"]
    ]
    with (
        tarfile.open(source_archive, "r") as source,
        tarfile.open(destination, "w") as target,
    ):
        for member in source.getmembers():
            extracted = source.extractfile(member)
            assert extracted is not None
            body = extracted.read()
            if member.name == "metadata.json":
                body = json.dumps(metadata).encode()
            name = member.name.replace(
                "generated/target/", "generated/target-preflight/", 1
            )
            info = tarfile.TarInfo(name)
            info.size = len(body)
            target.addfile(info, io.BytesIO(body))
    return destination


def test_extracts_exactly_bound_source_preflight_candidate(tmp_path):
    atomic_source_input = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v2",
            "source_bundle": ["us/statute/7/2015/f"],
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [],
        }
    )
    archive, metadata = _archive(tmp_path)
    replacement = _rewrite_as_source_preflight(
        archive, tmp_path / "source-preflight.tar", metadata, atomic_source_input
    )

    result = extract_candidate(
        _args(tmp_path, replacement, atomic_source_json=atomic_source_input)
    )

    assert result["runner"] == "openai-gpt-5.6-sol"


def test_rejects_source_preflight_candidate_for_different_bundle(tmp_path):
    atomic_source_input = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v2",
            "source_bundle": ["us/statute/7/2015/f"],
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [],
        }
    )
    archive, metadata = _archive(tmp_path)
    replacement = _rewrite_as_source_preflight(
        archive, tmp_path / "source-preflight.tar", metadata, atomic_source_input
    )

    with pytest.raises(ValueError, match="metadata mismatch: atomic_source_input"):
        extract_candidate(
            _args(
                tmp_path,
                replacement,
                atomic_source_json=json.dumps(
                    {
                        "schema": "axiom-encode/atomic-source-transaction/v2",
                        "source_bundle": ["us/guidance/different"],
                        "canonical_refresh_bundle": [],
                        "primary_required_test_cases": [],
                    }
                ),
            )
        )


def test_rejects_source_preflight_lane_without_expected_source(tmp_path):
    archive, metadata = _archive(tmp_path)
    replacement = _rewrite_as_source_preflight(
        archive,
        tmp_path / "source-preflight.tar",
        metadata,
        json.dumps(
            {
                "schema": "axiom-encode/atomic-source-transaction/v2",
                "source_bundle": [],
                "canonical_refresh_bundle": [],
                "primary_required_test_cases": [],
            }
        ),
    )

    with pytest.raises(ValueError, match="only the target lane"):
        extract_candidate(_args(tmp_path, replacement))


def test_rejects_metadata_identity_mismatch(tmp_path):
    archive, _ = _archive(tmp_path)

    with pytest.raises(ValueError, match="metadata mismatch: rulespec_ref"):
        extract_candidate(_args(tmp_path, archive, rulespec_ref="e" * 40))


def test_allows_explicit_rulespec_base_advance_for_later_workflow_proof(tmp_path):
    archive, _ = _archive(tmp_path)

    result = extract_candidate(
        _args(
            tmp_path,
            archive,
            rulespec_ref="e" * 40,
            allow_rulespec_base_advance=True,
        )
    )

    assert result["source_rulespec_ref"] == "d" * 40


def test_rejects_malformed_source_rulespec_ref_during_base_advance(tmp_path):
    archive, metadata = _archive(tmp_path)
    metadata["rulespec_ref"] = "not-a-commit"
    replacement = _rewrite_metadata(
        archive,
        tmp_path / "malformed-rulespec-ref.tar",
        metadata,
    )

    with pytest.raises(ValueError, match="metadata mismatch: rulespec_ref"):
        extract_candidate(
            _args(
                tmp_path,
                replacement,
                rulespec_ref="e" * 40,
                allow_rulespec_base_advance=True,
            )
        )


def test_rejects_candidate_digest_mismatch(tmp_path):
    archive, metadata = _archive(tmp_path)
    candidate_entry = next(
        item for item in metadata["files"] if item["path"].endswith("1437c-1.yaml")
    )
    candidate_entry["sha256"] = "0" * 64
    replacement = _rewrite_metadata(
        archive,
        tmp_path / "tampered.tar",
        metadata,
    )

    with pytest.raises(ValueError, match="digest mismatch"):
        extract_candidate(_args(tmp_path, replacement))


def test_rejects_duplicate_archive_member(tmp_path):
    archive, _ = _archive(tmp_path)
    replacement = tmp_path / "duplicate.tar"
    with tarfile.open(archive, "r") as source, tarfile.open(replacement, "w") as target:
        for member in source.getmembers():
            body = source.extractfile(member).read()
            target.addfile(member, io.BytesIO(body))
            if member.name == "metadata.json":
                target.addfile(member, io.BytesIO(body))

    with pytest.raises(ValueError, match="duplicate member path"):
        extract_candidate(_args(tmp_path, replacement))


def test_rejects_archive_member_flood(tmp_path):
    archive = tmp_path / "flood.tar"
    with tarfile.open(archive, "w") as bundle:
        for index in range(MAX_ARCHIVE_MEMBERS + 1):
            info = tarfile.TarInfo(f"entry-{index}.txt")
            bundle.addfile(info, io.BytesIO())

    with pytest.raises(ValueError, match="member limit"):
        extract_candidate(_args(tmp_path, archive))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dependent_citation", "us/statute/42/other"),
        ("existing_signed_imports_input", '["us/statutes/import.yaml"]'),
        ("legacy_exact_dependent_rulespec_path", "us/statutes/dependent.yaml"),
        ("legacy_retained_successor_rulespec_paths_input", '["old.yaml"]'),
        ("queue_dispatcher_run_id", "999"),
        ("queue_id", "queue"),
        ("queue_item_generation_sha256", "a" * 64),
        ("queue_item_id", "item"),
        ("queue_manifest_sha256", "b" * 64),
        ("replace_legacy_rulespec_path", "us/statutes/old.yaml"),
        ("second_dependent_citation", "us/statute/42/second"),
        (
            "second_legacy_exact_dependent_rulespec_path",
            "us/statutes/second.yaml",
        ),
    ],
)
def test_rejects_incompatible_prior_run_mode(tmp_path, field, value):
    archive, metadata = _archive(tmp_path)
    metadata[field] = value
    replacement = _rewrite_metadata(
        archive,
        tmp_path / "incompatible.tar",
        metadata,
    )

    with pytest.raises(ValueError, match=f"single-target run: {field}"):
        extract_candidate(_args(tmp_path, replacement))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "source_bundle_input",
            '["us/statute/42/source"]',
            "single-target run: source_bundle_input",
        ),
        (
            "canonical_refresh_bundle_input",
            '[{"citation":"x"}]',
            "single-target run: canonical_refresh_bundle_input",
        ),
        (
            "atomic_source_input",
            '["us/statute/42/source"]',
            "metadata mismatch: atomic_source_input",
        ),
        (
            "atomic_source_input",
            '{"canonical_refresh_bundle":[{"citation":"x"}]}',
            "metadata mismatch: atomic_source_input",
        ),
        (
            "atomic_source_input",
            json.dumps(
                {
                    "schema": "axiom-encode/atomic-source-transaction/v2",
                    "source_bundle": [],
                    "canonical_refresh_bundle": [],
                    "primary_required_test_cases": [
                        {
                            "name": "must not replay a review contract",
                            "period": {
                                "period_kind": "tax_year",
                                "start": "2025-01-01",
                                "end": "2025-12-31",
                            },
                            "input": {"us:test#input": True},
                            "required_output": {"us:test#output": 1},
                        }
                    ],
                }
            ),
            "metadata mismatch: atomic_source_input",
        ),
    ],
)
def test_rejects_unbound_nonempty_atomic_source_metadata(
    tmp_path, field, value, message
):
    archive, metadata = _archive(tmp_path)
    if field == "atomic_source_input":
        del metadata["source_bundle_input"]
    metadata[field] = value
    replacement = _rewrite_metadata(
        archive,
        tmp_path / "nonempty-atomic-source.tar",
        metadata,
    )

    with pytest.raises(ValueError, match=message):
        extract_candidate(_args(tmp_path, replacement))


def test_rejects_expected_canonical_refresh_bundle(tmp_path):
    archive, _ = _archive(tmp_path)

    with pytest.raises(ValueError, match="does not support canonical refresh"):
        extract_candidate(
            _args(
                tmp_path,
                archive,
                atomic_source_json=json.dumps(
                    {
                        "schema": "axiom-encode/atomic-source-transaction/v2",
                        "source_bundle": [],
                        "canonical_refresh_bundle": [{"citation": "x"}],
                        "primary_required_test_cases": [],
                    }
                ),
            )
        )


def test_rejects_expected_source_bundle_mixed_with_required_tests(tmp_path):
    archive, _ = _archive(tmp_path)

    with pytest.raises(ValueError, match="expected atomic source input is invalid"):
        extract_candidate(
            _args(
                tmp_path,
                archive,
                atomic_source_json=json.dumps(
                    {
                        "schema": "axiom-encode/atomic-source-transaction/v2",
                        "source_bundle": ["us/statute/7/2015/f"],
                        "canonical_refresh_bundle": [],
                        "primary_required_test_cases": [
                            {
                                "name": "control",
                                "period": {
                                    "period_kind": "tax_year",
                                    "start": "2025-01-01",
                                    "end": "2025-12-31",
                                },
                                "input": {"us:test#input": True},
                                "required_output": {"us:test#output": 1},
                            }
                        ],
                    }
                ),
            )
        )


def test_rejects_missing_legacy_and_atomic_source_metadata(tmp_path):
    archive, metadata = _archive(tmp_path)
    del metadata["source_bundle_input"]
    replacement = _rewrite_metadata(
        archive,
        tmp_path / "missing-source-mode.tar",
        metadata,
    )

    with pytest.raises(ValueError, match="single-target run: source_bundle_input"):
        extract_candidate(_args(tmp_path, replacement))


@pytest.mark.parametrize("field", sorted(SINGLE_TARGET_MODE_FIELDS))
def test_rejects_missing_prior_run_mode_field(tmp_path, field):
    archive, metadata = _archive(tmp_path)
    del metadata[field]
    replacement = _rewrite_metadata(
        archive,
        tmp_path / "missing-mode-field.tar",
        metadata,
    )

    with pytest.raises(ValueError, match=f"single-target run: {field}"):
        extract_candidate(_args(tmp_path, replacement))


def test_candidate_size_bound_is_shared_at_exact_limit(tmp_path):
    archive, _ = _archive(tmp_path, candidate=b"x" * MAX_CANDIDATE_BYTES)

    result = extract_candidate(_args(tmp_path, archive))

    assert Path(result["root"], result["path"]).stat().st_size == MAX_CANDIDATE_BYTES


def test_candidate_size_bound_rejects_limit_plus_one(tmp_path):
    archive, _ = _archive(tmp_path, candidate=b"x" * (MAX_CANDIDATE_BYTES + 1))

    with pytest.raises(ValueError, match="exceeds its size limit"):
        extract_candidate(_args(tmp_path, archive))
