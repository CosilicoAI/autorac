from __future__ import annotations

import hashlib
import io
import json
import tarfile
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.extract_repair_candidate import MAX_ARCHIVE_MEMBERS, extract_candidate


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
    assert (root / result["path"]).read_text() == "format: rulespec/v1\nrules: []\n"
    assert (root / "statutes/42/1437c-1.test.yaml").read_text() == "[]\n"


def test_rejects_metadata_identity_mismatch(tmp_path):
    archive, _ = _archive(tmp_path)

    with pytest.raises(ValueError, match="metadata mismatch: rulespec_ref"):
        extract_candidate(_args(tmp_path, archive, rulespec_ref="e" * 40))


def test_rejects_candidate_digest_mismatch(tmp_path):
    archive, metadata = _archive(tmp_path)
    candidate_entry = next(
        item for item in metadata["files"] if item["path"].endswith("1437c-1.yaml")
    )
    candidate_entry["sha256"] = "0" * 64
    replacement = tmp_path / "tampered.tar"
    with tarfile.open(archive, "r") as source, tarfile.open(replacement, "w") as target:
        for member in source.getmembers():
            body = source.extractfile(member).read()
            if member.name == "metadata.json":
                body = json.dumps(metadata).encode()
                member.size = len(body)
            target.addfile(member, io.BytesIO(body))

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
