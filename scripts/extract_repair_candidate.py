#!/usr/bin/env python3
"""Verify and extract one preserved targeted-reencode repair candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import tarfile
from pathlib import Path, PurePosixPath

SCHEMA = "axiom-encode/failed-reencode-diagnostics/v1"
FAILED_CANDIDATE_SCHEMA = "axiom-encode/failed-encode-candidate/v1"
FAILED_CANDIDATE_KEYS = {
    "attempt_count",
    "citation",
    "encoder_version",
    "issues",
    "path",
    "rulespec_sha256",
    "schema",
    "tests_sha256",
}
ATOMIC_ROOTS = frozenset(
    {"guidance", "manuals", "policies", "programs", "regulations", "statutes"}
)
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 8192
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
RUNNER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
JURISDICTION_PATTERN = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]+)*")
VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,127}")
MAX_RETAINED_ISSUES = 4096
CONTRACT = runpy.run_path(
    Path(__file__).parents[1] / "src/axiom_encode/repair_candidate_contract.py"
)
BACKFILL_CONTRACT = runpy.run_path(
    Path(__file__).with_name("prepare_signed_backfill.py")
)
SPLIT_ATOMIC_SOURCE_INPUT = BACKFILL_CONTRACT["split_atomic_source_input"]
MAX_CANDIDATE_BYTES = CONTRACT["VALIDATION_RETRY_CANDIDATE_MAX_FILE_BYTES"]
SINGLE_TARGET_MODE_FIELDS = {
    "dependent_citation": None,
    "existing_signed_imports_input": "[]",
    "legacy_exact_dependent_rulespec_path": None,
    "legacy_retained_successor_rulespec_paths_input": "[]",
    "queue_dispatcher_run_id": None,
    "queue_id": None,
    "queue_item_generation_sha256": None,
    "queue_item_id": None,
    "queue_manifest_sha256": None,
    "replace_legacy_rulespec_path": None,
    "second_dependent_citation": None,
    "second_legacy_exact_dependent_rulespec_path": None,
}


def _normalized_member_name(raw_name: str) -> str:
    name = raw_name.removeprefix("./")
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("repair artifact contains an unsafe member path")
    return path.as_posix()


def _member_index(bundle: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    index: dict[str, tarfile.TarInfo] = {}
    for member_number, member in enumerate(bundle, start=1):
        if member_number > MAX_ARCHIVE_MEMBERS:
            raise ValueError("repair artifact exceeds its member limit")
        name = _normalized_member_name(member.name)
        if name in index:
            raise ValueError("repair artifact contains a duplicate member path")
        index[name] = member
    return index


def _regular_member(
    members: dict[str, tarfile.TarInfo], expected: str
) -> tarfile.TarInfo:
    member = members.get(expected)
    if member is None or not member.isfile():
        raise ValueError(f"repair artifact must contain one regular {expected}")
    return member


def _read_member(
    bundle: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    max_bytes: int,
) -> bytes:
    if member.size < 0 or member.size > max_bytes:
        raise ValueError(
            f"repair artifact member exceeds its size limit: {member.name}"
        )
    source = bundle.extractfile(member)
    if source is None:
        raise ValueError(f"repair artifact member is unreadable: {member.name}")
    data = source.read(max_bytes + 1)
    if len(data) != member.size or len(data) > max_bytes:
        raise ValueError(
            f"repair artifact member size changed while reading: {member.name}"
        )
    return data


def _metadata_file_map(metadata: dict[str, object]) -> dict[str, dict[str, object]]:
    raw_files = metadata.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("repair metadata files must be an array")
    files: dict[str, dict[str, object]] = {}
    for raw_entry in raw_files:
        if not isinstance(raw_entry, dict):
            raise ValueError("repair metadata file entries must be objects")
        raw_path = raw_entry.get("path")
        digest = raw_entry.get("sha256")
        size = raw_entry.get("size")
        if (
            not isinstance(raw_path, str)
            or _normalized_member_name(raw_path) != raw_path
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or raw_path in files
        ):
            raise ValueError("repair metadata contains an invalid file entry")
        files[raw_path] = raw_entry
    return files


def _verified_generated_file(
    bundle: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    files: dict[str, dict[str, object]],
    relative_path: str,
) -> bytes:
    entry = files.get(relative_path)
    if entry is None:
        raise ValueError(f"repair metadata does not bind {relative_path}")
    member = _regular_member(members, f"generated/{relative_path}")
    data = _read_member(bundle, member, max_bytes=MAX_CANDIDATE_BYTES)
    if (
        len(data) != entry["size"]
        or hashlib.sha256(data).hexdigest() != entry["sha256"]
    ):
        raise ValueError(f"repair candidate digest mismatch: {relative_path}")
    return data


def _retained_candidate(
    bundle: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    files: dict[str, dict[str, object]],
    *,
    repair_lane: str,
    expected_module: str,
    citation: str,
) -> tuple[bytes, bytes] | None:
    root = f"{repair_lane}/final-rejected-candidate"
    issues_path = f"{root}/issues.json"
    candidate_path = f"{root}/{expected_module}"
    tests_path = f"{root}/{expected_module.removesuffix('.yaml')}.test.yaml"
    retained_paths = {path for path in files if path.startswith(f"{root}/")}
    expected_paths = {issues_path, candidate_path, tests_path}
    if retained_paths and retained_paths != expected_paths:
        raise ValueError("retained repair candidate must bind exactly three files")
    if issues_path not in files:
        return None
    try:
        metadata = json.loads(
            _verified_generated_file(bundle, members, files, issues_path).decode(
                "utf-8", errors="strict"
            )
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("retained repair candidate metadata is invalid") from exc
    if (
        not isinstance(metadata, dict)
        or set(metadata) != FAILED_CANDIDATE_KEYS
        or metadata.get("schema") != FAILED_CANDIDATE_SCHEMA
        or metadata.get("citation") != citation
        or metadata.get("path") != expected_module
        or not isinstance(metadata.get("encoder_version"), str)
        or VERSION_PATTERN.fullmatch(metadata["encoder_version"]) is None
        or not isinstance(metadata.get("attempt_count"), int)
        or isinstance(metadata["attempt_count"], bool)
        or metadata["attempt_count"] < 1
        or not isinstance(metadata.get("issues"), list)
        or not metadata["issues"]
        or len(metadata["issues"]) > MAX_RETAINED_ISSUES
        or any(not isinstance(issue, str) or not issue for issue in metadata["issues"])
        or not isinstance(metadata.get("rulespec_sha256"), str)
        or SHA256_PATTERN.fullmatch(metadata["rulespec_sha256"]) is None
        or not isinstance(metadata.get("tests_sha256"), str)
        or SHA256_PATTERN.fullmatch(metadata["tests_sha256"]) is None
    ):
        raise ValueError("retained repair candidate metadata is invalid")
    candidate = _verified_generated_file(bundle, members, files, candidate_path)
    tests = _verified_generated_file(
        bundle,
        members,
        files,
        tests_path,
    )
    if (
        hashlib.sha256(candidate).hexdigest() != metadata["rulespec_sha256"]
        or hashlib.sha256(tests).hexdigest() != metadata["tests_sha256"]
    ):
        raise ValueError("retained repair candidate digest mismatch")
    return candidate, tests


def _expected_module_path(country: str, replace_rulespec_path: str) -> str:
    path = PurePosixPath(replace_rulespec_path)
    jurisdiction = path.parts[0] if path.parts else ""
    if (
        path.is_absolute()
        or path.as_posix() != replace_rulespec_path
        or len(path.parts) < 3
        or JURISDICTION_PATTERN.fullmatch(jurisdiction) is None
        or (jurisdiction != country and not jurisdiction.startswith(f"{country}-"))
        or path.parts[1] not in ATOMIC_ROOTS
        or path.suffix != ".yaml"
        or path.name.endswith(".test.yaml")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("replace RuleSpec path is not canonical for the country")
    return PurePosixPath(*path.parts[1:]).as_posix()


def _repair_lane_for_atomic_source(
    metadata: dict[str, object], expected_atomic_source: object
) -> tuple[str, list[str]]:
    try:
        expected = SPLIT_ATOMIC_SOURCE_INPUT(expected_atomic_source)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("expected atomic source input is invalid") from exc
    if expected["canonical_refresh_bundle"]:
        raise ValueError("repair replay does not support canonical refresh bundles")
    has_atomic = "atomic_source_input" in metadata
    has_legacy_source = "source_bundle_input" in metadata
    has_legacy_refresh = "canonical_refresh_bundle_input" in metadata
    if has_atomic:
        if has_legacy_source or has_legacy_refresh:
            raise ValueError(
                "repair artifact is not a compatible single-target run: "
                "atomic_source_input"
            )
        raw_atomic = metadata["atomic_source_input"]
        try:
            split = SPLIT_ATOMIC_SOURCE_INPUT(raw_atomic)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                "repair artifact is not a compatible single-target run: "
                "atomic_source_input"
            ) from exc
        if split != expected:
            raise ValueError("repair artifact metadata mismatch: atomic_source_input")
        if not expected["source_bundle"]:
            return "target", ["target"]
        generated_lanes = metadata.get("generated_lanes")
        if generated_lanes == ["target-preflight"]:
            return "target-preflight", ["target-preflight"]
        source_lanes = [
            f"source-{index:02d}"
            for index in range(1, len(expected["source_bundle"]) + 1)
        ]
        final_lanes = sorted(["target", "target-preflight", *source_lanes])
        replayed_final_lanes = sorted(["target", *source_lanes])
        if generated_lanes in (final_lanes, replayed_final_lanes):
            return "target", generated_lanes
        raise ValueError(
            "repair artifact generated lanes do not bind a target preflight "
            "or final composed target"
        )
    if expected != {
        "canonical_refresh_bundle": [],
        "primary_required_test_cases": [],
        "source_bundle": [],
    }:
        raise ValueError(
            "legacy repair metadata cannot bind a nonempty atomic source input"
        )
    if not has_legacy_source or metadata["source_bundle_input"] != "[]":
        raise ValueError(
            "repair artifact is not a compatible single-target run: source_bundle_input"
        )
    if has_legacy_refresh and metadata["canonical_refresh_bundle_input"] != "[]":
        raise ValueError(
            "repair artifact is not a compatible single-target run: "
            "canonical_refresh_bundle_input"
        )
    return "target", ["target"]


def _write_candidate_pair(
    root: Path,
    relative_path: str,
    candidate: bytes,
    tests: bytes,
) -> None:
    output = root / relative_path
    test_output = output.with_suffix(".test.yaml")
    output.parent.mkdir(parents=True, exist_ok=False)
    for path, data in ((output, candidate), (test_output, tests)):
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as target:
                descriptor = -1
                target.write(data)
                target.flush()
                os.fsync(target.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def _source_repair_candidates(
    bundle: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    files: dict[str, dict[str, object]],
    *,
    destination: Path,
    country: str,
    source_citations: list[str],
    source_rulespec_paths_json: str | None,
) -> list[dict[str, str]]:
    if source_rulespec_paths_json is None:
        return []
    try:
        source_rulespec_paths = json.loads(source_rulespec_paths_json)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("expected source RuleSpec paths are invalid") from exc
    if (
        not isinstance(source_rulespec_paths, list)
        or len(source_rulespec_paths) != len(source_citations)
        or any(not isinstance(path, str) for path in source_rulespec_paths)
    ):
        raise ValueError("expected source RuleSpec paths are invalid")

    extracted: list[dict[str, str]] = []
    for index, (citation, rulespec_path) in enumerate(
        zip(source_citations, source_rulespec_paths, strict=True), start=1
    ):
        lane = f"source-{index:02d}"
        expected_module = _expected_module_path(country, rulespec_path)
        candidate_pattern = re.compile(
            rf"{re.escape(lane)}/({RUNNER_PATTERN.pattern})/"
            rf"{re.escape(expected_module)}"
        )
        runners = sorted(
            {
                match.group(1)
                for path in files
                if (match := candidate_pattern.fullmatch(path)) is not None
                and f"{lane}/{match.group(1)}/"
                f"{expected_module.removesuffix('.yaml')}.test.yaml"
                in files
            }
        )
        if "openai-gpt-5.6-sol" in runners:
            runner = "openai-gpt-5.6-sol"
        elif len(runners) == 1:
            runner = runners[0]
        else:
            raise ValueError(
                f"repair artifact does not bind one final source candidate: {lane}"
            )
        candidate_path = f"{lane}/{runner}/{expected_module}"
        tests_path = (
            f"{lane}/{runner}/{expected_module.removesuffix('.yaml')}.test.yaml"
        )
        candidate = _verified_generated_file(bundle, members, files, candidate_path)
        tests = _verified_generated_file(bundle, members, files, tests_path)
        root = destination / "source-candidates" / lane / runner
        _write_candidate_pair(root, expected_module, candidate, tests)
        extracted.append(
            {
                "citation": citation,
                "lane": lane,
                "path": expected_module,
                "root": str(root),
                "runner": runner,
                "rulespec_sha256": hashlib.sha256(candidate).hexdigest(),
                "tests_sha256": hashlib.sha256(tests).hexdigest(),
            }
        )
    return extracted


def extract_candidate(args: argparse.Namespace) -> dict[str, object]:
    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    expected_fields = {
        "citation": args.citation,
        "country": args.country,
        "encoder_commit": args.encoder_commit,
        "corpus_ref": args.corpus_ref,
        "rules_engine_ref": args.rules_engine_ref,
        "replace_rulespec_path": args.replace_rulespec_path,
        "workflow_run_id": args.workflow_run_id,
    }
    expected_module = _expected_module_path(
        args.country,
        args.replace_rulespec_path,
    )

    with tarfile.open(args.archive, mode="r:") as bundle:
        members = _member_index(bundle)
        metadata_member = _regular_member(members, "metadata.json")
        metadata = json.loads(
            _read_member(
                bundle,
                metadata_member,
                max_bytes=MAX_METADATA_BYTES,
            ).decode("utf-8", errors="strict")
        )
        if not isinstance(metadata, dict) or metadata.get("schema") != SCHEMA:
            raise ValueError("repair artifact metadata schema is invalid")
        for field, expected in expected_fields.items():
            if field not in metadata or metadata[field] != expected:
                raise ValueError(f"repair artifact metadata mismatch: {field}")
        source_rulespec_ref = metadata.get("rulespec_ref")
        if (
            not isinstance(source_rulespec_ref, str)
            or COMMIT_PATTERN.fullmatch(source_rulespec_ref) is None
        ):
            raise ValueError("repair artifact metadata mismatch: rulespec_ref")
        if (
            source_rulespec_ref != args.rulespec_ref
            and not args.allow_rulespec_base_advance
        ):
            raise ValueError("repair artifact metadata mismatch: rulespec_ref")
        for field, expected in SINGLE_TARGET_MODE_FIELDS.items():
            if field not in metadata or metadata[field] != expected:
                raise ValueError(
                    f"repair artifact is not a compatible single-target run: {field}"
                )
        try:
            expected_atomic_source = SPLIT_ATOMIC_SOURCE_INPUT(args.atomic_source_json)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("expected atomic source input is invalid") from exc
        repair_lane, expected_generated_lanes = _repair_lane_for_atomic_source(
            metadata, args.atomic_source_json
        )
        if metadata.get("workflow_run_attempt") != 1:
            raise ValueError("repair artifact must come from workflow attempt 1")
        if metadata.get("failed_steps") != ["encode_apply"]:
            raise ValueError(
                "repair artifact is not an encode/apply validation failure"
            )
        if metadata.get("generated_lanes") != expected_generated_lanes:
            raise ValueError(
                f"repair artifact must contain only the {repair_lane} lane"
            )

        files = _metadata_file_map(metadata)
        repair_pattern = re.compile(
            rf"{re.escape(repair_lane)}/({RUNNER_PATTERN.pattern})/"
            rf"{re.escape(expected_module[:-5])}\.repair\.json"
        )
        repair_matches = [
            (path, repair_pattern.fullmatch(path))
            for path in files
            if repair_pattern.fullmatch(path) is not None
        ]
        if len(repair_matches) != 1:
            raise ValueError("repair artifact must bind one final repair manifest")
        repair_path, repair_match = repair_matches[0]
        assert repair_match is not None
        runner = repair_match.group(1)
        repair_payload = json.loads(
            _verified_generated_file(bundle, members, files, repair_path).decode(
                "utf-8", errors="strict"
            )
        )
        if (
            not isinstance(repair_payload, dict)
            or repair_payload.get("schema_version") != "axiom-encode/repair-manifest/v1"
            or repair_payload.get("citation") != args.citation
            or repair_payload.get("runner") != runner
        ):
            raise ValueError("final repair manifest identity is invalid")

        retained = _retained_candidate(
            bundle,
            members,
            files,
            repair_lane=repair_lane,
            expected_module=expected_module,
            citation=args.citation,
        )
        if retained is None:
            candidate_path = f"{repair_lane}/{runner}/{expected_module}"
            test_path = candidate_path.removesuffix(".yaml") + ".test.yaml"
            candidate = _verified_generated_file(bundle, members, files, candidate_path)
            tests = _verified_generated_file(bundle, members, files, test_path)
        else:
            candidate, tests = retained
            runner = "retained-best"

        source_rulespec_paths_json = getattr(args, "source_rulespec_paths_json", None)
        if source_rulespec_paths_json is not None and repair_lane != "target":
            raise ValueError(
                "source repair candidates require a final composed target artifact"
            )
        source_candidates = _source_repair_candidates(
            bundle,
            members,
            files,
            destination=destination,
            country=args.country,
            source_citations=expected_atomic_source["source_bundle"],
            source_rulespec_paths_json=source_rulespec_paths_json,
        )

    root = destination / runner
    _write_candidate_pair(root, expected_module, candidate, tests)

    return {
        "root": str(root),
        "path": expected_module,
        "rulespec_sha256": hashlib.sha256(candidate).hexdigest(),
        "tests_sha256": hashlib.sha256(tests).hexdigest(),
        "runner": runner,
        "source_rulespec_ref": source_rulespec_ref,
        "source_candidates": source_candidates,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--citation", required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--encoder-commit", required=True)
    parser.add_argument("--corpus-ref", required=True)
    parser.add_argument("--rules-engine-ref", required=True)
    parser.add_argument("--rulespec-ref", required=True)
    parser.add_argument("--allow-rulespec-base-advance", action="store_true")
    parser.add_argument("--atomic-source-json", required=True)
    parser.add_argument("--replace-rulespec-path", required=True)
    parser.add_argument("--source-rulespec-paths-json")
    parser.add_argument("--workflow-run-id", required=True)
    return parser


def main() -> int:
    print(json.dumps(extract_candidate(_parser().parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
