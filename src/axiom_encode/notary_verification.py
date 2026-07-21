"""Strict, non-mutating verification receipts for RuleSpec checkouts.

This module deliberately does not share the generated-apply overlay.  The
overlay is an installation surface and is allowed to repair generated output;
the notary profile verifies the bytes already present in a clean checkout.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from axiom_encode import __version__

from .constants import (
    RULESPEC_ATOMIC_MODULE_ROOTS,
    RULESPEC_FILE_SUFFIX,
    RULESPEC_TEST_FILE_SUFFIX,
)
from .harness.eval_evidence import scrub_attestation_signing_keys
from .harness.evals import (
    _deterministic_tree_identity,
    _git_checkout_execution_identity,
)
from .harness.policyengine_runtime import PolicyEngineRuntime
from .harness.validator_pipeline import (
    PipelineResult,
    ValidatorPipeline,
    resolve_axiom_rules_engine_binary,
)
from .repo_routing import (
    atomic_rulespec_module_paths,
    canonical_rulespec_repo_name,
    find_policy_repo_root,
    inspect_canonical_rulespec_checkout,
)
from .toolchain import (
    VALIDATION_WAIVER_SET_PATH,
    load_rulespec_local_corpus_release,
    verify_rulespec_validation_waiver_set,
)
from .validation_waivers import load_validation_waivers

NOTARY_RECEIPT_SCHEMA_ID = "axiom/notary-verification-receipt/v0"
NOTARY_RECEIPT_SCHEMA_STATUS = "PROVISIONAL"
NOTARY_PROFILE_ID = "axiom/notary-verifier/strict-v0"
MIN_POLICYENGINE_MATCH = 0.95

GateReproducibility = Literal["public", "restricted-pinned", "ci-attested"]
GateStatus = Literal["passed", "failed", "reduced"]

_PUBLIC_GATES = (
    "subject-clean",
    "corpus-release-binding",
    "compile",
    "proof-revalidation",
    "companion-tests",
    "grounding-contract",
    "layout-inspection",
    "waiver-set-verification",
    "policy-repo-nonmutation",
)
_RESTRICTED_PINNED_GATES = ("policyengine-oracle",)
_CI_ATTESTED_GATES = (
    "rulespec-reviewer",
    "formula-reviewer",
    "parameter-reviewer",
    "integration-reviewer",
)
_REVIEWER_GATES = (
    ("rulespec_reviewer", "rulespec-reviewer"),
    ("formula_reviewer", "formula-reviewer"),
    ("parameter_reviewer", "parameter-reviewer"),
    ("integration_reviewer", "integration-reviewer"),
)
_GIT_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class NotaryVerificationError(RuntimeError):
    """The strict verifier could not produce trustworthy evidence."""


@dataclass(frozen=True, slots=True)
class NotaryVerificationResult:
    """Receipt and diagnostic outcome from one strict verification run."""

    receipt: dict[str, Any]
    passed: bool
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CleanGitIdentity:
    root: Path
    commit: str
    tree: str


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return the canonical JSON encoding used by the provisional receipt."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_receipt_body_bytes(receipt: Mapping[str, Any]) -> bytes:
    """Canonicalize a receipt body without its detached-style self-hash."""

    body = copy.deepcopy(dict(receipt))
    body.pop("receipt_sha256", None)
    return canonical_json_bytes(body)


def receipt_body_sha256(receipt: Mapping[str, Any]) -> str:
    """Hash only the canonical receipt body, excluding ``receipt_sha256``."""

    return hashlib.sha256(canonical_receipt_body_bytes(receipt)).hexdigest()


def attach_receipt_sha256(body: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached-style, self-hashed receipt without mutating ``body``."""

    receipt = copy.deepcopy(dict(body))
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = receipt_body_sha256(receipt)
    return receipt


def canonical_receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    """Serialize a complete receipt in its sole accepted JSON representation."""

    expected = receipt_body_sha256(receipt)
    if receipt.get("receipt_sha256") != expected:
        raise NotaryVerificationError(
            "receipt_sha256 does not match the canonical receipt body"
        )
    return canonical_json_bytes(receipt)


def gate_reproducibility(gate: str) -> GateReproducibility:
    """Return the charter classification for one strict-profile gate."""

    if gate in _PUBLIC_GATES:
        return "public"
    if gate in _RESTRICTED_PINNED_GATES:
        return "restricted-pinned"
    if gate in _CI_ATTESTED_GATES:
        return "ci-attested"
    raise ValueError(f"Unknown notary verification gate: {gate}")


def _gate(gate: str, status: GateStatus) -> dict[str, str]:
    return {
        "gate": gate,
        "status": status,
        "reproducibility": gate_reproducibility(gate),
    }


def _resolve_existing_directory(raw_path: Path, *, label: str) -> Path:
    raw = Path(os.path.abspath(Path(raw_path).expanduser()))
    cursor = Path(raw.anchor)
    for part in raw.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise NotaryVerificationError(f"{label} path contains a symlink: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise NotaryVerificationError(f"{label} path does not exist: {raw}") from exc
    if not resolved.is_dir():
        raise NotaryVerificationError(f"{label} path is not a directory: {raw}")
    return resolved


def _resolve_policy_checkout(raw_path: Path) -> Path:
    root = _resolve_existing_directory(raw_path, label="RuleSpec checkout")
    inspection = inspect_canonical_rulespec_checkout(
        root,
        allow_composition_specs=True,
    )
    if inspection.name is None:
        raise NotaryVerificationError(
            "RuleSpec checkout must be the exact canonical rulespec-<country> "
            f"checkout ({inspection.rejection}): {root}"
        )
    return root


def _git_text(root: Path, *args: str) -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise NotaryVerificationError("Cannot inspect Git identity: git is unavailable")
    try:
        completed = subprocess.run(
            [git_executable, "-C", str(root), *args],
            capture_output=True,
            check=False,
            text=True,
            env=scrub_attestation_signing_keys(),
        )
    except OSError as exc:
        raise NotaryVerificationError(
            f"Cannot inspect {root.name} Git identity: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise NotaryVerificationError(
            f"Cannot inspect {root.name} Git identity: {detail or 'git failed'}"
        )
    return completed.stdout.strip()


def _require_clean_git_checkout(root: Path, *, label: str) -> _CleanGitIdentity:
    identity = _git_checkout_execution_identity(root)
    commit = identity.get("commit")
    identity_root = identity.get("path")
    if identity.get("kind") != "git" or identity_root != str(root):
        raise NotaryVerificationError(
            f"{label} must be the top level of a Git worktree: {root}"
        )
    if identity.get("dirty") is not False:
        raise NotaryVerificationError(
            f"{label} worktree is dirty; commit or remove every tracked and "
            f"untracked change before verification: {root}"
        )
    if not isinstance(commit, str) or _GIT_OID_RE.fullmatch(commit) is None:
        raise NotaryVerificationError(f"{label} HEAD is not a full Git commit SHA")
    tree = _git_text(root, "rev-parse", "HEAD^{tree}")
    if _GIT_OID_RE.fullmatch(tree) is None:
        raise NotaryVerificationError(f"{label} HEAD tree is not a full Git tree SHA")
    return _CleanGitIdentity(root=root, commit=commit, tree=tree)


def _policy_content_identity(root: Path) -> tuple[str, int]:
    identity = _deterministic_tree_identity(
        root,
        excluded_directory_names=frozenset({".git"}),
    )
    tree_sha256 = identity.get("tree_sha256")
    file_count = identity.get("file_count")
    if (
        identity.get("state") != "directory"
        or not isinstance(tree_sha256, str)
        or _SHA256_RE.fullmatch(tree_sha256) is None
        or not isinstance(file_count, int)
    ):
        raise NotaryVerificationError("Cannot hash the RuleSpec checkout contents")
    return tree_sha256, file_count


def _encoder_package_identity() -> dict[str, object]:
    identity = _deterministic_tree_identity(
        Path(__file__).resolve().parent,
        excluded_directory_names=frozenset({"__pycache__"}),
    )
    tree_sha256 = identity.get("tree_sha256")
    file_count = identity.get("file_count")
    if (
        identity.get("state") != "directory"
        or not isinstance(tree_sha256, str)
        or _SHA256_RE.fullmatch(tree_sha256) is None
        or not isinstance(file_count, int)
    ):
        raise NotaryVerificationError("Cannot hash the executing axiom-encode package")
    return {"tree_sha256": tree_sha256, "file_count": file_count}


def _sha256_file(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise NotaryVerificationError(f"{label} is not a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise NotaryVerificationError(f"Cannot hash {label}: {path}") from exc
    return {"sha256": digest.hexdigest(), "size": size}


def _axiom_rules_engine_execution_identity(root: Path) -> dict[str, object]:
    """Bind the executable selected by the existing validator search order."""

    try:
        candidate = resolve_axiom_rules_engine_binary(root)
        relative = candidate.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise NotaryVerificationError(str(exc)) from exc
    return {
        "path": relative.as_posix(),
        **_sha256_file(candidate, label="Axiom rules engine executable"),
    }


def _assert_same_clean_checkout(
    expected: _CleanGitIdentity,
    *,
    label: str,
) -> None:
    actual = _require_clean_git_checkout(expected.root, label=label)
    if actual.commit != expected.commit or actual.tree != expected.tree:
        raise NotaryVerificationError(
            f"{label} commit or tree changed during strict verification"
        )


def _assert_output_outside_policy_repo(receipt_out: Path, policy_root: Path) -> Path:
    raw = Path(os.path.abspath(Path(receipt_out).expanduser()))
    cursor = Path(raw.anchor)
    for part in raw.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise NotaryVerificationError(
                f"Receipt output path contains a symlink: {raw}"
            )
    resolved = raw.resolve(strict=False)
    if resolved == policy_root or policy_root in resolved.parents:
        raise NotaryVerificationError(
            "--receipt-out must be outside the verified policy repository"
        )
    if raw.exists() and not raw.is_file():
        raise NotaryVerificationError(f"Receipt output is not a regular file: {raw}")
    return raw


def _is_primary_rulespec_module(path: Path, *, checkout: Path) -> bool:
    content_root = find_policy_repo_root(path)
    if content_root is None or content_root.parent != checkout:
        return False
    try:
        relative = path.relative_to(content_root)
    except ValueError:
        return False
    return (
        path.suffix == RULESPEC_FILE_SUFFIX
        and not path.name.endswith(RULESPEC_TEST_FILE_SUFFIX)
        and len(relative.parts) >= 2
        and relative.parts[0] in RULESPEC_ATOMIC_MODULE_ROOTS
    )


def _resolve_changed_target(raw_target: Path, *, checkout: Path) -> Path:
    candidate = Path(raw_target)
    if not candidate.is_absolute():
        candidate = checkout / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        candidate.relative_to(checkout)
    except ValueError as exc:
        raise NotaryVerificationError(
            f"Changed target is outside the policy checkout: {raw_target}"
        ) from exc
    if candidate.name.endswith(RULESPEC_TEST_FILE_SUFFIX):
        candidate = candidate.with_name(
            candidate.name.removesuffix(RULESPEC_TEST_FILE_SUFFIX)
            + RULESPEC_FILE_SUFFIX
        )
    try:
        candidate = candidate.resolve(strict=True)
    except OSError as exc:
        raise NotaryVerificationError(
            f"Changed RuleSpec target does not exist: {raw_target}"
        ) from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise NotaryVerificationError(
            f"Changed RuleSpec target is not a regular file: {raw_target}"
        )
    if not _is_primary_rulespec_module(candidate, checkout=checkout):
        raise NotaryVerificationError(
            "Changed target must be a primary atomic RuleSpec module or its "
            f"companion test: {raw_target}"
        )
    return candidate


def _whole_repo_targets(checkout: Path) -> tuple[Path, ...]:
    try:
        modules = atomic_rulespec_module_paths(checkout)
    except ValueError as exc:
        raise NotaryVerificationError(str(exc)) from exc
    if not modules:
        raise NotaryVerificationError(
            "Whole-repository verification found no atomic RuleSpec modules"
        )
    return modules


def resolve_notary_targets(
    checkout: Path,
    *,
    changed_files: Sequence[Path] = (),
    whole_repo: bool = False,
) -> tuple[Path, ...]:
    """Resolve exactly one explicit changed-file or whole-repository target set."""

    if whole_repo == bool(changed_files):
        raise NotaryVerificationError(
            "Choose exactly one target set: --changed-files or --whole-repo"
        )
    if whole_repo:
        return _whole_repo_targets(checkout)
    return tuple(
        sorted(
            {_resolve_changed_target(path, checkout=checkout) for path in changed_files}
        )
    )


def _pipeline_issue_strings(
    relative_target: str,
    result: PipelineResult,
) -> list[str]:
    issues: list[str] = []
    for name, validation in sorted(result.results.items()):
        if validation.passed:
            continue
        details = list(validation.issues or ())
        if not details and validation.error:
            details = [validation.error]
        if not details:
            details = ["gate failed without a diagnostic"]
        issues.extend(f"{relative_target} [{name}]: {detail}" for detail in details)
    return issues


def _result_gate_passed(results: Sequence[PipelineResult], gate: str) -> bool:
    return bool(results) and all(
        gate in result.results and result.results[gate].passed for result in results
    )


def _deterministic_gate_passed(
    results: Sequence[PipelineResult],
    gate: str,
) -> bool:
    if not results:
        return False
    for result in results:
        ci_result = result.results.get("ci")
        if ci_result is None:
            return False
        outcomes = ci_result.details.get("deterministic_gates")
        if not isinstance(outcomes, dict) or outcomes.get(gate) is not True:
            return False
    return True


def _oracle_passed(results: Sequence[PipelineResult]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    for result in results:
        oracle = result.results.get("policyengine")
        if oracle is None:
            issues.append("PolicyEngine oracle result is missing")
            continue
        if not oracle.passed:
            issues.append("PolicyEngine oracle reported failure")
        if oracle.score is None or oracle.score < MIN_POLICYENGINE_MATCH:
            rendered = (
                "no comparable score" if oracle.score is None else f"{oracle.score:.1%}"
            )
            issues.append(
                "PolicyEngine oracle score "
                f"{rendered} is below {MIN_POLICYENGINE_MATCH:.0%}"
            )
        coverage = oracle.details.get("coverage")
        if isinstance(coverage, dict) and int(coverage.get("unmapped", 0) or 0):
            issues.append(
                "PolicyEngine oracle left "
                f"{int(coverage.get('unmapped', 0) or 0)} output(s) unclassified"
            )
    return not issues, issues


def _portable_policyengine_identity(
    runtime: PolicyEngineRuntime | None,
) -> dict[str, object] | None:
    if runtime is None:
        return None
    identity = runtime.canonical_identity()
    fields = (
        "schema",
        "country",
        "trusted_git_commit",
        "official_tree_sha256",
        "checkout_execution_tree_sha256",
        "venv_execution_tree_sha256",
        "pyproject_sha256",
        "uv_lock_sha256",
        "locked_versions",
    )
    portable = {field: identity[field] for field in fields if field in identity}
    portable["identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(portable)
    ).hexdigest()
    return portable


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    raw = canonical_receipt_bytes(receipt)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    except OSError as exc:
        raise NotaryVerificationError(f"Cannot write notary receipt: {path}") from exc


def _utc_timestamp(now: datetime | None = None) -> str:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise NotaryVerificationError("Notary receipt timestamp must be timezone-aware")
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def run_notary_verification(
    *,
    policy_repo_path: Path,
    corpus_path: Path,
    axiom_rules_engine_path: Path,
    receipt_out: Path,
    changed_files: Sequence[Path] = (),
    whole_repo: bool = False,
    policyengine_runtime_root: Path | None = None,
    rulespec_dependency_roots: Sequence[Path] = (),
    allow_reduced: bool = False,
    now: datetime | None = None,
) -> NotaryVerificationResult:
    """Run the strict profile and emit one unsigned, content-addressed receipt."""

    timestamp = _utc_timestamp(now)
    waiver_date = datetime.fromisoformat(timestamp.removesuffix("Z") + "+00:00").date()
    policy_root = _resolve_policy_checkout(policy_repo_path)
    policy_git = _require_clean_git_checkout(
        policy_root,
        label="RuleSpec checkout",
    )
    policy_content_before = _policy_content_identity(policy_root)
    output_path = _assert_output_outside_policy_repo(receipt_out, policy_root)
    corpus_root = _resolve_existing_directory(corpus_path, label="Axiom Corpus")
    engine_root = _resolve_existing_directory(
        axiom_rules_engine_path,
        label="Axiom rules engine",
    )
    engine_git = _require_clean_git_checkout(
        engine_root,
        label="Axiom rules engine",
    )
    engine_execution_before = _axiom_rules_engine_execution_identity(engine_root)

    dependency_roots = tuple(
        _resolve_policy_checkout(root) for root in rulespec_dependency_roots
    )
    if len(set(dependency_roots)) != len(dependency_roots):
        raise NotaryVerificationError(
            "--rulespec-dependency-root contains a duplicate checkout"
        )
    if policy_root in dependency_roots:
        raise NotaryVerificationError(
            "The verified policy checkout cannot also be a dependency root"
        )
    dependency_git = tuple(
        _require_clean_git_checkout(root, label="RuleSpec dependency checkout")
        for root in dependency_roots
    )

    try:
        checkout_modules = atomic_rulespec_module_paths(policy_root)
    except ValueError as exc:
        raise NotaryVerificationError(str(exc)) from exc
    if not checkout_modules:
        raise NotaryVerificationError(
            "Strict verification found no atomic RuleSpec modules"
        )
    targets = resolve_notary_targets(
        policy_root,
        changed_files=changed_files,
        whole_repo=whole_repo,
    )
    waiver_sha256 = verify_rulespec_validation_waiver_set(policy_root)
    waivers = load_validation_waivers(
        policy_root / VALIDATION_WAIVER_SET_PATH,
        repo_root=policy_root,
        today=waiver_date,
    )
    corpus_release = load_rulespec_local_corpus_release(policy_root, corpus_root)

    module_content_roots = {
        module: policy_root / module.relative_to(policy_root).parts[0]
        for module in checkout_modules
    }
    try:
        admitted_content_roots = tuple(
            dict.fromkeys(module_content_roots[target] for target in targets)
        )
    except KeyError as exc:
        raise NotaryVerificationError(
            "A target is absent from the strict RuleSpec layout scan"
        ) from exc

    policyengine_runtime: PolicyEngineRuntime | None = None
    oracle_reduced = policyengine_runtime_root is None
    if oracle_reduced and not allow_reduced:
        raise NotaryVerificationError(
            "PolicyEngine oracle dependency is absent; strict verification "
            "fails closed unless --allow-reduced is passed"
        )
    if policyengine_runtime_root is not None:
        runtime_root = _resolve_existing_directory(
            policyengine_runtime_root,
            label="PolicyEngine runtime",
        )
        policyengine_runtime = PolicyEngineRuntime.for_rulespec_root(
            runtime_root,
            policy_repo_root=admitted_content_roots[0],
        )
        for content_root in admitted_content_roots[1:]:
            policyengine_runtime.assert_matches_rulespec_root(content_root)
        policyengine_runtime.assert_unchanged()

    encoder_package_before = _encoder_package_identity()
    pipelines: dict[Path, ValidatorPipeline] = {}
    pipeline_results: list[PipelineResult] = []
    issues: list[str] = []
    for target in targets:
        content_root = module_content_roots[target]
        pipeline = pipelines.get(content_root)
        if pipeline is None:
            pipeline = ValidatorPipeline(
                policy_repo_path=content_root,
                axiom_rules_path=engine_root,
                enable_oracles=policyengine_runtime is not None,
                oracle_validators=("policyengine",)
                if policyengine_runtime is not None
                else (),
                policyengine_runtime=policyengine_runtime,
                require_policy_proofs=True,
                enforce_repository_layout=True,
                local_corpus_release=corpus_release,
                rulespec_dependency_roots=dependency_roots,
            )
            pipelines[content_root] = pipeline
        result = pipeline.validate(target, skip_reviewers=False)
        pipeline_results.append(result)
        relative_target = target.relative_to(policy_root).as_posix()
        issues.extend(_pipeline_issue_strings(relative_target, result))

    compile_passed = _result_gate_passed(pipeline_results, "compile")
    ci_passed = _result_gate_passed(pipeline_results, "ci")
    deterministic_statuses = {
        gate: _deterministic_gate_passed(pipeline_results, gate)
        for gate in (
            "proof-revalidation",
            "companion-tests",
            "grounding-contract",
            "layout-inspection",
        )
    }
    for gate, passed in deterministic_statuses.items():
        if not passed and ci_passed:
            issues.append(f"{gate}: deterministic gate did not complete successfully")
    reviewer_statuses = {
        receipt_gate: _result_gate_passed(pipeline_results, result_name)
        for result_name, receipt_gate in _REVIEWER_GATES
    }
    if policyengine_runtime is None:
        oracle_passed = True
        oracle_issues: list[str] = []
        oracle_status: GateStatus = "reduced"
    else:
        oracle_passed, oracle_issues = _oracle_passed(pipeline_results)
        issues.extend(f"policyengine-oracle: {issue}" for issue in oracle_issues)
        oracle_status = "passed" if oracle_passed else "failed"

    validators_passed = (
        compile_passed
        and ci_passed
        and all(deterministic_statuses.values())
        and all(reviewer_statuses.values())
        and oracle_passed
        and all(result.all_passed for result in pipeline_results)
    )

    _assert_same_clean_checkout(policy_git, label="RuleSpec checkout")
    policy_content_after = _policy_content_identity(policy_root)
    if policy_content_after != policy_content_before:
        raise NotaryVerificationError(
            "Strict verification mutated files inside the policy repository"
        )
    _assert_same_clean_checkout(engine_git, label="Axiom rules engine")
    if _axiom_rules_engine_execution_identity(engine_root) != engine_execution_before:
        raise NotaryVerificationError(
            "The Axiom rules engine executable changed during verification"
        )
    for identity in dependency_git:
        _assert_same_clean_checkout(identity, label="RuleSpec dependency checkout")
    if policyengine_runtime is not None:
        policyengine_runtime.assert_unchanged()
    if _encoder_package_identity() != encoder_package_before:
        raise NotaryVerificationError(
            "The executing axiom-encode package changed during verification"
        )

    gate_outcomes = [
        _gate("subject-clean", "passed"),
        _gate("corpus-release-binding", "passed"),
        _gate("compile", "passed" if compile_passed else "failed"),
        *[
            _gate(gate, "passed" if deterministic_statuses[gate] else "failed")
            for gate in (
                "proof-revalidation",
                "companion-tests",
                "grounding-contract",
                "layout-inspection",
            )
        ],
        _gate("waiver-set-verification", "passed"),
        _gate("policyengine-oracle", oracle_status),
        *[
            _gate(name, "passed" if reviewer_statuses[name] else "failed")
            for name in _CI_ATTESTED_GATES
        ],
        _gate("policy-repo-nonmutation", "passed"),
    ]

    status = "passed"
    if not validators_passed:
        status = "failed"
    elif oracle_reduced:
        status = "passed-reduced"

    relative_targets = [
        target.relative_to(policy_root).as_posix() for target in targets
    ]
    body: dict[str, Any] = {
        "schema_id": NOTARY_RECEIPT_SCHEMA_ID,
        "schema_status": NOTARY_RECEIPT_SCHEMA_STATUS,
        "status": status,
        "subject_tree": policy_git.tree,
        "subject_commit": policy_git.commit,
        "targets": {
            "mode": "whole-repo" if whole_repo else "changed-files",
            "files": relative_targets,
        },
        "dependencies": {
            "corpus_release": {
                "name": corpus_release.name,
                "content_sha256": corpus_release.content_sha256,
            },
            "axiom_rules_engine": {
                "commit": engine_git.commit,
                "executable": engine_execution_before,
            },
            "axiom_encode": {
                "package": "axiom-encode",
                "version": __version__,
                "package_identity": encoder_package_before,
            },
            "policyengine_oracle": _portable_policyengine_identity(
                policyengine_runtime
            ),
            "rulespec_dependencies": [
                {
                    "repository": canonical_rulespec_repo_name(identity.root),
                    "commit": identity.commit,
                    "tree": identity.tree,
                }
                for identity in dependency_git
            ],
        },
        "waiver_set": {
            "sha256": waiver_sha256,
            "count": len(waivers.active_paths),
        },
        "gates": gate_outcomes,
        "run": {
            "encoder_version": __version__,
            "profile_id": NOTARY_PROFILE_ID,
            "timestamp": timestamp,
        },
    }
    receipt = attach_receipt_sha256(body)
    _write_receipt(output_path, receipt)
    return NotaryVerificationResult(
        receipt=receipt,
        passed=validators_passed,
        issues=tuple(issues),
    )


__all__ = [
    "MIN_POLICYENGINE_MATCH",
    "NOTARY_PROFILE_ID",
    "NOTARY_RECEIPT_SCHEMA_ID",
    "NOTARY_RECEIPT_SCHEMA_STATUS",
    "NotaryVerificationError",
    "NotaryVerificationResult",
    "attach_receipt_sha256",
    "canonical_json_bytes",
    "canonical_receipt_body_bytes",
    "canonical_receipt_bytes",
    "gate_reproducibility",
    "receipt_body_sha256",
    "resolve_notary_targets",
    "run_notary_verification",
]
