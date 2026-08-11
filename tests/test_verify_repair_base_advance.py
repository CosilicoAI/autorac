from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.verify_repair_base_advance import verify_base_advance


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True
    ).strip()


def _commit(repository: Path, message: str) -> str:
    subprocess.run(["git", "-C", str(repository), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", message], check=True)
    return _git(repository, "rev-parse", "HEAD")


def _repository(
    tmp_path: Path,
    *,
    candidate_path: str = "statutes/42/1437c-1.yaml",
    jurisdiction: str = "us",
) -> tuple[Path, str]:
    repository = tmp_path / "rulespec-us"
    repository.mkdir()
    subprocess.run(["git", "-C", str(repository), "init", "-q"], check=True)
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test")
    rulespec_path = Path(jurisdiction) / candidate_path
    payloads = {
        rulespec_path: "format: rulespec/v1\n",
        rulespec_path.with_suffix(".test.yaml"): "tests: []\n",
        Path(".axiom/encoding-manifests") / rulespec_path.with_suffix(".json"): "{}\n",
    }
    for relative_path, content in payloads.items():
        destination = repository / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    return repository, _commit(repository, "source")


def test_accepts_ancestor_base_when_repair_target_identity_is_unchanged(
    tmp_path: Path,
) -> None:
    repository, source_ref = _repository(tmp_path)
    (repository / "unrelated.txt").write_text("advance\n", encoding="utf-8")
    current_ref = _commit(repository, "unrelated advance")

    verify_base_advance(
        repository,
        country="us",
        source_ref=source_ref,
        current_ref=current_ref,
        candidate_path="statutes/42/1437c-1.yaml",
    )


def test_accepts_state_jurisdiction_repository_path(tmp_path: Path) -> None:
    candidate_path = "statutes/47/297/4.yaml"
    repository, source_ref = _repository(
        tmp_path,
        candidate_path=candidate_path,
        jurisdiction="us-la",
    )
    (repository / "unrelated.txt").write_text("advance\n", encoding="utf-8")
    current_ref = _commit(repository, "unrelated advance")

    verify_base_advance(
        repository,
        country="us",
        source_ref=source_ref,
        current_ref=current_ref,
        candidate_path=candidate_path,
        rulespec_path="us-la/statutes/47/297/4.yaml",
    )


def test_rejects_mismatched_state_candidate_repository_path(
    tmp_path: Path,
) -> None:
    candidate_path = "statutes/47/297/4.yaml"
    repository, source_ref = _repository(
        tmp_path,
        candidate_path=candidate_path,
        jurisdiction="us-la",
    )

    with pytest.raises(ValueError, match="repository RuleSpec path"):
        verify_base_advance(
            repository,
            country="us",
            source_ref=source_ref,
            current_ref=source_ref,
            candidate_path=candidate_path,
            rulespec_path="us-la/statutes/47/297/8.yaml",
        )


@pytest.mark.parametrize(
    "rulespec_path",
    [
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
    ],
)
def test_rejects_noncanonical_state_repository_path(
    tmp_path: Path,
    rulespec_path: str,
) -> None:
    candidate_path = "statutes/47/297/4.yaml"
    repository, source_ref = _repository(
        tmp_path,
        candidate_path=candidate_path,
        jurisdiction="us-la",
    )

    with pytest.raises(ValueError, match="repository RuleSpec path"):
        verify_base_advance(
            repository,
            country="us",
            source_ref=source_ref,
            current_ref=source_ref,
            candidate_path=candidate_path,
            rulespec_path=rulespec_path,
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "us/statutes/42/1437c-1.yaml",
        "us/statutes/42/1437c-1.test.yaml",
        ".axiom/encoding-manifests/us/statutes/42/1437c-1.json",
    ],
)
def test_rejects_target_identity_missing_at_source_base(
    tmp_path: Path,
    relative_path: str,
) -> None:
    repository, _ = _repository(tmp_path)
    subprocess.run(
        ["git", "-C", str(repository), "rm", "-q", "--", relative_path],
        check=True,
    )
    source_ref = _commit(repository, "source without identity path")

    with pytest.raises(ValueError, match="missing at its source RuleSpec base"):
        verify_base_advance(
            repository,
            country="us",
            source_ref=source_ref,
            current_ref=source_ref,
            candidate_path="statutes/42/1437c-1.yaml",
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "us/statutes/42/1437c-1.yaml",
        "us/statutes/42/1437c-1.test.yaml",
        ".axiom/encoding-manifests/us/statutes/42/1437c-1.json",
    ],
)
def test_rejects_advance_that_changes_repair_target_identity(
    tmp_path: Path,
    relative_path: str,
) -> None:
    repository, source_ref = _repository(tmp_path)
    (repository / relative_path).write_text("changed\n", encoding="utf-8")
    current_ref = _commit(repository, "target changed")

    with pytest.raises(ValueError, match="target identity changed"):
        verify_base_advance(
            repository,
            country="us",
            source_ref=source_ref,
            current_ref=current_ref,
            candidate_path="statutes/42/1437c-1.yaml",
        )


def test_rejects_dotted_section_manifest_change_at_real_manifest_path(
    tmp_path: Path,
) -> None:
    candidate_path = "regulations/10-ccr/4.100.yaml"
    repository, source_ref = _repository(
        tmp_path,
        candidate_path=candidate_path,
    )
    manifest = repository / ".axiom/encoding-manifests/us/regulations/10-ccr/4.100.json"
    manifest.write_text("changed\n", encoding="utf-8")
    current_ref = _commit(repository, "dotted manifest changed")

    with pytest.raises(ValueError, match="target identity changed"):
        verify_base_advance(
            repository,
            country="us",
            source_ref=source_ref,
            current_ref=current_ref,
            candidate_path=candidate_path,
        )


def test_accepts_rerooted_base_when_repair_target_identity_is_unchanged(
    tmp_path: Path,
) -> None:
    repository, source_ref = _repository(tmp_path)
    (repository / "source-only.txt").write_text("source\n", encoding="utf-8")
    source_ref = _commit(repository, "source branch")
    subprocess.run(
        ["git", "-C", str(repository), "switch", "-qc", "other", f"{source_ref}^"],
        check=True,
    )
    (repository / "other.txt").write_text("other\n", encoding="utf-8")
    current_ref = _commit(repository, "other history")

    verify_base_advance(
        repository,
        country="us",
        source_ref=source_ref,
        current_ref=current_ref,
        candidate_path="statutes/42/1437c-1.yaml",
    )


def test_accepts_rerooted_base_with_exact_canonical_manifest_relocation(
    tmp_path: Path,
) -> None:
    repository, source_ref = _repository(tmp_path)
    (repository / "source-only.txt").write_text("source\n", encoding="utf-8")
    source_ref = _commit(repository, "source branch")
    subprocess.run(
        ["git", "-C", str(repository), "switch", "-qc", "other", f"{source_ref}^"],
        check=True,
    )
    legacy_manifest = Path(".axiom/encoding-manifests/us/statutes/42/1437c-1.json")
    canonical_manifest = Path(".axiom/encoding-manifests/statutes/42/1437c-1.json")
    (repository / canonical_manifest).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "mv",
            legacy_manifest.as_posix(),
            canonical_manifest.as_posix(),
        ],
        check=True,
    )
    current_ref = _commit(repository, "canonical manifest relocation")

    verify_base_advance(
        repository,
        country="us",
        source_ref=source_ref,
        current_ref=current_ref,
        candidate_path="statutes/42/1437c-1.yaml",
    )


def test_rejects_ambiguous_legacy_and_canonical_manifests(tmp_path: Path) -> None:
    repository, source_ref = _repository(tmp_path)
    canonical_manifest = (
        repository / ".axiom/encoding-manifests/statutes/42/1437c-1.json"
    )
    canonical_manifest.parent.mkdir(parents=True, exist_ok=True)
    canonical_manifest.write_text("{}\n", encoding="utf-8")
    current_ref = _commit(repository, "duplicate canonical manifest")

    with pytest.raises(ValueError, match="ambiguous at its current RuleSpec base"):
        verify_base_advance(
            repository,
            country="us",
            source_ref=source_ref,
            current_ref=current_ref,
            candidate_path="statutes/42/1437c-1.yaml",
        )


def test_rejects_tree_instead_of_tracked_blob(tmp_path: Path) -> None:
    repository, _ = _repository(tmp_path)
    candidate = repository / "us/statutes/42/1437c-1.yaml"
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "rm",
            "-q",
            "--",
            candidate.relative_to(repository),
        ],
        check=True,
    )
    candidate.mkdir()
    (candidate / "child").write_text("not a rulespec blob\n", encoding="utf-8")
    source_ref = _commit(repository, "tree at candidate path")

    with pytest.raises(ValueError, match="not a tracked blob"):
        verify_base_advance(
            repository,
            country="us",
            source_ref=source_ref,
            current_ref=source_ref,
            candidate_path="statutes/42/1437c-1.yaml",
        )


def test_rejects_rerooted_base_when_repair_target_identity_changes(
    tmp_path: Path,
) -> None:
    repository, source_ref = _repository(tmp_path)
    (repository / "source-only.txt").write_text("source\n", encoding="utf-8")
    source_ref = _commit(repository, "source branch")
    subprocess.run(
        ["git", "-C", str(repository), "switch", "-qc", "other", f"{source_ref}^"],
        check=True,
    )
    (repository / "us/statutes/42/1437c-1.yaml").write_text(
        "changed\n", encoding="utf-8"
    )
    current_ref = _commit(repository, "changed target on other history")

    with pytest.raises(ValueError, match="target identity changed"):
        verify_base_advance(
            repository,
            country="us",
            source_ref=source_ref,
            current_ref=current_ref,
            candidate_path="statutes/42/1437c-1.yaml",
        )


def test_ignores_local_git_blob_replacement(tmp_path: Path) -> None:
    repository, source_ref = _repository(tmp_path)
    candidate = repository / "us/statutes/42/1437c-1.yaml"
    candidate.write_text("replacement content\n", encoding="utf-8")
    current_ref = _commit(repository, "changed candidate")
    source_blob = _git(
        repository,
        "--no-replace-objects",
        "rev-parse",
        f"{source_ref}:us/statutes/42/1437c-1.yaml",
    )
    current_blob = _git(
        repository,
        "--no-replace-objects",
        "rev-parse",
        f"{current_ref}:us/statutes/42/1437c-1.yaml",
    )
    subprocess.run(
        ["git", "-C", str(repository), "replace", source_blob, current_blob],
        check=True,
    )

    with pytest.raises(ValueError, match="target identity changed"):
        verify_base_advance(
            repository,
            country="us",
            source_ref=source_ref,
            current_ref=current_ref,
            candidate_path="statutes/42/1437c-1.yaml",
        )


def test_rejects_current_ref_that_is_not_checkout_head(tmp_path: Path) -> None:
    repository, source_ref = _repository(tmp_path)
    (repository / "unrelated.txt").write_text("advance\n", encoding="utf-8")
    _commit(repository, "advance")

    with pytest.raises(ValueError, match="not the checkout HEAD"):
        verify_base_advance(
            repository,
            country="us",
            source_ref=source_ref,
            current_ref=source_ref,
            candidate_path="statutes/42/1437c-1.yaml",
        )


@pytest.mark.parametrize(
    ("country", "candidate_path"),
    [
        ("../us", "statutes/42/1437c-1.yaml"),
        ("us", "../statutes/42/1437c-1.yaml"),
        ("us", "/statutes/42/1437c-1.yaml"),
        ("us", "statutes//42/1437c-1.yaml"),
        ("us", "statutes/./42/1437c-1.yaml"),
        ("us", "statutes/42/1437c-1.test.yaml"),
    ],
)
def test_rejects_noncanonical_identity_paths(
    tmp_path: Path,
    country: str,
    candidate_path: str,
) -> None:
    repository, source_ref = _repository(tmp_path)

    with pytest.raises(ValueError, match="path is not canonical"):
        verify_base_advance(
            repository,
            country=country,
            source_ref=source_ref,
            current_ref=source_ref,
            candidate_path=candidate_path,
        )
