from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.prepare_signed_backfill import (
    REVIEWED_RULESPEC_PR_BASE_BRANCHES,
    REVIEWED_RULESPEC_REFS,
    branch_name,
    stage_authorized_changes,
    validate_country,
    validate_dependent_cascade,
    validate_queue_tracking,
    validate_rulespec_base,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    return repo


def _add_origin_main(repo: Path) -> str:
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "remote", "add", "origin", str(repo))
    _git(repo, "update-ref", "refs/remotes/origin/main", base)
    return base


def _write_signed_change(repo: Path) -> tuple[Path, Path]:
    rule = repo / "us/regulations/example.yaml"
    rule.parent.mkdir(parents=True)
    rule.write_text("rules: []\n", encoding="utf-8")
    manifest = repo / ".axiom/encoding-manifests/us/regulations/example.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "axiom-encode/applied-rulespec/v5",
                "applied_files": [{"path": "us/regulations/example.yaml"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return rule, manifest


def _write_module(
    repo: Path,
    relative: str,
    *,
    imports: tuple[str, ...] = (),
) -> Path:
    path = repo / "us" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "format: rulespec/v1\n"
        + (
            "imports:\n" + "".join(f"  - {value}\n" for value in imports)
            if imports
            else "imports: []\n"
        )
        + "rules: []\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("country", ["../x", "us/x", "${{ inputs.x }}", "us\n"])
def test_validate_country_rejects_adversarial_values(country: str) -> None:
    with pytest.raises(ValueError, match="two-letter lowercase"):
        validate_country(country)


def test_validate_queue_tracking_accepts_complete_or_empty_metadata() -> None:
    assert validate_queue_tracking("", "", "", "") == "ad-hoc"
    assert (
        validate_queue_tracking(
            "us-snap-or-ut-2026-07",
            "ut-0001",
            "a" * 64,
            "b" * 64,
        )
        == "tracked"
    )


@pytest.mark.parametrize(
    ("queue_id", "item_id", "digest", "generation", "message"),
    [
        ("queue", "", "a" * 64, "b" * 64, "supplied together"),
        ("queue/unsafe", "ut-0001", "a" * 64, "b" * 64, "queue_id"),
        ("queue", "ut/0001", "a" * 64, "b" * 64, "queue_item_id"),
        ("queue", "ut-0001", "abc", "b" * 64, "queue_manifest_sha256"),
        ("queue", "ut-0001", "a" * 64, "abc", "queue_item_generation"),
    ],
)
def test_validate_queue_tracking_rejects_incomplete_or_unsafe_metadata(
    queue_id: str,
    item_id: str,
    digest: str,
    generation: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_queue_tracking(queue_id, item_id, digest, generation)


def test_stage_authorized_changes_stages_only_manifest_and_applied_files(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    rule, manifest = _write_signed_change(repo)

    stage_authorized_changes(repo)

    assert _git(repo, "diff", "--cached", "--name-only").splitlines() == sorted(
        [str(manifest.relative_to(repo)), str(rule.relative_to(repo))]
    )


def test_stage_authorized_changes_rejects_unexpected_file(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write_signed_change(repo)
    (repo / "encoder-surprise.txt").write_text("must not publish\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside signed manifest authorization"):
        stage_authorized_changes(repo)

    assert _git(repo, "diff", "--cached", "--name-only") == ""


def test_stage_rejects_manifest_authorizing_non_rulespec_path(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _rule, manifest = _write_signed_change(repo)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["applied_files"] = [{"path": ".github/workflows/pwn.yml"}]
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    unexpected = repo / ".github/workflows/pwn.yml"
    unexpected.parent.mkdir(parents=True)
    unexpected.write_text("name: unexpected\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a canonical RuleSpec YAML path"):
        stage_authorized_changes(repo)

    assert _git(repo, "diff", "--cached", "--name-only") == ""


def test_rerun_attempt_uses_recoverable_distinct_branch() -> None:
    first = branch_name("us", "12345", "1")
    rerun = branch_name("us", "12345", "2")

    assert first == "axiom/signed-backfill-us-12345-1"
    assert rerun == "axiom/signed-backfill-us-12345-2"
    assert rerun != first


def test_validate_dependent_cascade_accepts_only_direct_dependent(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _write_module(repo, "regulations/42-cfr/435/555.yaml")
    dependent = _write_module(
        repo,
        "regulations/42-cfr/435/559.yaml",
        imports=("us:regulations/42-cfr/435/555#target_rule",),
    )

    assert validate_dependent_cascade(
        repo,
        "us/regulation/42/435/555",
        "us/regulation/42/435/559",
    ) == (dependent.relative_to(repo / "us"),)


def test_validate_dependent_cascade_uses_nondefault_replacement_target(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    replacement = _write_module(
        repo,
        "policies/income_tax/pilot_liability_pipeline.yaml",
    )
    dependent = _write_module(
        repo,
        "regulations/42-cfr/435/559.yaml",
        imports=("us:policies/income_tax/pilot_liability_pipeline#target_rule",),
    )
    _write_module(repo, "regulations/42-cfr/435/555.yaml")

    assert validate_dependent_cascade(
        repo,
        "us/regulation/42/435/555",
        "us/regulation/42/435/559",
        target_rulespec_path=replacement.relative_to(repo).as_posix(),
    ) == (dependent.relative_to(repo / "us"),)


@pytest.mark.parametrize(
    "replacement_path",
    [
        "../outside.yaml",
        "us-nc/policies/income_tax/pipeline.yaml",
        "us/policies/income_tax/pipeline.test.yaml",
    ],
)
def test_validate_dependent_cascade_rejects_invalid_replacement_target(
    tmp_path: Path,
    replacement_path: str,
) -> None:
    repo = _repo(tmp_path)
    _write_module(repo, "regulations/42-cfr/435/555.yaml")
    _write_module(
        repo,
        "regulations/42-cfr/435/559.yaml",
        imports=("us:regulations/42-cfr/435/555",),
    )

    with pytest.raises(ValueError, match="target RuleSpec path must be"):
        validate_dependent_cascade(
            repo,
            "us/regulation/42/435/555",
            "us/regulation/42/435/559",
            target_rulespec_path=replacement_path,
        )


def test_validate_dependent_cascade_rejects_symlinked_replacement_parent(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "target.yaml").write_text(
        "rules:\n  target_rule:\n    kind: constant\n    value: 1\n",
        encoding="utf-8",
    )
    linked_parent = repo / "us" / "policies" / "linked"
    linked_parent.parent.mkdir(parents=True)
    linked_parent.symlink_to(outside, target_is_directory=True)
    _write_module(
        repo,
        "regulations/42-cfr/435/559.yaml",
        imports=("us:policies/linked/target#target_rule",),
    )

    with pytest.raises(
        ValueError,
        match="target citation has no regular baseline RuleSpec module",
    ):
        validate_dependent_cascade(
            repo,
            "us/regulation/42/435/555",
            "us/regulation/42/435/559",
            target_rulespec_path="us/policies/linked/target.yaml",
        )


def test_validate_dependent_cascade_rejects_symlinked_jurisdiction_root(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    outside = tmp_path / "outside-us"
    _write_module(
        outside.parent / "rulespec-outside",
        "policies/income_tax/target.yaml",
    )
    outside_source = outside.parent / "rulespec-outside" / "us"
    (repo / "us").symlink_to(outside_source, target_is_directory=True)
    _write_module(
        outside.parent / "rulespec-outside",
        "regulations/42-cfr/435/559.yaml",
        imports=("us:policies/income_tax/target#target_rule",),
    )

    with pytest.raises(
        ValueError,
        match="target citation has no regular baseline RuleSpec module",
    ):
        validate_dependent_cascade(
            repo,
            "us/regulation/42/435/555",
            "us/regulation/42/435/559",
            target_rulespec_path="us/policies/income_tax/target.yaml",
        )


def test_validate_dependent_cascade_rejects_unrelated_dependent(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _write_module(repo, "regulations/42-cfr/435/555.yaml")
    _write_module(
        repo,
        "regulations/42-cfr/435/559.yaml",
        imports=("us:regulations/42-cfr/435/555",),
    )
    _write_module(repo, "regulations/42-cfr/435/561.yaml")

    with pytest.raises(ValueError, match="does not exactly match"):
        validate_dependent_cascade(
            repo,
            "us/regulation/42/435/555",
            "us/regulation/42/435/561",
        )


def test_validate_dependent_cascade_accepts_all_direct_dependents(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _write_module(repo, "regulations/42-cfr/435/555.yaml")
    dependents = []
    for section in ("559", "561"):
        dependents.append(
            _write_module(
                repo,
                f"regulations/42-cfr/435/{section}.yaml",
                imports=("regulations/42-cfr/435/555",),
            ).relative_to(repo / "us")
        )

    assert validate_dependent_cascade(
        repo,
        "us/regulation/42/435/555",
        "us/regulation/42/435/559",
        "us/regulation/42/435/561",
    ) == tuple(dependents)


def test_validate_dependent_cascade_rejects_incomplete_direct_dependents(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _write_module(repo, "regulations/42-cfr/435/555.yaml")
    for section in ("559", "561"):
        _write_module(
            repo,
            f"regulations/42-cfr/435/{section}.yaml",
            imports=("regulations/42-cfr/435/555",),
        )

    with pytest.raises(ValueError, match="does not exactly match"):
        validate_dependent_cascade(
            repo,
            "us/regulation/42/435/555",
            "us/regulation/42/435/559",
        )


def test_validate_rulespec_base_accepts_main_ancestor(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = _add_origin_main(repo)

    assert validate_rulespec_base(repo, "us", base, open_pr=True) == "main"


def test_validate_rulespec_base_rejects_non_main_pr_for_main_ancestor(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    base = _add_origin_main(repo)

    with pytest.raises(ValueError, match="must target main"):
        validate_rulespec_base(
            repo,
            "us",
            base,
            open_pr=True,
            pr_base_branch="hard-cut/canonical-layout-us",
        )


def test_validate_rulespec_base_rejects_stale_main_pr_base(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    stale_base = _add_origin_main(repo)
    (repo / "README.md").write_text("advanced\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "advance main")
    advanced_base = _git(repo, "rev-parse", "HEAD")
    _git(
        repo,
        "update-ref",
        "refs/remotes/origin/main",
        advanced_base,
    )
    _git(repo, "checkout", "--detach", stale_base)

    with pytest.raises(ValueError, match="exact pull request base branch tip"):
        validate_rulespec_base(repo, "us", stale_base, open_pr=True)


@pytest.mark.parametrize(
    ("country", "reviewed_ref"),
    [
        ("us", "b61918da93fe8a1a29b35b9330aef2085291a5d0"),
        ("ca", "f60f7a84c30e38c7d4961d70647eb0457e7d76c2"),
    ],
)
def test_validate_rulespec_base_accepts_exact_reviewed_head_artifact_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    country: str,
    reviewed_ref: str,
) -> None:
    repo = tmp_path / f"rulespec-{country}"
    assert REVIEWED_RULESPEC_REFS == frozenset(
        {
            ("us", "b61918da93fe8a1a29b35b9330aef2085291a5d0"),
            ("ca", "f60f7a84c30e38c7d4961d70647eb0457e7d76c2"),
        }
    )
    assert REVIEWED_RULESPEC_PR_BASE_BRANCHES == frozenset(
        {("us", "hard-cut/canonical-layout-us")}
    )
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill._git",
        lambda _repo, *_args: f"{reviewed_ref}\n".encode(),
    )
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1),
    )

    assert (
        validate_rulespec_base(repo, country, reviewed_ref, open_pr=False)
        == "reviewed-head-artifact"
    )

    with pytest.raises(ValueError, match="artifact-only"):
        validate_rulespec_base(repo, country, reviewed_ref, open_pr=True)


def test_validate_rulespec_base_accepts_exact_reviewed_protected_branch_tip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "rulespec-us"
    reviewed_ref = "b61918da93fe8a1a29b35b9330aef2085291a5d0"
    git_calls: list[tuple[str, ...]] = []

    def fake_git(_repo: Path, *args: str) -> bytes:
        git_calls.append(args)
        return f"{reviewed_ref}\n".encode()

    monkeypatch.setattr("scripts.prepare_signed_backfill._git", fake_git)
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1),
    )

    assert (
        validate_rulespec_base(
            repo,
            "us",
            reviewed_ref,
            open_pr=True,
            pr_base_branch="hard-cut/canonical-layout-us",
        )
        == "reviewed-head-pr"
    )
    assert (
        "rev-parse",
        "refs/remotes/origin/hard-cut/canonical-layout-us",
    ) in git_calls


def test_validate_rulespec_base_rejects_stale_reviewed_protected_branch_tip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "rulespec-us"
    reviewed_ref = "b61918da93fe8a1a29b35b9330aef2085291a5d0"

    def fake_git(_repo: Path, *args: str) -> bytes:
        if args[-1] == "refs/remotes/origin/hard-cut/canonical-layout-us":
            return f"{'f' * 40}\n".encode()
        return f"{reviewed_ref}\n".encode()

    monkeypatch.setattr("scripts.prepare_signed_backfill._git", fake_git)
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1),
    )

    with pytest.raises(ValueError, match="exact pull request base branch tip"):
        validate_rulespec_base(
            repo,
            "us",
            reviewed_ref,
            open_pr=True,
            pr_base_branch="hard-cut/canonical-layout-us",
        )


def test_validate_rulespec_base_rejects_unapproved_reviewed_pr_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "rulespec-us"
    reviewed_ref = "b61918da93fe8a1a29b35b9330aef2085291a5d0"
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill._git",
        lambda _repo, *_args: f"{reviewed_ref}\n".encode(),
    )
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1),
    )

    with pytest.raises(ValueError, match="artifact-only"):
        validate_rulespec_base(
            repo,
            "us",
            reviewed_ref,
            open_pr=True,
            pr_base_branch="migration/other",
        )


@pytest.mark.parametrize(
    "retired_ref",
    [
        "991f5375b92dffca57b08069093c24a463365cbc",
        "10f7a16ef4a40cf1e26d6273e1aff9ebb79d002f",
        "670e6d6642c70168a4ecfcd7ccfc47c3e7cf51c3",
        "08f7d595d7d4b9ed8565eb90b1e19308fd7aecdd",
        "eb43611e4fdc2d3bdd93f123a2ee5e0b97b2fed0",
    ],
)
def test_validate_rulespec_base_rejects_retired_reviewed_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retired_ref: str,
) -> None:
    repo = tmp_path / "rulespec-us"
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill._git",
        lambda _repo, *_args: f"{retired_ref}\n".encode(),
    )
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1),
    )

    with pytest.raises(ValueError, match="neither on main nor an approved"):
        validate_rulespec_base(repo, "us", retired_ref, open_pr=False)


def test_validate_rulespec_base_rejects_unreviewed_non_main_head(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _add_origin_main(repo)
    _git(repo, "commit", "--allow-empty", "-m", "unreviewed")
    head = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(ValueError, match="neither on main nor an approved"):
        validate_rulespec_base(repo, "us", head, open_pr=False)
