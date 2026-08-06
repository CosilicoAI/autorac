"""Tests for RuleSpec repository identity helpers."""

import subprocess
from pathlib import Path

import pytest

from axiom_encode.cli import _rulespec_test_relation_request_name
from axiom_encode.concepts.jurisdiction import jurisdiction_prefix
from axiom_encode.constants import (
    RULESPEC_ATOMIC_MODULE_ROOTS,
    RULESPEC_COMPOSITION_SPEC_ROOT,
    RULESPEC_FILESYSTEM_ROOTS,
)
from axiom_encode.harness.dependency_stubs import UnsafeRulespecContextPath
from axiom_encode.harness.validator_pipeline import (
    ValidatorPipeline as _ValidatorPipeline,
)
from axiom_encode.harness.validator_pipeline import (
    _candidate_rulespec_repo_roots,
    _canonical_rulespec_compile_path,
)
from axiom_encode.repo_routing import (
    _rulespec_routing_cache_scope,
    candidate_jurisdiction_content_dirs,
    canonical_rulespec_repo_name,
    canonical_rulespec_root_identity,
    find_policy_repo_root,
    inspect_canonical_rulespec_checkout,
    is_composition_policy_repo_root,
    is_policy_repo_root,
    is_rulespec_source_checkout_root,
    jurisdiction_subdir_names,
)


class ValidatorPipeline(_ValidatorPipeline):
    """Test convenience wrapper for corpus-free routing helpers."""

    def __init__(self, *args, local_corpus_release=None, **kwargs):
        super().__init__(
            *args,
            local_corpus_release=local_corpus_release,
            **kwargs,
        )


def _init_checkout(path: Path, origin: str) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", origin],
        cwd=path,
        check=True,
        capture_output=True,
    )


def test_rulespec_filesystem_and_atomic_root_contracts_are_distinct():
    assert RULESPEC_ATOMIC_MODULE_ROOTS == frozenset(
        {"legislation", "policies", "regulations", "statutes"}
    )
    assert RULESPEC_COMPOSITION_SPEC_ROOT == "programs"
    assert RULESPEC_FILESYSTEM_ROOTS == (
        RULESPEC_ATOMIC_MODULE_ROOTS | {RULESPEC_COMPOSITION_SPEC_ROOT}
    )


def test_composition_checkout_identity_admits_only_top_level_programs(tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    (checkout / "programs/us/snap").mkdir(parents=True)

    assert is_policy_repo_root(checkout) is False
    assert is_composition_policy_repo_root(checkout) is True
    assert is_rulespec_source_checkout_root(checkout) is True
    assert inspect_canonical_rulespec_checkout(
        checkout, allow_composition_specs=True
    ) == ("rulespec-us", None)
    assert jurisdiction_subdir_names(checkout, allow_composition_specs=True) == set()

    (checkout / "statutes").mkdir()
    assert is_composition_policy_repo_root(checkout) is False
    assert is_rulespec_source_checkout_root(checkout) is True
    assert inspect_canonical_rulespec_checkout(
        checkout, allow_composition_specs=True
    ) == (None, "atomic-root-at-checkout")


def test_composition_checkout_identity_rejects_symlinked_programs(tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    external_programs = tmp_path / "external-programs"
    external_programs.mkdir()
    (checkout / "programs").symlink_to(external_programs, target_is_directory=True)

    assert is_composition_policy_repo_root(checkout) is False
    assert is_rulespec_source_checkout_root(checkout) is False


def test_composition_checkout_preserves_atomic_jurisdiction_identity(tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    (checkout / "programs/us/snap").mkdir(parents=True)
    module = checkout / "us-co/policies/snap/example.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\n", encoding="utf-8")

    assert canonical_rulespec_root_identity(checkout / "us-co") == ("rulespec-us/us-co")
    assert find_policy_repo_root(module) == checkout / "us-co"
    assert canonical_rulespec_repo_name(checkout) == "rulespec-us"
    assert canonical_rulespec_repo_name(module) == "rulespec-us-co"
    assert jurisdiction_subdir_names(checkout, allow_composition_specs=True) == {
        "us-co"
    }
    assert candidate_jurisdiction_content_dirs(checkout, "us-co") == [
        checkout / "us-co"
    ]


def test_canonical_rulespec_root_identity_is_stable_for_direct_content_root(tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us-co"
    policy_root.mkdir()

    assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us-co"
    assert canonical_rulespec_repo_name(policy_root) == "rulespec-us-co"
    assert jurisdiction_prefix(policy_root) == "us-co"


def test_canonical_rulespec_root_identity_rejects_flat_and_aliased_roots(tmp_path):
    flat = tmp_path / "rulespec-us-co"
    flat.mkdir()
    alias = tmp_path / "rulespec-us-clean.abcd"
    _init_checkout(alias, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    (alias / "us").mkdir()

    assert canonical_rulespec_root_identity(flat) is None
    assert canonical_rulespec_root_identity(alias / "us") is None
    assert canonical_rulespec_repo_name(alias) is None


def test_country_checkout_identity_requires_existing_directory(tmp_path):
    nonexistent = tmp_path / "rulespec-us"
    regular_file = tmp_path / "files" / "rulespec-us"
    regular_file.parent.mkdir()
    regular_file.write_text("not a checkout\n")

    assert is_policy_repo_root(nonexistent) is False
    assert canonical_rulespec_repo_name(nonexistent) is None
    assert is_policy_repo_root(regular_file) is False
    assert canonical_rulespec_repo_name(regular_file) is None


@pytest.mark.parametrize(
    "root_name",
    ["legislation", "policies", "regulations", "statutes"],
)
def test_country_checkout_rejects_flat_root_level_content(tmp_path, root_name):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    (checkout / "us").mkdir()
    (checkout / root_name).mkdir()

    assert is_policy_repo_root(checkout) is False
    assert canonical_rulespec_root_identity(checkout / "us") is None


@pytest.mark.parametrize(
    "jurisdiction",
    ["us-", "us--co", "us-co.backup", "us-_scratch"],
)
def test_canonical_identity_rejects_malformed_jurisdiction_names(
    tmp_path,
    jurisdiction,
):
    checkout = tmp_path / "rulespec-us"
    (checkout / "us").mkdir(parents=True)
    malformed = checkout / jurisdiction
    malformed.mkdir()

    assert canonical_rulespec_root_identity(malformed) is None
    assert jurisdiction not in jurisdiction_subdir_names(checkout)


def test_canonical_identity_rejects_symlinked_checkout_and_content_roots(tmp_path):
    real_checkout = tmp_path / "real" / "rulespec-us"
    real_content = real_checkout / "us-co"
    real_content.mkdir(parents=True)
    checkout_alias = tmp_path / "alias" / "rulespec-us"
    checkout_alias.parent.mkdir()
    checkout_alias.symlink_to(real_checkout, target_is_directory=True)
    content_alias = real_checkout / "us-ny"
    external_content = tmp_path / "external-us-ny"
    external_content.mkdir()
    content_alias.symlink_to(external_content, target_is_directory=True)

    assert canonical_rulespec_root_identity(checkout_alias / "us-co") is None
    assert candidate_jurisdiction_content_dirs(checkout_alias, "us-co") == []
    assert canonical_rulespec_root_identity(content_alias) is None
    assert find_policy_repo_root(content_alias / "statutes" / "1.yaml") is None


def test_axiom_workspace_inside_monorepo_is_not_canonical_checkout(tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    (checkout / "us").mkdir()
    (checkout / "us-dc").mkdir()
    axiom_workspace = checkout / "_axiom"
    (axiom_workspace / "rulespec-us" / "us").mkdir(parents=True)

    assert canonical_rulespec_repo_name(axiom_workspace) is None
    assert candidate_jurisdiction_content_dirs(axiom_workspace, "us") == []
    assert (
        canonical_rulespec_root_identity(axiom_workspace / "rulespec-us" / "us") is None
    )


def test_canonical_identity_observes_new_parent_git_boundary(tmp_path):
    checkout = tmp_path / "rulespec-us"
    policy_root = checkout / "us-co"
    policy_root.mkdir(parents=True)

    assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us-co"

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)

    assert canonical_rulespec_root_identity(policy_root) is None


def test_canonical_identity_observes_changed_git_origin(tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    policy_root.mkdir()

    assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us"

    subprocess.run(
        ["git", "remote", "set-url", "origin", "https://example.com/not-us.git"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )

    assert canonical_rulespec_root_identity(policy_root) is None
    assert inspect_canonical_rulespec_checkout(
        checkout, allow_composition_specs=True
    ) == (None, "git-origin-name-mismatch")


def test_routing_cache_reuses_git_identity_only_inside_bounded_scope(
    monkeypatch,
    tmp_path,
):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    policy_root.mkdir()
    original_run = subprocess.run
    git_calls = []

    def counting_run(command, *args, **kwargs):
        git_calls.append(tuple(command))
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr("axiom_encode.repo_routing.subprocess.run", counting_run)

    with _rulespec_routing_cache_scope():
        assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us"
        assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us"

    assert len(git_calls) == 6

    assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us"
    assert len(git_calls) == 8


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, 10),
        ("2", 10),
        ("invalid", 10),
        ("17", 17),
    ],
)
def test_git_identity_probes_use_safe_configurable_timeout(
    monkeypatch,
    tmp_path,
    configured,
    expected,
):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    policy_root.mkdir()
    original_run = subprocess.run
    observed_timeouts = []

    if configured is None:
        monkeypatch.delenv(
            "AXIOM_ENCODE_GIT_PROBE_TIMEOUT_SECONDS",
            raising=False,
        )
    else:
        monkeypatch.setenv(
            "AXIOM_ENCODE_GIT_PROBE_TIMEOUT_SECONDS",
            configured,
        )

    def recording_run(command, *args, **kwargs):
        observed_timeouts.append(kwargs.get("timeout"))
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr("axiom_encode.repo_routing.subprocess.run", recording_run)

    with _rulespec_routing_cache_scope():
        assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us"

    assert len(observed_timeouts) == 6
    assert set(observed_timeouts) == {expected}


def test_routing_cache_invalidates_changed_git_origin_inside_scope(tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    policy_root.mkdir()

    with _rulespec_routing_cache_scope():
        assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us"
        subprocess.run(
            [
                "git",
                "remote",
                "set-url",
                "origin",
                "https://example.com/not-us.git",
            ],
            cwd=checkout,
            check=True,
            capture_output=True,
        )
        assert canonical_rulespec_root_identity(policy_root) is None


def test_routing_cache_retries_origin_mutated_during_admission(
    monkeypatch,
    tmp_path,
):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    policy_root.mkdir()
    from axiom_encode import repo_routing

    original_origin_name = repo_routing._git_origin_repo_name
    mutated = False

    def mutate_after_origin_read(root: str):
        nonlocal mutated
        origin_name = original_origin_name(root)
        if not mutated:
            mutated = True
            subprocess.run(
                [
                    "git",
                    "remote",
                    "set-url",
                    "origin",
                    "https://example.com/not-us.git",
                ],
                cwd=checkout,
                check=True,
                capture_output=True,
            )
        return origin_name

    monkeypatch.setattr(repo_routing, "_git_origin_repo_name", mutate_after_origin_read)

    with _rulespec_routing_cache_scope():
        assert canonical_rulespec_root_identity(policy_root) is None
        assert inspect_canonical_rulespec_checkout(
            checkout, allow_composition_specs=True
        ) == (None, "git-origin-name-mismatch")


def test_routing_cache_tracks_missing_conditional_git_include(tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    policy_root.mkdir()
    include_path = tmp_path / "future.conf"
    subprocess.run(
        [
            "git",
            "config",
            "--local",
            "includeIf.gitdir:**/rulespec-us/.git.path",
            str(include_path),
        ],
        cwd=checkout,
        check=True,
        capture_output=True,
    )

    with _rulespec_routing_cache_scope():
        assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us"
        include_path.write_text(
            '[url "https://example.com/not-us.git"]\n'
            "\tinsteadOf = https://github.com/TheAxiomFoundation/rulespec-us.git\n",
            encoding="utf-8",
        )
        assert (
            subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=checkout,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            == "https://example.com/not-us.git"
        )
        assert canonical_rulespec_root_identity(policy_root) is None
        assert inspect_canonical_rulespec_checkout(
            checkout, allow_composition_specs=True
        ) == (None, "git-origin-name-mismatch")


def test_routing_cache_tracks_missing_system_git_config(monkeypatch, tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    policy_root.mkdir()
    system_config = tmp_path / "future-system.conf"
    original_run = subprocess.run

    def system_config_run(command, *args, **kwargs):
        if command[-2:] == ["var", "GIT_CONFIG_SYSTEM"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"{system_config}\n",
                stderr="",
            )
        if command[-3:] == ["remote", "get-url", "origin"] and system_config.exists():
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="https://example.com/not-us.git\n",
                stderr="",
            )
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr("axiom_encode.repo_routing.subprocess.run", system_config_run)

    with _rulespec_routing_cache_scope():
        assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us"
        system_config.write_text(
            '[url "https://example.com/not-us.git"]\n'
            "\tinsteadOf = https://github.com/TheAxiomFoundation/rulespec-us.git\n",
            encoding="utf-8",
        )
        assert canonical_rulespec_root_identity(policy_root) is None
        assert inspect_canonical_rulespec_checkout(
            checkout, allow_composition_specs=True
        ) == (None, "git-origin-name-mismatch")


def test_routing_cache_retries_config_inputs_mutated_during_discovery(
    monkeypatch,
    tmp_path,
):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    policy_root.mkdir()
    include_path = tmp_path / "future.conf"
    from axiom_encode import repo_routing

    original_config_inputs = repo_routing._git_config_input_paths
    discoveries = 0

    def add_include_after_discovery(root: str):
        nonlocal discoveries
        config_inputs = original_config_inputs(root)
        discoveries += 1
        if discoveries == 1:
            subprocess.run(
                [
                    "git",
                    "config",
                    "--local",
                    "include.path",
                    str(include_path),
                ],
                cwd=checkout,
                check=True,
                capture_output=True,
            )
        return config_inputs

    monkeypatch.setattr(
        repo_routing,
        "_git_config_input_paths",
        add_include_after_discovery,
    )

    with _rulespec_routing_cache_scope():
        assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us"
        assert discoveries == 4
        include_path.write_text(
            '[url "https://example.com/not-us.git"]\n'
            "\tinsteadOf = https://github.com/TheAxiomFoundation/rulespec-us.git\n",
            encoding="utf-8",
        )
        assert canonical_rulespec_root_identity(policy_root) is None


def test_routing_cache_disables_custom_global_config_path_switch(
    monkeypatch,
    tmp_path,
):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    policy_root.mkdir()
    initial_global = tmp_path / "initial-global.conf"
    replacement_global = tmp_path / "replacement-global.conf"
    initial_global.write_text("", encoding="utf-8")
    replacement_global.write_text(
        '[url "https://example.com/not-us.git"]\n'
        "\tinsteadOf = https://github.com/TheAxiomFoundation/rulespec-us.git\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(initial_global))

    with _rulespec_routing_cache_scope() as cache:
        assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us"
        assert cache.checkout_inspections == {}
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(replacement_global))
        assert canonical_rulespec_root_identity(policy_root) is None


@pytest.mark.parametrize(
    ("config_key", "initial_value", "replacement_value"),
    [
        (
            "remote.origin.url",
            "https://github.com/TheAxiomFoundation/rulespec-us.git",
            "https://example.com/not-us.git",
        ),
        (
            "url.https://example.com/not-us.git.insteadOf",
            "https://unused.example/",
            "https://github.com/TheAxiomFoundation/rulespec-us.git",
        ),
    ],
)
def test_routing_cache_disables_mutable_command_scoped_git_config(
    monkeypatch,
    tmp_path,
    config_key,
    initial_value,
    replacement_value,
):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    policy_root.mkdir()
    if config_key == "remote.origin.url":
        subprocess.run(
            ["git", "config", "--local", "--unset-all", "remote.origin.url"],
            cwd=checkout,
            check=True,
            capture_output=True,
        )
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", config_key)
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", initial_value)

    with _rulespec_routing_cache_scope() as cache:
        assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us"
        assert cache.checkout_inspections == {}
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", replacement_value)
        assert canonical_rulespec_root_identity(policy_root) is None


def test_routing_cache_disables_command_scoped_git_include(monkeypatch, tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    policy_root.mkdir()
    include_path = tmp_path / "command-include.conf"
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "include.path")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(include_path))

    with _rulespec_routing_cache_scope() as cache:
        assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us"
        assert cache.checkout_inspections == {}
        include_path.write_text(
            '[url "https://example.com/not-us.git"]\n'
            "\tinsteadOf = https://github.com/TheAxiomFoundation/rulespec-us.git\n",
            encoding="utf-8",
        )
        assert canonical_rulespec_root_identity(policy_root) is None


def test_routing_cache_disables_symlinked_git_config(tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    policy_root.mkdir()
    config_path = checkout / ".git" / "config"
    config_target = tmp_path / "git-config-target"
    config_path.replace(config_target)
    config_path.symlink_to(config_target)

    with _rulespec_routing_cache_scope():
        assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us"
        subprocess.run(
            [
                "git",
                "remote",
                "set-url",
                "origin",
                "https://example.com/not-us.git",
            ],
            cwd=checkout,
            check=True,
            capture_output=True,
        )
        assert canonical_rulespec_root_identity(policy_root) is None


def test_routing_cache_invalidates_linked_worktree_config_change(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init"], cwd=source, check=True, capture_output=True)
    (source / "tracked").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked"], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Axiom Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "initial",
        ],
        cwd=source,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "remote",
            "add",
            "origin",
            "https://github.com/TheAxiomFoundation/rulespec-us.git",
        ],
        cwd=source,
        check=True,
    )
    checkout = tmp_path / "rulespec-us"
    subprocess.run(
        ["git", "worktree", "add", "-b", "test-worktree", str(checkout)],
        cwd=source,
        check=True,
        capture_output=True,
    )
    policy_root = checkout / "us"
    policy_root.mkdir()

    assert (checkout / ".git").is_file()
    with _rulespec_routing_cache_scope():
        assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us"
        subprocess.run(
            [
                "git",
                "remote",
                "set-url",
                "origin",
                "https://example.com/not-us.git",
            ],
            cwd=checkout,
            check=True,
            capture_output=True,
        )
        assert canonical_rulespec_root_identity(policy_root) is None


def test_routing_cache_retries_rejected_checkout_identity(tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://example.com/not-us.git")
    policy_root = checkout / "us"
    policy_root.mkdir()

    with _rulespec_routing_cache_scope():
        assert canonical_rulespec_root_identity(policy_root) is None
        subprocess.run(
            [
                "git",
                "remote",
                "set-url",
                "origin",
                "https://github.com/TheAxiomFoundation/rulespec-us.git",
            ],
            cwd=checkout,
            check=True,
            capture_output=True,
        )
        assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us"


def test_canonical_identity_rejects_observed_git_boundary_when_git_unavailable(
    monkeypatch,
    tmp_path,
):
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(["git", "init"], cwd=outer, check=True, capture_output=True)
    nested_root = outer / "nested" / "rulespec-us" / "us-co"
    nested_root.mkdir(parents=True)
    plain_root = tmp_path / "plain" / "rulespec-us" / "us-co"
    plain_root.mkdir(parents=True)

    def unavailable_git(*_args, **_kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr("axiom_encode.repo_routing.subprocess.run", unavailable_git)

    assert canonical_rulespec_root_identity(nested_root) is None
    assert canonical_rulespec_root_identity(plain_root) == "rulespec-us/us-co"


@pytest.mark.parametrize(
    ("error", "rejection"),
    [
        (FileNotFoundError(2, "git unavailable"), "git-top-level-probe-oserror-2"),
        (
            subprocess.TimeoutExpired(["git", "rev-parse"], 10),
            "git-top-level-probe-timeout",
        ),
    ],
)
def test_checkout_inspection_categorizes_git_probe_launch_failures(
    monkeypatch,
    tmp_path,
    error,
    rejection,
):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")

    def failed_git(*_args, **_kwargs):
        raise error

    monkeypatch.setattr("axiom_encode.repo_routing.subprocess.run", failed_git)

    assert inspect_canonical_rulespec_checkout(
        checkout, allow_composition_specs=True
    ) == (None, rejection)


def test_candidate_roots_reject_noncanonical_checkout_alias(tmp_path):
    checkout = tmp_path / "rulespec-us-clean.abcd"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")

    with pytest.raises(
        UnsafeRulespecContextPath,
        match="exact direct jurisdiction child",
    ):
        _candidate_rulespec_repo_roots("rulespec-us", checkout)


def test_compile_roots_reject_temp_checkout_alias(tmp_path):
    checkout = tmp_path / "rulespec-us-clean.abcd"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")

    pipeline = ValidatorPipeline(
        policy_repo_path=checkout,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )
    with pytest.raises(
        UnsafeRulespecContextPath,
        match="exact direct jurisdiction child",
    ):
        pipeline._rulespec_compile_roots()


def test_canonical_compile_path_rejects_temp_checkout_alias(tmp_path):
    checkout = tmp_path / "rulespec-us-clean.abcd"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    rules_file = checkout / "statutes" / "26" / "63" / "f.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("format: rulespec/v1\nrules: []\n")

    with pytest.raises(
        UnsafeRulespecContextPath,
        match="exact direct jurisdiction child",
    ):
        _canonical_rulespec_compile_path(rules_file, checkout)


def test_canonical_compile_path_uses_direct_monorepo_jurisdiction_path(
    tmp_path,
):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us-co"
    rules_file = policy_root / "regulations" / "10-ccr-2506-1" / "4.407.31.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("format: rulespec/v1\nrules: []\n")

    compile_path = _canonical_rulespec_compile_path(rules_file, policy_root)

    assert compile_path == rules_file.resolve()
    assert compile_path.parts[-5:] == (
        "rulespec-us",
        "us-co",
        "regulations",
        "10-ccr-2506-1",
        "4.407.31.yaml",
    )
    assert compile_path.resolve() == rules_file.resolve()


def test_canonical_compile_path_keeps_canonical_country_content_root_segment(
    tmp_path,
):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    rules_file = policy_root / "statutes" / "26" / "3121" / "i.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("format: rulespec/v1\nrules: []\n")

    compile_path = _canonical_rulespec_compile_path(rules_file, policy_root)

    assert compile_path == rules_file
    assert compile_path.parts[-5:] == (
        "us",
        "statutes",
        "26",
        "3121",
        "i.yaml",
    )


def test_compile_roots_expose_only_explicit_monorepo_checkout(tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us-co"
    policy_root.mkdir()

    pipeline = ValidatorPipeline(
        policy_repo_path=policy_root,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )
    assert pipeline._rulespec_compile_roots() == (checkout.resolve(),)
    assert pipeline._rulespec_root_args() == [
        "--rulespec-root",
        str(checkout.resolve()),
    ]


def test_dataset_preserves_canonical_same_repo_refs(tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    rules_file = policy_root / "statutes" / "26" / "63" / "f.yaml"
    imported_file = policy_root / "statutes" / "26" / "151.yaml"
    rules_file.parent.mkdir(parents=True)
    imported_file.parent.mkdir(parents=True, exist_ok=True)
    rules_file.write_text(
        """format: rulespec/v1
rules:
  - name: spouse_of_taxpayer_for_subsection_f
    kind: data_relation
    data_relation:
      predicate: us:concepts/tax#spouse_of_taxpayer_for_subsection_f
      arity: 2
      arguments:
        - name: spouse
          type: Person
        - name: tax_unit
          type: TaxUnit
  - name: taxpayer_blind_additional_amount_entitlement
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: taxpayer_is_blind_at_close_of_taxable_year
"""
    )
    imported_file.write_text(
        """format: rulespec/v1
rules:
  - name: exemption_individual_eligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: tin_included_on_return_claiming_exemption
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_root,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    dataset = pipeline._build_rulespec_dataset(
        {
            "us:statutes/26/63/f#relation.spouse_of_taxpayer_for_subsection_f": [
                {
                    "id": "spouse",
                    "us:statutes/26/151#input.tin_included_on_return_claiming_exemption": True,
                }
            ],
            "us:statutes/26/63/f#input.taxpayer_is_blind_at_close_of_taxable_year": True,
        },
        period={"start": "2026-01-01", "end": "2026-12-31"},
        query_entity="TaxUnit",
        query_entity_id="case-1",
        require_legal_input_keys=True,
    )

    assert dataset["relations"][0]["name"] == (
        "us:statutes/26/63/f#relation.spouse_of_taxpayer_for_subsection_f"
    )
    assert dataset["inputs"][0]["name"] == (
        "us:statutes/26/151#input.tin_included_on_return_claiming_exemption"
    )
    assert dataset["inputs"][1]["name"] == (
        "us:statutes/26/63/f#input.taxpayer_is_blind_at_close_of_taxable_year"
    )


def test_dataset_preserves_country_subdir_canonical_refs(
    tmp_path,
):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    policy_root.mkdir()
    rules_file = policy_root / "regulations" / "7-cfr" / "275" / "23" / "d" / "2.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
rules:
  - name: payment_liability_amount
    kind: derived
    entity: StateAgency
    dtype: Money
    period: Year
    versions:
      - effective_from: '2003-01-01'
        formula: payment_error_rate
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_root,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    dataset = pipeline._build_rulespec_dataset(
        {
            "us:regulations/7-cfr/275/23/d/2#input.payment_error_rate": 0.08,
        },
        period={"start": "2003-01-01", "end": "2003-12-31"},
        query_entity="StateAgency",
        query_entity_id="state-agency",
        require_legal_input_keys=True,
    )

    assert dataset["inputs"][0]["name"] == (
        "us:regulations/7-cfr/275/23/d/2#input.payment_error_rate"
    )


def test_cli_relation_request_preserves_country_subdir_canonical_refs(tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    policy_root.mkdir()

    relation_name = _rulespec_test_relation_request_name(
        "us:regulations/7-cfr/275/23/d/1#relation.state_agencies_for_quality_control_fiscal_year",
        policy_repo_path=policy_root,
    )

    assert relation_name == (
        "us:regulations/7-cfr/275/23/d/1"
        "#relation.state_agencies_for_quality_control_fiscal_year"
    )
