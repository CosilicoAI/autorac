"""Tests for raw-Git tree manifest construction and differencing."""

from __future__ import annotations

import hashlib

import pytest

from axiom_encode.notary.canonical import jcs_dumps, sha256_hex
from axiom_encode.notary.manifest import (
    ManifestDiffEntry,
    build_tree_manifest,
    fsck_clean,
    manifest_diff,
    manifest_sha256,
    validate_manifest,
)
from axiom_encode.notary.refusal import Refusal


def _digest(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _assert_refusal(result: object, detail: str) -> Refusal:
    assert isinstance(result, Refusal)
    assert result.code == "structural"
    assert result.detail == detail
    return result


def test_clean_repo_manifest_and_digest_are_stable(git_repo) -> None:
    git_repo.commit({"added.txt": b"added\n"}, message="second clean commit")
    first = build_tree_manifest(git_repo.path, "HEAD")
    second = build_tree_manifest(git_repo.path, "HEAD")

    assert (
        first
        == second
        == [
            ("added.txt", "100644", _digest(b"added\n")),
            ("seed.txt", "100644", _digest(b"seed\n")),
        ]
    )
    assert manifest_sha256(first) == manifest_sha256(second)
    assert manifest_sha256(first) == sha256_hex(
        jcs_dumps([list(entry) for entry in first])
    )
    assert fsck_clean(git_repo.path) is True


def test_manifest_reads_referenced_bytes_without_replace_objects(git_repo) -> None:
    original_oid = git_repo.rev_parse("HEAD:seed.txt")
    replacement_oid = git_repo.hash_blob(b"replacement bytes\n")
    git_repo.git("replace", original_oid, replacement_oid)

    assert build_tree_manifest(git_repo.path, "HEAD") == [
        ("seed.txt", "100644", _digest(b"seed\n"))
    ]


def test_manifest_refuses_tag_object_in_regular_file_entry(git_repo) -> None:
    blob = git_repo.hash_blob(b"tag target\n")
    git_repo.git("tag", "-a", "blob-tag", blob, "-m", "tag a blob")
    tag = git_repo.rev_parse("refs/tags/blob-tag")
    tree = git_repo.raw_tree([("100644", b"tagged", tag)])

    _assert_refusal(
        build_tree_manifest(git_repo.path, tree),
        "blob_object_unreadable",
    )


def test_manifest_refuses_commit_object_in_directory_entry(git_repo) -> None:
    commit = git_repo.rev_parse("HEAD")
    tree = git_repo.raw_tree([("40000", b"not-a-tree", commit)])

    _assert_refusal(
        build_tree_manifest(git_repo.path, tree),
        "tree_object_unreadable",
    )


def test_manifest_refuses_tag_object_in_directory_entry(git_repo) -> None:
    actual_tree = git_repo.rev_parse("HEAD^{tree}")
    git_repo.git("tag", "-a", "tree-tag", actual_tree, "-m", "tag a tree")
    tag = git_repo.rev_parse("refs/tags/tree-tag")
    tree = git_repo.raw_tree([("40000", b"tagged-tree", tag)])

    _assert_refusal(
        build_tree_manifest(git_repo.path, tree),
        "tree_object_unreadable",
    )


@pytest.mark.parametrize(
    ("factory", "detail"),
    [
        ("gitlink_tree", "gitlink"),
        ("non_utf8_tree", "non_utf8_path"),
        ("symlink_tree", "symlink"),
        ("inadmissible_mode_tree", "inadmissible_mode=100600"),
        ("empty_subtree_tree", "empty_subtree"),
        ("duplicate_flattened_path_tree", "duplicate_flattened_terminal_path"),
    ],
)
def test_pathological_tree_is_refused(git_repo, factory: str, detail: str) -> None:
    tree = getattr(git_repo, factory)()

    _assert_refusal(build_tree_manifest(git_repo.path, tree), detail)


@pytest.mark.parametrize("changed", [False, True], ids=["unchanged", "changed"])
def test_non_utf8_path_is_refused_whether_changed_or_unchanged(
    git_repo,
    changed: bool,
) -> None:
    base = git_repo.non_utf8_tree(b"base invalid bytes\n")
    subject = git_repo.non_utf8_tree(
        b"subject invalid bytes\n" if changed else b"base invalid bytes\n",
        stable_contents=b"subject stable bytes\n",
    )

    _assert_refusal(build_tree_manifest(git_repo.path, base), "non_utf8_path")
    _assert_refusal(build_tree_manifest(git_repo.path, subject), "non_utf8_path")


def test_non_utf8_directory_name_is_refused(git_repo) -> None:
    blob = git_repo.hash_blob(b"nested\n")
    subtree = git_repo.mktree([("100644", "blob", blob, b"file.txt")])
    tree = git_repo.mktree([("040000", "tree", subtree, b"bad-\xff")])

    _assert_refusal(build_tree_manifest(git_repo.path, tree), "non_utf8_path")


def test_root_empty_tree_is_refused(git_repo) -> None:
    tree = git_repo.mktree([])

    refusal = _assert_refusal(
        build_tree_manifest(git_repo.path, tree),
        "empty_subtree",
    )
    assert refusal.path is None


def test_two_trees_differing_only_by_gitlink_are_both_refused(git_repo) -> None:
    first_target = git_repo.rev_parse("HEAD")
    second_target = git_repo.commit(
        {"seed.txt": b"second commit\n"},
        message="second gitlink target",
    )
    first_tree = git_repo.mktree([("160000", "commit", first_target, b"dependency")])
    second_tree = git_repo.mktree([("160000", "commit", second_target, b"dependency")])

    _assert_refusal(build_tree_manifest(git_repo.path, first_tree), "gitlink")
    _assert_refusal(build_tree_manifest(git_repo.path, second_tree), "gitlink")


def test_tree_wide_refusal_precedence_is_in_required_order(git_repo) -> None:
    blob = git_repo.hash_blob(b"entry\n")
    commit = git_repo.rev_parse("HEAD")
    empty_tree = git_repo.mktree([])
    cases = [
        (
            [("100644", b"bad-\xff", blob), ("160000", b"z", commit)],
            "gitlink",
        ),
        (
            [("120000", b"a", blob), ("100644", b"bad-\xff", blob)],
            "non_utf8_path",
        ),
        (
            [("100600", b"a", blob), ("120000", b"z", blob)],
            "symlink",
        ),
        (
            [("40000", b"a", empty_tree), ("100600", b"z", blob)],
            "inadmissible_mode=100600",
        ),
        (
            [
                ("100644", b"a", blob),
                ("100644", b"a", blob),
                ("40000", b"z", empty_tree),
            ],
            "empty_subtree",
        ),
    ]

    for entries, expected_detail in cases:
        tree = git_repo.raw_tree(entries)
        _assert_refusal(
            build_tree_manifest(git_repo.path, tree),
            expected_detail,
        )


def test_manifest_validation_refuses_sort_order_violation() -> None:
    manifest = [
        ("z", "100644", "0" * 64),
        ("a", "100644", "1" * 64),
    ]

    assert validate_manifest(manifest) == Refusal(
        "structural",
        "a",
        "manifest_out_of_order",
    )


def test_manifest_build_sorts_paths_by_utf8_bytes(git_repo) -> None:
    blob = git_repo.hash_blob(b"same bytes\n")
    first = "\ue000"
    second = "\U00010000"
    tree = git_repo.raw_tree(
        [
            ("100644", second.encode(), blob),
            ("100644", first.encode(), blob),
        ]
    )

    assert build_tree_manifest(git_repo.path, tree) == [
        (first, "100644", _digest(b"same bytes\n")),
        (second, "100644", _digest(b"same bytes\n")),
    ]


def test_manifest_diff_is_exact_for_all_change_kinds(git_repo) -> None:
    base = git_repo.commit(
        {
            "delete.txt": b"delete me\n",
            "mode.txt": b"same bytes\n",
            "modify.txt": b"before\n",
            "stay.txt": b"stay\n",
        },
        message="base",
    )
    subject = git_repo.commit(
        {
            "add.txt": b"added\n",
            "delete.txt": None,
            "mode.txt": b"same bytes\n",
            "modify.txt": b"after\n",
        },
        message="subject",
        executable={"mode.txt"},
    )
    base_manifest = build_tree_manifest(git_repo.path, base)
    subject_manifest = build_tree_manifest(git_repo.path, subject)
    assert not isinstance(base_manifest, Refusal)
    assert not isinstance(subject_manifest, Refusal)

    assert manifest_diff(base_manifest, subject_manifest) == [
        ManifestDiffEntry(
            "add.txt",
            None,
            None,
            "100644",
            _digest(b"added\n"),
        ),
        ManifestDiffEntry(
            "delete.txt",
            "100644",
            _digest(b"delete me\n"),
            None,
            None,
        ),
        ManifestDiffEntry(
            "mode.txt",
            "100644",
            _digest(b"same bytes\n"),
            "100755",
            _digest(b"same bytes\n"),
        ),
        ManifestDiffEntry(
            "modify.txt",
            "100644",
            _digest(b"before\n"),
            "100644",
            _digest(b"after\n"),
        ),
    ]


def test_manifest_diff_represents_rename_as_addition_and_deletion() -> None:
    digest = "0" * 64

    assert manifest_diff(
        [("old.txt", "100644", digest)],
        [("new.txt", "100644", digest)],
    ) == [
        ManifestDiffEntry("new.txt", None, None, "100644", digest),
        ManifestDiffEntry("old.txt", "100644", digest, None, None),
    ]


def test_fsck_clean_reports_dirty_raw_tree(git_repo) -> None:
    assert fsck_clean(git_repo.path) is True
    git_repo.duplicate_flattened_path_tree()

    assert fsck_clean(git_repo.path) is False


def test_unresolvable_tree_is_typed_refusal(git_repo) -> None:
    assert build_tree_manifest(git_repo.path, "does-not-exist") == Refusal(
        "structural",
        None,
        "tree_unresolvable",
    )
