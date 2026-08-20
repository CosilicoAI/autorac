"""Tiny Git repository builders, including deliberately malformed trees."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

type MktreeEntry = tuple[str, str, str, bytes]
type RawTreeEntry = tuple[str, bytes, str]


@dataclass(slots=True)
class GitRepoBuilder:
    """Build ordinary commits and raw pathological Git objects."""

    path: Path

    def git(
        self,
        *args: str,
        input_bytes: bytes | None = None,
        check: bool = True,
    ) -> bytes:
        result = subprocess.run(
            ["git", "-C", str(self.path), *args],
            input=input_bytes,
            check=False,
            capture_output=True,
        )
        if check and result.returncode != 0:
            raise AssertionError(
                f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace')}"
            )
        return result.stdout

    def rev_parse(self, revision: str) -> str:
        return self.git("rev-parse", "--verify", revision).decode().strip()

    def commit(
        self,
        changes: Mapping[str, bytes | None],
        *,
        message: str,
        executable: Iterable[str] = (),
    ) -> str:
        executable_paths = set(executable)
        for relative, contents in changes.items():
            destination = self.path / relative
            if contents is None:
                if destination.exists():
                    destination.unlink()
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)
            destination.chmod(0o755 if relative in executable_paths else 0o644)

        self.git("add", "-A")
        for relative in executable_paths:
            self.git("update-index", "--chmod=+x", "--", relative)
        self.git("commit", "--quiet", "-m", message)
        return self.rev_parse("HEAD")

    def hash_blob(self, contents: bytes) -> str:
        return (
            self.git(
                "hash-object",
                "-w",
                "--stdin",
                input_bytes=contents,
            )
            .decode()
            .strip()
        )

    def mktree(self, entries: Sequence[MktreeEntry]) -> str:
        payload = b"".join(
            f"{mode} {object_type} {oid}\t".encode() + path + b"\0"
            for mode, object_type, oid, path in entries
        )
        return self.git("mktree", "-z", input_bytes=payload).decode().strip()

    def raw_tree(self, entries: Sequence[RawTreeEntry]) -> str:
        payload = b"".join(
            mode.encode() + b" " + path + b"\0" + bytes.fromhex(oid)
            for mode, path, oid in entries
        )
        return (
            self.git(
                "hash-object",
                "-w",
                "--literally",
                "-t",
                "tree",
                "--stdin",
                input_bytes=payload,
            )
            .decode()
            .strip()
        )

    def gitlink_tree(self, target: str | None = None) -> str:
        target = target or self.rev_parse("HEAD")
        inner = self.mktree([("160000", "commit", target, b"dependency")])
        return self.mktree([("040000", "tree", inner, b"nested")])

    def non_utf8_tree(
        self,
        contents: bytes = b"invalid path contents\n",
        *,
        stable_contents: bytes = b"stable\n",
    ) -> str:
        invalid_blob = self.hash_blob(contents)
        stable_blob = self.hash_blob(stable_contents)
        return self.mktree(
            [
                ("100644", "blob", invalid_blob, b"bad-\xff"),
                ("100644", "blob", stable_blob, b"stable.txt"),
            ]
        )

    def symlink_tree(self) -> str:
        target = self.hash_blob(b"target.txt")
        inner = self.mktree([("120000", "blob", target, b"link")])
        return self.mktree([("040000", "tree", inner, b"nested")])

    def empty_subtree_tree(self) -> str:
        empty_tree = self.mktree([])
        regular_blob = self.hash_blob(b"visible\n")
        return self.mktree(
            [
                ("040000", "tree", empty_tree, b"empty"),
                ("100644", "blob", regular_blob, b"visible.txt"),
            ]
        )

    def inadmissible_mode_tree(self) -> str:
        blob = self.hash_blob(b"bad mode\n")
        return self.raw_tree([("100600", b"mode.txt", blob)])

    def duplicate_flattened_path_tree(self) -> str:
        blob = self.hash_blob(b"duplicate\n")
        subtree = self.mktree([("100644", "blob", blob, b"leaf")])
        return self.raw_tree(
            [
                ("40000", b"dir", subtree),
                ("100644", b"dir/leaf", blob),
            ]
        )


@pytest.fixture
def git_repo(tmp_path: Path) -> GitRepoBuilder:
    repo = tmp_path / "repo"
    subprocess.run(
        ["git", "init", "--quiet", "--object-format=sha1", str(repo)],
        check=True,
        capture_output=True,
    )
    builder = GitRepoBuilder(repo)
    builder.git("config", "user.name", "Notary Test")
    builder.git("config", "user.email", "notary@example.test")
    builder.git("config", "commit.gpgsign", "false")
    builder.git("config", "core.autocrlf", "false")
    builder.git("config", "core.filemode", "true")
    builder.commit({"seed.txt": b"seed\n"}, message="seed")
    return builder
