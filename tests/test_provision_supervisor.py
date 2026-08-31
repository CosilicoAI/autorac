"""Unit coverage for the provisioner's containment logic.

The root/ELF end-to-end paths run on the hosted image in
.github/workflows/provision-selftest.yml; these tests pin the pure logic —
prefix-boundary predicates, rpath rewriting, staging order, and the
copy-equivalent preflight — with no root or patchelf required.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "provision_verification_supervisor",
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "provision_verification_supervisor.py",
)
provisioner = importlib.util.module_from_spec(_SPEC)
# Register in sys.modules before exec so the module resolves itself by name.
sys.modules[_SPEC.name] = provisioner
_SPEC.loader.exec_module(provisioner)


class TestPathInside:
    def test_absolute_outside(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        assert not provisioner._path_inside("/opt/hostedtoolcache/x/lib", runtime)

    def test_inside(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        (runtime / "lib").mkdir(parents=True)
        assert provisioner._path_inside(str(runtime / "lib"), runtime)

    def test_sibling_prefix_collision_outside(self, tmp_path):
        """<runtime>-evil must not pass as inside (the startswith trap)."""
        runtime = (tmp_path / "runtime").resolve()
        assert not provisioner._path_inside(str(tmp_path / "runtime-evil"), runtime)

    def test_dotdot_traversal_outside(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        sneaky = str(runtime / "lib" / ".." / ".." / "outside")
        assert not provisioner._path_inside(sneaky, runtime)

    def test_outside_alias_symlinked_into_runtime_is_outside(self, tmp_path):
        """/outside/alias -> <runtime>/lib resolves inside, but the loader reads
        the literal (retargetable) path: lexical check keeps it out."""
        runtime = (tmp_path / "runtime").resolve()
        (runtime / "lib").mkdir(parents=True)
        alias = tmp_path / "outside" / "alias"
        alias.parent.mkdir()
        alias.symlink_to(runtime / "lib")
        assert not provisioner._path_inside(str(alias), runtime)

    def test_inside_alias_symlinked_out_is_outside(self, tmp_path):
        """<runtime>/alias/../lib normalizes lexically to <runtime>/lib, but the
        loader follows `alias` (to outside) BEFORE `..`: resolve check keeps it
        out. This is the case lexical-only would wrongly accept."""
        runtime = (tmp_path / "runtime").resolve()
        (runtime / "lib").mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (runtime / "alias").symlink_to(outside)
        assert not provisioner._path_inside(
            str(runtime / "alias" / ".." / "lib"), runtime
        )


class TestPinnedCodexCLI:
    def _archive(self, tmp_path: Path, binary: bytes) -> Path:
        source = tmp_path / "package/vendor/aarch64-apple-darwin/bin/codex"
        source.parent.mkdir(parents=True)
        source.write_bytes(binary)
        archive = tmp_path / "codex.tgz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(tmp_path / "package", arcname="package")
        return archive

    def test_installs_exact_hash_pinned_binary(self, tmp_path, monkeypatch):
        binary = b"#!/bin/sh\necho codex-cli-test\n"
        archive = self._archive(tmp_path, binary)
        monkeypatch.setattr(provisioner.sys, "platform", "darwin")
        monkeypatch.setattr(provisioner.platform, "machine", lambda: "arm64")
        monkeypatch.setitem(
            provisioner._CODEX_CLI_PINS[("darwin", "arm64")],
            "sha256",
            provisioner.hashlib.sha256(binary).hexdigest(),
        )
        destination = tmp_path / "runtime"
        destination.mkdir()
        config = provisioner._install_pinned_codex_cli(destination, archive)
        assert (destination / "bin/codex").read_bytes() == binary
        assert config["sha256"] == provisioner.hashlib.sha256(binary).hexdigest()

    def test_hash_mismatch_hard_fails_and_removes_binary(self, tmp_path, monkeypatch):
        archive = self._archive(tmp_path, b"tampered")
        monkeypatch.setattr(provisioner.sys, "platform", "darwin")
        monkeypatch.setattr(provisioner.platform, "machine", lambda: "arm64")
        monkeypatch.setitem(
            provisioner._CODEX_CLI_PINS[("darwin", "arm64")],
            "sha256",
            "0" * 64,
        )
        destination = tmp_path / "runtime"
        destination.mkdir()
        with pytest.raises(SystemExit, match="sha256 mismatch"):
            provisioner._install_pinned_codex_cli(destination, archive)
        assert not (destination / "bin/codex").exists()


class TestRpathComponentInside:
    def _runtime(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        (runtime / "lib").mkdir(parents=True)
        return runtime, runtime / "lib"

    def test_origin_relative_staying_inside_is_safe(self, tmp_path):
        runtime, object_dir = self._runtime(tmp_path)
        assert provisioner._rpath_component_inside("$ORIGIN", object_dir, runtime)
        assert provisioner._rpath_component_inside(
            "$ORIGIN/../lib", object_dir, runtime
        )
        assert provisioner._rpath_component_inside("${ORIGIN}/foo", object_dir, runtime)

    def test_origin_climbing_out_is_unsafe(self, tmp_path):
        runtime, object_dir = self._runtime(tmp_path)
        assert not provisioner._rpath_component_inside(
            "$ORIGIN/../../../../usr/lib", object_dir, runtime
        )

    def test_bogus_origin_token_not_expanded(self, tmp_path):
        """$ORIGIN_evil is not the gABI token (identifier char follows), so it
        stays unexpanded, is not absolute, and is therefore unsafe — not
        silently accepted as a rewritten <dir>_evil path."""
        runtime, object_dir = self._runtime(tmp_path)
        assert not provisioner._rpath_component_inside(
            "$ORIGIN_evil", object_dir, runtime
        )

    def test_empty_and_cwd_relative_are_unsafe(self, tmp_path):
        runtime, object_dir = self._runtime(tmp_path)
        assert not provisioner._rpath_component_inside("", object_dir, runtime)
        assert not provisioner._rpath_component_inside("lib", object_dir, runtime)
        assert not provisioner._rpath_component_inside(".", object_dir, runtime)
        # a leading-space value must not be silently trimmed into an abs path
        assert not provisioner._rpath_component_inside(" /x", object_dir, runtime)

    def test_absolute_outside_is_unsafe(self, tmp_path):
        runtime, object_dir = self._runtime(tmp_path)
        assert not provisioner._rpath_component_inside(
            "/opt/hostedtoolcache/Python/3.14.0/x64/lib", object_dir, runtime
        )


class TestOriginExpand:
    def test_only_real_token_expands(self, tmp_path):
        object_dir = Path("/rt/lib")
        assert provisioner._origin_expand("$ORIGIN/x", object_dir) == "/rt/lib/x"
        assert provisioner._origin_expand("${ORIGIN}/x", object_dir) == "/rt/lib/x"
        # identifier char after ORIGIN → not the token, left verbatim
        assert provisioner._origin_expand("$ORIGIN_evil", object_dir) == "$ORIGIN_evil"
        assert provisioner._origin_expand("$ORIGINX", object_dir) == "$ORIGINX"


class TestRewriteRpathForObject:
    def test_mixed_list_rewrites_only_unsafe_preserving_origin(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        object_dir = runtime / "lib"
        object_dir.mkdir(parents=True)
        target = str(runtime / "lib")
        mixed = "$ORIGIN/../lib:/opt/hostedtoolcache/Python/3.14.0/x64/lib"
        new_rpath, changed = provisioner._rewrite_rpath_for_object(
            mixed, object_dir, runtime, target
        )
        assert changed
        assert new_rpath == f"$ORIGIN/../lib:{target}"

    def test_all_safe_list_is_unchanged(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        object_dir = runtime / "lib"
        object_dir.mkdir(parents=True)
        target = str(runtime / "lib")
        _, changed = provisioner._rewrite_rpath_for_object(
            "$ORIGIN:$ORIGIN/../lib", object_dir, runtime, target
        )
        assert not changed

    def test_empty_and_escaping_collapse_deduplicated(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        object_dir = runtime / "lib"
        object_dir.mkdir(parents=True)
        target = str(runtime / "lib")
        # empty component + two escapes + one origin-safe → target once, then origin
        new_rpath, changed = provisioner._rewrite_rpath_for_object(
            ":/one/outside:/two/outside:$ORIGIN", object_dir, runtime, target
        )
        assert changed
        assert new_rpath == f"{target}:$ORIGIN"


class TestStageRuntimeTree:
    def _make_source(self, tmp_path: Path) -> Path:
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        source = tmp_path / "source"
        site = source / "lib" / version / "site-packages"
        site.mkdir(parents=True)
        (source / "bin").mkdir()
        (source / "bin" / "python3").write_text("fake interpreter")
        (site / "toolcache-only.txt").write_text("replaced wholesale")
        return source

    def test_stdlib_zip_is_purged_across_platlibdir_and_abi(self, tmp_path):
        """The canonical stdlib zip is on sys.path even absent; purge it in lib,
        lib64 (PLATLIBDIR), and the free-threaded ``t`` name."""
        source = self._make_source(tmp_path)
        major, minor = sys.version_info.major, sys.version_info.minor
        for rel in (
            f"lib/python{major}{minor}.zip",
            f"lib64/python{major}{minor}.zip",
            f"lib/python{major}{minor}t.zip",
        ):
            zip_path = source / rel
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(b"PK\x03\x04")
        staged = tmp_path / "staged-empty"
        staged.mkdir()
        runtime = tmp_path / "dest" / "python"
        runtime.parent.mkdir()
        provisioner._stage_runtime_tree(source, runtime, staged)
        assert not list(runtime.rglob(f"python{major}{minor}*.zip"))

    def test_staged_startup_hooks_are_purged(self, tmp_path):
        """Every importable/path-config startup carrier in the STAGED tree must
        be gone: the launcher runs -I (site enabled), which executes them. A
        bare `sitecustomize.py` glob would leave the package dir, .pyc, and
        extension-module forms importable, and ._pth/pybuilddir.txt reconfigure
        sys.path."""
        source = self._make_source(tmp_path)
        staged = tmp_path / "staged"
        staged.mkdir()
        (staged / "evil.pth").write_text("import os; os.system('pwned')")
        (staged / "python._pth").write_text("import site")
        (staged / "pybuilddir.txt").write_text("build")
        (staged / "sitecustomize.py").write_text("print('pwned')")
        (staged / "sitecustomize.pyc").write_bytes(b"\x00")
        (staged / "sitecustomize.cpython-314-x86_64-linux-gnu.so").write_bytes(b"\x00")
        (staged / "usercustomize.py").write_text("print('pwned')")
        pkg = staged / "sitecustomize"  # importable package form
        pkg.mkdir()
        (pkg / "__init__.py").write_text("import os; os.system('pwned')")
        # A carrier nested INSIDE a doomed dir: exercises parent-then-child
        # deletion (the guard must skip the already-removed child, not raise).
        (pkg / "nested.pth").write_text("import os")
        (staged / "yaml.py").write_text("legit = True")
        runtime = tmp_path / "dest" / "python"
        runtime.parent.mkdir()
        provisioner._stage_runtime_tree(source, runtime, staged)
        for orphan in (
            "*.pth",
            "*._pth",
            "pybuilddir.txt",
            "sitecustomize*",
            "usercustomize*",
        ):
            assert not list(runtime.rglob(orphan)), orphan
        assert not any(p.name == "sitecustomize" for p in runtime.rglob("*"))
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        site = runtime / "lib" / version / "site-packages"
        assert (site / "yaml.py").read_text() == "legit = True"
        assert not (site / "toolcache-only.txt").exists()


class TestSanePrefixPreflight:
    def test_forbidden_system_prefix_refused(self):
        with pytest.raises(SystemExit, match="not a self-contained"):
            provisioner._assert_sane_source_prefix(Path("/usr"), None, 100)

    def test_prefix_outside_required_parent_refused(self, tmp_path):
        prefix = tmp_path / "prefix"
        prefix.mkdir()
        required = tmp_path / "elsewhere"
        required.mkdir()
        with pytest.raises(SystemExit, match="not under required parent"):
            provisioner._assert_sane_source_prefix(prefix, required, 100)

    def test_file_cap_refuses(self, tmp_path):
        prefix = tmp_path / "prefix"
        prefix.mkdir()
        for index in range(5):
            (prefix / f"file{index}").write_text("x")
        with pytest.raises(SystemExit, match="more than 3 files"):
            provisioner._assert_sane_source_prefix(prefix, None, 3)

    def test_symlink_escaping_prefix_refused(self, tmp_path):
        """copytree(symlinks=False) would follow this into the outside world,
        so the copy-equivalent count must refuse it up front."""
        prefix = tmp_path / "prefix"
        prefix.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "huge").write_text("x")
        (prefix / "leak").symlink_to(outside)
        with pytest.raises(SystemExit, match="symlink escapes the source prefix"):
            provisioner._assert_sane_source_prefix(prefix, None, 100)

    def test_internal_diamond_counted_like_the_copy(self, tmp_path):
        """lib64 -> lib is copied twice by copytree, not a cycle: it must pass
        (and count double, exactly like the copy it bounds)."""
        prefix = tmp_path / "prefix"
        lib = prefix / "lib"
        lib.mkdir(parents=True)
        (lib / "libpython.so").write_text("x")
        (prefix / "lib64").symlink_to(lib)
        provisioner._assert_sane_source_prefix(prefix, None, 100)
        with pytest.raises(SystemExit, match="more than 1 files"):
            provisioner._assert_sane_source_prefix(prefix, None, 1)

    def test_ancestor_cycle_refused(self, tmp_path):
        prefix = tmp_path / "prefix"
        inner = prefix / "inner"
        inner.mkdir(parents=True)
        (inner / "loop").symlink_to(prefix)
        with pytest.raises(SystemExit, match="symlink cycle"):
            provisioner._assert_sane_source_prefix(prefix, None, 100)


class TestTrustedGit:
    def test_accepts_root_owned_system_git(self):
        git = shutil.which("git")
        if git is None:
            pytest.skip("Git is required")
        resolved = Path(git).resolve()
        assert provisioner._resolve_trusted_git(resolved) == resolved

    def test_rejects_relative_path(self):
        with pytest.raises(SystemExit, match="absolute and normalized"):
            provisioner._resolve_trusted_git(Path("git"))

    def test_rejects_symlinked_path(self, tmp_path):
        git = shutil.which("git")
        if git is None:
            pytest.skip("Git is required")
        alias = tmp_path / "git"
        alias.symlink_to(Path(git).resolve())
        with pytest.raises(SystemExit, match="contains a symlink"):
            provisioner._resolve_trusted_git(alias)

    @pytest.mark.parametrize("symlink", [False, True])
    def test_installed_wrapper_refuses_existing_path(self, tmp_path, symlink):
        git = shutil.which("git")
        if git is None:
            pytest.skip("Git is required")
        tool_directory = tmp_path / "tools"
        tool_directory.mkdir()
        wrapper = tool_directory / "git"
        sentinel = tmp_path / "sentinel"
        sentinel.write_text("unchanged\n")
        if symlink:
            wrapper.symlink_to(sentinel)
        else:
            wrapper.write_text("unchanged\n")

        with pytest.raises(SystemExit, match="wrapper path already exists"):
            provisioner._install_trusted_git_wrapper(
                tool_directory,
                Path(sys.executable).resolve(),
                provisioner._resolve_trusted_git(Path(git).resolve()),
            )

        assert sentinel.read_text() == "unchanged\n"
        if not symlink:
            assert wrapper.read_text() == "unchanged\n"

    def test_installed_wrapper_blocks_local_executable_config(self, tmp_path):
        git = shutil.which("git")
        if git is None:
            pytest.skip("Git is required")
        destination = tmp_path / "destination"
        destination.mkdir()
        wrapper = provisioner._install_trusted_git_wrapper(
            destination,
            Path(sys.executable).resolve(),
            provisioner._resolve_trusted_git(Path(git).resolve()),
        )
        repository = tmp_path / "rulespec-us"
        subprocess.run([git, "init", "--quiet", str(repository)], check=True)
        (repository / "a.txt").write_text("original\n")
        (repository / ".gitattributes").write_text("*.txt diff=hostile\n")
        subprocess.run([git, "-C", str(repository), "add", "."], check=True)
        subprocess.run(
            [
                git,
                "-c",
                "user.name=Axiom test",
                "-c",
                "user.email=test@axiom.invalid",
                "-C",
                str(repository),
                "commit",
                "--quiet",
                "-m",
                "fixture",
            ],
            check=True,
        )
        marker = tmp_path / "helper-executed"
        helper = tmp_path / "hostile-helper"
        helper.write_text(
            f"#!{Path(sys.executable).resolve()}\n"
            "import sys\n"
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).touch()\n"
            "sys.stdout.buffer.write(sys.stdin.buffer.read())\n"
        )
        helper.chmod(0o755)
        for key in (
            "core.fsmonitor",
            "diff.external",
            "diff.hostile.textconv",
        ):
            subprocess.run(
                [git, "-C", str(repository), "config", key, str(helper)], check=True
            )
        subprocess.run(
            [
                git,
                "-C",
                str(repository),
                "remote",
                "add",
                "origin",
                "https://github.com/TheAxiomFoundation/rulespec-us.git",
            ],
            check=True,
        )
        (repository / "a.txt").write_text("changed\n")
        (repository / "untracked.txt").write_text("untracked\n")
        clean_environment = {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.fsmonitor",
            "GIT_CONFIG_VALUE_0": str(helper),
            "GIT_EXTERNAL_DIFF": str(helper),
            "HOME": str(tmp_path),
            "PATH": str(destination),
        }
        object_id = subprocess.check_output(
            [git, "-C", str(repository), "rev-parse", "HEAD:a.txt"], text=True
        ).strip()
        head = subprocess.check_output(
            [git, "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        allowed_commands = (
            (["rev-parse", "HEAD"], None),
            (["rev-parse", "HEAD^{tree}"], None),
            (["rev-parse", "--is-inside-work-tree"], None),
            (["rev-parse", "--show-toplevel"], None),
            (["rev-parse", "--verify", "HEAD^{commit}"], None),
            (["rev-parse", f"{head}^{{commit}}"], None),
            (["rev-parse", f"{head}^{{tree}}"], None),
            (["status", "--porcelain"], None),
            (["status", "--porcelain", "--untracked-files=no"], None),
            (
                [
                    "status",
                    "--porcelain=v1",
                    "-z",
                    "--untracked-files=all",
                    "--ignored=matching",
                ],
                None,
            ),
            (
                [
                    "status",
                    "--porcelain=v1",
                    "-z",
                    "--untracked-files=all",
                ],
                None,
            ),
            (["remote", "get-url", "origin"], None),
            (["diff", "--binary", "HEAD", "--", "a.txt"], None),
            (
                [
                    "diff",
                    "--name-only",
                    "--no-renames",
                    "--diff-filter=ACDMRT",
                    "-z",
                    "HEAD",
                ],
                None,
            ),
            (
                [
                    "diff",
                    "--name-only",
                    "--no-renames",
                    "--diff-filter=ACDMRT",
                    "-z",
                    "HEAD",
                    "HEAD",
                ],
                None,
            ),
            (
                ["diff", "--name-only", "--diff-filter=ACMRT", "HEAD..HEAD"],
                None,
            ),
            (["ls-files", "-z"], None),
            (["ls-files", "--stage", "-z"], None),
            (["ls-files", "--others", "--exclude-standard", "-z"], None),
            (
                [
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                    "-z",
                    "--",
                    "untracked.txt",
                ],
                None,
            ),
            (
                [
                    "log",
                    "--format=%H",
                    "--",
                    "pyproject.toml",
                    "src/axiom_encode/__init__.py",
                    "uv.lock",
                ],
                None,
            ),
            (["ls-tree", "-z", "HEAD", "--", "a.txt"], None),
            (
                [
                    "ls-tree",
                    "-z",
                    "--full-tree",
                    "HEAD",
                    "--",
                    "us/statutes/42/1437c–1.yaml",
                ],
                None,
            ),
            (["ls-tree", "-r", "-z", "HEAD"], None),
            (["ls-tree", "-r", "-l", "-z", "--full-tree", head], None),
            (
                [
                    "grep",
                    "-z",
                    "-l",
                    "-a",
                    "-F",
                    "-e",
                    "original",
                    head,
                    "--",
                ],
                None,
            ),
            (["cat-file", "blob", object_id], None),
            (["cat-file", "--batch"], f"{object_id}\n".encode()),
            (["merge-base", "--is-ancestor", "HEAD", "HEAD"], None),
            (["rev-list", "--parents", "-n", "1", "HEAD"], None),
            (["show", "HEAD:a.txt"], None),
        )
        for command, command_input in allowed_commands:
            subprocess.run(
                [str(wrapper), "-C", str(repository), *command],
                input=command_input,
                check=True,
                capture_output=True,
                env=clean_environment,
            )
        subprocess.run(
            [str(wrapper), "rev-parse", "--show-prefix"],
            cwd=repository,
            check=True,
            capture_output=True,
            env=clean_environment,
        )
        assert not marker.exists()

        (repository / ".gitattributes").write_text(
            "*.txt diff=hostile filter=hostile\n"
        )
        for key in ("filter.hostile.clean", "filter.hostile.smudge"):
            subprocess.run(
                [git, "-C", str(repository), "config", key, str(helper)], check=True
            )
        filtered_worktree_commands = (
            ["status", "--porcelain"],
            ["status", "--porcelain", "--untracked-files=no"],
            [
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignored=matching",
            ],
            [
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
            ],
            ["diff", "--binary", "HEAD", "--", "a.txt"],
            [
                "diff",
                "--name-only",
                "--no-renames",
                "--diff-filter=ACDMRT",
                "-z",
                "HEAD",
            ],
        )
        for command in filtered_worktree_commands:
            refused_filter = subprocess.run(
                [str(wrapper), "-C", str(repository), *command],
                check=False,
                capture_output=True,
                text=True,
                env=clean_environment,
            )
            assert refused_filter.returncode != 0
            assert "refused worktree filter for" in refused_filter.stderr
        assert not marker.exists()
        for driver in ("unspecified", "unset"):
            (repository / ".gitattributes").write_text(f"*.txt filter={driver}\n")
            subprocess.run(
                [
                    git,
                    "-C",
                    str(repository),
                    "config",
                    f"filter.{driver}.clean",
                    str(helper),
                ],
                check=True,
            )
            refused_sentinel = subprocess.run(
                [
                    str(wrapper),
                    "-C",
                    str(repository),
                    "diff",
                    "--binary",
                    "HEAD",
                    "--",
                    "a.txt",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=clean_environment,
            )
            assert refused_sentinel.returncode != 0
            assert "refused worktree filter for" in refused_sentinel.stderr
            assert not marker.exists()

        refused_arguments = (
            ["cat-file", "--filters", "HEAD:a.txt"],
            ["diff", "--stat", "HEAD"],
            ["log", "--oneline"],
            ["ls-files", "--stage"],
            ["ls-files", "-z", "--stage"],
            ["ls-files", "--stage", "-z", "--debug"],
            ["ls-tree", "--name-only", "HEAD"],
            ["ls-tree", "--full-tree", "-z", "HEAD", "--", "a.txt"],
            ["ls-tree", "-z", "--full-tree", "HEAD", "--long", "--", "a.txt"],
            ["ls-tree", "-z", "--full-tree", "HEAD", "--"],
            ["ls-tree", "-r", "-l", "-z", head, "--full-tree"],
            ["ls-tree", "-r", "-l", "-z", "--full-tree", head, "--"],
            ["grep", "-z", "-l", "-a", "-F", head, "--"],
            ["grep", "-z", "-l", "-a", "-F", "-e", "--help", head, "--"],
            [
                "grep",
                "-z",
                "-l",
                "-a",
                "-F",
                "-e",
                "us:statutes/42/1437c–1",
                "--recurse-submodules",
                head,
                "--",
            ],
            ["rev-parse", f"{head}^{{blob}}"],
            ["rev-parse", f"{head}^{{tree}}", "--verify"],
            ["merge-base", "--octopus", "HEAD", "HEAD"],
            ["remote", "add", "blocked", "https://example.invalid/repo.git"],
            ["remote", "update"],
            ["rev-list", "--all"],
            ["rev-parse", "--git-dir"],
            ["show", "--stat", "HEAD"],
            ["status", "--short"],
        )
        for command in refused_arguments:
            refused = subprocess.run(
                [str(wrapper), "-C", str(repository), *command],
                check=False,
                capture_output=True,
                text=True,
                env=clean_environment,
            )
            assert refused.returncode != 0
            assert f"refused arguments for {command[0]}" in refused.stderr
        assert not marker.exists()
        assert (
            subprocess.run(
                [git, "-C", str(repository), "remote", "get-url", "blocked"],
                check=False,
                capture_output=True,
            ).returncode
            != 0
        )

        refused_command = subprocess.run(
            [str(wrapper), "-C", str(repository), "fetch"],
            check=False,
            capture_output=True,
            text=True,
            env=clean_environment,
        )
        assert refused_command.returncode != 0
        assert "refused command: fetch" in refused_command.stderr

        child_source = tmp_path / "child-source"
        subprocess.run([git, "init", "--quiet", str(child_source)], check=True)
        (child_source / "a.txt").write_text("original\n")
        (child_source / ".gitattributes").write_text("*.txt filter=hostile\n")
        subprocess.run([git, "-C", str(child_source), "add", "."], check=True)
        subprocess.run(
            [
                git,
                "-c",
                "user.name=Axiom test",
                "-c",
                "user.email=test@axiom.invalid",
                "-C",
                str(child_source),
                "commit",
                "--quiet",
                "-m",
                "fixture",
            ],
            check=True,
        )
        parent = tmp_path / "parent"
        subprocess.run([git, "init", "--quiet", str(parent)], check=True)
        subprocess.run(
            [
                git,
                "-c",
                "protocol.file.allow=always",
                "-C",
                str(parent),
                "submodule",
                "add",
                "--quiet",
                str(child_source),
                "sub",
            ],
            check=True,
        )
        subprocess.run(
            [
                git,
                "-c",
                "user.name=Axiom test",
                "-c",
                "user.email=test@axiom.invalid",
                "-C",
                str(parent),
                "commit",
                "--quiet",
                "-m",
                "fixture",
            ],
            check=True,
        )
        initialized_submodule = parent / "sub"
        subprocess.run(
            [
                git,
                "-C",
                str(initialized_submodule),
                "config",
                "filter.hostile.clean",
                str(helper),
            ],
            check=True,
        )
        (initialized_submodule / "a.txt").write_text("changed\n")
        gitlink_commands = (
            ["status", "--porcelain"],
            ["status", "--porcelain", "--untracked-files=no"],
            [
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignored=matching",
            ],
            ["diff", "--binary", "HEAD", "--", "sub"],
        )
        for command in gitlink_commands:
            refused_gitlink = subprocess.run(
                [str(wrapper), "-C", str(parent), *command],
                check=False,
                capture_output=True,
                text=True,
                env=clean_environment,
            )
            assert refused_gitlink.returncode != 0
            assert "refused gitlink at sub" in refused_gitlink.stderr
        assert not marker.exists()

    def test_installed_wrapper_stages_only_explicit_safe_paths(self, tmp_path):
        git = shutil.which("git")
        if git is None:
            pytest.skip("Git is required")
        destination = tmp_path / "destination"
        destination.mkdir()
        wrapper = provisioner._install_trusted_git_wrapper(
            destination,
            Path(sys.executable).resolve(),
            provisioner._resolve_trusted_git(Path(git).resolve()),
        )
        repository = (tmp_path / "rulespec-us").resolve()
        subprocess.run([git, "init", "--quiet", str(repository)], check=True)
        for name in ("a.txt", "b.txt"):
            (repository / name).write_text("original\n")
        subprocess.run([git, "-C", str(repository), "add", "."], check=True)
        subprocess.run(
            [
                git,
                "-c",
                "user.name=Axiom test",
                "-c",
                "user.email=test@axiom.invalid",
                "-C",
                str(repository),
                "commit",
                "--quiet",
                "-m",
                "fixture",
            ],
            check=True,
        )
        for name in ("a.txt", "b.txt"):
            (repository / name).write_text("changed\n")
        clean_environment = {
            "HOME": str(tmp_path),
            "PATH": str(destination),
        }

        subprocess.run(
            [str(wrapper), "-C", str(repository), "add", "--", "a.txt"],
            check=True,
            capture_output=True,
            env=clean_environment,
        )
        cached = subprocess.run(
            [
                str(wrapper),
                "-C",
                str(repository),
                "diff",
                "--cached",
                "--name-only",
                "--no-renames",
                "-z",
            ],
            check=True,
            capture_output=True,
            env=clean_environment,
        )
        assert cached.stdout == b"a.txt\0"

        refused_arguments = (
            ["add", "a.txt"],
            ["add", "--"],
            ["add", "--all"],
            ["add", "--", "../b.txt"],
            ["add", "--", "./b.txt"],
            ["add", "--", str(repository / "b.txt")],
            ["add", "--", ":(glob)*"],
            ["add", "--", "*.txt"],
            ["add", "--", "b?.txt"],
            ["add", "--", "[ab].txt"],
            ["add", "--", "b\\*.txt"],
            ["add", "--", "-b.txt"],
        )
        for command in refused_arguments:
            refused = subprocess.run(
                [str(wrapper), "-C", str(repository), *command],
                check=False,
                capture_output=True,
                text=True,
                env=clean_environment,
            )
            assert refused.returncode != 0
            assert "refused arguments for add" in refused.stderr
        refused_without_repository = subprocess.run(
            [str(wrapper), "add", "--", "b.txt"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            env=clean_environment,
        )
        assert refused_without_repository.returncode != 0
        assert "refused arguments for add" in refused_without_repository.stderr
        assert (
            subprocess.check_output(
                [git, "-C", str(repository), "diff", "--cached", "--name-only"]
            )
            == b"a.txt\n"
        )

        subprocess.run(
            [git, "-C", str(repository), "reset", "--quiet", "HEAD"], check=True
        )
        marker = tmp_path / "filter-executed"
        helper = tmp_path / "hostile-filter"
        helper.write_text(
            f"#!{Path(sys.executable).resolve()}\n"
            "import sys\n"
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).touch()\n"
            "sys.stdout.buffer.write(sys.stdin.buffer.read())\n"
        )
        helper.chmod(0o755)
        (repository / ".gitattributes").write_text("*.txt filter=hostile\n")
        subprocess.run(
            [
                git,
                "-C",
                str(repository),
                "config",
                "filter.hostile.clean",
                str(helper),
            ],
            check=True,
        )
        refused_filter = subprocess.run(
            [str(wrapper), "-C", str(repository), "add", "--", "b.txt"],
            check=False,
            capture_output=True,
            text=True,
            env=clean_environment,
        )
        assert refused_filter.returncode != 0
        assert "refused worktree filter for" in refused_filter.stderr
        assert not marker.exists()
        assert not subprocess.check_output(
            [git, "-C", str(repository), "diff", "--cached", "--name-only"]
        )
        for attribute in (
            "text eol=lf",
            "crlf",
            "ident",
            "working-tree-encoding=UTF-16",
        ):
            (repository / ".gitattributes").write_text(f"*.txt {attribute}\n")
            refused_attribute = subprocess.run(
                [str(wrapper), "-C", str(repository), "add", "--", "b.txt"],
                check=False,
                capture_output=True,
                text=True,
                env=clean_environment,
            )
            assert refused_attribute.returncode != 0
            assert "refused content-transforming attribute" in (
                refused_attribute.stderr
            )
            assert not subprocess.check_output(
                [git, "-C", str(repository), "diff", "--cached", "--name-only"]
            )

        (repository / ".gitattributes").unlink()
        subprocess.run(
            [git, "-C", str(repository), "config", "core.autocrlf", "true"],
            check=True,
        )
        (repository / "b.txt").write_bytes(b"changed\r\n")
        subprocess.run(
            [str(wrapper), "-C", str(repository), "add", "--", "b.txt"],
            check=True,
            capture_output=True,
            env=clean_environment,
        )
        staged_object = subprocess.check_output(
            [git, "-C", str(repository), "rev-parse", ":b.txt"], text=True
        ).strip()
        assert (
            subprocess.check_output(
                [git, "-C", str(repository), "cat-file", "blob", staged_object]
            )
            == b"changed\r\n"
        )

    def test_installed_wrapper_supports_migration_git_queries(
        self, tmp_path, monkeypatch
    ):
        from axiom_encode.cli import (
            _legacy_replacement_reference_inventory_issues,
            _rulespec_migration_base_blob,
            _rulespec_migration_git,
            _rulespec_migration_tracked_files,
        )

        git = shutil.which("git")
        if git is None:
            pytest.skip("Git is required")
        destination = tmp_path / "destination"
        destination.mkdir()
        provisioner._install_trusted_git_wrapper(
            destination,
            Path(sys.executable).resolve(),
            provisioner._resolve_trusted_git(Path(git).resolve()),
        )
        repository = tmp_path / "rulespec-us"
        subprocess.run([git, "init", "--quiet", str(repository)], check=True)
        unicode_path = repository / "us/statutes/42/1437c–1.yaml"
        unicode_path.parent.mkdir(parents=True)
        unicode_path.write_text("format: rulespec/v1\n")
        source_sha256 = hashlib.sha256(unicode_path.read_bytes()).hexdigest()
        executable = repository / "tools/check"
        executable.parent.mkdir()
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o755)
        subprocess.run([git, "-C", str(repository), "add", "."], check=True)
        subprocess.run(
            [
                git,
                "-c",
                "user.name=Axiom test",
                "-c",
                "user.email=test@axiom.invalid",
                "-C",
                str(repository),
                "commit",
                "--quiet",
                "-m",
                "fixture",
            ],
            check=True,
        )
        head = subprocess.check_output(
            [git, "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()

        monkeypatch.setenv("PATH", str(destination))

        assert _rulespec_migration_tracked_files(repository) == {
            Path("tools/check"): "100755",
            Path("us/statutes/42/1437c–1.yaml"): "100644",
        }
        assert (
            _rulespec_migration_base_blob(
                repository, head, Path("us/statutes/42/1437c–1.yaml")
            )
            == b"format: rulespec/v1\n"
        )
        assert re.fullmatch(
            r"[0-9a-f]{40}",
            _rulespec_migration_git(
                repository, "rev-parse", f"{head}^{{tree}}"
            ).strip(),
        )
        unicode_path.unlink()
        assert (
            _legacy_replacement_reference_inventory_issues(
                repository,
                base_commit=head,
                authoritative_replacements={
                    "us:statutes/42/1437c–1": "us:statutes/42/1437c-1"
                },
                legacy={
                    "manifest": {
                        "path": (
                            ".axiom/encoding-manifests/us/statutes/42/1437c–1.json"
                        ),
                        "sha256": "a" * 64,
                    },
                    "files": [
                        {
                            "path": "us/statutes/42/1437c–1.yaml",
                            "sha256": source_sha256,
                        }
                    ],
                },
                replacement={"rewrites": [], "scheduled_dependents": []},
                allow_pending_scheduled=True,
            )
            == []
        )

    def test_installed_wrapper_disables_partial_clone_lazy_fetch(self, tmp_path):
        git = shutil.which("git")
        if git is None:
            pytest.skip("Git is required")
        destination = tmp_path / "destination"
        destination.mkdir()
        wrapper = provisioner._install_trusted_git_wrapper(
            destination,
            Path(sys.executable).resolve(),
            provisioner._resolve_trusted_git(Path(git).resolve()),
        )
        source = tmp_path / "source"
        subprocess.run([git, "init", "--quiet", str(source)], check=True)
        (source / "a.txt").write_text("promised blob\n")
        subprocess.run([git, "-C", str(source), "add", "a.txt"], check=True)
        subprocess.run(
            [
                git,
                "-c",
                "user.name=Axiom test",
                "-c",
                "user.email=test@axiom.invalid",
                "-C",
                str(source),
                "commit",
                "--quiet",
                "-m",
                "fixture",
            ],
            check=True,
        )
        object_id = subprocess.check_output(
            [git, "-C", str(source), "rev-parse", "HEAD:a.txt"], text=True
        ).strip()
        subprocess.run(
            [git, "-C", str(source), "config", "uploadpack.allowFilter", "true"],
            check=True,
        )
        partial = tmp_path / "partial"
        subprocess.run(
            [
                git,
                "-c",
                "protocol.file.allow=always",
                "clone",
                "--quiet",
                "--filter=blob:none",
                "--no-checkout",
                source.as_uri(),
                str(partial),
            ],
            check=True,
        )
        no_lazy_environment = {**os.environ, "GIT_NO_LAZY_FETCH": "1"}
        missing = subprocess.run(
            [git, "-C", str(partial), "cat-file", "-e", object_id],
            check=False,
            capture_output=True,
            env=no_lazy_environment,
        )
        if missing.returncode == 0:
            pytest.skip("Git did not create a blobless partial clone")
        pack_root = partial / ".git" / "objects" / "pack"
        packs_before = {path.name for path in pack_root.iterdir()}
        wrapped = subprocess.run(
            [str(wrapper), "-C", str(partial), "cat-file", "blob", object_id],
            check=False,
            capture_output=True,
            env={"HOME": str(tmp_path), "PATH": str(destination)},
        )
        assert wrapped.returncode != 0
        assert {path.name for path in pack_root.iterdir()} == packs_before
        assert (
            subprocess.run(
                [git, "-C", str(partial), "cat-file", "-e", object_id],
                check=False,
                capture_output=True,
                env=no_lazy_environment,
            ).returncode
            != 0
        )

    def test_installed_wrapper_disables_replacement_objects(self, tmp_path):
        git = shutil.which("git")
        if git is None:
            pytest.skip("Git is required")
        destination = tmp_path / "destination"
        destination.mkdir()
        wrapper = provisioner._install_trusted_git_wrapper(
            destination,
            Path(sys.executable).resolve(),
            provisioner._resolve_trusted_git(Path(git).resolve()),
        )
        repository = tmp_path / "repository"
        subprocess.run([git, "init", "--quiet", str(repository)], check=True)
        original = b"original blob\n"
        replacement = b"replacement blob\n"
        (repository / "original.txt").write_bytes(original)
        (repository / "replacement.txt").write_bytes(replacement)
        original_id = subprocess.check_output(
            [git, "-C", str(repository), "hash-object", "-w", "original.txt"],
            text=True,
        ).strip()
        replacement_id = subprocess.check_output(
            [git, "-C", str(repository), "hash-object", "-w", "replacement.txt"],
            text=True,
        ).strip()
        subprocess.run(
            [git, "-C", str(repository), "replace", original_id, replacement_id],
            check=True,
        )
        assert (
            subprocess.check_output(
                [git, "-C", str(repository), "cat-file", "blob", original_id]
            )
            == replacement
        )
        wrapped = subprocess.run(
            [str(wrapper), "-C", str(repository), "cat-file", "blob", original_id],
            check=True,
            capture_output=True,
            env={"HOME": str(tmp_path), "PATH": str(destination)},
        )
        assert wrapped.stdout == original

    def test_installed_wrapper_disables_log_signature_verification(self, tmp_path):
        git = shutil.which("git")
        if git is None:
            pytest.skip("Git is required")
        destination = tmp_path / "destination"
        destination.mkdir()
        wrapper = provisioner._install_trusted_git_wrapper(
            destination,
            Path(sys.executable).resolve(),
            provisioner._resolve_trusted_git(Path(git).resolve()),
        )
        repository = tmp_path / "repository"
        subprocess.run([git, "init", "--quiet", str(repository)], check=True)
        (repository / "pyproject.toml").write_text('[project]\nname = "fixture"\n')
        subprocess.run(
            [git, "-C", str(repository), "add", "pyproject.toml"], check=True
        )
        subprocess.run(
            [
                git,
                "-c",
                "user.name=Axiom test",
                "-c",
                "user.email=test@axiom.invalid",
                "-C",
                str(repository),
                "commit",
                "--quiet",
                "-m",
                "fixture",
            ],
            check=True,
        )
        raw_commit = subprocess.check_output(
            [git, "-C", str(repository), "cat-file", "commit", "HEAD"]
        )
        headers, message = raw_commit.split(b"\n\n", 1)
        signed_commit = (
            headers
            + b"\ngpgsig -----BEGIN PGP SIGNATURE-----\n"
            + b" fake\n -----END PGP SIGNATURE-----\n\n"
            + message
        )
        signed_id = (
            subprocess.check_output(
                [
                    git,
                    "-C",
                    str(repository),
                    "hash-object",
                    "-t",
                    "commit",
                    "-w",
                    "--stdin",
                ],
                input=signed_commit,
                text=False,
            )
            .decode()
            .strip()
        )
        subprocess.run(
            [git, "-C", str(repository), "update-ref", "HEAD", signed_id], check=True
        )
        marker = tmp_path / "verifier-executed"
        helper = tmp_path / "fake-gpg"
        helper.write_text(
            f"#!{Path(sys.executable).resolve()}\n"
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).touch()\n"
            "raise SystemExit(1)\n"
        )
        helper.chmod(0o755)
        subprocess.run(
            [git, "-C", str(repository), "config", "log.showSignature", "true"],
            check=True,
        )
        subprocess.run(
            [git, "-C", str(repository), "config", "gpg.program", str(helper)],
            check=True,
        )
        log_arguments = [
            "log",
            "--format=%H",
            "--",
            "pyproject.toml",
            "src/axiom_encode/__init__.py",
            "uv.lock",
        ]
        subprocess.run(
            [git, "-C", str(repository), *log_arguments],
            check=False,
            capture_output=True,
        )
        assert marker.exists()
        marker.unlink()
        subprocess.run(
            [str(wrapper), "-C", str(repository), *log_arguments],
            check=True,
            capture_output=True,
            env={"HOME": str(tmp_path), "PATH": str(destination)},
        )
        assert not marker.exists()


class TestIsElf:
    def test_elf_magic(self, tmp_path):
        elf = tmp_path / "obj"
        elf.write_bytes(b"\x7fELF" + b"\x00" * 60)
        assert provisioner._is_elf(elf)

    def test_non_elf(self, tmp_path):
        script = tmp_path / "script"
        script.write_bytes(b"#!/bin/sh\n")
        assert not provisioner._is_elf(script)


class _FakePatchelf:
    """subprocess.run stand-in for a single object under test.

    Constructed with the object's current run path (a string, or None to
    simulate a --print-rpath returncode!=0 read failure). --set-rpath
    records and updates the object's current run path; a subsequent
    --print-rpath then returns that new value.
    """

    def __init__(self, rpath):
        # single object under test: sequence of run-path states over time
        self._current = rpath
        self.set_calls = []

    def __call__(self, cmd, *args, **kwargs):
        if "--print-rpath" in cmd:
            if self._current is None:
                return subprocess.CompletedProcess(cmd, 1, "", "no rpath")
            return subprocess.CompletedProcess(cmd, 0, self._current + "\n", "")
        if "--set-rpath" in cmd:
            value = cmd[cmd.index("--set-rpath") + 1]
            self.set_calls.append(value)
            self._current = value
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected patchelf call: {cmd}")


class TestRelocateElfRpaths:
    def _runtime_with_elf(self, tmp_path):
        runtime = (tmp_path / "python").resolve()
        (runtime / "lib").mkdir(parents=True)
        obj = runtime / "lib" / "libpython.so"
        obj.write_bytes(b"\x7fELF" + b"\x00" * 60)  # magic only; patchelf is stubbed
        return runtime, obj

    def test_escaping_runpath_is_repinned(self, tmp_path, monkeypatch):
        runtime, _ = self._runtime_with_elf(tmp_path)
        fake = _FakePatchelf("/opt/hostedtoolcache/Python/3.14/x64/lib")
        monkeypatch.setattr(provisioner.subprocess, "run", fake)
        assert provisioner._relocate_elf_rpaths(runtime, "patchelf") == 1
        assert fake.set_calls == [str(runtime / "lib")]

    def test_origin_only_runpath_untouched(self, tmp_path, monkeypatch):
        runtime, _ = self._runtime_with_elf(tmp_path)
        fake = _FakePatchelf("$ORIGIN")
        monkeypatch.setattr(provisioner.subprocess, "run", fake)
        assert provisioner._relocate_elf_rpaths(runtime, "patchelf") == 0
        assert fake.set_calls == []

    def test_unreadable_rpath_is_skipped(self, tmp_path, monkeypatch):
        # a .o-style object patchelf can't read a run path from → skipped, no set
        runtime, _ = self._runtime_with_elf(tmp_path)
        fake = _FakePatchelf(None)
        monkeypatch.setattr(provisioner.subprocess, "run", fake)
        assert provisioner._relocate_elf_rpaths(runtime, "patchelf") == 0
        assert fake.set_calls == []

    def test_rewrite_that_does_not_take_is_fatal(self, tmp_path, monkeypatch):
        """If --set-rpath silently failed to change the object, the confirming
        re-read must catch that an escaping component survives."""
        runtime, _ = self._runtime_with_elf(tmp_path)
        fake = _FakePatchelf("/opt/toolcache/lib")

        def stubborn(cmd, *a, **k):
            if "--set-rpath" in cmd:  # pretend the write was a no-op
                return subprocess.CompletedProcess(cmd, 0, "", "")
            return fake(cmd, *a, **k)

        monkeypatch.setattr(provisioner.subprocess, "run", stubborn)
        with pytest.raises(SystemExit, match="still escapes the runtime"):
            provisioner._relocate_elf_rpaths(runtime, "patchelf")

    # The real-patchelf rewrite round-trip is covered end-to-end on hosted Ubuntu
    # by provision-selftest.yml, which relocates a genuine escaping RUNPATH in the
    # toolchain libpython and confirms via /proc/self/maps that libpython loads
    # from inside the runtime. patchelf edits ELF sections that a byte-magic stub
    # can't model, so the rewrite itself is not unit-tested with a synthetic file.


class TestPrintRpath:
    def test_strips_only_trailing_newline(self, monkeypatch):
        # a leading space is part of the (malformed) value and must survive
        monkeypatch.setattr(
            provisioner.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 0, " /x:$ORIGIN\n", ""),
        )
        assert provisioner._print_rpath("patchelf", Path("obj")) == " /x:$ORIGIN"

    def test_nonzero_return_is_none(self, monkeypatch):
        monkeypatch.setattr(
            provisioner.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "err"),
        )
        assert provisioner._print_rpath("patchelf", Path("obj")) is None


class TestAssertSelfContained:
    """The empirical probe's fail-closed and shebang-safety guards. The probe
    shells out to the interpreter, so inject the maps report by stubbing
    subprocess.run; sys.platform is forced to linux to reach the maps checks."""

    def _run_with(self, monkeypatch, tmp_path, maps):
        runtime = (tmp_path / "python").resolve()
        (runtime / "bin").mkdir(parents=True)
        interpreter = runtime / "bin" / "python3"
        interpreter.write_text("x")
        source = (tmp_path / "src").resolve()
        source.mkdir()
        monkeypatch.setattr(provisioner.sys, "platform", "linux")
        # Stub the probe subprocess and its parse so the base_prefix/version
        # checks pass and we reach the maps checks with the injected report.
        monkeypatch.setattr(
            provisioner.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "{}", ""),
        )
        monkeypatch.setattr(
            provisioner.json,
            "loads",
            lambda _s: {
                "base_prefix": str(runtime),
                "version": list(sys.version_info[:2]),
                "maps": maps,
            },
        )
        provisioner._assert_self_contained(runtime, source, interpreter)
        return runtime

    def _interp(self, tmp_path):
        return str((tmp_path / "python").resolve() / "bin" / "python3")

    def test_empty_maps_fails_closed(self, tmp_path, monkeypatch):
        with pytest.raises(SystemExit, match="no /proc/self/maps entries"):
            self._run_with(monkeypatch, tmp_path, [])

    def test_interpreter_not_mapped_fails_closed(self, tmp_path, monkeypatch):
        # maps present but the provisioned interpreter binary is absent → the
        # probe did not actually run the runtime interpreter (vacuous pass).
        maps = ["/usr/lib/x86_64-linux-gnu/libc.so.6", "/lib64/ld-linux-x86-64.so.2"]
        with pytest.raises(SystemExit, match="did not map the provisioned interpreter"):
            self._run_with(monkeypatch, tmp_path, maps)

    def test_static_build_without_libpython_ok(self, tmp_path, monkeypatch):
        # A static-libpython interpreter (python-build-standalone) maps no
        # libpython.so; that is fine as long as its own binary is mapped.
        maps = [self._interp(tmp_path), "/usr/lib/x86_64-linux-gnu/libc.so.6"]
        self._run_with(monkeypatch, tmp_path, maps)  # no raise

    def test_system_libc_allowed_shared_libpython_pinned(self, tmp_path, monkeypatch):
        runtime = (tmp_path / "python").resolve()
        maps = [
            self._interp(tmp_path),
            str(runtime / "lib" / "libpython3.so"),
            "/usr/lib/x86_64-linux-gnu/libc.so.6",
            "/lib64/ld-linux-x86-64.so.2",
        ]
        self._run_with(monkeypatch, tmp_path, maps)  # no raise

    def test_source_prefix_mapping_rejected(self, tmp_path, monkeypatch):
        source = (tmp_path / "src").resolve()
        maps = [self._interp(tmp_path), str(source / "lib" / "libpython3.so")]
        with pytest.raises(SystemExit, match="maps code from the source"):
            self._run_with(monkeypatch, tmp_path, maps)

    def test_stray_libpython_rejected(self, tmp_path, monkeypatch):
        maps = [self._interp(tmp_path), "/opt/elsewhere/lib/libpython3.so"]
        with pytest.raises(SystemExit, match="libpython mapped outside the runtime"):
            self._run_with(monkeypatch, tmp_path, maps)

    def test_whitespace_destination_is_refused(self, tmp_path):
        runtime = tmp_path / "python"
        runtime.mkdir()
        bad_interp = tmp_path / "a b" / "python3"
        with pytest.raises(SystemExit, match="not shebang-safe"):
            provisioner._assert_self_contained(runtime, tmp_path, bad_interp)


@pytest.mark.skipif(shutil.which("git") is None, reason="git required")
@pytest.mark.skipif(shutil.which("git") is None, reason="git required")
class TestEncoderSnapshotProvisioning:
    """Cover the encode#1147 hardening: the runtime encoder is BUILT from a
    verified git-archive of the attested commit (not an arbitrary
    --site-packages), root git never scans a worktree, and the attestation is
    published safely and records the snapshot's declared version."""

    GIT = Path(shutil.which("git") or "git")
    OFFICIAL = "github.com/TheAxiomFoundation/axiom-encode"
    CODEX = {"version": "0.test", "sha256": "c" * 64}

    def _git(self, path: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.GIT), "-C", str(path), "-c", "safe.directory=*", *args],
            check=True,
            capture_output=True,
            text=True,
        )

    def _write_version_files(
        self, repo: Path, version: str, cli_body: str = "MARKER = 'committed'\n"
    ) -> None:
        (repo / "pyproject.toml").write_text(
            f'[project]\nname = "axiom-encode"\nversion = "{version}"\n'
        )
        (repo / "uv.lock").write_text(
            f'[[package]]\nname = "axiom-encode"\nversion = "{version}"\n'
            'source = { editable = "." }\n'
        )
        pkg = repo / "src" / "axiom_encode"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").write_text(f'__version__ = "{version}"\n')
        (pkg / "cli.py").write_text(cli_body)

    def _init_encoder_repo(
        self,
        path: Path,
        *,
        origin: str = "https://github.com/TheAxiomFoundation/axiom-encode.git",
        version: str = "9.9.9",
    ) -> str:
        path.mkdir(parents=True)
        self._git(path, "init", "-q")
        self._git(path, "config", "user.email", "t@example.com")
        self._git(path, "config", "user.name", "Test")
        self._git(path, "config", "commit.gpgsign", "false")
        self._git(path, "remote", "add", "origin", origin)
        # Two commits so the bump to `version` is a genuine, detectable increase.
        self._write_version_files(path, "0.0.1")
        self._git(path, "add", ".")
        self._git(path, "commit", "-q", "-m", "init")
        self._write_version_files(path, version)
        self._git(path, "add", ".")
        self._git(path, "commit", "-q", "-m", f"Bump to {version}")
        return self._git(path, "rev-parse", "HEAD").stdout.strip()

    def _export(self, repo: Path, commit: str, staging: Path):
        return provisioner._verify_and_export_encoder_snapshot(
            self.GIT, repo, commit, self.OFFICIAL, staging
        )

    def test_exports_declared_version_and_package(self, tmp_path):
        repo = tmp_path / "axiom-encode"
        head = self._init_encoder_repo(repo)
        staging = tmp_path / "staging"
        staging.mkdir()
        version, package = self._export(repo, head, staging)
        assert version == "9.9.9"
        assert (package / "__init__.py").read_text() == '__version__ = "9.9.9"\n'
        assert (package / "cli.py").read_text() == "MARKER = 'committed'\n"

    def test_export_uses_committed_tree_not_dirty_worktree(self, tmp_path):
        # Attest-A-stage-B: a dirty/hostile worktree must NOT reach the runtime;
        # the archive is taken from the committed tree at the pinned commit.
        repo = tmp_path / "axiom-encode"
        head = self._init_encoder_repo(repo)
        (repo / "src" / "axiom_encode" / "cli.py").write_text("MARKER = 'HOSTILE'\n")
        staging = tmp_path / "staging"
        staging.mkdir()
        _version, package = self._export(repo, head, staging)
        assert (package / "cli.py").read_text() == "MARKER = 'committed'\n"

    def test_git_no_replace_objects_defeats_replace_ref(self, tmp_path):
        # A replace ref that swaps the real cli.py blob for a hostile one must
        # NOT poison the export (GIT_NO_REPLACE_OBJECTS in the hardened env).
        repo = tmp_path / "axiom-encode"
        head = self._init_encoder_repo(repo)
        # Create a hostile blob and a replace mapping real->hostile for cli.py.
        real_blob = self._git(
            repo, "rev-parse", f"{head}:src/axiom_encode/cli.py"
        ).stdout.strip()
        hostile_file = tmp_path / "hostile.txt"
        hostile_file.write_text("MARKER = 'REPLACED'\n")
        hostile_blob = subprocess.run(
            [str(self.GIT), "-C", str(repo), "hash-object", "-w", str(hostile_file)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self._git(repo, "replace", real_blob, hostile_blob)
        # With replace refs honored, `show` would return the hostile content; the
        # hardened export must ignore replacements.
        staging = tmp_path / "staging"
        staging.mkdir()
        _version, package = self._export(repo, head, staging)
        assert (package / "cli.py").read_text() == "MARKER = 'committed'\n"

    def test_hardened_git_does_not_run_repo_drivers(self, tmp_path):
        # Hostile core.fsmonitor / clean / smudge drivers in the repo config must
        # never execute while the provisioner reads/exports the repo. They are
        # planted AFTER the clean commits, so only a provisioner read could fire
        # them; the bare, worktree-free git operations never do.
        repo = tmp_path / "axiom-encode"
        head = self._init_encoder_repo(repo)
        marker = tmp_path / "driver-ran"
        helper = tmp_path / "helper.sh"
        helper.write_text(f"#!/bin/sh\ntouch {marker}\ncat\n")
        helper.chmod(0o755)
        for key in (
            "core.fsmonitor",
            "filter.hostile.clean",
            "filter.hostile.smudge",
            "diff.hostile.textconv",
        ):
            self._git(repo, "config", key, str(helper))
        staging = tmp_path / "staging"
        staging.mkdir()
        self._export(repo, head, staging)
        assert not marker.exists()

    def test_export_subst_gitattribute_is_not_expanded(self, tmp_path):
        # A committed .gitattributes `export-subst` + a $Format:%s$ template would,
        # under `git archive`, expand the COMMIT SUBJECT into the module (arbitrary
        # code at encoder import) while cat-file/version binding see the benign
        # template. The raw ls-tree/cat-file export must emit the LITERAL template.
        repo = tmp_path / "axiom-encode"
        self._init_encoder_repo(repo)
        pkg = repo / "src" / "axiom_encode"
        (pkg / ".gitattributes").write_text("injected.py export-subst\n")
        (pkg / "injected.py").write_text('INJECTED = "$Format:%s$"\n')
        self._write_version_files(repo, "9.9.10")
        self._git(repo, "add", ".")
        self._git(
            repo,
            "commit",
            "-q",
            "-m",
            'x"; import os; os.system("touch PWNED")  #',
        )
        head = self._git(repo, "rev-parse", "HEAD").stdout.strip()
        staging = tmp_path / "staging"
        staging.mkdir()
        _version, package = self._export(repo, head, staging)
        injected = (package / "injected.py").read_text()
        assert injected == 'INJECTED = "$Format:%s$"\n'
        assert "os.system" not in injected
        assert "PWNED" not in injected

    def test_export_ignore_gitattribute_does_not_drop_module(self, tmp_path):
        # A committed .gitattributes `export-ignore` would drop a module from a
        # `git archive` (e.g. silently removing a guard). Every tracked blob must
        # still be exported.
        repo = tmp_path / "axiom-encode"
        self._init_encoder_repo(repo)
        pkg = repo / "src" / "axiom_encode"
        (pkg / ".gitattributes").write_text("secret_guard.py export-ignore\n")
        (pkg / "secret_guard.py").write_text("GUARD = True\n")
        self._write_version_files(repo, "9.9.10")
        self._git(repo, "add", ".")
        self._git(repo, "commit", "-q", "-m", "Bump to 9.9.10")
        head = self._git(repo, "rev-parse", "HEAD").stdout.strip()
        staging = tmp_path / "staging"
        staging.mkdir()
        _version, package = self._export(repo, head, staging)
        assert (package / "secret_guard.py").read_text() == "GUARD = True\n"

    def test_refuses_committed_symlink(self, tmp_path):
        repo = tmp_path / "axiom-encode"
        self._init_encoder_repo(repo)
        (repo / "src" / "axiom_encode" / "evil_link").symlink_to("/etc/passwd")
        self._write_version_files(repo, "9.9.10")
        self._git(repo, "add", ".")
        self._git(repo, "commit", "-q", "-m", "Bump to 9.9.10")
        head = self._git(repo, "rev-parse", "HEAD").stdout.strip()
        staging = tmp_path / "staging"
        staging.mkdir()
        with pytest.raises(SystemExit, match="symlink"):
            self._export(repo, head, staging)

    def test_refuses_committed_gitlink(self, tmp_path):
        repo = tmp_path / "axiom-encode"
        self._init_encoder_repo(repo)
        self._write_version_files(repo, "9.9.10")
        self._git(repo, "add", ".")
        # Stage a gitlink (submodule pointer) with no real submodule on disk.
        self._git(
            repo,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{'a' * 40},src/axiom_encode/vendored",
        )
        self._git(repo, "commit", "-q", "-m", "Bump to 9.9.10")
        head = self._git(repo, "rev-parse", "HEAD").stdout.strip()
        staging = tmp_path / "staging"
        staging.mkdir()
        with pytest.raises(SystemExit, match="gitlink|submodule"):
            self._export(repo, head, staging)

    def test_rejects_head_mismatch(self, tmp_path):
        repo = tmp_path / "axiom-encode"
        self._init_encoder_repo(repo)
        initial = self._git(repo, "rev-parse", "HEAD~1").stdout.strip()
        staging = tmp_path / "staging"
        staging.mkdir()
        with pytest.raises(SystemExit, match="HEAD"):
            self._export(repo, initial, staging)

    def test_rejects_absent_commit(self, tmp_path):
        repo = tmp_path / "axiom-encode"
        self._init_encoder_repo(repo)
        staging = tmp_path / "staging"
        staging.mkdir()
        with pytest.raises(SystemExit, match="not a commit object"):
            self._export(repo, "a" * 40, staging)

    def test_rejects_origin_mismatch(self, tmp_path):
        repo = tmp_path / "axiom-encode"
        head = self._init_encoder_repo(
            repo, origin="https://github.com/someone-else/axiom-encode.git"
        )
        staging = tmp_path / "staging"
        staging.mkdir()
        with pytest.raises(SystemExit, match="origin"):
            self._export(repo, head, staging)

    def test_rejects_inconsistent_version_metadata(self, tmp_path):
        repo = tmp_path / "axiom-encode"
        self._init_encoder_repo(repo)
        # __init__ disagrees with pyproject/uv.lock at HEAD.
        (repo / "src" / "axiom_encode" / "__init__.py").write_text(
            '__version__ = "0.0.0"\n'
        )
        self._git(repo, "add", ".")
        self._git(repo, "commit", "-q", "-m", "skew", "--no-verify")
        head = self._git(repo, "rev-parse", "HEAD").stdout.strip()
        staging = tmp_path / "staging"
        staging.mkdir()
        with pytest.raises(SystemExit, match="inconsistent"):
            self._export(repo, head, staging)

    def test_rejects_unversioned_encoder_change(self, tmp_path):
        repo = tmp_path / "axiom-encode"
        self._init_encoder_repo(repo)
        # Change encoder code AFTER the bump without bumping the version.
        (repo / "src" / "axiom_encode" / "cli.py").write_text("MARKER = 'drift'\n")
        self._git(repo, "add", ".")
        self._git(repo, "commit", "-q", "-m", "unversioned change", "--no-verify")
        head = self._git(repo, "rev-parse", "HEAD").stdout.strip()
        staging = tmp_path / "staging"
        staging.mkdir()
        with pytest.raises(SystemExit, match="after the latest version bump"):
            self._export(repo, head, staging)

    def test_allows_operational_reviewed_head_record_without_version_bump(
        self, tmp_path
    ):
        repo = tmp_path / "axiom-encode"
        self._init_encoder_repo(repo)
        record = repo / "data" / "reviewed-rulespec-refs" / "us" / f"{'a' * 40}.json"
        record.parent.mkdir(parents=True)
        record.write_text('{"operational":"policy"}\n')
        self._git(repo, "add", ".")
        self._git(repo, "commit", "-q", "-m", "admit reviewed head", "--no-verify")
        head = self._git(repo, "rev-parse", "HEAD").stdout.strip()
        staging = tmp_path / "staging"
        staging.mkdir()

        self._export(repo, head, staging)

        assert not (staging / "data").exists()

    def test_rejects_shallow_history(self, tmp_path):
        repo = tmp_path / "axiom-encode"
        self._init_encoder_repo(repo)
        shallow = tmp_path / "shallow"
        subprocess.run(
            [
                str(self.GIT),
                "-c",
                "safe.directory=*",
                "-c",
                "protocol.file.allow=always",
                "clone",
                "--depth",
                "1",
                "--no-local",
                f"file://{repo}",
                str(shallow),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self._git(shallow, "remote", "set-url", "origin", self.OFFICIAL_URL)
        head = self._git(shallow, "rev-parse", "HEAD").stdout.strip()
        staging = tmp_path / "staging"
        staging.mkdir()
        with pytest.raises(SystemExit, match="shallow"):
            self._export(shallow, head, staging)

    OFFICIAL_URL = "https://github.com/TheAxiomFoundation/axiom-encode.git"

    def test_own_git_dir_rejects_gitfile_pointer(self, tmp_path):
        repo = tmp_path / "axiom-encode"
        repo.mkdir()
        (repo / ".git").symlink_to(tmp_path / "elsewhere")
        with pytest.raises(SystemExit, match="not a plain directory"):
            provisioner._own_encoder_git_directory(repo, tmp_path / "staging-dir")

    def test_overlay_replaces_site_packages_axiom_encode(self, tmp_path):
        runtime = tmp_path / "python"
        site = (
            runtime
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        (site / "axiom_encode").mkdir(parents=True)
        (site / "axiom_encode" / "cli.py").write_text("MARKER = 'PIP_STAGED'\n")
        (site / "axiom_encode" / "stale.pth").write_text("import evil\n")
        package = tmp_path / "snapshot" / "axiom_encode"
        package.mkdir(parents=True)
        (package / "cli.py").write_text("MARKER = 'ATTESTED'\n")
        provisioner._overlay_encoder_package(runtime, package)
        assert (site / "axiom_encode" / "cli.py").read_text() == "MARKER = 'ATTESTED'\n"
        assert not (site / "axiom_encode" / "stale.pth").exists()

    def test_publishes_readonly_attestation_with_version(self, tmp_path):
        runtime = tmp_path / "python"
        runtime.mkdir()
        out = provisioner._publish_runtime_attestation(
            runtime, self.OFFICIAL, "a" * 40, "9.9.9", "b" * 64, self.CODEX
        )
        assert out == runtime / "runtime-attestation.json"
        assert stat.S_IMODE(out.stat().st_mode) == 0o444
        payload = json.loads(out.read_text())
        assert payload["schema"] == "axiom-encode/trusted-runtime-attestation/v1"
        assert payload["axiom_encode"] == {
            "origin_repository": self.OFFICIAL,
            "commit": "a" * 40,
            "version": "9.9.9",
            "package_tree_sha256": "b" * 64,
        }
        assert payload["codex_cli"] == self.CODEX
        assert (
            __import__("datetime")
            .datetime.fromisoformat(payload["provisioned_at"])
            .tzinfo
            is not None
        )

    def test_publish_refuses_preplaced_inode(self, tmp_path):
        runtime = tmp_path / "python"
        runtime.mkdir()
        (runtime / "runtime-attestation.json").write_text("{}\n")
        with pytest.raises(SystemExit, match="already exists"):
            provisioner._publish_runtime_attestation(
                runtime, self.OFFICIAL, "a" * 40, "9.9.9", "b" * 64, self.CODEX
            )

    def test_publish_refuses_preplaced_symlink(self, tmp_path):
        runtime = tmp_path / "python"
        runtime.mkdir()
        (runtime / "runtime-attestation.json").symlink_to(tmp_path / "target")
        with pytest.raises(SystemExit, match="already exists|cannot create"):
            provisioner._publish_runtime_attestation(
                runtime, self.OFFICIAL, "a" * 40, "9.9.9", "b" * 64, self.CODEX
            )

    def test_package_tree_sha256_matches_deterministic_tree_identity(self, tmp_path):
        # The provisioner's mirror MUST equal the evals primitive the CLI uses at
        # apply time, or the byte-binding check would false-reject every runtime.
        from axiom_encode.harness.evals import _deterministic_tree_identity

        package = tmp_path / "axiom_encode"
        (package / "sub").mkdir(parents=True)
        (package / "__init__.py").write_text('__version__ = "9.9.9"\n')
        (package / "cli.py").write_text("MARKER = 'x'\n")
        (package / "sub" / "mod.py").write_text("Y = 1\n")
        # A __pycache__ artifact must be excluded identically on both sides.
        (package / "__pycache__").mkdir()
        (package / "__pycache__" / "cli.pyc").write_bytes(b"\x00\x01")
        mirror = provisioner._deterministic_package_tree_sha256(package)
        canonical = _deterministic_tree_identity(
            package, excluded_directory_names=frozenset({"__pycache__"})
        )
        assert mirror == canonical["tree_sha256"]

    def test_github_repository_identity_normalizes_remotes(self):
        for url in (
            "https://github.com/TheAxiomFoundation/axiom-encode.git",
            "https://github.com/TheAxiomFoundation/axiom-encode",
            "git@github.com:TheAxiomFoundation/axiom-encode.git",
            "ssh://git@github.com/TheAxiomFoundation/axiom-encode.git",
        ):
            assert provisioner._github_repository_identity(url) == self.OFFICIAL
        assert (
            provisioner._github_repository_identity("https://example.com/x/y") is None
        )
