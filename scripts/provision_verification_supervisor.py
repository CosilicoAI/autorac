"""Build a protected verification-supervisor execution tree.

The load-bearing guarantee is EMPIRICAL: the provisioned launcher interpreter is
run under a minimal environment and ``/proc/self/maps`` is required to show it
maps its OWN binary from the root-owned runtime, maps NO file from the
(user-writable) source prefix, and pins any mapped ``libpython`` to the runtime —
even under a hostile ``LD_LIBRARY_PATH``. This bounds the launcher interpreter's
own load-time closure; it does NOT assert that every mapping is inside the runtime
(the root-owned system loader and libc legitimately map from ``/lib``, ``/usr/lib``).
The check observes exactly what the loader does (interpreter, every mapped library,
audit hooks, ``$ORIGIN`` expansion) with no dependence on hand-parsed ELF metadata.
The static passes below make that outcome reachable and durable:

* Source prefix: a system/FHS prefix, a prefix outside ``--require-prefix-under``,
  one exceeding the file cap, or one whose tree escapes itself via a symlink is
  refused BEFORE any copy.
* Startup code: every ``.pth``/``._pth``/``.egg-link``/``pyvenv.cfg``/
  ``pybuilddir.txt``/``__editable__*`` path-configuration carrier, the canonical
  stdlib ``pythonXY[t].zip``, and every importable ``sitecustomize``/
  ``usercustomize`` form (module, package dir, ``.pyc``, extension module) is
  purged AFTER the staged site-packages swap, so the launcher's ``-I``
  (site-enabled) start executes none of them.
* Run paths: every copied ELF whose ``DT_RPATH``/``DT_RUNPATH`` (read via
  ``patchelf --print-rpath``) escapes the runtime after per-object ``$ORIGIN``
  expansion — requiring both lexical and symlink-resolved containment — is
  repinned to ``<runtime>/lib`` as a forced ``DT_RPATH`` (which ``LD_LIBRARY_PATH``
  cannot outrank), then re-read to confirm nothing still escapes. This neutralizes
  the toolchain's absolute build-prefix RUNPATH so ``libpython`` loads from the
  runtime, and is the pass the maps check then verifies for the launcher.

Trust boundary and residual: the inputs are the GitHub-provided toolcache stdlib
and pinned pip dependencies, and the finished tree is chowned root and stripped
of group/other write by the caller (the supervisor re-verifies that ownership).
A pinned dependency carrying hostile ELF metadata (an absolute ``DT_NEEDED``, a
``DT_AUDIT`` hook, a decoy section) is NOT statically audited here — such a
dependency already executes its own code inside the signer, so metadata-level
auditing adds no protection beyond the dependency-provenance trust already
extended, and the run-path repin plus root-ownership bound the writable-path
vectors that do matter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Prefixes that are never a self-contained CPython runtime. Copying one of
# these (e.g. /usr, whose sys.base_prefix the system interpreter reports)
# means the caller resolved the wrong interpreter; fail before the copy.
FORBIDDEN_PREFIXES = (
    "/",
    "/usr",
    "/usr/local",
    "/opt",
    "/etc",
    "/bin",
    "/sbin",
    "/lib",
)

# A real CPython prefix (toolcache or standalone build) is ~20k files. Anything
# far past that is a wrong-prefix copy about to fill the disk. Overridable for
# legitimately larger local prefixes (--max-runtime-files).
DEFAULT_MAX_RUNTIME_FILES = 100_000

# Startup-code carriers that must not exist anywhere in the provisioned
# runtime, INCLUDING the staged site-packages (the launcher runs -I, which
# still processes site-packages .pth lines and imports sitecustomize at start).
# CPython imports sitecustomize/usercustomize by MODULE NAME, so every
# importable form must be covered — a bare `sitecustomize.py` glob leaves the
# package dir, the .pyc, and the extension-module forms executing.
_FORBIDDEN_STARTUP_GLOBS = (
    "*.pth",
    "*._pth",
    "*.egg-link",
    "pyvenv.cfg",
    "pybuilddir.txt",
    "__editable__*",
    "sitecustomize",
    "sitecustomize.*",
    "usercustomize",
    "usercustomize.*",
)


def _count_files_copy_equivalent(source: Path, limit: int) -> None:
    """Bound the copy the way copytree(symlinks=False) will actually perform it.

    copytree with symlinks=False FOLLOWS symlinks, so the count must too, and
    a link escaping the source prefix (`x -> /usr`) must refuse rather than
    count (or copy) the outside world. Diamonds (`lib64 -> lib`) are counted
    twice exactly as copytree copies them twice; only a directory whose real
    path appears among its own traversal ancestors — copytree's infinite
    recursion — refuses.
    """
    source_resolved = source.resolve(strict=True)

    def walk(directory: Path, ancestors: frozenset[Path], count: int) -> int:
        real_directory = directory.resolve(strict=True)
        if real_directory in ancestors:
            raise SystemExit(
                f"refusing to provision: symlink cycle at {directory} in {source}"
            )
        ancestors = ancestors | {real_directory}
        with os.scandir(directory) as entries:
            for entry in entries:
                entry_path = Path(entry.path)
                if entry.is_symlink():
                    target = entry_path.resolve(strict=False)
                    if not target.is_relative_to(source_resolved):
                        raise SystemExit(
                            "refusing to provision: symlink escapes the source "
                            f"prefix: {entry_path} -> {target}"
                        )
                if entry.is_dir(follow_symlinks=True):
                    count = walk(entry_path, ancestors, count)
                else:
                    count += 1
                    if count > limit:
                        raise SystemExit(
                            f"refusing to provision: {source} holds more than "
                            f"{limit} files — not a python runtime prefix"
                        )
        return count

    walk(source, frozenset(), 0)


def _assert_sane_source_prefix(
    source_runtime: Path, require_under: Path | None, max_files: int
) -> None:
    if str(source_runtime) in FORBIDDEN_PREFIXES:
        raise SystemExit(
            f"refusing to provision from {source_runtime}: not a self-contained "
            "python prefix (the invoking interpreter is a system python)"
        )
    if require_under is not None:
        resolved_parent = require_under.resolve(strict=True)
        if not source_runtime.is_relative_to(resolved_parent):
            raise SystemExit(
                f"refusing to provision: source prefix {source_runtime} is not "
                f"under required parent {resolved_parent}"
            )
    _count_files_copy_equivalent(source_runtime, max_files)


def _stage_runtime_tree(
    source_runtime: Path, runtime: Path, site_packages: Path
) -> None:
    """Copy the interpreter prefix, swap in staged site-packages, then purge.

    The purge MUST run after the site-packages swap: the staged tree is
    pip-produced content and could itself carry .pth/sitecustomize startup
    hooks that the launcher's -I mode would execute.
    """
    shutil.copytree(source_runtime, runtime, symlinks=False)
    target_site_packages = (
        runtime
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    shutil.rmtree(target_site_packages)
    shutil.copytree(site_packages.resolve(strict=True), target_site_packages)
    # CPython puts the canonical stdlib zip on sys.path unconditionally (even
    # when absent), and -I still lets `site` import modules — including
    # sitecustomize — from it. Its name/dir vary with PLATLIBDIR (lib vs lib64)
    # and the free-threaded ABI ("t" suffix, e.g. python313t.zip), so match the
    # version-prefixed zip anywhere in the copied tree; the full lib/ tree we
    # copied makes it redundant.
    zip_glob = f"python{sys.version_info.major}{sys.version_info.minor}*.zip"
    for stdlib_zip in runtime.rglob(zip_glob):
        if stdlib_zip.is_file():
            stdlib_zip.unlink()
    # Materialize every match BEFORE deleting: rglob is a lazy walker, and
    # rmtree-ing a directory it has yielded but not yet descended into would
    # make the walk fail. Collect first, then delete, guarding against a path
    # already removed by an earlier (ancestor) deletion.
    doomed = sorted(
        {
            match
            for pattern in _FORBIDDEN_STARTUP_GLOBS
            for match in runtime.rglob(pattern)
        }
    )
    for forbidden in doomed:
        if forbidden.is_symlink() or forbidden.is_file():
            forbidden.unlink()
        elif forbidden.is_dir():
            # `sitecustomize/` as a package directory is importable too.
            shutil.rmtree(forbidden)


# gABI: $ORIGIN / ${ORIGIN}; a bare $ORIGIN token ends at a non-identifier
# char, so "$ORIGIN_evil" is NOT the token and stays unexpanded (→ unsafe).
_ORIGIN_TOKEN = re.compile(r"\$(?:\{ORIGIN\}|ORIGIN(?![0-9A-Z_a-z]))")


def _is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError:
        return False


def _path_inside(path_text: str, runtime_resolved: Path) -> bool:
    """Inside iff BOTH the lexically normalized path AND the symlink-resolved
    path stay within the runtime.

    Lexical-only accepts ``/runtime/alias/../lib`` where ``alias`` is a symlink
    pointing outside (the loader follows the link before ``..``, landing out);
    resolve-only accepts ``/outside/alias`` symlinked into the runtime (the
    loader reads the literal, retargetable path). Requiring both closes each.
    """
    normalized = Path(os.path.normpath(path_text))
    if not normalized.is_relative_to(runtime_resolved):
        return False
    resolved = Path(path_text).resolve(strict=False)
    return resolved.is_relative_to(runtime_resolved)


def _origin_expand(component: str, object_dir: Path) -> str:
    return _ORIGIN_TOKEN.sub(str(object_dir), component)


def _rpath_component_inside(
    component: str, object_dir: Path, runtime_resolved: Path
) -> bool:
    """A DT_RPATH/DT_RUNPATH (or path-bearing DT_NEEDED) entry is safe only if,
    after per-object $ORIGIN expansion, it is absolute and — both lexically and
    after symlink resolution — strictly inside the runtime.

    ``object_dir`` is the ELF's directory; we provision in place at the final
    location, so $ORIGIN evaluation is exact. Empty, cwd-relative, and escaping
    entries are all unsafe.
    """
    if not component:
        return False
    expanded = _origin_expand(component, object_dir)
    if not expanded.startswith("/"):
        return False
    return _path_inside(expanded, runtime_resolved)


def _rewrite_rpath_for_object(
    rpath: str, object_dir: Path, runtime_resolved: Path, target: str
) -> tuple[str, bool]:
    """Keep safe components verbatim; collapse every unsafe one to the pinned
    lib dir. Returns (new_rpath, changed)."""
    parts: list[str] = []
    changed = False
    for component in rpath.split(":"):
        if _rpath_component_inside(component, object_dir, runtime_resolved):
            keep = component
        else:
            keep = target
            changed = True
        if keep not in parts:
            parts.append(keep)
    return ":".join(parts), changed


def _print_rpath(patchelf: str, path: Path) -> str | None:
    """Return the object's combined DT_RPATH/DT_RUNPATH, or None if patchelf
    cannot read it (a non-dynamic object such as a `.o`, or one with no run
    path). patchelf collapses RPATH and RUNPATH into a single reported value.
    """
    result = subprocess.run(
        [patchelf, "--print-rpath", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    # patchelf appends exactly one trailing newline; a leading space would be
    # part of a (malformed) run-path value, so strip only that newline.
    return result.stdout[:-1] if result.stdout.endswith("\n") else result.stdout


def _relocate_elf_rpaths(runtime: Path, patchelf: str) -> int:
    """Repin every copied ELF whose DT_RPATH/DT_RUNPATH escapes the runtime.

    The hosted-toolchain CPython is built --enable-shared with an absolute
    RUNPATH into its build prefix, so a byte-copy keeps loading libpython from
    the ORIGINAL (user-writable) location. Each escaping run-path component
    (after per-object $ORIGIN expansion, requiring both lexical and resolved
    containment) is rewritten to the pinned <runtime>/lib as a forced DT_RPATH,
    which LD_LIBRARY_PATH cannot outrank; patchelf --print-rpath then re-reads
    the object to confirm nothing still escapes.

    This is a best-effort hardening pass over trusted, pinned inputs (the
    toolcache stdlib and pinned pip deps). The load-bearing guarantee for the
    launcher interpreter is the empirical /proc/self/maps check in
    _assert_self_contained, not this static rewrite.
    """
    runtime_resolved = runtime.resolve(strict=True)
    target_rpath = str(runtime_resolved / "lib")
    rewritten = 0
    for path in sorted(runtime.rglob("*")):
        if path.is_symlink() or not path.is_file() or not _is_elf(path):
            continue
        current = _print_rpath(patchelf, path)
        if not current:
            continue
        new_rpath, changed = _rewrite_rpath_for_object(
            current, path.parent, runtime_resolved, target_rpath
        )
        if not changed:
            continue
        subprocess.run(
            [patchelf, "--force-rpath", "--set-rpath", new_rpath, str(path)],
            check=True,
        )
        after = _print_rpath(patchelf, path)
        _, still_escapes = _rewrite_rpath_for_object(
            after or "", path.parent, runtime_resolved, target_rpath
        )
        if still_escapes:
            raise SystemExit(
                f"{path}: run-path still escapes the runtime after relocation: "
                f"{after!r}"
            )
        rewritten += 1
    return rewritten


def _assert_self_contained(
    runtime: Path, source_runtime: Path, interpreter: Path
) -> None:
    """Prove the launcher's interpreter loads nothing from the source prefix and
    pins libpython to the runtime.

    Runs the interpreter exactly as the launcher will (-I, site enabled, so a
    surviving .pth would execute and show up) with an explicit minimal
    environment (no LD_LIBRARY_PATH). On Linux it then requires, from a
    non-empty /proc/self/maps: the provisioned interpreter's own binary is
    mapped (non-vacuity — proves the runtime interpreter actually ran, for both
    shared and static-libpython builds), NO file mapping under the
    (user-writable) source prefix, and every libpython mapping (if any) under
    the runtime. It deliberately does NOT require every mapping to be inside the
    runtime — the root-owned system loader and libc legitimately map from /lib,
    /usr/lib, etc. This bounds the launcher interpreter's own load-time closure.
    C-extensions the encoder imports later are OUTSIDE this probe: the run-path
    repin neutralizes their escaping DT_RPATH/DT_RUNPATH, but an absolute
    DT_NEEDED or DT_AUDIT in one of them is not covered here and remains bounded
    by the pinned-dependency provenance boundary documented below.

    The launcher is a `#!<interpreter> -I` shebang, and Linux truncates the
    shebang interpreter at whitespace, so a destination path containing
    whitespace would make the launcher select a different interpreter than this
    execve-based probe validated. Refuse such a path up front.
    """
    if any(character.isspace() for character in str(interpreter)):
        raise SystemExit(
            f"interpreter path is not shebang-safe (contains whitespace): {interpreter}"
        )
    probe_code = (
        "import json, sys\n"
        "maps = []\n"
        "try:\n"
        "    with open('/proc/self/maps') as handle:\n"
        "        for line in handle:\n"
        "            parts = line.split(None, 5)\n"
        "            if len(parts) == 6 and parts[5].startswith('/'):\n"
        "                maps.append(parts[5].strip())\n"
        "except FileNotFoundError:\n"
        "    pass\n"
        "print(json.dumps({\n"
        "    'base_prefix': sys.base_prefix,\n"
        "    'version': list(sys.version_info[:2]),\n"
        "    'maps': sorted(set(maps)),\n"
        "}))\n"
    )
    result = subprocess.run(
        [str(interpreter), "-I", "-c", probe_code],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    report = json.loads(result.stdout)
    runtime_resolved = runtime.resolve(strict=True)
    source_resolved = source_runtime.resolve(strict=True)
    if Path(report["base_prefix"]).resolve() != runtime_resolved:
        raise SystemExit(
            f"provisioned interpreter reports base_prefix {report['base_prefix']}, "
            f"expected {runtime}"
        )
    if tuple(report["version"]) != sys.version_info[:2]:
        raise SystemExit(
            f"provisioned interpreter is python {report['version']}, the "
            f"provisioning interpreter is {list(sys.version_info[:2])}"
        )
    if sys.platform == "linux":
        # Fail closed: an empty maps list means the probe observed nothing (a
        # kernel without /proc, a hardened readfile), so the containment claim
        # is unverified rather than satisfied.
        if not report["maps"]:
            raise SystemExit(
                "self-containment probe observed no /proc/self/maps entries; "
                "cannot verify the provisioned runtime"
            )
        # Non-vacuity, build-agnostic: the probe MUST have mapped this exact
        # interpreter binary from the runtime, proving it ran the provisioned
        # interpreter rather than passing on an empty/foreign map set. This holds
        # for both --enable-shared (libpython.so) and static-libpython builds
        # (e.g. python-build-standalone), where there is no shared libpython to
        # require.
        interpreter_resolved = str(interpreter.resolve(strict=True))
        if interpreter_resolved not in report["maps"]:
            raise SystemExit(
                "self-containment probe did not map the provisioned interpreter "
                f"{interpreter_resolved}; observed {sorted(report['maps'])[:5]}"
            )
        # /proc/self/maps paths are kernel-canonical; compare against resolved
        # bases so a symlinked prefix component cannot desync the containment.
        escaped = [m for m in report["maps"] if Path(m).is_relative_to(source_resolved)]
        if escaped:
            raise SystemExit(
                "provisioned interpreter still maps code from the source "
                f"prefix: {escaped}"
            )
        # Any libpython that IS mapped (shared builds) must come from the runtime;
        # a static build maps none, which is fine — libpython is baked into the
        # root-owned interpreter binary already required above.
        libpython = [m for m in report["maps"] if "libpython" in Path(m).name]
        stray = [m for m in libpython if not Path(m).is_relative_to(runtime_resolved)]
        if stray:
            raise SystemExit(f"libpython mapped outside the runtime: {stray}")


def _resolve_trusted_git(raw_git: Path) -> Path:
    """Admit one immutable, root-owned Git executable by exact path."""

    if not raw_git.is_absolute() or Path(os.path.normpath(raw_git)) != raw_git:
        raise SystemExit("trusted git path must be absolute and normalized")
    cursor = Path(raw_git.anchor)
    for part in raw_git.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise SystemExit(f"trusted git path contains a symlink: {raw_git}")
    try:
        resolved = raw_git.resolve(strict=True)
    except OSError as exc:
        raise SystemExit(f"trusted git path does not exist: {raw_git}") from exc
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode) or not os.access(resolved, os.X_OK):
        raise SystemExit(f"trusted git path is not an executable file: {resolved}")
    for component in (resolved, *resolved.parents):
        component_metadata = component.stat()
        if component_metadata.st_uid != 0:
            raise SystemExit(f"trusted git path is not root-owned: {component}")
        if component_metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise SystemExit(
                f"trusted git path is group- or other-writable: {component}"
            )
    if metadata.st_mode & (stat.S_ISUID | stat.S_ISGID):
        raise SystemExit("trusted git executable must not be setuid or setgid")
    return resolved


def _install_trusted_git_wrapper(
    tool_directory: Path, interpreter: Path, trusted_git: Path
) -> Path:
    """Install a narrowly scoped Git command broker into the isolated PATH."""

    wrapper = tool_directory / "git"
    content = (
        f"#!{interpreter} -I\n"
        "import os\n"
        "import re\n"
        "import subprocess\n"
        "import sys\n"
        f"git = {str(trusted_git)!r}\n"
        "args = sys.argv[1:]\n"
        "cursor = 0\n"
        "if args[:1] == ['-C']:\n"
        "    if len(args) < 3:\n"
        "        raise SystemExit('trusted git wrapper requires a command after -C')\n"
        "    cursor = 2\n"
        "allowed = {\n"
        "    'add', 'cat-file', 'diff', 'grep', 'log', 'ls-files', 'ls-tree',\n"
        "    'merge-base',\n"
        "    'remote', 'rev-list', 'rev-parse', 'show', 'status',\n"
        "}\n"
        "if cursor >= len(args) or args[cursor] not in allowed:\n"
        "    selected = args[cursor] if cursor < len(args) else '<missing>'\n"
        "    raise SystemExit(f'trusted git wrapper refused command: {selected}')\n"
        "command = args[cursor]\n"
        "options = args[cursor + 1:]\n"
        "def safe_ref(value):\n"
        "    return re.fullmatch(\n"
        "        r'(?:HEAD|main|origin/main|[0-9a-f]{40})(?:\\^\\{commit\\})?',\n"
        "        value,\n"
        "    ) is not None\n"
        "def safe_range(value):\n"
        "    parts = value.split('..')\n"
        "    return len(parts) == 2 and all(safe_ref(part) for part in parts)\n"
        "def safe_path(value):\n"
        "    return (\n"
        "        bool(value)\n"
        "        and not value.startswith(('-', ':', '/'))\n"
        "        and all(part not in {'', '.', '..'} for part in value.split('/'))\n"
        "    )\n"
        "def allowed_options():\n"
        "    if command == 'add':\n"
        "        repository = args[1] if args[:1] == ['-C'] else ''\n"
        "        return (\n"
        "            cursor == 2\n"
        "            and os.path.isabs(repository)\n"
        "            and os.path.normpath(repository) == repository\n"
        "            and os.path.realpath(repository) == repository\n"
        "            and os.path.isdir(repository)\n"
        "            and len(options) >= 2\n"
        "            and options[0] == '--'\n"
        "            and all(safe_path(value) for value in options[1:])\n"
        "        )\n"
        "    if command == 'cat-file':\n"
        "        return options == ['--batch'] or (\n"
        "            len(options) == 2\n"
        "            and options[0] == 'blob'\n"
        "            and re.fullmatch(r'[0-9a-f]{40}', options[1]) is not None\n"
        "        )\n"
        "    if command == 'diff':\n"
        "        if options == [\n"
        "            '--cached', '--name-only', '--no-renames', '-z',\n"
        "        ]:\n"
        "            return True\n"
        "        if (\n"
        "            options[:3] == ['--binary', 'HEAD', '--']\n"
        "            and all(safe_path(value) for value in options[3:])\n"
        "        ):\n"
        "            return True\n"
        "        changed_prefix = [\n"
        "            '--name-only', '--no-renames', '--diff-filter=ACDMRT', '-z'\n"
        "        ]\n"
        "        if options[:4] == changed_prefix:\n"
        "            return len(options[4:]) in {1, 2} and all(\n"
        "                safe_ref(value) for value in options[4:]\n"
        "            )\n"
        "        return (\n"
        "            options[:2] == ['--name-only', '--diff-filter=ACMRT']\n"
        "            and len(options) == 3\n"
        "            and safe_range(options[2])\n"
        "        )\n"
        "    if command == 'grep':\n"
        "        if options[:4] != ['-z', '-l', '-a', '-F']:\n"
        "            return False\n"
        "        option_cursor = 4\n"
        "        patterns = 0\n"
        "        while (\n"
        "            option_cursor + 1 < len(options)\n"
        "            and options[option_cursor] == '-e'\n"
        "            and safe_path(options[option_cursor + 1])\n"
        "            and patterns < 64\n"
        "        ):\n"
        "            option_cursor += 2\n"
        "            patterns += 1\n"
        "        return (\n"
        "            patterns > 0\n"
        "            and len(options) == option_cursor + 2\n"
        "            and safe_ref(options[option_cursor])\n"
        "            and options[option_cursor + 1] == '--'\n"
        "        )\n"
        "    if command == 'log':\n"
        "        return options == [\n"
        "            '--format=%H', '--', 'pyproject.toml',\n"
        "            'src/axiom_encode/__init__.py', 'uv.lock',\n"
        "        ]\n"
        "    if command == 'ls-files':\n"
        "        if options in (['-z'], ['--stage', '-z']):\n"
        "            return True\n"
        "        prefix = ['--others', '--exclude-standard', '-z']\n"
        "        if options == prefix:\n"
        "            return True\n"
        "        return (\n"
        "            options[:4] == [*prefix, '--']\n"
        "            and all(safe_path(value) for value in options[4:])\n"
        "        )\n"
        "    if command == 'ls-tree':\n"
        "        return (\n"
        "            len(options) == 5\n"
        "            and options[:2] == ['-z', '--full-tree']\n"
        "            and safe_ref(options[2])\n"
        "            and options[3] == '--'\n"
        "            and safe_path(options[4])\n"
        "        ) or (\n"
        "            len(options) == 5\n"
        "            and options[:4] == ['-r', '-l', '-z', '--full-tree']\n"
        "            and safe_ref(options[4])\n"
        "        ) or (\n"
        "            len(options) == 4\n"
        "            and options[0] == '-z'\n"
        "            and safe_ref(options[1])\n"
        "            and options[2] == '--'\n"
        "            and safe_path(options[3])\n"
        "        ) or (\n"
        "            len(options) == 3\n"
        "            and options[:2] == ['-r', '-z']\n"
        "            and safe_ref(options[2])\n"
        "        )\n"
        "    if command == 'merge-base':\n"
        "        return (\n"
        "            len(options) == 3\n"
        "            and options[0] == '--is-ancestor'\n"
        "            and all(safe_ref(value) for value in options[1:])\n"
        "        )\n"
        "    if command == 'remote':\n"
        "        return options == ['get-url', 'origin']\n"
        "    if command == 'rev-list':\n"
        "        return (\n"
        "            len(options) == 4\n"
        "            and options[:3] == ['--parents', '-n', '1']\n"
        "            and safe_ref(options[-1])\n"
        "        )\n"
        "    if command == 'rev-parse':\n"
        "        if options in (\n"
        "            ['HEAD'], ['HEAD^{tree}'], ['--is-inside-work-tree'],\n"
        "            ['--show-prefix'],\n"
        "            ['--show-toplevel'],\n"
        "        ):\n"
        "            return True\n"
        "        if (\n"
        "            len(options) == 1\n"
        "            and re.fullmatch(\n"
        "                r'[0-9a-f]{40}\\^\\{(?:commit|tree)\\}', options[0]\n"
        "            ) is not None\n"
        "        ):\n"
        "            return True\n"
        "        return (\n"
        "            len(options) == 2\n"
        "            and options[0] == '--verify'\n"
        "            and safe_ref(options[1])\n"
        "        )\n"
        "    if command == 'show':\n"
        "        if len(options) != 1 or ':' not in options[0]:\n"
        "            return False\n"
        "        ref, path = options[0].split(':', 1)\n"
        "        return safe_ref(ref) and safe_path(path)\n"
        "    if command == 'status':\n"
        "        return options in (\n"
        "            ['--porcelain'], ['--porcelain', '--untracked-files=no'],\n"
        "            [\n"
        "                '--porcelain=v1', '-z', '--untracked-files=all',\n"
        "                '--ignored=matching',\n"
        "            ],\n"
        "        )\n"
        "    return False\n"
        "if not allowed_options():\n"
        "    raise SystemExit(f'trusted git wrapper refused arguments for {command}')\n"
        "for name in tuple(os.environ):\n"
        "    if name.startswith('GIT_') or name == 'SSH_ASKPASS':\n"
        "        os.environ.pop(name, None)\n"
        "os.environ['GIT_ATTR_NOSYSTEM'] = '1'\n"
        "os.environ['GIT_CONFIG_GLOBAL'] = '/dev/null'\n"
        "os.environ['GIT_CONFIG_NOSYSTEM'] = '1'\n"
        "os.environ['GIT_NO_LAZY_FETCH'] = '1'\n"
        "os.environ['GIT_NO_REPLACE_OBJECTS'] = '1'\n"
        "os.environ['GIT_OPTIONAL_LOCKS'] = '0'\n"
        "os.environ['GIT_TERMINAL_PROMPT'] = '0'\n"
        "trusted = [\n"
        "    git, '--no-pager',\n"
        "    '-c', 'core.fsmonitor=false',\n"
        "    '-c', 'core.hooksPath=/dev/null',\n"
        "    '-c', 'credential.helper=',\n"
        "    '-c', 'core.attributesFile=/dev/null',\n"
        "    '-c', 'core.sshCommand=false',\n"
        "    '-c', 'protocol.ext.allow=never',\n"
        "    '-c', 'diff.external=/usr/bin/false',\n"
        "    '-c', 'log.showSignature=false',\n"
        "    '-c', 'submodule.recurse=false',\n"
        "]\n"
        "def worktree_sensitive():\n"
        "    if command == 'add':\n"
        "        return True\n"
        "    if command == 'status':\n"
        "        return True\n"
        "    if command != 'diff':\n"
        "        return False\n"
        "    if options[:3] == ['--binary', 'HEAD', '--']:\n"
        "        return True\n"
        "    prefix = [\n"
        "        '--name-only', '--no-renames', '--diff-filter=ACDMRT', '-z'\n"
        "    ]\n"
        "    return options[:4] == prefix and len(options) == 5\n"
        "def assert_no_worktree_filters():\n"
        "    repository = args[:cursor]\n"
        "    try:\n"
        "        staged = subprocess.run(\n"
        "            [*trusted, *repository, 'ls-files', '--stage', '-z'],\n"
        "            check=True, capture_output=True, env=os.environ,\n"
        "        ).stdout\n"
        "        for entry in staged.split(b'\\0'):\n"
        "            if not entry.startswith(b'160000 '):\n"
        "                continue\n"
        "            _metadata, _separator, path = entry.partition(b'\\t')\n"
        "            display = os.fsdecode(path)\n"
        "            raise SystemExit(\n"
        "                f'trusted git wrapper refused gitlink at {display}'\n"
        "            )\n"
        "        listed = subprocess.run(\n"
        "            [*trusted, *repository, 'ls-files', '-co',\n"
        "             '--exclude-standard', '-z'],\n"
        "            check=True, capture_output=True, env=os.environ,\n"
        "        ).stdout\n"
        "        if not listed:\n"
        "            return\n"
        "        checked = subprocess.run(\n"
        "            [*trusted, *repository, 'check-attr', '-z', '--stdin',\n"
        "             '--all'],\n"
        "            input=listed, check=True, capture_output=True, env=os.environ,\n"
        "        ).stdout\n"
        "    except (OSError, subprocess.CalledProcessError) as error:\n"
        "        raise SystemExit(\n"
        "            'trusted git wrapper could not audit worktree filters'\n"
        "        ) from error\n"
        "    records = checked.split(b'\\0')\n"
        "    if records and records[-1] == b'':\n"
        "        records.pop()\n"
        "    if len(records) % 3:\n"
        "        raise SystemExit('trusted git wrapper received invalid attributes')\n"
        "    for index in range(0, len(records), 3):\n"
        "        path, attribute, value = records[index:index + 3]\n"
        "        if attribute != b'filter':\n"
        "            continue\n"
        "        display = os.fsdecode(path)\n"
        "        raise SystemExit(\n"
        "            f'trusted git wrapper refused worktree filter for {display}'\n"
        "        )\n"
        "if worktree_sensitive():\n"
        "    assert_no_worktree_filters()\n"
        "if command == 'diff':\n"
        "    args[cursor + 1:cursor + 1] = ['--no-ext-diff', '--no-textconv']\n"
        "os.execv(git, [*trusted, *args])\n"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(wrapper, flags, 0o700)
    except FileExistsError as exc:
        raise SystemExit("trusted Git wrapper path already exists") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)
    wrapper.chmod(0o755)
    return wrapper


# Trusted-runtime identity attestation (encode#1147; extensible for #1158).
# The provisioned runtime is git-free, so the encoder cannot read its own Git
# identity at --apply time. When the caller supplies the encoder's clean source
# checkout, its commit, and its normalized origin, THIS root step verifies them
# and writes runtime-attestation.json into the interpreter prefix. --apply reads
# it (only under the trusted-runtime marker the signing supervisor sets) as the
# encoder identity. The file is a versioned envelope carrying an ``axiom_encode``
# sub-object; encode#1158 (pinned CLIs) will add a sibling member to the SAME
# file, so the CLI reader tolerates unknown siblings.
_RUNTIME_ATTESTATION_FILENAME = "runtime-attestation.json"
_RUNTIME_ATTESTATION_SCHEMA = "axiom-encode/trusted-runtime-attestation/v1"

# Moving either pin is a reviewed repository change.  The installer never asks
# npm for "latest" and the runtime disables update checks below.
_CODEX_CLI_VERSION = "0.144.0"
_CODEX_CLI_PINS = {
    ("darwin", "arm64"): {
        "url": "https://registry.npmjs.org/@openai/codex/-/codex-0.144.0-darwin-arm64.tgz",
        "member": "package/vendor/aarch64-apple-darwin/bin/codex",
        "sha256": "978740e6bcbd9af2f850823b723fb74f16d8d1e44de05f7dd6737ae631f72017",
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _install_pinned_codex_cli(
    destination: Path, archive: Path | None = None
) -> dict[str, str]:
    """Install only the reviewed Codex binary, failing closed on its digest."""

    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        machine = "arm64"
    elif machine in {"x86_64", "amd64"}:
        machine = "x64"
    pin = _CODEX_CLI_PINS.get((sys.platform, machine))
    if pin is None:
        raise SystemExit(
            f"refusing to provision: no reviewed Codex CLI pin for {sys.platform}/{machine}"
        )
    downloaded: Path | None = None
    if archive is None:
        descriptor, raw_path = tempfile.mkstemp(
            prefix="axiom-codex-cli-", suffix=".tgz"
        )
        os.close(descriptor)
        downloaded = Path(raw_path)
        try:
            with (
                urllib.request.urlopen(pin["url"], timeout=60) as response,
                downloaded.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
        except Exception as exc:
            downloaded.unlink(missing_ok=True)
            raise SystemExit(f"could not fetch pinned Codex CLI: {exc}") from exc
        archive = downloaded
    try:
        try:
            with tarfile.open(archive, "r:gz") as bundle:
                member = bundle.getmember(pin["member"])
                if not member.isfile() or member.issym() or member.islnk():
                    raise SystemExit(
                        "refusing pinned Codex archive member that is not a file"
                    )
                source = bundle.extractfile(member)
                if source is None:
                    raise SystemExit("pinned Codex archive does not contain its binary")
                bin_dir = destination / "bin"
                bin_dir.mkdir(mode=0o755, exist_ok=True)
                target = bin_dir / "codex"
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(target, flags, 0o755)
                with os.fdopen(descriptor, "wb") as output:
                    shutil.copyfileobj(source, output)
                    output.flush()
                    os.fsync(output.fileno())
        except (KeyError, tarfile.TarError) as exc:
            raise SystemExit(f"invalid pinned Codex CLI archive: {exc}") from exc
        actual = _sha256_file(target)
        if actual != pin["sha256"]:
            target.unlink(missing_ok=True)
            raise SystemExit(
                "refusing to provision: pinned Codex CLI sha256 mismatch "
                f"(expected {pin['sha256']}, got {actual})"
            )
        target.chmod(0o755)
        config = {
            "schema": "axiom-encode/trusted-codex-cli/v1",
            "version": _CODEX_CLI_VERSION,
            "sha256": actual,
            "path": str(target),
        }
        config_path = destination / "codex-cli.json"
        config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
        config_path.chmod(0o444)
        return config
    finally:
        if downloaded is not None:
            downloaded.unlink(missing_ok=True)


_ENCODER_PACKAGE_PATHSPEC = "src/axiom_encode"
# Files whose in-tree version must agree for the encoder to declare one version,
# and the paths whose change after a version bump is "unversioned encoder drift".
# Mirrors the invariant cli.py enforces in Git mode.
_ENCODER_VERSION_FILES = (
    "pyproject.toml",
    "src/axiom_encode/__init__.py",
    "uv.lock",
)
_ENCODER_VERSIONED_PREFIXES = ("src/axiom_encode/",)
_ENCODER_VERSIONED_FILES = frozenset({"pyproject.toml", "uv.lock"})
_GITHUB_REMOTE_PATTERNS = (
    r"https://github\.com/(?P<slug>[^/\s]+/[^/\s]+?)(?:\.git)?/?$",
    r"git@github\.com:(?P<slug>[^/\s]+/[^/\s]+?)(?:\.git)?$",
    r"ssh://git@github\.com/(?P<slug>[^/\s]+/[^/\s]+?)(?:\.git)?/?$",
)

# Command-line config that overrides ANY value the caller's repo config could
# carry, disabling every hook/driver git might otherwise EXECUTE while reading
# the repo. `-c` has the highest precedence, so a repo-local core.fsmonitor,
# hooksPath, clean/smudge/textconv driver, or external diff cannot fire.
_HARDENED_GIT_FLAGS = (
    "--no-pager",
    "-c",
    "core.fsmonitor=",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.attributesFile=/dev/null",
    "-c",
    "core.sshCommand=false",
    "-c",
    "protocol.ext.allow=never",
    "-c",
    "credential.helper=",
    "-c",
    "diff.external=/usr/bin/false",
    "-c",
    "log.showSignature=false",
    "-c",
    "submodule.recurse=false",
)


def _github_repository_identity(remote_url: str) -> str | None:
    """Return a credential-free canonical github.com/OWNER/REPO identity.

    Mirrors ``axiom_encode.harness.evals._github_repository_identity`` so the
    attestation records the SAME normalized origin form the CLI compares against.
    """

    value = remote_url.strip()
    for pattern in _GITHUB_REMOTE_PATTERNS:
        match = re.fullmatch(pattern, value)
        if match is not None:
            return f"github.com/{match.group('slug')}"
    return None


def _hardened_git_environment(home: Path) -> dict[str, str]:
    """A minimal environment that ignores every ambient/global/system Git config.

    No inherited ``GIT_*`` (proxy, ssh, askpass) crosses in, replace objects and
    lazy fetches are off, and ``HOME`` points at an empty root-owned directory so
    no ``~/.gitconfig`` is read.
    """

    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(home),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_OPTIONAL_LOCKS": "0",
    }


def _hardened_git(trusted_git: Path, git_dir: Path, home: Path, *args: str) -> bytes:
    """Run one read-only, worktree-free Git query against a ROOT-OWNED git dir.

    Every invocation targets ``--git-dir`` only (never a worktree), so git never
    scans or materializes working-tree files and thus never runs a repo
    fsmonitor, clean/smudge filter, or textconv driver. The git dir is root-owned
    by construction, so ``safe.directory`` is unnecessary and is NOT set.
    """

    try:
        completed = subprocess.run(
            [
                str(trusted_git),
                *_HARDENED_GIT_FLAGS,
                "--git-dir",
                str(git_dir),
                *args,
            ],
            check=True,
            capture_output=True,
            env=_hardened_git_environment(home),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", b"") or b""
        detail = (
            stderr.decode("utf-8", "replace").strip()
            if isinstance(stderr, bytes)
            else str(exc)
        )
        raise SystemExit(
            f"refusing to provision: hardened git {' '.join(args)} failed: "
            f"{detail or exc}"
        ) from exc
    return completed.stdout


def _hardened_git_text(trusted_git: Path, git_dir: Path, home: Path, *args: str) -> str:
    return _hardened_git(trusted_git, git_dir, home, *args).decode("utf-8").strip()


def _own_encoder_git_directory(encoder_git_root: Path, staging: Path) -> Path:
    """Copy the caller checkout's ``.git`` into a fresh root-owned directory.

    Owning the objects and refs (instead of running git against the caller-owned,
    user-writable checkout with ``safe.directory=*``) removes BOTH the
    dubious-ownership refusal and the caller's ability to race the repository
    mid-provision. All subsequent git runs bare against this root-owned copy.
    """

    caller_git = encoder_git_root / ".git"
    if caller_git.is_symlink() or not caller_git.is_dir():
        raise SystemExit(
            f"refusing to provision: {encoder_git_root}/.git is not a plain "
            "directory (worktree/submodule gitfile pointers are not accepted)"
        )
    git_dir = staging / "encoder.git"
    # symlinks=True copies links AS links, never following one out of the tree
    # during the copy; bare git never materializes a worktree from them.
    shutil.copytree(caller_git, git_dir, symlinks=True)
    # The copy is owned by the provisioning identity (root in production); a
    # different owner would mean a hijacked staging path.
    if git_dir.stat().st_uid != os.geteuid():
        raise SystemExit(
            "refusing to provision: copied encoder git dir is not owned by the "
            f"provisioning identity: {git_dir}"
        )
    return git_dir


def _extract_pyproject_version(content: str) -> str | None:
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', content)
    return match.group(1) if match else None


def _extract_package_init_version(content: str) -> str | None:
    match = re.search(r'(?m)^__version__\s*=\s*"([^"]+)"\s*$', content)
    return match.group(1) if match else None


def _extract_uv_lock_version(content: str) -> str | None:
    block = re.search(
        r'(?ms)^\[\[package\]\]\s*^name\s*=\s*"axiom-encode"\s*^version\s*=\s*"([^"]+)"',
        content,
    )
    return block.group(1) if block else None


def _encoder_versions_at_ref(
    trusted_git: Path, git_dir: Path, home: Path, ref: str
) -> dict[str, str | None]:
    """Return {pyproject, package, lock} versions declared at one commit."""

    def show(path: str) -> str:
        try:
            return _hardened_git_text(
                trusted_git, git_dir, home, "show", f"{ref}:{path}"
            )
        except SystemExit:
            return ""

    return {
        "pyproject": _extract_pyproject_version(show("pyproject.toml")),
        "package": _extract_package_init_version(show("src/axiom_encode/__init__.py")),
        "lock": _extract_uv_lock_version(show("uv.lock")),
    }


def _snapshot_declared_version(
    trusted_git: Path, git_dir: Path, home: Path, commit: str
) -> str:
    """Return the single version pyproject/__init__/uv.lock all declare at commit.

    Fails closed unless all three are present and identical, so the recorded
    version provably comes from the attested tree (not a runtime __version__).
    """

    versions = _encoder_versions_at_ref(trusted_git, git_dir, home, commit)
    declared = versions["pyproject"]
    if not declared or any(value != declared for value in versions.values()):
        raise SystemExit(
            "refusing to provision: encoder version metadata at "
            f"{commit[:12]} is inconsistent or missing: {versions}"
        )
    return declared


def _parse_numeric_version(version: str) -> tuple[int, ...] | None:
    parts = version.split(".")
    if not parts or any(not re.fullmatch(r"\d+", part) for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _encoder_version_increased(
    current: dict[str, str | None], previous: dict[str, str | None]
) -> bool:
    current_version = current.get("pyproject")
    previous_version = previous.get("pyproject")
    if not current_version:
        return False
    if not previous_version:
        return True
    current_key = _parse_numeric_version(current_version)
    previous_key = _parse_numeric_version(previous_version)
    if current_key is None or previous_key is None:
        return current_version > previous_version
    return current_key > previous_key


def _is_encoder_versioned_path(path: str) -> bool:
    return path in _ENCODER_VERSIONED_FILES or path.startswith(
        _ENCODER_VERSIONED_PREFIXES
    )


def _require_snapshot_version_bump(
    trusted_git: Path, git_dir: Path, home: Path, commit: str
) -> str:
    """Preserve, against the attested snapshot, the invariant that no encoder
    file changed after the latest version bump.

    This is the same guard cli.py enforces in Git mode, run here at provision
    time against the root-owned history so the runtime does not re-derive it by
    fiat. It fails closed if the commit's history is too shallow to locate the
    bump (the workflow must check the encoder out with full history).
    """

    if (
        _hardened_git_text(
            trusted_git, git_dir, home, "rev-parse", "--is-shallow-repository"
        )
        == "true"
    ):
        raise SystemExit(
            "refusing to provision: encoder git dir is shallow; the version-bump "
            "invariant needs full history (check the encoder out with "
            "fetch-depth: 0)"
        )
    lines = _hardened_git_text(
        trusted_git,
        git_dir,
        home,
        "log",
        "--format=%H",
        commit,
        "--",
        *_ENCODER_VERSION_FILES,
    ).splitlines()
    version_commit: str | None = None
    for candidate in (line.strip() for line in lines if line.strip()):
        current = _encoder_versions_at_ref(trusted_git, git_dir, home, candidate)
        if not current.get("pyproject") or not current.get("package"):
            continue
        parents = _hardened_git_text(
            trusted_git, git_dir, home, "rev-list", "--parents", "-n", "1", candidate
        ).split()
        if len(parents) <= 1:
            version_commit = candidate  # root commit that declares a version
            break
        previous = _encoder_versions_at_ref(trusted_git, git_dir, home, parents[1])
        if _encoder_version_increased(current, previous):
            version_commit = candidate
            break
    if not version_commit:
        raise SystemExit(
            "refusing to provision: no committed encoder version bump is reachable "
            f"from {commit[:12]} (check the encoder out with full history)"
        )
    changed = _hardened_git_text(
        trusted_git,
        git_dir,
        home,
        "diff",
        "--name-only",
        "--diff-filter=ACMRT",
        f"{version_commit}..{commit}",
    ).splitlines()
    unversioned = sorted(
        {
            path.strip()
            for path in changed
            if path.strip() and _is_encoder_versioned_path(path.strip())
        }
    )
    if unversioned:
        raise SystemExit(
            "refusing to provision: encoder files changed after the latest version "
            f"bump ({version_commit[:12]}) without a bump: {', '.join(unversioned)}"
        )
    return version_commit


def _export_encoder_package(
    trusted_git: Path, git_dir: Path, home: Path, commit: str, staging: Path
) -> Path:
    """Export the tracked ``src/axiom_encode`` tree at ``commit`` into a
    root-owned directory by reading raw tree objects.

    Deliberately NOT ``git archive``: archive honors in-tree ``.gitattributes``.
    ``export-subst`` would expand ``$Format:...$`` from the COMMIT MESSAGE into a
    module (arbitrary code at encoder import, inside the signing runtime) while
    ``git show``/``cat-file`` — and therefore the version binding — see the
    benign unexpanded template; ``export-ignore`` would silently drop a module.
    No git config disables either. ``ls-tree`` enumerates the exact tree objects
    and ``cat-file blob`` returns their raw bytes with ZERO attribute processing,
    so every tracked blob is emitted verbatim and export-ignore cannot hide one.
    Committed symlinks and gitlinks are refused rather than materialized.
    """

    listing = _hardened_git(
        trusted_git,
        git_dir,
        home,
        "ls-tree",
        "-r",
        "-z",
        commit,
        "--",
        _ENCODER_PACKAGE_PATHSPEC,
    )
    export_root = staging / "encoder-snapshot"
    export_root.mkdir()
    package = export_root / _ENCODER_PACKAGE_PATHSPEC
    package_root = package.resolve()
    entries = [entry for entry in listing.split(b"\0") if entry]
    if not entries:
        raise SystemExit(
            f"refusing to provision: {commit[:12]} has no tracked "
            f"{_ENCODER_PACKAGE_PATHSPEC} package"
        )
    for entry in entries:
        # `-z` emits raw (unquoted) "<mode> <type> <oid>\t<path>" records.
        meta, separator, raw_path = entry.partition(b"\t")
        fields = meta.split(b" ")
        if separator != b"\t" or len(fields) != 3:
            raise SystemExit(
                f"refusing to provision: malformed ls-tree entry: {entry!r}"
            )
        mode = fields[0].decode("ascii", "replace")
        object_type = fields[1]
        oid = fields[2].decode("ascii", "replace")
        relative = os.fsdecode(raw_path)
        if mode == "120000":
            raise SystemExit(
                "refusing to provision: encoder tree contains a committed symlink "
                f"at {relative}; refusing to materialize it"
            )
        if mode == "160000":
            raise SystemExit(
                "refusing to provision: encoder tree contains a committed "
                f"gitlink/submodule at {relative}; refusing to materialize it"
            )
        if object_type != b"blob" or mode not in ("100644", "100755"):
            raise SystemExit(
                "refusing to provision: unsupported encoder tree entry "
                f"{mode} {object_type!r} at {relative}"
            )
        if re.fullmatch(r"[0-9a-f]{40,64}", oid) is None:
            raise SystemExit(
                f"refusing to provision: malformed object id for {relative}: {oid}"
            )
        # ls-tree paths are already inside the pathspec, but normalize and confirm
        # containment as defense-in-depth against any traversal component.
        destination = (export_root / relative).resolve()
        if not destination.is_relative_to(package_root):
            raise SystemExit(
                f"refusing to provision: encoder tree path escapes the package: "
                f"{relative}"
            )
        blob = _hardened_git(trusted_git, git_dir, home, "cat-file", "blob", oid)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob)
        if mode == "100755":
            destination.chmod(0o755)
    if package.is_symlink() or not package.is_dir():
        raise SystemExit(
            f"refusing to provision: {commit[:12]} has no tracked "
            f"{_ENCODER_PACKAGE_PATHSPEC} package"
        )
    return package


def _tree_hash_update(hasher: "hashlib._Hash", relative_path: str, raw: bytes) -> None:
    relative_bytes = relative_path.encode("utf-8")
    hasher.update(len(relative_bytes).to_bytes(8, "big"))
    hasher.update(relative_bytes)
    hasher.update(len(raw).to_bytes(8, "big"))
    hasher.update(raw)


def _deterministic_package_tree_sha256(package: Path) -> str:
    """Replicate ``axiom_encode.harness.evals._deterministic_tree_identity``'s
    directory digest so the attestation records the SAME ``tree_sha256`` the CLI
    computes over the running package at apply time (byte-binding).

    Identical algorithm: domain-separated ``axiom-eval-tree-v1\\0`` seed, sorted
    walk that never follows symlinks, ``__pycache__`` excluded, symlinks and
    non-regular files rejected, and each ``<relpath>``/``<bytes>`` pair framed
    with 8-byte big-endian lengths. The provisioner cannot import the module
    (``-I -S`` standalone), so this mirror is kept in lockstep with it.
    """

    root = package.resolve()
    hasher = hashlib.sha256(b"axiom-eval-tree-v1\0")
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        retained: list[str] = []
        for name in sorted(directory_names):
            if name == "__pycache__":
                continue
            if (directory_path / name).is_symlink():
                raise SystemExit(
                    "refusing to provision: package tree contains a directory "
                    f"symlink: {directory_path / name}"
                )
            retained.append(name)
        directory_names[:] = retained
        for name in sorted(file_names):
            path = directory_path / name
            if path.is_symlink():
                raise SystemExit(
                    f"refusing to provision: package tree contains a file symlink: "
                    f"{path}"
                )
            if not path.is_file():
                raise SystemExit(
                    f"refusing to provision: package tree contains a non-regular "
                    f"file: {path}"
                )
            _tree_hash_update(
                hasher, path.relative_to(root).as_posix(), path.read_bytes()
            )
    return hasher.hexdigest()


def _overlay_encoder_package(runtime: Path, package: Path) -> Path:
    """Replace the runtime's ``axiom_encode`` with the attested snapshot bytes.

    This is what makes the executing package provably the attested commit: the
    caller-supplied ``--site-packages`` provides only third-party dependencies;
    the encoder itself comes from a raw ``ls-tree``+``cat-file`` export of the
    verified commit. Returns the installed package directory.
    """

    site_packages = (
        runtime
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if not site_packages.is_dir():
        raise SystemExit(
            f"refusing to provision: runtime site-packages missing at {site_packages}"
        )
    target = site_packages / "axiom_encode"
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    shutil.copytree(package, target, symlinks=True)
    # The attested tree carries no startup hooks, but re-run the purge over it so
    # the same guarantee the staged tree got also covers the overlaid package.
    doomed = sorted(
        {
            match
            for pattern in _FORBIDDEN_STARTUP_GLOBS
            for match in target.rglob(pattern)
        }
    )
    for forbidden in doomed:
        if forbidden.is_symlink() or forbidden.is_file():
            forbidden.unlink()
        elif forbidden.is_dir():
            shutil.rmtree(forbidden)
    return target


def _publish_runtime_attestation(
    runtime: Path,
    encoder_origin_repository: str,
    encoder_commit: str,
    declared_version: str,
    package_tree_sha256: str,
    codex_cli: dict[str, str] | None = None,
) -> Path:
    """Atomically publish the attestation into a root-owned prefix, fail-closed.

    The containing directory must be root-owned and is stripped of group/other
    write first. The file is created with ``O_EXCL | O_NOFOLLOW`` (a pre-placed
    inode or symlink at the predictable path aborts the provision), written and
    fsync'd, then pinned 0444. A retained write descriptor on a pre-existing
    inode cannot survive this because the inode is created fresh here.
    """

    metadata = runtime.stat()
    # Owned by the provisioning identity (root in production); a foreign owner
    # would mean the prefix was pre-seeded by someone else.
    if metadata.st_uid != os.geteuid():
        raise SystemExit(
            "refusing to provision: runtime prefix is not owned by the "
            f"provisioning identity: {runtime}"
        )
    runtime.chmod(metadata.st_mode & ~(stat.S_IWGRP | stat.S_IWOTH))
    path = runtime / _RUNTIME_ATTESTATION_FILENAME
    attestation = {
        "schema": _RUNTIME_ATTESTATION_SCHEMA,
        "provisioned_at": datetime.now(timezone.utc).isoformat(),
        "axiom_encode": {
            "origin_repository": encoder_origin_repository,
            "commit": encoder_commit,
            "version": declared_version,
            "package_tree_sha256": package_tree_sha256,
        },
    }
    if codex_cli is not None:
        attestation["codex_cli"] = {
            "version": codex_cli["version"],
            "sha256": codex_cli["sha256"],
        }
    payload = json.dumps(attestation, indent=2, sort_keys=True) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as exc:
        raise SystemExit(
            f"refusing to provision: attestation path already exists: {path}"
        ) from exc
    except OSError as exc:
        raise SystemExit(
            f"refusing to provision: cannot create attestation safely at {path}: {exc}"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o444)
    return path


def _verify_and_export_encoder_snapshot(
    trusted_git: Path,
    encoder_git_root: Path,
    encoder_commit: str,
    encoder_origin_repository: str,
    staging: Path,
) -> tuple[str, Path]:
    """Own the encoder git dir, verify the attested commit, and export its
    tracked ``src/axiom_encode``. Returns (declared_version, package_path)."""

    if re.fullmatch(r"[0-9a-f]{40}", encoder_commit) is None:
        raise SystemExit(
            "refusing to provision: --encoder-commit is not a full 40-hex SHA: "
            f"{encoder_commit}"
        )
    if re.fullmatch(r"github\.com/[^/\s]+/[^/\s]+", encoder_origin_repository) is None:
        raise SystemExit(
            "refusing to provision: --encoder-origin-repository is not a canonical "
            f"github.com/OWNER/REPO identity: {encoder_origin_repository}"
        )
    try:
        root = encoder_git_root.resolve(strict=True)
    except OSError as exc:
        raise SystemExit(
            f"refusing to provision: --encoder-git-root does not exist: "
            f"{encoder_git_root} ({exc})"
        ) from exc
    home = staging / "git-home"
    home.mkdir()
    git_dir = _own_encoder_git_directory(root, staging)
    try:
        resolved = _hardened_git_text(
            trusted_git,
            git_dir,
            home,
            "rev-parse",
            "--verify",
            f"{encoder_commit}^{{commit}}",
        )
    except SystemExit as exc:
        raise SystemExit(
            f"refusing to provision: {encoder_commit} is not a commit object in "
            "--encoder-git-root"
        ) from exc
    if resolved != encoder_commit:
        raise SystemExit(
            f"refusing to provision: {encoder_commit} is not a commit object in "
            "--encoder-git-root"
        )
    head = _hardened_git_text(trusted_git, git_dir, home, "rev-parse", "HEAD")
    if head != encoder_commit:
        raise SystemExit(
            f"refusing to provision: --encoder-git-root HEAD {head} does not match "
            f"--encoder-commit {encoder_commit}"
        )
    try:
        remote_url = _hardened_git_text(
            trusted_git, git_dir, home, "config", "--get", "remote.origin.url"
        )
    except SystemExit:
        remote_url = ""
    origin = _github_repository_identity(remote_url)
    if origin != encoder_origin_repository:
        raise SystemExit(
            f"refusing to provision: --encoder-git-root origin {origin!r} does not "
            f"match --encoder-origin-repository {encoder_origin_repository!r}"
        )
    declared_version = _snapshot_declared_version(
        trusted_git, git_dir, home, encoder_commit
    )
    _require_snapshot_version_bump(trusted_git, git_dir, home, encoder_commit)
    package = _export_encoder_package(
        trusted_git, git_dir, home, encoder_commit, staging
    )
    return declared_version, package


def _signing_trust_roots_payload(
    apply_root: str,
    eval_root: str,
    corpus_release_root: str,
    retired_corpus_release_roots: tuple[str, ...] = (),
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "axiom-encode/signing-trust-roots/v2",
        "apply_ed25519_public_key": apply_root,
        "eval_ed25519_public_key": eval_root,
        "corpus_release_ed25519_public_key": corpus_release_root,
    }
    if retired_corpus_release_roots:
        payload["schema"] = "axiom-encode/signing-trust-roots/v3"
        payload.pop("corpus_release_ed25519_public_key")
        payload["corpus_release_ed25519_public_keys"] = [
            corpus_release_root,
            *retired_corpus_release_roots,
        ]
    return payload


def provision(
    destination: Path,
    supervisor: Path,
    site_packages: Path,
    apply_root: str,
    eval_root: str,
    corpus_release_root: str,
    git: Path,
    require_prefix_under: Path | None = None,
    patchelf: str | None = None,
    max_runtime_files: int = DEFAULT_MAX_RUNTIME_FILES,
    encoder_origin_repository: str | None = None,
    encoder_commit: str | None = None,
    encoder_git_root: Path | None = None,
    codex_cli_archive: Path | None = None,
    install_pinned_codex_cli: bool = False,
    retired_corpus_release_roots: tuple[str, ...] = (),
) -> None:
    if any(not root.strip() for root in retired_corpus_release_roots):
        raise SystemExit(
            "refusing to provision: retired corpus release roots must not be empty"
        )
    encoder_attestation_args = (
        encoder_origin_repository,
        encoder_commit,
        encoder_git_root,
    )
    provision_encoder = all(arg is not None for arg in encoder_attestation_args)
    if (
        any(arg is not None for arg in encoder_attestation_args)
        and not provision_encoder
    ):
        raise SystemExit(
            "refusing to provision: --encoder-origin-repository, --encoder-commit, "
            "and --encoder-git-root must be supplied together or not at all"
        )
    trusted_git = _resolve_trusted_git(git)
    source_runtime = Path(sys.base_prefix).resolve(strict=True)
    source_interpreter = Path(sys.executable).resolve(strict=True)
    _assert_sane_source_prefix(source_runtime, require_prefix_under, max_runtime_files)

    # Verify + export the encoder from the attested commit BEFORE staging, into a
    # root-owned scratch tree that is discarded at the end. The overlaid bytes —
    # not an arbitrary caller --site-packages axiom_encode — are what execute and
    # what the attestation names, so the runtime provably IS the attested commit.
    encoder_staging: Path | None = None
    declared_version: str | None = None
    encoder_package: Path | None = None
    if provision_encoder:
        assert encoder_git_root is not None  # narrowed by provision_encoder
        assert encoder_commit is not None
        assert encoder_origin_repository is not None
        encoder_staging = Path(tempfile.mkdtemp(prefix="axiom-encoder-snapshot-"))
        declared_version, encoder_package = _verify_and_export_encoder_snapshot(
            trusted_git,
            encoder_git_root,
            encoder_commit,
            encoder_origin_repository,
            encoder_staging,
        )
    try:
        runtime = destination / "python"
        _stage_runtime_tree(source_runtime, runtime, site_packages)
        package_tree_sha256: str | None = None
        if provision_encoder:
            assert encoder_package is not None
            installed_package = _overlay_encoder_package(runtime, encoder_package)
            # Hash the INSTALLED package (exactly what the CLI hashes at apply
            # time via Path(__file__).parent) so the recorded tree digest binds
            # the executing bytes.
            package_tree_sha256 = _deterministic_package_tree_sha256(installed_package)
        if sys.platform == "linux":
            if not patchelf:
                patchelf = shutil.which("patchelf")
                if patchelf is None:
                    raise SystemExit(
                        "patchelf is required on linux to relocate the runtime "
                        "(install it or pass --patchelf)"
                    )
            _relocate_elf_rpaths(runtime, patchelf)
        interpreter = runtime / source_interpreter.relative_to(source_runtime)
        _assert_self_contained(runtime, source_runtime, interpreter)
        _install_trusted_git_wrapper(interpreter.parent, interpreter, trusted_git)
        codex_cli = (
            _install_pinned_codex_cli(destination, codex_cli_archive)
            if install_pinned_codex_cli
            else None
        )
        launcher = destination / "axiom-encode"
        launcher.write_text(
            f"#!{interpreter} -I\nraise SystemExit('launcher executed')\n"
        )
        launcher.chmod(0o755)
        trust = destination / "signing-trust-roots.json"
        trust_payload = _signing_trust_roots_payload(
            apply_root,
            eval_root,
            corpus_release_root,
            retired_corpus_release_roots,
        )
        trust.write_text(json.dumps(trust_payload, sort_keys=True) + "\n")
        trust.chmod(0o644)
        if provision_encoder:
            assert encoder_origin_repository is not None
            assert encoder_commit is not None
            assert declared_version is not None
            assert package_tree_sha256 is not None
            _publish_runtime_attestation(
                runtime,
                encoder_origin_repository,
                encoder_commit,
                declared_version,
                package_tree_sha256,
                codex_cli,
            )
        shutil.copy2(supervisor, destination / "axiom-encode-signing-supervisor")
        (destination / "axiom-encode-signing-supervisor").chmod(0o755)
    finally:
        if encoder_staging is not None:
            shutil.rmtree(encoder_staging, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--apply-root", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--corpus-release-root", required=True)
    parser.add_argument(
        "--retired-corpus-release-root",
        action="append",
        default=[],
        help="Retired verification-only corpus release root; repeat for each key.",
    )
    parser.add_argument(
        "--install-pinned-codex-cli",
        action="store_true",
        help="Fetch/install the repository-pinned Codex CLI for subscription generation.",
    )
    parser.add_argument(
        "--git",
        type=Path,
        required=True,
        help="Exact root-owned Git executable exposed inside the isolated PATH.",
    )
    parser.add_argument(
        "--require-prefix-under",
        type=Path,
        default=None,
        help="Refuse unless the invoking interpreter's prefix is under this "
        "directory (CI passes the tool-cache root).",
    )
    parser.add_argument(
        "--patchelf",
        default=None,
        help="patchelf binary for the linux RPATH relocation pass.",
    )
    parser.add_argument(
        "--max-runtime-files",
        type=int,
        default=DEFAULT_MAX_RUNTIME_FILES,
        help="Refuse prefixes holding more files than this (copy-equivalent "
        "count, following symlinks the way the copy will).",
    )
    parser.add_argument(
        "--codex-cli-archive",
        type=Path,
        default=None,
        help="Optional pre-fetched exact pinned npm archive.",
    )
    parser.add_argument(
        "--encoder-origin-repository",
        default=None,
        help="Canonical github.com/OWNER/REPO identity the encoder git dir's "
        "origin must normalize to. Recorded in runtime-attestation.json. Requires "
        "--encoder-commit and --encoder-git-root.",
    )
    parser.add_argument(
        "--encoder-commit",
        default=None,
        help="Full 40-hex commit to attest. The runtime's axiom_encode package is "
        "built from a git archive of THIS commit (not from --site-packages), and "
        "--encoder-git-root HEAD must equal it.",
    )
    parser.add_argument(
        "--encoder-git-root",
        type=Path,
        default=None,
        help="Encoder checkout whose .git is copied root-owned and verified "
        "(commit present, HEAD==commit, origin, declared version, version-bump "
        "invariant) before its src/axiom_encode is archived into the runtime. Must "
        "have full history (fetch-depth: 0) for the version-bump check.",
    )
    args = parser.parse_args()
    provision(
        args.destination.resolve(),
        args.supervisor.resolve(),
        args.site_packages.resolve(),
        args.apply_root,
        args.eval_root,
        args.corpus_release_root,
        args.git,
        args.require_prefix_under,
        args.patchelf,
        args.max_runtime_files,
        args.encoder_origin_repository,
        args.encoder_commit,
        args.encoder_git_root.resolve() if args.encoder_git_root is not None else None,
        args.codex_cli_archive.resolve()
        if args.codex_cli_archive is not None
        else None,
        args.install_pinned_codex_cli,
        tuple(args.retired_corpus_release_root),
    )
