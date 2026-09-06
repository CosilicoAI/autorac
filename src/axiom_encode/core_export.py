"""Package candidate bytes for the real Axiom core compiler.

This module deliberately does not parse RuleSpec or resolve imports. Callers
select an explicit candidate and dependency files; the native compiler owns
target syntax, import closure, and executable validity. Export is not validation
or signed admission, and does not mutate generated rules or acceptance tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axiom_encode.harness.evals import EvalResult

BUILD_SPEC_FORMAT = "axiom/build-spec/v0"
MAX_INPUT_BYTES = 16 * 1024 * 1024


class CoreExportError(ValueError):
    """An export failure that can be reported without a traceback."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _read_source(path: Path) -> bytes:
    # Paths are explicitly selected by the caller, not inferred from model
    # output or imports. Reject final-component symlinks and special files;
    # parent-directory containment/authentication is not an export guarantee.
    nofollow = getattr(os, "O_NOFOLLOW", None)
    nonblocking = getattr(os, "O_NONBLOCK", None)
    if nofollow is None or nonblocking is None:
        raise CoreExportError(
            "unsupported_platform",
            "safe regular-file reads require O_NOFOLLOW and O_NONBLOCK",
        )
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow | nonblocking)
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise CoreExportError("unsafe_input", f"not a regular file: {path}")
            if info.st_size > MAX_INPUT_BYTES:
                raise CoreExportError(
                    "input_too_large", f"source exceeds 16 MiB: {path}"
                )
            raw = source.read(MAX_INPUT_BYTES + 1)
    except OSError as error:
        raise CoreExportError("io_error", f"cannot read {path}: {error}") from error
    if len(raw) > MAX_INPUT_BYTES:
        raise CoreExportError("input_too_large", f"source exceeds 16 MiB: {path}")
    return raw


def _check_target(target: str) -> None:
    if not isinstance(target, str) or not target.strip():
        raise CoreExportError(
            "invalid_target", "module targets must be nonempty strings"
        )
    try:
        target.encode("utf-8")
    except UnicodeEncodeError as error:
        raise CoreExportError(
            "invalid_target", "module targets must be valid UTF-8"
        ) from error


def _encoded_spec(spec: dict) -> bytes:
    payload = (
        json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    # Include JSON escaping, keys, metadata, and the final newline in the core
    # CLI's reader limit, not just the source-file sizes.
    if len(payload) > MAX_INPUT_BYTES:
        raise CoreExportError(
            "build_spec_too_large", "serialized build spec exceeds 16 MiB"
        )
    return payload


def build_spec(
    root: str,
    candidate: Path,
    modules: Sequence[tuple[str, Path]] = (),
    expected_candidate_sha256: str | None = None,
) -> tuple[dict, dict]:
    """Return the exact core BuildSpec and separate, unvalidated export metadata.

    Hash the bytes actually read, before strict UTF-8 decoding. No newline or
    Unicode normalization takes place. The optional expected candidate hash is
    a byte pin supplied by the caller, not evidence of legal/source authority.
    """
    _check_target(root)
    if expected_candidate_sha256 is not None and (
        len(expected_candidate_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_candidate_sha256
        )
    ):
        raise CoreExportError(
            "invalid_digest", "candidate SHA-256 must be 64 lowercase hex characters"
        )
    selected = [
        (root, Path(candidate)),
        *[(target, Path(path)) for target, path in modules],
    ]
    seen: set[str] = set()
    for target, _ in selected:
        _check_target(target)
        if target in seen:
            raise CoreExportError(
                "duplicate_module", f"module target assigned more than once: {target}"
            )
        seen.add(target)

    sources: dict[str, str] = {}
    candidate_sha256 = ""
    total_bytes = 0
    for target, path in selected:
        raw = _read_source(path)
        total_bytes += len(raw)
        if total_bytes > MAX_INPUT_BYTES:
            raise CoreExportError(
                "build_spec_too_large", "combined source bytes exceed 16 MiB"
            )
        if target == root:
            candidate_sha256 = hashlib.sha256(raw).hexdigest()
            if (
                expected_candidate_sha256 is not None
                and candidate_sha256 != expected_candidate_sha256
            ):
                raise CoreExportError(
                    "candidate_digest_mismatch",
                    "candidate bytes differ from the expected SHA-256",
                )
        try:
            sources[target] = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise CoreExportError(
                "invalid_utf8", f"source is not valid UTF-8: {path}"
            ) from error
    spec = {"format": BUILD_SPEC_FORMAT, "root": root, "modules": sources}
    payload = _encoded_spec(spec)
    return spec, {
        "format": BUILD_SPEC_FORMAT,
        "assurance": "unvalidated_candidate",
        "build_spec_sha256": hashlib.sha256(payload).hexdigest(),
        "candidate_sha256": candidate_sha256,
        "module_count": len(sources),
    }


def build_spec_from_eval_result(
    result: EvalResult,
    *,
    root: str,
    modules: Sequence[tuple[str, Path]] = (),
) -> tuple[dict, dict]:
    """Export one caller-selected completed result with its mandatory byte pin.

    A failed validation may still produce a candidate useful for diagnosis.
    Neither success nor other result metadata confers validation or admission
    on the export. Dependencies remain explicit caller-selected inputs.
    """
    if not isinstance(result.output_file, str) or not result.output_file:
        raise CoreExportError(
            "missing_candidate", "evaluation result must name a candidate file"
        )
    digest = result.generated_output_sha256
    if not isinstance(digest, str):
        raise CoreExportError(
            "invalid_digest", "evaluation result must record a candidate SHA-256"
        )
    return build_spec(
        root,
        Path(result.output_file),
        modules,
        expected_candidate_sha256=digest,
    )


def run_export_core_build_spec(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="axiom-encode export-core-build-spec",
        description="Package exact candidate and dependency bytes for the Axiom core compiler; no validation or signing.",
    )
    parser.add_argument("--root", required=True, help="exact root module target")
    parser.add_argument(
        "--candidate",
        required=True,
        type=Path,
        help="generated candidate RuleSpec file",
    )
    parser.add_argument(
        "--module",
        action="append",
        nargs=2,
        metavar=("TARGET", "FILE"),
        default=[],
        help="explicit dependency mapping; may be repeated",
    )
    parser.add_argument(
        "--expect-candidate-sha256",
        help="optional expected digest from the generation record",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="new build-spec JSON file; never overwritten",
    )
    args = parser.parse_args(argv)
    try:
        spec, metadata = build_spec(
            args.root, args.candidate, args.module, args.expect_candidate_sha256
        )
        payload = _encoded_spec(spec)
        # Validate every source and the complete serialized size before any
        # output creation. Exclusive creation preserves existing files/symlinks.
        descriptor = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except (CoreExportError, OSError) as error:
        if isinstance(error, CoreExportError):
            code, message = error.code, error.message
        else:
            code, message = "io_error", str(error)
        print(
            json.dumps({"ok": False, "error": {"code": code, "message": message}}),
            file=sys.stderr,
        )
        return 1
    print(json.dumps({"ok": True, **metadata, "output": str(args.out)}, sort_keys=True))
    return 0
