"""Console-script dispatch for the ``axiom-encode`` command.

``pyproject.toml`` points the ``axiom-encode`` script here. Subcommands
implemented outside :mod:`axiom_encode.cli` are dispatched first; every
other invocation delegates to :func:`axiom_encode.cli.main` with
``sys.argv`` untouched, so the main command surface behaves exactly as if
it were registered directly.
"""

from __future__ import annotations

import sys


def main() -> int | None:
    # This must run before importing ``axiom_encode.cli``. Production private
    # keys remain inside external signers; the mutable Python process that can
    # later launch model subprocesses receives only a confined broker capability.
    from axiom_encode.signing_broker import (
        attach_signing_broker_from_environment,
        reject_direct_private_signing_environment,
    )

    reject_direct_private_signing_environment()
    attach_signing_broker_from_environment()

    argv = sys.argv[1:]
    if argv and argv[0] == "check-source-staleness":
        from axiom_encode.source_hash import run_check_source_staleness

        return run_check_source_staleness(argv[1:])
    if argv and argv[0] == "export-core-build-spec":
        from axiom_encode.core_export import run_export_core_build_spec

        return run_export_core_build_spec(argv[1:])
    from axiom_encode.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":  # pragma: no cover - exercised by attachment tests
    raise SystemExit(main())
