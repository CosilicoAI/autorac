#!/usr/bin/env python3
"""Compatibility entrypoint for protected signed-backfill publication helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_implementation_path = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "axiom_encode"
    / "prepare_signed_backfill.py"
)
_spec = importlib.util.spec_from_file_location(
    "_axiom_encode_prepare_signed_backfill",
    _implementation_path,
)
if _spec is None or _spec.loader is None:
    raise RuntimeError("could not load protected signed-backfill helper implementation")
_implementation = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _implementation
_spec.loader.exec_module(_implementation)

# ``runpy.run_path`` returns this wrapper's globals rather than consulting the
# replacement in ``sys.modules``.  Preserve the historical helper contract for
# callers that load this script that way (notably extract_repair_candidate.py).
for _name in dir(_implementation):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_implementation, _name)

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "stage":
        sys.argv[2:] = [
            "--corpus-path" if argument == "--corpus-root" else argument
            for argument in sys.argv[2:]
        ]
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
