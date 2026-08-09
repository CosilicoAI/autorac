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

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
