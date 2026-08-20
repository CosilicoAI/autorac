"""Canonical JSON and digest helpers for notary bodies."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable

import rfc8785

from .refusal import Refusal


class _DuplicateMember(ValueError):
    """Internal signal for a duplicate decoded JSON member name."""


class _InvalidConstant(ValueError):
    """Internal signal for a non-JSON numeric constant."""


def jcs_dumps(obj: object) -> bytes:
    """Serialize *obj* with RFC 8785 JSON Canonicalization Scheme rules."""
    return rfc8785.dumps(obj)


def sha256_hex(data: bytes) -> str:
    """Return the lowercase hexadecimal SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def body_digest(obj: object) -> str:
    """Return the SHA-256 digest of an object's JCS serialization."""
    return sha256_hex(jcs_dumps(obj))


def _unique_object(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateMember
        result[key] = value
    return result


def _reject_constant(_value: str) -> object:
    raise _InvalidConstant


def _invalid_value_detail(value: object) -> str | None:
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            try:
                current.encode("utf-8", errors="strict")
            except UnicodeEncodeError:
                return "invalid_unicode"
        elif isinstance(current, float) and not math.isfinite(current):
            return "number_out_of_range"
        elif isinstance(current, dict):
            pending.extend(current.keys())
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return None


def strict_parse(data: bytes) -> object | Refusal:
    """Parse strict UTF-8 JSON, refusing duplicates and invalid Unicode."""
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return Refusal("structural", None, "invalid_utf8")

    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except _DuplicateMember:
        return Refusal("structural", None, "duplicate_json_member")
    except _InvalidConstant:
        return Refusal("structural", None, "non_json_number")
    except (json.JSONDecodeError, RecursionError, ValueError):
        return Refusal("structural", None, "invalid_json")

    invalid_detail = _invalid_value_detail(value)
    if invalid_detail is not None:
        return Refusal("structural", None, invalid_detail)
    return value


def is_canonical(data: bytes) -> bool:
    """Return whether *data* is its parsed value's exact JCS serialization."""
    value = strict_parse(data)
    if isinstance(value, Refusal):
        return False
    try:
        return jcs_dumps(value) == data
    except (
        rfc8785.CanonicalizationError,
        RecursionError,
        TypeError,
        UnicodeError,
    ):
        return False
