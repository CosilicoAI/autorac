"""Universal bytewise comparator for named semantic arrays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .refusal import Refusal

SEMANTIC_ARRAY_SORT_KEYS: dict[str, str | None] = {
    "gates": "gate_id",
    "required_gates": "gate_id",
    "acceptable_outcomes": None,
    "reasons": None,
    "eligible_records": None,
    "unused_eligible_records": None,
    "ineligible_records": "store_name",
    "coverage_assignment": "path",
    "delta": "path",
    "actions": "ref_spec",
    "containers": "image",
    "inventories": "path",
}


def _element_key(
    name: str,
    sort_key: str | None,
    item: object,
    index: int,
) -> str | Refusal:
    path = f"{name}[{index}]"
    if sort_key is None:
        if not isinstance(item, str):
            return Refusal("structural", path, "semantic_array_key_not_string")
        return item
    if not isinstance(item, Mapping) or sort_key not in item:
        return Refusal("structural", path, "semantic_array_key_missing")
    value = item[sort_key]
    if not isinstance(value, str):
        return Refusal("structural", path, "semantic_array_key_not_string")
    return value


def validate_semantic_array(
    name: str,
    items: Sequence[object],
) -> None | Refusal:
    """Require the registered key to be strictly increasing in UTF-8 bytes."""
    if name not in SEMANTIC_ARRAY_SORT_KEYS:
        return Refusal("structural", name, "unknown_semantic_array")

    sort_key = SEMANTIC_ARRAY_SORT_KEYS[name]
    previous: bytes | None = None
    for index, item in enumerate(items):
        key = _element_key(name, sort_key, item, index)
        if isinstance(key, Refusal):
            return key
        try:
            encoded = key.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            return Refusal(
                "structural",
                f"{name}[{index}]",
                "semantic_array_key_invalid_unicode",
            )
        if previous is not None:
            if encoded == previous:
                return Refusal(
                    "structural",
                    f"{name}[{index}]",
                    "semantic_array_duplicate_key",
                )
            if encoded < previous:
                return Refusal(
                    "structural",
                    f"{name}[{index}]",
                    "semantic_array_out_of_order",
                )
        previous = encoded
    return None
