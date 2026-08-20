"""Tests for every registered semantic-array comparator."""

from __future__ import annotations

import pytest

from axiom_encode.notary.comparator import (
    SEMANTIC_ARRAY_SORT_KEYS,
    validate_semantic_array,
)
from axiom_encode.notary.refusal import Refusal

EXPECTED_REGISTRY = {
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


def _items(name: str, keys: list[str]) -> list[object]:
    sort_key = SEMANTIC_ARRAY_SORT_KEYS[name]
    if sort_key is None:
        return list(keys)
    return [{sort_key: key, "non_key_payload": index} for index, key in enumerate(keys)]


def test_semantic_array_registry_is_exact() -> None:
    assert SEMANTIC_ARRAY_SORT_KEYS == EXPECTED_REGISTRY


@pytest.mark.parametrize("name", EXPECTED_REGISTRY)
def test_each_semantic_array_accepts_utf8_byte_order(name: str) -> None:
    assert validate_semantic_array(name, _items(name, ["\ue000", "\U00010000"])) is None


@pytest.mark.parametrize("name", EXPECTED_REGISTRY)
def test_each_semantic_array_refuses_out_of_order_keys(name: str) -> None:
    result = validate_semantic_array(name, _items(name, ["\U00010000", "\ue000"]))

    assert isinstance(result, Refusal)
    assert result.detail == "semantic_array_out_of_order"


@pytest.mark.parametrize("name", EXPECTED_REGISTRY)
def test_each_semantic_array_refuses_duplicate_keys(name: str) -> None:
    result = validate_semantic_array(name, _items(name, ["same", "same"]))

    assert isinstance(result, Refusal)
    assert result.detail == "semantic_array_duplicate_key"


def test_unknown_semantic_array_is_refused() -> None:
    assert validate_semantic_array("unknown", []) == Refusal(
        "structural",
        "unknown",
        "unknown_semantic_array",
    )
