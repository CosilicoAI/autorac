"""Tests for strict JSON parsing and RFC 8785 serialization."""

from __future__ import annotations

import hashlib

import pytest

from axiom_encode.notary.canonical import (
    body_digest,
    is_canonical,
    jcs_dumps,
    sha256_hex,
    strict_parse,
)
from axiom_encode.notary.refusal import Refusal

RFC_SERIALIZATION = r"""{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],"string":"€$\u000f\nA'B\"\\\"/"}""".encode()
RFC_PROPERTY_ORDER = (
    '{"\\r":"Carriage Return","1":"One",'
    '"\u0080":"Control","ö":"Latin Small Letter O With Diaeresis",'
    '"€":"Euro Sign","😀":"Emoji: Grinning Face",'
    '"דּ":"Hebrew Letter Dalet With Dagesh"}'
).encode()


def test_digest_helpers_use_jcs_bytes() -> None:
    obj = {"z": 1, "a": [True, None]}
    canonical = b'{"a":[true,null],"z":1}'

    assert jcs_dumps(obj) == canonical
    assert sha256_hex(canonical) == hashlib.sha256(canonical).hexdigest()
    assert body_digest(obj) == hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize("data", [RFC_SERIALIZATION, RFC_PROPERTY_ORDER])
def test_rfc_8785_vectors_are_canonical(data: bytes) -> None:
    assert is_canonical(data)
    assert jcs_dumps(strict_parse(data)) == data


@pytest.mark.parametrize(
    "data",
    [
        b'{"member":1,"member":2}',
        b'{"member":1,"\\u006dember":2}',
        b'{"outer":{"member":1,"member":2}}',
    ],
)
def test_duplicate_json_member_is_refused(data: bytes) -> None:
    assert strict_parse(data) == Refusal(
        "structural",
        None,
        "duplicate_json_member",
    )


@pytest.mark.parametrize(
    ("data", "detail"),
    [
        (b'"\xff"', "invalid_utf8"),
        (rb'"\ud800"', "invalid_unicode"),
        (rb'"\ude00"', "invalid_unicode"),
        (rb'{"\ud800":"value"}', "invalid_unicode"),
    ],
)
def test_invalid_unicode_is_refused(data: bytes, detail: str) -> None:
    assert strict_parse(data) == Refusal("structural", None, detail)
    assert not is_canonical(data)


def test_valid_escaped_surrogate_pair_is_accepted() -> None:
    assert strict_parse(rb'"\ud83d\ude00"') == "😀"


@pytest.mark.parametrize("data", [b"1e999", b"[1e999]", b'{"value":1e999}'])
def test_out_of_range_number_is_refused_at_every_depth(data: bytes) -> None:
    assert strict_parse(data) == Refusal(
        "structural",
        None,
        "number_out_of_range",
    )


@pytest.mark.parametrize(
    "data",
    [
        b'{ "a":1}',
        b'{"b":2,"a":1}',
        b"1.0",
        b'{"a":1}\n',
    ],
)
def test_noncanonical_bytes_are_detected(data: bytes) -> None:
    assert not is_canonical(data)


def test_jcs_preserves_array_order() -> None:
    assert jcs_dumps(["z", "a"]) == b'["z","a"]'
    assert is_canonical(b'["z","a"]')
