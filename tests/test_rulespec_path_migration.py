from __future__ import annotations

import json
from pathlib import Path

import pytest

from axiom_encode.rulespec_path_migration import (
    PLAN_SCHEMA,
    PathMigrationPlanError,
    PlannedMove,
    canonical_destination,
    companion_path,
    exact_reference_replacements,
    load_plan_bytes,
    rewrite_exact_references,
    rulespec_identity,
)

BASE = "a" * 40


def _plan(*moves: tuple[str, str], **extra: object) -> bytes:
    payload: dict[str, object] = {
        "schema_version": PLAN_SCHEMA,
        "base_commit": BASE,
        "moves": [{"from": old, "to": new} for old, new in moves],
        **extra,
    }
    return json.dumps(payload).encode()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("us-la/statutes/47:294.yaml", "us-la/statutes/47/294.yaml"),
        (
            "us-nh/regulations/he-w-700/He-W 704/04.yaml",
            "us-nh/regulations/he-w-700/he-w-704/04.yaml",
        ),
        (
            "us/statutes/42/1437c\u20131.yaml",
            "us/statutes/42/1437c-1.yaml",
        ),
    ],
)
def test_unique_canonical_destination(source: str, expected: str) -> None:
    assert canonical_destination(Path(source)) == Path(expected)


def test_loads_minimal_exact_plan_and_hashes_canonical_payload() -> None:
    loaded = load_plan_bytes(
        _plan(("us-la/statutes/47:294.yaml", "us-la/statutes/47/294.yaml"))
    )

    assert loaded.base_commit == BASE
    assert loaded.moves == (
        PlannedMove(
            source=Path("us-la/statutes/47:294.yaml"),
            destination=Path("us-la/statutes/47/294.yaml"),
        ),
    )
    assert len(loaded.sha256) == 64
    assert loaded.canonical_bytes == (
        b'{"base_commit":"' + BASE.encode() + b'","moves":[{"from":'
        b'"us-la/statutes/47:294.yaml","to":"us-la/statutes/47/294.yaml"}],'
        b'"schema_version":"axiom-encode/rulespec-path-migration-plan/v1"}'
    )


@pytest.mark.parametrize(
    "raw",
    [
        _plan(
            ("us-la/statutes/47:294.yaml", "us-la/statutes/47/294.yaml"),
            reason="too much authority",
        ),
        _plan(("us-la/statutes/47:294.yaml", "us-la/statutes/47-294.yaml")),
        _plan(("us-la/statutes/47/294.yaml", "us-la/statutes/47/294.yaml")),
        _plan(("us-la/statutes/47:294.test.yaml", "us-la/statutes/47/294.test.yaml")),
        _plan(("us-la/statutes/../47:294.yaml", "us-la/statutes/47/294.yaml")),
        _plan(
            ("us-la/statutes/47:294.yaml", "us-la/statutes/47/294.yaml"),
            ("us-la/statutes/47:295.yaml", "us-la/statutes/47/294.yaml"),
        ),
    ],
)
def test_rejects_overbroad_unsafe_or_colliding_plan(raw: bytes) -> None:
    with pytest.raises(PathMigrationPlanError):
        load_plan_bytes(raw)


def test_derives_companion_and_only_plural_rulespec_identity_rewrites() -> None:
    move = PlannedMove(
        Path("us-la/statutes/47:32.yaml"),
        Path("us-la/statutes/47/32.yaml"),
    )

    replacements = exact_reference_replacements(
        [move],
        existing_companions={companion_path(move.source)},
    )

    assert replacements == {
        "us-la/statutes/47:32.yaml": "us-la/statutes/47/32.yaml",
        "us-la:statutes/47:32": "us-la:statutes/47/32",
        "us-la/statutes/47:32.test.yaml": "us-la/statutes/47/32.test.yaml",
    }
    assert rulespec_identity(move.source) == "us-la:statutes/47:32"
    assert "us-la/statute/47:32" not in replacements


def test_rewrites_exact_durable_references_but_preserves_legal_citation() -> None:
    raw = (
        b"import: us-la:statutes/47:32#rate\n"
        b"path: us-la/statutes/47:32.yaml\n"
        b"other_import: us-la:statutes/47:320#rate\n"
        b"other_path: archive-us-la/statutes/47:32.yaml.bak\n"
        b"corpus_citation_path: us-la/statute/47:32\n"
    )

    rewritten, counts = rewrite_exact_references(
        raw,
        {
            "us-la:statutes/47:32": "us-la:statutes/47/32",
            "us-la/statutes/47:32.yaml": "us-la/statutes/47/32.yaml",
        },
    )

    assert rewritten == (
        b"import: us-la:statutes/47/32#rate\n"
        b"path: us-la/statutes/47/32.yaml\n"
        b"other_import: us-la:statutes/47:320#rate\n"
        b"other_path: archive-us-la/statutes/47:32.yaml.bak\n"
        b"corpus_citation_path: us-la/statute/47:32\n"
    )
    assert counts == (
        {
            "from": "us-la/statutes/47:32.yaml",
            "to": "us-la/statutes/47/32.yaml",
            "count": 1,
        },
        {
            "from": "us-la:statutes/47:32",
            "to": "us-la:statutes/47/32",
            "count": 1,
        },
    )


def test_non_utf8_without_reference_is_ignored_but_match_fails_closed() -> None:
    replacements = {"old/path": "new/path"}
    assert rewrite_exact_references(b"\xff", replacements) == (b"\xff", ())
    with pytest.raises(PathMigrationPlanError, match="non-UTF-8"):
        rewrite_exact_references(b"\xffold/path", replacements)
