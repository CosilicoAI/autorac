"""Dispatch, ordering, and integrity tests for the parallel waiver audit."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode import cli

_RELEASE_IDENTITY = ("/corpus", "test-release", "0" * 64, "cHVibGljLWtleQ==")


def _release_stub():
    return SimpleNamespace(
        root=Path("/corpus"),
        name="test-release",
        content_sha256="0" * 64,
        public_key="cHVibGljLWtleQ==",
    )


class _FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


class _FakePool:
    """Deterministic in-process stand-in for ProcessPoolExecutor."""

    submitted: list[tuple] = []
    chunk_results: dict[str, list] = {}

    def __init__(self, max_workers):
        type(self).max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, chunk, *args):
        type(self).submitted.append((fn, tuple(chunk), args))
        return _FakeFuture(type(self).chunk_results[chunk[0]])


@pytest.fixture(autouse=True)
def _reset_fake_pool():
    _FakePool.submitted = []
    _FakePool.chunk_results = {}


def test_worker_count_defaults_to_cpu_bounded(monkeypatch):
    monkeypatch.delenv(cli._WAIVER_AUDIT_WORKERS_ENV, raising=False)
    with patch.object(cli.os, "cpu_count", return_value=4):
        assert cli._waiver_audit_worker_count(100) == 4
        assert cli._waiver_audit_worker_count(2) == 2
    with patch.object(cli.os, "cpu_count", return_value=32):
        assert cli._waiver_audit_worker_count(100) == 8


def test_worker_count_env_override(monkeypatch):
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, "3")
    assert cli._waiver_audit_worker_count(100) == 3
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, "0")
    assert cli._waiver_audit_worker_count(100) == 1
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, "1")
    assert cli._waiver_audit_worker_count(100) == 1
    # Overrides are hard-capped so a hostile value cannot fork-bomb the runner.
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, "3109")
    assert cli._waiver_audit_worker_count(3109) == cli._WAIVER_AUDIT_WORKERS_MAX


@pytest.mark.parametrize("raw", ["-1", "many"])
def test_worker_count_rejects_invalid_env(monkeypatch, raw):
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, raw)
    with pytest.raises(ValueError):
        cli._waiver_audit_worker_count(100)


def test_audit_partitions_are_exhaustive_disjoint_and_order_independent():
    classified = [(f"us/{index:03d}.yaml", "active") for index in range(17)]
    keys = '["us-ca", "us-ak", "us"]'
    partitions = [
        cli._validation_waiver_audit_partition(
            classified,
            partition_key=key,
            partition_keys_json=keys,
        )[0]
        for key in ("us", "us-ak", "us-ca")
    ]
    flattened = [item for partition in partitions for item in partition]
    assert sorted(flattened) == classified
    assert len(flattened) == len(set(flattened))
    assert cli._validation_waiver_audit_partition(
        classified,
        partition_key="us-ak",
        partition_keys_json='["us", "us-ca", "us-ak"]',
    )[0] == partitions[1]


@pytest.mark.parametrize(
    ("key", "keys", "message"),
    [
        ("us", "not-json", "valid JSON"),
        ("us", "[]", "non-empty JSON array"),
        ("us", '["us", "us"]', "duplicate"),
        ("us", '["us-ak"]', "occur exactly once"),
    ],
)
def test_audit_partition_rejects_ambiguous_manifests(key, keys, message):
    with pytest.raises(ValueError, match=message):
        cli._validation_waiver_audit_partition(
            [("us/a.yaml", "active")],
            partition_key=key,
            partition_keys_json=keys,
        )


@pytest.mark.parametrize(
    ("partition_key", "partition_keys_json"),
    [("us", None), (None, '["us"]')],
)
def test_audit_rejects_incomplete_partition_arguments(
    monkeypatch, tmp_path, partition_key, partition_keys_json
):
    import yaml as _yaml

    root = tmp_path / "rulespec-us"
    root.mkdir()
    ledger = {"validate_failures": {}}
    (root / "known-validation-gaps.yaml").write_text(_yaml.safe_dump(ledger))
    base = tmp_path / "base.yaml"
    base.write_text(_yaml.safe_dump(ledger))
    changed = tmp_path / "changed.txt"
    changed.write_text("")
    args = _audit_args(tmp_path, root, tmp_path / "c", tmp_path / "e", base, changed)
    args.partition_key = partition_key
    args.partition_keys_json = partition_keys_json

    with pytest.raises(ValueError, match="must be provided together"):
        cli._cmd_validation_waivers_audit(args)


def test_single_worker_delegates_to_serial(monkeypatch):
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, "1")
    sentinel = [{"path": "us/a.yaml", "passed": False, "fingerprint": "f"}]
    with patch.object(
        cli, "_fingerprint_validation_waiver_modules", return_value=sentinel
    ) as serial:
        result = cli._fingerprint_validation_waiver_modules_parallel(
            [Path("us/b.yaml"), Path("us/a.yaml")],
            root=Path("/repo"),
            corpus_path=Path("/corpus"),
            axiom_rules_path=Path("/engine"),
        )
    assert result == sentinel
    assert serial.call_count == 1
    # The serial path receives the same sorted order the parallel path uses.
    assert [str(p) for p in serial.call_args.args[0]] == ["us/a.yaml", "us/b.yaml"]


def _run_parallel(paths, monkeypatch, workers="2"):
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, workers)
    with (
        patch.object(
            cli, "load_rulespec_local_corpus_release", return_value=_release_stub()
        ),
        patch.object(cli.concurrent.futures, "ProcessPoolExecutor", _FakePool),
    ):
        return cli._fingerprint_validation_waiver_modules_parallel(
            [Path(p) for p in paths],
            root=Path("/repo"),
            corpus_path=Path("/corpus"),
            axiom_rules_path=Path("/engine"),
        )


def test_parallel_results_are_merged_and_sorted(monkeypatch):
    paths = [f"us/{index:03d}.yaml" for index in range(20)]
    # Contiguous chunks of the sorted list; return each chunk's rows reversed
    # to prove the merge re-establishes deterministic order.
    chunk = max(8, -(-len(paths) // (2 * 8)))
    for start in range(0, len(paths), chunk):
        piece = paths[start : start + chunk]
        _FakePool.chunk_results[piece[0]] = [
            {"path": path, "passed": False, "fingerprint": f"fp-{path}"}
            for path in reversed(piece)
        ]
    results = _run_parallel(list(reversed(paths)), monkeypatch)
    assert [row["path"] for row in results] == sorted(paths)
    submitted_paths = [
        path for _fn, chunk_paths, _args in _FakePool.submitted for path in chunk_paths
    ]
    assert submitted_paths == sorted(paths)
    for _fn, _chunk, args in _FakePool.submitted:
        assert args == (
            "/repo",
            "/corpus",
            "/engine",
            (),
            _RELEASE_IDENTITY,
        )


def test_parallel_detects_lost_results(monkeypatch):
    paths = [f"us/{index:03d}.yaml" for index in range(20)]
    chunk = max(8, -(-len(paths) // (2 * 8)))
    for start in range(0, len(paths), chunk):
        piece = paths[start : start + chunk]
        rows = [
            {"path": path, "passed": False, "fingerprint": f"fp-{path}"}
            for path in piece
        ]
        _FakePool.chunk_results[piece[0]] = rows[:-1] if start == 0 else rows
    with pytest.raises(RuntimeError, match="corrupted module set"):
        _run_parallel(paths, monkeypatch)


def test_parallel_detects_duplicate_masking_missing(monkeypatch):
    # Same cardinality, wrong multiset: one path duplicated, one omitted.
    paths = [f"us/{index:03d}.yaml" for index in range(20)]
    chunk = max(8, -(-len(paths) // (2 * 8)))
    for start in range(0, len(paths), chunk):
        piece = paths[start : start + chunk]
        rows = [
            {"path": path, "passed": False, "fingerprint": f"fp-{path}"}
            for path in piece
        ]
        if start == 0:
            rows[-1] = dict(rows[0])
        _FakePool.chunk_results[piece[0]] = rows
    with pytest.raises(RuntimeError, match="duplicated"):
        _run_parallel(paths, monkeypatch)


def test_parallel_propagates_worker_failure(monkeypatch):
    paths = [f"us/{index:03d}.yaml" for index in range(20)]
    chunk = max(8, -(-len(paths) // (2 * 8)))
    for start in range(0, len(paths), chunk):
        piece = paths[start : start + chunk]
        _FakePool.chunk_results[piece[0]] = (
            ValueError("worker exploded")
            if start == 0
            else [
                {"path": path, "passed": False, "fingerprint": f"fp-{path}"}
                for path in piece
            ]
        )
    with pytest.raises(ValueError, match="worker exploded"):
        _run_parallel(paths, monkeypatch)


def test_chunk_worker_builds_release_without_broker():
    sentinel = [{"path": "us/a.yaml", "passed": True, "fingerprint": "fp"}]
    built = {}

    def _fake_release(root, name, sha, key):
        built["identity"] = (str(root), name, sha, key)
        return "release-object"

    with (
        patch.object(cli, "LocalCorpusRelease", side_effect=_fake_release),
        patch.object(
            cli, "_fingerprint_validation_waiver_modules", return_value=sentinel
        ) as serial,
    ):
        result = cli._fingerprint_waiver_chunk(
            ["us/a.yaml"],
            "/repo",
            "/corpus",
            "/engine",
            (),
            _RELEASE_IDENTITY,
        )
    assert result == sentinel
    assert built["identity"] == _RELEASE_IDENTITY
    assert serial.call_args.kwargs["corpus_release"] == "release-object"


def test_fingerprint_batch_holds_no_shared_resolution_scope():
    # Partition-invariance regression: a batch-level resolution-cache scope
    # is reused by every nested per-module scope, so module A's resolution
    # state leaks into module B's failure messages and fingerprints become
    # partition-dependent (observed: 5/3,099 rulespec-us waiver fingerprints
    # changed between chunk layouts). The batch executor must leave scope
    # management to each ValidatorPipeline.validate call.
    calls = []
    routing_calls = []

    class _SpyScope:
        def __enter__(self):
            calls.append("entered")
            return self

        def __exit__(self, *exc):
            return False

    class _RoutingSpy:
        def __enter__(self):
            routing_calls.append("entered")
            return self

        def __exit__(self, *exc):
            return False

    with (
        patch.object(cli, "_rulespec_resolution_cache_scope", _SpyScope),
        patch.object(cli, "_rulespec_routing_cache_scope", _RoutingSpy),
        patch.object(
            cli, "_fingerprint_validation_waiver_modules_impl", return_value=[]
        ) as impl,
    ):
        result = cli._fingerprint_validation_waiver_modules(
            [Path("us/a.yaml")],
            root=Path("/repo"),
            corpus_path=Path("/corpus"),
            axiom_rules_path=Path("/engine"),
        )
    assert result == []
    assert impl.call_count == 1
    assert calls == []
    # The batch may share ROUTING admission (partition-invariant); it must
    # never share the resolution/identity cache.
    assert routing_calls == ["entered"]


def _audit_args(tmp_path, root, corpus, engine, base, changed):
    from types import SimpleNamespace

    return SimpleNamespace(
        root=root,
        corpus_path=corpus,
        axiom_rules_path=engine,
        protected_base=base,
        changed_paths=changed,
        json=False,
        rulespec_dependency_roots=(),
        rulespec_dependency_root=None,
    )


def test_audit_rechecks_discrepant_results_in_isolation(monkeypatch, capsys):
    # A load flake in the parallel pass must not become an accusation: the
    # audit re-executes any expectation-violating result alone and reports
    # on the isolated value.
    import yaml as _yaml

    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, "1")
    calls = []

    fp_good = "sha256:" + "a" * 64
    fp_flake = "sha256:" + "b" * 64

    def fake_parallel(modules, **kwargs):
        calls.append(("parallel", [str(m) for m in modules]))
        return [
            {
                "path": "us/statutes/x.yaml",
                "passed": False,
                "fingerprint": fp_flake,
                "outcome": {"validate": "flaky"},
            }
        ]

    def fake_serial(modules, **kwargs):
        calls.append(("serial", [str(m) for m in modules]))
        return [
            {
                "path": "us/statutes/x.yaml",
                "passed": False,
                "fingerprint": fp_good,
                "outcome": {"validate": "stable"},
            }
        ]

    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as td:
        root = _P(td) / "rulespec-us"
        (root / "us/statutes").mkdir(parents=True)
        (root / "us/statutes/x.yaml").write_text("format: rulespec/v1\n")
        ledger = {
            "validate_failures": {
                "us/statutes/x.yaml": {
                    "active": {
                        "fingerprint": fp_good,
                        "owner": "@MaxGhenis",
                        "issue": "https://github.com/TheAxiomFoundation/rulespec-us/issues/1",
                        "expires": (
                            __import__("datetime").date.today()
                            + __import__("datetime").timedelta(days=60)
                        ).isoformat(),
                    }
                }
            }
        }
        (root / "known-validation-gaps.yaml").write_text(_yaml.safe_dump(ledger))
        base = _P(td) / "base.yaml"
        base.write_text(_yaml.safe_dump(ledger))
        changed = _P(td) / "changed.txt"
        changed.write_text("")

        with (
            patch.object(
                cli, "_fingerprint_validation_waiver_modules_parallel", fake_parallel
            ),
            patch.object(cli, "_fingerprint_validation_waiver_modules", fake_serial),
        ):
            code = cli._cmd_validation_waivers_audit(
                _audit_args(
                    td, root, _P(td) / "corpus", _P(td) / "engine", base, changed
                )
            )

    assert code == 0
    assert [c[0] for c in calls] == ["parallel", "serial"]
    assert calls[1][1] == ["us/statutes/x.yaml"]
    out = capsys.readouterr().out
    assert "audit passed" in out


def test_audit_reports_discrepancy_that_survives_isolation(monkeypatch, capsys):
    import yaml as _yaml

    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, "1")
    fp_good = "sha256:" + "a" * 64
    fp_real_drift = "sha256:" + "c" * 64

    def fake_parallel(modules, **kwargs):
        return [
            {
                "path": "us/statutes/x.yaml",
                "passed": False,
                "fingerprint": fp_real_drift,
                "outcome": {"validate": "drifted"},
            }
        ]

    fake_serial = fake_parallel

    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as td:
        root = _P(td) / "rulespec-us"
        (root / "us/statutes").mkdir(parents=True)
        (root / "us/statutes/x.yaml").write_text("format: rulespec/v1\n")
        ledger = {
            "validate_failures": {
                "us/statutes/x.yaml": {
                    "active": {
                        "fingerprint": fp_good,
                        "owner": "@MaxGhenis",
                        "issue": "https://github.com/TheAxiomFoundation/rulespec-us/issues/1",
                        "expires": (
                            __import__("datetime").date.today()
                            + __import__("datetime").timedelta(days=60)
                        ).isoformat(),
                    }
                }
            }
        }
        (root / "known-validation-gaps.yaml").write_text(_yaml.safe_dump(ledger))
        base = _P(td) / "base.yaml"
        base.write_text(_yaml.safe_dump(ledger))
        changed = _P(td) / "changed.txt"
        changed.write_text("")

        with (
            patch.object(
                cli, "_fingerprint_validation_waiver_modules_parallel", fake_parallel
            ),
            patch.object(cli, "_fingerprint_validation_waiver_modules", fake_serial),
        ):
            code = cli._cmd_validation_waivers_audit(
                _audit_args(
                    td, root, _P(td) / "corpus", _P(td) / "engine", base, changed
                )
            )

    assert code == 1
    err = capsys.readouterr().err
    assert "fingerprint drifted" in err


def _ledger_for(paths_states, tmp_path):
    import datetime

    import yaml as _yaml

    expires = (datetime.date.today() + datetime.timedelta(days=60)).isoformat()
    entries = {}
    for path, states in paths_states.items():
        entry = {}
        for state, fp in states.items():
            entry[state] = {
                "fingerprint": fp,
                "owner": "@MaxGhenis",
                "issue": "https://github.com/TheAxiomFoundation/rulespec-us/issues/1",
                "expires": expires,
            }
        entries[path] = entry
    return _yaml.safe_dump({"validate_failures": entries})


@pytest.mark.parametrize(
    ("kind", "states", "row", "expect_recheck"),
    [
        # active must fail with the recorded fingerprint
        (
            "active-ok",
            {"active": "sha256:" + "a" * 64},
            {"passed": False, "fingerprint": "sha256:" + "a" * 64},
            False,
        ),
        (
            "active-stale",
            {"active": "sha256:" + "a" * 64},
            {"passed": True, "fingerprint": "sha256:" + "a" * 64},
            True,
        ),
        (
            "active-drift",
            {"active": "sha256:" + "a" * 64},
            {"passed": False, "fingerprint": "sha256:" + "b" * 64},
            True,
        ),
        # pending-only must pass
        (
            "pending-pass",
            {"pending": "sha256:" + "a" * 64},
            {"passed": True, "fingerprint": "sha256:" + "c" * 64},
            False,
        ),
        (
            "pending-fail",
            {"pending": "sha256:" + "a" * 64},
            {"passed": False, "fingerprint": "sha256:" + "c" * 64},
            True,
        ),
        # removed (base-active, absent from head) must pass
        (
            "removed-pass",
            "removed",
            {"passed": True, "fingerprint": "sha256:" + "c" * 64},
            False,
        ),
        (
            "removed-fail",
            "removed",
            {"passed": False, "fingerprint": "sha256:" + "c" * 64},
            True,
        ),
    ],
)
def test_recheck_selection_mirrors_verdicts(
    monkeypatch, kind, states, row, expect_recheck, tmp_path
):
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, "1")
    module_rel = "us/statutes/x.yaml"
    root = tmp_path / "rulespec-us"
    (root / "us/statutes").mkdir(parents=True)
    (root / "us/statutes/x.yaml").write_text("format: rulespec/v1\n")
    if states == "removed":
        head_states, base_states = {}, {module_rel: {"active": "sha256:" + "a" * 64}}
    else:
        head_states, base_states = {module_rel: states}, {module_rel: states}
    (root / "known-validation-gaps.yaml").write_text(_ledger_for(head_states, tmp_path))
    base = tmp_path / "base.yaml"
    base.write_text(_ledger_for(base_states, tmp_path))
    changed = tmp_path / "changed.txt"
    changed.write_text("")

    serial_calls = []

    def fake_parallel(modules, **kwargs):
        return [dict(row, path=module_rel, outcome={"v": 1})]

    def fake_serial(modules, **kwargs):
        serial_calls.append([str(m) for m in modules])
        return [dict(row, path=module_rel, outcome={"v": 1})]

    with (
        patch.object(
            cli, "_fingerprint_validation_waiver_modules_parallel", fake_parallel
        ),
        patch.object(cli, "_fingerprint_validation_waiver_modules", fake_serial),
    ):
        cli._cmd_validation_waivers_audit(
            _audit_args(tmp_path, root, tmp_path / "c", tmp_path / "e", base, changed)
        )
    assert bool(serial_calls) == expect_recheck, kind


def test_recheck_report_flags_and_counts(monkeypatch, capsys, tmp_path):
    import json as _json

    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, "1")
    module_rel = "us/statutes/x.yaml"
    fp_good = "sha256:" + "a" * 64
    fp_flake = "sha256:" + "b" * 64
    root = tmp_path / "rulespec-us"
    (root / "us/statutes").mkdir(parents=True)
    (root / "us/statutes/x.yaml").write_text("format: rulespec/v1\n")
    (root / "known-validation-gaps.yaml").write_text(
        _ledger_for({module_rel: {"active": fp_good}}, tmp_path)
    )
    base = tmp_path / "base.yaml"
    base.write_text(_ledger_for({module_rel: {"active": fp_good}}, tmp_path))
    changed = tmp_path / "changed.txt"
    changed.write_text("")

    def fake_parallel(modules, **kwargs):
        return [
            {
                "path": module_rel,
                "passed": False,
                "fingerprint": fp_flake,
                "outcome": {"v": "flake"},
            }
        ]

    def fake_serial(modules, **kwargs):
        return [
            {
                "path": module_rel,
                "passed": False,
                "fingerprint": fp_good,
                "outcome": {"v": "stable"},
            }
        ]

    args = _audit_args(tmp_path, root, tmp_path / "c", tmp_path / "e", base, changed)
    args.json = True
    with (
        patch.object(
            cli, "_fingerprint_validation_waiver_modules_parallel", fake_parallel
        ),
        patch.object(cli, "_fingerprint_validation_waiver_modules", fake_serial),
    ):
        code = cli._cmd_validation_waivers_audit(args)
    report = _json.loads(capsys.readouterr().out)
    assert code == 0 and report["success"] is True
    assert report["checked"] == 1
    (row_record,) = report["results"]
    assert row_record["isolated_recheck"] is True
    assert row_record["fingerprint"] == fp_good
    assert row_record["success"] is True


def test_systemic_discrepancy_fails_without_per_row_accusations(
    monkeypatch, capsys, tmp_path
):
    import json as _json

    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, "1")
    fp_good = "sha256:" + "a" * 64
    fp_bad = "sha256:" + "b" * 64
    count = 40  # > max(16, 1% of 40)
    root = tmp_path / "rulespec-us"
    (root / "us/statutes").mkdir(parents=True)
    states = {}
    for index in range(count):
        rel = f"us/statutes/m{index:03d}.yaml"
        (root / rel).write_text("format: rulespec/v1\n")
        states[rel] = {"active": fp_good}
    (root / "known-validation-gaps.yaml").write_text(_ledger_for(states, tmp_path))
    base = tmp_path / "base.yaml"
    base.write_text(_ledger_for(states, tmp_path))
    changed = tmp_path / "changed.txt"
    changed.write_text("")

    def fake_parallel(modules, **kwargs):
        return [
            {"path": str(m), "passed": False, "fingerprint": fp_bad, "outcome": {}}
            for m in modules
        ]

    def fake_serial(modules, **kwargs):
        raise AssertionError("systemic mode must not recheck serially")

    args = _audit_args(tmp_path, root, tmp_path / "c", tmp_path / "e", base, changed)
    args.json = True
    with (
        patch.object(
            cli, "_fingerprint_validation_waiver_modules_parallel", fake_parallel
        ),
        patch.object(cli, "_fingerprint_validation_waiver_modules", fake_serial),
    ):
        code = cli._cmd_validation_waivers_audit(args)
    report = _json.loads(capsys.readouterr().out)
    assert code == 1 and report["success"] is False
    assert len(report["errors"]) == 1
    assert "systemic audit discrepancy" in report["errors"][0]
    assert all(r.get("unverified_discrepancy") for r in report["results"])
    assert not any("error" in r for r in report["results"])
