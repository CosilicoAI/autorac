"""Tests for axiom_encode.run_log.v1 - schema, emission, backfill, publication."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from axiom_encode import __version__
from axiom_encode import run_log as rl
from axiom_encode import run_log_export as rx
from tests.eval_evidence_fixtures import (
    TEST_APPLY_PRIVATE_KEY_B64,
    TEST_APPLY_PUBLIC_KEY_B64,
)
from tests.signing_broker_fixtures import SigningBrokerFixture

# ---------------------------------------------------------------------------
# Pipeline spec / DAG
# ---------------------------------------------------------------------------


def test_pipeline_spec_dag_is_well_formed():
    spec = rl.pipeline_spec_dict()
    assert spec["pipeline_spec_version"] == rl.PIPELINE_SPEC_VERSION
    assert spec["run_log_schema"] == rl.SCHEMA_VERSION
    ids = {stage["id"] for stage in spec["stages"]}
    assert ids == set(rl.STAGE_IDS)
    # Every dependency edge points at a declared stage.
    for stage in spec["stages"]:
        for dep in stage["depends_on"]:
            assert dep in ids, f"{stage['id']} depends on undeclared {dep}"
    # Every funnel step references a declared stage.
    for step in spec["funnel_steps"]:
        assert step["stage"] in ids


def test_pipeline_spec_declares_forward_compatible_judge_stage():
    ids = {s.id for s in rl.PIPELINE_SPEC}
    assert "judge" in ids
    judge = next(s for s in rl.PIPELINE_SPEC if s.id == "judge")
    assert judge.category == rl.StageCategory.judge
    assert judge.optional is True


# ---------------------------------------------------------------------------
# Event validation + round-trip
# ---------------------------------------------------------------------------


def _event(**kw):
    base = dict(run_id="r1", seq=0, ts="2026-07-08T00:00:00+00:00", status="passed")
    base.update(kw)
    return rl.RunLogEvent(**base)


def test_event_round_trip_is_lossless():
    event = _event(
        stage="gate.oracle",
        status="failed",
        reason_code="oracle_mismatch",
        reason="score 0.9",
        duration_ms=1200,
        attrs={"oracle": "policyengine", "score": 0.9},
        findings=[
            rl.Finding(code="oracle_mismatch", severity="important", message="x")
        ],
    )
    line = event.to_json_line()
    restored = rl.RunLogEvent.from_json_line(line)
    assert restored.to_dict() == event.to_dict()
    # Wire uses "schema", not the python alias "schema_".
    assert json.loads(line)["schema"] == rl.SCHEMA_VERSION
    assert "schema_" not in json.loads(line)


def test_unknown_stage_fails_validation():
    with pytest.raises(ValidationError):
        _event(stage="not_a_real_stage")


def test_unknown_stage_fails_validation_from_dict():
    payload = {
        "schema": rl.SCHEMA_VERSION,
        "run_id": "r1",
        "seq": 0,
        "ts": "2026-07-08T00:00:00+00:00",
        "stage": "totally_made_up",
        "status": "passed",
    }
    with pytest.raises(ValidationError):
        rl.RunLogEvent.from_dict(payload)


def test_unknown_schema_fails_validation():
    with pytest.raises(ValidationError):
        rl.RunLogEvent.from_dict(
            {
                "schema": "axiom_encode.run_log.v999",
                "run_id": "r1",
                "seq": 0,
                "ts": "2026-07-08T00:00:00+00:00",
                "stage": "generate",
                "status": "passed",
            }
        )


def test_unknown_status_fails_validation():
    with pytest.raises(ValidationError):
        _event(stage="generate", status="maybe")


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        _event(stage="generate", not_a_field=1)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def test_writer_appends_and_sequences(tmp_path):
    writer = rl.RunLogWriter("run42", log_dir=tmp_path)
    writer.emit("generate", "passed")
    writer.emit("gate.ci", "failed", reason_code="ci_failure")
    lines = (tmp_path / "run42.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert [json.loads(x)["seq"] for x in lines] == [0, 1]
    # Re-opening continues the sequence rather than restarting.
    writer2 = rl.RunLogWriter("run42", log_dir=tmp_path)
    writer2.emit("apply", "passed")
    lines = (tmp_path / "run42.jsonl").read_text().strip().splitlines()
    assert [json.loads(x)["seq"] for x in lines] == [0, 1, 2]


def test_writer_is_non_fatal_on_bad_stage(tmp_path):
    writer = rl.RunLogWriter("run42", log_dir=tmp_path)
    result = writer.emit("nonexistent_stage", "passed")
    assert result is None
    assert writer.last_error is not None
    # A bad event does not create a corrupt line.
    assert (
        not (tmp_path / "run42.jsonl").exists()
        or (tmp_path / "run42.jsonl").read_text() == ""
    )


def test_writer_disabled_when_no_run_id(tmp_path):
    writer = rl.RunLogWriter("", log_dir=tmp_path)
    assert writer.emit("generate", "passed") is None


# ---------------------------------------------------------------------------
# Fold
# ---------------------------------------------------------------------------


def test_fold_run_computes_funnel_and_first_failure():
    events = [
        _event(
            seq=0,
            stage="generate",
            status="passed",
            attrs={"citation": "c", "backend": "codex", "model": "m"},
        ),
        _event(seq=1, stage="gate.ci", status="passed", duration_ms=1000),
        _event(
            seq=2, stage="gate.oracle", status="failed", reason_code="oracle_mismatch"
        ),
        _event(seq=3, stage="apply", status="passed"),
    ]
    record = rl.fold_run(events)
    assert record.funnel["generated"] is True
    assert record.funnel["gates_passed"] is True
    assert record.funnel["applied"] is True
    assert record.funnel["judged"] is False
    assert record.funnel["merged"] is False
    assert record.first_failure_stage == "gate.oracle"
    assert record.first_failure_reason_code == "oracle_mismatch"
    assert record.citation == "c"
    assert record.stage_duration_ms["gate.ci"] == 1000


def test_fold_run_terminal_status_wins_over_started():
    events = [
        _event(seq=0, stage="gate.ci", status="started"),
        _event(seq=1, stage="gate.ci", status="passed"),
    ]
    record = rl.fold_run(events)
    assert record.stage_status["gate.ci"] == "passed"


# ---------------------------------------------------------------------------
# Backfill (never fabricates un-sourced stages)
# ---------------------------------------------------------------------------


def _fake_run(**kw):
    from datetime import datetime, timezone

    base = dict(
        id="abc12345",
        timestamp=datetime(2026, 7, 8, tzinfo=timezone.utc),
        citation="us-ct/statute/17b-104",
        agent_type="codex:encoder",
        agent_model="gpt-5.5",
        total_duration_ms=5000,
        review_results=None,
        outcome={},
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_backfill_emits_generate_with_backfilled_flag():
    events = rx.synthesize_backfill_events(_fake_run(), None)
    generate = next(e for e in events if e.stage == "generate")
    assert generate.status == rl.StageStatus.passed.value
    assert generate.attrs["backfilled"] is True
    assert generate.attrs["citation"] == "us-ct/statute/17b-104"


def test_backfill_preserves_escalation_metadata_and_aggregate_duration():
    escalation = {
        "escalated": True,
        "failed_attempt_count": 2,
        "initial_model": "gpt-5.6-terra",
        "escalation_model": "gpt-5.6-sol",
    }
    events = rx.synthesize_backfill_events(
        _fake_run(
            total_duration_ms=12600,
            outcome={"generation_escalation": escalation},
        ),
        None,
    )
    generate = next(event for event in events if event.stage == "generate")
    assert generate.duration_ms == 12600
    assert generate.attrs["generation_escalation"] == escalation


def test_backfill_never_fabricates_downstream_or_judge_stages():
    run = _fake_run(
        outcome={
            "final_success": True,
            "status": "apply_applied",
            "standalone_validation_success": True,
        }
    )
    events = rx.synthesize_backfill_events(run, None)
    stages = {e.stage for e in events}
    # These were never captured historically; they must be absent, not guessed.
    assert "judge" not in stages
    assert "pr" not in stages
    assert "merge" not in stages
    assert "oracle_at_merge" not in stages
    assert "gate.compile" not in stages  # only gate.ci is source-backed historically
    assert "gate.ci" in stages
    assert "apply" in stages


def test_backfill_marks_sourced_generation_failure():
    # `apply_blocked_generation` is the one recorded signal that generation
    # itself failed; backfill must mark generate=failed rather than over-claim.
    run = _fake_run(
        outcome={"status": "apply_blocked_generation", "final_success": False}
    )
    events = rx.synthesize_backfill_events(run, None)
    generate = next(e for e in events if e.stage == "generate")
    assert generate.status == rl.StageStatus.failed.value
    assert generate.reason_code == "generation_failed"


def test_backfill_generate_passes_when_generation_reached_later_stages():
    for status in ("apply_applied", "apply_blocked_validation", "standalone_failed"):
        run = _fake_run(outcome={"status": status})
        generate = next(
            e for e in rx.synthesize_backfill_events(run, None) if e.stage == "generate"
        )
        assert generate.status == rl.StageStatus.passed.value, status


def test_backfill_uses_manifest_provenance_when_present(tmp_path):
    manifest = {
        "run_id": "abc12345",
        "backend": "codex",
        "model": "gpt-5.5",
        "generation_prompt_sha256": "deadbeef",
        "trace_file": "/tmp/trace.json",
        "trace_sha256": "cafef00d",
        "context_manifest_file": "/tmp/ctx.json",
        "context_manifest_sha256": "0011",
        "signature": {"key_id": "sha256:" + "a" * 64},
        "applied_files": [{"path": "statutes/17b-104.yaml", "sha256": "aa"}],
        "_manifest_path": "/tmp/m.json",
        "_manifest_sha256": "ff00",
    }
    events = rx.synthesize_backfill_events(
        _fake_run(outcome={"final_success": True}), manifest
    )
    generate = next(e for e in events if e.stage == "generate")
    assert generate.attrs["generation_prompt_sha256"] == "deadbeef"
    assert generate.attrs["trace_sha256"] == "cafef00d"
    apply_event = next(e for e in events if e.stage == "apply")
    assert apply_event.attrs["signature_key_id"] == "sha256:" + "a" * 64
    assert apply_event.attrs["manifest_sha_chain"]


def test_backfill_oracle_exact_match_convention():
    from axiom_encode.harness.encoding_db import ReviewResults

    run = _fake_run(review_results=ReviewResults(reviews=[], policyengine_match=1.0))
    events = rx.synthesize_backfill_events(run, None)
    oracle = next(e for e in events if e.stage == "gate.oracle")
    assert oracle.status == rl.StageStatus.passed.value
    assert oracle.attrs["score"] == 1.0

    run2 = _fake_run(review_results=ReviewResults(reviews=[], policyengine_match=0.5))
    oracle2 = next(
        e for e in rx.synthesize_backfill_events(run2, None) if e.stage == "gate.oracle"
    )
    assert oracle2.status == rl.StageStatus.failed.value
    assert oracle2.reason_code == "oracle_mismatch"


# ---------------------------------------------------------------------------
# Live emission (rich per-gate verdicts from EvalArtifactMetrics)
# ---------------------------------------------------------------------------


def test_live_emission_records_real_gate_verdicts(tmp_path):
    metrics = SimpleNamespace(
        compile_pass=True,
        compile_issues=[],
        ci_pass=False,
        ci_issues=["test x failed"],
        embedded_source_present=True,
        grounded_numeric_count=3,
        ungrounded_numeric_count=0,
        numeric_occurrence_issues=[],
        missing_source_numeric_occurrence_count=0,
        generalist_review_pass=None,
        policyengine_pass=None,
    )
    result = SimpleNamespace(
        success=False,
        output_file="/tmp/out.yaml",
        error=None,
        citation="us-ct/statute/17b-104",
        backend="codex",
        model="gpt-5.5",
        runner="codex-gpt-5.5",
        generation_prompt_sha256="abc",
        trace_file="/tmp/trace.json",
        context_manifest_file="/tmp/ctx.json",
        retry_count=0,
        duration_ms=4200,
        metrics=metrics,
    )
    writer = rx.emit_live_encode_events(
        result,
        "liverun1",
        {
            "final_success": False,
            "status": "apply_blocked_validation",
            "generation_escalation": {
                "escalated": True,
                "failed_attempt_count": 2,
                "initial_model": "gpt-5.6-terra",
                "escalation_model": "gpt-5.6-sol",
            },
        },
        total_duration_ms=12600,
        log_dir=tmp_path,
    )
    assert writer.last_error is None
    events = list(rl.iter_events(tmp_path / "liverun1.jsonl"))
    by_stage = {e.stage: e for e in events}
    assert by_stage["gate.compile"].status == "passed"
    assert by_stage["gate.ci"].status == "failed"
    assert by_stage["gate.ci"].reason_code == "ci_failure"
    assert by_stage["gate.grounding"].status == "passed"
    assert by_stage["apply"].status == "failed"
    assert by_stage["generate"].duration_ms == 12600
    assert by_stage["generate"].attrs["generation_escalation"] == {
        "escalated": True,
        "failed_attempt_count": 2,
        "initial_model": "gpt-5.6-terra",
        "escalation_model": "gpt-5.6-sol",
    }
    # Live logs are NOT flagged as backfill.
    assert by_stage["generate"].attrs.get("backfilled") is None


# ---------------------------------------------------------------------------
# Export + publish + staleness (end to end over a tiny sqlite db)
# ---------------------------------------------------------------------------


def _make_db(path, rows):
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE encoding_runs (id TEXT, timestamp TEXT, citation TEXT, "
        "agent_type TEXT, agent_model TEXT, total_duration_ms INTEGER, "
        "review_results_json TEXT, outcome_json TEXT)"
    )
    conn.executemany("INSERT INTO encoding_runs VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def test_export_publish_and_staleness_end_to_end(tmp_path):
    db = tmp_path / "encodings.db"
    _make_db(
        db,
        [
            (
                "run0001",
                "2026-07-01T00:00:00+00:00",
                "us/statute/a",
                "codex:encoder",
                "gpt-5.5",
                1000,
                json.dumps(
                    {
                        "reviews": [{"reviewer": "rulespec", "passed": True}],
                        "policyengine_match": 1.0,
                    }
                ),
                json.dumps(
                    {
                        "final_success": True,
                        "status": "apply_applied",
                        "standalone_validation_success": True,
                    }
                ),
            ),
            (
                "run0002",
                "2026-07-02T00:00:00+00:00",
                "us/statute/b",
                "codex:encoder",
                "gpt-5.5",
                2000,
                None,
                json.dumps(
                    {
                        "final_success": False,
                        "status": "apply_blocked_validation",
                        "standalone_validation_success": False,
                    }
                ),
            ),
        ],
    )
    log_dir = tmp_path / "logs"
    report = rx.export_backfill(db, [], log_dir=log_dir, limit=100)
    assert report["total_runs"] == 2
    assert report["exported"] == 2
    assert report["coverage_pct"]["run_id"] == 100.0
    assert report["coverage_pct"]["judge"] == 0.0  # the gap

    out_dir = tmp_path / "data"
    meta = rx.publish(log_dir, out_dir)
    assert meta["run_count"] == 2
    runs_index = json.loads((out_dir / rx.RUNS_INDEX_FILE).read_text())
    assert runs_index["run_count"] == 2
    # Aggregates precomputed for the renderer.
    buckets = {f["bucket"]: f["count"] for f in runs_index["aggregates"]["funnel"]}
    assert buckets["generated"] == 2
    assert buckets["applied"] == 1  # only run0001 applied
    assert buckets["merged"] == 0  # never captured
    assert (out_dir / rx.PIPELINE_SPEC_FILE).exists()

    # Fresh immediately after publish.
    status = rx.check_staleness(out_dir, log_dir)
    assert status["ok"] is True
    assert status["reason"] == "fresh"


def test_staleness_guard_flags_never_published(tmp_path):
    log_dir = tmp_path / "logs"
    writer = rl.RunLogWriter("r1", log_dir=log_dir)
    writer.emit("generate", "passed")
    status = rx.check_staleness(tmp_path / "empty_data", log_dir)
    assert status["ok"] is False
    assert status["reason"] == "never_published"


def test_backfill_preserves_live_logs(tmp_path):
    db = tmp_path / "encodings.db"
    _make_db(
        db,
        [
            (
                "runlive",
                "2026-07-01T00:00:00+00:00",
                "us/statute/a",
                "codex:encoder",
                "gpt-5.5",
                1000,
                None,
                json.dumps({"final_success": True}),
            )
        ],
    )
    log_dir = tmp_path / "logs"
    # Simulate a rich live log already on disk for this run.
    writer = rl.RunLogWriter("runlive", log_dir=log_dir)
    writer.emit("generate", "passed", attrs={"citation": "us/statute/a"})
    writer.emit("gate.compile", "passed")  # a verdict backfill cannot source
    rx.export_backfill(db, [], log_dir=log_dir, limit=100)
    events = list(rl.iter_events(log_dir / "runlive.jsonl"))
    stages = {e.stage for e in events}
    # The richer live gate.compile survived; backfill did not clobber it.
    assert "gate.compile" in stages


# ---------------------------------------------------------------------------
# Robustness / correctness regressions (from independent review)
# ---------------------------------------------------------------------------


def test_fold_run_retry_pass_is_not_double_counted():
    # A gate that failed then passed on retry must count as passed in the funnel
    # and NOT appear as the run's first failure.
    events = [
        _event(seq=0, stage="generate", status="passed"),
        _event(seq=1, stage="gate.ci", status="failed", reason_code="ci_failure"),
        _event(seq=2, stage="gate.ci", status="passed"),
        _event(seq=3, stage="apply", status="passed"),
    ]
    record = rl.fold_run(events)
    assert record.stage_status["gate.ci"] == "passed"
    assert record.funnel["gates_passed"] is True
    assert record.first_failure_stage is None
    assert record.first_failure_reason_code is None


def test_fold_run_first_failure_follows_pipeline_order():
    # Two stages end failed; the first in pipeline order is reported.
    events = [
        _event(seq=0, stage="apply", status="failed", reason_code="apply_blocked"),
        _event(seq=1, stage="gate.ci", status="failed", reason_code="ci_failure"),
    ]
    record = rl.fold_run(events)
    assert record.first_failure_stage == "gate.ci"
    assert record.first_failure_reason_code == "ci_failure"


def test_backfill_survives_malformed_timestamp_row(tmp_path):
    db = tmp_path / "encodings.db"
    _make_db(
        db,
        [
            ("bad1", "not-a-date", "us/a", "codex", "gpt-5.5", 0, None, "{}"),
            (
                "good1",
                "2026-07-02T00:00:00+00:00",
                "us/b",
                "codex",
                "gpt-5.5",
                0,
                None,
                json.dumps({"final_success": True}),
            ),
        ],
    )
    report = rx.export_backfill(db, [], log_dir=tmp_path / "logs", limit=100)
    # The malformed-timestamp row does not abort the backfill.
    assert report["total_runs"] == 2
    assert report["exported"] == 2


def test_build_manifest_index_rejects_unverified_json(tmp_path):
    repo = tmp_path / "rulespec-us"
    mdir = repo / ".axiom" / "encoding-manifests"
    mdir.mkdir(parents=True)
    (mdir / "list.json").write_text("[1, 2, 3]")  # valid JSON, not an object
    (mdir / "broken.json").write_text("{not json")
    (mdir / "ok.json").write_text(json.dumps({"run_id": "abc12345", "backend": "x"}))
    with pytest.raises(ValueError, match="Invalid apply manifests"):
        rx.build_manifest_index([repo])


def test_build_manifest_index_rejects_workspace_and_duplicate_checkouts(tmp_path):
    repo = tmp_path / "rulespec-us"
    repo.mkdir()

    with pytest.raises(ValueError, match="exact canonical"):
        rx.build_manifest_index([tmp_path])
    with pytest.raises(ValueError, match="Duplicate RuleSpec checkout"):
        rx.build_manifest_index([repo, repo])


def _write_verified_run_log_manifest(tmp_path, monkeypatch):
    repo = tmp_path / "rulespec-us"
    waiver = repo / "known-validation-gaps.yaml"
    waiver.parent.mkdir(parents=True, exist_ok=True)
    waiver.write_text("validate_failures: {}\n")
    waiver_sha256 = hashlib.sha256(waiver.read_bytes()).hexdigest()
    toolchain = repo / ".axiom/toolchain.toml"
    toolchain.parent.mkdir(parents=True, exist_ok=True)
    toolchain.write_text(
        "[toolchain]\n"
        'axiom_corpus_release = "run-log-test"\n'
        f'axiom_corpus_release_content_sha256 = "{"e" * 64}"\n'
        f'validation_waiver_set_sha256 = "{waiver_sha256}"\n'
    )
    rule_rel = "us/statutes/26/1.yaml"
    corpus_path = "us/statute/26/1"
    source_sha256 = "a" * 64
    rule = repo / rule_rel
    rule.parent.mkdir(parents=True, exist_ok=True)
    rule.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        f"    corpus_citation_path: {corpus_path}\n"
        f"    source_sha256: {source_sha256}\n"
        "rules: []\n"
    )
    provision_file = "data/corpus/provisions/us/statute/test.jsonl"
    source_attestation = {
        "requested_corpus_citation_path": corpus_path,
        "resolved_corpus_citation_path": corpus_path,
        "corpus_source": "local",
        "corpus_release": "run-log-test",
        "corpus_release_content_sha256": "e" * 64,
        "corpus_release_selector_sha256": "b" * 64,
        "provision_file": provision_file,
        "provision_file_sha256": "c" * 64,
        "row": {
            "provision_file": provision_file,
            "provision_file_sha256": "c" * 64,
            "line_number": 1,
            "record_id": "test:1",
            "citation_path": corpus_path,
            "jurisdiction": "us",
            "document_class": "statute",
            "version": "test",
            "source_path": "sources/us/statute/1",
            "source_as_of": "2026-01-01",
            "expression_date": "2026-01-01",
            "body_sha256": source_sha256,
        },
        "component_rows": [],
        "source_sha256": source_sha256,
        "resolved_text_sha256": source_sha256,
        "generation_input_sha256": "d" * 64,
        "rulespec_root": "rulespec-us/us",
        "source_as_of": "2026-01-01",
        "expression_date": "2026-01-01",
    }
    payload = {
        "schema_version": "axiom-encode/applied-rulespec/v5",
        "generated_at": "2026-07-11T00:00:00+00:00",
        "run_id": "abc12345",
        "backend": "codex",
        "tool": "axiom-encode encode --apply",
        "axiom_encode_version": __version__,
        "axiom_encode_git": {
            "root": "/repo/axiom-encode",
            "commit": "a" * 40,
            "dirty_tracked": False,
            "version": __version__,
            "version_commit": "b" * 40,
            "identity_source": "git",
        },
        "generation_prompt_sha256": None,
        "citation": corpus_path,
        "runner": "codex",
        "model": "run-log-test-model",
        "validation_waiver_set_sha256": waiver_sha256,
        "generated_output_root": str(tmp_path / "generated"),
        "generated_output_file": None,
        "generated_output_sha256": None,
        "trace_file": None,
        "trace_sha256": None,
        "context_manifest_file": None,
        "context_manifest_sha256": None,
        "source_attestation": source_attestation,
        "applied_files": [
            {
                "path": rule_rel,
                "sha256": hashlib.sha256(rule.read_bytes()).hexdigest(),
            }
        ],
        "validation_execution": {
            "schema": "axiom-encode/apply-validation-execution/v1",
            "axiom_encode": {
                "repository": "github.com/TheAxiomFoundation/axiom-encode",
                "commit": "a" * 40,
                "version": __version__,
                "identity_source": "git",
            },
            "axiom_rules_engine": {
                "repository": "github.com/TheAxiomFoundation/axiom-rules-engine",
                "commit": "e" * 40,
            },
            "policy_pre_apply": {
                "rulespec_root": "rulespec-us/us",
                "pre_apply_content_sha256": "a" * 64,
                "pre_apply_file_count": 1,
                "toolchain_contract_sha256": "b" * 64,
                "validation_waiver_set_sha256": waiver_sha256,
            },
            "rulespec_dependencies": [],
        },
    }
    from axiom_encode.cli import _sign_applied_encoding_manifest

    signing_broker = SigningBrokerFixture(
        apply_private_key=TEST_APPLY_PRIVATE_KEY_B64,
        apply_public_key=TEST_APPLY_PUBLIC_KEY_B64,
    )
    _sign_applied_encoding_manifest(payload, signing_broker)
    manifest = repo / ".axiom/encoding-manifests/us/statutes/26/1.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload))
    monkeypatch.setenv(
        "AXIOM_ENCODE_APPLY_SIGNING_PUBLIC_KEY",
        TEST_APPLY_PUBLIC_KEY_B64,
    )
    monkeypatch.setattr(
        "axiom_encode.signing_broker._active_broker",
        signing_broker,
    )
    monkeypatch.setattr("axiom_encode.signing_broker._active_broker_pid", None)
    return repo, rule, manifest


def test_build_manifest_index_accepts_verified_v4(tmp_path, monkeypatch):
    repo, _rule, _manifest = _write_verified_run_log_manifest(tmp_path, monkeypatch)

    index = rx.build_manifest_index([repo])

    assert set(index) == {"abc12345"}
    assert len(index["abc12345"]["_manifest_sha256"]) == 64


def test_build_manifest_index_reuses_one_receipt_proof_cache(tmp_path, monkeypatch):
    import axiom_encode.cli as cli_module

    repo, _rule, manifest = _write_verified_run_log_manifest(tmp_path, monkeypatch)
    second = manifest.with_name("second.json")
    second.write_bytes(manifest.read_bytes())
    original = cli_module._load_verified_applied_encoding_manifest_payload
    proof_calls = 0

    def counted_loader(*args, **kwargs):
        nonlocal proof_calls
        cache = kwargs["path_migration_receipt_proof_cache"]
        proof_key = (str(repo), "shared-receipt")
        if proof_key not in cache:
            proof_calls += 1
            cache[proof_key] = ()
        payload, root_prefix, digest, issues = original(*args, **kwargs)
        if payload is not None and Path(args[1]).name == "second.json":
            payload = dict(payload)
            payload["run_id"] = "second-run"
        return payload, root_prefix, digest, issues

    monkeypatch.setattr(
        cli_module,
        "_load_verified_applied_encoding_manifest_payload",
        counted_loader,
    )

    index = rx.build_manifest_index([repo])

    assert set(index) == {"abc12345", "second-run"}
    assert proof_calls == 1


def test_build_manifest_index_rejects_stale_or_tampered_manifest(tmp_path, monkeypatch):
    repo, rule, manifest = _write_verified_run_log_manifest(tmp_path, monkeypatch)
    rule.write_text("format: rulespec/v1\nrules: []\n# changed\n")
    with pytest.raises(ValueError, match="stale sha256"):
        rx.build_manifest_index([repo])

    _repo, _rule, manifest = _write_verified_run_log_manifest(tmp_path, monkeypatch)
    payload = json.loads(manifest.read_text())
    payload["backend"] = "openai"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid encoder apply manifest signature"):
        rx.build_manifest_index([repo])


def test_publish_and_staleness_tolerate_corrupt_line(tmp_path):
    log_dir = tmp_path / "logs"
    writer = rl.RunLogWriter("r1", log_dir=log_dir)
    writer.emit("generate", "passed")
    # Append a truncated/garbage line, as a crash mid-append would.
    with (log_dir / "r1.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"schema": "axiom_encode.run_log.v1", "run_id"\n')
    out_dir = tmp_path / "data"
    meta = rx.publish(log_dir, out_dir)  # must not raise
    assert meta["run_count"] == 1
    assert meta["skipped_malformed_lines"] >= 1
    status = rx.check_staleness(out_dir, log_dir)  # guard must not crash
    assert status["ok"] is True
