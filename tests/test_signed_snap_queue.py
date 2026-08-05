from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.prepare_signed_queue import (
    ACTIVATION_CHANGED_FILES,
    EXPECTED_COUNTS,
    PRIORITY_CITATIONS,
    _json_sha256,
    _verify_corpus_provenance,
    _verify_signed_release_binding,
    dispatch_queue_sha256,
    finalization_target_plan,
    finalize_and_repin,
    item_generation_sha256,
    pause_queue,
    queue_file_sha256,
    queue_object_file_sha256,
    reconcile_candidates,
    record_disposition,
    select_items,
    selectable_items,
    validate_queue,
    validate_release_pin,
    validate_tracked_dispatch,
    verify_activation_commit,
    verify_activation_evidence,
    verify_merge_authorization,
    verify_paused_transition,
)
from tests.release_object_fixtures import (
    TEST_RELEASE_PUBLIC_KEY,
    write_test_release_object,
)

ROOT = Path(__file__).parents[1]
QUEUE_PATH = ROOT / "data/encoding-queues/us-snap-or-ut-2026-07.json"


def _queue(*, active: bool = True) -> dict:
    payload = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    if active:
        payload["state"] = "active"
        payload.pop("pause_reason", None)
        payload["activation"] = {
            "check_runs_sha256": "0" * 64,
            "finalizer_head_sha": "a" * 40,
            "finalizer_run_attempt": 1,
            "finalizer_run_url": (
                "https://github.com/TheAxiomFoundation/axiom-encode/actions/runs/1"
            ),
            "previous_queue_object_sha256": "1" * 64,
            "pull_requests_sha256": "2" * 64,
            "rulespec_ref": payload["dispatch"]["rulespec_ref"],
            "schema": "axiom-encode/signed-encoding-queue-activation/v1",
            "workflow_runs_sha256": "3" * 64,
        }
    return payload


def test_committed_snap_queue_is_complete_and_valid() -> None:
    payload = _queue(active=False)

    validate_queue(payload)

    assert payload["expected_counts"] == EXPECTED_COUNTS
    assert tuple(item["citation"] for item in payload["items"][:4]) == (
        PRIORITY_CITATIONS
    )
    assert len(queue_file_sha256(QUEUE_PATH)) == 64
    assert payload["state"] in {"active", "paused"}
    assert payload["release"]["manifest_sha256"] == (
        "be18ff3f1557f4e30a1520e519e4d14b31c01122534ca1690294729bb7029691"
    )


def test_select_snap_queue_defaults_to_bounded_priority_tranche() -> None:
    selection = select_items(_queue(), item_ids="", limit=4)

    assert [item["id"] for item in selection["items"]] == [
        "ut-0001",
        "or-0001",
        "or-0002",
        "or-0003",
    ]
    assert selection["dispatch"]["max_batch_size"] == 4
    assert selection["issue"] == 1257


def test_paused_snap_queue_is_not_dispatchable() -> None:
    payload = _queue(active=False)

    assert selectable_items(payload)["items"] == []
    assert select_items(payload, item_ids="", limit=4)["items"] == []
    with pytest.raises(ValueError, match="paused"):
        select_items(payload, item_ids="ut-0001", limit=1)

    bypass = copy.deepcopy(payload)
    bypass["state"] = "active"
    bypass.pop("pause_reason")
    with pytest.raises(ValueError, match="finalization evidence"):
        validate_queue(bypass)


def test_active_snap_queue_can_be_paused_fail_closed() -> None:
    active = _queue()
    active_digest = queue_object_file_sha256(active)
    paused = pause_queue(
        active,
        reason="Tranche dispatched; awaiting finalization.",
        active_queue_sha256=active_digest,
    )

    assert paused["state"] == "paused"
    assert paused["pause_reason"] == "Tranche dispatched; awaiting finalization."
    assert "activation" not in paused
    assert paused["suspension"] == {
        "active_queue_sha256": active_digest,
        "schema": "axiom-encode/signed-encoding-queue-suspension/v1",
    }
    assert dispatch_queue_sha256(paused) == active_digest
    assert selectable_items(paused)["items"] == []


def test_pause_transition_preserves_exact_noncanonical_active_file_digest(
    tmp_path: Path,
) -> None:
    active = _queue()
    previous = tmp_path / "active.json"
    previous.write_text(json.dumps(active, separators=(",", ":")), encoding="utf-8")
    active_digest = queue_file_sha256(previous)
    assert active_digest != queue_object_file_sha256(active)
    paused = pause_queue(
        active,
        reason="Tranche dispatched; awaiting finalization.",
        active_queue_sha256=active_digest,
    )
    current = tmp_path / "paused.json"
    current.write_text(
        json.dumps(paused, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    verify_paused_transition(current, previous_queue_path=previous)
    assert paused["suspension"]["active_queue_sha256"] == active_digest


def test_paused_transition_rejects_manual_completion(tmp_path: Path) -> None:
    previous_payload = _queue(active=False)
    current_payload = copy.deepcopy(previous_payload)
    current_payload["items"][0]["status"] = "completed"
    current_payload["items"][0]["evidence"] = {
        "applied_manifest_path": ".axiom/encoding-manifests/test.json",
        "applied_manifest_sha256": "c" * 64,
        "artifact_metadata_sha256": "d" * 64,
        "generation_sha256": "b" * 64,
        "merge_commit": "a" * 40,
        "rulespec_pr_head_sha": "e" * 40,
        "target_run_head_sha": "f" * 40,
        "target_run_attempt": 1,
        "target_run_url": (
            "https://github.com/TheAxiomFoundation/axiom-encode/actions/runs/1"
        ),
        "type": "merged-rulespec-pr",
        "url": "https://github.com/TheAxiomFoundation/rulespec-us/pull/1",
    }
    previous = tmp_path / "previous.json"
    current = tmp_path / "current.json"
    previous.write_text(json.dumps(previous_payload), encoding="utf-8")
    current.write_text(json.dumps(current_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot complete"):
        verify_paused_transition(current, previous_queue_path=previous)


def test_paused_transition_accepts_only_pristine_complete_toolchain_repin(
    tmp_path: Path,
) -> None:
    previous_payload = _queue(active=False)
    current_payload = copy.deepcopy(previous_payload)
    current_payload["dispatch"].update(
        {
            "corpus_ref": "1" * 40,
            "rules_engine_ref": "2" * 40,
            "rulespec_ref": "3" * 40,
        }
    )
    current_payload["release"] = {
        "content_sha256": "4" * 64,
        "manifest_sha256": "5" * 64,
        "name": "replacement-signed-release",
    }
    previous = tmp_path / "previous.json"
    current = tmp_path / "current.json"
    previous.write_text(json.dumps(previous_payload), encoding="utf-8")
    current.write_text(json.dumps(current_payload), encoding="utf-8")

    verify_paused_transition(current, previous_queue_path=previous)

    current_payload["items"][0]["attempt"] = 2
    current.write_text(json.dumps(current_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="requires pristine items"):
        verify_paused_transition(current, previous_queue_path=previous)


def test_paused_transition_rejects_partial_toolchain_repin(tmp_path: Path) -> None:
    previous_payload = _queue(active=False)
    current_payload = copy.deepcopy(previous_payload)
    current_payload["dispatch"]["rules_engine_ref"] = "2" * 40
    current_payload["release"] = {
        "content_sha256": "4" * 64,
        "manifest_sha256": "5" * 64,
        "name": "replacement-signed-release",
    }
    previous = tmp_path / "previous.json"
    current = tmp_path / "current.json"
    previous.write_text(json.dumps(previous_payload), encoding="utf-8")
    current.write_text(json.dumps(current_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must replace every source pin"):
        verify_paused_transition(current, previous_queue_path=previous)


def test_selectable_snap_queue_returns_history_scan_pool() -> None:
    candidates = selectable_items(_queue())

    assert len(candidates["items"]) == 831
    assert candidates["items"][0]["id"] == "ut-0001"
    assert candidates["items"][-1]["id"] == "ut-0301"


def test_completed_snap_queue_returns_empty_successful_selection() -> None:
    payload = _queue()
    for item in payload["items"]:
        item["status"] = "blocked"
        item["evidence"] = {
            "note": "Durable terminal test disposition.",
            "type": "issue-comment",
            "url": (
                "https://github.com/TheAxiomFoundation/axiom-encode/issues/"
                "1257#issuecomment-123"
            ),
        }

    assert selectable_items(payload)["items"] == []
    assert select_items(payload, item_ids="", limit=4)["items"] == []


def test_select_snap_queue_preserves_explicit_item_order() -> None:
    selection = select_items(
        _queue(),
        item_ids="or-0002,ut-0001",
        limit=2,
    )

    assert [item["id"] for item in selection["items"]] == [
        "or-0002",
        "ut-0001",
    ]


@pytest.mark.parametrize(
    ("item_ids", "limit", "message"),
    [
        ("", 5, "between 1 and 4"),
        ("missing-0001", 1, "unknown queue item"),
        ("ut-0001,ut-0001", 2, "duplicates"),
        ("ut-0001,or-0001", 1, "exceed"),
    ],
)
def test_select_snap_queue_rejects_invalid_tranches(
    item_ids: str,
    limit: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        select_items(_queue(), item_ids=item_ids, limit=limit)


def test_validate_snap_queue_rejects_count_and_priority_drift() -> None:
    count_drift = copy.deepcopy(_queue())
    count_drift["items"].pop()
    with pytest.raises(ValueError, match="count"):
        validate_queue(count_drift)

    priority_drift = copy.deepcopy(_queue())
    priority_drift["items"][0]["citation"] = "us-ut/manual/other"
    with pytest.raises(ValueError, match="priority tranche"):
        validate_queue(priority_drift)


def test_select_snap_queue_rejects_nonselectable_item() -> None:
    payload = record_disposition(
        _queue(),
        item_id="ut-0001",
        status="blocked",
        evidence_url=(
            "https://github.com/TheAxiomFoundation/axiom-encode/issues/"
            "1257#issuecomment-123"
        ),
        note="The source is blocked pending a readable capture.",
    )

    with pytest.raises(ValueError, match="not selectable"):
        select_items(payload, item_ids="ut-0001", limit=1)


def test_terminal_snap_queue_dispositions_require_durable_evidence() -> None:
    payload = _queue()
    payload["items"][0]["status"] = "completed"
    with pytest.raises(ValueError, match="durable evidence"):
        validate_queue(payload)

    with pytest.raises(ValueError, match="must be blocked"):
        record_disposition(
            _queue(),
            item_id="ut-0001",
            status="completed",
            evidence_url=("https://github.com/TheAxiomFoundation/rulespec-us/pull/123"),
            note=None,
        )

    no_rule = record_disposition(
        _queue(),
        item_id="ut-0001",
        status="no-executable-rule",
        evidence_url=(
            "https://github.com/TheAxiomFoundation/axiom-encode/issues/"
            "1257#issuecomment-123"
        ),
        note="The source unit contains only navigation text.",
    )
    assert no_rule["items"][0]["evidence"]["type"] == "issue-comment"


def _run_history(
    item_id: str,
    *,
    status: str,
    conclusion: str | None,
    payload: dict | None = None,
) -> list:
    selectable_payload = copy.deepcopy(payload or _queue())
    selectable_payload["state"] = "active"
    selectable_payload.pop("pause_reason", None)
    selectable_payload.pop("suspension", None)
    selectable_payload["activation"] = copy.deepcopy(_queue()["activation"])
    selectable_payload["activation"]["rulespec_ref"] = selectable_payload["dispatch"][
        "rulespec_ref"
    ]
    selection = select_items(selectable_payload, item_ids=item_id, limit=1)
    generation = selection["items"][0]["generation_sha256"]
    return [
        {
            "workflow_runs": [
                {
                    "actor": {"login": "github-actions[bot]"},
                    "conclusion": conclusion,
                    "created_at": "2026-07-24T12:00:00Z",
                    "display_title": (
                        "Targeted signed RuleSpec re-encode "
                        f"[us-snap-or-ut-2026-07:{item_id}:{generation}] citation"
                    ),
                    "event": "workflow_dispatch",
                    "head_branch": "main",
                    "head_sha": "c" * 40,
                    "html_url": (
                        "https://github.com/TheAxiomFoundation/"
                        "axiom-encode/actions/runs/1"
                    ),
                    "id": 1,
                    "path": ".github/workflows/targeted-signed-reencode.yml",
                    "run_attempt": 1,
                    "status": status,
                }
            ]
        }
    ]


def _pr_history(
    item_id: str,
    *,
    state: str,
    merged: bool,
    generation_sha256: str | None = None,
    citation: str | None = None,
    base_commit: str | None = None,
    merge_commit_sha: str | None = None,
    payload: dict | None = None,
) -> list:
    selectable_payload = copy.deepcopy(payload or _queue())
    selectable_payload["state"] = "active"
    selectable_payload.pop("pause_reason", None)
    selectable_payload.pop("suspension", None)
    selectable_payload["activation"] = copy.deepcopy(_queue()["activation"])
    selectable_payload["activation"]["rulespec_ref"] = selectable_payload["dispatch"][
        "rulespec_ref"
    ]
    selection = select_items(selectable_payload, item_ids=item_id, limit=1)
    item = selection["items"][0]
    dispatch = selection["dispatch"]
    generation_sha256 = generation_sha256 or item["generation_sha256"]
    citation = citation or item["citation"]
    base_commit = base_commit or dispatch["rulespec_ref"]
    return [
        [
            {
                "base": {"ref": "hard-cut/canonical-layout-us"},
                "body": (
                    "Generated PR\n\n"
                    f"Citation: `{citation}`\n"
                    f"Base commit: `{base_commit}`\n"
                    "Base branch: `hard-cut/canonical-layout-us`\n"
                    f"Queue item: `us-snap-or-ut-2026-07/{item_id}`\n"
                    f"Queue generation SHA-256: `{generation_sha256}`\n"
                    "Axiom Encode run: https://github.com/"
                    "TheAxiomFoundation/axiom-encode/actions/runs/1"
                ),
                "head": {
                    "repo": {"full_name": "TheAxiomFoundation/rulespec-us"},
                    "sha": "d" * 40,
                },
                "html_url": "https://github.com/TheAxiomFoundation/rulespec-us/pull/1",
                "merged_at": "2026-07-24T13:00:00Z" if merged else None,
                "merge_commit_sha": (merge_commit_sha or "b" * 40 if merged else None),
                "number": 1,
                "state": state,
                "updated_at": "2026-07-24T13:00:00Z",
            }
        ]
    ]


def test_reconcile_queue_treats_merged_or_open_pr_as_durable_state() -> None:
    selection = select_items(_queue(), item_ids="", limit=2)
    prs = _pr_history("ut-0001", state="closed", merged=True)
    prs[0].extend(_pr_history("or-0001", state="open", merged=False)[0])

    result = reconcile_candidates(
        selection,
        pull_requests=prs,
        workflow_runs=_run_history(
            "ut-0001",
            status="completed",
            conclusion="success",
        ),
    )

    assert [item["reason"] for item in result["items"]] == [
        "merged-rulespec-pr",
        "open-rulespec-pr",
    ]
    assert not any(item["dispatchable"] for item in result["items"])


def test_reconcile_queue_does_not_treat_successful_run_as_completion() -> None:
    selection = select_items(_queue(), item_ids="ut-0001", limit=1)

    result = reconcile_candidates(
        selection,
        pull_requests=[],
        workflow_runs=_run_history(
            "ut-0001",
            status="completed",
            conclusion="success",
        ),
    )

    assert result["items"][0]["reason"] == "successful-run-without-durable-pr"
    assert result["items"][0]["dispatchable"] is False


def test_reconcile_queue_does_not_trust_merged_pr_body_without_target_run() -> None:
    selection = select_items(_queue(), item_ids="ut-0001", limit=1)

    result = reconcile_candidates(
        selection,
        pull_requests=_pr_history("ut-0001", state="closed", merged=True),
        workflow_runs=[],
    )

    assert result["items"][0]["reason"] == "untrusted-merged-rulespec-pr"
    assert result["items"][0]["dispatchable"] is False


def test_reconcile_queue_preserves_attempt_one_target_after_skipped_rerun() -> None:
    selection = select_items(_queue(), item_ids="ut-0001", limit=1)
    runs = _run_history(
        "ut-0001",
        status="completed",
        conclusion="success",
    )
    runs[0]["workflow_runs"][0]["run_attempt"] = 2

    result = reconcile_candidates(
        selection,
        pull_requests=_pr_history("ut-0001", state="closed", merged=True),
        workflow_runs=runs,
    )

    assert result["items"][0]["reason"] == "merged-rulespec-pr"
    assert result["items"][0]["dispatchable"] is False


def test_reconcile_queue_ignores_matching_marker_from_untrusted_fork() -> None:
    selection = select_items(_queue(), item_ids="ut-0001", limit=1)
    prs = _pr_history("ut-0001", state="open", merged=False)
    prs[0][0]["head"]["repo"]["full_name"] = "external/fork"

    result = reconcile_candidates(
        selection,
        pull_requests=prs,
        workflow_runs=[],
    )

    assert result["items"][0]["dispatchable"] is True
    assert result["items"][0]["reason"] == "new"


@pytest.mark.parametrize(
    "pr_kwargs",
    [
        {"generation_sha256": "0" * 64},
        {"citation": "us-ut/manual/stale-citation"},
        {"base_commit": "0" * 40},
    ],
)
def test_reconcile_queue_ignores_stale_pr_identity(pr_kwargs: dict) -> None:
    selection = select_items(_queue(), item_ids="ut-0001", limit=1)

    result = reconcile_candidates(
        selection,
        pull_requests=_pr_history(
            "ut-0001",
            state="closed",
            merged=True,
            **pr_kwargs,
        ),
        workflow_runs=[],
    )

    assert result["items"][0]["dispatchable"] is True
    assert result["items"][0]["reason"] == "new"


def test_retryable_disposition_creates_new_item_generation() -> None:
    original = select_items(_queue(), item_ids="ut-0001", limit=1)
    retried_queue = record_disposition(
        _queue(),
        item_id="ut-0001",
        status="retryable",
        evidence_url=None,
        note=None,
    )
    retried = select_items(retried_queue, item_ids="ut-0001", limit=1)

    assert retried["items"][0]["attempt"] == 2
    assert (
        retried["items"][0]["generation_sha256"]
        != original["items"][0]["generation_sha256"]
    )
    result = reconcile_candidates(
        retried,
        pull_requests=_pr_history("ut-0001", state="closed", merged=True),
        workflow_runs=[],
    )
    assert result["items"][0]["dispatchable"] is True


def _repin_fixture(tmp_path: Path) -> tuple[dict, Path, str, str]:
    payload = _queue()
    rulespec = tmp_path / "rulespec"
    toolchain = rulespec / ".axiom/toolchain.toml"
    toolchain.parent.mkdir(parents=True)
    release = payload["release"]
    toolchain.write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{release["name"]}"\n'
        "axiom_corpus_release_content_sha256 = "
        f'"{release["content_sha256"]}"\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", rulespec], check=True)
    subprocess.run(["git", "-C", rulespec, "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            rulespec,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "old base",
        ],
        check=True,
    )
    old_ref = subprocess.check_output(
        ["git", "-C", rulespec, "rev-parse", "HEAD"],
        text=True,
    ).strip()
    (rulespec / "next.txt").write_text("next base\n", encoding="utf-8")
    manifest = rulespec / ".axiom/encoding-manifests/test.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"signed": true}\n', encoding="utf-8")
    subprocess.run(["git", "-C", rulespec, "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            rulespec,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "new base",
        ],
        check=True,
    )
    new_ref = subprocess.check_output(
        ["git", "-C", rulespec, "rev-parse", "HEAD"],
        text=True,
    ).strip()
    subprocess.run(
        [
            "git",
            "-C",
            rulespec,
            "update-ref",
            "refs/remotes/origin/hard-cut/canonical-layout-us",
            new_ref,
        ],
        check=True,
    )
    payload["dispatch"]["rulespec_ref"] = old_ref
    payload["activation"]["rulespec_ref"] = old_ref
    validate_queue(payload)
    payload = pause_queue(
        payload,
        reason="Tranche dispatched; awaiting finalization.",
        active_queue_sha256=queue_object_file_sha256(payload),
    )
    return payload, rulespec, old_ref, new_ref


def _finalizer_evidence() -> dict:
    return {
        "check_runs": [
            {
                "check_runs": [
                    {
                        "conclusion": "success",
                        "name": "validate",
                        "status": "completed",
                    }
                ]
            }
        ],
        "finalizer_head_sha": "a" * 40,
        "finalizer_run_url": (
            "https://github.com/TheAxiomFoundation/axiom-encode/actions/runs/1"
        ),
        "target_evidence": {},
    }


def _target_evidence(payload: dict, rulespec: Path, item_id: str) -> dict:
    item = next(value for value in payload["items"] if value["id"] == item_id)
    manifest_path = ".axiom/encoding-manifests/test.json"
    manifest_digest = hashlib.sha256(
        (rulespec / manifest_path).read_bytes()
    ).hexdigest()
    run_url = "https://github.com/TheAxiomFoundation/axiom-encode/actions/runs/1"
    return {
        item_id: {
            "apply_manifests": {
                "items": [
                    {
                        "citation": item["citation"],
                        "path": manifest_path,
                        "sha256": manifest_digest,
                    }
                ],
                "schema": "axiom-encode/applied-manifest-inventory/v1",
            },
            "jobs": {
                "jobs": [
                    {
                        "conclusion": "success",
                        "name": "Queue protected signed RuleSpec re-encode",
                        "run_attempt": 1,
                        "status": "completed",
                    }
                ]
            },
            "metadata": {
                "citation": item["citation"],
                "encoder_commit": "c" * 40,
                "pr_base_branch": payload["dispatch"]["pr_base_branch"],
                "queue_id": payload["queue_id"],
                "queue_item_generation_sha256": item_generation_sha256(payload, item),
                "queue_item_id": item_id,
                "queue_manifest_sha256": dispatch_queue_sha256(payload),
                "rulespec_base": payload["dispatch"]["rulespec_ref"],
                "schema": "axiom-encode/targeted-reencode-artifact/v1",
                "workflow_run_attempt": 1,
                "workflow_run_id": "1",
            },
            "pull_request": {
                "head": {"sha": "d" * 40},
                "html_url": (
                    "https://github.com/TheAxiomFoundation/rulespec-us/pull/1"
                ),
                "number": 1,
            },
            "run": {
                "actor": {"login": "github-actions[bot]"},
                "conclusion": "success",
                "event": "workflow_dispatch",
                "head_branch": "main",
                "head_sha": "c" * 40,
                "html_url": run_url,
                "id": 1,
                "path": ".github/workflows/targeted-signed-reencode.yml",
                "run_attempt": 1,
                "status": "completed",
            },
        }
    }


def test_finalize_repin_activates_quiet_queue_and_records_merged_pr(
    tmp_path: Path,
) -> None:
    payload, rulespec, _, new_ref = _repin_fixture(tmp_path)
    evidence = _finalizer_evidence()
    evidence["target_evidence"] = _target_evidence(payload, rulespec, "ut-0001")
    assert (
        evidence["target_evidence"]["ut-0001"]["metadata"]["queue_manifest_sha256"]
        == payload["suspension"]["active_queue_sha256"]
    )
    updated = finalize_and_repin(
        payload,
        rulespec_root=rulespec,
        pull_requests=_pr_history(
            "ut-0001",
            state="closed",
            merged=True,
            merge_commit_sha=new_ref,
            payload=payload,
        ),
        workflow_runs=_run_history(
            "ut-0001",
            status="completed",
            conclusion="success",
            payload=payload,
        ),
        new_rulespec_ref=new_ref,
        reviewed_rulespec_refs=frozenset({("us", new_ref)}),
        **evidence,
    )

    assert updated["state"] == "active"
    assert "pause_reason" not in updated
    assert "suspension" not in updated
    assert updated["dispatch"]["rulespec_ref"] == new_ref
    assert updated["items"][0]["status"] == "completed"
    assert updated["items"][0]["evidence"]["merge_commit"] == new_ref


def test_finalize_repin_accepts_attempt_one_target_after_skipped_rerun(
    tmp_path: Path,
) -> None:
    payload, rulespec, _, new_ref = _repin_fixture(tmp_path)
    evidence = _finalizer_evidence()
    target_evidence = _target_evidence(payload, rulespec, "ut-0001")
    target = target_evidence["ut-0001"]
    target["run"]["run_attempt"] = 2
    target["jobs"]["jobs"].append(
        {
            "conclusion": "skipped",
            "name": "Queue protected signed RuleSpec re-encode",
            "run_attempt": 2,
            "status": "completed",
        }
    )
    evidence["target_evidence"] = target_evidence

    updated = finalize_and_repin(
        payload,
        rulespec_root=rulespec,
        pull_requests=_pr_history(
            "ut-0001",
            state="closed",
            merged=True,
            merge_commit_sha=new_ref,
            payload=payload,
        ),
        workflow_runs=_run_history(
            "ut-0001",
            status="completed",
            conclusion="success",
            payload=payload,
        ),
        new_rulespec_ref=new_ref,
        reviewed_rulespec_refs=frozenset({("us", new_ref)}),
        **evidence,
    )

    assert updated["items"][0]["evidence"]["target_run_attempt"] == 1
    target["jobs"]["jobs"][1]["conclusion"] = "success"
    with pytest.raises(ValueError, match="later attempt executed"):
        finalize_and_repin(
            payload,
            rulespec_root=rulespec,
            pull_requests=_pr_history(
                "ut-0001",
                state="closed",
                merged=True,
                merge_commit_sha=new_ref,
                payload=payload,
            ),
            workflow_runs=_run_history(
                "ut-0001",
                status="completed",
                conclusion="success",
                payload=payload,
            ),
            new_rulespec_ref=new_ref,
            reviewed_rulespec_refs=frozenset({("us", new_ref)}),
            **evidence,
        )


def test_finalization_plan_and_repin_require_exact_target_artifact(
    tmp_path: Path,
) -> None:
    payload, rulespec, _, new_ref = _repin_fixture(tmp_path)
    pull_requests = _pr_history(
        "ut-0001",
        state="closed",
        merged=True,
        merge_commit_sha=new_ref,
        payload=payload,
    )
    workflow_runs = _run_history(
        "ut-0001",
        status="completed",
        conclusion="success",
        payload=payload,
    )
    plan = finalization_target_plan(
        payload,
        pull_requests=pull_requests,
        workflow_runs=workflow_runs,
    )
    assert plan["items"] == [
        {
            "id": "ut-0001",
            "pull_request_url": (
                "https://github.com/TheAxiomFoundation/rulespec-us/pull/1"
            ),
            "target_run_url": (
                "https://github.com/TheAxiomFoundation/axiom-encode/actions/runs/1"
            ),
        }
    ]

    evidence = _finalizer_evidence()
    evidence["target_evidence"] = _target_evidence(payload, rulespec, "ut-0001")
    evidence["target_evidence"]["ut-0001"]["apply_manifests"]["items"][0]["sha256"] = (
        "0" * 64
    )
    with pytest.raises(
        ValueError,
        match="merged RuleSpec tree lacks the signed applied manifest",
    ):
        finalize_and_repin(
            payload,
            rulespec_root=rulespec,
            pull_requests=pull_requests,
            workflow_runs=workflow_runs,
            new_rulespec_ref=new_ref,
            reviewed_rulespec_refs=frozenset({("us", new_ref)}),
            **evidence,
        )


def test_finalize_repin_revalidates_existing_completed_items(tmp_path: Path) -> None:
    payload, rulespec, _, new_ref = _repin_fixture(tmp_path)
    pull_requests = _pr_history(
        "ut-0001",
        state="closed",
        merged=True,
        merge_commit_sha=new_ref,
        payload=payload,
    )
    target = _target_evidence(payload, rulespec, "ut-0001")["ut-0001"]
    applied = target["apply_manifests"]["items"][0]
    payload["items"][0]["status"] = "completed"
    payload["items"][0]["evidence"] = {
        "applied_manifest_path": applied["path"],
        "applied_manifest_sha256": applied["sha256"],
        "artifact_metadata_sha256": _json_sha256(target["metadata"]),
        "generation_sha256": item_generation_sha256(payload, payload["items"][0]),
        "merge_commit": new_ref,
        "rulespec_pr_head_sha": "d" * 40,
        "target_run_head_sha": "c" * 40,
        "target_run_attempt": 1,
        "target_run_url": (
            "https://github.com/TheAxiomFoundation/axiom-encode/actions/runs/1"
        ),
        "type": "merged-rulespec-pr",
        "url": "https://github.com/TheAxiomFoundation/rulespec-us/pull/1",
    }

    updated = finalize_and_repin(
        payload,
        rulespec_root=rulespec,
        pull_requests=pull_requests,
        workflow_runs=[],
        new_rulespec_ref=new_ref,
        reviewed_rulespec_refs=frozenset({("us", new_ref)}),
        **_finalizer_evidence(),
    )
    assert updated["items"][0]["status"] == "completed"

    pull_requests[0][0]["merge_commit_sha"] = "f" * 40
    with pytest.raises(ValueError, match="verified merged RuleSpec PR"):
        finalize_and_repin(
            payload,
            rulespec_root=rulespec,
            pull_requests=pull_requests,
            workflow_runs=[],
            new_rulespec_ref=new_ref,
            reviewed_rulespec_refs=frozenset({("us", new_ref)}),
            **_finalizer_evidence(),
        )


def test_verify_activation_commit_rejects_amendment_or_extra_file(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(_queue(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    base_sha = "a" * 40
    head_sha = "b" * 40
    tree_sha = "c" * 40
    changed_files = list(ACTIVATION_CHANGED_FILES)
    provenance = {
        "base_sha": base_sha,
        "changed_files": changed_files,
        "head_sha": head_sha,
        "queue_path": ACTIVATION_CHANGED_FILES[0],
        "queue_sha256": queue_file_sha256(queue),
        "repository": "TheAxiomFoundation/axiom-encode",
        "schema": "axiom-encode/snap-queue-activation-commit/v1",
        "tree_sha": tree_sha,
    }
    pull_request = {
        "base": {"ref": "main", "sha": base_sha},
        "head": {
            "repo": {"full_name": "TheAxiomFoundation/axiom-encode"},
            "sha": head_sha,
        },
        "state": "open",
    }

    verify_activation_commit(
        queue,
        provenance=provenance,
        pull_request=pull_request,
        current_base_sha=base_sha,
        current_head_sha=head_sha,
        current_tree_sha=tree_sha,
        current_changed_files=changed_files,
    )

    with pytest.raises(ValueError, match="commit or changed-file inventory"):
        verify_activation_commit(
            queue,
            provenance=provenance,
            pull_request=pull_request,
            current_base_sha=base_sha,
            current_head_sha="d" * 40,
            current_tree_sha=tree_sha,
            current_changed_files=changed_files,
        )
    with pytest.raises(ValueError, match="commit or changed-file inventory"):
        verify_activation_commit(
            queue,
            provenance=provenance,
            pull_request=pull_request,
            current_base_sha=base_sha,
            current_head_sha=head_sha,
            current_tree_sha=tree_sha,
            current_changed_files=[*changed_files, ".github/workflows/pwn.yml"],
        )


def test_verify_activation_evidence_authenticates_run_and_artifact(
    tmp_path: Path,
) -> None:
    payload, rulespec, _, new_ref = _repin_fixture(tmp_path)
    evidence = _finalizer_evidence()
    pull_requests = []
    workflow_runs = []
    updated = finalize_and_repin(
        payload,
        rulespec_root=rulespec,
        pull_requests=pull_requests,
        workflow_runs=workflow_runs,
        new_rulespec_ref=new_ref,
        reviewed_rulespec_refs=frozenset({("us", new_ref)}),
        **evidence,
    )
    queue = tmp_path / "queue.json"
    previous = tmp_path / "previous.json"
    finalized = tmp_path / "finalized.json"
    previous.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    serialized = json.dumps(updated, indent=2, sort_keys=True) + "\n"
    queue.write_text(serialized, encoding="utf-8")
    finalized.write_text(serialized, encoding="utf-8")
    finalizer_run = {
        "conclusion": "success",
        "event": "workflow_dispatch",
        "head_branch": "main",
        "head_sha": "a" * 40,
        "html_url": (
            "https://github.com/TheAxiomFoundation/axiom-encode/actions/runs/1"
        ),
        "path": ".github/workflows/finalize-signed-snap-queue.yml",
        "run_attempt": 1,
        "status": "completed",
    }
    finalizer_jobs = {
        "jobs": [
            {
                "conclusion": "success",
                "name": "Finalize protected SNAP queue tranche",
                "run_attempt": 1,
                "status": "completed",
            }
        ]
    }

    verify_activation_evidence(
        queue,
        previous_queue_path=previous,
        expected_base_sha="a" * 40,
        finalized_queue_path=finalized,
        check_runs=evidence["check_runs"],
        pull_requests=pull_requests,
        workflow_runs=workflow_runs,
        finalizer_run=finalizer_run,
        finalizer_jobs=finalizer_jobs,
        require_success=True,
    )

    finalizer_run["run_attempt"] = 2
    finalizer_jobs["jobs"].append(
        {
            "conclusion": "skipped",
            "name": "Finalize protected SNAP queue tranche",
            "run_attempt": 2,
            "status": "completed",
        }
    )
    verify_activation_evidence(
        queue,
        previous_queue_path=previous,
        expected_base_sha="a" * 40,
        finalized_queue_path=finalized,
        check_runs=evidence["check_runs"],
        pull_requests=pull_requests,
        workflow_runs=workflow_runs,
        finalizer_run=finalizer_run,
        finalizer_jobs=finalizer_jobs,
        require_success=True,
    )
    finalizer_jobs["jobs"][1]["conclusion"] = "success"
    with pytest.raises(ValueError, match="later attempt executed"):
        verify_activation_evidence(
            queue,
            previous_queue_path=previous,
            expected_base_sha="a" * 40,
            finalized_queue_path=finalized,
            check_runs=evidence["check_runs"],
            pull_requests=pull_requests,
            workflow_runs=workflow_runs,
            finalizer_run=finalizer_run,
            finalizer_jobs=finalizer_jobs,
            require_success=True,
        )
    finalizer_run["run_attempt"] = 1
    finalizer_jobs["jobs"].pop()

    finalized.write_text(json.dumps(updated), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the finalizer artifact"):
        verify_activation_evidence(
            queue,
            previous_queue_path=previous,
            expected_base_sha="a" * 40,
            finalized_queue_path=finalized,
            check_runs=evidence["check_runs"],
            pull_requests=pull_requests,
            workflow_runs=workflow_runs,
            finalizer_run=finalizer_run,
            finalizer_jobs=finalizer_jobs,
            require_success=True,
        )
    finalized.write_text(serialized, encoding="utf-8")

    finalizer_run["head_sha"] = "b" * 40
    with pytest.raises(ValueError, match="identity does not match"):
        verify_activation_evidence(
            queue,
            previous_queue_path=previous,
            expected_base_sha="a" * 40,
            finalized_queue_path=finalized,
            check_runs=evidence["check_runs"],
            pull_requests=pull_requests,
            workflow_runs=workflow_runs,
            finalizer_run=finalizer_run,
            finalizer_jobs=finalizer_jobs,
            require_success=True,
        )

    finalizer_run["head_sha"] = "a" * 40
    stale_previous = copy.deepcopy(payload)
    stale_previous["items"][0]["attempt"] = 2
    previous.write_text(
        json.dumps(stale_previous, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="base queue digest"):
        verify_activation_evidence(
            queue,
            previous_queue_path=previous,
            expected_base_sha="a" * 40,
            finalized_queue_path=finalized,
            check_runs=evidence["check_runs"],
            pull_requests=pull_requests,
            workflow_runs=workflow_runs,
            finalizer_run=finalizer_run,
            finalizer_jobs=finalizer_jobs,
            require_success=True,
        )


def test_verify_merge_authorization_rejects_direct_activation_merge(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(_queue(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_id = 9
    merge_commit = "b" * 40
    pr_head = "a" * 40
    run_url = (
        f"https://github.com/TheAxiomFoundation/axiom-encode/actions/runs/{run_id}"
    )
    authorization = {
        "activation_pr_head_sha": pr_head,
        "activation_pr_number": 123,
        "merge_commit": merge_commit,
        "merge_workflow_run_id": run_id,
        "merge_workflow_run_attempt": 1,
        "merge_workflow_run_url": run_url,
        "queue_path": "data/encoding-queues/us-snap-or-ut-2026-07.json",
        "queue_sha256": queue_file_sha256(queue),
        "repository": "TheAxiomFoundation/axiom-encode",
        "schema": "axiom-encode/snap-queue-merge-authorization/v1",
    }
    merge_run = {
        "conclusion": "success",
        "event": "workflow_dispatch",
        "head_branch": "main",
        "html_url": run_url,
        "path": ".github/workflows/merge-snap-queue-activation.yml",
        "run_attempt": 1,
        "status": "completed",
    }
    merge_jobs = {
        "jobs": [
            {
                "conclusion": "success",
                "name": "Merge reviewed SNAP queue activation",
                "run_attempt": 1,
                "status": "completed",
            }
        ]
    }
    pull_request = {
        "base": {"ref": "main"},
        "head": {
            "repo": {"full_name": "TheAxiomFoundation/axiom-encode"},
            "sha": pr_head,
        },
        "merge_commit_sha": merge_commit,
        "merged_at": "2026-07-24T13:00:00Z",
        "merged_by": {"login": "github-actions[bot]"},
        "number": 123,
        "state": "closed",
    }

    verify_merge_authorization(
        queue,
        authorization=authorization,
        merge_run=merge_run,
        merge_jobs=merge_jobs,
        pull_request=pull_request,
        current_head_sha="c" * 40,
        queue_change_sha=merge_commit,
    )

    merge_run["run_attempt"] = 2
    merge_jobs["jobs"].append(
        {
            "conclusion": "skipped",
            "name": "Merge reviewed SNAP queue activation",
            "run_attempt": 2,
            "status": "completed",
        }
    )
    verify_merge_authorization(
        queue,
        authorization=authorization,
        merge_run=merge_run,
        merge_jobs=merge_jobs,
        pull_request=pull_request,
        current_head_sha="c" * 40,
        queue_change_sha=merge_commit,
    )
    merge_jobs["jobs"][1]["conclusion"] = "success"
    with pytest.raises(ValueError, match="later attempt executed"):
        verify_merge_authorization(
            queue,
            authorization=authorization,
            merge_run=merge_run,
            merge_jobs=merge_jobs,
            pull_request=pull_request,
            current_head_sha="c" * 40,
            queue_change_sha=merge_commit,
        )
    merge_run["run_attempt"] = 1
    merge_jobs["jobs"].pop()

    pull_request["merged_by"]["login"] = "maintainer"
    with pytest.raises(ValueError, match="trusted workflow"):
        verify_merge_authorization(
            queue,
            authorization=authorization,
            merge_run=merge_run,
            merge_jobs=merge_jobs,
            pull_request=pull_request,
            current_head_sha="c" * 40,
            queue_change_sha=merge_commit,
        )


def test_finalize_repin_refuses_open_pr_or_active_run(tmp_path: Path) -> None:
    payload, rulespec, _, new_ref = _repin_fixture(tmp_path)
    with pytest.raises(ValueError, match="pull requests are open"):
        finalize_and_repin(
            payload,
            rulespec_root=rulespec,
            pull_requests=_pr_history(
                "ut-0001",
                state="open",
                merged=False,
                payload=payload,
            ),
            workflow_runs=[],
            new_rulespec_ref=new_ref,
            reviewed_rulespec_refs=frozenset({("us", new_ref)}),
            **_finalizer_evidence(),
        )

    with pytest.raises(ValueError, match="workflow runs are active"):
        finalize_and_repin(
            payload,
            rulespec_root=rulespec,
            pull_requests=[],
            workflow_runs=_run_history(
                "ut-0001",
                status="in_progress",
                conclusion=None,
                payload=payload,
            ),
            new_rulespec_ref=new_ref,
            reviewed_rulespec_refs=frozenset({("us", new_ref)}),
            **_finalizer_evidence(),
        )


def test_finalize_repin_requires_reviewed_allowlisted_base(tmp_path: Path) -> None:
    payload, rulespec, _, new_ref = _repin_fixture(tmp_path)
    with pytest.raises(ValueError, match="reviewed and allowlisted"):
        finalize_and_repin(
            payload,
            rulespec_root=rulespec,
            pull_requests=[],
            workflow_runs=[],
            new_rulespec_ref=new_ref,
            reviewed_rulespec_refs=frozenset(),
            **_finalizer_evidence(),
        )


def test_finalize_repin_requires_green_rulespec_checks(tmp_path: Path) -> None:
    payload, rulespec, _, new_ref = _repin_fixture(tmp_path)
    evidence = _finalizer_evidence()
    evidence["check_runs"][0]["check_runs"][0]["conclusion"] = "failure"
    with pytest.raises(ValueError, match="green check runs"):
        finalize_and_repin(
            payload,
            rulespec_root=rulespec,
            pull_requests=[],
            workflow_runs=[],
            new_rulespec_ref=new_ref,
            reviewed_rulespec_refs=frozenset({("us", new_ref)}),
            **evidence,
        )


def test_finalize_repin_requires_paused_exact_remote_tip(tmp_path: Path) -> None:
    payload, rulespec, old_ref, new_ref = _repin_fixture(tmp_path)
    active = copy.deepcopy(payload)
    active["state"] = "active"
    active.pop("pause_reason", None)
    active.pop("suspension", None)
    active["activation"] = copy.deepcopy(_queue()["activation"])
    active["activation"]["rulespec_ref"] = old_ref
    with pytest.raises(ValueError, match="requires a paused queue"):
        finalize_and_repin(
            active,
            rulespec_root=rulespec,
            pull_requests=[],
            workflow_runs=[],
            new_rulespec_ref=new_ref,
            reviewed_rulespec_refs=frozenset({("us", new_ref)}),
            **_finalizer_evidence(),
        )

    subprocess.run(
        [
            "git",
            "-C",
            rulespec,
            "update-ref",
            "refs/remotes/origin/hard-cut/canonical-layout-us",
            old_ref,
        ],
        check=True,
    )
    with pytest.raises(ValueError, match="exact checked-out remote branch tip"):
        finalize_and_repin(
            payload,
            rulespec_root=rulespec,
            pull_requests=[],
            workflow_runs=[],
            new_rulespec_ref=new_ref,
            reviewed_rulespec_refs=frozenset({("us", new_ref)}),
            **_finalizer_evidence(),
        )


def test_finalize_repin_requires_failed_run_disposition(tmp_path: Path) -> None:
    payload, rulespec, _, new_ref = _repin_fixture(tmp_path)
    with pytest.raises(ValueError, match="dispositions are required"):
        finalize_and_repin(
            payload,
            rulespec_root=rulespec,
            pull_requests=[],
            workflow_runs=_run_history(
                "ut-0001",
                status="completed",
                conclusion="failure",
                payload=payload,
            ),
            new_rulespec_ref=new_ref,
            reviewed_rulespec_refs=frozenset({("us", new_ref)}),
            **_finalizer_evidence(),
        )


def test_verify_corpus_provenance_rejects_wrong_head_and_checkout_dirt(
    tmp_path: Path,
) -> None:
    release_name = "test-release"
    release_dir = tmp_path / "manifests/releases"
    release_dir.mkdir(parents=True)
    provisions = tmp_path / "data/corpus/provisions"
    source_paths = [
        provisions / "us-or/manual/2026-07-16-or-programs-eligibility-notebook.jsonl",
        provisions
        / "us-ut/manual/2026-05-27-ut-manuals-r2026-07-15-self-contained.jsonl",
        provisions / "us-ut/manual/2026-07-13-recovery.jsonl",
    ]
    for source_path in source_paths:
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text("{}\n", encoding="utf-8")
    release = {
        "name": release_name,
        "scopes": [
            {
                "document_class": "manual",
                "jurisdiction": "us-or",
                "version": "2026-07-16-or-programs-eligibility-notebook",
            },
            {
                "document_class": "manual",
                "jurisdiction": "us-ut",
                "version": "2026-05-27-ut-manuals-r2026-07-15-self-contained",
            },
            {
                "document_class": "manual",
                "jurisdiction": "us-ut",
                "version": "2026-07-13-recovery",
            },
        ],
    }
    manifest = release_dir / f"{release_name}.json"
    manifest.write_text(json.dumps(release), encoding="utf-8")
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    subprocess.run(["git", "-C", tmp_path, "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            tmp_path,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    head = subprocess.check_output(
        ["git", "-C", tmp_path, "rev-parse", "HEAD"],
        text=True,
    ).strip()

    assert (
        len(
            _verify_corpus_provenance(
                tmp_path,
                corpus_ref=head,
                release_name=release_name,
                source_paths=source_paths,
            )
        )
        == 64
    )
    with pytest.raises(ValueError, match="does not match"):
        _verify_corpus_provenance(
            tmp_path,
            corpus_ref="a" * 40,
            release_name=release_name,
            source_paths=source_paths,
        )

    manifest.write_text(
        json.dumps({**release, "description": "dirty"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not clean"):
        _verify_corpus_provenance(
            tmp_path,
            corpus_ref=head,
            release_name=release_name,
            source_paths=source_paths,
        )

    manifest.write_text(
        json.dumps({**release, "name": "wrong-release"}), encoding="utf-8"
    )
    subprocess.run(
        [
            "git",
            "-C",
            tmp_path,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qam",
            "wrong release",
        ],
        check=True,
    )
    wrong_release_head = subprocess.check_output(
        ["git", "-C", tmp_path, "rev-parse", "HEAD"],
        text=True,
    ).strip()
    with pytest.raises(ValueError, match="name does not match"):
        _verify_corpus_provenance(
            tmp_path,
            corpus_ref=wrong_release_head,
            release_name=release_name,
            source_paths=source_paths,
        )

    missing_recovery = {
        **release,
        "scopes": [
            scope
            for scope in release["scopes"]
            if scope["version"] != "2026-07-13-recovery"
        ],
    }
    manifest.write_text(json.dumps(missing_recovery), encoding="utf-8")
    subprocess.run(
        [
            "git",
            "-C",
            tmp_path,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qam",
            "missing recovery scope",
        ],
        check=True,
    )
    missing_scope_head = subprocess.check_output(
        ["git", "-C", tmp_path, "rev-parse", "HEAD"],
        text=True,
    ).strip()
    with pytest.raises(ValueError, match="2026-07-13-recovery"):
        _verify_corpus_provenance(
            tmp_path,
            corpus_ref=missing_scope_head,
            release_name=release_name,
            source_paths=source_paths,
        )


def test_verify_signed_release_binding_authenticates_sources_and_toolchain(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    scopes = [
        ("us-or", "manual", "or-manual"),
        ("us-ut", "manual", "ut-manual"),
        ("us-ut", "manual", "ut-recovery"),
    ]
    source_paths = []
    for jurisdiction, document_class, version in scopes:
        source_path = (
            corpus
            / "data/corpus/provisions"
            / jurisdiction
            / document_class
            / f"{version}.jsonl"
        )
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text('{"citation_path":"fixture"}\n', encoding="utf-8")
        source_paths.append(source_path)
    subprocess.run(["git", "init", "-q", corpus], check=True)
    subprocess.run(["git", "-C", corpus, "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            corpus,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "corpus",
        ],
        check=True,
    )
    corpus_ref = subprocess.check_output(
        ["git", "-C", corpus, "rev-parse", "HEAD"],
        text=True,
    ).strip()
    release_name = "test-signed-release"
    release_sha = write_test_release_object(
        corpus,
        release_name,
        scopes,
        git_commit=corpus_ref,
    )
    release_object = corpus / "releases" / release_name / f"{release_sha}.json"
    public_key = tmp_path / "release-public-key"
    public_key.write_text(TEST_RELEASE_PUBLIC_KEY + "\n", encoding="utf-8")

    rulespec = tmp_path / "rulespec"
    toolchain = rulespec / ".axiom/toolchain.toml"
    toolchain.parent.mkdir(parents=True)
    toolchain.write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{release_name}"\n'
        f'axiom_corpus_release_content_sha256 = "{release_sha}"\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", rulespec], check=True)
    subprocess.run(["git", "-C", rulespec, "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            rulespec,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "rulespec",
        ],
        check=True,
    )
    rulespec_ref = subprocess.check_output(
        ["git", "-C", rulespec, "rev-parse", "HEAD"],
        text=True,
    ).strip()

    _verify_signed_release_binding(
        corpus,
        rulespec,
        corpus_ref=corpus_ref,
        rulespec_ref=rulespec_ref,
        release_name=release_name,
        release_content_sha256=release_sha,
        release_object_path=release_object,
        release_public_key_path=public_key,
        source_paths=source_paths,
    )

    source_paths[0].write_text('{"citation_path":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="differs from signed release"):
        _verify_signed_release_binding(
            corpus,
            rulespec,
            corpus_ref=corpus_ref,
            rulespec_ref=rulespec_ref,
            release_name=release_name,
            release_content_sha256=release_sha,
            release_object_path=release_object,
            release_public_key_path=public_key,
            source_paths=source_paths,
        )


def _tracked_dispatch_kwargs(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    dispatch = payload["dispatch"]
    selected = select_items(payload, item_ids="ut-0001", limit=1)["items"][0]
    return {
        "queue_id": payload["queue_id"],
        "item_id": "ut-0001",
        "manifest_sha256": queue_file_sha256(path),
        "item_generation_sha256_value": selected["generation_sha256"],
        "citation": payload["items"][0]["citation"],
        "country": dispatch["country"],
        "rulespec_ref": dispatch["rulespec_ref"],
        "pr_base_branch": dispatch["pr_base_branch"],
        "corpus_ref": dispatch["corpus_ref"],
        "rules_engine_ref": dispatch["rules_engine_ref"],
        "open_pr": dispatch["open_pr"],
    }


def test_validate_runtime_release_pin_matches_exact_queue(
    tmp_path: Path,
) -> None:
    payload = _queue()
    release = payload["release"]
    toolchain = tmp_path / "toolchain.toml"
    toolchain.write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{release["name"]}"\n'
        "axiom_corpus_release_content_sha256 = "
        f'"{release["content_sha256"]}"\n',
        encoding="utf-8",
    )

    validate_release_pin(
        QUEUE_PATH,
        toolchain_path=toolchain,
        manifest_sha256=queue_file_sha256(QUEUE_PATH),
    )

    toolchain.write_text(
        "[toolchain]\n"
        'axiom_corpus_release = "stale-release"\n'
        f'axiom_corpus_release_content_sha256 = "{"0" * 64}"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_release_pin(
            QUEUE_PATH,
            toolchain_path=toolchain,
            manifest_sha256=queue_file_sha256(QUEUE_PATH),
        )


def test_validate_tracked_dispatch_binds_exact_manifest_item_and_refs(
    tmp_path: Path,
) -> None:
    active_queue = tmp_path / "queue.json"
    active_queue.write_text(
        json.dumps(_queue(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_tracked_dispatch(
        active_queue,
        **_tracked_dispatch_kwargs(active_queue),
    )


def test_validate_tracked_dispatch_rejects_paused_queue() -> None:
    with pytest.raises(ValueError, match="paused"):
        validate_tracked_dispatch(
            QUEUE_PATH,
            **_tracked_dispatch_kwargs(QUEUE_PATH),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("manifest_sha256", "a" * 64, "digest"),
        ("item_generation_sha256_value", "a" * 64, "generation"),
        ("citation", "us-ut/manual/other", "citation"),
        ("rulespec_ref", "a" * 40, "trust inputs"),
        ("open_pr", False, "trust inputs"),
    ],
)
def test_validate_tracked_dispatch_rejects_spoofed_provenance(
    tmp_path: Path,
    field: str,
    value: str | bool,
    message: str,
) -> None:
    active_queue = tmp_path / "queue.json"
    active_queue.write_text(
        json.dumps(_queue(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    kwargs = _tracked_dispatch_kwargs(active_queue)
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        validate_tracked_dispatch(active_queue, **kwargs)
