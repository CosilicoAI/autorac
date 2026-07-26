from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

import scripts.prepare_signed_queue as signed_queue
from scripts.prepare_signed_queue import (
    ALL_STATE_EXPECTED_COUNTS,
    ALL_STATE_INVENTORY_PATH,
    ALL_STATE_INVENTORY_SHA256,
    ALL_STATE_QUEUE_ID,
    _logical_source_units,
    _queue_profile,
    build_all_state_snap_queue,
    validate_queue,
    verify_activation_commit,
    verify_merge_authorization,
)

ROOT = Path(__file__).parents[1]


def _all_state_queue() -> dict:
    items = []
    sequence = 0
    for jurisdiction, count in ALL_STATE_EXPECTED_COUNTS.items():
        if jurisdiction == "total":
            continue
        for index in range(1, count + 1):
            sequence += 1
            items.append(
                {
                    "attempt": 1,
                    "citation": (f"{jurisdiction}/manual/snap/source-{index:05d}"),
                    "id": f"{jurisdiction[3:]}-{index:05d}",
                    "jurisdiction": jurisdiction,
                    "label": f"{jurisdiction} source {index}",
                    "sequence": sequence,
                    "source_kind": "document",
                    "status": "pending",
                }
            )
    return {
        "schema": "axiom-encode/signed-encoding-queue/v1",
        "queue_id": ALL_STATE_QUEUE_ID,
        "state": "paused",
        "pause_reason": "Awaiting protected activation.",
        "description": "Test all-state queue.",
        "issue": 1287,
        "supersedes": ["us-snap-or-ut-2026-07"],
        "inventory": {
            "jurisdictions": 51,
            "path": ALL_STATE_INVENTORY_PATH,
            "sha256": ALL_STATE_INVENTORY_SHA256,
        },
        "release": {
            "content_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "name": "test-all-state-snap-release",
        },
        "dispatch": {
            "corpus_ref": "a" * 40,
            "country": "us",
            "max_batch_size": 4,
            "open_pr": True,
            "pr_base_branch": "hard-cut/canonical-layout-us",
            "rules_engine_ref": "b" * 40,
            "rulespec_ref": "c" * 40,
        },
        "expected_counts": ALL_STATE_EXPECTED_COUNTS,
        "items": items,
    }


def test_all_state_profile_covers_50_states_and_dc() -> None:
    profile = _queue_profile(ALL_STATE_QUEUE_ID)
    jurisdictions = set(profile["expected_counts"]) - {"total"}

    assert len(jurisdictions) == 51
    assert profile["expected_counts"]["total"] == sum(
        profile["expected_counts"][jurisdiction] for jurisdiction in jurisdictions
    )
    assert profile["issue"] == 1287
    assert profile["item_width"] == 5


def test_all_state_queue_accepts_five_digit_state_items() -> None:
    validate_queue(_all_state_queue())


def test_all_state_builder_is_deterministic_and_resolves_ma_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_root = tmp_path / "corpus"
    rulespec_root = tmp_path / "rulespec"
    rulespec_root.mkdir()
    jurisdictions = sorted(set(ALL_STATE_EXPECTED_COUNTS) - {"total"})
    expected_counts = {
        "total": len(jurisdictions),
        **dict.fromkeys(jurisdictions, 1),
    }
    monkeypatch.setattr(
        signed_queue,
        "ALL_STATE_EXPECTED_COUNTS",
        expected_counts,
    )

    states = []
    release_scopes = []
    expected_paths = []
    for jurisdiction in jurisdictions:
        inventory_version = f"test-{jurisdiction}-snap"
        release_version = inventory_version
        if jurisdiction == "us-ma":
            inventory_version = "2026-07-17-ma-dta-snap-regulations"
            release_version = "2026-07-24-ma-dta-regulations-snap-current-union"
        states.append(
            {
                "jurisdiction": jurisdiction,
                "queue_status": "published_current",
                "target_scope": {
                    "jurisdiction": jurisdiction,
                    "document_class": "regulation",
                    "version": inventory_version,
                },
            }
        )
        release_scopes.append(
            {
                "jurisdiction": jurisdiction,
                "document_class": "regulation",
                "version": release_version,
            }
        )
        source_path = (
            corpus_root
            / "data/corpus/provisions"
            / jurisdiction
            / "regulation"
            / f"{release_version}.jsonl"
        )
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(
            json.dumps(
                {
                    "citation_label": f"{jurisdiction} SNAP",
                    "citation_path": f"{jurisdiction}/regulation/snap",
                    "kind": "document",
                    "ordinal": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        expected_paths.append(source_path)

    inventory_path = corpus_root / ALL_STATE_INVENTORY_PATH
    inventory_path.parent.mkdir(parents=True)
    inventory_path.write_text(
        yaml.safe_dump({"program": "SNAP", "states": states}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        signed_queue,
        "ALL_STATE_INVENTORY_SHA256",
        hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
    )
    release_name = "test-all-state-snap-release"
    release_path = corpus_root / "manifests/releases" / f"{release_name}.json"
    release_path.parent.mkdir(parents=True)
    release_path.write_text(
        json.dumps({"scopes": release_scopes}),
        encoding="utf-8",
    )

    verified_paths = []
    monkeypatch.setattr(
        signed_queue,
        "_verify_corpus_provenance",
        lambda *args, **kwargs: "d" * 64,
    )

    def record_signed_binding(*args: object, **kwargs: object) -> None:
        verified_paths.extend(kwargs["source_paths"])

    monkeypatch.setattr(
        signed_queue,
        "_verify_signed_release_binding",
        record_signed_binding,
    )
    arguments = {
        "corpus_ref": "a" * 40,
        "rulespec_ref": "b" * 40,
        "rules_engine_ref": "c" * 40,
        "release_name": release_name,
        "release_content_sha256": "e" * 64,
        "release_object_path": tmp_path / "release-object.json",
        "release_public_key_path": tmp_path / "release-public-key",
        "state": "paused",
        "pause_reason": "Awaiting protected activation.",
    }

    first = build_all_state_snap_queue(
        corpus_root,
        rulespec_root,
        **arguments,
    )
    second = build_all_state_snap_queue(
        corpus_root,
        rulespec_root,
        **arguments,
    )

    assert first == second
    assert len(first["items"]) == 51
    assert first["items"][0]["id"] == "ak-00001"
    assert first["items"][-1]["id"] == "wy-00001"
    assert set(verified_paths) == set(expected_paths)
    assert any(
        path.name == "2026-07-24-ma-dta-regulations-snap-current-union.jsonl"
        for path in verified_paths
    )


def test_logical_source_units_prefer_document_topics() -> None:
    records = [
        {
            "citation_path": "us-ut/manual/topic-a",
            "citation_label": "Topic A",
            "kind": "document",
            "ordinal": 1,
        },
        {
            "citation_path": "us-ut/manual/topic-a/block-1",
            "citation_label": "Topic A block",
            "kind": "block",
            "ordinal": 2,
        },
        {
            "citation_path": "us-ut/manual/topic-b",
            "citation_label": "Topic B",
            "kind": "document",
            "ordinal": 3,
        },
    ]

    assert [record["citation_path"] for record in _logical_source_units(records)] == [
        "us-ut/manual/topic-a",
        "us-ut/manual/topic-b",
    ]


def test_logical_source_units_select_rules_from_code_hierarchy() -> None:
    records = [
        {
            "citation_path": "us-oh/regulation",
            "citation_label": "Collection",
            "kind": "collection",
            "ordinal": 1,
        },
        {
            "citation_path": "us-oh/regulation/chapter-1",
            "citation_label": "Chapter",
            "kind": "chapter",
            "ordinal": 2,
        },
        {
            "citation_path": "us-oh/regulation/chapter-1/rule-1",
            "citation_label": "Rule",
            "kind": "rule",
            "ordinal": 3,
        },
    ]

    assert [record["citation_path"] for record in _logical_source_units(records)] == [
        "us-oh/regulation/chapter-1/rule-1"
    ]


def test_logical_source_units_fall_back_from_unsafe_section_citations() -> None:
    records = [
        {
            "citation_path": "us-nh/regulation/he-w-700",
            "citation_label": "He-W 700",
            "kind": "document",
            "ordinal": 1,
        },
        {
            "citation_path": "us-nh/regulation/he-w-700/He-W 701.01",
            "citation_label": "He-W 701.01",
            "kind": "section",
            "ordinal": 2,
        },
    ]

    assert [record["citation_path"] for record in _logical_source_units(records)] == [
        "us-nh/regulation/he-w-700"
    ]


def test_protected_workflows_route_the_selected_queue_id() -> None:
    for filename in (
        "dispatch-signed-snap-queue.yml",
        "finalize-signed-snap-queue.yml",
        "merge-snap-queue-activation.yml",
    ):
        workflow = (ROOT / ".github/workflows" / filename).read_text(encoding="utf-8")
        assert "us-snap-all-states-2026-07" in workflow
        assert 'queue="data/encoding-queues/${QUEUE_ID}.json"' in workflow


def test_protected_workflows_enforce_one_way_all_state_cutover() -> None:
    validation = (
        ROOT / ".github/workflows/validate-snap-queue-activation.yml"
    ).read_text(encoding="utf-8")
    finalizer = (ROOT / ".github/workflows/finalize-signed-snap-queue.yml").read_text(
        encoding="utf-8"
    )
    merger = (ROOT / ".github/workflows/merge-snap-queue-activation.yml").read_text(
        encoding="utf-8"
    )

    assert "'data/encoding-queues/us-snap-*.json'" not in validation
    assert "QUEUE_PATH: ${{ steps.transition.outputs.queue_path }}" in validation
    assert '"${{ steps.transition.outputs.queue_path }}"' not in validation
    for workflow in (validation, finalizer, merger):
        assert "legacy SNAP queue is permanently superseded" in workflow


def test_all_state_activation_and_merge_authorization_bind_queue_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_path = tmp_path / f"{ALL_STATE_QUEUE_ID}.json"
    queue_path.write_text(
        json.dumps({"queue_id": ALL_STATE_QUEUE_ID, "state": "active"}) + "\n",
        encoding="utf-8",
    )
    queue_sha = hashlib.sha256(queue_path.read_bytes()).hexdigest()
    activation_path = f"data/encoding-queues/{ALL_STATE_QUEUE_ID}.json"
    changed_files = [
        activation_path,
        "pyproject.toml",
        "src/axiom_encode/__init__.py",
        "uv.lock",
    ]
    provenance = {
        "base_sha": "a" * 40,
        "changed_files": changed_files,
        "head_sha": "b" * 40,
        "queue_path": activation_path,
        "queue_sha256": queue_sha,
        "repository": "TheAxiomFoundation/axiom-encode",
        "schema": "axiom-encode/snap-queue-activation-commit/v1",
        "tree_sha": "c" * 40,
    }
    pull_request = {
        "state": "open",
        "base": {"ref": "main", "sha": "a" * 40},
        "head": {
            "sha": "b" * 40,
            "repo": {"full_name": "TheAxiomFoundation/axiom-encode"},
        },
    }
    verify_activation_commit(
        queue_path,
        provenance=provenance,
        pull_request=pull_request,
        current_base_sha="a" * 40,
        current_head_sha="b" * 40,
        current_tree_sha="c" * 40,
        current_changed_files=changed_files,
    )

    monkeypatch.setattr(signed_queue, "validate_queue", lambda payload: None)
    run_id = 123
    run_url = (
        f"https://github.com/TheAxiomFoundation/axiom-encode/actions/runs/{run_id}"
    )
    authorization = {
        "activation_pr_head_sha": "b" * 40,
        "activation_pr_number": 17,
        "merge_commit": "d" * 40,
        "merge_workflow_run_id": run_id,
        "merge_workflow_run_attempt": 1,
        "merge_workflow_run_url": run_url,
        "queue_path": activation_path,
        "queue_sha256": queue_sha,
        "repository": "TheAxiomFoundation/axiom-encode",
        "schema": "axiom-encode/snap-queue-merge-authorization/v1",
    }
    merge_run = {
        "event": "workflow_dispatch",
        "head_branch": "main",
        "html_url": run_url,
        "path": ".github/workflows/merge-snap-queue-activation.yml",
        "run_attempt": 1,
        "status": "completed",
        "conclusion": "success",
    }
    merge_jobs = {
        "jobs": [
            {
                "name": "Merge reviewed SNAP queue activation",
                "run_attempt": 1,
                "status": "completed",
                "conclusion": "success",
            }
        ]
    }
    merged_pull_request = {
        "number": 17,
        "state": "closed",
        "merged_at": "2026-07-26T00:00:00Z",
        "merge_commit_sha": "d" * 40,
        "base": {"ref": "main"},
        "head": {
            "sha": "b" * 40,
            "repo": {"full_name": "TheAxiomFoundation/axiom-encode"},
        },
        "merged_by": {"login": "github-actions[bot]"},
    }
    verify_merge_authorization(
        queue_path,
        authorization=authorization,
        merge_run=merge_run,
        merge_jobs=merge_jobs,
        pull_request=merged_pull_request,
        current_head_sha="e" * 40,
        queue_change_sha="d" * 40,
    )

    tampered = copy.deepcopy(authorization)
    tampered["queue_path"] = "data/encoding-queues/us-snap-or-ut-2026-07.json"
    with pytest.raises(
        ValueError,
        match="merge authorization does not match",
    ):
        verify_merge_authorization(
            queue_path,
            authorization=tampered,
            merge_run=merge_run,
            merge_jobs=merge_jobs,
            pull_request=merged_pull_request,
            current_head_sha="e" * 40,
            queue_change_sha="d" * 40,
        )


def test_all_state_preparation_uses_authenticated_release_builder() -> None:
    workflow = (ROOT / ".github/workflows/prepare-all-state-snap-queue.yml").read_text(
        encoding="utf-8"
    )

    assert "build-snap-all-states" in workflow
    assert "AXIOM_CORPUS_RELEASE_PUBLIC_KEY" in workflow
    assert "NEXT_PUBLIC_SUPABASE_ANON_KEY" in workflow
    assert "--state paused" in workflow
