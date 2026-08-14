from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from axiom_encode.toolchain import RuleSpecToolchainError


def load_queue_runner_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_snap_queue_until_idle.py"
    )
    module_name = f"run_snap_queue_until_idle_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_sha256_paths_ignores_file_names(tmp_path):
    module = load_queue_runner_module()
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    first_left = left / "alpha.txt"
    second_left = left / "beta.txt"
    first_right = right / "renamed-one.txt"
    second_right = right / "renamed-two.txt"
    first_left.write_text("alpha")
    second_left.write_text("beta")
    first_right.write_text("alpha")
    second_right.write_text("beta")

    left_digest = module.sha256_paths([first_left, second_left])
    right_digest = module.sha256_paths([first_right, second_right])

    assert left_digest == right_digest


def test_subprocess_env_drops_removed_policyengine_interpreter_overrides(
    monkeypatch,
):
    module = load_queue_runner_module()
    monkeypatch.setenv("AXIOM_ENCODE_POLICYENGINE_US_PYTHON", "/ambient/us")
    monkeypatch.setenv("AXIOM_ENCODE_POLICYENGINE_UK_PYTHON", "/ambient/uk")

    env = module.build_subprocess_env()

    assert "AXIOM_ENCODE_POLICYENGINE_US_PYTHON" not in env
    assert "AXIOM_ENCODE_POLICYENGINE_UK_PYTHON" not in env


def test_iter_manifest_queue_candidates_includes_federal_rules_us(
    tmp_path, monkeypatch
):
    module = load_queue_runner_module()
    axiom_encode_root = tmp_path / "axiom_encode"
    benchmarks = axiom_encode_root / "benchmarks"
    benchmarks.mkdir(parents=True)
    manifest = benchmarks / "us_snap_federal_demo_refresh.yaml"
    manifest.write_text(
        "name: Federal demo\n"
        "cases:\n"
        "  - kind: source\n"
        "    name: federal_snap_demo\n"
        "    corpus_citation_path: us/statute/7/2017/a\n"
    )
    monkeypatch.setattr(module, "AXIOM_ENCODE_ROOT", axiom_encode_root)
    monkeypatch.setattr(
        module,
        "sha256_corpus_source",
        lambda corpus_citation_path, corpus_root, policy_repo_root: "corpus-source-sha",
    )

    candidates = module.iter_manifest_queue_candidates(
        tmp_path / "axiom-corpus",
        tmp_path / "rulespec-us",
    )

    assert candidates == [
        {
            "name": "federal_snap_demo",
            "manifest": str(manifest),
            "corpus_citation_path": "us/statute/7/2017/a",
            "corpus_source_sha256": "corpus-source-sha",
            "source_repo": "rulespec-us",
        }
    ]


def test_infer_repo_routes_state_citations_to_canonical_us_monorepo():
    module = load_queue_runner_module()

    assert module.infer_repo("us/statute/7/2017/a") == "rulespec-us"
    assert module.infer_repo("us-tn/regulation/demo/snap") == "rulespec-us"
    assert module.infer_repo("nz/statute/demo") == "rulespec-nz"


def test_sha256_corpus_source_propagates_toolchain_configuration_errors(
    tmp_path, monkeypatch
):
    module = load_queue_runner_module()
    workspace = tmp_path / "workspace"
    axiom_encode_root = workspace / "axiom-encode"
    (workspace / "rulespec-us").mkdir(parents=True)
    axiom_encode_root.mkdir()
    monkeypatch.setattr(module, "AXIOM_ENCODE_ROOT", axiom_encode_root)

    with pytest.raises(RuleSpecToolchainError, match="toolchain.toml"):
        module.sha256_corpus_source(
            "us-tn/regulation/demo/snap",
            workspace / "axiom-corpus",
            workspace / "rulespec-us",
        )


def test_sha256_corpus_source_binds_named_release_identity(tmp_path, monkeypatch):
    module = load_queue_runner_module()
    workspace = tmp_path / "workspace"
    axiom_encode_root = workspace / "axiom-encode"
    axiom_encode_root.mkdir(parents=True)
    monkeypatch.setattr(module, "AXIOM_ENCODE_ROOT", axiom_encode_root)
    content_sha256 = "a" * 64

    def load_release(*_args, **_kwargs):
        return SimpleNamespace(
            name="snap-release",
            content_sha256=content_sha256,
        )

    monkeypatch.setattr(module, "load_rulespec_local_corpus_release", load_release)
    monkeypatch.setattr(
        module,
        "resolve_corpus_source_unit",
        lambda *_args, **_kwargs: SimpleNamespace(
            citation_path="us-tn/regulation/demo/snap",
            body="unchanged body",
        ),
    )

    first = module.sha256_corpus_source(
        "us-tn/regulation/demo/snap",
        workspace / "axiom-corpus",
        workspace / "rulespec-us",
    )
    content_sha256 = "b" * 64
    second = module.sha256_corpus_source(
        "us-tn/regulation/demo/snap",
        workspace / "axiom-corpus",
        workspace / "rulespec-us",
    )

    assert first != second


def test_run_eval_item_uses_manifest_runners_and_required_dependency_roots(
    tmp_path, monkeypatch
):
    module = load_queue_runner_module()
    axiom_encode_root = tmp_path / "axiom-encode"
    corpus_root = tmp_path / "axiom-corpus"
    axiom_rules_root = tmp_path / "axiom-rules-engine"
    policy_repo_root = tmp_path / "rulespec-us"
    policyengine_runtime_root = tmp_path / "policyengine-us"
    axiom_encode_root.mkdir()
    corpus_root.mkdir()
    axiom_rules_root.mkdir()
    policy_repo_root.mkdir()
    policyengine_runtime_root.mkdir()
    recorded: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["kwargs"] = kwargs
        return module.subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(module, "AXIOM_ENCODE_ROOT", axiom_encode_root)
    monkeypatch.setattr(module, "resolve_uv_bin", lambda: "uv")
    monkeypatch.setattr(module, "build_subprocess_env", lambda: {})
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    returncode = module.run_eval_item(
        {"manifest": "benchmarks/us_snap_demo_refresh.yaml"},
        reviewer_cli="claude",
        output_dir=tmp_path / "output",
        corpus_root=corpus_root,
        axiom_rules_root=axiom_rules_root,
        policy_repo_root=policy_repo_root,
        policyengine_runtime_root=policyengine_runtime_root,
    )

    assert returncode == 0
    cmd = recorded["cmd"]
    assert isinstance(cmd, list)
    corpus_flag = cmd.index("--corpus-path")
    assert cmd[corpus_flag + 1] == str(corpus_root)
    engine_flag = cmd.index("--axiom-rules-engine-path")
    assert cmd[engine_flag + 1] == str(axiom_rules_root)
    policy_flag = cmd.index("--policy-repo-path")
    assert cmd[policy_flag + 1] == str(policy_repo_root)
    runtime_flag = cmd.index("--policyengine-runtime-root")
    assert cmd[runtime_flag + 1] == str(policyengine_runtime_root)
    assert "--gpt-backend" not in cmd


def test_archive_eval_passes_required_dependency_roots(tmp_path, monkeypatch):
    module = load_queue_runner_module()
    axiom_encode_root = tmp_path / "axiom-encode"
    corpus_root = tmp_path / "axiom-corpus"
    axiom_rules_root = tmp_path / "axiom-rules-engine"
    policy_repo_root = tmp_path / "rulespec-us"
    policyengine_runtime_root = tmp_path / "policyengine-us"
    output_dir = tmp_path / "output"
    for path in (
        axiom_encode_root,
        corpus_root,
        axiom_rules_root,
        policy_repo_root,
        policyengine_runtime_root,
        output_dir,
    ):
        path.mkdir()
    recorded: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["kwargs"] = kwargs
        return module.subprocess.CompletedProcess(
            cmd,
            0,
            stdout=f"Archived eval suite to {tmp_path / 'archive'}\n",
            stderr="",
        )

    monkeypatch.setattr(module, "AXIOM_ENCODE_ROOT", axiom_encode_root)
    monkeypatch.setattr(module, "resolve_uv_bin", lambda: "uv")
    monkeypatch.setattr(module, "build_subprocess_env", lambda: {})
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    archived = module.archive_eval(
        output_dir,
        corpus_root=corpus_root,
        axiom_rules_root=axiom_rules_root,
        policy_repo_root=policy_repo_root,
        policyengine_runtime_root=policyengine_runtime_root,
    )

    assert archived == tmp_path / "archive"
    cmd = recorded["cmd"]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("--corpus-path") + 1] == str(corpus_root)
    assert cmd[cmd.index("--axiom-rules-engine-path") + 1] == str(axiom_rules_root)
    assert cmd[cmd.index("--policy-repo-path") + 1] == str(policy_repo_root)
    assert cmd[cmd.index("--policyengine-runtime-root") + 1] == str(
        policyengine_runtime_root
    )


def test_sync_queue_with_manifests_requeues_unattested_tracking_version(
    monkeypatch,
):
    module = load_queue_runner_module()
    data = {
        "items": [
            {
                "name": "snap_demo",
                "status": "done",
                "manifest": "/tmp/demo.yaml",
                "corpus_citation_path": "us/statute/7/2017/a",
                "manifest_sha256": "manifest-sha",
                "corpus_source_sha256": "old-source-sha",
                "note": "closed fully ready",
            }
        ]
    }
    monkeypatch.setattr(
        module,
        "iter_manifest_queue_candidates",
        lambda corpus_root, policy_repo_root: [
            {
                "name": "snap_demo",
                "manifest": "/tmp/demo.yaml",
                "corpus_citation_path": "us/statute/7/2017/a",
                "corpus_source_sha256": "new-source-sha",
                "source_repo": "rulespec-us-demo",
            }
        ],
    )
    monkeypatch.setattr(module, "sha256_file", lambda path: "manifest-sha")

    changed, added, retired, refreshed = module.sync_queue_with_manifests(
        data,
        Path("/tmp/axiom-corpus"),
        Path("/tmp/rulespec-us"),
    )

    assert changed is True
    assert added == []
    assert retired == []
    assert refreshed == ["snap_demo"]
    assert data["items"][0]["status"] == "queued"
    assert data["items"][0]["source_tracking_version"] == module.SOURCE_TRACKING_VERSION
    assert data["items"][0]["note"] == "requeued after manifest/source identity change"


def test_sync_queue_with_manifests_requeues_on_source_change_after_tracking_set(
    monkeypatch,
):
    module = load_queue_runner_module()
    data = {
        "items": [
            {
                "name": "snap_demo",
                "status": "done",
                "manifest": "/tmp/demo.yaml",
                "corpus_citation_path": "us/statute/7/2017/a",
                "manifest_sha256": "manifest-sha",
                "corpus_source_sha256": "old-source-sha",
                "source_tracking_version": module.SOURCE_TRACKING_VERSION,
                "output_dir": "/tmp/out",
                "archive_path": "/tmp/archive",
                "finished_at": "2026-04-15T00:00:00Z",
                "note": "closed fully ready",
            }
        ]
    }
    monkeypatch.setattr(
        module,
        "iter_manifest_queue_candidates",
        lambda corpus_root, policy_repo_root: [
            {
                "name": "snap_demo",
                "manifest": "/tmp/demo.yaml",
                "corpus_citation_path": "us/statute/7/2017/a",
                "corpus_source_sha256": "new-source-sha",
                "source_repo": "rulespec-us-demo",
            }
        ],
    )
    monkeypatch.setattr(module, "sha256_file", lambda path: "manifest-sha")

    changed, added, retired, refreshed = module.sync_queue_with_manifests(
        data,
        Path("/tmp/axiom-corpus"),
        Path("/tmp/rulespec-us"),
    )

    assert changed is True
    assert added == []
    assert retired == []
    assert refreshed == ["snap_demo"]
    assert data["items"][0]["status"] == "queued"
    assert data["items"][0]["output_dir"] is None
    assert data["items"][0]["archive_path"] is None
    assert data["items"][0]["finished_at"] is None
    assert data["items"][0]["note"] == "requeued after manifest/source identity change"


def test_sync_queue_does_not_relabel_in_flight_run_identity(monkeypatch):
    module = load_queue_runner_module()
    data = {
        "items": [
            {
                "name": "snap_demo",
                "status": "running",
                "manifest": "/tmp/old-demo.yaml",
                "corpus_citation_path": "us/statute/7/2017/a",
                "manifest_sha256": "old-manifest-sha",
                "corpus_source_sha256": "old-source-sha",
                "source_tracking_version": 1,
            }
        ]
    }
    monkeypatch.setattr(
        module,
        "iter_manifest_queue_candidates",
        lambda corpus_root, policy_repo_root: [
            {
                "name": "snap_demo",
                "manifest": "/tmp/new-demo.yaml",
                "corpus_citation_path": "us/statute/7/2017/a",
                "corpus_source_sha256": "new-source-sha",
                "source_repo": "rulespec-us",
            }
        ],
    )
    monkeypatch.setattr(module, "sha256_file", lambda path: "new-manifest-sha")

    changed, added, retired, refreshed = module.sync_queue_with_manifests(
        data,
        Path("/tmp/axiom-corpus"),
        Path("/tmp/rulespec-us"),
    )

    assert changed is False
    assert added == []
    assert retired == []
    assert refreshed == []
    assert data["items"][0]["manifest"] == "/tmp/old-demo.yaml"
    assert data["items"][0]["manifest_sha256"] == "old-manifest-sha"
    assert data["items"][0]["corpus_source_sha256"] == "old-source-sha"
    assert data["items"][0]["source_tracking_version"] == 1


def test_reconcile_ready_output_items_marks_revalidated_blocked_item_done(tmp_path):
    module = load_queue_runner_module()
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "summary.json").write_text('{"all_ready": true}')
    data = {
        "items": [
            {
                "name": "snap_demo",
                "status": "blocked",
                "output_dir": str(output_dir),
                "note": "eval-suite exited with code 1",
            }
        ]
    }

    changed, reconciled = module.reconcile_ready_output_items(data)

    assert changed is True
    assert reconciled == ["snap_demo"]
    assert data["items"][0]["status"] == "done"
    assert "source_tracking_version" not in data["items"][0]
    assert data["items"][0]["note"] == "closed fully ready after revalidation"


def test_reconcile_stale_running_items_marks_ready_orphan_done(tmp_path):
    module = load_queue_runner_module()
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "summary.json").write_text('{"all_ready": true}')
    data = {
        "items": [
            {
                "name": "snap_demo",
                "status": "running",
                "output_dir": str(output_dir),
                "note": "started with backend `codex`",
            }
        ]
    }

    changed = module.reconcile_stale_running_items(data, [])

    assert changed is True
    assert data["items"][0]["status"] == "done"
    assert "source_tracking_version" not in data["items"][0]
    assert (
        data["items"][0]["note"] == "closed fully ready after orphaned eval completed"
    )
    assert data["event_log"][0]["message"] == (
        "snap_demo was left in `running`, but its output is ready; marked done."
    )


def test_build_run_record_marks_unknown_row_costs_as_unknown(tmp_path):
    module = load_queue_runner_module()
    item = {"name": "snap_demo", "corpus_source_sha256": "abc123"}

    def record_for(rows):
        return module.build_run_record(
            item,
            reviewer_cli="codex",
            returncode=0,
            summary=None,
            results={"results": rows},
            archive_path=None,
            status="done",
            note="",
            policy_repo_root=tmp_path / "policy",
            policyengine_runtime_root=tmp_path / "runtime",
        )

    known = record_for(
        [
            {"estimated_cost_usd": 1.25, "input_tokens": 10},
            {"estimated_cost_usd": 0.75, "input_tokens": 10},
        ]
    )
    assert known["metrics"]["estimated_cost_usd"] == 2.0

    unknown = record_for(
        [
            {"estimated_cost_usd": 1.25, "input_tokens": 10},
            {"estimated_cost_usd": None, "input_tokens": 10},
        ]
    )
    assert unknown["metrics"]["estimated_cost_usd"] is None

    # A row with no usage at all (e.g. a failed launch) contributes no cost
    # and does not poison the aggregate.
    no_usage = record_for(
        [
            {"estimated_cost_usd": 1.25, "input_tokens": 10},
            {"estimated_cost_usd": None},
        ]
    )
    assert no_usage["metrics"]["estimated_cost_usd"] == 1.25
