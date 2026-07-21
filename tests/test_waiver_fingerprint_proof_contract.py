"""Contract: base proof validation is inside the waiver-audit fingerprint.

The shared validate workflow skips ``validate_failures``-waived modules in
both the validate step and the standalone proof-validate step
(TheAxiomFoundation/.github#38). That skip is sound only while the waiver
audit's fingerprinted validate outcome still includes base proof validation
— otherwise a standalone-only proof failure could hide behind a waiver
without drifting its fingerprint. These tests pin that contract so a future
refactor cannot silently decouple proof validation from the audited
pipeline.
"""

import json
import subprocess
from pathlib import Path

from axiom_encode import validation_waivers
from axiom_encode.harness import validator_pipeline
from axiom_encode.harness.proof_validator import validate_rulespec_proofs
from axiom_encode.harness.validator_pipeline import ValidatorPipeline
from tests.release_object_fixtures import bind_test_corpus_release

MODULE_WITH_UNRESOLVED_PROOF_SOURCE = """\
format: rulespec/v1
module:
  title: Example levy
  jurisdiction: nz
  citation_path: nz/statutes/example/levy
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: nz/statute/example/levy
rules:
  - name: example_levy_rate
    output: nz:statutes/example/levy#example_levy_rate
    metadata:
      proof:
        atoms:
          - kind: source
            path: versions[0].formula
            source:
              corpus_citation_path: nz/guidance/example/levy-rates
              excerpt: "The levy rate is 0.05"
    versions:
      - effective_from: '2026-04-01'
        formula: |-
          0.05
"""


def test_unresolved_proof_source_fails_base_proof_validation():
    """The same base proof check the standalone command runs reports an
    unresolved citation when the provision is absent from source_texts —
    the failure class .github#38's skip relies on the fingerprint to
    carry."""
    result = validate_rulespec_proofs(
        MODULE_WITH_UNRESOLVED_PROOF_SOURCE,
        require_policy_proofs=True,
        source_texts={},
    )

    assert result.passed is False
    assert any("Proof source unresolved" in issue for issue in result.issues)


def test_fingerprint_is_sensitive_to_base_proof_failures():
    """A validate outcome carrying a base proof failure fingerprints
    differently from one without it, so a waived module cannot change its
    proof outcome without fingerprint drift."""
    companion = {"present": True, "passed": True, "path": None, "cases": 1}
    clean = {"passed": False, "validators": {"ci": {"issues": ["other issue"]}}}
    with_proof_failure = {
        "passed": False,
        "validators": {
            "ci": {
                "issues": [
                    "other issue",
                    "Proof source unresolved: rule `example_levy_rate` proof "
                    "atom 0 cites `nz/guidance/example/levy-rates`, which was "
                    "not found in corpus.provisions.",
                ]
            }
        },
    }

    assert validation_waivers.fingerprint_outcome(
        clean, companion
    ) != validation_waivers.fingerprint_outcome(with_proof_failure, companion)


def _write_provisions(corpus_root: Path) -> None:
    """One statute provision (the module's source) and no guidance class,
    so the proof atom's guidance citation cannot resolve."""
    provisions = (
        corpus_root
        / "data"
        / "corpus"
        / "provisions"
        / "nz"
        / "statute"
        / "test-version.jsonl"
    )
    provisions.parent.mkdir(parents=True)
    row = {
        "id": "row-test-version",
        "citation_path": "nz/statute/example/levy",
        "body": "The levy rate is 0.05",
        "jurisdiction": "nz",
        "document_class": "statute",
        "version": "test-version",
        "source_path": "sources/nz/statute/test-version/source.xml",
        "source_as_of": "2026-01-02",
        "expression_date": "2026-01-01",
    }
    provisions.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")


def test_pipeline_ci_reports_unresolved_proof_sources(tmp_path, monkeypatch):
    """The audited ci pipeline (whose outcome the waiver audit fingerprints)
    itself reports unresolved proof sources. If a refactor moves this check
    out to the standalone proof-validate command only, the workflow's waiver
    skip of that step would hide proof failures — this test forces such a
    refactor to confront the contract."""
    policy_repo = tmp_path / "rulespec-nz"
    module_dir = policy_repo / "nz" / "statutes" / "example"
    module_dir.mkdir(parents=True)
    rules_file = module_dir / "levy.yaml"
    rules_file.write_text(MODULE_WITH_UNRESOLVED_PROOF_SOURCE)

    corpus_root = tmp_path / "axiom-corpus"
    _write_provisions(corpus_root)
    release = bind_test_corpus_release(
        corpus_root,
        "test-release",
        [("nz", "statute", "test-version")],
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        local_corpus_release=release,
        enable_oracles=False,
        require_policy_proofs=True,
    )

    def fake_compile(rules_file, output_path):
        return (
            subprocess.CompletedProcess(["axiom-rules-engine"], 0, "", ""),
            {"program": {"parameters": [], "derived": [], "relations": []}},
        )

    monkeypatch.setattr(pipeline, "_compile_rulespec_to_artifact", fake_compile)

    with validator_pipeline._authoritative_corpus_scope(release):
        result = pipeline._run_rulespec_ci(rules_file)

    assert not result.passed
    assert any(
        "Proof source unresolved" in issue and "nz/guidance/example/levy-rates" in issue
        for issue in result.issues
    ), (
        "The ci pipeline no longer reports unresolved proof sources; the "
        "shared workflow's waiver skip of the standalone proof-validate step "
        "is only sound while proof failures stay inside the audited, "
        "fingerprinted validate outcome. Issues were:\n" + "\n".join(result.issues)
    )
    assert result.details["deterministic_gates"]["proof-revalidation"] is False
    assert result.details["deterministic_gates"]["grounding-contract"] is True
    assert set(result.details["deterministic_gates"]) == {
        "proof-revalidation",
        "companion-tests",
        "grounding-contract",
        "layout-inspection",
    }


def test_pipeline_ci_preflight_fails_deterministic_gate_details_closed(tmp_path):
    policy_repo = tmp_path / "rulespec-nz" / "nz"
    rules_file = policy_repo / "statutes" / "example" / "invalid.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("format: [\n", encoding="utf-8")
    corpus_root = tmp_path / "axiom-corpus"
    _write_provisions(corpus_root)
    release = bind_test_corpus_release(
        corpus_root,
        "test-release",
        [("nz", "statute", "test-version")],
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        local_corpus_release=release,
        enable_oracles=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert result.details == {
        "deterministic_gates": {
            "proof-revalidation": False,
            "companion-tests": False,
            "grounding-contract": False,
            "layout-inspection": False,
        }
    }
