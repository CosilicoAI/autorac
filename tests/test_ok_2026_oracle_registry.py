import hashlib
import re
import tomllib
from pathlib import Path

import yaml
from axiom_oracles.bridges.coverage import build_policyengine_coverage_report
from axiom_oracles.bridges.registry import load_policyengine_registry

MODULE = "us-ok:policies/income_tax/pilot_liability_pipeline"
OUTPUT_NAME = "ok_pit_pilot_income_tax_liability"
POLICYENGINE_VARIABLE = "ok_income_tax_before_credits"
ORACLE_MERGE = "e1374eb30c582639f8f71f9bf9c22ba93b6e36f4"
ENCODER_VERSION = "0.2.1745"
FALLBACK_TEXT = """  - legal_id_prefix: "us-ok:"
    country: us
    mapping_type: not_comparable
    candidate_priority: P4
    rationale: PolicyEngine-US does not model OK agency policy manuals or state regulations at output granularity; comparable state outputs carry exact mappings which take precedence over this prefix."""


def _module_mappings(document):
    prefix = f"{MODULE}#"
    return {
        item["legal_id"].removeprefix(prefix): item
        for item in document["mappings"]
        if item.get("legal_id", "").startswith(prefix)
    }


def _shared_material(text: str) -> str:
    start = text.index(
        "  # Oklahoma's bounded TY2026 individual income tax before credits."
    )
    exact_end_marker = (
        "    rationale: On the reviewed nonnegative completed-return boundary, "
        "both outputs apply Oklahoma's enacted tax-year-2026 four-band schedule "
        "to Oklahoma taxable income using the statutory single-or-separate "
        "versus joint, surviving-spouse, or head-of-household schedule branch, "
        "before credits, payments, or final annual liability."
    )
    exact_end = text.index(exact_end_marker, start) + len(exact_end_marker)
    fallback_start = text.index(FALLBACK_TEXT)
    fallback_end = fallback_start + len(FALLBACK_TEXT)
    return f"{text[start:exact_end]}\n\n{text[fallback_start:fallback_end]}"


def _write_synthetic_module(root: Path) -> None:
    path = root / "us-ok/policies/income_tax/pilot_liability_pipeline.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "format: rulespec/v1\n"
        "rules:\n"
        f"  - name: {OUTPUT_NAME}\n"
        "    kind: derived\n"
        "    versions:\n"
        "      - effective_from: '2026-01-01'\n"
        "        formula: 0\n"
    )


def test_packaged_ok_2026_registry_has_one_exact_direct_mapping() -> None:
    root = Path(__file__).parents[1]
    path = root / "src/axiom_encode/oracles/policyengine/mappings/us.yaml"
    document = yaml.safe_load(path.read_text())
    mappings = _module_mappings(document)

    assert set(mappings) == {OUTPUT_NAME}
    mapping = mappings[OUTPUT_NAME]
    assert mapping["mapping_type"] == "direct_variable"
    assert mapping["policyengine_variable"] == POLICYENGINE_VARIABLE
    assert (
        mapping["program"],
        mapping["entity"],
        mapping["period"],
        mapping["unit"],
        mapping["comparison"],
    ) == ("tax", "tax_unit", "year", "USD", "money")
    assert "candidate_priority" not in mapping
    assert "reviewed nonnegative completed-return boundary" in mapping["rationale"]
    for bounded_scope in (
        "tax-year-2026",
        "four-band schedule",
        "before credits",
        "payments",
        "final annual liability",
    ):
        assert bounded_scope in mapping["rationale"]
    for excluded_name in (
        "input.ok_pit_pilot_state_taxable_income",
        "input.ok_pit_pilot_filing_status_uses_wide_schedule",
        "ok_pit_pilot_taxable_income",
        "ok_pit_pilot_schedule_tax",
    ):
        assert excluded_name not in mappings


def test_packaged_ok_2026_runtime_pin_version_and_precedence_are_exact() -> None:
    import axiom_oracles.bridges.registry as runtime_registry_module

    root = Path(__file__).parents[1]
    bundled_path = root / "src/axiom_encode/oracles/policyengine/mappings/us.yaml"
    runtime_path = (
        Path(runtime_registry_module.__file__).with_name("mappings") / "us.yaml"
    )
    bundled_text = bundled_path.read_text()
    runtime_text = runtime_path.read_text()
    bundled_document = yaml.safe_load(bundled_text)
    runtime_document = yaml.safe_load(runtime_text)

    assert _module_mappings(bundled_document) == _module_mappings(runtime_document)
    assert _shared_material(bundled_text) == _shared_material(runtime_text)
    assert bundled_text.count(FALLBACK_TEXT) == 1
    assert runtime_text.count(FALLBACK_TEXT) == 1
    assert hashlib.sha256(FALLBACK_TEXT.encode()).hexdigest() == (
        "1c977a456afdb80469308da94b9dda0e630c53f79e504706ae11ebd8f5a3057a"
    )
    assert hashlib.sha256(_shared_material(bundled_text).encode()).hexdigest() == (
        "971de99c5084eda2a4de0ebc1fd763605d58524d81279033128c3b8d1db70fde"
    )

    registry = load_policyengine_registry()
    mapping = registry.mapping_for_legal_id(
        f"{MODULE}#{OUTPUT_NAME}",
        country="us",
    )
    assert mapping is not None
    assert mapping.match_type == "exact"
    assert mapping.mapping_type == "direct_variable"
    assert mapping.policyengine_variable == POLICYENGINE_VARIABLE
    assert mapping.entity == "tax_unit"
    assert mapping.period == "year"
    assert mapping.unit == "USD"
    assert mapping.comparison == "money"

    for output_name in (
        "input.ok_pit_pilot_state_taxable_income",
        "input.ok_pit_pilot_filing_status_uses_wide_schedule",
        "ok_pit_pilot_taxable_income",
        "ok_pit_pilot_schedule_tax",
        "future_unmapped_output",
    ):
        fallback = registry.mapping_for_legal_id(
            f"{MODULE}#{output_name}",
            country="us",
        )
        assert fallback is not None
        assert fallback.legal_id == "us-ok:"
        assert fallback.match_type == "prefix"
        assert fallback.mapping_type == "not_comparable"
        assert fallback.candidate_priority == "P4"

    dependency_pin = re.search(
        r"axiom-oracles@[0-9a-f]{40}",
        (root / "pyproject.toml").read_text(),
    )
    assert dependency_pin is not None
    assert dependency_pin.group(0).removeprefix("axiom-oracles@") == ORACLE_MERGE
    lock_text = (root / "uv.lock").read_text()
    assert lock_text.count(f"?rev={ORACLE_MERGE}") == 2
    assert f"?rev={ORACLE_MERGE}#{ORACLE_MERGE}" in lock_text
    lock = tomllib.loads(lock_text)
    encoder_package = next(
        package for package in lock["package"] if package["name"] == "axiom-encode"
    )
    assert encoder_package["version"] == ENCODER_VERSION
    project = tomllib.loads((root / "pyproject.toml").read_text())
    assert project["project"]["version"] == ENCODER_VERSION
    assert (
        (root / "src/axiom_encode/__init__.py")
        .read_text()
        .startswith(f'__version__ = "{ENCODER_VERSION}"')
    )


def test_policyengine_coverage_classifies_only_bounded_ok_2026_output(
    tmp_path: Path,
) -> None:
    rulespec_root = tmp_path / "rulespec-us"
    _write_synthetic_module(rulespec_root)

    report = build_policyengine_coverage_report(rulespec_root, program="tax")

    assert report["total_outputs"] == 1
    assert report["status_counts"] == {"comparable": 1}
    assert len(report["items"]) == 1
    item = report["items"][0]
    assert item["rule_name"] == OUTPUT_NAME
    assert item["policyengine_variable"] == POLICYENGINE_VARIABLE
