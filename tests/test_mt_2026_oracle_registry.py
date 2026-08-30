import hashlib
import re
import tomllib
from pathlib import Path

import yaml
from axiom_oracles.bridges.coverage import build_policyengine_coverage_report
from axiom_oracles.bridges.registry import load_policyengine_registry

MODULE = "us-mt:policies/income_tax/pilot_liability_pipeline"
OUTPUT_NAME = "mt_pit_pilot_income_tax_liability"
POLICYENGINE_VARIABLE = "mt_income_tax_before_non_refundable_credits_joint"
ORACLE_MERGE = "e1374eb30c582639f8f71f9bf9c22ba93b6e36f4"
ENCODER_VERSION = "0.2.1749"
FALLBACK_TEXT = """  - legal_id_prefix: "us-mt:"
    country: us
    mapping_type: not_comparable
    candidate_priority: P4
    rationale: PolicyEngine-US does not model every Montana statute, agency policy manual, or state regulation at output granularity; independently reviewed comparable outputs carry exact mappings which take precedence over this jurisdiction-wide fallback."""


def _module_mappings(document):
    prefix = f"{MODULE}#"
    return {
        item["legal_id"].removeprefix(prefix): item
        for item in document["mappings"]
        if item.get("legal_id", "").startswith(prefix)
    }


def _shared_material(text: str) -> str:
    start = text.index("  # Montana's bounded TY2026 tax before nonrefundable credits.")
    exact_end_marker = (
        "    rationale: Both outputs compose Montana's complete temporary "
        "tax-year-2026 ordinary-income and net-long-term-capital-gain schedules "
        "under MCA 15-30-2103 from the same completed-return boundaries, before "
        "nonrefundable credits; this narrow mapping excludes credits, payments, "
        "and final annual liability."
    )
    exact_end = text.index(exact_end_marker, start) + len(exact_end_marker)
    fallback_start = text.index(FALLBACK_TEXT)
    fallback_end = fallback_start + len(FALLBACK_TEXT)
    return f"{text[start:exact_end]}\n\n{text[fallback_start:fallback_end]}"


def _write_synthetic_module(root: Path) -> None:
    path = root / "us-mt/policies/income_tax/pilot_liability_pipeline.yaml"
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


def test_packaged_mt_2026_registry_has_one_exact_direct_mapping() -> None:
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
    for bounded_scope in (
        "complete temporary tax-year-2026",
        "ordinary-income",
        "net-long-term-capital-gain",
        "MCA 15-30-2103",
        "completed-return boundaries",
        "before nonrefundable credits",
        "credits",
        "payments",
        "final annual liability",
    ):
        assert bounded_scope in mapping["rationale"]
    for excluded_name in (
        "input.mt_pit_pilot_state_taxable_income",
        "input.mt_pit_pilot_section_1222_net_long_term_capital_gain",
        "input.mt_pit_pilot_filing_status_joint_or_surviving_spouse",
        "input.mt_pit_pilot_filing_status_head_of_household",
        "mt_pit_pilot_taxable_income",
        "mt_pit_pilot_net_long_term_capital_gain",
        "mt_pit_pilot_nonqualified_taxable_income",
        "mt_pit_pilot_filing_status_threshold",
        "mt_pit_pilot_ordinary_income_tax",
        "mt_pit_pilot_capital_gain_lower_band",
        "mt_pit_pilot_capital_gains_tax",
    ):
        assert excluded_name not in mappings


def test_packaged_mt_2026_runtime_pin_version_and_precedence_are_exact() -> None:
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
        "e87e011316a69dac2f45fee3e33d3340d57e0b5d9bb8f3a74f2fe719d4211efb"
    )
    assert hashlib.sha256(_shared_material(bundled_text).encode()).hexdigest() == (
        "9c92210f6d6e63d7dfe46c22da19c8fc7a9149640efcb0cadd6b86d73f7a2cf0"
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
        "input.mt_pit_pilot_state_taxable_income",
        "input.mt_pit_pilot_section_1222_net_long_term_capital_gain",
        "input.mt_pit_pilot_filing_status_joint_or_surviving_spouse",
        "input.mt_pit_pilot_filing_status_head_of_household",
        "mt_pit_pilot_taxable_income",
        "mt_pit_pilot_net_long_term_capital_gain",
        "mt_pit_pilot_nonqualified_taxable_income",
        "mt_pit_pilot_filing_status_threshold",
        "mt_pit_pilot_ordinary_income_tax",
        "mt_pit_pilot_capital_gain_lower_band",
        "mt_pit_pilot_capital_gains_tax",
        "future_unmapped_output",
    ):
        fallback = registry.mapping_for_legal_id(
            f"{MODULE}#{output_name}",
            country="us",
        )
        assert fallback is not None
        assert fallback.legal_id == "us-mt:"
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


def test_policyengine_coverage_classifies_only_bounded_mt_2026_output(
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
