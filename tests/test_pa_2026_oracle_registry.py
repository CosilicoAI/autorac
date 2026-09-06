import hashlib
import re
import tomllib
from pathlib import Path

import yaml
from axiom_oracles.bridges.coverage import build_policyengine_coverage_report
from axiom_oracles.bridges.registry import load_policyengine_registry

MODULE = "us-pa:policies/income_tax/pilot_liability_pipeline"
INPUT_NAME = "pa_pit_pilot_state_taxable_income"
DIRECT_VARIABLES = {
    "pa_pit_pilot_taxable_income": "pa_adjusted_taxable_income",
    "pa_pit_pilot_income_tax_liability": "pa_income_tax_before_forgiveness",
}
ORACLE_MERGE = "e1374eb30c582639f8f71f9bf9c22ba93b6e36f4"
ENCODER_VERSION = "0.2.1767"
FALLBACK_TEXT = """  - legal_id_prefix: "us-pa:"
    country: us
    mapping_type: not_comparable
    candidate_priority: P4
    rationale: PolicyEngine-US does not model PA agency policy manuals or state regulations at output granularity; comparable state outputs carry exact mappings which take precedence over this prefix."""


def _module_mappings(document):
    prefix = f"{MODULE}#"
    return {
        item["legal_id"].removeprefix(prefix): item
        for item in document["mappings"]
        if item.get("legal_id", "").startswith(prefix)
    }


def _shared_material(text: str) -> str:
    start = text.index(
        "  # Pennsylvania's bounded TY2026 resident tax before forgiveness."
    )
    exact_end_marker = (
        "    rationale: On the reviewed nonnegative completed-return boundary, "
        "both outputs apply Pennsylvania's tax-year-2026 3.07 percent rate to "
        "adjusted taxable income before tax forgiveness, credits, payments, or "
        "final annual liability."
    )
    exact_end = text.index(exact_end_marker, start) + len(exact_end_marker)
    fallback_start = text.index(FALLBACK_TEXT)
    fallback_end = fallback_start + len(FALLBACK_TEXT)
    return f"{text[start:exact_end]}\n\n{text[fallback_start:fallback_end]}"


def _write_synthetic_module(root: Path) -> None:
    path = root / "us-pa/policies/income_tax/pilot_liability_pipeline.yaml"
    path.parent.mkdir(parents=True)
    rules = "\n".join(
        "  - name: "
        f"{name}\n"
        "    kind: derived\n"
        "    versions:\n"
        "      - effective_from: '2026-01-01'\n"
        "        formula: 0"
        for name in DIRECT_VARIABLES
    )
    path.write_text(f"format: rulespec/v1\nrules:\n{rules}\n")


def test_packaged_pa_2026_registry_has_exact_two_direct_mappings() -> None:
    root = Path(__file__).parents[1]
    path = root / "src/axiom_encode/oracles/policyengine/mappings/us.yaml"
    document = yaml.safe_load(path.read_text())
    mappings = _module_mappings(document)

    assert set(mappings) == set(DIRECT_VARIABLES)
    for output_name, variable in DIRECT_VARIABLES.items():
        mapping = mappings[output_name]
        assert mapping["mapping_type"] == "direct_variable"
        assert mapping["policyengine_variable"] == variable
        assert (
            mapping["program"],
            mapping["entity"],
            mapping["period"],
            mapping["unit"],
            mapping["comparison"],
        ) == ("tax", "tax_unit", "year", "USD", "money")
        assert "candidate_priority" not in mapping
        assert (
            "reviewed nonnegative completed-return boundary" in (mapping["rationale"])
        )

    liability = mappings["pa_pit_pilot_income_tax_liability"]
    for bounded_scope in (
        "3.07 percent",
        "before tax forgiveness",
        "credits",
        "payments",
        "final annual liability",
    ):
        assert bounded_scope in liability["rationale"]
    assert f"input.{INPUT_NAME}" not in mappings


def test_packaged_pa_2026_runtime_pin_version_and_precedence_are_exact() -> None:
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
        "36c34549d21576398741ca559e0d923b09126364b22cb1f7bdc5c3da025c6111"
    )
    assert hashlib.sha256(_shared_material(bundled_text).encode()).hexdigest() == (
        "a81fd41cefe8ae15c991d29c4676bce3aa970e8d850c3d1cefd90ef590f3ac97"
    )

    registry = load_policyengine_registry()
    for output_name, variable in DIRECT_VARIABLES.items():
        mapping = registry.mapping_for_legal_id(
            f"{MODULE}#{output_name}",
            country="us",
        )
        assert mapping is not None
        assert mapping.match_type == "exact"
        assert mapping.mapping_type == "direct_variable"
        assert mapping.policyengine_variable == variable
        assert mapping.entity == "tax_unit"
        assert mapping.period == "year"
        assert mapping.unit == "USD"
        assert mapping.comparison == "money"

    fallback = registry.mapping_for_legal_id(
        f"{MODULE}#future_unmapped_output",
        country="us",
    )
    assert fallback is not None
    assert fallback.legal_id == "us-pa:"
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


def test_policyengine_coverage_classifies_only_bounded_pa_2026_outputs(
    tmp_path: Path,
) -> None:
    rulespec_root = tmp_path / "rulespec-us"
    _write_synthetic_module(rulespec_root)

    report = build_policyengine_coverage_report(rulespec_root, program="tax")

    assert report["total_outputs"] == 2
    assert report["status_counts"] == {"comparable": 2}
    items = {item["rule_name"]: item for item in report["items"]}
    assert set(items) == set(DIRECT_VARIABLES)
    assert {
        name: item["policyengine_variable"] for name, item in items.items()
    } == DIRECT_VARIABLES
