"""A recorded evaluation candidate cannot change unnoticed on its way to core.

Fixtures are synthetic transport data, not policy calculations or legal evidence.
"""

import copy
from dataclasses import replace

import pytest

from axiom_encode import core_export
from tests.core_eval_fixture import candidate_result

ROOT = "zz:policies/synthetic"
DEPENDENCY = "zz:policies/explicit"


@pytest.fixture
def result(tmp_path):
    candidate = tmp_path / "candidate.yaml"
    candidate.write_bytes("# SYNTHETIC café 雪\r\nnot_yaml: [\r\n".encode())
    return candidate_result(candidate)


@pytest.mark.parametrize("success", [True, False])
def test_selected_result_preserves_bytes_metadata_and_original_result(
    result, tmp_path, success
):
    result = replace(
        result,
        success=success,
        error=None if success else "synthetic validation failure",
        admission={"synthetic_claim": "must not transfer"},
        source_attestation={"synthetic_claim": "must not transfer"},
    )
    before = copy.deepcopy(result.to_dict())
    dependency = tmp_path / "dependency.yaml"
    dependency.write_bytes(b"# SYNTHETIC explicit dependency\r\n")
    expected = core_export.build_spec(
        ROOT,
        result.output_file,
        [(DEPENDENCY, dependency)],
        expected_candidate_sha256=result.generated_output_sha256,
    )

    exported = core_export.build_spec_from_eval_result(
        result, root=ROOT, modules=[(DEPENDENCY, dependency)]
    )

    assert exported == expected
    assert exported[0]["modules"][ROOT].encode() == (
        "# SYNTHETIC café 雪\r\nnot_yaml: [\r\n".encode()
    )
    assert set(exported[0]) == {"format", "root", "modules"}
    assert exported[1]["assurance"] == "unvalidated_candidate"
    assert result.to_dict() == before


def test_changed_successful_candidate_fails_before_reading_dependencies(
    result, tmp_path
):
    from pathlib import Path

    Path(result.output_file).write_bytes(b"# SYNTHETIC changed after evaluation\n")
    with pytest.raises(core_export.CoreExportError) as error:
        core_export.build_spec_from_eval_result(
            result, root=ROOT, modules=[(DEPENDENCY, tmp_path / "absent.yaml")]
        )
    assert error.value.code == "candidate_digest_mismatch"


@pytest.mark.parametrize("digest", [None, "", 23, "a" * 63, "A" * 64, "g" * 64])
def test_missing_or_malformed_recorded_digest_cannot_disable_pin(result, digest):
    with pytest.raises(core_export.CoreExportError) as error:
        core_export.build_spec_from_eval_result(
            replace(result, generated_output_sha256=digest), root=ROOT
        )
    assert error.value.code == "invalid_digest"


@pytest.mark.parametrize("output_file", [None, "", 23])
def test_missing_candidate_binding_rejects_failed_result(result, output_file):
    with pytest.raises(core_export.CoreExportError) as error:
        core_export.build_spec_from_eval_result(
            replace(result, success=False, timed_out=True, output_file=output_file),
            root=ROOT,
        )
    assert error.value.code == "missing_candidate"


def test_deleted_candidate_does_not_fall_back_to_other_completed_result(
    result, tmp_path
):
    from pathlib import Path

    other_file = tmp_path / "other.yaml"
    other_file.write_bytes(b"# SYNTHETIC other result\n")
    other = candidate_result(other_file)
    Path(result.output_file).unlink()
    with pytest.raises(core_export.CoreExportError) as error:
        core_export.build_spec_from_eval_result(result, root=ROOT)
    assert error.value.code == "io_error"
    spec, metadata = core_export.build_spec_from_eval_result(other, root=ROOT)
    assert spec["modules"][ROOT].encode() == other_file.read_bytes()
    assert metadata["candidate_sha256"] == other.generated_output_sha256


@pytest.mark.parametrize(
    "kind,code",
    [
        ("duplicate", "duplicate_module"),
        ("symlink", "io_error"),
        ("directory", "io_error"),
        ("utf8", "invalid_utf8"),
        ("oversize", "input_too_large"),
    ],
)
def test_explicit_dependencies_keep_exporter_rejections(result, tmp_path, kind, code):
    dependency = tmp_path / "dependency.yaml"
    target = DEPENDENCY
    if kind == "symlink":
        dependency.symlink_to(result.output_file)
    elif kind == "directory":
        dependency.mkdir()
    elif kind == "utf8":
        dependency.write_bytes(b"\xff")
    elif kind == "oversize":
        with dependency.open("wb") as output:
            output.truncate(core_export.MAX_INPUT_BYTES + 1)
    else:
        target = ROOT
        dependency.write_bytes(b"# SYNTHETIC duplicate\n")
    with pytest.raises(core_export.CoreExportError) as error:
        core_export.build_spec_from_eval_result(
            result, root=ROOT, modules=[(target, dependency)]
        )
    assert error.value.code == code
