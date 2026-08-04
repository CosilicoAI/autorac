"""Integration and adversarial tests for the compiled signing supervisor."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tarfile
import threading
from base64 import b64decode, b64encode
from contextlib import contextmanager
from pathlib import Path

import _cffi_backend
import cryptography
import pytest
import yaml
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from axiom_encode import __version__
from axiom_encode.cli import (
    APPLIED_ENCODING_MANIFEST_SCHEMA,
    APPLIED_ENCODING_MODEL_TOOL,
    _sign_applied_encoding_manifest,
)
from axiom_encode.harness.dependency_stubs import validate_explicit_context_file
from axiom_encode.harness.evals import resolve_corpus_source_unit
from tests.release_object_fixtures import bind_test_corpus_release
from tests.signing_broker_fixtures import SigningBrokerFixture

ROOT = Path(__file__).parents[1]
SUPERVISOR_PACKAGE = "./cmd/axiom-encode-signing-supervisor"
PRIVATE_ENV_NAMES = (
    "AXIOM_ENCODE_APPLY_SIGNING_KEY",
    "AXIOM_ENCODE_APPLY_SIGNING_PRIVATE_KEY",
    "AXIOM_ENCODE_EVAL_SIGNING_PRIVATE_KEY",
)
PUBLIC_ENV_NAMES = (
    "AXIOM_ENCODE_APPLY_SIGNING_PUBLIC_KEY",
    "AXIOM_ENCODE_EVAL_SIGNING_PUBLIC_KEY",
    "AXIOM_CORPUS_RELEASE_PUBLIC_KEY",
)
SIGNATURE_DOMAIN = b"axiom-encode/external-signer-sign/v2\0"


def _keypair(seed: bytes) -> tuple[str, Ed25519PrivateKey]:
    private_key = Ed25519PrivateKey.from_private_bytes(seed)
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return b64encode(public_key).decode("ascii"), private_key


@pytest.fixture(scope="session")
def signing_supervisor(tmp_path_factory: pytest.TempPathFactory) -> Path:
    go = shutil.which("go")
    if go is None:
        pytest.skip("Go is required to build the signing supervisor")
    build_dir = tmp_path_factory.mktemp("signing-supervisor-build").resolve()
    binary = build_dir / "axiom-encode-signing-supervisor-test-fixture"
    subprocess.run(
        [
            go,
            "build",
            "-trimpath",
            "-buildvcs=false",
            "-ldflags=-buildid=",
            "-tags=signing_supervisor_test_fixture",
            "-o",
            str(binary),
            SUPERVISOR_PACKAGE,
        ],
        cwd=ROOT,
        env={**os.environ, "CGO_ENABLED": "0"},
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return binary


@pytest.fixture(scope="session")
def trusted_python_runtime(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Path]:
    source_interpreter = Path(sys.executable).resolve()
    source_runtime = Path(sys.base_prefix).resolve()
    runtime_root = tmp_path_factory.mktemp("trusted-python-runtime").resolve()
    shutil.copytree(source_runtime, runtime_root, dirs_exist_ok=True, symlinks=False)
    for forbidden in (
        *runtime_root.rglob("*.pth"),
        *runtime_root.rglob("*.egg-link"),
        *runtime_root.rglob("sitecustomize.py"),
        *runtime_root.rglob("usercustomize.py"),
        *runtime_root.rglob("pyvenv.cfg"),
        *runtime_root.rglob("__editable__*"),
    ):
        if forbidden.is_file():
            forbidden.unlink()
    interpreter = runtime_root / source_interpreter.relative_to(source_runtime)
    site_packages = (
        runtime_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    site_packages.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        Path(cryptography.__file__).resolve().parent,
        site_packages / "cryptography",
        dirs_exist_ok=True,
        symlinks=False,
    )
    shutil.copy2(Path(_cffi_backend.__file__).resolve(), site_packages)
    package_root = site_packages / "axiom_encode"
    package_root.mkdir()
    (package_root / "__init__.py").write_text('"""Trusted fixture package."""\n')
    shutil.copy2(ROOT / "src" / "axiom_encode" / "signing_broker.py", package_root)
    shutil.copy2(
        ROOT / "src" / "axiom_encode" / "_trusted_signing_bootstrap.py",
        package_root,
    )
    (package_root / "entrypoint.py").write_text(
        """from __future__ import annotations
import json
import os
import subprocess
import sys
from base64 import b64encode

def main():
    from axiom_encode.signing_broker import (
        SigningBrokerError,
        get_signing_broker,
        scrub_private_signing_environment,
    )
    broker = get_signing_broker()
    descriptor = broker._connection.fileno()
    fork_read, fork_write = os.pipe()
    fork_pid = os.fork()
    if fork_pid == 0:
        os.close(fork_read)
        state = {}
        try:
            get_signing_broker()
        except SigningBrokerError:
            state[\"broker\"] = \"closed\"
        else:
            state[\"broker\"] = \"open\"
        try:
            os.fstat(descriptor)
        except OSError:
            state[\"descriptor\"] = \"closed\"
        else:
            state[\"descriptor\"] = \"open\"
        os.write(fork_write, json.dumps(state).encode())
        os._exit(0)
    os.close(fork_write)
    fork_state = json.loads(os.read(fork_read, 4096))
    os.close(fork_read)
    os.waitpid(fork_pid, 0)
    child_code = \"import json,os,sys; fd=int(sys.argv[1]); state='open';\\ntry: os.fstat(fd)\\nexcept OSError: state='closed'\\nprint(json.dumps({'environment':dict(os.environ),'descriptor':state}))\"
    child = subprocess.run(
        [sys.executable, \"-I\", \"-S\", \"-c\", child_code, str(descriptor)],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_private_signing_environment(),
    )
    if os.environ.get("CODEX_HOME"):
        from pathlib import Path
        codex_home = Path(os.environ["CODEX_HOME"])
        codex_auth = codex_home / "auth.json"
        codex_auth_before_refresh = codex_auth.read_text()
        codex_auth.write_text(
            '{"token":"new"}\\n'
        )
    result = {
        \"isolated\": sys.flags.isolated,
        \"no_site\": sys.flags.no_site,
        \"package_origin\": __import__(\"axiom_encode\").__file__,
        \"sys_path\": sys.path,
        \"environment\": dict(os.environ),
        \"child\": json.loads(child.stdout),
        \"fork\": fork_state,
        \"capabilities\": sorted(broker.capabilities),
        \"roots\": {
            \"apply\": b64encode(broker.apply_public_key_raw).decode(\"ascii\"),
            \"eval\": b64encode(broker.eval_public_key_raw).decode(\"ascii\"),
            \"corpus_release\": b64encode(
                broker.corpus_release_public_key_raw
            ).decode(\"ascii\"),
            \"corpus_release_keys\": [
                b64encode(public_key).decode(\"ascii\")
                for public_key in broker.corpus_release_public_keys_raw
            ],
        },
    }
    if os.environ.get("CODEX_HOME"):
        metadata = codex_home.stat()
        result["codex_home_mode"] = metadata.st_mode & 0o777
        result["codex_home_uid"] = metadata.st_uid
        result["codex_auth_read_path"] = str(codex_auth)
        result["codex_auth_before_refresh"] = codex_auth_before_refresh
    if \"apply_ed25519\" in broker.capabilities:
        result[\"apply_signature\"] = b64encode(
            broker.apply_ed25519_sign(b\"compiled-apply-boundary\")
        ).decode(\"ascii\")
    if \"eval_ed25519\" in broker.capabilities:
        result[\"eval_signature\"] = b64encode(
            broker.eval_ed25519_sign(b\"compiled-eval-boundary\")
        ).decode(\"ascii\")
    print(json.dumps(result, sort_keys=True))
    broker.close()
    return 0
"""
    )
    return interpreter, runtime_root, package_root


@pytest.fixture(scope="session")
def trusted_real_cli_runtime(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Path]:
    """Build a hermetic runtime containing the real CLI and its dependencies."""

    source_interpreter = Path(sys.executable).resolve()
    source_runtime = Path(sys.base_prefix).resolve()
    runtime_root = tmp_path_factory.mktemp("trusted-real-cli-runtime").resolve()
    shutil.copytree(source_runtime, runtime_root, dirs_exist_ok=True, symlinks=False)
    interpreter = runtime_root / source_interpreter.relative_to(source_runtime)
    site_packages = (
        runtime_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    source_site_packages = Path(pytest.__file__).resolve().parents[1]
    shutil.copytree(
        source_site_packages,
        site_packages,
        dirs_exist_ok=True,
        symlinks=False,
        ignore=shutil.ignore_patterns(
            "axiom_encode",
            "*.pth",
            "*.egg-link",
            "sitecustomize.py",
            "usercustomize.py",
            "pyvenv.cfg",
            "__editable__*",
            "__pycache__",
            "*.pyc",
        ),
    )
    for forbidden in (
        *runtime_root.rglob("*.pth"),
        *runtime_root.rglob("*.egg-link"),
        *runtime_root.rglob("sitecustomize.py"),
        *runtime_root.rglob("usercustomize.py"),
        *runtime_root.rglob("pyvenv.cfg"),
        *runtime_root.rglob("__editable__*"),
    ):
        if forbidden.is_file():
            forbidden.unlink()

    package_root = runtime_root / "src" / "axiom_encode"
    shutil.copytree(
        ROOT / "src" / "axiom_encode",
        package_root,
        symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(ROOT / "pyproject.toml", runtime_root)
    shutil.copy2(ROOT / "uv.lock", runtime_root)
    (runtime_root / ".gitignore").write_text(
        "*\n"
        "!/.gitignore\n"
        "!/pyproject.toml\n"
        "!/uv.lock\n"
        "!/src/\n"
        "!/src/axiom_encode/\n"
        "!/src/axiom_encode/**\n"
        "/src/axiom_encode/**/__pycache__/\n"
        "/src/axiom_encode/**/*.pyc\n"
    )

    real_git = shutil.which("git")
    if real_git is None:
        pytest.skip("Git is required for guarded encoder identity verification")
    git_wrapper = interpreter.parent / "git"
    git_wrapper.write_text(
        f"#!{interpreter}\n"
        "import os\n"
        "import sys\n"
        f"git = {str(Path(real_git).resolve())!r}\n"
        "os.execv(git, [git, *sys.argv[1:]])\n"
    )
    git_wrapper.chmod(0o700)

    git_environment = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": str(runtime_root),
        "PATH": os.environ.get("PATH", ""),
    }
    subprocess.run(
        [real_git, "init", "--quiet", str(runtime_root)],
        check=True,
        env=git_environment,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            real_git,
            "-C",
            str(runtime_root),
            "add",
            ".gitignore",
            "pyproject.toml",
            "uv.lock",
            "src/axiom_encode",
        ],
        check=True,
        env=git_environment,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            real_git,
            "-c",
            "user.name=Axiom test fixture",
            "-c",
            "user.email=fixture@axiom-foundation.org",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(runtime_root),
            "commit",
            "--quiet",
            "--no-gpg-sign",
            "-m",
            "Build trusted real CLI fixture",
        ],
        check=True,
        env=git_environment,
        capture_output=True,
        text=True,
    )
    head = subprocess.run(
        [real_git, "-C", str(runtime_root), "rev-parse", "HEAD"],
        check=True,
        env=git_environment,
        capture_output=True,
        text=True,
    ).stdout.strip()
    # Model the root-provisioned runtime end to end: the supervisor sets
    # AXIOM_ENCODE_TRUSTED_RUNTIME=1, so _apply_encoder_execution_identity takes
    # the attestation branch and byte-binds the running package to
    # package_tree_sha256. Record the digest of the runtime's own package and a
    # commit that agrees with the live checkout (the agreement path).
    from axiom_encode.harness.evals import _deterministic_tree_identity

    package_tree_sha256 = _deterministic_tree_identity(
        package_root, excluded_directory_names=frozenset({"__pycache__"})
    )["tree_sha256"]
    (runtime_root / "runtime-attestation.json").write_text(
        json.dumps(
            {
                "schema": "axiom-encode/trusted-runtime-attestation/v1",
                "provisioned_at": "2026-07-18T00:00:00+00:00",
                "axiom_encode": {
                    "origin_repository": "github.com/TheAxiomFoundation/axiom-encode",
                    "commit": head,
                    "version": __version__,
                    "package_tree_sha256": package_tree_sha256,
                },
            }
        )
    )
    return interpreter, runtime_root, package_root


def _runtime_arguments(runtime: tuple[Path, Path, Path]) -> list[str]:
    _interpreter, runtime_root, package_root = runtime
    import_roots = [package_root.parent]
    site_packages = (
        runtime_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if site_packages.is_dir() and site_packages not in import_roots:
        import_roots.append(site_packages)
    arguments = [
        "--trusted-python-runtime-root",
        str(runtime_root),
    ]
    for import_root in import_roots:
        arguments.extend(("--trusted-python-import-root", str(import_root)))
    arguments.extend(("--trusted-python-package-root", str(package_root)))
    return arguments


def _launcher(
    tmp_path: Path, runtime: tuple[Path, Path, Path], body: str | None = None
) -> Path:
    interpreter, _runtime_root, _package_root = runtime
    launcher = tmp_path.resolve() / "axiom-encode"
    launcher.write_text(
        f"#!{interpreter} -I\n"
        + (body if body is not None else "raise SystemExit('launcher executed')\n")
    )
    launcher.chmod(0o700)
    return launcher


def _trust_config(
    tmp_path: Path,
    apply_public: str,
    eval_public: str,
    corpus_release_public: str | None = None,
    corpus_release_public_keys: tuple[str, ...] | None = None,
) -> Path:
    if corpus_release_public is None:
        corpus_release_public, _private_key = _keypair(b"\x17" * 32)
    path = tmp_path.resolve() / "signing-trust-roots.json"
    payload = {
        "schema": "axiom-encode/signing-trust-roots/v2",
        "apply_ed25519_public_key": apply_public,
        "eval_ed25519_public_key": eval_public,
        "corpus_release_ed25519_public_key": corpus_release_public,
    }
    if corpus_release_public_keys is not None:
        payload["schema"] = "axiom-encode/signing-trust-roots/v3"
        payload.pop("corpus_release_ed25519_public_key")
        payload["corpus_release_ed25519_public_keys"] = corpus_release_public_keys
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    path.chmod(0o600)
    return path


def _write_signed_corpus_release(
    corpus_root: Path,
    *,
    release_name: str,
    citation_path: str,
    version: str,
    body: str,
):
    jurisdiction, document_class, *_rest = citation_path.split("/")
    provision = (
        corpus_root
        / "data"
        / "corpus"
        / "provisions"
        / jurisdiction
        / document_class
        / f"{version}.jsonl"
    )
    provision.parent.mkdir(parents=True, exist_ok=True)
    provision.write_text(
        json.dumps(
            {
                "id": f"fixture:{citation_path}",
                "citation_path": citation_path,
                "body": body,
                "jurisdiction": jurisdiction,
                "document_class": document_class,
                "version": version,
                "source_path": "sources/supervisor-fixture.txt",
                "source_as_of": "2026-07-11",
                "expression_date": "2026-07-11",
            }
        )
        + "\n"
    )
    return bind_test_corpus_release(
        corpus_root,
        release_name,
        [(jurisdiction, document_class, version)],
    )


def _write_rulespec_toolchain(rulespec_root: Path, release) -> str:
    waiver = rulespec_root / "known-validation-gaps.yaml"
    waiver.parent.mkdir(parents=True, exist_ok=True)
    waiver.write_text("validate_failures: {}\n")
    waiver_sha256 = hashlib.sha256(waiver.read_bytes()).hexdigest()
    toolchain = rulespec_root / ".axiom" / "toolchain.toml"
    toolchain.parent.mkdir()
    toolchain.write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{release.name}"\n'
        f'axiom_corpus_release_content_sha256 = "{release.content_sha256}"\n'
        f'validation_waiver_set_sha256 = "{waiver_sha256}"\n'
    )
    return waiver_sha256


def _runtime_commit(runtime_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(runtime_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_signed_guard_fixture(
    tmp_path: Path,
    runtime_root: Path,
    apply_public: str,
) -> tuple[Path, Path]:
    rulespec_root = tmp_path / "rulespec-us"
    corpus_root = tmp_path / "axiom-corpus"
    citation_path = "us/statute/1"
    body = "The signed supervisor source is authoritative.\n"
    release = _write_signed_corpus_release(
        corpus_root,
        release_name="supervisor-guard-release",
        citation_path=citation_path,
        version="2026-supervisor-guard",
        body=body,
    )
    waiver_sha256 = _write_rulespec_toolchain(rulespec_root, release)
    source_unit = resolve_corpus_source_unit(citation_path, release)
    source_attestation = dict(source_unit.source_attestation)
    source_attestation["generation_input_sha256"] = hashlib.sha256(
        body.encode()
    ).hexdigest()
    source_attestation["rulespec_root"] = "rulespec-us/us"

    rule = rulespec_root / "us" / "statutes" / "1.yaml"
    rule.parent.mkdir(parents=True)
    rule.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        f"    corpus_citation_path: {citation_path}\n"
        f"    source_sha256: {source_attestation['source_sha256']}\n"
        "rules: []\n"
    )
    commit = _runtime_commit(runtime_root)
    payload = {
        "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
        "generated_at": "2026-07-11T00:00:00+00:00",
        "tool": APPLIED_ENCODING_MODEL_TOOL,
        "axiom_encode_version": __version__,
        "axiom_encode_git": {
            "root": str(runtime_root),
            "commit": commit,
            "dirty_tracked": False,
            "version": __version__,
            "version_commit": commit,
            "identity_source": "git",
        },
        "generation_prompt_sha256": None,
        "run_id": None,
        "citation": "1 USC 1",
        "runner": "codex:fixture",
        "backend": "codex",
        "model": "fixture",
        "validation_waiver_set_sha256": waiver_sha256,
        "generated_output_root": str(tmp_path / "generated"),
        "generated_output_file": None,
        "generated_output_sha256": None,
        "trace_file": None,
        "trace_sha256": None,
        "context_manifest_file": None,
        "context_manifest_sha256": None,
        "applied_files": [
            {
                "path": "us/statutes/1.yaml",
                "sha256": hashlib.sha256(rule.read_bytes()).hexdigest(),
            }
        ],
        "source_attestation": source_attestation,
        "validation_execution": {
            "schema": "axiom-encode/apply-validation-execution/v1",
            "axiom_encode": {
                "repository": "github.com/TheAxiomFoundation/axiom-encode",
                "commit": commit,
                "version": __version__,
                "identity_source": "git",
            },
            "axiom_rules_engine": {
                "repository": ("github.com/TheAxiomFoundation/axiom-rules-engine"),
                "commit": "e" * 40,
            },
            "policy_pre_apply": {
                "rulespec_root": "rulespec-us/us",
                "pre_apply_content_sha256": "c" * 64,
                "pre_apply_file_count": 0,
                "toolchain_contract_sha256": "d" * 64,
                "validation_waiver_set_sha256": waiver_sha256,
            },
            "rulespec_dependencies": [],
        },
    }
    apply_private = b64encode(b"\xab" * 32).decode("ascii")
    _sign_applied_encoding_manifest(
        payload,
        SigningBrokerFixture(
            apply_private_key=apply_private,
            apply_public_key=apply_public,
        ),
    )
    manifest = (
        rulespec_root / ".axiom" / "encoding-manifests" / "us" / "statutes" / "1.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return rulespec_root, corpus_root


def _current_engine_root() -> Path | None:
    configured = os.environ.get("AXIOM_RULES_ENGINE_ROOT")
    candidates = [
        *(Path(configured).expanduser() for _ in (0,) if configured),
        ROOT.parent / "axiom-rules-engine-canonical-loader-hard-cut",
        ROOT.parents[1] / "axiom-rules-engine",
    ]
    for root in candidates:
        if not root.is_dir():
            continue
        for binary in (
            root / "target" / "debug" / "axiom-rules-engine",
            root / "target" / "release" / "axiom-rules-engine",
            root / "axiom-rules-engine",
        ):
            if not binary.is_file() or not os.access(binary, os.X_OK):
                continue
            probe = subprocess.run(
                [str(binary), "compile", "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if probe.returncode == 0 and "--rulespec-root" in (
                probe.stdout + probe.stderr
            ):
                return root.resolve()
    return None


def _write_current_engine_fixture(tmp_path: Path) -> tuple[Path, Path]:
    rulespec_root = tmp_path / "rulespec-us"
    corpus_root = tmp_path / "axiom-corpus"
    citation_path = "us/guidance/example/sua"
    release = _write_signed_corpus_release(
        corpus_root,
        release_name="supervisor-current-engine-release",
        citation_path=citation_path,
        version="2026-supervisor-engine",
        body="The standard utility allowance for a household is $451.\n",
    )
    _write_rulespec_toolchain(rulespec_root, release)
    rules = rulespec_root / "us" / "policies" / "example" / "rules.yaml"
    rules.parent.mkdir(parents=True)
    rules.write_text(
        """format: rulespec/v1
module:
  summary: The standard utility allowance is $451.
  source_verification:
    corpus_citation_path: us/guidance/example/sua
rules:
  - name: standard_utility_allowance_value
    kind: parameter
    source: us/guidance/example/sua
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '451'
  - name: standard_utility_allowance
    kind: derived
    source: us/guidance/example/sua
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: standard_utility_allowance_value
"""
    )
    rules.with_name("rules.test.yaml").write_text(
        """- name: signed corpus and current engine
  period: 2026-01
  input: {}
  output:
    us:policies/example/rules#standard_utility_allowance: 451
"""
    )
    return rules, corpus_root


def _receive_exact(connection: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    while length:
        chunk = connection.recv(length)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        length -= len(chunk)
    return b"".join(chunks)


def _serve_external_signer(
    connection: socket.socket,
    private_key: Ed25519PrivateKey,
    behavior: str,
) -> None:
    try:
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        while True:
            try:
                header = _receive_exact(connection, 4)
            except EOFError:
                return
            request = json.loads(
                _receive_exact(connection, struct.unpack(">I", header)[0]).decode()
            )
            request_id = request["id"]
            scope = request.get("scope")
            if request.get("version") != 2:
                response = {
                    "version": 2,
                    "id": request_id,
                    "ok": False,
                    "error": "unsupported protocol",
                }
            elif request.get("op") == "challenge":
                nonce = b64decode(request["challenge"], validate=True)
                message = (
                    b"axiom-encode/external-signer-challenge/v2\0"
                    + scope.encode("ascii")
                    + b"\0"
                    + nonce
                )
                signature = private_key.sign(message)
                if behavior == "wrong_challenge_signature":
                    signature = private_key.sign(b"wrong")
                response = {
                    "version": 2,
                    "id": request_id,
                    "ok": True,
                    "public_key": b64encode(public_key).decode("ascii"),
                    "signature": b64encode(signature).decode("ascii"),
                }
                if behavior == "extra_challenge_field":
                    response["legacy"] = True
            elif request.get("op") == "sign":
                payload = b64decode(request["payload"], validate=True)
                message = SIGNATURE_DOMAIN + scope.encode("ascii") + b"\0" + payload
                signature = private_key.sign(message)
                if behavior == "wrong_sign_signature":
                    signature = private_key.sign(payload)
                response = {
                    "version": 2,
                    "id": request_id,
                    "ok": True,
                    "signature": b64encode(signature).decode("ascii"),
                }
                if behavior == "extra_sign_field":
                    response["legacy"] = True
            else:
                response = {
                    "version": 2,
                    "id": request_id,
                    "ok": False,
                    "error": "unsupported request",
                }
            if behavior == "legacy_v1_response":
                response["version"] = 1
            raw = json.dumps(response, separators=(",", ":"), sort_keys=True).encode()
            connection.sendall(struct.pack(">I", len(raw)) + raw)
    finally:
        connection.close()


@contextmanager
def _signers(*keys: Ed25519PrivateKey, behavior: str = "valid"):
    supervisor_connections: list[socket.socket] = []
    threads: list[threading.Thread] = []
    for key in keys:
        signer_connection, supervisor_connection = socket.socketpair()
        thread = threading.Thread(
            target=_serve_external_signer,
            args=(signer_connection, key, behavior),
            daemon=True,
        )
        thread.start()
        supervisor_connections.append(supervisor_connection)
        threads.append(thread)
    try:
        yield [connection.fileno() for connection in supervisor_connections]
    finally:
        for connection in supervisor_connections:
            connection.close()
        for thread in threads:
            thread.join(timeout=5)


def _invoke(
    supervisor: Path,
    runtime: tuple[Path, Path, Path],
    launcher: Path,
    trust_config: Path,
    descriptors: list[int],
    *,
    environment: dict[str, str] | None = None,
    command_args: tuple[str, ...] = (),
    supervisor_args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    signer_arguments: list[str] = []
    if descriptors:
        signer_arguments.extend(("--apply-signer-fd", str(descriptors[0])))
    if len(descriptors) > 1:
        signer_arguments.extend(("--eval-signer-fd", str(descriptors[1])))
    return subprocess.run(
        [
            str(supervisor),
            *signer_arguments,
            "--trusted-signing-roots",
            str(trust_config),
            *supervisor_args,
            *_runtime_arguments(runtime),
            "--",
            str(launcher),
            *command_args,
        ],
        env={} if environment is None else environment,
        pass_fds=tuple(descriptors),
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_subscription_auth_is_isolated_refreshed_and_wiped(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    apply_public, _ = _keypair(b"\xab" * 32)
    eval_public, _ = _keypair(b"\xcd" * 32)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    trusted = tmp_path / "trusted"
    (trusted / "bin").mkdir(parents=True)
    legacy_scratch = trusted / "runtime-codex-homes"
    legacy_scratch.mkdir(mode=0o700)
    legacy_scratch.chmod(0o000)
    codex = trusted / "bin/codex"
    codex.write_text("#!/bin/sh\nexit 0\n")
    codex.chmod(0o755)
    digest = hashlib.sha256(codex.read_bytes()).hexdigest()
    config = trusted / "codex-cli.json"
    config.write_text(
        json.dumps(
            {
                "schema": "axiom-encode/trusted-codex-cli/v1",
                "version": "test",
                "sha256": digest,
                "path": str(codex),
            }
        )
        + "\n"
    )
    config.chmod(0o444)
    auth = tmp_path / "operator-auth.json"
    auth.write_text('{"token":"old"}\n')
    auth.chmod(0o600)
    outbox = tmp_path / "refreshed-auth.json"
    operator_home_auth = tmp_path / "operator-home/.codex/auth.json"
    operator_home_auth.parent.mkdir(parents=True)
    operator_home_auth.write_text('{"must":"not-cross"}\n')
    launcher = _launcher(tmp_path, trusted_python_runtime)
    completed = _invoke(
        signing_supervisor,
        trusted_python_runtime,
        launcher,
        trust_config,
        [],
        environment={"HOME": str(tmp_path / "operator-home")},
        supervisor_args=(
            "--codex-subscription-auth",
            str(auth),
            "--codex-auth-outbox",
            str(outbox),
            "--trusted-codex-cli-config",
            str(config),
        ),
    )
    legacy_scratch.chmod(0o700)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    codex_home = Path(result["environment"]["CODEX_HOME"])
    assert codex_home.name.startswith("axiom-codex-")
    assert not codex_home.exists()
    assert result["codex_home_mode"] == 0o700
    assert result["codex_home_uid"] == os.geteuid()
    assert Path(result["codex_auth_read_path"]) == codex_home / "auth.json"
    assert Path(result["codex_auth_read_path"]) != operator_home_auth
    assert json.loads(result["codex_auth_before_refresh"]) == {"token": "old"}
    assert result["environment"]["AXIOM_ENCODE_TRUSTED_CODEX_BIN"] == str(codex)
    assert result["environment"]["AXIOM_ENCODE_TRUSTED_CODEX_SHA256"] == digest
    assert result["environment"]["AXIOM_ENCODE_TRUSTED_CODEX_VERSION"] == "test"
    assert str(codex.parent) not in result["environment"]["PATH"].split(os.pathsep)
    assert result["environment"]["HOME"] != str(tmp_path / "operator-home")
    assert result["child"]["descriptor"] == "closed"
    assert json.loads(operator_home_auth.read_text()) == {"must": "not-cross"}
    assert json.loads(outbox.read_text()) == {"token": "new"}


def test_subscription_tampered_binary_hard_fails_before_child(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    apply_public, _ = _keypair(b"\xab" * 32)
    eval_public, _ = _keypair(b"\xcd" * 32)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    codex = tmp_path / "codex"
    codex.write_text("tampered")
    codex.chmod(0o755)
    config = tmp_path / "codex-cli.json"
    config.write_text(
        json.dumps(
            {
                "schema": "axiom-encode/trusted-codex-cli/v1",
                "version": "test",
                "sha256": "0" * 64,
                "path": str(codex),
            }
        )
    )
    config.chmod(0o444)
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    launcher = _launcher(
        tmp_path, trusted_python_runtime, body="raise RuntimeError('must not execute')"
    )
    completed = _invoke(
        signing_supervisor,
        trusted_python_runtime,
        launcher,
        trust_config,
        [],
        supervisor_args=(
            "--codex-subscription-auth",
            str(auth),
            "--codex-auth-outbox",
            str(tmp_path / "out.json"),
            "--trusted-codex-cli-config",
            str(config),
        ),
    )
    assert completed.returncode == 2
    assert "sha256 mismatch" in completed.stderr


@pytest.mark.parametrize(
    "outbox_kind",
    ["symlink", "fifo", "socket", "device", "writable-directory"],
)
def test_subscription_refuses_unsafe_auth_outbox(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
    outbox_kind: str,
) -> None:
    apply_public, _ = _keypair(b"\xab" * 32)
    eval_public, _ = _keypair(b"\xcd" * 32)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n")
    codex.chmod(0o755)
    config = tmp_path / "codex-cli.json"
    config.write_text(
        json.dumps(
            {
                "schema": "axiom-encode/trusted-codex-cli/v1",
                "version": "test",
                "sha256": hashlib.sha256(codex.read_bytes()).hexdigest(),
                "path": str(codex),
            }
        )
    )
    config.chmod(0o444)
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    target = tmp_path / "target.json"
    target.write_text('{"preserve":true}\n')
    outbox_socket = None
    if outbox_kind == "symlink":
        outbox = tmp_path / "out.json"
        outbox.symlink_to(target)
    elif outbox_kind == "fifo":
        outbox = tmp_path / "out.json"
        os.mkfifo(outbox)
    elif outbox_kind == "socket":
        socket_directory = Path.cwd() / f".outbox-{os.getpid()}"
        socket_directory.mkdir(mode=0o700)
        outbox = socket_directory / "o"
        outbox_socket = socket.socket(socket.AF_UNIX)
        try:
            outbox_socket.bind(str(outbox))
        except PermissionError:
            outbox_socket.close()
            socket_directory.rmdir()
            pytest.skip("sandbox does not permit creating Unix-domain sockets")
    elif outbox_kind == "device":
        outbox = Path("/dev/null")
    else:
        unsafe_directory = tmp_path / "unsafe-outbox"
        unsafe_directory.mkdir(mode=0o777)
        unsafe_directory.chmod(0o777)
        outbox = unsafe_directory / "out.json"
    try:
        completed = _invoke(
            signing_supervisor,
            trusted_python_runtime,
            _launcher(tmp_path, trusted_python_runtime),
            trust_config,
            [],
            supervisor_args=(
                "--codex-subscription-auth",
                str(auth),
                "--codex-auth-outbox",
                str(outbox),
                "--trusted-codex-cli-config",
                str(config),
            ),
        )
    finally:
        if outbox_socket is not None:
            outbox_socket.close()
            shutil.rmtree(socket_directory)
    assert completed.returncode == 2
    assert "outbox" in completed.stderr
    assert json.loads(target.read_text()) == {"preserve": True}


def test_subscription_refuses_ambient_codex_home_outside_scratch(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    apply_public, _ = _keypair(b"\xab" * 32)
    eval_public, _ = _keypair(b"\xcd" * 32)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    completed = _invoke(
        signing_supervisor,
        trusted_python_runtime,
        _launcher(tmp_path, trusted_python_runtime),
        trust_config,
        [],
        environment={"CODEX_HOME": str(tmp_path / "outside")},
        supervisor_args=(
            "--codex-subscription-auth",
            str(tmp_path / "auth.json"),
            "--codex-auth-outbox",
            str(tmp_path / "out.json"),
            "--trusted-codex-cli-config",
            str(tmp_path / "codex-cli.json"),
        ),
    )
    assert completed.returncode == 2
    assert "ambient CODEX_HOME is outside supervisor custody" in completed.stderr


def test_fixture_binary_is_explicitly_nonpublishable(signing_supervisor: Path) -> None:
    completed = subprocess.run(
        [str(signing_supervisor), "--build-kind"], capture_output=True, text=True
    )
    assert completed.stdout.strip() == "test-fixture-nonpublishable"


def test_every_invocation_requires_protected_three_root_config(
    signing_supervisor: Path,
) -> None:
    completed = subprocess.run(
        [str(signing_supervisor), "--", "/nonexistent/axiom-encode"],
        env={},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "--trusted-signing-roots is required" in completed.stderr


@pytest.mark.skipif(
    "AXIOM_ENCODE_PRODUCTION_SIGNING_FIXTURE" not in os.environ,
    reason="root-owned production fixture is prepared only in signing CI",
)
def test_untagged_production_binary_root_owned_end_to_end() -> None:
    fixture = Path(os.environ["AXIOM_ENCODE_PRODUCTION_SIGNING_FIXTURE"])
    supervisor = fixture / "axiom-encode-signing-supervisor"
    runtime_root = fixture / "python"
    interpreter = runtime_root / Path(sys.executable).resolve().relative_to(
        Path(sys.base_prefix).resolve()
    )
    package_root = (
        runtime_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages/axiom_encode"
    )
    completed_kind = subprocess.run(
        [str(supervisor), "--build-kind"], capture_output=True, text=True
    )
    assert completed_kind.stdout.strip() == "production"
    apply_public, apply_key = _keypair(b"\xab" * 32)
    eval_public, eval_key = _keypair(b"\xcd" * 32)
    with _signers(apply_key, eval_key) as descriptors:
        completed = _invoke(
            supervisor,
            (interpreter, runtime_root, package_root),
            fixture / "axiom-encode",
            fixture / "signing-trust-roots.json",
            descriptors,
        )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    apply_key.public_key().verify(
        b64decode(result["apply"]),
        SIGNATURE_DOMAIN + b"apply_ed25519\0production-apply",
    )
    eval_key.public_key().verify(
        b64decode(result["eval"]),
        SIGNATURE_DOMAIN + b"eval_ed25519\0production-eval",
    )
    assert apply_public != eval_public


def test_compiled_supervisor_uses_isolated_direct_runtime_and_domain_signatures(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    apply_public, apply_key = _keypair(b"\xab" * 32)
    eval_public, eval_key = _keypair(b"\xcd" * 32)
    launcher = _launcher(tmp_path, trusted_python_runtime)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    sentinels = {
        "OPENAI_API_KEY": "openai-sentinel",
        "ANTHROPIC_API_KEY": "anthropic-sentinel",
        "PATH": "/hostile/bin",
        "PYTHONPATH": "/hostile/python",
        "GIT_CONFIG_GLOBAL": "/hostile/gitconfig",
        "AWS_SECRET_ACCESS_KEY": "aws-sentinel",
        "GH_TOKEN": "gh-sentinel",
        "AXIOM_ENCODE_SUPABASE_SECRET_KEY": "supabase-sentinel",
        "OTEL_EXPORTER_OTLP_HEADERS": "otel-sentinel",
        "HTTPS_PROXY": "http://proxy.invalid",
    }
    with _signers(apply_key, eval_key) as descriptors:
        completed = _invoke(
            signing_supervisor,
            trusted_python_runtime,
            launcher,
            trust_config,
            descriptors,
            environment=sentinels,
        )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["isolated"] == 1
    assert result["no_site"] == 1
    assert Path(result["package_origin"]) == trusted_python_runtime[2] / "__init__.py"
    assert result["capabilities"] == ["apply_ed25519", "eval_ed25519"]
    for name in (
        *PRIVATE_ENV_NAMES,
        *PUBLIC_ENV_NAMES,
        "AXIOM_ENCODE_SIGNING_BROKER_FD",
        "AXIOM_ENCODE_SIGNING_BROKER_PID",
        "AXIOM_ENCODE_SIGNING_BROKER_ACTIVE",
    ):
        assert name not in result["environment"]
    assert all(
        Path(path).is_relative_to(trusted_python_runtime[1])
        for path in result["sys_path"]
    )
    parent_only = {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AXIOM_ENCODE_SUPABASE_SECRET_KEY",
        "OTEL_EXPORTER_OTLP_HEADERS",
    }
    for name, value in sentinels.items():
        if name in parent_only:
            assert result["environment"][name] == value
        elif name in {"PATH", "GIT_CONFIG_GLOBAL"}:
            assert result["environment"][name] != value
        else:
            assert name not in result["environment"]
        if name not in parent_only:
            assert value not in completed.stdout
    assert {"LANG", "PYTHONDONTWRITEBYTECODE"} <= set(result["child"]["environment"])
    for name, value in sentinels.items():
        assert result["child"]["environment"].get(name) != value
    assert result["child"]["environment"]["PATH"] == result["environment"]["PATH"]
    assert result["child"]["environment"]["HOME"] == result["environment"]["HOME"]
    assert result["child"]["environment"]["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert result["child"]["descriptor"] == "closed"
    assert result["fork"] == {"broker": "closed", "descriptor": "closed"}
    apply_key.public_key().verify(
        b64decode(result["apply_signature"]),
        SIGNATURE_DOMAIN + b"apply_ed25519\0compiled-apply-boundary",
    )
    eval_key.public_key().verify(
        b64decode(result["eval_signature"]),
        SIGNATURE_DOMAIN + b"eval_ed25519\0compiled-eval-boundary",
    )
    with pytest.raises(Exception):
        eval_key.public_key().verify(
            b64decode(result["apply_signature"]),
            SIGNATURE_DOMAIN + b"eval_ed25519\0compiled-apply-boundary",
        )


def test_python_startup_rejects_corpus_public_root_environment() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "src/axiom_encode/_trusted_signing_bootstrap.py")],
        env={"AXIOM_CORPUS_RELEASE_PUBLIC_KEY": "counterfeit"},
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "authenticated broker" in completed.stderr
    assert "AXIOM_CORPUS_RELEASE_PUBLIC_KEY" in completed.stderr


def test_verification_only_invocation_exposes_roots_without_signing_capability(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    apply_public, _apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    corpus_release_public, _corpus_release_key = _keypair(b"\x17" * 32)
    completed = _invoke(
        signing_supervisor,
        trusted_python_runtime,
        _launcher(tmp_path, trusted_python_runtime),
        _trust_config(
            tmp_path,
            apply_public,
            eval_public,
            corpus_release_public,
        ),
        [],
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["capabilities"] == []
    assert result["roots"] == {
        "apply": apply_public,
        "eval": eval_public,
        "corpus_release": corpus_release_public,
        "corpus_release_keys": [corpus_release_public],
    }


def test_v3_trust_config_exposes_ordered_corpus_release_keyring(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    apply_public, _apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    current_public, _current_key = _keypair(b"\x18" * 32)
    retired_public, _retired_key = _keypair(b"\x17" * 32)
    completed = _invoke(
        signing_supervisor,
        trusted_python_runtime,
        _launcher(tmp_path, trusted_python_runtime),
        _trust_config(
            tmp_path,
            apply_public,
            eval_public,
            corpus_release_public_keys=(current_public, retired_public),
        ),
        [],
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["roots"] == {
        "apply": apply_public,
        "eval": eval_public,
        "corpus_release": current_public,
        "corpus_release_keys": [current_public, retired_public],
    }


@pytest.mark.parametrize("mutation", ["empty", "malformed", "wrong_length", "conflict"])
def test_v3_trust_config_rejects_invalid_corpus_release_keyrings(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
    mutation: str,
) -> None:
    apply_public, _apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    current_public, _current_key = _keypair(b"\x18" * 32)
    other_public, _other_key = _keypair(b"\x19" * 32)
    trust_config = _trust_config(
        tmp_path,
        apply_public,
        eval_public,
        corpus_release_public_keys=(current_public,),
    )
    payload = json.loads(trust_config.read_text())
    if mutation == "empty":
        payload["corpus_release_ed25519_public_keys"] = []
    elif mutation == "malformed":
        payload["corpus_release_ed25519_public_keys"] = ["not-base64!!"]
    elif mutation == "wrong_length":
        payload["corpus_release_ed25519_public_keys"] = [b64encode(b"short").decode()]
    else:
        payload["corpus_release_ed25519_public_key"] = other_public
    trust_config.write_text(json.dumps(payload) + "\n")

    completed = _invoke(
        signing_supervisor,
        trusted_python_runtime,
        _launcher(tmp_path, trusted_python_runtime),
        trust_config,
        [],
    )

    assert completed.returncode == 2
    assert "corpus release public key" in completed.stderr


def test_verification_only_supervisor_accepts_retired_release_key_from_v3_keyring(
    signing_supervisor: Path,
    trusted_real_cli_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    apply_public, _apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    current_corpus_release_public, _current_corpus_release_key = _keypair(b"\x18" * 32)
    retired_corpus_release_public, _retired_corpus_release_key = _keypair(b"\x17" * 32)
    rulespec_root, corpus_root = _write_signed_guard_fixture(
        tmp_path,
        trusted_real_cli_runtime[1],
        apply_public,
    )
    completed = _invoke(
        signing_supervisor,
        trusted_real_cli_runtime,
        _launcher(tmp_path, trusted_real_cli_runtime),
        _trust_config(
            tmp_path,
            apply_public,
            eval_public,
            corpus_release_public_keys=(
                current_corpus_release_public,
                retired_corpus_release_public,
            ),
        ),
        [],
        command_args=(
            "guard-generated",
            "--repo",
            str(rulespec_root),
            "--corpus-path",
            str(corpus_root),
            "--all",
            "--json",
        ),
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "repo": str(rulespec_root.resolve()),
        "passed": True,
        "issues": [],
    }


def test_supervised_validate_uses_signed_release_and_current_engine_end_to_end(
    signing_supervisor: Path,
    trusted_real_cli_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    engine_root = _current_engine_root()
    if engine_root is None:
        pytest.skip("current axiom-rules-engine binary is not available")
    rules, corpus_root = _write_current_engine_fixture(tmp_path)
    apply_public, _apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    corpus_release_public, _corpus_release_key = _keypair(b"\x17" * 32)
    completed = _invoke(
        signing_supervisor,
        trusted_real_cli_runtime,
        _launcher(tmp_path, trusted_real_cli_runtime),
        _trust_config(
            tmp_path,
            apply_public,
            eval_public,
            corpus_release_public,
        ),
        [],
        command_args=(
            "validate",
            str(rules),
            "--corpus-path",
            str(corpus_root),
            "--axiom-rules-engine-path",
            str(engine_root),
            "--skip-reviewers",
        ),
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "CI: ✓" in completed.stdout
    assert "Result: ✓ PASSED" in completed.stdout


@pytest.mark.parametrize("name", PRIVATE_ENV_NAMES)
@pytest.mark.parametrize("value", ["secret", ""])
def test_every_legacy_and_current_private_environment_name_is_fatal(
    signing_supervisor: Path,
    name: str,
    value: str,
) -> None:
    completed = subprocess.run(
        [str(signing_supervisor), "--help"],
        env={name: value},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert name in completed.stderr


@pytest.mark.parametrize("name", PUBLIC_ENV_NAMES)
@pytest.mark.parametrize("value", ["counterfeit", ""])
def test_environment_public_roots_cannot_define_trust(
    signing_supervisor: Path,
    name: str,
    value: str,
) -> None:
    completed = subprocess.run(
        [str(signing_supervisor), "--help"],
        env={name: value},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "must come from --trusted-signing-roots" in completed.stderr


@pytest.mark.parametrize("aliased", ["apply_eval", "apply_corpus", "eval_corpus"])
def test_aliased_roots_fail_even_for_verification_only_invocation(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
    aliased: str,
) -> None:
    apply_public, _apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    corpus_release_public, _corpus_release_key = _keypair(b"\x17" * 32)
    if aliased == "apply_eval":
        eval_public = apply_public
    elif aliased == "apply_corpus":
        corpus_release_public = apply_public
    else:
        corpus_release_public = eval_public
    launcher = _launcher(tmp_path, trusted_python_runtime)
    trust_config = _trust_config(
        tmp_path,
        apply_public,
        eval_public,
        corpus_release_public,
    )
    completed = _invoke(
        signing_supervisor,
        trusted_python_runtime,
        launcher,
        trust_config,
        [],
    )
    assert completed.returncode == 2
    assert "must be distinct" in completed.stderr


@pytest.mark.parametrize(
    "mutation",
    ["old_schema", "missing_eval", "missing_corpus", "extra_field"],
)
def test_trust_config_is_exact_and_has_no_legacy_shape(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
    mutation: str,
) -> None:
    apply_public, apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    launcher = _launcher(tmp_path, trusted_python_runtime)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    payload = json.loads(trust_config.read_text())
    if mutation == "old_schema":
        payload["schema"] = "axiom-encode/signing-trust-roots/v1"
    elif mutation == "missing_eval":
        payload.pop("eval_ed25519_public_key")
    elif mutation == "missing_corpus":
        payload.pop("corpus_release_ed25519_public_key")
    else:
        payload["legacy"] = True
    trust_config.write_text(json.dumps(payload) + "\n")
    with _signers(apply_key) as descriptors:
        completed = _invoke(
            signing_supervisor,
            trusted_python_runtime,
            launcher,
            trust_config,
            descriptors,
        )
    assert completed.returncode == 2
    assert "trust-root config" in completed.stderr


@pytest.mark.parametrize(
    "forbidden_name",
    ["attack.pth", "sitecustomize.py", "pyvenv.cfg", "__editable__attack.py"],
)
def test_runtime_startup_and_editable_injection_is_rejected_before_attachment(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
    forbidden_name: str,
) -> None:
    apply_public, apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    launcher = _launcher(tmp_path, trusted_python_runtime)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    forbidden = trusted_python_runtime[2].parent / forbidden_name
    forbidden.write_text("raise SystemExit('injected')\n")
    try:
        with _signers(apply_key) as descriptors:
            completed = _invoke(
                signing_supervisor,
                trusted_python_runtime,
                launcher,
                trust_config,
                descriptors,
            )
    finally:
        forbidden.unlink()
    assert completed.returncode == 2
    assert "forbidden startup or editable injection" in completed.stderr
    assert completed.stdout == ""


@pytest.mark.parametrize(
    ("behavior", "expected"),
    [
        ("wrong_challenge_signature", "challenge response is invalid"),
        ("legacy_v1_response", "initialization failed"),
        ("extra_challenge_field", "initialization failed"),
        ("wrong_sign_signature", "External apply signer failed"),
        ("extra_sign_field", "External apply signer failed"),
    ],
)
def test_invalid_external_signer_response_fails_closed(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
    behavior: str,
    expected: str,
) -> None:
    apply_public, apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    launcher = _launcher(tmp_path, trusted_python_runtime)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    with _signers(apply_key, behavior=behavior) as descriptors:
        completed = _invoke(
            signing_supervisor,
            trusted_python_runtime,
            launcher,
            trust_config,
            descriptors,
        )
    assert completed.returncode != 0
    assert expected in completed.stderr


def test_non_socket_signer_descriptor_is_rejected(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    apply_public, _apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    launcher = _launcher(tmp_path, trusted_python_runtime)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    read_fd, write_fd = os.pipe()
    try:
        completed = _invoke(
            signing_supervisor,
            trusted_python_runtime,
            launcher,
            trust_config,
            [read_fd],
        )
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert completed.returncode == 2
    assert "connected socket" in completed.stderr


# --- Production external apply-signer binary (cmd/axiom-encode-apply-signer) ---
#
# The tests above use an in-process Python signer to drive the broker. These
# exercise the real, separately compiled production external signer as the
# broker's protocol v2 socket peer, proving wire compatibility with the actual
# supervisor rather than a re-implementation.

_APPLY_SIGNER_PACKAGE = "./cmd/axiom-encode-apply-signer"
_SIGNER_REPOSITORY = "TheAxiomFoundation/rulespec-uk"
_SIGNER_WORKFLOW_REF = (
    "TheAxiomFoundation/rulespec-uk/.github/workflows/bulk-encode.yml@refs/heads/main"
)


@pytest.fixture(scope="session")
def apply_signer_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    go = shutil.which("go")
    if go is None:
        pytest.skip("Go is required to build the apply signer")
    build_dir = tmp_path_factory.mktemp("apply-signer-build").resolve()
    binary = build_dir / "axiom-encode-apply-signer"
    subprocess.run(
        [
            go,
            "build",
            "-trimpath",
            "-buildvcs=false",
            "-ldflags=-buildid=",
            "-o",
            str(binary),
            _APPLY_SIGNER_PACKAGE,
        ],
        cwd=ROOT,
        env={**os.environ, "CGO_ENABLED": "0"},
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return binary


@contextmanager
def _external_apply_signer(binary: Path, seed: bytes, audit_path: Path):
    """Run the production signer binary as the broker's protocol v2 peer.

    The signer's audit stream is redirected to a file, not a pipe: the signer
    stays alive until the socket tears down in this manager's finally, so reading
    a pipe inline would deadlock.
    """

    signer_connection, supervisor_connection = socket.socketpair()
    key_read, key_write = os.pipe()
    os.write(key_write, b64encode(seed))
    os.close(key_write)
    with open(audit_path, "w") as audit_file:
        process = subprocess.Popen(
            [
                str(binary),
                "serve",
                "--scope",
                "apply_ed25519",
                "--socket-fd",
                str(signer_connection.fileno()),
                "--key-fd",
                str(key_read),
                "--expected-github-repository",
                _SIGNER_REPOSITORY,
                "--allowed-workflow-ref",
                _SIGNER_WORKFLOW_REF,
                "--allowed-event-name",
                "workflow_dispatch",
            ],
            pass_fds=(signer_connection.fileno(), key_read),
            env={
                "GITHUB_ACTIONS": "true",
                "GITHUB_REPOSITORY": _SIGNER_REPOSITORY,
                "GITHUB_WORKFLOW_REF": _SIGNER_WORKFLOW_REF,
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "GITHUB_SHA": "0" * 40,
                "GITHUB_RUN_ID": "1",
                "PATH": os.environ.get("PATH", ""),
            },
            stdout=audit_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    signer_connection.close()
    os.close(key_read)
    try:
        yield [supervisor_connection.fileno()]
    finally:
        supervisor_connection.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_production_apply_signer_binary_signs_through_real_broker(
    signing_supervisor: Path,
    apply_signer_binary: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    seed = b"\x2a" * 32
    apply_public, apply_key = _keypair(seed)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    launcher = _launcher(tmp_path, trusted_python_runtime)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    audit_path = tmp_path / "signer-audit.log"
    with _external_apply_signer(apply_signer_binary, seed, audit_path) as descriptors:
        completed = _invoke(
            signing_supervisor,
            trusted_python_runtime,
            launcher,
            trust_config,
            descriptors,
        )
    signer_output = audit_path.read_text()
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["capabilities"] == ["apply_ed25519"]
    # The apply signature the real broker obtained from the production binary
    # verifies with the public half over the exact apply domain.
    apply_key.public_key().verify(
        b64decode(result["apply_signature"]),
        SIGNATURE_DOMAIN + b"apply_ed25519\0compiled-apply-boundary",
    )
    # It is scope-bound: the apply signature must not verify as an eval one.
    with pytest.raises(Exception):
        apply_key.public_key().verify(
            b64decode(result["apply_signature"]),
            b"axiom-encode/external-signer-sign/v2\0eval_ed25519\0compiled-apply-boundary",
        )
    # The signer's audit stream records the signing event and never the key.
    assert "event=sign" in signer_output
    assert b64encode(seed).decode() not in signer_output


def test_production_apply_signer_binary_rejects_wrong_ci_context(
    apply_signer_binary: Path,
) -> None:
    seed = b"\x2a" * 32
    signer_connection, supervisor_connection = socket.socketpair()
    key_read, key_write = os.pipe()
    os.write(key_write, b64encode(seed))
    os.close(key_write)
    try:
        completed = subprocess.run(
            [
                str(apply_signer_binary),
                "serve",
                "--scope",
                "apply_ed25519",
                "--socket-fd",
                str(signer_connection.fileno()),
                "--key-fd",
                str(key_read),
                "--expected-github-repository",
                _SIGNER_REPOSITORY,
                "--allowed-workflow-ref",
                _SIGNER_WORKFLOW_REF,
                "--allowed-event-name",
                "workflow_dispatch",
            ],
            pass_fds=(signer_connection.fileno(), key_read),
            env={
                "GITHUB_ACTIONS": "true",
                # Fork-controlled repository: must be refused before the key is read.
                "GITHUB_REPOSITORY": "attacker/rulespec-uk",
                "GITHUB_WORKFLOW_REF": _SIGNER_WORKFLOW_REF,
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "PATH": os.environ.get("PATH", ""),
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        signer_connection.close()
        supervisor_connection.close()
        os.close(key_read)
    assert completed.returncode == 2
    assert "does not match the expected repository" in completed.stderr


def test_targeted_signed_reencode_shell_steps_have_valid_syntax(tmp_path: Path) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )

    for job_name, job in workflow["jobs"].items():
        for index, step in enumerate(job.get("steps", [])):
            command = step.get("run")
            if command is None:
                continue
            script = tmp_path / f"{job_name}-{index}.bash"
            script.write_text(command, encoding="utf-8")
            subprocess.run(["bash", "-n", str(script)], check=True)


def test_targeted_signed_reencode_workflow_is_main_dispatch_only() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    trigger = workflow.get("on", workflow.get(True))
    assert set(trigger) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"actions": "read", "contents": "read"}
    assert workflow["concurrency"] == {
        "group": (
            "targeted-signed-reencode-${{ "
            "github.actor == 'github-actions[bot]' && inputs.queue_id && "
            "inputs.queue_item_generation_sha256 || github.run_id }}"
        ),
        "cancel-in-progress": False,
    }

    def concurrency_suffix(
        actor: str, queue_id: str, generation_sha: str, run_id: str
    ) -> str:
        return (
            generation_sha
            if actor == "github-actions[bot]" and queue_id and generation_sha
            else run_id
        )

    assert (
        concurrency_suffix("github-actions[bot]", "snap", "queue-generation", "run-1")
        == "queue-generation"
    )
    assert (
        concurrency_suffix("manual-user", "snap", "queue-generation", "run-2")
        == "run-2"
    )
    assert concurrency_suffix("manual-user", "", "queue-generation", "run-3") == "run-3"
    assert concurrency_suffix("github-actions[bot]", "snap", "", "run-4") == "run-4"
    inputs = trigger["workflow_dispatch"]["inputs"]
    assert "allowlisted reviewed SHA" in inputs["rulespec_ref"]["description"]
    assert "artifact-only" in inputs["rulespec_ref"]["description"]
    assert inputs["country"] == {
        "description": "Canonical RuleSpec country checkout (for rulespec-<country>)",
        "required": True,
        "default": "us",
        "type": "string",
    }
    assert inputs["open_pr"]["type"] == "boolean"
    assert inputs["open_pr"]["default"] is False
    assert inputs["pr_base_branch"]["type"] == "string"
    assert inputs["pr_base_branch"]["default"] == "main"
    assert inputs["source_bundle_json"] == {
        "description": (
            "JSON array of additional same-jurisdiction citations to encode as "
            "signed atomic imports"
        ),
        "required": False,
        "default": "[]",
        "type": "string",
    }
    assert inputs["existing_signed_imports_json"] == {
        "description": (
            "JSON array of tracked same-jurisdiction signed-v5 modules to reuse "
            "as direct imports"
        ),
        "required": False,
        "default": "[]",
        "type": "string",
    }
    assert inputs["replace_rulespec_path"]["required"] is False
    assert inputs["replace_legacy_rulespec_path"]["required"] is False
    assert inputs["dependent_citation"]["required"] is False
    assert inputs["dependent_review_finding"]["required"] is False
    assert inputs["second_dependent_citation"]["required"] is False
    assert inputs["second_dependent_review_finding"]["required"] is False
    assert inputs["queue_id"]["required"] is False
    assert inputs["queue_item_id"]["required"] is False
    assert inputs["queue_manifest_sha256"]["required"] is False
    assert inputs["queue_item_generation_sha256"]["required"] is False
    assert inputs["queue_dispatcher_run_id"]["required"] is False
    assert "[${{ inputs.queue_id || 'adhoc' }}:" in workflow["run-name"]

    job = workflow["jobs"]["encode"]
    assert job["name"] == "Queue protected signed RuleSpec re-encode"
    assert job["environment"] == "production-signing"
    assert "github.ref == 'refs/heads/main'" in job["if"]
    assert "github.actor == 'github-actions[bot]'" in job["if"]
    assert "github.run_attempt == 1" in job["if"]
    steps = job["steps"]
    country_step = next(
        step for step in steps if step.get("name") == "Validate country routing input"
    )
    assert "prepare_signed_backfill.py" in country_step["run"]
    assert 'validate-country "$COUNTRY"' in country_step["run"]
    assert "validate-queue-tracking" in country_step["run"]
    assert '"$QUEUE_ID" "$QUEUE_ITEM_ID"' in country_step["run"]
    assert '"$QUEUE_MANIFEST_SHA256"' in country_step["run"]
    assert '"$QUEUE_ITEM_GENERATION_SHA256"' in country_step["run"]
    assert "prepare_signed_queue.py" in country_step["run"]
    assert "validate-dispatch" in country_step["run"]
    assert "QUEUE_DISPATCHER_RUN_ID" in country_step["env"]
    assert (
        "queue target must be created by an authenticated dispatcher"
        in (country_step["run"])
    )
    assert 'GITHUB_RUN_ATTEMPT" != "1"' in country_step["run"]
    assert "dispatch-signed-snap-queue.yml" in country_step["run"]
    assert ".run_attempt >= 1" in country_step["run"]
    assert "jobs?filter=all&per_page=100" in country_step["run"]
    assert "Dispatch protected SNAP queue" in country_step["run"]
    assert '.conclusion == "skipped"' in country_step["run"]
    assert "snap-queue-reconciliation-$QUEUE_DISPATCHER_RUN_ID" in (country_step["run"])
    assert "snap-queue-plan.json" in country_step["run"]
    assert "dispatched-run-records.jsonl" in country_step["run"]
    assert "workflow_run_id == $run_id" in country_step["run"]
    assert "workflow_run_attempt == 1" in country_step["run"]
    assert (
        '"axiom-encode/data/encoding-queues/${QUEUE_ID}.json"' in (country_step["run"])
    )
    assert steps.index(country_step) < next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Checkout canonical RuleSpec country"
    )
    checkout_steps = [
        step for step in steps if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert checkout_steps
    assert all(step["with"]["persist-credentials"] is False for step in checkout_steps)
    assert all(step["with"]["fetch-depth"] == 0 for step in checkout_steps)

    source_bundle_step = next(
        step for step in steps if step.get("name") == "Validate atomic source inputs"
    )
    source_bundle_command = source_bundle_step["run"]
    assert source_bundle_step["env"]["SOURCE_BUNDLE_JSON"] == (
        "${{ inputs.source_bundle_json }}"
    )
    assert source_bundle_step["env"]["EXISTING_SIGNED_IMPORTS_JSON"] == (
        "${{ inputs.existing_signed_imports_json }}"
    )
    assert 'parse-source-bundle "${SOURCE_BUNDLE_JSON:-[]}"' in (source_bundle_command)
    assert 'parse-existing-signed-imports "$RULESPEC_CHECKOUT"' in (
        source_bundle_command
    )
    assert 'existing_import_args+=(--source-citation "$source_citation")' in (
        source_bundle_command
    )
    assert 'existing_import_args+=(--exclude-rulespec-path "$reserved_path")' in (
        source_bundle_command
    )
    assert '--primary-citation "$CITATION"' in source_bundle_command
    assert 'source_bundle_args+=(--exclude-citation "$DEPENDENT_CITATION")' in (
        source_bundle_command
    )
    assert (
        "queue-authorized re-encodes cannot add source inputs until queue "
        "generation identity binds them"
    ) in source_bundle_command
    assert (
        "source-bundle replacements cannot include dependent migrations"
        in source_bundle_command
    )
    assert steps.index(source_bundle_step) > next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Install encoder"
    )

    identity_step = next(
        step
        for step in steps
        if step.get("name") == "Verify immutable checkout identities"
    )
    identity_command = identity_step["run"]
    assert "^[0-9a-f]{40}$" in identity_command
    assert "rev-parse HEAD" in identity_command
    assert "merge-base --is-ancestor" in identity_command
    assert "validate-rulespec-base" in identity_command
    assert '"$RULESPEC_REF" "$OPEN_PR" "$PR_BASE_BRANCH"' in identity_command
    assert identity_step["env"]["OPEN_PR"] == "${{ inputs.open_pr }}"
    assert identity_step["env"]["PR_BASE_BRANCH"] == ("${{ inputs.pr_base_branch }}")
    assert '"https://github.com/TheAxiomFoundation/rulespec-$COUNTRY"' in (
        identity_command
    )

    release_step = next(
        step
        for step in steps
        if step.get("name") == "Fetch pinned signed corpus release object"
    )
    assert release_step["env"] == {
        "SUPABASE_URL": "https://swocpijqqahhuwtuahwc.supabase.co",
        "SUPABASE_ANON_KEY": "${{ vars.NEXT_PUBLIC_SUPABASE_ANON_KEY }}",
        "RULESPEC_CHECKOUT": "rulespec-${{ inputs.country }}",
        "QUEUE_ID": "${{ inputs.queue_id }}",
        "QUEUE_MANIFEST_SHA256": "${{ inputs.queue_manifest_sha256 }}",
    }
    release_command = release_step["run"]
    assert "materialize_corpus_release.py" in release_command
    assert "$RULESPEC_CHECKOUT/.axiom/toolchain.toml" in release_command
    assert 'pin --toolchain "$toolchain"' in release_command
    assert "validate-release-pin" in release_command
    assert '--manifest-sha256 "$QUEUE_MANIFEST_SHA256"' in release_command
    assert 'mktemp "$RUNNER_TEMP/' in release_command
    assert "/rest/v1/release_objects?select=release_object" in release_command
    assert "content_sha256=eq.${release_sha}&limit=2" in release_command
    assert "--proto '=https' --proto-redir '=https' --tlsv1.2" in release_command
    assert "--max-filesize 16777216" in release_command
    assert "Accept-Profile: corpus" in release_command
    assert (
        'materialize --toolchain "$toolchain" --response "$response"' in release_command
    )
    assert "--corpus-root axiom-corpus" in release_command
    assert 'merge-base --is-ancestor "$release_commit" HEAD' in release_command

    provision_step = next(
        step
        for step in steps
        if step.get("name") == "Provision protected signing supervisor"
    )
    assert "sudo chown 0:0 /opt" in provision_step["run"]
    assert "sudo chmod go-w /opt" in provision_step["run"]
    assert "--git /usr/bin/git" in provision_step["run"]
    assert (
        '--encoder-git-root "$GITHUB_WORKSPACE/axiom-encode"' in provision_step["run"]
    )
    assert '--encoder-commit "$GITHUB_SHA"' in provision_step["run"]
    assert (
        "--encoder-origin-repository "
        "github.com/TheAxiomFoundation/axiom-encode" in provision_step["run"]
    )

    routing_step = next(
        step
        for step in steps
        if step.get("name") == "Verify protected RuleSpec routing"
    )
    routing_command = routing_step["run"]
    assert "trusted_path=/opt/axiom-verification/python/bin" in routing_command
    assert 'env -i PATH="$trusted_path" HOME="$trusted_home"' in routing_command
    assert "canonical_rulespec_repo_name(checkout)" in routing_command
    assert "inspect_canonical_rulespec_checkout" in routing_command
    assert '_harden_signing_capability_process(role="routing-probe")' in routing_command
    assert "libc.prctl(38, 1, 0, 0, 0)" in routing_command
    assert "protected RuleSpec routing rejected checkout" in routing_command
    assert "hardened RuleSpec routing rejected checkout" in routing_command

    cascade_step = next(
        step for step in steps if step.get("name") == "Validate dependent cascade"
    )
    assert "inputs.dependent_citation != ''" in cascade_step["if"]
    assert "inputs.second_dependent_citation != ''" in cascade_step["if"]
    assert "validate-dependent-cascade" in cascade_step["run"]
    assert 'dependent_citations=("$DEPENDENT_CITATION")' in cascade_step["run"]
    assert (
        'dependent_citations+=("$SECOND_DEPENDENT_CITATION")' in (cascade_step["run"])
    )
    assert (
        '"$RULESPEC_CHECKOUT" "$CITATION" "${dependent_citations[@]}"'
        not in (cascade_step["run"])
    )
    assert cascade_step["env"]["REPLACE_RULESPEC_PATH"] == (
        "${{ inputs.replace_rulespec_path }}"
    )
    assert 'cascade_target="$REPLACE_RULESPEC_PATH"' in cascade_step["run"]
    assert 'cascade_target="$REPLACE_LEGACY_RULESPEC_PATH"' in cascade_step["run"]
    assert (
        'cascade_args+=(--target-rulespec-path "$cascade_target")'
        in cascade_step["run"]
    )
    assert 'cascade_args+=("${dependent_citations[@]}")' in cascade_step["run"]
    assert '"${cascade_args[@]}"' in cascade_step["run"]

    signed_import_step = next(
        step for step in steps if step.get("name") == "Verify existing signed imports"
    )
    signed_import_command = signed_import_step["run"]
    assert steps.index(cascade_step) < steps.index(signed_import_step)
    assert "/opt/axiom-verification/axiom-encode-signing-supervisor" in (
        signed_import_command
    )
    assert "signed-import-inventory" in signed_import_command
    assert '--base-ref "$RULESPEC_REF"' in signed_import_command
    assert '--rulespec-path "$existing_import_path"' in signed_import_command
    assert '> "$RUNNER_TEMP/existing-signed-import-inventory.json"' in (
        signed_import_command
    )

    apply_step = next(
        step
        for step in steps
        if step.get("name") == "Encode, review, validate, and apply"
    )
    assert apply_step["id"] == "encode_apply"
    assert apply_step["env"]["AXIOM_ENCODE_APPLY_SIGNING_KEY"] == (
        "${{ secrets.AXIOM_ENCODE_APPLY_SIGNING_KEY }}"
    )
    assert "AXIOM_ENCODE_APPLY_CHECKOUT" not in apply_step["env"]
    command = apply_step["run"]
    assert "run_signed_encode()" in command
    assert "/opt/axiom-verification/axiom-encode-apply-signer run" in command
    assert "--key-env AXIOM_ENCODE_APPLY_SIGNING_KEY" in command
    assert (
        "TheAxiomFoundation/axiom-encode/.github/workflows/"
        "targeted-signed-reencode.yml@refs/heads/main" in command
    )
    assert "--allowed-event-name workflow_dispatch" in command
    assert "--apply" in command
    assert "--require-complete-source-unit" in command
    assert "--skip-reviewers" not in command
    assert 'mktemp -d "$RUNNER_TEMP/axiom-targeted-review-finding.XXXXXX"' in command
    assert 'review_finding_path="$review_finding_dir/review-finding.txt"' in command
    assert "$GITHUB_WORKSPACE/axiom-rules-engine/.axiom-targeted" not in command
    assert "printf '%s\\n' \"$review_finding\"" in command
    assert 'args+=(--review-findings "$review_finding_path")' in command
    assert "args+=(--apply-target-only)" in command
    assert 'args+=(--replace-rulespec-path "$replacement_path")' in command
    assert 'local require_direct_imports="$7"' in command
    assert 'args+=(--required-import-rulespec-path "$required_import_path")' in (
        command
    )
    assert "--legacy-dependent-rulespec-path" in command
    assert 'citation-rulespec-path "$DEPENDENT_CITATION"' in command
    assert '[ "$target_only" = "true" ]' in command
    assert (
        "queue-authorized re-encodes cannot override the RuleSpec target path"
        in command
    )
    assert '--output "$RUNNER_TEMP/generated/$output_lane"' in command
    assert '"$SECOND_DEPENDENT_CITATION"' in command
    assert '"$SECOND_DEPENDENT_REVIEW_FINDING" false dependent-2 "" "" false' in command
    assert '"$CITATION" "$REVIEW_FINDING" true target \\\n' in command
    assert '"$DEPENDENT_CITATION" "$DEPENDENT_REVIEW_FINDING" \\\n' in command
    assert '"$REPLACE_RULESPEC_PATH" "$final_legacy_source_path"' in command
    assert '"$CITATION" "$REVIEW_FINDING" false \\\n' in command
    assert "dependent review finding is required with dependent citation" in command
    assert "parse-source-bundle" in command
    assert "parse-existing-signed-imports" in command
    assert '"$RUNNER_TEMP/existing-signed-import-paths.txt"' in command
    assert "source_bundle_count + existing_import_count" in command
    assert '> "$RUNNER_TEMP/source-bundle.json"' in command
    assert '> "$RUNNER_TEMP/source-bundle-citations.txt"' in command
    assert 'source_lane="$(printf \'source-%02d\' "$source_index")"' in command
    assert '"$source_citation" "" true "$source_lane" "" "" false' in command
    assert '"$CITATION" "" false target-preflight \\' in command
    assert '"$REPLACE_LEGACY_RULESPEC_PATH" false' in command
    assert 'final_legacy_source_path=""' in command
    assert (
        "source-bundle replacements cannot include dependent migrations" in command
    )
    assert "Canonicalize signed replacement target before source bundle" in command
    assert "checkpoint_signed_changes()" in command
    assert 'local guard_status="$?"' in command
    assert 'jq . "$RUNNER_TEMP/checkpoint-guard-generated.json" >&2' in command
    assert '"$workflow_python" "$backfill_helper" stage "$RULESPEC_CHECKOUT"' in (
        command
    )
    assert 'commit -m "$message"' in command
    source_loop = "while IFS= read -r source_citation; do"
    assert command.rindex(source_loop) < command.index('"$CITATION" "$REVIEW_FINDING"')
    assert "Compose signed source bundle for ${CITATION}" in command
    assert (
        "queue-authorized re-encodes cannot add source inputs until queue "
        "generation identity binds them"
    ) in command
    assert steps.index(cascade_step) < steps.index(apply_step)
    assert steps.index(signed_import_step) < steps.index(apply_step)

    failure_package_step = next(
        step
        for step in steps
        if step.get("name") == "Package failed re-encode diagnostics"
    )
    assert steps.index(
        upload_step := next(
            step
            for step in steps
            if step.get("name") == "Upload signed re-encode artifact"
        )
    ) + 1 == steps.index(failure_package_step)
    assert failure_package_step["if"] == "${{ failure() && !cancelled() }}"
    assert set(failure_package_step["env"]) == {
        "CITATION",
        "CORPUS_REF",
        "COUNTRY",
        "DEPENDENT_CITATION",
        "ENCODE_APPLY_CONCLUSION",
        "ENCODE_APPLY_OUTCOME",
        "EXISTING_SIGNED_IMPORTS_JSON",
        "FINALIZE_SIGNED_REENCODE_ARTIFACT_CONCLUSION",
        "FINALIZE_SIGNED_REENCODE_ARTIFACT_OUTCOME",
        "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
        "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON",
        "OPEN_PR",
        "PACKAGE_EXACT_GENERATED_CHANGES_CONCLUSION",
        "PACKAGE_EXACT_GENERATED_CHANGES_OUTCOME",
        "COMMIT_REVIEWED_LANE_CHANGES_CONCLUSION",
        "COMMIT_REVIEWED_LANE_CHANGES_OUTCOME",
        "PR_BASE_BRANCH",
        "PUBLISH_LANE_PULL_REQUEST_CONCLUSION",
        "PUBLISH_LANE_PULL_REQUEST_OUTCOME",
        "QUEUE_DISPATCHER_RUN_ID",
        "QUEUE_ID",
        "QUEUE_ITEM_GENERATION_SHA256",
        "QUEUE_ITEM_ID",
        "QUEUE_MANIFEST_SHA256",
        "REPLACE_LEGACY_RULESPEC_PATH",
        "REPLACE_RULESPEC_PATH",
        "RULES_ENGINE_REF",
        "RULESPEC_REF",
        "SECOND_DEPENDENT_CITATION",
        "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
        "SOURCE_BUNDLE_JSON",
        "UPLOAD_SIGNED_REENCODE_ARTIFACT_CONCLUSION",
        "UPLOAD_SIGNED_REENCODE_ARTIFACT_OUTCOME",
        "VERIFY_EXISTING_SIGNED_IMPORT_INTEGRITY_CONCLUSION",
        "VERIFY_EXISTING_SIGNED_IMPORT_INTEGRITY_OUTCOME",
        "VERIFY_GENERATED_PROVENANCE_CONCLUSION",
        "VERIFY_GENERATED_PROVENANCE_OUTCOME",
    }
    assert "toJSON(steps)" not in json.dumps(failure_package_step)
    assert ".outputs" not in json.dumps(failure_package_step["env"])
    failure_package_command = failure_package_step["run"]
    assert "/opt/axiom-verification/python/bin/python -I -" in (failure_package_command)
    assert "read_bounded_regular_file" in failure_package_command
    assert "followlinks=False" in failure_package_command
    assert "generated diagnostics exceed entry limit" in failure_package_command
    assert "generated diagnostics exceed file limit" in failure_package_command
    assert "generated diagnostics exceed size limit" in failure_package_command
    assert 'tar -cf "$RUNNER_TEMP/targeted-reencode-failure.tar"' in (
        failure_package_command
    )
    assert '-C "$artifact" .' in failure_package_command
    assert '"failed_steps": failed_steps' in failure_package_command
    assert '"generated_lanes": generated_lanes' in failure_package_command
    assert '"queue_item_generation_sha256"' in failure_package_command
    assert '"step_outcomes": step_outcomes' in failure_package_command
    assert '"schema": "axiom-encode/failed-reencode-diagnostics/v1"' in (
        failure_package_command
    )

    failure_upload_step = next(
        step
        for step in steps
        if step.get("name") == "Upload failed re-encode diagnostics"
    )
    assert steps.index(failure_package_step) + 1 == steps.index(failure_upload_step)
    assert "failure() && !cancelled()" in failure_upload_step["if"]
    assert (
        "steps.package_failed_reencode_diagnostics.outcome == 'success'"
        in (failure_upload_step["if"])
    )
    assert failure_upload_step["with"] == {
        "name": (
            "targeted-reencode-failure-${{ github.run_id }}-${{ github.run_attempt }}"
        ),
        "path": "${{ runner.temp }}/targeted-reencode-failure.tar",
        "if-no-files-found": "error",
        "retention-days": 90,
    }

    integrity_step = next(
        step
        for step in steps
        if step.get("name") == "Verify existing signed import integrity"
    )
    assert integrity_step["id"] == "verify_existing_signed_import_integrity"
    integrity_command = integrity_step["run"]
    assert steps.index(apply_step) < steps.index(integrity_step)
    assert "signed-import-inventory" in integrity_command
    assert "existing-signed-import-inventory-final.json" in integrity_command
    assert "cmp --silent" in integrity_command

    package_step = next(
        step for step in steps if step.get("name") == "Package exact generated changes"
    )
    assert package_step["id"] == "package_exact_generated_changes"
    package_command = package_step["run"]
    assert package_step["env"]["REVIEW_FINDING_PRESENT"] == (
        "${{ inputs.review_finding != '' }}"
    )
    assert package_step["env"]["DEPENDENT_REVIEW_FINDING_PRESENT"] == (
        "${{ inputs.dependent_review_finding != '' }}"
    )
    assert package_step["env"]["PR_BASE_BRANCH"] == ("${{ inputs.pr_base_branch }}")
    assert package_step["env"]["RULESPEC_REF"] == "${{ inputs.rulespec_ref }}"
    assert '"$RULESPEC_REF" > "$artifact/tracked.patch"' in package_command
    assert '"$RULESPEC_REF" HEAD >> "$artifact/status.txt"' in package_command
    assert "diff --binary --full-index HEAD" not in package_command
    assert 'cp "$RUNNER_TEMP/source-bundle.json" "$artifact/source-bundle.json"' in (
        package_command
    )
    assert '"$artifact/existing-signed-imports.json"' in package_command
    assert '"$artifact/signed-import-inventory.json"' in package_command
    assert '"$artifact/context-manifest.json"' in package_command
    assert '".axiom/encoding-manifests"' in package_command
    assert 'citation = effective.get("citation")' in package_command
    assert 'evidence_manifest["context_manifest_file"]' in package_command
    assert "evidence_manifest.get(" in package_command
    assert "is still pending" in package_command
    assert "primary_paths != [normalized_replacement_path]" in package_command
    assert "replacement apply manifest primary path does not " in package_command
    assert "match the requested RuleSpec target" in package_command
    assert 'finding.get("content")' in package_command
    assert 'finding.get("sha256")' in package_command
    assert '"dependent-context-manifest.json"' in package_command
    assert 'f"{source_lane}-context-manifest.json"' in package_command
    assert 'os.environ["RULESPEC_REF"], "--"' in package_command
    assert '"source_bundle": json.loads(' in package_command
    assert '"existing_signed_imports": json.loads(' in package_command
    assert '"signed_import_inventory_sha256": hashlib.sha256(' in package_command
    assert '"rulespec_base": os.environ["RULESPEC_REF"]' in package_command
    assert '"rulespec_generated_head": rev(os.environ["RULESPEC_CHECKOUT"])' in (
        package_command
    )
    assert '"pr_base_branch": os.environ["PR_BASE_BRANCH"]' in package_command
    assert '"queue_id": os.environ.get("QUEUE_ID") or None' in package_command
    assert '"queue_item_id": os.environ.get("QUEUE_ITEM_ID") or None' in (
        package_command
    )
    assert '"queue_item_generation_sha256": (' in package_command
    assert '"queue_manifest_sha256": (' in package_command
    assert '"workflow_run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"])' in (
        package_command
    )
    trusted_python = "/opt/axiom-verification/python/bin/python"
    assert f"workflow_python=({trusted_python} -I)" in package_command
    assert '"${workflow_python[@]}" - \\\n' in package_command

    commit_step = next(
        step
        for step in steps
        if step.get("name") == "Commit reviewed lane changes locally"
    )
    assert commit_step["id"] == "commit_reviewed_lane_changes"
    assert f"workflow_python=({trusted_python} -I)" in commit_step["run"]
    assert '"${workflow_python[@]}" \\\n' in commit_step["run"]
    assert (
        "axiom-encode/scripts/prepare_signed_backfill.py stage \\\n"
        in (commit_step["run"])
    )

    guard_step = next(
        step for step in steps if step.get("name") == "Verify generated provenance"
    )
    assert guard_step["id"] == "verify_generated_provenance"
    assert "guard-generated" in guard_step["run"]
    assert '--base-ref "$RULESPEC_REF"' in guard_step["run"]
    assert "guard_ref_args" not in guard_step["run"]
    assert 'guard_status="$?"' in guard_step["run"]
    assert 'jq . "$RUNNER_TEMP/guard-generated.json" >&2' in guard_step["run"]

    secret_steps = [
        step
        for step in steps
        if "AXIOM_ENCODE_APPLY_SIGNING_KEY" in (step.get("env") or {})
    ]
    assert secret_steps == [apply_step]

    publish_step = next(
        step
        for step in steps
        if step.get("name") == "Push lane branch and open draft pull request"
    )
    assert publish_step["id"] == "publish_lane_pull_request"
    assert publish_step["if"] == "${{ inputs.open_pr }}"
    assert publish_step["env"]["GH_TOKEN"] == "${{ secrets.AXIOM_REPO_TOKEN }}"
    assert publish_step["env"]["PR_BASE_BRANCH"] == ("${{ inputs.pr_base_branch }}")
    assert "AXIOM_ENCODE_APPLY_SIGNING_KEY" not in publish_step["env"]
    publish_command = publish_step["run"]
    assert f"workflow_python=({trusted_python} -I)" in publish_command
    assert 'repo="TheAxiomFoundation/rulespec-${COUNTRY}"' in publish_command
    assert '"$COUNTRY" "$GITHUB_RUN_ID" "$GITHUB_RUN_ATTEMPT"' in publish_command
    assert "core.hooksPath=/dev/null" in publish_command
    assert "fetch --no-tags origin \\\n" in publish_command
    assert "refs/remotes/origin/${PR_BASE_BRANCH}" in publish_command
    assert '" = "$RULESPEC_REF"' in publish_command
    assert '"HEAD:refs/heads/${branch}"' in publish_command
    assert publish_command.count('"HEAD:refs/heads/${branch}"') == 1
    assert "gh api --method POST" in publish_command
    assert '-f base="$PR_BASE_BRANCH"' in publish_command
    assert "-F draft=true" in publish_command
    assert "Queue item:" in publish_command
    assert "Queue generation SHA-256:" in publish_command
    assert "Queue manifest SHA-256:" in publish_command
    assert "'.base.ref == $branch and .base.sha == $sha'" in publish_command
    assert "pulls/${pr_number}" in publish_command
    assert "-f state=closed" in publish_command
    assert '":refs/heads/${branch}"' in publish_command
    assert "created pull request does not target the reviewed base SHA" in (
        publish_command
    )
    assert "SHA256SUMS" not in publish_command
    assert not any(
        " push " in f" {step.get('run', '')} "
        for step in steps[: steps.index(publish_step)]
    )

    checksum_step = next(
        step
        for step in steps
        if step.get("name") == "Finalize signed re-encode artifact checksums"
    )
    assert checksum_step["id"] == "finalize_signed_reencode_artifact"
    assert "if" not in checksum_step
    checksum_command = checksum_step["run"]
    assert 'artifact="$RUNNER_TEMP/targeted-reencode"' in checksum_command
    assert 'cd "$artifact"' in checksum_command
    assert "sha256sum * > SHA256SUMS" in checksum_command
    assert 'sha256sum "$RUNNER_TEMP/targeted-reencode"/*' not in checksum_command
    assert upload_step["id"] == "upload_signed_reencode_artifact"
    assert upload_step["with"]["name"] == (
        "targeted-reencode-${{ github.run_id }}-${{ github.run_attempt }}"
    )
    assert steps.index(checksum_step) + 1 == steps.index(upload_step)
    assert steps.index(failure_upload_step) == len(steps) - 1


def test_targeted_signed_reencode_packages_bounded_failure_diagnostics(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    step = next(
        item
        for item in workflow["jobs"]["encode"]["steps"]
        if item.get("name") == "Package failed re-encode diagnostics"
    )
    command = step["run"].replace(
        "/opt/axiom-verification/python/bin/python",
        sys.executable,
    )
    generated = tmp_path / "generated" / "target" / "model"
    generated.mkdir(parents=True)
    (generated / "statutes").mkdir()
    (generated / "statutes/54a:4-7.test.yaml").write_text("format: rulespec/v1\n")
    (generated / "target.repair.json").write_text('{"outcome": "blocked"}\n')
    (generated / "ignored.bin").write_bytes(b"not diagnostic output")
    (tmp_path / "guard-generated.json").write_text(
        json.dumps(
            {
                "repo": "/runner/rulespec-us",
                "passed": False,
                "issues": ["waiver transition requires protected base"],
            }
        )
        + "\n"
    )
    env = {
        **os.environ,
        "CITATION": "us/statute/42/1437c-1",
        "CORPUS_REF": "corpus-ref",
        "COUNTRY": "us",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_RUN_ID": "1234",
        "GITHUB_SHA": "encoder-ref",
        "ENCODE_APPLY_CONCLUSION": "failure",
        "ENCODE_APPLY_OUTCOME": "failure",
        "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": '["us/statutes/old.yaml"]',
        "RULES_ENGINE_REF": "rules-engine-ref",
        "RULESPEC_REF": "rulespec-ref",
        "RUNNER_TEMP": str(tmp_path),
        "VERIFY_GENERATED_PROVENANCE_CONCLUSION": "skipped",
        "VERIFY_GENERATED_PROVENANCE_OUTCOME": "skipped",
        "QUEUE_ID": "us-snap-all-states-2026-07",
        "QUEUE_ITEM_GENERATION_SHA256": "generation-sha",
        "QUEUE_ITEM_ID": "us/statute/42/1437c-1",
        "QUEUE_MANIFEST_SHA256": "manifest-sha",
    }

    subprocess.run(["bash", "-c", command], env=env, check=True)

    archive = tmp_path / "targeted-reencode-failure.tar"
    with tarfile.open(archive, mode="r") as bundle:
        file_member_names = {
            member.name.removeprefix("./")
            for member in bundle.getmembers()
            if member.isfile()
        }
        assert file_member_names == {
            "generated/target/model/statutes/54a:4-7.test.yaml",
            "generated/target/model/target.repair.json",
            "guards/guard-generated.json",
            "metadata.json",
        }
        assert "generated/target/model/ignored.bin" not in file_member_names
        target = bundle.extractfile(
            "./generated/target/model/statutes/54a:4-7.test.yaml"
        )
        repair = bundle.extractfile("./generated/target/model/target.repair.json")
        guard = bundle.extractfile("./guards/guard-generated.json")
        metadata_file = bundle.extractfile("./metadata.json")
        assert target is not None
        assert repair is not None
        assert guard is not None
        assert metadata_file is not None
        assert target.read() == b"format: rulespec/v1\n"
        assert json.loads(repair.read()) == {"outcome": "blocked"}
        assert json.loads(guard.read()) == {
            "repo": "/runner/rulespec-us",
            "passed": False,
            "issues": ["waiver transition requires protected base"],
        }
        metadata = json.loads(metadata_file.read())
    assert metadata["schema"] == "axiom-encode/failed-reencode-diagnostics/v1"
    assert metadata["workflow_run_id"] == "1234"
    assert metadata["failed_steps"] == ["encode_apply"]
    assert metadata["generated_lanes"] == ["target"]
    assert metadata["legacy_retained_successor_rulespec_paths_input"] == (
        '["us/statutes/old.yaml"]'
    )
    assert metadata["queue_id"] == "us-snap-all-states-2026-07"
    assert metadata["queue_item_generation_sha256"] == "generation-sha"
    assert metadata["queue_item_id"] == "us/statute/42/1437c-1"
    assert metadata["queue_manifest_sha256"] == "manifest-sha"
    assert metadata["step_outcomes"] == {
        "encode_apply": {"conclusion": "failure", "outcome": "failure"},
        "verify_generated_provenance": {
            "conclusion": "skipped",
            "outcome": "skipped",
        },
    }
    assert [item["path"] for item in metadata["files"]] == [
        "target/model/statutes/54a:4-7.test.yaml",
        "target/model/target.repair.json",
    ]
    target_inventory = metadata["files"][0]
    assert target_inventory["size"] == len(b"format: rulespec/v1\n")
    assert (
        target_inventory["sha256"]
        == hashlib.sha256(b"format: rulespec/v1\n").hexdigest()
    )


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("symlink", "Could not safely open"),
        ("oversize", "safety limit"),
        ("malformed", "not valid JSON"),
        ("wrong_shape", "invalid shape"),
    ],
)
def test_targeted_signed_reencode_rejects_unsafe_guard_diagnostics(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    step = next(
        item
        for item in workflow["jobs"]["encode"]["steps"]
        if item.get("name") == "Package failed re-encode diagnostics"
    )
    command = step["run"].replace(
        "/opt/axiom-verification/python/bin/python",
        sys.executable,
    )
    (tmp_path / "generated").mkdir()
    guard = tmp_path / "guard-generated.json"
    if mutation == "symlink":
        outside = tmp_path / "outside.json"
        outside.write_text('{"repo":"outside","passed":false,"issues":[]}\n')
        guard.symlink_to(outside)
    elif mutation == "oversize":
        guard.write_bytes(b"x" * (1024 * 1024 + 1))
    elif mutation == "malformed":
        guard.write_text("{\n")
    else:
        guard.write_text(
            json.dumps(
                {
                    "repo": "/runner/rulespec-us",
                    "passed": False,
                    "issues": [],
                    "unexpected": True,
                }
            )
            + "\n"
        )

    completed = subprocess.run(
        ["bash", "-c", command],
        env={
            **os.environ,
            "CITATION": "us/statute/42/1437c-1",
            "CORPUS_REF": "corpus-ref",
            "COUNTRY": "us",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_RUN_ID": "1234",
            "GITHUB_SHA": "encoder-ref",
            "RULES_ENGINE_REF": "rules-engine-ref",
            "RULESPEC_REF": "rulespec-ref",
            "RUNNER_TEMP": str(tmp_path),
        },
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert expected_error in completed.stderr
    assert not (tmp_path / "targeted-reencode-failure.tar").exists()


def test_targeted_signed_reencode_preserves_checkpoint_guard_failure(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Encode, review, validate, and apply"
    )
    checkpoint = command.split("checkpoint_signed_changes() {", 1)[1].split(
        '\n}\n\nfinal_legacy_source_path=',
        1,
    )[0]
    guard_stub = tmp_path / "guard-stub"
    guard_stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' "
        '\'{"repo":"/runner/rulespec-us","passed":false,'
        '"issues":["checkpoint manifest mismatch"]}\'\n'
        "exit 7\n"
    )
    guard_stub.chmod(0o700)
    script = (
        "set -euo pipefail\n"
        "checkpoint_signed_changes() {"
        + checkpoint.replace(
            "/opt/axiom-verification/axiom-encode-signing-supervisor",
            '"$GUARD_STUB"',
        )
        + "\n}\ncheckpoint_signed_changes test\n"
    )

    completed = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "GITHUB_WORKSPACE": str(tmp_path),
            "GUARD_STUB": str(guard_stub),
            "RULESPEC_CHECKOUT": str(tmp_path / "rulespec-us"),
            "RUNNER_TEMP": str(tmp_path),
        },
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 7
    assert "checkpoint manifest mismatch" in completed.stderr
    assert json.loads((tmp_path / "checkpoint-guard-generated.json").read_text()) == {
        "repo": "/runner/rulespec-us",
        "passed": False,
        "issues": ["checkpoint manifest mismatch"],
    }


def test_targeted_signed_reencode_rejects_symlinked_failure_diagnostics(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    step = next(
        item
        for item in workflow["jobs"]["encode"]["steps"]
        if item.get("name") == "Package failed re-encode diagnostics"
    )
    command = step["run"].replace(
        "/opt/axiom-verification/python/bin/python",
        sys.executable,
    )
    generated = tmp_path / "generated"
    generated.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("secret\n")
    (generated / "candidate.yaml").symlink_to(outside)
    env = {
        **os.environ,
        "CITATION": "us/statute/42/1437c-1",
        "CORPUS_REF": "corpus-ref",
        "COUNTRY": "us",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_RUN_ID": "1234",
        "GITHUB_SHA": "encoder-ref",
        "ENCODE_APPLY_CONCLUSION": "failure",
        "ENCODE_APPLY_OUTCOME": "failure",
        "RULES_ENGINE_REF": "rules-engine-ref",
        "RULESPEC_REF": "rulespec-ref",
        "RUNNER_TEMP": str(tmp_path),
    }

    completed = subprocess.run(
        ["bash", "-c", command],
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "non-regular file" in completed.stderr
    assert not (
        tmp_path / "targeted-reencode-failure/generated/candidate.yaml"
    ).exists()
    assert not (tmp_path / "targeted-reencode-failure.tar").exists()


def test_signed_snap_queue_dispatcher_is_bounded_and_idempotent() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/dispatch-signed-snap-queue.yml").read_text()
    )
    trigger = workflow.get("on", workflow.get(True))
    assert set(trigger) == {"workflow_dispatch"}
    inputs = trigger["workflow_dispatch"]["inputs"]
    assert inputs["dispatch"]["type"] == "boolean"
    assert inputs["dispatch"]["default"] is False
    assert inputs["limit"]["options"] == ["1", "2", "3", "4"]
    assert workflow["permissions"] == {
        "actions": "write",
        "contents": "read",
        "issues": "write",
    }
    assert workflow["concurrency"] == {
        "group": "dispatch-signed-snap-queue",
        "cancel-in-progress": False,
    }

    job = workflow["jobs"]["dispatch"]
    assert job["name"] == "Dispatch protected SNAP queue"
    assert "github.ref == 'refs/heads/main'" in job["if"]
    assert "github.run_attempt == 1" in job["if"]
    assert "environment" not in job
    steps = job["steps"]
    checkout = steps[0]
    assert checkout["with"]["persist-credentials"] is False
    assert checkout["with"]["fetch-depth"] == 0

    selection = next(
        step
        for step in steps
        if step.get("name") == "Select and validate bounded queue tranche"
    )
    selection_command = selection["run"]
    assert 'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"' in selection_command
    assert "prepare_signed_queue.py validate" in selection_command
    assert "activation_merge_run_id" in inputs
    assert "verify-merge-authorization" in selection_command
    assert '--merge-jobs "$RUNNER_TEMP/activation-merge-jobs.json"' in (
        selection_command
    )
    assert "snap-queue-merge-authorization-$ACTIVATION_MERGE_RUN_ID" in (
        selection_command
    )
    assert "git log --first-parent" in selection_command
    assert "commits/$rulespec_ref/check-runs?per_page=100" in selection_command
    assert "prepare_signed_queue.py select" in selection_command
    assert '--item-ids "$ITEM_IDS" --limit "$LIMIT"' in selection_command
    assert "prepare_signed_queue.py candidates" in selection_command
    assert 'if [ -n "$ITEM_IDS" ]' not in selection_command
    assert "prepare_signed_queue.py sha256" in selection_command

    dispatch = next(
        step
        for step in steps
        if step.get("name") == "Plan or dispatch idempotent protected runs"
    )
    assert "if" not in dispatch
    assert dispatch["env"]["DO_DISPATCH"] == "${{ inputs.dispatch }}"
    assert dispatch["env"]["GH_TOKEN"] == "${{ github.token }}"
    command = dispatch["run"]
    assert "rulespec-us/pulls?state=all" in command
    assert "prepare_signed_queue.py reconcile" in command
    assert "retry_failed" not in inputs
    assert "--retry-failed" not in command
    assert "snap-queue-reconciled.json" in command
    assert "the current tranche must be finalized" in command
    assert "any(.items[]; .dispatchable == false)" in command
    assert "targeted-signed-reencode.yml/dispatches" not in command
    assert "actions/workflows/$workflow/dispatches" in command
    assert 'ref: "main"' in command
    assert 'open_pr: "true"' in command
    assert "queue_item_generation_sha256" in command
    assert "queue_manifest_sha256" in command
    assert "queue_dispatcher_run_id: $dispatcher_run_id" in command
    assert command.index(
        'queue_id="$(jq -r \'.queue_id\' "$candidates")"'
    ) < command.index('> "$selection"')
    assert 'if [ "$selected_count" -ge "$LIMIT" ]' in command
    assert "signed-encoding-queue-plan/v1" in command
    assert 'if [ "$DO_DISPATCH" = "true" ]' in command
    assert "X-GitHub-Api-Version: 2026-03-10" in command
    assert "workflow_run_id" in command
    assert "workflow_run_attempt: 1" in command
    assert "dispatched-run-records.jsonl" in command
    assert "sleep 5" not in command
    assert 'gh issue comment "$issue"' in command

    upload = next(
        step
        for step in steps
        if step.get("name") == "Upload queue reconciliation record"
    )
    assert upload["with"]["retention-days"] == 90
    assert upload["with"]["if-no-files-found"] == "warn"
    assert "snap-queue-seed.json" in upload["with"]["path"]
    assert "snap-queue-reconciled.json" in upload["with"]["path"]
    assert "snap-queue-plan.json" in upload["with"]["path"]
    assert "dispatched-run-records.jsonl" in upload["with"]["path"]


def test_signed_snap_queue_finalizer_uses_live_fail_closed_evidence() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/finalize-signed-snap-queue.yml").read_text()
    )
    trigger = workflow.get("on", workflow.get(True))
    assert set(trigger) == {"workflow_dispatch"}
    assert workflow["permissions"] == {
        "actions": "read",
        "contents": "write",
        "pull-requests": "write",
    }
    assert workflow["concurrency"] == {
        "group": "finalize-signed-snap-queue",
        "cancel-in-progress": False,
    }
    job = workflow["jobs"]["finalize"]
    assert job["name"] == "Finalize protected SNAP queue tranche"
    assert "github.run_attempt == 1" in job["if"]
    assert "github.ref == 'refs/heads/main'" in job["if"]
    steps = job["steps"]
    checkouts = [
        step for step in steps if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert len(checkouts) == 2
    assert all(step["with"]["persist-credentials"] is False for step in checkouts)
    assert all(step["with"]["fetch-depth"] == 0 for step in checkouts)

    evidence = next(
        step for step in steps if step.get("name") == "Fetch live finalization evidence"
    )
    command = evidence["run"]
    assert 'test "$(jq -r \'.state\' "$queue")" = "paused"' in command
    assert "git/ref/heads/hard-cut/canonical-layout-us" in command
    assert "commits/$NEW_RULESPEC_REF/check-runs?per_page=100" in command
    assert '.status == "completed"' in command
    assert 'IN("success", "neutral", "skipped")' in command
    assert "rulespec-us/pulls?state=all" in command
    assert "targeted-signed-reencode.yml/runs" in command
    assert "finalize-repin" in command
    assert "finalization-target-plan" in command
    assert "targeted-reencode-$run_id" in command
    assert "jobs?filter=all&per_page=100" in command
    assert "--slurpfile jobs" in command
    assert "sha256sum -c SHA256SUMS" in command
    assert '--target-evidence "$RUNNER_TEMP/target-evidence.json"' in command
    assert '"$queue" rulespec-us' in command
    assert '--check-runs "$RUNNER_TEMP/rulespec-check-runs.json"' in command
    assert '--finalizer-head-sha "$GITHUB_SHA"' in command
    assert "--finalizer-run-url" in command
    assert command.count("\"$branch_ref\" --jq '.object.sha'") == 2

    activation = next(
        step
        for step in steps
        if step.get("name") == "Create exact queue activation commit"
    )
    activation_command = activation["run"]
    assert "uv version --bump patch" in activation_command
    assert "uv lock --offline" in activation_command
    assert 'test "$(git rev-parse refs/remotes/origin/main)" = "$GITHUB_SHA"' in (
        activation_command
    )
    assert "activation-commit.json" in activation_command
    assert "activation-changed-files.json" in activation_command
    assert "HEAD^{tree}" in activation_command
    upload = next(
        step for step in steps if step.get("name") == "Upload finalization evidence"
    )
    assert "activation-commit.json" in upload["with"]["path"]
    assert steps.index(activation) < steps.index(upload)

    publish = next(
        step
        for step in steps
        if step.get("name") == "Open reviewed queue activation pull request"
    )
    publish_command = publish["run"]
    assert "gh api --method POST" in publish_command
    assert "-F draft=true" in publish_command
    assert "required independent review-fix cycle" in publish_command
    assert steps.index(upload) < steps.index(publish)


def test_snap_queue_activation_checks_and_merge_revalidate_live_state() -> None:
    validate = yaml.safe_load(
        (ROOT / ".github/workflows/validate-snap-queue-activation.yml").read_text()
    )
    trigger = validate.get("on", validate.get(True))
    assert set(trigger) == {"pull_request"}
    assert validate["permissions"] == {"actions": "read", "contents": "read"}
    validate_command = validate["jobs"]["validate"]["steps"][1]["run"]
    assert "verify-activation" in validate_command
    assert "verify-paused-transition" in validate_command
    assert '--previous-queue "$previous_queue"' in validate_command
    assert '--expected-base-sha "$BASE_SHA"' in validate_command
    assert "--require-success false" in validate_command
    assert "verify-activation-commit" in validate_command
    assert '--finalizer-jobs "$RUNNER_TEMP/finalizer-jobs.json"' in validate_command
    assert "snap-queue-finalization-$run_id" in validate_command
    assert "git/ref/heads/hard-cut/canonical-layout-us" in validate_command
    assert 'echo "initial=$initial" >> "$GITHUB_OUTPUT"' in validate_command

    steps = validate["jobs"]["validate"]["steps"]
    initial_checkouts = [
        step
        for step in steps
        if step.get("name", "").startswith("Checkout exact queue")
    ]
    assert len(initial_checkouts) == 3
    assert all(
        step["if"] == "${{ steps.transition.outputs.initial == 'true' }}"
        for step in initial_checkouts
    )
    assert {step["with"]["repository"] for step in initial_checkouts} == {
        "TheAxiomFoundation/axiom-corpus",
        "TheAxiomFoundation/rulespec-us",
        "TheAxiomFoundation/axiom-rules-engine",
    }
    provenance = next(
        step
        for step in steps
        if step.get("name") == "Regenerate and authenticate initial queue"
    )
    provenance_command = provenance["run"]
    assert (
        "AXIOM_CORPUS_RELEASE_PUBLIC_KEY"
        in provenance["env"]["CORPUS_RELEASE_PUBLIC_KEY"]
    )
    assert "release_objects?select=release_object" in provenance_command
    assert "build_command=build-snap-all-states" in provenance_command
    assert "build_command=build-snap" in provenance_command
    assert '"$build_command" initial-axiom-corpus initial-rulespec-us' in (
        provenance_command
    )
    assert "unsupported initial SNAP queue" in provenance_command
    assert "--state paused" in provenance_command
    assert "cmp --silent" in provenance_command

    merge = yaml.safe_load(
        (ROOT / ".github/workflows/merge-snap-queue-activation.yml").read_text()
    )
    trigger = merge.get("on", merge.get(True))
    assert set(trigger) == {"workflow_dispatch"}
    assert merge["permissions"] == {
        "actions": "read",
        "contents": "write",
        "pull-requests": "write",
    }
    assert "github.ref == 'refs/heads/main'" in merge["jobs"]["merge"]["if"]
    assert "github.run_attempt == 1" in merge["jobs"]["merge"]["if"]
    assert merge["jobs"]["merge"]["name"] == "Merge reviewed SNAP queue activation"
    steps = merge["jobs"]["merge"]["steps"]
    checkout = next(
        step for step in steps if step.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"]["ref"] == "${{ steps.pull.outputs.base_sha }}"
    assert checkout["with"]["persist-credentials"] is False
    command = next(
        step["run"]
        for step in steps
        if step.get("name") == "Revalidate evidence and merge against live RuleSpec tip"
    )
    assert "--require-success true" in command
    assert "verify-activation-commit" in command
    assert 'test "$(git rev-parse HEAD)" = "$BASE_SHA"' in command
    assert 'git fetch --no-tags origin "$HEAD_SHA"' in command
    assert "--current-changed-files" in command
    assert '--finalizer-jobs "$RUNNER_TEMP/finalizer-jobs.json"' in command
    assert "merge_workflow_run_attempt: 1" in command
    assert "commits/$HEAD_SHA/check-runs?per_page=100" in command
    assert "commits/$rulespec_ref/check-runs?per_page=100" in command
    assert '--previous-queue "$RUNNER_TEMP/previous-snap-queue.json"' in command
    assert '--expected-base-sha "$BASE_SHA"' in command
    assert "git/ref/heads/hard-cut/canonical-layout-us" in command
    assert '--match-head-commit "$HEAD_SHA"' in command
    assert "git log --first-parent" in command
    upload = next(
        step
        for step in steps
        if step.get("name") == "Upload trusted queue merge authorization"
    )
    assert (
        "snap-queue-merge-authorization-${{ github.run_id }}" == upload["with"]["name"]
    )
    assert upload["with"]["retention-days"] == 90


def _prepare_empty_signed_import_inputs(runner_temp: Path) -> None:
    (runner_temp / "existing-signed-imports.json").write_text("[]\n")
    (runner_temp / "existing-signed-import-paths.txt").write_text("")


@pytest.mark.parametrize("dependent_count", [0, 1, 2])
def test_targeted_signed_reencode_orders_target_and_dependents(
    tmp_path: Path,
    dependent_count: int,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Encode, review, validate, and apply"
    )
    command = command.replace(
        "/opt/axiom-verification/axiom-encode-apply-signer run",
        '"$SIGNER_STUB"',
    )

    calls_path = tmp_path / "calls.jsonl"
    signer_stub = tmp_path / "signer-stub"
    signer_stub.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

with Path(os.environ["CALLS_PATH"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
"""
    )
    signer_stub.chmod(0o700)
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    _prepare_empty_signed_import_inputs(runner_temp)

    environment = {
        **os.environ,
        "AXIOM_ENCODE_APPLY_SIGNING_KEY": "test-key",
        "AXIOM_TEST_PYTHON": sys.executable,
        "CALLS_PATH": str(calls_path),
        "CITATION": "us/regulation/42/435/555",
        "DEPENDENT_CITATION": "",
        "DEPENDENT_REVIEW_FINDING": "",
        "GITHUB_WORKSPACE": str(tmp_path),
        "REVIEW_FINDING": "Preserve the target source.",
        "RULESPEC_CHECKOUT": str(tmp_path / "rulespec-us"),
        "RULESPEC_REF": "a" * 40,
        "RUNNER_TEMP": str(runner_temp),
        "SECOND_DEPENDENT_CITATION": "",
        "SECOND_DEPENDENT_REVIEW_FINDING": "",
        "SIGNER_STUB": str(signer_stub),
        "SOURCE_BUNDLE_JSON": "[]",
    }
    if dependent_count >= 1:
        environment.update(
            {
                "DEPENDENT_CITATION": "us/regulation/42/435/559",
                "DEPENDENT_REVIEW_FINDING": "Preserve the dependent source.",
            }
        )
    if dependent_count == 2:
        environment.update(
            {
                "SECOND_DEPENDENT_CITATION": "us/regulation/42/435/561",
                "SECOND_DEPENDENT_REVIEW_FINDING": (
                    "Preserve the second dependent source."
                ),
            }
        )

    subprocess.run(
        ["bash", "-c", command],
        check=True,
        env=environment,
    )

    calls = [
        json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == dependent_count + 1
    encode_args = [call[call.index("--") + 1 :] for call in calls]
    assert all(
        args.count("--require-complete-source-unit") == 1 for args in encode_args
    )
    assert encode_args[0][-1] == "us/regulation/42/435/555"
    assert ("--apply-target-only" in encode_args[0]) is (dependent_count > 0)
    assert (
        Path(encode_args[0][encode_args[0].index("--review-findings") + 1])
        .read_text(encoding="utf-8")
        .strip()
        == "Preserve the target source."
    )
    if dependent_count >= 1:
        assert encode_args[1][-1] == "us/regulation/42/435/559"
        assert ("--apply-target-only" in encode_args[1]) is (dependent_count == 2)
        assert (
            Path(encode_args[1][encode_args[1].index("--review-findings") + 1])
            .read_text(encoding="utf-8")
            .strip()
            == "Preserve the dependent source."
        )
    if dependent_count == 2:
        assert encode_args[2][-1] == "us/regulation/42/435/561"
        assert "--apply-target-only" not in encode_args[2]
        assert (
            Path(encode_args[2][encode_args[2].index("--review-findings") + 1])
            .read_text(encoding="utf-8")
            .strip()
            == "Preserve the second dependent source."
        )


def test_targeted_signed_reencode_composes_nonempty_source_bundle(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Encode, review, validate, and apply"
    ).replace(
        "/opt/axiom-verification/axiom-encode-apply-signer run",
        '"$SIGNER_STUB"',
    )
    before_checkpoint, checkpoint_and_after = command.split(
        "checkpoint_signed_changes() {",
        1,
    )
    _checkpoint_body, after_checkpoint = checkpoint_and_after.split(
        '\n}\n\nfinal_legacy_source_path=',
        1,
    )
    command = (
        before_checkpoint
        + "checkpoint_signed_changes() {\n"
        + '  printf \'%s\\n\' "$1" >> "$CHECKPOINTS_PATH"\n'
        + '  : > "$RUNNER_TEMP/checkpoint-guard-generated.json"\n'
        + '}\n\nfinal_legacy_source_path='
        + after_checkpoint
    )

    calls_path = tmp_path / "calls.jsonl"
    checkpoints_path = tmp_path / "checkpoints.txt"
    signer_stub = tmp_path / "signer-stub"
    signer_stub.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[sys.argv.index("--") + 1:]
checkout = Path(os.environ["RULESPEC_CHECKOUT"])
legacy = checkout / os.environ["LEGACY_REPLACEMENT_FILE"]
legacy_test = legacy.with_name(f"{legacy.stem}.test.yaml")
if "--replace-legacy-rulespec-path" in args:
    if "--apply-target-only" in args:
        raise SystemExit("legacy preflight must validate the complete checkout")
    if not legacy.is_file() or not legacy_test.is_file():
        raise SystemExit("legacy predecessor was not present for preflight")
    legacy.unlink()
    legacy_test.unlink()
elif legacy.exists() or legacy_test.exists():
    raise SystemExit("source encoding started before legacy predecessor retirement")

with Path(os.environ["CALLS_PATH"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
""",
        encoding="utf-8",
    )
    signer_stub.chmod(0o700)
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    _prepare_empty_signed_import_inputs(runner_temp)
    primary = "us-ri/statute/44-30-2.6"
    replacement_path = "us-ri/statutes/44-30-2.6.yaml"
    legacy_replacement_path = "us-ri/statutes/44:30-2.6.yaml"
    checkout = tmp_path / "rulespec-us"
    canonical = checkout / replacement_path
    legacy = checkout / legacy_replacement_path
    canonical.parent.mkdir(parents=True)
    canonical.write_text("format: rulespec/v1\nrules: []\n")
    canonical.with_name(f"{canonical.stem}.test.yaml").write_text("[]\n")
    legacy.write_text("format: rulespec/v1\nrules: []\n")
    legacy.with_name(f"{legacy.stem}.test.yaml").write_text("[]\n")
    sources = [
        "us-ri/statute/44-30-1",
        "us-ri/guidance/revenue/2026/rate-schedule",
    ]

    subprocess.run(
        ["bash", "-c", command],
        check=True,
        env={
            **os.environ,
            "AXIOM_ENCODE_APPLY_SIGNING_KEY": "test-key",
            "AXIOM_TEST_PYTHON": sys.executable,
            "CALLS_PATH": str(calls_path),
            "CHECKPOINTS_PATH": str(checkpoints_path),
            "CITATION": primary,
            "DEPENDENT_CITATION": "",
            "DEPENDENT_REVIEW_FINDING": "",
            "GITHUB_WORKSPACE": str(tmp_path),
            "LEGACY_REPLACEMENT_FILE": legacy_replacement_path,
            "REPLACE_LEGACY_RULESPEC_PATH": legacy_replacement_path,
            "REPLACE_RULESPEC_PATH": replacement_path,
            "REVIEW_FINDING": "Preserve the composed target semantics.",
            "RULESPEC_CHECKOUT": str(tmp_path / "rulespec-us"),
            "RULESPEC_REF": "a" * 40,
            "RUNNER_TEMP": str(runner_temp),
            "SECOND_DEPENDENT_CITATION": "",
            "SECOND_DEPENDENT_REVIEW_FINDING": "",
            "SIGNER_STUB": str(signer_stub),
            "SOURCE_BUNDLE_JSON": json.dumps(sources),
        },
    )

    calls = [
        json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == 4
    encode_args = [call[call.index("--") + 1 :] for call in calls]
    preflight_args = encode_args[0]
    assert preflight_args[-1] == primary
    assert "--apply-target-only" not in preflight_args
    assert "--review-findings" not in preflight_args
    replacement_index = preflight_args.index("--replace-rulespec-path")
    assert preflight_args[replacement_index + 1] == replacement_path
    legacy_index = preflight_args.index("--replace-legacy-rulespec-path")
    assert preflight_args[legacy_index + 1] == legacy_replacement_path
    assert "--required-import-rulespec-path" not in preflight_args

    for index, source in enumerate(sources, start=1):
        assert encode_args[index][-1] == source
        assert "--apply-target-only" in encode_args[index]
        assert "--required-import-rulespec-path" not in encode_args[index]

    primary_args = encode_args[-1]
    assert primary_args[-1] == primary
    assert "--apply-target-only" not in primary_args
    assert "--replace-legacy-rulespec-path" not in primary_args
    required_indexes = [
        index
        for index, value in enumerate(primary_args)
        if value == "--required-import-rulespec-path"
    ]
    assert [primary_args[index + 1] for index in required_indexes] == [
        "us-ri/statutes/44-30-1.yaml",
        "us-ri/policies/revenue/2026/rate-schedule.yaml",
    ]
    assert checkpoints_path.read_text(encoding="utf-8").splitlines() == [
        "Canonicalize signed replacement target before source bundle",
        f"Add signed source module for {sources[0]}",
        f"Add signed source module for {sources[1]}",
        f"Compose signed source bundle for {primary}",
    ]
    assert canonical.is_file()
    assert not legacy.exists()
    assert not legacy.with_name(f"{legacy.stem}.test.yaml").exists()


def test_targeted_signed_reencode_rejects_source_bundle_replacement_dependents_early(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Validate atomic source inputs"
    )
    checkout = tmp_path / "rulespec-us"
    checkout.mkdir()

    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CITATION": "us-ri/statute/44-30-2.6",
            "DEPENDENT_CITATION": "us-ri/statute/44-30-2.7",
            "EXISTING_SIGNED_IMPORTS_JSON": "[]",
            "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
            "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": "[]",
            "QUEUE_ID": "",
            "REPLACE_LEGACY_RULESPEC_PATH": "us-ri/statutes/44:30-2.6.yaml",
            "REPLACE_RULESPEC_PATH": "us-ri/statutes/44-30-2.6.yaml",
            "RULESPEC_CHECKOUT": str(checkout),
            "SECOND_DEPENDENT_CITATION": "",
            "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
            "SOURCE_BUNDLE_JSON": '["us-ri/statute/44-30-1"]',
        },
    )

    assert completed.returncode != 0
    assert (
        "source-bundle replacements cannot include dependent migrations"
        in completed.stderr
    )


def test_targeted_signed_reencode_reuses_verified_existing_import(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Encode, review, validate, and apply"
    ).replace(
        "/opt/axiom-verification/axiom-encode-apply-signer run",
        '"$SIGNER_STUB"',
    )

    calls_path = tmp_path / "calls.jsonl"
    signer_stub = tmp_path / "signer-stub"
    signer_stub.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

with Path(os.environ["CALLS_PATH"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
""",
        encoding="utf-8",
    )
    signer_stub.chmod(0o700)

    rulespec_repo = tmp_path / "rulespec-us"
    existing_path = "us-ri/statutes/44-30-1.yaml"
    manifest_path = (
        rulespec_repo / ".axiom/encoding-manifests/us-ri/statutes/44-30-1.json"
    )
    rule = rulespec_repo / existing_path
    rule.parent.mkdir(parents=True)
    rule.write_text("format: rulespec/v1\nrules: []\n")
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"schema_version": "axiom-encode/applied-rulespec/v5"}) + "\n"
    )
    subprocess.run(["git", "-C", str(rulespec_repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(rulespec_repo), "add", "."], check=True)

    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    normalized_existing = json.dumps([existing_path], separators=(",", ":"))
    (runner_temp / "existing-signed-imports.json").write_text(
        normalized_existing + "\n"
    )
    (runner_temp / "existing-signed-import-paths.txt").write_text(existing_path + "\n")

    subprocess.run(
        ["bash", "-c", command],
        check=True,
        env={
            **os.environ,
            "AXIOM_ENCODE_APPLY_SIGNING_KEY": "test-key",
            "AXIOM_TEST_PYTHON": sys.executable,
            "CALLS_PATH": str(calls_path),
            "CITATION": "us-ri/statute/44-30-2.6",
            "DEPENDENT_CITATION": "",
            "DEPENDENT_REVIEW_FINDING": "",
            "EXISTING_SIGNED_IMPORTS_JSON": normalized_existing,
            "GITHUB_WORKSPACE": str(tmp_path),
            "REVIEW_FINDING": "Preserve direct composition semantics.",
            "RULESPEC_CHECKOUT": str(rulespec_repo),
            "RULESPEC_REF": "a" * 40,
            "RUNNER_TEMP": str(runner_temp),
            "SECOND_DEPENDENT_CITATION": "",
            "SECOND_DEPENDENT_REVIEW_FINDING": "",
            "SIGNER_STUB": str(signer_stub),
            "SOURCE_BUNDLE_JSON": "[]",
        },
    )

    calls = [
        json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == 1
    encode_args = calls[0][calls[0].index("--") + 1 :]
    assert encode_args[-1] == "us-ri/statute/44-30-2.6"
    required_index = encode_args.index("--required-import-rulespec-path")
    assert encode_args[required_index + 1] == existing_path


@pytest.mark.parametrize("with_dependent", [False, True])
def test_targeted_signed_reencode_runs_replacement_target(
    tmp_path: Path,
    with_dependent: bool,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Encode, review, validate, and apply"
    ).replace(
        "/opt/axiom-verification/axiom-encode-apply-signer run",
        '"$SIGNER_STUB"',
    )
    calls_path = tmp_path / "calls.jsonl"
    signer_stub = tmp_path / "signer-stub"
    signer_stub.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

with Path(os.environ["CALLS_PATH"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
"""
    )
    signer_stub.chmod(0o700)
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    _prepare_empty_signed_import_inputs(runner_temp)
    replacement_path = "us-nc/policies/income_tax/pilot_liability_pipeline.yaml"
    dependent_citation = "us-nc/statute/105/105-153.5" if with_dependent else ""
    dependent_finding = (
        "Preserve the resident pipeline semantics." if with_dependent else ""
    )

    environment = {
        **os.environ,
        "AXIOM_ENCODE_APPLY_SIGNING_KEY": "test-key",
        "AXIOM_TEST_PYTHON": sys.executable,
        "CALLS_PATH": str(calls_path),
        "CITATION": "us-nc/statute/105/105-153.7",
        "DEPENDENT_CITATION": dependent_citation,
        "DEPENDENT_REVIEW_FINDING": dependent_finding,
        "GITHUB_WORKSPACE": str(tmp_path),
        "REPLACE_LEGACY_RULESPEC_PATH": (
            "us-nc/policies/income_tax/PILOT_LIABILITY_PIPELINE.yaml"
            if with_dependent
            else ""
        ),
        "REPLACE_RULESPEC_PATH": replacement_path,
        "REVIEW_FINDING": "Preserve all supported existing semantics.",
        "RULESPEC_CHECKOUT": str(tmp_path / "rulespec-us"),
        "RULESPEC_REF": "a" * 40,
        "RUNNER_TEMP": str(runner_temp),
        "SECOND_DEPENDENT_CITATION": "",
        "SECOND_DEPENDENT_REVIEW_FINDING": "",
        "SIGNER_STUB": str(signer_stub),
        "SOURCE_BUNDLE_JSON": "[]",
    }
    subprocess.run(
        ["bash", "-c", command],
        check=True,
        env=environment,
    )

    calls = [
        json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == (2 if with_dependent else 1)
    encode_args = [call[call.index("--") + 1 :] for call in calls]
    assert ("--apply-target-only" in encode_args[0]) is with_dependent
    assert encode_args[0][encode_args[0].index("--replace-rulespec-path") + 1] == (
        replacement_path
    )
    assert encode_args[0][-1] == "us-nc/statute/105/105-153.7"
    if with_dependent:
        assert "--replace-legacy-rulespec-path" in encode_args[0]
        assert "--legacy-dependent-rulespec-path" in encode_args[0]
        assert "--replace-rulespec-path" not in encode_args[1]
        assert "--apply-target-only" not in encode_args[1]
        assert encode_args[1][-1] == dependent_citation

    blocked = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env={**environment, "QUEUE_ID": "us-snap-all-states-2026-07"},
    )
    assert blocked.returncode != 0
    assert (
        "queue-authorized re-encodes cannot override the RuleSpec target path"
        in blocked.stderr
    )


def test_targeted_signed_reencode_passes_exact_legacy_dependents_atomically(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Encode, review, validate, and apply"
    ).replace(
        "/opt/axiom-verification/axiom-encode-apply-signer run",
        '"$SIGNER_STUB"',
    )
    calls_path = tmp_path / "calls.jsonl"
    signer_stub = tmp_path / "signer-stub"
    signer_stub.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

with Path(os.environ["CALLS_PATH"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
"""
    )
    signer_stub.chmod(0o700)
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    _prepare_empty_signed_import_inputs(runner_temp)
    source = "us-la/statutes/47:32.yaml"
    destination = "us-la/statutes/47/32.yaml"
    exact_dependents = [
        "us-la/policies/income_tax/2026_resident_core.yaml",
        "us-la/policies/income_tax/pilot_liability_pipeline.yaml",
    ]
    retained_successors = [
        "us-la/statutes/47:294.yaml",
        "us-la/statutes/47:295.yaml",
        "us-la/statutes/47:297/4.yaml",
        "us-la/statutes/47:297/8.yaml",
    ]
    environment = {
        **os.environ,
        "AXIOM_ENCODE_APPLY_SIGNING_KEY": "test-key",
        "AXIOM_TEST_PYTHON": sys.executable,
        "CALLS_PATH": str(calls_path),
        "CITATION": "us-la/statute/47:32",
        "DEPENDENT_CITATION": "",
        "DEPENDENT_REVIEW_FINDING": "",
        "GITHUB_WORKSPACE": str(tmp_path),
        "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": exact_dependents[0],
        "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": json.dumps(
            retained_successors
        ),
        "REPLACE_LEGACY_RULESPEC_PATH": source,
        "REPLACE_RULESPEC_PATH": destination,
        "REVIEW_FINDING": "Preserve all supported existing semantics.",
        "RULESPEC_CHECKOUT": str(tmp_path / "rulespec-us"),
        "RULESPEC_REF": "a" * 40,
        "RUNNER_TEMP": str(runner_temp),
        "SECOND_DEPENDENT_CITATION": "",
        "SECOND_DEPENDENT_REVIEW_FINDING": "",
        "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": exact_dependents[1],
        "SIGNER_STUB": str(signer_stub),
        "SOURCE_BUNDLE_JSON": "[]",
    }

    subprocess.run(["bash", "-c", command], check=True, env=environment)

    calls = [
        json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == 1
    encode_args = calls[0][calls[0].index("--") + 1 :]
    assert encode_args[encode_args.index("--replace-rulespec-path") + 1] == destination
    assert (
        encode_args[encode_args.index("--replace-legacy-rulespec-path") + 1] == source
    )
    exact_indexes = [
        index
        for index, value in enumerate(encode_args)
        if value == "--legacy-exact-dependent-rulespec-path"
    ]
    assert [encode_args[index + 1] for index in exact_indexes] == exact_dependents
    retained_indexes = [
        index
        for index, value in enumerate(encode_args)
        if value == "--legacy-retained-successor-rulespec-path"
    ]
    assert [encode_args[index + 1] for index in retained_indexes] == (
        retained_successors
    )
    assert "--apply-target-only" not in encode_args


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (
            {
                "SECOND_DEPENDENT_CITATION": "us/regulation/42/435/561",
                "SECOND_DEPENDENT_REVIEW_FINDING": "Preserve the second source.",
            },
            "first dependent citation is required before a second dependent",
        ),
        (
            {
                "DEPENDENT_CITATION": "us/regulation/42/435/559",
                "DEPENDENT_REVIEW_FINDING": "Preserve the dependent source.",
                "SECOND_DEPENDENT_CITATION": "us/regulation/42/435/559",
                "SECOND_DEPENDENT_REVIEW_FINDING": "Preserve the second source.",
            },
            "second dependent citation must be unique",
        ),
        (
            {
                "DEPENDENT_CITATION": "us/regulation/42/435/559",
                "DEPENDENT_REVIEW_FINDING": "Preserve the dependent source.",
                "SECOND_DEPENDENT_REVIEW_FINDING": "Preserve the second source.",
            },
            "second dependent citation is required with its review finding",
        ),
        (
            {
                "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": (
                    "us-la/policies/income_tax/pilot_liability_pipeline.yaml"
                ),
            },
            "exact legacy dependents require a legacy source replacement",
        ),
        (
            {
                "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": (
                    "us-la/policies/income_tax/2026_resident_core.yaml"
                ),
                "REPLACE_LEGACY_RULESPEC_PATH": "us-la/statutes/47:32.yaml",
                "REPLACE_RULESPEC_PATH": "us-la/statutes/47/32.yaml",
                "DEPENDENT_CITATION": "us-la/statute/47:294",
                "DEPENDENT_REVIEW_FINDING": "Preserve dependent semantics.",
            },
            "exact and model-regenerated dependent modes cannot be combined",
        ),
    ],
)
def test_targeted_signed_reencode_rejects_invalid_second_dependent_inputs(
    tmp_path: Path,
    overrides: dict[str, str],
    error: str,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Encode, review, validate, and apply"
    )
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    _prepare_empty_signed_import_inputs(runner_temp)
    environment = {
        **os.environ,
        "AXIOM_ENCODE_APPLY_SIGNING_KEY": "test-key",
        "CITATION": "us/regulation/42/435/555",
        "DEPENDENT_CITATION": "",
        "DEPENDENT_REVIEW_FINDING": "",
        "GITHUB_WORKSPACE": str(tmp_path),
        "REVIEW_FINDING": "Preserve the target source.",
        "RULESPEC_CHECKOUT": str(tmp_path / "rulespec-us"),
        "RULESPEC_REF": "a" * 40,
        "RUNNER_TEMP": str(runner_temp),
        "SECOND_DEPENDENT_CITATION": "",
        "SECOND_DEPENDENT_REVIEW_FINDING": "",
        "SOURCE_BUNDLE_JSON": "[]",
        **overrides,
    }

    completed = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert error in completed.stderr


def test_targeted_review_finding_temp_file_is_valid_context(tmp_path: Path) -> None:
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    _prepare_empty_signed_import_inputs(runner_temp)
    policy_root = tmp_path / "rulespec-us" / "us"
    policy_root.mkdir(parents=True)
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'path="$(mktemp "$RUNNER_TEMP/'
            'axiom-targeted-review-finding.XXXXXX.txt")"; '
            "printf '%s\\n' 'Preserve the supported provision.' > \"$path\"; "
            "printf '%s' \"$path\"",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "RUNNER_TEMP": str(runner_temp)},
    )

    finding_path = Path(completed.stdout)
    assert finding_path.parent == runner_temp
    assert finding_path.suffix == ".txt"
    assert validate_explicit_context_file(finding_path, policy_root) == finding_path


def test_targeted_artifact_packages_signed_review_context(tmp_path: Path) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    package_command = next(
        step
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Package exact generated changes"
    )["run"]
    marker = (
        '"${workflow_python[@]}" - \\\n'
        '  "$artifact/context-manifest.json" \\\n'
        "  \"$artifact/apply-manifests.json\" <<'PY'\n"
    )
    script = package_command.split(marker, 1)[1].split(
        '\nPY\n"${workflow_python[@]}" - "$artifact/metadata.json"', 1
    )[0]

    rulespec = tmp_path / "rulespec-nz"
    rulespec.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=rulespec, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=rulespec,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=rulespec, check=True)
    base = rulespec / "README.md"
    base.write_text("fixture\n")
    subprocess.run(["git", "add", "README.md"], cwd=rulespec, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=rulespec, check=True)
    rulespec_ref = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=rulespec, text=True
    ).strip()
    (tmp_path / "source-bundle.json").write_text("[]\n", encoding="utf-8")

    citation = "us-la/statute/47:294"
    review_content = "Preserve every supported provision.\n"
    context_payload = {
        "citation": citation,
        "review_findings_files": [
            {
                "content": review_content,
                "sha256": hashlib.sha256(review_content.encode()).hexdigest(),
            }
        ],
    }
    context_bytes = json.dumps(context_payload, sort_keys=True).encode()
    context_path = tmp_path / "generated" / "target" / "context-manifest.json"
    context_path.parent.mkdir(parents=True)
    context_path.write_bytes(context_bytes)
    applied_manifest = {
        "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
        "citation": citation,
        "context_manifest_file": str(context_path),
        "context_manifest_sha256": hashlib.sha256(context_bytes).hexdigest(),
    }
    applied_path = (
        rulespec / ".axiom" / "encoding-manifests" / "statutes" / "47" / "294.yaml.json"
    )
    applied_path.parent.mkdir(parents=True)
    applied_path.write_text(json.dumps(applied_manifest))

    packaged_context = tmp_path / "artifact" / "context-manifest.json"
    packaged_inventory = tmp_path / "artifact" / "apply-manifests.json"
    packaged_context.parent.mkdir()
    completed = subprocess.run(
        [sys.executable, "-", str(packaged_context), str(packaged_inventory)],
        cwd=tmp_path,
        input=script,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CITATION": citation,
            "PYTHONPATH": str(ROOT / "src"),
            "REVIEW_FINDING_PRESENT": "true",
            "RUNNER_TEMP": str(tmp_path),
            "RULESPEC_CHECKOUT": "rulespec-nz",
            "RULESPEC_REF": rulespec_ref,
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert packaged_context.read_bytes() == context_bytes
    inventory = json.loads(packaged_inventory.read_text())
    assert inventory["schema"] == "axiom-encode/applied-manifest-inventory/v1"
    assert inventory["items"] == [
        {
            "citation": citation,
            "path": applied_path.relative_to(rulespec).as_posix(),
            "sha256": hashlib.sha256(applied_path.read_bytes()).hexdigest(),
        }
    ]


def _targeted_package_script() -> str:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    package_command = next(
        step
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Package exact generated changes"
    )["run"]
    marker = (
        '"${workflow_python[@]}" - \\\n'
        '  "$artifact/context-manifest.json" \\\n'
        "  \"$artifact/apply-manifests.json\" <<'PY'\n"
    )
    return package_command.split(marker, 1)[1].split(
        '\nPY\n"${workflow_python[@]}" - "$artifact/metadata.json"', 1
    )[0]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("success", None),
        ("missing-input", "retained successors differ from inputs"),
        ("extra-input", "retained successors differ from inputs"),
        ("reordered-input", "retained successors differ from inputs"),
        ("tampered-evidence", "retained predecessor file digest differs"),
        ("missing-deleted-manifest", "deleted manifest inventory differs"),
        ("extra-deleted-manifest", "deleted manifest inventory differs"),
    ],
)
def test_targeted_artifact_packages_v4_retained_successor_closure(
    tmp_path: Path,
    mutation: str,
    error: str | None,
) -> None:
    script = _targeted_package_script()
    rulespec = tmp_path / "rulespec-us"
    rulespec.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=rulespec, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=rulespec,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=rulespec, check=True)

    def write(path: str, raw: bytes) -> Path:
        target = rulespec / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return target

    def evidence(path: Path) -> dict[str, str]:
        return {
            "path": path.relative_to(rulespec).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    main_old = write("us/statutes/47:32.yaml", b"old main\n")
    main_old_manifest = write(
        ".axiom/encoding-manifests/us/statutes/47:32.json", b"old main manifest\n"
    )
    unrelated_manifest = write(
        ".axiom/encoding-manifests/us/statutes/unrelated.json", b"unrelated\n"
    )
    retained_rows: list[dict[str, object]] = []
    retained_sources = [
        "us/statutes/47:294.yaml",
        "us/statutes/47:295.yaml",
    ]
    for number in (294, 295):
        old = write(f"us/statutes/47:{number}.yaml", f"old {number}\n".encode())
        old_test = write(f"us/statutes/47:{number}.test.yaml", b"[]\n")
        old_manifest = write(
            f".axiom/encoding-manifests/us/statutes/47:{number}.json",
            f"old manifest {number}\n".encode(),
        )
        successor = write(f"us/statutes/47/{number}.yaml", old.read_bytes())
        successor_test = write(
            f"us/statutes/47/{number}.test.yaml", old_test.read_bytes()
        )
        successor_files = [evidence(successor), evidence(successor_test)]
        original_payload = {
            "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
            "tool": "axiom-encode encode --apply",
            "backend": "codex",
            "citation": f"us/statute/47:{number}",
            "applied_files": successor_files,
        }
        original_raw = (
            json.dumps(original_payload, indent=2, sort_keys=True) + "\n"
        ).encode()
        successor_manifest = write(
            f".axiom/encoding-manifests/us/statutes/47/{number}.json",
            original_raw,
        )
        retained_rows.append(
            {
                "source": old.relative_to(rulespec).as_posix(),
                "destination": successor.relative_to(rulespec).as_posix(),
                "legacy_owner_class": "v1-manual-hmac-untrusted",
                "legacy_manifest": evidence(old_manifest),
                "legacy_files": [evidence(old), evidence(old_test)],
                "successor_manifest": {
                    **evidence(successor_manifest),
                    "payload": original_payload,
                },
                "successor_files": successor_files,
            }
        )
    subprocess.run(["git", "add", "."], cwd=rulespec, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=rulespec, check=True)
    rulespec_ref = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=rulespec, text=True
    ).strip()

    main_old.unlink()
    main_old_manifest.unlink()
    for row in retained_rows:
        for item in row["legacy_files"]:
            (rulespec / item["path"]).unlink()
        (rulespec / row["legacy_manifest"]["path"]).unlink()
    main_live = write("us/statutes/47/32.yaml", b"new main\n")
    citation = "us/statute/47:32"
    review_content = "Retain the canonical successors.\n"
    context_payload = {
        "citation": citation,
        "review_findings_files": [
            {
                "content": review_content,
                "sha256": hashlib.sha256(review_content.encode()).hexdigest(),
            }
        ],
    }
    context_bytes = json.dumps(context_payload, sort_keys=True).encode()
    context_path = tmp_path / "generated/target/context-manifest.json"
    context_path.parent.mkdir(parents=True)
    context_path.write_bytes(context_bytes)
    nested = {
        "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
        "tool": "axiom-encode encode --apply",
        "backend": "codex",
        "citation": citation,
        "context_manifest_file": str(context_path),
        "context_manifest_sha256": hashlib.sha256(context_bytes).hexdigest(),
        "applied_files": [evidence(main_live)],
    }
    receipt_payload = {
        "schema_version": "axiom-encode/legacy-fresh-reencode-receipt/v4",
        "replacement": {
            "source": "us/statutes/47:32.yaml",
            "destination": "us/statutes/47/32.yaml",
            "scheduled_dependents": [],
            "exact_dependents": [],
            "retained_successors": retained_rows,
            "metadata_reconciliations": [],
        },
    }
    receipt_path = rulespec / ".axiom/legacy-replacements" / f"{'a' * 64}.json"
    receipt_path.parent.mkdir(parents=True)

    if mutation == "tampered-evidence":
        retained_rows[0]["legacy_files"][0]["sha256"] = "0" * 64
    if mutation == "missing-deleted-manifest":
        first_manifest = retained_rows[0]["legacy_manifest"]
        (rulespec / first_manifest["path"]).write_text("old manifest 294\n")
    if mutation == "extra-deleted-manifest":
        unrelated_manifest.unlink()

    receipt_path.write_text(json.dumps(receipt_payload, sort_keys=True) + "\n")
    receipt_digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    for row in retained_rows:
        refreshed = {
            "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
            "tool": (
                "axiom-encode encode --apply --legacy-retained-successor-rulespec-path"
            ),
            "applied_files": row["successor_files"],
            "retained_successor_manifest": row["successor_manifest"]["payload"],
            "legacy_migration": {
                "receipt_path": receipt_path.relative_to(rulespec).as_posix(),
                "receipt_sha256": receipt_digest,
                "source": row["source"],
                "destination": row["destination"],
                "legacy_manifest_path": row["legacy_manifest"]["path"],
                "legacy_manifest_sha256": row["legacy_manifest"]["sha256"],
                "successor_manifest_sha256": row["successor_manifest"]["sha256"],
            },
        }
        (rulespec / row["successor_manifest"]["path"]).write_text(
            json.dumps(refreshed, sort_keys=True) + "\n"
        )
    outer = {
        "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
        "tool": "axiom-encode encode --apply --replace-legacy-rulespec-path",
        "replacement_manifest": nested,
        "replacement": {
            "receipt_path": receipt_path.relative_to(rulespec).as_posix(),
            "receipt_sha256": receipt_digest,
        },
    }
    outer_path = write(
        ".axiom/encoding-manifests/us/statutes/47/32.json",
        (json.dumps(outer, sort_keys=True) + "\n").encode(),
    )
    (tmp_path / "source-bundle.json").write_text("[]\n")
    selected_inputs = list(retained_sources)
    if mutation == "missing-input":
        selected_inputs.pop()
    elif mutation == "extra-input":
        selected_inputs.append("us/statutes/47:296.yaml")
    elif mutation == "reordered-input":
        selected_inputs.reverse()
    packaged_context = tmp_path / "artifact/context-manifest.json"
    packaged_inventory = tmp_path / "artifact/apply-manifests.json"
    packaged_context.parent.mkdir()
    completed = subprocess.run(
        [sys.executable, "-", str(packaged_context), str(packaged_inventory)],
        cwd=tmp_path,
        input=script,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CITATION": citation,
            "PYTHONPATH": str(ROOT / "src"),
            "REVIEW_FINDING_PRESENT": "true",
            "RUNNER_TEMP": str(tmp_path),
            "RULESPEC_CHECKOUT": "rulespec-us",
            "RULESPEC_REF": rulespec_ref,
            "REPLACE_LEGACY_RULESPEC_PATH": "us/statutes/47:32.yaml",
            "REPLACE_RULESPEC_PATH": "us/statutes/47/32.yaml",
            "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": json.dumps(
                selected_inputs
            ),
        },
    )

    if error is not None:
        assert completed.returncode != 0
        assert error in completed.stderr
        return
    assert completed.returncode == 0, completed.stderr
    inventory_paths = {
        item["path"] for item in json.loads(packaged_inventory.read_text())["items"]
    }
    assert outer_path.relative_to(rulespec).as_posix() in inventory_paths
    assert {str(row["successor_manifest"]["path"]) for row in retained_rows}.issubset(
        inventory_paths
    )


@pytest.mark.parametrize("receipt_version", [1, 2, 3])
def test_targeted_artifact_preserves_pre_v4_replacement_receipts(
    tmp_path: Path,
    receipt_version: int,
) -> None:
    script = _targeted_package_script()
    rulespec = tmp_path / "rulespec-us"
    rulespec.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=rulespec, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=rulespec,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=rulespec, check=True)
    old = rulespec / "us/statutes/47:32.yaml"
    old_manifest = rulespec / ".axiom/encoding-manifests/us/statutes/47:32.json"
    old.parent.mkdir(parents=True)
    old_manifest.parent.mkdir(parents=True)
    old.write_text("old\n")
    old_manifest.write_text("old manifest\n")
    subprocess.run(["git", "add", "."], cwd=rulespec, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=rulespec, check=True)
    rulespec_ref = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=rulespec, text=True
    ).strip()
    old.unlink()
    old_manifest.unlink()
    live = rulespec / "us/statutes/47/32.yaml"
    live.parent.mkdir(parents=True)
    live.write_text("new\n")
    citation = "us/statute/47:32"
    context_payload = {"citation": citation, "review_findings_files": []}
    context_bytes = json.dumps(context_payload, sort_keys=True).encode()
    context_path = tmp_path / "generated/target/context-manifest.json"
    context_path.parent.mkdir(parents=True)
    context_path.write_bytes(context_bytes)
    nested = {
        "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
        "tool": "axiom-encode encode --apply",
        "backend": "codex",
        "citation": citation,
        "context_manifest_file": str(context_path),
        "context_manifest_sha256": hashlib.sha256(context_bytes).hexdigest(),
        "applied_files": [],
    }
    receipt_replacement = {
        "source": "us/statutes/47:32.yaml",
        "destination": "us/statutes/47/32.yaml",
        "scheduled_dependents": [],
    }
    if receipt_version >= 2:
        receipt_replacement["exact_dependents"] = []
    if receipt_version >= 3:
        receipt_replacement["destination_predecessor_class"] = (
            "canonicalized-unowned-duplicate"
        )
        receipt_replacement["destination_predecessor_files"] = []
    receipt_payload = {
        "schema_version": (
            f"axiom-encode/legacy-fresh-reencode-receipt/v{receipt_version}"
        ),
        "replacement": receipt_replacement,
    }
    receipt_path = rulespec / ".axiom/legacy-replacements" / f"{'b' * 64}.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt_payload) + "\n")
    outer = {
        "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
        "tool": "axiom-encode encode --apply --replace-legacy-rulespec-path",
        "replacement_manifest": nested,
        "replacement": {
            "receipt_path": receipt_path.relative_to(rulespec).as_posix(),
            "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        },
    }
    outer_path = rulespec / ".axiom/encoding-manifests/us/statutes/47/32.json"
    outer_path.parent.mkdir(parents=True, exist_ok=True)
    outer_path.write_text(json.dumps(outer) + "\n")
    (tmp_path / "source-bundle.json").write_text("[]\n")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            "-",
            str(artifact / "context-manifest.json"),
            str(artifact / "apply-manifests.json"),
        ],
        cwd=tmp_path,
        input=script,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CITATION": citation,
            "PYTHONPATH": str(ROOT / "src"),
            "REVIEW_FINDING_PRESENT": "false",
            "RUNNER_TEMP": str(tmp_path),
            "RULESPEC_CHECKOUT": "rulespec-us",
            "RULESPEC_REF": rulespec_ref,
            "REPLACE_LEGACY_RULESPEC_PATH": "us/statutes/47:32.yaml",
            "REPLACE_RULESPEC_PATH": "us/statutes/47/32.yaml",
        },
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("target_lane", "dependent_lane", "dependent_context_citation", "error"),
    [
        ("target", "dependent", None, None),
        (
            "dependent",
            "target",
            None,
            "signed context manifest is outside assigned target lane",
        ),
        (
            "target",
            "target",
            None,
            "signed context manifest is outside assigned dependent lane",
        ),
        (
            "target",
            "dependent",
            "us/regulation/42/435/555",
            "signed context manifest citation does not match",
        ),
    ],
)
def test_targeted_artifact_enforces_target_and_dependent_context_lanes(
    tmp_path: Path,
    target_lane: str,
    dependent_lane: str,
    dependent_context_citation: str | None,
    error: str | None,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    package_command = next(
        step
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Package exact generated changes"
    )["run"]
    marker = (
        '"${workflow_python[@]}" - \\\n'
        '  "$artifact/context-manifest.json" \\\n'
        "  \"$artifact/apply-manifests.json\" <<'PY'\n"
    )
    script = package_command.split(marker, 1)[1].split(
        '\nPY\n"${workflow_python[@]}" - "$artifact/metadata.json"', 1
    )[0]

    rulespec = tmp_path / "rulespec-us"
    rulespec.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=rulespec, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=rulespec,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=rulespec, check=True)
    base = rulespec / "README.md"
    base.write_text("fixture\n")
    subprocess.run(["git", "add", "README.md"], cwd=rulespec, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=rulespec, check=True)
    rulespec_ref = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=rulespec, text=True
    ).strip()
    (tmp_path / "source-bundle.json").write_text("[]\n", encoding="utf-8")

    target_citation = "us/regulation/42/435/555"
    dependent_citation = "us/regulation/42/435/559"
    second_dependent_citation = "us/regulation/42/435/561"
    generated_root = tmp_path / "generated"
    contexts: dict[str, bytes] = {}
    for citation, context_citation, lane, section, finding in (
        (
            target_citation,
            target_citation,
            target_lane,
            "555",
            "Preserve the target source.\n",
        ),
        (
            dependent_citation,
            dependent_context_citation or dependent_citation,
            dependent_lane,
            "559",
            "Preserve the dependent source.\n",
        ),
        (
            second_dependent_citation,
            second_dependent_citation,
            "dependent-2",
            "561",
            "Preserve the second dependent source.\n",
        ),
    ):
        context_payload = {
            "citation": context_citation,
            "review_findings_files": [
                {
                    "content": finding,
                    "sha256": hashlib.sha256(finding.encode()).hexdigest(),
                }
            ],
        }
        context_bytes = json.dumps(context_payload, sort_keys=True).encode()
        context_path = (
            generated_root
            / lane
            / "_eval_workspaces"
            / section
            / "context-manifest.json"
        )
        context_path.parent.mkdir(parents=True)
        context_path.write_bytes(context_bytes)
        contexts[citation] = context_bytes

        applied_manifest = {
            "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
            "citation": citation,
            "context_manifest_file": str(context_path),
            "context_manifest_sha256": hashlib.sha256(context_bytes).hexdigest(),
        }
        applied_path = (
            rulespec
            / ".axiom"
            / "encoding-manifests"
            / "regulations"
            / "42-cfr"
            / "435"
            / f"{section}.yaml.json"
        )
        applied_path.parent.mkdir(parents=True, exist_ok=True)
        applied_path.write_text(json.dumps(applied_manifest))

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    packaged_target = artifact / "context-manifest.json"
    packaged_inventory = artifact / "apply-manifests.json"
    completed = subprocess.run(
        [sys.executable, "-", str(packaged_target), str(packaged_inventory)],
        cwd=tmp_path,
        input=script,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CITATION": target_citation,
            "DEPENDENT_CITATION": dependent_citation,
            "DEPENDENT_REVIEW_FINDING_PRESENT": "true",
            "PYTHONPATH": str(ROOT / "src"),
            "SECOND_DEPENDENT_CITATION": second_dependent_citation,
            "SECOND_DEPENDENT_REVIEW_FINDING_PRESENT": "true",
            "REVIEW_FINDING_PRESENT": "true",
            "RUNNER_TEMP": str(tmp_path),
            "RULESPEC_CHECKOUT": "rulespec-us",
            "RULESPEC_REF": rulespec_ref,
        },
    )

    if error is not None:
        assert completed.returncode != 0
        assert error in completed.stderr
        return

    assert completed.returncode == 0, completed.stderr
    assert packaged_target.read_bytes() == contexts[target_citation]
    assert (artifact / "dependent-context-manifest.json").read_bytes() == contexts[
        dependent_citation
    ]
    assert (artifact / "dependent-2-context-manifest.json").read_bytes() == contexts[
        second_dependent_citation
    ]


def test_apply_signing_key_migration_workflow_is_main_only() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/migrate-apply-signing-key.yml").read_text()
    )
    trigger = workflow.get("on", workflow.get(True))
    assert set(trigger) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}

    job = workflow["jobs"]["migrate"]
    assert job["environment"] == "signing-key-migration"
    assert "github.ref == 'refs/heads/main'" in job["if"]
    steps = job["steps"]
    assert not any(
        step.get("uses", "").startswith("actions/checkout@") for step in steps
    )
    setup_go = next(
        step for step in steps if step.get("name") == "Install pinned Go toolchain"
    )
    assert setup_go["uses"] == (
        "actions/setup-go@924ae3a1cded613372ab5595356fb5720e22ba16"
    )
    assert setup_go["with"] == {"go-version": "1.26.1", "cache": False}

    encrypt_step = next(
        step for step in steps if step.get("name") == "Encrypt inherited signing key"
    )
    assert encrypt_step["env"]["APPLY_SIGNING_KEY"] == (
        "${{ secrets.AXIOM_ENCODE_APPLY_SIGNING_KEY }}"
    )
    command = encrypt_step["run"]
    assert "rsa_padding_mode:oaep" in command
    assert "rsa_oaep_md:sha256" in command
    assert "rsa_mgf1_md:sha256" in command
    assert "unset APPLY_SIGNING_KEY" in command
    assert '"$rsa_bits" -lt 3072' in command
    assert "base64.StdEncoding.Strict().DecodeString" in command
    assert "x509.ParsePKCS8PrivateKey" in command
    assert "bytes.Equal(derivedPublic, trustedPublic)" in command
    assert "sha256sum --check SHA256SUMS" in command
    assert 'rm -f "$plaintext"' in command
    assert 'echo "$APPLY_SIGNING_KEY"' not in command
    upload_step = next(
        step
        for step in steps
        if step.get("name") == "Upload encrypted migration artifact"
    )
    assert upload_step["with"]["path"] == (
        "${{ runner.temp }}/apply-signing-key-migration"
    )
    assert upload_step["with"]["retention-days"] == 1

    secret_steps = [
        step for step in steps if "APPLY_SIGNING_KEY" in (step.get("env") or {})
    ]
    assert secret_steps == [encrypt_step]


def _migration_openssl() -> Path:
    candidates = (
        Path("/opt/homebrew/opt/openssl@3/bin/openssl"),
        Path("/usr/local/opt/openssl@3/bin/openssl"),
        Path(shutil.which("openssl") or "/missing"),
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        version = subprocess.run(
            [candidate, "version"],
            check=False,
            capture_output=True,
            text=True,
        )
        if version.returncode == 0 and version.stdout.startswith("OpenSSL 3."):
            return candidate
    pytest.skip("OpenSSL 3 is required for the executable migration workflow test")


def _run_apply_key_migration(
    tmp_path: Path,
    *,
    run_name: str,
    signing_key: str,
    apply_public_key: str,
    recipient_public_key: bytes,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    go = Path(shutil.which("go") or "/opt/homebrew/bin/go")
    if not go.is_file():
        pytest.skip("Go is required for the executable migration workflow test")
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/migrate-apply-signing-key.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["migrate"]["steps"]
        if step.get("name") == "Encrypt inherited signing key"
    )
    runner_temp = tmp_path / run_name
    runner_temp.mkdir()
    shim_dir = runner_temp / "bin"
    shim_dir.mkdir()
    (shim_dir / "go").symlink_to(go)
    (shim_dir / "openssl").symlink_to(_migration_openssl())
    sha256sum = shim_dir / "sha256sum"
    sha256sum.write_text(
        "#!/usr/bin/env python3\n"
        "import hashlib, pathlib, sys\n"
        "if sys.argv[1:2] == ['--check']:\n"
        "    for line in pathlib.Path(sys.argv[2]).read_text().splitlines():\n"
        "        expected, name = line.split('  ', 1)\n"
        "        actual = hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()\n"
        "        if actual != expected:\n"
        "            raise SystemExit(f'{name}: FAILED')\n"
        "        print(f'{name}: OK')\n"
        "else:\n"
        "    for name in sys.argv[1:]:\n"
        "        digest = hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()\n"
        "        print(f'{digest}  {name}')\n"
    )
    sha256sum.chmod(0o700)
    completed = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "RUNNER_TEMP": str(runner_temp),
            "APPLY_SIGNING_KEY": signing_key,
            "APPLY_PUBLIC_KEY": apply_public_key,
            "RECIPIENT_PUBLIC_KEY": b64encode(recipient_public_key).decode("ascii"),
            "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
        },
    )
    return completed, runner_temp / "apply-signing-key-migration"


def test_apply_signing_key_migration_round_trip_and_artifact_allowlist(
    tmp_path: Path,
) -> None:
    seed = bytes(range(32))
    apply_public_key, private_key = _keypair(seed)
    recipient_private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=3072
    )
    recipient_public_key = recipient_private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    raw_seed = b64encode(seed).decode("ascii")
    pkcs8_pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode("ascii")
    serializations = {
        "raw-seed": (raw_seed, raw_seed),
        "raw-seed-whitespace": (f" \n{raw_seed}\n ", raw_seed),
        "pkcs8-pem": (pkcs8_pem, pkcs8_pem.strip()),
    }

    for run_name, (
        serialized_private_key,
        expected_private_key,
    ) in serializations.items():
        completed, artifact = _run_apply_key_migration(
            tmp_path,
            run_name=run_name,
            signing_key=serialized_private_key,
            apply_public_key=apply_public_key,
            recipient_public_key=recipient_public_key,
        )

        assert completed.returncode == 0, completed.stderr
        assert sorted(path.name for path in artifact.iterdir()) == [
            "SHA256SUMS",
            "apply-public-key.txt",
            "apply-signing-key.oaep-sha256.bin",
        ]
        checksum_manifest = (artifact / "SHA256SUMS").read_text()
        assert str(tmp_path) not in checksum_manifest
        for line in checksum_manifest.splitlines():
            expected_digest, relative_name = line.split("  ", 1)
            assert (
                hashlib.sha256((artifact / relative_name).read_bytes()).hexdigest()
                == expected_digest
            )
        decrypted = recipient_private_key.decrypt(
            (artifact / "apply-signing-key.oaep-sha256.bin").read_bytes(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        assert decrypted.decode("ascii") == expected_private_key

    escaped_public_pem = (
        private_key.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
        .replace("\n", "\\n")
    )
    escaped_public, artifact = _run_apply_key_migration(
        tmp_path,
        run_name="escaped-public-pem",
        signing_key=raw_seed,
        apply_public_key=escaped_public_pem,
        recipient_public_key=recipient_public_key,
    )
    assert escaped_public.returncode == 0, escaped_public.stderr
    decrypted = recipient_private_key.decrypt(
        (artifact / "apply-signing-key.oaep-sha256.bin").read_bytes(),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    assert decrypted.decode("ascii") == raw_seed


def test_apply_signing_key_migration_rejects_mismatched_or_malformed_key(
    tmp_path: Path,
) -> None:
    seed = bytes(range(32))
    apply_public_key, _private_key = _keypair(seed)
    wrong_public_key, _wrong_private_key = _keypair(bytes(reversed(range(32))))
    recipient_private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=3072
    )
    recipient_public_key = recipient_private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    mismatched, _artifact = _run_apply_key_migration(
        tmp_path,
        run_name="mismatched",
        signing_key=b64encode(seed).decode("ascii"),
        apply_public_key=wrong_public_key,
        recipient_public_key=recipient_public_key,
    )
    malformed, _artifact = _run_apply_key_migration(
        tmp_path,
        run_name="malformed",
        signing_key="!" * 44,
        apply_public_key=apply_public_key,
        recipient_public_key=recipient_public_key,
    )
    pkcs8_pem = _private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode("ascii")
    trailing_data, _artifact = _run_apply_key_migration(
        tmp_path,
        run_name="trailing-data",
        signing_key=f"{pkcs8_pem}not-part-of-the-pem",
        apply_public_key=apply_public_key,
        recipient_public_key=recipient_public_key,
    )
    private_der = _private_key.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    internal_trailing_der = b64encode(private_der + b"X").decode("ascii")
    internal_trailing_pem = "\n".join(
        [
            "-----BEGIN PRIVATE KEY-----",
            *(
                internal_trailing_der[index : index + 64]
                for index in range(0, len(internal_trailing_der), 64)
            ),
            "-----END PRIVATE KEY-----",
        ]
    )
    trailing_der, _artifact = _run_apply_key_migration(
        tmp_path,
        run_name="trailing-der",
        signing_key=internal_trailing_pem,
        apply_public_key=apply_public_key,
        recipient_public_key=recipient_public_key,
    )

    canonical_seed = b64encode(seed).decode("ascii")
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    padding_index = alphabet.index(canonical_seed[-2])
    assert padding_index % 4 == 0
    noncanonical_seed = f"{canonical_seed[:-2]}{alphabet[padding_index + 1]}="
    noncanonical_private, _artifact = _run_apply_key_migration(
        tmp_path,
        run_name="noncanonical-private-base64",
        signing_key=noncanonical_seed,
        apply_public_key=apply_public_key,
        recipient_public_key=recipient_public_key,
    )
    public_padding_index = alphabet.index(apply_public_key[-2])
    assert public_padding_index % 4 == 0
    noncanonical_apply_public = (
        f"{apply_public_key[:-2]}{alphabet[public_padding_index + 1]}="
    )
    noncanonical_public, _artifact = _run_apply_key_migration(
        tmp_path,
        run_name="noncanonical-public-base64",
        signing_key=canonical_seed,
        apply_public_key=noncanonical_apply_public,
        recipient_public_key=recipient_public_key,
    )

    assert mismatched.returncode != 0
    assert "does not match the trusted apply public key" in mismatched.stderr
    assert malformed.returncode != 0
    assert "not strict base64 or PKCS8 PEM" in malformed.stderr
    assert trailing_data.returncode != 0
    assert "PEM contains trailing data" in trailing_data.stderr
    assert trailing_der.returncode != 0
    assert "contains trailing ASN.1 data" in trailing_der.stderr
    assert noncanonical_private.returncode != 0
    assert "not strict base64 or PKCS8 PEM" in noncanonical_private.stderr
    assert noncanonical_public.returncode != 0
    assert "not strict base64 Ed25519 material" in noncanonical_public.stderr


def test_apply_signing_key_migration_rejects_weak_recipient_rsa(
    tmp_path: Path,
) -> None:
    seed = bytes(range(32))
    apply_public_key, _private_key = _keypair(seed)
    weak_recipient = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    weak_public_key = weak_recipient.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    completed, _artifact = _run_apply_key_migration(
        tmp_path,
        run_name="weak-rsa",
        signing_key=b64encode(seed).decode("ascii"),
        apply_public_key=apply_public_key,
        recipient_public_key=weak_public_key,
    )

    assert completed.returncode != 0
    assert "must be RSA with at least 3072 bits" in completed.stderr
