# Axiom Encode

AI-assisted Axiom RuleSpec encoding infrastructure. This repo owns the automation layer for turning source legal text into jurisdiction-repo `*.yaml` RuleSpec artifacts and validating them with `axiom-rules-engine`.

## What this does

`axiom-encode` sits between the corpus and the jurisdiction rules repos in
the Axiom pipeline:

```
axiom-corpus (signed source releases)
  → axiom-encode (model generation + compile/proof/oracle gates)
  → rulespec-* repos (RuleSpec YAML + companion tests + signed apply manifests)
```

Given a citation (for example `26 USC 32(a)(1)`), it resolves the exact
provision text from the signed corpus release pinned by the target RuleSpec
checkout, prompts a model to draft the RuleSpec encoding, and only applies
output that passes the validation gates: RuleSpec compilation, proof-tree
validation against release-bound source text, companion tests, and oracle
comparisons where configured. Applies are recorded as signed manifests under
the target repo's `.axiom/encoding-manifests/`.

A checkout whose `.axiom/toolchain.toml` still declares the legacy
commit-ref schema (a repo mid-migration to signed-release pins) fails
current `axiom-encode` toolchain validation by design; the legacy file's
`axiom_encode_version` field names the encoder release that matches it.

## Who runs this

The Foundation team, on its supervised encoding runtime: generation runs on
operator-controlled compute, and the migrated jurisdiction repos require
RuleSpec content changes to carry the signed encoding manifests this
pipeline produces. The chartered admission model — where generation runs,
what gets signed, and what CI is allowed to do — is
[axiom-encode#1192](https://github.com/TheAxiomFoundation/axiom-encode/issues/1192).

If you are not running the supervised pipeline, you generally do not need
this repo:

- **Use the encodings** — consume published RuleSpec from the `rulespec-*`
  repos and the signed corpus releases they pin.
- **Contribute sources** — scrapers and corpus ingest live in
  [axiom-scrapers](https://github.com/TheAxiomFoundation/axiom-scrapers) and
  [axiom-corpus](https://github.com/TheAxiomFoundation/axiom-corpus).
- **Report problems** — a wrong or missing encoding is an issue on the
  relevant `rulespec-*` repo; fixes route through the supervised pipeline.

## Installation

```bash
uv sync
```

Without [uv](https://docs.astral.sh/uv/), create and activate a virtualenv,
then `pip install -e ".[dev]"`.

## Usage

```bash
axiom-encode encode "26 USC 32(a)(1)" \
  --corpus-path ~/TheAxiomFoundation/axiom-corpus \
  --policy-repo-path ~/TheAxiomFoundation/rulespec-us \
  --axiom-rules-engine-path ~/TheAxiomFoundation/axiom-rules-engine \
  --output /tmp/axiom-encode-encodings \
  --apply

axiom-encode validate ~/TheAxiomFoundation/rulespec-us/us/statutes/26/32/a/1.yaml \
  --corpus-path ~/TheAxiomFoundation/axiom-corpus \
  --axiom-rules-engine-path ~/TheAxiomFoundation/axiom-rules-engine
axiom-encode proof-validate ~/TheAxiomFoundation/rulespec-us/us/statutes/26/32/a/1.yaml \
  --corpus-path ~/TheAxiomFoundation/axiom-corpus
axiom-encode test \
  --root ~/TheAxiomFoundation/rulespec-us/us \
  --axiom-rules-engine-path ~/TheAxiomFoundation/axiom-rules-engine
axiom-encode inventory --root ~/TheAxiomFoundation/rulespec-us
```

`encode` validates generated output in a temporary overlay of the canonical
RuleSpec checkout before applying it. Standalone `validate` and
`proof-validate` accept only files inside that checkout; generated files under
the output directory are not a second validation layout.

`test --root` accepts exactly one jurisdiction content root in the canonical
`rulespec-<country>/<jurisdiction>` layout. Cross-country imports require an
explicit `--rulespec-dependency-root` pointing to the canonical country
checkout; ambient environment and sibling checkout discovery are not used.

`encode` resolves the requested citation to exactly one active
`corpus.provisions` row before model generation. Each RuleSpec checkout must
pin one signed release object in `.axiom/toolchain.toml` through
`[toolchain].axiom_corpus_release` and
`[toolchain].axiom_corpus_release_content_sha256`, and bind the exact raw bytes of the
canonical `known-validation-gaps.yaml` through
`[toolchain].validation_waiver_set_sha256`. Corpus-backed commands require an
explicit canonical local `axiom-corpus` checkout and an authenticated signing
broker initialized from the protected three-root config. The corpus-release
root comes only from that broker, never from the environment. Commands verify
`releases/<name>/<content_sha256>.json` and authorize artifacts only through its
signed `content.artifacts` inventory; there is no mutable `current` release,
directory-scan authorization, legacy layout, ambient checkout discovery, or
remote corpus resolver. Multiple matching active
rows are rejected as ambiguous. If the named release or an unambiguous provision
is unavailable, encoding stops before calling a model. Supabase run/session sync
is a separate telemetry feature and never supplies legal source text.

`encode` defaults to `--backend codex` with `gpt-5.6-terra`. Each section gets
up to two validator-rejected generations on that model, then one generation
with `gpt-5.6-sol`; use
`--escalate-after`, `--escalation-model`, or `--no-escalation` to override that
policy. Claude/Fable capacity is reserved for orchestration, gating, and review
rather than YAML generation. The Codex backend authenticates through the Codex
CLI's `~/.codex/auth.json`
(created by `codex login`, or an `OPENAI_API_KEY` recorded there); `CODEX_HOME`
overrides the directory and `OPENAI_API_KEY` in the environment also satisfies
the check. When neither is present `encode` stops with a clear error before
starting a run. Other backends stay available explicitly with
`--backend claude` or `--backend openai`.

`proof-validate` checks explicit RuleSpec proof trees without reviewers or
oracles. Proof atoms must cite immutable release-bound corpus text or an
explicit hashed RuleSpec import; mutable source-claim references are rejected.
Strict proof validation is enabled per file with
`module.proof_validation.required: true`.

`inventory` counts checked-in RuleSpec files in the exact canonical country
checkout passed to `--root`, separating source/provision encodings from
composition modules and reporting rule-kind totals. It does not discover
sibling repositories. Use `--json` when feeding the current inventory into
status reports or dashboards.

## Run telemetry and repair manifests

`axiom-encode encode` records each completed eval-backed run in the local
encoding DB and creates a linked SDK-style session with
`session_id=encode-<run_id>`. The session stores concise request/result/outcome
events, issue events when the final workflow fails, token counts, model, cwd,
and `axiom-encode` version so the public ops dashboard can show run health and
session telemetry from the same encode path.

When Supabase write credentials are configured, `encode` syncs both records:

```bash
export AXIOM_ENCODE_SUPABASE_URL=...
export AXIOM_ENCODE_SUPABASE_SECRET_KEY=...
axiom-encode encode "26 USC 32(a)(1)" \
  --corpus-path ~/TheAxiomFoundation/axiom-corpus \
  --policy-repo-path ~/TheAxiomFoundation/rulespec-us \
  --axiom-rules-engine-path ~/TheAxiomFoundation/axiom-rules-engine \
  --output /tmp/axiom-encode-encodings
```

The command prints `supabase_sync=run+session` when both records are uploaded.
If the credentials are missing, a `WARNING: run ... was NOT synced` line goes to
stderr — the run will not appear on the ops dashboard until reconciled. Use
`axiom-encode sync-agent-sessions --session encode-<run_id>` to replay a
single local session sync.

Because automation often runs in throwaway checkouts (where the local encoding
DB is deleted with the workspace), the signed apply manifests committed under
`.axiom/encoding-manifests/` in each rulespec repo are the durable record of
applied encodings. Reconcile Supabase with them at any time — the sync is
idempotent (rows are keyed by the manifest's original run id) and uses
`data_source=apply_manifest`:

Run `sync-applied-runs` under the same protected signing supervisor and three-root
configuration used by apply. Verification-only invocations omit signer
descriptors. Add `--dry-run` to preview.

The sync scans canonical country checkouts, not retired flat jurisdiction
repositories, and verifies every manifest against the configured Ed25519 public
key before using it as telemetry evidence. Verification never requires access
to private signing material.

Failed encode runs write a sibling `*.repair.json` file next to the generated
RuleSpec candidate. That manifest includes the run ID, session ID, citation,
trace path, generated output path, and actions for inspecting the trace,
and rerunning the same citation. The manifest does not include a manual repair
action; live RuleSpec changes should come from a new encode run. When
`--apply` validates and applies cleanly, the final outcome is successful even
if standalone validation failed before the generated file was checked in the
policy-repo overlay, and no repair manifest is written.

## Applying generated RuleSpec

Generated concept names are reviewed against the naming contract in
[`axiom-rules-engine/docs/concept-naming.md`](https://github.com/TheAxiomFoundation/axiom-rules-engine/blob/main/docs/concept-naming.md):
fragment patterns, the upstream-concept rule, and lowercase acronym tokens
with uppercase reserved for source-stated legal designators (the name lint
rejects accidental mixed case; all-uppercase tokens pass it, so casing is a
review item). `encode --apply` does not update the display acronym
registries described in the contract's Acronyms section; new acronym tokens
are flagged in review and closed by registry updates or an owned follow-up.

Live jurisdiction RuleSpec must be installed with `axiom-encode encode --apply`,
not by editing YAML in place. `--apply` validates the generated main file,
companion test, and direct dependent RuleSpec modules inside a temporary
policy-repo overlay, copies the validated files into the target rules repo, and
writes a signed JSON apply manifest under `.axiom/encoding-manifests/`.
For model-generated RuleSpec, the apply step mechanically stamps
`module.source_verification.source_sha256` and records the resolver-owned source
attestation in that manifest. The attestation binds the release-object content
digest, subordinate selector digest, provision artifact digest and row identity,
stored source-body digest, resolved corpus-text digest, and source/effective dates. Model output cannot
override those values, and apply fails if the attestation is incomplete or does
not match the RuleSpec source path.
Apply and retire commands must be launched by a trusted supervisor that attaches
the signing-broker socket before Python starts. The broker holds the Ed25519
signing capability out of process while the private key remains in an external
signer; the CLI receives only the broker capability. All three pairwise-distinct
apply, eval, and corpus-release public roots come from one root-owned,
caller-nonwritable JSON config; environment values do not define signing trust.
Without an attached broker initialized from that protected config, the encoder
refuses to install or retire RuleSpec.

The trusted supervisor is a separate Go-standard-library deployment artifact;
installing the Python package alone does not install it. Build it reproducibly
from the same reviewed revision:

```bash
mkdir -p build
CGO_ENABLED=0 go build -trimpath -buildvcs=false -ldflags='-buildid=' \
  -o build/axiom-encode-signing-supervisor \
  ./cmd/axiom-encode-signing-supervisor
```

After a trusted external signer has exposed an operation-scoped Unix socket on
inherited descriptor 3, launch apply through an immutable root-owned Python
runtime and `axiom-encode` package:

```bash
build/axiom-encode-signing-supervisor \
  --apply-signer-fd 3 \
  --trusted-signing-roots /etc/axiom/signing-trust-roots.json \
  --trusted-python-runtime-root /opt/axiom/python \
  --trusted-python-import-root \
    /opt/axiom/python/lib/python3.13/site-packages \
  --trusted-python-package-root \
    /opt/axiom/python/lib/python3.13/site-packages/axiom_encode \
  -- /opt/axiom/python/bin/axiom-encode encode ... --apply
```

Use `--eval-signer-fd 4` with the distinct eval signer selected by the protected
three-root config for
evaluation signing, or pass both signer options when both capabilities are
required. Raw private keys are never accepted by the supervisor, broker, or
Python process.
See [Trusted signing supervisor](docs/trusted-signing-supervisor.md) for the
external signer protocol, immutable execution checks, CI build, and platform
boundary.

Retire an applied module through the signed removal path. `--policy-repo-path`
is the canonical country checkout, and each target is jurisdiction-prefixed:

```bash
build/axiom-encode-signing-supervisor \
  --apply-signer-fd 3 \
  --trusted-signing-roots /etc/axiom/signing-trust-roots.json \
  --trusted-python-runtime-root /opt/axiom/python \
  --trusted-python-import-root \
    /opt/axiom/python/lib/python3.13/site-packages \
  --trusted-python-package-root \
    /opt/axiom/python/lib/python3.13/site-packages/axiom_encode \
  -- /opt/axiom/python/bin/axiom-encode \
  retire us/statutes/26/32/a/1.yaml \
  --policy-repo-path ~/TheAxiomFoundation/rulespec-us \
  --corpus-path ~/TheAxiomFoundation/axiom-corpus \
  --reason "superseded by the replacement source module"
```

`retire` accepts only a module already covered by a verified model
`encode --apply` manifest and includes its companion test automatically.

Canonicalize paths of already-authenticated v5 model encodings with the
separate path-only workflow. The plan is deliberately minimal and binds one
clean RuleSpec HEAD:

```json
{
  "schema_version": "axiom-encode/rulespec-path-migration-plan/v1",
  "base_commit": "0123456789abcdef0123456789abcdef01234567",
  "moves": [
    {
      "from": "us-la/statutes/47:32.yaml",
      "to": "us-la/statutes/47/32.yaml"
    }
  ]
}
```

Run it under the same protected apply-signing supervisor:

```bash
axiom-encode migrate-rulespec-paths \
  --plan /tmp/canonical-paths.json \
  --policy-repo-path ~/TheAxiomFoundation/rulespec-us \
  --corpus-path ~/TheAxiomFoundation/axiom-corpus
```

The destination must be the command's unique normalization of colon, whitespace,
case, or non-ASCII-dash path components. Companion tests and exact durable
RuleSpec references are derived rather than caller-authored. Legal
`corpus_citation_path` strings are preserved. The command accepts only files
with exactly one valid current-v5 model owner; legacy HMAC, manual,
deterministic, unmanifested, overlapping, or stale provenance fails closed.
It signs a plan/HEAD/tree/corpus/waiver/hash-bound receipt under
`.axiom/path-migrations/`, embeds the prior signed model manifest in each
replacement manifest, and installs moves, reference rewrites, receipts, and
manifests in one recoverable transaction.

Repository CI should run:

```bash
axiom-encode guard-generated \
  --repo "$GITHUB_WORKSPACE" \
  --corpus-path "$AXIOM_CORPUS_PATH" \
  --all
```

That guard protects atomic RuleSpec YAML under `legislation/`, `policies/`,
`regulations/`, and `statutes/` unless it matches a root-owned encoder apply
manifest whose signature, signed local release source identity, waiver-set
identity, and applied-file hashes are valid. The fifth canonical filesystem
root, `programs/`, contains declarative `axiom-compose` ProgramSpecs and is
deliberately outside every atomic RuleSpec encoding, manifest, signing,
validation, proof, waiver, import, concept, judge, and source-hash surface.
Program scope changes can be applied or checked deterministically with
`program-scope-sync`; they are admitted only by their compose-and-Rust build
gate. All YAML filenames use the canonical `.yaml` extension; `.yml` is
rejected. Running the whole-repository guard also makes a toolchain-only release
migration fail until every existing atomic manifest is regenerated against the
new immutable release.

## Source pinning and staleness

RuleSpec modules ground to legal text through exactly one
`module.source_verification.corpus_citation_path`. The block accepts only that
singular path and an optional `source_sha256` pin — the SHA-256 hex
digest of the full stored corpus provision body that supplied the encoding
source — plus
in-content `module.encoding_provenance` (`encoder`, `model`, `run_id`,
`reviewed_by`) and `module.validation` (oracle, `matches`/`mismatches`/
`pending` status, `last_run` date) blocks. These fields are inert at runtime.

Check every pinned module in a jurisdiction checkout against a local
`axiom-corpus` checkout:

```bash
axiom-encode check-source-staleness \
  --rulespec-root ~/TheAxiomFoundation/rulespec-us \
  --corpus-path ~/TheAxiomFoundation/axiom-corpus
```

The command exits `0` when every `source_sha256` pin still matches the active
corpus body and `1` when any module is stale (hash mismatch, ambiguity,
resolution failure, or a provision that can no longer be found). Generation,
validation, proof checking, staleness, and oracle corpus inspection all use the
same release-aware resolver. Metadata-only parent nodes are composed from their
unambiguous active descendants; child requests resolved from a stored parent
pin the complete parent body so any parent-body change flips staleness red.

`axiom_encode.source_hash` exposes release-bound building blocks for stamping
at encode time:

- `resolved_source_verification_block(release, citation_path)` resolves a
  named-release corpus citation and safely pins its complete stored body, including
  when the requested child is sliced from a parent row.
- `provenance_block(model, run_id)` builds `module.encoding_provenance`
  with the current `axiom-encode/<version>` as the encoder.
- `check_staleness(rulespec_root, release)` returns
  `StaleModule(module_path, pinned_sha, current_sha, resolution_error)` records
  for stale modules, for programmatic use.

`axiom-encode encode --apply` owns source-pin stamping for generated RuleSpec;
callers should not derive that pin from sliced prompt text. Oracle runners can
append `module.validation` entries as results land.

## Export candidates to Axiom core

`export-core-build-spec` packages a candidate and explicitly selected dependencies
for the real `axiom-core` compiler. It preserves their exact UTF-8 bytes, including
line endings, and emits an `axiom/build-spec/v0` JSON file:

```bash
axiom-encode export-core-build-spec \
  --root us:statutes/26/32/a/1 \
  --candidate /tmp/encodings/statutes/26/32/a/1.yaml \
  --module us:statutes/26/32/b /explicit/rulespec-us/us/statutes/26/32/b.yaml \
  --out /tmp/candidate.build-spec.json
```

Supply each dependency as its whole module, even when an import selects a symbol
with `#fragment`. The exporter does not infer dependencies from eval context,
hydrate a workspace, parse or repair RuleSpec, or normalize module targets. The
core compiler checks target syntax, the complete import closure, and executable
validity; it rejects missing and unused modules. An export can therefore succeed
while compilation fails.

The JSON file contains only `format`, `root`, and `modules`. A separate stdout
receipt reports `assurance: "unvalidated_candidate"`, the SHA-256 of the emitted
JSON bytes (including the final newline), the candidate's exact-byte SHA-256,
and the module count. To bind the export to a recorded generation, pass
`--expect-candidate-sha256` with its 64-character lowercase hexadecimal
`EvalResult.generated_output_sha256`. This hashes the generated RuleSpec file;
it is distinct from the corpus-body `source_sha256` described above. Neither hash
establishes legal validity or authority to install the candidate.

Inputs must be regular UTF-8 files; final-component symlinks and special files are
rejected. Parent directories are caller-selected and are not authenticated or
confined by this command. Each source, the combined sources, and the serialized
build spec are limited to 16 MiB. The output parent must already exist. The output
is created exclusively with owner-only permissions, so an existing file or
symlink is never overwritten. An interrupted write can leave a partial new file.
Generation, validation, and signed `--apply` retain their existing behavior.

Run the cross-repository tests with a built, real core executable:

```bash
AXIOM_CORE_BIN=/absolute/path/to/axiom-core/target/debug/axiom-core \
  uv run pytest --no-cov -q tests/test_core_export_integration.py
```

Without `AXIOM_CORE_BIN`, these tests skip explicitly; a configured but unusable
binary fails. Public encoder CI runs the exporter tests. The private core
repository can run the cross-repository tests against a pinned encoder revision
without granting this public repository access to core.

## Eval suites and readiness gates

Use manifest-driven benchmark suites when you want an explicit readiness answer instead
of ad hoc spot checks.

```bash
build/axiom-encode-signing-supervisor \
  --eval-signer-fd 4 \
  --trusted-signing-roots /etc/axiom/signing-trust-roots.json \
  --trusted-python-runtime-root /opt/axiom/python \
  --trusted-python-import-root \
    /opt/axiom/python/lib/python3.13/site-packages \
  --trusted-python-package-root \
    /opt/axiom/python/lib/python3.13/site-packages/axiom_encode \
  -- /opt/axiom/python/bin/axiom-encode eval-suite \
  benchmarks/us_snap_co_child_support_deduction_option_refresh.yaml \
  --output /tmp/axiom-encode-snap-co-child-support \
  --corpus-path ../axiom-corpus \
  --axiom-rules-engine-path ../axiom-rules-engine \
  --policy-repo-path ../rulespec-us \
  --policyengine-runtime-root /absolute/path/to/policyengine-us
build/axiom-encode-signing-supervisor \
  --eval-signer-fd 4 \
  --trusted-signing-roots /etc/axiom/signing-trust-roots.json \
  --trusted-python-runtime-root /opt/axiom/python \
  --trusted-python-import-root \
    /opt/axiom/python/lib/python3.13/site-packages \
  --trusted-python-package-root \
    /opt/axiom/python/lib/python3.13/site-packages/axiom_encode \
  -- /opt/axiom/python/bin/axiom-encode eval-suite \
  benchmarks/us_snap_federal_reconstruction_seed.yaml \
  --output /tmp/axiom-encode-snap-federal-seed \
  --corpus-path ../axiom-corpus \
  --axiom-rules-engine-path ../axiom-rules-engine \
  --policy-repo-path ../rulespec-us
axiom-encode eval-suite-archive /tmp/axiom-encode-snap-federal-seed \
  --corpus-path ../axiom-corpus \
  --axiom-rules-engine-path ../axiom-rules-engine \
  --policy-repo-path ../rulespec-us
```

- `benchmarks/us_snap_*_refresh.yaml` manifests are corpus-backed SNAP refresh lanes.
- `benchmarks/us_co_*` manifests exercise Colorado Works repair and seed cases.

Source cases have one machine source identity: a canonical
`corpus_citation_path` resolved from the RuleSpec checkout's pinned, signed
corpus release object. `name` is only a human-readable case label.

```yaml
cases:
  - kind: source
    name: snap-earned-income-deduction
    corpus_citation_path: us/statute/7/2014/e/2/B
```

The suite loader rejects the removed `source_id` field instead of translating
or retaining it as a compatibility alias. The `encode` and standalone
`eval-source` commands likewise derive their targets from their required corpus
citations; neither has a second source-identity override. A replay manifest is
regenerable only when its module already occupies the path derived from that
citation. Migrate or re-encode modules at any older logical path. Suite outputs
created under the old contract cannot be resumed, revalidated, reported, or
archived under the new contract. Rerun those suites into a fresh output
directory so every state record, ledger row, result, and report is bound to the
canonical corpus identity from the start.

Eval-suite result/evidence schema v5 content-addresses the generated RuleSpec, model
trace, and context manifest for every result. Resume, revalidation, reporting,
and archive admission safely reload those files and reject missing files or
SHA-256 mismatches. Verdict schema v5 carries a dedicated scope-domain-bound
Ed25519 signature over
the complete admission envelope: the immutable run UUID and original start time;
manifest and case identities; the manifest-declared runner set; corpus release
name, content, and selector digests; encoder, engine, RuleSpec, toolchain, and
waiver identities; plus the runner identity and mode, generated-output/trace/
context digests, prompt and token accounting, estimated and actual cost,
retrieved files and unexpected accesses, source attestation, and complete
validation verdict. Resume admission checks that signed envelope against current
inputs before rerunning compile, CI, reviewers, and configured oracles, then
compares the complete fresh metrics, success, and derivable error. Revalidation
instead authenticates the old envelope and bound generation bytes, recomputes
every result that has a generated output, and transactionally replaces the
verdict/ledger/results/summary/state closure. Old suite evidence and verdict
schemas are rejected rather than translated.

PolicyEngine cases also require `--policyengine-runtime-root` naming an absolute,
clean official `PolicyEngine/policyengine-<country>` checkout with a committed
`pyproject.toml`, committed `uv.lock`, and checkout-owned `.venv`. The canonical
active RuleSpec root determines the country; no manifest or CLI country override
exists. Axiom Encode freezes and signs the runtime's Git, lockfile, interpreter,
installed-version, module-origin, and package-tree identity, re-probes it around
oracle execution, and rejects mutation or resume/revalidation/report/archive
against a different runtime. It never discovers or installs a PolicyEngine
runtime from the environment. TAXSIM and the former `all` selector are removed.

Suite manifests are exact schemas: `runners` is the sole nonempty runner
declaration, and `eval-suite` has no CLI or ambient backend override. Manifests
must declare finite, in-range success, compile, CI, zero-ungrounded, and
generalist-review gates; PolicyEngine oracle cases must also declare their
PolicyEngine gate. Unknown fields, coercible strings, booleans-as-numbers, and
null thresholds are rejected.

Fresh `eval-suite` runs, resumed runs, and `eval-suite-revalidate` require an
externally attached broker with the eval-signing capability initialized from
the protected three-root config. Each config key accepts Ed25519 PKIX PEM or
exactly 32 raw bytes encoded with base64.
As with apply signing, run these mutation paths through the compiled trusted
supervisor with an operation-scoped external signer socket. The external signer
retains the private key and the broker verifies every returned signature.
`eval-suite-report` and `eval-suite-archive` only verify existing evidence and
use the protected broker roots without a signing capability. Never place private
signing material in the model-capable CLI, a manifest, or suite output.

Each suite reports:
- success rate
- compile pass rate
- CI pass rate
- zero-ungrounded rate
- PolicyEngine pass rate on oracle-mappable cases
- mean estimated cost

A configured maximum-cost gate fails unless every result has authenticated,
finite, nonnegative cost and token accounting; missing cost is not treated as a
free case.

The command exits `0` only when all readiness gates pass.

Use `eval-suite-archive` when a completed run is worth citing later. It first
revalidates the manifest, corpus release, RuleSpec checkout, waiver set,
encoder, engine, runners, and full case-by-runner ledger against the current
local inputs. It then copies the suite output tree into a durable local registry
under `artifacts/eval-suites` by default, rewrites archived JSON/JSONL artifact
paths away from `/tmp`, refreshes the result/evidence hashes, and appends a
record to `artifacts/eval-suites/index.jsonl`.

## PolicyEngine population oracles

Use `snap-populace-compare` to compare a SNAP composition module against
PolicyEngine over Populace records:

```bash
uv run --with "$HOME/PolicyEngine/populace/packages/populace-data[us]" \
  --with numpy \
  axiom-encode snap-populace-compare \
  --jurisdiction us-ny \
  --utility-projection policyengine-type \
  --positive-snap-only
```

The command runs `axiom-rules-engine` once over projected Populace records and
compares `us:statutes/7/2017/a#snap_regular_month_allotment` to PolicyEngine
`snap_normal_allotment`. Use `--jurisdiction us-co` or `--jurisdiction us-ny`;
add `--fail-on-mismatch` in CI when exact parity is expected, or
`--min-match-rate` when a documented upstream oracle gap makes a population
threshold more appropriate.

Populace oracle commands prefer a local engine-native H5 artifact before using
the `populace-data` package loader. To force a local dataset, set the most
specific available environment variable, for example
`AXIOM_POPULACE_US_2024_H5=/path/to/populace_us_2024.h5`. The loader also
checks `AXIOM_POPULACE_US_H5`, `AXIOM_POPULACE_H5`,
`AXIOM_POPULACE_DATASET`, and `AXIOM_POPULACE_DATA_PATH`.

Add `--external-oracle snapscreener` for a diagnostic comparison against the
public SnapScreener browser calculator. The command fetches or uses a local
`api.js`, records its SHA256 in stdout, and adds SnapScreener result columns to
`--write-csv`. This is a cross-check for implementation disagreements, not a
vendored dependency or legal source.

## Methods and paper notes

The repo now keeps paper-oriented internal documentation for benchmark-relevant
changes and current evidence.

- `docs/axiom-encode-methods-log.md` tracks the last meaningful harness changes,
  their hypotheses, and the evidence path to justify them later.
- `docs/rulespec-proof-validation.md` defines the proof-tree contract that keeps
  immutable corpus evidence and executable RuleSpec separated.
