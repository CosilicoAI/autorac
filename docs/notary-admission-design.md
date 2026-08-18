# Notary admission: design v27

Status: draft for sign-off. Implements the #1192 charter with the #1506
diff-coverage delta, under the build decision recorded on both issues
(dual-verdict, 2026-08-17). Version 27 folds design-review rounds 1–26
(one hundred thirty-three blocking findings; the record lives on #1507). Nothing
admission-capable merges until the §9 preconditions are satisfied and this
document is approved by the charter's gate: an independent cross-family
review of this concrete design plus Max's named sign-off, with every §11
decision closed.

## 1. Claim and threat model

The notary signs one narrow claim:

> Verification report `R` describes subject tree `H1` (manifest digest
> `T1`), reached from predecessor state `T0` recorded by chain predecessor
> `C` (a receipt, a transition record, or genesis): under the verifier
> profile and protected-path policy committed at the base, every
> protected-content terminal-entry change from `T0` to `T1` is covered
> by the unique valid assignment over the eligible lineage set; every
> gate the profile requires is declared with an outcome the profile
> accepts; unprotected terminal-entry changes are enumerated in `R`; and the protected
> signing policy authorized this acceptance.

Two classes of statement inside that claim have different evidentiary
strength, and the receipt does not blur them. **Recomputed invariants** —
tree manifests, the diff, protected-path classification, lineage
eligibility and authentication, the coverage assignment, the unprotected
enumeration, chain linkage — are computed by trusted pinned code from
content-addressed inputs, independently of anything the candidate
executed, and cannot be forged by candidate code. **Gate outcomes** are
declarations produced by running the candidate's own gates; a compromised
candidate weakening its gates is the residual risk the charter accepts,
unchanged in kind from today, and the receipt words them as declarations
("declared with an outcome the profile accepts"), never as re-established
facts.

Admission is a statement about the *delta*: the changes since the chain
predecessor are covered. It never upgrades the provenance of older
bytes. Pre-epoch content remains exactly what the genesis inventories
say — v5-attested or unattested baseline — no matter how many receipts
later sit above it. **State-identical chain advances are forbidden**: a
receipt whose base manifest equals its subject manifest refuses at
verification — a no-op receipt admits nothing, and permitting one opens
a stale-green race in which a no-op finalization voids a concurrently
verified sibling after its merge window.

The notary does not claim content was model-generated, and never will.
Lineage records say where bytes came from; the notary says the recomputed
invariants held, the gates were declared acceptable, and admission chains
from the last finalized state.

The scenario this design closes: an agent with repository write access
hand-edits a published rule and obtains a legitimate signature through the
sanctioned path, with no record of the edit's origin. Two walls stop the
*undeclared* hand edit — declared corrections are intentionally
admissible, loudly. The capability wall: the notary key never enters any
runner; an external typed signer validates an independently authenticated
job identity, and generation surfaces hold only the non-authorizing
producer key. The coverage wall: coverage is a recomputed invariant — the
trusted side re-derives it from git data and lineage records, so neither a
hand edit nor a forged report can pass it.

Residual risks accepted, per the charter: a verifier soundness bug
deterministically accepts wrong law; candidate-controlled gates can be
weakened (declared outcomes only); reviewer compromise authorizes
validator-passing bad content. Mitigations are the strict profile, the
trusted preflight, golden regeneration QA, and oracles — not additional
signatures.

## 2. Artifacts, schemas, and signature envelope

### 2.1 Conventions shared by every artifact

**Serialization.** Bodies are canonical JSON per **RFC 8785 (JCS)** —
the complete named contract for property ordering, string escaping, and
number serialization — over I-JSON-restricted input: no invalid Unicode,
no duplicate keys; every digest, id, and mode in these schemas is a JSON
string, so numeric-precision edge cases cannot arise. Bodies are UTF-8
bytes, content-addressed by SHA-256. Every body carries `schema` and
`lane` (`lane` is the canonical `owner/repo` name, compared
case-sensitively byte-for-byte). Schemas are closed-world: an unknown
field, missing field, wrong type, or duplicate key is a parse refusal.

**Value encodings, normatively.** One table, no exceptions:

| Kind | JSON encoding |
|---|---|
| SHA-256 digest (`*_sha256`, filenames, broker payload) | 64-char lowercase hex string, no prefix; GitHub artifact digests have their `sha256:` prefix stripped before entry |
| Git object id (`*_git_oid`, incl. `workflow_sha_git_oid`) | 40-char lowercase hex string (SHA-1 object format in the pilot) |
| Ed25519 signature | `signature_base64`: standard base64 with padding (RFC 4648 §4) |
| `signer_spki_sha256` | SHA-256 of the DER-encoded SubjectPublicKeyInfo, hex as above |
| `run_id`, `run_attempt`, `check_run_id`, `approve_check_run_id`, `artifact_id` | JSON strings, canonical decimal, no leading zeros, never numbers |
| `chain_predecessor_kind` | exactly one of `"genesis"`, `"receipt"`, `"transition"` |
| `tier` (profile only) | exactly one of `"public"`, `"restricted"`, `"ci-attested"` — never a report or receipt field; consumers read tiers from the profile the receipt binds |
| `ref` | the fully qualified Git ref string (`refs/...`) |
| `workflow_sha_git_oid` | the commit the workflow file was loaded from: **context-derived at candidate time** (`github.workflow_sha` in the `recompute` job), then **authenticated by equality** to `approve`'s OIDC `workflow_sha` claim at signing — one run, one workflow, so the equality is exact; reusable workflows are prohibited; never the run's `head_sha` |
| Entry modes | six-character octal strings: `"100644"` and `"100755"` (the pilot admits no symlinks anywhere — charter requirement 4 rejects symlink deltas, and refusing the mode tree-wide is the total form) |
| Paths | UTF-8 strings; sorting is bytewise over the UTF-8 encoding |
| Semantic arrays | **one comparator everywhere**: elements order bytewise over the UTF-8 encoding of the element's sort key, which each array names — `gates` by `gate_id`, `required_gates` by `gate_id`, `acceptable_outcomes` and `reasons` by their string value, `eligible_records`/`unused_eligible_records` by digest, `ineligible_records` by `store_name`, `coverage_assignment` by `path`, dependency `actions` by `ref_spec`, `containers` by `image`, inventories by `path`; set-valued arrays are strictly unique on their key (JCS canonicalizes objects, not arrays — this row is what makes array bytes deterministic) |

**Tree manifests.** A tree manifest is the recursively flattened list of
**terminal entries** (blobs and symlinks; directories appear only through
their contents' paths): the canonical JSON array of
`[path, mode, entry_sha256]` triples sorted bytewise by path, where
`entry_sha256` is SHA-256 over the entry's raw blob bytes — for symlinks,
the raw target bytes of the symlink blob, whatever they are. Pilot
totality rules, applied to **whole trees** at verification, genesis, and
finalization — not merely to diffs: a gitlink (mode 160000) anywhere in
the tree is a refusal; a non-UTF-8 path anywhere in the tree is a
refusal; terminal modes outside {"100644", "100755"} are a refusal —
symlinks refuse tree-wide in the pilot, per charter requirement 4.
Within these
rules the manifest function is total and exact: two trees with equal
manifests are terminal-entry-identical. Structural malformation refuses:
input trees must be fsck-clean; a duplicate flattened terminal path —
which Git itself treats as an fsck error but can technically encode —
is a manifest refusal, so per-path endpoints are always singular and
manifest set-difference is exact; and **any directory entry resolving to
the empty tree is a refusal** — an empty subtree is a real tree-entry
change the terminal manifest cannot see, and refusing it keeps the
claim's terminal-entry wording and the manifest's blindness aligned
rather than quietly divergent.

**Signature envelope.** Signing uses the existing broker frame verbatim:

```
"axiom-encode/external-signer-sign/v2" || 0x00 || scope || 0x00 || payload
```

with `payload` the 64-char lowercase-hex ASCII digest of the body. Scopes
are bound by a normative role table — the scope is the *signing role's*
identifier, which for countersignatures deliberately differs from the
body's schema:

| Role | Scope | Body signed |
|---|---|---|
| producer | `axiom/lineage-generation/v1` | generation event |
| actor | `axiom/lineage-correction/v1` | correction event |
| review | `axiom/lineage-correction-review/v1` | correction event (same body digest) |
| genesis | `axiom/notary-genesis/v1` | genesis |
| transition | `axiom/notary-transition/v1` | transition record |
| approver | `axiom/notary-approval/v1` | receipt candidate (its body digest) |
| admin-approver | `axiom/notary-admin-approval/v1` | genesis or transition administrative candidate (its body digest); signed by the **administrative** key of §8, never the correction-reviewer key — approving trust-root rotation is a wider power than validating a correction |
| notary | `axiom/notary-receipt/v1` | notary receipt |

Signatures are detached files beside the body, under a **normative
store-name grammar**: a record body is named `<digest>.json` where
`<digest>` is exactly the lowercase 64-hex SHA-256 of the body's raw
bytes — **the stem is the address; the `.json` suffix is grammar, not
digest** — and a signature sidecar is named `<digest>.json.<role>.sig`
with `<role>` one of the store's lineage scopes (`producer`, `actor`,
`review`; chain artifacts live on the chain branch, never in the
store). Each sidecar is canonical JSON:
`{schema: "axiom/detached-signature/v1", body_sha256, scope,
signer_spki_sha256, signature_base64}`. Every cross-scope verification
must fail, legacy `apply_ed25519`/`eval_ed25519` included (§10 tests the
full pairwise matrix).

**Typed signing.** The genesis, transition, and notary scopes are signed
only by the external typed signer (§5): it accepts typed fields,
reconstructs the canonical body itself, computes the digest, and signs. No
caller can submit arbitrary bytes to those scopes, genesis included.
Lineage scopes are signed at their origins (supervised runtime; actor and
reviewer tooling).

### 2.2 Generation event — `axiom/lineage-generation/v1`

Signed by the producer key (non-authorizing). Fields: `schema`,
`lane`, `epoch_sha256`; `runtime_identity`, `model`, `cli_version` —
non-empty JSON strings; `cli_sha256` — a digest; `prompt_sha256s` — a
sorted, strictly unique digest array; `emitted_at` — an RFC 3339 UTC
string, informational and unverified; and `transitions`, sorted
bytewise by path:

```
{path, before_blob_sha256 | null, before_mode | null,
 after_blob_sha256 | null, after_mode | null,
 patch_note_sha256 | null}
```

Modes are `"100644"` or `"100755"`, `null` exactly when the corresponding
blob digest is `null`. A mode-only change is a transition shape the
schema can carry, but **the pilot refuses protected-path mode changes
outright** — charter requirement 4 rejects executable-mode deltas at
preflight, and the pilot honors that wall rather than admitting covered
mode transitions; relaxing it is the §11 charter-alignment decision. `patch_note_sha256` is **opaque audit
metadata**: a producer-chosen digest of whatever diff rendering the
runtime archived. It is never verified, carries no algorithm contract, and
no refusal depends on it — endpoint blob digests and modes are the sole
ground truth.

### 2.3 Correction event — `axiom/lineage-correction/v1`

Signed by the actor (role `actor`) and countersigned by a protected
reviewer key distinct from the actor's (role `review`, per the §2.1 role
table). Fields: `schema`, `lane`, `epoch_sha256`; `actor` and
`reason` — non-empty JSON strings; `predecessor_record_sha256 | null`;
`transitions` as §2.2. The
countersignature validates lineage; it never authorizes merge. Corrections
are first-class and loud: the sanctioned response to a wrong encoding
remains fix-the-encoder-and-re-encode, and a correction event is the
recorded exception, never the quiet path.

### 2.4 Verification reports (Job 1, unsigned) — pass and refusal variants

Refusals cannot carry fields their stage never established (an ambiguity
refusal has no unique assignment; a structural refusal has no manifest),
so the report is **two closed schemas**, not one with impossible fields.

`axiom/notary-report-refusal/v1`: `{schema, lane, epoch_sha256,
subject_commit_git_oid, stage, refusal, established}` — `established`
is a closed object of the report's contextual fields in fixed
dependency order (repository and predecessor resolution, then
manifests and chain fields, then policy and profile digests, then
record lists), each nullable with the **prefix property**: a non-null
field implies every field before it is non-null, and a field is
non-null exactly when its computation completed before the failing
check. Its exact ordered members: `base_commit_git_oid`,
`chain_predecessor_sha256`, `chain_predecessor_kind`,
`base_tree_manifest_sha256`, `subject_tree_manifest_sha256`,
`profile_sha256`, `path_policy_sha256`, `eligible_records`,
`ineligible_records` — types as in the pass schema. A refusal during
repository or predecessor resolution is a `resolution`-stage refusal —
`chain-unresolvable` when the finalized chain state cannot be
reconstructed, `subject-unresolvable` when the subject commit or tree
cannot be resolved — and carries the truthful prefix resolution
actually established: all-null when chain reconstruction failed (it
runs first), the three chain fields when the subject failed after the
chain resolved. Stage
maxima: `resolution`, `structural`, and `preflight` refusals stop
wherever the prefix stopped; `eligibility` and `assignment` refusals may establish through
the record lists — **an assignment-stage refusal never carries an
assignment or unused list** (an ambiguity refusal has neither a unique
assignment nor a determinate unused partition); only `gates`-stage
refusals carry `coverage_assignment`, `unused_eligible_records`,
`unprotected_changes`, and `gates` (coverage succeeded there).

| `stage` | `established` content |
|---|---|
| `"resolution"` | the actual completed prefix — all-null when chain reconstruction fails (it runs first); the three chain fields when subject resolution fails after the chain resolved |
| `"structural"` | the actual completed prefix — a late structural failure (an empty subtree found while building the subject manifest, after resolution and the base manifest succeeded) truthfully carries the resolved predecessor and base fields; manifests establish in fixed order, **base first, then subject**, so identical faults establish identical prefixes |
| `"preflight"` | whatever prefix the failing check permitted — a `policy-invalid` for an absent profile truthfully carries `profile_sha256: null` |
| `"eligibility"` | prefix through the record lists |
| `"assignment"` | prefix through the record lists — **never** an assignment or unused partition (an ambiguity has neither) |
| `"gates"` | the full prefix, plus the gates-stage extras below |

Formally the refusal report is a **discriminated union on `stage`**:
the common shape (`schema`, `lane`, `epoch_sha256`,
`subject_commit_git_oid`, `stage`, `refusal`, `established`) for every
stage, and the `"gates"` variant alone extending it with
`coverage_assignment`, `unused_eligible_records`,
`unprotected_changes`, and `gates` — closed-world parsing accepts
exactly the variant its discriminator names. The prefix property is
the single rule for `established`; the table describes what it yields
per stage and never overrides it — including for structural refusals,
whose prefixes reflect how far establishment actually got. Only `"gates"`-stage refusals
additionally carry `coverage_assignment`, `unused_eligible_records`,
`unprotected_changes`, and `gates` — there coverage succeeded, so all
four are determinate.

A `"gates"`-stage refusal does carry its unique assignment — coverage
succeeded; a gate outcome did not. The refusal member is named
`refusal` in both variants (never `diff_coverage`), and its single
closed code enum is: `"subject-unresolvable"`,
`"chain-unresolvable"`, `"uncovered-path"`, `"ambiguous-assignment"`,
`"inconsistent-chain"`, `"record-cycle"`, `"no-valid-execution"`,
`"inadmissible-entry"`, `"structural"`, `"state-identical"`,
`"trust-surface-change"`, `"policy-invalid"`, `"predecessor-stale"`,
`"gate-unacceptable"`, `"gate-missing"`, `"gate-extra"` — with
templates for the added codes: `state-identical` → "base and subject
states are identical"; `trust-surface-change` → "ordinary candidate
changes a trust surface"; `policy-invalid` → "required policy, profile, or key registry absent
or invalid at the base" (registry invalidity refuses under this code);
a non-representable offending path reports `path: null` with the
template unchanged — the I-JSON path contract never carries non-UTF-8
bytes; `subject-unresolvable` → "subject commit or tree cannot be
resolved" (the common `subject_commit_git_oid` field carries the
requested oid, which is input, not resolution); `chain-unresolvable` →
"finalized chain state cannot be reconstructed" (an absent, malformed,
or invalid-to-reconstruction chain — distinct from `predecessor-stale`,
whose chain resolved and whose base is simply not the tip);
`predecessor-stale` → "base is
not the finalized chain tip"; `gate-extra` → "gate outside the
profile". `ineligible_records`
entries carry `reasons`: the **sorted array of every applicable reason
code**, so a multi-fault record has one deterministic representation.
Refusal reports are diagnostics; no downstream artifact ever binds
one.

`axiom/notary-report-pass/v1` — the only variant the trusted side will
reconcile — content-addressed output of the verification workflow.
Fields: `schema`,
`lane`, `epoch_sha256`; `subject_commit_git_oid`,
`subject_tree_manifest_sha256`; `base_commit_git_oid`,
`base_tree_manifest_sha256`; `chain_predecessor_sha256`,
`chain_predecessor_kind`; `profile_sha256`, `path_policy_sha256`;
`corpus_release` `{name, content_sha256}` — derived normatively from
the repository-root `.axiom/toolchain.toml` at the **base** per the
existing toolchain contract, with `.axiom/toolchain.toml` itself a §4
trust surface (transition-only); `waiver_set_sha256`;
`eligible_records` and `unused_eligible_records` (sorted digest arrays);
`coverage_assignment`: array sorted bytewise by path of
`{path, record_sha256s}` — for each changed protected path, the record
digests consumed for its chain, in chain order (the unique assignment of
§3.3); `gates`: array sorted
by `gate_id`, **one entry per gate id** (duplicates are a parse refusal),
each `{gate_id, outcome}` — tiers are not report fields at all: a gate's
tier comes from the digest-bound profile, so a report cannot strengthen a
classification the profile did not grant; `diff_coverage`: exactly `"pass"` — the pass
variant carries no refusal shapes, and `waived`/`not-run` exist in
neither variant; `unprotected_changes`
(sorted paths); `ineligible_records`: sorted array of
`{store_name, reasons}` — `store_name` is the literal name string
relative to `.axiom/lineage/`, suffix and any `/` separators included,
whatever bytes it holds within the UTF-8 tree domain; the array sorts
bytewise by `store_name`. (Eligible records remain digest-keyed — they
passed the address check, so stem and digest coincide; ineligible
entries are name-keyed precisely because their names may not be
addresses.)
`reasons` is the sorted array of every applicable code **whose
prerequisites are satisfied**, from the closed enum with this
prerequisite order: `address-mismatch`, `unrecognized-store-name`, and
`malformed-record` are always computable — with
`unrecognized-store-name` **exclusive**: a non-grammar name's contents
are never evaluated, so it is that entry's sole reason; `invalid-signature`,
`wrong-lane`, `wrong-epoch`,
`duplicate-transition-paths`, and `unprotected-path-transition` apply
only when the body parsed (never beside `malformed-record`). Entries
are keyed by `store_name` under the §2.6 classification, total over
every tree: a name matching the §2.1 body grammar whose stem differs
from the file's raw-byte digest is an `address-mismatch` entry; a name
matching neither the body grammar nor the
sidecar-of-a-newly-introduced-body pattern is an
`unrecognized-store-name` entry — each under its literal name — so a
valid body added at its correct address *and* at an alias yields one
eligible record plus one ineligible entry, with no identity collision
and no unrepresentable key. Same shape in
pass and refusal variants; `dependency_pins_sha256`: the digest of a
canonical dependency inventory, a body of schema
`axiom/notary-dependency-inventory/v1` with the closed shape
`{schema, lane, actions: [[ref_spec, git_oid], …] sorted by ref_spec
with strictly unique ref_spec keys,
containers: [[image, digest_sha256], …] sorted by image with strictly
unique image keys, python_lock_sha256, verifier: {repo, git_oid}}`, where
`python_lock_sha256` is SHA-256 over the raw bytes of `uv.lock` at
`verifier.git_oid` in `verifier.repo`; `verifier`: `{repo, git_oid}`,
the repository-qualified identity of the verifier code that ran.
`waiver_set_sha256` adopts the existing toolchain contract exactly:
SHA-256 over the raw bytes of the repository-root
`known-validation-gaps.yaml` at the base, which current mechanics
require to be a present, bounded regular file — an absent or non-regular
waiver file is a refusal, not an empty hash. The refusal variant's `refusal` member is itself closed:
`{code, path | null, detail}` with the single code enum of §2.4
(coverage and gate codes alike), `detail` a JSON string — and
deterministic under simultaneous faults, with all three members fixed:
the evaluation order is total — resolution (finalized-chain
reconstruction, then subject resolution), structural (§2.1 manifest
rules in listed order), preflight (§4 in listed order), state-identical,
eligibility (§3.2 in listed order), assignment (§3.3 rules 1–4 in
order), gates (missing, then extra, then unacceptable; within each,
bytewise-least `gate_id`) — checks run in that total order **globally across all
paths before the next check** (every path's rule-1 existence check
precedes any path's rule-2 chain validation, so `uncovered-path` on
any path wins over `inconsistent-chain` on another); the reported
refusal is the first failing check, `path` the bytewise-least affected
path within it, `null` otherwise; `detail` is the fixed template string defined per code — normatively:
`uncovered-path` → "no eligible transition covers this path";
`ambiguous-assignment` → "more than one valid assignment";
`inconsistent-chain` → "eligible transitions exist but no valid chain";
`record-cycle` → "consumed records admit no execution order";
`no-valid-execution` → "no topological order yields realizable trees";
`inadmissible-entry` → "entry mode or type inadmissible";
`subject-unresolvable` → "subject commit or tree cannot be resolved";
`chain-unresolvable` → "finalized chain state cannot be reconstructed";
`structural` → "tree or store structurally malformed";
`state-identical` → "base and subject states are identical";
`gate-missing` → "required gate absent";
`gate-unacceptable` → "gate outcome outside the acceptable set" —
so identical faults produce identical bytes. Code selection between
`uncovered-path` and `inconsistent-chain` is fixed: `uncovered-path`
when no eligible transition names the path at all,
`inconsistent-chain` when transitions name it but no valid chain
exists. The preflight order is the closed list: repository resolution,
chain-predecessor admissibility, policy and profile presence, registry
validity, trust-surface refusals in §4's listed sentence order. The
state-identical refusal carries code `"state-identical"`, stage
`"preflight"`. There is no `diff_coverage` refusal
object anywhere; the pass variant's `diff_coverage` is the literal
`"pass"` and nothing else.

The profile is likewise a closed committed schema at a normative path:
`.axiom/notary/profile.json`, schema `axiom/notary-profile/v1`, fields
`{schema, lane, required_gates, oracle_policy}` with `required_gates`
sorted by `gate_id` and **strictly unique per gate_id**, each
`{gate_id, acceptable_outcomes, tier}` where `acceptable_outcomes` is a
sorted string array, and `oracle_policy`
exactly one of `"fail-closed"` or `"reduced-tier"` — so `profile_sha256`
has exactly one preimage. Every committed policy body (`path-policy`,
`transition-path-policy`, `profile`) carries `schema` and `lane` like
any other body.

The report is a *proposal*. Refusal reports are diagnostic artifacts, and
even a pass report authorizes nothing: every **recomputed invariant**
is re-derived by the trusted side (§5) before signing — gates alone are
validated declarations, per §1's stated residual — so a
candidate-forged report (pass verdict, trimmed unprotected list,
invented assignment) fails reconciliation rather than getting signed.

### 2.5 Notary receipt (signed) — `axiom/notary-receipt/v1`

The only artifact that, once finalized (§7), authorizes admission — in
two layers, candidate and wrapper, each a closed schema.

**The candidate** (`axiom/notary-receipt-candidate/v1`) is produced by
the `recompute` job and carries exactly: the full §2.4 pass-report
field set with `schema` replaced by the candidate's own identifier — every recomputed field including `diff_coverage` (exactly
`"pass"`), `eligible_records`, `unused_eligible_records`,
`ineligible_records`, `coverage_assignment`, `unprotected_changes`, and
`gates` (validated declarations: deduplicated, profile-complete,
outcomes acceptable) — plus `report_sha256` (the reconciled proposal)
and `job1`: `{workflow_ref, workflow_sha_git_oid, ref, run_id,
run_attempt, check_run_id, conclusion, artifact_name, artifact_id,
artifact_sha256}` — `conclusion` must be the literal `"success"`;
`artifact_name` is a non-empty JSON string validated against the
pinned workflow's declared artifact name. (`authorization.environment`
— a wrapper field, not a candidate field — is likewise a non-empty
string validated against the `notary-signing` environment name where
§2.5 defines the wrapper.) `workflow_ref` is the OIDC
`workflow_ref` claim
string verbatim (its `@`-suffix must equal `ref` or the candidate
refuses), and `job1.check_run_id` is **`verify`'s** check-run id from
the jobs-for-run lookup — the `approve` job's own OIDC `check_run_id`
appears only in the wrapper's authorization block, never here. The
candidate contains no authorization — nothing circular.

The `job1` block is **run-scoped provenance, stated as such**: GitHub
artifacts carry no producing-job or attempt identity, so it claims only
that the named artifact with this digest existed in this run, which the
pilot requires to contain exactly one artifact of that name. The
narrowing is safe because no recomputed invariant depends on the
artifact — the trusted side re-derives them all, and gate declarations
carry §1's residual by construction — and gate declarations
attribute to the proposing run as a whole: a different job in the
candidate's own run forging them is the same trust domain and the same
accepted residual as the candidate weakening its gates. An OIDC-bound
upload attestation tightening this to job-level is future work, out of
milestone one.

**The final receipt** (`axiom/notary-receipt/v1`) is exactly
`{schema, lane, epoch_sha256, candidate_sha256, authorization}` with
`authorization`: `{environment, approve_check_run_id,
approval_signature_sha256}` — the `approve` job's own OIDC
`check_run_id`, and the digest of the detached
`axiom/notary-approval/v1` signature file. Candidate fields are
referenced by digest, never flattened, and the candidate body and
approval-signature file **must be published on the chain branch beside
the receipt** — a receipt whose candidate or approval file is
unpublished is unverifiable and invalid to reconstruction. The reviewer
approval signs the candidate digest; the signer assembles the wrapper
**by derivation, never by input**: wrapper `lane` and `epoch_sha256`
are copied from the candidate and verified equal to the chain's;
`authorization.environment` is taken from `approve`'s authenticated
OIDC claims, not from any caller value. The publisher and
reconstruction re-verify candidate/wrapper/chain lane and epoch
equality; a valid candidate for one lane or epoch cannot be wrapped
into another. Genesis and transitions are distinct schemas; a receipt
cannot claim their role.

### 2.6 Storage

Lineage records live in the lane repository under `.axiom/lineage/`,
append-only, one body file per record named by its content digest,
detached signatures beside it. The lineage directory is outside the
protected-content coverage domain but inside the preflight's structural
checks: schema-valid, authenticated, correctly content-addressed,
append-only; deleting or mutating an existing record is a refusal.
Base-present records are **inert history**: the structural rule
enforces append-only discipline alone (no deletion, no mutation) and
never re-validates their contents — an invalid record that once rode
in as ineligible sits harmless forever, since eligibility is the only
door and it never reopens for base-present files. (Re-validating the
inherited store would let one admitted bad record permanently poison
every successor.) A **newly introduced** record that is malformed,
misaddressed, or badly signed is an eligibility-stage ineligible
record, enumerated with its reasons — never a structural refusal — so
one bad new record cannot veto an otherwise covered candidate, now or
later. **Every newly introduced store file is classified exactly
once** by the §2.1 grammar: a name matching the body grammar with stem
equal to its raw-byte digest enters the record pipeline (eligible, or
ineligible for content reasons); a sidecar naming a newly introduced
body is signature evidence, verified with that body and never
separately enumerated (an absent or invalid required sidecar is the
body's `invalid-signature` reason); everything else — a body-grammar
name with a wrong stem (`address-mismatch`), or any other name: an
orphan sidecar, a sidecar naming a base-present body, a nested name, a
non-lineage role suffix, a non-hex or uppercase stem
(`unrecognized-store-name`) — is an ineligible entry under its literal
name, riding in as enumerated, inert history.
Reports, receipts, genesis, and transition records publish per §7 and are
never part of the subject tree.

## 3. The diff-coverage predicate

### 3.1 Protected paths

The policy is a committed file at the base
(`.axiom/notary/path-policy.json`):
`{schema: "axiom/notary-path-policy/v1", lane, rules: [{action:
"include" | "exclude", prefix}]}`. A prefix matches path `p` iff
`p == prefix` or `p` starts with `prefix + "/"` (component-wise; `rules`
never matches `rules-evil/x`). Last matching rule wins; a path matching no
rule is unprotected. Classification uses the policy committed at the
base. Tree-wide totality (UTF-8, no gitlinks, admissible modes) is
enforced by §2.1's manifest rules, so classification is total over every
manifest the system accepts.

### 3.2 Eligible lineage set

Records present under `.axiom/lineage/` in the subject tree and absent at
the base — introduced by this candidate — carrying this `lane` and
`epoch_sha256`, with valid signatures per the role table. The sorted
digest list is bound into report and receipt. Replay of **previously
published** records is dead: one merged by an earlier candidate is
present at the base and ineligible ever after; cross-lane and
cross-epoch records refuse on their bindings; records whose
before-digests no longer match the base refuse in replay (§3.3). A
never-published record stockpiled from an abandoned attempt can be
introduced later if its endpoints recur — within the coverage claim
(the bytes are authentically lineage-attested); event-to-attempt
freshness binding is named future work, not claimed.

### 3.3 The predicate, deterministic and total

Compute the tree-entry diff between the base and subject manifests —
manifest set-difference is normative; the Git equivalent is
`git diff-tree -r --raw -z --no-renames` (recursive: without `-r`,
diff-tree reports directory names, which would let nested protected
changes hide behind an unprotected directory entry): per-path additions,
deletions, and modifications, including mode-only modifications. Renames
do not exist at this level. Over the protected subset:

1. **Unique valid assignment.** Record hygiene first: a record's
   transitions must have unique paths (duplicates refuse the record at
   parse), and every transition path in an eligible record must be
   protected under the base policy — a record touching any unprotected
   path is ineligible, listed with its reason, and can never be
   consumed, so consumption semantics never depend on unprotected diff
   entries. A valid assignment then maps every changed protected path to
   exactly one non-forking, non-cyclic chain of transitions drawn from
   eligible records — records consumed atomically (every transition of a
   consumed record used against the diff, or none), no transition
   against an unchanged path, endpoints per rule 2. If no valid
   assignment exists, refuse (uncovered or inconsistent). **If more than
   one valid assignment exists, refuse as ambiguous** — a candidate
   avoids this by not shipping redundant covering records. Records
   unusable in any valid assignment (dead-end retries) are legal, listed
   as unused, and create no ambiguity.

   Two whole-record rules complete the assignment's consistency. First,
   **a repository-state execution must exist**: build the consumed-record
   dependency graph (an edge from record X to record Y wherever, in some
   path's chain, a transition of Y consumes the state a transition of X
   produced); the graph must be acyclic, and applying whole records in a
   topological order must replay **the protected-state projection**:
   starting from the base's protected entries, every transition's
   before-state must be current when its record applies, and the
   replayed projection must terminate at the subject's protected
   entries, and **every intermediate projection must be realizable as a
   Git tree** — prefix-free, no path simultaneously a terminal entry
   and a directory prefix of another. Acceptance is existential and
   therefore deterministic as a predicate: the assignment passes iff
   **some** topological order yields all-realizable intermediates with
   every endpoint current; if none does, the refusal is
   `no-valid-execution`, a sibling of `record-cycle`. Unprotected
   entries — including the candidate's own new
   lineage files, which every ordinary candidate necessarily adds — are
   outside the replay and accounted separately (`unprotected_changes`
   and the §2.6/§3.2 lineage checks); a replay defined over the whole
   tree would falsely reject every nonempty candidate — cross-path cycles between records (X and Y each feeding the
   other on different paths) refuse as `record-cycle`, because no
   repository state ever contained what the subject cherry-picks.
   Second, **correction predecessors have predicate meaning, scoped to
   the current candidate**: a consumed correction event's
   `predecessor_record_sha256` must be null or equal the digest of the
   immediately preceding consumed record in the same path's chain
   within this assignment; anything else refuses. A correction landing
   in a later candidate than the record it amends uses null — the
   historical relationship belongs in `reason`, since cross-receipt
   pointers would force references to permanently ineligible records.
   It is lineage, verified — not opaque metadata implementations may
   ignore.
2. **Blob-and-mode ground truth.** Each chain's first
   `(before_blob_sha256, before_mode)` equals the path's state at the
   base (`(null, null)` for additions); each link's after-state equals
   the next link's before-state; the final after-state equals the path's
   state in the subject (`(null, null)` for deletions). Mode-only changes
   need a covering transition like any other change.
3. **Admissible entries and the mode wall, domain-total.** The pilot's
   protected domain is **100644-only**: a protected entry with mode
   `"100755"` — at the base, at the subject, or inside any transition —
   refuses. This is stronger than an endpoint or chain rule and closes
   every laundering shape at once, including delete-in-one-receipt,
   recreate-executable-in-the-next: no admissible executable state
   exists anywhere in the protected domain, so there is nothing to
   launder toward. (Every rulespec file is a 100644 YAML file; the
   restriction costs the pilot nothing.) Executable support, if ever
   wanted, is the §11 charter-alignment decision and arrives with the
   chain-state tombstone machinery a cross-receipt wall needs.
   Protected paths are regular 100644 blobs wherever they exist — symlinks are inadmissible in the
   protected domain, and gitlinks are refused tree-wide by §2.1.
   File/directory replacements decompose into entry deletions and
   additions and are covered as such.
4. **Binary verdict.** The predicate outcome is `"pass"` (the pass
   variant's `diff_coverage` literal) or a typed refusal (the refusal
   variant's `refusal` member).
   From the enforcement epoch there is no `waived` and no `not-run`. This
   is deliberately stricter than consumer-side declaration completeness.

The entire predicate — manifests, diff, classification, eligibility,
authentication, assignment uniqueness, unprotected enumeration — is a
pure function of git data and lineage records. It runs twice: in Job 1
(producing the report) and independently on the trusted side (§5), which
is why a forged report cannot survive.

Advisory shadow runs are permitted only before a lane's enforcement
epoch.

## 4. Verifier profile P and trusted preflight

The profile is a committed definition at the base, digest-bound into
report and receipt. It defines the required gate set, each gate's
acceptable outcomes, and the oracle policy. Against the current apply
path: non-mutating (no repairs; repairable-but-unrepaired refuses);
oracles on (licensed/unavailable oracles per the §11 decision — fail
closed or visibly reduced-tier, never silent); reviewers means
deterministic checks plus protected-environment human approval (the
validator pipeline's LLM reviewers are QA outside the admission path);
no caller switches (skip flags and caller-disableable guards have no
notary equivalents).

The trusted preflight (pinned code, before any candidate code executes)
resolves the canonical repository, the chain predecessor and its recorded
state, the policy and profile at the base, and the complete tree-entry
diff. Ordinary candidates changing workflows, action pins, verifier pins,
trust roots, the path policy, the profile, waiver policy,
repository-structure rules, or lineage-store integrity refuse; those
surfaces move only by transition record (§6.4).

## 5. Execution, trust boundary, and the external signer

**One workflow run, four named jobs.** All jobs share the **same
workflow run** — the topology is normative because it makes workflow
identity derivable: any job's OIDC `workflow_sha` covers the whole run,
and no per-job claim GitHub does not issue is ever needed. The jobs,
each on a fresh ephemeral runner, none reusable:

| Job | Runs candidate code | Token | OIDC use |
|---|---|---|---|
| `verify` (Job 1) | yes | `contents: read` only, no secrets, no environment | none |
| `recompute` | no (pinned code) | `contents: read`, `actions: read` | reads `github.workflow_sha` from the run context — the authorized candidate-time source; the signer later compares it against `approve`'s OIDC `workflow_sha` claim and refuses on mismatch |
| `approve` | no (pinned code) | `id-token: write` only — the control-plane lookups (`actions: read`) belong to the **signer's own credential**, not to any job | presents identity to the signer |
| `publish` | no (pinned code) | the two §9.7b credentials, vended by the publisher-token broker against its OIDC | `id-token: write`; presents identity to the publisher-token broker |

`publish` (and the finalizer below) run in a dedicated protected
**`notary-publishing` environment** with its administrator bypass
disallowed and audited — but environment rules bind reviewers and refs,
not workflow identity, so the credentials themselves are not
environment secrets at all: `publish` presents its own OIDC
(`id-token: write`) and a **separately deployed publisher-token
broker** — holding no notary key, no signing operation, and nothing but
the two App credentials — vends the two publisher tokens against the
exact identity (repository, `workflow_ref` and `workflow_sha`, ref,
run, attempt, environment), the same authentication discipline the
signing broker applies to `approve`. The two brokers are distinct
deployments by requirement: the signing broker is forbidden
repository-write capability, and the publisher-token broker is
forbidden signing capability. A revoked, rerun, or foreign workflow
gets nothing. Minting authority never enters any job; the current
publisher's in-job App minting is what this replaces. Finalization
necessarily runs after the merge — but its trust does not ride any
lane workflow file, because a workflow-rotation transition changes the
very file GitHub would load post-merge, and the successor code is not
yet finalized. So **finalization is a typed broker operation**: the
`notary-finalize.yml` workflow (triggered by the protected content
ref, in the publishing environment) merely *requests* finalization
with its OIDC identity; the **publisher-token broker itself**
re-derives the merged tip's manifest, validates the signed pending
artifact, its chain predecessor, and the state-machine rules, and
executes the CAS commit with its own credential. Even
not-yet-finalized workflow code can only ask; it validates nothing and
holds nothing, so squash rewrites and credential-bearing workflow
rotations stay harmless.

Job conclusions are read through the jobs-for-run endpoint (by job name
and attempt), never self-asserted; the `approve` job's identity is
bound the only way OIDC permits — its token's `check_run_id` claim,
reconciled through jobs-for-run to the expected job name and attempt
(GitHub's claims carry no job name). The receipt's `job1.check_run_id`
is the `verify` job's check run, recorded from that same lookup.

**Job 1 — verify (secretless, candidate-executing).** Runs on a fresh
ephemeral runner (GitHub-hosted or dedicated ephemeral pool) with no
signing capability, no model credentials, no repository-write credential.
Runs the preflight and predicate from pinned code, executes the
candidate's gates, and emits the report artifact. Because candidate code
executes in this job, **nothing Job 1 emits is trusted**: its report is a
proposal to be reconciled.

**Trusted recomputation and signing.** The signer is **external**: the
notary key never enters any runner. The trusted signing component (the
supervisor/broker service, extended with typed notary operations)
validates an independently authenticated caller identity via OIDC claims
(workflow path, job, run id, attempt, repository, ref) rather than
trusting runner-resident state, and then, itself or through a dedicated
trusted job running only pinned code on a fresh runner:

1. Reads the Job-1 run from the Actions control plane (workflow path and
   git oid, ref, run id/attempt, conclusion, artifact digest); wrong
   workflow, wrong ref, non-success conclusion, or artifact mismatch
   refuses.
2. **Recomputes every claim-bearing invariant from content-addressed
   inputs**: both tree manifests (with §2.1 totality rules), the diff,
   protected classification under the base-committed policy, lineage
   eligibility and signature validity, the unique coverage assignment,
   the unprotected enumeration, and chain-predecessor admissibility
   (§6/§7). Any divergence from the report refuses — this, not the
   report, is what the receipt's recomputed fields mean.
3. Validates the declared gates: one entry per profile-required gate id,
   no extras outside the profile, outcomes within the profile's
   acceptable set, tiers valid. Gate outcomes remain declarations (§1).
4. Requires `diff_coverage` exactly `"pass"` as recomputed. A refusal
   report — or a pass report that fails recomputation — is never
   signable, regardless of its workflow's success.
5. Reconstructs the §2.5 receipt body from typed fields and signs. It
   exposes no arbitrary-byte operation for the genesis, transition, or
   notary scopes.

**Human authorization approves a completed receipt, not a pipeline.**
GitHub environment approval gates a job before it starts, so the
recomputing job cannot be the approved job — reviewers would be
approving work not yet run, which the charter explicitly forbids. The
trusted flow is therefore three stages: the **recomputation stage**
(fresh runner, pinned code, unapproved) performs the reconciliation
above and publishes the immutable, content-addressed **receipt
candidate** — the completed reconciled body, unsigned; the **approval
stage** is the `approve` job in the dedicated `notary-signing`
environment (required reviewers, no self-approval, **administrator
bypass disallowed and audited** — GitHub's separate bypass setting must
be off, and §10 tests the bypassed case; the environment holds no
generation credentials and no write tokens, unlike today's
`production-signing`, which provisions generation workflows with the
apply key, a model credential, and a write-capable token, and is
structurally disqualified). Platform approval alone cannot bind bytes —
neither OIDC nor the jobs API authenticates job inputs — so the binding
is cryptographic: **a protected reviewer signs the candidate digest**
under its own scope, `axiom/notary-approval/v1` (a detached signature
by a reviewer key from §8's ceremony, distinct from every automation
key), and that signature is the durable, signer-verifiable approval
evidence. The **signing step** is the external signer, which validates
the `approve` job's OIDC identity, re-reads the candidate by digest,
**verifies the reviewer approval signature over that exact digest**,
and signs. An environment-bypassed job carries no reviewer signature
and refuses. Approval is thereby digest-bound to a completed receipt by
signature, not by platform semantics; key material never reaches a
runner. (This resolves the §11 approval-wording decision in the
stronger form: `authorization.approval_signature_sha256` is the
reviewer approval-signature reference.)

Effective-permission constraints, not deployment assumptions (§9): the
external signer's own deployment holds no model, generation, or
repository-write credentials, and its caller allowlist is exact — the
`approve` job alone, identified by repository, workflow path
and `workflow_sha`, job, run, attempt, ref, and the `notary-signing`
environment, with `id-token: write` allocated for the OIDC proof and
`actions: read` for control-plane lookups and nothing further — the
allowlisted caller of the **signing** operation is the `approve` job;
`publish` presents OIDC only to the separate publisher-token broker,
never to the signer. Job 1's
token is audited to `contents: read` with no other permissions, no
secrets, and no environment. The trusted job's `GITHUB_TOKEN` is
read-only and it references no secrets beyond the OIDC exchange. The
publisher's identity is denied the notary signing operation by that same
allowlist — and because `checks:write` is itself merge-authorizing, the
publisher is pinned candidate-free code that validates the signed
receipt and the current finalized predecessor before ever emitting the
required check.

**Publisher — separate job, two-keyed.** Holds two separately
constrained credentials per §9.7b, from **two distinct App
registrations** — a chain credential (contents-write
on the notary repository alone) and a lane credential (checks-write and
contents-read on the lane repository alone) — provisioned to it, never
minted by it; publishes per §7; holds no signing capability.

Trust roots for every step resolve through one normative committed
registry: `.axiom/notary/keys.json` at the base, schema
`axiom/notary-key-registry/v1` — a closed object mapping each role to
its sorted (by `spki_sha256`), strictly unique array of
`{spki_sha256, public_key_spki_der_base64}` entries: **the key bytes
themselves** (standard base64 with padding, RFC 4648 §4, of the DER
SubjectPublicKeyInfo; decoding must yield a valid Ed25519 SPKI whose
SHA-256 equals the beside fingerprint, else the registry refuses) —
because a fingerprint alone cannot verify a signature. Registry role names are exactly the §2.1 role-table roles —
`producer`, `actor`, `review`, `admin-approver`, `approver` — plus
`notary` (whose entry mirrors the SPKI pinned in the lane's consumer
verification spec; a mismatch refuses) and `corpus-release` (the
signed-corpus-release verification root the gate execution requires —
today provisioned from an organization variable, which §9.10
eliminates in favor of this registry entry; Job 1's toolchain
verification reads it from here). The three typed scopes —
`genesis`, `transition`, and `notary` — all resolve to the **`notary`**
registry entry: the external typed signer signs all three with the
notary key, and the admin-approver's separate signature is what
authorizes the administrative ones. **Role-key separation is total, not enumerated**: the SPKI sets of
all roles — the seven registry roles and `legacy_apply_root` — are
**pairwise disjoint**, and any SPKI appearing under two roles refuses
the registry. Four consecutive review rounds each found a pair a
sparse forbidden-pair matrix had missed, so the contract is the
universal rule, with no permitted alias left for an implementer to
weigh. Two consequences carry their rationale by name. A `notary`
alias would mean a private key outside the typed signer can freshly
sign the notary domain (domain separation prevents replay, not
signing). A `legacy_apply_root` alias would reopen pre-epoch
laundering: §9.15's freeze pins *which* key is the v5 root, never
*when* its signatures were made, so an aliased holder could freshly
sign v5 records over hand-edited bytes right up to the lane lock and
have genesis bless them as `v5_attested` — and the production signing
broker already rejects an apply/corpus-release root collision, so
totality restores deployed behavior rather than adding a constraint.
The same shape covers every other pair — an `admin-approver` alias of
`approver`, for one, would escalate ordinary reviewer compromise into
authority over trust-surface rotation. Whether one person may hold two
roles' keys is §8 custody policy; the registry contract is about keys:
no SPKI serves two roles. (The charter's separate-key requirement made
checkable; the cross-scope matrix alone cannot detect deliberate key
reuse across scopes.) For every other scope,
scope-to-registry resolution is the role table itself: a detached signature under a scope verifies
against the key bytes registered for that scope's role, never against
the signature file's self-declared fingerprint (which is
cross-checked, not trusted). The registry is a §4 trust surface: it
moves only by transition, which is what key rotation is.
Runtime organization variables are not trust anchors anywhere in the
notary path.

## 6. Admission chain, epoch, genesis, and transitions

### 6.1 The chain

Every receipt and transition names its chain predecessor (`genesis`,
`receipt`, or `transition`) and the predecessor's recorded state; the new
artifact's base tree manifest must equal that state. The chain consists
of **finalized** artifacts only (§7): an unfinalized or superseded signed
artifact is void and never chain-eligible. An uncovered edit cannot
launder itself by becoming a later verification's base: the state that
skipped it was never finalized as tip.

### 6.2 Genesis — `axiom/notary-genesis/v1`

Produced only through the typed signer under administrative authorization
(§6.4's authority). Its body binds: `schema`, `lane`,
`genesis_commit_git_oid`, `genesis_tree_manifest_sha256`, and two frozen
inventories forming an **exhaustive, disjoint, exactly-once partition of
the protected paths** at genesis — where **every protected entry must be
mode 100644**: the domain-total wall of §3.3 applies at genesis and
activation exactly as in ordinary coverage, so a pre-epoch hand chmod
cannot hide under a path/blob-only v5 record (current v5 applied_files
bind no modes) and then poison every successor's base (the typed signer verifies the
partition property against the manifest before signing):

- `bootstrap_policies`: an object with exactly four members —
  `{path_policy_sha256, transition_path_policy_sha256, profile_sha256,
  key_registry_sha256}` — the prospective policy and registry bodies §7
  binds; the ceremony's administrative key must be authorized by that
  prospective registry, verified by the typed signer at genesis (the
  bootstrap breaks the predecessor-roots recursion by binding its own
  root set, exactly as it binds its policies);
- `activation_spec_template_sha256`: the §7 epoch-placeholder template
  digest; `consumer_spec_path`: the lane path of the operative consumer
  verification spec file — the fifth bootstrap path is named, not
  guessed, so a digest-matching decoy elsewhere binds nothing; and
  `notary_repository`: the `owner/repo` string of the dedicated chain
  repository (all part of the closed genesis schema, so the epoch
  digest is uniquely constructible and the bindings are inside the
  schema, not beside it);
- `legacy_apply_root`: `{raw_key_id, public_key_spki_der_base64}` —
  the exact legacy apply public key that authenticates v5 records at
  genesis. `raw_key_id` follows the current formula normatively:
  `"sha256:" + lowercase-hex SHA-256 of the raw 32-byte Ed25519 key`;
  the raw key wraps into the DER SPKI for the second field, and the
  pair must be mutually consistent or genesis refuses. Binding alone
  proves consistency, not history — so §9 requires the historical
  root's fingerprint to be **frozen independently before genesis**
  (recorded in the sign-off packet and compared at ceremony time
  against the currently provisioned production value, witnessed), and
  the anti-reclassification claim is scoped to that precondition: given
  the independently frozen fingerprint, root substitution at genesis
  cannot reclassify pre-epoch hand edits as attested. One residual is
  inherent to grandfathering and stated rather than claimed away: a
  fresh v5 record signed under the *true* frozen root before the lane
  lock is indistinguishable from a historical one, so `v5_attested`
  trust is bounded by the legacy key's custody up to the lock — §5's
  total `legacy_apply_root` exclusion keeps every registry-role holder
  out of that window, and the lock closes it (a locked, pinned tip
  admits no new store files). The organization
  variable supplying this root today is eliminated by §9.10;
- `v5_attested`: sorted `[path, entry_sha256, record_sha256]` — paths
  whose blobs are vouched at genesis by a legacy record that passes the
  **current v5 authentication contract**: schema exactly
  `axiom-encode/applied-rulespec/v5`, producer signature valid over the
  canonical unsigned bytes, encoder identity and waiver checks per the
  pinned v5 verifier, and path/blob agreement with the genesis tree.
  `record_sha256` is SHA-256 over the record's raw file bytes. Two
  qualifying records for one path is a genesis refusal (clean up first).
- `baseline_unattested`: sorted `[path, entry_sha256]` — everything
  else, visibly, including any pre-epoch hand edits. Recorded is not
  blessed: permanent "unattested baseline" provenance.

The epoch identifier is the genesis body's content digest; the body
carries no epoch field, so there is no self-hash fixed point. A second
genesis for a lane refuses. Genesis covers nothing and admits nothing.

### 6.3 Dual-era rule

Post-epoch, a v5 record vouches only for a path whose blob is
byte-identical to its frozen `v5_attested` pair. Any post-epoch change to
any protected path requires notary-era lineage; new v5 records have no
post-epoch standing; `baseline_unattested` paths stay unattested until a
covered change replaces them.

### 6.4 Transition records — `axiom/notary-transition/v1`

Trust-surface updates (path policy, profile, workflow or action pins,
trust roots, waiver policy, key rotation) cannot ride ordinary receipts —
the preflight refuses them — so the transition record exists to advance
the chain across them. Closed schema: `schema`, `lane`, `epoch_sha256`,
`chain_predecessor_sha256`, `chain_predecessor_kind`,
`base_tree_manifest_sha256`, `subject_tree_manifest_sha256`,
`subject_commit_git_oid`, `delta`: sorted array of
`{path, before_entry_sha256 | null, before_mode | null,
after_entry_sha256 | null, after_mode | null}` restricted to
trust-surface paths, and `reason`.

Rules: a transition whose delta is empty, or whose base manifest
equals its subject manifest, refuses — the state-identical ban of §1
covers transitions and receipts alike, and the signer and publisher
both enforce it; transition eligibility has a deterministic oracle — a
committed
`.axiom/notary/transition-path-policy.json` (same rule shape and
semantics as the path policy, schema
`axiom/notary-transition-path-policy/v1`), evaluated under the
**predecessor** state, enumerating the trust-surface paths (workflows,
action pins, verifier pins, trust roots, the three policy files, waiver
policy, repository-structure rules); the recomputed base→subject diff
must equal `delta` exactly, every delta path must match the
transition-path policy, and no delta path may be protected content — a
protected-content or unclassified change smuggled into a transition
refuses. The two policies have opposite obligations toward the
reserved prefixes: the **path policy** (protected-content domain) is
invalid if it protects any `.axiom/lineage/` or `.axiom/notary/` path;
the **transition-path policy** must **include** the `.axiom/notary/`
trust files — that is precisely what transitions rotate — and is
invalid if it covers `.axiom/lineage/` (the store moves by eligibility,
never by transition);
signatures and authorization are evaluated **under the predecessor
state's trust roots** (the old keys and policy authorize the handover, so
a successor-controlled key cannot self-authorize its own installation);
transitions (and genesis) follow the same digest-bound authorization
pattern as receipts: the typed signer emits the **administrative
candidate** (the exact genesis or transition body, unsigned), the
holder of the predecessor-state **administrative key** — per the §2.1
role table, never the correction-reviewer key — signs its digest under
`axiom/notary-admin-approval/v1` (for genesis, the administrative key
named by the bootstrap ceremony), and the signer signs only after
verifying that approval signature over the exact digest — environment
approval alone binds no bytes here either, and a swapped-body
transition after platform approval refuses on the missing
matching-digest signature; and transitions are void
until finalized (§7), exactly like receipts. For merge gating, a pending
transition plays the pending receipt's role for its own subject (§7): it
is the merge-authorizing artifact for exactly its enumerated delta.
Ordinary admission resumes from the finalized transition's recorded
state.

## 7. Two-phase publication and the canonical chain

The canonical chain lives on a **protected branch** — `refs/heads/chain`
— in a **dedicated notary repository** (`<lane-repo>-notary`, bound by
name into genesis and the lane's consumer verification spec). The
separate repository is the capability boundary rulesets cannot provide
inside the lane repo: a Contents-write App token is repository-scoped
and can always manipulate qualified custom refs of the repository it
holds, so single-ref confinement inside the lane repository is not
implementable — whereas the **chain App's** contents-write
installation covers only the notary repository, and in the lane
repository the publisher holds a credential from the **distinct lane
App** (§9.7b: a separate registration with its own private key), whose
installation holds `checks: write` and
`contents: read` — sufficient for source-bound required checks, which
branch protection binds by the creating App's id; no
Commit-Status-path permission is taken, because an unused
merge-authorizing permission is attack surface — with runtime tokens
downscoped per operation. The lane repository's refs are
physically outside both Apps' write capability. The chain branch's
ruleset in the notary repository: only the chain App may push; no
force pushes; no deletion; linear history. Its contents: content-addressed artifact files (reports,
receipts, transitions, genesis, and void markers) plus `HEAD.json`
(`{schema: "axiom/notary-head/v1", tip_sha256, tip_kind}`) as a
convenience pointer. Two marker schemas complete the branch's contents:
`axiom/notary-finalization/v1` (`{schema, lane, epoch_sha256,
target_sha256, target_kind, merged_tip_manifest_sha256, sequence}`) and
`axiom/notary-void/v1` (`{schema, lane, epoch_sha256, target_sha256,
target_kind, reason}`), each content-addressed like every other
artifact. Marker authentication is structural: the branch accepts
commits from the chain App alone, linear history, no force pushes —
so markers are exactly what the pinned publisher committed, in order.

The chain state machine, exhaustively: an artifact is *pending* when its
body is on the branch with no marker naming it; *finalized* when exactly
one finalization marker names it; *void* when a void marker names it.
Finalized and void are terminal and mutually exclusive — the first
marker in branch history wins, and any later marker naming the same
target is invalid, rejected by reconstruction. A finalization whose
target was never pending on the branch, whose `target_kind` mismatches,
or whose base does not equal the then-current finalized tip is invalid.
Genesis bootstraps the branch: the branch's first two commits must be
exactly the genesis body and its finalization marker, published by the
same CAS branch creation — a competing genesis loses the atomic
first-push and any later genesis body or marker is invalid by the
second-genesis rule. **Bootstrap is explicitly acyclic, policy-bound, and freeze-audited.**
No writer in this design can atomically couple a lane read with a
notary-repository branch creation, so the freeze is administrative and
audited rather than claimed as CAS: before genesis signing, the lane's
protected branch is **locked** (a lock ruleset barring all merges), the
signer verifies via the API both that the lock is active and that
`genesis_commit_git_oid` equals the locked tip, and the lock is
retained until activation finalization. A locked branch is read-only —
including to the activation merge — so the lock names its **sole,
audited bypass**: the bootstrap administrative actor, whose one
permitted action is merging the activation commit, listed in §9 as a
security-critical authority. The guarantee does not rest on that
actor's discipline: activation refuses unless the recomputed
base→subject state installs exactly the five bootstrap paths — the
epoch-pinned consumer spec, the three policies, the key registry —
**defined by final installed values, not by diff membership**: each
path's subject bytes must equal its genesis-bound digest (with the
epoch substitution for the spec), whether or not the path already held
those bytes at genesis, and no other path may change. A bypass misused
for any other merge produces an activation that cannot finalize and a
bootstrap that voids. A genesis attempt observing a moved tip or an absent lock
refuses.

Genesis also **binds its prospective policies and the activation
bytes**: the body carries `bootstrap_policies` — the digests of the
exact path-policy, transition-path-policy, and profile bodies the lane
will adopt — and `activation_spec_template_sha256`, the digest of the
prospective consumer-spec file with the epoch value replaced by the
fixed placeholder token
`"0000000000000000000000000000000000000000000000000000000000000000"`
(the epoch cannot be bound literally: it is the genesis digest itself).
Activation's spec file must equal that template with the placeholder
substituted by the actual epoch — byte-exact, mechanically checkable —
so no other field of the consumer-spec file (a notary SPKI, an anchor
pin) can ride the activation delta. The genesis partition of protected
paths is computed under the bound path policy, so the policy defining
genesis provenance is fixed at signing and cannot be selected
afterward. The lane's epoch pin cannot
live inside the genesis tree (that would be `epoch → tree → body →
epoch`); it lands immediately after, in the **bootstrap activation
transition** — the chain's second artifact, predecessor genesis, whose
delta installs the five bootstrap paths by final value — each
byte-equal to its genesis-bound digest (epoch substituted in the spec),
paths already correct at genesis permitted to be absent from the diff,
no other path changing; its eligibility oracle
is the genesis-bound transition-path policy. Enforcement begins at
activation finalization; the lock spans the whole interval, so no
ordinary candidate can slip between. The stale-genesis laundering path
is closed by the audited freeze: genesis is valid only over the locked
tip, and the activation transition chains from it before anything else
may.

**Verification does not trust the pointer — and the branch carries its
own preimages.** Genesis publishes the full policy, profile, and
key-registry bodies (not digests alone) beside its body on the chain
branch, and every transition that changes them publishes the new
bodies likewise, so administrative approvals and historical keys
verify from the branch alone after any number of rotations. The chain's truth is
reconstructible by rule from the branch alone: the valid chain is the
unique sequence of signed artifacts starting at genesis in which each
artifact's base manifest equals its predecessor's subject manifest, each
was finalized by an on-branch finalization commit, and no void marker
names it. Because the ruleset forbids force pushes and deletions, a
rollback, a void erasure, or a pointer rewrite is either blocked outright
or visible as history the reconstruction rejects. `HEAD.json` diverging
from the reconstruction is itself a refusal.

Admission is two-phase, resolving the ordering circularity between
"receipt must exist to authorize the merge" and "the merged state must
equal what was verified":

1. **Pending.** Job 1 verifies the candidate head (subject tree `T1`);
   the signer signs the receipt (or transition). The publisher commits it
   to the chain branch as *pending* and sets the required status check on
   the candidate — a check whose required context is **bound to the
   lane App** in branch protection, so a same-named context from
   Actions or any other integration does not satisfy the requirement. A
   pending artifact authorizes exactly one thing: the merge of a head
   whose tree manifest is `T1` onto the finalized tip it names as base.
2. **Freshness.** The protected content branch requires strict
   up-to-date-with-base merges (or a merge queue whose `merge_group`
   head is itself verified). A pending artifact binds its base manifest;
   when another candidate finalizes first, the stale candidate's base no
   longer matches the finalized tip, the strict requirement forces a new
   head, and the superseded pending artifact is voided by the publisher
   rather than left green. The race sol named — two green pendings, the
   second merging onto a moved base — is closed before merge, not
   discovered after.
3. **Finalize.** After the merge, the finalize workflow *requests*
   finalization with its OIDC identity, and the **publisher-token
   broker** — per §5 the sole holder of the chain credential —
   recomputes the merged tip's tree manifest. Finalization requires all of: the merged tip manifest
   equals the artifact's subject manifest; **the artifact's
   `chain_predecessor_sha256` and kind equal the current finalized tip
   exactly** — manifest-state agreement is not enough, because two
   pending artifacts for the same tree naming the same predecessor are
   distinct signed statements and only one may enter the chain; and the
   marker's `lane`, `epoch_sha256`, `target_kind`, and
   `merged_tip_manifest_sha256` validate against the artifact. On
   success the finalization marker commits with the next `sequence` — a
   finalization-only counter: canonical decimal string, no leading
   zeros, genesis's finalization is `"1"`, each subsequent finalization
   exactly one greater; a gap or repeat invalidates the branch to
   reconstruction; void markers carry no sequence and order by branch
   history and
   `HEAD.json` advances — compare-and-swap via the atomic branch
   update. **All other pending artifacts naming the superseded tip as
   predecessor are voided in the same publisher pass** (same-state
   siblings included). Otherwise finalization refuses, the artifact is
   permanently void (a void marker commits to the chain branch; a
   voided digest can never finalize), and verification reruns from the
   true finalized tip. A pending artifact whose candidate never merges
   simply never finalizes; the chain never advanced, so nothing
   strands.

Squash and merge-queue rewrites are therefore harmless exactly when they
preserve the verified tree, and void the attempt exactly when they do
not. Commits locate; trees bind.

## 8. Key ceremony

Before the typed signer holds any notary-scope key: generate
`notary_ed25519` with documented custodians, fingerprint published in
committed code (consumer verification specs pin the SPKI), storage and
rotation procedure (rotation is a §6.4 transition evaluated under
predecessor roots), revocation and recovery. The producer, actor,
reviewer, and administrative keys get the same treatment under their own
scopes. §10's pairwise cross-scope matrix is part of ceremony acceptance.

## 9. Preconditions (nothing admission-capable merges before these)

1. This document approved — threat model, schemas and envelope,
   predicate, profile, chain/epoch/genesis/transition rules, two-phase
   publication, and the §10 suite — by the charter's gate: independent
   cross-family review plus **Max's named sign-off**, with every §11
   decision closed.
2. The reviewers-semantics resolution in §4 accepted.
2a. The bootstrap authority named and audited: the single administrative
   actor holding the lane-lock bypass, its custody, its one permitted
   action (merging the activation commit), and its removal at
   activation finalization — a security-critical, single-use authority.
2b. The `notary-signing` environment's administrator-bypass setting
   disallowed and audited (a bypassed approval carries no reviewer
   signature and refuses regardless, but the platform gate is
   configured off, not merely compensated for).
2c. The `notary-publishing` environment created with its administrator
   bypass disallowed and audited, and the **publisher-token broker
   deployed** as a distinct service (no notary key, no signing
   operation, only the two App credentials) vending against exact OIDC
   identity — environment secrets are not the mechanism, because
   environments gate reviewers and refs, not workflow identity.
3. The dedicated `notary-signing` environment created (required
   reviewers, protected refs, no self-approval), holding no generation
   credentials and no write tokens; generation workflows migrated off
   any environment involved in notary authorization (#1194 re-scoped).
4. **External signer with identity binding**: the notary key held only
   by the external typed signer; OIDC-claim validation of the requesting
   repository, `workflow_ref` and `workflow_sha`, `check_run_id`
   (reconciled via jobs-for-run to the expected trusted-job name and
   attempt), run, attempt, ref, and `notary-signing` environment,
   against an exact allowlist; reusable workflows prohibited for Job 1
   and the trusted job; `id-token: write` and
   `actions: read` allocated explicitly and nothing further; no raw
   notary key material on any runner; Job 1's token audited to
   `contents: read` with no secrets and no environment.
5. **Compute isolation**: fresh ephemeral runners for the verification
   and trusted jobs.
6. Trusted recomputation implemented (§5): typed operations for
   genesis, transition, receipt, **and finalization** (the last a
   broker-side validate-and-CAS, never a vended credential); every
   claim-bearing invariant
   re-derived from content-addressed inputs before signing; typed
   operations for genesis, transition, and receipt scopes; no raw-byte
   signing for those scopes.
7. The publisher split implemented: the chain on its protected branch
   with an App-restricted, no-force-push, no-deletion ruleset; the
   required admission check bound to the lane App (a same-named
   context from any other source must not satisfy it); strict
   up-to-date-with-base merges (or verified merge-group heads) on the
   protected content branch; and the two-phase pending/finalize protocol
   with supersession voiding and permanent void markers.
7a. Effective-permission constraints audited per §5: signer deployment
   credentials, exact caller allowlist, Job-1 and trusted-job token
   audits, and publisher signing denial.
7b. Publisher write confinement is physical and two-keyed: **two
   distinct App registrations with separate private keys and disjoint
   installation permissions** — token downscoping is not an
   alternative, because the App credential that mints a downscoped
   token can equally mint an undownscoped one across everything its
   installation grants, and the audit therefore covers the **root App
   permissions**, not only delivered tokens: a chain App with
   contents-write on the notary repository alone, and a lane App
   with an installation holding `checks: write` and `contents: read`
   on the lane repository alone (no Commit-Status permission — branch
   protection binds the required check by App id), runtime tokens
   downscoped per operation.
   The publisher job holds no App-minting authority — tokens are
   provisioned to it, never minted by it. The publisher runs pinned
   candidate-free code and validates **every merge-authorizing
   artifact** — receipt or transition alike — signature, typed fields,
   and current finalized predecessor under predecessor-state roots,
   before emitting the App-bound required check.
8. Signed-leg billing and abuse policy operational (#1193).
9. Key ceremony completed per §8.
10. Org-variable trust anchors eliminated from the signing path.
11. rulespec-nz consumer-side pins landed: notary SPKI and epoch in the
    lane's committed verification spec; the pending-artifact check
    required and non-bypassable in branch protection; genesis activated
    atomically with enforcement.
12. The generated-file guard hardcoded in the protected shared workflow
    (caller boolean retired) — merge-time defense in depth during
    dual-era operation.
13. Retirement of lane signed-apply legs (#1195) is a cutover exit
    criterion, not a pilot precondition.
14. Golden-regeneration QA retained and audited — the charter keeps it
    as the distributional defense, §1 relies on it as a mitigation, and
    its removal fails this precondition.
15. The legacy apply root's fingerprint frozen independently before
    genesis (sign-off packet record, compared witnessed against the
    production value at ceremony time), and the corpus-release root
    migrated from its organization variable into the registry's
    `corpus-release` entry — Job 1's toolchain verification must read
    it from the registry before any lane activates.
16. Genesis-time equality refusals: `legacy_apply_root` equal to
    **any** registry entry refuses — `approver` matters concretely
    because legacy signed-apply retirement is not a pilot
    precondition, so an aliased apply authority could forge receipt
    approvals; `corpus-release` matters because an aliased service-key
    holder could freshly sign v5 records over hand-edited bytes before
    the lane lock and have genesis bless them (§5's window argument —
    and the production broker already rejects this collision); the
    registry validator enforces §5's total pairwise disjointness.

## 10. Negative-test floor

Grouped by refusal site; each case is a distinct test before the pilot
gates anything.

**Schemas, encodings, signatures:** unknown/missing field; wrong type;
duplicate JSON member; invalid Unicode in a body; number where string
required (`run_id`); digest with wrong case, length, or retained
`sha256:` prefix; unsorted semantic array (gates, records, assignment,
inventories); wrong content-address filename; malformed
detached-signature file; wrong scope per the role table (review
signature under the correction scope and conversely); signature by the
wrong key for a scope; invalid producer/actor/review/genesis/transition/
receipt signature; countersignature by the actor; raw-byte signing
attempt against genesis, transition, or notary scopes; full pairwise
cross-scope matrix including legacy operations.

**Manifests and totality:** gitlink anywhere in the tree; non-UTF-8 path
anywhere in the tree (changed or unchanged); inadmissible terminal mode;
symlink inside the protected domain; any symlink anywhere refused (charter requirement 4, tree-wide in the
pilot); manifest sort-order violation; two trees
differing only by a gitlink refused rather than treated as
manifest-equal.

**Lineage store:** mutated existing record; deleted existing record;
wrong lane; wrong epoch; mode present on a null side; mode outside the
admissible set.

**Eligibility and coverage:** record present at the base (replay);
uncovered protected change; protected mode-only change refused
(charter wall); overlapping
chains; transition against an unchanged path; forked, cyclic, or
discontinuous chain; wrong starting or terminal blob/mode; malformed
null-sides; partial record consumption; duplicate transition paths
within a record; record containing an unprotected-path transition
(ineligible, with reason); **ambiguous assignment (two redundant
covering records; split-vs-direct chains)**; dead-end retry record
accepted as unused without ambiguity (positive control); the cross-path
record-order cycle (two records feeding each other on different paths)
refusing as record-cycle; correction predecessor naming a nonexistent
record; correction predecessor mismatching the preceding consumed
record; `unused_eligible_records` incomplete (a consumed record listed,
or an unused one omitted); nested
protected change reported only as its directory by a non-recursive diff
(must refuse or be impossible under the normative diff);
`waived`/`not-run` post-epoch; path-policy precedence cases
(include-then-exclude; `rules-evil/x`; unmatched default-unprotected).

**Report reconciliation (trusted side):** forged pass report over a
refusing diff; trimmed, padded, duplicated, or misclassified
`unprotected_changes`; wrong or non-unique `coverage_assignment`;
`unused_eligible_records` not a subset or overlapping consumed records;
duplicate gate ids; missing required gate; extra-profile gate; outcome
outside the acceptable set; invalid tier; report/receipt copied-field
mismatch; tree-manifest mismatch on recomputation; profile or
path-policy digest mismatch at the base.

**Preflight and profile:** ordinary candidate touching any §4 trust
surface; repairable-but-unrepaired subject; oracle-disabled attempt;
skip-flag attempt; profile or policy not committed at the base.

**Chain, epoch, genesis, transitions:** base not the finalized tip;
wrong chain predecessor digest or kind enum; second genesis; genesis
with a non-exhaustive, overlapping, or duplicate-path partition; genesis
entry citing a record failing the v5 contract (wrong schema family,
invalid signature, wrong encoder identity, wrong waiver binding,
path/blob disagreement); transition whose enumerated delta contains a
non-trust-surface path; nonexistent
`record_sha256`; duplicate qualifying v5 records for one path; genesis
carrying coverage; state-identical receipt (base manifest equals subject manifest)
refused; mode-preserving transition rule refusing a
100644→100755→100644 chain at its first link; protected 100644 addition and
deletion admissible (positive controls — the domain-total wall refuses
100755 anywhere, additions included); post-epoch change vouched only by v5; v5
record whose blob differs from its frozen pair; transition whose
recomputed diff differs from its enumerated delta (smuggled content);
transition evaluated under successor roots (self-authorizing rotation);
unfinalized transition used as predecessor; ordinary receipt attempting
a trust-surface change.

**Signer, publication, finalization:** OIDC identity mismatch (wrong
workflow, run, repository, ref, or environment); Job-1 job conclusion
non-success via the jobs-for-run lookup; duplicate artifact name within
the run (run-scope uniqueness violated); artifact digest mismatch;
replayed artifact under a new run; signing a refusal report; stale base
at recomputation; pending artifact treated as chain tip; superseded
pending artifact voided when another candidate finalizes first (positive
control) and refused if presented after; required-check context from a
non-App source (same-named Actions check) not satisfying branch
protection; merge-group head differing from the verified subject;
finalization with merged-tip manifest differing from the verified
subject (voids, permanently); voided digest re-finalization attempt;
finalization when the artifact's base is not the current finalized tip;
chain-branch force-push and deletion rejected by ruleset; unauthorized
chain rollback or void-marker erasure visible to reconstruction;
`HEAD.json` diverging from chain reconstruction; publisher writes outside the chain branch of the notary repository are
not ruleset-preventable (contents-write is repository-scoped) and are
harmless by construction — reconstruction reads only the chain branch —
tested as: a non-chain ref in the notary repository is ignored by
reconstruction; `HEAD.json` naming an unpublished or unsigned
artifact; recomputation divergence in `eligible_records`,
`corpus_release`, `waiver_set_sha256`, `dependency_pins_sha256`, or the
`verifier` identity; malformed finalization or void marker;
finalization without a pending target; void-after-finalize rejected
(first marker wins); conflicting markers for one target; finalization
whose base is not the then-current tip; competing genesis publication
losing the atomic branch creation; wrong `workflow_sha` (head_sha
substituted, or any value diverging from the trusted job's OIDC
claim); signer-environment or attempt binding mismatch; profile-tier
strengthening impossible by construction (tiers absent from reports —
positive control); duplicate flattened tree path refused; same-state
sibling pending artifact voided at finalization (positive control) and
rejected if presented later; marker with wrong lane, epoch, target
kind, or tip manifest; marker sequence gap or repeat invalidating the
branch; transition delta path unmatched by the transition-path policy;
transition validated by the publisher under predecessor-state roots
(positive control) and rejected when its signature or predecessor
fails; empty subtree entry refused; genesis attempt without an active
lane lock, or against a tip that moved, refused (stale-genesis
laundering closed); lane movement between genesis signing and
publication impossible under the audited lock (positive control);
ordinary candidate between genesis and activation refused; activation
transition changing any path outside the five named bootstrap paths
refused; activation leaving any bootstrap path's installed bytes
unequal to its genesis-bound digest refused (installed values, not
diff membership); activation installing a policy file whose
digest differs from bootstrap_policies refused; genesis with absent
bootstrap_policies refused; wrong
`check_run_id` or a trusted job in a reusable or separate-run topology
refused; duplicate profile gate ids and duplicate inventory
ref_spec/image keys refused; `workflow_ref`/`ref` suffix mismatch
refused; publisher lane-content write attempt failing (two-credential
split, positive control); publisher job holding App-minting authority
forbidden by audit; one App registration serving both publisher
credentials, a shared App private key, or a root installation granting
repositories or permissions beyond its enumerated scope, each refused
by the §9.7b audit; publisher identity attempting the notary signing
operation refused; a non-publisher updating the chain branch, or
non-linear history, rejected; void marker carrying a sequence refused;
genesis finalization sequence not "1" refused; sequence "01" or
non-decimal text refused; fsck-dirty tree refused beyond the
duplicate-path case; absent, non-regular, or oversized waiver file
refused; refusal report carrying an assignment or stage-unestablished
fields refused; approval-stage input digest mismatching the receipt
candidate refused; signing without a digest-bound approval refused;
invalid OIDC signature, issuer, audience, or expiry refused; unknown
stage, refusal-code, or ineligible-reason value refused; multi-fault
ineligible record represented by its full sorted reason array (positive
control); wrong `baseline_unattested` entry digest refused; genesis
commit/tree-manifest mismatch refused; wrong or intervening first chain
commits refused; premature lock removal or an unauthorized bootstrap
bypass merge producing an activation that cannot finalize (voids);
environment-admin approval bypass carrying no reviewer signature
refused; reviewer approval signature over the wrong digest refused;
`"gates"`-stage refusal carrying its assignment (positive control);
approval signature invalid, by the wrong key, or under the wrong scope
refused; approval signature over a different candidate digest refused;
receipt whose candidate_sha256 mismatches the recomputed candidate
refused; activation consumer-spec differing from the genesis-bound
template beyond the epoch substitution refused (same-file smuggling);
replay over the whole tree instead of the protected projection would
reject a valid candidate (positive control for the projection rule);
a finalization request from a wrong workflow, ref, run, or
environment refused by the broker (no finalizer credential is ever
vended — finalization is a broker-side operation);
publisher-token broker asked to sign, or signing broker asked for a
write token, refused (capability separation of the two brokers);
publishing-environment administrator bypass tested off; signature
validated against a self-declared fingerprint instead of the registry
refused; key absent from the registry role refused; registry change
riding an ordinary candidate refused (trust surface); candidate
workflow_sha diverging from approve's OIDC claim refused; wrapper
lane or epoch diverging from candidate or chain refused;
authorization.environment not derived from OIDC refused; untyped or
empty lineage identity fields refused; empty correction `reason`
refused; simultaneous-fault refusal
reporting the first rule, least path, and template detail (positive
control); state-identical transition (empty delta) refused by signer
and publisher; delete-and-recreate chain through a differing
intermediate mode refused (chain-wide agreement); registry entry whose
key bytes hash to a different spki_sha256 refused; scope resolved to
the wrong registry role refused; genesis over a locked tip whose
ceremony key is absent from the genesis-bound prospective registry
refused; activation leaving any bootstrap path's installed bytes
unequal to its genesis-bound digest refused; bootstrap path already
byte-correct at genesis absent from the activation diff accepted
(positive control — installed values, not diff membership);
generation-side SPKI colliding with an admission-side role refused;
registry key bytes failing base64, DER, Ed25519, or fingerprint
validation refused; protected 100755 entry refused at base, subject,
and inside a transition (domain-total wall); delete-then-recreate-
executable across two finalized receipts refused by the same rule
(positive control); **a table-driven collision suite over every unordered role pair** —
§5's separation is total, so the table is the full pairwise matrix
over the seven registry roles plus `legacy_apply_root`: all
twenty-eight pairs distinct refusal cases (`notary`×`review`,
`admin-approver`×`approver`, and `corpus-release`×`legacy_apply_root`
included) — with an all-distinct registry accepted (positive
control); finalization requested by rotated not-yet-finalized workflow
code validated and executed by the broker, not the workflow (positive
control); broker refusing finalization when the pending artifact,
predecessor, or merged tree fails validation; clean-room chain
reconstruction succeeding from the branch alone after a key rotation
(positive control — published preimages); missing or wrong historical
policy, profile, or registry preimage on the branch failing
reconstruction; genesis or activation containing a 100755 protected
entry refused; cross-receipt correction
with a null predecessor and the amended record named in reason
accepted (positive control); noncanonical decimal id ("01") refused; invalid
RFC 3339 emitted_at refused; registry notary entry differing from the
consumer pin refused; waiver bytes disagreeing with the base toolchain
pin refused; equal-manifest transition carrying a forged nonempty
delta refused; bad newly-introduced record enumerated ineligible while
the candidate passes (positive control — structural scope);
non-hex or uppercase store filename enumerated
unrecognized-store-name under its literal store_name (positive
control — total keying); a body-grammar name whose stem differs from
its raw-byte digest enumerated address-mismatch; an honest body at
`<digest>.json` accepted — the stem is the address, the suffix is
grammar (positive control); an orphan sidecar, a sidecar naming a
base-present body, a nested store name, and a non-lineage role suffix
each enumerated unrecognized-store-name; the sidecar of an eligible
newly-introduced body not separately enumerated (positive control);
transition-path policy covering .axiom/notary accepted and covering
.axiom/lineage refused (the opposite obligations); corpus-release root
absent from the registry refused; invalid corpus-release signature
refused; raw_key_id/SPKI pair inconsistent refused; legacy root equal
to any registry entry refused (each role a distinct test,
corpus-release and producer included); finalized receipt whose
referenced report is unpublished invalid to reconstruction;
absent-profile preflight refusal carrying a truthful null digest
(positive control);
base-present invalid record inert — successor with an honest diff
passes (positive control — no lane poison); chain-reconstruction
failure refusing chain-unresolvable with a truthful all-null
established prefix (positive control); absent subject commit refusing
subject-unresolvable with the three chain fields established (the
chain resolved first); malformed or invalid-to-reconstruction
predecessor artifact refusing chain-unresolvable — distinct from
predecessor-stale, which requires a resolved chain;
established-prefix violation refused;
wrong-address new record enumerated address-mismatch; malformed new
record enumerated malformed-record; absent Job-1 artifact refused;
reusable approve, publish, or finalizer job refused; unsorted or
duplicate key-registry role array refused; consumer-spec decoy at a
non-named path binding nothing (positive control); v5 record validated
against a key other than the genesis-bound legacy root refused (root
substitution); alias occurrence ineligible with address-mismatch while
the correct-path occurrence stays eligible (positive control);
malformed body carrying only computable reasons (positive control);
late structural failure carrying the resolved base prefix (positive
control); reconciliation rejecting an omitted or spurious
ineligible_records occurrence, or an omitted or extra applicable
reason;
reusable verify job refused; duplicate prompt_sha256s refused; empty
or wrong pinned artifact_name refused; review/admin-approver SPKI
collision refused (a member of the total suite); policy
protecting .axiom/lineage or .axiom/notary refused as policy-invalid; intermediate replay projection with a path both
terminal and directory-prefix refused as no-valid-execution;
noncanonical JCS bytes or wrong semantic-array order refused; genesis
inventory containing a path outside the protected domain refused;
genesis or transition lacking a matching-digest administrative approval
signature refused (swapped-body-after-platform-approval); path policy
attempting to protect `.axiom/lineage` refused; bootstrap bypass
authority retained after activation finalization failing the §9.2a
audit; administrator bypass of the required lane check tested;
receipt with unpublished candidate or approval file invalid to
reconstruction; sorted-but-duplicate entries refused in each set-valued
array (acceptable_outcomes, eligible_records, unused_eligible_records,
ineligible_records, reasons); protected-path mode change refused
(charter wall, pilot); publisher token request from a revoked or rerun
workflow identity vended nothing; well-typed but
invalid path-policy action or profile oracle_policy refused; Actions
artifact_id mismatch refused; multi-fault ineligible record carrying
one deterministic sorted reasons array in the pass variant (positive
control).

## 11. Decisions for sign-off

- ProgramSpec scope: atomic RuleSpec only in the pilot (recommended), or
  extend the path policy to composition outputs.
- Licensed or unavailable oracles: fail closed, or visibly reduced-tier
  receipt.
- Approval wording — resolved in §5's stronger form:
  `authorization.approval_signature_sha256` binds durable
  digest-bound reviewer evidence, or records "the protected signing
  policy authorized this receipt" (honest for plain environment
  approval; the stronger form needs an explicit approval artifact).
- Custody model for the producer, actor, reviewer, and administrative
  keys (the notary key is fixed by §5/§8); reviewer custody is the open
  question deferred from the rulespec-nz custody ruling.
- Charter alignment on modes: whether covered executable-mode
  transitions become admissible post-pilot (charter requirement 4
  amendment) or the wall stays permanent.
- Newly protected paths: when a transition expands the path policy, the
  newly covered paths' current entries are inventoried in the transition
  body and carry "transition-initialized" provenance (administrative,
  visible, distinct from v5-attested and unattested-baseline) — the
  recommended semantics; alternative: require such paths to enter empty
  and be populated by covered changes.

## 12. Out of scope for milestone one

Witnessed lineage chains (dual RFC 3161 — sequenced behind the notary as
chartered); historical backfill; rename modeling (tree-entry
decomposition makes it unnecessary); gitlink/submodule support (refused
tree-wide in the pilot); fleet-wide shared-workflow conversion; v5
retirement; the other eight lanes; ProgramSpec admission unless §11
decides otherwise.
