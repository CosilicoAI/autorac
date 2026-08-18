# Notary admission: design v4

Status: draft for sign-off. Implements the #1192 charter with the #1506
diff-coverage delta, under the build decision recorded on both issues
(dual-verdict, 2026-08-17). Version 4 folds design-review rounds 1–3
(twenty-five blocking findings; the record lives on #1507). Nothing
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
> protected-content tree-entry change from `T0` to `T1` is covered by the
> unique valid assignment over the eligible lineage set; every gate the
> profile requires is declared with an outcome the profile accepts;
> unprotected tree-entry changes are enumerated in `R`; and the protected
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
predecessor are covered. It never upgrades the provenance of older bytes.
Pre-epoch content remains exactly what the genesis inventories say —
v5-attested or unattested baseline — no matter how many receipts later sit
above it; an empty-delta receipt admits nothing and changes no provenance
status.

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

**Serialization.** Bodies are canonical JSON (receipt-canonical
serialization: UTF-16 code-unit key order, ECMAScript number formatting),
UTF-8 bytes, content-addressed by SHA-256 of the body bytes. Every body
carries `schema` and `lane`. Schemas are closed-world: an unknown field,
missing field, or wrong type is a parse refusal.

**Value encodings, normatively.** One table, no exceptions:

| Kind | JSON encoding |
|---|---|
| SHA-256 digest (`*_sha256`, filenames, broker payload) | 64-char lowercase hex string, no prefix; GitHub artifact digests have their `sha256:` prefix stripped before entry |
| Git object id (`*_git_oid`, incl. `workflow_git_oid`) | 40-char lowercase hex string (SHA-1 object format in the pilot) |
| Ed25519 signature | `signature_base64`: standard base64 with padding (RFC 4648 §4) |
| `signer_spki_sha256` | SHA-256 of the DER-encoded SubjectPublicKeyInfo, hex as above |
| `run_id`, `run_attempt` | JSON strings (decimal), never numbers |
| `chain_predecessor_kind` | exactly one of `"genesis"`, `"receipt"`, `"transition"` |
| Paths | UTF-8 strings; sorting is bytewise over the UTF-8 encoding |

**Tree manifests.** A tree manifest binds **every entry** reachable from a
tree, not only blobs: the canonical JSON array of
`[path, mode, entry_sha256]` sorted bytewise by path, where `entry_sha256`
is SHA-256 over blob content for blob modes and over the UTF-8 target
string for symlinks. Pilot totality rules, applied to **whole trees** at
verification, genesis, and finalization — not merely to diffs: a gitlink
(mode 160000) anywhere in the tree is a refusal; a non-UTF-8 path anywhere
in the tree is a refusal; modes outside {100644, 100755, 120000} are a
refusal, and 120000 only outside the protected domain. Within these rules
the manifest is a complete binding of the tree: two trees with equal
manifests are entry-identical.

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
| notary | `axiom/notary-receipt/v1` | notary receipt |

Signatures are detached files beside the body (`<digest>.json`,
`<digest>.json.<role>.sig`), each canonical JSON:
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

Signed by the producer key (non-authorizing). Fields: `schema`, `lane`,
`epoch_sha256`, `runtime_identity`, `model`, `cli_version`, `cli_sha256`,
`prompt_sha256s`, `emitted_at` (informational, unverified), and
`transitions`, sorted bytewise by path:

```
{path, before_blob_sha256 | null, before_mode | null,
 after_blob_sha256 | null, after_mode | null,
 patch_note_sha256 | null}
```

Modes are `"100644"` or `"100755"`, `null` exactly when the corresponding
blob digest is `null`. A mode-only change is a real transition (equal blob
digests, differing modes). `patch_note_sha256` is **opaque audit
metadata**: a producer-chosen digest of whatever diff rendering the
runtime archived. It is never verified, carries no algorithm contract, and
no refusal depends on it — endpoint blob digests and modes are the sole
ground truth.

### 2.3 Correction event — `axiom/lineage-correction/v1`

Signed by the actor (role `actor`) and countersigned by a protected
reviewer key distinct from the actor's (role `review`, per the §2.1 role
table). Fields: `schema`, `lane`, `epoch_sha256`, `actor`, `reason`,
`predecessor_record_sha256 | null`, `transitions` as §2.2. The
countersignature validates lineage; it never authorizes merge. Corrections
are first-class and loud: the sanctioned response to a wrong encoding
remains fix-the-encoder-and-re-encode, and a correction event is the
recorded exception, never the quiet path.

### 2.4 Verification report (Job 1, unsigned) — `axiom/notary-report/v1`

Content-addressed output of the verification workflow. Fields: `schema`,
`lane`, `epoch_sha256`; `subject_commit_git_oid`,
`subject_tree_manifest_sha256`; `base_commit_git_oid`,
`base_tree_manifest_sha256`; `chain_predecessor_sha256`,
`chain_predecessor_kind`; `profile_sha256`, `path_policy_sha256`;
`corpus_release` `{name, content_sha256}`; `waiver_set_sha256`;
`eligible_records` and `unused_eligible_records` (sorted digest arrays);
`coverage_assignment` (per changed protected path, the ordered record
digests consumed — the unique assignment of §3.3); `gates`: array sorted
by `gate_id`, **one entry per gate id** (duplicates are a parse refusal),
each `{gate_id, outcome, tier}`; `diff_coverage`: `"pass"` or a typed
refusal object (never `waived`, never `not-run`); `unprotected_changes`
(sorted paths); `dependency_pins_sha256`; `verifier_commit_git_oid`.

The report is a *proposal*. Refusal reports are diagnostic artifacts, and
even a pass report authorizes nothing: every claim-bearing field is
recomputed by the trusted side (§5) before signing, so a candidate-forged
report — pass verdict, trimmed unprotected list, invented assignment —
fails reconciliation rather than getting signed.

### 2.5 Notary receipt (signed) — `axiom/notary-receipt/v1`

The only artifact that, once finalized (§7), authorizes admission. Fields:
`schema`, `lane`, `epoch_sha256`; `report_sha256`; the §2.4 chain and
tree fields, each independently recomputed by the trusted side; `gates`
as declared (deduplicated, profile-complete, outcomes acceptable);
`job1`: `{workflow_path, workflow_git_oid, ref, run_id, run_attempt,
conclusion, artifact_sha256}` as read from the Actions control plane;
`authorization`: `{environment, approval_context}` (§11 wording
decision). Genesis and transitions are distinct schemas; a receipt cannot
claim their role.

### 2.6 Storage

Lineage records live in the lane repository under `.axiom/lineage/`,
append-only, one body file per record named by its content digest,
detached signatures beside it. The lineage directory is outside the
protected-content coverage domain but inside the preflight's structural
checks: schema-valid, authenticated, correctly content-addressed,
append-only; deleting or mutating an existing record is a refusal.
Reports, receipts, genesis, and transition records publish per §7 and are
never part of the subject tree.

## 3. The diff-coverage predicate

### 3.1 Protected paths

The policy is a committed file at the base
(`.axiom/notary/path-policy.json`):
`{schema: "axiom/notary-path-policy/v1", rules: [{action:
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
digest list is bound into report and receipt. Replay is dead: a record
merged by an earlier candidate is present at the base and ineligible ever
after; cross-lane and cross-epoch records refuse on their bindings; stale
records refuse in replay (§3.3).

### 3.3 The predicate, deterministic and total

Compute the tree-entry diff between the base and subject manifests
(equivalently `git diff-tree --no-renames`): per-path additions,
deletions, and modifications, including mode-only modifications. Renames
do not exist at this level. Over the protected subset:

1. **Unique valid assignment.** A valid assignment maps every changed
   protected path to exactly one non-forking, non-cyclic chain of
   transitions drawn from eligible records — records consumed atomically
   (all transitions used, or none), no transition against an unchanged
   path, endpoints per rule 2. If no valid assignment exists, refuse
   (uncovered or inconsistent). **If more than one valid assignment
   exists, refuse as ambiguous** — a candidate avoids this by not
   shipping redundant covering records. Records unusable in any valid
   assignment (dead-end retries) are legal, listed as unused, and create
   no ambiguity.
2. **Blob-and-mode ground truth.** Each chain's first
   `(before_blob_sha256, before_mode)` equals the path's state at the
   base (`(null, null)` for additions); each link's after-state equals
   the next link's before-state; the final after-state equals the path's
   state in the subject (`(null, null)` for deletions). Mode-only changes
   need a covering transition like any other change.
3. **Admissible entries.** Protected paths are regular blobs (100644 or
   100755) wherever they exist — symlinks are inadmissible in the
   protected domain, and gitlinks are refused tree-wide by §2.1.
   File/directory replacements decompose into entry deletions and
   additions and are covered as such.
4. **Binary verdict.** `diff_coverage` is `"pass"` or a typed refusal.
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

**Human authorization** gates the signing step through the dedicated
`notary-signing` environment (required reviewers, protected refs, no
self-approval) — an environment that holds no generation credentials and
no write tokens, unlike today's `production-signing`, which provisions
generation workflows with the apply key, a model credential, and a
write-capable token and is structurally disqualified. Environment
approval releases the *request* to the external signer; it never releases
key material to a runner.

**Publisher — separate job.** Holds only a token whose write capability
is confined to the notary ref by repository ruleset (App tokens scope to
repositories, not refs); publishes per §7; holds no signing capability.

Trust roots for every step are committed at or digest-pinned to the
protected base. Runtime organization variables are not trust anchors
anywhere in the notary path.

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
the protected paths** at genesis (the typed signer verifies the
partition property against the manifest before signing):

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

Rules: the recomputed base→subject diff must equal `delta` exactly — a
protected-content or unrelated change smuggled into a transition refuses;
signatures and authorization are evaluated **under the predecessor
state's trust roots** (the old keys and policy authorize the handover, so
a successor-controlled key cannot self-authorize its own installation);
transitions are produced only through the typed signer under the
`notary-signing` environment's human approval; and transitions are void
until finalized (§7), exactly like receipts. For merge gating, a pending
transition plays the pending receipt's role for its own subject (§7): it
is the merge-authorizing artifact for exactly its enumerated delta.
Ordinary admission resumes from the finalized transition's recorded
state.

## 7. Two-phase publication and the canonical chain

The canonical chain lives at a dedicated ref
(`refs/notary/chain`) in the lane repository: content-addressed artifact
files plus a single `HEAD.json` naming the current finalized tip
(`{schema: "axiom/notary-head/v1", tip_sha256, tip_kind}`). All updates
are compare-and-swap; the publisher's ruleset confines writes to this
ref.

Admission is two-phase, which resolves the ordering circularity between
"receipt must exist to authorize the merge" and "the merged state must
equal what was verified":

1. **Pending.** Job 1 verifies the candidate head (subject tree `T1`);
   the signer signs the receipt (or transition). The publisher writes it
   to the notary ref as *pending* and sets the required status check on
   the candidate. A pending artifact authorizes exactly one thing: the
   merge of a head whose tree manifest is `T1`. It is not the chain tip.
2. **Finalize.** After the merge, the finalizer (publisher job,
   triggered on the protected ref) recomputes the merged tip's tree
   manifest. If it equals `T1`, it advances `HEAD.json` to the artifact
   by CAS — the artifact is now finalized and chain-eligible. If it
   differs (conflicted queue merge, stale squash), finalization refuses,
   the artifact is permanently void (a voided digest is recorded and can
   never finalize), and verification reruns from the true finalized tip.
   A pending artifact whose candidate never merges simply never
   finalizes; the chain never advanced, so nothing strands.

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
3. The dedicated `notary-signing` environment created (required
   reviewers, protected refs, no self-approval), holding no generation
   credentials and no write tokens; generation workflows migrated off
   any environment involved in notary authorization (#1194 re-scoped).
4. **External signer with identity binding**: the notary key held only
   by the external typed signer; OIDC-claim validation of the requesting
   workflow/job/run; no raw notary key material on any runner.
5. **Compute isolation**: fresh ephemeral runners for the verification
   and trusted jobs.
6. Trusted recomputation implemented (§5): every claim-bearing invariant
   re-derived from content-addressed inputs before signing; typed
   operations for genesis, transition, and receipt scopes; no raw-byte
   signing for those scopes.
7. The publisher split implemented, with a repository ruleset confining
   the publisher token's writes to the notary ref, and the two-phase
   pending/finalize protocol with CAS and permanent voiding.
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

## 10. Negative-test floor

Grouped by refusal site; each case is a distinct test before the pilot
gates anything.

**Schemas, encodings, signatures:** unknown/missing field; wrong type;
number where string required (`run_id`); digest with wrong case, length,
or retained `sha256:` prefix; wrong content-address filename; malformed
detached-signature file; wrong scope per the role table (review
signature under the correction scope and conversely); signature by the
wrong key for a scope; invalid producer/actor/review/genesis/transition/
receipt signature; countersignature by the actor; raw-byte signing
attempt against genesis, transition, or notary scopes; full pairwise
cross-scope matrix including legacy operations.

**Manifests and totality:** gitlink anywhere in the tree; non-UTF-8 path
anywhere in the tree (changed or unchanged); inadmissible mode; symlink
inside the protected domain; manifest sort-order violation; two trees
differing only by a gitlink refused rather than treated as
manifest-equal.

**Lineage store:** mutated existing record; deleted existing record;
wrong lane; wrong epoch; mode present on a null side; mode outside the
admissible set.

**Eligibility and coverage:** record present at the base (replay);
uncovered protected change; uncovered mode-only change; overlapping
chains; transition against an unchanged path; forked, cyclic, or
discontinuous chain; wrong starting or terminal blob/mode; malformed
null-sides; partial record consumption; **ambiguous assignment (two
redundant covering records; split-vs-direct chains)**; dead-end retry
record accepted as unused without ambiguity (positive control);
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
invalid signature, path/blob disagreement); nonexistent
`record_sha256`; duplicate qualifying v5 records for one path; genesis
carrying coverage; empty-delta first receipt leaving baseline provenance
unchanged (positive control); post-epoch change vouched only by v5; v5
record whose blob differs from its frozen pair; transition whose
recomputed diff differs from its enumerated delta (smuggled content);
transition evaluated under successor roots (self-authorizing rotation);
unfinalized transition used as predecessor; ordinary receipt attempting
a trust-surface change.

**Signer, publication, finalization:** OIDC identity mismatch (wrong
workflow, job, run, repository, or ref); Job-1 run non-success; artifact
digest mismatch; replayed artifact under a new run; signing a refusal
report; stale base at recomputation; pending artifact treated as chain
tip; finalization with merged-tip manifest differing from the verified
subject (voids, permanently); voided digest re-finalization attempt; CAS
loss on `HEAD.json`; publisher push outside the notary ref rejected by
ruleset; `HEAD.json` naming an unpublished or unsigned artifact.

## 11. Decisions for sign-off

- ProgramSpec scope: atomic RuleSpec only in the pilot (recommended), or
  extend the path policy to composition outputs.
- Licensed or unavailable oracles: fail closed, or visibly reduced-tier
  receipt.
- Approval wording: `authorization.approval_context` binds durable
  digest-bound reviewer evidence, or records "the protected signing
  policy authorized this receipt" (honest for plain environment
  approval; the stronger form needs an explicit approval artifact).
- Custody model for the producer, actor, reviewer, and administrative
  keys (the notary key is fixed by §5/§8); reviewer custody is the open
  question deferred from the rulespec-nz custody ruling.

## 12. Out of scope for milestone one

Witnessed lineage chains (dual RFC 3161 — sequenced behind the notary as
chartered); historical backfill; rename modeling (tree-entry
decomposition makes it unnecessary); gitlink/submodule support (refused
tree-wide in the pilot); fleet-wide shared-workflow conversion; v5
retirement; the other eight lanes; ProgramSpec admission unless §11
decides otherwise.
