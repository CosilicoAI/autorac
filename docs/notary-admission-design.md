# Notary admission: design v3

Status: draft for sign-off. Implements the #1192 charter with the #1506
diff-coverage delta, under the build decision recorded on both issues
(dual-verdict, 2026-08-17). Version 3 folds design-review rounds 1 and 2
(seventeen blocking findings total; the review record lives on #1507).
Nothing admission-capable merges until the preconditions in §9 are
satisfied and this document is approved by the charter's gate: an
independent cross-family review of this concrete design plus Max's named
sign-off, with every §11 decision closed.

## 1. Claim and threat model

The notary signs one narrow claim:

> Verification report `R` describes subject tree `H1` (manifest digest
> `T1`), reached from predecessor state `T0` recorded by chain predecessor
> `C` (a receipt, a transition record, or genesis): under the verifier
> profile and protected-path policy committed at the base, every
> protected-content tree-entry change from `T0` to `T1` is covered by the
> eligible lineage set; every gate the profile requires produced an
> outcome the profile accepts; unprotected tree-entry changes are
> enumerated in `R`; and the protected signing policy authorized this
> acceptance.

Admission is a statement about the *delta*: the changes since the chain
predecessor are covered. It never upgrades the provenance of older bytes.
Pre-epoch content remains exactly what the genesis inventories say it is —
v5-attested or unattested baseline — no matter how many receipts later sit
above it; an empty-delta receipt admits nothing and changes no provenance
status. The claim covers exactly what the predicate checks
(protected-content tree-entry changes), and unprotected changes are
enumerated in the report rather than silently excluded.

The notary does not claim content was model-generated, and never will —
that claim bred provenance laundering (the Path R pattern) and is retired
with the v5 era. Lineage records say where bytes came from; the notary
says the deterministic gates passed, coverage held, and admission chains
from the last admitted state.

The scenario this design closes: an agent with repository write access
hand-edits a published rule and obtains a legitimate signature through the
sanctioned path, with no record of the edit's origin. Two walls stop the
*undeclared* hand edit — declared corrections are intentionally
admissible, loudly. The capability wall: no generation surface holds the
merge-authorizing notary capability; the notary key is reachable only by
the typed signing job, on isolated compute, in a dedicated notary-only
environment. (Generation surfaces do hold the non-authorizing producer
key — that signs lineage, not admission.) The coverage wall: a changed
protected path with no covering eligible lineage refuses at verification,
and the signer independently re-enforces that verdict before signing.

Residual risks accepted, per the charter: a verifier soundness bug
deterministically accepts wrong law; candidate-controlled tests can be
weakened; reviewer compromise authorizes validator-passing bad content.
Mitigations are the strict profile, the trusted preflight, golden
regeneration QA, and oracles — not additional signatures.

## 2. Artifacts, schemas, and signature envelope

### 2.1 Conventions shared by every artifact

**Serialization.** Bodies are canonical JSON (receipt-canonical
serialization: UTF-16 code-unit key order, ECMAScript number formatting),
UTF-8 bytes, content-addressed by SHA-256 of the body bytes. Every body
carries `schema` (its full identifier, e.g. `axiom/notary-receipt/v1` —
the same full string serves as the signing scope; there are no shortened
scope names) and `lane` (canonical `owner/repo`). Schemas are
closed-world: an unknown field, a missing field, or a wrong type is a
parse refusal.

**Identifiers and digests.** Two disjoint conventions, never mixed:

- `*_git_oid`: a Git object id in the repository's object format (SHA-1
  in the pilot repositories). Used to *locate* objects and to interact
  with Git, the Actions control plane, and CAS. Never the
  collision-resistance-bearing binding.
- `*_sha256`: SHA-256 over defined content bytes. Blob digests are over
  raw blob content. A **tree manifest digest** is SHA-256 over the
  canonical JSON array of `[path, mode, content_sha256]` triples, sorted
  by path, covering every blob reachable from the tree — the
  collision-resistant description of a tree state, independent of Git's
  object format.

**Signature envelope.** Signing uses the existing broker frame verbatim:

```
"axiom-encode/external-signer-sign/v2" || 0x00 || scope || 0x00 || payload
```

where `scope` is the artifact's full schema identifier and `payload` is
the 64-byte lowercase-hex ASCII encoding of the body's SHA-256.
Signatures are detached files beside the body: `<digest>.json` and
`<digest>.json.<role>.sig`, where `<role>` is `producer`, `actor`,
`review`, `genesis`, `transition`, or `notary`. Each `.sig` file is
canonical JSON: `{schema: "axiom/detached-signature/v1", body_sha256,
scope, signer_spki_sha256, signature_base64}`. A signature under one
scope must fail verification under every other scope, legacy
`apply_ed25519`/`eval_ed25519` operations included (§10 tests the full
pairwise matrix).

**Typed signing.** For the notary and transition scopes, the trusted
signing component accepts only the typed fields of §2.5/§6.4,
reconstructs the canonical body itself, computes its digest, and signs.
No caller can submit arbitrary bytes to those scopes. Lineage scopes
(producer, actor, review) are signed at their origins (supervised
runtime; actor and reviewer tooling) under the same envelope.

### 2.2 Generation event — `axiom/lineage-generation/v1`

Signed by the supervised-runtime producer key (non-authorizing). Fields:
`schema`, `lane`, `epoch_sha256` (§6), `runtime_identity`, `model`,
`cli_version`, `cli_sha256`, `prompt_sha256s`, `emitted_at`
(informational, unverified), and `transitions`: an array sorted by path,
each:

```
{path, before_blob_sha256 | null, before_mode | null,
 after_blob_sha256 | null, after_mode | null,
 patch_sha256, patch_algorithm}
```

Modes are restricted to `"100644"` and `"100755"`; the mode is `null`
exactly when the corresponding blob digest is `null` (absent side). A
mode-only change is a real transition (same blob digests, differing
modes). `patch_algorithm` names a versioned canonical recipe
(`git-patch-v1`: `git diff --no-renames --full-index --binary -U3` with
`core.autocrlf=false`, no textconv, no external diff, attributes
ignored); the patch digest is audit metadata — blob digests and modes are
ground truth.

### 2.3 Correction event — `axiom/lineage-correction/v1`

Signed by the actor's key (role `actor`) and countersigned by a protected
reviewer key distinct from the actor's (role `review`, scope
`axiom/lineage-correction-review/v1` over the same body digest). Fields:
`schema`, `lane`, `epoch_sha256`, `actor`, `reason`,
`predecessor_record_sha256 | null`, and `transitions` as in §2.2. The
countersignature validates lineage — it never authorizes merge.
Corrections are first-class and loud: the sanctioned response to a wrong
encoding remains fix-the-encoder-and-re-encode, and a correction event is
the recorded exception, never the quiet path.

### 2.4 Verification report (Job 1, unsigned) — `axiom/notary-report/v1`

Content-addressed, produced by the secretless verify job. Fields:

- `schema`, `lane`, `epoch_sha256`
- `subject_commit_git_oid`, `subject_tree_manifest_sha256`
- `base_commit_git_oid`, `base_tree_manifest_sha256`
- `chain_predecessor_sha256` (receipt, transition record, or genesis body
  digest) and `chain_predecessor_kind`
- `profile_sha256`, `path_policy_sha256` (digests of the definition files
  committed at the base)
- `corpus_release`: `{name, content_sha256}` (the toolchain tuple)
- `waiver_set_sha256`
- `eligible_records`: sorted array of record digests (§3.2), and
  `unused_eligible_records`: sorted subset not consumed by the assignment
- `gates`: sorted array of `{gate_id, outcome, tier}`, `tier` in
  `public | restricted | ci-attested`
- `diff_coverage`: `"pass"` or a typed refusal object (never `waived`,
  never `not-run`)
- `unprotected_changes`: sorted array of changed paths outside the
  protected domain
- `dependency_pins_sha256`, `verifier_commit_git_oid`

A report may record a refusal — refusal reports are diagnostic artifacts.
Nothing downstream may sign one: the signer enforces acceptance itself
(§5), never inferring it from a job's success.

### 2.5 Notary receipt (Job 2, signed) — `axiom/notary-receipt/v1`

The only merge-authorizing artifact, signed under the notary scope by
`notary_ed25519`. Fields:

- `schema`, `lane`, `epoch_sha256`
- `report_sha256`
- `subject_commit_git_oid`, `subject_tree_manifest_sha256`,
  `base_tree_manifest_sha256`, `chain_predecessor_sha256`,
  `chain_predecessor_kind` — each independently re-derived by the signing
  component and required to match the report (a mismatch is a refusal,
  §5)
- `job1`: `{workflow_path, workflow_sha, ref, run_id, run_attempt,
  conclusion, artifact_sha256}` as read from the Actions control plane by
  the signing component itself
- `authorization`: `{environment, approval_context}` (§11 wording
  decision)

There is no `genesis` flag: genesis and transitions are distinct schemas
(§6), and a receipt claiming their role fails its closed schema.

### 2.6 Storage

Lineage records live in the lane repository under `.axiom/lineage/`,
append-only, one body file per record named by its content digest, with
detached signatures beside it. The lineage directory is outside the
protected-content coverage domain (its additions need no covering record)
but inside the preflight's structural checks: schema-valid,
authenticated, correctly content-addressed, append-only. Deleting or
mutating an existing lineage record is a refusal. Reports, receipts,
genesis, and transition records publish per §7 and are never part of the
subject tree.

## 3. The diff-coverage predicate

### 3.1 Protected paths, defined normatively

The protected-path policy is a committed file at the base
(`.axiom/notary/path-policy.json`):

```
{schema: "axiom/notary-path-policy/v1",
 rules: [{action: "include" | "exclude", prefix: "<path prefix>"}]}
```

Semantics, exactly: a prefix matches path `p` iff `p == prefix` or `p`
begins with `prefix + "/"` — component-wise, so `rules` never matches
`rules-evil/x`. Rules are evaluated in order; the **last** matching rule
decides; a path matching no rule is **unprotected**. Prefixes and paths
are UTF-8 strings. Classification uses the policy committed at the base —
a candidate cannot reclassify paths in its own favor, and a policy change
is a trust-surface change ordinary candidates cannot carry (§4; it moves
by transition record, §6.4).

Totality rule for undecodable names: any changed path anywhere in the
base→subject diff that is not valid UTF-8 is a refusal in the pilot —
protected or not — because neither the policy nor the report's JSON
arrays can represent it faithfully.

### 3.2 Eligible lineage set

Eligible records are exactly the records whose body files are present
under `.axiom/lineage/` in the subject tree and absent at the base —
introduced by this candidate — and which carry this `lane` and this
`epoch_sha256`. The sorted digest list is bound into the report. This
defeats replay: a record merged by an earlier candidate is present at the
base and ineligible ever after; a cross-lane or cross-epoch record
refuses on its bindings; a stale record whose before-digests no longer
match the base refuses in replay (§3.3).

Records are consumed atomically: all of a record's transitions used by
the covering assignment, or none. A partially matching record is a
refusal. Unused eligible records are legal (a retried generation) and are
listed in the report — never invisible.

### 3.3 The predicate

Compute the tree-entry diff from base tree to subject tree with rename
detection disabled (`git diff-tree --no-renames` semantics): a set of
per-path entry changes — additions, deletions, modifications, including
mode-only modifications. Renames do not exist at this level; a moved file
is a deletion and an addition, each needing coverage. Then, over the
protected subset:

1. **Closed-world assignment.** Every changed protected path maps to
   exactly one non-forking, non-cyclic chain of transitions drawn from
   eligible records — no missing, no overlapping, no extra transitions
   against unchanged paths.
2. **Blob-and-mode ground truth.** Each chain's first
   `(before_blob_sha256, before_mode)` equals the path's state at the
   base (`(null, null)` for an addition); each link's after-state equals
   the next link's before-state; the last after-state equals the path's
   state in the subject tree (`(null, null)` for a deletion). A mode-only
   change requires a covering transition like any other change.
3. **Admissible entries only.** Protected paths must be regular blobs
   (mode 100644 or 100755) wherever they exist. A gitlink, symlink, or
   other mode at a protected path — either side — is a refusal.
   File-to-directory and directory-to-file replacements decompose into
   entry deletions and additions and are covered as such.
4. **Binary verdict.** `diff_coverage` is `"pass"` or a typed refusal.
   From the enforcement epoch there is no `waived` and no `not-run`; an
   emergency change without a countersigned correction event is a
   refusal. This is deliberately stricter than consumer-side declaration
   completeness, which accepts declared `waived`/`not-run` outcomes as
   well-formed.

Advisory shadow runs (compute and publish the verdict without gating) are
permitted only before a lane's enforcement epoch.

## 4. Verifier profile P and trusted preflight

The profile is a committed definition at the base, digest-bound into the
report. It defines the gate set, each gate's acceptable outcomes, and the
oracle policy. It is not the current apply path, and the differences are
the point:

- **Non-mutating.** No deterministic repairs. The subject is verified as
  presented or refused; a repairable-but-unrepaired subject refuses.
- **Oracles on.** The apply overlay's oracle disablement does not exist
  here. Licensed or unavailable oracles follow the §11 decision — fail
  closed, or a visibly reduced-tier receipt; never silent.
- **Reviewers means deterministic checks plus protected-environment human
  approval.** The validator pipeline's LLM reviewers are QA outside the
  admission path, beside golden regeneration; under the charter's
  no-model-calls rule they cannot gate the notary.
- **No caller switches.** Skip flags and caller-disableable guards have
  no notary equivalents.

The trusted preflight (runs in Job 1, from pinned code, before any
candidate code executes) resolves: the canonical repository; the chain
predecessor and its recorded state (§6); the path policy and profile at
the base; the complete tree-entry diff. It refuses ordinary candidates
that change workflows, action pins, verifier pins, trust roots, the path
policy, the profile, waiver policy, repository-structure rules, or the
lineage store's integrity (§2.6). Those surfaces move only by transition
record (§6.4).

## 5. Jobs, compute isolation, and capability separation

**Job 1 — verify (secretless, isolated).** Runs candidate code under the
strict profile on a **fresh ephemeral runner** (GitHub-hosted, or a
dedicated ephemeral pool with per-job teardown) with no signing
capability, no model credentials, and no repository-write credential.
Emits the §2.4 report as a content-addressed artifact.

**Job 2 — sign (typed, notary-only, isolated).** Runs on a fresh
ephemeral runner in a **dedicated `notary-signing` environment** — not
`production-signing`, which today provisions generation workflows with
the apply key, a model credential, and a write-capable token, and is
structurally disqualified from holding the notary key. The environment
enforces required human reviewers, protected refs, no self-approval, and
Job-2-only secret access; the signer additionally binds the requesting
workflow and job identity, so a leaked approval cannot be replayed from
elsewhere. Job 2 runs no candidate code and holds no model or
repository-write credential. Its trusted component:

1. Reads the Job-1 run from the Actions control plane: workflow path and
   sha, ref, run id and attempt, conclusion, artifact digest. Wrong
   workflow, wrong ref, non-success conclusion, or artifact mismatch
   refuses.
2. Independently re-fetches and re-hashes the subject and base trees
   (recomputing both tree manifests), the report body, and the profile
   and path-policy files at the base; independently re-derives the chain
   predecessor and its admissibility (§6). Any mismatch with the
   report's fields discards the run and forces complete reverification.
3. **Enforces the acceptance predicate itself**: `diff_coverage` is
   exactly `"pass"`; every gate the profile requires is present with an
   outcome the profile accepts; the report's eligible and unused record
   lists are well-formed against the trees. A refusal report — however
   successfully its workflow concluded — is never signable. Actions
   success is context, not acceptance.
4. Reconstructs the §2.5 receipt body from typed fields and signs its
   digest under the notary scope. It exposes no arbitrary-byte signing
   operation.

**Publisher — a third job.** Verification and signing forbid
repository-write capability, so neither publishes. A separate publisher
job holds only a token whose write capability is confined to the receipt
publication ref — enforced by a repository ruleset, since App tokens
scope to repositories, not refs — and pushes per §7. It holds no signing
capability.

Trust roots for all jobs — public keys, profile and policy digests,
workflow and action pins, waiver authorities — are committed at or
digest-pinned to the protected base. Runtime organization variables are
not trust anchors anywhere in the notary path; eliminating the existing
`vars.*` provisioning from the signing path is a §9 precondition.

## 6. Admission chain, epoch, genesis, and transitions

### 6.1 The chain

Every receipt names its chain predecessor: a prior receipt, a transition
record, or genesis. The receipt's `base_tree_manifest_sha256` must equal
the predecessor's subject (for receipts and transitions) or recorded
baseline (for genesis). Verifying from any other base refuses: an
uncovered edit cannot launder itself by becoming the base of the next
verification, because the state that skipped it is not the recorded
chain tip. Commit ids locate; tree manifests bind — so squash and
merge-queue rewrites are harmless when they preserve the verified tree,
and void the run when they do not (§7).

### 6.2 Genesis — `axiom/notary-genesis/v1`

A distinct artifact signed under its own scope by the administrative
authority (§6.4's signer). Its body binds: `schema`, `lane`,
`genesis_commit_git_oid`, `genesis_tree_manifest_sha256`, and two frozen
inventories over the protected domain at genesis:

- `v5_attested`: sorted `[path, content_sha256, record_sha256]` triples —
  paths whose blobs are vouched by an authenticated v5 record at genesis;
- `baseline_unattested`: sorted `[path, content_sha256]` pairs — paths
  with no authenticating record, recorded visibly, including any
  pre-epoch hand edits. Recorded is not blessed: these bytes carry
  "unattested baseline" provenance permanently.

The **epoch identifier is the genesis body's content digest**. The body
contains no epoch field — it *is* the epoch — so every other artifact's
`epoch_sha256` refers to it without a self-hash fixed point. A second
genesis for a lane refuses; genesis covers nothing and admits nothing.

### 6.3 Dual-era rule

Post-epoch, a v5 record vouches only for a path whose blob is
byte-identical to its `v5_attested` pair in the frozen inventory. Any
post-epoch change to any protected path requires notary-era lineage; new
v5 records have no post-epoch standing; a path in `baseline_unattested`
stays unattested until a covered change replaces it.

### 6.4 Transition records — `axiom/notary-transition/v1`

Privileged trust-surface updates (path policy, profile, workflow or
action pins, trust roots, waiver policy, key rotation) cannot receive
ordinary receipts — the preflight refuses them from candidates — and
without a dedicated mechanism the chain would deadlock at the first key
rotation. A transition record is that mechanism: signed under its own
scope by the administrative authority through the `notary-signing`
environment's human approval, its body binds `schema`, `lane`,
`epoch_sha256`, `chain_predecessor_sha256` and kind, the old and new
tree manifest digests, the enumerated trust-surface delta (paths and
before/after digests), and `reason`. A transition covers only
trust-surface paths — a protected-content change smuggled into a
transition refuses — and becomes a valid chain predecessor. Ordinary
admission resumes from its recorded state.

## 7. Publication and the canonical head

The publisher pushes the receipt (with its report) to the lane's
canonical attestation ref, compare-and-swap. A signed receipt is **void
until published**: the chain reads only the canonical ref, so a
CAS-losing or stale signed receipt never becomes chain-eligible, and
re-verification from the true tip is the only path forward. At
publication the publisher verifies the protected ref's tip tree manifest
equals the receipt's subject tree manifest; a merge that produced a
different tree (a conflicted queue merge, a stale squash) fails the
check, voids the receipt, and reruns. The receipt distinguishes
subject/base tree manifests, subject commit oid, and
`verifier_commit_git_oid`; the attestation ref supplies location without
entering the signed body.

## 8. Key ceremony

Before Job 2 exists: generate `notary_ed25519` with documented
custodians, fingerprint published in committed code (consumer
verification specs pin the SPKI), storage and rotation procedure
(rotation is a §6.4 transition), revocation and recovery procedure. The
producer, actor, reviewer, and administrative keys get the same treatment
under their own scopes. §10's pairwise cross-scope matrix is part of
ceremony acceptance.

## 9. Preconditions (nothing admission-capable merges before these)

1. This document approved — threat model, schemas and envelope,
   predicate, profile, chain/epoch/genesis/transition rules, and the §10
   suite — by the charter's gate: an independent cross-family review of
   the concrete design plus **Max's named sign-off**, with every §11
   decision closed.
2. The reviewers-semantics resolution in §4 accepted.
3. The dedicated `notary-signing` environment created and enforcing:
   required reviewers, protected refs, no self-approval, Job-2-only
   secret access — with generation workflows migrated off any
   environment that will hold the notary key (#1194 re-scoped).
4. **Compute isolation**: Jobs 1 and 2 on fresh ephemeral runners (or a
   dedicated ephemeral pool), and signer-side workflow/job identity
   binding, so no candidate-code persistence can reach the notary
   operation.
5. The authenticated Job-1 handoff, typed Job-2 reconstruction, and
   Job-2 acceptance enforcement of §5 implemented in the trusted signing
   component (no raw-byte notary scope).
6. The publisher split of §5 implemented, with a repository ruleset
   confining the publisher token's writes to the receipt ref.
7. Signed-leg billing and abuse policy operational (#1193).
8. Notary key ceremony completed per §8, transition mechanism included.
9. Org-variable trust anchors eliminated from the signing path in favor
   of committed or digest-pinned roots.
10. rulespec-nz consumer-side pins landed: the notary SPKI and epoch in
    the lane's committed verification spec; the notary check required and
    non-bypassable in branch protection; genesis activated atomically
    with enforcement.
11. The generated-file guard hardcoded in the protected shared workflow
    (its caller-controlled boolean retired) — merge-time defense in depth
    during dual-era operation.
12. Retirement of lane signed-apply legs (#1195) is a cutover exit
    criterion, not a pilot precondition; dual-era operation is expected.

## 10. Negative-test floor

Grouped by refusal site; each case is a distinct test before the pilot
gates anything.

**Schemas and signatures:** unknown field; missing field; wrong type;
wrong content-address filename; malformed detached-signature file;
signature by the wrong key for a scope; invalid producer signature;
invalid actor signature; missing or invalid reviewer countersignature;
countersignature by the actor; countersignature by an untrusted key;
invalid genesis signature; invalid transition signature; invalid receipt
signature; raw-byte signing attempt against the notary or transition
scope; the full pairwise cross-scope matrix (all six scopes plus legacy
`apply_ed25519`/`eval_ed25519`) — every cross-scope verification fails.

**Lineage store:** mutated existing record; deleted existing record;
record with wrong lane; record with wrong epoch; mode field present on a
null side; mode value outside the admissible set.

**Eligibility and coverage:** record present at the base (replay);
uncovered protected change; uncovered mode-only change; overlapping
chains; extra transition against an unchanged path; forked chain; cyclic
chain; discontinuous chain; wrong starting blob or mode; wrong terminal
blob or mode; malformed null-sides; partial record consumption; wrong
patch digest or unknown `patch_algorithm`; gitlink at a protected path;
symlink at a protected path; disallowed mode at a protected path;
non-UTF-8 changed path anywhere in the diff; `waived`/`not-run` coverage
outcome post-epoch; path-policy precedence cases (`rules/private/x`
under include-then-exclude; `rules-evil/x` against prefix `rules`;
unmatched path defaulting to unprotected).

**Preflight and profile:** ordinary candidate touching workflows, pins,
trust roots, path policy, profile, waiver policy, repository-structure
rules, or lineage-store integrity; repairable-but-unrepaired subject;
failed gate; gate outcome outside the profile's acceptable set; missing
required gate; oracle-disabled attempt; skip-flag attempt; profile or
path policy not committed at the base.

**Chain, epoch, genesis, transitions:** base not the chain tip; wrong
chain predecessor digest; second genesis; genesis carrying coverage;
receipt claiming a predecessor role its schema forbids; empty-delta
first receipt leaving baseline provenance unchanged (positive control);
post-epoch change vouched only by v5; v5 record whose blob differs from
its frozen pair; malformed or path-mismatched grandfather inventory
entry; transition carrying a protected-content change; transition with
an unenumerated trust-surface delta; ordinary receipt attempting a
trust-surface change; chain advance over an unpublished (void) receipt.

**Job 2 and publication:** report-body digest mismatch; subject or base
tree-manifest mismatch on re-hash; profile or path-policy re-hash
mismatch; copied-field mismatch between report and receipt; Job-1 run
from the wrong workflow, sha, or ref; non-success Job-1 conclusion;
artifact digest mismatch; replayed Job-1 artifact under a new run;
**signing a refusal report**; signing with a missing required gate;
invalid authorization reference; wrong requesting workflow/job identity
at the signer; CAS loss voids the receipt; stale signed receipt never
chain-eligible; published-tip tree differing from receipt tree
(squash/queue rewrite) voids and reruns; receipt-child commit with any
extra delta; publisher push outside the receipt ref rejected by ruleset.

## 11. Decisions for sign-off

- ProgramSpec scope: atomic RuleSpec only in the pilot (recommended), or
  extend the path policy to composition outputs.
- Licensed or unavailable oracles: fail closed, or visibly reduced-tier
  receipt.
- Approval wording: `authorization.approval_context` binds durable
  digest-bound reviewer evidence, or records "the protected signing
  policy authorized this receipt" (honest for plain environment
  approval, which approves a deployment, not a displayed digest; the
  stronger form needs an explicit approval artifact).
- Custody model for the producer, actor, reviewer, and administrative
  keys (the notary key is fixed by §5/§8); reviewer custody is the open
  question deferred from the rulespec-nz custody ruling.

## 12. Out of scope for milestone one

Witnessed lineage chains (dual RFC 3161 — sequenced behind the notary as
chartered); historical backfill; rename modeling (tree-entry
decomposition makes it unnecessary); fleet-wide shared-workflow
conversion; v5 retirement; the other eight lanes; ProgramSpec admission
unless §11 decides otherwise.
