# Notary admission: design v2

Status: draft for sign-off. Implements the #1192 charter with the #1506
diff-coverage delta, under the build decision recorded on both issues
(dual-verdict, 2026-08-17). Version 2 folds the first cross-family design
review (nine blocking findings; the review record lives on #1507). Nothing
admission-capable merges until the preconditions in §9 are satisfied and
this document is approved by the charter's gate: an independent
cross-family review of this concrete design plus Max's named sign-off.

## 1. Claim and threat model

The notary signs one narrow claim:

> Verification report `R` describes subject commit `S` (tree `H1`): under
> verifier profile `P` and protected-path policy committed at base `B`,
> every protected-content tree-entry change from `B` to `H1` is covered by
> the eligible lineage set `L`, every gate in `P` produced the recorded
> outcome, unprotected tree-entry changes are enumerated in `R`, `B` is the
> previously admitted tip, and the protected signing policy authorized this
> acceptance.

The claim covers exactly what the predicate checks: protected-content
tree-entry changes. Unprotected changes (documentation, tooling) are not
silently outside the statement — the report enumerates them, so a reader
sees precisely what was and was not under coverage.

The notary does not claim content was model-generated, and never will —
that claim bred provenance laundering (the Path R pattern) and is retired
with the v5 era. Lineage records say where bytes came from; the notary says
the deterministic gates passed, coverage held, and admission chains from
the last admitted state.

The scenario this design closes: an agent with repository write access
hand-edits a published rule and obtains a legitimate signature through the
sanctioned path, with no record of the edit's origin. Two walls stop the
*undeclared* hand edit — declared corrections are intentionally admissible,
loudly. The capability wall: no generation surface holds the
merge-authorizing notary capability; the notary key lives in a dedicated
notary-only environment reachable only by the typed signing job, which runs
no candidate code. (Generation surfaces do hold the non-authorizing
producer key — that signs lineage, not admission.) The coverage wall: a
changed protected path with no covering eligible lineage refuses at
verification, regardless of who approves.

Residual risks accepted, per the charter: a verifier soundness bug
deterministically accepts wrong law; candidate-controlled tests can be
weakened; reviewer compromise authorizes validator-passing bad content.
Mitigations are the strict profile, the trusted preflight, golden
regeneration QA, and oracles — not additional signatures.

## 2. Artifacts, schemas, and signature envelope

### 2.1 Serialization and framing, shared by every artifact

- Bodies are canonical JSON (receipt-canonical serialization: UTF-16
  code-unit key order, ECMAScript number formatting), UTF-8 bytes,
  content-addressed by SHA-256. Digest strings are lowercase hex, always
  written into fields whose names end in `_sha256`; no multihash prefixes.
- Every body carries `schema` (a closed identifier, e.g.
  `axiom/notary-receipt/v1`) and `lane` (the canonical `owner/repo`).
  Unknown fields are a parse refusal; missing fields are a parse refusal;
  schemas are closed-world.
- Signatures are detached: `<digest>.json` beside `<digest>.json.sig`. No
  in-body signature fields exist, so no exclusion rules exist.
- Signing input is the broker's existing framing, with the domain as the
  scope: `sign(key, domain || 0x00 || body_sha256_bytes)`. The domain
  strings below are the complete scope set; a signature made under one
  domain must fail verification under every other (§10 tests the full
  pairwise matrix, legacy `apply_ed25519`/`eval_ed25519` included).
- The signer for the notary domain is a typed operation: the trusted
  signing component receives the typed fields of §2.4, reconstructs the
  canonical body itself, computes its digest, and signs. No caller can
  submit arbitrary bytes to the notary domain.

### 2.2 Generation event — `axiom/lineage-generation/v1`

Signed by the supervised-runtime producer key (non-authorizing). Fields:
`schema`, `lane`, `epoch_sha256` (§6), `runtime_identity`, `model`,
`cli_version`, `cli_sha256`, `prompt_sha256s`, `emitted_at` (informational,
unverified), and `transitions`: an array sorted by path, each
`{path, before_blob_sha256 | null, after_blob_sha256 | null,
patch_sha256, patch_algorithm}`. `null` before-side is an addition; `null`
after-side is a deletion. `patch_algorithm` names a versioned canonical
patch recipe (`git-patch-v1`: `git diff --no-renames --full-index --binary
-U3` with `core.autocrlf=false`, no textconv, no external diff, path
attributes ignored); the patch digest is audit metadata — blob digests are
ground truth.

### 2.3 Correction event — `axiom/lineage-correction/v1`

Signed by the actor's key and countersigned (a second detached signature
under `axiom/lineage-correction-review/v1`) by a protected reviewer key
distinct from the actor's. Fields: `schema`, `lane`, `epoch_sha256`,
`actor`, `reason`, `predecessor_record_sha256 | null`, and `transitions` as
in §2.2. The countersignature validates lineage — it never authorizes
merge. Corrections are first-class and loud: the sanctioned response to a
wrong encoding remains fix-the-encoder-and-re-encode, and a correction
event is the recorded exception, never the quiet path.

### 2.4 Verification report (Job 1, unsigned) — `axiom/notary-report/v1`

Content-addressed, produced by the secretless verify job. Fields:

- `schema`, `lane`, `epoch_sha256`
- `subject_commit`, `subject_tree_sha256`
- `base_commit`, `base_tree_sha256`
- `previous_receipt_sha256` (the receipt this admission chains from;
  genesis reference for the first)
- `profile_sha256` (digest of the profile definition file committed at
  `B`), `path_policy_sha256` (likewise; §3.1)
- `corpus_release`: `{name, content_sha256}` (the toolchain tuple)
- `waiver_set_sha256`
- `eligible_records`: sorted array of record digests (§3.2), and
  `eligible_records_sha256` over that array's canonical bytes
- `gates`: sorted array of `{gate_id, outcome, tier}` with `tier` in
  `public | restricted | ci-attested`
- `diff_coverage`: `"pass"` or a refusal object (never `waived`, never
  `not-run`)
- `unprotected_changes`: sorted array of paths changed outside the
  protected domain
- `dependency_pins_sha256`: digest of the exact resolved dependency set of
  the verifier run
- `verifier_commit`: the axiom-encode commit whose code ran Job 1

The report cannot contain the authorization, which does not exist until
Job 2; the receipt binds the two.

### 2.5 Notary receipt (Job 2, signed) — `axiom/notary-receipt/v1`

The only artifact signed under the notary domain, by `notary_ed25519`.
Fields:

- `schema`, `lane`, `epoch_sha256`
- `report_sha256`
- `subject_commit`, `subject_tree_sha256`, `base_commit`,
  `previous_receipt_sha256` (copied from the report after independent
  re-derivation; a mismatch is a refusal, §5)
- `job1`: `{workflow_path, workflow_sha, ref, run_id, run_attempt,
  conclusion, artifact_sha256}` as read from the Actions control plane by
  the signing component itself
- `authorization`: `{environment, approval_context}` — the protected
  signing policy reference (see §11 for the wording decision)
- `genesis`: boolean, `false` for every ordinary receipt (§6)

### 2.6 Storage

Lineage records live in the lane repository under `.axiom/lineage/`,
append-only, one body file per record named by its content digest, with its
detached signatures beside it. The lineage directory is outside the
protected-content coverage domain (its additions need no covering record)
but inside the preflight's structural checks: schema-valid, authenticated,
correctly content-addressed, append-only. Deleting or mutating an existing
lineage record is a refusal. Receipts and reports publish per §7 and are
never part of the subject tree.

## 3. The diff-coverage predicate

### 3.1 Protected paths, defined normatively

The protected-path policy is a committed file at `B`
(`.axiom/notary/path-policy.json`): an ordered list of include/exclude
prefix rules over UTF-8 paths, closed under the same canonical JSON rules.
Its digest enters the report as `path_policy_sha256`. Classification uses
the policy committed at `B` — a candidate cannot reclassify paths in its
own favor, and a policy change is a trust-surface change the preflight
refuses from candidates (§4). For the rulespec-nz pilot the policy covers
the atomic rule roots; whether `programs/` composition outputs enter the
domain is a §11 decision.

### 3.2 Eligible lineage set

Eligible records for a verification are exactly the records whose body
files are present under `.axiom/lineage/` in `H1` and absent in `B` —
introduced by this candidate — and which carry this `lane` and this
`epoch_sha256`. The sorted digest list is bound into the report. This rule
is what defeats replay: a record merged by an earlier candidate is present
in `B` and therefore ineligible ever after; a record from another lane or
epoch refuses on its bindings; and transitions chain from `B`'s actual
blobs (§3.3), so a stale record whose before-digests no longer match `B`
refuses even inside its introducing candidate.

Records are consumed atomically: all of a record's transitions must be used
by the covering assignment, or the record is unused entirely. A partially
matching record is a refusal, not a partial cover. Unused eligible records
are listed in the report (they are legal — a retried generation — but
never invisible).

### 3.3 The predicate

Compute the tree-entry diff from `B` to `H1` with rename detection
disabled (`git diff-tree --no-renames` semantics): the diff is a set of
per-path entry changes — additions, deletions, and modifications. Renames
do not exist at this level and need no special rule; a moved file is a
deletion needing a deletion transition and an addition needing an addition
transition. Then, over the protected subset:

1. **Closed-world assignment.** Every changed protected path maps to
   exactly one non-forking chain of transitions drawn from eligible
   records — no missing, no overlapping, no extra transitions against
   unchanged paths, no cycles, no forks.
2. **Blob-digest ground truth.** Each chain's first before-digest equals
   the path's blob at `B` (or `null` for an addition); each link's
   after-digest equals the next link's before-digest; the last
   after-digest equals the path's blob at `H1` (or `null` for a deletion).
3. **Admissible entries only.** Protected paths must be regular blobs
   (mode 100644 or 100755) on both sides where they exist. A gitlink,
   symlink, or other mode at a protected path — at `B` or `H1` — is a
   refusal. Protected paths must be valid UTF-8; a non-UTF-8 path inside
   the protected domain is a refusal. File-to-directory and
   directory-to-file replacements decompose into entry deletions and
   additions and are covered as such.
4. **Binary verdict.** `diff_coverage` is `"pass"` or a typed refusal.
   From the enforcement epoch there is no `waived` and no `not-run`; an
   emergency change without a countersigned correction event is a refusal.
   This is deliberately stricter than consumer-side declaration
   completeness, which by design accepts declared `waived`/`not-run`
   outcomes as well-formed.

Advisory shadow runs (compute and publish the verdict without gating) are
permitted only before a lane's enforcement epoch.

## 4. Verifier profile P and trusted preflight

The profile is a committed definition at `B`, digest-bound into the report.
It is not the current apply path, and the differences are the point:

- **Non-mutating.** No deterministic repairs. The subject is verified as
  presented or refused. A repairable-but-unrepaired subject refuses.
- **Oracles on.** The apply overlay's oracle disablement does not exist
  here. Licensed or unavailable oracles follow the §11 decision — fail
  closed, or a visibly reduced-tier receipt; never silent.
- **Reviewers means deterministic checks plus protected-environment human
  approval.** The validator pipeline's LLM reviewers are QA outside the
  admission path, beside golden regeneration; under the charter's
  no-model-calls rule they cannot gate the notary.
- **No caller switches.** Skip flags and caller-disableable guards have no
  notary equivalents.

The trusted preflight (runs in Job 1, from pinned code, before any
candidate code executes) resolves: the canonical repository; the admissible
base (`B` must equal the tip admitted by `previous_receipt` — §6); the
path policy and profile at `B`; the complete tree-entry diff. It refuses
candidate changes to workflows, action pins, verifier pins, trust roots,
the path policy, the profile, waiver policy, repository structure rules,
and the lineage-record store's integrity (§2.6). Those surfaces move only
through separately privileged flows.

## 5. Jobs and capability separation

**Job 1 — verify (secretless).** Runs candidate code under the strict
profile with no signing capability, no model credentials, and no
repository-write credential. Emits the §2.4 report as a content-addressed
artifact.

**Job 2 — sign (typed, notary-only).** Runs in a **dedicated
`notary-signing` environment** — not `production-signing`, which today
provisions generation workflows with the apply key, a model credential,
and later a write-capable token, and is therefore structurally
disqualified from holding the notary key. The environment enforces
required human reviewers, protected refs, no self-approval, and
Job-2-only secret access. Job 2 runs no candidate code and holds no model
or repository-write credential. Its trusted component:

1. Reads the Job-1 run from the Actions control plane: workflow path and
   sha, ref, run id and attempt, conclusion, artifact digest. A wrong
   workflow, wrong ref, non-success conclusion, or artifact digest
   mismatch is a refusal.
2. Independently re-fetches and re-hashes `S`, `H1`, `B`, the report body,
   the profile and path-policy files at `B`, and re-derives
   `previous_receipt` admissibility. Any mismatch discards the run and
   forces complete reverification.
3. Reconstructs the §2.5 receipt body from typed fields and signs its
   digest under the notary domain. It exposes no arbitrary-byte signing
   operation.

**Publisher — a third job.** Both verification and signing forbid
repository-write capability, so neither publishes. A separate publisher
job — the pattern the current signed-apply workflow already uses for its
App-token step — holds only a write token scoped to the receipt
publication ref, and pushes per §7. It holds no signing capability.

Trust roots for all jobs — public keys, profile and policy digests,
workflow and action pins, waiver authorities — are committed at or
digest-pinned to the protected base. Runtime organization variables are
not trust anchors anywhere in the notary path; eliminating the existing
`vars.*` provisioning from the signing path is a §9 precondition.

## 6. Admission chain, epoch, and genesis

Admission is a chain, receipt to receipt:

- Every ordinary receipt names `previous_receipt_sha256`, and its
  `base_commit` must be that receipt's `subject_commit`. Verifying any
  other base refuses. An uncovered edit therefore cannot launder itself by
  becoming the base of the next verification: the base that skipped it is
  not the admitted tip.
- **Genesis** is a distinct artifact — `axiom/notary-genesis/v1`, signed
  under its own domain — not an ordinary receipt. It binds: the lane, the
  epoch identifier (`epoch_sha256` = digest of the genesis body), the
  pinned genesis commit and tree, and a **frozen grandfather set**: the
  digest of the exact v5 record and blob pairs present at genesis. Genesis
  records state; it does not assert the baseline content was admitted, and
  it covers nothing. It cannot recur: a second genesis for a lane refuses.
- The first ordinary receipt uses the genesis as `previous_receipt` and
  the genesis commit as `B`.
- **Dual-era rule.** Pre-epoch content stands under the frozen v5
  grandfather set: a v5 manifest vouches post-epoch only for a path whose
  blob is byte-identical to its grandfathered pair. Any post-epoch change
  to any protected path requires notary-era lineage; v5 records cover the
  pre-epoch bytes only, and new v5 records have no post-epoch standing.

Pilot bootstrap on rulespec-nz: pin the lane's current main commit as
genesis. No historical lineage is invented; the baseline is grandfathered
as-is, visibly, in the frozen set — including any pre-epoch hand edits,
which genesis records but does not bless as admitted.

## 7. Publication

The publisher pushes the receipt (and its report) as a detached
attestation ref referencing `subject_commit`, compare-and-swap;
alternatively as a mechanical receipt-only child commit whose sole delta
is the receipt files. If the protected ref or the candidate moves between
verification and publication, the CAS fails, the verification is
discarded, and the run repeats from the new admitted tip. The receipt
distinguishes `subject_commit`, `subject_tree_sha256`, and
`verifier_commit`; the attestation ref supplies the publication location
without entering the signed body.

## 8. Key ceremony

Before Job 2 exists: generate `notary_ed25519` with documented custodians,
fingerprint published in committed code (consumer verification specs pin
the SPKI), storage and rotation procedure, revocation and recovery
procedure. The producer key, actor keys, and reviewer countersignature key
get the same treatment under their own domains. §10's pairwise cross-domain
matrix is part of ceremony acceptance. Rotation is loud, by reviewed change
to committed pins, exactly as the consumer-side verifier already requires.

## 9. Preconditions (nothing admission-capable merges before these)

1. This document approved — threat model, schemas and signature envelope,
   predicate, profile, chain/epoch/genesis rules, and the §10 suite — by
   the charter's gate: an independent cross-family review of the concrete
   design plus **Max's named sign-off**, with every §11 decision closed.
2. The reviewers-semantics resolution in §4 accepted.
3. The dedicated `notary-signing` environment created and enforcing:
   required reviewers, protected refs, no self-approval, Job-2-only secret
   access (#1194 re-scoped to the new environment) — and the generation
   workflows migrated off any environment that will hold the notary key.
4. The authenticated Job-1 handoff and typed Job-2 reconstruction of §5
   implemented in the trusted signing component (no raw-byte notary
   scope).
5. The publisher split of §5 implemented; neither verify nor sign holds a
   write credential.
6. Signed-leg billing and abuse policy operational (#1193).
7. Notary key ceremony completed per §8.
8. Org-variable trust anchors eliminated from the signing path in favor of
   committed or digest-pinned roots.
9. rulespec-nz consumer-side pins landed: the notary SPKI and epoch in the
   lane's committed verification spec; the notary check required and
   non-bypassable in branch protection; genesis activated atomically with
   enforcement.
10. The generated-file guard hardcoded in the protected shared workflow
    (its caller-controlled boolean retired), per the charter's migration
    step 3 — merge-time defense in depth during dual-era operation.
11. Retirement of lane signed-apply legs (#1195) is a cutover exit
    criterion, not a pilot precondition; dual-era operation is expected.

## 10. Negative-test floor

Grouped by where the refusal fires; each case is a distinct test before
the pilot gates anything.

**Lineage and records:** malformed record schema; unknown field; wrong
content-address filename; unauthenticated generation record (producer
signature invalid or wrong key); correction without countersignature;
countersignature by the actor; countersignature by an untrusted key;
mutated existing record; deleted existing record; record with wrong lane;
record with wrong epoch.

**Eligibility and coverage:** record present at `B` (replay); uncovered
protected change; overlapping chains; extra transition against an
unchanged path; forked chain; cyclic chain; discontinuous chain (wrong
intermediate digest); wrong starting blob; wrong terminal blob; malformed
null-sides (addition with before-digest; deletion with after-digest);
partial record consumption; wrong patch digest or unknown
`patch_algorithm` (audit-metadata refusal); gitlink at a protected path;
symlink at a protected path; disallowed mode; non-UTF-8 protected path;
`waived`/`not-run` coverage outcome post-epoch.

**Preflight and profile:** candidate touching workflows, pins, trust
roots, path policy, profile, waiver policy, or lineage-store integrity;
repairable-but-unrepaired subject; failed gate; oracle-disabled attempt;
skip-flag attempt; profile or path policy not committed at `B`.

**Chain, epoch, genesis:** base not the admitted tip; wrong
`previous_receipt`; second genesis; genesis with a non-empty covered set;
ordinary receipt claiming genesis; post-epoch change vouched only by v5;
v5 record whose blob differs from its frozen grandfathered pair.

**Job 2 and publication:** report-body digest mismatch; stale `B` or `H1`
at re-hash; Job-1 run from the wrong workflow or ref; non-success Job-1
conclusion; artifact digest mismatch; replayed Job-1 artifact under a new
run id; malformed typed receipt fields; invalid authorization reference;
CAS loss; receipt-child commit with any extra delta.

**Domains:** the full pairwise signature matrix across
`lineage-generation/v1`, `lineage-correction/v1`,
`lineage-correction-review/v1`, `notary-genesis/v1`, `notary-receipt/v1`,
`apply_ed25519`, and `eval_ed25519` — every cross-domain verification
fails.

## 11. Decisions for sign-off

- ProgramSpec scope: atomic RuleSpec only in the pilot (recommended), or
  extend the path policy to composition outputs.
- Licensed or unavailable oracles: fail closed, or visibly reduced-tier
  receipt.
- Approval wording: `authorization` binds durable digest-bound reviewer
  evidence, or records "the protected signing policy authorized this
  receipt" (honest for plain environment approval, which approves a
  deployment, not a displayed digest; the stronger form needs an explicit
  approval artifact).
- Custody model for the producer, actor, and reviewer keys (the notary key
  is fixed by §5/§8); reviewer custody is the open question deferred from
  the rulespec-nz custody ruling.

## 12. Out of scope for milestone one

Witnessed lineage chains (dual RFC 3161 — sequenced behind the notary as
chartered); historical backfill; rename *modeling* (tree-entry
decomposition makes it unnecessary); fleet-wide shared-workflow
conversion; v5 retirement; the other eight lanes; ProgramSpec admission
unless §11 decides otherwise.
