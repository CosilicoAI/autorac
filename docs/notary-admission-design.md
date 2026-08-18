# Notary admission: design v1

Status: draft for sign-off. Implements the #1192 charter with the #1506
diff-coverage delta, under the build decision recorded on both issues
(dual-verdict, 2026-08-17). Nothing admission-capable merges until the
preconditions in §9 are satisfied and this document is approved.

## 1. Claim and threat model

The notary signs one narrow claim:

> Tree `H1` was accepted under verifier profile `P`, protected base `B`,
> corpus release `Y`, waiver set `W`; every content change from `B` to `H1`
> is covered by authenticated lineage; the protected signing policy
> authorized this receipt.

It does not claim content was model-generated, and it never will — that
claim bred provenance laundering (the Path R pattern) and is retired with
the v5 era. Lineage records say where bytes came from; the notary says the
deterministic gates passed and coverage held.

The scenario this design closes: an agent with repository write access
hand-edits a published rule and obtains a legitimate signature through the
sanctioned path. Two independent walls stop it. The capability wall: no
generator holds signing capability — Job 2 alone reaches the notary key,
inside a protected environment with required human reviewers, and runs no
candidate code. The coverage wall: a changed path with no covering lineage
record refuses at preflight, regardless of who approves. A hand edit
therefore either dies unsigned or enters as a declared correction event —
attributed, countersigned, and visible to the fix-the-encoder loop.

Residual risks accepted, per the charter: a verifier soundness bug
deterministically accepts wrong law; candidate-controlled tests can be
weakened; reviewer compromise authorizes validator-passing bad content.
Mitigations are the strict profile, the trusted preflight, golden
regeneration QA, and oracles — not additional signatures.

## 2. Records

Three record kinds, all canonical JSON (receipt-canonical serialization:
UTF-16 code-unit key order, ECMAScript number formatting), content-addressed
by SHA-256, each signed under its own domain with its own key. Domains are
single-purpose byte prefixes; cross-domain verification must fail and §10
tests prove it.

| Record | Domain | Signer | Authorizes merge? |
|---|---|---|---|
| Generation event | `axiom/lineage-generation/v1` | supervised-runtime producer key | No |
| Correction event | `axiom/lineage-correction/v1` | actor key + distinct protected reviewer countersignature | No |
| Notary receipt | `axiom/notary-acceptance/v1` | `notary_ed25519` (Job 2 broker only) | **Yes** |

**Generation event.** Emitted by the supervised local runtime (the #1158
machinery) at generation time. Fields: runtime identity, model, CLI version
and digest, prompt digests, output tree digest, and per-path transitions
(§3). Authenticated as a runtime emission by the producer signature —
content addressing alone proves integrity, not origin.

**Correction event.** A declared human or agent fix. Fields: actor, reason,
predecessor record, per-path transitions, canonical patch digest. Valid only
with a countersignature from a protected reviewer distinct from the actor,
over the event digest. The countersignature validates lineage; it does not
authorize merge. Corrections are first-class and loud — the sanctioned
response to a wrong encoding remains fix-the-encoder-and-re-encode, and a
correction event is the recorded exception, never the quiet path.

**Notary receipt.** The only record Job 2 signs. Fields: `subject_commit`,
`subject_tree` (`H1`), `protected_base` (`B`), verifier profile digest
(`P`), corpus release (`Y`), waiver set digest (`W`), per-gate outcomes with
reproducibility tier (publicly reproducible / restricted pinned inputs /
CI-attested), the diff-coverage verdict, Job 1 run identity, and the
authorization reference. The existing `apply_ed25519` domain and v5
manifests are never reinterpreted; under the dual-era rule they keep their
generator-produced meaning permanently.

Lineage records live in the lane repository under `.axiom/lineage/`,
append-only, one file per record named by its content digest. The lineage
directory is outside the protected-content coverage domain (its additions
need no covering record — that would regress infinitely) but inside the
preflight's structural checks: schema-valid, authenticated, append-only,
correctly content-addressed. Deleting or mutating an existing lineage record
is a preflight refusal.

## 3. The diff-coverage predicate

For the complete git-object diff from `B` to `H1`, restricted to protected
content paths:

1. Every changed path maps to exactly one non-forking chain of transitions
   drawn from lineage records — closed-world set equality: no missing, no
   overlapping, no extra transitions.
2. Each transition binds the full before-blob digest and after-blob digest
   for its path, plus a versioned canonical patch digest (`git-patch-v1`:
   `git diff` with pinned flags, retained for audit; the blob digests are
   ground truth because hunk computation is algorithm-dependent).
3. Replaying each path's chain from its blob at `B` must terminate exactly
   at its blob at `H1`.
4. Additions and deletions are modeled explicitly (null-digest sides).
   Renames, mode changes, and symlinks are rejected in the pilot rather than
   modeled.
5. The verdict is binary. From the enforcement epoch, `waived`, `not-run`,
   missing coverage, or an emergency "correction" without a valid
   countersigned event is a refusal. Notary-v1 acceptance requires exactly
   `diff-coverage=pass` — deliberately stricter than consumer-side
   declaration completeness, which by design accepts declared `waived` and
   `not-run` outcomes as well-formed.

Advisory shadow runs (compute and publish the verdict without gating) are
permitted only before a lane's enforcement epoch, to burn in the predicate.

## 4. Verifier profile P

The notary profile is not the current apply path, and the differences are
the point:

- **Non-mutating.** No deterministic repairs. The subject is verified as
  presented or refused.
- **Oracles on.** The apply overlay's `enable_oracles=False` does not exist
  here. Licensed or unavailable oracles follow the sign-off decision in
  §11 — fail closed, or a visibly reduced-tier receipt; never silent.
- **Reviewers means deterministic checks plus protected-environment human
  approval.** Today's "reviewers" in the validator pipeline are LLM
  reviewers; under the charter's no-model-calls rule they cannot be part of
  the notary profile. LLM review remains QA outside the admission path,
  beside golden regeneration.
- **No skips.** `--skip-reviewers` and caller-disableable guards have no
  notary equivalents. The generated-file guard's merge-time role continues
  during dual-era operation, but the notary does not consult caller
  booleans.

The profile is itself content-addressed; `P` in the receipt is the digest of
the profile definition committed at `B`.

## 5. Two jobs, one capability split

**Job 1 — verify (secretless).** Runs candidate code under the strict
profile in an environment with no signing capability, no model credentials,
and no repository-write credential. Produces the content-addressed unsigned
receipt body: subject digests, gate outcomes and tiers, diff-coverage
verdict, exact dependency pins, run identity.

**Job 2 — sign (narrow, typed).** Runs in the `production-signing`
environment with required reviewers enforced (#1194). Runs no candidate
code. Independently re-fetches and re-hashes `B`, `H1`, and the receipt
body; parses only the typed notary schema; refuses on any mismatch by
forcing complete reverification; holds no model or repository-write
credential; signs only `axiom/notary-acceptance/v1`. Environment approval
gates Job 2 so the human approves a completed receipt, never a
not-yet-run pipeline.

Trust roots for both jobs — public keys, profile digests, policies,
workflow and action pins, waiver authorities — are committed at or
digest-pinned to the protected base. Runtime organization variables are not
trust anchors anywhere in the notary path; the existing `vars.*`
provisioning in the signing workflows is eliminated as a precondition (§9).

## 6. Epoch and bootstrap

Per-repo enforcement epochs, pilot on rulespec-nz:

1. Pin the lane's current main commit as the one-time enforcement genesis.
2. Issue a genesis receipt over the empty `B → H1` range. No historical
   lineage is invented; pre-epoch content is covered by the dual-era rule
   (v5 manifests keep their meaning; untouched content needs nothing).
3. Every subsequent non-empty diff of protected content must pass coverage.
4. Cross-class supersession: a post-epoch change to a v5-covered path
   requires notary-v1 lineage for the change; the v5 record continues to
   describe the pre-epoch bytes only.

## 7. Publication

Receipts publish as detached attestations referencing `subject_commit`,
pushed compare-and-swap; alternatively a mechanical receipt-only child
commit whose sole delta is the receipt. If the protected base or the
candidate moves between verification and publication, the verification is
discarded and rerun from the new `B`. The manifest distinguishes
`subject_commit`, `subject_tree`, `attestation_commit`, and
`verifier_commit` so no reader conflates what was verified with where the
receipt landed.

## 8. Key ceremony

Before Job 2 exists: generate `notary_ed25519` with documented custodians,
fingerprint publication in committed code (consumer verification specs pin
the SPKI), storage and rotation procedure, revocation and recovery
procedure, and negative tests proving apply-domain signatures cannot verify
as notary signatures and conversely. The producer and reviewer keys for
lineage records get the same treatment with their own domains. Rotation is
loud, by reviewed change to committed pins, exactly as the consumer-side
verifier already requires.

## 9. Preconditions (nothing admission-capable merges before these)

1. This document approved — threat model, canonical signed bytes, profile,
   epoch and bootstrap rules, and the §10 suite — by the human gate the
   charter names.
2. The reviewers-semantics resolution in §4 accepted (deterministic checks
   plus human environment approval; LLM review is QA only).
3. `production-signing` environment enforcing required reviewers, protected
   refs, no self-approval, Job-2-only secret access (#1194).
4. Signed-leg billing and abuse policy operational (#1193).
5. Notary key ceremony completed per §8.
6. Org-variable trust anchors eliminated from the signing path in favor of
   committed or digest-pinned roots.
7. Retirement of lane signed-apply legs (#1195) is a cutover exit
   criterion, not a pilot precondition; dual-era operation is expected.

## 10. Negative-test suite (acceptance floor for Job 1)

Each is a distinct refusal with its own test before the pilot gates
anything: uncovered changed path; overlapping chains; extra transition
covering an unchanged path; forked chain; replay terminating at the wrong
blob; correction event without countersignature; countersignature by the
actor; generation event not authenticated by the producer key; mutated or
deleted lineage record; rename, mode change, symlink in the diff; candidate
touching workflows, pins, trust roots, waiver policy, or repository
structure; `waived`/`not-run` coverage outcome post-epoch; stale `B` or
`H1` at Job 2; receipt-body mismatch at Job 2; cross-domain signature
verification in both directions; genesis receipt claiming a non-empty
range.

## 11. Decisions for sign-off

- ProgramSpec scope: atomic RuleSpec only in the pilot (recommended), or
  extend to composition outputs.
- Licensed or unavailable oracles: fail closed, or visibly reduced-tier
  receipt.
- Approval wording: bind durable reviewer evidence, or "the protected
  signing policy authorized this receipt" (recommended; matches the
  environment-approval mechanism).
- Custody model for the three key roles (notary / producer / reviewer):
  broker environment for the notary per §5; producer stays with the
  supervised runtime; reviewer custody is the open question deferred from
  the rulespec-nz custody ruling.

## 12. Out of scope for milestone one

Witnessed lineage chains (dual RFC 3161 — sequenced behind the notary as
chartered); historical backfill; rename/mode/symlink modeling; fleet-wide
shared-workflow conversion; v5 retirement; the other eight lanes; ProgramSpec
admission unless §11 decides otherwise.
