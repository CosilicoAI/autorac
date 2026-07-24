# Notary acceptance protocol — design-round draft v0

Status: **DRAFT for the axiom-encode#1192 design round.** Nothing here is
normative until the cross-family gate passes and Max signs off. Terminology:
MUST/SHOULD/MAY per RFC 2119. This document specifies the target admission
architecture; the charter (#1192) records why it was chosen.

## 1. Summary and claim inversion

Today the apply manifest's Ed25519 signature attests *generation provenance*:
"the supervised encoder at a pinned commit produced these bytes from pinned
inputs." That claim cannot be re-executed by anyone (generation is
stochastic), and maintaining it forces re-generation in CI whenever content
needs a signature (Path R), doubling model cost and review load and admitting
human corrections only by laundering them through steered re-runs.

This protocol inverts the signed claim. The signature attests **deterministic
verification plus authorization**:

> Tree `T` was accepted by verifier profile `P` against corpus release `Y`,
> engine `E`, protected base `B`, and waiver set `W`, and the protected
> signing policy authorized this acceptance.

That claim is falsifiable: any party with the inputs can re-run the verifier
and check it. Generation and correction history remain first-class records —
retained, content-addressed, eventually witnessed (receipt#7) — but they are
**lineage, not authority**. No model call ever runs on the canonical CI path.

## 2. Records

### 2.1 Generation event (non-authorizing)

Produced by the operator-side supervised runtime at generation time.

```json
{
  "schema": "axiom/generation-event/v0",
  "output_sha256": "<H0: hash of the raw, as-emitted encoder output>",
  "raw_output_ref": "<content-addressed location of the retained raw output>",
  "model": "<model id>",
  "codex_cli_version": "...", "codex_cli_sha256": "...",
  "prompt_sha256": "...",
  "runtime_identity": { "...": "supervised-runtime attestation fields" },
  "recorded_at": "<UTC>"
}
```

The raw output MUST be retained (not merely hashed); a bare hash proves
divergence but does not let an auditor inspect what changed.

### 2.2 Correction event (non-authorizing)

Records a human or tooling transformation `H0 → H1`:

```json
{
  "schema": "axiom/correction-event/v0",
  "from_sha256": "<H0>", "to_sha256": "<H1>",
  "actor": "<identity>", "reason": "<free text>",
  "diff_sha256": "<hash of the patch>", "recorded_at": "<UTC>"
}
```

Corrections are honest history. The notary never distinguishes corrected from
uncorrected content in its *authority* decision — both pass the identical
verifier — but the lineage lets auditors and drift QA distinguish model error
from later intervention.

### 2.3 Notary receipt (authorizing)

The only record whose signature governs merge. Two-stage: an unsigned
**verification receipt** (Job 1 output) and the signed **notary statement**
(Job 2 output) that embeds the former's digest.

Verification receipt (`axiom/notary-verification-receipt/v0`, content-addressed):

- `subject_commit`, `subject_tree` — the exact verified state. Dirty
  worktrees MUST be refused.
- `base_commit` — the protected-branch ancestor the diff was computed from.
- Dependency identities: corpus release name + content sha256, engine commit,
  encoder version + package identity, verifier profile id.
- `waiver_set_sha256` + count. An acceptance under waivers means "accepted
  under waiver set W," never "correct"; the receipt MUST carry it.
- Per-gate outcomes: `{gate, status, reproducibility}` where reproducibility ∈
  `public` | `restricted-pinned` | `ci-attested`. Gates with unavailable
  restricted inputs (e.g. licensed oracle bundles) MUST fail the run unless a
  visible reduced-tier mode was explicitly requested, and the receipt MUST
  say so. Silent degradation is prohibited. (**DECIDED, Max 2026-07-22**:
  ratified as implemented in the leg-1 profile — fail-closed default,
  explicit `--allow-reduced` yields a visible `passed-reduced` receipt, and a
  genuine oracle discrepancy fails even under `--allow-reduced`.)
- Run identity (encoder version, profile, UTC), `receipt_sha256` self-hash
  over canonical bytes.

Notary statement (`axiom/notary-acceptance/v1`, signed):

- The verification receipt digest, the subject/base identities re-hashed
  independently by Job 2, the authorization context (see §4), the signing
  policy epoch, and the repository + lane identity. Domain-separated signing
  input; see §3.

## 3. Domains and keys

- New scope `notary_ed25519`, new keypair, new trust-root entry. The signing
  input MUST be domain-separated with `axiom/notary-acceptance/v1` baked into
  the constructed bytes (not caller-assembled).
- `apply_ed25519` and every existing v5 manifest are **frozen in meaning**:
  they attest supervised generation, forever. No code path may re-emit,
  re-wrap, or re-interpret a v5 manifest as a notary statement or vice versa.
  Cross-era supersession rules live in §6.
- Key custody is unchanged in kind: broker-only, fd-delivered, Actions-bound,
  environment-gated. The broker learns the new domain; nothing else about
  custody changes. (Signature *format* mechanics later consolidate onto
  `receipt.sign` — TheAxiomFoundation/receipt#7 — with distinct roles for
  notary signer vs generation producer vs operator witness.)

## 4. Protocol

### 4.0 Preflight (trusted)

Resolves canonical repository identity, protected base `B`, candidate commit
`X`, and the complete git-object diff `B..X`. The preflight MUST reject
candidate changes to authority surfaces: workflow files, verifier pins, trust
roots, waiver policy files, repository-structure declarations, executable
modes, symlinks. Those move only through separately privileged flows.

### 4.1 Job 1 — verification (secretless)

Runs candidate code (compile, tests, reviewers, oracles) and therefore MUST
hold no signing secret and no broker capability. Executes verifier profile
`P` — non-mutating, no repairs, oracles on, reviewers on (the strict profile;
first implementation leg) — and emits the verification receipt.

### 4.2 Job 2 — authorization and signing (no candidate code)

Gated by the `production-signing` environment **with required reviewers**
(prerequisite: axiom-encode#1194). The approval therefore covers a *completed*
verification receipt: the reviewer can inspect the exact SHA, diff, and gate
outcomes before approving, because GitHub environment approval happens before
the job starts and Job 1 has already finished.

Job 2 MUST NOT execute candidate code. It independently re-fetches and
re-hashes the immutable subject, validates the verification receipt through
the Actions control plane (run identity, workflow ref, artifact digest),
constructs the notary statement, and requests exactly one `notary_ed25519`
signature from the broker.

Authorization wording (**DECIDED, Max 2026-07-22**): the statement claims
"the protected signing policy authorized this receipt" — exactly what is
provable from the environment gate. Reviewer identity remains queryable
through GitHub's deployment-approval records but is not embedded in the
signed claim.

### 4.3 Publication — the X/X+1 rule

Verifying `X` and adding its receipt produces `X+1`. Either:

1. **Detached**: the receipt references `X` and lives outside the tree
   (release asset / receipts store); or
2. **Mechanical child**: `X+1`'s only delta over `X` is the receipt file,
   created by the notary flow itself and pushed compare-and-swap while the
   branch still points at `X`.

The statement distinguishes `subject_commit`, `subject_tree`,
`attestation_commit`, and `verifier_commit` so no consumer conflates them.
Branch races, squash rewrites, or stale approvals MUST cause verification
failure downstream (the guard recomputes `subject_tree` from what actually
landed).

## 5. Guard semantics (consumer side)

`run-generated-guard` evolves into the notary guard: content changes on a
protected lane require, for every touched protected file, coverage by either
(a) a valid notary-v1 statement whose `subject_tree` matches, or (b) an
untouched-since-epoch file covered by a frozen v5 manifest. The guard MUST be
hardcoded in the protected shared workflow (no caller-controlled disable —
TheAxiomFoundation/.github#55); migration bypasses live in a differently
named, non-required check.

## 6. Migration — dual-era, per-repo epochs

1. Ship the dual-class verifier (v5 frozen + notary-v1) with distinct keys.
2. Per-repo enforcement epoch recorded in protected configuration. After the
   epoch: every new/changed/deleted protected file requires notary-v1; v5
   covers only untouched pre-epoch content.
3. Cross-class rules: a notary statement supersedes a v5 manifest for a file
   it covers; nothing downgrades notary-v1 → v5. Retirement/deletion flows
   learn the new class.
4. Backfill: re-verify + notarize each lane against its exact
   release/toolchain/waiver tuple (no model calls, no re-review of reviewed
   content), then retire the CI model-generation surfaces
   (targeted-signed-reencode generation leg; lane signed-apply legs —
   axiom-encode#1195).

## 7. Threat notes (accepted, bounded, mitigated)

- A verifier soundness bug deterministically accepts wrong law → mitigations:
  strict profile breadth, oracle execution, golden-regeneration drift QA
  (retained unchanged as the distributional check), shrink-only waivers.
- Candidate-weakened tests → preflight authority-surface rejection + reviewer
  gate + oracle independence.
- Verifier compromise must not become signing capability → the Job 1/Job 2
  capability split is load-bearing; a combined job is a rejected
  implementation.
- Replay/rollback/cross-lane reuse → repository, lane, base, tree, policy
  epoch, and domain bindings in the statement.
- Reviewer compromise or rubber-stamping → required reviewers named per lane;
  the receipt records who/what authorized.

## 8. Out of scope for v1 (explicit)

- ProgramSpec / composition outputs (**DECIDED, Max 2026-07-22**): v1 admits
  atomic RuleSpec only, matching the current manifest boundary; ProgramSpecs
  keep their compose admission path. Extension is a later, separately gated
  round.
- Witnessed lineage chains (dual RFC 3161): sequenced behind the notary
  (receipt#7); until then generation/correction records are self-asserted
  operator metadata plus git history.
- Tier-A batching mechanics (one receipt per immutable wave tree, per-item
  results retained): designed when Tier-A scheduling is real; the protocol
  above does not preclude it.

## 9. Acceptance criteria for this design round

1. Cross-family review (sol conditions from the 2026-07-21 round-2 report are
   the starting checklist) with no unresolved CONFIRMED objection.
2. ~~The three open decisions decided by Max and recorded here~~ **DONE
   2026-07-22** (§2.3 ratified-as-built, §4.2 narrow wording, §8 atomic-only).
3. ~~Required reviewers configured on `production-signing` (#1194)~~ **DONE
   2026-07-22** (MaxGhenis required on `production-signing` and
   `signing-key-migration`; API-verified).
4. The strict verifier profile leg merged with its receipt schema marked
   provisional-consistent with §2.3.
