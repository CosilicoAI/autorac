# Reviewed RuleSpec head admission

The normal targeted signed re-encode workflow starts from an exact commit on a
RuleSpec repository's `main` branch. It does not require an admission record.

An admission record is an exceptional authorization for an independently
reviewed RuleSpec commit that is not yet an ancestor of `main`. Authorization is
fail-closed and applies only to the exact country and 40-character commit SHA in
the record. Updating, merging into, or rebasing the RuleSpec branch creates a new
head that is not covered by the old record.

## Adding a reviewed head

Add exactly one file:

```text
data/reviewed-rulespec-refs/<country>/<rulespec-ref>.json
```

Use canonical, sorted, two-space-indented JSON with a trailing newline:

```json
{
  "country": "us",
  "purpose": "Reviewed repair head for rulespec-us PR 1234.",
  "review_url": "https://github.com/TheAxiomFoundation/axiom-encode/pull/5678",
  "rulespec_ref": "0123456789abcdef0123456789abcdef01234567",
  "schema": "axiom-encode/reviewed-rulespec-ref/v1"
}
```

The `review_url` identifies the existing Axiom Encode issue/PR or country
RuleSpec PR that carries the review context. Keep the RuleSpec branch frozen at
`rulespec_ref` until the protected run has consumed it. Reviewed heads are
artifact-only unless the existing protected-base rules separately authorize
opening a PR.

Admission records are operational policy loaded from the trusted Axiom Encode
checkout. They are deliberately absent from built wheels; outside a repository
checkout, the exceptional-head authorization set is empty. A record-only PR does
not change the encoder package and does not bump the package version. It should
add its own changelog fragment and focused review evidence. Do not combine
unrelated admissions in one record or edit an existing record to authorize a new
SHA.

## Removal

Deleting a record immediately removes that exceptional authorization from new
workflow runs. Make removal a dedicated, reviewed change that explains why the
head is retired. Removing a record does not alter artifacts already produced by
a prior protected run.

This ledger is an interim mechanism. The notary admission architecture tracked
in issue #1192 may replace reviewed-head generation authorization.
