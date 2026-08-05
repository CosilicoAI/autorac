# Changelog

All notable changes to Axiom Encode will be documented here.

- Recognize line-leading capital dotted legal subsections such as `A.` and `B.`
  as complete-source structural boundaries. This prevents a preceding
  parenthesized paragraph from absorbing a later subsection and producing
  impossible mixed formula obligations.

- Reject source-add dispatches when the primary or bundled canonical RuleSpec
  destination already exists in the pinned checkout. Existing modules must use
  the authenticated canonical-refresh contract, so mode mistakes fail before
  model execution or signing instead of reaching an invalid full-tree compile.

- Classify affirmative duties introduced by a `notwithstanding` exemption as
  enabling exception effects, so complete-source companion tests can prove
  obligations that remain operative despite a separate filing exemption.

- Add a broker-authenticated `migrate-rulespec-paths` workflow for
  current-v5 model encodings. It permits only deterministic engine-safe path
  normalization and exact durable-reference rewrites, preserves legal corpus
  citations and nested model provenance, binds the clean repository base/tree,
  corpus, waiver set, plan, paths, and hashes in a signed receipt, and installs
  the complete migration transactionally. Persisted verification recomputes
  every move and dependent rewrite from the receipt-bound Git base, enforces the
  current encoder identity, uses no-follow bounded reads and sanitized Git
  inspection, preserves path-only `0644` file semantics in live verification,
  and reuses one digest-bound receipt proof per multi-manifest operation.
