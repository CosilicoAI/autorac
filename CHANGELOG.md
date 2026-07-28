# Changelog

All notable changes to Axiom Encode will be documented here.

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
