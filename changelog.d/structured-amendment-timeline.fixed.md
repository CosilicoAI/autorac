Inject the full newest-first amendment timeline when ingest metadata explicitly
targets a consolidation document or one of its provisions. Structured targets
are authoritative machine-readable declarations needed for legal time slices,
including in-force amendments and forward-dated overlays; the legacy two-act cap
now applies only to name-tier false-positive matching. The existing 12,000-body
and 32,000-aggregate context caps remain the volume bounds. On aggregate overflow,
name-tier documents are dropped oldest-first before any structured document;
structured documents are dropped oldest-first only when no name-tier document
remains, and every omission is recorded in the workspace manifest. The legacy
name-tier-only path now also records documents omitted by its two-document and
aggregate-context limits without changing rendered context; its known separator
under-accounting remains tracked separately in #1432.

Closes #1428.
