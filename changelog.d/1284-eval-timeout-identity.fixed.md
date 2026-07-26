EncodeBench execution identity v3 now binds the effective Claude, Codex, and
OpenAI timeout policies, the per-case generation/retry wall budget, and
generation retry limits. That budget bounds backend waits, OpenAI requests and
backoffs, responses that arrive after the deadline, empty-artifact retries, and
all suite retries. Each runner-case receives an independent fresh budget, so
runner order and runner-set membership cannot consume another runner's
allowance or suppress its retries. Deterministic artifact validation and
optional reviewers are post-generation work and are not misrepresented as
preemptible by this budget.
Capability boards require the complete v3 identity and refuse missing or
unexpected toolchain fields. Terminal encoder timeout evidence stops suite-level
retries, bounding Claude and Codex timeout paths to the two documented
artifact-generation attempts instead of six. Runtime retry loops now consume
the same bound constants, and completed historical runs recover their
digest-verified suite retry count before rebuilding the live identity, so
nondefault policies remain verifiable without trusting other persisted
execution fields.
