EncodeBench execution identity v3 now binds the effective Claude, Codex, and
OpenAI timeout policies, the overall per-case wall budget, and generation retry
limits. The shared case budget bounds backend waits, OpenAI requests and
backoffs, and all suite retries. Terminal encoder timeout evidence stops
suite-level retries, bounding Claude and Codex timeout paths to the two
documented artifact-generation attempts instead of six. Runtime retry loops now
consume the same bound constants, and completed historical runs recover their
digest-verified suite retry count before rebuilding the live identity, so
nondefault policies remain verifiable without trusting other persisted
execution fields.
