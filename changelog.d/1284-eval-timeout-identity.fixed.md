EncodeBench execution identity v3 now binds the effective Claude, Codex, and
OpenAI timeout policies, the per-case generation/retry wall budget, and
generation retry limits. That budget charges each attempt's source, workspace,
and prompt setup and bounds backend waits, OpenAI requests and backoffs,
responses that arrive after the deadline, artifact materialization,
empty-artifact retries, and all suite retries. Artifacts whose materialization
crosses the deadline are securely discarded and persist as terminal case-budget
timeouts. Each runner-case receives an independent fresh budget, so runner
order and runner-set membership cannot consume another runner's allowance or
suppress its retries. Deterministic artifact validation and optional reviewers
pause rather than consume the generation budget, so they cannot suppress a
permitted generation retry and are not misrepresented as preemptible work.
Capability boards require the complete v3 identity, validate every nested
checkout and nonempty directory RuleSpec-root field, verify the complete sealed
PolicyEngine runtime-v2 identity and wrapper digest, require that runtime
whenever result rows carry PolicyEngine oracle evidence, and refuse missing or
unexpected toolchain fields. Git checkout identities must carry exactly the
producer-owned scope for the encoder and each RuleSpec root, while the rules
engine must remain whole-checkout scoped; a real producer-to-board admission
lock protects that contract. Terminal encoder timeout evidence stops suite-level
retries, bounding Claude and Codex timeout paths to the two documented
artifact-generation attempts instead of six.
Runtime retry loops now consume the same bound constants, and completed
historical runs recover their digest-verified suite retry count before
rebuilding the live identity, so nondefault policies remain verifiable without
trusting other persisted execution fields.
