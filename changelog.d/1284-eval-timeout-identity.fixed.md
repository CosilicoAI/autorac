EncodeBench execution identity v3 now binds the effective Claude, Codex, and
OpenAI timeout policies, the per-case generation/retry wall budget, and
generation retry limits. That budget charges each attempt's source, workspace,
and prompt setup and bounds backend waits, OpenAI requests and backoffs,
responses that arrive after the deadline, artifact materialization,
empty-artifact retries, and all suite retries. Artifacts whose materialization
crosses the deadline are securely discarded from both the result tree and any
direct-write backend workspace, then persist as terminal case-budget timeouts.
Each runner-case receives an independent fresh budget, so runner order and
runner-set membership cannot consume another runner's allowance or suppress its
retries. Deterministic artifact validation and optional reviewers pause rather
than consume the generation budget, so they cannot suppress a permitted
generation retry and are not misrepresented as preemptible work.
Capability boards require the complete v3 identity, validate every nested
checkout and nonempty directory RuleSpec-root field, verify the complete sealed
PolicyEngine runtime-v2 identity and wrapper digest (including its canonical
repository and trusted import-path topology), require that runtime whenever
result rows carry PolicyEngine oracle evidence, and refuse missing or unexpected
toolchain fields. Admission also mirrors the producer's official/checkout tree
count equality, canonical RuleSpec runtime-pin path, and Python-version/stdlib
binding. Cross-host PolicyEngine paths normalize to stable semantic anchors
while import order, relative module origins, and sealed-root layout remain
score-affecting. Git checkout identities must carry exactly the producer-owned
scope for the encoder and each RuleSpec root, while the rules engine must remain
whole-checkout scoped; real producer-to-board admission locks protect both
contracts. Every durable row's admission context must bind back to the suite's
exact execution identity and digest, and PolicyEngine metrics must bind to that
same sealed runtime. Terminal encoder timeout evidence stops suite-level
retries, bounding Claude and Codex timeout paths to the two documented
artifact-generation attempts instead of six.
Runtime retry loops now consume the same bound constants, and completed
historical runs recover their digest-verified suite retry count before
rebuilding the live identity, so nondefault policies remain verifiable without
trusting other persisted execution fields.
