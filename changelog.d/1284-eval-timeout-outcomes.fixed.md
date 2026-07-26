Exhausted encoder timeouts now persist as typed terminal outcomes in eval result
rows and render as distinct timeout cells on capability boards without entering
artifact compile, CI, grounding, or reviewer denominators. Durable rows retain
the exact triggering Codex wall/idle or OpenAI connect/read timeout class and
threshold even when a longer case deadline also clamps the request, count every
exhausted OpenAI request attempt (including an alarm wrapped by the HTTP
stack), retain that history across eventual HTTP responses, and preserve
earlier generation-time timeout evidence when an artifact is revalidated.
Codex process completion observed at or after the configured wall deadline is
now rejected using a monotonic clock instead of being admitted as an on-time
artifact. Returned or directly written target artifacts from every timed-out
backend attempt are securely discarded before validation; when the final
attempt times out, timeout classification takes precedence over any apparent
artifact validation failure. Capability boards reject contradictory
terminal-timeout rows that claim a generated artifact. Eval results and
summaries now use schema v6 with explicit artifact and timeout counts, while
machine-readable capability boards use schema `axiom-encode/eval-board/v2` for
the expanded outcome contract.
Board artifact denominators now require producer-shaped, content-bound output,
trace, and context-manifest path/digest pairs. Successful PolicyEngine rows
must carry a passing oracle outcome, passing outcomes may omit an advisory
score, and artifact-bearing failures cannot evade oracle denominators by
dropping their metrics. Board admission also requires the producer's core
artifact digest fields and validates optional validator-verdict path/digest
pairs without treating verdict-only failure evidence as a generated artifact;
the producer applies the same PolicyEngine outcome invariants before persistence.
