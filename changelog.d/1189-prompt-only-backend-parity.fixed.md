Claude, Codex, and OpenAI eval runners now receive the same prompt bytes with
the complete source and every declared context file inlined. Context is never
silently truncated or skipped: prompts outside the shared receiver envelope
fail as `context_overflow`. Claude keeps tools disabled, while Codex runs
read-only in a fresh empty scratch workspace. This is detection-based
isolation, not an operating-system sandbox: reported tool activity is a
terminal integrity failure that voids the row, but host-visible reads are not
prevented. Prompt-generated paths are opaque and location-independent, and
disabling corpus context injection now excludes amendment files as well as
their banner. Local CLI prompts are streamed as exact UTF-8 bytes over standard
input, avoiding operating-system command-line size limits.

OpenAI Responses must report a completed response and completed output, use the
model's 128,000-token output ceiling, and reject incomplete or max-token output
as `output_truncated`; Agent API max-token stops are likewise rejected. Claude
and Codex terminal envelopes are checked explicitly, including truncation on
non-success envelopes. Usage-limit diagnostics are classified before raw
receiver text is reduced to hashes and byte counts, allowing the suite circuit
breaker to stop without persisting the diagnostic. Output from any receiver
error or terminal partial is cleared before artifact materialization so it
cannot be scored.
Capability boards render overflow, truncation, and integrity failures
distinctly and never score their artifacts. Eval suites preflight each local
CLI's version and required flags before case dispatch, execute that exact
launcher, and bind both launcher and resolved native-receiver digests into
execution identity v5 and each local result row. Suite results and summaries
use schema v8; OpenAI rows continue to bind the endpoint, response model,
service tier, and request ceiling.
