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
and Codex terminal envelopes are checked explicitly, and output from any
receiver error or terminal partial is cleared before artifact materialization
so it cannot be scored.
Capability boards render overflow, truncation, and integrity failures
distinctly and never score their artifacts. Eval suites preflight each local
CLI's version and required flags once before case dispatch, execute that exact
binary, and require Claude/Codex versions plus the Codex executable digest, and
OpenAI endpoint, response model, service tier, and request ceiling, in
result/verdict schema v7.
