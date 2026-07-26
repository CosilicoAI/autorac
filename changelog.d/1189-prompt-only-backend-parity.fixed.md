Claude, Codex, and OpenAI eval runners now receive the same prompt bytes with
the complete source and every declared context file inlined. Context is never
silently truncated or skipped: prompts outside the shared receiver envelope
fail as `context_overflow`. Claude keeps tools disabled, while Codex runs
read-only in a fresh empty scratch workspace and treats undeclared reads as
terminal integrity failures.

OpenAI Responses must report a completed response and completed output, use the
model's 128,000-token output ceiling, and reject incomplete or max-token output
as `output_truncated`; Agent API max-token stops are likewise rejected.
Capability boards render overflow, truncation, and integrity failures
distinctly and never score their artifacts. Eval suites preflight each local
CLI's version and required flags once before case dispatch, execute that exact
binary, and bind Claude/Codex versions plus OpenAI endpoint, response model,
service tier, and request ceiling into result/verdict schema v7.
