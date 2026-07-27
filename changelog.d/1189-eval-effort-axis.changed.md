Eval runner specs now accept `[name=]backend:model[@effort]`. Omitting the
suffix records and uses the receiver default without sending an effort option.
Declared Codex effort uses the strict `model_reasoning_effort` configuration,
Claude receives `--effort`, and the OpenAI Responses API receives
`reasoning.effort`; unsupported backend-specific levels fail at manifest load.
Execution identity v6 binds each runner's requested effort, capability-board
folds reject effort changes under the same runner name, and board output labels
the column as requested effort. Claude effort is recorded as a request only:
Claude 5 thinking remains adaptive and the flag is not claimed to force a
measurable behavior change.
