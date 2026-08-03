# Model constants — single source of truth for all model references.
# Change the model in ONE place, it changes everywhere.

# Full model ID for Agent SDK (API) backend
DEFAULT_MODEL = "claude-opus-4-6"

# Short-form model names for Claude Code CLI backend
DEFAULT_CLI_MODEL = "opus"
REVIEWER_CLI_MODEL = "opus"

# Default model for direct OpenAI Responses and Codex-backed GPT evals.
# This is also the *generator* family the LLM judges must never share (see
# below): same-model self-review correlates errors (the 9/9 identical
# hardcoded-600,000 incident is the cautionary tale).
DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
DEFAULT_OPENAI_ESCALATION_MODEL = "gpt-5.6-sol"
DEFAULT_OPENAI_ESCALATE_AFTER = 2

# LLM judge models (maximum-traceability part 2). Cross-family by design: the
# generator is a GPT model, so judges run on a Claude-family model. Volume runs
# on Haiku; low-confidence verdicts escalate to Sonnet. Both are overridable via
# AXIOM_JUDGE_MODEL / AXIOM_JUDGE_ESCALATION_MODEL.
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_ESCALATION_MODEL = "claude-sonnet-4-5"

# Canonical RuleSpec filesystem contract. ``programs`` is canonical content,
# but it contains declarative axiom-compose ProgramSpecs rather than atomic
# ``rulespec/v1`` modules. Every encoder, validator, signer, manifest, import,
# proof, waiver, concept, judge, and source-hash surface must use the atomic
# four-root set; layout/routing checks alone use all five filesystem roots.
RULESPEC_COMPOSITION_SPEC_ROOT = "programs"
RULESPEC_ATOMIC_MODULE_ROOTS = frozenset(
    {"legislation", "policies", "regulations", "statutes"}
)
RULESPEC_FILESYSTEM_ROOTS = frozenset(
    {*RULESPEC_ATOMIC_MODULE_ROOTS, RULESPEC_COMPOSITION_SPEC_ROOT}
)
RULESPEC_FILE_SUFFIX = ".yaml"
RULESPEC_TEST_FILE_SUFFIX = ".test.yaml"

# Bounded number of atomic source modules that one protected encode may
# generate or reuse. The TY2026 state-PIT inventory currently needs at most 29
# additional source modules for one target.
MAX_SIGNED_SOURCE_MODULES = 64
