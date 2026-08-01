__version__ = "0.2.1201"
# Axiom Encode - AI-assisted RuleSpec encoding
# Self-contained encoding infrastructure -- no external plugin dependencies.


from .constants import (
    DEFAULT_CLI_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OPENAI_MODEL,
    REVIEWER_CLI_MODEL,
)
from .harness import (
    AgentSDKBackend,
    # Calibration Metrics
    CalibrationMetrics,
    CalibrationSnapshot,
    ClaudeCodeBackend,
    CodexCLIBackend,
    ComplexityFactors,
    # Encoder Backends
    EncoderBackend,
    EncoderRequest,
    EncoderResponse,
    # Encoding DB
    EncodingDB,
    EncodingRun,
    EvalResult,
    EvalRunnerSpec,
    Iteration,
    IterationError,
    PipelineResult,
    PredictionScores,
    ProofValidationResult,
    ReviewResult,
    ReviewResults,
    ValidationResult,
    # Validator Pipeline
    ValidatorPipeline,
    compute_calibration,
    create_run,
    evaluate_artifact,
    find_rulespec_proof_issues,
    get_calibration_trend,
    parse_runner_spec,
    print_calibration_report,
    run_model_eval,
    save_calibration_snapshot,
    validate_file,
    validate_rulespec_proofs,
)

__all__ = [
    "__version__",
    "DEFAULT_MODEL",
    "DEFAULT_CLI_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "REVIEWER_CLI_MODEL",
    "EncodingDB",
    "EncodingRun",
    "ComplexityFactors",
    "IterationError",
    "Iteration",
    "ReviewResult",
    "ReviewResults",
    "create_run",
    "ValidatorPipeline",
    "ValidationResult",
    "PipelineResult",
    "validate_file",
    "ProofValidationResult",
    "find_rulespec_proof_issues",
    "validate_rulespec_proofs",
    # Encoder Backends
    "EncoderBackend",
    "ClaudeCodeBackend",
    "CodexCLIBackend",
    "AgentSDKBackend",
    "EncoderRequest",
    "EncoderResponse",
    "PredictionScores",
    "EvalRunnerSpec",
    "EvalResult",
    "parse_runner_spec",
    "evaluate_artifact",
    "run_model_eval",
    # Calibration Metrics
    "CalibrationMetrics",
    "CalibrationSnapshot",
    "compute_calibration",
    "print_calibration_report",
    "save_calibration_snapshot",
    "get_calibration_trend",
]
