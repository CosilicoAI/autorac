"""Construct an actual evaluation result around synthetic transport fixtures."""

import hashlib
from pathlib import Path

from axiom_encode.harness.evals import EvalResult


def candidate_result(candidate: Path) -> EvalResult:
    return EvalResult(
        citation="zz/policies/synthetic",
        runner="synthetic",
        backend="synthetic",
        model="synthetic",
        mode="cold",
        output_file=str(candidate),
        trace_file="",
        context_manifest_file="",
        generated_output_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        trace_sha256=None,
        context_manifest_sha256=None,
        duration_ms=0,
        success=True,
        error=None,
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        reasoning_output_tokens=0,
        estimated_cost_usd=None,
        actual_cost_usd=None,
        retrieved_files=[],
        unexpected_accesses=[],
        metrics=None,
    )
