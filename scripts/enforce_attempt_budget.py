#!/usr/bin/env python3
"""Fail-fast budget for repeated failed re-encode dispatches of one citation.

Motivation (axiom-encode#1495): individual citations burned 20-31 targeted
re-encode runs in a single week at API prices, all of them ad hoc
dispatches retrying a persistently failing gate. This guard runs as a
cheap unprotected job before the production-signing environment gate and
before any model call. When the same citation has already failed
``attempt_budget`` consecutive times (no intervening success) inside the
lookback window, an ad hoc dispatch is blocked with triage instructions.

Queue-driven dispatches never reach this script in CI — the
attempt_budget job's own condition skips them so the guard can never
stall a tranche. The ``queue_id`` report-only branch below survives as
defense in depth for odd dispatch paths and local use.

The guard is a cost damper, not a security gate: if the GitHub API cannot
be reached it fails open with a loud warning. Stdlib-only because it runs
on a bare runner before any toolchain setup.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

DEFAULT_BUDGET = 3
DEFAULT_LOOKBACK_DAYS = 7
# Conclusions that consumed model spend and count against the budget.
FAILURE_CONCLUSIONS = {"failure", "timed_out"}
# Conclusions that neither consume budget nor reset the streak.
NEUTRAL_CONCLUSIONS = {
    "cancelled",
    "skipped",
    "neutral",
    "action_required",
    "startup_failure",
    "stale",
}
MAX_RUN_PAGES = 5
RUNS_PER_PAGE = 100
# Job name of the protected encode job; a counted failure must have reached
# it (not skipped) to prove the run actually spent model tokens. Runs this
# guard itself blocked conclude "failure" with the encode job skipped and
# must not feed back into the streak.
ENCODE_JOB_NAME = "Queue protected signed RuleSpec re-encode"
ENCODE_STEP_NAME = "Encode, review, validate, and apply"
# At most this many per-run jobs lookups per evaluation; beyond the cap a
# failure is counted without verification (conservative toward blocking).
MAX_JOB_LOOKUPS = 10


@dataclass
class Decision:
    """Outcome of evaluating the failed-attempt streak for a citation."""

    streak: int
    budget: int
    counted_runs: list[dict] = field(default_factory=list)

    @property
    def exhausted(self) -> bool:
        return self.streak >= self.budget


def run_citation(run_name: object) -> str | None:
    """Extract the citation from a targeted re-encode run name.

    Run names look like ``Targeted signed RuleSpec re-encode
    [<queue>:<item>:<sha>] <citation>``; citations are corpus paths and
    never contain ``"] "``, so the segment after the last ``"] "`` is the
    citation.
    """
    if not isinstance(run_name, str) or "] " not in run_name:
        return None
    return run_name.rsplit("] ", 1)[-1]


def evaluate_attempt_budget(
    runs: list[dict],
    *,
    citation: str,
    budget: int,
    current_run_id: int,
    spent_model_tokens: Callable[[dict], bool] | None = None,
) -> Decision:
    """Count consecutive completed failures for ``citation``.

    ``runs`` may arrive in any order; they are sorted newest-first here.
    The streak counts failure conclusions from the most recent completed
    run backwards and stops at the first success. Neutral conclusions
    (cancelled and friends) are skipped without resetting the streak.

    ``spent_model_tokens`` classifies whether a run actually reached the
    encode job. It filters both directions: failure runs that never
    reached a model call — above all, runs this guard itself blocked — do
    not extend the streak, and phantom successes that never ran encode
    (e.g. a queue run re-run where encode requires attempt 1, leaving
    only successful/skipped jobs) do not reset it. Both are neutral.
    """
    decision = Decision(streak=0, budget=budget)
    matching = [
        run
        for run in runs
        if run_citation(run.get("name")) == citation
        and run.get("id") != current_run_id
        and run.get("status") == "completed"
    ]
    matching.sort(key=lambda run: str(run.get("created_at") or ""), reverse=True)
    for run in matching:
        conclusion = run.get("conclusion")
        if conclusion == "success":
            if spent_model_tokens is not None and not spent_model_tokens(run):
                continue
            break
        if conclusion in NEUTRAL_CONCLUSIONS:
            continue
        if conclusion in FAILURE_CONCLUSIONS:
            if spent_model_tokens is not None and not spent_model_tokens(run):
                continue
            decision.streak += 1
            decision.counted_runs.append(
                {
                    "id": run.get("id"),
                    "created_at": run.get("created_at"),
                    "conclusion": conclusion,
                    "html_url": run.get("html_url"),
                }
            )
            continue
        # Unknown conclusion values are treated as neutral so that new
        # GitHub statuses cannot spuriously block encodes.
    return decision


def make_encode_job_checker(
    fetch_jobs: Callable[[int], list[dict]],
    *,
    max_lookups: int = MAX_JOB_LOOKUPS,
    encode_job_name: str = ENCODE_JOB_NAME,
    encode_step_name: str = ENCODE_STEP_NAME,
) -> Callable[[dict], bool]:
    """Build a ``spent_model_tokens`` classifier backed by the jobs API.

    A run counts as model-spending only when its encode job contains the
    actual encode step with a non-skipped conclusion. Looking only at the
    enclosing job is insufficient: trusted-repair and other preflight steps
    can fail that job before the model runs. Past ``max_lookups`` (or on
    lookup errors) the classification is conservative toward blocking in both
    directions: unverified failures count against the budget and
    unverified successes stay neutral rather than resetting the streak.
    Bounded so a long history cannot stall the guard.
    """
    lookups = 0

    def _unverified(run: dict) -> bool:
        return run.get("conclusion") != "success"

    def spent_model_tokens(run: dict) -> bool:
        nonlocal lookups
        run_id = run.get("id")
        if not isinstance(run_id, int) or lookups >= max_lookups:
            return _unverified(run)
        lookups += 1
        try:
            jobs = fetch_jobs(run_id)
        except Exception:
            return _unverified(run)
        encode_jobs = [
            job
            for job in jobs
            if isinstance(job, dict) and job.get("name") == encode_job_name
        ]
        if not encode_jobs:
            return False
        encode_steps = [
            step
            for job in encode_jobs
            for step in (job.get("steps") or [])
            if isinstance(step, dict) and step.get("name") == encode_step_name
        ]
        return any(step.get("conclusion") != "skipped" for step in encode_steps)

    return spent_model_tokens


def _fetch_completed_runs(
    *,
    repo: str,
    workflow_file: str,
    token: str,
    lookback_days: int,
) -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    runs: list[dict] = []
    for page in range(1, MAX_RUN_PAGES + 1):
        query = urllib.parse.urlencode(
            {
                "status": "completed",
                "created": f">={since}",
                "per_page": str(RUNS_PER_PAGE),
                "page": str(page),
            }
        )
        url = (
            f"https://api.github.com/repos/{repo}/actions/workflows/"
            f"{workflow_file}/runs?{query}"
        )
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "axiom-encode-attempt-budget",
            },
        )
        payload = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = json.load(response)
                break
            except (urllib.error.URLError, TimeoutError, ValueError):
                if attempt == 0:
                    time.sleep(2)
                else:
                    raise
        page_runs = (payload or {}).get("workflow_runs") or []
        runs.extend(run for run in page_runs if isinstance(run, dict))
        if len(page_runs) < RUNS_PER_PAGE:
            break
    return runs


def _fetch_run_jobs(*, repo: str, token: str, run_id: int) -> list[dict]:
    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=30"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "axiom-encode-attempt-budget",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    jobs = (payload or {}).get("jobs") or []
    return [job for job in jobs if isinstance(job, dict)]


def _append_summary(lines: list[str]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _render_summary(
    decision: Decision, *, citation: str, enforced: bool, blocked: bool
) -> list[str]:
    lines = ["### Failed-attempt budget", ""]
    lines.append(
        f"- Citation: `{citation}` — {decision.streak} consecutive failed "
        f"run(s) in the lookback window (budget {decision.budget})."
    )
    for run in decision.counted_runs:
        lines.append(
            f"  - {run.get('created_at')} — {run.get('conclusion')} — "
            f"{run.get('html_url')}"
        )
    if blocked:
        lines.append(
            "- **Blocked before the signing environment and any model call.** "
            "Triage the failing gate first (see axiom-encode#1494 for why "
            "regeneration alone rarely fixes these). To proceed anyway, set "
            "the repository variable `ATTEMPT_BUDGET_OVERRIDE` to `true` (or "
            "raise `ATTEMPT_BUDGET`), re-dispatch, and unset it after triage."
        )
    elif not enforced:
        lines.append(
            "- Report-only: queue-driven dispatch, or override was set; not blocking."
        )
    return lines


def main() -> int:
    citation = os.environ.get("CITATION", "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GH_TOKEN", "")
    workflow_file = os.environ.get("WORKFLOW_FILE", "targeted-signed-reencode.yml")
    queue_id = os.environ.get("QUEUE_ID", "").strip()
    override = os.environ.get("ATTEMPT_BUDGET_OVERRIDE", "").strip().lower()
    try:
        budget = int(os.environ.get("ATTEMPT_BUDGET", str(DEFAULT_BUDGET)))
    except ValueError:
        budget = DEFAULT_BUDGET
    if budget < 1:
        budget = DEFAULT_BUDGET
    try:
        lookback_days = int(os.environ.get("LOOKBACK_DAYS", str(DEFAULT_LOOKBACK_DAYS)))
    except ValueError:
        lookback_days = DEFAULT_LOOKBACK_DAYS
    try:
        current_run_id = int(os.environ.get("GITHUB_RUN_ID", "0"))
    except ValueError:
        current_run_id = 0

    if not citation or not repo or not token:
        print(
            "attempt-budget: missing CITATION/GITHUB_REPOSITORY/GH_TOKEN; "
            "failing open.",
            file=sys.stderr,
        )
        return 0

    try:
        runs = _fetch_completed_runs(
            repo=repo,
            workflow_file=workflow_file,
            token=token,
            lookback_days=lookback_days,
        )
    except Exception as error:  # noqa: BLE001 - fail open on API trouble
        print(
            f"attempt-budget: could not list workflow runs ({error}); failing open.",
            file=sys.stderr,
        )
        return 0

    decision = evaluate_attempt_budget(
        runs,
        citation=citation,
        budget=budget,
        current_run_id=current_run_id,
        spent_model_tokens=make_encode_job_checker(
            lambda run_id: _fetch_run_jobs(repo=repo, token=token, run_id=run_id)
        ),
    )
    enforced = queue_id == "" and override != "true"
    blocked = enforced and decision.exhausted
    print(
        f"attempt-budget: citation={citation} streak={decision.streak} "
        f"budget={budget} enforced={enforced} blocked={blocked}"
    )
    if decision.counted_runs or blocked or override == "true":
        _append_summary(
            _render_summary(
                decision, citation=citation, enforced=enforced, blocked=blocked
            )
        )
    if override == "true":
        print("attempt-budget: override set by dispatcher; not blocking.")
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
