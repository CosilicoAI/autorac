"""Tests for the failed-attempt budget guard (axiom-encode#1495)."""

from __future__ import annotations

import pytest

import scripts.enforce_attempt_budget as budget_mod
from scripts.enforce_attempt_budget import (
    ENCODE_JOB_NAME,
    ENCODE_STEP_NAME,
    evaluate_attempt_budget,
    make_encode_job_checker,
    run_citation,
)

CITATION = "us-ky/statute/krs/141.020/document-1"


def _run(
    run_id: int,
    created_at: str,
    conclusion: str,
    *,
    citation: str = CITATION,
    queue: str = "adhoc:adhoc:adhoc",
    status: str = "completed",
) -> dict:
    return {
        "id": run_id,
        "name": f"Targeted signed RuleSpec re-encode [{queue}] {citation}",
        "created_at": created_at,
        "status": status,
        "conclusion": conclusion,
        "html_url": f"https://github.com/org/repo/actions/runs/{run_id}",
    }


class TestRunCitation:
    def test_adhoc_run_name(self) -> None:
        name = f"Targeted signed RuleSpec re-encode [adhoc:adhoc:adhoc] {CITATION}"
        assert run_citation(name) == CITATION

    def test_queue_run_name(self) -> None:
        name = f"Targeted signed RuleSpec re-encode [q1:item2:abc123] {CITATION}"
        assert run_citation(name) == CITATION

    def test_name_without_bracket_separator(self) -> None:
        assert run_citation("some unrelated run") is None

    def test_non_string_name(self) -> None:
        assert run_citation(None) is None


class TestEvaluateAttemptBudget:
    def test_consecutive_failures_reach_budget(self) -> None:
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "failure"),
            _run(3, "2026-08-12T03:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs, citation=CITATION, budget=3, current_run_id=99
        )
        assert decision.streak == 3
        assert decision.exhausted
        assert [run["id"] for run in decision.counted_runs] == [3, 2, 1]

    def test_success_resets_streak(self) -> None:
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "success"),
            _run(3, "2026-08-12T03:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs, citation=CITATION, budget=3, current_run_id=99
        )
        assert decision.streak == 1
        assert not decision.exhausted

    def test_cancelled_runs_do_not_break_or_count(self) -> None:
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "cancelled"),
            _run(3, "2026-08-12T03:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs, citation=CITATION, budget=2, current_run_id=99
        )
        assert decision.streak == 2
        assert decision.exhausted

    def test_timed_out_counts_as_failure(self) -> None:
        runs = [_run(1, "2026-08-12T01:00:00Z", "timed_out")]
        decision = evaluate_attempt_budget(
            runs, citation=CITATION, budget=1, current_run_id=99
        )
        assert decision.streak == 1
        assert decision.exhausted

    def test_unknown_conclusion_is_neutral(self) -> None:
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "some_future_conclusion"),
            _run(3, "2026-08-12T03:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs, citation=CITATION, budget=3, current_run_id=99
        )
        assert decision.streak == 2

    def test_current_run_excluded(self) -> None:
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs, citation=CITATION, budget=2, current_run_id=2
        )
        assert decision.streak == 1
        assert not decision.exhausted

    def test_other_citations_ignored(self) -> None:
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(
                2,
                "2026-08-12T02:00:00Z",
                "failure",
                citation="us-ms/statute/27-7-5",
            ),
        ]
        decision = evaluate_attempt_budget(
            runs, citation=CITATION, budget=2, current_run_id=99
        )
        assert decision.streak == 1

    def test_incomplete_runs_ignored(self) -> None:
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "", status="in_progress"),
        ]
        decision = evaluate_attempt_budget(
            runs, citation=CITATION, budget=1, current_run_id=99
        )
        assert decision.streak == 1

    def test_order_independent_input(self) -> None:
        runs = [
            _run(3, "2026-08-12T03:00:00Z", "failure"),
            _run(1, "2026-08-12T01:00:00Z", "success"),
            _run(2, "2026-08-12T02:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs, citation=CITATION, budget=3, current_run_id=99
        )
        assert decision.streak == 2

    def test_no_matching_runs(self) -> None:
        decision = evaluate_attempt_budget(
            [], citation=CITATION, budget=3, current_run_id=99
        )
        assert decision.streak == 0
        assert not decision.exhausted


class TestEncodeJobChecker:
    """Guard-blocked runs must not feed back into the streak."""

    @staticmethod
    def _jobs_fixture(jobs_by_run: dict[int, list[dict]]):
        calls: list[int] = []

        def fetch(run_id: int) -> list[dict]:
            calls.append(run_id)
            return jobs_by_run[run_id]

        return fetch, calls

    def test_blocked_runs_are_neutral(self) -> None:
        # Runs 2 and 3 were blocked by the guard itself: run-level failure,
        # encode job skipped. Only run 1 actually reached the model.
        fetch, _ = self._jobs_fixture(
            {
                1: [self._encode_job("failure")],
                2: [self._encode_job("skipped")],
                3: [{"name": "Enforce failed-attempt budget", "conclusion": "failure"}],
            }
        )
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "failure"),
            _run(3, "2026-08-12T03:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs,
            citation=CITATION,
            budget=3,
            current_run_id=99,
            spent_model_tokens=make_encode_job_checker(fetch),
        )
        assert decision.streak == 1
        assert not decision.exhausted

    @staticmethod
    def _encode_job(step_conclusion: str, *, job_conclusion: str = "failure") -> dict:
        return {
            "name": ENCODE_JOB_NAME,
            "conclusion": job_conclusion,
            "steps": [
                {"name": ENCODE_STEP_NAME, "conclusion": step_conclusion},
            ],
        }

    def test_preflight_failure_inside_encode_job_is_neutral(self) -> None:
        fetch, _ = self._jobs_fixture(
            {
                1: [
                    {
                        "name": ENCODE_JOB_NAME,
                        "conclusion": "failure",
                        "steps": [
                            {
                                "name": "Resolve trusted prior-run repair candidate",
                                "conclusion": "failure",
                            },
                            {"name": ENCODE_STEP_NAME, "conclusion": "skipped"},
                        ],
                    }
                ]
            }
        )
        decision = evaluate_attempt_budget(
            [_run(1, "2026-08-12T01:00:00Z", "failure")],
            citation=CITATION,
            budget=1,
            current_run_id=99,
            spent_model_tokens=make_encode_job_checker(fetch),
        )
        assert decision.streak == 0
        assert not decision.exhausted

    def test_real_failures_still_count(self) -> None:
        fetch, _ = self._jobs_fixture(
            {run_id: [self._encode_job("failure")] for run_id in (1, 2, 3)}
        )
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "failure"),
            _run(3, "2026-08-12T03:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs,
            citation=CITATION,
            budget=3,
            current_run_id=99,
            spent_model_tokens=make_encode_job_checker(fetch),
        )
        assert decision.streak == 3
        assert decision.exhausted

    def test_lookup_cap_counts_conservatively(self) -> None:
        fetch, calls = self._jobs_fixture(
            {
                run_id: [self._encode_job("skipped", job_conclusion="skipped")]
                for run_id in range(1, 6)
            }
        )
        runs = [
            _run(run_id, f"2026-08-12T0{run_id}:00:00Z", "failure")
            for run_id in range(1, 6)
        ]
        decision = evaluate_attempt_budget(
            runs,
            citation=CITATION,
            budget=3,
            current_run_id=99,
            spent_model_tokens=make_encode_job_checker(fetch, max_lookups=2),
        )
        # Two newest runs verified as blocked (neutral); the remaining three
        # count unverified once the cap is hit.
        assert len(calls) == 2
        assert decision.streak == 3

    def test_fetch_error_counts_conservatively(self) -> None:
        def fetch(run_id: int) -> list[dict]:
            raise RuntimeError("jobs API down")

        runs = [_run(1, "2026-08-12T01:00:00Z", "failure")]
        decision = evaluate_attempt_budget(
            runs,
            citation=CITATION,
            budget=1,
            current_run_id=99,
            spent_model_tokens=make_encode_job_checker(fetch),
        )
        assert decision.streak == 1

    def test_unverified_success_past_cap_stays_neutral(self) -> None:
        # Ten blocked failures (all verified neutral) exhaust the lookup
        # cap; an unverifiable phantom success below them must not reset
        # the streak fed by genuine failures further down.
        jobs_by_run: dict[int, list[dict]] = {
            run_id: [self._encode_job("skipped", job_conclusion="skipped")]
            for run_id in range(5, 15)
        }
        fetch, calls = self._jobs_fixture(jobs_by_run)
        runs = (
            # Genuine failures, oldest.
            [
                _run(run_id, f"2026-08-12T0{run_id}:00:00Z", "failure")
                for run_id in (1, 2, 3)
            ]
            # Phantom success above them (encode skipped, unverifiable
            # once the cap is spent).
            + [_run(4, "2026-08-12T04:00:00Z", "success")]
            # Ten newest guard-blocked failures that consume the cap.
            + [
                _run(run_id, f"2026-08-12T{run_id}:00:00Z", "failure")
                for run_id in range(5, 15)
            ]
        )
        decision = evaluate_attempt_budget(
            runs,
            citation=CITATION,
            budget=3,
            current_run_id=99,
            spent_model_tokens=make_encode_job_checker(fetch, max_lookups=10),
        )
        assert len(calls) == 10
        # Blocked pile neutral, phantom success neutral (unverified),
        # genuine failures below count unverified: streak reaches budget.
        assert decision.streak == 3
        assert decision.exhausted

    def test_phantom_success_does_not_reset_streak(self) -> None:
        # A queue run re-run: encode requires attempt 1, so attempt 2 has
        # only successful/skipped jobs and the run concludes "success"
        # without any encode having succeeded. It must not lift the block.
        fetch, _ = self._jobs_fixture(
            {
                1: [self._encode_job("failure")],
                2: [self._encode_job("failure")],
                3: [self._encode_job("skipped", job_conclusion="skipped")],
                4: [self._encode_job("failure")],
            }
        )
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "failure"),
            _run(3, "2026-08-12T03:00:00Z", "success"),
            _run(4, "2026-08-12T04:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs,
            citation=CITATION,
            budget=3,
            current_run_id=99,
            spent_model_tokens=make_encode_job_checker(fetch),
        )
        assert decision.streak == 3
        assert decision.exhausted

    def test_real_success_still_resets_streak(self) -> None:
        fetch, _ = self._jobs_fixture(
            {
                1: [self._encode_job("failure")],
                2: [self._encode_job("success", job_conclusion="success")],
                3: [self._encode_job("failure")],
            }
        )
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "success"),
            _run(3, "2026-08-12T03:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs,
            citation=CITATION,
            budget=3,
            current_run_id=99,
            spent_model_tokens=make_encode_job_checker(fetch),
        )
        assert decision.streak == 1
        assert not decision.exhausted


class TestMainExitContract:
    """The workflow consumes only main()'s exit code — pin it."""

    FAILING_HISTORY = [
        _run(run_id, f"2026-08-12T0{run_id}:00:00Z", "failure") for run_id in (1, 2, 3)
    ]

    def _set_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        queue_id: str = "",
        repair_run_id: str = "",
        override: str = "",
        budget: str = "",
    ) -> None:
        monkeypatch.setenv("CITATION", CITATION)
        monkeypatch.setenv("GITHUB_REPOSITORY", "org/repo")
        monkeypatch.setenv("GH_TOKEN", "test-token")
        monkeypatch.setenv("GITHUB_RUN_ID", "99")
        monkeypatch.setenv("QUEUE_ID", queue_id)
        monkeypatch.setenv("REPAIR_RUN_ID", repair_run_id)
        monkeypatch.setenv("ATTEMPT_BUDGET_OVERRIDE", override)
        monkeypatch.setenv("ATTEMPT_BUDGET", budget)
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    def _stub_api(
        self,
        monkeypatch: pytest.MonkeyPatch,
        runs: list[dict],
        jobs: list[dict] | None = None,
    ) -> None:
        monkeypatch.setattr(budget_mod, "_fetch_completed_runs", lambda **kwargs: runs)
        monkeypatch.setattr(
            budget_mod,
            "_fetch_run_jobs",
            lambda **kwargs: (
                jobs
                if jobs is not None
                else [TestEncodeJobChecker._encode_job("failure")]
            ),
        )

    def test_blocked_adhoc_exits_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch)
        self._stub_api(monkeypatch, self.FAILING_HISTORY)
        assert budget_mod.main() == 1

    def test_queue_dispatch_reports_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch, queue_id="q-123")
        self._stub_api(monkeypatch, self.FAILING_HISTORY)
        assert budget_mod.main() == 0

    def test_repair_replay_reports_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch, repair_run_id="32201076681")
        self._stub_api(monkeypatch, self.FAILING_HISTORY)
        assert budget_mod.main() == 0

    def test_override_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch, override="true")
        self._stub_api(monkeypatch, self.FAILING_HISTORY)
        assert budget_mod.main() == 0

    def test_under_budget_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch)
        self._stub_api(monkeypatch, self.FAILING_HISTORY[:2])
        assert budget_mod.main() == 0

    def test_invalid_budget_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch, budget="not-a-number")
        self._stub_api(monkeypatch, self.FAILING_HISTORY)
        assert budget_mod.main() == 1

    def test_api_error_fails_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch)

        def boom(**kwargs: object) -> list[dict]:
            raise RuntimeError("api down")

        monkeypatch.setattr(budget_mod, "_fetch_completed_runs", boom)
        assert budget_mod.main() == 0

    def test_missing_env_fails_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "CITATION",
            "GITHUB_REPOSITORY",
            "GH_TOKEN",
            "QUEUE_ID",
            "REPAIR_RUN_ID",
            "ATTEMPT_BUDGET",
            "ATTEMPT_BUDGET_OVERRIDE",
        ):
            monkeypatch.delenv(name, raising=False)
        assert budget_mod.main() == 0
