from __future__ import annotations

from agent.loop import Task, TaskResult
from agent.memory.learning_service import EpisodeSummary, LearningService, RewardSignal


def test_learning_service_scores_successful_test_run() -> None:
    service = LearningService()
    summary = service.summarize_episode(
        Task(content="run the test suite"),
        TaskResult(
            task=Task(content="run the test suite"),
            output="All tests passed",
            success=True,
            elapsed_ms=120.0,
            status="succeeded",
            tool_calls=2,
        ),
    )

    assert summary.event_kind == "success"
    assert summary.reward.score > 1.0
    assert "task_succeeded" in summary.reward.reasons
    assert "tests_passed" in summary.reward.reasons


def test_learning_service_scores_failure_and_auth_issues() -> None:
    service = LearningService()
    summary = service.summarize_episode(
        Task(content="deploy the app"),
        TaskResult(
            task=Task(content="deploy the app"),
            output="Host key verification failed",
            success=False,
            elapsed_ms=150.0,
            status="failed",
            tool_calls=11,
        ),
    )

    assert summary.event_kind == "failure"
    assert summary.reward.score < 0
    assert "task_failed" in summary.reward.reasons
    assert "environment_or_auth_failure" in summary.reward.reasons
    assert "high_tool_call_count" in summary.reward.reasons


def test_learning_service_scores_waiting_state_separately() -> None:
    service = LearningService()
    summary = service.summarize_episode(
        Task(content="ask user for the environment"),
        TaskResult(
            task=Task(content="ask user for the environment"),
            output="Waiting for user input",
            success=None,
            elapsed_ms=50.0,
            status="waiting_for_user",
            tool_calls=1,
        ),
    )

    assert summary.event_kind == "waiting"
    assert "waiting_for_user" in summary.reward.reasons
    assert "clarification_requested" in summary.reward.reasons
    assert summary.reward.score >= 0


def test_learning_service_scores_timeouts_and_user_corrections() -> None:
    service = LearningService()
    summary = service.summarize_episode(
        Task(content="recover after correction"),
        TaskResult(
            task=Task(content="recover after correction"),
            output="Timed out and user corrected the answer",
            success=False,
            elapsed_ms=75.0,
            status="failed",
            tool_calls=4,
        ),
    )

    assert "timeout_detected" in summary.reward.reasons
    assert "user_corrected" in summary.reward.reasons


def test_learning_service_rewards_approval_and_verification_success() -> None:
    service = LearningService()
    result = TaskResult(
        task=Task(content="deploy the release"),
        output="Approval confirmed. Verified health checks after deploy.",
        success=True,
        elapsed_ms=220.0,
        status="succeeded",
        tool_calls=5,
    )

    summary = service.summarize_episode(Task(content="deploy the release"), result)

    assert "approval_handled" in summary.reward.reasons
    assert "verification_completed" in summary.reward.reasons
    assert service.should_promote_success(result, summary) is True


def test_learning_service_avoids_promoting_low_signal_success() -> None:
    service = LearningService()
    result = TaskResult(
        task=Task(content="small success"),
        output="done",
        success=True,
        elapsed_ms=50.0,
        status="succeeded",
        tool_calls=2,
    )

    summary = service.summarize_episode(Task(content="small success"), result)

    assert service.should_promote_success(result, summary) is False


def test_learning_service_promotes_high_signal_failures() -> None:
    service = LearningService()
    result = TaskResult(
        task=Task(content="deploy the app"),
        output="Host key verification failed",
        success=False,
        elapsed_ms=150.0,
        status="failed",
        tool_calls=11,
    )

    summary = service.summarize_episode(Task(content="deploy the app"), result)

    assert service.should_promote_failure(result, summary) is True


def test_learning_service_waiting_without_question_is_slightly_negative() -> None:
    service = LearningService()
    summary = service.summarize_episode(
        Task(content="wait for clarification"),
        TaskResult(
            task=Task(content="wait for clarification"),
            output="No response yet",
            success=None,
            elapsed_ms=10.0,
            status="waiting_for_user",
            tool_calls=0,
        ),
    )

    assert summary.reward.score < 0
    assert "clarification_requested" not in summary.reward.reasons


def test_learning_service_marks_failed_memory_reuse() -> None:
    service = LearningService()
    summary = service.summarize_episode(
        Task(content="retry deploy"),
        TaskResult(
            task=Task(content="retry deploy"),
            output="Permission denied",
            success=False,
            elapsed_ms=25.0,
            status="failed",
            tool_calls=2,
            retrieved_memory_ids=[1],
        ),
    )

    assert "memory_reuse_did_not_help" in summary.reward.reasons


def test_learning_service_does_not_score_memory_reuse_for_waiting_state() -> None:
    service = LearningService()
    summary = service.summarize_episode(
        Task(content="wait on deploy approval"),
        TaskResult(
            task=Task(content="wait on deploy approval"),
            output="Waiting for user input",
            success=None,
            elapsed_ms=25.0,
            status="waiting_for_user",
            tool_calls=1,
            retrieved_memory_ids=[1],
        ),
    )

    assert "memory_reuse_helped" not in summary.reward.reasons
    assert "memory_reuse_did_not_help" not in summary.reward.reasons


def test_learning_service_uses_score_fallback_for_unknown_status() -> None:
    service = LearningService()
    summary = service.summarize_episode(
        Task(content="background bookkeeping"),
        TaskResult(
            task=Task(content="background bookkeeping"),
            output="",
            success=None,
            elapsed_ms=5.0,
            status="unknown",
            tool_calls=0,
        ),
    )

    assert summary.event_kind == "success"


def test_learning_service_promotes_long_high_effort_success_without_special_keywords() -> None:
    service = LearningService()
    result = TaskResult(
        task=Task(content="document migration"),
        output="Wrote a detailed migration checklist and validation notes for the next operator.",
        success=True,
        elapsed_ms=80.0,
        status="succeeded",
        tool_calls=6,
    )

    summary = service.summarize_episode(Task(content="document migration"), result)

    assert service.should_promote_success(result, summary) is True


def test_learning_service_does_not_promote_success_when_result_not_confirmed() -> None:
    service = LearningService()
    result = TaskResult(
        task=Task(content="tentative success"),
        output="Looks okay",
        success=None,
        elapsed_ms=15.0,
        status="succeeded",
        tool_calls=1,
    )
    episode = EpisodeSummary(
        summary="tentative",
        reward=RewardSignal(score=1.0, reasons=["task_succeeded"]),
        event_kind="success",
    )

    assert service.should_promote_success(result, episode) is False


def test_learning_service_does_not_promote_waiting_failure_summary() -> None:
    service = LearningService()
    result = TaskResult(
        task=Task(content="wait"),
        output="Need more input",
        success=None,
        elapsed_ms=5.0,
        status="waiting_for_user",
        tool_calls=0,
    )
    episode = EpisodeSummary(
        summary="waiting",
        reward=RewardSignal(score=-0.2, reasons=["waiting_for_user"]),
        event_kind="failure",
    )

    assert service.should_promote_failure(result, episode) is False

