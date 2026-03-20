from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.loop import Task, TaskResult


@dataclass(slots=True)
class RewardSignal:
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EpisodeSummary:
    summary: str
    reward: RewardSignal
    event_kind: str
    details: dict[str, Any] = field(default_factory=dict)


class LearningService:
    """Derive structured reliability signals from task execution."""

    def summarize_episode(self, task: "Task", result: "TaskResult") -> EpisodeSummary:
        reasons: list[str] = []
        score = 0.0
        output = (result.output or "").lower()
        question = (result.question or "").strip()
        content = (task.content or "").lower()
        combined = " ".join(part for part in (content, output, question.lower()) if part)

        if result.status == "succeeded" or result.success is True:
            score += 1.0
            reasons.append("task_succeeded")
        if result.status == "failed" or result.success is False:
            score -= 1.0
            reasons.append("task_failed")
        if result.status == "waiting_for_user":
            reasons.append("waiting_for_user")
            if question or "waiting for user input" in output:
                score += 0.15
                reasons.append("clarification_requested")
            else:
                score -= 0.1
        if result.tool_calls:
            if result.tool_calls <= 3 and (result.status == "succeeded" or result.success is True):
                score += 0.15
                reasons.append("efficient_tool_use")
            elif result.tool_calls >= 10 and (result.status == "failed" or result.success is False):
                score -= 0.1
                reasons.append("high_tool_call_count")

        if "test passed" in output or "tests passed" in output:
            score += 0.35
            reasons.append("tests_passed")
        if "approval" in combined and any(
            phrase in combined
            for phrase in ("confirmed", "granted", "approved", "check approval", "needs approval", "before deploy")
        ):
            score += 0.25
            reasons.append("approval_handled")
        if any(
            phrase in combined
            for phrase in ("verified health", "verification summary", "verified", "health check")
        ):
            score += 0.2
            reasons.append("verification_completed")
        if "timed out" in output or "timeout" in output:
            score -= 0.4
            reasons.append("timeout_detected")
        if "permission denied" in output or "host key verification failed" in output:
            score -= 0.45
            reasons.append("environment_or_auth_failure")
        if "corrected by user" in output or "user corrected" in output:
            score -= 0.4
            reasons.append("user_corrected")
        if result.retrieved_memory_ids or result.retrieved_procedure_ids:
            if result.status == "succeeded" or result.success is True:
                score += 0.2
                reasons.append("memory_reuse_helped")
            elif result.status == "failed" or result.success is False:
                score -= 0.1
                reasons.append("memory_reuse_did_not_help")

        if result.status == "waiting_for_user":
            event_kind = "waiting"
        elif result.status == "failed" or result.success is False:
            event_kind = "failure"
        elif result.status == "succeeded" or result.success is True:
            event_kind = "success"
        else:
            event_kind = "success" if score >= 0 else "failure"

        summary = (
            f"Task `{task.content[:120]}` finished with status `{result.status}` "
            f"after {result.tool_calls} tool calls."
        )
        details = {
            "task_status": result.status,
            "tool_calls": result.tool_calls,
            "elapsed_ms": result.elapsed_ms,
            "question": result.question,
            "output_excerpt": (result.output or "")[:500],
        }
        return EpisodeSummary(
            summary=summary,
            reward=RewardSignal(score=round(score, 3), reasons=reasons),
            event_kind=event_kind,
            details=details,
        )

    def should_promote_success(self, result: "TaskResult", episode: EpisodeSummary) -> bool:
        if episode.event_kind != "success":
            return False
        if result.status == "waiting_for_user" or result.success is not True:
            return False
        high_signal_reasons = {
            "tests_passed",
            "approval_handled",
            "verification_completed",
            "memory_reuse_helped",
        }
        if high_signal_reasons.intersection(episode.reward.reasons):
            return True
        if result.tool_calls >= 6 and len((result.output or "").strip()) >= 40:
            return True
        return episode.reward.score >= 1.45

    def should_promote_failure(self, result: "TaskResult", episode: EpisodeSummary) -> bool:
        if episode.event_kind != "failure":
            return False
        if result.status == "waiting_for_user":
            return False
        if episode.reward.score <= -0.5:
            return True
        failure_reasons = {
            "task_failed",
            "timeout_detected",
            "environment_or_auth_failure",
            "user_corrected",
            "memory_reuse_did_not_help",
        }
        return bool(failure_reasons.intersection(episode.reward.reasons))

