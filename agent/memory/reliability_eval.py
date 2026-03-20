from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ReplayScenario:
    name: str
    query: str
    expected_substrings: tuple[str, ...]


@dataclass(slots=True)
class RewardReplayScenario:
    name: str
    task_content: str
    status: str
    success: bool | None
    output: str
    tool_calls: int = 0
    question: str | None = None
    retrieved_memory_ids: tuple[int, ...] = ()
    retrieved_procedure_ids: tuple[int, ...] = ()
    expected_reasons: tuple[str, ...] = ()
    min_score: float | None = None
    max_score: float | None = None
    expect_success_promotion: bool = False
    expect_failure_promotion: bool = False


async def run_reliability_eval(store: Any, scenarios: list[ReplayScenario]) -> dict[str, Any]:
    """Replay simple retrieval scenarios and score memory reuse quality."""

    total = len(scenarios)
    if total == 0:
        return {"scenarios": 0, "passed": 0, "pass_rate": 1.0, "results": []}

    results: list[dict[str, Any]] = []
    passed = 0
    for scenario in scenarios:
        payload = await store.search_learning_context(scenario.query, limit=3)
        text = str(payload.get("text", ""))
        matched = all(expected in text for expected in scenario.expected_substrings)
        if matched:
            passed += 1
        results.append(
            {
                "name": scenario.name,
                "matched": matched,
                "query": scenario.query,
                "memory_ids": list(payload.get("memory_ids", [])),
                "procedure_ids": list(payload.get("procedure_ids", [])),
            }
        )
    return {
        "scenarios": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3),
        "results": results,
    }


def run_reward_replay_eval(service: Any, task_factory: Any, result_factory: Any, scenarios: list[RewardReplayScenario]) -> dict[str, Any]:
    total = len(scenarios)
    if total == 0:
        return {"scenarios": 0, "passed": 0, "pass_rate": 1.0, "results": []}

    passed = 0
    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        task = task_factory(content=scenario.task_content)
        result = result_factory(
            task=task,
            output=scenario.output,
            success=scenario.success,
            elapsed_ms=100.0,
            status=scenario.status,
            tool_calls=scenario.tool_calls,
            question=scenario.question,
            retrieved_memory_ids=list(scenario.retrieved_memory_ids),
            retrieved_procedure_ids=list(scenario.retrieved_procedure_ids),
        )
        episode = service.summarize_episode(task, result)
        reasons_ok = all(reason in episode.reward.reasons for reason in scenario.expected_reasons)
        min_ok = scenario.min_score is None or episode.reward.score >= scenario.min_score
        max_ok = scenario.max_score is None or episode.reward.score <= scenario.max_score
        success_promotion_ok = (
            service.should_promote_success(result, episode) == scenario.expect_success_promotion
        )
        failure_promotion_ok = (
            service.should_promote_failure(result, episode) == scenario.expect_failure_promotion
        )
        matched = reasons_ok and min_ok and max_ok and success_promotion_ok and failure_promotion_ok
        if matched:
            passed += 1
        results.append(
            {
                "name": scenario.name,
                "matched": matched,
                "score": episode.reward.score,
                "reasons": list(episode.reward.reasons),
                "event_kind": episode.event_kind,
                "promote_success": service.should_promote_success(result, episode),
                "promote_failure": service.should_promote_failure(result, episode),
            }
        )
    return {
        "scenarios": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3),
        "results": results,
    }

