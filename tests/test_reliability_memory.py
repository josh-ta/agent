from __future__ import annotations

import pytest

from agent.loop import Task, TaskResult
from agent.memory.learning_service import LearningService
from agent.memory.reliability_eval import (
    ReplayScenario,
    RewardReplayScenario,
    run_reliability_eval,
    run_reward_replay_eval,
)
from agent.memory.sqlite_store import SQLiteStore
from agent.secret_store import SecretStore, mask_secret


@pytest.mark.asyncio
@pytest.mark.integration
async def test_learning_context_persists_across_store_restarts(isolated_paths) -> None:
    db_path = isolated_paths["workspace"] / "reliability.db"
    store = SQLiteStore(db_path)
    await store.init()
    await store.save_memory_fact("The staging environment needs approval before deploy.")
    await store.save_lesson("Check deployment approval before running release commands.", kind="mistake")
    await store.save_procedure(
        trigger_text="deploy or release task",
        checklist="Confirm approval, then run the deployment command, then verify health.",
    )
    await store.close()

    reopened = SQLiteStore(db_path)
    await reopened.init()
    try:
        payload = await reopened.search_learning_context("deploy the release", limit=3)
    finally:
        await reopened.close()

    assert "Relevant facts" in payload["text"]
    assert "Known pitfalls" in payload["text"]
    assert "Preferred procedures" in payload["text"]
    assert payload["memory_ids"]
    assert payload["procedure_ids"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_feedback_and_pinning_influence_reliability_scores(sqlite_store) -> None:
    memory_id = await sqlite_store.save_memory_item(
        kind="fact",
        content="Use the blue deployment workflow for zero-downtime releases.",
    )
    procedure_id = await sqlite_store.save_procedure(
        trigger_text="zero downtime deploy",
        checklist="Use the blue deployment workflow and verify health checks.",
    )

    await sqlite_store.record_feedback(
        task_id="task-1",
        feedback_kind="approved",
        score=2.0,
        memory_item_id=memory_id,
        procedure_id=procedure_id,
        details={"source": "operator"},
    )
    await sqlite_store.pin_memory_item(memory_id)
    await sqlite_store.pin_procedure(procedure_id)
    await sqlite_store._cleanup()

    assert sqlite_store._db is not None
    async with sqlite_store._db.execute(
        "SELECT pinned, success_credit FROM memory_items WHERE id=?",
        (memory_id,),
    ) as cur:
        memory_row = await cur.fetchone()
    async with sqlite_store._db.execute(
        "SELECT pinned, success_credit FROM procedures WHERE id=?",
        (procedure_id,),
    ) as cur:
        procedure_row = await cur.fetchone()

    assert memory_row is not None
    assert procedure_row is not None
    assert memory_row["pinned"] == 1
    assert procedure_row["pinned"] == 1
    assert memory_row["success_credit"] > 0
    assert procedure_row["success_credit"] > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_reliability_eval_reports_memory_reuse(sqlite_store) -> None:
    await sqlite_store.save_memory_fact("The parser requires the routing fixture.")
    await sqlite_store.save_procedure(
        trigger_text="parser or routing task",
        checklist="Load the routing fixture before running parser tests.",
    )

    report = await run_reliability_eval(
        sqlite_store,
        [
            ReplayScenario(
                name="parser-fixture",
                query="fix the parser tests",
                expected_substrings=("routing fixture", "Load the routing fixture"),
            )
        ],
    )

    assert report["scenarios"] == 1
    assert report["passed"] == 1
    assert report["pass_rate"] == 1.0


@pytest.mark.asyncio
async def test_reliability_eval_handles_empty_scenarios(sqlite_store) -> None:
    report = await run_reliability_eval(sqlite_store, [])

    assert report == {"scenarios": 0, "passed": 0, "pass_rate": 1.0, "results": []}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_reliability_eval_reports_failed_match(sqlite_store) -> None:
    await sqlite_store.save_memory_fact("Use the routing fixture.")

    report = await run_reliability_eval(
        sqlite_store,
        [ReplayScenario(name="missing", query="parser", expected_substrings=("not present",))],
    )

    assert report["passed"] == 0
    assert report["results"][0]["matched"] is False


def test_secret_store_redacts_values_and_keeps_metadata(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.secret_store.settings.agent_secrets_master_key", "unit-test-key")
    store = SecretStore(tmp_path / "agent-secrets.json")
    store.set(
        "LOGIN_PASSWORD",
        "hunter2",
        purpose="dashboard login",
        scope="staging",
        allowed_tools=["browser_fill_secret"],
    )

    redacted = store.redact_text("Use hunter2 to log in.")
    entries = store.list_entries()

    assert mask_secret("hunter2") in redacted
    assert "hunter2" not in redacted
    assert entries[0]["purpose"] == "dashboard login"
    assert entries[0]["allowed_tools"] == ["browser_fill_secret"]


def test_reward_replay_eval_matches_transcript_like_scenarios() -> None:
    report = run_reward_replay_eval(
        LearningService(),
        Task,
        TaskResult,
        [
            RewardReplayScenario(
                name="approval-gated-deploy",
                task_content="deploy the release",
                status="succeeded",
                success=True,
                output="Approval confirmed. Verified health checks after deploy.",
                tool_calls=5,
                expected_reasons=("task_succeeded", "approval_handled", "verification_completed"),
                min_score=1.4,
                expect_success_promotion=True,
            ),
            RewardReplayScenario(
                name="clarification-wait",
                task_content="ask which environment to use",
                status="waiting_for_user",
                success=None,
                output="Waiting for user input",
                tool_calls=1,
                question="Which environment should I use?",
                expected_reasons=("waiting_for_user", "clarification_requested"),
                min_score=0.0,
                max_score=0.5,
                expect_success_promotion=False,
                expect_failure_promotion=False,
            ),
            RewardReplayScenario(
                name="host-key-failure",
                task_content="deploy the app",
                status="failed",
                success=False,
                output="Host key verification failed",
                tool_calls=11,
                expected_reasons=("task_failed", "environment_or_auth_failure"),
                max_score=-1.3,
                expect_failure_promotion=True,
            ),
            RewardReplayScenario(
                name="fixture-reuse-success",
                task_content="fix the parser tests",
                status="succeeded",
                success=True,
                output="Tests passed after loading the routing fixture.",
                tool_calls=4,
                retrieved_memory_ids=(1,),
                expected_reasons=("tests_passed", "memory_reuse_helped"),
                min_score=1.5,
                expect_success_promotion=True,
            ),
        ],
    )

    assert report["scenarios"] == 4
    assert report["passed"] == 4
    assert report["pass_rate"] == 1.0


def test_reward_replay_eval_handles_empty_scenarios() -> None:
    report = run_reward_replay_eval(LearningService(), Task, TaskResult, [])

    assert report == {"scenarios": 0, "passed": 0, "pass_rate": 1.0, "results": []}


def test_reward_replay_eval_reports_mismatched_scenario() -> None:
    report = run_reward_replay_eval(
        LearningService(),
        Task,
        TaskResult,
        [
            RewardReplayScenario(
                name="mismatch",
                task_content="deploy",
                status="succeeded",
                success=True,
                output="done",
                tool_calls=1,
                expected_reasons=("approval_handled",),
                expect_success_promotion=True,
            )
        ],
    )

    assert report["passed"] == 0
    assert report["results"][0]["matched"] is False

