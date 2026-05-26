from __future__ import annotations

from agent import metrics


def test_metrics_counters_and_prometheus_text() -> None:
    metrics.inc_task_completed(success=True)
    metrics.inc_task_completed(success=False)
    metrics.inc_permission_denied()
    metrics.inc_shell_blocked()
    metrics.inc_context_warn()

    text = metrics.prometheus_text()
    assert 'outcome="success"' in text
    assert 'outcome="failure"' in text
    assert "agent_permission_denied_total" in text
    assert "agent_shell_blocked_total" in text
    assert "agent_context_token_warn_total" in text

    metrics.Metrics.inc_task_completed(success=True)
    assert "agent_tasks_completed_total" in metrics.Metrics.prometheus_text()
