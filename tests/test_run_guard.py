from agent.run_guard import RunGuard


def test_run_guard_begin_end_mismatch_returns_false() -> None:
    g = RunGuard()
    assert g.begin_run(1) is True
    assert g.end_run(2) is False
    assert g.end_run(1) is True


def test_run_guard_rejects_second_begin_until_end() -> None:
    g = RunGuard()
    assert g.begin_run(1) is True
    assert g.begin_run(2) is False
    assert g.end_run(1) is True
    assert g.begin_run(2) is True
