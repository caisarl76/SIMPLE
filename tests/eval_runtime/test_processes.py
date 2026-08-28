# tests/eval_runtime/test_processes.py
from __future__ import annotations

import signal

import pytest

from simple.eval_runtime.contracts import OwnedDeadline, ProcessIdentity, RuntimeBlocked
from simple.eval_runtime.processes import terminate_owned_process


class FakeProcessOps:
    def __init__(self, identities: list[ProcessIdentity | None], alive: list[bool]):
        self.identities = identities
        self.alive = alive
        self.signals: list[int] = []
        self.wait_deadlines: list[float] = []

    def inspect(self, pid: int) -> ProcessIdentity | None:
        return self.identities.pop(0)

    def signal_pidfd(self, pid: int, sig: int) -> None:
        self.signals.append(sig)

    def wait_dead(self, pid: int, deadline: float) -> bool:
        self.wait_deadlines.append(deadline)
        return self.alive.pop(0)


def test_cleanup_uses_int_term_kill_and_fresh_post_kill_wait() -> None:
    expected = ProcessIdentity(42, 900, "a" * 64)
    ops = FakeProcessOps([expected, expected, expected, expected], [False, False, True])
    clock_values = iter((1000.0, 1001.0, 1002.0))
    terminate_owned_process(
        ops,
        expected,
        OwnedDeadline(1200.0),
        stage_seconds=2.0,
        clock=lambda: next(clock_values),
    )
    assert ops.signals == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    assert len(ops.wait_deadlines) == 3
    assert ops.wait_deadlines[-1] > ops.wait_deadlines[-2]


def test_identity_drift_never_signals() -> None:
    expected = ProcessIdentity(42, 900, "a" * 64)
    replacement = ProcessIdentity(42, 901, "a" * 64)
    ops = FakeProcessOps([replacement], [])
    with pytest.raises(RuntimeBlocked, match="FOREIGN_PROCESS"):
        terminate_owned_process(ops, expected, OwnedDeadline(1200.0), stage_seconds=2.0)
    assert ops.signals == []
