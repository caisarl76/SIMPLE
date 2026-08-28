# src/simple/eval_runtime/processes.py
from __future__ import annotations

import signal
import time
from typing import Any

from .contracts import OwnedDeadline, ProcessIdentity, RuntimeBlocked


def terminate_owned_process(
    ops: Any,
    expected: ProcessIdentity,
    overall: OwnedDeadline,
    *,
    stage_seconds: float,
    clock=time.monotonic,
) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        actual = ops.inspect(expected.pid)
        if actual is None:
            return
        if actual != expected:
            raise RuntimeBlocked("FOREIGN_PROCESS", f"pid={expected.pid}")
        ops.signal_pidfd(expected.pid, sig)
        stage_deadline = min(overall.expires_at, clock() + stage_seconds)
        if ops.wait_dead(expected.pid, stage_deadline):
            return
    post_kill_deadline = min(overall.expires_at, clock() + stage_seconds)
    if not ops.wait_dead(expected.pid, post_kill_deadline):
        raise RuntimeBlocked("CLEANUP_FAILED", f"pid={expected.pid} survived SIGKILL")
