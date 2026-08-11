import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import numpy as np

from scripts.psi0_simple_real_bridge import ShutdownCoordinator


def wait_for(path, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def assert_port_rebinds(port):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def run_shutdown_scenario(tmp_path, scenario):
    report_path = tmp_path / f"{scenario}-report.json"
    ready_path = tmp_path / f"{scenario}-ready"
    command = [
        sys.executable,
        "scripts/tests/bridge_subprocess_fixture.py",
        "--scenario",
        scenario,
        "--report",
        str(report_path),
        "--ready",
        str(ready_path),
    ]
    process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1])
    wait_for(ready_path)
    started = time.monotonic()
    os.kill(process.pid, signal.SIGINT)
    process.wait(timeout=7.0)
    elapsed = time.monotonic() - started
    assert process.returncode == 0
    report = json.loads(report_path.read_text())
    for port in report["owned_ports"]:
        assert_port_rebinds(port)
    return elapsed, report


def test_ctrl_c_before_first_valid_state_publishes_nothing(tmp_path):
    elapsed_s, report = run_shutdown_scenario(tmp_path, "no-state")
    goals = report["goals"]
    inference_requests = report["inference_requests"]
    assert goals == []
    assert inference_requests == []
    assert elapsed_s <= 0.5
    assert report["publisher_closed"] is True
    assert report["camera_closed"] is True
    assert report["terminal_restored"] is True
    assert report["live_non_daemon_bridge_threads"] == []


def test_ctrl_c_with_five_second_request_publishes_exact_final_hold(tmp_path):
    elapsed_s, report = run_shutdown_scenario(tmp_path, "inflight-five-second")
    assert report["request_accepted"] is True
    assert elapsed_s <= 6.5
    goals = report["goals"]
    assert len(goals) == 25
    canonical = json.dumps(goals[0]["goal"], sort_keys=True, separators=(",", ":"))
    assert all(
        json.dumps(entry["goal"], sort_keys=True, separators=(",", ":")) == canonical
        for entry in goals
    )
    assert all(entry["goal"]["navigate_cmd"] == [0.0, 0.0, 0.0, 0.0] for entry in goals)
    target = np.asarray(goals[0]["goal"]["target_upper_body_pose"])
    np.testing.assert_array_less(np.asarray(report["goal_lower_bounds"]) - 1e-7, target)
    np.testing.assert_array_less(target, np.asarray(report["goal_upper_bounds"]) + 1e-7)
    assert (
        float(np.float32(0.20))
        <= goals[0]["goal"]["base_height_command"][0]
        <= float(np.float32(0.74))
    )
    deltas = np.diff([entry["scheduled_at"] for entry in goals])
    np.testing.assert_allclose(deltas, 0.02, rtol=0, atol=0.002)
    assert report["publish_attempts"] == 25
    assert report["publisher_closed_after_publish_count"] == 25
    assert report["publisher_closed"] is True
    assert report["camera_closed"] is True
    assert report["terminal_restored"] is True
    assert report["live_non_daemon_bridge_threads"] == []


class AdvancingClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)


class BudgetedCloser:
    def __init__(self, clock, consume=0.0, busy=False):
        self.clock = clock
        self.consume = consume
        self.busy = busy
        self.calls = []

    def close(self, timeout_s):
        self.calls.append((self.clock(), timeout_s))
        self.clock.sleep(min(self.consume, timeout_s))


class BudgetedSink(BudgetedCloser):
    def __init__(self, clock):
        super().__init__(clock)
        self.published = 0

    def publish(self, goal):
        self.published += 1
        return True


class ShutdownBridge:
    def __init__(self, hold):
        self.hold = hold

    def stop(self):
        return None

    def build_bounded_shutdown_hold(self):
        return self.hold


class FailingShutdownBridge(ShutdownBridge):
    def stop(self):
        raise RuntimeError("stop failed")

    def build_bounded_shutdown_hold(self):
        raise RuntimeError("hold failed")


def test_every_cleanup_timeout_shares_one_absolute_long_path_deadline():
    clock = AdvancingClock()
    sink = BudgetedSink(clock)
    camera = BudgetedCloser(clock, consume=0.5)
    worker = BudgetedCloser(clock, consume=5.5, busy=True)
    remaining_resources = [BudgetedCloser(clock) for _ in range(5)]
    coordinator = ShutdownCoordinator(
        bridge=ShutdownBridge(hold=object()),
        command_sink=sink,
        camera=camera,
        worker=worker,
        state_source=remaining_resources[0],
        keyboard=remaining_resources[1],
        ownership_guard=remaining_resources[2],
        ros_runtime=remaining_resources[3],
        metrics=remaining_resources[4],
        clock=clock,
        sleep=clock.sleep,
    )
    report = coordinator.close()
    assert sink.published == 25
    assert report.deadline_at == 6.5
    assert report.finished_at <= report.deadline_at
    for resource in [sink, camera, worker, *remaining_resources]:
        for called_at, timeout_s in resource.calls:
            assert 0.0 <= timeout_s <= max(0.0, report.deadline_at - called_at)


def test_no_state_cleanup_uses_one_half_second_deadline():
    clock = AdvancingClock()
    sink = BudgetedSink(clock)
    resources = [BudgetedCloser(clock) for _ in range(7)]
    coordinator = ShutdownCoordinator(
        bridge=ShutdownBridge(hold=None),
        command_sink=sink,
        camera=resources[0],
        worker=resources[1],
        state_source=resources[2],
        keyboard=resources[3],
        ownership_guard=resources[4],
        ros_runtime=resources[5],
        metrics=resources[6],
        clock=clock,
        sleep=clock.sleep,
    )
    report = coordinator.close()
    assert sink.published == 0
    assert report.deadline_at == 0.5
    assert report.finished_at <= 0.5


def test_exhausted_deadline_still_attempts_terminal_and_all_cleanup():
    clock = AdvancingClock()
    sink = BudgetedSink(clock)
    camera = BudgetedCloser(clock, consume=0.5)
    worker = BudgetedCloser(clock)
    state = BudgetedCloser(clock)
    keyboard = BudgetedCloser(clock)
    ownership = BudgetedCloser(clock)
    ros = BudgetedCloser(clock)
    metrics = BudgetedCloser(clock)
    coordinator = ShutdownCoordinator(
        bridge=ShutdownBridge(hold=None),
        command_sink=sink,
        camera=camera,
        worker=worker,
        state_source=state,
        keyboard=keyboard,
        ownership_guard=ownership,
        ros_runtime=ros,
        metrics=metrics,
        clock=clock,
        sleep=clock.sleep,
    )
    report = coordinator.close()
    assert camera.calls == [(0.0, 0.5)]
    for resource in (worker, state, keyboard, ownership, ros, metrics):
        assert resource.calls == [(0.5, 0.0)]
    assert any("deadline exhausted" in error for error in report.cleanup_errors)


def test_bridge_stop_failure_cannot_skip_terminal_or_metrics_cleanup():
    clock = AdvancingClock()
    resources = [BudgetedCloser(clock) for _ in range(7)]
    coordinator = ShutdownCoordinator(
        bridge=FailingShutdownBridge(hold=None),
        command_sink=BudgetedSink(clock),
        camera=resources[0],
        worker=resources[1],
        state_source=resources[2],
        keyboard=resources[3],
        ownership_guard=resources[4],
        ros_runtime=resources[5],
        metrics=resources[6],
        clock=clock,
        sleep=clock.sleep,
    )
    report = coordinator.close()
    assert all(len(resource.calls) == 1 for resource in resources)
    assert any(error == "bridge_stop: stop failed" for error in report.cleanup_errors)
    assert any(error == "bounded_hold: hold failed" for error in report.cleanup_errors)
