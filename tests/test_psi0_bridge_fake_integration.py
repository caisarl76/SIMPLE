import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from scripts.psi0_simple_real_bridge import ComposedCameraReader, HttpInferenceWorker
from scripts.tests.fake_composed_camera_server import running_fake_camera
from scripts.tests.fake_psi0_rtc_server import running_fake_policy
from simple.baselines.client import HttpActionClient
from simple.deploy.psi0_simple_bridge import (
    BridgeState,
    PolicyContract,
    Psi0SimpleBridge,
    TimedCameraFrame,
    TimedRobotState,
)
from tests.psi0_bridge_testkit import make_joint_contract


CONTRACT_PATH = Path("scripts/tests/fixtures/psi0_policy_contract_test_v2.json")


def assert_loopback_port_rebinds(port):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))


def wait_for_json(path, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text())
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def rtc_history(seq, tick, committed, reset=False):
    history = {
        "session_id": "fake-session",
        "request_seq": seq,
        "observation_tick": tick,
        "rtc_delay_steps": 6,
        "committed_actions": committed,
    }
    if reset:
        history["reset"] = True
    return history


def query_fake(client, history):
    return client.query_rtc_action(
        {"rgb_head_stereo_left": np.zeros((8, 8, 3), np.uint8)},
        "pick up the object",
        {"states": np.zeros((1, 32), np.float32)},
        {},
        history=history,
        dataset="simple",
    )


def run_wall_clock_fake_bridge(port, duration_s):
    contract = PolicyContract.from_dict(json.loads(CONTRACT_PATH.read_text()))
    worker = HttpInferenceWorker(
        client=HttpActionClient("127.0.0.1", port, timeout=5.0),
        clock=time.monotonic,
        contract=contract,
    )
    bridge = Psi0SimpleBridge(
        contract, make_joint_contract(), worker, time.monotonic, start_tick=0
    )

    def refresh_inputs():
        now = time.monotonic()
        bridge.update_inputs(
            TimedRobotState(np.zeros(43, np.float32), now),
            TimedCameraFrame(np.zeros((8, 8, 3), np.uint8), now, None),
        )

    refresh_inputs()
    bridge.tick()
    bridge.activate()
    deadline = time.monotonic() + duration_s
    policy_actions_after_fault = 0
    while time.monotonic() < deadline:
        tick_started = time.monotonic()
        refresh_inputs()
        result = bridge.tick()
        if bridge.state is BridgeState.FAULT and result.source_kind == "policy":
            policy_actions_after_fault += 1
        time.sleep(max(0.0, 0.02 - (time.monotonic() - tick_started)))
    worker.close(timeout_s=1.0)
    return SimpleNamespace(
        request_started_at=worker.metrics.first_request_started_at,
        fault_at=bridge.metrics.first_fault_at,
        old_generation_results_discarded=(
            bridge.metrics.discarded_old_generation_results
        ),
        policy_actions_after_fault=policy_actions_after_fault,
        requests_submitted=worker.metrics.requests_submitted,
    )


def test_fake_policy_contract_r0_successor_and_normal_latency_are_exact():
    expected_contract = json.loads(CONTRACT_PATH.read_text())
    with running_fake_policy(CONTRACT_PATH, normal_latency_s=0.05) as fake:
        client = HttpActionClient("127.0.0.1", fake.port, timeout=1.0)
        assert client.get_contract() == expected_contract
        hold = np.zeros((6, 36), np.float32)
        started = time.monotonic()
        r0 = query_fake(client, rtc_history(0, 100, hold, reset=True))
        elapsed = time.monotonic() - started
        assert elapsed >= 0.045
        assert fake.records[0].applied_latency_s == pytest.approx(0.05)
        assert r0.action.shape == (24, 36)
        assert r0.metadata == {
            "session_id": "fake-session",
            "request_seq": 0,
            "observation_tick": 100,
            "prediction_horizon": 30,
            "execution_horizon": 24,
            "rtc_delay_steps": 6,
            "first_action_tick": 106,
        }
        committed = np.arange(6 * 36, dtype=np.float32).reshape(6, 36) / 1000
        r1 = query_fake(client, rtc_history(1, 124, committed))
        assert r1.metadata["first_action_tick"] == 130
        np.testing.assert_array_equal(fake.records[1].committed_actions, committed)
        assert fake.records[1].reset is False
        assert fake.max_concurrent_requests == 1


def test_point_30_second_one_shot_latency_faults_at_deadline_and_is_discarded():
    with running_fake_policy(CONTRACT_PATH, normal_latency_s=0.05) as fake:
        fake.delay_next_request(0.30)
        report = run_wall_clock_fake_bridge(fake.port, duration_s=0.45)
        assert report.request_started_at is not None
        assert report.fault_at <= report.request_started_at + 0.14
        assert report.old_generation_results_discarded == 1
        assert report.policy_actions_after_fault == 0
        assert report.requests_submitted == 1
        assert fake.max_concurrent_requests == 1


def test_cli_control_endpoint_delays_exact_next_request_across_processes(tmp_path):
    ready = tmp_path / "policy-ready.json"
    token = "test-control-token"
    command = [
        sys.executable,
        "scripts/tests/fake_psi0_rtc_server.py",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--normal-latency-s",
        "0.05",
        "--contract",
        str(CONTRACT_PATH),
        "--control-token",
        token,
        "--ready-json",
        str(ready),
    ]
    process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1])
    port = None
    try:
        port = wait_for_json(ready)["port"]
        base = f"http://127.0.0.1:{port}"
        armed = requests.post(
            f"{base}/test-control/arm-next-delay",
            json={"token": token, "latency_s": 0.30},
            timeout=0.5,
        )
        assert armed.status_code == 202
        assert armed.json() == {"armed_request_seq": 0, "latency_s": 0.30}
        client = HttpActionClient("127.0.0.1", port, timeout=1.0)
        started = time.monotonic()
        query_fake(
            client,
            rtc_history(0, 100, np.zeros((6, 36), np.float32), reset=True),
        )
        assert time.monotonic() - started >= 0.295
        status = requests.get(
            f"{base}/test-control/status",
            headers={"X-Test-Control-Token": token},
            timeout=0.5,
        ).json()
        assert status["records"][0]["request_seq"] == 0
        assert status["records"][0]["applied_latency_s"] == pytest.approx(0.30)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=1.0)
    assert process.returncode == 0
    assert port is not None
    assert_loopback_port_rebinds(port)


def test_fake_camera_round_trip_and_bounded_close():
    with running_fake_camera(key="rgb_head_stereo_left") as fake:
        reader = ComposedCameraReader(
            "127.0.0.1", fake.port, "rgb_head_stereo_left", "rgb"
        )
        frame = reader.wait_for_frame(timeout_s=1.0)
        left = frame.image[:, :20].mean(axis=(0, 1))
        right = frame.image[:, 44:].mean(axis=(0, 1))
        assert left[0] > 200 and left[2] < 30
        assert right[2] > 200 and right[0] < 30
        started = time.monotonic()
        reader.close(timeout_s=0.5)
        assert time.monotonic() - started <= 0.5
    assert_loopback_port_rebinds(fake.port)
