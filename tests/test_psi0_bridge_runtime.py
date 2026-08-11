import json
import os
import pty
import termios
import threading
from types import SimpleNamespace

import msgpack
import msgpack_numpy as mnp
import numpy as np
import pytest

from scripts.psi0_simple_real_bridge import (
    ConnectionEvidenceRecorder,
    DisabledInferenceWorker,
    FiftyHzLoop,
    HttpInferenceWorker,
    JsonlMetrics,
    LocalKeyboard,
    ObservationOnlyShadowBridge,
    PreflightError,
    RosGoalPublisher,
    count_real_interface_connections,
    handle_keyboard_events,
    unpack_dict_message,
)
from simple.baselines.client import RtcActionResponse
from simple.deploy.psi0_simple_bridge import (
    ActivationRefused,
    BridgeMode,
    BridgeState,
    Goal,
    PolicyContract,
    Psi0SimpleBridge,
    RtcRequest,
    TimedCameraFrame,
    TimedRobotState,
)
from tests.psi0_bridge_testkit import (
    ManualClock,
    fresh_inputs,
    make_joint_contract,
    make_policy_contract,
    policy_payload,
)


def valid_runtime_request(generation=1):
    return RtcRequest(
        generation=generation,
        session_id="runtime-session",
        request_seq=0,
        observation_tick=100,
        history_tick=99,
        observation=np.zeros((1, 32), np.float32),
        image=np.zeros((8, 8, 3), np.uint8),
        committed_actions=np.zeros((6, 36), np.float32),
        reset=True,
        deadline_at=106 / 50.0,
    )


def valid_http_rtc_response():
    return RtcActionResponse(
        action=np.zeros((24, 36), np.float32),
        metadata={
            "session_id": "runtime-session",
            "request_seq": 0,
            "observation_tick": 100,
            "prediction_horizon": 30,
            "execution_horizon": 24,
            "rtc_delay_steps": 6,
            "first_action_tick": 106,
        },
    )


def test_ros_state_decoder_accepts_wbc_one_byte_bytes_elements():
    expected = np.arange(43, dtype=np.float32)
    packed = msgpack.packb({"q": expected}, default=mnp.encode)
    message = SimpleNamespace(
        data=tuple(bytes([value]) for value in packed),
    )

    payload = unpack_dict_message(message)

    np.testing.assert_array_equal(payload["q"], expected)


def test_ros_state_decoder_preserves_standard_integer_elements():
    expected = np.arange(43, dtype=np.float32)
    packed = msgpack.packb({"q": expected}, default=mnp.encode)
    message = SimpleNamespace(data=list(packed))

    payload = unpack_dict_message(message)

    np.testing.assert_array_equal(payload["q"], expected)


def test_ros_goal_publisher_uses_wbc_one_byte_bytes_elements():
    class CapturingNode:
        def __init__(self):
            self.message = None

        def create_publisher(self, _message_type, _topic, _depth):
            return SimpleNamespace(
                publish=lambda message: setattr(self, "message", message),
            )

    node = CapturingNode()
    publisher = RosGoalPublisher(SimpleNamespace(node=node), "test-goal")
    goal = Goal(
        target_upper_body_pose=np.zeros(31, np.float32),
        base_height_command=np.asarray([0.74], np.float32),
        navigate_cmd=np.zeros(4, np.float32),
        timestamp=1.0,
        target_time=1.02,
    )

    assert publisher.publish(goal) is True

    assert node.message is not None
    assert all(
        type(element) is bytes and len(element) == 1 for element in node.message.data
    )
    packed = bytes(value for element in node.message.data for value in element)
    payload = msgpack.unpackb(packed, object_hook=mnp.decode, raw=False)
    np.testing.assert_array_equal(payload["target_upper_body_pose"], np.zeros(31))
    np.testing.assert_array_equal(payload["navigate_cmd"], np.zeros(4))


def test_local_keyboard_accepts_only_p_and_restores_terminal():
    master, slave = pty.openpty()
    original = termios.tcgetattr(slave)
    try:
        with LocalKeyboard(slave) as keyboard:
            os.write(master, b"xP]ppq")
            assert keyboard.poll(timeout_s=0.1) == ("p", "p")
        assert termios.tcgetattr(slave) == original
    finally:
        os.close(master)
        os.close(slave)


class BlockingHttpClient:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def query_rtc_action(self, *args, **kwargs):
        self.calls += 1
        self.entered.set()
        assert self.release.wait(2.0)
        return valid_http_rtc_response()


def test_one_blocked_worker_never_blocks_tick_thread_or_allows_replacement():
    client = BlockingHttpClient()
    worker = HttpInferenceWorker(
        client=client,
        clock=ManualClock(100),
        contract=PolicyContract.from_dict(
            policy_payload(
                prediction_horizon=30,
                execution_horizon=24,
                rtc_delay_steps=6,
                rtc_training_max_delay=7,
            )
        ),
    )
    thread_identity = worker.thread.ident
    worker.submit(valid_runtime_request())
    assert client.entered.wait(0.2)
    tick_times = []
    loop = FiftyHzLoop(clock=ManualClock(100), sleep=lambda _: None)
    loop.run_n(50, lambda scheduled: tick_times.append(scheduled))
    assert len(tick_times) == 50
    np.testing.assert_allclose(np.diff(tick_times), 0.02, rtol=0, atol=1e-12)
    assert worker.busy is True
    with pytest.raises(RuntimeError, match="busy"):
        worker.submit(valid_runtime_request(generation=2))
    assert worker.thread.ident == thread_identity
    assert client.calls == 1
    client.release.set()
    worker.close(timeout_s=0.5)
    assert worker.busy is False
    assert worker.thread.ident == thread_identity


def test_queued_result_keeps_worker_busy_until_main_thread_drains_it():
    class ImmediateHttpClient:
        def query_rtc_action(self, *args, **kwargs):
            return valid_http_rtc_response()

    result_recorded = threading.Event()

    def record_event(event, **_fields):
        if event == "result":
            result_recorded.set()

    worker = HttpInferenceWorker(
        client=ImmediateHttpClient(),
        clock=ManualClock(100),
        contract=PolicyContract.from_dict(
            policy_payload(
                prediction_horizon=30,
                execution_horizon=24,
                rtc_delay_steps=6,
                rtc_training_max_delay=7,
            )
        ),
        event_sink=record_event,
    )
    worker.submit(valid_runtime_request())
    assert result_recorded.wait(0.2)
    assert worker.busy is True
    assert worker.poll() is not None
    assert worker.busy is False
    worker.close(timeout_s=0.5)


def test_every_shadow_metric_and_preview_is_uncertified(tmp_path):
    path = tmp_path / "shadow.jsonl"
    bridge = SimpleNamespace(state=BridgeState.PAUSED, generation=3)
    metrics = JsonlMetrics(
        path,
        mode=BridgeMode.SHADOW,
        policy_certified=False,
        clock=lambda: 1.25,
    )
    metrics.write_event("request", request_seq=7)
    metrics.write_event("result", request_seq=7)
    metrics.write_event("preview", navigate_cmd=[0.0] * 4)
    metrics.write("tick", bridge, published=False, previewed=True)
    with pytest.raises(ValueError, match="owned by JsonlMetrics"):
        metrics.write_event("override", policy_certified=True)
    metrics.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "request",
        "result",
        "preview",
        "tick",
    ]
    assert all(record["policy_certified"] is False for record in records)


def test_full_shadow_tick_metric_reports_invalid_input_instead_of_null(tmp_path):
    clock = ManualClock(100)
    inference = SimpleNamespace(busy=False)
    bridge = Psi0SimpleBridge(
        make_policy_contract(),
        make_joint_contract(),
        inference,
        clock,
        start_tick=100,
    )
    bridge.update_inputs(*fresh_inputs(clock))
    bad = TimedRobotState(np.r_[np.nan, np.zeros(42)], clock())
    bridge.update_inputs(bad, fresh_inputs(clock)[1])
    input_valid, input_error = bridge.input_status()

    path = tmp_path / "full-shadow.jsonl"
    metrics = JsonlMetrics(
        path,
        mode=BridgeMode.SHADOW,
        policy_certified=False,
        clock=lambda: 2.0,
    )
    metrics.write(
        "tick",
        bridge,
        published=False,
        previewed=False,
        input_valid=input_valid,
        input_error=input_error,
        camera_diagnostic=bridge.camera_diagnostic,
    )
    metrics.close()
    record = json.loads(path.read_text())
    assert record["input_valid"] is False
    assert record["input_error"] == "measured state finite"


def test_rejected_activation_is_reported_and_next_local_p_is_accepted():
    class ToggleBridge:
        state = BridgeState.PAUSED

        def __init__(self):
            self.calls = 0

        def handle_toggle(self):
            self.calls += 1
            if self.calls == 1:
                raise ActivationRefused("physical worker busy")
            self.state = BridgeState.ACTIVE

    class Keyboard:
        def poll(self, _timeout_s):
            return ("p",)

    class Metrics:
        def __init__(self):
            self.events = []

        def write_event(self, event, **fields):
            self.events.append((event, fields))

    bridge = ToggleBridge()
    metrics = Metrics()
    handle_keyboard_events(bridge, Keyboard(), metrics)
    assert bridge.state is BridgeState.PAUSED
    assert metrics.events == [
        (
            "activation_refused",
            {"state": "paused", "reason": "physical worker busy"},
        )
    ]
    handle_keyboard_events(bridge, Keyboard(), metrics)
    assert bridge.state is BridgeState.ACTIVE
    assert bridge.calls == 2


def test_missing_shadow_contract_disables_inference_without_exiting():
    class Metrics:
        def __init__(self):
            self.events = []

        def write_event(self, event, **fields):
            self.events.append((event, fields))

    worker = DisabledInferenceWorker()
    bridge = ObservationOnlyShadowBridge(
        worker, make_joint_contract(), ManualClock(100), start_tick=100
    )
    metrics = Metrics()
    keyboard = SimpleNamespace(poll=lambda _timeout_s: ("p",))
    handle_keyboard_events(bridge, keyboard, metrics)
    result = bridge.tick()
    assert bridge.state is BridgeState.PAUSED
    assert result.goal is None
    assert worker.busy is False
    assert metrics.events[0][0] == "activation_refused"


def test_contractless_shadow_validates_state_camera_freshness_and_skew():
    clock = ManualClock(500)
    worker = DisabledInferenceWorker()
    bridge = ObservationOnlyShadowBridge(
        worker, make_joint_contract(), clock, start_tick=500
    )
    state = TimedRobotState(np.zeros(43, np.float32), clock())
    camera = TimedCameraFrame(np.zeros((8, 8, 3), np.uint8), clock(), None)
    bridge.update_inputs(state, camera)
    assert bridge.observation_valid is True
    assert bridge.input_error is None
    accepted_state = bridge.last_valid_state
    accepted_snapshot = bridge.last_snapshot

    bad_shape = TimedRobotState(np.zeros(42, np.float32), clock())
    bridge.update_inputs(bad_shape, camera)
    assert bridge.observation_valid is False
    assert bridge.input_error == "measured state shape"
    assert bridge.last_valid_state is accepted_state
    assert bridge.last_snapshot is accepted_snapshot

    clock.set_tick(510)
    fresh_state = TimedRobotState(np.zeros(43, np.float32), clock())
    skewed_camera = TimedCameraFrame(np.zeros((8, 8, 3), np.uint8), 10.0, None)
    bridge.update_inputs(fresh_state, skewed_camera)
    assert bridge.observation_valid is False
    assert bridge.input_error == "receive-time skew"
    assert bridge.last_snapshot is accepted_snapshot

    invalid_time_camera = TimedCameraFrame(np.zeros((8, 8, 3), np.uint8), np.nan, None)
    bridge.update_inputs(fresh_state, invalid_time_camera)
    assert bridge.observation_valid is False
    assert bridge.input_error == "camera receive time"

    diagnostic_camera = TimedCameraFrame(np.zeros((8, 8, 3), np.uint8), clock(), np.nan)
    bridge.update_inputs(fresh_state, diagnostic_camera)
    assert bridge.observation_valid is True
    assert bridge.input_error is None
    assert bridge.camera_diagnostic is not None


def test_connection_evidence_requires_successful_runtime_observations():
    times = iter((1.0, 1.1, 1.2, 2.0, 2.1, 2.2))
    recorder = ConnectionEvidenceRecorder(clock=lambda: next(times))
    with pytest.raises(PreflightError, match="missing observed"):
        recorder.snapshot(required_components={"wbc", "camera", "policy"})
    recorder.observe_wbc_response({"env_type": "sim", "interface": "lo"})
    recorder.observe_camera_frame(
        "127.0.0.1",
        TimedCameraFrame(
            np.zeros((8, 8, 3), np.uint8),
            received_at=1.05,
            producer_timestamp=1.0,
        ),
    )
    recorder.observe_policy_contract("localhost", policy_payload())
    evidence = recorder.snapshot(required_components={"wbc", "camera", "policy"})
    assert [record["component"] for record in evidence] == [
        "wbc",
        "camera",
        "policy",
    ]
    assert count_real_interface_connections(evidence) == 0
    assert all(type(record["observed_at"]) is float for record in evidence)
    with pytest.raises(PreflightError, match="duplicate"):
        recorder.observe_policy_contract("localhost", policy_payload())

    remote = ConnectionEvidenceRecorder(clock=lambda: next(times))
    remote.observe_wbc_response({"env_type": "sim", "interface": "lo"})
    remote.observe_camera_frame(
        "127.0.0.1",
        TimedCameraFrame(
            np.zeros((8, 8, 3), np.uint8),
            received_at=2.05,
            producer_timestamp=2.0,
        ),
    )
    remote.observe_policy_contract("192.0.2.10", policy_payload())
    evidence = remote.snapshot(required_components={"wbc", "camera", "policy"})
    assert count_real_interface_connections(evidence) == 1
