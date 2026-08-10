from copy import deepcopy

import numpy as np
import pytest

from simple.deploy.psi0_simple_bridge import (
    ActivationRefused,
    BridgeState,
    Psi0SimpleBridge,
    RtcResult,
    TimedCameraFrame,
    TimedRobotState,
)
from tests.psi0_bridge_testkit import (
    ManualClock,
    fresh_inputs,
    make_joint_contract,
    make_policy_contract,
    sentinel_actions,
)


class BlockingInference:
    def __init__(self):
        self.requests = []
        self.result = None
        self.physical_busy = False

    @property
    def busy(self):
        return self.physical_busy or self.result is not None

    def submit(self, request):
        if self.physical_busy:
            raise RuntimeError("physical worker busy")
        self.requests.append(deepcopy(request))
        self.physical_busy = True

    def poll(self):
        result, self.result = self.result, None
        return result

    def release(self, result):
        self.result = result
        self.physical_busy = False


def ready_bridge(inference):
    clock = ManualClock(99)
    bridge = Psi0SimpleBridge(
        make_policy_contract(),
        make_joint_contract(),
        inference,
        clock,
        start_tick=99,
    )
    bridge.update_inputs(*fresh_inputs(clock))
    return bridge, clock


def assert_policy_buffers_empty(status):
    assert status.scheduled_ticks == ()
    assert status.staged_first_tick is None
    assert status.staged_count == 0
    assert status.committed_ticks == ()
    assert status.deadline is None
    assert status.logical_request_active is False


def test_startup_requires_one_consumed_paused_hold():
    inference = BlockingInference()
    bridge, _ = ready_bridge(inference)
    assert bridge.state is BridgeState.PAUSED
    with pytest.raises(ActivationRefused, match="paused hold"):
        bridge.handle_toggle()
    assert inference.requests == []
    hold = bridge.tick()
    assert hold.source_kind == "hold"
    bridge.handle_toggle()
    assert bridge.state is BridgeState.ACTIVE
    assert len(inference.requests) == 1


@pytest.mark.parametrize("transition", ["pause", "enter_fault", "stop"])
def test_exit_transition_atomically_invalidates_every_policy_buffer(transition):
    inference = BlockingInference()
    bridge, clock = ready_bridge(inference)
    bridge.tick()
    clock.set_tick(100)
    bridge.update_inputs(*fresh_inputs(clock))
    bridge.activate()
    before = bridge.status()
    if transition == "enter_fault":
        getattr(bridge, transition)("test fault")
    else:
        getattr(bridge, transition)()
    after = bridge.status()
    assert after.generation == before.generation + 1
    assert_policy_buffers_empty(after)


def test_fault_holds_zero_navigation_and_rearm_waits_for_physical_idle():
    inference = BlockingInference()
    bridge, clock = ready_bridge(inference)
    bridge.tick()
    clock.set_tick(100)
    bridge.update_inputs(*fresh_inputs(clock))
    bridge.activate()
    old_request = inference.requests[0]
    bridge.enter_fault("deadline")
    fault_tick = bridge.tick()
    assert bridge.state is BridgeState.FAULT
    np.testing.assert_array_equal(fault_tick.goal.navigate_cmd, np.zeros(4, np.float32))
    with pytest.raises(ActivationRefused, match="worker busy"):
        bridge.handle_toggle()
    late = RtcResult(
        generation=old_request.generation,
        request_seq=old_request.request_seq,
        completed_at=clock(),
        actions=sentinel_actions(103, 5),
        metadata={
            "session_id": old_request.session_id,
            "request_seq": 0,
            "observation_tick": 100,
            "prediction_horizon": 8,
            "execution_horizon": 5,
            "rtc_delay_steps": 3,
            "first_action_tick": 103,
        },
    )
    inference.release(late)
    with pytest.raises(ActivationRefused, match="worker busy"):
        bridge.handle_toggle()
    assert bridge.state is BridgeState.FAULT
    assert len(inference.requests) == 1
    bridge.update_inputs(*fresh_inputs(clock))
    bridge.tick()
    assert bridge.metrics.discarded_old_generation_results == 1
    bridge.handle_toggle()
    assert bridge.state is BridgeState.ACTIVE
    assert len(inference.requests) == 2


def test_active_invalid_state_latches_one_stable_fault_reason():
    inference = BlockingInference()
    bridge, clock = ready_bridge(inference)
    bridge.tick()
    clock.set_tick(100)
    bridge.update_inputs(*fresh_inputs(clock))
    bridge.activate()
    bad = TimedRobotState(np.r_[np.nan, np.zeros(42)], clock())
    bridge.update_inputs(bad, fresh_inputs(clock)[1])
    first = bridge.tick()
    second = bridge.tick()
    assert bridge.state is BridgeState.FAULT
    assert bridge.fault_reason == "measured state finite"
    assert bridge.metrics.fault_transitions == 1
    np.testing.assert_array_equal(first.goal.navigate_cmd, np.zeros(4, np.float32))
    np.testing.assert_array_equal(second.goal.navigate_cmd, np.zeros(4, np.float32))


@pytest.mark.parametrize("received_at", [np.nan, "not-a-time"])
def test_invalid_camera_receive_time_becomes_active_fault(received_at):
    inference = BlockingInference()
    bridge, clock = ready_bridge(inference)
    bridge.tick()
    clock.set_tick(100)
    bridge.update_inputs(*fresh_inputs(clock))
    bridge.activate()
    state, _ = fresh_inputs(clock)
    bad_camera = TimedCameraFrame(np.zeros((8, 8, 3), np.uint8), received_at, None)
    bridge.update_inputs(state, bad_camera)
    result = bridge.tick()
    assert bridge.state is BridgeState.FAULT
    assert bridge.fault_reason == "camera receive time"
    assert result.source_kind == "hold"


def test_paused_full_bridge_reports_invalid_input_without_faulting():
    inference = BlockingInference()
    bridge, clock = ready_bridge(inference)
    assert bridge.input_status() == (True, None)
    assert bridge.observation_valid is True
    assert bridge.input_error is None

    bad = TimedRobotState(np.r_[np.nan, np.zeros(42)], clock())
    bridge.update_inputs(bad, fresh_inputs(clock)[1])
    assert bridge.input_status() == (False, "measured state finite")
    assert bridge.observation_valid is False
    assert bridge.input_error == "measured state finite"
    assert bridge.state is BridgeState.PAUSED
    assert inference.requests == []


def test_paused_hold_is_captured_once_and_ignores_later_measurements():
    bridge, clock = ready_bridge(BlockingInference())
    first = bridge.tick()
    clock.set_tick(100)
    changed_q = np.zeros(43, np.float32)
    changed_q[0] = 0.1
    changed = TimedRobotState(changed_q, clock())
    bridge.update_inputs(changed, fresh_inputs(clock)[1])
    second = bridge.tick()
    assert first.source_kind == second.source_kind == "hold"
    np.testing.assert_array_equal(first.psi0_action, second.psi0_action)


def test_initial_activation_and_fault_rearm_each_increment_generation():
    inference = BlockingInference()
    bridge, clock = ready_bridge(inference)
    bridge.tick()
    clock.set_tick(100)
    bridge.update_inputs(*fresh_inputs(clock))
    initial = bridge.generation
    bridge.activate()
    assert bridge.generation == initial + 1
    bridge.enter_fault("test")
    assert bridge.generation == initial + 2
    inference.physical_busy = False
    bridge.tick()
    bridge.activate()
    assert bridge.generation == initial + 3


def test_malformed_metadata_type_latches_fault_instead_of_escaping():
    inference = BlockingInference()
    bridge, clock = ready_bridge(inference)
    bridge.tick()
    clock.set_tick(100)
    bridge.update_inputs(*fresh_inputs(clock))
    bridge.activate()
    request = inference.requests[0]
    inference.release(
        RtcResult(
            generation=request.generation,
            request_seq=request.request_seq,
            completed_at=clock(),
            actions=sentinel_actions(103, 5),
            metadata={
                "session_id": request.session_id,
                "request_seq": "0",
                "observation_tick": 100,
                "prediction_horizon": 8,
                "execution_horizon": 5,
                "rtc_delay_steps": 3,
                "first_action_tick": 103,
            },
        )
    )
    result = bridge.tick()
    assert bridge.state is BridgeState.FAULT
    assert "wrong type" in bridge.fault_reason
    assert result.source_kind == "hold"
