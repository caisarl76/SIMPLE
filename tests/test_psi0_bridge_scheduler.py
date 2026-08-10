from dataclasses import replace

import numpy as np
import pytest

from simple.deploy.psi0_simple_bridge import (
    PolicyContract,
    Psi0SimpleBridge,
    RtcRequest,
    RtcResult,
    validate_rtc_result,
)
from tests.psi0_bridge_testkit import (
    ImmediateInference,
    ManualClock,
    fresh_inputs,
    make_joint_contract,
    make_policy_contract,
    policy_payload,
    sentinel_actions,
)


def test_rtc_sentinel_has_exact_request_and_handoff_ticks():
    clock = ManualClock(99)
    contract = make_policy_contract()
    inference = ImmediateInference(clock, contract)
    bridge = Psi0SimpleBridge(
        contract, make_joint_contract(), inference, clock, start_tick=99
    )
    robot_state, frame = fresh_inputs(clock)
    bridge.update_inputs(robot_state, frame)
    paused = bridge.tick()
    assert paused.source_kind == "hold"
    clock.set_tick(100)
    robot_state, frame = fresh_inputs(clock)
    bridge.update_inputs(robot_state, frame)
    bridge.activate()
    results = {}
    for tick in range(100, 113):
        clock.set_tick(tick)
        robot_state, frame = fresh_inputs(clock)
        bridge.update_inputs(robot_state, frame)
        results[tick] = bridge.tick()

    requests = inference.requests
    assert requests[0].observation_tick == 100
    assert requests[0].first_action_tick == 103
    assert [results[tick].source_kind for tick in range(100, 103)] == ["hold"] * 3
    assert [results[tick].source_tick for tick in range(103, 108)] == list(
        range(103, 108)
    )
    assert requests[1].observation_tick == 105
    assert requests[1].history_tick == 104
    assert requests[1].committed_global_ticks == (105, 106, 107)
    assert [results[tick].source_tick for tick in range(108, 113)] == list(
        range(108, 113)
    )
    committed = np.stack([results[tick].psi0_action for tick in range(105, 108)])
    np.testing.assert_array_equal(requests[1].committed_actions, committed)
    assert bridge.command_history_tick == 112
    assert [result.tick for result in results.values()] == list(range(100, 113))


@pytest.fixture
def clock():
    return ManualClock(100)


@pytest.mark.parametrize(
    "updates,error",
    [
        ({"rtc_delay_steps": 1}, "RTC horizon"),
        ({"rtc_delay_steps": 6}, "RTC horizon"),
        ({"prediction_horizon": 7}, "RTC horizon"),
        ({"rtc_training_max_delay": 3}, "RTC horizon"),
        ({"rtc_endpoint": "/act"}, "legacy policy endpoint"),
    ],
)
def test_policy_contract_rejects_invalid_rtc_tuple_or_endpoint(updates, error):
    with pytest.raises(ValueError, match=error):
        PolicyContract.from_dict(policy_payload(**updates))


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("schema", 123),
        ("observation_dim", "32"),
        ("prediction_horizon", 8.0),
        ("rtc_delay_steps", True),
        ("rtc_enabled", 1),
        ("test_only", "false"),
        ("source_episode_index", "7"),
    ],
)
def test_policy_contract_never_coerces_malformed_json_types(field, bad_value):
    with pytest.raises(TypeError, match=field):
        PolicyContract.from_dict(policy_payload(**{field: bad_value}))


def test_policy_contract_rejects_missing_and_extra_keys():
    missing = policy_payload()
    missing.pop("converter_commit")
    with pytest.raises(TypeError, match="keys"):
        PolicyContract.from_dict(missing)
    with pytest.raises(TypeError, match="keys"):
        PolicyContract.from_dict(policy_payload(extra="not allowed"))


def valid_request_and_result(clock, contract):
    request = RtcRequest(
        generation=1,
        session_id="session",
        request_seq=2,
        observation_tick=100,
        history_tick=99,
        observation=np.zeros((1, 32), np.float32),
        image=np.zeros((4, 4, 3), np.uint8),
        committed_actions=sentinel_actions(100, 3),
        reset=False,
        deadline_at=103 / 50.0,
    )
    result = RtcResult(
        generation=1,
        request_seq=2,
        completed_at=100 / 50.0 + 3 / 50.0,
        actions=sentinel_actions(103, 5),
        metadata={
            "session_id": "session",
            "request_seq": 2,
            "observation_tick": 100,
            "prediction_horizon": 8,
            "execution_horizon": 5,
            "rtc_delay_steps": 3,
            "first_action_tick": 103,
        },
    )
    return request, result


@pytest.mark.parametrize("rows", [4, 6])
def test_response_length_is_exact(clock, rows):
    contract = make_policy_contract()
    request, result = valid_request_and_result(clock, contract)
    result = replace(result, actions=sentinel_actions(103, rows))
    with pytest.raises(ValueError, match="shape"):
        validate_rtc_result(request, result, contract, make_joint_contract())


@pytest.mark.parametrize(
    "key,bad_value",
    [
        ("session_id", "other"),
        ("request_seq", 3),
        ("observation_tick", 99),
        ("prediction_horizon", 9),
        ("execution_horizon", 4),
        ("rtc_delay_steps", 2),
        ("first_action_tick", 102),
    ],
)
def test_every_response_metadata_mismatch_is_rejected(clock, key, bad_value):
    contract = make_policy_contract()
    request, result = valid_request_and_result(clock, contract)
    metadata = dict(result.metadata)
    metadata[key] = bad_value
    with pytest.raises(ValueError, match=key):
        validate_rtc_result(
            request,
            replace(result, metadata=metadata),
            contract,
            make_joint_contract(),
        )


def test_missing_extra_and_wrong_type_metadata_are_rejected(clock):
    contract = make_policy_contract()
    request, result = valid_request_and_result(clock, contract)
    for metadata in (
        {key: value for key, value in result.metadata.items() if key != "session_id"},
        {**result.metadata, "extra": 1},
        {**result.metadata, "request_seq": "2"},
        {**result.metadata, "request_seq": True},
    ):
        with pytest.raises(
            (TypeError, ValueError), match="metadata|request_seq|session_id"
        ):
            validate_rtc_result(
                request,
                replace(result, metadata=metadata),
                contract,
                make_joint_contract(),
            )


def test_nonfinite_out_of_bounds_and_late_results_are_rejected(clock):
    contract = make_policy_contract()
    request, result = valid_request_and_result(clock, contract)
    for actions in (
        np.where(np.indices(result.actions.shape)[1] == 0, np.nan, result.actions),
        np.where(np.indices(result.actions.shape)[1] == 31, 0.1, result.actions),
    ):
        with pytest.raises(ValueError):
            validate_rtc_result(
                request,
                replace(result, actions=actions),
                contract,
                make_joint_contract(),
            )
    with pytest.raises(ValueError, match="deadline"):
        validate_rtc_result(
            request,
            replace(result, completed_at=result.completed_at + 1e-6),
            contract,
            make_joint_contract(),
        )
