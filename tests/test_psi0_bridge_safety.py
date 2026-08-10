import numpy as np
import pytest

from simple.deploy.psi0_simple_bridge import (
    Goal,
    JointContract,
    PSI0_ACTION_JOINT_NAMES,
    TimedCameraFrame,
    TimedRobotState,
    accept_measured_state,
    apply_slew_limit,
    build_bounded_hold,
    validate_action_suffix,
    validate_synchronized_snapshot,
)


@pytest.fixture
def contract():
    names = (
        *(f"leg_joint_{index}" for index in range(12)),
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        *PSI0_ACTION_JOINT_NAMES,
    )
    assert len(names) == len(set(names)) == 43
    return JointContract(
        names,
        names[12:],
        np.full(43, -2.0, np.float32),
        np.full(43, 2.0, np.float32),
    )


def state(q=None, received_at=10.0):
    return TimedRobotState(
        q=np.zeros(43, np.float32) if q is None else np.asarray(q, np.float32),
        received_at=received_at,
    )


def camera(received_at=10.0):
    return TimedCameraFrame(
        image=np.zeros((4, 4, 3), np.uint8),
        received_at=received_at,
        producer_timestamp=None,
    )


@pytest.mark.parametrize(
    "sample,error",
    [
        (state(np.zeros(42)), "shape"),
        (state(np.r_[np.nan, np.zeros(42)]), "finite"),
        (state(np.r_[2.051, np.zeros(42)]), "joint bounds"),
    ],
)
def test_invalid_state_never_replaces_last_valid(contract, sample, error):
    prior = state(np.full(43, 0.25), received_at=9.99)
    accepted, reason = accept_measured_state(prior, sample, contract, now=10.0)
    assert accepted is prior
    assert error in reason


@pytest.mark.parametrize("received_at", [9.899, 10.001])
def test_stale_or_future_state_never_replaces_last_valid(contract, received_at):
    prior = state(np.full(43, 0.25), received_at=9.99)
    candidate = state(np.full(43, 0.5), received_at=received_at)
    accepted, reason = accept_measured_state(prior, candidate, contract, now=10.0)
    assert accepted is prior
    assert reason == "measured state stale"


def test_state_tolerance_accepts_point_05_and_rejects_beyond(contract):
    q = np.zeros(43, np.float32)
    q[0] = 2.05
    accepted, reason = accept_measured_state(None, state(q), contract, now=10.0)
    assert accepted is not None and reason is None
    q[0] = 2.0501
    accepted, reason = accept_measured_state(None, state(q), contract, now=10.0)
    assert accepted is None and reason == "measured joint bounds"


@pytest.mark.parametrize(
    "state_time,camera_time,now,error",
    [
        (9.899, 10.0, 10.0, "state stale"),
        (10.0, 9.749, 10.0, "camera stale"),
        (10.0, 9.899, 10.0, "receive-time skew"),
        (10.0, np.nan, 10.0, "camera receive time"),
        (10.0, "not-a-time", 10.0, "camera receive time"),
    ],
)
def test_snapshot_freshness_and_skew_are_independent(
    state_time, camera_time, now, error
):
    with pytest.raises(ValueError, match=error):
        validate_synchronized_snapshot(
            state(received_at=state_time), camera(received_at=camera_time), now
        )


@pytest.mark.parametrize("producer_time", [None, np.nan, np.inf, "bad-time"])
def test_producer_timestamp_is_sanitized_but_never_gates_control(producer_time):
    frame = TimedCameraFrame(np.zeros((4, 4, 3), np.uint8), 10.0, producer_time)
    snapshot = validate_synchronized_snapshot(state(), frame, now=10.0)
    assert snapshot.camera.producer_timestamp is None
    assert snapshot.camera.producer_timestamp_diagnostic is not None


def test_finite_producer_timestamp_is_preserved_as_diagnostic_metadata():
    frame = TimedCameraFrame(np.zeros((4, 4, 3), np.uint8), 10.0, 123.5)
    snapshot = validate_synchronized_snapshot(state(), frame, now=10.0)
    assert snapshot.camera.producer_timestamp == 123.5
    assert snapshot.camera.producer_timestamp_diagnostic is None


def valid_suffix(rows=24):
    action = np.zeros((rows, 36), np.float32)
    action[:, 31] = 0.74
    return action


@pytest.mark.parametrize(
    "column,value,error",
    [
        (0, 2.0001, "joint bounds"),
        (31, 0.1999, "height"),
        (32, 0.5001, "planar navigation"),
        (33, -0.5001, "planar navigation"),
        (34, 1.0001, "turning"),
        (35, np.pi + 1e-4, "target yaw"),
        (7, np.nan, "finite"),
    ],
)
def test_one_bad_element_rejects_the_whole_suffix(contract, column, value, error):
    actions = valid_suffix()
    actions[13, column] = value
    original = actions.copy()
    with pytest.raises(ValueError, match=error):
        validate_action_suffix(actions, expected_s=24, contract=contract)
    np.testing.assert_array_equal(actions, original)


@pytest.mark.parametrize("shape", [(23, 36), (25, 36), (24, 35), (36,)])
def test_suffix_shape_is_exact(contract, shape):
    with pytest.raises(ValueError, match=r"expected action suffix shape \(24, 36\)"):
        validate_action_suffix(np.zeros(shape, np.float32), 24, contract)


PER_TICK_LIMITS = {
    "arm": 1.0 / 50.0,
    "hand": 2.0 / 50.0,
    "waist": 0.5 / 50.0,
    "height": 0.1 / 50.0,
    "planar_navigation": 0.5 / 50.0,
    "turning": 2.0 / 50.0,
    "target_yaw": 1.0 / 50.0,
}


def test_one_tick_slew_limits_every_action_group(contract):
    previous = valid_suffix(1)[0]
    requested = np.full(36, 1.5, np.float32)
    requested[31] = 0.74
    requested[35] = -np.pi + 0.01
    previous[35] = np.pi - 0.01
    limited = apply_slew_limit(previous, requested, contract, dt=0.02)
    assert (
        np.max(np.abs(limited[:14] - previous[:14])) <= PER_TICK_LIMITS["hand"] + 1e-7
    )
    assert (
        np.max(np.abs(limited[14:28] - previous[14:28]))
        <= PER_TICK_LIMITS["arm"] + 1e-7
    )
    assert (
        np.max(np.abs(limited[28:31] - previous[28:31]))
        <= PER_TICK_LIMITS["waist"] + 1e-7
    )
    assert abs(limited[31] - previous[31]) <= PER_TICK_LIMITS["height"] + 1e-7
    assert (
        np.max(np.abs(limited[32:34] - previous[32:34]))
        <= PER_TICK_LIMITS["planar_navigation"] + 1e-7
    )
    assert abs(limited[34] - previous[34]) <= PER_TICK_LIMITS["turning"] + 1e-7
    yaw_delta = ((limited[35] - previous[35] + np.pi) % (2 * np.pi)) - np.pi
    assert abs(yaw_delta) <= PER_TICK_LIMITS["target_yaw"] + 1e-7


def safe_goal(upper_names, *, height=0.74):
    return Goal(
        target_upper_body_pose=np.zeros(len(upper_names), np.float32),
        base_height_command=np.array([height], np.float32),
        navigate_cmd=np.ones(4, np.float32),
        timestamp=1.0,
        target_time=1.02,
    )


def test_tolerated_measured_overshoot_is_clamped_for_hold(contract):
    q = np.zeros(43, np.float32)
    q[12] = 2.04
    result = build_bounded_hold(10.0, state(q), None, contract)
    assert result.source == "measured_clamped"
    assert result.goal.target_upper_body_pose[0] == np.float32(2.0)
    assert result.clamped_joints[0][0] == "waist_yaw_joint"
    assert result.clamped_joints[0][2] == 2.0
    np.testing.assert_array_equal(result.goal.navigate_cmd, np.zeros(4, np.float32))


def test_invalid_measured_state_cannot_be_hold_source(contract):
    q = np.zeros(43, np.float32)
    q[12] = 2.051
    assert build_bounded_hold(10.0, state(q), None, contract) is None


def test_stale_state_falls_back_to_frozen_last_safe_goal(contract):
    previous = safe_goal(contract.upper_body_joint_names, height=0.8)
    result = build_bounded_hold(10.0, state(received_at=9.0), previous, contract)
    assert result.source == "last_safe_published"
    assert result.goal.base_height_command[0] == pytest.approx(0.74)
    np.testing.assert_array_equal(result.goal.navigate_cmd, np.zeros(4, np.float32))
    frozen = result.goal.target_upper_body_pose.copy()
    previous.target_upper_body_pose[:] = 1.0
    np.testing.assert_array_equal(result.goal.target_upper_body_pose, frozen)


def test_no_safe_source_returns_none_and_r0_hold_is_identical(contract):
    assert build_bounded_hold(10.0, None, None, contract) is None
    result = build_bounded_hold(10.0, state(), None, contract)
    prefix = np.repeat(result.psi0_action[None], 6, axis=0)
    assert prefix.shape == (6, 36)
    np.testing.assert_array_equal(prefix, np.broadcast_to(prefix[0], prefix.shape))
