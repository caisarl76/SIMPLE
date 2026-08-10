from dataclasses import dataclass
from enum import Enum

import numpy as np


class BridgeMode(str, Enum):
    SHADOW = "shadow"
    SIM_CONTROL = "sim-control"


class BridgeState(str, Enum):
    PAUSED = "paused"
    ACTIVE = "active"
    FAULT = "fault"
    STOPPED = "stopped"


@dataclass(frozen=True)
class Goal:
    target_upper_body_pose: np.ndarray
    base_height_command: np.ndarray
    navigate_cmd: np.ndarray
    timestamp: float
    target_time: float


@dataclass(frozen=True)
class JointContract:
    joint_names: tuple[str, ...]
    upper_body_joint_names: tuple[str, ...]
    lower_position_limits: np.ndarray
    upper_position_limits: np.ndarray


@dataclass(frozen=True)
class TimedRobotState:
    q: np.ndarray
    received_at: float


@dataclass(frozen=True)
class TimedCameraFrame:
    image: np.ndarray
    received_at: float
    producer_timestamp: float | None
    producer_timestamp_diagnostic: str | None = None


@dataclass(frozen=True)
class InputSnapshot:
    state: TimedRobotState
    camera: TimedCameraFrame


@dataclass(frozen=True)
class HoldResult:
    goal: Goal
    psi0_action: np.ndarray
    source: str
    clamped_joints: tuple[tuple[str, float, float], ...]


PSI0_STATE_JOINT_NAMES = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
PSI0_ACTION_JOINT_NAMES = PSI0_STATE_JOINT_NAMES
PSI0_WAIST_ACTION_NAMES = (
    "waist_roll_joint",
    "waist_pitch_joint",
    "waist_yaw_joint",
)
PSI0_UPPER_ACTION_NAMES = frozenset(
    (*PSI0_ACTION_JOINT_NAMES, *PSI0_WAIST_ACTION_NAMES)
)


def _upper_names(value):
    names = tuple(value)
    if (
        len(names) != 31
        or len(set(names)) != 31
        or set(names) != PSI0_UPPER_ACTION_NAMES
    ):
        raise ValueError("expected one-to-one 31-joint upper body")
    return names


def build_psi0_observation(q, joint_names, command_history_rpyh):
    q = np.asarray(q, dtype=np.float32)
    names = tuple(joint_names)
    name_to_index = {name: index for index, name in enumerate(names)}
    if q.shape != (43,) or len(names) != 43 or len(name_to_index) != 43:
        raise ValueError("expected one-to-one 43-joint state")
    if not np.isfinite(q).all() or any(
        name not in name_to_index for name in PSI0_STATE_JOINT_NAMES
    ):
        raise ValueError("state is non-finite or missing a PSI0 joint")
    history = np.asarray(command_history_rpyh, dtype=np.float32)
    if history.shape != (4,) or not np.isfinite(history).all():
        raise ValueError("expected [roll,pitch,yaw,height] history")
    values = [q[name_to_index[name]] for name in PSI0_STATE_JOINT_NAMES]
    result = np.concatenate([np.asarray(values, np.float32), history])[None]
    return np.ascontiguousarray(result, dtype=np.float32)


def map_psi0_action_to_goal(action, upper_body_joint_names, now):
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (36,) or not np.isfinite(action).all():
        raise ValueError("expected finite 36-D PSI0 action")
    names = _upper_names(upper_body_joint_names)
    named = dict(zip(PSI0_ACTION_JOINT_NAMES, action[:28], strict=True))
    named.update(dict(zip(PSI0_WAIST_ACTION_NAMES, action[28:31], strict=True)))
    timestamp = float(now)
    if not np.isfinite(timestamp):
        raise ValueError("goal timestamp must be finite")
    return Goal(
        target_upper_body_pose=np.asarray(
            [named[name] for name in names], dtype=np.float32
        ),
        base_height_command=np.ascontiguousarray(action[31:32]),
        navigate_cmd=np.ascontiguousarray(action[32:36]),
        timestamp=timestamp,
        target_time=timestamp + 0.02,
    )


def goal_to_psi0_action(goal, upper_body_joint_names):
    names = _upper_names(upper_body_joint_names)
    upper = np.asarray(goal.target_upper_body_pose, dtype=np.float32)
    height = np.asarray(goal.base_height_command, dtype=np.float32)
    navigation = np.asarray(goal.navigate_cmd, dtype=np.float32)
    if upper.shape != (31,) or height.shape != (1,) or navigation.shape != (4,):
        raise ValueError("goal arrays must have shapes (31,), (1,), and (4,)")
    if not all(np.isfinite(value).all() for value in (upper, height, navigation)):
        raise ValueError("goal arrays must be finite")
    named = dict(zip(names, upper, strict=True))
    action = np.concatenate(
        [
            np.asarray([named[name] for name in PSI0_ACTION_JOINT_NAMES], np.float32),
            np.asarray([named[name] for name in PSI0_WAIST_ACTION_NAMES], np.float32),
            height,
            navigation,
        ]
    )
    return np.ascontiguousarray(action, dtype=np.float32)


def validate_measured_state(sample, contract, now, tolerance=0.05):
    if type(sample) is not TimedRobotState:
        raise ValueError("measured state type")
    if type(now) not in (float, int) or not np.isfinite(now):
        raise ValueError("measured validation time")
    if type(sample.received_at) not in (float, int) or not np.isfinite(
        sample.received_at
    ):
        raise ValueError("measured receive time")
    age = float(now) - float(sample.received_at)
    if age < 0.0 or age > 0.10:
        raise ValueError("measured state stale")
    q = np.asarray(sample.q)
    if q.shape != (43,):
        raise ValueError("measured state shape")
    if not np.issubdtype(q.dtype, np.number) or not np.isfinite(q).all():
        raise ValueError("measured state finite")
    if len(contract.joint_names) != 43 or len(set(contract.joint_names)) != 43:
        raise ValueError("measured state one-to-one joint names")
    lower = np.asarray(contract.lower_position_limits, np.float32)
    upper = np.asarray(contract.upper_position_limits, np.float32)
    if lower.shape != (43,) or upper.shape != (43,):
        raise ValueError("joint contract limit shape")
    q32 = q.astype(np.float32, copy=True)
    if np.any(q32 < lower - tolerance) or np.any(q32 > upper + tolerance):
        raise ValueError("measured joint bounds")
    return TimedRobotState(q32, float(sample.received_at))


def accept_measured_state(last_valid, candidate, contract, now):
    try:
        accepted = validate_measured_state(candidate, contract, now)
    except ValueError as error:
        return last_valid, str(error)
    return accepted, None


def sanitize_producer_timestamp(value):
    if type(value) in (float, int) and np.isfinite(value):
        return float(value), None
    if value is None:
        return None, "producer timestamp missing"
    return None, "producer timestamp ignored: nonnumeric or nonfinite"


def validate_synchronized_snapshot(state, camera, now):
    if type(state) is not TimedRobotState:
        raise ValueError("state missing or wrong type")
    if type(camera) is not TimedCameraFrame:
        raise ValueError("camera missing or wrong type")
    if type(now) not in (float, int) or not np.isfinite(now):
        raise ValueError("snapshot validation time")
    if type(state.received_at) not in (float, int) or not np.isfinite(
        state.received_at
    ):
        raise ValueError("state receive time")
    if type(camera.received_at) not in (float, int) or not np.isfinite(
        camera.received_at
    ):
        raise ValueError("camera receive time")
    state_age = float(now) - float(state.received_at)
    camera_age = float(now) - float(camera.received_at)
    if state_age < 0.0 or state_age > 0.10:
        raise ValueError("state stale")
    if camera_age < 0.0 or camera_age > 0.25:
        raise ValueError("camera stale")
    if abs(state.received_at - camera.received_at) > 0.10:
        raise ValueError("receive-time skew")
    image = np.asarray(camera.image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("camera shape/dtype")
    producer_timestamp, diagnostic = sanitize_producer_timestamp(
        camera.producer_timestamp
    )
    if camera.producer_timestamp_diagnostic is not None:
        diagnostic = camera.producer_timestamp_diagnostic
    return InputSnapshot(
        TimedRobotState(np.asarray(state.q, np.float32).copy(), state.received_at),
        TimedCameraFrame(
            np.ascontiguousarray(image).copy(),
            camera.received_at,
            producer_timestamp,
            diagnostic,
        ),
    )


def _action_joint_limits(contract):
    index = {name: i for i, name in enumerate(contract.joint_names)}
    action_names = (
        *PSI0_ACTION_JOINT_NAMES,
        "waist_roll_joint",
        "waist_pitch_joint",
        "waist_yaw_joint",
    )
    if len(index) != 43 or any(name not in index for name in action_names):
        raise ValueError("joint contract does not cover PSI0 action names")
    lower = np.asarray(contract.lower_position_limits, np.float32)
    upper = np.asarray(contract.upper_position_limits, np.float32)
    selected = np.asarray([index[name] for name in action_names], np.int64)
    return lower[selected], upper[selected]


def validate_action_suffix(actions, expected_s, contract):
    array = np.asarray(actions)
    expected_shape = (expected_s, 36)
    if array.shape != expected_shape:
        raise ValueError(f"expected action suffix shape {expected_shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("action suffix finite")
    values = array.astype(np.float32, copy=False)
    lower, upper = _action_joint_limits(contract)
    if np.any(values[:, :31] < lower) or np.any(values[:, :31] > upper):
        raise ValueError("action joint bounds")
    if np.any(values[:, 31] < 0.20) or np.any(values[:, 31] > 0.74):
        raise ValueError("action height")
    if np.any(np.abs(values[:, 32:34]) > 0.5):
        raise ValueError("action planar navigation")
    if np.any(np.abs(values[:, 34]) > 1.0):
        raise ValueError("action turning")
    if np.any(values[:, 35] < -np.pi) or np.any(values[:, 35] > np.pi):
        raise ValueError("action target yaw")
    return values.copy()


def apply_slew_limit(previous_action, requested_action, contract, dt=0.02):
    previous = np.asarray(previous_action, np.float32)
    requested = np.asarray(requested_action, np.float32)
    if previous.shape != (36,) or requested.shape != (36,):
        raise ValueError("slew actions must both have shape (36,)")
    if not np.isfinite(previous).all() or not np.isfinite(requested).all():
        raise ValueError("slew actions must be finite")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("slew dt must be positive and finite")
    rates = np.asarray(
        [2.0] * 14 + [1.0] * 14 + [0.5] * 3 + [0.1] + [0.5] * 2 + [2.0] + [1.0],
        np.float32,
    )
    limited = previous + np.clip(requested - previous, -rates * dt, rates * dt)
    yaw_delta = (requested[35] - previous[35] + np.pi) % (2 * np.pi) - np.pi
    limited[35] = previous[35] + np.clip(yaw_delta, -dt, dt)
    limited[35] = (limited[35] + np.pi) % (2 * np.pi) - np.pi
    return limited.astype(np.float32, copy=False)


def _immutable_float32(values):
    result = np.asarray(values, np.float32).copy()
    result.setflags(write=False)
    return result


def build_bounded_hold(now, last_valid_state, last_safe_goal, contract):
    index = {name: i for i, name in enumerate(contract.joint_names)}
    lower = np.asarray(contract.lower_position_limits, np.float32)
    upper = np.asarray(contract.upper_position_limits, np.float32)
    upper_indices = np.asarray(
        [index[name] for name in contract.upper_body_joint_names], np.int64
    )
    clamped = []

    measured_usable = False
    if (
        last_valid_state is not None
        and 0.0 <= now - last_valid_state.received_at <= 0.10
    ):
        try:
            measured = validate_measured_state(last_valid_state, contract, now)
        except ValueError:
            measured_usable = False
        else:
            measured_usable = True
    if measured_usable:
        raw_upper = measured.q[upper_indices]
        target = np.clip(raw_upper, lower[upper_indices], upper[upper_indices])
        for name, raw, bounded in zip(
            contract.upper_body_joint_names, raw_upper, target, strict=True
        ):
            if raw != bounded:
                clamped.append((name, float(raw), float(bounded)))
        raw_height = (
            float(last_safe_goal.base_height_command[0])
            if last_safe_goal is not None
            else 0.74
        )
        source = "measured_clamped"
    elif last_safe_goal is not None:
        raw_upper = np.asarray(last_safe_goal.target_upper_body_pose, np.float32)
        if raw_upper.shape != (31,):
            raise ValueError("last safe upper-body hold shape")
        target = np.clip(raw_upper, lower[upper_indices], upper[upper_indices])
        for name, raw, bounded in zip(
            contract.upper_body_joint_names, raw_upper, target, strict=True
        ):
            if raw != bounded:
                clamped.append((name, float(raw), float(bounded)))
        raw_height = float(last_safe_goal.base_height_command[0])
        source = "last_safe_published"
    else:
        return None

    height = float(np.clip(raw_height, 0.20, 0.74))
    goal = Goal(
        target_upper_body_pose=_immutable_float32(target),
        base_height_command=_immutable_float32([height]),
        navigate_cmd=_immutable_float32(np.zeros(4, np.float32)),
        timestamp=float(now),
        target_time=float(now + 0.02),
    )
    action = _immutable_float32(
        goal_to_psi0_action(goal, contract.upper_body_joint_names)
    )
    return HoldResult(goal, action, source, tuple(clamped))
