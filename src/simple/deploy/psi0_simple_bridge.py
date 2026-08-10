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
