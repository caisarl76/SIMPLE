import re
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Protocol
import uuid

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


@dataclass(frozen=True)
class PolicyContract:
    schema: str
    checkpoint_sha256: str
    dataset_manifest_sha256: str
    raw_episode_sha256: str
    processed_episode_sha256: str
    source_episode_index: int
    processed_episode_index: int
    converter_commit: str
    server_commit: str
    converter_layout: str
    prediction_horizon: int
    execution_horizon: int
    rtc_delay_steps: int
    rtc_training_max_delay: int
    action_frequency_hz: int
    observation_dim: int
    action_dim: int
    endpoint: str
    request_semantics: str
    response_semantics: str
    image_key: str
    camera_color_order: str
    rtc_enabled: bool
    test_only: bool

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PolicyContract":
        wire_types = {
            "schema": str,
            "test_only": bool,
            "checkpoint_sha256": str,
            "dataset_manifest_sha256": str,
            "raw_episode_sha256": str,
            "processed_episode_sha256": str,
            "source_episode_index": int,
            "processed_episode_index": int,
            "converter_commit": str,
            "server_commit": str,
            "converter_layout": str,
            "observation_dim": int,
            "action_dim": int,
            "action_frequency_hz": int,
            "prediction_horizon": int,
            "execution_horizon": int,
            "rtc_delay_steps": int,
            "rtc_training_max_delay": int,
            "rtc_enabled": bool,
            "rtc_endpoint": str,
            "request_semantics": str,
            "response_semantics": str,
            "image_key": str,
            "camera_color_order": str,
        }
        if type(payload) is not dict or set(payload) != set(wire_types):
            raise TypeError("policy contract keys do not exactly match v2 schema")
        for key, expected_type in wire_types.items():
            if type(payload[key]) is not expected_type:
                raise TypeError(
                    f"policy contract field {key} must be {expected_type.__name__}"
                )
        contract = cls(
            schema=payload["schema"],
            checkpoint_sha256=payload["checkpoint_sha256"],
            dataset_manifest_sha256=payload["dataset_manifest_sha256"],
            raw_episode_sha256=payload["raw_episode_sha256"],
            processed_episode_sha256=payload["processed_episode_sha256"],
            source_episode_index=payload["source_episode_index"],
            processed_episode_index=payload["processed_episode_index"],
            converter_commit=payload["converter_commit"],
            server_commit=payload["server_commit"],
            converter_layout=payload["converter_layout"],
            prediction_horizon=payload["prediction_horizon"],
            execution_horizon=payload["execution_horizon"],
            rtc_delay_steps=payload["rtc_delay_steps"],
            rtc_training_max_delay=payload["rtc_training_max_delay"],
            action_frequency_hz=payload["action_frequency_hz"],
            observation_dim=payload["observation_dim"],
            action_dim=payload["action_dim"],
            endpoint=payload["rtc_endpoint"],
            request_semantics=payload["request_semantics"],
            response_semantics=payload["response_semantics"],
            image_key=payload["image_key"],
            camera_color_order=payload["camera_color_order"],
            rtc_enabled=payload["rtc_enabled"],
            test_only=payload["test_only"],
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        if self.schema != "simple.psi0.policy-contract.v2":
            raise ValueError("unsupported policy contract schema")
        if not re.fullmatch(r"[0-9a-f]{64}", self.checkpoint_sha256):
            raise ValueError("invalid checkpoint SHA-256")
        if not re.fullmatch(r"[0-9a-f]{64}", self.dataset_manifest_sha256):
            raise ValueError("invalid dataset-manifest SHA-256")
        if not re.fullmatch(r"[0-9a-f]{64}", self.raw_episode_sha256):
            raise ValueError("invalid raw-episode SHA-256")
        if not re.fullmatch(r"[0-9a-f]{64}", self.processed_episode_sha256):
            raise ValueError("invalid processed-episode SHA-256")
        if self.source_episode_index < 0 or self.processed_episode_index < 0:
            raise ValueError("episode indices must be nonnegative")
        if not re.fullmatch(r"[0-9a-f]{40}", self.converter_commit):
            raise ValueError("invalid converter commit")
        if not re.fullmatch(r"[0-9a-f]{40}", self.server_commit):
            raise ValueError("invalid PSI0 server commit")
        if self.converter_layout != "g1_simple_32_rpyh_v2":
            raise ValueError("unsupported PSI0 converter layout")
        if not self.rtc_enabled:
            raise ValueError("RTC must be enabled")
        if (
            self.action_frequency_hz != 50
            or self.observation_dim != 32
            or self.action_dim != 36
        ):
            raise ValueError("policy dimensions/frequency do not match bridge")
        d = self.rtc_delay_steps
        s = self.execution_horizon
        p = self.prediction_horizon
        if not (2 <= d <= s and d + s <= p and d < self.rtc_training_max_delay):
            raise ValueError("invalid RTC horizon/delay contract")
        if self.endpoint != "/act-rtc-v1":
            raise ValueError("legacy policy endpoint is ineligible")
        if self.request_semantics != "exact-post-slew-committed-prefix":
            raise ValueError("unsupported RTC request semantics")
        if self.response_semantics != "denormalized-executable-suffix":
            raise ValueError("unsupported RTC response semantics")
        if self.image_key != "rgb_head_stereo_left" or self.camera_color_order != "rgb":
            raise ValueError("policy image contract does not match bridge")


@dataclass(frozen=True)
class RtcRequest:
    generation: int
    session_id: str
    request_seq: int
    observation_tick: int
    history_tick: int
    observation: np.ndarray
    image: np.ndarray
    committed_actions: np.ndarray
    reset: bool
    deadline_at: float

    @property
    def first_action_tick(self) -> int:
        return self.observation_tick + self.committed_actions.shape[0]

    @property
    def committed_global_ticks(self) -> tuple[int, ...]:
        return tuple(range(self.observation_tick, self.first_action_tick))


@dataclass(frozen=True)
class RtcResult:
    generation: int
    request_seq: int
    completed_at: float
    actions: np.ndarray | None
    metadata: dict[str, object] | None
    error: str | None = None


class InferencePort(Protocol):
    @property
    def busy(self) -> bool:
        raise NotImplementedError

    def submit(self, request: RtcRequest) -> None:
        raise NotImplementedError

    def poll(self) -> RtcResult | None:
        raise NotImplementedError


class ActivationRefused(RuntimeError):
    """Expected fail-closed refusal of a local activation request."""


RTC_RESPONSE_FIELDS = (
    "session_id",
    "request_seq",
    "observation_tick",
    "prediction_horizon",
    "execution_horizon",
    "rtc_delay_steps",
    "first_action_tick",
)


def validate_rtc_result(request, result, contract, joints):
    if result.error is not None:
        raise ValueError(f"inference error: {result.error}")
    if type(result.completed_at) is not float or not np.isfinite(result.completed_at):
        raise ValueError("result completion time must be finite float")
    if result.completed_at > request.deadline_at:
        raise ValueError("RTC result missed monotonic deadline")
    metadata = result.metadata
    if type(metadata) is not dict or set(metadata) != set(RTC_RESPONSE_FIELDS):
        raise TypeError("RTC response metadata key set is invalid")
    expected = {
        "session_id": request.session_id,
        "request_seq": request.request_seq,
        "observation_tick": request.observation_tick,
        "prediction_horizon": contract.prediction_horizon,
        "execution_horizon": contract.execution_horizon,
        "rtc_delay_steps": contract.rtc_delay_steps,
        "first_action_tick": request.first_action_tick,
    }
    for key, value in expected.items():
        expected_type = str if key == "session_id" else int
        if type(metadata[key]) is not expected_type:
            raise TypeError(f"RTC response {key} has wrong type")
        if metadata[key] != value:
            raise ValueError(f"RTC response {key} mismatch")
    actions = np.asarray(result.actions)
    validate_action_suffix(actions, contract.execution_horizon, joints)
    return actions.astype(np.float32, copy=True)


@dataclass(frozen=True)
class BridgeStatus:
    state: BridgeState
    generation: int
    scheduled_ticks: tuple[int, ...]
    staged_first_tick: int | None
    staged_count: int
    committed_ticks: tuple[int, ...]
    deadline: float | None
    logical_request_active: bool


@dataclass
class BridgeMetrics:
    discarded_old_generation_results: int = 0
    discarded_late_results: int = 0
    fault_transitions: int = 0
    requests_submitted: int = 0
    results_accepted: int = 0
    first_fault_at: float | None = None


@dataclass(frozen=True)
class TickResult:
    tick: int
    goal: Goal | None
    psi0_action: np.ndarray | None
    source_tick: int | None
    source_kind: str


class Psi0SimpleBridge:
    def __init__(
        self,
        contract,
        joints,
        inference,
        clock,
        *,
        start_tick=0,
        consume_goal=None,
    ):
        self.contract = contract
        self.joints = joints
        self.inference = inference
        self.clock = clock
        self._consume_goal = consume_goal or (lambda goal: True)
        self._lock = threading.RLock()
        self._epoch = clock() - start_tick / contract.action_frequency_hz
        self.state = BridgeState.PAUSED
        self.generation = 0
        self.tick_index = start_tick
        self.session_id = uuid.uuid4().hex
        self.request_seq = 0
        self._request = None
        self._handoff_tick = None
        self._next_request_tick = None
        self._scheduled_actions = {}
        self._scheduled_kinds = {}
        self._frozen_ticks = set()
        self._staged_first_tick = None
        self._staged_actions = None
        self.command_history_rpyh = np.array([0.0, 0.0, 0.0, 0.74], dtype=np.float32)
        self.command_history_tick = start_tick - 1
        self._latest_camera = None
        self._latest_input_error = "no valid state"
        self.last_valid_state = None
        self.last_safe_goal = None
        self._last_safe_action = None
        self._captured_hold = None
        self._hold_consumed = False
        self.fault_reason = None
        self.metrics = BridgeMetrics()

    def _scheduled_time(self, tick):
        return self._epoch + tick / self.contract.action_frequency_hz

    def status(self):
        with self._lock:
            committed = ()
            deadline = None
            if self._request is not None:
                committed = self._request.committed_global_ticks
                deadline = self._request.deadline_at
            return BridgeStatus(
                state=self.state,
                generation=self.generation,
                scheduled_ticks=tuple(sorted(self._scheduled_actions)),
                staged_first_tick=self._staged_first_tick,
                staged_count=(
                    0 if self._staged_actions is None else len(self._staged_actions)
                ),
                committed_ticks=committed,
                deadline=deadline,
                logical_request_active=self._request is not None,
            )

    def _clear_policy_locked(self):
        self._request = None
        self._handoff_tick = None
        self._next_request_tick = None
        self._scheduled_actions.clear()
        self._scheduled_kinds.clear()
        self._frozen_ticks.clear()
        self._staged_first_tick = None
        self._staged_actions = None

    def _capture_hold_locked(self):
        self._captured_hold = build_bounded_hold(
            self.clock(), self.last_valid_state, self.last_safe_goal, self.joints
        )
        self._hold_consumed = False
        return self._captured_hold

    def _transition_out_locked(self, state, reason=None):
        if self.state is BridgeState.STOPPED:
            return
        if state is BridgeState.FAULT and self.state is BridgeState.FAULT:
            return
        self.generation += 1
        self._clear_policy_locked()
        self.state = state
        self.fault_reason = reason if state is BridgeState.FAULT else None
        self._capture_hold_locked()
        if state is BridgeState.FAULT:
            self.metrics.fault_transitions += 1
            if self.metrics.first_fault_at is None:
                self.metrics.first_fault_at = self.clock()

    def pause(self):
        with self._lock:
            self._transition_out_locked(BridgeState.PAUSED)

    def enter_fault(self, reason):
        with self._lock:
            self._transition_out_locked(BridgeState.FAULT, str(reason))

    def stop(self):
        with self._lock:
            self._transition_out_locked(BridgeState.STOPPED)

    def update_inputs(self, state, camera):
        with self._lock:
            accepted, reason = accept_measured_state(
                self.last_valid_state, state, self.joints, self.clock()
            )
            self.last_valid_state = accepted
            self._latest_input_error = reason
            if (
                type(camera) is not TimedCameraFrame
                or type(camera.image) is not np.ndarray
                or camera.image.ndim != 3
                or camera.image.shape[2] != 3
                or camera.image.dtype != np.uint8
            ):
                self._latest_camera = None
                self._latest_input_error = "camera shape/dtype"
            else:
                self._latest_camera = camera

    def _input_status_locked(self):
        if self._latest_input_error is not None:
            return False, self._latest_input_error
        try:
            self._snapshot_locked()
        except ValueError as error:
            return False, str(error)
        return True, None

    def input_status(self):
        with self._lock:
            return self._input_status_locked()

    @property
    def observation_valid(self):
        return self.input_status()[0]

    @property
    def input_error(self):
        return self.input_status()[1]

    @property
    def camera_diagnostic(self):
        with self._lock:
            if self._latest_camera is None:
                return None
            _timestamp, diagnostic = sanitize_producer_timestamp(
                self._latest_camera.producer_timestamp
            )
            return self._latest_camera.producer_timestamp_diagnostic or diagnostic

    def _snapshot_locked(self):
        if self._latest_input_error is not None:
            raise ValueError(self._latest_input_error)
        return validate_synchronized_snapshot(
            self.last_valid_state, self._latest_camera, self.clock()
        )

    def _make_request_locked(self, tick, committed, reset):
        snapshot = self._snapshot_locked()
        committed = np.asarray(committed, np.float32).copy()
        if committed.shape != (self.contract.rtc_delay_steps, 36):
            raise ValueError("committed prefix must have exact (d,36) shape")
        request = RtcRequest(
            generation=self.generation,
            session_id=self.session_id,
            request_seq=self.request_seq,
            observation_tick=tick,
            history_tick=tick - 1,
            observation=build_psi0_observation(
                snapshot.state.q,
                self.joints.joint_names,
                self.command_history_rpyh,
            ).copy(),
            image=np.ascontiguousarray(snapshot.camera.image).copy(),
            committed_actions=committed,
            reset=reset,
            deadline_at=self._scheduled_time(tick + self.contract.rtc_delay_steps),
        )
        self.inference.submit(request)
        self._request = request
        self._handoff_tick = request.first_action_tick
        self.request_seq += 1
        self.metrics.requests_submitted += 1
        return request

    def _consume_locked(self, tick, action, source_kind, source_tick):
        action = np.asarray(action, np.float32).copy()
        goal = map_psi0_action_to_goal(
            action, self.joints.upper_body_joint_names, self.clock()
        )
        if self._consume_goal(goal) is not True:
            self._transition_out_locked(BridgeState.FAULT, "command sink rejected")
            return TickResult(tick, None, None, None, "none")
        self.last_safe_goal = goal
        self._last_safe_action = action
        self.command_history_rpyh = action[[28, 29, 30, 31]].copy()
        self.command_history_tick = tick
        if source_kind == "hold":
            self._hold_consumed = True
        return TickResult(tick, goal, action, source_tick, source_kind)

    def _hold_tick_locked(self, tick):
        hold = self._captured_hold
        if hold is None:
            hold = self._capture_hold_locked()
        if hold is None:
            return TickResult(tick, None, None, None, "none")
        return self._consume_locked(tick, hold.psi0_action, "hold", None)

    def activate(self):
        with self._lock:
            if self.state not in (BridgeState.PAUSED, BridgeState.FAULT):
                raise RuntimeError("bridge is not paused or faulted")
            if not self._hold_consumed:
                raise ActivationRefused("one consumed paused hold is required")
            try:
                self._snapshot_locked()
            except ValueError as error:
                raise ActivationRefused(str(error)) from error
            if self.inference.busy:
                raise ActivationRefused("physical worker busy")
            if self._captured_hold is None:
                raise ActivationRefused("no bounded hold action")
            committed_action = np.asarray(
                self._captured_hold.psi0_action, np.float32
            ).copy()
            self.generation += 1
            self._clear_policy_locked()
            self.session_id = uuid.uuid4().hex
            self.request_seq = 0
            self.fault_reason = None
            self.state = BridgeState.ACTIVE
            self._captured_hold = None
            r0 = self.tick_index
            d = self.contract.rtc_delay_steps
            committed = np.repeat(committed_action[None], d, axis=0)
            for offset, action in enumerate(committed):
                tick = r0 + offset
                self._scheduled_actions[tick] = action.copy()
                self._scheduled_kinds[tick] = "hold"
                self._frozen_ticks.add(tick)
            try:
                self._make_request_locked(r0, committed, reset=True)
            except Exception as error:
                self._transition_out_locked(BridgeState.FAULT, str(error))
                raise
            self._next_request_tick = r0 + self.contract.execution_horizon

    def handle_toggle(self):
        with self._lock:
            state = self.state
        if state is BridgeState.ACTIVE:
            self.pause()
        elif state in (BridgeState.PAUSED, BridgeState.FAULT):
            self.activate()
        else:
            raise RuntimeError("stopped bridge cannot be toggled")

    def _drain_result_locked(self):
        result = self.inference.poll()
        if result is None:
            return
        if result.generation != self.generation:
            self.metrics.discarded_old_generation_results += 1
            return
        request = self._request
        if request is None or result.request_seq != request.request_seq:
            self._transition_out_locked(BridgeState.FAULT, "unexpected RTC result")
            return
        if result.error is not None:
            self._transition_out_locked(BridgeState.FAULT, result.error)
            return
        try:
            actions = validate_rtc_result(request, result, self.contract, self.joints)
        except (TypeError, ValueError) as error:
            if "deadline" in str(error):
                self.metrics.discarded_late_results += 1
            self._transition_out_locked(BridgeState.FAULT, str(error))
            return
        self._staged_first_tick = request.first_action_tick
        self._staged_actions = np.asarray(actions, np.float32).copy()
        self._request = None
        self.metrics.results_accepted += 1

    def _install_staged_locked(self, tick):
        if self._staged_first_tick != tick or self._staged_actions is None:
            self._transition_out_locked(BridgeState.FAULT, "RTC handoff underrun")
            return False
        for offset, action in enumerate(self._staged_actions):
            global_tick = tick + offset
            self._scheduled_actions[global_tick] = action.copy()
            self._scheduled_kinds[global_tick] = "policy"
            self._frozen_ticks.discard(global_tick)
        self._staged_first_tick = None
        self._staged_actions = None
        self._handoff_tick = None
        return True

    def _freeze_successor_prefix_locked(self, request_tick):
        previous = self._last_safe_action
        if previous is None or self.command_history_tick != request_tick - 1:
            raise ValueError("command history does not end at r-1")
        committed = []
        for tick in range(request_tick, request_tick + self.contract.rtc_delay_steps):
            requested = self._scheduled_actions.get(tick)
            if requested is None:
                raise ValueError(f"missing committed action at tick {tick}")
            limited = apply_slew_limit(previous, requested, self.joints, dt=0.02)
            self._scheduled_actions[tick] = limited.copy()
            self._frozen_ticks.add(tick)
            committed.append(limited.copy())
            previous = limited
        return np.stack(committed)

    def _active_tick_locked(self, tick):
        try:
            self._snapshot_locked()
        except ValueError as error:
            self._transition_out_locked(BridgeState.FAULT, str(error))
            return self._hold_tick_locked(tick)

        self._drain_result_locked()
        if self.state is not BridgeState.ACTIVE:
            return self._hold_tick_locked(tick)

        if self._handoff_tick == tick:
            if not self._install_staged_locked(tick):
                return self._hold_tick_locked(tick)

        if self._next_request_tick == tick:
            if self.inference.busy or self._request is not None:
                self._transition_out_locked(
                    BridgeState.FAULT, "physical worker busy at successor"
                )
                return self._hold_tick_locked(tick)
            try:
                committed = self._freeze_successor_prefix_locked(tick)
                self._make_request_locked(tick, committed, reset=False)
            except Exception as error:
                self._transition_out_locked(BridgeState.FAULT, str(error))
                return self._hold_tick_locked(tick)
            self._next_request_tick = tick + self.contract.execution_horizon

        requested = self._scheduled_actions.pop(tick, None)
        source_kind = self._scheduled_kinds.pop(tick, None)
        if requested is None:
            self._transition_out_locked(BridgeState.FAULT, "scheduled action underrun")
            return self._hold_tick_locked(tick)
        if tick in self._frozen_ticks:
            action = requested
            self._frozen_ticks.remove(tick)
        else:
            action = apply_slew_limit(
                self._last_safe_action, requested, self.joints, dt=0.02
            )
        if source_kind not in {"hold", "policy"}:
            self._transition_out_locked(BridgeState.FAULT, "scheduled kind missing")
            return self._hold_tick_locked(tick)
        source_tick = None if source_kind == "hold" else tick
        return self._consume_locked(tick, action, source_kind, source_tick)

    def tick(self):
        with self._lock:
            tick = self.tick_index
            if self.state is BridgeState.STOPPED:
                return TickResult(tick, None, None, None, "none")
            if self.state is BridgeState.ACTIVE:
                result = self._active_tick_locked(tick)
            else:
                self._drain_result_locked()
                result = self._hold_tick_locked(tick)
            self.tick_index += 1
            return result

    def build_bounded_shutdown_hold(self):
        with self._lock:
            result = self._captured_hold
            if result is None:
                result = self._capture_hold_locked()
            return None if result is None else result.goal
