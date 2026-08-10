import argparse
import base64
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Callable

import msgpack
import msgpack_numpy as mnp
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import ByteMultiArray
from std_srvs.srv import Trigger
import zmq

import decoupled_wbc
from decoupled_wbc.control.main.model_contract import (
    build_model_contract,
    digest_model_contract,
)
from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from decoupled_wbc.control.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from decoupled_wbc.control.sensor.sensor_server import ImageMessageSchema
from simple.baselines.client import HttpActionClient
from simple.deploy.psi0_simple_bridge import (
    BridgeMode,
    JointContract,
    PolicyContract,
    TimedCameraFrame,
    TimedRobotState,
    sanitize_producer_timestamp,
)


CONTROL_GOAL_TOPIC = "ControlPolicy/upper_body_pose"


class PreflightError(RuntimeError):
    pass


def _tcp_port(value):
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be in [1,65535]")
    return port


def build_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(BridgeMode.SHADOW.value, BridgeMode.SIM_CONTROL.value),
    )
    parser.add_argument("--server-host", required=True)
    parser.add_argument("--server-port", required=True, type=_tcp_port)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--policy-contract")
    parser.add_argument("--camera-host", required=True)
    parser.add_argument("--camera-port", required=True, type=_tcp_port)
    parser.add_argument("--camera-source-key", required=True)
    parser.add_argument(
        "--camera-color-order",
        choices=("rgb", "bgr"),
        default="rgb",
    )
    parser.add_argument("--ros-domain-id", type=int, default=42)
    parser.add_argument("--unitree-domain-id", type=int, default=42)
    parser.add_argument("--metrics-jsonl", required=True)
    return parser


def build_policy_client(host, port):
    if type(host) is not str or not host:
        raise ValueError("policy host must be a non-empty string")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("policy port must be an integer in [1,65535]")
    return HttpActionClient(host, port, timeout=5.0)


POLICY_FIELD_MAP = (
    ("schema", "schema"),
    ("test_only", "test_only"),
    ("checkpoint_sha256", "checkpoint_sha256"),
    ("dataset_manifest_sha256", "dataset_manifest_sha256"),
    ("raw_episode_sha256", "raw_episode_sha256"),
    ("processed_episode_sha256", "processed_episode_sha256"),
    ("source_episode_index", "source_episode_index"),
    ("processed_episode_index", "processed_episode_index"),
    ("converter_commit", "converter_commit"),
    ("server_commit", "server_commit"),
    ("converter_layout", "converter_layout"),
    ("observation_dim", "observation_dim"),
    ("action_dim", "action_dim"),
    ("action_frequency_hz", "action_frequency_hz"),
    ("prediction_horizon", "prediction_horizon"),
    ("execution_horizon", "execution_horizon"),
    ("rtc_delay_steps", "rtc_delay_steps"),
    ("rtc_training_max_delay", "rtc_training_max_delay"),
    ("rtc_enabled", "rtc_enabled"),
    ("rtc_endpoint", "endpoint"),
    ("request_semantics", "request_semantics"),
    ("response_semantics", "response_semantics"),
    ("image_key", "image_key"),
    ("camera_color_order", "camera_color_order"),
)


@dataclass(frozen=True)
class PolicyPreflightResult:
    policy_certified: bool
    mismatched_fields: tuple[str, ...]


def compare_policy_contracts(local, server, mode, wbc_env_type):
    mismatches = tuple(
        wire_name
        for wire_name, attribute in POLICY_FIELD_MAP
        if getattr(local, attribute) != getattr(server, attribute)
    )
    if (local.test_only or server.test_only) and wbc_env_type != "sim":
        raise PreflightError("test-only policy contract requires sim WBC")
    if mismatches and mode is not BridgeMode.SHADOW:
        raise PreflightError("policy contract mismatch: " + ",".join(mismatches))
    certified = mode is BridgeMode.SIM_CONTROL and not mismatches
    return PolicyPreflightResult(certified, mismatches)


class BoundedWbcConfigClient:
    def __init__(self, *, clock, wait_once, request_once=None, destroy):
        self._clock = clock
        self._wait_once = wait_once
        self._request_once = request_once or (
            lambda timeout: (_ for _ in ()).throw(RuntimeError("no requester"))
        )
        self._destroy = destroy

    def get_config(self, timeout_s=3.0):
        deadline = self._clock() + timeout_s
        try:
            while True:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise TimeoutError(
                        f"WBC config service timed out after {timeout_s:.1f}s"
                    )
                if self._wait_once(min(0.05, remaining)):
                    break
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise TimeoutError(
                    f"WBC config service timed out after {timeout_s:.1f}s"
                )
            payload = self._request_once(remaining)
            if type(payload) is not dict:
                raise PreflightError("WBC config response must be a dictionary")
            return payload
        finally:
            self._destroy()


def create_ros_wbc_config_client(service_name, domain_id):
    if os.environ.get("ROS_DOMAIN_ID") != str(domain_id):
        raise PreflightError("ROS_DOMAIN_ID does not match requested domain")
    context = rclpy.context.Context()
    context.init()
    node = rclpy.create_node("psi0_wbc_preflight", context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    client = node.create_client(Trigger, service_name)

    def request_once(timeout_s):
        future = client.call_async(Trigger.Request())
        executor.spin_until_future_complete(future, timeout_sec=timeout_s)
        if not future.done() or future.result() is None:
            raise TimeoutError("WBC config response timed out")
        response = future.result()
        if not response.success:
            raise PreflightError(f"WBC config service failed: {response.message}")
        packed = base64.b64decode(response.message.encode("ascii"), validate=True)
        payload = msgpack.unpackb(packed, object_hook=mnp.decode, raw=False)
        if type(payload) is not dict:
            raise PreflightError("WBC config response must be a dictionary")
        return payload

    def destroy():
        executor.remove_node(node)
        node.destroy_client(client)
        node.destroy_node()
        executor.shutdown()
        context.shutdown()

    return BoundedWbcConfigClient(
        clock=time.monotonic,
        wait_once=lambda seconds: client.wait_for_service(timeout_sec=seconds),
        request_once=request_once,
        destroy=destroy,
    )


@dataclass(frozen=True)
class ValidatedWbc:
    joint_contract: JointContract


EXPECTED_WBC_FIELDS = {
    "env_type": "sim",
    "interface": "lo",
    "simulator": "mujoco",
    "messaging_backend": "ros2",
    "control_frequency": 50,
    "enable_waist": True,
    "with_hands": True,
    "wbc_version": "gear_wbc",
    "wbc_policy_class": "G1DecoupledWholeBodyPolicy",
    "wbc_model_path": (
        "policy/GR00T-WholeBodyControl-Balance.onnx,"
        "policy/GR00T-WholeBodyControl-Walk.onnx"
    ),
}


def _first_difference(actual, expected, path="model_contract"):
    if type(actual) is not type(expected):
        return f"{path} type"
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            return f"{path} keys"
        for key in expected:
            difference = _first_difference(actual[key], expected[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(actual) != len(expected):
            return f"{path} length"
        for index, expected_value in enumerate(expected):
            difference = _first_difference(
                actual[index], expected_value, f"{path}[{index}]"
            )
            if difference is not None:
                return difference
        return None
    return None if actual == expected else path


def _extract_attested_joint_contract(actual):
    if type(actual) is not dict:
        raise PreflightError("WBC payload must be a dictionary")
    contract = actual.get("model_contract")
    transmitted = actual.get("model_contract_sha256")
    if type(contract) is not dict or type(transmitted) is not str:
        raise PreflightError("model contract digest fields are missing")
    try:
        recomputed = digest_model_contract(contract)
    except (TypeError, ValueError) as error:
        raise PreflightError(f"model contract digest input: {error}") from error
    if transmitted != recomputed:
        raise PreflightError("model contract digest mismatch")
    if (
        set(contract)
        != {
            "schema",
            "git",
            "robot_model",
            "urdf",
            "onnx_models",
        }
        or contract.get("schema") != "decoupled_wbc.g1-model-contract.v1"
    ):
        raise PreflightError("connected WBC model contract schema")
    git = contract.get("git")
    if (
        type(git) is not dict
        or set(git) != {"commit", "working_tree_clean"}
        or type(git.get("commit")) is not str
        or re.fullmatch(r"[0-9a-f]{40}", git["commit"]) is None
        or type(git.get("working_tree_clean")) is not bool
    ):
        raise PreflightError("connected WBC Git identity schema")
    urdf = contract.get("urdf")
    if (
        type(urdf) is not dict
        or set(urdf) != {"relative_path", "sha256"}
        or type(urdf.get("relative_path")) is not str
        or type(urdf.get("sha256")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", urdf["sha256"]) is None
    ):
        raise PreflightError("connected WBC URDF identity schema")
    onnx_models = contract.get("onnx_models")
    if type(onnx_models) is not list or len(onnx_models) != 2:
        raise PreflightError("connected WBC onnx identity schema")
    for model_entry, role in zip(onnx_models, ("balance", "walk"), strict=True):
        if (
            type(model_entry) is not dict
            or set(model_entry)
            != {"role", "relative_path", "sha256", "input", "output"}
            or model_entry.get("role") != role
            or type(model_entry.get("relative_path")) is not str
            or type(model_entry.get("sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", model_entry["sha256"]) is None
        ):
            raise PreflightError("connected WBC onnx identity schema")
        for direction, size in (("input", 516), ("output", 15)):
            tensor = model_entry.get(direction)
            if (
                type(tensor) is not dict
                or set(tensor) != {"name", "shape", "feature_size"}
                or type(tensor.get("name")) is not str
                or type(tensor.get("shape")) is not list
                or not tensor["shape"]
                or any(type(dim) not in (str, int) for dim in tensor["shape"])
                or type(tensor.get("feature_size")) is not int
                or tensor["feature_size"] != size
                or tensor["shape"][-1] != size
            ):
                raise PreflightError("connected WBC onnx tensor schema")
    try:
        model = contract["robot_model"]
        if (
            type(model) is not dict
            or set(model)
            != {
                "name",
                "joint_names",
                "lower_position_limits",
                "upper_position_limits",
                "upper_body_joint_names",
            }
            or type(model.get("name")) is not str
            or any(
                type(model.get(key)) is not list
                for key in (
                    "joint_names",
                    "lower_position_limits",
                    "upper_position_limits",
                    "upper_body_joint_names",
                )
            )
        ):
            raise TypeError("robot model identity schema")
        names = tuple(model["joint_names"])
        upper = tuple(model["upper_body_joint_names"])
        lower = np.asarray(model["lower_position_limits"], np.float32)
        high = np.asarray(model["upper_position_limits"], np.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise PreflightError(f"connected WBC joint contract: {error}") from error
    if (
        len(names) != 43
        or len(set(names)) != 43
        or len(upper) != 31
        or len(set(upper)) != 31
        or any(type(name) is not str for name in names)
        or any(type(name) is not str for name in upper)
        or any(name not in names for name in upper)
    ):
        raise PreflightError("connected WBC joint_names/upper_body dimensions")
    if (
        lower.shape != (43,)
        or high.shape != (43,)
        or not np.isfinite(lower).all()
        or not np.isfinite(high).all()
        or not np.all(lower < high)
    ):
        raise PreflightError("connected WBC effective limits")
    return ValidatedWbc(JointContract(names, upper, lower, high)), contract


def validate_connected_wbc(
    *, actual, expected_contract, expected_gitlink_sha, required_domain_id
):
    validated, contract = _extract_attested_joint_contract(actual)
    required = {**EXPECTED_WBC_FIELDS, "domain_id": required_domain_id}
    for key, expected in required.items():
        if key not in actual or type(actual[key]) is not type(expected):
            raise PreflightError(f"WBC field {key} missing or wrong type")
        if actual[key] != expected:
            raise PreflightError(f"WBC field {key} mismatch")
    if contract.get("git", {}).get("commit") != expected_gitlink_sha:
        raise PreflightError("model_contract.git.commit differs from root gitlink")
    difference = _first_difference(contract, expected_contract)
    if difference is not None:
        raise PreflightError(f"connected WBC {difference} mismatch")
    return validated


def inspect_shadow_wbc(
    *, actual, expected_contract, expected_gitlink_sha, required_domain_id
):
    validated, contract = _extract_attested_joint_contract(actual)
    mismatches = []
    required = {**EXPECTED_WBC_FIELDS, "domain_id": required_domain_id}
    for key, expected in required.items():
        if type(actual.get(key)) is not type(expected) or actual.get(key) != expected:
            mismatches.append(f"config.{key}")
    difference = _first_difference(contract, expected_contract)
    if difference is not None:
        mismatches.append(difference)
    elif contract.get("git", {}).get("commit") != expected_gitlink_sha:
        mismatches.append("model_contract.git.commit")
    return validated, tuple(mismatches)


def validate_then_create_publisher(
    payload,
    expected_contract,
    expected_gitlink_sha,
    required_domain_id,
    *,
    publisher_factory,
):
    validated = validate_connected_wbc(
        actual=payload,
        expected_contract=expected_contract,
        expected_gitlink_sha=expected_gitlink_sha,
        required_domain_id=required_domain_id,
    )
    return validated, publisher_factory()


class RosRuntime:
    def __init__(self, domain_id):
        if os.environ.get("ROS_DOMAIN_ID") != str(domain_id):
            raise PreflightError("ROS_DOMAIN_ID does not match requested domain")
        self.context = rclpy.context.Context()
        self.context.init()
        self.node = rclpy.create_node("psi0_simple_bridge", context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(
            target=self.executor.spin,
            name="psi0-ros-executor",
            daemon=True,
        )
        self.thread.start()

    def counts(self, topic):
        return self.node.count_publishers(topic), self.node.count_subscribers(topic)

    def close(self, timeout_s):
        deadline = time.monotonic() + max(0.0, timeout_s)
        self.executor.shutdown(timeout_sec=max(0.0, deadline - time.monotonic()))
        self.thread.join(max(0.0, deadline - time.monotonic()))
        if self.thread.is_alive():
            raise RuntimeError("ROS executor did not stop within shutdown budget")
        self.node.destroy_node()
        self.context.shutdown()


class RosGoalPublisher:
    def __init__(self, runtime, topic):
        self._runtime = runtime
        self._publisher = runtime.node.create_publisher(ByteMultiArray, topic, 1)
        self.closed = False
        self.publish_attempts = 0

    def publish(self, goal):
        if self.closed:
            raise RuntimeError("goal publisher is closed")
        payload = {
            "target_upper_body_pose": goal.target_upper_body_pose.copy(),
            "base_height_command": goal.base_height_command.copy(),
            "navigate_cmd": goal.navigate_cmd.copy(),
            "timestamp": float(goal.timestamp),
            "target_time": float(goal.target_time),
        }
        if set(payload) != {
            "target_upper_body_pose",
            "base_height_command",
            "navigate_cmd",
            "timestamp",
            "target_time",
        }:
            raise AssertionError("goal payload key drift")
        packed = msgpack.packb(payload, default=mnp.encode, use_bin_type=True)
        message = ByteMultiArray()
        message.data = list(packed)
        self.publish_attempts += 1
        self._publisher.publish(message)
        return True

    def close(self, timeout_s=0.0):
        if not self.closed:
            self.closed = True
            self._runtime.node.destroy_publisher(self._publisher)


def validate_goal_ownership_preflight(mode, graph):
    before = tuple(graph.counts(CONTROL_GOAL_TOPIC))
    if len(before) != 2 or any(type(value) is not int or value < 0 for value in before):
        raise PreflightError("goal publisher/subscription counts are invalid")
    if mode is BridgeMode.SIM_CONTROL and before != (0, 1):
        raise PreflightError(
            f"expected 0 publishers/1 subscription before bridge, got {before}"
        )
    return before


def establish_goal_ownership(mode, graph, publisher_factory):
    validate_goal_ownership_preflight(mode, graph)
    if mode is BridgeMode.SHADOW:
        return None
    publisher = publisher_factory()
    after = graph.counts(CONTROL_GOAL_TOPIC)
    if after != (1, 1):
        close = getattr(publisher, "close", None)
        if close is not None:
            close()
        raise PreflightError(
            f"expected 1 publisher/1 subscription after bridge, got {after}"
        )
    return publisher


class GoalOwnershipGuard:
    def __init__(self, mode, graph, expected_counts):
        self.mode = BridgeMode(mode)
        self.graph = graph
        self.expected_counts = tuple(expected_counts)

    def check(self):
        if self.mode is not BridgeMode.SHADOW:
            return
        current = tuple(self.graph.counts(CONTROL_GOAL_TOPIC))
        if current != self.expected_counts:
            raise PreflightError(
                "shadow goal publisher counts changed: "
                f"{self.expected_counts} -> {current}"
            )

    def close(self, timeout_s=0.0):
        self.check()


@dataclass(frozen=True)
class RuntimeAdapters:
    state_source: "RosStateSource"
    camera_reader: "ComposedCameraReader"
    goal_publisher: RosGoalPublisher | None
    ros_runtime: RosRuntime | None = None


@dataclass(frozen=True)
class RuntimeDependencyFactories:
    state_source: Callable[[], object]
    camera_reader: Callable[[], object]
    graph: Callable[[], object]


def build_runtime_adapters(
    *, mode, preflight_result, publisher_factory, test_dependencies
):
    if preflight_result.publisher_required != (mode is BridgeMode.SIM_CONTROL):
        raise PreflightError("preflight publisher requirement does not match mode")
    graph = test_dependencies.graph()
    if mode is BridgeMode.SHADOW and tuple(graph.counts(CONTROL_GOAL_TOPIC)) != tuple(
        preflight_result.goal_counts_at_preflight
    ):
        raise PreflightError("shadow goal counts changed after preflight")
    publisher = establish_goal_ownership(mode, graph, publisher_factory)
    state_source = None
    camera_reader = None
    try:
        state_source = test_dependencies.state_source()
        camera_reader = test_dependencies.camera_reader()
        return RuntimeAdapters(state_source, camera_reader, publisher)
    except Exception:
        for adapter in (camera_reader, state_source, publisher):
            close = getattr(adapter, "close", None)
            if close is not None:
                close()
        raise


def unpack_dict_message(message):
    packed = bytes(message.data)
    payload = msgpack.unpackb(packed, object_hook=mnp.decode, raw=False)
    if type(payload) is not dict:
        raise ValueError("ROS state payload must be a dictionary")
    return payload


class RosStateSource:
    def __init__(self, runtime, topic, clock=time.monotonic):
        self._runtime = runtime
        self._clock = clock
        self._lock = threading.Lock()
        self._latest = None
        self._error = None
        self._subscription = runtime.node.create_subscription(
            ByteMultiArray, topic, self._callback, 1
        )

    def _callback(self, message):
        try:
            payload = unpack_dict_message(message)
            q = np.asarray(payload["q"], np.float32)
            sample = TimedRobotState(q.copy(), self._clock())
            error = None
        except Exception as caught:
            sample = None
            error = f"state decode: {caught}"
        with self._lock:
            self._latest = sample
            self._error = error

    def poll(self):
        with self._lock:
            if self._error is not None:
                raise ValueError(self._error)
            sample, self._latest = self._latest, None
            return sample

    def close(self, timeout_s=0.0):
        self._runtime.node.destroy_subscription(self._subscription)


def decode_camera_message(serialized, *, key, color_order, received_at):
    schema = ImageMessageSchema.deserialize(serialized)
    if key not in schema.images:
        available = ",".join(sorted(schema.images))
        raise KeyError(f"{key}; available camera keys: {available}")
    image = np.asarray(schema.images[key])
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("camera image must be HxWx3 uint8")
    if color_order == "bgr":
        image = image[..., ::-1]
    elif color_order != "rgb":
        raise ValueError("camera color order must be rgb or bgr")
    if type(schema.timestamps) is dict:
        raw_producer_timestamp = schema.timestamps.get(key)
        producer_timestamp, diagnostic = sanitize_producer_timestamp(
            raw_producer_timestamp
        )
    else:
        producer_timestamp = None
        diagnostic = "producer timestamp mapping ignored: wrong type"
    return TimedCameraFrame(
        np.ascontiguousarray(image),
        float(received_at),
        producer_timestamp,
        diagnostic,
    )


class ComposedCameraReader:
    def __init__(self, host, port, key, color_order, clock=time.monotonic):
        if host in {"0.0.0.0", "*"}:
            raise ValueError("camera client requires a concrete host")
        self._endpoint = f"tcp://{host}:{port}"
        self._key = key
        self._color_order = color_order
        self._clock = clock
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._condition = threading.Condition()
        self._latest = None
        self._error = None
        self._thread = threading.Thread(
            target=self._run,
            name="psi0-camera-reader",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(0.5):
            raise TimeoutError("camera reader did not initialize")

    def _run(self):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        try:
            socket.setsockopt(zmq.SUBSCRIBE, b"")
            socket.setsockopt(zmq.CONFLATE, 1)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(self._endpoint)
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            self._ready.set()
            while not self._stop.is_set():
                if socket not in dict(poller.poll(100)):
                    continue
                packed = socket.recv(zmq.NOBLOCK)
                serialized = msgpack.unpackb(packed, object_hook=mnp.decode, raw=False)
                frame = decode_camera_message(
                    serialized,
                    key=self._key,
                    color_order=self._color_order,
                    received_at=self._clock(),
                )
                with self._condition:
                    self._latest = frame
                    self._condition.notify_all()
        except Exception as caught:
            with self._condition:
                self._error = caught
                self._condition.notify_all()
        finally:
            socket.close(linger=0)
            context.term()
            self._ready.set()

    def poll(self):
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"camera reader failed: {self._error}")
            frame, self._latest = self._latest, None
            return frame

    def wait_for_frame(self, timeout_s):
        deadline = self._clock() + timeout_s
        with self._condition:
            while self._latest is None and self._error is None:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise TimeoutError("camera frame timed out")
                self._condition.wait(remaining)
            if self._error is not None:
                raise RuntimeError(f"camera reader failed: {self._error}")
            frame, self._latest = self._latest, None
            return frame

    def close(self, timeout_s=0.5):
        self._stop.set()
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise RuntimeError("camera reader did not stop within timeout")


@dataclass(frozen=True)
class PreflightResult:
    policy_certified: bool
    policy_mismatched_fields: tuple[str, ...]
    wbc_mismatched_fields: tuple[str, ...]
    joint_contract: JointContract
    publisher_required: bool
    goal_counts_at_preflight: tuple[int, int]
    runtime_policy_contract: PolicyContract | None


def run_preflight(
    *,
    mode,
    local_policy,
    server_policy,
    wbc_payload,
    graph,
    expected_model_contract,
    expected_gitlink_sha,
    required_domain_id=42,
):
    mode = BridgeMode(mode)
    if mode is BridgeMode.SHADOW:
        validated_wbc, wbc_mismatches = inspect_shadow_wbc(
            actual=wbc_payload,
            expected_contract=expected_model_contract,
            expected_gitlink_sha=expected_gitlink_sha,
            required_domain_id=required_domain_id,
        )
    else:
        validated_wbc = validate_connected_wbc(
            actual=wbc_payload,
            expected_contract=expected_model_contract,
            expected_gitlink_sha=expected_gitlink_sha,
            required_domain_id=required_domain_id,
        )
        wbc_mismatches = ()
    parsed = {}
    policy_errors = []
    for label, payload in (("local", local_policy), ("server", server_policy)):
        if payload is None:
            policy_errors.append(f"{label} policy contract unavailable")
            continue
        try:
            parsed[label] = PolicyContract.from_dict(payload)
        except (TypeError, ValueError) as error:
            policy_errors.append(f"{label} policy contract: {error}")
    if policy_errors:
        if mode is not BridgeMode.SHADOW:
            raise PreflightError(policy_errors[0])
        policy_result = PolicyPreflightResult(False, tuple(policy_errors))
    else:
        try:
            policy_result = compare_policy_contracts(
                parsed["local"],
                parsed["server"],
                mode,
                wbc_payload.get("env_type"),
            )
        except PreflightError as error:
            if mode is not BridgeMode.SHADOW:
                raise
            policy_result = PolicyPreflightResult(False, (str(error),))
    runtime_contract = parsed.get("local") or parsed.get("server")
    goal_counts = validate_goal_ownership_preflight(mode, graph)
    return PreflightResult(
        policy_certified=(
            policy_result.policy_certified if mode is BridgeMode.SIM_CONTROL else False
        ),
        policy_mismatched_fields=policy_result.mismatched_fields,
        wbc_mismatched_fields=wbc_mismatches,
        joint_contract=validated_wbc.joint_contract,
        publisher_required=(mode is BridgeMode.SIM_CONTROL),
        goal_counts_at_preflight=goal_counts,
        runtime_policy_contract=runtime_contract,
    )


@dataclass(frozen=True)
class LocalWbcIdentity:
    root_gitlink_sha: str
    model_contract: dict[str, object]


def _git_stdout(arguments, cwd):
    return subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_local_wbc_identity(repository_root):
    root = Path(repository_root).resolve()
    nested = root / "third_party/decoupled_wbc"
    gitlink = _git_stdout(["git", "rev-parse", "HEAD:third_party/decoupled_wbc"], root)
    nested_head = _git_stdout(["git", "rev-parse", "HEAD"], nested)
    nested_status = _git_stdout(
        ["git", "status", "--porcelain", "--untracked-files=all"], nested
    )
    if nested_head != gitlink:
        raise PreflightError("nested HEAD differs from root gitlink")
    if nested_status:
        raise PreflightError("nested WBC checkout is not clean")
    config = ControlLoopConfig(
        interface="sim",
        simulator="mujoco",
        messaging_backend="ros2",
        enable_waist=True,
        with_hands=True,
        domain_id=42,
    )
    model = instantiate_g1_robot_model(
        waist_location="lower_and_upper_body",
        high_elbow_pose=config.high_elbow_pose,
    )
    contract = build_model_contract(
        robot_model=model,
        config=config,
        repository_root=Path(decoupled_wbc.__file__).resolve().parent,
    )
    if contract["git"] != {"commit": gitlink, "working_tree_clean": True}:
        raise PreflightError("local WBC model contract Git identity")
    return LocalWbcIdentity(gitlink, contract)
