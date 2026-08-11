import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import pty
import secrets
import signal
import socket
import subprocess
import sys
import termios
import threading
import time
from types import MappingProxyType

import numpy as np
import requests


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))


class SmokeSafetyError(RuntimeError):
    pass


CONTROL_GOAL_TOPIC = "ControlPolicy/upper_body_pose"
CONNECTION_EVIDENCE_KEYS = {
    "component",
    "transport",
    "endpoint",
    "real_interface",
    "observed_at",
}
CONNECTION_COMPONENTS = {"wbc", "camera", "policy"}


def count_real_interface_connections(evidence):
    if type(evidence) is not list or not 1 <= len(evidence) <= 3:
        raise SmokeSafetyError("connection evidence record count")
    components = [record.get("component") for record in evidence]
    if len(set(components)) != len(components) or not set(components) <= (
        CONNECTION_COMPONENTS
    ):
        raise SmokeSafetyError("connection evidence component set")
    for record in evidence:
        if type(record) is not dict or set(record) != CONNECTION_EVIDENCE_KEYS:
            raise SmokeSafetyError("connection evidence record schema")
        if (
            type(record["transport"]) is not str
            or type(record["endpoint"]) is not str
            or type(record["real_interface"]) is not bool
            or type(record["observed_at"]) is not float
            or not np.isfinite(record["observed_at"])
        ):
            raise SmokeSafetyError("connection evidence record types")
    return sum(record["real_interface"] for record in evidence)


@dataclass(frozen=True)
class SmokeConfig:
    duration_s: float
    ros_domain_id: int
    unitree_domain_id: int
    wbc_interface: str
    camera_host: str
    camera_port: int
    policy_host: str
    policy_port: int
    control_token: str
    policy_ready_json: str
    camera_ready_json: str
    bridge_metrics_jsonl: str
    smoke_report_json: str
    output_dir: str


@dataclass(frozen=True)
class ChildSpec:
    name: str
    argv: tuple[str, ...]
    env: object
    use_pty: bool = False
    ready_json: str | None = None


@dataclass(frozen=True)
class LaunchPlan:
    config: SmokeConfig
    children: tuple[ChildSpec, ...]

    def child(self, name):
        matches = [child for child in self.children if child.name == name]
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]


def _inside_output(path, output_dir):
    candidate = Path(path).resolve()
    root = Path(output_dir).resolve()
    return candidate != root and candidate.is_relative_to(root)


def build_launch_plan(config):
    if type(config) is not SmokeConfig:
        raise SmokeSafetyError("SmokeConfig is required")
    if config.duration_s != 15.0:
        raise SmokeSafetyError("certified smoke duration is exactly 15 seconds")
    if config.ros_domain_id != 42 or config.unitree_domain_id != 42:
        raise SmokeSafetyError("isolated ROS and Unitree domains must both be 42")
    if config.wbc_interface != "lo":
        raise SmokeSafetyError("smoke network interface must be loopback")
    if config.camera_host != "127.0.0.1" or config.policy_host != "127.0.0.1":
        raise SmokeSafetyError("fake services must be loopback-only")
    if config.camera_port != 15555 or config.policy_port != 22086:
        raise SmokeSafetyError("certified smoke ports must be 15555/22086")
    if type(config.control_token) is not str or not config.control_token:
        raise SmokeSafetyError("fake-policy control token is required")
    output_paths = (
        config.policy_ready_json,
        config.camera_ready_json,
        config.bridge_metrics_jsonl,
        config.smoke_report_json,
        *(
            str(Path(config.output_dir) / f"psi0-smoke-{name}.log")
            for name in ("wbc", "camera", "policy", "bridge")
        ),
    )
    if any(not _inside_output(path, config.output_dir) for path in output_paths):
        raise SmokeSafetyError("all artifacts must be inside output_dir")
    existing = [path for path in output_paths if Path(path).exists()]
    if existing:
        raise SmokeSafetyError(f"smoke artifact already exists: {existing[0]}")

    environment = dict(os.environ)
    parent_pythonpath = environment.get("PYTHONPATH")
    required_pythonpath = "third_party/decoupled_wbc"
    environment.update(
        {
            "ROS_DOMAIN_ID": "42",
            "UNITREE_DOMAIN_ID": "42",
            "PYTHONPATH": (
                required_pythonpath
                if not parent_pythonpath
                else os.pathsep.join((required_pythonpath, parent_pythonpath))
            ),
        }
    )
    env = MappingProxyType(environment)
    wbc = ChildSpec(
        "wbc",
        (
            "uv",
            "run",
            "--group",
            "sonic",
            "python",
            "-m",
            "decoupled_wbc.control.main.teleop.run_g1_control_loop",
            "--interface",
            "sim",
            "--simulator",
            "mujoco",
            "--messaging-backend",
            "ros2",
            "--enable-waist",
            "--with-hands",
            "--domain-id",
            "42",
            "--no-enable-onscreen",
            "--no-enable-offscreen",
        ),
        env,
    )
    camera = ChildSpec(
        "camera",
        (
            "uv",
            "run",
            "--group",
            "sonic",
            "python",
            "scripts/tests/fake_composed_camera_server.py",
            "--host",
            config.camera_host,
            "--port",
            str(config.camera_port),
            "--ready-json",
            config.camera_ready_json,
        ),
        env,
        ready_json=config.camera_ready_json,
    )
    policy = ChildSpec(
        "policy",
        (
            "uv",
            "run",
            "--group",
            "sonic",
            "python",
            "scripts/tests/fake_psi0_rtc_server.py",
            "--host",
            config.policy_host,
            "--port",
            str(config.policy_port),
            "--contract",
            "scripts/tests/fixtures/psi0_policy_contract_test_v2.json",
            "--normal-latency-s",
            "0.05",
            "--control-token",
            config.control_token,
            "--ready-json",
            config.policy_ready_json,
        ),
        env,
        ready_json=config.policy_ready_json,
    )
    bridge = ChildSpec(
        "bridge",
        (
            "uv",
            "run",
            "--group",
            "sonic",
            "python",
            "scripts/psi0_simple_real_bridge.py",
            "--mode",
            "sim-control",
            "--server-host",
            config.policy_host,
            "--server-port",
            str(config.policy_port),
            "--instruction",
            "smoke test hold",
            "--policy-contract",
            "scripts/tests/fixtures/psi0_policy_contract_test_v2.json",
            "--camera-host",
            config.camera_host,
            "--camera-port",
            str(config.camera_port),
            "--camera-source-key",
            "rgb_head_stereo_left",
            "--camera-color-order",
            "rgb",
            "--ros-domain-id",
            "42",
            "--unitree-domain-id",
            "42",
            "--metrics-jsonl",
            config.bridge_metrics_jsonl,
        ),
        env,
        use_pty=True,
    )
    return LaunchPlan(config, (wbc, camera, policy, bridge))


def arm_next_policy_delay(host, port, token, latency_s, *, session=requests):
    if host != "127.0.0.1" or type(token) is not str or not token:
        raise SmokeSafetyError("fake-policy control must be authenticated loopback")
    response = session.post(
        f"http://{host}:{port}/test-control/arm-next-delay",
        json={"token": token, "latency_s": latency_s},
        timeout=0.5,
    )
    if response.status_code != 202:
        raise SmokeSafetyError(
            f"fake-policy control returned status {response.status_code}"
        )
    response.raise_for_status()
    payload = response.json()
    if type(payload) is not dict or set(payload) != {
        "armed_request_seq",
        "latency_s",
    }:
        raise SmokeSafetyError("malformed fake-policy control response")
    if type(payload["armed_request_seq"]) is not int:
        raise SmokeSafetyError("malformed armed request sequence")
    if type(payload["latency_s"]) is not float or payload["latency_s"] != latency_s:
        raise SmokeSafetyError("fake-policy delay acknowledgement mismatch")
    return payload["armed_request_seq"]


@dataclass
class ChildRecord:
    pid: int
    pgid: int
    name: str
    argv: tuple[str, ...]
    started_at: float
    process: object | None = None


def _process_group_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


class OwnedChildren:
    def __init__(
        self,
        killpg=os.killpg,
        waitpid=os.waitpid,
        clock=time.monotonic,
        sleep=time.sleep,
        group_alive=_process_group_alive,
    ):
        self._killpg = killpg
        self._waitpid = waitpid
        self._clock = clock
        self._sleep = sleep
        self._group_alive = group_alive
        self.records = []
        self._closed = False
        self._closing = False

    def record(self, *, pid, pgid, name, argv, started_at, process=None):
        if (
            self._closed
            or self._closing
            or any(record.pid == pid for record in self.records)
        ):
            raise RuntimeError("invalid child ownership record")
        record = ChildRecord(pid, pgid, name, tuple(argv), float(started_at), process)
        self.records.append(record)
        return record

    def process_group(self, name):
        matches = [record.pgid for record in self.records if record.name == name]
        if len(matches) != 1:
            raise SmokeSafetyError(f"expected one owned child named {name!r}")
        return matches[0]

    def _poll(self, record, errors):
        try:
            return record.process.poll()
        except Exception as error:
            errors.append(f"poll {record.name}({record.pid}): {error}")
            return None

    def _live(self, records, errors):
        live = []
        for record in records:
            leader_alive = self._poll(record, errors) is None
            try:
                group_alive = self._group_alive(record.pgid)
            except Exception as error:
                errors.append(f"probe group {record.name}({record.pgid}): {error}")
                group_alive = True
            if leader_alive or group_alive:
                live.append(record)
        return live

    def _signal(self, records, sig, errors):
        for record in records:
            try:
                self._killpg(record.pgid, sig)
            except ProcessLookupError:
                continue
            except Exception as error:
                errors.append(f"signal {record.name}({record.pgid}) {sig}: {error}")

    def _wait_live_until(self, records, deadline, errors):
        while True:
            live = self._live(records, errors)
            if not live or self._clock() >= deadline:
                return live
            self._sleep(min(0.01, deadline - self._clock()))

    def live_pids(self):
        errors = []
        real = [record for record in self.records if record.process is not None]
        return [record.pid for record in self._live(real, errors)]

    def exit_codes(self):
        return {
            record.name: record.process.poll()
            for record in self.records
            if record.process is not None
        }

    def close(self, deadline=None):
        if self._closed:
            return
        if self._closing:
            raise SmokeSafetyError("child cleanup is already in progress")
        self._closing = True
        errors = []
        injected = [
            record for record in reversed(self.records) if record.process is None
        ]
        real = [
            record for record in reversed(self.records) if record.process is not None
        ]
        live = []
        injected_ok = True
        try:
            for record in injected:
                try:
                    self._killpg(record.pgid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                except Exception as error:
                    injected_ok = False
                    errors.append(f"signal {record.name}({record.pgid}): {error}")
                try:
                    self._waitpid(record.pid, 0)
                except ChildProcessError:
                    pass
                except Exception as error:
                    injected_ok = False
                    errors.append(f"wait {record.name}({record.pid}): {error}")

            if real:
                if type(deadline) not in (int, float) or not np.isfinite(deadline):
                    errors.append("real child cleanup requires a finite deadline")
                    deadline = self._clock()
                live = self._live(real, errors)
                if deadline <= self._clock():
                    errors.append("shared child cleanup deadline exhausted")
                else:
                    self._signal(live, signal.SIGINT, errors)
                    live = self._wait_live_until(
                        live, min(deadline, self._clock() + 0.5), errors
                    )
                    self._signal(live, signal.SIGTERM, errors)
                    live = self._wait_live_until(
                        live, min(deadline, self._clock() + 0.2), errors
                    )

                live = self._live(real, errors)
                self._signal(live, signal.SIGKILL, errors)
                live = self._wait_live_until(live, deadline, errors)
                if live:
                    errors.append(
                        "live children: "
                        + ",".join(f"{record.name}({record.pid})" for record in live)
                    )
            self._closed = not live and injected_ok
        finally:
            self._closing = False
        if errors:
            raise SmokeSafetyError("child cleanup failed: " + "; ".join(errors))


def close_smoke_resources(
    owner,
    descriptors,
    logs,
    deadline,
    *,
    close_fd=os.close,
):
    errors = []
    try:
        owner.close(deadline=deadline)
    except Exception as error:
        errors.append(f"children: {error}")
        if getattr(owner, "_closed", True) is False:
            try:
                owner.close(deadline=deadline)
            except Exception as retry_error:
                errors.append(f"children retry: {retry_error}")
    for descriptor in descriptors:
        if descriptor is None:
            continue
        try:
            close_fd(descriptor)
        except Exception as error:
            errors.append(f"descriptor {descriptor}: {error}")
    for index, log in enumerate(logs):
        try:
            log.close()
        except Exception as error:
            errors.append(f"log {index}: {error}")
    if errors:
        raise SmokeSafetyError("smoke resource cleanup failed: " + "; ".join(errors))


def wait_ready_json(
    path,
    expected_host,
    expected_port,
    deadline,
    clock=time.monotonic,
):
    path = Path(path)
    while clock() < deadline:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            expected = {"host": expected_host, "port": expected_port}
            if payload != expected:
                raise SmokeSafetyError(f"readiness mismatch: {payload!r}")
            return payload
        time.sleep(min(0.01, max(0.0, deadline - clock())))
    raise TimeoutError(f"readiness timeout: {path}")


def sleep_until(deadline, clock=time.monotonic, sleep=time.sleep):
    sleep(max(0.0, deadline - clock()))


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SmokeSafetyError(
                    f"invalid metrics JSONL line {line_number}: {error}"
                ) from error
    return records


def wait_bridge_preflight(path, deadline, clock=time.monotonic):
    path = Path(path)
    while clock() < deadline:
        if path.exists() and any(
            record.get("event") == "preflight_complete" for record in read_jsonl(path)
        ):
            return
        time.sleep(min(0.01, max(0.0, deadline - clock())))
    raise TimeoutError("bridge preflight readiness timeout")


def default_wbc_preflight(
    deadline,
    *,
    client_factory=None,
    clock=time.monotonic,
):
    remaining = deadline - clock()
    if remaining <= 0.0:
        raise TimeoutError("WBC readiness deadline")
    if client_factory is None:
        from scripts.psi0_simple_real_bridge import create_ros_wbc_config_client

        client_factory = create_ros_wbc_config_client
    client = client_factory("WBCPolicy/robot_config", domain_id=42)
    payload = client.get_config(timeout_s=min(10.0, remaining))
    if type(payload) is not dict:
        raise SmokeSafetyError("WBC configuration payload must be an object")
    if payload.get("env_type") != "sim" or payload.get("interface") != "lo":
        raise SmokeSafetyError("WBC is not isolated sim/loopback")
    return payload


def default_goal_counts(runtime_factory=None):
    if runtime_factory is None:
        from scripts.psi0_simple_real_bridge import RosRuntime

        runtime_factory = RosRuntime
    runtime = runtime_factory(42)
    try:
        return list(runtime.counts(CONTROL_GOAL_TOPIC))
    finally:
        runtime.close(timeout_s=0.5)


def policy_status(config, session=requests):
    response = session.get(
        f"http://{config.policy_host}:{config.policy_port}/test-control/status",
        headers={"X-Test-Control-Token": config.control_token},
        timeout=0.5,
    )
    response.raise_for_status()
    payload = response.json()
    if type(payload) is not dict or set(payload) != {
        "last_started_request_seq",
        "active_requests",
        "max_concurrent_requests",
        "records",
    }:
        raise SmokeSafetyError("malformed fake-policy status")
    if (
        type(payload["last_started_request_seq"]) is not int
        or type(payload["active_requests"]) is not int
        or type(payload["max_concurrent_requests"]) is not int
        or type(payload["records"]) is not list
    ):
        raise SmokeSafetyError("malformed fake-policy status types")
    return payload


def port_rebinds(port):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    return True


@dataclass(frozen=True)
class PortRebindCheck:
    ok: bool
    failures: tuple[str, ...]


def check_all_ports_rebound(ports, rebind=port_rebinds):
    failures = []
    for port in ports:
        try:
            rebound = rebind(port)
        except Exception as error:
            failures.append(f"port {port}: {error}")
            continue
        if rebound is not True:
            failures.append(f"port {port} did not rebind")
    return PortRebindCheck(not failures, tuple(failures))


def _live_non_daemon_threads():
    return [
        thread.name
        for thread in threading.enumerate()
        if thread is not threading.current_thread()
        and not thread.daemon
        and thread.is_alive()
    ]


def default_measure_worker_idle(records, scenario_started_at):
    return measured_bridge_worker_idle_at(records, scenario_started_at)


class DefaultSmokeHooks:
    clock = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)

    @staticmethod
    def resolve_popen():
        return subprocess.Popen

    owner_factory = staticmethod(OwnedChildren)
    wbc_preflight = staticmethod(default_wbc_preflight)
    goal_counts = staticmethod(default_goal_counts)
    wait_ready = staticmethod(wait_ready_json)
    openpty = staticmethod(pty.openpty)
    terminal_get = staticmethod(termios.tcgetattr)
    wait_bridge = staticmethod(wait_bridge_preflight)
    write = staticmethod(os.write)
    arm_delay = staticmethod(arm_next_policy_delay)
    policy_status = staticmethod(policy_status)
    signal_group = staticmethod(os.killpg)
    read_metrics = staticmethod(read_jsonl)
    measure_worker_idle = staticmethod(default_measure_worker_idle)
    port_rebind = staticmethod(port_rebinds)
    close_fd = staticmethod(os.close)
    live_threads = staticmethod(_live_non_daemon_threads)


def measured_real_interface_connections(records):
    preflight = [
        record for record in records if record.get("event") == "preflight_complete"
    ]
    if len(preflight) != 1:
        raise SmokeSafetyError("expected one preflight connection record")
    record = preflight[0]
    evidence = record.get("connection_evidence")
    if (
        type(evidence) is not list
        or not all(type(item) is dict for item in evidence)
        or {item.get("component") for item in evidence} != {"wbc", "camera", "policy"}
    ):
        raise SmokeSafetyError(
            "smoke requires observed WBC, camera, and policy connections"
        )
    measured = count_real_interface_connections(evidence)
    if (
        type(record.get("real_interface_connections")) is not int
        or record["real_interface_connections"] != measured
    ):
        raise SmokeSafetyError("real-interface count differs from evidence")
    return measured


def measured_bridge_worker_idle_at(records, scenario_started_at):
    candidates = [
        record
        for record in records
        if record.get("event") == "tick"
        and type(record.get("monotonic_s")) in (int, float)
        and 12.9 <= record["monotonic_s"] - scenario_started_at <= 13.0
    ]
    if not candidates:
        raise SmokeSafetyError("no bridge tick measured near second 13")
    latest = max(candidates, key=lambda record: record["monotonic_s"])
    if latest.get("worker_busy") is not False:
        raise SmokeSafetyError("bridge-owned worker is not idle at second 13")
    return latest["monotonic_s"] - scenario_started_at


def collect_smoke_report(
    records,
    *,
    scenario_started_at,
    armed_seq,
    delayed_record,
    goal_counts_before,
    goal_counts_running,
    goal_counts_after,
    terminal_restored,
    ports_rebound,
    bridge_exit_code,
    live_children_after,
    child_exit_codes,
    live_threads_after=None,
):
    ticks = [record for record in records if record.get("event") == "tick"]
    publishes = [record for record in ticks if record.get("published") is True]
    fault_ticks = [record for record in ticks if record.get("state") == "fault"]
    if not fault_ticks:
        raise SmokeSafetyError("metrics contain no lowercase fault tick")
    if not publishes:
        raise SmokeSafetyError("metrics contain no published ticks")
    fault_at = fault_ticks[0]["monotonic_s"] - scenario_started_at
    requests_ = [
        {
            "request_seq": record["request_seq"],
            "observation_tick": record["observation_tick"],
            "time_s": record["monotonic_s"] - scenario_started_at,
            "committed_actions": record["committed_actions"],
        }
        for record in records
        if record.get("event") == "request"
    ]
    unpublished_ticks = [
        {
            "tick": record["tick"],
            "time_s": record["monotonic_s"] - scenario_started_at,
            "source_kind": record.get("source_kind"),
            "state": record.get("state"),
        }
        for record in ticks
        if record.get("published") is not True
        and -0.02 <= record["monotonic_s"] - scenario_started_at <= 13.05
    ]
    executed = [
        {"tick": record["tick"], "post_slew_action": record["psi0_action"]}
        for record in publishes
        if record.get("psi0_action") is not None
    ]
    published_ticks = [
        {
            "tick": record["tick"],
            "time_s": record["monotonic_s"] - scenario_started_at,
            "source_kind": record.get("source_kind"),
            "state": record.get("state"),
        }
        for record in publishes
    ]
    request_record = next(
        record
        for record in records
        if record.get("event") == "request" and record.get("request_seq") == armed_seq
    )
    relative_publish_times = [
        record["monotonic_s"] - scenario_started_at for record in publishes
    ]
    before_fault = [value for value in relative_publish_times if value < fault_at]
    after_fault = [value for value in relative_publish_times if value >= fault_at]
    gaps = np.diff(relative_publish_times)
    first_fault_action = np.asarray(fault_ticks[0]["psi0_action"], np.float32)
    final_tick = ticks[-1]
    return {
        "steady_phases": [
            {"publish_times": before_fault},
            {"publish_times": after_fault},
        ],
        "published_ticks": published_ticks,
        "unpublished_ticks": unpublished_ticks,
        "requests": requests_,
        "executed_actions": executed,
        "blocked_main_loop_max_gap_s": float(np.max(gaps)),
        "delayed_request_started_at": (
            request_record["monotonic_s"] - scenario_started_at
        ),
        "armed_request_seq": armed_seq,
        "delayed_request_record": delayed_record,
        "fault_at": fault_at,
        "first_fault_goal_navigation": first_fault_action[32:36].tolist(),
        "policy_actions_after_fault": sum(
            record.get("source_kind") == "policy" for record in fault_ticks
        ),
        "old_generation_results_discarded": final_tick[
            "discarded_old_generation_results"
        ],
        "late_results_discarded": final_tick["discarded_late_results"],
        "worker_idle_at_s": measured_bridge_worker_idle_at(
            records, scenario_started_at
        ),
        "goal_counts_before": goal_counts_before,
        "goal_counts_running": goal_counts_running,
        "goal_counts_after": goal_counts_after,
        "live_children_after": list(live_children_after),
        "child_exit_codes": dict(child_exit_codes),
        "bridge_exit_code": bridge_exit_code,
        "live_threads_after": (
            _live_non_daemon_threads()
            if live_threads_after is None
            else list(live_threads_after)
        ),
        "terminal_restored": terminal_restored,
        "ports_rebound": ports_rebound,
        "real_interface_connections": measured_real_interface_connections(records),
        "extra_goal_publishers": max(0, goal_counts_running[0] - 1),
    }


@dataclass(frozen=True)
class SmokeValidation:
    ok: bool
    failures: tuple[str, ...]


def _exact_value(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _exact_value(left, right) for left, right in zip(actual, expected)
        )
    if type(expected) is dict:
        return set(actual) == set(expected) and all(
            _exact_value(actual[key], expected[key]) for key in expected
        )
    return actual == expected


_REPORT_KEYS = frozenset(
    {
        "steady_phases",
        "published_ticks",
        "unpublished_ticks",
        "requests",
        "executed_actions",
        "blocked_main_loop_max_gap_s",
        "delayed_request_started_at",
        "armed_request_seq",
        "delayed_request_record",
        "fault_at",
        "first_fault_goal_navigation",
        "policy_actions_after_fault",
        "old_generation_results_discarded",
        "late_results_discarded",
        "worker_idle_at_s",
        "goal_counts_before",
        "goal_counts_running",
        "goal_counts_after",
        "live_children_after",
        "child_exit_codes",
        "bridge_exit_code",
        "live_threads_after",
        "terminal_restored",
        "ports_rebound",
        "real_interface_connections",
        "extra_goal_publishers",
    }
)
_REQUEST_KEYS = frozenset(
    {"request_seq", "observation_tick", "time_s", "committed_actions"}
)
_EXPECTED_REQUEST_COUNT = 18
_REQUEST_TICK_SPACING = 24
_COMMITTED_ACTION_COUNT = 6
_ACTION_DIMENSION = 36


def _finite_float_vector(value, size):
    return (
        type(value) is list
        and len(value) == size
        and all(type(item) is float and np.isfinite(item) for item in value)
    )


def _finite_float_matrix(value, rows, columns):
    return (
        type(value) is list
        and len(value) == rows
        and all(_finite_float_vector(row, columns) for row in value)
    )


def validate_smoke_report(report):
    failures = []

    # Validate the evidence boundary before inspecting any report value. In
    # particular, this prevents credentials or other accidental fields from
    # silently becoming part of the persisted smoke report contract.
    if type(report) is not dict or set(report) != _REPORT_KEYS:
        return SmokeValidation(
            False,
            ("report_schema: expected exact pre-validation top-level key set",),
        )

    phases = report.get("steady_phases")
    phase_times = []
    if type(phases) is not list or len(phases) != 2:
        failures.append("steady_phases: expected two phases")
    else:
        for index, phase in enumerate(phases):
            if type(phase) is not dict or set(phase) != {"publish_times"}:
                failures.append(f"steady_phases[{index}]: record schema")
                phase_times.append(None)
                continue
            publish_times = phase["publish_times"]
            if type(publish_times) is not list or not all(
                type(value) is float and np.isfinite(value) for value in publish_times
            ):
                failures.append(f"steady_phases[{index}]: timestamps")
                phase_times.append(None)
                continue
            times = np.asarray(publish_times, np.float64)
            phase_times.append(times)
            if len(times) < 2:
                failures.append(f"steady_phases[{index}]: timestamps")
                continue
            gaps = np.diff(times)
            hz = (len(times) - 1) / (times[-1] - times[0])
            if not 49.0 <= hz <= 51.0:
                failures.append(f"steady_phases[{index}]: rate")
            if np.max(gaps) > 0.060:
                failures.append(f"maximum_gap: steady_phases[{index}]")

    published = report.get("published_ticks")
    timeline = []
    timeline_by_tick = {}
    active = []
    if type(published) is not list:
        failures.append("published_ticks: expected list")
    else:
        for entry in published:
            if type(entry) is not dict or set(entry) != {
                "tick",
                "time_s",
                "source_kind",
                "state",
            }:
                failures.append("published_ticks: record schema")
                continue
            tick = entry["tick"]
            time_s = entry["time_s"]
            if (
                type(tick) is not int
                or type(time_s) is not float
                or not np.isfinite(time_s)
                or type(entry["source_kind"]) is not str
                or entry["source_kind"] not in {"hold", "policy"}
                or type(entry["state"]) is not str
                or entry["state"] not in {"paused", "active", "fault"}
            ):
                failures.append("published_ticks: record value")
                continue
            if tick in timeline_by_tick:
                failures.append("published_ticks: duplicate tick")
            timeline.append(entry)
            timeline_by_tick[tick] = entry

    unpublished = report.get("unpublished_ticks")
    if type(unpublished) is not list or unpublished:
        failures.append(
            "unpublished_ticks: expected exact empty list for certified window"
        )

    if timeline:
        timeline_times = np.asarray([entry["time_s"] for entry in timeline], np.float64)
        timeline_tick_values = [entry["tick"] for entry in timeline]
        if not np.all(np.diff(timeline_times) > 0.0):
            failures.append("published_ticks: timestamps must increase")
        if len(timeline_times) > 1 and np.max(np.diff(timeline_times)) > 0.060:
            failures.append("published_ticks: maximum gap exceeds 0.060")
        if timeline_tick_values != list(
            range(timeline_tick_values[0], timeline_tick_values[-1] + 1)
        ):
            failures.append("published_ticks: discontinuous global ticks")
        if len(phase_times) == 2 and all(
            times is not None and len(times) for times in phase_times
        ):
            combined_phase_times = np.concatenate(phase_times)
            if not np.array_equal(timeline_times, combined_phase_times):
                failures.append("published_ticks: differs from steady phase evidence")

        if not (-0.02 <= timeline_times[0] <= 0.04) or timeline_times[-1] < 12.90:
            failures.append("timeline_coverage: expected publications from 0 to 13s")

        paused = [entry for entry in timeline if 0.0 <= entry["time_s"] < 3.0]
        active = [entry for entry in timeline if 3.0 <= entry["time_s"] < 11.0]
        finishing = [entry for entry in timeline if 11.0 <= entry["time_s"] <= 13.05]
        if (
            len(paused) < 145
            or not paused
            or paused[0]["time_s"] > 0.04
            or paused[-1]["time_s"] < 2.94
        ):
            failures.append("paused_phase: insufficient 0-3s coverage")
        elif any(
            entry["source_kind"] != "hold" or entry["state"] != "paused"
            for entry in paused
        ):
            failures.append("paused_phase: publications must be paused hold")
        if (
            len(active) < 390
            or not active
            or active[0]["time_s"] > 3.04
            or active[-1]["time_s"] < 10.96
            or len(finishing) < 90
            or not finishing
            or finishing[0]["time_s"] > 11.04
            or finishing[-1]["time_s"] < 12.90
        ):
            failures.append("phase_coverage: active/finishing timeline incomplete")
        delayed_start = report.get("delayed_request_started_at")
        if type(delayed_start) is float and np.isfinite(delayed_start):
            blocked_end = delayed_start + 0.30
            blocked = [
                entry
                for entry in timeline
                if delayed_start - 1e-9 <= entry["time_s"] <= blocked_end + 1e-9
            ]
            if (
                len(blocked) < 13
                or blocked[0]["time_s"] > delayed_start + 0.02
                or blocked[-1]["time_s"] < blocked_end - 0.02
                or np.max(np.diff([entry["time_s"] for entry in blocked]), initial=0.0)
                > 0.060
            ):
                failures.append("blocked_http_interval: publications did not continue")

    executed = report.get("executed_actions")
    executed_by_tick = {}
    if type(executed) is not list:
        failures.append("executed_actions: expected list")
    else:
        for entry in executed:
            if type(entry) is not dict or set(entry) != {
                "tick",
                "post_slew_action",
            }:
                failures.append("executed_actions: record schema")
                continue
            tick = entry["tick"]
            if type(tick) is not int or not _finite_float_vector(
                entry["post_slew_action"], _ACTION_DIMENSION
            ):
                failures.append("executed_actions: record value")
                continue
            if tick in executed_by_tick:
                failures.append("executed_actions: duplicate tick")
            executed_by_tick[tick] = np.asarray(entry["post_slew_action"], np.float64)
        ticks = sorted(executed_by_tick)
        if timeline and ticks != [entry["tick"] for entry in timeline]:
            failures.append(
                "executed_actions: tick keys must exactly cover published timeline"
            )

    requests_ = report.get("requests")
    parsed_requests = []
    if type(requests_) is not list:
        failures.append("request_coverage: requests must be a list")
    else:
        if len(requests_) != _EXPECTED_REQUEST_COUNT:
            failures.append("request_coverage: expected requests seq 0 through 17")
        for entry in requests_:
            if type(entry) is not dict or set(entry) != _REQUEST_KEYS:
                failures.append("requests: record schema")
                continue
            request_seq = entry["request_seq"]
            tick = entry["observation_tick"]
            request_time = entry["time_s"]
            committed_value = entry["committed_actions"]
            valid_fields = True
            if type(request_seq) is not int or type(tick) is not int:
                failures.append("requests: record value")
                valid_fields = False
            if type(request_time) is not float or not np.isfinite(request_time):
                failures.append("request_timing: expected finite float timestamps")
                valid_fields = False
            if not _finite_float_matrix(
                committed_value, _COMMITTED_ACTION_COUNT, _ACTION_DIMENSION
            ):
                failures.append("committed_actions: expected finite (6,36)")
                valid_fields = False
            if not valid_fields:
                continue
            committed = np.asarray(committed_value, np.float64)
            parsed_requests.append((request_seq, tick, request_time, committed))

        if len(parsed_requests) == _EXPECTED_REQUEST_COUNT:
            request_sequences = [entry[0] for entry in parsed_requests]
            request_ticks = [entry[1] for entry in parsed_requests]
            request_times = [entry[2] for entry in parsed_requests]
            if request_sequences != list(range(_EXPECTED_REQUEST_COUNT)):
                failures.append(
                    "request_sequence: expected consecutive seq 0 through 17"
                )
            if any(
                later - earlier != _REQUEST_TICK_SPACING
                for earlier, later in zip(request_ticks, request_ticks[1:])
            ):
                failures.append("request_spacing: expected 24 ticks")
            if any(
                later <= earlier
                for earlier, later in zip(request_times, request_times[1:])
            ) or any(
                abs((later - earlier) - 0.48) > 0.04
                for earlier, later in zip(request_times, request_times[1:])
            ):
                failures.append(
                    "request_timing: expected increasing 24-tick cadence at 50 Hz"
                )
            if any(
                abs(request_time - (3.0 + request_seq * 0.48)) > 0.04
                for request_seq, _, request_time, _ in parsed_requests
            ):
                failures.append(
                    "request_absolute_timing: every request must be within 0.04s "
                    "of 3.00 + request_seq*0.48"
                )
            if (
                not active
                or request_ticks[0] != active[0]["tick"]
                or not 2.98 <= request_times[0] <= 3.04
                or abs(request_times[0] - active[0]["time_s"]) > 0.04
            ):
                failures.append(
                    "first_request_alignment: R0 must match first active tick near 3s"
                )
            if any(
                tick not in timeline_by_tick
                or abs(timeline_by_tick[tick]["time_s"] - request_time) > 0.04
                for _, tick, request_time, _ in parsed_requests
            ):
                failures.append(
                    "request_timing: request times must align with observation ticks"
                )

        r0_hold_supported = False
        policy_successors = 0
        for request_seq, tick, _, committed in parsed_requests:
            if any(
                global_tick not in executed_by_tick
                for global_tick in range(tick, tick + _COMMITTED_ACTION_COUNT)
            ):
                failures.append("committed_actions: executed tick missing")
                continue
            actual = np.stack(
                [
                    executed_by_tick[global_tick]
                    for global_tick in range(tick, tick + _COMMITTED_ACTION_COUNT)
                ]
            )
            if not np.array_equal(committed, actual):
                failures.append("committed_actions: differs from post-slew execution")
                continue
            source_kinds = [
                timeline_by_tick.get(global_tick, {}).get("source_kind")
                for global_tick in range(tick, tick + _COMMITTED_ACTION_COUNT)
            ]
            states = [
                timeline_by_tick.get(global_tick, {}).get("state")
                for global_tick in range(tick, tick + _COMMITTED_ACTION_COUNT)
            ]
            if request_seq == 0:
                r0_hold_supported = (
                    source_kinds == ["hold"] * 6 and states == ["active"] * 6
                )
            elif source_kinds == ["policy"] * 6 and states == ["active"] * 6:
                policy_successors += 1
        if not r0_hold_supported:
            failures.append("r0_committed_hold: expected six committed hold ticks")
        if policy_successors != _EXPECTED_REQUEST_COUNT - 1:
            failures.append(
                "repeated_handoffs/active_policy_execution: every successor prefix "
                "must be backed by active policy ticks"
            )

    armed_seq = report.get("armed_request_seq")
    if type(armed_seq) is not int or armed_seq != _EXPECTED_REQUEST_COUNT - 1:
        failures.append("armed_request_seq: expected unique final request seq 17")
    elif parsed_requests and (
        parsed_requests[-1][0] != armed_seq
        or sum(entry[0] == armed_seq for entry in parsed_requests) != 1
    ):
        failures.append("armed_request_seq: must identify unique final request")

    exact = {
        "first_fault_goal_navigation": [0.0, 0.0, 0.0, 0.0],
        "policy_actions_after_fault": 0,
        "old_generation_results_discarded": 1,
        "late_results_discarded": 0,
        "goal_counts_before": [0, 1],
        "goal_counts_running": [1, 1],
        "goal_counts_after": [0, 1],
        "live_children_after": [],
        "bridge_exit_code": 0,
        "live_threads_after": [],
        "terminal_restored": True,
        "ports_rebound": True,
        "real_interface_connections": 0,
        "extra_goal_publishers": 0,
    }
    for field, expected in exact.items():
        if not _exact_value(report.get(field), expected):
            failures.append(f"{field}: expected {expected!r}")
    child_exit_codes = report.get("child_exit_codes")
    if (
        type(child_exit_codes) is not dict
        or set(child_exit_codes) != {"wbc", "camera", "policy", "bridge"}
        or any(type(code) is not int for code in child_exit_codes.values())
    ):
        failures.append("child_exit_codes: every owned child must be reaped")
    blocked_gap = report.get("blocked_main_loop_max_gap_s")
    if (
        type(blocked_gap) is not float
        or not np.isfinite(blocked_gap)
        or blocked_gap > 0.060
    ):
        failures.append("blocked_main_loop_max_gap_s: over 0.060")
    started = report.get("delayed_request_started_at")
    fault_at = report.get("fault_at")
    started_valid = type(started) is float and np.isfinite(started)
    fault_valid = type(fault_at) is float and np.isfinite(fault_at)
    if not started_valid:
        failures.append("delayed_request_started_at: missing finite timestamp")
    elif not 11.0 <= started <= 11.5:
        failures.append("delayed_request_started_at: expected in [11.0, 11.5]")
    elif parsed_requests and abs(started - parsed_requests[-1][2]) > 1e-9:
        failures.append("delayed_request_started_at: differs from final request")
    if not fault_valid:
        failures.append("fault_at: missing finite timestamp")
    if started_valid and fault_valid and not started <= fault_at <= started + 0.14:
        failures.append(
            "fault_at: expected request_start <= fault <= request_start+0.14"
        )
    fault_timeline = [entry for entry in timeline if entry["state"] == "fault"]
    if fault_valid and (
        not fault_timeline or abs(fault_at - fault_timeline[0]["time_s"]) > 1e-9
    ):
        failures.append("fault_at: differs from first fault publication")
    fault_boundary = next(
        (
            entry
            for entry in timeline
            if fault_valid and abs(entry["time_s"] - fault_at) <= 1e-9
        ),
        None,
    )
    if fault_valid and fault_boundary is None:
        failures.append("fault_at: no publication at fault boundary")

    if active and fault_valid:
        first_active_time = active[0]["time_s"]
        pre_fault = [
            entry
            for entry in timeline
            if first_active_time - 1e-9 <= entry["time_s"] < fault_at
        ]
        post_fault = [entry for entry in timeline if entry["time_s"] >= fault_at - 1e-9]
        if any(entry["state"] != "active" for entry in pre_fault):
            failures.append(
                "active_state/pre_fault_state: every tick from R0 to fault must be active"
            )
        if len(pre_fault) < _COMMITTED_ACTION_COUNT or (
            any(
                entry["source_kind"] != "hold"
                for entry in pre_fault[:_COMMITTED_ACTION_COUNT]
            )
            or any(
                entry["source_kind"] != "policy"
                for entry in pre_fault[_COMMITTED_ACTION_COUNT:]
            )
        ):
            failures.append(
                "pre_fault_source_sequence/active_source_sequence: expected exactly "
                "six R0 hold ticks followed by policy ticks until fault"
            )
        if any(entry["state"] != "fault" for entry in post_fault):
            failures.append(
                "post_fault_state: every chronological tick at/after fault must remain fault"
            )
        if any(entry["source_kind"] != "hold" for entry in post_fault):
            failures.append(
                "post_fault_source: every chronological tick at/after fault must be hold"
            )
        observed_policy_after_fault = sum(
            entry["source_kind"] == "policy" for entry in post_fault
        )
        if report.get("policy_actions_after_fault") != observed_policy_after_fault:
            failures.append(
                "policy_actions_after_fault: differs from chronological publications"
            )

    if fault_boundary is not None:
        first_fault_action = executed_by_tick.get(fault_boundary["tick"])
        reported_navigation = report.get("first_fault_goal_navigation")
        if (
            first_fault_action is None
            or not _finite_float_vector(reported_navigation, 4)
            or not np.array_equal(
                first_fault_action[32:36],
                np.asarray(reported_navigation, np.float64),
            )
        ):
            failures.append(
                "first_fault_goal_navigation: differs from first fault action"
            )
    worker_idle = report.get("worker_idle_at_s")
    if (
        type(worker_idle) is not float
        or not np.isfinite(worker_idle)
        or worker_idle > 13.0
    ):
        failures.append("worker_idle_at_s: exceeds second 13 or is non-finite")
    expected_delay = {
        "request_seq": report.get("armed_request_seq"),
        "applied_latency_s": 0.30,
    }
    if not _exact_value(report.get("delayed_request_record"), expected_delay):
        failures.append("delayed_request_record: acknowledgement mismatch")
    return SmokeValidation(not failures, tuple(failures))


def launch(config, hooks=None):
    plan = build_launch_plan(config)
    hooks = hooks or DefaultSmokeHooks()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    owner = hooks.owner_factory(clock=hooks.clock, sleep=hooks.sleep)
    popen = hooks.resolve_popen()
    logs = []
    bridge_master = bridge_slave = None
    bridge_original = None
    smoke_deadline = None
    report_args = None

    def cleanup():
        cleanup_deadline = (
            smoke_deadline if smoke_deadline is not None else hooks.clock() + 1.0
        )
        close_smoke_resources(
            owner,
            (bridge_master, bridge_slave),
            logs,
            cleanup_deadline,
            close_fd=hooks.close_fd,
        )

    try:

        def start(spec, stdin=None):
            log = (output / f"psi0-smoke-{spec.name}.log").open("x", encoding="utf-8")
            logs.append(log)
            process = popen(
                spec.argv,
                env=dict(spec.env),
                stdin=stdin,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            owner.record(
                pid=process.pid,
                pgid=process.pid,
                name=spec.name,
                argv=spec.argv,
                started_at=hooks.clock(),
                process=process,
            )
            return process

        start(plan.child("wbc"))
        hooks.wbc_preflight(hooks.clock() + 10.0)
        goal_counts_before = hooks.goal_counts()
        if goal_counts_before != [0, 1]:
            raise SmokeSafetyError(f"WBC preflight graph: {goal_counts_before}")

        start(plan.child("camera"))
        hooks.wait_ready(
            config.camera_ready_json,
            config.camera_host,
            config.camera_port,
            hooks.clock() + 1.0,
            hooks.clock,
        )
        start(plan.child("policy"))
        hooks.wait_ready(
            config.policy_ready_json,
            config.policy_host,
            config.policy_port,
            hooks.clock() + 1.0,
            hooks.clock,
        )

        bridge_master, bridge_slave = hooks.openpty()
        bridge_original = hooks.terminal_get(bridge_slave)
        bridge_process = start(plan.child("bridge"), stdin=bridge_slave)
        hooks.wait_bridge(config.bridge_metrics_jsonl, hooks.clock() + 3.0, hooks.clock)
        scenario_started_at = hooks.clock()
        smoke_deadline = scenario_started_at + config.duration_s

        sleep_until(scenario_started_at + 3.0, hooks.clock, hooks.sleep)
        hooks.write(bridge_master, b"p")
        goal_counts_running = hooks.goal_counts()
        if goal_counts_running != [1, 1]:
            raise SmokeSafetyError(f"bridge graph ownership: {goal_counts_running}")

        sleep_until(scenario_started_at + 11.0, hooks.clock, hooks.sleep)
        armed_seq = hooks.arm_delay(
            config.policy_host,
            config.policy_port,
            config.control_token,
            0.30,
        )
        sleep_until(scenario_started_at + 13.0, hooks.clock, hooks.sleep)
        contemporaneous_records = hooks.read_metrics(config.bridge_metrics_jsonl)
        hooks.measure_worker_idle(contemporaneous_records, scenario_started_at)
        status = hooks.policy_status(config)
        if status["active_requests"] != 0:
            raise SmokeSafetyError("fake policy still has an active request")
        matches = [
            record
            for record in status["records"]
            if record == {"request_seq": armed_seq, "applied_latency_s": 0.30}
        ]
        if len(matches) != 1:
            raise SmokeSafetyError("one-shot delay did not reach armed request")

        hooks.signal_group(owner.process_group("bridge"), signal.SIGINT)
        bridge_exit_code = bridge_process.wait(
            timeout=max(0.0, smoke_deadline - hooks.clock())
        )
        goal_counts_after = hooks.goal_counts()
        if goal_counts_after != [0, 1]:
            raise SmokeSafetyError(f"publisher cleanup graph: {goal_counts_after}")
        terminal_restored = hooks.terminal_get(bridge_slave) == bridge_original
        records = hooks.read_metrics(config.bridge_metrics_jsonl)
        report_args = (
            records,
            scenario_started_at,
            armed_seq,
            matches[0],
            goal_counts_before,
            goal_counts_running,
            goal_counts_after,
            terminal_restored,
            bridge_exit_code,
        )
    except BaseException as primary_error:
        try:
            cleanup()
        except Exception as cleanup_error:
            if not isinstance(primary_error, Exception):
                raise primary_error from cleanup_error
            raise SmokeSafetyError(
                f"smoke operation failed: {primary_error}; "
                f"cleanup failed: {cleanup_error}"
            ) from primary_error
        raise
    else:
        cleanup()

    if report_args is None:
        raise AssertionError("smoke report inputs were not collected")
    port_rebind_check = check_all_ports_rebound(
        (config.camera_port, config.policy_port), hooks.port_rebind
    )
    report = collect_smoke_report(
        report_args[0],
        scenario_started_at=report_args[1],
        armed_seq=report_args[2],
        delayed_record=report_args[3],
        goal_counts_before=report_args[4],
        goal_counts_running=report_args[5],
        goal_counts_after=report_args[6],
        terminal_restored=report_args[7],
        ports_rebound=port_rebind_check.ok,
        bridge_exit_code=report_args[8],
        live_children_after=owner.live_pids(),
        child_exit_codes=owner.exit_codes(),
        live_threads_after=hooks.live_threads(),
    )
    validation = validate_smoke_report(report)
    report["validation"] = {
        "ok": validation.ok,
        "failures": list(validation.failures),
    }
    Path(config.smoke_report_json).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if not validation.ok:
        raise SmokeSafetyError("; ".join(validation.failures))
    if smoke_deadline is None or hooks.clock() > smoke_deadline:
        raise SmokeSafetyError("smoke cleanup exceeded shared second-15 deadline")
    return report


def build_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--duration-s", required=True, type=float)
    parser.add_argument("--unitree-domain-id", required=True, type=int)
    parser.add_argument("--camera-port", required=True, type=int)
    parser.add_argument("--policy-port", required=True, type=int)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--control-token", default=None, help=argparse.SUPPRESS)
    return parser


def allocate_smoke_run_directory(
    output_root,
    *,
    now_ns=time.time_ns,
    token_hex=secrets.token_hex,
):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"psi0-smoke-{now_ns()}-{token_hex(4)}"
    output.mkdir(mode=0o755, exist_ok=False)
    return output


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = allocate_smoke_run_directory(args.output_dir)
    print(f"smoke run directory: {output}", flush=True)
    token = secrets.token_hex(16) if args.control_token is None else args.control_token
    config = SmokeConfig(
        duration_s=args.duration_s,
        ros_domain_id=42,
        unitree_domain_id=args.unitree_domain_id,
        wbc_interface="lo",
        camera_host="127.0.0.1",
        camera_port=args.camera_port,
        policy_host="127.0.0.1",
        policy_port=args.policy_port,
        control_token=token,
        policy_ready_json=str(output / "psi0-smoke-policy-ready.json"),
        camera_ready_json=str(output / "psi0-smoke-camera-ready.json"),
        bridge_metrics_jsonl=str(output / "psi0-smoke-bridge.jsonl"),
        smoke_report_json=str(output / "psi0-smoke-report.json"),
        output_dir=str(output),
    )
    launch(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
