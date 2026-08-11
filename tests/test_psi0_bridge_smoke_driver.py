import builtins
from dataclasses import FrozenInstanceError
import importlib.util
import os
import signal
import subprocess
import sys

import numpy as np
import pytest

import scripts.tests.smoke_psi0_simple_bridge as smoke_driver
from scripts.tests.smoke_psi0_simple_bridge import (
    OwnedChildren,
    SmokeConfig,
    SmokeSafetyError,
    allocate_smoke_run_directory,
    arm_next_policy_delay,
    build_launch_plan,
    close_smoke_resources,
    collect_smoke_report,
    default_wbc_preflight,
    launch,
    measured_bridge_worker_idle_at,
    measured_real_interface_connections,
    validate_smoke_report,
)
from simple.deploy.psi0_simple_bridge import (
    PSI0_ACTION_JOINT_NAMES,
    PSI0_WAIST_ACTION_NAMES,
)


EXPECTED_WBC_ARGS = (
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
)

EXPECTED_CAMERA_ARGS = (
    "uv",
    "run",
    "--group",
    "sonic",
    "python",
    "scripts/tests/fake_composed_camera_server.py",
    "--host",
    "127.0.0.1",
    "--port",
    "15555",
    "--ready-json",
    "outputs/psi0-smoke-camera-ready.json",
)

EXPECTED_POLICY_ARGS = (
    "uv",
    "run",
    "--group",
    "sonic",
    "python",
    "scripts/tests/fake_psi0_rtc_server.py",
    "--host",
    "127.0.0.1",
    "--port",
    "22086",
    "--contract",
    "scripts/tests/fixtures/psi0_policy_contract_test_v2.json",
    "--normal-latency-s",
    "0.05",
    "--control-token",
    "smoke-control-token",
    "--ready-json",
    "outputs/psi0-smoke-policy-ready.json",
)

EXPECTED_BRIDGE_ARGS = (
    "uv",
    "run",
    "--group",
    "sonic",
    "python",
    "scripts/psi0_simple_real_bridge.py",
    "--mode",
    "sim-control",
    "--server-host",
    "127.0.0.1",
    "--server-port",
    "22086",
    "--instruction",
    "smoke test hold",
    "--policy-contract",
    "scripts/tests/fixtures/psi0_policy_contract_test_v2.json",
    "--camera-host",
    "127.0.0.1",
    "--camera-port",
    "15555",
    "--camera-source-key",
    "rgb_head_stereo_left",
    "--camera-color-order",
    "rgb",
    "--ros-domain-id",
    "42",
    "--unitree-domain-id",
    "42",
    "--metrics-jsonl",
    "outputs/psi0-smoke-bridge.jsonl",
)

EXPECTED_REPORT_KEYS = {
    "certification_boundary",
    "pre_window_diagnostics",
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

EXPECTED_REQUEST_KEYS = {
    "request_seq",
    "observation_tick",
    "time_s",
    "committed_actions",
}


@pytest.fixture(autouse=True)
def isolated_smoke_driver_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def valid_smoke_config(**updates):
    values = {
        "duration_s": 15.0,
        "ros_domain_id": 42,
        "unitree_domain_id": 42,
        "wbc_interface": "lo",
        "camera_host": "127.0.0.1",
        "camera_port": 15555,
        "policy_host": "127.0.0.1",
        "policy_port": 22086,
        "control_token": "smoke-control-token",
        "policy_ready_json": "outputs/psi0-smoke-policy-ready.json",
        "camera_ready_json": "outputs/psi0-smoke-camera-ready.json",
        "bridge_metrics_jsonl": "outputs/psi0-smoke-bridge.jsonl",
        "smoke_report_json": "outputs/psi0-smoke-report.json",
        "output_dir": "outputs",
    }
    values.update(updates)
    return SmokeConfig(**values)


def test_smoke_driver_import_has_no_ros_or_wbc_dependency(monkeypatch):
    forbidden = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "rclpy" or name.startswith("scripts.psi0_simple_real_bridge"):
            forbidden.append(name)
            raise AssertionError(f"forbidden import: {name}")
        return original_import(name, *args, **kwargs)

    module_name = "isolated_smoke_driver_import_test"
    spec = importlib.util.spec_from_file_location(module_name, smoke_driver.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    assert forbidden == []


def test_default_popen_is_resolved_at_launch_time(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(smoke_driver.subprocess, "Popen", sentinel)
    assert smoke_driver.DefaultSmokeHooks().resolve_popen() is sentinel


def test_smoke_launch_plan_is_immutable_exact_and_normal_latency_is_50ms():
    config = valid_smoke_config()
    plan = build_launch_plan(config)
    with pytest.raises(FrozenInstanceError):
        config.duration_s = 12.0
    with pytest.raises(FrozenInstanceError):
        plan.children = ()
    assert tuple(child.name for child in plan.children) == (
        "wbc",
        "camera",
        "policy",
        "bridge",
    )
    assert plan.children[0].argv == EXPECTED_WBC_ARGS
    expected_environment = dict(os.environ)
    parent_pythonpath = expected_environment.get("PYTHONPATH")
    expected_environment.update(
        {
            "ROS_DOMAIN_ID": "42",
            "UNITREE_DOMAIN_ID": "42",
            "PYTHONPATH": (
                "third_party/decoupled_wbc"
                if not parent_pythonpath
                else os.pathsep.join(("third_party/decoupled_wbc", parent_pythonpath))
            ),
        }
    )
    assert [child.argv for child in plan.children] == [
        EXPECTED_WBC_ARGS,
        EXPECTED_CAMERA_ARGS,
        EXPECTED_POLICY_ARGS,
        EXPECTED_BRIDGE_ARGS,
    ]
    assert [dict(child.env) for child in plan.children] == [expected_environment] * 4
    assert [child.use_pty for child in plan.children] == [False, False, False, True]
    assert [child.ready_json for child in plan.children] == [
        None,
        "outputs/psi0-smoke-camera-ready.json",
        "outputs/psi0-smoke-policy-ready.json",
        None,
    ]
    camera = plan.child("camera")
    assert camera.argv[camera.argv.index("--host") + 1] == "127.0.0.1"
    assert camera.argv[camera.argv.index("--port") + 1] == "15555"
    assert camera.argv[camera.argv.index("--ready-json") + 1] == (
        "outputs/psi0-smoke-camera-ready.json"
    )
    policy = plan.child("policy")
    assert policy.argv[policy.argv.index("--normal-latency-s") + 1] == "0.05"
    assert policy.argv[policy.argv.index("--host") + 1] == "127.0.0.1"
    assert policy.argv[policy.argv.index("--port") + 1] == "22086"
    assert policy.argv[policy.argv.index("--control-token") + 1] == (
        "smoke-control-token"
    )
    assert policy.argv[policy.argv.index("--ready-json") + 1] == (
        "outputs/psi0-smoke-policy-ready.json"
    )
    bridge = plan.child("bridge")
    assert bridge.use_pty is True
    assert bridge.argv[bridge.argv.index("--mode") + 1] == "sim-control"
    assert bridge.argv[bridge.argv.index("--policy-contract") + 1] == (
        "scripts/tests/fixtures/psi0_policy_contract_test_v2.json"
    )
    assert bridge.argv[bridge.argv.index("--camera-source-key") + 1] == (
        "rgb_head_stereo_left"
    )
    assert bridge.argv[bridge.argv.index("--camera-color-order") + 1] == "rgb"
    assert bridge.argv[bridge.argv.index("--metrics-jsonl") + 1] == (
        "outputs/psi0-smoke-bridge.jsonl"
    )
    assert all(child.env["ROS_DOMAIN_ID"] == "42" for child in plan.children)
    assert all(child.env["UNITREE_DOMAIN_ID"] == "42" for child in plan.children)


def test_launch_plan_prepends_required_pythonpath_and_preserves_parent(monkeypatch):
    parent_pythonpath = os.pathsep.join(("/parent/site-packages", "/parent/dist"))
    monkeypatch.setenv("PYTHONPATH", parent_pythonpath)
    plan = build_launch_plan(valid_smoke_config())
    expected = os.pathsep.join(("third_party/decoupled_wbc", parent_pythonpath))
    assert [child.env["PYTHONPATH"] for child in plan.children] == [expected] * 4


@pytest.mark.parametrize("parent_pythonpath", [None, ""])
def test_launch_plan_uses_exact_required_pythonpath_without_nonempty_parent(
    monkeypatch, parent_pythonpath
):
    if parent_pythonpath is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", parent_pythonpath)
    plan = build_launch_plan(valid_smoke_config())
    assert [child.env["PYTHONPATH"] for child in plan.children] == [
        "third_party/decoupled_wbc"
    ] * 4


class FakeControlResponse:
    status_code = 202

    def __init__(self, payload=None):
        self.payload = payload or {"armed_request_seq": 17, "latency_s": 0.30}

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class RecordingControlSession:
    def __init__(self, response=None):
        self.response = response or FakeControlResponse()
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_smoke_arms_delay_through_cross_process_control_endpoint():
    session = RecordingControlSession()
    armed_seq = arm_next_policy_delay(
        "127.0.0.1",
        22086,
        "smoke-control-token",
        0.30,
        session=session,
    )
    assert armed_seq == 17
    assert session.calls == [
        (
            "http://127.0.0.1:22086/test-control/arm-next-delay",
            {
                "json": {"token": "smoke-control-token", "latency_s": 0.30},
                "timeout": 0.5,
            },
        )
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"armed_request_seq": True, "latency_s": 0.30},
        {"armed_request_seq": 17, "latency_s": 0.3, "extra": None},
        {"armed_request_seq": 17, "latency_s": 1},
        {"armed_request_seq": 17, "latency_s": 0.31},
        [17, 0.30],
    ],
)
def test_delay_control_response_schema_and_types_are_strict(payload):
    with pytest.raises(SmokeSafetyError):
        arm_next_policy_delay(
            "127.0.0.1",
            22086,
            "smoke-control-token",
            0.30,
            session=RecordingControlSession(FakeControlResponse(payload)),
        )


class StaticWbcConfigClient:
    def __init__(self, payload):
        self.payload = payload
        self.timeouts = []

    def get_config(self, timeout_s):
        self.timeouts.append(timeout_s)
        return self.payload


def test_smoke_wbc_preflight_reads_the_producer_top_level_payload():
    payload = {"env_type": "sim", "interface": "lo", "model_contract": {}}
    client = StaticWbcConfigClient(payload)
    result = default_wbc_preflight(
        10.0,
        client_factory=lambda *args, **kwargs: client,
        clock=lambda: 9.0,
    )
    assert result is payload
    assert client.timeouts == [1.0]


def test_smoke_wbc_preflight_passes_full_ten_second_budget_to_client():
    payload = {"env_type": "sim", "interface": "lo"}
    client = StaticWbcConfigClient(payload)
    result = default_wbc_preflight(
        20.0,
        client_factory=lambda *args, **kwargs: client,
        clock=lambda: 10.0,
    )
    assert result is payload
    assert client.timeouts == [10.0]


def smoke_action_bounds():
    lower = np.r_[np.full(31, -1.0), 0.20, np.zeros(4)].astype(np.float32)
    upper = np.r_[np.full(31, 1.0), 0.74, np.zeros(4)].astype(np.float32)
    return lower, upper


def bounded_hold_action(value=0.0):
    action = np.full(36, value, np.float32)
    action[31] = 0.74
    action[32:36] = 0.0
    return action.tolist()


def startup_tick(tick, monotonic_s, **updates):
    record = {
        "event": "tick",
        "tick": tick,
        "monotonic_s": float(monotonic_s),
        "state": "paused",
        "published": True,
        "source_kind": "hold",
        "input_valid": True,
        "input_error": None,
        "psi0_action": bounded_hold_action(),
    }
    record.update(updates)
    return record


def test_smoke_action_bounds_follow_connected_wbc_joint_names():
    ordered = (*PSI0_ACTION_JOINT_NAMES, *PSI0_WAIST_ACTION_NAMES)
    names = tuple(f"lower_{index}" for index in range(12)) + ordered
    lower = [float(-index - 1) for index in range(43)]
    upper = [float(index + 1) for index in range(43)]
    payload = {
        "model_contract": {
            "robot_model": {
                "joint_names": list(names),
                "lower_position_limits": lower,
                "upper_position_limits": upper,
            }
        }
    }

    action_lower, action_upper = smoke_driver.smoke_action_bounds(payload)

    np.testing.assert_array_equal(action_lower[:31], lower[12:])
    np.testing.assert_array_equal(action_upper[:31], upper[12:])
    np.testing.assert_array_equal(
        action_lower[31:], np.asarray([0.20, 0.0, 0.0, 0.0, 0.0], np.float32)
    )
    np.testing.assert_array_equal(
        action_upper[31:], np.asarray([0.74, 0.0, 0.0, 0.0, 0.0], np.float32)
    )


def test_bridge_startup_boundary_uses_second_qualifying_tick_recorded_time():
    lower, upper = smoke_action_bounds()
    records = [
        {"event": "preflight_complete"},
        startup_tick(
            100,
            4.98,
            published=False,
            source_kind="none",
            input_valid=False,
            input_error="no valid state",
            psi0_action=None,
        ),
        startup_tick(101, 5.00),
        startup_tick(102, 5.02),
    ]

    boundary = smoke_driver.find_certification_boundary(records, lower, upper)

    assert boundary.tick == 102
    assert boundary.monotonic_s == 5.02


@pytest.mark.parametrize(
    "first_updates,second_updates",
    [
        ({"state": "active"}, {}),
        ({"published": False}, {}),
        ({"source_kind": "policy"}, {}),
        ({"input_valid": False}, {}),
        ({"psi0_action": bounded_hold_action(1.01)}, {}),
        ({"psi0_action": [*bounded_hold_action()[:32], 0.1, 0.0, 0.0, 0.0]}, {}),
        ({}, {"psi0_action": bounded_hold_action(0.1)}),
        ({}, {"tick": 103}),
    ],
)
def test_bridge_startup_boundary_rejects_nonqualifying_or_nonconsecutive_pair(
    first_updates, second_updates
):
    lower, upper = smoke_action_bounds()
    first = startup_tick(101, 5.00)
    first.update(first_updates)
    second = startup_tick(102, 5.02)
    second.update(second_updates)
    records = [
        {"event": "preflight_complete"},
        first,
        second,
    ]

    assert smoke_driver.find_certification_boundary(records, lower, upper) is None


def test_bridge_startup_wait_returns_recorded_boundary_not_poll_time(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text("placeholder", encoding="utf-8")
    lower, upper = smoke_action_bounds()
    records = [
        {"event": "preflight_complete"},
        startup_tick(101, 5.00),
        startup_tick(102, 5.02),
    ]

    boundary = smoke_driver.wait_bridge_startup(
        path,
        lower,
        upper,
        deadline=10.0,
        clock=lambda: 7.5,
        read_metrics=lambda _path: records,
    )

    assert boundary == smoke_driver.CertificationBoundary(102, 5.02)


def test_smoke_wbc_preflight_rejects_a_nonexistent_wrapper_schema():
    client = StaticWbcConfigClient(
        {"control_loop_config": {"env_type": "sim", "interface": "lo"}}
    )
    with pytest.raises(SmokeSafetyError, match="sim/loopback"):
        default_wbc_preflight(
            10.0,
            client_factory=lambda *args, **kwargs: client,
            clock=lambda: 9.0,
        )


def test_two_exact_smoke_commands_allocate_unique_preserved_runs(tmp_path):
    nonces = iter(("run-a", "run-b"))
    kwargs = {
        "now_ns": lambda: 123456789,
        "token_hex": lambda length: next(nonces),
    }
    first = allocate_smoke_run_directory(tmp_path, **kwargs)
    marker = first / "psi0-smoke-report.json"
    marker.write_text("first run", encoding="utf-8")
    second = allocate_smoke_run_directory(tmp_path, **kwargs)
    assert first == tmp_path / "psi0-smoke-123456789-run-a"
    assert second == tmp_path / "psi0-smoke-123456789-run-b"
    assert marker.read_text(encoding="utf-8") == "first run"
    assert list(second.iterdir()) == []


def test_the_exact_task_13_cli_can_be_invoked_twice(monkeypatch, tmp_path):
    run_directories = iter((tmp_path / "run-1", tmp_path / "run-2"))
    allocation_roots = []
    launched = []

    def allocate(root):
        allocation_roots.append(root)
        output = next(run_directories)
        output.mkdir()
        return output

    monkeypatch.setattr(smoke_driver, "allocate_smoke_run_directory", allocate)
    monkeypatch.setattr(smoke_driver, "launch", launched.append)
    monkeypatch.setattr(
        smoke_driver.secrets, "token_hex", lambda length: "a" * (2 * length)
    )
    argv = [
        "--duration-s",
        "15",
        "--unitree-domain-id",
        "42",
        "--camera-port",
        "15555",
        "--policy-port",
        "22086",
    ]
    assert smoke_driver.main(argv) == 0
    assert smoke_driver.main(argv) == 0
    assert allocation_roots == ["outputs", "outputs"]
    assert [config.output_dir for config in launched] == [
        str(tmp_path / "run-1"),
        str(tmp_path / "run-2"),
    ]
    assert all(config.control_token == "a" * 32 for config in launched)


def test_cli_rejects_an_explicitly_empty_injected_control_token(monkeypatch):
    monkeypatch.setattr(smoke_driver, "launch", build_launch_plan)
    monkeypatch.setattr(smoke_driver.secrets, "token_hex", lambda length: "b" * 8)
    with pytest.raises(SmokeSafetyError, match="control token"):
        smoke_driver.main(
            [
                "--duration-s",
                "15",
                "--unitree-domain-id",
                "42",
                "--camera-port",
                "15555",
                "--policy-port",
                "22086",
                "--control-token",
                "",
            ]
        )


def test_collector_uses_lowercase_fault_and_old_generation_counter():
    zero = bounded_hold_action()
    records = [
        {
            "event": "preflight_complete",
            "connection_evidence": [
                {
                    "component": "wbc",
                    "transport": "dds-service-response",
                    "endpoint": "lo",
                    "real_interface": False,
                    "observed_at": 0.5,
                },
                {
                    "component": "camera",
                    "transport": "decoded-frame",
                    "endpoint": "127.0.0.1",
                    "real_interface": False,
                    "observed_at": 0.6,
                },
                {
                    "component": "policy",
                    "transport": "http-contract-response",
                    "endpoint": "127.0.0.1",
                    "real_interface": False,
                    "observed_at": 0.7,
                },
            ],
            "real_interface_connections": 0,
        },
        {
            "event": "tick",
            "tick": -1,
            "published": True,
            "state": "paused",
            "source_kind": "hold",
            "psi0_action": zero,
            "monotonic_s": -0.02,
            "input_valid": True,
            "input_error": None,
            "worker_busy": False,
            "discarded_late_results": 0,
            "discarded_old_generation_results": 0,
        },
        {
            "event": "tick",
            "tick": 0,
            "published": True,
            "state": "paused",
            "source_kind": "hold",
            "psi0_action": zero,
            "monotonic_s": 0.0,
            "input_valid": True,
            "input_error": None,
            "worker_busy": False,
            "discarded_late_results": 0,
            "discarded_old_generation_results": 0,
        },
        {
            "event": "request",
            "request_seq": 17,
            "observation_tick": 100,
            "committed_actions": np.zeros((6, 36), np.float32).tolist(),
            "monotonic_s": 11.0,
        },
        {
            "event": "tick",
            "tick": 100,
            "published": True,
            "state": "active",
            "source_kind": "hold",
            "psi0_action": zero,
            "monotonic_s": 11.00,
            "worker_busy": True,
            "discarded_late_results": 0,
            "discarded_old_generation_results": 0,
        },
        {
            "event": "tick",
            "tick": 101,
            "published": True,
            "state": "fault",
            "source_kind": "hold",
            "psi0_action": zero,
            "monotonic_s": 11.02,
            "worker_busy": True,
            "discarded_late_results": 0,
            "discarded_old_generation_results": 1,
        },
        {
            "event": "tick",
            "tick": 199,
            "published": True,
            "state": "fault",
            "source_kind": "hold",
            "psi0_action": zero,
            "monotonic_s": 12.98,
            "worker_busy": False,
            "discarded_late_results": 0,
            "discarded_old_generation_results": 1,
        },
    ]
    assert measured_bridge_worker_idle_at(records, 0.0) == 12.98
    report = collect_smoke_report(
        records,
        scenario_started_at=0.0,
        certification_boundary=smoke_driver.CertificationBoundary(0, 0.0),
        armed_seq=17,
        delayed_record={"request_seq": 17, "applied_latency_s": 0.30},
        goal_counts_before=[0, 1],
        goal_counts_running=[1, 1],
        goal_counts_after=[0, 1],
        terminal_restored=True,
        ports_rebound=True,
        bridge_exit_code=0,
        live_children_after=[],
        child_exit_codes={"wbc": -2, "camera": 0, "policy": 0, "bridge": 0},
    )
    assert report["fault_at"] == 11.02
    assert report["requests"] == [
        {
            "request_seq": 17,
            "observation_tick": 100,
            "time_s": 11.0,
            "committed_actions": np.zeros((6, 36), np.float32).tolist(),
        }
    ]
    assert report["old_generation_results_discarded"] == 1
    assert report["late_results_discarded"] == 0
    records[0]["connection_evidence"][2]["real_interface"] = True
    with pytest.raises(SmokeSafetyError, match="differs from evidence"):
        measured_real_interface_connections(records)


@pytest.mark.parametrize(
    "updates",
    [
        {"duration_s": 14.99},
        {"wbc_interface": "eth0"},
        {"camera_host": "0.0.0.0"},
        {"policy_host": "192.168.1.2"},
        {"ros_domain_id": 41},
        {"unitree_domain_id": 0},
        {"camera_port": 0},
        {"policy_port": 0},
        {"control_token": ""},
        {"policy_ready_json": "../outside.json"},
    ],
)
def test_invalid_isolation_is_rejected_before_any_popen(monkeypatch, updates):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: calls.append((a, k)))
    with pytest.raises(SmokeSafetyError):
        launch(valid_smoke_config(**updates))
    assert calls == []


def test_preexisting_artifact_is_rejected_before_any_popen(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: calls.append((a, k)))
    output = smoke_driver.Path("outputs")
    output.mkdir()
    (output / "psi0-smoke-bridge.jsonl").write_text("occupied", encoding="utf-8")
    with pytest.raises(SmokeSafetyError, match="already exists"):
        launch(valid_smoke_config())
    assert calls == []


def test_cleanup_signals_and_reaps_only_recorded_process_groups():
    events = []
    owner = OwnedChildren(
        killpg=lambda pgid, sig: events.append(("killpg", pgid, sig)),
        waitpid=lambda pid, flags: events.append(("waitpid", pid, flags)) or (pid, 0),
    )
    owner.record(pid=101, pgid=101, name="wbc", argv=("wbc",), started_at=1.0)
    owner.record(pid=202, pgid=202, name="bridge", argv=("bridge",), started_at=2.0)
    owner.close()
    assert events == [
        ("killpg", 202, signal.SIGINT),
        ("waitpid", 202, 0),
        ("killpg", 101, signal.SIGINT),
        ("waitpid", 101, 0),
    ]


def test_owned_child_record_returns_and_resolves_the_recorded_process_group():
    owner = OwnedChildren(
        killpg=lambda pgid, sig: None, waitpid=lambda pid, flags: (pid, 0)
    )
    record = owner.record(
        pid=202,
        pgid=909,
        name="bridge",
        argv=("bridge",),
        started_at=2.0,
    )
    assert record.pgid == 909
    assert owner.process_group("bridge") == 909


def test_real_child_cleanup_uses_one_shared_absolute_deadline():
    now = [0.0]
    terminated = set()
    events = []

    class Process:
        def __init__(self, pgid):
            self.pgid = pgid

        def poll(self):
            return 0 if self.pgid in terminated else None

    def killpg(pgid, sig):
        events.append((pgid, sig, now[0]))
        if sig == signal.SIGTERM:
            terminated.add(pgid)

    owner = OwnedChildren(
        killpg=killpg,
        clock=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
        group_alive=lambda pgid: pgid not in terminated,
    )
    for pid in (101, 202, 303):
        owner.record(
            pid=pid,
            pgid=pid,
            name=str(pid),
            argv=(str(pid),),
            started_at=0.0,
            process=Process(pid),
        )
    owner.close(deadline=1.0)
    assert now[0] <= 1.0
    assert terminated == {101, 202, 303}
    assert owner.live_pids() == []
    assert owner.exit_codes() == {"101": 0, "202": 0, "303": 0}
    assert [event[:2] for event in events] == [
        (303, signal.SIGINT),
        (202, signal.SIGINT),
        (101, signal.SIGINT),
        (303, signal.SIGTERM),
        (202, signal.SIGTERM),
        (101, signal.SIGTERM),
    ]


def test_cleanup_tracks_a_live_descendant_after_the_recorded_leader_exits():
    now = [0.0]
    group_live = [True]
    signals = []

    class ExitedLeader:
        @staticmethod
        def poll():
            return 0

    owner = OwnedChildren(
        killpg=lambda pgid, sig: signals.append((pgid, sig)),
        clock=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
        group_alive=lambda pgid: group_live[0],
    )
    owner.record(
        pid=202,
        pgid=909,
        name="bridge",
        argv=("bridge",),
        started_at=0.0,
        process=ExitedLeader(),
    )

    assert owner.live_pids() == [202]
    with pytest.raises(SmokeSafetyError, match="live children"):
        owner.close(deadline=0.71)
    assert signals == [
        (909, signal.SIGINT),
        (909, signal.SIGTERM),
        (909, signal.SIGKILL),
    ]
    assert {pgid for pgid, _ in signals} == {909}
    assert owner.live_pids() == [202]
    assert owner._closed is False

    group_live[0] = False
    owner.close(deadline=1.0)
    assert owner.live_pids() == []
    assert owner._closed is True


def test_child_cleanup_continues_after_signal_failure_and_can_retry():
    now = [0.0]
    terminated = set()
    fail_303 = [True]
    events = []

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return 0 if self.pid in terminated else None

    def killpg(pgid, sig):
        events.append((pgid, sig))
        if pgid == 303 and fail_303[0]:
            raise OSError("injected signal failure")
        terminated.add(pgid)

    owner = OwnedChildren(
        killpg=killpg,
        clock=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
        group_alive=lambda pgid: pgid not in terminated,
    )
    for pid in (101, 202, 303):
        owner.record(
            pid=pid,
            pgid=pid,
            name=str(pid),
            argv=(str(pid),),
            started_at=0.0,
            process=Process(pid),
        )
    with pytest.raises(SmokeSafetyError, match="303"):
        owner.close(deadline=0.8)
    assert (202, signal.SIGINT) in events
    assert (101, signal.SIGINT) in events
    assert (303, signal.SIGKILL) in events
    assert owner.live_pids() == [303]
    assert owner._closed is False

    fail_303[0] = False
    owner.close(deadline=2.0)
    assert owner.live_pids() == []
    assert owner._closed is True


def test_resource_cleanup_closes_pty_and_logs_when_child_cleanup_raises():
    events = []

    class FailingOwner:
        def close(self, deadline):
            events.append(("owner", deadline))
            raise RuntimeError("child cleanup failed")

    class Log:
        def __init__(self, name):
            self.name = name

        def close(self):
            events.append(("log", self.name))

    with pytest.raises(SmokeSafetyError, match="child cleanup failed"):
        close_smoke_resources(
            FailingOwner(),
            (10, 11),
            (Log("a"), Log("b")),
            15.0,
            close_fd=lambda descriptor: events.append(("fd", descriptor)),
        )
    assert events == [
        ("owner", 15.0),
        ("fd", 10),
        ("fd", 11),
        ("log", "a"),
        ("log", "b"),
    ]


def test_resource_cleanup_retries_an_owner_that_still_has_live_children():
    calls = []

    class RetriableOwner:
        _closed = False

        def close(self, deadline):
            calls.append(deadline)
            if len(calls) == 1:
                raise RuntimeError("transient killpg failure")
            self._closed = True

    with pytest.raises(SmokeSafetyError, match="transient killpg failure"):
        close_smoke_resources(RetriableOwner(), (), (), 15.0)
    assert calls == [15.0, 15.0]


def test_exhausted_cleanup_sigkills_every_live_group_and_remains_retryable():
    now = [1.0]
    allow_termination = [False]
    events = []

    class Process:
        def __init__(self, pid):
            self.pid = pid
            self.returncode = None

        def poll(self):
            return self.returncode

    processes = {pid: Process(pid) for pid in (101, 202, 303)}

    def killpg(pgid, sig):
        events.append((pgid, sig))
        if allow_termination[0]:
            processes[pgid].returncode = -int(sig)

    owner = OwnedChildren(
        killpg=killpg,
        clock=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
        group_alive=lambda pgid: processes[pgid].returncode is None,
    )
    for pid in processes:
        owner.record(
            pid=pid,
            pgid=pid,
            name=str(pid),
            argv=(str(pid),),
            started_at=0.0,
            process=processes[pid],
        )
    with pytest.raises(SmokeSafetyError, match="deadline exhausted"):
        close_smoke_resources(owner, (), (), deadline=1.0)
    assert [pgid for pgid, sig in events if sig == signal.SIGKILL] == [
        303,
        202,
        101,
        303,
        202,
        101,
    ]
    assert owner.live_pids() == [101, 202, 303]
    assert owner._closed is False

    allow_termination[0] = True
    owner.close(deadline=2.0)
    assert owner.live_pids() == []
    assert owner._closed is True


def test_port_rebind_checks_do_not_short_circuit_after_a_failure():
    calls = []

    def rebind(port):
        calls.append(port)
        return port == 22086

    result = smoke_driver.check_all_ports_rebound((15555, 22086), rebind)
    assert result.ok is False
    assert result.failures == ("port 15555 did not rebind",)
    assert calls == [15555, 22086]


def test_port_rebind_checks_continue_after_an_exception_and_return_diagnostics():
    calls = []

    def rebind(port):
        calls.append(port)
        if port == 15555:
            raise OSError("injected bind failure")
        return True

    result = smoke_driver.check_all_ports_rebound((15555, 22086), rebind)
    assert result.ok is False
    assert result.failures == ("port 15555: injected bind failure",)
    assert calls == [15555, 22086]


def _passing_metrics_records():
    zero = bounded_hold_action()
    records = [
        {
            "event": "preflight_complete",
            "connection_evidence": [
                {
                    "component": component,
                    "transport": transport,
                    "endpoint": endpoint,
                    "real_interface": False,
                    "observed_at": float(index),
                }
                for index, (component, transport, endpoint) in enumerate(
                    (
                        ("wbc", "dds-service-response", "lo"),
                        ("camera", "decoded-frame", "127.0.0.1"),
                        ("policy", "http-contract-response", "127.0.0.1"),
                    ),
                    1,
                )
            ],
            "real_interface_connections": 0,
        }
    ]
    records.extend(
        (
            {
                "event": "tick",
                "tick": 98,
                "published": False,
                "state": "paused",
                "source_kind": "none",
                "psi0_action": None,
                "monotonic_s": -0.04,
                "input_valid": False,
                "input_error": "no valid state",
                "worker_busy": False,
                "discarded_late_results": 0,
                "discarded_old_generation_results": 0,
            },
            {
                "event": "tick",
                "tick": 99,
                "published": True,
                "state": "paused",
                "source_kind": "hold",
                "psi0_action": zero,
                "monotonic_s": -0.02,
                "input_valid": True,
                "input_error": None,
                "worker_busy": False,
                "discarded_late_results": 0,
                "discarded_old_generation_results": 0,
            },
        )
    )
    # P=30, s=24, d=6 at 50 Hz: R0 is the first active tick at 3.00 s,
    # each successor is 24 ticks (0.48 s) later, and R17 is at 11.16 s.
    for request_seq in range(18):
        tick = 250 + request_seq * 24
        monotonic_s = 3.0 + request_seq * 0.48
        records.append(
            {
                "event": "request",
                "request_seq": request_seq,
                "observation_tick": tick,
                "committed_actions": np.tile(
                    np.asarray(zero, np.float32), (6, 1)
                ).tolist(),
                "monotonic_s": monotonic_s,
            }
        )
    for offset in range(650):
        monotonic_s = 0.02 * offset
        fault = monotonic_s >= 11.28
        r0_committed_hold = 3.0 <= monotonic_s < 3.12
        if monotonic_s < 3.0 or r0_committed_hold or fault:
            source_kind = "hold"
        else:
            source_kind = "policy"
        records.append(
            {
                "event": "tick",
                "tick": 100 + offset,
                "published": True,
                "state": (
                    "fault" if fault else "paused" if monotonic_s < 3.0 else "active"
                ),
                "source_kind": source_kind,
                "psi0_action": zero,
                "monotonic_s": monotonic_s,
                "input_valid": True,
                "input_error": None,
                "worker_busy": not (12.9 <= monotonic_s <= 13.0),
                "discarded_late_results": 0,
                "discarded_old_generation_results": 1 if fault else 0,
            }
        )
    return records


def test_collector_separates_pre_window_diagnostics_at_recorded_boundary():
    report = collect_smoke_report(
        _passing_metrics_records(),
        scenario_started_at=0.0,
        certification_boundary=smoke_driver.CertificationBoundary(100, 0.0),
        armed_seq=17,
        delayed_record={"request_seq": 17, "applied_latency_s": 0.30},
        goal_counts_before=[0, 1],
        goal_counts_running=[1, 1],
        goal_counts_after=[0, 1],
        terminal_restored=True,
        ports_rebound=True,
        bridge_exit_code=0,
        live_children_after=[],
        child_exit_codes={"wbc": -2, "camera": 0, "policy": 0, "bridge": 0},
        live_threads_after=[],
    )

    assert report["certification_boundary"] == {
        "tick": 100,
        "monotonic_s": 0.0,
    }
    assert [entry["tick"] for entry in report["pre_window_diagnostics"]] == [98, 99]
    assert report["pre_window_diagnostics"][0]["published"] is False
    assert report["pre_window_diagnostics"][0]["input_error"] == "no valid state"
    assert report["pre_window_diagnostics"][1]["published"] is True
    assert report["published_ticks"][0]["tick"] == 100
    assert report["published_ticks"][0]["time_s"] == 0.0
    assert report["unpublished_ticks"] == []


def test_launch_preserves_both_operation_and_cleanup_failures(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    config = valid_smoke_config(
        output_dir=str(output),
        policy_ready_json=str(output / "psi0-smoke-policy-ready.json"),
        camera_ready_json=str(output / "psi0-smoke-camera-ready.json"),
        bridge_metrics_jsonl=str(output / "psi0-smoke-bridge.jsonl"),
        smoke_report_json=str(output / "psi0-smoke-report.json"),
    )
    primary = RuntimeError("primary readiness sentinel")
    events = []
    next_pid = [100]

    class Log:
        def __init__(self, name):
            self.name = name

        def close(self):
            events.append(("close_log", self.name))

    def fake_open(path, *args, **kwargs):
        log = Log(str(path))
        events.append(("open_log", log.name))
        return log

    monkeypatch.setattr(smoke_driver.Path, "open", fake_open)

    class Process:
        def __init__(self):
            next_pid[0] += 1
            self.pid = next_pid[0]

        @staticmethod
        def poll():
            return None

    class CleanupFailureOwner:
        _closed = False

        @staticmethod
        def record(**kwargs):
            events.append(("record", kwargs["pgid"]))

        @staticmethod
        def close(deadline):
            events.append(("owner_close", deadline))
            raise RuntimeError("cleanup sentinel")

    owner = CleanupFailureOwner()

    class Hooks:
        clock = staticmethod(lambda: 0.0)
        sleep = staticmethod(lambda duration: None)
        resolve_popen = staticmethod(lambda: lambda argv, **kwargs: Process())
        owner_factory = staticmethod(lambda **kwargs: owner)
        wbc_preflight = staticmethod(lambda deadline: None)
        action_bounds = staticmethod(lambda payload: smoke_action_bounds())
        goal_counts = staticmethod(lambda: [0, 1])
        wait_ready = staticmethod(lambda *args: None)
        openpty = staticmethod(lambda: (10, 11))
        terminal_get = staticmethod(lambda fd: ["original"])

        @staticmethod
        def wait_bridge(path, lower, upper, deadline, clock):
            raise primary

        close_fd = staticmethod(lambda fd: events.append(("close_fd", fd)))

    with pytest.raises(SmokeSafetyError) as caught:
        launch(config, hooks=Hooks())
    assert [event for event in events if event[0] == "owner_close"] == [
        ("owner_close", 1.0),
        ("owner_close", 1.0),
    ]
    assert [event for event in events if event[0] == "close_fd"] == [
        ("close_fd", 10),
        ("close_fd", 11),
    ]
    assert len([event for event in events if event[0] == "close_log"]) == 4
    assert "primary readiness sentinel" in str(caught.value)
    assert "cleanup sentinel" in str(caught.value)
    assert caught.value.__cause__ is primary


def test_launch_is_injected_ordered_bounded_and_uses_shared_deadline(
    tmp_path, monkeypatch
):
    output = tmp_path / "run"
    output.mkdir()
    config = valid_smoke_config(
        output_dir=str(output),
        policy_ready_json=str(output / "psi0-smoke-policy-ready.json"),
        camera_ready_json=str(output / "psi0-smoke-camera-ready.json"),
        bridge_metrics_jsonl=str(output / "psi0-smoke-bridge.jsonl"),
        smoke_report_json=str(output / "psi0-smoke-report.json"),
    )
    events = []
    now = [0.0]
    next_pid = [100]
    processes = {}
    popen_calls = []

    class Process:
        def __init__(self, name):
            next_pid[0] += 1
            self.pid = next_pid[0]
            self.name = name
            self.returncode = None
            processes[self.pid] = self

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            events.append(("bridge_wait", timeout))
            self.returncode = 0
            return 0

    def fake_popen(argv, **kwargs):
        name = ("wbc", "camera", "policy", "bridge")[len(processes)]
        events.append(("popen", name))
        popen_calls.append((name, tuple(argv), dict(kwargs)))
        return Process(name)

    monkeypatch.setattr(smoke_driver.subprocess, "Popen", fake_popen)

    class Hooks:
        clock = staticmethod(lambda: now[0])
        sleep = staticmethod(
            lambda duration: (
                events.append(("sleep", duration)),
                now.__setitem__(0, now[0] + duration),
            )[-1]
        )

        @staticmethod
        def resolve_popen():
            events.append(("resolve_popen",))
            return smoke_driver.DefaultSmokeHooks().resolve_popen()

        @staticmethod
        def popen(*args, **kwargs):
            raise AssertionError("launch bypassed dynamic Popen resolution")

        @staticmethod
        def owner_factory(**kwargs):
            def killpg(pgid, sig):
                events.append(("cleanup_signal", pgid, sig, now[0]))
                process = processes[pgid]
                if process.returncode is None:
                    process.returncode = -int(sig)

            class RecordingOwner(OwnedChildren):
                def record(self, **record_kwargs):
                    record = super().record(**record_kwargs)
                    events.append(
                        (
                            "record",
                            record.name,
                            record.pid,
                            record.pgid,
                        )
                    )
                    return record

                def close(self, deadline=None):
                    events.append(("cleanup_deadline", deadline))
                    return super().close(deadline)

            return RecordingOwner(
                killpg=killpg,
                group_alive=lambda pgid: processes[pgid].returncode is None,
                **kwargs,
            )

        @staticmethod
        def wbc_preflight(deadline):
            events.append(("wbc_ready", deadline))
            return {"env_type": "sim", "interface": "lo"}

        action_bounds = staticmethod(lambda payload: smoke_action_bounds())

        goal_results = iter(([0, 1], [1, 1], [0, 1]))

        @staticmethod
        def goal_counts():
            result = list(next(Hooks.goal_results))
            events.append(("graph", result))
            return result

        @staticmethod
        def wait_ready(path, host, port, deadline, clock):
            events.append(("ready", port, deadline))
            return {"host": host, "port": port}

        @staticmethod
        def openpty():
            events.append(("openpty",))
            return 10, 11

        @staticmethod
        def terminal_get(fd):
            events.append(("terminal", fd))
            return ["original"]

        @staticmethod
        def wait_bridge(path, lower, upper, deadline, clock):
            expected_lower, expected_upper = smoke_action_bounds()
            np.testing.assert_array_equal(lower, expected_lower)
            np.testing.assert_array_equal(upper, expected_upper)
            events.append(("bridge_ready", deadline))
            now[0] = 0.5
            return smoke_driver.CertificationBoundary(100, 0.0)

        @staticmethod
        def write(fd, data):
            events.append(("write", fd, data, now[0]))
            return len(data)

        @staticmethod
        def arm_delay(host, port, token, latency):
            events.append(("arm", latency, now[0]))
            return 17

        @staticmethod
        def policy_status(config):
            events.append(("policy_status", now[0]))
            return {
                "last_started_request_seq": 17,
                "active_requests": 0,
                "max_concurrent_requests": 1,
                "records": [{"request_seq": 17, "applied_latency_s": 0.30}],
            }

        @staticmethod
        def signal_group(pgid, sig):
            events.append(("bridge_signal", pgid, sig, now[0]))
            processes[pgid].returncode = 0

        getpgid = staticmethod(
            lambda pid: (_ for _ in ()).throw(
                AssertionError("PGID must not be re-resolved")
            )
        )

        @staticmethod
        def read_metrics(path):
            events.append(("read_metrics", now[0]))
            return _passing_metrics_records()

        @staticmethod
        def measure_worker_idle(records, scenario_started_at):
            events.append(("bridge_idle", now[0]))
            return measured_bridge_worker_idle_at(records, scenario_started_at)

        port_rebind = staticmethod(lambda port: events.append(("rebind", port)) or True)
        close_fd = staticmethod(lambda fd: events.append(("close_fd", fd)))
        live_threads = staticmethod(lambda: [])

    plan = build_launch_plan(config)
    report = launch(config, hooks=Hooks())
    assert [call[0] for call in popen_calls] == [
        "wbc",
        "camera",
        "policy",
        "bridge",
    ]
    for name, argv, kwargs in popen_calls:
        spec = plan.child(name)
        assert argv == spec.argv
        assert kwargs["env"] == dict(spec.env)
        assert kwargs["stdin"] == (11 if name == "bridge" else None)
        assert kwargs["stdout"].name == str(output / f"psi0-smoke-{name}.log")
        assert kwargs["stderr"] is subprocess.STDOUT
        assert kwargs["start_new_session"] is True

    first_sleep = next(
        index for index, event in enumerate(events) if event[0] == "sleep"
    )
    assert events[:first_sleep] == [
        ("resolve_popen",),
        ("popen", "wbc"),
        ("record", "wbc", 101, 101),
        ("wbc_ready", 10.0),
        ("graph", [0, 1]),
        ("popen", "camera"),
        ("record", "camera", 102, 102),
        ("ready", 15555, 1.0),
        ("popen", "policy"),
        ("record", "policy", 103, 103),
        ("ready", 22086, 1.0),
        ("openpty",),
        ("terminal", 11),
        ("popen", "bridge"),
        ("record", "bridge", 104, 104),
        ("bridge_ready", 10.0),
    ]
    startup_names = [
        event[0]
        for event in events
        if event[0]
        in {
            "resolve_popen",
            "popen",
            "record",
            "wbc_ready",
            "graph",
            "ready",
            "openpty",
            "terminal",
            "bridge_ready",
        }
    ]
    assert startup_names[:17] == [
        "resolve_popen",
        "popen",
        "record",
        "wbc_ready",
        "graph",
        "popen",
        "record",
        "ready",
        "popen",
        "record",
        "ready",
        "openpty",
        "terminal",
        "popen",
        "record",
        "bridge_ready",
        "graph",
    ]
    event_names = [event[0] for event in events]
    assert event_names.index("wbc_ready") < event_names.index("graph")
    assert event_names.index("graph") < event_names.index("ready")
    assert [event for event in events if event[0] == "wbc_ready"] == [
        ("wbc_ready", 10.0)
    ]
    assert [event for event in events if event[0] == "ready"] == [
        ("ready", 15555, 1.0),
        ("ready", 22086, 1.0),
    ]
    assert [event for event in events if event[0] == "bridge_ready"] == [
        ("bridge_ready", 10.0)
    ]
    assert [event for event in events if event[0] == "write"] == [
        ("write", 10, b"p", 3.0)
    ]
    assert [event for event in events if event[0] == "arm"] == [("arm", 0.30, 11.0)]
    assert [event for event in events if event[0] == "policy_status"] == [
        ("policy_status", 13.0)
    ]
    second_13_events = [
        event[0]
        for event in events
        if event[0] in {"read_metrics", "bridge_idle", "policy_status", "bridge_signal"}
    ]
    assert second_13_events[:4] == [
        "read_metrics",
        "bridge_idle",
        "policy_status",
        "bridge_signal",
    ]
    assert [event for event in events if event[0] == "read_metrics"] == [
        ("read_metrics", 13.0),
        ("read_metrics", 13.0),
    ]
    assert [event for event in events if event[0] == "bridge_signal"] == [
        ("bridge_signal", processes[104].pid, signal.SIGINT, 13.0)
    ]
    assert [event for event in events if event[0] == "bridge_wait"] == [
        ("bridge_wait", 2.0)
    ]
    assert [event for event in events if event[0] == "cleanup_deadline"] == [
        ("cleanup_deadline", 15.0)
    ]
    assert max(event[3] for event in events if event[0] == "cleanup_signal") <= 15.0
    assert report["validation"] == {"ok": True, "failures": []}
    assert report["certification_boundary"] == {
        "tick": 100,
        "monotonic_s": 0.0,
    }
    assert [entry["tick"] for entry in report["pre_window_diagnostics"]] == [98, 99]
    assert report["child_exit_codes"] == {
        "wbc": -2,
        "camera": -2,
        "policy": -2,
        "bridge": 0,
    }


def passing_smoke_report():
    publish_times = [index * 0.02 for index in range(650)]
    published_ticks = [
        {
            "tick": 100 + index,
            "time_s": monotonic_s,
            "source_kind": (
                "hold" if monotonic_s < 3.12 or monotonic_s >= 11.28 else "policy"
            ),
            "state": (
                "fault"
                if monotonic_s >= 11.28
                else "paused"
                if monotonic_s < 3.0
                else "active"
            ),
        }
        for index, monotonic_s in enumerate(publish_times)
    ]
    executed_actions = []
    by_tick = {}
    for tick, timeline_entry in zip(range(100, 750), published_ticks):
        action = np.zeros(36, np.float32)
        action[31] = 0.74
        if timeline_entry["source_kind"] == "policy":
            action[0] = tick / 10_000.0
        by_tick[tick] = action
        executed_actions.append({"tick": tick, "post_slew_action": action.tolist()})
    return {
        "certification_boundary": {"tick": 100, "monotonic_s": 1000.0},
        "pre_window_diagnostics": [
            {
                "tick": 98,
                "monotonic_s": 999.96,
                "published": False,
                "state": "paused",
                "source_kind": "none",
                "input_valid": False,
                "input_error": "no valid state",
                "psi0_action": None,
            },
            {
                "tick": 99,
                "monotonic_s": 999.98,
                "published": True,
                "state": "paused",
                "source_kind": "hold",
                "input_valid": True,
                "input_error": None,
                "psi0_action": by_tick[100].tolist(),
            },
        ],
        "steady_phases": [
            {"publish_times": [value for value in publish_times if value < 11.28]},
            {"publish_times": [value for value in publish_times if value >= 11.28]},
        ],
        "published_ticks": published_ticks,
        "unpublished_ticks": [],
        "requests": [
            {
                "request_seq": request_seq,
                "observation_tick": tick,
                "time_s": 3.0 + request_seq * 0.48,
                "committed_actions": np.stack(
                    [by_tick[index] for index in range(tick, tick + 6)]
                ).tolist(),
            }
            for request_seq, tick in (
                (request_seq, 250 + request_seq * 24) for request_seq in range(18)
            )
        ],
        "executed_actions": executed_actions,
        "blocked_main_loop_max_gap_s": 0.02,
        "delayed_request_started_at": 11.16,
        "armed_request_seq": 17,
        "delayed_request_record": {"request_seq": 17, "applied_latency_s": 0.30},
        "fault_at": 11.28,
        "first_fault_goal_navigation": [0.0, 0.0, 0.0, 0.0],
        "policy_actions_after_fault": 0,
        "old_generation_results_discarded": 1,
        "late_results_discarded": 0,
        "worker_idle_at_s": 12.8,
        "goal_counts_before": [0, 1],
        "goal_counts_running": [1, 1],
        "goal_counts_after": [0, 1],
        "live_children_after": [],
        "child_exit_codes": {"wbc": -2, "camera": 0, "policy": 0, "bridge": 0},
        "bridge_exit_code": 0,
        "live_threads_after": [],
        "terminal_restored": True,
        "ports_rebound": True,
        "real_interface_connections": 0,
        "extra_goal_publishers": 0,
    }


def test_passing_smoke_report_meets_every_bound():
    report = passing_smoke_report()
    assert set(report) == EXPECTED_REPORT_KEYS
    assert all(set(entry) == EXPECTED_REQUEST_KEYS for entry in report["requests"])
    assert [
        (entry["request_seq"], entry["observation_tick"], entry["time_s"])
        for entry in report["requests"]
    ] == [
        (request_seq, 250 + request_seq * 24, 3.0 + request_seq * 0.48)
        for request_seq in range(18)
    ]
    result = validate_smoke_report(report)
    assert result.ok is True
    assert result.failures == ()


def test_smoke_rejects_unattested_or_misaligned_certification_boundary():
    mutations = []

    wrong_tick = passing_smoke_report()
    wrong_tick["certification_boundary"]["tick"] = 101
    mutations.append((wrong_tick, "certification_boundary"))

    nonfinite_time = passing_smoke_report()
    nonfinite_time["certification_boundary"]["monotonic_s"] = float("nan")
    mutations.append((nonfinite_time, "certification_boundary"))

    missing_first_qualifier = passing_smoke_report()
    missing_first_qualifier["pre_window_diagnostics"] = []
    mutations.append((missing_first_qualifier, "pre_window_diagnostics"))

    changed_hold = passing_smoke_report()
    changed_hold["pre_window_diagnostics"][-1]["psi0_action"][0] = 0.1
    mutations.append((changed_hold, "certification_boundary_action"))

    for report, expected in mutations:
        result = validate_smoke_report(report)
        assert result.ok is False
        assert any(expected in failure for failure in result.failures)


def test_smoke_rejects_nonexact_top_level_and_nested_report_schemas():
    mutations = []

    secret_extra = passing_smoke_report()
    secret_extra["control_token"] = "must-not-be-reported"
    mutations.append((secret_extra, "report_schema"))

    arbitrary_extra = passing_smoke_report()
    arbitrary_extra["unexpected"] = None
    mutations.append((arbitrary_extra, "report_schema"))

    missing_top_level = passing_smoke_report()
    missing_top_level.pop("extra_goal_publishers")
    mutations.append((missing_top_level, "report_schema"))

    phase_extra = passing_smoke_report()
    phase_extra["steady_phases"][0]["extra"] = []
    mutations.append((phase_extra, "steady_phases[0]: record schema"))

    request_extra = passing_smoke_report()
    request_extra["requests"][0]["token"] = "secret"
    mutations.append((request_extra, "requests: record schema"))

    request_missing = passing_smoke_report()
    request_missing["requests"][0].pop("time_s")
    mutations.append((request_missing, "requests: record schema"))

    for report, expected in mutations:
        result = validate_smoke_report(report)
        assert result.ok is False
        assert any(expected in failure for failure in result.failures)


def test_smoke_rejects_nonexact_evidence_value_types():
    mutations = []

    boolean_sequence = passing_smoke_report()
    boolean_sequence["requests"][0]["request_seq"] = False
    mutations.append((boolean_sequence, "requests: record value"))

    float_tick = passing_smoke_report()
    float_tick["requests"][0]["observation_tick"] = 250.0
    mutations.append((float_tick, "requests: record value"))

    integer_request_time = passing_smoke_report()
    integer_request_time["requests"][0]["time_s"] = 3
    mutations.append((integer_request_time, "request_timing"))

    integer_action = passing_smoke_report()
    integer_action["requests"][0]["committed_actions"][0][0] = 0
    mutations.append((integer_action, "committed_actions"))

    integer_phase_time = passing_smoke_report()
    integer_phase_time["steady_phases"][0]["publish_times"][0] = 0
    mutations.append((integer_phase_time, "steady_phases[0]: timestamps"))

    integer_timeline_time = passing_smoke_report()
    integer_timeline_time["published_ticks"][0]["time_s"] = 0
    mutations.append((integer_timeline_time, "published_ticks: record value"))

    integer_executed_action = passing_smoke_report()
    integer_executed_action["executed_actions"][0]["post_slew_action"][0] = 0
    mutations.append((integer_executed_action, "executed_actions: record value"))

    for report, expected in mutations:
        result = validate_smoke_report(report)
        assert result.ok is False
        assert any(expected in failure for failure in result.failures)


def test_smoke_rejects_incomplete_or_incoherent_full_request_evidence():
    mutations = []

    nonfinite_time = passing_smoke_report()
    nonfinite_time["requests"][4]["time_s"] = float("nan")
    mutations.append((nonfinite_time, "request_timing"))

    nonincreasing_time = passing_smoke_report()
    nonincreasing_time["requests"][4]["time_s"] = nonincreasing_time["requests"][3][
        "time_s"
    ]
    mutations.append((nonincreasing_time, "request_timing"))

    sequence_gap = passing_smoke_report()
    sequence_gap["requests"][5]["request_seq"] = 99
    mutations.append((sequence_gap, "request_sequence"))

    tick_spacing = passing_smoke_report()
    tick_spacing["requests"][5]["observation_tick"] += 1
    mutations.append((tick_spacing, "request_spacing"))

    first_request_misaligned = passing_smoke_report()
    first_request_misaligned["requests"][0]["observation_tick"] += 1
    first_request_misaligned["requests"][0]["time_s"] = 2.9
    mutations.append((first_request_misaligned, "first_request_alignment"))

    missing_request = passing_smoke_report()
    missing_request["requests"].pop(8)
    mutations.append((missing_request, "request_coverage"))

    duplicate_final = passing_smoke_report()
    duplicate_final["requests"][16]["request_seq"] = 17
    mutations.append((duplicate_final, "request_sequence"))

    wrong_armed_final = passing_smoke_report()
    wrong_armed_final["armed_request_seq"] = 16
    mutations.append((wrong_armed_final, "armed_request_seq"))

    wrong_final_start = passing_smoke_report()
    wrong_final_start["requests"][-1]["time_s"] = 10.9
    wrong_final_start["delayed_request_started_at"] = 10.9
    mutations.append((wrong_final_start, "delayed_request_started_at"))

    unbacked_successor = passing_smoke_report()
    unbacked_successor["published_ticks"][274 - 100]["source_kind"] = "hold"
    mutations.append((unbacked_successor, "repeated_handoffs"))

    for report, expected in mutations:
        result = validate_smoke_report(report)
        assert result.ok is False
        assert any(expected in failure for failure in result.failures)


def test_smoke_rejects_cumulative_51hz_request_time_drift():
    report = passing_smoke_report()
    tick_period_s = 1.0 / 51.0
    last_tick = 762
    for tick in range(750, last_tick + 1):
        report["published_ticks"].append(
            {
                "tick": tick,
                "time_s": 3.0 + (tick - 250) * tick_period_s,
                "source_kind": "hold",
                "state": "fault",
            }
        )
        report["executed_actions"].append(
            {"tick": tick, "post_slew_action": np.zeros(36, np.float32).tolist()}
        )
    for entry in report["published_ticks"]:
        if entry["tick"] >= 250:
            entry["time_s"] = 3.0 + (entry["tick"] - 250) * tick_period_s
    for request in report["requests"]:
        request["time_s"] = 3.0 + (request["observation_tick"] - 250) * tick_period_s
    fault_at = 3.0 + (664 - 250) * tick_period_s
    report["delayed_request_started_at"] = report["requests"][-1]["time_s"]
    report["fault_at"] = fault_at
    for entry in report["published_ticks"]:
        if entry["time_s"] < fault_at:
            entry["state"] = "paused" if entry["time_s"] < 3.0 else "active"
            entry["source_kind"] = "hold" if entry["tick"] < 256 else "policy"
        else:
            entry["state"] = "fault"
            entry["source_kind"] = "hold"
    report["steady_phases"] = [
        {
            "publish_times": [
                entry["time_s"]
                for entry in report["published_ticks"]
                if entry["time_s"] < fault_at
            ]
        },
        {
            "publish_times": [
                entry["time_s"]
                for entry in report["published_ticks"]
                if entry["time_s"] >= fault_at
            ]
        },
    ]
    assert report["requests"][0]["time_s"] == 3.0
    assert report["requests"][-1]["time_s"] == 11.0
    result = validate_smoke_report(report)
    assert result.ok is False
    assert any("request_absolute_timing" in failure for failure in result.failures)


def test_smoke_rejects_two_late_requests_and_48_sparse_executed_actions():
    report = passing_smoke_report()
    report["requests"] = report["requests"][-2:]
    report["executed_actions"] = report["executed_actions"][-48:]
    result = validate_smoke_report(report)
    assert result.ok is False
    assert any("request_coverage" in failure for failure in result.failures)
    assert any("executed_actions" in failure for failure in result.failures)


def test_collector_preserves_unpublished_raw_tick_and_validator_rejects_it():
    records = _passing_metrics_records()
    target_index = next(
        index
        for index, record in enumerate(records)
        if record.get("event") == "tick"
        and record.get("published") is True
        and record["monotonic_s"] >= 11.30
    )
    published_replacement = dict(records[target_index])
    unpublished = {
        **published_replacement,
        "published": False,
        "state": "active",
        "source_kind": "policy",
    }
    records[target_index : target_index + 1] = [unpublished, published_replacement]
    report = collect_smoke_report(
        records,
        scenario_started_at=0.0,
        certification_boundary=smoke_driver.CertificationBoundary(100, 0.0),
        armed_seq=17,
        delayed_record={"request_seq": 17, "applied_latency_s": 0.30},
        goal_counts_before=[0, 1],
        goal_counts_running=[1, 1],
        goal_counts_after=[0, 1],
        terminal_restored=True,
        ports_rebound=True,
        bridge_exit_code=0,
        live_children_after=[],
        child_exit_codes={"wbc": -2, "camera": 0, "policy": 0, "bridge": 0},
        live_threads_after=[],
    )
    assert report.get("unpublished_ticks") == [
        {
            "tick": unpublished["tick"],
            "time_s": unpublished["monotonic_s"],
            "source_kind": "policy",
            "state": "active",
        }
    ]
    result = validate_smoke_report(report)
    assert result.ok is False
    assert any("unpublished_ticks" in failure for failure in result.failures)


def test_smoke_rejects_nonempty_unpublished_tick_evidence_explicitly():
    report = passing_smoke_report()
    report["unpublished_ticks"] = [
        {
            "tick": 665,
            "time_s": 11.30,
            "source_kind": "policy",
            "state": "active",
        }
    ]
    result = validate_smoke_report(report)
    assert result.ok is False
    assert any("unpublished_ticks" in failure for failure in result.failures)


@pytest.mark.parametrize(
    "delayed_start,fault_at",
    [
        (10.99, 11.10),
        (11.51, 11.60),
        (11.16, 11.15),
        (11.16, 11.301),
    ],
)
def test_smoke_rejects_off_window_or_noncausal_delayed_fault_timing(
    delayed_start, fault_at
):
    report = passing_smoke_report()
    report["requests"][-1]["time_s"] = delayed_start
    report["delayed_request_started_at"] = delayed_start
    report["fault_at"] = fault_at
    result = validate_smoke_report(report)
    assert result.ok is False
    assert any(
        "delayed_request_started_at" in failure or "fault_at" in failure
        for failure in result.failures
    )


def test_smoke_rejects_missing_or_wrong_design_phase_execution():
    mutations = []

    no_paused = passing_smoke_report()
    no_paused["published_ticks"] = [
        entry for entry in no_paused["published_ticks"] if entry["time_s"] >= 3.0
    ]
    mutations.append((no_paused, "paused_phase"))

    paused_policy = passing_smoke_report()
    paused_policy["published_ticks"][50]["source_kind"] = "policy"
    mutations.append((paused_policy, "paused_phase"))

    all_hold = passing_smoke_report()
    for entry in all_hold["published_ticks"]:
        if 3.0 <= entry["time_s"] < 11.0:
            entry["source_kind"] = "hold"
    mutations.append((all_hold, "active_source_sequence"))

    only_six_policy = passing_smoke_report()
    active = [
        entry
        for entry in only_six_policy["published_ticks"]
        if 3.0 <= entry["time_s"] < 11.0
    ]
    for entry in active[12:]:
        entry["source_kind"] = "hold"
    mutations.append((only_six_policy, "active_source_sequence"))

    mid_sequence_hold = passing_smoke_report()
    next(
        entry
        for entry in mid_sequence_hold["published_ticks"]
        if entry["time_s"] == 5.0
    )["source_kind"] = "hold"
    mutations.append((mid_sequence_hold, "active_source_sequence"))

    non_active_state = passing_smoke_report()
    next(
        entry for entry in non_active_state["published_ticks"] if entry["time_s"] == 5.0
    )["state"] = "paused"
    mutations.append((non_active_state, "active_state"))

    sample_thinned = passing_smoke_report()
    sample_thinned["published_ticks"] = sample_thinned["published_ticks"][::2]
    mutations.append((sample_thinned, "phase_coverage"))

    endpoint_truncated = passing_smoke_report()
    endpoint_truncated["published_ticks"] = [
        entry
        for entry in endpoint_truncated["published_ticks"]
        if entry["time_s"] <= 12.5
    ]
    mutations.append((endpoint_truncated, "timeline_coverage"))

    for report, expected in mutations:
        result = validate_smoke_report(report)
        assert result.ok is False
        assert any(expected in failure for failure in result.failures)


def test_smoke_rejects_hold_source_gap_between_second_11_and_r17():
    report = passing_smoke_report()
    for entry in report["published_ticks"]:
        if 11.0 <= entry["time_s"] < 11.16:
            entry["source_kind"] = "hold"
    result = validate_smoke_report(report)
    assert result.ok is False
    assert any("pre_fault_source_sequence" in failure for failure in result.failures)


@pytest.mark.parametrize(
    "state,source_kind,expected",
    [
        ("active", "policy", "post_fault_state"),
        ("active", "hold", "post_fault_state"),
        ("fault", "policy", "post_fault_source"),
    ],
)
def test_smoke_requires_latched_fault_hold_for_every_chronological_post_fault_tick(
    state, source_kind, expected
):
    report = passing_smoke_report()
    for entry in report["published_ticks"]:
        if entry["time_s"] >= 11.30:
            entry["state"] = state
            entry["source_kind"] = source_kind
    result = validate_smoke_report(report)
    assert result.ok is False
    assert any(expected in failure for failure in result.failures)


def test_smoke_cross_checks_first_fault_navigation_against_executed_action():
    report = passing_smoke_report()
    first_fault_tick = next(
        entry["tick"]
        for entry in report["published_ticks"]
        if entry["time_s"] == report["fault_at"]
    )
    executed = next(
        entry
        for entry in report["executed_actions"]
        if entry["tick"] == first_fault_tick
    )
    executed["post_slew_action"][32] = 0.1
    result = validate_smoke_report(report)
    assert result.ok is False
    assert any("first_fault_goal_navigation" in failure for failure in result.failures)


def test_smoke_rejects_publication_gap_during_blocked_http_request():
    report = passing_smoke_report()
    report["published_ticks"] = [
        entry
        for entry in report["published_ticks"]
        if not 11.08 <= entry["time_s"] <= 11.24
    ]
    for phase in report["steady_phases"]:
        phase["publish_times"] = [
            value for value in phase["publish_times"] if not 11.08 <= value <= 11.24
        ]
    result = validate_smoke_report(report)
    assert result.ok is False
    assert any("blocked_http_interval" in failure for failure in result.failures)


@pytest.mark.parametrize(
    "field,value",
    [
        ("delayed_request_started_at", float("nan")),
        ("delayed_request_started_at", float("inf")),
        ("fault_at", float("nan")),
        ("fault_at", float("inf")),
        ("worker_idle_at_s", float("nan")),
        ("worker_idle_at_s", float("inf")),
        ("blocked_main_loop_max_gap_s", float("nan")),
        ("blocked_main_loop_max_gap_s", float("inf")),
    ],
)
def test_smoke_rejects_nonfinite_scalar_timing(field, value):
    report = passing_smoke_report()
    report[field] = value
    result = validate_smoke_report(report)
    assert result.ok is False
    assert any(field in failure for failure in result.failures)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_smoke_rejects_nonfinite_phase_and_timeline_timestamps(value):
    report = passing_smoke_report()
    report["steady_phases"][0]["publish_times"][10] = value
    report["published_ticks"][10]["time_s"] = value
    result = validate_smoke_report(report)
    assert result.ok is False
    assert any("steady_phases" in failure for failure in result.failures)
    assert any("published_ticks" in failure for failure in result.failures)


@pytest.mark.parametrize(
    "field,value",
    [
        ("blocked_main_loop_max_gap_s", 0.061),
        ("delayed_request_record", {"request_seq": 18, "applied_latency_s": 0.30}),
        (
            "delayed_request_record",
            {"request_seq": 17.0, "applied_latency_s": 0.30},
        ),
        ("fault_at", 11.141),
        ("first_fault_goal_navigation", [0.0, 0.0, 0.1, 0.0]),
        ("policy_actions_after_fault", 1),
        ("policy_actions_after_fault", False),
        ("old_generation_results_discarded", 0),
        ("old_generation_results_discarded", True),
        ("late_results_discarded", 1),
        ("late_results_discarded", False),
        ("worker_idle_at_s", 13.01),
        ("goal_counts_before", [1, 1]),
        ("goal_counts_running", [2, 1]),
        ("goal_counts_after", [1, 1]),
        ("live_children_after", [123]),
        ("bridge_exit_code", 1),
        ("bridge_exit_code", False),
        (
            "child_exit_codes",
            {"wbc": -2, "camera": 0, "policy": None, "bridge": 0},
        ),
        ("terminal_restored", False),
        ("ports_rebound", False),
        ("real_interface_connections", 1),
        ("real_interface_connections", False),
        ("extra_goal_publishers", 1),
    ],
)
def test_each_smoke_failure_is_reported(field, value):
    report = passing_smoke_report()
    report[field] = value
    result = validate_smoke_report(report)
    assert result.ok is False
    assert any(field in failure for failure in result.failures)


def test_smoke_rejects_rate_gap_request_spacing_shape_and_tick_discontinuity():
    mutations = []
    low_rate = passing_smoke_report()
    low_rate["steady_phases"][0]["publish_times"] = [
        3.0 + index * 0.021 for index in range(200)
    ]
    mutations.append((low_rate, "steady_phases"))
    gap = passing_smoke_report()
    gap["steady_phases"][0]["publish_times"][50] += 0.061
    mutations.append((gap, "maximum_gap"))
    spacing = passing_smoke_report()
    spacing["requests"][1]["observation_tick"] = 125
    mutations.append((spacing, "request_spacing"))
    shape = passing_smoke_report()
    shape["requests"][1]["committed_actions"] = shape["requests"][1][
        "committed_actions"
    ][:5]
    mutations.append((shape, "committed_actions"))
    skipped = passing_smoke_report()
    skipped["executed_actions"] = [
        entry for entry in skipped["executed_actions"] if entry["tick"] != 630
    ]
    mutations.append((skipped, "executed_actions"))
    mismatch = passing_smoke_report()
    mismatch["requests"][1]["committed_actions"][2][0] = 0.01
    mutations.append((mismatch, "committed_actions"))
    for report, expected in mutations:
        result = validate_smoke_report(report)
        assert result.ok is False
        assert any(expected in failure for failure in result.failures)
