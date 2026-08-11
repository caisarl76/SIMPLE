from types import SimpleNamespace

import pytest

from scripts.psi0_simple_real_bridge import (
    LocalKeyboard,
    PreflightError,
    RuntimeDependencyFactories,
    build_parser,
    build_policy_client,
    build_runtime_adapters,
    load_local_policy_payload,
    validate_domain_selection,
)
from simple.baselines.client import HttpActionClient
from simple.deploy.psi0_simple_bridge import BridgeMode


def test_cli_exposes_only_shadow_and_sim_control():
    parser = build_parser()
    help_text = parser.format_help()
    assert "{shadow,sim-control}" in help_text
    assert "real-control" not in help_text
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "real-control"])


def test_cli_rgb_default_and_no_low_level_interface_options():
    parser = build_parser()
    assert parser.get_default("camera_color_order") == "rgb"
    destinations = {action.dest for action in parser._actions}
    assert not destinations & {
        "robot_interface",
        "network_interface",
        "low_level_topic",
        "real_control",
        "dds_topic",
    }
    assert parser.get_default("policy_contract") is None


def test_only_sim_control_is_locked_to_isolated_domain_42():
    assert validate_domain_selection(BridgeMode.SIM_CONTROL, 42, 42) == (42, 42)
    with pytest.raises(PreflightError, match="isolated domain 42"):
        validate_domain_selection(BridgeMode.SIM_CONTROL, 7, 9)
    assert validate_domain_selection(BridgeMode.SHADOW, 7, 9) == (7, 9)
    with pytest.raises(PreflightError, match=r"\[0,232\]"):
        validate_domain_selection(BridgeMode.SHADOW, -1, 9)


def test_shadow_does_not_require_or_unconditionally_read_local_contract(tmp_path):
    missing = tmp_path / "missing-contract.json"
    assert load_local_policy_payload(None, BridgeMode.SHADOW) is None
    assert load_local_policy_payload(missing, BridgeMode.SHADOW) is None
    with pytest.raises(PreflightError, match="required in sim-control"):
        load_local_policy_payload(None, BridgeMode.SIM_CONTROL)
    with pytest.raises(PreflightError, match="cannot read"):
        load_local_policy_payload(missing, BridgeMode.SIM_CONTROL)


def test_bridge_http_factory_uses_five_seconds_but_generic_default_is_none():
    bridge_client = build_policy_client("127.0.0.1", 22086)
    generic_client = HttpActionClient("127.0.0.1", 22086)
    assert bridge_client.timeout == 5.0
    assert generic_client.timeout is None


class InMemoryAdapter:
    def __init__(self, payload=None):
        self.payload = payload
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class InMemoryWbcConfigClient(InMemoryAdapter):
    def get_config(self, timeout_s=3.0):
        assert timeout_s == 3.0
        return self.payload


class InMemoryStateSource(InMemoryAdapter):
    pass


class InMemoryCameraReader(InMemoryAdapter):
    pass


class FakeGraph:
    def __init__(self, publishers, subscriptions):
        self.publishers = publishers
        self.subscriptions = subscriptions

    def counts(self, topic):
        return self.publishers, self.subscriptions


def valid_adapter_dependencies():
    return RuntimeDependencyFactories(
        state_source=lambda: InMemoryStateSource(),
        camera_reader=lambda: InMemoryCameraReader(),
        graph=lambda: FakeGraph(publishers=0, subscriptions=1),
    )


def test_shadow_factory_never_calls_goal_publisher():
    calls = []
    adapters = build_runtime_adapters(
        mode=BridgeMode.SHADOW,
        preflight_result=SimpleNamespace(
            publisher_required=False, goal_counts_at_preflight=(0, 1)
        ),
        publisher_factory=lambda: calls.append("publisher"),
        test_dependencies=valid_adapter_dependencies(),
    )
    assert adapters.goal_publisher is None
    assert calls == []


def test_shadow_rejects_publisher_count_change_after_preflight():
    calls = []
    dependencies = RuntimeDependencyFactories(
        state_source=lambda: calls.append("state"),
        camera_reader=lambda: calls.append("camera"),
        graph=lambda: FakeGraph(publishers=1, subscriptions=1),
    )
    with pytest.raises(PreflightError, match="changed after preflight"):
        build_runtime_adapters(
            mode=BridgeMode.SHADOW,
            preflight_result=SimpleNamespace(
                publisher_required=False, goal_counts_at_preflight=(0, 1)
            ),
            publisher_factory=lambda: calls.append("publisher"),
            test_dependencies=dependencies,
        )
    assert calls == []


def test_local_keyboard_toggle_set_is_exact():
    assert LocalKeyboard.ACCEPTED_KEYS == (b"p",)
