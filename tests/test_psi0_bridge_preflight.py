import copy

import pytest

from decoupled_wbc.control.main.model_contract import digest_model_contract
from scripts.psi0_simple_real_bridge import (
    BoundedWbcConfigClient,
    GoalOwnershipGuard,
    PreflightError,
    compare_policy_contracts,
    establish_goal_ownership,
    run_preflight,
    validate_connected_wbc,
    validate_then_create_publisher,
)
from simple.deploy.psi0_simple_bridge import BridgeMode, PolicyContract
from tests.psi0_bridge_testkit import (
    ManualClock,
    make_joint_contract,
    policy_payload,
)


POLICY_WIRE_FIELDS = tuple(policy_payload())


def test_matching_test_contract_is_accepted_only_for_sim_wbc():
    local = PolicyContract.from_dict(policy_payload())
    server = PolicyContract.from_dict(policy_payload())
    result = compare_policy_contracts(local, server, BridgeMode.SIM_CONTROL, "sim")
    assert result.policy_certified is True
    assert result.mismatched_fields == ()
    shadow = compare_policy_contracts(local, server, BridgeMode.SHADOW, "sim")
    assert shadow.policy_certified is False
    assert shadow.mismatched_fields == ()
    with pytest.raises(PreflightError, match="test-only.*sim"):
        compare_policy_contracts(local, server, BridgeMode.SIM_CONTROL, "real")


@pytest.mark.parametrize("field", POLICY_WIRE_FIELDS)
def test_every_policy_wire_field_is_compared(field):
    local_payload = policy_payload()
    server_payload = policy_payload()
    value = server_payload[field]
    if type(value) is bool:
        server_payload[field] = not value
    elif type(value) is int:
        server_payload[field] = value + 1
    else:
        server_payload[field] = ("0" * len(value)) if value else "different"
    try:
        server = PolicyContract.from_dict(server_payload)
    except (TypeError, ValueError):
        with pytest.raises((TypeError, ValueError)):
            PolicyContract.from_dict(server_payload)
        return
    local = PolicyContract.from_dict(local_payload)
    with pytest.raises(PreflightError, match=field):
        compare_policy_contracts(local, server, BridgeMode.SIM_CONTROL, "sim")
    shadow = compare_policy_contracts(local, server, BridgeMode.SHADOW, "sim")
    assert shadow.policy_certified is False
    assert shadow.mismatched_fields == (field,)


def test_shadow_contract_failure_never_calls_publisher_factory():
    result = run_preflight(
        mode=BridgeMode.SHADOW,
        local_policy=policy_payload(),
        server_policy={"malformed": True},
        wbc_payload=valid_wbc_payload(),
        graph=FakeGraph(publishers=0, subscriptions=1),
        expected_model_contract=valid_model_contract(),
        expected_gitlink_sha="1" * 40,
    )
    assert result.policy_certified is False
    assert result.publisher_required is False
    assert result.runtime_policy_contract is not None


def test_shadow_missing_local_contract_uses_uncertified_server_contract():
    result = run_preflight(
        mode=BridgeMode.SHADOW,
        local_policy=None,
        server_policy=policy_payload(),
        wbc_payload=valid_wbc_payload(),
        graph=FakeGraph(publishers=0, subscriptions=1),
        expected_model_contract=valid_model_contract(),
        expected_gitlink_sha="1" * 40,
    )
    assert result.policy_certified is False
    assert result.runtime_policy_contract == PolicyContract.from_dict(policy_payload())
    assert result.policy_mismatched_fields == ("local policy contract unavailable",)


def test_shadow_with_no_usable_contract_remains_observation_only():
    result = run_preflight(
        mode=BridgeMode.SHADOW,
        local_policy=None,
        server_policy=None,
        wbc_payload=valid_wbc_payload(),
        graph=FakeGraph(publishers=0, subscriptions=1),
        expected_model_contract=valid_model_contract(),
        expected_gitlink_sha="1" * 40,
    )
    assert result.policy_certified is False
    assert result.runtime_policy_contract is None
    assert result.publisher_required is False


def test_sim_control_rejects_missing_local_contract():
    with pytest.raises(PreflightError, match="local policy contract unavailable"):
        run_preflight(
            mode=BridgeMode.SIM_CONTROL,
            local_policy=None,
            server_policy=policy_payload(),
            wbc_payload=valid_wbc_payload(),
            graph=FakeGraph(publishers=0, subscriptions=1),
            expected_model_contract=valid_model_contract(),
            expected_gitlink_sha="1" * 40,
        )


def test_shadow_reports_wbc_differences_but_keeps_structural_joint_contract():
    payload = copy.deepcopy(valid_wbc_payload())
    payload["env_type"] = "real"
    payload["interface"] = "robot0"
    payload["model_contract"]["git"]["commit"] = "9" * 40
    payload["model_contract_sha256"] = digest_model_contract(payload["model_contract"])
    graph = FakeGraph(publishers=2, subscriptions=1)
    result = run_preflight(
        mode=BridgeMode.SHADOW,
        local_policy=policy_payload(),
        server_policy=policy_payload(),
        wbc_payload=payload,
        graph=graph,
        expected_model_contract=valid_model_contract(),
        expected_gitlink_sha="1" * 40,
    )
    assert result.policy_certified is False
    assert result.publisher_required is False
    assert result.goal_counts_at_preflight == (2, 1)
    assert result.joint_contract.joint_names == make_joint_contract().joint_names
    assert result.wbc_mismatched_fields == (
        "config.env_type",
        "config.interface",
        "model_contract.git.commit",
    )


WBC_REQUIRED = {
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
    "domain_id": 42,
}


def valid_model_contract():
    joints = list(make_joint_contract().joint_names)
    return {
        "schema": "decoupled_wbc.g1-model-contract.v1",
        "git": {"commit": "1" * 40, "working_tree_clean": True},
        "robot_model": {
            "name": "g1_29dof_with_hand",
            "joint_names": joints,
            "lower_position_limits": [-2.0] * 43,
            "upper_position_limits": [2.0] * 43,
            "upper_body_joint_names": joints[12:],
        },
        "urdf": {
            "relative_path": (
                "control/robot_model/model_data/g1/g1_29dof_with_hand.urdf"
            ),
            "sha256": "2" * 64,
        },
        "onnx_models": [
            {
                "role": role,
                "relative_path": (f"sim2mujoco/resources/robots/g1/policy/{filename}"),
                "sha256": digest,
                "input": {
                    "name": "observations",
                    "shape": ["dynamic", 516],
                    "feature_size": 516,
                },
                "output": {
                    "name": "actions",
                    "shape": ["dynamic", 15],
                    "feature_size": 15,
                },
            }
            for role, filename, digest in (
                ("balance", "GR00T-WholeBodyControl-Balance.onnx", "3" * 64),
                ("walk", "GR00T-WholeBodyControl-Walk.onnx", "4" * 64),
            )
        ],
    }


def valid_wbc_payload():
    contract = valid_model_contract()
    return {
        **WBC_REQUIRED,
        "model_contract": contract,
        "model_contract_sha256": digest_model_contract(contract),
    }


def test_connected_wbc_payload_matches_every_top_level_and_model_identity():
    actual = valid_wbc_payload()
    expected = valid_model_contract()
    validated = validate_connected_wbc(
        actual=actual,
        expected_contract=expected,
        expected_gitlink_sha="1" * 40,
        required_domain_id=42,
    )
    assert validated.joint_contract.joint_names == tuple(
        expected["robot_model"]["joint_names"]
    )
    assert validated.joint_contract.upper_body_joint_names == tuple(
        expected["robot_model"]["upper_body_joint_names"]
    )
    assert actual["model_contract_sha256"] == digest_model_contract(expected)


def set_path(payload, path, value):
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.mark.parametrize(
    "path,value,error",
    [
        (("env_type",), "real", "env_type"),
        (("domain_id",), 0, "domain_id"),
    ],
)
def test_top_level_wbc_mutation_fails_before_publisher(path, value, error):
    payload = copy.deepcopy(valid_wbc_payload())
    set_path(payload, path, value)
    calls = []
    with pytest.raises(PreflightError, match=error):
        validate_then_create_publisher(
            payload,
            valid_model_contract(),
            "1" * 40,
            42,
            publisher_factory=lambda: calls.append("publisher"),
        )
    assert calls == []


def test_stale_transmitted_model_digest_fails_before_field_comparison():
    payload = copy.deepcopy(valid_wbc_payload())
    set_path(payload, ("model_contract", "robot_model", "name"), "wrong")
    calls = []
    with pytest.raises(PreflightError, match="digest"):
        validate_then_create_publisher(
            payload,
            valid_model_contract(),
            "1" * 40,
            42,
            publisher_factory=lambda: calls.append("publisher"),
        )
    assert calls == []


@pytest.mark.parametrize(
    "path,value,error",
    [
        (("git", "commit"), "0" * 40, "git"),
        (("git", "working_tree_clean"), False, "clean"),
        (("robot_model", "name"), "wrong", "robot_model"),
        (("robot_model", "joint_names", 0), "wrong", "joint_names"),
        (("robot_model", "upper_body_joint_names", 0), "wrong", "upper_body"),
        (("robot_model", "lower_position_limits", 12), -9.0, "limits"),
        (("urdf", "sha256"), "0" * 64, "urdf"),
        (("onnx_models", 0, "sha256"), "0" * 64, "onnx"),
        (("onnx_models", 1, "input", "feature_size"), 515, "onnx"),
    ],
)
def test_recomputed_digest_still_reports_model_field_mismatch(path, value, error):
    payload = copy.deepcopy(valid_wbc_payload())
    set_path(payload["model_contract"], path, value)
    payload["model_contract_sha256"] = digest_model_contract(payload["model_contract"])
    calls = []
    with pytest.raises(PreflightError, match=error):
        validate_then_create_publisher(
            payload,
            valid_model_contract(),
            "1" * 40,
            42,
            publisher_factory=lambda: calls.append("publisher"),
        )
    assert calls == []


def test_transmitted_digest_value_mutation_reports_digest():
    payload = copy.deepcopy(valid_wbc_payload())
    payload["model_contract_sha256"] = "0" * 64
    with pytest.raises(PreflightError, match="digest"):
        validate_then_create_publisher(
            payload,
            valid_model_contract(),
            "1" * 40,
            42,
            publisher_factory=lambda: None,
        )


def test_valid_wbc_validation_then_factory_calls_publisher_once():
    calls = []
    validated, publisher = validate_then_create_publisher(
        valid_wbc_payload(),
        valid_model_contract(),
        "1" * 40,
        42,
        publisher_factory=lambda: calls.append("publisher") or object(),
    )
    assert validated.joint_contract.joint_names == make_joint_contract().joint_names
    assert publisher is not None
    assert calls == ["publisher"]


def test_wbc_service_timeout_is_three_seconds_and_destroys_resources():
    clock = ManualClock()
    events = []
    adapter = BoundedWbcConfigClient(
        clock=clock,
        wait_once=lambda timeout: (
            clock.set_tick(clock.tick + int(timeout * 50)) or False
        ),
        destroy=lambda: events.append("destroy"),
    )
    with pytest.raises(TimeoutError, match="3.0"):
        adapter.get_config(timeout_s=3.0)
    assert clock() == pytest.approx(3.0)
    assert events == ["destroy"]


class FakeGraph:
    def __init__(self, publishers, subscriptions):
        self.publishers = publishers
        self.subscriptions = subscriptions

    def counts(self, topic):
        assert topic == "ControlPolicy/upper_body_pose"
        return self.publishers, self.subscriptions


def test_sim_control_owns_the_only_goal_publisher():
    graph = FakeGraph(0, 1)
    events = []

    def create():
        events.append("create")
        graph.publishers += 1
        return object()

    publisher = establish_goal_ownership(BridgeMode.SIM_CONTROL, graph, create)
    assert publisher is not None
    assert events == ["create"]
    assert graph.counts("ControlPolicy/upper_body_pose") == (1, 1)


@pytest.mark.parametrize("counts", [(1, 1), (0, 0), (2, 1), (0, 2)])
def test_sim_control_rejects_wrong_preflight_counts_without_publish(counts):
    calls = []
    with pytest.raises(PreflightError, match="0 publishers/1 subscription"):
        establish_goal_ownership(
            BridgeMode.SIM_CONTROL,
            FakeGraph(*counts),
            lambda: calls.append("publisher"),
        )
    assert calls == []


def test_shadow_never_constructs_a_publisher_or_changes_counts():
    graph = FakeGraph(2, 1)
    calls = []
    assert (
        establish_goal_ownership(
            BridgeMode.SHADOW, graph, lambda: calls.append("publisher")
        )
        is None
    )
    assert graph.counts("ControlPolicy/upper_body_pose") == (2, 1)
    assert calls == []


def test_shadow_goal_counts_are_checked_during_and_at_end_of_lifetime():
    graph = FakeGraph(2, 1)
    guard = GoalOwnershipGuard(BridgeMode.SHADOW, graph, (2, 1))
    guard.check()
    graph.publishers = 3
    with pytest.raises(PreflightError, match="counts changed"):
        guard.check()
    with pytest.raises(PreflightError, match="counts changed"):
        guard.close()
