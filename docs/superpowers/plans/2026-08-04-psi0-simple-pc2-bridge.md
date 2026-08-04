# PSI0 SIMPLE PC2 Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the approved fail-closed PSI0-to-decoupled-WBC bridge for `shadow` and isolated MuJoCo `sim-control` modes, without adding a real-robot control path.

**Architecture:** A transport-independent core owns named-joint mapping, input validation, bounded holds, RTC scheduling, action validation, slew limiting, and the PAUSED/ACTIVE/FAULT/STOPPED state machine. A thin PC2 runtime owns ROS2/ZMQ/HTTP/keyboard resources and injects them into the core. The decoupled-WBC submodule exposes a canonical model contract only after its model and policy load, and the root bridge refuses to create a publisher until WBC, policy, graph-ownership, camera, and state contracts pass.

**Tech Stack:** Python 3.10, NumPy, requests, pytest, ROS2/rclpy, ZeroMQ/msgpack, OpenCV JPEG codec, ONNX Runtime, MuJoCo, uv, Git submodules.

---

## Non-negotiable safety boundaries

- This plan implements only `shadow` and `sim-control`. Do not add a `real-control` enum value, CLI option, network-interface fallback, or Unitree low-level publisher.
- Do not start a real WBC, real DDS loop, or robot process while implementing or running unit tests.
- Run the simulation smoke test only with ROS domain 42, Unitree domain 42, Linux loopback, and zero pre-existing goal publishers.
- The current live PSI0 checkpoint and `/act` endpoint remain ineligible for `sim-control`. Root-repository tests use the explicitly test-only `/act-rtc-v1` fake.
- The companion live PSI0 server change is out of this repository. Record its required API in the README, but do not claim live-policy readiness without a matching server commit and certified policy contract.
- `third_party/decoupled_wbc` is a submodule. Do not update the root gitlink until the nested commit is available from a remote. A push is an external write and requires repository authority.
- Preserve the user's untracked `.codegraph/` directory throughout all tasks.

## File map

### Root SIMPLE repository

- Create `src/simple/deploy/__init__.py`: export the bridge's public transport-independent types.
- Create `src/simple/deploy/psi0_simple_bridge.py`: contracts, named mappings, validation, bounded holds, slew limiter, RTC scheduler, state machine, and worker result lifecycle.
- Create `scripts/psi0_simple_real_bridge.py`: CLI configuration, bounded WBC preflight, ROS graph ownership, ROS state/goal adapters, composed-camera reader, local keyboard, inference worker, metrics, and shutdown.
- Create `scripts/certify_psi0_policy_contract.py`: deterministic converter-layout certification and v2 policy-contract writer.
- Modify `src/simple/baselines/client.py`: optional request timeout, `/contract` fetch, and `/act-rtc-v1` request/metadata response while retaining `/act` compatibility.
- Modify `scripts/postprocess_psi0.py`: chronological RPY conversion and `0.74` initial height.
- Create root tests under `tests/` split by converter, HTTP transport, mapping, safety/holds, scheduler, preflight/camera, runtime shutdown, and certification.
- Create `scripts/tests/fake_psi0_rtc_server.py`: fake v2 contract and deterministic RTC HTTP server.
- Create `scripts/tests/fake_composed_camera_server.py`: real `ImageMessageSchema` JPEG publisher with red/blue sentinels.
- Create `scripts/tests/bridge_subprocess_fixture.py`: injectable fake-runtime subprocess used for signal and shutdown tests.
- Create `scripts/tests/smoke_psi0_simple_bridge.py`: isolated 15-second MuJoCo orchestration and metrics assertions.
- Create `scripts/tests/fixtures/psi0_policy_contract_test_v2.json`: explicitly test-only P=30/s=24/d=6 policy contract.
- Modify `README.md`: document shadow/simulation launch order, controls, shutdown ownership, and live-policy blockers.

### `third_party/decoupled_wbc` nested repository

- Modify `control/main/teleop/configs/configs.py`: serialize `env_type`, add `domain_id`, and propagate it to `DOMAIN_ID`.
- Modify `control/robot_model/supplemental_info/g1/g1_supplemental_info.py`: correct only the seven reviewed right-hand limits.
- Create `control/main/model_contract.py`: canonical JSON, Git/URDF/ONNX hashes, ordered names/effective limits, and contract digest.
- Modify `control/main/teleop/run_g1_control_loop.py`: construct model and policy before publishing the attested config service.
- Create `tests/control/main/test_model_contract.py`: digest, schema, order, hashes, and mutation tests.
- Create `tests/control/main/teleop/test_g1_control_loop_contract.py`: config serialization and service-construction order tests.
- Create `tests/control/robot_model/test_g1_effective_limits.py`: URDF/effective-limit audit with the two shoulder allowlist entries.

## Task 0: Create the isolated implementation worktree

**Files:**
- No file changes.

- [ ] **Step 1: Confirm the approved base and clean tracked state**

Run:

```bash
git rev-parse --short HEAD
git status --short
```

Expected: HEAD contains this plan commit; the only status entry is `?? .codegraph/`.

- [ ] **Step 2: Create the feature worktree**

Run:

```bash
git worktree add /tmp/simple-psi0-bridge -b feat/psi0-simple-bridge main
cd /tmp/simple-psi0-bridge
git submodule update --init --recursive third_party/decoupled_wbc
```

Expected: the worktree is on `feat/psi0-simple-bridge`, and the submodule resolves to the root gitlink.

- [ ] **Step 3: Create the nested delivery branch before editing the detached submodule**

Run:

```bash
git -C third_party/decoupled_wbc switch -c psi0-simple-bridge-contract
git -C third_party/decoupled_wbc status --short --branch
```

Expected: `## psi0-simple-bridge-contract` and no nested file changes.

- [ ] **Step 4: Record the safety baseline**

Run:

```bash
pgrep -af 'run_g1_control_loop|psi0_simple_real_bridge|fake_psi0_rtc_server|mujoco'
```

Expected: no process owned by this implementation worktree. If a matching process belongs to the user, do not stop it; record it and choose isolated ports/domains.

## Task 1: Correct and lock the 32-D converter contract

**Files:**
- Modify: `scripts/postprocess_psi0.py:111-125,166-190`
- Create: `tests/test_postprocess_psi0.py`

- [ ] **Step 1: Write the failing chronological-layout test**

Create `tests/test_postprocess_psi0.py` with:

```python
import numpy as np

from scripts.postprocess_psi0 import (
    build_proprio_obs,
    build_vectors,
    initial_command,
)


def test_build_vectors_uses_chronological_roll_pitch_yaw_history():
    proprio = np.arange(3 * 43, dtype=np.float32).reshape(3, 43)
    history = np.zeros((3, 9), dtype=np.float32)
    history[:, 3:6] = np.array(
        [[10, 20, 30], [11, 21, 31], [12, 22, 32]], dtype=np.float32
    )  # source columns: yaw, pitch, roll
    history[:, 6] = [0.70, 0.71, 0.72]
    cmd = np.zeros((3, 9), dtype=np.float32)
    action = np.zeros((3, 43), dtype=np.float32)

    states, _ = build_vectors(
        proprio, cmd, history, action, np.zeros(3), np.zeros(3)
    )
    *_, torso_rpy, height = build_proprio_obs(proprio, history)

    expected_rpy = np.array(
        [[30, 20, 10], [31, 21, 11], [32, 22, 12]], dtype=np.float32
    )
    np.testing.assert_array_equal(states[:, 28:31], expected_rpy)
    np.testing.assert_array_equal(torso_rpy, expected_rpy)
    np.testing.assert_array_equal(states[:, 31:32], height)


def test_initial_history_height_is_point_74():
    assert initial_command.dtype == np.float32
    assert initial_command[6] == np.float32(0.74)
```

- [ ] **Step 2: Run the test and verify the legacy converter fails**

Run:

```bash
uv run --group dev pytest -q tests/test_postprocess_psi0.py
```

Expected: both assertions exposing row reversal and `0.75` fail.

- [ ] **Step 3: Make the minimal converter correction**

In `scripts/postprocess_psi0.py`, replace the two legacy expressions with:

```python
history_cmd[:to, 3:6][:, ::-1],  # source yaw,pitch,roll -> roll,pitch,yaw
```

and:

```python
initial_command = np.array(
    [0, 0, 0, 0, 0, 0, 0.74, 0.74, 0.74], dtype=np.float32
)
```

- [ ] **Step 4: Run focused tests and formatting checks**

Run:

```bash
uv run --group dev pytest -q tests/test_postprocess_psi0.py
uv run --group dev ruff check tests/test_postprocess_psi0.py
uv run --group dev ruff check --select E9,F63,F7,F82 scripts/postprocess_psi0.py
uv run --group dev ruff format --check tests/test_postprocess_psi0.py
```

Expected: 2 tests pass and both Ruff commands exit zero.

- [ ] **Step 5: Commit the converter correction**

```bash
git add scripts/postprocess_psi0.py tests/test_postprocess_psi0.py
git commit -m "fix: correct PSI0 RPY history conversion"
```

## Task 2: Add deterministic policy-contract certification

**Files:**
- Create: `scripts/certify_psi0_policy_contract.py`
- Create: `tests/test_certify_psi0_policy_contract.py`

- [ ] **Step 1: Write failing certification tests**

The test must exercise corrected, row-reversed, and off-by-one candidates independently:

```python
import numpy as np
import pytest

from scripts.certify_psi0_policy_contract import certify_layout


def history_fixture():
    history = np.zeros((5, 9), dtype=np.float32)
    history[:, 3:6] = np.arange(15, dtype=np.float32).reshape(5, 3)
    history[:, 6] = 0.74
    return history


def test_certify_layout_accepts_only_corrected_causal_candidate():
    history = history_fixture()
    stored = np.zeros((5, 32), dtype=np.float32)
    stored[:, 28:31] = history[:, 3:6][:, ::-1]
    stored[:, 31] = 0.74
    result = certify_layout(stored, history)
    assert result == {
        "layout": "g1_simple_32_rpyh_v2",
        "corrected_match": True,
        "legacy_row_reversed_match": False,
        "off_by_one_match": False,
    }


@pytest.mark.parametrize("candidate", ["legacy", "off_by_one"])
def test_certify_layout_rejects_noncausal_or_shifted_data(candidate):
    history = history_fixture()
    stored = np.zeros((5, 32), dtype=np.float32)
    source = history[:, 3:6][::-1] if candidate == "legacy" else history[[0, 0, 1, 2, 3], 3:6][:, ::-1]
    stored[:, 28:31] = source
    stored[:, 31] = 0.74
    with pytest.raises(ValueError, match="not uniquely certified"):
        certify_layout(stored, history)
```

- [ ] **Step 2: Verify the new module is absent**

Run:

```bash
uv run --group dev pytest -q tests/test_certify_psi0_policy_contract.py
```

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the classifier and contract writer**

Create a CLI that accepts `--stored-state-npy`, `--history-command-npy`, `--checkpoint`, `--dataset-manifest`, `--server-commit`, `--prediction-horizon`, `--execution-horizon`, `--rtc-delay-steps`, `--rtc-training-max-delay`, and `--output`. Its core comparison is:

```python
def certify_layout(stored_state: np.ndarray, history_cmd: np.ndarray) -> dict[str, object]:
    stored = np.asarray(stored_state, dtype=np.float32)
    history = np.asarray(history_cmd, dtype=np.float32)
    if stored.ndim != 2 or stored.shape[1] < 32:
        raise ValueError(f"expected stored state shape (T,>=32), got {stored.shape}")
    if history.shape != (stored.shape[0], 9):
        raise ValueError(f"expected history shape {(stored.shape[0], 9)}, got {history.shape}")
    corrected = history[:, 3:6][:, ::-1]
    legacy = history[:, 3:6][::-1]
    previous = corrected[np.maximum(np.arange(len(corrected)) - 1, 0)]
    corrected_match = bool(np.array_equal(stored[:, 28:31], corrected))
    legacy_match = bool(np.array_equal(stored[:, 28:31], legacy))
    off_by_one_match = bool(np.array_equal(stored[:, 28:31], previous))
    height_match = bool(np.array_equal(stored[:, 31], np.full(len(stored), 0.74, np.float32)))
    if not height_match or not corrected_match or legacy_match or off_by_one_match:
        raise ValueError("processed episode is not uniquely certified for g1_simple_32_rpyh_v2")
    return {
        "layout": "g1_simple_32_rpyh_v2",
        "corrected_match": True,
        "legacy_row_reversed_match": False,
        "off_by_one_match": False,
    }
```

Write JSON atomically with schema `simple.psi0.policy-contract.v2`, SHA-256 hashes of the checkpoint and dataset manifest, the provided server commit, dimensions `32/36`, `50 Hz`, `/act-rtc-v1`, RGB key/order, committed-prefix/executable-suffix semantics, RTC fields, and the certification result. Reject a non-40-character hexadecimal server commit and any RTC tuple that violates the approved inequalities.

- [ ] **Step 4: Run certification tests and CLI help**

Run:

```bash
uv run --group dev pytest -q tests/test_certify_psi0_policy_contract.py
uv run python scripts/certify_psi0_policy_contract.py --help
uv run --group dev ruff check scripts/certify_psi0_policy_contract.py tests/test_certify_psi0_policy_contract.py
```

Expected: 3 tests pass, help lists every required argument, and Ruff exits zero.

- [ ] **Step 5: Commit the certification tool**

```bash
git add scripts/certify_psi0_policy_contract.py tests/test_certify_psi0_policy_contract.py
git commit -m "feat: certify PSI0 policy contracts"
```

## Task 3: Extend the HTTP client without changing legacy defaults

**Files:**
- Modify: `src/simple/baselines/client.py:65-165`
- Create: `tests/test_http_action_client.py`

- [ ] **Step 1: Write failing timeout and RTC metadata tests**

Use a recording fake session and assert these public contracts:

```python
def test_legacy_query_keeps_unbounded_default(recording_session):
    client = HttpActionClient("policy", 22085, session=recording_session)
    client.query_action({}, "test", {}, {})
    assert recording_session.calls[0]["url"].endswith("/act")
    assert recording_session.calls[0]["timeout"] is None


def test_rtc_query_uses_versioned_path_and_returns_metadata(recording_session):
    client = HttpActionClient("policy", 22085, timeout=5.0, session=recording_session)
    response = client.query_rtc_action(
        {"rgb_head_stereo_left": np.zeros((4, 4, 3), np.uint8)},
        "test",
        {"states": np.zeros((1, 32), np.float32)},
        {},
        history={
            "session_id": "s",
            "request_seq": 2,
            "observation_tick": 100,
            "rtc_delay_steps": 3,
            "committed_actions": np.zeros((3, 36), np.float32),
        },
    )
    assert recording_session.calls[0]["url"].endswith("/act-rtc-v1")
    assert recording_session.calls[0]["timeout"] == 5.0
    assert response.metadata["first_action_tick"] == 103
```

Also test `get_contract(timeout=2.0)`, connection/read timeout propagation, non-200 responses, malformed JSON, and that RTC metadata is not silently synthesized by the client.

- [ ] **Step 2: Run the tests and verify constructor/signature failures**

Run:

```bash
uv run --group dev pytest -q tests/test_http_action_client.py
```

Expected: failures show missing `session`, `timeout`, `get_contract`, and `query_rtc_action` support.

- [ ] **Step 3: Implement the compatible transport surface**

Add:

```python
@dataclass(frozen=True)
class RtcActionResponse:
    action: np.ndarray
    metadata: dict[str, Any]
    err: float = 0.0


class HttpActionClient:
    def __init__(
        self,
        server_ip: str,
        server_port: int,
        timeout: float | None = None,
        session: requests.Session | None = None,
    ):
        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout
        self.session = session or requests.Session()

    def get_contract(self, timeout: float = 2.0) -> dict[str, Any]:
        response = self.session.get(
            f"http://{self.server_ip}:{self.server_port}/contract", timeout=timeout
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise RuntimeError("policy contract response must be a JSON object")
        return result
```

Refactor the existing request serialization into one private `_post(path, request)` method that always passes `timeout=self.timeout`. Keep `query_action()` on `/act` with the same return tuple. Implement `query_rtc_action()` on `/act-rtc-v1`; require JSON keys `action` and `metadata`, deserialize NumPy payloads, and return `RtcActionResponse` without adding missing metadata.

- [ ] **Step 4: Run the HTTP and existing baseline import tests**

Run:

```bash
uv run --group dev pytest -q tests/test_http_action_client.py
uv run python -c 'from simple.baselines.client import HttpActionClient; assert HttpActionClient("x", 1).timeout is None'
uv run --group dev ruff check tests/test_http_action_client.py
uv run --group dev ruff check --select E9,F63,F7,F82 src/simple/baselines/client.py
```

Expected: all HTTP tests pass, legacy default is `None`, and Ruff exits zero.

- [ ] **Step 5: Commit the client extension**

```bash
git add src/simple/baselines/client.py tests/test_http_action_client.py
git commit -m "feat: add versioned PSI0 RTC client"
```

## Task 4: Repair and test the nested WBC configuration and effective limits

**Files:**
- Modify: `third_party/decoupled_wbc/control/main/teleop/configs/configs.py:1-68,71-200`
- Modify: `third_party/decoupled_wbc/control/robot_model/supplemental_info/g1/g1_supplemental_info.py:148-154`
- Create: `third_party/decoupled_wbc/tests/control/main/teleop/test_g1_control_loop_contract.py`
- Create: `third_party/decoupled_wbc/tests/control/robot_model/test_g1_effective_limits.py`

- [ ] **Step 1: Write failing config serialization tests**

Test the actual dataclass and actual YAML override:

```python
def test_sim_config_serializes_resolved_environment_and_domain():
    config = ControlLoopConfig(interface="sim", domain_id=42)
    payload = config.to_dict()
    assert payload["env_type"] == "sim"
    assert payload["interface"] == "lo"
    assert payload["domain_id"] == 42
    assert config.load_wbc_yaml()["DOMAIN_ID"] == 42
```

- [ ] **Step 2: Write the failing named limit audit**

Instantiate `lower_and_upper_body`, parse `g1_29dof_with_hand.urdf`, and compare all 31 `upper_body` joints. Require exact URDF equality within `1e-7` except:

```python
SHOULDER_ALLOWLIST = {
    "left_shoulder_roll_joint": (0.19, 2.2515),
    "right_shoulder_roll_joint": (-2.2515, -0.19),
}
```

For every effective joint, accept midpoint/endpoints and reject values `1e-4` outside. Assert the seven right-hand ranges exactly match the approved design table.

- [ ] **Step 3: Verify both nested tests fail**

Run:

```bash
uv run --group sonic pytest -q \
  third_party/decoupled_wbc/tests/control/main/teleop/test_g1_control_loop_contract.py \
  third_party/decoupled_wbc/tests/control/robot_model/test_g1_effective_limits.py
```

Expected: `env_type`/`domain_id` assertions and the seven right-hand assertions fail.

- [ ] **Step 4: Add serialized fields and domain propagation**

Import `field` and declare inside `BaseConfig`:

```python
env_type: Literal["sim", "real"] = field(init=False)
domain_id: int = 0
```

Add `"DOMAIN_ID": config.domain_id` to `override_wbc_config()` and retain the existing `resolve_interface()` assignment.

- [ ] **Step 5: Correct only the approved right-hand values**

Set:

```python
"right_hand_thumb_0_joint": [-1.04719755, 1.04719755],
"right_hand_thumb_1_joint": [-1.04719755, 0.72431163],
"right_hand_thumb_2_joint": [-1.74532925, 0],
"right_hand_index_0_joint": [0, 1.57079632],
"right_hand_index_1_joint": [0, 1.74532925],
"right_hand_middle_0_joint": [0, 1.57079632],
"right_hand_middle_1_joint": [0, 1.74532925],
```

Do not change either shoulder-roll entry.

- [ ] **Step 6: Run nested focused tests and lint**

Run:

```bash
uv run --group sonic pytest -q \
  third_party/decoupled_wbc/tests/control/main/teleop/test_g1_control_loop_contract.py \
  third_party/decoupled_wbc/tests/control/robot_model/test_g1_effective_limits.py
uv run --group dev ruff check \
  third_party/decoupled_wbc/tests/control/main/teleop/test_g1_control_loop_contract.py \
  third_party/decoupled_wbc/tests/control/robot_model/test_g1_effective_limits.py
uv run --group dev ruff check --select E9,F63,F7,F82 \
  third_party/decoupled_wbc/control/main/teleop/configs/configs.py \
  third_party/decoupled_wbc/control/robot_model/supplemental_info/g1/g1_supplemental_info.py
```

Expected: focused tests and Ruff pass.

- [ ] **Step 7: Keep the nested changes uncommitted until Task 5 includes the attestation helper**

Run:

```bash
git -C third_party/decoupled_wbc diff --check
git -C third_party/decoupled_wbc status --short
```

Expected: only the two nested source files and two new nested tests are listed.

## Task 5: Build and serve the connected WBC model attestation

**Files:**
- Create: `third_party/decoupled_wbc/control/main/model_contract.py`
- Modify: `third_party/decoupled_wbc/control/main/teleop/run_g1_control_loop.py:38-97`
- Create: `third_party/decoupled_wbc/tests/control/main/test_model_contract.py`
- Modify: `third_party/decoupled_wbc/tests/control/main/teleop/test_g1_control_loop_contract.py`

- [ ] **Step 1: Write failing canonicalization and mutation tests**

The contract test must instantiate the real `lower_and_upper_body` model and assert:

```python
def test_model_contract_is_canonical_and_complete():
    model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
    contract = build_model_contract(
        robot_model=model,
        config=ControlLoopConfig(interface="sim", enable_waist=True, domain_id=42),
        repository_root=Path(decoupled_wbc.__file__).resolve().parent,
        git_identity=GitIdentity(commit="a" * 40, working_tree_clean=True),
    )
    assert contract["schema"] == "decoupled_wbc.g1-model-contract.v1"
    assert contract["working_tree_clean"] is True
    assert len(contract["joint_names"]) == 43
    assert len(contract["lower_position_limits"]) == 43
    assert len(contract["upper_position_limits"]) == 43
    assert len(contract["upper_body_joint_names"]) == 31
    assert [entry["input_size"] for entry in contract["onnx_models"]] == [516, 516]
    assert [entry["output_size"] for entry in contract["onnx_models"]] == [15, 15]
    assert digest_model_contract(contract) == digest_model_contract(dict(contract))
```

Build one valid fixture contract with dependency injection for Git and ONNX inspection, then mutate each of: right-hand limit, ordered joint name, URDF hash, ONNX hash, nested SHA, and clean-tree flag. Assert each mutation changes the digest and fails `validate_expected_model_contract()`.

- [ ] **Step 2: Write the failing service-construction order test**

Monkeypatch `instantiate_g1_robot_model`, `G1Env`, `get_wbc_policy`, `build_model_contract_payload`, and `create_service_server` to append event names. Use a loop manager whose `ok()` immediately returns `False`. Assert:

```python
assert events.index("robot_model") < events.index("wbc_policy")
assert events.index("wbc_policy") < events.index("model_contract")
assert events.index("model_contract") < events.index("service_server")
```

Also assert the service payload contains both `model_contract` and `model_contract_sha256`.

- [ ] **Step 3: Run the nested tests and verify missing helper/order failures**

Run:

```bash
uv run --group sonic pytest -q \
  third_party/decoupled_wbc/tests/control/main/test_model_contract.py \
  third_party/decoupled_wbc/tests/control/main/teleop/test_g1_control_loop_contract.py
```

Expected: collection fails for the absent helper, and after the helper test imports, the old service order assertion fails.

- [ ] **Step 4: Implement canonical contract primitives**

Create `control/main/model_contract.py` with these exact public functions:

```python
MODEL_CONTRACT_SCHEMA = "decoupled_wbc.g1-model-contract.v1"


@dataclass(frozen=True)
class GitIdentity:
    commit: str
    working_tree_clean: bool


def canonical_json(value: dict[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest_model_contract(contract: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json(contract)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
```

`build_model_contract(robot_model, config, repository_root)` must:

1. call `inspect_git_identity(repository_root)` when no test identity is injected; it runs `git -C repository_root rev-parse HEAD`, requires a 40-character SHA, runs `git -C repository_root status --porcelain --untracked-files=all`, and requires empty output;
2. permit tests to inject immutable `GitIdentity(commit, working_tree_clean)` so TDD can run while the not-yet-committed nested files are intentionally dirty;
3. obtain all 43 names from `robot_model.joint_names` and use `robot_model.dof_index(name)` to read effective lower/upper values in that same order;
4. obtain the 31 upper names by filtering the ordered 43-name list through `get_joint_group_indices("upper_body")`;
5. hash `g1_29dof_with_hand.urdf`;
6. resolve both configured ONNX paths under `sim2mujoco/resources/robots/g1`, hash them, and inspect their first input/output shapes with CPU ONNX Runtime;
7. emit JSON-native strings, lists, integers, floats, and booleans only.

Add `validate_expected_model_contract(actual, expected)` that compares canonical JSON and raises `ValueError("connected WBC model contract mismatch")` on any difference. Add `build_model_contract_payload(config, robot_model)` that returns:

```python
repository_root = Path(__file__).resolve().parents[2]
contract = build_model_contract(robot_model, config, repository_root)
payload = config.to_dict()
payload["model_contract"] = contract
payload["model_contract_sha256"] = digest_model_contract(contract)
return payload
```

- [ ] **Step 5: Move service publication after successful model and policy construction**

In `run_g1_control_loop.main()` preserve manager creation, then perform this order:

```python
wbc_config = config.load_wbc_yaml()
waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
robot_model = instantiate_g1_robot_model(
    waist_location=waist_location, high_elbow_pose=config.high_elbow_pose
)
env = G1Env(
    env_name=config.env_name,
    robot_model=robot_model,
    config=wbc_config,
    wbc_version=config.wbc_version,
    messaging_backend=backend,
)
if env.sim and not config.sim_sync_mode:
    env.start_simulator()
wbc_policy = get_wbc_policy("g1", robot_model, wbc_config, config.upper_body_joint_speed)
service_payload = build_model_contract_payload(config, robot_model)
robot_config_server = create_service_server(backend, ROBOT_CONFIG_TOPIC, service_payload)
```

Keep `robot_config_server` alive for the function lifetime and close it in `finally` when the backend exposes `close()`.

- [ ] **Step 6: Run all nested contract/limit tests**

Run:

```bash
uv run --group sonic pytest -q \
  third_party/decoupled_wbc/tests/control/main/test_model_contract.py \
  third_party/decoupled_wbc/tests/control/main/teleop/test_g1_control_loop_contract.py \
  third_party/decoupled_wbc/tests/control/robot_model/test_g1_effective_limits.py
git -C third_party/decoupled_wbc diff --check
```

Expected: focused tests pass and the nested diff is whitespace-clean.

- [ ] **Step 7: Commit the complete nested change**

Run from the root worktree:

```bash
git -C third_party/decoupled_wbc add \
  control/main/model_contract.py \
  control/main/teleop/configs/configs.py \
  control/main/teleop/run_g1_control_loop.py \
  control/robot_model/supplemental_info/g1/g1_supplemental_info.py \
  tests/control/main/test_model_contract.py \
  tests/control/main/teleop/test_g1_control_loop_contract.py \
  tests/control/robot_model/test_g1_effective_limits.py
git -C third_party/decoupled_wbc commit -m "feat: attest G1 WBC model contract"
```

Expected: one nested commit on `psi0-simple-bridge-contract`.

- [ ] **Step 8: Verify the production Git inspector against the now-clean nested commit**

Run:

```bash
PYTHONPATH=third_party/decoupled_wbc uv run --group sonic python -c '
from pathlib import Path
import decoupled_wbc
from decoupled_wbc.control.main.model_contract import inspect_git_identity
root = Path(decoupled_wbc.__file__).resolve().parent
identity = inspect_git_identity(root)
assert identity.working_tree_clean is True
assert len(identity.commit) == 40
'
```

Expected: exit zero using the real nested Git SHA and clean-tree check.

## Task 6: Publish the nested commit before updating the root gitlink

**Files:**
- Modify: `third_party/decoupled_wbc` root gitlink
- Modify: `.gitmodules` only if an authorized HTTPS fork replaces the current remote

- [ ] **Step 1: Capture the exact delivery coordinates**

Run:

```bash
git -C third_party/decoupled_wbc rev-parse HEAD
git -C third_party/decoupled_wbc remote get-url origin
git -C third_party/decoupled_wbc status --porcelain --untracked-files=all
```

Expected: a 40-character nested SHA, the configured fetch URL, and empty status.

- [ ] **Step 2: Obtain authority before the external write**

If authority to push `psi0-simple-bridge-contract` to the displayed origin has not been explicitly granted, stop execution here and request it. Do not update the root gitlink while blocked.

- [ ] **Step 3: Push the named nested branch**

After authority is granted, run:

```bash
git -C third_party/decoupled_wbc push -u origin psi0-simple-bridge-contract
```

If the push is rejected, stop and request an authorized HTTPS fork URL. After receiving it, set that URL as the nested origin and update the root `.gitmodules` URL to the same fetchable HTTPS location before retrying the named-branch push.

- [ ] **Step 4: Verify remote reachability before staging the gitlink**

Run:

```bash
git -C third_party/decoupled_wbc ls-remote --exit-code origin refs/heads/psi0-simple-bridge-contract
git -C third_party/decoupled_wbc rev-parse HEAD
```

Expected: the first column returned by `ls-remote` exactly equals the second command's SHA.

- [ ] **Step 5: Commit the reachable gitlink in SIMPLE**

Run:

```bash
git add third_party/decoupled_wbc .gitmodules
git commit -m "build: update attested decoupled WBC"
```

If `.gitmodules` is unchanged, Git stages only the gitlink.

## Task 7: Implement named PSI0 observation/action mapping

**Files:**
- Create: `src/simple/deploy/__init__.py`
- Create: `src/simple/deploy/psi0_simple_bridge.py`
- Create: `tests/test_psi0_bridge_mapping.py`

- [ ] **Step 1: Write failing 43-D to 32-D named-sentinel tests**

Build `q_by_name = {name: float(index)}` from the attested ordered 43 names. Assert every output position against explicit names, not slices:

```python
expected_state_names = (
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
state = build_psi0_observation(q, joint_names, np.array([101, 102, 103, 0.74]))
np.testing.assert_array_equal(state[0, :28], [q_by_name[name] for name in expected_state_names])
np.testing.assert_array_equal(state[0, 28:], [101, 102, 103, 0.74])
```

- [ ] **Step 2: Write failing 36-D to 31-D and inverse mapping tests**

Use unique per-dimension sentinels. Assert waist action RPY becomes goal yaw/roll/pitch by name, arm/hand names land in the connected 31-name order, height/nav are preserved, and `goal_to_psi0_action()` exactly reconstructs the bounded 36-D hold.

- [ ] **Step 3: Verify the deploy package is absent**

Run:

```bash
uv run --group dev pytest -q tests/test_psi0_bridge_mapping.py
```

Expected: collection fails with `ModuleNotFoundError: simple.deploy`.

- [ ] **Step 4: Define immutable public core types**

Create in `psi0_simple_bridge.py`:

```python
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
```

Export these plus the mapping functions from `src/simple/deploy/__init__.py`.

- [ ] **Step 5: Implement mapping exclusively by joint name**

Define `PSI0_STATE_JOINT_NAMES` and `PSI0_ACTION_JOINT_NAMES` as explicit tuples. Implement:

```python
def build_psi0_observation(q, joint_names, command_history_rpyh):
    q = np.asarray(q, dtype=np.float32)
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    if q.shape != (43,) or len(name_to_index) != 43:
        raise ValueError("expected one-to-one 43-joint state")
    history = np.asarray(command_history_rpyh, dtype=np.float32)
    if history.shape != (4,):
        raise ValueError("expected [roll,pitch,yaw,height] history")
    values = [q[name_to_index[name]] for name in PSI0_STATE_JOINT_NAMES]
    return np.concatenate([np.asarray(values, np.float32), history])[None]
```

`map_psi0_action_to_goal(action, upper_body_joint_names, now)` must create a named action dictionary for 28 limb joints plus waist roll/pitch/yaw, gather it in the supplied 31-name order, and return finite float32 arrays with shapes `(31,)`, `(1,)`, `(4,)`. `goal_to_psi0_action()` performs the exact named inverse and forces no implicit clipping.

- [ ] **Step 6: Run mapping tests and commit**

Run:

```bash
uv run --group dev pytest -q tests/test_psi0_bridge_mapping.py
uv run --group dev ruff check src/simple/deploy tests/test_psi0_bridge_mapping.py
git add src/simple/deploy tests/test_psi0_bridge_mapping.py
git commit -m "feat: add named PSI0 bridge mappings"
```

Expected: mapping tests and Ruff pass before the commit.

## Task 8: Implement safety validation, slew limiting, and bounded holds

**Files:**
- Modify: `src/simple/deploy/psi0_simple_bridge.py`
- Create: `tests/test_psi0_bridge_safety.py`

- [ ] **Step 1: Write failing state validity and snapshot freshness tests**

Cover shape, finite values, one-to-one names, `0.05` measured tolerance, `0.10 s` state age, `0.25 s` camera age, and `0.10 s` receive-time skew. Verify an invalid state never replaces `last_valid_q`; in ACTIVE it produces one latched fault reason.

- [ ] **Step 2: Write failing whole-chunk bound and slew tests**

Test exact `(s,36)` shape, finite values, all named effective joint bounds, height `[0.20,0.74]`, planar navigation `[-0.5,0.5]`, turning `[-1,1]`, and yaw `[-pi,pi]`. Assert one invalid element rejects the entire suffix without clipping. Then assert one 20 ms slew step is bounded by:

```python
PER_TICK_LIMITS = {
    "arm": 1.0 / 50.0,
    "hand": 2.0 / 50.0,
    "waist": 0.5 / 50.0,
    "height": 0.1 / 50.0,
    "planar_navigation": 0.5 / 50.0,
    "turning": 2.0 / 50.0,
    "target_yaw": 1.0 / 50.0,
}
```

Use shortest-path angular delta for target yaw.

- [ ] **Step 3: Write failing bounded-hold source-order tests**

Assert all four cases:

1. a measured joint `0.04` rad beyond its effective limit is valid and hold-clamped;
2. `0.051` rad beyond is invalid;
3. stale measured state selects the final-post-slew `last_safe_published` target with zero navigation;
4. no valid state and no prior publication returns `None` and publishes nothing.

Also assert base height clamps to `[0.20,0.74]`, captured holds remain fixed, and the inverse 36-D R0 hold repeats identical values.

- [ ] **Step 4: Implement validation and hold primitives**

Add these immutable dataclasses:

```python
@dataclass(frozen=True)
class TimedRobotState:
    q: np.ndarray
    received_at: float


@dataclass(frozen=True)
class TimedCameraFrame:
    image: np.ndarray
    received_at: float
    producer_timestamp: float | None


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
```

Implement these functions with no I/O:

```python
validate_measured_state(sample, contract, now, tolerance=0.05)
validate_synchronized_snapshot(state, camera, now)
validate_action_suffix(actions, expected_s, contract)
apply_slew_limit(previous_action, requested_action, contract, dt=0.02)
build_bounded_hold(now, last_valid_state, last_safe_goal, contract)
```

`HoldResult.source` is exactly `"measured_clamped"` or `"last_safe_published"`; it includes a tuple of `(joint_name, unclamped, clamped)` records. The state tolerance never changes command limits.

- [ ] **Step 5: Run safety tests and commit**

Run:

```bash
uv run --group dev pytest -q tests/test_psi0_bridge_mapping.py tests/test_psi0_bridge_safety.py
uv run --group dev ruff check src/simple/deploy tests/test_psi0_bridge_mapping.py tests/test_psi0_bridge_safety.py
git add src/simple/deploy/psi0_simple_bridge.py tests/test_psi0_bridge_safety.py
git commit -m "feat: enforce PSI0 bridge safety bounds"
```

Expected: mapping and safety suites pass.

## Task 9: Implement the deterministic RTC scheduler and fault state machine

**Files:**
- Modify: `src/simple/deploy/psi0_simple_bridge.py`
- Create: `tests/test_psi0_bridge_scheduler.py`
- Create: `tests/test_psi0_bridge_lifecycle.py`

- [ ] **Step 1: Write the time-indexed P=8/s=5/d=3 sentinel test**

Use a manual clock, permissive synthetic joint bounds, and bounded unique action value `0.001 * global_tick + 0.000001 * dimension`. Activate at tick 100 and assert:

```python
assert requests[0].observation_tick == 100
assert requests[0].first_action_tick == 103
assert published_ticks[100:103] == ["hold", "hold", "hold"]
assert executed_source_ticks[103:108] == [103, 104, 105, 106, 107]
assert requests[1].observation_tick == 105
assert requests[1].history_tick == 104
assert requests[1].committed_global_ticks == (105, 106, 107)
assert executed_source_ticks[108:113] == [108, 109, 110, 111, 112]
```

Assert the R1 committed actions are the actual post-slew commands later published at 105-107, with no duplicate or skipped tick and no history advance when actions are merely reserved.

- [ ] **Step 2: Write failing protocol/deadline tests**

Parameterize rejection for `d<2`, `d>s`, `d+s>P`, `d>=rtc_training_max_delay`, response lengths `s-1/s+1`, non-finite and out-of-bounds suffixes, missing/mismatched session/sequence/observation/P/s/d/first-action metadata, legacy endpoint, and completion timestamp after `r+d`.

- [ ] **Step 3: Write failing lifecycle and blocked-worker tests**

Assert:

- startup is PAUSED and cannot request before one bounded hold publication;
- `p` toggles PAUSED→ACTIVE and ACTIVE→PAUSED;
- pause/fault/stop increments generation and atomically clears current, staged, committed, deadline, and logical request fields;
- one late result with an old generation is discarded;
- physical worker busy remains true after logical fault and prevents rearm;
- a second worker is never created;
- the first tick after fault publishes zero navigation;
- FAULT requires a later `p` while inputs are fresh and the worker is idle.

- [ ] **Step 4: Run tests and observe missing scheduler types**

Run:

```bash
uv run --group dev pytest -q \
  tests/test_psi0_bridge_scheduler.py \
  tests/test_psi0_bridge_lifecycle.py
```

Expected: failures identify absent `PolicyContract`, `RtcRequest`, `RtcResult`, `InferencePort`, and `Psi0SimpleBridge` behavior.

- [ ] **Step 5: Define the RTC interfaces and exact contract validation**

Add `import re` with the other core imports, then add:

```python
@dataclass(frozen=True)
class PolicyContract:
    schema: str
    checkpoint_sha256: str
    dataset_manifest_sha256: str
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
        return cls(
            schema=str(payload["schema"]),
            checkpoint_sha256=str(payload["checkpoint_sha256"]),
            dataset_manifest_sha256=str(payload["dataset_manifest_sha256"]),
            server_commit=str(payload["server_commit"]),
            converter_layout=str(payload["converter_layout"]),
            prediction_horizon=int(payload["prediction_horizon"]),
            execution_horizon=int(payload["execution_horizon"]),
            rtc_delay_steps=int(payload["rtc_delay_steps"]),
            rtc_training_max_delay=int(payload["rtc_training_max_delay"]),
            action_frequency_hz=int(payload["action_frequency_hz"]),
            observation_dim=int(payload["observation_dim"]),
            action_dim=int(payload["action_dim"]),
            endpoint=str(payload["rtc_endpoint"]),
            request_semantics=str(payload["request_semantics"]),
            response_semantics=str(payload["response_semantics"]),
            image_key=str(payload["image_key"]),
            camera_color_order=str(payload["camera_color_order"]),
            rtc_enabled=payload["rtc_enabled"] is True,
            test_only=payload.get("test_only", False) is True,
        )

    def validate(self) -> None:
        if self.schema != "simple.psi0.policy-contract.v2":
            raise ValueError("unsupported policy contract schema")
        if not re.fullmatch(r"[0-9a-f]{64}", self.checkpoint_sha256):
            raise ValueError("invalid checkpoint SHA-256")
        if not re.fullmatch(r"[0-9a-f]{64}", self.dataset_manifest_sha256):
            raise ValueError("invalid dataset-manifest SHA-256")
        if not re.fullmatch(r"[0-9a-f]{40}", self.server_commit):
            raise ValueError("invalid PSI0 server commit")
        if self.converter_layout != "g1_simple_32_rpyh_v2":
            raise ValueError("unsupported PSI0 converter layout")
        if not self.rtc_enabled:
            raise ValueError("RTC must be enabled")
        if self.action_frequency_hz != 50 or self.observation_dim != 32 or self.action_dim != 36:
            raise ValueError("policy dimensions/frequency do not match bridge")
        d, s, p = self.rtc_delay_steps, self.execution_horizon, self.prediction_horizon
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
```

These protocol bodies define the injected boundary only; no production I/O belongs in the core file.

- [ ] **Step 6: Implement scheduler state with global tick ownership**

`Psi0SimpleBridge` owns these fields under one lock: `state`, `generation`, `tick_index`, `session_id`, `request_seq`, `request_tick`, `handoff_tick`, `next_request_tick`, `scheduled_actions: dict[int,np.ndarray]`, `staged_first_tick`, `staged_actions`, `command_history_rpyh`, `last_valid_state`, `last_safe_goal`, `fault_reason`, and logical-request-active. Initialize `command_history_rpyh` exactly as `np.array([0.0, 0.0, 0.0, 0.74], dtype=np.float32)`. The injected `InferencePort.busy` remains separate.

Implement `activate()` to require PAUSED, fresh synchronized inputs, idle worker, and one successfully published bounded hold. At `r0`, schedule exactly `d` copies of the inverse 36-D hold, submit reset R0, and set handoff `r0+d` and next request `r0+s`.

Implement `tick()` in the approved order:

1. validate latest inputs and drain at most one completed result;
2. reject stale generation, bad metadata, bad suffix, or completion after the monotonic deadline corresponding to `r+d`;
3. at a response handoff, install exactly `s` suffix actions starting at `r+d`;
4. at `next_request_tick`, precompute and freeze final post-slew commands for ticks `r:r+d`, submit them with observation/history from `r-1`, then increment next request tick by `s`;
5. select the scheduled action for the current tick, or atomically enter FAULT if the required suffix is absent;
6. validate/map/slew/publish or shadow-preview the command;
7. only after successful publication/preview consumption, update `last_safe_goal` and `[roll,pitch,yaw,height]` history.

Implement `enter_fault(reason)`, `pause()`, and `stop()` as single locked transitions that increment generation and clear all logical action/request state. Never clear `InferencePort.busy` from the core.

- [ ] **Step 7: Run scheduler/lifecycle tests and commit**

Run:

```bash
uv run --group dev pytest -q \
  tests/test_psi0_bridge_mapping.py \
  tests/test_psi0_bridge_safety.py \
  tests/test_psi0_bridge_scheduler.py \
  tests/test_psi0_bridge_lifecycle.py
uv run --group dev ruff check src/simple/deploy tests/test_psi0_bridge_*.py
git add src/simple/deploy/psi0_simple_bridge.py \
  tests/test_psi0_bridge_scheduler.py tests/test_psi0_bridge_lifecycle.py
git commit -m "feat: add fail-closed PSI0 RTC scheduler"
```

Expected: all pure-core tests pass and no external transport is initialized.

## Task 10: Implement policy/WBC preflight and camera/state adapters

**Files:**
- Create: `scripts/psi0_simple_real_bridge.py`
- Create: `tests/test_psi0_bridge_preflight.py`
- Create: `tests/test_psi0_bridge_camera.py`

- [ ] **Step 1: Write failing policy contract comparison tests**

Load one local v2 contract and one server dictionary through `PolicyContract.from_dict()`. Require exact equality for checkpoint/dataset hashes, server commit, converter layout, dimensions, frequency, P/s/d, exclusive training delay, endpoint/semantics, image key, RGB order, and `test_only`. In `sim-control`, any missing/mismatch is fatal. A matching `test_only=true` pair is accepted only when the connected WBC reports `env_type=sim`; otherwise it is fatal. In shadow, return `policy_certified=False` and a stable list of mismatched fields without creating a publisher.

- [ ] **Step 2: Write failing WBC payload/digest tests**

Use a fixture response with the approved top-level values and nested model contract. Test:

- three-second bounded service wait and clean destruction on timeout;
- response digest recomputation;
- root gitlink SHA equality and clean local nested tree;
- full equality of 43/31 ordered names, effective limits, URDF/ONNX hashes/signatures, and contract digest;
- every top-level WBC field in the approved table;
- rejection before publisher construction for each mutation.

- [ ] **Step 3: Write failing graph-ownership and camera codec tests**

Graph tests require `0 publishers/1 subscription` before creating the `sim-control` goal publisher and `1/1` after. Reject `1/1` or `0/0` preflight. Shadow records counts and never creates a publisher.

Camera tests serialize spatially separate red and blue patches through the real nested `ImageMessageSchema`. Assert `rgb` leaves numeric channels unchanged and `bgr` swaps once, the selected key is mandatory, output is contiguous `uint8` HWC, and reader receive time uses the injected monotonic clock.

- [ ] **Step 4: Verify the CLI module is absent**

Run:

```bash
PYTHONPATH=third_party/decoupled_wbc uv run --group dev --group sonic pytest -q \
  tests/test_psi0_bridge_preflight.py tests/test_psi0_bridge_camera.py
```

Expected: collection fails for missing runtime adapters.

- [ ] **Step 5: Define an argparse surface with no real-control path**

Expose exactly these operational options:

```text
--mode {shadow,sim-control}
--server-host HOST
--server-port PORT
--instruction TEXT
--policy-contract PATH
--camera-host HOST
--camera-port PORT
--camera-source-key KEY
--camera-color-order {rgb,bgr}
--ros-domain-id 42
--unitree-domain-id 42
--metrics-jsonl PATH
```

Validate `BridgeMode(args.mode)` and ensure no parser branch accepts `real`, `real-control`, a robot NIC, or a low-level DDS topic.

- [ ] **Step 6: Implement focused runtime adapters**

Create concrete `BoundedWbcConfigClient`, `RosStateSource`, `RosGoalPublisher`, and `ComposedCameraReader` classes in the script. Compose them with:

```python
@dataclass(frozen=True)
class RuntimeAdapters:
    wbc_config_client: BoundedWbcConfigClient
    state_source: RosStateSource
    camera_reader: ComposedCameraReader
    goal_publisher: RosGoalPublisher | None
```

`BoundedWbcConfigClient.get_config(timeout_s=3.0)` owns a temporary rclpy node/client, loops `wait_for_service()` in short intervals until the monotonic deadline, waits for the async result only within the remaining budget, and then destroys the client/node. `RosStateSource.poll()` returns `TimedRobotState | None`; `RosGoalPublisher.publish(goal)` serializes the five approved goal fields; every adapter exposes `close()`. The camera socket is created, polled with 100 ms timeout, decoded, and closed only inside its reader thread; `ComposedCameraReader.close(timeout_s=0.5)` signals and joins that thread.

Implement canonical local model-contract building by importing the nested helper, and read the expected nested SHA with:

```python
subprocess.run(
    ["git", "rev-parse", "HEAD:third_party/decoupled_wbc"],
    cwd=repository_root,
    check=True,
    capture_output=True,
    text=True,
)
```

- [ ] **Step 7: Implement preflight sequencing before publisher construction**

`run_bridge()` must perform, in order: parse/validate mode; WBC bounded query; connected/local model digest comparison; local/server policy contract comparison; ROS graph ownership check; then publisher construction only for `sim-control`. Shadow never calls the publisher factory. Record every result in metrics.

- [ ] **Step 8: Run adapter/preflight tests and commit**

Run:

```bash
PYTHONPATH=third_party/decoupled_wbc uv run --group dev --group sonic pytest -q \
  tests/test_psi0_bridge_preflight.py tests/test_psi0_bridge_camera.py
uv run --group dev ruff check scripts/psi0_simple_real_bridge.py \
  tests/test_psi0_bridge_preflight.py tests/test_psi0_bridge_camera.py
git add scripts/psi0_simple_real_bridge.py \
  tests/test_psi0_bridge_preflight.py tests/test_psi0_bridge_camera.py
git commit -m "feat: add bounded PSI0 bridge preflight"
```

Expected: all preflight/camera tests pass, including proof that rejected preflight never constructs a publisher.

## Task 11: Implement the local keyboard, asynchronous inference, metrics, and bounded shutdown

**Files:**
- Modify: `scripts/psi0_simple_real_bridge.py`
- Create: `tests/test_psi0_bridge_runtime.py`
- Create: `tests/test_psi0_bridge_shutdown.py`
- Create: `scripts/tests/bridge_subprocess_fixture.py`

- [ ] **Step 1: Write failing keyboard and single-worker tests**

Use a pseudo-terminal. Assert the runtime starts PAUSED, only local byte `p` creates a toggle event, terminal settings restore in all exits, blocked HTTP never blocks 50 Hz ticks, one worker is used for the process lifetime, and a busy invalidated request refuses rearm until it physically returns.

- [ ] **Step 2: Write failing no-state Ctrl-C subprocess test**

Launch `bridge_subprocess_fixture.py`, send Ctrl-C before any state, and assert:

```python
assert goals == []
assert inference_requests == []
assert elapsed_s <= 0.5
assert report["publisher_closed"] is True
assert report["camera_closed"] is True
assert report["terminal_restored"] is True
assert report["live_non_daemon_bridge_threads"] == []
```

Rebind every fixture port after exit.

- [ ] **Step 3: Write failing five-second in-flight shutdown test**

The local handler signals after accepting `/act-rtc-v1`, then withholds response beyond the client's five-second read timeout. Immediately send Ctrl-C. Assert exit within 6.5 seconds, exactly 25 identical bounded zero-navigation messages on consecutive 20 ms scheduled ticks, no 26th message, publisher closure immediately after message 25, camera/terminal cleanup, no live non-daemon bridge thread, and reusable ports.

- [ ] **Step 4: Implement the inference worker**

`HttpInferenceWorker` owns one daemon thread and one `HttpActionClient(timeout=5.0)`. It has a capacity-one input slot, capacity-one result slot, a physical-busy event, and methods matching `InferencePort`. `submit()` rejects when busy; the thread converts `RtcRequest` to `/act-rtc-v1` history with reset only for R0, timestamps completion with monotonic time, and always clears physical busy in `finally`. It never retries and never starts a replacement worker.

- [ ] **Step 5: Implement local-keyboard and metrics ownership**

`LocalKeyboard` owns the terminal file descriptor, saves original `termios`, enters cbreak, polls with `select`, enqueues only `p`, and restores settings in `finally`. `JsonlMetrics` writes one compact record per preflight, state transition, request, result, publish/preview, fault, and shutdown event; every record includes mode, state, generation, monotonic time, `published`, and `policy_certified`.

- [ ] **Step 6: Implement exact shutdown branching**

On Ctrl-C:

1. latch stop, increment generation, and clear all policy buffers/bookkeeping;
2. call `build_bounded_hold(now)`;
3. if it returns a hold, publish exactly 25 identical goals on consecutive 20 ms schedule ticks, then close the publisher immediately and forbid a 26th publish;
4. if it returns `None`, publish zero messages and continue immediately;
5. signal camera exit, prevent new inference work, join the worker for at most 5.5 seconds, close messaging, restore terminal, and exit;
6. enforce the overall 6.5-second bound for the in-flight case and 0.5-second bound when neither final hold nor HTTP join is pending.

The runtime never signals or terminates the WBC, simulator, policy server, or any robot process.

- [ ] **Step 7: Run runtime/shutdown tests and commit**

Run:

```bash
PYTHONPATH=third_party/decoupled_wbc uv run --group dev --group sonic pytest -q \
  tests/test_psi0_bridge_runtime.py tests/test_psi0_bridge_shutdown.py
uv run --group dev ruff check scripts/psi0_simple_real_bridge.py \
  scripts/tests/bridge_subprocess_fixture.py \
  tests/test_psi0_bridge_runtime.py tests/test_psi0_bridge_shutdown.py
git add scripts/psi0_simple_real_bridge.py scripts/tests/bridge_subprocess_fixture.py \
  tests/test_psi0_bridge_runtime.py tests/test_psi0_bridge_shutdown.py
git commit -m "feat: add safe PSI0 bridge runtime lifecycle"
```

Expected: both shutdown branches and single-worker lifecycle pass without ROS, ZMQ, GPU, DDS, or MuJoCo.

## Task 12: Add protocol-faithful fake servers and integration tests

**Files:**
- Create: `scripts/tests/fake_psi0_rtc_server.py`
- Create: `scripts/tests/fake_composed_camera_server.py`
- Create: `scripts/tests/fixtures/psi0_policy_contract_test_v2.json`
- Create: `tests/test_psi0_bridge_fake_integration.py`

- [ ] **Step 1: Create the explicit test-only contract fixture**

Write JSON containing:

```json
{
  "schema": "simple.psi0.policy-contract.v2",
  "test_only": true,
  "checkpoint_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "dataset_manifest_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "server_commit": "cccccccccccccccccccccccccccccccccccccccc",
  "converter_layout": "g1_simple_32_rpyh_v2",
  "observation_dim": 32,
  "action_dim": 36,
  "action_frequency_hz": 50,
  "prediction_horizon": 30,
  "execution_horizon": 24,
  "rtc_delay_steps": 6,
  "rtc_training_max_delay": 7,
  "rtc_enabled": true,
  "rtc_endpoint": "/act-rtc-v1",
  "request_semantics": "exact-post-slew-committed-prefix",
  "response_semantics": "denormalized-executable-suffix",
  "image_key": "rgb_head_stereo_left",
  "camera_color_order": "rgb"
}
```

- [ ] **Step 2: Write the failing fake-integration tests**

Start each fake on an OS-assigned loopback port. Assert `/contract` matches the fixture. Send R0 with `observation_tick=100`, `d=6`, and six identical holds; require `(24,36)` actions and exact echoed metadata with first tick 106. Send R1 with unique committed actions and assert the fake records them unchanged. Configure one request for `0.30 s` latency and prove the runtime faults at the RTC deadline, discards the late response, and does not send a second concurrent request.

For the camera fake, require the selected key, JPEG round trip, red/blue dominance, and reader shutdown within 0.5 seconds.

- [ ] **Step 3: Implement the fake RTC HTTP server**

Use `ThreadingHTTPServer` bound to `127.0.0.1`. Implement only `GET /contract` and `POST /act-rtc-v1`; return 404 for `/act`. Deserialize requests with `RequestMessage.deserialize()`. Validate the supplied `(d,36)` committed prefix. Generate a safe stationary suffix from the final committed command:

```python
actions = np.repeat(committed_actions[-1:], execution_horizon, axis=0).astype(np.float32)
actions[:, 32:36] = 0.0
```

Return NumPy-serialized actions and metadata echoing session, sequence, observation tick, P/s/d, and `first_action_tick=observation_tick+d`. Expose thread-safe request records and one-shot configurable latency; close and join the server thread explicitly.

- [ ] **Step 4: Implement the fake composed-camera server**

Use the real nested `ImageMessageSchema` and `SensorServer` on loopback. Publish a contiguous RGB image whose left patch is `[255,0,0]` and right patch is `[0,0,255]`, plus the selected key timestamp. Expose a stop event, close the socket with `LINGER=0`, join the thread, and make port reuse an assertion.

- [ ] **Step 5: Run fake integration tests and commit**

Run:

```bash
PYTHONPATH=third_party/decoupled_wbc uv run --group dev --group sonic pytest -q \
  tests/test_psi0_bridge_fake_integration.py
uv run --group dev ruff check scripts/tests/fake_psi0_rtc_server.py \
  scripts/tests/fake_composed_camera_server.py tests/test_psi0_bridge_fake_integration.py
git add scripts/tests/fake_psi0_rtc_server.py \
  scripts/tests/fake_composed_camera_server.py \
  scripts/tests/fixtures/psi0_policy_contract_test_v2.json \
  tests/test_psi0_bridge_fake_integration.py
git commit -m "test: add PSI0 bridge protocol fakes"
```

Expected: fake policy/camera integration passes and every port can be rebound.

## Task 13: Add the isolated 15-second MuJoCo smoke test

**Files:**
- Create: `scripts/tests/smoke_psi0_simple_bridge.py`
- Create: `tests/test_psi0_bridge_smoke_driver.py`

- [ ] **Step 1: Write a failing smoke-driver orchestration test**

Mock `subprocess.Popen` and assert the driver prepares exactly four children: MuJoCo WBC, fake camera, fake RTC server, and bridge in a pseudo-terminal. Assert all commands use ROS domain 42, Unitree domain 42, Linux loopback, disabled on/offscreen rendering, waist enabled, hands enabled, and fixed ports 15555/22086. Assert the driver refuses any non-loopback WBC interface or domain mismatch before spawning.

- [ ] **Step 2: Implement process ownership and cleanup**

The driver must:

1. preflight that goal graph counts are `0 publishers/1 subscription` after WBC startup and before bridge startup;
2. launch every child in its own process group and record PID/command/start time;
3. wait for each bounded readiness signal;
4. send local `p` through the bridge pseudo-terminal at second 3;
5. make the fake policy's next request at second 11 take exactly 0.30 seconds;
6. require worker idle at second 13, send Ctrl-C through the pseudo-terminal, and finish by second 15;
7. reap only its own child process groups in `finally`, never using broad `pkill` or killing unrelated processes;
8. verify terminal restoration and rebind every port.

The WBC command is:

```bash
ROS_DOMAIN_ID=42 PYTHONPATH=third_party/decoupled_wbc \
uv run --group sonic python -m decoupled_wbc.control.main.teleop.run_g1_control_loop \
  --interface sim \
  --simulator mujoco \
  --messaging-backend ros2 \
  --enable-waist \
  --with-hands \
  --domain-id 42 \
  --no-enable-onscreen \
  --no-enable-offscreen
```

Before accepting the exact boolean spelling, run the module with `--help` and update the smoke command to the spelling Tyro actually prints; the test must assert that exact final command.

- [ ] **Step 3: Implement metric assertions**

Parse JSONL and captured goals. Require 49-51 Hz mean publication per steady phase, maximum 60 ms gap, no blocked-main-loop gap, request spacing of 24 ticks with six immutable post-slew committed actions, no tick duplication/skip, fault within 0.14 seconds of the delayed request, first fault goal navigation exactly zero, no later policy action, discarded late result, idle worker before shutdown, goal publisher count back to zero, no live child/thread, and zero real-interface connections.

- [ ] **Step 4: Run the driver unit test without starting MuJoCo**

Run:

```bash
uv run --group dev pytest -q tests/test_psi0_bridge_smoke_driver.py
uv run --group dev ruff check scripts/tests/smoke_psi0_simple_bridge.py \
  tests/test_psi0_bridge_smoke_driver.py
```

Expected: orchestration and fail-closed command validation pass with all process calls mocked.

- [ ] **Step 5: Commit the smoke driver before executing it**

```bash
git add scripts/tests/smoke_psi0_simple_bridge.py tests/test_psi0_bridge_smoke_driver.py
git commit -m "test: add isolated PSI0 bridge smoke driver"
```

- [ ] **Step 6: Run the real isolated simulation smoke test**

Before running, confirm there is no real-interface WBC or existing goal publisher. Then run:

```bash
ROS_DOMAIN_ID=42 uv run --group sonic python scripts/tests/smoke_psi0_simple_bridge.py \
  --duration-s 15 \
  --unitree-domain-id 42 \
  --camera-port 15555 \
  --policy-port 22086
```

Expected: exit zero with a metrics summary containing all approved smoke pass criteria. This initializes Unitree SDK2 DDS only on loopback/domain 42; it does not contact a robot.

## Task 14: Document operations and the external live-server handoff

**Files:**
- Modify: `README.md`
- Test: `tests/test_psi0_bridge_cli.py`

- [ ] **Step 1: Add CLI contract tests**

Assert `--help` lists only `shadow` and `sim-control`; invalid `real-control` exits nonzero; shadow constructs no publisher; `p` is the only activation toggle; and policy client timeout is exactly five seconds while generic `HttpActionClient` remains unbounded by default.

- [ ] **Step 2: Add the README deployment section**

Document:

- this milestone is simulation/shadow only;
- PC2 inputs, 43→32 and 36→31+1+4 named mappings;
- WBC command with `--interface sim --enable-waist --with-hands --domain-id 42`;
- fake server/camera commands and the bridge shadow/sim-control commands;
- startup PAUSED, local `p` transitions, FAULT rearm behavior, Ctrl-C conditional final hold, and WBC ownership after bridge exit;
- camera key/color validation and saved-sample gate;
- the current checkpoint/RTX 3060 latency failure and why `/act` is ineligible;
- the required external `/act-rtc-v1` request/response metadata and policy-contract fields;
- explicit warning never to run the C++ SONIC controller and Python decoupled WBC real low-level loops concurrently.

Use this shadow command pattern:

```bash
PYTHONPATH=third_party/decoupled_wbc uv run --group sonic python \
  scripts/psi0_simple_real_bridge.py \
  --mode shadow \
  --server-host 127.0.0.1 \
  --server-port 22086 \
  --instruction "pick up the object" \
  --policy-contract scripts/tests/fixtures/psi0_policy_contract_test_v2.json \
  --camera-host 127.0.0.1 \
  --camera-port 15555 \
  --camera-source-key rgb_head_stereo_left \
  --camera-color-order rgb \
  --ros-domain-id 42 \
  --unitree-domain-id 42 \
  --metrics-jsonl outputs/psi0_bridge_shadow.jsonl
```

State plainly that the test-only contract cannot certify a live policy or any future real-control mode.

- [ ] **Step 3: Run CLI tests and documentation command checks**

Run:

```bash
PYTHONPATH=third_party/decoupled_wbc uv run --group dev --group sonic pytest -q \
  tests/test_psi0_bridge_cli.py
PYTHONPATH=third_party/decoupled_wbc uv run --group sonic python \
  scripts/psi0_simple_real_bridge.py --help
```

Expected: CLI tests pass and help exactly matches the documented options.

- [ ] **Step 4: Commit documentation and CLI acceptance tests**

```bash
git add README.md tests/test_psi0_bridge_cli.py
git commit -m "docs: add PSI0 bridge simulation workflow"
```

## Task 15: Run complete verification and prove clean recursive delivery

**Files:**
- No intended source changes; fix discovered failures in the owning task's files and commit them atomically.

- [ ] **Step 1: Run all root deterministic tests**

Run:

```bash
PYTHONPATH=third_party/decoupled_wbc uv run --group dev --group sonic pytest -q \
  tests/test_postprocess_psi0.py \
  tests/test_certify_psi0_policy_contract.py \
  tests/test_http_action_client.py \
  tests/test_psi0_bridge_mapping.py \
  tests/test_psi0_bridge_safety.py \
  tests/test_psi0_bridge_scheduler.py \
  tests/test_psi0_bridge_lifecycle.py \
  tests/test_psi0_bridge_preflight.py \
  tests/test_psi0_bridge_camera.py \
  tests/test_psi0_bridge_runtime.py \
  tests/test_psi0_bridge_shutdown.py \
  tests/test_psi0_bridge_fake_integration.py \
  tests/test_psi0_bridge_smoke_driver.py \
  tests/test_psi0_bridge_cli.py
```

Expected: every listed test passes with zero skips.

- [ ] **Step 2: Run nested deterministic tests at the recorded gitlink**

Run:

```bash
uv run --group sonic pytest -q \
  third_party/decoupled_wbc/tests/control/main/test_model_contract.py \
  third_party/decoupled_wbc/tests/control/main/teleop/test_g1_control_loop_contract.py \
  third_party/decoupled_wbc/tests/control/robot_model/test_g1_effective_limits.py
```

Expected: all nested contract tests pass.

- [ ] **Step 3: Run lint, formatting, and whitespace checks**

Run full Ruff checks on every new file and fatal-error checks on modified legacy files:

```bash
uv run --group dev ruff check src/simple/deploy scripts/psi0_simple_real_bridge.py \
  scripts/certify_psi0_policy_contract.py \
  scripts/tests/fake_psi0_rtc_server.py \
  scripts/tests/fake_composed_camera_server.py \
  scripts/tests/bridge_subprocess_fixture.py \
  scripts/tests/smoke_psi0_simple_bridge.py tests
uv run --group dev ruff format --check src/simple/deploy scripts/psi0_simple_real_bridge.py \
  scripts/certify_psi0_policy_contract.py \
  scripts/tests/fake_psi0_rtc_server.py \
  scripts/tests/fake_composed_camera_server.py \
  scripts/tests/bridge_subprocess_fixture.py \
  scripts/tests/smoke_psi0_simple_bridge.py tests
uv run --group dev ruff check --select E9,F63,F7,F82 \
  src/simple/baselines/client.py scripts/postprocess_psi0.py
git diff --check main...HEAD
git -C third_party/decoupled_wbc diff --check
```

Expected: every command exits zero.

- [ ] **Step 4: Re-run the isolated 15-second smoke test**

Run the exact Task 13 command and preserve its metrics JSONL as a verification artifact. Expected: all pass criteria and zero real-interface connections.

- [ ] **Step 5: Verify the external nested branch still contains the exact gitlink SHA**

Run:

```bash
git -C third_party/decoupled_wbc ls-remote --exit-code origin refs/heads/psi0-simple-bridge-contract
git rev-parse HEAD:third_party/decoupled_wbc
```

Expected: the remote branch SHA and root gitlink SHA are identical.

- [ ] **Step 6: Prove delivery from a clean non-local recursive clone**

Run:

```bash
SIMPLE_CLEAN_CHECKOUT=$(mktemp -d /tmp/simple-psi0-clean.XXXXXX)
git clone --no-local --branch feat/psi0-simple-bridge \
  /home/jihun/work/SIMPLE "$SIMPLE_CLEAN_CHECKOUT"
git -C "$SIMPLE_CLEAN_CHECKOUT" submodule update --init --recursive
git -C "$SIMPLE_CLEAN_CHECKOUT" rev-parse HEAD:third_party/decoupled_wbc
git -C "$SIMPLE_CLEAN_CHECKOUT/third_party/decoupled_wbc" rev-parse HEAD
git -C "$SIMPLE_CLEAN_CHECKOUT" status --short
```

Expected: both nested SHAs match, recursive initialization succeeds from the recorded remote, and the clean clone has no status entries.

Run the focused root and nested test commands again from `$SIMPLE_CLEAN_CHECKOUT`; expected: the same passing counts as Steps 1-2.

- [ ] **Step 7: Record final repository and process state**

Run:

```bash
git status --short
git log --oneline --decorate -12
pgrep -af 'run_g1_control_loop|psi0_simple_real_bridge|fake_psi0_rtc_server|mujoco'
```

Expected: feature worktree status is clean, commits are atomic and ordered, and no implementation-owned runtime process remains. Do not remove or stop a process that predates the test harness.

## Requirement coverage audit

| Approved requirement | Implemented/tested in |
|---|---|
| Correct causal RPY history and `0.74` | Tasks 1-2 |
| Certified local policy contract and fake-only exception | Tasks 2, 10, 12 |
| WBC env/domain serialization and 31-joint configuration | Task 4 |
| Correct hand limits with shoulder allowlist | Task 4 |
| Connected WBC Git/joint/limit/URDF/ONNX attestation | Tasks 5-6, 10 |
| Exact named 43→32 and 36→31+1+4 mapping | Task 7 |
| RGB/BGR codec and viewpoint key contract | Tasks 10, 12 |
| State bounds, freshness, skew, and last-valid behavior | Task 8 |
| Bounded holds and no-hold startup/shutdown | Tasks 8, 11 |
| R0 full RTC metadata and committed hold prefix | Task 9 |
| 103/108 sentinel handoffs and post-slew prefix | Task 9 |
| Whole-suffix bounds and per-tick slew limits | Task 8 |
| PAUSED/ACTIVE/FAULT/STOPPED and local `p` | Tasks 9, 11 |
| Single asynchronous worker, deadlines, late generation discard | Tasks 9, 11-12 |
| Publisher ownership and shadow no-publisher guarantee | Task 10 |
| Exact 25-message shutdown and no 26th message | Task 11 |
| Short smoke stall and separate long shutdown test | Tasks 11-13 |
| Isolated loopback/domain-42 MuJoCo smoke test | Task 13 |
| Reachable nested commit and clean recursive checkout | Tasks 6, 15 |
| No real-control path and documented later gates | Tasks 0, 10, 14-15 |

## Execution checkpoints

1. Stop after Task 5 if the nested tests do not pass.
2. Stop at Task 6 until explicit push authority and a fetchable remote are available.
3. Stop before Task 13 if ROS graph ownership is not exactly `0 publishers/1 WBC subscription`, if any interface is not loopback, or if either domain is not 42.
4. Stop before replacing the fake policy server. The external PSI0 repository must independently deliver `/act-rtc-v1`, a matching commit in the certified contract, and warmed p99 latency at or below `(d-1)/50` seconds.
5. Completing this plan does not authorize real-robot deployment. That remains a separate reviewed design and implementation phase.
