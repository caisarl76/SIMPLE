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
- Modify `scripts/postprocess_psi0.py`: chronological RPY conversion, `0.74` initial height, and per-episode raw/converter provenance.
- Create root tests under `tests/` split by converter, HTTP transport, mapping, safety/holds, scheduler, preflight/camera, runtime shutdown, and certification.
- Create `scripts/tests/fake_psi0_rtc_server.py`: fake v2 contract and deterministic RTC HTTP server.
- Create `scripts/tests/fake_composed_camera_server.py`: real `ImageMessageSchema` JPEG publisher with red/blue sentinels.
- Create `scripts/tests/bridge_subprocess_fixture.py`: injectable fake-runtime subprocess used for signal and shutdown tests.
- Create `scripts/tests/smoke_psi0_simple_bridge.py`: isolated 15-second MuJoCo orchestration and metrics assertions.
- Create `scripts/benchmark_psi0_rtc_server.py`: 10-request warmup plus 100-request live RTC certification and co-located evidence bundle.
- Create `tests/test_benchmark_psi0_rtc_server.py`: deterministic failure accounting and p99/evidence-bundle tests.
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

## Plan execution rule

Every test snippet below is literal minimum test code, not pseudocode. Helper names are introduced in the same task before a later test uses them. During implementation, complete one checkbox, run the named focused test, and only then continue; no checkbox may be expanded into an unreviewed multi-hour action. When a code block contains several functions, add and run one function at a time in source order (each is a separate 2-5 minute red/green action).

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
import hashlib

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


def test_conversion_provenance_binds_source_episode_and_converter(tmp_path):
    from scripts import postprocess_psi0

    raw = tmp_path / "episode_000007.parquet"
    raw.write_bytes(b"raw-episode-seven")
    result = postprocess_psi0.build_conversion_provenance(
        source_path=raw,
        source_episode_index=7,
        skip=60,
        downsample=2,
        converter_commit="1" * 40,
    )
    assert set(result) == {
        "source_episode_index",
        "source_parquet_sha256",
        "skip",
        "downsample",
        "converter_commit",
    }
    assert result["source_episode_index"] == 7
    assert result["source_parquet_sha256"] == hashlib.sha256(
        b"raw-episode-seven"
    ).hexdigest()
    assert result["skip"] == 60
    assert result["downsample"] == 2
    assert result["converter_commit"] == "1" * 40
```

- [ ] **Step 2: Run the test and verify the legacy converter fails**

Run:

```bash
uv run --group dev pytest -q tests/test_postprocess_psi0.py
```

Expected: the two layout assertions exposing row reversal and `0.75` fail, and collection also reports the missing provenance helper.

- [ ] **Step 3: Correct the two converter values**

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

- [ ] **Step 4: Add a deterministic provenance record to each processed episode**

Add `sha256_file()`, `resolve_converter_commit()`, and this pure helper:

```python
def build_conversion_provenance(
    source_path: Path,
    source_episode_index: int,
    skip: int,
    downsample: int,
    converter_commit: str,
) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", converter_commit) is None:
        raise ValueError("converter commit must be a 40-character lowercase Git SHA")
    return {
        "source_episode_index": source_episode_index,
        "source_parquet_sha256": sha256_file(source_path),
        "skip": skip,
        "downsample": downsample,
        "converter_commit": converter_commit,
    }
```

`resolve_converter_commit()` runs `git log -1 --format=%H -- scripts/postprocess_psi0.py` from the SIMPLE repository root and rejects a malformed result. Resolve it once at CLI startup. Store the returned `conversion_provenance` dictionary in each row written to `meta/episodes.jsonl`, using the exact `data_path`, source `ep_index`, `args.skip`, and `args.downsample` for that row. The later certification tool must not accept provenance supplied only on its command line.

- [ ] **Step 5: Run focused tests and formatting checks**

Run:

```bash
uv run --group dev pytest -q tests/test_postprocess_psi0.py
uv run --group dev ruff check tests/test_postprocess_psi0.py
uv run --group dev ruff check --select E9,F63,F7,F82 scripts/postprocess_psi0.py
uv run --group dev ruff format --check tests/test_postprocess_psi0.py
```

Expected: 3 tests pass and both Ruff commands exit zero.

- [ ] **Step 6: Commit the converter correction and provenance writer**

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
import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.certify_psi0_policy_contract import (
    build_policy_contract_payload,
    certify_layout,
    load_bound_episode,
)


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


def write_bound_episode(tmp_path, *, recorded_raw_hash=None, converter_commit="2" * 40):
    raw_path = tmp_path / "episode_000007.parquet"
    processed_path = tmp_path / "episode_000003.parquet"
    episodes_path = tmp_path / "episodes.jsonl"
    cmd = np.zeros((5, 9), dtype=np.float32)
    cmd[:, 3:6] = np.arange(15, dtype=np.float32).reshape(5, 3)
    pq.write_table(pa.table({"observation.amo_policy_command": cmd.tolist()}), raw_path)
    initial = np.array([0, 0, 0, 0, 0, 0, 0.74, 0.74, 0.74], np.float32)
    history = np.concatenate([initial[None], cmd[:-1]], axis=0)
    states = np.zeros((5, 32), dtype=np.float32)
    states[:, 28:31] = history[:, 3:6][:, ::-1]
    states[:, 31] = history[:, 6]
    pq.write_table(
        pa.table({"states": states.tolist(), "episode_index": [3] * 5}),
        processed_path,
    )
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    record = {
        "episode_index": 3,
        "length": 5,
        "conversion_provenance": {
            "source_episode_index": 7,
            "source_parquet_sha256": recorded_raw_hash or raw_hash,
            "skip": 0,
            "downsample": 1,
            "converter_commit": converter_commit,
        },
    }
    episodes_path.write_text(json.dumps(record) + "\n")
    return raw_path, processed_path, episodes_path, states, history


def test_load_bound_episode_proves_same_raw_and_processed_episode(tmp_path):
    raw, processed, episodes, expected_states, expected_history = write_bound_episode(
        tmp_path
    )
    result = load_bound_episode(raw, processed, episodes)
    np.testing.assert_array_equal(result.stored_state, expected_states)
    np.testing.assert_array_equal(result.history_cmd, expected_history)
    assert result.source_episode_index == 7
    assert result.processed_episode_index == 3
    assert result.converter_commit == "2" * 40
    assert result.raw_episode_sha256 == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert result.processed_episode_sha256 == hashlib.sha256(
        processed.read_bytes()
    ).hexdigest()


def test_load_bound_episode_rejects_cross_episode_or_unattested_converter(tmp_path):
    raw, processed, episodes, *_ = write_bound_episode(
        tmp_path, recorded_raw_hash="f" * 64
    )
    with pytest.raises(ValueError, match="raw episode hash mismatch"):
        load_bound_episode(raw, processed, episodes)
    raw, processed, episodes, *_ = write_bound_episode(
        tmp_path, converter_commit="not-a-commit"
    )
    with pytest.raises(ValueError, match="converter commit"):
        load_bound_episode(raw, processed, episodes)


def test_policy_contract_records_bound_episode_and_converter_commit(tmp_path):
    raw, processed, episodes, *_ = write_bound_episode(tmp_path)
    bound = load_bound_episode(raw, processed, episodes)
    payload = build_policy_contract_payload(
        bound_episode=bound,
        checkpoint_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        server_commit="c" * 40,
        prediction_horizon=30,
        execution_horizon=24,
        rtc_delay_steps=6,
        rtc_training_max_delay=7,
    )
    assert payload["source_episode_index"] == 7
    assert payload["processed_episode_index"] == 3
    assert payload["raw_episode_sha256"] == bound.raw_episode_sha256
    assert payload["processed_episode_sha256"] == bound.processed_episode_sha256
    assert payload["converter_commit"] == "2" * 40
```

- [ ] **Step 2: Verify the new module is absent**

Run:

```bash
uv run --group dev pytest -q tests/test_certify_psi0_policy_contract.py
```

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement and test the same-episode loader**

Create immutable `BoundEpisode` with fields `stored_state`, `history_cmd`, `source_episode_index`, `processed_episode_index`, `raw_episode_sha256`, `processed_episode_sha256`, and `converter_commit`. `load_bound_episode(raw, processed, episodes_jsonl)` must perform these exact checks before returning:

```python
processed_index = parse_episode_index(processed.name)
source_index = parse_episode_index(raw.name)
processed_indices = set(pq.read_table(processed, columns=["episode_index"])["episode_index"].to_pylist())
if processed_indices != {processed_index}:
    raise ValueError("processed episode_index does not match filename")
records = [json.loads(line) for line in episodes_jsonl.read_text().splitlines() if line]
matches = [record for record in records if record["episode_index"] == processed_index]
if len(matches) != 1:
    raise ValueError("expected exactly one processed episode metadata record")
record = matches[0]
provenance = record["conversion_provenance"]
if type(provenance["source_episode_index"]) is not int or provenance["source_episode_index"] != source_index:
    raise ValueError("source episode index mismatch")
if provenance["source_parquet_sha256"] != sha256_file(raw):
    raise ValueError("raw episode hash mismatch")
if re.fullmatch(r"[0-9a-f]{40}", provenance["converter_commit"]) is None:
    raise ValueError("invalid converter commit")
```

Read raw `observation.amo_policy_command`, reconstruct history with the recorded `skip` and `downsample`, read processed `states`, require equal frame counts and the recorded processed length, then compute both file hashes. Do not accept standalone `.npy` inputs: they cannot prove same-episode provenance.

- [ ] **Step 4: Implement the classifier and contract writer**

Create a CLI that accepts `--raw-episode-parquet`, `--processed-episode-parquet`, `--processed-episodes-jsonl`, `--checkpoint`, `--dataset-manifest`, `--server-commit`, `--prediction-horizon`, `--execution-horizon`, `--rtc-delay-steps`, `--rtc-training-max-delay`, and `--output`. It calls `load_bound_episode()` and passes only that result to its core comparison:

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

Write JSON atomically with schema `simple.psi0.policy-contract.v2`, SHA-256 hashes of the checkpoint and dataset manifest, the provided server commit, dimensions `32/36`, `50 Hz`, `/act-rtc-v1`, RGB key/order, committed-prefix/executable-suffix semantics, RTC fields, and the certification result. Also write the bound episode's exact `source_episode_index`, `processed_episode_index`, `raw_episode_sha256`, `processed_episode_sha256`, and `converter_commit`. Reject a non-40-character hexadecimal converter or server commit and any RTC tuple that violates the approved inequalities.

Complete and run one focused test after each bounded subaction:

- [ ] Add only `certify_layout()` and its three causal-layout cases.
- [ ] Add only `build_policy_contract(bound_episode, ...)` and assert its exact key/type set.
- [ ] Add SHA/path validation and atomic new-file writing without CLI parsing.
- [ ] Add the argparse entry point and verify all required flags with `--help`.

- [ ] **Step 5: Run certification tests and CLI help**

Run:

```bash
uv run --group dev pytest -q tests/test_certify_psi0_policy_contract.py
uv run python scripts/certify_psi0_policy_contract.py --help
uv run --group dev ruff check scripts/certify_psi0_policy_contract.py tests/test_certify_psi0_policy_contract.py
```

Expected: 6 collected cases pass, help lists every required artifact argument, and Ruff exits zero.

- [ ] **Step 6: Commit the certification tool**

```bash
git add scripts/certify_psi0_policy_contract.py tests/test_certify_psi0_policy_contract.py
git commit -m "feat: certify PSI0 policy contracts"
```

## Task 3: Extend the HTTP client without changing legacy defaults

**Files:**
- Modify: `src/simple/baselines/client.py:65-165`
- Create: `tests/test_http_action_client.py`

- [ ] **Step 1: Write failing timeout and RTC metadata tests**

Create the recording transport in `tests/test_http_action_client.py`; it is the only fixture used by this file:

```python
from copy import deepcopy

import numpy as np
import pytest
import requests

from simple.baselines.client import (
    HttpActionClient,
    RequestMessage,
    convert_numpy_in_dict,
    numpy_serialize,
)


RTC_METADATA = {
    "session_id": "s",
    "request_seq": 0,
    "observation_tick": 100,
    "prediction_horizon": 30,
    "execution_horizon": 24,
    "rtc_delay_steps": 6,
    "first_action_tick": 106,
}


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = "fake response"

    def json(self):
        return deepcopy(self.payload)

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(str(self.status_code))


class RecordingSession:
    def __init__(self):
        self.calls = []
        self.post_response = FakeResponse(
            convert_numpy_in_dict(
                {"action": np.zeros((24, 36), np.float32), "metadata": RTC_METADATA},
                numpy_serialize,
            )
        )
        self.get_response = FakeResponse({"schema": "simple.psi0.policy-contract.v2"})

    def post(self, url, *, json, timeout):
        self.calls.append({"method": "POST", "url": url, "json": json, "timeout": timeout})
        return self.post_response

    def get(self, url, *, timeout):
        self.calls.append({"method": "GET", "url": url, "timeout": timeout})
        return self.get_response


@pytest.fixture
def recording_session():
    return RecordingSession()


def test_legacy_query_keeps_unbounded_default(recording_session):
    client = HttpActionClient("policy", 22085, session=recording_session)
    client.query_action({}, "instruction", {}, {})
    assert recording_session.calls[0]["url"].endswith("/act")
    assert recording_session.calls[0]["timeout"] is None


def test_r0_serialization_is_exact_and_response_metadata_is_preserved(recording_session):
    client = HttpActionClient("policy", 22085, timeout=5.0, session=recording_session)
    image = np.zeros((4, 4, 3), np.uint8)
    state = np.arange(32, dtype=np.float32)[None]
    committed = np.arange(6 * 36, dtype=np.float32).reshape(6, 36)
    response = client.query_rtc_action(
        {"rgb_head_stereo_left": image},
        "pick up the object",
        {"states": state},
        {},
        history={
            "reset": True,
            "session_id": "s",
            "request_seq": 0,
            "observation_tick": 100,
            "rtc_delay_steps": 6,
            "committed_actions": committed,
        },
        dataset="simple",
    )
    call = recording_session.calls[0]
    request = RequestMessage.deserialize(call["json"])
    assert call["url"] == "http://policy:22085/act-rtc-v1"
    assert call["timeout"] == 5.0
    assert set(call["json"]) == {
        "image", "instruction", "history", "state", "condition",
        "gt_action", "dataset_name", "timestamp",
    }
    assert set(request.image) == {"rgb_head_stereo_left"}
    np.testing.assert_array_equal(request.image["rgb_head_stereo_left"], image)
    assert set(request.state) == {"states"}
    np.testing.assert_array_equal(request.state["states"], state)
    assert request.condition == {}
    assert request.gt_action == []
    assert request.dataset_name == "simple"
    assert request.instruction == "pick up the object"
    assert set(request.history) == {
        "reset", "session_id", "request_seq", "observation_tick",
        "rtc_delay_steps", "committed_actions",
    }
    assert request.history["reset"] is True
    np.testing.assert_array_equal(request.history["committed_actions"], committed)
    assert set(response.metadata) == set(RTC_METADATA)
    assert response.metadata == RTC_METADATA
    assert response.action.shape == (24, 36)


def test_successor_omits_reset_and_keeps_complete_history(recording_session):
    client = HttpActionClient("policy", 22085, timeout=5.0, session=recording_session)
    committed = np.full((6, 36), 0.125, np.float32)
    client.query_rtc_action(
        {"rgb_head_stereo_left": np.zeros((2, 2, 3), np.uint8)},
        "continue",
        {"states": np.zeros((1, 32), np.float32)},
        {},
        history={
            "session_id": "s",
            "request_seq": 1,
            "observation_tick": 124,
            "rtc_delay_steps": 6,
            "committed_actions": committed,
        },
        dataset="simple",
    )
    request = RequestMessage.deserialize(recording_session.calls[0]["json"])
    assert "reset" not in request.history
    assert set(request.history) == {
        "session_id", "request_seq", "observation_tick",
        "rtc_delay_steps", "committed_actions",
    }
    np.testing.assert_array_equal(request.history["committed_actions"], committed)


def test_contract_timeout_is_explicit(recording_session):
    client = HttpActionClient("policy", 22085, session=recording_session)
    assert client.get_contract(timeout=2.0) == {
        "schema": "simple.psi0.policy-contract.v2"
    }
    assert recording_session.calls == [{
        "method": "GET",
        "url": "http://policy:22085/contract",
        "timeout": 2.0,
    }]
```

Append these failure cases; `rtc_call()` contains the exact R0 call from the preceding test so it cannot change the payload contract:

```python
def rtc_call(client):
    return client.query_rtc_action(
        {"rgb_head_stereo_left": np.zeros((2, 2, 3), np.uint8)},
        "instruction",
        {"states": np.zeros((1, 32), np.float32)},
        {},
        history={
            "reset": True,
            "session_id": "s",
            "request_seq": 0,
            "observation_tick": 100,
            "rtc_delay_steps": 6,
            "committed_actions": np.zeros((6, 36), np.float32),
        },
        dataset="simple",
    )


@pytest.mark.parametrize("error", [requests.ConnectTimeout(), requests.ReadTimeout()])
def test_transport_timeout_is_propagated(error):
    class RaisingSession:
        def post(self, *args, **kwargs):
            raise error

    with pytest.raises(type(error)):
        rtc_call(HttpActionClient("policy", 22085, timeout=5.0, session=RaisingSession()))


def test_non_200_response_is_rejected(recording_session):
    recording_session.post_response = FakeResponse({}, status_code=500)
    with pytest.raises(requests.HTTPError):
        rtc_call(HttpActionClient("policy", 22085, timeout=5.0, session=recording_session))


def test_missing_metadata_is_not_synthesized(recording_session):
    recording_session.post_response = FakeResponse(
        convert_numpy_in_dict({"action": np.zeros((24, 36), np.float32)}, numpy_serialize)
    )
    with pytest.raises(RuntimeError, match="metadata"):
        rtc_call(HttpActionClient("policy", 22085, timeout=5.0, session=recording_session))


def test_malformed_json_is_rejected(recording_session):
    recording_session.post_response.json = lambda: (_ for _ in ()).throw(ValueError("bad json"))
    with pytest.raises(RuntimeError, match="bad json"):
        rtc_call(HttpActionClient("policy", 22085, timeout=5.0, session=recording_session))
```

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

Refactor the existing request serialization into one private `_post(path, request)` method that always passes `timeout=self.timeout`. Keep `query_action()` on `/act` with the same return tuple. Implement `query_rtc_action()` on `/act-rtc-v1`; require JSON keys `action` and `metadata`, deserialize NumPy payloads, require metadata's exact seven-key set shown in `RTC_METADATA` with `str` for `session_id` and exact `int` (not `bool`) for the other six keys, and return `RtcActionResponse` without adding or coercing missing metadata.

Implement in four bounded subactions, running the matching test after each:

- [ ] Add constructor injection/defaults only.
- [ ] Add `get_contract()` and its timeout/status/JSON validation.
- [ ] Extract `_post()` and rerun the legacy `/act` compatibility test.
- [ ] Add `/act-rtc-v1` serialization plus strict response parsing.

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
import tyro

from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig


def test_sim_config_serializes_resolved_environment_and_domain():
    config = ControlLoopConfig(interface="sim", domain_id=42)
    payload = config.to_dict()
    assert payload["env_type"] == "sim"
    assert payload["interface"] == "lo"
    assert payload["domain_id"] == 42
    assert config.load_wbc_yaml()["DOMAIN_ID"] == 42


def test_tyro_smoke_flags_are_exact():
    help_text = tyro.extras.get_parser(ControlLoopConfig).format_help()
    for flag in (
        "--enable-waist", "--with-hands", "--domain-id",
        "--no-enable-onscreen", "--no-enable-offscreen",
    ):
        assert flag in help_text
```

- [ ] **Step 2: Write the failing named limit audit**

Create `test_g1_effective_limits.py` with the actual model and URDF parser:

```python
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model


SHOULDER_ALLOWLIST = {
    "left_shoulder_roll_joint": (0.19, 2.2515),
    "right_shoulder_roll_joint": (-2.2515, -0.19),
}

RIGHT_HAND_EXPECTED = {
    "right_hand_thumb_0_joint": (-1.04719755, 1.04719755),
    "right_hand_thumb_1_joint": (-1.04719755, 0.72431163),
    "right_hand_thumb_2_joint": (-1.74532925, 0.0),
    "right_hand_index_0_joint": (0.0, 1.57079632),
    "right_hand_index_1_joint": (0.0, 1.74532925),
    "right_hand_middle_0_joint": (0.0, 1.57079632),
    "right_hand_middle_1_joint": (0.0, 1.74532925),
}


def urdf_limits():
    path = Path(__file__).resolve().parents[3] / (
        "control/robot_model/model_data/g1/g1_29dof_with_hand.urdf"
    )
    result = {}
    for joint in ET.parse(path).getroot().findall("joint"):
        limit = joint.find("limit")
        if limit is not None and "lower" in limit.attrib:
            result[joint.attrib["name"]] = (
                float(limit.attrib["lower"]), float(limit.attrib["upper"])
            )
    return result


@pytest.fixture(scope="module")
def effective_limits():
    model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
    upper_names = tuple(model.joint_names[index] for index in model.get_joint_group_indices("upper_body"))
    assert len(upper_names) == 31
    return model, upper_names


def test_effective_limits_equal_urdf_except_reviewed_shoulders(effective_limits):
    model, upper_names = effective_limits
    urdf = urdf_limits()
    for name in upper_names:
        index = model.dof_index(name)
        expected = SHOULDER_ALLOWLIST.get(name, urdf[name])
        np.testing.assert_allclose(
            [model.lower_joint_limits[index], model.upper_joint_limits[index]],
            expected,
            rtol=0,
            atol=1e-7,
            err_msg=name,
        )


def test_right_hand_limits_match_approved_table(effective_limits):
    model, _ = effective_limits
    for name, expected in RIGHT_HAND_EXPECTED.items():
        index = model.dof_index(name)
        np.testing.assert_allclose(
            [model.lower_joint_limits[index], model.upper_joint_limits[index]],
            expected,
            rtol=0,
            atol=1e-7,
            err_msg=name,
        )


def test_every_effective_boundary_accepts_inside_and_rejects_outside(effective_limits):
    model, upper_names = effective_limits
    for name in upper_names:
        index = model.dof_index(name)
        lower = float(model.lower_joint_limits[index])
        upper = float(model.upper_joint_limits[index])
        for value in (lower, (lower + upper) / 2.0, upper):
            assert lower <= value <= upper, name
        assert not lower <= lower - 1e-4 <= upper, name
        assert not lower <= upper + 1e-4 <= upper, name
```

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
RIGHT_HAND_LIMITS = {
"right_hand_thumb_0_joint": [-1.04719755, 1.04719755],
"right_hand_thumb_1_joint": [-1.04719755, 0.72431163],
"right_hand_thumb_2_joint": [-1.74532925, 0],
"right_hand_index_0_joint": [0, 1.57079632],
"right_hand_index_1_joint": [0, 1.74532925],
"right_hand_middle_0_joint": [0, 1.57079632],
"right_hand_middle_1_joint": [0, 1.74532925],
}
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

The wire schema is exact. Add this recursive assertion helper and run it against the real `lower_and_upper_body` model with injected deterministic ONNX identities:

```python
import copy
from pathlib import Path
import re

import pytest

import decoupled_wbc
from decoupled_wbc.control.main.model_contract import (
    GitIdentity, build_model_contract, digest_model_contract,
    validate_expected_model_contract,
)
from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model

MODEL_KEYS = {"schema", "git", "robot_model", "urdf", "onnx_models"}
GIT_KEYS = {"commit", "working_tree_clean"}
ROBOT_KEYS = {
    "name", "joint_names", "lower_position_limits", "upper_position_limits",
    "upper_body_joint_names",
}
FILE_KEYS = {"relative_path", "sha256"}
ONNX_KEYS = {"role", "relative_path", "sha256", "input", "output"}
TENSOR_KEYS = {"name", "shape", "feature_size"}


def fake_onnx_inspector(path):
    return {
        "input": {"name": "observations", "shape": ["dynamic", 516], "feature_size": 516},
        "output": {"name": "actions", "shape": ["dynamic", 15], "feature_size": 15},
    }


def assert_exact_model_contract_types(contract):
    assert set(contract) == MODEL_KEYS
    assert type(contract["schema"]) is str
    assert set(contract["git"]) == GIT_KEYS
    assert type(contract["git"]["commit"]) is str
    assert type(contract["git"]["working_tree_clean"]) is bool
    robot = contract["robot_model"]
    assert set(robot) == ROBOT_KEYS
    assert robot["name"] == "g1_29dof_with_hand"
    for key in ("joint_names", "lower_position_limits", "upper_position_limits", "upper_body_joint_names"):
        assert type(robot[key]) is list
    assert all(type(name) is str for name in robot["joint_names"])
    assert all(type(value) is float for value in robot["lower_position_limits"])
    assert all(type(value) is float for value in robot["upper_position_limits"])
    assert all(type(name) is str for name in robot["upper_body_joint_names"])
    assert set(contract["urdf"]) == FILE_KEYS
    assert contract["urdf"]["relative_path"] == (
        "control/robot_model/model_data/g1/g1_29dof_with_hand.urdf"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", contract["urdf"]["sha256"])
    assert type(contract["onnx_models"]) is list
    assert [entry["role"] for entry in contract["onnx_models"]] == ["balance", "walk"]
    for entry in contract["onnx_models"]:
        assert set(entry) == ONNX_KEYS
        assert type(entry["relative_path"]) is str
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        for direction, feature_size in (("input", 516), ("output", 15)):
            signature = entry[direction]
            assert set(signature) == TENSOR_KEYS
            assert type(signature["name"]) is str
            assert type(signature["shape"]) is list
            assert all(type(dim) in (str, int) for dim in signature["shape"])
            assert type(signature["feature_size"]) is int
            assert signature["feature_size"] == feature_size


def test_model_contract_is_canonical_and_complete():
    model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
    contract = build_model_contract(
        robot_model=model,
        config=ControlLoopConfig(interface="sim", enable_waist=True, domain_id=42),
        repository_root=Path(decoupled_wbc.__file__).resolve().parent,
        git_identity=GitIdentity(commit="a" * 40, working_tree_clean=True),
        onnx_inspector=fake_onnx_inspector,
    )
    assert contract["schema"] == "decoupled_wbc.g1-model-contract.v1"
    assert_exact_model_contract_types(contract)
    assert contract["git"] == {"commit": "a" * 40, "working_tree_clean": True}
    assert len(contract["robot_model"]["joint_names"]) == 43
    assert len(contract["robot_model"]["lower_position_limits"]) == 43
    assert len(contract["robot_model"]["upper_position_limits"]) == 43
    assert len(contract["robot_model"]["upper_body_joint_names"]) == 31
    assert digest_model_contract(contract) == digest_model_contract(dict(contract))
```

Append this executable mutation matrix:

```python
@pytest.fixture
def valid_contract():
    return build_model_contract(
        robot_model=instantiate_g1_robot_model(waist_location="lower_and_upper_body"),
        config=ControlLoopConfig(interface="sim", enable_waist=True, domain_id=42),
        repository_root=Path(decoupled_wbc.__file__).resolve().parent,
        git_identity=GitIdentity(commit="a" * 40, working_tree_clean=True),
        onnx_inspector=fake_onnx_inspector,
    )


@pytest.mark.parametrize(
    "path,value",
    [
        (("robot_model", "lower_position_limits", 35), -9.0),
        (("robot_model", "joint_names", 0), "reordered_joint"),
        (("urdf", "sha256"), "0" * 64),
        (("onnx_models", 0, "sha256"), "0" * 64),
        (("git", "commit"), "0" * 40),
        (("git", "working_tree_clean"), False),
        (("robot_model", "name"), "another_model"),
    ],
)
def test_each_identity_mutation_changes_digest_and_is_rejected(valid_contract, path, value):
    actual = copy.deepcopy(valid_contract)
    target = actual
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if path[0] == "model_contract":
        payload["model_contract_sha256"] = digest_model_contract(
            payload["model_contract"]
        )
    assert digest_model_contract(actual) != digest_model_contract(valid_contract)
    with pytest.raises(ValueError, match="connected WBC model contract mismatch"):
        validate_expected_model_contract(actual, valid_contract)
```

- [ ] **Step 2: Write the failing service-construction order test**

Make construction injectable through `_build_attested_components(config, backend, factories)`. Add this complete test to `test_g1_control_loop_contract.py`:

```python
from types import SimpleNamespace

from decoupled_wbc.control.main.teleop.run_g1_control_loop import (
    _build_attested_components,
)


def test_service_is_created_only_after_model_policy_and_attestation():
    events = []
    robot_model = object()
    policy = object()
    service = object()
    factories = SimpleNamespace(
        robot_model=lambda **kwargs: events.append("robot_model") or robot_model,
        environment=lambda **kwargs: SimpleNamespace(sim=False),
        policy=lambda *args, **kwargs: events.append("wbc_policy") or policy,
        model_contract_payload=lambda config, model: (
            events.append("model_contract")
            or {"model_contract": {"schema": "decoupled_wbc.g1-model-contract.v1"},
                "model_contract_sha256": "d" * 64}
        ),
        service_server=lambda backend, topic, payload: (
            events.append("service_server") or service
        ),
    )
    components = _build_attested_components(
        ControlLoopConfig(interface="sim", enable_waist=True, domain_id=42),
        backend=object(),
        factories=factories,
    )
    assert events == ["robot_model", "wbc_policy", "model_contract", "service_server"]
    assert events.index("robot_model") < events.index("wbc_policy")
    assert events.index("wbc_policy") < events.index("model_contract")
    assert events.index("model_contract") < events.index("service_server")
    assert components.robot_model is robot_model
    assert components.wbc_policy is policy
    assert components.robot_config_server is service
    assert set(components.service_payload) >= {
        "model_contract", "model_contract_sha256"
    }
```

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
7. emit exactly the nested key schema asserted in Step 1: Git identity under `git`; model name/names/limits under `robot_model`; URDF identity under `urdf`; and ONNX identities under `onnx_models`. Normalize dynamic ONNX dimensions to the literal string `"dynamic"` and emit JSON-native strings, lists, integers, floats, and booleans only.

Add `validate_expected_model_contract(actual, expected)` that compares canonical JSON and raises `ValueError("connected WBC model contract mismatch")` on any difference. Add `build_model_contract_payload(config, robot_model)` that returns:

```python
repository_root = Path(__file__).resolve().parents[2]
contract = build_model_contract(robot_model, config, repository_root)
payload = config.to_dict()
payload["model_contract"] = contract
payload["model_contract_sha256"] = digest_model_contract(contract)
return payload
```

Complete this step through the following bounded subactions, running the relevant single test each time:

- [ ] Add canonical JSON, SHA-256, and exact recursive type validation.
- [ ] Add injected/production Git identity inspection.
- [ ] Add ordered model name/joint/limit/upper-body extraction.
- [ ] Add URDF relative-path/hash identity.
- [ ] Add one ONNX tensor-signature inspector, then parameterize it for balance/walk.
- [ ] Add full expected-contract comparison and payload digest assembly.

- [ ] **Step 5: Add the injectable attested-component constructor**

Add immutable `AttestedComponents` and `ProductionFactories`. Implement `_build_attested_components()` with exactly the ordering in the Step 2 test, and return the environment, model, policy, service payload, and service server. Keep each factory call on its own line so a construction exception prevents every later call.

- [ ] **Step 6: Move main-loop service publication behind the tested constructor**

In `run_g1_control_loop.main()` preserve manager creation, call `_build_attested_components()`, unpack it, and retain the following production construction order inside the helper:

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

- [ ] **Step 7: Run all nested contract/limit tests**

Run:

```bash
uv run --group sonic pytest -q \
  third_party/decoupled_wbc/tests/control/main/test_model_contract.py \
  third_party/decoupled_wbc/tests/control/main/teleop/test_g1_control_loop_contract.py \
  third_party/decoupled_wbc/tests/control/robot_model/test_g1_effective_limits.py
git -C third_party/decoupled_wbc diff --check
```

Expected: focused tests pass and the nested diff is whitespace-clean.

- [ ] **Step 8: Commit the complete nested change**

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

- [ ] **Step 9: Verify the production Git inspector against the now-clean nested commit**

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

Create `tests/test_psi0_bridge_mapping.py`; assert every output position against explicit names, not slices:

```python
import numpy as np
import pytest

from simple.deploy.psi0_simple_bridge import (
    build_psi0_observation, goal_to_psi0_action, map_psi0_action_to_goal,
)


EXPECTED_LIMB_NAMES = (
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
EXPECTED_UPPER_NAMES = (
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    *EXPECTED_LIMB_NAMES,
)
TEST_JOINT_NAMES = (
    *(f"leg_joint_{index}" for index in range(12)),
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    *EXPECTED_LIMB_NAMES,
)


def test_state_mapping_is_exactly_named():
    q = np.arange(43, dtype=np.float32)
    q_by_name = dict(zip(TEST_JOINT_NAMES, q, strict=True))
    state = build_psi0_observation(
        q, TEST_JOINT_NAMES, np.array([101, 102, 103, 0.74], np.float32)
    )
    np.testing.assert_array_equal(
        state[0, :28], [q_by_name[name] for name in EXPECTED_LIMB_NAMES]
    )
    np.testing.assert_array_equal(state[0, 28:], [101, 102, 103, 0.74])
```

- [ ] **Step 2: Write failing 36-D to 31-D and inverse mapping tests**

Append this literal named test; `expected_state_names` is promoted to module constant `EXPECTED_LIMB_NAMES` in Step 1:

```python
def test_action_mapping_and_inverse_are_exactly_named():
    action = np.arange(36, dtype=np.float32) / 100.0
    connected_upper_names = (
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "right_shoulder_yaw_joint", "left_hand_index_1_joint",
        *tuple(
            name for name in EXPECTED_LIMB_NAMES
            if name not in {"right_shoulder_yaw_joint", "left_hand_index_1_joint"}
        ),
    )
    assert len(connected_upper_names) == 31
    goal = map_psi0_action_to_goal(action, connected_upper_names, now=12.5)
    by_name = dict(zip(EXPECTED_LIMB_NAMES, action[:28], strict=True))
    by_name.update({
        "waist_roll_joint": action[28],
        "waist_pitch_joint": action[29],
        "waist_yaw_joint": action[30],
    })
    np.testing.assert_array_equal(
        goal.target_upper_body_pose,
        np.asarray([by_name[name] for name in connected_upper_names], np.float32),
    )
    np.testing.assert_array_equal(goal.base_height_command, action[31:32])
    np.testing.assert_array_equal(goal.navigate_cmd, action[32:36])
    assert goal.timestamp == 12.5
    assert goal.target_time == 12.52
    np.testing.assert_array_equal(
        goal_to_psi0_action(goal, connected_upper_names), action
    )


@pytest.mark.parametrize(
    "bad_names",
    [
        EXPECTED_UPPER_NAMES[:-1],
        EXPECTED_UPPER_NAMES[:-1] + (EXPECTED_UPPER_NAMES[0],),
        EXPECTED_UPPER_NAMES[:-1] + ("unknown_joint",),
    ],
)
def test_action_mapping_rejects_missing_duplicate_or_unknown_names(bad_names):
    with pytest.raises(ValueError, match="one-to-one 31-joint upper body"):
        map_psi0_action_to_goal(np.zeros(36, np.float32), bad_names, now=0.0)
```

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

Start `tests/test_psi0_bridge_safety.py` with these complete fixtures and cases:

```python
import numpy as np
import pytest

from simple.deploy.psi0_simple_bridge import (
    Goal, JointContract, PSI0_ACTION_JOINT_NAMES, TimedCameraFrame,
    TimedRobotState, accept_measured_state, apply_slew_limit,
    build_bounded_hold, validate_action_suffix, validate_synchronized_snapshot,
)


@pytest.fixture
def contract():
    names = (
        *(f"leg_joint_{index}" for index in range(12)),
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        *PSI0_ACTION_JOINT_NAMES,
    )
    assert len(names) == len(set(names)) == 43
    return JointContract(
        names, names[12:], np.full(43, -2.0, np.float32),
        np.full(43, 2.0, np.float32),
    )


def state(q=None, received_at=10.0):
    return TimedRobotState(
        q=np.zeros(43, np.float32) if q is None else np.asarray(q, np.float32),
        received_at=received_at,
    )


def camera(received_at=10.0):
    return TimedCameraFrame(
        image=np.zeros((4, 4, 3), np.uint8), received_at=received_at,
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
    ],
)
def test_snapshot_freshness_and_skew_are_independent(
    state_time, camera_time, now, error
):
    with pytest.raises(ValueError, match=error):
        validate_synchronized_snapshot(
            state(received_at=state_time), camera(received_at=camera_time), now
        )
```

`accept_measured_state()` returns `(unchanged_last_valid, stable_reason)` on failure and `(candidate, None)` on success. Task 9 adds the executable ACTIVE-state test that proves this stable reason is latched once.

- [ ] **Step 2: Write failing whole-chunk bound and slew tests**

Append the exact whole-chunk matrix and slew constants:

```python
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
    assert np.max(np.abs(limited[:14] - previous[:14])) <= PER_TICK_LIMITS["hand"] + 1e-7
    assert np.max(np.abs(limited[14:28] - previous[14:28])) <= PER_TICK_LIMITS["arm"] + 1e-7
    assert np.max(np.abs(limited[28:31] - previous[28:31])) <= PER_TICK_LIMITS["waist"] + 1e-7
    assert abs(limited[31] - previous[31]) <= PER_TICK_LIMITS["height"] + 1e-7
    assert np.max(np.abs(limited[32:34] - previous[32:34])) <= PER_TICK_LIMITS["planar_navigation"] + 1e-7
    assert abs(limited[34] - previous[34]) <= PER_TICK_LIMITS["turning"] + 1e-7
    yaw_delta = ((limited[35] - previous[35] + np.pi) % (2 * np.pi)) - np.pi
    assert abs(yaw_delta) <= PER_TICK_LIMITS["target_yaw"] + 1e-7
```

- [ ] **Step 3: Write failing bounded-hold source-order tests**

Append all four source-order branches:

```python
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
```

- [ ] **Step 4: Add the immutable safety data types**

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

Run only the import/constructor cases after adding the dataclasses; do not implement validation in this checkbox.

- [ ] **Step 5: Implement measured-state and synchronized-snapshot validation**

Implement these pure functions and run only the state/freshness cases from Step 1:

```python
validate_measured_state(sample, contract, now, tolerance=0.05)
accept_measured_state(last_valid, candidate, contract, now)
validate_synchronized_snapshot(state, camera, now)
```

- [ ] **Step 6: Implement whole-suffix validation and one-tick slew limiting**

Implement and run the Step 2 cases:

```python
validate_action_suffix(actions, expected_s, contract)
apply_slew_limit(previous_action, requested_action, contract, dt=0.02)
```

- [ ] **Step 7: Implement bounded hold selection and immutable capture**

Implement and run the Step 3 cases:

```python
build_bounded_hold(now, last_valid_state, last_safe_goal, contract)
```

`HoldResult.source` is exactly `"measured_clamped"` or `"last_safe_published"`; it includes a tuple of `(joint_name, unclamped, clamped)` records. The state tolerance never changes command limits.

- [ ] **Step 8: Run safety tests and commit**

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
- Create: `tests/psi0_bridge_testkit.py`
- Create: `tests/test_psi0_bridge_scheduler.py`
- Create: `tests/test_psi0_bridge_lifecycle.py`

- [ ] **Step 1: Write the time-indexed P=8/s=5/d=3 sentinel test**

Create `tests/psi0_bridge_testkit.py` first. These definitions are normative fixtures used by Tasks 9-11:

```python
from copy import deepcopy

import numpy as np

from simple.deploy.psi0_simple_bridge import (
    JointContract, PolicyContract, PSI0_ACTION_JOINT_NAMES, RtcResult,
    TimedCameraFrame, TimedRobotState,
)


class ManualClock:
    def __init__(self, tick=0):
        self.tick = tick

    def __call__(self):
        return self.tick / 50.0

    def set_tick(self, tick):
        self.tick = tick


def policy_payload(**updates):
    payload = {
        "schema": "simple.psi0.policy-contract.v2",
        "test_only": True,
        "checkpoint_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "raw_episode_sha256": "c" * 64,
        "processed_episode_sha256": "d" * 64,
        "source_episode_index": 7,
        "processed_episode_index": 3,
        "converter_commit": "e" * 40,
        "server_commit": "f" * 40,
        "converter_layout": "g1_simple_32_rpyh_v2",
        "observation_dim": 32,
        "action_dim": 36,
        "action_frequency_hz": 50,
        "prediction_horizon": 8,
        "execution_horizon": 5,
        "rtc_delay_steps": 3,
        "rtc_training_max_delay": 4,
        "rtc_enabled": True,
        "rtc_endpoint": "/act-rtc-v1",
        "request_semantics": "exact-post-slew-committed-prefix",
        "response_semantics": "denormalized-executable-suffix",
        "image_key": "rgb_head_stereo_left",
        "camera_color_order": "rgb",
    }
    payload.update(updates)
    return payload


def test_policy_contract(**updates):
    return PolicyContract.from_dict(policy_payload(**updates))


def test_joint_contract():
    names = (
        *(f"leg_joint_{index}" for index in range(12)),
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        *PSI0_ACTION_JOINT_NAMES,
    )
    return JointContract(
        names, names[12:], np.full(43, -2.0, np.float32),
        np.full(43, 2.0, np.float32),
    )


def fresh_inputs(clock):
    return (
        TimedRobotState(np.zeros(43, np.float32), clock()),
        TimedCameraFrame(np.zeros((8, 8, 3), np.uint8), clock(), None),
    )


def sentinel_actions(first_tick, count):
    actions = np.empty((count, 36), np.float32)
    for row, global_tick in enumerate(range(first_tick, first_tick + count)):
        actions[row] = 0.30 + 0.001 * (global_tick - 100) + 0.000001 * np.arange(36)
    actions[:, 31] = 0.50
    actions[:, 32:36] = 0.0
    return actions


class ImmediateInference:
    def __init__(self, clock, contract):
        self.clock = clock
        self.contract = contract
        self.requests = []
        self.pending = []
        self.submissions = 0

    @property
    def busy(self):
        return False

    def submit(self, request):
        self.submissions += 1
        self.requests.append(deepcopy(request))
        first = request.observation_tick + self.contract.rtc_delay_steps
        metadata = {
            "session_id": request.session_id,
            "request_seq": request.request_seq,
            "observation_tick": request.observation_tick,
            "prediction_horizon": self.contract.prediction_horizon,
            "execution_horizon": self.contract.execution_horizon,
            "rtc_delay_steps": self.contract.rtc_delay_steps,
            "first_action_tick": first,
        }
        self.pending.append(RtcResult(
            generation=request.generation,
            request_seq=request.request_seq,
            completed_at=self.clock(),
            actions=sentinel_actions(first, self.contract.execution_horizon),
            metadata=metadata,
        ))

    def poll(self):
        return self.pending.pop(0) if self.pending else None
```

Then create `tests/test_psi0_bridge_scheduler.py`. `Psi0SimpleBridge.tick()` returns immutable `TickResult(tick, goal, psi0_action, source_tick, source_kind)`; this makes tick provenance observable without reaching into private fields:

```python
def test_rtc_sentinel_has_exact_request_and_handoff_ticks():
    clock = ManualClock(99)
    contract = test_policy_contract()
    inference = ImmediateInference(clock, contract)
    bridge = Psi0SimpleBridge(contract, test_joint_contract(), inference, clock, start_tick=99)
    robot_state, frame = fresh_inputs(clock)
    bridge.update_inputs(robot_state, frame)
    paused = bridge.tick()
    assert paused.source_kind == "hold"
    clock.set_tick(100)
    robot_state, frame = fresh_inputs(clock)
    bridge.update_inputs(robot_state, frame)
    bridge.activate()
    results = {}
    for tick in range(100, 113):
        clock.set_tick(tick)
        robot_state, frame = fresh_inputs(clock)
        bridge.update_inputs(robot_state, frame)
        results[tick] = bridge.tick()

    requests = inference.requests
    assert requests[0].observation_tick == 100
    assert requests[0].first_action_tick == 103
    assert [results[tick].source_kind for tick in range(100, 103)] == ["hold"] * 3
    assert [results[tick].source_tick for tick in range(103, 108)] == list(range(103, 108))
    assert requests[1].observation_tick == 105
    assert requests[1].history_tick == 104
    assert requests[1].committed_global_ticks == (105, 106, 107)
    assert [results[tick].source_tick for tick in range(108, 113)] == list(range(108, 113))
    committed = np.stack([results[tick].psi0_action for tick in range(105, 108)])
    np.testing.assert_array_equal(requests[1].committed_actions, committed)
    assert bridge.command_history_tick == 112
    assert [result.tick for result in results.values()] == list(range(100, 113))
```

- [ ] **Step 2: Write failing protocol/deadline tests**

Append strict contract-type tests first:

```python
from dataclasses import replace


@pytest.fixture
def clock():
    return ManualClock(100)


@pytest.mark.parametrize(
    "updates,error",
    [
        ({"rtc_delay_steps": 1}, "RTC horizon"),
        ({"rtc_delay_steps": 6}, "RTC horizon"),
        ({"prediction_horizon": 7}, "RTC horizon"),
        ({"rtc_training_max_delay": 3}, "RTC horizon"),
        ({"rtc_endpoint": "/act"}, "legacy policy endpoint"),
    ],
)
def test_policy_contract_rejects_invalid_rtc_tuple_or_endpoint(updates, error):
    with pytest.raises(ValueError, match=error):
        PolicyContract.from_dict(policy_payload(**updates))


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("schema", 123),
        ("observation_dim", "32"),
        ("prediction_horizon", 8.0),
        ("rtc_delay_steps", True),
        ("rtc_enabled", 1),
        ("test_only", "false"),
        ("source_episode_index", "7"),
    ],
)
def test_policy_contract_never_coerces_malformed_json_types(field, bad_value):
    with pytest.raises(TypeError, match=field):
        PolicyContract.from_dict(policy_payload(**{field: bad_value}))


def test_policy_contract_rejects_missing_and_extra_keys():
    missing = policy_payload()
    missing.pop("converter_commit")
    with pytest.raises(TypeError, match="keys"):
        PolicyContract.from_dict(missing)
    with pytest.raises(TypeError, match="keys"):
        PolicyContract.from_dict(policy_payload(extra="not allowed"))
```

Then test the pure response validator. It must require exactly `(s,36)`, all seven exact metadata keys/types, whole-suffix bounds, and completion no later than the monotonic deadline:

```python
def valid_request_and_result(clock, contract):
    request = RtcRequest(
        generation=1,
        session_id="session",
        request_seq=2,
        observation_tick=100,
        history_tick=99,
        observation=np.zeros((1, 32), np.float32),
        image=np.zeros((4, 4, 3), np.uint8),
        committed_actions=sentinel_actions(100, 3),
        reset=False,
    )
    result = RtcResult(
        generation=1,
        request_seq=2,
        completed_at=100 / 50.0 + 3 / 50.0,
        actions=sentinel_actions(103, 5),
        metadata={
            "session_id": "session", "request_seq": 2,
            "observation_tick": 100, "prediction_horizon": 8,
            "execution_horizon": 5, "rtc_delay_steps": 3,
            "first_action_tick": 103,
        },
    )
    return request, result


@pytest.mark.parametrize("rows", [4, 6])
def test_response_length_is_exact(clock, rows):
    contract = test_policy_contract()
    request, result = valid_request_and_result(clock, contract)
    result = replace(result, actions=sentinel_actions(103, rows))
    with pytest.raises(ValueError, match="shape"):
        validate_rtc_result(request, result, contract, test_joint_contract())


@pytest.mark.parametrize(
    "key,bad_value",
    [
        ("session_id", "other"), ("request_seq", 3),
        ("observation_tick", 99), ("prediction_horizon", 9),
        ("execution_horizon", 4), ("rtc_delay_steps", 2),
        ("first_action_tick", 102),
    ],
)
def test_every_response_metadata_mismatch_is_rejected(clock, key, bad_value):
    contract = test_policy_contract()
    request, result = valid_request_and_result(clock, contract)
    metadata = dict(result.metadata)
    metadata[key] = bad_value
    with pytest.raises(ValueError, match=key):
        validate_rtc_result(
            request, replace(result, metadata=metadata), contract,
            test_joint_contract(),
        )


def test_missing_extra_and_wrong_type_metadata_are_rejected(clock):
    contract = test_policy_contract()
    request, result = valid_request_and_result(clock, contract)
    for metadata in (
        {key: value for key, value in result.metadata.items() if key != "session_id"},
        {**result.metadata, "extra": 1},
        {**result.metadata, "request_seq": "2"},
        {**result.metadata, "request_seq": True},
    ):
        with pytest.raises((TypeError, ValueError), match="metadata|request_seq|session_id"):
            validate_rtc_result(
                request, replace(result, metadata=metadata), contract,
                test_joint_contract(),
            )


def test_nonfinite_out_of_bounds_and_late_results_are_rejected(clock):
    contract = test_policy_contract()
    request, result = valid_request_and_result(clock, contract)
    for actions in (
        np.where(np.indices(result.actions.shape)[1] == 0, np.nan, result.actions),
        np.where(np.indices(result.actions.shape)[1] == 31, 0.1, result.actions),
    ):
        with pytest.raises(ValueError):
            validate_rtc_result(
                request, replace(result, actions=actions), contract,
                test_joint_contract(),
            )
    with pytest.raises(ValueError, match="deadline"):
        validate_rtc_result(
            request, replace(result, completed_at=result.completed_at + 1e-6),
            contract, test_joint_contract(),
        )
```

Add and run each test function separately before moving to lifecycle behavior.

- [ ] **Step 3: Write failing lifecycle and blocked-worker tests**

Create `tests/test_psi0_bridge_lifecycle.py` with an explicit blocking port and atomic-state assertions:

```python
class BlockingInference:
    def __init__(self):
        self.requests = []
        self.result = None
        self.physical_busy = False

    @property
    def busy(self):
        return self.physical_busy

    def submit(self, request):
        if self.physical_busy:
            raise RuntimeError("physical worker busy")
        self.requests.append(deepcopy(request))
        self.physical_busy = True

    def poll(self):
        result, self.result = self.result, None
        return result

    def release(self, result):
        self.result = result
        self.physical_busy = False


def ready_bridge(inference):
    clock = ManualClock(99)
    bridge = Psi0SimpleBridge(
        test_policy_contract(), test_joint_contract(), inference, clock,
        start_tick=99,
    )
    bridge.update_inputs(*fresh_inputs(clock))
    return bridge, clock


def assert_policy_buffers_empty(status):
    assert status.scheduled_ticks == ()
    assert status.staged_first_tick is None
    assert status.staged_count == 0
    assert status.committed_ticks == ()
    assert status.deadline is None
    assert status.logical_request_active is False


def test_startup_requires_one_consumed_paused_hold():
    inference = BlockingInference()
    bridge, _ = ready_bridge(inference)
    assert bridge.state is BridgeState.PAUSED
    with pytest.raises(RuntimeError, match="paused hold"):
        bridge.handle_toggle()
    assert inference.requests == []
    hold = bridge.tick()
    assert hold.source_kind == "hold"
    bridge.handle_toggle()
    assert bridge.state is BridgeState.ACTIVE
    assert len(inference.requests) == 1


@pytest.mark.parametrize("transition", ["pause", "enter_fault", "stop"])
def test_exit_transition_atomically_invalidates_every_policy_buffer(transition):
    inference = BlockingInference()
    bridge, clock = ready_bridge(inference)
    bridge.tick()
    clock.set_tick(100)
    bridge.update_inputs(*fresh_inputs(clock))
    bridge.activate()
    before = bridge.status()
    getattr(bridge, transition)("test fault") if transition == "enter_fault" else getattr(bridge, transition)()
    after = bridge.status()
    assert after.generation == before.generation + 1
    assert_policy_buffers_empty(after)


def test_fault_holds_zero_navigation_and_rearm_waits_for_physical_idle():
    inference = BlockingInference()
    bridge, clock = ready_bridge(inference)
    bridge.tick()
    clock.set_tick(100)
    bridge.update_inputs(*fresh_inputs(clock))
    bridge.activate()
    old_request = inference.requests[0]
    bridge.enter_fault("deadline")
    fault_tick = bridge.tick()
    assert bridge.state is BridgeState.FAULT
    np.testing.assert_array_equal(fault_tick.goal.navigate_cmd, np.zeros(4, np.float32))
    with pytest.raises(RuntimeError, match="worker busy"):
        bridge.handle_toggle()
    late = RtcResult(
        generation=old_request.generation,
        request_seq=old_request.request_seq,
        completed_at=clock(),
        actions=sentinel_actions(103, 5),
        metadata={
            "session_id": old_request.session_id, "request_seq": 0,
            "observation_tick": 100, "prediction_horizon": 8,
            "execution_horizon": 5, "rtc_delay_steps": 3,
            "first_action_tick": 103,
        },
    )
    inference.release(late)
    bridge.update_inputs(*fresh_inputs(clock))
    bridge.tick()
    assert bridge.metrics.discarded_old_generation_results == 1
    bridge.handle_toggle()
    assert bridge.state is BridgeState.ACTIVE
    assert len(inference.requests) == 2


def test_active_invalid_state_latches_one_stable_fault_reason():
    inference = BlockingInference()
    bridge, clock = ready_bridge(inference)
    bridge.tick()
    clock.set_tick(100)
    bridge.update_inputs(*fresh_inputs(clock))
    bridge.activate()
    bad = TimedRobotState(np.r_[np.nan, np.zeros(42)], clock())
    bridge.update_inputs(bad, fresh_inputs(clock)[1])
    first = bridge.tick()
    second = bridge.tick()
    assert bridge.state is BridgeState.FAULT
    assert bridge.fault_reason == "measured state finite"
    assert bridge.metrics.fault_transitions == 1
    np.testing.assert_array_equal(first.goal.navigate_cmd, np.zeros(4, np.float32))
    np.testing.assert_array_equal(second.goal.navigate_cmd, np.zeros(4, np.float32))
```

The core exposes immutable `status()` and counters only for deterministic assertions; mutating buffers remains private. Runtime Task 11 separately proves one physical worker object for the process lifetime.

- [ ] **Step 4: Run tests and observe missing scheduler types**

Run:

```bash
uv run --group dev pytest -q \
  tests/test_psi0_bridge_scheduler.py \
  tests/test_psi0_bridge_lifecycle.py
```

Expected: failures identify absent `PolicyContract`, `RtcRequest`, `RtcResult`, `InferencePort`, and `Psi0SimpleBridge` behavior.

- [ ] **Step 5: Define strict policy-contract parsing and validation**

Add `import re` with the other core imports, then add:

```python
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
                raise TypeError(f"policy contract field {key} must be {expected_type.__name__}")
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


```

Run only `test_policy_contract_*` after this checkbox.

- [ ] **Step 6: Define RTC request/result interfaces and the pure response validator**

Add:

```python
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

These protocol bodies define the injected boundary only; no production I/O belongs in the core file. Implement `validate_rtc_result()` against the exact Step 2 schema and run only the protocol/deadline cases.

- [ ] **Step 7: Add scheduler state, immutable status, and atomic exit transitions**

`Psi0SimpleBridge` owns these fields under one lock: `state`, `generation`, `tick_index`, `session_id`, `request_seq`, `request_tick`, `handoff_tick`, `next_request_tick`, `scheduled_actions: dict[int,np.ndarray]`, `staged_first_tick`, `staged_actions`, `command_history_rpyh`, `last_valid_state`, `last_safe_goal`, `fault_reason`, and logical-request-active. Initialize `command_history_rpyh` exactly as `np.array([0.0, 0.0, 0.0, 0.74], dtype=np.float32)`. The injected `InferencePort.busy` remains separate.

Implement immutable `BridgeStatus`/metrics snapshots plus `enter_fault(reason)`, `pause()`, and `stop()` as single locked transitions that increment generation and clear all logical action/request state. Never clear `InferencePort.busy` from the core. Run `test_exit_transition_*` before proceeding.

- [ ] **Step 8: Implement paused holds, R0 activation, and physical-busy rearm**

Implement `activate()` to require PAUSED, fresh synchronized inputs, idle worker, and one successfully published bounded hold. At `r0`, schedule exactly `d` copies of the inverse 36-D hold, submit reset R0, and set handoff `r0+d` and next request `r0+s`.

Implement `handle_toggle()` for PAUSED/ACTIVE/FAULT and run the startup/rearm tests. Do not implement successor scheduling in this checkbox.

- [ ] **Step 9: Implement tick order, successor scheduling, and history ownership**

Implement `tick()` in the approved order:

1. validate latest inputs and drain at most one completed result;
2. reject stale generation, bad metadata, bad suffix, or completion after the monotonic deadline corresponding to `r+d`;
3. at a response handoff, install exactly `s` suffix actions starting at `r+d`;
4. at `next_request_tick`, precompute and freeze final post-slew commands for ticks `r:r+d`, submit them with observation/history from `r-1`, then increment next request tick by `s`;
5. select the scheduled action for the current tick, or atomically enter FAULT if the required suffix is absent;
6. validate/map/slew/publish or shadow-preview the command;
7. only after successful publication/preview consumption, update `last_safe_goal` and `[roll,pitch,yaw,height]` history.

Return immutable `TickResult` after each consumed command and advance history only there. Run the P=8/s=5/d=3 sentinel, then the remaining lifecycle tests.

Build `tick()` through these independently tested subactions:

- [ ] Drain/discard one result by generation without installing actions.
- [ ] Validate/stage one on-time suffix without changing committed ticks.
- [ ] Install a staged suffix only at its exact handoff tick.
- [ ] Freeze the next `d` post-slew actions and submit one successor at `r0+s`.
- [ ] Select current action or enter an atomic underrun fault.
- [ ] Produce `TickResult` and advance command history only after consumption.

- [ ] **Step 10: Run scheduler/lifecycle tests and commit**

Run:

```bash
uv run --group dev pytest -q \
  tests/test_psi0_bridge_mapping.py \
  tests/test_psi0_bridge_safety.py \
  tests/test_psi0_bridge_scheduler.py \
  tests/test_psi0_bridge_lifecycle.py
uv run --group dev ruff check src/simple/deploy tests/test_psi0_bridge_*.py
git add src/simple/deploy/psi0_simple_bridge.py \
  tests/psi0_bridge_testkit.py tests/test_psi0_bridge_scheduler.py \
  tests/test_psi0_bridge_lifecycle.py
git commit -m "feat: add fail-closed PSI0 RTC scheduler"
```

Expected: all pure-core tests pass and no external transport is initialized.

## Task 10: Implement policy/WBC preflight and camera/state adapters

**Files:**
- Create: `scripts/psi0_simple_real_bridge.py`
- Create: `tests/test_psi0_bridge_preflight.py`
- Create: `tests/test_psi0_bridge_camera.py`

- [ ] **Step 1: Write failing policy contract comparison tests**

Start `tests/test_psi0_bridge_preflight.py` with an exhaustive field mutation matrix:

```python
import copy

import pytest

from decoupled_wbc.control.main.model_contract import digest_model_contract
from scripts.psi0_simple_real_bridge import (
    BoundedWbcConfigClient, PreflightError,
    compare_policy_contracts, establish_goal_ownership, run_preflight,
    validate_connected_wbc, validate_then_create_publisher,
)
from simple.deploy.psi0_simple_bridge import BridgeMode, PolicyContract
from tests.psi0_bridge_testkit import ManualClock, policy_payload, test_joint_contract


POLICY_WIRE_FIELDS = tuple(policy_payload())


def test_matching_test_contract_is_accepted_only_for_sim_wbc():
    local = PolicyContract.from_dict(policy_payload())
    server = PolicyContract.from_dict(policy_payload())
    result = compare_policy_contracts(local, server, BridgeMode.SIM_CONTROL, "sim")
    assert result.policy_certified is True
    assert result.mismatched_fields == ()
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
    calls = []
    result = run_preflight(
        mode=BridgeMode.SHADOW,
        local_policy=policy_payload(),
        server_policy={"malformed": True},
        wbc_payload=valid_wbc_payload(),
        graph=FakeGraph(publishers=0, subscriptions=1),
        publisher_factory=lambda: calls.append("publisher"),
    )
    assert result.policy_certified is False
    assert calls == []
```

No fixture is implicit: `valid_wbc_payload()` and `FakeGraph` are defined literally in Steps 2-3 of this same file.

- [ ] **Step 2: Write failing WBC payload/digest tests**

Append a fully typed wire fixture:

```python
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
    joints = list(test_joint_contract().joint_names)
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
            "relative_path": "control/robot_model/model_data/g1/g1_29dof_with_hand.urdf",
            "sha256": "2" * 64,
        },
        "onnx_models": [
            {
                "role": role,
                "relative_path": f"sim2mujoco/resources/robots/g1/policy/{filename}",
                "sha256": digest,
                "input": {"name": "observations", "shape": ["dynamic", 516], "feature_size": 516},
                "output": {"name": "actions", "shape": ["dynamic", 15], "feature_size": 15},
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


@pytest.mark.parametrize(
    "path,value,error",
    [
        (("env_type",), "real", "env_type"),
        (("domain_id",), 0, "domain_id"),
        (("model_contract_sha256",), "0" * 64, "digest"),
        (("model_contract", "git", "commit"), "0" * 40, "git"),
        (("model_contract", "git", "working_tree_clean"), False, "clean"),
        (("model_contract", "robot_model", "name"), "wrong", "robot_model"),
        (("model_contract", "robot_model", "joint_names", 0), "wrong", "joint_names"),
        (("model_contract", "robot_model", "upper_body_joint_names", 0), "wrong", "upper_body"),
        (("model_contract", "robot_model", "lower_position_limits", 12), -9.0, "limits"),
        (("model_contract", "urdf", "sha256"), "0" * 64, "urdf"),
        (("model_contract", "onnx_models", 0, "sha256"), "0" * 64, "onnx"),
        (("model_contract", "onnx_models", 1, "input", "feature_size"), 515, "onnx"),
    ],
)
def test_wbc_mutation_fails_before_publisher(path, value, error):
    payload = copy.deepcopy(valid_wbc_payload())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    calls = []
    with pytest.raises(PreflightError, match=error):
        validate_then_create_publisher(
            payload, valid_model_contract(), "1" * 40, 42,
            publisher_factory=lambda: calls.append("publisher"),
        )
    assert calls == []


def test_wbc_service_timeout_is_three_seconds_and_destroys_resources():
    clock = ManualClock()
    events = []
    adapter = BoundedWbcConfigClient(
        clock=clock,
        wait_once=lambda timeout: clock.set_tick(clock.tick + int(timeout * 50)) or False,
        destroy=lambda: events.append("destroy"),
    )
    with pytest.raises(TimeoutError, match="3.0"):
        adapter.get_config(timeout_s=3.0)
    assert clock() == pytest.approx(3.0)
    assert events == ["destroy"]
```

The production validator recomputes the digest before comparing expected content. For any nested mutation, either the transmitted digest is stale and fails digest validation, or a recomputed malicious digest still fails full expected-contract equality.

- [ ] **Step 3: Write failing graph-ownership and camera codec tests**

Append exact graph ownership tests:

```python
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
            BridgeMode.SIM_CONTROL, FakeGraph(*counts),
            lambda: calls.append("publisher"),
        )
    assert calls == []


def test_shadow_never_constructs_a_publisher_or_changes_counts():
    graph = FakeGraph(2, 1)
    calls = []
    assert establish_goal_ownership(
        BridgeMode.SHADOW, graph, lambda: calls.append("publisher")
    ) is None
    assert graph.counts("ControlPolicy/upper_body_pose") == (2, 1)
    assert calls == []
```

Create `tests/test_psi0_bridge_camera.py` with the real codec and explicit RGB default:

```python
import numpy as np
import pytest

from decoupled_wbc.control.sensor.sensor_server import ImageMessageSchema
from scripts.psi0_simple_real_bridge import build_parser, decode_camera_message
from tests.psi0_bridge_testkit import ManualClock


def encoded_sentinel():
    image = np.zeros((32, 64, 3), np.uint8)
    image[:, :24] = [240, 0, 0]
    image[:, 40:] = [0, 0, 240]
    return ImageMessageSchema(
        timestamps={"rgb_head_stereo_left": 4.5},
        images={"rgb_head_stereo_left": image},
    ).serialize()


def test_parser_default_camera_color_order_is_rgb():
    parser = build_parser()
    assert parser.get_default("camera_color_order") == "rgb"


@pytest.mark.parametrize("color_order", ["rgb", "bgr"])
def test_codec_applies_configured_channel_transform_exactly_once(color_order):
    clock = ManualClock(500)
    frame = decode_camera_message(
        encoded_sentinel(), key="rgb_head_stereo_left",
        color_order=color_order, received_at=clock(),
    )
    assert frame.image.dtype == np.uint8
    assert frame.image.flags.c_contiguous
    assert frame.image.shape == (32, 64, 3)
    assert frame.received_at == 10.0
    assert frame.producer_timestamp == 4.5
    left = frame.image[:, :20].mean(axis=(0, 1))
    right = frame.image[:, 44:].mean(axis=(0, 1))
    if color_order == "rgb":
        assert left[0] > 200 and left[2] < 30
        assert right[2] > 200 and right[0] < 30
    else:
        assert left[2] > 200 and left[0] < 30
        assert right[0] > 200 and right[2] < 30


def test_camera_key_is_mandatory():
    with pytest.raises(KeyError, match="rgb_head_stereo_left"):
        decode_camera_message(
            encoded_sentinel(), key="missing", color_order="rgb", received_at=1.0
        )
```

This test establishes numeric channel order after the actual JPEG codec; viewpoint validation remains a saved-sample/manual gate because a codec cannot prove physical viewpoint.

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
--camera-color-order {rgb,bgr}  # argparse choices=("rgb","bgr"), default="rgb"
--ros-domain-id 42
--unitree-domain-id 42
--metrics-jsonl PATH
```

Validate `BridgeMode(args.mode)` and ensure no parser branch accepts `real`, `real-control`, a robot NIC, or a low-level DDS topic.

- [ ] **Step 6: Implement strict local/server policy comparison**

Implement `compare_policy_contracts()` and run only the Step 1 mutation matrix. Keep shadow's mismatches sorted in policy wire-field order; never use arbitrary set order.

- [ ] **Step 7: Implement the bounded WBC config client and connected-model validator**

Implement `BoundedWbcConfigClient`, local expected-contract construction, root-gitlink lookup, digest recomputation, and exact full comparison. `get_config(timeout_s=3.0)` owns a temporary rclpy node/client, loops `wait_for_service()` in at most 50 ms intervals until the monotonic deadline, waits for the async result only within the remaining budget, and destroys client/node in `finally`. Read the expected nested SHA with:

```python
subprocess.run(
    ["git", "rev-parse", "HEAD:third_party/decoupled_wbc"],
    cwd=repository_root,
    check=True,
    capture_output=True,
    text=True,
)
```

Run the Step 2 tests before adding publishers or camera code.

Complete in this order:

- [ ] Implement the 50 ms polling/three-second deadline and unconditional destruction.
- [ ] Parse and type-check the exact connected WBC payload/digest.
- [ ] Resolve and validate the root gitlink plus clean local nested identity.
- [ ] Build the local expected model contract and compare full canonical JSON.

- [ ] **Step 8: Implement graph ownership and the goal publisher adapter**

Implement `establish_goal_ownership()`, `RosGoalPublisher`, and only the five approved `Goal` fields. Run the graph tests and prove rejected preflight never invokes the publisher factory.

- [ ] **Step 9: Implement state and camera adapters with bounded close**

Create concrete `RosStateSource` and `ComposedCameraReader` classes in the script. Compose all adapters with:

```python
@dataclass(frozen=True)
class RuntimeAdapters:
    wbc_config_client: BoundedWbcConfigClient
    state_source: RosStateSource
    camera_reader: ComposedCameraReader
    goal_publisher: RosGoalPublisher | None
```

`RosStateSource.poll()` returns `TimedRobotState | None`; every adapter exposes `close()`. The camera socket is created, polled with 100 ms timeout, decoded, and closed only inside its reader thread; `ComposedCameraReader.close(timeout_s=0.5)` signals and joins that thread. Run the camera tests after this checkbox.

Complete in four bounded subactions:

- [ ] Add state polling only.
- [ ] Add pure codec/color conversion only.
- [ ] Add camera reader-thread polling only.
- [ ] Add reader-thread socket close/join only.

- [ ] **Step 10: Implement preflight sequencing before publisher construction**

`run_bridge()` must perform, in order: parse/validate mode; WBC bounded query; connected/local model digest comparison; local/server policy contract comparison; ROS graph ownership check; then publisher construction only for `sim-control`. Shadow never calls the publisher factory. Record every result in metrics.

- [ ] **Step 11: Run adapter/preflight tests and commit**

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

Create `tests/test_psi0_bridge_runtime.py` with these literal ownership cases:

```python
import os
import pty
import termios
import threading

import numpy as np
import pytest

from scripts.psi0_simple_real_bridge import FiftyHzLoop, HttpInferenceWorker, LocalKeyboard
from simple.baselines.client import RtcActionResponse
from simple.deploy.psi0_simple_bridge import RtcRequest
from tests.psi0_bridge_testkit import ManualClock


def valid_runtime_request(generation=1):
    return RtcRequest(
        generation=generation,
        session_id="runtime-session",
        request_seq=0,
        observation_tick=100,
        history_tick=99,
        observation=np.zeros((1, 32), np.float32),
        image=np.zeros((8, 8, 3), np.uint8),
        committed_actions=np.zeros((6, 36), np.float32),
        reset=True,
    )


def valid_http_rtc_response():
    return RtcActionResponse(
        action=np.zeros((24, 36), np.float32),
        metadata={
            "session_id": "runtime-session", "request_seq": 0,
            "observation_tick": 100, "prediction_horizon": 30,
            "execution_horizon": 24, "rtc_delay_steps": 6,
            "first_action_tick": 106,
        },
    )


def test_local_keyboard_accepts_only_p_and_restores_terminal():
    master, slave = pty.openpty()
    original = termios.tcgetattr(slave)
    try:
        with LocalKeyboard(slave) as keyboard:
            os.write(master, b"xP]ppq")
            assert keyboard.poll(timeout_s=0.1) == ("p", "p")
        assert termios.tcgetattr(slave) == original
    finally:
        os.close(master)
        os.close(slave)


class BlockingHttpClient:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def query_rtc_action(self, *args, **kwargs):
        self.calls += 1
        self.entered.set()
        assert self.release.wait(2.0)
        return valid_http_rtc_response()


def test_one_blocked_worker_never_blocks_tick_thread_or_allows_replacement():
    client = BlockingHttpClient()
    worker = HttpInferenceWorker(client=client, clock=ManualClock(100))
    thread_identity = worker.thread.ident
    worker.submit(valid_runtime_request())
    assert client.entered.wait(0.2)
    tick_times = []
    loop = FiftyHzLoop(clock=ManualClock(100), sleep=lambda _: None)
    loop.run_n(50, lambda scheduled: tick_times.append(scheduled))
    assert len(tick_times) == 50
    np.testing.assert_allclose(np.diff(tick_times), 0.02, rtol=0, atol=1e-12)
    assert worker.busy is True
    with pytest.raises(RuntimeError, match="busy"):
        worker.submit(valid_runtime_request(generation=2))
    assert worker.thread.ident == thread_identity
    assert client.calls == 1
    client.release.set()
    worker.close(timeout_s=0.5)
    assert worker.busy is False
    assert worker.thread.ident == thread_identity
```

No pytest fixture is implicit in these tests.

- [ ] **Step 2: Write failing no-state Ctrl-C subprocess test**

Create `tests/test_psi0_bridge_shutdown.py`. The subprocess helper owns only files and ports under `tmp_path`:

```python
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import numpy as np


def wait_for(path, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def assert_port_rebinds(port):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def run_shutdown_scenario(tmp_path, scenario):
    report_path = tmp_path / f"{scenario}-report.json"
    ready_path = tmp_path / f"{scenario}-ready"
    command = [
        sys.executable,
        "scripts/tests/bridge_subprocess_fixture.py",
        "--scenario", scenario,
        "--report", str(report_path),
        "--ready", str(ready_path),
    ]
    process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1])
    wait_for(ready_path)
    started = time.monotonic()
    os.kill(process.pid, signal.SIGINT)
    process.wait(timeout=7.0)
    elapsed = time.monotonic() - started
    assert process.returncode == 0
    report = json.loads(report_path.read_text())
    for port in report["owned_ports"]:
        assert_port_rebinds(port)
    return elapsed, report


def test_ctrl_c_before_first_valid_state_publishes_nothing(tmp_path):
    elapsed_s, report = run_shutdown_scenario(tmp_path, "no-state")
    goals = report["goals"]
    inference_requests = report["inference_requests"]
    assert goals == []
    assert inference_requests == []
    assert elapsed_s <= 0.5
    assert report["publisher_closed"] is True
    assert report["camera_closed"] is True
    assert report["terminal_restored"] is True
    assert report["live_non_daemon_bridge_threads"] == []
```

- [ ] **Step 3: Write failing five-second in-flight shutdown test**

Append the long-path assertion; the fixture scenario's local HTTP handler writes `request_accepted=true` only after accepting `/act-rtc-v1`, then withholds its response for 5.2 seconds:

```python
def test_ctrl_c_with_five_second_request_publishes_exact_final_hold(tmp_path):
    elapsed_s, report = run_shutdown_scenario(tmp_path, "inflight-five-second")
    assert report["request_accepted"] is True
    assert elapsed_s <= 6.5
    goals = report["goals"]
    assert len(goals) == 25
    canonical = json.dumps(goals[0]["goal"], sort_keys=True, separators=(",", ":"))
    assert all(
        json.dumps(entry["goal"], sort_keys=True, separators=(",", ":")) == canonical
        for entry in goals
    )
    assert all(entry["goal"]["navigate_cmd"] == [0.0, 0.0, 0.0, 0.0] for entry in goals)
    deltas = np.diff([entry["scheduled_at"] for entry in goals])
    np.testing.assert_allclose(deltas, 0.02, rtol=0, atol=0.002)
    assert report["publish_attempts"] == 25
    assert report["publisher_closed_after_publish_count"] == 25
    assert report["publisher_closed"] is True
    assert report["camera_closed"] is True
    assert report["terminal_restored"] is True
    assert report["live_non_daemon_bridge_threads"] == []
```

`publish_attempts == 25` is the explicit no-26th assertion. The report is flushed only after publisher closure, so closure ordering is observable rather than inferred.

- [ ] **Step 4: Implement the inference worker**

`HttpInferenceWorker` owns one daemon thread and one `HttpActionClient(timeout=5.0)`. It has a capacity-one input slot, capacity-one result slot, a physical-busy event, and methods matching `InferencePort`. `submit()` rejects when busy. The thread calls `query_rtc_action()` with image exactly `{contract.image_key: request.image}`, state exactly `{"states": request.observation}`, the configured instruction, condition `{}`, dataset `"simple"`, `gt_action=[]`, and history containing the five successor keys plus `reset=True` only when `request.reset` is true. It timestamps completion with monotonic time and always clears physical busy in `finally`. It never retries and never starts a replacement worker. Reuse the exact Task 3 recording-session tests at this worker boundary to prove R0 carries reset and successors omit it.

Implement in bounded subactions:

- [ ] Add capacity-one queues/event and one daemon thread with no HTTP call.
- [ ] Add exact `RtcRequest` serialization and recording-session assertions.
- [ ] Add result/error conversion and `finally` busy clearing.
- [ ] Add bounded join without replacement/retry behavior.

- [ ] **Step 5: Implement local-keyboard and metrics ownership**

`LocalKeyboard` owns the terminal file descriptor, saves original `termios`, enters cbreak, polls with `select`, enqueues only `p`, and restores settings in `finally`. `JsonlMetrics` writes one compact record per preflight, state transition, request, result, publish/preview, fault, and shutdown event; every record includes mode, state, generation, monotonic time, `published`, and `policy_certified`.

- [ ] Implement keyboard save/cbreak entry.
- [ ] Implement nonblocking `p`-only polling.
- [ ] Implement unconditional terminal restoration.
- [ ] Implement one compact metrics-record serializer.

- [ ] **Step 6: Implement the zero-I/O shutdown subprocess fixture**

Implement both exact `--scenario` choices in `bridge_subprocess_fixture.py`. It binds OS-assigned loopback ports and reports them, uses in-memory state/camera/publisher adapters, and never imports rclpy, Unitree, MuJoCo, or a robot adapter. `no-state` writes readiness without producing a state. `inflight-five-second` provides one valid state and bounded hold, starts a loopback `/act-rtc-v1` handler that sets `request_accepted`, then waits 5.2 seconds. Both install the production signal/shutdown coordinator and atomically write the exact JSON keys asserted in Steps 2-3.

- [ ] **Step 7: Implement exact shutdown branching**

On Ctrl-C:

1. latch stop, increment generation, and clear all policy buffers/bookkeeping;
2. call `build_bounded_hold(now)`;
3. if it returns a hold, publish exactly 25 identical goals on consecutive 20 ms schedule ticks, then close the publisher immediately and forbid a 26th publish;
4. if it returns `None`, publish zero messages and continue immediately;
5. signal camera exit, prevent new inference work, join the worker for at most 5.5 seconds, close messaging, restore terminal, and exit;
6. enforce the overall 6.5-second bound for the in-flight case and 0.5-second bound when neither final hold nor HTTP join is pending.

The runtime never signals or terminates the WBC, simulator, policy server, or any robot process.

- [ ] **Step 8: Run runtime/shutdown tests and commit**

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
  "raw_episode_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
  "processed_episode_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
  "source_episode_index": 7,
  "processed_episode_index": 3,
  "converter_commit": "ffffffffffffffffffffffffffffffffffffffff",
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

Create `tests/test_psi0_bridge_fake_integration.py`; both fakes are context managers that allocate their own loopback port and expose request records:

```python
from pathlib import Path
from types import SimpleNamespace
import json
import time

import numpy as np
import pytest

from scripts.psi0_simple_real_bridge import ComposedCameraReader, HttpInferenceWorker
from scripts.tests.fake_composed_camera_server import running_fake_camera
from scripts.tests.fake_psi0_rtc_server import running_fake_policy
from simple.baselines.client import HttpActionClient
from simple.deploy.psi0_simple_bridge import (
    BridgeState, PolicyContract, Psi0SimpleBridge, TimedCameraFrame,
    TimedRobotState,
)
from tests.psi0_bridge_testkit import test_joint_contract

CONTRACT_PATH = Path("scripts/tests/fixtures/psi0_policy_contract_test_v2.json")


def rtc_history(seq, tick, committed, reset=False):
    history = {
        "session_id": "fake-session", "request_seq": seq,
        "observation_tick": tick, "rtc_delay_steps": 6,
        "committed_actions": committed,
    }
    if reset:
        history["reset"] = True
    return history


def query_fake(client, history):
    return client.query_rtc_action(
        {"rgb_head_stereo_left": np.zeros((8, 8, 3), np.uint8)},
        "pick up the object",
        {"states": np.zeros((1, 32), np.float32)},
        {}, history=history, dataset="simple",
    )


def run_wall_clock_fake_bridge(port, duration_s):
    contract = PolicyContract.from_dict(json.loads(CONTRACT_PATH.read_text()))
    worker = HttpInferenceWorker(
        client=HttpActionClient("127.0.0.1", port, timeout=5.0),
        clock=time.monotonic,
    )
    bridge = Psi0SimpleBridge(
        contract, test_joint_contract(), worker, time.monotonic, start_tick=0
    )
    def refresh_inputs():
        now = time.monotonic()
        bridge.update_inputs(
            TimedRobotState(np.zeros(43, np.float32), now),
            TimedCameraFrame(np.zeros((8, 8, 3), np.uint8), now, None),
        )
    refresh_inputs()
    bridge.tick()
    bridge.activate()
    deadline = time.monotonic() + duration_s
    policy_actions_after_fault = 0
    while time.monotonic() < deadline:
        tick_started = time.monotonic()
        refresh_inputs()
        result = bridge.tick()
        if bridge.state is BridgeState.FAULT and result.source_kind == "policy":
            policy_actions_after_fault += 1
        time.sleep(max(0.0, 0.02 - (time.monotonic() - tick_started)))
    worker.close(timeout_s=1.0)
    return SimpleNamespace(
        request_started_at=worker.metrics.first_request_started_at,
        fault_at=bridge.metrics.first_fault_at,
        late_results_discarded=bridge.metrics.discarded_old_generation_results,
        policy_actions_after_fault=policy_actions_after_fault,
        requests_submitted=worker.metrics.requests_submitted,
    )


def test_fake_policy_contract_r0_successor_and_normal_latency_are_exact():
    expected_contract = json.loads(CONTRACT_PATH.read_text())
    with running_fake_policy(CONTRACT_PATH, normal_latency_s=0.05) as fake:
        client = HttpActionClient("127.0.0.1", fake.port, timeout=1.0)
        assert client.get_contract() == expected_contract
        hold = np.zeros((6, 36), np.float32)
        started = time.monotonic()
        r0 = query_fake(client, rtc_history(0, 100, hold, reset=True))
        elapsed = time.monotonic() - started
        assert elapsed >= 0.045
        assert fake.records[0].applied_latency_s == pytest.approx(0.05)
        assert r0.action.shape == (24, 36)
        assert r0.metadata == {
            "session_id": "fake-session", "request_seq": 0,
            "observation_tick": 100, "prediction_horizon": 30,
            "execution_horizon": 24, "rtc_delay_steps": 6,
            "first_action_tick": 106,
        }
        committed = np.arange(6 * 36, dtype=np.float32).reshape(6, 36) / 1000
        r1 = query_fake(client, rtc_history(1, 124, committed))
        assert r1.metadata["first_action_tick"] == 130
        np.testing.assert_array_equal(fake.records[1].committed_actions, committed)
        assert fake.records[1].reset is False
        assert fake.max_concurrent_requests == 1


def test_point_30_second_one_shot_latency_faults_at_deadline_and_is_discarded():
    with running_fake_policy(CONTRACT_PATH, normal_latency_s=0.05) as fake:
        fake.delay_next_request(0.30)
        report = run_wall_clock_fake_bridge(fake.port, duration_s=0.45)
        assert report.request_started_at is not None
        assert report.fault_at <= report.request_started_at + 0.14
        assert report.late_results_discarded == 1
        assert report.policy_actions_after_fault == 0
        assert report.requests_submitted == 1
        assert fake.max_concurrent_requests == 1


def test_fake_camera_round_trip_and_bounded_close():
    with running_fake_camera(key="rgb_head_stereo_left") as fake:
        reader = ComposedCameraReader(
            "127.0.0.1", fake.port, "rgb_head_stereo_left", "rgb"
        )
        frame = reader.wait_for_frame(timeout_s=1.0)
        left = frame.image[:, :20].mean(axis=(0, 1))
        right = frame.image[:, 44:].mean(axis=(0, 1))
        assert left[0] > 200 and left[2] < 30
        assert right[2] > 200 and right[0] < 30
        started = time.monotonic()
        reader.close(timeout_s=0.5)
        assert time.monotonic() - started <= 0.5
    assert_loopback_port_rebinds(fake.port)
```

The helper contains no ROS, DDS, MuJoCo, or GPU dependency.

- [ ] **Step 3: Implement the fake RTC HTTP server**

Use `ThreadingHTTPServer` bound to `127.0.0.1` and an OS-assigned port. `running_fake_policy(..., normal_latency_s=0.05)` must default to exactly `0.05`, record the actual selected delay per request, and track active/max-concurrent handler counts under a lock. Implement only `GET /contract` and `POST /act-rtc-v1`; return 404 for `/act`. Deserialize requests with `RequestMessage.deserialize()`. Validate `dataset_name="simple"`, empty condition, exact image/state dictionaries, R0-only reset, full history key sets, and the supplied `(d,36)` committed prefix. Generate a safe stationary suffix from the final committed command:

```python
actions = np.repeat(committed_actions[-1:], execution_horizon, axis=0).astype(np.float32)
actions[:, 32:36] = 0.0
```

Return NumPy-serialized actions and metadata with the exact seven keys/types from Task 3, echoing session, sequence, observation tick, P/s/d, and `first_action_tick=observation_tick+d`. Expose thread-safe request records and `delay_next_request(seconds)`; close and join the server thread explicitly.

Implement one route/behavior per bounded subaction:

- [ ] Bind loopback/OS-assigned port and implement bounded close/join.
- [ ] Add exact `GET /contract` only.
- [ ] Add request deserialization and exact field/history validation.
- [ ] Add stationary suffix plus exact seven-key metadata response.
- [ ] Add locked concurrency counters and default/one-shot latency.

- [ ] **Step 4: Implement the fake composed-camera server**

Do not subclass or instantiate `SensorServer`, because it binds `tcp://*`. Use only the real nested `ImageMessageSchema` for serialization. The fake owns a `zmq.Context` and PUB socket, sets `LINGER=0`, and binds an OS-assigned endpoint obtained from `socket.bind_to_random_port("tcp://127.0.0.1")`. Publish `msgpack.packb(schema.serialize(), use_bin_type=True)` for a contiguous RGB image whose left patch is `[240,0,0]` and right patch is `[0,0,240]`, plus the selected key timestamp. The publisher socket/context are created, used, and closed in the same server thread. Expose a stop event, join within 0.5 seconds, and assert the exact loopback port can be rebound.

- [ ] Add the loopback PUB-thread lifecycle and port rebind test.
- [ ] Add real-schema JPEG serialization and red/blue sentinel publication.

Both fake scripts expose `--host 127.0.0.1 --port PORT`; they reject any other host. The policy fake additionally exposes `--normal-latency-s 0.05`. Port `0` requests an OS-assigned port for pytest, while the smoke driver uses fixed isolated ports 15555/22086.

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

Create `tests/test_psi0_bridge_smoke_driver.py` and assert the complete immutable launch plan before mocking process creation:

```python
import signal
import subprocess

import numpy as np
import pytest

from scripts.tests.smoke_psi0_simple_bridge import (
    OwnedChildren, SmokeConfig, SmokeSafetyError, build_launch_plan, launch,
    validate_smoke_report,
)

EXPECTED_WBC_ARGS = (
    "uv", "run", "--group", "sonic", "python", "-m",
    "decoupled_wbc.control.main.teleop.run_g1_control_loop",
    "--interface", "sim", "--simulator", "mujoco",
    "--messaging-backend", "ros2", "--enable-waist", "--with-hands",
    "--domain-id", "42", "--no-enable-onscreen", "--no-enable-offscreen",
)


def valid_smoke_config(**updates):
    values = {
        "duration_s": 15.0, "ros_domain_id": 42, "unitree_domain_id": 42,
        "wbc_interface": "lo", "camera_host": "127.0.0.1",
        "camera_port": 15555, "policy_host": "127.0.0.1",
        "policy_port": 22086,
    }
    values.update(updates)
    return SmokeConfig(**values)


def test_smoke_launch_plan_is_exact_and_normal_policy_latency_is_50ms():
    plan = build_launch_plan(valid_smoke_config())
    assert tuple(child.name for child in plan.children) == (
        "wbc", "camera", "policy", "bridge"
    )
    assert plan.children[0].argv == EXPECTED_WBC_ARGS
    assert plan.children[0].env["ROS_DOMAIN_ID"] == "42"
    assert plan.children[0].env["UNITREE_DOMAIN_ID"] == "42"
    assert plan.children[0].env["PYTHONPATH"] == "third_party/decoupled_wbc"
    camera = plan.child("camera")
    assert camera.argv[-4:] == (
        "--host", "127.0.0.1", "--port", "15555"
    )
    policy = plan.child("policy")
    assert "--normal-latency-s" in policy.argv
    index = policy.argv.index("--normal-latency-s")
    assert policy.argv[index + 1] == "0.05"
    assert policy.argv[-4:] == (
        "--host", "127.0.0.1", "--port", "22086"
    )
    bridge = plan.child("bridge")
    assert bridge.use_pty is True
    assert "--mode" in bridge.argv and bridge.argv[bridge.argv.index("--mode") + 1] == "sim-control"
    assert all(child.env["ROS_DOMAIN_ID"] == "42" for child in plan.children)
    assert all(child.env["UNITREE_DOMAIN_ID"] == "42" for child in plan.children)


@pytest.mark.parametrize(
    "updates",
    [
        {"wbc_interface": "eth0"}, {"camera_host": "0.0.0.0"},
        {"policy_host": "192.168.1.2"}, {"ros_domain_id": 41},
        {"unitree_domain_id": 0},
    ],
)
def test_invalid_isolation_is_rejected_before_any_popen(monkeypatch, updates):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: calls.append((a, k)))
    with pytest.raises(SmokeSafetyError):
        launch(valid_smoke_config(**updates))
    assert calls == []
```

The exact Tyro flags are fixed in `EXPECTED_WBC_ARGS`; there is no implementation-time CLI spelling decision.

- [ ] **Step 2: Implement immutable launch-plan validation**

Implement `SmokeConfig`, `ChildSpec`, `LaunchPlan`, and `build_launch_plan()`. Run both Step 1 tests. Do not call `Popen` in this checkbox.

- [ ] **Step 3: Implement process-group ownership and bounded readiness**

Append this unit test before implementing the process owner:

```python
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
        ("killpg", 202, signal.SIGINT), ("waitpid", 202, 0),
        ("killpg", 101, signal.SIGINT), ("waitpid", 101, 0),
    ]
```

Launch each child with `start_new_session=True`, record its PID/PGID/command/start time immediately, and wait for each named readiness event with a monotonic timeout. Cleanup walks only the recorded list in reverse order. No broad process search or `pkill` is permitted.

- [ ] **Step 4: Implement the timed smoke scenario**

Implement and unit-test each timeline action separately:

- [ ] Preflight `0 publishers/1 subscription` after WBC readiness and before bridge launch.
- [ ] Launch each child through the tested `OwnedChildren` registry.
- [ ] Wait for each named bounded readiness signal.
- [ ] Send local `p` through the bridge pseudo-terminal at second 3.
- [ ] Configure the fake policy's next request for exactly 0.30 seconds at second 11.
- [ ] Require worker idle at second 13, send Ctrl-C, and finish by second 15.
- [ ] Reap only owned process groups in `finally`.
- [ ] Verify terminal restoration and rebind every owned port.

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

- [ ] **Step 5: Write and implement exact metric assertions**

Append a generated passing report and one-field mutation matrix:

```python
def passing_smoke_report():
    publish_times = [3.0 + index * 0.02 for index in range(400)]
    committed_array = np.arange(6 * 36, dtype=np.float32).reshape(6, 36) / 1000
    committed_array[:, 31] = 0.5
    committed_array[:, 32:36] = 0.0
    committed = committed_array.tolist()
    return {
        "steady_phases": [{"publish_times": publish_times[:200]},
                          {"publish_times": publish_times[200:]}],
        "requests": [
            {"tick": 100, "committed_actions": committed},
            {"tick": 124, "committed_actions": committed},
            {"tick": 148, "committed_actions": committed},
        ],
        "executed_ticks": list(range(100, 172)),
        "blocked_main_loop_max_gap_s": 0.02,
        "delayed_request_started_at": 11.0,
        "fault_at": 11.12,
        "first_fault_goal_navigation": [0.0, 0.0, 0.0, 0.0],
        "policy_actions_after_fault": 0,
        "late_results_discarded": 1,
        "worker_idle_at_s": 12.8,
        "goal_counts_before": [0, 1],
        "goal_counts_running": [1, 1],
        "goal_counts_after": [0, 1],
        "live_children_after": [],
        "live_threads_after": [],
        "terminal_restored": True,
        "ports_rebound": True,
        "real_interface_connections": 0,
        "extra_goal_publishers": 0,
    }


def test_passing_smoke_report_meets_every_bound():
    result = validate_smoke_report(passing_smoke_report())
    assert result.ok is True
    assert result.failures == ()


@pytest.mark.parametrize(
    "field,value",
    [
        ("blocked_main_loop_max_gap_s", 0.061),
        ("fault_at", 11.141),
        ("first_fault_goal_navigation", [0.0, 0.0, 0.1, 0.0]),
        ("policy_actions_after_fault", 1),
        ("late_results_discarded", 0),
        ("worker_idle_at_s", 13.01),
        ("goal_counts_after", [1, 1]),
        ("live_children_after", [123]),
        ("terminal_restored", False),
        ("ports_rebound", False),
        ("real_interface_connections", 1),
        ("extra_goal_publishers", 1),
    ],
)
def test_each_smoke_failure_is_reported(field, value):
    report = passing_smoke_report()
    report[field] = value
    result = validate_smoke_report(report)
    assert result.ok is False
    assert any(field in failure for failure in result.failures)
```

Append the generated-array failures as executable cases:

```python
def test_smoke_rejects_rate_gap_request_spacing_shape_and_tick_discontinuity():
    mutations = []
    low_rate = passing_smoke_report()
    low_rate["steady_phases"][0]["publish_times"] = [3.0 + index * 0.021 for index in range(200)]
    mutations.append((low_rate, "steady_phases"))
    gap = passing_smoke_report()
    gap["steady_phases"][0]["publish_times"][50] += 0.061
    mutations.append((gap, "maximum_gap"))
    spacing = passing_smoke_report()
    spacing["requests"][1]["tick"] = 125
    mutations.append((spacing, "request_spacing"))
    shape = passing_smoke_report()
    shape["requests"][1]["committed_actions"] = shape["requests"][1]["committed_actions"][:5]
    mutations.append((shape, "committed_actions"))
    skipped = passing_smoke_report()
    skipped["executed_ticks"].remove(130)
    mutations.append((skipped, "executed_ticks"))
    for report, expected in mutations:
        result = validate_smoke_report(report)
        assert result.ok is False
        assert any(expected in failure for failure in result.failures)
```

`validate_smoke_report()` computes per-phase mean frequency from first/last time and count, rejects outside 49-51 Hz, rejects any adjacent gap over 0.060 seconds, requires request tick differences of 24, exact `(6,36)` committed arrays, and an uninterrupted executed-tick sequence.

- [ ] **Step 6: Run the driver unit test without starting MuJoCo**

Run:

```bash
uv run --group dev pytest -q tests/test_psi0_bridge_smoke_driver.py
uv run --group dev ruff check scripts/tests/smoke_psi0_simple_bridge.py \
  tests/test_psi0_bridge_smoke_driver.py
```

Expected: orchestration and fail-closed command validation pass with all process calls mocked.

- [ ] **Step 7: Commit the smoke driver before executing it**

```bash
git add scripts/tests/smoke_psi0_simple_bridge.py tests/test_psi0_bridge_smoke_driver.py
git commit -m "test: add isolated PSI0 bridge smoke driver"
```

- [ ] **Step 8: Run the real isolated simulation smoke test**

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

Create `tests/test_psi0_bridge_cli.py` with no runtime imports beyond the parser/factories:

```python
import pytest

from scripts.psi0_simple_real_bridge import (
    LocalKeyboard, RuntimeDependencyFactories, build_parser,
    build_policy_client, build_runtime_adapters,
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
        "robot_interface", "network_interface", "low_level_topic",
        "real_control", "dds_topic",
    }


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
        wbc_config_client=lambda: InMemoryWbcConfigClient({"env_type": "sim"}),
        state_source=lambda: InMemoryStateSource(),
        camera_reader=lambda: InMemoryCameraReader(),
        graph=lambda: FakeGraph(publishers=0, subscriptions=1),
    )


def test_shadow_factory_never_calls_goal_publisher():
    calls = []
    adapters = build_runtime_adapters(
        mode=BridgeMode.SHADOW,
        publisher_factory=lambda: calls.append("publisher"),
        test_dependencies=valid_adapter_dependencies(),
    )
    assert adapters.goal_publisher is None
    assert calls == []


def test_local_keyboard_toggle_set_is_exact():
    assert LocalKeyboard.ACCEPTED_KEYS == (b"p",)
```

None of these test doubles imports or opens ROS, ZMQ, DDS, or a network socket.

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
- the exact Task 15 live-certification command, 10 discarded warmups, 100 measured requests, zero-failure rule, higher-method p99 limit `(d-1)/50`, and seven co-located evidence files;
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

## Task 15: Add the external live-policy certification bundle tool

**Files:**
- Create: `scripts/benchmark_psi0_rtc_server.py`
- Create: `tests/test_benchmark_psi0_rtc_server.py`
- Modify: `README.md`

- [ ] **Step 1: Write the exact 10-warmup/100-measured benchmark tests**

Create a transport-injected unit test; no live server is contacted:

```python
import json

import numpy as np
import pytest

from scripts.benchmark_psi0_rtc_server import (
    BenchmarkReport, BenchmarkRequestError, BenchmarkSample,
    BenchmarkTransportResponse, benchmark_server, sha256_file,
    write_certification_bundle,
)
from simple.deploy.psi0_simple_bridge import PolicyContract
from tests.psi0_bridge_testkit import policy_payload


FAILURE_KEYS = (
    "timeout", "http", "decode", "shape", "metadata", "bounds", "late"
)


class FakeBenchmarkTransport:
    def __init__(self, measured_latencies, failure_at=None, failure_kind=None):
        self.measured_latencies = list(measured_latencies)
        self.failure_at = failure_at
        self.failure_kind = failure_kind
        self.calls = []
        self.contract_calls = 0

    def get_contract(self):
        self.contract_calls += 1
        return live_policy_payload()

    def query(self, request_index, sample, history):
        self.calls.append((request_index, history))
        measured_index = request_index - 10
        if measured_index == self.failure_at:
            raise BenchmarkRequestError(self.failure_kind)
        latency = 0.02 if request_index < 10 else self.measured_latencies[measured_index]
        return valid_benchmark_response(history, latency_s=latency)


def live_policy_payload(**updates):
    payload = policy_payload(
        test_only=False,
        prediction_horizon=30,
        execution_horizon=24,
        rtc_delay_steps=6,
        rtc_training_max_delay=7,
    )
    payload.update(updates)
    return payload


def representative_samples(count):
    return tuple(
        BenchmarkSample(
            image=np.full((8, 8, 3), index % 256, np.uint8),
            state=np.full((1, 32), index / 1000.0, np.float32),
            instruction=f"representative instruction {index}",
        )
        for index in range(count)
    )


def valid_benchmark_response(history, latency_s):
    first_tick = history["observation_tick"] + history["rtc_delay_steps"]
    action = np.zeros((24, 36), np.float32)
    action[:, 31] = 0.5
    return BenchmarkTransportResponse(
        action=action,
        metadata={
            "session_id": history["session_id"],
            "request_seq": history["request_seq"],
            "observation_tick": history["observation_tick"],
            "prediction_horizon": 30,
            "execution_horizon": 24,
            "rtc_delay_steps": 6,
            "first_action_tick": first_tick,
        },
        latency_s=latency_s,
    )


def test_benchmark_discards_ten_warmups_and_certifies_one_hundred_requests(tmp_path):
    transport = FakeBenchmarkTransport([0.05] * 99 + [0.099])
    report = benchmark_server(
        transport=transport,
        samples=representative_samples(100),
        contract=PolicyContract.from_dict(live_policy_payload()),
        warmup_requests=10,
        measured_requests=100,
    )
    assert len(transport.calls) == 110
    assert transport.contract_calls == 1
    assert report == BenchmarkReport(
        warmup_requests=10,
        measured_requests=100,
        successes=100,
        failures={key: 0 for key in FAILURE_KEYS},
        p99_latency_s=0.099,
        latency_limit_s=0.10,
        certified=True,
    )
    assert [call[0] for call in transport.calls] == list(range(110))
    assert transport.calls[0][1]["reset"] is True
    assert all("reset" not in history for _, history in transport.calls[1:])


def test_p99_uses_higher_method_and_point_101_fails_gate():
    report = benchmark_server(
        transport=FakeBenchmarkTransport([0.05] * 99 + [0.101]),
        samples=representative_samples(100),
        contract=PolicyContract.from_dict(live_policy_payload()),
        warmup_requests=10,
        measured_requests=100,
    )
    assert report.p99_latency_s == 0.101
    assert report.latency_limit_s == 0.10
    assert report.certified is False


@pytest.mark.parametrize("failure_kind", FAILURE_KEYS)
def test_every_failure_class_is_counted_and_prevents_certification(failure_kind):
    report = benchmark_server(
        transport=FakeBenchmarkTransport(
            [0.05] * 100, failure_at=17, failure_kind=failure_kind
        ),
        samples=representative_samples(100),
        contract=PolicyContract.from_dict(live_policy_payload()),
        warmup_requests=10,
        measured_requests=100,
    )
    assert report.successes == 99
    assert report.failures[failure_kind] == 1
    assert sum(report.failures.values()) == 1
    assert report.certified is False
```

No external fixtures are used.

- [ ] **Step 2: Write the co-located evidence-bundle test**

```python
def write_representative_npz(path, samples):
    np.savez(
        path,
        images=np.stack([sample.image for sample in samples]),
        states=np.stack([sample.state for sample in samples]),
        instructions=np.asarray([sample.instruction for sample in samples]),
    )


def test_certified_bundle_contains_exact_contract_hash_and_commit_evidence(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    dataset_manifest = tmp_path / "dataset.json"
    samples = tmp_path / "samples.npz"
    checkpoint.write_bytes(b"checkpoint")
    dataset_manifest.write_bytes(b"dataset")
    write_representative_npz(samples, representative_samples(100))
    contract_payload = live_policy_payload(
        checkpoint_sha256=sha256_file(checkpoint),
        dataset_manifest_sha256=sha256_file(dataset_manifest),
    )
    contract_path = tmp_path / "policy-contract.json"
    contract_path.write_text(json.dumps(contract_payload))
    output = tmp_path / "bundle"
    write_certification_bundle(
        output_dir=output,
        report=BenchmarkReport(10, 100, 100, {key: 0 for key in FAILURE_KEYS}, 0.05, 0.10, True),
        policy_contract_path=contract_path,
        fetched_server_contract=contract_payload,
        checkpoint_path=checkpoint,
        dataset_manifest_path=dataset_manifest,
        samples_path=samples,
    )
    assert {path.name for path in output.iterdir()} == {
        "latency_report.json", "policy_contract.json", "checkpoint.sha256",
        "dataset_manifest.sha256", "request_samples.sha256",
        "server_commit.txt", "converter_commit.txt",
    }
    assert json.loads((output / "policy_contract.json").read_text()) == contract_payload
    assert (output / "checkpoint.sha256").read_text().strip() == sha256_file(checkpoint)
    assert (output / "dataset_manifest.sha256").read_text().strip() == sha256_file(dataset_manifest)
    assert (output / "request_samples.sha256").read_text().strip() == sha256_file(samples)
    assert (output / "server_commit.txt").read_text().strip() == contract_payload["server_commit"]
    assert (output / "converter_commit.txt").read_text().strip() == contract_payload["converter_commit"]
    latency = json.loads((output / "latency_report.json").read_text())
    assert latency["warmup_requests"] == 10
    assert latency["measured_requests"] == latency["successes"] == 100
    assert latency["failures"] == {key: 0 for key in FAILURE_KEYS}
    assert latency["p99_method"] == "higher"
    assert latency["certified"] is True
```

The bundle writer first verifies the checkpoint and dataset hashes equal the contract, verifies the fetched server contract equals the local contract, writes all seven files into a new temporary sibling directory, fsyncs, then atomically renames that directory to the requested new output path. It refuses an existing output path and never overwrites evidence.

- [ ] **Step 3: Implement representative request loading and response classification**

Require an NPZ with arrays `images` (`N,H,W,3`, contiguous `uint8`) and `states` (`N,1,32`, finite `float32`) and `N >= 100`. Fetch `/contract` exactly once before warmup, parse it strictly, and require full equality with the local contract. Implement one classifier for each exact failure key. Each request uses dataset `simple`, empty condition, selected RGB key, R0-only reset, monotonically increasing request sequence/observation ticks, and the full committed prefix. Validate response shape, metadata, bounds, and deadline with the same production functions as the bridge.

- [ ] **Step 4: Implement warmed measurement and higher-method p99**

Issue exactly 10 warmup requests and exclude them from every measured statistic. Select samples deterministically as `samples[request_index % len(samples)]`, so the subsequent 100 measured requests cover every representative sample exactly once. Issue exactly 100 measured requests, never retry, and account for every attempt as exactly one success or one failure category. Compute:

```python
p99_latency_s = float(np.quantile(success_latencies, 0.99, method="higher"))
latency_limit_s = (contract.rtc_delay_steps - 1) / contract.action_frequency_hz
certified = (
    len(success_latencies) == 100
    and sum(failures.values()) == 0
    and p99_latency_s <= latency_limit_s
)
```

Return nonzero from the CLI unless `certified` is true; a failed run may write `latency_report.json` to a separately named diagnostic directory, but must not write a certified bundle.

- [ ] **Step 5: Implement the evidence writer and exact CLI**

Expose this command without defaults for artifact identity paths:

```bash
uv run python scripts/benchmark_psi0_rtc_server.py \
  --server-url http://SERVER:22085 \
  --policy-contract artifacts/psi0/policy-contract.json \
  --checkpoint artifacts/psi0/checkpoint.pt \
  --dataset-manifest artifacts/psi0/dataset-manifest.json \
  --representative-samples artifacts/psi0/representative-requests.npz \
  --warmup-requests 10 \
  --measured-requests 100 \
  --output-dir artifacts/psi0/live-certification
```

The CLI rejects any warmup value other than 10 or measured value other than 100 so the documented gate cannot be weakened accidentally.

- [ ] **Step 6: Run unit tests, lint, and commit**

Run:

```bash
uv run --group dev pytest -q tests/test_benchmark_psi0_rtc_server.py
uv run --group dev ruff check scripts/benchmark_psi0_rtc_server.py \
  tests/test_benchmark_psi0_rtc_server.py
uv run --group dev ruff format --check scripts/benchmark_psi0_rtc_server.py \
  tests/test_benchmark_psi0_rtc_server.py
git add scripts/benchmark_psi0_rtc_server.py \
  tests/test_benchmark_psi0_rtc_server.py README.md
git commit -m "feat: certify live PSI0 RTC latency"
```

Expected: all deterministic benchmark/bundle tests pass. Do not run the live command during implementation unless the external server and representative artifacts have separately been provided.

## Task 16: Run complete verification and prove clean recursive delivery

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
  tests/test_psi0_bridge_cli.py \
  tests/test_benchmark_psi0_rtc_server.py
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
  scripts/tests/smoke_psi0_simple_bridge.py \
  scripts/benchmark_psi0_rtc_server.py tests
uv run --group dev ruff format --check src/simple/deploy scripts/psi0_simple_real_bridge.py \
  scripts/certify_psi0_policy_contract.py \
  scripts/tests/fake_psi0_rtc_server.py \
  scripts/tests/fake_composed_camera_server.py \
  scripts/tests/bridge_subprocess_fixture.py \
  scripts/tests/smoke_psi0_simple_bridge.py \
  scripts/benchmark_psi0_rtc_server.py tests
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
| Same-episode raw/processed binding and converter-commit provenance | Tasks 1-2 |
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
| Live gate: 10 warmups, 100 measured, failure counts, higher-method p99, evidence bundle | Task 15 |
| Reachable nested commit and clean recursive checkout | Tasks 6, 16 |
| No real-control path and documented later gates | Tasks 0, 10, 14-16 |

## Execution checkpoints

1. Stop after Task 5 if the nested tests do not pass.
2. Stop at Task 6 until explicit push authority and a fetchable remote are available.
3. Stop before Task 13 if ROS graph ownership is not exactly `0 publishers/1 WBC subscription`, if any interface is not loopback, or if either domain is not 42.
4. Stop before replacing the fake policy server unless the exact Task 15 command produces a certified seven-file bundle: matching `/act-rtc-v1` contract/commits/hashes, 10 discarded warmups, 100/100 successful measured requests, zero categorized failures, and higher-method p99 at or below `(d-1)/50` seconds.
5. Completing this plan does not authorize real-robot deployment. That remains a separate reviewed design and implementation phase.
