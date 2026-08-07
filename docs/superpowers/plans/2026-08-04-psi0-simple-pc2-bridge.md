# PSI0 SIMPLE PC2 Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the approved fail-closed PSI0-to-decoupled-WBC bridge for `shadow` and isolated MuJoCo `sim-control` modes, without adding a real-robot control path.

**Architecture:** A transport-independent core owns named-joint mapping, input validation, bounded holds, RTC scheduling, action validation, slew limiting, and the PAUSED/ACTIVE/FAULT/STOPPED state machine. A thin PC2 runtime owns ROS2/ZMQ/HTTP/keyboard resources and injects them into the core. The decoupled-WBC submodule exposes a canonical model contract only after its model and policy load, and the root bridge refuses to create a publisher until WBC, policy, graph-ownership, camera, and state contracts pass.

**Tech Stack:** Python 3.10, NumPy, requests, pytest, ROS2/rclpy, ZeroMQ/msgpack, OpenCV JPEG codec, ONNX Runtime, MuJoCo, uv, Git submodules.

---

## Non-negotiable safety boundaries

- This plan implements only `shadow` and `sim-control`. Do not add a `real-control` enum value, CLI option, network-interface fallback, or Unitree low-level publisher.
- `sim-control` alone is locked to loopback/domain 42. `shadow` may use explicitly selected valid ROS/Unitree domains to observe a later real system, but it never creates a goal publisher; if neither local nor server policy contract is usable it remains observation-only and refuses `p` without exiting.
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

Every test snippet below is literal minimum test code, not pseudocode. Helper names are introduced in the same task before a later test uses them. Source blocks targeting the same file are cumulative in task order. A file's first production block owns the imports needed at that task; if a later task explicitly says to extend the import header, move that exact import block to the top of the file rather than inserting it at the current source position. No other production block introduces a mid-file import. During implementation, complete one checkbox, run the named focused test, and only then continue; no checkbox may be expanded into an unreviewed multi-hour action. When a code block contains several functions, add and run one function at a time in source order (each is a separate 2-5 minute red/green action). After assembling each changed Python file, run `ruff format` on that exact file before its listed `ruff format --check`/`ruff check` gate; formatting is mechanical and does not permit behavioral changes.

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

Extend the existing import header with `import hashlib` and `import re`. Add these complete helpers:

```python
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_converter_commit(repository_root: Path) -> str:
    commit = subprocess.run(
        [
            "git", "log", "-1", "--format=%H", "--",
            "scripts/postprocess_psi0.py",
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("could not resolve converter source commit")
    return commit


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

Immediately after parsing CLI arguments, resolve the identity exactly once:

```python
repository_root = Path(__file__).resolve().parents[1]
converter_commit = resolve_converter_commit(repository_root)
```

Immediately before each `episodes.append(...)`, construct the record from that iteration's real source path/index and add it under the exact `conversion_provenance` key in the episode dictionary:

```python
conversion_provenance = build_conversion_provenance(
    source_path=data_path,
    source_episode_index=ep_index,
    skip=args.skip,
    downsample=args.downsample,
    converter_commit=converter_commit,
)
```

The `episodes.append({...})` dictionary includes `"conversion_provenance": conversion_provenance` alongside its existing fields. The later certification tool must not accept provenance supplied only on its command line.

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
    cmd[:, 6] = 0.74
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
    assert payload["test_only"] is False
```

- [ ] **Step 2: Verify the new module is absent**

Run:

```bash
uv run --group dev pytest -q tests/test_certify_psi0_policy_contract.py
```

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement and test the same-episode loader**

Create `scripts/certify_psi0_policy_contract.py` with this complete loader. It accepts only Parquet episode files and binds the processed row to the raw source through the converter-written metadata:

```python
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq


EPISODE_NAME = re.compile(r"episode_(\d{6})\.parquet")
INITIAL_COMMAND = np.asarray(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.74, 0.74, 0.74],
    np.float32,
)


@dataclass(frozen=True)
class BoundEpisode:
    stored_state: np.ndarray
    history_cmd: np.ndarray
    source_episode_index: int
    processed_episode_index: int
    raw_episode_sha256: str
    processed_episode_sha256: str
    converter_commit: str


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_episode_index(name):
    match = EPISODE_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid episode parquet filename: {name}")
    return int(match.group(1))


def _metadata_record(path, processed_index):
    records = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid episodes metadata line {line_number}"
            ) from error
        if type(record) is not dict:
            raise ValueError("episode metadata record must be an object")
        records.append(record)
    matches = [
        record for record in records
        if record.get("episode_index") == processed_index
    ]
    if len(matches) != 1:
        raise ValueError("expected exactly one processed episode metadata record")
    return matches[0]


def load_bound_episode(raw, processed, episodes_jsonl):
    raw = Path(raw)
    processed = Path(processed)
    episodes_jsonl = Path(episodes_jsonl)
    source_index = parse_episode_index(raw.name)
    processed_index = parse_episode_index(processed.name)
    record = _metadata_record(episodes_jsonl, processed_index)
    provenance = record.get("conversion_provenance")
    required_provenance = {
        "source_episode_index", "source_parquet_sha256", "skip",
        "downsample", "converter_commit",
    }
    if type(provenance) is not dict or set(provenance) != required_provenance:
        raise ValueError("conversion provenance keys")
    if (
        type(provenance["source_episode_index"]) is not int
        or provenance["source_episode_index"] != source_index
    ):
        raise ValueError("source episode index mismatch")
    raw_hash = sha256_file(raw)
    if provenance["source_parquet_sha256"] != raw_hash:
        raise ValueError("raw episode hash mismatch")
    converter_commit = provenance["converter_commit"]
    if (
        type(converter_commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}", converter_commit) is None
    ):
        raise ValueError("invalid converter commit")
    skip = provenance["skip"]
    downsample = provenance["downsample"]
    if type(skip) is not int or skip < 0:
        raise ValueError("conversion skip must be a nonnegative integer")
    if type(downsample) is not int or downsample <= 0:
        raise ValueError("conversion downsample must be a positive integer")

    raw_table = pq.read_table(
        raw, columns=["observation.amo_policy_command"]
    )
    command = np.asarray(
        raw_table["observation.amo_policy_command"].to_pylist(),
        dtype=np.float32,
    )
    if command.ndim != 2 or command.shape[1] != 9:
        raise ValueError("raw command must have shape (T,9)")
    if not np.isfinite(command).all() or skip >= len(command):
        raise ValueError("raw command is non-finite or skip removes all frames")
    full_history = np.concatenate(
        [INITIAL_COMMAND[None], command[:-1]], axis=0
    )
    history = np.ascontiguousarray(full_history[skip::downsample])

    processed_table = pq.read_table(
        processed, columns=["states", "episode_index"]
    )
    processed_indices = set(
        processed_table["episode_index"].to_pylist()
    )
    if processed_indices != {processed_index}:
        raise ValueError("processed episode_index does not match filename")
    stored = np.asarray(
        processed_table["states"].to_pylist(), dtype=np.float32
    )
    if stored.ndim != 2 or stored.shape[1] < 32:
        raise ValueError("processed states must have shape (T,>=32)")
    if not np.isfinite(stored).all():
        raise ValueError("processed states must be finite")
    if type(record.get("length")) is not int:
        raise ValueError("processed metadata length must be an integer")
    if len(stored) != len(history) or len(stored) != record["length"]:
        raise ValueError("raw history and processed frame counts differ")
    return BoundEpisode(
        stored_state=np.ascontiguousarray(stored),
        history_cmd=history,
        source_episode_index=source_index,
        processed_episode_index=processed_index,
        raw_episode_sha256=raw_hash,
        processed_episode_sha256=sha256_file(processed),
        converter_commit=converter_commit,
    )
```

The filename parser rejects standalone `.npy` inputs before any array read, because they cannot prove same-episode provenance.

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


def validate_policy_contract_payload(payload):
    types = {
        "schema": str, "test_only": bool, "checkpoint_sha256": str,
        "dataset_manifest_sha256": str, "raw_episode_sha256": str,
        "processed_episode_sha256": str, "source_episode_index": int,
        "processed_episode_index": int, "converter_commit": str,
        "server_commit": str, "converter_layout": str, "observation_dim": int,
        "action_dim": int, "action_frequency_hz": int,
        "prediction_horizon": int, "execution_horizon": int,
        "rtc_delay_steps": int, "rtc_training_max_delay": int,
        "rtc_enabled": bool, "rtc_endpoint": str, "request_semantics": str,
        "response_semantics": str, "image_key": str,
        "camera_color_order": str,
    }
    if type(payload) is not dict or set(payload) != set(types):
        raise TypeError("policy contract keys do not exactly match v2 schema")
    for key, expected in types.items():
        if type(payload[key]) is not expected:
            raise TypeError(f"policy contract field {key} must be {expected.__name__}")
    for key in (
        "checkpoint_sha256", "dataset_manifest_sha256", "raw_episode_sha256",
        "processed_episode_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", payload[key]) is None:
            raise ValueError(key)
    for key in ("converter_commit", "server_commit"):
        if re.fullmatch(r"[0-9a-f]{40}", payload[key]) is None:
            raise ValueError(key)
    d = payload["rtc_delay_steps"]
    s = payload["execution_horizon"]
    p = payload["prediction_horizon"]
    if not (2 <= d <= s and d + s <= p and d < payload["rtc_training_max_delay"]):
        raise ValueError("invalid RTC horizon/delay contract")


def build_policy_contract_payload(
    *, bound_episode, checkpoint_sha256, dataset_manifest_sha256,
    server_commit, prediction_horizon, execution_horizon,
    rtc_delay_steps, rtc_training_max_delay,
):
    certification = certify_layout(
        bound_episode.stored_state, bound_episode.history_cmd
    )
    payload = {
        "schema": "simple.psi0.policy-contract.v2",
        "test_only": False,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "raw_episode_sha256": bound_episode.raw_episode_sha256,
        "processed_episode_sha256": bound_episode.processed_episode_sha256,
        "source_episode_index": bound_episode.source_episode_index,
        "processed_episode_index": bound_episode.processed_episode_index,
        "converter_commit": bound_episode.converter_commit,
        "server_commit": server_commit,
        "converter_layout": certification["layout"],
        "observation_dim": 32,
        "action_dim": 36,
        "action_frequency_hz": 50,
        "prediction_horizon": prediction_horizon,
        "execution_horizon": execution_horizon,
        "rtc_delay_steps": rtc_delay_steps,
        "rtc_training_max_delay": rtc_training_max_delay,
        "rtc_enabled": True,
        "rtc_endpoint": "/act-rtc-v1",
        "request_semantics": "exact-post-slew-committed-prefix",
        "response_semantics": "denormalized-executable-suffix",
        "image_key": "rgb_head_stereo_left",
        "camera_color_order": "rgb",
    }
    validate_policy_contract_payload(payload)
    return payload
```

Add the complete atomic writer and CLI below. `os.link()` publishes the completed temporary file without an overwrite race; an existing destination raises `FileExistsError` and is left unchanged:

```python
def atomic_write_policy_contract(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--raw-episode-parquet", required=True)
    parser.add_argument("--processed-episode-parquet", required=True)
    parser.add_argument("--processed-episodes-jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--server-commit", required=True)
    parser.add_argument("--prediction-horizon", required=True, type=int)
    parser.add_argument("--execution-horizon", required=True, type=int)
    parser.add_argument("--rtc-delay-steps", required=True, type=int)
    parser.add_argument(
        "--rtc-training-max-delay", required=True, type=int
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    bound = load_bound_episode(
        args.raw_episode_parquet,
        args.processed_episode_parquet,
        args.processed_episodes_jsonl,
    )
    payload = build_policy_contract_payload(
        bound_episode=bound,
        checkpoint_sha256=sha256_file(args.checkpoint),
        dataset_manifest_sha256=sha256_file(args.dataset_manifest),
        server_commit=args.server_commit,
        prediction_horizon=args.prediction_horizon,
        execution_horizon=args.execution_horizon,
        rtc_delay_steps=args.rtc_delay_steps,
        rtc_training_max_delay=args.rtc_training_max_delay,
    )
    atomic_write_policy_contract(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The causal certification is represented by the strict `converter_layout` field and does not add an unrecognized JSON key. Reject a non-40-character hexadecimal converter or server commit and any RTC tuple that violates the approved inequalities. The writer accepts no `test_only` argument: only the checked-in fake fixture in Task 12 may set it true. Task 9 owns `PolicyContract`; to preserve TDD task order, this task defines an identical private `validate_policy_contract_payload()` key/type/value validator, and Task 9 replaces it with the shared parser without changing the emitted bytes.

Complete and run one focused test after each bounded subaction:

- [ ] Add only `certify_layout()` and its three causal-layout cases.
- [ ] Add only `build_policy_contract_payload(bound_episode=..., ...)` and assert its exact key/type set.
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

    @property
    def timestamp(self):
        return str(datetime.now()).replace(" ", "_").replace(":", "-")

    def get_contract(self, timeout: float = 2.0) -> dict[str, Any]:
        response = self.session.get(
            f"http://{self.server_ip}:{self.server_port}/contract", timeout=timeout
        )
        response.raise_for_status()
        try:
            result = response.json()
        except Exception as error:
            raise RuntimeError(f"invalid policy contract JSON: {error}") from error
        if type(result) is not dict:
            raise RuntimeError("policy contract response must be a JSON object")
        return result

    def _post(self, path: str, request: RequestMessage) -> dict[str, Any]:
        response = self.session.post(
            f"http://{self.server_ip}:{self.server_port}{path}",
            json=request.serialize(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        try:
            payload = response.json()
            payload = convert_numpy_in_dict(payload, numpy_deserialize)
        except Exception as error:
            raise RuntimeError(str(error)) from error
        if type(payload) is not dict:
            raise RuntimeError("policy response must be a JSON object")
        return payload

    def query_action(
        self, image_dict, instruction, state_dict, condition_dict,
        history=None, dataset="grasp", gt_action=None,
    ):
        if history is None:
            history = {key: [] for key in image_dict}
        if gt_action is None:
            gt_action = []
        request = RequestMessage(
            image_dict, instruction, history, state_dict, condition_dict,
            gt_action, dataset, self.timestamp,
        )
        try:
            parsed = ResponseMessage.deserialize(self._post("/act", request))
        except (requests.Timeout, requests.HTTPError):
            raise
        except Exception as error:
            raise RuntimeError(str(error)) from error
        trajectory = parsed.traj_image
        if not isinstance(trajectory, np.ndarray) or trajectory.ndim != 3:
            trajectory = None
        return parsed.action, parsed.err, trajectory

    def query_rtc_action(
        self, image_dict, instruction, state_dict, condition_dict,
        *, history, dataset="simple",
    ) -> RtcActionResponse:
        request = RequestMessage(
            image_dict, instruction, history, state_dict, condition_dict,
            [], dataset, self.timestamp,
        )
        payload = self._post("/act-rtc-v1", request)
        if set(payload) != {"action", "metadata"}:
            raise RuntimeError("RTC response requires action and metadata")
        metadata = payload["metadata"]
        metadata_types = {
            "session_id": str,
            "request_seq": int,
            "observation_tick": int,
            "prediction_horizon": int,
            "execution_horizon": int,
            "rtc_delay_steps": int,
            "first_action_tick": int,
        }
        if type(metadata) is not dict or set(metadata) != set(metadata_types):
            raise RuntimeError("RTC response metadata key set")
        for key, expected_type in metadata_types.items():
            if type(metadata[key]) is not expected_type:
                raise RuntimeError(f"RTC response metadata {key} type")
        action = payload["action"]
        if type(action) is not np.ndarray:
            raise RuntimeError("RTC response action must be a NumPy array")
        return RtcActionResponse(action=action, metadata=dict(metadata))
```

Replace the old `HttpActionClient` class with this class rather than retaining two `query_action()` definitions. `_post()` always passes `timeout=self.timeout`; legacy `/act` keeps the same return tuple. `/act-rtc-v1` requires exactly `action` and `metadata`, deserializes NumPy payloads, requires metadata's exact seven-key set with `str` for `session_id` and exact `int` (not `bool`) for the other six keys, and returns `RtcActionResponse` without adding or coercing missing metadata.

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
import pytest
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


def test_model_contract_rejects_wrong_onnx_feature_signature():
    def wrong_inspector(path):
        del path
        return {
            "input": {
                "name": "observations", "shape": ["dynamic", 515],
                "feature_size": 515,
            },
            "output": {
                "name": "actions", "shape": ["dynamic", 15],
                "feature_size": 15,
            },
        }

    with pytest.raises(ValueError, match="516"):
        build_model_contract(
            robot_model=instantiate_g1_robot_model(
                waist_location="lower_and_upper_body"
            ),
            config=ControlLoopConfig(
                interface="sim", enable_waist=True, domain_id=42
            ),
            repository_root=Path(decoupled_wbc.__file__).resolve().parent,
            git_identity=GitIdentity(
                commit="a" * 40, working_tree_clean=True
            ),
            onnx_inspector=wrong_inspector,
        )
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
    assert digest_model_contract(actual) != digest_model_contract(valid_contract)
    with pytest.raises(ValueError, match="connected WBC model contract mismatch"):
        validate_expected_model_contract(actual, valid_contract)
```

- [ ] **Step 2: Write the failing service-construction order test**

Make construction injectable through `_build_attested_components(config, backend, factories)`. Add this complete test to `test_g1_control_loop_contract.py`:

```python
def test_service_is_created_only_after_model_policy_and_attestation():
    from types import SimpleNamespace

    from decoupled_wbc.control.main.teleop.run_g1_control_loop import (
        _build_attested_components,
    )

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


def test_attested_construction_rolls_back_every_created_resource():
    from types import SimpleNamespace

    import pytest

    from decoupled_wbc.control.main.teleop.run_g1_control_loop import (
        _build_attested_components,
    )

    cases = (
        ("policy", ["environment.close"]),
        ("model_contract", ["policy.close", "environment.close"]),
        ("service", ["policy.close", "environment.close"]),
    )
    for failure, expected_tail in cases:
        events = []
        environment = SimpleNamespace(
            sim=True,
            start_simulator=lambda: events.append("simulator.start"),
            close=lambda: events.append("environment.close"),
        )
        policy = SimpleNamespace(
            close=lambda: events.append("policy.close")
        )

        def make_policy(*_args, **_kwargs):
            events.append("policy")
            if failure == "policy":
                raise RuntimeError("policy failed")
            return policy

        def make_contract(_config, _model):
            events.append("model_contract")
            if failure == "model_contract":
                raise RuntimeError("attestation failed")
            return {
                "model_contract": {},
                "model_contract_sha256": "d" * 64,
            }

        def make_service(_backend, _topic, _payload):
            events.append("service")
            raise RuntimeError("service failed")

        factories = SimpleNamespace(
            robot_model=lambda **_kwargs: object(),
            environment=lambda **_kwargs: environment,
            policy=make_policy,
            model_contract_payload=make_contract,
            service_server=make_service,
        )
        with pytest.raises(RuntimeError, match="failed"):
            _build_attested_components(
                ControlLoopConfig(
                    interface="sim", enable_waist=True, domain_id=42
                ),
                backend=object(),
                factories=factories,
            )
        assert events[-len(expected_tail):] == expected_tail


def make_runtime_factories(failure=None):
    from types import SimpleNamespace

    events = []

    def start_simulator():
        events.append("simulator.start")
        raise RuntimeError("simulator start failed")

    environment = SimpleNamespace(
        sim=failure == "simulator_start",
        start_simulator=start_simulator,
        close=lambda: events.append("environment.close"),
    )
    policy = SimpleNamespace(close=lambda: events.append("policy.close"))
    service = SimpleNamespace(close=lambda: events.append("service.close"))
    manager = SimpleNamespace(
        shutdown=lambda: events.append("manager.shutdown"),
    )

    def create_rate(_frequency):
        events.append("rate")
        if failure == "rate":
            raise RuntimeError("rate failed")
        return SimpleNamespace()

    manager.create_rate = create_rate

    def robot_model(**_kwargs):
        events.append("robot_model")
        if failure == "robot_model":
            raise RuntimeError("robot model failed")
        return object()

    def environment_factory(**_kwargs):
        events.append("environment")
        if failure == "environment":
            raise RuntimeError("environment failed")
        return environment

    def policy_factory(*_args, **_kwargs):
        events.append("policy")
        if failure == "policy":
            raise RuntimeError("policy failed")
        return policy

    def model_contract_payload(_config, _model):
        events.append("model_contract")
        if failure == "model_contract":
            raise RuntimeError("model contract failed")
        return {
            "model_contract": {}, "model_contract_sha256": "d" * 64,
        }

    def service_server(_backend, _topic, _payload):
        events.append("service")
        if failure == "service":
            raise RuntimeError("service failed")
        return service

    def publisher(_backend, topic):
        events.append(f"publisher:{topic}")
        if failure == "publisher":
            raise RuntimeError("publisher failed")
        return SimpleNamespace()

    class Dispatcher:
        def register(self, _listener):
            events.append("dispatcher.register")
            if failure == "dispatcher_register":
                raise RuntimeError("dispatcher register failed")

        def start(self):
            events.append("dispatcher.start")
            if failure == "dispatcher_start":
                raise RuntimeError("dispatcher start failed")

        def stop(self):
            events.append("dispatcher.stop")

    def subscriber(_backend, _topic, **_kwargs):
        events.append("subscriber")
        if failure == "subscriber":
            raise RuntimeError("subscriber failed")
        return SimpleNamespace()

    def make_simple(stage):
        events.append(stage)
        if failure == stage:
            raise RuntimeError(f"{stage} failed")
        return SimpleNamespace()

    def dispatcher_factory(_kind):
        events.append("dispatcher")
        if failure == "dispatcher":
            raise RuntimeError("dispatcher failed")
        return Dispatcher()

    factories = SimpleNamespace(
        loop_manager=lambda _backend, **_kwargs: (
            events.append("manager") or manager
        ),
        robot_model=robot_model,
        environment=environment_factory,
        policy=policy_factory,
        model_contract_payload=model_contract_payload,
        service_server=service_server,
        publisher=publisher,
        telemetry=lambda **_kwargs: make_simple("telemetry"),
        keyboard_listener=lambda: make_simple("keyboard_listener"),
        keyboard_estop=lambda: make_simple("keyboard_estop"),
        dispatcher=dispatcher_factory,
        subscriber=subscriber,
    )
    return factories, events


@pytest.mark.parametrize(
    "failure,expected_cleanup",
    [
        ("robot_model", ["manager.shutdown"]),
        ("environment", ["manager.shutdown"]),
        ("simulator_start", ["environment.close", "manager.shutdown"]),
        ("policy", ["environment.close", "manager.shutdown"]),
        (
            "model_contract",
            ["policy.close", "environment.close", "manager.shutdown"],
        ),
        (
            "service",
            ["policy.close", "environment.close", "manager.shutdown"],
        ),
        (
            "publisher",
            ["service.close", "policy.close", "environment.close",
             "manager.shutdown"],
        ),
        (
            "telemetry",
            ["service.close", "policy.close", "environment.close",
             "manager.shutdown"],
        ),
        (
            "keyboard_listener",
            ["service.close", "policy.close", "environment.close",
             "manager.shutdown"],
        ),
        (
            "keyboard_estop",
            ["service.close", "policy.close", "environment.close",
             "manager.shutdown"],
        ),
        (
            "dispatcher",
            ["service.close", "policy.close", "environment.close",
             "manager.shutdown"],
        ),
        (
            "dispatcher_register",
            ["dispatcher.stop", "service.close", "policy.close",
             "environment.close", "manager.shutdown"],
        ),
        (
            "dispatcher_start",
            ["dispatcher.stop", "service.close", "policy.close",
             "environment.close", "manager.shutdown"],
        ),
        (
            "rate",
            ["dispatcher.stop", "service.close", "policy.close",
             "environment.close", "manager.shutdown"],
        ),
        (
            "subscriber",
            ["dispatcher.stop", "service.close", "policy.close",
             "environment.close", "manager.shutdown"],
        ),
    ],
)
def test_every_complete_startup_failure_rolls_back_all_owned_resources(
    failure, expected_cleanup,
):
    from decoupled_wbc.control.main.teleop.run_g1_control_loop import (
        _build_owned_runtime,
    )

    factories, events = make_runtime_factories(failure)
    with pytest.raises(RuntimeError, match="failed"):
        _build_owned_runtime(
            ControlLoopConfig(
                interface="sim", enable_waist=True, domain_id=42
            ),
            factories=factories,
        )
    assert events[-len(expected_cleanup):] == expected_cleanup
    assert events.count("manager.shutdown") == 1


def test_successful_runtime_cleanup_is_reverse_order_complete_and_idempotent():
    from decoupled_wbc.control.main.teleop.run_g1_control_loop import (
        _build_owned_runtime,
    )

    factories, events = make_runtime_factories()
    runtime = _build_owned_runtime(
        ControlLoopConfig(interface="sim", enable_waist=True, domain_id=42),
        factories=factories,
    )
    runtime.close()
    first_close_events = list(events)
    runtime.close()
    assert events == first_close_events
    assert events[-5:] == [
        "dispatcher.stop", "service.close", "policy.close",
        "environment.close", "manager.shutdown",
    ]


def test_runtime_cleanup_failure_cannot_skip_later_owned_resources():
    from decoupled_wbc.control.main.teleop.run_g1_control_loop import (
        _build_owned_runtime,
    )

    factories, events = make_runtime_factories()
    runtime = _build_owned_runtime(
        ControlLoopConfig(interface="sim", enable_waist=True, domain_id=42),
        factories=factories,
    )

    def fail_service_close():
        events.append("service.close.failed")
        raise RuntimeError("service close failed")

    runtime.components.robot_config_server.close = fail_service_close
    with pytest.raises(RuntimeError, match="service close failed"):
        runtime.close()
    assert events[-5:] == [
        "dispatcher.stop", "service.close.failed", "policy.close",
        "environment.close", "manager.shutdown",
    ]
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
from dataclasses import dataclass

import hashlib
import json
from pathlib import Path
import re
import subprocess

import onnxruntime as ort


MODEL_CONTRACT_SCHEMA = "decoupled_wbc.g1-model-contract.v1"
URDF_RELATIVE_PATH = (
    "control/robot_model/model_data/g1/g1_29dof_with_hand.urdf"
)
ONNX_ROOT = Path("sim2mujoco/resources/robots/g1")


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


def inspect_git_identity(repository_root):
    root = Path(repository_root)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git", "-C", str(root), "status", "--porcelain",
            "--untracked-files=all",
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("WBC Git commit is not a full lowercase SHA")
    if status:
        raise RuntimeError("WBC working tree is not clean")
    return GitIdentity(commit=commit, working_tree_clean=True)


def _normalized_shape(shape):
    return [dimension if type(dimension) is int else "dynamic" for dimension in shape]


def inspect_onnx_signature(path):
    session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("WBC ONNX must have exactly one input and one output")

    def tensor(node, expected_size):
        shape = _normalized_shape(node.shape)
        if not shape or shape[-1] != expected_size:
            raise ValueError(f"WBC ONNX feature size must be {expected_size}")
        return {
            "name": str(node.name), "shape": shape,
            "feature_size": expected_size,
        }

    return {
        "input": tensor(inputs[0], 516),
        "output": tensor(outputs[0], 15),
    }


def _validated_tensor_signature(value, expected_size):
    if type(value) is not dict or set(value) != {
        "name", "shape", "feature_size",
    }:
        raise ValueError("ONNX tensor signature key set")
    if type(value["name"]) is not str or not value["name"]:
        raise ValueError("ONNX tensor name")
    if (
        type(value["shape"]) is not list or not value["shape"]
        or any(type(dimension) not in (str, int) for dimension in value["shape"])
        or value["shape"][-1] != expected_size
        or type(value["feature_size"]) is not int
        or value["feature_size"] != expected_size
    ):
        raise ValueError(f"ONNX tensor feature size must be {expected_size}")
    return {
        "name": value["name"], "shape": list(value["shape"]),
        "feature_size": expected_size,
    }


def build_model_contract(
    robot_model, config, repository_root, *, git_identity=None,
    onnx_inspector=inspect_onnx_signature,
):
    root = Path(repository_root).resolve()
    identity = git_identity or inspect_git_identity(root)
    if (
        re.fullmatch(r"[0-9a-f]{40}", identity.commit) is None
        or identity.working_tree_clean is not True
    ):
        raise ValueError("invalid clean WBC Git identity")
    names = tuple(robot_model.joint_names)
    indices = tuple(robot_model.dof_index(name) for name in names)
    if len(names) != 43 or len(set(names)) != 43 or indices != tuple(range(43)):
        raise ValueError("G1 model must expose 43 unique ordered joints")
    lower = [float(robot_model.lower_joint_limits[index]) for index in indices]
    upper = [float(robot_model.upper_joint_limits[index]) for index in indices]
    if any(not low < high for low, high in zip(lower, upper, strict=True)):
        raise ValueError("G1 effective position limits are invalid")
    upper_indices = tuple(robot_model.get_joint_group_indices("upper_body"))
    upper_names = [names[index] for index in upper_indices]
    if len(upper_names) != 31 or len(set(upper_names)) != 31:
        raise ValueError("G1 upper body must expose 31 unique joints")

    urdf = root / URDF_RELATIVE_PATH
    model_paths = tuple(config.wbc_model_path.split(","))
    if model_paths != (
        "policy/GR00T-WholeBodyControl-Balance.onnx",
        "policy/GR00T-WholeBodyControl-Walk.onnx",
    ):
        raise ValueError("unexpected WBC ONNX model paths")
    onnx_models = []
    for role, relative in zip(("balance", "walk"), model_paths, strict=True):
        model_path = root / ONNX_ROOT / relative
        signature = onnx_inspector(model_path)
        if type(signature) is not dict or set(signature) != {"input", "output"}:
            raise ValueError("ONNX inspector key set")
        onnx_models.append({
            "role": role,
            "relative_path": (ONNX_ROOT / relative).as_posix(),
            "sha256": sha256_file(model_path),
            "input": _validated_tensor_signature(signature["input"], 516),
            "output": _validated_tensor_signature(signature["output"], 15),
        })
    return {
        "schema": MODEL_CONTRACT_SCHEMA,
        "git": {
            "commit": identity.commit,
            "working_tree_clean": identity.working_tree_clean,
        },
        "robot_model": {
            "name": "g1_29dof_with_hand",
            "joint_names": list(names),
            "lower_position_limits": lower,
            "upper_position_limits": upper,
            "upper_body_joint_names": upper_names,
        },
        "urdf": {
            "relative_path": URDF_RELATIVE_PATH,
            "sha256": sha256_file(urdf),
        },
        "onnx_models": onnx_models,
    }


def validate_expected_model_contract(actual, expected):
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError("connected WBC model contract mismatch")


def build_model_contract_payload(config, robot_model):
    repository_root = Path(__file__).resolve().parents[2]
    contract = build_model_contract(robot_model, config, repository_root)
    payload = config.to_dict()
    payload["model_contract"] = contract
    payload["model_contract_sha256"] = digest_model_contract(contract)
    return payload
```

The injected Git identity and ONNX inspector are test seams only; production uses the clean nested checkout and CPU ONNX Runtime paths shown above. Every emitted value is JSON-native, and dynamic ONNX dimensions normalize to the literal string `"dynamic"`.

Complete this step through the following bounded subactions, running the relevant single test each time:

- [ ] Add canonical JSON, SHA-256, and exact recursive type validation.
- [ ] Add injected/production Git identity inspection.
- [ ] Add ordered model name/joint/limit/upper-body extraction.
- [ ] Add URDF relative-path/hash identity.
- [ ] Add one ONNX tensor-signature inspector, then parameterize it for balance/walk.
- [ ] Add full expected-contract comparison and payload digest assembly.

- [ ] **Step 5: Add the injectable attested-component constructor**

Extend the existing import header with the two imports at the start of this block, then add the complete constructor and cleanup helper to `run_g1_control_loop.py`. Keep each factory call on its own line so a construction exception prevents every later call:

```python
from dataclasses import dataclass

from decoupled_wbc.control.main.model_contract import (
    build_model_contract_payload,
)


@dataclass(frozen=True)
class AttestedComponents:
    environment: object
    robot_model: object
    wbc_policy: object
    service_payload: dict[str, object]
    robot_config_server: object


class ProductionFactories:
    def loop_manager(self, backend, **kwargs):
        return create_loop_manager(backend, **kwargs)

    def robot_model(self, **kwargs):
        return instantiate_g1_robot_model(**kwargs)

    def environment(self, **kwargs):
        return G1Env(**kwargs)

    def policy(self, *args, **kwargs):
        return get_wbc_policy(*args, **kwargs)

    def model_contract_payload(self, config, model):
        return build_model_contract_payload(config, model)

    def service_server(self, backend, topic, payload):
        return create_service_server(backend, topic, payload)

    def publisher(self, backend, topic):
        return create_publisher(backend, topic)

    def telemetry(self, **kwargs):
        return Telemetry(**kwargs)

    def keyboard_listener(self):
        return KeyboardListenerPublisher()

    def keyboard_estop(self):
        return KeyboardEStop()

    def dispatcher(self, kind):
        if kind == "raw":
            return KeyboardDispatcher()
        if kind == "ros":
            return ROSKeyboardDispatcher()
        raise ValueError(
            f"Invalid keyboard dispatcher: {kind}, use 'raw' or 'ros'"
        )

    def subscriber(self, backend, topic, **kwargs):
        return create_subscriber(backend, topic, **kwargs)


def _build_attested_components(config, backend, factories=None):
    factories = factories or ProductionFactories()
    created = []
    try:
        wbc_config = config.load_wbc_yaml()
        waist_location = (
            "lower_and_upper_body" if config.enable_waist else "lower_body"
        )
        robot_model = factories.robot_model(
            waist_location=waist_location,
            high_elbow_pose=config.high_elbow_pose,
        )
        environment = factories.environment(
            env_name=config.env_name,
            robot_model=robot_model,
            config=wbc_config,
            wbc_version=config.wbc_version,
            messaging_backend=backend,
        )
        created.append(("environment", environment))
        if environment.sim and not config.sim_sync_mode:
            environment.start_simulator()
        wbc_policy = factories.policy(
            "g1", robot_model, wbc_config, config.upper_body_joint_speed
        )
        created.append(("policy", wbc_policy))
        service_payload = factories.model_contract_payload(config, robot_model)
        robot_config_server = factories.service_server(
            backend, ROBOT_CONFIG_TOPIC, service_payload
        )
        created.append(("service", robot_config_server))
        return AttestedComponents(
            environment=environment,
            robot_model=robot_model,
            wbc_policy=wbc_policy,
            service_payload=service_payload,
            robot_config_server=robot_config_server,
        )
    except Exception as error:
        rollback_errors = _rollback_attested_construction(created)
        if rollback_errors:
            raise RuntimeError(
                f"{error}; construction rollback: "
                + "; ".join(rollback_errors)
            ) from error
        raise


def _rollback_attested_construction(created):
    errors = []
    for name, resource in reversed(created):
        close = getattr(resource, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception as error:
            errors.append(f"{name}: {error}")
    return tuple(errors)


def _close_attested_components(components):
    errors = []
    for name, resource in (
        ("service", components.robot_config_server),
        ("policy", components.wbc_policy),
        ("environment", components.environment),
    ):
        operation = getattr(resource, "close", None)
        if not callable(operation):
            continue
        try:
            operation()
        except Exception as error:
            errors.append(f"{name}: {error}")
    if errors:
        raise RuntimeError("attested cleanup failed: " + "; ".join(errors))


class CleanupOwner:
    def __init__(self):
        self._operations = []
        self._closed = False

    def own(self, name, operation):
        if self._closed:
            raise RuntimeError("cannot add a resource to a closed owner")
        self._operations.append((name, operation))

    def close(self):
        if self._closed:
            return ()
        self._closed = True
        errors = []
        for name, operation in reversed(self._operations):
            try:
                operation()
            except Exception as error:
                errors.append(f"{name}: {error}")
        return tuple(errors)


@dataclass(frozen=True)
class WbcRuntime:
    manager: object
    components: AttestedComponents
    data_exp_pub: object
    lower_body_policy_status_pub: object
    joint_safety_status_pub: object
    telemetry: object
    keyboard_listener_pub: object
    keyboard_estop: object
    dispatcher: object
    rate: object
    upper_body_policy_subscriber: object
    owner: CleanupOwner

    def close(self):
        errors = self.owner.close()
        if errors:
            raise RuntimeError("WBC cleanup failed: " + "; ".join(errors))


def _build_owned_runtime(config, factories=None):
    factories = factories or ProductionFactories()
    backend = config.messaging_backend
    if backend == "zmq" and config.keyboard_dispatcher_type == "ros":
        print(
            "ZMQ backend: forcing keyboard_dispatcher_type='raw' "
            "(ROS dispatcher unavailable)"
        )
        config.keyboard_dispatcher_type = "raw"
    owner = CleanupOwner()
    try:
        manager = factories.loop_manager(
            backend, node_name=CONTROL_NODE_NAME
        )
        owner.own("manager", manager.shutdown)

        components = _build_attested_components(
            config, backend, factories=factories
        )
        owner.own(
            "attested_components",
            lambda: _close_attested_components(components),
        )

        data_exp_pub = factories.publisher(backend, STATE_TOPIC_NAME)
        lower_body_policy_status_pub = factories.publisher(
            backend, LOWER_BODY_POLICY_STATUS_TOPIC
        )
        joint_safety_status_pub = factories.publisher(
            backend, JOINT_SAFETY_STATUS_TOPIC
        )
        telemetry = factories.telemetry(window_size=100)
        keyboard_listener_pub = factories.keyboard_listener()
        keyboard_estop = factories.keyboard_estop()

        dispatcher = factories.dispatcher(config.keyboard_dispatcher_type)
        owner.own("dispatcher", dispatcher.stop)
        for listener in (
            components.environment,
            components.wbc_policy,
            keyboard_listener_pub,
            keyboard_estop,
        ):
            dispatcher.register(listener)
        dispatcher.start()

        rate = manager.create_rate(config.control_frequency)
        zmq_kw = {"host": config.zmq_host} if backend == "zmq" else {}
        upper_body_policy_subscriber = factories.subscriber(
            backend, CONTROL_GOAL_TOPIC, **zmq_kw
        )
        return WbcRuntime(
            manager=manager,
            components=components,
            data_exp_pub=data_exp_pub,
            lower_body_policy_status_pub=lower_body_policy_status_pub,
            joint_safety_status_pub=joint_safety_status_pub,
            telemetry=telemetry,
            keyboard_listener_pub=keyboard_listener_pub,
            keyboard_estop=keyboard_estop,
            dispatcher=dispatcher,
            rate=rate,
            upper_body_policy_subscriber=upper_body_policy_subscriber,
            owner=owner,
        )
    except Exception as error:
        rollback_errors = owner.close()
        if rollback_errors:
            raise RuntimeError(
                f"{error}; complete startup rollback: "
                + "; ".join(rollback_errors)
            ) from error
        raise
```

- [ ] **Step 6: Put the complete WBC startup lifecycle behind one tested owner**

Delete every resource construction statement from `main()`—including manager, service, publishers, telemetry, environment/simulator, policy, keyboard resources, dispatcher, rate, and subscriber. The first resource-owning statement must be the following call; nothing with a `close()`, `shutdown()`, `stop()`, child thread, ROS/DDS/ZMQ handle, or simulator may be constructed before it:

```python
runtime = _build_owned_runtime(config)
manager = runtime.manager
components = runtime.components
env = components.environment
robot_model = components.robot_model
wbc_policy = components.wbc_policy
data_exp_pub = runtime.data_exp_pub
lower_body_policy_status_pub = runtime.lower_body_policy_status_pub
joint_safety_status_pub = runtime.joint_safety_status_pub
telemetry = runtime.telemetry
dispatcher = runtime.dispatcher
rate = runtime.rate
upper_body_policy_subscriber = runtime.upper_body_policy_subscriber
```

Keep the existing control-loop body after these assignments. Replace its `finally` body with this single owner close:

```python
runtime.close()
```

`_build_owned_runtime()` registers manager shutdown immediately after manager creation, delegates component construction to the internally rollback-safe attested constructor, then registers the component cleanup before it constructs any publisher, dispatcher, or subscriber. It registers `dispatcher.stop` before the first dispatcher callback or thread is started. Any later failure closes all registered resources in reverse ownership order and continues through cleanup failures. `main()` therefore has no partial-startup path outside the owner. Run Ruff against the complete modified file, not these insertion fragments in isolation. The existing file already owns the imports used by `ProductionFactories`; Step 5 adds only the new `dataclass` and model-contract imports.

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
```

Export these plus the mapping functions from `src/simple/deploy/__init__.py`.

- [ ] **Step 5: Implement mapping exclusively by joint name**

Define the joint tuples and all three transforms literally:

```python
PSI0_STATE_JOINT_NAMES = (
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint", "left_hand_middle_0_joint",
    "left_hand_middle_1_joint", "left_hand_index_0_joint",
    "left_hand_index_1_joint", "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
PSI0_ACTION_JOINT_NAMES = PSI0_STATE_JOINT_NAMES
PSI0_WAIST_ACTION_NAMES = (
    "waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint",
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
    action = np.concatenate([
        np.asarray([named[name] for name in PSI0_ACTION_JOINT_NAMES], np.float32),
        np.asarray([named[name] for name in PSI0_WAIST_ACTION_NAMES], np.float32),
        height,
        navigation,
    ])
    return np.ascontiguousarray(action, dtype=np.float32)
```

Neither direction clips or reorders by numeric slice; every reordering is performed through the explicit joint-name dictionaries above.

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


@pytest.mark.parametrize("received_at", [9.899, 10.001])
def test_stale_or_future_state_never_replaces_last_valid(
    contract, received_at,
):
    prior = state(np.full(43, 0.25), received_at=9.99)
    candidate = state(np.full(43, 0.5), received_at=received_at)
    accepted, reason = accept_measured_state(
        prior, candidate, contract, now=10.0
    )
    assert accepted is prior
    assert reason == "measured state stale"


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
        (10.0, np.nan, 10.0, "camera receive time"),
        (10.0, "not-a-time", 10.0, "camera receive time"),
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
def validate_measured_state(sample, contract, now, tolerance=0.05):
    if type(sample) is not TimedRobotState:
        raise ValueError("measured state type")
    if type(now) not in (float, int) or not np.isfinite(now):
        raise ValueError("measured validation time")
    if type(sample.received_at) not in (float, int) or not np.isfinite(sample.received_at):
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


def validate_synchronized_snapshot(state, camera, now):
    if type(state) is not TimedRobotState:
        raise ValueError("state missing or wrong type")
    if type(camera) is not TimedCameraFrame:
        raise ValueError("camera missing or wrong type")
    if type(now) not in (float, int) or not np.isfinite(now):
        raise ValueError("snapshot validation time")
    if (
        type(state.received_at) not in (float, int)
        or not np.isfinite(state.received_at)
    ):
        raise ValueError("state receive time")
    if (
        type(camera.received_at) not in (float, int)
        or not np.isfinite(camera.received_at)
    ):
        raise ValueError("camera receive time")
    if (
        camera.producer_timestamp is not None
        and (
            type(camera.producer_timestamp) not in (float, int)
            or not np.isfinite(camera.producer_timestamp)
        )
    ):
        raise ValueError("camera producer time")
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
    return InputSnapshot(
        TimedRobotState(np.asarray(state.q, np.float32).copy(), state.received_at),
        TimedCameraFrame(
            np.ascontiguousarray(image).copy(), camera.received_at,
            camera.producer_timestamp,
        ),
    )
```

- [ ] **Step 6: Implement whole-suffix validation and one-tick slew limiting**

Implement and run the Step 2 cases:

```python
def _action_joint_limits(contract):
    index = {name: i for i, name in enumerate(contract.joint_names)}
    action_names = (
        *PSI0_ACTION_JOINT_NAMES,
        "waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint",
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
        [2.0] * 14 + [1.0] * 14 + [0.5] * 3 + [0.1]
        + [0.5] * 2 + [2.0] + [1.0],
        np.float32,
    )
    limited = previous + np.clip(requested - previous, -rates * dt, rates * dt)
    yaw_delta = (requested[35] - previous[35] + np.pi) % (2 * np.pi) - np.pi
    limited[35] = previous[35] + np.clip(yaw_delta, -dt, dt)
    limited[35] = (limited[35] + np.pi) % (2 * np.pi) - np.pi
    return limited.astype(np.float32, copy=False)
```

- [ ] **Step 7: Implement bounded hold selection and immutable capture**

Implement and run the Step 3 cases:

```python
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
    if last_valid_state is not None and 0.0 <= now - last_valid_state.received_at <= 0.10:
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
            if last_safe_goal is not None else 0.74
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


def make_policy_contract(**updates):
    return PolicyContract.from_dict(policy_payload(**updates))


def make_joint_contract():
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
from dataclasses import replace

import numpy as np
import pytest

from simple.deploy.psi0_simple_bridge import (
    PolicyContract,
    Psi0SimpleBridge,
    RtcRequest,
    RtcResult,
    validate_rtc_result,
)
from tests.psi0_bridge_testkit import (
    ImmediateInference,
    ManualClock,
    fresh_inputs,
    make_joint_contract,
    make_policy_contract,
    policy_payload,
    sentinel_actions,
)


def test_rtc_sentinel_has_exact_request_and_handoff_ticks():
    clock = ManualClock(99)
    contract = make_policy_contract()
    inference = ImmediateInference(clock, contract)
    bridge = Psi0SimpleBridge(contract, make_joint_contract(), inference, clock, start_tick=99)
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
        deadline_at=103 / 50.0,
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
    contract = make_policy_contract()
    request, result = valid_request_and_result(clock, contract)
    result = replace(result, actions=sentinel_actions(103, rows))
    with pytest.raises(ValueError, match="shape"):
        validate_rtc_result(request, result, contract, make_joint_contract())


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
    contract = make_policy_contract()
    request, result = valid_request_and_result(clock, contract)
    metadata = dict(result.metadata)
    metadata[key] = bad_value
    with pytest.raises(ValueError, match=key):
        validate_rtc_result(
            request, replace(result, metadata=metadata), contract,
            make_joint_contract(),
        )


def test_missing_extra_and_wrong_type_metadata_are_rejected(clock):
    contract = make_policy_contract()
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
                make_joint_contract(),
            )


def test_nonfinite_out_of_bounds_and_late_results_are_rejected(clock):
    contract = make_policy_contract()
    request, result = valid_request_and_result(clock, contract)
    for actions in (
        np.where(np.indices(result.actions.shape)[1] == 0, np.nan, result.actions),
        np.where(np.indices(result.actions.shape)[1] == 31, 0.1, result.actions),
    ):
        with pytest.raises(ValueError):
            validate_rtc_result(
                request, replace(result, actions=actions), contract,
                make_joint_contract(),
            )
    with pytest.raises(ValueError, match="deadline"):
        validate_rtc_result(
            request, replace(result, completed_at=result.completed_at + 1e-6),
            contract, make_joint_contract(),
        )
```

Add and run each test function separately before moving to lifecycle behavior.

- [ ] **Step 3: Write failing lifecycle and blocked-worker tests**

Create `tests/test_psi0_bridge_lifecycle.py` with an explicit blocking port and atomic-state assertions:

```python
from copy import deepcopy

import numpy as np
import pytest

from simple.deploy.psi0_simple_bridge import (
    ActivationRefused,
    BridgeState,
    Psi0SimpleBridge,
    RtcResult,
    TimedCameraFrame,
    TimedRobotState,
)
from tests.psi0_bridge_testkit import (
    ManualClock,
    fresh_inputs,
    make_joint_contract,
    make_policy_contract,
    sentinel_actions,
)


class BlockingInference:
    def __init__(self):
        self.requests = []
        self.result = None
        self.physical_busy = False

    @property
    def busy(self):
        return self.physical_busy or self.result is not None

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
        make_policy_contract(), make_joint_contract(), inference, clock,
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
    with pytest.raises(ActivationRefused, match="paused hold"):
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
    with pytest.raises(ActivationRefused, match="worker busy"):
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
    with pytest.raises(ActivationRefused, match="worker busy"):
        bridge.handle_toggle()
    assert bridge.state is BridgeState.FAULT
    assert len(inference.requests) == 1
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


@pytest.mark.parametrize("received_at", [np.nan, "not-a-time"])
def test_invalid_camera_receive_time_becomes_active_fault(received_at):
    inference = BlockingInference()
    bridge, clock = ready_bridge(inference)
    bridge.tick()
    clock.set_tick(100)
    bridge.update_inputs(*fresh_inputs(clock))
    bridge.activate()
    state, _ = fresh_inputs(clock)
    bad_camera = TimedCameraFrame(
        np.zeros((8, 8, 3), np.uint8), received_at, None
    )
    bridge.update_inputs(state, bad_camera)
    result = bridge.tick()
    assert bridge.state is BridgeState.FAULT
    assert bridge.fault_reason == "camera receive time"
    assert result.source_kind == "hold"


def test_paused_hold_is_captured_once_and_ignores_later_measurements():
    bridge, clock = ready_bridge(BlockingInference())
    first = bridge.tick()
    clock.set_tick(100)
    changed_q = np.zeros(43, np.float32)
    changed_q[0] = 0.1
    changed = TimedRobotState(changed_q, clock())
    bridge.update_inputs(changed, fresh_inputs(clock)[1])
    second = bridge.tick()
    assert first.source_kind == second.source_kind == "hold"
    np.testing.assert_array_equal(first.psi0_action, second.psi0_action)


def test_initial_activation_and_fault_rearm_each_increment_generation():
    inference = BlockingInference()
    bridge, clock = ready_bridge(inference)
    bridge.tick()
    clock.set_tick(100)
    bridge.update_inputs(*fresh_inputs(clock))
    initial = bridge.generation
    bridge.activate()
    assert bridge.generation == initial + 1
    bridge.enter_fault("test")
    assert bridge.generation == initial + 2
    inference.physical_busy = False
    bridge.tick()
    bridge.activate()
    assert bridge.generation == initial + 3


def test_malformed_metadata_type_latches_fault_instead_of_escaping():
    inference = BlockingInference()
    bridge, clock = ready_bridge(inference)
    bridge.tick()
    clock.set_tick(100)
    bridge.update_inputs(*fresh_inputs(clock))
    bridge.activate()
    request = inference.requests[0]
    inference.release(RtcResult(
        generation=request.generation,
        request_seq=request.request_seq,
        completed_at=clock(),
        actions=sentinel_actions(103, 5),
        metadata={
            "session_id": request.session_id, "request_seq": "0",
            "observation_tick": 100, "prediction_horizon": 8,
            "execution_horizon": 5, "rtc_delay_steps": 3,
            "first_action_tick": 103,
        },
    ))
    result = bridge.tick()
    assert bridge.state is BridgeState.FAULT
    assert "wrong type" in bridge.fault_reason
    assert result.source_kind == "hold"
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

Extend the existing import header (not the current insertion point) with these exact imports. They are all used by the completed Task 9 file:

```python
import re
import threading
from typing import Protocol
import uuid
```

Then add:

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
```

These protocol bodies define the injected boundary only; no production I/O belongs in the core file. Add the exact validator and run only the protocol/deadline cases:

```python
RTC_RESPONSE_FIELDS = (
    "session_id", "request_seq", "observation_tick", "prediction_horizon",
    "execution_horizon", "rtc_delay_steps", "first_action_tick",
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
```

Key order is not semantically significant; key set, exact Python JSON types, and values are.

- [ ] **Step 7: Add scheduler state, immutable status, and atomic exit transitions**

Add the following state types and class skeleton verbatim. `consume_goal` is the only effectful core callback; shadow supplies a preview recorder and sim-control supplies the already-preflighted ROS publisher. Returning false is a fail-closed sink rejection.

```python
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
        self, contract, joints, inference, clock, *, start_tick=0,
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
        self.command_history_rpyh = np.array(
            [0.0, 0.0, 0.0, 0.74], dtype=np.float32
        )
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
```

Import `threading` and `uuid` at module scope. Never mutate `InferencePort.busy` from the core. Run `test_exit_transition_*` before proceeding.

- [ ] Add immutable status/result types and mutable counters; run import tests.
- [ ] Add constructor/epoch initialization; run the paused-state assertion.
- [ ] Add `status()` and `_clear_policy_locked()`; run buffer snapshot assertions.
- [ ] Add the three public exit transitions; run only `test_exit_transition_*`.

- [ ] **Step 8: Implement paused holds, R0 activation, and physical-busy rearm**

Add these methods to the class. This fixes R0's observation tick, first-action tick, exact committed hold prefix, and deadline without a separate reset schema:

```python
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
                snapshot.state.q, self.joints.joint_names,
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
```

Run the startup/rearm tests. Do not add successor scheduling in this checkbox.

- [ ] Add input acceptance and synchronized snapshot retrieval.
- [ ] Add exact request construction/deadline calculation with an injected recorder.
- [ ] Add consumed hold/history ownership and run startup-hold tests.
- [ ] Add R0 activation/toggle branches and run physical-busy rearm tests.

- [ ] **Step 9: Implement tick order, successor scheduling, and history ownership**

Add the following complete transition methods. No request appends behind a remaining queue: each accepted suffix is installed by global tick, and every successor observes the state at its own `r` while committing the already scheduled `r:r+d` prefix.

```python
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
            actions = validate_rtc_result(
                request, result, self.contract, self.joints
            )
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
        for tick in range(
            request_tick, request_tick + self.contract.rtc_delay_steps
        ):
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
```

`validate_rtc_result()` returns a copied `(s,36)` array after exact metadata, whole-suffix bound, and `result.completed_at <= request.deadline_at` checks. It never installs state. Return immutable `TickResult` after each consumed command and advance history only in `_consume_locked()`. Run the P=8/s=5/d=3 sentinel, then the remaining lifecycle tests.

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
    BoundedWbcConfigClient, GoalOwnershipGuard, PreflightError,
    compare_policy_contracts, establish_goal_ownership, run_preflight,
    validate_connected_wbc, validate_then_create_publisher,
)
from simple.deploy.psi0_simple_bridge import BridgeMode, PolicyContract
from tests.psi0_bridge_testkit import ManualClock, make_joint_contract, policy_payload


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
    assert result.runtime_policy_contract == PolicyContract.from_dict(
        policy_payload()
    )
    assert result.policy_mismatched_fields == (
        "local policy contract unavailable",
    )


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
    payload["model_contract_sha256"] = digest_model_contract(
        payload["model_contract"]
    )
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
        "config.env_type", "config.interface", "model_contract.git.commit",
    )
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
            payload, valid_model_contract(), "1" * 40, 42,
            publisher_factory=lambda: calls.append("publisher"),
        )
    assert calls == []


def test_stale_transmitted_model_digest_fails_before_field_comparison():
    payload = copy.deepcopy(valid_wbc_payload())
    set_path(payload, ("model_contract", "robot_model", "name"), "wrong")
    calls = []
    with pytest.raises(PreflightError, match="digest"):
        validate_then_create_publisher(
            payload, valid_model_contract(), "1" * 40, 42,
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
            payload, valid_model_contract(), "1" * 40, 42,
            publisher_factory=lambda: calls.append("publisher"),
        )
    assert calls == []


def test_transmitted_digest_value_mutation_reports_digest():
    payload = copy.deepcopy(valid_wbc_payload())
    payload["model_contract_sha256"] = "0" * 64
    with pytest.raises(PreflightError, match="digest"):
        validate_then_create_publisher(
            payload, valid_model_contract(), "1" * 40, 42,
            publisher_factory=lambda: None,
        )


def test_valid_wbc_validation_then_factory_calls_publisher_once():
    calls = []
    validated, publisher = validate_then_create_publisher(
        valid_wbc_payload(), valid_model_contract(), "1" * 40, 42,
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


def test_shadow_goal_counts_are_checked_during_and_at_end_of_lifetime():
    graph = FakeGraph(2, 1)
    guard = GoalOwnershipGuard(BridgeMode.SHADOW, graph, (2, 1))
    guard.check()
    graph.publishers = 3
    with pytest.raises(PreflightError, match="counts changed"):
        guard.check()
    with pytest.raises(PreflightError, match="counts changed"):
        guard.close()
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

Add these complete definitions to `scripts/psi0_simple_real_bridge.py`:

```python
import argparse
import base64
from dataclasses import dataclass
import ipaddress
import json
import os
from pathlib import Path
import queue
import re
import select
import subprocess
import sys
import termios
import threading
import time
import tty
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
    build_model_contract, digest_model_contract,
)
from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from decoupled_wbc.control.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from decoupled_wbc.control.sensor.sensor_server import ImageMessageSchema
from simple.baselines.client import HttpActionClient
from simple.deploy.psi0_simple_bridge import (
    ActivationRefused, BridgeMetrics, BridgeMode, BridgeState, JointContract,
    PolicyContract, Psi0SimpleBridge, RtcResult, TickResult,
    TimedCameraFrame, TimedRobotState, accept_measured_state,
    validate_synchronized_snapshot,
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
        "--mode", required=True,
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
        "--camera-color-order", choices=("rgb", "bgr"), default="rgb",
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
```

`main()` immediately constructs `BridgeMode(args.mode)`. The parser has no branch for `real`, `real-control`, a robot NIC, or a low-level DDS topic, and `build_policy_client()` is the only production construction site for the bridge HTTP client.

- [ ] **Step 6: Implement strict local/server policy comparison**

Implement this exact wire-order comparison and run only the Step 1 mutation matrix:

```python
POLICY_FIELD_MAP = (
    ("schema", "schema"), ("test_only", "test_only"),
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
    ("rtc_enabled", "rtc_enabled"), ("rtc_endpoint", "endpoint"),
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
        wire_name for wire_name, attribute in POLICY_FIELD_MAP
        if getattr(local, attribute) != getattr(server, attribute)
    )
    if (local.test_only or server.test_only) and wbc_env_type != "sim":
        raise PreflightError("test-only policy contract requires sim WBC")
    if mismatches and mode is not BridgeMode.SHADOW:
        raise PreflightError(
            "policy contract mismatch: " + ",".join(mismatches)
        )
    certified = mode is BridgeMode.SIM_CONTROL and not mismatches
    return PolicyPreflightResult(certified, mismatches)
```

The tuple is the canonical wire order; never use set iteration for error order.

- [ ] **Step 7: Implement the bounded WBC config client and connected-model validator**

Implement `BoundedWbcConfigClient`, local expected-contract construction, root-gitlink lookup, digest recomputation, and exact full comparison. Use this complete deadline wrapper; the injectable form is what the Step 2 timeout test invokes:

```python
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
                    raise TimeoutError(f"WBC config service timed out after {timeout_s:.1f}s")
                if self._wait_once(min(0.05, remaining)):
                    break
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise TimeoutError(f"WBC config service timed out after {timeout_s:.1f}s")
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
```

The temporary client uses its own rclpy context, so its unconditional `context.shutdown()` cannot stop the runtime context created later. The complete `_git_stdout()` and `build_local_wbc_identity()` definitions in Step 10 own the root-gitlink lookup; there is no bare subprocess fragment in this step.

Use this exact digest-first validator; nested field tests recompute the digest specifically to reach `_first_difference()`:

```python
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
            difference = _first_difference(
                actual[key], expected[key], f"{path}.{key}"
            )
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
    if set(contract) != {
        "schema", "git", "robot_model", "urdf", "onnx_models",
    } or contract.get("schema") != "decoupled_wbc.g1-model-contract.v1":
        raise PreflightError("connected WBC model contract schema")
    git = contract.get("git")
    if (
        type(git) is not dict or set(git) != {"commit", "working_tree_clean"}
        or type(git.get("commit")) is not str
        or re.fullmatch(r"[0-9a-f]{40}", git["commit"]) is None
        or type(git.get("working_tree_clean")) is not bool
    ):
        raise PreflightError("connected WBC Git identity schema")
    urdf = contract.get("urdf")
    if (
        type(urdf) is not dict or set(urdf) != {"relative_path", "sha256"}
        or type(urdf.get("relative_path")) is not str
        or type(urdf.get("sha256")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", urdf["sha256"]) is None
    ):
        raise PreflightError("connected WBC URDF identity schema")
    onnx_models = contract.get("onnx_models")
    if type(onnx_models) is not list or len(onnx_models) != 2:
        raise PreflightError("connected WBC ONNX identity schema")
    for model_entry, role in zip(onnx_models, ("balance", "walk"), strict=True):
        if (
            type(model_entry) is not dict
            or set(model_entry) != {
                "role", "relative_path", "sha256", "input", "output",
            }
            or model_entry.get("role") != role
            or type(model_entry.get("relative_path")) is not str
            or type(model_entry.get("sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", model_entry["sha256"]) is None
        ):
            raise PreflightError("connected WBC ONNX identity schema")
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
                raise PreflightError("connected WBC ONNX tensor schema")
    try:
        model = contract["robot_model"]
        if (
            type(model) is not dict
            or set(model) != {
                "name", "joint_names", "lower_position_limits",
                "upper_position_limits", "upper_body_joint_names",
            }
            or type(model.get("name")) is not str
            or any(
                type(model.get(key)) is not list
                for key in (
                    "joint_names", "lower_position_limits",
                    "upper_position_limits", "upper_body_joint_names",
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
        len(names) != 43 or len(set(names)) != 43
        or len(upper) != 31 or len(set(upper)) != 31
        or any(type(name) is not str for name in names)
        or any(type(name) is not str for name in upper)
        or any(name not in names for name in upper)
    ):
        raise PreflightError("connected WBC joint_names/upper_body dimensions")
    if (
        lower.shape != (43,) or high.shape != (43,)
        or not np.isfinite(lower).all() or not np.isfinite(high).all()
        or not np.all(lower < high)
    ):
        raise PreflightError("connected WBC effective limits")
    return ValidatedWbc(JointContract(names, upper, lower, high)), contract


def validate_connected_wbc(
    *, actual, expected_contract, expected_gitlink_sha, required_domain_id,
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
    *, actual, expected_contract, expected_gitlink_sha, required_domain_id,
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
    payload, expected_contract, expected_gitlink_sha, required_domain_id,
    *, publisher_factory,
):
    validated = validate_connected_wbc(
        actual=payload,
        expected_contract=expected_contract,
        expected_gitlink_sha=expected_gitlink_sha,
        required_domain_id=required_domain_id,
    )
    return validated, publisher_factory()
```

Run the Step 2 tests before adding publishers or camera code.

Complete in this order:

- [ ] Implement the 50 ms polling/three-second deadline and unconditional destruction.
- [ ] Parse and type-check the exact connected WBC payload/digest.
- [ ] Resolve and validate the root gitlink plus clean local nested identity.
- [ ] Build the local expected model contract and compare full canonical JSON.

- [ ] **Step 8: Implement graph ownership and the goal publisher adapter**

Add these exact ROS runtime/publisher boundaries. They use native `ByteMultiArray` plus the repository's `msgpack_numpy` wire format, but do not use its global `ROSManager`, which cannot provide isolated ownership or bounded destruction:

```python
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
            target=self.executor.spin, name="psi0-ros-executor", daemon=True
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
            "target_upper_body_pose", "base_height_command", "navigate_cmd",
            "timestamp", "target_time",
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
    if (
        len(before) != 2 or any(type(value) is not int or value < 0 for value in before)
    ):
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
```

Run the graph tests and prove rejected preflight never invokes the publisher factory.

- [ ] Add the isolated ROS context/executor and a close-only lifecycle test.
- [ ] Add exact five-field goal serialization with a fake rclpy publisher.
- [ ] Add before/after graph count enforcement and run all ownership cases.

- [ ] **Step 9: Implement state and camera adapters with bounded close**

Create concrete `RosStateSource` and `ComposedCameraReader` classes in the script. Compose all adapters with:

```python
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
    *, mode, preflight_result, publisher_factory, test_dependencies,
):
    if preflight_result.publisher_required != (mode is BridgeMode.SIM_CONTROL):
        raise PreflightError("preflight publisher requirement does not match mode")
    graph = test_dependencies.graph()
    if (
        mode is BridgeMode.SHADOW
        and tuple(graph.counts(CONTROL_GOAL_TOPIC))
        != tuple(preflight_result.goal_counts_at_preflight)
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
```

Production creates one `RosRuntime`, then passes factories closing over it; after construction, it reconstructs `RuntimeAdapters(adapters.state_source, adapters.camera_reader, adapters.goal_publisher, runtime)` exactly as shown in `run_bridge()` below. `RuntimeAdapters.close()` is not added: the shutdown coordinator owns and closes each field once in its defined order.

`RosStateSource.poll()` returns `TimedRobotState | None`. Add this complete implementation, including the callback-time receive timestamp rather than a poll-time timestamp:

```python
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
    if key not in schema.images or key not in schema.timestamps:
        raise KeyError(key)
    image = np.asarray(schema.images[key])
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("camera image must be HxWx3 uint8")
    if color_order == "bgr":
        image = image[..., ::-1]
    elif color_order != "rgb":
        raise ValueError("camera color order must be rgb or bgr")
    return TimedCameraFrame(
        np.ascontiguousarray(image), float(received_at),
        float(schema.timestamps[key]),
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
            target=self._run, name="psi0-camera-reader", daemon=True
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
                    serialized, key=self._key, color_order=self._color_order,
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
```

The ZMQ socket/context are created, used, and closed only in `_run()`. Every adapter exposes `close()`. Run the camera tests after this checkbox.

Complete in four bounded subactions:

- [ ] Add state polling only.
- [ ] Add pure codec/color conversion only.
- [ ] Add camera reader-thread polling only.
- [ ] Add reader-thread socket close/join only.

- [ ] **Step 10: Implement preflight sequencing before publisher construction**

Implement the preflight as a pure orchestrator with the exact call order below. Construction of state/camera/worker/keyboard happens only after this function returns; publisher construction appears exactly once and last:

```python
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
    *, mode, local_policy, server_policy, wbc_payload, graph,
    expected_model_contract, expected_gitlink_sha, required_domain_id=42,
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
    for label, payload in (
        ("local", local_policy), ("server", server_policy)
    ):
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
                parsed["local"], parsed["server"], mode,
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
            policy_result.policy_certified
            if mode is BridgeMode.SIM_CONTROL else False
        ),
        policy_mismatched_fields=policy_result.mismatched_fields,
        wbc_mismatched_fields=wbc_mismatches,
        joint_contract=validated_wbc.joint_contract,
        publisher_required=(mode is BridgeMode.SIM_CONTROL),
        goal_counts_at_preflight=goal_counts,
        runtime_policy_contract=runtime_contract,
    )
```

The real `run_bridge()` calls: parser/mode validation; `create_ros_wbc_config_client(...).get_config(3.0)`; local contract read; HTTP server contract fetch; `run_preflight(...)`; and only then `build_runtime_adapters(...)`. `run_preflight()` performs the read-only `0 publishers/1 subscription` check and never receives or invokes a publisher factory. `build_runtime_adapters()` is the sole caller of `establish_goal_ownership()` and creates the publisher exactly once. Record each completed phase in metrics.

Construct the local WBC identity with this exact helper; it proves the root gitlink, nested HEAD, nested cleanliness, and locally hashed model all describe one checkout:

```python
@dataclass(frozen=True)
class LocalWbcIdentity:
    root_gitlink_sha: str
    model_contract: dict[str, object]


def _git_stdout(arguments, cwd):
    return subprocess.run(
        arguments, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def build_local_wbc_identity(repository_root):
    root = Path(repository_root).resolve()
    nested = root / "third_party/decoupled_wbc"
    gitlink = _git_stdout(
        ["git", "rev-parse", "HEAD:third_party/decoupled_wbc"], root
    )
    nested_head = _git_stdout(["git", "rev-parse", "HEAD"], nested)
    nested_status = _git_stdout(
        ["git", "status", "--porcelain", "--untracked-files=all"], nested
    )
    if nested_head != gitlink:
        raise PreflightError("nested HEAD differs from root gitlink")
    if nested_status:
        raise PreflightError("nested WBC checkout is not clean")
    config = ControlLoopConfig(
        interface="sim", simulator="mujoco", messaging_backend="ros2",
        enable_waist=True, with_hands=True, domain_id=42,
    )
    model = instantiate_g1_robot_model(
        waist_location="lower_and_upper_body",
        high_elbow_pose=config.high_elbow_pose,
    )
    contract = build_model_contract(
        robot_model=model, config=config,
        repository_root=Path(decoupled_wbc.__file__).resolve().parent,
    )
    if contract["git"] != {
        "commit": gitlink, "working_tree_clean": True
    }:
        raise PreflightError("local WBC model contract Git identity")
    return LocalWbcIdentity(gitlink, contract)
```

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
import json
import os
import pty
import termios
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.psi0_simple_real_bridge import (
    ConnectionEvidenceRecorder, DisabledInferenceWorker, FiftyHzLoop,
    HttpInferenceWorker, JsonlMetrics, LocalKeyboard,
    ObservationOnlyShadowBridge, PreflightError, handle_keyboard_events,
    count_real_interface_connections,
)
from simple.baselines.client import RtcActionResponse
from simple.deploy.psi0_simple_bridge import (
    ActivationRefused, BridgeMode, BridgeState, PolicyContract, RtcRequest,
    TimedCameraFrame, TimedRobotState,
)
from tests.psi0_bridge_testkit import (
    ManualClock, make_joint_contract, policy_payload,
)


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
        deadline_at=106 / 50.0,
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
    worker = HttpInferenceWorker(
        client=client, clock=ManualClock(100),
        contract=PolicyContract.from_dict(policy_payload(
            prediction_horizon=30, execution_horizon=24,
            rtc_delay_steps=6, rtc_training_max_delay=7,
        )),
    )
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


def test_queued_result_keeps_worker_busy_until_main_thread_drains_it():
    class ImmediateHttpClient:
        def query_rtc_action(self, *args, **kwargs):
            return valid_http_rtc_response()

    result_recorded = threading.Event()

    def record_event(event, **_fields):
        if event == "result":
            result_recorded.set()

    worker = HttpInferenceWorker(
        client=ImmediateHttpClient(), clock=ManualClock(100),
        contract=PolicyContract.from_dict(policy_payload(
            prediction_horizon=30, execution_horizon=24,
            rtc_delay_steps=6, rtc_training_max_delay=7,
        )),
        event_sink=record_event,
    )
    worker.submit(valid_runtime_request())
    assert result_recorded.wait(0.2)
    assert worker.busy is True
    assert worker.poll() is not None
    assert worker.busy is False
    worker.close(timeout_s=0.5)


def test_every_shadow_metric_and_preview_is_uncertified(tmp_path):
    path = tmp_path / "shadow.jsonl"
    bridge = SimpleNamespace(
        state=BridgeState.PAUSED, generation=3,
    )
    metrics = JsonlMetrics(
        path, mode=BridgeMode.SHADOW, policy_certified=False,
        clock=lambda: 1.25,
    )
    metrics.write_event("request", request_seq=7)
    metrics.write_event("result", request_seq=7)
    metrics.write_event("preview", navigate_cmd=[0.0] * 4)
    metrics.write("tick", bridge, published=False, previewed=True)
    with pytest.raises(ValueError, match="owned by JsonlMetrics"):
        metrics.write_event("override", policy_certified=True)
    metrics.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "request", "result", "preview", "tick",
    ]
    assert all(record["policy_certified"] is False for record in records)


def test_rejected_activation_is_reported_and_next_local_p_is_accepted():
    class ToggleBridge:
        state = BridgeState.PAUSED

        def __init__(self):
            self.calls = 0

        def handle_toggle(self):
            self.calls += 1
            if self.calls == 1:
                raise ActivationRefused("physical worker busy")
            self.state = BridgeState.ACTIVE

    class Keyboard:
        def poll(self, _timeout_s):
            return ("p",)

    class Metrics:
        def __init__(self):
            self.events = []

        def write_event(self, event, **fields):
            self.events.append((event, fields))

    bridge = ToggleBridge()
    metrics = Metrics()
    handle_keyboard_events(bridge, Keyboard(), metrics)
    assert bridge.state is BridgeState.PAUSED
    assert metrics.events == [(
        "activation_refused",
        {"state": "paused", "reason": "physical worker busy"},
    )]
    handle_keyboard_events(bridge, Keyboard(), metrics)
    assert bridge.state is BridgeState.ACTIVE
    assert bridge.calls == 2


def test_missing_shadow_contract_disables_inference_without_exiting():
    class Metrics:
        def __init__(self):
            self.events = []

        def write_event(self, event, **fields):
            self.events.append((event, fields))

    worker = DisabledInferenceWorker()
    bridge = ObservationOnlyShadowBridge(
        worker, make_joint_contract(), ManualClock(100), start_tick=100
    )
    metrics = Metrics()
    keyboard = SimpleNamespace(poll=lambda _timeout_s: ("p",))
    handle_keyboard_events(bridge, keyboard, metrics)
    result = bridge.tick()
    assert bridge.state is BridgeState.PAUSED
    assert result.goal is None
    assert worker.busy is False
    assert metrics.events[0][0] == "activation_refused"


def test_contractless_shadow_validates_state_camera_freshness_and_skew():
    clock = ManualClock(500)
    worker = DisabledInferenceWorker()
    bridge = ObservationOnlyShadowBridge(
        worker, make_joint_contract(), clock, start_tick=500
    )
    state = TimedRobotState(np.zeros(43, np.float32), clock())
    camera = TimedCameraFrame(
        np.zeros((8, 8, 3), np.uint8), clock(), None
    )
    bridge.update_inputs(state, camera)
    assert bridge.observation_valid is True
    assert bridge.input_error is None
    accepted_state = bridge.last_valid_state
    accepted_snapshot = bridge.last_snapshot

    bad_shape = TimedRobotState(np.zeros(42, np.float32), clock())
    bridge.update_inputs(bad_shape, camera)
    assert bridge.observation_valid is False
    assert bridge.input_error == "measured state shape"
    assert bridge.last_valid_state is accepted_state
    assert bridge.last_snapshot is accepted_snapshot

    clock.set_tick(510)
    fresh_state = TimedRobotState(np.zeros(43, np.float32), clock())
    skewed_camera = TimedCameraFrame(
        np.zeros((8, 8, 3), np.uint8), 10.0, None
    )
    bridge.update_inputs(fresh_state, skewed_camera)
    assert bridge.observation_valid is False
    assert bridge.input_error == "receive-time skew"
    assert bridge.last_snapshot is accepted_snapshot

    invalid_time_camera = TimedCameraFrame(
        np.zeros((8, 8, 3), np.uint8), np.nan, None
    )
    bridge.update_inputs(fresh_state, invalid_time_camera)
    assert bridge.observation_valid is False
    assert bridge.input_error == "camera receive time"


def test_connection_evidence_requires_successful_runtime_observations():
    times = iter((1.0, 1.1, 1.2, 2.0, 2.1, 2.2))
    recorder = ConnectionEvidenceRecorder(clock=lambda: next(times))
    with pytest.raises(PreflightError, match="missing observed"):
        recorder.snapshot(required_components={"wbc", "camera", "policy"})
    recorder.observe_wbc_response({"env_type": "sim", "interface": "lo"})
    recorder.observe_camera_frame(
        "127.0.0.1",
        TimedCameraFrame(
            np.zeros((8, 8, 3), np.uint8), received_at=1.05,
            producer_timestamp=1.0,
        ),
    )
    recorder.observe_policy_contract("localhost", policy_payload())
    evidence = recorder.snapshot(
        required_components={"wbc", "camera", "policy"}
    )
    assert [record["component"] for record in evidence] == [
        "wbc", "camera", "policy",
    ]
    assert count_real_interface_connections(evidence) == 0
    assert all(type(record["observed_at"]) is float for record in evidence)
    with pytest.raises(PreflightError, match="duplicate"):
        recorder.observe_policy_contract("localhost", policy_payload())

    remote = ConnectionEvidenceRecorder(clock=lambda: next(times))
    remote.observe_wbc_response({"env_type": "sim", "interface": "lo"})
    remote.observe_camera_frame(
        "127.0.0.1",
        TimedCameraFrame(
            np.zeros((8, 8, 3), np.uint8), received_at=2.05,
            producer_timestamp=2.0,
        ),
    )
    remote.observe_policy_contract("192.0.2.10", policy_payload())
    evidence = remote.snapshot(
        required_components={"wbc", "camera", "policy"}
    )
    assert count_real_interface_connections(evidence) == 1
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

from scripts.psi0_simple_real_bridge import ShutdownCoordinator


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

Append the long-path assertion. The fixture uses the production `HttpInferenceWorker` and `HttpActionClient` against a local `/act-rtc-v1` handler. The handler records acceptance, withholds its response for 5.2 seconds (beyond the client's five-second timeout), and is fully joined before process exit:

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
    target = np.asarray(goals[0]["goal"]["target_upper_body_pose"])
    np.testing.assert_array_less(
        np.asarray(report["goal_lower_bounds"]) - 1e-7, target
    )
    np.testing.assert_array_less(
        target, np.asarray(report["goal_upper_bounds"]) + 1e-7
    )
    assert 0.20 <= goals[0]["goal"]["base_height_command"][0] <= 0.74
    deltas = np.diff([entry["scheduled_at"] for entry in goals])
    np.testing.assert_allclose(deltas, 0.02, rtol=0, atol=0.002)
    assert report["publish_attempts"] == 25
    assert report["publisher_closed_after_publish_count"] == 25
    assert report["publisher_closed"] is True
    assert report["camera_closed"] is True
    assert report["terminal_restored"] is True
    assert report["live_non_daemon_bridge_threads"] == []


class AdvancingClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)


class BudgetedCloser:
    def __init__(self, clock, consume=0.0, busy=False):
        self.clock = clock
        self.consume = consume
        self.busy = busy
        self.calls = []

    def close(self, timeout_s):
        self.calls.append((self.clock(), timeout_s))
        self.clock.sleep(min(self.consume, timeout_s))


class BudgetedSink(BudgetedCloser):
    def __init__(self, clock):
        super().__init__(clock)
        self.published = 0

    def publish(self, goal):
        self.published += 1
        return True


class ShutdownBridge:
    def __init__(self, hold):
        self.hold = hold

    def stop(self):
        return None

    def build_bounded_shutdown_hold(self):
        return self.hold


class FailingShutdownBridge(ShutdownBridge):
    def stop(self):
        raise RuntimeError("stop failed")

    def build_bounded_shutdown_hold(self):
        raise RuntimeError("hold failed")


def test_every_cleanup_timeout_shares_one_absolute_long_path_deadline():
    clock = AdvancingClock()
    sink = BudgetedSink(clock)
    camera = BudgetedCloser(clock, consume=0.5)
    worker = BudgetedCloser(clock, consume=5.5, busy=True)
    remaining_resources = [BudgetedCloser(clock) for _ in range(5)]
    coordinator = ShutdownCoordinator(
        bridge=ShutdownBridge(hold=object()), command_sink=sink,
        camera=camera, worker=worker, state_source=remaining_resources[0],
        keyboard=remaining_resources[1], ownership_guard=remaining_resources[2],
        ros_runtime=remaining_resources[3], metrics=remaining_resources[4],
        clock=clock, sleep=clock.sleep,
    )
    report = coordinator.close()
    assert sink.published == 25
    assert report.deadline_at == 6.5
    assert report.finished_at <= report.deadline_at
    for resource in [sink, camera, worker, *remaining_resources]:
        for called_at, timeout_s in resource.calls:
            assert 0.0 <= timeout_s <= max(
                0.0, report.deadline_at - called_at
            )


def test_no_state_cleanup_uses_one_half_second_deadline():
    clock = AdvancingClock()
    sink = BudgetedSink(clock)
    resources = [BudgetedCloser(clock) for _ in range(7)]
    coordinator = ShutdownCoordinator(
        bridge=ShutdownBridge(hold=None), command_sink=sink,
        camera=resources[0], worker=resources[1], state_source=resources[2],
        keyboard=resources[3], ownership_guard=resources[4],
        ros_runtime=resources[5], metrics=resources[6], clock=clock,
        sleep=clock.sleep,
    )
    report = coordinator.close()
    assert sink.published == 0
    assert report.deadline_at == 0.5
    assert report.finished_at <= 0.5


def test_exhausted_deadline_still_attempts_terminal_and_all_cleanup():
    clock = AdvancingClock()
    sink = BudgetedSink(clock)
    camera = BudgetedCloser(clock, consume=0.5)
    worker = BudgetedCloser(clock)
    state = BudgetedCloser(clock)
    keyboard = BudgetedCloser(clock)
    ownership = BudgetedCloser(clock)
    ros = BudgetedCloser(clock)
    metrics = BudgetedCloser(clock)
    coordinator = ShutdownCoordinator(
        bridge=ShutdownBridge(hold=None), command_sink=sink,
        camera=camera, worker=worker, state_source=state,
        keyboard=keyboard, ownership_guard=ownership,
        ros_runtime=ros, metrics=metrics, clock=clock, sleep=clock.sleep,
    )
    report = coordinator.close()
    assert camera.calls == [(0.0, 0.5)]
    for resource in (worker, state, keyboard, ownership, ros, metrics):
        assert resource.calls == [(0.5, 0.0)]
    assert any("deadline exhausted" in error for error in report.cleanup_errors)


def test_bridge_stop_failure_cannot_skip_terminal_or_metrics_cleanup():
    clock = AdvancingClock()
    resources = [BudgetedCloser(clock) for _ in range(7)]
    coordinator = ShutdownCoordinator(
        bridge=FailingShutdownBridge(hold=None),
        command_sink=BudgetedSink(clock), camera=resources[0],
        worker=resources[1], state_source=resources[2],
        keyboard=resources[3], ownership_guard=resources[4],
        ros_runtime=resources[5], metrics=resources[6],
        clock=clock, sleep=clock.sleep,
    )
    report = coordinator.close()
    assert all(len(resource.calls) == 1 for resource in resources)
    assert any(error == "bridge_stop: stop failed" for error in report.cleanup_errors)
    assert any(error == "bounded_hold: hold failed" for error in report.cleanup_errors)
```

`publish_attempts == 25` is the explicit no-26th assertion. The report is flushed only after publisher closure, so closure ordering is observable rather than inferred.

- [ ] **Step 4: Implement the inference worker**

`HttpInferenceWorker` owns one daemon thread and one `HttpActionClient(timeout=5.0)`. Add this complete implementation; `contract` and `instruction` are constructor arguments in production, while the runtime unit test may inject their defaults:

```python
@dataclass
class WorkerMetrics:
    requests_submitted: int = 0
    first_request_started_at: float | None = None


class HttpInferenceWorker:
    def __init__(
        self, *, client, clock, contract, instruction="instruction",
        event_sink=None,
    ):
        self.client = client
        self.clock = clock
        self.contract = contract
        self.instruction = instruction
        self._event_sink = event_sink or (lambda _event, **_fields: None)
        self._input = queue.Queue(maxsize=1)
        self._output = queue.Queue(maxsize=1)
        self._physical_busy = threading.Event()
        self._stopping = threading.Event()
        self.metrics = WorkerMetrics()
        self.thread = threading.Thread(
            target=self._run, name="psi0-http-worker", daemon=True
        )
        self.thread.start()

    @property
    def busy(self):
        return self._physical_busy.is_set() or not self._output.empty()

    def submit(self, request):
        if self._stopping.is_set():
            raise ActivationRefused("inference worker is stopping")
        if self.busy:
            raise ActivationRefused("inference worker busy")
        self._physical_busy.set()
        try:
            self._input.put_nowait(request)
        except queue.Full as error:
            self._physical_busy.clear()
            raise ActivationRefused("inference request queue is full") from error
        self.metrics.requests_submitted += 1
        self._event_sink(
            "request", generation=request.generation,
            request_seq=request.request_seq,
            observation_tick=request.observation_tick,
            committed_actions=request.committed_actions.tolist(),
        )

    def _serialize_history(self, request):
        history = {
            "session_id": request.session_id,
            "request_seq": request.request_seq,
            "observation_tick": request.observation_tick,
            "rtc_delay_steps": self.contract.rtc_delay_steps,
            "committed_actions": request.committed_actions.copy(),
        }
        if request.reset:
            history["reset"] = True
        return history

    def _run(self):
        while True:
            try:
                request = self._input.get(timeout=0.05)
            except queue.Empty:
                if self._stopping.is_set():
                    return
                continue
            if self._stopping.is_set():
                self._physical_busy.clear()
                return
            started_at = self.clock()
            if self.metrics.first_request_started_at is None:
                self.metrics.first_request_started_at = started_at
            try:
                response = self.client.query_rtc_action(
                    {self.contract.image_key: request.image},
                    self.instruction,
                    {"states": request.observation},
                    {},
                    history=self._serialize_history(request),
                    dataset="simple",
                )
                result = RtcResult(
                    generation=request.generation,
                    request_seq=request.request_seq,
                    completed_at=self.clock(),
                    actions=np.asarray(response.action, np.float32).copy(),
                    metadata=dict(response.metadata),
                )
            except Exception as error:
                result = RtcResult(
                    generation=request.generation,
                    request_seq=request.request_seq,
                    completed_at=self.clock(), actions=None, metadata=None,
                    error=f"{type(error).__name__}: {error}",
                )
            try:
                self._output.put_nowait(result)
            finally:
                self._physical_busy.clear()
            self._event_sink(
                "result", generation=result.generation,
                request_seq=result.request_seq,
                completed_at=result.completed_at, error=result.error,
            )
            if self._stopping.is_set():
                return

    def poll(self):
        try:
            return self._output.get_nowait()
        except queue.Empty:
            return None

    def close(self, timeout_s):
        self._stopping.set()
        self.thread.join(timeout_s)
        if self.thread.is_alive():
            raise RuntimeError("inference worker failed to stop within timeout")
        self._physical_busy.clear()
        for pending in (self._input, self._output):
            while True:
                try:
                    pending.get_nowait()
                except queue.Empty:
                    break
```

`WorkerMetrics` is a dataclass with `requests_submitted: int = 0` and `first_request_started_at: float | None = None`. `busy` is true while HTTP is running **or while a completed result remains undrained**, so a local rearm cannot race the main thread's result drain. Both production and tests pass the parsed contract explicitly. The worker never retries or starts a replacement thread. Reuse the exact Task 3 recording-session tests at this boundary to prove R0 carries reset and successors omit it.

Implement in bounded subactions:

- [ ] Add capacity-one queues/event and one daemon thread with no HTTP call.
- [ ] Add exact `RtcRequest` serialization and recording-session assertions.
- [ ] Add result/error conversion and `finally` busy clearing.
- [ ] Add bounded join without replacement/retry behavior.

- [ ] **Step 5: Implement local-keyboard and metrics ownership**

Use these exact implementations:

```python
class LocalKeyboard:
    ACCEPTED_KEYS = (b"p",)

    def __init__(self, fd):
        self.fd = fd
        self._original = None

    def __enter__(self):
        if not os.isatty(self.fd):
            raise RuntimeError("local keyboard requires a TTY")
        self._original = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def poll(self, timeout_s=0.0):
        accepted = []
        readable, _, _ = select.select([self.fd], [], [], timeout_s)
        while readable:
            value = os.read(self.fd, 1)
            if value in self.ACCEPTED_KEYS:
                accepted.append(value.decode("ascii"))
            readable, _, _ = select.select([self.fd], [], [], 0.0)
        return tuple(accepted)

    def close(self, timeout_s=0.0):
        if self._original is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._original)
            self._original = None

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class JsonlMetrics:
    def __init__(
        self, path, *, mode, policy_certified, clock=time.monotonic,
    ):
        self._file = Path(path).open("x", encoding="utf-8")
        self._mode = mode.value
        self._policy_certified = bool(policy_certified)
        self._clock = clock
        self._closed = False
        self._lock = threading.Lock()

    def write(self, event, bridge, *, published, **fields):
        if self._closed:
            raise RuntimeError("metrics writer is closed")
        if "policy_certified" in fields:
            raise ValueError("policy certification is owned by JsonlMetrics")
        payload = {
            "event": event, "mode": self._mode, "state": bridge.state.value,
            "generation": bridge.generation, "monotonic_s": self._clock(),
            "published": bool(published),
            "policy_certified": self._policy_certified, **fields,
        }
        with self._lock:
            self._file.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self._file.flush()

    def write_event(self, event, **fields):
        if self._closed:
            raise RuntimeError("metrics writer is closed")
        if "policy_certified" in fields:
            raise ValueError("policy certification is owned by JsonlMetrics")
        payload = {
            "event": event, "mode": self._mode,
            "monotonic_s": self._clock(),
            "policy_certified": self._policy_certified, **fields,
        }
        with self._lock:
            self._file.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self._file.flush()

    def close(self, timeout_s=0.0):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._file.close()
```

The runtime writes one record per preflight, state transition, request, result, publish/preview, fault, and shutdown event.

- [ ] Implement keyboard save/cbreak entry.
- [ ] Implement nonblocking `p`-only polling.
- [ ] Implement unconditional terminal restoration.
- [ ] Implement one compact metrics-record serializer.

- [ ] **Step 6: Implement the zero-I/O shutdown subprocess fixture**

Create `scripts/tests/bridge_subprocess_fixture.py` with this complete content. It owns only an OS-assigned loopback HTTP port and never imports rclpy, Unitree, MuJoCo, ZMQ, or a robot adapter:

```python
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.psi0_simple_real_bridge import (  # noqa: E402
    HttpInferenceWorker, ShutdownCoordinator,
)
from simple.baselines.client import (  # noqa: E402
    HttpActionClient, convert_numpy_in_dict, numpy_serialize,
)
from simple.deploy.psi0_simple_bridge import (  # noqa: E402
    PolicyContract, RtcRequest,
)


class FixtureBridge:
    def __init__(self, with_hold):
        self._hold = None
        if with_hold:
            self._hold = SimpleNamespace(
                target_upper_body_pose=np.zeros(31, np.float32),
                base_height_command=np.asarray([0.74], np.float32),
                navigate_cmd=np.zeros(4, np.float32),
                timestamp=0.0,
                target_time=0.02,
            )

    def stop(self):
        return None

    def build_bounded_shutdown_hold(self):
        return self._hold


class RecordingSink:
    def __init__(self):
        self.goals = []
        self.publish_attempts = 0
        self.closed = False
        self.publisher_closed_after_publish_count = None

    def publish(self, goal):
        if self.closed:
            raise RuntimeError("publish after close")
        self.publish_attempts += 1
        self.goals.append({
            "scheduled_at": time.monotonic(),
            "goal": {
                "target_upper_body_pose": goal.target_upper_body_pose.tolist(),
                "base_height_command": goal.base_height_command.tolist(),
                "navigate_cmd": goal.navigate_cmd.tolist(),
                "timestamp": float(goal.timestamp),
                "target_time": float(goal.target_time),
            },
        })
        return True

    def close(self, timeout_s):
        self.closed = True
        self.publisher_closed_after_publish_count = self.publish_attempts


class ClosingResource:
    def __init__(self):
        self.closed = False

    def close(self, timeout_s):
        self.closed = True


class IdleWorker(ClosingResource):
    busy = False


class DelayedRtcHandler(BaseHTTPRequestHandler):
    def log_message(self, _format, *args):
        return None

    def do_POST(self):
        if self.path != "/act-rtc-v1":
            self.send_error(404)
            return
        size = int(self.headers["Content-Length"])
        self.rfile.read(size)
        self.server.request_accepted.set()
        time.sleep(5.2)
        response = convert_numpy_in_dict({
            "action": np.zeros((24, 36), np.float32),
            "metadata": {
                "session_id": "fixture-session", "request_seq": 0,
                "observation_tick": 100, "prediction_horizon": 30,
                "execution_horizon": 24, "rtc_delay_steps": 6,
                "first_action_tick": 106,
            },
        }, numpy_serialize)
        body = json.dumps(response, separators=(",", ":")).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


class DelayedRtcHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = False
    block_on_close = True


class RtcServerHarness:
    def __init__(self):
        self.request_accepted = threading.Event()
        self.server = DelayedRtcHttpServer(
            ("127.0.0.1", 0), DelayedRtcHandler
        )
        self.server.request_accepted = self.request_accepted
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.01),
            name="fixture-rtc-server", daemon=False,
        )
        self.thread.start()

    def close(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        self.server.shutdown()
        self.thread.join(max(0.0, deadline - time.monotonic()))
        if self.thread.is_alive():
            raise RuntimeError("fixture RTC accept loop missed deadline")
        self.server.server_close()
        if time.monotonic() > deadline:
            raise RuntimeError("fixture RTC handler missed deadline")


def fixture_policy_contract():
    return PolicyContract.from_dict({
        "schema": "simple.psi0.policy-contract.v2", "test_only": True,
        "checkpoint_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "raw_episode_sha256": "c" * 64,
        "processed_episode_sha256": "d" * 64,
        "source_episode_index": 7, "processed_episode_index": 3,
        "converter_commit": "e" * 40, "server_commit": "f" * 40,
        "converter_layout": "g1_simple_32_rpyh_v2",
        "observation_dim": 32, "action_dim": 36,
        "action_frequency_hz": 50, "prediction_horizon": 30,
        "execution_horizon": 24, "rtc_delay_steps": 6,
        "rtc_training_max_delay": 7, "rtc_enabled": True,
        "rtc_endpoint": "/act-rtc-v1",
        "request_semantics": "exact-post-slew-committed-prefix",
        "response_semantics": "denormalized-executable-suffix",
        "image_key": "rgb_head_stereo_left", "camera_color_order": "rgb",
    })


def start_inflight_request(server):
    contract = fixture_policy_contract()
    worker = HttpInferenceWorker(
        client=HttpActionClient("127.0.0.1", server.port, timeout=5.0),
        clock=time.monotonic, contract=contract,
    )
    committed = np.zeros((6, 36), np.float32)
    committed[:, 31] = 0.74
    worker.submit(RtcRequest(
        generation=1, session_id="fixture-session", request_seq=0,
        observation_tick=100, history_tick=99,
        observation=np.zeros((1, 32), np.float32),
        image=np.zeros((8, 8, 3), np.uint8),
        committed_actions=committed, reset=True,
        deadline_at=time.monotonic() + 0.12,
    ))
    return worker


def atomic_write_json(path, payload):
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")))
    os.replace(temporary, destination)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", required=True,
        choices=("no-state", "inflight-five-second"),
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--ready", required=True)
    args = parser.parse_args(argv)

    inflight = args.scenario == "inflight-five-second"
    bridge = FixtureBridge(with_hold=inflight)
    sink = RecordingSink()
    camera = ClosingResource()
    server = RtcServerHarness() if inflight else None
    worker = start_inflight_request(server) if inflight else IdleWorker()
    inference_requests = [{"request_seq": 0}] if inflight else []
    state = ClosingResource()
    ros = ClosingResource()
    keyboard = ClosingResource()
    metrics = ClosingResource()
    ownership_guard = ClosingResource()
    coordinator = ShutdownCoordinator(
        bridge=bridge, command_sink=sink, camera=camera, worker=worker,
        state_source=state, ownership_guard=ownership_guard,
        ros_runtime=ros, keyboard=keyboard, metrics=metrics,
    )

    interrupted = threading.Event()
    signal.signal(signal.SIGINT, lambda _signum, _frame: interrupted.set())
    if inflight and not server.request_accepted.wait(0.2):
        raise RuntimeError("fixture worker did not accept request")
    Path(args.ready).touch(exist_ok=False)
    interrupted.wait()
    shutdown = coordinator.close()
    fixture_errors = []
    if server is not None:
        try:
            server.close(max(0.0, shutdown.deadline_at - time.monotonic()))
        except Exception as error:
            fixture_errors.append(f"fixture_server: {error}")

    current = threading.current_thread()
    live = [
        thread.name for thread in threading.enumerate()
        if thread is not current and not thread.daemon and thread.is_alive()
    ]
    atomic_write_json(args.report, {
        "owned_ports": [] if server is None else [server.port],
        "goal_lower_bounds": [-2.0] * 31,
        "goal_upper_bounds": [2.0] * 31,
        "goals": sink.goals,
        "publish_attempts": sink.publish_attempts,
        "publisher_closed_after_publish_count": (
            sink.publisher_closed_after_publish_count
        ),
        "publisher_closed": sink.closed,
        "camera_closed": camera.closed,
        "terminal_restored": keyboard.closed,
        "inference_requests": inference_requests,
        "request_accepted": (
            False if server is None else server.request_accepted.is_set()
        ),
        "live_non_daemon_bridge_threads": live,
        "shutdown_started_at": shutdown.started_at,
        "shutdown_deadline_at": shutdown.deadline_at,
        "shutdown_finished_at": shutdown.finished_at,
        "cleanup_errors": [*shutdown.cleanup_errors, *fixture_errors],
    })
    return 0 if not shutdown.cleanup_errors and not fixture_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Implement exact shutdown branching**

On Ctrl-C:

1. latch stop, increment generation, and clear all policy buffers/bookkeeping;
2. call `build_bounded_hold(now)`;
3. if it returns a hold, publish exactly 25 identical goals on consecutive 20 ms schedule ticks, then close the publisher immediately and forbid a 26th publish;
4. if it returns `None`, publish zero messages and continue immediately;
5. signal camera exit, prevent new inference work, join the worker for at most 5.5 seconds, close messaging, restore terminal, and exit;
6. enforce the overall 6.5-second bound for the in-flight case and 0.5-second bound when neither final hold nor HTTP join is pending.

Implement that ordering with the following coordinator and loop. `publisher` is `None` in shadow; `preview_sink.close()` is passed in its place so resource ownership is still explicit.

```python
@dataclass(frozen=True)
class ShutdownReport:
    final_hold_publishes: int
    publisher_closed_after_publish_count: int
    started_at: float
    deadline_at: float
    finished_at: float
    cleanup_errors: tuple[str, ...]


class ShutdownCoordinator:
    def __init__(
        self, *, bridge, command_sink, camera, worker, state_source,
        ownership_guard, ros_runtime, keyboard, metrics,
        clock=time.monotonic, sleep=time.sleep,
    ):
        self.bridge = bridge
        self.command_sink = command_sink
        self.camera = camera
        self.worker = worker
        self.state_source = state_source
        self.ownership_guard = ownership_guard
        self.ros_runtime = ros_runtime
        self.keyboard = keyboard
        self.metrics = metrics
        self.clock = clock
        self.sleep = sleep
        self._closed = False

    def close(self):
        if self._closed:
            raise RuntimeError("shutdown coordinator may run only once")
        self._closed = True
        started_at = self.clock()
        errors = []
        try:
            self.bridge.stop()
        except Exception as error:
            errors.append(f"bridge_stop: {error}")
        hold = None
        try:
            hold = self.bridge.build_bounded_shutdown_hold()
        except Exception as error:
            errors.append(f"bounded_hold: {error}")
        try:
            worker_busy = bool(self.worker.busy)
        except Exception as error:
            errors.append(f"worker_busy: {error}")
            worker_busy = True
        long_path = hold is not None or worker_busy
        deadline_at = started_at + (6.5 if long_path else 0.5)

        def remaining(cap=None):
            value = max(0.0, deadline_at - self.clock())
            return value if cap is None else min(value, cap)

        def close_resource(name, resource, cap=None):
            if resource is None:
                return
            budget = remaining(cap)
            if budget <= 0.0:
                errors.append(f"{name}: overall shutdown deadline exhausted")
            try:
                resource.close(timeout_s=budget)
            except Exception as error:
                errors.append(f"{name}: {error}")
            if self.clock() > deadline_at:
                errors.append(f"{name}: exceeded overall shutdown deadline")

        published = 0
        try:
            if hold is not None and self.command_sink is not None:
                first = self.clock()
                for index in range(25):
                    scheduled_at = first + index * 0.02
                    if scheduled_at > deadline_at:
                        raise TimeoutError("final hold exceeds overall deadline")
                    self.sleep(max(0.0, scheduled_at - self.clock()))
                    if self.clock() > deadline_at:
                        raise TimeoutError("final hold missed overall deadline")
                    if self.command_sink.publish(hold) is not True:
                        raise RuntimeError("final hold publish rejected")
                    published += 1
        except Exception as error:
            errors.append(f"final_hold: {error}")
        finally:
            close_resource("publisher", self.command_sink)

        close_resource("camera", self.camera, 0.5)
        close_resource("worker", self.worker, 5.5)
        close_resource("state", self.state_source)
        close_resource("keyboard", self.keyboard)
        close_resource("ownership_guard", self.ownership_guard)
        close_resource("ros", self.ros_runtime)
        close_resource("metrics", self.metrics)
        finished_at = self.clock()
        if finished_at > deadline_at and not any(
            "overall shutdown deadline" in error for error in errors
        ):
            errors.append("cleanup: exceeded overall shutdown deadline")
        return ShutdownReport(
            final_hold_publishes=published,
            publisher_closed_after_publish_count=published,
            started_at=started_at,
            deadline_at=deadline_at,
            finished_at=finished_at,
            cleanup_errors=tuple(errors),
        )


class FiftyHzLoop:
    def __init__(self, clock=time.monotonic, sleep=time.sleep):
        self.clock = clock
        self.sleep = sleep
        self._next = None

    def run_n(self, count, callback):
        first = self.clock() if self._next is None else self._next
        for index in range(count):
            scheduled = first + index * 0.02
            self.sleep(max(0.0, scheduled - self.clock()))
            callback(scheduled)
        self._next = first + count * 0.02


def handle_keyboard_events(bridge, keyboard, metrics):
    for key in keyboard.poll(0.0):
        if key != "p":
            continue
        try:
            bridge.handle_toggle()
        except ActivationRefused as error:
            metrics.write_event(
                "activation_refused",
                state=bridge.state.value,
                reason=str(error),
            )


def run_runtime(
    bridge, adapters, keyboard, coordinator, metrics, ownership_guard,
):
    latest_state = None
    latest_camera = None

    def one_tick(_scheduled):
        nonlocal latest_state, latest_camera
        ownership_guard.check()
        state = adapters.state_source.poll()
        camera = adapters.camera_reader.poll()
        latest_state = state if state is not None else latest_state
        latest_camera = camera if camera is not None else latest_camera
        if latest_state is not None and latest_camera is not None:
            bridge.update_inputs(latest_state, latest_camera)
        handle_keyboard_events(bridge, keyboard, metrics)
        result = bridge.tick()
        metrics.write(
            "tick", bridge,
            published=(
                adapters.goal_publisher is not None and result.goal is not None
            ),
            tick=result.tick, source_kind=result.source_kind,
            previewed=(
                adapters.goal_publisher is None and result.goal is not None
            ),
            psi0_action=(
                None if result.psi0_action is None
                else result.psi0_action.tolist()
            ),
            worker_busy=bridge.inference.busy,
            input_valid=getattr(bridge, "observation_valid", None),
            input_error=getattr(bridge, "input_error", None),
            discarded_late_results=bridge.metrics.discarded_late_results,
            discarded_old_generation_results=(
                bridge.metrics.discarded_old_generation_results
            ),
        )

    loop = FiftyHzLoop()
    try:
        while bridge.state is not BridgeState.STOPPED:
            loop.run_n(1, one_tick)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_report = coordinator.close()
    return shutdown_report


class ShadowPreviewSink:
    def __init__(self, metrics):
        self.metrics = metrics
        self.closed = False

    def publish(self, goal):
        if self.closed:
            raise RuntimeError("preview sink is closed")
        self.metrics.write_event(
            "preview",
            target_upper_body_pose=goal.target_upper_body_pose.tolist(),
            base_height_command=goal.base_height_command.tolist(),
            navigate_cmd=goal.navigate_cmd.tolist(),
        )
        return True

    def close(self, timeout_s=0.0):
        self.closed = True


class DisabledInferenceWorker:
    busy = False

    def __init__(self):
        self.closed = False

    def close(self, timeout_s=0.0):
        self.closed = True


class ObservationOnlyShadowBridge:
    def __init__(self, inference, joints, clock, *, start_tick):
        self.inference = inference
        self.joints = joints
        self.clock = clock
        self.tick_index = start_tick
        self.state = BridgeState.PAUSED
        self.generation = 0
        self.metrics = BridgeMetrics()
        self.last_valid_state = None
        self.last_snapshot = None
        self.observation_valid = False
        self.input_error = "no synchronized inputs"

    def update_inputs(self, state, camera):
        accepted, reason = accept_measured_state(
            self.last_valid_state, state, self.joints, self.clock()
        )
        self.last_valid_state = accepted
        if reason is not None:
            self.observation_valid = False
            self.input_error = reason
            return
        try:
            snapshot = validate_synchronized_snapshot(
                accepted, camera, self.clock()
            )
        except ValueError as error:
            self.observation_valid = False
            self.input_error = str(error)
            return
        self.last_snapshot = snapshot
        self.observation_valid = True
        self.input_error = None

    def handle_toggle(self):
        raise ActivationRefused(
            "no usable policy contract; shadow remains observation-only"
        )

    def tick(self):
        result = TickResult(
            self.tick_index, None, None, None, "none"
        )
        self.tick_index += 1
        return result

    def stop(self):
        if self.state is not BridgeState.STOPPED:
            self.generation += 1
            self.state = BridgeState.STOPPED

    def build_bounded_shutdown_hold(self):
        return None


def _is_loopback_host(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class ConnectionEvidenceRecorder:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._records = {}

    def _observe(self, component, transport, endpoint, real_interface):
        if component in self._records:
            raise PreflightError(f"duplicate observed connection: {component}")
        if type(endpoint) is not str or not endpoint:
            raise PreflightError(f"invalid observed endpoint: {component}")
        self._records[component] = {
            "component": component,
            "transport": transport,
            "endpoint": endpoint,
            "real_interface": bool(real_interface),
            "observed_at": float(self._clock()),
        }

    def observe_wbc_response(self, payload):
        if type(payload) is not dict:
            raise PreflightError("WBC response observation requires payload")
        interface = payload.get("interface")
        env_type = payload.get("env_type")
        if type(interface) is not str or type(env_type) is not str:
            raise PreflightError("WBC response lacks interface identity")
        self._observe(
            "wbc", "dds-service-response", interface,
            not (env_type == "sim" and interface == "lo"),
        )

    def observe_camera_frame(self, host, frame):
        if (
            type(frame) is not TimedCameraFrame
            or type(frame.image) is not np.ndarray
            or frame.image.dtype != np.uint8
            or frame.image.ndim != 3
            or frame.image.shape[2] != 3
        ):
            raise PreflightError("camera observation requires a decoded frame")
        self._observe(
            "camera", "decoded-frame", host, not _is_loopback_host(host)
        )

    def observe_policy_contract(self, host, payload):
        if type(payload) is not dict or not payload:
            raise PreflightError("policy observation requires contract response")
        self._observe(
            "policy", "http-contract-response", host,
            not _is_loopback_host(host),
        )

    def snapshot(self, required_components):
        required = set(required_components)
        missing = required - set(self._records)
        if missing:
            raise PreflightError(
                "missing observed connections: " + ",".join(sorted(missing))
            )
        order = ("wbc", "camera", "policy")
        evidence = [dict(self._records[name]) for name in order if name in self._records]
        count_real_interface_connections(evidence)
        return evidence


def count_real_interface_connections(evidence):
    expected_keys = {
        "component", "transport", "endpoint", "real_interface", "observed_at",
    }
    allowed_components = {"wbc", "camera", "policy"}
    if type(evidence) is not list or not 1 <= len(evidence) <= 3:
        raise PreflightError("connection evidence record count")
    components = [record.get("component") for record in evidence]
    if len(set(components)) != len(components) or not set(components) <= allowed_components:
        raise PreflightError("connection evidence component set")
    for record in evidence:
        if type(record) is not dict or set(record) != expected_keys:
            raise PreflightError("connection evidence record schema")
        if (
            type(record["transport"]) is not str
            or type(record["endpoint"]) is not str
            or type(record["real_interface"]) is not bool
            or type(record["observed_at"]) is not float
            or not np.isfinite(record["observed_at"])
        ):
            raise PreflightError("connection evidence record types")
    return sum(record["real_interface"] for record in evidence)


def _close_partial(resources):
    errors = []
    seen = set()
    for name, resource in reversed(resources):
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))
        try:
            resource.close(timeout_s=0.5)
        except Exception as error:
            errors.append(f"{name}: {error}")
    return tuple(errors)


def validate_domain_selection(mode, ros_domain_id, unitree_domain_id):
    mode = BridgeMode(mode)
    for label, value in (
        ("ROS", ros_domain_id), ("Unitree", unitree_domain_id)
    ):
        if type(value) is not int or not 0 <= value <= 232:
            raise PreflightError(f"{label} domain must be an integer in [0,232]")
    if (
        mode is BridgeMode.SIM_CONTROL
        and (ros_domain_id, unitree_domain_id) != (42, 42)
    ):
        raise PreflightError("sim-control requires isolated domain 42")
    return ros_domain_id, unitree_domain_id


def load_local_policy_payload(path, mode):
    mode = BridgeMode(mode)
    if path is None:
        if mode is BridgeMode.SIM_CONTROL:
            raise PreflightError("--policy-contract is required in sim-control")
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        if mode is BridgeMode.SIM_CONTROL:
            raise PreflightError(f"cannot read local policy contract: {error}") from error
        return None
    return payload


def run_bridge(args):
    mode = BridgeMode(args.mode)
    validate_domain_selection(
        mode, args.ros_domain_id, args.unitree_domain_id
    )
    os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
    os.environ["UNITREE_DOMAIN_ID"] = str(args.unitree_domain_id)
    repository_root = Path(__file__).resolve().parents[1]
    resources = []
    coordinator = None
    connection_recorder = ConnectionEvidenceRecorder()
    try:
        runtime = RosRuntime(args.ros_domain_id)
        resources.append(("ros", runtime))
        wbc_payload = create_ros_wbc_config_client(
            "WBCPolicy/robot_config", args.ros_domain_id
        ).get_config(3.0)
        connection_recorder.observe_wbc_response(wbc_payload)
        local_policy_payload = load_local_policy_payload(
            args.policy_contract, mode
        )
        policy_client = build_policy_client(args.server_host, args.server_port)
        try:
            server_policy_payload = policy_client.get_contract(timeout=2.0)
            connection_recorder.observe_policy_contract(
                args.server_host, server_policy_payload
            )
        except Exception:
            if mode is not BridgeMode.SHADOW:
                raise
            server_policy_payload = None
        local_wbc = build_local_wbc_identity(repository_root)
        preflight = run_preflight(
            mode=mode, local_policy=local_policy_payload,
            server_policy=server_policy_payload, wbc_payload=wbc_payload,
            graph=runtime, expected_model_contract=local_wbc.model_contract,
            expected_gitlink_sha=local_wbc.root_gitlink_sha,
            required_domain_id=args.ros_domain_id,
        )
        factories = RuntimeDependencyFactories(
            state_source=lambda: RosStateSource(runtime, "G1Env/env_state_act"),
            camera_reader=lambda: ComposedCameraReader(
                args.camera_host, args.camera_port, args.camera_source_key,
                args.camera_color_order,
            ),
            graph=lambda: runtime,
        )
        adapters = build_runtime_adapters(
            mode=mode, preflight_result=preflight,
            publisher_factory=lambda: RosGoalPublisher(
                runtime, CONTROL_GOAL_TOPIC
            ),
            test_dependencies=factories,
        )
        adapters = RuntimeAdapters(
            adapters.state_source, adapters.camera_reader,
            adapters.goal_publisher, runtime,
        )
        resources.extend((
            ("publisher", adapters.goal_publisher),
            ("state", adapters.state_source),
            ("camera", adapters.camera_reader),
        ))
        observed_camera = adapters.camera_reader.wait_for_frame(timeout_s=1.0)
        connection_recorder.observe_camera_frame(
            args.camera_host, observed_camera
        )
        ownership_guard = GoalOwnershipGuard(
            mode, runtime, preflight.goal_counts_at_preflight
        )
        resources.append(("ownership_guard", ownership_guard))
        metrics = JsonlMetrics(
            args.metrics_jsonl, mode=mode,
            policy_certified=preflight.policy_certified,
        )
        resources.append(("metrics", metrics))
        command_sink = (
            adapters.goal_publisher
            if adapters.goal_publisher is not None
            else ShadowPreviewSink(metrics)
        )
        resources.append(("command_sink", command_sink))
        contract = preflight.runtime_policy_contract
        if contract is None:
            worker = DisabledInferenceWorker()
            bridge = ObservationOnlyShadowBridge(
                worker, preflight.joint_contract, time.monotonic,
                start_tick=int(time.monotonic() * 50),
            )
        else:
            worker = HttpInferenceWorker(
                client=policy_client, clock=time.monotonic, contract=contract,
                instruction=args.instruction, event_sink=metrics.write_event,
            )
            bridge = Psi0SimpleBridge(
                contract, preflight.joint_contract, worker, time.monotonic,
                start_tick=int(
                    time.monotonic() * contract.action_frequency_hz
                ),
                consume_goal=command_sink.publish,
            )
        resources.append(("worker", worker))
        keyboard = LocalKeyboard(sys.stdin.fileno())
        keyboard.__enter__()
        resources.append(("keyboard", keyboard))
        required_connections = {"wbc", "camera"}
        if mode is BridgeMode.SIM_CONTROL:
            required_connections.add("policy")
        connection_evidence = connection_recorder.snapshot(
            required_components=required_connections
        )
        metrics.write_event(
            "preflight_complete",
            policy_mismatched_fields=list(preflight.policy_mismatched_fields),
            wbc_mismatched_fields=list(preflight.wbc_mismatched_fields),
            publisher_required=preflight.publisher_required,
            goal_counts_at_preflight=list(
                preflight.goal_counts_at_preflight
            ),
            connection_evidence=connection_evidence,
            real_interface_connections=count_real_interface_connections(
                connection_evidence
            ),
        )
        coordinator = ShutdownCoordinator(
            bridge=bridge, command_sink=command_sink,
            camera=adapters.camera_reader, worker=worker,
            state_source=adapters.state_source,
            ownership_guard=ownership_guard, ros_runtime=runtime,
            keyboard=keyboard, metrics=metrics,
        )
        report = run_runtime(
            bridge, adapters, keyboard, coordinator, metrics,
            ownership_guard,
        )
        if report.cleanup_errors:
            raise RuntimeError(
                "bridge shutdown errors: " + "; ".join(report.cleanup_errors)
            )
        return 0
    except Exception as error:
        cleanup_errors = ()
        if coordinator is not None and not coordinator._closed:
            report = coordinator.close()
            cleanup_errors = report.cleanup_errors
        elif coordinator is None:
            cleanup_errors = _close_partial(resources)
        if cleanup_errors:
            raise RuntimeError(
                f"{error}; cleanup: " + "; ".join(cleanup_errors)
            ) from error
        raise


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run_bridge(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

In shadow, the command sink's `publish()` records a preview and returns true but never owns a ROS publisher; its `close()` only closes that recorder. The preflight accepts attested structural joint data while reporting configuration/model differences, `JsonlMetrics` stamps `policy_certified=false` onto every event, and `GoalOwnershipGuard` checks the initial publisher count on every tick and again before ROS teardown. If no bounded hold exists, the loop above performs zero final calls. If it exists, the coordinator calls `publish()` exactly 25 times with the same `Goal` object, closes the sink immediately afterward, and has no code path for a 26th call. The runtime never signals or terminates the WBC, simulator, policy server, or any robot process.

- [ ] Add idempotence rejection and bridge-stop/hold capture only.
- [ ] Add the exact 25-tick publisher branch and run the fake-clock unit case.
- [ ] Add the no-hold zero-publish branch and run the pre-state case.
- [ ] Add ordered cleanup with error accumulation and run both subprocess cases.
- [ ] Add the persistent-deadline 50 Hz loop and run the 50-scheduled-tick case.
- [ ] Wire `run_runtime()` and prove unexpected exceptions are re-raised after cleanup.

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
import signal
import socket
import subprocess
import sys
import time

import numpy as np
import pytest
import requests

from scripts.psi0_simple_real_bridge import ComposedCameraReader, HttpInferenceWorker
from scripts.tests.fake_composed_camera_server import running_fake_camera
from scripts.tests.fake_psi0_rtc_server import running_fake_policy
from simple.baselines.client import HttpActionClient
from simple.deploy.psi0_simple_bridge import (
    BridgeState, PolicyContract, Psi0SimpleBridge, TimedCameraFrame,
    TimedRobotState,
)
from tests.psi0_bridge_testkit import make_joint_contract

CONTRACT_PATH = Path("scripts/tests/fixtures/psi0_policy_contract_test_v2.json")


def assert_loopback_port_rebinds(port):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))


def wait_for_json(path, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text())
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


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
        contract=contract,
    )
    bridge = Psi0SimpleBridge(
        contract, make_joint_contract(), worker, time.monotonic, start_tick=0
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
        old_generation_results_discarded=(
            bridge.metrics.discarded_old_generation_results
        ),
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
        assert report.old_generation_results_discarded == 1
        assert report.policy_actions_after_fault == 0
        assert report.requests_submitted == 1
        assert fake.max_concurrent_requests == 1


def test_cli_control_endpoint_delays_exact_next_request_across_processes(tmp_path):
    ready = tmp_path / "policy-ready.json"
    token = "test-control-token"
    command = [
        sys.executable, "scripts/tests/fake_psi0_rtc_server.py",
        "--host", "127.0.0.1", "--port", "0",
        "--normal-latency-s", "0.05",
        "--contract", str(CONTRACT_PATH),
        "--control-token", token, "--ready-json", str(ready),
    ]
    process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1])
    port = None
    try:
        port = wait_for_json(ready)["port"]
        base = f"http://127.0.0.1:{port}"
        armed = requests.post(
            f"{base}/test-control/arm-next-delay",
            json={"token": token, "latency_s": 0.30}, timeout=0.5,
        )
        assert armed.status_code == 202
        assert armed.json() == {"armed_request_seq": 0, "latency_s": 0.30}
        client = HttpActionClient("127.0.0.1", port, timeout=1.0)
        started = time.monotonic()
        query_fake(client, rtc_history(0, 100, np.zeros((6, 36), np.float32), reset=True))
        assert time.monotonic() - started >= 0.295
        status = requests.get(
            f"{base}/test-control/status",
            headers={"X-Test-Control-Token": token}, timeout=0.5,
        ).json()
        assert status["records"][0]["request_seq"] == 0
        assert status["records"][0]["applied_latency_s"] == pytest.approx(0.30)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=1.0)
    assert process.returncode == 0
    assert port is not None
    assert_loopback_port_rebinds(port)


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

Create `scripts/tests/fake_psi0_rtc_server.py` with the following literal implementation. It validates `dataset_name="simple"`, empty condition, exact image/state dictionaries, R0-only reset, full history key sets, and the supplied `(d,36)` committed prefix. It generates a safe stationary suffix from the final committed command:

```python
import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import threading
import time

import numpy as np

from simple.baselines.client import (
    RequestMessage, convert_numpy_in_dict, numpy_serialize,
)


OUTER_KEYS = {
    "image", "instruction", "history", "state", "condition",
    "gt_action", "dataset_name", "timestamp",
}
BASE_HISTORY_KEYS = {
    "session_id", "request_seq", "observation_tick",
    "rtc_delay_steps", "committed_actions",
}


class ProtocolError(RuntimeError):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


@dataclass(frozen=True)
class RequestRecord:
    request_seq: int
    observation_tick: int
    committed_actions: np.ndarray
    reset: bool
    applied_latency_s: float


@dataclass
class FakePolicyState:
    contract: dict
    normal_latency_s: float
    control_token: str | None
    lock: threading.Lock = field(default_factory=threading.Lock)
    delay_by_seq: dict[int, float] = field(default_factory=dict)
    records: list[RequestRecord] = field(default_factory=list)
    last_started_request_seq: int = -1
    active_requests: int = 0
    max_concurrent_requests: int = 0

    def arm_next(self, latency_s):
        if type(latency_s) not in (int, float) or type(latency_s) is bool:
            raise ProtocolError(400, "latency_s must be numeric")
        latency_s = float(latency_s)
        if not 0.0 <= latency_s <= 1.0:
            raise ProtocolError(400, "latency_s outside [0,1]")
        with self.lock:
            target = self.last_started_request_seq + 1
            if target in self.delay_by_seq:
                raise ProtocolError(409, "next request already armed")
            self.delay_by_seq[target] = latency_s
        return target, latency_s


def _validate_request(payload, state):
    if type(payload) is not dict or set(payload) != OUTER_KEYS:
        raise ProtocolError(400, "request key set")
    request = RequestMessage.deserialize(payload)
    contract = state.contract
    if request.dataset_name != "simple" or request.condition != {}:
        raise ProtocolError(400, "dataset/condition")
    if type(request.image) is not dict or set(request.image) != {contract["image_key"]}:
        raise ProtocolError(400, "image dictionary")
    if type(request.state) is not dict or set(request.state) != {"states"}:
        raise ProtocolError(400, "state dictionary")
    image = np.asarray(request.image[contract["image_key"]])
    states = np.asarray(request.state["states"])
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ProtocolError(400, "image value")
    if states.dtype != np.float32 or states.shape != (1, 32) or not np.isfinite(states).all():
        raise ProtocolError(400, "state value")
    if request.gt_action != [] or type(request.instruction) is not str:
        raise ProtocolError(400, "gt_action/instruction")
    history = request.history
    if type(history) is not dict:
        raise ProtocolError(400, "history type")
    seq = history.get("request_seq")
    expected_keys = BASE_HISTORY_KEYS | ({"reset"} if seq == 0 else set())
    if set(history) != expected_keys:
        raise ProtocolError(400, "history key set")
    if type(history["session_id"]) is not str or not history["session_id"]:
        raise ProtocolError(400, "session_id")
    for key in ("request_seq", "observation_tick", "rtc_delay_steps"):
        if type(history[key]) is not int:
            raise ProtocolError(400, f"{key} type")
    if seq == 0 and history["reset"] is not True:
        raise ProtocolError(400, "R0 reset")
    if history["rtc_delay_steps"] != contract["rtc_delay_steps"]:
        raise ProtocolError(400, "rtc delay")
    committed = np.asarray(history["committed_actions"])
    expected = (contract["rtc_delay_steps"], contract["action_dim"])
    if committed.dtype != np.float32 or committed.shape != expected:
        raise ProtocolError(400, "committed prefix")
    if not np.isfinite(committed).all():
        raise ProtocolError(400, "committed prefix finite")
    return request, history, committed.copy()


class FakePolicyHandler(BaseHTTPRequestHandler):
    server_version = "SIMPLEFakeRTC/1"

    @property
    def state(self):
        return self.server.state

    def log_message(self, _format, *args):
        return None

    def _read_json(self):
        try:
            size = int(self.headers.get("Content-Length", ""))
            return json.loads(self.rfile.read(size))
        except Exception as error:
            raise ProtocolError(400, f"invalid JSON: {error}") from error

    def _send(self, status, payload):
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        return (
            self.state.control_token is not None
            and self.headers.get("X-Test-Control-Token")
            == self.state.control_token
        )

    def do_GET(self):
        if self.path == "/contract":
            self._send(200, self.state.contract)
            return
        if self.path == "/test-control/status":
            if not self._authorized():
                self._send(403, {"error": "forbidden"})
                return
            with self.state.lock:
                payload = {
                    "last_started_request_seq": self.state.last_started_request_seq,
                    "active_requests": self.state.active_requests,
                    "max_concurrent_requests": self.state.max_concurrent_requests,
                    "records": [
                        {"request_seq": record.request_seq,
                         "applied_latency_s": record.applied_latency_s}
                        for record in self.state.records
                    ],
                }
            self._send(200, payload)
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            if self.path == "/test-control/arm-next-delay":
                payload = self._read_json()
                if type(payload) is not dict or set(payload) != {"token", "latency_s"}:
                    raise ProtocolError(400, "control key set")
                if payload["token"] != self.state.control_token:
                    raise ProtocolError(403, "forbidden")
                seq, delay = self.state.arm_next(payload["latency_s"])
                self._send(202, {"armed_request_seq": seq, "latency_s": delay})
                return
            if self.path != "/act-rtc-v1":
                raise ProtocolError(404, "not found")
            request, history, committed = _validate_request(
                self._read_json(), self.state
            )
            seq = history["request_seq"]
            with self.state.lock:
                expected = self.state.last_started_request_seq + 1
                if seq != expected:
                    raise ProtocolError(409, f"expected request_seq {expected}")
                self.state.last_started_request_seq = seq
                delay = self.state.delay_by_seq.pop(
                    seq, self.state.normal_latency_s
                )
                self.state.active_requests += 1
                self.state.max_concurrent_requests = max(
                    self.state.max_concurrent_requests,
                    self.state.active_requests,
                )
            recorded = False
            try:
                time.sleep(delay)
                contract = self.state.contract
                actions = np.repeat(
                    committed[-1:], contract["execution_horizon"], axis=0
                ).astype(np.float32)
                actions[:, 32:36] = 0.0
                metadata = {
                    "session_id": history["session_id"],
                    "request_seq": seq,
                    "observation_tick": history["observation_tick"],
                    "prediction_horizon": contract["prediction_horizon"],
                    "execution_horizon": contract["execution_horizon"],
                    "rtc_delay_steps": contract["rtc_delay_steps"],
                    "first_action_tick": (
                        history["observation_tick"] + contract["rtc_delay_steps"]
                    ),
                }
                response = convert_numpy_in_dict(
                    {"action": actions, "metadata": metadata}, numpy_serialize
                )
                with self.state.lock:
                    self.state.active_requests -= 1
                    self.state.records.append(RequestRecord(
                        seq, history["observation_tick"], committed,
                        history.get("reset", False), delay,
                    ))
                    recorded = True
                self._send(200, response)
            finally:
                if not recorded:
                    with self.state.lock:
                        self.state.active_requests -= 1
        except ProtocolError as error:
            self._send(error.status, {"error": str(error)})


class LoopbackThreadingServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False


class FakePolicyHandle:
    def __init__(self, server, thread):
        self.server = server
        self.thread = thread
        self.port = server.server_address[1]

    @property
    def records(self):
        with self.server.state.lock:
            return tuple(self.server.state.records)

    @property
    def max_concurrent_requests(self):
        with self.server.state.lock:
            return self.server.state.max_concurrent_requests

    def delay_next_request(self, seconds):
        return self.server.state.arm_next(seconds)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(0.5)
        if self.thread.is_alive():
            raise RuntimeError("fake policy server did not stop")


def start_fake_policy(contract_path, normal_latency_s, control_token=None, port=0):
    contract = json.loads(Path(contract_path).read_text())
    server = LoopbackThreadingServer(("127.0.0.1", port), FakePolicyHandler)
    server.state = FakePolicyState(contract, float(normal_latency_s), control_token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return FakePolicyHandle(server, thread)


@contextmanager
def running_fake_policy(contract_path, normal_latency_s=0.05):
    handle = start_fake_policy(contract_path, normal_latency_s)
    try:
        yield handle
    finally:
        handle.close()


def atomic_ready(path, host, port):
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps({"host": host, "port": port}))
    os.replace(temporary, destination)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--normal-latency-s", required=True, type=float)
    parser.add_argument("--control-token", required=True)
    parser.add_argument("--ready-json", required=True)
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1" or not args.control_token:
        parser.error("authenticated loopback is required")
    handle = start_fake_policy(
        args.contract, args.normal_latency_s, args.control_token, args.port
    )
    stopped = threading.Event()
    signal.signal(signal.SIGINT, lambda _signum, _frame: stopped.set())
    atomic_ready(args.ready_json, args.host, handle.port)
    stopped.wait()
    handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Implement one route/behavior per bounded subaction:

- [ ] Bind loopback/OS-assigned port and implement bounded close/join.
- [ ] Add exact `GET /contract` only.
- [ ] Add request deserialization and exact field/history validation.
- [ ] Add stationary suffix plus exact seven-key metadata response.
- [ ] Add locked concurrency counters and default/one-shot latency.
- [ ] Add the tokenized arm/status routes and run the cross-process test.
- [ ] Add atomic readiness JSON, SIGINT close, and port-rebind assertion.

- [ ] **Step 4: Implement the fake composed-camera server**

Create `scripts/tests/fake_composed_camera_server.py` with this complete implementation. It does not subclass `SensorServer`, because that class binds `tcp://*`; it only reuses the real `ImageMessageSchema` codec:

```python
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import threading
import time

import msgpack
import numpy as np
import zmq

from decoupled_wbc.control.sensor.sensor_server import ImageMessageSchema


def sentinel_rgb():
    image = np.zeros((48, 64, 3), np.uint8)
    image[:, :20] = [240, 0, 0]
    image[:, 44:] = [0, 0, 240]
    return np.ascontiguousarray(image)


class FakeCameraHandle:
    def __init__(self, key, host="127.0.0.1", port=0):
        if host != "127.0.0.1":
            raise ValueError("fake camera is loopback-only")
        self.key = key
        self.host = host
        self.requested_port = port
        self.port = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._error = None
        self._thread = threading.Thread(
            target=self._run, name="fake-composed-camera", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(0.5):
            raise TimeoutError("fake camera did not bind")
        if self._error is not None:
            raise RuntimeError(f"fake camera failed: {self._error}")

    def _run(self):
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        try:
            socket.setsockopt(zmq.LINGER, 0)
            endpoint = f"tcp://{self.host}"
            if self.requested_port == 0:
                self.port = socket.bind_to_random_port(endpoint)
            else:
                self.port = self.requested_port
                socket.bind(f"{endpoint}:{self.port}")
            image = sentinel_rgb()
            self._ready.set()
            while not self._stop.is_set():
                schema = ImageMessageSchema(
                    timestamps={self.key: time.monotonic()},
                    images={self.key: image},
                )
                socket.send(
                    msgpack.packb(schema.serialize(), use_bin_type=True),
                    flags=zmq.NOBLOCK,
                )
                self._stop.wait(0.03)
        except Exception as error:
            self._error = error
            self._ready.set()
        finally:
            socket.close(linger=0)
            context.term()

    def close(self):
        self._stop.set()
        self._thread.join(0.5)
        if self._thread.is_alive():
            raise RuntimeError("fake camera did not stop within 0.5 seconds")
        if self._error is not None:
            raise RuntimeError(f"fake camera failed: {self._error}")


@contextmanager
def running_fake_camera(key="rgb_head_stereo_left"):
    handle = FakeCameraHandle(key)
    try:
        yield handle
    finally:
        handle.close()


def atomic_ready(path, host, port):
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps({"host": host, "port": port}))
    os.replace(temporary, destination)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--key", default="rgb_head_stereo_left")
    parser.add_argument("--ready-json", required=True)
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1":
        parser.error("fake camera is loopback-only")
    handle = FakeCameraHandle(args.key, args.host, args.port)
    stopped = threading.Event()
    signal.signal(signal.SIGINT, lambda _signum, _frame: stopped.set())
    atomic_ready(args.ready_json, args.host, handle.port)
    stopped.wait()
    handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] Add the loopback PUB-thread lifecycle and port rebind test.
- [ ] Add real-schema JPEG serialization and red/blue sentinel publication.

Both fake scripts expose `--host 127.0.0.1 --port PORT --ready-json PATH`; they reject any other host. The policy fake additionally exposes `--contract PATH --normal-latency-s 0.05 --control-token TOKEN`. Port `0` requests an OS-assigned port for pytest, while the isolated smoke driver keeps its approved fixed ports 15555/22086 and consumes both readiness files before continuing.

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

import scripts.tests.smoke_psi0_simple_bridge as smoke_driver
from scripts.tests.smoke_psi0_simple_bridge import (
    OwnedChildren, SmokeConfig, SmokeSafetyError, arm_next_policy_delay,
    allocate_smoke_run_directory, build_launch_plan, collect_smoke_report,
    close_smoke_resources, default_wbc_preflight, launch,
    measured_bridge_worker_idle_at,
    measured_real_interface_connections, validate_smoke_report,
)

EXPECTED_WBC_ARGS = (
    "uv", "run", "--group", "sonic", "python", "-m",
    "decoupled_wbc.control.main.teleop.run_g1_control_loop",
    "--interface", "sim", "--simulator", "mujoco",
    "--messaging-backend", "ros2", "--enable-waist", "--with-hands",
    "--domain-id", "42", "--no-enable-onscreen", "--no-enable-offscreen",
)


@pytest.fixture(autouse=True)
def isolated_smoke_driver_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def valid_smoke_config(**updates):
    values = {
        "duration_s": 15.0, "ros_domain_id": 42, "unitree_domain_id": 42,
        "wbc_interface": "lo", "camera_host": "127.0.0.1",
        "camera_port": 15555, "policy_host": "127.0.0.1",
        "policy_port": 22086, "control_token": "smoke-control-token",
        "policy_ready_json": "outputs/psi0-smoke-policy-ready.json",
        "camera_ready_json": "outputs/psi0-smoke-camera-ready.json",
        "bridge_metrics_jsonl": "outputs/psi0-smoke-bridge.jsonl",
        "smoke_report_json": "outputs/psi0-smoke-report.json",
        "output_dir": "outputs",
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
    assert camera.argv[camera.argv.index("--host") + 1] == "127.0.0.1"
    assert camera.argv[camera.argv.index("--port") + 1] == "15555"
    assert camera.argv[camera.argv.index("--ready-json") + 1] == (
        "outputs/psi0-smoke-camera-ready.json"
    )
    policy = plan.child("policy")
    assert "--normal-latency-s" in policy.argv
    index = policy.argv.index("--normal-latency-s")
    assert policy.argv[index + 1] == "0.05"
    assert policy.argv[policy.argv.index("--host") + 1] == "127.0.0.1"
    assert policy.argv[policy.argv.index("--port") + 1] == "22086"
    assert policy.argv[policy.argv.index("--control-token") + 1] == "smoke-control-token"
    assert policy.argv[policy.argv.index("--ready-json") + 1] == (
        "outputs/psi0-smoke-policy-ready.json"
    )
    bridge = plan.child("bridge")
    assert bridge.use_pty is True
    assert "--mode" in bridge.argv and bridge.argv[bridge.argv.index("--mode") + 1] == "sim-control"
    assert all(child.env["ROS_DOMAIN_ID"] == "42" for child in plan.children)
    assert all(child.env["UNITREE_DOMAIN_ID"] == "42" for child in plan.children)


class FakeControlResponse:
    status_code = 202

    def raise_for_status(self):
        return None

    def json(self):
        return {"armed_request_seq": 17, "latency_s": 0.30}


class RecordingControlSession:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeControlResponse()


def test_smoke_arms_delay_through_cross_process_control_endpoint():
    session = RecordingControlSession()
    armed_seq = arm_next_policy_delay(
        "127.0.0.1", 22086, "smoke-control-token", 0.30,
        session=session,
    )
    assert armed_seq == 17
    assert session.calls == [(
        "http://127.0.0.1:22086/test-control/arm-next-delay",
        {
            "json": {"token": "smoke-control-token", "latency_s": 0.30},
            "timeout": 0.5,
        },
    )]


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
        10.0, client_factory=lambda *args, **kwargs: client,
        clock=lambda: 9.0,
    )
    assert result is payload
    assert client.timeouts == [1.0]


def test_smoke_wbc_preflight_rejects_a_nonexistent_wrapper_schema():
    client = StaticWbcConfigClient({
        "control_loop_config": {"env_type": "sim", "interface": "lo"}
    })
    with pytest.raises(SmokeSafetyError, match="sim/loopback"):
        default_wbc_preflight(
            10.0, client_factory=lambda *args, **kwargs: client,
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
        "--duration-s", "15", "--unitree-domain-id", "42",
        "--camera-port", "15555", "--policy-port", "22086",
    ]
    assert smoke_driver.main(argv) == 0
    assert smoke_driver.main(argv) == 0
    assert allocation_roots == ["outputs", "outputs"]
    assert [config.output_dir for config in launched] == [
        str(tmp_path / "run-1"), str(tmp_path / "run-2"),
    ]


def test_collector_uses_lowercase_fault_and_old_generation_counter():
    zero = np.zeros(36, np.float32).tolist()
    records = [
        {
            "event": "preflight_complete",
            "connection_evidence": [
                {
                    "component": "wbc", "transport": "dds-service-response",
                    "endpoint": "lo", "real_interface": False,
                    "observed_at": 0.5,
                },
                {
                    "component": "camera", "transport": "decoded-frame",
                    "endpoint": "127.0.0.1", "real_interface": False,
                    "observed_at": 0.6,
                },
                {
                    "component": "policy", "transport": "http-contract-response",
                    "endpoint": "127.0.0.1", "real_interface": False,
                    "observed_at": 0.7,
                },
            ],
            "real_interface_connections": 0,
        },
        {
            "event": "request", "request_seq": 17,
            "observation_tick": 100,
            "committed_actions": np.zeros((6, 36), np.float32).tolist(),
            "monotonic_s": 11.0,
        },
        {
            "event": "tick", "tick": 100, "published": True,
            "state": "active", "source_kind": "hold",
            "psi0_action": zero, "monotonic_s": 11.00,
            "worker_busy": True,
            "discarded_late_results": 0,
            "discarded_old_generation_results": 0,
        },
        {
            "event": "tick", "tick": 101, "published": True,
            "state": "fault", "source_kind": "hold",
            "psi0_action": zero, "monotonic_s": 11.02,
            "worker_busy": True,
            "discarded_late_results": 0,
            "discarded_old_generation_results": 1,
        },
        {
            "event": "tick", "tick": 199, "published": True,
            "state": "fault", "source_kind": "hold",
            "psi0_action": zero, "monotonic_s": 12.98,
            "worker_busy": False,
            "discarded_late_results": 0,
            "discarded_old_generation_results": 1,
        },
    ]
    assert measured_bridge_worker_idle_at(records, 0.0) == 12.98
    report = collect_smoke_report(
        records, scenario_started_at=0.0, armed_seq=17,
        delayed_record={"request_seq": 17, "applied_latency_s": 0.30},
        goal_counts_before=[0, 1],
        goal_counts_running=[1, 1], goal_counts_after=[0, 1],
        terminal_restored=True, ports_rebound=True, bridge_exit_code=0,
        live_children_after=[],
        child_exit_codes={
            "wbc": -2, "camera": 0, "policy": 0, "bridge": 0,
        },
    )
    assert report["fault_at"] == 11.02
    assert report["old_generation_results_discarded"] == 1
    assert report["late_results_discarded"] == 0
    records[0]["connection_evidence"][2]["real_interface"] = True
    with pytest.raises(SmokeSafetyError, match="differs from evidence"):
        measured_real_interface_connections(records)


@pytest.mark.parametrize(
    "updates",
    [
        {"wbc_interface": "eth0"}, {"camera_host": "0.0.0.0"},
        {"policy_host": "192.168.1.2"}, {"ros_domain_id": 41},
        {"unitree_domain_id": 0}, {"camera_port": 0},
        {"policy_port": 0},
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

Implement `SmokeConfig`, `ChildSpec`, `LaunchPlan`, and `build_launch_plan()`. The policy `ChildSpec` must pass the config's exact token/readiness path through `--control-token` and `--ready-json`; reject an empty token, a non-loopback host, a readiness path outside the configured run directory, or any existing artifact file. On every invocation, the production CLI atomically creates a new `psi0-smoke-<time_ns>-<nonce>` directory below `--output-dir`, prints that path, and never removes or overwrites an earlier run. It creates `control_token=secrets.token_hex(16)` once when no token is injected by a unit test and never writes it to metrics. Run all Step 1 tests. Do not call `Popen` in this checkbox.

```python
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

from scripts.psi0_simple_real_bridge import (  # noqa: E402
    CONTROL_GOAL_TOPIC, RosRuntime, count_real_interface_connections,
    create_ros_wbc_config_client,
)


class SmokeSafetyError(RuntimeError):
    pass


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
    if not config.control_token:
        raise SmokeSafetyError("fake-policy control token is required")
    output_paths = (
        config.policy_ready_json, config.camera_ready_json,
        config.bridge_metrics_jsonl, config.smoke_report_json,
        *(str(Path(config.output_dir) / f"psi0-smoke-{name}.log")
          for name in ("wbc", "camera", "policy", "bridge")),
    )
    if any(not _inside_output(path, config.output_dir) for path in output_paths):
        raise SmokeSafetyError("all artifacts must be inside output_dir")
    existing = [path for path in output_paths if Path(path).exists()]
    if existing:
        raise SmokeSafetyError(f"smoke artifact already exists: {existing[0]}")

    environment = dict(os.environ)
    environment.update({
        "ROS_DOMAIN_ID": "42", "UNITREE_DOMAIN_ID": "42",
        "PYTHONPATH": "third_party/decoupled_wbc",
    })
    env = MappingProxyType(environment)
    wbc = ChildSpec("wbc", (
        "uv", "run", "--group", "sonic", "python", "-m",
        "decoupled_wbc.control.main.teleop.run_g1_control_loop",
        "--interface", "sim", "--simulator", "mujoco",
        "--messaging-backend", "ros2", "--enable-waist", "--with-hands",
        "--domain-id", "42", "--no-enable-onscreen", "--no-enable-offscreen",
    ), env)
    camera = ChildSpec("camera", (
        "uv", "run", "--group", "sonic", "python",
        "scripts/tests/fake_composed_camera_server.py",
        "--host", config.camera_host, "--port", str(config.camera_port),
        "--ready-json", config.camera_ready_json,
    ), env, ready_json=config.camera_ready_json)
    policy = ChildSpec("policy", (
        "uv", "run", "--group", "sonic", "python",
        "scripts/tests/fake_psi0_rtc_server.py",
        "--host", config.policy_host, "--port", str(config.policy_port),
        "--contract", "scripts/tests/fixtures/psi0_policy_contract_test_v2.json",
        "--normal-latency-s", "0.05",
        "--control-token", config.control_token,
        "--ready-json", config.policy_ready_json,
    ), env, ready_json=config.policy_ready_json)
    bridge = ChildSpec("bridge", (
        "uv", "run", "--group", "sonic", "python",
        "scripts/psi0_simple_real_bridge.py", "--mode", "sim-control",
        "--server-host", config.policy_host,
        "--server-port", str(config.policy_port),
        "--instruction", "smoke test hold",
        "--policy-contract",
        "scripts/tests/fixtures/psi0_policy_contract_test_v2.json",
        "--camera-host", config.camera_host,
        "--camera-port", str(config.camera_port),
        "--camera-source-key", "rgb_head_stereo_left",
        "--camera-color-order", "rgb", "--ros-domain-id", "42",
        "--unitree-domain-id", "42",
        "--metrics-jsonl", config.bridge_metrics_jsonl,
    ), env, use_pty=True)
    return LaunchPlan(config, (wbc, camera, policy, bridge))
```

Add the complete control helper used by the timeline:

```python
def arm_next_policy_delay(host, port, token, latency_s, *, session=requests):
    if host != "127.0.0.1" or not token:
        raise SmokeSafetyError("fake-policy control must be authenticated loopback")
    response = session.post(
        f"http://{host}:{port}/test-control/arm-next-delay",
        json={"token": token, "latency_s": latency_s},
        timeout=0.5,
    )
    response.raise_for_status()
    payload = response.json()
    if type(payload) is not dict or set(payload) != {
        "armed_request_seq", "latency_s"
    }:
        raise SmokeSafetyError("malformed fake-policy control response")
    if type(payload["armed_request_seq"]) is not int:
        raise SmokeSafetyError("malformed armed request sequence")
    if type(payload["latency_s"]) is not float or payload["latency_s"] != latency_s:
        raise SmokeSafetyError("fake-policy delay acknowledgement mismatch")
    return payload["armed_request_seq"]
```

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
        killpg=killpg, clock=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    )
    for pid in (101, 202, 303):
        owner.record(
            pid=pid, pgid=pid, name=str(pid), argv=(str(pid),),
            started_at=0.0, process=Process(pid),
        )
    owner.close(deadline=1.0)
    assert now[0] <= 1.0
    assert terminated == {101, 202, 303}
    assert owner.live_pids() == []
    assert owner.exit_codes() == {"101": 0, "202": 0, "303": 0}
    assert [event[:2] for event in events] == [
        (303, signal.SIGINT), (202, signal.SIGINT), (101, signal.SIGINT),
        (303, signal.SIGTERM), (202, signal.SIGTERM),
        (101, signal.SIGTERM),
    ]


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
        killpg=killpg, clock=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    )
    for pid in (101, 202, 303):
        owner.record(
            pid=pid, pgid=pid, name=str(pid), argv=(str(pid),),
            started_at=0.0, process=Process(pid),
        )
    with pytest.raises(SmokeSafetyError, match="303"):
        owner.close(deadline=0.8)
    assert (202, signal.SIGINT) in events
    assert (101, signal.SIGINT) in events
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
            FailingOwner(), (10, 11), (Log("a"), Log("b")), 15.0,
            close_fd=lambda descriptor: events.append(("fd", descriptor)),
        )
    assert events == [
        ("owner", 15.0), ("fd", 10), ("fd", 11),
        ("log", "a"), ("log", "b"),
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
```

Use this exact owner and readiness helper. Launch each child with `start_new_session=True`, record it immediately, and walk only that list in reverse. `close(deadline=...)` catches failures per child and per operation, continues through every remaining process group, and signals each escalation phase to all still-live process groups before waiting concurrently against the same absolute deadline. It never grants a fresh timeout per child. A final SIGKILL pass still runs after an exhausted deadline or earlier cleanup error. The owner becomes closed only after no real child remains (and all injected waits succeeded), so a failed cleanup with a live child can be retried. The injection-only unit branch preserves the exact `waitpid` assertion above:

```python
@dataclass
class ChildRecord:
    pid: int
    pgid: int
    name: str
    argv: tuple[str, ...]
    started_at: float
    process: object | None = None


class OwnedChildren:
    def __init__(
        self, killpg=os.killpg, waitpid=os.waitpid,
        clock=time.monotonic, sleep=time.sleep,
    ):
        self._killpg = killpg
        self._waitpid = waitpid
        self._clock = clock
        self._sleep = sleep
        self.records = []
        self._closed = False
        self._closing = False

    def record(self, *, pid, pgid, name, argv, started_at, process=None):
        if (
            self._closed or self._closing
            or any(record.pid == pid for record in self.records)
        ):
            raise RuntimeError("invalid child ownership record")
        self.records.append(ChildRecord(
            pid, pgid, name, tuple(argv), float(started_at), process
        ))

    def _poll(self, record, errors):
        try:
            return record.process.poll()
        except Exception as error:
            errors.append(f"poll {record.name}({record.pid}): {error}")
            return None

    def _live(self, records, errors):
        return [
            record for record in records
            if self._poll(record, errors) is None
        ]

    def _signal(self, records, sig, errors):
        for record in records:
            try:
                self._killpg(record.pgid, sig)
            except ProcessLookupError:
                continue
            except Exception as error:
                errors.append(
                    f"signal {record.name}({record.pgid}) {sig}: {error}"
                )

    def _wait_live_until(self, records, deadline, errors):
        while True:
            live = self._live(records, errors)
            if not live or self._clock() >= deadline:
                return live
            self._sleep(min(0.01, deadline - self._clock()))

    def live_pids(self):
        return [
            record.pid for record in self.records
            if record.process is not None and record.process.poll() is None
        ]

    def exit_codes(self):
        return {
            record.name: record.process.poll()
            for record in self.records if record.process is not None
        }

    def close(self, deadline=None):
        if self._closed:
            return
        if self._closing:
            raise SmokeSafetyError("child cleanup is already in progress")
        self._closing = True
        errors = []
        injected = [record for record in reversed(self.records) if record.process is None]
        real = [record for record in reversed(self.records) if record.process is not None]
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
                if (
                    type(deadline) not in (int, float)
                    or not np.isfinite(deadline)
                ):
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

                # This last best-effort phase is unconditional, even if the
                # shared deadline or an earlier signal operation failed.
                live = self._live(real, errors)
                self._signal(live, signal.SIGKILL, errors)
                live = self._wait_live_until(live, deadline, errors)
                if live:
                    errors.append(
                        "live children: "
                        + ",".join(
                            f"{record.name}({record.pid})" for record in live
                        )
                    )
            self._closed = not live and injected_ok
        finally:
            self._closing = False
        if errors:
            raise SmokeSafetyError("child cleanup failed: " + "; ".join(errors))


def close_smoke_resources(
    owner, descriptors, logs, deadline, *, close_fd=os.close,
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
        raise SmokeSafetyError(
            "smoke resource cleanup failed: " + "; ".join(errors)
        )


def wait_ready_json(path, expected_host, expected_port, deadline, clock=time.monotonic):
    path = Path(path)
    while clock() < deadline:
        if path.exists():
            payload = json.loads(path.read_text())
            expected = {"host": expected_host, "port": expected_port}
            if payload != expected:
                raise SmokeSafetyError(f"readiness mismatch: {payload!r}")
            return payload
        time.sleep(min(0.01, max(0.0, deadline - clock())))
    raise TimeoutError(f"readiness timeout: {path}")
```

- [ ] **Step 4: Implement the timed smoke scenario**

Implement the timeline with these literal helpers and `launch()`; the injectable hooks keep the unit test free of process creation, while the CLI uses the defaults:

```python
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
            record.get("event") == "preflight_complete"
            for record in read_jsonl(path)
        ):
            return
        time.sleep(min(0.01, max(0.0, deadline - clock())))
    raise TimeoutError("bridge preflight readiness timeout")


def default_wbc_preflight(
    deadline, *, client_factory=create_ros_wbc_config_client,
    clock=time.monotonic,
):
    remaining = deadline - clock()
    if remaining <= 0.0:
        raise TimeoutError("WBC readiness deadline")
    client = client_factory(
        "WBCPolicy/robot_config", domain_id=42
    )
    payload = client.get_config(timeout_s=min(3.0, remaining))
    if type(payload) is not dict:
        raise SmokeSafetyError("WBC configuration payload must be an object")
    if payload.get("env_type") != "sim" or payload.get("interface") != "lo":
        raise SmokeSafetyError("WBC is not isolated sim/loopback")
    return payload


def default_goal_counts():
    runtime = RosRuntime(42)
    try:
        return list(runtime.counts(CONTROL_GOAL_TOPIC))
    finally:
        runtime.close(timeout_s=0.5)


def policy_status(config, session=requests):
    response = session.get(
        f"http://{config.policy_host}:{config.policy_port}/test-control/status",
        headers={"X-Test-Control-Token": config.control_token}, timeout=0.5,
    )
    response.raise_for_status()
    payload = response.json()
    if type(payload) is not dict or set(payload) != {
        "last_started_request_seq", "active_requests",
        "max_concurrent_requests", "records",
    }:
        raise SmokeSafetyError("malformed fake-policy status")
    return payload


def port_rebinds(port):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    return True


class DefaultSmokeHooks:
    clock = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)
    popen = staticmethod(subprocess.Popen)
    wbc_preflight = staticmethod(default_wbc_preflight)
    goal_counts = staticmethod(default_goal_counts)
    policy_status = staticmethod(policy_status)


def _spawn(spec, owner, log, *, stdin=None):
    process = subprocess.Popen(
        spec.argv, env=dict(spec.env), stdin=stdin, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    owner.record(
        pid=process.pid, pgid=os.getpgid(process.pid), name=spec.name,
        argv=spec.argv, started_at=time.monotonic(), process=process,
    )
    return process


def measured_real_interface_connections(records):
    preflight = [
        record for record in records
        if record.get("event") == "preflight_complete"
    ]
    if len(preflight) != 1:
        raise SmokeSafetyError("expected one preflight connection record")
    record = preflight[0]
    evidence = record.get("connection_evidence")
    if (
        type(evidence) is not list
        or not all(type(item) is dict for item in evidence)
        or {item.get("component") for item in evidence}
        != {"wbc", "camera", "policy"}
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
        record for record in records
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
    records, *, scenario_started_at, armed_seq, delayed_record,
    goal_counts_before, goal_counts_running, goal_counts_after,
    terminal_restored, ports_rebound, bridge_exit_code,
    live_children_after, child_exit_codes,
):
    ticks = [record for record in records if record.get("event") == "tick"]
    publishes = [record for record in ticks if record.get("published") is True]
    fault_ticks = [record for record in ticks if record.get("state") == "fault"]
    if not fault_ticks:
        raise SmokeSafetyError("metrics contain no lowercase fault tick")
    fault_at = fault_ticks[0]["monotonic_s"] - scenario_started_at
    requests_ = [
        {"tick": record["observation_tick"],
         "committed_actions": record["committed_actions"]}
        for record in records if record.get("event") == "request"
    ]
    executed = [
        {"tick": record["tick"], "post_slew_action": record["psi0_action"]}
        for record in publishes if record.get("psi0_action") is not None
    ]
    request_record = next(
        record for record in records
        if record.get("event") == "request"
        and record.get("request_seq") == armed_seq
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
            {"publish_times": before_fault}, {"publish_times": after_fault},
        ],
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
        "old_generation_results_discarded": (
            final_tick["discarded_old_generation_results"]
        ),
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
        "live_threads_after": [
            thread.name for thread in threading.enumerate()
            if thread is not threading.current_thread()
            and not thread.daemon and thread.is_alive()
        ],
        "terminal_restored": terminal_restored,
        "ports_rebound": ports_rebound,
        "real_interface_connections": measured_real_interface_connections(
            records
        ),
        "extra_goal_publishers": max(0, goal_counts_running[0] - 1),
    }


def launch(config, hooks=None):
    plan = build_launch_plan(config)  # all safety checks precede Popen
    hooks = hooks or DefaultSmokeHooks()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    owner = OwnedChildren(clock=hooks.clock, sleep=hooks.sleep)
    logs = []
    bridge_master = bridge_slave = None
    bridge_original = None
    report = None
    smoke_deadline = None
    try:
        def start(spec, stdin=None):
            log = (output / f"psi0-smoke-{spec.name}.log").open("x")
            logs.append(log)
            process = hooks.popen(
                spec.argv, env=dict(spec.env), stdin=stdin, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            owner.record(
                pid=process.pid, pgid=process.pid, name=spec.name,
                argv=spec.argv, started_at=hooks.clock(), process=process,
            )
            return process

        start(plan.child("wbc"))
        hooks.wbc_preflight(hooks.clock() + 3.0)
        goal_counts_before = hooks.goal_counts()
        if goal_counts_before != [0, 1]:
            raise SmokeSafetyError(f"WBC preflight graph: {goal_counts_before}")

        start(plan.child("camera"))
        wait_ready_json(
            config.camera_ready_json, config.camera_host, config.camera_port,
            hooks.clock() + 1.0, hooks.clock,
        )
        start(plan.child("policy"))
        wait_ready_json(
            config.policy_ready_json, config.policy_host, config.policy_port,
            hooks.clock() + 1.0, hooks.clock,
        )

        bridge_master, bridge_slave = pty.openpty()
        bridge_original = termios.tcgetattr(bridge_slave)
        bridge_process = start(plan.child("bridge"), stdin=bridge_slave)
        wait_bridge_preflight(
            config.bridge_metrics_jsonl, hooks.clock() + 3.0, hooks.clock
        )
        scenario_started_at = hooks.clock()
        smoke_deadline = scenario_started_at + 15.0
        sleep_until(scenario_started_at + 3.0, hooks.clock, hooks.sleep)
        os.write(bridge_master, b"p")
        goal_counts_running = hooks.goal_counts()
        if goal_counts_running != [1, 1]:
            raise SmokeSafetyError(f"bridge graph ownership: {goal_counts_running}")

        sleep_until(scenario_started_at + 11.0, hooks.clock, hooks.sleep)
        armed_seq = arm_next_policy_delay(
            config.policy_host, config.policy_port, config.control_token, 0.30
        )
        sleep_until(scenario_started_at + 13.0, hooks.clock, hooks.sleep)
        status = hooks.policy_status(config)
        if status["active_requests"] != 0:
            raise SmokeSafetyError("fake policy still has an active request")
        matches = [
            record for record in status["records"]
            if record == {"request_seq": armed_seq, "applied_latency_s": 0.30}
        ]
        if len(matches) != 1:
            raise SmokeSafetyError("one-shot delay did not reach armed request")
        os.killpg(os.getpgid(bridge_process.pid), signal.SIGINT)
        bridge_exit_code = bridge_process.wait(timeout=max(
            0.0, smoke_deadline - hooks.clock()
        ))
        goal_counts_after = hooks.goal_counts()
        if goal_counts_after != [0, 1]:
            raise SmokeSafetyError(f"publisher cleanup graph: {goal_counts_after}")
        terminal_restored = termios.tcgetattr(bridge_slave) == bridge_original
        records = read_jsonl(config.bridge_metrics_jsonl)
        report_args = (
            records, scenario_started_at, armed_seq, matches[0],
            goal_counts_before, goal_counts_running, goal_counts_after,
            terminal_restored, bridge_exit_code,
        )
    finally:
        cleanup_deadline = (
            smoke_deadline if smoke_deadline is not None
            else hooks.clock() + 1.0
        )
        close_smoke_resources(
            owner, (bridge_master, bridge_slave), logs, cleanup_deadline
        )

    ports_rebound = all(
        port_rebinds(port) for port in (config.camera_port, config.policy_port)
    )
    report = collect_smoke_report(
        report_args[0], scenario_started_at=report_args[1],
        armed_seq=report_args[2], delayed_record=report_args[3],
        goal_counts_before=report_args[4],
        goal_counts_running=report_args[5], goal_counts_after=report_args[6],
        terminal_restored=report_args[7], ports_rebound=ports_rebound,
        bridge_exit_code=report_args[8],
        live_children_after=owner.live_pids(),
        child_exit_codes=owner.exit_codes(),
    )
    validation = validate_smoke_report(report)
    report["validation"] = {
        "ok": validation.ok, "failures": list(validation.failures)
    }
    Path(config.smoke_report_json).write_text(json.dumps(report, indent=2))
    if not validation.ok:
        raise SmokeSafetyError("; ".join(validation.failures))
    if smoke_deadline is None or hooks.clock() > smoke_deadline:
        raise SmokeSafetyError("smoke cleanup exceeded shared second-15 deadline")
    return report


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", required=True, type=float)
    parser.add_argument("--unitree-domain-id", required=True, type=int)
    parser.add_argument("--camera-port", required=True, type=int)
    parser.add_argument("--policy-port", required=True, type=int)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--control-token", default=None, help=argparse.SUPPRESS)
    return parser


def allocate_smoke_run_directory(
    output_root, *, now_ns=time.time_ns, token_hex=secrets.token_hex,
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
    token = args.control_token or secrets.token_hex(16)
    config = SmokeConfig(
        duration_s=args.duration_s, ros_domain_id=42,
        unitree_domain_id=args.unitree_domain_id, wbc_interface="lo",
        camera_host="127.0.0.1", camera_port=args.camera_port,
        policy_host="127.0.0.1", policy_port=args.policy_port,
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
```

The implementation is divided into these bounded assertions:

- [ ] Preflight `0 publishers/1 subscription` after WBC readiness and before bridge launch.
- [ ] Launch each child through the tested `OwnedChildren` registry.
- [ ] Wait for each named bounded readiness signal.
- [ ] Send local `p` through the bridge pseudo-terminal at second 3.
- [ ] At second 11 call `arm_next_policy_delay(...)`, store the acknowledged `armed_request_seq`, and fail the smoke immediately on timeout/non-202/schema mismatch.
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

After the second-13 worker-idle check, query `GET /test-control/status` with the same token and require exactly one record whose `request_seq` equals the acknowledged sequence and whose `applied_latency_s` is exactly `0.30`. This is the cross-process proof that the intended request, rather than an arbitrary later request, received the injected delay.

- [ ] **Step 5: Write and implement exact metric assertions**

Append a generated passing report and one-field mutation matrix:

The armed 0.30-second response returns only after the handoff underrun has latched `FAULT` and advanced the bridge generation. Therefore the only valid accounting is exactly one `old_generation_results_discarded` and zero `late_results_discarded`; the latter counter is reserved for a response drained after its deadline but before a generation transition.

```python
def passing_smoke_report():
    publish_times = [3.0 + index * 0.02 for index in range(400)]
    executed_actions = []
    by_tick = {}
    for tick in range(100, 172):
        action = np.zeros(36, np.float32)
        action[31] = 0.5
        action[32] = (tick - 136) / 1000.0
        by_tick[tick] = action
        executed_actions.append({
            "tick": tick, "post_slew_action": action.tolist()
        })
    return {
        "steady_phases": [{"publish_times": publish_times[:200]},
                          {"publish_times": publish_times[200:]}],
        "requests": [
            {"tick": tick, "committed_actions": np.stack(
                [by_tick[index] for index in range(tick, tick + 6)]
            ).tolist()}
            for tick in (100, 124, 148)
        ],
        "executed_actions": executed_actions,
        "blocked_main_loop_max_gap_s": 0.02,
        "delayed_request_started_at": 11.0,
        "armed_request_seq": 17,
        "delayed_request_record": {
            "request_seq": 17, "applied_latency_s": 0.30,
        },
        "fault_at": 11.12,
        "first_fault_goal_navigation": [0.0, 0.0, 0.0, 0.0],
        "policy_actions_after_fault": 0,
        "old_generation_results_discarded": 1,
        "late_results_discarded": 0,
        "worker_idle_at_s": 12.8,
        "goal_counts_before": [0, 1],
        "goal_counts_running": [1, 1],
        "goal_counts_after": [0, 1],
        "live_children_after": [],
        "child_exit_codes": {
            "wbc": -2, "camera": 0, "policy": 0, "bridge": 0,
        },
        "bridge_exit_code": 0,
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
        ("delayed_request_record", {"request_seq": 18, "applied_latency_s": 0.30}),
        ("fault_at", 11.141),
        ("first_fault_goal_navigation", [0.0, 0.0, 0.1, 0.0]),
        ("policy_actions_after_fault", 1),
        ("old_generation_results_discarded", 0),
        ("late_results_discarded", 1),
        ("worker_idle_at_s", 13.01),
        ("goal_counts_after", [1, 1]),
        ("live_children_after", [123]),
        ("bridge_exit_code", 1),
        ("child_exit_codes", {
            "wbc": -2, "camera": 0, "policy": None, "bridge": 0,
        }),
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
    skipped["executed_actions"] = [
        entry for entry in skipped["executed_actions"] if entry["tick"] != 130
    ]
    mutations.append((skipped, "executed_actions"))
    mismatch = passing_smoke_report()
    mismatch["requests"][1]["committed_actions"][2][0] = 0.01
    mutations.append((mismatch, "committed_actions"))
    for report, expected in mutations:
        result = validate_smoke_report(report)
        assert result.ok is False
        assert any(expected in failure for failure in result.failures)
```

`validate_smoke_report()` is implemented literally below. In particular, every request's `(6,36)` committed prefix is compared to the actual post-slew action published for each corresponding global tick:

```python
@dataclass(frozen=True)
class SmokeValidation:
    ok: bool
    failures: tuple[str, ...]


def validate_smoke_report(report):
    failures = []

    phases = report.get("steady_phases")
    if type(phases) is not list or len(phases) != 2:
        failures.append("steady_phases: expected two phases")
    else:
        for index, phase in enumerate(phases):
            times = np.asarray(phase.get("publish_times", []), np.float64)
            if len(times) < 2 or not np.isfinite(times).all():
                failures.append(f"steady_phases[{index}]: timestamps")
                continue
            gaps = np.diff(times)
            hz = (len(times) - 1) / (times[-1] - times[0])
            if not 49.0 <= hz <= 51.0:
                failures.append(f"steady_phases[{index}]: rate")
            if np.max(gaps) > 0.060:
                failures.append(f"maximum_gap: steady_phases[{index}]")

    executed = report.get("executed_actions")
    executed_by_tick = {}
    if type(executed) is not list:
        failures.append("executed_actions: expected list")
    else:
        for entry in executed:
            if type(entry) is not dict or set(entry) != {"tick", "post_slew_action"}:
                failures.append("executed_actions: record schema")
                continue
            tick = entry["tick"]
            action = np.asarray(entry["post_slew_action"], np.float32)
            if type(tick) is not int or action.shape != (36,) or not np.isfinite(action).all():
                failures.append("executed_actions: record value")
                continue
            if tick in executed_by_tick:
                failures.append("executed_actions: duplicate tick")
            executed_by_tick[tick] = action
        ticks = sorted(executed_by_tick)
        if ticks and ticks != list(range(ticks[0], ticks[-1] + 1)):
            failures.append("executed_actions: discontinuous ticks")

    requests_ = report.get("requests")
    if type(requests_) is not list or len(requests_) < 2:
        failures.append("requests: missing")
    else:
        request_ticks = [entry.get("tick") for entry in requests_]
        if any(type(tick) is not int for tick in request_ticks):
            failures.append("requests: tick type")
        elif any(b - a != 24 for a, b in zip(request_ticks, request_ticks[1:])):
            failures.append("request_spacing: expected 24 ticks")
        for entry in requests_:
            tick = entry.get("tick")
            committed = np.asarray(entry.get("committed_actions"), np.float32)
            if committed.shape != (6, 36) or not np.isfinite(committed).all():
                failures.append("committed_actions: expected finite (6,36)")
                continue
            if type(tick) is not int or any(
                global_tick not in executed_by_tick
                for global_tick in range(tick, tick + 6)
            ):
                failures.append("committed_actions: executed tick missing")
                continue
            actual = np.stack([
                executed_by_tick[global_tick]
                for global_tick in range(tick, tick + 6)
            ])
            if not np.array_equal(committed, actual):
                failures.append("committed_actions: differs from post-slew execution")

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
        if report.get(field) != expected:
            failures.append(f"{field}: expected {expected!r}")
    child_exit_codes = report.get("child_exit_codes")
    if (
        type(child_exit_codes) is not dict
        or set(child_exit_codes) != {"wbc", "camera", "policy", "bridge"}
        or any(type(code) is not int for code in child_exit_codes.values())
    ):
        failures.append("child_exit_codes: every owned child must be reaped")
    if report.get("blocked_main_loop_max_gap_s", float("inf")) > 0.060:
        failures.append("blocked_main_loop_max_gap_s: over 0.060")
    started = report.get("delayed_request_started_at")
    fault_at = report.get("fault_at")
    if type(started) not in (int, float) or type(fault_at) not in (int, float):
        failures.append("fault_at: missing timestamp")
    elif fault_at > started + 0.14:
        failures.append("fault_at: latency exceeds 0.14 seconds")
    if report.get("worker_idle_at_s", float("inf")) > 13.0:
        failures.append("worker_idle_at_s: exceeds second 13")
    expected_delay = {
        "request_seq": report.get("armed_request_seq"),
        "applied_latency_s": 0.30,
    }
    if report.get("delayed_request_record") != expected_delay:
        failures.append("delayed_request_record: acknowledgement mismatch")
    return SmokeValidation(not failures, tuple(failures))
```

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

Expected: exit zero with a metrics summary containing all approved smoke pass criteria. Record the printed unique run directory; its metrics JSONL, report, and logs are immutable inputs to Task 16. Re-running this exact command allocates another directory and preserves the first. This initializes Unitree SDK2 DDS only on loopback/domain 42; it does not contact a robot.

## Task 14: Document operations and the external live-server handoff

**Files:**
- Modify: `README.md`
- Test: `tests/test_psi0_bridge_cli.py`

- [ ] **Step 1: Add CLI contract tests**

Create `tests/test_psi0_bridge_cli.py` with no runtime imports beyond the parser/factories:

```python
from types import SimpleNamespace

import pytest

from scripts.psi0_simple_real_bridge import (
    LocalKeyboard, PreflightError, RuntimeDependencyFactories, build_parser,
    build_policy_client, build_runtime_adapters, load_local_policy_payload,
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
        "robot_interface", "network_interface", "low_level_topic",
        "real_control", "dds_topic",
    }
    assert parser.get_default("policy_contract") is None


def test_only_sim_control_is_locked_to_isolated_domain_42():
    assert validate_domain_selection(BridgeMode.SIM_CONTROL, 42, 42) == (
        42, 42
    )
    with pytest.raises(PreflightError, match="isolated domain 42"):
        validate_domain_selection(BridgeMode.SIM_CONTROL, 7, 9)
    assert validate_domain_selection(BridgeMode.SHADOW, 7, 9) == (7, 9)
    with pytest.raises(PreflightError, match=r"\[0,232\]"):
        validate_domain_selection(BridgeMode.SHADOW, -1, 9)


def test_shadow_does_not_require_or_unconditionally_read_local_contract(
    tmp_path,
):
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
```

None of these test doubles imports or opens ROS, ZMQ, DDS, or a network socket.

- [ ] **Step 2: Add the README deployment section**

Document:

- this milestone is simulation/shadow only;
- PC2 inputs, 43→32 and 36→31+1+4 named mappings;
- WBC command with `--interface sim --enable-waist --with-hands --domain-id 42`;
- fake server/camera commands and the bridge shadow/sim-control commands;
- startup PAUSED, local `p` transitions, FAULT rearm behavior, Ctrl-C conditional final hold, and WBC ownership after bridge exit;
- `--policy-contract` is mandatory for `sim-control` but optional in `shadow`; shadow uses a valid server contract as uncertified runtime structure when the local file is missing, and becomes observation-only if neither contract is usable;
- domain 42/loopback is mandatory for `sim-control` and its smoke test, while shadow accepts explicit domain IDs for observation of a real system without ever creating a publisher;
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
from tests.psi0_bridge_testkit import make_joint_contract, policy_payload


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
        joint_contract=make_joint_contract(),
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
        joint_contract=make_joint_contract(),
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
        joint_contract=make_joint_contract(),
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


def test_bundle_refuses_hash_mismatch_and_existing_destination(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    dataset_manifest = tmp_path / "dataset.json"
    samples = tmp_path / "samples.npz"
    checkpoint.write_bytes(b"checkpoint")
    dataset_manifest.write_bytes(b"dataset")
    write_representative_npz(samples, representative_samples(100))
    payload = live_policy_payload(
        checkpoint_sha256="0" * 64,
        dataset_manifest_sha256=sha256_file(dataset_manifest),
    )
    contract_path = tmp_path / "policy-contract.json"
    contract_path.write_text(json.dumps(payload))
    report = BenchmarkReport(
        10, 100, 100, {key: 0 for key in FAILURE_KEYS},
        0.05, 0.10, True,
    )
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="checkpoint hash"):
        write_certification_bundle(
            output_dir=output, report=report,
            policy_contract_path=contract_path,
            fetched_server_contract=payload,
            checkpoint_path=checkpoint,
            dataset_manifest_path=dataset_manifest,
            samples_path=samples,
        )
    assert not output.exists()
    output.mkdir()
    with pytest.raises(FileExistsError):
        write_certification_bundle(
            output_dir=output, report=report,
            policy_contract_path=contract_path,
            fetched_server_contract=payload,
            checkpoint_path=checkpoint,
            dataset_manifest_path=dataset_manifest,
            samples_path=samples,
        )
```

The bundle writer first verifies the checkpoint and dataset hashes equal the contract, verifies the fetched server contract equals the local contract, writes all seven files into a new temporary sibling directory, fsyncs, then atomically renames that directory to the requested new output path. It refuses an existing output path and never overwrites evidence.

- [ ] **Step 3: Implement representative request loading and response classification**

Require an NPZ with arrays `images` (`N,H,W,3`, contiguous `uint8`), `states` (`N,1,32`, finite `float32`), and `instructions` (`N` strings), with `N >= 100`. Define the wire/result types and exact classifier as follows:

```python
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from urllib.parse import urlparse
import uuid

import numpy as np
import requests

from decoupled_wbc.control.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from simple.baselines.client import (
    RequestMessage, convert_numpy_in_dict, numpy_deserialize,
)
from simple.deploy.psi0_simple_bridge import (
    JointContract, PolicyContract, PSI0_ACTION_JOINT_NAMES,
    validate_action_suffix,
)


FAILURE_KEYS = (
    "timeout", "http", "decode", "shape", "metadata", "bounds", "late"
)


class BenchmarkRequestError(RuntimeError):
    def __init__(self, kind, detail=""):
        if kind not in FAILURE_KEYS:
            raise ValueError(f"unknown benchmark failure kind: {kind}")
        super().__init__(detail or kind)
        self.kind = kind


@dataclass(frozen=True)
class BenchmarkSample:
    image: np.ndarray
    state: np.ndarray
    instruction: str


@dataclass(frozen=True)
class BenchmarkTransportResponse:
    action: np.ndarray
    metadata: dict[str, object]
    latency_s: float


@dataclass(frozen=True)
class BenchmarkReport:
    warmup_requests: int
    measured_requests: int
    successes: int
    failures: dict[str, int]
    p99_latency_s: float
    latency_limit_s: float
    certified: bool


def load_representative_samples(path):
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"images", "states", "instructions"}:
            raise ValueError("sample NPZ keys must be images/states/instructions")
        images = payload["images"]
        states = payload["states"]
        instructions = payload["instructions"]
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("images must be contiguous uint8 (N,H,W,3)")
    if not images.flags.c_contiguous:
        raise ValueError("images must be contiguous")
    if states.dtype != np.float32 or states.shape != (len(images), 1, 32):
        raise ValueError("states must be float32 (N,1,32)")
    if not np.isfinite(states).all():
        raise ValueError("states must be finite")
    if instructions.shape != (len(images),) or instructions.dtype.kind not in "US":
        raise ValueError("instructions must be an N-element string array")
    if len(images) < 100:
        raise ValueError("at least 100 representative samples are required")
    return tuple(
        BenchmarkSample(images[i].copy(), states[i].copy(), str(instructions[i]))
        for i in range(len(images))
    )


def expected_metadata(history, contract):
    return {
        "session_id": history["session_id"],
        "request_seq": history["request_seq"],
        "observation_tick": history["observation_tick"],
        "prediction_horizon": contract.prediction_horizon,
        "execution_horizon": contract.execution_horizon,
        "rtc_delay_steps": contract.rtc_delay_steps,
        "first_action_tick": (
            history["observation_tick"] + contract.rtc_delay_steps
        ),
    }


def classify_response(response, history, contract, joint_contract):
    if type(response) is not BenchmarkTransportResponse:
        raise BenchmarkRequestError("decode", "transport response type")
    action = np.asarray(response.action)
    expected_shape = (contract.execution_horizon, contract.action_dim)
    if action.shape != expected_shape or action.dtype != np.float32:
        raise BenchmarkRequestError("shape", f"expected {expected_shape} float32")
    metadata = response.metadata
    expected = expected_metadata(history, contract)
    if type(metadata) is not dict or set(metadata) != set(expected):
        raise BenchmarkRequestError("metadata", "metadata key set")
    for key, value in expected.items():
        required_type = str if key == "session_id" else int
        if type(metadata[key]) is not required_type or metadata[key] != value:
            raise BenchmarkRequestError("metadata", key)
    try:
        validate_action_suffix(action, contract.execution_horizon, joint_contract)
    except ValueError as error:
        raise BenchmarkRequestError("bounds", str(error)) from error
    if (
        type(response.latency_s) is not float
        or not np.isfinite(response.latency_s)
        or response.latency_s > contract.rtc_delay_steps / contract.action_frequency_hz
    ):
        raise BenchmarkRequestError("late", "RTC response missed r+d")
    return response.latency_s
```

The concrete HTTP transport catches `requests.Timeout` as `timeout`, calls `raise_for_status()` and catches `requests.HTTPError` as `http`, and catches JSON/NumPy deserialization or response-construction errors as `decode`; it re-raises an existing `BenchmarkRequestError` unchanged. This assigns transport failures before the response priority `shape → metadata → bounds → late`, so one malformed response always has exactly one category.

Fetch `/contract` exactly once before warmup, parse it with `PolicyContract.from_dict()`, and require full dataclass equality with the local contract. Each request uses dataset `simple`, empty condition, exactly `{contract.image_key: sample.image}`, exactly `{"states": sample.state}`, `gt_action=[]`, R0-only reset, monotonically increasing sequence/observation ticks, and a finite exact `(d,36)` committed prefix.

Use this complete concrete transport:

```python
class HttpBenchmarkTransport:
    def __init__(
        self, server_url, image_key, timeout_s=5.0,
        session=requests, clock=time.monotonic,
    ):
        parsed = urlparse(server_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("server_url must be an absolute HTTP(S) URL")
        if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
            raise ValueError("server_url must not contain path/query/fragment")
        if type(image_key) is not str or not image_key:
            raise ValueError("image_key must be a non-empty string")
        self.base_url = server_url.rstrip("/")
        self.image_key = image_key
        self.timeout_s = timeout_s
        self.session = session
        self.clock = clock
        self.fetched_contract = None

    def get_contract(self):
        try:
            response = self.session.get(
                self.base_url + "/contract", timeout=self.timeout_s
            )
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout as error:
            raise BenchmarkRequestError("timeout", str(error)) from error
        except requests.HTTPError as error:
            raise BenchmarkRequestError("http", str(error)) from error
        except Exception as error:
            raise BenchmarkRequestError("decode", str(error)) from error
        if type(payload) is not dict:
            raise BenchmarkRequestError("decode", "contract response type")
        self.fetched_contract = payload
        return payload

    def query(self, request_index, sample, history):
        del request_index
        request = RequestMessage(
            image={self.image_key: sample.image},
            instruction=sample.instruction,
            history=history,
            state={"states": sample.state},
            condition={}, gt_action=[], dataset_name="simple",
            timestamp=str(time.time_ns()),
        )
        started = self.clock()
        try:
            response = self.session.post(
                self.base_url + "/act-rtc-v1",
                json=request.serialize(), timeout=self.timeout_s,
            )
            response.raise_for_status()
            payload = convert_numpy_in_dict(response.json(), numpy_deserialize)
            if type(payload) is not dict or set(payload) != {"action", "metadata"}:
                raise ValueError("RTC response key set")
            result = BenchmarkTransportResponse(
                action=payload["action"], metadata=payload["metadata"],
                latency_s=float(self.clock() - started),
            )
        except BenchmarkRequestError:
            raise
        except requests.Timeout as error:
            raise BenchmarkRequestError("timeout", str(error)) from error
        except requests.HTTPError as error:
            raise BenchmarkRequestError("http", str(error)) from error
        except Exception as error:
            raise BenchmarkRequestError("decode", str(error)) from error
        return result
```

- [ ] Add NPZ key/dtype/shape/finite validation and run loader cases.
- [ ] Add strict metadata construction/comparison and run metadata mutations.
- [ ] Add shape/bounds/late classification in priority order.
- [ ] Add HTTP exception mapping and prove every attempt yields one category.

- [ ] **Step 4: Implement warmed measurement and higher-method p99**

Implement the full loop below. A warmup failure aborts without creating any bundle; measured failures are counted and the loop continues without retry. The constant committed prefix is a safe stationary input contract, not a prediction reconstructed after a failed response.

```python
def benchmark_server(
    *, transport, samples, contract, joint_contract,
    warmup_requests=10, measured_requests=100,
):
    if warmup_requests != 10 or measured_requests != 100:
        raise ValueError("certification requires 10 warmups and 100 measured requests")
    if len(samples) < 100:
        raise ValueError("at least 100 representative samples are required")
    fetched = PolicyContract.from_dict(transport.get_contract())
    if fetched != contract:
        raise ValueError("server policy contract differs from local contract")

    joint_index = {
        name: index for index, name in enumerate(joint_contract.joint_names)
    }
    action_joint_names = (
        *PSI0_ACTION_JOINT_NAMES,
        "waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint",
    )
    if any(name not in joint_index for name in action_joint_names):
        raise ValueError("joint contract cannot build stationary prefix")
    stationary = np.zeros(36, np.float32)
    for action_index, name in enumerate(action_joint_names):
        model_index = joint_index[name]
        stationary[action_index] = np.clip(
            0.0,
            joint_contract.lower_position_limits[model_index],
            joint_contract.upper_position_limits[model_index],
        )
    stationary[31] = 0.74
    committed = np.repeat(
        stationary[None], contract.rtc_delay_steps, axis=0
    )
    session_id = "benchmark-" + uuid.uuid4().hex

    def make_history(request_index):
        history = {
            "session_id": session_id,
            "request_seq": request_index,
            "observation_tick": request_index * contract.execution_horizon,
            "rtc_delay_steps": contract.rtc_delay_steps,
            "committed_actions": committed.copy(),
        }
        if request_index == 0:
            history["reset"] = True
        return history

    for request_index in range(warmup_requests):
        sample = samples[request_index % len(samples)]
        history = make_history(request_index)
        try:
            response = transport.query(request_index, sample, history)
            classify_response(response, history, contract, joint_contract)
        except BenchmarkRequestError as error:
            raise RuntimeError(
                f"warmup request {request_index} failed: {error.kind}: {error}"
            ) from error

    failures = {key: 0 for key in FAILURE_KEYS}
    success_latencies = []
    for measured_index in range(measured_requests):
        request_index = warmup_requests + measured_index
        sample = samples[request_index % len(samples)]
        history = make_history(request_index)
        try:
            response = transport.query(request_index, sample, history)
            latency = classify_response(
                response, history, contract, joint_contract
            )
        except BenchmarkRequestError as error:
            failures[error.kind] += 1
        else:
            success_latencies.append(latency)

    p99_latency_s = (
        float(np.quantile(success_latencies, 0.99, method="higher"))
        if success_latencies else float("inf")
    )
    latency_limit_s = (
        contract.rtc_delay_steps - 1
    ) / contract.action_frequency_hz
    certified = (
        len(success_latencies) == measured_requests
        and sum(failures.values()) == 0
        and p99_latency_s <= latency_limit_s
    )
    return BenchmarkReport(
        warmup_requests, measured_requests, len(success_latencies), failures,
        p99_latency_s, latency_limit_s, certified,
    )
```

Because measured request indices are 10–109, `samples[request_index % 100]` visits samples 10–99 then 0–9 exactly once. Return nonzero from the CLI unless `certified` is true; a failed run may write `latency_report.json` to a separately named diagnostic directory, but must not write a certified bundle.

- [ ] Add exact contract fetch/equality and stationary committed-prefix creation.
- [ ] Add 10 warmups with abort-on-first-failure behavior.
- [ ] Add 100 measured attempts with no retry and exact failure accounting.
- [ ] Add higher-method p99/certification and run all gate mutations.

- [ ] **Step 5: Implement the evidence writer and exact CLI**

Implement the evidence writer, local effective-limit loader, parser, and entry point exactly:

```python
def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_fsynced(path, data):
    payload = data.encode("utf-8") if isinstance(data, str) else data
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_certification_bundle(
    *, output_dir, report, policy_contract_path, fetched_server_contract,
    checkpoint_path, dataset_manifest_path, samples_path,
):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    if type(report) is not BenchmarkReport or not report.certified:
        raise ValueError("only a certified report may create an evidence bundle")
    if (
        report.warmup_requests != 10
        or report.measured_requests != 100
        or report.successes != 100
        or report.failures != {key: 0 for key in FAILURE_KEYS}
    ):
        raise ValueError("certification report counters do not meet the gate")

    local_payload = json.loads(Path(policy_contract_path).read_text())
    local_contract = PolicyContract.from_dict(local_payload)
    server_contract = PolicyContract.from_dict(fetched_server_contract)
    if local_contract != server_contract or local_payload != fetched_server_contract:
        raise ValueError("fetched server contract differs from local contract")
    if local_contract.test_only:
        raise ValueError("test-only policy contract cannot be live-certified")
    checkpoint_hash = sha256_file(checkpoint_path)
    dataset_hash = sha256_file(dataset_manifest_path)
    sample_hash = sha256_file(samples_path)
    if checkpoint_hash != local_contract.checkpoint_sha256:
        raise ValueError("checkpoint hash differs from policy contract")
    if dataset_hash != local_contract.dataset_manifest_sha256:
        raise ValueError("dataset manifest hash differs from policy contract")
    for field in ("server_commit", "converter_commit"):
        if re.fullmatch(r"[0-9a-f]{40}", getattr(local_contract, field)) is None:
            raise ValueError(f"{field} must be a full lowercase Git SHA")

    latency_payload = asdict(report)
    latency_payload["p99_method"] = "higher"
    files = {
        "latency_report.json": (
            json.dumps(latency_payload, sort_keys=True, separators=(",", ":"))
            + "\n"
        ),
        "policy_contract.json": (
            json.dumps(local_payload, sort_keys=True, separators=(",", ":"))
            + "\n"
        ),
        "checkpoint.sha256": checkpoint_hash + "\n",
        "dataset_manifest.sha256": dataset_hash + "\n",
        "request_samples.sha256": sample_hash + "\n",
        "server_commit.txt": local_contract.server_commit + "\n",
        "converter_commit.txt": local_contract.converter_commit + "\n",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.", dir=output.parent
    ))
    try:
        for name, data in files.items():
            _write_fsynced(temporary / name, data)
        _fsync_directory(temporary)
        os.rename(temporary, output)
        _fsync_directory(output.parent)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def build_local_joint_contract():
    model = instantiate_g1_robot_model(
        waist_location="lower_and_upper_body"
    )
    names = tuple(model.joint_names)
    upper_indices = model.get_joint_group_indices("upper_body")
    upper_names = tuple(names[index] for index in upper_indices)
    if len(names) != 43 or len(upper_names) != 31:
        raise RuntimeError("local G1 model does not expose 43/31 joints")
    return JointContract(
        names, upper_names, model.lower_joint_limits.copy(),
        model.upper_joint_limits.copy(),
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--policy-contract", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--representative-samples", required=True)
    parser.add_argument("--warmup-requests", required=True, type=int)
    parser.add_argument("--measured-requests", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.warmup_requests != 10 or args.measured_requests != 100:
        raise SystemExit("live certification requires exactly 10 warmups/100 measured")
    contract_payload = json.loads(Path(args.policy_contract).read_text())
    contract = PolicyContract.from_dict(contract_payload)
    if contract.test_only:
        raise SystemExit("test-only contracts cannot be live-certified")
    samples = load_representative_samples(args.representative_samples)
    transport = HttpBenchmarkTransport(
        args.server_url, image_key=contract.image_key, timeout_s=5.0
    )
    report = benchmark_server(
        transport=transport, samples=samples, contract=contract,
        joint_contract=build_local_joint_contract(),
        warmup_requests=args.warmup_requests,
        measured_requests=args.measured_requests,
    )
    if not report.certified:
        print(json.dumps(asdict(report), sort_keys=True))
        return 1
    write_certification_bundle(
        output_dir=args.output_dir, report=report,
        policy_contract_path=args.policy_contract,
        fetched_server_contract=transport.fetched_contract,
        checkpoint_path=args.checkpoint,
        dataset_manifest_path=args.dataset_manifest,
        samples_path=args.representative_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

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
uv run --group dev ruff check \
  third_party/decoupled_wbc/control/main/model_contract.py \
  third_party/decoupled_wbc/control/main/teleop/run_g1_control_loop.py \
  third_party/decoupled_wbc/tests/control/main/test_model_contract.py \
  third_party/decoupled_wbc/tests/control/main/teleop/test_g1_control_loop_contract.py \
  third_party/decoupled_wbc/tests/control/robot_model/test_g1_effective_limits.py
uv run --group dev ruff format --check \
  third_party/decoupled_wbc/control/main/model_contract.py \
  third_party/decoupled_wbc/control/main/teleop/run_g1_control_loop.py \
  third_party/decoupled_wbc/tests/control/main/test_model_contract.py \
  third_party/decoupled_wbc/tests/control/main/teleop/test_g1_control_loop_contract.py \
  third_party/decoupled_wbc/tests/control/robot_model/test_g1_effective_limits.py
git diff --check main...HEAD
git -C third_party/decoupled_wbc diff --check
```

Expected: every command exits zero.

- [ ] **Step 4: Re-run the isolated 15-second smoke test**

Run the exact Task 13 command. It atomically allocates and prints a new timestamp-and-nonce run directory beneath `outputs`; require that path to differ from the Task 13 path, preserve both directories unchanged, and use the new directory's metrics JSONL as the verification artifact. Expected: all pass criteria and zero real-interface connections. If allocation collides or any artifact already exists in the newly allocated directory, the driver fails closed instead of deleting or overwriting data.

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
| WBC partial-construction rollback before service publication | Task 5 |
| Exact named 43→32 and 36→31+1+4 mapping | Task 7 |
| RGB/BGR codec and viewpoint key contract | Tasks 10, 12 |
| State bounds, freshness, skew, and last-valid behavior | Task 8 |
| Bounded holds and no-hold startup/shutdown | Tasks 8, 11 |
| R0 full RTC metadata and committed hold prefix | Task 9 |
| 103/108 sentinel handoffs and post-slew prefix | Task 9 |
| Whole-suffix bounds and per-tick slew limits | Task 8 |
| PAUSED/ACTIVE/FAULT/STOPPED and local `p` | Tasks 9, 11 |
| Nonfatal local activation refusal and later retry | Tasks 9, 11 |
| Single asynchronous worker, deadlines, late generation discard | Tasks 9, 11-12 |
| Publisher ownership and shadow no-publisher guarantee | Task 10 |
| Optional shadow contract, observation-only fallback, and selectable domains | Tasks 10-11, 14 |
| Exact 25-message shutdown and no 26th message | Task 11 |
| Short smoke stall and separate long shutdown test | Tasks 11-13 |
| Isolated loopback/domain-42 MuJoCo smoke test | Task 13 |
| Measured worker idleness, bridge exit, and reaped child processes | Task 13 |
| Live gate: 10 warmups, 100 measured, failure counts, higher-method p99, evidence bundle | Task 15 |
| Reachable nested commit and clean recursive checkout | Tasks 6, 16 |
| No real-control path and documented later gates | Tasks 0, 10, 14-16 |

## Execution checkpoints

1. Stop after Task 5 if the nested tests do not pass.
2. Stop at Task 6 until explicit push authority and a fetchable remote are available.
3. Stop before Task 13 if ROS graph ownership is not exactly `0 publishers/1 WBC subscription`, if any interface is not loopback, or if either domain is not 42.
4. Stop before replacing the fake policy server unless the exact Task 15 command produces a certified seven-file bundle: matching `/act-rtc-v1` contract/commits/hashes, 10 discarded warmups, 100/100 successful measured requests, zero categorized failures, and higher-method p99 at or below `(d-1)/50` seconds.
5. Completing this plan does not authorize real-robot deployment. That remains a separate reviewed design and implementation phase.
