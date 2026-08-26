# Dedicated PSI0 H100 Evaluation Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and certify the simulation-only runtime defined by the approved dedicated PSI0 H100 design, ending with official SIMPLE episodes 2 and 3 running against the protected step-40000 checkpoint while every H100, PC2, evaluator, relay, video, and cleanup artifact is attributable.

**Architecture:** Python code owns strict schemas, canonical manifests, lifecycle orchestration, GPU/process attribution, and immutable evidence. Five statically linked native executables own the security boundaries Python cannot provide: H100 publication, PC2 publication, detached remote helpers, the private-root evaluator supervisor, and the connected-FD policy relay. The existing evaluator gains one indivisible managed six-FD contract and remains backward compatible when that contract is absent; the deferred third-person camera remains out of scope.

**Tech Stack:** Python 3.10, Typer, pytest, NumPy, OpenCV, FFmpeg/FFprobe, Rust 1.75 with vendored crates and static CRT, systemd socket activation, Linux namespaces/cgroups/pidfds/openat2/renameat2/SCM_RIGHTS, Docker, SSH, NVIDIA SMI/NVML, MuJoCo, Isaac Sim, and the existing SIMPLE PSI0 client.

---

## Scope, source identity, and execution rules

The authoritative design is
`docs/superpowers/specs/2026-08-24-dedicated-psi0-h100-evaluation-runtime-design.md`
at design commit `0a3ad85029c5994cd017624e260276a3694a1b35`. Every task below is simulation-only. Do not launch Docker, SSH, H100 inference, Isaac, MuJoCo, DDS, or a robot until the task explicitly names the corresponding staged integration gate.

The implementation uses a two-commit provenance boundary:

- source commit `I` contains every implementation, unit test, native binary source, service/config file, and documentation change, but not the active runtime profile;
- approval commit `P` has first parent `I` and changes exactly
  `configs/psi0_h100_eval_runtime_v1.json`;
- all managed execution loads source from the sealed tree of `I` and profile bytes from `P`.

Never stage, remove, rotate, or copy the preserved untracked `outputs/` tree. Before every commit run `git status --short` and require that the task's listed files plus `?? outputs/` are the only entries.

## File and ownership map

| Path | Responsibility |
| --- | --- |
| `src/simple/eval_runtime/contracts.py` | Exact enums, JSON key sets, dataclasses, Boolean/type rejection, ID/path validation |
| `src/simple/eval_runtime/canonical.py` | Canonical JSON/JSONL, SHA-256, tree manifests, hash chains, atomic evidence writes |
| `src/simple/eval_runtime/processes.py` | Monotonic deadlines, PID/start-tick identity, bounded INT→TERM→KILL, command audit |
| `src/simple/eval_runtime/profile.py` | Commit `I`/`P` verification and strict 87-key runtime-profile loading |
| `src/simple/eval_runtime/pc2_closure.py` | Git-object closure construction, base-Python intake, write-ahead owner records, recovery generations |
| `src/simple/eval_runtime/assets.py` | Offline task-asset requirements, data-root seam, USD/Sdf HSSD normalization and probes |
| `src/simple/eval_runtime/installer_client.py` | Framed Unix/SCM_RIGHTS calls, locked-FD handoff, authorization/result transitions |
| `src/simple/eval_runtime/installer_manager.py` | Live handed-off OFD ownership, fail-closed reconciliation, and disk-backed restart planning |
| `src/simple/eval_runtime/remote.py` | Attested detached-helper launch/reconciliation over SSH |
| `src/simple/eval_runtime/lease.py` | Remote lease CAS, heartbeat, mutation ordering, stale/recovery claims |
| `src/simple/eval_runtime/gpu.py` | H100/PC2 UUID resolution, process attribution, CUDA probes, five-second monitors |
| `src/simple/eval_runtime/container.py` | Digest-pinned stopped-container create/reuse/inspect and protected tracer/server lifecycle |
| `src/simple/eval_runtime/runner_client.py` | Exact runner operations and FD-role schemas |
| `src/simple/eval_runtime/events.py` | Runtime-contract/event/ack schemas, pipe framing, order validation |
| `src/simple/eval_runtime/warmup.py` | Production `RequestMessage` warm-up and byte-exact digest |
| `src/simple/eval_runtime/evidence.py` | Immutable run tree, artifact hashes, verdicts, Markdown rendering |
| `src/simple/eval_runtime/manager.py` | Persist-before-action state machine and reverse-order cleanup |
| `src/simple/eval_runtime/cli.py` | `freeze-provenance`, `create`, `status`, `evaluate`, `stop`, and stale recovery |
| `native/psi0_eval_runtime/` | Static Rust workspace shared by the five native executables |
| `deploy/psi0_eval/` | Hashed systemd socket/service units and root-owned configuration templates |
| `configs/psi0_h100_eval_seccomp_v1.json` | Reviewed container seccomp profile included in the H100 source snapshot |
| `configs/psi0_h100_eval_runtime_v1.json` | Generated candidate installed only in approval commit `P` |
| `src/simple/cli/eval_decoupled_wbc.py` | Managed evaluator FD contract, durable WBC evidence, ordered events, video path routing |
| `src/simple/baselines/client.py` | Connected-FD request transport with no endpoint/reconnect fallback |
| `src/simple/baselines/psi0_decoupled_wbc.py` | Managed policy-FD construction path |
| `src/simple/envs/video_writer.py` | Exclusive raw creation and checked bounded transcode |
| `src/simple/envs/wrappers/video_recorder.py` | Deferred start, no overwrite, close-all structured finalization |
| `src/simple/scenes/hssd.py` and `src/simple/utils.py` | Sealed offline data root and shell-free normalized HSSD loading |
| `tests/eval_runtime/` | Simulator-free unit, contract, crash, race, protocol, and fake-boundary tests |
| `tests/integration/eval_runtime/` | Explicit staged system integration gates; never collected by the unit-test command |

## Fixed public Python interfaces

Later tasks must use these names and signatures exactly:

| Module | Fixed symbols |
| --- | --- |
| `contracts.py` | `RuntimeBlocked`, `TerminalState`, `RuntimeProfile`, `ProcessIdentity`, `OwnedDeadline`, `validate_identifier`, `validate_relative_path`, `parse_runtime_profile` |
| `canonical.py` | `canonical_json_bytes`, `sha256_bytes`, `canonical_tree_manifest`, `atomic_write_new_json`, `append_hash_chained_jsonl` |
| `manager.py` | `DedicatedPsi0Runtime.freeze_provenance`, `.create`, `.status`, `.evaluate`, `.stop` |
| `runner_client.py` | `RunnerClient.prepare_output`, `.loader_probe`, `.cuda_probe`, `.evaluate_episode`, `.finalize_episode`, `.recover_episode` |

The concrete signatures and bodies appear in the task that first creates each module. Later tasks must import those symbols rather than defining a second version.

The native wire protocol is canonical UTF-8 JSON preceded by one unsigned 32-bit big-endian length. A request contains exactly `schema_version`, `operation`, `request_id`, `profile_sha256`, `payload`, and `fd_roles`; a response contains exactly `schema_version`, `request_id`, `status`, `payload`, and `error`. The terminal installer response transfers exactly one still-locked, `FD_CLOEXEC` transaction-lock descriptor by `SCM_RIGHTS`. The manager persists the required authorization/result transition before closing that descriptor; EOF has no state-transition meaning.

## Global test commands

Unit tasks use this environment and must remain simulator-, SSH-, Docker-, and real-interface-free:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv/bin/python -m pytest -q -p no:cacheprovider tests/eval_runtime
```

Native tasks use the vendored dependency tree only:

```bash
CARGO_NET_OFFLINE=true cargo test --locked \
  --manifest-path native/psi0_eval_runtime/Cargo.toml
```

Static checks use the complete implementation file list stored in
`tests/eval_runtime/static_files.txt`; they do not use a stale hand-written subset:

```bash
xargs -a tests/eval_runtime/static_files.txt ruff check --no-cache
xargs -a tests/eval_runtime/static_files.txt ruff format --check --no-cache
.venv/bin/python -m compileall -q src/simple/eval_runtime \
  src/simple/cli/eval_decoupled_wbc.py src/simple/baselines \
  src/simple/envs src/simple/scenes
git diff --check 0a3ad85029c5994cd017624e260276a3694a1b35..HEAD
```

### Task 0: Freeze the baseline and create simulator-free test scaffolding

**Files:**
- Create: `tests/eval_runtime/__init__.py`
- Create: `tests/eval_runtime/conftest.py`
- Create: `tests/eval_runtime/fakes.py`
- Create: `tests/eval_runtime/static_files.txt`
- Create: `tests/integration/eval_runtime/README.md`

- [ ] **Step 1: Record the approved baseline without launching a runtime**

Run:

```bash
plan_head="$(git rev-parse HEAD)"
remote_head="$(git rev-parse origin/feature/psi0-simple-pc2-bridge)"
test "$plan_head" = "$remote_head"
printf '%s\n' "$plan_head"
git merge-base --is-ancestor 0a3ad85029c5994cd017624e260276a3694a1b35 HEAD
git status --short
git submodule status --recursive
git diff --check
```

Expected: both revisions are the same current plan commit;
`0a3ad85029c5994cd017624e260276a3694a1b35` is an ancestor and remains the immutable design-range base. Tracked output is empty; the only untracked path is `outputs/`; every submodule has a leading space; whitespace check is silent. Do not require `HEAD` itself to equal the design commit because execution begins from the subsequently reviewed plan commit.

- [ ] **Step 2: Add deterministic fixtures used by every contract test**

```python
# tests/eval_runtime/conftest.py
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "native: permits only reviewed local native test drivers"
    )


def digest(byte: bytes = b"x") -> str:
    return hashlib.sha256(byte).hexdigest()


def write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


@dataclass
class ManualClock:
    value: float = 1000.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@dataclass
class RecordingRunner:
    results: list[object] = field(default_factory=list)
    calls: list[tuple[list[str], float]] = field(default_factory=list)

    def run(self, argv: list[str], *, deadline: float, **_: Any) -> object:
        self.calls.append((list(argv), deadline))
        if not self.results:
            raise AssertionError(f"unexpected command: {argv!r}")
        return self.results.pop(0)


@pytest.fixture
def manual_clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def isolated_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    return root


@pytest.fixture
def fake_image_env():
    import gymnasium as gym
    import numpy as np

    class ImageEnv(gym.Env):
        observation_space = gym.spaces.Dict(
            {
                "head_stereo_left": gym.spaces.Box(0, 255, (36, 64, 3), np.uint8),
                "head_stereo_right": gym.spaces.Box(0, 255, (36, 64, 3), np.uint8),
            }
        )
        action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        _success = False

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return {
                "head_stereo_left": np.zeros((36, 64, 3), np.uint8),
                "head_stereo_right": np.zeros((36, 64, 3), np.uint8),
            }, {}

        def step(self, action):
            observation, _ = self.reset()
            return observation, 0.0, False, False, {}

    return ImageEnv()


@pytest.fixture(autouse=True)
def forbid_external_runtime(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    real_popen = subprocess.Popen
    repo = Path(__file__).parents[2].resolve()
    release = (
        repo / "native/psi0_eval_runtime/target/x86_64-unknown-linux-gnu/release"
    ).resolve()
    native_names = {
        "psi0-eval-install-input", "psi0-eval-install-pc2-input",
        "psi0-eval-remote-helper", "psi0-eval-run-pc2-evaluator",
        "psi0-eval-policy-relay",
    }

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"external runtime forbidden in unit tests: {args!r}")

    def constrained_popen(argv: object, *args: object, **kwargs: object):
        marked = request.node.get_closest_marker("native") is not None
        if not marked or not isinstance(argv, (list, tuple)) or not argv:
            return blocked(argv, *args, **kwargs)
        words = [os.fspath(item) for item in argv]
        cwd = Path(kwargs.get("cwd", repo)).resolve()
        if words[:2] == ["readelf", "-lWd"] and len(words) == 3:
            binary = Path(words[2]).resolve(strict=True)
            if binary.parent == release and binary.name in native_names:
                return real_popen(argv, *args, **kwargs)
        if words == ["cargo", "metadata", "--offline", "--locked", "--format-version", "1"]:
            if cwd == (repo / "native/psi0_eval_runtime").resolve():
                return real_popen(argv, *args, **kwargs)
        try:
            binary = Path(words[0]).resolve(strict=True)
        except FileNotFoundError:
            return blocked(argv, *args, **kwargs)
        if binary.parent == release and binary.name in native_names and "--test-root" in words:
            return real_popen(argv, *args, **kwargs)
        return blocked(argv, *args, **kwargs)

    monkeypatch.setattr("socket.create_connection", blocked)
    monkeypatch.setattr("subprocess.Popen", constrained_popen)
    monkeypatch.setenv("SIMPLE_EVAL_RUNTIME_UNIT_TEST", "1")
```

Create shared fakes with explicit state rather than magic mocks. These fakes are imported by the literal tests in later tasks:

```python
# tests/eval_runtime/fakes.py
from __future__ import annotations

import json
import os
import socket
import array
import fcntl
import hashlib
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


class InstallerHandoffHarness:
    def __init__(self, *, boundary: str):
        self.boundary = boundary
        self.duplicate_acquired = False
        self.persisted = False
        self.closed = False

    def queue_duplicate(self) -> None:
        self.duplicate_acquired = False

    def advance_to_boundary(self) -> None:
        self.duplicate_acquired = False

    def manager_persist_transition(self) -> None:
        self.persisted = True

    def manager_close_handoff(self) -> None:
        assert self.persisted
        self.closed = True

    def one_duplicate_acquires_after_revalidation(self) -> bool:
        self.duplicate_acquired = self.persisted and self.closed
        return self.duplicate_acquired


class RecoveryHarness:
    def __init__(self, *, crash: str | None = None):
        self.crash = crash
        self.install_request_bytes = b'{"operation":"install_base_python","token":"fixed"}'
        self.replayed_install_bytes = b""
        self.install_reference = "install:1"
        self._authorizations: list[SimpleNamespace] = []
        self._results: list[SimpleNamespace] = []
        self._journal = False
        self.root_journal_accesses = 0
        self.final_digest_paths_examined: list[str] = []
        self.unrelated_protected_objects_preserved = True
        self.manager_journal_writes = 0

    def append_recovery_generation(self) -> SimpleNamespace:
        number = len(self._authorizations) + 1
        previous = (
            self._authorizations[-1].authorization_sha256
            if self._authorizations else None
        )
        entry = SimpleNamespace(
            recovery_generation=number,
            active_reference=f"recovery:{number}",
            previous_entry_sha256=previous,
            request_digest=f"{number:064x}",
            authorization_sha256=f"{number + 100:064x}",
        )
        self._authorizations.append(entry)
        return entry

    def recover(self, generation: SimpleNamespace) -> SimpleNamespace:
        self.root_journal_accesses += 1
        previous = self._results[-1].result_sha256 if self._results else None
        result_entry = SimpleNamespace(
            recovery_generation=generation.recovery_generation,
            previous_result_sha256=previous,
            result_sha256=f"{generation.recovery_generation + 200:064x}",
        )
        self._results.append(result_entry)
        if self.crash == "mutation_child_before_prepared":
            return SimpleNamespace(status="ABORTED_BEFORE_PREPARED", result=result_entry)
        return SimpleNamespace(
            status="RECOVERED" if self._journal else "NO_JOURNAL_NO_MUTATION",
            result=result_entry,
        )

    def reactivate_install(self, generation: SimpleNamespace) -> None:
        assert generation is self._authorizations[-1]
        self.replayed_install_bytes = self.install_request_bytes

    def service_create_journal_then_drop_transport(self) -> None:
        self._journal = True

    def reject_install_against(self, reference: str) -> str:
        return "AUTHORIZATION_KIND" if reference.startswith("recovery:") else "accepted"

    def reject_recovery_against(self, reference: str) -> str:
        return "AUTHORIZATION_KIND" if reference.startswith("install:") else "accepted"


class ForkedInstallerService:
    def __init__(self, root: Path, pid: int):
        self.root = root
        self.pid = pid
        self.socket_path = root / "installer.sock"

    @classmethod
    def start(
        cls,
        root: Path,
        *,
        transaction_lock: Path,
        script: tuple[str, ...] | None = None,
    ) -> "ForkedInstallerService":
        root.mkdir(mode=0o700)
        transaction_lock.touch(mode=0o600)
        request_log = root / "requests.jsonl"
        response_log = root / "responses.jsonl"
        request_log.touch(mode=0o600)
        response_log.touch(mode=0o600)
        ready_read, ready_write = os.pipe2(os.O_CLOEXEC)
        pid = os.fork()
        if pid == 0:
            os.close(ready_read)
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(root / "installer.sock")
            listener.listen(4)
            os.write(ready_write, b"1")
            os.close(ready_write)
            install_count = 0
            journal_created = False
            script_index = 0
            try:
                while True:
                    connection, _ = listener.accept()
                    with connection:
                        body = cls._receive_request(connection)
                        if body == b"__stop__":
                            break
                        request = json.loads(body)
                        if script is not None:
                            response_status = script[
                                min(script_index, len(script) - 1)
                            ]
                            script_index += 1
                        elif request["operation"] == "install_base_python":
                            install_count += 1
                            response_status = (
                                "TRANSPORT_DROPPED"
                                if install_count == 1
                                else "TRANSPORT_DROPPED_AFTER_JOURNAL"
                            )
                        elif request["operation"] == "recover_base_python":
                            generation = request["recovery_generation"]
                            response_status = (
                                "NO_JOURNAL_NO_MUTATION"
                                if generation == 1 and not journal_created
                                else "RECOVERED"
                            )
                        else:
                            raise RuntimeError("INSTALLER_TEST_OPERATION")
                        cls._append_jsonl(
                            request_log,
                            {"body_hex": body.hex()},
                        )
                        cls._append_jsonl(
                            response_log,
                            {"status": response_status},
                        )
                        if response_status == "TRANSPORT_DROPPED":
                            continue
                        if response_status == "TRANSPORT_DROPPED_AFTER_JOURNAL":
                            journal = root / "service-journal.json"
                            journal.write_bytes(b'{"phase":"PREPARED"}\n')
                            with journal.open("rb") as stream:
                                os.fsync(stream.fileno())
                            cls._fsync_dir(root)
                            journal_created = True
                            continue
                        try:
                            cls._send_response(
                                connection,
                                transaction_lock=transaction_lock,
                                request=body,
                                status=response_status,
                            )
                        except BrokenPipeError:
                            continue
            finally:
                listener.close()
                os._exit(0)
        os.close(ready_write)
        if os.read(ready_read, 1) != b"1":
            raise RuntimeError("INSTALLER_TEST_SERVICE_START")
        os.close(ready_read)
        return cls(root, pid)

    @staticmethod
    def _receive_request(connection: socket.socket) -> bytes:
        header = ForkedInstallerService._receive_exact(connection, 4)
        length = struct.unpack("!I", header)[0]
        if length <= 0 or length > 1_048_576:
            raise RuntimeError("INSTALLER_TEST_REQUEST_LENGTH")
        return ForkedInstallerService._receive_exact(connection, length)

    @staticmethod
    def _receive_exact(connection: socket.socket, count: int) -> bytes:
        value = bytearray()
        while len(value) < count:
            chunk = connection.recv(count - len(value))
            if not chunk:
                raise RuntimeError("INSTALLER_TEST_REQUEST_EOF")
            value.extend(chunk)
        return bytes(value)

    @staticmethod
    def _send_response(
        connection: socket.socket,
        *,
        transaction_lock: Path,
        request: bytes,
        status: str,
    ) -> None:
        request_id = hashlib.sha256(request).hexdigest()
        payload = json.dumps(
            {
                "schema_version": 1,
                "request_id": request_id,
                "status": status,
                "payload": {},
                "error": None,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        lock_fd = os.open(
            transaction_lock,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        rights = array.array("i", [lock_fd])
        connection.sendmsg(
            [struct.pack("!I", len(payload)), payload],
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)],
        )
        os.close(lock_fd)

    @staticmethod
    def _append_jsonl(path: Path, value: dict[str, str]) -> None:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        ForkedInstallerService._fsync_dir(path.parent)

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def wait(self, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid == self.pid:
                if os.waitstatus_to_exitcode(status) != 0:
                    raise RuntimeError("INSTALLER_TEST_SERVICE_EXIT")
                return
            time.sleep(0.01)
        raise TimeoutError("INSTALLER_TEST_SERVICE_WAIT")

    def stop(self) -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(self.socket_path)
            connection.sendall(struct.pack("!I", 8) + b"__stop__")

    def request_bytes(self) -> list[bytes]:
        return [
            bytes.fromhex(json.loads(line)["body_hex"])
            for line in (self.root / "requests.jsonl").read_text().splitlines()
        ]

    def responses(self) -> list[str]:
        return [
            json.loads(line)["status"]
            for line in (self.root / "responses.jsonl").read_text().splitlines()
        ]


class FakeRemoteTransport:
    def __init__(self, *, disconnect_after: str | None = None, child_hangs: bool = False):
        self.disconnect_after = disconnect_after
        self.child_hangs = child_hangs
        self.launch_count = 0
        self.read_only_reconciliations = 0

    def launch(self, **request: Any) -> SimpleNamespace:
        self.launch_count += 1
        if self.disconnect_after == "detached_ready":
            raise ConnectionError("transport lost after durable acknowledgement")
        if self.child_hangs:
            return SimpleNamespace(
                state="timed_out", signal_sequence=["INT", "TERM", "KILL"],
                post_kill_alive=False, daemon_postcondition="exited",
            )
        return SimpleNamespace(state="completed")

    def reconcile(self, helper_id: str) -> SimpleNamespace:
        self.read_only_reconciliations += 1
        return SimpleNamespace(state="completed")


class RunnerHarness:
    def __init__(self, *, crash: str | None = None):
        self.crash = crash
        self.children: list[str] = []
        self.output_root_creations = 0
        self.cross_child_visibility_attempts: list[str] = []
        self._terminal: set[int] = set()

    def prepare_output(self) -> None:
        self.output_root_creations += 1

    def loader_probe(self) -> None:
        self.children.append("probe_loader")

    def cuda_probe(self) -> None:
        self.children.append("probe_cuda")

    def evaluate_episode(self, episode: int) -> None:
        if episode == 3 and 2 not in self._terminal:
            raise RuntimeError("PREVIOUS_EPISODE_NOT_TERMINAL")
        self.children.append(f"episode_{episode}")

    def finalize_episode(self, episode: int) -> None:
        self._terminal.add(episode)

    def recover_episode(self, episode: int, *, mode: str) -> SimpleNamespace:
        assert mode in {"live-manager", "stale-manager"}
        if self.crash == "foreign_cgroup_member":
            return SimpleNamespace(status="FOREIGN_BLOCKED")
        self._terminal.add(episode)
        return SimpleNamespace(
            status="RECOVERED", terminal=True, cgroup_members=(),
            unix_socket_inodes=(), upstream_socket_inode=None, foreign_signals=[],
        )


class FakeRelay:
    def __init__(self, relay_socket: socket.socket, *, action_shape: tuple[int, int]):
        self.socket = relay_socket
        self.action_shape = action_shape
        self.requests: list[int] = []
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self) -> None:
        self._thread.join(timeout=2.0)
        assert not self._thread.is_alive()

    def _serve(self) -> None:
        from simple.baselines.client import ResponseMessage, recv_policy_frame, send_policy_frame
        sequence, request = recv_policy_frame(self.socket.fileno())
        self.requests.append(sequence)
        response = ResponseMessage(np.zeros(self.action_shape, np.float32), 0.0).serialize()
        send_policy_frame(self.socket.fileno(), sequence, response)
        self.socket.close()


class ManagedEvaluatorHarness:
    def __init__(self, root: Path):
        self.root = root
        self.sonic_config = {"ENV_TYPE": "sim", "INTERFACE": "lo", "DOMAIN_ID": 0}
        self.events: list[dict[str, Any]] = []
        self.order: list[str] = []
        self.gym_make_calls = 0
        self.options = SimpleNamespace(run_id="run-a", episode_index=2, nonce="a" * 32)
        self.ops = self
        self._record: dict[str, Any] = {}

    @classmethod
    def valid(cls) -> "ManagedEvaluatorHarness":
        import tempfile
        return cls(Path(tempfile.mkdtemp(prefix="managed-evaluator-test-")))

    def write_record(self, value: dict[str, Any]) -> None:
        self._record = value
        self.order.append("evidence_fsync")

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append({"event": event, "payload": payload})
        if event == "runtime_contract":
            self.order.append("runtime_contract_write")
        elif event == "worker_init" and payload.get("status") == "creating_env":
            self.order.append("creating_env_write")

    def read_ack(self, timeout: float) -> dict[str, Any]:
        self.order.append("ack_read")
        return {"schema_version": 1, "accepted_sequence": 1}

    def make_record(self, config: dict[str, Any], *, status: str, error: str | None) -> dict[str, Any]:
        return {"schema_version": 1, "config": dict(config), "status": status, "error": error}

    def durable_record(self) -> dict[str, Any]:
        return self._record


class FakeVideoOps:
    def __init__(self, *, failure: str | None = None):
        self.failure = failure
        self.now = 100.0
        self.cleanup_complete = False

    def clock(self) -> float:
        return self.now

    def open_writer(self, path: Path, framerate: float, resolution: tuple[int, int]):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)

        class Writer:
            def write(self, image: np.ndarray) -> None:
                os.write(fd, b"frame")

            def release(self) -> None:
                os.fsync(fd)
                os.close(fd)

        return Writer()

    def transcode(self, raw: Path, temporary: Path, deadline: float) -> None:
        if self.failure in {"missing", "timeout", "nonzero", "malformed"}:
            temporary.write_bytes(b"diagnostic")
            self.cleanup_complete = True
            raise RuntimeError(self.failure)
        temporary.write_bytes(raw.read_bytes() + b"h264")

    def probe(self, path: Path) -> SimpleNamespace:
        if self.failure == "probe":
            self.cleanup_complete = True
            raise RuntimeError("probe")
        return SimpleNamespace(codec="h264", width=640, height=360, frame_count=1, duration=0.1)

    def publish(self, temporary: Path, final: Path) -> None:
        os.link(temporary, final)
        temporary.unlink()
        self.cleanup_complete = True


@dataclass
class ManagerHarness:
    dependencies: Any
    request: Any
    timeline: list[str] = field(default_factory=list)
    states: list[Any] = field(default_factory=list)
    started_episodes: list[int] = field(default_factory=list)

    @classmethod
    def success(cls, *, task_results: dict[int, bool] | None = None) -> "ManagerHarness":
        backend = SimpleNamespace(task_results=task_results or {2: True, 3: True})
        harness = cls(backend, SimpleNamespace(episodes=(2, 3)))
        backend.harness = harness
        return harness

    @classmethod
    def failure(cls, phase: str) -> "ManagerHarness":
        harness = cls.success()
        harness.dependencies.failure_phase = phase
        return harness


class CleanupHarness:
    def __init__(self, *, expired: bool):
        self.expired = expired
        self.calls: list[str] = []

    def close(self) -> None:
        self.calls.extend(["restore_terminal", "close_logs", "finalize_manifest"])


class FakeCliBackend:
    def __init__(self):
        self.mutations: list[str] = []
        self.actions: list[str] = []
        self.integration_calls: list[tuple[str, str | None]] = []
        self.integration_profile_approved = True
        self.retirement_state = "failed"
        self.retirement_calls: list[tuple[str, Path]] = []
        self.signals: list[int] = []

    def local_preflight(self) -> None:
        self.actions.append("local_preflight")

    def acquire_lease(self) -> None:
        self.actions.append("lease")

    def leased_preflight(self) -> None:
        self.actions.append("leased_preflight")

    def release_lease(self) -> None:
        self.actions.append("release_lease")

    def run_integration_fixture(
        self, operation: str, fixture: str | None
    ) -> dict[str, object]:
        self.integration_calls.append((operation, fixture))
        return {"schema_version": 1, "marker": operation}

    def require_integration_profile_binding(self) -> None:
        if not self.integration_profile_approved:
            raise RuntimeError("INTEGRATION_PROFILE_BINDING")

    def recover_owned_run(self, run_id: str, run_root: Path) -> None:
        from simple.eval_runtime.evidence import EvidenceStore

        self.retirement_calls.append((run_id, run_root))
        store = EvidenceStore.open_existing(run_root)
        if self.retirement_state == "foreign":
            raise RuntimeError("FOREIGN_BLOCKED")
        if self.retirement_state == "nonterminal":
            return
        if self.retirement_state == "unknown":
            finalize_failed_evidence(
                store,
                run_id=run_id,
                unknown_remote_state=True,
            )
            return
        if self.retirement_state == "pass":
            populate_complete_evidence(store, episodes=(2, 3))
            store.finalize_terminal_manifest(
                run_id=run_id,
                episodes=(2, 3),
                infrastructure_verdict="PASS",
                overall_verdict="PASS",
                task_results={"episode_2": True, "episode_3": True},
                owned_processes_alive=[],
                unknown_remote_state=False,
                unresolved_errors=[],
            )
            return
        finalize_failed_evidence(store, run_id=run_id)


def make_construction_record(*, phase: str):
    from simple.eval_runtime.pc2_closure import ConstructionRecord

    return ConstructionRecord.from_dict(
        {
            "schema_version": 1,
            "construction_attempt": 1,
            "attempt_status": "active",
            "operation_token_sha256": "a" * 64,
            "phase": phase,
            "cleanup_required": phase != "COMPLETE",
            "pending_action": None,
        }
    )


def make_lease_store(root: Path):
    from simple.eval_runtime.lease import LeaseStore

    now = [1_000_000_000]

    def clock_ns() -> int:
        now[0] += 1
        return now[0]

    return LeaseStore(root=root / "remote-control", clock_ns=clock_ns)


def make_expired_lease_store(root: Path, *, operation: str, cleanup_required: bool):
    from simple.eval_runtime.lease import LeaseStore

    store = LeaseStore(root=root / "remote-control", clock_ns=lambda: 100_000_000_000)
    store.install_expired_test_record(
        run_id="expired-run",
        operation=operation,
        cleanup_required=cleanup_required,
        heartbeat_monotonic_ns=1,
    )
    return store


def make_docker_inspect(*, state: str = "exited"):
    from simple.eval_runtime.container import DockerInspect

    return DockerInspect.from_dict(
        {
            "state": state,
            "image_id": "sha256:" + "a" * 64,
            "network_mode": "bridge",
            "published_ports": {},
            "mounts": [
                {"source": "/protected/inputs", "destination": "/inputs", "read_only": True},
                {"source": "/workloads/run-a", "destination": "/runtime", "read_only": False},
            ],
            "cap_drop": ["ALL"],
            "cap_add": ["SYS_PTRACE"],
            "no_new_privileges": True,
        }
    )


def make_container_contract():
    from simple.eval_runtime.container import ContainerContract

    return ContainerContract.from_dict(
        {
            "image_id": "sha256:" + "a" * 64,
            "network_mode": "bridge",
            "allow_published_ports": False,
            "required_readonly_destinations": ["/inputs"],
            "required_writable_destinations": ["/runtime"],
            "cap_drop": ["ALL"],
            "cap_add": ["SYS_PTRACE"],
            "no_new_privileges": True,
        }
    )


def make_runner_request(*, operation: str, fd_roles: list[str]):
    from simple.eval_runtime.runner_client import RunnerRequest

    return RunnerRequest.from_dict(
        {
            "schema_version": 1,
            "run_id": "run-a",
            "operation": operation,
            "operation_token_sha256": "a" * 64,
            "episode_index": 2 if "episode" in operation else None,
            "fd_roles": fd_roles,
        }
    )


def make_sandbox_manifest():
    from simple.eval_runtime.runner_client import SandboxManifest

    return SandboxManifest.from_dict(
        json.loads(
            (Path(__file__).parents[2] / "deploy/psi0_eval/pc2-runner-v1.json").read_text(
                encoding="utf-8"
            )
        )
    )


def make_event_reader(
    *, run_id: str = "run-a", episode_index: int = 2, evaluator_pid: int = 100
):
    from simple.eval_runtime.events import EventReader

    return EventReader(
        run_id=run_id,
        episode_index=episode_index,
        evaluator_pid=evaluator_pid,
        clock=lambda: 10.0,
    )


def make_runtime_event(
    *, sequence: int, event: str, payload: dict[str, Any] | None = None
):
    from simple.eval_runtime.events import RuntimeEvent

    return RuntimeEvent(
        schema_version=1,
        run_id="run-a",
        episode_index=2,
        evaluator_pid=100,
        sequence=sequence,
        event=event,
        payload=payload or {},
    )


def inject_event_malformation(reader: Any, malformation: str) -> None:
    malformed = {
        "utf8": (b"\xff\n", False),
        "json": (b"{]\n", False),
        "partial": (b'{"schema_version":1', True),
        "oversize": (b"x" * 4097, False),
        "early_eof": (b"", True),
    }
    payload, eof = malformed[malformation]
    reader.feed_bytes(payload, eof=eof)


def populate_complete_evidence(store: Any, *, episodes: tuple[int, ...]) -> None:
    from simple.eval_runtime.evidence import mandatory_evidence_paths

    for relative in mandatory_evidence_paths(episodes):
        store.write(relative, {"schema_version": 1, "status": "ok"})


def finalize_failed_evidence(
    store: Any,
    *,
    run_id: str,
    unknown_remote_state: bool = False,
) -> Any:
    manifest_path = store.root / "manifest.json"
    if manifest_path.exists():
        return store.verify_terminal_manifest(expected_run_id=run_id)
    for relative in (
        "cleanup/actions.json",
        "cleanup/processes-final.json",
        "cleanup/ports-final.json",
        "cleanup/container-final.json",
        "cleanup/remote-helpers-final.json",
    ):
        if not (store.root / relative).exists():
            store.write(relative, {"schema_version": 1, "status": "clean"})
    return store.finalize_terminal_manifest(
        run_id=run_id,
        episodes=(),
        infrastructure_verdict="FAIL",
        overall_verdict="FAIL",
        task_results={},
        owned_processes_alive=[],
        unknown_remote_state=unknown_remote_state,
        unresolved_errors=["INTERRUPTED_RETRY"],
    )
```

The production classes above must expose the exact `from_dict` constructors shown; `LeaseStore.install_expired_test_record` is enabled only when `SIMPLE_EVAL_RUNTIME_UNIT_TEST=1` and otherwise raises `RuntimeBlocked("TEST_API_DISABLED", ...)`. Task-specific tests may extend these fakes locally, but may not reference an undefined helper. The native tests, FD/filesystem tests, and integration gates—not these state-only fakes—own kernel-boundary proof.

Use function names `make_*`, never `test_*`, for fixture builders so pytest does not collect them as tests.

- [ ] **Step 3: Add the complete static-file inventory**

`tests/eval_runtime/static_files.txt` contains one path per line, sorted bytewise. Start with every Python path in the file map above and every `tests/eval_runtime/test_*.py` created by later tasks. Each later task updates this file in the same commit as a new Python file. Do not list Rust, JSON, Markdown, generated candidates, or preserved evidence.

- [ ] **Step 4: Document integration-test isolation**

```markdown
# Dedicated PSI0 runtime integration gates

These tests are never collected by `pytest tests/eval_runtime`.
Run only the exact gate command named in the implementation plan. Gates 0--6
must use fakes or model-free fixtures. Gates 7--9 require explicit operator
approval, VPN/SSH availability, an approved profile commit, PC2 GPU 1, and
H100 GPU 7. No gate authorizes real robot control.
```

- [ ] **Step 5: Verify collection is empty and commit**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  --collect-only -q -p no:cacheprovider tests/eval_runtime
```

Expected: `no tests collected`; no simulator import, SSH process, Docker process, or network socket is created.

```bash
git add tests/eval_runtime tests/integration/eval_runtime/README.md
git commit -m "test: scaffold dedicated PSI0 runtime"
```

### Task 1: Implement strict identifiers, profile parsing, and approval-commit verification

**Files:**
- Create: `src/simple/eval_runtime/__init__.py`
- Create: `src/simple/eval_runtime/contracts.py`
- Create: `src/simple/eval_runtime/profile.py`
- Create: `tests/eval_runtime/test_contracts.py`
- Create: `tests/eval_runtime/test_profile.py`
- Modify: `tests/eval_runtime/static_files.txt`

- [ ] **Step 1: Write failing strict-type and path tests**

```python
# tests/eval_runtime/test_contracts.py
from __future__ import annotations

from pathlib import PurePosixPath

import pytest

from simple.eval_runtime.contracts import (
    PROFILE_FIELDS,
    RuntimeBlocked,
    parse_runtime_profile,
    validate_identifier,
    validate_relative_path,
)

from .conftest import digest


def make_profile(*, source_commit: str = "b" * 40) -> dict[str, object]:
    value: dict[str, object] = {}
    for name, rule in PROFILE_FIELDS.items():
        value[name] = rule.example
    value.update(
        source_snapshot_host_path=f"/inputs/server/{value['source_tree_root_sha256']}",
        checkpoint_snapshot_host_path=f"/inputs/checkpoint/{value['checkpoint_tree_root_sha256']}",
        pc2_input_host_path=f"/inputs/pc2/{value['pc2_closure_id']}",
        pc2_base_python_host_path=f"/inputs/base/{value['pc2_base_python_tree_sha256']}",
    )
    value["container_seccomp_profile_host_path"] = (
        f"{value['source_snapshot_host_path']}/runtime-tools/psi0_h100_eval_seccomp_v1.json"
    )
    value["simple_source_commit"] = source_commit
    return value


@pytest.mark.parametrize(
    "value",
    [".", "..", "", "a/b", "a\\b", "é", " a", "a\x00b", "-leading"],
)
def test_identifier_rejects_dot_segments_and_unsafe_bytes(value: str) -> None:
    with pytest.raises(RuntimeBlocked, match="INVALID_IDENTIFIER"):
        validate_identifier(value, field="run_id")


def test_identifier_accepts_reviewed_alphabet() -> None:
    assert validate_identifier("run_20260826.a-1", field="run_id") == "run_20260826.a-1"


@pytest.mark.parametrize("value", ["", ".", "..", "a/../b", "/a", "a//b", "a/./b"])
def test_relative_path_has_no_empty_or_dot_segments(value: str) -> None:
    with pytest.raises(RuntimeBlocked, match="INVALID_PATH"):
        validate_relative_path(value, field="checkpoint_weight_relative_path")


def test_relative_path_returns_normalized_posix_path() -> None:
    assert validate_relative_path("weights/step-40000.pt", field="weight") == PurePosixPath(
        "weights/step-40000.pt"
    )


def test_profile_requires_the_exact_87_keys_and_exact_types() -> None:
    payload = make_profile()
    parsed = parse_runtime_profile(
        payload,
        blob_sha256=digest(b"profile"),
        approval_commit="a" * 40,
        source_commit="b" * 40,
    )
    assert set(parsed.values) == set(PROFILE_FIELDS)
    assert len(PROFILE_FIELDS) == 87

    for name in tuple(payload):
        missing = dict(payload)
        missing.pop(name)
        with pytest.raises(RuntimeBlocked, match="PROFILE_SCHEMA"):
            parse_runtime_profile(
                missing,
                blob_sha256=digest(b"profile"),
                approval_commit="a" * 40,
                source_commit="b" * 40,
            )


def test_profile_rejects_boolean_for_every_integer_field() -> None:
    payload = make_profile()
    for name, rule in PROFILE_FIELDS.items():
        if rule.kind == "positive_int" or rule.kind == "schema_one":
            malformed = dict(payload)
            malformed[name] = True
            with pytest.raises(RuntimeBlocked, match=f"PROFILE_TYPE:{name}"):
                parse_runtime_profile(
                    malformed,
                    blob_sha256=digest(b"profile"),
                    approval_commit="a" * 40,
                    source_commit="b" * 40,
                )
```

```python
# tests/eval_runtime/test_profile.py
from __future__ import annotations

from pathlib import Path

import pytest

from simple.eval_runtime.contracts import RuntimeBlocked
from simple.eval_runtime.profile import (
    GitObjectReader,
    exclusive_copy_profile_candidate,
    load_approved_profile,
)

from .test_contracts import make_profile
from .conftest import write_json


class FakeGit(GitObjectReader):
    def __init__(self, source: str, approval: str, changed: list[str], blob: bytes):
        self.source = source
        self.approval = approval
        self.changed = changed
        self.blob = blob

    def head(self) -> str:
        return self.approval

    def first_parent(self, commit: str) -> str:
        assert commit == self.approval
        return self.source

    def changed_paths(self, old: str, new: str) -> list[str]:
        assert (old, new) == (self.source, self.approval)
        return self.changed

    def blob_at(self, commit: str, path: str) -> bytes:
        assert commit == self.approval
        assert path == "configs/psi0_h100_eval_runtime_v1.json"
        return self.blob


def test_profile_commit_may_change_only_the_profile(tmp_path: Path) -> None:
    profile_path = write_json(tmp_path / "profile.json", make_profile())
    blob = profile_path.read_bytes()
    git = FakeGit("b" * 40, "a" * 40, ["configs/psi0_h100_eval_runtime_v1.json"], blob)
    loaded = load_approved_profile(git)
    assert loaded.source_commit == "b" * 40
    assert loaded.approval_commit == "a" * 40


def test_profile_simple_source_commit_must_equal_approval_parent(tmp_path: Path) -> None:
    profile_path = write_json(tmp_path / "profile.json", make_profile(source_commit="c" * 40))
    git = FakeGit(
        "b" * 40,
        "a" * 40,
        ["configs/psi0_h100_eval_runtime_v1.json"],
        profile_path.read_bytes(),
    )
    with pytest.raises(RuntimeBlocked, match="PROFILE_SOURCE_BINDING"):
        load_approved_profile(git)


@pytest.mark.parametrize(
    "changed",
    [[], ["README.md"], ["configs/psi0_h100_eval_runtime_v1.json", "README.md"]],
)
def test_profile_commit_rejects_any_other_diff(changed: list[str]) -> None:
    blob = (str(make_profile()).encode())
    git = FakeGit("b" * 40, "a" * 40, changed, blob)
    with pytest.raises(RuntimeBlocked, match="PROFILE_APPROVAL_COMMIT"):
        load_approved_profile(git)


def test_candidate_publication_is_exclusive_and_byte_identical(tmp_path: Path) -> None:
    source = tmp_path / "candidate.json"
    destination = tmp_path / "approved.json"
    source.write_bytes(b'{"schema_version":1}\n')
    exclusive_copy_profile_candidate(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode & 0o777 == 0o644
    with pytest.raises(FileExistsError):
        exclusive_copy_profile_candidate(source, destination)
```

- [ ] **Step 2: Run the focused tests and observe RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_contracts.py \
  tests/eval_runtime/test_profile.py
```

Expected: collection fails with `ModuleNotFoundError: simple.eval_runtime`.

- [ ] **Step 3: Implement exact validators and field rules**

`contracts.py` defines `FieldRule(kind, example, validator)`, the 87 profile fields in the exact order and with the exact validation rules from the approved design, and rejects `type(value) is not expected_type`. Use these concrete primitives:

```python
# src/simple/eval_runtime/contracts.py
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Callable, Mapping

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class RuntimeBlocked(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


class TerminalState(str, Enum):
    CLEAN = "CLEAN"
    FAILED = "FAILED"
    PROVENANCE_BLOCKED = "PROVENANCE_BLOCKED"
    STALE_OWNED_BLOCKED = "STALE_OWNED_BLOCKED"
    FOREIGN_BLOCKED = "FOREIGN_BLOCKED"


@dataclass(frozen=True, slots=True)
class FieldRule:
    kind: str
    example: object
    validator: Callable[[object, str], object]


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    values: Mapping[str, object]
    blob_sha256: str
    approval_commit: str
    source_commit: str


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    argv_sha256: str


@dataclass(frozen=True, slots=True)
class OwnedDeadline:
    expires_at: float

    def remaining(self, clock: Callable[[], float] = time.monotonic) -> float:
        return max(0.0, self.expires_at - clock())


def validate_identifier(value: object, *, field: str) -> str:
    if type(value) is not str or value in {".", ".."} or not IDENTIFIER.fullmatch(value):
        raise RuntimeBlocked("INVALID_IDENTIFIER", field)
    return value


def validate_relative_path(value: object, *, field: str) -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value or value.startswith("/"):
        raise RuntimeBlocked("INVALID_PATH", field)
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise RuntimeBlocked("INVALID_PATH", field)
    return PurePosixPath(*parts)


def parse_runtime_profile(
    data: object, *, blob_sha256: str, approval_commit: str, source_commit: str
) -> RuntimeProfile:
    if type(data) is not dict or set(data) != set(PROFILE_FIELDS):
        raise RuntimeBlocked("PROFILE_SCHEMA", "exact 87-key set required")
    parsed: dict[str, object] = {}
    for name, rule in PROFILE_FIELDS.items():
        try:
            parsed[name] = rule.validator(data[name], name)
        except RuntimeBlocked:
            raise
        except Exception as error:
            raise RuntimeBlocked(f"PROFILE_TYPE:{name}", str(error)) from error
    if parsed["simple_source_commit"] != source_commit:
        raise RuntimeBlocked(
            "PROFILE_SOURCE_BINDING", "simple_source_commit must equal approval parent"
        )
    _validate_profile_relationships(parsed)
    return RuntimeProfile(MappingProxyType(parsed), blob_sha256, approval_commit, source_commit)


# Populate PROFILE_FIELDS directly from the approved 87-key table.
def _exact_int(value: object, field: str, expected: int | None = None) -> int:
    if type(value) is not int or value <= 0 or (expected is not None and value != expected):
        raise RuntimeBlocked(f"PROFILE_TYPE:{field}", "positive exact integer required")
    return value


def _string(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise RuntimeBlocked(f"PROFILE_TYPE:{field}", "nonempty string required")
    return value


def _hex(value: object, field: str, pattern: re.Pattern[str]) -> str:
    text = _string(value, field)
    if pattern.fullmatch(text) is None:
        raise RuntimeBlocked(f"PROFILE_TYPE:{field}", "invalid hexadecimal identity")
    return text


def _absolute(value: object, field: str) -> str:
    text = _string(value, field)
    if not text.startswith("/") or "/../" in f"/{text}/" or "/./" in f"/{text}/":
        raise RuntimeBlocked(f"PROFILE_TYPE:{field}", "normalized absolute path required")
    return text


def _image_id(value: object, field: str) -> str:
    text = _string(value, field)
    if not text.startswith("sha256:") or HEX64.fullmatch(text[7:]) is None:
        raise RuntimeBlocked(f"PROFILE_TYPE:{field}", "digest-qualified image ID required")
    return text


def _image_reference(value: object, field: str) -> str:
    text = _string(value, field)
    prefix = "pytorch/pytorch@sha256:"
    if not text.startswith(prefix) or HEX64.fullmatch(text[len(prefix):]) is None:
        raise RuntimeBlocked(f"PROFILE_TYPE:{field}", "digest-qualified image reference required")
    return text


FIELD_KINDS = {
    "schema_version": "schema_one",
    "profile_id": "string",
    "image_reference": "image_reference",
    "image_id": "image_id",
    "source_snapshot_host_path": "absolute",
    "source_tree_root_sha256": "hex64",
    "source_entry_count": "positive_int",
    "source_regular_file_bytes": "positive_int",
    "source_completion_sha256": "hex64",
    "source_installer_receipt_sha256": "hex64",
    "source_mode_policy_version": "positive_int",
    "package_freeze_sha256": "hex64",
    "python_version": "string",
    "torch_version": "string",
    "torch_cuda_version": "string",
    "checkpoint_snapshot_host_path": "absolute",
    "checkpoint_weight_relative_path": "relative",
    "checkpoint_entry_count": "positive_int",
    "checkpoint_regular_file_bytes": "positive_int",
    "checkpoint_size": "positive_int",
    "checkpoint_sha256": "hex64",
    "checkpoint_tree_root_sha256": "hex64",
    "checkpoint_completion_sha256": "hex64",
    "checkpoint_installer_receipt_sha256": "hex64",
    "checkpoint_mode_policy_version": "positive_int",
    "checkpoint_tracer_path": "absolute",
    "checkpoint_tracer_sha256": "hex64",
    "checkpoint_tracer_version": "string",
    "checkpoint_tracer_argv_sha256": "hex64",
    "checkpoint_tracer_probe_sha256": "hex64",
    "checkpoint_tracer_probe_sentinel_sha256": "hex64",
    "checkpoint_tracer_probe_argv_sha256": "hex64",
    "container_seccomp_profile_host_path": "absolute",
    "container_seccomp_profile_sha256": "hex64",
    "container_security_contract_sha256": "hex64",
    "server_source_sha256": "hex64",
    "launcher_source_sha256": "hex64",
    "h100_roots_identity_sha256": "hex64",
    "h100_input_installer_sha256": "hex64",
    "simple_source_commit": "hex40",
    "simple_root_tree": "hex40",
    "recursive_gitlinks_sha256": "hex64",
    "pc2_closure_id": "hex64",
    "pc2_input_host_path": "absolute",
    "pc2_source_tree_sha256": "hex64",
    "pc2_venv_tree_sha256": "hex64",
    "pc2_base_python_host_path": "absolute",
    "pc2_base_python_tree_sha256": "hex64",
    "pc2_base_python_loader_relative_path": "relative",
    "pc2_base_python_loader_sha256": "hex64",
    "pc2_base_python_completion_sha256": "hex64",
    "pc2_base_python_root_identity_sha256": "hex64",
    "pc2_base_python_installer_receipt_sha256": "hex64",
    "pc2_package_freeze_sha256": "hex64",
    "pc2_import_origins_sha256": "hex64",
    "pc2_native_closure_sha256": "hex64",
    "pc2_episode_data_tree_sha256": "hex64",
    "pc2_task_assets_tree_sha256": "hex64",
    "pc2_asset_requirements_sha256": "hex64",
    "pc2_asset_normalization_results_sha256": "hex64",
    "pc2_runtime_identity_sha256": "hex64",
    "pc2_runtime_identity_sidecar_sha256": "hex64",
    "pc2_closure_completion_sha256": "hex64",
    "pc2_closure_root_identity_sha256": "hex64",
    "pc2_roots_identity_sha256": "hex64",
    "pc2_input_installer_sha256": "hex64",
    "pc2_installer_config_sha256": "hex64",
    "pc2_installer_service_unit_sha256": "hex64",
    "pc2_installer_socket_unit_sha256": "hex64",
    "pc2_installer_receipt_sha256": "hex64",
    "pc2_runner_launcher_sha256": "hex64",
    "pc2_runner_config_sha256": "hex64",
    "pc2_runner_service_unit_sha256": "hex64",
    "pc2_runner_socket_unit_sha256": "hex64",
    "pc2_runner_sandbox_contract_sha256": "hex64",
    "pc2_policy_relay_sha256": "hex64",
    "pc2_policy_relay_contract_sha256": "hex64",
    "pc2_evaluator_uid": "positive_int",
    "pc2_evaluator_gid": "positive_int",
    "pc2_mode_policy_version": "positive_int",
    "pc2_python_version": "string",
    "pc2_torch_version": "string",
    "pc2_torch_cuda_version": "string",
    "pc2_mujoco_version": "string",
    "pc2_isaac_version": "string",
    "pc2_nvidia_driver_version": "string",
    "pc2_cuda_driver_version": "string",
}


def _field_rule(kind: str) -> FieldRule:
    if kind == "schema_one":
        return FieldRule(kind, 1, lambda value, field: _exact_int(value, field, 1))
    if kind == "positive_int":
        return FieldRule(kind, 1, _exact_int)
    if kind == "hex64":
        return FieldRule(kind, "a" * 64, lambda value, field: _hex(value, field, HEX64))
    if kind == "hex40":
        return FieldRule(kind, "a" * 40, lambda value, field: _hex(value, field, HEX40))
    if kind == "absolute":
        return FieldRule(kind, "/protected/value", _absolute)
    if kind == "relative":
        return FieldRule(kind, "weights/value.bin", lambda value, field: str(validate_relative_path(value, field=field)))
    if kind == "image_id":
        return FieldRule(kind, "sha256:" + "a" * 64, lambda value, field: _image_id(value, field))
    if kind == "image_reference":
        return FieldRule(kind, "pytorch/pytorch@sha256:" + "a" * 64, lambda value, field: _image_reference(value, field))
    return FieldRule(kind, "value", _string)


PROFILE_FIELDS = MappingProxyType(
    {name: _field_rule(kind) for name, kind in FIELD_KINDS.items()}
)


def _validate_profile_relationships(values: dict[str, object]) -> None:
    required_suffixes = {
        "source_snapshot_host_path": f"/server/{values['source_tree_root_sha256']}",
        "checkpoint_snapshot_host_path": f"/checkpoint/{values['checkpoint_tree_root_sha256']}",
        "pc2_input_host_path": f"/{values['pc2_closure_id']}",
        "pc2_base_python_host_path": f"/{values['pc2_base_python_tree_sha256']}",
    }
    for field, suffix in required_suffixes.items():
        if not str(values[field]).endswith(suffix):
            raise RuntimeBlocked(f"PROFILE_RELATION:{field}", suffix)
    source = str(values["source_snapshot_host_path"]).rstrip("/")
    seccomp = str(values["container_seccomp_profile_host_path"])
    if not seccomp.startswith(source + "/"):
        raise RuntimeBlocked("PROFILE_RELATION:container_seccomp_profile_host_path", source)
```

`profile.py` runs Git only through argv lists and compares the exact profile blob read from `P`; JSON decoding uses `json.loads(blob)` and no worktree profile read:

```python
# src/simple/eval_runtime/profile.py
from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

from .contracts import RuntimeBlocked, RuntimeProfile, parse_runtime_profile

PROFILE_PATH = "configs/psi0_h100_eval_runtime_v1.json"


class GitObjectReader:
    def _run(self, *args: str) -> bytes:
        return subprocess.run(
            ["git", *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout

    def head(self) -> str:
        return self._run("rev-parse", "HEAD").decode("ascii").strip()

    def first_parent(self, commit: str) -> str:
        return self._run("rev-parse", f"{commit}^").decode("ascii").strip()

    def changed_paths(self, old: str, new: str) -> list[str]:
        return self._run("diff", "--name-only", old, new).decode("utf-8").splitlines()

    def blob_at(self, commit: str, path: str) -> bytes:
        return self._run("show", f"{commit}:{path}")


def load_approved_profile(git: GitObjectReader) -> RuntimeProfile:
    approval = git.head()
    source = git.first_parent(approval)
    if git.changed_paths(source, approval) != [PROFILE_PATH]:
        raise RuntimeBlocked("PROFILE_APPROVAL_COMMIT", "approval must change one path")
    blob = git.blob_at(approval, PROFILE_PATH)
    try:
        payload = json.loads(blob)
    except Exception as error:
        raise RuntimeBlocked("PROFILE_SCHEMA", str(error)) from error
    return parse_runtime_profile(
        payload,
        blob_sha256=hashlib.sha256(blob).hexdigest(),
        approval_commit=approval,
        source_commit=source,
    )


def exclusive_copy_profile_candidate(source: Path, destination: Path) -> None:
    source_fd = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    parent_fd = os.open(
        destination.parent, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    destination_fd = -1
    created_identity: tuple[int, int] | None = None
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise RuntimeBlocked("PROFILE_CANDIDATE_TYPE", str(source))
        destination_fd = os.open(
            destination.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o644,
            dir_fd=parent_fd,
        )
        destination_stat = os.fstat(destination_fd)
        created_identity = (destination_stat.st_dev, destination_stat.st_ino)
        while chunk := os.read(source_fd, 1024 * 1024):
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
        os.fchmod(destination_fd, 0o644)
        os.fsync(destination_fd)
        path_stat = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        if (path_stat.st_dev, path_stat.st_ino) != created_identity:
            raise RuntimeBlocked("PROFILE_PUBLICATION_RACE", destination.name)
        os.fsync(parent_fd)
    except Exception:
        if created_identity is not None:
            try:
                path_stat = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
                if (path_stat.st_dev, path_stat.st_ino) == created_identity:
                    os.unlink(destination.name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        os.close(parent_fd)
        os.close(source_fd)
```

- [ ] **Step 4: Run focused tests and the key-count verifier**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_contracts.py \
  tests/eval_runtime/test_profile.py
.venv/bin/python -c 'from simple.eval_runtime.contracts import PROFILE_FIELDS; assert len(PROFILE_FIELDS) == 87; print("87 strict profile fields")'
```

Expected: all tests pass and stdout contains `87 strict profile fields`.

- [ ] **Step 5: Commit**

```bash
git add src/simple/eval_runtime tests/eval_runtime
git commit -m "feat: add strict PSI0 runtime contracts"
```

### Task 2: Implement canonical JSON, no-follow tree manifests, immutable evidence writes, and process cleanup

**Files:**
- Create: `src/simple/eval_runtime/canonical.py`
- Create: `src/simple/eval_runtime/processes.py`
- Create: `tests/eval_runtime/test_canonical.py`
- Create: `tests/eval_runtime/test_processes.py`
- Modify: `tests/eval_runtime/static_files.txt`

- [ ] **Step 1: Write failing canonicalization and containment tests**

```python
# tests/eval_runtime/test_canonical.py
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from simple.eval_runtime.canonical import (
    ManifestPolicy,
    atomic_write_new_json,
    canonical_json_bytes,
    canonical_tree_manifest,
)
from simple.eval_runtime.contracts import RuntimeBlocked


def test_canonical_json_is_sorted_compact_and_rejects_nan() -> None:
    assert canonical_json_bytes({"z": 1, "a": [True, None]}) == b'{"a":[true,null],"z":1}'
    with pytest.raises(RuntimeBlocked, match="NON_CANONICAL_JSON"):
        canonical_json_bytes({"x": float("nan")})


def test_tree_manifest_is_mode_inclusive_and_byte_sorted(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "z").write_bytes(b"z")
    (root / "a").mkdir()
    (root / "a" / "x").write_bytes(b"xx")
    os.chmod(root / "z", 0o444)
    os.chmod(root / "a" / "x", 0o555)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        manifest = canonical_tree_manifest(root_fd, ManifestPolicy())
    finally:
        os.close(root_fd)
    lines = [json.loads(line) for line in manifest.bytes_value.splitlines()]
    assert [line["path"] for line in lines] == ["a", "a/x", "z"]
    assert lines[1]["mode"] == "0555"
    assert lines[2]["mode"] == "0444"
    assert manifest.regular_file_bytes == 3


@pytest.mark.parametrize("kind", ["absolute", "escape", "fifo", "socket"])
def test_manifest_rejects_escaping_links_and_special_files(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    if kind == "absolute":
        (root / "bad").symlink_to("/tmp/outside")
    elif kind == "escape":
        (root / "bad").symlink_to("../outside")
    elif kind == "fifo":
        os.mkfifo(root / "bad")
    else:
        import socket
        sock = socket.socket(socket.AF_UNIX)
        sock.bind(str(root / "bad"))
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with pytest.raises(RuntimeBlocked, match="TREE_ENTRY"):
            canonical_tree_manifest(root_fd, ManifestPolicy())
    finally:
        os.close(root_fd)
        if kind == "socket":
            sock.close()


def test_atomic_write_refuses_existing_target(tmp_path: Path) -> None:
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        first = atomic_write_new_json(parent_fd, "record.json", {"schema_version": 1})
        assert first.size > 0
        with pytest.raises(FileExistsError):
            atomic_write_new_json(parent_fd, "record.json", {"schema_version": 1})
    finally:
        os.close(parent_fd)
```

```python
# tests/eval_runtime/test_processes.py
from __future__ import annotations

import signal

import pytest

from simple.eval_runtime.contracts import OwnedDeadline, ProcessIdentity, RuntimeBlocked
from simple.eval_runtime.processes import terminate_owned_process


class FakeProcessOps:
    def __init__(self, identities: list[ProcessIdentity | None], alive: list[bool]):
        self.identities = identities
        self.alive = alive
        self.signals: list[int] = []
        self.wait_deadlines: list[float] = []

    def inspect(self, pid: int) -> ProcessIdentity | None:
        return self.identities.pop(0)

    def signal_pidfd(self, pid: int, sig: int) -> None:
        self.signals.append(sig)

    def wait_dead(self, pid: int, deadline: float) -> bool:
        self.wait_deadlines.append(deadline)
        return self.alive.pop(0)


def test_cleanup_uses_int_term_kill_and_fresh_post_kill_wait() -> None:
    expected = ProcessIdentity(42, 900, "a" * 64)
    ops = FakeProcessOps([expected, expected, expected, expected], [False, False, True])
    terminate_owned_process(ops, expected, OwnedDeadline(1200.0), stage_seconds=2.0)
    assert ops.signals == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    assert len(ops.wait_deadlines) == 3
    assert ops.wait_deadlines[-1] > ops.wait_deadlines[-2]


def test_identity_drift_never_signals() -> None:
    expected = ProcessIdentity(42, 900, "a" * 64)
    replacement = ProcessIdentity(42, 901, "a" * 64)
    ops = FakeProcessOps([replacement], [])
    with pytest.raises(RuntimeBlocked, match="FOREIGN_PROCESS"):
        terminate_owned_process(ops, expected, OwnedDeadline(1200.0), stage_seconds=2.0)
    assert ops.signals == []
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_canonical.py \
  tests/eval_runtime/test_processes.py
```

Expected: imports fail because `canonical.py` and `processes.py` do not exist.

- [ ] **Step 3: Implement canonical serialization and FD-relative traversal**

```python
# src/simple/eval_runtime/canonical.py
from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from dataclasses import dataclass
from typing import Mapping

from .contracts import RuntimeBlocked


@dataclass(frozen=True, slots=True)
class ManifestPolicy:
    allowed_absolute_prefixes: tuple[str, ...] = ()
    excluded_names: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class TreeManifest:
    bytes_value: bytes
    root_sha256: str
    entry_count: int
    regular_file_bytes: int


@dataclass(frozen=True, slots=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    sha256: str


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RuntimeBlocked("NON_CANONICAL_JSON", str(error)) from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_tree_manifest(root_fd: int, policy: ManifestPolicy) -> TreeManifest:
    entries: list[dict[str, object]] = []
    regular_bytes = 0

    def visit(parent_fd: int, prefix: tuple[str, ...]) -> None:
        nonlocal regular_bytes
        names = sorted(os.listdir(parent_fd), key=lambda name: os.fsencode(name))
        for name in names:
            if name in policy.excluded_names:
                continue
            relative = "/".join((*prefix, name))
            info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISDIR(info.st_mode):
                entries.append({"kind": "directory", "path": relative, "mode": f"{mode:04o}"})
                child_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
                try:
                    visit(child_fd, (*prefix, name))
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(info.st_mode):
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
                try:
                    payload = bytearray()
                    while block := os.read(fd, 1024 * 1024):
                        payload.extend(block)
                finally:
                    os.close(fd)
                regular_bytes += info.st_size
                entries.append({"kind": "file", "path": relative, "mode": f"{mode:04o}", "bytes": info.st_size, "sha256": sha256_bytes(bytes(payload))})
            elif stat.S_ISLNK(info.st_mode):
                target = os.readlink(name, dir_fd=parent_fd)
                if target.startswith("/") and not any(target == p or target.startswith(f"{p}/") for p in policy.allowed_absolute_prefixes):
                    raise RuntimeBlocked("TREE_ENTRY", relative)
                if not target.startswith("/") and any(part == ".." for part in target.split("/")):
                    raise RuntimeBlocked("TREE_ENTRY", relative)
                entries.append({"kind": "symlink", "path": relative, "target": target})
            else:
                raise RuntimeBlocked("TREE_ENTRY", relative)

    visit(root_fd, ())
    encoded = b"".join(canonical_json_bytes(entry) + b"\n" for entry in entries)
    return TreeManifest(encoded, sha256_bytes(encoded), len(entries), regular_bytes)


def atomic_write_new_json(parent_fd: int, name: str, value: object, mode: int = 0o444) -> FileIdentity:
    payload = canonical_json_bytes(value) + b"\n"
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    fd = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
    try:
        os.write(fd, payload)
        os.fsync(fd)
        os.fchmod(fd, mode)
        info = os.fstat(fd)
        os.link(temporary, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd, follow_symlinks=False)
        os.fsync(parent_fd)
    finally:
        os.close(fd)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
    return FileIdentity(info.st_dev, info.st_ino, info.st_size, sha256_bytes(payload))


def append_hash_chained_jsonl(fd: int, value: Mapping[str, object], previous: str | None) -> str:
    record = dict(value)
    record["previous_sha256"] = previous
    encoded = canonical_json_bytes(record) + b"\n"
    os.write(fd, encoded)
    os.fsync(fd)
    return sha256_bytes(encoded)
```

- [ ] **Step 4: Implement identity-checked bounded termination**

```python
# src/simple/eval_runtime/processes.py
from __future__ import annotations

import signal
import time
from typing import Any

from .contracts import OwnedDeadline, ProcessIdentity, RuntimeBlocked


def terminate_owned_process(
    ops: Any,
    expected: ProcessIdentity,
    overall: OwnedDeadline,
    *,
    stage_seconds: float,
    clock=time.monotonic,
) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        actual = ops.inspect(expected.pid)
        if actual is None:
            return
        if actual != expected:
            raise RuntimeBlocked("FOREIGN_PROCESS", f"pid={expected.pid}")
        ops.signal_pidfd(expected.pid, sig)
        stage_deadline = min(overall.expires_at, clock() + stage_seconds)
        if ops.wait_dead(expected.pid, stage_deadline):
            return
    post_kill_deadline = min(overall.expires_at, clock() + stage_seconds)
    if not ops.wait_dead(expected.pid, post_kill_deadline):
        raise RuntimeBlocked("CLEANUP_FAILED", f"pid={expected.pid} survived SIGKILL")
```

- [ ] **Step 5: Run GREEN and commit**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_canonical.py \
  tests/eval_runtime/test_processes.py
```

Expected: all tests pass.

```bash
git add src/simple/eval_runtime tests/eval_runtime
git commit -m "feat: add canonical PSI0 runtime evidence primitives"
```

### Task 3: Add the vendored static native workspace and framed FD protocol

**Files:**
- Create: `native/psi0_eval_runtime/Cargo.toml`
- Create: `native/psi0_eval_runtime/Cargo.lock`
- Create: `native/psi0_eval_runtime/.cargo/config.toml`
- Create: `native/psi0_eval_runtime/vendor/`
- Create: `native/psi0_eval_runtime/src/lib.rs`
- Create: `native/psi0_eval_runtime/src/protocol.rs`
- Create: `native/psi0_eval_runtime/src/linux.rs`
- Create: `native/psi0_eval_runtime/src/bin/psi0-eval-install-input.rs`
- Create: `native/psi0_eval_runtime/src/bin/psi0-eval-install-pc2-input.rs`
- Create: `native/psi0_eval_runtime/src/bin/psi0-eval-remote-helper.rs`
- Create: `native/psi0_eval_runtime/src/bin/psi0-eval-run-pc2-evaluator.rs`
- Create: `native/psi0_eval_runtime/src/bin/psi0-eval-policy-relay.rs`
- Create: `scripts/build_psi0_eval_native.sh`
- Create: `tests/eval_runtime/test_native_build.py`

- [ ] **Step 1: Write the failing native build and ELF-contract test**

```python
# tests/eval_runtime/test_native_build.py
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

BINARIES = (
    "psi0-eval-install-input",
    "psi0-eval-install-pc2-input",
    "psi0-eval-remote-helper",
    "psi0-eval-run-pc2-evaluator",
    "psi0-eval-policy-relay",
)


@pytest.mark.native
@pytest.mark.parametrize("name", BINARIES)
def test_release_binary_is_static_and_has_no_dynamic_dependencies(name: str) -> None:
    binary = Path("native/psi0_eval_runtime/target/x86_64-unknown-linux-gnu/release") / name
    assert binary.is_file()
    program = subprocess.run(["readelf", "-lWd", binary], check=True, text=True, capture_output=True)
    assert "INTERP" not in program.stdout
    assert "NEEDED" not in program.stdout


@pytest.mark.native
def test_native_workspace_is_offline_and_locked() -> None:
    result = subprocess.run(
        ["cargo", "metadata", "--offline", "--locked", "--format-version", "1"],
        cwd="native/psi0_eval_runtime",
        check=True,
        text=True,
        capture_output=True,
    )
    assert '"psi0-eval-runtime"' in result.stdout
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_native_build.py
```

Expected: all five binary assertions fail because the workspace does not exist.

- [ ] **Step 3: Create the locked workspace and shared protocol**

Use this exact manifest; run `cargo vendor --locked vendor` once while network access is explicitly approved, commit the resulting `Cargo.lock`, `.cargo/config.toml`, and `vendor/`, and afterward require `CARGO_NET_OFFLINE=true` for every build:

```toml
# native/psi0_eval_runtime/Cargo.toml
[package]
name = "psi0-eval-runtime"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"

[dependencies]
hex = "0.4.3"
libc = "0.2.158"
nix = { version = "0.27.1", default-features = false, features = ["fs", "mount", "process", "sched", "signal", "socket", "user"] }
serde = { version = "1.0.210", features = ["derive"] }
serde_json = "1.0.128"
sha2 = "0.10.8"
thiserror = "1.0.63"

[profile.release]
codegen-units = 1
lto = true
panic = "abort"
strip = true
```

```toml
# native/psi0_eval_runtime/.cargo/config.toml
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "vendor"

[build]
rustflags = ["-C", "target-feature=+crt-static"]
```

The shared protocol rejects duplicate/extra keys, frames above 1 MiB, reordered FD roles, non-`CLOEXEC` FDs, and ancillary truncation. Its core API is:

```rust
// native/psi0_eval_runtime/src/protocol.rs
use serde::{Deserialize, Serialize};
use std::os::fd::{OwnedFd, RawFd};

pub const MAX_FRAME_BYTES: usize = 1_048_576;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema_version: u8,
    pub operation: String,
    pub request_id: String,
    pub profile_sha256: String,
    pub payload: serde_json::Value,
    pub fd_roles: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Response {
    pub schema_version: u8,
    pub request_id: String,
    pub status: String,
    pub payload: serde_json::Value,
    pub error: Option<String>,
}

pub fn recv_request(socket: RawFd, expected_roles: &[&str]) -> Result<(Request, Vec<OwnedFd>), ProtocolError>;
pub fn send_response(socket: RawFd, response: &Response, handoff: Option<RawFd>) -> Result<(), ProtocolError>;
pub fn recv_exact_frame(fd: RawFd, maximum: usize) -> Result<Vec<u8>, ProtocolError>;
pub fn send_exact_frame(fd: RawFd, payload: &[u8]) -> Result<(), ProtocolError>;
```

Each binary initially parses one `--self-test` operation, prints its own SHA-256/ELF metadata as canonical JSON, and exits zero. No binary accepts a shell command, arbitrary executable, arbitrary path, environment extension, or network endpoint.

- [ ] **Step 4: Add the reproducible build script**

```bash
#!/usr/bin/env bash
# scripts/build_psi0_eval_native.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CARGO_HOME="${CARGO_HOME:-/tmp/psi0-eval-cargo-home}"
export CARGO_NET_OFFLINE=true
cargo build --offline --locked --release \
  --manifest-path "$root/native/psi0_eval_runtime/Cargo.toml" \
  --target x86_64-unknown-linux-gnu
for name in \
  psi0-eval-install-input \
  psi0-eval-install-pc2-input \
  psi0-eval-remote-helper \
  psi0-eval-run-pc2-evaluator \
  psi0-eval-policy-relay
do
  binary="$root/native/psi0_eval_runtime/target/x86_64-unknown-linux-gnu/release/$name"
  test -x "$binary"
  if readelf -lWd "$binary" | grep -Eq 'INTERP|NEEDED'; then
    echo "$name is not a static no-dependency executable" >&2
    exit 1
  fi
done
```

- [ ] **Step 5: Build, run GREEN, and commit**

Run:

```bash
bash scripts/build_psi0_eval_native.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider -m native tests/eval_runtime/test_native_build.py
CARGO_NET_OFFLINE=true cargo test --offline --locked \
  --manifest-path native/psi0_eval_runtime/Cargo.toml
```

Expected: five ELF tests, metadata test, and all Rust unit tests pass; every `readelf` check is silent.

```bash
git add native/psi0_eval_runtime scripts/build_psi0_eval_native.sh \
  tests/eval_runtime/test_native_build.py
git commit -m "build: add static PSI0 runtime helpers"
```

### Task 4: Seal SIMPLE, base Python, episode data, and offline task assets

**Files:**
- Create: `src/simple/eval_runtime/assets.py`
- Create: `src/simple/eval_runtime/pc2_closure.py`
- Create: `tests/eval_runtime/test_assets.py`
- Create: `tests/eval_runtime/test_pc2_closure.py`
- Modify: `src/simple/utils.py`
- Modify: `src/simple/scenes/hssd.py`
- Modify: `src/simple/assets/graspnet.py`
- Modify: `tests/eval_runtime/static_files.txt`

- [ ] **Step 1: Write failing offline-data and HSSD production-path tests**

```python
# tests/eval_runtime/test_assets.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from simple.eval_runtime.assets import (
    AssetRequirement,
    probe_episode_assets,
)
from simple.scenes.hssd import HssdSceneManager
from simple.utils import resolve_data_path, resolve_res_path


def test_data_root_seam_is_offline_and_never_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "sealed-data"
    root.mkdir()
    monkeypatch.setenv("SIMPLE_DATA_ROOT", str(root))
    monkeypatch.setenv("SIMPLE_ASSET_OFFLINE", "1")
    monkeypatch.setattr("simple.utils.snapshot_download", lambda **_: pytest.fail("network"))
    with pytest.raises(FileNotFoundError):
        resolve_data_path("assets/missing/item.usd", auto_download=True)


def test_hssd_load_uses_normalized_closure_without_shell_or_tmp_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "task-data"
    configs = yaml.safe_load(
        Path(resolve_res_path("hssd-scenes/config.yaml")).read_text(encoding="utf-8")
    )
    config = next(item for item in configs if item["uid"] == "scene0")
    scene_name = config["name"]
    scene = data / config["data_dir"] / scene_name
    scene.mkdir(parents=True)
    usd = scene / f"{scene_name}.usd"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    usd_sha256 = hashlib.sha256(usd.read_bytes()).hexdigest()
    (scene / "NORMALIZED.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scene_uid": "scene0",
                "scene_name": scene_name,
                "source_usd_sha256": "a" * 64,
                "normalized_usd_sha256": usd_sha256,
                "normalization_policy_version": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SIMPLE_DATA_ROOT", str(data))
    monkeypatch.setenv("SIMPLE_ASSET_OFFLINE", "1")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("subprocess"))
    monkeypatch.setattr("shutil.copytree", lambda *a, **k: pytest.fail("copytree"))
    before = set(Path("/tmp").iterdir())
    loaded = HssdSceneManager().load("hssd:scene0")
    assert loaded.uid == "hssd:scene0"
    assert loaded.name == "107734119_175999932"
    assert set(Path("/tmp").iterdir()) == before


def test_episode_asset_probe_rejects_missing_or_out_of_root_dependency(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    required = (
        AssetRequirement(
            "scene",
            "scenes/hssd/107734119_175999932/107734119_175999932.usd",
            "a" * 64,
        ),
        AssetRequirement("robot", "robots/g1/g1.usd", "b" * 64),
    )
    with pytest.raises(Exception, match="ASSET_PROBE"):
        probe_episode_assets(root, required, episode_index=2)
```

```python
# tests/eval_runtime/test_pc2_closure.py
from __future__ import annotations

from copy import deepcopy

import pytest

from simple.eval_runtime.pc2_closure import (
    BASE_PHASES,
    CLOSURE_PHASES,
    ClosureDescriptor,
    compute_closure_id,
)
from .fakes import make_construction_record


def make_descriptor() -> ClosureDescriptor:
    return ClosureDescriptor(
        schema_version=1,
        simple_commit="a" * 40,
        simple_root_tree="b" * 40,
        recursive_gitlinks_sha256="c" * 64,
        git_object_manifest_sha256="d" * 64,
        episode_data_manifest_sha256="e" * 64,
        asset_requirements_sha256="f" * 64,
        task_data_source_manifest_sha256="0" * 64,
        base_python_tree_sha256="1" * 64,
        mode_policy_version=1,
        hssd_policy_version=1,
        installer_sha256="2" * 64,
        locked_inputs_sha256="3" * 64,
    )


def test_closure_id_is_path_independent_and_requirements_sensitive() -> None:
    descriptor = make_descriptor()
    baseline = compute_closure_id(descriptor)
    assert baseline == compute_closure_id(descriptor)
    changed = deepcopy(descriptor)
    object.__setattr__(changed, "asset_requirements_sha256", "4" * 64)
    assert compute_closure_id(changed) != baseline
    assert not hasattr(descriptor, "pc2_input_host_path")
    assert not hasattr(descriptor, "pc2_asset_normalization_results_sha256")


def test_phase_orders_require_venv_before_hssd() -> None:
    assert CLOSURE_PHASES == (
        "ALLOCATED", "SOURCE", "EPISODE_DATA", "VENV", "TASK_DATA_COPY",
        "HSSD_NORMALIZED", "PAYLOAD_READY", "INSTALLING", "FINAL_RENAMED",
        "RECEIPT_CREATED", "COMPLETE",
    )
    assert BASE_PHASES == (
        "BASE_ALLOCATED", "BASE_COPIED", "BASE_METADATA_NORMALIZED",
        "BASE_FINAL_RENAMED", "BASE_RECEIPT_CREATED", "BASE_COMPLETE",
    )


@pytest.mark.parametrize("phase", CLOSURE_PHASES[:-1])
def test_crash_record_stays_cleanup_required_until_complete(phase: str) -> None:
    record = make_construction_record(phase=phase)
    assert record.cleanup_required is True
    assert record.pending_action is None or isinstance(record.pending_action, str)
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_assets.py \
  tests/eval_runtime/test_pc2_closure.py
```

Expected: imports fail for `simple.eval_runtime.assets` and `pc2_closure`.

- [ ] **Step 3: Add the explicit offline data-root seam and remove the HSSD shell hack**

Change `get_data_dir()` and `resolve_data_path()` so the environment root is authoritative and `SIMPLE_ASSET_OFFLINE=1` disables download even when a caller passes `auto_download=True`:

```python
# src/simple/utils.py
import importlib.resources as res
import os


def get_data_dir() -> str:
    configured = os.environ.get("SIMPLE_DATA_ROOT")
    if configured:
        root = os.path.realpath(configured)
        if not os.path.isabs(root) or not os.path.isdir(root):
            raise RuntimeError("SIMPLE_DATA_ROOT must name an existing absolute directory")
        return root
    return str(res.files("simple").parent.parent / "data")


def _asset_download_allowed(auto_download: bool) -> bool:
    return auto_download and os.environ.get("SIMPLE_ASSET_OFFLINE") != "1"
```

Use `_asset_download_allowed` at every download branch in `resolve_data_path` and `src/simple/assets/graspnet.py`. Move `hashlib` and `json` into `src/simple/scenes/hssd.py`'s existing top-level import block. Replace `HssdSceneManager._hack_fix_tmp_paths` with validation of the freeze-produced `NORMALIZED.json`; production loading performs no subprocess, copy, network, or global `/tmp` write:

```python
# src/simple/scenes/hssd.py
# ruff: noqa: F821
def _require_normalized_scene(
    self, *, scene_uid: str, scene_name: str, scene_dir: str, usd_path: str
) -> None:
    marker = os.path.join(scene_dir, "NORMALIZED.json")
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(usd_path)
    if os.environ.get("SIMPLE_ASSET_OFFLINE") != "1":
        return
    with open(marker, "rb") as stream:
        marker_bytes = stream.read()
    payload = json.loads(marker_bytes)
    keys = {
        "schema_version",
        "scene_uid",
        "scene_name",
        "source_usd_sha256",
        "normalized_usd_sha256",
        "normalization_policy_version",
    }
    if type(payload) is not dict or set(payload) != keys:
        raise RuntimeError("sealed HSSD normalization marker has wrong schema")
    if payload["schema_version"] != 1 or payload["normalization_policy_version"] != 1:
        raise RuntimeError("sealed HSSD normalization marker has wrong version")
    if payload["scene_uid"] != scene_uid or payload["scene_name"] != scene_name:
        raise RuntimeError("sealed HSSD normalization marker has wrong scene identity")
    with open(usd_path, "rb") as stream:
        normalized_sha256 = hashlib.sha256(stream.read()).hexdigest()
    if payload["normalized_usd_sha256"] != normalized_sha256:
        raise RuntimeError("sealed HSSD normalization marker hash mismatch")


def load(self, scene_uid: str) -> Scene:
    key = scene_uid.split(":", 1)[1] if ":" in scene_uid else scene_uid
    configs = {scene["uid"]: scene for scene in self.hssd_scenes}
    scene_name = configs[key]["name"]
    scene_dir = resolve_data_path(f"scenes/hssd/{scene_name}", auto_download=True)
    usd_path = os.path.join(scene_dir, f"{scene_name}.usd")
    self._require_normalized_scene(
        scene_uid=key, scene_name=scene_name, scene_dir=scene_dir, usd_path=usd_path
    )
    return HssdSuite(configs[key])
```

- [ ] **Step 4: Implement path-independent descriptors and write-ahead construction records**

```python
# src/simple/eval_runtime/pc2_closure.py
from __future__ import annotations

from dataclasses import asdict, dataclass

from .canonical import canonical_json_bytes, sha256_bytes

BASE_PHASES = (
    "BASE_ALLOCATED", "BASE_COPIED", "BASE_METADATA_NORMALIZED",
    "BASE_FINAL_RENAMED", "BASE_RECEIPT_CREATED", "BASE_COMPLETE",
)
CLOSURE_PHASES = (
    "ALLOCATED", "SOURCE", "EPISODE_DATA", "VENV", "TASK_DATA_COPY",
    "HSSD_NORMALIZED", "PAYLOAD_READY", "INSTALLING", "FINAL_RENAMED",
    "RECEIPT_CREATED", "COMPLETE",
)


@dataclass(frozen=True, slots=True)
class ClosureDescriptor:
    schema_version: int
    simple_commit: str
    simple_root_tree: str
    recursive_gitlinks_sha256: str
    git_object_manifest_sha256: str
    episode_data_manifest_sha256: str
    asset_requirements_sha256: str
    task_data_source_manifest_sha256: str
    base_python_tree_sha256: str
    mode_policy_version: int
    hssd_policy_version: int
    installer_sha256: str
    locked_inputs_sha256: str


@dataclass(frozen=True, slots=True)
class ConstructionRecord:
    schema_version: int
    construction_attempt: int
    attempt_status: str
    operation_token_sha256: str
    phase: str
    cleanup_required: bool
    pending_action: str | None

    @classmethod
    def from_dict(cls, data: object) -> "ConstructionRecord":
        keys = {
            "schema_version", "construction_attempt", "attempt_status",
            "operation_token_sha256", "phase", "cleanup_required", "pending_action",
        }
        if type(data) is not dict or set(data) != keys:
            raise ValueError("CONSTRUCTION_RECORD_SCHEMA")
        if data["schema_version"] != 1 or data["phase"] not in CLOSURE_PHASES:
            raise ValueError("CONSTRUCTION_RECORD_VALUE")
        return cls(**data)


def compute_closure_id(descriptor: ClosureDescriptor) -> str:
    return sha256_bytes(canonical_json_bytes(asdict(descriptor)))
```

Complete `Pc2ClosureBuilder` with these explicit operations and persist its owner record before each call: `allocate`, `copy_git_objects`, `copy_episode_data`, `build_relative_venv`, `copy_task_assets`, `normalize_hssd_with_staging_venv`, `probe_payload`, `authorize_install`, `mirror_final_renamed`, `mirror_receipt_created`, and `mark_complete`. The builder must:

- require byte-empty root and recursive submodule porcelain output;
- copy source from Git object blobs, never worktree paths;
- create/probe the protected base-Python snapshot before computing `ClosureDescriptor`;
- create `VENV` before `HSSD_NORMALIZED` and invoke only `<staging>/venv/bin/python` for USD/Sdf;
- freeze `asset-requirements.json` before closure-ID computation and write `hssd-normalization-results.json` as a separate deterministic output;
- reject absolute `.pth`, `.egg-link`, shebang, symlink, RPATH/RUNPATH, loader, native-library, import, or data origin outside the closure/base roots;
- set `cleanup_required=true` before allocation and clear it only after protected receipt and post-publish revalidation;
- never create or write the root-owned installer journal.

The exact HSSD result schema is:

```python
HSSD_RESULT_KEYS = {
    "schema_version", "requirements_sha256", "pc2_closure_id",
    "normalizer_policy_version", "usd_sdf_tool_identity_sha256",
    "source_layers", "normalized_layers", "old_to_new_mappings",
}
```

All arrays are sorted by raw UTF-8 path bytes; the document contains no timestamp, absolute host path, staging path, inode, or unordered mapping.

- [ ] **Step 5: Add base-Python and closure crash tests**

Extend `test_pc2_closure.py` with one parameterized fake operation driver that raises after each phase in `BASE_PHASES + CLOSURE_PHASES`, reloads the fsynced owner record, and asserts exactly one of:

```text
assert final_path.exists() is False
# or
assert verify_complete_closure(final_path, receipt_path).valid is True
```

For pre-`PREPARED` base-Python failures assert the intake-ID/operation-token journal has `final_tree_sha256 is None`, contains no publication phase, removes only the token-owned staging inode, terminalizes the attempt as `aborted-before-prepared`, increments `construction_attempt`, and creates a fresh token before retry. For at/post-`PREPARED` failures assert one non-null digest appears unchanged in every later record.

- [ ] **Step 6: Run GREEN and commit**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_assets.py \
  tests/eval_runtime/test_pc2_closure.py
```

Expected: all tests pass; the monkeypatched subprocess, copytree, and download functions are never called.

```bash
git add src/simple/eval_runtime src/simple/utils.py src/simple/scenes/hssd.py \
  src/simple/assets/graspnet.py tests/eval_runtime
git commit -m "feat: seal offline PC2 evaluation inputs"
```

### Task 5: Implement the privileged input installers and replay-safe recovery protocol

**Files:**
- Create: `src/simple/eval_runtime/installer_client.py`
- Create: `src/simple/eval_runtime/installer_manager.py`
- Create: `native/psi0_eval_runtime/src/installer.rs`
- Modify: `native/psi0_eval_runtime/src/bin/psi0-eval-install-input.rs`
- Modify: `native/psi0_eval_runtime/src/bin/psi0-eval-install-pc2-input.rs`
- Create: `deploy/psi0_eval/pc2-installer-v1.json`
- Create: `deploy/psi0_eval/psi0-eval-pc2-installer.socket`
- Create: `deploy/psi0_eval/psi0-eval-pc2-installer@.service`
- Create: `tests/eval_runtime/test_installer_protocol.py`
- Create: `tests/eval_runtime/test_installer_recovery.py`
- Create: `tests/eval_runtime/test_installer_manager_restart.py`
- Create: `tests/eval_runtime/test_systemd_contracts.py`
- Modify: `tests/eval_runtime/static_files.txt`

- [ ] **Step 1: Write failing handoff and recovery-generation tests**

```python
# tests/eval_runtime/test_installer_protocol.py
from __future__ import annotations

import array
import fcntl
import json
import os
import socket
import struct

import pytest

from simple.eval_runtime.installer_client import (
    HandoffPersistenceUncertain,
    InstallerReply,
    LockIdentity,
    apply_locked_reply,
    recv_installer_reply,
)
from .fakes import InstallerHandoffHarness


class RecordingHistory:
    def __init__(
        self,
        order: list[str],
        *,
        fail_at: tuple[str, str] | None = None,
    ):
        self.order = order
        self.fail_at = fail_at
        self.retained: list[int] = []

    def _step(self, name: str) -> None:
        if self.fail_at == (name, "before"):
            raise OSError(f"injected:before:{name}")
        self.order.append(name)
        if self.fail_at == (name, "after"):
            raise OSError(f"injected:after:{name}")

    def persist_reply(self, reply: InstallerReply) -> None:
        del reply
        for name in (
            "install_authorization",
            "recovery_authorization",
            "recovery_result",
            "owner_record_replace",
        ):
            self._step(name)

    def fsync_owner_record_and_parent(self) -> None:
        self._step("owner_record_fsync")
        self._step("owner_parent_fsync")

    def retain_uncertain_handoff(self, reply: InstallerReply, error: BaseException) -> None:
        del error
        self.retained.append(reply.handoff_fd)


def test_manager_fsyncs_transition_before_closing_handoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    order: list[str] = []
    lock = tmp_path / "transaction.lock"
    lock.touch(mode=0o600)
    expected = LockIdentity.from_path(lock)
    handoff_fd = os.open(lock, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    fcntl.flock(handoff_fd, fcntl.LOCK_EX)
    real_close = os.close
    monkeypatch.setattr(os, "close", lambda fd: order.append(f"close:{fd}"))
    reply = InstallerReply("PREPARED", {}, "a" * 64, handoff_fd)
    apply_locked_reply(RecordingHistory(order), reply, expected_lock=expected)
    assert order == [
        "install_authorization",
        "recovery_authorization",
        "recovery_result",
        "owner_record_replace",
        "owner_record_fsync",
        "owner_parent_fsync",
        f"close:{handoff_fd}",
    ]
    real_close(handoff_fd)


@pytest.mark.parametrize(
    "transition",
    [
        "install_authorization",
        "recovery_authorization",
        "recovery_result",
        "owner_record_replace",
        "owner_record_fsync",
        "owner_parent_fsync",
    ],
)
@pytest.mark.parametrize("edge", ["before", "after"])
def test_failed_transition_retains_handoff_and_blocks_waiter(
    transition: str, edge: str, tmp_path
) -> None:
    lock = tmp_path / "transaction.lock"
    lock.touch(mode=0o600)
    expected = LockIdentity.from_path(lock)
    handoff_fd = os.open(lock, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    fcntl.flock(handoff_fd, fcntl.LOCK_EX)
    history = RecordingHistory([], fail_at=(transition, edge))
    reply = InstallerReply("PREPARED", {}, "a" * 64, handoff_fd)
    with pytest.raises(HandoffPersistenceUncertain) as captured:
        apply_locked_reply(history, reply, expected_lock=expected)
    assert captured.value.handoff_fd == handoff_fd
    assert history.retained == [handoff_fd]
    contender = os.open(lock, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    with pytest.raises(BlockingIOError):
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(contender)
    os.close(handoff_fd)


@pytest.mark.parametrize(
    "boundary",
    [
        "before_response", "before_sendmsg", "ancillary_queued", "after_recvmsg",
        "service_local_close", "socket_shutdown", "before_eof", "after_eof",
    ],
)
def test_duplicate_instance_cannot_acquire_until_manager_transition(boundary: str) -> None:
    harness = InstallerHandoffHarness(boundary=boundary)
    harness.queue_duplicate()
    harness.advance_to_boundary()
    assert harness.duplicate_acquired is False
    harness.manager_persist_transition()
    assert harness.duplicate_acquired is False
    harness.manager_close_handoff()
    assert harness.one_duplicate_acquires_after_revalidation()


def _send_response_with_fds(sock: socket.socket, lock_fds: list[int]) -> None:
    payload = json.dumps(
        {
            "schema_version": 1,
            "request_id": "request-a",
            "status": "PREPARED",
            "payload": {},
            "error": None,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    frame = struct.pack("!I", len(payload)) + payload
    ancillary = []
    if lock_fds:
        ancillary.append(
            (socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", lock_fds))
        )
    sock.sendmsg([frame], ancillary)


@pytest.mark.parametrize("case", ["missing", "extra", "wrong_identity"])
def test_receiver_rejects_missing_extra_or_wrong_handoff(case: str, tmp_path) -> None:
    lock = tmp_path / "transaction.lock"
    lock.touch(mode=0o600)
    other = tmp_path / "other.lock"
    other.touch(mode=0o600)
    expected = LockIdentity.from_path(lock)
    manager, service = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    lock_fd = os.open(lock, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    other_fd = os.open(other, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    sent = {
        "missing": [],
        "extra": [lock_fd, other_fd],
        "wrong_identity": [other_fd],
    }[case]
    _send_response_with_fds(service, sent)
    with pytest.raises(RuntimeError, match="HANDOFF_FD_COUNT|HANDOFF_LOCK_IDENTITY"):
        recv_installer_reply(
            manager.fileno(), request_id="request-a", expected_lock=expected
        )
    os.close(lock_fd)
    os.close(other_fd)
    manager.close()
    service.close()


def test_receiver_preserves_same_ofd_lock_and_sets_cloexec(tmp_path) -> None:
    lock = tmp_path / "transaction.lock"
    lock.touch(mode=0o600)
    expected = LockIdentity.from_path(lock)
    manager, service = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    release_read, release_write = os.pipe()
    pid = os.fork()
    if pid == 0:
        manager.close()
        os.close(release_write)
        lock_fd = os.open(lock, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        _send_response_with_fds(service, [lock_fd])
        os.read(release_read, 1)
        os._exit(0)
    service.close()
    os.close(release_read)
    reply = None
    try:
        reply = recv_installer_reply(
            manager.fileno(), request_id="request-a", expected_lock=expected
        )
        assert fcntl.fcntl(reply.handoff_fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
        contender = os.open(lock, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.close(contender)
    finally:
        if reply is not None:
            os.close(reply.handoff_fd)
        os.write(release_write, b"x")
        os.close(release_write)
        os.waitpid(pid, 0)
        manager.close()


def test_receiver_rejects_intervening_independent_holder(tmp_path) -> None:
    lock = tmp_path / "transaction.lock"
    lock.touch(mode=0o600)
    expected = LockIdentity.from_path(lock)
    manager, service = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    holder = os.open(lock, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    passed = os.open(lock, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    fcntl.flock(holder, fcntl.LOCK_EX)
    _send_response_with_fds(service, [passed])
    os.close(passed)
    with pytest.raises(Exception, match="HANDOFF_LOCK_INTERVENED"):
        recv_installer_reply(manager.fileno(), request_id="request-a", expected_lock=expected)
    os.close(holder)
    manager.close()
    service.close()
```

```python
# tests/eval_runtime/test_installer_recovery.py
from __future__ import annotations

from .fakes import RecoveryHarness


def test_two_uncertain_cycles_use_hash_chained_recovery_generations() -> None:
    harness = RecoveryHarness()
    original = harness.install_request_bytes
    generation_1 = harness.append_recovery_generation()
    assert generation_1.recovery_generation == 1
    assert generation_1.previous_entry_sha256 is None
    result_1 = harness.recover(generation_1)
    assert result_1.status == "NO_JOURNAL_NO_MUTATION"
    assert result_1.result.previous_result_sha256 is None
    harness.reactivate_install(generation_1)
    assert harness.replayed_install_bytes == original
    harness.service_create_journal_then_drop_transport()
    generation_2 = harness.append_recovery_generation()
    assert generation_2.recovery_generation == 2
    assert generation_2.previous_entry_sha256 == generation_1.authorization_sha256
    assert generation_2.request_digest != generation_1.request_digest
    result_2 = harness.recover(generation_2)
    assert result_2.status == "RECOVERED"
    assert result_2.result.previous_result_sha256 == result_1.result.result_sha256
    assert harness.install_request_bytes == original


def test_install_and_recovery_authorizations_are_not_interchangeable() -> None:
    harness = RecoveryHarness()
    recovery = harness.append_recovery_generation()
    assert harness.reject_install_against(recovery.active_reference) == "AUTHORIZATION_KIND"
    assert harness.reject_recovery_against(harness.install_reference) == "AUTHORIZATION_KIND"
    assert harness.root_journal_accesses == 0


def test_pre_prepared_abort_never_derives_final_digest_path() -> None:
    harness = RecoveryHarness(crash="mutation_child_before_prepared")
    result = harness.recover(harness.append_recovery_generation())
    assert result.status == "ABORTED_BEFORE_PREPARED"
    assert harness.final_digest_paths_examined == []
    assert harness.unrelated_protected_objects_preserved is True
    assert harness.manager_journal_writes == 0
```

```python
# tests/eval_runtime/test_installer_manager_restart.py
from __future__ import annotations

import fcntl
import os

import pytest

from simple.eval_runtime.installer_manager import (
    DiskAuthorizationStore,
    InstallerOrchestrator,
)
from .fakes import ForkedInstallerService

INSTALL_BYTES = b'{"operation":"install_base_python","schema_version":1}\n'
CRASH_BOUNDARIES = (
    "before_authorization_append", "after_authorization_append",
    "before_active_reference_switch", "after_active_reference_switch",
    "before_request_delivery", "after_request_delivery",
    "before_terminal_handoff_receipt", "after_terminal_handoff_receipt",
    "before_result_append", "after_result_append",
    "before_install_reactivation", "after_install_reactivation",
    "before_terminal_clear", "after_terminal_clear",
    "before_handoff_close", "after_handoff_close",
    "before_resubmission", "after_resubmission",
)
LIVE_PERSISTENCE_BOUNDARIES = (
    "before_result_append", "after_result_append",
    "before_owner_record_replace", "after_owner_record_replace",
    "before_owner_record_fsync", "after_owner_record_fsync",
    "before_owner_parent_fsync", "after_owner_parent_fsync",
)

@pytest.mark.parametrize("boundary", CRASH_BOUNDARIES)
def test_fork_exit_restart_uses_real_wire_order_and_byte_exact_replay(
    boundary: str, tmp_path
) -> None:
    root = tmp_path / "manager-state"
    construction = tmp_path / "construction.lock"
    transaction = tmp_path / "transaction.lock"
    service = ForkedInstallerService.start(
        tmp_path / "service",
        transaction_lock=transaction,
    )
    DiskAuthorizationStore.bootstrap(
        root,
        install_request=INSTALL_BYTES,
        construction_lock=construction,
        transaction_lock=transaction,
    )
    first = InstallerOrchestrator.restart(root, service.socket_path)
    with pytest.raises(Exception, match="INSTALLER_TRANSPORT_UNCERTAIN"):
        first.begin_install()
    first.close_after_uncertain_transport()

    pid = os.fork()
    if pid == 0:
        manager = InstallerOrchestrator.restart(root, service.socket_path)
        manager.drive_to_terminal(
            fault_hook=lambda name: os._exit(86) if name == boundary else None,
        )
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 86

    restarted = InstallerOrchestrator.restart(root, service.socket_path)
    restarted.drive_to_terminal()
    snapshot = restarted.store.validated_snapshot()
    assert snapshot.install_request_bytes == INSTALL_BYTES
    assert snapshot.authorization_chain_valid is True
    assert snapshot.result_chain_valid is True
    assert snapshot.temporary_entries == ()
    assert snapshot.terminal is True
    service.stop()
    service.wait(timeout=5.0)
    requests = service.request_bytes()
    installs = [
        body for body in requests
        if body == INSTALL_BYTES
    ]
    assert installs == [INSTALL_BYTES, INSTALL_BYTES]
    responses = service.responses()
    assert responses[0] == "TRANSPORT_DROPPED"
    assert "NO_JOURNAL_NO_MUTATION" in responses
    assert "TRANSPORT_DROPPED_AFTER_JOURNAL" in responses
    assert responses[-1] == "RECOVERED"
    assert snapshot.replayed_install_sha256 == snapshot.install_request_sha256
    assert snapshot.terminal_response_status == "RECOVERED"


@pytest.mark.parametrize("boundary", LIVE_PERSISTENCE_BOUNDARIES)
def test_live_persistence_failure_retains_both_locks_until_reconciled(
    boundary: str, tmp_path
) -> None:
    root = tmp_path / "manager-state"
    construction = tmp_path / "construction.lock"
    transaction = tmp_path / "transaction.lock"
    DiskAuthorizationStore.bootstrap(
        root,
        install_request=INSTALL_BYTES,
        construction_lock=construction,
        transaction_lock=transaction,
    )
    service = ForkedInstallerService.start(
        tmp_path / "service",
        transaction_lock=transaction,
        script=("RECOVERED",),
    )
    manager = InstallerOrchestrator.restart(root, service.socket_path)
    with pytest.raises(Exception, match="HANDOFF_PERSISTENCE_UNCERTAIN"):
        manager.drive_to_terminal(
            fault_hook=lambda name: (_ for _ in ()).throw(OSError(name))
            if name == boundary
            else None,
        )
    contender = os.open(transaction, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    with pytest.raises(BlockingIOError):
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manager.reconcile_locked_handoff()
    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(contender)
    assert manager.store.validated_snapshot().terminal is True
    service.stop()
    service.wait(timeout=5.0)
```

`ForkedInstallerService` is the simulator-free kernel test service defined in Task 0. It listens on a real Unix socket in a separate `fork` child, reads the production length-prefixed request frame, appends the exact received bytes and response status to fsynced files, and sends terminal frames plus the real transaction-lock OFD with `sendmsg(SCM_RIGHTS)`. Its default state machine drops the first install response, returns the same `NO_JOURNAL_NO_MUTATION` result for every replay of recovery generation 1 until the manager persists it, fsyncs a journal and drops the byte-identical second install response, then returns the same `RECOVERED` result for generation 2 replays. A broken response socket never advances authorization state; request generation and the fsynced service journal determine the response. The explicit stop frame is sent only after the manager verifies terminal state. `InstallerOrchestrator` uses the production session sender and `recv_installer_reply`; no test calls a persist method with a prebuilt reply. The parameterized child can therefore exit before or after actual request delivery and actual terminal-frame-plus-handoff receipt. A fresh orchestrator reopens only the fsynced prefix, performs any required wire operation, and cannot mark terminal without a newly received, validated, durably persisted terminal response. The assertion proves the two actual install frames are byte-identical. `RecoveryHarness` remains only a pure hash-chain unit test. Native tests in Step 6 cover every service-side phase barrier. None calls systemd or requires root.

- [ ] **Step 2: Write failing systemd/config exactness tests**

```python
# tests/eval_runtime/test_systemd_contracts.py
from __future__ import annotations

import json
from pathlib import Path


def test_installer_unit_has_fixed_parent_write_roots_and_accept_socket() -> None:
    service = Path("deploy/psi0_eval/psi0-eval-pc2-installer@.service").read_text()
    socket = Path("deploy/psi0_eval/psi0-eval-pc2-installer.socket").read_text()
    assert "Accept=yes" in socket
    assert "%i" not in service
    assert "%I" not in service
    assert "ReadWritePaths=/mnt/data/psi0-simple-eval-inputs" in service
    assert "ReadWritePaths=/mnt/data/psi0-simple-eval-base-python" in service
    assert "CapabilityBoundingSet=CAP_SYS_ADMIN CAP_CHOWN CAP_FOWNER CAP_DAC_OVERRIDE" in service


def test_installer_config_contains_all_stable_lock_identities() -> None:
    config = json.loads(Path("deploy/psi0_eval/pc2-installer-v1.json").read_text())
    expected = {
        "schema_version", "lifecycle_uid", "input_uid", "input_gid",
        "closure_root", "base_python_root", "control_root", "staging_root",
        "closure_lock_basename", "closure_lock_device", "closure_lock_inode", "closure_lock_mount_id",
        "base_lock_basename", "base_lock_device", "base_lock_inode", "base_lock_mount_id",
        "transaction_lock_basename", "transaction_lock_device", "transaction_lock_inode",
        "transaction_lock_mount_id", "transaction_lock_uid", "transaction_lock_gid",
        "transaction_lock_mode", "mode_policy_version", "request_limit_bytes",
    }
    assert set(config) == expected
```

- [ ] **Step 3: Run RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_installer_protocol.py \
  tests/eval_runtime/test_installer_recovery.py \
  tests/eval_runtime/test_installer_manager_restart.py \
  tests/eval_runtime/test_systemd_contracts.py
```

Expected: imports and missing deployment-file assertions fail.

- [ ] **Step 4: Implement the exact authorization history and locked response rule**

`installer_client.py` uses immutable install bytes plus append-only recovery entries:

```python
# src/simple/eval_runtime/installer_client.py
import array
import fcntl
import hashlib
import json
import os
import socket
import stat
import struct
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class LockIdentity:
    device: int
    inode: int
    mount_id: int
    mode: int

    @classmethod
    def from_path(cls, path: os.PathLike[str]) -> "LockIdentity":
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            return lock_identity_from_fd(fd)
        finally:
            os.close(fd)


@dataclass(frozen=True, slots=True)
class RecoveryAuthorization:
    recovery_generation: int
    authorization_id: str
    recovery_token_sha256: str
    request_digest: str
    previous_entry_sha256: str | None


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    recovery_generation: int
    authorization_sha256: str
    request_sha256: str
    terminal_response_sha256: str
    terminal_phase: str
    transaction_lock_identity_sha256: str
    previous_result_sha256: str | None


@dataclass(frozen=True, slots=True)
class InstallerReply:
    status: str
    payload: Mapping[str, object]
    response_sha256: str
    handoff_fd: int


@dataclass(frozen=True, slots=True)
class DecodedInstallerResponse:
    status: str
    payload: Mapping[str, object]
    sha256: str


class HandoffPersistenceUncertain(RuntimeError):
    def __init__(self, handoff_fd: int, cause: BaseException):
        super().__init__(f"HANDOFF_PERSISTENCE_UNCERTAIN:{type(cause).__name__}")
        self.handoff_fd = handoff_fd


class TransportUncertain(RuntimeError):
    pass


def lock_identity_from_fd(fd: int) -> LockIdentity:
    status = os.fstat(fd)
    mount_id = None
    with open(f"/proc/self/fdinfo/{fd}", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("mnt_id:"):
                mount_id = int(line.split(":", 1)[1])
                break
    if mount_id is None:
        raise RuntimeError("HANDOFF_MOUNT_ID")
    return LockIdentity(
        status.st_dev, status.st_ino, mount_id, stat.S_IMODE(status.st_mode)
    )


def validate_locked_handoff_fd(fd: int, expected_lock: LockIdentity) -> None:
    if not fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC:
        raise RuntimeError("HANDOFF_NOT_CLOEXEC")
    if lock_identity_from_fd(fd) != expected_lock:
        raise RuntimeError("HANDOFF_LOCK_IDENTITY")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("HANDOFF_LOCK_INTERVENED") from error


def extract_exactly_one_rights_fd(ancillary: list[tuple[int, int, bytes]]) -> int:
    rights: list[int] = []
    invalid_kind = False
    for level, kind, data in ancillary:
        if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
            invalid_kind = True
            continue
        values = array.array("i")
        values.frombytes(data[: len(data) - (len(data) % values.itemsize)])
        rights.extend(values)
    if invalid_kind or len(rights) != 1:
        for fd in rights:
            os.close(fd)
        code = "HANDOFF_ANCILLARY_KIND" if invalid_kind else "HANDOFF_FD_COUNT"
        raise RuntimeError(code)
    return rights[0]


def close_all_rights_fds(ancillary: list[tuple[int, int, bytes]]) -> None:
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            values = array.array("i")
            values.frombytes(data[: len(data) - (len(data) % values.itemsize)])
            for fd in values:
                os.close(fd)


def _recv_exact(channel: socket.socket, count: int) -> bytes:
    value = bytearray()
    while len(value) < count:
        chunk = channel.recv(count - len(value))
        if not chunk:
            raise TransportUncertain("HANDOFF_FRAME_EOF")
        value.extend(chunk)
    return bytes(value)


def decode_exact_response_payload(payload: bytes, *, request_id: str) -> DecodedInstallerResponse:
    data = json.loads(payload)
    keys = {"schema_version", "request_id", "status", "payload", "error"}
    if type(data) is not dict or set(data) != keys:
        raise RuntimeError("HANDOFF_RESPONSE_SCHEMA")
    if data["schema_version"] != 1 or data["request_id"] != request_id:
        raise RuntimeError("HANDOFF_RESPONSE_IDENTITY")
    if type(data["status"]) is not str or type(data["payload"]) is not dict:
        raise RuntimeError("HANDOFF_RESPONSE_TYPE")
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    if canonical != payload:
        raise RuntimeError("HANDOFF_RESPONSE_NONCANONICAL")
    return DecodedInstallerResponse(
        data["status"], data["payload"], hashlib.sha256(payload).hexdigest()
    )


def recv_installer_reply(
    socket_fd: int, *, request_id: str, expected_lock: LockIdentity
) -> InstallerReply:
    duplicate = os.dup(socket_fd)
    with socket.socket(fileno=duplicate) as channel:
        header, ancillary, flags, _ = channel.recvmsg(
            4,
            socket.CMSG_SPACE(array.array("i", [0]).itemsize),
            socket.MSG_CMSG_CLOEXEC | socket.MSG_WAITALL,
        )
        if not header and not ancillary:
            raise TransportUncertain("HANDOFF_FRAME_EOF")
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            close_all_rights_fds(ancillary)
            raise RuntimeError("HANDOFF_ANCILLARY_TRUNCATED")
        received = extract_exactly_one_rights_fd(ancillary)
        try:
            if len(header) != 4:
                raise RuntimeError("HANDOFF_FRAME_HEADER")
            length = struct.unpack("!I", header)[0]
            if length <= 0 or length > 1_048_576:
                raise RuntimeError("HANDOFF_FRAME_LENGTH")
            payload = _recv_exact(channel, length)
        except Exception:
            os.close(received)
            raise
    try:
        validate_locked_handoff_fd(received, expected_lock)
        response = decode_exact_response_payload(payload, request_id=request_id)
        return InstallerReply(
            response.status, response.payload, response.sha256, received
        )
    except Exception:
        os.close(received)
        raise


class UnixInstallerSession:
    def __init__(
        self,
        socket_path: os.PathLike[str],
        *,
        expected_lock: LockIdentity,
    ) -> None:
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.connect(os.fspath(socket_path))
        self._expected_lock = expected_lock
        self._request_id: str | None = None

    def send_request(self, request_bytes: bytes) -> None:
        self._request_id = hashlib.sha256(request_bytes).hexdigest()
        frame = struct.pack("!I", len(request_bytes)) + request_bytes
        self._socket.sendall(frame)

    def receive_reply(self) -> InstallerReply:
        if self._request_id is None:
            raise RuntimeError("INSTALLER_REQUEST_NOT_SENT")
        return recv_installer_reply(
            self._socket.fileno(),
            request_id=self._request_id,
            expected_lock=self._expected_lock,
        )

    def close(self) -> None:
        self._socket.close()


def apply_locked_reply(
    history: Any, reply: InstallerReply, *, expected_lock: LockIdentity
) -> None:
    validate_locked_handoff_fd(reply.handoff_fd, expected_lock)
    try:
        history.persist_reply(reply)
        history.fsync_owner_record_and_parent()
    except BaseException as error:
        history.retain_uncertain_handoff(reply, error)
        raise HandoffPersistenceUncertain(reply.handoff_fd, error) from error
    os.close(reply.handoff_fd)
```

Add the concrete manager owner; `apply_locked_reply` becomes its small success-path helper rather than the owner of descriptor lifetime:

```python
# src/simple/eval_runtime/installer_manager.py
# ruff: noqa: F821
@dataclass(frozen=True, slots=True)
class WireDecision:
    request_bytes: bytes
    authorization_kind: str
    recovery_generation: int | None
    is_resubmission: bool


@dataclass(slots=True)
class RetainedHandoff:
    reply: InstallerReply
    cause: BaseException


class InstallerManagerOwnership:
    def __init__(
        self,
        *,
        store: DiskAuthorizationStore,
        construction_lock_fd: int,
    ) -> None:
        self.store = store
        self._construction_lock_fd = construction_lock_fd
        self._retained: RetainedHandoff | None = None
        self._closed = False
        store.validate_locked_construction_fd(construction_lock_fd)

    @classmethod
    def open(
        cls,
        *,
        store: DiskAuthorizationStore,
        construction_lock_fd: int,
    ) -> "InstallerManagerOwnership":
        return cls(store=store, construction_lock_fd=construction_lock_fd)

    @classmethod
    def restart(cls, root: Path) -> "InstallerManagerOwnership":
        store = DiskAuthorizationStore.open(root)
        construction_lock_fd = store.open_and_lock_construction()
        return cls(store=store, construction_lock_fd=construction_lock_fd)

    def persist_received_reply(
        self,
        decision: WireDecision,
        reply: InstallerReply,
        *,
        fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        if self._closed or self._retained is not None:
            raise RuntimeError("HANDOFF_OWNER_STATE")
        self.store.validate_locked_handoff(reply.handoff_fd)
        try:
            self.store.commit_received_result(
                decision,
                reply,
                fault_hook=fault_hook,
            )
            self.store.fsync_owner_and_parent(fault_hook=fault_hook)
            if fault_hook is not None:
                fault_hook("before_handoff_close")
        except BaseException as error:
            self._retained = RetainedHandoff(reply, error)
            raise HandoffPersistenceUncertain(reply.handoff_fd, error) from error
        os.close(reply.handoff_fd)
        if fault_hook is not None:
            fault_hook("after_handoff_close")

    def reconcile_locked_handoff(self) -> None:
        retained = self._retained
        if retained is None or self._closed:
            raise RuntimeError("NO_RETAINED_HANDOFF")
        self.store.validate_locked_construction_fd(self._construction_lock_fd)
        self.store.validate_locked_handoff(retained.reply.handoff_fd)
        self.store.reconcile_exact_reply(retained.reply)
        self.store.fsync_owner_and_parent()
        os.close(retained.reply.handoff_fd)
        self._retained = None
        self._close_construction_lock()

    def close_without_handoff(self) -> None:
        if self._retained is not None or self._closed:
            raise RuntimeError("HANDOFF_OWNER_STATE")
        self._close_construction_lock()

    def close_terminal_construction_lock(self) -> None:
        if self._retained is not None or self._closed:
            raise RuntimeError("HANDOFF_OWNER_STATE")
        if not self.store.validated_snapshot().terminal:
            raise RuntimeError("INSTALLER_NOT_TERMINAL")
        self._close_construction_lock()

    def _close_construction_lock(self) -> None:
        if not self._closed:
            os.close(self._construction_lock_fd)
            self._closed = True


class InstallerOrchestrator:
    def __init__(
        self,
        *,
        ownership: InstallerManagerOwnership,
        socket_path: Path,
    ) -> None:
        self.ownership = ownership
        self.store = ownership.store
        self.socket_path = socket_path

    @classmethod
    def restart(cls, root: Path, socket_path: Path) -> "InstallerOrchestrator":
        return cls(
            ownership=InstallerManagerOwnership.restart(root),
            socket_path=socket_path,
        )

    def begin_install(self) -> None:
        decision = self.store.prepare_initial_install_delivery()
        try:
            self._exchange(decision, fault_hook=None)
        except TransportUncertain as error:
            self.store.record_transport_uncertain(decision)
            self.store.fsync_owner_and_parent()
            raise RuntimeError("INSTALLER_TRANSPORT_UNCERTAIN") from error

    def close_after_uncertain_transport(self) -> None:
        if not self.store.validated_snapshot().transport_uncertain:
            raise RuntimeError("INSTALLER_NOT_UNCERTAIN")
        self.ownership.close_without_handoff()

    def reconcile_locked_handoff(self) -> None:
        self.ownership.reconcile_locked_handoff()

    def drive_to_terminal(
        self,
        *,
        fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        while not self.store.validated_snapshot().terminal:
            if self._complete_pending_local_transition(fault_hook=fault_hook):
                continue
            decision = self.store.prepare_next_wire_request(fault_hook=fault_hook)
            if decision.is_resubmission and fault_hook is not None:
                fault_hook("before_resubmission")
            try:
                reply = self._exchange(decision, fault_hook=fault_hook)
            except TransportUncertain:
                self.store.record_transport_uncertain(decision)
                self.store.fsync_owner_and_parent()
                continue
            if decision.is_resubmission and fault_hook is not None:
                fault_hook("after_resubmission")
            self.ownership.persist_received_reply(
                decision,
                reply,
                fault_hook=fault_hook,
            )
            if reply.status not in {"NO_JOURNAL_NO_MUTATION", "RECOVERED"}:
                raise RuntimeError("INSTALLER_RESPONSE_STATUS")
        self.ownership.close_terminal_construction_lock()

    def _complete_pending_local_transition(
        self,
        *,
        fault_hook: Callable[[str], None] | None,
    ) -> bool:
        pending = self.store.pending_local_transition()
        if pending == "REACTIVATE_INSTALL":
            if fault_hook is not None:
                fault_hook("before_install_reactivation")
            self.store.reactivate_byte_identical_install()
            self.store.fsync_owner_and_parent()
            if fault_hook is not None:
                fault_hook("after_install_reactivation")
            return True
        if pending == "CLEAR_TERMINAL_RECOVERY":
            if fault_hook is not None:
                fault_hook("before_terminal_clear")
            self.store.mark_terminal_from_fsynced_received_reply()
            self.store.fsync_owner_and_parent()
            if fault_hook is not None:
                fault_hook("after_terminal_clear")
            return True
        if pending is not None:
            raise RuntimeError("INSTALLER_PENDING_LOCAL_TRANSITION")
        return False

    def _exchange(
        self,
        decision: WireDecision,
        *,
        fault_hook: Callable[[str], None] | None,
    ) -> InstallerReply:
        session = UnixInstallerSession(
            self.socket_path,
            expected_lock=self.store.transaction_lock_identity,
        )
        try:
            if fault_hook is not None:
                fault_hook("before_request_delivery")
            session.send_request(decision.request_bytes)
            if fault_hook is not None:
                fault_hook("after_request_delivery")
                fault_hook("before_terminal_handoff_receipt")
            reply = session.receive_reply()
            if fault_hook is not None:
                fault_hook("after_terminal_handoff_receipt")
            return reply
        except (TransportUncertain, BrokenPipeError, ConnectionResetError) as error:
            raise TransportUncertain from error
        finally:
            session.close()
```

`DiskAuthorizationStore` is not an in-memory adapter. It exclusively creates canonical `install-request.bin`, append-only authorization/result JSONL, and an atomically replaced `owner-record.json` beneath a mode-`0700` manager root; every record and replacement is fsynced with its parent before its reference becomes active. Its root manifest pins the construction- and transaction-lock device/inode/mount identities and original install digest. `prepare_next_wire_request` places the authorization-append and active-reference fault hooks around their actual durable operations and returns a `WireDecision` containing the exact bytes that the session will send. `InstallerOrchestrator._exchange` alone owns the request-delivery and terminal-frame-plus-handoff hooks. `persist_received_reply` places result-append and handoff-close hooks around the received reply's actual durable transition. Reactivation, terminal-clear, and resubmission hooks surround those real orchestration operations, never a synthetic reply. `validated_snapshot`, `recovery_decision`, `reconcile_exact_reply`, and restart methods reopen all files with `openat2`/no-follow semantics, reject any invalid suffix or temporary entry, verify both independent hash chains and the active-reference digest, and never infer state from an un-fsynced pending file. Live reconciliation closes neither FD on any error. Restart acquires the pinned construction lock and uses only the fsynced prefix, but it can reach terminal only by driving the next required wire request through a newly opened production session and persisting the newly received response.

`AuthorizationHistory.append_recovery()` requires generation `last + 1`, a fresh token and authorization ID, `previous_entry_sha256=None` for generation 1 and otherwise the exact canonical digest of the preceding authorization entry, a monotonic authorization sequence, and one active reference. `RecoveryResultHistory.append()` separately requires a contiguous result prefix and `previous_result_sha256=None` for its first result and otherwise the preceding result digest. Authorization objects contain no result field, and result entries never participate in the authorization chain. `NO_JOURNAL_NO_MUTATION` reactivates the unchanged install authorization and byte-identical request; every further uncertainty appends a new recovery generation. Neither history overwrites or reorders an entry, marks an older entry active, or uses `pending_action` as authority.

`InstallerManagerOwnership` closes the handed-off OFD only on the success path after both owner-record and parent-directory fsyncs return. On any transition write, replacement, file fsync, or parent fsync failure it retains the still-open descriptor and the already-held stable construction-lock FD in `HANDOFF_PERSISTENCE_UNCERTAIN`. The manager starts no cleanup that would close either FD, sends no further request, and cannot report a terminal clean state. Live reconciliation reopens and compares the owner record and root journal while both locks remain held. Process death releases kernel locks; a fresh owner must acquire the construction lock and replay only the disk store's fsynced active prefix. The two production-backed test families cover live errors and `fork/_exit` at every approved authorization/reference/request/receipt/result/reactivation/clear/close/resubmission boundary.

- [ ] **Step 5: Implement both native installers**

Both binaries use `installer.rs` and enforce this order:

```text
SO_PEERCRED -> fixed request schema -> stable construction lock identity ->
preliminary no-follow current owner record and manager PID/start ticks ->
independently open root-owned transaction lock -> flock(LOCK_EX) ->
close the preliminary owner FD -> fresh openat2 of the current owner-record path ->
full current inode/content/schema/token/
phase/PID/start-ticks/authorization-sequence revalidation -> active authorization
kind/generation/digest -> bind the fresh owner digest into the journal ->
exclusively create/open root journal ->
append/fsync pending -> fork confined mutation child -> validate PREPARED ->
append/fsync PREPARED -> parent renameat2(RENAME_NOREPLACE) -> fsync parents ->
append FINAL_RENAMED -> exclusive receipt create/fsync -> append RECEIPT_CREATED ->
send one terminal response plus duplicated still-locked FD -> close service FD
```

The preliminary owner snapshot is never reused after acquiring the transaction lock. The fresh `openat2` result must be a regular file beneath the fixed control-root FD, match the current pathname's device/inode/mount identity, and pass the complete canonical-content authorization check. A test pauses instance A after its preliminary read, replaces the owner record through the manager's authorized atomic update, lets A acquire the transaction lock, and requires A to reject the stale token while instance B alone proceeds.

The service never calls `LOCK_UN`. Every install/recover instance opens a distinct file description for the exact configured transaction-lock inode. The mutation child receives only the validated token staging bind mount and result pipe, has no source/destination parent, receipt, journal, socket, or sibling-token FD, and may not call rename. The parent performs the one FD-relative rename with retained source-parent/destination-parent FDs and derived basenames. The child drops all capabilities before final read-back; the parent drops all setup/metadata capabilities before the terminal response.

Base-Python journals are keyed by `<intake-sha256>.<operation-token-sha256>.jsonl`; closure journals by `<closure-id>.<operation-token-sha256>.jsonl`. Before `PREPARED`, the only legal final digest is null and recovery removes only the exact token-owned staging inode after `STAGING_REMOVE_PENDING`; at or after `PREPARED`, every record must carry the first bound non-null digest. Root journals, completion metadata, and receipts are never within payload hashes.

- [ ] **Step 6: Add native crash/race tests**

In Rust unit/integration tests inject a barrier or `_exit(99)` at:

```text
before journal O_EXCL; after journal create; before mutation child PREPARED;
after PREPARED; before/after RENAME_PENDING; after rename; after FINAL_RENAMED;
before/after receipt O_EXCL; after receipt fsync; after RECEIPT_CREATED;
before response; before sendmsg; after sendmsg; before local FD close; during shutdown
```

For each point assert a restart produces either no final path or one manifest-valid immutable final path with exactly one byte-identical receipt. Add the mandatory generation-1 no-journal → identical install replay → post-journal uncertainty → generation-2 recovery sentinel. Run two concurrent installers and two concurrent recoverers; require one mutation winner and zero signals/deletions by losers.

Add a production-service owner-record race: instance A reads owner generation 1 and pauses before the transaction lock; the manager atomically installs generation 2; instance B queues; A acquires first, reopens the current owner path, and must return `AUTHORIZATION_STALE` without journal or mutation; B then reopens generation 2 and is the sole journal/mutation winner. Assert the service never authorizes from the preliminary FD or its cached bytes.

- [ ] **Step 7: Run GREEN, rebuild static binaries, and commit**

Run:

```bash
bash scripts/build_psi0_eval_native.sh
CARGO_NET_OFFLINE=true cargo test --offline --locked \
  --manifest-path native/psi0_eval_runtime/Cargo.toml installer
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_installer_protocol.py \
  tests/eval_runtime/test_installer_recovery.py \
  tests/eval_runtime/test_installer_manager_restart.py \
  tests/eval_runtime/test_systemd_contracts.py
```

Expected: all Python/Rust tests pass, including every handoff boundary and both recovery generations.

```bash
git add src/simple/eval_runtime/installer_client.py \
  src/simple/eval_runtime/installer_manager.py native/psi0_eval_runtime \
  deploy/psi0_eval tests/eval_runtime scripts/build_psi0_eval_native.sh
git commit -m "feat: add replay-safe PSI0 input installers"
```

### Task 6: Implement detached remote helpers, lease CAS, GPU attribution, and stopped-container lifecycle

**Files:**
- Create: `src/simple/eval_runtime/remote.py`
- Create: `src/simple/eval_runtime/lease.py`
- Create: `src/simple/eval_runtime/gpu.py`
- Create: `src/simple/eval_runtime/container.py`
- Modify: `native/psi0_eval_runtime/src/bin/psi0-eval-remote-helper.rs`
- Create: `configs/psi0_h100_eval_seccomp_v1.json`
- Create: `tests/eval_runtime/test_remote_helper.py`
- Create: `tests/eval_runtime/test_lease.py`
- Create: `tests/eval_runtime/test_gpu.py`
- Create: `tests/eval_runtime/test_container.py`
- Modify: `tests/eval_runtime/static_files.txt`

- [ ] **Step 1: Write failing lease and stale-recovery tests**

```python
# tests/eval_runtime/test_lease.py
from __future__ import annotations

import threading

import pytest

from simple.eval_runtime.lease import acquire_normal, claim_recovery
from .fakes import make_expired_lease_store, make_lease_store


def test_two_concurrent_managers_have_exactly_one_lease_winner(tmp_path) -> None:
    store = make_lease_store(tmp_path)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def contender(run_id: str) -> None:
        barrier.wait()
        outcomes.append(acquire_normal(store, run_id=run_id).status)

    threads = [threading.Thread(target=contender, args=(f"run-{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["ACQUIRED", "LEASE_BLOCKED"]


def test_heartbeat_does_not_wait_for_mutation_lock(tmp_path) -> None:
    store = make_lease_store(tmp_path)
    lease = acquire_normal(store, run_id="run-a").lease
    store.hold_mutation_lock()
    before = lease.heartbeat_monotonic_ns
    refreshed = store.heartbeat(lease)
    assert refreshed.heartbeat_monotonic_ns > before


@pytest.mark.parametrize("operation", ["freeze-provenance", "create", "evaluate"])
def test_recovery_requires_operation_specific_write_ahead_evidence(tmp_path, operation: str) -> None:
    store = make_expired_lease_store(tmp_path, operation=operation, cleanup_required=True)
    if operation == "freeze-provenance":
        store.write_freeze_journal()
    elif operation == "create":
        store.write_container_journal()
    else:
        store.write_evaluate_journal()
    claim = claim_recovery(store, expected_run_id=store.lease.run_id)
    assert claim.mode == "recovery"
    assert claim.generation == store.lease.generation + 1
    assert claim.recovery_of_token_sha256 == store.lease.token_sha256


def test_foreign_or_unknown_running_container_is_never_stopped(tmp_path) -> None:
    store = make_expired_lease_store(tmp_path, operation="evaluate", cleanup_required=True)
    store.write_evaluate_journal(container_owner="foreign")
    result = claim_recovery(store, expected_run_id=store.lease.run_id).reconcile()
    assert result.status == "FOREIGN_BLOCKED"
    assert result.signal_calls == []
    assert result.docker_stop_calls == []
```

- [ ] **Step 2: Write failing helper, GPU, and container tests**

```python
# tests/eval_runtime/test_remote_helper.py
from __future__ import annotations

from simple.eval_runtime.remote import RemoteHelperClient
from .fakes import FakeRemoteTransport


def test_ssh_disconnect_reconciles_detached_helper_without_duplicate_launch() -> None:
    transport = FakeRemoteTransport(disconnect_after="detached_ready")
    client = RemoteHelperClient(transport)
    result = client.run(helper_id="helper-a", operation="inspect", internal_seconds=10, outer_seconds=25)
    assert result.state == "completed"
    assert transport.launch_count == 1
    assert transport.read_only_reconciliations == 1


def test_internal_timeout_reaps_exact_child_group_and_records_daemon_postcondition() -> None:
    transport = FakeRemoteTransport(child_hangs=True)
    result = RemoteHelperClient(transport).run(
        helper_id="helper-timeout", operation="docker-stop", internal_seconds=10, outer_seconds=25
    )
    assert result.state == "timed_out"
    assert result.signal_sequence == ["INT", "TERM", "KILL"]
    assert result.post_kill_alive is False
    assert result.daemon_postcondition == "exited"
```

```python
# tests/eval_runtime/test_gpu.py
from __future__ import annotations

import pytest

from simple.eval_runtime.gpu import GpuGate, GpuInventory, GpuProcess


def test_h100_gate_accepts_idle_configured_container_but_no_compute_process() -> None:
    inventory = GpuInventory(uuid="GPU-7", compute_processes=(), configured_containers=("idle",))
    assert GpuGate(expected_uuid="GPU-7").validate_before_start(inventory).uuid == "GPU-7"


def test_foreign_compute_process_fails_without_signal() -> None:
    inventory = GpuInventory(
        uuid="GPU-7",
        compute_processes=(GpuProcess(pid=88, start_ticks=7, used_memory=100, owner="foreign"),),
        configured_containers=(),
    )
    with pytest.raises(Exception, match="GPU_FOREIGN"):
        GpuGate(expected_uuid="GPU-7").validate_before_start(inventory)


def test_pc2_monitor_rejects_missed_poll_and_non_descendant() -> None:
    gate = GpuGate(expected_uuid="GPU-PC2", allowed_processes={(99, 123)})
    assert gate.validate_poll(GpuInventory("GPU-PC2", (GpuProcess(99, 123, 50, "owned"),), ()), 5.0)
    with pytest.raises(Exception, match="GPU_POLL_MISSED"):
        gate.validate_poll(GpuInventory("GPU-PC2", (), ()), 10.1)
```

```python
# tests/eval_runtime/test_container.py
from __future__ import annotations

import pytest

from simple.eval_runtime.container import validate_container
from .fakes import make_container_contract, make_docker_inspect


def test_container_contract_has_no_published_ports_or_control_mount_alias() -> None:
    inspect = make_docker_inspect()
    validate_container(make_container_contract(), inspect)
    assert inspect.published_ports == {}
    assert inspect.network_mode != "host"
    assert all("control" not in mount.destination for mount in inspect.mounts)
    assert all("inputs" not in mount.source or mount.read_only for mount in inspect.mounts)


@pytest.mark.parametrize("state", ["created", "running", "paused", "restarting", "removing", "dead", "unknown"])
def test_evaluate_requires_initial_exited_state(state: str) -> None:
    inspect = make_docker_inspect(state=state)
    with pytest.raises(Exception, match="CONTAINER_NOT_EXITED"):
        validate_container(make_container_contract(), inspect, for_evaluate=True)
```

- [ ] **Step 3: Run RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_lease.py \
  tests/eval_runtime/test_remote_helper.py tests/eval_runtime/test_gpu.py \
  tests/eval_runtime/test_container.py
```

Expected: four import failures.

- [ ] **Step 4: Implement detached helper launch and reconciliation**

The static helper accepts one canonical request on stdin, validates its fixed operation enum, exclusively creates `transactions/<helper-id>/`, forks, and makes the child durable before the SSH-facing parent responds. Implement this exact child ordering:

```text
setsid -> child process group -> stdin /dev/null -> stdout/stderr owned logs ->
close inherited SSH descriptors -> install internal monotonic deadline ->
write/fsync detached_ready record -> release work-child start barrier
```

The transaction state is `launching`, `detached_ready`, then one of `completed`, `failed`, `timed_out`, or `cleanup_failed`. Every work child PID/start ticks/argv digest/parent/PGID is fsynced before it crosses the barrier. Deadline handling sends INT, waits, TERM, waits, KILL, then starts a new bounded post-KILL wait and records liveness plus the Docker/filesystem postcondition. `RemoteHelperClient` accepts launch only after a fresh read-only SSH reconciliation of the acknowledgement; disconnect before acknowledgement reconciles the helper ID before any retry.

- [ ] **Step 5: Implement lease/mutation lock ordering and operation-specific recovery**

`LeaseStore` validates the exact schema from the design and uses `lease.lock` only for acquire, heartbeat, claim, cleanup mark, and release. Mutation helpers take `mutation.lock` first and briefly take `lease.lock` only to revalidate; no code path may acquire them in the reverse order. Heartbeat is every 10 seconds, expiry is 45 seconds, and `cleanup_required` begins true for `freeze-provenance`, `create`, and `evaluate`.

Normal reclamation accepts only expired, `cleanup_required=false`, `pending_mutation=null`, exited/absent container, no port/server/GPU/helper. Recovery claim atomically changes token, increments generation, records old-token hash, and applies the exact freeze/create/evaluate predicate. Foreign or unknown identity never signals or stops.

- [ ] **Step 6: Implement UUID/process attribution and five-second monitors**

`gpu.py` parses structured `nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits` and `--query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits`. PC2's CUDA probe runs as a tracked child with `CUDA_VISIBLE_DEVICES=1`, allocates one tensor, and emits CUDA device-0 UUID; H100 container probe must emit exactly one CUDA UUID matching physical GPU 7. A poll is valid only when every compute PID maps through `/proc/<pid>/stat`, cgroup/container identities, and the current server/evaluator ancestry. Both episode monitors poll at most five seconds apart; each command deadline is ten seconds; a missed, malformed, or unknown poll fails.

- [ ] **Step 7: Implement the digest-pinned stopped container contract**

`container.py` generates Docker argv with the fixed name, `--gpus device=7`, `--init`, `--restart=no`, private bridge, no `-p`, 64 GiB private shm, read-only root, exact tmpfs mounts, `--cap-drop=ALL`, `--cap-add=SYS_PTRACE`, `no-new-privileges`, protected seccomp host path, protected source/checkpoint read-only mounts, and disjoint writable workload mount. It validates every `docker inspect` field and root alias. `create` starts only for model-free probes and stops to exact `exited`; it never starts the official server. `evaluate` rejects every initial state except exact `exited`.

The committed seccomp JSON has `defaultAction="SCMP_ACT_ERRNO"`, the fixed architecture list, only the syscall allowlist required by the unchanged server plus interruptible `strace`, and argument filters that deny process-VM writes, ptrace kill-on-exit behavior, daemonization, mount/namespace changes, and arbitrary bpf. The Docker inspect digest must match the protected seccomp bytes under the sealed server snapshot.

- [ ] **Step 8: Run GREEN and commit**

Run:

```bash
bash scripts/build_psi0_eval_native.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_lease.py \
  tests/eval_runtime/test_remote_helper.py tests/eval_runtime/test_gpu.py \
  tests/eval_runtime/test_container.py
```

Expected: all tests pass and no external-runtime guard fires.

```bash
git add src/simple/eval_runtime native/psi0_eval_runtime \
  configs/psi0_h100_eval_seccomp_v1.json tests/eval_runtime
git commit -m "feat: own dedicated H100 runtime lifecycle"
```

### Task 7: Implement the private-root evaluator supervisor and operation journals

**Files:**
- Create: `src/simple/eval_runtime/runner_client.py`
- Create: `native/psi0_eval_runtime/src/runner.rs`
- Modify: `native/psi0_eval_runtime/src/bin/psi0-eval-run-pc2-evaluator.rs`
- Create: `deploy/psi0_eval/pc2-runner-v1.json`
- Create: `deploy/psi0_eval/psi0-eval-pc2-runner.socket`
- Create: `deploy/psi0_eval/psi0-eval-pc2-runner@.service`
- Create: `tests/eval_runtime/test_runner_protocol.py`
- Create: `tests/eval_runtime/test_runner_recovery.py`
- Create: `tests/eval_runtime/test_runner_sandbox.py`
- Modify: `tests/eval_runtime/static_files.txt`

- [ ] **Step 1: Write failing operation/FD schema tests**

```python
# tests/eval_runtime/test_runner_protocol.py
from __future__ import annotations

import pytest

from simple.eval_runtime.runner_client import OPERATION_FD_ROLES
from .fakes import RunnerHarness, make_runner_request

EXPECTED = {
    "prepare_output": ("workload-parent-root",),
    "loader_probe": ("closure-root", "base-Python-root", "identity-sidecar", "workload-parent-root"),
    "cuda_probe": ("closure-root", "base-Python-root", "identity-sidecar", "workload-parent-root"),
    "evaluate_episode": (
        "event-write", "acknowledgement-read", "closure-root", "base-Python-root",
        "identity-sidecar", "workload-parent-root",
    ),
    "finalize_episode": ("workload-parent-root",),
    "recover_episode": ("workload-parent-root",),
}


def test_runner_operation_fd_roles_are_exact() -> None:
    assert OPERATION_FD_ROLES == EXPECTED


@pytest.mark.parametrize("operation", sorted(EXPECTED))
def test_runner_rejects_missing_extra_or_reordered_fd(operation: str) -> None:
    roles = list(EXPECTED[operation])
    mutations = [roles[:-1], roles + ["extra"], list(reversed(roles))]
    for malformed in mutations:
        if malformed == roles:
            continue
        with pytest.raises(Exception, match="FD_ROLES"):
            make_runner_request(operation=operation, fd_roles=malformed).validate()


def test_full_sequence_uses_distinct_children_under_one_output_root() -> None:
    harness = RunnerHarness()
    harness.prepare_output()
    harness.loader_probe()
    harness.cuda_probe()
    harness.evaluate_episode(2)
    harness.finalize_episode(2)
    harness.evaluate_episode(3)
    harness.finalize_episode(3)
    assert harness.children == ["probe_loader", "probe_cuda", "episode_2", "episode_3"]
    assert harness.output_root_creations == 1
    assert harness.cross_child_visibility_attempts == []
```

`RunnerHarness` is defined in this file as an adapter over the native `--test-root` protocol. It creates real temporary Unix sockets, FDs, files, locks, and forked dummy children but substitutes namespace/mount/GPU syscalls through the native test backend.

- [ ] **Step 2: Write failing sandbox and recovery tests**

```python
# tests/eval_runtime/test_runner_sandbox.py
from __future__ import annotations

from .fakes import make_sandbox_manifest


def test_sandbox_has_only_two_read_roots_and_one_operation_write_root() -> None:
    manifest = make_sandbox_manifest()
    assert manifest.network.interfaces == ("lo",)
    assert manifest.network.default_routes == ()
    assert manifest.network.host_namespace is False
    assert manifest.writable_mounts == ("/run-output", "/tmp")
    assert manifest.readonly_roots == ("/sealed/closure", "/sealed/base-python")
    forbidden = ("/home", "/mnt/data", "/run/psi0-simple-eval", "/etc/ld.so.cache")
    assert all(path not in manifest.visible_paths for path in forbidden)


def test_loader_argv_uses_open_fd_and_inhibits_host_cache() -> None:
    argv = make_sandbox_manifest().loader_argv(loader_fd=9)
    assert argv[:6] == [
        "9", "--inhibit-cache", "--library-path",
        "/sealed/closure/venv/lib:/sealed/base-python/lib:/sealed/base-python/lib64:/sealed/base-python/usr/lib:/sealed/base-python/usr/lib64",
        "--argv0", "/sealed/closure/venv/bin/python",
    ]
    assert argv[6] == "/sealed/closure/venv/bin/python"
```

```python
# tests/eval_runtime/test_runner_recovery.py
from __future__ import annotations

import pytest

from .fakes import RunnerHarness


@pytest.mark.parametrize(
    "crash",
    [
        "after_output_child_pending", "after_child_create", "after_upstream_connect_pending",
        "after_tcp_connect", "after_relay_start_pending", "after_relay_fork",
        "after_relay_started", "after_evaluator_launch", "during_policy_request",
    ],
)
def test_recovery_reaps_only_journaled_cgroup_and_sockets(crash: str) -> None:
    harness = RunnerHarness(crash=crash)
    harness.evaluate_episode(2)
    recovered = harness.recover_episode(2, mode="live-manager")
    assert recovered.terminal is True
    assert recovered.cgroup_members == ()
    assert recovered.unix_socket_inodes == ()
    assert recovered.upstream_socket_inode is None
    assert recovered.foreign_signals == []


def test_second_episode_is_blocked_when_first_cleanup_is_not_terminal() -> None:
    harness = RunnerHarness(crash="foreign_cgroup_member")
    harness.evaluate_episode(2)
    assert harness.recover_episode(2, mode="live-manager").status == "FOREIGN_BLOCKED"
    with pytest.raises(Exception, match="PREVIOUS_EPISODE_NOT_TERMINAL"):
        harness.evaluate_episode(3)
```

- [ ] **Step 3: Run RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_runner_protocol.py \
  tests/eval_runtime/test_runner_sandbox.py \
  tests/eval_runtime/test_runner_recovery.py
```

Expected: imports fail for `runner_client`.

- [ ] **Step 4: Implement Python request validation and client ownership**

`runner_client.py` uses the exact operation table above. It opens closure/base/sidecar/workload roots with `O_PATH|O_DIRECTORY|O_NOFOLLOW` or `openat2`, verifies mount/device/inode/profile identities, and transfers only the named descriptors. `prepare_output` is called exactly once. Probes and episodes use distinct absent children. `RunnerSession.close()` asks the same supervisor connection to stop evaluator descendants, close Unix/upstream sockets, stop relay, journal `DESCENDANTS_GONE`, then waits for service/cgroup absence and calls `finalize_episode`; it never signals a child directly.

- [ ] **Step 5: Implement the static runner and write-ahead episode journal**

The native runner verifies `SO_PEERCRED`, the fixed profile/config/binary identities, exact operation/FDS, and root/output identities before mutation. For `evaluate_episode`, append and fsync these phases in order:

```text
RESERVED -> OUTPUT_CHILD_PENDING -> OUTPUT_CHILD_CREATED ->
UPSTREAM_CONNECT_PENDING -> UPSTREAM_CONNECTED ->
RELAY_START_PENDING -> RELAY_STARTED ->
EVALUATOR_START_PENDING -> EVALUATOR_STARTED -> STOPPING ->
DESCENDANTS_GONE
```

Only a later `finalize_episode` or `recover_episode` instance may append `TERMINAL` after proving the original service cgroup, evaluator/relay PIDs, both Unix socket inodes, and upstream TCP inode absent. The operation lock is stable and root-owned. Recovery signals only exact recorded cgroup members; pending-socket recovery accepts zero sockets or one loopback connection attributable to the recorded tunnel. Multiple, non-loopback, foreign, or uninspectable sockets return `FOREIGN_BLOCKED` without a signal.

Before evaluator exec the runner unshares mount/network namespaces, pivots to an empty tmpfs root, mounts only the reviewed table, creates `lo` only, drops supplementary groups/capabilities to the dedicated evaluator UID/GID, sets no-new-privileges, and applies the reviewed seccomp/device policy. It opens the copied no-`PT_INTERP` dynamic loader relative to the base FD and calls `execveat(AT_EMPTY_PATH)` with the exact argv in the design. `loader_probe` performs loader `--verify`, `--list`, `python -I -S`, `LD_DEBUG`, syscall trace, and `/proc/<pid>/maps` reconciliation; every regular executable mapping/access must be below the two sealed roots.

- [ ] **Step 6: Add systemd runner units**

The socket uses `Accept=yes` at `/run/psi0-simple-eval/evaluator-launcher.sock`. The service runs the exact static launcher, uses fixed profile/config read paths, `IPAddressDeny=any`, `IPAddressAllow=localhost`, `RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6`, `KillMode=control-group`, bounded `RuntimeMaxSec`, no installer socket/control/staging mounts, and only the reviewed capability set until the runner drops it by phase. Unit tests parse every property and reject an override, extra writable host path, or added capability.

- [ ] **Step 7: Build, run GREEN, and commit**

Run:

```bash
bash scripts/build_psi0_eval_native.sh
CARGO_NET_OFFLINE=true cargo test --offline --locked \
  --manifest-path native/psi0_eval_runtime/Cargo.toml runner
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_runner_protocol.py \
  tests/eval_runtime/test_runner_sandbox.py \
  tests/eval_runtime/test_runner_recovery.py \
  tests/eval_runtime/test_systemd_contracts.py
```

Expected: all protocol, sandbox, crash, and unit-parsing tests pass without simulator import.

```bash
git add src/simple/eval_runtime/runner_client.py native/psi0_eval_runtime \
  deploy/psi0_eval tests/eval_runtime scripts/build_psi0_eval_native.sh
git commit -m "feat: add sealed PC2 evaluator supervisor"
```

### Task 8: Implement the connected-FD policy relay and deterministic warm-up

**Files:**
- Create: `src/simple/eval_runtime/warmup.py`
- Create: `native/psi0_eval_runtime/src/relay.rs`
- Modify: `native/psi0_eval_runtime/src/bin/psi0-eval-policy-relay.rs`
- Create: `deploy/psi0_eval/pc2-policy-relay-v1.json`
- Modify: `src/simple/baselines/client.py`
- Modify: `src/simple/baselines/psi0_decoupled_wbc.py`
- Create: `tests/eval_runtime/test_policy_relay.py`
- Create: `tests/eval_runtime/test_warmup.py`
- Modify: `tests/test_http_action_client.py`
- Modify: `tests/eval_runtime/static_files.txt`

- [ ] **Step 1: Write failing connected-FD and byte-exact warm-up tests**

```python
# tests/eval_runtime/test_warmup.py
from __future__ import annotations

import hashlib
import json

import numpy as np

from simple.eval_runtime.warmup import build_warmup


def test_warmup_uses_production_serializer_and_fixed_values() -> None:
    warmup = build_warmup()
    request = warmup.request
    assert set(request) == {
        "image", "instruction", "history", "state", "condition",
        "gt_action", "dataset_name", "timestamp",
    }
    assert request["instruction"] == "move forward to the table and pick up the object"
    assert request["history"] == {"reset": True}
    assert request["condition"] == {}
    assert request["gt_action"] == []
    assert request["dataset_name"] == "simple"
    assert request["timestamp"] == "1970-01-01_00-00-00"
    decoded = warmup.deserialize()
    np.testing.assert_array_equal(decoded.image["rgb_head_stereo_left"], np.zeros((360, 640, 3), np.uint8))
    expected_state = np.zeros((1, 32), np.float32)
    expected_state[0, 31] = 0.74
    np.testing.assert_array_equal(decoded.state["states"], expected_state)
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert warmup.canonical_request_sha256 == hashlib.sha256(canonical).hexdigest()
```

```python
# tests/eval_runtime/test_policy_relay.py
from __future__ import annotations

import socket

import pytest

from simple.baselines.client import FdActionClient
from .fakes import FakeRelay


def test_fd_client_round_trips_one_canonical_request_without_endpoint() -> None:
    client_socket, relay_socket = socket.socketpair()
    relay = FakeRelay(relay_socket, action_shape=(24, 36))
    relay.start()
    client = FdActionClient(client_socket.detach(), timeout=5.0)
    action, error, trajectory = client.query_action({}, "instruction", {}, {})
    assert action.shape == (24, 36)
    assert error == 0.0
    assert trajectory is None
    assert relay.requests == [1]
    relay.join()


def test_fd_client_has_no_host_port_url_or_reconnect_path() -> None:
    client_socket, relay_socket = socket.socketpair()
    client = FdActionClient(client_socket.detach(), timeout=5.0)
    assert not hasattr(client, "server_ip")
    assert not hasattr(client, "server_port")
    assert not hasattr(client, "session")
    relay_socket.close()
    with pytest.raises(Exception, match="POLICY_TRANSPORT_EOF"):
        client.query_action({}, "instruction", {}, {})
    assert client.reconnect_attempts == 0
```

`FakeRelay` is a thread-backed exact framing peer defined in the test file; it uses `RequestMessage.deserialize`, returns a `ResponseMessage.serialize()` object, enforces sequence 1 and one request in flight, and never opens a network socket.

- [ ] **Step 2: Run RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_warmup.py \
  tests/eval_runtime/test_policy_relay.py tests/test_http_action_client.py
```

Expected: `build_warmup` and `FdActionClient` imports fail; existing HTTP tests still pass when selected alone.

- [ ] **Step 3: Implement warm-up through the production serializer**

```python
# src/simple/eval_runtime/warmup.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from simple.baselines.client import RequestMessage


@dataclass(frozen=True, slots=True)
class Warmup:
    request: dict[str, object]
    canonical_request_sha256: str

    def deserialize(self) -> RequestMessage:
        return RequestMessage.deserialize(self.request)


def build_warmup() -> Warmup:
    state = np.zeros((1, 32), dtype=np.float32)
    state[0, 31] = np.float32(0.74)
    message = RequestMessage(
        image={"rgb_head_stereo_left": np.zeros((360, 640, 3), dtype=np.uint8)},
        instruction="move forward to the table and pick up the object",
        history={"reset": True},
        state={"states": state},
        condition={},
        gt_action=[],
        dataset_name="simple",
        timestamp="1970-01-01_00-00-00",
    )
    serialized = message.serialize()
    encoded = json.dumps(serialized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return Warmup(serialized, hashlib.sha256(encoded).hexdigest())
```

- [ ] **Step 4: Add `FdActionClient` without changing unmanaged HTTP behavior**

`FdActionClient` owns one already-connected `AF_UNIX SOCK_STREAM` FD, uses four-byte big-endian lengths and canonical JSON, maintains sequence starting at 1, allows one request in flight, caps frames at the reviewed contract, and treats EOF/timeout/malformed/status errors as terminal. Put the exact-length framing in module functions `recv_policy_frame(fd) -> tuple[int, dict[str, object]]` and `send_policy_frame(fd, sequence, payload) -> None`; both enforce the exact `{schema_version, sequence, payload}` key set and are shared by the test relay. The client exposes the same `query_action` signature as `HttpActionClient`; it has no host, port, URL, session, socket creation, resolver, or reconnect method. Existing `HttpActionClient` behavior and tests remain unchanged.

Change `Psi0DecoupledWbcAgent.__init__` to accept exactly one of unmanaged `(host, port)` or managed `policy_transport_fd`. When managed, instantiate `FdActionClient`; passing both or neither raises before any WBC/simulator construction.

- [ ] **Step 5: Implement the networkless static relay**

The runner passes only the Unix policy socket, one already-connected upstream TCP FD, barrier, diagnostic pipe, and fixed contract FD. The relay validates all FD types/inodes before barrier release, then converts one canonical framed request at a time into fixed `POST /act` HTTP/1.1 on the existing TCP FD. It returns sequence, status, length, and response bytes. The relay never receives or inspects host/port/URL, never creates/connects/binds/listens/resolves a socket, never pipelines, and never reconnects. Apply a seccomp allowlist whose negative test proves `socket`, `connect`, `bind`, `listen`, `getsockname`, `getpeername`, resolver file access, and namespace syscalls return `EPERM`.

- [ ] **Step 6: Add relay boundary and crash tests**

Test exact maximum sizes, partial reads/writes, wrong sequence, extra bytes, two simultaneous requests, HTTP non-200, upstream EOF, deadline, relay crash, and runner-detected tunnel drift. For each terminal transport failure assert evaluator failure, zero reconnect calls, relay exit, Unix socket absence, upstream socket absence, and a fresh independently journaled relay for the next episode.

- [ ] **Step 7: Build, run GREEN, and commit**

Run:

```bash
bash scripts/build_psi0_eval_native.sh
CARGO_NET_OFFLINE=true cargo test --offline --locked \
  --manifest-path native/psi0_eval_runtime/Cargo.toml relay
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_warmup.py \
  tests/eval_runtime/test_policy_relay.py tests/test_http_action_client.py
```

Expected: all relay/warm-up/client tests pass; legacy unbounded HTTP `/act` behavior remains green.

```bash
git add src/simple/eval_runtime/warmup.py src/simple/baselines \
  native/psi0_eval_runtime deploy/psi0_eval tests/eval_runtime \
  tests/test_http_action_client.py
git commit -m "feat: add connected-FD PSI0 policy transport"
```

### Task 9: Add the evaluator runtime contract, sequenced evidence, and Python cleanup lifecycle

**Files:**
- Create: `src/simple/eval_runtime/events.py`
- Modify: `src/simple/evals/api.py`
- Modify: `src/simple/cli/eval_decoupled_wbc.py`
- Create: `tests/eval_runtime/test_runtime_contract.py`
- Create: `tests/eval_runtime/test_runtime_events.py`
- Create: `tests/eval_runtime/test_evaluator_fd_plumbing.py`
- Create: `tests/eval_runtime/test_evaluator_python_lifecycle.py`
- Modify: `tests/eval_runtime/static_files.txt`

- [ ] **Step 1: Write failing contract-before-environment tests**

```python
# tests/eval_runtime/test_runtime_contract.py
from __future__ import annotations

import pytest

from simple.cli.eval_decoupled_wbc import _validate_managed_runtime
from .fakes import ManagedEvaluatorHarness


@pytest.mark.parametrize(
    ("field", "value"),
    [("ENV_TYPE", "real"), ("INTERFACE", "eth0"), ("DOMAIN_ID", 1)],
)
def test_unsafe_wbc_value_writes_rejection_before_gym_make(field: str, value: object) -> None:
    harness = ManagedEvaluatorHarness.valid()
    harness.sonic_config[field] = value
    with pytest.raises(Exception, match="UNSAFE_WBC_CONFIG"):
        _validate_managed_runtime(harness.options, harness.sonic_config, harness.ops)
    assert harness.durable_record()["status"] == "rejected"
    assert [event["event"] for event in harness.events] == ["runtime_contract"]
    assert harness.events[0]["payload"]["status"] == "rejected"
    assert harness.gym_make_calls == 0


def test_valid_contract_waits_for_ack_before_creating_environment() -> None:
    harness = ManagedEvaluatorHarness.valid()
    validated = _validate_managed_runtime(harness.options, harness.sonic_config, harness.ops)
    assert validated.status == "validated"
    assert harness.order == [
        "evidence_fsync", "runtime_contract_write", "ack_read", "creating_env_write"
    ]
    assert harness.gym_make_calls == 0
```

`ManagedEvaluatorHarness` is defined in the same test file. It creates a private temp output directory, pipe pairs, closure/base/sidecar directory FDs, a Unix socketpair for policy transport, exact profile/relay identities, and a fake `gym.make` counter. It uses real atomic evidence writes and pipe bytes.

- [ ] **Step 2: Write failing event schema/order and parent-plumbing tests**

```python
# tests/eval_runtime/test_runtime_events.py
from __future__ import annotations

import pytest

from .fakes import inject_event_malformation, make_event_reader, make_runtime_event


def test_event_sequence_is_contiguous_and_payload_keys_are_exact() -> None:
    reader = make_event_reader(run_id="run-a", episode_index=2, evaluator_pid=100)
    reader.accept(make_runtime_event(sequence=1, event="runtime_contract"))
    reader.accept(make_runtime_event(sequence=2, event="worker_init", payload={"status": "creating_env", "total_episodes": 1}))
    with pytest.raises(Exception, match="EVENT_SEQUENCE"):
        reader.accept(make_runtime_event(sequence=4, event="episode_start"))
    with pytest.raises(Exception, match="EVENT_PAYLOAD"):
        reader.accept(make_runtime_event(sequence=3, event="worker_status", payload={"status": "closing", "extra": 1}))


@pytest.mark.parametrize("malformation", ["utf8", "json", "partial", "oversize", "early_eof"])
def test_event_channel_malformation_fails_closed(malformation: str) -> None:
    reader = make_event_reader()
    with pytest.raises(Exception, match="EVENT_CHANNEL"):
        inject_event_malformation(reader, malformation)


def test_callback_only_cannot_satisfy_runtime_contract() -> None:
    reader = make_event_reader()
    reader.observe_tui_callback({"event": "runtime_contract"})
    with pytest.raises(Exception, match="EVENT_CHANNEL_SILENT"):
        reader.require_contract(deadline=reader.clock() + 1.0)
```

```python
# tests/eval_runtime/test_evaluator_fd_plumbing.py
from __future__ import annotations

import pytest

from simple.cli.eval_decoupled_wbc import EvalConfig, run_eval


MANAGED = {
    "runtime_evidence_path": "/run-output/wbc-runtime-contract.json",
    "runtime_evidence_run_id": "run-a",
    "runtime_evidence_nonce": "a" * 32,
    "runtime_event_fd": 3,
    "runtime_ack_fd": 4,
    "runtime_root_fd": 5,
    "runtime_base_python_root_fd": 6,
    "runtime_identity_fd": 7,
    "policy_transport_fd": 8,
}


@pytest.mark.parametrize("missing", sorted(MANAGED))
def test_six_fd_contract_and_three_options_are_indivisible(missing: str) -> None:
    values = dict(MANAGED)
    values.pop(missing)
    config = EvalConfig(env_id="simple/Fake-v0", policy="psi0_decoupled_wbc", **values)
    with pytest.raises(ValueError, match="managed runtime contract"):
        run_eval(config, show_progress=False)


def test_managed_contract_requires_one_worker() -> None:
    config = EvalConfig(
        env_id="simple/Fake-v0", policy="psi0_decoupled_wbc", num_workers=2, **MANAGED
    )
    with pytest.raises(ValueError, match="num_workers=1"):
        run_eval(config, show_progress=False)
```

```python
# tests/eval_runtime/test_evaluator_python_lifecycle.py
from __future__ import annotations

from collections import defaultdict

import pytest

from simple.cli.eval_decoupled_wbc import _execute_managed_worker_lifecycle

FAILURES = (
    "gym_make", "agent_make", "recorder_make", "reset", "stabilization", "policy_request", "render",
    "video_write", "result_persistence", "video_release", "agent_close",
    "environment_close", "terminal_persistence",
)


class SimulatorFreeWorkerOps:
    def __init__(self, fail_at: str | None):
        self.fail_at = fail_at
        self.calls: list[str] = []
        self.events: list[tuple[str, object]] = []
        self.persisted: dict[str, bool] | None = None
        self.terminal: tuple[tuple[str, ...], dict[str, bool]] | None = None

    def call(self, phase: str):
        self.calls.append(phase)
        if self.fail_at == phase:
            raise RuntimeError(phase)
        return object()

    def make_env(self):
        return self.call("gym_make")

    def make_agent(self, env):
        del env
        return self.call("agent_make")

    def episodes(self):
        return ("episode_2", "episode_3")

    def new_stats(self):
        return defaultdict(bool)

    def make_recorder(self, env, episode):
        del env, episode
        return self.call("recorder_make")

    def reset(self, recorder):
        del recorder
        return self.call("reset")

    def stabilize(self, env, observation):
        del env, observation
        self.call("stabilization")

    def request_policy(self, agent, observation):
        del agent, observation
        return self.call("policy_request")

    def render(self, env):
        del env
        return self.call("render")

    def write_video(self, recorder, frame):
        del recorder, frame
        self.call("video_write")

    def run_policy_episode(self, agent, recorder, observation, episode):
        del episode
        self.request_policy(agent, observation)
        frame = self.render(recorder)
        self.write_video(recorder, frame)

    def episode_success(self, env, episode):
        del env
        return episode == "episode_2"

    def report_episode_start(self, episode):
        self.events.append(("episode_start", episode))

    def report_episode_end(self, episode, success):
        self.events.append(("episode_end", (episode, success)))

    def persist_result(self, stats):
        self.call("result_persistence")
        self.persisted = dict(stats)

    def report_worker_closing(self, stats):
        self.events.append(("worker_status", ("closing", dict(stats))))

    def report_worker_done(self, stats):
        self.events.append(("worker_done", dict(stats)))

    def release_recorder(self, recorder):
        del recorder
        self.call("video_release")

    def close_agent(self, agent):
        del agent
        self.call("agent_close")

    def close_env(self, env):
        del env
        self.call("environment_close")

    def persist_terminal(self, errors, stats):
        self.call("terminal_persistence")
        self.terminal = (tuple(phase for phase, _ in errors), dict(stats))


def test_python_worker_success_returns_exact_statistics_and_events() -> None:
    ops = SimulatorFreeWorkerOps(None)
    result = _execute_managed_worker_lifecycle(ops)
    assert isinstance(result, defaultdict)
    assert dict(result) == {"episode_2": True, "episode_3": False}
    assert ops.persisted == dict(result)
    assert ops.events == [
        ("episode_start", "episode_2"),
        ("episode_end", ("episode_2", True)),
        ("episode_start", "episode_3"),
        ("episode_end", ("episode_3", False)),
        ("worker_status", ("closing", dict(result))),
        ("worker_done", dict(result)),
    ]
    assert ops.terminal == ((), dict(result))
    assert ops.calls.count("video_release") == 2
    assert ops.calls[-3:] == [
        "agent_close", "environment_close", "terminal_persistence",
    ]


@pytest.mark.parametrize("failure", FAILURES)
def test_python_worker_failure_attempts_every_owned_cleanup(failure: str) -> None:
    ops = SimulatorFreeWorkerOps(failure)
    with pytest.raises(Exception, match="EVALUATOR_WORKER_FAILED"):
        _execute_managed_worker_lifecycle(ops)
    actual_cleanup = [
        name for name in ops.calls
        if name in {
            "video_release", "agent_close", "environment_close",
            "terminal_persistence",
        }
    ]
    expected_tail = []
    if failure not in {"gym_make", "agent_make"}:
        expected_tail.append("agent_close")
    if failure != "gym_make":
        expected_tail.append("environment_close")
    expected_tail.append("terminal_persistence")
    assert actual_cleanup[-len(expected_tail):] == expected_tail
    if failure in {"gym_make", "agent_make", "recorder_make"}:
        expected_releases = 0
    elif failure in {
        "reset", "stabilization", "policy_request", "render",
        "video_write", "video_release",
    }:
        expected_releases = 1
    else:
        expected_releases = 2
    assert actual_cleanup.count("video_release") == expected_releases
```

- [ ] **Step 3: Run RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_runtime_contract.py \
  tests/eval_runtime/test_runtime_events.py \
  tests/eval_runtime/test_evaluator_fd_plumbing.py \
  tests/eval_runtime/test_evaluator_python_lifecycle.py
```

Expected: managed options, event classes, and the extracted Python lifecycle are absent.

- [ ] **Step 4: Add frozen config fields and validate them before any worker spawn**

Add these exact optional fields to `simple.evals.api.EvalConfig` and the Typer CLI:

```python
video_output_dir: str | None = None
runtime_evidence_path: str | None = None
runtime_evidence_run_id: str | None = None
runtime_evidence_nonce: str | None = None
runtime_event_fd: int | None = None
runtime_ack_fd: int | None = None
runtime_root_fd: int | None = None
runtime_base_python_root_fd: int | None = None
runtime_identity_fd: int | None = None
policy_transport_fd: int | None = None
```

Before creating logs, processes, agents, or environments, require either all nine managed values are `None`, or all are present, `num_workers == 1`, policy is `psi0_decoupled_wbc`, run/nonce/path are valid, FDs are distinct and have the correct direction/type, roots/identity match the sealed sidecar, and `host`/`port` were not explicitly supplied for managed mode. Parent `worker_kwargs` carry every value exactly. Unmanaged behavior remains unchanged.

- [ ] **Step 5: Implement exact event and acknowledgement schemas**

`events.py` declares `RUNTIME_RECORD_KEYS`, `EVENT_KEYS`, `ACK_KEYS`, and exact per-event payload key/type maps from the design. Use `type(value) is expected`, reject Boolean-as-integer, require `PIPE_BUF`-bounded single-write canonical lines, sequence starting at 1, and append receive timestamps only in the manager-owned copy. `EventReader.feed_bytes(data, *, eof)` owns incremental framing and rejects wrong PID/run/episode/identity, malformed/partial/oversize bytes, gaps, duplicate/out-of-order sequence, silence, and early EOF. `observe_tui_callback` records diagnostics in a separate counter and can never advance the safety-channel sequence or satisfy `require_contract`.

The runtime-contract record contains exactly the 35 keys in the approved schema: source/tree/gitlinks, closure/base/loader/root/runtime/sandbox/relay/sidecar identities, UID/GID, private network inventory, policy socket/relay identity, actual config, status/error, and creation identities. No layer calls Git. Values derive only from inherited descriptors plus supervisor launch record.

- [ ] **Step 6: Move `creating_env` behind durable evidence and acknowledgement**

Refactor `_run_eval_worker` in this exact order:

```python
# ruff: noqa: F821
managed = _validate_managed_runtime_options(
    runtime_evidence_path=runtime_evidence_path,
    runtime_evidence_run_id=runtime_evidence_run_id,
    runtime_evidence_nonce=runtime_evidence_nonce,
    runtime_event_fd=runtime_event_fd,
    runtime_ack_fd=runtime_ack_fd,
    runtime_root_fd=runtime_root_fd,
    runtime_base_python_root_fd=runtime_base_python_root_fd,
    runtime_identity_fd=runtime_identity_fd,
    policy_transport_fd=policy_transport_fd,
    num_workers=num_workers,
    policy=policy,
)
if managed is not None:
    runtime_record = managed.build_record(sonic_config)
    file_identity = managed.write_record_exclusive(runtime_record)
    report("runtime_contract", **managed.contract_payload(runtime_record, file_identity))
    managed.read_and_validate_ack(timeout=5.0, runtime_record=runtime_record, file_identity=file_identity)
report("worker_init", total_episodes=len(episode_indices), status="creating_env")
worker_ops = _ManagedWorkerOps.from_worker_arguments(
    env_id=env_id,
    make_kwargs=make_kwargs,
    managed=managed,
    # Pass the existing agent, recorder, policy, render, result, and event
    # dependencies here; construction remains deferred until make_env().
)
result = _execute_managed_worker_lifecycle(worker_ops)
```

Extract the Python-owned resource lifecycle and make `_run_eval_worker` execute its existing environment/agent/episode body through this production seam:

```python
# src/simple/cli/eval_decoupled_wbc.py
# ruff: noqa: F821
def _execute_managed_worker_lifecycle(ops):
    env = None
    agent = None
    stats = ops.new_stats()
    errors: list[tuple[str, BaseException]] = []
    try:
        env = ops.make_env()
        agent = ops.make_agent(env)
        for episode in ops.episodes():
            recorder = None
            episode_errors: list[tuple[str, BaseException]] = []
            ops.report_episode_start(episode)
            try:
                recorder = ops.make_recorder(env, episode)
                observation = ops.reset(recorder)
                ops.stabilize(env, observation)
                ops.run_policy_episode(agent, recorder, observation, episode)
                success = ops.episode_success(env, episode)
                stats[episode] = success
                ops.report_episode_end(episode, success)
            except BaseException as error:
                episode_errors.append(("episode_body", error))
            finally:
                if recorder is not None:
                    try:
                        ops.release_recorder(recorder)
                    except BaseException as error:
                        episode_errors.append(("video_release", error))
            if episode_errors:
                errors.extend(episode_errors)
                break
        if not errors:
            ops.persist_result(stats)
            ops.report_worker_closing(stats)
    except BaseException as error:
        errors.append(("body", error))
    finally:
        cleanup = (
            ("agent_close", None if agent is None else lambda: ops.close_agent(agent)),
            ("environment_close", None if env is None else lambda: ops.close_env(env)),
        )
        for phase, action in cleanup:
            if action is None:
                continue
            try:
                action()
            except BaseException as error:
                errors.append((phase, error))
        if not errors:
            try:
                ops.report_worker_done(stats)
            except BaseException as error:
                errors.append(("worker_done_event", error))
        try:
            ops.persist_terminal(tuple(errors), stats)
        except BaseException as error:
            errors.append(("terminal_persistence", error))
    if errors:
        summary = ",".join(phase for phase, _ in errors)
        raise RuntimeError(f"EVALUATOR_WORKER_FAILED:{summary}") from errors[0][1]
    return stats
```

The production `_ManagedWorkerOps.from_worker_arguments` constructor stores arguments only and acquires no resource. Its `make_env()` is the sole call site for the real `gym.make`; `episodes()` yields the existing assigned episode objects in their existing order; `new_stats()` returns `defaultdict(bool)` exactly as the current worker does. Move the current reset, per-episode agent reset, stabilization loop, gait setup, `while not episode_over` policy loop, step events, `_success` read, stats-line append, and episode-end metrics byte-for-byte into the corresponding adapter methods. `run_policy_episode` must retain the complete loop rather than executing one synthetic step. `persist_result(stats)` writes the existing `dict(stats)` payload; `_execute_managed_worker_lifecycle` returns the original `defaultdict` object that `run_eval` consumes, and `_run_eval_worker` immediately returns the `result` shown above without converting it to an observation or another type. Per-episode `VideoRecorder` creation/release remains inside the loop, while the outer lifecycle owns agent close, environment close, worker events, terminal persistence, and the exact statistics return. Thus the tested lifecycle, rather than a second direct construction path, owns every Python resource and the complete existing episode behavior. No simulator-free branch exists outside dependency injection. Each failure is accumulated, every later cleanup is still attempted, and the first error remains the exception cause. Worker entry catches the combined error only after terminal persistence and closes the event pipe. The Rust runner remains responsible only for evaluator/relay process, cgroup, namespace, FD, and socket cleanup.

If WBC config, network namespace, root/identity FD, policy socket/relay, record creation, event write, or acknowledgement validation fails, write a rejected record when possible, emit `runtime_contract(rejected)`, raise into the existing worker-error path, emit no `creating_env`, and call `gym.make` zero times. Every `report()` first performs the safety pipe write and only then invokes the TUI callback.

Wrap worker result persistence, video finalization, agent/client close, and `raw_env.close()` in nested `try/finally`; each cleanup error is recorded and later cleanup still runs.

- [ ] **Step 7: Run GREEN and commit**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_runtime_contract.py \
  tests/eval_runtime/test_runtime_events.py \
  tests/eval_runtime/test_evaluator_fd_plumbing.py \
  tests/eval_runtime/test_evaluator_python_lifecycle.py \
  tests/test_official_eval_compatibility.py
```

Expected: all tests pass; rejected configurations have zero environment construction; unmanaged compatibility stays green.

```bash
git add src/simple/eval_runtime/events.py src/simple/evals/api.py \
  src/simple/cli/eval_decoupled_wbc.py tests/eval_runtime \
  tests/test_official_eval_compatibility.py
git commit -m "feat: attest managed PSI0 evaluator startup"
```

### Task 10: Harden run-scoped standard video creation and finalization

**Files:**
- Modify: `src/simple/envs/video_writer.py`
- Modify: `src/simple/envs/wrappers/video_recorder.py`
- Modify: `src/simple/cli/eval_decoupled_wbc.py`
- Modify: `src/simple/evals/api.py`
- Create: `tests/eval_runtime/test_video_writer.py`
- Create: `tests/eval_runtime/test_video_recorder.py`
- Create: `tests/eval_runtime/test_video_cli_routing.py`
- Modify: `tests/eval_runtime/static_files.txt`

- [ ] **Step 1: Write failing exclusive-creation and raw-retention tests**

```python
# tests/eval_runtime/test_video_writer.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simple.envs.video_writer import VideoFinalizationError, VideoWriter
from .fakes import FakeVideoOps


@pytest.mark.parametrize(
    "existing_name",
    [
        "head.raw.mp4",
        "head.transcoding.mp4",
        "head_success.mp4",
        "head_failed.mp4",
    ],
)
def test_writer_refuses_any_existing_raw_tmp_or_verdict_path(
    tmp_path: Path, existing_name: str
) -> None:
    raw = tmp_path / "head.raw.mp4"
    existing = tmp_path / existing_name
    existing.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        VideoWriter(raw, 10, (640, 360))
    assert existing.read_bytes() == b"old"


@pytest.mark.parametrize("failure", ["missing", "timeout", "nonzero", "malformed", "probe"])
def test_finalization_failure_preserves_nonempty_raw_and_diagnostic_tmp(
    tmp_path: Path, failure: str
) -> None:
    ops = FakeVideoOps(failure=failure)
    writer = VideoWriter(tmp_path / "head.raw.mp4", 10, (640, 360), ops=ops)
    writer.write(np.zeros((360, 640, 3), np.uint8))
    with pytest.raises(VideoFinalizationError):
        writer.release(success=True, deadline=ops.clock() + 120.0)
    assert writer.filename.is_file()
    assert writer.filename.stat().st_size > 0
    assert not (tmp_path / "head_success.mp4").exists()
    assert ops.cleanup_complete is True


def test_success_keeps_raw_and_atomically_publishes_verified_verdict(tmp_path: Path) -> None:
    ops = FakeVideoOps()
    writer = VideoWriter(tmp_path / "head.raw.mp4", 10, (640, 360), ops=ops)
    writer.write(np.zeros((360, 640, 3), np.uint8))
    result = writer.release(success=False, deadline=ops.clock() + 120.0)
    assert writer.filename.is_file()
    assert result.final_path == tmp_path / "head_failed.mp4"
    assert result.final_path.is_file()
    assert result.probe.frame_count > 0
    assert result.probe.duration > 0
```

`FakeVideoOps` is defined in the test file. It emulates OpenCV release, FFmpeg argv/exit/deadline, FFprobe JSON, atomic rename, and INT/TERM/KILL reconciliation; no real codec process starts in unit tests.

- [ ] **Step 2: Write failing deferred-recorder and parent/worker routing tests**

```python
# tests/eval_runtime/test_video_recorder.py
from __future__ import annotations

import pytest

from simple.envs.wrappers.video_recorder import VideoRecorder


def test_reset_does_not_create_writer_and_start_is_exactly_once(fake_image_env, tmp_path) -> None:
    recorder = VideoRecorder(fake_image_env, video_folder=tmp_path, name_prefix="episode_2")
    observation, _ = recorder.reset()
    assert list(tmp_path.rglob("*.mp4")) == []
    recorder.start(observation)
    assert sorted(path.name for path in tmp_path.rglob("*.raw.mp4")) == [
        "head_stereo_left.raw.mp4", "head_stereo_right.raw.mp4"
    ]
    with pytest.raises(RuntimeError, match="already started"):
        recorder.start(observation)


def test_step_before_start_and_reset_while_active_fail(fake_image_env, tmp_path) -> None:
    recorder = VideoRecorder(fake_image_env, video_folder=tmp_path, name_prefix="episode_2")
    observation, _ = recorder.reset()
    with pytest.raises(RuntimeError, match="not started"):
        recorder.step(fake_image_env.action_space.sample())
    recorder.start(observation)
    with pytest.raises(RuntimeError, match="active recorder"):
        recorder.reset()


def test_release_attempts_all_cameras_and_returns_structured_failures(fake_image_env, tmp_path) -> None:
    recorder = VideoRecorder(fake_image_env, video_folder=tmp_path, name_prefix="episode_2")
    observation, _ = recorder.reset()
    recorder.start(observation)
    recorder.video_writers["head_stereo_left"].ops.failure = "timeout"
    results = recorder.release(deadline=recorder.clock() + 120.0)
    assert set(results) == {"head_stereo_left", "head_stereo_right"}
    assert results["head_stereo_left"].ok is False
    assert results["head_stereo_right"].ok is True
```

```python
# tests/eval_runtime/test_video_cli_routing.py
from __future__ import annotations

from simple.cli.eval_decoupled_wbc import _make_worker_kwargs
from simple.evals.api import EvalConfig


def test_video_output_dir_routes_parent_to_worker_exactly() -> None:
    config = EvalConfig(
        env_id="simple/Fake-v0", policy="psi0_decoupled_wbc",
        video_output_dir="/run-output/videos",
    )
    assert _make_worker_kwargs(config, sonic_config={})["video_output_dir"] == "/run-output/videos"


def test_unmanaged_omission_retains_legacy_root_selection() -> None:
    config = EvalConfig(env_id="simple/Fake-v0", policy="psi0_decoupled_wbc")
    assert _make_worker_kwargs(config, sonic_config={})["video_output_dir"] is None
```

- [ ] **Step 3: Run RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_video_writer.py \
  tests/eval_runtime/test_video_recorder.py \
  tests/eval_runtime/test_video_cli_routing.py
```

Expected: old writer deletes existing files, recorder opens on reset, and routing helper is absent.

- [ ] **Step 4: Implement exclusive raw creation and bounded checked finalization**

`VideoWriter` receives a path ending `.raw.mp4`, refuses existing raw/temp/verdict paths, opens OpenCV only after exclusive sentinel creation, and never deletes raw. `release(success, deadline)` releases OpenCV, checks raw nonempty, starts:

```python
# ruff: noqa: F821
[
    "ffmpeg", "-nostdin", "-n", "-i", str(raw_path),
    "-vcodec", "libx264", str(tmp_path),
]
```

with a shared monotonic deadline capped at 120 seconds. Nonzero/timeout follows identity-checked INT→TERM→KILL and fresh post-KILL wait. On zero exit, invoke FFprobe by argv, require codec, exact dimensions, positive frame count/duration, then atomically rename absent temp to verdict and fsync the directory. Failure preserves raw and moves any nonempty temporary to a unique `.diagnostic.<code>.mp4` name without replacing an existing file.

- [ ] **Step 5: Implement deferred recorder and worker `finally` lifecycle**

`VideoRecorder.reset` delegates reset but creates no directory/writer. `start(observation)` exclusively creates `<video_output_dir>/episode_<N>/` and each `<camera>.raw.mp4`, writes frame zero, and is callable once. `step` requires active state. `release` closes every camera even after failure and returns `dict[str, VideoFinalizeResult]`. `close` always calls release then `super().close()` and combines failures without skipping either.

In `_run_eval_worker`, create recorder before reset, stabilize through `sonic_env` without recorder writes, then call `env.start(observation)` exactly once. Pass exactly `/run-output/videos` as `video_output_dir`; `VideoRecorder` alone appends `episode_<N>` exactly once. Wrap episode evaluation, result persistence, video release, agent close, and environment close in `try/finally`; render/write/persistence/close failures remain infrastructure failures with raw artifacts retained. The FFmpeg path uses `-n`, opens neither the temporary nor either verdict path when any exists, and tests preserve every preexisting byte across constructor and finalization failures.

- [ ] **Step 6: Run GREEN and commit**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_video_writer.py \
  tests/eval_runtime/test_video_recorder.py \
  tests/eval_runtime/test_video_cli_routing.py \
  tests/test_official_eval_compatibility.py
```

Expected: all tests pass; no real FFmpeg or simulator starts.

```bash
git add src/simple/envs src/simple/cli/eval_decoupled_wbc.py \
  src/simple/evals/api.py tests/eval_runtime tests/test_official_eval_compatibility.py
git commit -m "fix: preserve run-scoped SIMPLE evaluation videos"
```

### Task 11: Implement immutable evidence, lifecycle state machine, CLI, and reverse-order cleanup

**Files:**
- Create: `src/simple/eval_runtime/evidence.py`
- Create: `src/simple/eval_runtime/manager.py`
- Create: `src/simple/eval_runtime/cli.py`
- Create: `scripts/psi0_eval_runtime.py`
- Modify: `pyproject.toml`
- Create: `tests/eval_runtime/test_evidence.py`
- Create: `tests/eval_runtime/test_manager.py`
- Create: `tests/eval_runtime/test_cli.py`
- Create: `tests/eval_runtime/test_cleanup.py`
- Modify: `tests/eval_runtime/static_files.txt`

- [ ] **Step 1: Write failing immutable evidence and verdict tests**

```python
# tests/eval_runtime/test_evidence.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from simple.eval_runtime.evidence import (
    EvidenceStore,
    Gate9HandoffReconciler,
    InfrastructureVerdict,
    create_run_id_handoff,
    read_run_id_handoff,
    remove_run_id_handoff,
)
from .fakes import finalize_failed_evidence, populate_complete_evidence


def test_run_root_and_every_manager_artifact_are_exclusive(tmp_path: Path) -> None:
    store = EvidenceStore.create(tmp_path, run_id="run-a")
    store.write("preflight/profile-hash.json", {"schema_version": 1, "sha256": "a" * 64})
    with pytest.raises(FileExistsError):
        EvidenceStore.create(tmp_path, run_id="run-a")
    with pytest.raises(FileExistsError):
        store.write("preflight/profile-hash.json", {"schema_version": 1, "sha256": "b" * 64})


def test_existing_run_child_is_never_reused_or_modified(tmp_path: Path) -> None:
    child = tmp_path / "run-collision"
    child.mkdir(mode=0o700)
    sentinel = child / "foreign.txt"
    sentinel.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        EvidenceStore.create(tmp_path, run_id="run-collision")
    assert sentinel.read_bytes() == b"keep"
    assert sorted(path.name for path in child.iterdir()) == ["foreign.txt"]


def test_pass_requires_all_mandatory_files_and_clean_terminal_state(tmp_path: Path) -> None:
    store = EvidenceStore.create(tmp_path, run_id="run-a")
    populate_complete_evidence(store, episodes=(2, 3))
    verdict = InfrastructureVerdict.evaluate(store)
    assert verdict.status == "PASS"
    (store.root / "cleanup/ports-final.json").unlink()
    assert InfrastructureVerdict.evaluate(store).status == "FAIL"


def test_manifest_redacts_raw_tokens_and_payloads(tmp_path: Path) -> None:
    store = EvidenceStore.create(tmp_path, run_id="run-a")
    store.record_command(
        argv=["ssh", "h100", "helper", "--token", "secret"],
        secret_positions={4},
        result={"request_payload": "large", "lease_token_sha256": "a" * 64},
    )
    encoded = (store.root / "commands.jsonl").read_text()
    assert "secret" not in encoded
    assert '"request_payload"' not in encoded
    assert "lease_token_sha256" in encoded


def test_run_id_handoff_is_exclusive_mode_0600_and_validated(tmp_path: Path) -> None:
    handoff = tmp_path / "evaluation-run-id"
    identity = create_run_id_handoff(handoff, "eval-a")
    assert identity.mode == 0o600
    assert read_run_id_handoff(handoff) == "eval-a"
    with pytest.raises(FileExistsError):
        create_run_id_handoff(handoff, "eval-b")
    handoff.chmod(0o644)
    with pytest.raises(Exception, match="RUN_ID_HANDOFF_MODE"):
        read_run_id_handoff(handoff)
    handoff.chmod(0o600)
    remove_run_id_handoff(handoff, expected_run_id="eval-a")
    assert not handoff.exists()


def test_interrupted_gate9_run_is_terminalized_preserved_and_retired(
    tmp_path: Path,
) -> None:
    output_parent = tmp_path / "workloads"
    output_parent.mkdir(mode=0o700)
    store = EvidenceStore.create(output_parent, run_id="eval-old")
    store.write("partial-evidence.json", {"schema_version": 1, "keep": True})
    handoff = tmp_path / "evaluation-run-id"
    create_run_id_handoff(handoff, "eval-old")

    def recover_owned(run_id: str, run_root: Path) -> None:
        assert run_id == "eval-old"
        finalize_failed_evidence(
            EvidenceStore.open_existing(run_root),
            run_id=run_id,
        )

    reconciler = Gate9HandoffReconciler(
        output_parent=output_parent,
        recover_owned_run=recover_owned,
    )
    receipt = reconciler.reconcile_and_retire(
        handoff, reason="interrupted-retry"
    )
    assert receipt.run_id == "eval-old"
    assert receipt.verdict == "FAIL"
    assert not handoff.exists()
    assert (store.root / "partial-evidence.json").read_text()
    assert (store.root / "operator-handoff-retired.json").is_file()
    verified = store.verify_terminal_manifest(expected_run_id="eval-old")
    assert "partial-evidence.json" in verified.artifact_paths
    create_run_id_handoff(handoff, "eval-new")
    assert read_run_id_handoff(handoff) == "eval-new"


def test_gate9_interrupt_before_run_root_creates_failed_evidence_root(
    tmp_path: Path,
) -> None:
    output_parent = tmp_path / "workloads"
    output_parent.mkdir(mode=0o700)
    handoff = tmp_path / "evaluation-run-id"
    create_run_id_handoff(handoff, "eval-prestart")

    def recover_owned(run_id: str, run_root: Path) -> None:
        assert run_id == "eval-prestart"
        assert run_root.is_dir()
        finalize_failed_evidence(
            EvidenceStore.open_existing(run_root),
            run_id=run_id,
        )

    receipt = Gate9HandoffReconciler(
        output_parent, recover_owned_run=recover_owned
    ).reconcile_and_retire(handoff, reason="interrupted-retry")
    root = output_parent / "eval-prestart"
    assert receipt.run_id == "eval-prestart"
    assert (root / "prestart-interruption.json").is_file()
    assert (root / "operator-handoff-retired.json").is_file()
    assert not handoff.exists()


@pytest.mark.parametrize(
    "boundary", ["after_terminalize", "after_receipt", "before_handoff_unlink"]
)
def test_gate9_retirement_is_restartable_at_every_write_boundary(
    tmp_path: Path, boundary: str
) -> None:
    output_parent = tmp_path / "workloads"
    output_parent.mkdir(mode=0o700)
    store = EvidenceStore.create(output_parent, run_id="eval-old")
    handoff = tmp_path / "evaluation-run-id"
    create_run_id_handoff(handoff, "eval-old")

    def recover_owned(run_id: str, run_root: Path) -> None:
        finalize_failed_evidence(
            EvidenceStore.open_existing(run_root),
            run_id=run_id,
        )

    reconciler = Gate9HandoffReconciler(output_parent, recover_owned_run=recover_owned)
    with pytest.raises(OSError, match=boundary):
        reconciler.reconcile_and_retire(
            handoff,
            reason="interrupted-retry",
            fault_hook=lambda name: (_ for _ in ()).throw(OSError(name))
            if name == boundary
            else None,
        )
    assert read_run_id_handoff(handoff) == "eval-old"
    receipt = reconciler.reconcile_and_retire(
        handoff, reason="interrupted-retry"
    )
    assert receipt.run_id == "eval-old"
    assert not handoff.exists()
    assert (store.root / "operator-handoff-retired.json").is_file()


def test_gate9_rejects_tampered_artifact_from_full_manifest(tmp_path: Path) -> None:
    output_parent = tmp_path / "workloads"
    output_parent.mkdir(mode=0o700)
    store = EvidenceStore.create(output_parent, run_id="eval-old")
    store.write("partial-evidence.json", {"schema_version": 1, "keep": True})
    handoff = tmp_path / "evaluation-run-id"
    original = create_run_id_handoff(handoff, "eval-old")

    def recover_then_tamper(run_id: str, run_root: Path) -> None:
        finalize_failed_evidence(
            EvidenceStore.open_existing(run_root),
            run_id=run_id,
        )
        (run_root / "partial-evidence.json").write_bytes(b"tampered\n")

    reconciler = Gate9HandoffReconciler(
        output_parent,
        recover_owned_run=recover_then_tamper,
    )
    with pytest.raises(Exception, match="MANIFEST_ARTIFACT_HASH"):
        reconciler.reconcile_and_retire(handoff, reason="interrupted-retry")
    current = handoff.stat()
    assert (current.st_dev, current.st_ino) == (original.device, original.inode)
    assert not (store.root / "operator-handoff-retired.json").exists()


def test_full_manifest_rejects_tampered_command_record(tmp_path: Path) -> None:
    store = EvidenceStore.create(tmp_path, run_id="eval-old")
    store.record_command(
        argv=["status"],
        secret_positions=set(),
        result={"returncode": 0},
    )
    finalize_failed_evidence(store, run_id="eval-old")
    with (store.root / "commands.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"sequence":999}\n')
    with pytest.raises(Exception, match="MANIFEST_ARTIFACT_HASH|MANIFEST_COMMAND_INDEX"):
        store.verify_terminal_manifest(expected_run_id="eval-old")


def test_gate9_rejects_old_seven_key_toy_manifest(tmp_path: Path) -> None:
    output_parent = tmp_path / "workloads"
    output_parent.mkdir(mode=0o700)
    store = EvidenceStore.create(output_parent, run_id="eval-old")
    toy = {
        "schema_version": 1,
        "run_id": "eval-old",
        "lifecycle": "TERMINAL",
        "infrastructure_verdict": "FAIL",
        "overall_verdict": "FAIL",
        "owned_processes_alive": [],
        "unknown_remote_state": False,
    }
    (store.root / "manifest.json").write_text(
        json.dumps(toy, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    handoff = tmp_path / "evaluation-run-id"
    create_run_id_handoff(handoff, "eval-old")
    reconciler = Gate9HandoffReconciler(
        output_parent,
        recover_owned_run=lambda run_id, run_root: None,
    )
    with pytest.raises(Exception, match="MANIFEST_SCHEMA"):
        reconciler.reconcile_and_retire(handoff, reason="interrupted-retry")
    assert read_run_id_handoff(handoff) == "eval-old"
    assert not (store.root / "operator-handoff-retired.json").exists()


@pytest.mark.parametrize("failure", ["foreign", "unknown", "nonterminal", "pass"])
def test_gate9_reconciliation_failure_retains_exact_handoff(
    tmp_path: Path, failure: str
) -> None:
    output_parent = tmp_path / "workloads"
    output_parent.mkdir(mode=0o700)
    store = EvidenceStore.create(output_parent, run_id="eval-old")
    handoff = tmp_path / "evaluation-run-id"
    original = create_run_id_handoff(handoff, "eval-old")

    def reject(run_id: str, run_root: Path) -> None:
        current_store = EvidenceStore.open_existing(run_root)
        if failure == "foreign":
            raise RuntimeError("FOREIGN")
        if failure == "nonterminal":
            current_store.write(
                "cleanup/actions.json",
                {"schema_version": 1, "status": "running"},
            )
            return
        if failure == "unknown":
            finalize_failed_evidence(
                current_store,
                run_id=run_id,
                unknown_remote_state=True,
            )
            return
        populate_complete_evidence(current_store, episodes=(2, 3))
        current_store.finalize_terminal_manifest(
            run_id=run_id,
            episodes=(2, 3),
            infrastructure_verdict="PASS",
            overall_verdict="PASS",
            task_results={"episode_2": True, "episode_3": True},
            owned_processes_alive=[],
            unknown_remote_state=False,
            unresolved_errors=[],
        )

    reconciler = Gate9HandoffReconciler(output_parent, recover_owned_run=reject)
    with pytest.raises(Exception):
        reconciler.reconcile_and_retire(handoff, reason="interrupted-retry")
    current = handoff.stat()
    assert (current.st_dev, current.st_ino) == (original.device, original.inode)
    assert read_run_id_handoff(handoff) == "eval-old"
    assert not (store.root / "operator-handoff-retired.json").exists()
```

- [ ] **Step 2: Write failing persist-before-action and cleanup tests**

```python
# tests/eval_runtime/test_manager.py
from __future__ import annotations

from simple.eval_runtime.manager import (
    LifecycleState,
    PersistedStateMachine,
    next_episodes,
    reduce_verdict,
)


def test_every_external_action_is_preceded_by_fsynced_state() -> None:
    timeline: list[str] = []
    machine = PersistedStateMachine(
        initial=LifecycleState.NEW,
        persist=lambda state: timeline.append(f"fsync_state:{state.value}"),
    )
    for state in (
        LifecycleState.LOCAL_ATTESTED, LifecycleState.LEASED,
        LifecycleState.PREFLIGHTED, LifecycleState.CONTAINER_STARTED,
        LifecycleState.SERVER_STARTED, LifecycleState.TUNNEL_READY,
        LifecycleState.SERVER_READY, LifecycleState.EVALUATING,
        LifecycleState.CLEANING, LifecycleState.CLEAN,
    ):
        machine.before_action(state, lambda state=state: timeline.append(f"external:{state.value}"))
    for index, event in enumerate(timeline):
        if event.startswith("external:"):
            assert timeline[index - 1].startswith("fsync_state:")
    assert machine.history == [
        LifecycleState.NEW,
        LifecycleState.LOCAL_ATTESTED,
        LifecycleState.LEASED,
        LifecycleState.PREFLIGHTED,
        LifecycleState.CONTAINER_STARTED,
        LifecycleState.SERVER_STARTED,
        LifecycleState.TUNNEL_READY,
        LifecycleState.SERVER_READY,
        LifecycleState.EVALUATING,
        LifecycleState.CLEANING,
        LifecycleState.CLEAN,
    ]


def test_episode_three_never_starts_after_episode_two_infrastructure_failure() -> None:
    assert next_episodes((2, 3), completed=2, infrastructure_clean=False) == ()
    assert next_episodes((2, 3), completed=2, infrastructure_clean=True) == (3,)


def test_task_failure_is_distinct_from_infrastructure_failure() -> None:
    summary = reduce_verdict(task_results={2: False, 3: True}, infrastructure_valid=True)
    assert summary.infrastructure_valid is True
    assert summary.task_results == {2: False, 3: True}
    assert summary.verdict == "FAIL"
```

```python
# tests/eval_runtime/test_cleanup.py
from __future__ import annotations

from simple.eval_runtime.manager import CleanupStack
from .fakes import CleanupHarness


def test_cleanup_is_reverse_order_and_one_failure_does_not_skip_later_actions() -> None:
    order: list[str] = []
    stack = CleanupStack()
    stack.push("container", lambda deadline: order.append("container"))
    stack.push("server", lambda deadline: (_ for _ in ()).throw(RuntimeError("server")))
    stack.push("tunnel", lambda deadline: order.append("tunnel"))
    errors = stack.close(shared_deadline=100.0)
    assert order == ["tunnel", "container"]
    assert [error.resource for error in errors] == ["server"]


def test_terminal_restoration_logs_and_manifest_run_even_after_deadline() -> None:
    harness = CleanupHarness(expired=True)
    harness.close()
    assert harness.calls[-3:] == ["restore_terminal", "close_logs", "finalize_manifest"]
```

`CleanupHarness` is the deterministic state-only fake from `tests/eval_runtime/fakes.py`; process and manager boundaries remain covered by direct dependency-injected tests and the integration drivers.

- [ ] **Step 3: Write failing CLI operation tests**

```python
# tests/eval_runtime/test_cli.py
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from simple.eval_runtime.cli import app
from simple.eval_runtime.evidence import (
    EvidenceStore,
    create_run_id_handoff,
    read_run_id_handoff,
)
from .fakes import FakeCliBackend

runner = CliRunner()


def test_cli_exposes_only_reviewed_operations() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("freeze-provenance", "create", "status", "evaluate", "stop"):
        assert command in result.stdout
    assert "force" not in result.stdout.lower()


def test_evaluate_restricts_initial_workload_to_episodes_two_and_three() -> None:
    result = runner.invoke(app, ["evaluate", "--episodes", "2,4", "--output-root", "/tmp/new"])
    assert result.exit_code != 0
    assert "episodes must be exactly 2,3" in result.stdout


def test_status_is_read_only() -> None:
    backend = FakeCliBackend()
    result = runner.invoke(app, ["status"], obj=backend)
    assert result.exit_code == 0
    assert backend.mutations == []


def test_create_preflight_only_releases_lease_without_container(tmp_path: Path) -> None:
    backend = FakeCliBackend()
    output_parent = tmp_path / "workloads"
    output_parent.mkdir(mode=0o700)
    result = runner.invoke(
        app,
        [
            "create", "--preflight-only", "--run-id", "gate4-a",
            "--output-root", str(output_parent),
        ],
        obj=backend,
    )
    assert result.exit_code == 0
    assert backend.actions == [
        "local_preflight", "lease", "leased_preflight", "release_lease"
    ]
    assert "docker_create" not in backend.actions


def test_stale_recovery_requires_explicit_run_id_and_has_no_force_flag() -> None:
    missing = runner.invoke(app, ["stop", "--recover-stale"])
    assert missing.exit_code != 0
    unsafe = runner.invoke(app, ["stop", "--recover-stale", "../run"])
    assert unsafe.exit_code != 0


def _retirement_groups(
    output_parent: Path,
    handoff: Path,
) -> dict[str, list[str]]:
    return {
        "recover": ["--recover-stale", "eval-old"],
        "output": ["--output-root", str(output_parent)],
        "handoff": ["--run-id-handoff", str(handoff)],
        "enable": ["--retire-terminal-handoff"],
        "reason": ["--retire-reason", "interrupted-retry"],
    }


def _flatten(groups: dict[str, list[str]]) -> list[str]:
    return [word for group in groups.values() for word in group]


def _make_retirement_fixture(tmp_path: Path):
    output_parent = tmp_path / "workloads"
    output_parent.mkdir(mode=0o700)
    store = EvidenceStore.create(output_parent, run_id="eval-old")
    store.write("partial-evidence.json", {"schema_version": 1})
    handoff = tmp_path / "evaluation-run-id"
    create_run_id_handoff(handoff, "eval-old")
    return output_parent, handoff


def test_complete_retirement_options_use_canonical_reconciler(tmp_path: Path) -> None:
    output_parent, handoff = _make_retirement_fixture(tmp_path)
    backend = FakeCliBackend()
    result = runner.invoke(
        app,
        ["stop", *_flatten(_retirement_groups(output_parent, handoff))],
        obj=backend,
    )
    assert result.exit_code == 0, result.stdout
    assert backend.retirement_calls == [
        ("eval-old", output_parent / "eval-old")
    ]
    assert not handoff.exists()
    assert (
        output_parent / "eval-old" / "operator-handoff-retired.json"
    ).is_file()


@pytest.mark.parametrize(
    "missing",
    ["recover", "output", "handoff", "enable", "reason"],
)
def test_every_partial_retirement_option_set_fails_before_recovery(
    tmp_path: Path,
    missing: str,
) -> None:
    output_parent, handoff = _make_retirement_fixture(tmp_path)
    groups = _retirement_groups(output_parent, handoff)
    groups.pop(missing)
    backend = FakeCliBackend()
    result = runner.invoke(app, ["stop", *_flatten(groups)], obj=backend)
    assert result.exit_code != 0
    assert backend.retirement_calls == []
    assert read_run_id_handoff(handoff) == "eval-old"


@pytest.mark.parametrize("state", ["pass", "foreign", "unknown", "nonterminal"])
def test_retirement_rejects_unacceptable_state_without_signal_or_unlink(
    tmp_path: Path,
    state: str,
) -> None:
    output_parent, handoff = _make_retirement_fixture(tmp_path)
    backend = FakeCliBackend()
    backend.retirement_state = state
    result = runner.invoke(
        app,
        ["stop", *_flatten(_retirement_groups(output_parent, handoff))],
        obj=backend,
    )
    assert result.exit_code != 0
    assert backend.signals == []
    assert read_run_id_handoff(handoff) == "eval-old"
    assert not (
        output_parent / "eval-old" / "operator-handoff-retired.json"
    ).exists()
```

- [ ] **Step 4: Run RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_evidence.py \
  tests/eval_runtime/test_manager.py tests/eval_runtime/test_cleanup.py \
  tests/eval_runtime/test_cli.py
```

Expected: evidence, manager, and CLI imports fail.

- [ ] **Step 5: Implement the exact evidence tree and verdict reducer**

`EvidenceStore.create(output_parent, run_id=...)` requires an existing, attested, nonsymlink directory equal to the configured workload-parent root. It opens that parent with `O_DIRECTORY|O_NOFOLLOW`, validates device/inode/owner/mode, and atomically creates only `<output_parent>/<run-id>` with `mkdirat`; the child must be absent. An existing child—empty or nonempty—fails before any read, write, lease, helper, or cleanup action and is never renamed, deleted, adopted, or inspected beyond its no-follow type/identity. The store writes only beneath the newly created child via `atomic_write_new_json` or append-only fsynced JSONL. Define `mandatory_evidence_paths(episodes: tuple[int, ...]) -> tuple[str, ...]` as the complete payload path list in the approved design, including distinct `probe_loader`, `probe_cuda`, `episode_2`, and `episode_3` runner children and all final cleanup records, but excluding the three late derived paths in `LATE_UNINDEXED_PATHS`. `manifest.json` hashes every allowlisted artifact and every canonical command record; `run-manifest.md` is rendered solely from that JSON. Raw token, SSH material, secrets, environment secrets, complete image/action payloads, and mutable intake paths are rejected by the evidence serializer.

Use one full final-manifest schema for normal acceptance, failed-run preservation, status verification, and Gate 9 retirement:

```python
# src/simple/eval_runtime/evidence.py (canonical final-manifest contract)
# ruff: noqa: F821
FINAL_MANIFEST_KEYS = frozenset(
    {
        "schema_version", "run_id", "lifecycle", "episodes",
        "infrastructure_verdict", "overall_verdict", "task_results",
        "profile_sha256", "artifact_index", "command_index",
        "state_transitions", "cleanup_summary", "unresolved_errors",
        "owned_processes_alive", "unknown_remote_state",
    }
)
ARTIFACT_ENTRY_KEYS = frozenset({"path", "sha256", "byte_count", "kind"})
COMMAND_ENTRY_KEYS = frozenset({"sequence", "record_sha256"})
LATE_UNINDEXED_PATHS = frozenset(
    {"manifest.json", "run-manifest.md", "operator-handoff-retired.json"}
)


@dataclass(frozen=True, slots=True)
class VerifiedTerminalManifest:
    run_id: str
    lifecycle: str
    infrastructure_verdict: str
    overall_verdict: str
    owned_processes_alive: tuple[int, ...]
    unknown_remote_state: bool
    artifact_paths: tuple[str, ...]
    manifest_bytes: bytes
    manifest_sha256: str


def _verify_terminal_manifest(
    store: EvidenceStore,
    *,
    expected_run_id: str,
) -> VerifiedTerminalManifest:
    manifest_bytes = store.read_bytes_nofollow("manifest.json")
    manifest = decode_canonical_json(manifest_bytes)
    if type(manifest) is not dict or set(manifest) != FINAL_MANIFEST_KEYS:
        raise RuntimeBlocked("MANIFEST_SCHEMA", expected_run_id)
    if (
        manifest["schema_version"] != 1
        or manifest["run_id"] != expected_run_id
        or manifest["lifecycle"] != "TERMINAL"
    ):
        raise RuntimeBlocked("MANIFEST_IDENTITY", expected_run_id)
    if (
        type(manifest["episodes"]) is not list
        or any(type(value) is not int for value in manifest["episodes"])
        or manifest["infrastructure_verdict"] not in {"PASS", "FAIL"}
        or manifest["overall_verdict"] not in {"PASS", "FAIL"}
        or type(manifest["task_results"]) is not dict
        or type(manifest["profile_sha256"]) is not str
        or type(manifest["state_transitions"]) is not list
        or type(manifest["cleanup_summary"]) is not dict
        or type(manifest["unresolved_errors"]) is not list
        or type(manifest["owned_processes_alive"]) is not list
        or any(type(value) is not int for value in manifest["owned_processes_alive"])
        or type(manifest["unknown_remote_state"]) is not bool
    ):
        raise RuntimeBlocked("MANIFEST_TYPE", expected_run_id)
    entries = manifest["artifact_index"]
    if type(entries) is not list:
        raise RuntimeBlocked("MANIFEST_ARTIFACT_SCHEMA", expected_run_id)
    indexed_paths: list[str] = []
    for entry in entries:
        if type(entry) is not dict or set(entry) != ARTIFACT_ENTRY_KEYS:
            raise RuntimeBlocked("MANIFEST_ARTIFACT_SCHEMA", expected_run_id)
        relative = validate_contained_relative_path(entry["path"])
        payload = store.read_bytes_nofollow(relative)
        if (
            len(payload) != entry["byte_count"]
            or hashlib.sha256(payload).hexdigest() != entry["sha256"]
        ):
            raise RuntimeBlocked("MANIFEST_ARTIFACT_HASH", relative)
        indexed_paths.append(relative)
    if indexed_paths != sorted(set(indexed_paths)):
        raise RuntimeBlocked("MANIFEST_ARTIFACT_ORDER", expected_run_id)
    actual_paths = store.list_regular_payload_paths(
        excluding=LATE_UNINDEXED_PATHS
    )
    if tuple(indexed_paths) != actual_paths:
        raise RuntimeBlocked("MANIFEST_ARTIFACT_COVERAGE", expected_run_id)
    command_records = store.read_canonical_command_records()
    expected_commands = [
        {
            "sequence": index,
            "record_sha256": hashlib.sha256(record).hexdigest(),
        }
        for index, record in enumerate(command_records, start=1)
    ]
    if manifest["command_index"] != expected_commands or any(
        type(item) is not dict or set(item) != COMMAND_ENTRY_KEYS
        for item in manifest["command_index"]
    ):
        raise RuntimeBlocked("MANIFEST_COMMAND_INDEX", expected_run_id)
    required_cleanup = {
        "cleanup/actions.json", "cleanup/processes-final.json",
        "cleanup/ports-final.json", "cleanup/container-final.json",
        "cleanup/remote-helpers-final.json",
    }
    if not required_cleanup.issubset(indexed_paths):
        raise RuntimeBlocked("MANIFEST_CLEANUP_COVERAGE", expected_run_id)
    if manifest["infrastructure_verdict"] == "PASS" and not set(
        mandatory_evidence_paths(tuple(manifest["episodes"]))
    ).issubset(indexed_paths):
        raise RuntimeBlocked("MANIFEST_PASS_COVERAGE", expected_run_id)
    return VerifiedTerminalManifest(
        run_id=manifest["run_id"],
        lifecycle=manifest["lifecycle"],
        infrastructure_verdict=manifest["infrastructure_verdict"],
        overall_verdict=manifest["overall_verdict"],
        owned_processes_alive=tuple(manifest["owned_processes_alive"]),
        unknown_remote_state=manifest["unknown_remote_state"],
        artifact_paths=tuple(indexed_paths),
        manifest_bytes=manifest_bytes,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
```

`EvidenceStore.finalize_terminal_manifest(...)` first freezes the command and state-transition streams, no-follow enumerates every current allowlisted regular payload, computes the sorted `artifact_index` and independent `command_index`, validates exact types, writes `manifest.json` with `O_EXCL` plus file/parent fsync, then immediately calls `_verify_terminal_manifest`. `EvidenceStore.verify_terminal_manifest` is the public bound form of that same verifier; no caller re-parses selected keys. A FAIL manifest may contain the fsynced operation prefix rather than every PASS-only artifact, but it must index every existing artifact and command and contain all five terminal cleanup records. PASS additionally requires every `mandatory_evidence_paths(episodes)` entry. Any changed, absent, extra, unindexed, duplicate, out-of-order, or path-escaping artifact/command makes verification fail. The Markdown rendering and Gate 9 retirement receipt are deliberately late derived files and are the only unindexed exceptions.

The same module imports `hashlib`, `json`, `os`, `stat`, `dataclass`, `Path`, and `Callable`, then implements `create_run_id_handoff`, `read_run_id_handoff`, `remove_run_id_handoff`, and `Gate9HandoffReconciler`. Creation uses `O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW`, mode `0600`, one validated run ID plus newline, file fsync, and parent fsync. Reading uses `O_RDONLY|O_CLOEXEC|O_NOFOLLOW` and requires a regular single-link file owned by the current UID, exact mode `0600`, bounded size, one newline, and a valid identifier. Removal reopens and revalidates the current path, requires the expected run ID plus the originally captured path/FD identity, unlinks by parent FD only after acceptance or authenticated failed-run reconciliation, and fsyncs the parent. The reconciler opens the attested output parent and exact existing run root without following links. If interruption occurred after handoff creation but before the run root, `EvidenceStore.open_or_create_reconciliation` alone may exclusively create that exact child and fsync an immutable `prestart-interruption.json` binding the handoff identity; no ordinary existing run is adopted by that path. It then invokes normal owned stale cleanup and calls the same `verify_terminal_manifest` used by final acceptance. Only a fully indexed canonical terminal infrastructure-FAIL manifest with zero live owned processes and known-clean remote state can authorize the immutable retirement receipt and exact handoff unlink. A toy subset, changed artifact, missing command digest, PASS, foreign/unknown ownership, incomplete cleanup, changed inode, mismatched run ID, or missing/nonterminal manifest retains the handoff and forbids a fresh Gate 9 run.

```python
# src/simple/eval_runtime/evidence.py (run-ID handoff helpers)
# ruff: noqa: F821
@dataclass(frozen=True, slots=True)
class RunIdHandoffIdentity:
    device: int
    inode: int
    mode: int


def _handoff_parent(path: Path) -> tuple[Path, int]:
    parent = path.parent.resolve(strict=True)
    if parent != path.parent.absolute() or path.name in {"", ".", ".."}:
        raise RuntimeBlocked("RUN_ID_HANDOFF_PARENT", str(path))
    parent_fd = os.open(
        parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    return parent, parent_fd


def _validate_handoff_fd(fd: int) -> RunIdHandoffIdentity:
    info = os.fstat(fd)
    mode = stat.S_IMODE(info.st_mode)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeBlocked("RUN_ID_HANDOFF_TYPE", "regular single-link file required")
    if info.st_uid != os.getuid():
        raise RuntimeBlocked("RUN_ID_HANDOFF_OWNER", str(info.st_uid))
    if mode != 0o600:
        raise RuntimeBlocked("RUN_ID_HANDOFF_MODE", oct(mode))
    return RunIdHandoffIdentity(info.st_dev, info.st_ino, mode)


def create_run_id_handoff(path: Path, run_id: str) -> RunIdHandoffIdentity:
    value = validate_identifier(run_id, field="run_id")
    _, parent_fd = _handoff_parent(path)
    fd = -1
    try:
        fd = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        encoded = (value + "\n").encode("ascii")
        if os.write(fd, encoded) != len(encoded):
            raise OSError("short run-ID handoff write")
        os.fsync(fd)
        identity = _validate_handoff_fd(fd)
        os.fsync(parent_fd)
        return identity
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(parent_fd)


def _read_run_id_handoff(path: Path) -> tuple[str, RunIdHandoffIdentity, int]:
    _, parent_fd = _handoff_parent(path)
    try:
        fd = os.open(
            path.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except Exception:
        os.close(parent_fd)
        raise
    try:
        identity = _validate_handoff_fd(fd)
        current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (identity.device, identity.inode):
            raise RuntimeBlocked("RUN_ID_HANDOFF_REPLACED", str(path))
        encoded = os.read(fd, 130)
        if len(encoded) > 129 or not encoded.endswith(b"\n") or encoded.count(b"\n") != 1:
            raise RuntimeBlocked("RUN_ID_HANDOFF_CONTENT", str(path))
        value = validate_identifier(encoded[:-1].decode("ascii"), field="run_id")
        return value, identity, parent_fd
    except Exception:
        os.close(parent_fd)
        raise
    finally:
        os.close(fd)


def read_run_id_handoff(path: Path) -> str:
    value, _, parent_fd = _read_run_id_handoff(path)
    os.close(parent_fd)
    return value


def remove_run_id_handoff(
    path: Path,
    *,
    expected_run_id: str,
    expected_identity: RunIdHandoffIdentity | None = None,
) -> None:
    value, identity, parent_fd = _read_run_id_handoff(path)
    try:
        if value != expected_run_id:
            raise RuntimeBlocked("RUN_ID_HANDOFF_VALUE", value)
        if expected_identity is not None and identity != expected_identity:
            raise RuntimeBlocked("RUN_ID_HANDOFF_IDENTITY", str(path))
        current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (identity.device, identity.inode):
            raise RuntimeBlocked("RUN_ID_HANDOFF_REPLACED", str(path))
        os.unlink(path.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


@dataclass(frozen=True, slots=True)
class Gate9RetirementReceipt:
    run_id: str
    verdict: str
    manifest_sha256: str
    handoff_device: int
    handoff_inode: int


class Gate9HandoffReconciler:
    def __init__(
        self,
        output_parent: Path,
        *,
        recover_owned_run: Callable[[str, Path], None],
    ) -> None:
        self.output_parent = EvidenceStore.validate_existing_parent(output_parent)
        self.recover_owned_run = recover_owned_run

    def reconcile_and_retire(
        self,
        handoff: Path,
        *,
        reason: str,
        fault_hook: Callable[[str], None] | None = None,
    ) -> Gate9RetirementReceipt:
        if reason != "interrupted-retry":
            raise RuntimeBlocked("RUN_ID_RETIRE_REASON", reason)
        run_id, handoff_identity, handoff_parent_fd = _read_run_id_handoff(handoff)
        os.close(handoff_parent_fd)
        store = EvidenceStore.open_or_create_reconciliation(
            self.output_parent,
            run_id=run_id,
            handoff_identity=handoff_identity,
        )
        self.recover_owned_run(run_id, store.root)
        if fault_hook is not None:
            fault_hook("after_terminalize")
        verified = store.verify_terminal_manifest(expected_run_id=run_id)
        if (
            verified.lifecycle != "TERMINAL"
            or verified.infrastructure_verdict != "FAIL"
            or verified.overall_verdict == "PASS"
            or verified.owned_processes_alive != ()
            or verified.unknown_remote_state is not False
        ):
            raise RuntimeBlocked("RUN_ID_RETIRE_NOT_FAILED_TERMINAL", run_id)
        receipt_payload = {
            "schema_version": 1,
            "run_id": run_id,
            "reason": reason,
            "verdict": verified.overall_verdict,
            "manifest_sha256": verified.manifest_sha256,
            "handoff_device": handoff_identity.device,
            "handoff_inode": handoff_identity.inode,
            "handoff_mode": handoff_identity.mode,
            "run_root_device": store.identity.device,
            "run_root_inode": store.identity.inode,
        }
        store.write_or_verify_exact(
            "operator-handoff-retired.json", receipt_payload
        )
        if fault_hook is not None:
            fault_hook("after_receipt")
            fault_hook("before_handoff_unlink")
        remove_run_id_handoff(
            handoff,
            expected_run_id=run_id,
            expected_identity=handoff_identity,
        )
        return Gate9RetirementReceipt(
            run_id=run_id,
            verdict=verified.overall_verdict,
            manifest_sha256=str(receipt_payload["manifest_sha256"]),
            handoff_device=handoff_identity.device,
            handoff_inode=handoff_identity.inode,
        )
```

`InfrastructureVerdict.evaluate` implements every PASS predicate in the approved design. Task verdicts are separate: a normal episode returning false leaves `infrastructure_valid=true` but overall verdict `FAIL`. Any missing evidence, unknown liveness, timeout, collision, GPU foreign process, hash mismatch, nonterminal runner/helper, live owned process/socket/port/container, or cleanup error makes infrastructure `FAIL`; it never upgrades an earlier failed run.

- [ ] **Step 6: Implement the lifecycle and unconditional cleanup**

Use exact states:

```python
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class LifecycleState(str, Enum):
    NEW = "NEW"
    LOCAL_ATTESTED = "LOCAL_ATTESTED"
    LEASED = "LEASED"
    PREFLIGHTED = "PREFLIGHTED"
    CONTAINER_STARTED = "CONTAINER_STARTED"
    SERVER_STARTED = "SERVER_STARTED"
    TUNNEL_READY = "TUNNEL_READY"
    SERVER_READY = "SERVER_READY"
    EVALUATING = "EVALUATING"
    CLEANING = "CLEANING"
    CLEAN = "CLEAN"
    FAILED = "FAILED"
    PROVENANCE_BLOCKED = "PROVENANCE_BLOCKED"
    STALE_OWNED_BLOCKED = "STALE_OWNED_BLOCKED"
    FOREIGN_BLOCKED = "FOREIGN_BLOCKED"


# Pure ordering/reduction primitives used by DedicatedPsi0Runtime.
@dataclass
class PersistedStateMachine:
    initial: LifecycleState
    persist: Callable[[LifecycleState], None]

    def __post_init__(self) -> None:
        self.history = [self.initial]

    def before_action(self, state: LifecycleState, action: Callable[[], None]) -> None:
        self.persist(state)
        self.history.append(state)
        action()


def next_episodes(
    requested: tuple[int, ...], *, completed: int, infrastructure_clean: bool
) -> tuple[int, ...]:
    if not infrastructure_clean:
        return ()
    return tuple(episode for episode in requested if episode > completed)


@dataclass(frozen=True, slots=True)
class VerdictSummary:
    infrastructure_valid: bool
    task_results: dict[int, bool]
    verdict: str


def reduce_verdict(
    *, task_results: dict[int, bool], infrastructure_valid: bool
) -> VerdictSummary:
    passed = infrastructure_valid and bool(task_results) and all(task_results.values())
    return VerdictSummary(infrastructure_valid, dict(task_results), "PASS" if passed else "FAIL")
```

Persist/fsync each transition before the next external action. Register cleanup before every resource start. Cleanup order is runner ownership unit/finalize, tunnel, tracer, server, remote evidence, container, local/remote postconditions. Every action receives one shared absolute deadline; failure is recorded and does not skip later actions. Terminal restoration, log closure, and manifest finalization execute unconditionally outside deadline short-circuiting. Foreign/unknown identity records `FOREIGN_BLOCKED` and sends no signal/stop for that resource.

- [ ] **Step 7: Implement the CLI and script entrypoints**

`cli.py` defines a Typer `app` with exact commands `freeze-provenance`, `create`, `status`, `evaluate`, and `stop`; only `stop` accepts `--recover-stale RUN_ID`. Its Gate 9 reconciliation form additionally requires the indivisible set `--output-root PARENT --run-id-handoff FILE --retire-terminal-handoff --retire-reason interrupted-retry`. Parse those five option groups into one `RetirementOptions` value before obtaining a backend or performing cleanup; zero groups means ordinary stop, all five means retirement, and every other subset is rejected. The complete form invokes `Gate9HandoffReconciler` with the normal identity-checked owned cleanup backend. The reconciler accepts only the canonical full-manifest verifier result; PASS, foreign, unknown, and nonterminal states retain the handoff and issue no signal. Every mutating command interprets `--output-root` as the existing attested workload-parent directory and combines it with the separately validated `--run-id`; it never accepts a full child path in `--output-root`. `create --preflight-only` is the Gate-4 form: it requires `<output-root>/<run-id>` to be absent, exclusively creates that child, executes the exact local and leased preflight used by normal `create`, including `prepare_output` and the distinct loader/CUDA probes, then releases the lease after proving no helper/container/workload exists. It never calls Docker create and records a terminal preflight-only manifest. Normal `create` has no changed behavior. `evaluate` likewise requires an existing approved parent plus an absent run-ID child, exact `--episodes 2,3`, optional unbound loopback `--local-port` default 22085, and existing approved profile/container. All subprocesses receive argv arrays and monotonic deadlines through injectable runners; no `shell=True`, `os.system`, glob kill, force option, or prior-output mutation exists.

`freeze-provenance --phase` accepts exactly `pc2`, `h100`, or `verify`; all three require the same existing run ID and immutable output root. `status` accepts optional `--run-id` plus `--verify-evidence` and remains read-only. `evaluate --stop-after` is absent by default and accepts only `warmup`; it is recorded as Gate-7 infrastructure evidence and exits through the full cleanup path before any runner episode operation. None of these options adds a sixth lifecycle command.

```python
# src/simple/eval_runtime/cli.py (indivisible Gate 9 parser)
# ruff: noqa: F821
@dataclass(frozen=True, slots=True)
class RetirementOptions:
    run_id: str
    output_parent: Path
    handoff: Path
    reason: str

    @classmethod
    def from_cli(
        cls,
        *,
        recover_stale: str | None,
        output_root: Path | None,
        run_id_handoff: Path | None,
        retire_terminal_handoff: bool,
        retire_reason: str | None,
    ) -> "RetirementOptions | None":
        present = (
            recover_stale is not None,
            output_root is not None,
            run_id_handoff is not None,
            retire_terminal_handoff,
            retire_reason is not None,
        )
        if not any(present):
            return None
        if not all(present):
            raise RuntimeBlocked("RETIREMENT_OPTIONS_PARTIAL", repr(present))
        if retire_reason != "interrupted-retry":
            raise RuntimeBlocked("RETIREMENT_REASON", str(retire_reason))
        return cls(
            run_id=validate_identifier(recover_stale, field="run_id"),
            output_parent=EvidenceStore.validate_existing_parent(output_root),
            handoff=run_id_handoff,
            reason=retire_reason,
        )


def execute_retirement(options: RetirementOptions, backend) -> None:
    Gate9HandoffReconciler(
        options.output_parent,
        recover_owned_run=backend.recover_owned_run,
    ).reconcile_and_retire(options.handoff, reason=options.reason)
```

```python
# scripts/psi0_eval_runtime.py
#!/usr/bin/env python3
from simple.eval_runtime.cli import app

if __name__ == "__main__":
    app()
```

Add:

```toml
# pyproject.toml [project.scripts]
psi0-eval-runtime = "simple.eval_runtime.cli:typer_main"
```

- [ ] **Step 8: Run GREEN and commit**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_evidence.py \
  tests/eval_runtime/test_manager.py tests/eval_runtime/test_cleanup.py \
  tests/eval_runtime/test_cli.py
.venv/bin/python scripts/psi0_eval_runtime.py --help
```

Expected: all tests pass; help lists five operations and no force option.

```bash
git add src/simple/eval_runtime scripts/psi0_eval_runtime.py pyproject.toml \
  tests/eval_runtime
git commit -m "feat: orchestrate dedicated PSI0 evaluation evidence"
```

### Task 12: Complete adversarial crash, race, boundary, and failure-phase coverage

**Files:**
- Modify: `src/simple/eval_runtime/cli.py`
- Create: `tests/eval_runtime/adversarial_adapters.py`
- Create: `tests/eval_runtime/test_crash_matrix.py`
- Create: `tests/eval_runtime/test_concurrency_matrix.py`
- Create: `tests/eval_runtime/test_failure_phases.py`
- Create: `tests/eval_runtime/test_security_boundaries.py`
- Create: `tests/eval_runtime/test_integration_cli_auth.py`
- Create: `tests/integration/eval_runtime/conftest.py`
- Create: `tests/integration/eval_runtime/test_local_publication.py`
- Create: `tests/integration/eval_runtime/test_runner_model_free.py`
- Create: `tests/integration/eval_runtime/test_owned_stale_recovery.py`
- Create: `tests/integration/eval_runtime/test_remote_fault_recovery.py`
- Modify: `tests/eval_runtime/static_files.txt`

- [ ] **Step 1: Add one executable crash matrix covering every write-ahead phase**

```python
# tests/eval_runtime/test_crash_matrix.py
from __future__ import annotations

import pytest

from .adversarial_adapters import NativeAdversarialAdapter

pytestmark = pytest.mark.native

BASE_CRASHES = (
    "base_allocated", "base_copied", "base_metadata_normalized",
    "mutation_child_before_prepared", "prepared", "rename_pending",
    "base_final_renamed", "base_receipt_create", "base_receipt_created",
    "base_complete",
)
CLOSURE_CRASHES = (
    "allocated", "source", "episode_data", "venv", "task_data_copy",
    "hssd_normalized", "payload_ready", "installing",
    "mutation_child_before_prepared", "prepared", "rename_pending",
    "final_renamed", "receipt_create", "receipt_created", "complete",
)
RUNNER_CRASHES = (
    "output_child_pending", "output_child_created", "upstream_connect_pending",
    "upstream_connected", "relay_start_pending", "relay_started",
    "evaluator_start_pending", "evaluator_started", "inflight_request",
    "stopping", "descendants_gone",
)


@pytest.mark.parametrize("phase", BASE_CRASHES)
def test_base_python_crash_is_recoverable(phase: str, tmp_path) -> None:
    result = NativeAdversarialAdapter(tmp_path).crash_and_recover("base", phase)
    assert result["final_absent"] or result["complete_and_receipted"]
    assert result["unknown_deletions"] == []


@pytest.mark.parametrize("phase", CLOSURE_CRASHES)
def test_closure_crash_is_recoverable(phase: str, tmp_path) -> None:
    result = NativeAdversarialAdapter(tmp_path).crash_and_recover("closure", phase)
    assert result["final_absent"] or result["complete_and_receipted"]
    assert result["unknown_deletions"] == []


@pytest.mark.parametrize("phase", RUNNER_CRASHES)
def test_runner_crash_has_terminal_owned_cleanup(phase: str, tmp_path) -> None:
    result = NativeAdversarialAdapter(tmp_path).crash_and_recover("runner", phase)
    assert result["terminal_journal"] is True
    assert result["cgroup_members"] == []
    assert result["socket_inodes"] == []
    assert result["foreign_signals"] == []
```

`NativeAdversarialAdapter` does not exist yet during RED. Its GREEN implementation below invokes only the five reviewed static binaries with a temporary `--test-root`; each helper forks real fixture children, persists real journals/locks/PID identities, exits at the named boundary, and reconstructs from disk in a fresh process. No success value is synthesized in Python.

- [ ] **Step 2: Add exact concurrency and authority tests**

```python
# tests/eval_runtime/test_concurrency_matrix.py
from __future__ import annotations

import pytest

from .adversarial_adapters import NativeAdversarialAdapter

pytestmark = pytest.mark.native


@pytest.mark.parametrize("resource", ["remote_lease", "base_python", "pc2_closure", "installer_recovery", "runner_recovery"])
def test_two_concurrent_claimers_have_one_winner(resource: str, tmp_path) -> None:
    result = NativeAdversarialAdapter(tmp_path).run_two(resource)
    assert result["winners"] == 1
    assert result["loser_mutations"] == []
    assert result["loser_signals"] == []


def test_evaluator_uid_cannot_reach_installer_or_construction_state(tmp_path) -> None:
    result = NativeAdversarialAdapter(tmp_path).authority_probe()
    assert result["installer_connect_errno"] in {13, 2}
    assert result["runner_control_connect_errno"] in {13, 2}
    assert result["visible_construction_paths"] == []


def test_same_account_swap_after_root_fd_open_is_rejected(tmp_path) -> None:
    result = NativeAdversarialAdapter(tmp_path).root_swap_probe()
    for key in (
        "chmod_failed", "rename_failed", "unlink_failed", "symlink_failed",
        "lookalike_swap_failed", "executed_original_fd_identities",
    ):
        assert result[key] is True
```

- [ ] **Step 3: Add native process/cgroup/socket cleanup failures**

```python
# tests/eval_runtime/test_failure_phases.py
from __future__ import annotations

import pytest

from .adversarial_adapters import NativeAdversarialAdapter

pytestmark = pytest.mark.native


@pytest.mark.parametrize(
    "phase",
    [
        "runner_before_relay", "relay_started", "evaluator_started",
        "event_pipe", "ack_pipe", "relay_eof", "evaluator_sigstop",
        "runner_terminal_write", "cgroup_reap", "socket_cleanup",
    ],
)
def test_native_failure_phase_runs_process_cleanup(phase: str, tmp_path) -> None:
    result = NativeAdversarialAdapter(tmp_path).failure_phase(phase)
    assert result["infrastructure_verdict"] == "FAIL"
    for key in (
        "runner_terminal", "service_cgroup_empty", "relay_absent",
        "evaluator_absent", "unix_sockets_absent", "upstream_socket_absent",
        "foreign_signals_empty",
    ):
        assert result[key] is True
```

Python-owned environment, recorder, render/write, result-persistence, and close failures are deliberately absent here: Task 9's `test_evaluator_python_lifecycle.py` injects those faults through the extracted production Python lifecycle. These native cases retain responsibility for process, cgroup, pipe, and socket cleanup only.

- [ ] **Step 4: Add syscall/path/network boundary assertions**

```python
# tests/eval_runtime/test_security_boundaries.py
from __future__ import annotations

import pytest

from .adversarial_adapters import NativeAdversarialAdapter

pytestmark = pytest.mark.native


@pytest.mark.parametrize("name", [".", "..", "a/b", "a\\b", "a\x00b", "é"])
def test_every_user_selected_segment_is_rejected_before_external_action(name: str, tmp_path) -> None:
    result = NativeAdversarialAdapter(tmp_path).identifier_preflight(name)
    assert result["status"] == "INVALID_IDENTIFIER"
    assert result["external_actions"] == []


@pytest.mark.parametrize(
    "syscall",
    ["socket", "connect", "bind", "listen", "mount", "setns", "unshare", "ptrace_kill"],
)
def test_relay_or_evaluator_forbidden_syscall_fails(syscall: str, tmp_path) -> None:
    result = NativeAdversarialAdapter(tmp_path).forbidden_syscall(syscall)
    assert result["errno"] == 1
    assert result["escape_effects"] == []
```

- [ ] **Step 5: Run RED, implement the native disk/process adapter, then run GREEN**

First run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_crash_matrix.py \
  tests/eval_runtime/test_concurrency_matrix.py \
  tests/eval_runtime/test_failure_phases.py \
  tests/eval_runtime/test_security_boundaries.py
```

Expected RED: collection fails with `ModuleNotFoundError: tests.eval_runtime.adversarial_adapters`. No native helper is invoked during RED.

Add the adapter. It never invents a result: every returned mapping is read from an exclusively created report written by one of the five reviewed release binaries.

```python
# tests/eval_runtime/adversarial_adapters.py
from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any


class NativeAdversarialAdapter:
    _BINARIES = {
        "installer": "psi0-eval-install-input",
        "pc2_installer": "psi0-eval-install-pc2-input",
        "remote": "psi0-eval-remote-helper",
        "runner": "psi0-eval-run-pc2-evaluator",
        "relay": "psi0-eval-policy-relay",
    }

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.repo = Path(__file__).parents[2].resolve()
        self.release = (
            self.repo
            / "native/psi0_eval_runtime/target/x86_64-unknown-linux-gnu/release"
        )
        self.reports = self.root / "reports"
        self.reports.mkdir(mode=0o700)

    def _run(
        self,
        binary_key: str,
        operation: str,
        *,
        arguments: tuple[str, ...] = (),
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        binary = (self.release / self._BINARIES[binary_key]).resolve(strict=True)
        report = self.reports / f"{binary.name}.{operation}.json"
        if report.exists():
            raise AssertionError(f"native report already exists: {report}")
        completed = subprocess.run(
            [
                str(binary),
                "--test-root",
                str(self.root),
                "--test-operation",
                operation,
                *arguments,
                "--report",
                str(report),
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        assert completed.returncode == 0, (
            f"{binary.name} {operation} failed: {completed.stderr}"
        )
        report_fd = os.open(report, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        with os.fdopen(report_fd, "rb") as fd:
            before = os.fstat(fd.fileno())
            assert stat.S_ISREG(before.st_mode)
            assert before.st_nlink == 1 and before.st_size > 0
            payload = json.load(fd)
            assert fd.read(1) == b""
            after = os.fstat(fd.fileno())
            assert (before.st_dev, before.st_ino, before.st_size) == (
                after.st_dev,
                after.st_ino,
                after.st_size,
            )
            current = report.stat(follow_symlinks=False)
            assert (after.st_dev, after.st_ino) == (current.st_dev, current.st_ino)
        assert type(payload) is dict
        assert payload["schema_version"] == 1
        assert payload["operation"] == operation
        assert type(payload["result"]) is dict
        return payload["result"]

    def crash_and_recover(self, kind: str, phase: str) -> dict[str, Any]:
        key = "pc2_installer" if kind in {"base", "closure"} else "runner"
        return self._run(
            key,
            "crash-and-recover",
            arguments=("--kind", kind, "--fault-point", phase),
        )

    def run_two(self, resource: str) -> dict[str, Any]:
        key = (
            "remote"
            if resource == "remote_lease"
            else (
                "pc2_installer"
                if resource in {"base_python", "pc2_closure", "installer_recovery"}
                else "runner"
            )
        )
        return self._run(
            key,
            "two-claimers",
            arguments=("--resource", resource),
        )

    def authority_probe(self) -> dict[str, Any]:
        return self._run("runner", "authority-probe")

    def root_swap_probe(self) -> dict[str, Any]:
        return self._run("runner", "root-swap-probe")

    def failure_phase(self, phase: str) -> dict[str, Any]:
        return self._run("runner", "failure-phase", arguments=("--fault-point", phase))

    def identifier_preflight(self, name: str) -> dict[str, Any]:
        return self._run(
            "runner",
            "identifier-preflight",
            arguments=("--input-hex", name.encode("utf-8").hex()),
        )

    def forbidden_syscall(self, syscall: str) -> dict[str, Any]:
        key = (
            "relay" if syscall in {"socket", "connect", "bind", "listen"} else "runner"
        )
        return self._run(key, "forbidden-syscall", arguments=("--syscall", syscall))
```

Implement the native `--test-root` drivers after the RED result. These options are accepted only when the root is an existing mode-`0700` directory owned by the caller, the report is a nonexistent direct child of `<test-root>/reports`, and `SIMPLE_EVAL_RUNTIME_UNIT_TEST=1`; production invocations reject every test-only option. Each driver must use the production journal, lock, identifier, root-FD, cgroup/process, cleanup, and seccomp code paths. Crash drivers fork a fixture child, inject `_exit(86)` at the named write-ahead boundary, then invoke recovery in a fresh child which reconstructs state exclusively from disk. Concurrency drivers use two children released from a pipe barrier and report their independently persisted outcomes. Failure drivers launch the inert relay/evaluator process tree and verify real PIDs, cgroups, pipe/socket inodes, journals, and cleanup postconditions; they do not emulate Python environment or video cleanup. The report is created with `O_CREAT|O_EXCL|O_NOFOLLOW`, canonical JSON, `fsync`, and parent-directory `fsync`; it echoes the operation and input values and contains the measured result only. Unit tests also invoke every driver twice with different roots and assert inode-disjoint journals, locks, reports, and child identities.

Rerun the same command. Expected GREEN: every parameterized case passes against real disk/process boundaries; every helper report is unique and nonempty; no network, Docker, SSH, GPU, simulator, or policy model starts.

- [ ] **Step 6: Write RED tests for authenticated integration-only CLI forms**

```python
# tests/eval_runtime/test_integration_cli_auth.py
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from simple.eval_runtime.cli import app
from .fakes import FakeCliBackend

runner = CliRunner()


def _start_ticks(pid: int) -> int:
    tail = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").rsplit(")", 1)[1]
    return int(tail.split()[19])


def _create_marker(
    path: Path,
    *,
    operation: str,
    fixture: str | None,
    token: str,
    report: Path,
) -> int:
    run_root = path.parents[1]
    output_parent = run_root.parent
    run_root_identity = run_root.stat()
    output_parent_identity = output_parent.stat()
    fd = os.open(
        path,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    identity = os.fstat(fd)
    payload = {
        "schema_version": 1,
        "run_id": run_root.name,
        "run_root_device": run_root_identity.st_dev,
        "run_root_inode": run_root_identity.st_ino,
        "output_parent": str(output_parent),
        "output_parent_device": output_parent_identity.st_dev,
        "output_parent_inode": output_parent_identity.st_ino,
        "operation": operation,
        "fixture": fixture,
        "token_sha256": hashlib.sha256(token.encode("ascii")).hexdigest(),
        "creator_pid": os.getpid(),
        "creator_start_ticks": _start_ticks(os.getpid()),
        "marker_device": identity.st_dev,
        "marker_inode": identity.st_ino,
        "marker_mode": stat.S_IMODE(identity.st_mode),
        "report_path": str(report),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    os.write(fd, encoded + b"\n")
    os.fsync(fd)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    os.lseek(fd, 0, os.SEEK_SET)
    return fd


def _authenticated_options(
    *,
    output_parent: Path,
    run_id: str,
    marker: Path,
    marker_fd: int,
    token: str,
    report: Path,
) -> list[str]:
    return [
        "--run-id",
        run_id,
        "--output-root",
        str(output_parent),
        "--integration-marker-path",
        str(marker),
        "--integration-marker-fd",
        str(marker_fd),
        "--integration-token",
        token,
        "--integration-report",
        str(report),
    ]


def _gate_paths(tmp_path: Path, operation: str) -> tuple[Path, str, Path, Path]:
    output_parent = tmp_path / "workloads"
    output_parent.mkdir(exist_ok=True, mode=0o700)
    run_id = f"gate-{operation}"
    run_root = output_parent / run_id
    run_root.mkdir(mode=0o700)
    auth = run_root / "integration-auth"
    evidence = run_root / "evidence"
    auth.mkdir(mode=0o700)
    evidence.mkdir(mode=0o700)
    return output_parent, run_id, auth / "gate.marker", evidence / "integration-report.json"


def test_integration_form_is_rejected_without_enablement_and_marker(tmp_path: Path) -> None:
    backend = FakeCliBackend()
    result = runner.invoke(
        app,
        ["freeze-provenance", "--phase", "pc2-crash-fixtures"],
        obj=backend,
    )
    assert result.exit_code != 0
    assert backend.integration_calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("argv", "operation", "fixture"),
    [
        (["freeze-provenance", "--phase", "pc2-crash-fixtures"], "local-publication", None),
        (["status", "--model-free-runner-probe"], "runner-model-free", None),
        (["stop", "--owned-stale-fixture", "freeze"], "owned-stale", "freeze"),
        (["stop", "--remote-fault-fixture", "ssh-sever"], "remote-fault", "ssh-sever"),
    ],
)
def test_authenticated_integration_form_is_single_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    operation: str,
    fixture: str | None,
) -> None:
    monkeypatch.setenv("SIMPLE_EVAL_INTEGRATION", "1")
    monkeypatch.setenv("SIMPLE_EVAL_RUNTIME_UNIT_TEST", "1")
    output_parent, run_id, marker, report = _gate_paths(tmp_path, operation)
    token = "ab" * 32
    marker_fd = _create_marker(
        marker,
        operation=operation,
        fixture=fixture,
        token=token,
        report=report,
    )
    backend = FakeCliBackend()
    options = _authenticated_options(
        output_parent=output_parent,
        run_id=run_id,
        marker=marker,
        marker_fd=marker_fd,
        token=token,
        report=report,
    )
    try:
        accepted = runner.invoke(app, [*argv, *options], obj=backend)
        assert accepted.exit_code == 0
        assert backend.integration_calls == [(operation, fixture)]
        assert json.loads(report.read_text(encoding="utf-8"))["marker"] == operation
        assert marker.with_suffix(".marker.consumed").is_file()
        manifest = json.loads(
            (output_parent / run_id / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["run_id"] == run_id
        assert manifest["lifecycle"] == "TERMINAL"
        replay = runner.invoke(app, [*argv, *options], obj=backend)
        assert replay.exit_code != 0
        assert backend.integration_calls == [(operation, fixture)]
    finally:
        os.close(marker_fd)


def _rewrite_marker(fd: int, payload: dict[str, object] | bytes) -> None:
    encoded = (
        payload
        if isinstance(payload, bytes)
        else json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    assert os.write(fd, encoded) == len(encoded)
    os.fsync(fd)
    os.lseek(fd, 0, os.SEEK_SET)


@pytest.mark.parametrize(
    "mutation",
    [
        "token", "inode", "creator_pid", "creator_start_ticks", "mode",
        "run_id", "run_root_inode", "output_parent_inode", "operation",
        "fixture", "report_path", "malformed", "consumed", "existing_report",
    ],
)
def test_each_integration_authentication_mutation_fails_independently(
    mutation: str, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SIMPLE_EVAL_INTEGRATION", "1")
    monkeypatch.setenv("SIMPLE_EVAL_RUNTIME_UNIT_TEST", "1")
    output_parent, run_id, marker, report = _gate_paths(tmp_path, "owned-stale")
    marker_fd = _create_marker(
        marker,
        operation="owned-stale",
        fixture="freeze",
        token="ab" * 32,
        report=report,
    )
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    cli_token = "ab" * 32
    consumed = marker.with_suffix(".marker.consumed")
    if mutation == "token":
        cli_token = "cd" * 32
    elif mutation == "inode":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(marker.read_bytes())
        replacement.chmod(0o600)
        os.replace(replacement, marker)
    elif mutation == "creator_pid":
        marker_payload["creator_pid"] = os.getpid() + 1
        _rewrite_marker(marker_fd, marker_payload)
    elif mutation == "creator_start_ticks":
        marker_payload["creator_start_ticks"] += 1
        _rewrite_marker(marker_fd, marker_payload)
    elif mutation == "mode":
        marker.chmod(0o640)
    elif mutation == "run_id":
        marker_payload["run_id"] = "gate-other"
        _rewrite_marker(marker_fd, marker_payload)
    elif mutation == "run_root_inode":
        marker_payload["run_root_inode"] += 1
        _rewrite_marker(marker_fd, marker_payload)
    elif mutation == "output_parent_inode":
        marker_payload["output_parent_inode"] += 1
        _rewrite_marker(marker_fd, marker_payload)
    elif mutation == "operation":
        marker_payload["operation"] = "remote-fault"
        _rewrite_marker(marker_fd, marker_payload)
    elif mutation == "fixture":
        marker_payload["fixture"] = "create"
        _rewrite_marker(marker_fd, marker_payload)
    elif mutation == "report_path":
        marker_payload["report_path"] = str(tmp_path / "other-report.json")
        _rewrite_marker(marker_fd, marker_payload)
    elif mutation == "malformed":
        _rewrite_marker(marker_fd, b"{\n")
    elif mutation == "consumed":
        consumed.write_bytes(b"foreign")
        consumed.chmod(0o600)
    elif mutation == "existing_report":
        report.write_bytes(b"foreign")
        report.chmod(0o600)
    report_before = report.read_bytes() if report.exists() else None
    backend = FakeCliBackend()
    try:
        result = runner.invoke(
            app,
            [
                "stop",
                "--owned-stale-fixture",
                "freeze",
                *_authenticated_options(
                    output_parent=output_parent,
                    run_id=run_id,
                    marker=marker,
                    marker_fd=marker_fd,
                    token=cli_token,
                    report=report,
                ),
            ],
            obj=backend,
        )
        assert result.exit_code != 0
        assert backend.integration_calls == []
        if report_before is None:
            assert not report.exists()
        else:
            assert report.read_bytes() == report_before
        if mutation != "consumed":
            assert not consumed.exists()
    finally:
        os.close(marker_fd)


def test_profile_binding_failure_precedes_consumption(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIMPLE_EVAL_INTEGRATION", "1")
    monkeypatch.setenv("SIMPLE_EVAL_RUNTIME_UNIT_TEST", "1")
    output_parent, run_id, marker, report = _gate_paths(
        tmp_path, "runner-model-free"
    )
    token = "ab" * 32
    marker_fd = _create_marker(
        marker,
        operation="runner-model-free",
        fixture=None,
        token=token,
        report=report,
    )
    backend = FakeCliBackend()
    backend.integration_profile_approved = False
    try:
        result = runner.invoke(
            app,
            [
                "status",
                "--model-free-runner-probe",
                *_authenticated_options(
                    output_parent=output_parent,
                    run_id=run_id,
                    marker=marker,
                    marker_fd=marker_fd,
                    token=token,
                    report=report,
                ),
            ],
            obj=backend,
        )
        assert result.exit_code != 0
        assert backend.integration_calls == []
        assert not marker.with_suffix(".marker.consumed").exists()
        assert not report.exists()
    finally:
        os.close(marker_fd)
```

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_integration_cli_auth.py
```

Expected RED: all four integration-only argument forms are rejected because their authenticated parser/dispatcher does not exist; the ordinary five-operation CLI remains unchanged.

Implement hidden integration options in `src/simple/eval_runtime/cli.py` only after RED. They are indivisible: output parent, run ID, exact fixture argument, marker path, inherited marker FD, 64-hex token, and contained report path must all be present. Validation requires `SIMPLE_EVAL_INTEGRATION=1`, an already approved profile whose `simple_source_commit == HEAD^`, an attested existing output parent, a fresh mode-`0700` run root created by the live pytest parent, a mode-`0600` regular marker created with `O_EXCL`, exact path/FD device and inode equality, exact marker owner and creator PID/start ticks, token digest, operation/fixture/report equality, and absent report/`.consumed` paths. The marker's run ID, output-parent path/device/inode, and run-root device/inode must equal the CLI arguments and current no-follow FDs. Production obtains and verifies source/profile binding through the same strict loader used by normal commands. Only in `SIMPLE_EVAL_RUNTIME_UNIT_TEST=1` may an in-process `CliRunner` use its injected backend's `require_integration_profile_binding()` result and permit creator identity to equal the current process; the unit-only branch is rejected unless both conditions hold. Profile and run-root validation precedes marker consumption. The CLI exclusively creates `.consumed`, fsyncs it and the parent, invokes the named fixture, then exclusively writes/fsyncs the report and terminal manifest inside that run. Invalid, stale, replaced, replayed, partial, path-escaping, profile-mismatched, or ordinary production invocations perform zero fixture/backend actions. Task 12 therefore explicitly modifies `cli.py`; Task 11's normal public contracts do not recognize these forms.

Rerun:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_integration_cli_auth.py
```

Expected GREEN: all four authenticated forms dispatch exactly once, every replay/replacement/token mismatch is rejected before backend action, and the normal five-operation CLI surface remains unchanged.

- [ ] **Step 7: Add the four explicitly gated integration drivers**

```python
# tests/integration/eval_runtime/conftest.py
from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest


UNEXPECTED_SKIPS: list[str] = []


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration_local: approved local Gate 3")
    config.addinivalue_line(
        "markers", "integration_owned_stale: approved owned-stale Gate 6"
    )
    config.addinivalue_line(
        "markers", "integration_remote_fault: approved remote-fault Gate 8"
    )


@dataclass(frozen=True, slots=True)
class GateResult:
    report: dict[str, object]
    stdout: str
    stderr: str
    run_id: str
    run_root: Path


def _start_ticks(pid: int) -> int:
    tail = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").rsplit(")", 1)[1]
    return int(tail.split()[19])


def _create_gate_marker(
    path: Path,
    *,
    operation: str,
    fixture: str | None,
    token: str,
    report: Path,
) -> int:
    run_root = path.parents[1]
    output_parent = run_root.parent
    run_root_identity = run_root.stat()
    output_parent_identity = output_parent.stat()
    fd = os.open(
        path,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    identity = os.fstat(fd)
    payload = {
        "schema_version": 1,
        "run_id": run_root.name,
        "run_root_device": run_root_identity.st_dev,
        "run_root_inode": run_root_identity.st_ino,
        "output_parent": str(output_parent),
        "output_parent_device": output_parent_identity.st_dev,
        "output_parent_inode": output_parent_identity.st_ino,
        "operation": operation,
        "fixture": fixture,
        "token_sha256": hashlib.sha256(token.encode("ascii")).hexdigest(),
        "creator_pid": os.getpid(),
        "creator_start_ticks": _start_ticks(os.getpid()),
        "marker_device": identity.st_dev,
        "marker_inode": identity.st_ino,
        "marker_mode": stat.S_IMODE(identity.st_mode),
        "report_path": str(report),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    os.write(fd, encoded + b"\n")
    os.fsync(fd)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    os.lseek(fd, 0, os.SEEK_SET)
    return fd


@pytest.fixture
def gate_runner():
    def run(
        *argv: str,
        marker: str,
        fixture: str | None = None,
        timeout: float = 300.0,
    ) -> GateResult:
        if os.environ.get("SIMPLE_EVAL_INTEGRATION") != "1":
            pytest.fail(
                "SIMPLE_EVAL_INTEGRATION=1 is mandatory for an integration gate",
                pytrace=False,
            )
        output_parent = Path(os.environ["SIMPLE_EVAL_GATE_OUTPUT_ROOT"]).resolve(
            strict=True
        )
        parent_info = output_parent.stat()
        assert stat.S_ISDIR(parent_info.st_mode)
        assert parent_info.st_uid == os.getuid()
        run_id = f"gate-{marker}-{secrets.token_hex(8)}"
        run_root = output_parent / run_id
        run_root.mkdir(mode=0o700)
        auth_root = run_root / "integration-auth"
        evidence_root = run_root / "evidence"
        auth_root.mkdir(mode=0o700)
        evidence_root.mkdir(mode=0o700)
        marker_path = auth_root / "gate.marker"
        report = evidence_root / "integration-report.json"
        token = secrets.token_hex(32)
        marker_fd = _create_gate_marker(
            marker_path,
            operation=marker,
            fixture=fixture,
            token=token,
            report=report,
        )
        try:
            completed = subprocess.run(
                [
                    *argv,
                    "--run-id",
                    run_id,
                    "--output-root",
                    str(output_parent),
                    "--integration-marker-path",
                    str(marker_path),
                    "--integration-marker-fd",
                    str(marker_fd),
                    "--integration-token",
                    token,
                    "--integration-report",
                    str(report),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                pass_fds=(marker_fd,),
            )
        finally:
            os.close(marker_fd)
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert payload["marker"] == marker
        assert payload["run_id"] == run_id
        assert Path(payload["run_root"]) == run_root
        journal_paths = payload["journal_paths"]
        assert isinstance(journal_paths, list) and journal_paths
        for relative in journal_paths:
            journal = (run_root / relative).resolve(strict=True)
            assert journal.is_relative_to(run_root)
            assert journal.is_file() and journal.stat().st_size > 0
        assert marker_path.with_suffix(".marker.consumed").is_file()
        manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["run_id"] == run_id
        assert manifest["lifecycle"] == "TERMINAL"
        assert manifest["integration_report_sha256"] == hashlib.sha256(
            report.read_bytes()
        ).hexdigest()
        return GateResult(
            payload, completed.stdout, completed.stderr, run_id, run_root
        )

    return run


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.skipped:
        UNEXPECTED_SKIPS.append(report.nodeid)


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if report.skipped:
        UNEXPECTED_SKIPS.append(report.nodeid)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del exitstatus
    if UNEXPECTED_SKIPS:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
```

```python
# tests/integration/eval_runtime/test_local_publication.py
import pytest


pytestmark = [pytest.mark.integration_local]


def test_all_local_publication_crash_points_and_races(gate_runner) -> None:
    result = gate_runner(
        ".venv/bin/psi0-eval-runtime", "freeze-provenance",
        "--phase", "pc2-crash-fixtures", marker="local-publication", timeout=1800.0,
    )
    assert result.report["crash_points_failed"] == []
    assert result.report["constructor_winners"] == 1
    assert result.report["recoverer_winners"] == 1
    assert result.report["unknown_deletions"] == []
```

```python
# tests/integration/eval_runtime/test_runner_model_free.py
import pytest


pytestmark = [pytest.mark.integration_local]


def test_private_root_loader_cuda_network_and_fake_relay(gate_runner) -> None:
    result = gate_runner(
        ".venv/bin/psi0-eval-runtime", "status", "--model-free-runner-probe",
        marker="runner-model-free", timeout=600.0,
    )
    assert result.report["operations"] == ["prepare_output", "loader_probe", "cuda_probe"]
    assert result.report["interfaces"] == ["lo"]
    assert result.report["default_routes"] == []
    assert result.report["host_mappings"] == []
    assert result.report["relay_reconnects"] == 0
```

```python
# tests/integration/eval_runtime/test_owned_stale_recovery.py
import pytest


pytestmark = [pytest.mark.integration_owned_stale]


@pytest.mark.parametrize("fixture", ["freeze", "create", "evaluate-inert"])
def test_two_recoverers_have_one_winner_and_no_foreign_signal(fixture: str, gate_runner) -> None:
    result = gate_runner(
        ".venv/bin/psi0-eval-runtime", "stop", "--owned-stale-fixture", fixture,
        marker="owned-stale", fixture=fixture, timeout=600.0,
    )
    assert result.report["recovery_winners"] == 1
    assert result.report["loser_signals"] == []
    assert result.report["foreign_signals"] == []
    assert result.report["container_final"] in {"absent", "exited"}
```

```python
# tests/integration/eval_runtime/test_remote_fault_recovery.py
import pytest


pytestmark = [pytest.mark.integration_remote_fault]


@pytest.mark.parametrize("fault", ["tunnel-interrupt", "helper-timeout", "ssh-sever"])
def test_remote_fault_is_reconciled_by_identity(fault: str, gate_runner) -> None:
    result = gate_runner(
        ".venv/bin/psi0-eval-runtime", "stop", "--remote-fault-fixture", fault,
        marker="remote-fault", fixture=fault, timeout=900.0,
    )
    assert result.report["detached_helper_terminal"] is True
    assert result.report["owned_children_alive"] == []
    assert result.report["foreign_signals"] == []
    assert result.report["container_final"] == "exited"
```

The four integration-only argument forms are compiled into the existing commands but hidden from normal help. Each invocation requires `SIMPLE_EVAL_INTEGRATION=1`, the approved profile, an existing attested workload parent, a fresh validated run ID, and an exclusively created mode-`0700` `<output-root>/<run-id>` containing only its mode-`0700` `integration-auth/` and `evidence/` children. The inherited O_EXCL marker must pass token, inode, creator PID/start-time, mode, operation, fixture, single-use, and contained report-path validation. The CLI adopts only that exact authenticated empty gate root, writes all native journals and reports beneath it, fsyncs the terminal manifest, and never removes or rewrites it. The report and manifest bind the run ID, absolute root identity, source/profile identity, fixture parameters, native journal digests, measured result, and cleanup postconditions. Ordinary production invocations cannot adopt a precreated root. Every staged integration command exports both `SIMPLE_EVAL_INTEGRATION=1` and `SIMPLE_EVAL_GATE_OUTPUT_ROOT`; missing enablement, missing root, or any unexpected pytest skip fails the session.

- [ ] **Step 8: Run the complete unit/native suite and commit**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider tests/eval_runtime
CARGO_NET_OFFLINE=true cargo test --offline --locked \
  --manifest-path native/psi0_eval_runtime/Cargo.toml
```

Expected: all tests pass with no skips except tests explicitly marked `integration` and not collected from this path.

```bash
git add tests/eval_runtime tests/integration/eval_runtime \
  native/psi0_eval_runtime src/simple/eval_runtime
git commit -m "test: cover dedicated PSI0 runtime failure boundaries"
```

### Task 13: Add operator provisioning, runtime documentation, and the source-commit gate

**Files:**
- Create: `configs/psi0_h100_eval_freeze_v1.json`
- Create: `deploy/psi0_eval/install_pc2_services.sh`
- Create: `deploy/psi0_eval/install_h100_installer.sh`
- Create: `docs/psi0_dedicated_sim_evaluation.md`
- Modify: `README.md`
- Create: `tests/eval_runtime/test_freeze_config.py`
- Create: `tests/eval_runtime/test_provisioning_scripts.py`

- [ ] **Step 1: Write failing freeze-config and provisioning tests**

```python
# tests/eval_runtime/test_freeze_config.py
from __future__ import annotations

import json
from pathlib import Path


def test_freeze_config_has_only_fixed_design_inputs() -> None:
    config = json.loads(Path("configs/psi0_h100_eval_freeze_v1.json").read_text())
    assert set(config) == {
        "schema_version", "ssh_target", "container_name", "image_reference",
        "host_gpu_index", "container_gpu_index", "container_server_port",
        "pc2_gpu_index", "default_pc2_port", "task", "policy", "episodes",
        "checkpoint_step", "checkpoint_intake_path", "checkpoint_weight_relative_path",
        "checkpoint_weight_size", "checkpoint_weight_sha256",
        "server_source_relative_path", "server_source_sha256",
        "launcher_source_relative_path", "launcher_source_sha256",
        "simulation_mode", "wbc_env_type", "wbc_interface", "wbc_domain_id",
    }
    assert config["episodes"] == [2, 3]
    assert config["host_gpu_index"] == 7
    assert config["pc2_gpu_index"] == 1
    assert config["wbc_env_type"] == "sim"
    assert config["wbc_interface"] == "lo"
    assert config["wbc_domain_id"] == 0
    assert "@sha256:" in config["image_reference"]


def test_freeze_config_names_exact_checkpoint_weight() -> None:
    config = json.loads(Path("configs/psi0_h100_eval_freeze_v1.json").read_text())
    assert config["checkpoint_weight_size"] == 6253648840
    assert config["checkpoint_weight_sha256"] == "27df2e24c5efd176b962d2b219565056fc5081b69e050821a313249e677dd0f9"
    assert config["checkpoint_weight_relative_path"]
```

```python
# tests/eval_runtime/test_provisioning_scripts.py
from __future__ import annotations

from pathlib import Path


def test_pc2_provisioning_uses_exact_paths_and_starts_only_socket_units() -> None:
    script = Path("deploy/psi0_eval/install_pc2_services.sh").read_text()
    assert "/usr/local/sbin/psi0-eval-install-pc2-input" in script
    assert "/usr/local/libexec/psi0-eval-run-pc2-evaluator" in script
    assert "/usr/local/libexec/psi0-eval-policy-relay" in script
    assert "systemctl enable --now psi0-eval-pc2-installer.socket psi0-eval-pc2-runner.socket" in script
    assert "systemctl start psi0-eval-pc2-installer@" not in script
    assert "systemctl start psi0-eval-pc2-runner@" not in script
    assert "systemctl restart" not in script


def test_h100_provisioning_never_touches_shared_training_container() -> None:
    script = Path("deploy/psi0_eval/install_h100_installer.sh").read_text()
    assert "/usr/local/sbin/psi0-eval-install-input" in script
    assert "jihun_psi0_sonic_train_gpu23_20260805" not in script
    assert "docker stop" not in script
    assert "docker restart" not in script
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_freeze_config.py \
  tests/eval_runtime/test_provisioning_scripts.py
```

Expected: all assertions fail because files are absent.

- [ ] **Step 3: Commit the exact fixed freeze input**

Create `configs/psi0_h100_eval_freeze_v1.json` as canonical pretty JSON using the exact fixed identities from the approved design. The image reference must already be digest-qualified and reviewed; do not resolve a mutable tag in `freeze-provenance`. The exact `checkpoint_weight_relative_path` must be determined read-only from the selected run before this commit, contain no dot segment, and be the only 6,253,648,840-byte file with the required digest. This file is an intake selector, not the active 87-key runtime profile.

- [ ] **Step 4: Add idempotent, inspect-first provisioning scripts**

Both scripts accept `--check` or `--install`. `--check` compares byte hashes, owner, group, mode, socket/service/config content, fixed lock device/inode/mount identities, evaluator UID/GID separation, and same-mount `RENAME_NOREPLACE` capability without mutation. `--install` uses only fixed `install(1)` argv, systemctl daemon-reload, and `systemctl enable --now` for the two socket units after explicit operator sudo approval; no service instance starts until an authenticated request arrives, and the script does not start/restart a service instance, container, simulator, or model. It refuses a mismatched preexisting file instead of overwriting unless the operator separately removes it outside this workflow.

- [ ] **Step 5: Document exact operator workflow and safety boundary**

`docs/psi0_dedicated_sim_evaluation.md` contains:

1. prerequisites, `/mnt/data` symlink dependency, VPN/SSH check, PC2 GPU 1/H100 GPU 7 inspection, evaluator UID, protected roots, and service provisioning;
2. unit/static verification commands;
3. freeze candidate and commit `I`/`P` rules;
4. gates 2--9 with the exact commands from Tasks 14--17;
5. immutable evidence paths and PASS interpretation;
6. recovery commands and `FOREIGN_BLOCKED` behavior;
7. explicit statements that third-person video, real robot, `/contract`, `/act-rtc-v1`, and bridge certification are out of scope.

Update README's `🎯 Evaluation in SIMPLE` section with a short link to this document and retain all existing unmanaged evaluation commands.

- [ ] **Step 6: Run GREEN and complete all static/unit checks**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/eval_runtime/test_freeze_config.py \
  tests/eval_runtime/test_provisioning_scripts.py
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider tests/eval_runtime \
  tests/test_http_action_client.py tests/test_official_eval_compatibility.py
bash scripts/build_psi0_eval_native.sh
CARGO_NET_OFFLINE=true cargo test --offline --locked \
  --manifest-path native/psi0_eval_runtime/Cargo.toml
xargs -a tests/eval_runtime/static_files.txt ruff check --no-cache
xargs -a tests/eval_runtime/static_files.txt ruff format --check --no-cache
.venv/bin/python -m compileall -q src/simple/eval_runtime \
  src/simple/cli/eval_decoupled_wbc.py src/simple/baselines \
  src/simple/envs src/simple/scenes
git diff --check 0a3ad85029c5994cd017624e260276a3694a1b35..HEAD
git status --short
```

Expected: all tests/checks pass; status lists only task files and `?? outputs/` before commit.

- [ ] **Step 7: Create source commit `I`, push it, and freeze its identity**

```bash
git add configs/psi0_h100_eval_freeze_v1.json deploy/psi0_eval \
  docs/psi0_dedicated_sim_evaluation.md README.md tests/eval_runtime
git commit -m "docs: add dedicated PSI0 simulation runtime workflow"
git push origin feature/psi0-simple-pc2-bridge
git rev-parse HEAD
git status --short
```

Expected: push succeeds; record stdout as source commit `I`; the only status entry is `?? outputs/`. From this point until approval commit `P`, make no source, test, native, service, config, or documentation change.

### Task 14: Run Gates 1--2 and generate the candidate profile without starting the dedicated container

**Files:**
- Generated, never committed: `/mnt/data/jihun/psi0-simple-eval-workloads/<freeze-run-id>/candidate/psi0_h100_eval_runtime_v1.json`
- Generated evidence: `/mnt/data/jihun/psi0-simple-eval-workloads/<freeze-run-id>/`
- No tracked file changes

- [ ] **Step 1: Re-run Gate 1 from source commit `I`**

Run the exact full commands from Task 13 Step 6. Expected: all pass; only `?? outputs/` is untracked.

- [ ] **Step 2: Inspect GPU/container ownership before provisioning or freeze**

Run read-only:

```bash
nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
ssh h100 nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits
ssh h100 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
ssh h100 docker inspect jihun_psi0_sonic_train_gpu23_20260805 \
  --format '{{json .State}} {{json .HostConfig.DeviceRequests}}'
```

Expected: PC2 physical GPU 1 and H100 physical GPU 7 are identifiable; no foreign active compute process occupies either selected GPU. The shared training container is inspection-only and unchanged. If ownership is busy or any query is unknown, stop this task without signalling anything.

- [ ] **Step 3: Provision/check fixed PC2 and H100 helpers with explicit operator approval**

First run both `--check` forms. If installation is absent, obtain explicit sudo approval, then run:

```bash
sudo bash deploy/psi0_eval/install_pc2_services.sh --install
sudo bash deploy/psi0_eval/install_pc2_services.sh --check
source_I="$(git rev-parse HEAD)"
remote_provision="/tmp/psi0-eval-provision-$source_I"
ssh h100 install -d -m 0700 "$remote_provision"
scp native/psi0_eval_runtime/target/x86_64-unknown-linux-gnu/release/psi0-eval-install-input \
  deploy/psi0_eval/install_h100_installer.sh "h100:$remote_provision/"
ssh -t h100 sudo bash "$remote_provision/install_h100_installer.sh" --install
ssh h100 sudo -n bash "$remote_provision/install_h100_installer.sh" --check
```

Expected: fixed executable/config/unit hashes, users/groups/modes, lock identities, same-device rename probes, and socket activation pass. No service instance, container, simulator, or model starts during provisioning. Remove the exact `$remote_provision` directory through the installer script's bounded `--cleanup-staging "$remote_provision"` operation and verify it is absent.

- [ ] **Step 4: Seal/adopt PC2 base Python and closure under the local write-ahead constructor**

Run:

```bash
run_id_file="/tmp/psi0-eval-freeze-run-id-$(id -u)"
test ! -e "$run_id_file"
run_id="freeze-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short=12 HEAD)-$(openssl rand -hex 4)"
printf '%s\n' "$run_id" > "$run_id_file"
.venv/bin/psi0-eval-runtime freeze-provenance \
  --freeze-config configs/psi0_h100_eval_freeze_v1.json \
  --run-id "$run_id" \
  --output-root /mnt/data/jihun/psi0-simple-eval-workloads \
  --phase pc2
```

Expected: the constructor requires clean root/submodules, builds/probes the protected base Python, creates the relative venv before HSSD normalization, seals exact episode data/assets, passes offline episode-2/3 probes and real `HssdSceneManager.load("hssd:scene0")`, publishes through the installer, and stops with complete receipt/identity evidence. It imports no simulator during the pre-construction asset probes and starts no H100 workload.

- [ ] **Step 5: Freeze H100 server/checkpoint inputs and candidate profile**

Resume the same immutable run:

```bash
run_id_file="/tmp/psi0-eval-freeze-run-id-$(id -u)"
run_id="$(sed -n '1p' "$run_id_file")"
.venv/bin/psi0-eval-runtime freeze-provenance \
  --freeze-config configs/psi0_h100_eval_freeze_v1.json \
  --run-id "$run_id" \
  --output-root /mnt/data/jihun/psi0-simple-eval-workloads \
  --phase h100
```

Expected: an exclusive remote lease and detached helpers no-follow copy/seal the full official server/.venv/offline HF cache and full checkpoint run; privileged installer receipts supply final mode-inclusive hashes; protected checkpoint named weight matches exact path/size/hash; server spot hashes match; offline probes pass in the digest-pinned image; no dedicated container is created. Heartbeats remain current through 1,800-second helpers.

- [ ] **Step 6: Complete Gate 2 by verifying the candidate without a workload**

Run:

```bash
run_id_file="/tmp/psi0-eval-freeze-run-id-$(id -u)"
run_id="$(sed -n '1p' "$run_id_file")"
.venv/bin/psi0-eval-runtime freeze-provenance \
  --freeze-config configs/psi0_h100_eval_freeze_v1.json \
  --run-id "$run_id" \
  --output-root /mnt/data/jihun/psi0-simple-eval-workloads \
  --phase verify
```

Expected: strict 87-key candidate, exact commit/tree/gitlinks, protected receipts/completions, PC2/H100 manifests, UUIDs, binaries/services/configs/sandbox, seccomp path, checkpoint path/size/hash, and zero writable alias all pass. Container remains absent or exited. `git status --short` still shows only `?? outputs/`. This completes Gate 2's freeze/verification half; Gates 3 and 4 remain forbidden until Task 15 creates and pushes approval commit `P`.

### Task 15: Create and review the profile-only approval commit `P`

**Files:**
- Create: `configs/psi0_h100_eval_runtime_v1.json`
- No other tracked changes

- [ ] **Step 1: Install the verified candidate with the reviewed exclusive-copy helper**

Run:

```bash
run_id_file="/tmp/psi0-eval-freeze-run-id-$(id -u)"
run_id="$(sed -n '1p' "$run_id_file")"
candidate="/mnt/data/jihun/psi0-simple-eval-workloads/$run_id/candidate/psi0_h100_eval_runtime_v1.json"
test -f "$candidate"
.venv/bin/python -c 'from pathlib import Path; from simple.eval_runtime.profile import exclusive_copy_profile_candidate; import sys; exclusive_copy_profile_candidate(Path(sys.argv[1]), Path(sys.argv[2]))' \
  "$candidate" configs/psi0_h100_eval_runtime_v1.json
cmp --silent "$candidate" configs/psi0_h100_eval_runtime_v1.json
```

Expected: the destination is exclusively created with bytes identical to the candidate; no runtime/container starts.

- [ ] **Step 2: Verify the approval diff is exactly one path**

Run:

```bash
test -z "$(git diff --name-only)"
test -z "$(git diff --cached --name-only)"
test "$(git status --short --untracked-files=normal)" = \
  $'?? configs/psi0_h100_eval_runtime_v1.json\n?? outputs/'
git diff --check
source_I="$(git rev-parse HEAD)"
.venv/bin/python -c 'import hashlib,json,sys; from pathlib import Path; from simple.eval_runtime.contracts import parse_runtime_profile; p=Path("configs/psi0_h100_eval_runtime_v1.json"); b=p.read_bytes(); d=json.loads(b); parse_runtime_profile(d, blob_sha256=hashlib.sha256(b).hexdigest(), approval_commit="a"*40, source_commit=sys.argv[1]); print(len(d))' "$source_I"
```

Expected: tracked and staged diffs are empty because the exclusively copied profile is still untracked; the exact short status contains only the new profile plus preserved `outputs/`; the profile's `simple_source_commit` equals actual source commit `I`; whitespace is clean; stdout is `87`.

- [ ] **Step 3: Commit `P`, verify parent/diff, and push**

```bash
source_I="$(git rev-parse HEAD)"
git add configs/psi0_h100_eval_runtime_v1.json
test "$(git diff --cached --name-only)" = "configs/psi0_h100_eval_runtime_v1.json"
git diff --cached --check
git commit -m "config: approve dedicated PSI0 evaluation runtime"
approval_P="$(git rev-parse HEAD)"
test "$(git rev-parse HEAD^)" = "$source_I"
test "$(git diff --name-only "$source_I".."$approval_P")" = "configs/psi0_h100_eval_runtime_v1.json"
git diff --check "$source_I".."$approval_P"
git push origin feature/psi0-simple-pc2-bridge
git status --short
run_id_file="/tmp/psi0-eval-freeze-run-id-$(id -u)"
unlink "$run_id_file"
```

Expected: all assertions pass; local/remote branch parity; only `?? outputs/` remains. Do not amend `P` and do not make another code commit before Gates 3--9 complete; any code change requires a new `I` and new profile freeze/approval cycle.

### Task 16: Run staged Gates 3--8 with no official episode execution

**Files:**
- Generated immutable evidence only: `/mnt/data/jihun/psi0-simple-eval-workloads/<gate-run-id>/`
- Remote transient evidence only: `/mnt/data01/jhkim/psi0-simple-eval-workloads/runs/<gate-run-id>/`
- No tracked changes

- [ ] **Step 1: Gate 3—run local crash/race/private-root/relay fixtures against approved `P`**

Run:

```bash
SIMPLE_EVAL_INTEGRATION=1 \
SIMPLE_EVAL_GATE_OUTPUT_ROOT=/mnt/data/jihun/psi0-simple-eval-workloads \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -rs -p no:cacheprovider \
  -m integration_local \
  tests/integration/eval_runtime/test_local_publication.py \
  tests/integration/eval_runtime/test_runner_model_free.py
```

Expected: zero skips; every base/closure crash point recovers; concurrent constructors/recoverers have one winner; same-account swaps fail; `prepare_output`, loader probe, CUDA probe, private loopback namespace, connected-FD fake relay, and manager/runner crash fixtures pass. Each pytest invocation prints and preserves its distinct `gate-<operation>-<nonce>` directory with authentication marker, consumed marker, native journals, report, and terminal manifest under the configured workload parent. The integration driver verifies that the active approved profile's `simple_source_commit` equals `P^`; no container, H100 server, model, simulator, or episode starts.

- [ ] **Step 2: Gate 4—run status plus the exact local/leased preflight**

Run:

```bash
preflight_run_id="preflight-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short=12 HEAD)-$(openssl rand -hex 4)"
.venv/bin/psi0-eval-runtime status
.venv/bin/psi0-eval-runtime create \
  --preflight-only \
  --run-id "$preflight_run_id" \
  --output-root /mnt/data/jihun/psi0-simple-eval-workloads
```

Expected: read-only status and the exact normal-create preflight validate path containment, mount aliases, both GPUs, protected receipts and source/profile binding, services/sandboxes, distinct runner output/probe children, private roots, loopback networking, policy connected-FD transport, remote helpers, and pre/post provenance. The preflight-only manifest is terminal and immutable; lease/helper cleanup passes; Docker create is never called; no container, server, model, simulator, evaluator episode, or policy request starts.

- [ ] **Step 3: Gate 5—create and attest the model-free stopped container**

Run:

```bash
gate_run_id="create-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short=12 HEAD)-$(openssl rand -hex 4)"
.venv/bin/psi0-eval-runtime create \
  --run-id "$gate_run_id" \
  --output-root /mnt/data/jihun/psi0-simple-eval-workloads
```

Expected: profile, receipt, manifest, and GPU checks pass; Docker creates the exact digest-pinned container with protected source/checkpoint/seccomp mounts; model-free offline, read-only, mount-alias, CUDA UUID, and interruptible tracer-detach probes pass; final Docker state is exactly `exited`; no server, model, tunnel, or evaluator remains.

- [ ] **Step 4: Verify Gate 5 evidence and stopped state independently**

Run read-only:

```bash
.venv/bin/psi0-eval-runtime status
ssh h100 docker inspect jihun_psi0_simple_eval_gpu7 --format '{{.State.Status}}'
ssh h100 docker port jihun_psi0_simple_eval_gpu7
ssh h100 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
```

Expected: status reports a clean reusable container; Docker stdout is `exited`; port output is empty; H100 GPU 7 has no process from the dedicated container. The shared training container is unchanged.

- [ ] **Step 5: Gate 6—exercise owned stale fixtures and concurrent recovery claims**

Run the explicit integration file, which creates only inert/model-free owned fixtures:

```bash
SIMPLE_EVAL_INTEGRATION=1 \
SIMPLE_EVAL_GATE_OUTPUT_ROOT=/mnt/data/jihun/psi0-simple-eval-workloads \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -rs -p no:cacheprovider \
  -m integration_owned_stale \
  tests/integration/eval_runtime/test_owned_stale_recovery.py
```

Expected for freeze staging, interrupted create, and inert `SIGSTOP` evaluate fixture: two claimers race, exactly one recovery token wins, old token is fenced, no foreign process is signalled, terminal operation-specific postcondition passes, container returns to `exited`, and each case retains an independent run-ID-bound immutable gate directory.

- [ ] **Step 6: Gate 7—start the unchanged official server only long enough for trace, warm-up, and cleanup**

Run:

```bash
server_run_id="warmup-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short=12 HEAD)-$(openssl rand -hex 4)"
.venv/bin/psi0-eval-runtime evaluate \
  --run-id "$server_run_id" \
  --output-root /mnt/data/jihun/psi0-simple-eval-workloads \
  --episodes 2,3 \
  --stop-after warmup
```

Expected: the exact interruptible tracer argv starts the unchanged official server against `/checkpoint`; trace proves one open of the profiled protected weight and no unexpected checkpoint; tracer detaches cleanly via tracer-only SIGINT; server reaches port 22185; loopback tunnel reaches it; canonical warm-up returns HTTP 200 and finite `(24, 36)`; H100 monitoring permits only the attested server tree; cleanup removes tracer/server/tunnel/helper/GPU processes and leaves the container `exited`. No simulator/evaluator/episode runs because `--stop-after warmup` is an enumerated Gate 7 mode.

- [ ] **Step 7: Gate 8—inject tunnel/helper/SSH faults and prove ownership cleanup**

Run:

```bash
SIMPLE_EVAL_INTEGRATION=1 \
SIMPLE_EVAL_GATE_OUTPUT_ROOT=/mnt/data/jihun/psi0-simple-eval-workloads \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -rs -p no:cacheprovider \
  -m integration_remote_fault \
  tests/integration/eval_runtime/test_remote_fault_recovery.py
```

Expected: one tunnel interruption, one internally timed-out helper, and one forcibly severed originating SSH transport during a live owned model-free mutation each produce a distinct run-ID-bound immutable gate directory, detached-helper reconciliation from a fresh SSH session, bounded exact-child cleanup, Docker daemon postcondition, clean lease/recovery outcome, no foreign signal, and exited container.

- [ ] **Step 8: Reverify protected inputs and exact postconditions**

Run:

```bash
.venv/bin/psi0-eval-runtime status
git status --short
git diff --check 0a3ad85029c5994cd017624e260276a3694a1b35..HEAD
```

Expected: source/checkpoint/PC2 manifests and root identities still equal the profile; no helper/runner/relay/tunnel/server/port/GPU process remains; container is exited; only `?? outputs/` is untracked; whitespace is clean.

### Task 17: Run official simulation episodes 2 and 3 and certify Gate 9 evidence

**Files:**
- Generated immutable run: `/mnt/data/jihun/psi0-simple-eval-workloads/<evaluation-run-id>/`
- Remote transient run: `/mnt/data01/jhkim/psi0-simple-eval-workloads/runs/<evaluation-run-id>/`
- Generated operator handoff: `/tmp/psi0-eval-evaluation-run-id-<uid>`
- No tracked changes

- [ ] **Step 1: Recheck selected GPUs and stopped ownership immediately before the run**

Run read-only:

```bash
.venv/bin/psi0-eval-runtime status
nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
ssh h100 nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits
ssh h100 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
```

Expected: PC2 GPU 1 and H100 GPU 7 match approved UUIDs and have no foreign compute process; dedicated container is `exited`; no lease, server, tunnel, runner, relay, or evaluator is live. If any state is busy or unknown, do not start.

- [ ] **Step 2: Run the exact official simulation-only evaluation**

Run:

```bash
evaluation_run_id_file="/tmp/psi0-eval-evaluation-run-id-$(id -u)"
if test -e "$evaluation_run_id_file"; then
  previous_evaluation_run_id="$(.venv/bin/python -c '
import sys
from pathlib import Path
from simple.eval_runtime.evidence import read_run_id_handoff
print(read_run_id_handoff(Path(sys.argv[1])))
' "$evaluation_run_id_file")"
  .venv/bin/psi0-eval-runtime stop \
    --recover-stale "$previous_evaluation_run_id" \
    --output-root /mnt/data/jihun/psi0-simple-eval-workloads \
    --run-id-handoff "$evaluation_run_id_file" \
    --retire-terminal-handoff \
    --retire-reason interrupted-retry
fi
test ! -e "$evaluation_run_id_file"
evaluation_run_id="eval-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short=12 HEAD)-$(openssl rand -hex 4)"
.venv/bin/python -c '
import sys
from pathlib import Path
from simple.eval_runtime.evidence import create_run_id_handoff
create_run_id_handoff(Path(sys.argv[1]), sys.argv[2])
' "$evaluation_run_id_file" "$evaluation_run_id"
evaluation_run_id="$(.venv/bin/python -c '
import sys
from pathlib import Path
from simple.eval_runtime.evidence import read_run_id_handoff
print(read_run_id_handoff(Path(sys.argv[1])))
' "$evaluation_run_id_file")"
.venv/bin/psi0-eval-runtime evaluate \
  --run-id "$evaluation_run_id" \
  --output-root /mnt/data/jihun/psi0-simple-eval-workloads \
  --episodes 2,3 \
  --local-port 22085
```

Expected: an absent handoff proceeds immediately. A retained handoff first authenticates its exact old run, completes only identity-owned cleanup, writes a terminal FAIL manifest plus immutable retirement receipt, preserves the whole old run, and retires the exact handoff inode. PASS, foreign, unknown, or nonterminal old state stops before a new ID is generated. Then one exclusive lease, protected preflight, container/server/tracer/tunnel/warm-up, runner probes, episode 2 and 3, reverse-order cleanup, exited container, and lease release all complete. The evaluator command is exactly the approved `psi0_decoupled_wbc`, `mujoco_isaac`, headless, one-worker, managed-FD, standard-video argv. No third-person flag or real interface exists.

- [ ] **Step 3: Validate runtime-contract/event ordering for both episodes**

Run:

```bash
evaluation_run_id_file="/tmp/psi0-eval-evaluation-run-id-$(id -u)"
evaluation_run_id="$(.venv/bin/python -c '
import sys
from pathlib import Path
from simple.eval_runtime.evidence import read_run_id_handoff
print(read_run_id_handoff(Path(sys.argv[1])))
' "$evaluation_run_id_file")"
.venv/bin/psi0-eval-runtime status \
  --run-id "$evaluation_run_id" \
  --verify-evidence
```

Expected for each episode: distinct nonce and evidence file; `ENV_TYPE=sim`, `INTERFACE=lo`, `DOMAIN_ID=0`; closure/base/sidecar/UID/GID/network/relay/socket identities equal the profile and runner response; event order is `runtime_contract(validated)` → exact acknowledgement → `worker_init(creating_env)` → ready/start/steps/end/closing/done; no gap, malformed frame, or early EOF; WBC evidence predates `gym.make`.

- [ ] **Step 4: Validate runner, relay, network, and GPU ownership between episodes**

Reload the exact run identity rather than relying on shell state, then run the same immutable verifier:

```bash
evaluation_run_id_file="/tmp/psi0-eval-evaluation-run-id-$(id -u)"
evaluation_run_id="$(.venv/bin/python -c '
import sys
from pathlib import Path
from simple.eval_runtime.evidence import read_run_id_handoff
print(read_run_id_handoff(Path(sys.argv[1])))
' "$evaluation_run_id_file")"
.venv/bin/psi0-eval-runtime status \
  --run-id "$evaluation_run_id" \
  --verify-evidence
```

The verifier must report for episode 2 before episode 3 starts:

```text
runner journal TERMINAL
original service cgroup absent
evaluator/worker/WBC descendants absent
relay absent
both Unix policy socket inodes absent
upstream TCP socket inode absent
PC2 GPU poll complete and no owned PID remains
H100 GPU poll complete with only attested server descendants
```

Episode 3 has an independent record with a different nonce, child directory, Unix sockets, upstream connection, relay PID, and evaluator PID. Any infrastructure cleanup failure blocks episode 3 and makes the run FAIL.

- [ ] **Step 5: Validate raw and checked head-stereo artifacts**

Reload the exact run identity and rerun the artifact-bearing evidence verifier:

```bash
evaluation_run_id_file="/tmp/psi0-eval-evaluation-run-id-$(id -u)"
evaluation_run_id="$(.venv/bin/python -c '
import sys
from pathlib import Path
from simple.eval_runtime.evidence import read_run_id_handoff
print(read_run_id_handoff(Path(sys.argv[1])))
' "$evaluation_run_id_file")"
.venv/bin/psi0-eval-runtime status \
  --run-id "$evaluation_run_id" \
  --verify-evidence
```

For every allowlisted camera in each episode require both:

```text
evaluator-output/episode_<N>/videos/episode_<N>/<camera>.raw.mp4
evaluator-output/episode_<N>/videos/episode_<N>/<camera>_<verdict>.mp4
```

The artifact verifier checks absent-before creation, retained nonempty raw, atomic verdict publication, SHA-256, H.264 codec, resolution, frame rate/count/duration, FFmpeg/FFprobe logs, and nonblank decodable first/middle/final frames. It performs no third-person or semantic-success classification.

- [ ] **Step 6: Verify final cleanup and manifest verdict**

Run read-only:

```bash
evaluation_run_id_file="/tmp/psi0-eval-evaluation-run-id-$(id -u)"
evaluation_run_id="$(.venv/bin/python -c '
import sys
from pathlib import Path
from simple.eval_runtime.evidence import read_run_id_handoff
print(read_run_id_handoff(Path(sys.argv[1])))
' "$evaluation_run_id_file")"
.venv/bin/psi0-eval-runtime status \
  --run-id "$evaluation_run_id" \
  --verify-evidence
ssh h100 docker inspect jihun_psi0_simple_eval_gpu7 --format '{{.State.Status}}'
ssh h100 docker port jihun_psi0_simple_eval_gpu7
pgrep -af 'psi0-eval|eval_decoupled_wbc|ssh -N -L 127.0.0.1:22085' || true
lsof -nP -iTCP:22085 -sTCP:LISTEN || true
git status --short
```

Expected: Docker state `exited`; no published port; no owned manager/helper/tunnel/runner/relay/evaluator/WBC process or listener; both selected GPUs have no owned process; protected pre/post manifests match; all mandatory evidence exists. `manifest.json` is the source of truth and `run-manifest.md` renders the same verdict. Task failure may make overall evaluation FAIL while infrastructure remains valid; only full infrastructure plus two task successes yields PASS. Git status remains only `?? outputs/`.

Keep the validated mode-`0600` handoff file through Task 18. It is removed only after the final source/profile and Gate 9 acceptance checks succeed.

### Task 18: Final source/profile audit and handoff

**Files:**
- No changes

- [ ] **Step 1: Verify local/remote commit parity and the `I`/`P` boundary**

Run:

```bash
git fetch origin feature/psi0-simple-pc2-bridge
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/feature/psi0-simple-pc2-bridge)"
source_I="$(git rev-parse HEAD^)"
approval_P="$(git rev-parse HEAD)"
test "$(git diff --name-only "$source_I".."$approval_P")" = "configs/psi0_h100_eval_runtime_v1.json"
git diff --check 0a3ad85029c5994cd017624e260276a3694a1b35..HEAD
git status --short
```

Expected: parity and one-profile-path assertions pass; range whitespace is clean; only `?? outputs/` remains.

- [ ] **Step 2: Re-run all unit, native, static, and compile gates**

Run the complete global commands at the top of this plan plus:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_http_action_client.py \
  tests/test_official_eval_compatibility.py
bash scripts/build_psi0_eval_native.sh
for binary in native/psi0_eval_runtime/target/x86_64-unknown-linux-gnu/release/psi0-eval-*; do
  readelf -lWd "$binary" | grep -Eq 'INTERP|NEEDED' && exit 1 || true
done
```

Expected: every test/check passes; all five native binaries remain static; no runtime process starts from these checks.

- [ ] **Step 3: Compare final evidence to the acceptance boundary**

Reload and validate the persisted run identity, use it on the Gate 9 directory, and remove the handoff only after the verifier succeeds:

```bash
evaluation_run_id_file="/tmp/psi0-eval-evaluation-run-id-$(id -u)"
evaluation_run_id="$(.venv/bin/python -c '
import sys
from pathlib import Path
from simple.eval_runtime.evidence import read_run_id_handoff
print(read_run_id_handoff(Path(sys.argv[1])))
' "$evaluation_run_id_file")"
.venv/bin/psi0-eval-runtime status \
  --run-id "$evaluation_run_id" \
  --verify-evidence
```

Require every PASS predicate in the design's Verdict Rules. Record separately:

- infrastructure verdict;
- episode 2 SIMPLE task verdict;
- episode 3 SIMPLE task verdict;
- overall evaluation verdict;
- profile approval commit `P` and source commit `I`;
- immutable local and remote evidence paths;
- explicit `UNCERTIFIED` status for the PC2 real-time bridge.

Only after those values are durably recorded and explicitly accepted, remove the handoff with its expected-value and inode checks:

```bash
evaluation_run_id_file="/tmp/psi0-eval-evaluation-run-id-$(id -u)"
evaluation_run_id="$(.venv/bin/python -c '
import sys
from pathlib import Path
from simple.eval_runtime.evidence import read_run_id_handoff
print(read_run_id_handoff(Path(sys.argv[1])))
' "$evaluation_run_id_file")"
.venv/bin/python -c '
import sys
from pathlib import Path
from simple.eval_runtime.evidence import remove_run_id_handoff
remove_run_id_handoff(Path(sys.argv[1]), expected_run_id=sys.argv[2])
' "$evaluation_run_id_file" "$evaluation_run_id"
test ! -e "$evaluation_run_id_file"
```

No result from this plan authorizes real robot control. Bridge certification remains separate and still requires corrected 32-D same-episode provenance, attested `/contract` and `/act-rtc-v1`, and 100 warmed requests with zero failures and p99 at most 0.10 seconds for `d=6`.

## Plan self-review checklist

- [ ] Every section of the approved design maps to at least one task: fixed identities/profile (1, 13--15), canonical manifests (2), PC2 closure/assets/base Python (4--5), remote helper/lease/container/GPU/tracer (6, 14--16), runner/private root/relay (7--8), evaluator WBC/event contract (9), standard video (10), evidence/lifecycle/cleanup (11), adversarial coverage (12), and staged gates (14--18).
- [ ] Search this plan for unfinished-marker patterns prohibited by the writing-plans skill; the result must be empty.
- [ ] Extract every fenced Python block and parse it with `ast.parse`; extract every fenced Bash block and run `bash -n`; correct every syntax error before committing this plan.
- [ ] Confirm every referenced production symbol is defined in a prior task or the same task and every referenced test fixture has a concrete definition in `tests/eval_runtime/conftest.py` or its test module.
- [ ] Confirm every code-changing task has RED, implementation, GREEN, and commit steps; operator-only staged gates make no source commit.
- [ ] Confirm `git diff --check 0a3ad85029c5994cd017624e260276a3694a1b35..HEAD` is silent and preserved `outputs/` is untouched.
