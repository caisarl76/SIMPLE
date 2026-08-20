# SIMPLE Third-Person Evaluation Camera Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a simulation-only, opt-in world-fixed third-person evaluation camera whose video artifact is backend-correct, policy-input-isolated, and safely finalized.

**Architecture:** Evaluation creates a canonical `CameraCfg` and passes it to `BaseDualSim` as an environment-owned sensor overlay. Registry construction becomes fresh for the complete nested dependency graph, sensor validation precedes simulator startup, and Isaac/MuJoCo implement the same world-frame camera contract. Video writers use checked bounded subprocesses, while shared lifecycle owners guarantee recorder, agent, wrapper, environment, and worker cleanup.

**Tech Stack:** Python 3.10, Gymnasium, NumPy, OpenCV, FFmpeg/FFprobe, MuJoCo, Isaac Sim, Typer, pytest, Ruff.

---

## Constraints and file map

- Simulation only. Do not start Unitree real DDS, a robot control loop, or a real-interface bridge.
- Preserve the untracked `outputs/` directory and existing evaluation evidence.
- Every production behavior follows red-green-refactor. A test must fail for the expected missing behavior before its implementation is added.
- The single official-policy E2E is the last task; unit and backend frame checks must pass first.

**Create:**

- `src/simple/evals/third_person_camera.py` — canonical camera, option validation, and environment kwargs.
- `src/simple/evals/lifecycle.py` — aggregate cleanup, episode-video ownership, and child reaping.
- `scripts/verify_third_person_camera.py` — one-reset backend frame verifier with no policy execution.
- `tests/test_third_person_eval_camera.py` — factory, registry, task ownership, and constructor-order tests.
- `tests/test_third_person_eval_backends.py` — Isaac/MuJoCo camera adapter and mixed-output tests.
- `tests/test_third_person_policy_isolation.py` — literal request-byte tests for both PSI0 agents.
- `tests/test_eval_video_finalization.py` — FFmpeg/fallback/raw-preservation and recorder deadline tests.
- `tests/test_eval_worker_cleanup.py` — episode/resource/child cleanup tests.

**Modify:**

- `src/simple/core/registry.py` — uncached nested construction context.
- `src/simple/core/task.py` — copy mutable task metadata per task instance.
- `src/simple/envs/base_dual_env.py` — isolated sensors, validation order, rollback, idempotent close.
- `src/simple/engines/isaacsim.py` — `eye_in_world` parent and configured clipping.
- `src/simple/engines/mujoco.py` — `eye_in_world` parent and effective global clipping.
- `src/simple/envs/video_writer.py` — checked bounded atomic finalization.
- `src/simple/envs/wrappers/video_recorder.py` — shared deadline and aggregate failures.
- `src/simple/evals/api.py` — frozen configuration field and validation.
- `src/simple/evals/env_runner.py` — public configuration, sensor kwargs, and lifecycle ownership.
- `src/simple/cli/eval.py` — option/worker plumbing and lifecycle ownership.
- `src/simple/cli/eval_decoupled_wbc.py` — option/worker plumbing and lifecycle ownership.

## Task 0: Record the clean simulation-only baseline

**Files:**

- Inspect: `docs/superpowers/specs/2026-08-18-simple-third-person-eval-camera-design.md`
- Inspect: all files in the map above

- [ ] **Step 1: Confirm the approved revision and worktree state**

Run:

```bash
git rev-parse HEAD
git merge-base --is-ancestor 83c175c HEAD
git status --short
git submodule status --recursive
```

Expected: the approved design commit `83c175c` is an ancestor of HEAD; only
preserved `outputs/` is untracked; submodules have no leading `-` or `+`.

- [ ] **Step 2: Confirm no evaluation or robot process is active**

Run:

```bash
pgrep -af 'eval-decoupled-wbc|eval-worker|serve_psi0|run_g1_control_loop|psi0_simple_real_bridge' || true
lsof -nP -iTCP:22085 -sTCP:LISTEN || true
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
```

Expected: no process owned by this task and no listener on 22085. Record unrelated GPU jobs and do not stop them.

- [ ] **Step 3: Run the focused pre-change suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_official_eval_compatibility.py \
  tests/test_http_action_client.py
```

Expected: PASS.

- [ ] **Step 4: Record the legacy lint/format debt without changing it**

Run the `legacy_modified_files` array from Task 11 Step 2 against full Ruff and
format checks, saving their output under the preserved evidence root:

```bash
mkdir -p outputs/third-person-camera-baseline
legacy_modified_files=(
  src/simple/core/registry.py
  src/simple/core/task.py
  src/simple/envs/base_dual_env.py
  src/simple/engines/isaacsim.py
  src/simple/engines/mujoco.py
  src/simple/envs/video_writer.py
  src/simple/envs/wrappers/video_recorder.py
  src/simple/evals/api.py
  src/simple/evals/env_runner.py
  src/simple/cli/eval.py
  src/simple/cli/eval_decoupled_wbc.py
)
set +e
ruff check --no-cache "${legacy_modified_files[@]}" \
  > outputs/third-person-camera-baseline/ruff.txt 2>&1
ruff_baseline_exit=$?
ruff format --check --no-cache "${legacy_modified_files[@]}" \
  > outputs/third-person-camera-baseline/format.txt 2>&1
format_baseline_exit=$?
set -e
test "$ruff_baseline_exit" -eq 1
test "$format_baseline_exit" -eq 1
```

Expected: the baseline records 46 existing Ruff findings and nine existing
format deltas. Do not edit unrelated lines to clear them in this feature.

## Task 1: Make registry construction fresh through nested dependencies

**Files:**

- Modify: `src/simple/core/registry.py`
- Modify: `src/simple/core/task.py`
- Create: `tests/test_third_person_eval_camera.py`

- [ ] **Step 1: Write failing registry tests**

Create the test file with these imports and tests:

```python
from __future__ import annotations

from dataclasses import dataclass

from simple.core.registry import RegistryMixin
from simple.core.task import Task


class ProbeDependencyRegistry(RegistryMixin[object]):
    pass


class ProbeTaskRegistry(RegistryMixin[object]):
    pass


@dataclass
class ProbeDependency:
    token: object


class ProbeTask:
    def __init__(self) -> None:
        self.dependency = ProbeDependencyRegistry.make("third-person-probe-dependency")


class ProbeMetadataTask(Task):
    metadata = {"render_hz": 30}

    def __init__(self, render_hz):
        super().__init__(dr=None, render_hz=render_hz)

    @property
    def layout(self):
        return None

    @property
    def instruction(self):
        return "probe"

    @property
    def action_space(self):
        return None


def install_probe_registry_entries() -> None:
    ProbeDependencyRegistry._registry["third-person-probe-dependency"] = (
        lambda: ProbeDependency(object())
    )
    ProbeTaskRegistry._registry["third-person-probe-task"] = ProbeTask
    ProbeDependencyRegistry._instances.pop("third-person-probe-dependency", None)
    ProbeTaskRegistry._instances.pop("third-person-probe-task", None)


def remove_probe_registry_entries() -> None:
    for key in ("third-person-probe-dependency", "third-person-probe-task"):
        ProbeTaskRegistry._registry.pop(key, None)
        ProbeTaskRegistry._instances.pop(key, None)


def test_make_keeps_legacy_singleton_behavior() -> None:
    install_probe_registry_entries()
    try:
        first = ProbeTaskRegistry.make("third-person-probe-task")
        second = ProbeTaskRegistry.make("third-person-probe-task")
        assert first is second
        assert first.dependency is second.dependency
    finally:
        remove_probe_registry_entries()


def test_make_fresh_isolates_task_and_nested_registry_dependencies() -> None:
    install_probe_registry_entries()
    try:
        first = ProbeTaskRegistry.make_fresh("third-person-probe-task")
        second = ProbeTaskRegistry.make_fresh("third-person-probe-task")
        assert first is not second
        assert first.dependency is not second.dependency
        assert first.dependency.token is not second.dependency.token
    finally:
        remove_probe_registry_entries()


def test_make_fresh_unknown_uid_matches_make_error() -> None:
    try:
        ProbeTaskRegistry.make_fresh("third-person-missing")
    except ValueError as error:
        assert str(error) == "No class registered under uid 'third-person-missing'"
    else:
        raise AssertionError("make_fresh accepted an unknown registry UID")


def test_task_metadata_is_owned_by_each_instance() -> None:
    first = ProbeMetadataTask(render_hz=10)
    second = ProbeMetadataTask(render_hz=20)
    assert first.metadata == {"render_hz": 10}
    assert second.metadata == {"render_hz": 20}
    assert ProbeMetadataTask.metadata == {"render_hz": 30}
```

The nested assertion is required because official task constructors call
`RobotRegistry.make`; fresh task objects that still share a robot are not
independent live environments.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_third_person_eval_camera.py
```

Expected: `test_make_fresh_*` fails because `make_fresh` does not exist; the legacy singleton test passes.

- [ ] **Step 3: Implement nested fresh construction**

Replace `RegistryMixin` in `src/simple/core/registry.py` with this behavior while retaining the license header and type variables:

```python
from contextvars import ContextVar
from typing import ClassVar, Generic, Type, TypeVar

T = TypeVar("T")
_S = TypeVar("_S")
_FRESH_CONSTRUCTION_DEPTH: ContextVar[int] = ContextVar(
    "simple_registry_fresh_construction_depth", default=0
)


class RegistryMixin(Generic[T]):
    _registry: ClassVar[dict[str, Type[T]]] = {}  # type: ignore
    _instances: ClassVar[dict[str, T]] = {}  # type: ignore

    @classmethod
    def register(cls, uid: str):
        def wrapper(subclass: Type[_S]) -> Type[_S]:
            cls._registry[uid] = subclass  # type: ignore
            return subclass

        return wrapper

    @classmethod
    def _construct(cls, uid: str, *args, **kwargs) -> T:
        if uid not in cls._registry:
            raise ValueError(f"No class registered under uid '{uid}'")
        return cls._registry[uid](*args, **kwargs)

    @classmethod
    def make(cls, uid: str, *args, **kwargs) -> T:
        if _FRESH_CONSTRUCTION_DEPTH.get() > 0:
            return cls._construct(uid, *args, **kwargs)
        if uid not in cls._instances:
            cls._instances[uid] = cls._construct(uid, *args, **kwargs)
        return cls._instances[uid]

    @classmethod
    def make_fresh(cls, uid: str, *args, **kwargs) -> T:
        depth = _FRESH_CONSTRUCTION_DEPTH.get()
        reset_token = _FRESH_CONSTRUCTION_DEPTH.set(depth + 1)
        try:
            return cls._construct(uid, *args, **kwargs)
        finally:
            _FRESH_CONSTRUCTION_DEPTH.reset(reset_token)
```

- [ ] **Step 4: Run registry tests and verify GREEN**

At the beginning of `Task.__init__`, before `self.metadata.update`, copy the
subclass mapping so constructor overrides cannot mutate other live tasks:

```python
import copy

self.metadata = copy.deepcopy(type(self).metadata)
```

Run the Task 1 Step 2 command.

Expected: 4 passed.

- [ ] **Step 5: Commit the registry unit**

```bash
git add src/simple/core/registry.py src/simple/core/task.py \
  tests/test_third_person_eval_camera.py
git commit -m "fix: isolate fresh registry dependency graphs"
```

## Task 2: Isolate task sensors and validate before simulator startup

**Files:**

- Modify: `src/simple/envs/base_dual_env.py`
- Modify: `tests/test_third_person_eval_camera.py`

- [ ] **Step 1: Add constructor-order, collision, and two-live-task tests**

First extend the import block at the top of
`tests/test_third_person_eval_camera.py` with:

```python
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

import simple.envs.base_dual_env as base_dual_env
from simple.envs.base_dual_env import BaseDualSim
from simple.sensors.config import CameraCfg
from simple.tasks.registry import TaskRegistry
```

Then append the following test support and tests below the Task 1 tests; do not
place new imports between test functions:

```python


def camera_cfg(name: str = "probe", near: float = 0.2, far: float = 5.0) -> CameraCfg:
    return CameraCfg(
        uid=name,
        mount="eye_in_world",
        width=64,
        height=36,
        focal_length=1.88,
        fov=float(np.deg2rad(90.0)),
        near=near,
        far=far,
        pose={"distance": 2.5, "polar": float(np.deg2rad(60.0)), "azimuth": 0.0},
    )


class LiveProbeTask:
    sensor_cfgs = {"head": camera_cfg("head")}

    def __init__(self) -> None:
        self.robot = SimpleNamespace(identity=object())
        self._layout = SimpleNamespace(cameras={})
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    @property
    def layout(self):
        return self._layout

    @property
    def observation_space(self):
        return spaces.Dict(
            {key: value.observation_space for key, value in self.sensor_cfgs.items()}
        )

    def reset(self) -> None:
        self._layout = SimpleNamespace(cameras=deepcopy(self.sensor_cfgs))


class FakeSimulator:
    def __init__(self, task, events, name):
        self.task = task
        self.events = events
        self.name = name
        events.append(f"create:{name}")

    def close(self):
        self.events.append(f"close:{self.name}")


def install_live_task() -> None:
    TaskRegistry._registry["third-person-live-task"] = LiveProbeTask
    TaskRegistry._instances.pop("third-person-live-task", None)


def remove_live_task() -> None:
    TaskRegistry._registry.pop("third-person-live-task", None)
    TaskRegistry._instances.pop("third-person-live-task", None)


def install_fake_simulators(monkeypatch, events):
    monkeypatch.setattr(
        base_dual_env,
        "_construct_mujoco_simulator",
        lambda task, headless: FakeSimulator(task, events, "mujoco"),
    )
    monkeypatch.setattr(
        base_dual_env,
        "_construct_isaac_simulator",
        lambda task, headless: FakeSimulator(task, events, "isaac"),
    )


def test_two_live_environments_keep_task_sensor_robot_and_layout_state_isolated(monkeypatch):
    install_live_task()
    events = []
    install_fake_simulators(monkeypatch, events)
    try:
        opt_in = BaseDualSim(
            "third-person-live-task",
            sim_mode="mujoco",
            extra_sensor_cfgs={"third_person": camera_cfg("third-person")},
        )
        opt_out = BaseDualSim("third-person-live-task", sim_mode="mujoco")
        opt_in.task.reset()
        opt_out.task.reset()

        assert opt_in.task is not opt_out.task
        assert opt_in.task.robot is not opt_out.task.robot
        assert opt_in.task.sensor_cfgs is not opt_out.task.sensor_cfgs
        assert opt_in.task.layout is not opt_out.task.layout
        assert set(opt_in.task.sensor_cfgs) == {"head", "third_person"}
        assert set(opt_out.task.sensor_cfgs) == {"head"}
        assert set(opt_in.task.layout.cameras) == {"head", "third_person"}
        assert set(opt_out.task.layout.cameras) == {"head"}

        opt_out.task.reset()
        assert set(opt_in.task.layout.cameras) == {"head", "third_person"}
        opt_in.close()
        opt_in.close()
        assert events.count("close:mujoco") == 1
        assert set(opt_out.task.layout.cameras) == {"head"}
        opt_out.close()
    finally:
        remove_live_task()


def test_sensor_collision_fails_before_any_simulator_resource(monkeypatch):
    install_live_task()
    events = []
    install_fake_simulators(monkeypatch, events)
    monkeypatch.setattr(
        BaseDualSim,
        "_init_isaac",
        lambda self, headless, webrtc: events.append("init:isaac"),
    )
    try:
        try:
            BaseDualSim(
                "third-person-live-task",
                sim_mode="mujoco_isaac",
                extra_sensor_cfgs={"head": camera_cfg("collision")},
            )
        except ValueError as error:
            assert "sensor key collision: head" in str(error)
        else:
            raise AssertionError("sensor collision was accepted")
        assert events == []
    finally:
        remove_live_task()


def test_incompatible_mujoco_clipping_fails_before_resources(monkeypatch):
    install_live_task()
    events = []
    install_fake_simulators(monkeypatch, events)
    try:
        try:
            BaseDualSim(
                "third-person-live-task",
                sim_mode="mujoco",
                extra_sensor_cfgs={
                    "third_person": camera_cfg("third-person", near=0.3, far=5.0)
                },
            )
        except ValueError as error:
            assert "camera clipping mismatch" in str(error)
        else:
            raise AssertionError("incompatible camera clipping was accepted")
        assert events == []
    finally:
        remove_live_task()


def test_preconstructed_task_rejects_sensor_injection_without_mutation(monkeypatch):
    task = LiveProbeTask()
    original_sensor_mapping = task.sensor_cfgs
    events = []
    install_fake_simulators(monkeypatch, events)
    with pytest.raises(ValueError, match="task UID string"):
        BaseDualSim(
            task,
            sim_mode="mujoco",
            extra_sensor_cfgs={"third_person": camera_cfg("third-person")},
        )
    assert task.sensor_cfgs is original_sensor_mapping
    assert set(task.sensor_cfgs) == {"head"}
    assert events == []
```

- [ ] **Step 2: Run the three new tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_third_person_eval_camera.py -k 'live_environments or collision or clipping'
```

Expected: failures for missing `extra_sensor_cfgs` support/factory seams and cached task use.

- [ ] **Step 3: Add sensor preparation and factory seams**

Add these functions above `BaseDualSim` in `base_dual_env.py`:

```python
import copy
import math
import sys
from collections.abc import Mapping

from simple.core.actor import CameraEntity
from simple.sensors.config import CameraCfg, SensorCfg


def _construct_isaac_simulator(task, headless):
    from simple.engines.isaacsim import IsaacSimSimulator

    return IsaacSimSimulator(task, headless=headless)


def _construct_mujoco_simulator(task, headless):
    from simple.engines import MujocoSimulator

    return MujocoSimulator(task, headless=headless)


def _validate_camera(name: str, cfg: CameraCfg) -> None:
    numeric = (cfg.width, cfg.height, cfg.focal_length, cfg.fov, cfg.near, cfg.far)
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError(f"camera {name} contains non-finite fields")
    if cfg.width <= 0 or cfg.height <= 0:
        raise ValueError(f"camera {name} resolution must be positive")
    if cfg.focal_length <= 0:
        raise ValueError(f"camera {name} focal_length must be positive")
    if not 0.0 < cfg.fov < math.pi:
        raise ValueError(f"camera {name} fov must be radians in (0, pi)")
    if cfg.near <= 0.0 or cfg.far <= cfg.near:
        raise ValueError(f"camera {name} clipping must satisfy 0 < near < far")
    if cfg.mount not in {"eye_in_hand", "eye_in_head", "eye_on_base", "eye_in_world"}:
        raise ValueError(f"camera {name} has unsupported mount {cfg.mount!r}")
    CameraEntity(name, cfg)


def _prepare_task_sensors(
    task,
    extra_sensor_cfgs: Mapping[str, SensorCfg] | None,
    sim_mode: str,
) -> None:
    base = copy.deepcopy(dict(task.sensor_cfgs))
    extra = copy.deepcopy(dict(extra_sensor_cfgs or {}))
    collision = sorted(set(base).intersection(extra))
    if collision:
        raise ValueError("sensor key collision: " + ", ".join(collision))
    for name, cfg in extra.items():
        if not isinstance(cfg, CameraCfg):
            raise TypeError(f"evaluation sensor {name} must be CameraCfg")
        _validate_camera(name, cfg)
    merged = {**base, **extra}
    if "mujoco" in sim_mode and "third_person" in extra:
        expected = (extra["third_person"].near, extra["third_person"].far)
        for name, cfg in merged.items():
            if isinstance(cfg, CameraCfg) and not (
                math.isclose(cfg.near, expected[0], rel_tol=0.0, abs_tol=1e-9)
                and math.isclose(cfg.far, expected[1], rel_tol=0.0, abs_tol=1e-9)
            ):
                raise ValueError(
                    f"camera clipping mismatch: {name}={(cfg.near, cfg.far)} "
                    f"third_person={expected}"
                )
    task.sensor_cfgs = merged
```

- [ ] **Step 4: Replace `BaseDualSim.__init__` and make close idempotent**

Use this constructor/close ownership logic; keep `_init_isaac`, `spin`, and the
Gym superclass behavior unchanged:

```python
def __init__(
    self,
    task,
    sim_mode="mujoco_isaac",
    headless=True,
    webrtc=False,
    extra_sensor_cfgs=None,
    *args,
    **kwargs,
):
    global _ISAAC_LOADED
    self.headless = headless
    self.webrtc = webrtc
    self.sim_mode = sim_mode
    self.isaac = None
    self.mujoco = None
    self._closed = False
    self._owns_isaac_app = False

    if isinstance(task, str):
        self.task = TaskRegistry.make_fresh(task, *args, **kwargs)
        _prepare_task_sensors(self.task, extra_sensor_cfgs, sim_mode)
    else:
        if extra_sensor_cfgs:
            raise ValueError(
                "extra_sensor_cfgs requires a task UID string for fresh ownership"
            )
        self.task = task
    self.action_space = self.task.action_space
    self.observation_space = self.task.observation_space

    try:
        if "isaac" in sim_mode and not _ISAAC_LOADED:
            self._init_isaac(headless, webrtc)
            self._owns_isaac_app = True
        if "isaac" in sim_mode:
            self.isaac = _construct_isaac_simulator(self.task, headless)
        self.mujoco = _construct_mujoco_simulator(
            self.task, headless=("isaac" in sim_mode) or headless
        )
    except BaseException:
        cleanup_errors = []
        if self.mujoco is not None:
            try:
                self.mujoco.close()
            except Exception as error:
                cleanup_errors.append(error)
        if self.isaac is not None:
            close_isaac = getattr(self.isaac, "close", None)
            if callable(close_isaac):
                try:
                    close_isaac()
                except Exception as error:
                    cleanup_errors.append(error)
        if self._owns_isaac_app:
            try:
                _close_simulation_app()
            except Exception as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            print(
                "constructor cleanup failures: "
                + "; ".join(map(str, cleanup_errors)),
                file=sys.stderr,
            )
        raise


def close(self):
    if self._closed:
        return
    self._closed = True
    errors = []
    if self.mujoco is not None:
        try:
            self.mujoco.close()
        except Exception as error:
            errors.append(error)
    if self.isaac is not None:
        close_isaac = getattr(self.isaac, "close", None)
        if callable(close_isaac):
            try:
                close_isaac()
            except Exception as error:
                errors.append(error)
    if self._owns_isaac_app:
        try:
            _close_simulation_app()
        except Exception as error:
            errors.append(error)
    try:
        super().close()
    except Exception as error:
        errors.append(error)
    if errors:
        raise RuntimeError("environment cleanup failed: " + "; ".join(map(str, errors)))
```

Add the global helper used by rollback and normal close:

```python
def _close_simulation_app() -> None:
    global _ISAAC_LOADED, _SIMULATION_APP
    app = _SIMULATION_APP
    try:
        if app is not None:
            app.close()
    finally:
        _SIMULATION_APP = None
        _ISAAC_LOADED = False
```

- [ ] **Step 5: Add and pass rollback tests**

Append these tests:

```python
def test_isaac_constructor_failure_closes_owned_application(monkeypatch):
    install_live_task()
    events = []

    class FakeApp:
        def close(self):
            events.append("close:app")

    def fake_init(self, headless, webrtc):
        events.append("init:isaac")
        base_dual_env._ISAAC_LOADED = True
        base_dual_env._SIMULATION_APP = FakeApp()

    def fail_isaac(task, headless):
        events.append("create:isaac")
        raise RuntimeError("isaac failed")

    monkeypatch.setattr(BaseDualSim, "_init_isaac", fake_init)
    monkeypatch.setattr(base_dual_env, "_construct_isaac_simulator", fail_isaac)
    base_dual_env._ISAAC_LOADED = False
    base_dual_env._SIMULATION_APP = None
    try:
        try:
            BaseDualSim("third-person-live-task", sim_mode="mujoco_isaac")
        except RuntimeError as error:
            assert str(error) == "isaac failed"
        else:
            raise AssertionError("Isaac construction failure was swallowed")
        assert events == ["init:isaac", "create:isaac", "close:app"]
        assert base_dual_env._ISAAC_LOADED is False
        assert base_dual_env._SIMULATION_APP is None
    finally:
        remove_live_task()


def test_mujoco_constructor_failure_closes_isaac_then_application(monkeypatch):
    install_live_task()
    events = []

    class FakeApp:
        def close(self):
            events.append("close:app")

    def fake_init(self, headless, webrtc):
        events.append("init:isaac")
        base_dual_env._ISAAC_LOADED = True
        base_dual_env._SIMULATION_APP = FakeApp()

    monkeypatch.setattr(BaseDualSim, "_init_isaac", fake_init)
    monkeypatch.setattr(
        base_dual_env,
        "_construct_isaac_simulator",
        lambda task, headless: FakeSimulator(task, events, "isaac"),
    )

    def fail_mujoco(task, headless):
        events.append("create:mujoco")
        raise RuntimeError("mujoco failed")

    monkeypatch.setattr(base_dual_env, "_construct_mujoco_simulator", fail_mujoco)
    base_dual_env._ISAAC_LOADED = False
    base_dual_env._SIMULATION_APP = None
    try:
        try:
            BaseDualSim("third-person-live-task", sim_mode="mujoco_isaac")
        except RuntimeError as error:
            assert str(error) == "mujoco failed"
        else:
            raise AssertionError("MuJoCo construction failure was swallowed")
        assert events == [
            "init:isaac",
            "create:isaac",
            "create:mujoco",
            "close:isaac",
            "close:app",
        ]
        assert base_dual_env._ISAAC_LOADED is False
        assert base_dual_env._SIMULATION_APP is None
    finally:
        remove_live_task()


def test_constructor_cleanup_failure_does_not_mask_original(monkeypatch, capsys):
    install_live_task()
    events = []

    class FakeApp:
        def close(self):
            events.append("close:app")

    class FailingIsaac:
        def close(self):
            events.append("close:isaac")
            raise RuntimeError("isaac cleanup failed")

    def fake_init(self, headless, webrtc):
        base_dual_env._ISAAC_LOADED = True
        base_dual_env._SIMULATION_APP = FakeApp()

    monkeypatch.setattr(BaseDualSim, "_init_isaac", fake_init)
    monkeypatch.setattr(
        base_dual_env,
        "_construct_isaac_simulator",
        lambda task, headless: FailingIsaac(),
    )
    monkeypatch.setattr(
        base_dual_env,
        "_construct_mujoco_simulator",
        lambda task, headless: (_ for _ in ()).throw(
            ValueError("mujoco construction failed")
        ),
    )
    base_dual_env._ISAAC_LOADED = False
    base_dual_env._SIMULATION_APP = None
    try:
        with pytest.raises(ValueError, match="mujoco construction failed"):
            BaseDualSim("third-person-live-task", sim_mode="mujoco_isaac")
        assert events == ["close:isaac", "close:app"]
        assert "isaac cleanup failed" in capsys.readouterr().err
    finally:
        remove_live_task()
```

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_third_person_eval_camera.py
```

Expected: all Task 1 and Task 2 tests pass.

- [ ] **Step 6: Commit task ownership and ordering**

```bash
git add src/simple/envs/base_dual_env.py tests/test_third_person_eval_camera.py
git commit -m "fix: validate isolated task sensors before simulation"
```

## Task 3: Add the canonical camera and evaluation configuration plumbing

**Files:**

- Create: `src/simple/evals/third_person_camera.py`
- Modify: `src/simple/evals/api.py`
- Modify: `src/simple/evals/env_runner.py`
- Modify: `src/simple/cli/eval.py`
- Modify: `src/simple/cli/eval_decoupled_wbc.py`
- Modify: `tests/test_third_person_eval_camera.py`

- [ ] **Step 1: Write failing factory/config tests**

Change the existing dataclass import at the top of
`tests/test_third_person_eval_camera.py` to `from dataclasses import dataclass,
fields`, then add these new-module imports to that same top-level import block:

```python
from simple.evals.third_person_camera import (
    THIRD_PERSON_SENSOR_KEY,
    make_environment_sensor_kwargs,
    make_third_person_camera,
    validate_video_options,
)
```

Append these tests without adding imports between existing tests:

```python


def test_third_person_factory_is_exact_and_fresh() -> None:
    first = make_third_person_camera()
    second = make_third_person_camera()
    assert THIRD_PERSON_SENSOR_KEY == "third_person"
    assert first is not second
    assert first.uid == "simple_eval_third_person_v1"
    assert first.mount == "eye_in_world"
    assert first.resolution == (640, 360)
    assert first.focal_length == 1.88
    assert first.fov == float(np.deg2rad(90.0))
    assert first.near == 0.2
    assert first.far == 5.0
    assert first.pose == {
        "distance": 2.5,
        "polar": float(np.deg2rad(60.0)),
        "azimuth": 0.0,
    }
    first.pose["distance"] = 99.0
    assert second.pose["distance"] == 2.5


def test_environment_sensor_kwargs_are_opt_in() -> None:
    assert make_environment_sensor_kwargs(False) == {}
    enabled = make_environment_sensor_kwargs(True)
    assert set(enabled) == {"extra_sensor_cfgs"}
    assert set(enabled["extra_sensor_cfgs"]) == {"third_person"}


def test_third_person_without_video_is_rejected() -> None:
    try:
        validate_video_options(third_person_video=True, save_video=False)
    except ValueError as error:
        assert str(error) == "third_person_video requires save_video"
    else:
        raise AssertionError("invalid video option combination was accepted")


def test_both_eval_config_types_expose_disabled_default() -> None:
    from simple.evals.api import EvalConfig as WorkerEvalConfig
    from simple.evals.env_runner import EvalConfig as PublicEvalConfig

    worker_names = {field.name for field in fields(WorkerEvalConfig)}
    public_names = {field.name for field in fields(PublicEvalConfig)}
    assert "third_person_video" in worker_names
    assert "third_person_video" in public_names
    assert WorkerEvalConfig(env_id="simple/probe", policy="psi0").third_person_video is False
    assert PublicEvalConfig(env_id="simple/probe").third_person_video is False
```

- [ ] **Step 2: Run factory/config tests and verify RED**

Run the Task 2 Step 5 command.

Expected: collection fails because `simple.evals.third_person_camera` does not exist.

- [ ] **Step 3: Create the canonical camera module**

```python
from __future__ import annotations

import numpy as np

from simple.sensors.config import CameraCfg

THIRD_PERSON_SENSOR_KEY = "third_person"


def make_third_person_camera() -> CameraCfg:
    return CameraCfg(
        uid="simple_eval_third_person_v1",
        mount="eye_in_world",
        width=640,
        height=360,
        focal_length=1.88,
        fov=float(np.deg2rad(90.0)),
        near=0.2,
        far=5.0,
        pose={
            "distance": 2.5,
            "polar": float(np.deg2rad(60.0)),
            "azimuth": 0.0,
        },
    )


def make_environment_sensor_kwargs(enabled: bool) -> dict:
    if not enabled:
        return {}
    return {
        "extra_sensor_cfgs": {
            THIRD_PERSON_SENSOR_KEY: make_third_person_camera(),
        }
    }


def validate_video_options(*, third_person_video: bool, save_video: bool) -> None:
    if third_person_video and not save_video:
        raise ValueError("third_person_video requires save_video")
```

- [ ] **Step 4: Add both configuration fields and validation points**

Add `third_person_video: bool = False` immediately after `save_video` in both
dataclasses. Call `validate_video_options` at the start of both CLI `run_eval`
functions and in `EnvRunner.__init__`, before directories, policy health, data,
or environments are touched.

The exact call is:

```python
validate_video_options(
    third_person_video=config.third_person_video,
    save_video=config.save_video,
)
```

- [ ] **Step 5: Plumb the option through testable parent and worker builders**

Add this Typer parameter to both `main` functions and both worker functions:

```python
third_person_video: Annotated[
    bool,
    typer.Option("--third-person-video/--no-third-person-video"),
] = False,
```

Add the same key to `EvalConfig(...)` and the values unpacked from `config`.
In `src/simple/cli/eval.py`, add this complete parent builder and make both the
single-worker call and `ctx.Process(..., args=(..., worker_kwargs, ...))` use
its return value:

```python
def _build_worker_kwargs(config: EvalConfig) -> dict[str, Any]:
    return {
        "env_id": config.env_id,
        "policy": config.policy,
        "split": config.split,
        "host": config.host,
        "port": config.port,
        "data_format": config.data_format,
        "sim_mode": config.sim_mode,
        "headless": config.headless,
        "eval_dir": config.eval_dir,
        "max_episode_steps": config.max_episode_steps,
        "num_episodes": config.num_episodes,
        "episode_start": config.episode_start,
        "data_dir": config.data_dir,
        "rollout_save_dir": config.rollout_save_dir,
        "success_criteria": config.success_criteria,
        "save_video": config.save_video,
        "third_person_video": config.third_person_video,
    }
```

Add the corresponding complete builder to
`src/simple/cli/eval_decoupled_wbc.py`:

```python
def _build_worker_kwargs(
    config: EvalConfig,
    *,
    sonic_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "env_id": config.env_id,
        "policy": config.policy,
        "split": config.split,
        "host": config.host,
        "port": config.port,
        "data_format": config.data_format,
        "sim_mode": config.sim_mode,
        "headless": config.headless,
        "eval_dir": config.eval_dir,
        "max_episode_steps": config.max_episode_steps,
        "num_episodes": config.num_episodes,
        "episode_start": config.episode_start,
        "data_dir": config.data_dir,
        "rollout_save_dir": config.rollout_save_dir,
        "success_criteria": config.success_criteria,
        "save_video": config.save_video,
        "third_person_video": config.third_person_video,
        "sonic_config": sonic_config,
    }
```

In each CLI module add this worker-local constructor and replace its direct
`gym.make(env_id, **make_kwargs)` call with it:

```python
def _make_worker_environment(
    env_id: str,
    *,
    make_kwargs: dict[str, Any],
    third_person_video: bool,
):
    environment_kwargs = dict(make_kwargs)
    environment_kwargs.update(
        make_environment_sensor_kwargs(third_person_video)
    )
    return gym.make(env_id, **environment_kwargs)
```

The worker call is exactly:

```python
environment = _make_worker_environment(
    env_id,
    make_kwargs=make_kwargs,
    third_person_video=third_person_video,
)
```

Add this identical process-construction seam to both CLI modules and use it in
their multi-worker loops:

```python
def _make_worker_process(
    context,
    *,
    worker_result_path,
    worker_id,
    num_workers,
    worker_kwargs,
    log_path,
    progress_connection,
):
    return context.Process(
        target=_run_eval_worker_entry,
        args=(
            worker_result_path,
            worker_id,
            num_workers,
            worker_kwargs,
            log_path,
            progress_connection,
        ),
        name=f"eval-worker-{worker_id}",
    )
```

In `EnvRunner._make_env`, extend its local `make_kwargs` exactly once:

```python
make_kwargs.update(
    make_environment_sensor_kwargs(self.config.third_person_video)
)
```

Do not add an empty `extra_sensor_cfgs` mapping when disabled.

- [ ] **Step 6: Add parent-to-worker and worker-to-environment regression tests**

Add these imports to the file's top-level import block. Append the exact CLI
tests without mid-file imports:

```python
import pytest
import typer
from typer.testing import CliRunner

from simple.cli import eval as eval_cli
from simple.cli import eval_decoupled_wbc as decoupled_cli
```

```python
def test_both_cli_entrypoints_preserve_third_person_flag(monkeypatch) -> None:
    for module in (eval_cli, decoupled_cli):
        captured = []
        monkeypatch.setattr(
            module,
            "run_eval",
            lambda config, **kwargs: captured.append(config),
        )
        app = typer.Typer()
        app.command()(module.main)
        result = CliRunner().invoke(
            app,
            ["simple/Probe-v0", "psi0", "train", "--third-person-video"],
        )
        assert result.exit_code == 0, result.output
        assert len(captured) == 1
        assert captured[0].third_person_video is True
        assert captured[0].save_video is True


def test_both_run_eval_functions_validate_before_environment_setup(monkeypatch) -> None:
    from simple.cli import eval as eval_cli
    from simple.cli import eval_decoupled_wbc as decoupled_cli
    from simple.evals.api import EvalConfig

    invalid = EvalConfig(
        env_id="simple/Probe-v0",
        policy="psi0",
        third_person_video=True,
        save_video=False,
    )
    monkeypatch.setattr(
        decoupled_cli,
        "_make_sonic_config",
        lambda: (_ for _ in ()).throw(AssertionError("setup ran")),
    )
    for module in (eval_cli, decoupled_cli):
        try:
            module.run_eval(invalid, show_progress=False)
        except ValueError as error:
            assert str(error) == "third_person_video requires save_video"
        else:
            raise AssertionError(f"{module.__name__} accepted invalid video options")


@pytest.mark.parametrize("enabled", [False, True])
def test_both_parent_builders_supply_exact_worker_flag(enabled) -> None:
    from simple.cli import eval as eval_cli
    from simple.cli import eval_decoupled_wbc as decoupled_cli
    from simple.evals.api import EvalConfig

    config = EvalConfig(
        env_id="simple/Probe-v0",
        policy="psi0",
        third_person_video=enabled,
    )
    ordinary = eval_cli._build_worker_kwargs(config)
    decoupled = decoupled_cli._build_worker_kwargs(
        config,
        sonic_config={"ENV_NAME": "simple"},
    )
    assert ordinary["third_person_video"] is enabled
    assert decoupled["third_person_video"] is enabled
    assert "sonic_config" not in ordinary
    assert decoupled["sonic_config"] == {"ENV_NAME": "simple"}


@pytest.mark.parametrize("module", [eval_cli, decoupled_cli])
@pytest.mark.parametrize("enabled", [False, True])
def test_spawned_process_receives_the_exact_built_worker_kwargs(module, enabled):
    calls = []

    class FakeContext:
        def Process(self, **kwargs):
            calls.append(kwargs)
            return object()

    worker_kwargs = {"third_person_video": enabled}
    process = module._make_worker_process(
        FakeContext(),
        worker_result_path="worker.pkl",
        worker_id=2,
        num_workers=3,
        worker_kwargs=worker_kwargs,
        log_path="worker.log",
        progress_connection=object(),
    )
    assert process is not None
    assert calls[0]["args"][3] is worker_kwargs
    assert calls[0]["args"][3]["third_person_video"] is enabled


@pytest.mark.parametrize("module", [eval_cli, decoupled_cli])
@pytest.mark.parametrize("enabled", [False, True])
def test_both_workers_pass_fresh_environment_camera_kwargs(
    monkeypatch, module, enabled
) -> None:
    calls = []
    monkeypatch.setattr(
        module.gym,
        "make",
        lambda env_id, **kwargs: calls.append((env_id, kwargs)) or object(),
    )
    for _ in range(2):
        module._make_worker_environment(
            "simple/Probe-v0",
            make_kwargs={"sim_mode": "mujoco"},
            third_person_video=enabled,
        )
    assert [env_id for env_id, _ in calls] == [
        "simple/Probe-v0",
        "simple/Probe-v0",
    ]
    if not enabled:
        assert all("extra_sensor_cfgs" not in kwargs for _, kwargs in calls)
        return
    first = calls[0][1]["extra_sensor_cfgs"]["third_person"]
    second = calls[1][1]["extra_sensor_cfgs"]["third_person"]
    assert first is not second
    assert first.uid == second.uid == "simple_eval_third_person_v1"
```

Add `import pytest` and the two CLI module imports used by the parametrized
test to the existing top-level import block. Do not import them between tests.
The production parent must assign `worker_kwargs = _build_worker_kwargs(...)`
before branching on `num_workers`; no branch may rebuild or mutate that
dictionary. Thus this test covers the exact object passed to the direct worker
call and as the `ctx.Process` worker argument.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_third_person_eval_camera.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit camera/config plumbing**

```bash
git add src/simple/evals/third_person_camera.py src/simple/evals/api.py \
  src/simple/evals/env_runner.py src/simple/cli/eval.py \
  src/simple/cli/eval_decoupled_wbc.py tests/test_third_person_eval_camera.py
git commit -m "feat: add opt-in third-person evaluation camera"
```

## Task 4: Implement backend adapters and asymmetric marker verification

**Files:**

- Modify: `src/simple/evals/third_person_camera.py`
- Modify: `src/simple/envs/base_dual_env.py`
- Modify: `src/simple/engines/isaacsim.py`
- Modify: `src/simple/engines/mujoco.py`
- Create: `tests/test_third_person_eval_backends.py`

- [ ] **Step 1: Write pure adapter and marker-analysis tests**

Create `tests/test_third_person_eval_backends.py` with imports that are safe in
the ordinary `.venv`. In particular, this file must not import
`simple.engines.isaacsim`; that module imports `carb` and may be loaded only
after `BaseDualSim._init_isaac` has created `SimulationApp`.

```python
from types import SimpleNamespace

import numpy as np

from simple.core.actor import CameraEntity
from simple.evals.third_person_camera import (
    analyze_verification_frame,
    apply_mujoco_clipping,
    isaac_camera_parent_path,
    make_third_person_camera,
    make_verification_markers,
)


def third_person_entity() -> CameraEntity:
    return CameraEntity("third_person", make_third_person_camera())


def test_canonical_world_pose_is_exact() -> None:
    camera = third_person_entity()
    np.testing.assert_allclose(
        camera.pose.position,
        [2.1650635094610964, 0.0, 1.2500000000000002],
        atol=1e-7,
    )


def test_pure_isaac_adapter_uses_workspace_parent() -> None:
    assert (
        isaac_camera_parent_path(
            "eye_in_world",
            workspace_prim_path="/World/workspace",
            robot_prim_path="/World/workspace/Robot/g1",
            wrist_prim_path="/World/workspace/Robot/g1/wrist",
            head_prim_path="/World/workspace/Robot/g1/head",
        )
        == "/World/workspace"
    )


def test_mujoco_clipping_is_converted_from_metres_to_extent_units() -> None:
    model = SimpleNamespace(
        stat=SimpleNamespace(extent=4.0),
        vis=SimpleNamespace(map=SimpleNamespace(znear=0.0, zfar=0.0)),
    )
    apply_mujoco_clipping(model, near=0.2, far=5.0)
    assert model.vis.map.znear * model.stat.extent == 0.2
    assert model.vis.map.zfar * model.stat.extent == 5.0


def test_asymmetric_marker_analyzer_checks_order_center_and_clipping() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[170:190, 230:250] = (255, 0, 0)
    frame[170:190, 310:330] = (0, 255, 0)
    frame[170:190, 390:410] = (0, 0, 255)
    report = analyze_verification_frame(frame)
    assert report["shape"] == [360, 640, 3]
    assert 230 <= report["marker_centers"]["left_red"][0] <= 250
    assert abs(report["marker_centers"]["center_green"][0] - 320) <= 12
    assert 390 <= report["marker_centers"]["right_blue"][0] <= 410
    assert report["marker_order_ok"] is True
    assert report["center_projection_ok"] is True
    assert report["near_magenta_pixels"] == 0
    assert report["far_cyan_pixels"] == 0
    assert report["clipping_ok"] is True


def test_marker_geometry_has_inside_near_and_far_sentinels() -> None:
    markers = {marker.name: marker for marker in make_verification_markers()}
    assert set(markers) == {
        "left_red",
        "center_green",
        "right_blue",
        "near_magenta",
        "far_cyan",
    }
    assert markers["near_magenta"].distance == 0.1
    assert markers["far_cyan"].distance == 5.5
    assert all(
        markers[name].distance == 1.2
        for name in ("left_red", "center_green", "right_blue")
    )
```

- [ ] **Step 2: Run pure tests and verify RED without importing Isaac**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_third_person_eval_backends.py
.venv/bin/python -c \
  'import sys; import simple.evals.third_person_camera; assert "carb" not in sys.modules; assert "simple.engines.isaacsim" not in sys.modules'
```

Expected: the test process imports neither `carb` nor the Isaac engine, and
the first command fails because the pure adapter/marker helpers do not exist.

- [ ] **Step 3: Implement the backend-neutral adapter and marker contract**

Append these complete definitions to `third_person_camera.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationMarker:
    name: str
    distance: float
    horizontal_offset: float
    rgb: tuple[float, float, float]


def make_verification_markers() -> tuple[VerificationMarker, ...]:
    return (
        VerificationMarker("left_red", 1.2, -0.30, (1.0, 0.0, 0.0)),
        VerificationMarker("center_green", 1.2, 0.0, (0.0, 1.0, 0.0)),
        VerificationMarker("right_blue", 1.2, 0.30, (0.0, 0.0, 1.0)),
        VerificationMarker("near_magenta", 0.1, 0.0, (1.0, 0.0, 1.0)),
        VerificationMarker("far_cyan", 5.5, 2.0, (0.0, 1.0, 1.0)),
    )


def isaac_camera_parent_path(
    mount: str,
    *,
    workspace_prim_path: str,
    robot_prim_path: str,
    wrist_prim_path: str,
    head_prim_path: str,
) -> str:
    parents = {
        "eye_in_world": workspace_prim_path,
        "eye_on_base": robot_prim_path,
        "eye_in_hand": wrist_prim_path,
        "eye_in_head": head_prim_path,
    }
    try:
        return parents[mount]
    except KeyError as cause:
        raise ValueError(f"Unsupported camera mount: {mount}") from cause


def apply_mujoco_clipping(model, *, near: float, far: float) -> None:
    extent = float(model.stat.extent)
    if not np.isfinite(extent) or extent <= 0.0:
        raise RuntimeError(f"invalid MuJoCo model extent: {extent}")
    model.vis.map.znear = near / extent
    model.vis.map.zfar = far / extent


def _dominant_mask(frame: np.ndarray, channel: int) -> np.ndarray:
    primary = frame[..., channel].astype(np.int16)
    others = [frame[..., index].astype(np.int16) for index in range(3) if index != channel]
    return (primary >= 160) & (primary - others[0] >= 60) & (primary - others[1] >= 60)


def _expected_pixel(marker: VerificationMarker) -> tuple[float, float]:
    fx = 320.0
    return 319.5 + fx * marker.horizontal_offset / marker.distance, 179.5


def _centroid(
    mask: np.ndarray,
    name: str,
    expected: tuple[float, float],
) -> list[float]:
    column_grid = np.arange(mask.shape[1])[None, :]
    row_grid = np.arange(mask.shape[0])[:, None]
    roi = (
        (np.abs(column_grid - expected[0]) <= 40.0)
        & (np.abs(row_grid - expected[1]) <= 40.0)
    )
    rows, columns = np.nonzero(mask & roi)
    if len(columns) < 16:
        raise AssertionError(f"{name} marker has only {len(columns)} pixels")
    return [float(columns.mean()), float(rows.mean())]


def analyze_verification_frame(frame: np.ndarray) -> dict:
    if frame.dtype != np.uint8 or frame.shape != (360, 640, 3):
        raise AssertionError(f"unexpected frame contract: {frame.shape} {frame.dtype}")
    red = _dominant_mask(frame, 0)
    green = _dominant_mask(frame, 1)
    blue = _dominant_mask(frame, 2)
    markers = {marker.name: marker for marker in make_verification_markers()}
    expected = {name: _expected_pixel(marker) for name, marker in markers.items()}
    centers = {
        "left_red": _centroid(red, "left_red", expected["left_red"]),
        "center_green": _centroid(green, "center_green", expected["center_green"]),
        "right_blue": _centroid(blue, "right_blue", expected["right_blue"]),
    }
    magenta = (frame[..., 0] >= 160) & (frame[..., 2] >= 160) & (frame[..., 1] <= 100)
    cyan = (frame[..., 1] >= 160) & (frame[..., 2] >= 160) & (frame[..., 0] <= 100)
    order_ok = centers["left_red"][0] < centers["center_green"][0] < centers["right_blue"][0]
    projection_errors = {
        name: float(np.linalg.norm(np.asarray(center) - np.asarray(expected[name])))
        for name, center in centers.items()
    }
    center_ok = projection_errors["center_green"] <= 32.0
    near_x, near_y = expected["near_magenta"]
    far_x, far_y = expected["far_cyan"]
    columns = np.arange(frame.shape[1])[None, :]
    rows = np.arange(frame.shape[0])[:, None]
    near_roi = (np.abs(columns - near_x) <= 40.0) & (np.abs(rows - near_y) <= 40.0)
    far_roi = (np.abs(columns - far_x) <= 40.0) & (np.abs(rows - far_y) <= 40.0)
    near_pixels = int((magenta & near_roi).sum())
    far_pixels = int((cyan & far_roi).sum())
    return {
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "marker_centers": centers,
        "expected_marker_centers": {
            name: [float(value) for value in pixel]
            for name, pixel in expected.items()
        },
        "marker_projection_errors": projection_errors,
        "marker_order_ok": order_ok,
        "center_projection_ok": center_ok,
        "near_magenta_pixels": near_pixels,
        "far_cyan_pixels": far_pixels,
        "clipping_ok": near_pixels <= 4 and far_pixels <= 4,
    }
```

Keep these helpers free of Isaac imports. Ruff may reflow the two long boolean
expressions; `ruff format` is authoritative.

- [ ] **Step 4: Add a verifier-only scene-marker hook before engine startup**

Add `_third_person_verification_markers: bool = False` as an explicit
keyword-only `BaseDualSim.__init__` argument. After the fresh task's isolated
camera mapping has been merged and validated, but before `_init_isaac` or
either engine constructor, call this helper only when the flag is true:

```python
def _inject_third_person_verification_markers(task) -> None:
    from simple.assets.primitive import Box
    from simple.evals.third_person_camera import make_verification_markers

    camera = task.layout.cameras["third_person"]
    camera_position = np.asarray(camera.pose.position, dtype=np.float64)
    forward = -camera_position / np.linalg.norm(camera_position)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    for marker in make_verification_markers():
        position = camera_position + marker.distance * forward
        position = position + marker.horizontal_offset * right
        box = Box(
            size=[0.10, 0.10, 0.10],
            position=position.tolist(),
            quaternion=[1.0, 0.0, 0.0, 0.0],
        )
        box.material = {
            "verification_rgb": marker.rgb,
            "collision": False,
        }
        task.layout.actors[f"__camera_verify_{marker.name}"] = box
```

Add `import numpy as np` to `base_dual_env.py`. Reject the hook with
`ValueError` unless the isolated `third_person` camera is present. The hook is
private, absent from both evaluation CLIs, and used only by the backend
verification script in Task 10.

- [ ] **Step 5: Implement MuJoCo camera, clipping, and visual-only markers**

Import `apply_mujoco_clipping` from the neutral module. Add the
`eye_in_world` `_build_camera` branch before `eye_in_hand`:

```text
elif camera.mount == "eye_in_world":
    cam_q = t3d.quaternions.qmult(camera.pose.quaternion, q_isaac_mujoco)
    self._add_mujoco_camera(
        self.mj_worldbody,
        camera,
        name=cname,
        pos=camera.pose.position,
        quat=cam_q,
    )
```

Immediately after `mjSpec.compile()`, call `apply_mujoco_clipping` with the
`third_person` camera's exact near/far values before constructing a renderer.
In `_build_primitive`, keep the current table behavior and use this exact
marker-only branch when `verification_rgb` is present:

```python
marker_rgb = getattr(actor, "material", {}).get("verification_rgb")
geom_kwargs = {}
if marker_rgb is not None:
    geom_kwargs.update(
        rgba=[*marker_rgb, 1.0],
        contype=0,
        conaffinity=0,
    )
table.add_geom(
    name=f"{table_name}_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=table_size,
    condim=6,
    friction=[2, 0.04, 0.0005],
    priority=10,
    **geom_kwargs,
)
```

- [ ] **Step 6: Implement Isaac adapter use, clipping, and visual-only markers**

Use `isaac_camera_parent_path` in `add_cameras`; construct all four candidate
paths first, then select by mount. Replace the hard-coded clipping call with:

```python
isaacsim_camera.set_clipping_range(
    cameraEntity.cam_cfg.near,
    cameraEntity.cam_cfg.far,
)
```

After the existing table setup in `__update_scene`, create every verification
primitive with this complete helper. It deliberately applies no collision
API:

```python
def _setup_verification_marker(self, prim_path: str, marker_box) -> None:
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)
    omni.kit.commands.execute(
        "CreateMeshPrimWithDefaultXform",
        prim_type="Cube",
        prim_path=prim_path,
    )
    prim = stage.GetPrimAtPath(prim_path)
    position = [float(value) for value in marker_box.pose.position]
    size = [float(value) for value in marker_box.size]
    rgb = [float(value) for value in marker_box.material["verification_rgb"]]
    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*position))
    prim.GetAttribute("xformOp:scale").Set(Gf.Vec3d(*size))
    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
```

The call site is:

```python
for actor_name, actor in self.task.layout.actors.items():
    if actor_name.startswith("__camera_verify_"):
        self._setup_verification_marker(
            f"{self.workspace_prim_path}/{actor_name}", actor
        )
```

No test in the ordinary `.venv` imports this module. Task 10 performs the
initialized Isaac integration and reads the camera's effective clipping range.

- [ ] **Step 7: Add MuJoCo construction and mixed-output unit coverage**

Append these imports and tests to the existing backend test file:

```python
from simple.engines.mujoco import MujocoSimulator
from simple.envs.sonic_loco_manip import SonicLocoManipEnv


def test_mujoco_eye_in_world_attaches_to_worldbody() -> None:
    simulator = MujocoSimulator.__new__(MujocoSimulator)
    worldbody = object()
    calls = []
    simulator.mj_worldbody = worldbody
    simulator._add_mujoco_camera = (
        lambda parent, camera, **kwargs: calls.append((parent, camera, kwargs))
    )
    camera = third_person_entity()
    simulator._build_camera("third_person", camera)
    parent, got_camera, kwargs = calls[0]
    assert parent is worldbody
    assert got_camera is camera
    np.testing.assert_allclose(kwargs["pos"], camera.pose.position, atol=1e-7)


def test_mixed_mode_selects_isaac_and_mujoco_only_selects_mujoco() -> None:
    mujoco_frame = np.full((2, 3, 3), 11, dtype=np.uint8)
    isaac_frame = np.full((2, 3, 3), 22, dtype=np.uint8)
    env = SonicLocoManipEnv.__new__(SonicLocoManipEnv)
    env.task = SimpleNamespace(metadata={})
    env.mujoco = SimpleNamespace(render=lambda: {"third_person": mujoco_frame})
    env.isaac = SimpleNamespace(render=lambda: {"third_person": isaac_frame})
    np.testing.assert_array_equal(env._render_frame()["third_person"], isaac_frame)
    env.isaac = None
    np.testing.assert_array_equal(env._render_frame()["third_person"], mujoco_frame)
```

Run the Task 4 Step 2 commands again. Expected: all tests pass, and the import
sentinel proves the pure gate is independent of Isaac runtime packages.

- [ ] **Step 8: Commit backend semantics**

```bash
git add src/simple/evals/third_person_camera.py src/simple/envs/base_dual_env.py \
  src/simple/engines/isaacsim.py src/simple/engines/mujoco.py \
  tests/test_third_person_eval_backends.py
git commit -m "feat: verify third-person camera backend semantics"
```

## Task 5: Prove both PSI0 request payloads are byte-identical

**Files:**

- Create: `tests/test_third_person_policy_isolation.py`

- [ ] **Step 1: Add the literal request capture test**

Create `tests/test_third_person_policy_isolation.py` with the complete literal
capture fixture. A zero-length action chunk forces both agents to stop before
action mapping, robot access, or WBC work:

```python
from collections import deque
from copy import deepcopy
import json

import numpy as np
import pytest

from simple.baselines.client import (
    HttpActionClient,
    RequestMessage,
    ResponseMessage,
)
from simple.baselines.psi0 import Psi0Agent
from simple.baselines.psi0_decoupled_wbc import Psi0DecoupledWbcAgent


FROZEN_TIMESTAMP = "2026-08-18_00-00-00.000000"


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return deepcopy(self._payload)

    def raise_for_status(self):
        return None


class RecordingSession:
    def __init__(self):
        self.calls = []
        self.response = FakeResponse(
            ResponseMessage(
                action=np.zeros((0, 36), dtype=np.float32),
                err=0.0,
            ).serialize()
        )

    def post(self, url, *, json, timeout):
        self.calls.append(
            {"url": url, "json": deepcopy(json), "timeout": timeout}
        )
        return self.response


def canonical_bytes(payload: dict) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def make_agent(agent_type, session):
    agent = agent_type.__new__(agent_type)
    agent._action_queue = deque()
    agent._reset_history = True
    agent._global_step_idx = 0
    agent.upsample_factor = 1
    agent.client = HttpActionClient(
        "policy", 22085, timeout=5.0, session=session
    )
    if agent_type is Psi0Agent:
        agent._session_id = "psi0-fixed-session"
        agent._last_cmd_torso_rpyh = np.array(
            [0.0, 0.0, 0.0, 0.75], dtype=np.float32
        )
    else:
        agent._last_cmd_torso_rpyh = np.array(
            [0.0, 0.0, 0.0, 0.74], dtype=np.float32
        )
    return agent


def capture_request(agent_type, observation):
    session = RecordingSession()
    agent = make_agent(agent_type, session)
    with pytest.raises(StopIteration, match="No more queued actions"):
        agent.get_action(
            observation,
            instruction="pick up the object",
            info={"episode_index": 7},
        )
    assert len(session.calls) == 1
    assert session.calls[0]["url"] == "http://policy:22085/act"
    assert session.calls[0]["timeout"] == 5.0
    return session.calls[0]["json"]


@pytest.mark.parametrize(
    "agent_type", [Psi0Agent, Psi0DecoupledWbcAgent]
)
def test_third_person_observation_does_not_change_policy_request(
    monkeypatch, agent_type
):
    monkeypatch.setattr(
        HttpActionClient,
        "timestamp",
        property(lambda self: FROZEN_TIMESTAMP),
    )
    joint_qpos = np.arange(43, dtype=np.float32) / 100.0
    head = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    ordinary_observation = {
        "joint_qpos": joint_qpos.copy(),
        "head_stereo_left": head.copy(),
    }
    augmented_observation = {
        **ordinary_observation,
        "third_person": np.full((360, 640, 3), 173, dtype=np.uint8),
    }

    ordinary = capture_request(agent_type, ordinary_observation)
    augmented = capture_request(agent_type, augmented_observation)
    assert canonical_bytes(ordinary) == canonical_bytes(augmented)

    request = RequestMessage.deserialize(augmented)
    assert set(request.image) == {"rgb_head_stereo_left"}
    np.testing.assert_array_equal(request.image["rgb_head_stereo_left"], head)
    assert set(request.state) == {"states"}
    assert request.state["states"].shape == (1, 32)
    assert request.state["states"].dtype == np.float32
    assert request.instruction == "pick up the object"
    assert request.condition == {}
    assert request.gt_action == []
    assert request.dataset_name == "simple"
    assert request.timestamp == FROZEN_TIMESTAMP
    if agent_type is Psi0Agent:
        assert request.history == {
            "reset": True,
            "session_id": "psi0-fixed-session",
            "episode_index": 7,
            "step_index": 0,
        }
    else:
        assert request.history == {"reset": True}
```

- [ ] **Step 2: Run the isolation tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_third_person_policy_isolation.py
```

Expected: both agent parameter cases pass without a server, GPU, simulator, or WBC construction.

- [ ] **Step 3: Commit the characterization contract**

```bash
git add tests/test_third_person_policy_isolation.py
git commit -m "test: lock PSI0 third-person input isolation"
```

## Task 6: Replace shell-based video finalization with checked atomic output

**Files:**

- Modify: `src/simple/envs/video_writer.py`
- Create: `tests/test_eval_video_finalization.py`

Rerun ownership is explicit. Evaluator resets carry `task_id`, so
`VideoRecorder.reset` removes the owned `episode_<N>` directory before opening
new writers. Direct `VideoWriter` reuse additionally rotates any canonical
prior verdict to `third_person_previous_<verdict>.mp4`. A successful new
verdict removes that backup and leaves exactly one canonical artifact. A
failed transcode leaves the new raw MP4 plus the non-canonical previous backup,
and leaves neither canonical success nor canonical failure. Temporary files
are never retained.

- [ ] **Step 1: Write failing success, failure, timeout, and fallback tests**

Create `tests/test_eval_video_finalization.py` with these literal writer cases.
The rerun contract retains at most one immediately previous artifact under a
non-canonical `previous_*` name while the new run is in progress. A completed
new verdict removes those backups; a failed transcode preserves both the new
raw file and the prior backup, but never leaves an old canonical verdict:

```python
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from simple.envs import video_writer as video_writer_module
from simple.envs.video_writer import VideoFinalizationError, VideoWriter


class FakeCvWriter:
    def __init__(self):
        self.release_count = 0

    def release(self):
        self.release_count += 1


@pytest.fixture
def writer_factory(monkeypatch, tmp_path):
    created = []

    def make_cv_writer(*args, **kwargs):
        writer = FakeCvWriter()
        created.append(writer)
        return writer

    monkeypatch.setattr(
        video_writer_module.cv2, "VideoWriter", make_cv_writer
    )
    monkeypatch.setattr(
        video_writer_module.cv2, "VideoWriter_fourcc", lambda *args: 0
    )

    def make():
        raw_path = tmp_path / "episode" / "third_person.mp4"
        writer = VideoWriter(str(raw_path), 50, (640, 360))
        raw_path.write_bytes(b"raw-video")
        return writer, raw_path, created[-1]

    return make


def test_writer_success_is_atomic_and_idempotent(
    monkeypatch, writer_factory
):
    writer, raw_path, cv_writer = writer_factory()
    calls = []
    monkeypatch.setattr(video_writer_module.shutil, "which", lambda _: "/ffmpeg")

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        assert "-nostdin" in argv
        Path(argv[-1]).write_bytes(b"h264-video")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(video_writer_module.subprocess, "run", run)
    final_path = Path(writer.release(success=True))
    assert final_path.name == "third_person_success.mp4"
    assert final_path.read_bytes() == b"h264-video"
    assert not raw_path.exists()
    assert cv_writer.release_count == 1
    assert Path(writer.release(success=True)) == final_path
    assert len(calls) == 1


@pytest.mark.parametrize("failure_kind", ["nonzero", "timeout"])
def test_writer_failure_preserves_raw(
    monkeypatch, writer_factory, failure_kind
):
    writer, raw_path, cv_writer = writer_factory()
    monkeypatch.setattr(video_writer_module.shutil, "which", lambda _: "/ffmpeg")

    def run(argv, **kwargs):
        Path(argv[-1]).write_bytes(b"partial")
        if failure_kind == "nonzero":
            raise subprocess.CalledProcessError(
                1, argv, stderr=b"codec failed"
            )
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(video_writer_module.subprocess, "run", run)
    with pytest.raises(VideoFinalizationError) as raised:
        writer.release(success=False)
    assert raised.value.raw_path == str(raw_path)
    assert raw_path.read_bytes() == b"raw-video"
    assert not raw_path.with_name("third_person_failed.mp4").exists()
    assert not raw_path.with_name("third_person_failed.tmp.mp4").exists()
    assert cv_writer.release_count == 1
    with pytest.raises(VideoFinalizationError) as repeated:
        writer.release(success=False)
    assert repeated.value is raised.value


def test_writer_without_ffmpeg_atomically_keeps_verdict_name(
    monkeypatch, writer_factory
):
    writer, raw_path, _ = writer_factory()
    monkeypatch.setattr(video_writer_module.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        video_writer_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("subprocess should not run")
        ),
    )
    final_path = Path(writer.release(success=False))
    assert final_path.name == "third_person_failed.mp4"
    assert final_path.read_bytes() == b"raw-video"
    assert not raw_path.exists()


@pytest.mark.parametrize(
    ("old_verdict", "new_success"),
    [("success", False), ("failed", True)],
)
def test_opposite_verdict_rerun_has_one_canonical_artifact(
    monkeypatch, tmp_path, old_verdict, new_success
):
    monkeypatch.setattr(video_writer_module.cv2, "VideoWriter", lambda *args: FakeCvWriter())
    monkeypatch.setattr(video_writer_module.cv2, "VideoWriter_fourcc", lambda *args: 0)
    monkeypatch.setattr(video_writer_module.shutil, "which", lambda _: None)
    episode = tmp_path / "episode"
    episode.mkdir()
    raw_path = episode / "third_person.mp4"
    old_path = episode / f"third_person_{old_verdict}.mp4"
    old_path.write_bytes(b"old-verdict")
    writer = VideoWriter(str(raw_path), 50, (640, 360))
    assert not old_path.exists()
    assert (episode / f"third_person_previous_{old_verdict}.mp4").exists()
    raw_path.write_bytes(b"new-video")
    final_path = Path(writer.release(success=new_success))
    expected = "success" if new_success else "failed"
    assert final_path.name == f"third_person_{expected}.mp4"
    assert sorted(path.name for path in episode.glob("third_person_*.mp4")) == [
        f"third_person_{expected}.mp4"
    ]


def test_failed_transcode_preserves_raw_and_prior_backup_without_stale_verdict(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(video_writer_module.cv2, "VideoWriter", lambda *args: FakeCvWriter())
    monkeypatch.setattr(video_writer_module.cv2, "VideoWriter_fourcc", lambda *args: 0)
    monkeypatch.setattr(video_writer_module.shutil, "which", lambda _: "/ffmpeg")
    episode = tmp_path / "episode"
    episode.mkdir()
    raw_path = episode / "third_person.mp4"
    old_success = episode / "third_person_success.mp4"
    old_success.write_bytes(b"old-success")
    writer = VideoWriter(str(raw_path), 50, (640, 360))
    raw_path.write_bytes(b"new-raw")

    def fail(argv, **kwargs):
        Path(argv[-1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, argv, stderr=b"codec failed")

    monkeypatch.setattr(video_writer_module.subprocess, "run", fail)
    with pytest.raises(VideoFinalizationError):
        writer.release(success=False)
    assert raw_path.read_bytes() == b"new-raw"
    assert (episode / "third_person_previous_success.mp4").read_bytes() == b"old-success"
    assert not (episode / "third_person_success.mp4").exists()
    assert not (episode / "third_person_failed.mp4").exists()
    assert not (episode / "third_person_failed.tmp.mp4").exists()
```

- [ ] **Step 2: Run finalization tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_eval_video_finalization.py -k 'writer'
```

Expected: failures because the current code uses `os.system` and deletes raw output unchecked.

- [ ] **Step 3: Implement `VideoFinalizationError` and atomic release**

Replace `VideoWriter.release` with an implementation using these exact states:

```python
class VideoFinalizationError(RuntimeError):
    def __init__(self, raw_path: str, reason: str):
        self.raw_path = raw_path
        self.reason = reason
        super().__init__(f"video finalization failed; raw preserved at {raw_path}: {reason}")
```

Initialize `self._raw_closed = False`, `self._finalized_path = None`, and
`self._finalization_error = None` in `__init__`. Before opening the OpenCV
writer, rotate both canonical verdict names with this exact helper so an
interrupted rerun cannot expose a stale verdict as current:

```python
def _rotate_previous_verdicts(filename: str) -> tuple[str, ...]:
    stem = filename[:-4]
    backups = []
    for verdict in ("success", "failed"):
        canonical = f"{stem}_{verdict}.mp4"
        backup = f"{stem}_previous_{verdict}.mp4"
        if os.path.exists(backup):
            os.unlink(backup)
        if os.path.exists(canonical):
            os.replace(canonical, backup)
            backups.append(backup)
    for verdict in ("success", "failed"):
        temporary = f"{stem}_{verdict}.tmp.mp4"
        if os.path.exists(temporary):
            os.unlink(temporary)
    return tuple(backups)
```

Set `self._previous_verdict_paths = _rotate_previous_verdicts(filename)`
before constructing `cv2.VideoWriter`. Add this helper method:

```python
def _discard_previous_verdicts(self) -> None:
    for path in self._previous_verdict_paths:
        if os.path.exists(path):
            os.unlink(path)
    self._previous_verdict_paths = ()
```

Implement release as:

```python
def release(self, success=True, *, deadline=None):
    if self._finalized_path is not None:
        return self._finalized_path
    if self._finalization_error is not None:
        raise self._finalization_error
    if not self._raw_closed:
        self.video_writer.release()
        self._raw_closed = True

    suffix = "success" if success else "failed"
    final_path = f"{self.filename[:-4]}_{suffix}.mp4"
    temporary_path = f"{self.filename[:-4]}_{suffix}.tmp.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        os.replace(self.filename, final_path)
        self._discard_previous_verdicts()
        self._finalized_path = final_path
        return final_path

    remaining = 60.0 if deadline is None else deadline - time.monotonic()
    if remaining <= 0.0:
        error = VideoFinalizationError(self.filename, "deadline expired")
        self._finalization_error = error
        raise error
    argv = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-i",
        self.filename,
        "-vcodec",
        "libx264",
        temporary_path,
    ]
    try:
        subprocess.run(
            argv,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=remaining,
        )
        if not os.path.isfile(temporary_path) or os.path.getsize(temporary_path) == 0:
            raise RuntimeError("ffmpeg produced no output")
        os.replace(temporary_path, final_path)
        os.unlink(self.filename)
        self._discard_previous_verdicts()
    except Exception as cause:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        reason = getattr(cause, "stderr", None) or str(cause)
        if isinstance(reason, bytes):
            reason = reason.decode("utf-8", "replace")
        error = VideoFinalizationError(self.filename, str(reason))
        self._finalization_error = error
        raise error from cause
    self._finalized_path = final_path
    return final_path
```

Add imports for `subprocess` and `time`; remove unused shell-finalization imports.

- [ ] **Step 4: Run writer tests and verify GREEN**

Run the Task 6 Step 2 command.

Expected: all writer cases pass.

- [ ] **Step 5: Commit atomic finalization**

```bash
git add src/simple/envs/video_writer.py tests/test_eval_video_finalization.py
git commit -m "fix: bound and preserve evaluation video finalization"
```

## Task 7: Give `VideoRecorder` one shared deadline and aggregate all failures

**Files:**

- Modify: `src/simple/envs/wrappers/video_recorder.py`
- Modify: `tests/test_eval_video_finalization.py`

- [ ] **Step 1: Write failing shared-deadline tests**

Add these imports to the existing import block at the top of
`tests/test_eval_video_finalization.py`:

```python
from simple.envs.wrappers import video_recorder as video_recorder_module
from simple.envs.wrappers.video_recorder import (
    VideoRecorder,
    VideoRecorderFinalizationError,
)
```

Then append the complete recorder cases:

```python


class FakeReleaseWriter:
    def __init__(self, name, events, error=None):
        self.name = name
        self.events = events
        self.error = error

    def release(self, success, *, deadline):
        self.events.append((self.name, success, deadline))
        if self.error is not None:
            raise self.error


class FakeWrappedEnv:
    def __init__(self, success, close_error=None):
        self.unwrapped = SimpleNamespace(_success=success)
        self.close_count = 0
        self.close_error = close_error

    def close(self):
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error


def make_recorder(*, success=True):
    recorder = VideoRecorder.__new__(VideoRecorder)
    recorder.env = FakeWrappedEnv(success)
    recorder.video_writers = {}
    recorder._is_released = False
    recorder._release_error = None
    return recorder


def test_recorder_uses_one_deadline_and_is_idempotent(monkeypatch):
    monkeypatch.setattr(video_recorder_module.time, "monotonic", lambda: 100.0)
    events = []
    recorder = make_recorder(success=True)
    recorder.video_writers = {
        name: FakeReleaseWriter(name, events)
        for name in ("head", "third_person", "wrist")
    }
    recorder.release(timeout_seconds=60.0)
    assert events == [
        ("head", True, 160.0),
        ("third_person", True, 160.0),
        ("wrist", True, 160.0),
    ]
    recorder.release(timeout_seconds=60.0)
    assert len(events) == 3


def test_recorder_aggregates_failures_without_skipping_writers(monkeypatch):
    monkeypatch.setattr(video_recorder_module.time, "monotonic", lambda: 5.0)
    events = []
    recorder = make_recorder(success=False)
    recorder.video_writers = {
        "head": FakeReleaseWriter("head", events, RuntimeError("head failed")),
        "third_person": FakeReleaseWriter("third_person", events),
        "wrist": FakeReleaseWriter("wrist", events, RuntimeError("wrist failed")),
    }
    with pytest.raises(VideoRecorderFinalizationError) as raised:
        recorder.release(timeout_seconds=60.0)
    assert [name for name, _ in raised.value.failures] == ["head", "wrist"]
    assert [event[0] for event in events] == ["head", "third_person", "wrist"]
    assert {event[2] for event in events} == {65.0}


def test_recorder_close_always_closes_wrapped_env(monkeypatch):
    monkeypatch.setattr(video_recorder_module.time, "monotonic", lambda: 5.0)
    events = []
    recorder = make_recorder(success=False)
    recorder.video_writers = {
        "head": FakeReleaseWriter("head", events, RuntimeError("finalize"))
    }
    with pytest.raises(VideoRecorderFinalizationError, match="finalize"):
        recorder.close()
    assert recorder.env.close_count == 1


def test_recorder_close_keeps_finalization_failure_primary(monkeypatch, capsys):
    monkeypatch.setattr(video_recorder_module.time, "monotonic", lambda: 5.0)
    events = []
    recorder = make_recorder(success=False)
    recorder.env.close_error = RuntimeError("wrapped close")
    recorder.video_writers = {
        "head": FakeReleaseWriter("head", events, RuntimeError("finalize"))
    }
    with pytest.raises(VideoRecorderFinalizationError, match="finalize"):
        recorder.close()
    assert "wrapped close" in capsys.readouterr().err
```

- [ ] **Step 2: Run recorder tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_eval_video_finalization.py -k 'recorder'
```

Expected: failures because current recorder release has no deadline, explicit verdict, or aggregation.

- [ ] **Step 3: Implement recorder aggregation**

Add:

```python
class VideoRecorderFinalizationError(RuntimeError):
    def __init__(self, failures):
        self.failures = tuple(failures)
        detail = "; ".join(f"{name}: {error}" for name, error in self.failures)
        super().__init__(f"video recorder finalization failed: {detail}")
```

Initialize `self._release_error = None`. Replace release with:

```python
def release(self, success=None, *, timeout_seconds=60.0):
    if self._release_error is not None:
        raise self._release_error
    if self._is_released:
        return
    verdict = bool(getattr(self.unwrapped, "_success", False)) if success is None else bool(success)
    deadline = time.monotonic() + timeout_seconds
    failures = []
    for name, video_writer in self.video_writers.items():
        try:
            video_writer.release(verdict, deadline=deadline)
        except Exception as error:
            failures.append((name, error))
    self._is_released = True
    if failures:
        self._release_error = VideoRecorderFinalizationError(failures)
        raise self._release_error
```

Add `import time`. Make `close` call `release()` inside `try` and always call
`super().close()`, preserving the finalization error as the primary error with
this exact implementation:

```python
def close(self):
    release_error = None
    close_error = None
    try:
        self.release()
    except BaseException as error:
        release_error = error
    try:
        super().close()
    except BaseException as error:
        close_error = error
    if release_error is not None:
        if close_error is not None:
            print(f"wrapped environment cleanup failed: {close_error}", file=sys.stderr)
        raise release_error
    if close_error is not None:
        raise close_error
```

Add `sys` to the module imports.

- [ ] **Step 4: Run all video tests and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_eval_video_finalization.py
git add src/simple/envs/wrappers/video_recorder.py tests/test_eval_video_finalization.py
git commit -m "fix: share evaluation recorder finalization deadline"
```

Expected: all video tests pass before commit.

## Task 8: Add reusable episode/resource/worker lifecycle ownership

**Files:**

- Create: `src/simple/evals/lifecycle.py`
- Create: `tests/test_eval_worker_cleanup.py`

- [ ] **Step 1: Write failing lifecycle unit tests**

Create `tests/test_eval_worker_cleanup.py` with these executable lifecycle
fakes and assertions:

```python
import pytest

from simple.evals import lifecycle
from simple.evals.lifecycle import (
    CleanupError,
    EpisodeVideo,
    ResourceOwner,
    construct_worker_stack,
    monitor_parent_workers,
    stop_worker_processes,
)


class Closeable:
    def __init__(self, name, events, error=None):
        self.name = name
        self.events = events
        self.error = error

    def close(self):
        self.events.append(self.name)
        if self.error is not None:
            raise self.error


class FakeRecorder:
    def __init__(self):
        self.calls = []

    def release(self, *, success):
        self.calls.append(success)


def test_resource_owner_closes_once_in_reverse_order():
    events = []
    raw = Closeable("raw", events)
    rollout = Closeable("rollout", events)
    agent = Closeable("agent", events)
    with ResourceOwner() as owner:
        owner.own("raw", raw)
        owner.own("rollout", rollout)
        owner.own("duplicate_rollout", rollout)
        owner.own("agent", agent)
    assert events == ["agent", "rollout", "raw"]


def test_resource_owner_closes_wrapper_then_raw_fallback():
    events = []
    raw = Closeable("raw", events)
    wrapper = Closeable("wrapper", events)
    agent = Closeable("agent", events)
    with ResourceOwner() as owner:
        owner.own("environment", raw)
        owner.own("agent", agent)
        owner.own_before("wrapper", wrapper, before_name="agent")
    assert events == ["agent", "wrapper", "raw"]


def test_resource_owner_disown_keeps_runner_owned_environment_open():
    events = []
    raw = Closeable("raw", events)
    with ResourceOwner() as owner:
        owner.own("environment", raw)
        assert owner.disown("environment", raw) is raw
    assert events == []


def test_resource_owner_aggregates_without_skipping():
    events = []
    with pytest.raises(CleanupError) as raised:
        with ResourceOwner() as owner:
            owner.own("raw", Closeable("raw", events, RuntimeError("raw")))
            owner.own(
                "rollout",
                Closeable("rollout", events, RuntimeError("rollout")),
            )
            owner.own("agent", Closeable("agent", events, RuntimeError("agent")))
    assert events == ["agent", "rollout", "raw"]
    assert [name for name, _ in raised.value.failures] == [
        "agent",
        "rollout",
        "raw",
    ]


def test_rollout_exception_remains_primary(capsys):
    with pytest.raises(ValueError, match="rollout"):
        with ResourceOwner() as owner:
            owner.own(
                "raw", Closeable("raw", [], RuntimeError("cleanup failed"))
            )
            raise ValueError("rollout")
    assert "cleanup failed" in capsys.readouterr().err


@pytest.mark.parametrize("verdict", [False, True])
def test_episode_video_uses_explicit_verdict(verdict):
    recorder = FakeRecorder()
    with EpisodeVideo(recorder) as video:
        assert video.env is recorder
        video.finish(verdict)
    assert recorder.calls == [verdict]


def test_episode_video_exception_finalizes_failed_once():
    recorder = FakeRecorder()
    with pytest.raises(RuntimeError, match="episode"):
        with EpisodeVideo(recorder):
            raise RuntimeError("episode")
    assert recorder.calls == [False]


@pytest.mark.parametrize(
    ("worker_kind", "wrapper_names"),
    [
        ("ordinary", ("stand", "timelimit")),
        ("decoupled", ("timelimit",)),
    ],
)
def test_construct_worker_stack_rolls_back_every_completed_stage(
    worker_kind, wrapper_names
):
    construction_order = ("environment", *wrapper_names, "agent", "rollout")
    for failure_stage in (*construction_order, "after_rollout"):
        events = []

        def make(name):
            events.append(f"make:{name}")
            if failure_stage == name:
                raise RuntimeError(f"{name} failed")
            return Closeable(name, events)

        with pytest.raises(RuntimeError, match="failed"):
            with ResourceOwner() as owner:
                construct_worker_stack(
                    owner,
                    environment_factory=lambda: make("environment"),
                    wrapper_factories=tuple(
                        (name, lambda env, name=name: make(name))
                        for name in wrapper_names
                    ),
                    agent_factory=lambda env: make("agent"),
                    rollout_factory=lambda env, agent: make("rollout"),
                )
                if failure_stage == "after_rollout":
                    raise RuntimeError("after_rollout failed")
        completed = [
            name
            for name in construction_order
            if f"make:{name}" in events and name != failure_stage
        ]
        ownership_order = [
            name for name in completed if name not in {"agent", "rollout"}
        ]
        if "rollout" in completed:
            ownership_order.append("rollout")
        if "agent" in completed:
            ownership_order.append("agent")
        closed = [event for event in events if not event.startswith("make:")]
        assert closed == list(reversed(ownership_order)), (
            worker_kind,
            failure_stage,
            events,
        )


class FakeProcess:
    def __init__(self, name, pid, events, *, terminate_error=None):
        self.name = name
        self.pid = pid
        self._alive = True
        self.events = events
        self.terminate_error = terminate_error

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.events.append((self.name, "terminate"))
        if self.terminate_error is not None:
            raise self.terminate_error
        self._alive = False

    def join(self, timeout):
        self.events.append((self.name, "join", timeout))

    def kill(self):
        self.events.append((self.name, "kill"))
        self._alive = False


def test_stop_workers_has_one_deadline_and_never_skips_cleanup(monkeypatch):
    events = []
    first = FakeProcess("first", 101, events)
    second = FakeProcess("second", 102, events)
    third = FakeProcess(
        "third", 103, events, terminate_error=RuntimeError("terminate")
    )
    processes = [first, second, third]

    def kill(pid, signal_number):
        events.append((pid, "signal", signal_number))
        if pid == first.pid:
            first._alive = False

    clock_values = iter([10.0, 76.0, 76.0, 82.0, 82.0, 82.0, 82.0])
    monkeypatch.setattr(lifecycle.os, "kill", kill)
    failures = stop_worker_processes(
        processes,
        grace_seconds=65.0,
        clock=lambda: next(clock_values),
        sleep=lambda _: (_ for _ in ()).throw(
            AssertionError("shared deadline should already be expired")
        ),
    )
    assert [name for name, _ in failures] == ["third"]
    assert all(any(event[:2] == (proc.name, "join") for event in events) for proc in processes)
    assert ("third", "kill") in events
    assert all(not proc.is_alive() for proc in processes)


class FakeConnection:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def close(self):
        self.events.append((self.name, "close"))


def test_parent_keyboard_interrupt_always_stops_joins_closes_and_restores(monkeypatch):
    events = []
    processes = [FakeProcess(f"worker-{index}", 200 + index, events) for index in range(3)]
    connections = [FakeConnection(f"pipe-{index}", events) for index in range(3)]

    def interrupt():
        raise KeyboardInterrupt

    def stop(items, **kwargs):
        assert items == processes
        events.append(("workers", "stopped"))
        for process in items:
            process._alive = False
            process.join(timeout=0.0)
        return []

    monkeypatch.setattr(lifecycle, "stop_worker_processes", stop)
    with pytest.raises(KeyboardInterrupt):
        monitor_parent_workers(
            interrupt,
            processes=processes,
            connections=connections,
            restore_terminal=lambda: events.append(("terminal", "restored")),
        )
    assert events.count(("workers", "stopped")) == 1
    assert [(name, action) for name, action in events if name.startswith("pipe-")] == [
        ("pipe-0", "close"),
        ("pipe-1", "close"),
        ("pipe-2", "close"),
    ]
    assert ("terminal", "restored") in events
    assert all(not process.is_alive() for process in processes)
```

- [ ] **Step 2: Run lifecycle tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_eval_worker_cleanup.py
```

Expected: import failure for `simple.evals.lifecycle`.

- [ ] **Step 3: Implement the lifecycle module**

```python
from __future__ import annotations

import os
import signal
import sys
import time


class CleanupError(RuntimeError):
    def __init__(self, failures):
        self.failures = tuple(failures)
        detail = "; ".join(f"{name}: {error}" for name, error in self.failures)
        super().__init__(f"cleanup failed: {detail}")


class ResourceOwner:
    def __init__(self):
        self._resources = []
        self._closed = False

    def own(self, name, resource):
        self._resources.append((name, resource))
        return resource

    def own_before(self, name, resource, *, before_name):
        for index, (owned_name, _) in enumerate(self._resources):
            if owned_name == before_name:
                self._resources.insert(index, (name, resource))
                return resource
        raise KeyError(f"resource ownership not found: {before_name}")

    def disown(self, name, resource):
        for index, (owned_name, owned_resource) in enumerate(self._resources):
            if owned_name == name and owned_resource is resource:
                self._resources.pop(index)
                return resource
        raise KeyError(f"resource ownership not found: {name}")

    def close(self):
        if self._closed:
            return []
        self._closed = True
        failures = []
        seen = set()
        for name, resource in reversed(self._resources):
            identity = id(resource)
            if identity in seen:
                continue
            seen.add(identity)
            close = getattr(resource, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as error:
                failures.append((name, error))
        return failures

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        failures = self.close()
        if not failures:
            return False
        cleanup_error = CleanupError(failures)
        if exc is None:
            raise cleanup_error
        print(cleanup_error, file=sys.stderr)
        return False


class EpisodeVideo:
    def __init__(self, recorder):
        self.env = recorder
        self._finished = False
        self._success = False

    def finish(self, success: bool):
        self._finished = True
        self._success = bool(success)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        success = self._finished and self._success and exc is None
        try:
            self.env.release(success=success)
        except Exception as error:
            if exc is None:
                raise
            print(f"video cleanup failed: {error}", file=sys.stderr)
        return False


def construct_worker_stack(
    owner,
    *,
    environment_factory,
    wrapper_factories,
    agent_factory,
    rollout_factory=None,
):
    environment = owner.own("environment", environment_factory())
    for name, factory in wrapper_factories:
        environment = owner.own(name, factory(environment))
    agent = owner.own("agent", agent_factory(environment))
    if rollout_factory is not None:
        environment = owner.own_before(
            "rollout",
            rollout_factory(environment, agent),
            before_name="agent",
        )
    return environment, agent


def stop_worker_processes(
    processes,
    *,
    grace_seconds=65.0,
    clock=time.monotonic,
    sleep=time.sleep,
):
    failures = []

    def is_alive(process):
        try:
            return bool(process.is_alive())
        except Exception as error:
            failures.append((process.name, error))
            return True

    for process in processes:
        if is_alive(process) and process.pid is not None:
            try:
                os.kill(process.pid, signal.SIGINT)
            except Exception as error:
                failures.append((process.name, error))
    deadline = clock() + grace_seconds
    while any(is_alive(process) for process in processes) and clock() < deadline:
        sleep(min(0.05, max(0.0, deadline - clock())))
    for process in processes:
        if is_alive(process):
            try:
                process.terminate()
            except Exception as error:
                failures.append((process.name, error))
    termination_deadline = clock() + 5.0
    while (
        any(is_alive(process) for process in processes)
        and clock() < termination_deadline
    ):
        sleep(min(0.05, max(0.0, termination_deadline - clock())))
    for process in processes:
        if is_alive(process) and hasattr(process, "kill"):
            try:
                process.kill()
            except Exception as error:
                failures.append((process.name, error))
    for process in processes:
        try:
            process.join(timeout=max(0.0, termination_deadline - clock()))
        except Exception as error:
            failures.append((process.name, error))
    return failures


def monitor_parent_workers(
    monitor,
    *,
    processes,
    connections,
    restore_terminal,
):
    result = None
    primary = None
    traceback = None
    try:
        result = monitor()
    except BaseException as error:
        primary = error
        traceback = error.__traceback__

    failures = list(stop_worker_processes(processes))
    for index, connection in enumerate(connections):
        try:
            connection.close()
        except Exception as error:
            failures.append((f"connection_{index}", error))
    try:
        restore_terminal()
    except Exception as error:
        failures.append(("terminal", error))

    if primary is not None:
        if failures:
            print(CleanupError(failures), file=sys.stderr)
        raise primary.with_traceback(traceback)
    if failures:
        raise CleanupError(failures)
    return result
```

- [ ] **Step 4: Run lifecycle tests and verify GREEN**

Run the Task 8 Step 2 command.

Expected: all lifecycle cases pass without child processes.

- [ ] **Step 5: Commit lifecycle primitives**

```bash
git add src/simple/evals/lifecycle.py tests/test_eval_worker_cleanup.py
git commit -m "feat: own evaluation resource cleanup"
```

## Task 9: Apply lifecycle ownership to library and both CLI evaluators

**Files:**

- Modify: `src/simple/evals/env_runner.py`
- Modify: `src/simple/cli/eval.py`
- Modify: `src/simple/cli/eval_decoupled_wbc.py`
- Modify: `tests/test_eval_worker_cleanup.py`

- [ ] **Step 1: Add evaluator integration tests with injected fakes**

Add `from types import SimpleNamespace`, `import numpy as np`, both CLI module
imports, and `simple.envs.wrappers.video_recorder as video_recorder_module` to
the top-level import block of `tests/test_eval_worker_cleanup.py`. Append these
executable episode integration fakes and tests:

```python
class FakeEpisodeEnv:
    def __init__(self, events, *, success, failure_phase=None):
        self.events = events
        self.failure_phase = failure_phase
        self.unwrapped = SimpleNamespace(_success=success)

    def reset(self, **kwargs):
        self.events.append("env.reset")
        if self.failure_phase == "reset":
            raise RuntimeError("reset failed")
        return {"joint_qpos": np.zeros(43, np.float32)}, {"episode_index": 0}

    def step(self, action):
        self.events.append("env.step")
        if self.failure_phase == "render":
            self.render()
        if self.failure_phase == "step":
            raise RuntimeError("step failed")
        return {"joint_qpos": np.zeros(43, np.float32)}, 0.0, True, False, {}

    def render(self):
        self.events.append("env.render")
        raise RuntimeError("render failed")

    def close(self):
        self.events.append("environment.close")


class FakeEpisodePolicy:
    def __init__(self, events, *, failure_phase=None):
        self.events = events
        self.failure_phase = failure_phase
        self._wbc_policy = SimpleNamespace(
            lower_body_policy=SimpleNamespace(
                use_policy_action=False,
                gait_indices=None,
            )
        )

    def reset(self, **kwargs):
        self.events.append("agent.reset")
        if self.failure_phase == "agent_reset":
            raise RuntimeError("agent reset failed")

    def get_action(self, observation, **kwargs):
        self.events.append("agent.get_action")
        if self.failure_phase == "get_action":
            raise RuntimeError("get action failed")
        return object()

    def close(self):
        self.events.append("agent.close")


class FakeEpisodeRecorder:
    def __init__(self, env, events):
        self.env = env
        self.events = events
        self.calls = []

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        result = self.env.step(action)
        if self.env.failure_phase == "video_write":
            self.events.append("video.write")
            raise RuntimeError("video write failed")
        return result

    def _init_writers(self, observation):
        self.events.append("video.reseed")

    def release(self, *, success):
        self.calls.append(success)
        self.events.append(f"video.release:{success}")


@pytest.mark.parametrize("module", [eval_cli, decoupled_cli])
@pytest.mark.parametrize(
    "failure_phase",
    [None, "reset", "agent_reset", "get_action", "step", "render", "video_write"],
)
@pytest.mark.parametrize("task_success", [False, True])
def test_cli_episode_helpers_finalize_one_explicit_verdict(
    module, failure_phase, task_success
):
    events = []
    raw_env = FakeEpisodeEnv(
        events,
        success=task_success,
        failure_phase=failure_phase,
    )
    recorder = FakeEpisodeRecorder(raw_env, events)
    agent = FakeEpisodePolicy(events, failure_phase=failure_phase)
    common = {
        "raw_env": raw_env,
        "agent": agent,
        "task": SimpleNamespace(instruction="probe"),
        "env_conf": {},
        "episode": [],
        "policy": "psi0",
        "task_id": "episode_0",
        "report": lambda *args, **kwargs: None,
        "step_update_every": 5,
    }
    if module is decoupled_cli:
        common.update(
            sonic_env=raw_env,
            robot=SimpleNamespace(stabilized=True),
            control_dt=0.02,
        )

    expected_verdict = failure_phase is None and task_success
    if failure_phase is None:
        with EpisodeVideo(recorder) as video:
            _, success = module._execute_episode(video.env, **common)
            video.finish(success)
    else:
        with pytest.raises(RuntimeError, match="failed"):
            with EpisodeVideo(recorder) as video:
                module._execute_episode(video.env, **common)
    assert recorder.calls == [expected_verdict]
    assert events.count(f"video.release:{expected_verdict}") == 1


@pytest.mark.parametrize(
    "failure_phase", [None, "reset", "get_action", "step", "render", "video_write"]
)
def test_env_runner_episode_uses_the_same_verdict_contract(
    monkeypatch, tmp_path, failure_phase
):
    from simple.evals.env_runner import EnvRunner

    events = []
    raw_env = FakeEpisodeEnv(events, success=True, failure_phase=failure_phase)
    recorder = FakeEpisodeRecorder(raw_env, events)
    monkeypatch.setattr(
        video_recorder_module,
        "VideoRecorder",
        lambda **kwargs: recorder,
    )
    runner = EnvRunner.__new__(EnvRunner)
    runner.config = SimpleNamespace(
        save_video=True,
        eval_dir=str(tmp_path),
        split="train",
    )
    runner._raw_env = raw_env
    runner._render_hz = 50
    runner.task = SimpleNamespace(instruction="probe")
    runner._default_reset_kwargs = lambda *args: {}
    policy = FakeEpisodePolicy(events, failure_phase=failure_phase)
    if failure_phase is None:
        result = runner.run_episode(
            policy,
            {},
            [],
            episode_idx=0,
            policy_name="probe",
        )
        assert result.success is True
    else:
        with pytest.raises(RuntimeError, match="failed"):
            runner.run_episode(
                policy,
                {},
                [],
                episode_idx=0,
                policy_name="probe",
            )
    assert recorder.calls == [failure_phase is None]


@pytest.mark.parametrize("module", [eval_cli, decoupled_cli])
def test_result_persistence_failure_keeps_primary_and_closes_every_resource(module):
    events = []
    recorder = FakeRecorder()

    def persist(kind, payload):
        events.append("persist")
        raise RuntimeError("persist failed")

    with pytest.raises(RuntimeError, match="persist failed"):
        with ResourceOwner() as owner:
            environment, agent = construct_worker_stack(
                owner,
                environment_factory=lambda: Closeable("environment.close", events),
                wrapper_factories=(
                    ("timelimit", lambda env: Closeable("timelimit.close", events)),
                ),
                agent_factory=lambda env: Closeable("agent.close", events),
                rollout_factory=lambda env, agent: Closeable("rollout.close", events),
            )
            assert environment is not None and agent is not None
            with EpisodeVideo(recorder) as video:
                video.finish(True)
            module._persist_completed_worker(
                {"episode_0": True},
                persist_payload=persist,
                report=lambda *args, **kwargs: None,
            )
    assert recorder.calls == [True]
    assert events == [
        "persist",
        "agent.close",
        "rollout.close",
        "timelimit.close",
        "environment.close",
    ]
```

The `render` and `video_write` cases are deliberately distinct: the former
raises from the engine-facing environment step before a frame is returned;
the latter raises from the recorder after the delegated step. Together with
Task 6's literal transcode failures, these cover every required video/render
failure phase. Task 8's executable construction and parent-interruption tests
cover rollback and `KeyboardInterrupt`; no cleanup acceptance criterion is
left as prose.

- [ ] **Step 2: Run integration tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_eval_worker_cleanup.py -k 'evaluator or env_runner or parent'
```

Expected: failures showing normal-path-only recorder/environment cleanup.

- [ ] **Step 3: Own worker resources in both CLI modules**

Import `EpisodeVideo`, `ResourceOwner`, and `stop_worker_processes`. Add the
following helper to `src/simple/cli/eval.py`; this is the complete ordinary
episode reset and policy loop extracted from the current worker:

```python
def _execute_episode(
    env,
    *,
    raw_env,
    agent,
    task,
    env_conf,
    episode,
    policy,
    task_id,
    report,
    step_update_every,
):
    observation, info = env.reset(
        options={"state_dict": env_conf, "task_id": task_id}
    )
    if policy == "vlt":
        reset_kwargs = {
            "camera_infos": env_conf["camera_info"],
            "episode": episode,
            "condition": "forward_all",
            "save_cond_images": True,
        }
    else:
        reset_kwargs = {}
    agent.reset(**reset_kwargs)
    frame_idx = 0
    episode_over = False
    while not episode_over:
        try:
            action = agent.get_action(
                observation,
                info=info,
                instruction=task.instruction,
            )
            observation, _, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated
            frame_idx += 1
            if (
                frame_idx == 1
                or frame_idx % step_update_every == 0
                or episode_over
            ):
                report("episode_step", episode=task_id, step=frame_idx)
        except StopIteration:
            episode_over = True
    return frame_idx, bool(raw_env.unwrapped._success)
```

Add this complete decoupled helper to
`src/simple/cli/eval_decoupled_wbc.py`:

```python
def _execute_episode(
    env,
    *,
    raw_env,
    sonic_env,
    agent,
    robot,
    task,
    env_conf,
    episode,
    policy,
    task_id,
    report,
    step_update_every,
    control_dt,
):
    observation, info = env.reset(
        options={"state_dict": env_conf, "task_id": task_id}
    )
    if policy == "vlt":
        reset_kwargs = {
            "camera_infos": env_conf["camera_info"],
            "episode": episode,
            "condition": "forward_all",
            "save_cond_images": True,
        }
    else:
        reset_kwargs = {}
    agent.reset(**reset_kwargs)
    agent._wbc_policy.lower_body_policy.use_policy_action = True

    sim_cnt = 0
    while not robot.stabilized and sim_cnt < 300:
        step_start = time.monotonic()
        action = agent.get_stabilize_action(observation)
        stabilization_result = sonic_env.step(action)
        observation = stabilization_result[0]
        info = stabilization_result[-1]
        sonic_env.update_viewer()
        sonic_env.update_reward()
        sleep_time = control_dt - (time.monotonic() - step_start)
        if sleep_time > 0:
            time.sleep(sleep_time)
        sim_cnt += 1

    init_writers = getattr(env, "_init_writers", None)
    if callable(init_writers):
        init_writers(observation)
    agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros(
        (1), dtype=torch.float32
    )

    frame_idx = 0
    episode_over = False
    while not episode_over:
        try:
            action = agent.get_action(
                observation,
                info=info,
                instruction=task.instruction,
            )
            observation, _, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated
            frame_idx += 1
            if (
                frame_idx == 1
                or frame_idx % step_update_every == 0
                or episode_over
            ):
                report("episode_step", episode=task_id, step=frame_idx)
        except StopIteration:
            episode_over = True
    return frame_idx, bool(raw_env.unwrapped._success)
```

Add this identical result helper to both CLI modules. It creates one explicit
failure seam for the executable persistence test and keeps persistence inside
the worker's `ResourceOwner` scope:

```python
def _persist_completed_worker(stats, *, persist_payload, report):
    payload = dict(stats)
    persist_payload("ok", payload)
    report(
        "worker_status",
        status="closing",
    )
    report(
        "worker_done",
        completed_episodes=len(payload),
        successes=sum(payload.values()),
    )
    return payload
```

Replace each worker's three inline persistence/report calls with:

```python
_persist_completed_worker(
    stats,
    persist_payload=persist_payload,
    report=report,
)
```

Replace the construction and episode section of the ordinary worker with this
literal ownership sequence. Register every wrapper and the raw environment so
a wrapper close failure cannot suppress the raw fallback close. The idempotent
`BaseDualSim.close` makes repeated delegated closes safe:

```python
with ResourceOwner() as resources:
    policy_module = importlib.import_module(f"simple.baselines.{policy}")
    agent_clazz = getattr(policy_module, f"{snake_to_pascal(policy)}Agent")

    def make_environment():
        return _make_worker_environment(
            env_id,
            make_kwargs=make_kwargs,
            third_person_video=third_person_video,
        )

    def maybe_time_limit(environment):
        episode_limit = max_episode_steps
        if episode_limit is None:
            episode_limit = environment.unwrapped.task.metadata.get(
                "max_episode_steps"
            )
        if episode_limit is None:
            return environment
        return TimeLimit(environment, max_episode_steps=episode_limit)

    def make_agent(environment):
        task = environment.unwrapped.task
        return agent_clazz(task.robot, host, port)

    rollout_factory = None
    if rollout_save_dir is not None:
        from simple.envs.lerobot import LerobotRecorder

        rollout_factory = lambda environment, agent: LerobotRecorder(
            env=environment, root_dir=rollout_save_dir, agent=agent
        )

    environment, agent = construct_worker_stack(
        resources,
        environment_factory=make_environment,
        wrapper_factories=(
            ("stand_environment", StandStabilizationWrapper),
            ("timelimit_environment", maybe_time_limit),
        ),
        agent_factory=make_agent,
        rollout_factory=rollout_factory,
    )
    task = environment.unwrapped.task

    stats = defaultdict(bool)
    for eps_idx in episode_indices:
        env_conf, episode = get_episode(dataset, eps_idx)
        task_id = f"episode_{eps_idx}"
        report("episode_start", episode=task_id)
        episode_start_time = time.perf_counter()
        episode_arguments = {
            "raw_env": environment,
            "agent": agent,
            "task": task,
            "env_conf": env_conf,
            "episode": episode,
            "policy": policy,
            "task_id": task_id,
            "report": report,
            "step_update_every": step_update_every,
        }
        recorder = None
        if save_video:
            recorder = VideoRecorder(
                env=environment,
                video_folder=eval_output_dir,
                name_prefix=task_id,
                framerate=render_hz,
                write_png=False,
            )
        if recorder is None:
            frame_idx, is_success = _execute_episode(
                environment, **episode_arguments
            )
        else:
            with EpisodeVideo(recorder) as video_run:
                frame_idx, is_success = _execute_episode(
                    video_run.env, **episode_arguments
                )
                video_run.finish(is_success)
        stats[task_id] = is_success
        episode_seconds = time.perf_counter() - episode_start_time
        _append_eval_stats_line(eval_dir, f"{task_id}: {is_success} \n")
        report(
            "episode_end",
            episode=task_id,
            step=frame_idx,
            completed_episodes=len(stats),
            successes=sum(stats.values()),
            episode_seconds=episode_seconds,
            steps_per_second=(
                frame_idx / episode_seconds if episode_seconds > 0 else 0.0
            ),
        )

    _persist_completed_worker(
        stats,
        persist_payload=persist_payload,
        report=report,
    )
return stats
```

Use the same block in the decoupled worker with these exact construction and
helper-argument substitutions:

```python
with ResourceOwner() as resources:
    policy_module = importlib.import_module(f"simple.baselines.{policy}")
    agent_clazz = getattr(policy_module, f"{snake_to_pascal(policy)}Agent")

    def make_environment():
        return _make_worker_environment(
            env_id,
            make_kwargs=make_kwargs,
            third_person_video=third_person_video,
        )

    def maybe_time_limit(environment):
        episode_limit = max_episode_steps
        if episode_limit is None:
            episode_limit = environment.unwrapped.task.metadata.get(
                "max_episode_steps"
            )
        if episode_limit is None:
            return environment
        return TimeLimit(environment, max_episode_steps=episode_limit)

    def make_agent(environment):
        task = environment.unwrapped.task
        return agent_clazz(
            task.robot,
            host,
            port,
            sonic_config=sonic_config,
        )

    rollout_factory = None
    if rollout_save_dir is not None:
        from simple.envs.lerobot import LerobotRecorder

        rollout_factory = lambda environment, agent: LerobotRecorder(
            env=environment, root_dir=rollout_save_dir, agent=agent
        )

    environment, agent = construct_worker_stack(
        resources,
        environment_factory=make_environment,
        wrapper_factories=(("timelimit_environment", maybe_time_limit),),
        agent_factory=make_agent,
        rollout_factory=rollout_factory,
    )
    sonic_env = environment.unwrapped
    task = sonic_env.task
    if success_criteria is not None:
        task.success_criteria = success_criteria
        task.metadata["success_criteria"] = success_criteria

    stats = defaultdict(bool)
    for eps_idx in episode_indices:
        env_conf, episode = get_episode(dataset, eps_idx)
        task_id = f"episode_{eps_idx}"
        report("episode_start", episode=task_id)
        episode_start_time = time.perf_counter()
        episode_arguments = {
            "raw_env": environment,
            "sonic_env": sonic_env,
            "agent": agent,
            "robot": task.robot,
            "task": task,
            "env_conf": env_conf,
            "episode": episode,
            "policy": policy,
            "task_id": task_id,
            "report": report,
            "step_update_every": step_update_every,
            "control_dt": control_dt,
        }
        recorder = None
        if save_video:
            recorder = VideoRecorder(
                env=environment,
                video_folder=eval_output_dir,
                name_prefix=task_id,
                framerate=render_hz,
                write_png=False,
            )
        if recorder is None:
            frame_idx, is_success = _execute_episode(
                environment, **episode_arguments
            )
        else:
            with EpisodeVideo(recorder) as video_run:
                frame_idx, is_success = _execute_episode(
                    video_run.env, **episode_arguments
                )
                video_run.finish(is_success)
        stats[task_id] = is_success
        episode_seconds = time.perf_counter() - episode_start_time
        _append_eval_stats_line(eval_dir, f"{task_id}: {is_success} \n")
        report(
            "episode_end",
            episode=task_id,
            step=frame_idx,
            completed_episodes=len(stats),
            successes=sum(stats.values()),
            episode_seconds=episode_seconds,
            steps_per_second=(
                frame_idx / episode_seconds if episode_seconds > 0 else 0.0
            ),
        )

    _persist_completed_worker(
        stats,
        persist_payload=persist_payload,
        report=report,
    )
return stats
```

Keep the existing setup-ready report immediately before `stats` in each block.
Remove both direct `raw_env.close()` and explicit per-episode `env.release()`
calls. Import `construct_worker_stack` together with `EpisodeVideo`,
`ResourceOwner`, and `monitor_parent_workers` in both CLI modules.

- [ ] **Step 4: Own `EnvRunner.run_episode` recording**

Import `EpisodeVideo` and `ResourceOwner` from `simple.evals.lifecycle` in
`env_runner.py`.

In `run_episode`, retain the current reset and policy loop but put it in a local
`execute_episode(env)` closure returning `EvalEpisodeResult`. Invoke that
closure with this exact ownership branch:

```python
reset_kwargs = dict(
    self._default_reset_kwargs(env_config, episode_data)
)
reset_options = dict(reset_kwargs.get("options", {}))
reset_options["task_id"] = task_id
reset_kwargs["options"] = reset_options
```

Use `reset_kwargs` in the closure's `env.reset(...)` call. Supplying `task_id`
makes `VideoRecorder.reset` clear the owned episode directory before creating
new writers; Task 6's per-writer rotation remains the direct-use fallback.

```python
if not self.config.save_video:
    return execute_episode(base_env or self._raw_env)
from simple.envs.wrappers.video_recorder import VideoRecorder

output_dir = eval_output_dir or (
    Path(self.config.eval_dir)
    / self.policy_output_name(policy_name)
    / self.config.split
)
recorder = VideoRecorder(
    env=base_env or self._raw_env,
    video_folder=str(output_dir),
    name_prefix=task_id,
    framerate=self._render_hz,
    write_png=False,
)
with EpisodeVideo(recorder) as video_run:
    result = execute_episode(video_run.env)
    video_run.finish(result.success)
    return result
```

Own the policy and optional rollout wrapper for the whole `EnvRunner.run`
method without making a normal non-recording run close the runner-owned raw
environment:

```python
with ResourceOwner() as resources:
    resources.own("environment", self._raw_env)
    agent, resolved_policy_name = self._resolve_policy(
        policy,
        policy_name=policy_name,
        policy_reset_fn=policy_reset_fn,
    )
    resources.own("agent", agent)
    rollout_env = self._raw_env
    closes_raw_environment = False
    if self.config.rollout_save_dir:
        from simple.envs.lerobot import LerobotRecorder

        rollout_env = LerobotRecorder(
            env=self._raw_env,
            root_dir=self.config.rollout_save_dir,
            agent=agent,
        )
        resources.own_before(
            "rollout_environment", rollout_env, before_name="agent"
        )
        closes_raw_environment = True
    else:
        resources.disown("environment", self._raw_env)
    try:
        return self._run_resolved_policy(
            agent,
            resolved_policy_name=resolved_policy_name,
            rollout_env=rollout_env,
            num_episodes=num_episodes,
            episode_start=episode_start,
            episode_reset_kwargs_fn=episode_reset_kwargs_fn,
            progress_callback=progress_callback,
            progress_reporter=progress_reporter,
        )
    finally:
        if closes_raw_environment:
            self._closed = True
```

Extract the current episode-index loop and `EvalResult` construction verbatim
into `_run_resolved_policy` with the keyword-only signature shown by this call;
it contains no construction or cleanup. In `EnvRunner.close`, retain the
existing `_closed` guard. Do not defer a recorder to `atexit`; only the Isaac
application close may retain the existing library-specific deferral.

- [ ] **Step 5: Bound multi-worker interruption**

Extract this exact process-start and receive loop as the local
`monitor_workers()` function in both CLI parents:

```python
def monitor_workers():
    with Live(
        render_progress(env_id, policy, worker_states, log_path),
        console=console,
        refresh_per_second=4,
        auto_refresh=show_progress,
    ) as live:
        for wid in range(num_workers):
            worker_result_path = str(result_dir / f"worker_{wid}.pkl")
            recv_conn, send_conn = ctx.Pipe(duplex=False)
            owned_connections.extend((recv_conn, send_conn))
            process = ctx.Process(
                target=_run_eval_worker_entry,
                args=(
                    worker_result_path,
                    wid,
                    num_workers,
                    worker_kwargs,
                    worker_log_paths[wid],
                    send_conn,
                ),
                name=f"eval-worker-{wid}",
            )
            try:
                process.start()
            except BaseException:
                send_conn.close()
                recv_conn.close()
                raise
            send_conn.close()
            progress_readers[wid] = recv_conn
            procs.append(process)

        while any(process.is_alive() for process in procs) or progress_readers:
            ready = (
                wait(list(progress_readers.values()), timeout=0.2)
                if progress_readers
                else []
            )
            for connection in ready:
                try:
                    wid, payload = connection.recv()
                except EOFError:
                    for key, value in list(progress_readers.items()):
                        if value is connection:
                            value.close()
                            del progress_readers[key]
                            break
                    continue
                update_progress(worker_states, wid, payload)
                if show_progress:
                    live.update(
                        render_progress(
                            env_id, policy, worker_states, log_path
                        ),
                        refresh=False,
                    )
```

Initialize `owned_connections = []` beside `procs` before defining the local
function. Replace each multi-worker monitor's `try/finally` with the shared,
executable exception-preservation path tested in Task 8:

```python
def restore_parent_terminal():
    restore_cursor(console)
    if terminal_stream is not None:
        terminal_stream.close()


monitor_parent_workers(
    monitor_workers,
    processes=procs,
    connections=owned_connections,
    restore_terminal=restore_parent_terminal,
)
```

On normal exit `stop_worker_processes` only joins dead children and sends no
signal. Every receive pipe remains in `owned_connections` even after EOF so
the parent helper attempts every close; repeated `close()` calls are harmless.

- [ ] **Step 6: Run integration and existing evaluator tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_eval_worker_cleanup.py \
  tests/test_official_eval_compatibility.py
```

Expected: PASS.

- [ ] **Step 7: Commit evaluator lifecycle integration**

```bash
git add src/simple/evals/env_runner.py src/simple/cli/eval.py \
  src/simple/cli/eval_decoupled_wbc.py tests/test_eval_worker_cleanup.py
git commit -m "fix: finalize evaluation artifacts on every exit"
```

## Task 10: Add a policy-free backend frame verifier

**Files:**

- Create: `scripts/verify_third_person_camera.py`
- Modify: `tests/test_third_person_eval_backends.py`

- [ ] **Step 1: Write failing CLI parsing and report tests**

Add `import json`, `import pytest`, and the verifier import below to the
top-level import block of `tests/test_third_person_eval_backends.py`, then
append the literal tests without adding imports between test functions:

```python
from scripts.verify_third_person_camera import (
    build_parser,
    read_effective_clipping,
    validate_frame,
    write_artifacts,
)


def test_verifier_parser_is_strict():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--env-id",
            "simple/Probe-v0",
            "--data-dir",
            "/dataset",
            "--output-dir",
            "/output",
            "--sim-mode",
            "mujoco",
        ]
    )
    assert args.episode_index == 0
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--env-id",
                "simple/Probe-v0",
                "--data-dir",
                "/dataset",
                "--output-dir",
                "/output",
                "--sim-mode",
                "mujoco_isaac",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args([])


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((360, 640, 3), dtype=np.float32),
        np.zeros((360, 640, 3), dtype=np.uint8),
        np.full((360, 640, 3), 22, dtype=np.uint8),
    ],
)
def test_verifier_rejects_invalid_frames(frame):
    with pytest.raises(ValueError):
        validate_frame(frame)


def test_verifier_writes_exact_report(tmp_path):
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[170:190, 230:250] = (255, 0, 0)
    frame[170:190, 310:330] = (0, 255, 0)
    frame[170:190, 390:410] = (0, 0, 255)
    validate_frame(frame)
    report_path = write_artifacts(
        frame,
        output_dir=tmp_path,
        env_id="simple/Probe-v0",
        sim_mode="mujoco",
        episode_index=0,
        effective_clipping=(0.2, 5.0),
    )
    report = json.loads(report_path.read_text())
    assert set(report) == {
        "env_id",
        "sim_mode",
        "episode_index",
        "shape",
        "dtype",
        "minimum",
        "maximum",
        "mean",
        "unique_rgb_count",
        "png_sha256",
        "marker_validation",
        "effective_clipping",
        "effective_clipping_ok",
    }
    assert report["shape"] == [360, 640, 3]
    assert report["dtype"] == "uint8"
    assert report["marker_validation"]["marker_order_ok"] is True
    assert report["marker_validation"]["center_projection_ok"] is True
    assert report["marker_validation"]["clipping_ok"] is True
    assert report["effective_clipping"] == [0.2, 5.0]
    assert report["effective_clipping_ok"] is True
    assert len(report["png_sha256"]) == 64
    assert (tmp_path / "third_person.png").is_file()


def test_effective_clipping_reads_initialized_backend_without_importing_isaac():
    mujoco_env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            mujoco=SimpleNamespace(
                mjModel=SimpleNamespace(
                    stat=SimpleNamespace(extent=4.0),
                    vis=SimpleNamespace(
                        map=SimpleNamespace(znear=0.05, zfar=1.25)
                    ),
                )
            )
        )
    )
    assert read_effective_clipping(mujoco_env, "mujoco") == (0.2, 5.0)
    isaac_env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            isaac=SimpleNamespace(
                cameras={
                    "third_person": SimpleNamespace(
                        get_clipping_range=lambda: (0.2, 5.0)
                    )
                }
            )
        )
    )
    assert read_effective_clipping(isaac_env, "isaac") == (0.2, 5.0)
```

- [ ] **Step 2: Run script tests and verify RED**

Run the Task 4 Step 2 command.

Expected: import failure for the new script.

- [ ] **Step 3: Implement the verifier**

Create `scripts/verify_third_person_camera.py` exactly as follows:

```python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

import simple.envs as _simple_envs  # noqa: F401
from simple.datasets.lerobot import get_episode_lerobot
from simple.evals.lifecycle import ResourceOwner
from simple.evals.third_person_camera import (
    THIRD_PERSON_SENSOR_KEY,
    analyze_verification_frame,
    make_environment_sensor_kwargs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--sim-mode", required=True, choices=("mujoco", "isaac")
    )
    return parser


def validate_frame(frame: np.ndarray) -> dict:
    if not isinstance(frame, np.ndarray):
        raise ValueError("third-person frame must be a numpy array")
    try:
        report = analyze_verification_frame(frame)
    except AssertionError as cause:
        raise ValueError(str(cause)) from cause
    failed = [
        key
        for key in ("marker_order_ok", "center_projection_ok", "clipping_ok")
        if report[key] is not True
    ]
    if failed:
        raise ValueError(f"third-person marker validation failed: {failed}")
    return report


def read_effective_clipping(env, sim_mode: str) -> tuple[float, float]:
    if sim_mode == "mujoco":
        model = env.unwrapped.mujoco.mjModel
        extent = float(model.stat.extent)
        return (
            float(model.vis.map.znear) * extent,
            float(model.vis.map.zfar) * extent,
        )
    if sim_mode == "isaac":
        near, far = env.unwrapped.isaac.cameras[
            THIRD_PERSON_SENSOR_KEY
        ].get_clipping_range()
        return float(near), float(far)
    raise ValueError(f"unsupported verification backend: {sim_mode}")


def write_artifacts(
    frame: np.ndarray,
    *,
    output_dir: Path,
    env_id: str,
    sim_mode: str,
    episode_index: int,
    effective_clipping: tuple[float, float],
) -> Path:
    marker_validation = validate_frame(frame)
    clipping_ok = bool(
        np.allclose(effective_clipping, (0.2, 5.0), rtol=0.0, atol=1e-6)
    )
    if not clipping_ok:
        raise ValueError(f"unexpected effective clipping: {effective_clipping}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "third_person.png"
    report_path = output_dir / "report.json"
    if png_path.exists() or report_path.exists():
        raise FileExistsError(f"verification artifact already exists: {output_dir}")
    Image.fromarray(frame, mode="RGB").save(png_path)
    report = {
        "env_id": env_id,
        "sim_mode": sim_mode,
        "episode_index": episode_index,
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "minimum": int(frame.min()),
        "maximum": int(frame.max()),
        "mean": float(frame.mean()),
        "unique_rgb_count": int(
            np.unique(frame.reshape(-1, 3), axis=0).shape[0]
        ),
        "png_sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
        "marker_validation": marker_validation,
        "effective_clipping": [float(value) for value in effective_clipping],
        "effective_clipping_ok": clipping_ok,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report_path


def make_loopback_sonic_config() -> dict:
    from simple.cli.eval_decoupled_wbc import _make_sonic_config

    sonic_config = _make_sonic_config()
    sonic_config["ENV_NAME"] = "simple"
    sonic_config["INTERFACE"] = "lo"
    sonic_config["DOMAIN_ID"] = 42
    return sonic_config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episode_index < 0:
        raise ValueError("episode index must be nonnegative")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=args.env_id,
        root=args.data_dir,
        video_backend="pyav",
    )
    env_conf, _ = get_episode_lerobot(dataset, args.episode_index)
    make_kwargs = {
        "sim_mode": args.sim_mode,
        "render_hz": dataset.meta.fps,
        "headless": True,
        "sonic_config": make_loopback_sonic_config(),
        "_third_person_verification_markers": True,
        **make_environment_sensor_kwargs(True),
    }
    with ResourceOwner() as resources:
        env = resources.own("environment", gym.make(args.env_id, **make_kwargs))
        observation, _ = env.reset(
            options={
                "state_dict": env_conf,
                "task_id": f"episode_{args.episode_index}",
            }
        )
        frame = observation[THIRD_PERSON_SENSOR_KEY]
        effective_clipping = read_effective_clipping(env, args.sim_mode)
        report_path = write_artifacts(
            frame,
            output_dir=Path(args.output_dir),
            env_id=args.env_id,
            sim_mode=args.sim_mode,
            episode_index=args.episode_index,
            effective_clipping=effective_clipping,
        )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run unit tests and commit the verifier**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_third_person_eval_backends.py
git add scripts/verify_third_person_camera.py tests/test_third_person_eval_backends.py
git commit -m "test: add third-person backend frame verifier"
```

## Task 11: Run static, focused, and full regression gates

**Files:** all implementation and test files above

- [ ] **Step 1: Run the approved focused suite**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_third_person_eval_camera.py \
  tests/test_third_person_eval_backends.py \
  tests/test_third_person_policy_isolation.py \
  tests/test_eval_video_finalization.py \
  tests/test_eval_worker_cleanup.py
```

Expected: PASS.

- [ ] **Step 2: Run static checks**

```bash
legacy_modified_files=(
  src/simple/core/registry.py
  src/simple/core/task.py
  src/simple/envs/base_dual_env.py
  src/simple/engines/isaacsim.py
  src/simple/engines/mujoco.py
  src/simple/envs/video_writer.py
  src/simple/envs/wrappers/video_recorder.py
  src/simple/evals/api.py
  src/simple/evals/env_runner.py
  src/simple/cli/eval.py
  src/simple/cli/eval_decoupled_wbc.py
)
new_python_files=(
  src/simple/evals/third_person_camera.py
  src/simple/evals/lifecycle.py
  scripts/verify_third_person_camera.py
  tests/test_third_person_eval_camera.py
  tests/test_third_person_eval_backends.py
  tests/test_third_person_policy_isolation.py
  tests/test_eval_video_finalization.py
  tests/test_eval_worker_cleanup.py
)
ruff check --no-cache --select E9,F63,F7,F82 \
  "${legacy_modified_files[@]}" "${new_python_files[@]}"
ruff check --no-cache "${new_python_files[@]}"
ruff format --check --no-cache "${new_python_files[@]}"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  src/simple/evals/third_person_camera.py \
  src/simple/evals/lifecycle.py \
  scripts/verify_third_person_camera.py
git diff --check 83c175c..HEAD
git diff --check
```

Expected: every command exits zero.

The first whitespace command covers the complete implementation range from the
approved design commit; the second covers any still-uncommitted correction.
The narrowed legacy-file rule set is intentional: the pre-existing files have
46 unrelated Ruff findings and nine pre-existing format deltas on commit
`83c175c`. Do not expand this feature into a repository-wide style cleanup.
Every newly created file still receives the full Ruff and format gates.

- [ ] **Step 3: Run the full isolated test suite**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider tests
```

Expected: PASS. If a failure is load-sensitive, rerun only to characterize it;
do not claim a pass until the full suite exits zero in one run.

- [ ] **Step 4: Commit only if verification required a correction**

```bash
git add src/simple scripts/verify_third_person_camera.py tests
git commit -m "test: complete third-person evaluation coverage"
```

Skip this commit when the worktree has no tracked correction after Steps 1-3.

## Task 12: Verify both render backends and one official PSI0 episode

**Files/artifacts:**

- Create at runtime: `outputs/third-person-camera-verification/<run-id>/`
- Preserve: every raw/final video and JSON/PNG report from this run

- [ ] **Step 1: Reconfirm simulation-only isolation**

Execute Steps 1-8 in one shell so the ownership PIDs and immutable run path are
not lost. Create the run directory without overwriting earlier evidence, then
run the Task 0 Step 2 process checks:

```bash
verification_root=outputs/third-person-camera-verification
mkdir -p "$verification_root"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short=12 HEAD)"
verification_dir="$verification_root/$run_id"
mkdir "$verification_dir"
printf '%s\n' "$verification_dir" > "$verification_root/latest-run.txt"
pgrep -af 'eval-decoupled-wbc|eval-worker|serve_psi0|run_g1_control_loop|psi0_simple_real_bridge' || true
lsof -nP -iTCP:22085 -sTCP:LISTEN || true
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
```

Expected: the local port is unused and no process is owned by this task. The
verifier itself overwrites the SONIC values with `ENV_NAME=simple`,
`INTERFACE=lo`, and `DOMAIN_ID=42`. Do not continue if another domain-42
simulation or any real-interface process is active.

- [ ] **Step 2: Run the MuJoCo-only frame verifier**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/verify_third_person_camera.py \
  --env-id simple/G1WholebodyXMovePickTeleop-v0 \
  --data-dir data/evals/simple-eval/G1WholebodyXMovePickTeleop-v0/dr-level-0 \
  --episode-index 0 \
  --sim-mode mujoco \
  --output-dir "$verification_dir/mujoco"
```

Expected: PNG and JSON report exist, shape is `[360, 640, 3]`, and the report
passes `validate_frame`.

- [ ] **Step 3: Run the initialized Isaac-only frame verifier**

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/verify_third_person_camera.py \
  --env-id simple/G1WholebodyXMovePickTeleop-v0 \
  --data-dir data/evals/simple-eval/G1WholebodyXMovePickTeleop-v0/dr-level-0 \
  --episode-index 0 \
  --sim-mode isaac \
  --output-dir "$verification_dir/isaac"
```

Expected: a distinct PNG/report pair with the same shape. Both reports require
red/green/blue centroid ordering, the green marker within 32 pixels of optical
center, no visible near-magenta or far-cyan sentinel, and measured clipping
`[0.2, 5.0]`. The Isaac engine import occurs only after `SimulationApp` startup
inside `gym.make`; the ordinary `.venv` unit gate never imports `carb`.

- [ ] **Step 4: Start the official server/tunnel and record provenance**

Use the existing H100 container only as a dependency/GPU environment. Do not
trust its source copy because it omits `.git`; clone the official repository at
the immutable commit below into the already-mounted model volume and execute
that exact source through `PYTHONPATH`. The server-file blob in the provisioned
container was independently matched to official history, but the acceptance
run uses the fresh Git checkout so its whole-tree identity is attestable.
The guarded EXIT/INT/TERM traps are installed before the first command that can
start the remote server or local tunnel. Both owned processes use bounded
INT→TERM→KILL escalation; no cleanup path performs an unchecked blocking wait.

```bash
psi0_commit=a32e57a3fabb8590c80677f9cd3d1fc3db60eb06
psi0_container=jihun_psi0_sonic_train_gpu23_20260805
remote_source=/mnt/data01/jhkim/model_weight/Psi0/verification-source-$psi0_commit
container_source=/hfm/cache/checkpoints/psi0/verification-source-$psi0_commit
remote_run=/mnt/data01/jhkim/model_weight/Psi0/simple-checkpoints/g1wholebodyxmovepick-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604022205
container_run=/hfm/cache/checkpoints/psi0/simple-checkpoints/g1wholebodyxmovepick-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604022205
remote_log="$remote_run/third-person-verification-$run_id.log"
container_log="$container_run/third-person-verification-$run_id.log"
remote_pid_file="$remote_run/third-person-verification-$run_id.pid"
container_pid_file="$container_run/third-person-verification-$run_id.pid"

cleanup_done=0
tunnel_started=0
tunnel_pid=""
tunnel_process_exit_code=""
remote_server_may_exist=0
remote_server_pid=""

local_pid_alive() {
  test -n "$1" && kill -0 "$1" 2>/dev/null
}

wait_local_pid_bounded() {
  local pid="$1"
  local iterations="$2"
  local attempt
  for attempt in $(seq 1 "$iterations"); do
    if ! local_pid_alive "$pid"; then
      wait "$pid"
      tunnel_process_exit_code=$?
      return 0
    fi
    sleep 0.2
  done
  return 124
}

stop_tunnel_bounded() {
  if test "$tunnel_started" -ne 1; then
    return 0
  fi
  if ! local_pid_alive "$tunnel_pid"; then
    wait "$tunnel_pid"
    tunnel_process_exit_code=$?
    return 0
  fi
  kill -INT "$tunnel_pid" 2>/dev/null || true
  wait_local_pid_bounded "$tunnel_pid" 25 && return 0
  kill -TERM "$tunnel_pid" 2>/dev/null || true
  wait_local_pid_bounded "$tunnel_pid" 25 && return 0
  kill -KILL "$tunnel_pid" 2>/dev/null || true
  wait_local_pid_bounded "$tunnel_pid" 25
}

remote_server_alive() {
  test -n "$remote_server_pid" && timeout 10s ssh h100 \
    "docker exec '$psi0_container' kill -0 '$remote_server_pid'" \
    >/dev/null 2>&1
}

wait_remote_server_bounded() {
  local iterations="$1"
  local attempt
  for attempt in $(seq 1 "$iterations"); do
    if ! remote_server_alive; then
      return 0
    fi
    sleep 1
  done
  return 124
}

stop_remote_server_bounded() {
  if test "$remote_server_may_exist" -ne 1; then
    return 0
  fi
  if test -z "$remote_server_pid"; then
    remote_server_pid="$(timeout 10s ssh h100 \
      "docker exec '$psi0_container' cat '$container_pid_file'" 2>/dev/null || true)"
  fi
  if ! remote_server_alive; then
    return 0
  fi
  timeout 10s ssh h100 \
    "docker exec '$psi0_container' kill -INT '$remote_server_pid'" \
    >/dev/null 2>&1 || true
  wait_remote_server_bounded 20 && return 0
  timeout 10s ssh h100 \
    "docker exec '$psi0_container' kill -TERM '$remote_server_pid'" \
    >/dev/null 2>&1 || true
  wait_remote_server_bounded 5 && return 0
  timeout 10s ssh h100 \
    "docker exec '$psi0_container' kill -KILL '$remote_server_pid'" \
    >/dev/null 2>&1 || true
  wait_remote_server_bounded 5
}

cleanup_owned_infrastructure() {
  if test "$cleanup_done" -eq 1; then
    return
  fi
  cleanup_done=1
  trap '' INT TERM
  set +e
  stop_tunnel_bounded
  tunnel_cleanup_code=$?
  printf '%s\n' "$tunnel_cleanup_code" > "$verification_dir/tunnel-cleanup-exit-code.txt"
  printf '%s\n' "$tunnel_process_exit_code" > "$verification_dir/tunnel-exit-code.txt"
  stop_remote_server_bounded
  remote_cleanup_code=$?
  if remote_server_alive; then
    printf '%s\n' true > "$verification_dir/server-alive-after-cleanup.txt"
  else
    printf '%s\n' false > "$verification_dir/server-alive-after-cleanup.txt"
  fi
  printf '%s\n' "$remote_cleanup_code" > "$verification_dir/server-cleanup-exit-code.txt"
  timeout 10s ssh h100 \
    "docker exec '$psi0_container' cat '$container_log'" \
    > "$verification_dir/server.log" 2>&1 || true
  timeout 10s ssh h100 \
    "test ! -e '$remote_pid_file' || mv '$remote_pid_file' '$remote_pid_file.stopped'" \
    >/dev/null 2>&1 || true
  set -e
}

trap cleanup_owned_infrastructure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ssh h100 "docker exec '$psi0_container' nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv,noheader && docker exec '$psi0_container' nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader" > "$verification_dir/h100-gpu-preflight.txt"
ssh h100 "if test -d '$remote_source/.git'; then test \"\$(git -C '$remote_source' rev-parse HEAD)\" = '$psi0_commit' && test -z \"\$(git -C '$remote_source' status --porcelain)\"; else test ! -e '$remote_source' && git clone https://github.com/physical-superintelligence-lab/Psi0.git '$remote_source' && git -C '$remote_source' checkout --detach '$psi0_commit'; fi"
ssh h100 "git -C '$remote_source' rev-parse HEAD && git -C '$remote_source' status --porcelain" > "$verification_dir/psi0-source.txt"
ssh h100 "sha256sum '$remote_run/checkpoints/ckpt_40000/model.safetensors'" > "$verification_dir/checkpoint.sha256"

remote_server_may_exist=1
ssh h100 "docker exec '$psi0_container' bash -lc 'test ! -e \"$container_pid_file\"; test -f \"$container_source/src/psi/deploy/psi0_serve_simple.py\"; cd /workspace/Psi0; nohup env PYTHONPATH=\"$container_source/src\" /workspace/Psi0/.venv/bin/python -m psi.deploy.psi0_serve_simple --host 0.0.0.0 --port 22185 --device cuda:0 --policy=psi0 --run-dir=\"$container_run\" --ckpt-step=40000 --action-exec-horizon=24 --rtc > \"$container_log\" 2>&1 & echo \$! > \"$container_pid_file\"'"
remote_server_pid="$(ssh h100 "docker exec '$psi0_container' cat '$container_pid_file'")"
printf '%s\n' "$remote_server_pid" > "$verification_dir/server.pid"

container_ip="$(ssh h100 "docker inspect '$psi0_container' --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'")"
ssh -N -L "127.0.0.1:22085:$container_ip:22185" h100 > "$verification_dir/tunnel.log" 2>&1 &
tunnel_pid=$!
tunnel_started=1
printf '%s\n' "$tunnel_pid" > "$verification_dir/tunnel.pid"

for attempt in $(seq 1 180); do
  if curl --fail --silent --show-error http://127.0.0.1:22085/health > "$verification_dir/health.json"; then
    break
  fi
  sleep 1
done
kill -0 "$tunnel_pid"
curl --fail --silent --show-error http://127.0.0.1:22085/health > "$verification_dir/health.json"
curl --silent --show-error --output "$verification_dir/info-response.txt" --write-out '%{http_code}\n' http://127.0.0.1:22085/info > "$verification_dir/info-status.txt"
printf '%s\n' "PYTHONPATH=$container_source/src /workspace/Psi0/.venv/bin/python -m psi.deploy.psi0_serve_simple --host 0.0.0.0 --port 22185 --device cuda:0 --policy=psi0 --run-dir=$container_run --ckpt-step=40000 --action-exec-horizon=24 --rtc" > "$verification_dir/server-command.txt"
```

Expected: source HEAD equals the recorded commit with an empty status, the
6,253,648,840-byte `model.safetensors` receives a SHA-256 manifest, health is
200, and only the owned loopback tunnel exposes it locally. `/info` is recorded
whether it returns 200 or 404; artifact lookup below handles either naming
case. Inspect `h100-gpu-preflight.txt` before the server command and stop if
container GPU index 0 already has an unrelated compute PID; do not select or
terminate another user's process.

- [ ] **Step 5: Run exactly one official policy episode**

```bash
printf '%s\n' "SIMPLE_DISABLE_TUI=1 CUDA_VISIBLE_DEVICES=1 timeout --signal=INT --kill-after=75s 1800s .venv/bin/eval-decoupled-wbc simple/G1WholebodyXMovePickTeleop-v0 psi0_decoupled_wbc train --data-format lerobot --data-dir data/evals/simple-eval/G1WholebodyXMovePickTeleop-v0/dr-level-0 --host 127.0.0.1 --port 22085 --sim-mode mujoco_isaac --headless --num-episodes 1 --episode-start 0 --num-workers 1 --third-person-video --save-video" > "$verification_dir/eval-command.txt"
set +e
SIMPLE_DISABLE_TUI=1 CUDA_VISIBLE_DEVICES=1 \
timeout --signal=INT --kill-after=75s 1800s \
  .venv/bin/eval-decoupled-wbc \
  simple/G1WholebodyXMovePickTeleop-v0 psi0_decoupled_wbc train \
  --data-format lerobot \
  --data-dir data/evals/simple-eval/G1WholebodyXMovePickTeleop-v0/dr-level-0 \
  --host 127.0.0.1 --port 22085 \
  --sim-mode mujoco_isaac --headless \
  --num-episodes 1 --episode-start 0 --num-workers 1 \
  --third-person-video --save-video
eval_exit_code=$?
set -e
printf '%s\n' "$eval_exit_code" > "$verification_dir/eval-exit-code.txt"
test "$eval_exit_code" -eq 0
```

Expected: normal success/failure verdict and exact artifact
`episode_0/third_person_<verdict>.mp4` beneath the approved decoupled output
root.

- [ ] **Step 6: Validate the final artifact**

Resolve exactly one verdict file, then run FFprobe without a placeholder:

```bash
mapfile -t third_person_paths < <(
  find data/evals/psi0_decoupled_wbc -type f \
    \( -name third_person_success.mp4 -o -name third_person_failed.mp4 \) \
    -newer "$verification_dir/health.json" -print
)
test "${#third_person_paths[@]}" -eq 1
third_person_path="${third_person_paths[0]}"
episode_dir="$(dirname "$third_person_path")"
printf '%s\n' "$third_person_path" > "$verification_dir/third-person-path.txt"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,nb_frames,duration \
  -of json "$third_person_path" > "$verification_dir/third-person-ffprobe.json"
test "$(find "$episode_dir" -maxdepth 1 -type f -name 'head_stereo_left_*.mp4' | wc -l)" -eq 1
test ! -e "$episode_dir/third_person.mp4"

ffmpeg -nostdin -y -i "$third_person_path" -vf "select=eq(n\\,0)" -frames:v 1 "$verification_dir/frame-first.png"
ffmpeg -nostdin -y -i "$third_person_path" -vf thumbnail -frames:v 1 "$verification_dir/frame-middle.png"
ffmpeg -nostdin -y -sseof -0.1 -i "$third_person_path" -frames:v 1 "$verification_dir/frame-last.png"
.venv/bin/python -c 'from pathlib import Path; import numpy as np; from PIL import Image; paths=[Path(p) for p in __import__("sys").argv[1:]]; frames=[np.asarray(Image.open(p).convert("RGB")) for p in paths]; assert all(f.shape == (360, 640, 3) for f in frames); assert all(np.unique(f.reshape(-1, 3), axis=0).shape[0] >= 32 for f in frames)' "$verification_dir/frame-first.png" "$verification_dir/frame-middle.png" "$verification_dir/frame-last.png"
```

Expected: width 640, height 360, and positive frame count/duration. Decode
first/middle/last frames, confirm non-empty scene content, confirm the ordinary
head-stereo artifact exists, and confirm raw `third_person.mp4` is absent after
successful finalization.

- [ ] **Step 7: Stop only owned infrastructure and audit cleanup**

Interrupt only the PIDs recorded by Step 4. Preserve the server log and PID
record, and leave the shared container and immutable source checkout running or
present for other users:

```bash
cleanup_owned_infrastructure
trap - EXIT INT TERM
test "$(cat "$verification_dir/server-alive-after-cleanup.txt")" = false
test "$(cat "$verification_dir/tunnel-cleanup-exit-code.txt")" -eq 0
test "$(cat "$verification_dir/server-cleanup-exit-code.txt")" -eq 0
case "$(cat "$verification_dir/tunnel-exit-code.txt")" in
  0|130|137|143) ;;
  *) false ;;
esac
! local_pid_alive "$tunnel_pid"
! remote_server_alive

pgrep -af 'eval-decoupled-wbc|eval-worker|ffmpeg|serve_psi0' || true
lsof -nP -iTCP:22085 -sTCP:LISTEN || true
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
```

Expected: no owned evaluator, child, FFmpeg, tunnel listener, or remote server.
Do not stop unrelated jobs.

- [ ] **Step 8: Save the verification manifest and final commit**

Write `manifest.json` in the run directory containing SIMPLE HEAD, PSI0 HEAD,
checkpoint and artifact hashes, literal commands, exit codes, selected
interfaces/domain, health/info responses, FFprobe JSON, frame paths, task
verdict, owned PIDs, and cleanup audit. Build and validate it with these exact
commands:

```bash
simple_commit="$(git rev-parse HEAD)"
recorded_psi0_commit="$(head -n 1 "$verification_dir/psi0-source.txt")"
test "$recorded_psi0_commit" = "$psi0_commit"
checkpoint_sha256="$(awk '{print $1}' "$verification_dir/checkpoint.sha256")"
third_person_sha256="$(sha256sum "$third_person_path" | awk '{print $1}')"
eval_exit_code="$(cat "$verification_dir/eval-exit-code.txt")"
tunnel_exit_code="$(cat "$verification_dir/tunnel-exit-code.txt")"
tunnel_cleanup_exit_code="$(cat "$verification_dir/tunnel-cleanup-exit-code.txt")"
server_cleanup_exit_code="$(cat "$verification_dir/server-cleanup-exit-code.txt")"
info_status="$(cat "$verification_dir/info-status.txt")"
server_pid="$(cat "$verification_dir/server.pid")"
tunnel_pid="$(cat "$verification_dir/tunnel.pid")"
server_alive_after_cleanup="$(cat "$verification_dir/server-alive-after-cleanup.txt")"
case "$(basename "$third_person_path")" in
  third_person_success.mp4) task_verdict=success ;;
  third_person_failed.mp4) task_verdict=failed ;;
  *) false ;;
esac
if lsof -nP -iTCP:22085 -sTCP:LISTEN >/dev/null; then
  local_listener_after_cleanup=true
else
  local_listener_after_cleanup=false
fi

jq -n \
  --arg schema simple.third-person-camera-verification.v1 \
  --arg simple_commit "$simple_commit" \
  --arg psi0_commit "$recorded_psi0_commit" \
  --arg checkpoint_path "$remote_run/checkpoints/ckpt_40000/model.safetensors" \
  --arg checkpoint_sha256 "$checkpoint_sha256" \
  --arg artifact_path "$third_person_path" \
  --arg artifact_sha256 "$third_person_sha256" \
  --arg task_verdict "$task_verdict" \
  --argjson eval_exit_code "$eval_exit_code" \
  --argjson tunnel_exit_code "$tunnel_exit_code" \
  --argjson tunnel_cleanup_exit_code "$tunnel_cleanup_exit_code" \
  --argjson server_cleanup_exit_code "$server_cleanup_exit_code" \
  --argjson info_status "$info_status" \
  --argjson server_pid "$server_pid" \
  --argjson tunnel_pid "$tunnel_pid" \
  --argjson server_alive_after_cleanup "$server_alive_after_cleanup" \
  --argjson local_listener_after_cleanup "$local_listener_after_cleanup" \
  --rawfile health "$verification_dir/health.json" \
  --rawfile info_response "$verification_dir/info-response.txt" \
  --rawfile server_command "$verification_dir/server-command.txt" \
  --rawfile eval_command "$verification_dir/eval-command.txt" \
  --slurpfile ffprobe "$verification_dir/third-person-ffprobe.json" \
  --slurpfile mujoco "$verification_dir/mujoco/report.json" \
  --slurpfile isaac "$verification_dir/isaac/report.json" \
  '{
    schema: $schema,
    simple_commit: $simple_commit,
    psi0_commit: $psi0_commit,
    checkpoint: {path: $checkpoint_path, sha256: $checkpoint_sha256},
    simulation: {interface: "lo", domain_id: 42, mode: "mujoco_isaac"},
    backend_reports: {mujoco: $mujoco[0], isaac: $isaac[0]},
    policy_server: {
      command: ($server_command | rtrimstr("\n")),
      health: ($health | rtrimstr("\n")),
      info_http_status: $info_status,
      info_response: ($info_response | rtrimstr("\n")),
      pid: $server_pid
    },
    evaluator: {
      command: ($eval_command | rtrimstr("\n")),
      exit_code: $eval_exit_code,
      task_verdict: $task_verdict
    },
    artifact: {
      path: $artifact_path,
      sha256: $artifact_sha256,
      ffprobe: $ffprobe[0],
      reviewed_frames: ["frame-first.png", "frame-middle.png", "frame-last.png"]
    },
    cleanup: {
      tunnel_pid: $tunnel_pid,
      tunnel_exit_code: $tunnel_exit_code,
      tunnel_cleanup_exit_code: $tunnel_cleanup_exit_code,
      server_cleanup_exit_code: $server_cleanup_exit_code,
      server_alive: $server_alive_after_cleanup,
      local_listener: $local_listener_after_cleanup
    }
  }' > "$verification_dir/manifest.json"

jq -e '[.. | select(. == null)] | length == 0' "$verification_dir/manifest.json"
jq -e '.evaluator.exit_code == 0 and .cleanup.tunnel_cleanup_exit_code == 0 and .cleanup.server_cleanup_exit_code == 0 and .cleanup.server_alive == false and .cleanup.local_listener == false' "$verification_dir/manifest.json"
jq -e 'all(.backend_reports[]; .marker_validation.marker_order_ok == true and .marker_validation.center_projection_ok == true and .marker_validation.clipping_ok == true and .effective_clipping_ok == true and .effective_clipping == [0.2, 5.0])' "$verification_dir/manifest.json"
jq -e '.artifact.ffprobe.streams | length == 1 and .[0].width == 640 and .[0].height == 360 and (.[0].nb_frames | tonumber) > 0 and (.[0].duration | tonumber) > 0' "$verification_dir/manifest.json"
git status --short
git log --oneline --decorate -12
git diff --check 83c175c..HEAD
git diff --check
```

Expected: implementation/tests are committed; only preserved `outputs/`
evidence is untracked. Do not commit model weights or generated videos.
