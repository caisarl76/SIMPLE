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
        "psi0-eval-install-input",
        "psi0-eval-install-pc2-input",
        "psi0-eval-remote-helper",
        "psi0-eval-run-pc2-evaluator",
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
        if words == [
            "cargo",
            "metadata",
            "--offline",
            "--locked",
            "--format-version",
            "1",
        ]:
            if cwd == (repo / "native/psi0_eval_runtime").resolve():
                return real_popen(argv, *args, **kwargs)
        try:
            binary = Path(words[0]).resolve(strict=True)
        except FileNotFoundError:
            return blocked(argv, *args, **kwargs)
        if (
            binary.parent == release
            and binary.name in native_names
            and "--test-root" in words
        ):
            return real_popen(argv, *args, **kwargs)
        return blocked(argv, *args, **kwargs)

    monkeypatch.setattr("socket.create_connection", blocked)
    monkeypatch.setattr("subprocess.Popen", constrained_popen)
    monkeypatch.setenv("SIMPLE_EVAL_RUNTIME_UNIT_TEST", "1")
