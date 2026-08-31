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
    binary = (
        Path("native/psi0_eval_runtime/target/x86_64-unknown-linux-gnu/release") / name
    )
    assert binary.is_file()
    program = subprocess.run(
        ["readelf", "-lWd", binary], check=True, text=True, capture_output=True
    )
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
