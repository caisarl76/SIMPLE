import subprocess
from pathlib import Path

import numpy as np
import pytest

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

    states, _ = build_vectors(proprio, cmd, history, action, np.zeros(3), np.zeros(3))
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


def test_converter_identity_rejects_dirty_script_bytes(tmp_path):
    from scripts import postprocess_psi0

    script = tmp_path / "scripts" / "postprocess_psi0.py"
    script.parent.mkdir()
    script.write_bytes(Path(postprocess_psi0.__file__).read_bytes())
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True
    )
    subprocess.run(
        ["git", "add", str(script.relative_to(tmp_path))], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "commit", "-qm", "add converter"], cwd=tmp_path, check=True)

    identity = postprocess_psi0.resolve_converter_identity(tmp_path, script)
    assert (
        identity.commit
        == subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert identity.script_sha256 == postprocess_psi0.sha256_bytes(script.read_bytes())

    script.write_bytes(script.read_bytes() + b"#")
    with pytest.raises(
        RuntimeError, match="executed converter differs from its recorded Git blob"
    ):
        postprocess_psi0.resolve_converter_identity(tmp_path, script)
