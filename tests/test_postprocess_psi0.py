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
    assert (
        result["source_parquet_sha256"]
        == hashlib.sha256(b"raw-episode-seven").hexdigest()
    )
    assert result["skip"] == 60
    assert result["downsample"] == 2
    assert result["converter_commit"] == "1" * 40
