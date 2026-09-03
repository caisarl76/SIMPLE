import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from scripts import postprocess_psi0 as converter


def _valid_vectors(row_count):
    return {
        "states": np.zeros((row_count, 32), np.float32),
        "action": np.zeros((row_count, 36), np.float32),
        "observation.hand_joints": np.zeros((row_count, 14), np.float32),
        "observation.arm_joints": np.zeros((row_count, 14), np.float32),
        "observation.leg_joints": np.zeros((row_count, 15), np.float32),
        "observation.prev_torso_rpy": np.zeros((row_count, 3), np.float32),
        "observation.prev_height": np.full((row_count, 1), 0.74, np.float32),
    }


def test_retained_indices_are_the_only_cardinality_source():
    selected = converter.make_retained_indices(214, skip=60, downsample=4)
    np.testing.assert_array_equal(selected, np.arange(60, 214, 4, dtype=np.int64))
    assert len(selected) == 39


@pytest.mark.parametrize("frame_count", [59, 60])
def test_zero_retained_rows_fail(frame_count):
    with pytest.raises(ValueError, match="frame_count must be greater than skip"):
        converter.make_retained_indices(frame_count, skip=60, downsample=1)


def test_output_schema_is_exact():
    schema = converter.output_schema()
    expected = {
        "states": pa.list_(pa.float32(), 32),
        "action": pa.list_(pa.float32(), 36),
        "observation.hand_joints": pa.list_(pa.float32(), 14),
        "observation.arm_joints": pa.list_(pa.float32(), 14),
        "observation.leg_joints": pa.list_(pa.float32(), 15),
        "observation.prev_torso_rpy": pa.list_(pa.float32(), 3),
        "observation.prev_height": pa.list_(pa.float32(), 1),
        "timestamp": pa.float32(),
        "frame_index": pa.int64(),
        "episode_index": pa.int64(),
        "index": pa.int64(),
        "task_index": pa.int64(),
        "next.done": pa.bool_(),
    }
    assert schema.names == list(expected)
    assert {field.name: field.type for field in schema} == expected


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_finite_array_rejects_nonfinite_values(bad):
    values = np.zeros((2, 32), dtype=np.float32)
    values[1, 3] = bad
    with pytest.raises(ValueError, match="states contains nonfinite values"):
        converter.require_finite("states", values)


def test_stats_are_retained_only_and_include_count():
    source = np.arange(8, dtype=np.float32).reshape(4, 2)
    retained = source[[1, 3]]
    block = converter.stats_block(retained)
    assert block["count"] == [2]
    np.testing.assert_allclose(block["mean"], [4.0, 5.0])
    assert block["mean"] != converter.stats_block(source)["mean"]
    json.dumps(block, allow_nan=False)


def test_row_table_uses_exact_indices_timestamps_and_terminal_flag():
    vectors = {
        "states": np.zeros((3, 32), np.float32),
        "action": np.zeros((3, 36), np.float32),
        "observation.hand_joints": np.zeros((3, 14), np.float32),
        "observation.arm_joints": np.zeros((3, 14), np.float32),
        "observation.leg_joints": np.zeros((3, 15), np.float32),
        "observation.prev_torso_rpy": np.zeros((3, 3), np.float32),
        "observation.prev_height": np.full((3, 1), 0.74, np.float32),
    }
    table = converter.build_output_table(
        vectors=vectors,
        output_episode_index=2,
        global_offset=5,
        output_task_index=1,
        output_fps=Fraction(50, 1),
    )
    assert table.schema == converter.output_schema()
    assert table["frame_index"].to_pylist() == [0, 1, 2]
    assert table["episode_index"].to_pylist() == [2, 2, 2]
    assert table["index"].to_pylist() == [5, 6, 7]
    assert table["task_index"].to_pylist() == [1, 1, 1]
    assert table["next.done"].to_pylist() == [False, False, True]
    assert table["timestamp"].type == pa.float32()
    np.testing.assert_array_equal(
        np.asarray(table["timestamp"]),
        np.asarray([0.0, 0.02, 0.04], dtype=np.float32),
    )


def test_row_table_timestamps_round_exact_fraction_quotients_once():
    output_fps = Fraction(30000, 1001)
    table = converter.build_output_table(
        vectors=_valid_vectors(11),
        output_episode_index=0,
        global_offset=0,
        output_task_index=0,
        output_fps=output_fps,
    )
    expected = np.asarray(
        [Fraction(i, 1) / output_fps for i in range(11)], dtype=np.float32
    )
    np.testing.assert_array_equal(np.asarray(table["timestamp"]), expected)


@pytest.mark.parametrize("output_fps", [Fraction(0, 1), Fraction(-1, 1)])
def test_row_table_rejects_nonpositive_fps(output_fps):
    with pytest.raises(ValueError, match="output_fps must be positive"):
        converter.build_output_table(
            vectors=_valid_vectors(2),
            output_episode_index=0,
            global_offset=0,
            output_task_index=0,
            output_fps=output_fps,
        )


def test_row_table_rejects_timestamp_float32_overflow():
    with pytest.raises(ValueError, match="timestamp contains nonfinite values"):
        converter.build_output_table(
            vectors=_valid_vectors(2),
            output_episode_index=0,
            global_offset=0,
            output_task_index=0,
            output_fps=Fraction(1, 10**1000),
        )


def test_row_table_rejects_wrong_vector_width():
    vectors = _valid_vectors(3)
    vectors["action"] = np.zeros((3, 35), np.float32)
    with pytest.raises(ValueError, match=r"action must have shape \(3, 36\)"):
        converter.build_output_table(
            vectors=vectors,
            output_episode_index=0,
            global_offset=0,
            output_task_index=0,
            output_fps=Fraction(50, 1),
        )


def test_row_table_rejects_mismatched_vector_row_cardinality():
    vectors = _valid_vectors(3)
    vectors["action"] = np.zeros((2, 36), np.float32)
    with pytest.raises(ValueError, match=r"action must have shape \(3, 36\)"):
        converter.build_output_table(
            vectors=vectors,
            output_episode_index=0,
            global_offset=0,
            output_task_index=0,
            output_fps=Fraction(50, 1),
        )


def test_atomic_write_new_bytes_is_exclusive_and_durable(tmp_path, monkeypatch):
    path = tmp_path / "payload.json"
    fsynced = []
    real_fsync = converter.os.fsync

    def record_fsync(fd):
        fsynced.append(converter.os.fstat(fd).st_ino)
        real_fsync(fd)

    monkeypatch.setattr(converter.os, "fsync", record_fsync)
    converter.atomic_write_new_bytes(path, b'{"finite":true}\n')

    assert path.read_bytes() == b'{"finite":true}\n'
    assert len(fsynced) >= 3  # payload, parent after link, parent after temp removal
    assert list(tmp_path.glob(".payload.json.tmp-*")) == []

    with pytest.raises(FileExistsError):
        converter.atomic_write_new_bytes(path, b"replacement\n")
    assert path.read_bytes() == b'{"finite":true}\n'
    assert list(tmp_path.glob(".payload.json.tmp-*")) == []


def test_atomic_write_new_bytes_does_not_follow_final_symlink(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"target\n")
    path = tmp_path / "payload.json"
    path.symlink_to(target.name)

    with pytest.raises(FileExistsError):
        converter.atomic_write_new_bytes(path, b"replacement\n")

    assert target.read_bytes() == b"target\n"
    assert path.is_symlink()
    assert list(tmp_path.glob(".payload.json.tmp-*")) == []


def test_atomic_write_rejects_temporary_path_swap_at_link(tmp_path, monkeypatch):
    path = tmp_path / "payload.json"
    expected = b'{"source":"trusted"}\n'
    attacker = b'{"source":"attacker"}\n'
    real_link = converter.os.link
    attacked = False

    def swap_then_link(source, destination, **kwargs):
        nonlocal attacked
        attacked = True
        source_dir_fd = kwargs.get("src_dir_fd")
        if source_dir_fd is None:
            converter.os.unlink(source)
            Path(source).write_bytes(attacker)
        else:
            converter.os.unlink(source, dir_fd=source_dir_fd)
            fd = converter.os.open(
                source,
                converter.os.O_WRONLY
                | converter.os.O_CREAT
                | converter.os.O_EXCL
                | converter.os.O_CLOEXEC
                | converter.os.O_NOFOLLOW,
                0o644,
                dir_fd=source_dir_fd,
            )
            try:
                converter.os.write(fd, attacker)
            finally:
                converter.os.close(fd)
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(converter.os, "link", swap_then_link)
    with pytest.raises(RuntimeError, match="identity|digest|payload"):
        converter.atomic_write_new_bytes(path, expected)

    assert attacked
    assert not path.exists() or path.read_bytes() != expected


def test_atomic_write_rejects_parent_path_swap_at_link(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    saved_parent = tmp_path / "saved-parent"
    path = parent / "payload.json"
    expected = b'{"source":"trusted"}\n'
    attacker = b'{"source":"attacker"}\n'
    real_link = converter.os.link
    attacked = False

    def swap_parent_then_link(source, destination, **kwargs):
        nonlocal attacked
        attacked = True
        temp_name = Path(converter.os.fsdecode(source)).name
        parent.rename(saved_parent)
        parent.mkdir()
        (parent / temp_name).write_bytes(attacker)
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(converter.os, "link", swap_parent_then_link)
    with pytest.raises(RuntimeError, match="identity changed"):
        converter.atomic_write_new_bytes(path, expected)

    assert attacked
    assert not path.exists() or path.read_bytes() != expected


def test_atomic_write_closes_payload_fd_when_cleanup_fails(tmp_path, monkeypatch):
    path = tmp_path / "payload.json"
    descriptors_before = set(converter.os.listdir("/proc/self/fd"))
    real_open = converter.os.open
    real_close = converter.os.close
    real_unlink = converter.os.unlink
    real_path_unlink = Path.unlink
    payload_fd = None
    closed = []

    def recording_open(name, flags, *args, **kwargs):
        nonlocal payload_fd
        fd = real_open(name, flags, *args, **kwargs)
        if flags & converter.os.O_EXCL:
            payload_fd = fd
        return fd

    def recording_close(fd):
        closed.append(fd)
        real_close(fd)

    def fail_temp_unlink(name, *args, **kwargs):
        if Path(converter.os.fsdecode(name)).name.startswith(".payload.json.tmp-"):
            raise OSError("injected cleanup failure")
        return real_unlink(name, *args, **kwargs)

    def fail_temp_path_unlink(name, *args, **kwargs):
        if name.name.startswith(".payload.json.tmp-"):
            raise OSError("injected cleanup failure")
        return real_path_unlink(name, *args, **kwargs)

    monkeypatch.setattr(converter.os, "open", recording_open)
    monkeypatch.setattr(converter.os, "close", recording_close)
    monkeypatch.setattr(converter.os, "unlink", fail_temp_unlink)
    monkeypatch.setattr(Path, "unlink", fail_temp_path_unlink)
    try:
        with pytest.raises(OSError, match="injected cleanup failure"):
            converter.atomic_write_new_bytes(path, b"payload\n")
        assert payload_fd is not None
        assert payload_fd in closed
        assert set(converter.os.listdir("/proc/self/fd")) == descriptors_before
    finally:
        if payload_fd is not None and payload_fd not in closed:
            real_close(payload_fd)
