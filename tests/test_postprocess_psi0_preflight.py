import argparse
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from psi0_converter_fixtures import make_source_episode
from scripts import postprocess_psi0 as converter


IDENTITY = converter.ConverterIdentity(commit="0" * 40, script_sha256="1" * 64)
VIDEO_KEY = "observation.rgb_head_stereo_left"
REQUIRED_SOURCE_COLUMNS = (
    "observation.joint_qpos",
    "observation.amo_policy_command",
    "observation.amo_policy_target_yaw",
    "observation.amo_policy_turning_flag",
    "action",
    "task_index",
)


def assert_preflight_rejected_without_output(call, output: Path) -> None:
    parent = output.parent
    before = sorted(
        (path.relative_to(parent), path.lstat().st_mode, path.lstat().st_size)
        for path in parent.rglob("*")
    )
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        call()
    after = sorted(
        (path.relative_to(parent), path.lstat().st_mode, path.lstat().st_size)
        for path in parent.rglob("*")
    )
    assert after == before
    assert not output.exists()
    assert not list(parent.glob(f".{output.name}.staging-*"))


def _args(source: Path | str, output: Path, **overrides: object) -> argparse.Namespace:
    values = {
        "sim_root": str(source),
        "out_dir": str(output),
        "skip": 0,
        "downsample": 1,
        "total_episodes": 100,
        "fps": "4",
        "video_key": VIDEO_KEY,
        "chunks_size": 1000,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _preflight(source: Path | str, output: Path, **overrides: object):
    return converter.preflight_conversion(_args(source, output, **overrides), IDENTITY)


def _parquet(root: Path, episode_index: int = 0) -> Path:
    return root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"


def _video(root: Path, episode_index: int = 0) -> Path:
    return (
        root / "videos" / "chunk-000" / VIDEO_KEY / f"episode_{episode_index:06d}.mp4"
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    return table.set_column(index, name, values)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("skip", -1),
        ("downsample", 0),
        ("downsample", -1),
        ("fps", "0"),
        ("fps", "-1"),
        ("fps", "nan"),
        ("fps", "inf"),
        ("chunks_size", 0),
        ("chunks_size", -1),
        ("total_episodes", 0),
        ("total_episodes", -1),
    ],
)
def test_preflight_rejects_invalid_global_arguments(tmp_path, name, value):
    source = make_source_episode(tmp_path / "source")
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(
        lambda: _preflight(source, output, **{name: value}), output
    )


def test_preflight_rejects_timestamp_float32_overflow(tmp_path):
    source = make_source_episode(tmp_path / "source")
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(
        lambda: _preflight(source, output, fps=f"1/{10**100}"), output
    )


def test_preflight_rejects_missing_source_root(tmp_path):
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(
        lambda: _preflight(tmp_path / "missing", output), output
    )


@pytest.mark.parametrize("name", ["info.json", "tasks.jsonl", "episodes.jsonl"])
def test_preflight_rejects_missing_metadata(tmp_path, name):
    source = make_source_episode(tmp_path / "source")
    (source / "meta" / name).unlink()
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(lambda: _preflight(source, output), output)


def test_preflight_rejects_existing_destination(tmp_path):
    source = make_source_episode(tmp_path / "source")
    output = tmp_path / "output"
    output.symlink_to(tmp_path / "missing-target")
    assert_preflight_rejected_without_output(lambda: _preflight(source, output), output)
    assert output.is_symlink()


@pytest.mark.parametrize("column", REQUIRED_SOURCE_COLUMNS)
def test_preflight_rejects_each_missing_required_column(tmp_path, column):
    source = make_source_episode(tmp_path / "source")
    path = _parquet(source)
    table = pq.read_table(path)
    pq.write_table(table.drop([column]), path)
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(lambda: _preflight(source, output), output)


def test_preflight_rejects_scalar_instead_of_list(tmp_path):
    source = make_source_episode(tmp_path / "source")
    path = _parquet(source)
    table = pq.read_table(path)
    table = _replace_column(
        table,
        "observation.joint_qpos",
        pa.array(np.arange(len(table), dtype=np.float32)),
    )
    pq.write_table(table, path)
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(lambda: _preflight(source, output), output)


def test_preflight_rejects_list_instead_of_scalar(tmp_path):
    source = make_source_episode(tmp_path / "source")
    path = _parquet(source)
    table = pq.read_table(path)
    values = pa.FixedSizeListArray.from_arrays(
        pa.array(np.zeros(len(table), dtype=np.float32)), 1
    )
    table = _replace_column(table, "observation.amo_policy_target_yaw", values)
    pq.write_table(table, path)
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(lambda: _preflight(source, output), output)


def test_preflight_rejects_wrong_vector_width(tmp_path):
    source = make_source_episode(tmp_path / "source")
    path = _parquet(source)
    table = pq.read_table(path)
    values = pa.FixedSizeListArray.from_arrays(
        pa.array(np.zeros(len(table) * 42, dtype=np.float32)), 42
    )
    table = _replace_column(table, "observation.joint_qpos", values)
    pq.write_table(table, path)
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(lambda: _preflight(source, output), output)


def test_preflight_rejects_wrong_source_dtype(tmp_path):
    source = make_source_episode(tmp_path / "source")
    path = _parquet(source)
    table = pq.read_table(path)
    values = pa.FixedSizeListArray.from_arrays(
        pa.array(np.zeros(len(table) * 43, dtype=np.float64)), 43
    )
    table = _replace_column(table, "action", values)
    pq.write_table(table, path)
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(lambda: _preflight(source, output), output)


def test_preflight_rejects_task_index_row_count_mismatch(tmp_path, monkeypatch):
    source = make_source_episode(tmp_path / "source")
    output = tmp_path / "output"
    real_read_table = converter.pq.read_table

    def shortened_task_index(path, *, columns=None):
        table = real_read_table(path, columns=columns)
        return table.slice(0, len(table) - 1)

    monkeypatch.setattr(converter.pq, "read_table", shortened_task_index)
    assert_preflight_rejected_without_output(lambda: _preflight(source, output), output)


def test_preflight_rejects_non_int64_task_index(tmp_path):
    source = make_source_episode(tmp_path / "source")
    path = _parquet(source)
    table = pq.read_table(path)
    table = _replace_column(
        table, "task_index", pa.array(np.zeros(len(table), dtype=np.int32))
    )
    pq.write_table(table, path)
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(lambda: _preflight(source, output), output)


def test_preflight_rejects_nonconstant_task_index(tmp_path):
    source = make_source_episode(tmp_path / "source")
    path = _parquet(source)
    table = pq.read_table(path)
    values = np.zeros(len(table), dtype=np.int64)
    values[-1] = 1
    table = _replace_column(table, "task_index", pa.array(values))
    pq.write_table(table, path)
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(lambda: _preflight(source, output), output)


@pytest.mark.parametrize("skip", [8, 9])
def test_preflight_rejects_zero_retained_rows(tmp_path, skip):
    source = make_source_episode(tmp_path / "source", frames=8)
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(
        lambda: _preflight(source, output, skip=skip), output
    )


@pytest.mark.parametrize(
    "rows",
    [
        [
            {"task_index": 0, "task": "test task"},
            {"task_index": 0, "task": "another task"},
        ],
        [{"task_index": -1, "task": "test task"}],
        [{"task_index": 0.5, "task": "test task"}],
        [{"task_index": True, "task": "test task"}],
        [{"task_index": 0, "task": ""}],
        [{"task_index": 0, "task": "   "}],
        [{"task_index": 0, "task": 3}],
        [{"task_index": 1, "task": "test task"}],
    ],
)
def test_preflight_rejects_malformed_or_missing_task_metadata(tmp_path, rows):
    source = make_source_episode(tmp_path / "source")
    _write_jsonl(source / "meta" / "tasks.jsonl", rows)
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(lambda: _preflight(source, output), output)


def _episode_row(source: Path) -> dict[str, object]:
    return _read_jsonl(source / "meta" / "episodes.jsonl")[0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("episode_index", -1),
        ("episode_index", 0.5),
        ("episode_index", True),
        ("length", 0),
        ("length", -1),
        ("length", 7),
        ("length", 8.0),
        ("tasks", []),
        ("tasks", [""]),
        ("tasks", [3]),
        ("tasks", "test task"),
        ("environment_config", "not json"),
        ("environment_config", "NaN"),
        ("environment_config", {}),
        ("tasks", ["different task"]),
        ("episode_index", 1),
    ],
)
def test_preflight_rejects_malformed_or_missing_episode_metadata(
    tmp_path, field, value
):
    source = make_source_episode(tmp_path / "source")
    row = _episode_row(source)
    row[field] = value
    _write_jsonl(source / "meta" / "episodes.jsonl", [row])
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(lambda: _preflight(source, output), output)


def test_preflight_rejects_duplicate_episode_metadata(tmp_path):
    source = make_source_episode(tmp_path / "source")
    row = _episode_row(source)
    _write_jsonl(source / "meta" / "episodes.jsonl", [row, row])
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(lambda: _preflight(source, output), output)


def test_preflight_rejects_missing_video(tmp_path):
    source = make_source_episode(tmp_path / "source")
    _video(source).unlink()
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(lambda: _preflight(source, output), output)


def test_preflight_rejects_duplicate_video_candidate(tmp_path):
    source = make_source_episode(tmp_path / "source")
    duplicate = source / "videos" / "chunk-001" / VIDEO_KEY
    duplicate.mkdir(parents=True)
    (duplicate / "episode_000000.mp4").write_bytes(_video(source).read_bytes())
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(lambda: _preflight(source, output), output)


def test_preflight_rejects_video_without_video_stream(tmp_path):
    source = make_source_episode(tmp_path / "source")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=1",
            "-c:a",
            "aac",
            str(_video(source)),
        ],
        check=True,
    )
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(lambda: _preflight(source, output), output)


def _ffprobe_payload(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("avg_frame_rate", "5/1"),
        ("avg_frame_rate", "nan"),
        ("avg_frame_rate", "0/1"),
        ("r_frame_rate", "nan"),
        ("r_frame_rate", "0/0"),
    ],
)
def test_preflight_rejects_ambiguous_nonfinite_or_zero_video_rate(
    tmp_path, monkeypatch, field, value
):
    source = make_source_episode(tmp_path / "source")
    payload = _ffprobe_payload(_video(source))
    video_stream = next(
        stream for stream in payload["streams"] if stream["codec_type"] == "video"
    )
    video_stream[field] = value

    def fake_run(argv, **kwargs):
        assert argv[0] == "ffprobe"
        return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")

    monkeypatch.setattr(converter.subprocess, "run", fake_run)
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(lambda: _preflight(source, output), output)


def test_preflight_rejects_video_frame_count_mismatch(tmp_path):
    source = make_source_episode(tmp_path / "source", frames=8)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=4",
            "-frames:v",
            "7",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(_video(source)),
        ],
        check=True,
    )
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(lambda: _preflight(source, output), output)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("codec_name", ""),
        ("sample_fmt", None),
        ("sample_rate", "not-an-integer"),
        ("sample_rate", "0"),
        ("channels", 0),
        ("channel_layout", ""),
    ],
)
def test_preflight_rejects_malformed_audio_identity(
    tmp_path, monkeypatch, field, value
):
    source = make_source_episode(tmp_path / "source")
    payload = _ffprobe_payload(_video(source))
    audio = {
        "codec_type": "audio",
        "codec_name": "aac",
        "sample_fmt": "fltp",
        "sample_rate": "48000",
        "channels": 2,
        "channel_layout": "stereo",
    }
    audio[field] = value
    payload["streams"].append(audio)

    def fake_run(argv, **kwargs):
        assert argv[0] == "ffprobe"
        return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")

    monkeypatch.setattr(converter.subprocess, "run", fake_run)
    output = tmp_path / "output"
    assert_preflight_rejected_without_output(lambda: _preflight(source, output), output)


def test_probe_media_returns_canonical_identity(tmp_path):
    source = make_source_episode(tmp_path / "source")
    identity = converter.probe_media(_video(source))
    assert identity.codec_name == "h264"
    assert identity.pixel_format == "yuv420p"
    assert (identity.width, identity.height) == (64, 48)
    assert identity.average_frame_rate == "4"
    assert identity.nominal_frame_rate == "4"
    assert Fraction(identity.duration) > 0
    assert identity.frame_count == 8
    assert identity.audio_streams == ()


def test_preflight_orders_sources_and_remaps_tasks_by_first_episode_occurrence(
    tmp_path,
):
    source_b = make_source_episode(
        tmp_path / "source-b",
        episode_index=9,
        task_index=5,
        task_text="shared task",
    )
    source_a = make_source_episode(
        tmp_path / "source-a",
        episode_index=7,
        task_index=3,
        task_text="first task",
    )
    make_source_episode(
        source_a,
        episode_index=2,
        task_index=8,
        task_text="shared task",
    )
    episode_rows = _read_jsonl(source_a / "meta" / "episodes.jsonl")
    episode_rows.append(
        {
            "episode_index": 7,
            "tasks": ["first task"],
            "length": 8,
            "environment_config": "{}",
        }
    )
    _write_jsonl(source_a / "meta" / "episodes.jsonl", list(reversed(episode_rows)))
    task_rows = _read_jsonl(source_a / "meta" / "tasks.jsonl")
    task_rows.insert(0, {"task_index": 3, "task": "first task"})
    _write_jsonl(source_a / "meta" / "tasks.jsonl", list(reversed(task_rows)))

    output = tmp_path / "output"
    plan = _preflight(str(tmp_path / "source-*"), output)

    assert plan.output_path == output.resolve()
    assert plan.output_fps == Fraction(4, 1)
    assert [episode.source_root for episode in plan.episodes] == [
        source_a.resolve(),
        source_a.resolve(),
        source_b.resolve(),
    ]
    assert [episode.source_episode_index for episode in plan.episodes] == [2, 7, 9]
    assert [episode.output_episode_index for episode in plan.episodes] == [0, 1, 2]
    assert [episode.task_text for episode in plan.episodes] == [
        "shared task",
        "first task",
        "shared task",
    ]
    assert [episode.output_task_index for episode in plan.episodes] == [0, 1, 0]
    assert plan.tasks == (
        {"task_index": 0, "task": "shared task"},
        {"task_index": 1, "task": "first task"},
    )
    assert [episode.environment_config for episode in plan.episodes] == ["{}"] * 3
    assert all(
        not episode.retained_indices.flags.writeable for episode in plan.episodes
    )
