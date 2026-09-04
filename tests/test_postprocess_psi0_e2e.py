import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from psi0_converter_fixtures import make_source_episode
from scripts import postprocess_psi0 as converter


VECTOR_WIDTHS = {
    "states": 32,
    "action": 36,
    "observation.hand_joints": 14,
    "observation.arm_joints": 14,
    "observation.leg_joints": 15,
    "observation.prev_torso_rpy": 3,
    "observation.prev_height": 1,
}

GLOBAL_STATS_FIELDS = {
    "states",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "next.done",
}


def _strict_json(payload: bytes):
    def reject_constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    return json.loads(payload, parse_constant=reject_constant)


def _strict_jsonl(path: Path):
    return [_strict_json(line) for line in path.read_bytes().splitlines() if line]


def _make_plan(root: Path):
    make_source_episode(
        root / "source-a",
        episode_index=5,
        frames=6,
        fps=4,
        task_index=7,
        task_text="first complete instruction",
    )
    make_source_episode(
        root / "source-b",
        episode_index=2,
        frames=7,
        fps=4,
        task_index=11,
        task_text="second complete instruction",
    )
    args = SimpleNamespace(
        skip=1,
        downsample=2,
        chunks_size=1,
        total_episodes=2,
        fps="4",
        video_key="observation.rgb_head_stereo_left",
        out_dir=str(root / "processed"),
        sim_root=str(root / "source-*"),
    )
    identity = converter.ConverterIdentity("a" * 40, "b" * 64)
    return converter.preflight_conversion(args, identity)


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    return _make_plan(tmp_path_factory.mktemp("staged-plan"))


def _column_values(table: pa.Table, name: str) -> np.ndarray:
    if name in VECTOR_WIDTHS:
        return np.asarray(table[name].to_pylist(), dtype=np.float32)
    return np.asarray(table[name].to_pylist())


def _media_dict(value):
    return json.loads(json.dumps(asdict(value), allow_nan=False))


def _independent_stats(values):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    return {
        "mean": array.mean(0, dtype=np.float64).astype(np.float32).tolist(),
        "std": array.std(0, dtype=np.float64).astype(np.float32).tolist(),
        "min": array.min(0).astype(np.float32).tolist(),
        "max": array.max(0).astype(np.float32).tolist(),
        "q01": np.quantile(array, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).astype(np.float32).tolist(),
        "count": [len(array)],
    }


def _parquet_bytes_with_qpos_offset(path, offset):
    table = pq.read_table(path)
    qpos = np.asarray(table["observation.joint_qpos"].to_pylist(), dtype=np.float32)
    qpos += np.float32(offset)
    flattened = pa.array(qpos.reshape(-1), type=pa.float32())
    replacement = pa.FixedSizeListArray.from_arrays(flattened, 43)
    table = table.set_column(
        table.schema.get_field_index("observation.joint_qpos"),
        "observation.joint_qpos",
        replacement,
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def _parquet_bytes_with_task_index(path, task_index):
    table = pq.read_table(path)
    replacement = pa.array(
        np.full(table.num_rows, task_index, dtype=np.int64), type=pa.int64()
    )
    table = table.set_column(
        table.schema.get_field_index("task_index"), "task_index", replacement
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def _expected_tree(plan, staging):
    result = {
        "CONVERSION_STATUS.json",
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
        "meta/stats.json",
        "meta/stats_psi0.json",
        "meta/relative_stats.json",
        "meta/lang_map.json",
        "meta/modality.json",
        "meta/conversion_provenance.json",
    }
    for episode in plan.episodes:
        chunk = episode.output_episode_index // plan.chunks_size
        result.add(
            f"data/chunk-{chunk:03d}/episode_{episode.output_episode_index:06d}.parquet"
        )
        result.add(
            "videos/"
            f"chunk-{chunk:03d}/egocentric/episode_{episode.output_episode_index:06d}.mp4"
        )
    actual = {
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    assert actual == result
    assert not (staging / "meta" / "conversion_manifest.json").exists()


def test_generate_staged_dataset_emits_exact_retained_contract(tmp_path, plan):
    staging = tmp_path / ".processed.staging-00000000"
    converter.generate_staged_dataset(plan, staging)

    _expected_tree(plan, staging)
    status = _strict_json((staging / "CONVERSION_STATUS.json").read_bytes())
    staging_identity = staging.stat()
    assert status == {
        "schema_version": 1,
        "staging": {
            "st_dev": staging_identity.st_dev,
            "st_ino": staging_identity.st_ino,
        },
        "state": "in_progress",
    }

    task_rows = _strict_jsonl(staging / "meta" / "tasks.jsonl")
    assert task_rows == [
        {
            "category": "",
            "description": task["task"],
            "task": task["task"],
            "task_index": task["task_index"],
        }
        for task in plan.tasks
    ]
    task_by_index = {row["task_index"]: row for row in task_rows}

    episode_rows = _strict_jsonl(staging / "meta" / "episodes.jsonl")
    episode_stats = _strict_jsonl(staging / "meta" / "episodes_stats.jsonl")
    assert len(episode_rows) == len(plan.episodes)
    assert len(episode_stats) == len(plan.episodes)

    global_offset = 0
    global_arrays = {name: [] for name in GLOBAL_STATS_FIELDS}
    episode_provenance_digests = []
    output_video_count = 0
    for episode, row, stats_row in zip(
        plan.episodes, episode_rows, episode_stats, strict=True
    ):
        chunk = episode.output_episode_index // plan.chunks_size
        parquet_path = staging / (
            f"data/chunk-{chunk:03d}/episode_{episode.output_episode_index:06d}.parquet"
        )
        table = pq.read_table(parquet_path)
        retained_count = len(episode.retained_indices)
        assert table.schema == converter.output_schema()
        assert table.num_rows == retained_count
        assert all(len(table[name]) == retained_count for name in table.column_names)
        for name in set(VECTOR_WIDTHS) | {"timestamp"}:
            assert np.isfinite(_column_values(table, name)).all()

        assert table["frame_index"].to_pylist() == list(range(retained_count))
        assert (
            table["episode_index"].to_pylist()
            == [episode.output_episode_index] * retained_count
        )
        assert table["index"].to_pylist() == list(
            range(global_offset, global_offset + retained_count)
        )
        assert (
            table["task_index"].to_pylist()
            == [episode.output_task_index] * retained_count
        )
        assert table["next.done"].to_pylist() == [False] * (retained_count - 1) + [True]
        expected_timestamp = np.asarray(
            [Fraction(i, 1) / plan.output_fps for i in range(retained_count)],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            np.asarray(table["timestamp"]), expected_timestamp
        )

        source = pq.read_table(episode.parquet_path)
        qpos = np.asarray(
            source["observation.joint_qpos"].to_pylist(), dtype=np.float32
        )
        command = np.asarray(
            source["observation.amo_policy_command"].to_pylist(), dtype=np.float32
        )
        target_yaw = np.asarray(
            source["observation.amo_policy_target_yaw"], dtype=np.float32
        )
        turning = np.asarray(
            source["observation.amo_policy_turning_flag"], dtype=np.float32
        )
        action = np.asarray(source["action"].to_pylist(), dtype=np.float32)
        history = np.concatenate(
            [converter.initial_command[None], command[:-1]], axis=0
        )
        states, actions = converter.build_vectors(
            qpos, command, history, action, target_yaw, turning
        )
        hand, arm, leg, torso, height = converter.build_proprio_obs(qpos, history)
        expected_vectors = {
            "states": states[episode.retained_indices],
            "action": actions[episode.retained_indices],
            "observation.hand_joints": hand[episode.retained_indices],
            "observation.arm_joints": arm[episode.retained_indices],
            "observation.leg_joints": leg[episode.retained_indices],
            "observation.prev_torso_rpy": torso[episode.retained_indices],
            "observation.prev_height": height[episode.retained_indices],
        }
        for name, expected in expected_vectors.items():
            np.testing.assert_array_equal(_column_values(table, name), expected)

        assert set(stats_row) == {"episode_index", "stats"}
        assert stats_row["episode_index"] == episode.output_episode_index
        assert set(stats_row["stats"]) == {"action", "timestamp"}
        assert stats_row["stats"] == {
            "action": _independent_stats(_column_values(table, "action")),
            "timestamp": _independent_stats(_column_values(table, "timestamp")),
        }
        assert all(
            block["count"] == [retained_count] for block in stats_row["stats"].values()
        )

        assert set(row) == {
            "episode_index",
            "tasks",
            "length",
            "dataset_from_index",
            "dataset_to_index",
            "robot_type",
            "instruction",
            "environment_config",
            "conversion_provenance",
        }
        assert row["episode_index"] == episode.output_episode_index
        assert row["tasks"] == [episode.output_task_index]
        assert row["length"] == retained_count
        assert row["dataset_from_index"] == global_offset
        assert row["dataset_to_index"] == global_offset + retained_count - 1
        assert row["robot_type"] == "g1"
        assert row["instruction"] == task_by_index[episode.output_task_index]
        assert row["environment_config"] == episode.environment_config

        video_path = staging / (
            "videos/"
            f"chunk-{chunk:03d}/egocentric/episode_{episode.output_episode_index:06d}.mp4"
        )
        output_media = converter.probe_media(video_path)
        assert converter.media_profile(output_media) == plan.output_media
        assert output_media.frame_count == retained_count
        output_video_count += 1

        provenance = row["conversion_provenance"]
        assert provenance == {
            "converter_commit": plan.converter.commit,
            "converter_script_sha256": plan.converter.script_sha256,
            "downsample": plan.downsample,
            "output_media": _media_dict(output_media),
            "output_video_sha256": converter.sha256_file(video_path),
            "requested_output_fps": str(plan.output_fps),
            "requested_video_key": plan.video_key,
            "retained_count": retained_count,
            "skip": plan.skip,
            "source_episode_index": episode.source_episode_index,
            "source_media": _media_dict(episode.source_media),
            "source_parquet_sha256": episode.parquet_sha256,
            "source_video_sha256": episode.video_sha256,
        }
        episode_provenance_digests.append(
            converter.sha256_bytes(converter.canonical_json_bytes(provenance))
        )

        for name in GLOBAL_STATS_FIELDS:
            global_arrays[name].append(_column_values(table, name))
        global_offset += retained_count

    stats_bytes = (staging / "meta" / "stats.json").read_bytes()
    assert stats_bytes == (staging / "meta" / "stats_psi0.json").read_bytes()
    global_stats = _strict_json(stats_bytes)
    assert set(global_stats) == GLOBAL_STATS_FIELDS
    total_frames = global_offset
    for name in GLOBAL_STATS_FIELDS:
        emitted = np.concatenate(global_arrays[name], axis=0)
        assert global_stats[name] == _independent_stats(emitted)
        assert global_stats[name]["count"] == [total_frames]

    info = _strict_json((staging / "meta" / "info.json").read_bytes())
    assert set(info) == {
        "codebase_version",
        "robot_type",
        "total_episodes",
        "total_frames",
        "total_tasks",
        "total_videos",
        "total_chunks",
        "chunks_size",
        "fps",
        "data_path",
        "video_path",
        "features",
    }
    assert info["total_episodes"] == len(plan.episodes)
    assert info["total_frames"] == total_frames
    assert info["total_tasks"] == len(plan.tasks)
    assert info["total_videos"] == output_video_count
    assert info["total_chunks"] == 2
    assert info["features"]["states"]["shape"] == [32]
    assert info["features"]["action"]["shape"] == [36]
    profile = plan.output_media
    assert info["features"]["observation.images.egocentric"] == {
        "dtype": "video",
        "shape": [profile.height, profile.width, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "has_audio": bool(profile.audio_streams),
            "video.channels": 3,
            "video.codec": profile.codec_name,
            "video.fps": float(Fraction(profile.average_frame_rate)),
            "video.height": profile.height,
            "video.is_depth_map": False,
            "video.pix_fmt": profile.pixel_format,
            "video.width": profile.width,
        },
    }
    assert set(info["features"]) == {
        "observation.images.egocentric",
        *converter.output_schema().names,
    }

    dataset_provenance = _strict_json(
        (staging / "meta" / "conversion_provenance.json").read_bytes()
    )
    assert dataset_provenance["converter"] == asdict(plan.converter)
    assert dataset_provenance["invocation"] == {
        "chunks_size": plan.chunks_size,
        "downsample": plan.downsample,
        "output_fps": str(plan.output_fps),
        "output_path": str(plan.output_path),
        "skip": plan.skip,
        "total_episodes": len(plan.episodes),
        "video_key": plan.video_key,
    }
    assert [entry["path"] for entry in dataset_provenance["input_roots"]] == [
        str(path) for path in sorted({episode.source_root for episode in plan.episodes})
    ]
    assert dataset_provenance["media_mode"] == plan.media_mode
    assert dataset_provenance["output_media"] == _media_dict(plan.output_media)
    assert dataset_provenance["episode_provenance_sha256"] == (
        episode_provenance_digests
    )
    assert dataset_provenance["output_schema_sha256"] == (
        converter.output_schema_sha256()
    )

    assert _strict_json((staging / "meta" / "relative_stats.json").read_bytes()) == {}
    assert _strict_json((staging / "meta" / "lang_map.json").read_bytes()) == {}
    assert _strict_json((staging / "meta" / "modality.json").read_bytes()) == (
        converter.modality_dict()
    )
    for path in staging.rglob("*.json"):
        _strict_json(path.read_bytes())
    for path in staging.rglob("*.jsonl"):
        _strict_jsonl(path)


def test_preflight_plan_freezes_ordered_input_root_identities(plan):
    assert [identity.path for identity in plan.input_roots] == [
        path for path in sorted({episode.source_root for episode in plan.episodes})
    ]
    for identity in plan.input_roots:
        metadata = identity.path.lstat()
        assert (identity.st_dev, identity.st_ino) == (
            metadata.st_dev,
            metadata.st_ino,
        )


def test_generation_accepts_caller_initialized_staging_status(tmp_path, plan):
    staging = tmp_path / ".processed.staging-initialized"
    staging.mkdir()
    converter.write_in_progress_status(staging, plan)
    status = _strict_json((staging / "CONVERSION_STATUS.json").read_bytes())
    identity = staging.stat()
    assert status == {
        "schema_version": 1,
        "staging": {"st_dev": identity.st_dev, "st_ino": identity.st_ino},
        "state": "in_progress",
    }

    converter.generate_staged_dataset(plan, staging)

    _expected_tree(plan, staging)


def test_generation_rejects_mismatched_or_replaced_staging_identity(tmp_path, plan):
    mismatched = tmp_path / ".processed.staging-mismatched"
    mismatched.mkdir()
    identity = mismatched.stat()
    converter.atomic_write_new_bytes(
        mismatched / "CONVERSION_STATUS.json",
        converter.canonical_json_bytes(
            {
                "schema_version": 1,
                "staging": {
                    "st_dev": identity.st_dev,
                    "st_ino": identity.st_ino + 1,
                },
                "state": "in_progress",
            }
        ),
    )
    with pytest.raises(RuntimeError, match="staging identity"):
        converter.generate_staged_dataset(plan, mismatched)

    staging = tmp_path / ".processed.staging-replaced"
    staging.mkdir()
    converter.write_in_progress_status(staging, plan)
    preserved = tmp_path / "preserved-staging"
    staging.rename(preserved)
    staging.mkdir()
    shutil.copyfile(
        preserved / "CONVERSION_STATUS.json",
        staging / "CONVERSION_STATUS.json",
    )
    with pytest.raises(RuntimeError, match="staging identity"):
        converter.generate_staged_dataset(plan, staging)


def test_preflight_parquet_snapshot_cannot_mix_metadata_and_later_digest(
    tmp_path, monkeypatch
):
    source = make_source_episode(
        tmp_path / "source",
        frames=6,
        fps=4,
        task_index=7,
        task_text="trusted task",
    )
    parquet_path = source / "data/chunk-000/episode_000000.parquet"
    trusted_bytes = parquet_path.read_bytes()
    attacker_bytes = _parquet_bytes_with_task_index(parquet_path, 0)
    trusted_sha256 = converter.sha256_bytes(trusted_bytes)
    actual_read_table = converter.pq.read_table
    swapped = False

    def swap_after_metadata_read(where, *args, **kwargs):
        nonlocal swapped
        result = actual_read_table(where, *args, **kwargs)
        if kwargs.get("columns") == ["task_index"] and not swapped:
            swapped = True
            replacement = parquet_path.with_name("replacement.parquet")
            replacement.write_bytes(attacker_bytes)
            os.replace(replacement, parquet_path)
        return result

    monkeypatch.setattr(converter.pq, "read_table", swap_after_metadata_read)
    args = SimpleNamespace(
        skip=0,
        downsample=1,
        chunks_size=1000,
        total_episodes=1,
        fps="4",
        video_key="observation.rgb_head_stereo_left",
        out_dir=str(tmp_path / "processed"),
        sim_root=str(source),
    )
    identity = converter.ConverterIdentity("a" * 40, "b" * 64)

    plan = converter.preflight_conversion(args, identity)

    assert swapped
    assert plan.episodes[0].source_task_index == 7
    assert plan.episodes[0].parquet_sha256 == trusted_sha256
    with pytest.raises(RuntimeError, match="source Parquet"):
        converter.generate_staged_dataset(
            plan, tmp_path / ".processed.staging-mixed-preflight"
        )


def test_generation_rejects_post_preflight_parquet_path_replacement(tmp_path, plan):
    episode = plan.episodes[0]
    source = episode.parquet_path
    original = source.with_name("preflight-original.parquet")
    attacker = source.with_name("attacker.parquet")
    attacker.write_bytes(_parquet_bytes_with_qpos_offset(source, 1000))
    source.rename(original)
    attacker.rename(source)
    staging = tmp_path / ".processed.staging-replaced-parquet"

    try:
        with pytest.raises(RuntimeError, match="source Parquet"):
            converter.generate_staged_dataset(plan, staging)
    finally:
        source.unlink()
        original.rename(source)

    assert not (staging / "meta" / "conversion_manifest.json").exists()


def test_generation_reads_verified_parquet_snapshot_during_transient_mutation(
    tmp_path, plan, monkeypatch
):
    episode = plan.episodes[0]
    source = episode.parquet_path
    original_bytes = source.read_bytes()
    attacker_bytes = _parquet_bytes_with_qpos_offset(source, 1000)
    source_inode = source.stat().st_ino
    actual_read_table = converter.pq.read_table
    attacked = False

    def mutate_during_read(where, *args, **kwargs):
        nonlocal attacked
        if attacked:
            return actual_read_table(where, *args, **kwargs)
        attacked = True
        with source.open("r+b") as stream:
            stream.write(attacker_bytes)
            stream.truncate()
        assert source.stat().st_ino == source_inode
        try:
            return actual_read_table(where, *args, **kwargs)
        finally:
            with source.open("r+b") as stream:
                stream.write(original_bytes)
                stream.truncate()

    monkeypatch.setattr(converter.pq, "read_table", mutate_during_read)
    staging = tmp_path / ".processed.staging-transient-parquet"
    converter.generate_staged_dataset(plan, staging)

    assert attacked
    assert source.read_bytes() == original_bytes
    output = pq.read_table(staging / "data/chunk-000/episode_000000.parquet")
    clean_source = pq.read_table(pa.BufferReader(original_bytes))
    qpos = np.asarray(
        clean_source["observation.joint_qpos"].to_pylist(), dtype=np.float32
    )
    command = np.asarray(
        clean_source["observation.amo_policy_command"].to_pylist(),
        dtype=np.float32,
    )
    history = np.concatenate([converter.initial_command[None], command[:-1]], axis=0)
    clean_states, _ = converter.build_vectors(
        qpos,
        command,
        history,
        np.asarray(clean_source["action"].to_pylist(), dtype=np.float32),
        np.asarray(clean_source["observation.amo_policy_target_yaw"], dtype=np.float32),
        np.asarray(
            clean_source["observation.amo_policy_turning_flag"], dtype=np.float32
        ),
    )
    np.testing.assert_array_equal(
        _column_values(output, "states"), clean_states[episode.retained_indices]
    )


def test_generation_rejects_matching_profile_video_replacement_after_helper(
    tmp_path, plan, monkeypatch
):
    staging = tmp_path / ".processed.staging-replaced-video"
    attacker = tmp_path / "attacker.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=640x360:rate=4",
            "-frames:v",
            str(len(plan.episodes[0].retained_indices)),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(attacker),
        ],
        check=True,
    )
    actual_write = converter.write_episode_video
    attacked = False

    def replace_after_write(**kwargs):
        nonlocal attacked
        result = actual_write(**kwargs)
        if not attacked:
            attacked = True
            os.replace(attacker, kwargs["destination"])
        return result

    monkeypatch.setattr(converter, "write_episode_video", replace_after_write)
    with pytest.raises(RuntimeError, match="output video identity changed"):
        converter.generate_staged_dataset(plan, staging)

    assert attacked
    assert not (staging / "meta" / "conversion_manifest.json").exists()


def test_generation_rejects_same_inode_video_mutation_after_helper(
    tmp_path, plan, monkeypatch
):
    staging = tmp_path / ".processed.staging-mutated-video"
    actual_write = converter.write_episode_video
    attacked = False

    def mutate_after_write(**kwargs):
        nonlocal attacked
        result = actual_write(**kwargs)
        if not attacked:
            attacked = True
            destination = kwargs["destination"]
            metadata = destination.stat()
            payload = bytearray(destination.read_bytes())
            payload[-1] ^= 1
            with destination.open("r+b") as stream:
                stream.write(payload)
                stream.truncate()
            assert destination.stat().st_ino == metadata.st_ino
            assert destination.stat().st_size == metadata.st_size
        return result

    monkeypatch.setattr(converter, "write_episode_video", mutate_after_write)
    with pytest.raises(RuntimeError, match="output video SHA-256"):
        converter.generate_staged_dataset(plan, staging)

    assert attacked
    assert not (staging / "meta" / "conversion_manifest.json").exists()


def test_generation_revalidates_frozen_input_root_before_consumption(tmp_path):
    plan = _make_plan(tmp_path / "inputs")
    source_root = plan.input_roots[0].path
    preserved_root = source_root.with_name("preserved-source-a")
    source_root.rename(preserved_root)
    shutil.copytree(preserved_root, source_root)
    staging = tmp_path / ".processed.staging-root-swap"

    try:
        with pytest.raises(RuntimeError, match="input root identity changed"):
            converter.generate_staged_dataset(plan, staging)
    finally:
        shutil.rmtree(source_root)
        preserved_root.rename(source_root)

    assert not (staging / "meta" / "conversion_manifest.json").exists()


def test_generation_chunk_write_cannot_escape_pinned_staging_tree(
    tmp_path, plan, monkeypatch
):
    staging = tmp_path / ".processed.staging-chunk-swap"
    preserved_data = tmp_path / "preserved-data"
    external_data = tmp_path / "external-data"
    external_chunk = external_data / "chunk-000"
    external_chunk.mkdir(parents=True)
    actual_write = converter._write_parquet_new
    attacked = False

    def swap_data_ancestor(path, table, **kwargs):
        nonlocal attacked
        if not attacked:
            attacked = True
            (staging / "data").rename(preserved_data)
            (staging / "data").symlink_to(external_data, target_is_directory=True)
        return actual_write(path, table, **kwargs)

    monkeypatch.setattr(converter, "_write_parquet_new", swap_data_ancestor)
    with pytest.raises(RuntimeError, match="identity changed"):
        converter.generate_staged_dataset(plan, staging)

    assert attacked
    assert not (external_chunk / "episode_000000.parquet").exists()
    assert not (staging / "meta" / "conversion_manifest.json").exists()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", list(VECTOR_WIDTHS) + ["timestamp"])
def test_generation_rejects_every_nonfinite_output_before_arrow_and_manifest(
    tmp_path, plan, monkeypatch, field, bad
):
    staging = tmp_path / ".processed.staging-invalid"
    if field == "timestamp":
        invalid_plan = replace(plan)
        original = converter.require_finite

        def invalid_timestamp(name, values):
            if name == "timestamp":
                values = np.asarray(values).copy()
                values[0] = bad
            return original(name, values)

        monkeypatch.setattr(converter, "require_finite", invalid_timestamp)
    else:
        invalid_plan = plan
        if field in {"states", "action"}:
            original = converter.build_vectors

            def invalid_vectors(*args, **kwargs):
                states, actions = original(*args, **kwargs)
                result = [states.copy(), actions.copy()]
                result[0 if field == "states" else 1][
                    plan.episodes[0].retained_indices[0], 0
                ] = bad
                return tuple(result)

            monkeypatch.setattr(converter, "build_vectors", invalid_vectors)
        else:
            original = converter.build_proprio_obs
            output_index = {
                "observation.hand_joints": 0,
                "observation.arm_joints": 1,
                "observation.leg_joints": 2,
                "observation.prev_torso_rpy": 3,
                "observation.prev_height": 4,
            }[field]

            def invalid_proprio(*args, **kwargs):
                result = [value.copy() for value in original(*args, **kwargs)]
                result[output_index][plan.episodes[0].retained_indices[0], 0] = bad
                return tuple(result)

            monkeypatch.setattr(converter, "build_proprio_obs", invalid_proprio)

    with pytest.raises((ValueError, OverflowError), match="nonfinite"):
        converter.generate_staged_dataset(invalid_plan, staging)

    assert not (staging / "meta" / "conversion_manifest.json").exists()


def test_generation_refuses_existing_staging_tree(tmp_path, plan):
    staging = tmp_path / ".processed.staging-existing"
    staging.mkdir()
    sentinel = staging / "sentinel"
    sentinel.write_bytes(b"preserve\n")

    with pytest.raises(FileExistsError):
        converter.generate_staged_dataset(plan, staging)

    assert sentinel.read_bytes() == b"preserve\n"


def _conversion_argv(source: Path, output: Path) -> list[str]:
    return [
        "--sim-root",
        str(source),
        "--out-dir",
        str(output),
        "--skip",
        "0",
        "--downsample",
        "1",
        "--total-episodes",
        "1",
        "--fps",
        "4",
        "--video-key",
        "observation.rgb_head_stereo_left",
        "--chunks-size",
        "1000",
    ]


def _recorded_converter_identity() -> converter.ConverterIdentity:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", "scripts/postprocess_psi0.py"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = subprocess.run(
        ["git", "show", f"{commit}:scripts/postprocess_psi0.py"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    return converter.ConverterIdentity(commit, converter.sha256_bytes(payload))


def _failed_staging(output: Path) -> Path:
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4()}"
    staging.mkdir(mode=0o755)
    error = RuntimeError("synthetic generation failure")
    (staging / "CONVERSION_STATUS.json").write_bytes(
        converter._failed_status_bytes(staging, error)
    )
    return staging


def _complete_staging(output: Path) -> Path:
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4()}"
    staging.mkdir(mode=0o755)
    for name in ("data", "videos", "meta"):
        (staging / name).mkdir(mode=0o755)
    (staging / "data" / "payload.bin").write_bytes(b"certified payload\n")
    (staging / "CONVERSION_STATUS.json").write_bytes(
        converter._in_progress_status_bytes(staging.stat(follow_symlinks=False))
    )
    manifest = converter.build_payload_manifest(staging)
    manifest_payload = converter.canonical_json_bytes(manifest)
    (staging / "meta" / "conversion_manifest.json").write_bytes(manifest_payload)
    (staging / "CONVERSION_STATUS.json").write_bytes(
        converter._complete_status_bytes(
            staging,
            manifest,
            manifest_payload,
            converter.ConverterIdentity("a" * 40, "b" * 64),
        )
    )
    for path in staging.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    staging.chmod(0o555)
    return staging


def _replace_sealed_json(path: Path, value: object) -> None:
    path.parent.chmod(0o755)
    path.chmod(0o644)
    path.write_bytes(converter.canonical_json_bytes(value))
    path.chmod(0o444)
    path.parent.chmod(0o555)


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int, int, str | None]]:
    snapshot = {}
    for path in [root, *sorted(root.rglob("*"))]:
        metadata = path.lstat()
        relative = "." if path == root else path.relative_to(root).as_posix()
        digest = converter.sha256_file(path) if stat.S_ISREG(metadata.st_mode) else None
        snapshot[relative] = (
            stat.S_IFMT(metadata.st_mode),
            stat.S_IMODE(metadata.st_mode),
            metadata.st_ino,
            digest,
        )
    return snapshot


def test_cli_accepts_exact_conversion_options_and_legacy_episode_alias(tmp_path):
    parser = converter.build_parser()
    common = [
        "--sim-root",
        str(tmp_path / "source-*"),
        "--out-dir",
        str(tmp_path / "processed"),
        "--skip",
        "3",
        "--downsample",
        "2",
        "--fps",
        "25/2",
        "--video-key",
        "camera",
        "--chunks-size",
        "17",
        "--preflight-only",
    ]

    dashed = parser.parse_args([*common, "--total-episodes", "9"])
    underscored = parser.parse_args([*common, "--total_episodes", "9"])

    for args in (dashed, underscored):
        assert args.sim_root == str(tmp_path / "source-*")
        assert args.out_dir == str(tmp_path / "processed")
        assert args.skip == 3
        assert args.downsample == 2
        assert args.total_episodes == 9
        assert args.fps == Fraction(25, 2)
        assert args.video_key == "camera"
        assert args.chunks_size == 17
        assert args.preflight_only is True
        assert args.certify_psi0_root is None
        assert args.certify_psi0_commit is None
        assert args.certify_python is None


@pytest.mark.parametrize("fps", ["0", "-1", "not-a-rate"])
def test_cli_rejects_nonpositive_or_invalid_fps(fps):
    with pytest.raises(SystemExit):
        converter.build_parser().parse_args(["--out-dir", "/tmp/output", "--fps", fps])


@pytest.mark.parametrize("chunks_size", ["0", "-1", "not-an-integer"])
def test_cli_rejects_nonpositive_or_invalid_chunks_size(chunks_size):
    with pytest.raises(SystemExit):
        converter.build_parser().parse_args(
            ["--out-dir", "/tmp/output", "--chunks-size", chunks_size]
        )


@pytest.mark.parametrize(
    "options",
    [
        ["--certify-psi0-root", "/tmp/psi0"],
        ["--certify-psi0-commit", "a" * 40],
        ["--certify-python", sys.executable],
        [
            "--certify-psi0-root",
            "/tmp/psi0",
            "--certify-psi0-commit",
            "a" * 40,
        ],
    ],
)
def test_cli_rejects_partial_certification_option_sets(tmp_path, options):
    source = make_source_episode(tmp_path / "source", frames=2, fps=4)
    with pytest.raises(SystemExit):
        converter.main([*_conversion_argv(source, tmp_path / "output"), *options])


@pytest.mark.parametrize(
    "options",
    [
        ["--inspect-preserved-staging", "/tmp/.output.staging-" + str(uuid.uuid4())],
        [
            "--remove-preserved-staging",
            "/tmp/.output.staging-" + str(uuid.uuid4()),
            "--expected-status-sha256",
            "a" * 64,
            "--confirm-remove",
            "/tmp/.output.staging-" + str(uuid.uuid4()),
        ],
    ],
)
def test_cli_maintenance_modes_reject_conversion_and_certification_options(options):
    with pytest.raises(SystemExit):
        converter.main(
            [
                "--out-dir",
                "/tmp/output",
                *options,
                "--sim-root",
                "/tmp/source",
            ]
        )


@pytest.mark.parametrize(
    "abbreviated",
    [
        ["--ski", "1"],
        ["--certify-psi0-r", "/tmp/psi0"],
    ],
)
def test_cli_maintenance_rejects_abbreviated_conversion_and_certification_options(
    tmp_path,
    abbreviated,
):
    output = tmp_path / "output"
    staging = _failed_staging(output)
    before = _tree_snapshot(staging)

    with pytest.raises(SystemExit):
        converter.main(
            [
                "--out-dir",
                str(output),
                "--inspect-preserved-staging",
                str(staging),
                *abbreviated,
            ]
        )

    assert _tree_snapshot(staging) == before


def test_cli_rejects_combined_inspection_and_removal():
    staging = "/tmp/.output.staging-" + str(uuid.uuid4())
    with pytest.raises(SystemExit):
        converter.main(
            [
                "--out-dir",
                "/tmp/output",
                "--inspect-preserved-staging",
                staging,
                "--remove-preserved-staging",
                staging,
                "--expected-status-sha256",
                "a" * 64,
                "--confirm-remove",
                staging,
            ]
        )


def test_cli_subprocess_attests_real_committed_script(tmp_path):
    source = make_source_episode(tmp_path / "source", frames=3, fps=4)
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "postprocess_psi0.py"
    shutil.copyfile(Path(converter.__file__), script)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "add", "scripts/postprocess_psi0.py"], cwd=repository, check=True
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Task Nine",
            "-c",
            "user.email=task9@example.invalid",
            "commit",
            "-qm",
            "record converter",
        ],
        cwd=repository,
        check=True,
    )
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    output = tmp_path / "processed"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--preflight-only",
            *_conversion_argv(source, output),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert report["selected_episodes"] == 1
    assert report["retained_frames"] == 3
    assert (
        expected_commit
        == subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", "scripts/postprocess_psi0.py"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert not output.exists()
    assert not list(tmp_path.glob(".processed.staging-*"))


@pytest.mark.parametrize("entrypoint", ["script", "module"])
def test_cli_entrypoints_complete_conversion_and_certification(tmp_path, entrypoint):
    from scripts.certify_psi0_dataset import validate_evidence_terminal

    source = make_source_episode(tmp_path / "source", frames=3, fps=4)
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "postprocess_psi0.py"
    certifier = scripts / "certify_psi0_dataset.py"
    shutil.copyfile(Path(converter.__file__), script)
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "scripts/certify_psi0_dataset.py",
        certifier,
    )
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "add",
            "scripts/postprocess_psi0.py",
            "scripts/certify_psi0_dataset.py",
        ],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Task Twelve",
            "-c",
            "user.email=task12@example.invalid",
            "commit",
            "-qm",
            "record converter and certifier",
        ],
        cwd=repository,
        check=True,
    )
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_script_sha256 = converter.sha256_file(script)
    psi0_root, psi0_commit, import_marker = _make_task9_fake_psi0_checkout(tmp_path)
    output = tmp_path / "processed"
    command = (
        [sys.executable, str(script)]
        if entrypoint == "script"
        else [sys.executable, "-m", "scripts.postprocess_psi0"]
    )

    completed = subprocess.run(
        [
            *command,
            *_conversion_argv(source, output),
            "--certify-psi0-root",
            str(psi0_root),
            "--certify-psi0-commit",
            psi0_commit,
            "--certify-python",
            sys.executable,
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.is_dir()
    assert not list(tmp_path.glob(".processed.staging-*"))
    validation = subprocess.run(
        [sys.executable, str(certifier), str(output), "--print-tree-digest"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert validation.returncode == 0, validation.stderr
    assert re.fullmatch(r"[0-9a-f]{64}\n", validation.stdout)
    assert _strict_json((output / "meta/info.json").read_bytes())["total_frames"] == 3
    evidence_roots = list(tmp_path.glob(".processed.certification-*"))
    assert len(evidence_roots) == 1
    assert validate_evidence_terminal(evidence_roots[0])["verdict"] == "PASS"
    assert import_marker.read_text() == "imported"
    provenance = _strict_json((output / "meta/conversion_provenance.json").read_bytes())
    assert provenance["converter"] == {
        "commit": expected_commit,
        "script_sha256": expected_script_sha256,
    }
    assert (
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )


@pytest.mark.parametrize(
    ("entrypoint", "attack_location"),
    [
        ("converter-script", "script"),
        ("converter-script", "pythonpath"),
        ("converter-module", "cwd"),
        ("certifier-script", "script"),
        ("certifier-script", "pythonpath"),
        ("certifier-module", "cwd"),
    ],
)
def test_cli_entrypoints_ignore_shadowable_startup_paths(
    tmp_path, entrypoint, attack_location
):
    source = make_source_episode(tmp_path / "source", frames=3, fps=4)
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    converter_script = scripts / "postprocess_psi0.py"
    certifier_script = scripts / "certify_psi0_dataset.py"
    shutil.copyfile(Path(converter.__file__), converter_script)
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "scripts/certify_psi0_dataset.py",
        certifier_script,
    )
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "add",
            "scripts/postprocess_psi0.py",
            "scripts/certify_psi0_dataset.py",
        ],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Startup Test",
            "-c",
            "user.email=startup@example.invalid",
            "commit",
            "-qm",
            "record entrypoints",
        ],
        cwd=repository,
        check=True,
    )
    marker = tmp_path / f"{entrypoint}-{attack_location}-executed"
    if attack_location == "script":
        attack_root = scripts
    elif attack_location == "cwd":
        attack_root = repository
    else:
        attack_root = tmp_path / "pythonpath-attack"
        attack_root.mkdir()
    (attack_root / "argparse.py").write_text(
        f"open({str(marker)!r}, 'w').write('executed')\n"
        "raise RuntimeError('shadow argparse executed')\n"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    if attack_location == "pythonpath":
        environment["PYTHONPATH"] = str(attack_root)
    if entrypoint == "converter-script":
        command = [
            sys.executable,
            str(converter_script),
            "--preflight-only",
            *_conversion_argv(source, tmp_path / "output"),
        ]
    elif entrypoint == "converter-module":
        command = [
            sys.executable,
            "-m",
            "scripts.postprocess_psi0",
            "--preflight-only",
            *_conversion_argv(source, tmp_path / "output"),
        ]
    elif entrypoint == "certifier-script":
        command = [sys.executable, str(certifier_script), "--help"]
    else:
        command = [sys.executable, "-m", "scripts.certify_psi0_dataset", "--help"]

    completed = subprocess.run(
        command,
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()


def test_converter_module_rejects_regular_scripts_package_before_conversion(tmp_path):
    source = make_source_episode(tmp_path / "source", frames=3, fps=4)
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(Path(converter.__file__), scripts / "postprocess_psi0.py")
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "scripts/certify_psi0_dataset.py",
        scripts / "certify_psi0_dataset.py",
    )
    (scripts / "__init__.py").write_text("")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "add", "scripts"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Module Boundary",
            "-c",
            "user.email=module@example.invalid",
            "commit",
            "-qm",
            "record regular package",
        ],
        cwd=repository,
        check=True,
    )
    output = tmp_path / "output"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.postprocess_psi0",
            *_conversion_argv(source, output),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "module execution requires a namespace scripts package" in completed.stderr
    assert not output.exists()
    assert not list(tmp_path.glob(".output.staging-*"))


def test_certificate_array_expansion_is_executable(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    certificate = parent / (
        ".output.certification-" + "a" * 64 + "-00000000-0000-0000-0000-000000000001"
    )
    certificate.mkdir()
    script = r"""
set -euo pipefail
PARENT=$1
mapfile -t CERTS < <(find "$PARENT" -maxdepth 1 -type d -name '.output.certification-*' -print | sort)
test "${#CERTS[@]}" -eq 1
CERT_ROOT=${CERTS[0]}
test "$CERT_ROOT" = "$PARENT/.output.certification-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-00000000-0000-0000-0000-000000000001"
"""
    subprocess.run(["bash", "-c", script, "bash", str(parent)], check=True)


def test_preserved_staging_inspection_is_no_follow_and_read_only(tmp_path):
    output = tmp_path / "output"
    staging = _failed_staging(output)
    nested = staging / "nested"
    nested.mkdir()
    (nested / "payload").write_bytes(b"payload")
    (nested / "leaf-link").symlink_to(tmp_path / "outside")
    before = _tree_snapshot(staging)

    report = converter.inspect_preserved_staging(staging, output)

    assert _tree_snapshot(staging) == before
    assert report["schema_version"] == 1
    assert report["staging_path"] == str(staging.absolute())
    assert report["output_path"] == str(output.absolute())
    assert report["failure_classification"] == "pre_completion_failed"
    assert report["status"]["bytes"] == (staging / "CONVERSION_STATUS.json").read_text()
    assert report["status"]["sha256"] == converter.sha256_file(
        staging / "CONVERSION_STATUS.json"
    )
    assert report["manifest"]["state"] == "absent"
    assert [entry["path"] for entry in report["tree"]] == sorted(
        before, key=lambda value: value.encode("utf-8")
    )
    assert (
        next(item for item in report["tree"] if item["path"] == "nested/leaf-link")[
            "type"
        ]
        == "symlink"
    )


def test_preserved_staging_inspection_never_reads_through_manifest_ancestor(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output"
    staging = _failed_staging(output)
    external_meta = tmp_path / "external-meta"
    external_meta.mkdir()
    (external_meta / "conversion_manifest.json").write_bytes(b"external")
    (staging / "meta").symlink_to(external_meta, target_is_directory=True)
    actual_read = converter._read_regular_file_bytes

    def reject_external_read(path, label):
        if path == staging / "meta/conversion_manifest.json":
            pytest.fail("inspection followed an untrusted manifest ancestor")
        return actual_read(path, label)

    monkeypatch.setattr(converter, "_read_regular_file_bytes", reject_external_read)

    report = converter.inspect_preserved_staging(staging, output)

    assert report["manifest"]["state"] in {"absent", "invalid"}
    assert (
        next(item for item in report["tree"] if item["path"] == "meta")["type"]
        == "symlink"
    )


def test_preserved_complete_staging_inspection_validates_full_bound_tree(tmp_path):
    output = tmp_path / "output"
    staging = _complete_staging(output)

    report = converter.inspect_preserved_staging(staging, output)

    assert report["manifest"]["state"] == "valid"
    assert report["failure_classification"] == "complete_unpublished"


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_status_field",
        "empty_entries",
        "wrong_hash",
        "extra_file",
        "missing_file",
        "wrong_root_identity",
    ],
)
def test_preserved_complete_staging_inspection_rejects_corrupt_binding(
    tmp_path,
    corruption,
):
    output = tmp_path / "output"
    staging = _complete_staging(output)
    status_path = staging / "CONVERSION_STATUS.json"
    manifest_path = staging / "meta" / "conversion_manifest.json"

    if corruption in {"missing_status_field", "wrong_root_identity"}:
        status = _strict_json(status_path.read_bytes())
        if corruption == "missing_status_field":
            del status["entry_count"]
        else:
            status["staging"]["st_ino"] += 1
        _replace_sealed_json(status_path, status)
    elif corruption in {"empty_entries", "wrong_hash"}:
        manifest = _strict_json(manifest_path.read_bytes())
        if corruption == "empty_entries":
            manifest["entries"] = []
        else:
            manifest["entries"][0]["sha256"] = "0" * 64
        _replace_sealed_json(manifest_path, manifest)
    elif corruption == "extra_file":
        data = staging / "data"
        data.chmod(0o755)
        (data / "extra.bin").write_bytes(b"extra\n")
        (data / "extra.bin").chmod(0o444)
        data.chmod(0o555)
    else:
        data = staging / "data"
        data.chmod(0o755)
        (data / "payload.bin").unlink()
        data.chmod(0o555)
    before = _tree_snapshot(staging)

    report = converter.inspect_preserved_staging(staging, output)

    assert _tree_snapshot(staging) == before
    assert report["manifest"]["state"] == "invalid"
    assert report["failure_classification"] == "completion_uncertain_unpublished"


def test_preserved_staging_removal_rejects_status_bound_to_another_root(tmp_path):
    output = tmp_path / "output"
    staging = _complete_staging(output)
    status_path = staging / "CONVERSION_STATUS.json"
    status = _strict_json(status_path.read_bytes())
    status["staging"]["st_ino"] += 1
    _replace_sealed_json(status_path, status)
    status_sha256 = converter.sha256_file(status_path)
    before = _tree_snapshot(staging)

    with pytest.raises(RuntimeError, match="identity|root"):
        converter.remove_preserved_staging(
            staging,
            output,
            expected_status_sha256=status_sha256,
            confirm_remove=str(staging),
        )

    assert _tree_snapshot(staging) == before


def test_preserved_staging_removal_is_fd_relative_and_preserves_siblings(tmp_path):
    output = tmp_path / "output"
    staging = _failed_staging(output)
    nested = staging / "nested"
    nested.mkdir()
    payload = nested / "payload"
    payload.write_bytes(b"payload")
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    (nested / "leaf-link").symlink_to(outside)
    hard_link = nested / "hard-link"
    os.link(outside, hard_link)
    sibling = tmp_path / "unrelated"
    sibling.mkdir()
    (sibling / "sentinel").write_bytes(b"preserve")
    status_sha256 = converter.sha256_file(staging / "CONVERSION_STATUS.json")

    converter.remove_preserved_staging(
        staging,
        output,
        expected_status_sha256=status_sha256,
        confirm_remove=str(staging.absolute()),
    )

    assert not os.path.lexists(staging)
    assert outside.read_bytes() == b"outside"
    assert (sibling / "sentinel").read_bytes() == b"preserve"


def test_preserved_staging_removal_holds_lock_through_mutation_and_parent_fsync(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output"
    staging = _failed_staging(output)
    (staging / "payload").write_bytes(b"payload")
    status_sha256 = converter.sha256_file(staging / "CONVERSION_STATUS.json")
    actual_unlink = converter.os.unlink
    actual_fsync = converter.os.fsync
    mutation_seen = False
    parent_fsync_seen = False

    def checked_unlink(*args, **kwargs):
        nonlocal mutation_seen
        converter.assert_conversion_lock_held(output)
        mutation_seen = True
        return actual_unlink(*args, **kwargs)

    def checked_fsync(fd):
        nonlocal parent_fsync_seen
        if mutation_seen:
            converter.assert_conversion_lock_held(output)
            parent_fsync_seen = True
        return actual_fsync(fd)

    monkeypatch.setattr(converter.os, "unlink", checked_unlink)
    monkeypatch.setattr(converter.os, "fsync", checked_fsync)

    converter.remove_preserved_staging(
        staging,
        output,
        expected_status_sha256=status_sha256,
        confirm_remove=str(staging),
    )

    assert mutation_seen
    assert parent_fsync_seen


def test_preserved_staging_output_race_before_recursive_mutation_keeps_tree(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output"
    staging = _failed_staging(output)
    nested = staging / "nested"
    nested.mkdir()
    (nested / "payload").write_bytes(b"payload")
    before = _tree_snapshot(staging)
    status_sha256 = converter.sha256_file(staging / "CONVERSION_STATUS.json")
    actual = converter._require_output_absent_at

    def create_output(parent_fd, output_name, boundary):
        if boundary == "before_recursive_mutation":
            os.mkdir(output_name, dir_fd=parent_fd)
        return actual(parent_fd, output_name, boundary)

    monkeypatch.setattr(converter, "_require_output_absent_at", create_output)

    with pytest.raises(FileExistsError):
        converter.remove_preserved_staging(
            staging,
            output,
            expected_status_sha256=status_sha256,
            confirm_remove=str(staging),
        )

    assert _tree_snapshot(staging) == before
    assert output.is_dir()


def test_preserved_staging_output_race_before_root_rmdir_fails_closed(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output"
    staging = _failed_staging(output)
    (staging / "payload").write_bytes(b"payload")
    status_sha256 = converter.sha256_file(staging / "CONVERSION_STATUS.json")
    actual = converter._require_output_absent_at

    def create_output(parent_fd, output_name, boundary):
        if boundary == "before_staging_rmdir":
            os.mkdir(output_name, dir_fd=parent_fd)
        return actual(parent_fd, output_name, boundary)

    monkeypatch.setattr(converter, "_require_output_absent_at", create_output)

    with pytest.raises(FileExistsError):
        converter.remove_preserved_staging(
            staging,
            output,
            expected_status_sha256=status_sha256,
            confirm_remove=str(staging),
        )

    assert staging.is_dir()
    assert output.is_dir()


@pytest.mark.parametrize(
    "failure", ["wrong_digest", "wrong_confirmation", "output_exists"]
)
def test_preserved_staging_removal_rejects_wrong_authorization(tmp_path, failure):
    output = tmp_path / "output"
    staging = _failed_staging(output)
    status_sha256 = converter.sha256_file(staging / "CONVERSION_STATUS.json")
    confirmation = str(staging.absolute())
    if failure == "wrong_digest":
        status_sha256 = "0" * 64
    elif failure == "wrong_confirmation":
        confirmation = str(staging) + "-wrong"
    else:
        output.mkdir()

    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        converter.remove_preserved_staging(
            staging,
            output,
            expected_status_sha256=status_sha256,
            confirm_remove=confirmation,
        )

    assert staging.is_dir()


@pytest.mark.parametrize(
    "staging_text",
    [
        "/tmp/not-a-staging-root",
        "/tmp/.other.staging-00000000-0000-0000-0000-000000000000",
        "/tmp/../tmp/.output.staging-00000000-0000-0000-0000-000000000000",
    ],
)
def test_preserved_staging_rejects_bad_path_contract(staging_text, tmp_path):
    output = tmp_path / "output"
    with pytest.raises((ValueError, FileNotFoundError)):
        converter.inspect_preserved_staging(Path(staging_text), output)


def test_preserved_staging_rejects_symlink_root(tmp_path):
    output = tmp_path / "output"
    target = _failed_staging(output)
    link = output.parent / f".{output.name}.staging-{uuid.uuid4()}"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises((ValueError, RuntimeError, OSError)):
        converter.inspect_preserved_staging(link, output)


def test_preserved_staging_removal_rejects_live_lock_holder(tmp_path):
    output = tmp_path / "output"
    staging = _failed_staging(output)
    status_sha256 = converter.sha256_file(staging / "CONVERSION_STATUS.json")

    with converter.conversion_lock(output):
        with pytest.raises(BlockingIOError):
            converter.remove_preserved_staging(
                staging,
                output,
                expected_status_sha256=status_sha256,
                confirm_remove=str(staging.absolute()),
            )

    assert staging.is_dir()


def test_preserved_staging_open_rejects_path_swap(tmp_path, monkeypatch):
    output = tmp_path / "output"
    staging = _failed_staging(output)
    original = staging.with_name(staging.name + ".preserved")
    replacement = staging.with_name(staging.name + ".replacement")
    replacement.mkdir()
    (replacement / "sentinel").write_bytes(b"replacement")
    actual_open = converter.os.open
    swapped = False

    def swap_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if not swapped and path == os.fsencode(staging.name):
            swapped = True
            staging.rename(original)
            staging.symlink_to(replacement, target_is_directory=True)
        return actual_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(converter.os, "open", swap_before_open)
    started = time.monotonic()
    with pytest.raises((OSError, RuntimeError, ValueError)):
        converter.inspect_preserved_staging(staging, output)
    assert time.monotonic() - started < 2
    assert swapped
    assert (replacement / "sentinel").read_bytes() == b"replacement"
    assert original.is_dir()


def test_preserved_staging_removal_rejects_path_swap_before_open(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output"
    staging = _failed_staging(output)
    status_sha256 = converter.sha256_file(staging / "CONVERSION_STATUS.json")
    original = staging.with_name(staging.name + ".preserved")
    replacement = staging.with_name(staging.name + ".replacement")
    replacement.mkdir()
    (replacement / "sentinel").write_bytes(b"replacement")
    actual_open = converter.os.open
    swapped = False

    def swap_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if not swapped and path == os.fsencode(staging.name):
            swapped = True
            staging.rename(original)
            staging.symlink_to(replacement, target_is_directory=True)
        return actual_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(converter.os, "open", swap_before_open)
    started = time.monotonic()
    with pytest.raises((OSError, RuntimeError, ValueError)):
        converter.remove_preserved_staging(
            staging,
            output,
            expected_status_sha256=status_sha256,
            confirm_remove=str(staging),
        )
    assert time.monotonic() - started < 2
    assert swapped
    assert (replacement / "sentinel").read_bytes() == b"replacement"
    assert original.is_dir()


@pytest.mark.parametrize("operation", ["inspect", "remove"])
def test_preserved_staging_parent_swap_after_lock_uses_no_reopened_parent(
    tmp_path,
    monkeypatch,
    operation,
):
    parent = tmp_path / "parent"
    parent.mkdir()
    output = parent / "output"
    staging = _failed_staging(output)
    (staging / "payload").write_bytes(b"original")
    status_sha256 = converter.sha256_file(staging / "CONVERSION_STATUS.json")
    original_before = _tree_snapshot(staging)
    displaced_parent = tmp_path / "displaced-parent"
    actual_open = converter._open_preserved_staging
    actual_read = converter._read_regular_at
    replacement_before = None
    replacement_read = False
    swapped = False

    def swap_then_open(*args, **kwargs):
        nonlocal replacement_before, swapped
        if not swapped:
            swapped = True
            parent.rename(displaced_parent)
            parent.mkdir()
            replacement = parent / staging.name
            shutil.copytree(displaced_parent / staging.name, replacement)
            (replacement / "payload").write_bytes(b"replacement")
            replacement_before = _tree_snapshot(replacement)
        return actual_open(*args, **kwargs)

    def track_replacement_read(directory_fd, name, label):
        nonlocal replacement_read
        replacement = parent / staging.name
        if (
            replacement.exists()
            and os.fstat(directory_fd).st_ino == replacement.stat().st_ino
        ):
            replacement_read = True
        return actual_read(directory_fd, name, label)

    monkeypatch.setattr(converter, "_open_preserved_staging", swap_then_open)
    monkeypatch.setattr(converter, "_read_regular_at", track_replacement_read)

    with pytest.raises((RuntimeError, FileNotFoundError)):
        if operation == "inspect":
            converter.inspect_preserved_staging(staging, output)
        else:
            converter.remove_preserved_staging(
                staging,
                output,
                expected_status_sha256=status_sha256,
                confirm_remove=str(staging),
            )

    assert swapped
    assert replacement_before is not None
    assert _tree_snapshot(displaced_parent / staging.name) == original_before
    assert _tree_snapshot(parent / staging.name) == replacement_before
    assert replacement_read is False


def test_cli_maintenance_branches_before_converter_attestation_and_preflight(
    tmp_path,
    monkeypatch,
    capsys,
):
    output = tmp_path / "output"
    staging = _failed_staging(output)
    monkeypatch.setattr(
        converter,
        "resolve_converter_identity",
        lambda *_: pytest.fail("maintenance attempted converter attestation"),
    )
    monkeypatch.setattr(
        converter,
        "preflight_conversion",
        lambda *_: pytest.fail("maintenance attempted source preflight"),
    )

    assert (
        converter.main(
            [
                "--out-dir",
                str(output),
                "--inspect-preserved-staging",
                str(staging),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["staging_path"] == str(staging)


def test_original_run_never_certifies_publication_uncertain(tmp_path, monkeypatch):
    source = make_source_episode(tmp_path / "source", frames=8, fps=4)
    output = tmp_path / "processed"
    psi0_root = tmp_path / "psi0"
    psi0_root.mkdir()
    certifier_calls = []

    class FailDestinationParentFsync(converter.PublicationFilesystem):
        def fsync_directory(self, path):
            if path == output.parent and output.exists():
                raise OSError("injected destination-parent fsync failure")
            super().fsync_directory(path)

    monkeypatch.setattr(
        converter,
        "resolve_converter_identity",
        lambda *_: _recorded_converter_identity(),
    )
    monkeypatch.setattr(
        converter,
        "PublicationFilesystem",
        FailDestinationParentFsync,
    )
    monkeypatch.setattr(
        converter,
        "certify_published_dataset",
        lambda *args, **kwargs: certifier_calls.append((args, kwargs)),
    )
    args = converter.build_parser().parse_args(
        [
            *_conversion_argv(source, output),
            "--certify-psi0-root",
            str(psi0_root),
            "--certify-psi0-commit",
            "c" * 40,
            "--certify-python",
            sys.executable,
        ]
    )

    with pytest.raises(converter.PublicationUncertainError):
        converter.run_conversion(args)

    assert output.is_dir()
    assert certifier_calls == []
    assert list(tmp_path.glob(".processed.certification-*")) == []


def test_post_parent_fsync_hook_bypasses_failure_reporting_and_certification(
    tmp_path,
    monkeypatch,
):
    source = make_source_episode(tmp_path / "source", frames=8, fps=4)
    output = tmp_path / "processed"
    psi0_root = tmp_path / "psi0"
    psi0_root.mkdir()
    failure_reports = []
    certifier_calls = []

    def fail_after_parent_fsync(point):
        if point == "after_destination_parent_fsync":
            raise RuntimeError("injected post-parent-fsync hook failure")

    monkeypatch.setattr(
        converter,
        "resolve_converter_identity",
        lambda *_: converter.ConverterIdentity("a" * 40, "b" * 64),
    )
    monkeypatch.setattr(converter, "no_fault", fail_after_parent_fsync)
    monkeypatch.setattr(
        converter,
        "report_conversion_failure",
        lambda *args: failure_reports.append(args),
    )
    monkeypatch.setattr(
        converter,
        "certify_published_dataset",
        lambda *args, **kwargs: certifier_calls.append((args, kwargs)),
    )
    args = converter.build_parser().parse_args(
        [
            *_conversion_argv(source, output),
            "--certify-psi0-root",
            str(psi0_root),
            "--certify-psi0-commit",
            "c" * 40,
            "--certify-python",
            sys.executable,
        ]
    )

    with pytest.raises(converter.PublishedBoundaryError) as caught:
        converter.run_conversion(args)

    assert caught.value.publication.state == "published"
    assert output.is_dir()
    assert failure_reports == []
    assert certifier_calls == []
    assert list(tmp_path.glob(".processed.certification-*")) == []


def _make_task9_fake_psi0_checkout(root: Path) -> tuple[Path, str, Path]:
    checkout = root / "fake-psi0"
    package = checkout / "src/psi/data/lerobot"
    package.mkdir(parents=True)
    for init in (
        checkout / "src/psi/__init__.py",
        checkout / "src/psi/data/__init__.py",
        checkout / "src/psi/data/lerobot/__init__.py",
    ):
        init.write_text("")
    import_marker = root / "fake-psi0-imported"
    (package / "compat.py").write_text(
        f"""\
import json
import os
from pathlib import Path

import torch

LEROBOT_LAYOUT = "fake"
Path({str(import_marker)!r}).write_text("imported")


class LeRobotDataset:
    def __init__(self, *, repo_id, root):
        assert repo_id == "simple-certified"
        self.root = Path(root)
        info = json.loads((self.root / "meta/info.json").read_text())
        self.total = info["total_frames"]
        height, width, _channels = info["features"]["observation.images.egocentric"]["shape"]
        self.height = height
        self.width = width
        cache_root = Path(os.environ["HOME"]).parent
        assert Path(os.environ["HF_HOME"]) == cache_root / "hf"
        assert Path(os.environ["HF_DATASETS_CACHE"]) == cache_root / "datasets"
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"
        (Path(os.environ["HF_HOME"]) / "task9").mkdir()
        (Path(os.environ["HF_HOME"]) / "task9/cache.bin").write_bytes(b"cache")

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        return {{
            "observation.images.egocentric": torch.zeros(
                (3, self.height, self.width), dtype=torch.float32
            ),
            "states": torch.zeros((32,), dtype=torch.float32),
            "action": torch.zeros((36,), dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.int64),
        }}
""",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "src"], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Task Nine",
            "-c",
            "user.email=task9@example.invalid",
            "commit",
            "-qm",
            "fake pinned PSI0",
        ],
        cwd=checkout,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return checkout, commit, import_marker


def _direct_children() -> tuple[int, ...]:
    children_path = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
    if not children_path.exists():
        return ()
    return tuple(int(value) for value in children_path.read_text().split())


def test_complete_synthetic_pipeline_transcodes_certifies_and_recertifies(
    tmp_path,
    monkeypatch,
):
    from scripts.certify_psi0_dataset import (
        certify_published_dataset as certify,
        validate_dataset,
        validate_evidence_terminal,
    )

    source_a = make_source_episode(
        tmp_path / "source-a",
        episode_index=4,
        frames=5,
        fps=4,
        size="64x48",
        video_codec="libx264",
        task_index=2,
        task_text="first synthetic task",
    )
    source_b = make_source_episode(
        tmp_path / "source-b",
        episode_index=7,
        frames=6,
        fps=5,
        size="80x60",
        video_codec="mpeg4",
        task_index=3,
        task_text="second synthetic task",
    )
    source_before = {source: _tree_snapshot(source) for source in (source_a, source_b)}
    checkout, commit, import_marker = _make_task9_fake_psi0_checkout(tmp_path)
    output = tmp_path / "processed"
    monkeypatch.setattr(
        converter,
        "resolve_converter_identity",
        lambda *_: _recorded_converter_identity(),
    )
    args = converter.build_parser().parse_args(
        [
            "--sim-root",
            str(tmp_path / "source-*"),
            "--out-dir",
            str(output),
            "--skip",
            "1",
            "--downsample",
            "2",
            "--total-episodes",
            "2",
            "--fps",
            "4",
            "--video-key",
            "observation.rgb_head_stereo_left",
            "--chunks-size",
            "1",
            "--certify-psi0-root",
            str(checkout),
            "--certify-psi0-commit",
            commit,
            "--certify-python",
            sys.executable,
        ]
    )
    children_before = _direct_children()

    publication = converter.run_conversion(args)

    assert isinstance(publication, converter.PublicationResult)
    assert publication.state == "published"
    validation = validate_dataset(output, expected=None, require_final_modes=True)
    assert validation.total_episodes == 2
    assert validation.total_frames == 5
    info = json.loads((output / "meta/info.json").read_text())
    assert info["features"]["observation.images.egocentric"]["video_info"] == {
        "has_audio": False,
        "video.channels": 3,
        "video.codec": "h264",
        "video.fps": 4.0,
        "video.height": 360,
        "video.is_depth_map": False,
        "video.pix_fmt": "yuv420p",
        "video.width": 640,
    }
    assert info["features"]["observation.images.egocentric"]["shape"] == [
        360,
        640,
        3,
    ]
    certificates = sorted(tmp_path.glob(".processed.certification-*"))
    assert len(certificates) == 1
    terminal = validate_evidence_terminal(certificates[0])
    assert terminal["verdict"] == "PASS"
    loader = json.loads((certificates[0] / "psi0-loader-result.json").read_text())
    assert loader["result"]["visited_indices"] == list(range(5))
    assert import_marker.read_text() == "imported"
    assert {
        source: _tree_snapshot(source) for source in (source_a, source_b)
    } == source_before
    assert _direct_children() == children_before

    with converter.conversion_lock(output):
        second = certify(
            publication,
            psi0_root=checkout,
            psi0_commit=commit,
            python=Path(sys.executable),
            certificate_uuid=uuid.UUID("00000000-0000-0000-0000-000000000009"),
        )
    assert second != certificates[0]
    assert validate_evidence_terminal(second)["verdict"] == "PASS"
    assert len(list(tmp_path.glob(".processed.certification-*"))) == 2


def test_run_conversion_preflight_failure_creates_no_staging(tmp_path, monkeypatch):
    source = make_source_episode(tmp_path / "source", frames=2, fps=4)
    output = tmp_path / "processed"
    monkeypatch.setattr(
        converter,
        "resolve_converter_identity",
        lambda *_: converter.ConverterIdentity("a" * 40, "b" * 64),
    )
    args = converter.build_parser().parse_args(_conversion_argv(source, output))
    args.skip = 2

    with pytest.raises(ValueError, match="greater than skip"):
        converter.run_conversion(args)

    assert not output.exists()
    assert not list(tmp_path.glob(".processed.staging-*"))


def test_run_conversion_generation_failure_preserves_failed_staging(
    tmp_path,
    monkeypatch,
):
    source = make_source_episode(tmp_path / "source", frames=2, fps=4)
    output = tmp_path / "processed"
    monkeypatch.setattr(
        converter,
        "resolve_converter_identity",
        lambda *_: converter.ConverterIdentity("a" * 40, "b" * 64),
    )
    monkeypatch.setattr(
        converter,
        "generate_staged_dataset",
        lambda *_: (_ for _ in ()).throw(RuntimeError("injected generation failure")),
    )
    args = converter.build_parser().parse_args(_conversion_argv(source, output))

    with pytest.raises(RuntimeError, match="injected generation failure"):
        converter.run_conversion(args)

    staging = next(tmp_path.glob(".processed.staging-*"))
    assert (
        json.loads((staging / "CONVERSION_STATUS.json").read_text())["state"]
        == "failed"
    )
    assert not output.exists()


@pytest.mark.parametrize(
    ("fault_point", "error_type", "classification", "published"),
    [
        (
            "after_payload_close",
            converter.PublicationStateError,
            "pre_completion_failed",
            False,
        ),
        (
            "after_complete_temp_creation",
            converter.PublicationStateError,
            "completion_uncertain_unpublished",
            False,
        ),
        (
            "after_complete_root_fsync",
            converter.PublicationStateError,
            "completion_uncertain_unpublished",
            False,
        ),
        (
            "after_publication_rename",
            converter.PublicationUncertainError,
            "publication_uncertain",
            True,
        ),
    ],
)
def test_run_conversion_negative_durability_matrix(
    tmp_path,
    monkeypatch,
    fault_point,
    error_type,
    classification,
    published,
):
    source = make_source_episode(tmp_path / "source", frames=2, fps=4)
    output = tmp_path / "processed"
    monkeypatch.setattr(
        converter,
        "resolve_converter_identity",
        lambda *_: converter.ConverterIdentity("a" * 40, "b" * 64),
    )

    def inject(point):
        if point == fault_point:
            raise RuntimeError(f"injected {point}")

    monkeypatch.setattr(converter, "no_fault", inject)
    args = converter.build_parser().parse_args(_conversion_argv(source, output))

    with pytest.raises(error_type) as caught:
        converter.run_conversion(args)

    if published:
        assert output.is_dir()
        assert not list(tmp_path.glob(".processed.staging-*"))
    else:
        staging = next(tmp_path.glob(".processed.staging-*"))
        report = converter.inspect_preserved_staging(staging, output)
        assert report["failure_classification"] == classification
        if fault_point == "after_complete_root_fsync":
            assert caught.value.state == "complete_unpublished"


def test_run_conversion_destination_collision_preserves_both_roots(
    tmp_path,
    monkeypatch,
):
    source = make_source_episode(tmp_path / "source", frames=2, fps=4)
    output = tmp_path / "processed"
    monkeypatch.setattr(
        converter,
        "resolve_converter_identity",
        lambda *_: converter.ConverterIdentity("a" * 40, "b" * 64),
    )
    actual_validate = converter.validate_staged_dataset

    def validate_then_collide(staging, plan):
        result = actual_validate(staging, plan)
        output.mkdir()
        (output / "sentinel").write_bytes(b"collision")
        return result

    monkeypatch.setattr(converter, "validate_staged_dataset", validate_then_collide)
    args = converter.build_parser().parse_args(_conversion_argv(source, output))

    with pytest.raises(converter.PublicationStateError) as caught:
        converter.run_conversion(args)

    assert caught.value.state == "complete_unpublished"
    assert (output / "sentinel").read_bytes() == b"collision"
    assert len(list(tmp_path.glob(".processed.staging-*"))) == 1


def test_run_conversion_certification_failure_keeps_published_output(
    tmp_path,
    monkeypatch,
):
    source = make_source_episode(tmp_path / "source", frames=2, fps=4)
    output = tmp_path / "processed"
    psi0_root = tmp_path / "psi0"
    psi0_root.mkdir()
    monkeypatch.setattr(
        converter,
        "resolve_converter_identity",
        lambda *_: converter.ConverterIdentity("a" * 40, "b" * 64),
    )
    monkeypatch.setattr(
        converter,
        "certify_published_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected certification failure")
        ),
    )
    args = converter.build_parser().parse_args(
        [
            *_conversion_argv(source, output),
            "--certify-psi0-root",
            str(psi0_root),
            "--certify-psi0-commit",
            "c" * 40,
            "--certify-python",
            sys.executable,
        ]
    )

    with pytest.raises(RuntimeError, match="injected certification failure"):
        converter.run_conversion(args)

    assert output.is_dir()
    assert not list(tmp_path.glob(".processed.staging-*"))
