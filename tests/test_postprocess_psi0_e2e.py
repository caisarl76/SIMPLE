import json
import os
import shutil
import subprocess
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
