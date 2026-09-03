#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


@dataclass(frozen=True)
class ConverterIdentity:
    commit: str
    script_sha256: str


@dataclass(frozen=True)
class MediaIdentity:
    codec_name: str
    pixel_format: str
    width: int
    height: int
    average_frame_rate: str
    nominal_frame_rate: str
    duration: str
    frame_count: int
    audio_streams: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class MediaProfile:
    codec_name: str
    pixel_format: str
    width: int
    height: int
    average_frame_rate: str
    nominal_frame_rate: str
    audio_streams: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class EpisodePlan:
    source_root: Path
    source_episode_index: int
    output_episode_index: int
    parquet_path: Path
    video_path: Path
    parquet_sha256: str
    video_sha256: str
    frame_count: int
    retained_indices: np.ndarray
    source_task_index: int
    output_task_index: int
    task_text: str
    environment_config: str
    source_media: MediaIdentity


@dataclass(frozen=True)
class ConversionPlan:
    output_path: Path
    output_fps: Fraction
    skip: int
    downsample: int
    chunks_size: int
    video_key: str
    media_mode: Literal["copy_all", "transcode_all"]
    output_media: MediaProfile
    tasks: tuple[dict[str, object], ...]
    episodes: tuple[EpisodePlan, ...]
    converter: ConverterIdentity


REQUIRED_SOURCE_TYPES = {
    "observation.joint_qpos": pa.list_(pa.float32(), 43),
    "observation.amo_policy_command": pa.list_(pa.float32(), 9),
    "observation.amo_policy_target_yaw": pa.float32(),
    "observation.amo_policy_turning_flag": pa.float32(),
    "action": pa.list_(pa.float32(), 43),
    "task_index": pa.int64(),
}


# joint orders are re-ordered to match G1Sonic.joint_names
STATE_SLICES = [
    ("left_hand_thumb", 29, 32),
    ("left_hand_middle", 34, 36),
    ("left_hand_index", 32, 34),
    ("right_hand", 36, 43),
    ("left_arm", 15, 22),
    ("right_arm", 22, 29),
]

ACTION_SLICES = [
    ("left_hand_thumb", 29, 32),
    ("left_hand_middle", 34, 36),
    ("left_hand_index", 32, 34),
    ("right_hand", 36, 43),
    ("left_arm", 15, 22),
    ("right_arm", 22, 29),
    ("torso_rp", 13, 15),  # waist roll/pitch
    ("torso_y", 12, 13),  # waist yaw
]


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    def _json_default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, separators=(",", ":"), default=_json_default)
            f.write("\n")


def canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    separators = None if pretty else (",", ":")
    text = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=separators,
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_rate(value: str) -> Fraction:
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid media rate: {value!r}") from exc
    if result <= 0:
        raise ValueError(f"media rate must be positive: {value!r}")
    return result


def _required_string(mapping: dict[str, object], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"media field {field!r} must be a nonempty string")
    return value


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"media field {field!r} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"media field {field!r} must be a positive integer") from exc
    if result <= 0 or str(result) != str(value):
        raise ValueError(f"media field {field!r} must be a positive integer")
    return result


def _optional_frame_count(value: object, field: str) -> int | None:
    if value is None or value in ("", "N/A"):
        return None
    return _positive_integer(value, field)


def _positive_duration(value: object) -> str:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ValueError("media duration must be a positive finite number")
    try:
        duration = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("media duration must be a positive finite number") from exc
    if duration <= 0:
        raise ValueError("media duration must be a positive finite number")
    return str(duration)


def probe_media(path: Path) -> MediaIdentity:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required but was not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe failed for source video: {path}") from exc

    try:
        payload = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("ffprobe returned malformed JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("streams"), list):
        raise ValueError("ffprobe output must contain a stream list")
    streams = payload["streams"]
    if not all(isinstance(stream, dict) for stream in streams):
        raise ValueError("ffprobe stream entries must be objects")
    video_streams = [
        stream for stream in streams if stream.get("codec_type") == "video"
    ]
    if len(video_streams) != 1:
        raise ValueError("source media must contain exactly one video stream")
    video = video_streams[0]

    average_rate = parse_rate(_required_string(video, "avg_frame_rate"))
    nominal_rate = parse_rate(_required_string(video, "r_frame_rate"))
    if average_rate != nominal_rate:
        raise ValueError("source video frame rate is ambiguous")

    read_frames = _optional_frame_count(video.get("nb_read_frames"), "nb_read_frames")
    declared_frames = _optional_frame_count(video.get("nb_frames"), "nb_frames")
    if read_frames is None:
        if declared_frames is None:
            raise ValueError("source video frame count is unavailable")
        frame_count = declared_frames
    else:
        if declared_frames is not None and declared_frames != read_frames:
            raise ValueError("source video frame counts disagree")
        frame_count = read_frames

    duration_value = video.get("duration")
    if duration_value in (None, "", "N/A"):
        media_format = payload.get("format")
        if not isinstance(media_format, dict):
            raise ValueError("ffprobe output is missing media duration")
        duration_value = media_format.get("duration")
    duration = _positive_duration(duration_value)

    audio_streams = []
    for stream in streams:
        if stream.get("codec_type") != "audio":
            continue
        audio_streams.append(
            {
                "codec_name": _required_string(stream, "codec_name"),
                "sample_fmt": _required_string(stream, "sample_fmt"),
                "sample_rate": _positive_integer(
                    stream.get("sample_rate"), "sample_rate"
                ),
                "channels": _positive_integer(stream.get("channels"), "channels"),
                "channel_layout": _required_string(stream, "channel_layout"),
            }
        )

    return MediaIdentity(
        codec_name=_required_string(video, "codec_name"),
        pixel_format=_required_string(video, "pix_fmt"),
        width=_positive_integer(video.get("width"), "width"),
        height=_positive_integer(video.get("height"), "height"),
        average_frame_rate=str(average_rate),
        nominal_frame_rate=str(nominal_rate),
        duration=duration,
        frame_count=frame_count,
        audio_streams=tuple(audio_streams),
    )


def _require_nonnegative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _require_positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"could not read metadata: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"metadata must contain a JSON object: {path}")
    return value


def _task_mapping(path: Path) -> dict[int, str]:
    rows = load_jsonl(path)
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("task metadata rows must be objects")
        task_index = row.get("task_index")
        if type(task_index) is not int or task_index < 0:
            raise ValueError("task_index must be a nonnegative integer")
        task = row.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a nonempty string")
        if task_index in result:
            raise ValueError(f"duplicate task_index: {task_index}")
        result[task_index] = task
    return result


def _episode_mapping(path: Path) -> dict[int, dict[str, object]]:
    rows = load_jsonl(path)
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("episode metadata rows must be objects")
        episode_index = row.get("episode_index")
        if type(episode_index) is not int or episode_index < 0:
            raise ValueError("episode_index must be a nonnegative integer")
        length = row.get("length")
        if type(length) is not int or length <= 0:
            raise ValueError("episode length must be a positive integer")
        tasks = row.get("tasks")
        if (
            not isinstance(tasks, list)
            or not tasks
            or any(not isinstance(task, str) or not task.strip() for task in tasks)
        ):
            raise ValueError("episode tasks must be a nonempty list of strings")
        environment_config = row.get("environment_config")
        if not isinstance(environment_config, str):
            raise ValueError("environment_config must be a JSON string")

        def reject_nonfinite_json(value: str) -> None:
            raise ValueError(f"invalid JSON constant: {value}")

        try:
            json.loads(environment_config, parse_constant=reject_nonfinite_json)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("environment_config must contain valid JSON") from exc
        if episode_index in result:
            raise ValueError(f"duplicate episode_index: {episode_index}")
        result[episode_index] = row
    return result


def _media_profile(identity: MediaIdentity) -> MediaProfile:
    return MediaProfile(
        codec_name=identity.codec_name,
        pixel_format=identity.pixel_format,
        width=identity.width,
        height=identity.height,
        average_frame_rate=identity.average_frame_rate,
        nominal_frame_rate=identity.nominal_frame_rate,
        audio_streams=identity.audio_streams,
    )


def _validate_timestamp_range(output_fps: Fraction, row_count: int) -> None:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            maximum = np.asarray(
                [Fraction(row_count - 1, 1) / output_fps], dtype=np.float32
            )
    except (OverflowError, ValueError) as exc:
        raise ValueError("requested FPS produces nonfinite float32 timestamps") from exc
    if not np.isfinite(maximum).all():
        raise ValueError("requested FPS produces nonfinite float32 timestamps")


def preflight_conversion(
    args: argparse.Namespace, converter: ConverterIdentity
) -> ConversionPlan:
    skip = _require_nonnegative_integer(args.skip, "skip")
    downsample = _require_positive_integer(args.downsample, "downsample")
    chunks_size = _require_positive_integer(args.chunks_size, "chunks_size")
    total_episodes = _require_positive_integer(args.total_episodes, "total_episodes")
    output_fps = parse_rate(str(args.fps))
    video_key = args.video_key
    if (
        not isinstance(video_key, str)
        or not video_key.strip()
        or Path(video_key).name != video_key
    ):
        raise ValueError("video_key must be one nonempty path component")

    requested_output = Path(args.out_dir).expanduser()
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError(f"destination already exists: {requested_output}")
    output_path = requested_output.resolve(strict=False)

    matched_roots = glob.glob(str(args.sim_root))
    if not matched_roots:
        raise ValueError(f"no source roots matched: {args.sim_root}")
    source_roots = sorted(
        {Path(match).expanduser().resolve(strict=True) for match in matched_roots}
    )

    candidates = []
    for source_root in source_roots:
        if not source_root.is_dir():
            raise ValueError(f"source root is not a directory: {source_root}")
        metadata_paths = {
            name: source_root / "meta" / name
            for name in ("info.json", "tasks.jsonl", "episodes.jsonl")
        }
        for path in metadata_paths.values():
            if not path.is_file():
                raise ValueError(f"required metadata file is missing: {path}")
        _load_json_object(metadata_paths["info.json"])
        tasks = _task_mapping(metadata_paths["tasks.jsonl"])
        episodes = _episode_mapping(metadata_paths["episodes.jsonl"])

        data_files = sorted((source_root / "data").glob("chunk-*/episode_*.parquet"))
        seen_episode_indices = set()
        for parquet_path in data_files:
            match = re.fullmatch(r"episode_(\d+)", parquet_path.stem)
            if match is None:
                raise ValueError(f"invalid source episode filename: {parquet_path}")
            source_episode_index = int(match.group(1))
            if source_episode_index in seen_episode_indices:
                raise ValueError(
                    f"duplicate source episode index: {source_episode_index}"
                )
            seen_episode_indices.add(source_episode_index)
            candidates.append(
                (
                    source_root,
                    source_episode_index,
                    parquet_path.resolve(strict=True),
                    tasks,
                    episodes,
                )
            )

    selected_candidates = candidates[:total_episodes]
    if not selected_candidates:
        raise ValueError("no source episodes were found")

    plans = []
    output_tasks = []
    output_task_by_text = {}
    for (
        source_root,
        source_episode_index,
        parquet_path,
        tasks,
        episodes,
    ) in selected_candidates:
        schema = pq.read_schema(parquet_path)
        for name, expected_type in REQUIRED_SOURCE_TYPES.items():
            index = schema.get_field_index(name)
            if index < 0:
                raise ValueError(f"required source column is missing: {name}")
            actual_type = schema.field(index).type
            if actual_type != expected_type:
                raise ValueError(
                    f"source column {name!r} has type {actual_type}, "
                    f"expected {expected_type}"
                )

        parquet_file = pq.ParquetFile(parquet_path)
        frame_count = parquet_file.metadata.num_rows
        task_table = pq.read_table(parquet_path, columns=["task_index"])
        if len(task_table) != frame_count:
            raise ValueError("task_index row count does not match Parquet metadata")
        task_values = task_table["task_index"].to_pylist()
        if not task_values or any(type(value) is not int for value in task_values):
            raise ValueError("task_index values must be int64 values")
        source_task_index = task_values[0]
        if any(value != source_task_index for value in task_values):
            raise ValueError("task_index must be constant within an episode")

        retained_indices = make_retained_indices(frame_count, skip, downsample)
        retained_indices.setflags(write=False)
        _validate_timestamp_range(output_fps, len(retained_indices))

        video_candidates = sorted(
            path.resolve(strict=True)
            for path in (source_root / "videos").glob(
                f"chunk-*/{video_key}/episode_{source_episode_index:06d}.mp4"
            )
            if path.is_file()
        )
        if len(video_candidates) != 1:
            raise ValueError(
                "source episode must have exactly one matching video candidate"
            )
        video_path = video_candidates[0]
        source_media = probe_media(video_path)
        if source_media.frame_count != frame_count:
            raise ValueError("source video and Parquet frame counts differ")

        task_text = tasks.get(source_task_index)
        if task_text is None:
            raise ValueError(f"task metadata lookup failed: {source_task_index}")
        episode = episodes.get(source_episode_index)
        if episode is None:
            raise ValueError(f"episode metadata lookup failed: {source_episode_index}")
        if episode["length"] != frame_count:
            raise ValueError("episode metadata length does not match Parquet rows")
        if task_text not in episode["tasks"]:
            raise ValueError("episode task metadata does not match task_index")

        output_task_index = output_task_by_text.get(task_text)
        if output_task_index is None:
            output_task_index = len(output_tasks)
            output_task_by_text[task_text] = output_task_index
            output_tasks.append({"task_index": output_task_index, "task": task_text})

        plans.append(
            EpisodePlan(
                source_root=source_root,
                source_episode_index=source_episode_index,
                output_episode_index=len(plans),
                parquet_path=parquet_path,
                video_path=video_path,
                parquet_sha256=sha256_file(parquet_path),
                video_sha256=sha256_file(video_path),
                frame_count=frame_count,
                retained_indices=retained_indices,
                source_task_index=source_task_index,
                output_task_index=output_task_index,
                task_text=task_text,
                environment_config=episode["environment_config"],
                source_media=source_media,
            )
        )

    source_profiles = [_media_profile(plan.source_media) for plan in plans]
    requested_rate = str(output_fps)
    copy_all = (
        skip == 0
        and downsample == 1
        and all(profile == source_profiles[0] for profile in source_profiles)
        and source_profiles[0].average_frame_rate == requested_rate
        and source_profiles[0].nominal_frame_rate == requested_rate
    )
    if copy_all:
        media_mode: Literal["copy_all", "transcode_all"] = "copy_all"
        output_media = source_profiles[0]
    else:
        media_mode = "transcode_all"
        output_media = MediaProfile(
            codec_name="h264",
            pixel_format="yuv420p",
            width=640,
            height=360,
            average_frame_rate=requested_rate,
            nominal_frame_rate=requested_rate,
            audio_streams=(),
        )

    return ConversionPlan(
        output_path=output_path,
        output_fps=output_fps,
        skip=skip,
        downsample=downsample,
        chunks_size=chunks_size,
        video_key=video_key,
        media_mode=media_mode,
        output_media=output_media,
        tasks=tuple(output_tasks),
        episodes=tuple(plans),
        converter=converter,
    )


def resolve_converter_commit(repository_root: Path) -> str:
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", "scripts/postprocess_psi0.py"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("could not resolve converter source commit")
    return commit


def resolve_converter_identity(
    repository_root: Path, script_path: Path
) -> ConverterIdentity:
    repository_root = repository_root.resolve(strict=True)
    script_path = script_path.resolve(strict=True)
    expected = repository_root / "scripts" / "postprocess_psi0.py"
    if script_path != expected:
        raise RuntimeError(f"unexpected converter path: {script_path}")
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", "scripts/postprocess_psi0.py"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("could not resolve converter source commit")
    committed = subprocess.run(
        ["git", "show", f"{commit}:scripts/postprocess_psi0.py"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    ).stdout
    executed = script_path.read_bytes()
    if executed != committed:
        raise RuntimeError("executed converter differs from its recorded Git blob")
    return ConverterIdentity(commit=commit, script_sha256=sha256_bytes(executed))


def make_retained_indices(frame_count: int, skip: int, downsample: int) -> np.ndarray:
    if skip < 0:
        raise ValueError("skip must be nonnegative")
    if downsample <= 0:
        raise ValueError("downsample must be positive")
    if frame_count <= skip:
        raise ValueError("frame_count must be greater than skip")
    result = np.arange(skip, frame_count, downsample, dtype=np.int64)
    if result.size == 0 or not np.all(result[1:] > result[:-1]):
        raise ValueError("retained indices must be nonempty and strictly increasing")
    return result


def require_finite(name: str, values: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    if not np.issubdtype(result.dtype, np.floating):
        raise ValueError(f"{name} must be floating point")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite values")
    return result


def output_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("states", pa.list_(pa.float32(), 32)),
            pa.field("action", pa.list_(pa.float32(), 36)),
            pa.field("observation.hand_joints", pa.list_(pa.float32(), 14)),
            pa.field("observation.arm_joints", pa.list_(pa.float32(), 14)),
            pa.field("observation.leg_joints", pa.list_(pa.float32(), 15)),
            pa.field("observation.prev_torso_rpy", pa.list_(pa.float32(), 3)),
            pa.field("observation.prev_height", pa.list_(pa.float32(), 1)),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
            pa.field("next.done", pa.bool_()),
        ]
    )


def build_conversion_provenance(
    source_path: Path,
    source_episode_index: int,
    skip: int,
    downsample: int,
    converter_commit: str,
) -> dict:
    if re.fullmatch(r"[0-9a-f]{40}", converter_commit) is None:
        raise ValueError("converter_commit must be a lowercase 40-character Git SHA")
    return {
        "source_episode_index": source_episode_index,
        "source_parquet_sha256": sha256_file(source_path),
        "skip": skip,
        "downsample": downsample,
        "converter_commit": converter_commit,
    }


def modality_dict():
    def _entry(start, end, original_key, absolute=True):
        return {
            "start": start,
            "end": end,
            "rotation_type": None,
            "absolute": absolute,
            "dtype": "float32",
            "original_key": original_key,
        }

    return {
        "state": {
            "left_hand": _entry(0, 7, "states"),
            "right_hand": _entry(7, 14, "states"),
            "left_arm": _entry(14, 21, "states"),
            "right_arm": _entry(21, 28, "states"),
            "rpy": _entry(28, 31, "states"),
            "height": _entry(31, 32, "states"),
        },
        "action": {
            "left_hand": _entry(0, 7, "action"),
            "right_hand": _entry(7, 14, "action"),
            "left_arm": _entry(14, 21, "action"),
            "right_arm": _entry(21, 28, "action"),
            "rpy": _entry(28, 31, "action"),
            "height": _entry(31, 32, "action"),
            "torso_vx": _entry(32, 33, "action", absolute=False),
            "torso_vy": _entry(33, 34, "action", absolute=False),
            "torso_vyaw": _entry(34, 35, "action", absolute=False),
            "target_yaw": _entry(35, 36, "action"),
        },
        "video": {"rs_view": {"original_key": "observation.images.egocentric"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def build_vectors(proprio, cmd, history_cmd, action, target_yaw, turning_flag):
    to = proprio.shape[0]

    # states: match to_psi0_state_format ordering
    states = np.concatenate(
        [proprio[:, s:e] for _, s, e in STATE_SLICES]
        + [
            history_cmd[:to, 3:6][:, ::-1],  # torso_rpy
            history_cmd[:to, 6:7],  # base height
        ],
        axis=1,
    ).astype(np.float32)

    # actions: match to_psi0_action_format ordering
    actions = np.concatenate(
        [action[:, s:e] for _, s, e in ACTION_SLICES]
        + [
            cmd[:, 6:7],  # base height
            cmd[:, 0:2],  # vx, vy
            turning_flag.reshape(-1, 1)[:, 0:1],  # turning flag
            target_yaw.reshape(-1, 1)[:, 0:1],  # target yaw
        ],
        axis=1,
    ).astype(np.float32)
    return states, actions


def build_proprio_obs(proprio, history_cmd):
    # hand joints: left thumb(29:32), left index(32:34), left middle(34:36),
    #              right thumb(36:39), right index(39:41), right middle(41:43)
    hand = np.concatenate(
        [
            proprio[:, 29:32],
            proprio[:, 32:34],
            proprio[:, 34:36],
            proprio[:, 36:39],
            proprio[:, 39:41],
            proprio[:, 41:43],
        ],
        axis=1,
    ).astype(np.float32)

    # arm joints: left arm(15:22), right arm(22:29)
    arm = np.concatenate([proprio[:, 15:22], proprio[:, 22:29]], axis=1).astype(
        np.float32
    )

    # leg joints: 12 leg joints + 3 waist joints (12:15)
    leg = np.concatenate([proprio[:, 0:12], proprio[:, 12:15]], axis=1).astype(
        np.float32
    )

    # torso rpy
    torso_rpy = history_cmd[: proprio.shape[0], 3:6][:, ::-1].astype(np.float32)
    # base height from command
    prev_height = history_cmd[: proprio.shape[0], 6:7].astype(np.float32)

    return hand, arm, leg, torso_rpy, prev_height


def stats_block(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    array = require_finite("statistics input", np.asarray(values, dtype=np.float32))
    if array.shape[0] == 0:
        raise ValueError("statistics input is empty")
    if array.ndim == 1:
        array = array[:, None]
    block = {
        "mean": array.mean(0, dtype=np.float64).astype(np.float32).tolist(),
        "std": array.std(0, dtype=np.float64).astype(np.float32).tolist(),
        "min": array.min(0).astype(np.float32).tolist(),
        "max": array.max(0).astype(np.float32).tolist(),
        "q01": np.quantile(array, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).astype(np.float32).tolist(),
        "count": [int(array.shape[0])],
    }
    canonical_json_bytes(block)
    return block


def build_output_table(
    *,
    vectors: dict[str, np.ndarray],
    output_episode_index: int,
    global_offset: int,
    output_task_index: int,
    output_fps: Fraction,
) -> pa.Table:
    if output_fps <= 0:
        raise ValueError("output_fps must be positive")

    vector_widths = {
        "states": 32,
        "action": 36,
        "observation.hand_joints": 14,
        "observation.arm_joints": 14,
        "observation.leg_joints": 15,
        "observation.prev_torso_rpy": 3,
        "observation.prev_height": 1,
    }
    if set(vectors) != set(vector_widths):
        raise ValueError("vectors must contain exactly the output vector fields")

    arrays = []
    row_count = None
    for name, width in vector_widths.items():
        values = np.ascontiguousarray(vectors[name], dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"{name} must have shape (n, {width})")
        if row_count is None:
            row_count = values.shape[0]
        if values.shape != (row_count, width):
            raise ValueError(f"{name} must have shape ({row_count}, {width})")
        require_finite(name, values)
        flattened = pa.array(values.reshape(-1), type=pa.float32())
        arrays.append(pa.FixedSizeListArray.from_arrays(flattened, width))

    assert row_count is not None
    timestamp_quotients = [
        Fraction(frame, 1) / output_fps for frame in range(row_count)
    ]
    with np.errstate(over="ignore"):
        timestamp = np.asarray(timestamp_quotients, dtype=np.float32)
    require_finite("timestamp", timestamp)
    frame_index = np.arange(row_count, dtype=np.int64)
    episode_index = np.full(row_count, output_episode_index, dtype=np.int64)
    index = np.arange(global_offset, global_offset + row_count, dtype=np.int64)
    task_index = np.full(row_count, output_task_index, dtype=np.int64)
    done = np.zeros(row_count, dtype=np.bool_)
    if row_count:
        done[-1] = True

    arrays.extend(
        [
            pa.array(timestamp, type=pa.float32()),
            pa.array(frame_index, type=pa.int64()),
            pa.array(episode_index, type=pa.int64()),
            pa.array(index, type=pa.int64()),
            pa.array(task_index, type=pa.int64()),
            pa.array(done, type=pa.bool_()),
        ]
    )
    return pa.Table.from_arrays(arrays, schema=output_schema())


# default history command for the initial state
initial_command = np.array([0, 0, 0, 0, 0, 0, 0.74, 0.74, 0.74], dtype=np.float32)
default_fps = 50


def write_downsampled_video(
    src_video: Path, dst_video: Path, skip: int, downsample: int, fps: int
):
    if skip < 0:
        raise ValueError(f"skip must be >= 0, got {skip}")
    if downsample <= 0:
        raise ValueError(f"downsample must be > 0, got {downsample}")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    if skip == 0 and downsample == 1:
        shutil.copyfile(src_video, dst_video)
        return

    vf = (
        f"select='gte(n\\,{skip})*not(mod(n-{skip}\\,{downsample}))',"
        f"setpts=N/({fps}*TB)"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src_video),
        "-an",
        "-vf",
        vf,
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dst_video),
    ]

    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required to write downsampled videos but was not found in PATH"
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for source video: {src_video} -> {dst_video} (exit code {result.returncode})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim-root",
        required=True,
        help="Path or glob pattern, e.g. data/datagen*/simple/G1WholebodyBendPick-v0",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--skip", type=int, default=60)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--total_episodes", type=int, default=100)
    parser.add_argument("--fps", type=int, default=default_fps)
    parser.add_argument("--video-key", default="observation.rgb_head_stereo_left")
    parser.add_argument("--chunks-size", type=int, default=1000)
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    converter_commit = resolve_converter_commit(repository_root)

    # last_episode_idx = 0
    episode_idx = 0
    last_index = 0
    all_tasks = []

    total_frames = 0
    episodes = []
    episode_stats_rows = []

    all_states = []
    all_actions = []
    all_timestamp = []
    all_frame_index = []
    all_episode_index = []
    all_index = []
    all_task_index = []
    all_done = []

    out_dir = Path(args.out_dir).resolve()
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    print("created output directories at", out_dir)

    sim_info = None

    all_sim_roots = sorted(Path(p).resolve() for p in glob.glob(args.sim_root))
    for sim_root in all_sim_roots:
        if episode_idx >= args.total_episodes:
            print(
                f"Reached total_episodes={args.total_episodes}, stopping further processing."
            )
            break

        print(f"Merging data: {sim_root}")

        sim_info = json.loads((sim_root / "meta" / "info.json").read_text())
        sim_tasks = load_jsonl(sim_root / "meta" / "tasks.jsonl")
        episodes_info = load_jsonl(sim_root / "meta" / "episodes.jsonl")

        original_fps = float(sim_info["fps"])
        if args.fps / original_fps != args.downsample:
            print(
                f"Warning: The specified fps {args.fps} is not consistent with the original fps {original_fps} and downsample factor {args.downsample}."
                f"The timestamps will be computed based on the specified fps."
            )

        curr_task_index_to_new_task_index = {}

        def merge_task(task, all_tasks):
            for t in all_tasks:
                if t["task"] == task:
                    return t["task_index"]
            all_tasks.append(
                {
                    "task_index": len(all_tasks),
                    "task": task,
                }
            )
            return len(all_tasks) - 1

        for t in sim_tasks:
            task_index = t["task_index"]
            task = t["task"]
            curr_task_index_to_new_task_index[task_index] = merge_task(task, all_tasks)

        data_files = sorted((sim_root / "data").glob("chunk-*/episode_*.parquet"))
        for data_path in tqdm(data_files):  # for each episode
            if episode_idx >= args.total_episodes:
                print(
                    f"Reached total_episodes={args.total_episodes}, stopping further processing."
                )
                break

            ep_index = int(data_path.stem.split("_")[-1])
            chunk_id = episode_idx // args.chunks_size

            table = pq.read_table(data_path)
            proprio = np.asarray(
                table["observation.joint_qpos"].to_pylist(), dtype=np.float32
            )
            cmd = np.asarray(
                table["observation.amo_policy_command"].to_pylist(), dtype=np.float32
            )
            target_yaw = np.asarray(
                table["observation.amo_policy_target_yaw"].to_pylist(), dtype=np.float32
            )
            turning_flag = np.asanyarray(
                table["observation.amo_policy_turning_flag"].to_pylist(),
                dtype=np.float32,
            )
            action = np.asarray(table["action"].to_pylist(), dtype=np.float32)

            history_cmd = np.concatenate([initial_command[None, :], cmd[:-1]], axis=0)
            states, actions = build_vectors(
                proprio, cmd, history_cmd, action, target_yaw, turning_flag
            )
            hand_joints, arm_joints, leg_joints, torso_rpy, prev_height = (
                build_proprio_obs(proprio, history_cmd)
            )

            m = states.shape[0]
            assert m >= args.skip, (
                f"Episode {episode_idx} has only {m} frames, which is less than skip={args.skip}"
            )
            n = (m - args.skip) // args.downsample

            done = np.zeros((n,), dtype=bool)
            if n > 0:
                done[-1] = True

            # frame_index = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
            frame_index = np.asarray(range(n), dtype=np.int64)

            # episode_index = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
            episode_index = np.asarray([episode_idx] * n, dtype=np.int64)

            # index = np.asarray(table["index"].to_pylist(), dtype=np.int64)
            index = np.asarray(table["index"].to_pylist(), dtype=np.int64) + last_index

            # timesteps = np.asarray([round(ts * original_fps) for ts in table["timestamp"].to_pylist()], dtype=np.float32)
            timestamp = frame_index * 1.0 / args.fps

            curr_task_indices = np.asarray(
                table["task_index"].to_pylist(), dtype=np.int64
            )
            assert np.all(curr_task_indices == curr_task_indices[0]), (
                f"Episode {episode_idx} has multiple task indices: {set(curr_task_indices)}"
            )
            new_task_index = curr_task_index_to_new_task_index[curr_task_indices[0]]
            task_index = np.asarray([new_task_index] * n, dtype=np.int64)

            out_table = pa.table(
                {
                    "states": states[args.skip :][:: args.downsample].tolist(),
                    "action": actions[args.skip :][:: args.downsample].tolist(),
                    "observation.hand_joints": hand_joints[args.skip :][
                        :: args.downsample
                    ].tolist(),
                    "observation.arm_joints": arm_joints[args.skip :][
                        :: args.downsample
                    ].tolist(),
                    "observation.leg_joints": leg_joints[args.skip :][
                        :: args.downsample
                    ].tolist(),
                    "observation.prev_torso_rpy": torso_rpy[args.skip :][
                        :: args.downsample
                    ].tolist(),
                    "observation.prev_height": prev_height[args.skip :][
                        :: args.downsample
                    ].tolist(),
                    "timestamp": timestamp,
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "index": index[args.skip :][:: args.downsample],
                    "task_index": task_index,
                    "next.done": done,
                }
            )

            out_data_dir = out_dir / "data" / f"chunk-{chunk_id:03d}"
            out_data_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                out_table, out_data_dir / f"episode_{episode_idx:06d}.parquet"
            )

            src_chunk = data_path.parent.name
            src_episode_idx = int(data_path.stem.split("_")[-1])
            src_video = (
                sim_root
                / "videos"
                / src_chunk
                / args.video_key
                / f"episode_{src_episode_idx:06d}.mp4"
            )
            dst_video_dir = out_dir / "videos" / f"chunk-{chunk_id:03d}" / "egocentric"
            dst_video_dir.mkdir(parents=True, exist_ok=True)
            dst_video = dst_video_dir / f"episode_{episode_idx:06d}.mp4"
            if src_video.exists():
                write_downsampled_video(
                    src_video=src_video,
                    dst_video=dst_video,
                    skip=args.skip,
                    downsample=args.downsample,
                    fps=args.fps,
                )

            total_frames += n
            ep_task = task_index[0]  # ) if len(task_index) else 0
            conversion_provenance = build_conversion_provenance(
                source_path=data_path,
                source_episode_index=ep_index,
                skip=args.skip,
                downsample=args.downsample,
                converter_commit=converter_commit,
            )
            episodes.append(
                {
                    "episode_index": episode_idx,
                    "tasks": [ep_task],
                    "length": n,
                    "dataset_from_index": total_frames - n,
                    "dataset_to_index": total_frames - 1,
                    "robot_type": "g1",
                    "instruction": all_tasks[task_index[0]],
                    "environment_config": episodes_info[ep_index]["environment_config"],
                    "conversion_provenance": conversion_provenance,
                }
            )

            ep_stats = {
                "episode_index": episode_idx,
                "stats": {
                    "action": {**stats_block(actions), "count": [int(n)]},
                    "timestamp": {**stats_block(timestamp), "count": [int(n)]},
                },
            }
            episode_stats_rows.append(ep_stats)

            all_states.append(states)
            all_actions.append(actions)
            all_timestamp.append(timestamp)
            all_frame_index.append(frame_index)
            all_episode_index.append(episode_index)
            all_index.append(index)
            all_task_index.append(task_index)
            all_done.append(done.astype(np.float32))

            last_index += n
            episode_idx += 1

    all_states = (
        np.concatenate(all_states, axis=0)
        if all_states
        else np.zeros((0, 32), dtype=np.float32)
    )
    all_actions = (
        np.concatenate(all_actions, axis=0)
        if all_actions
        else np.zeros((0, 36), dtype=np.float32)
    )
    all_timestamp = (
        np.concatenate(all_timestamp, axis=0)
        if all_timestamp
        else np.zeros((0,), dtype=np.float32)
    )
    all_frame_index = (
        np.concatenate(all_frame_index, axis=0)
        if all_frame_index
        else np.zeros((0,), dtype=np.float32)
    )
    all_episode_index = (
        np.concatenate(all_episode_index, axis=0)
        if all_episode_index
        else np.zeros((0,), dtype=np.float32)
    )
    all_index = (
        np.concatenate(all_index, axis=0)
        if all_index
        else np.zeros((0,), dtype=np.float32)
    )
    all_task_index = (
        np.concatenate(all_task_index, axis=0)
        if all_task_index
        else np.zeros((0,), dtype=np.float32)
    )
    all_done = (
        np.concatenate(all_done, axis=0)
        if all_done
        else np.zeros((0,), dtype=np.float32)
    )

    task_by_index = {}
    tasks_rows = []
    for t in all_tasks:
        ti = t.get("task_index", 0)
        task_by_index[int(ti)] = t.get("task", "")
        tasks_rows.append(
            {
                "task_index": int(ti),
                "task": t.get("task", ""),
                "category": "",
                "description": t.get("task", ""),
            }
        )

    meta_dir = out_dir / "meta"
    write_jsonl(meta_dir / "tasks.jsonl", tasks_rows)
    print("Wrote tasks.jsonl with", len(tasks_rows), "tasks")
    write_jsonl(
        meta_dir / "episodes.jsonl", sorted(episodes, key=lambda r: r["episode_index"])
    )
    print("Wrote episodes.jsonl with", len(episodes), "episodes")
    write_jsonl(
        meta_dir / "episodes_stats.jsonl",
        sorted(episode_stats_rows, key=lambda r: r["episode_index"]),
    )
    print("Wrote episodes_stats.jsonl")

    assert sim_info is not None
    video_feat = sim_info["features"][args.video_key]

    info = {
        "codebase_version": "v2.1",
        "robot_type": "g1",
        "total_episodes": len(episodes),
        "total_frames": int(total_frames),
        "total_tasks": len(tasks_rows),
        "total_videos": len(episodes),
        "total_chunks": math.ceil(len(episodes) / args.chunks_size)
        if args.chunks_size
        else 1,
        "chunks_size": args.chunks_size,
        "fps": args.fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/egocentric/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.egocentric": {
                "dtype": video_feat.get("dtype", "video"),
                "shape": video_feat.get("shape", [360, 640, 3]),
                "names": ["height", "width", "channel"],
                "video_info": video_feat.get("info", video_feat.get("video_info", {})),
            },
            "observation.hand_joints": {
                "dtype": "float32",
                "shape": [14],
                "names": ["hand_joints"],
            },
            "observation.arm_joints": {
                "dtype": "float32",
                "shape": [14],
                "names": ["arm_joints"],
            },
            "observation.leg_joints": {
                "dtype": "float32",
                "shape": [15],
                "names": ["leg_joints"],
            },
            "observation.prev_torso_rpy": {
                "dtype": "float32",
                "shape": [3],
                "names": ["prev_roll", "prev_pitch", "prev_yaw"],
            },
            "observation.prev_height": {
                "dtype": "float32",
                "shape": [1],
                "names": ["prev_height"],
            },
            "states": {"dtype": "float32", "shape": [-1]},
            "action": {"dtype": "float32", "shape": [-1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4))
    print("Wrote info.json")

    stats = {
        "states": stats_block(all_states),
        "action": stats_block(all_actions),
        "timestamp": stats_block(all_timestamp),
        "frame_index": stats_block(all_frame_index),
        "episode_index": stats_block(all_episode_index),
        "index": stats_block(all_index),
        "task_index": stats_block(all_task_index),
        "next.done": stats_block(all_done),
    }
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=4))
    print("Wrote stats.json")
    (meta_dir / "stats_psi0.json").write_text(json.dumps(stats, indent=4))
    print("Wrote stats_psi0.json")
    (meta_dir / "relative_stats.json").write_text("{}")
    print("Wrote relative_stats.json")
    (meta_dir / "lang_map.json").write_text("{}")
    print("Wrote lang_map.json")
    (meta_dir / "modality.json").write_text(json.dumps(modality_dict(), indent=2))
    print("Wrote modality.json")


if __name__ == "__main__":
    main()
