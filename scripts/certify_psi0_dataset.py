#!/usr/bin/env python3
"""Independent validation and sibling certification for PSI0 converter output."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import math
import os
import stat
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import NoReturn

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from scripts import postprocess_psi0 as converter  # noqa: E402


_HEX_40 = frozenset("0123456789abcdef")
_FLOAT_COLUMNS = (
    "states",
    "action",
    "observation.hand_joints",
    "observation.arm_joints",
    "observation.leg_joints",
    "observation.prev_torso_rpy",
    "observation.prev_height",
    "timestamp",
)
_GLOBAL_STATS_COLUMNS = (
    "states",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "next.done",
)
_OUTPUT_SCHEMA = pa.schema(
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
_FEATURE_METADATA = {
    "states": {"dtype": "float32", "shape": [32]},
    "action": {"dtype": "float32", "shape": [36]},
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
    "timestamp": {"dtype": "float32", "shape": [1]},
    "frame_index": {"dtype": "int64", "shape": [1]},
    "episode_index": {"dtype": "int64", "shape": [1]},
    "index": {"dtype": "int64", "shape": [1]},
    "task_index": {"dtype": "int64", "shape": [1]},
    "next.done": {"dtype": "bool", "shape": [1]},
}
_INFO_FIELDS = {
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
_FIXED_METADATA = frozenset(
    {
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
        "CONVERSION_STATUS.json",
    }
)
_EVIDENCE_FILES = (
    "certificate-request.json",
    "dataset-root-identity.json",
    "dataset-validation.json",
    "source-validation.json",
    "psi0-environment.json",
    "psi0-loader-command.json",
    "psi0-loader-cache.json",
    "psi0-loader-result.json",
)
_FAILURE_STAGE_BY_COMPLETED_COUNT = (
    "request",
    "dataset_identity",
    "dataset_validation",
    "source_validation",
    "psi0_environment",
    "psi0_loader_command",
    "psi0_loader_cache",
    "psi0_loader_result",
    "terminal",
)
_EPISODE_PROVENANCE_FIELDS = {
    "source_episode_index",
    "source_parquet_sha256",
    "source_video_sha256",
    "requested_video_key",
    "requested_output_fps",
    "skip",
    "downsample",
    "retained_count",
    "source_media",
    "output_video_sha256",
    "output_media",
    "converter_commit",
    "converter_script_sha256",
}
_COMPAT_RELATIVE_PATH = "src/psi/data/lerobot/compat.py"
_MANDATORY_CACHE_DIRECTORIES = frozenset(
    {"home", "hf", "datasets", "xdg", "torch", "tmp"}
)
_INET_SOCKET_AUDIT_EVENTS = frozenset(
    {
        "socket.__new__",
        "socket.bind",
        "socket.connect",
        "socket.connect_ex",
        "socket.sendto",
    }
)
_RESOLVER_AUDIT_EVENTS = frozenset(
    {
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.gethostbyname_ex",
        "socket.getnameinfo",
    }
)
_PROCESS_ESCAPE_AUDIT_EVENTS = frozenset(
    {
        "os.exec",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.posix_spawnp",
        "os.system",
        "subprocess.Popen",
    }
)
_DENIED_LOADER_AUDIT_EVENTS = tuple(
    sorted(
        _INET_SOCKET_AUDIT_EVENTS
        | _RESOLVER_AUDIT_EVENTS
        | _PROCESS_ESCAPE_AUDIT_EVENTS,
        key=str.encode,
    )
)
_LOADER_AUDIT_EVENT_CATEGORIES = {
    **{event: "inet_socket" for event in _INET_SOCKET_AUDIT_EVENTS},
    **{event: "resolver" for event in _RESOLVER_AUDIT_EVENTS},
    **{event: "process_escape" for event in _PROCESS_ESCAPE_AUDIT_EVENTS},
}


@dataclass(frozen=True)
class DatasetExpectations:
    """Immutable generation facts captured by preflight."""

    plan: converter.ConversionPlan

    @classmethod
    def from_plan(cls, plan: converter.ConversionPlan) -> DatasetExpectations:
        return cls(plan=plan)


@dataclass(frozen=True)
class DatasetValidationResult:
    dataset_root: Path
    total_episodes: int
    total_frames: int
    total_tasks: int
    total_videos: int
    tree_digest: str
    manifest_sha256: str | None
    complete_status_sha256: str | None
    source_entries: tuple[dict[str, object], ...]


class DatasetValidationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class PublishedUncertifiedError(RuntimeError):
    def __init__(self, evidence_root: Path, code: str, message: str) -> None:
        absolute_evidence_root = evidence_root.absolute()
        super().__init__(
            f"PUBLISHED_UNCERTIFIED {code}: {message}; "
            f"evidence_root={absolute_evidence_root}"
        )
        self.evidence_root = absolute_evidence_root
        self.code = code
        self.message = message


class EvidenceDurabilityUncertain(RuntimeError):
    pass


class DurableEvidenceWriteError(RuntimeError):
    def __init__(self, name: str, original: BaseException) -> None:
        super().__init__(f"durable evidence write failed for {name}: {original}")
        self.name = name
        self.original = original


class CacheCleanupError(RuntimeError):
    pass


def _evidence_fault(_point: str) -> None:
    return


def _loader_fault(_point: str) -> None:
    return


def _fail(code: str, message: str) -> NoReturn:
    raise DatasetValidationError(code, message)


def _json_exact_equal(actual: object, expected: object) -> bool:
    """Compare decoded JSON values without Python's bool/int aliasing."""

    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return set(actual) == set(expected) and all(
            _json_exact_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _json_exact_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _json_numbers_are_finite(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_json_numbers_are_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_json_numbers_are_finite(item) for item in value)
    return True


def _strict_json_bytes(path: Path, code: str) -> tuple[object, bytes]:
    try:
        payload = converter._read_regular_file_bytes(path, path.name)
        value = json.loads(
            payload,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid constant {value}")
            ),
        )
    except (
        OSError,
        RuntimeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        _fail(code, f"cannot read canonical JSON {path}: {exc}")
    if payload != converter.canonical_json_bytes(value):
        _fail(code, f"noncanonical JSON bytes: {path}")
    return value, payload


def _strict_json_object(path: Path, code: str) -> tuple[dict[str, object], bytes]:
    value, payload = _strict_json_bytes(path, code)
    if not isinstance(value, dict):
        _fail(code, f"expected JSON object: {path}")
    return value, payload


def _strict_jsonl(path: Path, code: str) -> tuple[list[dict[str, object]], bytes]:
    try:
        payload = converter._read_regular_file_bytes(path, path.name)
        rows: list[dict[str, object]] = []
        for line in payload.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                _fail(code, f"unterminated JSONL row: {path}")
            value = json.loads(
                line,
                parse_constant=lambda item: (_ for _ in ()).throw(
                    ValueError(f"invalid constant {item}")
                ),
            )
            if not isinstance(value, dict) or line != converter.canonical_json_bytes(
                value
            ):
                _fail(code, f"noncanonical JSONL row: {path}")
            rows.append(value)
    except DatasetValidationError:
        raise
    except (
        OSError,
        RuntimeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        _fail(code, f"cannot read canonical JSONL {path}: {exc}")
    return rows, payload


def _independent_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if not array.size or not np.isfinite(array).all():
        _fail("STATS_VALUE", "statistics input is empty or nonfinite")
    result = {
        "mean": array.mean(0, dtype=np.float64).astype(np.float32).tolist(),
        "std": array.std(0, dtype=np.float64).astype(np.float32).tolist(),
        "min": array.min(0).astype(np.float32).tolist(),
        "max": array.max(0).astype(np.float32).tolist(),
        "q01": np.quantile(array, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).astype(np.float32).tolist(),
        "count": [int(array.shape[0])],
    }
    converter.canonical_json_bytes(result)
    return result


def _media_mapping(
    identity: converter.MediaIdentity | converter.MediaProfile,
) -> dict[str, object]:
    return json.loads(json.dumps(asdict(identity), allow_nan=False))


def _media_profile_mapping(identity: converter.MediaIdentity) -> dict[str, object]:
    return {
        "codec_name": identity.codec_name,
        "pixel_format": identity.pixel_format,
        "width": identity.width,
        "height": identity.height,
        "average_frame_rate": identity.average_frame_rate,
        "nominal_frame_rate": identity.nominal_frame_rate,
        "audio_streams": list(identity.audio_streams),
    }


def _expected_paths(
    info: dict[str, object], episodes: list[dict[str, object]]
) -> tuple[set[str], set[str], set[str]]:
    try:
        chunks_size = info["chunks_size"]
        data_template = info["data_path"]
        video_template = info["video_path"]
        if type(chunks_size) is not int or chunks_size <= 0:
            raise ValueError("invalid chunks_size")
        if not isinstance(data_template, str) or not isinstance(video_template, str):
            raise ValueError("invalid path template")
        data: set[str] = set()
        videos: set[str] = set()
        for row in episodes:
            index = row["episode_index"]
            if type(index) is not int or index < 0:
                raise ValueError("invalid episode index")
            chunk = index // chunks_size
            data.add(data_template.format(episode_chunk=chunk, episode_index=index))
            videos.add(video_template.format(episode_chunk=chunk, episode_index=index))
    except (KeyError, ValueError, IndexError) as exc:
        _fail("METADATA_PATHS", str(exc))
    if any(
        Path(path).is_absolute() or ".." in Path(path).parts for path in data | videos
    ):
        _fail("METADATA_PATHS", "unsafe generated metadata path")
    return data, videos, set(_FIXED_METADATA) | data | videos


def _published_tree_error_code(root: Path, error: BaseException) -> str:
    message = str(error)
    lowered = message.lower()
    if os.path.lexists(root / "conversion_manifest.json"):
        return "TREE_RESERVED"
    if "link count" in lowered:
        return "TREE_LINK"
    if (
        "unsupported staged tree entry" in lowered
        or "not a regular file" in lowered
        or "is a symlink" in lowered
    ):
        return "TREE_TYPE"
    if "identity differs for path:" in message:
        relative = message.rsplit(":", 1)[-1].strip()
        try:
            manifest = json.loads(
                converter._read_regular_file_bytes(
                    root / "meta/conversion_manifest.json", "payload manifest"
                )
            )
            recorded = next(
                entry
                for entry in manifest["entries"]
                if isinstance(entry, dict) and entry.get("path") == relative
            )
            metadata = (root / relative).lstat()
        except (OSError, RuntimeError, ValueError, KeyError, StopIteration, TypeError):
            return "TREE_MANIFEST"
        return "TREE_SIZE" if recorded.get("size") != metadata.st_size else "TREE_HASH"
    if "membership differs" in lowered:
        return "TREE_MEMBERSHIP"
    if "staging identity differs" in lowered:
        return "TREE_STATUS_IDENTITY"
    return "TREE_MANIFEST"


def _validate_tree(
    root: Path,
    expected_members: set[str],
    *,
    require_final_modes: bool,
) -> tuple[str, str | None, str | None]:
    try:
        metadata = root.lstat()
    except OSError as exc:
        _fail("TREE_ROOT", str(exc))
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_nlink < 2:
        _fail("TREE_ROOT", "dataset root is not a safe directory")

    manifest_path = root / "meta/conversion_manifest.json"
    status_path = root / "CONVERSION_STATUS.json"
    manifest_sha: str | None = None
    status_sha: str | None = None
    if manifest_path.exists():
        try:
            manifest = converter.validate_payload_manifest(root)
            manifest_payload = converter._read_regular_file_bytes(
                manifest_path, "payload manifest"
            )
            manifest_sha = converter.sha256_bytes(manifest_payload)
            status, status_payload = converter._validate_complete_status(root)
            if status.get("manifest_sha256") != manifest_sha:
                raise RuntimeError("status manifest digest differs")
            status_sha = converter.sha256_bytes(status_payload)
            listed = {entry["path"] for entry in manifest["entries"]}
            if listed != expected_members - {"CONVERSION_STATUS.json"}:
                raise RuntimeError("manifest semantic membership differs")
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            _fail(_published_tree_error_code(root, exc), str(exc))
    else:
        if require_final_modes:
            _fail("TREE_MANIFEST", "published tree has no payload manifest")
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        if actual != expected_members:
            difference = actual ^ expected_members
            if difference and all(path.startswith("videos/") for path in difference):
                _fail("VIDEO_MEMBERSHIP", "staged video membership differs")
            _fail("TREE_MEMBERSHIP", "staged tree membership differs")
        status, _ = _strict_json_object(status_path, "TREE_STATUS")
        if status.get("state") != "in_progress":
            _fail("TREE_STATUS", "staged status is not in_progress")

    if require_final_modes:
        try:
            converter._validate_final_modes(root)
        except (OSError, RuntimeError) as exc:
            _fail("TREE_MODE", str(exc))
    try:
        files, _ = converter._canonical_tree_paths(root)
        digest_entries = [
            {
                "path": relative,
                "size": path.lstat().st_size,
                "sha256": converter.sha256_file(path),
            }
            for relative, path in files
        ]
    except (OSError, RuntimeError) as exc:
        _fail("TREE_MANIFEST", str(exc))
    return (
        converter.sha256_bytes(converter.canonical_json_bytes(digest_entries)),
        manifest_sha,
        status_sha,
    )


def _validate_info(
    info: dict[str, object],
    tasks: list[dict[str, object]],
    episodes: list[dict[str, object]],
    *,
    expected: DatasetExpectations | None,
) -> None:
    if (
        set(info) != _INFO_FIELDS
        or info.get("codebase_version") != "v2.1"
        or info.get("robot_type") != "g1"
    ):
        _fail("METADATA_SHAPE", "dataset info schema differs")
    lengths = [row.get("length") for row in episodes]
    if any(type(length) is not int or length <= 0 for length in lengths):
        _fail("METADATA_TOTALS", "invalid episode lengths")
    totals = {
        "total_episodes": len(episodes),
        "total_tasks": len(tasks),
        "total_videos": len(episodes),
        "total_frames": sum(lengths),
    }
    if any(
        type(info.get(key)) is not int or info.get(key) != value
        for key, value in totals.items()
    ):
        _fail("METADATA_TOTALS", "dataset totals differ")
    chunks_size = info.get("chunks_size")
    if type(chunks_size) is not int or chunks_size <= 0:
        _fail("METADATA_CHUNKS", "invalid chunks_size")
    expected_chunks = math.ceil(len(episodes) / chunks_size)
    if (
        type(info.get("total_chunks")) is not int
        or info.get("total_chunks") != expected_chunks
    ):
        _fail("METADATA_CHUNKS", "total_chunks differs")
    fps = info.get("fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        _fail("METADATA_VIDEO", "invalid dataset FPS")
    if (
        info.get("data_path")
        != "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        or info.get("video_path")
        != "videos/chunk-{episode_chunk:03d}/egocentric/episode_{episode_index:06d}.mp4"
    ):
        _fail("METADATA_PATHS", "path templates differ")
    features = info.get("features")
    if not isinstance(features, dict):
        _fail("METADATA_SHAPE", "features is not an object")
    if set(features) != {"observation.images.egocentric", *_FEATURE_METADATA}:
        _fail("METADATA_SHAPE", "feature key set differs")
    for name, expected_feature in _FEATURE_METADATA.items():
        if not _json_exact_equal(features.get(name), expected_feature):
            _fail("METADATA_SHAPE", f"feature metadata differs: {name}")
    image = features.get("observation.images.egocentric")
    if (
        not isinstance(image, dict)
        or set(image) != {"dtype", "shape", "names", "video_info"}
        or image.get("dtype") != "video"
        or image.get("names") != ["height", "width", "channel"]
        or not isinstance(image.get("video_info"), dict)
    ):
        _fail("METADATA_VIDEO", "video metadata is absent")
    if expected is not None:
        plan = expected.plan
        if chunks_size != plan.chunks_size or info.get("fps") != float(plan.output_fps):
            _fail("METADATA_CHUNKS", "preflight settings differ")
        if len(tasks) != len(plan.tasks) or len(episodes) != len(plan.episodes):
            _fail("METADATA_CARDINALITY", "preflight cardinality differs")
        video_info = image["video_info"]
        expected_profile = plan.output_media
        if not _json_exact_equal(
            video_info,
            {
                "has_audio": bool(expected_profile.audio_streams),
                "video.channels": 3,
                "video.codec": expected_profile.codec_name,
                "video.fps": float(Fraction(expected_profile.average_frame_rate)),
                "video.height": expected_profile.height,
                "video.is_depth_map": False,
                "video.pix_fmt": expected_profile.pixel_format,
                "video.width": expected_profile.width,
            },
        ):
            _fail("METADATA_VIDEO", "video_info differs from preflight")


def _validate_metadata_rows(
    root: Path,
    tasks: list[dict[str, object]],
    episodes: list[dict[str, object]],
    expected: DatasetExpectations | None,
) -> None:
    task_keys = {"task_index", "task", "category", "description"}
    task_texts: list[str] = []
    for index, task in enumerate(tasks):
        if (
            set(task) != task_keys
            or type(task.get("task_index")) is not int
            or task.get("task_index") != index
            or not isinstance(task.get("task"), str)
            or not task["task"]
            or task.get("category") != ""
            or task.get("description") != task.get("task")
        ):
            _fail("METADATA_CARDINALITY", "task row differs")
        task_texts.append(task["task"])
    if len(set(task_texts)) != len(task_texts):
        _fail("METADATA_CARDINALITY", "task texts are not unique")
    episode_keys = {
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
    offset = 0
    referenced_task_indices: set[int] = set()
    for index, episode in enumerate(episodes):
        task_references = episode.get("tasks")
        length = episode.get("length")
        if (
            set(episode) != episode_keys
            or type(episode.get("episode_index")) is not int
            or episode.get("episode_index") != index
            or not isinstance(task_references, list)
            or len(task_references) != 1
            or type(task_references[0]) is not int
            or not 0 <= task_references[0] < len(tasks)
            or type(length) is not int
            or length <= 0
            or type(episode.get("dataset_from_index")) is not int
            or episode.get("dataset_from_index") != offset
            or type(episode.get("dataset_to_index")) is not int
            or episode.get("dataset_to_index") != offset + length - 1
            or episode.get("robot_type") != "g1"
            or not _json_exact_equal(
                episode.get("instruction"), tasks[task_references[0]]
            )
            or not isinstance(episode.get("environment_config"), str)
            or not isinstance(episode.get("conversion_provenance"), dict)
        ):
            _fail("METADATA_CARDINALITY", "episode row differs")
        try:
            environment = json.loads(
                episode["environment_config"],
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"invalid constant {value}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            _fail("METADATA_CARDINALITY", f"environment_config is invalid: {exc}")
        if not _json_numbers_are_finite(environment):
            _fail("METADATA_CARDINALITY", "environment_config is nonfinite")
        referenced_task_indices.add(task_references[0])
        offset += length
    if referenced_task_indices != set(range(len(tasks))):
        _fail("METADATA_CARDINALITY", "task references are not exact")
    relative_stats, _ = _strict_json_object(
        root / "meta/relative_stats.json", "METADATA_SHAPE"
    )
    language_map, _ = _strict_json_object(root / "meta/lang_map.json", "METADATA_SHAPE")
    modality, _ = _strict_json_object(root / "meta/modality.json", "METADATA_SHAPE")
    if (
        relative_stats
        or language_map
        or not _json_exact_equal(modality, converter.modality_dict())
    ):
        _fail("METADATA_SHAPE", "fixed metadata differs")
    if expected is not None:
        plan = expected.plan
        expected_tasks = [
            {
                "task_index": task["task_index"],
                "task": task["task"],
                "category": "",
                "description": task["task"],
            }
            for task in plan.tasks
        ]
        if not _json_exact_equal(tasks, expected_tasks):
            _fail("METADATA_CARDINALITY", "tasks differ from preflight")
        for row, planned in zip(episodes, plan.episodes, strict=True):
            if (
                row["episode_index"] != planned.output_episode_index
                or row["tasks"] != [planned.output_task_index]
                or row["length"] != len(planned.retained_indices)
                or row["environment_config"] != planned.environment_config
            ):
                _fail("METADATA_CARDINALITY", "episode differs from preflight")


def _validate_table(
    table: pa.Table,
    episode: dict[str, object],
    *,
    global_offset: int,
    output_fps: Fraction,
) -> dict[str, np.ndarray]:
    if table.schema != _OUTPUT_SCHEMA:
        _fail("ARROW_SCHEMA", "Parquet schema differs")
    length = episode.get("length")
    if type(length) is not int or length <= 0 or table.num_rows != length:
        _fail("ARROW_ROWS", "Parquet row count differs")
    if any(len(table[name]) != length for name in table.column_names):
        _fail("ARROW_ROWS", "column row count differs")
    values = {name: np.asarray(table[name].to_pylist()) for name in table.column_names}
    for name in _FLOAT_COLUMNS:
        try:
            finite = np.isfinite(values[name]).all()
        except TypeError:
            finite = False
        if not finite:
            _fail("NUMERIC_NONFINITE", f"nonfinite value: {name}")
    episode_index = episode.get("episode_index")
    tasks = episode.get("tasks")
    if values["frame_index"].tolist() != list(range(length)):
        _fail("INDEX_LOCAL", "frame_index differs")
    if values["index"].tolist() != list(range(global_offset, global_offset + length)):
        _fail("INDEX_GLOBAL", "global index differs")
    if values["episode_index"].tolist() != [episode_index] * length:
        _fail("INDEX_EPISODE", "episode_index differs")
    if (
        not isinstance(tasks, list)
        or len(tasks) != 1
        or values["task_index"].tolist() != tasks * length
    ):
        _fail("INDEX_TASK", "task_index differs")
    expected_timestamp = np.asarray(
        [Fraction(index, 1) / output_fps for index in range(length)],
        dtype=np.float32,
    )
    if not np.array_equal(values["timestamp"], expected_timestamp):
        _fail("INDEX_TIMESTAMP", "timestamp differs")
    if values["next.done"].tolist() != [False] * (length - 1) + [True]:
        _fail("INDEX_DONE", "terminal flags differ")
    return values


def _validate_stats(
    root: Path,
    episode_rows: list[dict[str, object]],
    all_values: dict[str, list[np.ndarray]],
) -> None:
    rows, _ = _strict_jsonl(root / "meta/episodes_stats.jsonl", "STATS_SCOPE")
    if len(rows) != len(episode_rows):
        _fail("STATS_SCOPE", "episode statistics cardinality differs")
    for row, episode, action, timestamp in zip(
        rows,
        episode_rows,
        all_values["action"],
        all_values["timestamp"],
        strict=True,
    ):
        if type(row.get("episode_index")) is not int or row.get(
            "episode_index"
        ) != episode.get("episode_index"):
            _fail("STATS_SCOPE", "episode statistics scope differs")
        stats = row.get("stats")
        if not isinstance(stats, dict) or set(stats) != {"action", "timestamp"}:
            _fail("STATS_SCOPE", "episode statistics fields differ")
        for name, values in (("action", action), ("timestamp", timestamp)):
            actual = stats[name]
            expected = _independent_stats(values)
            if not isinstance(actual, dict) or not _json_exact_equal(
                actual.get("count"), expected["count"]
            ):
                _fail("STATS_COUNT", f"episode count differs: {name}")
            if not _json_exact_equal(actual, expected):
                _fail("STATS_VALUE", f"episode statistics differ: {name}")
    stats, stats_bytes = _strict_json_object(root / "meta/stats.json", "STATS_VALUE")
    _, psi0_bytes = _strict_json_object(root / "meta/stats_psi0.json", "STATS_BYTES")
    if stats_bytes != psi0_bytes:
        _fail("STATS_BYTES", "global statistics files differ")
    if set(stats) != set(_GLOBAL_STATS_COLUMNS):
        _fail("STATS_SCOPE", "global statistics fields differ")
    for name in _GLOBAL_STATS_COLUMNS:
        expected = _independent_stats(np.concatenate(all_values[name], axis=0))
        actual = stats[name]
        if not isinstance(actual, dict) or not _json_exact_equal(
            actual.get("count"), expected["count"]
        ):
            _fail("STATS_COUNT", f"global count differs: {name}")
        if not _json_exact_equal(actual, expected):
            _fail("STATS_VALUE", f"global statistics differ: {name}")


def _validate_converter_identity(
    identity: dict[str, object], expected: DatasetExpectations | None
) -> None:
    if set(identity) != {"commit", "script_sha256"}:
        _fail("PROVENANCE_CONVERTER", "converter identity fields differ")
    commit = identity.get("commit")
    digest = identity.get("script_sha256")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(char not in _HEX_40 for char in commit)
    ):
        _fail("PROVENANCE_CONVERTER", "invalid converter commit")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in _HEX_40 for char in digest)
    ):
        _fail("PROVENANCE_CONVERTER", "invalid converter digest")
    if expected is not None and not _json_exact_equal(
        identity, asdict(expected.plan.converter)
    ):
        _fail("PROVENANCE_CONVERTER", "converter differs from preflight")
    if expected is not None:
        return
    repository = Path(__file__).resolve().parents[1]
    try:
        payload = subprocess.run(
            ["git", "show", f"{commit}:scripts/postprocess_psi0.py"],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        _fail("PROVENANCE_CONVERTER", f"converter commit unavailable: {exc}")
    if converter.sha256_bytes(payload) != digest:
        _fail("PROVENANCE_CONVERTER", "converter blob digest differs")


def _find_source(roots: list[Path], pattern: str, digest: object, code: str) -> Path:
    candidates = [path for root in roots for path in root.glob(pattern)]
    matches = []
    for path in candidates:
        try:
            payload = converter._read_regular_file_bytes(path, "recorded source")
        except (OSError, RuntimeError):
            continue
        if converter.sha256_bytes(payload) == digest:
            matches.append(path)
    if len(matches) != 1:
        _fail(code, "recorded source cannot be uniquely reopened")
    return matches[0]


def _source_derived_vectors(
    parquet: Path, *, skip: int, downsample: int
) -> dict[str, np.ndarray]:
    """Reconstruct output vectors without trusting generator vector helpers."""

    try:
        table = pq.read_table(parquet)
        qpos = np.asarray(table["observation.joint_qpos"].to_pylist(), dtype=np.float32)
        command = np.asarray(
            table["observation.amo_policy_command"].to_pylist(), dtype=np.float32
        )
        target_yaw = np.asarray(
            table["observation.amo_policy_target_yaw"].to_pylist(), dtype=np.float32
        )
        turning = np.asarray(
            table["observation.amo_policy_turning_flag"].to_pylist(), dtype=np.float32
        )
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    except (OSError, KeyError, pa.ArrowException, ValueError) as exc:
        _fail("PROVENANCE_SOURCE", f"cannot reconstruct source vectors: {exc}")
    if not (
        len(qpos) == len(command) == len(target_yaw) == len(turning) == len(action)
    ):
        _fail("PROVENANCE_SOURCE", "source column cardinality differs")
    initial = np.asarray([0, 0, 0, 0, 0, 0, 0.74, 0.74, 0.74], dtype=np.float32)
    history = np.concatenate([initial[None], command[:-1]], axis=0)
    indices = np.arange(skip, len(qpos), downsample, dtype=np.int64)
    states = np.concatenate(
        [
            qpos[:, 29:32],
            qpos[:, 34:36],
            qpos[:, 32:34],
            qpos[:, 36:43],
            qpos[:, 15:22],
            qpos[:, 22:29],
            history[:, 3:6][:, ::-1],
            history[:, 6:7],
        ],
        axis=1,
    ).astype(np.float32)
    actions = np.concatenate(
        [
            action[:, 29:32],
            action[:, 34:36],
            action[:, 32:34],
            action[:, 36:43],
            action[:, 15:22],
            action[:, 22:29],
            action[:, 13:15],
            action[:, 12:13],
            command[:, 6:7],
            command[:, 0:2],
            turning[:, None],
            target_yaw[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    hand = np.concatenate(
        [
            qpos[:, 29:32],
            qpos[:, 32:34],
            qpos[:, 34:36],
            qpos[:, 36:39],
            qpos[:, 39:41],
            qpos[:, 41:43],
        ],
        axis=1,
    ).astype(np.float32)
    return {
        "states": states[indices],
        "action": actions[indices],
        "observation.hand_joints": hand[indices],
        "observation.arm_joints": np.concatenate(
            [qpos[:, 15:22], qpos[:, 22:29]], axis=1
        ).astype(np.float32)[indices],
        "observation.leg_joints": np.concatenate(
            [qpos[:, :12], qpos[:, 12:15]], axis=1
        ).astype(np.float32)[indices],
        "observation.prev_torso_rpy": history[:, 3:6][:, ::-1].astype(np.float32)[
            indices
        ],
        "observation.prev_height": history[:, 6:7].astype(np.float32)[indices],
    }


def _validate_provenance(
    root: Path,
    info: dict[str, object],
    provenance: dict[str, object],
    episodes: list[dict[str, object]],
    output_videos: list[Path],
    output_values: list[dict[str, np.ndarray]],
    *,
    expected: DatasetExpectations | None,
) -> tuple[dict[str, object], ...]:
    if set(provenance) != {
        "converter",
        "invocation",
        "input_roots",
        "media_mode",
        "output_media",
        "episode_provenance_sha256",
        "output_schema_sha256",
    }:
        _fail("PROVENANCE_SOURCE", "dataset provenance fields differ")
    converter_identity = provenance.get("converter")
    if not isinstance(converter_identity, dict):
        _fail("PROVENANCE_CONVERTER", "converter identity absent")
    _validate_converter_identity(converter_identity, expected)
    invocation = provenance.get("invocation")
    roots_data = provenance.get("input_roots")
    if not isinstance(invocation, dict) or not isinstance(roots_data, list):
        _fail("PROVENANCE_SOURCE", "source invocation or roots absent")
    if set(invocation) != {
        "output_path",
        "skip",
        "downsample",
        "output_fps",
        "video_key",
        "chunks_size",
        "total_episodes",
    }:
        _fail("PROVENANCE_SOURCE", "invocation fields differ")
    skip_value = invocation.get("skip")
    downsample_value = invocation.get("downsample")
    chunks_value = invocation.get("chunks_size")
    episode_count_value = invocation.get("total_episodes")
    if (
        type(skip_value) is not int
        or skip_value < 0
        or type(downsample_value) is not int
        or downsample_value <= 0
        or type(chunks_value) is not int
        or chunks_value <= 0
        or type(episode_count_value) is not int
        or episode_count_value <= 0
        or not isinstance(invocation.get("output_path"), str)
        or not isinstance(invocation.get("output_fps"), str)
        or not isinstance(invocation.get("video_key"), str)
        or not invocation["video_key"]
    ):
        _fail("PROVENANCE_SOURCE", "source invocation types are invalid")
    if expected is not None:
        plan = expected.plan
        expected_invocation = {
            "output_path": str(plan.output_path),
            "skip": plan.skip,
            "downsample": plan.downsample,
            "output_fps": str(plan.output_fps),
            "video_key": plan.video_key,
            "chunks_size": plan.chunks_size,
            "total_episodes": len(plan.episodes),
        }
        if not _json_exact_equal(invocation, expected_invocation):
            _fail("PROVENANCE_SOURCE", "invocation differs from preflight")
        if provenance.get("media_mode") != plan.media_mode or not _json_exact_equal(
            provenance.get("output_media"), _media_mapping(plan.output_media)
        ):
            _fail("PROVENANCE_OUTPUT", "media provenance differs from preflight")
    try:
        output_fps = Fraction(str(invocation["output_fps"]))
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        _fail("PROVENANCE_SOURCE", f"invalid output FPS provenance: {exc}")
    if (
        (expected is None and invocation.get("output_path") != str(root.absolute()))
        or invocation.get("total_episodes") != len(episodes)
        or invocation.get("chunks_size") != info["chunks_size"]
        or float(output_fps) != info["fps"]
    ):
        _fail("PROVENANCE_SOURCE", "invocation differs from canonical dataset")
    roots: list[Path] = []
    for item in roots_data:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            _fail("PROVENANCE_SOURCE", "invalid input root identity")
        path = Path(item["path"])
        try:
            metadata = path.lstat()
        except OSError as exc:
            _fail("PROVENANCE_SOURCE", str(exc))
        actual = {
            "path": str(path),
            "st_dev": metadata.st_dev,
            "st_ino": metadata.st_ino,
            "st_uid": metadata.st_uid,
            "st_gid": metadata.st_gid,
            "st_mode": stat.S_IMODE(metadata.st_mode),
        }
        if not _json_exact_equal(item, actual) or not stat.S_ISDIR(metadata.st_mode):
            _fail("PROVENANCE_SOURCE", "input root identity differs")
        roots.append(path)
    if roots != sorted(roots, key=lambda path: str(path).encode()) or len(
        set(roots)
    ) != len(roots):
        _fail("PROVENANCE_SOURCE", "input roots are not unique and ordered")
    requested_key = invocation.get("video_key")
    if not isinstance(requested_key, str):
        _fail("PROVENANCE_SOURCE", "video key absent")
    digest_order: list[str] = []
    source_entries: list[dict[str, object]] = []
    source_media_identities: list[converter.MediaIdentity] = []
    output_media_identities: list[converter.MediaIdentity] = []
    video_digest_pairs: list[tuple[object, object]] = []
    for row, output_video, emitted in zip(
        episodes, output_videos, output_values, strict=True
    ):
        item = row.get("conversion_provenance")
        if not isinstance(item, dict) or set(item) != _EPISODE_PROVENANCE_FIELDS:
            _fail("PROVENANCE_SOURCE", "episode provenance absent")
        if item.get("requested_video_key") != requested_key or item.get(
            "requested_output_fps"
        ) != invocation.get("output_fps"):
            _fail("PROVENANCE_SOURCE", "episode invocation provenance differs")
        if type(item.get("retained_count")) is not int or item.get(
            "retained_count"
        ) != row.get("length"):
            _fail("PROVENANCE_SOURCE", "episode retained count differs")
        if item.get("converter_commit") != converter_identity.get("commit") or item.get(
            "converter_script_sha256"
        ) != converter_identity.get("script_sha256"):
            _fail("PROVENANCE_CONVERTER", "episode converter identity differs")
        source_episode = item.get("source_episode_index")
        if type(source_episode) is not int or source_episode < 0:
            _fail("PROVENANCE_SOURCE", "source episode index invalid")
        parquet = _find_source(
            roots,
            f"data/chunk-*/episode_{source_episode:06d}.parquet",
            item.get("source_parquet_sha256"),
            "PROVENANCE_SOURCE",
        )
        video = _find_source(
            roots,
            f"videos/chunk-*/{requested_key}/episode_{source_episode:06d}.mp4",
            item.get("source_video_sha256"),
            "PROVENANCE_SOURCE",
        )
        try:
            source_media = converter.probe_media(video)
        except (OSError, RuntimeError, ValueError) as exc:
            _fail("PROVENANCE_SOURCE", str(exc))
        if not _json_exact_equal(
            item.get("source_media"), _media_mapping(source_media)
        ):
            _fail("PROVENANCE_SOURCE", "source media identity differs")
        source_media_identities.append(source_media)
        skip = item.get("skip")
        downsample = item.get("downsample")
        if (
            type(skip) is not int
            or skip < 0
            or type(downsample) is not int
            or downsample <= 0
            or skip != invocation.get("skip")
            or downsample != invocation.get("downsample")
        ):
            _fail("PROVENANCE_SOURCE", "selection provenance is invalid")
        derived = _source_derived_vectors(parquet, skip=skip, downsample=downsample)
        if item.get("retained_count") != len(derived["states"]):
            _fail("PROVENANCE_SOURCE", "retained source count differs")
        for name, values in derived.items():
            if not np.array_equal(values, emitted[name]):
                _fail(
                    "PROVENANCE_OUTPUT",
                    f"emitted values differ from source: {name}",
                )
        if item.get("output_video_sha256") != converter.sha256_file(output_video):
            _fail("PROVENANCE_OUTPUT", "output video digest differs")
        video_digest_pairs.append(
            (item.get("source_video_sha256"), item.get("output_video_sha256"))
        )
        try:
            output_media = converter.probe_media(output_video)
        except (OSError, RuntimeError, ValueError) as exc:
            _fail("VIDEO_PROFILE", str(exc))
        if not _json_exact_equal(
            item.get("output_media"), _media_mapping(output_media)
        ):
            _fail("PROVENANCE_OUTPUT", "output media identity differs")
        output_media_identities.append(output_media)
        if not _json_exact_equal(
            provenance.get("output_media"), _media_profile_mapping(output_media)
        ):
            _fail("PROVENANCE_OUTPUT", "dataset output media differs")
        digest_order.append(
            converter.sha256_bytes(converter.canonical_json_bytes(item))
        )
        source_entries.append(
            {
                "source_parquet": str(parquet),
                "source_parquet_sha256": item["source_parquet_sha256"],
                "source_video": str(video),
                "source_video_sha256": item["source_video_sha256"],
            }
        )
    if provenance.get("episode_provenance_sha256") != digest_order:
        _fail("PROVENANCE_ORDER", "episode provenance digest order differs")
    skip = invocation.get("skip")
    downsample = invocation.get("downsample")
    copy_eligible = (
        skip == 0
        and downsample == 1
        and bool(source_media_identities)
        and all(
            _media_profile_mapping(media)
            == _media_profile_mapping(source_media_identities[0])
            for media in source_media_identities
        )
        and all(
            Fraction(media.average_frame_rate) == output_fps
            and Fraction(media.nominal_frame_rate) == output_fps
            for media in source_media_identities
        )
        and all(
            media.frame_count == len(emitted["states"])
            for media, emitted in zip(
                source_media_identities, output_values, strict=True
            )
        )
    )
    expected_mode = "copy_all" if copy_eligible else "transcode_all"
    if provenance.get("media_mode") != expected_mode:
        _fail("PROVENANCE_OUTPUT", "dataset media mode differs")
    if expected_mode == "copy_all":
        if any(source != output for source, output in video_digest_pairs):
            _fail("PROVENANCE_OUTPUT", "copied media bytes differ")
        if any(
            source != output
            for source, output in zip(
                source_media_identities, output_media_identities, strict=True
            )
        ):
            _fail("PROVENANCE_OUTPUT", "copied media identity differs")
    else:
        canonical = {
            "codec_name": "h264",
            "pixel_format": "yuv420p",
            "width": 640,
            "height": 360,
            "average_frame_rate": str(output_fps),
            "nominal_frame_rate": str(output_fps),
            "audio_streams": [],
        }
        if provenance.get("output_media") != canonical:
            _fail("PROVENANCE_OUTPUT", "canonical output profile differs")
    schema_identity = {
        "fields": [
            {"name": field.name, "nullable": field.nullable, "type": str(field.type)}
            for field in _OUTPUT_SCHEMA
        ]
    }
    if provenance.get("output_schema_sha256") != converter.sha256_bytes(
        converter.canonical_json_bytes(schema_identity)
    ):
        _fail("PROVENANCE_OUTPUT", "output schema digest differs")
    return tuple(source_entries)


def validate_dataset(
    dataset_root: Path,
    *,
    expected: DatasetExpectations | None,
    require_final_modes: bool,
) -> DatasetValidationResult:
    """Independently validate an emitted dataset and all recorded sources."""

    root = dataset_root.absolute()
    if (root / "meta/conversion_manifest.json").exists():
        try:
            converter.validate_payload_manifest(root)
            converter._validate_complete_status(root)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            _fail(_published_tree_error_code(root, exc), str(exc))
        if require_final_modes:
            try:
                converter._validate_final_modes(root)
            except (OSError, RuntimeError) as exc:
                _fail("TREE_MODE", str(exc))
    info, _ = _strict_json_object(root / "meta/info.json", "METADATA_TOTALS")
    tasks, _ = _strict_jsonl(root / "meta/tasks.jsonl", "METADATA_CARDINALITY")
    episodes, _ = _strict_jsonl(root / "meta/episodes.jsonl", "METADATA_CARDINALITY")
    provenance, _ = _strict_json_object(
        root / "meta/conversion_provenance.json", "PROVENANCE_SOURCE"
    )
    _validate_info(info, tasks, episodes, expected=expected)
    _validate_metadata_rows(root, tasks, episodes, expected)
    data_paths, video_paths, expected_members = _expected_paths(info, episodes)
    tree_digest, manifest_sha, status_sha = _validate_tree(
        root,
        expected_members,
        require_final_modes=require_final_modes,
    )

    if [row.get("task_index") for row in tasks] != list(range(len(tasks))):
        _fail("METADATA_CARDINALITY", "task indices are not contiguous")
    if [row.get("episode_index") for row in episodes] != list(range(len(episodes))):
        _fail("METADATA_CARDINALITY", "episode indices are not contiguous")
    output_fps_value = info.get("fps")
    try:
        output_fps = Fraction(str(output_fps_value))
    except (ValueError, ZeroDivisionError) as exc:
        _fail("METADATA_VIDEO", str(exc))
    if output_fps <= 0:
        _fail("METADATA_VIDEO", "output FPS is nonpositive")

    global_offset = 0
    all_values: dict[str, list[np.ndarray]] = {
        name: [] for name in _GLOBAL_STATS_COLUMNS
    }
    ordered_videos: list[Path] = []
    episode_values: list[dict[str, np.ndarray]] = []
    for episode, data_relative, video_relative in zip(
        episodes,
        sorted(data_paths),
        sorted(video_paths),
        strict=True,
    ):
        if episode.get("dataset_from_index") != global_offset:
            _fail("INDEX_GLOBAL", "episode start differs")
        try:
            table = pq.read_table(root / data_relative)
        except (OSError, pa.ArrowException) as exc:
            _fail("ARROW_SCHEMA", str(exc))
        values = _validate_table(
            table,
            episode,
            global_offset=global_offset,
            output_fps=output_fps,
        )
        length = table.num_rows
        if episode.get("dataset_to_index") != global_offset + length - 1:
            _fail("INDEX_GLOBAL", "episode end differs")
        for name in _GLOBAL_STATS_COLUMNS:
            all_values[name].append(values[name])
        video = root / video_relative
        try:
            identity = converter.probe_media(video)
        except (OSError, RuntimeError, ValueError) as exc:
            _fail("VIDEO_PROFILE", str(exc))
        if identity.frame_count != length:
            _fail("VIDEO_PROFILE", "video frame count differs")
        image = info["features"]["observation.images.egocentric"]
        profile = image["video_info"]
        actual_profile = {
            "has_audio": bool(identity.audio_streams),
            "video.channels": 3,
            "video.codec": identity.codec_name,
            "video.fps": float(Fraction(identity.average_frame_rate)),
            "video.height": identity.height,
            "video.is_depth_map": False,
            "video.pix_fmt": identity.pixel_format,
            "video.width": identity.width,
        }
        if not _json_exact_equal(profile, actual_profile) or not _json_exact_equal(
            image.get("shape"),
            [identity.height, identity.width, 3],
        ):
            _fail("VIDEO_PROFILE", "video differs from dataset-wide profile")
        ordered_videos.append(video)
        episode_values.append(values)
        global_offset += length
    _validate_stats(root, episodes, all_values)
    source_entries = _validate_provenance(
        root,
        info,
        provenance,
        episodes,
        ordered_videos,
        episode_values,
        expected=expected,
    )
    return DatasetValidationResult(
        dataset_root=root,
        total_episodes=len(episodes),
        total_frames=global_offset,
        total_tasks=len(tasks),
        total_videos=len(ordered_videos),
        tree_digest=tree_digest,
        manifest_sha256=manifest_sha,
        complete_status_sha256=status_sha,
        source_entries=source_entries,
    )


def _write_evidence_file(root_fd: int, name: str, value: object) -> dict[str, object]:
    if Path(name).name != name or name not in {
        *_EVIDENCE_FILES,
        "PASS.json",
        "FAIL.json",
    }:
        raise ValueError("unsafe evidence filename")
    payload = converter.canonical_json_bytes(value)
    temporary = f".{name}.tmp-{uuid.uuid4()}"
    fd = None
    destination_fd = None
    temporary_exists = False
    published = False
    durable = False
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=root_fd,
        )
        temporary_exists = True
        _evidence_fault(f"{name}:after_temp_create")
        remaining = memoryview(payload)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("evidence write made no progress")
            remaining = remaining[written:]
        _evidence_fault(f"{name}:after_write")
        os.fsync(fd)
        _evidence_fault(f"{name}:after_payload_fsync")
        os.fchmod(fd, 0o444)
        os.fsync(fd)
        _evidence_fault(f"{name}:after_mode_fsync")
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError("unsafe evidence file")
        converter._rename_noreplace_at(
            root_fd,
            os.fsencode(temporary),
            root_fd,
            os.fsencode(name),
            expected_identity=(metadata.st_dev, metadata.st_ino),
        )
        temporary_exists = False
        published = True
        _evidence_fault(f"{name}:after_rename")
        os.fsync(root_fd)
        durable = True
        _evidence_fault(f"{name}:after_root_fsync")
        destination_fd = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
        visible = os.fstat(destination_fd)
        if (
            not stat.S_ISREG(visible.st_mode)
            or visible.st_nlink != 1
            or stat.S_IMODE(visible.st_mode) != 0o444
            or visible.st_size != len(payload)
            or converter._sha256_descriptor(destination_fd)
            != converter.sha256_bytes(payload)
        ):
            raise RuntimeError("durable evidence file differs")
    except BaseException as exc:
        if published and not durable:
            try:
                os.unlink(name, dir_fd=root_fd)
                os.fsync(root_fd)
            except BaseException:
                pass
            raise EvidenceDurabilityUncertain(
                f"evidence durability is uncertain for {name}: {exc}"
            ) from exc
        if durable:
            raise DurableEvidenceWriteError(name, exc) from exc
        raise
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        if fd is not None:
            os.close(fd)
        if temporary_exists:
            try:
                os.unlink(temporary, dir_fd=root_fd)
                os.fsync(root_fd)
            except FileNotFoundError:
                pass
    return {
        "path": name,
        "size": len(payload),
        "sha256": converter.sha256_bytes(payload),
    }


def _completed_evidence_entries(root_fd: int) -> list[dict[str, object]]:
    names = sorted(os.listdir(root_fd), key=lambda value: value.encode())
    if any(
        name.startswith(".") or name in {"PASS.json", "FAIL.json"} for name in names
    ):
        raise EvidenceDurabilityUncertain("evidence root contains an incomplete member")
    if set(names) != set(_EVIDENCE_FILES[: len(names)]):
        raise EvidenceDurabilityUncertain(
            "evidence completion is not a canonical prefix"
        )
    entries: list[dict[str, object]] = []
    for name in _EVIDENCE_FILES[: len(names)]:
        fd = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
        try:
            metadata = os.fstat(fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o444
            ):
                raise EvidenceDurabilityUncertain("completed evidence member is unsafe")
            entries.append(
                {
                    "path": name,
                    "size": metadata.st_size,
                    "sha256": converter._sha256_descriptor(fd),
                }
            )
        finally:
            os.close(fd)
    return entries


def _hash_dataset_member(root_fd: int, components: tuple[str, ...]) -> str:
    directory_fd = os.dup(root_fd)
    member_fd = None
    try:
        for component in components[:-1]:
            child_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            metadata = os.fstat(child_fd)
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError("dataset metadata parent is not a directory")
            os.close(directory_fd)
            directory_fd = child_fd
        member_fd = os.open(
            components[-1],
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        before = os.fstat(member_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError("dataset identity member is unsafe")
        digest = converter._sha256_descriptor(member_fd)
        after = os.fstat(member_fd)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise RuntimeError("dataset identity member changed while hashing")
        return digest
    finally:
        if member_fd is not None:
            os.close(member_fd)
        os.close(directory_fd)


def _dataset_root_identity(
    publication: converter.PublicationResult,
) -> dict[str, object]:
    fd = os.open(
        publication.dataset_root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        before = os.fstat(fd)
        if not stat.S_ISDIR(before.st_mode):
            raise DatasetValidationError(
                "PUBLICATION_IDENTITY", "dataset root is not a directory"
            )
        converter._validate_descriptor_path(
            fd, publication.dataset_root, expected_type="directory"
        )
        manifest_sha256 = _hash_dataset_member(fd, ("meta", "conversion_manifest.json"))
        complete_status_sha256 = _hash_dataset_member(fd, ("CONVERSION_STATUS.json",))
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise DatasetValidationError(
                "PUBLICATION_IDENTITY", "dataset root changed while hashing"
            )
        if (
            manifest_sha256 != publication.manifest_sha256
            or complete_status_sha256 != publication.complete_status_sha256
        ):
            raise DatasetValidationError(
                "PUBLICATION_IDENTITY", "publication result differs from disk"
            )
        return {
            "path": str(publication.dataset_root.absolute()),
            "st_dev": after.st_dev,
            "st_ino": after.st_ino,
            "st_uid": after.st_uid,
            "st_gid": after.st_gid,
            "st_mode": after.st_mode,
            "st_nlink": after.st_nlink,
            "manifest_sha256": manifest_sha256,
            "complete_status_sha256": complete_status_sha256,
        }
    except DatasetValidationError:
        raise
    except (OSError, RuntimeError) as exc:
        raise DatasetValidationError("PUBLICATION_IDENTITY", str(exc)) from exc
    finally:
        os.close(fd)


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_40 for character in value)
    ):
        raise DatasetValidationError(
            "PUBLICATION_IDENTITY", f"{label} is not a lowercase SHA-256"
        )
    return value


def _collect_psi0_environment(
    psi0_root: Path, psi0_commit: str, python: Path
) -> dict[str, object]:
    source, executable = _validate_psi0_checkout(psi0_root, psi0_commit, python)
    probe = """
import importlib.metadata
import json
import platform

packages = {}
for name in ("av", "datasets", "numpy", "pyarrow", "torch", "torchvision"):
    packages[name] = importlib.metadata.version(name)
distributions = sorted(
    (
        {
            "name": distribution.metadata.get("Name", ""),
            "version": distribution.version,
        }
        for distribution in importlib.metadata.distributions()
    ),
    key=lambda item: (item["name"], item["version"]),
)
print(
    json.dumps(
        {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
            "distributions": distributions,
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
)
"""
    try:
        completed = subprocess.run(
            [str(executable), "-I", "-c", probe],
            env={"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1"},
            check=True,
            capture_output=True,
            text=False,
        )
        probed = json.loads(completed.stdout)
    except (
        OSError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        raise DatasetValidationError("PSI0_ENVIRONMENT", str(exc)) from exc
    if (
        not isinstance(probed, dict)
        or completed.stdout != converter.canonical_json_bytes(probed)
        or set(probed) != {"python_version", "platform", "packages", "distributions"}
    ):
        raise DatasetValidationError(
            "PSI0_ENVIRONMENT", "Python environment probe is not canonical"
        )
    details = {
        "psi0_root": str(source.parent),
        "psi0_src": str(source),
        "psi0_commit": psi0_commit,
        "tracked_status": [],
        "python": str(executable),
        "python_realpath": str(executable.resolve(strict=True)),
        **probed,
    }
    details["compat_module"] = _compat_module_identity(source, psi0_commit)
    details["distribution_manifest_sha256"] = converter.sha256_bytes(
        converter.canonical_json_bytes(details["distributions"])
    )
    return {
        "schema_version": 1,
        "details": details,
        "details_sha256": converter.sha256_bytes(
            converter.canonical_json_bytes(details)
        ),
    }


def _validate_psi0_checkout(
    psi0_root: Path, psi0_commit: str, python: Path
) -> tuple[Path, Path]:
    if (
        not isinstance(psi0_commit, str)
        or len(psi0_commit) != 40
        or any(character not in _HEX_40 for character in psi0_commit)
    ):
        raise DatasetValidationError(
            "PSI0_CHECKOUT", "PSI0 commit is not a lowercase 40-hex object name"
        )
    try:
        root = psi0_root.resolve(strict=True)
        source = (root / "src").resolve(strict=True)
    except OSError as exc:
        raise DatasetValidationError("PSI0_CHECKOUT", str(exc)) from exc
    if not root.is_dir() or not source.is_dir() or source.parent != root:
        raise DatasetValidationError(
            "PSI0_CHECKOUT", "PSI0 root or its direct src child is not a directory"
        )
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status_output = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DatasetValidationError("PSI0_CHECKOUT", str(exc)) from exc
    if head != psi0_commit:
        raise DatasetValidationError(
            "PSI0_CHECKOUT", f"PSI0 HEAD differs: expected {psi0_commit}, got {head}"
        )
    if status_output:
        raise DatasetValidationError("PSI0_CHECKOUT", "PSI0 tracked state is dirty")
    executable = Path(os.path.abspath(python))
    try:
        metadata = executable.stat()
    except OSError as exc:
        raise DatasetValidationError("PSI0_ENVIRONMENT", str(exc)) from exc
    if not stat.S_ISREG(metadata.st_mode) or not os.access(executable, os.X_OK):
        raise DatasetValidationError(
            "PSI0_ENVIRONMENT", "certification Python is not executable"
        )
    return source, executable


def _compat_module_identity(source: Path, commit: str) -> dict[str, object]:
    repository_root = source.parent
    module_path = source / "psi/data/lerobot/compat.py"
    try:
        resolved_module = module_path.resolve(strict=True)
        if (
            not resolved_module.is_relative_to(source)
            or resolved_module != module_path.absolute()
        ):
            raise RuntimeError("PSI0 compat module escapes the pinned source root")
        metadata = resolved_module.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError("PSI0 compat module is not a single-link regular file")
        disk_digest = converter.sha256_file(resolved_module)
        blob = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "show",
                f"{commit}:{_COMPAT_RELATIVE_PATH}",
            ],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise DatasetValidationError("PSI0_CHECKOUT", str(exc)) from exc
    blob_digest = converter.sha256_bytes(blob)
    if disk_digest != blob_digest:
        raise DatasetValidationError(
            "PSI0_CHECKOUT", "PSI0 compat module differs from its committed blob"
        )
    return {
        "commit": commit,
        "relative_path": _COMPAT_RELATIVE_PATH,
        "origin": str(resolved_module),
        "blob_sha256": blob_digest,
    }


def _validate_checkout_bytecode_caches(source: Path) -> None:
    root_fd: int | None = None

    def source_name_for_cache(name: str) -> str:
        cache_tag = sys.implementation.cache_tag
        if not isinstance(cache_tag, str) or not cache_tag:
            raise DatasetValidationError(
                "PSI0_CHECKOUT", "Python cache tag is unavailable"
            )
        plain_suffix = f".{cache_tag}.pyc"
        optimization: str | None = None
        if name.endswith(plain_suffix):
            stem = name[: -len(plain_suffix)]
        else:
            body = name.removesuffix(".pyc")
            stem, marker, optimization_value = body.rpartition(f".{cache_tag}.opt-")
            if not marker or not optimization_value or not optimization_value.isalnum():
                raise DatasetValidationError(
                    "PSI0_CHECKOUT", "checkout bytecode cache name is noncanonical"
                )
            optimization = optimization_value
        if not stem:
            raise DatasetValidationError(
                "PSI0_CHECKOUT", "checkout bytecode cache has no source name"
            )
        source_name = f"{stem}.py"
        expected_name = Path(
            importlib.util.cache_from_source(
                source_name, optimization=optimization if optimization else ""
            )
        ).name
        if expected_name != name:
            raise DatasetValidationError(
                "PSI0_CHECKOUT", "checkout bytecode cache name is noncanonical"
            )
        return source_name

    def validate_timestamp_pyc(
        directory_fd: int,
        source_directory_fd: int | None,
        name: str,
        path_metadata: os.stat_result,
    ) -> None:
        if (
            source_directory_fd is None
            or not stat.S_ISREG(path_metadata.st_mode)
            or path_metadata.st_nlink != 1
        ):
            raise DatasetValidationError(
                "PSI0_CHECKOUT", "checkout bytecode is not a canonical cache file"
            )
        source_name = source_name_for_cache(name)
        source_metadata = os.stat(
            source_name, dir_fd=source_directory_fd, follow_symlinks=False
        )
        if not stat.S_ISREG(source_metadata.st_mode) or source_metadata.st_nlink != 1:
            raise DatasetValidationError(
                "PSI0_CHECKOUT", "checkout bytecode source is not a regular file"
            )
        source_fd = os.open(
            source_name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=source_directory_fd,
        )
        try:
            stable_source = os.fstat(source_fd)
            if (
                not stat.S_ISREG(stable_source.st_mode)
                or stable_source.st_nlink != 1
                or (stable_source.st_dev, stable_source.st_ino)
                != (source_metadata.st_dev, source_metadata.st_ino)
            ):
                raise DatasetValidationError(
                    "PSI0_CHECKOUT", "checkout bytecode source changed"
                )
        finally:
            os.close(source_fd)
        bytecode_fd = os.open(
            name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        try:
            before = os.fstat(bytecode_fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or (before.st_dev, before.st_ino)
                != (path_metadata.st_dev, path_metadata.st_ino)
            ):
                raise DatasetValidationError(
                    "PSI0_CHECKOUT", "checkout bytecode changed before opening"
                )
            header = os.pread(bytecode_fd, 16, 0)
            after = os.fstat(bytecode_fd)
            if (
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or after.st_nlink != 1
                or len(header) != 16
                or header[:4] != importlib.util.MAGIC_NUMBER
                or int.from_bytes(header[4:8], "little") != 0
                or int.from_bytes(header[8:12], "little")
                != int(stable_source.st_mtime) & 0xFFFFFFFF
                or int.from_bytes(header[12:16], "little")
                != stable_source.st_size & 0xFFFFFFFF
            ):
                raise DatasetValidationError(
                    "PSI0_CHECKOUT",
                    "checkout-local Python bytecode is not a stable timestamp cache",
                )
        finally:
            os.close(bytecode_fd)

    def walk(directory_fd: int, source_directory_fd: int | None = None) -> None:
        for name in sorted(os.listdir(directory_fd), key=os.fsencode):
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if name.endswith(".pyc"):
                validate_timestamp_pyc(
                    directory_fd, source_directory_fd, name, metadata
                )
                continue
            if not stat.S_ISDIR(metadata.st_mode):
                continue
            child_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            try:
                stable = os.fstat(child_fd)
                if (stable.st_dev, stable.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    raise RuntimeError("PSI0 checkout directory changed during scan")
                walk(
                    child_fd,
                    directory_fd if name == "__pycache__" else None,
                )
            finally:
                os.close(child_fd)

    try:
        root_fd = os.open(
            source,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        converter._validate_descriptor_path(root_fd, source, expected_type="directory")
        walk(root_fd)
        converter._validate_descriptor_path(root_fd, source, expected_type="directory")
    except DatasetValidationError:
        raise
    except (OSError, RuntimeError) as exc:
        raise DatasetValidationError("PSI0_CHECKOUT", str(exc)) from exc
    finally:
        if root_fd is not None:
            os.close(root_fd)


def _worker_environment(cache_root: Path) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(cache_root / "home"),
        "HF_HOME": str(cache_root / "hf"),
        "HF_DATASETS_CACHE": str(cache_root / "datasets"),
        "XDG_CACHE_HOME": str(cache_root / "xdg"),
        "TORCH_HOME": str(cache_root / "torch"),
        "TMPDIR": str(cache_root / "tmp"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
    }


def _prepare_worker_pycache_prefix(tmp_root: Path) -> Path:
    try:
        resolved_tmp = tmp_root.resolve(strict=True)
        if resolved_tmp != tmp_root.absolute():
            raise RuntimeError("loader TMPDIR is not an exact resolved path")
        tmp_fd = os.open(
            resolved_tmp,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            metadata = os.fstat(tmp_fd)
            if stat.S_IMODE(metadata.st_mode) != 0o700:
                raise RuntimeError("loader TMPDIR mode differs")
            os.mkdir("pycache", 0o700, dir_fd=tmp_fd)
            prefix_fd = os.open(
                "pycache",
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=tmp_fd,
            )
            try:
                prefix_metadata = os.fstat(prefix_fd)
                if stat.S_IMODE(prefix_metadata.st_mode) != 0o700 or os.listdir(
                    prefix_fd
                ):
                    raise RuntimeError("loader bytecode cache is not private and empty")
                os.fsync(prefix_fd)
            finally:
                os.close(prefix_fd)
            os.fsync(tmp_fd)
        finally:
            os.close(tmp_fd)
    except (OSError, RuntimeError) as exc:
        raise DatasetValidationError("PSI0_LOADER_VALIDATION", str(exc)) from exc
    prefix = resolved_tmp / "pycache"
    sys.pycache_prefix = str(prefix)
    return prefix


def _install_checkout_source_only_importer(source: Path) -> None:
    def source_only_path_hook(path_entry: str) -> importlib.machinery.FileFinder:
        try:
            resolved = Path(path_entry).resolve(strict=True)
        except (OSError, TypeError) as exc:
            raise ImportError from exc
        if resolved != source and not resolved.is_relative_to(source):
            raise ImportError
        return importlib.machinery.FileFinder(
            str(resolved),
            (
                importlib.machinery.SourceFileLoader,
                importlib.machinery.SOURCE_SUFFIXES,
            ),
            (
                importlib.machinery.ExtensionFileLoader,
                importlib.machinery.EXTENSION_SUFFIXES,
            ),
        )

    sys.path_hooks.insert(0, source_only_path_hook)
    for cached_path in tuple(sys.path_importer_cache):
        try:
            resolved = Path(cached_path).resolve(strict=True)
        except (OSError, TypeError):
            continue
        if resolved == source or resolved.is_relative_to(source):
            sys.path_importer_cache.pop(cached_path, None)


def _create_loader_cache(root_fd: int) -> None:
    os.mkdir(".loader-cache", 0o700, dir_fd=root_fd)
    cache_fd = os.open(
        ".loader-cache",
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=root_fd,
    )
    try:
        os.fchmod(cache_fd, 0o700)
        for name in ("home", "hf", "datasets", "xdg", "torch", "tmp"):
            os.mkdir(name, 0o700, dir_fd=cache_fd)
            child_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=cache_fd,
            )
            try:
                os.fchmod(child_fd, 0o700)
                os.fsync(child_fd)
            finally:
                os.close(child_fd)
        os.fsync(cache_fd)
    finally:
        os.close(cache_fd)
    os.fsync(root_fd)


def _loader_cache_manifest(root_fd: int) -> list[dict[str, object]]:
    cache_fd = os.open(
        ".loader-cache",
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=root_fd,
    )
    entries: list[dict[str, object]] = []

    root_metadata = os.fstat(cache_fd)
    if stat.S_IMODE(root_metadata.st_mode) != 0o700:
        os.close(cache_fd)
        raise RuntimeError("loader-cache root mode differs")
    top_level = set(os.listdir(cache_fd))
    for name in _MANDATORY_CACHE_DIRECTORIES:
        if name not in top_level:
            os.close(cache_fd)
            raise RuntimeError(f"mandatory loader-cache directory is absent: {name}")
        metadata = os.stat(name, dir_fd=cache_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            os.close(cache_fd)
            raise RuntimeError(f"mandatory loader-cache directory differs: {name}")

    def walk(directory_fd: int, prefix: str) -> None:
        for name in sorted(os.listdir(directory_fd), key=os.fsencode):
            if not name or name in {".", ".."} or "/" in name:
                raise RuntimeError("unsafe loader-cache member name")
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            relative = f"{prefix}/{name}" if prefix else name
            if stat.S_ISDIR(metadata.st_mode):
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    stable = os.fstat(child_fd)
                    if (stable.st_dev, stable.st_ino) != (
                        metadata.st_dev,
                        metadata.st_ino,
                    ):
                        raise RuntimeError("loader-cache directory changed")
                    entries.append(
                        {
                            "path": relative,
                            "type": "directory",
                            "size": metadata.st_size,
                            "sha256": None,
                        }
                    )
                    walk(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                member_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    before = os.fstat(member_fd)
                    digest = converter._sha256_descriptor(member_fd)
                    after = os.fstat(member_fd)
                    if (before.st_dev, before.st_ino, before.st_size) != (
                        after.st_dev,
                        after.st_ino,
                        after.st_size,
                    ) or before.st_nlink != 1:
                        raise RuntimeError("loader-cache file changed")
                    entries.append(
                        {
                            "path": relative,
                            "type": "regular",
                            "size": after.st_size,
                            "sha256": digest,
                        }
                    )
                finally:
                    os.close(member_fd)
            else:
                raise RuntimeError("unsafe loader-cache member")

    try:
        walk(cache_fd, "")
    finally:
        os.close(cache_fd)
    return entries


def _remove_loader_cache(root_fd: int) -> None:
    cache_fd = os.open(
        ".loader-cache",
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=root_fd,
    )

    def remove_children(directory_fd: int) -> None:
        for name in sorted(os.listdir(directory_fd), key=os.fsencode):
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    stable = os.fstat(child_fd)
                    if (stable.st_dev, stable.st_ino) != (
                        metadata.st_dev,
                        metadata.st_ino,
                    ):
                        raise RuntimeError("loader-cache directory changed")
                    remove_children(child_fd)
                finally:
                    os.close(child_fd)
                os.rmdir(name, dir_fd=directory_fd)
            elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                os.unlink(name, dir_fd=directory_fd)
            else:
                raise RuntimeError("unsafe loader-cache member")
            os.fsync(directory_fd)

    try:
        remove_children(cache_fd)
    finally:
        os.close(cache_fd)
    os.rmdir(".loader-cache", dir_fd=root_fd)
    os.fsync(root_fd)
    try:
        os.stat(".loader-cache", dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise RuntimeError("loader cache remains after cleanup")


def _read_worker_result(
    root_fd: int, name: str, expected_identity: tuple[int, int]
) -> tuple[dict[str, object], bytes]:
    fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=root_fd)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino) != expected_identity
        ):
            raise RuntimeError("loader result identity differs")
        chunks: list[bytes] = []
        while chunk := os.read(fd, 1024 * 1024):
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise RuntimeError("loader result changed while reading")
    finally:
        os.close(fd)
    value = json.loads(
        payload,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"invalid constant {item}")
        ),
    )
    if not isinstance(value, dict) or payload != converter.canonical_json_bytes(value):
        raise RuntimeError("loader result is not canonical JSON")
    return value, payload


def _loader_audit_policy_is_valid(value: object) -> bool:
    if (
        not isinstance(value, dict)
        or set(value) != {"audit_hook", "denied_events", "violations"}
        or value.get("audit_hook") != "deny_inet_resolver_process_escape"
        or value.get("denied_events") != list(_DENIED_LOADER_AUDIT_EVENTS)
        or not isinstance(value.get("violations"), list)
    ):
        return False
    return not any(
        not isinstance(item, dict)
        or set(item) != {"event", "category"}
        or not isinstance(item.get("event"), str)
        or item.get("category") != _LOADER_AUDIT_EVENT_CATEGORIES.get(item.get("event"))
        for item in value["violations"]
    )


def _validate_worker_result(
    value: dict[str, object],
    *,
    result: DatasetValidationResult,
    psi0_source: Path,
    environment: dict[str, str],
    module_identity: dict[str, object],
) -> None:
    base_fields = {
        "schema_version",
        "verdict",
        "sys_path_0",
        "visited_indices",
        "episode_ranges",
        "offline_environment",
        "tensor_contract",
        "module_provenance",
        "network_policy",
    }
    verdict = value.get("verdict")
    expected_fields = base_fields | (
        set() if verdict == "PASS" else {"error_code", "message"}
    )
    if (
        set(value) != expected_fields
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or verdict not in {"PASS", "FAIL"}
        or value.get("sys_path_0") != str(psi0_source)
        or not _json_exact_equal(value.get("offline_environment"), environment)
        or not _json_exact_equal(value.get("module_provenance"), module_identity)
    ):
        raise DatasetValidationError(
            "PSI0_LOADER_VALIDATION", "loader result schema differs"
        )
    visited = value.get("visited_indices")
    if not isinstance(visited, list) or any(
        type(index) is not int for index in visited
    ):
        raise DatasetValidationError(
            "PSI0_LOADER_VALIDATION", "loader visited indices are invalid"
        )
    network_policy = value.get("network_policy")
    if not _loader_audit_policy_is_valid(network_policy):
        raise DatasetValidationError(
            "PSI0_LOADER_VALIDATION", "loader network policy differs"
        )
    if verdict == "FAIL":
        if (
            not isinstance(value.get("error_code"), str)
            or not value["error_code"]
            or not isinstance(value.get("message"), str)
        ):
            raise DatasetValidationError(
                "PSI0_LOADER_VALIDATION", "loader failure identity is invalid"
            )
        return
    if network_policy["violations"]:
        raise DatasetValidationError(
            "PSI0_LOADER_VALIDATION", "loader recorded a network-policy violation"
        )
    expected_indices = list(range(result.total_frames))
    if visited != expected_indices:
        raise DatasetValidationError(
            "PSI0_LOADER_VALIDATION", "loader traversal was incomplete"
        )
    expected_ranges: list[list[int]] = []
    episodes, _ = _strict_jsonl(
        result.dataset_root / "meta/episodes.jsonl", "PSI0_LOADER_VALIDATION"
    )
    for episode in episodes:
        expected_ranges.append(
            [episode["dataset_from_index"], episode["dataset_to_index"]]
        )
    if value.get("episode_ranges") != expected_ranges:
        raise DatasetValidationError(
            "PSI0_LOADER_VALIDATION", "loader episode ranges differ"
        )
    info, _ = _strict_json_object(
        result.dataset_root / "meta/info.json", "PSI0_LOADER_VALIDATION"
    )
    image_shape = info["features"]["observation.images.egocentric"]["shape"]
    expected_contract = {
        "action": {"dtype": "torch.float32", "shape": [36]},
        "observation.images.egocentric": {
            "dtype": "torch.float32",
            "shape": [3, image_shape[0], image_shape[1]],
        },
        "states": {"dtype": "torch.float32", "shape": [32]},
    }
    if not _json_exact_equal(value.get("tensor_contract"), expected_contract):
        raise DatasetValidationError(
            "PSI0_LOADER_VALIDATION", "loader tensor contract differs"
        )


def _run_psi0_loader(
    *,
    root_fd: int,
    evidence_root: Path,
    result: DatasetValidationResult,
    psi0_root: Path,
    psi0_commit: str,
    python: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    psi0_source, executable = _validate_psi0_checkout(psi0_root, psi0_commit, python)
    module_identity = _compat_module_identity(psi0_source, psi0_commit)
    cache_root = evidence_root / ".loader-cache"
    _create_loader_cache(root_fd)
    environment = _worker_environment(cache_root)
    result_name = f".psi0-loader-result-{uuid.uuid4()}.tmp"
    argv = [
        str(executable),
        "-I",
        "scripts/certify_psi0_dataset.py",
        "--loader-worker",
        "--psi0-src",
        str(psi0_source),
        "--dataset-root",
        str(result.dataset_root),
        "--result",
        str(evidence_root / result_name),
    ]
    command = {
        "schema_version": 1,
        "argv": argv,
        "environment": environment,
        "environment_sha256": converter.sha256_bytes(
            converter.canonical_json_bytes(environment)
        ),
    }
    process: subprocess.CompletedProcess[str] | None = None
    worker_value: dict[str, object]
    worker_payload: bytes
    cache_entries: list[dict[str, object]] = []
    result_created = False
    try:
        _loader_fault("after_cache_setup")
        checked_source, checked_executable = _validate_psi0_checkout(
            psi0_root, psi0_commit, python
        )
        checked_module = _compat_module_identity(checked_source, psi0_commit)
        if (
            checked_source != psi0_source
            or checked_executable != executable
            or not _json_exact_equal(checked_module, module_identity)
        ):
            raise DatasetValidationError(
                "PSI0_CHECKOUT", "PSI0 checkout changed during loader setup"
            )
        _validate_checkout_bytecode_caches(checked_source)
        result_fd = os.open(
            result_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=root_fd,
        )
        result_created = True
        try:
            result_metadata = os.fstat(result_fd)
            request_payload = converter.canonical_json_bytes(
                {
                    "schema_version": 1,
                    "expected_commit": psi0_commit,
                    "module_identity": module_identity,
                }
            )
            remaining = memoryview(request_payload)
            while remaining:
                written = os.write(result_fd, remaining)
                if written <= 0:
                    raise OSError("loader request write made no progress")
                remaining = remaining[written:]
            os.fsync(result_fd)
        finally:
            os.close(result_fd)
        result_identity = (result_metadata.st_dev, result_metadata.st_ino)
        process = subprocess.run(
            argv,
            cwd=_REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            worker_value, worker_payload = _read_worker_result(
                root_fd, result_name, result_identity
            )
            _validate_worker_result(
                worker_value,
                result=result,
                psi0_source=psi0_source,
                environment=environment,
                module_identity=module_identity,
            )
        except BaseException as exc:
            worker_value = {
                "schema_version": 1,
                "verdict": "FAIL",
                "sys_path_0": str(psi0_source),
                "visited_indices": [],
                "episode_ranges": [],
                "offline_environment": environment,
                "tensor_contract": {},
                "module_provenance": module_identity,
                "network_policy": {
                    "audit_hook": "deny_inet_resolver_process_escape",
                    "denied_events": list(_DENIED_LOADER_AUDIT_EVENTS),
                    "violations": [],
                },
                "error_code": "WORKER_RESULT_INVALID",
                "message": str(exc),
            }
            worker_payload = converter.canonical_json_bytes(worker_value)
        expected_returncode = 0 if worker_value.get("verdict") == "PASS" else 1
        if process.returncode != expected_returncode:
            execution_error = RuntimeError(
                f"loader worker exited {process.returncode}: {process.stderr}"
            )
            worker_value = {
                **worker_value,
                "verdict": "FAIL",
                "error_code": "WORKER_EXIT",
                "message": str(execution_error),
            }
            worker_payload = converter.canonical_json_bytes(worker_value)
        cache_entries = _loader_cache_manifest(root_fd)
    finally:
        cleanup_errors: list[str] = []
        if result_created:
            try:
                os.unlink(result_name, dir_fd=root_fd)
                os.fsync(root_fd)
            except BaseException as exc:
                cleanup_errors.append(f"result cleanup: {exc}")
        try:
            _remove_loader_cache(root_fd)
        except BaseException as exc:
            cleanup_errors.append(f"cache cleanup: {exc}")
        if cleanup_errors:
            raise CacheCleanupError("; ".join(cleanup_errors))
    wrapped_result = {
        "schema_version": 1,
        "returncode": process.returncode if process is not None else None,
        "stdout": process.stdout if process is not None else "",
        "stderr": process.stderr if process is not None else "",
        "result_sha256": converter.sha256_bytes(worker_payload),
        "result": worker_value,
    }
    return command, {"schema_version": 1, "entries": cache_entries}, wrapped_result


def _write_worker_result(path: Path, value: dict[str, object]) -> None:
    payload = converter.canonical_json_bytes(value)
    fd = os.open(path, os.O_WRONLY | os.O_TRUNC | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError("unsafe loader result destination")
        remaining = memoryview(payload)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("loader result write made no progress")
            remaining = remaining[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _loader_worker_main(
    *, psi0_src: Path, dataset_root: Path, result_path: Path
) -> int:
    visited: list[int] = []
    episode_ranges: list[list[int]] = []
    source_string = str(psi0_src.absolute())
    tensor_contract: dict[str, object] = {}
    module_provenance: dict[str, object] = {}
    network_violations: list[dict[str, object]] = []
    network_policy = {
        "audit_hook": "deny_inet_resolver_process_escape",
        "denied_events": list(_DENIED_LOADER_AUDIT_EVENTS),
        "violations": network_violations,
    }
    offline_keys = (
        "PATH",
        "HOME",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "TMPDIR",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "PYTHONNOUSERSITE",
    )
    offline_environment = {
        key: os.environ[key] for key in offline_keys if key in os.environ
    }
    try:
        worker_request, _ = _strict_json_object(result_path, "PSI0_LOADER_VALIDATION")
        expected_commit = worker_request.get("expected_commit")
        expected_module = worker_request.get("module_identity")
        if (
            set(worker_request)
            != {"schema_version", "expected_commit", "module_identity"}
            or type(worker_request.get("schema_version")) is not int
            or worker_request.get("schema_version") != 1
            or not isinstance(expected_commit, str)
            or not isinstance(expected_module, dict)
        ):
            raise RuntimeError("loader worker request differs")
        source = psi0_src.resolve(strict=True)
        source_string = str(source)
        if not source.is_dir():
            raise RuntimeError("PSI0 source is not a directory")
        checked_source, _ = _validate_psi0_checkout(
            source.parent, expected_commit, Path(sys.executable)
        )
        module_provenance = _compat_module_identity(checked_source, expected_commit)
        if checked_source != source or not _json_exact_equal(
            module_provenance, expected_module
        ):
            raise RuntimeError("PSI0 checkout changed before worker import")
        _validate_checkout_bytecode_caches(checked_source)
        sys.dont_write_bytecode = True
        _prepare_worker_pycache_prefix(Path(os.environ["TMPDIR"]))
        _install_checkout_source_only_importer(checked_source)
        import socket

        def deny_network_and_process_escape(
            event: str, arguments: tuple[object, ...]
        ) -> None:
            category: str | None = None
            if event in _RESOLVER_AUDIT_EVENTS:
                category = "resolver"
            elif event in _PROCESS_ESCAPE_AUDIT_EVENTS:
                category = "process_escape"
            elif event == "socket.__new__" and len(arguments) >= 2:
                try:
                    family = int(arguments[1])
                except (TypeError, ValueError):
                    family = -1
                if family in {socket.AF_INET, socket.AF_INET6}:
                    category = "inet_socket"
            elif event in _INET_SOCKET_AUDIT_EVENTS and arguments:
                family = getattr(arguments[0], "family", None)
                if family in {socket.AF_INET, socket.AF_INET6}:
                    category = "inet_socket"
            if category is None:
                return
            network_violations.append({"event": event, "category": category})
            raise PermissionError(
                "network, resolver, and child-process access is forbidden"
            )

        sys.addaudithook(deny_network_and_process_escape)
        sys.path.insert(0, source_string)
        recorded_sys_path_0 = sys.path[0]
        if recorded_sys_path_0 != source_string:
            raise RuntimeError("PSI0 source is not first on sys.path")
        from psi.data.lerobot.compat import LeRobotDataset

        import torch

        module = sys.modules.get("psi.data.lerobot.compat")
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError("PSI0 compat module has no file origin")
        imported_origin = Path(module_file).resolve(strict=True)
        if (
            imported_origin != Path(module_provenance["origin"])
            or converter.sha256_file(imported_origin)
            != module_provenance["blob_sha256"]
        ):
            raise RuntimeError("imported PSI0 compat module differs from pinned blob")

        info, _ = _strict_json_object(
            dataset_root / "meta/info.json", "PSI0_LOADER_VALIDATION"
        )
        episodes, _ = _strict_jsonl(
            dataset_root / "meta/episodes.jsonl", "PSI0_LOADER_VALIDATION"
        )
        total_frames = info.get("total_frames")
        features = info.get("features")
        if type(total_frames) is not int or total_frames <= 0:
            raise RuntimeError("dataset total_frames is invalid")
        if not isinstance(features, dict):
            raise RuntimeError("dataset features are invalid")
        image_feature = features.get("observation.images.egocentric")
        if not isinstance(image_feature, dict):
            raise RuntimeError("dataset image feature is invalid")
        image_shape = image_feature.get("shape")
        if (
            not isinstance(image_shape, list)
            or len(image_shape) != 3
            or any(type(dimension) is not int for dimension in image_shape)
        ):
            raise RuntimeError("dataset image shape is invalid")
        height, width, channels = image_shape
        if channels != 3:
            raise RuntimeError("dataset image channels differ")
        expected_start = 0
        for episode in episodes:
            start = episode.get("dataset_from_index")
            end = episode.get("dataset_to_index")
            if (
                type(start) is not int
                or type(end) is not int
                or start != expected_start
                or end < start
            ):
                raise RuntimeError("episode loader range is invalid")
            episode_ranges.append([start, end])
            expected_start = end + 1
        if expected_start != total_frames:
            raise RuntimeError("episode loader ranges do not cover the dataset")

        dataset = LeRobotDataset(repo_id="simple-certified", root=dataset_root)
        if len(dataset) != total_frames:
            raise RuntimeError("loader length differs from dataset metadata")

        expected_tensors = {
            "observation.images.egocentric": (3, height, width),
            "states": (32,),
            "action": (36,),
        }
        tensor_contract = {
            name: {"dtype": "torch.float32", "shape": list(shape)}
            for name, shape in expected_tensors.items()
        }
        for index in range(total_frames):
            sample = dataset[index]
            if not isinstance(sample, dict):
                raise RuntimeError("loader sample is not a mapping")
            loaded_index = sample.get("index")
            if (
                not isinstance(loaded_index, torch.Tensor)
                or loaded_index.numel() != 1
                or loaded_index.dtype != torch.int64
            ):
                raise RuntimeError("loader sample index is not an int64 scalar")
            observed_index = int(loaded_index.item())
            visited.append(observed_index)
            if observed_index != index:
                raise RuntimeError(
                    f"loader index differs: expected {index}, got {observed_index}"
                )
            for name, shape in expected_tensors.items():
                tensor = sample.get(name)
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError(f"loader tensor is absent: {name}")
                if tensor.dtype != torch.float32:
                    raise RuntimeError(f"loader tensor dtype differs: {name}")
                if tuple(tensor.shape) != shape:
                    raise RuntimeError(f"loader tensor shape differs: {name}")
                if not bool(torch.isfinite(tensor).all().item()):
                    raise RuntimeError(f"loader tensor is nonfinite: {name}")
        if visited != list(range(total_frames)):
            raise RuntimeError("loader traversal was incomplete")
        for start, end in episode_ranges:
            if start not in visited or end not in visited:
                raise RuntimeError("loader traversal missed an episode boundary")
        value = {
            "schema_version": 1,
            "verdict": "PASS",
            "sys_path_0": recorded_sys_path_0,
            "visited_indices": visited,
            "episode_ranges": episode_ranges,
            "offline_environment": offline_environment,
            "tensor_contract": tensor_contract,
            "module_provenance": module_provenance,
            "network_policy": network_policy,
        }
        return_code = 0
    except BaseException as exc:
        value = {
            "schema_version": 1,
            "verdict": "FAIL",
            "sys_path_0": source_string,
            "visited_indices": visited,
            "episode_ranges": episode_ranges,
            "offline_environment": offline_environment,
            "tensor_contract": tensor_contract,
            "module_provenance": module_provenance,
            "network_policy": network_policy,
            "error_code": type(exc).__name__.upper(),
            "message": str(exc),
        }
        return_code = 1
    _write_worker_result(result_path, value)
    return return_code


def _terminal_value(
    *,
    certificate_id: uuid.UUID,
    verdict: str,
    identity: dict[str, object],
    entries: list[dict[str, object]],
    stage: str | None = None,
    error_code: str | None = None,
    message: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "certificate_uuid": str(certificate_id),
        "verdict": verdict,
        "dataset_root_identity": identity,
        "manifest_sha256": identity["manifest_sha256"],
        "complete_status_sha256": identity["complete_status_sha256"],
        "entries": sorted(entries, key=lambda item: str(item["path"]).encode()),
    }
    if verdict == "FAIL":
        value.update({"stage": stage, "error_code": error_code, "message": message})
    return value


def _validate_loader_evidence(
    *,
    children: dict[str, Path],
    evidence_root: Path,
    request: dict[str, object],
    identity: dict[str, object],
    terminal_verdict: str,
) -> None:
    environment: dict[str, object] | None = None
    details: dict[str, object] | None = None
    environment_path = children.get("psi0-environment.json")
    if environment_path is not None:
        environment, _ = _strict_json_object(environment_path, "EVIDENCE_TERMINAL")
        details_value = environment.get("details")
        if (
            set(environment) != {"schema_version", "details", "details_sha256"}
            or type(environment.get("schema_version")) is not int
            or environment.get("schema_version") != 1
            or not isinstance(details_value, dict)
            or environment.get("details_sha256")
            != converter.sha256_bytes(converter.canonical_json_bytes(details_value))
        ):
            raise ValueError("PSI0 environment evidence differs")
        details = details_value
        expected_detail_keys = {
            "psi0_root",
            "psi0_src",
            "psi0_commit",
            "tracked_status",
            "python",
            "python_realpath",
            "python_version",
            "platform",
            "packages",
            "distributions",
            "distribution_manifest_sha256",
            "compat_module",
        }
        packages = details.get("packages")
        distributions = details.get("distributions")
        compat_module = details.get("compat_module")
        if (
            set(details) != expected_detail_keys
            or details.get("psi0_root") != request.get("psi0_root")
            or details.get("psi0_commit") != request.get("psi0_commit")
            or details.get("tracked_status") != []
            or details.get("python") != request.get("python")
            or not isinstance(details.get("psi0_src"), str)
            or not isinstance(details.get("python_realpath"), str)
            or not isinstance(details.get("python_version"), str)
            or not isinstance(details.get("platform"), str)
            or not isinstance(packages, dict)
            or set(packages)
            != {"av", "datasets", "numpy", "pyarrow", "torch", "torchvision"}
            or any(
                not isinstance(value, str) or not value for value in packages.values()
            )
            or not isinstance(distributions, list)
            or any(
                not isinstance(item, dict)
                or set(item) != {"name", "version"}
                or not isinstance(item["name"], str)
                or not isinstance(item["version"], str)
                for item in distributions
            )
            or distributions
            != sorted(distributions, key=lambda item: (item["name"], item["version"]))
            or details.get("distribution_manifest_sha256")
            != converter.sha256_bytes(converter.canonical_json_bytes(distributions))
            or not isinstance(compat_module, dict)
            or set(compat_module)
            != {"commit", "relative_path", "origin", "blob_sha256"}
            or compat_module.get("commit") != details.get("psi0_commit")
            or compat_module.get("relative_path") != _COMPAT_RELATIVE_PATH
            or compat_module.get("origin")
            != str(Path(details["psi0_src"]) / "psi/data/lerobot/compat.py")
            or not isinstance(compat_module.get("blob_sha256"), str)
            or len(compat_module["blob_sha256"]) != 64
            or any(
                character not in _HEX_40 for character in compat_module["blob_sha256"]
            )
        ):
            raise ValueError("PSI0 environment details differ")

    command: dict[str, object] | None = None
    command_path = children.get("psi0-loader-command.json")
    if command_path is not None:
        if details is None:
            raise ValueError("PSI0 loader command lacks environment evidence")
        command, _ = _strict_json_object(command_path, "EVIDENCE_TERMINAL")
        worker_environment = command.get("environment")
        argv = command.get("argv")
        if (
            set(command)
            != {"schema_version", "argv", "environment", "environment_sha256"}
            or type(command.get("schema_version")) is not int
            or command.get("schema_version") != 1
            or not isinstance(worker_environment, dict)
            or command.get("environment_sha256")
            != converter.sha256_bytes(
                converter.canonical_json_bytes(worker_environment)
            )
            or set(worker_environment)
            != {
                "PATH",
                "HOME",
                "HF_HOME",
                "HF_DATASETS_CACHE",
                "XDG_CACHE_HOME",
                "TORCH_HOME",
                "TMPDIR",
                "HF_HUB_OFFLINE",
                "HF_DATASETS_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "PYTHONNOUSERSITE",
            }
            or worker_environment.get("PATH") != "/usr/bin:/bin"
            or any(
                worker_environment.get(name) != "1"
                for name in (
                    "HF_HUB_OFFLINE",
                    "HF_DATASETS_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "PYTHONNOUSERSITE",
                )
            )
            or not isinstance(argv, list)
            or len(argv) != 10
            or argv[0] != details["python"]
            or argv[1:5]
            != [
                "-I",
                "scripts/certify_psi0_dataset.py",
                "--loader-worker",
                "--psi0-src",
            ]
            or argv[5] != details["psi0_src"]
            or argv[6:8] != ["--dataset-root", identity["path"]]
            or argv[8] != "--result"
        ):
            raise ValueError("PSI0 loader command evidence differs")
        cache_root = evidence_root / ".loader-cache"
        expected_cache_paths = {
            "HOME": cache_root / "home",
            "HF_HOME": cache_root / "hf",
            "HF_DATASETS_CACHE": cache_root / "datasets",
            "XDG_CACHE_HOME": cache_root / "xdg",
            "TORCH_HOME": cache_root / "torch",
            "TMPDIR": cache_root / "tmp",
        }
        if any(
            worker_environment.get(key) != str(path)
            for key, path in expected_cache_paths.items()
        ):
            raise ValueError("PSI0 loader cache paths differ")
        result_path = Path(argv[9])
        if (
            result_path.parent != evidence_root
            or not result_path.name.startswith(".psi0-loader-result-")
            or not result_path.name.endswith(".tmp")
            or os.path.lexists(result_path)
        ):
            raise ValueError("PSI0 loader result path differs")

    cache_path = children.get("psi0-loader-cache.json")
    if cache_path is not None:
        if command is None:
            raise ValueError("PSI0 loader cache lacks command evidence")
        cache, _ = _strict_json_object(cache_path, "EVIDENCE_TERMINAL")
        cache_entries = cache.get("entries")
        if (
            set(cache) != {"schema_version", "entries"}
            or type(cache.get("schema_version")) is not int
            or cache.get("schema_version") != 1
            or not isinstance(cache_entries, list)
        ):
            raise ValueError("PSI0 loader cache evidence differs")
        paths: list[str] = []
        for entry in cache_entries:
            if not isinstance(entry, dict) or set(entry) != {
                "path",
                "type",
                "size",
                "sha256",
            }:
                raise ValueError("PSI0 loader cache entry schema differs")
            path = entry.get("path")
            entry_type = entry.get("type")
            digest = entry.get("sha256")
            path_parts = path.split("/") if isinstance(path, str) else []
            if (
                not isinstance(path, str)
                or not path
                or Path(path).is_absolute()
                or any(part in {"", ".", ".."} for part in path_parts)
                or path_parts[0] not in _MANDATORY_CACHE_DIRECTORIES
                or entry_type not in {"directory", "regular"}
                or type(entry.get("size")) is not int
                or entry["size"] < 0
                or (entry_type == "directory" and digest is not None)
                or (
                    entry_type == "regular"
                    and (
                        not isinstance(digest, str)
                        or len(digest) != 64
                        or any(character not in _HEX_40 for character in digest)
                    )
                )
            ):
                raise ValueError("PSI0 loader cache entry differs")
            paths.append(path)
        if paths != sorted(paths, key=str.encode) or len(paths) != len(set(paths)):
            raise ValueError("PSI0 loader cache entries are not ordered and unique")
        entries_by_path = {entry["path"]: entry for entry in cache_entries}
        for path, entry in entries_by_path.items():
            parts = path.split("/")
            for length in range(1, len(parts)):
                ancestor = entries_by_path.get("/".join(parts[:length]))
                if ancestor is None or ancestor["type"] != "directory":
                    raise ValueError(
                        f"PSI0 loader cache ancestor differs for {entry['path']}"
                    )
        top_level = {
            entry["path"]: entry for entry in cache_entries if "/" not in entry["path"]
        }
        if set(top_level) != _MANDATORY_CACHE_DIRECTORIES or any(
            top_level[name]["type"] != "directory"
            or top_level[name]["sha256"] is not None
            for name in _MANDATORY_CACHE_DIRECTORIES
        ):
            raise ValueError("PSI0 loader mandatory cache roots differ")

    result_path = children.get("psi0-loader-result.json")
    if result_path is not None:
        if command is None or cache_path is None:
            raise ValueError("PSI0 loader result lacks preceding evidence")
        loader_result, _ = _strict_json_object(result_path, "EVIDENCE_TERMINAL")
        worker_result = loader_result.get("result")
        if (
            set(loader_result)
            != {
                "schema_version",
                "returncode",
                "stdout",
                "stderr",
                "result_sha256",
                "result",
            }
            or type(loader_result.get("schema_version")) is not int
            or loader_result.get("schema_version") != 1
            or type(loader_result.get("returncode")) is not int
            or not isinstance(loader_result.get("stdout"), str)
            or not isinstance(loader_result.get("stderr"), str)
            or not isinstance(worker_result, dict)
            or loader_result.get("result_sha256")
            != converter.sha256_bytes(converter.canonical_json_bytes(worker_result))
        ):
            raise ValueError("PSI0 loader result evidence differs")
        worker_verdict = worker_result.get("verdict")
        worker_fields = {
            "schema_version",
            "verdict",
            "sys_path_0",
            "visited_indices",
            "episode_ranges",
            "offline_environment",
            "tensor_contract",
            "module_provenance",
            "network_policy",
        } | (set() if worker_verdict == "PASS" else {"error_code", "message"})
        network_policy = worker_result.get("network_policy")
        if (
            set(worker_result) != worker_fields
            or type(worker_result.get("schema_version")) is not int
            or worker_result.get("schema_version") != 1
            or worker_verdict not in {"PASS", "FAIL"}
            or worker_result.get("sys_path_0") != details["psi0_src"]
            or not _json_exact_equal(
                worker_result.get("module_provenance"), details["compat_module"]
            )
            or not _json_exact_equal(
                worker_result.get("offline_environment"), command["environment"]
            )
            or not isinstance(worker_result.get("visited_indices"), list)
            or any(type(index) is not int for index in worker_result["visited_indices"])
            or not isinstance(worker_result.get("episode_ranges"), list)
            or not isinstance(worker_result.get("tensor_contract"), dict)
            or not _loader_audit_policy_is_valid(network_policy)
            or (worker_verdict == "PASS" and loader_result["returncode"] != 0)
            or (worker_verdict == "FAIL" and loader_result["returncode"] == 0)
        ):
            raise ValueError("PSI0 loader worker result differs")
        if terminal_verdict == "PASS" and worker_verdict != "PASS":
            raise ValueError("PASS evidence contains a failed PSI0 loader result")
        if worker_verdict == "PASS":
            info, _ = _strict_json_object(
                Path(identity["path"]) / "meta/info.json", "EVIDENCE_TERMINAL"
            )
            episodes, _ = _strict_jsonl(
                Path(identity["path"]) / "meta/episodes.jsonl",
                "EVIDENCE_TERMINAL",
            )
            total_frames = info.get("total_frames")
            features = info.get("features")
            image = (
                features.get("observation.images.egocentric")
                if isinstance(features, dict)
                else None
            )
            image_shape = image.get("shape") if isinstance(image, dict) else None
            if (
                type(total_frames) is not int
                or total_frames <= 0
                or not isinstance(image_shape, list)
                or len(image_shape) != 3
                or any(type(value) is not int for value in image_shape)
                or any(value <= 0 for value in image_shape)
                or image_shape[2] != 3
            ):
                raise ValueError("PSI0 loader dataset semantics differ")
            expected_ranges: list[list[int]] = []
            expected_start = 0
            for episode in episodes:
                start = episode.get("dataset_from_index")
                end = episode.get("dataset_to_index")
                if (
                    type(start) is not int
                    or type(end) is not int
                    or start != expected_start
                    or end < start
                ):
                    raise ValueError("PSI0 loader episode ranges differ")
                expected_ranges.append([start, end])
                expected_start = end + 1
            if (
                expected_start != total_frames
                or worker_result["visited_indices"] != list(range(total_frames))
                or worker_result["episode_ranges"] != expected_ranges
                or not _json_exact_equal(
                    worker_result["tensor_contract"],
                    {
                        "action": {"dtype": "torch.float32", "shape": [36]},
                        "observation.images.egocentric": {
                            "dtype": "torch.float32",
                            "shape": [3, image_shape[0], image_shape[1]],
                        },
                        "states": {"dtype": "torch.float32", "shape": [32]},
                    },
                )
                or network_policy["violations"]
            ):
                raise ValueError("PSI0 loader result differs from bound dataset")


def validate_evidence_terminal(evidence_root: Path) -> dict[str, object]:
    root_metadata = evidence_root.lstat()
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_IMODE(root_metadata.st_mode) != 0o555
    ):
        raise ValueError("evidence root mode or type differs")
    children = {path.name: path for path in evidence_root.iterdir()}
    terminals = [
        children[name] for name in ("PASS.json", "FAIL.json") if name in children
    ]
    if len(terminals) != 1:
        raise ValueError("evidence must contain exactly one terminal record")
    terminal_metadata = terminals[0].lstat()
    if (
        not stat.S_ISREG(terminal_metadata.st_mode)
        or terminal_metadata.st_nlink != 1
        or stat.S_IMODE(terminal_metadata.st_mode) != 0o444
    ):
        raise ValueError("terminal record type, link count, or mode differs")
    terminal, _ = _strict_json_object(terminals[0], "EVIDENCE_TERMINAL")
    verdict = "PASS" if terminals[0].name == "PASS.json" else "FAIL"
    required_keys = {
        "schema_version",
        "certificate_uuid",
        "verdict",
        "dataset_root_identity",
        "manifest_sha256",
        "complete_status_sha256",
        "entries",
    }
    if verdict == "FAIL":
        required_keys |= {"stage", "error_code", "message"}
    if (
        set(terminal) != required_keys
        or type(terminal.get("schema_version")) is not int
        or terminal.get("schema_version") != 1
        or terminal.get("verdict") != verdict
    ):
        raise ValueError("terminal record schema differs")
    certificate_uuid = terminal.get("certificate_uuid")
    if not isinstance(certificate_uuid, str):
        raise ValueError("terminal certificate UUID is invalid")
    try:
        certificate_id = uuid.UUID(certificate_uuid)
    except ValueError as exc:
        raise ValueError("terminal certificate UUID is invalid") from exc
    identity = terminal.get("dataset_root_identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
        raise ValueError("terminal dataset identity is absent")
    dataset_root = Path(identity["path"])
    dataset_fd = os.open(
        dataset_root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        metadata = os.fstat(dataset_fd)
        converter._validate_descriptor_path(
            dataset_fd, dataset_root, expected_type="directory"
        )
        manifest_sha256 = _hash_dataset_member(
            dataset_fd, ("meta", "conversion_manifest.json")
        )
        complete_status_sha256 = _hash_dataset_member(
            dataset_fd, ("CONVERSION_STATUS.json",)
        )
        converter._validate_descriptor_path(
            dataset_fd, dataset_root, expected_type="directory"
        )
    finally:
        os.close(dataset_fd)
    actual_identity = {
        "path": str(dataset_root.absolute()),
        "st_dev": metadata.st_dev,
        "st_ino": metadata.st_ino,
        "st_uid": metadata.st_uid,
        "st_gid": metadata.st_gid,
        "st_mode": metadata.st_mode,
        "st_nlink": metadata.st_nlink,
        "manifest_sha256": manifest_sha256,
        "complete_status_sha256": complete_status_sha256,
    }
    if not _json_exact_equal(identity, actual_identity):
        raise ValueError("terminal dataset identity differs")
    if (
        terminal.get("manifest_sha256") != identity["manifest_sha256"]
        or terminal.get("complete_status_sha256") != identity["complete_status_sha256"]
    ):
        raise ValueError("terminal publication digests differ")
    entries = terminal.get("entries")
    if not isinstance(entries, list) or any(
        not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}
        for entry in entries
    ):
        raise ValueError("terminal evidence entry schema differs")
    if any(
        not isinstance(entry["path"], str)
        or type(entry["size"]) is not int
        or entry["size"] < 0
        or not isinstance(entry["sha256"], str)
        or len(entry["sha256"]) != 64
        or any(character not in _HEX_40 for character in entry["sha256"])
        for entry in entries
    ):
        raise ValueError("terminal evidence entry values differ")
    if entries != sorted(entries, key=lambda item: str(item.get("path", "")).encode()):
        raise ValueError("terminal entries are not canonically ordered")
    expected_names = {entry["path"] for entry in entries}
    if len(expected_names) != len(entries):
        raise ValueError("terminal entries contain duplicates")
    expected_prefix = set(_EVIDENCE_FILES[: len(entries)])
    if verdict == "PASS":
        if len(entries) != len(_EVIDENCE_FILES) or expected_names != set(
            _EVIDENCE_FILES
        ):
            raise ValueError("PASS evidence is incomplete")
    elif (
        not entries
        or expected_names != expected_prefix
        or terminal.get("stage") != _FAILURE_STAGE_BY_COMPLETED_COUNT[len(entries)]
        or not isinstance(terminal.get("error_code"), str)
        or not terminal["error_code"]
        or not isinstance(terminal.get("message"), str)
    ):
        raise ValueError("FAIL evidence completion prefix differs")
    actual_names = set(children) - {terminals[0].name}
    if expected_names != actual_names:
        raise ValueError("terminal evidence membership differs")
    for entry in entries:
        name = entry["path"]
        if (
            not isinstance(name, str)
            or not name
            or name in {"PASS.json", "FAIL.json"}
            or Path(name).name != name
        ):
            raise ValueError("terminal evidence path is unsafe")
        path = children[name]
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("unsafe evidence member")
        if (
            metadata.st_size != entry["size"]
            or converter.sha256_file(path) != entry["sha256"]
        ):
            raise ValueError("evidence member identity differs")
        if stat.S_IMODE(metadata.st_mode) != 0o444:
            raise ValueError("evidence member mode differs")
    request, _ = _strict_json_object(
        children["certificate-request.json"], "EVIDENCE_TERMINAL"
    )
    if (
        set(request)
        != {
            "schema_version",
            "certificate_uuid",
            "dataset_root",
            "psi0_root",
            "psi0_commit",
            "python",
        }
        or type(request.get("schema_version")) is not int
        or request.get("schema_version") != 1
        or request.get("certificate_uuid") != terminal["certificate_uuid"]
        or request.get("dataset_root") != identity["path"]
    ):
        raise ValueError("evidence identity binding differs")
    if "dataset-root-identity.json" in children:
        recorded_identity, _ = _strict_json_object(
            children["dataset-root-identity.json"], "EVIDENCE_TERMINAL"
        )
        if not _json_exact_equal(recorded_identity, identity):
            raise ValueError("evidence identity binding differs")
    expected_root = dataset_root.parent / (
        f".{dataset_root.name}.certification-"
        f"{identity['manifest_sha256']}-{certificate_id}"
    )
    if evidence_root.absolute() != expected_root.absolute():
        raise ValueError("evidence root path differs from terminal identity")
    _validate_loader_evidence(
        children=children,
        evidence_root=evidence_root,
        request=request,
        identity=identity,
        terminal_verdict=verdict,
    )
    return terminal


def _error_identity(error: BaseException) -> tuple[str, str]:
    original = error.original if isinstance(error, DurableEvidenceWriteError) else error
    code = (
        original.code
        if isinstance(original, DatasetValidationError)
        else type(original).__name__.upper()
    )
    return code, str(original)


def _remove_terminal_if_present(root_fd: int, name: str) -> None:
    try:
        os.fchmod(root_fd, 0o700)
        os.unlink(name, dir_fd=root_fd)
        os.fsync(root_fd)
    except FileNotFoundError:
        pass


def _seal_evidence_root(root_fd: int, parent_fd: int) -> None:
    os.fchmod(root_fd, 0o555)
    os.fsync(root_fd)
    _evidence_fault("evidence_root:after_root_fsync")
    os.fsync(parent_fd)
    _evidence_fault("evidence_root:after_parent_fsync")


def _finish_failed_evidence(
    *,
    root_fd: int,
    parent_fd: int,
    evidence_root: Path,
    certificate_id: uuid.UUID,
    identity: dict[str, object] | None,
    error: BaseException,
) -> NoReturn:
    error_code, message = _error_identity(error)
    if isinstance(error, EvidenceDurabilityUncertain) or identity is None:
        os.close(root_fd)
        os.close(parent_fd)
        raise PublishedUncertifiedError(evidence_root, error_code, message) from error
    try:
        entries = _completed_evidence_entries(root_fd)
        if not entries:
            raise EvidenceDurabilityUncertain("no durable evidence prefix exists")
        stage = _FAILURE_STAGE_BY_COMPLETED_COUNT[len(entries)]
        terminal = _terminal_value(
            certificate_id=certificate_id,
            verdict="FAIL",
            identity=identity,
            entries=entries,
            stage=stage,
            error_code=error_code,
            message=message,
        )
        try:
            _write_evidence_file(root_fd, "FAIL.json", terminal)
        except DurableEvidenceWriteError as terminal_error:
            if terminal_error.name != "FAIL.json":
                raise
        _seal_evidence_root(root_fd, parent_fd)
        validate_evidence_terminal(evidence_root)
    except BaseException:
        try:
            _remove_terminal_if_present(root_fd, "FAIL.json")
        finally:
            os.close(root_fd)
            os.close(parent_fd)
        raise PublishedUncertifiedError(evidence_root, error_code, message) from error
    os.close(root_fd)
    os.close(parent_fd)
    raise PublishedUncertifiedError(evidence_root, error_code, message) from error


def _certify_published_dataset_locked(
    publication: converter.PublicationResult,
    *,
    psi0_root: Path,
    psi0_commit: str,
    python: Path,
    certificate_uuid: uuid.UUID | None = None,
) -> Path:
    _require_sha256(publication.manifest_sha256, "manifest digest")
    _require_sha256(publication.complete_status_sha256, "complete-status digest")
    verified_identity = _dataset_root_identity(publication)
    certificate_id = certificate_uuid or uuid.uuid4()
    evidence_root = publication.dataset_root.parent / (
        f".{publication.dataset_root.name}.certification-"
        f"{publication.manifest_sha256}-{certificate_id}"
    )
    parent_fd = os.open(
        evidence_root.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    evidence_name = converter._validated_basename(evidence_root, "evidence root")
    root_fd = None
    try:
        converter._validate_descriptor_path(
            parent_fd, evidence_root.parent, expected_type="directory"
        )
        os.mkdir(evidence_name, 0o700, dir_fd=parent_fd)
        root_fd = os.open(
            evidence_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        _evidence_fault("evidence_root:after_mkdir")
        created = converter._validate_descriptor_entry(
            root_fd,
            parent_fd,
            evidence_name,
            expected_type="directory",
        )
        if stat.S_IMODE(created.st_mode) != 0o700:
            os.fchmod(root_fd, 0o700)
            os.fsync(root_fd)
        os.fsync(parent_fd)
        _evidence_fault("evidence_root:after_creation_parent_fsync")
    except FileExistsError:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)
        raise
    except BaseException as exc:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)
        raise PublishedUncertifiedError(
            evidence_root, "EVIDENCE_ROOT_UNCERTAIN", str(exc)
        ) from exc
    assert root_fd is not None
    entries: list[dict[str, object]] = []
    identity: dict[str, object] | None = verified_identity
    try:
        entries.append(
            _write_evidence_file(
                root_fd,
                "certificate-request.json",
                {
                    "schema_version": 1,
                    "certificate_uuid": str(certificate_id),
                    "dataset_root": str(publication.dataset_root.absolute()),
                    "psi0_root": str(psi0_root.resolve(strict=False)),
                    "psi0_commit": psi0_commit,
                    "python": str(python.absolute()),
                },
            )
        )
        entries.append(
            _write_evidence_file(root_fd, "dataset-root-identity.json", identity)
        )
        result = validate_dataset(
            publication.dataset_root, expected=None, require_final_modes=True
        )
        if (
            result.manifest_sha256 != publication.manifest_sha256
            or result.complete_status_sha256 != publication.complete_status_sha256
        ):
            raise DatasetValidationError(
                "PUBLICATION_IDENTITY", "publication digests differ"
            )
        entries.append(
            _write_evidence_file(
                root_fd,
                "dataset-validation.json",
                {
                    "schema_version": 1,
                    "total_episodes": result.total_episodes,
                    "total_frames": result.total_frames,
                    "total_tasks": result.total_tasks,
                    "total_videos": result.total_videos,
                    "tree_digest": result.tree_digest,
                },
            )
        )
        entries.append(
            _write_evidence_file(
                root_fd,
                "source-validation.json",
                {"schema_version": 1, "entries": list(result.source_entries)},
            )
        )
        environment = _collect_psi0_environment(psi0_root, psi0_commit, python)
        entries.append(
            _write_evidence_file(root_fd, "psi0-environment.json", environment)
        )
        loader_command, loader_cache, loader_result = _run_psi0_loader(
            root_fd=root_fd,
            evidence_root=evidence_root,
            result=result,
            psi0_root=psi0_root,
            psi0_commit=psi0_commit,
            python=python,
        )
        entries.append(
            _write_evidence_file(
                root_fd,
                "psi0-loader-command.json",
                loader_command,
            )
        )
        entries.append(
            _write_evidence_file(
                root_fd,
                "psi0-loader-cache.json",
                loader_cache,
            )
        )
        entries.append(
            _write_evidence_file(
                root_fd,
                "psi0-loader-result.json",
                loader_result,
            )
        )
        worker_result = loader_result.get("result")
        if (
            not isinstance(worker_result, dict)
            or worker_result.get("verdict") != "PASS"
        ):
            message = (
                worker_result.get("message", "loader validation failed")
                if isinstance(worker_result, dict)
                else "loader result is absent"
            )
            raise DatasetValidationError("PSI0_LOADER_VALIDATION", str(message))
        terminal_name = "PASS.json"
        terminal = _terminal_value(
            certificate_id=certificate_id,
            verdict="PASS",
            identity=identity,
            entries=entries,
        )
    except CacheCleanupError as exc:
        os.close(root_fd)
        os.close(parent_fd)
        raise PublishedUncertifiedError(
            evidence_root, "CACHE_CLEANUP_FAILED", str(exc)
        ) from exc
    except BaseException as exc:
        _finish_failed_evidence(
            root_fd=root_fd,
            parent_fd=parent_fd,
            evidence_root=evidence_root,
            certificate_id=certificate_id,
            identity=identity,
            error=exc,
        )
    try:
        _write_evidence_file(root_fd, terminal_name, terminal)
        _seal_evidence_root(root_fd, parent_fd)
        validate_evidence_terminal(evidence_root)
    except BaseException as exc:
        _remove_terminal_if_present(root_fd, "PASS.json")
        _finish_failed_evidence(
            root_fd=root_fd,
            parent_fd=parent_fd,
            evidence_root=evidence_root,
            certificate_id=certificate_id,
            identity=identity,
            error=exc,
        )
    os.close(root_fd)
    os.close(parent_fd)
    return evidence_root


def certify_published_dataset(
    publication: converter.PublicationResult,
    *,
    psi0_root: Path,
    psi0_commit: str,
    python: Path,
    certificate_uuid: uuid.UUID | None = None,
) -> Path:
    """Validate a published dataset and durably write sibling evidence."""

    if publication.state != "published":
        raise converter.PublicationUncertainError(publication)
    converter.assert_conversion_lock_held(publication.dataset_root)
    return _certify_published_dataset_locked(
        publication,
        psi0_root=psi0_root,
        psi0_commit=psi0_commit,
        python=python,
        certificate_uuid=certificate_uuid,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--loader-worker", action="store_true")
    parser.add_argument("--psi0-src", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--print-tree-digest", action="store_true")
    parser.add_argument("--recover-publication-uncertain", action="store_true")
    parser.add_argument("--psi0-root", type=Path)
    parser.add_argument("--psi0-commit")
    parser.add_argument("--python", type=Path)
    args = parser.parse_args(argv)
    if args.loader_worker:
        if args.psi0_src is None or args.dataset_root is None or args.result is None:
            parser.error("loader worker requires PSI0 src, dataset root, and result")
        return _loader_worker_main(
            psi0_src=args.psi0_src,
            dataset_root=args.dataset_root,
            result_path=args.result,
        )
    if args.dataset_root is None:
        parser.error("--dataset-root is required")
    try:
        if args.print_tree_digest:
            result = validate_dataset(
                args.dataset_root, expected=None, require_final_modes=True
            )
            print(result.tree_digest)
            return 0
        if not args.recover_publication_uncertain:
            parser.error("certification requires --recover-publication-uncertain")
        if args.psi0_root is None or args.psi0_commit is None or args.python is None:
            parser.error("certification requires PSI0 root, commit, and Python")
        with converter.conversion_lock(args.dataset_root):
            result = validate_dataset(
                args.dataset_root, expected=None, require_final_modes=True
            )
            converter.PublicationFilesystem().fsync_directory(args.dataset_root.parent)
            if result.manifest_sha256 is None or result.complete_status_sha256 is None:
                raise DatasetValidationError(
                    "PUBLICATION_IDENTITY", "published metadata absent"
                )
            publication = converter.PublicationResult(
                dataset_root=args.dataset_root,
                manifest_sha256=result.manifest_sha256,
                complete_status_sha256=result.complete_status_sha256,
                state="published",
            )
            certify_published_dataset(
                publication,
                psi0_root=args.psi0_root,
                psi0_commit=args.psi0_commit,
                python=args.python,
            )
        return 0
    except (
        DatasetValidationError,
        PublishedUncertifiedError,
        OSError,
        RuntimeError,
    ) as exc:
        print(f"PUBLISHED_UNCERTIFIED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
