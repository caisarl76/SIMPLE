#!/usr/bin/env python3
import argparse
import ctypes
import errno
import fcntl
import glob
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import uuid
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Literal

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
class WrittenMedia:
    identity: MediaIdentity
    sha256: str
    st_dev: int
    st_ino: int
    st_size: int


@dataclass(frozen=True)
class SourceRootIdentity:
    path: Path
    st_dev: int
    st_ino: int
    st_uid: int
    st_gid: int
    st_mode: int


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
    input_roots: tuple[SourceRootIdentity, ...]
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


def _validate_descriptor_path(
    fd: int,
    path: Path,
    *,
    expected_type: Literal["file", "directory"],
    require_single_link: bool = False,
) -> os.stat_result:
    descriptor = os.fstat(fd)
    try:
        pathname = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"opened {expected_type} path disappeared: {path}") from exc
    predicate = stat.S_ISREG if expected_type == "file" else stat.S_ISDIR
    if not predicate(descriptor.st_mode) or not predicate(pathname.st_mode):
        raise RuntimeError(f"unsafe {expected_type}: {path}")
    if (descriptor.st_dev, descriptor.st_ino) != (pathname.st_dev, pathname.st_ino):
        raise RuntimeError(f"opened {expected_type} identity changed: {path}")
    if require_single_link and (descriptor.st_nlink != 1 or pathname.st_nlink != 1):
        raise RuntimeError(f"opened {expected_type} has unsafe link count: {path}")
    return descriptor


def _sha256_descriptor(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while block := os.read(fd, 1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


_MFD_CLOEXEC = 0x0001
_MFD_ALLOW_SEALING = 0x0002
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_REQUIRED_SNAPSHOT_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008


def _verify_snapshot_seals(fd: int) -> None:
    try:
        actual_seals = fcntl.fcntl(fd, _F_GET_SEALS)
    except OSError as exc:
        raise RuntimeError("sealed memfd verification failed") from exc
    if actual_seals & _REQUIRED_SNAPSHOT_SEALS != _REQUIRED_SNAPSHOT_SEALS:
        raise RuntimeError("sealed memfd is missing required write seals")


def _create_sealed_snapshot(source_fd: int, *, expected_sha256: str) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        memfd_create = libc.memfd_create
    except AttributeError as exc:
        raise RuntimeError("sealed memfd support is unavailable") from exc
    memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    memfd_create.restype = ctypes.c_int
    snapshot_fd = memfd_create(
        b"psi0-source-video",
        _MFD_CLOEXEC | _MFD_ALLOW_SEALING,
    )
    if snapshot_fd < 0:
        error_number = ctypes.get_errno()
        raise RuntimeError(f"sealed memfd creation failed: {os.strerror(error_number)}")
    try:
        if not fcntl.fcntl(snapshot_fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC:
            raise RuntimeError("sealed memfd is missing close-on-exec")
        digest = hashlib.sha256()
        os.lseek(source_fd, 0, os.SEEK_SET)
        while block := os.read(source_fd, 1024 * 1024):
            digest.update(block)
            remaining = memoryview(block)
            while remaining:
                written = os.write(snapshot_fd, remaining)
                if written <= 0:
                    raise OSError("snapshot copy made no forward progress")
                remaining = remaining[written:]
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("source video SHA-256 differs during snapshot")
        os.fsync(snapshot_fd)
        try:
            fcntl.fcntl(snapshot_fd, _F_ADD_SEALS, _REQUIRED_SNAPSHOT_SEALS)
        except OSError as exc:
            raise RuntimeError("sealed memfd write protection failed") from exc
        _verify_snapshot_seals(snapshot_fd)
        os.lseek(snapshot_fd, 0, os.SEEK_SET)
        return snapshot_fd
    except BaseException:
        os.close(snapshot_fd)
        raise


def fsync_directory(path: Path, *, directory_fd: int | None = None) -> None:
    owns_fd = directory_fd is None
    fd = _open_validated_directory(path) if owns_fd else directory_fd
    assert fd is not None
    try:
        _validate_descriptor_path(fd, path, expected_type="directory")
        os.fsync(fd)
        _validate_descriptor_path(fd, path, expected_type="directory")
    finally:
        if owns_fd:
            os.close(fd)


def copy_file_exclusive(
    source: Path,
    destination_parent_fd: int,
    destination_name: bytes,
    *,
    expected_sha256: str,
) -> tuple[str, int]:
    source_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    destination_flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    source_fd = os.open(source, source_flags)
    try:
        _validate_descriptor_path(source_fd, source, expected_type="file")
        source_digest = _sha256_descriptor(source_fd)
        if source_digest != expected_sha256:
            raise RuntimeError("source video SHA-256 differs from preflight identity")
        os.lseek(source_fd, 0, os.SEEK_SET)

        destination_fd: int | None = os.open(
            destination_name,
            destination_flags,
            0o644,
            dir_fd=destination_parent_fd,
        )
        verify_fd: int | None = None
        try:
            assert destination_fd is not None
            destination_identity = _validate_descriptor_entry(
                destination_fd,
                destination_parent_fd,
                destination_name,
                expected_type="file",
                require_single_link=True,
            )
            copied_digest = hashlib.sha256()
            while block := os.read(source_fd, 1024 * 1024):
                copied_digest.update(block)
                remaining = memoryview(block)
                while remaining:
                    written = os.write(destination_fd, remaining)
                    if written <= 0:
                        raise OSError("copy made no forward progress")
                    remaining = remaining[written:]
            os.fsync(destination_fd)
            after_fsync = _validate_descriptor_entry(
                destination_fd,
                destination_parent_fd,
                destination_name,
                expected_type="file",
                require_single_link=True,
            )
            if (after_fsync.st_dev, after_fsync.st_ino) != (
                destination_identity.st_dev,
                destination_identity.st_ino,
            ):
                raise RuntimeError("destination video identity changed during copy")

            _validate_descriptor_path(source_fd, source, expected_type="file")
            if copied_digest.hexdigest() != source_digest:
                raise RuntimeError("copied video SHA-256 differs from source")
            verify_fd = os.open(
                destination_name,
                source_flags,
                dir_fd=destination_parent_fd,
            )
            reopened_identity = _validate_descriptor_entry(
                verify_fd,
                destination_parent_fd,
                destination_name,
                expected_type="file",
                require_single_link=True,
            )
            if (reopened_identity.st_dev, reopened_identity.st_ino) != (
                destination_identity.st_dev,
                destination_identity.st_ino,
            ):
                raise RuntimeError("destination verification identity changed")
            destination_digest = _sha256_descriptor(verify_fd)
            if destination_digest != source_digest:
                raise RuntimeError("destination video SHA-256 differs from source")

            writer_to_close = destination_fd
            destination_fd = None
            os.close(writer_to_close)
            final_identity = _validate_descriptor_entry(
                verify_fd,
                destination_parent_fd,
                destination_name,
                expected_type="file",
                require_single_link=True,
            )
            if (final_identity.st_dev, final_identity.st_ino) != (
                destination_identity.st_dev,
                destination_identity.st_ino,
            ):
                raise RuntimeError("destination verification identity changed")
            if _sha256_descriptor(verify_fd) != destination_digest:
                raise RuntimeError("destination video changed during verification")
            result_fd = verify_fd
            verify_fd = None
            return destination_digest, result_fd
        finally:
            if verify_fd is not None:
                os.close(verify_fd)
            if destination_fd is not None:
                os.close(destination_fd)
    finally:
        os.close(source_fd)


def _validated_basename(path: Path, label: str) -> bytes:
    name = path.name
    if not name or name in (".", "..") or path.parent / name != path:
        raise ValueError(f"{label} must have a valid basename")
    return os.fsencode(name)


def _open_validated_directory(path: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        _validate_descriptor_path(fd, path, expected_type="directory")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _validate_descriptor_entry(
    fd: int,
    parent_fd: int,
    name: str | bytes,
    *,
    expected_type: Literal["file", "directory"],
    require_single_link: bool = False,
) -> os.stat_result:
    descriptor = os.fstat(fd)
    try:
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"opened {expected_type} entry disappeared") from exc
    predicate = stat.S_ISREG if expected_type == "file" else stat.S_ISDIR
    if not predicate(descriptor.st_mode) or not predicate(entry.st_mode):
        raise RuntimeError(f"unsafe {expected_type} entry")
    if (descriptor.st_dev, descriptor.st_ino) != (entry.st_dev, entry.st_ino):
        raise RuntimeError(f"opened {expected_type} entry identity changed")
    if require_single_link and (descriptor.st_nlink != 1 or entry.st_nlink != 1):
        raise RuntimeError(f"opened {expected_type} entry has unsafe link count")
    return descriptor


def _rename_noreplace_at(
    source_parent_fd: int,
    source_name: bytes,
    destination_parent_fd: int,
    destination_name: bytes,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    source_metadata = os.stat(
        source_name, dir_fd=source_parent_fd, follow_symlinks=False
    )
    if not (
        stat.S_ISREG(source_metadata.st_mode) or stat.S_ISDIR(source_metadata.st_mode)
    ):
        raise RuntimeError("unsafe rename source")
    if stat.S_ISREG(source_metadata.st_mode) and source_metadata.st_nlink != 1:
        raise RuntimeError("rename source has unsafe link count")
    if (
        expected_identity is not None
        and (
            source_metadata.st_dev,
            source_metadata.st_ino,
        )
        != expected_identity
    ):
        raise RuntimeError("rename source identity changed before publication")

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is unavailable") from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            source_parent_fd,
            source_name,
            destination_parent_fd,
            destination_name,
            1,
        )
        != 0
    ):
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number))
        raise OSError(error_number, os.strerror(error_number))

    destination_metadata = os.stat(
        destination_name,
        dir_fd=destination_parent_fd,
        follow_symlinks=False,
    )
    if (destination_metadata.st_dev, destination_metadata.st_ino) != (
        source_metadata.st_dev,
        source_metadata.st_ino,
    ):
        raise RuntimeError("renamed output identity differs from source")
    if (
        stat.S_ISREG(destination_metadata.st_mode)
        and destination_metadata.st_nlink != 1
    ):
        raise RuntimeError("renamed output has unsafe link count")


def rename_noreplace(
    source: Path,
    destination: Path,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    source_name = _validated_basename(source, "source")
    destination_name = _validated_basename(destination, "destination")
    source_parent_fd = _open_validated_directory(source.parent)
    try:
        destination_parent_fd = _open_validated_directory(destination.parent)
        try:
            _rename_noreplace_at(
                source_parent_fd,
                source_name,
                destination_parent_fd,
                destination_name,
                expected_identity=expected_identity,
            )
            _validate_descriptor_path(
                source_parent_fd, source.parent, expected_type="directory"
            )
            _validate_descriptor_path(
                destination_parent_fd,
                destination.parent,
                expected_type="directory",
            )
        finally:
            os.close(destination_parent_fd)
    finally:
        os.close(source_parent_fd)


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


def probe_media(path: Path, *, pass_fds: tuple[int, ...] = ()) -> MediaIdentity:
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
            pass_fds=pass_fds,
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


def _snapshot_regular_file(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[bytes, str]:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = _validate_descriptor_path(
            fd,
            path,
            expected_type="file",
            require_single_link=True,
        )
        digest = hashlib.sha256()
        chunks = []
        os.lseek(fd, 0, os.SEEK_SET)
        while block := os.read(fd, 1024 * 1024):
            digest.update(block)
            chunks.append(block)
        payload = b"".join(chunks)
        actual_sha256 = digest.hexdigest()
        if len(payload) != before.st_size:
            raise RuntimeError(f"{label} size changed while being snapshotted")
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise RuntimeError(f"{label} SHA-256 differs from preflight identity")
        after = _validate_descriptor_path(
            fd,
            path,
            expected_type="file",
            require_single_link=True,
        )
        if (after.st_dev, after.st_ino, after.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise RuntimeError(f"{label} identity changed while being snapshotted")
        if _sha256_descriptor(fd) != actual_sha256:
            raise RuntimeError(f"{label} changed while being snapshotted")
        return payload, actual_sha256
    finally:
        os.close(fd)


def media_selection_key(value: MediaIdentity) -> tuple[object, ...]:
    return (
        value.codec_name,
        value.pixel_format,
        value.width,
        value.height,
        value.average_frame_rate,
        value.nominal_frame_rate,
        value.audio_streams,
    )


def media_profile(value: MediaIdentity) -> MediaProfile:
    return MediaProfile(
        codec_name=value.codec_name,
        pixel_format=value.pixel_format,
        width=value.width,
        height=value.height,
        average_frame_rate=value.average_frame_rate,
        nominal_frame_rate=value.nominal_frame_rate,
        audio_streams=value.audio_streams,
    )


def decide_media_mode(
    episodes: tuple[EpisodePlan, ...],
    *,
    skip: int,
    downsample: int,
    output_fps: Fraction,
) -> tuple[Literal["copy_all", "transcode_all"], MediaProfile]:
    if not episodes:
        raise ValueError("at least one episode is required")
    if skip < 0 or downsample <= 0 or output_fps <= 0:
        raise ValueError("invalid media selection arguments")

    first_source = episodes[0].source_media
    first_key = media_selection_key(first_source)
    copy_all = (
        skip == 0
        and downsample == 1
        and all(
            media_selection_key(episode.source_media) == first_key
            for episode in episodes
        )
        and all(
            parse_rate(episode.source_media.average_frame_rate) == output_fps
            and parse_rate(episode.source_media.nominal_frame_rate) == output_fps
            for episode in episodes
        )
        and all(
            episode.source_media.frame_count == len(episode.retained_indices)
            for episode in episodes
        )
    )
    if copy_all:
        return "copy_all", media_profile(first_source)

    requested_rate = str(output_fps)
    return "transcode_all", MediaProfile(
        codec_name="h264",
        pixel_format="yuv420p",
        width=640,
        height=360,
        average_frame_rate=requested_rate,
        nominal_frame_rate=requested_rate,
        audio_streams=(),
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

    input_roots = []
    for source_root in source_roots:
        source_root_fd = _open_validated_directory(source_root)
        try:
            source_root_metadata = _validate_descriptor_path(
                source_root_fd,
                source_root,
                expected_type="directory",
            )
            input_roots.append(
                SourceRootIdentity(
                    path=source_root,
                    st_dev=source_root_metadata.st_dev,
                    st_ino=source_root_metadata.st_ino,
                    st_uid=source_root_metadata.st_uid,
                    st_gid=source_root_metadata.st_gid,
                    st_mode=stat.S_IMODE(source_root_metadata.st_mode),
                )
            )
        finally:
            os.close(source_root_fd)

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
        parquet_payload, parquet_sha256 = _snapshot_regular_file(
            parquet_path,
            label="source Parquet",
        )
        schema = pq.read_schema(pa.BufferReader(parquet_payload))
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

        parquet_file = pq.ParquetFile(pa.BufferReader(parquet_payload))
        frame_count = parquet_file.metadata.num_rows
        task_table = pq.read_table(
            pa.BufferReader(parquet_payload), columns=["task_index"]
        )
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
                parquet_sha256=parquet_sha256,
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

    media_mode, output_media = decide_media_mode(
        tuple(plans),
        skip=skip,
        downsample=downsample,
        output_fps=output_fps,
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
        input_roots=tuple(input_roots),
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
    try:
        with np.errstate(over="ignore"):
            timestamp = np.asarray(timestamp_quotients, dtype=np.float32)
    except OverflowError as exc:
        raise ValueError("timestamp contains nonfinite values") from exc
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


def output_schema_sha256() -> str:
    identity = {
        "fields": [
            {
                "name": field.name,
                "nullable": field.nullable,
                "type": str(field.type),
            }
            for field in output_schema()
        ]
    }
    return sha256_bytes(canonical_json_bytes(identity))


def _atomic_write_new_bytes_at(
    path: Path,
    payload: bytes,
    mode: int,
    *,
    parent_fd: int,
) -> None:
    destination_name = _validated_basename(path, "destination")
    temporary_name = os.fsencode(f".{path.name}.tmp-{uuid.uuid4().hex}")
    payload_fd = None
    temporary_read_fd = None
    destination_fd = None
    temporary_identity: tuple[int, int] | None = None
    temporary_exists = False
    primary_error: BaseException | None = None
    try:
        payload_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            mode,
            dir_fd=parent_fd,
        )
        temporary_exists = True
        created = _validate_descriptor_entry(
            payload_fd,
            parent_fd,
            temporary_name,
            expected_type="file",
            require_single_link=True,
        )
        temporary_identity = (created.st_dev, created.st_ino)
        with os.fdopen(payload_fd, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        written = _validate_descriptor_entry(
            payload_fd,
            parent_fd,
            temporary_name,
            expected_type="file",
            require_single_link=True,
        )
        if (
            written.st_dev,
            written.st_ino,
            written.st_size,
        ) != (*temporary_identity, len(payload)):
            raise RuntimeError("temporary payload identity or size changed")
        expected_sha256 = sha256_bytes(payload)
        temporary_read_fd = os.open(
            temporary_name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        verified = _validate_descriptor_entry(
            temporary_read_fd,
            parent_fd,
            temporary_name,
            expected_type="file",
            require_single_link=True,
        )
        if (verified.st_dev, verified.st_ino) != temporary_identity:
            raise RuntimeError("temporary payload identity changed before verification")
        if _sha256_descriptor(temporary_read_fd) != expected_sha256:
            raise RuntimeError("temporary payload digest differs from requested bytes")
        _validate_descriptor_path(parent_fd, path.parent, expected_type="directory")
        os.link(
            temporary_name,
            destination_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        destination_fd = os.open(
            destination_name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        linked = _validate_descriptor_entry(
            destination_fd,
            parent_fd,
            destination_name,
            expected_type="file",
        )
        if (linked.st_dev, linked.st_ino) != temporary_identity:
            raise RuntimeError("published payload identity differs from temporary")
        if linked.st_nlink != 2 or linked.st_size != len(payload):
            raise RuntimeError("published payload has unsafe metadata")
        if _sha256_descriptor(destination_fd) != expected_sha256:
            raise RuntimeError("published payload digest differs from requested bytes")
        fsync_directory(path.parent, directory_fd=parent_fd)
        os.unlink(temporary_name, dir_fd=parent_fd)
        temporary_exists = False
        fsync_directory(path.parent, directory_fd=parent_fd)
        final = _validate_descriptor_entry(
            destination_fd,
            parent_fd,
            destination_name,
            expected_type="file",
            require_single_link=True,
        )
        if (
            final.st_dev,
            final.st_ino,
            final.st_size,
        ) != (*temporary_identity, len(payload)):
            raise RuntimeError("final payload identity or size changed")
        if _sha256_descriptor(destination_fd) != expected_sha256:
            raise RuntimeError("final payload digest differs from requested bytes")
        _validate_descriptor_path(parent_fd, path.parent, expected_type="directory")
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_error = None
        if temporary_exists and temporary_identity is not None:
            try:
                visible = os.stat(
                    temporary_name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if (visible.st_dev, visible.st_ino) != temporary_identity:
                    raise RuntimeError("temporary payload identity changed")
                os.unlink(temporary_name, dir_fd=parent_fd)
                fsync_directory(path.parent, directory_fd=parent_fd)
            except FileNotFoundError:
                pass
            except BaseException as exc:
                cleanup_error = exc
        if destination_fd is not None:
            os.close(destination_fd)
        if temporary_read_fd is not None:
            os.close(temporary_read_fd)
        if payload_fd is not None:
            os.close(payload_fd)
        if cleanup_error is not None and primary_error is None:
            raise cleanup_error


def atomic_write_new_bytes(path: Path, payload: bytes, mode: int = 0o644) -> None:
    parent_fd = _open_validated_directory(path.parent)
    try:
        _atomic_write_new_bytes_at(
            path,
            payload,
            mode,
            parent_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)


def _canonical_jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def _media_mapping(value: MediaIdentity | MediaProfile) -> dict[str, object]:
    mapping = asdict(value)
    mapping["audio_streams"] = list(mapping["audio_streams"])
    return mapping


def _write_parquet_new(
    path: Path,
    table: pa.Table,
    *,
    parent_fd: int | None = None,
) -> None:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    payload = sink.getvalue().to_pybytes()
    if parent_fd is None:
        atomic_write_new_bytes(path, payload)
    else:
        _atomic_write_new_bytes_at(path, payload, 0o644, parent_fd=parent_fd)


def _read_verified_source_parquet(
    episode: EpisodePlan,
) -> tuple[pa.Table, str]:
    fd = os.open(
        episode.parquet_path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        before = _validate_descriptor_path(
            fd,
            episode.parquet_path,
            expected_type="file",
            require_single_link=True,
        )
        digest = hashlib.sha256()
        chunks = []
        os.lseek(fd, 0, os.SEEK_SET)
        while block := os.read(fd, 1024 * 1024):
            digest.update(block)
            chunks.append(block)
        payload = b"".join(chunks)
        consumed_sha256 = digest.hexdigest()
        if len(payload) != before.st_size:
            raise RuntimeError("source Parquet size changed while being read")
        if consumed_sha256 != episode.parquet_sha256:
            raise RuntimeError("source Parquet SHA-256 differs from preflight identity")
        after_read = _validate_descriptor_path(
            fd,
            episode.parquet_path,
            expected_type="file",
            require_single_link=True,
        )
        if (after_read.st_dev, after_read.st_ino, after_read.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise RuntimeError("source Parquet identity changed while being read")

        table = pq.read_table(pa.BufferReader(payload))

        after_decode = _validate_descriptor_path(
            fd,
            episode.parquet_path,
            expected_type="file",
            require_single_link=True,
        )
        if (after_decode.st_dev, after_decode.st_ino, after_decode.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise RuntimeError("source Parquet identity changed while being decoded")
        if _sha256_descriptor(fd) != consumed_sha256:
            raise RuntimeError("source Parquet changed while being decoded")
        return table, consumed_sha256
    finally:
        os.close(fd)


def _feature_metadata() -> dict[str, dict[str, object]]:
    return {
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


def _in_progress_status_bytes(staging_metadata: os.stat_result) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": 1,
            "staging": {
                "st_dev": staging_metadata.st_dev,
                "st_ino": staging_metadata.st_ino,
            },
            "state": "in_progress",
        }
    )


def write_in_progress_status(staging: Path, plan: ConversionPlan) -> None:
    del plan
    staging_fd = _open_validated_directory(staging)
    try:
        staging_metadata = _validate_descriptor_path(
            staging_fd,
            staging,
            expected_type="directory",
        )
        _atomic_write_new_bytes_at(
            staging / "CONVERSION_STATUS.json",
            _in_progress_status_bytes(staging_metadata),
            0o644,
            parent_fd=staging_fd,
        )
        _validate_descriptor_path(
            staging_fd,
            staging,
            expected_type="directory",
        )
    finally:
        os.close(staging_fd)


def _validate_in_progress_status_at(staging: Path, staging_fd: int) -> None:
    staging_metadata = _validate_descriptor_path(
        staging_fd,
        staging,
        expected_type="directory",
    )
    if os.listdir(staging_fd) != ["CONVERSION_STATUS.json"]:
        raise FileExistsError("existing staging tree is not freshly initialized")
    status_fd = os.open(
        b"CONVERSION_STATUS.json",
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=staging_fd,
    )
    try:
        _validate_descriptor_entry(
            status_fd,
            staging_fd,
            b"CONVERSION_STATUS.json",
            expected_type="file",
            require_single_link=True,
        )
        os.lseek(status_fd, 0, os.SEEK_SET)
        payload = b"".join(iter(lambda: os.read(status_fd, 4096), b""))
        if payload != _in_progress_status_bytes(staging_metadata):
            raise RuntimeError("staging identity or in_progress status differs")
    finally:
        os.close(status_fd)
    _validate_descriptor_path(
        staging_fd,
        staging,
        expected_type="directory",
    )


def _validate_in_progress_status(staging: Path) -> None:
    staging_fd = _open_validated_directory(staging)
    try:
        _validate_in_progress_status_at(staging, staging_fd)
    finally:
        os.close(staging_fd)


def _create_pinned_directory(
    parent_fd: int,
    parent_path: Path,
    name: str,
    *,
    mode: int = 0o755,
) -> int:
    child_path = parent_path / name
    child_name = _validated_basename(child_path, "directory")
    _validate_descriptor_path(parent_fd, parent_path, expected_type="directory")
    os.mkdir(child_name, mode=mode, dir_fd=parent_fd)
    child_fd = None
    try:
        child_fd = os.open(
            child_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        _validate_descriptor_entry(
            child_fd,
            parent_fd,
            child_name,
            expected_type="directory",
        )
        _validate_descriptor_path(child_fd, child_path, expected_type="directory")
        fsync_directory(parent_path, directory_fd=parent_fd)
        _validate_descriptor_path(parent_fd, parent_path, expected_type="directory")
        return child_fd
    except BaseException:
        if child_fd is not None:
            os.close(child_fd)
        raise


def _validate_input_root(identity: SourceRootIdentity) -> None:
    fd = _open_validated_directory(identity.path)
    try:
        metadata = _validate_descriptor_path(
            fd,
            identity.path,
            expected_type="directory",
        )
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_uid,
            metadata.st_gid,
            stat.S_IMODE(metadata.st_mode),
        ) != (
            identity.st_dev,
            identity.st_ino,
            identity.st_uid,
            identity.st_gid,
            identity.st_mode,
        ):
            raise RuntimeError("input root identity changed after preflight")
    finally:
        os.close(fd)


def _prepare_staging_root(staging: Path, plan: ConversionPlan) -> None:
    try:
        staging.mkdir(mode=0o755)
    except FileExistsError:
        _validate_in_progress_status(staging)
        return
    write_in_progress_status(staging, plan)
    _validate_in_progress_status(staging)


class _PinnedStagingTree:
    def __init__(self, staging: Path) -> None:
        self.staging = staging
        self.data_path = staging / "data"
        self.video_path = staging / "videos"
        self.meta_path = staging / "meta"
        self._fds: list[int] = []
        self._chunks: dict[int, tuple[Path, int, Path, int, Path, int]] = {}
        try:
            self.staging_fd = self._keep(_open_validated_directory(staging))
            _validate_in_progress_status_at(staging, self.staging_fd)
            self.data_fd = self._keep(
                _create_pinned_directory(self.staging_fd, staging, "data")
            )
            self.video_fd = self._keep(
                _create_pinned_directory(self.staging_fd, staging, "videos")
            )
            self.meta_fd = self._keep(
                _create_pinned_directory(self.staging_fd, staging, "meta")
            )
            self.validate_base()
        except BaseException:
            self.close()
            raise

    def _keep(self, fd: int) -> int:
        self._fds.append(fd)
        return fd

    def validate_base(self) -> None:
        _validate_descriptor_path(
            self.staging_fd,
            self.staging,
            expected_type="directory",
        )
        _validate_descriptor_entry(
            self.data_fd,
            self.staging_fd,
            b"data",
            expected_type="directory",
        )
        _validate_descriptor_path(
            self.data_fd,
            self.data_path,
            expected_type="directory",
        )
        _validate_descriptor_entry(
            self.video_fd,
            self.staging_fd,
            b"videos",
            expected_type="directory",
        )
        _validate_descriptor_path(
            self.video_fd,
            self.video_path,
            expected_type="directory",
        )
        _validate_descriptor_entry(
            self.meta_fd,
            self.staging_fd,
            b"meta",
            expected_type="directory",
        )
        _validate_descriptor_path(
            self.meta_fd,
            self.meta_path,
            expected_type="directory",
        )

    def episode_directories(self, chunk_index: int) -> tuple[Path, int, Path, int]:
        existing = self._chunks.get(chunk_index)
        if existing is not None:
            self._validate_chunk(chunk_index, existing)
            return existing[0], existing[1], existing[4], existing[5]

        self.validate_base()
        chunk_name = f"chunk-{chunk_index:03d}"
        data_chunk_path = self.data_path / chunk_name
        data_chunk_fd = self._keep(
            _create_pinned_directory(self.data_fd, self.data_path, chunk_name)
        )
        video_chunk_path = self.video_path / chunk_name
        video_chunk_fd = self._keep(
            _create_pinned_directory(self.video_fd, self.video_path, chunk_name)
        )
        egocentric_path = video_chunk_path / "egocentric"
        egocentric_fd = self._keep(
            _create_pinned_directory(
                video_chunk_fd,
                video_chunk_path,
                "egocentric",
            )
        )
        result = (
            data_chunk_path,
            data_chunk_fd,
            video_chunk_path,
            video_chunk_fd,
            egocentric_path,
            egocentric_fd,
        )
        self._chunks[chunk_index] = result
        self._validate_chunk(chunk_index, result)
        return data_chunk_path, data_chunk_fd, egocentric_path, egocentric_fd

    def _validate_chunk(
        self,
        chunk_index: int,
        directories: tuple[Path, int, Path, int, Path, int],
    ) -> None:
        self.validate_base()
        (
            data_chunk_path,
            data_chunk_fd,
            video_chunk_path,
            video_chunk_fd,
            egocentric_path,
            egocentric_fd,
        ) = directories
        chunk_name = f"chunk-{chunk_index:03d}"
        _validate_descriptor_entry(
            data_chunk_fd,
            self.data_fd,
            chunk_name,
            expected_type="directory",
        )
        _validate_descriptor_path(
            data_chunk_fd,
            data_chunk_path,
            expected_type="directory",
        )
        _validate_descriptor_entry(
            video_chunk_fd,
            self.video_fd,
            chunk_name,
            expected_type="directory",
        )
        _validate_descriptor_path(
            video_chunk_fd,
            video_chunk_path,
            expected_type="directory",
        )
        _validate_descriptor_entry(
            egocentric_fd,
            video_chunk_fd,
            b"egocentric",
            expected_type="directory",
        )
        _validate_descriptor_path(
            egocentric_fd,
            egocentric_path,
            expected_type="directory",
        )

    def close(self) -> None:
        while self._fds:
            os.close(self._fds.pop())


# default history command for the initial state
initial_command = np.array([0, 0, 0, 0, 0, 0, 0.74, 0.74, 0.74], dtype=np.float32)
default_fps = 50


def _validate_output_media(
    identity: MediaIdentity,
    *,
    output_profile: MediaProfile,
    retained_count: int,
) -> None:
    if identity.frame_count != retained_count:
        raise RuntimeError("output media frame count differs from retained rows")
    if media_profile(identity) != output_profile:
        raise RuntimeError("output media profile differs from dataset profile")


def _select_filter(retained_indices: np.ndarray, output_fps: Fraction) -> str:
    indices = np.asarray(retained_indices)
    if (
        indices.ndim != 1
        or indices.size == 0
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0)
        or np.any(indices[1:] <= indices[:-1])
    ):
        raise ValueError("retained indices must be nonempty increasing integers")
    selected = "+".join(f"eq(n\\,{int(index)})" for index in indices)
    return f"select={selected},scale=640:360:flags=lanczos,setpts=N/({output_fps}*TB)"


def write_episode_video(
    *,
    episode: EpisodePlan,
    destination: Path,
    media_mode: Literal["copy_all", "transcode_all"],
    output_fps: Fraction,
    output_profile: MediaProfile,
    before_media_publish: Callable[[], None] | None = None,
    return_artifact: bool = False,
    destination_parent_fd: int | None = None,
) -> MediaIdentity | WrittenMedia:
    if output_fps <= 0:
        raise ValueError("output_fps must be positive")
    retained_count = len(episode.retained_indices)
    if retained_count == 0:
        raise ValueError("an episode video must contain at least one retained frame")

    if media_mode == "copy_all":
        if (
            output_profile != media_profile(episode.source_media)
            or episode.source_media.frame_count != retained_count
        ):
            raise ValueError(
                "copy_all episode does not match the dataset media profile"
            )
        parent_fd = (
            _open_validated_directory(destination.parent)
            if destination_parent_fd is None
            else os.dup(destination_parent_fd)
        )
        destination_fd = None
        try:
            _validate_descriptor_path(
                parent_fd,
                destination.parent,
                expected_type="directory",
            )
            destination_name = _validated_basename(destination, "destination")
            copied_sha256, destination_fd = copy_file_exclusive(
                episode.video_path,
                parent_fd,
                destination_name,
                expected_sha256=episode.video_sha256,
            )
            if copied_sha256 != episode.video_sha256:
                raise RuntimeError("copied destination SHA-256 differs from source")
            pinned_destination_path = Path(
                f"/proc/self/fd/{parent_fd}/{destination.name}"
            )
            output_identity = probe_media(
                pinned_destination_path,
                pass_fds=(parent_fd,),
            )
            final_metadata = _validate_descriptor_entry(
                destination_fd,
                parent_fd,
                destination_name,
                expected_type="file",
                require_single_link=True,
            )
            if _sha256_descriptor(destination_fd) != episode.video_sha256:
                raise RuntimeError("copied destination changed while being probed")
            _validate_output_media(
                output_identity,
                output_profile=output_profile,
                retained_count=retained_count,
            )
            fsync_directory(destination.parent, directory_fd=parent_fd)
            _validate_descriptor_path(
                parent_fd, destination.parent, expected_type="directory"
            )
            final_identity = probe_media(
                Path(f"/proc/self/fd/{destination_fd}"),
                pass_fds=(destination_fd,),
            )
            _validate_output_media(
                final_identity,
                output_profile=output_profile,
                retained_count=retained_count,
            )
            _validate_descriptor_path(
                parent_fd, destination.parent, expected_type="directory"
            )
            _validate_descriptor_entry(
                destination_fd,
                parent_fd,
                destination_name,
                expected_type="file",
                require_single_link=True,
            )
            final_sha256 = _sha256_descriptor(destination_fd)
            if final_sha256 != episode.video_sha256:
                raise RuntimeError("copied destination changed before completion")
            artifact = WrittenMedia(
                identity=final_identity,
                sha256=final_sha256,
                st_dev=final_metadata.st_dev,
                st_ino=final_metadata.st_ino,
                st_size=final_metadata.st_size,
            )
            return artifact if return_artifact else artifact.identity
        finally:
            if destination_fd is not None:
                os.close(destination_fd)
            os.close(parent_fd)
    if media_mode != "transcode_all":
        raise ValueError(f"unsupported media mode: {media_mode!r}")

    canonical_profile = MediaProfile(
        codec_name="h264",
        pixel_format="yuv420p",
        width=640,
        height=360,
        average_frame_rate=str(output_fps),
        nominal_frame_rate=str(output_fps),
        audio_streams=(),
    )
    if output_profile != canonical_profile:
        raise ValueError("transcode_all requires the canonical output media profile")

    source_fd = os.open(episode.video_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    snapshot_fd = None
    parent_fd = None
    private_fd = None
    artifact_fd = None
    try:
        _validate_descriptor_path(source_fd, episode.video_path, expected_type="file")
        if _sha256_descriptor(source_fd) != episode.video_sha256:
            raise RuntimeError("source video SHA-256 differs from preflight identity")
        pinned_source_path = Path(f"/proc/self/fd/{source_fd}")
        source_media = probe_media(pinned_source_path, pass_fds=(source_fd,))
        if source_media != episode.source_media:
            raise RuntimeError("source video media differs from preflight identity")
        _validate_descriptor_path(source_fd, episode.video_path, expected_type="file")
        if _sha256_descriptor(source_fd) != episode.video_sha256:
            raise RuntimeError("source video changed after media validation")
        snapshot_fd = _create_sealed_snapshot(
            source_fd,
            expected_sha256=episode.video_sha256,
        )
        pinned_snapshot_path = Path(f"/proc/self/fd/{snapshot_fd}")

        parent_fd = (
            _open_validated_directory(destination.parent)
            if destination_parent_fd is None
            else os.dup(destination_parent_fd)
        )
        _validate_descriptor_path(
            parent_fd,
            destination.parent,
            expected_type="directory",
        )
        destination_name = _validated_basename(destination, "destination")
        private_name = f".{destination.name}.media-{uuid.uuid4().hex}"
        os.mkdir(private_name, mode=0o700, dir_fd=parent_fd)
        private_fd = os.open(
            private_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        private_metadata = _validate_descriptor_entry(
            private_fd,
            parent_fd,
            private_name,
            expected_type="directory",
        )
        if stat.S_IMODE(private_metadata.st_mode) != 0o700:
            raise RuntimeError("private media directory has unsafe metadata")
        private_directory = destination.parent / private_name
        fsync_directory(destination.parent, directory_fd=parent_fd)
        pinned_artifact_path = Path(f"/proc/self/fd/{private_fd}/artifact.mp4")
        command = [
            "ffmpeg",
            "-n",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(pinned_snapshot_path),
            "-vf",
            _select_filter(episode.retained_indices, output_fps),
            "-vsync",
            "cfr",
            "-r",
            str(output_fps),
            "-frames:v",
            str(retained_count),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(pinned_artifact_path),
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                pass_fds=(snapshot_fd, private_fd),
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required but was not found in PATH") from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for source video: {episode.video_path} "
                f"(exit code {result.returncode})"
            )
        _validate_descriptor_path(source_fd, episode.video_path, expected_type="file")
        if _sha256_descriptor(source_fd) != episode.video_sha256:
            raise RuntimeError("source video changed during transcoding")
        _verify_snapshot_seals(snapshot_fd)
        if _sha256_descriptor(snapshot_fd) != episode.video_sha256:
            raise RuntimeError("sealed source snapshot changed during transcoding")
        artifact_fd = os.open(
            "artifact.mp4",
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=private_fd,
        )
        artifact_metadata = _validate_descriptor_entry(
            artifact_fd,
            private_fd,
            "artifact.mp4",
            expected_type="file",
            require_single_link=True,
        )
        artifact_identity = (artifact_metadata.st_dev, artifact_metadata.st_ino)
        output_identity = probe_media(
            pinned_artifact_path,
            pass_fds=(private_fd,),
        )
        _validate_descriptor_entry(
            artifact_fd,
            private_fd,
            "artifact.mp4",
            expected_type="file",
            require_single_link=True,
        )
        _validate_output_media(
            output_identity,
            output_profile=output_profile,
            retained_count=retained_count,
        )
        os.fsync(artifact_fd)
        _validate_descriptor_entry(
            artifact_fd,
            private_fd,
            "artifact.mp4",
            expected_type="file",
            require_single_link=True,
        )
        artifact_sha256 = _sha256_descriptor(artifact_fd)
        fsync_directory(private_directory, directory_fd=private_fd)
        if before_media_publish is not None:
            before_media_publish()
        _validate_descriptor_path(
            parent_fd, destination.parent, expected_type="directory"
        )
        private_metadata = _validate_descriptor_entry(
            private_fd,
            parent_fd,
            private_name,
            expected_type="directory",
        )
        if stat.S_IMODE(private_metadata.st_mode) != 0o700:
            raise RuntimeError("private media directory mode changed")
        _validate_descriptor_path(
            private_fd, private_directory, expected_type="directory"
        )
        _validate_descriptor_entry(
            artifact_fd,
            private_fd,
            "artifact.mp4",
            expected_type="file",
            require_single_link=True,
        )
        if _sha256_descriptor(artifact_fd) != artifact_sha256:
            raise RuntimeError("validated media artifact changed before publication")
        _rename_noreplace_at(
            private_fd,
            b"artifact.mp4",
            parent_fd,
            destination_name,
            expected_identity=artifact_identity,
        )
        final_metadata = _validate_descriptor_entry(
            artifact_fd,
            parent_fd,
            destination_name,
            expected_type="file",
            require_single_link=True,
        )
        if _sha256_descriptor(artifact_fd) != artifact_sha256:
            raise RuntimeError("published media differs from validated artifact")
        fsync_directory(destination.parent, directory_fd=parent_fd)
        _validate_descriptor_entry(
            artifact_fd,
            parent_fd,
            destination_name,
            expected_type="file",
            require_single_link=True,
        )
        if _sha256_descriptor(artifact_fd) != artifact_sha256:
            raise RuntimeError("published media changed after parent fsync")
        os.rmdir(private_name, dir_fd=parent_fd)
        fsync_directory(destination.parent, directory_fd=parent_fd)
        final_identity = probe_media(
            Path(f"/proc/self/fd/{artifact_fd}"),
            pass_fds=(artifact_fd,),
        )
        _validate_output_media(
            final_identity,
            output_profile=output_profile,
            retained_count=retained_count,
        )
        _validate_descriptor_entry(
            artifact_fd,
            parent_fd,
            destination_name,
            expected_type="file",
            require_single_link=True,
        )
        final_sha256 = _sha256_descriptor(artifact_fd)
        if final_sha256 != artifact_sha256:
            raise RuntimeError("published media changed before completion")
        artifact = WrittenMedia(
            identity=final_identity,
            sha256=final_sha256,
            st_dev=final_metadata.st_dev,
            st_ino=final_metadata.st_ino,
            st_size=final_metadata.st_size,
        )
        return artifact if return_artifact else artifact.identity
    finally:
        if artifact_fd is not None:
            os.close(artifact_fd)
        if private_fd is not None:
            os.close(private_fd)
        if parent_fd is not None:
            os.close(parent_fd)
        if snapshot_fd is not None:
            os.close(snapshot_fd)
        os.close(source_fd)


def _verify_written_media(
    path: Path,
    expected: WrittenMedia,
    *,
    output_profile: MediaProfile,
    retained_count: int,
    parent_fd: int | None = None,
) -> WrittenMedia:
    owned_parent_fd = (
        _open_validated_directory(path.parent)
        if parent_fd is None
        else os.dup(parent_fd)
    )
    name = _validated_basename(path, "output video")
    fd = None
    try:
        _validate_descriptor_path(
            owned_parent_fd,
            path.parent,
            expected_type="directory",
        )
        fd = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=owned_parent_fd,
        )
        before = _validate_descriptor_entry(
            fd,
            owned_parent_fd,
            name,
            expected_type="file",
            require_single_link=True,
        )
        if (before.st_dev, before.st_ino, before.st_size) != (
            expected.st_dev,
            expected.st_ino,
            expected.st_size,
        ):
            raise RuntimeError("output video identity changed after writing")
        before_sha256 = _sha256_descriptor(fd)
        if before_sha256 != expected.sha256:
            raise RuntimeError("output video SHA-256 changed after writing")
        identity = probe_media(
            Path(f"/proc/self/fd/{fd}"),
            pass_fds=(fd,),
        )
        _validate_output_media(
            identity,
            output_profile=output_profile,
            retained_count=retained_count,
        )
        if identity != expected.identity:
            raise RuntimeError("output video media identity changed after writing")
        after = _validate_descriptor_entry(
            fd,
            owned_parent_fd,
            name,
            expected_type="file",
            require_single_link=True,
        )
        if (after.st_dev, after.st_ino, after.st_size) != (
            expected.st_dev,
            expected.st_ino,
            expected.st_size,
        ):
            raise RuntimeError("output video identity changed during verification")
        after_sha256 = _sha256_descriptor(fd)
        if after_sha256 != before_sha256:
            raise RuntimeError("output video SHA-256 changed during verification")
        _validate_descriptor_path(
            owned_parent_fd,
            path.parent,
            expected_type="directory",
        )
        return WrittenMedia(
            identity=identity,
            sha256=after_sha256,
            st_dev=after.st_dev,
            st_ino=after.st_ino,
            st_size=after.st_size,
        )
    finally:
        if fd is not None:
            os.close(fd)
        os.close(owned_parent_fd)


def generate_staged_dataset(plan: ConversionPlan, staging: Path) -> None:
    _prepare_staging_root(staging, plan)
    tree = _PinnedStagingTree(staging)
    try:
        _generate_staged_dataset(plan, tree)
    finally:
        tree.close()


def _generate_staged_dataset(plan: ConversionPlan, tree: _PinnedStagingTree) -> None:
    meta_root = tree.meta_path

    tasks = [
        {
            "task_index": task["task_index"],
            "task": task["task"],
            "category": "",
            "description": task["task"],
        }
        for task in plan.tasks
    ]
    task_by_index = {task["task_index"]: task for task in tasks}
    if len(task_by_index) != len(tasks):
        raise ValueError("output task indices must be unique")

    episode_rows = []
    episode_stats_rows = []
    episode_provenance_digests = []
    global_values: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "states",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
            "next.done",
        )
    }
    global_offset = 0
    output_video_count = 0

    input_root_by_path = {identity.path: identity for identity in plan.input_roots}
    if len(input_root_by_path) != len(plan.input_roots):
        raise ValueError("input root identities must have unique paths")
    for input_root in plan.input_roots:
        _validate_input_root(input_root)

    for episode in plan.episodes:
        input_root = input_root_by_path.get(episode.source_root)
        if input_root is None:
            raise ValueError("episode source root is absent from the preflight plan")
        _validate_input_root(input_root)
        table, consumed_parquet_sha256 = _read_verified_source_parquet(episode)
        qpos = np.asarray(table["observation.joint_qpos"].to_pylist(), dtype=np.float32)
        command = np.asarray(
            table["observation.amo_policy_command"].to_pylist(), dtype=np.float32
        )
        target_yaw = np.asarray(
            table["observation.amo_policy_target_yaw"], dtype=np.float32
        )
        turning = np.asarray(
            table["observation.amo_policy_turning_flag"], dtype=np.float32
        )
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        history = np.concatenate([initial_command[None], command[:-1]], axis=0)
        states, actions = build_vectors(
            qpos, command, history, action, target_yaw, turning
        )
        hand, arm, leg, torso, height = build_proprio_obs(qpos, history)
        indices = episode.retained_indices
        selected = {
            "states": states[indices],
            "action": actions[indices],
            "observation.hand_joints": hand[indices],
            "observation.arm_joints": arm[indices],
            "observation.leg_joints": leg[indices],
            "observation.prev_torso_rpy": torso[indices],
            "observation.prev_height": height[indices],
        }
        output_table = build_output_table(
            vectors=selected,
            output_episode_index=episode.output_episode_index,
            global_offset=global_offset,
            output_task_index=episode.output_task_index,
            output_fps=plan.output_fps,
        )
        retained_count = len(indices)
        if output_table.num_rows != retained_count:
            raise RuntimeError("output row count differs from retained indices")

        chunk_index = episode.output_episode_index // plan.chunks_size
        (
            data_directory,
            data_directory_fd,
            video_directory,
            video_directory_fd,
        ) = tree.episode_directories(chunk_index)
        data_path = (
            data_directory / f"episode_{episode.output_episode_index:06d}.parquet"
        )
        _write_parquet_new(
            data_path,
            output_table,
            parent_fd=data_directory_fd,
        )

        video_path = video_directory / f"episode_{episode.output_episode_index:06d}.mp4"
        written_media = write_episode_video(
            episode=episode,
            destination=video_path,
            media_mode=plan.media_mode,
            output_fps=plan.output_fps,
            output_profile=plan.output_media,
            return_artifact=True,
            destination_parent_fd=video_directory_fd,
        )
        if not isinstance(written_media, WrittenMedia):
            raise RuntimeError("video writer did not return descriptor-bound evidence")
        verified_media = _verify_written_media(
            video_path,
            written_media,
            output_profile=plan.output_media,
            retained_count=retained_count,
            parent_fd=video_directory_fd,
        )
        _validate_input_root(input_root)
        output_media = verified_media.identity
        output_video_sha256 = verified_media.sha256
        output_video_count += 1

        episode_provenance = {
            "source_episode_index": episode.source_episode_index,
            "source_parquet_sha256": consumed_parquet_sha256,
            "source_video_sha256": episode.video_sha256,
            "requested_video_key": plan.video_key,
            "requested_output_fps": str(plan.output_fps),
            "skip": plan.skip,
            "downsample": plan.downsample,
            "retained_count": retained_count,
            "source_media": _media_mapping(episode.source_media),
            "output_video_sha256": output_video_sha256,
            "output_media": _media_mapping(output_media),
            "converter_commit": plan.converter.commit,
            "converter_script_sha256": plan.converter.script_sha256,
        }
        episode_provenance_digests.append(
            sha256_bytes(canonical_json_bytes(episode_provenance))
        )
        instruction = task_by_index.get(episode.output_task_index)
        if instruction is None or instruction["task"] != episode.task_text:
            raise ValueError("episode task mapping does not match emitted task")
        episode_rows.append(
            {
                "episode_index": episode.output_episode_index,
                "tasks": [episode.output_task_index],
                "length": retained_count,
                "dataset_from_index": global_offset,
                "dataset_to_index": global_offset + retained_count - 1,
                "robot_type": "g1",
                "instruction": instruction,
                "environment_config": episode.environment_config,
                "conversion_provenance": episode_provenance,
            }
        )

        table_values = {
            name: np.asarray(output_table[name].to_pylist()) for name in global_values
        }
        table_values["states"] = selected["states"]
        table_values["action"] = selected["action"]
        episode_stats_rows.append(
            {
                "episode_index": episode.output_episode_index,
                "stats": {
                    "action": stats_block(selected["action"]),
                    "timestamp": stats_block(table_values["timestamp"]),
                },
            }
        )
        for name, values in table_values.items():
            if len(values) != retained_count:
                raise RuntimeError(
                    f"global accumulator {name!r} has incorrect cardinality"
                )
            global_values[name].append(values)
        global_offset += retained_count

    total_frames = global_offset
    if output_video_count != len(plan.episodes):
        raise RuntimeError("output video count differs from episode count")
    global_stats = {
        name: stats_block(np.concatenate(parts, axis=0))
        for name, parts in global_values.items()
    }
    if any(block["count"] != [total_frames] for block in global_stats.values()):
        raise RuntimeError("global statistics count differs from output rows")

    profile = plan.output_media
    image_feature = {
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
    info = {
        "codebase_version": "v2.1",
        "robot_type": "g1",
        "total_episodes": len(plan.episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": output_video_count,
        "total_chunks": math.ceil(len(plan.episodes) / plan.chunks_size),
        "chunks_size": plan.chunks_size,
        "fps": float(plan.output_fps),
        "data_path": (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/egocentric/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.images.egocentric": image_feature,
            **_feature_metadata(),
        },
    }

    input_roots = [
        {
            "path": str(identity.path),
            "st_dev": identity.st_dev,
            "st_ino": identity.st_ino,
            "st_uid": identity.st_uid,
            "st_gid": identity.st_gid,
            "st_mode": identity.st_mode,
        }
        for identity in plan.input_roots
    ]
    dataset_provenance = {
        "converter": asdict(plan.converter),
        "invocation": {
            "output_path": str(plan.output_path),
            "skip": plan.skip,
            "downsample": plan.downsample,
            "output_fps": str(plan.output_fps),
            "video_key": plan.video_key,
            "chunks_size": plan.chunks_size,
            "total_episodes": len(plan.episodes),
        },
        "input_roots": input_roots,
        "media_mode": plan.media_mode,
        "output_media": _media_mapping(plan.output_media),
        "episode_provenance_sha256": episode_provenance_digests,
        "output_schema_sha256": output_schema_sha256(),
    }

    metadata_payloads = {
        "tasks.jsonl": _canonical_jsonl_bytes(tasks),
        "episodes.jsonl": _canonical_jsonl_bytes(episode_rows),
        "episodes_stats.jsonl": _canonical_jsonl_bytes(episode_stats_rows),
        "info.json": canonical_json_bytes(info),
        "relative_stats.json": canonical_json_bytes({}),
        "lang_map.json": canonical_json_bytes({}),
        "modality.json": canonical_json_bytes(modality_dict()),
        "conversion_provenance.json": canonical_json_bytes(dataset_provenance),
    }
    stats_payload = canonical_json_bytes(global_stats)
    metadata_payloads["stats.json"] = stats_payload
    metadata_payloads["stats_psi0.json"] = stats_payload
    for name, payload in metadata_payloads.items():
        tree.validate_base()
        _atomic_write_new_bytes_at(
            meta_root / name,
            payload,
            0o644,
            parent_fd=tree.meta_fd,
        )


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
