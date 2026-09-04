#!/usr/bin/env python3
# ruff: noqa: E402
import sys as _startup_sys


def _sanitize_entrypoint_sys_path() -> None:
    if __name__ != "__main__":
        return
    trusted_roots = tuple(
        root.rstrip("/\\")
        for root in {_startup_sys.prefix, _startup_sys.base_prefix}
        if root
    )

    def trusted(entry: object) -> bool:
        if not isinstance(entry, str) or not entry:
            return False
        normalized = entry.replace("\\", "/").rstrip("/")
        for root in trusted_roots:
            normalized_root = root.replace("\\", "/")
            if normalized == normalized_root:
                return True
            prefix = normalized_root + "/"
            if normalized.startswith(prefix):
                relative_parts = normalized[len(prefix) :].split("/")
                return all(part not in {"", ".", ".."} for part in relative_parts)
        return False

    _startup_sys.path[:] = [entry for entry in _startup_sys.path if trusted(entry)]


_sanitize_entrypoint_sys_path()
sys = _startup_sys

import argparse
import contextlib
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


def _bootstrap_direct_script_imports() -> None:
    if __name__ != "__main__":
        return
    scripts_root = Path(__file__).resolve().parent
    running_module = sys.modules[__name__]
    canonical_name = "scripts.postprocess_psi0"
    package = sys.modules.get("scripts")
    if __package__ in {None, ""}:
        if package is not None:
            raise RuntimeError("conflicting scripts package during direct execution")
        package = type(sys)("scripts")
        package.__package__ = "scripts"
        package.__path__ = [str(scripts_root)]
        sys.modules["scripts"] = package
    else:
        # Python resolves and may execute scripts/__init__.py before module code.
        # Therefore -m is supported only when that initial resolution produced
        # this repository's namespace package; direct-path execution has no such
        # pre-execution package boundary.
        if (
            __package__ != "scripts"
            or getattr(__spec__, "name", None) != canonical_name
            or package is None
            or getattr(package, "__file__", None) is not None
        ):
            raise RuntimeError("module execution requires a namespace scripts package")
        package.__path__ = [str(scripts_root)]
    imported_module = sys.modules.get("scripts.postprocess_psi0")
    if imported_module is not None and imported_module is not running_module:
        raise RuntimeError("conflicting converter module identity")
    sys.modules[canonical_name] = running_module
    package.postprocess_psi0 = running_module


_bootstrap_direct_script_imports()


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


@dataclass(frozen=True)
class PublicationResult:
    dataset_root: Path
    manifest_sha256: str
    complete_status_sha256: str
    state: Literal["published", "publication_uncertain"]


class PublicationUncertainError(RuntimeError):
    def __init__(self, publication: PublicationResult) -> None:
        super().__init__(
            "publication rename completed but parent durability is uncertain"
        )
        self.publication = publication


class PublishedBoundaryError(RuntimeError):
    def __init__(self, publication: PublicationResult) -> None:
        super().__init__("failure after durable publication")
        self.publication = publication


class PublicationStateError(RuntimeError):
    def __init__(
        self,
        state: Literal[
            "pre_completion_failed",
            "completion_uncertain_unpublished",
            "complete_unpublished",
        ],
    ) -> None:
        super().__init__(state)
        self.state = state


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


def _validate_conversion_lock_metadata(metadata: os.stat_result) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("conversion lock is not a regular file")
    if metadata.st_uid != os.getuid():
        raise RuntimeError("conversion lock has wrong owner")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise RuntimeError("conversion lock has wrong mode")
    if metadata.st_nlink != 1:
        raise RuntimeError("conversion lock has wrong link count")


def _validate_conversion_anchor_metadata(metadata: os.stat_result) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("conversion lock anchor is not a directory")
    if metadata.st_uid != os.getuid():
        raise RuntimeError("conversion lock anchor has wrong owner")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise RuntimeError("conversion lock anchor has wrong mode")


_INHERITED_CONVERSION_LOCK_FDS: set[int] = set()


@dataclass(frozen=True)
class _ConversionLockOwnership:
    pid: int
    lock_fd: int
    lock_identity: tuple[int, int]
    parent_fd: int
    parent_identity: tuple[int, int]


_OWNED_CONVERSION_LOCKS: dict[Path, _ConversionLockOwnership] = {}


def _conversion_lock_key(output: Path) -> Path:
    _validated_basename(output, "conversion output")
    return output.parent.resolve(strict=True) / output.name


def _close_inherited_conversion_lock_fds() -> None:
    for fd in tuple(_INHERITED_CONVERSION_LOCK_FDS):
        try:
            os.close(fd)
        except OSError:
            pass
    _INHERITED_CONVERSION_LOCK_FDS.clear()
    _OWNED_CONVERSION_LOCKS.clear()


os.register_at_fork(after_in_child=_close_inherited_conversion_lock_fds)


def assert_conversion_lock_held(output: Path) -> None:
    key = _conversion_lock_key(output)
    ownership = _OWNED_CONVERSION_LOCKS.get(key)
    if ownership is None or ownership.pid != os.getpid():
        raise RuntimeError("certification requires the active conversion lock")
    try:
        lock_metadata = os.fstat(ownership.lock_fd)
        parent_metadata = os.fstat(ownership.parent_fd)
    except OSError as exc:
        raise RuntimeError("conversion lock ownership descriptor is invalid") from exc
    if (lock_metadata.st_dev, lock_metadata.st_ino) != ownership.lock_identity or (
        parent_metadata.st_dev,
        parent_metadata.st_ino,
    ) != ownership.parent_identity:
        raise RuntimeError("conversion lock ownership identity changed")
    _validate_conversion_lock_metadata(lock_metadata)
    lock_name = _validated_basename(
        output.parent / f".{output.name}.conversion.lock", "conversion lock"
    )
    visible = _validate_descriptor_entry(
        ownership.lock_fd,
        ownership.parent_fd,
        lock_name,
        expected_type="file",
        require_single_link=True,
    )
    if (visible.st_dev, visible.st_ino) != ownership.lock_identity:
        raise RuntimeError("visible conversion lock identity changed")
    _validate_descriptor_path(
        ownership.parent_fd, output.parent, expected_type="directory"
    )


def _conversion_lock_ownership(output: Path) -> _ConversionLockOwnership:
    assert_conversion_lock_held(output)
    ownership = _OWNED_CONVERSION_LOCKS.get(_conversion_lock_key(output))
    if ownership is None or ownership.pid != os.getpid():
        raise RuntimeError("conversion lock ownership disappeared")
    return ownership


def _validate_owned_parent(ownership: _ConversionLockOwnership, output: Path) -> None:
    metadata = os.fstat(ownership.parent_fd)
    if (metadata.st_dev, metadata.st_ino) != ownership.parent_identity:
        raise RuntimeError("conversion parent descriptor identity changed")
    _validate_descriptor_path(
        ownership.parent_fd,
        output.parent,
        expected_type="directory",
    )


@contextlib.contextmanager
def conversion_lock(output: Path):
    owner_pid = os.getpid()
    lock_path = output.parent / f".{output.name}.conversion.lock"
    lock_name = _validated_basename(lock_path, "conversion lock")
    anchor_path = output.parent / f".{output.name}.conversion.lock-anchor"
    anchor_name = _validated_basename(anchor_path, "conversion lock anchor")
    parent_fd = _open_validated_directory(output.parent)
    _INHERITED_CONVERSION_LOCK_FDS.add(parent_fd)
    anchor_fd = None
    fd = None
    key = None
    ownership = None
    try:
        _validate_descriptor_path(parent_fd, output.parent, expected_type="directory")
        try:
            os.mkdir(anchor_name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        else:
            fsync_directory(output.parent, directory_fd=parent_fd)
        anchor_fd = os.open(
            anchor_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        _INHERITED_CONVERSION_LOCK_FDS.add(anchor_fd)
        anchor_metadata = _validate_descriptor_entry(
            anchor_fd,
            parent_fd,
            anchor_name,
            expected_type="directory",
        )
        _validate_conversion_anchor_metadata(anchor_metadata)
        _validate_descriptor_path(
            anchor_fd,
            anchor_path,
            expected_type="directory",
        )
        fcntl.flock(anchor_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked_anchor = _validate_descriptor_entry(
            anchor_fd,
            parent_fd,
            anchor_name,
            expected_type="directory",
        )
        _validate_conversion_anchor_metadata(locked_anchor)
        _validate_descriptor_path(
            anchor_fd,
            anchor_path,
            expected_type="directory",
        )
        fd = os.open(
            lock_name,
            os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        _INHERITED_CONVERSION_LOCK_FDS.add(fd)
        metadata = _validate_descriptor_entry(
            fd,
            parent_fd,
            lock_name,
            expected_type="file",
            require_single_link=True,
        )
        _validate_conversion_lock_metadata(metadata)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = _validate_descriptor_entry(
            fd,
            parent_fd,
            lock_name,
            expected_type="file",
            require_single_link=True,
        )
        _validate_conversion_lock_metadata(locked)
        _validate_descriptor_path(parent_fd, output.parent, expected_type="directory")
        key = _conversion_lock_key(output)
        if key in _OWNED_CONVERSION_LOCKS:
            raise RuntimeError("conversion lock is already registered")
        parent_metadata = os.fstat(parent_fd)
        ownership = _ConversionLockOwnership(
            pid=owner_pid,
            lock_fd=fd,
            lock_identity=(locked.st_dev, locked.st_ino),
            parent_fd=parent_fd,
            parent_identity=(parent_metadata.st_dev, parent_metadata.st_ino),
        )
        _OWNED_CONVERSION_LOCKS[key] = ownership
        try:
            yield fd
        finally:
            if os.getpid() == owner_pid:
                locked = _validate_descriptor_entry(
                    fd,
                    parent_fd,
                    lock_name,
                    expected_type="file",
                    require_single_link=True,
                )
                _validate_conversion_lock_metadata(locked)
                locked_anchor = _validate_descriptor_entry(
                    anchor_fd,
                    parent_fd,
                    anchor_name,
                    expected_type="directory",
                )
                _validate_conversion_anchor_metadata(locked_anchor)
                _validate_descriptor_path(
                    anchor_fd,
                    anchor_path,
                    expected_type="directory",
                )
                _validate_descriptor_path(
                    parent_fd,
                    output.parent,
                    expected_type="directory",
                )
    finally:
        if os.getpid() == owner_pid:
            if key is not None and _OWNED_CONVERSION_LOCKS.get(key) == ownership:
                del _OWNED_CONVERSION_LOCKS[key]
            if fd is not None:
                _INHERITED_CONVERSION_LOCK_FDS.discard(fd)
                os.close(fd)
            if anchor_fd is not None:
                _INHERITED_CONVERSION_LOCK_FDS.discard(anchor_fd)
                os.close(anchor_fd)
            _INHERITED_CONVERSION_LOCK_FDS.discard(parent_fd)
            os.close(parent_fd)


_MANIFEST_RELATIVE_PATH = "meta/conversion_manifest.json"
_STATUS_RELATIVE_PATH = "CONVERSION_STATUS.json"
_RESERVED_MANIFEST_PATHS = {
    _MANIFEST_RELATIVE_PATH,
    _STATUS_RELATIVE_PATH,
}
_PAYLOAD_ROOTS = ("data", "videos", "meta")
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _validate_manifest_member_metadata(metadata: object, relative_path: str) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"manifest member is not a regular file: {relative_path}")
    if metadata.st_nlink != 1:
        raise RuntimeError(f"manifest member has unsafe link count: {relative_path}")


@dataclass(frozen=True)
class _OpenManifestMember:
    relative_path: str
    fd: int
    st_dev: int
    st_ino: int
    st_size: int
    st_mtime_ns: int
    st_ctime_ns: int
    sha256: str


@dataclass(frozen=True)
class _OpenManifestDirectory:
    relative_path: str
    fd: int
    st_dev: int
    st_ino: int
    st_mtime_ns: int
    st_ctime_ns: int


@dataclass
class _StagedTreeCapture:
    files: dict[str, _OpenManifestMember]
    directories: dict[str, _OpenManifestDirectory]

    def close(self) -> None:
        while self.files:
            _, member = self.files.popitem()
            os.close(member.fd)
        while self.directories:
            _, directory = self.directories.popitem()
            os.close(directory.fd)


def _member_metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_manifest_directory_identity(
    relative_path: str,
    fd: int,
    metadata: os.stat_result,
) -> _OpenManifestDirectory:
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"manifest directory is not a directory: {relative_path}")
    return _OpenManifestDirectory(
        relative_path=relative_path,
        fd=fd,
        st_dev=metadata.st_dev,
        st_ino=metadata.st_ino,
        st_mtime_ns=metadata.st_mtime_ns,
        st_ctime_ns=metadata.st_ctime_ns,
    )


def _expected_directory_identity(
    directory: _OpenManifestDirectory,
) -> tuple[int, ...]:
    return (
        directory.st_dev,
        directory.st_ino,
        directory.st_mtime_ns,
        directory.st_ctime_ns,
    )


def _open_manifest_member(
    parent_fd: int,
    name: str,
    relative_path: str,
) -> _OpenManifestMember:
    path_metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    _validate_manifest_member_metadata(path_metadata, relative_path)
    fd = os.open(
        name,
        os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    try:
        before = _validate_descriptor_entry(
            fd,
            parent_fd,
            name,
            expected_type="file",
            require_single_link=True,
        )
        _validate_manifest_member_metadata(before, relative_path)
        digest = _sha256_descriptor(fd)
        after = _validate_descriptor_entry(
            fd,
            parent_fd,
            name,
            expected_type="file",
            require_single_link=True,
        )
        _validate_manifest_member_metadata(after, relative_path)
        if _member_metadata_identity(after) != _member_metadata_identity(before):
            raise RuntimeError(
                f"manifest member changed while hashing: {relative_path}"
            )
        if _sha256_descriptor(fd) != digest:
            raise RuntimeError(
                f"manifest member changed while hashing: {relative_path}"
            )
        return _OpenManifestMember(
            relative_path=relative_path,
            fd=fd,
            st_dev=after.st_dev,
            st_ino=after.st_ino,
            st_size=after.st_size,
            st_mtime_ns=after.st_mtime_ns,
            st_ctime_ns=after.st_ctime_ns,
            sha256=digest,
        )
    except BaseException:
        os.close(fd)
        raise


def _capture_manifest_directory(
    directory_fd: int,
    relative_directory: str,
    capture: _StagedTreeCapture,
) -> None:
    captured_directory = capture.directories[relative_directory]
    before = os.fstat(directory_fd)
    if _directory_metadata_identity(before) != _expected_directory_identity(
        captured_directory
    ):
        raise RuntimeError(
            f"manifest directory changed before scan: {relative_directory}"
        )
    for name in os.listdir(directory_fd):
        if not isinstance(name, str) or not name or name in (".", ".."):
            raise RuntimeError("manifest traversal encountered an unsafe name")
        relative_path = f"{relative_directory}/{name}"
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"manifest member is a symlink: {relative_path}")
        if stat.S_ISDIR(metadata.st_mode):
            if relative_path in capture.directories:
                raise RuntimeError(f"duplicate manifest directory: {relative_path}")
            child_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            try:
                child_before = _validate_descriptor_entry(
                    child_fd,
                    directory_fd,
                    name,
                    expected_type="directory",
                )
                capture.directories[relative_path] = _open_manifest_directory_identity(
                    relative_path,
                    child_fd,
                    child_before,
                )
            except BaseException:
                os.close(child_fd)
                raise
            _capture_manifest_directory(child_fd, relative_path, capture)
            child_after = _validate_descriptor_entry(
                child_fd,
                directory_fd,
                name,
                expected_type="directory",
            )
            if _directory_metadata_identity(
                child_after
            ) != _expected_directory_identity(capture.directories[relative_path]):
                raise RuntimeError(
                    f"manifest directory identity changed: {relative_path}"
                )
            continue
        if relative_path in capture.files:
            raise RuntimeError(f"duplicate manifest member path: {relative_path}")
        capture.files[relative_path] = _open_manifest_member(
            directory_fd,
            name,
            relative_path,
        )
    after = os.fstat(directory_fd)
    if _directory_metadata_identity(after) != _expected_directory_identity(
        captured_directory
    ):
        raise RuntimeError(
            f"manifest directory changed during scan: {relative_directory}"
        )


def _capture_staged_tree(
    staging: Path | None,
    staging_fd: int,
) -> _StagedTreeCapture:
    capture = _StagedTreeCapture(files={}, directories={})
    try:
        captured_root_fd = os.dup(staging_fd)
        root_metadata = os.fstat(captured_root_fd)
        capture.directories["."] = _open_manifest_directory_identity(
            ".",
            captured_root_fd,
            root_metadata,
        )
        root_names = set(os.listdir(captured_root_fd))
        expected_root_names = {*_PAYLOAD_ROOTS, _STATUS_RELATIVE_PATH}
        if root_names != expected_root_names:
            raise RuntimeError(
                "staged tree has files or directories outside payload layout"
            )
        for root_name in _PAYLOAD_ROOTS:
            root_fd = os.open(
                root_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=captured_root_fd,
            )
            try:
                root_before = _validate_descriptor_entry(
                    root_fd,
                    captured_root_fd,
                    root_name,
                    expected_type="directory",
                )
                capture.directories[root_name] = _open_manifest_directory_identity(
                    root_name,
                    root_fd,
                    root_before,
                )
            except BaseException:
                os.close(root_fd)
                raise
            _capture_manifest_directory(root_fd, root_name, capture)
            root_after = _validate_descriptor_entry(
                root_fd,
                captured_root_fd,
                root_name,
                expected_type="directory",
            )
            if _directory_metadata_identity(root_after) != _expected_directory_identity(
                capture.directories[root_name]
            ):
                raise RuntimeError(f"manifest directory identity changed: {root_name}")
        capture.files[_STATUS_RELATIVE_PATH] = _open_manifest_member(
            captured_root_fd,
            _STATUS_RELATIVE_PATH,
            _STATUS_RELATIVE_PATH,
        )
        root_after = os.fstat(captured_root_fd)
        if _directory_metadata_identity(root_after) != _expected_directory_identity(
            capture.directories["."]
        ):
            raise RuntimeError("manifest staging root changed during scan")
        if staging is not None:
            _validate_descriptor_path(staging_fd, staging, expected_type="directory")
        return capture
    except BaseException:
        capture.close()
        raise


def _revalidate_open_capture(capture: _StagedTreeCapture) -> None:
    for path, directory in capture.directories.items():
        metadata = os.fstat(directory.fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"manifest directory changed type: {path}")
        if _directory_metadata_identity(metadata) != _expected_directory_identity(
            directory
        ):
            raise RuntimeError(f"manifest directory identity changed: {path}")
    for path, member in capture.files.items():
        metadata = os.fstat(member.fd)
        _validate_manifest_member_metadata(metadata, path)
        if _member_metadata_identity(metadata) != (
            member.st_dev,
            member.st_ino,
            member.st_size,
            member.st_mtime_ns,
            member.st_ctime_ns,
        ):
            raise RuntimeError(f"manifest member identity changed: {path}")
        if _sha256_descriptor(member.fd) != member.sha256:
            raise RuntimeError(f"manifest member SHA-256 changed: {path}")


def _compare_tree_captures(
    initial: _StagedTreeCapture,
    final: _StagedTreeCapture,
) -> None:
    if set(initial.directories) != set(final.directories):
        raise RuntimeError("manifest directory membership or identity changed")
    for path, before_directory in initial.directories.items():
        after_directory = final.directories[path]
        if _expected_directory_identity(
            before_directory
        ) != _expected_directory_identity(after_directory):
            raise RuntimeError(f"manifest directory changed during validation: {path}")
    if set(initial.files) != set(final.files):
        raise RuntimeError("manifest file membership changed")
    for path, before in initial.files.items():
        after = final.files[path]
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.sha256,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.sha256,
        ):
            raise RuntimeError(f"manifest member changed during validation: {path}")


def _read_open_manifest_member(member: _OpenManifestMember) -> bytes:
    os.lseek(member.fd, 0, os.SEEK_SET)
    payload = b"".join(iter(lambda: os.read(member.fd, 1024 * 1024), b""))
    if len(payload) != member.st_size or sha256_bytes(payload) != member.sha256:
        raise RuntimeError("payload manifest changed while being read")
    return payload


def _strict_canonical_json_member(
    member: _OpenManifestMember,
    label: str,
) -> object:
    payload = _read_open_manifest_member(member)
    try:
        value = json.loads(
            payload,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if payload != canonical_json_bytes(value):
        raise ValueError(f"{label} bytes are not canonical")
    return value


def _validate_status_staging_identity(
    capture: _StagedTreeCapture,
    staging_fd: int,
) -> None:
    status = _strict_canonical_json_member(
        capture.files[_STATUS_RELATIVE_PATH],
        "conversion status",
    )
    if not isinstance(status, dict):
        raise RuntimeError("conversion status has no staging identity")
    staging_identity = status.get("staging")
    if not isinstance(staging_identity, dict):
        raise RuntimeError("conversion status has no staging identity")
    expected_device = staging_identity.get("st_dev")
    expected_inode = staging_identity.get("st_ino")
    if type(expected_device) is not int or type(expected_inode) is not int:
        raise RuntimeError("conversion status has an invalid staging identity")
    actual = os.fstat(staging_fd)
    if (expected_device, expected_inode) != (actual.st_dev, actual.st_ino):
        raise RuntimeError("conversion status staging identity differs from root")


def build_payload_manifest(staging: Path) -> dict[str, object]:
    staging_fd = _open_validated_directory(staging)
    initial = None
    final = None
    try:
        initial = _capture_staged_tree(staging, staging_fd)
        _validate_status_staging_identity(initial, staging_fd)
        entries = [
            {
                "path": path,
                "size": initial.files[path].st_size,
                "sha256": initial.files[path].sha256,
            }
            for path in sorted(
                initial.files,
                key=lambda value: value.encode("utf-8"),
            )
            if path not in _RESERVED_MANIFEST_PATHS
        ]
        _revalidate_open_capture(initial)
        final = _capture_staged_tree(staging, staging_fd)
        _compare_tree_captures(initial, final)
        _revalidate_open_capture(initial)
        _validate_descriptor_path(staging_fd, staging, expected_type="directory")
        return {"schema_version": 1, "entries": entries}
    finally:
        if final is not None:
            final.close()
        if initial is not None:
            initial.close()
        os.close(staging_fd)


def _validate_manifest_path(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("manifest entry path must be a string")
    if value in _RESERVED_MANIFEST_PATHS:
        raise ValueError(f"manifest entry path is reserved: {value}")
    components = value.split("/")
    if (
        not value
        or value.startswith("/")
        or any(component in ("", ".", "..") for component in components)
        or components[0] not in _PAYLOAD_ROOTS
    ):
        raise ValueError(f"unsafe manifest entry path: {value!r}")
    return value


def _validate_payload_manifest_capture(
    capture: _StagedTreeCapture,
) -> tuple[dict[str, object], bytes]:
    if set(_RESERVED_MANIFEST_PATHS) - set(capture.files):
        raise ValueError("staged tree is missing a reserved path")
    manifest_member = capture.files[_MANIFEST_RELATIVE_PATH]
    manifest_payload = _read_open_manifest_member(manifest_member)
    manifest = _strict_canonical_json_member(manifest_member, "payload manifest")
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "entries",
    }:
        raise ValueError("payload manifest has an invalid schema")
    if manifest["schema_version"] != 1 or type(manifest["schema_version"]) is not int:
        raise ValueError("payload manifest has an invalid schema version")
    entries = manifest["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("payload manifest entries must be a nonempty list")

    expected_paths = []
    expected_by_path: dict[str, tuple[int, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "size",
            "sha256",
        }:
            raise ValueError("payload manifest entry has an invalid schema")
        path = _validate_manifest_path(entry["path"])
        size = entry["size"]
        digest = entry["sha256"]
        if type(size) is not int or size < 0:
            raise ValueError("payload manifest entry size must be nonnegative")
        if not isinstance(digest, str) or _LOWERCASE_SHA256.fullmatch(digest) is None:
            raise ValueError("payload manifest entry has malformed SHA-256")
        if path in expected_by_path:
            raise ValueError(f"payload manifest has duplicate path: {path}")
        expected_paths.append(path)
        expected_by_path[path] = (size, digest)
    if expected_paths != sorted(
        expected_paths,
        key=lambda value: value.encode("utf-8"),
    ):
        raise ValueError("payload manifest paths are not in canonical order")

    actual_payload = {
        path: (member.st_size, member.sha256)
        for path, member in capture.files.items()
        if path not in _RESERVED_MANIFEST_PATHS
    }
    if set(expected_by_path) != set(actual_payload):
        raise ValueError("payload manifest membership differs from staged tree")
    for path, expected_identity in expected_by_path.items():
        if actual_payload[path] != expected_identity:
            raise ValueError(f"payload manifest identity differs for path: {path}")

    required_directories = {".", *_PAYLOAD_ROOTS}
    for path in expected_paths:
        parts = path.split("/")[:-1]
        required_directories.update(
            "/".join(parts[:index]) for index in range(1, len(parts) + 1)
        )
    if set(capture.directories) != required_directories:
        raise ValueError(
            "payload manifest directory membership differs from staged tree"
        )
    return manifest, manifest_payload


def validate_payload_manifest(staging: Path) -> dict[str, object]:
    staging_fd = _open_validated_directory(staging)
    initial = None
    final = None
    try:
        initial = _capture_staged_tree(staging, staging_fd)
        _validate_status_staging_identity(initial, staging_fd)
        manifest, _ = _validate_payload_manifest_capture(initial)

        _revalidate_open_capture(initial)
        final = _capture_staged_tree(staging, staging_fd)
        _compare_tree_captures(initial, final)
        _revalidate_open_capture(initial)
        _validate_descriptor_path(staging_fd, staging, expected_type="directory")
        return manifest
    finally:
        if final is not None:
            final.close()
        if initial is not None:
            initial.close()
        os.close(staging_fd)


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


def atomic_replace_status(
    path: Path,
    payload: bytes,
    mode: int,
    fault: Callable[[str], None],
) -> None:
    if path.name != _STATUS_RELATIVE_PATH:
        raise ValueError("only CONVERSION_STATUS.json may be replaced")
    parent_fd = _open_validated_directory(path.parent)
    destination_name = _validated_basename(path, "conversion status")
    temporary_name = os.fsencode(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary_fd = None
    destination_fd = None
    temporary_exists = False
    try:
        destination_fd = os.open(
            destination_name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        existing = _validate_descriptor_entry(
            destination_fd,
            parent_fd,
            destination_name,
            expected_type="file",
            require_single_link=True,
        )
        temporary_fd = os.open(
            temporary_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            mode,
            dir_fd=parent_fd,
        )
        temporary_exists = True
        created = _validate_descriptor_entry(
            temporary_fd,
            parent_fd,
            temporary_name,
            expected_type="file",
            require_single_link=True,
        )
        temporary_identity = (created.st_dev, created.st_ino)
        fault("after_complete_temp_creation")
        remaining = memoryview(payload)
        while remaining:
            written = os.write(temporary_fd, remaining)
            if written <= 0:
                raise OSError("conversion status write made no forward progress")
            remaining = remaining[written:]
        written_metadata = _validate_descriptor_entry(
            temporary_fd,
            parent_fd,
            temporary_name,
            expected_type="file",
            require_single_link=True,
        )
        if (
            written_metadata.st_dev,
            written_metadata.st_ino,
            written_metadata.st_size,
        ) != (*temporary_identity, len(payload)):
            raise RuntimeError("temporary conversion status identity changed")
        fault("before_complete_temp_fsync")
        os.fsync(temporary_fd)
        fault("after_complete_temp_fsync")
        if _sha256_descriptor(temporary_fd) != sha256_bytes(payload):
            raise RuntimeError("temporary conversion status bytes changed")
        _validate_descriptor_path(parent_fd, path.parent, expected_type="directory")
        current = _validate_descriptor_entry(
            destination_fd,
            parent_fd,
            destination_name,
            expected_type="file",
            require_single_link=True,
        )
        if (current.st_dev, current.st_ino) != (existing.st_dev, existing.st_ino):
            raise RuntimeError("conversion status identity changed before replace")
        os.replace(
            temporary_name,
            destination_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_exists = False
        fault("after_complete_status_rename")
        os.fsync(parent_fd)
        visible_fd = os.open(
            destination_name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            visible = _validate_descriptor_entry(
                visible_fd,
                parent_fd,
                destination_name,
                expected_type="file",
                require_single_link=True,
            )
            if visible.st_size != len(payload):
                raise RuntimeError("conversion status size changed")
            if _sha256_descriptor(visible_fd) != sha256_bytes(payload):
                raise RuntimeError("conversion status digest changed")
        finally:
            os.close(visible_fd)
        _validate_descriptor_path(parent_fd, path.parent, expected_type="directory")
    finally:
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except FileNotFoundError:
                pass
        if temporary_fd is not None:
            os.close(temporary_fd)
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(parent_fd)


class PublicationFilesystem:
    def fsync_file(self, path: Path) -> None:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        fd = os.open(path, flags)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise RuntimeError(f"unsafe file for fsync: {path}")
            os.fsync(fd)
        finally:
            os.close(fd)

    def fsync_directory(self, path: Path) -> None:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def chmod(self, path: Path, mode: int) -> None:
        fd = os.open(
            path,
            os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            before = os.fstat(fd)
            if not (stat.S_ISREG(before.st_mode) or stat.S_ISDIR(before.st_mode)):
                raise RuntimeError(f"unsafe chmod target: {path}")
            if before.st_nlink != 1 and stat.S_ISREG(before.st_mode):
                raise RuntimeError(f"multiply linked chmod target: {path}")
            os.fchmod(fd, mode)
            after = os.fstat(fd)
            if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
                raise RuntimeError(f"chmod target identity changed: {path}")
            reopened = os.open(
                path,
                os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
            try:
                visible = os.fstat(reopened)
                if (visible.st_dev, visible.st_ino) != (
                    before.st_dev,
                    before.st_ino,
                ):
                    raise RuntimeError(f"chmod pathname was replaced: {path}")
                if stat.S_IMODE(visible.st_mode) != mode:
                    raise RuntimeError(f"chmod mode did not persist: {path}")
            finally:
                os.close(reopened)
        finally:
            os.close(fd)

    def atomic_write_new(self, path: Path, payload: bytes, mode: int) -> None:
        atomic_write_new_bytes(path, payload, mode)

    def atomic_replace_status(
        self,
        path: Path,
        payload: bytes,
        mode: int,
        fault: Callable[[str], None],
    ) -> None:
        atomic_replace_status(path, payload, mode, fault)

    def rename_noreplace(
        self,
        source: Path,
        destination: Path,
        *,
        expected_identity: tuple[int, int] | None = None,
    ) -> None:
        rename_noreplace(
            source,
            destination,
            expected_identity=expected_identity,
        )


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


def validate_staged_dataset(staging: Path, plan: ConversionPlan):
    # Import lazily to keep the validator's dependency on this module acyclic.
    # The conversion orchestrator calls this after generation and immediately
    # before durable publication.
    from scripts.certify_psi0_dataset import DatasetExpectations, validate_dataset

    return validate_dataset(
        staging,
        expected=DatasetExpectations.from_plan(plan),
        require_final_modes=False,
    )


def _canonical_tree_paths(
    root: Path,
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    files: list[tuple[str, Path]] = []
    directories: list[tuple[str, Path]] = [(".", root)]
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directory_names.sort(key=lambda value: value.encode("utf-8"))
        file_names.sort(key=lambda value: value.encode("utf-8"))
        for name in directory_names:
            path = directory_path / name
            metadata = path.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(f"unsafe canonical directory: {path}")
            directories.append((path.relative_to(root).as_posix(), path))
        for name in file_names:
            path = directory_path / name
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise RuntimeError(f"unsafe canonical file: {path}")
            files.append((path.relative_to(root).as_posix(), path))
    files.sort(key=lambda item: item[0].encode("utf-8"))
    directories.sort(
        key=lambda item: (
            -(0 if item[0] == "." else item[0].count("/") + 1),
            item[0].encode("utf-8"),
        )
    )
    return files, directories


def _write_manifest_new(
    path: Path,
    payload: bytes,
    filesystem: PublicationFilesystem,
    fault: Callable[[str], None],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o644,
    )
    temporary_exists = True
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("manifest write made no forward progress")
            remaining = remaining[written:]
        os.close(fd)
        fd = -1
        filesystem.fsync_file(temporary)
        fault("after_manifest_file_fsync")
        rename_noreplace(temporary, path)
        temporary_exists = False
        fault("after_manifest_rename")
        filesystem.fsync_directory(path.parent)
        fault("after_meta_fsync")
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary_exists:
            try:
                temporary.unlink()
                filesystem.fsync_directory(path.parent)
            except FileNotFoundError:
                pass


def _complete_status_bytes(
    staging: Path,
    manifest: dict[str, object],
    manifest_payload: bytes,
    converter: ConverterIdentity,
) -> bytes:
    entries = manifest["entries"]
    if not isinstance(entries, list):
        raise ValueError("payload manifest entries must be a list")
    covered_bytes = 0
    for entry in entries:
        if not isinstance(entry, dict) or type(entry.get("size")) is not int:
            raise ValueError("payload manifest entry has invalid size")
        covered_bytes += entry["size"]
    metadata = staging.stat(follow_symlinks=False)
    return canonical_json_bytes(
        {
            "schema_version": 1,
            "staging": {"st_dev": metadata.st_dev, "st_ino": metadata.st_ino},
            "state": "complete",
            "manifest_sha256": sha256_bytes(manifest_payload),
            "manifest_size": len(manifest_payload),
            "entry_count": len(entries),
            "covered_bytes": covered_bytes,
            "converter": asdict(converter),
        }
    )


def _failed_status_bytes(staging: Path, error: BaseException) -> bytes:
    metadata = staging.stat(follow_symlinks=False)
    return canonical_json_bytes(
        {
            "schema_version": 1,
            "staging": {"st_dev": metadata.st_dev, "st_ino": metadata.st_ino},
            "state": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )


def _read_regular_file_bytes(path: Path, label: str) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"unsafe {label}")
        payload = b"".join(iter(lambda: os.read(fd, 1024 * 1024), b""))
        after = os.fstat(fd)
        if (after.st_dev, after.st_ino, after.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ) or after.st_nlink != 1:
            raise RuntimeError(f"{label} identity changed")
        if len(payload) != before.st_size:
            raise RuntimeError(f"{label} size changed")
    finally:
        os.close(fd)
    return payload


def _read_canonical_json_file(
    path: Path, label: str
) -> tuple[dict[str, object], bytes]:
    payload = _read_regular_file_bytes(path, label)
    try:
        value = json.loads(
            payload,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict) or payload != canonical_json_bytes(value):
        raise ValueError(f"{label} bytes are not canonical")
    return value, payload


def _validate_complete_status(
    root: Path,
    expected_payload: bytes | None = None,
) -> tuple[dict[str, object], bytes]:
    status, payload = _read_canonical_json_file(
        root / _STATUS_RELATIVE_PATH,
        "conversion status",
    )
    if status.get("state") != "complete":
        raise RuntimeError("conversion status is not complete")
    root_metadata = root.stat(follow_symlinks=False)
    staging_identity = status.get("staging")
    if not isinstance(staging_identity, dict) or (
        staging_identity.get("st_dev"),
        staging_identity.get("st_ino"),
    ) != (root_metadata.st_dev, root_metadata.st_ino):
        raise RuntimeError("complete status root identity differs")
    manifest_path = root / _MANIFEST_RELATIVE_PATH
    manifest_payload = _read_regular_file_bytes(manifest_path, "payload manifest")
    manifest = validate_payload_manifest(root)
    entries = manifest["entries"]
    assert isinstance(entries, list)
    expected_fields = {
        "manifest_sha256": sha256_bytes(manifest_payload),
        "manifest_size": len(manifest_payload),
        "entry_count": len(entries),
        "covered_bytes": sum(entry["size"] for entry in entries),
    }
    if any(status.get(field) != value for field, value in expected_fields.items()):
        raise RuntimeError("complete status differs from payload manifest")
    if expected_payload is not None and payload != expected_payload:
        raise RuntimeError("complete status bytes changed")
    return status, payload


def _require_in_progress_status(root: Path) -> None:
    status, payload = _read_canonical_json_file(
        root / _STATUS_RELATIVE_PATH,
        "conversion status",
    )
    if status.get("state") != "in_progress":
        raise RuntimeError("conversion status is not in_progress")
    metadata = root.stat(follow_symlinks=False)
    if payload != _in_progress_status_bytes(metadata):
        raise RuntimeError("in_progress status differs from staging identity")


def _validate_final_modes(root: Path) -> None:
    files, directories = _canonical_tree_paths(root)
    for relative_path, path in files:
        metadata = path.lstat()
        if stat.S_IMODE(metadata.st_mode) != 0o444:
            raise RuntimeError(f"canonical file has wrong mode: {relative_path}")
    for relative_path, path in directories:
        metadata = path.lstat()
        if stat.S_IMODE(metadata.st_mode) != 0o555:
            raise RuntimeError(f"canonical directory has wrong mode: {relative_path}")


def _replace_with_failed_status(
    staging: Path,
    error: BaseException,
    filesystem: PublicationFilesystem,
) -> None:
    filesystem.atomic_replace_status(
        staging / _STATUS_RELATIVE_PATH,
        _failed_status_bytes(staging, error),
        0o644,
        lambda _point: None,
    )


def inspect_publication_state(staging: Path, output: Path) -> str:
    if os.path.lexists(output):
        if os.path.lexists(staging):
            raise RuntimeError("both staging and canonical output exist")
        output_fd = _open_validated_directory(output)
        os.close(output_fd)
        PublicationFilesystem().fsync_directory(output.parent)
        _validate_complete_status(output)
        _validate_final_modes(output)
        return "published"
    if not os.path.lexists(staging):
        raise FileNotFoundError("neither staging nor canonical output exists")
    status, _ = _read_canonical_json_file(
        staging / _STATUS_RELATIVE_PATH,
        "conversion status",
    )
    state = status.get("state")
    if state == "failed":
        return "pre_completion_failed"
    if state == "in_progress":
        return "completion_uncertain_unpublished"
    if state != "complete":
        raise RuntimeError("staging has an unknown conversion state")
    try:
        _validate_complete_status(staging)
        _validate_final_modes(staging)
    except (OSError, ValueError, RuntimeError):
        return "completion_uncertain_unpublished"
    return "complete_unpublished"


def _absolute_lexical_path(value: str | os.PathLike[str], label: str) -> Path:
    text = os.fspath(value)
    if not os.path.isabs(text):
        raise ValueError(f"{label} must be absolute")
    if any(component in {".", ".."} for component in text.split(os.sep)):
        raise ValueError(f"{label} must not contain dot segments")
    return Path(text)


def _validate_preserved_staging_paths(
    staging_value: str | os.PathLike[str],
    output_value: str | os.PathLike[str],
) -> tuple[Path, Path]:
    staging = _absolute_lexical_path(staging_value, "staging path")
    output = _absolute_lexical_path(output_value, "output path")
    _validated_basename(output, "output path")
    expected = re.fullmatch(
        rf"\.{re.escape(output.name)}\.staging-"
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        staging.name,
    )
    if expected is None:
        raise ValueError("staging path has the wrong direct-sibling name")
    if staging.parent != output.parent:
        raise ValueError("staging and output must be direct siblings")
    return staging, output


def _require_output_absent_at(
    parent_fd: int,
    output_name: bytes,
    boundary: str,
) -> None:
    try:
        os.stat(output_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise FileExistsError(f"canonical output exists at {boundary}")


def _open_preserved_staging(
    staging: Path,
    output: Path,
    ownership: _ConversionLockOwnership,
) -> tuple[int, os.stat_result]:
    parent_fd = ownership.parent_fd
    _validate_owned_parent(ownership, output)
    output_name = _validated_basename(output, "canonical output")
    _require_output_absent_at(parent_fd, output_name, "initial validation")
    staging_fd = None
    try:
        staging_name = _validated_basename(staging, "preserved staging")
        pathname = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(pathname.st_mode):
            raise RuntimeError("preserved staging is not a directory")
        staging_fd = os.open(
            staging_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        opened = _validate_descriptor_entry(
            staging_fd,
            parent_fd,
            staging_name,
            expected_type="directory",
        )
        if (opened.st_dev, opened.st_ino) != (pathname.st_dev, pathname.st_ino):
            raise RuntimeError("preserved staging identity changed while opening")
        _validate_owned_parent(ownership, output)
        return staging_fd, opened
    except BaseException:
        if staging_fd is not None:
            os.close(staging_fd)
        raise


def _read_regular_at(
    directory_fd: int, name: str, label: str
) -> tuple[bytes, os.stat_result]:
    encoded = os.fsencode(name)
    pathname = os.stat(encoded, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(pathname.st_mode) or pathname.st_nlink != 1:
        raise RuntimeError(f"unsafe {label}")
    fd = os.open(
        encoded,
        os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    try:
        opened = _validate_descriptor_entry(
            fd,
            directory_fd,
            encoded,
            expected_type="file",
            require_single_link=True,
        )
        if (opened.st_dev, opened.st_ino) != (pathname.st_dev, pathname.st_ino):
            raise RuntimeError(f"{label} identity changed while opening")
        payload = b"".join(iter(lambda: os.read(fd, 1024 * 1024), b""))
        after = _validate_descriptor_entry(
            fd,
            directory_fd,
            encoded,
            expected_type="file",
            require_single_link=True,
        )
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ) or len(payload) != opened.st_size:
            raise RuntimeError(f"{label} changed while being read")
        return payload, opened
    finally:
        os.close(fd)


def _tree_entry_type(metadata: os.stat_result) -> str:
    if stat.S_ISDIR(metadata.st_mode):
        return "directory"
    if stat.S_ISREG(metadata.st_mode):
        return "file"
    if stat.S_ISLNK(metadata.st_mode):
        return "symlink"
    return "special"


def _inspect_tree_at(
    directory_fd: int,
    relative: str,
    root_device: int,
) -> list[dict[str, object]]:
    root_metadata = os.fstat(directory_fd)
    result = [
        {
            "path": relative,
            "type": "directory",
            "st_dev": root_metadata.st_dev,
            "st_ino": root_metadata.st_ino,
            "st_nlink": root_metadata.st_nlink,
            "mode": f"{stat.S_IMODE(root_metadata.st_mode):04o}",
            "size": root_metadata.st_size,
        }
    ]
    for name in sorted(os.listdir(directory_fd), key=lambda item: item.encode("utf-8")):
        encoded = os.fsencode(name)
        metadata = os.stat(encoded, dir_fd=directory_fd, follow_symlinks=False)
        child_relative = name if relative == "." else f"{relative}/{name}"
        entry_type = _tree_entry_type(metadata)
        result.append(
            {
                "path": child_relative,
                "type": entry_type,
                "st_dev": metadata.st_dev,
                "st_ino": metadata.st_ino,
                "st_nlink": metadata.st_nlink,
                "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
                "size": metadata.st_size,
            }
        )
        if entry_type != "directory":
            continue
        if metadata.st_dev != root_device:
            raise RuntimeError("preserved staging contains a mounted directory")
        child_fd = os.open(
            encoded,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        try:
            opened = _validate_descriptor_entry(
                child_fd,
                directory_fd,
                encoded,
                expected_type="directory",
            )
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise RuntimeError("preserved tree changed during inspection")
            descendants = _inspect_tree_at(child_fd, child_relative, root_device)
            result.extend(descendants[1:])
        finally:
            os.close(child_fd)
    return result


def _canonical_json_object(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict) or payload != canonical_json_bytes(value):
        raise ValueError(f"{label} bytes are not canonical")
    return value


def _validate_status_record_at(
    status: dict[str, object],
    payload: bytes,
    staging_fd: int,
) -> str:
    root = os.fstat(staging_fd)
    staging_identity = status.get("staging")
    if not isinstance(staging_identity, dict) or set(staging_identity) != {
        "st_dev",
        "st_ino",
    }:
        raise RuntimeError("conversion status has an invalid staging identity")
    if any(type(staging_identity.get(field)) is not int for field in staging_identity):
        raise RuntimeError("conversion status has an invalid staging identity")
    if (staging_identity["st_dev"], staging_identity["st_ino"]) != (
        root.st_dev,
        root.st_ino,
    ):
        raise RuntimeError("conversion status root identity differs")
    state = status.get("state")
    common = {"schema_version", "staging", "state"}
    if (
        status.get("schema_version") != 1
        or type(status.get("schema_version")) is not int
    ):
        raise ValueError("conversion status has an invalid schema version")
    if state == "in_progress":
        if set(status) != common or payload != _in_progress_status_bytes(root):
            raise ValueError("in_progress status has an invalid schema or binding")
    elif state == "failed":
        if set(status) != common | {"error", "error_type"} or not all(
            isinstance(status.get(field), str) for field in ("error", "error_type")
        ):
            raise ValueError("failed status has an invalid schema")
    elif state == "complete":
        if set(status) != common | {
            "manifest_sha256",
            "manifest_size",
            "entry_count",
            "covered_bytes",
            "converter",
        }:
            raise ValueError("complete status has an invalid schema")
        for field in ("manifest_size", "entry_count", "covered_bytes"):
            if type(status[field]) is not int or status[field] < 0:
                raise ValueError(f"complete status has invalid {field}")
        if (
            not isinstance(status["manifest_sha256"], str)
            or _LOWERCASE_SHA256.fullmatch(status["manifest_sha256"]) is None
        ):
            raise ValueError("complete status has invalid manifest SHA-256")
        converter = status["converter"]
        if (
            not isinstance(converter, dict)
            or set(converter) != {"commit", "script_sha256"}
            or not isinstance(converter["commit"], str)
            or re.fullmatch(r"[0-9a-f]{40}", converter["commit"]) is None
            or not isinstance(converter["script_sha256"], str)
            or _LOWERCASE_SHA256.fullmatch(converter["script_sha256"]) is None
        ):
            raise ValueError("complete status has invalid converter identity")
    else:
        raise ValueError("conversion status has an unknown state")
    return state


def _validate_final_capture_modes(
    capture: _StagedTreeCapture,
    staging_fd: int,
) -> None:
    root_device = os.fstat(staging_fd).st_dev
    for path, member in capture.files.items():
        metadata = os.fstat(member.fd)
        _validate_manifest_member_metadata(metadata, path)
        if metadata.st_dev != root_device or stat.S_IMODE(metadata.st_mode) != 0o444:
            raise RuntimeError(f"canonical file has wrong identity or mode: {path}")
    for path, directory in capture.directories.items():
        metadata = os.fstat(directory.fd)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_dev != root_device
            or metadata.st_nlink < 1
            or stat.S_IMODE(metadata.st_mode) != 0o555
        ):
            raise RuntimeError(
                f"canonical directory has wrong identity or mode: {path}"
            )


def _validate_complete_tree_at(
    staging_fd: int,
) -> tuple[dict[str, object], bytes]:
    initial = None
    final = None
    try:
        initial = _capture_staged_tree(None, staging_fd)
        manifest, manifest_payload = _validate_payload_manifest_capture(initial)
        status_member = initial.files[_STATUS_RELATIVE_PATH]
        status_payload = _read_open_manifest_member(status_member)
        status = _strict_canonical_json_member(status_member, "conversion status")
        if not isinstance(status, dict):
            raise ValueError("conversion status is not an object")
        if _validate_status_record_at(status, status_payload, staging_fd) != "complete":
            raise ValueError("conversion status is not complete")
        entries = manifest["entries"]
        assert isinstance(entries, list)
        expected = {
            "manifest_sha256": sha256_bytes(manifest_payload),
            "manifest_size": len(manifest_payload),
            "entry_count": len(entries),
            "covered_bytes": sum(entry["size"] for entry in entries),
        }
        if any(status[field] != value for field, value in expected.items()):
            raise RuntimeError("complete status differs from payload manifest")
        _validate_final_capture_modes(initial, staging_fd)
        _revalidate_open_capture(initial)
        final = _capture_staged_tree(None, staging_fd)
        _compare_tree_captures(initial, final)
        _revalidate_open_capture(initial)
        return manifest, manifest_payload
    finally:
        if final is not None:
            final.close()
        if initial is not None:
            initial.close()


def _inspect_manifest_at(staging_fd: int) -> dict[str, object]:
    try:
        metadata = os.stat(b"meta", dir_fd=staging_fd, follow_symlinks=False)
    except FileNotFoundError:
        return {"state": "absent"}
    if not stat.S_ISDIR(metadata.st_mode):
        return {"state": "invalid", "error": "meta is not a directory"}
    meta_fd = None
    try:
        meta_fd = os.open(
            b"meta",
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=staging_fd,
        )
        opened = _validate_descriptor_entry(
            meta_fd,
            staging_fd,
            b"meta",
            expected_type="directory",
        )
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise RuntimeError("meta identity changed during manifest inspection")
        try:
            os.stat(
                b"conversion_manifest.json",
                dir_fd=meta_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return {"state": "absent"}
        value, manifest_payload = _validate_complete_tree_at(staging_fd)
        return {
            "state": "valid",
            "sha256": sha256_bytes(manifest_payload),
            "size": len(manifest_payload),
            "entry_count": len(value["entries"]),
        }
    except (OSError, RuntimeError, ValueError) as exc:
        return {"state": "invalid", "error": str(exc)}
    finally:
        if meta_fd is not None:
            os.close(meta_fd)


def _inspect_preserved_staging_locked(staging: Path, output: Path) -> dict[str, object]:
    ownership = _conversion_lock_ownership(output)
    parent_fd = ownership.parent_fd
    staging_fd, staging_metadata = _open_preserved_staging(staging, output, ownership)
    try:
        status_payload, status_metadata = _read_regular_at(
            staging_fd, _STATUS_RELATIVE_PATH, "conversion status"
        )
        status_value = None
        try:
            status_value = _canonical_json_object(
                status_payload,
                "conversion status",
            )
            state = _validate_status_record_at(
                status_value,
                status_payload,
                staging_fd,
            )
            status_error = None
        except (OSError, RuntimeError, ValueError) as exc:
            state = status_value.get("state") if status_value is not None else None
            status_error = str(exc)
        tree = _inspect_tree_at(staging_fd, ".", staging_metadata.st_dev)
        manifest = _inspect_manifest_at(staging_fd)
        if status_error is not None:
            if manifest.get("state") == "valid":
                manifest = {"state": "invalid", "error": status_error}
            classification = "completion_uncertain_unpublished"
        elif state == "failed":
            classification = "pre_completion_failed"
        elif state == "in_progress":
            classification = "completion_uncertain_unpublished"
        elif state == "complete":
            final_modes = all(
                (entry["type"] == "file" and entry["mode"] == "0444")
                or (entry["type"] == "directory" and entry["mode"] == "0555")
                for entry in tree
            )
            if manifest.get("state") == "valid" and final_modes:
                classification = "complete_unpublished"
            else:
                classification = "completion_uncertain_unpublished"
        else:
            classification = "completion_uncertain_unpublished"

        _validate_owned_parent(ownership, output)
        visible = _validate_descriptor_entry(
            staging_fd,
            parent_fd,
            _validated_basename(staging, "preserved staging"),
            expected_type="directory",
        )
        if (visible.st_dev, visible.st_ino) != (
            staging_metadata.st_dev,
            staging_metadata.st_ino,
        ):
            raise RuntimeError("preserved staging changed during inspection")
        return {
            "schema_version": 1,
            "staging_path": str(staging),
            "output_path": str(output),
            "root_identity": {
                "st_dev": staging_metadata.st_dev,
                "st_ino": staging_metadata.st_ino,
            },
            "status": {
                "bytes": status_payload.decode("utf-8"),
                "sha256": sha256_bytes(status_payload),
                "size": status_metadata.st_size,
                "mode": f"{stat.S_IMODE(status_metadata.st_mode):04o}",
            },
            "manifest": manifest,
            "failure_classification": classification,
            "tree": tree,
        }
    finally:
        os.close(staging_fd)


def inspect_preserved_staging(
    staging_value: str | os.PathLike[str],
    output_value: str | os.PathLike[str],
) -> dict[str, object]:
    staging, output = _validate_preserved_staging_paths(staging_value, output_value)
    with conversion_lock(output):
        return _inspect_preserved_staging_locked(staging, output)


def _remove_tree_contents_at(directory_fd: int, root_device: int) -> None:
    for name in sorted(os.listdir(directory_fd), key=lambda item: item.encode("utf-8")):
        encoded = os.fsencode(name)
        metadata = os.stat(encoded, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            if metadata.st_dev != root_device:
                raise RuntimeError("preserved staging contains a mounted directory")
            child_fd = os.open(
                encoded,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            try:
                opened = _validate_descriptor_entry(
                    child_fd,
                    directory_fd,
                    encoded,
                    expected_type="directory",
                )
                if (opened.st_dev, opened.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    raise RuntimeError("preserved tree changed during removal")
                _remove_tree_contents_at(child_fd, root_device)
                os.fsync(child_fd)
                visible = _validate_descriptor_entry(
                    child_fd,
                    directory_fd,
                    encoded,
                    expected_type="directory",
                )
                if (visible.st_dev, visible.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    raise RuntimeError("preserved directory changed during removal")
            finally:
                os.close(child_fd)
            os.rmdir(encoded, dir_fd=directory_fd)
        else:
            os.unlink(encoded, dir_fd=directory_fd)
        os.fsync(directory_fd)


def remove_preserved_staging(
    staging_value: str | os.PathLike[str],
    output_value: str | os.PathLike[str],
    *,
    expected_status_sha256: str,
    confirm_remove: str,
) -> None:
    staging, output = _validate_preserved_staging_paths(staging_value, output_value)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_status_sha256):
        raise ValueError("expected status SHA-256 must be 64 lowercase hex characters")
    if confirm_remove != str(staging):
        raise ValueError("removal confirmation does not exactly match staging path")
    with conversion_lock(output):
        _remove_preserved_staging_locked(
            staging,
            output,
            expected_status_sha256=expected_status_sha256,
        )


def _remove_preserved_staging_locked(
    staging: Path,
    output: Path,
    *,
    expected_status_sha256: str,
) -> None:
    ownership = _conversion_lock_ownership(output)
    parent_fd = ownership.parent_fd
    output_name = _validated_basename(output, "canonical output")
    staging_name = _validated_basename(staging, "preserved staging")
    staging_fd, staging_metadata = _open_preserved_staging(staging, output, ownership)
    try:
        status_payload, _ = _read_regular_at(
            staging_fd, _STATUS_RELATIVE_PATH, "conversion status"
        )
        if sha256_bytes(status_payload) != expected_status_sha256:
            raise ValueError("conversion status SHA-256 does not match authorization")
        status = _canonical_json_object(status_payload, "conversion status")
        _validate_status_record_at(status, status_payload, staging_fd)
        _validate_owned_parent(ownership, output)
        visible = _validate_descriptor_entry(
            staging_fd,
            parent_fd,
            staging_name,
            expected_type="directory",
        )
        if (visible.st_dev, visible.st_ino) != (
            staging_metadata.st_dev,
            staging_metadata.st_ino,
        ):
            raise RuntimeError("preserved staging changed before removal")
        _require_output_absent_at(
            parent_fd,
            output_name,
            "before_recursive_mutation",
        )
        _remove_tree_contents_at(staging_fd, staging_metadata.st_dev)
        _validate_owned_parent(ownership, output)
        visible = _validate_descriptor_entry(
            staging_fd,
            parent_fd,
            staging_name,
            expected_type="directory",
        )
        if (visible.st_dev, visible.st_ino) != (
            staging_metadata.st_dev,
            staging_metadata.st_ino,
        ):
            raise RuntimeError("preserved staging changed before root removal")
        _require_output_absent_at(
            parent_fd,
            output_name,
            "before_staging_rmdir",
        )
        os.rmdir(staging_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        try:
            os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("preserved staging removal did not persist")
        _validate_owned_parent(ownership, output)
    finally:
        os.close(staging_fd)


def publish_staged_dataset(
    staging: Path,
    output: Path,
    converter: ConverterIdentity,
    filesystem: PublicationFilesystem,
    fault: Callable[[str], None],
) -> PublicationResult:
    completion_write_started = False
    durable_complete = False
    publication_renamed = False
    publication_durable = False
    published_result: PublicationResult | None = None
    manifest_sha256 = ""
    complete_status_sha256 = ""
    try:
        fault("after_payload_close")

        manifest = build_payload_manifest(staging)
        fault("after_staged_validation")

        manifest_payload = canonical_json_bytes(manifest)
        manifest_sha256 = sha256_bytes(manifest_payload)
        _write_manifest_new(
            staging / _MANIFEST_RELATIVE_PATH,
            manifest_payload,
            filesystem,
            fault,
        )

        files, directories = _canonical_tree_paths(staging)
        for relative_path, path in files:
            filesystem.fsync_file(path)
            fault(f"after_each_payload_fsync:{relative_path}")
        for relative_path, path in directories:
            filesystem.fsync_directory(path)
            fault(f"after_each_precomplete_directory_fsync:{relative_path}")

        if validate_payload_manifest(staging) != manifest:
            raise RuntimeError("staged manifest changed before completion")
        _require_in_progress_status(staging)
        fault("after_precomplete_revalidation")

        complete_payload = _complete_status_bytes(
            staging,
            manifest,
            manifest_payload,
            converter,
        )
        complete_status_sha256 = sha256_bytes(complete_payload)
        completion_write_started = True
        filesystem.atomic_replace_status(
            staging / _STATUS_RELATIVE_PATH,
            complete_payload,
            0o644,
            fault,
        )
        durable_complete = True
        fault("after_complete_root_fsync")

        files, directories = _canonical_tree_paths(staging)
        for relative_path, path in files:
            filesystem.chmod(path, 0o444)
            fault(f"after_each_chmod:{relative_path}")
        for relative_path, path in directories:
            filesystem.chmod(path, 0o555)
            fault(f"after_each_chmod:{relative_path}")

        for relative_path, path in files:
            filesystem.fsync_file(path)
            fault(f"after_each_final_file_fsync:{relative_path}")
        for relative_path, path in directories:
            filesystem.fsync_directory(path)
            fault(f"after_each_final_directory_fsync:{relative_path}")

        if staging.parent != output.parent:
            raise ValueError("staging and canonical output must be siblings")
        parent_fd = _open_validated_directory(staging.parent)
        staging_fd = None
        try:
            staging_name = _validated_basename(staging, "staging root")
            output_name = _validated_basename(output, "canonical output")
            staging_fd = os.open(
                staging_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            pinned = _validate_descriptor_entry(
                staging_fd,
                parent_fd,
                staging_name,
                expected_type="directory",
            )
            expected_staging_identity = (pinned.st_dev, pinned.st_ino)

            if validate_payload_manifest(staging) != manifest:
                raise RuntimeError("sealed manifest changed before publication")
            _validate_complete_status(staging, complete_payload)
            _validate_final_modes(staging)
            pinned_after_validation = _validate_descriptor_entry(
                staging_fd,
                parent_fd,
                staging_name,
                expected_type="directory",
            )
            if (
                pinned_after_validation.st_dev,
                pinned_after_validation.st_ino,
            ) != expected_staging_identity:
                raise RuntimeError("validated staging root identity changed")
            fault("after_final_revalidation")

            if os.path.lexists(output):
                raise FileExistsError(f"canonical output already exists: {output}")
            fault("before_publication_rename")
            _validate_descriptor_path(
                parent_fd,
                staging.parent,
                expected_type="directory",
            )
            before_rename = _validate_descriptor_entry(
                staging_fd,
                parent_fd,
                staging_name,
                expected_type="directory",
            )
            if (
                before_rename.st_dev,
                before_rename.st_ino,
            ) != expected_staging_identity:
                raise RuntimeError("staging root was replaced before publication")
            if validate_payload_manifest(staging) != manifest:
                raise RuntimeError("sealed manifest changed at publication boundary")
            _validate_complete_status(staging, complete_payload)
            _validate_final_modes(staging)
            final_source = _validate_descriptor_entry(
                staging_fd,
                parent_fd,
                staging_name,
                expected_type="directory",
            )
            if (final_source.st_dev, final_source.st_ino) != expected_staging_identity:
                raise RuntimeError("staging root changed during final revalidation")
            _validate_descriptor_path(
                parent_fd,
                staging.parent,
                expected_type="directory",
            )
            if os.path.lexists(output):
                raise FileExistsError(f"canonical output already exists: {output}")
            filesystem.rename_noreplace(
                staging,
                output,
                expected_identity=expected_staging_identity,
            )
            publication_renamed = True
            published = _validate_descriptor_entry(
                staging_fd,
                parent_fd,
                output_name,
                expected_type="directory",
            )
            if (published.st_dev, published.st_ino) != expected_staging_identity:
                raise RuntimeError("published root differs from validated staging root")
            _validate_descriptor_path(
                parent_fd,
                output.parent,
                expected_type="directory",
            )
            try:
                fault("after_publication_rename")
                _validate_descriptor_path(
                    parent_fd,
                    output.parent,
                    expected_type="directory",
                )
                _validate_descriptor_entry(
                    staging_fd,
                    parent_fd,
                    output_name,
                    expected_type="directory",
                )
                filesystem.fsync_directory(output.parent)
                _validate_descriptor_path(
                    parent_fd,
                    output.parent,
                    expected_type="directory",
                )
                durable_root = _validate_descriptor_entry(
                    staging_fd,
                    parent_fd,
                    output_name,
                    expected_type="directory",
                )
                if (
                    durable_root.st_dev,
                    durable_root.st_ino,
                ) != expected_staging_identity:
                    raise RuntimeError(
                        "durable output differs from validated staging root"
                    )
                os.fsync(parent_fd)
                publication_durable = True
                published_result = PublicationResult(
                    dataset_root=output,
                    manifest_sha256=manifest_sha256,
                    complete_status_sha256=complete_status_sha256,
                    state="published",
                )
            except BaseException as exc:
                if publication_durable:
                    if published_result is None:
                        published_result = PublicationResult(
                            dataset_root=output,
                            manifest_sha256=manifest_sha256,
                            complete_status_sha256=complete_status_sha256,
                            state="published",
                        )
                    raise PublishedBoundaryError(published_result) from exc
                uncertain = PublicationResult(
                    dataset_root=output,
                    manifest_sha256=manifest_sha256,
                    complete_status_sha256=complete_status_sha256,
                    state="publication_uncertain",
                )
                raise PublicationUncertainError(uncertain) from exc
        finally:
            cleanup_error = None
            for descriptor in (staging_fd, parent_fd):
                if descriptor is None:
                    continue
                try:
                    os.close(descriptor)
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            if cleanup_error is not None:
                raise cleanup_error
        if published_result is None:
            raise AssertionError("durable publication has no result")
        try:
            fault("after_destination_parent_fsync")
        except BaseException as exc:
            raise PublishedBoundaryError(published_result) from exc
        return published_result
    except (PublicationUncertainError, PublishedBoundaryError):
        raise
    except BaseException as exc:
        if publication_durable:
            if published_result is None:
                published_result = PublicationResult(
                    dataset_root=output,
                    manifest_sha256=manifest_sha256,
                    complete_status_sha256=complete_status_sha256,
                    state="published",
                )
            raise PublishedBoundaryError(published_result) from exc
        if publication_renamed:
            uncertain = PublicationResult(
                dataset_root=output,
                manifest_sha256=manifest_sha256,
                complete_status_sha256=complete_status_sha256,
                state="publication_uncertain",
            )
            raise PublicationUncertainError(uncertain) from exc
        if (
            durable_complete
            and not os.path.lexists(staging)
            and os.path.lexists(output)
        ):
            uncertain = PublicationResult(
                dataset_root=output,
                manifest_sha256=manifest_sha256,
                complete_status_sha256=complete_status_sha256,
                state="publication_uncertain",
            )
            raise PublicationUncertainError(uncertain) from exc
        if durable_complete:
            raise PublicationStateError("complete_unpublished") from exc
        if completion_write_started:
            raise PublicationStateError("completion_uncertain_unpublished") from exc
        try:
            _replace_with_failed_status(staging, exc, filesystem)
        except BaseException as report_error:
            raise RuntimeError(
                "publication and durable failure reporting both failed: "
                f"{type(exc).__name__}: {exc}; "
                f"{type(report_error).__name__}: {report_error}"
            ) from report_error
        raise PublicationStateError("pre_completion_failed") from exc


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


def no_fault(point: str) -> None:
    del point


def create_unique_staging(plan: ConversionPlan) -> Path:
    output = plan.output_path
    assert_conversion_lock_held(output)
    if os.path.lexists(output):
        raise FileExistsError(f"canonical output already exists: {output}")
    parent_fd = _open_validated_directory(output.parent)
    try:
        for _attempt in range(100):
            staging = output.parent / f".{output.name}.staging-{uuid.uuid4()}"
            name = _validated_basename(staging, "staging root")
            try:
                os.mkdir(name, mode=0o755, dir_fd=parent_fd)
            except FileExistsError:
                continue
            staging_fd = None
            try:
                staging_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
                _validate_descriptor_entry(
                    staging_fd,
                    parent_fd,
                    name,
                    expected_type="directory",
                )
                _validate_descriptor_path(
                    staging_fd, staging, expected_type="directory"
                )
                os.fsync(parent_fd)
                return staging
            finally:
                if staging_fd is not None:
                    os.close(staging_fd)
        raise FileExistsError("could not allocate a unique staging directory")
    finally:
        os.close(parent_fd)


def preflight_report(plan: ConversionPlan) -> dict[str, object]:
    return {
        "schema_version": 1,
        "output_path": str(plan.output_path),
        "selected_episodes": len(plan.episodes),
        "source_frames": sum(item.frame_count for item in plan.episodes),
        "retained_frames": sum(len(item.retained_indices) for item in plan.episodes),
        "media_mode": plan.media_mode,
        "output_profile": asdict(plan.output_media),
        "episodes": [
            {
                "source_episode_index": item.source_episode_index,
                "output_episode_index": item.output_episode_index,
                "source_frames": item.frame_count,
                "retained_frames": len(item.retained_indices),
                "source_media": asdict(item.source_media),
            }
            for item in plan.episodes
        ],
    }


def report_conversion_failure(staging: Path, error: BaseException) -> None:
    print(
        f"CONVERSION_FAILED: preserved staging {staging}: "
        f"{type(error).__name__}: {error}",
        file=sys.stderr,
    )
    if not os.path.lexists(staging):
        return
    if isinstance(error, PublicationStateError):
        return
    try:
        status, _ = _read_canonical_json_file(
            staging / _STATUS_RELATIVE_PATH, "conversion status"
        )
    except (OSError, RuntimeError, ValueError):
        return
    if status.get("state") == "in_progress":
        _replace_with_failed_status(staging, error, PublicationFilesystem())


def certify_published_dataset(
    publication: PublicationResult,
    *,
    psi0_root: Path,
    psi0_commit: str,
    python: Path,
) -> Path:
    from scripts.certify_psi0_dataset import certify_published_dataset as certify

    return certify(
        publication,
        psi0_root=psi0_root,
        psi0_commit=psi0_commit,
        python=python,
    )


def run_conversion(args: argparse.Namespace) -> PublicationResult | ConversionPlan:
    repository = Path(__file__).resolve().parents[1]
    identity = resolve_converter_identity(repository, Path(__file__))
    output = Path(args.out_dir).resolve(strict=False)
    with conversion_lock(output):
        plan = preflight_conversion(args, identity)
        if args.preflight_only:
            print(canonical_json_bytes(preflight_report(plan)).decode(), end="")
            return plan
        staging = create_unique_staging(plan)
        write_in_progress_status(staging, plan)
        try:
            generate_staged_dataset(plan, staging)
            validate_staged_dataset(staging, plan)
            result = publish_staged_dataset(
                staging,
                output,
                identity,
                PublicationFilesystem(),
                no_fault,
            )
        except (PublicationUncertainError, PublishedBoundaryError):
            raise
        except BaseException as exc:
            report_conversion_failure(staging, exc)
            raise
        if result.state != "published":
            raise AssertionError("publish_staged_dataset returned a non-success state")
        if args.certify_psi0_root is not None:
            certify_published_dataset(
                result,
                psi0_root=Path(args.certify_psi0_root),
                psi0_commit=args.certify_psi0_commit,
                python=Path(args.certify_python),
            )
        return result


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


def _lowercase_hex_argument(length: int, label: str) -> Callable[[str], str]:
    def parse(value: str) -> str:
        if re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
            raise argparse.ArgumentTypeError(
                f"{label} must be exactly {length} lowercase hexadecimal characters"
            )
        return value

    return parse


def _positive_integer_argument(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--sim-root")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--skip", type=int, default=60)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument(
        "--total-episodes",
        "--total_episodes",
        dest="total_episodes",
        type=int,
        default=100,
    )
    parser.add_argument("--fps", type=parse_rate, default=Fraction(default_fps, 1))
    parser.add_argument("--video-key", default="observation.rgb_head_stereo_left")
    parser.add_argument("--chunks-size", type=_positive_integer_argument, default=1000)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--certify-psi0-root")
    parser.add_argument(
        "--certify-psi0-commit",
        type=_lowercase_hex_argument(40, "PSI0 commit"),
    )
    parser.add_argument("--certify-python")
    parser.add_argument("--inspect-preserved-staging")
    parser.add_argument("--remove-preserved-staging")
    parser.add_argument(
        "--expected-status-sha256",
        type=_lowercase_hex_argument(64, "status SHA-256"),
    )
    parser.add_argument("--confirm-remove")
    return parser


def _argv_contains(raw: list[str], option: str) -> bool:
    return any(value == option or value.startswith(option + "=") for value in raw)


def _validate_cli_mode(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    raw: list[str],
) -> Literal["conversion", "inspection", "removal"]:
    inspection = args.inspect_preserved_staging is not None
    removal = args.remove_preserved_staging is not None
    if inspection and removal:
        parser.error("inspection and removal modes are mutually exclusive")
    if inspection or removal:
        forbidden = (
            "--sim-root",
            "--skip",
            "--downsample",
            "--total-episodes",
            "--total_episodes",
            "--fps",
            "--video-key",
            "--chunks-size",
            "--preflight-only",
            "--certify-psi0-root",
            "--certify-psi0-commit",
            "--certify-python",
        )
        if any(_argv_contains(raw, option) for option in forbidden):
            parser.error("maintenance mode cannot include conversion options")
        if inspection:
            if (
                args.expected_status_sha256 is not None
                or args.confirm_remove is not None
            ):
                parser.error("inspection does not accept removal authorization")
            return "inspection"
        if args.expected_status_sha256 is None or args.confirm_remove is None:
            parser.error("removal requires status digest and exact confirmation")
        return "removal"

    if args.expected_status_sha256 is not None or args.confirm_remove is not None:
        parser.error("removal authorization requires --remove-preserved-staging")
    if args.sim_root is None:
        parser.error("conversion requires --sim-root")
    certification = (
        args.certify_psi0_root,
        args.certify_psi0_commit,
        args.certify_python,
    )
    if sum(value is not None for value in certification) not in {0, 3}:
        parser.error("certification requires PSI0 root, commit, and Python")
    return "conversion"


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw)
    mode = _validate_cli_mode(parser, args, raw)
    if mode == "inspection":
        report = inspect_preserved_staging(
            args.inspect_preserved_staging,
            args.out_dir,
        )
        print(canonical_json_bytes(report).decode(), end="")
        return 0
    if mode == "removal":
        staging = _absolute_lexical_path(args.remove_preserved_staging, "staging path")
        remove_preserved_staging(
            staging,
            args.out_dir,
            expected_status_sha256=args.expected_status_sha256,
            confirm_remove=args.confirm_remove,
        )
        print(
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "removed": str(staging),
                    "output_path": str(
                        _absolute_lexical_path(args.out_dir, "output path")
                    ),
                }
            ).decode(),
            end="",
        )
        return 0
    run_conversion(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
