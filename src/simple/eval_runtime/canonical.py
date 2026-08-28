# src/simple/eval_runtime/canonical.py
from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from dataclasses import dataclass
from typing import Mapping

from .contracts import RuntimeBlocked


@dataclass(frozen=True, slots=True)
class ManifestPolicy:
    allowed_absolute_prefixes: tuple[str, ...] = ()
    excluded_names: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class TreeManifest:
    bytes_value: bytes
    root_sha256: str
    entry_count: int
    regular_file_bytes: int


@dataclass(frozen=True, slots=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    sha256: str


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RuntimeBlocked("NON_CANONICAL_JSON", str(error)) from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_tree_manifest(root_fd: int, policy: ManifestPolicy) -> TreeManifest:
    entries: list[dict[str, object]] = []
    regular_bytes = 0

    def visit(parent_fd: int, prefix: tuple[str, ...]) -> None:
        nonlocal regular_bytes
        names = sorted(os.listdir(parent_fd), key=lambda name: os.fsencode(name))
        for name in names:
            if name in policy.excluded_names:
                continue
            relative = "/".join((*prefix, name))
            info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISDIR(info.st_mode):
                entries.append(
                    {"kind": "directory", "path": relative, "mode": f"{mode:04o}"}
                )
                child_fd = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd
                )
                try:
                    visit(child_fd, (*prefix, name))
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(info.st_mode):
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
                try:
                    payload = bytearray()
                    while block := os.read(fd, 1024 * 1024):
                        payload.extend(block)
                finally:
                    os.close(fd)
                regular_bytes += info.st_size
                entries.append(
                    {
                        "kind": "file",
                        "path": relative,
                        "mode": f"{mode:04o}",
                        "bytes": info.st_size,
                        "sha256": sha256_bytes(bytes(payload)),
                    }
                )
            elif stat.S_ISLNK(info.st_mode):
                target = os.readlink(name, dir_fd=parent_fd)
                if target.startswith("/") and not any(
                    target == p or target.startswith(f"{p}/")
                    for p in policy.allowed_absolute_prefixes
                ):
                    raise RuntimeBlocked("TREE_ENTRY", relative)
                if not target.startswith("/") and any(
                    part == ".." for part in target.split("/")
                ):
                    raise RuntimeBlocked("TREE_ENTRY", relative)
                entries.append({"kind": "symlink", "path": relative, "target": target})
            else:
                raise RuntimeBlocked("TREE_ENTRY", relative)

    visit(root_fd, ())
    encoded = b"".join(canonical_json_bytes(entry) + b"\n" for entry in entries)
    return TreeManifest(encoded, sha256_bytes(encoded), len(entries), regular_bytes)


def atomic_write_new_json(
    parent_fd: int, name: str, value: object, mode: int = 0o444
) -> FileIdentity:
    payload = canonical_json_bytes(value) + b"\n"
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    fd = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
    try:
        os.write(fd, payload)
        os.fsync(fd)
        os.fchmod(fd, mode)
        info = os.fstat(fd)
        os.link(
            temporary,
            name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.fsync(parent_fd)
    finally:
        os.close(fd)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
    return FileIdentity(info.st_dev, info.st_ino, info.st_size, sha256_bytes(payload))


def append_hash_chained_jsonl(
    fd: int, value: Mapping[str, object], previous: str | None
) -> str:
    record = dict(value)
    record["previous_sha256"] = previous
    encoded = canonical_json_bytes(record) + b"\n"
    os.write(fd, encoded)
    os.fsync(fd)
    return sha256_bytes(encoded)
