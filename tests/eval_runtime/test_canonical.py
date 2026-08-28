# tests/eval_runtime/test_canonical.py
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from simple.eval_runtime.canonical import (
    ManifestPolicy,
    atomic_write_new_json,
    canonical_json_bytes,
    canonical_tree_manifest,
)
from simple.eval_runtime.contracts import RuntimeBlocked


def test_canonical_json_is_sorted_compact_and_rejects_nan() -> None:
    assert (
        canonical_json_bytes({"z": 1, "a": [True, None]}) == b'{"a":[true,null],"z":1}'
    )
    with pytest.raises(RuntimeBlocked, match="NON_CANONICAL_JSON"):
        canonical_json_bytes({"x": float("nan")})


def test_tree_manifest_is_mode_inclusive_and_byte_sorted(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "z").write_bytes(b"z")
    (root / "a").mkdir()
    (root / "a" / "x").write_bytes(b"xx")
    os.chmod(root / "z", 0o444)
    os.chmod(root / "a" / "x", 0o555)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        manifest = canonical_tree_manifest(root_fd, ManifestPolicy())
    finally:
        os.close(root_fd)
    lines = [json.loads(line) for line in manifest.bytes_value.splitlines()]
    assert [line["path"] for line in lines] == ["a", "a/x", "z"]
    assert lines[1]["mode"] == "0555"
    assert lines[2]["mode"] == "0444"
    assert manifest.regular_file_bytes == 3


@pytest.mark.parametrize("kind", ["absolute", "escape", "fifo", "socket"])
def test_manifest_rejects_escaping_links_and_special_files(
    tmp_path: Path, kind: str
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    if kind == "absolute":
        (root / "bad").symlink_to("/tmp/outside")
    elif kind == "escape":
        (root / "bad").symlink_to("../outside")
    elif kind == "fifo":
        os.mkfifo(root / "bad")
    else:
        import socket

        sock = socket.socket(socket.AF_UNIX)
        sock.bind(str(root / "bad"))
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with pytest.raises(RuntimeBlocked, match="TREE_ENTRY"):
            canonical_tree_manifest(root_fd, ManifestPolicy())
    finally:
        os.close(root_fd)
        if kind == "socket":
            sock.close()


def test_atomic_write_refuses_existing_target(tmp_path: Path) -> None:
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        first = atomic_write_new_json(parent_fd, "record.json", {"schema_version": 1})
        assert first.size > 0
        with pytest.raises(FileExistsError):
            atomic_write_new_json(parent_fd, "record.json", {"schema_version": 1})
    finally:
        os.close(parent_fd)
