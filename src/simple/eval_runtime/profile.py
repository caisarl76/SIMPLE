# src/simple/eval_runtime/profile.py
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
import subprocess

from .contracts import RuntimeBlocked, RuntimeProfile, parse_runtime_profile

PROFILE_PATH = "configs/psi0_h100_eval_runtime_v1.json"


class GitObjectReader:
    def _run(self, *args: str) -> bytes:
        return subprocess.run(
            ["git", *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout

    def head(self) -> str:
        return self._run("rev-parse", "HEAD").decode("ascii").strip()

    def first_parent(self, commit: str) -> str:
        return self._run("rev-parse", f"{commit}^").decode("ascii").strip()

    def changed_paths(self, old: str, new: str) -> list[str]:
        return self._run("diff", "--name-only", old, new).decode("utf-8").splitlines()

    def blob_at(self, commit: str, path: str) -> bytes:
        return self._run("show", f"{commit}:{path}")


def load_approved_profile(git: GitObjectReader) -> RuntimeProfile:
    approval = git.head()
    source = git.first_parent(approval)
    if git.changed_paths(source, approval) != [PROFILE_PATH]:
        raise RuntimeBlocked("PROFILE_APPROVAL_COMMIT", "approval must change one path")
    blob = git.blob_at(approval, PROFILE_PATH)
    try:
        payload = json.loads(blob)
    except Exception as error:
        raise RuntimeBlocked("PROFILE_SCHEMA", str(error)) from error
    return parse_runtime_profile(
        payload,
        blob_sha256=hashlib.sha256(blob).hexdigest(),
        approval_commit=approval,
        source_commit=source,
    )


def exclusive_copy_profile_candidate(source: Path, destination: Path) -> None:
    source_fd = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    parent_fd = os.open(
        destination.parent, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    destination_fd = -1
    created_identity: tuple[int, int] | None = None
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise RuntimeBlocked("PROFILE_CANDIDATE_TYPE", str(source))
        destination_fd = os.open(
            destination.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o644,
            dir_fd=parent_fd,
        )
        destination_stat = os.fstat(destination_fd)
        created_identity = (destination_stat.st_dev, destination_stat.st_ino)
        while chunk := os.read(source_fd, 1024 * 1024):
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
        os.fchmod(destination_fd, 0o644)
        os.fsync(destination_fd)
        path_stat = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        if (path_stat.st_dev, path_stat.st_ino) != created_identity:
            raise RuntimeBlocked("PROFILE_PUBLICATION_RACE", destination.name)
        os.fsync(parent_fd)
    except Exception:
        if created_identity is not None:
            try:
                path_stat = os.stat(
                    destination.name, dir_fd=parent_fd, follow_symlinks=False
                )
                if (path_stat.st_dev, path_stat.st_ino) == created_identity:
                    os.unlink(destination.name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        os.close(parent_fd)
        os.close(source_fd)
