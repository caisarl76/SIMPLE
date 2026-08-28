# tests/eval_runtime/test_profile.py
from __future__ import annotations

from pathlib import Path

import pytest

from simple.eval_runtime.contracts import RuntimeBlocked
from simple.eval_runtime.profile import (
    GitObjectReader,
    exclusive_copy_profile_candidate,
    load_approved_profile,
)

from .test_contracts import make_profile
from .conftest import write_json


class FakeGit(GitObjectReader):
    def __init__(self, source: str, approval: str, changed: list[str], blob: bytes):
        self.source = source
        self.approval = approval
        self.changed = changed
        self.blob = blob

    def head(self) -> str:
        return self.approval

    def first_parent(self, commit: str) -> str:
        assert commit == self.approval
        return self.source

    def changed_paths(self, old: str, new: str) -> list[str]:
        assert (old, new) == (self.source, self.approval)
        return self.changed

    def blob_at(self, commit: str, path: str) -> bytes:
        assert commit == self.approval
        assert path == "configs/psi0_h100_eval_runtime_v1.json"
        return self.blob


def test_profile_commit_may_change_only_the_profile(tmp_path: Path) -> None:
    profile_path = write_json(tmp_path / "profile.json", make_profile())
    blob = profile_path.read_bytes()
    git = FakeGit("b" * 40, "a" * 40, ["configs/psi0_h100_eval_runtime_v1.json"], blob)
    loaded = load_approved_profile(git)
    assert loaded.source_commit == "b" * 40
    assert loaded.approval_commit == "a" * 40


def test_profile_simple_source_commit_must_equal_approval_parent(
    tmp_path: Path,
) -> None:
    profile_path = write_json(
        tmp_path / "profile.json", make_profile(source_commit="c" * 40)
    )
    git = FakeGit(
        "b" * 40,
        "a" * 40,
        ["configs/psi0_h100_eval_runtime_v1.json"],
        profile_path.read_bytes(),
    )
    with pytest.raises(RuntimeBlocked, match="PROFILE_SOURCE_BINDING"):
        load_approved_profile(git)


@pytest.mark.parametrize(
    "changed",
    [[], ["README.md"], ["configs/psi0_h100_eval_runtime_v1.json", "README.md"]],
)
def test_profile_commit_rejects_any_other_diff(changed: list[str]) -> None:
    blob = str(make_profile()).encode()
    git = FakeGit("b" * 40, "a" * 40, changed, blob)
    with pytest.raises(RuntimeBlocked, match="PROFILE_APPROVAL_COMMIT"):
        load_approved_profile(git)


def test_candidate_publication_is_exclusive_and_byte_identical(tmp_path: Path) -> None:
    source = tmp_path / "candidate.json"
    destination = tmp_path / "approved.json"
    source.write_bytes(b'{"schema_version":1}\n')
    exclusive_copy_profile_candidate(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode & 0o777 == 0o644
    with pytest.raises(FileExistsError):
        exclusive_copy_profile_candidate(source, destination)
