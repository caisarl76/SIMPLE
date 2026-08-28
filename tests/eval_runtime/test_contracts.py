# tests/eval_runtime/test_contracts.py
from __future__ import annotations

from pathlib import PurePosixPath

import pytest

from simple.eval_runtime.contracts import (
    PROFILE_FIELDS,
    RuntimeBlocked,
    parse_runtime_profile,
    validate_identifier,
    validate_relative_path,
)

from .conftest import digest


def make_profile(*, source_commit: str = "b" * 40) -> dict[str, object]:
    value: dict[str, object] = {}
    for name, rule in PROFILE_FIELDS.items():
        value[name] = rule.example
    value.update(
        source_snapshot_host_path=f"/inputs/server/{value['source_tree_root_sha256']}",
        checkpoint_snapshot_host_path=f"/inputs/checkpoint/{value['checkpoint_tree_root_sha256']}",
        pc2_input_host_path=f"/inputs/pc2/{value['pc2_closure_id']}",
        pc2_base_python_host_path=f"/inputs/base/{value['pc2_base_python_tree_sha256']}",
    )
    value["container_seccomp_profile_host_path"] = (
        f"{value['source_snapshot_host_path']}/runtime-tools/psi0_h100_eval_seccomp_v1.json"
    )
    value["simple_source_commit"] = source_commit
    return value


@pytest.mark.parametrize(
    "value",
    [".", "..", "", "a/b", "a\\b", "é", " a", "a\x00b", "-leading"],
)
def test_identifier_rejects_dot_segments_and_unsafe_bytes(value: str) -> None:
    with pytest.raises(RuntimeBlocked, match="INVALID_IDENTIFIER"):
        validate_identifier(value, field="run_id")


def test_identifier_accepts_reviewed_alphabet() -> None:
    assert validate_identifier("run_20260826.a-1", field="run_id") == "run_20260826.a-1"


@pytest.mark.parametrize("value", ["", ".", "..", "a/../b", "/a", "a//b", "a/./b"])
def test_relative_path_has_no_empty_or_dot_segments(value: str) -> None:
    with pytest.raises(RuntimeBlocked, match="INVALID_PATH"):
        validate_relative_path(value, field="checkpoint_weight_relative_path")


def test_relative_path_returns_normalized_posix_path() -> None:
    assert validate_relative_path(
        "weights/step-40000.pt", field="weight"
    ) == PurePosixPath("weights/step-40000.pt")


def test_profile_requires_the_exact_87_keys_and_exact_types() -> None:
    payload = make_profile()
    parsed = parse_runtime_profile(
        payload,
        blob_sha256=digest(b"profile"),
        approval_commit="a" * 40,
        source_commit="b" * 40,
    )
    assert set(parsed.values) == set(PROFILE_FIELDS)
    assert len(PROFILE_FIELDS) == 87

    for name in tuple(payload):
        missing = dict(payload)
        missing.pop(name)
        with pytest.raises(RuntimeBlocked, match="PROFILE_SCHEMA"):
            parse_runtime_profile(
                missing,
                blob_sha256=digest(b"profile"),
                approval_commit="a" * 40,
                source_commit="b" * 40,
            )


def test_profile_rejects_boolean_for_every_integer_field() -> None:
    payload = make_profile()
    for name, rule in PROFILE_FIELDS.items():
        if rule.kind == "positive_int" or rule.kind == "schema_one":
            malformed = dict(payload)
            malformed[name] = True
            with pytest.raises(RuntimeBlocked, match=f"PROFILE_TYPE:{name}"):
                parse_runtime_profile(
                    malformed,
                    blob_sha256=digest(b"profile"),
                    approval_commit="a" * 40,
                    source_commit="b" * 40,
                )
