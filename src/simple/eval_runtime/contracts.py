# src/simple/eval_runtime/contracts.py
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Callable, Mapping

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class RuntimeBlocked(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


class TerminalState(str, Enum):
    CLEAN = "CLEAN"
    FAILED = "FAILED"
    PROVENANCE_BLOCKED = "PROVENANCE_BLOCKED"
    STALE_OWNED_BLOCKED = "STALE_OWNED_BLOCKED"
    FOREIGN_BLOCKED = "FOREIGN_BLOCKED"


@dataclass(frozen=True, slots=True)
class FieldRule:
    kind: str
    example: object
    validator: Callable[[object, str], object]


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    values: Mapping[str, object]
    blob_sha256: str
    approval_commit: str
    source_commit: str


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    argv_sha256: str


@dataclass(frozen=True, slots=True)
class OwnedDeadline:
    expires_at: float

    def remaining(self, clock: Callable[[], float] = time.monotonic) -> float:
        return max(0.0, self.expires_at - clock())


def validate_identifier(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or value in {".", ".."}
        or not IDENTIFIER.fullmatch(value)
    ):
        raise RuntimeBlocked("INVALID_IDENTIFIER", field)
    return value


def validate_relative_path(value: object, *, field: str) -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value or value.startswith("/"):
        raise RuntimeBlocked("INVALID_PATH", field)
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise RuntimeBlocked("INVALID_PATH", field)
    return PurePosixPath(*parts)


def parse_runtime_profile(
    data: object, *, blob_sha256: str, approval_commit: str, source_commit: str
) -> RuntimeProfile:
    if type(data) is not dict or set(data) != set(PROFILE_FIELDS):
        raise RuntimeBlocked("PROFILE_SCHEMA", "exact 87-key set required")
    parsed: dict[str, object] = {}
    for name, rule in PROFILE_FIELDS.items():
        try:
            parsed[name] = rule.validator(data[name], name)
        except RuntimeBlocked:
            raise
        except Exception as error:
            raise RuntimeBlocked(f"PROFILE_TYPE:{name}", str(error)) from error
    if parsed["simple_source_commit"] != source_commit:
        raise RuntimeBlocked(
            "PROFILE_SOURCE_BINDING", "simple_source_commit must equal approval parent"
        )
    _validate_profile_relationships(parsed)
    return RuntimeProfile(
        MappingProxyType(parsed), blob_sha256, approval_commit, source_commit
    )


# Populate PROFILE_FIELDS directly from the approved 87-key table.
def _exact_int(value: object, field: str, expected: int | None = None) -> int:
    if (
        type(value) is not int
        or value <= 0
        or (expected is not None and value != expected)
    ):
        raise RuntimeBlocked(f"PROFILE_TYPE:{field}", "positive exact integer required")
    return value


def _string(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise RuntimeBlocked(f"PROFILE_TYPE:{field}", "nonempty string required")
    return value


def _hex(value: object, field: str, pattern: re.Pattern[str]) -> str:
    text = _string(value, field)
    if pattern.fullmatch(text) is None:
        raise RuntimeBlocked(f"PROFILE_TYPE:{field}", "invalid hexadecimal identity")
    return text


def _absolute(value: object, field: str) -> str:
    text = _string(value, field)
    if not text.startswith("/") or "/../" in f"/{text}/" or "/./" in f"/{text}/":
        raise RuntimeBlocked(
            f"PROFILE_TYPE:{field}", "normalized absolute path required"
        )
    return text


def _image_id(value: object, field: str) -> str:
    text = _string(value, field)
    if not text.startswith("sha256:") or HEX64.fullmatch(text[7:]) is None:
        raise RuntimeBlocked(
            f"PROFILE_TYPE:{field}", "digest-qualified image ID required"
        )
    return text


def _image_reference(value: object, field: str) -> str:
    text = _string(value, field)
    prefix = "pytorch/pytorch@sha256:"
    if not text.startswith(prefix) or HEX64.fullmatch(text[len(prefix) :]) is None:
        raise RuntimeBlocked(
            f"PROFILE_TYPE:{field}", "digest-qualified image reference required"
        )
    return text


FIELD_KINDS = {
    "schema_version": "schema_one",
    "profile_id": "string",
    "image_reference": "image_reference",
    "image_id": "image_id",
    "source_snapshot_host_path": "absolute",
    "source_tree_root_sha256": "hex64",
    "source_entry_count": "positive_int",
    "source_regular_file_bytes": "positive_int",
    "source_completion_sha256": "hex64",
    "source_installer_receipt_sha256": "hex64",
    "source_mode_policy_version": "positive_int",
    "package_freeze_sha256": "hex64",
    "python_version": "string",
    "torch_version": "string",
    "torch_cuda_version": "string",
    "checkpoint_snapshot_host_path": "absolute",
    "checkpoint_weight_relative_path": "relative",
    "checkpoint_entry_count": "positive_int",
    "checkpoint_regular_file_bytes": "positive_int",
    "checkpoint_size": "positive_int",
    "checkpoint_sha256": "hex64",
    "checkpoint_tree_root_sha256": "hex64",
    "checkpoint_completion_sha256": "hex64",
    "checkpoint_installer_receipt_sha256": "hex64",
    "checkpoint_mode_policy_version": "positive_int",
    "checkpoint_tracer_path": "absolute",
    "checkpoint_tracer_sha256": "hex64",
    "checkpoint_tracer_version": "string",
    "checkpoint_tracer_argv_sha256": "hex64",
    "checkpoint_tracer_probe_sha256": "hex64",
    "checkpoint_tracer_probe_sentinel_sha256": "hex64",
    "checkpoint_tracer_probe_argv_sha256": "hex64",
    "container_seccomp_profile_host_path": "absolute",
    "container_seccomp_profile_sha256": "hex64",
    "container_security_contract_sha256": "hex64",
    "server_source_sha256": "hex64",
    "launcher_source_sha256": "hex64",
    "h100_roots_identity_sha256": "hex64",
    "h100_input_installer_sha256": "hex64",
    "simple_source_commit": "hex40",
    "simple_root_tree": "hex40",
    "recursive_gitlinks_sha256": "hex64",
    "pc2_closure_id": "hex64",
    "pc2_input_host_path": "absolute",
    "pc2_source_tree_sha256": "hex64",
    "pc2_venv_tree_sha256": "hex64",
    "pc2_base_python_host_path": "absolute",
    "pc2_base_python_tree_sha256": "hex64",
    "pc2_base_python_loader_relative_path": "relative",
    "pc2_base_python_loader_sha256": "hex64",
    "pc2_base_python_completion_sha256": "hex64",
    "pc2_base_python_root_identity_sha256": "hex64",
    "pc2_base_python_installer_receipt_sha256": "hex64",
    "pc2_package_freeze_sha256": "hex64",
    "pc2_import_origins_sha256": "hex64",
    "pc2_native_closure_sha256": "hex64",
    "pc2_episode_data_tree_sha256": "hex64",
    "pc2_task_assets_tree_sha256": "hex64",
    "pc2_asset_requirements_sha256": "hex64",
    "pc2_asset_normalization_results_sha256": "hex64",
    "pc2_runtime_identity_sha256": "hex64",
    "pc2_runtime_identity_sidecar_sha256": "hex64",
    "pc2_closure_completion_sha256": "hex64",
    "pc2_closure_root_identity_sha256": "hex64",
    "pc2_roots_identity_sha256": "hex64",
    "pc2_input_installer_sha256": "hex64",
    "pc2_installer_config_sha256": "hex64",
    "pc2_installer_service_unit_sha256": "hex64",
    "pc2_installer_socket_unit_sha256": "hex64",
    "pc2_installer_receipt_sha256": "hex64",
    "pc2_runner_launcher_sha256": "hex64",
    "pc2_runner_config_sha256": "hex64",
    "pc2_runner_service_unit_sha256": "hex64",
    "pc2_runner_socket_unit_sha256": "hex64",
    "pc2_runner_sandbox_contract_sha256": "hex64",
    "pc2_policy_relay_sha256": "hex64",
    "pc2_policy_relay_contract_sha256": "hex64",
    "pc2_evaluator_uid": "positive_int",
    "pc2_evaluator_gid": "positive_int",
    "pc2_mode_policy_version": "positive_int",
    "pc2_python_version": "string",
    "pc2_torch_version": "string",
    "pc2_torch_cuda_version": "string",
    "pc2_mujoco_version": "string",
    "pc2_isaac_version": "string",
    "pc2_nvidia_driver_version": "string",
    "pc2_cuda_driver_version": "string",
}


def _field_rule(kind: str) -> FieldRule:
    if kind == "schema_one":
        return FieldRule(kind, 1, lambda value, field: _exact_int(value, field, 1))
    if kind == "positive_int":
        return FieldRule(kind, 1, _exact_int)
    if kind == "hex64":
        return FieldRule(kind, "a" * 64, lambda value, field: _hex(value, field, HEX64))
    if kind == "hex40":
        return FieldRule(kind, "a" * 40, lambda value, field: _hex(value, field, HEX40))
    if kind == "absolute":
        return FieldRule(kind, "/protected/value", _absolute)
    if kind == "relative":
        return FieldRule(
            kind,
            "weights/value.bin",
            lambda value, field: str(validate_relative_path(value, field=field)),
        )
    if kind == "image_id":
        return FieldRule(
            kind, "sha256:" + "a" * 64, lambda value, field: _image_id(value, field)
        )
    if kind == "image_reference":
        return FieldRule(
            kind,
            "pytorch/pytorch@sha256:" + "a" * 64,
            lambda value, field: _image_reference(value, field),
        )
    return FieldRule(kind, "value", _string)


PROFILE_FIELDS = MappingProxyType(
    {name: _field_rule(kind) for name, kind in FIELD_KINDS.items()}
)


def _validate_profile_relationships(values: dict[str, object]) -> None:
    required_suffixes = {
        "source_snapshot_host_path": f"/server/{values['source_tree_root_sha256']}",
        "checkpoint_snapshot_host_path": f"/checkpoint/{values['checkpoint_tree_root_sha256']}",
        "pc2_input_host_path": f"/{values['pc2_closure_id']}",
        "pc2_base_python_host_path": f"/{values['pc2_base_python_tree_sha256']}",
    }
    for field, suffix in required_suffixes.items():
        if not str(values[field]).endswith(suffix):
            raise RuntimeBlocked(f"PROFILE_RELATION:{field}", suffix)
    source = str(values["source_snapshot_host_path"]).rstrip("/")
    seccomp = str(values["container_seccomp_profile_host_path"])
    if not seccomp.startswith(source + "/"):
        raise RuntimeBlocked(
            "PROFILE_RELATION:container_seccomp_profile_host_path", source
        )
