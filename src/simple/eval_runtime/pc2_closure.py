from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .canonical import canonical_json_bytes, sha256_bytes

BASE_PHASES = (
    "BASE_ALLOCATED",
    "BASE_COPIED",
    "BASE_METADATA_NORMALIZED",
    "BASE_FINAL_RENAMED",
    "BASE_RECEIPT_CREATED",
    "BASE_COMPLETE",
)
CLOSURE_PHASES = (
    "ALLOCATED",
    "SOURCE",
    "EPISODE_DATA",
    "VENV",
    "TASK_DATA_COPY",
    "HSSD_NORMALIZED",
    "PAYLOAD_READY",
    "INSTALLING",
    "FINAL_RENAMED",
    "RECEIPT_CREATED",
    "COMPLETE",
)

HSSD_RESULT_KEYS = {
    "schema_version",
    "requirements_sha256",
    "pc2_closure_id",
    "normalizer_policy_version",
    "usd_sdf_tool_identity_sha256",
    "source_layers",
    "normalized_layers",
    "old_to_new_mappings",
}


@dataclass(frozen=True, slots=True)
class ClosureDescriptor:
    schema_version: int
    simple_commit: str
    simple_root_tree: str
    recursive_gitlinks_sha256: str
    git_object_manifest_sha256: str
    episode_data_manifest_sha256: str
    asset_requirements_sha256: str
    task_data_source_manifest_sha256: str
    base_python_tree_sha256: str
    mode_policy_version: int
    hssd_policy_version: int
    installer_sha256: str
    locked_inputs_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("CLOSURE_DESCRIPTOR_VERSION")
        if len(self.simple_commit) != 40 or len(self.simple_root_tree) != 40:
            raise ValueError("CLOSURE_DESCRIPTOR_GIT_ID")
        for field in (
            self.recursive_gitlinks_sha256,
            self.git_object_manifest_sha256,
            self.episode_data_manifest_sha256,
            self.asset_requirements_sha256,
            self.task_data_source_manifest_sha256,
            self.base_python_tree_sha256,
            self.installer_sha256,
            self.locked_inputs_sha256,
        ):
            if len(field) != 64 or any(ch not in "0123456789abcdef" for ch in field):
                raise ValueError("CLOSURE_DESCRIPTOR_SHA256")


@dataclass(frozen=True, slots=True)
class ConstructionRecord:
    schema_version: int
    construction_attempt: int
    attempt_status: str
    operation_token_sha256: str
    phase: str
    cleanup_required: bool
    pending_action: str | None

    @classmethod
    def from_dict(cls, data: object) -> "ConstructionRecord":
        keys = {
            "schema_version",
            "construction_attempt",
            "attempt_status",
            "operation_token_sha256",
            "phase",
            "cleanup_required",
            "pending_action",
        }
        if type(data) is not dict or set(data) != keys:
            raise ValueError("CONSTRUCTION_RECORD_SCHEMA")
        if data["schema_version"] != 1 or data["phase"] not in CLOSURE_PHASES:
            raise ValueError("CONSTRUCTION_RECORD_VALUE")
        if (
            type(data["construction_attempt"]) is not int
            or data["construction_attempt"] < 1
        ):
            raise ValueError("CONSTRUCTION_RECORD_ATTEMPT")
        token = data["operation_token_sha256"]
        if (
            type(token) is not str
            or len(token) != 64
            or any(ch not in "0123456789abcdef" for ch in token)
        ):
            raise ValueError("CONSTRUCTION_RECORD_TOKEN")
        cleanup_required = data["cleanup_required"]
        if type(cleanup_required) is not bool or cleanup_required != (
            data["phase"] != "COMPLETE"
        ):
            raise ValueError("CONSTRUCTION_RECORD_CLEANUP")
        if (
            data["pending_action"] is not None
            and type(data["pending_action"]) is not str
        ):
            raise ValueError("CONSTRUCTION_RECORD_PENDING")
        return cls(**data)


def compute_closure_id(descriptor: ClosureDescriptor) -> str:
    return sha256_bytes(canonical_json_bytes(asdict(descriptor)))


def load_construction_record(path: Path) -> ConstructionRecord:
    payload = Path(path).read_bytes()
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("CONSTRUCTION_RECORD_JSON") from exc
    if payload != canonical_json_bytes(decoded) + b"\n":
        raise ValueError("CONSTRUCTION_RECORD_CANONICAL")
    return ConstructionRecord.from_dict(decoded)


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("short write while persisting construction record")
        offset += written


def _persist_construction_record(path: Path, record: ConstructionRecord) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary = f".{path.name}.{uuid.uuid4().hex}.tmp"
    fd = -1
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        _write_all(fd, canonical_json_bytes(asdict(record)) + b"\n")
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temporary, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)


class Pc2ClosureBuilder:
    """Write-ahead state machine for the unprivileged closure build phases.

    Privileged publication is deliberately supplied as callbacks. This keeps the
    builder unable to create root-owned journals while ensuring every mutation
    has a durable pending action and every completed phase is fsynced.
    """

    def __init__(
        self,
        owner_record_path: Path,
        *,
        construction_attempt: int,
        operation_token_sha256: str,
    ) -> None:
        if construction_attempt < 1:
            raise ValueError("CONSTRUCTION_ATTEMPT")
        if len(operation_token_sha256) != 64 or any(
            ch not in "0123456789abcdef" for ch in operation_token_sha256
        ):
            raise ValueError("CONSTRUCTION_TOKEN")
        self._owner_record_path = Path(owner_record_path)
        self._construction_attempt = construction_attempt
        self._operation_token_sha256 = operation_token_sha256
        if self._owner_record_path.exists():
            record = load_construction_record(self._owner_record_path)
            if (
                record.construction_attempt != construction_attempt
                or record.operation_token_sha256 != operation_token_sha256
            ):
                raise ValueError("CONSTRUCTION_OWNER_MISMATCH")

    def _record(
        self,
        *,
        phase: str,
        pending_action: str | None,
        complete: bool = False,
    ) -> ConstructionRecord:
        record = ConstructionRecord(
            schema_version=1,
            construction_attempt=self._construction_attempt,
            attempt_status="complete" if complete else "active",
            operation_token_sha256=self._operation_token_sha256,
            phase=phase,
            cleanup_required=not complete,
            pending_action=pending_action,
        )
        _persist_construction_record(self._owner_record_path, record)
        return record

    def _step(
        self,
        *,
        action_name: str,
        expected_phase: str | None,
        resulting_phase: str,
        action: Callable[[], object],
        complete: bool = False,
    ) -> None:
        if self._owner_record_path.exists():
            current = load_construction_record(self._owner_record_path)
            if current.pending_action is not None:
                raise RuntimeError("CONSTRUCTION_RECOVERY_REQUIRED")
            if current.phase != expected_phase:
                raise RuntimeError(
                    f"CONSTRUCTION_PHASE:expected={expected_phase}:actual={current.phase}"
                )
            pending_phase = current.phase
        else:
            if expected_phase is not None:
                raise RuntimeError("CONSTRUCTION_RECORD_MISSING")
            pending_phase = resulting_phase
        self._record(phase=pending_phase, pending_action=action_name)
        action()
        self._record(phase=resulting_phase, pending_action=None, complete=complete)

    def allocate(self, action: Callable[[], object]) -> None:
        self._step(
            action_name="allocate",
            expected_phase=None,
            resulting_phase="ALLOCATED",
            action=action,
        )

    def copy_git_objects(self, action: Callable[[], object]) -> None:
        self._step(
            action_name="copy_git_objects",
            expected_phase="ALLOCATED",
            resulting_phase="SOURCE",
            action=action,
        )

    def copy_episode_data(self, action: Callable[[], object]) -> None:
        self._step(
            action_name="copy_episode_data",
            expected_phase="SOURCE",
            resulting_phase="EPISODE_DATA",
            action=action,
        )

    def build_relative_venv(self, action: Callable[[], object]) -> None:
        self._step(
            action_name="build_relative_venv",
            expected_phase="EPISODE_DATA",
            resulting_phase="VENV",
            action=action,
        )

    def copy_task_assets(self, action: Callable[[], object]) -> None:
        self._step(
            action_name="copy_task_assets",
            expected_phase="VENV",
            resulting_phase="TASK_DATA_COPY",
            action=action,
        )

    def normalize_hssd_with_staging_venv(self, action: Callable[[], object]) -> None:
        self._step(
            action_name="normalize_hssd_with_staging_venv",
            expected_phase="TASK_DATA_COPY",
            resulting_phase="HSSD_NORMALIZED",
            action=action,
        )

    def probe_payload(self, action: Callable[[], object]) -> None:
        self._step(
            action_name="probe_payload",
            expected_phase="HSSD_NORMALIZED",
            resulting_phase="PAYLOAD_READY",
            action=action,
        )

    def authorize_install(self, action: Callable[[], object]) -> None:
        self._step(
            action_name="authorize_install",
            expected_phase="PAYLOAD_READY",
            resulting_phase="INSTALLING",
            action=action,
        )

    def mirror_final_renamed(self, action: Callable[[], object]) -> None:
        self._step(
            action_name="mirror_final_renamed",
            expected_phase="INSTALLING",
            resulting_phase="FINAL_RENAMED",
            action=action,
        )

    def mirror_receipt_created(self, action: Callable[[], object]) -> None:
        self._step(
            action_name="mirror_receipt_created",
            expected_phase="FINAL_RENAMED",
            resulting_phase="RECEIPT_CREATED",
            action=action,
        )

    def mark_complete(self, action: Callable[[], object]) -> None:
        self._step(
            action_name="mark_complete",
            expected_phase="RECEIPT_CREATED",
            resulting_phase="COMPLETE",
            action=action,
            complete=True,
        )
