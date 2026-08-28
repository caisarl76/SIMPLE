# tests/eval_runtime/fakes.py
from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import array
import fcntl
import stat
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np


class InstallerHandoffHarness:
    def __init__(self, *, boundary: str):
        self.boundary = boundary
        self.duplicate_acquired = False
        self.persisted = False
        self.closed = False

    def queue_duplicate(self) -> None:
        self.duplicate_acquired = False

    def advance_to_boundary(self) -> None:
        self.duplicate_acquired = False

    def manager_persist_transition(self) -> None:
        self.persisted = True

    def manager_close_handoff(self) -> None:
        assert self.persisted
        self.closed = True

    def one_duplicate_acquires_after_revalidation(self) -> bool:
        self.duplicate_acquired = self.persisted and self.closed
        return self.duplicate_acquired


class RecoveryHarness:
    def __init__(self, *, crash: str | None = None):
        self.crash = crash
        self.install_request_bytes = (
            b'{"operation":"install_base_python","token":"fixed"}'
        )
        self.replayed_install_bytes = b""
        self.install_reference = "install:1"
        self._authorizations: list[SimpleNamespace] = []
        self._results: list[SimpleNamespace] = []
        self._journal = False
        self.root_journal_accesses = 0
        self.final_digest_paths_examined: list[str] = []
        self.unrelated_protected_objects_preserved = True
        self.manager_journal_writes = 0

    def append_recovery_generation(self) -> SimpleNamespace:
        number = len(self._authorizations) + 1
        previous = (
            self._authorizations[-1].canonical_entry_sha256
            if self._authorizations
            else None
        )
        canonical_entry = {
            "recovery_generation": number,
            "previous_entry_sha256": previous,
            "request_digest": f"{number:064x}",
            "authorization_sha256": f"{number + 100:064x}",
        }
        entry = SimpleNamespace(
            **canonical_entry,
            active_reference=f"recovery:{number}",
            canonical_entry_sha256=hashlib.sha256(
                json.dumps(
                    canonical_entry, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        )
        self._authorizations.append(entry)
        return entry

    def recover(self, generation: SimpleNamespace) -> SimpleNamespace:
        self.root_journal_accesses += 1
        previous = self._results[-1].result_sha256 if self._results else None
        result_entry = SimpleNamespace(
            recovery_generation=generation.recovery_generation,
            previous_result_sha256=previous,
            result_sha256=f"{generation.recovery_generation + 200:064x}",
        )
        self._results.append(result_entry)
        if self.crash == "mutation_child_before_prepared":
            return SimpleNamespace(
                status="ABORTED_BEFORE_PREPARED", result=result_entry
            )
        return SimpleNamespace(
            status="RECOVERED" if self._journal else "NO_JOURNAL_NO_MUTATION",
            result=result_entry,
        )

    def reactivate_install(self, generation: SimpleNamespace) -> None:
        assert generation is self._authorizations[-1]
        self.replayed_install_bytes = self.install_request_bytes

    def service_create_journal_then_drop_transport(self) -> None:
        self._journal = True

    def reject_install_against(self, reference: str) -> str:
        return "AUTHORIZATION_KIND" if reference.startswith("recovery:") else "accepted"

    def reject_recovery_against(self, reference: str) -> str:
        return "AUTHORIZATION_KIND" if reference.startswith("install:") else "accepted"


def fixture_publication_authorization(
    request: dict[str, object],
    *,
    previous: dict[str, object] | None,
    previous_transition: dict[str, object] | None = None,
    previous_owner_record: bytes | None = None,
) -> dict[str, object]:
    from simple.eval_runtime.installer_client import (
        encode_authorization_owner_projection,
        encode_authorization_owner_record,
        encode_owner_transition,
        encode_pending_publication_authorization,
        encode_publication_authorization,
        validate_owner_transition_against_records,
    )

    prior_transition_or_entry = (
        previous_transition
        if previous_transition is not None
        else None
        if previous is None
        else previous["owner_transition"]
    )
    prior_transition = (
        None
        if prior_transition_or_entry is None
        else prior_transition_or_entry.get("transition", prior_transition_or_entry)
    )
    if previous_owner_record is None:
        if previous is not None:
            raise RuntimeError("INSTALLER_TEST_PREVIOUS_OWNER_REQUIRED")
        previous_owner_record = encode_authorization_owner_record(
            _fixture_unactivated_owner_mapping(request)
        )
    previous_owner = json.loads(previous_owner_record)
    sequence = previous_owner["authorization_sequence"] + 1
    provisional_authorization = encode_pending_publication_authorization(
        request=request,
        authorization_sequence=sequence,
        previous_authorization=previous,
        recovery_reason=(
            None
            if request["payload"]["recovery_generation"] is None
            else "transport_uncertain"
        ),
    )
    provisional_owner = _fixture_owner_mapping_for_authorization(
        provisional_authorization,
        request=request,
        previous_owner_record=previous_owner_record,
    )
    target_projection = encode_authorization_owner_projection(
        provisional_owner, creates_authorization=True
    )
    previous_active = previous_owner["active_service_authorization"]
    if previous_active is None:
        previous_active_authorization = None
    elif previous_active["kind"] == "install":
        previous_active_authorization = previous_owner["install_authorization"]
    else:
        previous_active_authorization = next(
            entry
            for entry in previous_owner["recovery_authorizations"]
            if entry["authorization_sequence"]
            == previous_owner["authorization_sequence"]
        )
    transition = encode_owner_transition(
        authorization_id=request["payload"]["authorization_id"],
        authorization_sequence=sequence,
        from_authorization_id=(
            None
            if previous_active_authorization is None
            else previous_active_authorization["authorization_id"]
        ),
        from_authorization_sequence=previous_owner["authorization_sequence"],
        from_owner_record_sha256=hashlib.sha256(previous_owner_record).hexdigest(),
        from_pending_action=previous_owner["pending_action"],
        previous_transition=prior_transition,
        reason="activate_authorization",
        to_owner_record_projection_sha256=hashlib.sha256(target_projection).hexdigest(),
        to_pending_action=request["operation"],
    )
    authorization = encode_publication_authorization(
        request=request,
        authorization_sequence=sequence,
        previous_authorization=previous,
        owner_transition=transition,
        recovery_reason=(
            None
            if request["payload"]["recovery_generation"] is None
            else "transport_uncertain"
        ),
    )
    target_owner_record = fixture_owner_record_for_authorization(
        authorization,
        request=request,
        previous_owner_record=previous_owner_record,
    )
    validate_owner_transition_against_records(
        transition,
        from_owner_record=previous_owner_record,
        to_owner_record=target_owner_record,
        expected_authorization=authorization,
        creates_authorization=True,
    )
    return authorization


def _fixture_unactivated_owner_mapping(
    request: dict[str, object],
) -> dict[str, object]:
    payload = request["payload"]
    base_python = request["operation"].endswith("base_python")
    return {
        "schema_version": 1,
        "construction_kind": "base_python" if base_python else "closure",
        "closure_id": None if base_python else payload["destination_id"],
        "descriptor_sha256": (
            None if base_python else payload["descriptor_or_intake_sha256"]
        ),
        "base_python_intake_sha256": (
            payload["descriptor_or_intake_sha256"] if base_python else None
        ),
        "run_id": "fixture-install-run",
        "owner_token_sha256": payload["owner_token_sha256"],
        "operation_token_sha256": payload["operation_token_sha256"],
        "host_name": "fixture-host",
        "boot_id": "fixture-boot-id",
        "owner_pid": os.getpid(),
        "owner_start_ticks": 1,
        "state": "CONSTRUCTING",
        "phase": "BASE_COPIED" if base_python else "PAYLOAD_READY",
        "staging_path": "/fixture/staging",
        "final_path": None,
        "construction_attempt": (
            payload["construction_attempt"] if base_python else None
        ),
        "attempt_status": "active" if base_python else None,
        "cleanup_required": True,
        "pending_action": "prepare_install_authorization",
        "pending_local_transition": None,
        "heartbeat_monotonic_ns": 1,
        "heartbeat_wall_ns": 1,
        "last_error": None,
        "install_authorization": None,
        "recovery_authorizations": [],
        "recovery_results": [],
        "active_service_authorization": None,
        "active_authorization_sha256": None,
        "authorization_sequence": 0,
        "transport_uncertain": False,
        "response_uncertain": False,
        "last_preacquisition_status": None,
        "locked_response_records": [],
        "publication_mirrors": [],
        "protected_journal_validated": False,
        "protected_receipt_validated": False,
        "published_root_validated": False,
        "component_hashes": {},
        "final_tree_sha256": None,
        "receipt_sha256": None,
        "terminal_response_status": None,
    }


def fixture_unactivated_owner_record(request: dict[str, object]) -> bytes:
    from simple.eval_runtime.installer_client import (
        encode_authorization_owner_record,
    )

    return encode_authorization_owner_record(
        _fixture_unactivated_owner_mapping(request)
    )


def _fixture_owner_mapping_for_authorization(
    authorization: dict[str, object],
    *,
    request: dict[str, object],
    previous_owner_record: bytes,
) -> dict[str, object]:
    owner = json.loads(previous_owner_record)
    payload = request["payload"]
    recovery_generation = payload["recovery_generation"]
    if recovery_generation is None:
        owner["install_authorization"] = authorization
        owner["recovery_authorizations"] = []
        owner["recovery_results"] = []
        active_kind = "install"
        generation = 0
    else:
        owner["recovery_authorizations"] = [
            *owner["recovery_authorizations"],
            authorization,
        ]
        active_kind = "recovery"
        generation = recovery_generation
    owner.update(
        {
            "active_service_authorization": {
                "kind": active_kind,
                "generation": generation,
            },
            "active_authorization_sha256": authorization["authorization_sha256"],
            "authorization_sequence": authorization["authorization_sequence"],
            "cleanup_required": True,
            "pending_action": request["operation"],
            "pending_local_transition": None,
            "operation_token_sha256": payload["operation_token_sha256"],
            "owner_token_sha256": payload["owner_token_sha256"],
            "state": "INSTALLING",
        }
    )
    return owner


def fixture_owner_record_for_authorization(
    authorization: dict[str, object],
    *,
    request: dict[str, object],
    previous_owner_record: bytes,
) -> bytes:
    from simple.eval_runtime.installer_client import (
        encode_authorization_owner_record,
    )

    return encode_authorization_owner_record(
        _fixture_owner_mapping_for_authorization(
            authorization,
            request=request,
            previous_owner_record=previous_owner_record,
        )
    )


def fixture_owner_transition_entry(
    authorization: dict[str, object],
    *,
    previous_owner_record: bytes,
    target_owner_record: bytes,
    creates_authorization: bool,
) -> dict[str, object]:
    from simple.eval_runtime.installer_client import (
        encode_owner_transition_entry,
    )

    return encode_owner_transition_entry(
        authorization["owner_transition"],
        from_owner_record=previous_owner_record,
        to_owner_record=target_owner_record,
        expected_authorization=authorization,
        creates_authorization=creates_authorization,
    )


def fixture_owner_record_for_reactivation(
    install_authorization: dict[str, object],
    transition: dict[str, object],
    *,
    previous_owner_record: bytes,
    terminal_phase: str = "NO_JOURNAL_NO_MUTATION",
) -> bytes:
    from simple.eval_runtime.installer_client import (
        encode_authorization_owner_record,
        encode_recovery_result,
    )

    owner = json.loads(previous_owner_record)
    recovery = owner["recovery_authorizations"][-1]
    owner["recovery_results"] = [
        *owner["recovery_results"],
        encode_recovery_result(
            recovery_authorization=recovery,
            terminal_response_sha256="b" * 64,
            terminal_phase=terminal_phase,
            transaction_lock_identity_sha256="c" * 64,
            previous_result=(
                owner["recovery_results"][-1] if owner["recovery_results"] else None
            ),
        ),
    ]
    owner.update(
        {
            "active_service_authorization": {
                "kind": "install",
                "generation": 0,
            },
            "active_authorization_sha256": install_authorization[
                "authorization_sha256"
            ],
            "authorization_sequence": transition["to_authorization_sequence"],
            "pending_action": transition["to_pending_action"],
        }
    )
    return encode_authorization_owner_record(owner)


def fixture_install_reactivation_transition(
    install_authorization: dict[str, object],
    *,
    previous_authorization: dict[str, object],
    previous_transition: dict[str, object],
    previous_owner_record: bytes,
    terminal_phase: str = "NO_JOURNAL_NO_MUTATION",
) -> dict[str, object]:
    from simple.eval_runtime.installer_client import (
        encode_authorization_owner_projection,
        encode_owner_transition,
        validate_owner_transition_against_records,
    )

    previous_owner = json.loads(previous_owner_record)
    sequence = previous_owner["authorization_sequence"] + 1
    provisional_transition = {
        "to_authorization_sequence": sequence,
        "to_pending_action": "install_base_python",
    }
    provisional_owner = fixture_owner_record_for_reactivation(
        install_authorization,
        provisional_transition,
        previous_owner_record=previous_owner_record,
        terminal_phase=terminal_phase,
    )
    target_projection = encode_authorization_owner_projection(
        json.loads(provisional_owner), creates_authorization=False
    )
    transition = encode_owner_transition(
        authorization_id=install_authorization["authorization_id"],
        authorization_sequence=sequence,
        from_authorization_id=previous_authorization["authorization_id"],
        from_authorization_sequence=previous_owner["authorization_sequence"],
        from_owner_record_sha256=hashlib.sha256(previous_owner_record).hexdigest(),
        from_pending_action=previous_owner["pending_action"],
        previous_transition=previous_transition,
        reason="no_journal_install_reactivation",
        to_owner_record_projection_sha256=hashlib.sha256(target_projection).hexdigest(),
        to_pending_action="install_base_python",
    )
    validate_owner_transition_against_records(
        transition,
        from_owner_record=previous_owner_record,
        to_owner_record=fixture_owner_record_for_reactivation(
            install_authorization,
            transition,
            previous_owner_record=previous_owner_record,
            terminal_phase=terminal_phase,
        ),
        expected_authorization=install_authorization,
        creates_authorization=False,
    )
    return transition


def fixture_reactivation_transition_entry(
    install_authorization: dict[str, object],
    transition: dict[str, object],
    *,
    previous_owner_record: bytes,
    target_owner_record: bytes,
) -> dict[str, object]:
    from simple.eval_runtime.installer_client import (
        encode_owner_transition_entry,
    )

    return encode_owner_transition_entry(
        transition,
        from_owner_record=previous_owner_record,
        to_owner_record=target_owner_record,
        expected_authorization=install_authorization,
        creates_authorization=False,
    )


class ForkedInstallerService:
    def __init__(
        self, root: Path, pid: int, *, expected_exit_codes: set[int] | None = None
    ):
        self.root = root
        self.pid = pid
        self.socket_path = root / "installer.sock"
        self._stop_requested = False
        self._expected_exit_codes = expected_exit_codes or {0}

    @classmethod
    def start(
        cls,
        root: Path,
        *,
        construction_lock: Path,
        transaction_lock: Path,
        script: tuple[str, ...] | None = None,
        response_fault: str | None = None,
        publication_crash_after: str | None = None,
        manager_state_root: Path | None = None,
    ) -> "ForkedInstallerService":
        root.mkdir(mode=0o700)
        transaction_lock.touch(mode=0o600)
        request_log = root / "requests.jsonl"
        response_log = root / "responses.jsonl"
        request_log.touch(mode=0o600)
        response_log.touch(mode=0o600)
        ready_read, ready_write = os.pipe2(os.O_CLOEXEC)
        pid = os.fork()
        if pid == 0:
            os.close(ready_read)
            if publication_crash_after is not None:
                os.environ["SIMPLE_FIXTURE_CRASH_AFTER_PUBLICATION_PHASE"] = (
                    publication_crash_after
                )
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(os.fspath(root / "installer.sock"))
            listener.listen(4)
            os.write(ready_write, b"1")
            os.close(ready_write)
            install_count = 0
            journal_created = False
            install_request = None
            install_authorization = None
            publication: dict[str, object] | None = None
            recovery_requests: list[dict[str, object]] = []
            recovery_authorizations: list[dict[str, object]] = []
            owner_transition_history: list[dict[str, object]] = []
            owner_record = None
            script_index = 0
            try:
                while True:
                    connection, _ = listener.accept()
                    with connection:
                        try:
                            body, request, construction_fd = cls._receive_request(
                                connection,
                                construction_lock=construction_lock,
                            )
                            try:
                                if manager_state_root is not None:
                                    # This preflight prevents the fixture's
                                    # local convenience state from inventing
                                    # an actor. _send_response reopens the
                                    # same files after taking transaction.lock
                                    # and only that second read authorizes work.
                                    preliminary = cls._reload_manager_evidence(
                                        manager_state_root, request=request
                                    )
                                    install_request = preliminary["install_request"]
                                    install_authorization = preliminary[
                                        "install_authorization"
                                    ]
                                    recovery_requests = list(
                                        preliminary["recovery_requests"]
                                    )
                                    recovery_authorizations = list(
                                        preliminary["recovery_authorizations"]
                                    )
                                    owner_transition_history = list(
                                        preliminary["owner_transition_history"]
                                    )
                                    owner_record = preliminary["owner_record"]
                                if script is not None:
                                    response_status = script[
                                        min(script_index, len(script) - 1)
                                    ]
                                    script_index += 1
                                elif request["operation"] == "install_base_python":
                                    install_count += 1
                                    response_status = (
                                        "TRANSPORT_DROPPED"
                                        if install_count == 1
                                        else "TRANSPORT_DROPPED_AFTER_JOURNAL"
                                    )
                                elif request["operation"] == "recover_base_python":
                                    generation = request["payload"][
                                        "recovery_generation"
                                    ]
                                    response_status = (
                                        "NO_JOURNAL_NO_MUTATION"
                                        if generation == 1 and not journal_created
                                        else "RECOVERED"
                                    )
                                else:
                                    raise RuntimeError("INSTALLER_TEST_OPERATION")
                                cls._append_jsonl(
                                    request_log,
                                    {
                                        "body_hex": body.hex(),
                                        "construction_lock_validated": True,
                                        "fd_roles": request["fd_roles"],
                                        "request_id": request["request_id"],
                                    },
                                )
                                cls._append_jsonl(
                                    response_log,
                                    {
                                        "request_id": request["request_id"],
                                        "status": response_fault or response_status,
                                    },
                                )
                                if response_status == "TRANSPORT_DROPPED":
                                    if request["operation"].startswith("install_"):
                                        install_request = request
                                        install_authorization = (
                                            fixture_publication_authorization(
                                                request, previous=None
                                            )
                                        )
                                        unactivated = fixture_unactivated_owner_record(
                                            request
                                        )
                                        owner_record = (
                                            fixture_owner_record_for_authorization(
                                                install_authorization,
                                                request=request,
                                                previous_owner_record=unactivated,
                                            )
                                        )
                                        owner_transition_history = [
                                            fixture_owner_transition_entry(
                                                install_authorization,
                                                previous_owner_record=unactivated,
                                                target_owner_record=owner_record,
                                                creates_authorization=True,
                                            )
                                        ]
                                    continue
                                if response_status == "INSTALLER_TRANSACTION_BUSY":
                                    contender = os.open(
                                        transaction_lock,
                                        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                                    )
                                    try:
                                        try:
                                            fcntl.flock(
                                                contender,
                                                fcntl.LOCK_EX | fcntl.LOCK_NB,
                                            )
                                        except BlockingIOError:
                                            pass
                                        else:
                                            raise RuntimeError(
                                                "INSTALLER_TEST_BUSY_NOT_CONTENDED"
                                            )
                                    finally:
                                        os.close(contender)
                                bound_install_request = install_request
                                if bound_install_request is None and request[
                                    "operation"
                                ].startswith("install_"):
                                    bound_install_request = request
                                bound_install_authorization = install_authorization
                                if bound_install_authorization is None and request[
                                    "operation"
                                ].startswith("install_"):
                                    bound_install_authorization = (
                                        fixture_publication_authorization(
                                            request, previous=None
                                        )
                                    )
                                if request["operation"].startswith("install_"):
                                    actor_authorization = bound_install_authorization
                                    if owner_record is None:
                                        unactivated = fixture_unactivated_owner_record(
                                            request
                                        )
                                        owner_record = (
                                            fixture_owner_record_for_authorization(
                                                actor_authorization,
                                                request=request,
                                                previous_owner_record=unactivated,
                                            )
                                        )
                                        actor_transition_entry = (
                                            fixture_owner_transition_entry(
                                                actor_authorization,
                                                previous_owner_record=unactivated,
                                                target_owner_record=owner_record,
                                                creates_authorization=True,
                                            )
                                        )
                                    else:
                                        actor_transition_entry = next(
                                            entry
                                            for entry in owner_transition_history
                                            if entry["transition"]["transition_sha256"]
                                            == actor_authorization["owner_transition"][
                                                "transition_sha256"
                                            ]
                                        )
                                    actor_owner_record = owner_record
                                else:
                                    known_ids = [
                                        item["request_id"] for item in recovery_requests
                                    ]
                                    if request["request_id"] in known_ids:
                                        actor_authorization = recovery_authorizations[
                                            known_ids.index(request["request_id"])
                                        ]
                                        actor_owner_record = owner_record
                                        actor_transition_entry = next(
                                            entry
                                            for entry in owner_transition_history
                                            if entry["transition"]["transition_sha256"]
                                            == actor_authorization["owner_transition"][
                                                "transition_sha256"
                                            ]
                                        )
                                    else:
                                        previous_authorization = (
                                            recovery_authorizations[-1]
                                            if recovery_authorizations
                                            else bound_install_authorization
                                        )
                                        previous_transition = (
                                            owner_transition_history[-1]
                                            if owner_transition_history
                                            else None
                                        )
                                        if owner_record is None:
                                            raise RuntimeError(
                                                "INSTALLER_TEST_OWNER_RECORD_MISSING"
                                            )
                                        actor_authorization = (
                                            fixture_publication_authorization(
                                                request,
                                                previous=previous_authorization,
                                                previous_transition=(
                                                    previous_transition
                                                ),
                                                previous_owner_record=owner_record,
                                            )
                                        )
                                        actor_owner_record = (
                                            fixture_owner_record_for_authorization(
                                                actor_authorization,
                                                request=request,
                                                previous_owner_record=owner_record,
                                            )
                                        )
                                        actor_transition_entry = (
                                            fixture_owner_transition_entry(
                                                actor_authorization,
                                                previous_owner_record=owner_record,
                                                target_owner_record=(
                                                    actor_owner_record
                                                ),
                                                creates_authorization=True,
                                            )
                                        )
                                publication = cls._send_response(
                                    connection,
                                    transaction_lock=transaction_lock,
                                    request=request,
                                    status=response_status,
                                    fault=response_fault,
                                    install_request=bound_install_request,
                                    install_authorization=(bound_install_authorization),
                                    actor_authorization=actor_authorization,
                                    publication=publication,
                                    prior_recovery_requests=tuple(recovery_requests),
                                    prior_recovery_authorizations=tuple(
                                        recovery_authorizations
                                    ),
                                    owner_transition_history=tuple(
                                        owner_transition_history
                                        + (
                                            []
                                            if any(
                                                entry["transition"]
                                                == actor_authorization[
                                                    "owner_transition"
                                                ]
                                                for entry in owner_transition_history
                                            )
                                            else [actor_transition_entry]
                                        )
                                    ),
                                    manager_state_root=manager_state_root,
                                )
                                if manager_state_root is not None:
                                    # The manager, not the fake service, owns
                                    # every post-response transition. The next
                                    # request reopens those newly durable files.
                                    continue
                                if response_status in {
                                    "INSTALLED",
                                    "TRANSPORT_DROPPED_AFTER_JOURNAL",
                                }:
                                    install_request = request
                                    install_authorization = actor_authorization
                                    if not any(
                                        entry["transition"]
                                        == actor_authorization["owner_transition"]
                                        for entry in owner_transition_history
                                    ):
                                        owner_transition_history.append(
                                            actor_transition_entry
                                        )
                                        owner_record = actor_owner_record
                                    journal_created = True
                                elif response_status in {
                                    "NO_JOURNAL_NO_MUTATION",
                                    "ABORTED_BEFORE_PREPARED",
                                    "RECOVERED",
                                }:
                                    if not any(
                                        recorded["request_id"] == request["request_id"]
                                        for recorded in recovery_requests
                                    ):
                                        recovery_requests.append(request)
                                        recovery_authorizations.append(
                                            actor_authorization
                                        )
                                        owner_transition_history.append(
                                            actor_transition_entry
                                        )
                                        owner_record = actor_owner_record
                                        if response_status == (
                                            "NO_JOURNAL_NO_MUTATION"
                                        ):
                                            reactivation = (
                                                fixture_install_reactivation_transition(
                                                    bound_install_authorization,
                                                    previous_authorization=(
                                                        actor_authorization
                                                    ),
                                                    previous_transition=(
                                                        owner_transition_history[-1][
                                                            "transition"
                                                        ]
                                                    ),
                                                    previous_owner_record=owner_record,
                                                )
                                            )
                                            reactivated_owner_record = (
                                                fixture_owner_record_for_reactivation(
                                                    bound_install_authorization,
                                                    reactivation,
                                                    previous_owner_record=owner_record,
                                                )
                                            )
                                            owner_transition_history.append(
                                                fixture_reactivation_transition_entry(
                                                    bound_install_authorization,
                                                    reactivation,
                                                    previous_owner_record=owner_record,
                                                    target_owner_record=(
                                                        reactivated_owner_record
                                                    ),
                                                )
                                            )
                                            owner_record = reactivated_owner_record
                                    if response_status == "ABORTED_BEFORE_PREPARED":
                                        install_request = None
                                        install_authorization = None
                                        publication = None
                                        journal_created = False
                                        install_count = 0
                                        recovery_requests = []
                                        recovery_authorizations = []
                                        owner_transition_history = []
                                        owner_record = None
                            finally:
                                os.close(construction_fd)
                        except BrokenPipeError:
                            continue
            finally:
                listener.close()
                os._exit(0)
        os.close(ready_write)
        if os.read(ready_read, 1) != b"1":
            raise RuntimeError("INSTALLER_TEST_SERVICE_START")
        os.close(ready_read)
        return cls(
            root,
            pid,
            expected_exit_codes=({87} if publication_crash_after is not None else {0}),
        )

    @staticmethod
    def _receive_request(
        connection: socket.socket,
        *,
        construction_lock: Path,
    ) -> tuple[bytes, dict[str, object], int]:
        header, ancillary, flags, _ = connection.recvmsg(
            4,
            socket.CMSG_SPACE(array.array("i", [0]).itemsize),
            socket.MSG_CMSG_CLOEXEC | socket.MSG_WAITALL,
        )
        rights = array.array("i")
        invalid_kind = False
        for level, kind, data in ancillary:
            if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                invalid_kind = True
            else:
                rights.frombytes(data[: len(data) - len(data) % rights.itemsize])
        if (
            flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC)
            or not header
            or len(header) > 4
            or invalid_kind
            or len(rights) != 1
        ):
            for fd in rights:
                os.close(fd)
            raise RuntimeError("INSTALLER_TEST_REQUEST_HEADER_OR_ANCILLARY")
        construction_fd = rights[0]
        if len(header) < 4:
            try:
                header += ForkedInstallerService._receive_exact(
                    connection, 4 - len(header)
                )
            except BaseException:
                os.close(construction_fd)
                raise
        length = struct.unpack("!I", header)[0]
        if length <= 0 or length > 1_048_576:
            os.close(construction_fd)
            raise RuntimeError("INSTALLER_TEST_REQUEST_LENGTH")
        try:
            body = ForkedInstallerService._receive_exact(connection, length)
            request = json.loads(body)
            if type(request) is not dict or set(request) != {
                "schema_version",
                "operation",
                "request_id",
                "profile_sha256",
                "payload",
                "fd_roles",
            }:
                raise RuntimeError("INSTALLER_TEST_REQUEST_SCHEMA")
            if (
                json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
                != body
            ):
                raise RuntimeError("INSTALLER_TEST_REQUEST_NONCANONICAL")
            if request["schema_version"] != 1:
                raise RuntimeError("INSTALLER_TEST_REQUEST_VERSION")
            if (
                request["operation"]
                not in {
                    "install_closure",
                    "recover_closure",
                    "install_base_python",
                    "recover_base_python",
                }
                or type(request["request_id"]) is not str
                or not request["request_id"]
                or type(request["profile_sha256"]) is not str
                or len(request["profile_sha256"]) != 64
            ):
                raise RuntimeError("INSTALLER_TEST_REQUEST_VALUE")
            if request["fd_roles"] != ["construction_lock"]:
                raise RuntimeError("INSTALLER_TEST_REQUEST_FD_ROLES")
            payload = request["payload"]
            if type(payload) is not dict or set(payload) != {
                "authorization_id",
                "construction_attempt",
                "descriptor_or_intake_sha256",
                "destination_id",
                "expected_destination_kind",
                "expected_source_kind",
                "operation_token_sha256",
                "owner_token_sha256",
                "recovery_generation",
                "recovery_token_sha256",
                "source_id",
            }:
                raise RuntimeError("INSTALLER_TEST_REQUEST_PAYLOAD")
            recovery = request["operation"].startswith("recover_")
            generation = payload["recovery_generation"]
            recovery_token = payload["recovery_token_sha256"]
            if recovery:
                if type(generation) is not int or generation <= 0:
                    raise RuntimeError("INSTALLER_TEST_RECOVERY_GENERATION")
                if type(recovery_token) is not str or len(recovery_token) != 64:
                    raise RuntimeError("INSTALLER_TEST_RECOVERY_TOKEN")
            elif generation is not None or recovery_token is not None:
                raise RuntimeError("INSTALLER_TEST_INSTALL_GENERATION")
            contender = os.open(
                construction_lock, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            )
            try:
                status = os.fstat(construction_fd)
                expected = os.fstat(contender)
                if (
                    not (fcntl.fcntl(construction_fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC)
                    or (status.st_dev, status.st_ino)
                    != (expected.st_dev, expected.st_ino)
                    or ForkedInstallerService._mount_id(construction_fd)
                    != ForkedInstallerService._mount_id(contender)
                    or stat.S_IMODE(status.st_mode) != 0o600
                ):
                    raise RuntimeError("INSTALLER_TEST_CONSTRUCTION_LOCK_IDENTITY")
                try:
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    raise RuntimeError("INSTALLER_TEST_CONSTRUCTION_LOCK_NOT_HELD")
            finally:
                os.close(contender)
            return body, request, construction_fd
        except BaseException:
            os.close(construction_fd)
            raise

    @staticmethod
    def _mount_id(fd: int) -> int:
        for line in Path(f"/proc/self/fdinfo/{fd}").read_text().splitlines():
            if line.startswith("mnt_id:"):
                return int(line.split(":", 1)[1])
        raise RuntimeError("INSTALLER_TEST_LOCK_MOUNT_ID")

    @staticmethod
    def _receive_exact(connection: socket.socket, count: int) -> bytes:
        value = bytearray()
        while len(value) < count:
            chunk = connection.recv(count - len(value))
            if not chunk:
                raise RuntimeError("INSTALLER_TEST_REQUEST_EOF")
            value.extend(chunk)
        return bytes(value)

    @staticmethod
    def _send_response(
        connection: socket.socket,
        *,
        transaction_lock: Path,
        request: dict[str, object],
        status: str,
        fault: str | None = None,
        install_request: dict[str, object] | None,
        install_authorization: dict[str, object] | None,
        actor_authorization: dict[str, object],
        publication: dict[str, object] | None,
        prior_recovery_requests: tuple[dict[str, object], ...],
        prior_recovery_authorizations: tuple[dict[str, object], ...],
        owner_transition_history: tuple[dict[str, object], ...],
        manager_state_root: Path | None,
    ) -> dict[str, object] | None:
        wire_status = (
            "INSTALLED" if status == "TRANSPORT_DROPPED_AFTER_JOURNAL" else status
        )
        if wire_status == "INSTALLER_TRANSACTION_BUSY":
            payload = ForkedInstallerService._encode_response(
                request=request,
                status=wire_status,
                publication=None,
                fault=fault,
            )
            connection.sendall(struct.pack("!I", len(payload)) + payload)
            return publication
        lock_fd = os.open(
            transaction_lock,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        foreign_fd = None
        replayed_install_publication = False
        replayed_install_response: bytes | None = None
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if manager_state_root is not None:
                evidence = ForkedInstallerService._reload_manager_evidence(
                    manager_state_root, request=request
                )
                install_request = evidence["install_request"]
                install_authorization = evidence["install_authorization"]
                prior_recovery_requests = tuple(evidence["recovery_requests"])
                prior_recovery_authorizations = tuple(
                    evidence["recovery_authorizations"]
                )
                owner_transition_history = tuple(evidence["owner_transition_history"])
                actor_authorization = evidence["active_authorization"]
            if wire_status in {"INSTALLED", "RECOVERED"}:
                if install_request is None:
                    raise RuntimeError("INSTALLER_TEST_INSTALL_REQUEST_MISSING")
                if install_authorization is None:
                    raise RuntimeError("INSTALLER_TEST_INSTALL_AUTHORIZATION_MISSING")
                if wire_status == "INSTALLED":
                    if request != install_request:
                        raise RuntimeError("INSTALLER_TEST_INSTALL_REQUEST_DRIFT")
                    journal = (
                        transaction_lock.parent
                        / "protected-installer"
                        / "journals"
                        / f"{request['payload']['operation_token_sha256']}.jsonl"
                    )
                    try:
                        journal.lstat()
                    except FileNotFoundError:
                        journal_present = False
                    else:
                        journal_present = True
                    if journal_present:
                        (
                            publication,
                            replayed_install_response,
                        ) = ForkedInstallerService._replay_existing_install_publication(
                            transaction_lock.parent,
                            install_request=install_request,
                            install_authorization=install_authorization,
                            recovery_requests=prior_recovery_requests,
                            recovery_authorizations=(prior_recovery_authorizations),
                            owner_transition_history=(owner_transition_history),
                        )
                        replayed_install_publication = True
                    elif publication is not None:
                        raise RuntimeError(
                            "INSTALLER_TEST_INSTALL_JOURNAL_MISSING_WITH_CACHE"
                        )
                    else:
                        component_names = (
                            ("base_python",)
                            if request["operation"].endswith("base_python")
                            else (
                                "source",
                                "venv",
                                "episode_data",
                                "task_data",
                                "hssd_normalization_results",
                                "runtime_identity",
                            )
                        )
                        publication = (
                            ForkedInstallerService._materialize_protected_publication(
                                transaction_lock.parent,
                                install_request=install_request,
                                install_authorization=install_authorization,
                                component_names=component_names,
                                recovery_requests=prior_recovery_requests,
                                recovery_authorizations=(prior_recovery_authorizations),
                                owner_transition_history=(owner_transition_history),
                            )
                        )
                elif publication is None:
                    publication = ForkedInstallerService._resume_protected_publication(
                        transaction_lock.parent,
                        install_request=install_request,
                        install_authorization=install_authorization,
                        prior_recovery_requests=prior_recovery_requests,
                        prior_recovery_authorizations=(prior_recovery_authorizations),
                        recovery_request=request,
                        recovery_authorization=actor_authorization,
                        owner_transition_history=owner_transition_history,
                    )
            protected_response_faults = {
                "protected_response_receipt_digest",
                "protected_response_tree_digest",
            }
            if replayed_install_publication:
                if replayed_install_response is None or fault is not None:
                    raise RuntimeError("INSTALLER_TEST_INSTALL_REPLAY_WIRE_DRIFT")
                payload = replayed_install_response
            else:
                payload = ForkedInstallerService._encode_response(
                    request=request,
                    status=wire_status,
                    publication=publication,
                    fault=fault if fault in protected_response_faults else None,
                )
            wire_faults = {
                "malformed_schema",
                "request_id_mismatch",
                "payload_identity_mismatch",
                "digest_encoding",
                "digest_length",
                "component_missing",
                "component_extra",
                "component_type",
                "root_identity_type",
                "root_identity_missing",
                "root_identity_extra",
                "root_mode",
                "success_error",
                "terminal_phase",
                "busy_error_missing",
                "busy_error_code",
                "busy_error_retryable_type",
                "busy_error_extra",
                "busy_mutation_field",
            }
            wire_payload = (
                ForkedInstallerService._encode_response(
                    request=request,
                    status=wire_status,
                    publication=publication,
                    fault=fault,
                )
                if fault in wire_faults
                else payload
            )
            if (
                wire_status
                in {
                    "INSTALLED",
                    "NO_JOURNAL_NO_MUTATION",
                    "ABORTED_BEFORE_PREPARED",
                    "RECOVERED",
                }
                and not replayed_install_publication
            ):
                ForkedInstallerService._record_terminal_response(
                    transaction_lock.parent,
                    request_id=request["request_id"],
                    operation_token_sha256=request["payload"]["operation_token_sha256"],
                    payload=payload,
                    fault_hook=lambda name: (
                        os._exit(87)
                        if os.environ.get(
                            "SIMPLE_FIXTURE_CRASH_AFTER_PUBLICATION_PHASE"
                        )
                        == name
                        else None
                    ),
                )
            if wire_status == "INSTALLED" and not replayed_install_publication:
                ForkedInstallerService._write_install_publication_journal(
                    publication,
                    install_request=install_request,
                    install_authorization=install_authorization,
                    recovery_requests=prior_recovery_requests,
                    recovery_authorizations=prior_recovery_authorizations,
                    owner_transition_history=owner_transition_history,
                    terminal_response=payload,
                )
            elif wire_status == "RECOVERED":
                ForkedInstallerService._append_recovery_publication_journal(
                    publication,
                    install_request=install_request,
                    install_authorization=install_authorization,
                    prior_recovery_requests=prior_recovery_requests,
                    prior_recovery_authorizations=(prior_recovery_authorizations),
                    recovery_request=request,
                    recovery_authorization=actor_authorization,
                    owner_transition_history=owner_transition_history,
                    terminal_response=payload,
                )
            elif wire_status == "ABORTED_BEFORE_PREPARED":
                ForkedInstallerService._abort_pre_prepared_attempt(
                    transaction_lock.parent,
                    install_request=install_request,
                    install_authorization=install_authorization,
                    prior_recovery_requests=prior_recovery_requests,
                    prior_recovery_authorizations=prior_recovery_authorizations,
                    recovery_request=request,
                    recovery_authorization=actor_authorization,
                    owner_transition_history=owner_transition_history,
                    terminal_response=payload,
                )
            if wire_status in {
                "INSTALLED",
                "NO_JOURNAL_NO_MUTATION",
                "ABORTED_BEFORE_PREPARED",
                "RECOVERED",
            }:
                if fault in {
                    "protected_journal",
                    "protected_receipt",
                    "protected_root_mode",
                    "protected_component",
                    "protected_response_digest",
                }:
                    ForkedInstallerService._mutate_protected_publication(
                        publication, fault=fault
                    )
            if status == "TRANSPORT_DROPPED_AFTER_JOURNAL":
                return publication
            handoff_fd = lock_fd
            if fault == "wrong_handoff":
                foreign = transaction_lock.with_name("foreign-transaction.lock")
                foreign.touch(mode=0o600)
                foreign_fd = os.open(
                    foreign, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
                )
                fcntl.flock(foreign_fd, fcntl.LOCK_EX)
                handoff_fd = foreign_fd
            rights = array.array(
                "i", [] if fault == "missing_handoff" else [handoff_fd]
            )
            body = (
                wire_payload[: max(1, len(wire_payload) // 2)]
                if fault == "partial_body"
                else wire_payload
            )
            ancillary = (
                [] if not rights else [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)]
            )
            connection.sendmsg([struct.pack("!I", len(wire_payload)), body], ancillary)
            return publication
        finally:
            if foreign_fd is not None:
                os.close(foreign_fd)
            os.close(lock_fd)

    @staticmethod
    def _reload_manager_evidence(
        manager_state_root: Path,
        *,
        request: dict[str, object],
    ) -> dict[str, object]:
        """Reopen the real post-crash records after transaction acquisition."""
        from simple.eval_runtime.installer_client import (
            validate_owner_history_for_service,
        )
        from simple.eval_runtime.installer_manager import DiskAuthorizationStore

        store = DiskAuthorizationStore.open_readonly(manager_state_root)
        evidence = store.read_service_evidence()
        if evidence["active_request"] != request:
            raise RuntimeError("INSTALLER_TEST_ACTIVE_REQUEST_DRIFT")
        validate_owner_history_for_service(
            evidence["owner_transition_history"],
            current_owner_record=evidence["owner_record"],
        )
        active = evidence["active_authorization"]
        if (
            active is None
            or active["request_sha256"]
            != hashlib.sha256(
                json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        ):
            raise RuntimeError("INSTALLER_TEST_ACTIVE_AUTHORIZATION_DRIFT")
        return evidence

    @staticmethod
    def _encode_response(
        *,
        request: dict[str, object],
        status: str,
        publication: dict[str, object] | None,
        fault: str | None,
    ) -> bytes:
        request_payload = request["payload"]
        post_prepared = status in {"INSTALLED", "RECOVERED"}
        if status == "INSTALLED" and not request["operation"].startswith("install_"):
            raise RuntimeError("INSTALLER_TEST_RESPONSE_OPERATION")
        if status in {
            "NO_JOURNAL_NO_MUTATION",
            "ABORTED_BEFORE_PREPARED",
            "RECOVERED",
        } and not request["operation"].startswith("recover_"):
            raise RuntimeError("INSTALLER_TEST_RESPONSE_OPERATION")
        if post_prepared and publication is None:
            raise RuntimeError("INSTALLER_TEST_RESPONSE_PUBLICATION")
        response_payload = {
            "authorization_id": request_payload["authorization_id"],
            "completion_sha256": publication["completion_sha256"]
            if publication
            else None,
            "component_hashes": dict(publication["component_hashes"])
            if publication
            else None,
            "final_tree_sha256": publication["final_tree_sha256"]
            if publication
            else None,
            "operation": request["operation"],
            "operation_token_sha256": request_payload["operation_token_sha256"],
            "owner_token_sha256": request_payload["owner_token_sha256"],
            "receipt_sha256": publication["receipt_sha256"] if publication else None,
            "recovery_generation": request_payload["recovery_generation"],
            "recovery_token_sha256": request_payload["recovery_token_sha256"],
            "root_identity": dict(publication["root_identity"])
            if publication
            else None,
            "terminal_phase": (
                "BASE_COMPLETE"
                if post_prepared and request["operation"].endswith("base_python")
                else "COMPLETE"
                if post_prepared
                else (None if status == "INSTALLER_TRANSACTION_BUSY" else status)
            ),
        }
        response = {
            "schema_version": 1,
            "request_id": request["request_id"],
            "status": status,
            "payload": response_payload,
            "error": (
                {"code": status, "retryable": True}
                if status == "INSTALLER_TRANSACTION_BUSY"
                else None
            ),
        }
        component_names = tuple(response_payload["component_hashes"] or ())
        if fault == "malformed_schema":
            response.pop("error")
        elif fault == "request_id_mismatch":
            response["request_id"] = "not-the-request"
        elif fault == "payload_identity_mismatch":
            response_payload["authorization_id"] = "not-the-authorization"
        elif fault == "digest_encoding":
            response_payload["completion_sha256"] = "A" * 64
        elif fault == "digest_length":
            response_payload["completion_sha256"] = "a" * 63
        elif fault == "component_missing":
            response_payload["component_hashes"].pop(component_names[0])
        elif fault == "component_extra":
            response_payload["component_hashes"]["unexpected"] = "7" * 64
        elif fault == "component_type":
            response_payload["component_hashes"][component_names[0]] = True
        elif fault == "root_identity_type":
            response_payload["root_identity"]["inode"] = True
        elif fault == "root_identity_missing":
            response_payload["root_identity"].pop("gid")
        elif fault == "root_identity_extra":
            response_payload["root_identity"]["path"] = "/forged"
        elif fault == "root_mode":
            response_payload["root_identity"]["mode"] = 0o755
        elif fault == "success_error":
            response["error"] = {"code": "FORGED", "retryable": False}
        elif fault == "terminal_phase":
            response_payload["terminal_phase"] = "RECEIPT_CREATED"
        elif fault == "protected_response_receipt_digest":
            response_payload["receipt_sha256"] = "8" * 64
        elif fault == "protected_response_tree_digest":
            response_payload["final_tree_sha256"] = "9" * 64
        elif fault == "busy_error_missing":
            response["error"] = None
        elif fault == "busy_error_code":
            response["error"]["code"] = "OTHER"
        elif fault == "busy_error_retryable_type":
            response["error"]["retryable"] = 1
        elif fault == "busy_error_extra":
            response["error"]["detail"] = "forged"
        elif fault == "busy_mutation_field":
            response_payload["receipt_sha256"] = "8" * 64
        return json.dumps(response, sort_keys=True, separators=(",", ":")).encode()

    @staticmethod
    def _materialize_protected_publication(
        control_parent: Path,
        *,
        install_request: dict[str, object],
        install_authorization: dict[str, object],
        component_names: tuple[str, ...],
        recovery_requests: tuple[dict[str, object], ...],
        recovery_authorizations: tuple[dict[str, object], ...],
        owner_transition_history: tuple[dict[str, object], ...],
    ) -> dict[str, object]:
        from simple.eval_runtime.installer_client import (
            append_publication_phase,
            encode_publication_receipt,
        )

        protected = control_parent / "protected-installer"
        published_parent = protected / "published"
        staging_parent = published_parent
        receipts = protected / "receipts"
        journals = protected / "journals"
        responses = protected / "response-artifacts"
        for directory in (
            protected,
            published_parent,
            staging_parent,
            receipts,
            journals,
            responses,
        ):
            directory.mkdir(mode=0o700, exist_ok=True)
        receipt_parent_identity = (
            ForkedInstallerService._ensure_fixture_installer_config(
                control_parent, receipts=receipts, responses=responses
            )
        )
        response_operation_identity = (
            ForkedInstallerService._ensure_response_operation_directory(
                control_parent,
                operation_token_sha256=install_request["payload"][
                    "operation_token_sha256"
                ],
                fault_hook=lambda name: (
                    os._exit(87)
                    if os.environ.get("SIMPLE_FIXTURE_CRASH_AFTER_PUBLICATION_PHASE")
                    == name
                    else None
                ),
            )
        )
        component_hashes = {
            name: hashlib.sha256(f"fixture:{name}\n".encode()).hexdigest()
            for name in component_names
        }
        final_tree_sha256 = hashlib.sha256(
            json.dumps(component_hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        staged = staging_parent / (
            f".staging-{install_request['payload']['operation_token_sha256']}"
        )
        staged.mkdir(mode=0o700, exist_ok=False)
        source_parent_identity = ForkedInstallerService._path_identity(staging_parent)
        staging_identity = ForkedInstallerService._path_identity(staged)
        journal_path = journals / (
            f"{install_request['payload']['operation_token_sha256']}.jsonl"
        )
        publication: dict[str, object] = {
            "journal_path": os.fspath(journal_path),
            "receipt_parent_identity": receipt_parent_identity,
            "response_operation_identity": response_operation_identity,
            "source_parent_identity": source_parent_identity,
            "staging_identity": staging_identity,
        }
        initial = append_publication_phase(
            b"",
            install_request=install_request,
            install_authorization=install_authorization,
            recovery_requests=recovery_requests,
            recovery_authorizations=recovery_authorizations,
            actor_request=install_request,
            actor_authorization=install_authorization,
            phase="INITIAL",
            publication=publication,
            observed_root_mode=0o700,
            owner_transition_history=owner_transition_history,
        )
        ForkedInstallerService._persist_journal_candidate(
            journal_path, existing=b"", candidate=initial
        )
        for name, digest in component_hashes.items():
            path = staged / f"{name}.sha256"
            path.write_text(f"{digest}\n", encoding="ascii")
            path.chmod(0o444)
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        completion = json.dumps(
            {
                "component_hashes": component_hashes,
                "final_tree_sha256": final_tree_sha256,
                "schema_version": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        completion_path = staged / "COMPLETE.json"
        completion_path.write_bytes(completion)
        completion_path.chmod(0o444)
        with completion_path.open("rb") as stream:
            os.fsync(stream.fileno())
        staged.chmod(0o555)
        ForkedInstallerService._fsync_dir(staged)
        ForkedInstallerService._fsync_dir(staging_parent)
        root_status = staged.stat()
        root_fd = os.open(staged, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            mount_id = ForkedInstallerService._mount_id(root_fd)
        finally:
            os.close(root_fd)
        root_identity = {
            "device": root_status.st_dev,
            "gid": root_status.st_gid,
            "inode": root_status.st_ino,
            "mode": stat.S_IMODE(root_status.st_mode),
            "mount_id": mount_id,
            "uid": root_status.st_uid,
        }
        if {
            key: root_identity[key]
            for key in ("device", "inode", "mount_id", "uid", "gid")
        } != {
            key: staging_identity[key]
            for key in ("device", "inode", "mount_id", "uid", "gid")
        }:
            raise RuntimeError("INSTALLER_TEST_STAGING_IDENTITY_DRIFT")
        publication.update(
            {
                "completion_sha256": hashlib.sha256(completion).hexdigest(),
                "component_hashes": component_hashes,
                "final_tree_sha256": final_tree_sha256,
                "root_identity": root_identity,
            }
        )
        for phase in ("PREPARED", "RENAME_PENDING"):
            current = journal_path.read_bytes()
            candidate = append_publication_phase(
                current,
                install_request=install_request,
                install_authorization=install_authorization,
                recovery_requests=recovery_requests,
                recovery_authorizations=recovery_authorizations,
                actor_request=install_request,
                actor_authorization=install_authorization,
                phase=phase,
                publication=publication,
                observed_root_mode=0o555,
                owner_transition_history=owner_transition_history,
            )
            ForkedInstallerService._persist_journal_candidate(
                journal_path, existing=current, candidate=candidate
            )
        published = published_parent / final_tree_sha256
        os.rename(staged, published)
        ForkedInstallerService._fsync_dir(staging_parent)
        ForkedInstallerService._fsync_dir(published_parent)
        if ForkedInstallerService._path_identity(published) != root_identity:
            raise RuntimeError("INSTALLER_TEST_RENAME_IDENTITY")
        current = journal_path.read_bytes()
        candidate = append_publication_phase(
            current,
            install_request=install_request,
            install_authorization=install_authorization,
            recovery_requests=recovery_requests,
            recovery_authorizations=recovery_authorizations,
            actor_request=install_request,
            actor_authorization=install_authorization,
            phase="FINAL_RENAMED",
            publication=publication,
            observed_root_mode=0o555,
            owner_transition_history=owner_transition_history,
        )
        ForkedInstallerService._persist_journal_candidate(
            journal_path, existing=current, candidate=candidate
        )
        receipt_document = encode_publication_receipt(
            install_request=install_request, publication=publication
        )
        receipt_path = receipts / f"{final_tree_sha256}.json"
        receipt_path.write_bytes(receipt_document)
        receipt_path.chmod(0o444)
        with receipt_path.open("rb") as stream:
            os.fsync(stream.fileno())
        ForkedInstallerService._fsync_dir(receipts)
        receipt_sha256 = hashlib.sha256(receipt_document).hexdigest()
        ForkedInstallerService._validate_existing_receipt(
            receipts,
            receipt_path.name,
            expected_bytes=receipt_document,
            expected_parent_identity=receipt_parent_identity,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
        )
        publication["receipt_sha256"] = receipt_sha256
        return {
            **publication,
            "journal_path": os.fspath(journal_path),
            "published_path": os.fspath(published),
            "receipt_inode": receipt_path.stat().st_ino,
            "receipt_path": os.fspath(receipt_path),
            "receipt_sha256": receipt_sha256,
        }

    @staticmethod
    def _replay_existing_install_publication(
        control_parent: Path,
        *,
        install_request: dict[str, object],
        install_authorization: dict[str, object],
        recovery_requests: tuple[dict[str, object], ...],
        recovery_authorizations: tuple[dict[str, object], ...],
        owner_transition_history: tuple[dict[str, object], ...],
    ) -> tuple[dict[str, object], bytes]:
        """Validate a complete install publication without creating anything."""
        from simple.eval_runtime.installer_client import (
            INSTALL_PUBLICATION_PHASES,
            decode_exact_publication_journal,
            encode_publication_receipt,
        )

        protected = control_parent / "protected-installer"
        token = install_request["payload"]["operation_token_sha256"]
        journal = protected / "journals" / f"{token}.jsonl"
        journal_bytes = journal.read_bytes()
        records = decode_exact_publication_journal(
            journal_bytes,
            install_request=install_request,
            install_authorization=install_authorization,
            recovery_requests=recovery_requests,
            recovery_authorizations=recovery_authorizations,
            owner_transition_history=owner_transition_history,
        )
        if tuple(record["phase"] for record in records) != (
            *INSTALL_PUBLICATION_PHASES,
        ):
            raise RuntimeError("INSTALLER_TEST_INSTALL_REPLAY_PREFIX")
        terminal = records[-1]
        publication = {
            key: terminal[key]
            for key in (
                "completion_sha256",
                "component_hashes",
                "final_tree_sha256",
                "root_identity",
                "receipt_sha256",
            )
        }
        published = protected / "published" / publication["final_tree_sha256"]
        staging = protected / "published" / f".staging-{token}"
        if staging.exists() or (
            ForkedInstallerService._path_identity(published)
            != publication["root_identity"]
        ):
            raise RuntimeError("INSTALLER_TEST_INSTALL_REPLAY_ROOT")
        expected_names = {"COMPLETE.json"}
        for name, digest in publication["component_hashes"].items():
            component = published / f"{name}.sha256"
            if component.read_bytes() != f"{digest}\n".encode("ascii"):
                raise RuntimeError("INSTALLER_TEST_INSTALL_REPLAY_COMPONENT")
            expected_names.add(component.name)
        if {path.name for path in published.iterdir()} != expected_names:
            raise RuntimeError("INSTALLER_TEST_INSTALL_REPLAY_NAMESPACE")
        completion = (published / "COMPLETE.json").read_bytes()
        if hashlib.sha256(completion).hexdigest() != publication["completion_sha256"]:
            raise RuntimeError("INSTALLER_TEST_INSTALL_REPLAY_COMPLETION")

        receipt = protected / "receipts" / f"{publication['final_tree_sha256']}.json"
        receipt_bytes = encode_publication_receipt(
            install_request=install_request, publication=publication
        )
        receipt_contract = ForkedInstallerService._configured_receipt_contract(
            control_parent
        )
        ForkedInstallerService._validate_existing_receipt(
            receipt.parent,
            receipt.name,
            expected_bytes=receipt_bytes,
            expected_parent_identity=receipt_contract["receipt_parent_identity"],
            expected_uid=receipt_contract["input_uid"],
            expected_gid=receipt_contract["input_gid"],
        )
        if hashlib.sha256(receipt_bytes).hexdigest() != publication["receipt_sha256"]:
            raise RuntimeError("INSTALLER_TEST_INSTALL_REPLAY_RECEIPT")
        expected_response = ForkedInstallerService._encode_response(
            request=install_request,
            status="INSTALLED",
            publication=publication,
            fault=None,
        )
        terminal_response_sha256 = hashlib.sha256(expected_response).hexdigest()
        if terminal["terminal_response_sha256"] != terminal_response_sha256:
            raise RuntimeError("INSTALLER_TEST_INSTALL_REPLAY_TERMINAL_DIGEST")
        stored_response, response_identity = (
            ForkedInstallerService._read_existing_install_replay_response(
                control_parent,
                operation_token_sha256=token,
                request_id=install_request["request_id"],
                expected_operation_identity=records[0]["response_operation_identity"],
                expected_payload=expected_response,
                expected_payload_sha256=terminal_response_sha256,
            )
        )
        return (
            {
                **publication,
                "journal_path": os.fspath(journal),
                "published_path": os.fspath(published),
                "receipt_inode": receipt.stat().st_ino,
                "receipt_path": os.fspath(receipt),
                "response_operation_identity": response_identity,
            },
            stored_response,
        )

    @staticmethod
    def _read_existing_install_replay_response(
        control_parent: Path,
        *,
        operation_token_sha256: str,
        request_id: str,
        expected_operation_identity: dict[str, int],
        expected_payload: bytes,
        expected_payload_sha256: str,
    ) -> tuple[bytes, dict[str, int]]:
        """Read existing response evidence without mkdir, append, chmod, or fsync."""
        from simple.eval_runtime.installer_client import (
            _stable_publication_identity,
            decode_exact_response_directory_journal,
        )

        protected = control_parent / "protected-installer"
        configured = ForkedInstallerService._configured_receipt_contract(control_parent)
        protected_fd = os.open(
            protected,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        journal_parent_fd = response_root_fd = operation_fd = -1

        def read_regular_at(
            parent_fd: int, basename: str, *, expected_mode: int
        ) -> tuple[bytes, dict[str, int]]:
            fd = os.open(
                basename,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            try:
                before = os.fstat(fd)
                identity = ForkedInstallerService._fd_identity(fd)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or before.st_uid != configured["input_uid"]
                    or before.st_gid != configured["input_gid"]
                    or stat.S_IMODE(before.st_mode) != expected_mode
                    or before.st_size > 1 << 20
                ):
                    raise RuntimeError("INSTALLER_TEST_INSTALL_REPLAY_FILE_IDENTITY")
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = os.read(fd, 65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > 1 << 20:
                        raise RuntimeError("INSTALLER_TEST_INSTALL_REPLAY_FILE_SIZE")
                    chunks.append(chunk)
                after = os.fstat(fd)
                if (
                    before.st_dev,
                    before.st_ino,
                    before.st_mode,
                    before.st_nlink,
                    before.st_uid,
                    before.st_gid,
                    before.st_size,
                ) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_nlink,
                    after.st_uid,
                    after.st_gid,
                    after.st_size,
                ):
                    raise RuntimeError("INSTALLER_TEST_INSTALL_REPLAY_FILE_DRIFT")
                return b"".join(chunks), identity
            finally:
                os.close(fd)

        try:
            journal_parent_fd = os.open(
                "response-directory-journals",
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=protected_fd,
            )
            response_root_fd = os.open(
                "response-artifacts",
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=protected_fd,
            )
            journal_basename = f"{operation_token_sha256}.jsonl"
            journal_bytes, journal_identity = read_regular_at(
                journal_parent_fd,
                journal_basename,
                expected_mode=0o444,
            )
            records = tuple(
                decode_exact_response_directory_journal(
                    journal_bytes,
                    operation_token_sha256=operation_token_sha256,
                )
            )
            if tuple(record["phase"] for record in records) != (
                "PENDING",
                "CREATED",
            ):
                raise RuntimeError("INSTALLER_TEST_INSTALL_REPLAY_DIRECTORY_JOURNAL")
            response_root_identity = ForkedInstallerService._fd_identity(
                response_root_fd
            )
            if response_root_identity != configured["response_parent_identity"] or any(
                record["response_root_identity"] != response_root_identity
                for record in records
            ):
                raise RuntimeError("INSTALLER_TEST_INSTALL_REPLAY_RESPONSE_ROOT")
            operation_fd = os.open(
                operation_token_sha256,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=response_root_fd,
            )
            operation_identity = ForkedInstallerService._fd_identity(operation_fd)
            if (
                records[-1]["operation_identity"] != operation_identity
                or _stable_publication_identity(operation_identity)
                != expected_operation_identity
            ):
                raise RuntimeError("INSTALLER_TEST_INSTALL_REPLAY_OPERATION_IDENTITY")
            basename = f"{hashlib.sha256(request_id.encode()).hexdigest()}.bin"
            if set(os.listdir(operation_fd)) != {basename}:
                raise RuntimeError("INSTALLER_TEST_INSTALL_REPLAY_ARTIFACT_SET")
            stored, artifact_identity = read_regular_at(
                operation_fd, basename, expected_mode=0o444
            )
            if artifact_identity["mount_id"] != operation_identity["mount_id"]:
                raise RuntimeError("INSTALLER_TEST_INSTALL_REPLAY_ARTIFACT_MOUNT")
            if (
                stored != expected_payload
                or hashlib.sha256(stored).hexdigest() != expected_payload_sha256
            ):
                raise RuntimeError("INSTALLER_TEST_INSTALL_REPLAY_ARTIFACT_BYTES")
            journal_after, journal_identity_after = read_regular_at(
                journal_parent_fd, journal_basename, expected_mode=0o444
            )
            stored_after, artifact_identity_after = read_regular_at(
                operation_fd, basename, expected_mode=0o444
            )
            if (
                journal_after != journal_bytes
                or journal_identity_after != journal_identity
                or stored_after != stored
                or artifact_identity_after != artifact_identity
                or set(os.listdir(operation_fd)) != {basename}
                or ForkedInstallerService._fd_identity(operation_fd)
                != operation_identity
                or ForkedInstallerService._fd_identity(response_root_fd)
                != response_root_identity
            ):
                raise RuntimeError("INSTALLER_TEST_INSTALL_REPLAY_EVIDENCE_DRIFT")
            return stored, operation_identity
        finally:
            for fd in (operation_fd, response_root_fd, journal_parent_fd):
                if fd >= 0:
                    os.close(fd)
            os.close(protected_fd)

    @staticmethod
    def _resume_protected_publication(
        control_parent: Path,
        *,
        install_request: dict[str, object],
        install_authorization: dict[str, object],
        prior_recovery_requests: tuple[dict[str, object], ...],
        prior_recovery_authorizations: tuple[dict[str, object], ...],
        recovery_request: dict[str, object],
        recovery_authorization: dict[str, object],
        owner_transition_history: tuple[dict[str, object], ...],
    ) -> dict[str, object]:
        from simple.eval_runtime.installer_client import (
            INSTALL_PUBLICATION_PHASES,
            append_publication_phase,
            decode_exact_publication_journal,
            encode_publication_receipt,
            _stable_publication_identity,
        )

        protected = control_parent / "protected-installer"
        published_parent = protected / "published"
        receipts = protected / "receipts"
        journals = protected / "journals"
        token = install_request["payload"]["operation_token_sha256"]
        journal = journals / f"{token}.jsonl"
        prefix = journal.read_bytes()
        actor_known_before_resume = any(
            request["request_id"] == recovery_request["request_id"]
            for request in prior_recovery_requests
        )
        prefix_transition_history = owner_transition_history
        if (
            not actor_known_before_resume
            and prefix_transition_history
            and prefix_transition_history[-1]["transition"]
            == recovery_authorization["owner_transition"]
        ):
            prefix_transition_history = prefix_transition_history[:-1]
        records = decode_exact_publication_journal(
            prefix,
            install_request=install_request,
            install_authorization=install_authorization,
            recovery_requests=prior_recovery_requests,
            recovery_authorizations=prior_recovery_authorizations,
            owner_transition_history=prefix_transition_history,
        )
        install_count = sum(
            record["phase"] in INSTALL_PUBLICATION_PHASES
            or record["phase"] == "RECOVERY_ORPHAN_INSTALL_RESPONSE_ADOPTED"
            for record in records
        )
        result_record = next(
            (
                record
                for record in reversed(records)
                if record["final_tree_sha256"] is not None
            ),
            None,
        )
        staged = published_parent / f".staging-{token}"
        component_names = (
            ("base_python",)
            if install_request["operation"].endswith("base_python")
            else (
                "source",
                "venv",
                "episode_data",
                "task_data",
                "hssd_normalization_results",
                "runtime_identity",
            )
        )
        component_hashes = {
            name: hashlib.sha256(f"fixture:{name}\n".encode()).hexdigest()
            for name in component_names
        }
        final_tree_sha256 = hashlib.sha256(
            json.dumps(component_hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        published = published_parent / final_tree_sha256
        active_root = staged if staged.exists() else published
        publication: dict[str, object] = {
            "journal_path": os.fspath(journal),
            "response_operation_identity": records[0]["response_operation_identity"],
            "source_parent_identity": ForkedInstallerService._path_identity(
                published_parent
            ),
            "staging_identity": ForkedInstallerService._path_identity(active_root),
        }
        response_operation_identity = (
            ForkedInstallerService._ensure_response_operation_directory(
                control_parent,
                operation_token_sha256=token,
                require_existing_journal=True,
            )
        )
        if (
            _stable_publication_identity(response_operation_identity)
            != records[0]["response_operation_identity"]
        ):
            raise RuntimeError("INSTALLER_TEST_RESPONSE_OPERATION_IDENTITY")
        if result_record is not None:
            publication.update(
                {
                    key: result_record[key]
                    for key in (
                        "completion_sha256",
                        "component_hashes",
                        "final_tree_sha256",
                        "root_identity",
                    )
                }
            )
        receipt_path = receipts / f"{final_tree_sha256}.json"
        receipt_contract = ForkedInstallerService._configured_receipt_contract(
            control_parent
        )
        expected_receipt_parent_identity = receipt_contract["receipt_parent_identity"]
        receipt_sha256 = next(
            (
                record["receipt_sha256"]
                for record in reversed(records)
                if record["phase"]
                in {
                    "RECEIPT_CREATED",
                    "RECOVERY_ORPHAN_INSTALL_RESPONSE_ADOPTED",
                }
            ),
            None,
        )
        receipt_document = encode_publication_receipt(
            install_request=install_request, publication=publication
        )
        # Recovery never fabricates an install-authored RECEIPT_CREATED
        # suffix.  It first appends RECOVERY_STARTED, then records any orphan
        # install response in a recovery-authored adoption record below.
        if receipt_sha256 is not None:
            if receipt_sha256 != hashlib.sha256(receipt_document).hexdigest():
                raise RuntimeError("INSTALLER_TEST_RECOVERY_RECEIPT_DIGEST")
            ForkedInstallerService._validate_existing_receipt(
                receipts,
                receipt_path.name,
                expected_bytes=receipt_document,
                expected_parent_identity=expected_receipt_parent_identity,
                expected_uid=receipt_contract["input_uid"],
                expected_gid=receipt_contract["input_gid"],
            )
        current = prefix
        actor_already_started = any(
            record["actor_request_id"] == recovery_request["request_id"]
            for record in records
        )
        if actor_already_started:
            recovery_requests = prior_recovery_requests
            recovery_authorizations = prior_recovery_authorizations
            candidate = current
        else:
            candidate = append_publication_phase(
                current,
                install_request=install_request,
                install_authorization=install_authorization,
                recovery_requests=prior_recovery_requests,
                recovery_authorizations=prior_recovery_authorizations,
                actor_request=recovery_request,
                actor_authorization=recovery_authorization,
                phase="RECOVERY_STARTED",
                publication=publication,
                observed_root_mode=records[-1]["observed_root_mode"],
                receipt_sha256=receipt_sha256,
                owner_transition_history=owner_transition_history,
            )
            ForkedInstallerService._persist_journal_candidate(
                journal, existing=current, candidate=candidate
            )
            current = candidate
            recovery_requests = (*prior_recovery_requests, recovery_request)
            recovery_authorizations = (
                *prior_recovery_authorizations,
                recovery_authorization,
            )
        if install_count < 2:
            raise RuntimeError("INSTALLER_TEST_PRE_PREPARED_REQUIRES_ABORT")
        if install_count < 3:
            current = append_publication_phase(
                current,
                install_request=install_request,
                install_authorization=install_authorization,
                recovery_requests=recovery_requests,
                recovery_authorizations=recovery_authorizations,
                actor_request=recovery_request,
                actor_authorization=recovery_authorization,
                phase="RENAME_PENDING",
                publication=publication,
                observed_root_mode=0o555,
                owner_transition_history=owner_transition_history,
            )
            ForkedInstallerService._persist_journal_candidate(
                journal, existing=candidate, candidate=current
            )
            candidate = current
            install_count = 3
        if install_count < 4:
            if staged.exists():
                os.rename(staged, published)
                ForkedInstallerService._fsync_dir(published_parent)
            if (
                ForkedInstallerService._path_identity(published)
                != (publication["root_identity"])
            ):
                raise RuntimeError("INSTALLER_TEST_RECOVERY_RENAME_IDENTITY")
            current = append_publication_phase(
                current,
                install_request=install_request,
                install_authorization=install_authorization,
                recovery_requests=recovery_requests,
                recovery_authorizations=recovery_authorizations,
                actor_request=recovery_request,
                actor_authorization=recovery_authorization,
                phase="FINAL_RENAMED",
                publication=publication,
                observed_root_mode=0o555,
                owner_transition_history=owner_transition_history,
            )
            ForkedInstallerService._persist_journal_candidate(
                journal, existing=candidate, candidate=current
            )
            candidate = current
            install_count = 4
        if install_count < 5:
            try:
                ForkedInstallerService._validate_existing_receipt(
                    receipts,
                    receipt_path.name,
                    expected_bytes=receipt_document,
                    expected_parent_identity=(expected_receipt_parent_identity),
                    expected_uid=receipt_contract["input_uid"],
                    expected_gid=receipt_contract["input_gid"],
                )
            except FileNotFoundError:
                fd = os.open(
                    receipt_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                    0o444,
                )
                try:
                    if os.write(fd, receipt_document) != len(receipt_document):
                        raise RuntimeError("INSTALLER_TEST_RECEIPT_SHORT_WRITE")
                    os.fsync(fd)
                finally:
                    os.close(fd)
            ForkedInstallerService._fsync_dir(receipts)
            if (
                os.environ.get("SIMPLE_FIXTURE_CRASH_AFTER_PUBLICATION_PHASE")
                == "RECEIPT_FSYNCED"
            ):
                os._exit(87)
            receipt_sha256 = hashlib.sha256(receipt_document).hexdigest()
            adopted_identity = ForkedInstallerService._validate_existing_receipt(
                receipts,
                receipt_path.name,
                expected_bytes=receipt_document,
                expected_parent_identity=expected_receipt_parent_identity,
                expected_uid=receipt_contract["input_uid"],
                expected_gid=receipt_contract["input_gid"],
            )
            publication.update(
                {
                    "published_path": os.fspath(published),
                    "receipt_inode": adopted_identity["inode"],
                    "receipt_parent_identity": expected_receipt_parent_identity,
                    "receipt_path": os.fspath(receipt_path),
                    "receipt_sha256": receipt_sha256,
                }
            )
            orphan_install_response = ForkedInstallerService._encode_response(
                request=install_request,
                status="INSTALLED",
                publication=publication,
                fault=None,
            )
            ForkedInstallerService._record_terminal_response(
                control_parent,
                request_id=install_request["request_id"],
                operation_token_sha256=token,
                payload=orphan_install_response,
            )
            current = append_publication_phase(
                current,
                install_request=install_request,
                install_authorization=install_authorization,
                recovery_requests=recovery_requests,
                recovery_authorizations=recovery_authorizations,
                actor_request=recovery_request,
                actor_authorization=recovery_authorization,
                phase="RECOVERY_ORPHAN_INSTALL_RESPONSE_ADOPTED",
                publication=publication,
                observed_root_mode=0o555,
                receipt_sha256=receipt_sha256,
                adopted_install_response=orphan_install_response,
                owner_transition_history=owner_transition_history,
            )
            ForkedInstallerService._persist_journal_candidate(
                journal, existing=candidate, candidate=current
            )
            candidate = current
            install_count = 5
        if receipt_sha256 is None:
            raise RuntimeError("INSTALLER_TEST_RECOVERY_RECEIPT_MISSING")
        final_receipt_identity = ForkedInstallerService._validate_existing_receipt(
            receipts,
            receipt_path.name,
            expected_bytes=receipt_document,
            expected_parent_identity=expected_receipt_parent_identity,
            expected_uid=receipt_contract["input_uid"],
            expected_gid=receipt_contract["input_gid"],
        )
        publication.update(
            {
                "published_path": os.fspath(published),
                "receipt_inode": final_receipt_identity["inode"],
                "receipt_parent_identity": expected_receipt_parent_identity,
                "receipt_path": os.fspath(receipt_path),
                "receipt_sha256": receipt_sha256,
            }
        )
        return publication

    @staticmethod
    def _abort_pre_prepared_attempt(
        control_parent: Path,
        *,
        install_request: dict[str, object],
        install_authorization: dict[str, object],
        prior_recovery_requests: tuple[dict[str, object], ...],
        prior_recovery_authorizations: tuple[dict[str, object], ...],
        recovery_request: dict[str, object],
        recovery_authorization: dict[str, object],
        owner_transition_history: tuple[dict[str, object], ...],
        terminal_response: bytes,
    ) -> None:
        from simple.eval_runtime.installer_client import (
            append_publication_phase,
            decode_exact_publication_journal,
        )

        protected = control_parent / "protected-installer"
        published_parent = protected / "published"
        token = install_request["payload"]["operation_token_sha256"]
        staged = published_parent / f".staging-{token}"
        journal = protected / "journals" / f"{token}.jsonl"
        existing = journal.read_bytes()
        actor_known_before_abort = any(
            request["request_id"] == recovery_request["request_id"]
            for request in prior_recovery_requests
        )
        prefix_transition_history = owner_transition_history
        if (
            not actor_known_before_abort
            and prefix_transition_history
            and prefix_transition_history[-1]["transition"]
            == recovery_authorization["owner_transition"]
        ):
            prefix_transition_history = prefix_transition_history[:-1]
        records = decode_exact_publication_journal(
            existing,
            install_request=install_request,
            install_authorization=install_authorization,
            recovery_requests=prior_recovery_requests,
            recovery_authorizations=prior_recovery_authorizations,
            owner_transition_history=prefix_transition_history,
        )
        phases = [record["phase"] for record in records]
        legal_prefixes = (
            ["INITIAL"],
            ["INITIAL", "RECOVERY_STARTED"],
            ["INITIAL", "RECOVERY_STARTED", "STAGING_REMOVE_PENDING"],
            [
                "INITIAL",
                "RECOVERY_STARTED",
                "STAGING_REMOVE_PENDING",
                "ABORTED_BEFORE_PREPARED",
            ],
        )
        if phases not in legal_prefixes:
            raise RuntimeError("INSTALLER_TEST_ABORT_PREFIX")
        publication = {
            "response_operation_identity": records[0]["response_operation_identity"],
            "source_parent_identity": {
                **records[0]["source_parent_identity"],
                "mode": 0o700,
            },
            "staging_identity": {
                **records[0]["staging_identity"],
                "mode": 0o700,
            },
        }
        actor_known = any(
            request["request_id"] == recovery_request["request_id"]
            for request in prior_recovery_requests
        )
        histories = (
            prior_recovery_requests
            if actor_known
            else (*prior_recovery_requests, recovery_request),
            prior_recovery_authorizations
            if actor_known
            else (*prior_recovery_authorizations, recovery_authorization),
        )
        current = existing
        if phases == ["INITIAL"]:
            candidate = append_publication_phase(
                current,
                install_request=install_request,
                install_authorization=install_authorization,
                recovery_requests=prior_recovery_requests,
                recovery_authorizations=prior_recovery_authorizations,
                actor_request=recovery_request,
                actor_authorization=recovery_authorization,
                phase="RECOVERY_STARTED",
                publication=publication,
                observed_root_mode=0o700,
                owner_transition_history=owner_transition_history,
            )
            ForkedInstallerService._persist_journal_candidate(
                journal, existing=current, candidate=candidate
            )
            current = candidate
            phases.append("RECOVERY_STARTED")
        staging_removal_already_pending = phases[-1] == "STAGING_REMOVE_PENDING"
        if phases[-1] == "RECOVERY_STARTED":
            candidate = append_publication_phase(
                current,
                install_request=install_request,
                install_authorization=install_authorization,
                recovery_requests=histories[0],
                recovery_authorizations=histories[1],
                actor_request=recovery_request,
                actor_authorization=recovery_authorization,
                phase="STAGING_REMOVE_PENDING",
                publication=publication,
                observed_root_mode=0o700,
                owner_transition_history=owner_transition_history,
            )
            ForkedInstallerService._persist_journal_candidate(
                journal, existing=current, candidate=candidate
            )
            current = candidate
            phases.append("STAGING_REMOVE_PENDING")
        if phases[-1] == "STAGING_REMOVE_PENDING":
            if staged.exists():
                ForkedInstallerService._remove_tree_beneath(
                    published_parent,
                    staged.name,
                    expected_identity=publication["staging_identity"],
                    fault_hook=lambda name: (
                        os._exit(87)
                        if os.environ.get(
                            "SIMPLE_FIXTURE_CRASH_AFTER_PUBLICATION_PHASE"
                        )
                        == name
                        else None
                    ),
                )
            elif staged.is_symlink():
                raise RuntimeError("INSTALLER_TEST_ABORT_STAGING_SYMLINK")
            elif not staging_removal_already_pending:
                raise RuntimeError("INSTALLER_TEST_ABORT_STAGING_MISSING")
            else:
                ForkedInstallerService._fsync_absent_staging_parent(
                    published_parent,
                    expected_identity=publication["source_parent_identity"],
                    fault_hook=lambda name: (
                        os._exit(87)
                        if os.environ.get(
                            "SIMPLE_FIXTURE_CRASH_AFTER_PUBLICATION_PHASE"
                        )
                        == name
                        else None
                    ),
                )
            if (
                os.environ.get("SIMPLE_FIXTURE_CRASH_AFTER_PUBLICATION_PHASE")
                == "STAGING_REMOVED"
            ):
                os._exit(87)
            candidate = append_publication_phase(
                current,
                install_request=install_request,
                install_authorization=install_authorization,
                recovery_requests=histories[0],
                recovery_authorizations=histories[1],
                actor_request=recovery_request,
                actor_authorization=recovery_authorization,
                phase="ABORTED_BEFORE_PREPARED",
                publication=publication,
                observed_root_mode=0o700,
                terminal_response=terminal_response,
                owner_transition_history=owner_transition_history,
            )
            ForkedInstallerService._persist_journal_candidate(
                journal, existing=current, candidate=candidate
            )
            current = candidate
        final_records = decode_exact_publication_journal(
            current,
            install_request=install_request,
            install_authorization=install_authorization,
            recovery_requests=histories[0],
            recovery_authorizations=histories[1],
            owner_transition_history=owner_transition_history,
        )
        if (
            final_records[-1]["phase"] != "ABORTED_BEFORE_PREPARED"
            or final_records[-1]["terminal_response_sha256"]
            != hashlib.sha256(terminal_response).hexdigest()
        ):
            raise RuntimeError("INSTALLER_TEST_ABORT_RESPONSE_DRIFT")

    @staticmethod
    def _remove_tree_beneath(
        parent: Path,
        basename: str,
        *,
        expected_identity: dict[str, int],
        fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        if basename in {"", ".", ".."} or "/" in basename:
            raise RuntimeError("INSTALLER_TEST_ABORT_STAGING_BASENAME")
        parent_fd = os.open(
            parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        root_fd = -1
        try:
            root_fd = os.open(
                basename,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            status = os.fstat(root_fd)
            actual = {
                "device": status.st_dev,
                "gid": status.st_gid,
                "inode": status.st_ino,
                "mount_id": ForkedInstallerService._mount_id(root_fd),
                "uid": status.st_uid,
            }
            stable_expected = {
                key: expected_identity[key]
                for key in ("device", "gid", "inode", "mount_id", "uid")
            }
            if actual != stable_expected or not stat.S_ISDIR(status.st_mode):
                raise RuntimeError("INSTALLER_TEST_ABORT_STAGING_IDENTITY")
            ForkedInstallerService._remove_directory_contents_fd(
                root_fd, expected_mount_id=actual["mount_id"]
            )
            os.close(root_fd)
            root_fd = -1
            os.rmdir(basename, dir_fd=parent_fd)
            if fault_hook is not None:
                fault_hook("after_exact_staging_rmdir")
                fault_hook("before_staging_parent_fsync")
            os.fsync(parent_fd)
            if fault_hook is not None:
                fault_hook("after_staging_parent_fsync")
        finally:
            if root_fd >= 0:
                os.close(root_fd)
            os.close(parent_fd)

    @staticmethod
    def _fsync_absent_staging_parent(
        parent: Path,
        *,
        expected_identity: dict[str, int],
        fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        parent_fd = os.open(
            parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        try:
            status = os.fstat(parent_fd)
            actual = {
                "device": status.st_dev,
                "gid": status.st_gid,
                "inode": status.st_ino,
                "mode": stat.S_IMODE(status.st_mode),
                "mount_id": ForkedInstallerService._mount_id(parent_fd),
                "uid": status.st_uid,
            }
            if actual != expected_identity or not stat.S_ISDIR(status.st_mode):
                raise RuntimeError("INSTALLER_TEST_ABORT_PARENT_IDENTITY")
            if fault_hook is not None:
                fault_hook("before_staging_parent_fsync")
            os.fsync(parent_fd)
            if fault_hook is not None:
                fault_hook("after_staging_parent_fsync")
        finally:
            os.close(parent_fd)

    @staticmethod
    def _remove_directory_contents_fd(
        directory_fd: int, *, expected_mount_id: int
    ) -> None:
        os.fchmod(directory_fd, 0o700)
        for name in os.listdir(directory_fd):
            if name in {".", ".."} or "/" in name:
                raise RuntimeError("INSTALLER_TEST_ABORT_CHILD_BASENAME")
            child_status = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(child_status.st_mode):
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    if ForkedInstallerService._mount_id(child_fd) != (
                        expected_mount_id
                    ):
                        raise RuntimeError("INSTALLER_TEST_ABORT_CHILD_MOUNT")
                    ForkedInstallerService._remove_directory_contents_fd(
                        child_fd, expected_mount_id=expected_mount_id
                    )
                finally:
                    os.close(child_fd)
                os.rmdir(name, dir_fd=directory_fd)
            else:
                os.unlink(name, dir_fd=directory_fd)
        os.fsync(directory_fd)

    @staticmethod
    def _write_install_publication_journal(
        publication: dict[str, object],
        *,
        install_request: dict[str, object],
        install_authorization: dict[str, object],
        recovery_requests: tuple[dict[str, object], ...],
        recovery_authorizations: tuple[dict[str, object], ...],
        owner_transition_history: tuple[dict[str, object], ...],
        terminal_response: bytes,
    ) -> None:
        from simple.eval_runtime.installer_client import (
            append_publication_phase,
            decode_exact_publication_journal,
        )

        journal = Path(publication["journal_path"])
        existing = journal.read_bytes()
        records = decode_exact_publication_journal(
            existing,
            install_request=install_request,
            install_authorization=install_authorization,
            recovery_requests=recovery_requests,
            recovery_authorizations=recovery_authorizations,
            owner_transition_history=owner_transition_history,
        )
        if records[-1]["phase"] == "RECEIPT_CREATED":
            if (
                records[-1]["actor_request_id"] != install_request["request_id"]
                or records[-1]["terminal_response_sha256"]
                != hashlib.sha256(terminal_response).hexdigest()
                or records[-1]["receipt_sha256"] != publication["receipt_sha256"]
            ):
                raise RuntimeError("INSTALLER_TEST_TERMINAL_INSTALL_REPLAY_DRIFT")
            return
        if records[-1]["phase"] != "FINAL_RENAMED":
            raise RuntimeError("INSTALLER_TEST_INSTALL_TERMINAL_PREFIX")
        candidate = append_publication_phase(
            existing,
            install_request=install_request,
            install_authorization=install_authorization,
            recovery_requests=recovery_requests,
            recovery_authorizations=recovery_authorizations,
            actor_request=install_request,
            actor_authorization=install_authorization,
            phase="RECEIPT_CREATED",
            publication=publication,
            observed_root_mode=0o555,
            receipt_sha256=publication["receipt_sha256"],
            terminal_response=terminal_response,
            owner_transition_history=owner_transition_history,
        )
        ForkedInstallerService._persist_journal_candidate(
            journal, existing=existing, candidate=candidate
        )

    @staticmethod
    def _persist_journal_candidate(
        journal: Path, *, existing: bytes, candidate: bytes
    ) -> None:
        if not candidate.startswith(existing):
            raise RuntimeError("INSTALLER_TEST_JOURNAL_PREFIX_CHANGED")
        suffix = candidate[len(existing) :]
        if not suffix:
            return
        flags = os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        if existing:
            journal.chmod(0o600)
            flags |= os.O_APPEND
        else:
            flags |= os.O_CREAT | os.O_EXCL
        fd = os.open(journal, flags, 0o600)
        try:
            if os.write(fd, suffix) != len(suffix):
                raise RuntimeError("INSTALLER_TEST_JOURNAL_SHORT_APPEND")
            os.fsync(fd)
            os.fchmod(fd, 0o444)
            os.fsync(fd)
        finally:
            os.close(fd)
        ForkedInstallerService._fsync_dir(journal.parent)
        crash_after = os.environ.get("SIMPLE_FIXTURE_CRASH_AFTER_PUBLICATION_PHASE")
        if crash_after is not None:
            last = json.loads(candidate[:-1].split(b"\n")[-1])
            if last["phase"] == crash_after:
                os._exit(87)

    @staticmethod
    def _append_recovery_publication_journal(
        publication: dict[str, object],
        *,
        install_request: dict[str, object],
        install_authorization: dict[str, object],
        prior_recovery_requests: tuple[dict[str, object], ...],
        prior_recovery_authorizations: tuple[dict[str, object], ...],
        recovery_request: dict[str, object],
        recovery_authorization: dict[str, object],
        owner_transition_history: tuple[dict[str, object], ...],
        terminal_response: bytes,
    ) -> None:
        from simple.eval_runtime.installer_client import (
            RECOVERY_PUBLICATION_PHASES,
            append_publication_phase,
            decode_exact_publication_journal,
        )

        journal = Path(publication["journal_path"])
        receipt = Path(publication["receipt_path"])
        prefix = journal.read_bytes()
        receipt_bytes = receipt.read_bytes()
        receipt_inode = receipt.stat().st_ino
        if (
            receipt_inode != publication["receipt_inode"]
            or hashlib.sha256(receipt_bytes).hexdigest()
            != publication["receipt_sha256"]
        ):
            raise RuntimeError("INSTALLER_TEST_RECEIPT_CHANGED_BEFORE_RECOVERY")
        if any(
            recorded["request_id"] == recovery_request["request_id"]
            for recorded in prior_recovery_requests
        ):
            records = decode_exact_publication_journal(
                prefix,
                install_request=install_request,
                install_authorization=install_authorization,
                recovery_requests=prior_recovery_requests,
                recovery_authorizations=prior_recovery_authorizations,
                owner_transition_history=owner_transition_history,
            )
            if records[-1]["phase"] == "RECOVERY_RESPONSE_CREATED":
                if (
                    records[-1]["actor_request_id"] != recovery_request["request_id"]
                    or records[-1]["terminal_response_sha256"]
                    != hashlib.sha256(terminal_response).hexdigest()
                ):
                    raise RuntimeError("INSTALLER_TEST_RECOVERY_REPLAY_DRIFT")
                return
        current = prefix
        actor_already_started = any(
            json.loads(line)["actor_request_id"] == recovery_request["request_id"]
            for line in prefix.decode("utf-8").splitlines()
        )
        actor_known = any(
            request["request_id"] == recovery_request["request_id"]
            for request in prior_recovery_requests
        )
        recovery_requests = (
            prior_recovery_requests
            if actor_known
            else (*prior_recovery_requests, recovery_request)
        )
        recovery_authorizations = (
            prior_recovery_authorizations
            if actor_known
            else (*prior_recovery_authorizations, recovery_authorization)
        )
        last_phase = json.loads(prefix.decode("utf-8").splitlines()[-1])["phase"]
        if last_phase == "RECOVERY_PUBLICATION_VALIDATED":
            phases = ("RECOVERY_RESPONSE_CREATED",)
        elif actor_already_started:
            phases = RECOVERY_PUBLICATION_PHASES[1:]
        else:
            phases = RECOVERY_PUBLICATION_PHASES
        for phase in phases:
            candidate = append_publication_phase(
                current,
                install_request=install_request,
                install_authorization=install_authorization,
                recovery_requests=(
                    recovery_requests
                    if phase != "RECOVERY_STARTED"
                    else prior_recovery_requests
                ),
                recovery_authorizations=(
                    recovery_authorizations
                    if phase != "RECOVERY_STARTED"
                    else prior_recovery_authorizations
                ),
                actor_request=recovery_request,
                actor_authorization=recovery_authorization,
                phase=phase,
                publication=publication,
                observed_root_mode=0o555,
                receipt_sha256=publication["receipt_sha256"],
                terminal_response=(
                    terminal_response if phase == "RECOVERY_RESPONSE_CREATED" else None
                ),
                owner_transition_history=owner_transition_history,
            )
            ForkedInstallerService._persist_journal_candidate(
                journal, existing=current, candidate=candidate
            )
            current = candidate
        if (
            receipt.read_bytes() != receipt_bytes
            or receipt.stat().st_ino != receipt_inode
            or hashlib.sha256(receipt.read_bytes()).hexdigest()
            != publication["receipt_sha256"]
        ):
            raise RuntimeError("INSTALLER_TEST_RECEIPT_CHANGED_AFTER_RECOVERY")

    @staticmethod
    def _ensure_response_operation_directory(
        control_parent: Path,
        *,
        operation_token_sha256: str,
        fault_hook: Callable[[str], None] | None = None,
        require_existing_journal: bool = False,
    ) -> dict[str, int]:
        """Write-ahead authenticate one operation directory before use."""
        if len(operation_token_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in operation_token_sha256
        ):
            raise RuntimeError("INSTALLER_TEST_RESPONSE_OPERATION_TOKEN")
        protected = control_parent / "protected-installer"
        response_root = protected / "response-artifacts"
        journal_root = protected / "response-directory-journals"
        for directory in (protected, journal_root):
            directory.mkdir(mode=0o700, exist_ok=True)
        journal = journal_root / f"{operation_token_sha256}.jsonl"
        from simple.eval_runtime.installer_client import (
            append_response_directory_phase,
            decode_exact_response_directory_journal,
        )

        existing = journal.read_bytes() if journal.exists() else b""
        records = list(
            decode_exact_response_directory_journal(
                existing,
                operation_token_sha256=operation_token_sha256,
                allow_empty=True,
            )
        )
        if require_existing_journal and not records:
            raise RuntimeError("INSTALLER_TEST_RESPONSE_DIRECTORY_JOURNAL_MISSING")

        response_root_fd = os.open(
            response_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        response_root_identity = ForkedInstallerService._fd_identity(response_root_fd)
        if records and records[0]["response_root_identity"] != (response_root_identity):
            os.close(response_root_fd)
            raise RuntimeError("INSTALLER_TEST_RESPONSE_DIRECTORY_ROOT_IDENTITY")

        def append_record(
            phase: str, operation_identity: dict[str, int] | None
        ) -> None:
            existing = journal.read_bytes() if journal.exists() else b""
            candidate = append_response_directory_phase(
                existing,
                operation_token_sha256=operation_token_sha256,
                phase=phase,
                response_root_identity=response_root_identity,
                operation_identity=operation_identity,
            )
            ForkedInstallerService._persist_journal_candidate(
                journal, existing=existing, candidate=candidate
            )
            records[:] = decode_exact_response_directory_journal(
                candidate, operation_token_sha256=operation_token_sha256
            )

        pending_was_durable = bool(records)
        try:
            if not records:
                # Reject a directory that predates PENDING, even when empty.
                try:
                    os.stat(
                        operation_token_sha256,
                        dir_fd=response_root_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    pass
                else:
                    raise RuntimeError("INSTALLER_TEST_RESPONSE_DIRECTORY_ORPHAN")
                append_record("PENDING", None)
                if fault_hook is not None:
                    fault_hook("after_response_directory_pending")
            elif records[0]["operation_identity"] is not None:
                raise RuntimeError("INSTALLER_TEST_RESPONSE_DIRECTORY_PENDING")

            if len(records) == 1:
                if pending_was_durable:
                    try:
                        operation_fd = os.open(
                            operation_token_sha256,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                            dir_fd=response_root_fd,
                        )
                    except FileNotFoundError:
                        os.mkdir(
                            operation_token_sha256,
                            mode=0o700,
                            dir_fd=response_root_fd,
                        )
                        operation_fd = os.open(
                            operation_token_sha256,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                            dir_fd=response_root_fd,
                        )
                else:
                    os.mkdir(
                        operation_token_sha256,
                        mode=0o700,
                        dir_fd=response_root_fd,
                    )
                    operation_fd = os.open(
                        operation_token_sha256,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                        dir_fd=response_root_fd,
                    )
                try:
                    observed = ForkedInstallerService._fd_identity(operation_fd)
                    if observed["mode"] != 0o700 or os.listdir(operation_fd):
                        raise RuntimeError("INSTALLER_TEST_RESPONSE_DIRECTORY_ORPHAN")
                finally:
                    os.close(operation_fd)
                if fault_hook is not None:
                    fault_hook("after_response_directory_mkdir")
                os.fsync(response_root_fd)
                if fault_hook is not None:
                    fault_hook("after_response_directory_parent_fsync")
                append_record("CREATED", observed)
                if fault_hook is not None:
                    fault_hook("after_response_directory_created")
            expected = records[1]["operation_identity"]
            operation_fd = os.open(
                operation_token_sha256,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=response_root_fd,
            )
            try:
                final_identity = ForkedInstallerService._fd_identity(operation_fd)
            finally:
                os.close(operation_fd)
            if (
                records[0]["response_root_identity"]
                != records[1]["response_root_identity"]
                or records[1]["phase"] != "CREATED"
                or expected is None
                or final_identity != expected
            ):
                raise RuntimeError("INSTALLER_TEST_RESPONSE_DIRECTORY_IDENTITY")
            return expected
        finally:
            os.close(response_root_fd)

    @staticmethod
    def _record_terminal_response(
        control_parent: Path,
        *,
        request_id: str,
        operation_token_sha256: str,
        payload: bytes,
        fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        if len(operation_token_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in operation_token_sha256
        ):
            raise RuntimeError("INSTALLER_TEST_RESPONSE_OPERATION_TOKEN")
        response_root = control_parent / "protected-installer" / ("response-artifacts")
        expected_identity = ForkedInstallerService._ensure_response_operation_directory(
            control_parent,
            operation_token_sha256=operation_token_sha256,
            fault_hook=fault_hook,
        )
        responses = response_root / operation_token_sha256
        if ForkedInstallerService._path_identity(responses) != expected_identity:
            raise RuntimeError("INSTALLER_TEST_RESPONSE_OPERATION_IDENTITY")
        basename = hashlib.sha256(request_id.encode()).hexdigest()
        path = responses / f"{basename}.bin"
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        temporary = responses / f".{basename}.{payload_sha256}.pending"

        def fault(name: str) -> None:
            if fault_hook is not None:
                fault_hook(name)

        def validate_final() -> None:
            existing = path.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(existing.st_mode)
                or existing.st_nlink != 1
                or existing.st_uid != os.getuid()
                or existing.st_gid != os.getgid()
                or stat.S_IMODE(existing.st_mode) != 0o444
                or path.read_bytes() != payload
            ):
                raise RuntimeError("INSTALLER_TEST_RESPONSE_REPLAY_DRIFT")

        if path.exists():
            # The fixture publishes with link(2)+unlink(2), the portable Python
            # analogue of the native renameat2(RENAME_NOREPLACE) path.  A crash
            # between those calls may leave the exact temporary name as the
            # second link to an otherwise complete final inode.
            if temporary.exists():
                final_stat = path.stat(follow_symlinks=False)
                temp_stat = temporary.stat(follow_symlinks=False)
                if (
                    (final_stat.st_dev, final_stat.st_ino)
                    != (temp_stat.st_dev, temp_stat.st_ino)
                    or final_stat.st_nlink != 2
                    or temporary.read_bytes() != payload
                ):
                    raise RuntimeError("INSTALLER_TEST_RESPONSE_TEMP_CONFLICT")
                temporary.unlink()
                ForkedInstallerService._fsync_dir(responses)
            validate_final()
            # Re-fsyncing the parent is idempotent and closes the crash window
            # where atomic publication was visible but its directory entry had
            # not yet reached durable storage.
            ForkedInstallerService._fsync_dir(responses)
            return

        # An unpublished temporary inode is never evidence.  Reuse it only
        # when it is already complete and immutable; otherwise remove exactly
        # this request-and-payload-derived entry, fsync, and rebuild it.
        if temporary.exists():
            temp_stat = temporary.stat(follow_symlinks=False)
            reusable = (
                stat.S_ISREG(temp_stat.st_mode)
                and temp_stat.st_nlink == 1
                and temp_stat.st_uid == os.getuid()
                and temp_stat.st_gid == os.getgid()
                and stat.S_IMODE(temp_stat.st_mode) == 0o444
                and temporary.read_bytes() == payload
            )
            if not reusable:
                if (
                    not stat.S_ISREG(temp_stat.st_mode)
                    or temp_stat.st_nlink != 1
                    or temp_stat.st_uid != os.getuid()
                    or temp_stat.st_gid != os.getgid()
                    or stat.S_IMODE(temp_stat.st_mode) != 0o600
                ):
                    raise RuntimeError("INSTALLER_TEST_RESPONSE_TEMP_IDENTITY")
                temporary.unlink()
                ForkedInstallerService._fsync_dir(responses)
            else:
                temp_fd = os.open(temporary, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
                try:
                    os.fsync(temp_fd)
                finally:
                    os.close(temp_fd)

        fault("before_response_temporary_create")
        if not temporary.exists():
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
            )
            try:
                fault("after_response_temporary_create")
                offset = 0
                while offset < len(payload):
                    written = os.write(fd, payload[offset:])
                    if written <= 0:
                        raise RuntimeError("INSTALLER_TEST_RESPONSE_SHORT_WRITE")
                    offset += written
                    fault("during_response_temporary_write")
                fault("after_response_temporary_write")
                os.fsync(fd)
                fault("after_response_temporary_data_fsync")
                os.fchmod(fd, 0o444)
                fault("after_response_temporary_chmod")
                os.fsync(fd)
                fault("after_response_temporary_metadata_fsync")
            finally:
                os.close(fd)

        fault("before_response_atomic_publish")
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            validate_final()
            return
        fault("after_response_atomic_publish")
        temporary.unlink()
        fault("after_response_temporary_unlink")
        ForkedInstallerService._fsync_dir(responses)
        fault("after_response_parent_fsync")
        validate_final()

    @staticmethod
    def _fd_identity(fd: int) -> dict[str, int]:
        status = os.fstat(fd)
        return {
            "device": status.st_dev,
            "gid": status.st_gid,
            "inode": status.st_ino,
            "mode": stat.S_IMODE(status.st_mode),
            "mount_id": ForkedInstallerService._mount_id(fd),
            "uid": status.st_uid,
        }

    @staticmethod
    def _path_identity(path: Path) -> dict[str, int]:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            return ForkedInstallerService._fd_identity(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _ensure_fixture_installer_config(
        control_parent: Path, *, receipts: Path, responses: Path
    ) -> dict[str, int]:
        protected = control_parent / "protected-installer"
        config = protected / "installer-config.json"
        identity = ForkedInstallerService._path_identity(receipts)
        document = json.dumps(
            {
                "input_gid": os.getgid(),
                "input_uid": os.getuid(),
                "receipt_parent_identity": identity,
                "response_parent_identity": (
                    ForkedInstallerService._path_identity(responses)
                ),
                "schema_version": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if config.exists():
            if config.read_bytes() != document:
                raise RuntimeError("INSTALLER_TEST_CONFIG_DRIFT")
            return identity
        fd = os.open(
            config,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o444,
        )
        try:
            if os.write(fd, document) != len(document):
                raise RuntimeError("INSTALLER_TEST_CONFIG_SHORT_WRITE")
            os.fsync(fd)
        finally:
            os.close(fd)
        ForkedInstallerService._fsync_dir(protected)
        return identity

    @staticmethod
    def _configured_receipt_contract(
        control_parent: Path,
    ) -> dict[str, object]:
        config = control_parent / "protected-installer" / "installer-config.json"
        value = json.loads(config.read_bytes())
        if (
            type(value) is not dict
            or set(value)
            != {
                "input_gid",
                "input_uid",
                "receipt_parent_identity",
                "response_parent_identity",
                "schema_version",
            }
            or value["schema_version"] != 1
        ):
            raise RuntimeError("INSTALLER_TEST_CONFIG_SCHEMA")
        return value

    @staticmethod
    def _validate_existing_receipt(
        parent: Path,
        basename: str,
        *,
        expected_bytes: bytes,
        expected_parent_identity: dict[str, int],
        expected_uid: int,
        expected_gid: int,
    ) -> dict[str, int]:
        if basename in {"", ".", ".."} or "/" in basename:
            raise RuntimeError("INSTALLER_TEST_RECEIPT_BASENAME")
        if type(expected_parent_identity) is not dict or set(
            expected_parent_identity
        ) != {"device", "gid", "inode", "mode", "mount_id", "uid"}:
            raise RuntimeError("INSTALLER_TEST_RECEIPT_PARENT_IDENTITY")
        if any(
            type(expected_parent_identity[key]) is not int
            for key in expected_parent_identity
        ):
            raise RuntimeError("INSTALLER_TEST_RECEIPT_PARENT_IDENTITY")
        parent_fd = os.open(
            parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        receipt_fd = -1
        final_fd = -1
        try:
            parent_status = os.fstat(parent_fd)
            parent_actual = {
                "device": parent_status.st_dev,
                "gid": parent_status.st_gid,
                "inode": parent_status.st_ino,
                "mode": stat.S_IMODE(parent_status.st_mode),
                "mount_id": ForkedInstallerService._mount_id(parent_fd),
                "uid": parent_status.st_uid,
            }
            if parent_actual != expected_parent_identity or not stat.S_ISDIR(
                parent_status.st_mode
            ):
                raise RuntimeError("INSTALLER_TEST_RECEIPT_PARENT_IDENTITY")
            receipt_fd = os.open(
                basename,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            parent_mount = expected_parent_identity["mount_id"]
            before = os.fstat(receipt_fd)
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError("INSTALLER_TEST_RECEIPT_TYPE")
            if before.st_nlink != 1:
                raise RuntimeError("INSTALLER_TEST_RECEIPT_LINK_COUNT")
            if (before.st_uid, before.st_gid) != (expected_uid, expected_gid):
                raise RuntimeError("INSTALLER_TEST_RECEIPT_OWNER")
            if stat.S_IMODE(before.st_mode) != 0o444:
                raise RuntimeError("INSTALLER_TEST_RECEIPT_MODE")
            if ForkedInstallerService._mount_id(receipt_fd) != parent_mount:
                raise RuntimeError("INSTALLER_TEST_RECEIPT_MOUNT")
            chunks = bytearray()
            while len(chunks) <= len(expected_bytes):
                chunk = os.read(receipt_fd, len(expected_bytes) + 1 - len(chunks))
                if not chunk:
                    break
                chunks.extend(chunk)
            if bytes(chunks) != expected_bytes:
                raise RuntimeError("INSTALLER_TEST_RECEIPT_BYTES")
            os.fsync(receipt_fd)
            after = os.fstat(receipt_fd)
            path_status = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
            stable = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_uid",
                "st_gid",
                "st_size",
            )
            if any(
                getattr(before, field) != getattr(after, field)
                or getattr(before, field) != getattr(path_status, field)
                for field in stable
            ):
                raise RuntimeError("INSTALLER_TEST_RECEIPT_DRIFT")
            final_fd = os.open(
                basename,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            final_status = os.fstat(final_fd)
            if (
                any(
                    getattr(before, field) != getattr(final_status, field)
                    for field in stable
                )
                or ForkedInstallerService._mount_id(final_fd) != parent_mount
            ):
                raise RuntimeError("INSTALLER_TEST_RECEIPT_FINAL_REOPEN")
            final_bytes = bytearray()
            while len(final_bytes) <= len(expected_bytes):
                chunk = os.read(final_fd, len(expected_bytes) + 1 - len(final_bytes))
                if not chunk:
                    break
                final_bytes.extend(chunk)
            if bytes(final_bytes) != expected_bytes:
                raise RuntimeError("INSTALLER_TEST_RECEIPT_FINAL_REOPEN")
            return {
                "device": before.st_dev,
                "gid": before.st_gid,
                "inode": before.st_ino,
                "mode": stat.S_IMODE(before.st_mode),
                "mount_id": parent_mount,
                "nlink": before.st_nlink,
                "uid": before.st_uid,
            }
        finally:
            if final_fd >= 0:
                os.close(final_fd)
            if receipt_fd >= 0:
                os.close(receipt_fd)
            os.close(parent_fd)

    @staticmethod
    def _mutate_protected_publication(
        publication: dict[str, object], *, fault: str
    ) -> None:
        if fault == "protected_journal":
            changed = Path(publication["journal_path"])
            changed.chmod(0o600)
            changed.write_text('{"phase":"FORGED"}\n', encoding="ascii")
            changed.chmod(0o444)
        elif fault == "protected_response_digest":
            changed = Path(publication["journal_path"])
            records = changed.read_text(encoding="ascii").splitlines()
            terminal = json.loads(records[-1])
            terminal["terminal_response_sha256"] = "0" * 64
            records[-1] = json.dumps(terminal, sort_keys=True, separators=(",", ":"))
            changed.chmod(0o600)
            changed.write_text("\n".join(records) + "\n", encoding="ascii")
            changed.chmod(0o444)
        elif fault == "protected_receipt":
            changed = Path(publication["receipt_path"])
            changed.chmod(0o600)
            changed.write_bytes(b"{}")
            changed.chmod(0o444)
        elif fault == "protected_root_mode":
            changed = Path(publication["published_path"])
            changed.chmod(0o755)
        elif fault == "protected_component":
            published = Path(publication["published_path"])
            published.chmod(0o755)
            changed = next(published.glob("*.sha256"))
            changed.chmod(0o644)
            changed.write_bytes(b"0" * 64 + b"\n")
            changed.chmod(0o444)
            published.chmod(0o555)
        else:
            raise RuntimeError("INSTALLER_TEST_PROTECTED_FAULT")
        fd = os.open(changed, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        ForkedInstallerService._fsync_dir(changed.parent)

    @staticmethod
    def _append_jsonl(path: Path, value: dict[str, object]) -> None:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        ForkedInstallerService._fsync_dir(path.parent)

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def wait(self, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid == self.pid:
                exit_code = os.waitstatus_to_exitcode(status)
                expected = (
                    {-signal.SIGTERM}
                    if self._stop_requested
                    else self._expected_exit_codes
                )
                if exit_code not in expected:
                    raise RuntimeError("INSTALLER_TEST_SERVICE_EXIT")
                return
            time.sleep(0.01)
        raise TimeoutError("INSTALLER_TEST_SERVICE_WAIT")

    def stop(self) -> None:
        self._stop_requested = True
        os.kill(self.pid, signal.SIGTERM)

    def request_bytes(self) -> list[bytes]:
        return [
            bytes.fromhex(json.loads(line)["body_hex"])
            for line in (self.root / "requests.jsonl").read_text().splitlines()
        ]

    def responses(self) -> list[str]:
        return [
            json.loads(line)["status"]
            for line in (self.root / "responses.jsonl").read_text().splitlines()
        ]

    def response_records(self) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in (self.root / "responses.jsonl").read_text().splitlines()
        ]

    def request_records(self) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in (self.root / "requests.jsonl").read_text().splitlines()
        ]


class FakeRemoteTransport:
    def __init__(
        self, *, disconnect_after: str | None = None, child_hangs: bool = False
    ):
        self.disconnect_after = disconnect_after
        self.child_hangs = child_hangs
        self.launch_count = 0
        self.read_only_reconciliations = 0

    def launch(self, **request: Any) -> SimpleNamespace:
        self.launch_count += 1
        if self.disconnect_after == "detached_ready":
            raise ConnectionError("transport lost after durable acknowledgement")
        if self.child_hangs:
            return SimpleNamespace(
                state="timed_out",
                signal_sequence=["INT", "TERM", "KILL"],
                post_kill_alive=False,
                daemon_postcondition="exited",
            )
        return SimpleNamespace(state="completed")

    def reconcile(self, helper_id: str) -> SimpleNamespace:
        self.read_only_reconciliations += 1
        return SimpleNamespace(state="completed")


class RunnerHarness:
    def __init__(self, *, crash: str | None = None):
        self.crash = crash
        self.children: list[str] = []
        self.output_root_creations = 0
        self.cross_child_visibility_attempts: list[str] = []
        self._terminal: set[int] = set()

    def prepare_output(self) -> None:
        self.output_root_creations += 1

    def loader_probe(self) -> None:
        self.children.append("probe_loader")

    def cuda_probe(self) -> None:
        self.children.append("probe_cuda")

    def evaluate_episode(self, episode: int) -> None:
        if episode == 3 and 2 not in self._terminal:
            raise RuntimeError("PREVIOUS_EPISODE_NOT_TERMINAL")
        self.children.append(f"episode_{episode}")

    def finalize_episode(self, episode: int) -> None:
        self._terminal.add(episode)

    def recover_episode(self, episode: int, *, mode: str) -> SimpleNamespace:
        assert mode in {"live-manager", "stale-manager"}
        if self.crash == "foreign_cgroup_member":
            return SimpleNamespace(status="FOREIGN_BLOCKED")
        self._terminal.add(episode)
        return SimpleNamespace(
            status="RECOVERED",
            terminal=True,
            cgroup_members=(),
            unix_socket_inodes=(),
            upstream_socket_inode=None,
            foreign_signals=[],
        )


class FakeRelay:
    def __init__(self, relay_socket: socket.socket, *, action_shape: tuple[int, int]):
        self.socket = relay_socket
        self.action_shape = action_shape
        self.requests: list[int] = []
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self) -> None:
        self._thread.join(timeout=2.0)
        assert not self._thread.is_alive()

    def _serve(self) -> None:
        from simple.baselines.client import (
            ResponseMessage,
            recv_policy_frame,
            send_policy_frame,
        )

        sequence, request = recv_policy_frame(self.socket.fileno())
        self.requests.append(sequence)
        response = ResponseMessage(
            np.zeros(self.action_shape, np.float32), 0.0
        ).serialize()
        send_policy_frame(self.socket.fileno(), sequence, response)
        self.socket.close()


class ManagedEvaluatorHarness:
    def __init__(self, root: Path):
        self.root = root
        self.sonic_config = {"ENV_TYPE": "sim", "INTERFACE": "lo", "DOMAIN_ID": 0}
        self.events: list[dict[str, Any]] = []
        self.order: list[str] = []
        self.gym_make_calls = 0
        self.options = SimpleNamespace(run_id="run-a", episode_index=2, nonce="a" * 32)
        self.ops = self
        self._record: dict[str, Any] = {}

    @classmethod
    def valid(cls) -> "ManagedEvaluatorHarness":
        import tempfile

        return cls(Path(tempfile.mkdtemp(prefix="managed-evaluator-test-")))

    def write_record(self, value: dict[str, Any]) -> None:
        self._record = value
        self.order.append("evidence_fsync")

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append({"event": event, "payload": payload})
        if event == "runtime_contract":
            self.order.append("runtime_contract_write")
        elif event == "worker_init" and payload.get("status") == "creating_env":
            self.order.append("creating_env_write")

    def read_ack(self, timeout: float) -> dict[str, Any]:
        self.order.append("ack_read")
        return {"schema_version": 1, "accepted_sequence": 1}

    def make_record(
        self, config: dict[str, Any], *, status: str, error: str | None
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "config": dict(config),
            "status": status,
            "error": error,
        }

    def durable_record(self) -> dict[str, Any]:
        return self._record


class FakeVideoOps:
    def __init__(self, *, failure: str | None = None):
        self.failure = failure
        self.now = 100.0
        self.cleanup_complete = False

    def clock(self) -> float:
        return self.now

    def open_writer(self, path: Path, framerate: float, resolution: tuple[int, int]):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)

        class Writer:
            def write(self, image: np.ndarray) -> None:
                os.write(fd, b"frame")

            def release(self) -> None:
                os.fsync(fd)
                os.close(fd)

        return Writer()

    def transcode(self, raw: Path, temporary: Path, deadline: float) -> None:
        if self.failure in {"missing", "timeout", "nonzero", "malformed"}:
            temporary.write_bytes(b"diagnostic")
            self.cleanup_complete = True
            raise RuntimeError(self.failure)
        temporary.write_bytes(raw.read_bytes() + b"h264")

    def probe(self, path: Path) -> SimpleNamespace:
        if self.failure == "probe":
            self.cleanup_complete = True
            raise RuntimeError("probe")
        return SimpleNamespace(
            codec="h264", width=640, height=360, frame_count=1, duration=0.1
        )

    def publish(self, temporary: Path, final: Path) -> None:
        os.link(temporary, final)
        temporary.unlink()
        self.cleanup_complete = True


@dataclass
class ManagerHarness:
    dependencies: Any
    request: Any
    timeline: list[str] = field(default_factory=list)
    states: list[Any] = field(default_factory=list)
    started_episodes: list[int] = field(default_factory=list)

    @classmethod
    def success(
        cls, *, task_results: dict[int, bool] | None = None
    ) -> "ManagerHarness":
        backend = SimpleNamespace(task_results=task_results or {2: True, 3: True})
        harness = cls(backend, SimpleNamespace(episodes=(2, 3)))
        backend.harness = harness
        return harness

    @classmethod
    def failure(cls, phase: str) -> "ManagerHarness":
        harness = cls.success()
        harness.dependencies.failure_phase = phase
        return harness


class CleanupHarness:
    def __init__(self, *, expired: bool):
        self.expired = expired
        self.calls: list[str] = []

    def close(self) -> None:
        self.calls.extend(["restore_terminal", "close_logs", "finalize_manifest"])


class FakeCliBackend:
    def __init__(self):
        self.mutations: list[str] = []
        self.actions: list[str] = []
        self.integration_calls: list[tuple[str, str | None]] = []
        self.integration_profile_approved = True
        self.retirement_state = "failed"
        self.retirement_calls: list[tuple[str, Path]] = []
        self.signals: list[int] = []

    def local_preflight(self) -> None:
        self.actions.append("local_preflight")

    def acquire_lease(self) -> None:
        self.actions.append("lease")

    def leased_preflight(self) -> None:
        self.actions.append("leased_preflight")

    def release_lease(self) -> None:
        self.actions.append("release_lease")

    def run_integration_fixture(
        self, operation: str, fixture: str | None
    ) -> dict[str, object]:
        self.integration_calls.append((operation, fixture))
        return {"schema_version": 1, "marker": operation}

    def require_integration_profile_binding(self) -> None:
        if not self.integration_profile_approved:
            raise RuntimeError("INTEGRATION_PROFILE_BINDING")

    def recover_owned_run(self, run_id: str, run_root: Path) -> None:
        from simple.eval_runtime.evidence import EvidenceStore

        self.retirement_calls.append((run_id, run_root))
        store = EvidenceStore.open_existing(run_root)
        if self.retirement_state == "foreign":
            raise RuntimeError("FOREIGN_BLOCKED")
        if self.retirement_state == "nonterminal":
            return
        if self.retirement_state == "unknown":
            finalize_failed_evidence(
                store,
                run_id=run_id,
                unknown_remote_state=True,
            )
            return
        if self.retirement_state in {"pass", "task-failed"}:
            task_failed = self.retirement_state == "task-failed"
            populate_complete_evidence(store, episodes=(2, 3))
            store.finalize_terminal_manifest(
                run_id=run_id,
                episodes=(2, 3),
                infrastructure_verdict="PASS",
                overall_verdict="FAIL" if task_failed else "PASS",
                task_results={
                    "episode_2": not task_failed,
                    "episode_3": True,
                },
                owned_processes_alive=[],
                unknown_remote_state=False,
                unresolved_errors=[],
            )
            return
        finalize_failed_evidence(store, run_id=run_id)


def make_construction_record(*, phase: str):
    from simple.eval_runtime.pc2_closure import ConstructionRecord

    return ConstructionRecord.from_dict(
        {
            "schema_version": 1,
            "construction_attempt": 1,
            "attempt_status": "active",
            "operation_token_sha256": "a" * 64,
            "phase": phase,
            "cleanup_required": phase != "COMPLETE",
            "pending_action": None,
        }
    )


def make_lease_store(root: Path):
    from simple.eval_runtime.lease import LeaseStore

    now = [1_000_000_000]

    def clock_ns() -> int:
        now[0] += 1
        return now[0]

    return LeaseStore(root=root / "remote-control", clock_ns=clock_ns)


def make_expired_lease_store(root: Path, *, operation: str, cleanup_required: bool):
    from simple.eval_runtime.lease import LeaseStore

    store = LeaseStore(root=root / "remote-control", clock_ns=lambda: 100_000_000_000)
    store.install_expired_test_record(
        run_id="expired-run",
        operation=operation,
        cleanup_required=cleanup_required,
        heartbeat_monotonic_ns=1,
    )
    return store


def make_docker_inspect(*, state: str = "exited"):
    from simple.eval_runtime.container import DockerInspect

    return DockerInspect.from_dict(
        {
            "state": state,
            "image_id": "sha256:" + "a" * 64,
            "network_mode": "bridge",
            "published_ports": {},
            "mounts": [
                {
                    "source": "/protected/inputs",
                    "destination": "/inputs",
                    "read_only": True,
                },
                {
                    "source": "/workloads/run-a",
                    "destination": "/runtime",
                    "read_only": False,
                },
            ],
            "cap_drop": ["ALL"],
            "cap_add": ["SYS_PTRACE"],
            "no_new_privileges": True,
        }
    )


def make_container_contract():
    from simple.eval_runtime.container import ContainerContract

    return ContainerContract.from_dict(
        {
            "image_id": "sha256:" + "a" * 64,
            "network_mode": "bridge",
            "allow_published_ports": False,
            "required_readonly_destinations": ["/inputs"],
            "required_writable_destinations": ["/runtime"],
            "cap_drop": ["ALL"],
            "cap_add": ["SYS_PTRACE"],
            "no_new_privileges": True,
        }
    )


def make_runner_request(*, operation: str, fd_roles: list[str]):
    from simple.eval_runtime.runner_client import RunnerRequest

    return RunnerRequest.from_dict(
        {
            "schema_version": 1,
            "run_id": "run-a",
            "operation": operation,
            "operation_token_sha256": "a" * 64,
            "episode_index": 2 if "episode" in operation else None,
            "fd_roles": fd_roles,
        }
    )


def make_sandbox_manifest():
    from simple.eval_runtime.runner_client import SandboxManifest

    return SandboxManifest.from_dict(
        json.loads(
            (
                Path(__file__).parents[2] / "deploy/psi0_eval/pc2-runner-v1.json"
            ).read_text(encoding="utf-8")
        )
    )


def make_event_reader(
    *, run_id: str = "run-a", episode_index: int = 2, evaluator_pid: int = 100
):
    from simple.eval_runtime.events import EventReader

    return EventReader(
        run_id=run_id,
        episode_index=episode_index,
        evaluator_pid=evaluator_pid,
        clock=lambda: 10.0,
    )


def make_runtime_event(
    *, sequence: int, event: str, payload: dict[str, Any] | None = None
):
    from simple.eval_runtime.events import RuntimeEvent

    return RuntimeEvent(
        schema_version=1,
        run_id="run-a",
        episode_index=2,
        evaluator_pid=100,
        sequence=sequence,
        event=event,
        payload=payload or {},
    )


def inject_event_malformation(reader: Any, malformation: str) -> None:
    malformed = {
        "utf8": (b"\xff\n", False),
        "json": (b"{]\n", False),
        "partial": (b'{"schema_version":1', True),
        "oversize": (b"x" * 4097, False),
        "early_eof": (b"", True),
    }
    payload, eof = malformed[malformation]
    reader.feed_bytes(payload, eof=eof)


def populate_complete_evidence(store: Any, *, episodes: tuple[int, ...]) -> None:
    from simple.eval_runtime.evidence import mandatory_evidence_paths

    for relative in mandatory_evidence_paths(episodes):
        store.write(relative, {"schema_version": 1, "status": "ok"})


def finalize_failed_evidence(
    store: Any,
    *,
    run_id: str,
    unknown_remote_state: bool = False,
) -> Any:
    manifest_path = store.root / "manifest.json"
    if manifest_path.exists():
        return store.verify_terminal_manifest(expected_run_id=run_id)
    for relative in (
        "cleanup/actions.json",
        "cleanup/processes-final.json",
        "cleanup/ports-final.json",
        "cleanup/container-final.json",
        "cleanup/remote-helpers-final.json",
    ):
        if not (store.root / relative).exists():
            store.write(relative, {"schema_version": 1, "status": "clean"})
    return store.finalize_terminal_manifest(
        run_id=run_id,
        episodes=(),
        infrastructure_verdict="FAIL",
        overall_verdict="FAIL",
        task_results={},
        owned_processes_alive=[],
        unknown_remote_state=unknown_remote_state,
        unresolved_errors=["INTERRUPTED_RETRY"],
    )
