from __future__ import annotations

from copy import deepcopy

import pytest

from simple.eval_runtime.pc2_closure import (
    BASE_PHASES,
    CLOSURE_PHASES,
    ClosureDescriptor,
    ConstructionRecord,
    Pc2ClosureBuilder,
    compute_closure_id,
    load_construction_record,
)

from .fakes import make_construction_record


def make_descriptor() -> ClosureDescriptor:
    return ClosureDescriptor(
        schema_version=1,
        simple_commit="a" * 40,
        simple_root_tree="b" * 40,
        recursive_gitlinks_sha256="c" * 64,
        git_object_manifest_sha256="d" * 64,
        episode_data_manifest_sha256="e" * 64,
        asset_requirements_sha256="f" * 64,
        task_data_source_manifest_sha256="0" * 64,
        base_python_tree_sha256="1" * 64,
        mode_policy_version=1,
        hssd_policy_version=1,
        installer_sha256="2" * 64,
        locked_inputs_sha256="3" * 64,
    )


def test_closure_id_is_path_independent_and_requirements_sensitive() -> None:
    descriptor = make_descriptor()
    baseline = compute_closure_id(descriptor)
    assert baseline == compute_closure_id(descriptor)
    changed = deepcopy(descriptor)
    object.__setattr__(changed, "asset_requirements_sha256", "4" * 64)
    assert compute_closure_id(changed) != baseline
    assert not hasattr(descriptor, "pc2_input_host_path")
    assert not hasattr(descriptor, "pc2_asset_normalization_results_sha256")


def test_phase_orders_require_venv_before_hssd() -> None:
    assert CLOSURE_PHASES == (
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
    assert BASE_PHASES == (
        "BASE_ALLOCATED",
        "BASE_COPIED",
        "BASE_METADATA_NORMALIZED",
        "BASE_FINAL_RENAMED",
        "BASE_RECEIPT_CREATED",
        "BASE_COMPLETE",
    )


@pytest.mark.parametrize("phase", CLOSURE_PHASES[:-1])
def test_crash_record_stays_cleanup_required_until_complete(phase: str) -> None:
    record = make_construction_record(phase=phase)
    assert record.cleanup_required is True
    assert record.pending_action is None or isinstance(record.pending_action, str)


def test_builder_persists_pending_action_before_each_operation(tmp_path) -> None:
    owner_record = tmp_path / "owner-record.json"
    builder = Pc2ClosureBuilder(
        owner_record,
        construction_attempt=1,
        operation_token_sha256="a" * 64,
    )
    observed: list[str] = []

    def action() -> None:
        record = load_construction_record(owner_record)
        assert record.pending_action == "allocate"
        assert record.cleanup_required is True
        observed.append(record.pending_action)

    builder.allocate(action)
    assert observed == ["allocate"]
    assert load_construction_record(owner_record).pending_action is None


def test_builder_crash_leaves_fsynced_recoverable_record(tmp_path) -> None:
    owner_record = tmp_path / "owner-record.json"
    builder = Pc2ClosureBuilder(
        owner_record,
        construction_attempt=2,
        operation_token_sha256="b" * 64,
    )

    def crash() -> None:
        raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        builder.allocate(crash)
    record = load_construction_record(owner_record)
    assert record.construction_attempt == 2
    assert record.pending_action == "allocate"
    assert record.cleanup_required is True


def test_builder_only_clears_cleanup_after_complete(tmp_path) -> None:
    owner_record = tmp_path / "owner-record.json"
    builder = Pc2ClosureBuilder(
        owner_record,
        construction_attempt=1,
        operation_token_sha256="c" * 64,
    )

    def no_op() -> None:
        return None

    builder.allocate(no_op)
    builder.copy_git_objects(no_op)
    builder.copy_episode_data(no_op)
    builder.build_relative_venv(no_op)
    builder.copy_task_assets(no_op)
    builder.normalize_hssd_with_staging_venv(no_op)
    builder.probe_payload(no_op)
    builder.authorize_install(no_op)
    builder.mirror_final_renamed(no_op)
    builder.mirror_receipt_created(no_op)
    builder.mark_complete(no_op)
    record = load_construction_record(owner_record)
    assert record == ConstructionRecord(
        schema_version=1,
        construction_attempt=1,
        attempt_status="complete",
        operation_token_sha256="c" * 64,
        phase="COMPLETE",
        cleanup_required=False,
        pending_action=None,
    )
