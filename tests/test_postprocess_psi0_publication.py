import hashlib
import json
import multiprocessing
import os
import shutil
import socket
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from psi0_converter_fixtures import make_source_episode
from scripts import postprocess_psi0 as converter


def _hold_conversion_lock(output: Path, ready, release) -> None:
    with converter.conversion_lock(output):
        ready.set()
        release.wait(timeout=10)


def _exit_while_holding_lock(output: Path, ready) -> None:
    with converter.conversion_lock(output):
        ready.set()
        os._exit(71)


def _wait_in_forked_child(ready, release) -> None:
    ready.set()
    release.wait(timeout=10)


def _make_plan(root: Path):
    source = make_source_episode(root / "source", frames=3, fps=3)
    args = SimpleNamespace(
        skip=0,
        downsample=1,
        chunks_size=1000,
        total_episodes=1,
        fps="3",
        video_key="observation.rgb_head_stereo_left",
        out_dir=str(root / "processed"),
        sim_root=str(source),
    )
    identity = converter.ConverterIdentity("a" * 40, "b" * 64)
    return converter.preflight_conversion(args, identity)


@pytest.fixture(scope="module")
def generated_staging(tmp_path_factory):
    root = tmp_path_factory.mktemp("manifest-source")
    plan = _make_plan(root)
    staging = root / ".processed.staging-template"
    converter.generate_staged_dataset(plan, staging)
    return staging


def _copy_staging(generated_staging: Path, tmp_path: Path) -> Path:
    staging = tmp_path / ".processed.staging-test"
    shutil.copytree(generated_staging, staging)
    (staging / "CONVERSION_STATUS.json").write_bytes(
        converter._in_progress_status_bytes(staging.stat())
    )
    return staging


def _write_manifest(staging: Path) -> dict[str, object]:
    manifest = converter.build_payload_manifest(staging)
    path = staging / "meta" / "conversion_manifest.json"
    path.write_bytes(converter.canonical_json_bytes(manifest))
    return manifest


def _rewrite_manifest(staging: Path, manifest: dict[str, object]) -> None:
    (staging / "meta" / "conversion_manifest.json").write_bytes(
        converter.canonical_json_bytes(manifest)
    )


def _entry_paths(manifest: dict[str, object]) -> list[str]:
    entries = manifest["entries"]
    assert isinstance(entries, list)
    return [entry["path"] for entry in entries]


def _enumerated_regular_files(staging: Path) -> set[str]:
    result = set()
    for directory, directory_names, file_names in os.walk(staging):
        directory_names.sort()
        for name in file_names:
            path = Path(directory) / name
            if stat.S_ISREG(path.lstat().st_mode):
                result.add(path.relative_to(staging).as_posix())
    return result


def test_conversion_lock_blocks_a_real_process_without_creating_staging(tmp_path):
    output = tmp_path / "processed"
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_conversion_lock,
        args=(output, ready, release),
    )
    holder.start()
    assert ready.wait(timeout=10)

    with pytest.raises(BlockingIOError):
        with converter.conversion_lock(output):
            pytest.fail("contender acquired a held conversion lock")
    assert not list(tmp_path.glob(".processed.staging-*"))

    release.set()
    holder.join(timeout=10)
    assert holder.exitcode == 0


def test_conversion_lock_is_kernel_released_after_os_exit(tmp_path):
    output = tmp_path / "processed"
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    holder = context.Process(
        target=_exit_while_holding_lock,
        args=(output, ready),
    )
    holder.start()
    assert ready.wait(timeout=10)
    holder.join(timeout=10)
    assert holder.exitcode == 71

    lock_path = tmp_path / ".processed.conversion.lock"
    original_identity = (lock_path.stat().st_dev, lock_path.stat().st_ino)
    with converter.conversion_lock(output) as fd:
        replacement_identity = os.fstat(fd)
        assert (replacement_identity.st_dev, replacement_identity.st_ino) == (
            original_identity
        )
    assert lock_path.exists()
    assert not list(tmp_path.glob(".processed.staging-*"))


@pytest.mark.parametrize("kind", ["symlink", "directory", "wrong_mode", "hard_link"])
def test_conversion_lock_rejects_unsafe_persistent_path_before_staging(tmp_path, kind):
    output = tmp_path / "processed"
    lock_path = tmp_path / ".processed.conversion.lock"
    if kind == "symlink":
        target = tmp_path / "lock-target"
        target.write_bytes(b"")
        target.chmod(0o600)
        lock_path.symlink_to(target)
    elif kind == "directory":
        lock_path.mkdir(mode=0o600)
    elif kind == "wrong_mode":
        lock_path.write_bytes(b"")
        lock_path.chmod(0o644)
    else:
        target = tmp_path / "lock-target"
        target.write_bytes(b"")
        target.chmod(0o600)
        os.link(target, lock_path)

    with pytest.raises((OSError, RuntimeError)):
        with converter.conversion_lock(output):
            pytest.fail("unsafe conversion lock was accepted")
    assert not list(tmp_path.glob(".processed.staging-*"))


def test_conversion_lock_detects_path_swap_during_acquisition(tmp_path, monkeypatch):
    output = tmp_path / "processed"
    lock_path = tmp_path / ".processed.conversion.lock"
    original_flock = converter.fcntl.flock

    def swap_after_lock(fd, operation):
        original_flock(fd, operation)
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            return
        displaced = tmp_path / "displaced.lock"
        lock_path.rename(displaced)
        lock_path.write_bytes(b"")
        lock_path.chmod(0o600)

    monkeypatch.setattr(converter.fcntl, "flock", swap_after_lock)
    with pytest.raises(RuntimeError, match="identity"):
        with converter.conversion_lock(output):
            pytest.fail("swapped lock pathname was accepted")
    assert not list(tmp_path.glob(".processed.staging-*"))


def test_conversion_lock_file_replacement_cannot_create_split_brain(tmp_path):
    output = tmp_path / "processed"
    lock_path = tmp_path / ".processed.conversion.lock"
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_conversion_lock,
        args=(output, ready, release),
    )
    holder.start()
    assert ready.wait(timeout=10)
    try:
        lock_path.rename(tmp_path / "displaced.lock")
        lock_path.write_bytes(b"")
        lock_path.chmod(0o600)
        with pytest.raises((BlockingIOError, RuntimeError)):
            with converter.conversion_lock(output):
                pytest.fail("replacement lock file created split-brain ownership")
    finally:
        release.set()
        holder.join(timeout=10)
    assert holder.exitcode != 0


def test_conversion_lock_allows_different_outputs_in_one_parent(tmp_path):
    first_output = tmp_path / "processed-a"
    second_output = tmp_path / "processed-b"
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_conversion_lock,
        args=(first_output, ready, release),
    )
    holder.start()
    assert ready.wait(timeout=10)
    try:
        with converter.conversion_lock(second_output):
            pass
    finally:
        release.set()
        holder.join(timeout=10)
    assert holder.exitcode == 0


@pytest.mark.parametrize("kind", ["symlink", "file", "wrong_mode"])
def test_conversion_lock_rejects_unsafe_anchor_before_staging(tmp_path, kind):
    output = tmp_path / "processed"
    anchor = tmp_path / ".processed.conversion.lock-anchor"
    if kind == "symlink":
        target = tmp_path / "anchor-target"
        target.mkdir(mode=0o700)
        anchor.symlink_to(target, target_is_directory=True)
    elif kind == "file":
        anchor.write_bytes(b"")
        anchor.chmod(0o700)
    else:
        anchor.mkdir(mode=0o755)

    with pytest.raises((OSError, RuntimeError)):
        with converter.conversion_lock(output):
            pytest.fail("unsafe conversion lock anchor was accepted")
    assert not list(tmp_path.glob(".processed.staging-*"))


def test_conversion_lock_rejects_anchor_replacement(tmp_path):
    output = tmp_path / "processed"
    anchor = tmp_path / ".processed.conversion.lock-anchor"

    with pytest.raises(RuntimeError, match="identity"):
        with converter.conversion_lock(output):
            anchor.rename(tmp_path / "displaced-anchor")
            anchor.mkdir(mode=0o700)


def test_conversion_lock_rejects_output_parent_replacement(tmp_path):
    output_parent = tmp_path / "output-parent"
    output_parent.mkdir()
    output = output_parent / "processed"

    with pytest.raises(RuntimeError, match="identity|disappeared"):
        with converter.conversion_lock(output):
            output_parent.rename(tmp_path / "displaced-parent")
            output_parent.mkdir()


def test_forked_child_does_not_prolong_parent_conversion_lock(tmp_path):
    output = tmp_path / "processed"
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    release = context.Event()
    with converter.conversion_lock(output):
        child = context.Process(
            target=_wait_in_forked_child,
            args=(ready, release),
        )
        child.start()
        assert ready.wait(timeout=10)

    try:
        with converter.conversion_lock(output):
            pass
    finally:
        release.set()
        child.join(timeout=10)
    assert child.exitcode == 0


def test_conversion_lock_direct_fork_child_finalizer_does_not_close_reused_fd(
    tmp_path,
):
    output = tmp_path / "processed"
    manager = converter.conversion_lock(output)
    manager.__enter__()
    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(read_fd)
        reused_fd = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
        result = b"ok"
        try:
            manager.__exit__(None, None, None)
            os.fstat(reused_fd)
        except BaseException as exc:
            result = f"bad:{type(exc).__name__}".encode()
        os.write(write_fd, result)
        os._exit(0)

    os.close(write_fd)
    try:
        manager.__exit__(None, None, None)
        _, status = os.waitpid(child_pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert os.read(read_fd, 128) == b"ok"
    finally:
        os.close(read_fd)


def test_conversion_lock_releases_normally_for_immediate_reacquisition(tmp_path):
    output = tmp_path / "processed"
    with converter.conversion_lock(output):
        pass
    with converter.conversion_lock(output):
        pass


def test_conversion_lock_rejects_wrong_owner_when_privilege_fixture_is_available(
    tmp_path,
):
    if os.geteuid() != 0:
        pytest.skip("safe wrong-owner fixture requires an isolated root test process")
    output = tmp_path / "processed"
    lock_path = tmp_path / ".processed.conversion.lock"
    lock_path.write_bytes(b"")
    lock_path.chmod(0o600)
    os.chown(lock_path, 65534, 65534)
    with pytest.raises(RuntimeError, match="owner"):
        with converter.conversion_lock(output):
            pytest.fail("wrong-owner conversion lock was accepted")


def test_manifest_covers_exact_generated_regular_file_set(generated_staging, tmp_path):
    staging = _copy_staging(generated_staging, tmp_path)
    manifest = _write_manifest(staging)

    assert converter.validate_payload_manifest(staging) == manifest
    assert _enumerated_regular_files(staging) == set(_entry_paths(manifest)) | {
        "meta/conversion_manifest.json",
        "CONVERSION_STATUS.json",
    }
    assert _entry_paths(manifest) == sorted(
        _entry_paths(manifest), key=lambda path: path.encode("utf-8")
    )
    for entry in manifest["entries"]:
        assert set(entry) == {"path", "size", "sha256"}
        path = staging / entry["path"]
        assert entry["size"] == path.stat().st_size
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_manifest_rejects_payload_membership_changes(
    generated_staging, tmp_path, mutation
):
    staging = _copy_staging(generated_staging, tmp_path)
    manifest = _write_manifest(staging)
    if mutation == "missing":
        (staging / _entry_paths(manifest)[0]).unlink()
    else:
        (staging / "data" / "extra.bin").write_bytes(b"extra")

    with pytest.raises((ValueError, RuntimeError)):
        converter.validate_payload_manifest(staging)


@pytest.mark.parametrize("kind", ["symlink", "fifo", "socket", "device"])
def test_manifest_rejects_nonregular_members(generated_staging, tmp_path, kind):
    staging = _copy_staging(generated_staging, tmp_path)
    _write_manifest(staging)
    path = staging / "meta" / f"unsafe-{kind}"
    live_socket = None
    if kind == "symlink":
        path.symlink_to(staging / "meta" / "info.json")
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "socket":
        live_socket = socket.socket(socket.AF_UNIX)
        try:
            live_socket.bind(str(path))
        except PermissionError:
            live_socket.close()
            pytest.skip("sandbox does not permit creating a Unix socket fixture")
    else:
        try:
            os.mknod(path, stat.S_IFCHR | 0o600, os.makedev(1, 3))
        except PermissionError:
            pytest.skip("safe device-node fixture requires CAP_MKNOD")
    try:
        with pytest.raises((ValueError, RuntimeError)):
            converter.validate_payload_manifest(staging)
    finally:
        if live_socket is not None:
            live_socket.close()


def test_manifest_rejects_hard_linked_member(generated_staging, tmp_path):
    staging = _copy_staging(generated_staging, tmp_path)
    _write_manifest(staging)
    os.link(staging / "meta" / "info.json", staging / "meta" / "info-copy.json")

    with pytest.raises((ValueError, RuntimeError), match="link"):
        converter.validate_payload_manifest(staging)


@pytest.mark.parametrize("field", ["size", "sha256"])
def test_manifest_rejects_wrong_file_identity(generated_staging, tmp_path, field):
    staging = _copy_staging(generated_staging, tmp_path)
    manifest = _write_manifest(staging)
    entry = manifest["entries"][0]
    if field == "size":
        entry[field] += 1
    else:
        entry[field] = "0" * 64
    _rewrite_manifest(staging, manifest)

    with pytest.raises((ValueError, RuntimeError)):
        converter.validate_payload_manifest(staging)


@pytest.mark.parametrize(
    "reserved", ["meta/conversion_manifest.json", "CONVERSION_STATUS.json"]
)
def test_manifest_rejects_reserved_self_or_status_entry(
    generated_staging, tmp_path, reserved
):
    staging = _copy_staging(generated_staging, tmp_path)
    manifest = _write_manifest(staging)
    manifest["entries"].append({"path": reserved, "size": 1, "sha256": "0" * 64})
    manifest["entries"].sort(key=lambda entry: entry["path"].encode("utf-8"))
    _rewrite_manifest(staging, manifest)

    with pytest.raises(ValueError, match="reserved"):
        converter.validate_payload_manifest(staging)


@pytest.mark.parametrize("mutation", ["unsorted", "duplicate"])
def test_manifest_rejects_noncanonical_path_order(
    generated_staging, tmp_path, mutation
):
    staging = _copy_staging(generated_staging, tmp_path)
    manifest = _write_manifest(staging)
    entries = manifest["entries"]
    if mutation == "unsorted":
        entries[0], entries[1] = entries[1], entries[0]
    else:
        entries.insert(1, dict(entries[0]))
    _rewrite_manifest(staging, manifest)

    with pytest.raises(ValueError, match="order|duplicate"):
        converter.validate_payload_manifest(staging)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/data/file",
        "data/./file",
        "data/../file",
        "data//file",
        "outside/file",
    ],
)
def test_manifest_rejects_unsafe_or_out_of_root_entry_path(
    generated_staging, tmp_path, unsafe_path
):
    staging = _copy_staging(generated_staging, tmp_path)
    manifest = _write_manifest(staging)
    manifest["entries"][0]["path"] = unsafe_path
    manifest["entries"].sort(key=lambda entry: entry["path"].encode("utf-8"))
    _rewrite_manifest(staging, manifest)

    with pytest.raises(ValueError, match="path"):
        converter.validate_payload_manifest(staging)


@pytest.mark.parametrize("digest", ["A" * 64, "0" * 63, "g" * 64, 7])
def test_manifest_rejects_uppercase_or_malformed_digest(
    generated_staging, tmp_path, digest
):
    staging = _copy_staging(generated_staging, tmp_path)
    manifest = _write_manifest(staging)
    manifest["entries"][0]["sha256"] = digest
    _rewrite_manifest(staging, manifest)

    with pytest.raises(ValueError, match="SHA-256"):
        converter.validate_payload_manifest(staging)


def test_manifest_builder_rejects_file_outside_payload_roots(
    generated_staging, tmp_path
):
    staging = _copy_staging(generated_staging, tmp_path)
    (staging / "outside.bin").write_bytes(b"outside")

    with pytest.raises((ValueError, RuntimeError), match="root|layout"):
        converter.build_payload_manifest(staging)


@pytest.mark.parametrize("reserved", ["manifest", "status"])
def test_manifest_rejects_missing_reserved_path(generated_staging, tmp_path, reserved):
    staging = _copy_staging(generated_staging, tmp_path)
    _write_manifest(staging)
    path = (
        staging / "meta" / "conversion_manifest.json"
        if reserved == "manifest"
        else staging / "CONVERSION_STATUS.json"
    )
    path.unlink()

    with pytest.raises((OSError, ValueError, RuntimeError)):
        converter.validate_payload_manifest(staging)


@pytest.mark.parametrize(
    "extra", ["conversion_manifest.json", "meta/CONVERSION_STATUS.json"]
)
def test_manifest_rejects_extra_reserved_path(generated_staging, tmp_path, extra):
    staging = _copy_staging(generated_staging, tmp_path)
    _write_manifest(staging)
    path = staging / extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"{}\n")

    with pytest.raises((ValueError, RuntimeError)):
        converter.validate_payload_manifest(staging)


def test_manifest_rejects_noncanonical_json_bytes(generated_staging, tmp_path):
    staging = _copy_staging(generated_staging, tmp_path)
    manifest = _write_manifest(staging)
    (staging / "meta" / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical"):
        converter.validate_payload_manifest(staging)


def test_manifest_rejects_unlisted_file_inserted_after_initial_listdir(
    generated_staging, tmp_path, monkeypatch
):
    staging = _copy_staging(generated_staging, tmp_path)
    _write_manifest(staging)
    original_listdir = converter.os.listdir
    inserted = False

    def insert_after_listing(path):
        nonlocal inserted
        names = original_listdir(path)
        try:
            opened_path = Path(os.readlink(f"/proc/self/fd/{path}"))
        except (OSError, TypeError):
            return names
        if not inserted and opened_path == staging / "meta":
            (staging / "meta" / "late-member.json").write_bytes(b"{}\n")
            inserted = True
        return names

    monkeypatch.setattr(converter.os, "listdir", insert_after_listing)
    with pytest.raises((ValueError, RuntimeError)):
        converter.validate_payload_manifest(staging)
    assert inserted


def test_manifest_rejects_file_inserted_during_final_listdir(
    generated_staging, tmp_path, monkeypatch
):
    staging = _copy_staging(generated_staging, tmp_path)
    _write_manifest(staging)
    original_listdir = converter.os.listdir
    meta_listings = 0
    inserted = False

    def insert_during_final_listing(path):
        nonlocal meta_listings, inserted
        names = original_listdir(path)
        try:
            opened_path = Path(os.readlink(f"/proc/self/fd/{path}"))
        except (OSError, TypeError):
            return names
        if opened_path == staging / "meta":
            meta_listings += 1
            if meta_listings == 2:
                (staging / "meta" / "late-final-member.json").write_bytes(b"{}\n")
                inserted = True
        return names

    monkeypatch.setattr(converter.os, "listdir", insert_during_final_listing)
    with pytest.raises((ValueError, RuntimeError)):
        converter.validate_payload_manifest(staging)
    assert inserted


def test_manifest_rejects_member_mutated_immediately_after_hash(
    generated_staging, tmp_path, monkeypatch
):
    staging = _copy_staging(generated_staging, tmp_path)
    _write_manifest(staging)
    target = staging / "meta" / "info.json"
    original_sha256_descriptor = converter._sha256_descriptor
    target_hash_calls = 0
    mutated = False

    def mutate_after_hash(fd):
        nonlocal target_hash_calls, mutated
        digest = original_sha256_descriptor(fd)
        try:
            opened_path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:
            return digest
        if opened_path == target:
            target_hash_calls += 1
            if target_hash_calls == 2:
                payload = target.read_bytes()
                replacement = bytes([payload[0] ^ 1]) + payload[1:]
                target.write_bytes(replacement)
                mutated = True
        return digest

    monkeypatch.setattr(converter, "_sha256_descriptor", mutate_after_hash)
    with pytest.raises((ValueError, RuntimeError)):
        converter.validate_payload_manifest(staging)
    assert mutated


def test_manifest_rejects_full_staging_path_replacement_before_manifest_read(
    generated_staging, tmp_path, monkeypatch
):
    staging = _copy_staging(generated_staging, tmp_path)
    _write_manifest(staging)
    displaced = tmp_path / "displaced-staging"
    original_validate = converter._validate_descriptor_path
    staging_validations = 0
    replaced = False

    def replace_after_initial_enumeration(fd, path, **kwargs):
        nonlocal staging_validations, replaced
        result = original_validate(fd, path, **kwargs)
        if path == staging:
            staging_validations += 1
            if staging_validations == 2:
                staging.rename(displaced)
                shutil.copytree(displaced, staging)
                replaced = True
        return result

    monkeypatch.setattr(
        converter,
        "_validate_descriptor_path",
        replace_after_initial_enumeration,
    )
    with pytest.raises(RuntimeError, match="identity"):
        converter.validate_payload_manifest(staging)
    assert replaced


def test_manifest_rejects_precall_root_replacement_with_stale_status_identity(
    generated_staging, tmp_path
):
    staging = _copy_staging(generated_staging, tmp_path)
    _write_manifest(staging)
    displaced = tmp_path / "displaced-before-validation"
    staging.rename(displaced)
    shutil.copytree(displaced, staging)

    with pytest.raises(RuntimeError, match="status|staging identity"):
        converter.validate_payload_manifest(staging)


@pytest.mark.parametrize(
    "unsafe_mode",
    [stat.S_IFSOCK | 0o600, stat.S_IFBLK | 0o600, stat.S_IFCHR | 0o600],
)
def test_manifest_type_predicate_rejects_all_nonregular_inode_modes(unsafe_mode):
    with pytest.raises(RuntimeError, match="not a regular file"):
        converter._validate_manifest_member_metadata(
            SimpleNamespace(st_mode=unsafe_mode, st_nlink=1),
            "meta/unsafe",
        )
