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
                metadata = (staging / "meta").stat()
                os.utime(
                    staging / "meta",
                    ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000),
                )
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


def _identity() -> converter.ConverterIdentity:
    return converter.ConverterIdentity("a" * 40, "b" * 64)


def _status(staging_or_output: Path) -> dict[str, object]:
    return json.loads((staging_or_output / "CONVERSION_STATUS.json").read_bytes())


def _assert_sealed_dataset(root: Path) -> None:
    assert stat.S_IMODE(root.stat().st_mode) == 0o555
    for directory, directory_names, file_names in os.walk(root):
        directory_names.sort()
        file_names.sort()
        assert stat.S_IMODE(Path(directory).stat().st_mode) == 0o555
        for name in file_names:
            path = Path(directory) / name
            assert stat.S_ISREG(path.lstat().st_mode)
            assert path.stat().st_nlink == 1
            assert stat.S_IMODE(path.stat().st_mode) == 0o444


def test_publication_filesystem_uses_fchmod_and_seals_file_and_directory(
    tmp_path, monkeypatch
):
    assert os.chmod not in os.supports_follow_symlinks
    filesystem = converter.PublicationFilesystem()
    regular = tmp_path / "payload"
    regular.write_bytes(b"payload")
    directory = tmp_path / "directory"
    directory.mkdir()
    opened_descriptors = []
    original_fchmod = os.fchmod

    def record_fchmod(fd, mode):
        opened_descriptors.append((fd, mode, os.fstat(fd).st_ino))
        original_fchmod(fd, mode)

    monkeypatch.setattr(converter.os, "fchmod", record_fchmod)
    filesystem.chmod(regular, 0o444)
    filesystem.chmod(directory, 0o555)

    assert [item[1] for item in opened_descriptors] == [0o444, 0o555]
    assert opened_descriptors[0][2] == regular.stat().st_ino
    assert opened_descriptors[1][2] == directory.stat().st_ino
    assert stat.S_IMODE(regular.stat().st_mode) == 0o444
    assert stat.S_IMODE(directory.stat().st_mode) == 0o555


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_publication_filesystem_chmod_rejects_unsafe_targets(tmp_path, kind):
    filesystem = converter.PublicationFilesystem()
    target = tmp_path / "target"
    if kind == "symlink":
        regular = tmp_path / "regular"
        regular.write_bytes(b"payload")
        target.symlink_to(regular)
    else:
        os.mkfifo(target)

    with pytest.raises((OSError, RuntimeError)):
        filesystem.chmod(target, 0o444)


def test_publication_filesystem_chmod_detects_post_open_path_swap(
    tmp_path, monkeypatch
):
    filesystem = converter.PublicationFilesystem()
    target = tmp_path / "target"
    target.write_bytes(b"original")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replacement")
    original_fchmod = os.fchmod

    def swap_after_fchmod(fd, mode):
        original_fchmod(fd, mode)
        target.rename(tmp_path / "displaced")
        replacement.rename(target)

    monkeypatch.setattr(converter.os, "fchmod", swap_after_fchmod)
    with pytest.raises(RuntimeError, match="replaced|identity"):
        filesystem.chmod(target, 0o444)


def test_atomic_replace_status_exposes_all_durability_boundaries(tmp_path):
    status = tmp_path / "CONVERSION_STATUS.json"
    status.write_bytes(b"old\n")
    events = []
    payload = b'{"state":"complete"}\n'

    converter.atomic_replace_status(status, payload, 0o644, events.append)

    assert events == [
        "after_complete_temp_creation",
        "before_complete_temp_fsync",
        "after_complete_temp_fsync",
        "after_complete_status_rename",
    ]
    assert status.read_bytes() == payload
    assert not list(tmp_path.glob(".CONVERSION_STATUS.json.tmp-*"))


@pytest.mark.parametrize("kind", ["symlink", "fifo", "hard_link"])
def test_atomic_replace_status_rejects_unsafe_existing_target(tmp_path, kind):
    status = tmp_path / "CONVERSION_STATUS.json"
    if kind == "symlink":
        target = tmp_path / "target"
        target.write_bytes(b"target")
        status.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(status)
    else:
        target = tmp_path / "target"
        target.write_bytes(b"target")
        os.link(target, status)

    with pytest.raises((OSError, RuntimeError)):
        converter.atomic_replace_status(status, b"new\n", 0o644, lambda _: None)


def test_atomic_replace_status_has_no_cross_device_fallback(tmp_path, monkeypatch):
    status = tmp_path / "CONVERSION_STATUS.json"
    status.write_bytes(b"old\n")

    def fail_replace(*args, **kwargs):
        del args, kwargs
        raise OSError(converter.errno.EXDEV, "cross-device")

    monkeypatch.setattr(converter.os, "replace", fail_replace)
    with pytest.raises(OSError) as caught:
        converter.atomic_replace_status(status, b"new\n", 0o644, lambda _: None)
    assert caught.value.errno == converter.errno.EXDEV
    assert status.read_bytes() == b"old\n"
    assert not list(tmp_path.glob(".CONVERSION_STATUS.json.tmp-*"))


class _RecordingFilesystem(converter.PublicationFilesystem):
    def __init__(self):
        self.events = []

    def fsync_file(self, path):
        super().fsync_file(path)
        self.events.append(("fsync_file", Path(path)))

    def fsync_directory(self, path):
        super().fsync_directory(path)
        self.events.append(("fsync_directory", Path(path)))

    def chmod(self, path, mode):
        super().chmod(path, mode)
        self.events.append(("chmod", Path(path), mode))

    def atomic_replace_status(self, path, payload, mode, fault):
        self.events.append(("atomic_replace_status:start", Path(path)))
        super().atomic_replace_status(path, payload, mode, fault)
        self.events.append(("atomic_replace_status:done", Path(path)))

    def rename_noreplace(self, source, destination, *, expected_identity=None):
        super().rename_noreplace(
            source,
            destination,
            expected_identity=expected_identity,
        )
        self.events.append(
            (
                "rename_noreplace",
                Path(source),
                Path(destination),
                expected_identity,
            )
        )


def test_durable_publication_orders_flush_fsync_seal_validate_and_rename(
    generated_staging, tmp_path
):
    staging = _copy_staging(generated_staging, tmp_path)
    output = tmp_path / "processed"
    filesystem = _RecordingFilesystem()
    faults = []

    result = converter.publish_staged_dataset(
        staging, output, _identity(), filesystem, faults.append
    )

    assert result.state == "published"
    assert result.dataset_root == output
    assert not staging.exists()
    assert converter.validate_payload_manifest(output)["schema_version"] == 1
    assert _status(output)["state"] == "complete"
    _assert_sealed_dataset(output)
    assert faults[0:2] == ["after_payload_close", "after_staged_validation"]
    assert (
        faults.index("after_manifest_file_fsync")
        < faults.index("after_manifest_rename")
        < faults.index("after_meta_fsync")
    )
    assert faults.index("after_precomplete_revalidation") < faults.index(
        "before_complete_temp_fsync"
    )
    assert (
        faults.index("after_complete_temp_fsync")
        < faults.index("after_complete_status_rename")
        < faults.index("after_complete_root_fsync")
    )
    assert (
        faults.index("after_final_revalidation")
        < faults.index("before_publication_rename")
        < faults.index("after_publication_rename")
    )
    assert faults[-1] == "after_destination_parent_fsync"
    rename_index = next(
        index
        for index, event in enumerate(filesystem.events)
        if event[0] == "rename_noreplace"
    )
    parent_fsync_index = max(
        index
        for index, event in enumerate(filesystem.events)
        if event[0] == "fsync_directory" and event[1] == output.parent
    )
    assert rename_index < parent_fsync_index


@pytest.mark.parametrize(
    ("point", "state"),
    [
        ("after_payload_close", "pre_completion_failed"),
        ("after_staged_validation", "pre_completion_failed"),
        ("after_manifest_file_fsync", "pre_completion_failed"),
        ("after_manifest_rename", "pre_completion_failed"),
        ("after_meta_fsync", "pre_completion_failed"),
        ("after_precomplete_revalidation", "pre_completion_failed"),
        ("after_complete_temp_creation", "completion_uncertain_unpublished"),
        ("before_complete_temp_fsync", "completion_uncertain_unpublished"),
        ("after_complete_temp_fsync", "completion_uncertain_unpublished"),
        ("after_complete_status_rename", "completion_uncertain_unpublished"),
        ("after_complete_root_fsync", "complete_unpublished"),
        ("after_final_revalidation", "complete_unpublished"),
        ("before_publication_rename", "complete_unpublished"),
    ],
)
def test_durable_publication_fault_classification(
    generated_staging, tmp_path, point, state
):
    staging = _copy_staging(generated_staging, tmp_path)
    output = tmp_path / "processed"

    def fail(selected):
        if selected == point:
            raise OSError(f"fault:{point}")

    with pytest.raises(converter.PublicationStateError) as caught:
        converter.publish_staged_dataset(
            staging,
            output,
            _identity(),
            converter.PublicationFilesystem(),
            fail,
        )

    assert caught.value.state == state
    assert staging.is_dir()
    assert not output.exists()
    visible_state = _status(staging)["state"]
    if state == "pre_completion_failed":
        assert visible_state == "failed"
        assert stat.S_IMODE(staging.stat().st_mode) == 0o755
    elif state == "complete_unpublished":
        assert visible_state == "complete"
    else:
        assert visible_state in {"in_progress", "complete"}


def test_every_dynamic_durability_fault_point_is_exercised(generated_staging, tmp_path):
    observed = []
    staging = _copy_staging(generated_staging, tmp_path)
    converter.publish_staged_dataset(
        staging,
        tmp_path / "processed",
        _identity(),
        converter.PublicationFilesystem(),
        observed.append,
    )
    prefixes = {
        "after_each_payload_fsync:",
        "after_each_precomplete_directory_fsync:",
        "after_each_chmod:",
        "after_each_final_file_fsync:",
        "after_each_final_directory_fsync:",
    }
    for prefix in prefixes:
        points = [point for point in observed if point.startswith(prefix)]
        assert points, prefix


def test_complete_status_root_fsync_failure_is_completion_uncertain(
    generated_staging, tmp_path, monkeypatch
):
    staging = _copy_staging(generated_staging, tmp_path)
    output = tmp_path / "processed"
    after_status_rename = False
    original_fsync = os.fsync

    def observe(point):
        nonlocal after_status_rename
        if point == "after_complete_status_rename":
            after_status_rename = True

    def fail_root_fsync(fd):
        if after_status_rename and stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("complete root fsync failed")
        original_fsync(fd)

    monkeypatch.setattr(converter.os, "fsync", fail_root_fsync)
    with pytest.raises(converter.PublicationStateError) as caught:
        converter.publish_staged_dataset(
            staging,
            output,
            _identity(),
            converter.PublicationFilesystem(),
            observe,
        )
    assert caught.value.state == "completion_uncertain_unpublished"
    assert _status(staging)["state"] == "complete"
    assert not output.exists()


def test_precomplete_files_are_fsynced_before_bottom_up_directories(
    generated_staging, tmp_path
):
    staging = _copy_staging(generated_staging, tmp_path)
    observed = []
    converter.publish_staged_dataset(
        staging,
        tmp_path / "processed",
        _identity(),
        converter.PublicationFilesystem(),
        observed.append,
    )
    file_points = [
        index
        for index, point in enumerate(observed)
        if point.startswith("after_each_payload_fsync:")
    ]
    directory_points = [
        (index, point.split(":", 1)[1])
        for index, point in enumerate(observed)
        if point.startswith("after_each_precomplete_directory_fsync:")
    ]
    assert max(file_points) < min(index for index, _ in directory_points)
    directory_paths = [path for _, path in directory_points]
    assert directory_paths[-1] == "."
    assert [path.count("/") for path in directory_paths] == sorted(
        [path.count("/") for path in directory_paths], reverse=True
    )

    output = tmp_path / "processed"
    expected_files = _enumerated_regular_files(output)
    observed_files = {
        point.split(":", 1)[1]
        for point in observed
        if point.startswith("after_each_payload_fsync:")
    }
    assert observed_files == expected_files
    chmod_indices = [
        index
        for index, point in enumerate(observed)
        if point.startswith("after_each_chmod:")
    ]
    final_file_indices = [
        index
        for index, point in enumerate(observed)
        if point.startswith("after_each_final_file_fsync:")
    ]
    final_directory_points = [
        (index, point.split(":", 1)[1])
        for index, point in enumerate(observed)
        if point.startswith("after_each_final_directory_fsync:")
    ]
    assert max(chmod_indices) < min(final_file_indices)
    assert max(final_file_indices) < min(index for index, _ in final_directory_points)
    assert final_directory_points[-1][1] == "."


def test_every_dynamic_fault_preserves_the_required_disk_state(
    generated_staging, tmp_path
):
    discovery_root = tmp_path / "discovery"
    discovery_root.mkdir()
    discovery_staging = _copy_staging(generated_staging, discovery_root)
    observed = []
    converter.publish_staged_dataset(
        discovery_staging,
        discovery_root / "processed",
        _identity(),
        converter.PublicationFilesystem(),
        observed.append,
    )
    dynamic = [
        point
        for point in observed
        if point.startswith(
            (
                "after_each_payload_fsync:",
                "after_each_precomplete_directory_fsync:",
                "after_each_chmod:",
                "after_each_final_file_fsync:",
                "after_each_final_directory_fsync:",
            )
        )
    ]
    assert len(dynamic) == len(set(dynamic))

    for index, target in enumerate(dynamic):
        case = tmp_path / f"fault-{index}"
        case.mkdir()
        staging = _copy_staging(generated_staging, case)
        output = case / "processed"

        def fail(point):
            if point == target:
                raise OSError(f"fault:{target}")

        with pytest.raises(converter.PublicationStateError) as caught:
            converter.publish_staged_dataset(
                staging,
                output,
                _identity(),
                converter.PublicationFilesystem(),
                fail,
            )
        expected = (
            "pre_completion_failed"
            if target.startswith(
                (
                    "after_each_payload_fsync:",
                    "after_each_precomplete_directory_fsync:",
                )
            )
            else "complete_unpublished"
        )
        assert caught.value.state == expected, target
        assert staging.is_dir(), target
        assert not output.exists(), target
        assert _status(staging)["state"] == (
            "failed" if expected == "pre_completion_failed" else "complete"
        )


def test_destination_collision_preserves_staging_and_destination(
    generated_staging, tmp_path
):
    staging = _copy_staging(generated_staging, tmp_path)
    output = tmp_path / "processed"
    output.mkdir()
    marker = output / "existing"
    marker.write_bytes(b"untouched")

    with pytest.raises(converter.PublicationStateError) as caught:
        converter.publish_staged_dataset(
            staging,
            output,
            _identity(),
            converter.PublicationFilesystem(),
            lambda _: None,
        )

    assert caught.value.state == "complete_unpublished"
    assert staging.is_dir()
    assert _status(staging)["state"] == "complete"
    assert marker.read_bytes() == b"untouched"


def test_prepublication_staging_root_swap_cannot_publish_attacker_tree(
    generated_staging, tmp_path
):
    staging = _copy_staging(generated_staging, tmp_path)
    output = tmp_path / "processed"
    displaced = tmp_path / "validated-complete-staging"

    def swap_at_boundary(point):
        if point != "before_publication_rename":
            return
        staging.rename(displaced)
        shutil.copytree(displaced, staging)
        staging.chmod(0o755)
        (staging / "attacker-diagnostic").write_bytes(b"not validated")

    with pytest.raises(converter.PublicationStateError) as caught:
        converter.publish_staged_dataset(
            staging,
            output,
            _identity(),
            converter.PublicationFilesystem(),
            swap_at_boundary,
        )

    assert caught.value.state == "complete_unpublished"
    assert not output.exists()
    assert staging.is_dir()
    assert (staging / "attacker-diagnostic").read_bytes() == b"not validated"
    assert displaced.is_dir()
    assert _status(displaced)["state"] == "complete"
    _assert_sealed_dataset(displaced)


def test_prepublication_manifest_member_mutation_is_revalidated(
    generated_staging, tmp_path
):
    staging = _copy_staging(generated_staging, tmp_path)
    output = tmp_path / "processed"
    info = staging / "meta" / "info.json"

    def mutate_at_boundary(point):
        if point != "before_publication_rename":
            return
        original = info.read_bytes()
        mutated = original.replace(b'"robot_type":"g1"', b'"robot_type":"x1"')
        assert mutated != original
        assert len(mutated) == len(original)
        info.chmod(0o644)
        info.write_bytes(mutated)
        info.chmod(0o444)

    with pytest.raises(converter.PublicationStateError) as caught:
        converter.publish_staged_dataset(
            staging,
            output,
            _identity(),
            converter.PublicationFilesystem(),
            mutate_at_boundary,
        )

    assert caught.value.state == "complete_unpublished"
    assert staging.is_dir()
    assert not output.exists()
    assert _status(staging)["state"] == "complete"
    assert stat.S_IMODE(info.stat().st_mode) == 0o444


def test_publication_parent_swap_cannot_claim_durable_output(
    generated_staging, tmp_path
):
    parent = tmp_path / "output-parent"
    parent.mkdir()
    staging = _copy_staging(generated_staging, parent)
    output = parent / "processed"
    displaced_parent = tmp_path / "displaced-output-parent"

    def swap_after_rename(point):
        if point != "after_publication_rename":
            return
        parent.rename(displaced_parent)
        parent.mkdir()

    with pytest.raises(converter.PublicationUncertainError) as caught:
        converter.publish_staged_dataset(
            staging,
            output,
            _identity(),
            converter.PublicationFilesystem(),
            swap_after_rename,
        )

    assert caught.value.publication.state == "publication_uncertain"
    assert not output.exists()
    assert (displaced_parent / "processed").is_dir()
    assert _status(displaced_parent / "processed")["state"] == "complete"


def test_post_rename_validation_failure_remains_publication_uncertain(
    generated_staging, tmp_path
):
    staging = _copy_staging(generated_staging, tmp_path)
    output = tmp_path / "processed"
    displaced = tmp_path / "renamed-but-not-validated"

    class MoveDestinationAfterRename(converter.PublicationFilesystem):
        def rename_noreplace(self, source, destination, *, expected_identity=None):
            super().rename_noreplace(
                source,
                destination,
                expected_identity=expected_identity,
            )
            destination.rename(displaced)

    with pytest.raises(converter.PublicationUncertainError) as caught:
        converter.publish_staged_dataset(
            staging,
            output,
            _identity(),
            MoveDestinationAfterRename(),
            lambda _: None,
        )

    assert caught.value.publication.state == "publication_uncertain"
    assert not staging.exists()
    assert not output.exists()
    assert displaced.is_dir()
    assert _status(displaced)["state"] == "complete"


def test_destination_parent_durability_uses_held_parent_descriptor(
    generated_staging, tmp_path, monkeypatch
):
    parent = tmp_path / "output-parent"
    parent.mkdir()
    parent_identity = (parent.stat().st_dev, parent.stat().st_ino)
    staging = _copy_staging(generated_staging, parent)
    output = parent / "processed"
    displaced_parent = tmp_path / "temporarily-displaced-parent"
    fsynced_directories = []
    original_fsync = os.fsync

    def record_fsync(fd):
        metadata = os.fstat(fd)
        if stat.S_ISDIR(metadata.st_mode):
            fsynced_directories.append((metadata.st_dev, metadata.st_ino))
        original_fsync(fd)

    class SwapAndRestoreParentDuringPathFsync(converter.PublicationFilesystem):
        def fsync_directory(self, path):
            if Path(path) != parent or not output.exists() or staging.exists():
                return super().fsync_directory(path)
            parent.rename(displaced_parent)
            parent.mkdir()
            try:
                super().fsync_directory(parent)
            finally:
                parent.rmdir()
                displaced_parent.rename(parent)

    monkeypatch.setattr(converter.os, "fsync", record_fsync)
    result = converter.publish_staged_dataset(
        staging,
        output,
        _identity(),
        SwapAndRestoreParentDuringPathFsync(),
        lambda _: None,
    )

    assert result.state == "published"
    assert parent_identity in fsynced_directories
    assert output.is_dir()


def test_held_parent_descriptor_fsync_failure_is_publication_uncertain(
    generated_staging, tmp_path, monkeypatch
):
    parent = tmp_path / "output-parent"
    parent.mkdir()
    parent_identity = (parent.stat().st_dev, parent.stat().st_ino)
    staging = _copy_staging(generated_staging, parent)
    output = parent / "processed"
    original_fsync = os.fsync
    publication_parent_fsyncs = 0

    def fail_second_publication_parent_fsync(fd):
        nonlocal publication_parent_fsyncs
        metadata = os.fstat(fd)
        identity = (metadata.st_dev, metadata.st_ino)
        if (
            stat.S_ISDIR(metadata.st_mode)
            and identity == parent_identity
            and output.exists()
            and not staging.exists()
        ):
            publication_parent_fsyncs += 1
            if publication_parent_fsyncs == 2:
                raise OSError("held parent descriptor fsync failed")
        original_fsync(fd)

    monkeypatch.setattr(converter.os, "fsync", fail_second_publication_parent_fsync)
    with pytest.raises(converter.PublicationUncertainError) as caught:
        converter.publish_staged_dataset(
            staging,
            output,
            _identity(),
            converter.PublicationFilesystem(),
            lambda _: None,
        )

    assert publication_parent_fsyncs == 2
    assert caught.value.publication.state == "publication_uncertain"
    assert output.is_dir()


@pytest.mark.parametrize("failed_validation", ["path", "entry"])
def test_no_fallible_revalidation_follows_authoritative_parent_fsync(
    generated_staging, tmp_path, monkeypatch, failed_validation
):
    staging = _copy_staging(generated_staging, tmp_path)
    output = tmp_path / "processed"
    captured_parent_fd = None
    authoritative_parent_fsync = False
    post_fsync_revalidations = []
    original_validate_entry = converter._validate_descriptor_entry
    original_validate_path = converter._validate_descriptor_path
    original_fsync = os.fsync

    def capture_parent_descriptor(
        fd,
        parent_fd,
        name,
        *,
        expected_type,
        require_single_link=False,
    ):
        nonlocal captured_parent_fd
        result = original_validate_entry(
            fd,
            parent_fd,
            name,
            expected_type=expected_type,
            require_single_link=require_single_link,
        )
        if name == os.fsencode(output.name) and expected_type == "directory":
            captured_parent_fd = parent_fd
            if authoritative_parent_fsync and failed_validation == "entry":
                post_fsync_revalidations.append("entry")
                raise OSError("entry revalidation followed authoritative parent fsync")
        return result

    def record_authoritative_fsync(fd):
        nonlocal authoritative_parent_fsync
        original_fsync(fd)
        if captured_parent_fd == fd:
            authoritative_parent_fsync = True

    def reject_post_fsync_revalidation(
        fd,
        path,
        *,
        expected_type,
        require_single_link=False,
    ):
        if (
            authoritative_parent_fsync
            and failed_validation == "path"
            and Path(path) == output.parent
        ):
            post_fsync_revalidations.append("path")
            raise OSError("revalidation followed authoritative parent fsync")
        return original_validate_path(
            fd,
            path,
            expected_type=expected_type,
            require_single_link=require_single_link,
        )

    monkeypatch.setattr(
        converter,
        "_validate_descriptor_entry",
        capture_parent_descriptor,
    )
    monkeypatch.setattr(converter.os, "fsync", record_authoritative_fsync)
    monkeypatch.setattr(
        converter,
        "_validate_descriptor_path",
        reject_post_fsync_revalidation,
    )

    result = converter.publish_staged_dataset(
        staging,
        output,
        _identity(),
        converter.PublicationFilesystem(),
        lambda _: None,
    )

    assert authoritative_parent_fsync
    assert post_fsync_revalidations == []
    assert result.state == "published"
    assert output.is_dir()


@pytest.mark.parametrize("failed_close", ["staging", "parent"])
def test_descriptor_close_failure_after_parent_durability_is_published(
    generated_staging, tmp_path, monkeypatch, failed_close
):
    staging = _copy_staging(generated_staging, tmp_path)
    output = tmp_path / "processed"
    captured = {"staging": None, "parent": None}
    authoritative_parent_fsync = False
    close_failed = False
    original_validate = converter._validate_descriptor_entry
    original_fsync = os.fsync
    original_close = os.close

    def capture_publisher_descriptors(
        fd,
        parent_fd,
        name,
        *,
        expected_type,
        require_single_link=False,
    ):
        result = original_validate(
            fd,
            parent_fd,
            name,
            expected_type=expected_type,
            require_single_link=require_single_link,
        )
        if name == os.fsencode(output.name) and expected_type == "directory":
            captured["staging"] = fd
            captured["parent"] = parent_fd
        return result

    def record_authoritative_fsync(fd):
        nonlocal authoritative_parent_fsync
        original_fsync(fd)
        if captured["parent"] == fd:
            authoritative_parent_fsync = True

    def close_then_fail(fd):
        nonlocal close_failed
        original_close(fd)
        if (
            authoritative_parent_fsync
            and not close_failed
            and captured[failed_close] == fd
        ):
            close_failed = True
            raise OSError(f"injected {failed_close} descriptor close failure")

    monkeypatch.setattr(
        converter,
        "_validate_descriptor_entry",
        capture_publisher_descriptors,
    )
    monkeypatch.setattr(converter.os, "fsync", record_authoritative_fsync)
    monkeypatch.setattr(converter.os, "close", close_then_fail)

    with pytest.raises(converter.PublishedBoundaryError) as caught:
        converter.publish_staged_dataset(
            staging,
            output,
            _identity(),
            converter.PublicationFilesystem(),
            lambda _: None,
        )

    assert close_failed
    assert caught.value.publication.state == "published"
    assert caught.value.publication.dataset_root == output
    assert output.is_dir()
    assert not staging.exists()
    assert converter.validate_payload_manifest(output)["schema_version"] == 1
    assert _status(output)["state"] == "complete"
    _assert_sealed_dataset(output)
    for fd in captured.values():
        assert fd is not None
        with pytest.raises(OSError):
            os.fstat(fd)


def test_after_publication_rename_is_uncertain_and_visible(generated_staging, tmp_path):
    staging = _copy_staging(generated_staging, tmp_path)
    output = tmp_path / "processed"

    def fail(point):
        if point == "after_publication_rename":
            raise OSError("crash after rename")

    with pytest.raises(converter.PublicationUncertainError) as caught:
        converter.publish_staged_dataset(
            staging,
            output,
            _identity(),
            converter.PublicationFilesystem(),
            fail,
        )

    assert caught.value.publication.state == "publication_uncertain"
    assert caught.value.publication.dataset_root == output
    assert output.is_dir()
    assert not staging.exists()
    assert _status(output)["state"] == "complete"
    _assert_sealed_dataset(output)


def test_destination_parent_fsync_failure_is_publication_uncertain(
    generated_staging, tmp_path
):
    staging = _copy_staging(generated_staging, tmp_path)
    output = tmp_path / "processed"

    class FailDestinationParentFsync(converter.PublicationFilesystem):
        def fsync_directory(self, path):
            if Path(path) == output.parent and output.exists() and not staging.exists():
                raise OSError("destination parent fsync failed")
            super().fsync_directory(path)

    with pytest.raises(converter.PublicationUncertainError) as caught:
        converter.publish_staged_dataset(
            staging,
            output,
            _identity(),
            FailDestinationParentFsync(),
            lambda _: None,
        )
    assert caught.value.publication.state == "publication_uncertain"
    assert output.is_dir()
    assert not staging.exists()
    assert _status(output)["state"] == "complete"


def test_rename_helper_post_syscall_failure_is_publication_uncertain(
    generated_staging, tmp_path
):
    staging = _copy_staging(generated_staging, tmp_path)
    output = tmp_path / "processed"

    class FailAfterRename(converter.PublicationFilesystem):
        def rename_noreplace(self, source, destination, *, expected_identity=None):
            super().rename_noreplace(
                source,
                destination,
                expected_identity=expected_identity,
            )
            raise OSError("post-rename validation was interrupted")

    with pytest.raises(converter.PublicationUncertainError) as caught:
        converter.publish_staged_dataset(
            staging,
            output,
            _identity(),
            FailAfterRename(),
            lambda _: None,
        )
    assert caught.value.publication.state == "publication_uncertain"
    assert output.is_dir()
    assert not staging.exists()
    assert _status(output)["state"] == "complete"


def test_complete_is_never_rewritten_to_failed(generated_staging, tmp_path):
    staging = _copy_staging(generated_staging, tmp_path)
    filesystem = _RecordingFilesystem()

    def fail(point):
        if point == "after_complete_root_fsync":
            raise OSError("after durable complete")

    with pytest.raises(converter.PublicationStateError) as caught:
        converter.publish_staged_dataset(
            staging,
            tmp_path / "processed",
            _identity(),
            filesystem,
            fail,
        )
    assert caught.value.state == "complete_unpublished"
    assert _status(staging)["state"] == "complete"
    assert [event[0] for event in filesystem.events].count(
        "atomic_replace_status:start"
    ) == 1


def test_after_destination_parent_fsync_exception_is_durably_published(
    generated_staging, tmp_path
):
    staging = _copy_staging(generated_staging, tmp_path)
    output = tmp_path / "processed"

    def fail(point):
        if point == "after_destination_parent_fsync":
            raise OSError("post-publication hook")

    with pytest.raises(converter.PublishedBoundaryError) as caught:
        converter.publish_staged_dataset(
            staging,
            output,
            _identity(),
            converter.PublicationFilesystem(),
            fail,
        )

    assert caught.value.publication.state == "published"
    assert caught.value.publication.dataset_root == output
    assert output.is_dir()
    assert not staging.exists()
    assert converter.validate_payload_manifest(output)["schema_version"] == 1
    assert _status(output)["state"] == "complete"
    _assert_sealed_dataset(output)


def _exit_at_publication_fault(staging, output, point):
    def exit_at(selected):
        if selected == point:
            os._exit(73)

    converter.publish_staged_dataset(
        staging,
        output,
        _identity(),
        converter.PublicationFilesystem(),
        exit_at,
    )


@pytest.mark.parametrize(
    "point",
    [
        "before_complete_temp_fsync",
        "after_complete_temp_creation",
        "after_complete_temp_fsync",
        "after_complete_status_rename",
        "after_complete_root_fsync",
        "before_publication_rename",
        "after_publication_rename",
    ],
)
def test_crash_at_complete_and_publication_boundaries_is_inspectable(
    generated_staging, tmp_path, point
):
    staging = _copy_staging(generated_staging, tmp_path)
    output = tmp_path / "processed"
    context = multiprocessing.get_context("fork")
    child = context.Process(
        target=_exit_at_publication_fault,
        args=(staging, output, point),
    )
    child.start()
    child.join(timeout=30)
    assert child.exitcode == 73

    with converter.conversion_lock(output):
        state = converter.inspect_publication_state(staging, output)
    if point in {"after_publication_rename"}:
        # Restart inspection establishes destination-parent durability.
        assert state == "published"
        assert output.is_dir()
        assert not staging.exists()
    elif point == "before_publication_rename":
        assert state == "complete_unpublished"
        assert staging.is_dir()
    else:
        # A crash cannot prove the complete-status root fsync occurred until
        # sealing provides the next durable on-disk milestone.
        assert state == "completion_uncertain_unpublished"
        assert staging.is_dir()
    assert not list(tmp_path.glob(".processed.certification-*"))


def test_after_destination_parent_fsync_exit_restarts_as_published(
    generated_staging, tmp_path
):
    staging = _copy_staging(generated_staging, tmp_path)
    output = tmp_path / "processed"
    context = multiprocessing.get_context("fork")
    child = context.Process(
        target=_exit_at_publication_fault,
        args=(staging, output, "after_destination_parent_fsync"),
    )
    child.start()
    child.join(timeout=30)
    assert child.exitcode == 73

    with converter.conversion_lock(output):
        assert converter.inspect_publication_state(staging, output) == "published"
        root_fd = os.open(
            output,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        os.close(root_fd)
        assert converter.validate_payload_manifest(output)["schema_version"] == 1
        assert _status(output)["state"] == "complete"
        _assert_sealed_dataset(output)
        assert not staging.exists()
    assert not list(tmp_path.glob(".processed.certification-*"))
