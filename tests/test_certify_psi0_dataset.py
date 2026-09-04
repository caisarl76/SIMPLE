import importlib.util
import json
import os
import py_compile
import shutil
import stat
import struct
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from psi0_converter_fixtures import make_source_episode
from scripts import certify_psi0_dataset as certifier
from scripts import postprocess_psi0 as converter
from scripts.certify_psi0_dataset import (
    DatasetExpectations,
    DatasetValidationError,
    PublishedUncertifiedError,
    certify_published_dataset as _production_certify_published_dataset,
    validate_dataset,
    validate_evidence_terminal,
)


VECTOR_FIELDS = {
    "states": 32,
    "action": 36,
    "observation.hand_joints": 14,
    "observation.arm_joints": 14,
    "observation.leg_joints": 15,
    "observation.prev_torso_rpy": 3,
    "observation.prev_height": 1,
}
SCALAR_FIELDS = (
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "next.done",
)
LOADER_DENIED_AUDIT_EVENTS = sorted(
    (
        "os.exec",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.posix_spawnp",
        "os.system",
        "socket.__new__",
        "socket.bind",
        "socket.connect",
        "socket.connect_ex",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.gethostbyname_ex",
        "socket.getnameinfo",
        "socket.sendto",
        "subprocess.Popen",
    ),
    key=str.encode,
)


def certify_published_dataset(*args, **kwargs):
    """Keep pre-Task-8 evidence durability tests focused and fast."""

    psi0_root = Path(kwargs["psi0_root"]).absolute()
    repository_root = Path(__file__).resolve().parents[1]
    if psi0_root != repository_root:
        return _production_certify_published_dataset(*args, **kwargs)
    original = certifier._run_psi0_loader
    original_collect = certifier._collect_psi0_environment
    should_stub_collection = (
        getattr(original_collect, "__name__", "") == "_collect_psi0_environment"
    )
    if should_stub_collection:

        def collect_stub(root, commit, python):
            details = {
                "psi0_root": str(root.absolute()),
                "psi0_src": str(root.absolute() / "src"),
                "psi0_commit": commit,
                "tracked_status": [],
                "python": str(python.absolute()),
                "python_realpath": str(python.resolve(strict=False)),
                "python_version": "task-7-test",
                "platform": "task-7-test",
                "packages": {
                    name: "task-7-test"
                    for name in (
                        "av",
                        "datasets",
                        "numpy",
                        "pyarrow",
                        "torch",
                        "torchvision",
                    )
                },
                "distributions": [],
            }
            details["compat_module"] = {
                "commit": commit,
                "relative_path": "src/psi/data/lerobot/compat.py",
                "origin": str(root.absolute() / "src/psi/data/lerobot/compat.py"),
                "blob_sha256": "0" * 64,
            }
            details["distribution_manifest_sha256"] = converter.sha256_bytes(
                converter.canonical_json_bytes(details["distributions"])
            )
            return {
                "schema_version": 1,
                "details": details,
                "details_sha256": converter.sha256_bytes(
                    converter.canonical_json_bytes(details)
                ),
            }

        certifier._collect_psi0_environment = collect_stub

    def loader_stub(**loader_kwargs):
        validation = loader_kwargs["result"]
        evidence_root = loader_kwargs["evidence_root"]
        environment = certifier._worker_environment(evidence_root / ".loader-cache")
        episodes = [
            json.loads(line)
            for line in (validation.dataset_root / "meta/episodes.jsonl")
            .read_text()
            .splitlines()
        ]
        info = json.loads((validation.dataset_root / "meta/info.json").read_text())
        image_shape = info["features"]["observation.images.egocentric"]["shape"]
        result_value = {
            "schema_version": 1,
            "verdict": "PASS",
            "sys_path_0": str(loader_kwargs["psi0_root"].absolute() / "src"),
            "visited_indices": list(range(validation.total_frames)),
            "episode_ranges": [
                [row["dataset_from_index"], row["dataset_to_index"]] for row in episodes
            ],
            "offline_environment": environment,
            "tensor_contract": {
                "action": {"dtype": "torch.float32", "shape": [36]},
                "observation.images.egocentric": {
                    "dtype": "torch.float32",
                    "shape": [3, image_shape[0], image_shape[1]],
                },
                "states": {"dtype": "torch.float32", "shape": [32]},
            },
            "module_provenance": {
                "commit": loader_kwargs["psi0_commit"],
                "relative_path": "src/psi/data/lerobot/compat.py",
                "origin": str(
                    loader_kwargs["psi0_root"].absolute()
                    / "src/psi/data/lerobot/compat.py"
                ),
                "blob_sha256": "0" * 64,
            },
            "network_policy": {
                "audit_hook": "deny_inet_resolver_process_escape",
                "denied_events": LOADER_DENIED_AUDIT_EVENTS,
                "violations": [],
            },
        }
        result_name = ".psi0-loader-result-task-7-test.tmp"
        argv = [
            str(loader_kwargs["python"].absolute()),
            "-I",
            "scripts/certify_psi0_dataset.py",
            "--loader-worker",
            "--psi0-src",
            str(loader_kwargs["psi0_root"].absolute() / "src"),
            "--dataset-root",
            str(validation.dataset_root),
            "--result",
            str(evidence_root / result_name),
        ]
        return (
            {
                "schema_version": 1,
                "argv": argv,
                "environment": environment,
                "environment_sha256": converter.sha256_bytes(
                    converter.canonical_json_bytes(environment)
                ),
            },
            {
                "schema_version": 1,
                "entries": [
                    {
                        "path": name,
                        "type": "directory",
                        "size": 0,
                        "sha256": None,
                    }
                    for name in sorted(
                        ("home", "hf", "datasets", "xdg", "torch", "tmp"),
                        key=str.encode,
                    )
                ],
            },
            {
                "schema_version": 1,
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "result_sha256": converter.sha256_bytes(
                    converter.canonical_json_bytes(result_value)
                ),
                "result": result_value,
            },
        )

    certifier._run_psi0_loader = loader_stub
    try:
        return _production_certify_published_dataset(*args, **kwargs)
    finally:
        certifier._run_psi0_loader = original
        if should_stub_collection:
            certifier._collect_psi0_environment = original_collect


def _make_fake_psi0_checkout(
    root: Path,
    *,
    behavior: str = "valid",
) -> tuple[Path, str, Path]:
    checkout = root / f"psi0-{behavior}"
    package = checkout / "src/psi/data/lerobot"
    package.mkdir(parents=True)
    for init in (
        checkout / "src/psi/__init__.py",
        checkout / "src/psi/data/__init__.py",
        checkout / "src/psi/data/lerobot/__init__.py",
    ):
        init.write_text("")
    import_marker = root / f"{behavior}-imported"
    module = f"""\
import json
import os
import sys
from pathlib import Path

import torch

LEROBOT_LAYOUT = "fake"
BEHAVIOR = {behavior!r}
IMPORT_MARKER = Path({str(import_marker)!r})
if BEHAVIOR == "pycache_prefix":
    expected_prefix = Path(os.environ["TMPDIR"]) / "pycache"
    assert sys.pycache_prefix == str(expected_prefix)
    assert expected_prefix.is_dir()
    assert not list(expected_prefix.iterdir())
if BEHAVIOR in ("sourceless_pyc", "race_sourceless_pyc"):
    import evil
if BEHAVIOR == "urllib3_import":
    from urllib3.util.connection import allowed_gai_family

    assert callable(allowed_gai_family)
IMPORT_MARKER.write_text("imported")


class LeRobotDataset:
    def __init__(self, *, repo_id, root):
        assert repo_id == "simple-certified"
        self.root = Path(root)
        info = json.loads((self.root / "meta/info.json").read_text())
        self.total = info["total_frames"]
        shape = info["features"]["observation.images.egocentric"]["shape"]
        self.height, self.width = shape[:2]
        cache_root = Path(os.environ["HOME"]).parent
        expected = {{
            "HOME": cache_root / "home",
            "HF_HOME": cache_root / "hf",
            "HF_DATASETS_CACHE": cache_root / "datasets",
            "XDG_CACHE_HOME": cache_root / "xdg",
            "TORCH_HOME": cache_root / "torch",
            "TMPDIR": cache_root / "tmp",
        }}
        assert all(Path(os.environ[key]) == value for key, value in expected.items())
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["PYTHONNOUSERSITE"] == "1"
        self.visits = Path(os.environ["TMPDIR"]) / "visited.jsonl"
        (Path(os.environ["HF_HOME"]) / "fake-hub").mkdir()
        (Path(os.environ["HF_HOME"]) / "fake-hub/index.bin").write_bytes(b"hub")
        (Path(os.environ["HF_DATASETS_CACHE"]) / "fake-dataset.bin").write_bytes(b"dataset")

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        with self.visits.open("a") as stream:
            stream.write(f"{{index}}\\n")
        if BEHAVIOR == "network" and index == 0:
            import socket

            socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()
        if BEHAVIOR == "caught_network" and index == 0:
            import socket

            try:
                socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()
            except PermissionError:
                pass
        if BEHAVIOR == "resolver" and index == 0:
            import socket

            socket.getaddrinfo("localhost", 80)
        if BEHAVIOR == "subprocess" and index == 0:
            import subprocess

            subprocess.run(["/bin/true"], check=True)
        if BEHAVIOR == "missing" and index == 2:
            raise IndexError("missing synthetic row")
        returned_index = index
        if BEHAVIOR == "duplicate" and index == 2:
            returned_index = 1
        elif BEHAVIOR == "gap" and index == 2:
            returned_index = 3
        state_shape = (31,) if BEHAVIOR == "wrong_shape" and index == 1 else (32,)
        dtype = torch.float64 if BEHAVIOR == "wrong_dtype" and index == 1 else torch.float32
        states = torch.zeros(state_shape, dtype=dtype)
        if BEHAVIOR == "nonfinite" and index == 1:
            states[0] = float("nan")
        return {{
            "observation.images.egocentric": torch.zeros(
                (3, self.height, self.width), dtype=torch.float32
            ),
            "states": states,
            "action": torch.zeros((36,), dtype=torch.float32),
            "index": torch.tensor(returned_index, dtype=torch.int64),
        }}
"""
    (package / "compat.py").write_text(module)
    if behavior == "race_sourceless_pyc":
        evil_marker = root / "race-sourceless-pyc-executed"
        evil_source = root / "race-evil-source.py"
        evil_bytecode = root / "race-evil.pyc"
        evil_source.write_text(
            "from pathlib import Path\n"
            f"Path({str(evil_marker)!r}).write_text('executed')\n"
        )
        py_compile.compile(
            str(evil_source),
            cfile=str(evil_bytecode),
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
        )
        payload = evil_bytecode.read_bytes()
        evil_source.unlink()
        evil_bytecode.unlink()
        (checkout / "src/psi/__init__.py").write_text(
            "from pathlib import Path\n"
            f"Path({str(checkout / 'src/evil.pyc')!r}).write_bytes({payload!r})\n"
        )
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "src"], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Task Eight",
            "-c",
            "user.email=task8@example.invalid",
            "commit",
            "-qm",
            "fake pinned PSI0",
        ],
        cwd=checkout,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return checkout, commit, import_marker


def _install_unchecked_hash_compat_pyc(checkout: Path, marker: Path) -> None:
    compat = checkout / "src/psi/data/lerobot/compat.py"
    source = compat.read_text()
    needle = 'IMPORT_MARKER.write_text("imported")\n'
    malicious = source.replace(
        needle,
        needle + f'Path({str(marker)!r}).write_text("unchecked pyc executed")\n',
        1,
    )
    assert malicious != source
    malicious_source = checkout.parent / "malicious-compat.py"
    malicious_source.write_text(malicious)
    pycache = compat.parent / "__pycache__"
    pycache.mkdir()
    cache_path = Path(importlib.util.cache_from_source(str(compat)))
    py_compile.compile(
        str(malicious_source),
        cfile=str(cache_path),
        dfile=str(compat),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    assert (
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )


def _replace_with_malicious_timestamp_compat_pyc(checkout: Path, marker: Path) -> Path:
    compat = checkout / "src/psi/data/lerobot/compat.py"
    source = compat.read_text()
    needle = 'IMPORT_MARKER.write_text("imported")\n'
    malicious = source.replace(
        needle,
        needle + f'Path({str(marker)!r}).write_text("timestamp pyc executed")\n',
        1,
    )
    assert malicious != source
    malicious_source = checkout.parent / "malicious-timestamp-compat.py"
    malicious_source.write_text(malicious)
    cache_path = Path(importlib.util.cache_from_source(str(compat)))
    py_compile.compile(
        str(malicious_source),
        cfile=str(cache_path),
        dfile=str(compat),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    payload = bytearray(cache_path.read_bytes())
    source_metadata = compat.stat()
    payload[8:12] = struct.pack("<I", int(source_metadata.st_mtime) & 0xFFFFFFFF)
    payload[12:16] = struct.pack("<I", source_metadata.st_size & 0xFFFFFFFF)
    cache_path.write_bytes(payload)
    assert int.from_bytes(payload[4:8], "little") == 0
    return cache_path


def _certify_with_fake_loader(
    publication: converter.PublicationResult,
    checkout: Path,
    commit: str,
    certificate_id: uuid.UUID,
) -> Path:
    with converter.conversion_lock(publication.dataset_root):
        return certify_published_dataset(
            publication,
            psi0_root=checkout,
            psi0_commit=commit,
            python=Path(sys.executable),
            certificate_uuid=certificate_id,
        )


def _identity_for_recorded_blob() -> converter.ConverterIdentity:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", "scripts/postprocess_psi0.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = subprocess.run(
        ["git", "show", f"{commit}:scripts/postprocess_psi0.py"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return converter.ConverterIdentity(commit, converter.sha256_bytes(payload))


def _make_plan(root: Path) -> converter.ConversionPlan:
    make_source_episode(
        root / "source-a",
        episode_index=3,
        frames=5,
        fps=4,
        task_index=8,
        task_text="first instruction",
    )
    make_source_episode(
        root / "source-b",
        episode_index=7,
        frames=6,
        fps=4,
        task_index=10,
        task_text="second instruction",
    )
    return converter.preflight_conversion(
        SimpleNamespace(
            skip=1,
            downsample=2,
            chunks_size=1,
            total_episodes=2,
            fps="4",
            video_key="observation.rgb_head_stereo_left",
            out_dir=str(root / "dataset"),
            sim_root=str(root / "source-*"),
        ),
        _identity_for_recorded_blob(),
    )


def _make_copy_plan(root: Path) -> converter.ConversionPlan:
    make_source_episode(
        root / "copy-source-a",
        episode_index=1,
        frames=3,
        fps=4,
        task_index=2,
        task_text="copy first",
    )
    make_source_episode(
        root / "copy-source-b",
        episode_index=2,
        frames=4,
        fps=4,
        task_index=3,
        task_text="copy second",
    )
    return converter.preflight_conversion(
        SimpleNamespace(
            skip=0,
            downsample=1,
            chunks_size=1,
            total_episodes=2,
            fps="4",
            video_key="observation.rgb_head_stereo_left",
            out_dir=str(root / "copy-dataset"),
            sim_root=str(root / "copy-source-*"),
        ),
        _identity_for_recorded_blob(),
    )


def _make_unit_plan(root: Path) -> converter.ConversionPlan:
    make_source_episode(
        root / "unit-source",
        episode_index=0,
        frames=1,
        fps=1,
        task_index=0,
        task_text="unit",
    )
    return converter.preflight_conversion(
        SimpleNamespace(
            skip=0,
            downsample=1,
            chunks_size=1,
            total_episodes=1,
            fps="1",
            video_key="observation.rgb_head_stereo_left",
            out_dir=str(root / "unit-dataset"),
            sim_root=str(root / "unit-source"),
        ),
        _identity_for_recorded_blob(),
    )


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("certify-production")
    plan = _make_plan(root)
    staging = root / ".dataset.staging-valid"
    converter.generate_staged_dataset(plan, staging)
    expected = DatasetExpectations.from_plan(plan)
    assert (
        validate_dataset(
            staging, expected=expected, require_final_modes=False
        ).total_episodes
        == 2
    )
    publication = converter.publish_staged_dataset(
        staging,
        plan.output_path,
        plan.converter,
        converter.PublicationFilesystem(),
        lambda _point: None,
    )
    assert (
        validate_dataset(
            plan.output_path, expected=None, require_final_modes=True
        ).total_frames
        == 5
    )
    return root, plan, publication


@pytest.fixture(scope="module")
def copy_generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("certify-copy-production")
    plan = _make_copy_plan(root)
    staging = root / ".copy-dataset.staging-valid"
    converter.generate_staged_dataset(plan, staging)
    publication = converter.publish_staged_dataset(
        staging,
        plan.output_path,
        plan.converter,
        converter.PublicationFilesystem(),
        lambda _point: None,
    )
    return root, plan, publication


@pytest.fixture(scope="module")
def unit_generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("certify-unit-production")
    plan = _make_unit_plan(root)
    staging = root / ".unit-dataset.staging-valid"
    converter.generate_staged_dataset(plan, staging)
    return root, plan, staging


@pytest.fixture(scope="module")
def staged_template(tmp_path_factory, generated):
    _, plan, _ = generated
    staging = tmp_path_factory.mktemp("certify-staged") / ".dataset.staging-template"
    converter.generate_staged_dataset(plan, staging)
    return plan, staging


@pytest.fixture(scope="module")
def certified(generated):
    _, _, publication = generated
    with converter.conversion_lock(publication.dataset_root):
        evidence = certify_published_dataset(
            publication,
            psi0_root=Path(__file__).resolve().parents[1],
            psi0_commit="0" * 40,
            python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
            certificate_uuid=uuid.UUID("00000000-0000-0000-0000-000000000456"),
        )
    validate_evidence_terminal(evidence)
    return evidence


def _writable_copy(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    for path in [destination, *destination.rglob("*")]:
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            path.chmod(0o644)
    return destination


def _reseal_published(root: Path) -> converter.PublicationResult:
    status = json.loads((root / "CONVERSION_STATUS.json").read_text())
    converter_identity = converter.ConverterIdentity(**status["converter"])
    root_metadata = root.stat(follow_symlinks=False)
    status["staging"] = {
        "st_dev": root_metadata.st_dev,
        "st_ino": root_metadata.st_ino,
    }
    (root / "CONVERSION_STATUS.json").write_bytes(
        converter.canonical_json_bytes(status)
    )
    manifest = converter.build_payload_manifest(root)
    manifest_payload = converter.canonical_json_bytes(manifest)
    (root / "meta/conversion_manifest.json").write_bytes(manifest_payload)
    status_payload = converter._complete_status_bytes(
        root, manifest, manifest_payload, converter_identity
    )
    (root / "CONVERSION_STATUS.json").write_bytes(status_payload)
    for path in [root, *root.rglob("*")]:
        path.chmod(0o555 if path.is_dir() else 0o444)
    return converter.PublicationResult(
        dataset_root=root,
        manifest_sha256=converter.sha256_bytes(manifest_payload),
        complete_status_sha256=converter.sha256_bytes(status_payload),
        state="published",
    )


def _published_copy(publication, destination: Path) -> converter.PublicationResult:
    root = _writable_copy(publication.dataset_root, destination)
    _rewrite_json(
        root / "meta/conversion_provenance.json",
        lambda value: value["invocation"].__setitem__(
            "output_path", str(root.absolute())
        ),
    )
    return _reseal_published(root)


def _unseal(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)


def _seal_evidence_copy(root: Path) -> None:
    for path in root.iterdir():
        if path.is_file() and not path.is_symlink():
            path.chmod(0o444)
    root.chmod(0o555)


def _rewrite_episode_provenance_digests(root: Path) -> None:
    rows = [
        json.loads(line)
        for line in (root / "meta/episodes.jsonl").read_text().splitlines()
    ]
    digests = [
        converter.sha256_bytes(
            converter.canonical_json_bytes(row["conversion_provenance"])
        )
        for row in rows
    ]
    _rewrite_json(
        root / "meta/conversion_provenance.json",
        lambda value: value.__setitem__("episode_provenance_sha256", digests),
    )


def _coherently_mutated_publication(
    publication: converter.PublicationResult,
    destination: Path,
    mutation,
    *,
    rehash_episode_provenance: bool = True,
) -> converter.PublicationResult:
    root = _writable_copy(publication.dataset_root, destination)
    _rewrite_json(
        root / "meta/conversion_provenance.json",
        lambda value: value["invocation"].__setitem__(
            "output_path", str(root.absolute())
        ),
    )
    mutation(root)
    if rehash_episode_provenance:
        _rewrite_episode_provenance_digests(root)
    return _reseal_published(root)


def _rewrite_table(path: Path, transform) -> None:
    table = transform(pq.read_table(path))
    pq.write_table(table, path)


def _first_parquet(root: Path) -> Path:
    return sorted(root.glob("data/chunk-*/episode_*.parquet"))[0]


def _rewrite_json(path: Path, transform) -> None:
    value = json.loads(path.read_text())
    transform(value)
    path.write_bytes(converter.canonical_json_bytes(value))


def _rewrite_jsonl(path: Path, transform) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    transform(rows)
    path.write_bytes(b"".join(converter.canonical_json_bytes(row) for row in rows))


def _assert_code(root: Path, expected, code: str, *, final=False) -> None:
    with pytest.raises(DatasetValidationError) as caught:
        validate_dataset(root, expected=expected, require_final_modes=final)
    assert caught.value.code == code, str(caught.value)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("name", "ARROW_SCHEMA"),
        ("order", "ARROW_SCHEMA"),
        ("width", "ARROW_SCHEMA"),
        ("value_type", "ARROW_SCHEMA"),
        ("scalar_type", "ARROW_SCHEMA"),
        ("unequal_count", "ARROW_ROWS"),
    ],
)
def test_semantic_arrow_mutations_fail_stably(tmp_path, generated, mutation, code):
    _, plan, _ = generated
    root = tmp_path / mutation
    converter.generate_staged_dataset(plan, root)
    path = _first_parquet(root)

    def mutate(table):
        if mutation == "name":
            return table.rename_columns(["wrong", *table.column_names[1:]])
        if mutation == "order":
            return table.select(
                [table.column_names[1], table.column_names[0], *table.column_names[2:]]
            )
        if mutation == "width":
            values = np.asarray(table["states"].to_pylist(), dtype=np.float32)[:, :-1]
            array = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), 31)
            return table.set_column(0, "states", array)
        if mutation == "value_type":
            values = np.asarray(table["states"].to_pylist(), dtype=np.float64)
            array = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), 32)
            return table.set_column(0, "states", array)
        if mutation == "scalar_type":
            return table.set_column(
                table.schema.get_field_index("timestamp"),
                "timestamp",
                pa.array(table["timestamp"].to_pylist(), type=pa.float64()),
            )
        arrays = [
            table[name].slice(0, len(table) - (name == "action"))
            for name in table.column_names
        ]
        return pa.Table.from_arrays(arrays, names=table.column_names)

    if mutation == "unequal_count":
        # Arrow tables cannot have unequal columns; a zero-row episode is the on-disk
        # cardinality corruption with an otherwise valid schema.
        pq.write_table(pq.read_table(path).slice(0, 0), path)
    else:
        _rewrite_table(path, mutate)
    _assert_code(root, DatasetExpectations.from_plan(plan), code)


def test_semantic_production_staged_validation_seam_uses_preflight(tmp_path, generated):
    _, plan, _ = generated
    root = tmp_path / "production-validation-seam"
    converter.generate_staged_dataset(plan, root)
    assert converter.validate_staged_dataset(root, plan).total_frames == 5


@pytest.mark.parametrize("column", [*VECTOR_FIELDS, "timestamp"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_semantic_each_float_column_rejects_nonfinite(
    tmp_path, generated, column, value
):
    _, plan, _ = generated
    root = tmp_path / f"nonfinite-{column.replace('.', '-')}-{str(value)}"
    converter.generate_staged_dataset(plan, root)
    path = _first_parquet(root)

    def mutate(table):
        index = table.schema.get_field_index(column)
        if column in VECTOR_FIELDS:
            values = np.asarray(table[column].to_pylist(), dtype=np.float32)
            values[0, 0] = value
            array = pa.FixedSizeListArray.from_arrays(
                pa.array(values.ravel(), type=pa.float32()), VECTOR_FIELDS[column]
            )
        else:
            values = np.asarray(table[column].to_pylist(), dtype=np.float32)
            values[0] = value
            array = pa.array(values, type=pa.float32())
        return table.set_column(index, column, array)

    _rewrite_table(path, mutate)
    _assert_code(root, DatasetExpectations.from_plan(plan), "NUMERIC_NONFINITE")


@pytest.mark.parametrize(
    ("column", "values", "code"),
    [
        ("frame_index", [0, 2], "INDEX_LOCAL"),
        ("frame_index", [0, 0], "INDEX_LOCAL"),
        ("index", [0, 0], "INDEX_GLOBAL"),
        ("index", [0, 2], "INDEX_GLOBAL"),
        ("episode_index", [9, 9], "INDEX_EPISODE"),
        ("task_index", [9, 9], "INDEX_TASK"),
        ("timestamp", [0.0, 9.0], "INDEX_TIMESTAMP"),
        ("next.done", [True, False], "INDEX_DONE"),
    ],
)
def test_semantic_index_mutations_fail_stably(
    tmp_path, generated, column, values, code
):
    _, plan, _ = generated
    root = tmp_path / f"index-{column}"
    converter.generate_staged_dataset(plan, root)
    path = _first_parquet(root)

    def mutate(table):
        index = table.schema.get_field_index(column)
        arrow_type = table.schema.field(column).type
        return table.set_column(index, column, pa.array(values, type=arrow_type))

    _rewrite_table(path, mutate)
    _assert_code(root, DatasetExpectations.from_plan(plan), code)


@pytest.mark.parametrize(
    "column",
    [
        "states",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "next.done",
    ],
)
@pytest.mark.parametrize(
    "statistic", ["mean", "std", "min", "max", "q01", "q99", "count"]
)
def test_semantic_every_global_statistic_field_is_recomputed(
    tmp_path, staged_template, column, statistic
):
    plan, template = staged_template
    root = shutil.copytree(template, tmp_path / f"global-{column}-{statistic}")

    def mutate(value):
        value[column][statistic][0] = 999 if statistic == "count" else 999.0

    _rewrite_json(root / "meta/stats.json", mutate)
    (root / "meta/stats_psi0.json").write_bytes((root / "meta/stats.json").read_bytes())
    expected_code = "STATS_COUNT" if statistic == "count" else "STATS_VALUE"
    _assert_code(root, DatasetExpectations.from_plan(plan), expected_code)


@pytest.mark.parametrize("block", ["action", "timestamp"])
@pytest.mark.parametrize(
    "statistic", ["mean", "std", "min", "max", "q01", "q99", "count"]
)
def test_semantic_every_episode_statistic_field_is_recomputed(
    tmp_path, staged_template, block, statistic
):
    plan, template = staged_template
    root = shutil.copytree(template, tmp_path / f"episode-{block}-{statistic}")

    def mutate(rows):
        rows[0]["stats"][block][statistic][0] = 999 if statistic == "count" else 999.0

    _rewrite_jsonl(root / "meta/episodes_stats.jsonl", mutate)
    expected_code = "STATS_COUNT" if statistic == "count" else "STATS_VALUE"
    _assert_code(root, DatasetExpectations.from_plan(plan), expected_code)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("mean", "STATS_VALUE"),
        ("std", "STATS_VALUE"),
        ("min", "STATS_VALUE"),
        ("max", "STATS_VALUE"),
        ("q01", "STATS_VALUE"),
        ("q99", "STATS_VALUE"),
        ("count", "STATS_COUNT"),
        ("scope", "STATS_SCOPE"),
        ("global_scope", "STATS_SCOPE"),
        ("bytes", "STATS_BYTES"),
    ],
)
def test_semantic_statistics_mutations_fail_stably(tmp_path, generated, mutation, code):
    _, plan, _ = generated
    root = tmp_path / f"stats-{mutation}"
    converter.generate_staged_dataset(plan, root)
    if mutation == "bytes":
        (root / "meta/stats_psi0.json").write_bytes(b"{}\n")
    elif mutation == "scope":
        _rewrite_jsonl(
            root / "meta/episodes_stats.jsonl",
            lambda rows: rows[0].__setitem__("episode_index", 1),
        )
    elif mutation == "global_scope":

        def mutate_scope(value):
            value["wrong"] = value.pop("action")

        _rewrite_json(root / "meta/stats.json", mutate_scope)
        (root / "meta/stats_psi0.json").write_bytes(
            (root / "meta/stats.json").read_bytes()
        )
    else:
        _rewrite_json(
            root / "meta/stats.json",
            lambda value: value["action"].__setitem__(
                mutation,
                [999] if mutation == "count" else [999.0] * 36,
            ),
        )
        (root / "meta/stats_psi0.json").write_bytes(
            (root / "meta/stats.json").read_bytes()
        )
    _assert_code(root, DatasetExpectations.from_plan(plan), code)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("missing", "VIDEO_MEMBERSHIP"),
        ("extra", "VIDEO_MEMBERSHIP"),
        ("frame_count", "VIDEO_PROFILE"),
        ("codec", "VIDEO_PROFILE"),
        ("pixel_format", "VIDEO_PROFILE"),
        ("dimensions", "VIDEO_PROFILE"),
        ("fps", "VIDEO_PROFILE"),
        ("audio", "VIDEO_PROFILE"),
    ],
)
def test_semantic_video_mutations_fail_stably(tmp_path, generated, mutation, code):
    _, plan, _ = generated
    root = tmp_path / f"video-{mutation}"
    converter.generate_staged_dataset(plan, root)
    videos = sorted(root.glob("videos/chunk-*/egocentric/*.mp4"))
    if mutation == "missing":
        videos[0].unlink()
    elif mutation == "extra":
        shutil.copyfile(videos[0], videos[0].with_name("extra.mp4"))
    else:
        destination = videos[0].with_suffix(".new.mp4")
        args = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i"]
        if mutation == "audio":
            args += [
                "testsrc2=size=640x360:rate=4:duration=0.5",
                "-f",
                "lavfi",
                "-i",
                "sine=duration=0.5",
                "-shortest",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(destination),
            ]
        else:
            size = "320x240" if mutation == "dimensions" else "640x360"
            rate = "5" if mutation == "fps" else "4"
            duration = "0.75" if mutation == "frame_count" else "0.5"
            codec_name = "mpeg4" if mutation == "codec" else "libx264"
            pixel_format = "yuv444p" if mutation == "pixel_format" else "yuv420p"
            args += [
                f"testsrc2=size={size}:rate={rate}:duration={duration}",
                "-an",
                "-c:v",
                codec_name,
                "-pix_fmt",
                pixel_format,
                str(destination),
            ]
        subprocess.run(args, check=True)
        os.replace(destination, videos[0])
    _assert_code(root, DatasetExpectations.from_plan(plan), code)


@pytest.mark.parametrize(
    ("path", "mutation", "code"),
    [
        (
            "meta/info.json",
            lambda v: v.__setitem__("total_frames", 99),
            "METADATA_TOTALS",
        ),
        (
            "meta/info.json",
            lambda v: v.__setitem__("chunks_size", 9),
            "METADATA_CHUNKS",
        ),
        (
            "meta/info.json",
            lambda v: v.__setitem__("data_path", "wrong"),
            "METADATA_PATHS",
        ),
        (
            "meta/info.json",
            lambda v: v["features"]["states"].__setitem__("shape", [31]),
            "METADATA_SHAPE",
        ),
        (
            "meta/info.json",
            lambda v: v["features"]["observation.images.egocentric"][
                "video_info"
            ].__setitem__("video.width", 1),
            "METADATA_VIDEO",
        ),
    ],
)
def test_semantic_metadata_mutations_fail_stably(
    tmp_path, generated, path, mutation, code
):
    _, plan, _ = generated
    root = tmp_path / f"metadata-{code}"
    converter.generate_staged_dataset(plan, root)
    _rewrite_json(root / path, mutation)
    _assert_code(root, DatasetExpectations.from_plan(plan), code)


@pytest.mark.parametrize("metadata", ["tasks", "episodes"])
def test_semantic_metadata_cardinality_is_exact(tmp_path, generated, metadata):
    _, plan, _ = generated
    root = tmp_path / f"metadata-cardinality-{metadata}"
    converter.generate_staged_dataset(plan, root)
    path = root / f"meta/{metadata}.jsonl"
    _rewrite_jsonl(path, lambda rows: rows.pop())
    _assert_code(root, DatasetExpectations.from_plan(plan), "METADATA_TOTALS")


@pytest.mark.parametrize("feature", [*VECTOR_FIELDS, *SCALAR_FIELDS])
@pytest.mark.parametrize("field", ["dtype", "shape"])
def test_semantic_every_feature_dtype_and_shape_is_exact(
    tmp_path, generated, feature, field
):
    _, plan, _ = generated
    root = tmp_path / f"feature-{feature.replace('.', '-')}-{field}"
    converter.generate_staged_dataset(plan, root)
    _rewrite_json(
        root / "meta/info.json",
        lambda value: value["features"][feature].__setitem__(
            field, "float64" if field == "dtype" else [999]
        ),
    )
    _assert_code(root, DatasetExpectations.from_plan(plan), "METADATA_SHAPE")


@pytest.mark.parametrize(
    "feature",
    [
        "observation.hand_joints",
        "observation.arm_joints",
        "observation.leg_joints",
        "observation.prev_torso_rpy",
        "observation.prev_height",
    ],
)
def test_semantic_named_feature_names_are_exact(tmp_path, generated, feature):
    _, plan, _ = generated
    root = tmp_path / f"feature-names-{feature.replace('.', '-')}"
    converter.generate_staged_dataset(plan, root)
    _rewrite_json(
        root / "meta/info.json",
        lambda value: value["features"][feature].__setitem__("names", ["wrong"]),
    )
    _assert_code(root, DatasetExpectations.from_plan(plan), "METADATA_SHAPE")


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("codebase_version", "METADATA_SHAPE"),
        ("robot_type", "METADATA_SHAPE"),
        ("extra_info", "METADATA_SHAPE"),
        ("image_dtype", "METADATA_VIDEO"),
        ("image_names", "METADATA_VIDEO"),
        ("image_extra", "METADATA_VIDEO"),
    ],
)
def test_semantic_info_and_image_metadata_schemas_are_exact(
    tmp_path, generated, mutation, code
):
    _, plan, _ = generated
    root = tmp_path / f"metadata-schema-{mutation}"
    converter.generate_staged_dataset(plan, root)

    def mutate(value):
        if mutation == "codebase_version":
            value[mutation] = "wrong"
        elif mutation == "robot_type":
            value[mutation] = "wrong"
        elif mutation == "extra_info":
            value["extra"] = True
        else:
            image = value["features"]["observation.images.egocentric"]
            if mutation == "image_dtype":
                image["dtype"] = "image"
            elif mutation == "image_names":
                image["names"] = ["wrong"]
            else:
                image["extra"] = True

    _rewrite_json(root / "meta/info.json", mutate)
    _assert_code(root, DatasetExpectations.from_plan(plan), code)


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("total_episodes", "METADATA_TOTALS"),
        ("total_frames", "METADATA_TOTALS"),
        ("total_tasks", "METADATA_TOTALS"),
        ("total_videos", "METADATA_TOTALS"),
        ("total_chunks", "METADATA_CHUNKS"),
        ("chunks_size", "METADATA_CHUNKS"),
        ("fps", "METADATA_CHUNKS"),
        ("data_path", "METADATA_PATHS"),
        ("video_path", "METADATA_PATHS"),
    ],
)
def test_semantic_each_info_scalar_and_path_is_exact(tmp_path, generated, field, code):
    _, plan, _ = generated
    root = tmp_path / f"metadata-info-{field}"
    converter.generate_staged_dataset(plan, root)
    _rewrite_json(
        root / "meta/info.json",
        lambda value: value.__setitem__(
            field, "wrong" if field.endswith("path") else 999
        ),
    )
    _assert_code(root, DatasetExpectations.from_plan(plan), code)


@pytest.mark.parametrize(
    "field",
    [
        "has_audio",
        "video.channels",
        "video.codec",
        "video.fps",
        "video.height",
        "video.is_depth_map",
        "video.pix_fmt",
        "video.width",
    ],
)
def test_semantic_each_video_info_field_is_exact(tmp_path, generated, field):
    _, plan, _ = generated
    root = tmp_path / f"metadata-video-info-{field}"
    converter.generate_staged_dataset(plan, root)
    replacements = {
        "has_audio": True,
        "video.channels": 1,
        "video.codec": "av1",
        "video.fps": 5.0,
        "video.height": 1,
        "video.is_depth_map": True,
        "video.pix_fmt": "yuv444p",
        "video.width": 1,
    }
    _rewrite_json(
        root / "meta/info.json",
        lambda value: value["features"]["observation.images.egocentric"][
            "video_info"
        ].__setitem__(field, replacements[field]),
    )
    _assert_code(root, DatasetExpectations.from_plan(plan), "METADATA_VIDEO")


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("total_episodes", "METADATA_TOTALS"),
        ("total_frames", "METADATA_TOTALS"),
        ("total_tasks", "METADATA_TOTALS"),
        ("total_videos", "METADATA_TOTALS"),
        ("total_chunks", "METADATA_CHUNKS"),
        ("chunks_size", "METADATA_CHUNKS"),
        ("fps", "METADATA_VIDEO"),
    ],
)
def test_semantic_one_item_info_rejects_boolean_integer_values(
    tmp_path, unit_generated, field, code
):
    _, plan, template = unit_generated
    root = shutil.copytree(template, tmp_path / f"boolean-info-{field}")
    _rewrite_json(root / "meta/info.json", lambda value: value.__setitem__(field, True))
    _assert_code(root, DatasetExpectations.from_plan(plan), code)


@pytest.mark.parametrize(
    ("target", "field", "code"),
    [
        ("feature", "timestamp", "METADATA_SHAPE"),
        ("video", "has_audio", "METADATA_VIDEO"),
        ("episode_stats", "count", "STATS_COUNT"),
        ("global_stats", "count", "STATS_COUNT"),
        ("episode_provenance", "retained_count", "PROVENANCE_SOURCE"),
        ("source_media", "frame_count", "PROVENANCE_SOURCE"),
        ("output_media", "frame_count", "PROVENANCE_OUTPUT"),
    ],
)
def test_semantic_json_schema_rejects_boolean_integer_aliases(
    tmp_path, unit_generated, target, field, code
):
    _, plan, template = unit_generated
    root = shutil.copytree(template, tmp_path / f"boolean-schema-{target}-{field}")
    if target == "feature":
        _rewrite_json(
            root / "meta/info.json",
            lambda value: value["features"][field].__setitem__("shape", [True]),
        )
    elif target == "video":
        _rewrite_json(
            root / "meta/info.json",
            lambda value: value["features"]["observation.images.egocentric"][
                "video_info"
            ].__setitem__(field, 0),
        )
    elif target == "episode_stats":
        _rewrite_jsonl(
            root / "meta/episodes_stats.jsonl",
            lambda rows: rows[0]["stats"]["timestamp"].__setitem__(field, [True]),
        )
    elif target == "global_stats":
        _rewrite_json(
            root / "meta/stats.json",
            lambda value: value["timestamp"].__setitem__(field, [True]),
        )
        (root / "meta/stats_psi0.json").write_bytes(
            (root / "meta/stats.json").read_bytes()
        )
    else:

        def mutate(rows):
            provenance = rows[0]["conversion_provenance"]
            if target == "episode_provenance":
                provenance[field] = True
            else:
                provenance[target][field] = True

        _rewrite_jsonl(root / "meta/episodes.jsonl", mutate)
    _assert_code(root, DatasetExpectations.from_plan(plan), code)


@pytest.mark.parametrize(
    "field",
    [
        "task_index",
        "episode_index",
        "task_reference",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        "stats_episode_index",
    ],
)
def test_semantic_one_item_metadata_rejects_boolean_indices(
    tmp_path, unit_generated, field
):
    _, plan, template = unit_generated
    root = shutil.copytree(template, tmp_path / f"boolean-index-{field}")
    if field == "task_index":
        _rewrite_jsonl(
            root / "meta/tasks.jsonl",
            lambda rows: rows[0].__setitem__("task_index", False),
        )
        _rewrite_jsonl(
            root / "meta/episodes.jsonl",
            lambda rows: rows[0]["instruction"].__setitem__("task_index", False),
        )
    elif field == "stats_episode_index":
        _rewrite_jsonl(
            root / "meta/episodes_stats.jsonl",
            lambda rows: rows[0].__setitem__("episode_index", False),
        )
    else:
        key = "tasks" if field == "task_reference" else field
        replacement = (
            [False]
            if field == "task_reference"
            else (True if field == "length" else False)
        )
        _rewrite_jsonl(
            root / "meta/episodes.jsonl",
            lambda rows: rows[0].__setitem__(key, replacement),
        )
    with pytest.raises(DatasetValidationError) as caught:
        validate_dataset(
            root,
            expected=DatasetExpectations.from_plan(plan),
            require_final_modes=False,
        )
    assert caught.value.code in {
        "METADATA_TOTALS",
        "METADATA_CARDINALITY",
        "STATS_SCOPE",
    }


def test_semantic_instruction_rejects_boolean_task_index_alias(
    tmp_path, staged_template
):
    plan, template = staged_template
    root = shutil.copytree(template, tmp_path / "boolean-instruction-task-index")
    _rewrite_jsonl(
        root / "meta/episodes.jsonl",
        lambda rows: rows[1]["instruction"].__setitem__("task_index", True),
    )
    _assert_code(root, DatasetExpectations.from_plan(plan), "METADATA_CARDINALITY")


def test_semantic_modality_rejects_boolean_integer_alias(tmp_path, unit_generated):
    _, plan, template = unit_generated
    root = shutil.copytree(template, tmp_path / "boolean-modality-start")
    _rewrite_json(
        root / "meta/modality.json",
        lambda value: value["state"]["left_hand"].__setitem__("start", False),
    )
    _assert_code(root, DatasetExpectations.from_plan(plan), "METADATA_SHAPE")


def test_semantic_published_task_index_false_is_not_zero(tmp_path, generated):
    _, _, publication = generated

    def mutate(root):
        _rewrite_jsonl(
            root / "meta/tasks.jsonl",
            lambda rows: rows[0].__setitem__("task_index", False),
        )
        _rewrite_jsonl(
            root / "meta/episodes.jsonl",
            lambda rows: rows[0]["instruction"].__setitem__("task_index", False),
        )

    mutated = _coherently_mutated_publication(
        publication, tmp_path / "published-task-index-false", mutate
    )
    _assert_code(mutated.dataset_root, None, "METADATA_CARDINALITY", final=True)


@pytest.mark.parametrize(
    "mutation", ["invalid_environment", "unused_task", "duplicate_task_text"]
)
def test_semantic_published_metadata_requires_canonical_task_mapping(
    tmp_path, generated, mutation
):
    _, _, publication = generated

    def mutate(root):
        if mutation == "invalid_environment":
            _rewrite_jsonl(
                root / "meta/episodes.jsonl",
                lambda rows: rows[0].__setitem__("environment_config", "not json"),
            )
        elif mutation == "unused_task":
            _rewrite_jsonl(
                root / "meta/tasks.jsonl",
                lambda rows: rows.append(
                    {
                        "task_index": 2,
                        "task": "unused instruction",
                        "category": "",
                        "description": "unused instruction",
                    }
                ),
            )
            _rewrite_json(
                root / "meta/info.json",
                lambda value: value.__setitem__("total_tasks", 3),
            )
        else:

            def duplicate(rows):
                rows[1]["task"] = rows[0]["task"]
                rows[1]["description"] = rows[0]["task"]

            _rewrite_jsonl(root / "meta/tasks.jsonl", duplicate)
            _rewrite_jsonl(
                root / "meta/episodes.jsonl",
                lambda rows: rows[1].__setitem__(
                    "instruction",
                    {
                        "task_index": 1,
                        "task": rows[0]["instruction"]["task"],
                        "category": "",
                        "description": rows[0]["instruction"]["task"],
                    },
                ),
            )

    mutated = _coherently_mutated_publication(
        publication, tmp_path / f"published-{mutation}", mutate
    )
    with converter.conversion_lock(mutated.dataset_root):
        with pytest.raises(PublishedUncertifiedError) as caught:
            certify_published_dataset(
                mutated,
                psi0_root=Path(__file__).resolve().parents[1],
                psi0_commit="0" * 40,
                python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
                certificate_uuid=uuid.uuid4(),
            )
    assert caught.value.code == "METADATA_CARDINALITY"
    assert (caught.value.evidence_root / "FAIL.json").is_file()
    assert not (caught.value.evidence_root / "PASS.json").exists()
    validate_evidence_terminal(caught.value.evidence_root)


@pytest.mark.parametrize(
    "field", ["skip", "downsample", "chunks_size", "total_episodes"]
)
def test_semantic_published_invocation_rejects_boolean_integers(tmp_path, field):
    plan = _make_unit_plan(tmp_path / field)
    staging = tmp_path / field / ".unit-dataset.staging"
    converter.generate_staged_dataset(plan, staging)
    publication = converter.publish_staged_dataset(
        staging,
        plan.output_path,
        plan.converter,
        converter.PublicationFilesystem(),
        lambda _point: None,
    )
    _unseal(publication.dataset_root)
    _rewrite_json(
        publication.dataset_root / "meta/conversion_provenance.json",
        lambda value: value["invocation"].__setitem__(
            field, False if field == "skip" else True
        ),
    )
    publication = _reseal_published(publication.dataset_root)
    _assert_code(publication.dataset_root, None, "PROVENANCE_SOURCE", final=True)


@pytest.mark.parametrize(
    "field",
    ["task_index", "task", "category", "description", "extra"],
)
def test_semantic_each_task_metadata_field_is_exact(tmp_path, generated, field):
    _, plan, _ = generated
    root = tmp_path / f"metadata-task-{field}"
    converter.generate_staged_dataset(plan, root)

    def mutate(rows):
        replacements = {
            "task_index": 99,
            "task": "",
            "category": "wrong",
            "description": "wrong",
            "extra": True,
        }
        rows[0][field] = replacements[field]

    _rewrite_jsonl(root / "meta/tasks.jsonl", mutate)
    _assert_code(root, DatasetExpectations.from_plan(plan), "METADATA_CARDINALITY")


@pytest.mark.parametrize(
    "field",
    [
        "episode_index",
        "tasks",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        "robot_type",
        "instruction",
        "environment_config",
        "conversion_provenance",
        "extra",
    ],
)
def test_semantic_each_episode_metadata_field_is_exact(tmp_path, generated, field):
    _, plan, _ = generated
    root = tmp_path / f"metadata-episode-{field}"
    converter.generate_staged_dataset(plan, root)

    def mutate(rows):
        replacements = {
            "episode_index": 99,
            "tasks": [],
            "length": 0,
            "dataset_from_index": 99,
            "dataset_to_index": 99,
            "robot_type": "wrong",
            "instruction": {},
            "environment_config": "wrong",
            "conversion_provenance": None,
            "extra": True,
        }
        rows[0][field] = replacements[field]

    _rewrite_jsonl(root / "meta/episodes.jsonl", mutate)
    code = "METADATA_TOTALS" if field == "length" else "METADATA_CARDINALITY"
    _assert_code(root, DatasetExpectations.from_plan(plan), code)


@pytest.mark.parametrize(
    ("field", "replacement", "code"),
    [
        ("source_parquet_sha256", "0" * 64, "PROVENANCE_SOURCE"),
        ("source_video_sha256", "0" * 64, "PROVENANCE_SOURCE"),
        ("output_video_sha256", "0" * 64, "PROVENANCE_OUTPUT"),
        ("converter_commit", "0" * 40, "PROVENANCE_CONVERTER"),
        ("converter_script_sha256", "0" * 64, "PROVENANCE_CONVERTER"),
    ],
)
def test_semantic_provenance_mutations_fail_stably(
    tmp_path, generated, field, replacement, code
):
    _, plan, _ = generated
    root = tmp_path / f"provenance-{field}"
    converter.generate_staged_dataset(plan, root)
    _rewrite_jsonl(
        root / "meta/episodes.jsonl",
        lambda rows: rows[0]["conversion_provenance"].__setitem__(field, replacement),
    )
    _assert_code(root, DatasetExpectations.from_plan(plan), code)


def test_semantic_episode_provenance_digest_order_is_bound(tmp_path, generated):
    _, plan, _ = generated
    root = tmp_path / "digest-order"
    converter.generate_staged_dataset(plan, root)
    _rewrite_json(
        root / "meta/conversion_provenance.json",
        lambda value: value["episode_provenance_sha256"].reverse(),
    )
    _assert_code(root, DatasetExpectations.from_plan(plan), "PROVENANCE_ORDER")


@pytest.mark.parametrize(
    ("category", "field", "code"),
    [
        ("converter", "commit", "PROVENANCE_CONVERTER"),
        ("converter", "script_sha256", "PROVENANCE_CONVERTER"),
        ("invocation", "output_path", "PROVENANCE_SOURCE"),
        ("invocation", "skip", "PROVENANCE_SOURCE"),
        ("invocation", "downsample", "PROVENANCE_SOURCE"),
        ("invocation", "output_fps", "PROVENANCE_SOURCE"),
        ("invocation", "video_key", "PROVENANCE_SOURCE"),
        ("invocation", "chunks_size", "PROVENANCE_SOURCE"),
        ("invocation", "total_episodes", "PROVENANCE_SOURCE"),
        ("input_root", "path", "PROVENANCE_SOURCE"),
        ("input_root", "st_dev", "PROVENANCE_SOURCE"),
        ("input_root", "st_ino", "PROVENANCE_SOURCE"),
        ("input_root", "st_uid", "PROVENANCE_SOURCE"),
        ("input_root", "st_gid", "PROVENANCE_SOURCE"),
        ("input_root", "st_mode", "PROVENANCE_SOURCE"),
        ("dataset", "media_mode", "PROVENANCE_OUTPUT"),
        ("output_media", "codec_name", "PROVENANCE_OUTPUT"),
        ("output_media", "pixel_format", "PROVENANCE_OUTPUT"),
        ("output_media", "width", "PROVENANCE_OUTPUT"),
        ("output_media", "height", "PROVENANCE_OUTPUT"),
        ("output_media", "average_frame_rate", "PROVENANCE_OUTPUT"),
        ("output_media", "nominal_frame_rate", "PROVENANCE_OUTPUT"),
        ("output_media", "audio_streams", "PROVENANCE_OUTPUT"),
        ("dataset", "output_schema_sha256", "PROVENANCE_OUTPUT"),
    ],
)
def test_semantic_published_dataset_provenance_fields_are_independently_bound(
    tmp_path, generated, category, field, code
):
    _, _, publication = generated

    def mutate(root):
        def change(value):
            if category == "converter":
                value["converter"][field] = "0" * (40 if field == "commit" else 64)
            elif category == "invocation":
                replacements = {
                    "output_path": "/wrong",
                    "skip": 0,
                    "downsample": 1,
                    "output_fps": "5",
                    "video_key": "wrong",
                    "chunks_size": 999,
                    "total_episodes": 999,
                }
                value["invocation"][field] = replacements[field]
            elif category == "input_root":
                value["input_roots"][0][field] = (
                    "/wrong" if field == "path" else 999999999
                )
            elif category == "output_media":
                replacements = {
                    "codec_name": "av1",
                    "pixel_format": "yuv444p",
                    "width": 1,
                    "height": 1,
                    "average_frame_rate": "5",
                    "nominal_frame_rate": "5",
                    "audio_streams": [{"codec_name": "aac"}],
                }
                value["output_media"][field] = replacements[field]
            else:
                value[field] = "copy_all" if field == "media_mode" else "0" * 64

        _rewrite_json(root / "meta/conversion_provenance.json", change)

    mutated = _coherently_mutated_publication(
        publication, tmp_path / f"published-{category}-{field}", mutate
    )
    _assert_code(mutated.dataset_root, None, code, final=True)


@pytest.mark.parametrize(
    ("category", "field", "code"),
    [
        ("episode", "source_episode_index", "PROVENANCE_SOURCE"),
        ("episode", "source_parquet_sha256", "PROVENANCE_SOURCE"),
        ("episode", "source_video_sha256", "PROVENANCE_SOURCE"),
        ("episode", "requested_video_key", "PROVENANCE_SOURCE"),
        ("episode", "requested_output_fps", "PROVENANCE_SOURCE"),
        ("episode", "skip", "PROVENANCE_SOURCE"),
        ("episode", "downsample", "PROVENANCE_SOURCE"),
        ("episode", "retained_count", "PROVENANCE_SOURCE"),
        ("source_media", "codec_name", "PROVENANCE_SOURCE"),
        ("source_media", "pixel_format", "PROVENANCE_SOURCE"),
        ("source_media", "width", "PROVENANCE_SOURCE"),
        ("source_media", "height", "PROVENANCE_SOURCE"),
        ("source_media", "average_frame_rate", "PROVENANCE_SOURCE"),
        ("source_media", "nominal_frame_rate", "PROVENANCE_SOURCE"),
        ("source_media", "frame_count", "PROVENANCE_SOURCE"),
        ("source_media", "duration", "PROVENANCE_SOURCE"),
        ("source_media", "audio_streams", "PROVENANCE_SOURCE"),
        ("episode", "output_video_sha256", "PROVENANCE_OUTPUT"),
        ("output_media", "codec_name", "PROVENANCE_OUTPUT"),
        ("output_media", "pixel_format", "PROVENANCE_OUTPUT"),
        ("output_media", "width", "PROVENANCE_OUTPUT"),
        ("output_media", "height", "PROVENANCE_OUTPUT"),
        ("output_media", "average_frame_rate", "PROVENANCE_OUTPUT"),
        ("output_media", "nominal_frame_rate", "PROVENANCE_OUTPUT"),
        ("output_media", "frame_count", "PROVENANCE_OUTPUT"),
        ("output_media", "duration", "PROVENANCE_OUTPUT"),
        ("output_media", "audio_streams", "PROVENANCE_OUTPUT"),
        ("episode", "converter_commit", "PROVENANCE_CONVERTER"),
        ("episode", "converter_script_sha256", "PROVENANCE_CONVERTER"),
    ],
)
def test_semantic_published_episode_provenance_fields_are_independently_bound(
    tmp_path, generated, category, field, code
):
    _, _, publication = generated

    def mutate(root):
        def change(rows):
            episode = rows[0]["conversion_provenance"]
            target = episode if category == "episode" else episode[category]
            replacements = {
                "source_episode_index": 999,
                "source_parquet_sha256": "0" * 64,
                "source_video_sha256": "0" * 64,
                "requested_video_key": "wrong",
                "requested_output_fps": "5",
                "skip": 0,
                "downsample": 1,
                "retained_count": 999,
                "output_video_sha256": "0" * 64,
                "converter_commit": "0" * 40,
                "converter_script_sha256": "0" * 64,
                "codec_name": "av1",
                "pixel_format": "yuv444p",
                "width": 1,
                "height": 1,
                "average_frame_rate": "5",
                "nominal_frame_rate": "5",
                "frame_count": 999,
                "duration": "999",
                "audio_streams": [{"codec_name": "aac"}],
            }
            target[field] = replacements[field]

        _rewrite_jsonl(root / "meta/episodes.jsonl", change)

    mutated = _coherently_mutated_publication(
        publication, tmp_path / f"published-episode-{category}-{field}", mutate
    )
    _assert_code(mutated.dataset_root, None, code, final=True)


def test_semantic_published_provenance_order_and_input_root_order_are_bound(
    tmp_path, generated
):
    _, _, publication = generated
    for mutation, code in (
        ("episode_digest_order", "PROVENANCE_ORDER"),
        ("input_root_order", "PROVENANCE_SOURCE"),
    ):

        def mutate(root, *, mutation=mutation):
            _rewrite_json(
                root / "meta/conversion_provenance.json",
                lambda value: value[
                    "episode_provenance_sha256"
                    if mutation == "episode_digest_order"
                    else "input_roots"
                ].reverse(),
            )

        mutated = _coherently_mutated_publication(
            publication,
            tmp_path / f"published-{mutation}",
            mutate,
            rehash_episode_provenance=mutation != "episode_digest_order",
        )
        _assert_code(mutated.dataset_root, None, code, final=True)


@pytest.mark.parametrize(
    ("scope", "code"),
    [
        ("converter", "PROVENANCE_CONVERTER"),
        ("episode", "PROVENANCE_SOURCE"),
    ],
)
def test_semantic_published_provenance_rejects_unlisted_fields(
    tmp_path, generated, scope, code
):
    _, _, publication = generated

    def mutate(root):
        if scope == "converter":
            _rewrite_json(
                root / "meta/conversion_provenance.json",
                lambda value: value["converter"].__setitem__("extra", True),
            )
        else:
            _rewrite_jsonl(
                root / "meta/episodes.jsonl",
                lambda rows: rows[0]["conversion_provenance"].__setitem__(
                    "extra", True
                ),
            )

    mutated = _coherently_mutated_publication(
        publication, tmp_path / f"published-extra-{scope}", mutate
    )
    _assert_code(mutated.dataset_root, None, code, final=True)


def test_semantic_published_media_profile_validation_is_generator_independent(
    generated, monkeypatch
):
    _, _, publication = generated
    monkeypatch.setattr(
        converter,
        "media_profile",
        lambda _identity: (_ for _ in ()).throw(
            AssertionError("generator media helper was trusted")
        ),
    )
    assert (
        validate_dataset(
            publication.dataset_root, expected=None, require_final_modes=True
        ).total_episodes
        == 2
    )


def test_semantic_published_copy_all_requires_byte_identical_media(
    tmp_path, copy_generated
):
    _, plan, publication = copy_generated
    assert plan.media_mode == "copy_all"
    result = validate_dataset(
        publication.dataset_root, expected=None, require_final_modes=True
    )
    assert result.total_episodes == 2
    episode_rows = [
        json.loads(line)
        for line in (publication.dataset_root / "meta/episodes.jsonl")
        .read_text()
        .splitlines()
    ]
    assert all(
        row["conversion_provenance"]["source_video_sha256"]
        == row["conversion_provenance"]["output_video_sha256"]
        for row in episode_rows
    )

    def mutate(root):
        video = sorted(root.glob("videos/chunk-*/egocentric/*.mp4"))[0]
        with video.open("ab") as stream:
            stream.write(b"ignored-mp4-trailer")
        _rewrite_jsonl(
            root / "meta/episodes.jsonl",
            lambda rows: rows[0]["conversion_provenance"].__setitem__(
                "output_video_sha256", converter.sha256_file(video)
            ),
        )

    mutated = _coherently_mutated_publication(
        publication, tmp_path / "copy-all-trailer", mutate
    )
    _assert_code(mutated.dataset_root, None, "PROVENANCE_OUTPUT", final=True)


def test_semantic_published_tree_manifest_mode_and_link_are_checked(
    tmp_path, generated
):
    _, _, publication = generated
    for mutation, code in (
        ("extra", "TREE_MEMBERSHIP"),
        ("hash", "TREE_HASH"),
        ("size", "TREE_SIZE"),
        ("type", "TREE_TYPE"),
        ("mode", "TREE_MODE"),
        ("link", "TREE_LINK"),
        ("reserved", "TREE_RESERVED"),
    ):
        root = _published_copy(publication, tmp_path / mutation).dataset_root
        _unseal(root)
        if mutation == "extra":
            (root / "meta/extra").write_text("x")
        elif mutation == "hash":
            payload = bytearray((root / "meta/info.json").read_bytes())
            payload[0] ^= 1
            (root / "meta/info.json").write_bytes(payload)
        elif mutation == "size":
            with (root / "meta/info.json").open("ab") as stream:
                stream.write(b"x")
        elif mutation == "type":
            (root / "meta/info.json").unlink()
            (root / "meta/info.json").symlink_to("tasks.jsonl")
        elif mutation == "mode":
            (root / "meta/info.json").chmod(0o640)
        elif mutation == "link":
            os.link(root / "meta/info.json", root / "meta/alias.json")
        else:
            (root / "conversion_manifest.json").write_text("reserved")
        _assert_code(root, None, code, final=True)


def test_semantic_resealed_published_mutation_fixture_is_valid(tmp_path, generated):
    _, _, publication = generated
    repaired = _published_copy(publication, tmp_path / "repaired")
    assert (
        validate_dataset(
            repaired.dataset_root, expected=None, require_final_modes=True
        ).total_episodes
        == 2
    )


def _snapshot_tree(root: Path):
    return {
        path.relative_to(root).as_posix(): (
            path.lstat().st_dev,
            path.lstat().st_ino,
            path.lstat().st_mode,
            path.read_bytes() if path.is_file() else None,
        )
        for path in [root, *root.rglob("*")]
    }


def test_evidence_is_exclusive_sibling_sealed_and_terminal(generated, tmp_path):
    _, _, publication = generated
    certificate_id = uuid.UUID("00000000-0000-0000-0000-000000000123")
    before = _snapshot_tree(publication.dataset_root)
    with converter.conversion_lock(publication.dataset_root):
        evidence = certify_published_dataset(
            publication,
            psi0_root=Path(__file__).resolve().parents[1],
            psi0_commit=subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
            certificate_uuid=certificate_id,
        )
    assert evidence == publication.dataset_root.parent / (
        f".{publication.dataset_root.name}.certification-"
        f"{publication.manifest_sha256}-{certificate_id}"
    )
    assert not evidence.is_relative_to(publication.dataset_root)
    assert (evidence / "PASS.json").is_file()
    assert not (evidence / "FAIL.json").exists()
    assert stat.S_IMODE(evidence.stat().st_mode) == 0o555
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o444 for path in evidence.iterdir()
    )
    identity = json.loads((evidence / "dataset-root-identity.json").read_text())
    root_metadata = publication.dataset_root.stat(follow_symlinks=False)
    assert identity == {
        "path": str(publication.dataset_root.absolute()),
        "st_dev": root_metadata.st_dev,
        "st_ino": root_metadata.st_ino,
        "st_uid": root_metadata.st_uid,
        "st_gid": root_metadata.st_gid,
        "st_mode": root_metadata.st_mode,
        "st_nlink": root_metadata.st_nlink,
        "manifest_sha256": converter.sha256_file(
            publication.dataset_root / "meta/conversion_manifest.json"
        ),
        "complete_status_sha256": converter.sha256_file(
            publication.dataset_root / "CONVERSION_STATUS.json"
        ),
    }
    validate_evidence_terminal(evidence)
    assert _snapshot_tree(publication.dataset_root) == before
    with converter.conversion_lock(publication.dataset_root):
        with pytest.raises(FileExistsError):
            certify_published_dataset(
                publication,
                psi0_root=Path(__file__).resolve().parents[1],
                psi0_commit="0" * 40,
                python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
                certificate_uuid=certificate_id,
            )


def test_evidence_rejects_nonpublished_before_mkdir(generated):
    _, _, publication = generated
    uncertain = replace(publication, state="publication_uncertain")
    certificate_id = uuid.uuid4()
    expected_path = publication.dataset_root.parent / (
        f".{publication.dataset_root.name}.certification-"
        f"{publication.manifest_sha256}-{certificate_id}"
    )
    with pytest.raises(converter.PublicationUncertainError):
        certify_published_dataset(
            uncertain,
            psi0_root=Path(__file__).resolve().parents[1],
            psi0_commit="0" * 40,
            python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
            certificate_uuid=certificate_id,
        )
    assert not expected_path.exists()


def test_evidence_rejects_missing_lock_ownership_before_mkdir(generated):
    _, _, publication = generated
    certificate_id = uuid.uuid4()
    expected_path = publication.dataset_root.parent / (
        f".{publication.dataset_root.name}.certification-"
        f"{publication.manifest_sha256}-{certificate_id}"
    )
    with pytest.raises(RuntimeError, match="active conversion lock"):
        certify_published_dataset(
            publication,
            psi0_root=Path(__file__).resolve().parents[1],
            psi0_commit="0" * 40,
            python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
            certificate_uuid=certificate_id,
        )
    assert not expected_path.exists()


def test_evidence_forked_child_cannot_reuse_parent_lock_ownership(generated):
    _, _, publication = generated
    with converter.conversion_lock(publication.dataset_root):
        pid = os.fork()
        if pid == 0:
            try:
                converter.assert_conversion_lock_held(publication.dataset_root)
            except RuntimeError:
                os._exit(0)
            os._exit(1)
        waited, status = os.waitpid(pid, 0)
    assert waited == pid
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


@pytest.mark.parametrize(
    "mutation",
    ["result_manifest", "result_status", "disk_manifest", "disk_status"],
)
def test_evidence_root_identity_hashes_pinned_disk_and_preserves_original_error(
    tmp_path, generated, mutation
):
    _, _, original = generated
    publication = (
        _published_copy(original, tmp_path / mutation)
        if mutation.startswith("disk_")
        else original
    )
    if mutation == "result_manifest":
        publication = replace(publication, manifest_sha256="0" * 64)
    elif mutation == "result_status":
        publication = replace(publication, complete_status_sha256="0" * 64)
    elif mutation == "disk_manifest":
        _unseal(publication.dataset_root)
        with (publication.dataset_root / "meta/conversion_manifest.json").open(
            "ab"
        ) as stream:
            stream.write(b"x")
    elif mutation == "disk_status":
        _unseal(publication.dataset_root)
        with (publication.dataset_root / "CONVERSION_STATUS.json").open("ab") as stream:
            stream.write(b"x")
    certificate_id = uuid.uuid4()
    evidence = publication.dataset_root.parent / (
        f".{publication.dataset_root.name}.certification-"
        f"{publication.manifest_sha256}-{certificate_id}"
    )
    with converter.conversion_lock(publication.dataset_root):
        with pytest.raises(DatasetValidationError) as caught:
            certify_published_dataset(
                publication,
                psi0_root=Path(__file__).resolve().parents[1],
                psi0_commit="0" * 40,
                python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
                certificate_uuid=certificate_id,
            )
    assert caught.value.code == "PUBLICATION_IDENTITY"
    assert not evidence.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("manifest_sha256", "a" * 63 + "/"),
        ("manifest_sha256", "../escape"),
        ("manifest_sha256", "A" * 64),
        ("manifest_sha256", "." * 64),
        ("complete_status_sha256", "a" * 63 + "/"),
        ("complete_status_sha256", "A" * 64),
    ],
)
def test_evidence_rejects_unsafe_publication_digests_before_path_use(
    generated, field, value
):
    _, _, publication = generated
    parent = publication.dataset_root.parent
    # Materialize the persistent lock artifacts before taking the name snapshot.
    with converter.conversion_lock(publication.dataset_root):
        pass
    before = {path.relative_to(parent).as_posix() for path in parent.rglob("*")}
    invalid = replace(publication, **{field: value})
    with converter.conversion_lock(publication.dataset_root):
        with pytest.raises(DatasetValidationError) as caught:
            certify_published_dataset(
                invalid,
                psi0_root=Path(__file__).resolve().parents[1],
                psi0_commit="0" * 40,
                python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
                certificate_uuid=uuid.uuid4(),
            )
    assert caught.value.code == "PUBLICATION_IDENTITY"
    after = {path.relative_to(parent).as_posix() for path in parent.rglob("*")}
    assert after == before


def test_evidence_failure_is_terminal_and_preserves_dataset(generated, monkeypatch):
    _, _, publication = generated
    before = _snapshot_tree(publication.dataset_root)
    from scripts import certify_psi0_dataset as certifier

    monkeypatch.setattr(
        certifier,
        "_collect_psi0_environment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    certificate_id = uuid.uuid4()
    with converter.conversion_lock(publication.dataset_root):
        with pytest.raises(certifier.PublishedUncertifiedError) as caught:
            certify_published_dataset(
                publication,
                psi0_root=Path(__file__).resolve().parents[1],
                psi0_commit="0" * 40,
                python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
                certificate_uuid=certificate_id,
            )
    evidence = caught.value.evidence_root
    assert (evidence / "FAIL.json").is_file()
    assert not (evidence / "PASS.json").exists()
    validate_evidence_terminal(evidence)
    assert _snapshot_tree(publication.dataset_root) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "empty_entries",
        "partial_entries",
        "extra_regular",
        "extra_directory",
        "extra_symlink",
        "extra_fifo",
        "root_mode",
        "terminal_mode",
        "terminal_link",
        "terminal_symlink",
        "terminal_fifo",
        "terminal_bytes",
        "member_mode",
        "member_link",
        "member_bytes",
        "member_size",
        "unsafe_entry",
        "entry_schema",
        "entry_values",
    ],
)
def test_evidence_terminal_rejects_inexact_or_unsafe_members(
    tmp_path, certified, mutation
):
    root = _writable_copy(certified, tmp_path / mutation)
    terminal_path = root / "PASS.json"
    if mutation in {
        "empty_entries",
        "partial_entries",
        "unsafe_entry",
        "entry_schema",
        "entry_values",
    }:
        terminal = json.loads(terminal_path.read_text())
        if mutation == "empty_entries":
            terminal["entries"] = []
        elif mutation == "partial_entries":
            terminal["entries"] = terminal["entries"][:-1]
        else:
            if mutation == "unsafe_entry":
                terminal["entries"][0]["path"] = "../unsafe"
                terminal["entries"] = sorted(
                    terminal["entries"], key=lambda item: item["path"].encode()
                )
            elif mutation == "entry_schema":
                terminal["entries"][0] = None
            else:
                terminal["entries"][0]["path"] = []
        terminal_path.write_bytes(converter.canonical_json_bytes(terminal))
    elif mutation == "extra_regular":
        (root / "extra").write_text("x")
    elif mutation == "extra_directory":
        (root / "extra").mkdir()
    elif mutation == "extra_symlink":
        (root / "extra").symlink_to("certificate-request.json")
    elif mutation == "extra_fifo":
        os.mkfifo(root / "extra")
    elif mutation == "root_mode":
        pass
    elif mutation == "terminal_mode":
        terminal_path.chmod(0o640)
    elif mutation == "terminal_link":
        os.link(terminal_path, root / "terminal-alias")
    elif mutation == "terminal_symlink":
        terminal_path.unlink()
        terminal_path.symlink_to("certificate-request.json")
    elif mutation == "terminal_fifo":
        terminal_path.unlink()
        os.mkfifo(terminal_path)
    elif mutation == "terminal_bytes":
        terminal_path.write_bytes(terminal_path.read_bytes() + b"x")
    else:
        member = root / "certificate-request.json"
        if mutation == "member_mode":
            member.chmod(0o640)
        elif mutation == "member_link":
            os.link(member, root / "member-alias")
        elif mutation == "member_bytes":
            payload = bytearray(member.read_bytes())
            payload[-2] ^= 1
            member.write_bytes(payload)
        elif mutation == "member_size":
            member.write_bytes(member.read_bytes() + b"x")
    _seal_evidence_copy(root)
    if mutation == "root_mode":
        root.chmod(0o755)
    if mutation == "terminal_mode":
        terminal_path.chmod(0o640)
    if mutation == "member_mode":
        (root / "certificate-request.json").chmod(0o640)
    with pytest.raises((OSError, ValueError, DatasetValidationError)):
        validate_evidence_terminal(root)


@pytest.mark.parametrize("mutation", ["request_uuid", "dataset_identity"])
def test_evidence_terminal_cross_binds_request_and_dataset_identity(
    tmp_path, certified, mutation
):
    root = _writable_copy(certified, tmp_path / f"cross-bind-{mutation}")
    member_name = (
        "certificate-request.json"
        if mutation == "request_uuid"
        else "dataset-root-identity.json"
    )
    member = root / member_name
    if mutation == "request_uuid":
        _rewrite_json(
            member,
            lambda value: value.__setitem__("certificate_uuid", str(uuid.uuid4())),
        )
    else:
        _rewrite_json(
            member,
            lambda value: value.__setitem__("manifest_sha256", "0" * 64),
        )
    terminal_path = root / "PASS.json"
    terminal = json.loads(terminal_path.read_text())
    entry = next(item for item in terminal["entries"] if item["path"] == member_name)
    entry["size"] = member.stat().st_size
    entry["sha256"] = converter.sha256_file(member)
    terminal_path.write_bytes(converter.canonical_json_bytes(terminal))
    _seal_evidence_copy(root)
    with pytest.raises(ValueError, match="evidence identity binding differs"):
        validate_evidence_terminal(root)


@pytest.mark.parametrize("record", ["terminal", "request"])
def test_evidence_terminal_rejects_boolean_schema_version(tmp_path, certified, record):
    root = _writable_copy(certified, tmp_path / f"boolean-schema-{record}")
    terminal_path = root / "PASS.json"
    if record == "terminal":
        _rewrite_json(
            terminal_path, lambda value: value.__setitem__("schema_version", True)
        )
    else:
        request_path = root / "certificate-request.json"
        _rewrite_json(
            request_path, lambda value: value.__setitem__("schema_version", True)
        )
        terminal = json.loads(terminal_path.read_text())
        entry = next(
            item
            for item in terminal["entries"]
            if item["path"] == "certificate-request.json"
        )
        entry["size"] = request_path.stat().st_size
        entry["sha256"] = converter.sha256_file(request_path)
        terminal_path.write_bytes(converter.canonical_json_bytes(terminal))
    _seal_evidence_copy(root)
    with pytest.raises(ValueError):
        validate_evidence_terminal(root)


def test_evidence_terminal_is_bound_to_exact_sibling_name(tmp_path, certified):
    root = _writable_copy(certified, tmp_path / "arbitrary-certificate-copy")
    _seal_evidence_copy(root)
    with pytest.raises(ValueError, match="evidence root path differs"):
        validate_evidence_terminal(root)


def test_evidence_fail_terminal_requires_exact_durable_prefix(
    tmp_path, generated, monkeypatch
):
    from scripts import certify_psi0_dataset as certifier

    _, _, publication = generated
    monkeypatch.setattr(
        certifier,
        "_collect_psi0_environment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with converter.conversion_lock(publication.dataset_root):
        with pytest.raises(PublishedUncertifiedError) as caught:
            certify_published_dataset(
                publication,
                psi0_root=Path(__file__).resolve().parents[1],
                psi0_commit="0" * 40,
                python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
                certificate_uuid=uuid.uuid4(),
            )
    root = _writable_copy(caught.value.evidence_root, tmp_path / "fail-terminal")
    terminal_path = root / "FAIL.json"
    terminal = json.loads(terminal_path.read_text())
    terminal["entries"] = []
    terminal_path.write_bytes(converter.canonical_json_bytes(terminal))
    _seal_evidence_copy(root)
    with pytest.raises(ValueError, match="FAIL evidence completion prefix differs"):
        validate_evidence_terminal(root)


@pytest.mark.parametrize(
    "name",
    [
        "certificate-request.json",
        "dataset-root-identity.json",
        "dataset-validation.json",
        "source-validation.json",
        "psi0-environment.json",
        "psi0-loader-command.json",
        "psi0-loader-cache.json",
        "psi0-loader-result.json",
    ],
)
@pytest.mark.parametrize(
    "boundary", ["after_payload_fsync", "after_mode_fsync", "after_root_fsync"]
)
def test_evidence_each_file_fsync_failure_records_only_durable_prefix(
    generated, monkeypatch, name, boundary
):
    from scripts import certify_psi0_dataset as certifier

    _, _, publication = generated
    target = f"{name}:{boundary}"

    def fault(point):
        if point == target:
            raise RuntimeError(target)

    monkeypatch.setattr(certifier, "_evidence_fault", fault)
    certificate_id = uuid.uuid4()
    before = _snapshot_tree(publication.dataset_root)
    with converter.conversion_lock(publication.dataset_root):
        with pytest.raises(PublishedUncertifiedError) as caught:
            certify_published_dataset(
                publication,
                psi0_root=Path(__file__).resolve().parents[1],
                psi0_commit="0" * 40,
                python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
                certificate_uuid=certificate_id,
            )
    evidence = caught.value.evidence_root
    index = certifier._EVIDENCE_FILES.index(name)
    durable_count = index + (boundary == "after_root_fsync")
    terminal = evidence / "FAIL.json"
    if durable_count:
        assert terminal.is_file()
        value = validate_evidence_terminal(evidence)
        assert {entry["path"] for entry in value["entries"]} == set(
            certifier._EVIDENCE_FILES[:durable_count]
        )
    else:
        assert evidence.is_dir()
        assert not terminal.exists()
        assert not (evidence / "PASS.json").exists()
    assert _snapshot_tree(publication.dataset_root) == before


@pytest.mark.parametrize(
    "name",
    [
        "certificate-request.json",
        "dataset-root-identity.json",
        "dataset-validation.json",
        "source-validation.json",
        "psi0-environment.json",
        "psi0-loader-command.json",
        "psi0-loader-cache.json",
        "psi0-loader-result.json",
    ],
)
def test_evidence_post_rename_uncertainty_preserves_nonterminal_root(
    generated, monkeypatch, name
):
    from scripts import certify_psi0_dataset as certifier

    _, _, publication = generated
    target = f"{name}:after_rename"
    monkeypatch.setattr(
        certifier,
        "_evidence_fault",
        lambda point: (
            (_ for _ in ()).throw(RuntimeError(target)) if point == target else None
        ),
    )
    certificate_id = uuid.uuid4()
    with converter.conversion_lock(publication.dataset_root):
        with pytest.raises(PublishedUncertifiedError) as caught:
            certify_published_dataset(
                publication,
                psi0_root=Path(__file__).resolve().parents[1],
                psi0_commit="0" * 40,
                python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
                certificate_uuid=certificate_id,
            )
    evidence = caught.value.evidence_root
    assert evidence.is_dir()
    assert not (evidence / "PASS.json").exists()
    assert not (evidence / "FAIL.json").exists()


@pytest.mark.parametrize("terminal_name", ["PASS.json", "FAIL.json"])
@pytest.mark.parametrize(
    "boundary",
    ["after_payload_fsync", "after_mode_fsync", "after_rename", "after_root_fsync"],
)
def test_evidence_terminal_file_boundaries_never_leave_a_false_verdict(
    generated, monkeypatch, terminal_name, boundary
):
    from scripts import certify_psi0_dataset as certifier

    _, _, publication = generated
    if terminal_name == "FAIL.json":
        monkeypatch.setattr(
            certifier,
            "_collect_psi0_environment",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    target = f"{terminal_name}:{boundary}"
    monkeypatch.setattr(
        certifier,
        "_evidence_fault",
        lambda point: (
            (_ for _ in ()).throw(RuntimeError(target)) if point == target else None
        ),
    )
    with converter.conversion_lock(publication.dataset_root):
        with pytest.raises(PublishedUncertifiedError) as caught:
            certify_published_dataset(
                publication,
                psi0_root=Path(__file__).resolve().parents[1],
                psi0_commit="0" * 40,
                python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
                certificate_uuid=uuid.uuid4(),
            )
    evidence = caught.value.evidence_root
    has_fail = (evidence / "FAIL.json").exists()
    expected_fail = (terminal_name == "PASS.json" and boundary != "after_rename") or (
        terminal_name == "FAIL.json" and boundary == "after_root_fsync"
    )
    assert has_fail is expected_fail
    assert not (evidence / "PASS.json").exists()
    if has_fail:
        validate_evidence_terminal(evidence)


@pytest.mark.parametrize(
    "boundary",
    [
        "evidence_root:after_mkdir",
        "evidence_root:after_creation_parent_fsync",
        "evidence_root:after_root_fsync",
        "evidence_root:after_parent_fsync",
    ],
)
def test_evidence_root_and_parent_fsync_failures_never_claim_invalid_terminal(
    generated, monkeypatch, boundary
):
    from scripts import certify_psi0_dataset as certifier

    _, _, publication = generated
    monkeypatch.setattr(
        certifier,
        "_evidence_fault",
        lambda point: (
            (_ for _ in ()).throw(RuntimeError(boundary)) if point == boundary else None
        ),
    )
    with converter.conversion_lock(publication.dataset_root):
        with pytest.raises(PublishedUncertifiedError) as caught:
            certify_published_dataset(
                publication,
                psi0_root=Path(__file__).resolve().parents[1],
                psi0_commit="0" * 40,
                python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
                certificate_uuid=uuid.uuid4(),
            )
    evidence = caught.value.evidence_root
    assert evidence.is_dir()
    assert not (evidence / "PASS.json").exists()
    assert not (evidence / "FAIL.json").exists()


def test_evidence_fsync_boundaries_run_while_conversion_lock_is_owned(
    generated, monkeypatch
):
    from scripts import certify_psi0_dataset as certifier

    _, _, publication = generated
    events = []
    certificate_id = uuid.uuid4()
    expected_root = publication.dataset_root.parent / (
        f".{publication.dataset_root.name}.certification-"
        f"{publication.manifest_sha256}-{certificate_id}"
    )

    def record(point):
        converter.assert_conversion_lock_held(publication.dataset_root)
        if point == "evidence_root:after_mkdir":
            assert stat.S_IMODE(expected_root.stat().st_mode) == 0o700
        events.append(point)

    monkeypatch.setattr(certifier, "_evidence_fault", record)
    with converter.conversion_lock(publication.dataset_root):
        evidence = certify_published_dataset(
            publication,
            psi0_root=Path(__file__).resolve().parents[1],
            psi0_commit="0" * 40,
            python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
            certificate_uuid=certificate_id,
        )
    assert events[0:2] == [
        "evidence_root:after_mkdir",
        "evidence_root:after_creation_parent_fsync",
    ]
    for name in [*certifier._EVIDENCE_FILES, "PASS.json"]:
        positions = [
            events.index(f"{name}:{boundary}")
            for boundary in (
                "after_payload_fsync",
                "after_mode_fsync",
                "after_rename",
                "after_root_fsync",
            )
        ]
        assert positions == sorted(positions)
    assert events[-2:] == [
        "evidence_root:after_root_fsync",
        "evidence_root:after_parent_fsync",
    ]
    validate_evidence_terminal(evidence)


def test_evidence_incomplete_root_is_preserved_never_reused_and_retry_is_fresh(
    generated, monkeypatch
):
    from scripts import certify_psi0_dataset as certifier

    _, _, publication = generated
    first_id = uuid.uuid4()
    second_id = uuid.uuid4()
    target = "dataset-validation.json:after_rename"
    monkeypatch.setattr(
        certifier,
        "_evidence_fault",
        lambda point: (
            (_ for _ in ()).throw(RuntimeError(target)) if point == target else None
        ),
    )
    with converter.conversion_lock(publication.dataset_root):
        with pytest.raises(PublishedUncertifiedError) as first:
            certify_published_dataset(
                publication,
                psi0_root=Path(__file__).resolve().parents[1],
                psi0_commit="0" * 40,
                python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
                certificate_uuid=first_id,
            )
    before = _snapshot_tree(first.value.evidence_root)
    monkeypatch.setattr(certifier, "_evidence_fault", lambda _point: None)
    with converter.conversion_lock(publication.dataset_root):
        with pytest.raises(FileExistsError):
            certify_published_dataset(
                publication,
                psi0_root=Path(__file__).resolve().parents[1],
                psi0_commit="0" * 40,
                python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
                certificate_uuid=first_id,
            )
        retry = certify_published_dataset(
            publication,
            psi0_root=Path(__file__).resolve().parents[1],
            psi0_commit="0" * 40,
            python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
            certificate_uuid=second_id,
        )
    assert _snapshot_tree(first.value.evidence_root) == before
    assert retry != first.value.evidence_root
    validate_evidence_terminal(retry)


def test_evidence_real_exit_after_rename_preserves_incomplete_root_and_allows_retry(
    generated,
):
    from scripts import certify_psi0_dataset as certifier

    _, _, publication = generated
    crashed_id = uuid.uuid4()
    crashed_root = publication.dataset_root.parent / (
        f".{publication.dataset_root.name}.certification-"
        f"{publication.manifest_sha256}-{crashed_id}"
    )
    pid = os.fork()
    if pid == 0:
        target = "dataset-validation.json:after_rename"
        certifier._evidence_fault = lambda point: (
            os._exit(73) if point == target else None
        )
        try:
            with converter.conversion_lock(publication.dataset_root):
                certify_published_dataset(
                    publication,
                    psi0_root=Path(__file__).resolve().parents[1],
                    psi0_commit="0" * 40,
                    python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
                    certificate_uuid=crashed_id,
                )
        except BaseException:
            os._exit(74)
        os._exit(75)
    waited, status = os.waitpid(pid, 0)
    assert waited == pid
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 73
    assert crashed_root.is_dir()
    assert not (crashed_root / "PASS.json").exists()
    assert not (crashed_root / "FAIL.json").exists()

    with converter.conversion_lock(publication.dataset_root):
        retry = certify_published_dataset(
            publication,
            psi0_root=Path(__file__).resolve().parents[1],
            psi0_commit="0" * 40,
            python=Path("/home/jihun/work/SIMPLE/.venv/bin/python"),
            certificate_uuid=uuid.uuid4(),
        )
    validate_evidence_terminal(retry)


def test_evidence_uncertain_recovery_revalidates_and_fsyncs_under_lock(
    tmp_path, generated, monkeypatch
):
    from scripts import certify_psi0_dataset as certifier

    _, _, original = generated
    publication = _published_copy(original, tmp_path / "uncertain-recovery")
    before = _snapshot_tree(publication.dataset_root)
    events = []
    original_fsync = converter.PublicationFilesystem.fsync_directory
    original_certify = certify_published_dataset

    def fsync_directory(self, path):
        converter.assert_conversion_lock_held(publication.dataset_root)
        events.append(("parent_fsync", path))
        return original_fsync(self, path)

    def certify(publication_result, **kwargs):
        converter.assert_conversion_lock_held(publication.dataset_root)
        assert publication_result.state == "published"
        assert events == [("parent_fsync", publication.dataset_root.parent)]
        events.append(("certify", publication_result.dataset_root))
        return original_certify(publication_result, **kwargs)

    monkeypatch.setattr(
        converter.PublicationFilesystem, "fsync_directory", fsync_directory
    )
    monkeypatch.setattr(certifier, "certify_published_dataset", certify)
    assert (
        certifier.main(
            [
                "--dataset-root",
                str(publication.dataset_root),
                "--recover-publication-uncertain",
                "--psi0-root",
                str(Path(__file__).resolve().parents[1]),
                "--psi0-commit",
                "0" * 40,
                "--python",
                "/home/jihun/work/SIMPLE/.venv/bin/python",
            ]
        )
        == 0
    )
    assert events[0][0] == "parent_fsync"
    assert events[1][0] == "certify"
    assert _snapshot_tree(publication.dataset_root) == before
    evidence = list(
        publication.dataset_root.parent.glob(
            f".{publication.dataset_root.name}.certification-"
            f"{publication.manifest_sha256}-*"
        )
    )
    assert len(evidence) == 1
    validate_evidence_terminal(evidence[0])


def test_evidence_uncertain_recovery_rejects_corruption_before_evidence(
    tmp_path, generated
):
    from scripts import certify_psi0_dataset as certifier

    _, _, original = generated
    publication = _published_copy(original, tmp_path / "uncertain-corrupt")
    _unseal(publication.dataset_root)
    with (publication.dataset_root / "meta/info.json").open("ab") as stream:
        stream.write(b"x")
    before = _snapshot_tree(publication.dataset_root)
    assert (
        certifier.main(
            [
                "--dataset-root",
                str(publication.dataset_root),
                "--recover-publication-uncertain",
                "--psi0-root",
                str(Path(__file__).resolve().parents[1]),
                "--psi0-commit",
                "0" * 40,
                "--python",
                "/home/jihun/work/SIMPLE/.venv/bin/python",
            ]
        )
        == 1
    )
    assert not list(
        publication.dataset_root.parent.glob(
            f".{publication.dataset_root.name}.certification-*"
        )
    )
    assert _snapshot_tree(publication.dataset_root) == before


def test_evidence_print_tree_digest_is_read_only_and_requires_published_tree(
    tmp_path, generated, capsys
):
    from scripts import certify_psi0_dataset as certifier

    _, plan, publication = generated
    before = _snapshot_tree(publication.dataset_root)
    assert (
        certifier.main(
            ["--dataset-root", str(publication.dataset_root), "--print-tree-digest"]
        )
        == 0
    )
    printed = capsys.readouterr()
    assert (
        printed.out.strip()
        == validate_dataset(
            publication.dataset_root, expected=None, require_final_modes=True
        ).tree_digest
    )
    assert printed.err == ""
    assert _snapshot_tree(publication.dataset_root) == before

    staging = tmp_path / ".manifestless.staging"
    converter.generate_staged_dataset(plan, staging)
    for path in [staging, *staging.rglob("*")]:
        path.chmod(0o555 if path.is_dir() else 0o444)
    assert certifier.main(["--dataset-root", str(staging), "--print-tree-digest"]) == 1
    assert not list(staging.parent.glob(".manifestless.certification-*"))


def _read_canonical_evidence(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    value = json.loads(payload)
    assert payload == converter.canonical_json_bytes(value)
    return value


def test_psi0_loader_parent_uses_isolated_worker_and_traverses_every_row(
    tmp_path, generated
):
    _, _, publication = generated
    checkout, commit, import_marker = _make_fake_psi0_checkout(tmp_path)
    evidence = _certify_with_fake_loader(publication, checkout, commit, uuid.uuid4())

    assert import_marker.read_text() == "imported"
    assert not (evidence / ".loader-cache").exists()
    assert validate_evidence_terminal(evidence)["verdict"] == "PASS"

    environment = _read_canonical_evidence(evidence / "psi0-environment.json")
    details = environment["details"]
    assert environment["details_sha256"] == converter.sha256_bytes(
        converter.canonical_json_bytes(details)
    )
    assert details["psi0_commit"] == commit
    assert details["tracked_status"] == []
    assert details["psi0_root"] == str(checkout.resolve())
    assert details["packages"] == {
        name: details["packages"][name]
        for name in ("av", "datasets", "numpy", "pyarrow", "torch", "torchvision")
    }
    assert details["distributions"] == sorted(
        details["distributions"], key=lambda item: (item["name"], item["version"])
    )
    assert details["distribution_manifest_sha256"] == converter.sha256_bytes(
        converter.canonical_json_bytes(details["distributions"])
    )

    command = _read_canonical_evidence(evidence / "psi0-loader-command.json")
    assert command["argv"][1:3] == ["-I", "scripts/certify_psi0_dataset.py"]
    assert command["argv"][3] == "--loader-worker"
    worker_environment = command["environment"]
    assert set(worker_environment) == {
        "PATH",
        "HOME",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "TMPDIR",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "PYTHONNOUSERSITE",
    }
    cache_root = evidence / ".loader-cache"
    assert Path(worker_environment["HOME"]) == cache_root / "home"
    assert Path(worker_environment["HF_HOME"]) == cache_root / "hf"
    assert Path(worker_environment["HF_DATASETS_CACHE"]) == cache_root / "datasets"
    assert Path(worker_environment["XDG_CACHE_HOME"]) == cache_root / "xdg"
    assert Path(worker_environment["TORCH_HOME"]) == cache_root / "torch"
    assert Path(worker_environment["TMPDIR"]) == cache_root / "tmp"
    assert command["environment_sha256"] == converter.sha256_bytes(
        converter.canonical_json_bytes(worker_environment)
    )

    cache = _read_canonical_evidence(evidence / "psi0-loader-cache.json")
    assert cache["entries"]
    assert cache["entries"] == sorted(
        cache["entries"], key=lambda item: item["path"].encode()
    )
    assert {entry["type"] for entry in cache["entries"]} == {
        "directory",
        "regular",
    }
    visits = next(
        entry for entry in cache["entries"] if entry["path"] == "tmp/visited.jsonl"
    )
    assert visits["sha256"] == converter.sha256_bytes(b"0\n1\n2\n3\n4\n")
    result = _read_canonical_evidence(evidence / "psi0-loader-result.json")
    worker_result = result["result"]
    assert result["result_sha256"] == converter.sha256_bytes(
        converter.canonical_json_bytes(worker_result)
    )
    assert worker_result["verdict"] == "PASS"
    assert worker_result["visited_indices"] == list(range(5))
    assert worker_result["episode_ranges"] == [[0, 1], [2, 4]]
    assert worker_result["sys_path_0"] == str((checkout / "src").resolve())
    assert worker_result["module_provenance"] == {
        "blob_sha256": converter.sha256_bytes(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "show",
                    f"{commit}:src/psi/data/lerobot/compat.py",
                ],
                check=True,
                capture_output=True,
            ).stdout
        ),
        "commit": commit,
        "origin": str((checkout / "src/psi/data/lerobot/compat.py").resolve()),
        "relative_path": "src/psi/data/lerobot/compat.py",
    }
    assert worker_result["network_policy"] == {
        "audit_hook": "deny_inet_resolver_process_escape",
        "denied_events": LOADER_DENIED_AUDIT_EVENTS,
        "violations": [],
    }
    assert worker_result["tensor_contract"] == {
        "action": {"dtype": "torch.float32", "shape": [36]},
        "observation.images.egocentric": {
            "dtype": "torch.float32",
            "shape": [3, 360, 640],
        },
        "states": {"dtype": "torch.float32", "shape": [32]},
    }


def _coherently_rewrite_evidence_member(evidence: Path, name: str, mutate) -> None:
    evidence.chmod(0o755)
    member = evidence / name
    terminal_path = evidence / "PASS.json"
    member.chmod(0o644)
    terminal_path.chmod(0o644)
    value = json.loads(member.read_text())
    mutate(value)
    member.write_bytes(converter.canonical_json_bytes(value))
    terminal = json.loads(terminal_path.read_text())
    entry = next(item for item in terminal["entries"] if item["path"] == name)
    entry["size"] = member.stat().st_size
    entry["sha256"] = converter.sha256_file(member)
    terminal_path.write_bytes(converter.canonical_json_bytes(terminal))
    member.chmod(0o444)
    terminal_path.chmod(0o444)
    evidence.chmod(0o555)


@pytest.mark.parametrize(
    "mutation",
    [
        "visited_indices",
        "episode_ranges",
        "tensor_contract",
        "cache_empty",
        "cache_missing_root",
        "cache_cross_type",
        "cache_noncanonical_path",
        "cache_extra_root",
        "cache_missing_parent",
        "cache_descendant_under_file",
    ],
)
def test_psi0_loader_terminal_semantics_are_bound_to_dataset_and_cache_roots(
    tmp_path, generated, mutation
):
    _, _, publication = generated
    checkout, commit, _ = _make_fake_psi0_checkout(tmp_path)
    evidence = _certify_with_fake_loader(publication, checkout, commit, uuid.uuid4())

    if mutation.startswith("cache_"):

        def mutate_cache(value):
            entries = value["entries"]
            if mutation == "cache_empty":
                entries.clear()
            elif mutation == "cache_missing_root":
                entries[:] = [entry for entry in entries if entry["path"] != "home"]
            elif mutation == "cache_cross_type":
                home = next(entry for entry in entries if entry["path"] == "home")
                home["type"] = "regular"
                home["sha256"] = "0" * 64
            elif mutation == "cache_noncanonical_path":
                entries.append(
                    {
                        "path": "home//nested",
                        "type": "directory",
                        "size": 0,
                        "sha256": None,
                    }
                )
                entries.sort(key=lambda entry: entry["path"].encode())
            elif mutation == "cache_extra_root":
                entries.append(
                    {
                        "path": "rogue",
                        "type": "directory",
                        "size": 0,
                        "sha256": None,
                    }
                )
                entries.sort(key=lambda entry: entry["path"].encode())
            elif mutation == "cache_missing_parent":
                entries[:] = [
                    entry for entry in entries if entry["path"] != "hf/fake-hub"
                ]
            else:
                entries.append(
                    {
                        "path": "datasets/fake-dataset.bin/descendant",
                        "type": "directory",
                        "size": 0,
                        "sha256": None,
                    }
                )
                entries.sort(key=lambda entry: entry["path"].encode())

        _coherently_rewrite_evidence_member(
            evidence, "psi0-loader-cache.json", mutate_cache
        )
    else:

        def mutate_result(value):
            result = value["result"]
            if mutation == "visited_indices":
                result["visited_indices"] = [0, 1, 3, 2, 4]
            elif mutation == "episode_ranges":
                result["episode_ranges"] = [[0, 2], [3, 4]]
            else:
                result["tensor_contract"]["action"]["shape"] = [35]
            value["result_sha256"] = converter.sha256_bytes(
                converter.canonical_json_bytes(result)
            )

        _coherently_rewrite_evidence_member(
            evidence, "psi0-loader-result.json", mutate_result
        )

    with pytest.raises(ValueError, match="PSI0 loader"):
        validate_evidence_terminal(evidence)


@pytest.mark.parametrize(
    "behavior",
    ["wrong_shape", "wrong_dtype", "nonfinite", "gap", "duplicate", "missing"],
)
def test_psi0_loader_rejects_invalid_samples_and_cleans_cache(
    tmp_path, generated, behavior
):
    _, _, publication = generated
    checkout, commit, import_marker = _make_fake_psi0_checkout(
        tmp_path, behavior=behavior
    )
    with pytest.raises(PublishedUncertifiedError) as caught:
        _certify_with_fake_loader(publication, checkout, commit, uuid.uuid4())

    evidence = caught.value.evidence_root
    assert import_marker.is_file()
    assert not (evidence / ".loader-cache").exists()
    assert not list(evidence.glob(".psi0-loader-result-*.tmp"))
    terminal = validate_evidence_terminal(evidence)
    assert terminal["verdict"] == "FAIL"
    assert terminal["error_code"] == "PSI0_LOADER_VALIDATION"
    result = _read_canonical_evidence(evidence / "psi0-loader-result.json")
    assert result["result"]["verdict"] == "FAIL"
    assert result["result_sha256"] == converter.sha256_bytes(
        converter.canonical_json_bytes(result["result"])
    )


def test_psi0_loader_revalidates_checkout_after_cache_setup_before_import(
    tmp_path, generated, monkeypatch
):
    _, _, publication = generated
    checkout, commit, import_marker = _make_fake_psi0_checkout(tmp_path)

    def mutate_checkout(point):
        if point == "after_cache_setup":
            (checkout / "src/psi/data/lerobot/compat.py").write_text(
                "mutated after initial validation\n"
            )

    monkeypatch.setattr(certifier, "_loader_fault", mutate_checkout, raising=False)
    with pytest.raises(PublishedUncertifiedError) as caught:
        _certify_with_fake_loader(publication, checkout, commit, uuid.uuid4())

    assert not import_marker.exists()
    assert not (caught.value.evidence_root / "PASS.json").exists()


def test_psi0_loader_rejects_untracked_unchecked_hash_pyc_before_import(
    tmp_path, generated
):
    _, _, publication = generated
    checkout, commit, import_marker = _make_fake_psi0_checkout(tmp_path)
    pyc_marker = tmp_path / "unchecked-pyc-executed"
    _install_unchecked_hash_compat_pyc(checkout, pyc_marker)

    with pytest.raises(PublishedUncertifiedError) as caught:
        _certify_with_fake_loader(publication, checkout, commit, uuid.uuid4())

    evidence = caught.value.evidence_root
    assert not pyc_marker.exists()
    assert not import_marker.exists()
    assert not (evidence / "PASS.json").exists()


@pytest.mark.parametrize("kind", ["fifo", "multilink"])
def test_checkout_bytecode_scanner_rejects_unsafe_metadata_without_opening(
    tmp_path, monkeypatch, kind
):
    source = tmp_path / f"{kind}-source"
    pycache = source / "__pycache__"
    pycache.mkdir(parents=True)
    module = source / "module.py"
    module.write_text("VALUE = 1\n")
    cache_path = Path(importlib.util.cache_from_source(str(module)))
    if kind == "fifo":
        os.mkfifo(cache_path, 0o600)
    else:
        backing = tmp_path / "multilink-bytecode"
        backing.write_bytes(b"not opened")
        os.link(backing, cache_path)
    original_open = certifier.os.open

    def forbid_unsafe_open(path, flags, *args, **kwargs):
        if path == cache_path.name:
            pytest.fail("checkout scanner opened unsafe bytecode metadata")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(certifier.os, "open", forbid_unsafe_open)
    started = time.monotonic()
    with pytest.raises(DatasetValidationError, match="PSI0_CHECKOUT"):
        certifier._validate_checkout_bytecode_caches(source)
    assert time.monotonic() - started < 2.0


def test_checkout_bytecode_scanner_nonblocking_open_rejects_fifo_replacement(
    tmp_path, monkeypatch
):
    source = tmp_path / "swap-source"
    pycache = source / "__pycache__"
    pycache.mkdir(parents=True)
    module = source / "module.py"
    module.write_text("VALUE = 1\n")
    cache_path = Path(importlib.util.cache_from_source(str(module)))
    py_compile.compile(
        str(module),
        cfile=str(cache_path),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    original_open = certifier.os.open
    swapped = False

    def replace_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == cache_path.name and not swapped:
            assert flags & os.O_NONBLOCK
            dir_fd = kwargs["dir_fd"]
            os.unlink(path, dir_fd=dir_fd)
            os.mkfifo(path, 0o600, dir_fd=dir_fd)
            swapped = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(certifier.os, "open", replace_before_open)
    started = time.monotonic()
    with pytest.raises(DatasetValidationError, match="PSI0_CHECKOUT"):
        certifier._validate_checkout_bytecode_caches(source)
    assert swapped
    assert time.monotonic() - started < 2.0


def test_psi0_loader_rejects_direct_sourceless_timestamp_pyc_before_import(
    tmp_path, generated
):
    _, _, publication = generated
    checkout, commit, import_marker = _make_fake_psi0_checkout(
        tmp_path, behavior="sourceless_pyc"
    )
    evil_marker = tmp_path / "direct-evil-pyc-executed"
    evil_source = tmp_path / "evil-source.py"
    evil_source.write_text(
        f"from pathlib import Path\nPath({str(evil_marker)!r}).write_text('executed')\n"
    )
    py_compile.compile(
        str(evil_source),
        cfile=str(checkout / "src/evil.pyc"),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )

    with pytest.raises(PublishedUncertifiedError) as caught:
        _certify_with_fake_loader(publication, checkout, commit, uuid.uuid4())

    terminal = validate_evidence_terminal(caught.value.evidence_root)
    assert terminal["error_code"] == "PSI0_CHECKOUT"
    assert not evil_marker.exists()
    assert not import_marker.exists()


def test_psi0_loader_blocks_sourceless_pyc_added_after_worker_scan(tmp_path, generated):
    _, _, publication = generated
    checkout, commit, import_marker = _make_fake_psi0_checkout(
        tmp_path, behavior="race_sourceless_pyc"
    )
    evil_marker = tmp_path / "race-sourceless-pyc-executed"

    with pytest.raises(PublishedUncertifiedError) as caught:
        _certify_with_fake_loader(publication, checkout, commit, uuid.uuid4())

    assert not (caught.value.evidence_root / "PASS.json").exists()
    assert not evil_marker.exists()
    assert not import_marker.exists()


def test_psi0_loader_worker_rejects_untracked_unchecked_hash_pyc_before_import(
    tmp_path, generated
):
    _, _, publication = generated
    checkout, commit, import_marker = _make_fake_psi0_checkout(
        tmp_path, behavior="worker-pyc"
    )
    pyc_marker = tmp_path / "worker-unchecked-pyc-executed"
    _install_unchecked_hash_compat_pyc(checkout, pyc_marker)
    cache_root = tmp_path / "worker-cache"
    cache_root.mkdir(mode=0o700)
    for name in ("home", "hf", "datasets", "xdg", "torch", "tmp"):
        (cache_root / name).mkdir(mode=0o700)
    result_path = tmp_path / "worker-result.json"
    module_identity = certifier._compat_module_identity(checkout / "src", commit)
    result_path.write_bytes(
        converter.canonical_json_bytes(
            {
                "schema_version": 1,
                "expected_commit": commit,
                "module_identity": module_identity,
            }
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "scripts/certify_psi0_dataset.py",
            "--loader-worker",
            "--psi0-src",
            str(checkout / "src"),
            "--dataset-root",
            str(publication.dataset_root),
            "--result",
            str(result_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=certifier._worker_environment(cache_root),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert json.loads(result_path.read_bytes())["verdict"] == "FAIL"
    assert not pyc_marker.exists()
    assert not import_marker.exists()


def test_psi0_loader_redirects_bytecode_cache_before_import(tmp_path, generated):
    _, _, publication = generated
    checkout, commit, import_marker = _make_fake_psi0_checkout(
        tmp_path, behavior="pycache_prefix"
    )

    evidence = _certify_with_fake_loader(publication, checkout, commit, uuid.uuid4())

    assert import_marker.is_file()
    assert validate_evidence_terminal(evidence)["verdict"] == "PASS"


def test_task11_import_probe_timestamp_pyc_is_ignored_during_certification(
    tmp_path, generated
):
    _, _, publication = generated
    checkout, commit, import_marker = _make_fake_psi0_checkout(
        tmp_path, behavior="task11-probe"
    )
    source = checkout / "src"
    probe_environment = dict(os.environ)
    probe_environment.pop("PYTHONDONTWRITEBYTECODE", None)
    probe_environment.pop("PYTHONPYCACHEPREFIX", None)
    probe_environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    probe = subprocess.run(
        [sys.executable, "-I", "-", str(source)],
        input="""\
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(source))
from psi.data.lerobot.compat import LEROBOT_LAYOUT, LeRobotDataset

print(LEROBOT_LAYOUT, LeRobotDataset)
""",
        env=probe_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    checkout_pycs = sorted(source.rglob("*.pyc"))
    assert checkout_pycs
    assert all(
        int.from_bytes(path.read_bytes()[4:8], "little") == 0 for path in checkout_pycs
    )

    import_marker.unlink()
    pyc_payload_marker = tmp_path / "task11-probe-pyc-executed"
    _replace_with_malicious_timestamp_compat_pyc(checkout, pyc_payload_marker)

    evidence = _certify_with_fake_loader(publication, checkout, commit, uuid.uuid4())

    assert validate_evidence_terminal(evidence)["verdict"] == "PASS"
    assert import_marker.read_text() == "imported"
    assert not pyc_payload_marker.exists()


@pytest.mark.parametrize(
    ("behavior", "expected_event"),
    [
        ("network", "socket.__new__"),
        ("resolver", "socket.getaddrinfo"),
        ("subprocess", "subprocess.Popen"),
    ],
)
def test_psi0_loader_network_and_process_attempts_fail_and_clean_cache(
    tmp_path, generated, behavior, expected_event
):
    _, _, publication = generated
    checkout, commit, import_marker = _make_fake_psi0_checkout(
        tmp_path, behavior=behavior
    )
    with pytest.raises(PublishedUncertifiedError) as caught:
        _certify_with_fake_loader(publication, checkout, commit, uuid.uuid4())

    evidence = caught.value.evidence_root
    assert import_marker.is_file()
    assert not (evidence / ".loader-cache").exists()
    terminal = validate_evidence_terminal(evidence)
    assert terminal["verdict"] == "FAIL"
    result = _read_canonical_evidence(evidence / "psi0-loader-result.json")["result"]
    assert result["network_policy"]["violations"]
    assert result["network_policy"]["violations"][0]["event"] == expected_event


def test_psi0_loader_primes_urllib3_without_inet_capability_probe(
    tmp_path,
    generated,
):
    _, _, publication = generated
    checkout, commit, import_marker = _make_fake_psi0_checkout(
        tmp_path, behavior="urllib3_import"
    )

    evidence = _certify_with_fake_loader(publication, checkout, commit, uuid.uuid4())

    assert import_marker.is_file()
    assert validate_evidence_terminal(evidence)["verdict"] == "PASS"
    result = _read_canonical_evidence(evidence / "psi0-loader-result.json")
    assert result["returncode"] == 0
    assert result["result"]["network_policy"]["violations"] == []


def test_psi0_loader_caught_network_attempt_is_worker_failure(
    tmp_path,
    generated,
):
    _, _, publication = generated
    checkout, commit, import_marker = _make_fake_psi0_checkout(
        tmp_path, behavior="caught_network"
    )

    with pytest.raises(PublishedUncertifiedError) as caught:
        _certify_with_fake_loader(publication, checkout, commit, uuid.uuid4())

    evidence = caught.value.evidence_root
    assert import_marker.is_file()
    terminal = validate_evidence_terminal(evidence)
    assert terminal["verdict"] == "FAIL"
    result = _read_canonical_evidence(evidence / "psi0-loader-result.json")
    assert result["returncode"] == 1
    assert result["result"]["verdict"] == "FAIL"
    assert result["result"]["network_policy"]["violations"] == [
        {"category": "inet_socket", "event": "socket.__new__"}
    ]


@pytest.mark.parametrize("failure", ["dirty", "wrong_commit"])
def test_psi0_loader_rejects_unpinned_checkout_before_import(
    tmp_path, generated, failure
):
    _, _, publication = generated
    checkout, commit, import_marker = _make_fake_psi0_checkout(tmp_path)
    if failure == "dirty":
        (checkout / "src/psi/data/lerobot/compat.py").write_text("dirty tracked file\n")
    else:
        commit = "0" * 40

    with pytest.raises(PublishedUncertifiedError) as caught:
        _certify_with_fake_loader(publication, checkout, commit, uuid.uuid4())

    evidence = caught.value.evidence_root
    assert not import_marker.exists()
    assert not (evidence / ".loader-cache").exists()
    terminal = validate_evidence_terminal(evidence)
    assert terminal["verdict"] == "FAIL"
    assert terminal["error_code"] == "PSI0_CHECKOUT"


def test_psi0_loader_cache_cleanup_failure_forbids_terminal_evidence(
    tmp_path, generated, monkeypatch
):
    _, _, publication = generated
    checkout, commit, _ = _make_fake_psi0_checkout(tmp_path)

    def fail_cleanup(*_args, **_kwargs):
        raise OSError("injected cache cleanup failure")

    monkeypatch.setattr(certifier, "_remove_loader_cache", fail_cleanup, raising=False)
    with pytest.raises(PublishedUncertifiedError) as caught:
        _certify_with_fake_loader(publication, checkout, commit, uuid.uuid4())

    assert caught.value.code == "CACHE_CLEANUP_FAILED"
    evidence = caught.value.evidence_root
    assert (evidence / ".loader-cache").is_dir()
    assert not (evidence / "PASS.json").exists()
    assert not (evidence / "FAIL.json").exists()


def test_psi0_loader_cache_cleanup_cli_reports_absolute_evidence_root(
    tmp_path, generated, monkeypatch, capsys
):
    _, _, original = generated
    publication = _published_copy(original, tmp_path / "cleanup-cli")
    checkout, commit, _ = _make_fake_psi0_checkout(tmp_path)

    monkeypatch.setattr(
        certifier,
        "_remove_loader_cache",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cleanup failed")),
    )
    assert (
        certifier.main(
            [
                "--dataset-root",
                str(publication.dataset_root),
                "--recover-publication-uncertain",
                "--psi0-root",
                str(checkout),
                "--psi0-commit",
                commit,
                "--python",
                sys.executable,
            ]
        )
        == 1
    )
    evidence_roots = list(
        publication.dataset_root.parent.glob(
            f".{publication.dataset_root.name}.certification-*"
        )
    )
    assert len(evidence_roots) == 1
    stderr = capsys.readouterr().err
    assert "CACHE_CLEANUP_FAILED" in stderr
    assert str(evidence_roots[0].absolute()) in stderr
    assert not (evidence_roots[0] / "PASS.json").exists()
    assert not (evidence_roots[0] / "FAIL.json").exists()


@pytest.mark.production_loader
def test_real_pinned_psi0_loader_traverses_every_synthetic_row(tmp_path):
    psi0_root_value = os.environ.get("PSI0_PRODUCTION_ROOT")
    psi0_commit = os.environ.get("PSI0_PRODUCTION_COMMIT")
    assert psi0_root_value, "PSI0_PRODUCTION_ROOT is required"
    assert psi0_commit, "PSI0_PRODUCTION_COMMIT is required"
    psi0_root = Path(psi0_root_value)

    plan = _make_copy_plan(tmp_path)
    staging = tmp_path / ".copy-dataset.production-loader-staging"
    converter.generate_staged_dataset(plan, staging)
    publication = converter.publish_staged_dataset(
        staging,
        plan.output_path,
        plan.converter,
        converter.PublicationFilesystem(),
        lambda _point: None,
    )
    with converter.conversion_lock(publication.dataset_root):
        evidence = certify_published_dataset(
            publication,
            psi0_root=psi0_root,
            psi0_commit=psi0_commit,
            python=Path(sys.executable),
            certificate_uuid=uuid.uuid4(),
        )

    assert validate_evidence_terminal(evidence)["verdict"] == "PASS"
    assert not (evidence / ".loader-cache").exists()
    cache = _read_canonical_evidence(evidence / "psi0-loader-cache.json")
    assert any(
        entry["type"] == "regular"
        and Path(entry["path"]).parts[0] in {"hf", "datasets"}
        for entry in cache["entries"]
    )
    result = _read_canonical_evidence(evidence / "psi0-loader-result.json")
    worker_result = result["result"]
    assert worker_result["visited_indices"] == list(range(7))
    assert worker_result["episode_ranges"] == [[0, 2], [3, 6]]
    assert worker_result["network_policy"]["violations"] == []
    assert worker_result["tensor_contract"] == {
        "action": {"dtype": "torch.float32", "shape": [36]},
        "observation.images.egocentric": {
            "dtype": "torch.float32",
            "shape": [3, 48, 64],
        },
        "states": {"dtype": "torch.float32", "shape": [32]},
    }
