from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from simple.eval_runtime.assets import AssetRequirement, probe_episode_assets
from simple.core.simulator import Simulator
from simple.scenes.hssd import HssdSceneManager
from simple.utils import resolve_data_path, resolve_res_path


def test_data_root_seam_is_offline_and_never_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "sealed-data"
    root.mkdir()
    monkeypatch.setenv("SIMPLE_DATA_ROOT", str(root))
    monkeypatch.setenv("SIMPLE_ASSET_OFFLINE", "1")
    monkeypatch.setattr(
        "simple.utils.snapshot_download", lambda **_: pytest.fail("network")
    )
    with pytest.raises(FileNotFoundError):
        resolve_data_path("assets/missing/item.usd", auto_download=True)


def test_hssd_load_uses_normalized_closure_without_shell_or_tmp_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "task-data"
    configs = yaml.safe_load(
        Path(resolve_res_path("hssd-scenes/config.yaml")).read_text(encoding="utf-8")
    )
    config = next(item for item in configs if item["uid"] == "scene0")
    scene_name = config["name"]
    scene = data / config["data_dir"] / scene_name
    scene.mkdir(parents=True)
    usd = scene / f"{scene_name}.usd"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    usd_sha256 = hashlib.sha256(usd.read_bytes()).hexdigest()
    (scene / "NORMALIZED.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scene_uid": "scene0",
                "scene_name": scene_name,
                "source_usd_sha256": "a" * 64,
                "normalized_usd_sha256": usd_sha256,
                "normalization_policy_version": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SIMPLE_DATA_ROOT", str(data))
    monkeypatch.setenv("SIMPLE_ASSET_OFFLINE", "1")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("subprocess"))
    monkeypatch.setattr("shutil.copytree", lambda *a, **k: pytest.fail("copytree"))
    before = set(Path("/tmp").iterdir())
    loaded = HssdSceneManager().load("hssd:scene0")
    assert loaded.uid == "hssd:scene0"
    assert loaded.name == "107734119_175999932"
    assert set(Path("/tmp").iterdir()) == before


def test_episode_asset_probe_rejects_missing_or_out_of_root_dependency(
    tmp_path: Path,
) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    required = (
        AssetRequirement(
            "scene",
            "scenes/hssd/107734119_175999932/107734119_175999932.usd",
            "a" * 64,
        ),
        AssetRequirement("robot", "robots/g1/g1.usd", "b" * 64),
    )
    with pytest.raises(Exception, match="ASSET_PROBE"):
        probe_episode_assets(root, required, episode_index=2)


def test_simulator_data_resolvers_delegate_to_sealed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "sealed-data"
    target = root / "robots" / "g1.usd"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"usd")
    monkeypatch.setenv("SIMPLE_DATA_ROOT", str(root))
    monkeypatch.setenv("SIMPLE_ASSET_OFFLINE", "1")
    simulator = Simulator()
    assert simulator.get_data_dir() == str(root)
    assert simulator.resolve_data_path("robots/g1.usd") == str(target)
    with pytest.raises(FileNotFoundError):
        simulator.resolve_data_path("robots/missing.usd", create_if_not_exist=True)
