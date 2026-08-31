"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import TYPE_CHECKING

import numpy as np
import yaml
from dotenv import load_dotenv

from simple.core.scene import Scene, TabletopScene
from simple.scenes.scene_manager import SceneManager
from simple.utils import resolve_data_path, resolve_res_path

if TYPE_CHECKING:
    from simple.core import Asset

load_dotenv()


class HssdSuite(TabletopScene):
    name: str
    center_offset: list[float]
    center_orientation: list[float]

    def __init__(self, conf) -> None:
        self.uid = f"hssd:{conf['uid']}"
        self.conf = conf
        self.name = conf["name"]
        self.data_dir = f"{conf['data_dir']}/{conf['name']}"
        self.middle()

    def set_table(self, table: Asset) -> None:
        self.table = table

    def set_table2(self, table2: Asset) -> None:
        self.table2 = table2

    def dr(self) -> HssdSuite:
        """Randomize within this scene's configured ranges."""
        self.center_offset = np.random.uniform(
            self.conf["center_offset_limit_up"],
            self.conf["center_offset_limit_down"],
        )
        self.center_orientation = np.random.uniform(
            self.conf["center_orientation_limit_up"],
            self.conf["center_orientation_limit_down"],
        )
        return self

    def middle(self) -> HssdSuite:
        self.center_offset = [
            (upper + lower) / 2.0
            for upper, lower in zip(
                self.conf["center_offset_limit_up"],
                self.conf["center_offset_limit_down"],
                strict=True,
            )
        ]
        self.center_orientation = [
            (upper + lower) / 2.0
            for upper, lower in zip(
                self.conf["center_orientation_limit_up"],
                self.conf["center_orientation_limit_down"],
                strict=True,
            )
        ]
        return self


@SceneManager.register("hssd")
class HssdSceneManager(SceneManager):
    def __init__(self) -> None:
        config_file_path = resolve_res_path("hssd-scenes/config.yaml")
        with open(config_file_path, encoding="utf-8") as stream:
            self.hssd_scenes = yaml.safe_load(stream)

    def sample(self, exclude: list[str] | None = None) -> Scene:
        candidates = self.hssd_scenes
        if exclude:
            excluded = set(exclude)
            candidates = [item for item in candidates if item["uid"] not in excluded]
        if not candidates:
            raise ValueError("no HSSD scenes remain after exclusions")
        return HssdSuite(random.choice(candidates))

    def _require_normalized_scene(
        self, *, scene_uid: str, scene_name: str, scene_dir: str, usd_path: str
    ) -> None:
        if not os.path.isfile(usd_path):
            raise FileNotFoundError(usd_path)
        if os.environ.get("SIMPLE_ASSET_OFFLINE") != "1":
            return

        marker = os.path.join(scene_dir, "NORMALIZED.json")
        with open(marker, "rb") as stream:
            payload = json.loads(stream.read())
        keys = {
            "schema_version",
            "scene_uid",
            "scene_name",
            "source_usd_sha256",
            "normalized_usd_sha256",
            "normalization_policy_version",
        }
        if type(payload) is not dict or set(payload) != keys:
            raise RuntimeError("sealed HSSD normalization marker has wrong schema")
        if (
            payload["schema_version"] != 1
            or payload["normalization_policy_version"] != 1
        ):
            raise RuntimeError("sealed HSSD normalization marker has wrong version")
        if payload["scene_uid"] != scene_uid or payload["scene_name"] != scene_name:
            raise RuntimeError(
                "sealed HSSD normalization marker has wrong scene identity"
            )
        if any(
            type(payload[key]) is not str
            or len(payload[key]) != 64
            or any(ch not in "0123456789abcdef" for ch in payload[key])
            for key in ("source_usd_sha256", "normalized_usd_sha256")
        ):
            raise RuntimeError("sealed HSSD normalization marker has invalid hash")
        with open(usd_path, "rb") as stream:
            normalized_sha256 = hashlib.sha256(stream.read()).hexdigest()
        if payload["normalized_usd_sha256"] != normalized_sha256:
            raise RuntimeError("sealed HSSD normalization marker hash mismatch")

    def load(self, scene_uid: str) -> Scene:
        key = scene_uid.split(":", 1)[1] if ":" in scene_uid else scene_uid
        configs = {scene["uid"]: scene for scene in self.hssd_scenes}
        if key not in configs:
            raise KeyError(f"unknown HSSD scene: {scene_uid}")
        scene_name = configs[key]["name"]
        scene_dir = resolve_data_path(f"scenes/hssd/{scene_name}", auto_download=True)
        usd_path = os.path.join(scene_dir, f"{scene_name}.usd")
        self._require_normalized_scene(
            scene_uid=key,
            scene_name=scene_name,
            scene_dir=scene_dir,
            usd_path=usd_path,
        )
        return HssdSuite(configs[key])
