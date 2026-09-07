import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest

import simple.dr.material as material_module
from simple.dr.material import MaterialDR, MaterialDRCfg
from simple.core.actor import ObjectActor
from simple.dr.manager import TabletopGraspDRManager


class _Actor:
    def __init__(self):
        self.material = None
        self.shaders = None

    def set_material(self, value):
        self.material = value

    def set_shaders(self, value):
        self.shaders = value


class _Layout:
    def __init__(self):
        self.table = _Actor()
        self.robot = _Actor()
        self.object = ObjectActor.__new__(ObjectActor)
        self.object.material = None
        self.actors = {"table": self.table, "robot": self.robot, "object": self.object}


def _patch_material_resources(monkeypatch):
    table = [
        {"path": "table-a", "name": "table-a"},
        {"path": "table-b", "name": "table-b"},
    ]
    ground = [
        {"path": "ground-a", "name": "ground-a"},
        {"path": "ground-b", "name": "ground-b"},
    ]

    class _LoadedSplit:
        def item(self):
            return {"train": {"table": table, "ground": ground}}

    monkeypatch.setattr(
        material_module.np, "load", lambda *args, **kwargs: _LoadedSplit()
    )
    monkeypatch.setattr(
        material_module, "resolve_data_path", lambda path, auto_download=False: path
    )
    monkeypatch.setattr(material_module, "resolve_res_path", lambda path: path)


def _draw(dr, layout):
    dr._inner_state = None
    return copy.deepcopy(dr("train", layout))


def test_mutated_config_controls_fixed_draws_and_explicit_table_does_not_randomize_ground(
    monkeypatch,
):
    _patch_material_resources(monkeypatch)
    cfg = MaterialDRCfg(
        material_mode="rand_all",
        table_material={"path": "explicit", "name": "explicit"},
    )
    dr = MaterialDR(cfg)
    cfg.material_mode = "fixed"

    first = _draw(dr, _Layout())
    second = _draw(dr, _Layout())

    assert first == second
    assert first["table_material"] == {"path": "explicit", "name": "explicit"}
    assert first["ground_material"] == {
        "path": "vMaterials_2/Concrete/Concrete_Floor_Damage.mdl",
        "name": "Concrete_Floor_Damage",
    }
    assert first["robot_shader_params"] == {
        "reflection_roughness_constant": 0.5,
        "metallic_constant": 0.0,
        "specular_level": 0.0,
    }
    assert first["object_shader_params"] == [first["robot_shader_params"]]


def test_random_mode_varies_with_seed(monkeypatch):
    _patch_material_resources(monkeypatch)
    cfg = MaterialDRCfg(material_mode="rand_all")
    dr = MaterialDR(cfg)

    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(1)
        np.random.seed(1)
        first = _draw(dr, _Layout())
        random.seed(2)
        np.random.seed(2)
        second = _draw(dr, _Layout())
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

    assert first != second
    assert (
        first["table_material"] != second["table_material"]
        or first["robot_shader_params"] != second["robot_shader_params"]
    )


def test_loaded_state_replays_without_randomization(monkeypatch):
    _patch_material_resources(monkeypatch)
    dr = MaterialDR(MaterialDRCfg(material_mode="rand_all"))
    state = _draw(dr, _Layout())
    dr.load_state_dict(state)
    layout = _Layout()

    replay = dr("train", layout)

    assert replay == state
    assert layout.table.material == state["table_material"]
    assert layout.robot.shaders == state["robot_shader_params"]
    assert layout.object.material == state["object_shader_params"][0]


@pytest.mark.parametrize("level", [1, 2])
def test_manager_level_switches_material_to_repeatable_fixed_mode(monkeypatch, level):
    _patch_material_resources(monkeypatch)
    manager = TabletopGraspDRManager.__new__(TabletopGraspDRManager)
    manager.randomizers = {
        "material": MaterialDR(MaterialDRCfg(material_mode="rand_all")),
        "lighting": SimpleNamespace(cfg=SimpleNamespace(light_mode="rand_all")),
        "scene": SimpleNamespace(
            cfg=SimpleNamespace(
                scene_mode="rand_all", room_choices=["scene0", "scene1"]
            )
        ),
        "distractors": SimpleNamespace(cfg=SimpleNamespace(number_of_distractors=3)),
    }

    manager.set_level(level)
    material_dr = manager.randomizers["material"]
    first = _draw(material_dr, _Layout())
    second = _draw(material_dr, _Layout())

    assert material_dr.cfg.material_mode == "fixed"
    assert first == second
