"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from abc import ABC
from typing import Any

import importlib.resources as res
import numpy as np
from importlib.resources import as_file

from simple.utils import get_data_dir as shared_get_data_dir
from simple.utils import resolve_data_path as shared_resolve_data_path


class Simulator(ABC):
    def update_layout(self) -> None: ...

    def set_states(self, states: dict[str, Any]) -> None: ...

    def get_states(self) -> dict[str, Any]: ...

    def step(self) -> dict[str, np.ndarray] | None: ...

    def render(self, *args, **kwargs) -> dict[str, np.ndarray]: ...

    def resolve_res_path(self, rel_path=None) -> str:
        res_dir = self.get_res_dir()
        if not rel_path:
            return res_dir
        with as_file(res_dir / rel_path) as res_path:
            if not res_path.exists():
                raise FileNotFoundError(res_path)
        return str(res_path)

    def get_res_dir(self) -> str:
        return res.files("simple") / "resources"  # type: ignore

    def get_data_dir(self) -> str:
        return shared_get_data_dir()

    def resolve_data_path(self, rel_path=None, create_if_not_exist=False) -> str:
        return shared_resolve_data_path(
            rel_path,
            create_if_not_exist=create_if_not_exist,
            auto_download=create_if_not_exist,
        )
