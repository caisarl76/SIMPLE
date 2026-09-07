import os
from pathlib import Path


DEFAULT_OUTPUT_ROOT = "/mnt/data/jihun/datasets/SIMPLE"


def output_path(name: str) -> str:
    return str(
        Path(os.environ.get("SIMPLE_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)).expanduser()
        / name
    )
