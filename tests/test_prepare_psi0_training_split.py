import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from prepare_psi0_training_split import prepare


def test_split_copies_disjoint_episodes_and_rejects_existing_output(tmp_path):
    batch = tmp_path / "batch"
    level = batch / "raw/dr0/simple/task/level-0"
    (level / "meta").mkdir(parents=True)
    for name in ("info.json", "tasks.jsonl"):
        (level / "meta" / name).write_text("{}\n")
    (level / "meta/info.json").write_text('{"total_frames":30,"total_episodes":3}')
    rows = [
        {"episode_index": i, "length": 10, "environment_config": f"env-{i}"}
        for i in range(3)
    ]
    (level / "meta/episodes.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    (level / "meta/episodes_stats.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    for i in range(3):
        for directory, suffix in (
            ("data/chunk-000", ".parquet"),
            ("videos/cam", ".mp4"),
        ):
            path = level / directory / f"episode_{i:06d}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"episode-{i}".encode())
    (batch / "result.json").write_text('{"ok":true}\n')
    output = tmp_path / "out"
    prepare(batch, output)
    assert {
        p.name
        for p in (output / "raw/train/dr0/simple/task/level-0/data/chunk-000").iterdir()
    } == {"episode_000000.parquet", "episode_000001.parquet"}
    assert (
        output / "raw/val/dr0/simple/task/level-0/data/chunk-000/episode_000002.parquet"
    ).read_bytes() == b"episode-2"
    manifest = json.loads((output / "split.json").read_text())
    assert manifest["episodes"]["dr0/simple/task/level-0"] == {
        "train": [0, 1],
        "val": [2],
    }
    assert all(
        row["source_sha256"] == row["destination_sha256"]
        for row in manifest["copied"]
        if not row["transformed"]
    )
    assert (
        json.loads(
            (output / "raw/train/dr0/simple/task/level-0/meta/info.json").read_text()
        )["total_frames"]
        == 20
    )
    with pytest.raises(FileExistsError):
        prepare(batch, output)
