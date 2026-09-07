import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts import validate_simple_raw


def _raw(root: Path, *, finite=True):
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    frames = 61
    values = np.ones((frames, 43), dtype=np.float32)
    if not finite:
        values[0, 0] = np.nan
    arr = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), 43)
    pq.write_table(
        pa.Table.from_arrays([arr, arr], names=["action", "observation.joint_qpos"]),
        root / "data/chunk-000/episode_000000.parquet",
    )
    info = {
        "total_episodes": 1,
        "total_frames": frames,
        "chunks_size": 1000,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text(
        json.dumps(
            {"episode_index": 0, "length": frames, "environment_config": {"seed": 1}}
        )
        + "\n"
    )


def test_report_is_exclusive_and_canonicalizes_environment(tmp_path, monkeypatch):
    _raw(tmp_path)
    for camera in validate_simple_raw.CAMERAS:
        path = tmp_path / f"videos/chunk-000/{camera}/episode_000000.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(
        validate_simple_raw.subprocess,
        "check_output",
        lambda *a, **k: json.dumps(
            {
                "streams": [
                    {
                        "width": 640,
                        "height": 360,
                        "r_frame_rate": "50/1",
                        "nb_frames": 61,
                    }
                ]
            }
        ),
    )
    report = tmp_path / "report.json"
    result = validate_simple_raw.validate(tmp_path, 1, report)
    assert result["ok"] and result["environment_config_unique"] == 1
    try:
        validate_simple_raw.validate(tmp_path, 1, report)
    except FileExistsError:
        pass
    else:
        raise AssertionError("report must be exclusive")


def test_nonfinite_vector_fails(tmp_path, monkeypatch):
    _raw(tmp_path, finite=False)
    monkeypatch.setattr(
        validate_simple_raw.subprocess,
        "check_output",
        lambda *a, **k: json.dumps(
            {
                "streams": [
                    {
                        "width": 640,
                        "height": 360,
                        "r_frame_rate": "50/1",
                        "nb_frames": 61,
                    }
                ]
            }
        ),
    )
    result = validate_simple_raw.validate(tmp_path, 1, tmp_path / "report.json")
    assert not result["ok"] and any(
        "invalid action" in error for error in result["errors"]
    )
