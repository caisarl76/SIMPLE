import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _fixed_size_float32(values: np.ndarray, width: int) -> pa.FixedSizeListArray:
    flattened = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flattened, width)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_source_episode(
    root: Path,
    *,
    episode_index: int = 0,
    frames: int = 8,
    fps: int = 4,
    size: str = "64x48",
    video_codec: str = "libx264",
    task_index: int = 0,
    task_text: str = "test task",
) -> Path:
    data_dir = root / "data" / "chunk-000"
    video_dir = root / "videos" / "chunk-000" / "observation.rgb_head_stereo_left"
    meta_dir = root / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    episode_name = f"episode_{episode_index:06d}"
    parquet_path = data_dir / f"{episode_name}.parquet"
    qpos = np.arange(frames * 43, dtype=np.float32).reshape(frames, 43)
    command = np.arange(frames * 9, dtype=np.float32).reshape(frames, 9)
    action = (qpos + np.float32(0.5)).astype(np.float32)
    table = pa.Table.from_arrays(
        [
            _fixed_size_float32(qpos, 43),
            _fixed_size_float32(command, 9),
            pa.array(np.linspace(0.0, 0.5, frames, dtype=np.float32)),
            pa.array(np.zeros(frames, dtype=np.float32)),
            _fixed_size_float32(action, 43),
            pa.array(np.full(frames, task_index, dtype=np.int64)),
        ],
        names=[
            "observation.joint_qpos",
            "observation.amo_policy_command",
            "observation.amo_policy_target_yaw",
            "observation.amo_policy_turning_flag",
            "action",
            "task_index",
        ],
    )
    pq.write_table(table, parquet_path)

    video_path = video_dir / f"{episode_name}.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={size}:rate={fps}",
            "-frames:v",
            str(frames),
            "-c:v",
            video_codec,
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
    )

    (meta_dir / "info.json").write_text(
        json.dumps({"fps": fps}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(
        meta_dir / "tasks.jsonl",
        [{"task_index": task_index, "task": task_text}],
    )
    _write_jsonl(
        meta_dir / "episodes.jsonl",
        [
            {
                "episode_index": episode_index,
                "tasks": [task_text],
                "length": frames,
                "environment_config": "{}",
            }
        ],
    )
    return root
