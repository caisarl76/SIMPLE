#!/usr/bin/env python3
"""Validate one SIMPLE raw dataset and write an immutable evidence report."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

CAMERAS = (
    "observation.rgb_front_stereo_left",
    "observation.rgb_front_stereo_right",
    "observation.rgb_head_stereo_left",
    "observation.rgb_head_stereo_right",
)


def validate(root: Path, expected: int, report: Path) -> dict:
    import numpy as np
    import pyarrow.parquet as pq

    root, errors, started = root.resolve(), [], time.time()
    try:
        info = json.loads((root / "meta/info.json").read_text())
        records = [
            json.loads(x)
            for x in (root / "meta/episodes.jsonl").read_text().splitlines()
            if x.strip()
        ]
    except Exception as exc:
        raise RuntimeError(f"invalid raw metadata: {exc}") from exc
    for required in ("meta/info.json", "meta/episodes.jsonl"):
        if not (root / required).is_file():
            errors.append(f"missing required metadata file {required}")
    if not isinstance(info.get("video_path"), str):
        errors.append("info video_path must be a string")
    if not isinstance(info.get("chunks_size", 1000), int):
        errors.append("info chunks_size must be an integer")
    parquet = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    if len(parquet) != expected:
        errors.append(f"parquet count {len(parquet)} != {expected}")
    if len(records) != expected:
        errors.append(f"episodes metadata count {len(records)} != {expected}")
    if info.get("total_episodes") != len(parquet):
        errors.append("info total_episodes mismatch")
    if info.get("total_frames") != sum(r.get("length", 0) for r in records):
        errors.append("info total_frames mismatch")
    if [r.get("episode_index") for r in records] != list(range(expected)):
        errors.append("metadata indices are not contiguous")
    lengths = []
    for position, path in enumerate(parquet):
        tab = pq.read_table(path)
        n = tab.num_rows
        lengths.append(n)
        try:
            index = int(path.stem.rsplit("_", 1)[1])
        except ValueError:
            index = -1
        if index != position:
            errors.append(f"{path}: parquet indices are not contiguous")
        for col in ("action", "observation.joint_qpos"):
            vals = tab[col].to_pylist() if col in tab.column_names else []
            if col not in tab.column_names or any(
                v is None
                or len(v) != 43
                or not np.isfinite(np.asarray(v, dtype=np.float32)).all()
                for v in vals
            ):
                errors.append(f"{path}: invalid {col}")
        if index >= len(records) or index < 0 or records[index].get("length") != n:
            errors.append(f"{path}: metadata length mismatch")
        for cam in CAMERAS:
            if not isinstance(info.get("video_path"), str):
                continue
            video = root / info["video_path"].format(
                episode_chunk=index // int(info.get("chunks_size", 1000)),
                episode_index=index,
                video_key=cam,
            )
            if not video.exists():
                errors.append(f"missing video {video}")
                continue
            try:
                stream = json.loads(
                    subprocess.check_output(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-select_streams",
                            "v:0",
                            "-show_entries",
                            "stream=width,height,r_frame_rate,nb_frames",
                            "-of",
                            "json",
                            str(video),
                        ],
                        text=True,
                    )
                )["streams"][0]
                num, den = (stream.get("r_frame_rate", "0/1").split("/") + ["1"])[:2]
                if (
                    int(stream.get("width", 0)),
                    int(stream.get("height", 0)),
                    int(num) / int(den),
                    int(stream.get("nb_frames", 0)),
                ) != (640, 360, 50, n):
                    errors.append(f"{video}: geometry/rate/frame mismatch")
            except Exception as exc:
                errors.append(f"{video}: ffprobe failed: {exc}")
    if any(n <= 60 for n in lengths):
        errors.append("episode at or below 60 frames")
    env = [
        r.get("environment_config")
        for r in records
        if r.get("environment_config") is not None
    ]
    canonical = {json.dumps(v, sort_keys=True, separators=(",", ":")) for v in env}
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p != report
    }
    result = {
        "raw_root": str(root),
        "expected_episodes": expected,
        "actual_episodes": len(parquet),
        "total_frames": sum(lengths),
        "retained_frames": sum(max(n - 60, 0) for n in lengths),
        "episode_lengths": lengths,
        "environment_config_count": len(env),
        "environment_config_unique": len(canonical) if env else None,
        "source_sha256": hashes,
        "elapsed_seconds": time.time() - started,
        "ok": not errors,
        "errors": errors,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_root", type=Path)
    ap.add_argument("--expected-episodes", type=int, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args(argv)
    result = validate(args.raw_root, args.expected_episodes, args.report)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
