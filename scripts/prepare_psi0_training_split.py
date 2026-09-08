#!/usr/bin/env python3
"""Copy a corrected raw batch into disjoint train/validation episode trees."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


EP = re.compile(r"episode_(\d+)")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(batch_root, output_root):
    batch_root, output_root = Path(batch_root).resolve(), Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    raw = batch_root / "raw"
    levels = sorted(raw.glob("dr*/simple/*/level-*"))
    if not levels:
        raise ValueError(f"no raw level directories under {raw}")
    manifest = batch_root / "result.json"
    if not manifest.is_file():
        raise ValueError(f"missing source manifest: {manifest}")
    result_hash = sha256(manifest)
    result = json.loads(manifest.read_text())
    mapping, plans = {}, []
    for level in levels:
        episodes_path = level / "meta" / "episodes.jsonl"
        rows = [
            json.loads(line)
            for line in episodes_path.read_text().splitlines()
            if line.strip()
        ]
        ids = [int(row["episode_index"]) for row in rows]
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError(f"need at least two episodes: {level}")
        parquet_ids = {
            int(m.group(1))
            for p in (level / "data").rglob("*.parquet")
            if (m := EP.search(p.name))
        }
        video_ids = {
            int(m.group(1))
            for p in (level / "videos").rglob("*")
            if p.is_file() and (m := EP.search(p.name))
        }
        if parquet_ids != set(ids) or video_ids != set(ids):
            raise ValueError(f"episode files do not cover metadata exactly: {level}")
        val_id = max(ids)
        selected = {"train": set(ids) - {val_id}, "val": {val_id}}
        rel_level = level.relative_to(raw)
        mapping[str(rel_level)] = {k: sorted(v) for k, v in selected.items()}
        for split, wanted in selected.items():
            for source in level.rglob("*"):
                if not source.is_file():
                    continue
                match = EP.search(source.name)
                if match and int(match.group(1)) not in wanted:
                    continue
                plans.append(
                    (
                        source,
                        output_root
                        / "raw"
                        / split
                        / rel_level
                        / source.relative_to(level),
                        wanted,
                    )
                )
    output_root.mkdir(parents=True)
    copied = []
    for source, target, wanted in plans:
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.parent.name == "meta" and source.name in {
            "episodes.jsonl",
            "episodes_stats.jsonl",
        }:
            lines = source.read_text().splitlines()
            lines = [
                line
                for line in lines
                if not line.strip() or int(json.loads(line)["episode_index"]) in wanted
            ]
            target.write_text("\n".join(lines) + ("\n" if lines else ""))
            transformed = True
        elif source.name == "info.json":
            info = json.loads(source.read_text())
            if info:
                info["total_episodes"] = len(wanted)
                episode_rows = [
                    json.loads(line)
                    for line in (source.parent / "episodes.jsonl")
                    .read_text()
                    .splitlines()
                    if line.strip()
                ]
                info["total_frames"] = sum(
                    row["length"]
                    for row in episode_rows
                    if row["episode_index"] in wanted
                )
                info["total_videos"] = sum(
                    1
                    for p in source.parent.parent.joinpath("videos").rglob("*")
                    if p.is_file()
                    and (m := EP.search(p.name))
                    and int(m.group(1)) in wanted
                )
                info.pop("splits", None)
            target.write_text(json.dumps(info, indent=2) + "\n")
            transformed = True
        else:
            shutil.copy2(source, target)
            transformed = False
        source_hash, destination_hash = sha256(source), sha256(target)
        if not transformed and source_hash != destination_hash:
            raise ValueError(f"copy hash mismatch: {source}")
        copied.append(
            {
                "source": str(source),
                "destination": str(target),
                "source_sha256": source_hash,
                "destination_sha256": destination_hash,
                "transformed": transformed,
            }
        )
    (output_root / "split.json").write_text(
        json.dumps(
            {
                "source": str(batch_root),
                "source_manifest": result,
                "source_manifest_sha256": result_hash,
                "episodes": mapping,
                "copied": copied,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.batch_root, args.output_root)


if __name__ == "__main__":
    main()
