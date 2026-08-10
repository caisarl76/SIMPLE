#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq


EPISODE_NAME = re.compile(r"episode_(\d{6})\.parquet")
INITIAL_COMMAND = np.asarray(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.74, 0.74, 0.74],
    np.float32,
)


@dataclass(frozen=True)
class BoundEpisode:
    stored_state: np.ndarray
    history_cmd: np.ndarray
    source_episode_index: int
    processed_episode_index: int
    raw_episode_sha256: str
    processed_episode_sha256: str
    converter_commit: str


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_episode_index(name):
    match = EPISODE_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid episode parquet filename: {name}")
    return int(match.group(1))


def _metadata_record(path, processed_index):
    records = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid episodes metadata line {line_number}") from error
        if type(record) is not dict:
            raise ValueError("episode metadata record must be an object")
        records.append(record)
    matches = [
        record for record in records if record.get("episode_index") == processed_index
    ]
    if len(matches) != 1:
        raise ValueError("expected exactly one processed episode metadata record")
    return matches[0]


def load_bound_episode(raw, processed, episodes_jsonl):
    raw = Path(raw)
    processed = Path(processed)
    episodes_jsonl = Path(episodes_jsonl)
    source_index = parse_episode_index(raw.name)
    processed_index = parse_episode_index(processed.name)
    record = _metadata_record(episodes_jsonl, processed_index)
    provenance = record.get("conversion_provenance")
    required_provenance = {
        "source_episode_index",
        "source_parquet_sha256",
        "skip",
        "downsample",
        "converter_commit",
    }
    if type(provenance) is not dict or set(provenance) != required_provenance:
        raise ValueError("conversion provenance keys")
    if (
        type(provenance["source_episode_index"]) is not int
        or provenance["source_episode_index"] != source_index
    ):
        raise ValueError("source episode index mismatch")
    raw_hash = sha256_file(raw)
    if provenance["source_parquet_sha256"] != raw_hash:
        raise ValueError("raw episode hash mismatch")
    converter_commit = provenance["converter_commit"]
    if (
        type(converter_commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}", converter_commit) is None
    ):
        raise ValueError("invalid converter commit")
    skip = provenance["skip"]
    downsample = provenance["downsample"]
    if type(skip) is not int or skip < 0:
        raise ValueError("conversion skip must be a nonnegative integer")
    if type(downsample) is not int or downsample <= 0:
        raise ValueError("conversion downsample must be a positive integer")

    raw_table = pq.read_table(raw, columns=["observation.amo_policy_command"])
    command = np.asarray(
        raw_table["observation.amo_policy_command"].to_pylist(),
        dtype=np.float32,
    )
    if command.ndim != 2 or command.shape[1] != 9:
        raise ValueError("raw command must have shape (T,9)")
    if not np.isfinite(command).all() or skip >= len(command):
        raise ValueError("raw command is non-finite or skip removes all frames")
    full_history = np.concatenate([INITIAL_COMMAND[None], command[:-1]], axis=0)
    history = np.ascontiguousarray(full_history[skip::downsample])

    processed_table = pq.read_table(processed, columns=["states", "episode_index"])
    processed_indices = set(processed_table["episode_index"].to_pylist())
    if processed_indices != {processed_index}:
        raise ValueError("processed episode_index does not match filename")
    stored = np.asarray(processed_table["states"].to_pylist(), dtype=np.float32)
    if stored.ndim != 2 or stored.shape[1] < 32:
        raise ValueError("processed states must have shape (T,>=32)")
    if not np.isfinite(stored).all():
        raise ValueError("processed states must be finite")
    if type(record.get("length")) is not int:
        raise ValueError("processed metadata length must be an integer")
    if len(stored) != len(history) or len(stored) != record["length"]:
        raise ValueError("raw history and processed frame counts differ")
    return BoundEpisode(
        stored_state=np.ascontiguousarray(stored),
        history_cmd=history,
        source_episode_index=source_index,
        processed_episode_index=processed_index,
        raw_episode_sha256=raw_hash,
        processed_episode_sha256=sha256_file(processed),
        converter_commit=converter_commit,
    )


def certify_layout(
    stored_state: np.ndarray, history_cmd: np.ndarray
) -> dict[str, object]:
    stored = np.asarray(stored_state, dtype=np.float32)
    history = np.asarray(history_cmd, dtype=np.float32)
    if stored.ndim != 2 or stored.shape[1] < 32:
        raise ValueError(f"expected stored state shape (T,>=32), got {stored.shape}")
    if history.shape != (stored.shape[0], 9):
        raise ValueError(
            f"expected history shape {(stored.shape[0], 9)}, got {history.shape}"
        )
    corrected = history[:, 3:6][:, ::-1]
    legacy = history[:, 3:6][::-1]
    previous = corrected[np.maximum(np.arange(len(corrected)) - 1, 0)]
    corrected_match = bool(np.array_equal(stored[:, 28:31], corrected))
    legacy_match = bool(np.array_equal(stored[:, 28:31], legacy))
    off_by_one_match = bool(np.array_equal(stored[:, 28:31], previous))
    height_match = bool(
        np.array_equal(stored[:, 31], np.full(len(stored), 0.74, np.float32))
    )
    if not height_match or not corrected_match or legacy_match or off_by_one_match:
        raise ValueError(
            "processed episode is not uniquely certified for g1_simple_32_rpyh_v2"
        )
    return {
        "layout": "g1_simple_32_rpyh_v2",
        "corrected_match": True,
        "legacy_row_reversed_match": False,
        "off_by_one_match": False,
    }


def validate_policy_contract_payload(payload):
    types = {
        "schema": str,
        "test_only": bool,
        "checkpoint_sha256": str,
        "dataset_manifest_sha256": str,
        "raw_episode_sha256": str,
        "processed_episode_sha256": str,
        "source_episode_index": int,
        "processed_episode_index": int,
        "converter_commit": str,
        "server_commit": str,
        "converter_layout": str,
        "observation_dim": int,
        "action_dim": int,
        "action_frequency_hz": int,
        "prediction_horizon": int,
        "execution_horizon": int,
        "rtc_delay_steps": int,
        "rtc_training_max_delay": int,
        "rtc_enabled": bool,
        "rtc_endpoint": str,
        "request_semantics": str,
        "response_semantics": str,
        "image_key": str,
        "camera_color_order": str,
    }
    if type(payload) is not dict or set(payload) != set(types):
        raise TypeError("policy contract keys do not exactly match v2 schema")
    for key, expected in types.items():
        if type(payload[key]) is not expected:
            raise TypeError(f"policy contract field {key} must be {expected.__name__}")
    for key in (
        "checkpoint_sha256",
        "dataset_manifest_sha256",
        "raw_episode_sha256",
        "processed_episode_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", payload[key]) is None:
            raise ValueError(key)
    for key in ("converter_commit", "server_commit"):
        if re.fullmatch(r"[0-9a-f]{40}", payload[key]) is None:
            raise ValueError(key)
    d = payload["rtc_delay_steps"]
    s = payload["execution_horizon"]
    p = payload["prediction_horizon"]
    if not (2 <= d <= s and d + s <= p and d < payload["rtc_training_max_delay"]):
        raise ValueError("invalid RTC horizon/delay contract")


def build_policy_contract_payload(
    *,
    bound_episode,
    checkpoint_sha256,
    dataset_manifest_sha256,
    server_commit,
    prediction_horizon,
    execution_horizon,
    rtc_delay_steps,
    rtc_training_max_delay,
):
    certification = certify_layout(
        bound_episode.stored_state, bound_episode.history_cmd
    )
    payload = {
        "schema": "simple.psi0.policy-contract.v2",
        "test_only": False,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "raw_episode_sha256": bound_episode.raw_episode_sha256,
        "processed_episode_sha256": bound_episode.processed_episode_sha256,
        "source_episode_index": bound_episode.source_episode_index,
        "processed_episode_index": bound_episode.processed_episode_index,
        "converter_commit": bound_episode.converter_commit,
        "server_commit": server_commit,
        "converter_layout": certification["layout"],
        "observation_dim": 32,
        "action_dim": 36,
        "action_frequency_hz": 50,
        "prediction_horizon": prediction_horizon,
        "execution_horizon": execution_horizon,
        "rtc_delay_steps": rtc_delay_steps,
        "rtc_training_max_delay": rtc_training_max_delay,
        "rtc_enabled": True,
        "rtc_endpoint": "/act-rtc-v1",
        "request_semantics": "exact-post-slew-committed-prefix",
        "response_semantics": "denormalized-executable-suffix",
        "image_key": "rgb_head_stereo_left",
        "camera_color_order": "rgb",
    }
    validate_policy_contract_payload(payload)
    return payload


def atomic_write_policy_contract(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--raw-episode-parquet", required=True)
    parser.add_argument("--processed-episode-parquet", required=True)
    parser.add_argument("--processed-episodes-jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--server-commit", required=True)
    parser.add_argument("--prediction-horizon", required=True, type=int)
    parser.add_argument("--execution-horizon", required=True, type=int)
    parser.add_argument("--rtc-delay-steps", required=True, type=int)
    parser.add_argument("--rtc-training-max-delay", required=True, type=int)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    bound = load_bound_episode(
        args.raw_episode_parquet,
        args.processed_episode_parquet,
        args.processed_episodes_jsonl,
    )
    payload = build_policy_contract_payload(
        bound_episode=bound,
        checkpoint_sha256=sha256_file(args.checkpoint),
        dataset_manifest_sha256=sha256_file(args.dataset_manifest),
        server_commit=args.server_commit,
        prediction_horizon=args.prediction_horizon,
        execution_horizon=args.execution_horizon,
        rtc_delay_steps=args.rtc_delay_steps,
        rtc_training_max_delay=args.rtc_training_max_delay,
    )
    atomic_write_policy_contract(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
