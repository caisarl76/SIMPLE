import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.certify_psi0_policy_contract import (
    build_policy_contract_payload,
    certify_layout,
    load_bound_episode,
)


def history_fixture():
    history = np.zeros((5, 9), dtype=np.float32)
    history[:, 3:6] = np.arange(15, dtype=np.float32).reshape(5, 3)
    history[:, 6] = 0.74
    return history


def test_certify_layout_accepts_only_corrected_causal_candidate():
    history = history_fixture()
    stored = np.zeros((5, 32), dtype=np.float32)
    stored[:, 28:31] = history[:, 3:6][:, ::-1]
    stored[:, 31] = 0.74
    result = certify_layout(stored, history)
    assert result == {
        "layout": "g1_simple_32_rpyh_v2",
        "corrected_match": True,
        "legacy_row_reversed_match": False,
        "off_by_one_match": False,
    }


@pytest.mark.parametrize("candidate", ["legacy", "off_by_one"])
def test_certify_layout_rejects_noncausal_or_shifted_data(candidate):
    history = history_fixture()
    stored = np.zeros((5, 32), dtype=np.float32)
    source = (
        history[:, 3:6][::-1]
        if candidate == "legacy"
        else history[[0, 0, 1, 2, 3], 3:6][:, ::-1]
    )
    stored[:, 28:31] = source
    stored[:, 31] = 0.74
    with pytest.raises(ValueError, match="not uniquely certified"):
        certify_layout(stored, history)


def write_bound_episode(tmp_path, *, recorded_raw_hash=None, converter_commit="2" * 40):
    raw_path = tmp_path / "episode_000007.parquet"
    processed_path = tmp_path / "episode_000003.parquet"
    episodes_path = tmp_path / "episodes.jsonl"
    cmd = np.zeros((5, 9), dtype=np.float32)
    cmd[:, 3:6] = np.arange(15, dtype=np.float32).reshape(5, 3)
    cmd[:, 6] = 0.74
    pq.write_table(pa.table({"observation.amo_policy_command": cmd.tolist()}), raw_path)
    initial = np.array([0, 0, 0, 0, 0, 0, 0.74, 0.74, 0.74], np.float32)
    history = np.concatenate([initial[None], cmd[:-1]], axis=0)
    states = np.zeros((5, 32), dtype=np.float32)
    states[:, 28:31] = history[:, 3:6][:, ::-1]
    states[:, 31] = history[:, 6]
    pq.write_table(
        pa.table({"states": states.tolist(), "episode_index": [3] * 5}),
        processed_path,
    )
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    record = {
        "episode_index": 3,
        "length": 5,
        "conversion_provenance": {
            "source_episode_index": 7,
            "source_parquet_sha256": recorded_raw_hash or raw_hash,
            "skip": 0,
            "downsample": 1,
            "converter_commit": converter_commit,
        },
    }
    episodes_path.write_text(json.dumps(record) + "\n")
    return raw_path, processed_path, episodes_path, states, history


def test_load_bound_episode_proves_same_raw_and_processed_episode(tmp_path):
    raw, processed, episodes, expected_states, expected_history = write_bound_episode(
        tmp_path
    )
    result = load_bound_episode(raw, processed, episodes)
    np.testing.assert_array_equal(result.stored_state, expected_states)
    np.testing.assert_array_equal(result.history_cmd, expected_history)
    assert result.source_episode_index == 7
    assert result.processed_episode_index == 3
    assert result.converter_commit == "2" * 40
    assert result.raw_episode_sha256 == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert (
        result.processed_episode_sha256
        == hashlib.sha256(processed.read_bytes()).hexdigest()
    )


def test_load_bound_episode_rejects_cross_episode_or_unattested_converter(tmp_path):
    raw, processed, episodes, *_ = write_bound_episode(
        tmp_path, recorded_raw_hash="f" * 64
    )
    with pytest.raises(ValueError, match="raw episode hash mismatch"):
        load_bound_episode(raw, processed, episodes)
    raw, processed, episodes, *_ = write_bound_episode(
        tmp_path, converter_commit="not-a-commit"
    )
    with pytest.raises(ValueError, match="converter commit"):
        load_bound_episode(raw, processed, episodes)


def test_policy_contract_records_bound_episode_and_converter_commit(tmp_path):
    raw, processed, episodes, *_ = write_bound_episode(tmp_path)
    bound = load_bound_episode(raw, processed, episodes)
    payload = build_policy_contract_payload(
        bound_episode=bound,
        checkpoint_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        server_commit="c" * 40,
        prediction_horizon=30,
        execution_horizon=24,
        rtc_delay_steps=6,
        rtc_training_max_delay=7,
    )
    assert payload["source_episode_index"] == 7
    assert payload["processed_episode_index"] == 3
    assert payload["raw_episode_sha256"] == bound.raw_episode_sha256
    assert payload["processed_episode_sha256"] == bound.processed_episode_sha256
    assert payload["converter_commit"] == "2" * 40
    assert payload["test_only"] is False
