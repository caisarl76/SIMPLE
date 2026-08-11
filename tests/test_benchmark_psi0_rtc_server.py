import json

import numpy as np
import pytest
import requests

from scripts.benchmark_psi0_rtc_server import (
    FAILURE_KEYS,
    BenchmarkReport,
    BenchmarkRequestError,
    BenchmarkSample,
    BenchmarkTransportResponse,
    HttpBenchmarkTransport,
    benchmark_server,
    build_parser,
    classify_response,
    load_representative_samples,
    main,
    sha256_file,
    write_certification_bundle,
)
from simple.baselines.client import numpy_serialize
from simple.deploy.psi0_simple_bridge import PolicyContract
from tests.psi0_bridge_testkit import make_joint_contract, policy_payload


class FakeBenchmarkTransport:
    def __init__(self, measured_latencies, failure_at=None, failure_kind=None):
        self.measured_latencies = list(measured_latencies)
        self.failure_at = failure_at
        self.failure_kind = failure_kind
        self.calls = []
        self.contract_calls = 0

    def get_contract(self):
        self.contract_calls += 1
        return live_policy_payload()

    def query(self, request_index, sample, history):
        self.calls.append((request_index, sample, history))
        measured_index = request_index - 10
        if measured_index == self.failure_at:
            raise BenchmarkRequestError(self.failure_kind)
        latency = (
            0.02 if request_index < 10 else self.measured_latencies[measured_index]
        )
        return valid_benchmark_response(history, latency_s=latency)


def live_policy_payload(**updates):
    payload = policy_payload(
        test_only=False,
        prediction_horizon=30,
        execution_horizon=24,
        rtc_delay_steps=6,
        rtc_training_max_delay=7,
    )
    payload.update(updates)
    return payload


def representative_samples(count):
    return tuple(
        BenchmarkSample(
            image=np.full((8, 8, 3), index % 256, np.uint8),
            state=np.full((1, 32), index / 1000.0, np.float32),
            instruction=f"representative instruction {index}",
        )
        for index in range(count)
    )


def valid_benchmark_response(history, latency_s):
    first_tick = history["observation_tick"] + history["rtc_delay_steps"]
    action = np.zeros((24, 36), np.float32)
    action[:, 31] = 0.5
    return BenchmarkTransportResponse(
        action=action,
        metadata={
            "session_id": history["session_id"],
            "request_seq": history["request_seq"],
            "observation_tick": history["observation_tick"],
            "prediction_horizon": 30,
            "execution_horizon": 24,
            "rtc_delay_steps": 6,
            "first_action_tick": first_tick,
        },
        latency_s=latency_s,
    )


def test_benchmark_discards_ten_warmups_and_certifies_one_hundred_requests():
    samples = representative_samples(100)
    transport = FakeBenchmarkTransport([0.05] * 99 + [0.099])
    report = benchmark_server(
        transport=transport,
        samples=samples,
        contract=PolicyContract.from_dict(live_policy_payload()),
        joint_contract=make_joint_contract(),
        warmup_requests=10,
        measured_requests=100,
    )
    assert len(transport.calls) == 110
    assert transport.contract_calls == 1
    assert report == BenchmarkReport(
        warmup_requests=10,
        measured_requests=100,
        successes=100,
        failures={key: 0 for key in FAILURE_KEYS},
        p99_latency_s=0.099,
        latency_limit_s=0.10,
        certified=True,
    )
    assert [call[0] for call in transport.calls] == list(range(110))
    assert transport.calls[0][2]["reset"] is True
    assert all("reset" not in history for _, _, history in transport.calls[1:])
    assert [sample.instruction for _, sample, _ in transport.calls[10:]] == [
        sample.instruction for sample in (*samples[10:], *samples[:10])
    ]
    for _, _, history in transport.calls:
        assert history["committed_actions"].shape == (6, 36)
        assert history["committed_actions"].dtype == np.float32
        assert np.isfinite(history["committed_actions"]).all()


def test_p99_uses_higher_method_and_point_101_fails_gate():
    report = benchmark_server(
        transport=FakeBenchmarkTransport([0.05] * 99 + [0.101]),
        samples=representative_samples(100),
        contract=PolicyContract.from_dict(live_policy_payload()),
        joint_contract=make_joint_contract(),
        warmup_requests=10,
        measured_requests=100,
    )
    assert report.p99_latency_s == 0.101
    assert report.latency_limit_s == 0.10
    assert report.certified is False


@pytest.mark.parametrize("failure_kind", FAILURE_KEYS)
def test_every_failure_class_is_counted_and_prevents_certification(failure_kind):
    report = benchmark_server(
        transport=FakeBenchmarkTransport(
            [0.05] * 100,
            failure_at=17,
            failure_kind=failure_kind,
        ),
        samples=representative_samples(100),
        contract=PolicyContract.from_dict(live_policy_payload()),
        joint_contract=make_joint_contract(),
        warmup_requests=10,
        measured_requests=100,
    )
    assert report.successes == 99
    assert report.failures[failure_kind] == 1
    assert sum(report.failures.values()) == 1
    assert report.certified is False


def test_warmup_failure_aborts_immediately():
    transport = FakeBenchmarkTransport([0.05] * 100)
    transport.failure_at = -7
    transport.failure_kind = "timeout"
    with pytest.raises(RuntimeError, match="warmup request 3 failed: timeout"):
        benchmark_server(
            transport=transport,
            samples=representative_samples(100),
            contract=PolicyContract.from_dict(live_policy_payload()),
            joint_contract=make_joint_contract(),
        )
    assert len(transport.calls) == 4


def test_benchmark_requires_exact_counts_and_matching_server_contract():
    contract = PolicyContract.from_dict(live_policy_payload())
    with pytest.raises(ValueError, match="10 warmups and 100 measured"):
        benchmark_server(
            transport=FakeBenchmarkTransport([0.05] * 100),
            samples=representative_samples(100),
            contract=contract,
            joint_contract=make_joint_contract(),
            warmup_requests=9,
        )
    transport = FakeBenchmarkTransport([0.05] * 100)
    transport.get_contract = lambda: live_policy_payload(server_commit="1" * 40)
    with pytest.raises(ValueError, match="differs from local"):
        benchmark_server(
            transport=transport,
            samples=representative_samples(100),
            contract=contract,
            joint_contract=make_joint_contract(),
        )


def write_representative_npz(path, samples):
    np.savez(
        path,
        images=np.stack([sample.image for sample in samples]),
        states=np.stack([sample.state for sample in samples]),
        instructions=np.asarray([sample.instruction for sample in samples]),
    )


def test_representative_loader_accepts_only_exact_contiguous_finite_schema(tmp_path):
    path = tmp_path / "samples.npz"
    expected = representative_samples(100)
    write_representative_npz(path, expected)
    loaded = load_representative_samples(path)
    assert len(loaded) == 100
    assert np.array_equal(loaded[17].image, expected[17].image)
    assert np.array_equal(loaded[17].state, expected[17].state)
    assert loaded[17].instruction == expected[17].instruction


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"extra": np.zeros(100)}, "keys"),
        ({"images": np.zeros((100, 8, 8, 3), np.float32)}, "images"),
        ({"states": np.zeros((100, 32), np.float32)}, "states"),
        (
            {"states": np.full((100, 1, 32), np.nan, np.float32)},
            "finite",
        ),
        ({"instructions": np.arange(100)}, "instructions"),
        (
            {
                "images": np.zeros((99, 8, 8, 3), np.uint8),
                "states": np.zeros((99, 1, 32), np.float32),
                "instructions": np.asarray(["x"] * 99),
            },
            "at least 100",
        ),
    ],
)
def test_representative_loader_rejects_schema_mutations(tmp_path, updates, message):
    values = {
        "images": np.zeros((100, 8, 8, 3), np.uint8),
        "states": np.zeros((100, 1, 32), np.float32),
        "instructions": np.asarray(["x"] * 100),
    }
    values.update(updates)
    path = tmp_path / "samples.npz"
    np.savez(path, **values)
    with pytest.raises(ValueError, match=message):
        load_representative_samples(path)


@pytest.mark.parametrize(
    ("mutation", "failure_kind"),
    [
        ("shape", "shape"),
        ("dtype", "shape"),
        ("metadata_keys", "metadata"),
        ("metadata_value", "metadata"),
        ("bounds", "bounds"),
        ("late", "late"),
    ],
)
def test_response_classifier_uses_exact_priority(mutation, failure_kind):
    contract = PolicyContract.from_dict(live_policy_payload())
    history = {
        "session_id": "benchmark-test",
        "request_seq": 10,
        "observation_tick": 240,
        "rtc_delay_steps": 6,
        "committed_actions": np.zeros((6, 36), np.float32),
    }
    response = valid_benchmark_response(history, 0.05)
    if mutation == "shape":
        response = BenchmarkTransportResponse(
            response.action[:-1], response.metadata, response.latency_s
        )
    elif mutation == "dtype":
        response = BenchmarkTransportResponse(
            response.action.astype(np.float64), response.metadata, response.latency_s
        )
    elif mutation == "metadata_keys":
        response.metadata.pop("first_action_tick")
    elif mutation == "metadata_value":
        response.metadata["request_seq"] = 11
    elif mutation == "bounds":
        response.action[:, 31] = 0.75
    elif mutation == "late":
        response = BenchmarkTransportResponse(response.action, response.metadata, 0.121)
    with pytest.raises(BenchmarkRequestError) as caught:
        classify_response(response, history, contract, make_joint_contract())
    assert caught.value.kind == failure_kind


class FakeHttpResponse:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def json(self):
        return self.payload


class FakeHttpSession:
    def __init__(self, get_response=None, post_response=None):
        self.get_response = get_response
        self.post_response = post_response
        self.get_calls = []
        self.post_calls = []

    def get(self, *args, **kwargs):
        self.get_calls.append((args, kwargs))
        if isinstance(self.get_response, Exception):
            raise self.get_response
        return self.get_response

    def post(self, *args, **kwargs):
        self.post_calls.append((args, kwargs))
        if isinstance(self.post_response, Exception):
            raise self.post_response
        return self.post_response


def serialized_response(response):
    return {
        "action": numpy_serialize(response.action),
        "metadata": response.metadata,
    }


def test_http_transport_serializes_exact_rtc_request_and_decodes_response():
    payload = live_policy_payload()
    history = {
        "session_id": "benchmark-test",
        "request_seq": 0,
        "observation_tick": 0,
        "rtc_delay_steps": 6,
        "committed_actions": np.zeros((6, 36), np.float32),
        "reset": True,
    }
    expected = valid_benchmark_response(history, 0.05)
    session = FakeHttpSession(
        FakeHttpResponse(payload),
        FakeHttpResponse(serialized_response(expected)),
    )
    times = iter((10.0, 10.05))
    transport = HttpBenchmarkTransport(
        "http://127.0.0.1:22085",
        "rgb_head_stereo_left",
        session=session,
        clock=lambda: next(times),
    )
    assert transport.get_contract() == payload
    sample = representative_samples(1)[0]
    result = transport.query(0, sample, history)
    assert result.latency_s == pytest.approx(0.05)
    assert np.array_equal(result.action, expected.action)
    assert result.metadata == expected.metadata
    request = session.post_calls[0][1]["json"]
    assert set(request) == {
        "image",
        "instruction",
        "history",
        "state",
        "condition",
        "gt_action",
        "dataset_name",
        "timestamp",
    }
    assert set(request["image"]) == {"rgb_head_stereo_left"}
    assert set(request["state"]) == {"states"}
    assert request["condition"] == {}
    assert request["gt_action"] == []
    assert request["dataset_name"] == "simple"
    assert request["history"]["reset"] is True


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (requests.Timeout("timeout"), "timeout"),
        (FakeHttpResponse(error=requests.HTTPError("http")), "http"),
        (FakeHttpResponse(payload=[]), "decode"),
    ],
)
def test_http_contract_failure_mapping_is_exact(response, kind):
    transport = HttpBenchmarkTransport(
        "http://127.0.0.1:22085",
        "rgb_head_stereo_left",
        session=FakeHttpSession(get_response=response),
    )
    with pytest.raises(BenchmarkRequestError) as caught:
        transport.get_contract()
    assert caught.value.kind == kind


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (requests.Timeout("timeout"), "timeout"),
        (FakeHttpResponse(error=requests.HTTPError("http")), "http"),
        (FakeHttpResponse(payload=[]), "decode"),
    ],
)
def test_http_action_failure_mapping_is_exact(response, kind):
    transport = HttpBenchmarkTransport(
        "http://127.0.0.1:22085",
        "rgb_head_stereo_left",
        session=FakeHttpSession(post_response=response),
    )
    history = {
        "session_id": "benchmark-test",
        "request_seq": 0,
        "observation_tick": 0,
        "rtc_delay_steps": 6,
        "committed_actions": np.zeros((6, 36), np.float32),
        "reset": True,
    }
    with pytest.raises(BenchmarkRequestError) as caught:
        transport.query(0, representative_samples(1)[0], history)
    assert caught.value.kind == kind


def test_certified_bundle_contains_exact_contract_hash_and_commit_evidence(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    dataset_manifest = tmp_path / "dataset.json"
    samples = tmp_path / "samples.npz"
    checkpoint.write_bytes(b"checkpoint")
    dataset_manifest.write_bytes(b"dataset")
    write_representative_npz(samples, representative_samples(100))
    contract_payload = live_policy_payload(
        checkpoint_sha256=sha256_file(checkpoint),
        dataset_manifest_sha256=sha256_file(dataset_manifest),
    )
    contract_path = tmp_path / "policy-contract.json"
    contract_path.write_text(json.dumps(contract_payload))
    output = tmp_path / "bundle"
    write_certification_bundle(
        output_dir=output,
        report=BenchmarkReport(
            10,
            100,
            100,
            {key: 0 for key in FAILURE_KEYS},
            0.05,
            0.10,
            True,
        ),
        policy_contract_path=contract_path,
        fetched_server_contract=contract_payload,
        checkpoint_path=checkpoint,
        dataset_manifest_path=dataset_manifest,
        samples_path=samples,
    )
    assert {path.name for path in output.iterdir()} == {
        "latency_report.json",
        "policy_contract.json",
        "checkpoint.sha256",
        "dataset_manifest.sha256",
        "request_samples.sha256",
        "server_commit.txt",
        "converter_commit.txt",
    }
    assert json.loads((output / "policy_contract.json").read_text()) == (
        contract_payload
    )
    assert (output / "checkpoint.sha256").read_text().strip() == sha256_file(checkpoint)
    assert (output / "dataset_manifest.sha256").read_text().strip() == sha256_file(
        dataset_manifest
    )
    assert (output / "request_samples.sha256").read_text().strip() == sha256_file(
        samples
    )
    assert (output / "server_commit.txt").read_text().strip() == contract_payload[
        "server_commit"
    ]
    assert (output / "converter_commit.txt").read_text().strip() == (
        contract_payload["converter_commit"]
    )
    latency = json.loads((output / "latency_report.json").read_text())
    assert latency["warmup_requests"] == 10
    assert latency["measured_requests"] == latency["successes"] == 100
    assert latency["failures"] == {key: 0 for key in FAILURE_KEYS}
    assert latency["p99_method"] == "higher"
    assert latency["certified"] is True


def test_bundle_refuses_hash_mismatch_and_existing_destination(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    dataset_manifest = tmp_path / "dataset.json"
    samples = tmp_path / "samples.npz"
    checkpoint.write_bytes(b"checkpoint")
    dataset_manifest.write_bytes(b"dataset")
    write_representative_npz(samples, representative_samples(100))
    payload = live_policy_payload(
        checkpoint_sha256="0" * 64,
        dataset_manifest_sha256=sha256_file(dataset_manifest),
    )
    contract_path = tmp_path / "policy-contract.json"
    contract_path.write_text(json.dumps(payload))
    report = BenchmarkReport(
        10,
        100,
        100,
        {key: 0 for key in FAILURE_KEYS},
        0.05,
        0.10,
        True,
    )
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="checkpoint hash"):
        write_certification_bundle(
            output_dir=output,
            report=report,
            policy_contract_path=contract_path,
            fetched_server_contract=payload,
            checkpoint_path=checkpoint,
            dataset_manifest_path=dataset_manifest,
            samples_path=samples,
        )
    assert not output.exists()
    output.mkdir()
    with pytest.raises(FileExistsError):
        write_certification_bundle(
            output_dir=output,
            report=report,
            policy_contract_path=contract_path,
            fetched_server_contract=payload,
            checkpoint_path=checkpoint,
            dataset_manifest_path=dataset_manifest,
            samples_path=samples,
        )


def test_cli_requires_artifact_identity_paths_and_exact_counts():
    parser = build_parser()
    destinations = {action.dest: action for action in parser._actions}
    for name in (
        "server_url",
        "policy_contract",
        "checkpoint",
        "dataset_manifest",
        "representative_samples",
        "warmup_requests",
        "measured_requests",
        "output_dir",
    ):
        assert destinations[name].required is True


def test_cli_rejects_weakened_counts_before_reading_artifacts():
    with pytest.raises(SystemExit, match="exactly 10 warmups/100 measured"):
        main(
            [
                "--server-url",
                "http://127.0.0.1:22085",
                "--policy-contract",
                "missing-contract.json",
                "--checkpoint",
                "missing-checkpoint.pt",
                "--dataset-manifest",
                "missing-dataset.json",
                "--representative-samples",
                "missing-samples.npz",
                "--warmup-requests",
                "9",
                "--measured-requests",
                "100",
                "--output-dir",
                "missing-output",
            ]
        )


def test_cli_rejects_test_only_contract_before_network_or_sample_loading(tmp_path):
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(policy_payload(test_only=True)))
    with pytest.raises(SystemExit, match="test-only contracts"):
        main(
            [
                "--server-url",
                "http://127.0.0.1:22085",
                "--policy-contract",
                str(contract),
                "--checkpoint",
                "missing-checkpoint.pt",
                "--dataset-manifest",
                "missing-dataset.json",
                "--representative-samples",
                "missing-samples.npz",
                "--warmup-requests",
                "10",
                "--measured-requests",
                "100",
                "--output-dir",
                "missing-output",
            ]
        )
