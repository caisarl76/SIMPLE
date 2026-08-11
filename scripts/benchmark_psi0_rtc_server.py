import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from urllib.parse import urlparse
import uuid

import numpy as np
import requests

from decoupled_wbc.control.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from simple.baselines.client import (
    RequestMessage,
    convert_numpy_in_dict,
    numpy_deserialize,
)
from simple.deploy.psi0_simple_bridge import (
    PSI0_ACTION_JOINT_NAMES,
    JointContract,
    PolicyContract,
    validate_action_suffix,
)


FAILURE_KEYS = (
    "timeout",
    "http",
    "decode",
    "shape",
    "metadata",
    "bounds",
    "late",
)


class BenchmarkRequestError(RuntimeError):
    def __init__(self, kind, detail=""):
        if kind not in FAILURE_KEYS:
            raise ValueError(f"unknown benchmark failure kind: {kind}")
        super().__init__(detail or kind)
        self.kind = kind


@dataclass(frozen=True)
class BenchmarkSample:
    image: np.ndarray
    state: np.ndarray
    instruction: str


@dataclass(frozen=True)
class BenchmarkTransportResponse:
    action: np.ndarray
    metadata: dict[str, object]
    latency_s: float


@dataclass(frozen=True)
class BenchmarkReport:
    warmup_requests: int
    measured_requests: int
    successes: int
    failures: dict[str, int]
    p99_latency_s: float
    latency_limit_s: float
    certified: bool


def load_representative_samples(path):
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"images", "states", "instructions"}:
            raise ValueError("sample NPZ keys must be images/states/instructions")
        images = payload["images"]
        states = payload["states"]
        instructions = payload["instructions"]
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("images must be contiguous uint8 (N,H,W,3)")
    if not images.flags.c_contiguous:
        raise ValueError("images must be contiguous")
    if states.dtype != np.float32 or states.shape != (len(images), 1, 32):
        raise ValueError("states must be float32 (N,1,32)")
    if not np.isfinite(states).all():
        raise ValueError("states must be finite")
    if instructions.shape != (len(images),) or instructions.dtype.kind not in "US":
        raise ValueError("instructions must be an N-element string array")
    if len(images) < 100:
        raise ValueError("at least 100 representative samples are required")
    return tuple(
        BenchmarkSample(images[i].copy(), states[i].copy(), str(instructions[i]))
        for i in range(len(images))
    )


def expected_metadata(history, contract):
    return {
        "session_id": history["session_id"],
        "request_seq": history["request_seq"],
        "observation_tick": history["observation_tick"],
        "prediction_horizon": contract.prediction_horizon,
        "execution_horizon": contract.execution_horizon,
        "rtc_delay_steps": contract.rtc_delay_steps,
        "first_action_tick": (history["observation_tick"] + contract.rtc_delay_steps),
    }


def classify_response(response, history, contract, joint_contract):
    if type(response) is not BenchmarkTransportResponse:
        raise BenchmarkRequestError("decode", "transport response type")
    action = np.asarray(response.action)
    expected_shape = (contract.execution_horizon, contract.action_dim)
    if action.shape != expected_shape or action.dtype != np.float32:
        raise BenchmarkRequestError("shape", f"expected {expected_shape} float32")
    metadata = response.metadata
    expected = expected_metadata(history, contract)
    if type(metadata) is not dict or set(metadata) != set(expected):
        raise BenchmarkRequestError("metadata", "metadata key set")
    for key, value in expected.items():
        required_type = str if key == "session_id" else int
        if type(metadata[key]) is not required_type or metadata[key] != value:
            raise BenchmarkRequestError("metadata", key)
    try:
        validate_action_suffix(action, contract.execution_horizon, joint_contract)
    except ValueError as error:
        raise BenchmarkRequestError("bounds", str(error)) from error
    if (
        type(response.latency_s) is not float
        or not np.isfinite(response.latency_s)
        or response.latency_s > contract.rtc_delay_steps / contract.action_frequency_hz
    ):
        raise BenchmarkRequestError("late", "RTC response missed r+d")
    return response.latency_s


class HttpBenchmarkTransport:
    def __init__(
        self,
        server_url,
        image_key,
        timeout_s=5.0,
        session=requests,
        clock=time.monotonic,
    ):
        parsed = urlparse(server_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("server_url must be an absolute HTTP(S) URL")
        if (
            parsed.path not in {"", "/"}
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("server_url must not contain path/query/fragment")
        if type(image_key) is not str or not image_key:
            raise ValueError("image_key must be a non-empty string")
        self.base_url = server_url.rstrip("/")
        self.image_key = image_key
        self.timeout_s = timeout_s
        self.session = session
        self.clock = clock
        self.fetched_contract = None

    def get_contract(self):
        try:
            response = self.session.get(
                self.base_url + "/contract", timeout=self.timeout_s
            )
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout as error:
            raise BenchmarkRequestError("timeout", str(error)) from error
        except requests.HTTPError as error:
            raise BenchmarkRequestError("http", str(error)) from error
        except Exception as error:
            raise BenchmarkRequestError("decode", str(error)) from error
        if type(payload) is not dict:
            raise BenchmarkRequestError("decode", "contract response type")
        self.fetched_contract = payload
        return payload

    def query(self, request_index, sample, history):
        del request_index
        request = RequestMessage(
            image={self.image_key: sample.image},
            instruction=sample.instruction,
            history=history,
            state={"states": sample.state},
            condition={},
            gt_action=[],
            dataset_name="simple",
            timestamp=str(time.time_ns()),
        )
        started = self.clock()
        try:
            response = self.session.post(
                self.base_url + "/act-rtc-v1",
                json=request.serialize(),
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            payload = convert_numpy_in_dict(response.json(), numpy_deserialize)
            if type(payload) is not dict or set(payload) != {"action", "metadata"}:
                raise ValueError("RTC response key set")
            result = BenchmarkTransportResponse(
                action=payload["action"],
                metadata=payload["metadata"],
                latency_s=float(self.clock() - started),
            )
        except BenchmarkRequestError:
            raise
        except requests.Timeout as error:
            raise BenchmarkRequestError("timeout", str(error)) from error
        except requests.HTTPError as error:
            raise BenchmarkRequestError("http", str(error)) from error
        except Exception as error:
            raise BenchmarkRequestError("decode", str(error)) from error
        return result


def benchmark_server(
    *,
    transport,
    samples,
    contract,
    joint_contract,
    warmup_requests=10,
    measured_requests=100,
):
    if warmup_requests != 10 or measured_requests != 100:
        raise ValueError("certification requires 10 warmups and 100 measured requests")
    if len(samples) < 100:
        raise ValueError("at least 100 representative samples are required")
    fetched = PolicyContract.from_dict(transport.get_contract())
    if fetched != contract:
        raise ValueError("server policy contract differs from local contract")

    joint_index = {name: index for index, name in enumerate(joint_contract.joint_names)}
    action_joint_names = (
        *PSI0_ACTION_JOINT_NAMES,
        "waist_roll_joint",
        "waist_pitch_joint",
        "waist_yaw_joint",
    )
    if any(name not in joint_index for name in action_joint_names):
        raise ValueError("joint contract cannot build stationary prefix")
    stationary = np.zeros(36, np.float32)
    for action_index, name in enumerate(action_joint_names):
        model_index = joint_index[name]
        stationary[action_index] = np.clip(
            0.0,
            joint_contract.lower_position_limits[model_index],
            joint_contract.upper_position_limits[model_index],
        )
    stationary[31] = 0.74
    committed = np.repeat(stationary[None], contract.rtc_delay_steps, axis=0)
    session_id = "benchmark-" + uuid.uuid4().hex

    def make_history(request_index):
        history = {
            "session_id": session_id,
            "request_seq": request_index,
            "observation_tick": request_index * contract.execution_horizon,
            "rtc_delay_steps": contract.rtc_delay_steps,
            "committed_actions": committed.copy(),
        }
        if request_index == 0:
            history["reset"] = True
        return history

    for request_index in range(warmup_requests):
        sample = samples[request_index % len(samples)]
        history = make_history(request_index)
        try:
            response = transport.query(request_index, sample, history)
            classify_response(response, history, contract, joint_contract)
        except BenchmarkRequestError as error:
            raise RuntimeError(
                f"warmup request {request_index} failed: {error.kind}: {error}"
            ) from error

    failures = {key: 0 for key in FAILURE_KEYS}
    success_latencies = []
    for measured_index in range(measured_requests):
        request_index = warmup_requests + measured_index
        sample = samples[request_index % len(samples)]
        history = make_history(request_index)
        try:
            response = transport.query(request_index, sample, history)
            latency = classify_response(
                response,
                history,
                contract,
                joint_contract,
            )
        except BenchmarkRequestError as error:
            failures[error.kind] += 1
        else:
            success_latencies.append(latency)

    p99_latency_s = (
        float(np.quantile(success_latencies, 0.99, method="higher"))
        if success_latencies
        else float("inf")
    )
    latency_limit_s = (contract.rtc_delay_steps - 1) / contract.action_frequency_hz
    certified = (
        len(success_latencies) == measured_requests
        and sum(failures.values()) == 0
        and p99_latency_s <= latency_limit_s
    )
    return BenchmarkReport(
        warmup_requests,
        measured_requests,
        len(success_latencies),
        failures,
        p99_latency_s,
        latency_limit_s,
        certified,
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_fsynced(path, data):
    payload = data.encode("utf-8") if isinstance(data, str) else data
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_certification_bundle(
    *,
    output_dir,
    report,
    policy_contract_path,
    fetched_server_contract,
    checkpoint_path,
    dataset_manifest_path,
    samples_path,
):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    if type(report) is not BenchmarkReport or not report.certified:
        raise ValueError("only a certified report may create an evidence bundle")
    if (
        report.warmup_requests != 10
        or report.measured_requests != 100
        or report.successes != 100
        or report.failures != {key: 0 for key in FAILURE_KEYS}
    ):
        raise ValueError("certification report counters do not meet the gate")

    local_payload = json.loads(Path(policy_contract_path).read_text())
    local_contract = PolicyContract.from_dict(local_payload)
    server_contract = PolicyContract.from_dict(fetched_server_contract)
    if local_contract != server_contract or local_payload != fetched_server_contract:
        raise ValueError("fetched server contract differs from local contract")
    if local_contract.test_only:
        raise ValueError("test-only policy contract cannot be live-certified")
    checkpoint_hash = sha256_file(checkpoint_path)
    dataset_hash = sha256_file(dataset_manifest_path)
    sample_hash = sha256_file(samples_path)
    if checkpoint_hash != local_contract.checkpoint_sha256:
        raise ValueError("checkpoint hash differs from policy contract")
    if dataset_hash != local_contract.dataset_manifest_sha256:
        raise ValueError("dataset manifest hash differs from policy contract")
    for field in ("server_commit", "converter_commit"):
        if re.fullmatch(r"[0-9a-f]{40}", getattr(local_contract, field)) is None:
            raise ValueError(f"{field} must be a full lowercase Git SHA")

    latency_payload = asdict(report)
    latency_payload["p99_method"] = "higher"
    files = {
        "latency_report.json": (
            json.dumps(latency_payload, sort_keys=True, separators=(",", ":")) + "\n"
        ),
        "policy_contract.json": (
            json.dumps(local_payload, sort_keys=True, separators=(",", ":")) + "\n"
        ),
        "checkpoint.sha256": checkpoint_hash + "\n",
        "dataset_manifest.sha256": dataset_hash + "\n",
        "request_samples.sha256": sample_hash + "\n",
        "server_commit.txt": local_contract.server_commit + "\n",
        "converter_commit.txt": local_contract.converter_commit + "\n",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name, data in files.items():
            _write_fsynced(temporary / name, data)
        _fsync_directory(temporary)
        os.rename(temporary, output)
        _fsync_directory(output.parent)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def build_local_joint_contract():
    model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
    names = tuple(model.joint_names)
    upper_indices = model.get_joint_group_indices("upper_body")
    upper_names = tuple(names[index] for index in upper_indices)
    if len(names) != 43 or len(upper_names) != 31:
        raise RuntimeError("local G1 model does not expose 43/31 joints")
    return JointContract(
        names,
        upper_names,
        model.lower_joint_limits.copy(),
        model.upper_joint_limits.copy(),
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--policy-contract", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--representative-samples", required=True)
    parser.add_argument("--warmup-requests", required=True, type=int)
    parser.add_argument("--measured-requests", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.warmup_requests != 10 or args.measured_requests != 100:
        raise SystemExit("live certification requires exactly 10 warmups/100 measured")
    contract_payload = json.loads(Path(args.policy_contract).read_text())
    contract = PolicyContract.from_dict(contract_payload)
    if contract.test_only:
        raise SystemExit("test-only contracts cannot be live-certified")
    samples = load_representative_samples(args.representative_samples)
    transport = HttpBenchmarkTransport(
        args.server_url,
        image_key=contract.image_key,
        timeout_s=5.0,
    )
    report = benchmark_server(
        transport=transport,
        samples=samples,
        contract=contract,
        joint_contract=build_local_joint_contract(),
        warmup_requests=args.warmup_requests,
        measured_requests=args.measured_requests,
    )
    if not report.certified:
        print(json.dumps(asdict(report), sort_keys=True))
        return 1
    write_certification_bundle(
        output_dir=args.output_dir,
        report=report,
        policy_contract_path=args.policy_contract,
        fetched_server_contract=transport.fetched_contract,
        checkpoint_path=args.checkpoint,
        dataset_manifest_path=args.dataset_manifest,
        samples_path=args.representative_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
