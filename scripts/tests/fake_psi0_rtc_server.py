import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import struct
import threading
import time

import numpy as np

from simple.baselines.client import (
    RequestMessage,
    convert_numpy_in_dict,
    numpy_serialize,
)


OUTER_KEYS = {
    "image",
    "instruction",
    "history",
    "state",
    "condition",
    "gt_action",
    "dataset_name",
    "timestamp",
}
BASE_HISTORY_KEYS = {
    "session_id",
    "request_seq",
    "observation_tick",
    "rtc_delay_steps",
    "committed_actions",
}


class ProtocolError(RuntimeError):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


@dataclass(frozen=True)
class RequestRecord:
    request_seq: int
    observation_tick: int
    committed_actions: np.ndarray
    reset: bool
    applied_latency_s: float


@dataclass
class FakePolicyState:
    contract: dict
    normal_latency_s: float
    control_token: str | None
    lock: threading.Lock = field(default_factory=threading.Lock)
    delay_by_seq: dict[int, float] = field(default_factory=dict)
    records: list[RequestRecord] = field(default_factory=list)
    last_started_request_seq: int = -1
    active_requests: int = 0
    max_concurrent_requests: int = 0

    def arm_next(self, latency_s):
        if type(latency_s) not in (int, float) or type(latency_s) is bool:
            raise ProtocolError(400, "latency_s must be numeric")
        latency_s = float(latency_s)
        if not 0.0 <= latency_s <= 1.0:
            raise ProtocolError(400, "latency_s outside [0,1]")
        with self.lock:
            target = self.last_started_request_seq + 1
            if target in self.delay_by_seq:
                raise ProtocolError(409, "next request already armed")
            self.delay_by_seq[target] = latency_s
        return target, latency_s


def _validate_request(payload, state):
    if type(payload) is not dict or set(payload) != OUTER_KEYS:
        raise ProtocolError(400, "request key set")
    request = RequestMessage.deserialize(payload)
    contract = state.contract
    if request.dataset_name != "simple" or request.condition != {}:
        raise ProtocolError(400, "dataset/condition")
    if type(request.image) is not dict or set(request.image) != {contract["image_key"]}:
        raise ProtocolError(400, "image dictionary")
    if type(request.state) is not dict or set(request.state) != {"states"}:
        raise ProtocolError(400, "state dictionary")
    image = np.asarray(request.image[contract["image_key"]])
    states = np.asarray(request.state["states"])
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ProtocolError(400, "image value")
    if (
        states.dtype != np.float32
        or states.shape != (1, 32)
        or not np.isfinite(states).all()
    ):
        raise ProtocolError(400, "state value")
    if request.gt_action != [] or type(request.instruction) is not str:
        raise ProtocolError(400, "gt_action/instruction")
    history = request.history
    if type(history) is not dict:
        raise ProtocolError(400, "history type")
    seq = history.get("request_seq")
    expected_keys = BASE_HISTORY_KEYS | ({"reset"} if seq == 0 else set())
    if set(history) != expected_keys:
        raise ProtocolError(400, "history key set")
    if type(history["session_id"]) is not str or not history["session_id"]:
        raise ProtocolError(400, "session_id")
    for key in ("request_seq", "observation_tick", "rtc_delay_steps"):
        if type(history[key]) is not int:
            raise ProtocolError(400, f"{key} type")
    if seq == 0 and history["reset"] is not True:
        raise ProtocolError(400, "R0 reset")
    if history["rtc_delay_steps"] != contract["rtc_delay_steps"]:
        raise ProtocolError(400, "rtc delay")
    committed = np.asarray(history["committed_actions"])
    expected = (contract["rtc_delay_steps"], contract["action_dim"])
    if committed.dtype != np.float32 or committed.shape != expected:
        raise ProtocolError(400, "committed prefix")
    if not np.isfinite(committed).all():
        raise ProtocolError(400, "committed prefix finite")
    return request, history, committed.copy()


class FakePolicyHandler(BaseHTTPRequestHandler):
    server_version = "SIMPLEFakeRTC/1"

    @property
    def state(self):
        return self.server.state

    def log_message(self, _format, *args):
        return None

    def _read_json(self):
        try:
            size = int(self.headers.get("Content-Length", ""))
            return json.loads(self.rfile.read(size))
        except Exception as error:
            raise ProtocolError(400, f"invalid JSON: {error}") from error

    def _send(self, status, payload):
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        return (
            self.state.control_token is not None
            and self.headers.get("X-Test-Control-Token") == self.state.control_token
        )

    def do_GET(self):
        if self.path == "/contract":
            self._send(200, self.state.contract)
            return
        if self.path == "/test-control/status":
            if not self._authorized():
                self._send(403, {"error": "forbidden"})
                return
            with self.state.lock:
                payload = {
                    "last_started_request_seq": (self.state.last_started_request_seq),
                    "active_requests": self.state.active_requests,
                    "max_concurrent_requests": (self.state.max_concurrent_requests),
                    "records": [
                        {
                            "request_seq": record.request_seq,
                            "applied_latency_s": record.applied_latency_s,
                        }
                        for record in self.state.records
                    ],
                }
            self._send(200, payload)
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            if self.path == "/test-control/arm-next-delay":
                payload = self._read_json()
                if type(payload) is not dict or set(payload) != {
                    "token",
                    "latency_s",
                }:
                    raise ProtocolError(400, "control key set")
                if payload["token"] != self.state.control_token:
                    raise ProtocolError(403, "forbidden")
                seq, delay = self.state.arm_next(payload["latency_s"])
                self._send(
                    202,
                    {"armed_request_seq": seq, "latency_s": delay},
                )
                return
            if self.path != "/act-rtc-v1":
                raise ProtocolError(404, "not found")
            request, history, committed = _validate_request(
                self._read_json(), self.state
            )
            seq = history["request_seq"]
            with self.state.lock:
                expected = self.state.last_started_request_seq + 1
                if seq != expected:
                    raise ProtocolError(409, f"expected request_seq {expected}")
                self.state.last_started_request_seq = seq
                delay = self.state.delay_by_seq.pop(seq, self.state.normal_latency_s)
                self.state.active_requests += 1
                self.state.max_concurrent_requests = max(
                    self.state.max_concurrent_requests,
                    self.state.active_requests,
                )
            recorded = False
            try:
                time.sleep(delay)
                contract = self.state.contract
                actions = np.repeat(
                    committed[-1:], contract["execution_horizon"], axis=0
                ).astype(np.float32)
                actions[:, 32:36] = 0.0
                metadata = {
                    "session_id": history["session_id"],
                    "request_seq": seq,
                    "observation_tick": history["observation_tick"],
                    "prediction_horizon": contract["prediction_horizon"],
                    "execution_horizon": contract["execution_horizon"],
                    "rtc_delay_steps": contract["rtc_delay_steps"],
                    "first_action_tick": (
                        history["observation_tick"] + contract["rtc_delay_steps"]
                    ),
                }
                response = convert_numpy_in_dict(
                    {"action": actions, "metadata": metadata},
                    numpy_serialize,
                )
                with self.state.lock:
                    self.state.active_requests -= 1
                    self.state.records.append(
                        RequestRecord(
                            seq,
                            history["observation_tick"],
                            committed,
                            history.get("reset", False),
                            delay,
                        )
                    )
                    recorded = True
                self._send(200, response)
            finally:
                if not recorded:
                    with self.state.lock:
                        self.state.active_requests -= 1
        except ProtocolError as error:
            self._send(error.status, {"error": str(error)})


class LoopbackThreadingServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False

    def shutdown_request(self, request):
        request.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_LINGER,
            struct.pack("ii", 1, 0),
        )
        super().shutdown_request(request)


class FakePolicyHandle:
    def __init__(self, server, thread):
        self.server = server
        self.thread = thread
        self.port = server.server_address[1]

    @property
    def records(self):
        with self.server.state.lock:
            return tuple(self.server.state.records)

    @property
    def max_concurrent_requests(self):
        with self.server.state.lock:
            return self.server.state.max_concurrent_requests

    def delay_next_request(self, seconds):
        return self.server.state.arm_next(seconds)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(0.5)
        if self.thread.is_alive():
            raise RuntimeError("fake policy server did not stop")


def start_fake_policy(contract_path, normal_latency_s, control_token=None, port=0):
    contract = json.loads(Path(contract_path).read_text())
    server = LoopbackThreadingServer(("127.0.0.1", port), FakePolicyHandler)
    server.state = FakePolicyState(contract, float(normal_latency_s), control_token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return FakePolicyHandle(server, thread)


@contextmanager
def running_fake_policy(contract_path, normal_latency_s=0.05):
    handle = start_fake_policy(contract_path, normal_latency_s)
    try:
        yield handle
    finally:
        handle.close()


def atomic_ready(path, host, port):
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps({"host": host, "port": port}))
    os.replace(temporary, destination)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--normal-latency-s", required=True, type=float)
    parser.add_argument("--control-token", required=True)
    parser.add_argument("--ready-json", required=True)
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1" or not args.control_token:
        parser.error("authenticated loopback is required")
    handle = start_fake_policy(
        args.contract,
        args.normal_latency_s,
        args.control_token,
        args.port,
    )
    stopped = threading.Event()
    signal.signal(signal.SIGINT, lambda _signum, _frame: stopped.set())
    atomic_ready(args.ready_json, args.host, handle.port)
    stopped.wait()
    handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
