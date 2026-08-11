import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.psi0_simple_real_bridge import (  # noqa: E402
    HttpInferenceWorker,
    ShutdownCoordinator,
)
from simple.baselines.client import (  # noqa: E402
    HttpActionClient,
    convert_numpy_in_dict,
    numpy_serialize,
)
from simple.deploy.psi0_simple_bridge import (  # noqa: E402
    PolicyContract,
    RtcRequest,
)


class FixtureBridge:
    def __init__(self, with_hold):
        self._hold = None
        if with_hold:
            self._hold = SimpleNamespace(
                target_upper_body_pose=np.zeros(31, np.float32),
                base_height_command=np.asarray([0.74], np.float32),
                navigate_cmd=np.zeros(4, np.float32),
                timestamp=0.0,
                target_time=0.02,
            )

    def stop(self):
        return None

    def build_bounded_shutdown_hold(self):
        return self._hold


class RecordingSink:
    def __init__(self):
        self.goals = []
        self.publish_attempts = 0
        self.closed = False
        self.publisher_closed_after_publish_count = None

    def publish(self, goal):
        if self.closed:
            raise RuntimeError("publish after close")
        self.publish_attempts += 1
        self.goals.append(
            {
                "scheduled_at": time.monotonic(),
                "goal": {
                    "target_upper_body_pose": goal.target_upper_body_pose.tolist(),
                    "base_height_command": goal.base_height_command.tolist(),
                    "navigate_cmd": goal.navigate_cmd.tolist(),
                    "timestamp": float(goal.timestamp),
                    "target_time": float(goal.target_time),
                },
            }
        )
        return True

    def close(self, timeout_s):
        self.closed = True
        self.publisher_closed_after_publish_count = self.publish_attempts


class ClosingResource:
    def __init__(self):
        self.closed = False

    def close(self, timeout_s):
        self.closed = True


class IdleWorker(ClosingResource):
    busy = False


class DelayedRtcHandler(BaseHTTPRequestHandler):
    def log_message(self, _format, *args):
        return None

    def do_POST(self):
        if self.path != "/act-rtc-v1":
            self.send_error(404)
            return
        size = int(self.headers["Content-Length"])
        self.rfile.read(size)
        self.server.request_accepted.set()
        time.sleep(5.2)
        response = convert_numpy_in_dict(
            {
                "action": np.zeros((24, 36), np.float32),
                "metadata": {
                    "session_id": "fixture-session",
                    "request_seq": 0,
                    "observation_tick": 100,
                    "prediction_horizon": 30,
                    "execution_horizon": 24,
                    "rtc_delay_steps": 6,
                    "first_action_tick": 106,
                },
            },
            numpy_serialize,
        )
        body = json.dumps(response, separators=(",", ":")).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


class DelayedRtcHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = False
    block_on_close = True


class RtcServerHarness:
    def __init__(self):
        self.request_accepted = threading.Event()
        self.server = DelayedRtcHttpServer(("127.0.0.1", 0), DelayedRtcHandler)
        self.server.request_accepted = self.request_accepted
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.01),
            name="fixture-rtc-server",
            daemon=False,
        )
        self.thread.start()

    def close(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        self.server.shutdown()
        self.thread.join(max(0.0, deadline - time.monotonic()))
        if self.thread.is_alive():
            raise RuntimeError("fixture RTC accept loop missed deadline")
        self.server.server_close()
        if time.monotonic() > deadline:
            raise RuntimeError("fixture RTC handler missed deadline")


def fixture_policy_contract():
    return PolicyContract.from_dict(
        {
            "schema": "simple.psi0.policy-contract.v2",
            "test_only": True,
            "checkpoint_sha256": "a" * 64,
            "dataset_manifest_sha256": "b" * 64,
            "raw_episode_sha256": "c" * 64,
            "processed_episode_sha256": "d" * 64,
            "source_episode_index": 7,
            "processed_episode_index": 3,
            "converter_commit": "e" * 40,
            "server_commit": "f" * 40,
            "converter_layout": "g1_simple_32_rpyh_v2",
            "observation_dim": 32,
            "action_dim": 36,
            "action_frequency_hz": 50,
            "prediction_horizon": 30,
            "execution_horizon": 24,
            "rtc_delay_steps": 6,
            "rtc_training_max_delay": 7,
            "rtc_enabled": True,
            "rtc_endpoint": "/act-rtc-v1",
            "request_semantics": "exact-post-slew-committed-prefix",
            "response_semantics": "denormalized-executable-suffix",
            "image_key": "rgb_head_stereo_left",
            "camera_color_order": "rgb",
        }
    )


def start_inflight_request(server):
    contract = fixture_policy_contract()
    worker = HttpInferenceWorker(
        client=HttpActionClient("127.0.0.1", server.port, timeout=5.0),
        clock=time.monotonic,
        contract=contract,
    )
    committed = np.zeros((6, 36), np.float32)
    committed[:, 31] = 0.74
    worker.submit(
        RtcRequest(
            generation=1,
            session_id="fixture-session",
            request_seq=0,
            observation_tick=100,
            history_tick=99,
            observation=np.zeros((1, 32), np.float32),
            image=np.zeros((8, 8, 3), np.uint8),
            committed_actions=committed,
            reset=True,
            deadline_at=time.monotonic() + 0.12,
        )
    )
    return worker


def atomic_write_json(path, payload):
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")))
    os.replace(temporary, destination)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        required=True,
        choices=("no-state", "inflight-five-second"),
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--ready", required=True)
    args = parser.parse_args(argv)

    inflight = args.scenario == "inflight-five-second"
    bridge = FixtureBridge(with_hold=inflight)
    sink = RecordingSink()
    camera = ClosingResource()
    server = RtcServerHarness() if inflight else None
    worker = start_inflight_request(server) if inflight else IdleWorker()
    inference_requests = [{"request_seq": 0}] if inflight else []
    state = ClosingResource()
    ros = ClosingResource()
    keyboard = ClosingResource()
    metrics = ClosingResource()
    ownership_guard = ClosingResource()
    coordinator = ShutdownCoordinator(
        bridge=bridge,
        command_sink=sink,
        camera=camera,
        worker=worker,
        state_source=state,
        ownership_guard=ownership_guard,
        ros_runtime=ros,
        keyboard=keyboard,
        metrics=metrics,
    )

    interrupted = threading.Event()
    signal.signal(signal.SIGINT, lambda _signum, _frame: interrupted.set())
    if inflight and not server.request_accepted.wait(0.2):
        raise RuntimeError("fixture worker did not accept request")
    Path(args.ready).touch(exist_ok=False)
    interrupted.wait()
    shutdown = coordinator.close()
    fixture_errors = []
    if server is not None:
        try:
            server.close(max(0.0, shutdown.deadline_at - time.monotonic()))
        except Exception as error:
            fixture_errors.append(f"fixture_server: {error}")

    current = threading.current_thread()
    live = [
        thread.name
        for thread in threading.enumerate()
        if thread is not current and not thread.daemon and thread.is_alive()
    ]
    atomic_write_json(
        args.report,
        {
            "owned_ports": [] if server is None else [server.port],
            "goal_lower_bounds": [-2.0] * 31,
            "goal_upper_bounds": [2.0] * 31,
            "goals": sink.goals,
            "publish_attempts": sink.publish_attempts,
            "publisher_closed_after_publish_count": (
                sink.publisher_closed_after_publish_count
            ),
            "publisher_closed": sink.closed,
            "camera_closed": camera.closed,
            "terminal_restored": keyboard.closed,
            "inference_requests": inference_requests,
            "request_accepted": (
                False if server is None else server.request_accepted.is_set()
            ),
            "live_non_daemon_bridge_threads": live,
            "shutdown_started_at": shutdown.started_at,
            "shutdown_deadline_at": shutdown.deadline_at,
            "shutdown_finished_at": shutdown.finished_at,
            "cleanup_errors": [*shutdown.cleanup_errors, *fixture_errors],
        },
    )
    return 0 if not shutdown.cleanup_errors and not fixture_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
