import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import threading
import time

import msgpack
import numpy as np
import zmq

from decoupled_wbc.control.sensor.sensor_server import ImageMessageSchema


def sentinel_rgb():
    image = np.zeros((48, 64, 3), np.uint8)
    image[:, :20] = [240, 0, 0]
    image[:, 44:] = [0, 0, 240]
    return np.ascontiguousarray(image)


class FakeCameraHandle:
    def __init__(self, key, host="127.0.0.1", port=0):
        if host != "127.0.0.1":
            raise ValueError("fake camera is loopback-only")
        self.key = key
        self.host = host
        self.requested_port = port
        self.port = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._error = None
        self._thread = threading.Thread(
            target=self._run,
            name="fake-composed-camera",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(0.5):
            raise TimeoutError("fake camera did not bind")
        if self._error is not None:
            raise RuntimeError(f"fake camera failed: {self._error}")

    def _run(self):
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        try:
            socket.setsockopt(zmq.LINGER, 0)
            endpoint = f"tcp://{self.host}"
            if self.requested_port == 0:
                self.port = socket.bind_to_random_port(endpoint)
            else:
                self.port = self.requested_port
                socket.bind(f"{endpoint}:{self.port}")
            image = sentinel_rgb()
            self._ready.set()
            while not self._stop.is_set():
                schema = ImageMessageSchema(
                    timestamps={self.key: time.monotonic()},
                    images={self.key: image},
                )
                socket.send(
                    msgpack.packb(schema.serialize(), use_bin_type=True),
                    flags=zmq.NOBLOCK,
                )
                self._stop.wait(0.03)
        except Exception as error:
            self._error = error
            self._ready.set()
        finally:
            socket.close(linger=0)
            context.term()

    def close(self):
        self._stop.set()
        self._thread.join(0.5)
        if self._thread.is_alive():
            raise RuntimeError("fake camera did not stop within 0.5 seconds")
        if self._error is not None:
            raise RuntimeError(f"fake camera failed: {self._error}")


@contextmanager
def running_fake_camera(key="rgb_head_stereo_left"):
    handle = FakeCameraHandle(key)
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
    parser.add_argument("--key", default="rgb_head_stereo_left")
    parser.add_argument("--ready-json", required=True)
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1":
        parser.error("fake camera is loopback-only")
    handle = FakeCameraHandle(args.key, args.host, args.port)
    stopped = threading.Event()
    signal.signal(signal.SIGINT, lambda _signum, _frame: stopped.set())
    atomic_ready(args.ready_json, args.host, handle.port)
    stopped.wait()
    handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
