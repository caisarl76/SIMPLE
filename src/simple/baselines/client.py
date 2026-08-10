"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import os
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
# import draccus
import requests
from typing import Any, Dict, List, Union
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from base64 import b64decode, b64encode

def numpy_serialize(o):
    if isinstance(o, (np.ndarray, np.generic)):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(o.dtype),
            "shape": o.shape,
        }

    msg = f"Object of type {o.__class__.__name__} is not JSON serializable"
    raise TypeError(msg)


def numpy_deserialize(dct):
    if "__numpy__" in dct:
        np_obj = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return np_obj.reshape(shape) if (shape := dct["shape"]) else np_obj[0]
    return dct


def convert_numpy_in_dict(data, func):
    """
    Recursively processes a JSON-like dictionary, converting any NumPy arrays
    or lists of NumPy arrays into a serializable format using the provided function.

    Args:
        data: The JSON-like dictionary or object to process.
        func: A function to apply to each NumPy array to make it serializable.

    Returns:
        The processed dictionary or object with all NumPy arrays converted.
    """
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {key: convert_numpy_in_dict(value, func) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_in_dict(item, func) for item in data]
    elif isinstance(data, (np.ndarray, np.generic)):
        return func(data)
    else:
        return data
    
    
class Message(object):
    def __init__(self):
        pass
    
    def serialize(self):
        raise NotImplementedError
    
    @classmethod
    def deserialize(cls, response: Dict[str, Any]):
        raise NotImplementedError


class RequestMessage(Message):
    def __init__(self, image: Dict[str, Any], instruction: str, history: Dict[str, Any], state: Dict[str, Any], condition: Dict[str, Any], gt_action: Union[np.ndarray, List], dataset_name: str, timestamp: str):
        self.image, self.instruction, self.history, self.state, self.gt_action, self.dataset_name, self.timestamp = image, instruction, history, state, gt_action, dataset_name, timestamp
        self.condition = condition

    def serialize(self):
        msg = {
            "image": self.image,
            "instruction": self.instruction,
            "history": self.history,
            "state": self.state,
            "condition": self.condition,
            "gt_action": self.gt_action,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp
        }
        return convert_numpy_in_dict(msg, numpy_serialize)
    
    @classmethod
    def deserialize(cls, response: Dict[str, Any]):
        response = convert_numpy_in_dict(response, numpy_deserialize)
        return cls(
            image=response["image"],
            instruction=response["instruction"],
            history=response["history"],
            state=response["state"],
            condition=response["condition"],
            gt_action=response["gt_action"],
            dataset_name=response["dataset_name"],
            timestamp=response["timestamp"]
        )


class ResponseMessage(Message):
    def __init__(self, action: np.ndarray, err: float, traj_image: np.ndarray = None):
        self.action = action
        self.err = err
        self.traj_image = traj_image if traj_image is not None else np.zeros((1, 1, 3), dtype=np.uint8)
    
    def serialize(self):
        msg = {
            "action": self.action,
            "err": self.err,
            "traj_image": self.traj_image,
        }
        return convert_numpy_in_dict(msg, numpy_serialize)
    
    @classmethod
    def deserialize(cls, response: Dict[str, Any]):
        response = convert_numpy_in_dict(response, numpy_deserialize)
        err = response["err"] if "err" in response else 0.0
        traj_image = response["traj_image"] if "traj_image" in response else None
        if type(err) == str:
            print(f"[WARN] Server eror: {err}.")
        return cls(action=response["action"], err=err, traj_image=traj_image)



@dataclass(frozen=True)
class RtcActionResponse:
    action: np.ndarray
    metadata: dict[str, Any]
    err: float = 0.0


class HttpActionClient:
    def __init__(
        self,
        server_ip: str,
        server_port: int,
        timeout: float | None = None,
        session: requests.Session | None = None,
    ):
        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout
        self.session = session or requests.Session()

    @property
    def timestamp(self):
        return str(datetime.now()).replace(" ", "_").replace(":", "-")

    def get_contract(self, timeout: float = 2.0) -> dict[str, Any]:
        response = self.session.get(
            f"http://{self.server_ip}:{self.server_port}/contract", timeout=timeout
        )
        response.raise_for_status()
        try:
            result = response.json()
        except Exception as error:
            raise RuntimeError(f"invalid policy contract JSON: {error}") from error
        if type(result) is not dict:
            raise RuntimeError("policy contract response must be a JSON object")
        return result

    def _post(self, path: str, request: RequestMessage) -> dict[str, Any]:
        response = self.session.post(
            f"http://{self.server_ip}:{self.server_port}{path}",
            json=request.serialize(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        try:
            payload = response.json()
            payload = convert_numpy_in_dict(payload, numpy_deserialize)
        except Exception as error:
            raise RuntimeError(str(error)) from error
        if type(payload) is not dict:
            raise RuntimeError("policy response must be a JSON object")
        return payload

    def query_action(
        self,
        image_dict,
        instruction,
        state_dict,
        condition_dict,
        history=None,
        dataset="grasp",
        gt_action=None,
    ):
        if history is None:
            history = {key: [] for key in image_dict}
        if gt_action is None:
            gt_action = []

        request = RequestMessage(
            image_dict,
            instruction,
            history,
            state_dict,
            condition_dict,
            gt_action,
            dataset,
            self.timestamp,
        )
        try:
            parsed = ResponseMessage.deserialize(self._post("/act", request))
        except (requests.Timeout, requests.HTTPError):
            raise
        except Exception as error:
            raise RuntimeError(str(error)) from error
        trajectory = parsed.traj_image
        if not isinstance(trajectory, np.ndarray) or trajectory.ndim != 3:
            trajectory = None
        return parsed.action, parsed.err, trajectory

    def query_rtc_action(
        self,
        image_dict,
        instruction,
        state_dict,
        condition_dict,
        *,
        history,
        dataset="simple",
    ) -> RtcActionResponse:
        request = RequestMessage(
            image_dict,
            instruction,
            history,
            state_dict,
            condition_dict,
            [],
            dataset,
            self.timestamp,
        )
        payload = self._post("/act-rtc-v1", request)
        if set(payload) != {"action", "metadata"}:
            raise RuntimeError("RTC response requires action and metadata")
        metadata = payload["metadata"]
        metadata_types = {
            "session_id": str,
            "request_seq": int,
            "observation_tick": int,
            "prediction_horizon": int,
            "execution_horizon": int,
            "rtc_delay_steps": int,
            "first_action_tick": int,
        }
        if type(metadata) is not dict or set(metadata) != set(metadata_types):
            raise RuntimeError("RTC response metadata key set")
        for key, expected_type in metadata_types.items():
            if type(metadata[key]) is not expected_type:
                raise RuntimeError(f"RTC response metadata {key} type")
        action = payload["action"]
        if type(action) is not np.ndarray:
            raise RuntimeError("RTC response action must be a NumPy array")
        return RtcActionResponse(action=action, metadata=dict(metadata))
    

if __name__ == "__main__":
    server_ip = "localhost" #"172.17.0.1"
    server_port = 22085 #21000
    client = HttpActionClient(server_ip, server_port)
    
    from PIL import Image
    obs =  np.zeros((224, 224, 3), dtype=np.uint8) #np.array(Image.open("steore-left.png"), dtype=np.uint8)
    instruction = "Pick up red box."
    action = client.query_action(obs, instruction) # delta: xyz, rpy, openness
    print("unnormalized action: ", action)
