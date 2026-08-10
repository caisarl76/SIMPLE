from copy import deepcopy

import numpy as np

from simple.deploy.psi0_simple_bridge import (
    PSI0_ACTION_JOINT_NAMES,
    JointContract,
    PolicyContract,
    RtcResult,
    TimedCameraFrame,
    TimedRobotState,
)


class ManualClock:
    def __init__(self, tick=0):
        self.tick = tick

    def __call__(self):
        return self.tick / 50.0

    def set_tick(self, tick):
        self.tick = tick


def policy_payload(**updates):
    payload = {
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
        "prediction_horizon": 8,
        "execution_horizon": 5,
        "rtc_delay_steps": 3,
        "rtc_training_max_delay": 4,
        "rtc_enabled": True,
        "rtc_endpoint": "/act-rtc-v1",
        "request_semantics": "exact-post-slew-committed-prefix",
        "response_semantics": "denormalized-executable-suffix",
        "image_key": "rgb_head_stereo_left",
        "camera_color_order": "rgb",
    }
    payload.update(updates)
    return payload


def make_policy_contract(**updates):
    return PolicyContract.from_dict(policy_payload(**updates))


def make_joint_contract():
    names = (
        *(f"leg_joint_{index}" for index in range(12)),
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        *PSI0_ACTION_JOINT_NAMES,
    )
    return JointContract(
        names,
        names[12:],
        np.full(43, -2.0, np.float32),
        np.full(43, 2.0, np.float32),
    )


def fresh_inputs(clock):
    return (
        TimedRobotState(np.zeros(43, np.float32), clock()),
        TimedCameraFrame(np.zeros((8, 8, 3), np.uint8), clock(), None),
    )


def sentinel_actions(first_tick, count):
    actions = np.empty((count, 36), np.float32)
    for row, global_tick in enumerate(range(first_tick, first_tick + count)):
        actions[row] = 0.30 + 0.001 * (global_tick - 100) + 0.000001 * np.arange(36)
    actions[:, 31] = 0.50
    actions[:, 32:36] = 0.0
    return actions


class ImmediateInference:
    def __init__(self, clock, contract):
        self.clock = clock
        self.contract = contract
        self.requests = []
        self.pending = []
        self.submissions = 0

    @property
    def busy(self):
        return False

    def submit(self, request):
        self.submissions += 1
        self.requests.append(deepcopy(request))
        first = request.observation_tick + self.contract.rtc_delay_steps
        metadata = {
            "session_id": request.session_id,
            "request_seq": request.request_seq,
            "observation_tick": request.observation_tick,
            "prediction_horizon": self.contract.prediction_horizon,
            "execution_horizon": self.contract.execution_horizon,
            "rtc_delay_steps": self.contract.rtc_delay_steps,
            "first_action_tick": first,
        }
        self.pending.append(
            RtcResult(
                generation=request.generation,
                request_seq=request.request_seq,
                completed_at=self.clock(),
                actions=sentinel_actions(first, self.contract.execution_horizon),
                metadata=metadata,
            )
        )

    def poll(self):
        return self.pending.pop(0) if self.pending else None
