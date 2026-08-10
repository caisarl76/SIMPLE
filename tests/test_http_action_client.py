from copy import deepcopy

import numpy as np
import pytest
import requests

from simple.baselines.client import (
    HttpActionClient,
    RequestMessage,
    convert_numpy_in_dict,
    numpy_serialize,
)


RTC_METADATA = {
    "session_id": "s",
    "request_seq": 0,
    "observation_tick": 100,
    "prediction_horizon": 30,
    "execution_horizon": 24,
    "rtc_delay_steps": 6,
    "first_action_tick": 106,
}


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = "fake response"

    def json(self):
        return deepcopy(self.payload)

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(str(self.status_code))


class RecordingSession:
    def __init__(self):
        self.calls = []
        self.post_response = FakeResponse(
            convert_numpy_in_dict(
                {
                    "action": np.zeros((24, 36), np.float32),
                    "metadata": RTC_METADATA,
                },
                numpy_serialize,
            )
        )
        self.get_response = FakeResponse({"schema": "simple.psi0.policy-contract.v2"})

    def post(self, url, *, json, timeout):
        self.calls.append(
            {"method": "POST", "url": url, "json": json, "timeout": timeout}
        )
        return self.post_response

    def get(self, url, *, timeout):
        self.calls.append({"method": "GET", "url": url, "timeout": timeout})
        return self.get_response


@pytest.fixture
def recording_session():
    return RecordingSession()


def test_legacy_query_keeps_unbounded_default(recording_session):
    client = HttpActionClient("policy", 22085, session=recording_session)
    client.query_action({}, "instruction", {}, {})
    assert recording_session.calls[0]["url"].endswith("/act")
    assert recording_session.calls[0]["timeout"] is None


def test_r0_serialization_is_exact_and_response_metadata_is_preserved(
    recording_session,
):
    client = HttpActionClient("policy", 22085, timeout=5.0, session=recording_session)
    image = np.zeros((4, 4, 3), np.uint8)
    state = np.arange(32, dtype=np.float32)[None]
    committed = np.arange(6 * 36, dtype=np.float32).reshape(6, 36)
    response = client.query_rtc_action(
        {"rgb_head_stereo_left": image},
        "pick up the object",
        {"states": state},
        {},
        history={
            "reset": True,
            "session_id": "s",
            "request_seq": 0,
            "observation_tick": 100,
            "rtc_delay_steps": 6,
            "committed_actions": committed,
        },
        dataset="simple",
    )
    call = recording_session.calls[0]
    request = RequestMessage.deserialize(call["json"])
    assert call["url"] == "http://policy:22085/act-rtc-v1"
    assert call["timeout"] == 5.0
    assert set(call["json"]) == {
        "image",
        "instruction",
        "history",
        "state",
        "condition",
        "gt_action",
        "dataset_name",
        "timestamp",
    }
    assert set(request.image) == {"rgb_head_stereo_left"}
    np.testing.assert_array_equal(request.image["rgb_head_stereo_left"], image)
    assert set(request.state) == {"states"}
    np.testing.assert_array_equal(request.state["states"], state)
    assert request.condition == {}
    assert request.gt_action == []
    assert request.dataset_name == "simple"
    assert request.instruction == "pick up the object"
    assert set(request.history) == {
        "reset",
        "session_id",
        "request_seq",
        "observation_tick",
        "rtc_delay_steps",
        "committed_actions",
    }
    assert request.history["reset"] is True
    np.testing.assert_array_equal(request.history["committed_actions"], committed)
    assert set(response.metadata) == set(RTC_METADATA)
    assert response.metadata == RTC_METADATA
    assert response.action.shape == (24, 36)


def test_successor_omits_reset_and_keeps_complete_history(recording_session):
    client = HttpActionClient("policy", 22085, timeout=5.0, session=recording_session)
    committed = np.full((6, 36), 0.125, np.float32)
    client.query_rtc_action(
        {"rgb_head_stereo_left": np.zeros((2, 2, 3), np.uint8)},
        "continue",
        {"states": np.zeros((1, 32), np.float32)},
        {},
        history={
            "session_id": "s",
            "request_seq": 1,
            "observation_tick": 124,
            "rtc_delay_steps": 6,
            "committed_actions": committed,
        },
        dataset="simple",
    )
    request = RequestMessage.deserialize(recording_session.calls[0]["json"])
    assert "reset" not in request.history
    assert set(request.history) == {
        "session_id",
        "request_seq",
        "observation_tick",
        "rtc_delay_steps",
        "committed_actions",
    }
    np.testing.assert_array_equal(request.history["committed_actions"], committed)


def test_contract_timeout_is_explicit(recording_session):
    client = HttpActionClient("policy", 22085, session=recording_session)
    assert client.get_contract(timeout=2.0) == {
        "schema": "simple.psi0.policy-contract.v2"
    }
    assert recording_session.calls == [
        {
            "method": "GET",
            "url": "http://policy:22085/contract",
            "timeout": 2.0,
        }
    ]


def rtc_call(client):
    return client.query_rtc_action(
        {"rgb_head_stereo_left": np.zeros((2, 2, 3), np.uint8)},
        "instruction",
        {"states": np.zeros((1, 32), np.float32)},
        {},
        history={
            "reset": True,
            "session_id": "s",
            "request_seq": 0,
            "observation_tick": 100,
            "rtc_delay_steps": 6,
            "committed_actions": np.zeros((6, 36), np.float32),
        },
        dataset="simple",
    )


@pytest.mark.parametrize("error", [requests.ConnectTimeout(), requests.ReadTimeout()])
def test_transport_timeout_is_propagated(error):
    class RaisingSession:
        def post(self, *args, **kwargs):
            raise error

    with pytest.raises(type(error)):
        rtc_call(
            HttpActionClient("policy", 22085, timeout=5.0, session=RaisingSession())
        )


def test_non_200_response_is_rejected(recording_session):
    recording_session.post_response = FakeResponse({}, status_code=500)
    with pytest.raises(requests.HTTPError):
        rtc_call(
            HttpActionClient("policy", 22085, timeout=5.0, session=recording_session)
        )


def test_missing_metadata_is_not_synthesized(recording_session):
    recording_session.post_response = FakeResponse(
        convert_numpy_in_dict(
            {"action": np.zeros((24, 36), np.float32)}, numpy_serialize
        )
    )
    with pytest.raises(RuntimeError, match="metadata"):
        rtc_call(
            HttpActionClient("policy", 22085, timeout=5.0, session=recording_session)
        )


def test_malformed_json_is_rejected(recording_session):
    recording_session.post_response.json = lambda: (_ for _ in ()).throw(
        ValueError("bad json")
    )
    with pytest.raises(RuntimeError, match="bad json"):
        rtc_call(
            HttpActionClient("policy", 22085, timeout=5.0, session=recording_session)
        )
