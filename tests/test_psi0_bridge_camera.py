import numpy as np
import pytest

from decoupled_wbc.control.sensor.sensor_server import ImageMessageSchema
from scripts.psi0_simple_real_bridge import build_parser, decode_camera_message
from tests.psi0_bridge_testkit import ManualClock


def encoded_sentinel(producer_timestamp=4.5, include_timestamp=True):
    image = np.zeros((32, 64, 3), np.uint8)
    image[:, :24] = [240, 0, 0]
    image[:, 40:] = [0, 0, 240]
    timestamps = (
        {"rgb_head_stereo_left": producer_timestamp} if include_timestamp else {}
    )
    return ImageMessageSchema(
        timestamps=timestamps,
        images={"rgb_head_stereo_left": image},
    ).serialize()


def test_parser_default_camera_color_order_is_rgb():
    parser = build_parser()
    assert parser.get_default("camera_color_order") == "rgb"


@pytest.mark.parametrize("color_order", ["rgb", "bgr"])
def test_codec_applies_configured_channel_transform_exactly_once(color_order):
    clock = ManualClock(500)
    frame = decode_camera_message(
        encoded_sentinel(),
        key="rgb_head_stereo_left",
        color_order=color_order,
        received_at=clock(),
    )
    assert frame.image.dtype == np.uint8
    assert frame.image.flags.c_contiguous
    assert frame.image.shape == (32, 64, 3)
    assert frame.received_at == 10.0
    assert frame.producer_timestamp == 4.5
    left = frame.image[:, :20].mean(axis=(0, 1))
    right = frame.image[:, 44:].mean(axis=(0, 1))
    if color_order == "rgb":
        assert left[0] > 200 and left[2] < 30
        assert right[2] > 200 and right[0] < 30
    else:
        assert left[2] > 200 and left[0] < 30
        assert right[0] > 200 and right[2] < 30


def test_camera_key_is_mandatory():
    with pytest.raises(KeyError, match="rgb_head_stereo_left"):
        decode_camera_message(
            encoded_sentinel(),
            key="missing",
            color_order="rgb",
            received_at=1.0,
        )


@pytest.mark.parametrize(
    "payload",
    [
        encoded_sentinel(include_timestamp=False),
        encoded_sentinel(producer_timestamp=np.nan),
        encoded_sentinel(producer_timestamp=np.inf),
        encoded_sentinel(producer_timestamp="not-a-time"),
        {**encoded_sentinel(), "timestamps": "malformed"},
    ],
)
def test_invalid_or_missing_producer_time_is_diagnostic_only(payload):
    frame = decode_camera_message(
        payload,
        key="rgb_head_stereo_left",
        color_order="rgb",
        received_at=10.0,
    )
    assert frame.received_at == 10.0
    assert frame.producer_timestamp is None
    assert frame.producer_timestamp_diagnostic is not None
