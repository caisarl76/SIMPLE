import numpy as np
import pytest

from simple.deploy.psi0_simple_bridge import (
    build_psi0_observation,
    goal_to_psi0_action,
    map_psi0_action_to_goal,
)


EXPECTED_LIMB_NAMES = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
EXPECTED_UPPER_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    *EXPECTED_LIMB_NAMES,
)
TEST_JOINT_NAMES = (
    *(f"leg_joint_{index}" for index in range(12)),
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    *EXPECTED_LIMB_NAMES,
)


def test_state_mapping_is_exactly_named():
    q = np.arange(43, dtype=np.float32)
    q_by_name = dict(zip(TEST_JOINT_NAMES, q, strict=True))
    state = build_psi0_observation(
        q, TEST_JOINT_NAMES, np.array([101, 102, 103, 0.74], np.float32)
    )
    np.testing.assert_array_equal(
        state[0, :28], [q_by_name[name] for name in EXPECTED_LIMB_NAMES]
    )
    np.testing.assert_array_equal(
        state[0, 28:], np.asarray([101, 102, 103, 0.74], np.float32)
    )


def test_action_mapping_and_inverse_are_exactly_named():
    action = np.arange(36, dtype=np.float32) / 100.0
    connected_upper_names = (
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "right_shoulder_yaw_joint",
        "left_hand_index_1_joint",
        *tuple(
            name
            for name in EXPECTED_LIMB_NAMES
            if name not in {"right_shoulder_yaw_joint", "left_hand_index_1_joint"}
        ),
    )
    assert len(connected_upper_names) == 31
    goal = map_psi0_action_to_goal(action, connected_upper_names, now=12.5)
    by_name = dict(zip(EXPECTED_LIMB_NAMES, action[:28], strict=True))
    by_name.update(
        {
            "waist_roll_joint": action[28],
            "waist_pitch_joint": action[29],
            "waist_yaw_joint": action[30],
        }
    )
    np.testing.assert_array_equal(
        goal.target_upper_body_pose,
        np.asarray([by_name[name] for name in connected_upper_names], np.float32),
    )
    np.testing.assert_array_equal(goal.base_height_command, action[31:32])
    np.testing.assert_array_equal(goal.navigate_cmd, action[32:36])
    assert goal.timestamp == 12.5
    assert goal.target_time == 12.52
    np.testing.assert_array_equal(
        goal_to_psi0_action(goal, connected_upper_names), action
    )


@pytest.mark.parametrize(
    "bad_names",
    [
        EXPECTED_UPPER_NAMES[:-1],
        EXPECTED_UPPER_NAMES[:-1] + (EXPECTED_UPPER_NAMES[0],),
        EXPECTED_UPPER_NAMES[:-1] + ("unknown_joint",),
    ],
)
def test_action_mapping_rejects_missing_duplicate_or_unknown_names(bad_names):
    with pytest.raises(ValueError, match="one-to-one 31-joint upper body"):
        map_psi0_action_to_goal(np.zeros(36, np.float32), bad_names, now=0.0)
