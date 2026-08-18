from simple.tasks.g1_wholebody_xmove_bend_pick_teleop import (
    G1WholebodyXMoveBendPickTaskTeleop,
)


def test_xmove_bend_pick_uid_matches_official_eval_archive() -> None:
    assert (
        G1WholebodyXMoveBendPickTaskTeleop.uid == "g1_wholebody_xmove_bend_pick_teleop"
    )
