import math
from darwinbot import fitness, rules
from darwinbot.sim import Metrics


def _m(dx, yaw=0.0, progress=None):
    return Metrics(dx=dx, dy=0.0, energy=0.0, fell=False, unstable=False, duration=8.0,
                   mass=1.0, yaw=yaw, progress=dx if progress is None else progress)


def _ms(dx_turn):
    straight = _m(3.0)
    turn = lambda c: _m(dx_turn, yaw=-c * math.pi / 2, progress=0.0)
    half = _m(1.5)
    return [straight, turn(-1.0), turn(1.0), half]


def _score(tight, dx_turn):
    old = (fitness.TURN_SCALE, fitness.TURN_IN_PLACE, fitness.TURN_TIGHT)
    try:
        fitness.TURN_SCALE, fitness.TURN_IN_PLACE, fitness.TURN_TIGHT = 0.5, True, tight
        return fitness.task_score(_ms(dx_turn), "steer")
    finally:
        fitness.TURN_SCALE, fitness.TURN_IN_PLACE, fitness.TURN_TIGHT = old


def test_off_by_default_changes_nothing():
    assert fitness.DEFAULT_TURN_TIGHT == 0.0
    assert _score(0.0, 0.0) == _score(0.0, 1.0)


def test_pivot_earns_more_than_wide_arc():
    assert _score(1.0, 0.0) > _score(1.0, 1.0)


def test_wide_arc_gets_no_bonus():
    assert math.isclose(_score(1.0, 1.0), _score(0.0, 1.0))


def test_stored_and_applied():
    assert fitness.score_options(turn_tight=1.0)["turn_tight"] == 1.0
    old = fitness.TURN_TIGHT
    try:
        fitness.apply_score({"turn_tight": 0.7})
        assert fitness.TURN_TIGHT == 0.7
    finally:
        fitness.apply_score({}); fitness.TURN_TIGHT = old
