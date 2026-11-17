"""Forward progress counts only the second half of the episode, scaled by 2."""

from darwinbot import fitness
from darwinbot.sim import Metrics


def _m(dx, progress=None):
    return Metrics(dx=dx, dy=0.0, energy=0.0, fell=False, unstable=False,
                   duration=5.0, mass=1.0, progress=progress)


def test_default_progress_is_dx():
    assert _m(0.3).progress == 0.3
    assert fitness.score(_m(0.3)) == 0.3


def test_early_jump_then_still_scores_zero():
    # x_mid == x_end: all of dx came in the first half
    m = _m(0.28, progress=2 * (0.28 - 0.28))
    assert fitness.score(m) == 0.0


def test_constant_speed_keeps_progress():
    v, T = 0.1, 5.0
    x_mid, x_end = v * T / 2, v * T
    m = _m(x_end, progress=2 * (x_end - x_mid))
    assert abs(m.progress - m.dx) < 1e-12
    assert abs(fitness.score(m) - x_end) < 1e-12
