"""
test_sim.py
-----------
MuJoCo episode construction, standing, wheel drive and determinism.
"""

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import fitness
from darwinbot import morphology as mo
from darwinbot import terrain as te
from darwinbot.sim import Episode, Metrics


def _scale(value, lo, hi):
    return (value - lo) / (hi - lo)


def _genome(pairs):
    genes = np.zeros(mo.MORPH_LENGTH)
    genes[0] = _scale(0.40, *mo.BODY_BOUNDS[0])
    genes[1] = _scale(0.20, *mo.BODY_BOUNDS[1])
    for p, (x, thigh, shin, wheeled, wheel_rad) in enumerate(pairs):
        b = 2 + p * mo.PAIR_GENES
        genes[b] = 1.0
        genes[b + 1] = _scale(x, *mo.PAIR_BOUNDS[1])
        genes[b + 2] = _scale(0.0, *mo.PAIR_BOUNDS[2])
        genes[b + 3] = _scale(thigh, *mo.PAIR_BOUNDS[3])
        genes[b + 4] = _scale(shin, *mo.PAIR_BOUNDS[4])
        genes[b + 5] = _scale(0.02, *mo.PAIR_BOUNDS[5])
        genes[b + 6] = 1.0 if wheeled else 0.0
        genes[b + 7] = _scale(wheel_rad, *mo.PAIR_BOUNDS[7])
    return genes


LEGS = _genome([(0.4, 0.11, 0.11, False, 0.05),
                (-0.4, 0.11, 0.11, False, 0.05)])
WHEELS = _genome([(0.4, 0.10, 0.10, True, 0.06),
                  (-0.4, 0.10, 0.10, True, 0.06)])


def _ctrl(morph, amp=0.0, wheel=0.5):
    ctrl = np.full(co.CTRL_LENGTH, 0.5)
    ctrl[0] = 0.5
    for pair in morph.pairs:
        b = 1 + pair.index * co.PAIR_CTRL
        for j in range(len(co.JOINTS)):
            ctrl[b + j * 3] = amp
            ctrl[b + j * 3 + 1] = 0.5
            ctrl[b + j * 3 + 2] = 0.5
        ctrl[b + co.PAIR_CTRL - 2] = 0.5
        ctrl[b + co.PAIR_CTRL - 1] = wheel
    return ctrl


@pytest.mark.parametrize("name", list(te.TERRAINS))
def test_every_terrain_compiles(name):
    episode = Episode(mo.decode(LEGS), te.TERRAINS[name](), seconds=1.0)
    assert episode.model.nu == len(episode.layout)


def test_zero_amplitude_stands_still():
    morph = mo.decode(LEGS)
    episode = Episode(morph, te.flat(), seconds=1.0)
    metrics = episode.run(_ctrl(morph))

    assert not metrics.fell and not metrics.unstable
    assert abs(metrics.dx) < 0.02


def test_wheels_drive_forward():
    morph = mo.decode(WHEELS)
    episode = Episode(morph, te.flat(), seconds=2.0)
    metrics = episode.run(_ctrl(morph, wheel=0.9))

    assert not metrics.fell and not metrics.unstable
    assert metrics.dx > 0.1


def test_runs_are_deterministic():
    morph = mo.decode(WHEELS)
    ctrl = _ctrl(morph, wheel=0.7)
    first = Episode(morph, te.flat(), seconds=1.0).run(ctrl)
    second = Episode(morph, te.flat(), seconds=1.0).run(ctrl)

    assert first == second


def _metrics(**overrides):
    fields = dict(dx=0.0, dy=0.0, energy=0.0, fell=False, unstable=False,
                  duration=1.0, mass=1.0)
    fields.update(overrides)
    return Metrics(**fields)


def test_unstable_gets_unstable_score():
    assert fitness.score(_metrics(unstable=True, dx=5.0)) == fitness.UNSTABLE_SCORE


def test_fall_applies_penalty():
    upright = fitness.score(_metrics(dx=1.0))
    fallen = fitness.score(_metrics(dx=1.0, fell=True))
    assert upright - fallen == pytest.approx(fitness.FALL_PENALTY)


def test_aggregate_blends_mean_and_min():
    scores = [1.0, 2.0, 6.0]
    expected = 0.7 * np.mean(scores) + 0.3 * min(scores)
    assert fitness.aggregate(scores) == pytest.approx(expected)
