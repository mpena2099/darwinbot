"""
test_body_ground.py
-------------------
The share of control ticks the body itself (not a foot, wheel or tip) touches
the ground: measured in the Python loop and in the compiled one, near 1 for a
body lying on its torso, near 0 for one standing on its wheels, and the reason
a wheeled robot that drags its torso is named "belly" and no longer "roll".
"""

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import evolve
from darwinbot import fastloop
from darwinbot import morphology as mo
from darwinbot.sim import Behaviour, Episode
from darwinbot.terrain import flat

LEGS = mo.encode(mo.template(2, wheels=False, yaw=True))
WHEELS = mo.encode(mo.template(1, wheels=True, yaw=True, wheel_rad=0.06))


def _ctrl(morph, amp=0.0, wheel=0.5, sigma=0.5):
    """Mid genes everywhere; joints get `amp` on both harmonics, wheels `wheel`."""
    ctrl = np.full(co.CTRL_LENGTH, 0.5)
    for act in morph.layout():
        b = co.node_base(act.slot)
        if act.kind == "wheel":
            ctrl[b + co.OFFSET] = wheel
        else:
            ctrl[b + co.AMP1] = amp
            ctrl[b + co.AMP2] = amp
            ctrl[b + co.SIGMA] = sigma
    return ctrl


@pytest.mark.parametrize("genes", [LEGS, WHEELS])
def test_compiled_loop_counts_the_same_body_ground(monkeypatch, genes):
    """The bodies of tests/test_fastloop.py: the compiled loop counts the same
    ticks as the Python loop."""
    if fastloop._info() is None:
        pytest.skip("compiled loop unavailable on this platform")
    morph = mo.decode(genes)
    ep = Episode(morph, flat(), 2.0)
    ctrl = co.random_controller(np.random.default_rng(5))
    monkeypatch.setenv("DARWINBOT_PY_LOOP", "1")
    ref = ep.run(ctrl)
    monkeypatch.delenv("DARWINBOT_PY_LOOP")
    assert fastloop.prepare(ep) is not None
    fast = ep.run(ctrl)
    assert fast.body_ground == ref.body_ground
    assert 0.0 <= ref.body_ground <= 1.0


def test_a_body_on_its_torso_and_one_on_its_wheels():
    # no limbs at all: the torso itself rests on the ground
    flat_body = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    on_torso = Episode(flat_body, flat(), 2.0)
    assert on_torso.buildable
    m = on_torso.run(co.random_controller(np.random.default_rng(0)))
    assert m.body_ground > 0.8
    assert m.duty == 0.0             # no terminal point ever touches

    # wheels only, no drive: it stands on them and its body stays clear
    wheels = mo.decode(WHEELS)
    on_wheels = Episode(wheels, flat(), 2.0)
    m = on_wheels.run(_ctrl(wheels, amp=0.0, wheel=0.5))
    assert m.duty > 0.9 and m.wheel_share > 0.5
    assert m.body_ground < 0.05


def test_type_name_prefers_belly_over_roll():
    rolling = Behaviour(0.6, 0.3, 8.0, 0.9, 1.0, 4.0, 0.8)
    assert evolve.type_name(rolling).endswith("-belly")
    sliding = Behaviour(0.6, 0.3, 8.0, 0.9, 1.0, 4.0, 0.1)
    assert evolve.type_name(sliding).endswith("-roll")


def test_behaviour_of_takes_six_or_seven_values():
    old = Behaviour.of([0.6, 0.3, 8.0, 0.1, 2.5, 4.0])
    assert len(old) == 7 and old.body_ground == 0.0
    assert (old.duty, old.height, old.servos, old.wheel_share, old.cot, old.legs) == (
        0.6, 0.3, 8.0, 0.1, 2.5, 4.0)
    new = Behaviour.of([0.6, 0.3, 8.0, 0.1, 2.5, 4.0, 0.7])
    assert new.body_ground == 0.7
    assert Behaviour.of(old) is old
