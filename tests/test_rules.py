"""Optional score rules: off by default, each moves the score the intended way."""

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import evolve, fitness, rules
from darwinbot import morphology as mo
from darwinbot import terrain as te
from darwinbot.sim import Episode, Metrics


@pytest.fixture(autouse=True)
def reset():
    rules.set_active(())
    yield
    rules.set_active(())


def _m(dx=0.3, dy=0.0, yaw=0.0, **kw):
    return Metrics(dx=dx, dy=dy, energy=0.0, fell=False, unstable=False,
                   duration=8.0, mass=1.0, yaw=yaw, **kw)


def _steer(left, right, straight=None):
    return [straight or _m(), left, right]


def _turn(yaw_towards, dx=0.2):
    # command -1 (left) wants yaw > 0, command +1 (right) wants yaw < 0
    return _m(dx=dx, yaw=yaw_towards)


def test_names_default_off_and_archive():
    assert not any(rules.active(n) for n in rules.NAMES)
    with pytest.raises(ValueError):
        rules.normalise(["nonsense"])
    assert rules.normalise(["slip", "worse_turn"]) == ("worse_turn", "slip")
    assert evolve.Archive(terrains="flat").rules == ()
    a = evolve.Archive(terrains="flat")
    state = dict(a.__dict__)
    del state["rules"]
    b = evolve.Archive.__new__(evolve.Archive)
    b.__setstate__(state)
    assert b.rules == ()


def test_no_rule_keeps_current_score():
    ms = _steer(_turn(1.0), _turn(-0.2), _m(tilt=0.4, slip=2.0, servo_ground=1.0))
    base = fitness.task_score(ms, "steer")
    expected = 0.5 * fitness.score(ms[0], "forward") + 0.25 * (
        fitness.up(0.2, ms[1]) + fitness.up(0.2 * 0.2, ms[2]))
    assert base == pytest.approx(expected)
    assert fitness.score(_m(tilt=0.4, slip=2.0, servo_ground=1.0)) == pytest.approx(0.3)


def test_worse_turn_lowers_one_sided_robot():
    ms = _steer(_turn(1.0, dx=0.5), _turn(0.0, dx=0.5))   # turns left only
    base = fitness.task_score(ms, "steer")
    rules.set_active(["worse_turn"])
    assert fitness.task_score(ms, "steer") < base
    both = _steer(_turn(1.0, dx=0.5), _turn(-1.0, dx=0.5))
    rules.set_active(())
    b2 = fitness.task_score(both, "steer")
    rules.set_active(["worse_turn"])
    assert fitness.task_score(both, "steer") == pytest.approx(b2)   # symmetric robot unchanged


def test_turn_forward_drops_reverse_curve():
    rev = _steer(_turn(1.0, dx=-0.5), _turn(-1.0, dx=-0.5))
    fwd = _steer(_turn(1.0, dx=0.5), _turn(-1.0, dx=0.5))
    base_rev = fitness.task_score(rev, "steer")
    rules.set_active(["turn_forward"])
    assert fitness.task_score(rev, "steer") < base_rev
    rules.set_active(())
    base_fwd = fitness.task_score(fwd, "steer")
    rules.set_active(["turn_forward"])
    assert fitness.task_score(fwd, "steer") == pytest.approx(base_fwd)


@pytest.mark.parametrize("rule,field", [("tilt", "tilt"), ("slip", "slip"),
                                        ("servo_ground", "servo_ground")])
def test_penalty_rules(rule, field):
    clean, dirty = _m(), _m(**{field: 0.5})
    assert fitness.score(clean) == fitness.score(dirty)      # off: ignored
    rules.set_active([rule])
    assert fitness.score(dirty) < fitness.score(clean)
    assert fitness.score(clean) == pytest.approx(0.3)
    rules.set_active([rule])


def test_episode_measures_new_metrics():
    morph = mo.decode(mo.encode(mo.template(2, wheels=False, yaw=True)))
    ep = Episode(morph, te.TERRAINS["flat"](), seconds=1.5)
    ctrl = np.full(co.CTRL_LENGTH, 0.5)
    m = ep.run(ctrl)
    assert 0.0 <= m.tilt < 0.5
    assert m.slip >= 0.0 and np.isfinite(m.slip)
    assert 0.0 <= m.servo_ground <= 1.0
