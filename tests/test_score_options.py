"""Score settings are train.py options stored in the archive, not only env vars."""

import importlib.util
import math
import os
import pickle
import sys

import pytest

from darwinbot import evolve, fitness, rules
from darwinbot.sim import Metrics

ENV = ("DARWINBOT_BASAL_SCALE", "DARWINBOT_TURN_SCALE",
       "DARWINBOT_TURN_IN_PLACE", "DARWINBOT_TURN_CAP_DEG")


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    yield
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    rules.apply_score({})
    fitness.apply_score({})


def _m(dx=0.5, dy=0.0, yaw=0.0):
    return Metrics(dx=dx, dy=dy, energy=0.0, fell=False, unstable=False,
                   duration=8.0, mass=1.0, yaw=yaw, n_servos=6)


def _steer():
    return [_m(), _m(dx=0.2, yaw=1.0), _m(dx=0.2, yaw=-1.0), _m()]


def test_defaults_unchanged():
    assert evolve.score_options() == {}
    assert evolve.Archive(terrains="flat").score == {}
    assert rules.BASAL_SCALE == 1.0
    assert rules.BASAL_PER_SERVO == pytest.approx(0.01)
    assert rules.BASAL_PER_KG == pytest.approx(0.05)
    assert fitness.TURN_SCALE == pytest.approx(0.2)
    assert fitness.TURN_IN_PLACE is False
    assert fitness.TURN_CAP == pytest.approx(math.radians(90))


def test_flags_stored_and_survive_save_load():
    score = evolve.score_options(basal_scale=2.0, turn_scale=0.4,
                                 turn_in_place=True, turn_cap_deg=120)
    assert score == {"basal_scale": 2.0, "turn_scale": 0.4,
                     "turn_in_place": True, "turn_cap_deg": 120.0}
    a = evolve.Archive(terrains="flat", score=score)
    assert pickle.loads(pickle.dumps(a)).score == score
    assert a.like().score == score


def test_loaded_values_active_after_activate():
    score = {"basal_scale": 2.0, "turn_scale": 0.5,
             "turn_in_place": True, "turn_cap_deg": 10}
    rules.apply_score({})          # a fresh process state: module values at defaults
    fitness.apply_score({})
    a = pickle.loads(pickle.dumps(evolve.Archive(terrains="flat", score=score)))
    a.activate()
    assert rules.BASAL_PER_SERVO == pytest.approx(0.02)
    assert rules.BASAL_PER_KG == pytest.approx(0.1)
    assert fitness.TURN_SCALE == pytest.approx(0.5)
    assert fitness.TURN_IN_PLACE is True
    assert fitness.TURN_CAP == pytest.approx(math.radians(10))
    base = fitness.task_score(_steer(), "steer")
    evolve.apply_score({"turn_cap_deg": 120})
    assert fitness.task_score(_steer(), "steer") > base   # the cap changes the score


def test_env_overrides_stored_values(monkeypatch):
    for name, value in (("DARWINBOT_BASAL_SCALE", "3"),
                        ("DARWINBOT_TURN_SCALE", "0.7"),
                        ("DARWINBOT_TURN_IN_PLACE", "1"),
                        ("DARWINBOT_TURN_CAP_DEG", "45")):
        monkeypatch.setenv(name, value)
    a = evolve.Archive(terrains="flat", score={"basal_scale": 2.0, "turn_scale": 0.5,
                                              "turn_in_place": False, "turn_cap_deg": 10})
    a.activate()
    assert rules.BASAL_SCALE == 3.0
    assert rules.BASAL_PER_SERVO == pytest.approx(0.03)
    assert fitness.TURN_SCALE == pytest.approx(0.7)
    assert fitness.TURN_IN_PLACE is True
    assert fitness.TURN_CAP == pytest.approx(math.radians(45))
    # a new archive records the gate's values, so later runs can replay them
    assert evolve.score_options() == {"basal_scale": 3.0, "turn_scale": 0.7,
                                      "turn_in_place": True, "turn_cap_deg": 45.0}


def test_train_flags_reach_evolve_run_and_are_stored(tmp_path, monkeypatch):
    path = os.path.join(os.path.dirname(__file__), "..", "experiments", "train.py")
    spec = importlib.util.spec_from_file_location("train_score_options", path)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    calls = []
    monkeypatch.setattr(train.evolve, "run",
                        lambda **kw: calls.append(kw) or evolve.Archive(terrains="flat"))
    monkeypatch.setattr(sys, "argv", ["train.py", "--init", "1", "--batches", "0",
                                      "--batch-size", "1", "--workers", "1",
                                      "--variants", "0", "--seconds", "1",
                                      "--out", str(tmp_path), "--basal-scale", "2",
                                      "--turn-scale", "0.4", "--turn-in-place",
                                      "--turn-cap-deg", "120"])
    train.main()
    assert calls[-1]["score"] == {"basal_scale": 2.0, "turn_scale": 0.4,
                                  "turn_in_place": True, "turn_cap_deg": 120.0}


def test_old_archive_without_field_loads_with_defaults():
    a = evolve.Archive(terrains="flat", score={"basal_scale": 2.0})
    state = dict(a.__dict__)
    del state["score"]
    b = evolve.Archive.__new__(evolve.Archive)
    b.__setstate__(state)
    assert b.score == {}
    b.activate()
    assert rules.BASAL_SCALE == 1.0
    assert fitness.TURN_SCALE == pytest.approx(0.2)
