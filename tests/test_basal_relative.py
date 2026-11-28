"""Relative basal: a positive score keeps a share set by the body's parts."""

import importlib.util
import os
import pickle
import sys

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import evolve, fitness, rules
from darwinbot import morphology as mo
from darwinbot import terrain as te
from darwinbot.sim import Episode, Metrics

ENV = ("DARWINBOT_BASAL_SCALE", "DARWINBOT_BASAL_MODE",
       "DARWINBOT_BASAL_SERVO_SHARE", "DARWINBOT_BASAL_PART_SHARE")


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    rules.apply_score({})
    fitness.apply_score({})
    rules.set_active(())
    yield
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    rules.apply_score({})
    fitness.apply_score({})
    rules.set_active(())


def _m(dx=0.5, n_servos=0, n_parts=0, mass=1.0, **kw):
    return Metrics(dx=dx, dy=0.0, energy=0.0, fell=False, unstable=False,
                   duration=8.0, mass=mass, n_servos=n_servos, n_parts=n_parts, **kw)


def test_absolute_default_is_unchanged():
    m = _m(n_servos=6, n_parts=8, mass=1.5)
    base = fitness.task_score([m], "forward")
    rules.set_active(["basal"])
    cost = rules.BASAL_PER_SERVO * 6 + rules.BASAL_PER_KG * 1.5
    assert rules.BASAL_MODE == "absolute"
    assert evolve.score_options() == {}
    assert fitness.task_score([m], "forward") == pytest.approx(base - cost)


def test_relative_factor_on_a_known_body():
    m = _m(n_servos=6, n_parts=8)        # 8 modules, 2 of them without a servo
    base = fitness.task_score([m], "forward")
    assert base > 0
    rules.set_active(["basal"])
    evolve.apply_score(evolve.score_options(basal_mode="relative"))
    f = 6 * rules.BASAL_SERVO_SHARE + 2 * rules.BASAL_PART_SHARE
    assert fitness.task_score([m], "forward") == pytest.approx(base * (1 - f))


def test_relative_factor_is_capped():
    m = _m(n_servos=25, n_parts=25)
    base = fitness.task_score([m], "forward")
    rules.set_active(["basal"])
    evolve.apply_score(evolve.score_options(basal_mode="relative"))
    assert rules.BASAL_REL_CAP == 0.8
    assert fitness.task_score([m], "forward") == pytest.approx(
        base * (1 - rules.BASAL_REL_CAP))


def test_non_positive_scores_are_untouched():
    rules.set_active(["basal"])
    evolve.apply_score(evolve.score_options(basal_mode="relative",
                                            basal_servo_share=0.1))
    assert fitness.task_score([_m(dx=0.0, n_servos=6, n_parts=8)], "forward") == 0.0
    assert fitness.task_score([_m(dx=-0.4, n_servos=6, n_parts=8)], "forward") == pytest.approx(-0.4)


def test_relative_once_per_steer_evaluation():
    ms = [_m(n_servos=6, n_parts=8), _m(dx=0.2, yaw=1.0, n_servos=6, n_parts=8),
          _m(dx=0.2, yaw=-1.0, n_servos=6, n_parts=8), _m(n_servos=6, n_parts=8)]
    base = fitness.task_score(ms, "steer")
    rules.set_active(["basal"])
    evolve.apply_score(evolve.score_options(basal_mode="relative"))
    f = 6 * rules.BASAL_SERVO_SHARE + 2 * rules.BASAL_PART_SHARE
    assert fitness.task_score(ms, "steer") == pytest.approx(base * (1 - f))


def test_mode_without_the_rule_changes_nothing():
    m = _m(n_servos=6, n_parts=8)
    base = fitness.task_score([m], "forward")
    evolve.apply_score(evolve.score_options(basal_mode="relative"))
    assert fitness.task_score([m], "forward") == pytest.approx(base)


def test_settings_are_stored_and_restored():
    score = evolve.score_options(basal_scale=2.0, basal_mode="relative",
                                 basal_servo_share=0.05, basal_part_share=0.03)
    assert score == {"basal_scale": 2.0, "basal_mode": "relative",
                     "basal_servo_share": 0.05, "basal_part_share": 0.03}
    a = evolve.Archive(terrains="flat", score=score)
    assert pickle.loads(pickle.dumps(a)).score == score
    assert a.like().score == score
    rules.apply_score({})
    a.activate()
    assert rules.BASAL_PER_SERVO == pytest.approx(0.02)
    assert rules.BASAL_MODE == "relative"
    assert rules.BASAL_SERVO_SHARE == pytest.approx(0.05)
    assert rules.BASAL_PART_SHARE == pytest.approx(0.03)


def test_episode_fills_n_parts():
    morph = mo.decode(mo.encode(mo.template(2, wheels=False, yaw=True)))
    ep = Episode(morph, te.TERRAINS["flat"](), seconds=0.2)
    m = ep.run(np.full(co.CTRL_LENGTH, 0.5))
    assert m.n_parts == morph.n_nodes - 1
    assert m.n_parts > 0


def test_train_flags_reach_the_archive(tmp_path, monkeypatch):
    path = os.path.join(os.path.dirname(__file__), "..", "experiments", "train.py")
    spec = importlib.util.spec_from_file_location("train_basal_relative", path)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    calls = []
    monkeypatch.setattr(train.evolve, "run",
                        lambda **kw: calls.append(kw) or evolve.Archive(terrains="flat"))
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--init", "1", "--batches", "0", "--batch-size", "1",
        "--workers", "1", "--variants", "0", "--seconds", "1",
        "--out", str(tmp_path), "--rules", "basal", "--basal-mode", "relative",
        "--basal-servo-share", "0.05", "--basal-part-share", "0.03"])
    train.main()
    assert calls[-1]["score"] == {"basal_mode": "relative",
                                  "basal_servo_share": 0.05,
                                  "basal_part_share": 0.03}
