"""
test_drag_options.py
--------------------
The two optional experiments about robots that slide on their body: the score
rule `body_ground` and the sixth archive axis (`train.py --axes body`). Both are
off by default, and a default run is unchanged.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

from darwinbot import evolve, fitness, rules
from darwinbot.sim import Metrics


def _run(out, **kw):
    return evolve.run(terrains="flat", n_init=2, n_batches=1, batch_size=2,
                      budget_init=4, budget_child=3, workers=1, seed=7,
                      episode_seconds=0.4, variants=0, out_dir=str(out), **kw)


def _m(body_ground=0.0):
    return Metrics(dx=0.3, dy=0.0, energy=0.0, fell=False, unstable=False,
                   duration=8.0, mass=1.0, body_ground=body_ground)


@pytest.fixture(autouse=True)
def reset():
    rules.set_active(())
    yield
    rules.set_active(())


def test_body_ground_rule_costs_its_share():
    m = _m(body_ground=0.5)
    off = fitness.rule_cost(m)
    assert off == fitness.rule_cost(_m()) == 0.0
    rules.set_active(("body_ground",))
    assert fitness.rule_cost(m) == pytest.approx(off + 0.2 * m.body_ground)
    # the score pays it too, like the other per-episode rules
    rules.set_active(())
    base = fitness.score(m)
    rules.set_active(("body_ground",))
    assert fitness.score(m) == pytest.approx(base - 0.2 * m.body_ground)


def test_body_axes_run_creates_a_six_axis_archive(tmp_path):
    archive = _run(tmp_path / "a", axes="body")
    assert archive.axes == "body" and archive.n_descriptors == 6
    assert archive._grid.measure_dim == 6
    assert archive.cells
    path = tmp_path / "a" / "archive.pkl"
    loaded = evolve.load_archive(str(path))
    assert loaded.axes == "body" and loaded._grid.measure_dim == 6
    # resuming it needs no flag, and the flag on the other archive is refused
    again = evolve.run(resume=str(path), n_batches=0, workers=1, seed=7,
                       out_dir=str(tmp_path / "b"))
    assert again.n_descriptors == 6
    default = _run(tmp_path / "c")
    with pytest.raises(ValueError, match="axes"):
        _run(tmp_path / "d", resume=str(tmp_path / "c" / "archive.pkl"),
             axes="body")
    assert default.n_descriptors == 5


def test_default_run_is_five_axes_and_unchanged(tmp_path):
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b", axes="default")
    assert a.n_descriptors == 5 and a._grid.measure_dim == 5
    assert a.stats() == b.stats()
    ca, cb = a.cells, b.cells
    assert ca.keys() == cb.keys()
    for k in ca:
        assert ca[k].fitness == cb[k].fitness
        assert np.array_equal(ca[k].morph, cb[k].morph)
        assert np.array_equal(ca[k].ctrl, cb[k].ctrl)


def test_merge_refuses_archives_with_different_axes():
    with pytest.raises(ValueError, match="axes"):
        evolve.merge([evolve.Archive(terrains="flat"),
                      evolve.Archive(terrains="flat", axes="body")])


TRAIN_ARGS = ["train.py", "--init", "1", "--batches", "0", "--batch-size", "1",
              "--workers", "1", "--variants", "0", "--seconds", "1"]


def _train():
    path = os.path.join(os.path.dirname(__file__), "..", "experiments", "train.py")
    spec = importlib.util.spec_from_file_location("train_drag_options", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_accepts_the_body_axes_and_rule(tmp_path, monkeypatch):
    train = _train()
    calls = []
    monkeypatch.setattr(train.evolve, "run",
                        lambda **kw: calls.append(kw) or evolve.Archive(terrains="flat"))
    monkeypatch.setattr(sys, "argv", TRAIN_ARGS + [
        "--out", str(tmp_path), "--axes", "body", "--rules", "body_ground"])
    train.main()
    assert calls[-1]["axes"] == "body"
    assert calls[-1]["score_rules"] == ("body_ground",)


def test_fall_zero_rule_drops_the_progress_of_a_fallen_episode():
    def ep(fell, progress=1.0):
        return Metrics(dx=1.0, dy=0.0, energy=0.0, fell=fell, unstable=False,
                       duration=8.0, mass=1.0, progress=progress)
    fallen, walked = ep(True), ep(False)
    rules.set_active(())
    assert fitness.score(fallen) == pytest.approx(1.0 - fitness.FALL_PENALTY)
    assert fitness.score(walked) == pytest.approx(1.0)
    rules.set_active(("fall_zero",))
    assert fitness.score(fallen) == pytest.approx(-fitness.FALL_PENALTY)
    assert fitness.score(walked) == pytest.approx(1.0)
    # turn task: the heading change of a fallen episode earns nothing either
    turner = Metrics(dx=0.0, dy=0.0, energy=0.0, fell=True, unstable=False,
                     duration=8.0, mass=1.0, yaw=2.0)
    assert fitness.score(turner, "turn") == pytest.approx(-fitness.FALL_PENALTY)
    # steer task: a fallen straight episode loses its progress credit
    on = fitness.task_score([ep(True), walked, walked, ep(False, progress=0.5)], "steer")
    rules.set_active(())
    off = fitness.task_score([ep(True), walked, walked, ep(False, progress=0.5)], "steer")
    assert on < off


def test_energy_abs_rule_charges_the_energy_against_a_fixed_mass():
    from darwinbot import fitness, rules
    from darwinbot.sim import Metrics
    m = Metrics(dx=1.0, dy=0.0, fell=False, unstable=False, duration=8.0, energy=20.0, mass=2.0)
    rules.set_active(())
    base = fitness.rule_cost(m)
    rules.set_active(("energy_abs",))
    try:
        assert fitness.rule_cost(m) == pytest.approx(
            base + fitness.ENERGY_ABS_W * 20.0 / (fitness.M_REF * fitness.G))
    finally:
        rules.set_active(())
