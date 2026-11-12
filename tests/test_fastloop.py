"""The compiled episode loop (darwinbot/fastloop.py) against the Python loop
of Episode.run (DARWINBOT_PY_LOOP=1): same bodies, same controllers, same
Metrics. The two do the same operations in the same order, so they agree bit
for bit; the check allows 1e-9 relative on floats in case a platform's libm or
reduction order differs, and exact equality on the flags."""

import dataclasses
import warnings

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import fastloop
from darwinbot import morphology as mo
from darwinbot import randomize
from darwinbot.sim import Episode
from darwinbot.terrain import flat, get_terrains


def _random_body(seed):
    return mo.decode(mo.random_morphology(np.random.default_rng(seed)))


def _walker():
    return mo.decode(mo.encode(mo.template(2, wheels=False, yaw=True)))


def _wheeled():
    return mo.decode(mo.encode(mo.template(1, wheels=True, yaw=True, wheel_rad=0.06)))


def _faller():
    for seed in range(1, 200):      # a random body that tips over early
        m = _random_body(seed)
        if mo.violations(m) or m.n_joints < 2:
            continue
        ep = Episode(m, flat(), 8.0)
        if not ep.buildable:
            continue
        if ep.run(co.random_controller(np.random.default_rng(seed))).fell:
            return m, seed
    pytest.skip("no falling body found")


def _compare(a, b):
    da, db = dataclasses.asdict(a), dataclasses.asdict(b)
    for key in da:
        if isinstance(da[key], (bool, int)) or da[key] is None:
            assert da[key] == db[key], key
        else:
            assert da[key] == pytest.approx(db[key], rel=1e-9, abs=1e-12), key


def _both(monkeypatch, ep, ctrl, variant=None, command=0.0, speed=1.0):
    monkeypatch.setenv("DARWINBOT_PY_LOOP", "1")
    ref = ep.run(ctrl, variant, command, speed)
    monkeypatch.delenv("DARWINBOT_PY_LOOP")
    assert fastloop.prepare(ep) is not None
    return ref, ep.run(ctrl, variant, command, speed)


@pytest.fixture(autouse=True)
def _needs_fast():
    if fastloop._info() is None:
        pytest.skip("compiled loop unavailable on this platform")


def _case(name):
    if name == "walker":
        return _walker(), flat(), 0.0, 1, None
    if name == "wheeled":
        return _wheeled(), flat(), 0.0, 2, None
    if name == "faller":
        m, seed = _faller()
        return m, flat(), 0.0, seed, None
    if name == "payload":
        return _walker(), flat(), 0.8, 3, None
    if name == "variant":
        return _random_body(2), flat(), 0.0, 4, 2
    if name == "rough":
        return _random_body(4), get_terrains("rough", seed=1)[0], 0.0, 5, None
    if name == "rubble-variant":
        return _random_body(1), get_terrains("rubble", seed=2)[0], 0.0, 6, 1
    raise KeyError(name)


@pytest.mark.parametrize("name", ["walker", "wheeled", "faller", "payload",
                                  "variant", "rough", "rubble-variant"])
def test_same_metrics_as_the_python_loop(monkeypatch, name):
    morph, terrain, payload, seed, latency = _case(name)
    ep = Episode(morph, terrain, 4.0, payload)
    assert ep.buildable
    rng = np.random.default_rng(seed)
    ctrl = co.random_controller(rng)
    variant = None
    if latency is not None:
        variant = randomize.perturb(ep.model, rng)
        variant = randomize.Variant(variant.model, latency)
    for command, speed in ((0.0, 1.0), (-1.0, 1.0), (0.7, 0.4)):
        ref, fast = _both(monkeypatch, ep, ctrl, variant, command, speed)
        _compare(ref, fast)


def test_episode_can_be_rerun_and_k_is_kept(monkeypatch):
    ep = Episode(_walker(), flat(), 2.0)
    ctrl = co.random_controller(np.random.default_rng(9))
    ref, fast = _both(monkeypatch, ep, ctrl)
    k_fast = ep._k
    monkeypatch.setenv("DARWINBOT_PY_LOOP", "1")
    ep.run(ctrl)
    assert ep._k == k_fast
    _compare(ref, fast)
    monkeypatch.delenv("DARWINBOT_PY_LOOP")
    _compare(ref, ep.run(ctrl))


def test_falls_back_with_one_warning(monkeypatch):
    ep = Episode(_walker(), flat(), 1.0)
    ctrl = co.random_controller(np.random.default_rng(1))
    monkeypatch.setitem(fastloop._state, "done", False)
    monkeypatch.setitem(fastloop._state, "info", None)
    monkeypatch.setattr(fastloop, "_library_paths", lambda: ["/nonexistent/libmujoco.so"])
    with pytest.warns(RuntimeWarning, match="compiled episode loop unavailable"):
        first = ep.run(ctrl)
    with warnings.catch_warnings():
        warnings.simplefilter("error")        # no second warning
        second = ep.run(ctrl)
    _compare(first, second)


def test_blow_up_is_reported_the_same(monkeypatch):
    import mujoco
    ep = Episode(_walker(), flat(), 2.0)
    model = mujoco.MjModel.from_xml_string(ep.xml)
    model.opt.gravity[:] = (0.0, 0.0, -1e15)
    ctrl = co.random_controller(np.random.default_rng(1))
    ref, fast = _both(monkeypatch, ep, ctrl, randomize.Variant(model, 0))
    assert ref.unstable and fast.unstable
    _compare(ref, fast)
