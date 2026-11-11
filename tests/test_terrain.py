"""Terrains: every one compiles with a robot, seeds change the random ones."""

import mujoco
import numpy as np
import pytest

from darwinbot import morphology as mo
from darwinbot import terrain as te
from darwinbot.sim import Episode


@pytest.mark.parametrize("name", sorted(te.TERRAINS))
def test_every_terrain_builds_and_holds_a_robot(name):
    ep = Episode(mo.template(2), te.TERRAINS[name](), seconds=1.0)
    assert ep.buildable
    d = ep.data
    mujoco.mj_resetData(ep.model, d)
    for _ in range(500):
        mujoco.mj_step(ep.model, d)
    assert np.isfinite(d.qpos).all()
    assert d.qpos[2] > te.TERRAINS[name]().spawn_z    # resting on the ground, not in it


def test_presets_name_known_terrains():
    for names in te.PRESETS.values():
        assert all(n in te.TERRAINS for n in names)
    assert "rubble" in te.PRESETS["terrain"]


def test_seed_redraws_only_random_terrains():
    a = te.get_terrains("all", seed=1)
    b = te.get_terrains("all", seed=2)
    fixed = te.get_terrains("all")
    for ta, tb, tf in zip(a, b, fixed):
        if ta.name in te.SEEDED:
            assert ta.world + ta.assets != tb.world + tb.assets
        else:
            assert ta == tb == tf
    assert te.get_terrains("rubble", seed=5) == te.get_terrains("rubble", seed=5)


def test_downhill_spawns_on_the_plateau():
    t = te.downhill()
    assert t.spawn_z > 0.2
    ep = Episode(mo.template(1), t, seconds=1.0)
    assert ep.data.qpos[2] > t.spawn_z


def test_unknown_terrain_is_refused():
    with pytest.raises(ValueError):
        te.get_terrains("lava")


def test_terrain_arguments_on_the_command_line():
    a, b, c = te.get_terrains("steps:rise=0.08:run=0.3,slope:degrees=12,flat")
    assert a.name == "steps:rise=0.08:run=0.3" and b.name == "slope:degrees=12"
    assert a.world == te.steps(rise=0.08, run=0.3).world
    assert c == te.flat()
    assert [t.world for t in te.get_terrains("terrain", seed=5)] == \
           [t.world for t in te.get_terrains("flat,rough,steps,rubble", seed=5)]


@pytest.mark.parametrize("spec", ["steps:rise", "steps:height=1", "steps:rise=x",
                                  "steps:count=2.5", "rough:seed=3", "stairs"])
def test_bad_terrain_arguments_are_refused(spec):
    with pytest.raises(ValueError):
        te.get_terrains(spec)
