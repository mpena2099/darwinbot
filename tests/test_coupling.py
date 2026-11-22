"""
test_coupling.py
----------------
Phase coupling between joints: the partner map, the update rule, the neutral
old genome and the control file fields.
"""

import dataclasses
import math

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import evolve
from darwinbot import morphology as mo
from darwinbot import sim
from darwinbot.control_file import make_control_file, read_control_file, write_control_file
from darwinbot.sim import Episode
from darwinbot.terrain import flat


def _genes(slot, **kw):
    """Controller vector at mid values with some genes of one slot overridden."""
    ctrl = np.full(co.CTRL_LENGTH, 0.5)
    for name, v in kw.items():
        ctrl[co.node_base(slot) + getattr(co, name.upper())] = v
    return ctrl


def _compare(a, b):
    da, db = dataclasses.asdict(a), dataclasses.asdict(b)
    for key in da:
        if isinstance(da[key], (bool, int)) or da[key] is None:
            assert da[key] == db[key], key
        else:
            assert da[key] == pytest.approx(db[key], rel=1e-12, abs=1e-12), key


def test_partner_map_of_a_two_pair_template():
    morph = mo.template(2)
    # layout: hip, knee of the four legs, in depth-first order
    assert [a.kind for a in morph.layout()] == ["joint"] * 8
    # knees point at their hip; the four hip roots form a ring
    assert co.couple_partners(morph).tolist() == [6, 0, 0, 2, 2, 4, 4, 6]


def test_single_joint_has_no_partner():
    morph = mo.Morphology(0.15, 0.10, {0: mo.Node(0, "core"),
                                       1: mo.Node(1, "joint", 0)})
    assert co.couple_partners(morph).tolist() == [-1]
    chain = mo.Morphology(0.15, 0.10, {0: mo.Node(0, "core"),
                                       1: mo.Node(1, "joint", 0),
                                       2: mo.Node(2, "joint", 1)})
    assert co.couple_partners(chain).tolist() == [-1, 0]


def test_wheels_never_couple():
    morph = mo.template(1, wheels=True)
    layout = morph.layout()
    partner = co.couple_partners(morph)
    assert [a.kind for a in layout] == ["joint", "joint", "wheel",
                                        "joint", "joint", "wheel"]
    assert partner.tolist() == [3, 0, -1, 0, 3, -1]
    assert all(partner[i] == -1 for i, a in enumerate(layout) if a.kind == "wheel")


def test_wheels_ignore_the_coupling_genes():
    ctrl = _genes(2, couple_k=1.0, couple_lag=1.0)
    p = co.Params(ctrl, [mo.Actuator(2, "wheel", "wheel")])
    assert p.couple_k[0] == 0.0 and p.couple_lag[0] == 0.0


def _old_advance(theta, omega, sigma, force, dt, *rest):
    """The update rule before coupling: what the coupled code must reproduce
    when every k is zero."""
    for i in range(theta.shape[0]):
        n = force[i] / co.SERVO_TORQUE
        if n > 1.0:
            n = 1.0
        elif n < -1.0:
            n = -1.0
        theta[i] += dt * (omega - sigma[i] * n * math.cos(theta[i]))


def test_zero_coupling_is_the_old_controller(monkeypatch):
    """A controller stored as BODY_MODEL 23 migrates to k = 0 and runs exactly
    as the old update rule, with the partner map in place."""
    morph = mo.decode(mo.encode(mo.template(1, yaw=True)))
    old = np.random.default_rng(4).uniform(size=1 + mo.MAX_NODES * 10)
    _, ctrl = evolve.MIGRATIONS[23](morph, old)
    layout = morph.layout()
    params = co.Params(ctrl, layout, partner=co.couple_partners(morph))
    assert np.all(params.couple_k == 0.0) and np.all(params.couple_lag == 0.0)

    ep = Episode(morph, flat(), 2.0)
    assert ep.buildable
    real = sim.advance_phases
    monkeypatch.setenv("DARWINBOT_PY_LOOP", "1")
    monkeypatch.setattr(sim, "advance_phases", _old_advance)
    before = ep.run(ctrl)
    monkeypatch.setattr(sim, "advance_phases", real)
    after = ep.run(ctrl)
    _compare(before, after)


def test_coupled_joints_converge_in_phase():
    ctrl = np.full(co.CTRL_LENGTH, 0.5)
    b1, b2 = co.node_base(1), co.node_base(2)
    ctrl[b1 + co.COUPLE_K] = 1.0                 # k = COUPLE_RANGE[1] = 4 rad/s
    ctrl[b2 + co.COUPLE_K] = 1.0
    ctrl[b1 + co.COUPLE_LAG] = 0.5               # lag = 0 rad
    ctrl[b2 + co.COUPLE_LAG] = 0.5
    layout = [mo.Actuator(1, "joint", "pitch"), mo.Actuator(2, "joint", "pitch")]
    c = co.Controller(ctrl, layout, partner=np.array([1, 0]))
    assert c.params.couple_k.tolist() == [4.0, 4.0]
    assert np.allclose(c.params.couple_lag, 0.0)
    c.theta[:] = [0.0, 1.0]
    gaps = []
    dt = 0.001
    for _ in range(20000):
        c.advance(dt, np.zeros(2))
        gaps.append(abs(c.theta[0] - c.theta[1]))
    assert gaps[-1] < gaps[0] * 1e-3
    assert gaps[-1] < 1e-3


def test_migration_23_keeps_the_old_genes():
    old = np.random.default_rng(7).uniform(size=1 + mo.MAX_NODES * 10)
    genes = mo.random_morphology(np.random.default_rng(7))
    g, new = evolve.MIGRATIONS[23](genes, old)
    assert np.array_equal(g, genes)
    assert new.shape == (co.CTRL_LENGTH,) and new[0] == old[0]
    n_old = old[1:].reshape(mo.MAX_NODES, 10)
    n_new = new[1:].reshape(mo.MAX_NODES, co.CTRL_PER_NODE)
    assert np.array_equal(n_new[:, :10], n_old)
    assert np.all(n_new[:, co.COUPLE_K] == 0.0)
    assert np.all(n_new[:, co.COUPLE_LAG] == 0.5)   # gene 0.5 is lag 0 rad
    assert evolve.migration_path(23) == list(range(23, evolve.BODY_MODEL))


def test_control_file_format_5_round_trip(tmp_path):
    morph = mo.template(1, wheels=True)
    layout = morph.layout()
    ctrl = co.random_controller(np.random.default_rng(2))
    params = co.Params(ctrl, layout, partner=co.couple_partners(morph))
    ranges = {a.axis: 1.0 for a in layout}
    d = make_control_file("coupled", 0.0, layout, params, ranges)
    assert d["format"] == 5
    write_control_file(d, tmp_path / "c.json")
    assert read_control_file(tmp_path / "c.json") == d
    joints = [i for i, a in enumerate(layout) if a.kind == "joint"]
    assert [d["servos"][i]["couple_to"] for i in joints] == \
        params.partner[joints].tolist()
    for s in d["servos"]:
        if s["mode"] == "wheel":
            assert (s["couple_to"], s["couple_k"], s["couple_lag"]) == (-1, 0.0, 0.0)


def test_control_file_format_4_still_loads(tmp_path):
    morph = mo.template(1)
    layout = morph.layout()
    params = co.Params(co.random_controller(np.random.default_rng(3)), layout,
                       partner=co.couple_partners(morph))
    d = make_control_file("old", 0.0, layout, params,
                          {a.axis: 1.0 for a in layout})
    d["format"] = 4
    for s in d["servos"]:
        del s["couple_to"], s["couple_k"], s["couple_lag"]
    write_control_file(d, tmp_path / "c.json")
    r = read_control_file(tmp_path / "c.json")
    assert all(s["couple_to"] == -1 and s["couple_k"] == 0.0
               and s["couple_lag"] == 0.0 for s in r["servos"])
