"""
test_search_fixes.py
--------------------
The opt-in search and control options, all off by default: novelty-rank parent
selection, the c300 archive axes, the headroom clip mode and the clip_mode
field of the control file.
"""

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import evolve
from darwinbot import morphology as mo
from darwinbot.control_file import make_control_file, read_control_file, write_control_file
from darwinbot.sim import Behaviour

ACT = [mo.Actuator(1, "joint", "pitch"), mo.Actuator(2, "joint", "pitch"),
       mo.Actuator(3, "joint", "roll"), mo.Actuator(4, "wheel", "wheel")]


@pytest.fixture
def headroom():
    co.set_clip_mode("headroom")
    try:
        yield
    finally:
        co.set_clip_mode("command")


def _elite(fitness, duty, servos=4):
    behaviour = Behaviour(duty, 0.6, float(servos), 0.0, 10.0, 3.0)
    return evolve.Elite(np.full(evolve.morpho.MORPH_LENGTH, 0.5),
                        np.zeros(evolve.ctrl.CTRL_LENGTH), fitness, [fitness], 0,
                        np.array(behaviour, dtype=float))


# ---------------------------------------------------------------- novelty-rank

def _toy_archive(n=8):
    """n elites in n different cells, fitness rising with insertion order."""
    archive = evolve.Archive(terrains="flat")
    for servos in range(2, 17):
        for duty in np.linspace(0.0, 1.0, 11):
            before = len(archive.cells)
            archive.insert(_elite(float(before + 1), duty, servos))
            if len(archive.cells) == n:
                return archive
    raise AssertionError("could not fill the toy archive")


def test_novelty_rank_draws_high_objective_elites_more_often():
    archive = _toy_archive()
    assert len(archive.cells) == 8
    objective = np.array([e.fitness for e in archive.cells.values()])
    median = np.median(objective)
    shares, means = {}, {}
    for selection in ("novelty", "novelty-rank"):
        rng = np.random.default_rng(7)
        drawn = archive.sample_elites(20000, rng, selection=selection)["objective"]
        shares[selection] = np.mean(drawn > median)
        means[selection] = np.mean(drawn)
    assert shares["novelty-rank"] > shares["novelty"]
    assert means["novelty-rank"] > means["novelty"]


# ------------------------------------------------------------------ c300 axes

def test_c300_archive_has_300_cells():
    archive = evolve.Archive(terrains="flat", axes="c300")
    assert archive.insert(_elite(1.0, 0.5))
    assert archive.axes == "c300"
    assert archive.n_descriptors == 5
    assert archive.n_cells == 300
    assert len(archive._grid.centroids) == 300
    assert evolve._centroids(5, 300).shape == (300, 5)


def test_default_archive_keeps_800_cells():
    archive = evolve.Archive(terrains="flat")
    assert archive.insert(_elite(1.0, 0.5))
    assert archive.n_cells == 800
    assert len(archive._grid.centroids) == 800
    assert evolve._centroids(5).shape == (800, 5)
    assert np.array_equal(evolve._centroids(5), evolve._centroids(5, 800))


# -------------------------------------------------------------- clip modes

def test_default_clip_mode_is_command():
    assert co.CLIP_MODE == "command"
    with pytest.raises(ValueError, match="clip mode"):
        co.set_clip_mode("bogus")
    assert co.CLIP_MODE == "command"


def _steered_ctrl(seed=21, amp_gene=1.0):
    rng = np.random.default_rng(seed)
    ctrl = co.random_controller(rng)
    ctrl[0] = 1.0                            # high frequency: the clip bites
    for slot in (1, 2, 3):
        b = co.node_base(slot)
        ctrl[b + co.AMP1] = amp_gene
        ctrl[b + co.AMP2] = amp_gene
        ctrl[b + co.STEER_GAIN] = 0.75       # steer gain 0.5
    return ctrl


def test_headroom_scales_the_stronger_side(headroom):
    ctrl = _steered_ctrl()
    base = co.Params(ctrl, ACT, 0.0)
    j = 0                                    # a pitch joint
    peak_base = (base.amp1_base + 2.0 * base.amp2_base) * (base.omega + np.abs(base.sigma))
    assert peak_base[j] > co.SERVO_SPEED
    g = np.clip(1.0 + np.abs(base.steer_gain), 0.0, 2.0)
    k0 = np.minimum(1.0, co.SERVO_SPEED / np.maximum(peak_base * g, 1e-12))
    assert k0[j] < 1.0
    left = co.Params(ctrl, ACT, -1.0)
    right = co.Params(ctrl, ACT, 1.0)
    gl = np.clip(1.0 - base.steer_gain, 0.0, 2.0)
    gr = np.clip(1.0 + base.steer_gain, 0.0, 2.0)
    expected = k0[j] * abs(base.amp1_base[j] * gl[j] - base.amp1_base[j] * gr[j])
    assert abs(left.amp1[j] - right.amp1[j]) == pytest.approx(expected, rel=1e-9)
    # the safety clip did not bite at full speed: both sides fit the servo
    for p in (left, right):
        peak = (p.amp1 + 2.0 * p.amp2) * (p.omega + np.abs(p.sigma))
        assert peak[j] <= co.SERVO_SPEED * (1 + 1e-9)
    # the command clip mode clips only the stronger side, so it differs
    co.set_clip_mode("command")
    try:
        cmd_left = co.Params(ctrl, ACT, -1.0)
        cmd_right = co.Params(ctrl, ACT, 1.0)
    finally:
        co.set_clip_mode("headroom")
    assert not np.isclose(abs(cmd_left.amp1[j] - cmd_right.amp1[j]), expected)


def test_headroom_equals_command_when_nothing_clips(headroom):
    ctrl = _steered_ctrl(seed=22, amp_gene=0.1)
    ctrl[0] = 0.2                            # slow gait: no joint reaches the servo speed
    base = co.Params(ctrl, ACT, 0.0)
    peak_base = (base.amp1_base + 2.0 * base.amp2_base) * (base.omega + np.abs(base.sigma))
    g = np.clip(1.0 + np.abs(base.steer_gain), 0.0, 2.0)
    assert np.all((peak_base * g)[:3] < co.SERVO_SPEED)
    for command in (-1.0, 0.3, 1.0):
        head = co.Params(ctrl, ACT, command)
        co.set_clip_mode("command")
        try:
            plain = co.Params(ctrl, ACT, command)
        finally:
            co.set_clip_mode("headroom")
        assert np.array_equal(head.amp1, plain.amp1)
        assert np.array_equal(head.amp2, plain.amp2)
        assert np.array_equal(head.offset, plain.offset)


def test_default_mode_keeps_the_old_formula():
    ctrl = _steered_ctrl(seed=31)
    ctrl[co.node_base(4) + co.OFFSET] = 0.7  # wheel
    p = co.Params(ctrl, ACT, -0.6, 0.25)
    base = co.Params(ctrl, ACT, 0.0, 1.0)
    d = 1.0 - 0.25
    gain = (np.clip(1.0 - 0.6 * base.steer_gain, 0.0, 2.0)
            * np.clip(1.0 - d * base.speed_gain, 0.0, 2.0))
    amp1 = np.where(base.wheel, base.amp1_base, base.amp1_base * gain)
    amp2 = np.where(base.wheel, base.amp2_base, base.amp2_base * gain)
    peak = (amp1 + 2.0 * amp2) * (base.omega + np.abs(base.sigma))
    k = np.minimum(1.0, co.SERVO_SPEED / np.maximum(peak, 1e-12))
    assert np.any(k[:3] < 1.0)               # the clip really runs here
    assert np.allclose(p.amp1, amp1 * k)
    assert np.allclose(p.amp2, amp2 * k)


# ------------------------------------------------------- control file field

CF_LAYOUT = [(2, "joint", "pitch"), (5, "joint", "pitch"), (7, "joint", "roll"),
             (9, "wheel", "wheel")]
CF_ACT = [mo.Actuator(2, "joint", "pitch"), mo.Actuator(5, "joint", "pitch"),
          mo.Actuator(7, "joint", "roll"), mo.Actuator(9, "wheel", "wheel")]
RANGES = {"pitch": 1.2, "roll": 0.9}
SIGNS = [1, -1, 1, -1]


def _cf(clip_mode=None):
    params = co.Params(co.random_controller(np.random.default_rng(5)), CF_ACT, -0.5, 0.8)
    return make_control_file("test", 1.5, CF_LAYOUT, params, RANGES, signs=SIGNS,
                             clip_mode=clip_mode)


def test_control_file_has_no_clip_mode_by_default(tmp_path):
    d = _cf()
    assert "clip_mode" not in d
    write_control_file(d, tmp_path / "c.json")
    back = read_control_file(tmp_path / "c.json")
    assert "clip_mode" not in back
    assert back == d


def test_control_file_round_trips_headroom(tmp_path, headroom):
    d = _cf()                                # the mode comes from the controller
    assert d["clip_mode"] == "headroom"
    write_control_file(d, tmp_path / "c.json")
    back = read_control_file(tmp_path / "c.json")
    assert back["clip_mode"] == "headroom"
    assert back == d


def test_control_file_explicit_clip_mode_and_bad_value():
    assert _cf(clip_mode="headroom")["clip_mode"] == "headroom"
    assert "clip_mode" not in _cf(clip_mode="command")
    with pytest.raises(ValueError, match="clip_mode"):
        _cf(clip_mode="bogus")
