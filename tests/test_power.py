"""Tests for the electrical estimate (report only, never scored)."""

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import morphology as mo
from darwinbot import power
from darwinbot.sim import Episode
from darwinbot.terrain import get_terrains


def test_current_model():
    half = power.I_IDLE + 0.5 * (power.I_STALL - power.I_IDLE)
    assert power.servo_current(0.0) == pytest.approx(power.I_IDLE)
    assert power.servo_current(0.5 * power.TAU_STALL) == pytest.approx(half)
    assert power.servo_current(1.5 * power.TAU_STALL) == pytest.approx(power.I_STALL)
    assert power.servo_current(-3.0 * power.TAU_STALL) == pytest.approx(power.I_STALL)
    assert power.servo_current(np.array([0.0, 2.0 * power.TAU_STALL])) == \
        pytest.approx([power.I_IDLE, power.I_STALL])


def test_estimate_small_body():
    morph = mo.decode(mo.random_morphology(np.random.default_rng(3)))
    episode = Episode(morph, get_terrains("flat")[0], 2.0)
    ctrl = np.random.default_rng(4).uniform(0, 1, co.CTRL_LENGTH)
    r = power.estimate(episode, ctrl, 2.0)
    values = [r.mean_a, r.peak_a, r.peak_1s_a, r.runtime_min, r.c_needed]
    assert np.all(np.isfinite(values)) and np.all(np.array(values) >= 0)
    assert r.n_servos == len(episode.layout) == morph.n_servos
    assert r.peak_a >= r.peak_1s_a >= r.mean_a
    assert len(r.per_servo_peak_a) == r.n_servos
    assert r.c_needed == pytest.approx(r.peak_1s_a / r.capacity_ah)


def test_report_ends_with_the_caveat():
    morph = mo.decode(mo.random_morphology(np.random.default_rng(5)))
    episode = Episode(morph, get_terrains("flat")[0], 1.0)
    ctrl = np.random.default_rng(6).uniform(0, 1, co.CTRL_LENGTH)
    lines = power.report_lines(power.estimate(episode, ctrl, 1.0))
    assert lines[-1] == "estimate from simulated torque, not measured"
    assert any("runtime" in line for line in lines)
