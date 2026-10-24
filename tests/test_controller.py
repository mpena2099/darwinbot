"""
test_controller.py
------------------
CPG layout, active gene selection and signal shape.
"""

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import morphology as mo


def test_ctrl_length():
    assert co.PAIR_CTRL == len(co.JOINTS) * 3 + 2
    assert co.CTRL_LENGTH == 1 + mo.N_PAIRS * co.PAIR_CTRL


def test_active_genes_follow_pairs_and_wheels():
    pairs = [
        mo.LegPair(0, 0.0, 0.0, 0.10, 0.10, True, True, 0.05),
        mo.LegPair(2, 0.0, 0.0, 0.10, 0.10, False, False, 0.05),
    ]
    m = mo.Morphology(body_length=0.4, body_width=0.2, pairs=pairs)
    active = set(co.active_genes(m))

    assert 0 in active
    wheel_0 = 1 + co.PAIR_CTRL - 1
    wheel_2 = 1 + 2 * co.PAIR_CTRL + co.PAIR_CTRL - 1
    assert wheel_0 in active
    assert wheel_2 not in active

    yaw_0 = range(1, 1 + co.JOINT_CTRL)
    yaw_2 = range(1 + 2 * co.PAIR_CTRL, 1 + 2 * co.PAIR_CTRL + co.JOINT_CTRL)
    assert set(yaw_0) <= active
    assert not active.intersection(yaw_2)

    absent = range(1 + co.PAIR_CTRL, 1 + 2 * co.PAIR_CTRL)
    assert not active.intersection(absent)


def test_cpg_starts_at_zero_and_is_bounded():
    rng = np.random.default_rng(0)
    ctrl = co.random_controller(rng)
    layout = [(p, s, k) for p in range(mo.N_PAIRS) for s in (0, 1)
              for k in co.JOINTS]
    cpg = co.CPG(ctrl, layout)

    assert np.allclose(cpg(0.0), 0.0)
    for t in (co.RAMP_TIME, 1.3, 4.7):
        out = cpg(t)
        assert np.all(np.abs(out) <= np.abs(cpg.offset) + cpg.amp + 1e-12)


def test_ramp_scales_the_signal():
    ctrl = np.full(co.CTRL_LENGTH, 0.5)
    ctrl[1 + co.JOINTS.index("knee") * 3] = 0.0   # zero amplitude, ramp only
    cpg = co.CPG(ctrl, [(0, 0, "knee")])

    half = cpg(co.RAMP_TIME / 2)
    full = cpg(co.RAMP_TIME)
    assert np.allclose(half, 0.5 * full)


def test_right_phase_is_left_plus_side_shift():
    ctrl = np.full(co.CTRL_LENGTH, 0.5)
    phase_gene = 1 + co.JOINTS.index("hip_yaw") * 3 + 2
    shift_gene = 1 + len(co.JOINTS) * 3
    ctrl[phase_gene] = 0.25
    ctrl[shift_gene] = 0.4
    cpg = co.CPG(ctrl, [(0, 0, "hip_yaw"), (0, 1, "hip_yaw")])

    assert cpg.amp[0] == pytest.approx(cpg.amp[1])
    assert cpg.offset[0] == pytest.approx(cpg.offset[1])
    assert cpg.phase[1] - cpg.phase[0] == pytest.approx(0.4 * 2 * np.pi)


def test_cpg_amplitude_respects_servo_speed():
    ctrl = np.full(co.CTRL_LENGTH, 1.0)    # max frequency and amplitude
    cpg = co.CPG(ctrl, [(0, 0, "knee"), (0, 0, "wheel")])

    assert np.all(cpg.amp * cpg.omega <= co.SERVO_SPEED + 1e-12)
    assert cpg.amp[0] < co.AMP_RANGE[1]
    assert cpg.offset[1] == pytest.approx(co.WHEEL_RANGE[1])
