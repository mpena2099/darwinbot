"""
test_controller.py
------------------
Per-slot gene layout, active gene selection and the load-coupled phase update.
"""

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import morphology as mo

JOINT = mo.Actuator(1, "joint", "pitch")
WHEEL = mo.Actuator(2, "wheel", "wheel")


def _genes(slot, **kw):
    """Controller vector at mid values with some genes of one slot overridden."""
    ctrl = np.full(co.CTRL_LENGTH, 0.5)
    for name, v in kw.items():
        ctrl[co.node_base(slot) + getattr(co, name.upper())] = v
    return ctrl


def test_ctrl_length():
    assert co.CTRL_LENGTH == 1 + mo.MAX_NODES * co.CTRL_PER_NODE
    assert co.CTRL_PER_NODE == 8


def test_active_genes_follow_actuators():
    plain = mo.template(1)
    wheeled = mo.template(1, wheels=True)
    ap, aw = set(co.active_genes(plain)), set(co.active_genes(wheeled))
    assert 0 in ap and ap < set(range(co.CTRL_LENGTH))
    for act in plain.layout():
        b = co.node_base(act.slot)
        assert set(range(b, b + co.CTRL_PER_NODE)) <= ap
    for act in wheeled.layout():
        if act.kind == "wheel":
            b = co.node_base(act.slot)
            assert b + co.OFFSET in aw
            assert not {b + co.AMP1, b + co.SIGMA} & aw
    # genes of slots the body does not use stay out
    used = {a.slot for a in wheeled.layout()}
    free = next(s for s in range(1, mo.MAX_NODES) if s not in used)
    assert not set(range(co.node_base(free), co.node_base(free) + co.CTRL_PER_NODE)) & aw


def test_sigma_zero_is_an_open_loop_two_harmonic_cpg():
    ctrl = _genes(1, amp1=0.4, phase1=0.3, offset=0.7, amp2=0.3, phase2=0.8, sigma=0.5)
    c = co.Controller(ctrl, [JOINT])
    p = c.params
    assert p.sigma[0] == 0.0
    dt = 0.002
    rng = np.random.default_rng(0)
    for k in range(2000):
        t = k * dt
        out = c.step(t, dt, rng.normal(size=1))        # the load must not matter
        ref = min(1.0, t / co.RAMP_TIME) * (
            p.offset[0] + p.amp1[0] * np.sin(p.omega * t + p.phase1[0])
            + p.amp2[0] * np.sin(2 * p.omega * t + p.phase2[0]))
        assert out[0] == pytest.approx(ref, abs=1e-9)


def test_constant_load_shifts_phase_as_the_ode_says():
    sigma_gene = 1.0                                  # sigma = +SIGMA_MAX
    c = co.Controller(_genes(1, sigma=sigma_gene), [JOINT])
    force = 0.4 * co.SERVO_TORQUE                     # N = 0.4
    dt = 0.001
    theta = 0.0                                       # reference integration
    omega, sigma, n = c.omega, co.SIGMA_MAX, 0.4
    for k in range(3000):
        c.step(k * dt, dt, np.array([force]))
        theta += dt * (omega - sigma * n * np.cos(theta))
    assert c.theta[0] == pytest.approx(theta, rel=1e-9)
    assert abs(theta - omega * 3000 * dt) > 0.05      # and it really shifted

    c.reset()
    assert c.theta[0] == 0.0


def test_load_is_clipped_to_servo_torque():
    c1 = co.Controller(_genes(1, sigma=1.0), [JOINT])
    c2 = co.Controller(_genes(1, sigma=1.0), [JOINT])
    c1.advance(0.01, np.array([co.SERVO_TORQUE]))
    c2.advance(0.01, np.array([50 * co.SERVO_TORQUE]))
    assert c1.theta[0] == c2.theta[0]


def test_sigma_range_is_signed():
    lo = co.Controller(_genes(1, sigma=0.0), [JOINT]).params.sigma[0]
    hi = co.Controller(_genes(1, sigma=1.0), [JOINT]).params.sigma[0]
    assert lo == -co.SIGMA_MAX and hi == co.SIGMA_MAX


def test_amplitude_clip_keeps_peak_speed_and_proportion():
    ctrl = _genes(1, amp1=1.0, amp2=0.5)
    ctrl[0] = 1.0                                     # max frequency
    c = co.Controller(ctrl, [JOINT])
    p = c.params
    assert (p.amp1[0] + 2 * p.amp2[0]) * p.omega == pytest.approx(co.SERVO_SPEED)
    assert p.amp1[0] < co.AMP_RANGE[1]
    assert p.amp2[0] / p.amp1[0] == pytest.approx(0.5)

    ctrl[0] = 0.0                                     # slow: no clipping needed
    p = co.Controller(ctrl, [JOINT]).params
    assert p.amp1[0] == pytest.approx(co.AMP_RANGE[1])


def test_wheel_is_ramped_signed_speed_and_ignores_the_rest():
    fast = co.Controller(_genes(2, offset=1.0, amp1=1.0, sigma=1.0), [WHEEL])
    back = co.Controller(_genes(2, offset=0.0), [WHEEL])
    assert fast.params.sigma[0] == 0.0 and fast.params.amp1[0] == 0.0
    assert fast.targets(co.RAMP_TIME)[0] == pytest.approx(co.SERVO_SPEED)
    assert back.targets(co.RAMP_TIME)[0] == pytest.approx(-co.SERVO_SPEED)
    assert fast.targets(co.RAMP_TIME / 2)[0] == pytest.approx(co.SERVO_SPEED / 2)
    fast.advance(0.01, np.array([co.SERVO_TORQUE]))
    assert fast.theta[0] == pytest.approx(0.01 * fast.omega)
    assert fast.targets(co.RAMP_TIME)[0] == pytest.approx(co.SERVO_SPEED)


def test_starts_at_zero_and_ramps():
    rng = np.random.default_rng(0)
    layout = mo.template(2, wheels=True, yaw=True).layout()
    c = co.Controller(co.random_controller(rng), layout)
    assert np.allclose(c.targets(0.0), 0.0)
    half = c.targets(co.RAMP_TIME / 2)
    full = c.targets(co.RAMP_TIME)
    assert np.allclose(half, 0.5 * full)


def test_steering_command_scales_amplitude_and_shifts_offset():
    ctrl = _genes(1, amp1=0.4, offset=0.5, amp2=0.2)
    b = co.node_base(1)
    ctrl[b + co.STEER_GAIN], ctrl[b + co.STEER_SHIFT] = 1.0, 1.0   # gain +1, shift +0.4
    ctrl[0] = 0.0                       # slowest rhythm: no speed clip in the way
    straight = co.Params(ctrl, [JOINT])
    right = co.Params(ctrl, [JOINT], command=1.0)
    left = co.Params(ctrl, [JOINT], command=-1.0)
    assert right.amp1[0] > straight.amp1[0] > left.amp1[0] == 0.0
    assert np.isclose(right.offset[0] - straight.offset[0], co.STEER_SHIFT_RANGE[1])
    assert (right.amp1[0] + 2 * right.amp2[0]) * right.omega <= co.SERVO_SPEED + 1e-9


def test_steering_changes_wheel_speed_within_range():
    ctrl = _genes(2, offset=1.0)
    ctrl[co.node_base(2) + co.STEER_SHIFT] = 0.0                  # full negative change
    fast = co.Params(ctrl, [WHEEL]).offset[0]
    slowed = co.Params(ctrl, [WHEEL], command=1.0).offset[0]
    assert slowed < fast and slowed >= co.WHEEL_RANGE[0]
    assert co.Params(ctrl, [WHEEL], command=-1.0).offset[0] == co.WHEEL_RANGE[1]
