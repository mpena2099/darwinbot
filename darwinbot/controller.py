"""
controller.py
-------------
Load-coupled central pattern generator shared by every morphology.

Each joint has its own phase oscillator theta_i (simulation state). One global
frequency sets the natural rate omega; the servo load slows or speeds the
phase (Tegotae-style feedback, Owaki et al.):

    dtheta_i/dt = omega - sigma_i * N_i * cos(theta_i)
    target_i    = ramp(t) * (offset_i + amp1_i * sin(theta_i + phase1_i)
                                      + amp2_i * sin(2 theta_i + phase2_i))

N_i = clip(load_i / SERVO_TORQUE, -1, 1) is the servo load the STS3215 reports
on the real robot (in simulation, the actuator force). With sigma = 0 this is
an open-loop two-harmonic CPG. Wheels ignore everything but the offset gene,
which is their signed speed: target = ramp(t) * speed.

The controller vector has a fixed size for every morphology and is indexed by
the stable node slot id (see docs/plano-diversidade.md). Genes of slots that
are absent from the body are ignored, so a child can inherit its parent's
controller verbatim.

Layout (all genes in [0, 1])
  [0] frequency
  slot s (s = 0..MAX_NODES-1), CTRL_PER_NODE genes from 1 + s * CTRL_PER_NODE:
    amp1, phase1, offset, amp2, phase2, sigma

Amplitudes are clipped so no servo is asked to move faster than SERVO_SPEED:
(amp1 + 2 * amp2) * omega <= SERVO_SPEED, scaling both down proportionally.
"""

import numpy as np

from darwinbot.morphology import MAX_NODES, ActuatorLayout, Morphology

FREQ_RANGE   = (0.5, 3.0)     # Hz
AMP_RANGE    = (0.0, 0.8)     # rad, each harmonic
OFFSET_RANGE = (-0.6, 0.6)    # rad
SERVO_SPEED  = 5.45           # rad/s, STS3215 no-load speed at 7.4 V
SERVO_TORQUE = 1.91 * 0.7     # N*m: 7.4 V stall torque with a 30% margin
WHEEL_RANGE  = (-SERVO_SPEED, SERVO_SPEED)   # signed so wheels may roll either way
SIGMA_MAX    = 3.0            # rad/s; signed, load sign depends on the mounting
SIGMA_RANGE  = (-SIGMA_MAX, SIGMA_MAX)
CONTROL_DT   = 0.02           # s, the ESP32 updates servo targets at 50 Hz
RAMP_TIME    = 0.5            # s to fade the gait in from rest

(AMP1, PHASE1, OFFSET, AMP2, PHASE2, SIGMA) = range(6)
CTRL_PER_NODE = 6
CTRL_LENGTH   = 1 + MAX_NODES * CTRL_PER_NODE


def _scale(u, lo, hi):
    return lo + np.clip(u, 0.0, 1.0) * (hi - lo)


def node_base(slot: int) -> int:
    """Index of the first gene of a slot's block."""
    return 1 + slot * CTRL_PER_NODE


def active_genes(morph: Morphology) -> np.ndarray:
    """Indices of controller genes that affect this morphology."""
    idx = {0}
    for act in morph.layout():
        b = node_base(act.slot)
        if act.kind == "wheel":
            idx.add(b + OFFSET)
        else:
            idx.update(range(b, b + CTRL_PER_NODE))
    return np.array(sorted(idx), dtype=int)


def random_controller(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 1.0, CTRL_LENGTH)


# ---- the update rule, one place, so the firmware math can be documented ----

def ramp(t):
    """Fade-in factor, 0 at t = 0 and 1 from RAMP_TIME on."""
    return np.minimum(1.0, t / RAMP_TIME)


def normalised_load(force):
    """Servo load as a fraction of the maximum torque, in [-1, 1]."""
    return np.clip(force / SERVO_TORQUE, -1.0, 1.0)


def phase_rate(theta, omega, sigma, force):
    """dtheta/dt of one joint (or an array of them)."""
    return omega - sigma * normalised_load(force) * np.cos(theta)


def servo_targets(theta, t, p: "Params"):
    """Position targets (joints) and speeds (wheels) for phases theta at time t."""
    return ramp(t) * (p.offset + p.amp1 * np.sin(theta + p.phase1)
                      + p.amp2 * np.sin(2.0 * theta + p.phase2))


class Params:
    """Per-actuator constants decoded from a controller vector."""

    def __init__(self, ctrl: np.ndarray, layout: ActuatorLayout):
        ctrl = np.asarray(ctrl, dtype=float)
        assert ctrl.shape == (CTRL_LENGTH,), ctrl.shape
        self.omega = 2 * np.pi * _scale(ctrl[0], *FREQ_RANGE)

        n = len(layout)
        self.amp1   = np.zeros(n)
        self.phase1 = np.zeros(n)
        self.offset = np.zeros(n)
        self.amp2   = np.zeros(n)
        self.phase2 = np.zeros(n)
        self.sigma  = np.zeros(n)
        self.wheel  = np.zeros(n, dtype=bool)
        for a, act in enumerate(layout):
            g = ctrl[node_base(act.slot):node_base(act.slot) + CTRL_PER_NODE]
            if act.kind == "wheel":
                self.wheel[a] = True
                self.offset[a] = _scale(g[OFFSET], *WHEEL_RANGE)
                continue
            self.amp1[a]   = _scale(g[AMP1],   *AMP_RANGE)
            self.phase1[a] = _scale(g[PHASE1], 0.0, 2 * np.pi)
            self.offset[a] = _scale(g[OFFSET], *OFFSET_RANGE)
            self.amp2[a]   = _scale(g[AMP2],   *AMP_RANGE)
            self.phase2[a] = _scale(g[PHASE2], 0.0, 2 * np.pi)
            self.sigma[a]  = _scale(g[SIGMA],  *SIGMA_RANGE)
        # A servo cannot follow a signal whose peak speed exceeds its own.
        peak = (self.amp1 + 2.0 * self.amp2) * self.omega
        k = np.minimum(1.0, SERVO_SPEED / np.maximum(peak, 1e-12))
        self.amp1 = self.amp1 * k
        self.amp2 = self.amp2 * k


class Controller:
    """Stateful controller: owns the phases, advanced every physics step."""

    def __init__(self, ctrl: np.ndarray, layout: ActuatorLayout):
        self.params = Params(ctrl, layout)
        self.omega = self.params.omega
        self.theta = np.zeros(len(layout))

    def reset(self) -> None:
        self.theta[:] = 0.0

    def targets(self, t: float) -> np.ndarray:
        """Targets for the current phases (recomputed every control period)."""
        return servo_targets(self.theta, t, self.params)

    def advance(self, dt: float, load: np.ndarray) -> None:
        """Integrate the phases over one physics step (explicit Euler)."""
        self.theta += dt * phase_rate(self.theta, self.omega,
                                      self.params.sigma, load)

    def step(self, t: float, dt: float, load: np.ndarray) -> np.ndarray:
        """Targets at time t, then the phases advanced by dt under this load."""
        out = self.targets(t)
        self.advance(dt, load)
        return out
