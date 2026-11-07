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

Steering: a command c in [-1, 1] (0 straight, -1 left, +1 right) comes from
outside (a remote, later a sensor). Each joint scales both amplitudes by
(1 + c * gain) and shifts its offset by c * shift; each wheel adds c * shift
to its speed. Animals turn the same way, with one gait and one side stronger,
so the robot keeps a single rhythm and evolution finds how this body turns.

Speed: a command s in [0, 1] (1 = the gait as evolved, 0 = the slowest the body
learned) asks for a forward speed. With d = 1 - s, each joint scales both
amplitudes by (1 - d * speed_gain), kept in [0, 2], and shifts its offset by
d * speed_shift; each wheel adds d * speed_shift to its speed. The frequency is
shared by the whole body and stays put. At s = 1 nothing changes, so a robot
that never heard of the command walks as before. Steering and speed combine:
the amplitude factors multiply and the shifts add, then the speed clip.

The controller vector has a fixed size for every morphology and is indexed by
the stable node slot id (see morphology.py). Genes of slots that
are absent from the body are ignored, so a child can inherit its parent's
controller verbatim.

Layout (all genes in [0, 1])
  [0] frequency
  slot s (s = 0..MAX_NODES-1), CTRL_PER_NODE genes from 1 + s * CTRL_PER_NODE:
    amp1, phase1, offset, amp2, phase2, sigma, steer_gain, steer_shift,
    speed_gain, speed_shift

Amplitudes are clipped so no servo is asked to move faster than SERVO_SPEED:
(amp1 + 2 * amp2) * omega <= SERVO_SPEED, scaling both down proportionally.
"""

import math

import numpy as np
from numba import njit

from darwinbot.morphology import MAX_NODES, ActuatorLayout, Morphology

FREQ_RANGE   = (0.3, 1.5)     # Hz; the second harmonic doubles it. Robots and animals
                              # of this size step 1 to 2 times a second; at 3 Hz the
                              # gaits were a shiver of short, fast steps
AMP_RANGE    = (0.0, 0.8)     # rad, each harmonic
OFFSET_RANGE = (-0.6, 0.6)    # rad
SERVO_SPEED  = 4.71           # rad/s, STS3215 (12 V) no-load speed
LOAD_FULL_SCALE = 0.7        # fraction of the STS3215 stall torque that reads as N = 1
SERVO_TORQUE = 2.94 * LOAD_FULL_SCALE   # N*m: 12 V stall torque with a 30% margin
WHEEL_RANGE  = (-SERVO_SPEED, SERVO_SPEED)   # signed so wheels may roll either way
SIGMA_MAX    = 3.0            # rad/s; signed, load sign depends on the mounting
SIGMA_RANGE  = (-SIGMA_MAX, SIGMA_MAX)
STEER_GAIN_RANGE  = (-1.0, 1.0)     # amplitude factor 1 + c * gain, kept in [0, 2]
STEER_SHIFT_RANGE = (-0.4, 0.4)     # rad, offset shift at full command
WHEEL_STEER_RANGE = WHEEL_RANGE     # rad/s, wheel speed change at full command
SPEED_GAIN_RANGE  = (-1.0, 1.0)     # amplitude factor 1 - d * gain, kept in [0, 2]
SPEED_SHIFT_RANGE = (-0.4, 0.4)     # rad, offset shift at speed command 0
WHEEL_SPEED_RANGE = WHEEL_RANGE     # rad/s, wheel speed change at speed command 0
CONTROL_DT   = 0.02           # s, the ESP32 updates servo targets at 50 Hz
RAMP_TIME    = 0.5            # s to fade the gait in from rest

(AMP1, PHASE1, OFFSET, AMP2, PHASE2, SIGMA, STEER_GAIN, STEER_SHIFT,
 SPEED_GAIN, SPEED_SHIFT) = range(10)
CTRL_PER_NODE = 10
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
            idx.update((b + OFFSET, b + STEER_SHIFT, b + SPEED_SHIFT))
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


@njit(cache=True)
def advance_phases(theta, omega, sigma, force, dt):
    """theta += dt * phase_rate(theta, omega, sigma, force), in place, compiled:
    the physics loop calls it every step. Same operations in the same order as
    phase_rate, so the phases match the NumPy version bit for bit."""
    for i in range(theta.shape[0]):
        n = force[i] / SERVO_TORQUE
        if n > 1.0:
            n = 1.0
        elif n < -1.0:
            n = -1.0
        theta[i] += dt * (omega - sigma[i] * n * math.cos(theta[i]))


def servo_targets(theta, t, p: "Params"):
    """Position targets (joints) and speeds (wheels) for phases theta at time t."""
    return ramp(t) * (p.offset + p.amp1 * np.sin(theta + p.phase1)
                      + p.amp2 * np.sin(2.0 * theta + p.phase2))


class Params:
    """Per-actuator constants decoded from a controller vector, under a
    steering command (0 = straight) and a speed command (1 = full)."""

    def __init__(self, ctrl: np.ndarray, layout: ActuatorLayout,
                 command: float = 0.0, speed: float = 1.0):
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
        self.steer_gain  = np.zeros(n)
        self.steer_shift = np.zeros(n)
        self.speed_gain  = np.zeros(n)
        self.speed_shift = np.zeros(n)
        self.wheel  = np.zeros(n, dtype=bool)
        for a, act in enumerate(layout):
            g = ctrl[node_base(act.slot):node_base(act.slot) + CTRL_PER_NODE]
            if act.kind == "wheel":
                self.wheel[a] = True
                self.offset[a] = _scale(g[OFFSET], *WHEEL_RANGE)
                self.steer_shift[a] = _scale(g[STEER_SHIFT], *WHEEL_STEER_RANGE)
                self.speed_shift[a] = _scale(g[SPEED_SHIFT], *WHEEL_SPEED_RANGE)
                continue
            self.amp1[a]   = _scale(g[AMP1],   *AMP_RANGE)
            self.phase1[a] = _scale(g[PHASE1], 0.0, 2 * np.pi)
            self.offset[a] = _scale(g[OFFSET], *OFFSET_RANGE)
            self.amp2[a]   = _scale(g[AMP2],   *AMP_RANGE)
            self.phase2[a] = _scale(g[PHASE2], 0.0, 2 * np.pi)
            self.sigma[a]  = _scale(g[SIGMA],  *SIGMA_RANGE)
            self.steer_gain[a]  = _scale(g[STEER_GAIN], *STEER_GAIN_RANGE)
            self.steer_shift[a] = _scale(g[STEER_SHIFT], *STEER_SHIFT_RANGE)
            self.speed_gain[a]  = _scale(g[SPEED_GAIN], *SPEED_GAIN_RANGE)
            self.speed_shift[a] = _scale(g[SPEED_SHIFT], *SPEED_SHIFT_RANGE)
        # as decoded, before the commands and the speed clip: what the firmware
        # receives, so it can steer and clip on the robot the same way
        self.amp1_base, self.amp2_base = self.amp1.copy(), self.amp2.copy()
        self.offset_base = self.offset.copy()
        apply_commands(self, command, speed)
        # A servo cannot follow a signal whose peak speed exceeds its own.
        peak = (self.amp1 + 2.0 * self.amp2) * self.omega
        k = np.minimum(1.0, SERVO_SPEED / np.maximum(peak, 1e-12))
        self.amp1 = self.amp1 * k
        self.amp2 = self.amp2 * k


def apply_commands(p: "Params", command: float, speed: float = 1.0) -> None:
    """Apply the steering and speed commands to the decoded params, in place,
    always from the *_base values (before the speed clip, so a stronger side
    still respects the servo). The firmware does the same in set_command and
    set_speed."""
    c = float(np.clip(command, -1.0, 1.0))
    d = 1.0 - float(np.clip(speed, 0.0, 1.0))
    gain = (np.clip(1.0 + c * p.steer_gain, 0.0, 2.0)
            * np.clip(1.0 - d * p.speed_gain, 0.0, 2.0))
    shift = c * p.steer_shift + d * p.speed_shift
    p.amp1 = np.where(p.wheel, p.amp1_base, p.amp1_base * gain)
    p.amp2 = np.where(p.wheel, p.amp2_base, p.amp2_base * gain)
    p.offset = np.where(p.wheel, np.clip(p.offset_base + shift, *WHEEL_RANGE),
                        p.offset_base + shift)


class Controller:
    """Stateful controller: owns the phases, advanced every physics step."""

    def __init__(self, ctrl: np.ndarray, layout: ActuatorLayout,
                 command: float = 0.0, speed: float = 1.0):
        self.params = Params(ctrl, layout, command, speed)
        self.omega = self.params.omega
        self.theta = np.zeros(len(layout))

    def reset(self) -> None:
        self.theta[:] = 0.0

    def targets(self, t: float) -> np.ndarray:
        """Targets for the current phases (recomputed every control period)."""
        return servo_targets(self.theta, t, self.params)

    def advance(self, dt: float, load: np.ndarray) -> None:
        """Integrate the phases over one physics step (explicit Euler)."""
        advance_phases(self.theta, self.omega, self.params.sigma, load, dt)

    def step(self, t: float, dt: float, load: np.ndarray) -> np.ndarray:
        """Targets at time t, then the phases advanced by dt under this load."""
        out = self.targets(t)
        self.advance(dt, load)
        return out
