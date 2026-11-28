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

Coupling: a joint also feels its partner, the nearest joint ancestor (the limb
root when it has none), and the joints that name it as partner:

    + k_i * sin(theta_p(i) - theta_i - lag_i)
    + sum over j with p(j) = i of k_j * sin(theta_j - theta_i + lag_j)

The limb roots (joints without a joint ancestor) form a ring in layout order,
so a body whose limbs are all roots couples them in a cycle. Wheels never
couple. With every k = 0 the phases are exactly the load-coupled CPG above.

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
    speed_gain, speed_shift, couple_k, couple_lag

Amplitudes are clipped so no servo is asked to move faster than SERVO_SPEED:
(amp1 + 2 * amp2) * (omega + |sigma|) <= SERVO_SPEED, scaling both down proportionally.
In the default "command" clip mode this happens after the commands, so the
stronger steering side is clipped back to the servo speed. The "headroom" mode
scales the base amplitudes first by the headroom of the stronger side (see
CLIP_MODE), so both sides keep their proportion after the commands.
"""

import math

import numpy as np
from numba import njit

from darwinbot import hardware
from darwinbot.morphology import MAX_NODES, ActuatorLayout, Morphology

FREQ_RANGE   = (0.3, 1.5)     # Hz; the second harmonic doubles it. Robots and animals
                              # of this size step 1 to 2 times a second; at 3 Hz the
                              # gaits were a shiver of short, fast steps
AMP_RANGE    = (0.0, 0.8)     # rad, each harmonic
OFFSET_RANGE = (-0.6, 0.6)    # rad
SERVO_SPEED  = hardware.SERVO.speed          # rad/s, STS3215 (12 V) no-load speed
LOAD_FULL_SCALE = hardware.SERVO.load_full_scale   # fraction of the stall torque that reads as N = 1
SERVO_TORQUE = hardware.SERVO.stall_torque * LOAD_FULL_SCALE   # N*m: 12 V stall torque with a 30% margin
WHEEL_RANGE  = (-SERVO_SPEED, SERVO_SPEED)   # signed so wheels may roll either way
SIGMA_MAX    = 3.0            # rad/s; signed, load sign depends on the mounting
SIGMA_RANGE  = (-SIGMA_MAX, SIGMA_MAX)
STEER_GAIN_RANGE  = (-1.0, 1.0)     # amplitude factor 1 + c * gain, kept in [0, 2]
STEER_SHIFT_RANGE = (-0.4, 0.4)     # rad, offset shift at full command
WHEEL_STEER_RANGE = WHEEL_RANGE     # rad/s, wheel speed change at full command
SPEED_GAIN_RANGE  = (-1.0, 1.0)     # amplitude factor 1 - d * gain, kept in [0, 2]
SPEED_SHIFT_RANGE = (-0.4, 0.4)     # rad, offset shift at speed command 0
WHEEL_SPEED_RANGE = WHEEL_RANGE     # rad/s, wheel speed change at speed command 0
COUPLE_RANGE      = (0.0, 4.0)      # rad/s, phase coupling strength
COUPLE_LAG_RANGE  = (-np.pi, np.pi)  # rad, phase offset of the coupling
CONTROL_DT   = 0.02           # s, the ESP32 updates servo targets at 50 Hz
RAMP_TIME    = 0.5            # s to fade the gait in from rest

CLIP_MODES = ("command", "headroom")
CLIP_MODE  = "command"        # command: clip after the commands (default);
                              # headroom: scale the base amplitudes first so the
                              # stronger steering side keeps its proportion


def set_clip_mode(mode: str) -> None:
    """Select the servo speed clip mode (see CLIP_MODES)."""
    global CLIP_MODE
    if mode not in CLIP_MODES:
        raise ValueError(f"clip mode must be one of {CLIP_MODES}, got {mode!r}")
    CLIP_MODE = mode

(AMP1, PHASE1, OFFSET, AMP2, PHASE2, SIGMA, STEER_GAIN, STEER_SHIFT,
 SPEED_GAIN, SPEED_SHIFT, COUPLE_K, COUPLE_LAG) = range(12)
CTRL_PER_NODE = 12
CTRL_LENGTH   = 1 + MAX_NODES * CTRL_PER_NODE


def _scale(u, lo, hi):
    return lo + np.clip(u, 0.0, 1.0) * (hi - lo)


def node_base(slot: int) -> int:
    """Index of the first gene of a slot's block."""
    return 1 + slot * CTRL_PER_NODE


def copy_node_genes(ctrl: np.ndarray, copies) -> np.ndarray:
    """Controller of a child whose body copied limbs: for every (source slot,
    new slot, mirrored) of `copies`, in order, the new slot's gene block becomes
    the source's, so the twin starts with the gait of the limb it copies instead
    of unrelated genes. The twin walks in antiphase: PHASE1 moves half a cycle
    (the oscillator runs half a period later). PHASE2 is left alone, because
    half a period of the fundamental is a whole one of the second harmonic. A
    mirrored copy also swaps the steering sign, so the twin on the other side
    turns the other way: STEER_GAIN_RANGE and STEER_SHIFT_RANGE are symmetric
    about 0, so a gene g and 1 - g are opposite values. Returns a new vector."""
    out = np.array(ctrl, dtype=float)
    for src, new, mirrored in copies:
        a, b = node_base(src), node_base(new)
        block = out[a:a + CTRL_PER_NODE].copy()
        block[PHASE1] = (block[PHASE1] + 0.5) % 1.0
        if mirrored:
            block[STEER_GAIN] = 1.0 - block[STEER_GAIN]
            block[STEER_SHIFT] = 1.0 - block[STEER_SHIFT]
        out[b:b + CTRL_PER_NODE] = block
    return out


def couple_partners(morph: Morphology) -> np.ndarray:
    """Partner of every actuator in layout order (-1 for none): a joint's
    nearest joint ancestor, or for a limb root (no joint ancestor) the previous
    root in a ring over the roots; wheels have none."""
    layout = morph.layout()
    index = {act.slot: i for i, act in enumerate(layout)}
    partner = np.full(len(layout), -1, dtype=int)
    roots = []
    for i, act in enumerate(layout):
        if act.kind != "joint":
            continue
        p = morph.nodes[act.slot].parent
        while p >= 0:
            j = index.get(p)
            if j is not None and layout[j].kind == "joint":
                partner[i] = j
                break
            p = morph.nodes[p].parent
        if partner[i] < 0:
            roots.append(i)
    if len(roots) > 1:
        for r, i in enumerate(roots):
            partner[i] = roots[r - 1]
    return partner


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
    """Uniform genes, except the coupling strength, which starts at 0 so a new
    body begins with independent oscillators and evolution adds coupling only
    where it helps (random coupling scrambled young gaits: the first gate lost
    40% of the body types)."""
    ctrl = rng.uniform(0.0, 1.0, CTRL_LENGTH)
    ctrl[1 + COUPLE_K::CTRL_PER_NODE] = 0.0
    return ctrl


# ---- the update rule, one place, so the firmware math can be documented ----

def ramp(t):
    """Fade-in factor, 0 at t = 0 and 1 from RAMP_TIME on."""
    return np.minimum(1.0, t / RAMP_TIME)


def normalised_load(force):
    """Servo load as a fraction of the maximum torque, in [-1, 1]."""
    return np.clip(force / SERVO_TORQUE, -1.0, 1.0)


def phase_rate(theta, omega, sigma, force, couple_k=None, couple_lag=None,
               partner=None):
    """dtheta/dt of one joint (or an array of them), coupling terms evaluated
    on the phases given."""
    rate = omega - sigma * normalised_load(force) * np.cos(theta)
    if couple_k is not None and partner is not None:
        for j in range(len(theta)):
            k = couple_k[j]
            p = partner[j]
            if k != 0.0 and p >= 0:
                rate[j] += k * np.sin(theta[p] - theta[j] - couple_lag[j])
                rate[p] += k * np.sin(theta[j] - theta[p] + couple_lag[j])
    return rate


@njit(cache=True)
def advance_phases(theta, omega, sigma, force, dt, couple_k, couple_lag,
                   partner):
    """theta += dt * phase_rate(theta, omega, sigma, force, couple_k,
    couple_lag, partner), in place, compiled: the physics loop calls it every
    step. Every term reads the phases from the start of the step. Same
    operations in the same order as phase_rate, so the phases match the NumPy
    version bit for bit, and with every k = 0 exactly the old load-coupled
    update."""
    n = theta.shape[0]
    rate = np.empty(n)
    for i in range(n):
        load = force[i] / SERVO_TORQUE
        if load > 1.0:
            load = 1.0
        elif load < -1.0:
            load = -1.0
        rate[i] = omega - sigma[i] * load * math.cos(theta[i])
    for j in range(n):
        k = couple_k[j]
        p = partner[j]
        if k != 0.0 and p >= 0:
            rate[j] += k * math.sin(theta[p] - theta[j] - couple_lag[j])
            rate[p] += k * math.sin(theta[j] - theta[p] + couple_lag[j])
    for i in range(n):
        theta[i] += dt * rate[i]


def servo_targets(theta, t, p: "Params"):
    """Position targets (joints) and speeds (wheels) for phases theta at time t."""
    return ramp(t) * (p.offset + p.amp1 * np.sin(theta + p.phase1)
                      + p.amp2 * np.sin(2.0 * theta + p.phase2))


class Params:
    """Per-actuator constants decoded from a controller vector, under a
    steering command (0 = straight) and a speed command (1 = full)."""

    def __init__(self, ctrl: np.ndarray, layout: ActuatorLayout,
                 command: float = 0.0, speed: float = 1.0,
                 partner=None):
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
        self.couple_k    = np.zeros(n)
        self.couple_lag  = np.zeros(n)
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
            self.couple_k[a]    = _scale(g[COUPLE_K], *COUPLE_RANGE)
            self.couple_lag[a]  = _scale(g[COUPLE_LAG], *COUPLE_LAG_RANGE)
        self.partner = (np.full(n, -1, dtype=int) if partner is None
                        else np.asarray(partner, dtype=int))
        # as decoded, before the commands and the speed clip: what the firmware
        # receives, so it can steer and clip on the robot the same way
        self.amp1_base, self.amp2_base = self.amp1.copy(), self.amp2.copy()
        self.offset_base = self.offset.copy()
        scale = None
        if CLIP_MODE == "headroom":
            # Scale the base amplitudes by the headroom of the stronger steering
            # side before the commands, so a steered joint keeps both sides in
            # proportion instead of clipping only the stronger one. The usual
            # clip below stays as a safety net (it can still bite for speed_gain).
            peak_base = (self.amp1_base + 2.0 * self.amp2_base) * (
                self.omega + np.abs(self.sigma))
            g = np.clip(1.0 + np.abs(self.steer_gain), 0.0, 2.0)
            scale = np.minimum(1.0, SERVO_SPEED / np.maximum(peak_base * g, 1e-12))
        apply_commands(self, command, speed, scale)
        # A servo cannot follow a signal whose peak speed exceeds its own; with
        # load feedback the phase advances at up to omega + |sigma|.
        peak = (self.amp1 + 2.0 * self.amp2) * (self.omega + np.abs(self.sigma))
        k = np.minimum(1.0, SERVO_SPEED / np.maximum(peak, 1e-12))
        self.amp1 = self.amp1 * k
        self.amp2 = self.amp2 * k


def apply_commands(p: "Params", command: float, speed: float = 1.0,
                   scale=None) -> None:
    """Apply the steering and speed commands to the decoded params, in place,
    always from the *_base values (before the speed clip, so a stronger side
    still respects the servo). The firmware does the same in set_command and
    set_speed. scale, when given, multiplies the base amplitudes first."""
    c = float(np.clip(command, -1.0, 1.0))
    d = 1.0 - float(np.clip(speed, 0.0, 1.0))
    gain = (np.clip(1.0 + c * p.steer_gain, 0.0, 2.0)
            * np.clip(1.0 - d * p.speed_gain, 0.0, 2.0))
    shift = c * p.steer_shift + d * p.speed_shift
    amp1_base = p.amp1_base if scale is None else p.amp1_base * scale
    amp2_base = p.amp2_base if scale is None else p.amp2_base * scale
    p.amp1 = np.where(p.wheel, p.amp1_base, amp1_base * gain)
    p.amp2 = np.where(p.wheel, p.amp2_base, amp2_base * gain)
    p.offset = np.where(p.wheel, np.clip(p.offset_base + shift, *WHEEL_RANGE),
                        p.offset_base + shift)


class Controller:
    """Stateful controller: owns the phases, advanced every physics step."""

    def __init__(self, ctrl: np.ndarray, layout: ActuatorLayout,
                 command: float = 0.0, speed: float = 1.0, partner=None):
        self.params = Params(ctrl, layout, command, speed, partner)
        self.omega = self.params.omega
        self.theta = np.zeros(len(layout))

    def reset(self) -> None:
        self.theta[:] = 0.0

    def targets(self, t: float) -> np.ndarray:
        """Targets for the current phases (recomputed every control period)."""
        return servo_targets(self.theta, t, self.params)

    def advance(self, dt: float, load: np.ndarray) -> None:
        """Integrate the phases over one physics step (explicit Euler)."""
        p = self.params
        advance_phases(self.theta, self.omega, p.sigma, load, dt,
                       p.couple_k, p.couple_lag, p.partner)

    def step(self, t: float, dt: float, load: np.ndarray) -> np.ndarray:
        """Targets at time t, then the phases advanced by dt under this load."""
        out = self.targets(t)
        self.advance(dt, load)
        return out
