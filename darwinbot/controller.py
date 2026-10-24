"""
controller.py
-------------
Open-loop central pattern generator (CPG) shared by every morphology.

One global frequency drives all joints, so the gait is periodic. Each joint
of a pair has its own amplitude, offset and phase; the right leg replays the
left leg's signal shifted by a per-pair phase `side_shift`. That keeps gaits
left/right symmetric in shape while still allowing trot, pace or bound
(side_shift = pi, 0 or anything between).

The controller vector has a fixed size for every morphology. Genes of absent
pairs or absent wheels are simply ignored, which lets a child inherit its
parent's controller verbatim even when a pair or wheel appears or vanishes.

Layout (all genes in [0, 1])
  [0] frequency
  per pair p (PAIR_CTRL genes each):
    for joint in JOINTS: amplitude, offset, phase
    side_shift
    wheel_speed
"""

from typing import List, Tuple

import numpy as np

from darwinbot.morphology import Morphology, N_PAIRS

JOINTS = ("hip_yaw", "hip_pitch", "knee")

FREQ_RANGE  = (0.5, 3.0)     # Hz
AMP_RANGE   = (0.0, 0.8)     # rad
OFFSET_RANGE = (-0.6, 0.6)   # rad
WHEEL_RANGE = (-15.0, 15.0)  # rad/s, signed so wheels may roll either way
RAMP_TIME   = 0.5            # s to fade the gait in from rest

JOINT_CTRL = 3                                    # amp, offset, phase
PAIR_CTRL  = len(JOINTS) * JOINT_CTRL + 2         # + side_shift + wheel_speed
CTRL_LENGTH = 1 + N_PAIRS * PAIR_CTRL

# (pair_index, side, kind) for every actuator, in model order; side 0 = left
ActuatorLayout = List[Tuple[int, int, str]]


def _scale(u, lo, hi):
    return lo + np.clip(u, 0.0, 1.0) * (hi - lo)


def _pair_base(pair_index: int) -> int:
    return 1 + pair_index * PAIR_CTRL


def active_genes(morph: Morphology) -> np.ndarray:
    """Indices of controller genes that affect this morphology."""
    idx = [0]
    for pair in morph.pairs:
        b = _pair_base(pair.index)
        idx.extend(range(b, b + len(JOINTS) * JOINT_CTRL + 1))
        if pair.has_wheel:
            idx.append(b + PAIR_CTRL - 1)
    return np.array(idx, dtype=int)


def random_controller(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 1.0, CTRL_LENGTH)


class CPG:
    """Precomputed per-actuator signal parameters; call with time t."""

    def __init__(self, ctrl: np.ndarray, layout: ActuatorLayout):
        ctrl = np.asarray(ctrl, dtype=float)
        assert ctrl.shape == (CTRL_LENGTH,), ctrl.shape
        self.omega = 2 * np.pi * _scale(ctrl[0], *FREQ_RANGE)

        n = len(layout)
        self.amp    = np.zeros(n)
        self.offset = np.zeros(n)
        self.phase  = np.zeros(n)
        for a, (pair, side, kind) in enumerate(layout):
            b = _pair_base(pair)
            if kind == "wheel":
                self.offset[a] = _scale(ctrl[b + PAIR_CTRL - 1], *WHEEL_RANGE)
                continue
            j = b + JOINTS.index(kind) * JOINT_CTRL
            side_shift = _scale(ctrl[b + len(JOINTS) * JOINT_CTRL], 0.0, 2 * np.pi)
            self.amp[a]    = _scale(ctrl[j],     *AMP_RANGE)
            self.offset[a] = _scale(ctrl[j + 1], *OFFSET_RANGE)
            self.phase[a]  = _scale(ctrl[j + 2], 0.0, 2 * np.pi) + side * side_shift

    def __call__(self, t: float) -> np.ndarray:
        ramp = min(1.0, t / RAMP_TIME)
        return ramp * (self.offset + self.amp * np.sin(self.omega * t + self.phase))
