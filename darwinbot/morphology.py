"""
morphology.py
-------------
Body genome: a box torso with up to N_PAIRS bilaterally mirrored leg pairs.

Every gene lives in [0, 1] and is mapped linearly onto its physical range on
decode. Keeping the raw genome normalised makes mutation and crossover
scale-free: one sigma works for every gene.

Leg anatomy (one side; the other side is the mirror image across the XZ plane)
  hip   : pitch servo (lift/lower) + optional yaw servo (swing fore/aft)
  thigh : capsule, horizontal, pointing outward from the torso side
  knee  : pitch joint
  shin  : capsule, pointing straight down at rest
  foot  : optional wheel at the shin tip, axle along the body Y axis

Genome layout
  [0]  body_length
  [1]  body_width
  then per pair p (PAIR_GENES each):
  [+0] enabled     > 0.5 means the pair exists
  [+1] attach_x    position along the torso side, -1 (rear) .. +1 (front)
  [+2] splay       leg yaw at rest, rad (positive points the leg forward)
  [+3] thigh_len   snapped to LINK_LENGTHS (gene picks the catalogue index)
  [+4] shin_len    snapped to LINK_LENGTHS
  [+5] has_yaw     > 0.5 means the hip has a yaw servo
  [+6] has_wheel   > 0.5 means the foot is a wheel
  [+7] wheel_rad

Thigh and shin are printed links of fixed catalogue lengths with a constant
capsule radius, so every body is a list of real parts (see n_servos).
"""

from dataclasses import dataclass
from typing import List

import numpy as np

N_PAIRS = 3

LINK_LENGTHS = (0.06, 0.08, 0.10, 0.12)   # m, printed link catalogue
LINK_RADIUS  = 0.012                      # m, capsule radius of every link

BODY_BOUNDS = [
    (0.15, 0.40),   # body_length (m)
    (0.10, 0.25),   # body_width  (m)
]
PAIR_BOUNDS = [
    (0.0, 1.0),     # enabled
    (-1.0, 1.0),    # attach_x (fraction of half length)
    (-0.6, 0.6),    # splay (rad)
    (0.0, 1.0),     # thigh_len (index into LINK_LENGTHS)
    (0.0, 1.0),     # shin_len (index into LINK_LENGTHS)
    (0.0, 1.0),     # has_yaw
    (0.0, 1.0),     # has_wheel
    (0.03, 0.06),   # wheel_rad (m)
]
PAIR_GENES = len(PAIR_BOUNDS)
MORPH_LENGTH = len(BODY_BOUNDS) + N_PAIRS * PAIR_GENES
BOUNDS = BODY_BOUNDS + PAIR_BOUNDS * N_PAIRS


@dataclass
class LegPair:
    index:     int      # pair slot 0..N_PAIRS-1, stable across mutation
    attach_x:  float
    splay:     float
    thigh_len: float
    shin_len:  float
    has_yaw:   bool
    has_wheel: bool
    wheel_rad: float

    @property
    def radius(self) -> float:
        return LINK_RADIUS

    @property
    def n_servos(self) -> int:
        """Servos in both legs of the pair: hip pitch, knee, yaw, wheel."""
        return 2 * (2 + self.has_yaw + self.has_wheel)

    @property
    def reach(self) -> float:
        """Distance from hip height down to the ground contact point at rest."""
        if self.has_wheel:
            return self.shin_len + self.wheel_rad
        return self.shin_len + self.radius


@dataclass
class Morphology:
    body_length: float
    body_width:  float
    pairs:       List[LegPair]

    @property
    def n_legs(self) -> int:
        return 2 * len(self.pairs)

    @property
    def n_wheels(self) -> int:
        return 2 * sum(p.has_wheel for p in self.pairs)

    @property
    def n_servos(self) -> int:
        return sum(p.n_servos for p in self.pairs)

    @property
    def mean_leg_length(self) -> float:
        return float(np.mean([p.thigh_len + p.shin_len for p in self.pairs]))


def _scale(u: float, lo: float, hi: float) -> float:
    return lo + float(np.clip(u, 0.0, 1.0)) * (hi - lo)


def _snap(u: float) -> float:
    """Gene in [0, 1] -> nearest-below catalogue length."""
    i = min(int(np.clip(u, 0.0, 1.0) * len(LINK_LENGTHS)), len(LINK_LENGTHS) - 1)
    return LINK_LENGTHS[i]


def decode(genes: np.ndarray) -> Morphology:
    genes = np.asarray(genes, dtype=float)
    assert genes.shape == (MORPH_LENGTH,), genes.shape
    phys = [_scale(u, lo, hi) for u, (lo, hi) in zip(genes, BOUNDS)]

    enabled = [genes[2 + p * PAIR_GENES] > 0.5 for p in range(N_PAIRS)]
    if not any(enabled):
        # A body with no legs is a brick: keep the pair closest to switching on.
        best = int(np.argmax([genes[2 + p * PAIR_GENES] for p in range(N_PAIRS)]))
        enabled[best] = True

    pairs = []
    for p in range(N_PAIRS):
        if not enabled[p]:
            continue
        b = 2 + p * PAIR_GENES
        pairs.append(LegPair(
            index     = p,
            attach_x  = phys[b + 1],
            splay     = phys[b + 2],
            thigh_len = _snap(genes[b + 3]),
            shin_len  = _snap(genes[b + 4]),
            has_yaw   = genes[b + 5] > 0.5,
            has_wheel = genes[b + 6] > 0.5,
            wheel_rad = phys[b + 7],
        ))
    return Morphology(body_length=phys[0], body_width=phys[1], pairs=pairs)


def random_morphology(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 1.0, MORPH_LENGTH)


def mutate(genes: np.ndarray, rng: np.random.Generator,
           sigma: float = 0.08, flip_prob: float = 0.08) -> np.ndarray:
    """
    Gaussian step on every continuous gene, plus an independent chance of
    flipping each discrete switch (enabled, has_yaw, has_wheel). Flips are applied by
    reflecting the gene around 0.5 so the rest of its value is kept.
    """
    child = genes + rng.normal(0.0, sigma, genes.shape)
    for p in range(N_PAIRS):
        for offset in (0, 5, 6):
            idx = 2 + p * PAIR_GENES + offset
            if rng.random() < flip_prob:
                child[idx] = 1.0 - child[idx]
    return np.clip(child, 0.0, 1.0)


def descriptor(m: Morphology) -> tuple:
    """Features used as MAP-Elites axes: (pairs, wheeled pairs, servos)."""
    return (len(m.pairs), m.n_wheels // 2, m.n_servos)


def summary(m: Morphology) -> str:
    lines = [f"torso {m.body_length:.2f} x {m.body_width:.2f} m, "
             f"{m.n_legs} legs, {m.n_wheels} wheels, {m.n_servos} servos"]
    for p in m.pairs:
        foot = f"wheel r={p.wheel_rad:.3f}" if p.has_wheel else "foot"
        yaw = "yaw" if p.has_yaw else "no yaw"
        lines.append(
            f"  pair {p.index}: x={p.attach_x:+.2f} splay={np.degrees(p.splay):+.0f}deg "
            f"thigh={p.thigh_len:.2f} shin={p.shin_len:.2f} {yaw} {foot}"
        )
    return "\n".join(lines)
