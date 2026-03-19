"""
genome.py
---------
Defines the genome structure for a hybrid robot individual.

Each individual encodes:
  - Morphology: number of limbs, length, width, insertion angle
  - Wheels: presence, position along limb, radius, max torque
  - Controller: oscillator frequency, phase, amplitude per limb joint
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


# ── Genome layout constants ──────────────────────────────────────────────────

N_LIMBS = 4          # fixed for now; can be made evolvable later
GENES_PER_LIMB = 9   # see LimbGenes below

GENOME_LENGTH = N_LIMBS * GENES_PER_LIMB

# Gene index offsets within each limb block
G_LENGTH    = 0   # limb length (m)
G_WIDTH     = 1   # limb width / radius (m)
G_ANGLE     = 2   # insertion angle from body (rad)
G_HAS_WHEEL = 3   # wheel present: rounds to 0 or 1
G_WHEEL_POS = 4   # wheel position along limb (0.0 = base, 1.0 = tip)
G_WHEEL_RAD = 5   # wheel radius (m)
G_OSC_FREQ  = 6   # oscillator frequency (Hz)
G_OSC_PHASE = 7   # oscillator phase offset (rad)
G_OSC_AMP   = 8   # oscillator amplitude (rad)

# Gene bounds: (min, max) for each offset
BOUNDS = [
    (0.10, 0.50),        # G_LENGTH
    (0.03, 0.15),        # G_WIDTH
    (-1.57, 1.57),       # G_ANGLE  (-π/2 to π/2)
    (0.0,  1.0),         # G_HAS_WHEEL (continuous; rounded on decode)
    (0.3,  0.9),         # G_WHEEL_POS
    (0.04, 0.12),        # G_WHEEL_RAD
    (0.3,  3.0),         # G_OSC_FREQ
    (0.0,  6.2832),      # G_OSC_PHASE  (0 to 2π)
    (0.2,  1.2),         # G_OSC_AMP
]


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class LimbGenes:
    length:    float
    width:     float
    angle:     float
    has_wheel: int      # 0 or 1
    wheel_pos: float
    wheel_rad: float
    osc_freq:  float
    osc_phase: float
    osc_amp:   float


@dataclass
class RobotGenome:
    limbs: List[LimbGenes] = field(default_factory=list)

    @property
    def n_limbs(self):
        return len(self.limbs)


# ── Genome creation ──────────────────────────────────────────────────────────

def random_genome(rng: np.random.Generator = None) -> np.ndarray:
    """Return a random flat genome vector within valid bounds."""
    if rng is None:
        rng = np.random.default_rng()

    genes = np.zeros(GENOME_LENGTH)
    for limb_idx in range(N_LIMBS):
        base = limb_idx * GENES_PER_LIMB
        for gene_offset, (lo, hi) in enumerate(BOUNDS):
            genes[base + gene_offset] = rng.uniform(lo, hi)
    return genes


def clip_genome(genes: np.ndarray) -> np.ndarray:
    """Clip all genes to their valid bounds after mutation/crossover."""
    genes = genes.copy()
    for limb_idx in range(N_LIMBS):
        base = limb_idx * GENES_PER_LIMB
        for gene_offset, (lo, hi) in enumerate(BOUNDS):
            idx = base + gene_offset
            genes[idx] = np.clip(genes[idx], lo, hi)
    return genes


# ── Decode flat vector → RobotGenome ─────────────────────────────────────────

def decode(genes: np.ndarray) -> RobotGenome:
    """Convert a flat numpy vector into a RobotGenome dataclass."""
    genes = clip_genome(genes)
    limbs = []
    for i in range(N_LIMBS):
        b = i * GENES_PER_LIMB
        limbs.append(LimbGenes(
            length    = genes[b + G_LENGTH],
            width     = genes[b + G_WIDTH],
            angle     = genes[b + G_ANGLE],
            has_wheel = int(round(genes[b + G_HAS_WHEEL])),
            wheel_pos = genes[b + G_WHEEL_POS],
            wheel_rad = genes[b + G_WHEEL_RAD],
            osc_freq  = genes[b + G_OSC_FREQ],
            osc_phase = genes[b + G_OSC_PHASE],
            osc_amp   = genes[b + G_OSC_AMP],
        ))
    return RobotGenome(limbs=limbs)


# ── Controller signal ─────────────────────────────────────────────────────────

def oscillator_signal(limb: LimbGenes, t: float) -> float:
    """
    Returns the joint angle target for a limb at time t (seconds).
    Uses a simple sinusoidal Central Pattern Generator (CPG).
    """
    return limb.osc_amp * np.sin(2 * np.pi * limb.osc_freq * t + limb.osc_phase)


# ── Helpers ───────────────────────────────────────────────────────────────────

def genome_summary(genes: np.ndarray) -> str:
    """Human-readable summary of a genome."""
    robot = decode(genes)
    lines = ["RobotGenome summary:"]
    for i, limb in enumerate(robot.limbs):
        wheel_str = f"wheel@{limb.wheel_pos:.2f} r={limb.wheel_rad:.3f}" \
                    if limb.has_wheel else "no wheel"
        lines.append(
            f"  Limb {i}: len={limb.length:.3f}m  angle={np.degrees(limb.angle):.1f}°"
            f"  [{wheel_str}]  osc={limb.osc_freq:.2f}Hz amp={limb.osc_amp:.2f}rad"
        )
    return "\n".join(lines)
