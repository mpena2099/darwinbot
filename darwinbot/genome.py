"""
genome.py
---------
Genome structure for a hybrid robot with segmented limbs.

Each limb has N_SEGMENTS segments in series (like an insect leg),
with an optional wheel always at the tip of the last segment.

Limb gene layout
----------------
  [0]          angle          insertion angle from body (rad)
  [1..5]       segment 0      length, width, osc_freq, osc_phase, osc_amp
  [6..10]      segment 1      length, width, osc_freq, osc_phase, osc_amp
  [11..15]     segment 2      length, width, osc_freq, osc_phase, osc_amp
  [16]         has_wheel      0 or 1 (rounded on decode)
  [17]         wheel_rad      wheel radius (m)

Total: 1 + N_SEGMENTS × 5 + 2 = 18 genes per limb (for N_SEGMENTS=3)
       6 limbs × 18 genes      = 108 genes total
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


# ── Layout constants ──────────────────────────────────────────────────────────

N_LIMBS         = 6
N_SEGMENTS      = 3    # segments per limb — Tachikoma-style
GENES_PER_SEG   = 5    # length, width, osc_freq, osc_phase, osc_amp
GENES_PER_LIMB  = 1 + N_SEGMENTS * GENES_PER_SEG + 2  # angle + segs + wheel
GENOME_LENGTH   = N_LIMBS * GENES_PER_LIMB            # 108

# Offsets within each limb block
G_ANGLE    = 0                                    # insertion angle

# Segment gene offsets (relative to start of that segment's block)
GS_LENGTH    = 0
GS_WIDTH     = 1
GS_OSC_FREQ  = 2
GS_OSC_PHASE = 3
GS_OSC_AMP   = 4

# Wheel genes (after all segments)
G_HAS_WHEEL = 1 + N_SEGMENTS * GENES_PER_SEG
G_WHEEL_RAD = 1 + N_SEGMENTS * GENES_PER_SEG + 1


def seg_base(seg_idx: int) -> int:
    """Index of first gene of segment seg_idx within a limb block."""
    return 1 + seg_idx * GENES_PER_SEG


# ── Bounds ────────────────────────────────────────────────────────────────────
# One entry per gene within a limb block, in order.

BOUNDS = (
    [(-1.57, 1.57)]                          # G_ANGLE
    + [                                       # segments
        (0.06, 0.25),   # GS_LENGTH  (shorter per segment than before)
        (0.02, 0.08),   # GS_WIDTH
        (0.3,  3.0),    # GS_OSC_FREQ
        (0.0,  6.2832), # GS_OSC_PHASE
        (0.1,  1.0),    # GS_OSC_AMP
    ] * N_SEGMENTS
    + [
        (0.0,  1.0),    # G_HAS_WHEEL
        (0.04, 0.12),   # G_WHEEL_RAD
    ]
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SegmentGenes:
    length:    float
    width:     float
    osc_freq:  float
    osc_phase: float
    osc_amp:   float


@dataclass
class LimbGenes:
    angle:     float
    segments:  List[SegmentGenes]
    has_wheel: int        # 0 or 1
    wheel_rad: float

    @property
    def n_segments(self):
        return len(self.segments)


@dataclass
class RobotGenome:
    limbs: List[LimbGenes] = field(default_factory=list)

    @property
    def n_limbs(self):
        return len(self.limbs)


# ── Genome creation ───────────────────────────────────────────────────────────

def random_genome(rng: np.random.Generator = None) -> np.ndarray:
    """Return a random flat genome vector within valid bounds."""
    if rng is None:
        rng = np.random.default_rng()

    genes = np.zeros(GENOME_LENGTH)
    for limb_idx in range(N_LIMBS):
        limb_base = limb_idx * GENES_PER_LIMB
        for gene_offset, (lo, hi) in enumerate(BOUNDS):
            genes[limb_base + gene_offset] = rng.uniform(lo, hi)
    return genes


def clip_genome(genes: np.ndarray) -> np.ndarray:
    """Clip all genes to their valid bounds."""
    genes = genes.copy()
    for limb_idx in range(N_LIMBS):
        limb_base = limb_idx * GENES_PER_LIMB
        for gene_offset, (lo, hi) in enumerate(BOUNDS):
            idx = limb_base + gene_offset
            genes[idx] = np.clip(genes[idx], lo, hi)
    return genes


# ── Decode ────────────────────────────────────────────────────────────────────

def decode(genes: np.ndarray) -> RobotGenome:
    """Convert flat numpy vector into a RobotGenome."""
    genes = clip_genome(genes)
    limbs = []
    for i in range(N_LIMBS):
        b = i * GENES_PER_LIMB

        angle = genes[b + G_ANGLE]
        segs  = []
        for s in range(N_SEGMENTS):
            sb = b + seg_base(s)
            segs.append(SegmentGenes(
                length    = genes[sb + GS_LENGTH],
                width     = genes[sb + GS_WIDTH],
                osc_freq  = genes[sb + GS_OSC_FREQ],
                osc_phase = genes[sb + GS_OSC_PHASE],
                osc_amp   = genes[sb + GS_OSC_AMP],
            ))

        limbs.append(LimbGenes(
            angle     = angle,
            segments  = segs,
            has_wheel = int(round(genes[b + G_HAS_WHEEL])),
            wheel_rad = genes[b + G_WHEEL_RAD],
        ))
    return RobotGenome(limbs=limbs)


# ── Controller ────────────────────────────────────────────────────────────────

def segment_signal(seg: SegmentGenes, t: float) -> float:
    """
    Sinusoidal CPG signal for one segment joint at time t.
    Each segment has independent frequency, phase, and amplitude —
    allowing wave-like propagation along the limb.
    """
    return seg.osc_amp * np.sin(2 * np.pi * seg.osc_freq * t + seg.osc_phase)


# ── Summary ───────────────────────────────────────────────────────────────────

def genome_summary(genes: np.ndarray) -> str:
    """Human-readable summary of a genome."""
    robot = decode(genes)
    lines = [f"RobotGenome summary ({N_LIMBS} limbs × {N_SEGMENTS} segments):"]
    for i, limb in enumerate(robot.limbs):
        wheel_str = (f"wheel r={limb.wheel_rad:.3f}"
                     if limb.has_wheel else "no wheel")
        seg_strs  = [
            f"[{s.length:.2f}m {s.osc_freq:.1f}Hz φ{np.degrees(s.osc_phase):.0f}°]"
            for s in limb.segments
        ]
        lines.append(
            f"  Limb {i}: angle={np.degrees(limb.angle):.1f}°  "
            f"segs={'  '.join(seg_strs)}  {wheel_str}"
        )
    return "\n".join(lines)
