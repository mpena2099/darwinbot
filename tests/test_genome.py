"""
tests/test_genome.py
--------------------
Unit tests for genome encoding/decoding round-trips,
bounds clipping, and CPG signal generation.
"""

import numpy as np
import pytest

from darwinbot.genome import (
    random_genome, decode, clip_genome, genome_summary,
    segment_signal, GENOME_LENGTH, N_LIMBS, N_SEGMENTS,
    GENES_PER_LIMB, BOUNDS, G_HAS_WHEEL,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _all_in_bounds(genes: np.ndarray) -> bool:
    for limb_idx in range(N_LIMBS):
        base = limb_idx * GENES_PER_LIMB
        for offset, (lo, hi) in enumerate(BOUNDS):
            v = genes[base + offset]
            if not (lo <= v <= hi):
                return False
    return True


# ── Genome shape and bounds ───────────────────────────────────────────────────

def test_genome_length():
    g = random_genome()
    assert len(g) == GENOME_LENGTH, f"Expected {GENOME_LENGTH} genes, got {len(g)}"


def test_random_genome_in_bounds():
    rng = np.random.default_rng(42)
    g = random_genome(rng)
    assert _all_in_bounds(g), "random_genome() produced out-of-bounds genes"


def test_clip_genome_forces_bounds():
    # Create a genome violating every bound
    g = np.full(GENOME_LENGTH, 1e6)
    clipped = clip_genome(g)
    assert _all_in_bounds(clipped), "clip_genome() did not bring genes within bounds"

    g2 = np.full(GENOME_LENGTH, -1e6)
    clipped2 = clip_genome(g2)
    assert _all_in_bounds(clipped2), "clip_genome() did not clip negative values"


# ── Decode round-trip ─────────────────────────────────────────────────────────

def test_decode_structure():
    g = random_genome(np.random.default_rng(7))
    robot = decode(g)

    assert robot.n_limbs == N_LIMBS
    for limb in robot.limbs:
        assert len(limb.segments) == N_SEGMENTS
        assert limb.has_wheel in (0, 1)
        assert limb.wheel_rad > 0


def test_decode_has_wheel_is_binary():
    """has_wheel gene covers [0, 1]; after rounding it must be exactly 0 or 1."""
    rng = np.random.default_rng(99)
    for _ in range(20):
        g = random_genome(rng)
        robot = decode(g)
        for limb in robot.limbs:
            assert limb.has_wheel in (0, 1)


def test_decode_is_deterministic():
    g = random_genome(np.random.default_rng(0))
    r1 = decode(g)
    r2 = decode(g)
    assert r1.limbs[0].segments[0].length == r2.limbs[0].segments[0].length


# ── CPG signal ────────────────────────────────────────────────────────────────

def test_segment_signal_amplitude():
    from darwinbot.genome import SegmentGenes
    seg = SegmentGenes(length=0.1, width=0.03, osc_freq=1.0, osc_phase=0.0, osc_amp=0.5)
    # Peak of sin is 1 → max signal = amp
    t_peak = 0.25  # sin(2π × 1.0 × 0.25 + 0) = sin(π/2) = 1
    assert abs(segment_signal(seg, t_peak) - 0.5) < 1e-9

    t_zero = 0.0   # sin(0) = 0
    assert abs(segment_signal(seg, t_zero)) < 1e-9


def test_segment_signal_phase_shift():
    from darwinbot.genome import SegmentGenes
    seg0 = SegmentGenes(length=0.1, width=0.03, osc_freq=1.0, osc_phase=0.0, osc_amp=1.0)
    seg1 = SegmentGenes(length=0.1, width=0.03, osc_freq=1.0, osc_phase=np.pi, osc_amp=1.0)
    # At t=0: seg0 → 0, seg1 → sin(π) ≈ 0 but with phase, so seg0 leads seg1
    t = 0.25  # seg0 peak
    assert segment_signal(seg0, t) > 0
    assert segment_signal(seg1, t) < 0  # phase-shifted to trough


# ── Genome summary ────────────────────────────────────────────────────────────

def test_genome_summary_runs():
    g = random_genome(np.random.default_rng(1))
    summary = genome_summary(g)
    assert "RobotGenome" in summary
    assert "Limb 0" in summary
    assert "Limb 5" in summary
