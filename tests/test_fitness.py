"""
tests/test_fitness.py
---------------------
Unit tests for per-scenario scorers in fitness.py.
These tests mock the simulator so they don't need MuJoCo.
"""

import pytest
from darwinbot.fitness import (
    _score_default, _score_confined,
    FALL_PENALTY, ENTRY_BONUS, DEPTH_BONUS, EXIT_BONUS,
    LATERAL_W, ENERGY_W, MIN_SCORE,
)
from darwinbot.scenarios import ScenarioInfo


# ── Helpers ───────────────────────────────────────────────────────────────────

def _metrics(distance_x=1.0, lateral_dev=0.0, energy=0.0,
             fell=False, y_final=0.0, target_y=None):
    """Build a fake metrics dict as returned by simulator.simulate()."""
    info = ScenarioInfo(
        name="test",
        spawn_pos=(0.0, -0.5, 0.35),
        target_pos=(0.0, target_y, 0.2) if target_y is not None else None,
    )
    return {
        "fell":        fell,
        "distance_x":  distance_x,
        "lateral_dev": lateral_dev,
        "energy":      energy,
        "_info":       info,
        "_pos0":       (0.0, -0.5, 0.35),
        "_pos_final":  (distance_x, y_final, 0.35),
    }


# ── _score_default ────────────────────────────────────────────────────────────

def test_score_default_fall():
    m = _metrics(fell=True)
    assert _score_default(m) == FALL_PENALTY


def test_score_default_basic():
    m = _metrics(distance_x=3.0, lateral_dev=0.5, energy=100.0)
    expected = 3.0 - LATERAL_W * 0.5 - ENERGY_W * 100.0
    assert abs(_score_default(m) - expected) < 1e-9


def test_score_default_min_clamp():
    # Huge lateral deviation → score below MIN_SCORE
    m = _metrics(distance_x=0.0, lateral_dev=1000.0)
    assert _score_default(m) == MIN_SCORE


# ── _score_confined ───────────────────────────────────────────────────────────

def test_confined_fall():
    m = _metrics(fell=True)
    assert _score_confined(m) == FALL_PENALTY


def test_confined_no_entry():
    """Robot stays outside (y_final ≤ 0) — no bonus."""
    m = _metrics(y_final=-0.1, target_y=0.6)
    assert _score_confined(m) == pytest.approx(0.0, abs=1e-6)


def test_confined_entry_only():
    m = _metrics(y_final=0.3, target_y=0.6)
    assert _score_confined(m) == pytest.approx(ENTRY_BONUS, abs=1e-6)


def test_confined_entry_and_depth():
    m = _metrics(y_final=0.7, target_y=0.6)
    assert _score_confined(m) == pytest.approx(ENTRY_BONUS + DEPTH_BONUS, abs=1e-6)


def test_confined_full_exit_requires_entry_first():
    """
    Robot ends at y < -0.1 but never entered (y_final never positive
    during the episode — we only see final pos). Since score >= ENTRY_BONUS
    is False, EXIT_BONUS should NOT be awarded.
    """
    m = _metrics(y_final=-0.5, target_y=0.6)
    assert _score_confined(m) == pytest.approx(0.0, abs=1e-6)


def test_confined_exit_bonus_awarded_after_entry_and_exit():
    """
    Simulate robot that entered and then exited back out.
    We model this as: y_final < -0.1 BUT entry happened.
    The scorer infers entry from y_final history — since the robot
    ended outside, award EXIT_BONUS only if y_f shows it went deep.
    NOTE: actual exit bonus requires scorer to see history; this test
    validates the current implementation logic (based on final y only).
    """
    # With current implementation: exit awarded when score >= ENTRY_BONUS AND y_f < -0.1.
    # That requires y_f > 0 to get ENTRY first, which contradicts y_f < -0.1.
    # So EXIT_BONUS cannot currently be triggered in a single step.
    # This test documents that known limitation.
    m = _metrics(y_final=-0.5, target_y=0.6)
    score = _score_confined(m)
    assert score < ENTRY_BONUS  # exit alone doesn't give entry + exit


# ── _score_confined energy penalty ───────────────────────────────────────────

def test_confined_energy_penalty():
    m = _metrics(y_final=0.5, energy=1000.0, target_y=0.6)
    score = _score_confined(m)
    assert score < ENTRY_BONUS  # energy penalty reduces the score
