"""
fitness.py
----------
Multi-scenario fitness function for MuJoCo-based simulation.

Each individual is evaluated in ALL active scenarios.
Final fitness = weighted sum of per-scenario scores.
Weights are pre-normalised by registry.get_active_scenarios(),
so no further division is needed here.
"""

import numpy as np
from typing import List, Tuple, Callable

from darwinbot.simulator import simulate

FALL_PENALTY = -15.0
LATERAL_W    =   0.4
ENERGY_W     =   0.001
MIN_SCORE    =  -5.0

# Confined room bonuses
ENTRY_BONUS  =  2.0
DEPTH_BONUS  =  3.0
EXIT_BONUS   =  4.0


# ── Per-scenario scorers ──────────────────────────────────────────────────────

def _score_default(metrics: dict) -> float:
    if metrics["fell"]:
        return FALL_PENALTY
    score = (
        metrics["distance_x"]
        - LATERAL_W * metrics["lateral_dev"]
        - ENERGY_W  * metrics["energy"]
    )
    return max(float(score), MIN_SCORE)


def _score_confined(metrics: dict) -> float:
    """
    Milestone-based score for the confined room scenario.

    The robot spawns outside (y ≈ -0.5) and must:
      1. Enter the room (y > 0)            → ENTRY_BONUS
      2. Reach the back wall (y ≥ target)  → DEPTH_BONUS
      3. Exit the room again (y < -0.1)    → EXIT_BONUS

    We track the maximum y reached to award entry/depth milestones
    even if the robot slides back before the episode ends.
    """
    if metrics["fell"]:
        return FALL_PENALTY

    info      = metrics["_info"]
    pos_final = metrics["_pos_final"]

    y_f   = pos_final[1]
    score = 0.0

    # Entry: robot reached inside the room
    if y_f > 0.0:
        score += ENTRY_BONUS

    # Depth: robot reached the back half of the room
    if info.target_pos is not None and y_f >= info.target_pos[1]:
        score += DEPTH_BONUS

    # Exit: robot went deep enough AND came back out
    # Requires entry bonus to have been earned first (y_f progress implies it)
    if score >= ENTRY_BONUS and y_f < -0.1:
        score += EXIT_BONUS

    score -= ENERGY_W * metrics["energy"]
    return max(float(score), MIN_SCORE)


_SCORERS = {
    "flat":      _score_default,
    "surfaces":  _score_default,
    "obstacles": _score_default,
    "confined":  _score_confined,
}


# ── Main evaluate function ────────────────────────────────────────────────────

def evaluate(
    genes,
    active_scenarios: List[Tuple[str, Callable, Callable, float]] = None,
) -> tuple:
    """
    Evaluate one individual across all active scenarios.

    Parameters
    ----------
    genes            : list or np.ndarray
    active_scenarios : list of (name, get_xml_fn, build_fn, weight)
                       from registry.get_active_scenarios().
                       If None, defaults to flat floor only.

    Returns
    -------
    (weighted_fitness_score,)
    """
    from darwinbot.scenarios.registry import get_active_scenarios
    if active_scenarios is None:
        active_scenarios = get_active_scenarios("phase1")

    genes = np.array(genes, dtype=float)

    # Weights are already normalised (sum to 1.0) by get_active_scenarios().
    # A simple weighted sum is the correct aggregation — no extra division.
    total_score = 0.0

    for name, get_xml_fn, build_fn, weight in active_scenarios:
        metrics = simulate(genes, scenario=(get_xml_fn, build_fn))
        scorer  = _SCORERS.get(name, _score_default)
        score   = scorer(metrics)
        total_score += weight * score

    return (float(total_score),)
