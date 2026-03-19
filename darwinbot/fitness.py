"""
fitness.py
----------
Multi-scenario fitness function.

Each individual is evaluated in ALL active scenarios.
The final fitness is a weighted average of per-scenario scores,
with weights defined in scenarios/registry.py.

Per-scenario scoring
--------------------
flat / surfaces / obstacles:
    score = distance_x - LATERAL_W * lateral_dev - ENERGY_W * energy
    Fall  → FALL_PENALTY

confined:
    Scored on progress toward and through the room, not raw X distance.
    score = entry_bonus + depth_bonus + exit_bonus - ENERGY_W * energy
    Fall  → FALL_PENALTY
"""

import numpy as np
from typing import List, Tuple, Callable

from darwinbot.simulator import simulate

# Shared penalty/weight constants
FALL_PENALTY   = -15.0
LATERAL_W      =   0.4
ENERGY_W       =   0.001
MIN_SCORE      =  -5.0

# Confined scenario bonuses
ENTRY_BONUS    =  2.0    # crossed the opening threshold into room
DEPTH_BONUS    =  3.0    # reached back half of room
EXIT_BONUS     =  4.0    # exited back through the opening


# ── Per-scenario scorers ──────────────────────────────────────────────────────

def _score_default(metrics: dict) -> float:
    """Used for flat, surfaces, obstacles."""
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
    Scores progress in the confined room scenario.
    Uses Y displacement (robot moves in +Y direction into the room).
    """
    if metrics["fell"]:
        return FALL_PENALTY

    info      = metrics["_info"]
    pos0      = metrics["_pos0"]
    pos_final = metrics["_pos_final"]

    # Room interior spans Y = 0 .. ROOM_D
    # Robot spawns at Y ~ -0.5
    y0    = pos0[1]
    y_f   = pos_final[1]
    score = 0.0

    # Entered the room (crossed Y=0 threshold)
    if y_f > 0.0:
        score += ENTRY_BONUS

    # Reached the back half of the room
    if info.target_pos is not None:
        room_mid_y = info.target_pos[1]
        if y_f >= room_mid_y:
            score += DEPTH_BONUS

    # Exited back through the opening (returned to Y < 0 after entering)
    # Approximate: final Y is back near spawn
    if y_f < -0.1 and y0 < 0:
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
    active_scenarios: List[Tuple[str, Callable, float]] = None,
) -> tuple:
    """
    Evaluate one individual across all active scenarios.

    Parameters
    ----------
    genes            : list or np.ndarray (DEAP passes a list)
    active_scenarios : list of (name, builder_fn, weight) from registry.
                       If None, defaults to flat floor only.

    Returns
    -------
    (weighted_fitness_score,)  — tuple required by DEAP
    """
    from darwinbot.scenarios.registry import get_active_scenarios
    if active_scenarios is None:
        active_scenarios = get_active_scenarios("phase1")

    genes = np.array(genes, dtype=float)

    total_score  = 0.0
    total_weight = 0.0   # weights already normalised, but be safe

    for name, builder, weight in active_scenarios:
        metrics = simulate(genes, scenario_builder=builder)
        scorer  = _SCORERS.get(name, _score_default)
        score   = scorer(metrics)
        total_score  += weight * score
        total_weight += weight

    final = total_score / total_weight if total_weight > 0 else FALL_PENALTY
    return (float(final),)
