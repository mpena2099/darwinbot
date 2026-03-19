"""
fitness.py
----------
Defines the fitness function used by DEAP.

Objectives (single-objective for now, composite score):
  + Forward distance (primary)
  - Lateral deviation (penalised)
  - Energy consumption (penalised lightly)
  - Fall penalty (heavy)

Returns a tuple (score,) as required by DEAP.
"""

import numpy as np
from darwinbot.simulator import simulate

# Penalty / weight constants
FALL_PENALTY      = -15.0
LATERAL_WEIGHT    =   0.4   # multiplied by lateral deviation (m)
ENERGY_WEIGHT     =   0.001  # multiplied by energy proxy (J·s)
MIN_DISTANCE      =  -5.0   # floor to avoid extreme outliers


def evaluate(genes) -> tuple:
    """
    Evaluate one individual. Called by DEAP's toolbox.

    Parameters
    ----------
    genes : list or np.ndarray  (DEAP passes a list)

    Returns
    -------
    (fitness_score,)  — tuple required by DEAP
    """
    genes = np.array(genes, dtype=float)

    metrics = simulate(genes)

    if metrics["fell"]:
        return (FALL_PENALTY,)

    score = (
        metrics["distance_x"]
        - LATERAL_WEIGHT  * metrics["lateral_dev"]
        - ENERGY_WEIGHT   * metrics["energy"]
    )

    score = max(score, MIN_DISTANCE)
    return (float(score),)
