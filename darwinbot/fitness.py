"""
fitness.py
----------
Episode metrics -> scalar score, and the multi-terrain aggregate.

score = dx - LATERAL_W * |dy| - ENERGY_W * energy / (m g) - FALL_PENALTY * fell

energy / (m g) is in metres: the height the robot could have lifted itself
with the same work. Dividing by weight keeps heavy and light bodies on the
same footing, so the term behaves like a cost of transport.

For the "turn" task the goal is rotation in place instead of distance:

score = |yaw| - TURN_DRIFT_W * hypot(dx, dy) - ENERGY_W * energy / (m g) - FALL_PENALTY * fell

|yaw| is in radians and the sign does not matter, so a body may turn
whichever way suits it. Drifting away from the spawn point costs score.

A fall ends the episode but keeps the distance covered so far, so a robot
that runs 2 m and then tips over still beats one that never moves. The
fixed penalty only breaks ties in favour of staying upright.
"""

import math
from typing import Sequence

import numpy as np

from darwinbot.sim import Metrics

LATERAL_W    = 0.5
ENERGY_W     = 0.05
FALL_PENALTY = 0.5
TURN_DRIFT_W = 0.5
TASKS        = ("forward", "turn")
UNSTABLE_SCORE = -5.0
G = 9.81


def score(m: Metrics, task: str = "forward") -> float:
    if task not in TASKS:
        raise ValueError(f"unknown task '{task}', expected one of {TASKS}")
    if m.unstable:
        return UNSTABLE_SCORE
    cost = ENERGY_W * m.energy / (m.mass * G)
    if task == "turn":
        s = abs(m.yaw) - TURN_DRIFT_W * math.hypot(m.dx, m.dy) - cost
    else:
        s = m.dx - LATERAL_W * abs(m.dy) - cost
    if m.fell:
        s -= FALL_PENALTY
    return float(s)


def aggregate(scores: Sequence[float]) -> float:
    """
    Mean over terrains, blended with the worst one. Pure mean lets a robot
    ignore a terrain it fails on; the min term rewards robustness.
    """
    scores = np.asarray(scores, dtype=float)
    return float(0.7 * scores.mean() + 0.3 * scores.min())
